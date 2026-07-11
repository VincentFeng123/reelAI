"""
IngestionPipeline — the orchestrator.

One class, two public methods:

  * `ingest_url(source_url, ...) -> IngestResult`
      Timestamped transcript → Gemini cue segmentation → persistence for one YouTube URL.

  * `ingest_feed(feed_url, max_items=6, ...) -> IngestFeedResult`
      Resolve a YouTube channel or playlist URL to individual video URLs
      and call `ingest_url` for each with bounded concurrency.

The pipeline owns:
  * A bounded `ThreadPoolExecutor` for transcript and segmentation work in `ingest_feed`.
  * A process-wide YouTube provider rate limiter.

The pipeline is stateless beyond those two things. It does NOT cache results itself —
search evidence and transcript artifacts are cached in versioned DB tables.
"""

from __future__ import annotations

import collections
import hashlib
import logging
import math
import os
import threading
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, TimeoutError as FutureTimeoutError, wait
from pathlib import Path
from typing import Any, Callable

from ..db import DatabaseIntegrityError, dumps_json, fetch_one, get_conn, now_iso, upsert
from . import TERMS_NOTICE
from .errors import (
    BlockedVideoError,
    DownloadError,
    IngestError,
    RateLimitedError,
    SegmentationError,
    ServerlessUnavailable,
    TranscriptionError,
    UnsupportedSourceError,
)
from .logging_config import get_ingest_logger, log_event, new_trace_id, set_trace_id
from .metadata import (
    build_takeaways_for_ingest,
    fallback_ai_summary,
    format_attribution,
)
from .models import (
    IngestFeedItem,
    IngestFeedResult,
    IngestMetadata,
    IngestResult,
    IngestSearchItem,
    IngestSearchResult,
    IngestSegment,
    IngestTopicCutResult,
    IngestTranscriptCue,
    PlatformLiteral,
    ReelOutWithAttribution,
    YouTubeSourceRef,
)
from .persistence import (
    load_existing_reel,
    resolve_material_concept,
    store_ingest_metadata_blob,
    upsert_reel_row,
    upsert_video,
)
from ..clip_engine import run as clip_engine_run, search as clip_engine_search, bridge as clip_engine_bridge, metadata as clip_engine_meta, config as clip_engine_config  # noqa: F401
from ..clip_engine.cancellation import is_cancelled, raise_if_cancelled
from ..clip_engine.provider_runtime import GenerationContext
from ..clip_engine.errors import (
    CancellationError as _ClipCancellationError,
    ClipError as _ClipError,
    ProviderError as _ClipProviderError,
    TranscriptError as _ClipTranscriptError,
    TranscriptUnavailableError as _TranscriptUnavailableError,
    UnsupportedURLError as _ClipUnsupportedURLError,
)
from ..services.assessments import store_reel_assessment_question
from ..services.search_query_plan import (
    SearchQueryPlan,
    topic_signature_evidence,
)

logger: logging.Logger = get_ingest_logger(__name__)

# Shared wall-clock budget for one topic's concurrent clip+filter batch. A
# pathological set of videos must not multiply this deadline by video count.
INGEST_TOPIC_VIDEO_TIMEOUT_SEC = float(os.environ.get("INGEST_TOPIC_VIDEO_TIMEOUT_SEC", "180"))


def _run_clip(
    url: str,
    *,
    topic: str,
    language: str,
    should_cancel: Callable[[], bool] | None,
    generation_context: GenerationContext | None = None,
    deadline_monotonic: float | None = None,
) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "language": language,
        "generation_context": generation_context,
        "provider_cache": generation_context.cache_store if generation_context is not None else None,
    }
    if deadline_monotonic is not None:
        settings["deadline_monotonic"] = float(deadline_monotonic)
    kwargs = {
        "topic": topic,
        "settings": settings,
    }
    if should_cancel is None:
        return clip_engine_run.clip(url, **kwargs)
    return clip_engine_run.clip(url, **kwargs, should_cancel=should_cancel)


def _discover(
    topic: str,
    *,
    limit: int,
    exclude_video_ids: list[str],
    level: str | None,
    should_cancel: Callable[[], bool] | None,
    creative_commons_only: bool = False,
    preferred_video_duration: str = "any",
    language: str = "en",
    generation_context: GenerationContext | None = None,
    literal_topic: str | None = None,
    use_query_planner: bool = True,
    breadth: int | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "limit": limit,
        "exclude_video_ids": exclude_video_ids,
        "filters": {
            "creative_commons_only": bool(creative_commons_only),
            "duration": preferred_video_duration,
        },
        "language": language,
        "context": generation_context,
        "cache_store": generation_context.cache_store if generation_context is not None else None,
        "literal_topic": literal_topic or topic,
        "use_query_planner": bool(use_query_planner),
    }
    if breadth is not None:
        kwargs["breadth"] = max(1, int(breadth))
    if level is not None:
        kwargs["level"] = level
    if should_cancel is not None:
        kwargs["should_cancel"] = should_cancel
    return clip_engine_search.discover(topic, **kwargs)


def _is_valid_timestamped_supadata_transcript(transcript: dict[str, Any]) -> bool:
    """Verify the provenance markers and cue invariants used by the final gate."""
    if (
        transcript.get("source") != "supadata"
        or not str(transcript.get("artifact_key") or "").strip()
        or not isinstance(transcript.get("native_mode"), bool)
    ):
        return False
    segments = transcript.get("segments")
    if not isinstance(segments, list) or not segments:
        return False

    seen_ids: set[str] = set()
    previous_start = -1.0
    previous_end = -1.0
    for cue in segments:
        if not isinstance(cue, dict):
            return False
        cue_id = str(cue.get("cue_id") or "").strip()
        text = " ".join(str(cue.get("text") or "").split()).strip()
        try:
            start = float(cue.get("start"))
            end = float(cue.get("end"))
        except (TypeError, ValueError):
            return False
        if (
            not cue_id
            or cue_id in seen_ids
            or not text
            or not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or end <= start
            or start + 1e-9 < previous_start
            or end + 1e-9 < previous_end
        ):
            return False
        seen_ids.add(cue_id)
        previous_start = start
        previous_end = end
    return True


def _strict_topic_clips(
    clips: list[dict[str, Any]],
    transcript: dict[str, Any],
    query_plan: SearchQueryPlan | None,
) -> list[dict[str, Any]]:
    """Keep trusted timestamped windows proven by the exact topic signature."""
    if query_plan is None:
        # Compatibility for injected/mocked discover implementations. The real
        # topic search path always returns a validated plan.
        return clips
    if not _is_valid_timestamped_supadata_transcript(transcript):
        return []

    kept: list[dict[str, Any]] = []
    for clip in clips:
        cue_ids = clip.get("cue_ids")
        text = clip_engine_bridge.cue_text(transcript, cue_ids)
        if not text:
            text = clip_engine_bridge.window_text(
                transcript,
                float(clip.get("start") or 0.0),
                float(clip.get("end") or 0.0),
            )
        evidence = topic_signature_evidence(text, query_plan)
        if not evidence:
            continue
        clip["topic_evidence_terms"] = evidence[:8]
        kept.append(clip)
    return kept


def _retrieval_search_context(
    *,
    requested_topic: str,
    corrected_topic: str,
    video: dict[str, Any],
    query_plan: SearchQueryPlan | None,
    creative_commons_only: bool,
    source_duration: str,
) -> dict[str, Any]:
    context: dict[str, Any] = {
        "requested_topic": requested_topic,
        "corrected_topic": corrected_topic,
        "creative_commons_only": bool(creative_commons_only),
        "source_duration": source_duration,
        "matched_queries": list(video.get("matched_queries") or [])[:12],
        "matched_query_families": list(video.get("matched_families") or [])[:12],
        "matched_query_provenance": dict(video.get("matched_query_provenance") or {}),
    }
    if query_plan is not None:
        context.update(
            {
                "literal_query": query_plan.literal_query,
                "canonical_query": query_plan.canonical_query,
                "query_plan_version": query_plan.version,
                "query_plan_ai_status": query_plan.ai_status,
            }
        )
    return context


# --------------------------------------------------------------------- #
# Per-platform sliding-window rate limiter
# --------------------------------------------------------------------- #


class _PlatformRateLimiter:
    """
    Process-wide, thread-safe sliding-window counter keyed by platform.

    This sits ON TOP of the per-IP `_enforce_rate_limit` in `main.py` — that one caps
    what any single client can ask for, while this caps our total outbound traffic to
    any one platform so we don't accidentally DoS IG/TT/YT when two clients pile on.
    """

    # (limit_count, window_seconds)
    _DEFAULTS: dict[str, tuple[int, float]] = {
        "yt": (15, 60.0),
        "ig": (10, 60.0),
        "tt": (10, 60.0),
    }

    def __init__(self, overrides: dict[str, tuple[int, float]] | None = None) -> None:
        self._limits = {**self._DEFAULTS, **(overrides or {})}
        self._windows: dict[str, collections.deque[float]] = {p: collections.deque() for p in self._limits}
        self._lock = threading.Lock()

    def acquire(self, platform: str) -> None:
        limit_window = self._limits.get(platform)
        if limit_window is None:
            return
        limit, window = limit_window
        now = time.monotonic()
        cutoff = now - window
        with self._lock:
            deque_ = self._windows.setdefault(platform, collections.deque())
            while deque_ and deque_[0] < cutoff:
                deque_.popleft()
            if len(deque_) >= limit:
                oldest = deque_[0]
                retry_after = max(1.0, window - (now - oldest))
                raise RateLimitedError(
                    f"ingestion rate limit for platform={platform} exceeded",
                    retry_after_sec=retry_after,
                    detail=f"limit={limit}/{int(window)}s",
                )
            deque_.append(now)


# --------------------------------------------------------------------- #
# IngestionPipeline
# --------------------------------------------------------------------- #


class IngestionPipeline:
    """
    Orchestrates a single-URL or feed ingestion run.

    Dependencies are injected at construction time so the pipeline can be unit-tested
    with mocked services (see `backend/tests/test_ingestion_url.py`).
    """

    def __init__(
        self,
        *,
        youtube_service: Any,
        embedding_service: Any,
        settings: Any = None,
        rate_limiter: _PlatformRateLimiter | None = None,
        serverless_mode: bool | None = None,
    ) -> None:
        self._youtube_service = youtube_service
        self._embedding_service = embedding_service
        self._settings = settings
        self._openai_client = None
        self._rate_limiter = rate_limiter or _PlatformRateLimiter()
        if serverless_mode is None:
            serverless_mode = bool(
                os.environ.get("VERCEL")
                or os.environ.get("AWS_LAMBDA_FUNCTION_NAME")
                or os.environ.get("K_SERVICE")
            )
        self._serverless_mode = serverless_mode

    # --------------------------------------------------------------------- #
    # Single-URL ingest
    # --------------------------------------------------------------------- #

    def ingest_url(
        self,
        *,
        source_url: str,
        material_id: str | None = None,
        concept_id: str | None = None,
        target_clip_duration_sec: int = 45,
        target_clip_duration_min_sec: int = 15,
        target_clip_duration_max_sec: int = 60,
        language: str = "en",
        trace_id: str | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> IngestResult:
        raise_if_cancelled(should_cancel)
        effective_trace = set_trace_id(trace_id or new_trace_id())
        log_event(logger, logging.INFO, "ingest_start", source_url=source_url, material_id=material_id, concept_id=concept_id)
        started = time.monotonic()

        video_id = clip_engine_meta.extract_video_id(source_url)
        if not video_id:
            raise UnsupportedSourceError("Only YouTube URLs are supported.")

        self._rate_limiter.acquire("yt")

        try:
            engine_out = _run_clip(
                source_url,
                # concept_id is an opaque row id, NOT a topic — it must never
                # steer segmentation (it flows to _persist_ingest for row
                # association only, like ingest_feed).
                topic="",
                language=language,
                should_cancel=should_cancel,
            )
        except _ClipCancellationError:
            raise
        except _ClipUnsupportedURLError as exc:
            raise UnsupportedSourceError(str(exc)) from exc
        except _ClipTranscriptError as exc:
            raise TranscriptionError(str(exc)) from exc
        except _ClipError as exc:
            raise SegmentationError(str(exc)) from exc

        clips = engine_out["clips"]
        if not clips:
            raise SegmentationError("no on-topic clip could be produced")

        best = clip_engine_bridge.pick_best_clip(clips, target_clip_duration_sec, target_clip_duration_max_sec)

        raise_if_cancelled(should_cancel)
        meta = clip_engine_meta.youtube_metadata(video_id) or {}
        if not meta.get("duration_sec"):
            meta["duration_sec"] = engine_out["transcript"].get("duration")

        adapter_result = clip_engine_bridge.synth_adapter_result(video_id, source_url)
        metadata = clip_engine_bridge.to_metadata(video_id, meta, source_url)
        cues = clip_engine_bridge.to_cues(engine_out["transcript"])
        chosen = clip_engine_bridge.to_segment(best, engine_out["transcript"])
        snippet = clip_engine_bridge.window_text(engine_out["transcript"], chosen.t_start, chosen.t_end)[:7000]

        persisted = self._persist_ingest(
            adapter_result=adapter_result,
            metadata=metadata,
            cues=cues,
            chosen=chosen,
            snippet=snippet,
            material_id=material_id,
            concept_id=concept_id,
            clip_window=(chosen.t_start, chosen.t_end),
            target_max=int(target_clip_duration_max_sec),
            clip_details=best,
            should_cancel=should_cancel,
        )

        elapsed_ms = int((time.monotonic() - started) * 1000)
        log_event(
            logger,
            logging.INFO,
            "ingest_completed",
            source_url=source_url,
            reel_id=persisted.reel_id,
            platform="yt",
            source_id=video_id,
            t_start=chosen.t_start,
            t_end=chosen.t_end,
            elapsed_ms=elapsed_ms,
            author_handle=metadata.author_handle,
            author_name=metadata.author_name,
            duration_sec=metadata.duration_sec,
        )

        return IngestResult(
            reel=persisted,
            metadata=metadata,
            terms_notice=TERMS_NOTICE,
            trace_id=effective_trace,
        )

    # --------------------------------------------------------------------- #
    # Topic-aware multi-reel cut
    # --------------------------------------------------------------------- #

    def ingest_topic_cut(
        self,
        *,
        source_url: str,
        material_id: str | None = None,
        concept_id: str | None = None,
        language: str = "en",
        use_llm: bool = True,
        query: str | None = None,
        trace_id: str | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> IngestTopicCutResult:
        """
        Topic-aware variant of `ingest_url` that emits MULTIPLE reels per video.

        Routes through the clip engine (same as `ingest_url`), then applies
        query-relevance filtering via `clip_engine_bridge.filter_by_query`.
        Each kept clip is persisted via `_persist_ingest`, producing a list of
        `ReelOutWithAttribution` rows that decode cleanly into the iOS Reel struct.

        `use_llm` is accepted for API signature compatibility — the clip engine
        always uses its internal LLM; this param has no effect.
        """
        raise_if_cancelled(should_cancel)
        effective_trace = set_trace_id(trace_id or new_trace_id())
        log_event(
            logger,
            logging.INFO,
            "ingest_topic_cut_start",
            source_url=source_url,
            material_id=material_id,
            concept_id=concept_id,
            use_llm=use_llm,
        )
        started = time.monotonic()

        video_id = clip_engine_meta.extract_video_id(source_url)
        if not video_id:
            raise UnsupportedSourceError("Only YouTube URLs are supported.")

        self._rate_limiter.acquire("yt")

        try:
            engine_out = _run_clip(
                source_url,
                topic=(query or ""),
                language=language,
                should_cancel=should_cancel,
            )
        except _ClipCancellationError:
            raise
        except _ClipUnsupportedURLError as exc:
            raise UnsupportedSourceError(str(exc)) from exc
        except _ClipTranscriptError as exc:
            raise TranscriptionError(str(exc)) from exc
        except _ClipError as exc:
            raise SegmentationError(str(exc)) from exc

        kept = clip_engine_bridge.filter_by_query(
            engine_out["clips"], engine_out["transcript"], query
        )

        meta = clip_engine_meta.youtube_metadata(video_id) or {}
        duration = float(
            meta.get("duration_sec") or engine_out["transcript"].get("duration") or 0.0
        )

        is_short = bool(duration and duration < 60.0 and not kept)

        reels: list[ReelOutWithAttribution] = []
        metadata = clip_engine_bridge.to_metadata(video_id, meta, source_url)

        if not is_short and kept:
            adapter_result = clip_engine_bridge.synth_adapter_result(video_id, source_url)
            cues = clip_engine_bridge.to_cues(engine_out["transcript"])

            for clip in kept:
                raise_if_cancelled(should_cancel)
                chosen = clip_engine_bridge.to_segment(clip, engine_out["transcript"])
                snippet = clip_engine_bridge.window_text(
                    engine_out["transcript"], chosen.t_start, chosen.t_end
                )[:7000]
                persisted = self._persist_ingest(
                    adapter_result=adapter_result,
                    metadata=metadata,
                    cues=cues,
                    chosen=chosen,
                    snippet=snippet,
                    material_id=material_id,
                    concept_id=concept_id,
                    clip_window=(chosen.t_start, chosen.t_end),
                    target_max=180,
                    clip_details=clip,
                    should_cancel=should_cancel,
                )
                reels.append(persisted)

        elapsed_ms = int((time.monotonic() - started) * 1000)
        log_event(
            logger,
            logging.INFO,
            "ingest_topic_cut_completed",
            source_url=source_url,
            video_id=video_id,
            is_short=is_short,
            reel_count=len(reels),
            elapsed_ms=elapsed_ms,
        )

        return IngestTopicCutResult(
            source_url=source_url,
            video_id=video_id,
            is_short=is_short,
            classification_reason=("short" if is_short else "long-form"),
            duration_sec=duration,
            reel_count=len(reels),
            reels=reels,
            metadata=metadata,
            terms_notice=TERMS_NOTICE,
            trace_id=effective_trace,
        )

    # --------------------------------------------------------------------- #
    # Feed ingest
    # --------------------------------------------------------------------- #

    def ingest_feed(
        self,
        *,
        feed_url: str,
        max_items: int = 6,
        material_id: str | None = None,
        concept_id: str | None = None,
        target_clip_duration_sec: int = 45,
        target_clip_duration_min_sec: int = 15,
        target_clip_duration_max_sec: int = 60,
        language: str = "en",
        trace_id: str | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> IngestFeedResult:
        raise_if_cancelled(should_cancel)
        effective_trace = set_trace_id(trace_id or new_trace_id())
        log_event(logger, logging.INFO, "ingest_feed_start", feed_url=feed_url)
        self._rate_limiter.acquire("yt")

        urls = (
            [feed_url]
            if clip_engine_meta.extract_video_id(feed_url)
            else clip_engine_meta.resolve_feed_urls(feed_url, max_items)
        )
        raise_if_cancelled(should_cancel)

        items: list[IngestFeedItem] = []
        succeeded = 0
        failed = 0

        for url in urls:
            raise_if_cancelled(should_cancel)
            try:
                video_id = clip_engine_meta.extract_video_id(url)
                engine_out = _run_clip(
                    url, topic="", language=language, should_cancel=should_cancel
                )
                if not engine_out["clips"]:
                    items.append(IngestFeedItem(source_url=url, status="skipped"))
                    continue

                best = clip_engine_bridge.pick_best_clip(engine_out["clips"], target_clip_duration_sec, target_clip_duration_max_sec)

                meta = clip_engine_meta.youtube_metadata(video_id) or {}
                if not meta.get("duration_sec"):
                    meta["duration_sec"] = engine_out["transcript"].get("duration")

                adapter_result = clip_engine_bridge.synth_adapter_result(video_id, url)
                metadata = clip_engine_bridge.to_metadata(video_id, meta, url)
                cues = clip_engine_bridge.to_cues(engine_out["transcript"])
                chosen = clip_engine_bridge.to_segment(best, engine_out["transcript"])
                snippet = clip_engine_bridge.window_text(
                    engine_out["transcript"], chosen.t_start, chosen.t_end
                )[:7000]

                persisted = self._persist_ingest(
                    adapter_result=adapter_result,
                    metadata=metadata,
                    cues=cues,
                    chosen=chosen,
                    snippet=snippet,
                    material_id=material_id,
                    concept_id=concept_id,
                    clip_window=(chosen.t_start, chosen.t_end),
                    target_max=int(target_clip_duration_max_sec),
                    clip_details=best,
                    should_cancel=should_cancel,
                )

                items.append(IngestFeedItem(
                    source_url=url,
                    status="ok",
                    reel=persisted,
                    metadata=metadata,
                ))
                succeeded += 1

            except _ClipCancellationError:
                raise
            except Exception as exc:
                failed += 1
                items.append(IngestFeedItem(source_url=url, status="error", error=str(exc)))

        log_event(logger, logging.INFO, "ingest_feed_completed", feed_url=feed_url, total_resolved=len(urls), succeeded=succeeded, failed=failed)
        return IngestFeedResult(
            feed_url=feed_url,
            total_resolved=len(urls),
            succeeded=succeeded,
            failed=failed,
            items=items,
            terms_notice=TERMS_NOTICE,
            trace_id=effective_trace,
        )

    # --------------------------------------------------------------------- #
    # Topic search — multi-platform fan-out
    # --------------------------------------------------------------------- #

    def ingest_search(
        self,
        *,
        query: str,
        platforms: list[PlatformLiteral] | None = None,
        max_per_platform: int = 5,
        material_id: str | None = None,
        concept_id: str | None = None,
        target_clip_duration_sec: int = 45,
        target_clip_duration_min_sec: int = 15,
        target_clip_duration_max_sec: int = 60,
        language: str = "en",
        exclude_video_ids: list[str] | None = None,
        trace_id: str | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> IngestSearchResult:
        """
        Topic-based multi-platform search.

        Flow:
          1. For each requested platform, build a search URL via the adapter and call
             resolve_feed() to get a list of reel URLs. Failures on one platform do NOT
             abort the others — they're recorded in `per_platform_errors`.
          2. Dedup resolved URLs across platforms and against `exclude_video_ids`.
          3. Create (or reuse) a query-scoped sentinel material so the resulting feed
             is browsable via the existing /api/feed?material_id=... endpoint.
          4. Fan out ingest_url() calls across a thread pool. Each call goes through
             the full pipeline (download → transcribe → segment → persist).
          5. Collect results into a batch response. Infinite-scroll pagination is the
             caller's responsibility: pass every seen reel's source_id in
             `exclude_video_ids` on subsequent calls.

        This method is stateless w.r.t. concurrency — two callers with the same query
        get the same sentinel material and can race on the reels unique index, which
        is handled correctly by ingest_url() via load_existing_reel.
        """
        raise_if_cancelled(should_cancel)
        effective_trace = set_trace_id(trace_id or new_trace_id())

        # Coerce to YouTube-only regardless of what the caller requested.
        resolved_material_id = material_id or self._ensure_search_material(query)
        limit = min(int(max_per_platform), clip_engine_config.CLIP_SEARCH_MAX_VIDEOS)

        self._rate_limiter.acquire("yt")

        disc = _discover(
            query,
            limit=limit,
            exclude_video_ids=exclude_video_ids or [],
            level=None,
            should_cancel=should_cancel,
            literal_topic=query,
            use_query_planner=True,
            breadth=3,
        )
        query_plan = disc.get("query_plan")
        if not isinstance(query_plan, SearchQueryPlan):
            query_plan = None

        _search_warning = disc.get("warning")
        if _search_warning:
            log_event(logger, logging.WARNING, "ingest_search_warning", warning=_search_warning)

        items: list[IngestSearchItem] = []
        succeeded = 0
        failed = 0

        for v in disc["videos"]:
            raise_if_cancelled(should_cancel)
            try:
                v["_search_context"] = _retrieval_search_context(
                    requested_topic=query,
                    corrected_topic=str(disc.get("corrected") or query),
                    video=v,
                    query_plan=query_plan,
                    creative_commons_only=False,
                    source_duration="any",
                )
                engine_out = _run_clip(
                    v["url"], topic=query, language=language,
                    should_cancel=should_cancel,
                )
                eligible_clips = _strict_topic_clips(
                    list(engine_out["clips"]),
                    engine_out["transcript"],
                    query_plan,
                )
                if not eligible_clips:
                    items.append(IngestSearchItem(
                        platform="yt", source_url=v["url"], status="skipped"
                    ))
                    continue

                best = clip_engine_bridge.pick_best_clip(
                    eligible_clips,
                    target_clip_duration_sec,
                    target_clip_duration_max_sec,
                )
                best["search_context"] = {
                    **dict(v.get("_search_context") or {}),
                    "topic_evidence_terms": list(best.get("topic_evidence_terms") or [])[:8],
                }

                persisted, metadata = self._persist_engine_clip(
                    v=v,
                    clip=best,
                    engine_out=engine_out,
                    material_id=resolved_material_id,
                    concept_id=concept_id,
                    target_max=int(target_clip_duration_max_sec),
                    should_cancel=should_cancel,
                )

                items.append(IngestSearchItem(
                    platform="yt",
                    source_url=v["url"],
                    status="ok",
                    reel=persisted,
                    metadata=metadata,
                ))
                succeeded += 1

            except _ClipCancellationError:
                raise
            except Exception as exc:
                failed += 1
                items.append(IngestSearchItem(
                    platform="yt",
                    source_url=v["url"],
                    status="error",
                    error=str(exc),
                ))

        return IngestSearchResult(
            query=query,
            material_id=resolved_material_id,
            platforms=["yt"],
            per_platform_resolved={"yt": len(disc["videos"])},
            per_platform_succeeded={"yt": succeeded},
            per_platform_failed={"yt": failed},
            per_platform_errors={"yt": _search_warning} if _search_warning else {},
            total_resolved=len(disc["videos"]),
            succeeded=succeeded,
            failed=failed,
            items=items,
            terms_notice=TERMS_NOTICE + " Search is YouTube-only.",
            trace_id=effective_trace,
        )

    # --------------------------------------------------------------------- #
    # Material topic — multi-clip, one concept per call
    # --------------------------------------------------------------------- #

    def ingest_topic(
        self,
        *,
        topic: str,
        material_id: str,
        concept_id: str,
        generation_id: str | None = None,
        exclude_video_ids: list[str] | None = None,
        target_clip_duration_sec: int = 45,
        target_clip_duration_min_sec: int = 15,
        target_clip_duration_max_sec: int = 60,
        language: str = "en",
        knowledge_level: str | None = None,
        max_videos: int = 3,
        max_reels: int | None = None,
        on_reel_created: Callable[[ReelOutWithAttribution], None] | None = None,
        dry_run: bool = False,
        should_cancel: Callable[[], bool] | None = None,
        creative_commons_only: bool = False,
        preferred_video_duration: str = "any",
        generation_context: GenerationContext | None = None,
        literal_topic: str | None = None,
    ) -> tuple[list[ReelOutWithAttribution], list[str]]:
        """
        Route ONE study concept through the clip engine and persist EVERY
        relevance-surviving clip per video (multiple reels per video), unlike
        `ingest_search`'s one-best `pick_best_clip`. This is the per-concept
        engine the material→reels rewire calls.

        Returns `(reels, resolved_video_ids)`: `reels` in discover order then
        clip order within a video; `resolved_video_ids` = the video ids
        `discover` returned (a viability probe callers consume even under
        `dry_run`).

        Cost/latency guardrail: `max_videos` bounds the paid `run.clip` calls.
        Without `max_reels`, each video's clips persist as soon as that video
        finishes; the returned list is restored to discover order. With
        `max_reels`, fetch and persistence stay in discover order so the capped
        selection remains deterministic.
        """
        topic = " ".join(str(topic or "").split())
        if not topic:
            raise UnsupportedSourceError("A non-blank YouTube search topic is required.")
        raise_if_cancelled(should_cancel)
        limit = min(int(max_videos), clip_engine_config.CLIP_SEARCH_MAX_VIDEOS)

        self._rate_limiter.acquire("yt")

        # Defensively strip any `yt:`-prefixed ids (e.g. prior-generation reel rows
        # wired into the caller's exclusions) so a prefixed id can't leak into the
        # Supadata discover query, where it would never match a bare source id.
        bare_exclusions = [
            str(v or "").strip().split(":", 1)[-1]
            for v in (exclude_video_ids or [])
            if str(v or "").strip()
        ]

        try:
            disc = _discover(
                topic, limit=limit, exclude_video_ids=bare_exclusions, level=knowledge_level,
                should_cancel=should_cancel,
                creative_commons_only=creative_commons_only,
                preferred_video_duration=preferred_video_duration,
                language=language,
                generation_context=generation_context,
                literal_topic=literal_topic or topic,
                use_query_planner=True,
                breadth=3 if generation_context is not None and generation_context.budget.mode == "fast" else 6,
            )
        except _ClipProviderError:
            if generation_context is not None:
                generation_context.increment_counter("provider_failures")
            raise

        warning = disc.get("warning")
        if warning:
            log_event(logger, logging.WARNING, "ingest_topic_warning", warning=warning)

        corrected_topic = " ".join(str(disc.get("corrected") or topic).split()) or topic
        query_plan = disc.get("query_plan")
        if not isinstance(query_plan, SearchQueryPlan):
            query_plan = None
        for discovered_video in disc["videos"]:
            discovered_video["_search_context"] = _retrieval_search_context(
                requested_topic=topic,
                corrected_topic=corrected_topic,
                video=discovered_video,
                query_plan=query_plan,
                creative_commons_only=creative_commons_only,
                source_duration=preferred_video_duration,
            )
            if query_plan is not None:
                discovered_video["_query_plan"] = query_plan
        resolved_video_ids = [v["id"] for v in disc["videos"]]
        if generation_context is not None:
            generation_context.increment_counter("discovered_videos", len(resolved_video_ids))

        # dry_run: discover-only viability probe — no run.clip, no DB writes.
        if dry_run:
            return [], resolved_video_ids

        videos = disc["videos"]

        if not videos:
            return [], resolved_video_ids

        # FETCH stage: clip + relevance-filter each video concurrently under one
        # shared deadline. shutdown(wait=False) abandons stuck worker threads.
        executor = ThreadPoolExecutor(max_workers=min(4, len(videos)))
        batch_cancelled = threading.Event()

        def fetch_should_cancel() -> bool:
            return batch_cancelled.is_set() or is_cancelled(should_cancel)

        deadline = time.monotonic() + max(0.0, INGEST_TOPIC_VIDEO_TIMEOUT_SEC)
        for video in videos:
            video["_deadline_monotonic"] = deadline
        futures = [
            executor.submit(
                self._clip_and_filter,
                v,
                corrected_topic,
                language,
                fetch_should_cancel,
                generation_context,
            )
            for v in videos
        ]

        def fetch_result(v: dict[str, Any], future: Any, timeout: float):
            try:
                return future.result(timeout=timeout)
            except _ClipCancellationError:
                if batch_cancelled.is_set() and not is_cancelled(should_cancel):
                    return None
                raise
            except FutureTimeoutError:
                batch_cancelled.set()
                if generation_context is not None:
                    generation_context.increment_counter("transcript_timeouts")
                log_event(
                    logger,
                    logging.WARNING,
                    "ingest_topic_video_failed",
                    video_id=v.get("id"),
                    error=(
                        "shared clip fetch deadline exceeded "
                        f"({INGEST_TOPIC_VIDEO_TIMEOUT_SEC:g}s)"
                    ),
                )
            except _TranscriptUnavailableError as exc:
                if generation_context is not None:
                    generation_context.increment_counter("transcript_failures")
                    generation_context.increment_counter("provider_failures")
                log_event(
                    logger,
                    logging.INFO,
                    "ingest_topic_transcript_unavailable",
                    video_id=v.get("id"),
                    error=str(exc),
                )
            except _ClipTranscriptError as exc:
                if generation_context is not None:
                    generation_context.increment_counter("transcript_failures")
                log_event(
                    logger,
                    logging.INFO,
                    "ingest_topic_transcript_unavailable",
                    video_id=v.get("id"),
                    error=str(exc),
                )
            except _ClipProviderError as exc:
                if generation_context is not None:
                    generation_context.increment_counter("provider_failures")
                    is_transcript_failure = (
                        str(getattr(exc, "operation", "")).casefold() == "transcript"
                    )
                    if is_transcript_failure:
                        generation_context.increment_counter("transcript_failures")
                    message = f"{exc} {getattr(exc, 'detail', '') or ''}".casefold()
                    if (
                        is_transcript_failure
                        and ("timed out" in message or "timeout" in message or "deadline" in message)
                    ):
                        generation_context.increment_counter("transcript_timeouts")
                raise
            except _ClipError as exc:
                log_event(
                    logger,
                    logging.WARNING,
                    "ingest_topic_video_failed",
                    video_id=v.get("id"),
                    error=str(exc),
                )
            except Exception as exc:
                log_event(
                    logger,
                    logging.WARNING,
                    "ingest_topic_video_failed",
                    video_id=v.get("id"),
                    error=str(exc),
                )
            return None

        def persist_result(
            result: tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]],
            *,
            limit: int | None = None,
        ) -> list[ReelOutWithAttribution]:
            v, kept, engine_out = result
            persisted: list[ReelOutWithAttribution] = []
            for clip in kept if limit is None else kept[:max(0, limit)]:
                raise_if_cancelled(should_cancel)
                reel, _ = self._persist_engine_clip(
                    v=v,
                    clip=clip,
                    engine_out=engine_out,
                    material_id=material_id,
                    concept_id=concept_id,
                    target_max=int(target_clip_duration_max_sec),
                    generation_id=generation_id,
                    should_cancel=should_cancel,
                )
                persisted.append(reel)
                if generation_context is not None:
                    generation_context.increment_counter("persisted_clips")
                if on_reel_created is not None:
                    on_reel_created(reel)
            return persisted

        fetched: list[tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]] = []
        reels_by_video: dict[int, list[ReelOutWithAttribution]] = {}
        provider_errors: list[_ClipProviderError] = []
        try:
            # A numeric cap preserves the prior discover-order selection contract.
            if max_reels is not None:
                for v, future in zip(videos, futures):
                    raise_if_cancelled(should_cancel)
                    remaining = max(0.0, deadline - time.monotonic())
                    result = fetch_result(v, future, remaining)
                    if result is not None:
                        fetched.append(result)
            else:
                # Uncapped primary generation persists each completed video now.
                pending = {
                    future: (index, v)
                    for index, (v, future) in enumerate(zip(videos, futures))
                }
                while pending:
                    raise_if_cancelled(should_cancel)
                    remaining = max(0.0, deadline - time.monotonic())
                    done, _ = wait(
                        pending,
                        timeout=min(0.01, remaining),
                        return_when=FIRST_COMPLETED,
                    )
                    if not done:
                        if time.monotonic() < deadline:
                            continue
                        for _, v in pending.values():
                            log_event(
                                logger,
                                logging.WARNING,
                                "ingest_topic_video_failed",
                                video_id=v.get("id"),
                                error=(
                                    "shared clip fetch deadline exceeded "
                                    f"({INGEST_TOPIC_VIDEO_TIMEOUT_SEC:g}s)"
                                ),
                            )
                        if generation_context is not None:
                            generation_context.increment_counter(
                                "transcript_timeouts", len(pending)
                            )
                        break

                    for future in sorted(done, key=lambda item: pending[item][0]):
                        index, v = pending.pop(future)
                        try:
                            result = fetch_result(v, future, 0.0)
                        except _ClipProviderError as exc:
                            provider_errors.append(exc)
                            log_event(
                                logger,
                                logging.WARNING,
                                "ingest_topic_video_failed",
                                video_id=v.get("id"),
                                error=str(exc),
                            )
                            continue
                        if result is not None:
                            reels_by_video[index] = persist_result(result)
        finally:
            batch_cancelled.set()
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)

        if max_reels is not None:
            reels: list[ReelOutWithAttribution] = []
            for result in fetched:
                if len(reels) >= max_reels:
                    break
                reels.extend(persist_result(result, limit=max_reels - len(reels)))
            return reels, resolved_video_ids

        # Progressive callbacks reflect availability; restore discover order in
        # the final result for deterministic downstream inventory.
        reels = [
            reel
            for index in range(len(videos))
            for reel in reels_by_video.get(index, [])
        ]
        if not reels and provider_errors:
            raise provider_errors[0]
        return reels, resolved_video_ids

    def _clip_and_filter(
        self, v: dict[str, Any], topic: str, language: str,
        should_cancel: Callable[[], bool] | None = None,
        generation_context: GenerationContext | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
        """Fetch ONE discovered video's clips, score each (query relevance
        blended with the engine's informativeness so one-word topics still get
        a ranking signal). Returns `(v, scored_clips,
        engine_out)`, scored_clips sorted by score DESCENDING. Empty `clips`
        yields no clips (video is skipped)."""
        engine_out = _run_clip(
            v["url"], topic=topic, language=language,
            should_cancel=should_cancel,
            generation_context=generation_context,
            deadline_monotonic=v.get("_deadline_monotonic"),
        )
        transcript = engine_out["transcript"]
        query_plan = (
            v.get("_query_plan")
            if isinstance(v.get("_query_plan"), SearchQueryPlan)
            else None
        )
        trusted_transcript = _is_valid_timestamped_supadata_transcript(transcript)
        if generation_context is not None and trusted_transcript:
            generation_context.increment_counter("usable_transcripts")
        elif generation_context is not None and query_plan is not None:
            generation_context.increment_counter("transcript_failures")

        raw_clips = list(engine_out["clips"])
        if not raw_clips:
            if generation_context is not None:
                generation_context.increment_counter("gemini_empty_results")
            return v, [], engine_out
        eligible_clips = _strict_topic_clips(
            raw_clips,
            transcript,
            query_plan,
        )
        if generation_context is not None and trusted_transcript:
            generation_context.increment_counter(
                "topic_rejections", max(0, len(raw_clips) - len(eligible_clips))
            )
        for clip in eligible_clips:
            relevance = clip_engine_bridge.relevance_score(
                clip, engine_out["transcript"], topic
            )
            raw_info = clip.get("informativeness")
            informativeness = (
                0.5 if raw_info is None else max(0.0, min(1.0, float(raw_info)))
            )
            clip["score"] = relevance * (0.5 + 0.5 * informativeness)
            clip["search_context"] = {
                **dict(v.get("_search_context") or {}),
                "topic_evidence_terms": list(clip.get("topic_evidence_terms") or [])[:8],
            }
        kept = sorted(eligible_clips, key=lambda c: c["score"], reverse=True)
        return v, kept, engine_out

    def _ensure_search_material(self, query: str) -> str:
        """
        Idempotently create a query-scoped sentinel material so /api/feed can scope to
        one specific topic search's results. The material id is deterministic from
        the query text so re-running the same search reuses the same material.
        """
        normalized = " ".join((query or "").strip().lower().split())
        if not normalized:
            return "ingest-search:empty"
        query_hash = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]
        material_id = f"ingest-search:{query_hash}"
        concept_id = f"{material_id}:concept"

        try:
            with get_conn(transactional=True) as conn:
                try:
                    upsert(
                        conn,
                        "materials",
                        {
                            "id": material_id,
                            "subject_tag": normalized[:200],
                            "raw_text": (query or "")[:2000],
                            "source_type": "ingest-search",
                            "source_path": None,
                            "created_at": now_iso(),
                        },
                        pk="id",
                    )
                except DatabaseIntegrityError:
                    pass
                try:
                    keywords = [tok for tok in normalized.split() if tok][:10]
                    upsert(
                        conn,
                        "concepts",
                        {
                            "id": concept_id,
                            "material_id": material_id,
                            "title": (query or "").strip()[:200] or "Search",
                            "keywords_json": dumps_json(keywords),
                            "summary": "",
                            "embedding_json": None,
                            "created_at": now_iso(),
                        },
                        pk="id",
                    )
                except DatabaseIntegrityError:
                    pass
        except Exception:
            logger.exception("failed to ensure search material for query=%s", normalized)

        return material_id

    # --------------------------------------------------------------------- #
    # Helpers that need a DB connection
    # --------------------------------------------------------------------- #

    def _persist_engine_clip(
        self,
        *,
        v: dict[str, Any],
        clip: dict[str, Any],
        engine_out: dict[str, Any],
        material_id: str | None,
        concept_id: str | None,
        target_max: int,
        generation_id: str | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> tuple[ReelOutWithAttribution, IngestMetadata]:
        """
        Build the persist inputs for ONE engine clip of a discovered video `v`
        (source metadata from the discover dict, transcript fallbacks from
        `engine_out`) and persist it. Shared by `ingest_search` (its one
        `pick_best_clip`) and `ingest_topic` (every surviving clip).

        Returns `(reel, metadata)`; callers that only want the reel ignore the
        second element.
        """
        meta = {
            "title": v.get("title", ""),
            "author_name": v.get("channel", ""),
            "duration_sec": v.get("duration") or engine_out["transcript"].get("duration"),
            "thumbnail_url": v.get("thumbnail", ""),
            "view_count": v.get("view_count"),
            "upload_date_iso": v.get("upload_date"),
        }

        adapter_result = clip_engine_bridge.synth_adapter_result(v["id"], v["url"])
        metadata = clip_engine_bridge.to_metadata(v["id"], meta, v["url"])
        cues = clip_engine_bridge.to_cues(engine_out["transcript"])
        chosen = clip_engine_bridge.to_segment(clip, engine_out["transcript"])
        snippet = clip_engine_bridge.window_text(
            engine_out["transcript"], chosen.t_start, chosen.t_end
        )[:7000]

        reel = self._persist_ingest(
            adapter_result=adapter_result,
            metadata=metadata,
            cues=cues,
            chosen=chosen,
            snippet=snippet,
            material_id=material_id,
            concept_id=concept_id,
            clip_window=(chosen.t_start, chosen.t_end),
            target_max=target_max,
            generation_id=generation_id,
            clip_title=str(clip.get("title") or "").strip(),
            clip_difficulty=(
                None if clip.get("difficulty") is None else float(clip["difficulty"])
            ),
            clip_details=clip,
            should_cancel=should_cancel,
        )
        return reel, metadata

    def _persist_ingest(
        self,
        *,
        adapter_result: YouTubeSourceRef,
        metadata: IngestMetadata,
        cues: list[IngestTranscriptCue],
        chosen: IngestSegment,
        snippet: str,
        material_id: str | None,
        concept_id: str | None,
        clip_window: tuple[float, float],
        target_max: int,
        generation_id: str | None = None,
        clip_title: str = "",
        clip_difficulty: float | None = None,
        clip_details: dict[str, Any] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> ReelOutWithAttribution:
        raise_if_cancelled(should_cancel)
        clip_start = round(float(clip_window[0]), 3)
        clip_end = round(float(clip_window[1]), 3)
        clip_duration = clip_end - clip_start
        if clip_duration < 1.0 or clip_duration > 180.0:
            raise SegmentationError("Clip must stay within the 1-180 second safety envelope.")
        from .persistence import build_video_id  # local import to avoid cycle surprises

        video_id = build_video_id(adapter_result.platform, adapter_result.source_id)

        # Build the client-facing YouTube embed URL. floor(start)/ceil(end) with
        # a >=1s guard matches the engine's embed_url —
        # int()-truncating the end cut up to ~1s off every reel's final word.
        embed_start = int(clip_start)
        embed_end = max(embed_start + 1, math.ceil(clip_end))
        video_url = (
            f"https://www.youtube.com/embed/{adapter_result.source_id}"
            f"?start={embed_start}&end={embed_end}"
            "&modestbranding=1&rel=0&playsinline=1"
        )

        details = clip_details if isinstance(clip_details, dict) else {}
        generated_takeaways = details.get("takeaways")
        takeaways: list[str] = []
        if isinstance(generated_takeaways, list):
            seen_takeaways: set[str] = set()
            for value in generated_takeaways:
                text = " ".join(str(value or "").split()).strip()
                key = text.casefold()
                if not text or key in seen_takeaways:
                    continue
                seen_takeaways.add(key)
                takeaways.append(text[:280])
                if len(takeaways) >= 4:
                    break
        if len(takeaways) < 2:
            takeaways = build_takeaways_for_ingest(
                concept_title=clip_title or metadata.title or "",
                transcript_snippet=snippet,
                hashtags=metadata.hashtags,
                limit=3,
            )
            if clip_title:
                takeaways = ([clip_title] + [t for t in takeaways if t != clip_title])[:3]

        ai_summary = " ".join(str(details.get("summary") or "").split()).strip()[:700]
        if not ai_summary:
            ai_summary = fallback_ai_summary(
                concept_title=clip_title or metadata.title or "",
                video_title=metadata.title or "",
                video_description=metadata.description,
                transcript_snippet=snippet,
                takeaways=takeaways,
            )
        match_reason = " ".join(str(details.get("match_reason") or "").split()).strip()[:700]
        if not match_reason:
            matched_idea = clip_title or (takeaways[0] if takeaways else "") or metadata.title or "this topic"
            match_reason = f"This clip directly explains {matched_idea[:180]} using the source transcript."
        try:
            informativeness = float(details.get("informativeness"))
        except (TypeError, ValueError):
            informativeness = 0.6
        informativeness = max(0.6, min(1.0, informativeness))
        assessment = details.get("assessment")
        raise_if_cancelled(should_cancel)

        with get_conn(transactional=True) as conn:
            raise_if_cancelled(should_cancel)
            effective_material_id, effective_concept_id = resolve_material_concept(
                conn,
                material_id=material_id,
                concept_id=concept_id,
            )

            tombstone = None
            try:
                tombstone = fetch_one(
                    conn,
                    "SELECT video_id FROM blocked_video_tombstones WHERE video_id = ?",
                    (str(adapter_result.source_id or "").strip(),),
                )
            except Exception as exc:
                if "blocked_video_tombstones" not in str(exc).lower():
                    raise
            if tombstone:
                raise BlockedVideoError("This YouTube video has been removed by takedown.")

            raise_if_cancelled(should_cancel)
            upsert_video(conn, platform=adapter_result.platform, source_id=adapter_result.source_id, metadata=metadata)

            raise_if_cancelled(should_cancel)
            reel_id = f"ingest-{uuid.uuid4().hex[:16]}"
            inserted = upsert_reel_row(
                conn,
                reel_id=reel_id,
                material_id=effective_material_id,
                concept_id=effective_concept_id,
                video_id=video_id,
                video_url=video_url,
                t_start=clip_start,
                t_end=clip_end,
                transcript_snippet=snippet,
                takeaways=takeaways,
                base_score=float(chosen.score),
                generation_id=generation_id,
                difficulty=clip_difficulty,
                ai_summary=ai_summary,
                match_reason=match_reason,
                informativeness=informativeness,
                model_used=str(details.get("model_used") or ""),
                quality_degraded=bool(details.get("quality_degraded", False)),
                selected_cue_ids=[
                    str(cue_id)
                    for cue_id in (details.get("cue_ids") or [])
                    if str(cue_id or "").strip()
                ],
                search_context=(
                    dict(details.get("search_context") or {})
                    if isinstance(details.get("search_context"), dict)
                    else {}
                ),
            )

            if not inserted:
                # Unique index collision — load the existing row and reuse it.
                existing = load_existing_reel(
                    conn,
                    material_id=effective_material_id,
                    concept_id=effective_concept_id,
                    video_id=video_id,
                    t_start=clip_start,
                    t_end=clip_end,
                    generation_id=generation_id,
                )
                if existing:
                    reel_id = existing["id"]
                    # Still store metadata blob (may have changed since prior ingest).
                else:
                    raise DatabaseIntegrityError(
                        "Reel insert reported a unique collision but no matching row exists."
                    )
            raise_if_cancelled(should_cancel)
            if isinstance(assessment, dict):
                store_reel_assessment_question(
                    conn,
                    reel_id=reel_id,
                    prompt=str(assessment.get("prompt") or ""),
                    options=list(assessment.get("options") or []),
                    correct_index=assessment.get("correct_index"),
                    explanation=str(assessment.get("explanation") or ""),
                )
            raise_if_cancelled(should_cancel)
            store_ingest_metadata_blob(conn, reel_id=reel_id, metadata=metadata)
            raise_if_cancelled(should_cancel)

        clip_duration = max(0.0, clip_end - clip_start)

        # Window the whole-video cues to this clip's [start, end] and rebase to
        # clip-relative timestamps (legacy `_build_caption_cues` semantics), so the
        # client renders captions aligned to the trimmed clip, not the source video.
        captions = []
        for cue in cues:
            if not cue.text:
                continue
            if cue.end <= clip_start or cue.start >= clip_end:
                continue  # no overlap with the clip window
            captions.append(
                {
                    "start": max(0.0, cue.start - clip_start),
                    "end": min(clip_duration, cue.end - clip_start),
                    "text": cue.text,
                }
            )

        attribution = format_attribution(metadata)

        return ReelOutWithAttribution(
            reel_id=reel_id,
            material_id=effective_material_id,
            concept_id=effective_concept_id,
            concept_title=clip_title or metadata.title or "",
            video_title=metadata.title or "",
            channel_name=metadata.author_name or "",
            video_description=metadata.description,
            ai_summary=ai_summary,
            match_reason=match_reason,
            informativeness=informativeness,
            video_url=video_url,
            t_start=float(clip_start),
            t_end=float(clip_end),
            transcript_snippet=snippet,
            takeaways=takeaways,
            captions=captions,
            score=float(chosen.score),
            relevance_score=None,
            discovery_score=None,
            clipability_score=float(chosen.score),
            query_strategy="",
            retrieval_stage="",
            source_surface=f"ingest:{adapter_result.platform}",
            matched_terms=[],
            relevance_reason="",
            concept_position=None,
            total_concepts=None,
            video_duration_sec=int(metadata.duration_sec) if metadata.duration_sec else None,
            clip_duration_sec=float(clip_duration),
            model_used=str(details.get("model_used") or ""),
            quality_degraded=bool(details.get("quality_degraded", False)),
            selected_cue_ids=[
                str(cue_id)
                for cue_id in (details.get("cue_ids") or [])
                if str(cue_id or "").strip()
            ],
            source_attribution=attribution,
        )


# --------------------------------------------------------------------- #
# CLI smoke test entry point
# --------------------------------------------------------------------- #


def _cli_main(argv: list[str] | None = None) -> int:  # pragma: no cover
    import argparse
    import json as _json

    parser = argparse.ArgumentParser(description="Smoke-test a single ReelAI ingest")
    parser.add_argument("source_url", help="URL to ingest")
    parser.add_argument("--target", type=int, default=45)
    parser.add_argument("--min", dest="min_sec", type=int, default=15)
    parser.add_argument("--max", dest="max_sec", type=int, default=60)
    parser.add_argument("--language", default="en")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    # Construct minimal services for the CLI path. In a running FastAPI process these
    # are wired via main.py at import time.
    from .. import db as db_module
    from ..config import get_settings
    from ..services.embeddings import EmbeddingService
    from ..services.youtube import YouTubeService

    settings = get_settings()
    db_module.init_db()

    embedding_service = EmbeddingService()
    youtube_service = YouTubeService(settings=settings)

    pipeline = IngestionPipeline(
        youtube_service=youtube_service,
        embedding_service=embedding_service,
        settings=settings,
    )

    result = pipeline.ingest_url(
        source_url=args.source_url,
        target_clip_duration_sec=args.target,
        target_clip_duration_min_sec=args.min_sec,
        target_clip_duration_max_sec=args.max_sec,
        language=args.language,
    )
    print(_json.dumps(result.model_dump(), indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys as _sys

    raise SystemExit(_cli_main(_sys.argv[1:]))


__all__ = ["IngestionPipeline", "_PlatformRateLimiter"]
