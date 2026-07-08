"""
IngestionPipeline — the orchestrator.

One class, two public methods:

  * `ingest_url(source_url, ...) -> IngestResult`
      Download → transcribe → segment → persist one reel for a single URL.

  * `ingest_feed(feed_url, max_items=6, ...) -> IngestFeedResult`
      Resolve a profile / hashtag / playlist URL to a list of individual reel URLs
      and call `ingest_url` for each with bounded concurrency.

The pipeline owns:
  * A `ThreadPoolExecutor(max_workers=2)` for `ingest_feed` (kept small because each item
    is disk+network+CPU heavy and we have per-platform rate limits on top).
  * A process-wide sliding-window rate limiter keyed by platform (`yt`/`ig`/`tt`).

The pipeline is stateless beyond those two things. It does NOT cache results itself —
transcripts and summaries cache inside their respective modules via `llm_cache`.
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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from ..db import DatabaseIntegrityError, dumps_json, get_conn, now_iso, upsert
from . import TERMS_NOTICE
from .adapters.base import AdapterResult
from .errors import (
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
    brief_ai_summary,
    build_takeaways_for_ingest,
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
)
from .persistence import (
    ensure_sentinel_concept,
    ensure_sentinel_material,
    load_existing_reel,
    store_ingest_metadata_blob,
    upsert_reel_row,
    upsert_video,
)
from ..clip_engine import run as clip_engine_run, search as clip_engine_search, bridge as clip_engine_bridge, metadata as clip_engine_meta, config as clip_engine_config  # noqa: F401
from ..clip_engine.errors import (
    ClipError as _ClipError,
    TranscriptError as _ClipTranscriptError,
    UnsupportedURLError as _ClipUnsupportedURLError,
)

logger: logging.Logger = get_ingest_logger(__name__)

# Feed curation: keep only the best clips of each discovered video (ranked by
# relevance x informativeness) instead of persisting the engine's whole tiling.
INGEST_TOPIC_MAX_CLIPS_PER_VIDEO = int(os.environ.get("INGEST_TOPIC_MAX_CLIPS_PER_VIDEO", "3"))
# Per-video wall-clock budget for clip+filter (VidScout's feed abandons a
# video's job after 180s); a pathological video must not stall the generation.
INGEST_TOPIC_VIDEO_TIMEOUT_SEC = float(os.environ.get("INGEST_TOPIC_VIDEO_TIMEOUT_SEC", "180"))


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
    ) -> IngestResult:
        effective_trace = set_trace_id(trace_id or new_trace_id())
        log_event(logger, logging.INFO, "ingest_start", source_url=source_url, material_id=material_id, concept_id=concept_id)
        started = time.monotonic()

        video_id = clip_engine_meta.extract_video_id(source_url)
        if not video_id:
            raise UnsupportedSourceError("Only YouTube URLs are supported.")

        self._rate_limiter.acquire("yt")

        try:
            engine_out = clip_engine_run.clip(
                source_url,
                # concept_id is an opaque row id, NOT a topic — it must never
                # steer segmentation (it flows to _persist_ingest for row
                # association only, like ingest_feed).
                topic="",
                settings={"language": language},
            )
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

        meta = clip_engine_meta.youtube_metadata(video_id) or {}
        if not meta.get("duration_sec"):
            meta["duration_sec"] = engine_out["transcript"].get("duration")

        adapter_result = clip_engine_bridge.synth_adapter_result(video_id, source_url)
        metadata = clip_engine_bridge.to_metadata(video_id, meta, source_url)
        cues = clip_engine_bridge.to_cues(engine_out["transcript"])
        chosen = clip_engine_bridge.to_segment(best, engine_out["transcript"])
        snippet = clip_engine_bridge.window_text(engine_out["transcript"], chosen.t_start, chosen.t_end)[:700]

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
            engine_out = clip_engine_run.clip(
                source_url,
                topic=(query or ""),
                settings={"language": language},
            )
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
                chosen = clip_engine_bridge.to_segment(clip, engine_out["transcript"])
                snippet = clip_engine_bridge.window_text(
                    engine_out["transcript"], chosen.t_start, chosen.t_end
                )[:700]
                persisted = self._persist_ingest(
                    adapter_result=adapter_result,
                    metadata=metadata,
                    cues=cues,
                    chosen=chosen,
                    snippet=snippet,
                    material_id=material_id,
                    concept_id=concept_id,
                    clip_window=(chosen.t_start, chosen.t_end),
                    target_max=720,  # MAX_TOPIC_REEL_SEC = 12 * 60
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
    ) -> IngestFeedResult:
        effective_trace = set_trace_id(trace_id or new_trace_id())
        log_event(logger, logging.INFO, "ingest_feed_start", feed_url=feed_url)
        self._rate_limiter.acquire("yt")

        urls = clip_engine_meta.resolve_feed_urls(feed_url, max_items)

        items: list[IngestFeedItem] = []
        succeeded = 0
        failed = 0

        for url in urls:
            try:
                video_id = clip_engine_meta.extract_video_id(url)
                engine_out = clip_engine_run.clip(url, topic="", settings={"language": language})
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
                )[:700]

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
                )

                items.append(IngestFeedItem(
                    source_url=url,
                    status="ok",
                    reel=persisted,
                    metadata=metadata,
                ))
                succeeded += 1

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
        effective_trace = set_trace_id(trace_id or new_trace_id())

        # Coerce to YouTube-only regardless of what the caller requested.
        resolved_material_id = material_id or self._ensure_search_material(query)
        limit = min(int(max_per_platform), clip_engine_config.CLIP_SEARCH_MAX_VIDEOS)

        self._rate_limiter.acquire("yt")

        disc = clip_engine_search.discover(
            query, limit=limit, exclude_video_ids=exclude_video_ids or []
        )

        _search_warning = disc.get("warning")
        if _search_warning:
            log_event(logger, logging.WARNING, "ingest_search_warning", warning=_search_warning)

        items: list[IngestSearchItem] = []
        succeeded = 0
        failed = 0

        for v in disc["videos"]:
            try:
                engine_out = clip_engine_run.clip(
                    v["url"], topic=query, settings={"language": language}
                )
                if not engine_out["clips"]:
                    items.append(IngestSearchItem(
                        platform="yt", source_url=v["url"], status="skipped"
                    ))
                    continue

                best = clip_engine_bridge.pick_best_clip(engine_out["clips"], target_clip_duration_sec, target_clip_duration_max_sec)

                persisted, metadata = self._persist_engine_clip(
                    v=v,
                    clip=best,
                    engine_out=engine_out,
                    material_id=resolved_material_id,
                    concept_id=concept_id,
                    target_max=int(target_clip_duration_max_sec),
                )

                items.append(IngestSearchItem(
                    platform="yt",
                    source_url=v["url"],
                    status="ok",
                    reel=persisted,
                    metadata=metadata,
                ))
                succeeded += 1

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
        max_videos: int = 3,
        max_reels: int | None = None,
        on_reel_created: Callable[[ReelOutWithAttribution], None] | None = None,
        dry_run: bool = False,
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

        Cost/latency guardrail: `max_videos` bounds the paid `run.clip` calls;
        the per-video clip+filter FETCH runs concurrently, but persist +
        `on_reel_created` + the `max_reels` early-stop run SEQUENTIALLY in
        discover order so ordering and the global cap stay deterministic.
        """
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

        disc = clip_engine_search.discover(
            topic, limit=limit, exclude_video_ids=bare_exclusions
        )

        warning = disc.get("warning")
        if warning:
            log_event(logger, logging.WARNING, "ingest_topic_warning", warning=warning)

        resolved_video_ids = [v["id"] for v in disc["videos"]]

        # dry_run: discover-only viability probe — no run.clip, no DB writes.
        if dry_run:
            return [], resolved_video_ids

        videos = disc["videos"]

        # FETCH stage: clip + relevance-filter each video concurrently, with a
        # per-video wall-clock deadline; persistence stays single-threaded.
        # shutdown(wait=False) abandons a stuck video's thread rather than
        # blocking the whole generation on it.
        fetched: list[tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]] = []
        if videos:
            executor = ThreadPoolExecutor(max_workers=min(4, len(videos)))
            try:
                futures = [
                    executor.submit(self._clip_and_filter, v, topic, language)
                    for v in videos
                ]
                for v, future in zip(videos, futures):
                    try:
                        fetched.append(future.result(timeout=INGEST_TOPIC_VIDEO_TIMEOUT_SEC))
                    except Exception as exc:
                        log_event(
                            logger,
                            logging.WARNING,
                            "ingest_topic_video_failed",
                            video_id=v.get("id"),
                            error=str(exc),
                        )
            finally:
                executor.shutdown(wait=False, cancel_futures=True)

        # PERSIST stage: sequential, discover order — the `on_reel_created`
        # ordering and the (optional) `max_reels` early-stop stay deterministic.
        # Duration/kind/informativeness gates already ran inside the engine;
        # `_clip_and_filter` keeps only the best clips per video.
        reels: list[ReelOutWithAttribution] = []
        for v, kept, engine_out in fetched:
            for clip in kept:
                if max_reels is not None and len(reels) >= max_reels:
                    return reels, resolved_video_ids
                reel, _ = self._persist_engine_clip(
                    v=v,
                    clip=clip,
                    engine_out=engine_out,
                    material_id=material_id,
                    concept_id=concept_id,
                    target_max=int(target_clip_duration_max_sec),
                    generation_id=generation_id,
                )
                reels.append(reel)
                if on_reel_created is not None:
                    on_reel_created(reel)

        return reels, resolved_video_ids

    def _clip_and_filter(
        self, v: dict[str, Any], topic: str, language: str
    ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
        """Fetch ONE discovered video's clips, score each (query relevance
        blended with the engine's informativeness so one-word topics still get
        a ranking signal), and keep only the top
        ``INGEST_TOPIC_MAX_CLIPS_PER_VIDEO``. Returns `(v, scored_clips,
        engine_out)`, scored_clips sorted by score DESCENDING. Empty `clips`
        yields no clips (video is skipped)."""
        engine_out = clip_engine_run.clip(
            v["url"], topic=topic, settings={"language": language}
        )
        if not engine_out["clips"]:
            return v, [], engine_out
        for clip in engine_out["clips"]:
            relevance = clip_engine_bridge.relevance_score(
                clip, engine_out["transcript"], topic
            )
            raw_info = clip.get("informativeness")
            informativeness = (
                0.5 if raw_info is None else max(0.0, min(1.0, float(raw_info)))
            )
            clip["score"] = relevance * (0.5 + 0.5 * informativeness)
        kept = sorted(engine_out["clips"], key=lambda c: c["score"], reverse=True)
        return v, kept[:INGEST_TOPIC_MAX_CLIPS_PER_VIDEO], engine_out

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
        )[:700]

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
        )
        return reel, metadata

    def _persist_ingest(
        self,
        *,
        adapter_result: AdapterResult,
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
    ) -> ReelOutWithAttribution:
        clip_start, clip_end = clip_window
        from .persistence import build_video_id  # local import to avoid cycle surprises

        video_id = build_video_id(adapter_result.platform, adapter_result.source_id)

        # Build the client-facing video_url: YouTube honors start/end params; IG/TT don't.
        # floor(start)/ceil(end) with a >=1s guard, matching the engine's embed_url —
        # int()-truncating the end cut up to ~1s off every reel's final word.
        if adapter_result.platform == "yt":
            embed_start = int(clip_start)
            embed_end = max(embed_start + 1, math.ceil(clip_end))
            video_url = (
                f"https://www.youtube.com/embed/{adapter_result.source_id}"
                f"?start={embed_start}&end={embed_end}"
                "&modestbranding=1&rel=0&playsinline=1"
            )
        else:
            video_url = adapter_result.playback_url

        with get_conn(transactional=True) as conn:
            effective_material_id = material_id or ensure_sentinel_material(conn)
            effective_concept_id = concept_id or ensure_sentinel_concept(conn, effective_material_id)

            upsert_video(conn, platform=adapter_result.platform, source_id=adapter_result.source_id, metadata=metadata)

            # The engine's per-clip topic title (when present) describes what the
            # CLIP teaches; the video title is only a fallback. It also leads the
            # takeaways — takeaways_json is the only per-reel field that round-trips
            # to both clients, so this is what makes the title user-visible.
            takeaways = build_takeaways_for_ingest(
                concept_title=clip_title or metadata.title or "",
                transcript_snippet=snippet,
                hashtags=metadata.hashtags,
                limit=3,
            )
            if clip_title:
                takeaways = ([clip_title] + [t for t in takeaways if t != clip_title])[:3]

            ai_summary = brief_ai_summary(
                conn,
                concept_title=clip_title or metadata.title or "",
                video_title=metadata.title or "",
                video_description=metadata.description,
                transcript_snippet=snippet,
                takeaways=takeaways,
                cache_key_suffix=f"{adapter_result.platform}:{adapter_result.source_id}:{int(clip_start)}-{int(clip_end)}",
            )

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
            )

            if not inserted:
                # Unique index collision — load the existing row and reuse it.
                existing = load_existing_reel(
                    conn,
                    material_id=effective_material_id,
                    video_id=video_id,
                    t_start=clip_start,
                    t_end=clip_end,
                    generation_id=generation_id,
                )
                if existing:
                    reel_id = existing["id"]
                    # Still store metadata blob (may have changed since prior ingest).
            store_ingest_metadata_blob(conn, reel_id=reel_id, metadata=metadata)

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
            concept_title=metadata.title or "",
            video_title=metadata.title or "",
            channel_name=metadata.author_name or "",
            video_description=metadata.description,
            ai_summary=ai_summary,
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
