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
import json
import logging
import math
import os
import re
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
    load_reel_by_selection_candidate,
    resolve_material_concept,
    store_ingest_metadata_blob,
    update_reel_boundary_state,
    upsert_reel_row,
    upsert_video,
)
from ..clip_engine import (  # noqa: F401
    bridge as clip_engine_bridge,
    config as clip_engine_config,
    metadata as clip_engine_meta,
    run as clip_engine_run,
    search as clip_engine_search,
    silence as clip_engine_silence,
)
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
from ..services.knowledge_level import effective_level_target
from ...pipeline import gemini_segment as live_gemini_segment

logger: logging.Logger = get_ingest_logger(__name__)

# Shared wall-clock budget for one topic's concurrent clip+filter batch. A
# pathological set of videos must not multiply this deadline by video count.
INGEST_TOPIC_VIDEO_TIMEOUT_SEC = float(os.environ.get("INGEST_TOPIC_VIDEO_TIMEOUT_SEC", "180"))
INGEST_TOPIC_BOOTSTRAP_TIMEOUT_SEC = float(
    os.environ.get("INGEST_TOPIC_BOOTSTRAP_TIMEOUT_SEC", "45")
)


def _run_clip(
    url: str,
    *,
    topic: str,
    language: str,
    should_cancel: Callable[[], bool] | None,
    generation_context: GenerationContext | None = None,
    deadline_monotonic: float | None = None,
    candidate_rank: int | None = None,
    max_clips: int | None = None,
    retrieval_profile: str = "deep",
    knowledge_level: str | None = None,
    target_clip_duration_sec: int | None = None,
    target_clip_duration_min_sec: int | None = None,
    target_clip_duration_max_sec: int | None = None,
) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "language": language,
        "generation_context": generation_context,
        "provider_cache": generation_context.cache_store if generation_context is not None else None,
    }
    if knowledge_level:
        settings["_knowledge_level"] = str(knowledge_level).strip().lower()
    if target_clip_duration_sec is not None:
        settings["_segment_target_sec"] = int(target_clip_duration_sec)
    if target_clip_duration_min_sec is not None:
        settings["_segment_target_min_sec"] = int(target_clip_duration_min_sec)
    if target_clip_duration_max_sec is not None:
        settings["_segment_target_max_sec"] = int(target_clip_duration_max_sec)
    if candidate_rank is not None:
        settings["_segment_candidate_rank"] = int(candidate_rank)
    if max_clips is not None:
        settings["max_clips"] = max(1, min(8, int(max_clips)))
    if deadline_monotonic is not None:
        settings["deadline_monotonic"] = float(deadline_monotonic)
    if retrieval_profile == "bootstrap":
        # Bootstrap is the low-latency Flash path. Fail closed instead of
        # spending the shared job's one Pro fallback; localized Flash boundary
        # repair still runs inside the selector.
        settings["_segment_pro_fallback_gate"] = lambda **_kwargs: False
        settings["_segment_routing_mode"] = "flash_only"
        settings["_segment_thinking_level"] = "low"
    elif retrieval_profile == "deep":
        # Whole-transcript deep selection still relies on the deterministic
        # relevance, grounding, closure, and filler gates. Low thinking leaves
        # output room for more valid clips and avoids candidate JSON truncation.
        settings["_segment_thinking_level"] = "low"
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
    retrieval_profile: str = "deep",
    deadline_monotonic: float | None = None,
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
        "practice_fast": True,
        "retrieval_profile": retrieval_profile,
    }
    if deadline_monotonic is not None:
        kwargs["deadline_monotonic"] = float(deadline_monotonic)
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


def _supadata_boundary_diagnostics(
    transcript: dict[str, Any],
    clip: dict[str, Any],
) -> dict[str, Any] | None:
    """Validate one complete clip against its immutable Supadata cue range."""
    if not _is_valid_timestamped_supadata_transcript(transcript):
        return None
    cue_ids = [str(value or "").strip() for value in (clip.get("cue_ids") or [])]
    if not cue_ids or any(not value for value in cue_ids):
        return None
    segments = transcript["segments"]
    index_by_id = {
        str(segment.get("cue_id") or "").strip(): index
        for index, segment in enumerate(segments)
    }
    try:
        indices = [index_by_id[cue_id] for cue_id in cue_ids]
        start_sec = float(clip.get("start"))
        end_sec = float(clip.get("end"))
    except (KeyError, TypeError, ValueError):
        return None
    if (
        indices != list(range(indices[0], indices[-1] + 1))
        or not math.isfinite(start_sec)
        or not math.isfinite(end_sec)
        or start_sec < 0
        or end_sec <= start_sec
        or end_sec - start_sec > 180.0
    ):
        return None
    first = segments[indices[0]]
    last = segments[indices[-1]]
    first_start = float(first["start"])
    last_end = float(last["end"])
    if start_sec > first_start + 1e-3 or end_sec + 1e-3 < last_end:
        return None
    previous_end = (
        float(segments[indices[0] - 1]["end"])
        if indices[0] > 0
        else 0.0
    )
    next_start = (
        float(segments[indices[-1] + 1]["start"])
        if indices[-1] + 1 < len(segments)
        else end_sec
    )
    return {
        "method": "supadata_cue_timing",
        "acoustic_verified": False,
        "start_cue_id": cue_ids[0],
        "end_cue_id": cue_ids[-1],
        "start_padding_ms": round(max(0.0, first_start - start_sec) * 1000),
        "end_padding_ms": round(max(0.0, end_sec - last_end) * 1000),
        "preceding_gap_ms": round(max(0.0, first_start - previous_end) * 1000),
        "following_gap_ms": round(max(0.0, next_start - last_end) * 1000),
    }


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


def _topic_evidence(
    text: str,
    topic_terms: list[str],
    *,
    semantic_score: float | None = None,
) -> list[str]:
    """Return transcript-grounded lexical evidence or a strong semantic verdict."""
    from ..services.search_query_plan import semantic_query_family

    text_family = semantic_query_family(text)
    text_tokens = set(text_family.split())
    evidence: list[str] = []
    for raw_term in topic_terms:
        term = " ".join(str(raw_term or "").split())
        term_tokens = set(semantic_query_family(term).split())
        if not term_tokens:
            continue
        overlap = len(term_tokens.intersection(text_tokens))
        required = 1 if len(term_tokens) == 1 else math.ceil(0.67 * len(term_tokens))
        if overlap >= required:
            evidence.append(term)
    if evidence:
        return evidence[:8]
    if semantic_score is not None and semantic_score >= 0.72 and text.strip():
        return [f"semantic:{semantic_score:.2f}"]
    return []


def _grounded_topic_evidence_quote(text: str, quote: object) -> str:
    """Return a substantive exact quote grounded inside one selected clip.

    The selector supplies this semantic proof.  Keeping the check lexical and
    local prevents a one-word topic mention (for example, "biology" in course
    logistics) from becoming sufficient evidence on its own.
    """
    cleaned_quote = " ".join(str(quote or "").split()).strip()
    quote_words = re.findall(r"[\w+#'-]+", cleaned_quote.casefold())
    if not 5 <= len(quote_words) <= 40:
        return ""
    text_words = re.findall(r"[\w+#'-]+", str(text or "").casefold())
    width = len(quote_words)
    if not any(text_words[index : index + width] == quote_words for index in range(len(text_words) - width + 1)):
        return ""
    generic = {
        "a", "an", "and", "are", "course", "class", "for", "in", "intro",
        "introduction", "lecture", "of", "on", "the", "this", "to", "today",
        "university", "welcome", "we", "will", "you", "your",
    }
    content_words = {word for word in quote_words if len(word) >= 3 and word not in generic}
    return cleaned_quote if len(content_words) >= 2 else ""


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
        embedding_service: Any,
        youtube_service: Any = None,
        settings: Any = None,
        rate_limiter: _PlatformRateLimiter | None = None,
        serverless_mode: bool | None = None,
    ) -> None:
        # Compatibility-only test seam. Production retrieval is Supadata-only.
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
        meta = {"duration_sec": engine_out["transcript"].get("duration")}

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

        meta: dict[str, Any] = {}
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

        if not clip_engine_meta.extract_video_id(feed_url):
            raise UnsupportedSourceError(
                "Supadata-only feed ingestion requires a direct YouTube video URL."
            )
        urls = [feed_url]
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

                meta = {"duration_sec": engine_out["transcript"].get("duration")}

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
        retrieval_profile: str = "deep",
        analyzed_video_ids: set[str] | None = None,
        retrieved_video_ids: set[str] | None = None,
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
        Bootstrap analyzes up to three videos concurrently; deep retrieval starts
        with two and backfills one at a time only while the requested clip cap is
        unmet. Completed valid clips persist and stream immediately, and the
        returned list is restored to discover order.
        """
        topic = " ".join(str(topic or "").split())
        if not topic:
            raise UnsupportedSourceError("A non-blank YouTube search topic is required.")
        retrieval_profile = (
            "bootstrap" if str(retrieval_profile).strip().lower() == "bootstrap" else "deep"
        )
        raise_if_cancelled(should_cancel)
        provider_analysis_limit = int(max_videos)
        if generation_context is not None and not dry_run:
            provider_analysis_limit = min(
                provider_analysis_limit,
                generation_context.budget.remaining("segmentation"),
            )
        analysis_limit = max(
            0,
            min(provider_analysis_limit, clip_engine_config.CLIP_SEARCH_MAX_VIDEOS),
        )
        if analysis_limit <= 0:
            return [], []
        discovery_limit = (
            analysis_limit
            if retrieval_profile == "bootstrap"
            else min(
                clip_engine_config.CLIP_SEARCH_MAX_VIDEOS,
                max(analysis_limit, analysis_limit * 2),
            )
        )
        bootstrap_deadline = (
            time.monotonic() + max(0.0, INGEST_TOPIC_BOOTSTRAP_TIMEOUT_SEC)
            if retrieval_profile == "bootstrap"
            else None
        )

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
                topic, limit=discovery_limit, exclude_video_ids=bare_exclusions, level=knowledge_level,
                should_cancel=should_cancel,
                creative_commons_only=creative_commons_only,
                preferred_video_duration=preferred_video_duration,
                language=language,
                generation_context=generation_context,
                literal_topic=literal_topic or topic,
                use_query_planner=False,
                breadth=clip_engine_config.SEARCH_BREADTH,
                retrieval_profile=retrieval_profile,
                deadline_monotonic=bootstrap_deadline,
            )
        except _ClipProviderError:
            if generation_context is not None:
                generation_context.increment_counter("provider_failures")
            raise

        warning = disc.get("warning")
        if warning:
            log_event(logger, logging.WARNING, "ingest_topic_warning", warning=warning)

        authoritative_topic = " ".join(str(literal_topic or topic).split()) or topic
        corrected_topic = " ".join(str(disc.get("corrected") or authoritative_topic).split()) or authoritative_topic
        query_plan = disc.get("query_plan")
        if not isinstance(query_plan, SearchQueryPlan):
            query_plan = None
        topic_terms = [
            " ".join(str(term or "").split())
            for term in (disc.get("topic_terms") or [])
            if " ".join(str(term or "").split())
        ]
        if query_plan is not None:
            topic_terms.extend(
                term
                for term in (
                    query_plan.literal_query,
                    query_plan.canonical_query,
                    *query_plan.trusted_signature,
                )
                if term
            )
        topic_terms = list(dict.fromkeys(topic_terms))
        for discovered_video in disc["videos"]:
            discovered_video["_retrieval_profile"] = retrieval_profile
            discovered_video["_knowledge_level"] = knowledge_level
            discovered_video["_topic_terms"] = topic_terms
            discovered_video["_literal_topic"] = authoritative_topic
            discovered_video["_target_clip_duration_sec"] = int(
                target_clip_duration_sec
            )
            discovered_video["_target_clip_duration_min_sec"] = int(
                target_clip_duration_min_sec
            )
            discovered_video["_target_clip_duration_max_sec"] = int(
                target_clip_duration_max_sec
            )
            discovered_video["_search_context"] = _retrieval_search_context(
                requested_topic=authoritative_topic,
                corrected_topic=corrected_topic,
                video=discovered_video,
                query_plan=query_plan,
                creative_commons_only=creative_commons_only,
                source_duration=preferred_video_duration,
            )
            if query_plan is not None:
                discovered_video["_query_plan"] = query_plan
        resolved_video_ids = [v["id"] for v in disc["videos"][:analysis_limit]]
        if retrieved_video_ids is not None:
            retrieved_video_ids.update(
                str(video_id or "").strip().split(":", 1)[-1]
                for video_id in resolved_video_ids
                if str(video_id or "").strip()
            )
        if generation_context is not None:
            generation_context.increment_counter("discovered_videos", len(resolved_video_ids))

        # dry_run: discover-only viability probe — no run.clip, no DB writes.
        if dry_run:
            return [], resolved_video_ids

        videos = disc["videos"]

        if not videos:
            return [], resolved_video_ids

        requested_count = 3 if max_reels is None else max(0, int(max_reels))
        if requested_count == 0:
            return [], resolved_video_ids
        inventory_cap = requested_count + 2 if max_reels is None else requested_count
        minimum_valid = inventory_cap

        concurrent_video_count = min(
            3 if retrieval_profile == "bootstrap" else 2,
            len(videos),
            analysis_limit,
        )
        executor = ThreadPoolExecutor(max_workers=concurrent_video_count)
        require_acoustic_boundaries = bool(
            generation_context is not None
            and generation_context.require_acoustic_boundaries
        )
        audio_executor = (
            ThreadPoolExecutor(
                max_workers=concurrent_video_count,
                thread_name_prefix="clip-audio-prepare",
            )
            if require_acoustic_boundaries
            else None
        )
        audio_preparation_futures: dict[str, Any] = {}
        prepared_audio_results: dict[str, Any] = {}
        enrichment_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="clip-enrichment")
            if generation_context is not None and retrieval_profile == "deep"
            else None
        )
        enrichment_buffer: list[dict[str, Any]] = []
        enrichment_futures: list[Any] = []
        batch_cancelled = threading.Event()

        def fetch_should_cancel() -> bool:
            return batch_cancelled.is_set() or is_cancelled(should_cancel)

        shared_timeout_sec = (
            INGEST_TOPIC_BOOTSTRAP_TIMEOUT_SEC
            if retrieval_profile == "bootstrap"
            else INGEST_TOPIC_VIDEO_TIMEOUT_SEC
        )
        deadline = bootstrap_deadline or (
            time.monotonic() + max(0.0, INGEST_TOPIC_VIDEO_TIMEOUT_SEC)
        )
        for index, video in enumerate(videos):
            video["_deadline_monotonic"] = deadline
            video["_segment_candidate_rank"] = index

        def submit_video(index: int):
            analyzed_video_id = str(videos[index].get("id") or "").strip()
            if analyzed_video_ids is not None and analyzed_video_id:
                analyzed_video_ids.add(analyzed_video_id)
            videos[index]["_segment_max_candidates"] = min(
                8,
                max(1, inventory_cap - persisted_count + 2),
            )
            if (
                audio_executor is not None
                and analyzed_video_id
                and analyzed_video_id not in audio_preparation_futures
            ):
                audio_preparation_futures[analyzed_video_id] = audio_executor.submit(
                    clip_engine_silence.prepare_audio_source,
                    str(videos[index].get("url") or analyzed_video_id),
                    cancel_check=fetch_should_cancel,
                )
            return executor.submit(
                self._clip_and_filter,
                videos[index],
                authoritative_topic,
                language,
                fetch_should_cancel,
                generation_context,
            )

        def fetch_result(v: dict[str, Any], future: Any, timeout: float):
            completed = False
            try:
                result = future.result(timeout=timeout)
                completed = True
                if generation_context is not None:
                    record_cohort = getattr(
                        generation_context,
                        "record_pro_fallback_cohort_result",
                        None,
                    )
                    if retrieval_profile == "deep" and callable(record_cohort):
                        accepted_count = (
                            len(result[1])
                            if isinstance(result, tuple) and len(result) > 1
                            else 0
                        )
                        record_cohort(
                            candidate_rank=int(v.get("_segment_candidate_rank") or 0),
                            accepted_count=accepted_count,
                            fallback_eligible=False,
                        )
                return result
            except _ClipCancellationError:
                if batch_cancelled.is_set() and not is_cancelled(should_cancel):
                    return None
                raise
            except FutureTimeoutError:
                batch_cancelled.set()
                if generation_context is not None:
                    generation_context.increment_counter("clip_fetch_timeouts")
                log_event(
                    logger,
                    logging.WARNING,
                    "ingest_topic_video_failed",
                    video_id=v.get("id"),
                    topic=topic,
                    concept_id=concept_id,
                    generation_id=generation_id,
                    literal_query=(v.get("_search_context") or {}).get("literal_query"),
                    matched_queries=(v.get("_search_context") or {}).get("matched_queries"),
                    query_plan_ai_status=(v.get("_search_context") or {}).get("query_plan_ai_status"),
                    error=(
                        "shared clip fetch deadline exceeded "
                        f"({shared_timeout_sec:g}s)"
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
            finally:
                if not completed and generation_context is not None:
                    record_cohort = getattr(
                        generation_context,
                        "record_pro_fallback_cohort_result",
                        None,
                    )
                    if retrieval_profile == "deep" and callable(record_cohort):
                        record_cohort(
                            candidate_rank=int(v.get("_segment_candidate_rank") or 0),
                            accepted_count=0,
                            fallback_eligible=False,
                        )
            return None

        def prepared_audio_for(v: dict[str, Any]):
            video_id = str(v.get("id") or "").strip()
            if not require_acoustic_boundaries:
                return None
            if video_id in prepared_audio_results:
                return prepared_audio_results[video_id]
            future = audio_preparation_futures.get(video_id)
            if future is None:
                result = clip_engine_silence.AudioPreparationResult(
                    "unavailable",
                    diagnostics={"stage": "resolve", "reason": "audio_not_prepared"},
                )
            else:
                try:
                    result = future.result(timeout=max(0.1, deadline - time.monotonic()))
                except Exception:
                    result = clip_engine_silence.AudioPreparationResult(
                        "unavailable",
                        diagnostics={"stage": "resolve", "reason": "audio_prepare_failed"},
                    )
            prepared_audio_results[video_id] = result
            return result

        def record_boundary_unavailable(reason: str) -> None:
            if generation_context is None:
                return
            generation_context.increment_counter("boundary_unavailable")
            generation_context.record_segment_event({
                "event": "segment_completed",
                "rejection_reasons": [f"acoustic:{reason}"],
            })

        def enrich_persisted_batch(batch: list[dict[str, Any]]) -> None:
            if generation_context is None or not batch:
                return
            settings = {"_segment_budget_reserve": generation_context.reserve_gemini_call}
            updates, calls = live_gemini_segment.enrich_accepted_clips(
                batch,
                topic=authoritative_topic,
                settings=settings,
                deadline_monotonic=deadline,
                cancelled=should_cancel,
            )
            for call in calls:
                has_provider_response = bool(
                    call.get("finish_reason")
                    or call.get("prompt_tokens")
                    or call.get("prompt_token_count")
                    or call.get("candidate_tokens")
                    or call.get("candidates_token_count")
                    or call.get("thought_tokens")
                    or call.get("thoughts_token_count")
                    or call.get("total_tokens")
                    or call.get("total_token_count")
                )
                generation_context.record_gemini(
                    stage="enrichment",
                    attempt=max(1, int(call.get("retries") or 0) + 1),
                    model_used=str(call.get("model") or ""),
                    quality_degraded=False,
                    usage=call,
                    status_code=200 if has_provider_response else None,
                    error_code="" if has_provider_response else "model_call_failed",
                )
            if not updates:
                return
            try:
                with get_conn(transactional=True) as conn:
                    for reel_id, details in updates.items():
                        conn.execute(
                            """
                            UPDATE reels
                            SET ai_summary = ?, takeaways_json = ?, match_reason = ?
                            WHERE id = ?
                            """,
                            (
                                str(details.get("summary") or "")[:700],
                                dumps_json(list(details.get("takeaways") or [])[:4]),
                                str(details.get("match_reason") or "")[:700],
                                reel_id,
                            ),
                        )
            except Exception:
                logger.warning("accepted clip enrichment persistence failed", exc_info=True)

        def queue_enrichment(
            reel: ReelOutWithAttribution,
            clip: dict[str, Any],
            engine_out: dict[str, Any],
        ) -> None:
            if enrichment_executor is None:
                return
            reel_id = str(getattr(reel, "reel_id", "") or "").strip()
            if not reel_id:
                return
            clip_text = (
                clip_engine_bridge.cue_text(
                    engine_out["transcript"], clip.get("cue_ids")
                )
                or clip_engine_bridge.window_text(
                    engine_out["transcript"],
                    float(clip.get("start") or 0.0),
                    float(clip.get("end") or 0.0),
                )
            )
            enrichment_buffer.append({
                "clip_id": reel_id,
                "title": str(clip.get("title") or ""),
                "learning_objective": str(clip.get("learning_objective") or ""),
                "text": clip_text[:5000],
            })

        surfaceable_candidate_ids_by_video: dict[str, set[str]] = {}

        def persist_result(
            result: tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]],
            *,
            limit: int | None = None,
        ) -> list[ReelOutWithAttribution]:
            v, kept, engine_out = result
            persisted: list[ReelOutWithAttribution] = []
            surfaceable_candidate_ids = surfaceable_candidate_ids_by_video.setdefault(
                str(v.get("id") or ""), set()
            )
            surface_limit = None if limit is None else max(0, int(limit))
            for raw_clip in kept:
                if surface_limit is not None and len(persisted) >= surface_limit:
                    break
                raise_if_cancelled(should_cancel)
                clip = dict(raw_clip)
                search_context = dict(clip.get("search_context") or {})
                semantic_eligible = search_context.get("surface_eligible") is not False
                prerequisite_ids = {
                    str(value or "").strip()
                    for value in (clip.get("prerequisite_ids") or [])
                    if str(value or "").strip()
                }
                prerequisites_ready = prerequisite_ids.issubset(surfaceable_candidate_ids)
                surface_eligible = semantic_eligible and prerequisites_ready
                if not prerequisites_ready:
                    search_context["surface_reason"] = "prerequisite_not_surfaceable"

                if semantic_eligible and generation_context is not None:
                    caption_diagnostics = _supadata_boundary_diagnostics(
                        engine_out["transcript"], clip
                    )
                    if caption_diagnostics is None:
                        surface_eligible = False
                        search_context["boundary_status"] = "unavailable"
                        search_context["surface_reason"] = "supadata_boundary_unavailable"
                        record_boundary_unavailable("supadata_boundary_unavailable")
                    elif require_acoustic_boundaries:
                        original_start = float(clip.get("start") or 0.0)
                        original_end = float(clip.get("end") or 0.0)
                        acoustic_timeout = deadline - time.monotonic()
                        if acoustic_timeout <= 0:
                            acoustic = clip_engine_silence.SilenceVerificationResult(
                                "unavailable",
                                original_start,
                                original_end,
                                {"stage": "verify", "reason": "deadline_exceeded"},
                            )
                        else:
                            acoustic = clip_engine_silence.verify_acoustic_boundaries(
                                str(v.get("url") or v.get("id") or ""),
                                original_start,
                                original_end,
                                prepared=prepared_audio_for(v),
                                timeout_sec=min(
                                    clip_engine_silence.DEFAULT_TIMEOUT_SEC,
                                    acoustic_timeout,
                                ),
                                cancel_check=should_cancel,
                            )
                        acoustic_diagnostics = dict(acoustic.diagnostics or {})
                        first_start = (
                            original_start
                            + float(caption_diagnostics["start_padding_ms"]) / 1000.0
                        )
                        last_end = (
                            original_end
                            - float(caption_diagnostics["end_padding_ms"]) / 1000.0
                        )
                        adjusted_duration = acoustic.end_sec - acoustic.start_sec
                        acoustic_range_is_safe = bool(
                            acoustic.verified
                            and acoustic.start_sec + 1e-3
                            >= max(
                                0.0,
                                original_start
                                - clip_engine_silence.EDGE_WINDOW_SEC / 2.0,
                            )
                            and acoustic.end_sec
                            <= original_end
                            + clip_engine_silence.EDGE_WINDOW_SEC / 2.0
                            + 1e-3
                            and acoustic.start_sec <= first_start + 0.05
                            and acoustic.end_sec + 0.05 >= last_end
                            and 1.0 <= adjusted_duration <= 180.0
                        )
                        boundary_diagnostics = {
                            "method": "energy_silence",
                            "acoustic_verified": acoustic_range_is_safe,
                            "caption": caption_diagnostics,
                            "acoustic": acoustic_diagnostics,
                        }
                        search_context["boundary_diagnostics"] = boundary_diagnostics
                        if not acoustic_range_is_safe:
                            surface_eligible = False
                            search_context["boundary_status"] = "unavailable"
                            search_context["surface_reason"] = str(
                                acoustic_diagnostics.get("reason")
                                or (
                                    "acoustic_adjusted_duration_invalid"
                                    if acoustic.verified
                                    and not 1.0 <= adjusted_duration <= 180.0
                                    else "acoustic_boundary_outside_selected_range"
                                )
                            )
                            failure_stage = str(
                                acoustic_diagnostics.get("stage") or "verify"
                            ).strip()
                            record_boundary_unavailable(
                                f"{failure_stage}:{search_context['surface_reason']}"
                            )
                        else:
                            clip["start"] = round(float(acoustic.start_sec), 3)
                            clip["end"] = round(float(acoustic.end_sec), 3)
                            search_context["boundary_status"] = "verified"
                    else:
                        search_context["boundary_status"] = "caption_aligned"
                        search_context["boundary_diagnostics"] = caption_diagnostics
                else:
                    search_context.setdefault("boundary_status", "caption_aligned")

                if surface_eligible and bool(search_context.get("deferred_level")):
                    surface_eligible = False
                    search_context["surface_reason"] = "level_mismatch"

                search_context["surface_eligible"] = bool(surface_eligible)
                clip["search_context"] = search_context
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
                if generation_context is not None:
                    generation_context.increment_counter("stored_clips")
                if not surface_eligible:
                    if generation_context is not None:
                        generation_context.increment_counter("deferred_clips")
                    continue
                persisted.append(reel)
                candidate_id = str(clip.get("selection_candidate_id") or "").strip()
                if candidate_id:
                    surfaceable_candidate_ids.add(candidate_id)
                if generation_context is not None:
                    generation_context.increment_counter("persisted_clips")
                if on_reel_created is not None:
                    on_reel_created(reel)
                queue_enrichment(reel, clip, engine_out)
            return persisted

        reels_by_video: dict[int, list[ReelOutWithAttribution]] = {}
        completed_results: dict[
            int,
            tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]],
        ] = {}
        initial_resolved: set[int] = set()
        aggregate_fallback_checked = False
        provider_errors: list[_ClipProviderError] = []
        initial_count = min(2, len(videos), analysis_limit)
        if generation_context is not None and retrieval_profile == "deep":
            configure_fallback = getattr(
                generation_context, "configure_pro_fallback_gate", None
            )
            if callable(configure_fallback):
                configure_fallback(initial_count)
        valid_clip_count = 0
        persisted_count = 0
        bootstrap_attempted_indices: set[int] = set()

        def persist_bootstrap_sources() -> None:
            """Persist at most one best clip per completed source before reuse.

            Bootstrap waits for the two initial analyses, ranks their strongest
            clips together, and reserves the second slot for the second source.
            Later sources backfill empty/failed initial results.
            """
            nonlocal valid_clip_count, persisted_count
            if retrieval_profile != "bootstrap" or len(initial_resolved) < initial_count:
                return
            # Rank the two initial sources together. Concurrent source three is
            # considered only after those sources have had their first chance.
            for source_indices in (
                range(initial_count),
                range(initial_count, min(len(videos), analysis_limit)),
            ):
                source_candidates: list[
                    tuple[float, int, tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]]
                ] = []
                for source_index in source_indices:
                    source_result = completed_results.get(source_index)
                    if (
                        source_result is None
                        or source_index in bootstrap_attempted_indices
                        or not source_result[1]
                    ):
                        continue
                    source_candidates.append(
                        (
                            float(source_result[1][0].get("score") or 0.0),
                            source_index,
                            source_result,
                        )
                    )
                for _score, source_index, source_result in sorted(
                    source_candidates,
                    key=lambda item: (item[0], -item[1]),
                    reverse=True,
                ):
                    if persisted_count >= inventory_cap:
                        break
                    bootstrap_attempted_indices.add(source_index)
                    one_clip_result = (
                        source_result[0],
                        source_result[1][:1],
                        source_result[2],
                    )
                    persisted = persist_result(one_clip_result, limit=1)
                    reels_by_video.setdefault(source_index, []).extend(persisted)
                    persisted_count += len(persisted)
                if persisted_count >= inventory_cap:
                    break
            # Backfill decisions must follow clips that can actually stream,
            # rather than every low-value proposal returned by one source.
            valid_clip_count = persisted_count

        def clips_temporally_overlap(a: dict[str, Any], b: dict[str, Any]) -> bool:
            start = max(float(a.get("start") or 0.0), float(b.get("start") or 0.0))
            end = min(float(a.get("end") or 0.0), float(b.get("end") or 0.0))
            overlap = end - start
            if overlap <= 0:
                return False
            shorter = min(
                float(a.get("end") or 0.0) - float(a.get("start") or 0.0),
                float(b.get("end") or 0.0) - float(b.get("start") or 0.0),
            )
            return shorter > 0 and overlap / shorter >= 0.8

        def maybe_run_aggregate_fallback() -> None:
            nonlocal aggregate_fallback_checked, valid_clip_count, persisted_count
            if retrieval_profile != "deep":
                return
            if aggregate_fallback_checked or len(initial_resolved) < initial_count:
                return
            aggregate_fallback_checked = True
            initial_surfaceable = sum(
                len(reels_by_video.get(index, []))
                for index in range(initial_count)
            )
            if generation_context is None or initial_surfaceable >= 3:
                return
            viable_candidates = [
                index
                for index in range(initial_count)
                if index in completed_results
                and _is_valid_timestamped_supadata_transcript(
                    completed_results[index][2].get("transcript") or {}
                )
            ]
            if not viable_candidates:
                # Leave the Pro budget untouched so a later backfill selector can
                # still claim it after obtaining a usable transcript.
                return
            candidate_index = min(
                viable_candidates,
                key=lambda index: (len(completed_results[index][1]), index),
            )
            claim_fallback = getattr(
                generation_context,
                "claim_aggregate_pro_fallback",
                None,
            )
            if not callable(claim_fallback) or not claim_fallback(
                validated_count=initial_surfaceable
            ):
                return
            v, existing_clips, engine_out = completed_results[candidate_index]
            try:
                fallback_out = clip_engine_run.pro_boundary_fallback(
                    engine_out["transcript"],
                    topic=authoritative_topic,
                    video_id=str(v.get("id") or ""),
                    settings={
                        "generation_context": generation_context,
                        "deadline_monotonic": deadline,
                        "_segment_target_sec": int(target_clip_duration_sec),
                        "_segment_target_min_sec": int(
                            target_clip_duration_min_sec
                        ),
                        "_segment_target_max_sec": int(
                            target_clip_duration_max_sec
                        ),
                        "max_clips": min(
                            8,
                            max(1, inventory_cap - initial_surfaceable + 2),
                        ),
                    },
                    should_cancel=should_cancel,
                )
                fallback_result = self._clip_and_filter(
                    v,
                    authoritative_topic,
                    language,
                    should_cancel,
                    generation_context,
                    engine_out_override=fallback_out,
                    record_transcript_usage=False,
                )
            except (_ClipError, _ClipProviderError) as exc:
                log_event(
                    logger,
                    logging.WARNING,
                    "aggregate_pro_fallback_failed",
                    video_id=v.get("id"),
                    error=str(exc),
                )
                return
            accepted_candidate_ids = {
                str(clip.get("selection_candidate_id") or "").strip()
                for clip in existing_clips
                if str(clip.get("selection_candidate_id") or "").strip()
            }
            novelty_candidates = [
                clip
                for clip in fallback_result[1]
                if not any(
                    clips_temporally_overlap(clip, existing)
                    for existing in existing_clips
                )
            ]
            novel_clips: list[dict[str, Any]] = []
            pending_novel = list(novelty_candidates)
            satisfied_ids = set(accepted_candidate_ids)
            while pending_novel:
                deferred: list[dict[str, Any]] = []
                progress = False
                for clip in pending_novel:
                    prerequisite_ids = {
                        str(value).strip()
                        for value in (clip.get("prerequisite_ids") or [])
                        if str(value).strip()
                    }
                    if not prerequisite_ids.issubset(satisfied_ids):
                        deferred.append(clip)
                        continue
                    novel_clips.append(clip)
                    candidate_id = str(
                        clip.get("selection_candidate_id") or ""
                    ).strip()
                    if candidate_id:
                        satisfied_ids.add(candidate_id)
                    progress = True
                if not progress:
                    break
                pending_novel = deferred
            if not novel_clips:
                return
            novel_result = (fallback_result[0], novel_clips, fallback_result[2])
            valid_clip_count += len(novel_clips)
            if persisted_count < inventory_cap:
                persisted = persist_result(
                    novel_result,
                    limit=inventory_cap - persisted_count,
                )
                reels_by_video.setdefault(candidate_index, []).extend(persisted)
                persisted_count += len(persisted)

        pending = {
            submit_video(index): (index, videos[index])
            for index in range(concurrent_video_count)
        }
        all_futures = list(pending)
        next_index = concurrent_video_count
        try:
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
                    initial_resolved.update(
                        index for index, _video in pending.values() if index < initial_count
                    )
                    persist_bootstrap_sources()
                    for _, v in pending.values():
                        log_event(
                            logger,
                            logging.WARNING,
                            "ingest_topic_video_failed",
                            video_id=v.get("id"),
                            topic=topic,
                            concept_id=concept_id,
                            generation_id=generation_id,
                            literal_query=(v.get("_search_context") or {}).get("literal_query"),
                            matched_queries=(v.get("_search_context") or {}).get("matched_queries"),
                            query_plan_ai_status=(v.get("_search_context") or {}).get("query_plan_ai_status"),
                            error=(
                                "shared clip fetch deadline exceeded "
                                f"({shared_timeout_sec:g}s)"
                            ),
                        )
                    if generation_context is not None:
                        generation_context.increment_counter("clip_fetch_timeouts", len(pending))
                    break

                for future in sorted(done, key=lambda item: pending[item][0]):
                    index, v = pending.pop(future)
                    if index < initial_count:
                        initial_resolved.add(index)
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
                    if result is None:
                        continue
                    completed_results[index] = result
                    if retrieval_profile == "bootstrap":
                        persist_bootstrap_sources()
                    else:
                        valid_clip_count += len(result[1])
                        if persisted_count < inventory_cap:
                            persisted = persist_result(
                                result,
                                limit=inventory_cap - persisted_count,
                            )
                            reels_by_video[index] = persisted
                            persisted_count += len(persisted)

                if retrieval_profile == "bootstrap":
                    persist_bootstrap_sources()
                maybe_run_aggregate_fallback()

                if (
                    not pending
                    and persisted_count < minimum_valid
                    and persisted_count < inventory_cap
                    and next_index < min(len(videos), analysis_limit)
                ):
                    future = submit_video(next_index)
                    pending[future] = (next_index, videos[next_index])
                    all_futures.append(future)
                    next_index += 1
        finally:
            batch_cancelled.set()
            for future in all_futures:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            if audio_executor is not None:
                audio_executor.shutdown(wait=False, cancel_futures=True)

        if retrieval_profile == "bootstrap" and persisted_count < inventory_cap:
            # Only after every available source has had its first chance may a
            # later clip from an already used source fill the remaining slot.
            # Keep walking that source when an otherwise-good earlier clip was
            # acoustically unavailable; the inventory cap still limits emits.
            extras: list[
                tuple[
                    float,
                    int,
                    int,
                    tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]],
                ]
            ] = []
            for source_index, source_result in completed_results.items():
                if len(source_result[1]) <= 1:
                    continue
                for clip_index in range(1, len(source_result[1])):
                    extras.append(
                        (
                            float(source_result[1][clip_index].get("score") or 0.0),
                            source_index,
                            clip_index,
                            source_result,
                        )
                    )
            for _score, source_index, clip_index, source_result in sorted(
                extras, key=lambda item: (item[0], -item[1], -item[2]), reverse=True
            ):
                if persisted_count >= inventory_cap:
                    break
                extra_result = (
                    source_result[0],
                    source_result[1][clip_index : clip_index + 1],
                    source_result[2],
                )
                persisted = persist_result(extra_result, limit=1)
                reels_by_video.setdefault(source_index, []).extend(persisted)
                persisted_count += len(persisted)

        if enrichment_executor is not None:
            while enrichment_buffer:
                batch = enrichment_buffer[:3]
                del enrichment_buffer[:3]
                enrichment_futures.append(
                    enrichment_executor.submit(
                        enrich_persisted_batch,
                        batch,
                    )
                )
            for future in enrichment_futures:
                remaining = max(0.0, deadline - time.monotonic())
                if remaining <= 0:
                    future.cancel()
                    continue
                try:
                    future.result(timeout=remaining)
                except Exception:
                    logger.warning("accepted clip enrichment failed", exc_info=True)
            enrichment_executor.shutdown(wait=False, cancel_futures=True)

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
        engine_out_override: dict[str, Any] | None = None,
        record_transcript_usage: bool = True,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
        """Fetch ONE discovered video's clips, score each (query relevance
        blended with the engine's informativeness so one-word topics still get
        a ranking signal). Returns `(v, scored_clips,
        engine_out)`, scored_clips sorted by score DESCENDING. Empty `clips`
        yields no clips (video is skipped)."""
        engine_out = engine_out_override
        if engine_out is None:
            engine_out = _run_clip(
                v["url"], topic=topic, language=language,
                should_cancel=should_cancel,
                generation_context=generation_context,
                deadline_monotonic=v.get("_deadline_monotonic"),
                candidate_rank=v.get("_segment_candidate_rank"),
                max_clips=v.get("_segment_max_candidates"),
                retrieval_profile=str(v.get("_retrieval_profile") or "deep"),
                knowledge_level=str(v.get("_knowledge_level") or ""),
                target_clip_duration_sec=v.get("_target_clip_duration_sec"),
                target_clip_duration_min_sec=v.get(
                    "_target_clip_duration_min_sec"
                ),
                target_clip_duration_max_sec=v.get(
                    "_target_clip_duration_max_sec"
                ),
            )
        transcript = engine_out["transcript"]
        query_plan = (
            v.get("_query_plan")
            if isinstance(v.get("_query_plan"), SearchQueryPlan)
            else None
        )
        trusted_transcript = _is_valid_timestamped_supadata_transcript(transcript)
        if generation_context is not None and trusted_transcript and record_transcript_usage:
            generation_context.increment_counter("usable_transcripts")
        elif (
            generation_context is not None
            and record_transcript_usage
            and (query_plan is not None or v.get("_topic_terms"))
        ):
            generation_context.increment_counter("transcript_failures")

        raw_clips = list(engine_out["clips"])
        if not raw_clips:
            if generation_context is not None:
                generation_context.increment_counter("gemini_empty_results")
            return v, [], engine_out
        topic_terms = [str(term) for term in (v.get("_topic_terms") or []) if str(term).strip()]
        level_target = effective_level_target(v.get("_knowledge_level"), 0.0)

        def unit(value: object, default: float) -> float:
            try:
                parsed = float(value)
            except (TypeError, ValueError, OverflowError):
                return default
            return max(0.0, min(1.0, parsed)) if math.isfinite(parsed) else default

        transcript_texts = [
            (
                clip_engine_bridge.cue_text(transcript, clip.get("cue_ids"))
                or clip_engine_bridge.window_text(
                    transcript,
                    float(clip.get("start") or 0.0),
                    float(clip.get("end") or 0.0),
                )
            )
            for clip in raw_clips
        ]
        semantic_scores: list[float | None] = [None] * len(raw_clips)
        if (
            topic_terms
            and trusted_transcript
            and self._embedding_service is not None
            and getattr(self._embedding_service, "semantic_available", False) is True
        ):
            try:
                vectors = self._embedding_service.embed_semantic([*topic_terms, *transcript_texts])
                anchor_count = len(topic_terms)
                if vectors is not None and len(vectors) == anchor_count + len(raw_clips):
                    for index, vector in enumerate(vectors[anchor_count:]):
                        semantic_scores[index] = max(
                            float(vector.dot(anchor)) for anchor in vectors[:anchor_count]
                        )
            except Exception:
                logger.warning("semantic topic gate unavailable; using lexical evidence")

        kept: list[dict[str, Any]] = []
        for index, clip in enumerate(raw_clips):
            lexical_relevance = clip_engine_bridge.relevance_score(
                clip, engine_out["transcript"], topic
            )
            informativeness = unit(clip.get("informativeness"), 0.5)
            difficulty = unit(clip.get("difficulty"), 0.5)
            raw_topic_relevance = clip.get("topic_relevance")
            topic_relevance = (
                lexical_relevance
                if raw_topic_relevance is None
                else unit(raw_topic_relevance, lexical_relevance)
            )
            selector_topic_contract = any(
                key in clip
                for key in (
                    "directly_teaches_topic",
                    "substantive",
                    "factually_grounded",
                    "topic_evidence_quote",
                )
            )
            grounded_evidence_quote = ""
            if selector_topic_contract:
                grounded_evidence_quote = _grounded_topic_evidence_quote(
                    transcript_texts[index], clip.get("topic_evidence_quote")
                )
                if (
                    clip.get("directly_teaches_topic") is not True
                    or clip.get("substantive") is not True
                    or clip.get("factually_grounded") is not True
                    or not grounded_evidence_quote
                ):
                    if generation_context is not None:
                        generation_context.increment_counter("topic_rejections")
                    continue
                evidence = [grounded_evidence_quote]
            else:
                # Compatibility for legacy cached selector output. New selector
                # responses always carry the grounded semantic contract above.
                evidence = _topic_evidence(
                    transcript_texts[index],
                    topic_terms,
                    semantic_score=semantic_scores[index],
                ) if topic_terms and trusted_transcript else []
            if topic_terms and not evidence:
                if generation_context is not None:
                    generation_context.increment_counter("topic_rejections")
                continue
            clip["topic_evidence_terms"] = evidence

            has_selector_metadata = any(
                key in clip
                for key in (
                    "educational_importance",
                    "boundary_confidence",
                    "is_standalone",
                    "chain_id",
                    "prerequisite_ids",
                )
            )
            search_context = {
                **dict(v.get("_search_context") or {}),
                "topic_evidence_terms": evidence[:8],
                "directly_teaches_topic": bool(
                    clip.get("directly_teaches_topic", bool(evidence))
                ),
                "substantive": bool(clip.get("substantive", bool(evidence))),
                "factually_grounded": bool(clip.get("factually_grounded")),
                "topic_evidence_quote": grounded_evidence_quote,
                "surface_eligible": True,
                "deferred_level": abs(difficulty - level_target) > 0.35,
            }
            if has_selector_metadata:
                importance = unit(clip.get("educational_importance"), 0.5)
                boundary_confidence = unit(clip.get("boundary_confidence"), 0.5)
                uncertainty = str(clip.get("uncertainty") or "low").strip().lower()
                uncertainty_penalty = 0.05 if uncertainty == "medium" else 0.0
                content_score = (
                    0.45 * topic_relevance
                    + 0.35 * importance
                    + 0.20 * informativeness
                )
                clip["score"] = max(0.0, content_score - uncertainty_penalty)
                prerequisite_ids = clip.get("prerequisite_ids")
                source_namespace = str(v.get("id") or "unknown-video").strip()

                def namespaced_selection_id(value: object) -> str:
                    raw_value = str(value or "").strip()
                    if not raw_value:
                        return ""
                    prefix = f"{source_namespace}::"
                    return raw_value if raw_value.startswith(prefix) else f"{prefix}{raw_value}"

                selection_candidate_id = namespaced_selection_id(
                    clip.get("selection_candidate_id")
                )
                namespaced_prerequisites = (
                    [
                        namespaced_selection_id(item)
                        for item in prerequisite_ids
                        if str(item).strip()
                    ]
                    if isinstance(prerequisite_ids, list)
                    else []
                )
                raw_chain_id = str(clip.get("chain_id") or "").strip()
                chain_id = namespaced_selection_id(raw_chain_id) if raw_chain_id else ""
                clip["selection_candidate_id"] = selection_candidate_id
                clip["prerequisite_ids"] = namespaced_prerequisites
                clip["chain_id"] = chain_id
                search_context.update(
                    selection_contract_version="confidence_v1",
                    content_score=content_score,
                    topic_relevance=topic_relevance,
                    educational_importance=importance,
                    boundary_confidence=boundary_confidence,
                    is_standalone=bool(clip.get("is_standalone", True)),
                    chain_id=chain_id,
                    chain_position=clip.get("chain_position"),
                    selection_candidate_id=selection_candidate_id,
                    prerequisite_ids=namespaced_prerequisites,
                    uncertainty=uncertainty,
                    uncertainty_reasons=[
                        str(reason)
                        for reason in (clip.get("uncertainty_reasons") or [])
                        if str(reason).strip()
                    ],
                )
            else:
                level_fit = 1.0 - abs(difficulty - level_target)
                uncertainty_penalty = (
                    0.05
                    if str(clip.get("uncertainty") or "low").strip().lower()
                    == "medium"
                    else 0.0
                )
                clip["score"] = max(
                    0.0,
                    (
                        0.55 * topic_relevance
                        + 0.25 * informativeness
                        + 0.15 * level_fit
                        + 0.05 * lexical_relevance
                    )
                    - uncertainty_penalty,
                )
            clip["search_context"] = search_context
            kept.append(clip)

        nodes: dict[str, dict[str, Any]] = {}
        source_order: dict[str, int] = {}
        aliases: dict[str, str] = {}
        for clip_index, clip in enumerate(kept):
            node_id = f"clip-node-{clip_index}"
            nodes[node_id] = clip
            source_order[node_id] = clip_index
            candidate_id = str(clip.get("selection_candidate_id") or "").strip()
            if candidate_id:
                aliases[candidate_id] = node_id
        dependencies: dict[str, set[str]] = {node_id: set() for node_id in nodes}
        for node_id, clip in nodes.items():
            for prerequisite in clip.get("prerequisite_ids") or []:
                prerequisite_id = str(prerequisite or "").strip()
                if prerequisite_id:
                    dependencies[node_id].add(
                        aliases.get(prerequisite_id, f"missing:{prerequisite_id}")
                    )
        remaining_nodes = set(nodes)
        satisfied_nodes: set[str] = set()
        ordered_kept: list[dict[str, Any]] = []
        while remaining_nodes:
            eligible_nodes = [
                node_id
                for node_id in remaining_nodes
                if dependencies[node_id].issubset(satisfied_nodes)
            ]
            if not eligible_nodes:
                break
            if not ordered_kept:
                safe_first = [
                    node_id
                    for node_id in eligible_nodes
                    if (
                        not nodes[node_id].get("search_context", {}).get(
                            "selection_contract_version"
                        )
                        or (
                            bool(nodes[node_id].get("is_standalone"))
                            and unit(nodes[node_id].get("boundary_confidence"), 0.0) >= 0.80
                        )
                    )
                ]
                if not safe_first:
                    break
                eligible_nodes = safe_first
            chosen_id = max(
                eligible_nodes,
                key=lambda node_id: (
                    float(nodes[node_id].get("score") or 0.0),
                    -source_order[node_id],
                ),
            )
            ordered_kept.append(nodes[chosen_id])
            remaining_nodes.remove(chosen_id)
            satisfied_nodes.add(chosen_id)
        return v, ordered_kept, engine_out

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
        snippet = (
            clip_engine_bridge.cue_text(
                engine_out["transcript"], clip.get("cue_ids")
            )
            or clip_engine_bridge.window_text(
                engine_out["transcript"], chosen.t_start, chosen.t_end
            )
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
        selected_cue_ids = [
            str(cue_id)
            for cue_id in (details.get("cue_ids") or [])
            if str(cue_id or "").strip()
        ]
        selection_context = (
            dict(details.get("search_context") or {})
            if isinstance(details.get("search_context"), dict)
            else {}
        )
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

        learning_objective = " ".join(
            str(details.get("learning_objective") or "").split()
        ).strip()[:700]
        ai_summary = " ".join(str(details.get("summary") or "").split()).strip()[:700]
        if not ai_summary:
            ai_summary = learning_objective
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
            match_reason = " ".join(
                str(details.get("reason") or learning_objective).split()
            ).strip()[:700]
        if not match_reason:
            matched_idea = clip_title or (takeaways[0] if takeaways else "") or metadata.title or "this topic"
            match_reason = f"This clip directly explains {matched_idea[:180]} using the source transcript."
        try:
            informativeness = float(details.get("informativeness"))
        except (TypeError, ValueError):
            informativeness = 0.6
        informativeness = max(0.0, min(1.0, informativeness))
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
            existing_candidate = load_reel_by_selection_candidate(
                conn,
                material_id=effective_material_id,
                concept_id=effective_concept_id,
                video_id=video_id,
                generation_id=generation_id,
                selection_candidate_id=str(
                    selection_context.get("selection_candidate_id") or ""
                ),
            )
            if existing_candidate:
                reel_id = str(existing_candidate["id"])
                existing_context = {}
                try:
                    existing_context = json.loads(
                        str(existing_candidate.get("search_context_json") or "{}")
                    )
                except (TypeError, json.JSONDecodeError):
                    pass
                existing_surfaceable = (
                    isinstance(existing_context, dict)
                    and existing_context.get("surface_eligible") is True
                )
                incoming_surfaceable = selection_context.get("surface_eligible") is True
                if incoming_surfaceable or not existing_surfaceable:
                    update_reel_boundary_state(
                        conn,
                        reel_id=reel_id,
                        video_url=video_url,
                        t_start=clip_start,
                        t_end=clip_end,
                        transcript_snippet=snippet,
                        selected_cue_ids=selected_cue_ids,
                        search_context=selection_context,
                    )
                inserted = True
            else:
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
                    selected_cue_ids=selected_cue_ids,
                    search_context=selection_context,
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
        selected_cue_id_set = set(selected_cue_ids)
        for cue in cues:
            if not cue.text:
                continue
            if selected_cue_id_set and cue.cue_id not in selected_cue_id_set:
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
        try:
            chain_position = float(selection_context.get("chain_position") or 0.0)
        except (TypeError, ValueError):
            chain_position = 0.0

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
            difficulty=(
                0.5 if clip_difficulty is None else float(clip_difficulty)
            ),
            model_used=str(details.get("model_used") or ""),
            quality_degraded=bool(details.get("quality_degraded", False)),
            selected_cue_ids=[
                str(cue_id)
                for cue_id in (details.get("cue_ids") or [])
                if str(cue_id or "").strip()
            ],
            selection_contract_version=(
                str(selection_context.get("selection_contract_version") or "").strip()
                or None
            ),
            boundary_confidence=(
                float(selection_context["boundary_confidence"])
                if selection_context.get("boundary_confidence") is not None
                else None
            ),
            is_standalone=bool(selection_context.get("is_standalone", True)),
            chain_id=str(selection_context.get("chain_id") or ""),
            chain_position=chain_position,
            selection_candidate_id=str(
                selection_context.get("selection_candidate_id") or ""
            ),
            prerequisite_ids=[
                str(value)
                for value in (selection_context.get("prerequisite_ids") or [])
                if str(value).strip()
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

    settings = get_settings()
    db_module.init_db()

    embedding_service = EmbeddingService()
    pipeline = IngestionPipeline(
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
