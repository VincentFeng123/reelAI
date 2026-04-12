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
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from ..db import DatabaseIntegrityError, dumps_json, get_conn, now_iso, upsert
from . import TERMS_NOTICE
from .adapters.base import AdapterResult, BaseAdapter, PlatformCode
from .adapters.yt_dlp_adapter import YtDlpAdapter
from .download import TempWorkspace
from .errors import (
    DownloadError,
    IngestError,
    RateLimitedError,
    SegmentationError,
    ServerlessUnavailable,
    TranscriptionError,
    UnsupportedSourceError,
)
from .ffmpeg_tools import check_ffmpeg_available, probe_duration, silencedetect
from .logging_config import get_ingest_logger, log_event, new_trace_id, set_trace_id
from .metadata import (
    brief_ai_summary,
    build_takeaways_for_ingest,
    format_attribution,
    map_info_dict_to_metadata,
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
from .segment import normalize_clip_window, pick_segments, snippet_for_window
from .transcribe import transcribe

logger: logging.Logger = get_ingest_logger(__name__)


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
        openai_client: Any,
        settings: Any = None,
        adapters: list[BaseAdapter] | None = None,
        rate_limiter: _PlatformRateLimiter | None = None,
        serverless_mode: bool | None = None,
        feed_concurrency: int = 2,
    ) -> None:
        self._youtube_service = youtube_service
        self._embedding_service = embedding_service
        self._openai_client = openai_client
        self._settings = settings
        self._adapters: list[BaseAdapter] = adapters or [YtDlpAdapter()]
        self._rate_limiter = rate_limiter or _PlatformRateLimiter()
        if serverless_mode is None:
            serverless_mode = bool(
                os.environ.get("VERCEL")
                or os.environ.get("AWS_LAMBDA_FUNCTION_NAME")
                or os.environ.get("K_SERVICE")
            )
        self._serverless_mode = serverless_mode
        self._allow_openai_in_serverless = os.environ.get("ALLOW_OPENAI_IN_SERVERLESS") == "1"
        self._feed_executor = ThreadPoolExecutor(max_workers=max(1, int(feed_concurrency)))

    # --------------------------------------------------------------------- #
    # Adapter routing
    # --------------------------------------------------------------------- #

    def _pick_adapter(self, url: str) -> BaseAdapter:
        for adapter in self._adapters:
            try:
                if adapter.supports(url):
                    return adapter
            except Exception:
                logger.exception("adapter.supports crashed for %s (%s)", adapter.name, url)
        raise UnsupportedSourceError(f"No adapter supports URL: {url}")

    # --------------------------------------------------------------------- #
    # Preflight
    # --------------------------------------------------------------------- #

    def _preflight(self) -> None:
        if self._serverless_mode and not self._allow_openai_in_serverless:
            raise ServerlessUnavailable(
                "Reel ingestion is disabled in serverless mode. Set ALLOW_OPENAI_IN_SERVERLESS=1 to override."
            )
        if not check_ffmpeg_available():
            raise DownloadError(
                "ffmpeg/ffprobe are not installed on this host. "
                "Install via railpack.toml or the deploy Dockerfile."
            )

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

        self._preflight()

        adapter = self._pick_adapter(source_url)
        platform: PlatformCode = adapter.platform_for(source_url)
        self._rate_limiter.acquire(platform)

        with TempWorkspace() as workspace:
            adapter_result = adapter.resolve(source_url, workspace)
            metadata = map_info_dict_to_metadata(
                adapter_result.info_dict,
                platform=platform,
                source_url=source_url,
                source_id=adapter_result.source_id,
                playback_url=adapter_result.playback_url,
            )

            duration_sec = metadata.duration_sec
            if not duration_sec or duration_sec <= 0:
                try:
                    duration_sec = probe_duration(adapter_result.video_path)
                    metadata = metadata.model_copy(update={"duration_sec": duration_sec})
                except DownloadError as exc:
                    raise DownloadError(
                        "Could not determine video duration",
                        detail=exc.detail or exc.message,
                    ) from exc

            # Transcribe — opens its own short-lived connection for cache read/write.
            try:
                cues = self._transcribe_with_conn(
                    platform=platform,
                    source_id=adapter_result.source_id,
                    info_dict=adapter_result.info_dict,
                    video_path=adapter_result.video_path,
                    workspace=workspace,
                    language=language,
                )
            except (TranscriptionError, ServerlessUnavailable):
                raise
            except Exception as exc:
                raise TranscriptionError("unexpected error during transcription", detail=str(exc)) from exc

            if not cues:
                raise TranscriptionError("no transcript cues were produced")

            # Silence detection
            silence_ranges: list[tuple[float, float]] = []
            audio_path = workspace / "audio_16k.wav"
            try:
                if audio_path.exists():
                    silence_ranges = silencedetect(audio_path)
            except SegmentationError:
                logger.warning("silencedetect failed; continuing without silence boundaries")
                silence_ranges = []

            # Segmentation
            segments = pick_segments(
                cues,
                float(duration_sec or 0.0),
                target_sec=int(target_clip_duration_sec),
                min_sec=int(target_clip_duration_min_sec),
                max_sec=int(target_clip_duration_max_sec),
                silence_ranges=silence_ranges,
                top_k=1,
            )
            if not segments:
                raise SegmentationError("no valid clip window could be produced")
            chosen = segments[0]

            # Final window normalization (defense in depth)
            window = normalize_clip_window(
                chosen.t_start,
                chosen.t_end,
                int(duration_sec or 0),
                min_len=int(target_clip_duration_min_sec),
                max_len=int(target_clip_duration_max_sec),
            )
            if window is None:
                raise SegmentationError("normalize_clip_window rejected the chosen segment")
            clip_start, clip_end = float(window[0]), float(window[1])
            chosen = IngestSegment(t_start=clip_start, t_end=clip_end, text=chosen.text, score=chosen.score)

            snippet = snippet_for_window(cues, clip_start, clip_end, max_chars=700)

            # Persist
            persisted = self._persist_ingest(
                adapter_result=adapter_result,
                metadata=metadata,
                cues=cues,
                chosen=chosen,
                snippet=snippet,
                material_id=material_id,
                concept_id=concept_id,
                clip_window=(clip_start, clip_end),
                target_max=int(target_clip_duration_max_sec),
            )

        elapsed_ms = int((time.monotonic() - started) * 1000)
        log_event(
            logger,
            logging.INFO,
            "ingest_completed",
            source_url=source_url,
            reel_id=persisted.reel_id,
            platform=platform,
            source_id=adapter_result.source_id,
            author_handle=metadata.author_handle,
            author_name=metadata.author_name,
            duration_sec=metadata.duration_sec,
            t_start=chosen.t_start,
            t_end=chosen.t_end,
            elapsed_ms=elapsed_ms,
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

        Reuses the existing download → transcribe machinery, then hands the
        cues to `services.topic_cut.cut_video_into_topic_reels` which:
          1. Classifies the video as Short or long-form (Shorts are returned
             with an empty `reels` list — caller leaves them alone).
          2. For long-form: asks the LLM to identify topic boundaries by cue
             index, snaps the boundaries to natural cue gaps, and drops any
             segment outside [30s, 12min].
          3. Falls back to the lexical-novelty heuristic if no OpenAI client
             is available or the LLM call fails.

        Each TopicReel is then persisted via the same sentinel-material path
        used by `_persist_ingest`, producing a list of `ReelOutWithAttribution`
        rows that decode cleanly into the iOS `Reel` struct.

        Concurrency note: this method is single-threaded — topic segmentation
        is one LLM call, persistence is one transaction. The pipeline-wide
        rate limiter is acquired once per video, just like `ingest_url`.
        """
        # Local import to break the import cycle: services.topic_cut imports
        # nothing from the ingestion package, but we don't want this module to
        # take on a top-level dependency on services either.
        from ..services.topic_cut import (
            VideoClassification,
            cues_from_ingest_cues,
            cut_video_into_topic_reels,
        )

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

        self._preflight()

        adapter = self._pick_adapter(source_url)
        platform: PlatformCode = adapter.platform_for(source_url)
        self._rate_limiter.acquire(platform)

        with TempWorkspace() as workspace:
            adapter_result = adapter.resolve(source_url, workspace)
            metadata = map_info_dict_to_metadata(
                adapter_result.info_dict,
                platform=platform,
                source_url=source_url,
                source_id=adapter_result.source_id,
                playback_url=adapter_result.playback_url,
            )

            duration_sec = metadata.duration_sec
            if not duration_sec or duration_sec <= 0:
                try:
                    duration_sec = probe_duration(adapter_result.video_path)
                    metadata = metadata.model_copy(update={"duration_sec": duration_sec})
                except DownloadError as exc:
                    raise DownloadError(
                        "Could not determine video duration",
                        detail=exc.detail or exc.message,
                    ) from exc

            try:
                cues = self._transcribe_with_conn(
                    platform=platform,
                    source_id=adapter_result.source_id,
                    info_dict=adapter_result.info_dict,
                    video_path=adapter_result.video_path,
                    workspace=workspace,
                    language=language,
                )
            except (TranscriptionError, ServerlessUnavailable):
                raise
            except Exception as exc:
                raise TranscriptionError("unexpected error during transcription", detail=str(exc)) from exc

            if not cues:
                raise TranscriptionError("no transcript cues were produced")

            # Stash the info_dict so we can pass it (and its `chapters` key)
            # to topic_cut after the workspace exits. Most ingest paths only
            # need the metadata that map_info_dict_to_metadata extracted, but
            # topic_cut wants the raw `chapters` list which we've never
            # propagated until now.
            info_dict_snapshot: dict[str, Any] = dict(adapter_result.info_dict or {})

        # Outside the workspace ctx-manager: the temp dir + downloaded video are
        # gone, but cues + metadata are pure-Python and survive.
        topic_cues = cues_from_ingest_cues(cues)
        classification, topic_reels = cut_video_into_topic_reels(
            source_url,
            query=query,
            duration_sec=float(duration_sec or 0.0),
            openai_client=self._openai_client,
            use_llm=use_llm,
            transcript=topic_cues,
            info_dict=info_dict_snapshot,
        )

        persisted_reels: list[ReelOutWithAttribution] = []
        if not classification.is_short and topic_reels:
            persisted_reels = self._persist_topic_reels(
                topic_reels=topic_reels,
                cues=cues,
                adapter_result=adapter_result,
                metadata=metadata,
                material_id=material_id,
                concept_id=concept_id,
            )

        elapsed_ms = int((time.monotonic() - started) * 1000)
        log_event(
            logger,
            logging.INFO,
            "ingest_topic_cut_completed",
            source_url=source_url,
            video_id=classification.video_id,
            platform=platform,
            is_short=classification.is_short,
            reel_count=len(persisted_reels),
            elapsed_ms=elapsed_ms,
        )

        return IngestTopicCutResult(
            source_url=source_url,
            video_id=classification.video_id,
            is_short=classification.is_short,
            classification_reason=classification.reason,
            duration_sec=float(classification.duration_sec or duration_sec or 0.0),
            reel_count=len(persisted_reels),
            reels=persisted_reels,
            metadata=metadata,
            terms_notice=TERMS_NOTICE,
            trace_id=effective_trace,
        )

    def _persist_topic_reels(
        self,
        *,
        topic_reels: list[Any],  # list[TopicReel] — typed as Any to avoid the import
        cues: list[IngestTranscriptCue],
        adapter_result: AdapterResult,
        metadata: IngestMetadata,
        material_id: str | None,
        concept_id: str | None,
    ) -> list[ReelOutWithAttribution]:
        """
        Insert each TopicReel as a `reels` row and return the client-facing
        ReelOutWithAttribution list.

        Mirrors `_persist_ingest` per-row but in a single shared transaction so
        a 12-reel video doesn't open 12 connections. Each reel is uniquely
        keyed by `(material_id, video_id, t_start, t_end)`; if a duplicate
        sneaks through (e.g. a re-ingest of the same URL), we reuse the
        existing row's reel_id.
        """
        from .persistence import build_video_id  # local import per the existing pattern

        video_id = build_video_id(adapter_result.platform, adapter_result.source_id)
        attribution = format_attribution(metadata)
        out: list[ReelOutWithAttribution] = []

        with get_conn(transactional=True) as conn:
            effective_material_id = material_id or ensure_sentinel_material(conn)
            effective_concept_id = concept_id or ensure_sentinel_concept(conn, effective_material_id)
            upsert_video(
                conn,
                platform=adapter_result.platform,
                source_id=adapter_result.source_id,
                metadata=metadata,
            )

            for tr in topic_reels:
                clip_start = float(tr.t_start)
                clip_end = float(tr.t_end)

                if adapter_result.platform == "yt":
                    video_url = (
                        f"https://www.youtube.com/embed/{adapter_result.source_id}"
                        f"?start={int(clip_start)}&end={int(clip_end)}"
                        "&modestbranding=1&rel=0&playsinline=1"
                    )
                else:
                    video_url = adapter_result.playback_url

                snippet = snippet_for_window(cues, clip_start, clip_end, max_chars=700)
                takeaways = build_takeaways_for_ingest(
                    concept_title=tr.label or metadata.title or "",
                    transcript_snippet=snippet,
                    hashtags=metadata.hashtags,
                    limit=3,
                )

                # Per-segment AI summary, cached on disk via `brief_ai_summary`'s
                # llm_cache key. Skipping the LLM here when no client is configured
                # is fine — the topic_cut label is already a usable headline.
                ai_summary = brief_ai_summary(
                    conn,
                    openai_client=self._openai_client,
                    concept_title=tr.label or metadata.title or "",
                    video_title=metadata.title or "",
                    video_description=metadata.description,
                    transcript_snippet=snippet,
                    takeaways=takeaways,
                    cache_key_suffix=(
                        f"topic_cut:{adapter_result.platform}:"
                        f"{adapter_result.source_id}:{int(clip_start)}-{int(clip_end)}"
                    ),
                )

                reel_id = f"topic-{uuid.uuid4().hex[:16]}"
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
                    base_score=1.0,
                )
                if not inserted:
                    existing = load_existing_reel(
                        conn,
                        material_id=effective_material_id,
                        video_id=video_id,
                        t_start=clip_start,
                        t_end=clip_end,
                    )
                    if existing:
                        reel_id = existing["id"]

                store_ingest_metadata_blob(conn, reel_id=reel_id, metadata=metadata)

                clip_captions = [
                    {"start": cue.start, "end": cue.end, "text": cue.text}
                    for cue in cues
                    if cue.text and clip_start <= cue.start <= clip_end
                ]
                clip_duration = max(0.0, clip_end - clip_start)

                out.append(
                    ReelOutWithAttribution(
                        reel_id=reel_id,
                        material_id=effective_material_id,
                        concept_id=effective_concept_id,
                        concept_title=tr.label or metadata.title or "",
                        video_title=metadata.title or "",
                        video_description=metadata.description,
                        ai_summary=ai_summary,
                        video_url=video_url,
                        t_start=clip_start,
                        t_end=clip_end,
                        transcript_snippet=snippet,
                        takeaways=takeaways,
                        captions=clip_captions,
                        score=1.0,
                        relevance_score=None,
                        discovery_score=None,
                        clipability_score=1.0,
                        query_strategy="topic_cut",
                        retrieval_stage="topic_cut",
                        source_surface=f"ingest:{adapter_result.platform}:topic_cut",
                        matched_terms=[],
                        relevance_reason=tr.summary or "",
                        concept_position=None,
                        total_concepts=None,
                        video_duration_sec=int(metadata.duration_sec) if metadata.duration_sec else None,
                        clip_duration_sec=float(clip_duration),
                        source_attribution=attribution,
                    )
                )

        return out

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
        log_event(logger, logging.INFO, "ingest_feed_start", feed_url=feed_url, max_items=max_items)
        self._preflight()

        adapter = self._pick_adapter(feed_url)
        # We intentionally do NOT check platform rate limit for the feed discovery call —
        # flat_playlist is a single lightweight request. Per-URL limits apply when we
        # resolve each item.
        reel_urls = adapter.resolve_feed(feed_url, max_items=max_items)

        items: list[IngestFeedItem] = []

        def _one(reel_url: str) -> IngestFeedItem:
            try:
                result = self.ingest_url(
                    source_url=reel_url,
                    material_id=material_id,
                    concept_id=concept_id,
                    target_clip_duration_sec=target_clip_duration_sec,
                    target_clip_duration_min_sec=target_clip_duration_min_sec,
                    target_clip_duration_max_sec=target_clip_duration_max_sec,
                    language=language,
                    trace_id=effective_trace,  # share trace id across siblings
                )
                return IngestFeedItem(
                    source_url=reel_url,
                    status="ok",
                    reel=result.reel,
                    metadata=result.metadata,
                )
            except IngestError as exc:
                logger.warning("feed item failed url=%s error=%s", reel_url, exc)
                return IngestFeedItem(source_url=reel_url, status="error", error=exc.message)
            except Exception as exc:
                logger.exception("feed item crashed url=%s", reel_url)
                return IngestFeedItem(source_url=reel_url, status="error", error=str(exc))

        # Use the executor, but collect in the original order for stable responses.
        futures = [self._feed_executor.submit(_one, u) for u in reel_urls]
        for fut in futures:
            items.append(fut.result())

        succeeded = sum(1 for i in items if i.status == "ok")
        failed = sum(1 for i in items if i.status == "error")

        log_event(
            logger,
            logging.INFO,
            "ingest_feed_completed",
            feed_url=feed_url,
            total_resolved=len(reel_urls),
            succeeded=succeeded,
            failed=failed,
        )
        return IngestFeedResult(
            feed_url=feed_url,
            total_resolved=len(reel_urls),
            succeeded=succeeded,
            failed=failed,
            items=items,
            terms_notice=TERMS_NOTICE,
            trace_id=effective_trace,
        )

    # --------------------------------------------------------------------- #
    # Topic-cut variant for search — multiple query-filtered reels per URL
    # --------------------------------------------------------------------- #

    def _ingest_url_with_topic_cut(
        self,
        *,
        source_url: str,
        query: str,
        material_id: str | None = None,
        concept_id: str | None = None,
        language: str = "en",
        trace_id: str | None = None,
    ) -> list[ReelOutWithAttribution]:
        """
        Download + transcribe + topic-cut a single URL, returning only the
        topic reels that match *query*.  Used by `ingest_search` to produce
        multiple precise clips per video instead of a single heuristic window.

        Falls back to the standard `ingest_url` (single clip) when topic-cut
        produces no results (e.g. Shorts, missing transcript, all topics
        filtered out).
        """
        from ..services.topic_cut import (
            cues_from_ingest_cues,
            cut_video_into_topic_reels,
        )

        effective_trace = set_trace_id(trace_id or new_trace_id())
        self._preflight()

        adapter = self._pick_adapter(source_url)
        platform: PlatformCode = adapter.platform_for(source_url)
        self._rate_limiter.acquire(platform)

        with TempWorkspace() as workspace:
            adapter_result = adapter.resolve(source_url, workspace)
            metadata = map_info_dict_to_metadata(
                adapter_result.info_dict,
                platform=platform,
                source_url=source_url,
                source_id=adapter_result.source_id,
                playback_url=adapter_result.playback_url,
            )

            duration_sec = metadata.duration_sec
            if not duration_sec or duration_sec <= 0:
                try:
                    duration_sec = probe_duration(adapter_result.video_path)
                    metadata = metadata.model_copy(update={"duration_sec": duration_sec})
                except DownloadError as exc:
                    raise DownloadError(
                        "Could not determine video duration",
                        detail=exc.detail or exc.message,
                    ) from exc

            try:
                cues = self._transcribe_with_conn(
                    platform=platform,
                    source_id=adapter_result.source_id,
                    info_dict=adapter_result.info_dict,
                    video_path=adapter_result.video_path,
                    workspace=workspace,
                    language=language,
                )
            except (TranscriptionError, ServerlessUnavailable):
                raise
            except Exception as exc:
                raise TranscriptionError(
                    "unexpected error during transcription", detail=str(exc)
                ) from exc

            if not cues:
                raise TranscriptionError("no transcript cues were produced")

            info_dict_snapshot: dict[str, Any] = dict(adapter_result.info_dict or {})

        # Outside workspace — temp dir is gone, cues + metadata survive.
        topic_cues = cues_from_ingest_cues(cues)
        classification, topic_reels = cut_video_into_topic_reels(
            source_url,
            query=query,
            duration_sec=float(duration_sec or 0.0),
            openai_client=self._openai_client,
            use_llm=True,
            transcript=topic_cues,
            info_dict=info_dict_snapshot,
        )

        if not classification.is_short and topic_reels:
            persisted = self._persist_topic_reels(
                topic_reels=topic_reels,
                cues=cues,
                adapter_result=adapter_result,
                metadata=metadata,
                material_id=material_id,
                concept_id=concept_id,
            )
            if persisted:
                return persisted

        # Fallback: topic-cut produced nothing usable (Short, no transcript,
        # all topics filtered out). Return empty — caller handles gracefully.
        return []

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
        # Default to YouTube only. Instagram and TikTok robots.txt explicitly
        # disallow the ReelAIBot user agent, so resolving them from here would
        # always bounce at the robots.txt check in yt_dlp_adapter. Callers can
        # still opt in by passing a platforms list explicitly.
        effective_platforms: list[PlatformLiteral] = list(platforms) if platforms else ["yt"]
        exclude = {str(v).strip() for v in (exclude_video_ids or []) if v}

        log_event(
            logger,
            logging.INFO,
            "ingest_search_start",
            query=query,
            platforms=effective_platforms,
            max_per_platform=max_per_platform,
            exclude_count=len(exclude),
        )

        started = time.monotonic()
        self._preflight()

        adapter = self._pick_search_adapter()

        per_platform_resolved: dict[str, int] = {p: 0 for p in effective_platforms}
        per_platform_succeeded: dict[str, int] = {p: 0 for p in effective_platforms}
        per_platform_failed: dict[str, int] = {p: 0 for p in effective_platforms}
        per_platform_errors: dict[str, str] = {}

        resolved_urls: list[tuple[PlatformLiteral, str]] = []
        seen_urls: set[str] = set()
        seen_source_ids: set[str] = set()

        # ---- Phase 1: resolve each platform's URL list (best-effort per platform) ----
        for platform in effective_platforms:
            try:
                search_url = adapter.build_search_url(query, platform, max_per_platform)
            except IngestError as exc:
                per_platform_failed[platform] = per_platform_failed.get(platform, 0) + 1
                per_platform_errors[platform] = exc.message
                log_event(
                    logger,
                    logging.WARNING,
                    "search_url_build_failed",
                    platform=platform,
                    error=exc.message,
                )
                continue

            try:
                urls = adapter.resolve_feed(search_url, max_items=max_per_platform)
            except IngestError as exc:
                per_platform_failed[platform] = per_platform_failed.get(platform, 0) + 1
                per_platform_errors[platform] = exc.message
                log_event(
                    logger,
                    logging.WARNING,
                    "search_resolve_failed",
                    platform=platform,
                    error=exc.message,
                )
                continue
            except Exception as exc:
                per_platform_failed[platform] = per_platform_failed.get(platform, 0) + 1
                per_platform_errors[platform] = f"unexpected: {exc}"
                logger.exception(
                    "search resolve_feed crashed for platform=%s query=%s", platform, query
                )
                continue

            # Filter: skip duplicates, skip URLs whose extracted source_id is already seen.
            accepted_this_platform = 0
            for url in urls:
                if url in seen_urls:
                    continue
                source_id = None
                if hasattr(adapter, "extract_source_id_from_url"):
                    try:
                        source_id = adapter.extract_source_id_from_url(url, platform)
                    except Exception:
                        source_id = None
                if source_id and source_id in exclude:
                    continue
                if source_id and source_id in seen_source_ids:
                    continue
                seen_urls.add(url)
                if source_id:
                    seen_source_ids.add(source_id)
                resolved_urls.append((platform, url))
                accepted_this_platform += 1
            per_platform_resolved[platform] = accepted_this_platform

        total_resolved = len(resolved_urls)
        if total_resolved == 0:
            log_event(
                logger,
                logging.WARNING,
                "ingest_search_no_urls",
                query=query,
                per_platform_errors=per_platform_errors,
            )

        # ---- Phase 2: scope a material for the query and run ingestion ----
        if material_id:
            effective_material_id = material_id
            effective_concept_id = concept_id
        else:
            effective_material_id = self._ensure_search_material(query)
            # The search-scoped concept row is created alongside the material in
            # `_ensure_search_material` and its id is deterministic, so we derive it
            # here rather than doing another SELECT. Reels inserted below will JOIN
            # cleanly to this concept from the legacy `/api/feed` endpoint.
            effective_concept_id = concept_id or f"{effective_material_id}:concept"

        def _ingest_one(platform: PlatformLiteral, reel_url: str) -> list[IngestSearchItem]:
            # Try the topic-cut path first (YouTube only) — produces multiple
            # query-filtered reels per video.  Falls back to the legacy single-
            # clip path when topic-cut returns nothing.
            if platform == "yt":
                try:
                    reels = self._ingest_url_with_topic_cut(
                        source_url=reel_url,
                        query=query,
                        material_id=effective_material_id,
                        concept_id=effective_concept_id,
                        language=language,
                        trace_id=effective_trace,
                    )
                    if reels:
                        return [
                            IngestSearchItem(
                                platform=platform,
                                source_url=reel_url,
                                status="ok",
                                reel=r,
                                metadata=None,
                            )
                            for r in reels
                        ]
                    # topic-cut returned nothing (Short, all topics filtered) —
                    # fall through to legacy single-clip path below.
                except Exception:
                    logger.debug(
                        "topic-cut failed for %s, falling back to ingest_url",
                        reel_url,
                        exc_info=True,
                    )

            # Legacy single-clip fallback (also the primary path for IG/TT).
            try:
                result = self.ingest_url(
                    source_url=reel_url,
                    material_id=effective_material_id,
                    concept_id=effective_concept_id,
                    target_clip_duration_sec=target_clip_duration_sec,
                    target_clip_duration_min_sec=target_clip_duration_min_sec,
                    target_clip_duration_max_sec=target_clip_duration_max_sec,
                    language=language,
                    trace_id=effective_trace,
                )
                return [IngestSearchItem(
                    platform=platform,
                    source_url=reel_url,
                    status="ok",
                    reel=result.reel,
                    metadata=result.metadata,
                )]
            except RateLimitedError as exc:
                logger.warning(
                    "search item rate-limited platform=%s url=%s retry_after=%s",
                    platform,
                    reel_url,
                    exc.retry_after_sec,
                )
                return [IngestSearchItem(
                    platform=platform,
                    source_url=reel_url,
                    status="rate_limited",
                    error=exc.message,
                )]
            except IngestError as exc:
                logger.warning(
                    "search item failed platform=%s url=%s error=%s",
                    platform,
                    reel_url,
                    exc.message,
                )
                return [IngestSearchItem(
                    platform=platform,
                    source_url=reel_url,
                    status="error",
                    error=exc.message,
                )]
            except Exception as exc:
                logger.exception("search item crashed platform=%s url=%s", platform, reel_url)
                return [IngestSearchItem(
                    platform=platform,
                    source_url=reel_url,
                    status="error",
                    error=str(exc),
                )]

        items: list[IngestSearchItem] = []
        if resolved_urls:
            future_to_key = {
                self._feed_executor.submit(_ingest_one, platform, url): (platform, url)
                for platform, url in resolved_urls
            }
            for future in as_completed(future_to_key):
                platform, url = future_to_key[future]
                try:
                    batch = future.result()
                except Exception as exc:
                    logger.exception("search future raised unexpectedly platform=%s", platform)
                    batch = [IngestSearchItem(
                        platform=platform,
                        source_url=url,
                        status="error",
                        error=str(exc),
                    )]
                for item in batch:
                    items.append(item)
                    if item.status == "ok":
                        per_platform_succeeded[item.platform] = per_platform_succeeded.get(item.platform, 0) + 1
                    else:
                        per_platform_failed[item.platform] = per_platform_failed.get(item.platform, 0) + 1

        # Order items deterministically: resolved order, not completion order.
        url_to_order = {url: idx for idx, (_p, url) in enumerate(resolved_urls)}
        items.sort(key=lambda it: url_to_order.get(it.source_url, 1_000_000))

        succeeded = sum(1 for i in items if i.status == "ok")
        failed = sum(1 for i in items if i.status in ("error", "rate_limited"))

        elapsed_ms = int((time.monotonic() - started) * 1000)
        log_event(
            logger,
            logging.INFO,
            "ingest_search_completed",
            query=query,
            material_id=effective_material_id,
            total_resolved=total_resolved,
            succeeded=succeeded,
            failed=failed,
            per_platform_resolved=per_platform_resolved,
            per_platform_succeeded=per_platform_succeeded,
            per_platform_failed=per_platform_failed,
            elapsed_ms=elapsed_ms,
        )

        return IngestSearchResult(
            query=query,
            material_id=effective_material_id,
            platforms=effective_platforms,
            per_platform_resolved=per_platform_resolved,
            per_platform_succeeded=per_platform_succeeded,
            per_platform_failed=per_platform_failed,
            per_platform_errors=per_platform_errors,
            total_resolved=total_resolved,
            succeeded=succeeded,
            failed=failed,
            items=items,
            terms_notice=TERMS_NOTICE,
            trace_id=effective_trace,
        )

    def _pick_search_adapter(self) -> Any:
        """The search adapter is the first adapter that exposes build_search_url."""
        for adapter in self._adapters:
            if hasattr(adapter, "build_search_url"):
                return adapter
        raise UnsupportedSourceError("No search-capable adapter is installed")

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

    def _transcribe_with_conn(
        self,
        *,
        platform: PlatformLiteral,
        source_id: str,
        info_dict: dict[str, Any],
        video_path: Path,
        workspace: Path,
        language: str,
    ) -> list[IngestTranscriptCue]:
        with get_conn() as conn:
            return transcribe(
                conn,
                platform=platform,
                source_id=source_id,
                info_dict=info_dict,
                video_path=video_path,
                workspace=workspace,
                youtube_service=self._youtube_service,
                openai_client=self._openai_client,
                language=language,
                serverless_mode=self._serverless_mode,
                allow_openai_in_serverless=self._allow_openai_in_serverless,
            )

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
    ) -> ReelOutWithAttribution:
        clip_start, clip_end = clip_window
        from .persistence import build_video_id  # local import to avoid cycle surprises

        video_id = build_video_id(adapter_result.platform, adapter_result.source_id)

        # Build the client-facing video_url: YouTube honors start/end params; IG/TT don't.
        if adapter_result.platform == "yt":
            video_url = (
                f"https://www.youtube.com/embed/{adapter_result.source_id}"
                f"?start={int(clip_start)}&end={int(clip_end)}"
                "&modestbranding=1&rel=0&playsinline=1"
            )
        else:
            video_url = adapter_result.playback_url

        with get_conn(transactional=True) as conn:
            effective_material_id = material_id or ensure_sentinel_material(conn)
            effective_concept_id = concept_id or ensure_sentinel_concept(conn, effective_material_id)

            upsert_video(conn, platform=adapter_result.platform, source_id=adapter_result.source_id, metadata=metadata)

            takeaways = build_takeaways_for_ingest(
                concept_title=metadata.title or "",
                transcript_snippet=snippet,
                hashtags=metadata.hashtags,
                limit=3,
            )

            ai_summary = brief_ai_summary(
                conn,
                openai_client=self._openai_client,
                concept_title=metadata.title or "",
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
            )

            if not inserted:
                # Unique index collision — load the existing row and reuse it.
                existing = load_existing_reel(
                    conn,
                    material_id=effective_material_id,
                    video_id=video_id,
                    t_start=clip_start,
                    t_end=clip_end,
                )
                if existing:
                    reel_id = existing["id"]
                    # Still store metadata blob (may have changed since prior ingest).
            store_ingest_metadata_blob(conn, reel_id=reel_id, metadata=metadata)

        captions = [
            {"start": cue.start, "end": cue.end, "text": cue.text}
            for cue in cues
            if cue.text
        ]

        attribution = format_attribution(metadata)

        clip_duration = max(0.0, clip_end - clip_start)
        return ReelOutWithAttribution(
            reel_id=reel_id,
            material_id=effective_material_id,
            concept_id=effective_concept_id,
            concept_title=metadata.title or "",
            video_title=metadata.title or "",
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
    from ..services.openai_client import build_openai_client
    from ..services.youtube import YouTubeService

    settings = get_settings()
    db_module.init_db()

    openai_client = build_openai_client(
        api_key=settings.openai_api_key,
        timeout=float(getattr(settings, "openai_timeout_sec", 60.0)),
        enabled=bool(getattr(settings, "openai_enabled", False) and settings.openai_api_key),
    )
    embedding_service = EmbeddingService(
        client=openai_client,
        model=getattr(settings, "openai_embedding_model", "text-embedding-3-small"),
        enabled=bool(getattr(settings, "openai_enabled", False) and settings.openai_api_key),
    )
    youtube_service = YouTubeService(settings=settings)

    pipeline = IngestionPipeline(
        youtube_service=youtube_service,
        embedding_service=embedding_service,
        openai_client=openai_client,
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
