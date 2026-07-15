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
import unicodedata
import uuid
from concurrent.futures import (
    FIRST_COMPLETED,
    ThreadPoolExecutor,
    TimeoutError as FutureTimeoutError,
    as_completed,
    wait,
)
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
from ..services.knowledge_level import (
    difficulty_matches_knowledge_level,
    effective_level_target,
)

logger: logging.Logger = get_ingest_logger(__name__)

# Shared wall-clock budget for one topic's concurrent clip+filter batch. A
# pathological set of videos must not multiply this deadline by video count.
INGEST_TOPIC_VIDEO_TIMEOUT_SEC = float(os.environ.get("INGEST_TOPIC_VIDEO_TIMEOUT_SEC", "180"))
INGEST_TOPIC_BOOTSTRAP_TIMEOUT_SEC = float(
    os.environ.get("INGEST_TOPIC_BOOTSTRAP_TIMEOUT_SEC", "45")
)
INGEST_TOPIC_STRAGGLER_GRACE_SEC = float(
    os.environ.get("INGEST_TOPIC_STRAGGLER_GRACE_SEC", "8")
)
PARTIAL_CUE_MATERIALITY_SEC = 0.05
SPEECH_OWNERSHIP_EPSILON_SEC = 0.001
MAX_PLAUSIBLE_CAPTION_WORDS_PER_SEC = 8.0
CAPTION_PROJECTION_START_PREROLL_SEC = 0.15
PROJECTED_END_COVERAGE_SEC = 0.002
_QUOTE_WORD_RE = re.compile(
    r"[\w+#]+(?:['\u2018\u2019\u02bc-][\w+#]+)*",
    re.UNICODE,
)
_QUOTE_APOSTROPHES = str.maketrans(
    {"\u2018": "'", "\u2019": "'", "\u02bc": "'"}
)
_CAPTION_INCOMPLETE_END_RE = re.compile(
    r"\b(?:(?:and|but|or)(?:\s+(?:now|so|then))?|as|because|by|for|from|"
    r"if|in|into|of|on|since|than|that|to|until|when|where|which|while|"
    r"with)\s*[.!?]?[\"')\]]*$",
    re.IGNORECASE,
)
_CAPTION_FRESH_UNIT_ONSET_RE = re.compile(
    r"^\s*(?:(?:all\s+right|alright|okay|ok|so)\s*[,;:]?\s+)?(?:"
    r"(?:now|next)\b|the\s+(?:following|next)\s+(?:example|exercise|lesson|"
    r"problem|section|step|teaching\s+unit|topic)\b|"
    r"let(?:['’]?s|\s+us)\s+(?:begin|consider|look|move|solve|start|try|work)\b)",
    re.IGNORECASE,
)
_NO_SPACE_SCRIPT_NAME_MARKERS = (
    "BOPOMOFO",
    "CJK",
    "HANGUL",
    "HIRAGANA",
    "KATAKANA",
    "KHMER",
    "LAO",
    "MYANMAR",
    "THAI",
)


def _quote_token(value: str) -> str:
    return (
        unicodedata.normalize("NFKC", value)
        .translate(_QUOTE_APOSTROPHES)
        .casefold()
    )


def _caption_speech_units(text: str) -> int:
    """Count plausible spoken units without treating no-space scripts as one word."""
    units = 0
    for token in _QUOTE_WORD_RE.findall(str(text or "")):
        ordinary_run = False
        for character in token:
            if not character.isalnum():
                continue
            unicode_name = unicodedata.name(character, "")
            if any(
                marker in unicode_name
                for marker in _NO_SPACE_SCRIPT_NAME_MARKERS
            ):
                units += 1
            else:
                ordinary_run = True
        if ordinary_run:
            units += 1
    return units


def _quote_character_spans(text: str, quote: str) -> list[tuple[int, int]]:
    text_matches = list(_QUOTE_WORD_RE.finditer(str(text or "")))
    quote_tokens = [
        _quote_token(match.group(0))
        for match in _QUOTE_WORD_RE.finditer(str(quote or ""))
    ]
    if not quote_tokens or len(quote_tokens) > len(text_matches):
        return []
    text_tokens = [_quote_token(match.group(0)) for match in text_matches]
    width = len(quote_tokens)
    return [
        (
            text_matches[index].start(),
            text_matches[index + width - 1].end(),
        )
        for index in range(len(text_tokens) - width + 1)
        if text_tokens[index:index + width] == quote_tokens
    ]


def _literal_source_quote(
    text: str,
    quote: str,
    span: tuple[int, int],
) -> str:
    source = str(text or "")
    quote_matches = list(_QUOTE_WORD_RE.finditer(str(quote or "")))
    start, end = span
    if quote_matches:
        prefix = str(quote or "")[:quote_matches[0].start()]
        suffix = str(quote or "")[quote_matches[-1].end():]
        if prefix and start >= len(prefix) and source[start - len(prefix):start] == prefix:
            start -= len(prefix)
        if suffix and source[end:end + len(suffix)] == suffix:
            end += len(suffix)
    return source[start:end]


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
    if candidate_rank is not None:
        settings["_segment_candidate_rank"] = int(candidate_rank)
    if max_clips is not None:
        settings["max_clips"] = max(1, int(max_clips))
    if deadline_monotonic is not None:
        settings["deadline_monotonic"] = float(deadline_monotonic)
    settings["_segment_pro_fallback_gate"] = lambda **_kwargs: False
    settings["_segment_routing_mode"] = "flash_only"
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
    return clip_engine_run.is_valid_timestamped_supadata_transcript(transcript)


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


def _transcript_boundary_seed(
    transcript: dict[str, Any],
    raw_clip: dict[str, Any],
) -> tuple[dict[str, Any] | None, tuple[float, float]]:
    """Recover complete transcript cues without making edge metadata a quality gate."""

    strict = _supadata_boundary_diagnostics(transcript, raw_clip)
    if strict is not None:
        return strict, _complete_caption_speech_bounds(raw_clip, strict)
    segments = [
        segment
        for segment in (transcript.get("segments") or [])
        if isinstance(segment, dict)
    ]
    index_by_id = {
        str(segment.get("cue_id") or f"cue-{index}").strip(): index
        for index, segment in enumerate(segments)
    }
    requested_cue_ids = [
        str(value or "").strip() for value in (raw_clip.get("cue_ids") or [])
    ]
    indices = (
        [index_by_id[cue_id] for cue_id in requested_cue_ids]
        if requested_cue_ids
        and all(cue_id in index_by_id for cue_id in requested_cue_ids)
        else []
    )
    recovered_from_timestamps = False
    if not indices or indices != list(range(indices[0], indices[-1] + 1)):
        try:
            selected_start = float(raw_clip.get("start"))
            selected_end = float(raw_clip.get("end"))
        except (TypeError, ValueError, OverflowError):
            return None, (0.0, 0.0)
        if (
            not math.isfinite(selected_start)
            or not math.isfinite(selected_end)
            or selected_start < 0.0
            or selected_end <= selected_start
        ):
            return None, (0.0, 0.0)
        overlapping: list[int] = []
        for index, segment in enumerate(segments):
            try:
                cue_start = float(segment.get("start"))
                cue_end = float(segment.get("end"))
            except (TypeError, ValueError, OverflowError):
                continue
            if (
                math.isfinite(cue_start)
                and math.isfinite(cue_end)
                and cue_end > cue_start
                and cue_end > selected_start + 1e-3
                and cue_start < selected_end - 1e-3
            ):
                overlapping.append(index)
        if not overlapping:
            return None, (0.0, 0.0)
        indices = list(range(overlapping[0], overlapping[-1] + 1))
        recovered_from_timestamps = True
    try:
        start = float(segments[indices[0]]["start"])
        end = float(segments[indices[-1]]["end"])
        source_end = max(
            float(transcript.get("duration") or 0.0),
            *(float(segment.get("end") or 0.0) for segment in segments),
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return None, (0.0, 0.0)
    if (
        not math.isfinite(start)
        or not math.isfinite(end)
        or not math.isfinite(source_end)
        or start < 0.0
        or end <= start
        or end > source_end + 1e-3
    ):
        return None, (0.0, 0.0)
    cue_ids = [
        str(segments[index].get("cue_id") or f"cue-{index}").strip()
        for index in indices
    ]
    if any(not cue_id for cue_id in cue_ids):
        return None, (0.0, 0.0)
    if recovered_from_timestamps:
        # The selector's finite range is still transcript-grounded even if an
        # upstream cue identifier drifted. Canonicalizing it here keeps a good
        # educational candidate and gives persistence an immutable cue snapshot.
        raw_clip["cue_ids"] = cue_ids
        raw_clip["start"] = start
        raw_clip["end"] = end
    return {
        "method": "contiguous_transcript_cues",
        "acoustic_verified": False,
        "start_cue_id": cue_ids[0],
        "end_cue_id": cue_ids[-1],
        "start_padding_ms": 0,
        "end_padding_ms": 0,
        "preceding_gap_ms": 0,
        "following_gap_ms": 0,
        "fallback_from_strict_caption_diagnostics": True,
        "recovered_from_timestamp_range": recovered_from_timestamps,
    }, (start, end)


def _required_speech_bounds(
    raw_clip: dict[str, Any],
    caption_diagnostics: dict[str, Any],
) -> tuple[float, float]:
    """Recover immutable first/last speech from a selector's padded cue range."""
    padded_start = float(raw_clip.get("start") or 0.0)
    padded_end = float(raw_clip.get("end") or padded_start)
    cue_start, cue_end = (
        padded_start
        + float(caption_diagnostics["start_padding_ms"]) / 1000.0,
        padded_end
        - float(caption_diagnostics["end_padding_ms"]) / 1000.0,
    )
    try:
        explicit_start = float(raw_clip.get("required_first_speech_sec"))
    except (TypeError, ValueError, OverflowError):
        explicit_start = cue_start
    try:
        explicit_end = float(raw_clip.get("required_last_speech_sec"))
    except (TypeError, ValueError, OverflowError):
        explicit_end = cue_end
    required_start = (
        explicit_start
        if math.isfinite(explicit_start) and cue_start <= explicit_start < cue_end
        else cue_start
    )
    required_end = (
        explicit_end
        if math.isfinite(explicit_end) and cue_start < explicit_end <= cue_end
        else cue_end
    )
    return (
        (required_start, required_end)
        if required_end > required_start
        else (cue_start, cue_end)
    )


def _complete_caption_speech_bounds(
    raw_clip: dict[str, Any],
    caption_diagnostics: dict[str, Any],
) -> tuple[float, float]:
    """Keep the complete selected cues when exact lexical projection is unavailable."""

    without_partial_hints = dict(raw_clip)
    without_partial_hints.pop("required_first_speech_sec", None)
    without_partial_hints.pop("required_last_speech_sec", None)
    return _required_speech_bounds(without_partial_hints, caption_diagnostics)


def _boundary_evidence_grade(
    context: object,
    *,
    t_start: object | None = None,
    t_end: object | None = None,
) -> int:
    """Order persisted boundary evidence so retries can only improve a reel."""

    if (
        not isinstance(context, dict)
        or str(context.get("selection_contract_version") or "").strip()
        != "quality_silence_v27"
    ):
        return 0
    if (
        clip_engine_silence.persisted_boundary_is_verified(context)
        and clip_engine_silence.persisted_boundary_is_usable(
            context, t_start=t_start, t_end=t_end
        )
    ):
        return 2
    if clip_engine_silence.persisted_boundary_is_usable(
        context, t_start=t_start, t_end=t_end
    ):
        return 1
    return 0


def _overlapping_caption_end_handoff(
    transcript: dict[str, Any],
    raw_clip: dict[str, Any],
) -> tuple[float, dict[str, Any]] | None:
    """Use the next cue onset only as an acoustically proven overlap handoff.

    Hosted rolling captions commonly keep one cue displayed after the next cue
    begins.  The stale display end is not a speech timestamp.  This helper does
    not authorize a cut by itself; it records the adjacent onset that the
    acoustic gate must later prove lies inside one two-sided quiet run.
    """
    segments = [
        segment
        for segment in (transcript.get("segments") or [])
        if isinstance(segment, dict)
    ]
    index_by_id = {
        str(segment.get("cue_id") or f"cue-{index}"): index
        for index, segment in enumerate(segments)
    }
    cue_ids = [str(value or "").strip() for value in (raw_clip.get("cue_ids") or [])]
    if not cue_ids or any(cue_id not in index_by_id for cue_id in cue_ids):
        return None
    indices = [index_by_id[cue_id] for cue_id in cue_ids]
    if indices != list(range(indices[0], indices[-1] + 1)):
        return None
    last_index = indices[-1]
    if last_index + 1 >= len(segments):
        return None
    selected = segments[last_index]
    following = segments[last_index + 1]
    try:
        selected_start = float(selected.get("start"))
        display_end = float(selected.get("end"))
        next_onset = float(following.get("start"))
    except (TypeError, ValueError, OverflowError):
        return None
    selected_speech_units = _caption_speech_units(
        str(selected.get("text") or "")
    )
    selected_text = str(selected.get("text") or "").strip()
    following_text = str(following.get("text") or "").strip()
    minimum_selected_span = max(
        PARTIAL_CUE_MATERIALITY_SEC,
        selected_speech_units / MAX_PLAUSIBLE_CAPTION_WORDS_PER_SEC,
    )
    if (
        not math.isfinite(selected_start)
        or not math.isfinite(display_end)
        or not math.isfinite(next_onset)
        or next_onset < 0.0
        or not selected_speech_units
        or next_onset < selected_start + minimum_selected_span
        or next_onset + SPEECH_OWNERSHIP_EPSILON_SEC >= display_end
        or _CAPTION_INCOMPLETE_END_RE.search(selected_text)
        or not _CAPTION_FRESH_UNIT_ONSET_RE.match(following_text)
    ):
        return None
    return next_onset, {
        "mode": "next_cue_onset_two_sided_quiet",
        "selected_cue_id": cue_ids[-1],
        "next_cue_id": str(
            following.get("cue_id") or f"cue-{last_index + 1}"
        ),
        "display_end_sec": round(display_end, 3),
        "next_cue_onset_sec": round(next_onset, 3),
        "overlap_sec": round(display_end - next_onset, 3),
    }


def _caption_overlap_end_handoff_is_valid(
    transcript: dict[str, Any],
    raw_clip: dict[str, Any],
    *,
    required_end: float,
    projection_diagnostics: dict[str, Any],
) -> bool:
    canonical = _overlapping_caption_end_handoff(transcript, raw_clip)
    return bool(
        canonical is not None
        and math.isfinite(required_end)
        and abs(required_end - canonical[0]) <= SPEECH_OWNERSHIP_EPSILON_SEC
        and projection_diagnostics.get("caption_overlap_end_handoff")
        == canonical[1]
    )


def _interpolated_caption_edge_anchor(
    *,
    cue_text: str,
    quote: str,
    edge: str,
    cue_start_sec: float,
    cue_end_sec: float,
    occurrence: str | None = None,
) -> tuple[float, float] | None:
    """Estimate one uniquely grounded edge inside a timestamped caption cue.

    Some transcript providers omit native word timings. Uniform token
    interpolation is a cheap refinement for an explicit filler-edge marker.
    Ambiguous input returns ``None`` so the caller keeps the complete cue.
    """
    matches = list(_QUOTE_WORD_RE.finditer(str(cue_text or "")))
    spans = _quote_character_spans(cue_text, quote)
    selected_span = (
        spans[0]
        if occurrence == "first" and spans
        else (
            spans[-1]
            if occurrence == "last" and spans
            else (spans[0] if len(spans) == 1 else None)
        )
    )
    if (
        edge not in {"start", "end"}
        or selected_span is None
        or len(matches) < 2
        or not math.isfinite(cue_start_sec)
        or not math.isfinite(cue_end_sec)
        or cue_end_sec <= cue_start_sec
    ):
        return None
    span_start, span_end = selected_span
    selected = [
        index
        for index, match in enumerate(matches)
        if match.start() >= span_start and match.end() <= span_end
    ]
    if not selected or selected != list(range(selected[0], selected[-1] + 1)):
        return None
    duration = cue_end_sec - cue_start_sec
    token_count = len(matches)
    if (
        duration + SPEECH_OWNERSHIP_EPSILON_SEC
        < token_count / MAX_PLAUSIBLE_CAPTION_WORDS_PER_SEC
    ):
        return None

    def onset(index: int) -> float:
        return cue_start_sec + duration * (index / token_count)

    if edge == "start":
        if selected[0] == 0:
            return None
        required = max(
            cue_start_sec,
            onset(selected[0]) - CAPTION_PROJECTION_START_PREROLL_SEC,
        )
        return required, min(required, onset(selected[0] - 1))
    if selected[-1] + 1 >= token_count:
        return None
    last_onset = onset(selected[-1])
    excluded_onset = onset(selected[-1] + 1)
    required_end = min(
        last_onset + PROJECTED_END_COVERAGE_SEC,
        (last_onset + excluded_onset) / 2.0,
    )
    return required_end, excluded_onset


def _projected_speech_bounds(
    transcript: dict[str, Any],
    raw_clip: dict[str, Any],
    caption_diagnostics: dict[str, Any],
    prepared: object,
) -> tuple[tuple[float, float], dict[str, Any], str | None]:
    """Resolve selected speech edges from explicit native word onsets when available."""
    fallback = _required_speech_bounds(raw_clip, caption_diagnostics)
    edge_projection = raw_clip.get("edge_projection")
    projections = edge_projection if isinstance(edge_projection, dict) else {}

    source = getattr(prepared, "source", None)
    words = tuple(getattr(source, "lexical_words", ()) or ())
    if not words:
        if projections:
            segments = [
                segment
                for segment in (transcript.get("segments") or [])
                if isinstance(segment, dict)
            ]
            index_by_id = {
                str(segment.get("cue_id") or f"cue-{index}").strip(): index
                for index, segment in enumerate(segments)
            }
            selected_ids = [
                str(value or "").strip()
                for value in (raw_clip.get("cue_ids") or [])
            ]
            if (
                not selected_ids
                or any(cue_id not in index_by_id for cue_id in selected_ids)
            ):
                return fallback, {}, "projection_selected_cues_unavailable"
            selected_indices = [index_by_id[cue_id] for cue_id in selected_ids]
            if selected_indices != list(
                range(selected_indices[0], selected_indices[-1] + 1)
            ):
                return fallback, {}, "projection_selected_cues_unavailable"
            required_start, required_end = fallback
            estimated: dict[str, Any] = {}
            expected_cue_ids = {
                "start": str(caption_diagnostics.get("start_cue_id") or ""),
                "end": str(caption_diagnostics.get("end_cue_id") or ""),
            }
            for edge in ("start", "end"):
                marker = projections.get(edge)
                if not isinstance(marker, dict):
                    continue
                cue_id = str(marker.get("cue_id") or "").strip()
                quote = " ".join(str(marker.get("quote") or "").split())
                selected_index = (
                    selected_indices[0] if edge == "start" else selected_indices[-1]
                )
                if (
                    not cue_id
                    or cue_id != expected_cue_ids[edge]
                    or not quote
                    or index_by_id.get(cue_id) != selected_index
                ):
                    return fallback, {}, f"{edge}_projection_marker_invalid"
                cue = segments[selected_index]
                try:
                    cue_start = float(cue.get("start"))
                    cue_end = float(cue.get("end"))
                except (TypeError, ValueError, OverflowError):
                    return fallback, {}, f"{edge}_caption_interpolation_unavailable"
                if selected_index + 1 < len(segments):
                    try:
                        following_start = float(
                            segments[selected_index + 1].get("start")
                        )
                    except (TypeError, ValueError, OverflowError):
                        following_start = cue_end
                    if (
                        math.isfinite(following_start)
                        and cue_start < following_start < cue_end
                    ):
                        cue_end = following_start
                anchor = _interpolated_caption_edge_anchor(
                    cue_text=str(cue.get("text") or ""),
                    quote=quote,
                    edge=edge,
                    cue_start_sec=cue_start,
                    cue_end_sec=cue_end,
                    occurrence=str(marker.get("occurrence") or "").strip()
                    or None,
                )
                if anchor is None:
                    return fallback, {}, f"{edge}_caption_interpolation_unavailable"
                required_sec, excluded_sec = anchor
                if edge == "start":
                    required_start = required_sec
                else:
                    required_end = required_sec
                estimated[edge] = {
                    "cue_id": cue_id,
                    "quote": quote,
                    "mode": "caption_token_interpolation",
                    "required_speech_sec": round(required_sec, 3),
                    "excluded_neighbor_onset_sec": round(excluded_sec, 3),
                }
            overlap_handoff = (
                None
                if "end" in estimated or isinstance(projections.get("end"), dict)
                else _overlapping_caption_end_handoff(transcript, raw_clip)
            )
            if (
                overlap_handoff is not None
                and abs(fallback[1] - overlap_handoff[1]["display_end_sec"])
                > SPEECH_OWNERSHIP_EPSILON_SEC
            ):
                overlap_handoff = None
            if overlap_handoff is not None:
                required_end, overlap_diagnostics = overlap_handoff
            else:
                overlap_diagnostics = None
            if (
                not estimated
                or not math.isfinite(required_start)
                or not math.isfinite(required_end)
                or required_end <= required_start
            ):
                return fallback, {}, "caption_projection_invalid"
            return (
                (required_start, required_end),
                {
                    "caption_projection_verified": True,
                    **estimated,
                    **(
                        {"caption_overlap_end_handoff": overlap_diagnostics}
                        if overlap_diagnostics is not None
                        else {}
                    ),
                },
                None,
            )
        overlap_handoff = (
            None
            if isinstance(projections.get("end"), dict)
            else _overlapping_caption_end_handoff(transcript, raw_clip)
        )
        if (
            not projections
            and overlap_handoff is not None
            and abs(fallback[1] - overlap_handoff[1]["display_end_sec"])
            <= SPEECH_OWNERSHIP_EPSILON_SEC
        ):
            next_onset, overlap_diagnostics = overlap_handoff
            if next_onset <= fallback[0]:
                return fallback, {}, "caption_overlap_handoff_invalid"
            return (
                (fallback[0], next_onset),
                {"caption_overlap_end_handoff": overlap_diagnostics},
                None,
            )
        return (
            (fallback, {}, "lexical_timing_unavailable")
            if projections
            else (fallback, {}, None)
        )

    segments = [
        segment
        for segment in (transcript.get("segments") or [])
        if isinstance(segment, dict)
    ]
    index_by_id = {
        str(segment.get("cue_id") or f"cue-{index}").strip(): index
        for index, segment in enumerate(segments)
    }
    selected_ids = [
        str(value or "").strip() for value in (raw_clip.get("cue_ids") or [])
    ]
    if (
        not selected_ids
        or any(cue_id not in index_by_id for cue_id in selected_ids)
    ):
        return (
            (fallback, {}, "projection_selected_cues_unavailable")
            if projections
            else (fallback, {}, None)
        )
    selected_indices = [index_by_id[cue_id] for cue_id in selected_ids]
    if selected_indices != list(
        range(selected_indices[0], selected_indices[-1] + 1)
    ):
        return (
            (fallback, {}, "projection_selected_cues_unavailable")
            if projections
            else (fallback, {}, None)
        )
    first_index = selected_indices[0]
    last_index = selected_indices[-1]
    required_start, required_end = fallback
    verified: dict[str, Any] = {}
    expected_cue_ids = {
        "start": str(caption_diagnostics.get("start_cue_id") or ""),
        "end": str(caption_diagnostics.get("end_cue_id") or ""),
    }
    from ..clip_engine import lexical_timing

    for edge in ("start", "end"):
        marker = projections.get(edge)
        is_projected = isinstance(marker, dict)
        if is_projected:
            cue_id = str(marker.get("cue_id") or "").strip()
            quote = " ".join(str(marker.get("quote") or "").split())
            selected_index = first_index if edge == "start" else last_index
            cue = segments[selected_index]
            if (
                not cue_id
                or cue_id != expected_cue_ids[edge]
                or not quote
                or index_by_id.get(cue_id) != selected_index
            ):
                return fallback, {}, f"{edge}_projection_marker_invalid"
            context_text = str(cue.get("text") or "")
            context_start = float(cue.get("start") or 0.0)
            context_end = float(cue.get("end") or 0.0)
        else:
            quote = " ".join(
                str(raw_clip.get(f"{edge}_quote") or "").split()
            )
            if not quote:
                continue
            if edge == "start":
                if first_index == 0:
                    continue
                prior = segments[first_index - 1]
                selected = segments[first_index]
                context_text = (
                    f"{str(prior.get('text') or '')} "
                    f"{str(selected.get('text') or '')}"
                )
                context_start = float(prior.get("start") or 0.0)
                context_end = float(selected.get("end") or 0.0)
            else:
                if last_index + 1 >= len(segments):
                    continue
                selected = segments[last_index]
                following = segments[last_index + 1]
                context_text = (
                    f"{str(selected.get('text') or '')} "
                    f"{str(following.get('text') or '')}"
                )
                context_start = float(selected.get("start") or 0.0)
                context_end = float(following.get("end") or 0.0)
        anchor = lexical_timing.align_edge_anchor(
            words,
            cue_text=context_text,
            quote=quote,
            edge=edge,
            cue_start_sec=context_start,
            cue_end_sec=context_end,
            occurrence=(
                str(marker.get("occurrence") or "").strip()
                if is_projected
                else None
            )
            or None,
        )
        if anchor is None:
            if is_projected:
                return fallback, {}, f"{edge}_lexical_alignment_unavailable"
            continue
        excluded_sec = float(anchor.excluded_neighbor_onset_sec)
        required_sec = (
            float(anchor.quote_start_sec)
            if edge == "start"
            else (
                min(
                    float(anchor.quote_last_onset_sec)
                    + PROJECTED_END_COVERAGE_SEC,
                    (
                        float(anchor.quote_last_onset_sec)
                        + excluded_sec
                    )
                    / 2.0,
                )
                if is_projected
                else float(anchor.quote_last_onset_sec)
            )
        )
        if (
            not math.isfinite(required_sec)
            or not math.isfinite(excluded_sec)
            or (
                excluded_sec >= required_sec
                if edge == "start"
                else excluded_sec <= required_sec
            )
        ):
            if is_projected:
                return fallback, {}, f"{edge}_lexical_alignment_unavailable"
            continue
        if edge == "start":
            required_start = required_sec
        else:
            required_end = required_sec
        verified[edge] = {
            "cue_id": expected_cue_ids[edge],
            "quote": quote,
            "mode": "projected" if is_projected else "cue_edge",
            "required_speech_sec": round(required_sec, 3),
            "excluded_neighbor_onset_sec": round(excluded_sec, 3),
        }

    if (
        not math.isfinite(required_start)
        or not math.isfinite(required_end)
        or required_end <= required_start
    ):
        return fallback, {}, "lexical_projection_invalid"
    overlap_handoff = (
        None
        if "end" in verified or isinstance(projections.get("end"), dict)
        else _overlapping_caption_end_handoff(transcript, raw_clip)
    )
    if (
        overlap_handoff is not None
        and abs(fallback[1] - overlap_handoff[1]["display_end_sec"])
        > SPEECH_OWNERSHIP_EPSILON_SEC
    ):
        overlap_handoff = None
    if overlap_handoff is not None:
        required_end, overlap_diagnostics = overlap_handoff
    else:
        overlap_diagnostics = None
    if required_end <= required_start:
        return fallback, {}, "caption_overlap_handoff_invalid"
    return (
        (required_start, required_end),
        {
            **(
                {
                    "lexical_boundary_verified": True,
                    "lexical_projection_verified": any(
                        details.get("mode") == "projected"
                        for details in verified.values()
                    ),
                }
                if verified
                else {}
            ),
            **verified,
            **(
                {"caption_overlap_end_handoff": overlap_diagnostics}
                if overlap_diagnostics is not None
                else {}
            ),
        },
        None,
    )


def _selected_caption_cues(
    transcript: dict[str, Any],
    raw_clip: dict[str, Any],
    *,
    boundary_bounds: tuple[float, float] | None = None,
) -> list[dict[str, Any]]:
    """Snapshot the exact selected cues so later transcript refreshes cannot drift."""
    cue_ids = [
        str(cue_id or "").strip()
        for cue_id in (raw_clip.get("cue_ids") or [])
        if str(cue_id or "").strip()
    ]
    segments_by_id = {
        str(segment.get("cue_id") or f"cue-{index}").strip(): segment
        for index, segment in enumerate(transcript.get("segments") or [])
        if isinstance(segment, dict)
    }
    if not cue_ids or any(cue_id not in segments_by_id for cue_id in cue_ids):
        return []
    edge_projection = raw_clip.get("edge_projection")
    projection = edge_projection if isinstance(edge_projection, dict) else {}

    cues: list[dict[str, Any]] = []
    for index, cue_id in enumerate(cue_ids):
        segment = segments_by_id[cue_id]
        text = str(segment.get("text") or "")
        text_start, text_end = 0, len(text)
        start_marker = projection.get("start") if index == 0 else None
        end_marker = projection.get("end") if index == len(cue_ids) - 1 else None
        if isinstance(start_marker, dict) and str(start_marker.get("cue_id") or "") == cue_id:
            spans = _quote_character_spans(
                text, str(start_marker.get("quote") or "")
            )
            occurrence = str(start_marker.get("occurrence") or "").strip()
            if spans and (len(spans) == 1 or occurrence in {"first", "last"}):
                span = spans[-1] if occurrence == "last" else spans[0]
                text_start = span[0]
        if isinstance(end_marker, dict) and str(end_marker.get("cue_id") or "") == cue_id:
            spans = _quote_character_spans(
                text, str(end_marker.get("quote") or "")
            )
            occurrence = str(end_marker.get("occurrence") or "").strip()
            if spans and (len(spans) == 1 or occurrence in {"first", "last"}):
                span = spans[-1] if occurrence == "last" else spans[0]
                text_end = span[1]
        cue_start = float(segment["start"])
        cue_end = float(segment["end"])
        if boundary_bounds is not None:
            if isinstance(start_marker, dict):
                cue_start = float(boundary_bounds[0])
            if isinstance(end_marker, dict):
                cue_end = float(boundary_bounds[1])
        cues.append({
            "cue_id": cue_id,
            "start": cue_start,
            "end": cue_end,
            "text": text[text_start:text_end].strip(),
            "lang": str(segment.get("lang") or ""),
        })
    return cues


def _selection_snapshot_payload(
    selection_context: dict[str, Any],
    *,
    clip_start: float,
    clip_end: float,
    selected_cue_ids: list[str],
) -> tuple[str, list[dict[str, Any]]] | None:
    """Build the provisional snippet/captions from immutable selected evidence."""
    raw_cues = selection_context.get("selection_caption_cues")
    if not isinstance(raw_cues, list) or not raw_cues or clip_end <= clip_start:
        return None

    clip_len = max(0.2, clip_end - clip_start)
    selected_ids = {cue_id for cue_id in selected_cue_ids if cue_id}
    captions: list[dict[str, Any]] = []
    for index, entry in enumerate(raw_cues):
        if not isinstance(entry, dict):
            continue
        cue_id = str(entry.get("cue_id") or f"cue-{index}")
        if selected_ids and cue_id not in selected_ids:
            continue
        text = str(entry.get("text") or "").replace("\n", " ").strip()
        if not text:
            continue
        try:
            cue_start = float(entry.get("start") or 0.0)
            cue_end = float(entry["end"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if (
            not math.isfinite(cue_start)
            or not math.isfinite(cue_end)
            or cue_end <= cue_start
            or cue_end <= clip_start
            or cue_start >= clip_end
        ):
            continue

        relative_start = max(0.0, max(cue_start, clip_start) - clip_start)
        relative_end = min(clip_len, min(cue_end, clip_end) - clip_start)
        if relative_end - relative_start < 0.16:
            relative_end = min(clip_len, relative_start + 0.9)
        payload = {
            "start": round(float(relative_start), 2),
            "end": round(
                float(min(clip_len, max(relative_end, relative_start + 0.16))),
                2,
            ),
            "text": text,
        }
        if (
            captions
            and payload["text"] == captions[-1]["text"]
            and payload["start"] - captions[-1]["end"] <= 0.2
        ):
            captions[-1]["end"] = payload["end"]
        else:
            captions.append(payload)

    if not captions:
        return None
    return " ".join(caption["text"] for caption in captions)[:7000], captions


def _selected_speech_corridor(
    transcript: dict[str, Any],
    raw_clip: dict[str, Any],
    caption_diagnostics: dict[str, Any],
    *,
    source_end_sec: float | None = None,
    required_speech_bounds: tuple[float, float] | None = None,
    projection_diagnostics: dict[str, Any] | None = None,
) -> tuple[float, float, str | None]:
    """Use cue onsets as the acoustic fences around selected rolling captions."""
    segments = [
        segment
        for segment in list(transcript.get("segments") or [])
        if isinstance(segment, dict)
    ]
    transcript_end = max(
        [
            float(transcript.get("duration") or 0.0),
            *[float(segment.get("end") or 0.0) for segment in segments],
        ],
        default=0.0,
    )
    source_end = transcript_end
    try:
        candidate_source_end = float(source_end_sec)
    except (TypeError, ValueError, OverflowError):
        candidate_source_end = 0.0
    if math.isfinite(candidate_source_end) and candidate_source_end >= transcript_end:
        source_end = candidate_source_end
    if not segments:
        return 0.0, source_end, "selected_cue_range_unavailable"

    selected_ids = {
        str(cue_id)
        for cue_id in (raw_clip.get("cue_ids") or [])
        if str(cue_id or "").strip()
    }
    selected_indices = [
        index
        for index, segment in enumerate(segments)
        if str(segment.get("cue_id") or f"cue-{index}") in selected_ids
    ]
    if not selected_indices:
        return 0.0, source_end, "selected_cue_range_unavailable"

    first_index = min(selected_indices)
    last_index = max(selected_indices)
    if selected_indices != list(range(first_index, last_index + 1)):
        return 0.0, source_end, "selected_cue_range_unavailable"

    first_start = float(segments[first_index].get("start") or 0.0)
    last_end = float(segments[last_index].get("end") or first_start)
    required_start, required_end = (
        required_speech_bounds
        if required_speech_bounds is not None
        else _required_speech_bounds(raw_clip, caption_diagnostics)
    )
    timed_boundaries = (
        projection_diagnostics
        if isinstance(projection_diagnostics, dict)
        and (
            projection_diagnostics.get("lexical_boundary_verified") is True
            or projection_diagnostics.get("lexical_projection_verified") is True
            or projection_diagnostics.get("caption_projection_verified") is True
        )
        else {}
    )
    start_boundary = timed_boundaries.get("start")
    end_boundary = timed_boundaries.get("end")
    start_is_timed = isinstance(start_boundary, dict)
    end_is_timed = isinstance(end_boundary, dict)
    end_is_overlap_handoff = _caption_overlap_end_handoff_is_valid(
        transcript,
        raw_clip,
        required_end=required_end,
        projection_diagnostics=(
            projection_diagnostics
            if isinstance(projection_diagnostics, dict)
            else {}
        ),
    )
    start_is_partial = required_start > first_start + PARTIAL_CUE_MATERIALITY_SEC
    end_is_partial = required_end < last_end - PARTIAL_CUE_MATERIALITY_SEC
    if (
        (start_is_partial and not start_is_timed)
        or (
            end_is_partial
            and not end_is_timed
            and not end_is_overlap_handoff
        )
    ):
        return 0.0, source_end, "partial_cue_edge_requires_projection"

    start_limit = (
        float(start_boundary["excluded_neighbor_onset_sec"])
        if start_is_timed
        else 0.0
        if first_index == 0
        else min(
            first_start,
            float(segments[first_index - 1].get("end") or first_start),
        )
    )
    end_limit = (
        float(end_boundary["excluded_neighbor_onset_sec"])
        if end_is_timed
        else source_end
        if last_index + 1 >= len(segments)
        else float(segments[last_index + 1]["start"])
    )
    if start_limit > required_start:
        if start_limit - required_start <= SPEECH_OWNERSHIP_EPSILON_SEC:
            start_limit = required_start
        else:
            return start_limit, end_limit, "selected_cue_range_unavailable"
    if end_limit < required_end:
        if required_end - end_limit <= SPEECH_OWNERSHIP_EPSILON_SEC:
            end_limit = required_end
        else:
            return start_limit, end_limit, "selected_cue_range_unavailable"
    if end_limit <= start_limit:
        return start_limit, end_limit, "selected_cue_range_unavailable"
    return max(0.0, start_limit), min(source_end, end_limit), None


def _acoustic_boundary_plan(
    transcript: dict[str, Any],
    raw_clip: dict[str, Any],
    projection_diagnostics: dict[str, Any],
    *,
    speech_bounds: tuple[float, float],
    search_limits: tuple[float, float],
) -> tuple[float, float, bool, bool, bool, bool] | None:
    """Separate speech targets from their allowed acoustic search corridors."""
    segments = list(transcript.get("segments") or [])
    index_by_id = {
        str(segment.get("cue_id") or f"cue-{index}"): index
        for index, segment in enumerate(segments)
        if isinstance(segment, dict)
    }
    cue_ids = [str(cue_id) for cue_id in (raw_clip.get("cue_ids") or [])]
    if not cue_ids or any(cue_id not in index_by_id for cue_id in cue_ids):
        return None
    indices = [index_by_id[cue_id] for cue_id in cue_ids]
    if indices != list(range(indices[0], indices[-1] + 1)):
        return None
    first_index = min(indices)
    last_index = max(indices)
    start_target = float(speech_bounds[0])
    start_is_lexical = isinstance(projection_diagnostics.get("start"), dict)
    end_is_lexical = isinstance(projection_diagnostics.get("end"), dict)
    end_target = float(speech_bounds[1])
    search_start, search_end = search_limits
    if (
        not all(
            math.isfinite(value)
            for value in (start_target, end_target, search_start, search_end)
        )
        or start_target < search_start - SPEECH_OWNERSHIP_EPSILON_SEC
        or end_target > search_end + SPEECH_OWNERSHIP_EPSILON_SEC
        or end_target <= start_target
    ):
        return None
    start_two_sided = start_is_lexical or first_index > 0
    end_is_overlap_handoff = _caption_overlap_end_handoff_is_valid(
        transcript,
        raw_clip,
        required_end=end_target,
        projection_diagnostics=projection_diagnostics,
    )
    end_two_sided = (
        end_is_lexical
        or end_is_overlap_handoff
        or last_index + 1 < len(segments)
    )
    start_handoff = not start_is_lexical and (
        start_two_sided
        or (
            first_index == 0
            and start_target > search_start + SPEECH_OWNERSHIP_EPSILON_SEC
        )
    )
    end_handoff = end_is_overlap_handoff or (
        not end_is_lexical
        and last_index + 1 >= len(segments)
        and end_target < search_end - SPEECH_OWNERSHIP_EPSILON_SEC
    )
    return (
        start_target,
        end_target,
        start_handoff,
        end_handoff,
        start_two_sided,
        end_two_sided,
    )


def _transcript_aligned_result(
    acoustic: object,
    *,
    speech_bounds: tuple[float, float],
    search_limits: tuple[float, float],
    projection_diagnostics: dict[str, Any],
) -> clip_engine_silence.SilenceVerificationResult:
    """Use the selected complete transcript thought when audio is not preferred.

    Gemini's exact edge quotes and deterministic discourse checks choose the
    semantic unit. Native lexical timestamps, or conservative token
    interpolation for a uniquely grounded filler edge, may tighten a coarse
    cue. If refinement is uncertain the caller retains the whole cue.
    """

    if bool(getattr(acoustic, "verified", False)):
        return acoustic
    diagnostics = dict(getattr(acoustic, "diagnostics", {}) or {})
    reason = str(diagnostics.get("reason") or "sentence_boundary_selected").strip()
    stage = str(diagnostics.get("stage") or "transcript").strip()
    if reason == "cancelled":
        return acoustic
    required_start, required_end = speech_bounds
    semantic_start, semantic_end = search_limits
    overlap_handoff = projection_diagnostics.get(
        "caption_overlap_end_handoff"
    )
    overlap_fallback_end: float | None = None
    if (
        not isinstance(projection_diagnostics.get("end"), dict)
        and isinstance(overlap_handoff, dict)
        and overlap_handoff.get("mode") == "next_cue_onset_two_sided_quiet"
    ):
        try:
            next_onset = float(overlap_handoff["next_cue_onset_sec"])
            display_end = float(overlap_handoff["display_end_sec"])
        except (KeyError, TypeError, ValueError, OverflowError):
            pass
        else:
            if (
                math.isfinite(next_onset)
                and math.isfinite(display_end)
                and abs(next_onset - required_end)
                <= SPEECH_OWNERSHIP_EPSILON_SEC
                and display_end > next_onset
            ):
                # A rolling-caption onset is only an acoustic cut candidate.
                # Without verified quiet at that handoff, retain the complete
                # selected cue instead of treating its neighbor's onset as a
                # transcript-authoritative speech boundary.
                overlap_fallback_end = display_end
                semantic_end = max(semantic_end, display_end)
    final_start = required_start
    final_end = (
        overlap_fallback_end
        if overlap_fallback_end is not None
        else (
            semantic_end
            if isinstance(projection_diagnostics.get("end"), dict)
            else required_end
        )
    )
    if (
        not all(
            math.isfinite(value)
            for value in (
                required_start,
                required_end,
                semantic_start,
                semantic_end,
                final_start,
                final_end,
            )
        )
        or semantic_start > final_start
        or final_start > required_start
        or required_end > final_end
        or final_end > semantic_end
        or final_end <= final_start
    ):
        return acoustic
    return clip_engine_silence.SilenceVerificationResult(
        "context_aligned",
        round(final_start, 3),
        round(final_end, 3),
        {
            **diagnostics,
            "stage": stage,
            "reason": reason,
            "context_aligned": True,
            "required_speech_range": [
                round(required_start, 3),
                round(required_end, 3),
            ],
            "semantic_range": [
                round(semantic_start, 3),
                round(semantic_end, 3),
            ],
            "final_range": [round(final_start, 3), round(final_end, 3)],
        },
    )


def _context_result_range_is_safe(
    result: object,
    *,
    source_end_sec: float,
) -> bool:
    """Validate a transcript fallback without requiring perfect edge evidence."""

    if str(getattr(result, "status", "")) != "context_aligned":
        return False
    try:
        start = float(getattr(result, "start_sec"))
        end = float(getattr(result, "end_sec"))
        source_end = float(source_end_sec)
    except (TypeError, ValueError, OverflowError):
        return False
    diagnostics = dict(getattr(result, "diagnostics", {}) or {})
    final = diagnostics.get("final_range")
    if not isinstance(final, (list, tuple)) or len(final) != 2:
        return False
    try:
        final_start, final_end = (float(value) for value in final)
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(
        all(math.isfinite(value) for value in (start, end, source_end, final_start, final_end))
        and start >= 0.0
        and end > start
        and end <= source_end + 1e-3
        and abs(start - final_start) <= 1e-3
        and abs(end - final_end) <= 1e-3
    )


def _acoustic_range_is_safe(
    *,
    start_sec: float,
    end_sec: float,
    required_start_sec: float,
    required_end_sec: float,
    semantic_start_limit_sec: float,
    semantic_end_limit_sec: float,
    source_end_sec: float,
    diagnostics: dict[str, Any],
    require_start_handoff: bool = False,
    require_end_handoff: bool = False,
    require_start_two_sided: bool = False,
    require_end_two_sided: bool = False,
) -> bool:
    """Validate progressive and projected acoustic edges independently."""
    values = (
        start_sec,
        end_sec,
        required_start_sec,
        required_end_sec,
        semantic_start_limit_sec,
        semantic_end_limit_sec,
        source_end_sec,
    )
    if (
        not all(math.isfinite(value) for value in values)
        or start_sec < 0.0
        or end_sec <= start_sec
        or end_sec > source_end_sec + 1e-3
        or start_sec > required_start_sec + SPEECH_OWNERSHIP_EPSILON_SEC
        or (
            not require_end_handoff
            and end_sec + SPEECH_OWNERSHIP_EPSILON_SEC < required_end_sec
        )
    ):
        return False

    start_crosses_fence = start_sec < (
        semantic_start_limit_sec - SPEECH_OWNERSHIP_EPSILON_SEC
    )
    end_crosses_fence = end_sec > (
        semantic_end_limit_sec + SPEECH_OWNERSHIP_EPSILON_SEC
    )
    if (start_crosses_fence and not require_start_handoff) or (
        end_crosses_fence and not require_end_handoff
    ):
        return False
    if (
        require_start_two_sided
        and diagnostics.get("start_two_sided_required") is not True
    ) or (
        require_end_two_sided
        and diagnostics.get("end_two_sided_required") is not True
    ):
        return False

    cut_epsilon = 0.011
    tolerance = clip_engine_silence.HANDOFF_TIMESTAMP_TOLERANCE_SEC
    legacy_handoff = diagnostics.get("speech_handoff_verified") is True

    def handoff_is_safe(edge: str) -> bool:
        explicit_verified = diagnostics.get(f"{edge}_speech_handoff_verified")
        if explicit_verified is not True and not (
            explicit_verified is None and legacy_handoff
        ):
            return False
        require_two_sided = (
            require_start_two_sided if edge == "start" else require_end_two_sided
        )
        if diagnostics.get(f"{edge}_two_sided_required") is not require_two_sided:
            return False
        try:
            diagnostic_start_limit = float(
                diagnostics["semantic_start_limit_sec"]
            )
            diagnostic_end_limit = float(diagnostics["semantic_end_limit_sec"])
            quiet = tuple(
                float(value) for value in diagnostics[f"{edge}_quiet"]
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            return False
        if (
            len(quiet) != 2
            or not all(
                math.isfinite(value)
                for value in (diagnostic_start_limit, diagnostic_end_limit, *quiet)
            )
            or abs(diagnostic_start_limit - semantic_start_limit_sec) > 1e-3
            or abs(diagnostic_end_limit - semantic_end_limit_sec) > 1e-3
            or quiet[0] > quiet[1]
        ):
            return False

        required = required_start_sec if edge == "start" else required_end_sec
        cut = start_sec if edge == "start" else end_sec
        if not (
            quiet[0] <= required + tolerance
            and quiet[1] >= required - tolerance
            and quiet[0] - cut_epsilon <= cut <= quiet[1] + cut_epsilon
        ):
            return False
        if edge == "start":
            return bool(
                cut
                >= max(
                    0.0,
                    semantic_start_limit_sec
                    - clip_engine_silence.HANDOFF_OBSERVATION_HALO_SEC,
                )
                - cut_epsilon
                and (
                    not start_crosses_fence
                    or quiet[1] >= semantic_start_limit_sec - tolerance
                )
            )
        return bool(
            cut
            <= min(
                source_end_sec,
                semantic_end_limit_sec
                + clip_engine_silence.HANDOFF_OBSERVATION_HALO_SEC,
            )
            + cut_epsilon
            and (
                not end_crosses_fence
                or quiet[0] <= semantic_end_limit_sec + tolerance
            )
        )

    return bool(
        (not require_start_handoff or handoff_is_safe("start"))
        and (not require_end_handoff or handoff_is_safe("end"))
    )


def _prepared_media_end_sec(
    prepared: object,
    *,
    transcript_end_sec: float,
) -> float:
    """Use resolved media duration only when it safely encloses the transcript."""
    source = getattr(prepared, "source", None)
    try:
        duration = float(getattr(source, "duration_sec", None))
    except (TypeError, ValueError, OverflowError):
        return transcript_end_sec
    if not math.isfinite(duration) or duration < transcript_end_sec:
        return transcript_end_sec
    return duration


def _direct_adapter_duration_sec(
    transcript: dict[str, Any],
    verified_clips: list[dict[str, Any]],
) -> float:
    """Return known media duration, falling back to transcript timing."""
    candidates: list[float] = []
    raw_values = [
        transcript.get("duration"),
        *[
            clip.get("_verified_media_duration_sec")
            for clip in verified_clips
            if isinstance(clip, dict)
        ],
    ]
    for value in raw_values:
        try:
            duration = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(duration) and duration > 0.0:
            candidates.append(duration)
    return max(candidates, default=0.0)


def _verified_direct_adapter_clips(
    *,
    source_url: str,
    engine_out: dict[str, Any],
    should_cancel: Callable[[], bool] | None,
    limit: int | None = None,
    prepared_audio: clip_engine_silence.AudioPreparationResult | None = None,
    exact_topic: str = "",
    embedding_service: Any = None,
) -> list[dict[str, Any]]:
    """Return difficulty-ranked candidates with complete transcript cuts."""
    transcript = dict(engine_out.get("transcript") or {})
    segments = list(transcript.get("segments") or [])
    transcript_end = max(
        [
            float(transcript.get("duration") or 0.0),
            *[
                float(segment.get("end") or 0.0)
                for segment in segments
                if isinstance(segment, dict)
            ],
        ],
        default=0.0,
    )
    if transcript_end <= 0.0:
        return []

    candidates: list[dict[str, Any]] = []
    for raw_clip in list(engine_out.get("clips") or []):
        if not isinstance(raw_clip, dict):
            continue
        try:
            quality_scores = (
                float(raw_clip.get("informativeness")),
                float(raw_clip.get("topic_relevance")),
                float(raw_clip.get("educational_importance")),
            )
        except (TypeError, ValueError, OverflowError):
            continue
        clip_text = clip_engine_bridge.cue_text(transcript, raw_clip.get("cue_ids"))
        if not clip_text:
            clip_text = clip_engine_bridge.window_text(
                transcript,
                float(raw_clip.get("start") or 0.0),
                float(raw_clip.get("end") or 0.0),
            )
        grounded_evidence_quote = _grounded_topic_evidence_quote(
            clip_text, raw_clip.get("topic_evidence_quote")
        )
        if (
            any(not math.isfinite(score) for score in quality_scores)
            or any(score < 0.75 for score in quality_scores)
            or raw_clip.get("kind") != "educational"
            or raw_clip.get("directly_teaches_topic") is not True
            or raw_clip.get("substantive") is not True
            or raw_clip.get("factually_grounded") is not True
            or raw_clip.get("self_contained") is not True
            or raw_clip.get("is_standalone") is not True
            or any(
                not str(raw_clip.get(field) or "").strip()
                for field in ("title", "learning_objective", "facet", "reason")
            )
            or not grounded_evidence_quote
        ):
            continue
        candidate = dict(raw_clip)
        candidate["_quality_scores"] = quality_scores
        candidate["_grounded_topic_evidence_quote"] = grounded_evidence_quote
        candidates.append(candidate)
    if not candidates:
        return []

    candidates.sort(
        key=lambda candidate: (
            (
                0
                if float(candidate.get("difficulty") or 0.0) < 0.34
                else 1
                if float(candidate.get("difficulty") or 0.0) < 0.67
                else 2
            ),
            0
            if str(candidate.get("intent_role") or "primary").strip().lower()
            == "primary"
            else 1,
            -float(candidate.get("intent_coverage", 1.0) or 0.0),
            -min(candidate["_quality_scores"]),
            -(sum(candidate["_quality_scores"]) / 3.0),
            -float(candidate["_quality_scores"][1]),
            float(candidate.get("start") or 0.0),
            float(candidate.get("end") or 0.0),
            int(candidate.get("sequence_index") or 0),
            str(candidate.get("selection_candidate_id") or ""),
        )
    )

    # Audio lookup is intentionally not started here. Production boundaries are
    # transcript-authoritative; callers may still supply an already prepared
    # source for an opt-in acoustic refinement without changing the fallback.
    prepared = prepared_audio
    media_end = _prepared_media_end_sec(
        prepared,
        transcript_end_sec=transcript_end,
    )

    def verify(raw_clip: dict[str, Any]):
        diagnostics, complete_cue_bounds = _transcript_boundary_seed(
            transcript, raw_clip
        )
        if diagnostics is None:
            return diagnostics, None, {}, (0.0, 0.0)
        speech_bounds, projection, projection_error = _projected_speech_bounds(
            transcript,
            raw_clip,
            diagnostics,
            prepared,
        )
        start_sec, end_sec = speech_bounds
        if projection_error:
            speech_bounds = complete_cue_bounds
            start_sec, end_sec = speech_bounds
            projection = (
                {
                    "context_fallback": {
                        "stage": "transcript",
                        "reason": projection_error,
                    }
                }
                if projection_error
                else {}
            )
        search_start_limit, search_end_limit, corridor_error = _selected_speech_corridor(
            transcript,
            raw_clip,
            diagnostics,
            source_end_sec=media_end,
            required_speech_bounds=speech_bounds,
            projection_diagnostics=projection,
        )
        if corridor_error:
            speech_bounds = complete_cue_bounds
            projection = {
                **projection,
                "context_fallback": {
                    "stage": "transcript",
                    "reason": f"full_cue_fallback:{corridor_error}",
                },
            }
            transcript_boundary = _transcript_aligned_result(
                clip_engine_silence.SilenceVerificationResult(
                    "unavailable",
                    speech_bounds[0],
                    speech_bounds[1],
                    {
                        "stage": "transcript",
                        "reason": f"full_cue_fallback:{corridor_error}",
                    },
                ),
                speech_bounds=speech_bounds,
                search_limits=speech_bounds,
                projection_diagnostics={},
            )
            return diagnostics, transcript_boundary, projection, speech_bounds
        boundary_plan = _acoustic_boundary_plan(
            transcript,
            raw_clip,
            projection,
            speech_bounds=speech_bounds,
            search_limits=(search_start_limit, search_end_limit),
        )
        if boundary_plan is None:
            speech_bounds = complete_cue_bounds
            transcript_boundary = _transcript_aligned_result(
                clip_engine_silence.SilenceVerificationResult(
                    "unavailable",
                    speech_bounds[0],
                    speech_bounds[1],
                    {
                        "stage": "transcript",
                        "reason": "full_cue_fallback:boundary_plan_invalid",
                    },
                ),
                speech_bounds=speech_bounds,
                search_limits=speech_bounds,
                projection_diagnostics={},
            )
            return diagnostics, transcript_boundary, projection, speech_bounds
        (
            start_target,
            end_target,
            start_handoff,
            end_handoff,
            start_two_sided,
            end_two_sided,
        ) = boundary_plan
        if bool(getattr(prepared, "ready", False)):
            acoustic = clip_engine_silence.verify_acoustic_boundaries(
                source_url,
                start_target,
                end_target,
                search_start_limit_sec=search_start_limit,
                search_end_limit_sec=search_end_limit,
                require_speech_handoff=False,
                require_start_speech_handoff=start_handoff,
                require_end_speech_handoff=end_handoff,
                require_start_two_sided=start_two_sided,
                require_end_two_sided=end_two_sided,
                prepared=prepared,
                cancel_check=should_cancel,
            )
        else:
            acoustic = clip_engine_silence.SilenceVerificationResult(
                "unavailable",
                start_target,
                end_target,
                {
                    "stage": "transcript",
                    "reason": "complete_discourse_boundary",
                },
            )
        if (
            acoustic.verified
            and not _acoustic_range_is_safe(
                start_sec=float(acoustic.start_sec),
                end_sec=float(acoustic.end_sec),
                required_start_sec=start_target,
                required_end_sec=end_target,
                semantic_start_limit_sec=search_start_limit,
                semantic_end_limit_sec=search_end_limit,
                source_end_sec=media_end,
                diagnostics=dict(acoustic.diagnostics or {}),
                require_start_handoff=start_handoff,
                require_end_handoff=end_handoff,
                require_start_two_sided=start_two_sided,
                require_end_two_sided=end_two_sided,
            )
        ):
            acoustic = clip_engine_silence.SilenceVerificationResult(
                "unavailable",
                speech_bounds[0],
                speech_bounds[1],
                {
                    **dict(acoustic.diagnostics or {}),
                    "stage": "semantic_corridor",
                    "reason": "acoustic_refinement_unsafe",
                },
            )
        acoustic = _transcript_aligned_result(
            acoustic,
            speech_bounds=speech_bounds,
            search_limits=(search_start_limit, search_end_limit),
            projection_diagnostics=projection,
        )
        return diagnostics, acoustic, projection, speech_bounds

    if bool(getattr(prepared, "ready", False)):
        with ThreadPoolExecutor(
            max_workers=min(3, len(candidates)),
            thread_name_prefix="direct-clip-boundary-verify",
        ) as executor:
            verification_results = list(executor.map(verify, candidates))
    else:
        # Transcript-only checks are short, deterministic Python work. Running
        # them inline avoids thread scheduling overhead on the small Railway
        # instance without changing candidate order or delaying any I/O.
        verification_results = [verify(candidate) for candidate in candidates]

    verified: list[dict[str, Any]] = []
    for raw_clip, (caption, acoustic, projection, speech_bounds) in zip(
        candidates, verification_results, strict=True
    ):
        raise_if_cancelled(should_cancel)
        if caption is None:
            continue
        if acoustic is None or not (
            bool(getattr(acoustic, "verified", False))
            or str(getattr(acoustic, "status", "")) == "context_aligned"
        ):
            acoustic = _transcript_aligned_result(
                clip_engine_silence.SilenceVerificationResult(
                    "unavailable",
                    speech_bounds[0],
                    speech_bounds[1],
                    {
                        "stage": "transcript",
                        "reason": "boundary_refinement_unavailable",
                    },
                ),
                speech_bounds=speech_bounds,
                search_limits=speech_bounds,
                projection_diagnostics={},
            )
        strict_acoustic = bool(getattr(acoustic, "verified", False))
        context_aligned = str(getattr(acoustic, "status", "")) == "context_aligned"
        if strict_acoustic:
            semantic_start, semantic_end, corridor_error = _selected_speech_corridor(
                transcript,
                raw_clip,
                caption,
                source_end_sec=media_end,
                required_speech_bounds=speech_bounds,
                projection_diagnostics=projection,
            )
            boundary_plan = _acoustic_boundary_plan(
                transcript,
                raw_clip,
                projection,
                speech_bounds=speech_bounds,
                search_limits=(semantic_start, semantic_end),
            )
            if corridor_error or boundary_plan is None:
                boundary_range_is_safe = False
            else:
                (
                    start_target,
                    end_target,
                    start_handoff,
                    end_handoff,
                    start_two_sided,
                    end_two_sided,
                ) = boundary_plan
                boundary_range_is_safe = _acoustic_range_is_safe(
                    start_sec=float(acoustic.start_sec),
                    end_sec=float(acoustic.end_sec),
                    required_start_sec=start_target,
                    required_end_sec=end_target,
                    semantic_start_limit_sec=semantic_start,
                    semantic_end_limit_sec=semantic_end,
                    source_end_sec=media_end,
                    diagnostics=dict(acoustic.diagnostics or {}),
                    require_start_handoff=start_handoff,
                    require_end_handoff=end_handoff,
                    require_start_two_sided=start_two_sided,
                    require_end_two_sided=end_two_sided,
                )
            if not boundary_range_is_safe:
                acoustic = _transcript_aligned_result(
                    clip_engine_silence.SilenceVerificationResult(
                        "unavailable",
                        speech_bounds[0],
                        speech_bounds[1],
                        {
                            "stage": "transcript",
                            "reason": "acoustic_refinement_unsafe",
                        },
                    ),
                    speech_bounds=speech_bounds,
                    search_limits=speech_bounds,
                    projection_diagnostics={},
                )
                strict_acoustic = False
                context_aligned = (
                    str(getattr(acoustic, "status", "")) == "context_aligned"
                )
                boundary_range_is_safe = _context_result_range_is_safe(
                    acoustic,
                    source_end_sec=media_end,
                )
        else:
            boundary_range_is_safe = _context_result_range_is_safe(
                acoustic,
                source_end_sec=media_end,
            )
        if not boundary_range_is_safe:
            continue
        clip = dict(raw_clip)
        clip.pop("_quality_scores", None)
        clip.pop("_grounded_topic_evidence_quote", None)
        clip["start"] = round(float(acoustic.start_sec), 3)
        clip["end"] = round(float(acoustic.end_sec), 3)
        # Persistence may validate an optional acoustic tail against prepared
        # media; the normal transcript path keeps the source cue duration.
        clip["_verified_media_duration_sec"] = media_end
        informativeness, topic_relevance, educational_importance = raw_clip[
            "_quality_scores"
        ]
        quality_floor = min(
            informativeness, topic_relevance, educational_importance
        )
        quality_mean = (
            informativeness + topic_relevance + educational_importance
        ) / 3.0
        search_context = dict(clip.get("search_context") or {})
        search_context.update(
            selection_contract_version="quality_silence_v27",
            content_score=topic_relevance,
            quality_floor=quality_floor,
            quality_mean=quality_mean,
            informativeness=informativeness,
            topic_relevance=topic_relevance,
            educational_importance=educational_importance,
            directly_teaches_topic=True,
            substantive=True,
            factually_grounded=True,
            self_contained=True,
            is_standalone=True,
            speech_corridor_verified=True,
            transcript_artifact_key=str(
                transcript.get("artifact_key") or ""
            ).strip(),
            selection_caption_cues=_selected_caption_cues(
                transcript,
                clip,
                boundary_bounds=(clip["start"], clip["end"]),
            ),
            boundary_confidence=float(
                clip.get("boundary_confidence") or 0.0
            ),
            chain_id=str(clip.get("chain_id") or ""),
            chain_position=float(clip.get("chain_position") or 0.0),
            selection_candidate_id=str(
                clip.get("selection_candidate_id") or ""
            ),
            prerequisite_ids=list(clip.get("prerequisite_ids") or []),
            uncertainty=str(clip.get("uncertainty") or "low"),
            intent_role=str(clip.get("intent_role") or "primary").strip().lower(),
            intent_coverage=float(clip.get("intent_coverage", 1.0) or 0.0),
            intent_evidence=list(clip.get("intent_evidence") or []),
            topic_evidence_quote=str(
                clip.get("topic_evidence_quote") or ""
            ).strip(),
            source_rank=0,
            surface_eligible=True,
            boundary_status=("verified" if strict_acoustic else "context_aligned"),
            boundary_diagnostics={
                "method": "energy_silence" if strict_acoustic else "transcript_context",
                "acoustic_verified": strict_acoustic,
                "final_range": [clip["start"], clip["end"]],
                **({"context_aligned": True} if context_aligned else {}),
                "caption": caption,
                **(
                    {"acoustic": dict(acoustic.diagnostics or {})}
                    if strict_acoustic
                    else {"transcript": dict(acoustic.diagnostics or {})}
                ),
                **({"lexical_projection": projection} if projection else {}),
            },
        )
        clip["search_context"] = search_context
        verified.append(clip)
        if limit is not None and len(verified) >= max(0, int(limit)):
            break
    return verified


def _verified_direct_adapter_clip(
    *,
    source_url: str,
    engine_out: dict[str, Any],
    should_cancel: Callable[[], bool] | None,
) -> dict[str, Any] | None:
    clips = _verified_direct_adapter_clips(
        source_url=source_url,
        engine_out=engine_out,
        should_cancel=should_cancel,
        limit=1,
    )
    return clips[0] if clips else None


def _run_direct_clip(
    source_url: str,
    *,
    topic: str,
    language: str,
    should_cancel: Callable[[], bool] | None,
    generation_context: GenerationContext,
) -> dict[str, Any]:
    """Run the one whole-transcript selector without starting audio work."""
    return _run_clip(
        source_url,
        topic=topic,
        language=language,
        should_cancel=should_cancel,
        generation_context=generation_context,
    )


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
    quote_words = [
        _quote_token(match.group(0))
        for match in _QUOTE_WORD_RE.finditer(cleaned_quote)
    ]
    if not 5 <= len(quote_words) <= 40:
        return ""
    spans = _quote_character_spans(text, cleaned_quote)
    if not spans:
        return ""
    generic = {
        "a", "an", "and", "are", "course", "class", "for", "in", "intro",
        "introduction", "lecture", "of", "on", "the", "this", "to", "today",
        "university", "welcome", "we", "will", "you", "your",
    }
    content_words = {word for word in quote_words if len(word) >= 3 and word not in generic}
    if len(content_words) < 2:
        return ""
    return _literal_source_quote(str(text or ""), cleaned_quote, spans[0])


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
        target_clip_duration_sec: int | None = None,
        target_clip_duration_min_sec: int | None = None,
        target_clip_duration_max_sec: int | None = None,
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
            generation_context = GenerationContext(
                "fast",
                generation_id=f"url:{video_id}",
                require_acoustic_boundaries=False,
            )
            engine_out = _run_direct_clip(
                source_url,
                # concept_id is an opaque row id, NOT a topic — it must never
                # steer segmentation (it flows to _persist_ingest for row
                # association only, like ingest_feed).
                topic="",
                language=language,
                should_cancel=should_cancel,
                generation_context=generation_context,
            )
        except _ClipCancellationError:
            raise
        except _ClipUnsupportedURLError as exc:
            raise UnsupportedSourceError(str(exc)) from exc
        except _ClipTranscriptError as exc:
            raise TranscriptionError(str(exc)) from exc
        except _ClipError as exc:
            raise SegmentationError(str(exc)) from exc

        verified = _verified_direct_adapter_clips(
            source_url=source_url,
            engine_out=engine_out,
            should_cancel=should_cancel,
            limit=None,
            prepared_audio=None,
        )
        if not verified:
            raise SegmentationError(
                "no quality clip with complete transcript boundaries could be produced"
            )

        raise_if_cancelled(should_cancel)
        meta = {
            "duration_sec": _direct_adapter_duration_sec(
                engine_out["transcript"], verified
            )
        }

        adapter_result = clip_engine_bridge.synth_adapter_result(video_id, source_url)
        metadata = clip_engine_bridge.to_metadata(video_id, meta, source_url)
        cues = clip_engine_bridge.to_cues(engine_out["transcript"])
        chosen = clip_engine_bridge.to_segment(verified[0], engine_out["transcript"])
        persisted: ReelOutWithAttribution | None = None
        persisted_reels: list[ReelOutWithAttribution] = []
        for index, clip in enumerate(verified):
            raise_if_cancelled(should_cancel)
            segment = (
                chosen
                if index == 0
                else clip_engine_bridge.to_segment(clip, engine_out["transcript"])
            )
            snippet = clip_engine_bridge.window_text(
                engine_out["transcript"], segment.t_start, segment.t_end
            )[:7000]
            stored = self._persist_ingest(
                adapter_result=adapter_result,
                metadata=metadata,
                cues=cues,
                chosen=segment,
                snippet=snippet,
                material_id=material_id,
                concept_id=concept_id,
                clip_window=(segment.t_start, segment.t_end),
                target_max=0,
                clip_title=str(clip.get("title") or "").strip(),
                clip_difficulty=(
                    None
                    if clip.get("difficulty") is None
                    else float(clip["difficulty"])
                ),
                clip_details=clip,
                should_cancel=should_cancel,
            )
            persisted_reels.append(stored)
            if persisted is None:
                persisted = stored

        # Keep the legacy singular reel while the additive inventory exposes
        # every qualifying unit persisted above.
        assert persisted is not None

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
            reels=persisted_reels,
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

        Routes through the same whole-transcript selector and transcript boundary
        refinement as material generation, preserving every qualifying facet.
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
            generation_context = GenerationContext(
                "fast",
                generation_id=f"topic-cut:{video_id}",
                require_acoustic_boundaries=False,
            )
            engine_out = _run_direct_clip(
                source_url,
                topic=(query or ""),
                language=language,
                should_cancel=should_cancel,
                generation_context=generation_context,
            )
        except _ClipCancellationError:
            raise
        except _ClipUnsupportedURLError as exc:
            raise UnsupportedSourceError(str(exc)) from exc
        except _ClipTranscriptError as exc:
            raise TranscriptionError(str(exc)) from exc
        except _ClipError as exc:
            raise SegmentationError(str(exc)) from exc

        kept = _verified_direct_adapter_clips(
            source_url=source_url,
            engine_out=engine_out,
            should_cancel=should_cancel,
            limit=None,
            prepared_audio=None,
            exact_topic=(query or ""),
            embedding_service=self._embedding_service,
        )

        duration = _direct_adapter_duration_sec(
            engine_out["transcript"], kept
        )
        meta: dict[str, Any] = {"duration_sec": duration}

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
                    target_max=0,
                    clip_title=str(clip.get("title") or "").strip(),
                    clip_difficulty=(
                        None
                        if clip.get("difficulty") is None
                        else float(clip["difficulty"])
                    ),
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
        target_clip_duration_sec: int | None = None,
        target_clip_duration_min_sec: int | None = None,
        target_clip_duration_max_sec: int | None = None,
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
                generation_context = GenerationContext(
                    "fast",
                    generation_id=f"feed:{video_id}",
                    require_acoustic_boundaries=False,
                )
                engine_out = _run_clip(
                    url,
                    topic="",
                    language=language,
                    should_cancel=should_cancel,
                    generation_context=generation_context,
                )
                best = _verified_direct_adapter_clip(
                    source_url=url,
                    engine_out=engine_out,
                    should_cancel=should_cancel,
                )
                if best is None:
                    items.append(IngestFeedItem(source_url=url, status="skipped"))
                    continue

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
                    target_max=0,
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
        target_clip_duration_sec: int | None = None,
        target_clip_duration_min_sec: int | None = None,
        target_clip_duration_max_sec: int | None = None,
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

                best = _verified_direct_adapter_clip(
                    source_url=str(v["url"]),
                    engine_out={**engine_out, "clips": eligible_clips},
                    should_cancel=should_cancel,
                )
                if best is None:
                    items.append(IngestSearchItem(
                        platform="yt", source_url=v["url"], status="skipped"
                    ))
                    continue
                best["search_context"] = {
                    **dict(v.get("_search_context") or {}),
                    **dict(best.get("search_context") or {}),
                    "topic_evidence_terms": list(best.get("topic_evidence_terms") or [])[:8],
                }

                persisted, metadata = self._persist_engine_clip(
                    v=v,
                    clip=best,
                    engine_out=engine_out,
                    material_id=resolved_material_id,
                    concept_id=concept_id,
                    target_max=0,
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
        max_persisted_reels: int | None = None,
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
        All selected sources are analyzed concurrently. Completed valid clips
        persist and stream immediately, and the returned list is restored to
        discover order.
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
                topic, limit=discovery_limit, exclude_video_ids=bare_exclusions, level=None,
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
        for source_rank, discovered_video in enumerate(disc["videos"]):
            discovered_video["_retrieval_profile"] = retrieval_profile
            discovered_video["_knowledge_level"] = knowledge_level
            discovered_video["_topic_terms"] = topic_terms
            discovered_video["_literal_topic"] = authoritative_topic
            discovered_video["_search_context"] = _retrieval_search_context(
                requested_topic=authoritative_topic,
                corrected_topic=corrected_topic,
                video=discovered_video,
                query_plan=query_plan,
                creative_commons_only=creative_commons_only,
                source_duration=preferred_video_duration,
            )
            discovered_video["_search_context"]["source_rank"] = source_rank
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
        persistence_cap = (
            None
            if max_persisted_reels is None
            else max(0, int(max_persisted_reels))
        )
        stored_count = 0

        concurrent_video_count = min(3, len(videos), analysis_limit)
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
        audio_preparation_deadlines: dict[str, float] = {}
        acoustic_verification_deadlines: dict[str, float] = {}
        acoustic_phase_timeout_sec = (
            clip_engine_silence.DEFAULT_TIMEOUT_SEC
            if retrieval_profile == "bootstrap"
            else clip_engine_silence.DEEP_PHASE_TIMEOUT_SEC
        )
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
            videos[index].pop("_segment_max_candidates", None)
            if (
                audio_executor is not None
                and analyzed_video_id
                and analyzed_video_id not in audio_preparation_futures
            ):
                audio_preparation_futures[analyzed_video_id] = audio_executor.submit(
                    clip_engine_silence.prepare_audio_source,
                    str(videos[index].get("url") or analyzed_video_id),
                    cancel_check=fetch_should_cancel,
                    language=language,
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

        def audio_preparation_deadline_for(v: dict[str, Any]) -> float:
            source_key = str(v.get("id") or "")
            return audio_preparation_deadlines.setdefault(
                source_key,
                time.monotonic() + acoustic_phase_timeout_sec,
            )

        def prepared_audio_for(v: dict[str, Any]):
            video_id = str(v.get("id") or "").strip()
            if not require_acoustic_boundaries:
                return None
            if video_id in prepared_audio_results:
                return prepared_audio_results[video_id]
            phase_deadline = audio_preparation_deadline_for(v)
            future = audio_preparation_futures.get(video_id)
            if future is None:
                result = clip_engine_silence.AudioPreparationResult(
                    "unavailable",
                    diagnostics={"stage": "resolve", "reason": "audio_not_prepared"},
                )
            else:
                try:
                    remaining = phase_deadline - time.monotonic()
                    if remaining <= 0:
                        raise FutureTimeoutError
                    result = future.result(timeout=remaining)
                except FutureTimeoutError:
                    result = clip_engine_silence.AudioPreparationResult(
                        "unavailable",
                        diagnostics={"stage": "resolve", "reason": "deadline_exceeded"},
                    )
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

        surfaceable_candidate_ids_by_video: dict[str, set[str]] = {}

        def persist_result(
            result: tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]],
            *,
            limit: int | None = None,
        ) -> list[ReelOutWithAttribution]:
            nonlocal stored_count
            v, kept, engine_out = result
            persisted_by_index: dict[int, ReelOutWithAttribution] = {}
            level_deferred_by_index: dict[int, ReelOutWithAttribution] = {}
            level_deferred_stage_by_index: dict[int, int] = {}
            callback_indices: set[int] = set()
            if persistence_cap is not None and stored_count >= persistence_cap:
                return []
            surfaceable_candidate_ids = surfaceable_candidate_ids_by_video.setdefault(
                str(v.get("id") or ""), set()
            )
            surface_limit = None if limit is None else max(0, int(limit))
            candidate_clips = list(kept)
            storage_is_limited = bool(
                persistence_cap is not None
                and stored_count + len(candidate_clips) > persistence_cap
            )
            completion_order_safe = all(
                not any(
                    str(value or "").strip()
                    for value in (clip.get("prerequisite_ids") or [])
                )
                for clip in candidate_clips
            )

            transcript_segments = list(engine_out["transcript"].get("segments") or [])
            transcript_end = max(
                [
                    float(engine_out["transcript"].get("duration") or 0.0),
                    *[
                        float(segment.get("end") or 0.0)
                        for segment in transcript_segments
                        if isinstance(segment, dict)
                    ],
                ]
            )
            prepared_audio = (
                prepared_audio_for(v) if require_acoustic_boundaries else None
            )
            verification_deadline = None
            if require_acoustic_boundaries:
                source_key = str(v.get("id") or "")
                verification_deadline = acoustic_verification_deadlines.setdefault(
                    source_key,
                    time.monotonic() + acoustic_phase_timeout_sec,
                )
            media_end = _prepared_media_end_sec(
                prepared_audio,
                transcript_end_sec=transcript_end,
            )

            def verify_boundary(raw_clip: dict[str, Any]):
                caption, complete_cue_bounds = _transcript_boundary_seed(
                    engine_out["transcript"], raw_clip
                )
                if caption is None:
                    return caption, None, {}, (0.0, 0.0)
                speech_bounds, projection, projection_error = _projected_speech_bounds(
                    engine_out["transcript"],
                    raw_clip,
                    caption,
                    prepared_audio,
                )
                required_start, required_end = speech_bounds
                if projection_error:
                    speech_bounds = complete_cue_bounds
                    required_start, required_end = speech_bounds
                    projection = (
                        {
                            "context_fallback": {
                                "stage": "transcript",
                                "reason": projection_error,
                            }
                        }
                        if projection_error
                        else {}
                    )
                search_start_limit, search_end_limit, corridor_error = _selected_speech_corridor(
                    engine_out["transcript"],
                    raw_clip,
                    caption,
                    source_end_sec=media_end,
                    required_speech_bounds=speech_bounds,
                    projection_diagnostics=projection,
                )
                if corridor_error:
                    speech_bounds = complete_cue_bounds
                    projection = {
                        **projection,
                        "context_fallback": {
                            "stage": "transcript",
                            "reason": f"full_cue_fallback:{corridor_error}",
                        },
                    }
                    transcript_boundary = _transcript_aligned_result(
                        clip_engine_silence.SilenceVerificationResult(
                            "unavailable",
                            speech_bounds[0],
                            speech_bounds[1],
                            {
                                "stage": "transcript",
                                "reason": f"full_cue_fallback:{corridor_error}",
                            },
                        ),
                        speech_bounds=speech_bounds,
                        search_limits=speech_bounds,
                        projection_diagnostics={},
                    )
                    return caption, transcript_boundary, projection, speech_bounds
                boundary_plan = _acoustic_boundary_plan(
                    engine_out["transcript"],
                    raw_clip,
                    projection,
                    speech_bounds=speech_bounds,
                    search_limits=(search_start_limit, search_end_limit),
                )
                if boundary_plan is None:
                    speech_bounds = complete_cue_bounds
                    transcript_boundary = _transcript_aligned_result(
                        clip_engine_silence.SilenceVerificationResult(
                            "unavailable",
                            speech_bounds[0],
                            speech_bounds[1],
                            {
                                "stage": "transcript",
                                "reason": "full_cue_fallback:boundary_plan_invalid",
                            },
                        ),
                        speech_bounds=speech_bounds,
                        search_limits=speech_bounds,
                        projection_diagnostics={},
                    )
                    return caption, transcript_boundary, projection, speech_bounds
                (
                    start_target,
                    end_target,
                    start_handoff,
                    end_handoff,
                    start_two_sided,
                    end_two_sided,
                ) = boundary_plan
                if not require_acoustic_boundaries:
                    transcript_boundary = _transcript_aligned_result(
                        clip_engine_silence.SilenceVerificationResult(
                            "unavailable",
                            start_target,
                            end_target,
                            {
                                "stage": "transcript",
                                "reason": "complete_discourse_boundary",
                            },
                        ),
                        speech_bounds=speech_bounds,
                        search_limits=(search_start_limit, search_end_limit),
                        projection_diagnostics=projection,
                    )
                    return caption, transcript_boundary, projection, speech_bounds
                assert verification_deadline is not None
                remaining_verification_sec = (
                    verification_deadline - time.monotonic()
                )
                if remaining_verification_sec <= 0:
                    transcript_boundary = _transcript_aligned_result(
                        clip_engine_silence.SilenceVerificationResult(
                            "unavailable",
                            required_start,
                            required_end,
                            {
                                "stage": "transcript",
                                "reason": "audio_refinement_deadline_exceeded",
                            },
                        ),
                        speech_bounds=speech_bounds,
                        search_limits=(search_start_limit, search_end_limit),
                        projection_diagnostics=projection,
                    )
                    return caption, transcript_boundary, projection, speech_bounds
                acoustic = clip_engine_silence.verify_acoustic_boundaries(
                    str(v.get("url") or v.get("id") or ""),
                    start_target,
                    end_target,
                    search_start_limit_sec=search_start_limit,
                    search_end_limit_sec=search_end_limit,
                    require_speech_handoff=False,
                    require_start_speech_handoff=start_handoff,
                    require_end_speech_handoff=end_handoff,
                    require_start_two_sided=start_two_sided,
                    require_end_two_sided=end_two_sided,
                    prepared=prepared_audio,
                    timeout_sec=remaining_verification_sec,
                    cancel_check=should_cancel,
                )
                if (
                    acoustic.verified
                    and not _acoustic_range_is_safe(
                        start_sec=float(acoustic.start_sec),
                        end_sec=float(acoustic.end_sec),
                        required_start_sec=start_target,
                        required_end_sec=end_target,
                        semantic_start_limit_sec=search_start_limit,
                        semantic_end_limit_sec=search_end_limit,
                        source_end_sec=media_end,
                        diagnostics=dict(acoustic.diagnostics or {}),
                        require_start_handoff=start_handoff,
                        require_end_handoff=end_handoff,
                        require_start_two_sided=start_two_sided,
                        require_end_two_sided=end_two_sided,
                    )
                ):
                    acoustic = clip_engine_silence.SilenceVerificationResult(
                        "unavailable",
                        speech_bounds[0],
                        speech_bounds[1],
                        {
                            **dict(acoustic.diagnostics or {}),
                            "stage": "semantic_corridor",
                            "reason": "acoustic_refinement_unsafe",
                        },
                    )
                acoustic = _transcript_aligned_result(
                    acoustic,
                    speech_bounds=speech_bounds,
                    search_limits=(search_start_limit, search_end_limit),
                    projection_diagnostics=projection,
                )
                return caption, acoustic, projection, speech_bounds

            def boundary_results():
                if generation_context is None or not candidate_clips:
                    for index, clip in enumerate(candidate_clips):
                        yield index, (
                            _supadata_boundary_diagnostics(
                                engine_out["transcript"], clip
                            ),
                            None,
                            {},
                            (0.0, 0.0),
                        )
                    return
                if not require_acoustic_boundaries:
                    for index, clip in enumerate(candidate_clips):
                        yield index, verify_boundary(clip)
                    return
                boundary_executor = ThreadPoolExecutor(
                    max_workers=min(3, len(candidate_clips)),
                    thread_name_prefix="clip-boundary-verify",
                )
                futures = {
                    boundary_executor.submit(verify_boundary, clip): index
                    for index, clip in enumerate(candidate_clips)
                }
                yielded_indices: set[int] = set()
                try:
                    if verification_deadline is None:
                        ordered_futures = (
                            as_completed(futures)
                            if completion_order_safe and not storage_is_limited
                            else iter(futures)
                        )
                        for future in ordered_futures:
                            index = futures[future]
                            yielded_indices.add(index)
                            yield index, future.result()
                    elif completion_order_safe and not storage_is_limited:
                        remaining = max(
                            0.0, verification_deadline - time.monotonic()
                        )
                        try:
                            for future in as_completed(futures, timeout=remaining):
                                index = futures[future]
                                yielded_indices.add(index)
                                yield index, future.result()
                        except FutureTimeoutError:
                            pass
                    else:
                        for future, index in futures.items():
                            remaining = verification_deadline - time.monotonic()
                            if remaining <= 0:
                                break
                            try:
                                result = future.result(timeout=remaining)
                            except FutureTimeoutError:
                                break
                            yielded_indices.add(index)
                            yield index, result

                    if verification_deadline is not None:
                        unfinished = [
                            (future, index)
                            for future, index in futures.items()
                            if index not in yielded_indices
                        ]
                        for future, _index in unfinished:
                            future.cancel()
                        for _future, index in unfinished:
                            clip = candidate_clips[index]
                            caption, speech_bounds = _transcript_boundary_seed(
                                engine_out["transcript"], clip
                            )
                            timeout_result = _transcript_aligned_result(
                                clip_engine_silence.SilenceVerificationResult(
                                    "unavailable",
                                    speech_bounds[0],
                                    speech_bounds[1],
                                    {
                                        "stage": "transcript",
                                        "reason": "audio_refinement_deadline_exceeded",
                                    },
                                ),
                                speech_bounds=speech_bounds,
                                search_limits=speech_bounds,
                                projection_diagnostics={},
                            )
                            yield index, (
                                caption,
                                timeout_result,
                                {},
                                speech_bounds,
                            )
                finally:
                    boundary_executor.shutdown(wait=False, cancel_futures=True)

            for candidate_index, (
                caption_diagnostics,
                acoustic,
                projection_diagnostics,
                speech_bounds,
            ) in boundary_results():
                raise_if_cancelled(should_cancel)
                if persistence_cap is not None and stored_count >= persistence_cap:
                    break
                raw_clip = candidate_clips[candidate_index]
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
                boundary_verified_for_storage = False
                permanently_rejected = False
                if not prerequisites_ready:
                    search_context["surface_reason"] = "prerequisite_not_surfaceable"

                if generation_context is not None and caption_diagnostics is None:
                    surface_eligible = False
                    permanently_rejected = True
                    search_context["boundary_status"] = "unavailable"
                    search_context["surface_reason"] = "supadata_boundary_unavailable"
                    record_boundary_unavailable("supadata_boundary_unavailable")
                elif semantic_eligible and generation_context is not None:
                    if acoustic is not None:
                        assert acoustic is not None
                        acoustic_diagnostics = dict(acoustic.diagnostics or {})
                        semantic_start, semantic_end, corridor_error = (
                            _selected_speech_corridor(
                                engine_out["transcript"],
                                clip,
                                caption_diagnostics,
                                source_end_sec=media_end,
                                required_speech_bounds=speech_bounds,
                                projection_diagnostics=projection_diagnostics,
                            )
                        )
                        boundary_plan = _acoustic_boundary_plan(
                            engine_out["transcript"],
                            clip,
                            projection_diagnostics,
                            speech_bounds=speech_bounds,
                            search_limits=(semantic_start, semantic_end),
                        )
                        if boundary_plan is None:
                            start_target, end_target = speech_bounds
                            start_handoff = end_handoff = False
                            start_two_sided = end_two_sided = False
                        else:
                            (
                                start_target,
                                end_target,
                                start_handoff,
                                end_handoff,
                                start_two_sided,
                                end_two_sided,
                            ) = boundary_plan
                        strict_acoustic = bool(getattr(acoustic, "verified", False))
                        context_aligned = (
                            str(getattr(acoustic, "status", ""))
                            == "context_aligned"
                        )
                        acoustic_range_is_safe = bool(
                            strict_acoustic
                            and corridor_error is None
                            and boundary_plan is not None
                            and _acoustic_range_is_safe(
                                start_sec=float(acoustic.start_sec),
                                end_sec=float(acoustic.end_sec),
                                required_start_sec=start_target,
                                required_end_sec=end_target,
                                semantic_start_limit_sec=semantic_start,
                                semantic_end_limit_sec=semantic_end,
                                source_end_sec=media_end,
                                diagnostics=acoustic_diagnostics,
                                require_start_handoff=start_handoff,
                                require_end_handoff=end_handoff,
                                require_start_two_sided=start_two_sided,
                                require_end_two_sided=end_two_sided,
                            )
                        )
                        context_range_is_safe = _context_result_range_is_safe(
                            acoustic,
                            source_end_sec=media_end,
                        )
                        boundary_range_is_safe = (
                            acoustic_range_is_safe or context_range_is_safe
                        )
                        boundary_diagnostics = {
                            "method": (
                                "energy_silence"
                                if acoustic_range_is_safe
                                else "transcript_context"
                            ),
                            "acoustic_verified": acoustic_range_is_safe,
                            "final_range": [
                                round(float(acoustic.start_sec), 3),
                                round(float(acoustic.end_sec), 3),
                            ],
                            **(
                                {"context_aligned": True}
                                if context_range_is_safe
                                else {}
                            ),
                            "caption": caption_diagnostics,
                            **(
                                {"acoustic": acoustic_diagnostics}
                                if acoustic_range_is_safe
                                else {"transcript": acoustic_diagnostics}
                            ),
                        }
                        if projection_diagnostics:
                            boundary_diagnostics["lexical_projection"] = (
                                projection_diagnostics
                            )
                        search_context["boundary_diagnostics"] = boundary_diagnostics
                        if not boundary_range_is_safe:
                            surface_eligible = False
                            permanently_rejected = True
                            search_context["boundary_status"] = "unavailable"
                            search_context["surface_reason"] = str(
                                acoustic_diagnostics.get("reason")
                                or "acoustic_boundary_outside_source_or_required_speech"
                            )
                            failure_stage = str(
                                acoustic_diagnostics.get("stage") or "verify"
                            ).strip()
                            record_boundary_unavailable(
                                f"{failure_stage}:{search_context['surface_reason']}"
                            )
                        else:
                            boundary_verified_for_storage = acoustic_range_is_safe
                            clip["start"] = round(float(acoustic.start_sec), 3)
                            clip["end"] = round(float(acoustic.end_sec), 3)
                            search_context["selection_caption_cues"] = (
                                _selected_caption_cues(
                                    engine_out["transcript"],
                                    clip,
                                    boundary_bounds=(clip["start"], clip["end"]),
                                )
                            )
                            search_context["boundary_status"] = (
                                "verified"
                                if acoustic_range_is_safe
                                else "context_aligned"
                            )
                            search_context["speech_corridor_verified"] = True
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
                if permanently_rejected:
                    if generation_context is not None:
                        # ``deferred_clips`` remains the backward-compatible
                        # aggregate for every non-surfaceable candidate. The
                        # specific counter distinguishes terminal boundary
                        # rejection from reusable difficulty deferral.
                        generation_context.increment_counter("deferred_clips")
                        generation_context.increment_counter(
                            "permanently_rejected_clips"
                        )
                    continue
                created_new_reel = True

                def record_persistence_result(created: bool) -> None:
                    nonlocal created_new_reel
                    created_new_reel = created

                reel, _ = self._persist_engine_clip(
                    v=v,
                    clip=clip,
                    engine_out=engine_out,
                    material_id=material_id,
                    concept_id=concept_id,
                    target_max=0,
                    generation_id=generation_id,
                    on_persistence_result=record_persistence_result,
                    should_cancel=should_cancel,
                )
                if created_new_reel:
                    stored_count += 1
                if generation_context is not None:
                    if created_new_reel:
                        generation_context.increment_counter("stored_clips")
                    if boundary_verified_for_storage:
                        generation_context.increment_counter("verified_clips")
                if not surface_eligible:
                    if generation_context is not None:
                        generation_context.increment_counter("deferred_clips")
                        if search_context.get("surface_reason") == "level_mismatch":
                            generation_context.increment_counter(
                                "level_deferred_clips"
                            )
                    if search_context.get("surface_reason") == "level_mismatch":
                        level_deferred_by_index[candidate_index] = reel
                        try:
                            difficulty = max(
                                0.0, min(1.0, float(clip.get("difficulty", 0.5)))
                            )
                        except (TypeError, ValueError, OverflowError):
                            difficulty = 0.5
                        level_deferred_stage_by_index[candidate_index] = (
                            0 if difficulty < 0.34 else 1 if difficulty < 0.67 else 2
                        )
                    continue
                candidate_id = str(clip.get("selection_candidate_id") or "").strip()
                if candidate_id:
                    surfaceable_candidate_ids.add(candidate_id)
                persisted_by_index[candidate_index] = reel
                definitely_selected = (
                    surface_limit is None
                    or candidate_index < surface_limit
                    or (
                        not completion_order_safe
                        and len(persisted_by_index) <= surface_limit
                    )
                )
                if on_reel_created is not None and definitely_selected:
                    on_reel_created(reel)
                    callback_indices.add(candidate_index)

            selected_reels = persisted_by_index
            selected_indices = sorted(selected_reels)
            if not selected_indices and level_deferred_by_index:
                # Difficulty is an ordering preference, not a reason to make a
                # valid source appear empty. Stream the nearest already-ranked
                # level only when this source has no current-level candidate;
                # the final inventory remains authoritative across sources.
                target_stage = {
                    "beginner": 0,
                    "intermediate": 1,
                    "advanced": 2,
                }.get(str(v.get("_knowledge_level") or "").strip().lower(), 0)
                nearest_stage = min(
                    set(level_deferred_stage_by_index.values()),
                    key=lambda stage: (abs(stage - target_stage), stage),
                )
                selected_reels = {
                    candidate_index: reel
                    for candidate_index, reel in level_deferred_by_index.items()
                    if level_deferred_stage_by_index[candidate_index] == nearest_stage
                }
                selected_indices = sorted(selected_reels)
            if surface_limit is not None:
                selected_indices = selected_indices[:surface_limit]
            if on_reel_created is not None:
                for candidate_index in selected_indices:
                    if candidate_index not in callback_indices:
                        on_reel_created(selected_reels[candidate_index])
            if generation_context is not None:
                generation_context.increment_counter(
                    "persisted_clips", len(selected_indices)
                )
            return [selected_reels[index] for index in selected_indices]

        reels_by_video: dict[int, list[ReelOutWithAttribution]] = {}
        completed_results: dict[
            int,
            tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]],
        ] = {}
        straggler_deadline: float | None = None
        initial_resolved: set[int] = set()
        provider_errors: list[_ClipProviderError] = []
        initial_count = concurrent_video_count
        persisted_count = 0
        bootstrap_attempted_indices: set[int] = set()

        def persist_bootstrap_sources() -> None:
            """Persist at most one best clip per completed source before reuse.

            Bootstrap waits for the two initial analyses, ranks their strongest
            clips together, and reserves the second slot for the second source.
            Later sources backfill empty/failed initial results.
            """
            nonlocal persisted_count
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
        pending = {
            submit_video(index): (index, videos[index])
            for index in range(concurrent_video_count)
        }
        all_futures = list(pending)
        next_index = concurrent_video_count
        try:
            while pending:
                raise_if_cancelled(should_cancel)
                active_deadline = min(
                    deadline,
                    straggler_deadline if straggler_deadline is not None else deadline,
                )
                remaining = max(0.0, active_deadline - time.monotonic())
                done, _ = wait(
                    pending,
                    timeout=min(0.05, remaining),
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    if time.monotonic() < active_deadline:
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
                                "post-first-clip straggler grace exceeded"
                                if straggler_deadline is not None
                                and straggler_deadline < deadline
                                else "shared clip fetch deadline exceeded"
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
                        stored_before = stored_count
                        persisted = persist_result(
                            result,
                            limit=max(0, inventory_cap - persisted_count),
                        )
                        reels_by_video[index] = persisted
                        persisted_count += len(persisted)
                        if (
                            (persisted or stored_count > stored_before)
                            and straggler_deadline is None
                            and retrieval_profile == "deep"
                        ):
                            straggler_deadline = min(
                                deadline,
                                time.monotonic()
                                + max(0.0, INGEST_TOPIC_STRAGGLER_GRACE_SEC),
                            )

                if retrieval_profile == "bootstrap":
                    persist_bootstrap_sources()
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

        # Progressive callbacks reflect availability; restore discover order in
        # the final result for deterministic downstream inventory.
        reels = [
            reel
            for index in range(len(videos))
            for reel in reels_by_video.get(index, [])
        ]
        if not completed_results and provider_errors:
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
        if generation_context is not None and record_transcript_usage:
            if trusted_transcript:
                if engine_out.get("_transcript_usage_recorded") is not True:
                    generation_context.increment_counter("usable_transcripts")
            elif query_plan is not None or v.get("_topic_terms"):
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
        selector_topic_contracts = [
            any(
                key in clip
                for key in (
                    "directly_teaches_topic",
                    "substantive",
                    "factually_grounded",
                    "topic_evidence_quote",
                )
            )
            for clip in raw_clips
        ]
        grounded_selector_quotes = [
            (
                _grounded_topic_evidence_quote(
                    transcript_texts[index], clip.get("topic_evidence_quote")
                )
                if selector_topic_contracts[index]
                else ""
            )
            for index, clip in enumerate(raw_clips)
        ]
        semantic_scores: list[float | None] = [None] * len(raw_clips)
        if (
            topic_terms
            and any(not contract for contract in selector_topic_contracts)
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
            educational_importance = unit(
                clip.get("educational_importance"), 0.0
            )
            if (
                min(
                    informativeness,
                    topic_relevance,
                    educational_importance,
                )
                < 0.75
                or clip.get("self_contained") is not True
                or clip.get("is_standalone") is not True
            ):
                if generation_context is not None:
                    generation_context.increment_counter("topic_rejections")
                continue
            selector_topic_contract = selector_topic_contracts[index]
            grounded_evidence_quote = ""
            if selector_topic_contract:
                grounded_evidence_quote = grounded_selector_quotes[index]
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
                "self_contained": bool(clip.get("self_contained")),
                "topic_evidence_quote": grounded_evidence_quote,
                "transcript_artifact_key": str(
                    engine_out["transcript"].get("artifact_key") or ""
                ).strip(),
                "selection_caption_cues": _selected_caption_cues(
                    engine_out["transcript"], clip
                ),
                "surface_eligible": True,
                "deferred_level": not difficulty_matches_knowledge_level(
                    difficulty,
                    str(v.get("_knowledge_level") or ""),
                ),
            }
            if has_selector_metadata:
                importance = educational_importance
                boundary_confidence = unit(clip.get("boundary_confidence"), 0.5)
                uncertainty = str(clip.get("uncertainty") or "low").strip().lower()
                quality_floor = min(
                    informativeness, topic_relevance, importance
                )
                quality_mean = (
                    topic_relevance + importance + informativeness
                ) / 3.0
                content_score = topic_relevance
                clip["score"] = content_score
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
                    selection_contract_version="quality_silence_v27",
                    content_score=content_score,
                    quality_floor=quality_floor,
                    quality_mean=quality_mean,
                    informativeness=informativeness,
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
                    intent_role=str(
                        clip.get("intent_role") or "primary"
                    ).strip().lower(),
                    intent_coverage=unit(clip.get("intent_coverage"), 1.0),
                    intent_evidence=list(clip.get("intent_evidence") or []),
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
        target_difficulty_stage = (
            0 if level_target < 0.34 else 1 if level_target < 0.67 else 2
        )

        def difficulty_stage(node_id: str) -> int:
            difficulty = unit(nodes[node_id].get("difficulty"), 0.5)
            return 0 if difficulty < 0.34 else 1 if difficulty < 0.67 else 2

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
                    int(
                        not bool(
                            nodes[node_id]
                            .get("search_context", {})
                            .get("deferred_level")
                        )
                    ),
                    -abs(difficulty_stage(node_id) - target_difficulty_stage),
                    -difficulty_stage(node_id),
                    int(
                        str(nodes[node_id].get("intent_role") or "primary")
                        .strip()
                        .lower()
                        == "primary"
                    ),
                    unit(nodes[node_id].get("intent_coverage"), 1.0),
                    -unit(nodes[node_id].get("difficulty"), 0.5),
                    float(
                        nodes[node_id]
                        .get("search_context", {})
                        .get("topic_relevance", 0.0)
                    ),
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
        on_persistence_result: Callable[[bool], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> tuple[ReelOutWithAttribution, IngestMetadata]:
        """
        Build the persist inputs for ONE engine clip of a discovered video `v`
        (source metadata from the discover dict, transcript fallbacks from
        `engine_out`) and persist it. Shared by `ingest_search` and
        `ingest_topic`.

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
            on_persistence_result=on_persistence_result,
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
        on_persistence_result: Callable[[bool], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> ReelOutWithAttribution:
        raise_if_cancelled(should_cancel)
        clip_start = round(float(clip_window[0]), 3)
        clip_end = round(float(clip_window[1]), 3)
        clip_duration = clip_end - clip_start
        source_end_candidates = [
            float(cue.end)
            for cue in cues
            if math.isfinite(float(cue.end))
        ]
        if metadata.duration_sec is not None:
            try:
                source_end_candidates.append(float(metadata.duration_sec))
            except (TypeError, ValueError, OverflowError):
                pass
        source_end = max(source_end_candidates, default=clip_end)
        if (
            not math.isfinite(clip_start)
            or not math.isfinite(clip_end)
            or clip_start < 0.0
            or clip_end <= clip_start
            or clip_end > source_end + 1e-3
        ):
            raise SegmentationError("Clip timestamps must be ordered and within the source.")
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
        selection_snapshot = _selection_snapshot_payload(
            selection_context,
            clip_start=clip_start,
            clip_end=clip_end,
            selected_cue_ids=selected_cue_ids,
        )
        if selection_snapshot is not None:
            snippet, selection_snapshot_captions = selection_snapshot
        else:
            selection_snapshot_captions = None
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
            candidate_lookup = {
                "material_id": effective_material_id,
                "concept_id": effective_concept_id,
                "video_id": video_id,
                "generation_id": generation_id,
                "selection_candidate_id": str(
                    selection_context.get("selection_candidate_id") or ""
                ),
            }
            existing_candidate = load_reel_by_selection_candidate(
                conn, **candidate_lookup
            )
            if existing_candidate:
                created_new_reel = False
                retained_candidate: dict[str, Any] | None = None
                for _attempt in range(3):
                    reel_id = str(existing_candidate["id"])
                    existing_context: dict[str, Any] = {}
                    existing_context_json = str(
                        existing_candidate.get("search_context_json") or "{}"
                    )
                    try:
                        parsed_context = json.loads(existing_context_json)
                        if isinstance(parsed_context, dict):
                            existing_context = parsed_context
                    except (TypeError, json.JSONDecodeError):
                        pass
                    existing_surfaceable = (
                        existing_context.get("surface_eligible") is True
                    )
                    incoming_surfaceable = (
                        selection_context.get("surface_eligible") is True
                    )
                    existing_grade = _boundary_evidence_grade(
                        existing_context,
                        t_start=existing_candidate.get("t_start"),
                        t_end=existing_candidate.get("t_end"),
                    )
                    incoming_grade = _boundary_evidence_grade(
                        selection_context,
                        t_start=clip_start,
                        t_end=clip_end,
                    )
                    should_update = bool(
                        incoming_grade > existing_grade
                        or (
                            incoming_grade == existing_grade
                            and (incoming_surfaceable or not existing_surfaceable)
                        )
                    )
                    if not should_update:
                        retained_candidate = existing_candidate
                        break
                    if update_reel_boundary_state(
                        conn,
                        reel_id=reel_id,
                        video_url=video_url,
                        t_start=clip_start,
                        t_end=clip_end,
                        transcript_snippet=snippet,
                        selected_cue_ids=selected_cue_ids,
                        search_context=selection_context,
                        expected_search_context_json=existing_context_json,
                    ):
                        break
                    latest_candidate = load_reel_by_selection_candidate(
                        conn, **candidate_lookup
                    )
                    if latest_candidate is None:
                        raise DatabaseIntegrityError(
                            "Selection candidate disappeared during boundary promotion."
                        )
                    existing_candidate = latest_candidate
                else:
                    retained_candidate = existing_candidate
                if retained_candidate is not None:
                    reel_id = str(retained_candidate["id"])
                    clip_start = float(retained_candidate["t_start"])
                    clip_end = float(retained_candidate["t_end"])
                    video_url = str(retained_candidate.get("video_url") or video_url)
                    snippet = str(
                        retained_candidate.get("transcript_snippet") or snippet
                    )
                    try:
                        retained_ids = json.loads(
                            str(
                                retained_candidate.get("selected_cue_ids_json")
                                or "[]"
                            )
                        )
                    except (TypeError, json.JSONDecodeError):
                        retained_ids = []
                    if isinstance(retained_ids, list):
                        selected_cue_ids = [
                            str(value) for value in retained_ids if str(value).strip()
                        ]
                    try:
                        retained_context = json.loads(
                            str(
                                retained_candidate.get("search_context_json") or "{}"
                            )
                        )
                    except (TypeError, json.JSONDecodeError):
                        retained_context = {}
                    if isinstance(retained_context, dict):
                        selection_context = retained_context
                    retained_snapshot = _selection_snapshot_payload(
                        selection_context,
                        clip_start=clip_start,
                        clip_end=clip_end,
                        selected_cue_ids=selected_cue_ids,
                    )
                    if retained_snapshot is not None:
                        snippet, selection_snapshot_captions = retained_snapshot
                    else:
                        selection_snapshot_captions = None
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
                created_new_reel = bool(inserted)

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
        captions = selection_snapshot_captions or []
        if selection_snapshot_captions is None:
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

        persisted_reel = ReelOutWithAttribution(
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
            selected_cue_ids=list(selected_cue_ids),
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
            selection_quality_floor=(
                float(selection_context["quality_floor"])
                if selection_context.get("quality_floor") is not None
                else None
            ),
            selection_quality_mean=(
                float(selection_context["quality_mean"])
                if selection_context.get("quality_mean") is not None
                else None
            ),
            selection_topic_relevance=(
                float(selection_context["topic_relevance"])
                if selection_context.get("topic_relevance") is not None
                else None
            ),
            selection_source_rank=max(
                0, int(selection_context.get("source_rank") or 0)
            ),
            selection_intent_role=str(
                selection_context.get("intent_role") or "primary"
            ).strip().lower(),
            selection_intent_coverage=max(
                0.0,
                min(1.0, float(selection_context.get("intent_coverage", 1.0))),
            ),
            source_attribution=attribution,
        )
        if on_persistence_result is not None:
            on_persistence_result(created_new_reel)
        return persisted_reel


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
