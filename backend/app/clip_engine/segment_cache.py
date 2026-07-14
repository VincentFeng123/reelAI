"""Persistent cache for fully validated Gemini segmentation results."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from ... import config as pipeline_config
from ..db import dumps_json, fetch_one, get_conn, now_iso, upsert


logger = logging.getLogger(__name__)

SEGMENT_CACHE_VERSION = 5
SEGMENT_CACHE_TTL_SEC = 30 * 24 * 60 * 60
SELECTION_CONTRACT_VERSION = "quality_silence_v4"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _relevant_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    fine_snap = settings.get("segment_fine_snap")
    return {
        "fine_snap": (
            bool(pipeline_config.SEGMENT_FINE_SNAP)
            if fine_snap is None
            else bool(fine_snap)
        ),
        "language": " ".join(str(settings.get("language") or "en").split()).lower(),
    }


@lru_cache(maxsize=1)
def _segmenter_source_signature() -> str | None:
    """Invalidate cache entries whenever the active prompt or validators change."""
    try:
        from ...pipeline import gemini_segment

        source_path = Path(str(gemini_segment.__file__ or ""))
        return hashlib.sha256(source_path.read_bytes()).hexdigest()
    except (OSError, TypeError, ValueError):
        return None


def cache_enabled() -> bool:
    """Keep evaluation side effects and source-less releases out of the cache."""
    return (
        pipeline_config.SEGMENT_ROUTING_MODE != "shadow"
        and _segmenter_source_signature() is not None
    )


def segment_cache_key(
    *,
    video_id: str,
    topic: str,
    transcript: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> str:
    """Hash every input that can change the validated public clip list."""
    transcript_payload = {
        "artifact_key": str(transcript.get("artifact_key") or ""),
        "segments": transcript.get("segments") or [],
        "words": transcript.get("words") or [],
        "duration": transcript.get("duration") or transcript.get("duration_sec") or 0.0,
    }
    payload = {
        "version": SEGMENT_CACHE_VERSION,
        "selection_contract_version": SELECTION_CONTRACT_VERSION,
        "video_id": str(video_id or "").strip(),
        "topic": " ".join(str(topic or "").split()),
        "transcript_sha256": hashlib.sha256(
            _canonical_json(transcript_payload).encode("utf-8")
        ).hexdigest(),
        "flash_model": pipeline_config.SEGMENT_FLASH_MODEL,
        "segmenter_source_sha256": _segmenter_source_signature(),
        "settings": _relevant_settings(settings),
    }
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return (
        f"clip-segmentation:{SELECTION_CONTRACT_VERSION}:"
        f"v{SEGMENT_CACHE_VERSION}:{digest}"
    )


def _age_seconds(created_at: object) -> float:
    try:
        parsed = datetime.fromisoformat(str(created_at or "").replace("Z", "+00:00"))
    except ValueError:
        return float("inf")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds())


def _transcript_bounds(transcript: Mapping[str, Any]) -> tuple[float, float] | None:
    segments = transcript.get("segments")
    if not isinstance(segments, list) or not segments:
        return None
    ends: list[float] = []
    for segment in segments:
        if not isinstance(segment, Mapping):
            return None
        try:
            start = float(segment.get("start"))
            end = float(segment.get("end"))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            return None
        ends.append(end)
    # A finalized clip may include bounded silence before the first spoken cue.
    return 0.0, max(ends)


def _valid_assessment(value: object) -> bool:
    if value is None:
        return True
    if not isinstance(value, Mapping):
        return False
    options = value.get("options")
    correct_index = value.get("correct_index")
    return bool(
        str(value.get("prompt") or "").strip()
        and isinstance(options, list)
        and len(options) == 4
        and all(str(option or "").strip() for option in options)
        and len({" ".join(str(option).casefold().split()) for option in options}) == 4
        and isinstance(correct_index, int)
        and not isinstance(correct_index, bool)
        and 0 <= correct_index <= 3
        and str(value.get("explanation") or "").strip()
    )


def _valid_clips(
    value: object,
    *,
    transcript: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> list[dict[str, Any]] | None:
    if not isinstance(value, list):
        return None
    bounds = _transcript_bounds(transcript)
    if bounds is None:
        return None
    transcript_start, transcript_end = bounds
    clips: list[dict[str, Any]] = []
    previous_order: tuple[int, float, float, float, float, float, str] | None = None
    for index, raw in enumerate(value, start=1):
        if not isinstance(raw, dict):
            return None
        score_values = (
            raw.get("informativeness"),
            raw.get("topic_relevance"),
            raw.get("educational_importance"),
            raw.get("difficulty"),
        )
        if any(
            isinstance(score, bool) or not isinstance(score, (int, float))
            for score in score_values
        ):
            return None
        try:
            start = float(raw.get("start"))
            end = float(raw.get("end"))
            required_start = float(raw.get("required_first_speech_sec", start))
            required_end = float(raw.get("required_last_speech_sec", end))
            informativeness, topic_relevance, educational_importance, difficulty = (
                float(score) for score in score_values
            )
        except (TypeError, ValueError):
            return None
        uncertainty = str(raw.get("uncertainty") or "low").strip().lower()
        uncertainty_reasons = raw.get("uncertainty_reasons") or []
        evidence_quote = " ".join(str(raw.get("topic_evidence_quote") or "").split())
        evidence_words = re.findall(r"[\w+#'-]+", evidence_quote.casefold())
        window_parts: list[str] = []
        for segment in transcript.get("segments") or []:
            if not isinstance(segment, Mapping):
                continue
            try:
                segment_start = float(segment.get("start") or 0.0)
                segment_end = float(segment.get("end") or segment_start)
            except (TypeError, ValueError, OverflowError):
                continue
            if segment_end >= start and segment_start <= end:
                window_parts.append(str(segment.get("text") or ""))
        window_text = " ".join(window_parts)
        window_words = re.findall(r"[\w+#'-]+", window_text.casefold())
        evidence_width = len(evidence_words)
        evidence_grounded = bool(
            5 <= evidence_width <= 40
            and any(
                window_words[offset : offset + evidence_width] == evidence_words
                for offset in range(len(window_words) - evidence_width + 1)
            )
        )
        difficulty_stage = 0 if difficulty < 0.34 else 1 if difficulty < 0.67 else 2
        quality_floor = min(informativeness, topic_relevance, educational_importance)
        quality_mean = (
            informativeness + topic_relevance + educational_importance
        ) / 3.0
        order_key = (
            difficulty_stage,
            -quality_floor,
            -quality_mean,
            -topic_relevance,
            start,
            end,
            str(raw.get("selection_candidate_id") or ""),
        )
        if (
            not all(
                math.isfinite(number)
                for number in (
                    start,
                    end,
                    required_start,
                    required_end,
                    informativeness,
                    topic_relevance,
                    educational_importance,
                    difficulty,
                )
            )
            or start < transcript_start - 0.001
            or end > transcript_end + 0.001
            or (previous_order is not None and order_key < previous_order)
            or end <= start
            or required_start < start
            or required_end > end
            or required_end <= required_start
            or not 0 <= informativeness <= 1
            or not 0 <= topic_relevance <= 1
            or not 0 <= educational_importance <= 1
            or topic_relevance < 0.75
            or not 0 <= difficulty <= 1
            or raw.get("self_contained") is not True
            or raw.get("is_standalone") is not True
            or raw.get("directly_teaches_topic") is not True
            or raw.get("substantive") is not True
            or raw.get("factually_grounded") is not True
            or not evidence_grounded
            or raw.get("kind") != "educational"
            or uncertainty not in {"low", "medium"}
            or not isinstance(uncertainty_reasons, list)
            or any(
                not str(raw.get(field) or "").strip()
                for field in ("title", "learning_objective", "facet", "reason")
            )
            or any(field not in raw for field in ("summary", "takeaways", "match_reason", "assessment"))
            or raw.get("sequence_index") != index
            or not isinstance(raw.get("takeaways"), list)
            or not _valid_assessment(raw.get("assessment"))
        ):
            return None
        if any(
            min(end, float(prior["end"])) - max(start, float(prior["start"]))
            >= 0.8 * min(end - start, float(prior["end"]) - float(prior["start"]))
            for prior in clips
        ):
            return None
        clips.append(dict(raw))
        previous_order = order_key
    return clips


def load_segment_result(
    cache_key: str,
    *,
    video_id: str,
    transcript: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], str] | None:
    try:
        with get_conn() as conn:
            if fetch_one(
                conn,
                "SELECT video_id FROM blocked_video_tombstones WHERE video_id = ?",
                (video_id,),
            ):
                return None
            row = fetch_one(
                conn,
                "SELECT response_json, created_at FROM llm_cache WHERE cache_key = ?",
                (cache_key,),
            )
    except Exception as exc:
        logger.debug("Segmentation cache read unavailable: %s", exc)
        return None
    if not row or _age_seconds(row.get("created_at")) >= SEGMENT_CACHE_TTL_SEC:
        return None
    try:
        payload = json.loads(str(row.get("response_json") or "{}"))
    except (TypeError, json.JSONDecodeError):
        return None
    if (
        not isinstance(payload, dict)
        or payload.get("version") != SEGMENT_CACHE_VERSION
        or payload.get("selection_contract_version") != SELECTION_CONTRACT_VERSION
        or payload.get("video_id") != video_id
    ):
        return None
    clips = _valid_clips(payload.get("clips"), transcript=transcript, settings=settings)
    if clips is None:
        return None
    return clips, str(payload.get("notes") or "")


def store_segment_result(
    cache_key: str,
    clips: list[dict[str, Any]],
    notes: str,
    *,
    video_id: str,
    transcript: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> None:
    validated = _valid_clips(clips, transcript=transcript, settings=settings)
    if validated is None:
        return
    try:
        with get_conn(transactional=True) as conn:
            upsert(
                conn,
                "llm_cache",
                {
                    "cache_key": cache_key,
                    "response_json": dumps_json(
                        {
                            "version": SEGMENT_CACHE_VERSION,
                            "selection_contract_version": SELECTION_CONTRACT_VERSION,
                            "video_id": video_id,
                            "clips": validated,
                            "notes": str(notes or ""),
                        }
                    ),
                    "created_at": now_iso(),
                },
                pk="cache_key",
            )
    except Exception as exc:
        logger.debug("Segmentation cache write unavailable: %s", exc)
