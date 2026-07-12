"""Gemini educational segmentation over validated timestamped cue boundaries."""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from typing import Callable, Optional

from pydantic import BaseModel, Field

from .. import config
from .....pipeline.gemini_segment import segment_clips

ProgressCb = Optional[Callable[[float, str], None]]
_TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)


class _Assessment(BaseModel):
    prompt: str = ""
    options: list[str] = Field(default_factory=list)
    correct_index: int | float | bool | str | None = None
    explanation: str = ""
    cue_ids: list[str] = Field(default_factory=list)


class _Topic(BaseModel):
    title: str
    start_line: int = Field(strict=True)
    end_line: int = Field(strict=True)
    start_quote: str
    end_quote: str
    reason: str = ""
    facet: str = "other"
    kind: Optional[str] = None
    informativeness: Optional[float] = None
    topic_relevance: Optional[float] = None
    self_contained: Optional[bool] = None
    difficulty: float = 0.5
    summary: str = ""
    summary_cue_ids: list[str] = Field(default_factory=list)
    takeaways: list[str] = Field(default_factory=list)
    takeaway_cue_ids: list[list[str]] = Field(default_factory=list)
    match_reason: str = ""
    match_reason_cue_ids: list[str] = Field(default_factory=list)
    assessment: _Assessment | None = None


class _Plan(BaseModel):
    topics: list[_Topic]


_ACCEPT_KINDS = {"content", "educational"}


def _norm_informativeness(value: float) -> float:
    value = float(value)
    if value > 10.0:
        value /= 100.0
    elif value > 1.0:
        value /= 10.0
    return max(0.0, min(1.0, value))


def _norm_optional_confidence(value: Optional[float]) -> Optional[float]:
    return None if value is None else _norm_informativeness(value)


def _mmss(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


def _tokens(value: object) -> list[str]:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return [match.group(0) for match in _TOKEN_RE.finditer(normalized)]


def _contains_tokens(text: str, quote: str) -> bool:
    source = _tokens(text)
    target = _tokens(quote)
    if not source or not target or len(target) > len(source):
        return False
    width = len(target)
    return any(source[index:index + width] == target for index in range(len(source) - width + 1))


def _cue_id(segment: dict, index: int) -> str:
    return str(segment.get("cue_id") or f"cue-{index}").strip()


def _with_cue_ids(segments: Iterable[dict], *, offset: int = 0) -> list[dict]:
    result: list[dict] = []
    for local_index, segment in enumerate(segments):
        copied = dict(segment)
        copied["cue_id"] = _cue_id(copied, offset + local_index)
        result.append(copied)
    return result


def _prompts(lines: str, n: int, topic: str = "") -> tuple[str, str]:
    topic_rule = ""
    if topic.strip():
        topic_rule = (
            f"The viewer is studying {topic.strip()!r}. Return only clips that teach "
            "material relevant to that context.\n"
        )
    system = (
        "Select self-contained educational clips from timestamped transcript cues. "
        "Each transcript line has a local line index and an immutable cue_id. "
        "Use only the supplied lines. Do not repair, clamp, or invent indices. "
        "start_quote must occur verbatim in start_line after Unicode/case normalization; "
        "end_quote must occur in end_line. Clips begin at start_line's cue start and end "
        "at end_line's cue end.\n" + topic_rule +
        "Acceptable kind values are content or educational; label intros, admin, promos, "
        "and outros truthfully so they can be removed. Return informativeness, "
        "topic_relevance, self_contained, and difficulty on a 0..1 scale. Prefer complete "
        "20-90 second ideas, but complete clips may be 1-180 seconds. "
        "Every non-empty summary must include summary_cue_ids; every takeaway must have "
        "a same-position takeaway_cue_ids list; every match_reason must include "
        "match_reason_cue_ids; assessment must include cue_ids alongside prompt, exactly "
        "four distinct options, correct_index, and explanation. Grounding IDs must all be "
        "selected by that clip. Unsupported metadata will be discarded. "
        f"Line indices are strict integers from 0 through {n - 1}."
    )
    user = (
        f"Transcript batch ({n} cues, `[line|cue_id] MM:SS text`):\n\n{lines}\n\n"
        "Return {topics:[{title,start_line,end_line,start_quote,end_quote,reason,facet,kind,"
        "informativeness,topic_relevance,self_contained,difficulty,summary,summary_cue_ids,"
        "takeaways,takeaway_cue_ids,match_reason,match_reason_cue_ids,assessment:{prompt,"
        "options,correct_index,explanation,cue_ids}}]}."
    )
    return system, user


def _near_duplicate(a: dict, b: dict, threshold: float = 0.8) -> bool:
    overlap = min(float(a["end"]), float(b["end"])) - max(float(a["start"]), float(b["start"]))
    if overlap <= 0:
        return False
    shorter = min(float(a["end"]) - float(a["start"]), float(b["end"]) - float(b["start"]))
    return shorter > 0 and overlap / shorter >= threshold


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def _grounding_ids(value: object, selected: set[str]) -> list[str]:
    if not isinstance(value, list):
        return []
    ids = [_clean_text(item) for item in value]
    ids = list(dict.fromkeys(item for item in ids if item))
    return ids if ids and set(ids) <= selected else []


def _clean_list(value: object, *, maximum: int) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = _clean_text(item)
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
        if len(cleaned) >= maximum:
            break
    return cleaned


def _grounded_takeaways(topic: _Topic, selected: set[str]) -> tuple[list[str], list[list[str]]]:
    values = topic.takeaways if isinstance(topic.takeaways, list) else []
    refs = topic.takeaway_cue_ids if isinstance(topic.takeaway_cue_ids, list) else []
    kept_values: list[str] = []
    kept_refs: list[list[str]] = []
    seen: set[str] = set()
    for index, raw in enumerate(values):
        text = _clean_text(raw)
        ids = _grounding_ids(refs[index] if index < len(refs) else None, selected)
        key = text.casefold()
        if not text or not ids or key in seen:
            continue
        seen.add(key)
        kept_values.append(text)
        kept_refs.append(ids)
        if len(kept_values) >= 4:
            break
    return kept_values, kept_refs


def _validated_assessment(value: object, *, selected: set[str], grounding_text: str) -> dict | None:
    if isinstance(value, _Assessment):
        value = value.model_dump()
    if not isinstance(value, dict):
        return None
    cue_ids = _grounding_ids(value.get("cue_ids"), selected)
    prompt = _clean_text(value.get("prompt"))
    explanation = _clean_text(value.get("explanation"))
    options = _clean_list(value.get("options"), maximum=4)
    correct_index = value.get("correct_index")
    if (
        not cue_ids
        or not prompt
        or not explanation
        or len(options) != 4
        or isinstance(correct_index, bool)
        or not isinstance(correct_index, int)
        or not 0 <= correct_index < 4
    ):
        return None
    source_tokens = {token for token in _tokens(grounding_text) if len(token) >= 4}
    explanation_tokens = {token for token in _tokens(explanation) if len(token) >= 4}
    if source_tokens and not source_tokens & explanation_tokens:
        return None
    return {
        "prompt": prompt,
        "options": options,
        "correct_index": correct_index,
        "explanation": explanation,
        "cue_ids": cue_ids,
    }


def _quality_order(clips: list[dict], *, max_clips: int) -> list[dict]:
    ordered = sorted(
        clips,
        key=lambda clip: (
            float(clip["informativeness"]) + float(clip["topic_relevance"]),
            -(float(clip["end"]) - float(clip["start"])),
            -float(clip["start"]),
        ),
        reverse=True,
    )
    kept: list[dict] = []
    for candidate in ordered:
        if any(_near_duplicate(candidate, other) for other in kept):
            continue
        kept.append(candidate)
        if len(kept) >= max_clips:
            break
    kept.sort(key=lambda clip: (clip["start"], clip["end"]))
    for index, clip in enumerate(kept, 1):
        clip["sequence_index"] = index
    return kept


def _plan_to_clips(plan: _Plan, segs: list[dict], words: list[dict], settings: dict) -> list[dict]:
    del words  # Provider cues intentionally carry no synthetic word times.
    segments = _with_cue_ids(segs)
    count = len(segments)
    if not count:
        return []
    info_min = max(
        0.6,
        float(settings.get("segment_informativeness_min", config.SEGMENT_INFORMATIVENESS_MIN)),
    )
    relevance_min = max(
        0.6,
        float(settings.get("segment_topic_relevance_min", config.SEGMENT_TOPIC_RELEVANCE_MIN)),
    )
    raw: list[dict] = []
    for topic in plan.topics:
        kind = _clean_text(topic.kind).lower()
        info = _norm_optional_confidence(topic.informativeness)
        relevance = _norm_optional_confidence(topic.topic_relevance)
        if (
            kind not in _ACCEPT_KINDS
            or info is None
            or relevance is None
            or topic.self_contained is not True
            or info < info_min
            or relevance < relevance_min
        ):
            continue
        start_index = topic.start_line
        end_index = topic.end_line
        if (
            isinstance(start_index, bool)
            or isinstance(end_index, bool)
            or start_index < 0
            or end_index < start_index
            or end_index >= count
        ):
            continue
        start_cue = segments[start_index]
        end_cue = segments[end_index]
        if not _contains_tokens(str(start_cue.get("text") or ""), topic.start_quote):
            continue
        if not _contains_tokens(str(end_cue.get("text") or ""), topic.end_quote):
            continue
        start = round(float(start_cue["start"]), 3)
        end = round(float(end_cue["end"]), 3)
        if not 1.0 <= end - start <= 180.0:
            continue
        selected_cues = segments[start_index:end_index + 1]
        cue_ids = [_cue_id(cue, start_index + offset) for offset, cue in enumerate(selected_cues)]
        selected = set(cue_ids)
        clip_text = " ".join(_clean_text(cue.get("text")) for cue in selected_cues).strip()
        summary_ids = _grounding_ids(topic.summary_cue_ids, selected)
        summary = _clean_text(topic.summary) if summary_ids else ""
        takeaways, takeaway_ids = _grounded_takeaways(topic, selected)
        match_reason_ids = _grounding_ids(topic.match_reason_cue_ids, selected)
        match_reason = _clean_text(topic.match_reason) if match_reason_ids else ""
        assessment = _validated_assessment(
            topic.assessment,
            selected=selected,
            grounding_text=" ".join([clip_text, summary, *takeaways]),
        )
        raw.append(
            {
                "start": start,
                "end": end,
                "title": _clean_text(topic.title),
                "facet": _clean_text(topic.facet) or "other",
                "reason": _clean_text(topic.reason),
                "kind": kind,
                "informativeness": info,
                "topic_relevance": relevance,
                "self_contained": True,
                "difficulty": _norm_informativeness(topic.difficulty),
                "summary": summary,
                "summary_cue_ids": summary_ids if summary else [],
                "takeaways": takeaways,
                "takeaway_cue_ids": takeaway_ids,
                "match_reason": match_reason,
                "match_reason_cue_ids": match_reason_ids if match_reason else [],
                "assessment": assessment,
                "cue_ids": cue_ids,
                "start_cue_id": cue_ids[0],
                "end_cue_id": cue_ids[-1],
                "transcript_text": clip_text,
                "model_used": _clean_text(settings.get("_model_used")),
                "quality_degraded": bool(settings.get("_quality_degraded", False)),
            }
        )
    max_clips = min(40, max(0, int(settings.get("max_clips") or config.SEGMENT_MAX_CLIPS)))
    return _quality_order(raw, max_clips=max_clips)


def _estimate_tokens(value: str) -> int:
    return max(1, (len(value) + max(1, config.CHARS_PER_TOKEN) - 1) // max(1, config.CHARS_PER_TOKEN))


def _cue_batches(
    segments: list[dict],
    *,
    max_cues: int,
    max_input_tokens: int,
    overlap_cues: int,
) -> list[tuple[int, list[dict]]]:
    if not segments:
        return []
    max_cues = max(1, int(max_cues))
    overlap_cues = max(0, min(int(overlap_cues), max_cues - 1))
    token_budget = max(256, int(max_input_tokens) - 1800)
    batches: list[tuple[int, list[dict]]] = []
    start = 0
    while start < len(segments):
        end = start
        tokens = 0
        while end < len(segments) and end - start < max_cues:
            cue = segments[end]
            cost = _estimate_tokens(str(cue.get("text") or "")) + 12
            if end > start and tokens + cost > token_budget:
                break
            tokens += cost
            end += 1
        if end == start:
            end += 1
        batches.append((start, segments[start:end]))
        if end >= len(segments):
            break
        start = max(start + 1, end - overlap_cues)
    return batches
