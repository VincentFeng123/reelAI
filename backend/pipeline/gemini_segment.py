"""Guarded Gemini educational clip segmentation.

Production starts with a guarded Flash-first canary. Hybrid mode admits only
deterministic ``green`` Gemini 3.5 Flash results and reruns every uncertain or
invalid request with the configured Pro model. Shadow and Pro-only modes remain
available as explicit overrides.

The public contract stays ``segment_clips(...) -> (clips, notes)``.  Model names,
routing decisions, and call telemetry are logged internally and never added to a
clip, note, or API response.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from threading import Lock
from typing import Callable, Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)
from typing_extensions import Annotated

from .. import config

ProgressCb = Optional[Callable[[float, str], None]]
CancelledCb = Optional[Callable[[], bool]]

log = logging.getLogger("clipper.segment")

_WORD_RE = re.compile(r"[a-z0-9']+", re.IGNORECASE)
_NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

PRODUCTION_PRO_PROFILE = "production_pro_v0"
CORRECTED_PRO_PROFILE = "corrected_pro_v1"
FLASH_SINGLE_PROFILE = "flash_single_v1"
FLASH_SPLIT_PROFILE = "flash_split_v1"
# The guarded hybrid route uses this Flash profile. Profile changes remain gated
# independently from the routing rollout.
PRODUCTION_FLASH_PROFILE = FLASH_SINGLE_PROFILE
# Corrected Pro replaces this authority only after its own baseline gate clears.
# Every Pro route (control, shadow, fallback, rollback) uses the same selection.
AUTHORITATIVE_PRO_PROFILE = PRODUCTION_PRO_PROFILE
SEGMENT_PROFILES = (
    PRODUCTION_PRO_PROFILE,
    CORRECTED_PRO_PROFILE,
    FLASH_SINGLE_PROFILE,
    FLASH_SPLIT_PROFILE,
)

_TOTAL_DEADLINE_S = 150.0
_FLASH_SINGLE_TIMEOUT_S = 45.0
_FLASH_BOUNDARY_TIMEOUT_S = 45.0
_FLASH_ENRICH_TIMEOUT_S = 25.0
_PRO_TIMEOUT_S = 90.0
_SELECTION_OUTPUT_TOKENS = 24_576
_BOUNDARY_OUTPUT_TOKENS = 12_288
_ENRICH_OUTPUT_TOKENS = 24_576
_MIN_CLIP_S = 1.0
_MAX_CLIP_S = 180.0
_UNCERTAIN_DURATION_S = 150.0
_MIN_SCORE = 0.60
_GREEN_SCORE = 0.75
_MAX_CLIPS = 40
_DUPLICATE_OVERLAP = 0.8

_PRICING_VERSION = "gemini-standard-2026-07-11"
_PRICING_PER_MILLION = {
    "flash": {"input": 1.50, "output": 9.00},
    "pro": {"input": 2.00, "output": 12.00},
}

_flash_disable_lock = Lock()
_flash_disabled_reason: str | None = None


# ---------------------------------------------------------------------------
# Strict model schemas

class _StrictModel(BaseModel):
    # Gemini's response_schema endpoint rejects Pydantic's
    # ``additionalProperties: false`` representation for ``extra="forbid"``.
    # Required typed fields still constrain generation; semantic validation below
    # remains authoritative.
    model_config = ConfigDict(extra="forbid")


class _UncertaintyReason(str, Enum):
    BOUNDARY_AMBIGUOUS = "boundary_ambiguous"
    INCOMPLETE_CONTEXT = "incomplete_context"
    TOPIC_AMBIGUOUS = "topic_ambiguous"
    TRANSCRIPT_NOISE = "transcript_noise"
    OVERLAP_RISK = "overlap_risk"
    OTHER = "other"


class _AssessmentDraft(_StrictModel):
    prompt: _NonBlank
    options: list[_NonBlank] = Field(min_length=4, max_length=4)
    correct_index: int = Field(ge=0, le=3, strict=True)
    explanation: _NonBlank
    evidence_quote: _NonBlank

    @model_validator(mode="after")
    def _unique_options(self):
        normalized = {" ".join(option.split()).casefold() for option in self.options}
        if len(normalized) != 4:
            raise ValueError("assessment options must be distinct")
        if any("all of the above" in option.casefold() for option in self.options):
            raise ValueError("all-of-the-above options are not allowed")
        return self


class _BoundaryTopic(_StrictModel):
    start_line: int = Field(ge=0, strict=True)
    end_line: int = Field(ge=0, strict=True)
    start_quote: _NonBlank
    end_quote: _NonBlank
    title: _NonBlank
    facet: _NonBlank
    reason: _NonBlank
    informativeness: float = Field(ge=0.0, le=1.0, strict=True)
    topic_relevance: float = Field(ge=0.0, le=1.0, strict=True)
    difficulty: float = Field(ge=0.0, le=1.0, strict=True)
    self_contained: bool = Field(strict=True)
    uncertainty: Literal["low", "medium", "high"]
    uncertainty_reasons: list[_UncertaintyReason] = Field(max_length=6)

    @model_validator(mode="after")
    def _uncertainty_has_reason(self):
        if self.uncertainty != "low" and not self.uncertainty_reasons:
            raise ValueError("medium/high uncertainty requires a reason")
        return self


class _Topic(_BoundaryTopic):
    summary: _NonBlank
    takeaways: list[_NonBlank] = Field(min_length=2, max_length=4)
    match_reason: _NonBlank
    assessment: _AssessmentDraft | None

    @field_validator("takeaways")
    @classmethod
    def _distinct_takeaways(cls, value: list[str]) -> list[str]:
        if len({" ".join(item.split()).casefold() for item in value}) != len(value):
            raise ValueError("takeaways must be distinct")
        return value


class _Plan(_StrictModel):
    topics: list[_Topic] = Field(max_length=_MAX_CLIPS)


class _BoundaryPlan(_StrictModel):
    topics: list[_BoundaryTopic] = Field(max_length=_MAX_CLIPS)


class _EnrichmentItem(_StrictModel):
    clip_id: _NonBlank
    summary: _NonBlank
    takeaways: list[_NonBlank] = Field(min_length=2, max_length=4)
    match_reason: _NonBlank
    assessment: _AssessmentDraft

    @field_validator("takeaways")
    @classmethod
    def _distinct_takeaways(cls, value: list[str]) -> list[str]:
        if len({" ".join(item.split()).casefold() for item in value}) != len(value):
            raise ValueError("takeaways must be distinct")
        return value


class _EnrichmentPlan(_StrictModel):
    items: list[_EnrichmentItem] = Field(max_length=_MAX_CLIPS)


# The frozen production prompt remains available as an immutable evaluation
# baseline.  Its schema is deliberately permissive because that is part of the
# profile being measured; strict application validation still guards its output.
class _ProductionTopic(BaseModel):
    title: str
    start_line: int
    end_line: int
    start_quote: str
    end_quote: str
    reason: str = ""
    facet: str = "other"
    kind: Optional[str] = None
    informativeness: Optional[float] = None
    topic_relevance: Optional[float] = None
    self_contained: Optional[bool] = None
    difficulty: float = 0.5


class _LegacyTopic(_ProductionTopic):
    summary: str = ""
    takeaways: list[str] = Field(default_factory=list)
    match_reason: str = ""
    assessment: dict | None = None


class _LegacyPlan(BaseModel):
    topics: list[_LegacyTopic]


class _ProductionPlan(BaseModel):
    topics: list[_ProductionTopic]


# ---------------------------------------------------------------------------
# Prompt construction

_POLICY_AND_EXAMPLES = """Policy:
- Return only complete, substantive teaching units. Omit greetings, sponsors,
  administration, promos, outros, tangents, and partial explanations.
- Prefer fewer clips to forcing an incomplete idea. Keep results chronological.
- Contextual overlap is allowed only when both clips remain independently complete.
- Copy exact transcript line IDs and exact opening/closing quotes.
- A complete clip may be 1 to 180 seconds; prefer focused 20 to 90 second units.
- Do not provide chain-of-thought or hidden reasoning.

Examples:
KEEP this complete teaching unit:
[12] 01:20 Gradient descent updates model parameters in the direction that reduces loss.
[13] 01:28 Repeating that update moves the model until the loss reaches a minimum.
Use start_line=12, end_line=13,
start_quote="Gradient descent updates model parameters",
end_quote="until the loss reaches a minimum".

OMIT these non-units:
[20] 02:05 Welcome back, and thanks to today's sponsor.
[21] 02:12 The first half of the explanation is that the variable changes because...
The first line is framing/sponsorship; the second ends before the explanation is complete.
"""


def _topic_rule(topic: str) -> str:
    topic = topic.strip()
    if not topic:
        return "No topic filter was supplied; return every substantive educational unit."
    return (
        f"The viewer is studying {topic!r}. Return only units that directly teach that "
        "topic, and make each match_reason name the relevant idea."
    )


def _selection_fields(*, enriched: bool) -> str:
    fields = (
        "start_line, end_line, start_quote, end_quote, title, facet, reason, "
        "informativeness, topic_relevance, difficulty, self_contained, uncertainty, "
        "uncertainty_reasons"
    )
    if enriched:
        fields += (
            ", summary, takeaways (2-4 distinct grounded points), match_reason, and "
            "assessment {prompt (at most 16 words), exactly four distinct options "
            "(at most 8 words each), correct_index, explanation (one sentence, at most "
            "24 words), evidence_quote copied exactly from the selected clip}"
        )
    return fields


def _prompts(lines: str, n: int, topic: str = "") -> tuple[str, str]:
    """Gemini 3.5 single-pass prompt: policy/examples, context, task last."""
    system = (
        "You select self-contained educational clips from timestamped transcripts.\n\n"
        + _POLICY_AND_EXAMPLES
    )
    user = (
        f"{_topic_rule(topic)}\nLine IDs must be between 0 and {n - 1}.\n\n"
        f"Transcript ({n} lines, formatted `[index] MM:SS text`):\n{lines}\n\n"
        "Based on the preceding transcript, return the chronological educational units. "
        f"Every item must contain {_selection_fields(enriched=True)}. Return no item for "
        "material that is incomplete or non-educational."
    )
    return system, user


def _boundary_prompts(lines: str, n: int, topic: str = "") -> tuple[str, str]:
    system = (
        "You select self-contained educational clip boundaries from timestamped transcripts.\n\n"
        + _POLICY_AND_EXAMPLES
    )
    user = (
        f"{_topic_rule(topic)}\nLine IDs must be between 0 and {n - 1}.\n\n"
        f"Transcript ({n} lines, formatted `[index] MM:SS text`):\n{lines}\n\n"
        "Based on the preceding transcript, return only the chronological boundary selections. "
        f"Every item must contain {_selection_fields(enriched=False)}. Learning details and "
        "assessments are generated later, so do not include them."
    )
    return system, user


def _legacy_prompts(lines: str, n: int, topic: str = "") -> tuple[str, str]:
    """Pre-router production prompt used by ``production_pro_v0``."""
    topic_rule = ""
    if topic.strip():
        topic_rule = (
            f"The viewer is studying: {topic.strip()!r}. Only return clips that TEACH "
            "material relevant to that topic; skip unrelated sections entirely.\n"
        )
    system = (
        "You select self-contained CLIPS from a lecture/talk transcript for a short-form "
        "learning feed. First read and understand the WHOLE transcript. Then pick the "
        "SUBSTANTIVE teaching moments — one coherent idea, concept, worked example, or "
        "section, taught from its introduction through to its natural conclusion. Skip pure "
        "filler (greetings, admin, 'like and subscribe', tangents), course-logistics intros, "
        "and wrap-up outros.\n" + topic_rule +
        "For every clip return: title; start_line (the line where the idea is INTRODUCED); "
        "end_line (the line where it CLOSES); start_quote (the first ~6 words spoken at the "
        "start, copied verbatim from that line); end_quote (the last ~6 words, verbatim); a "
        "short reason; kind — one of content|educational|intro|outro|admin|promo; "
        "informativeness — 0.0 to 1.0, how much a motivated student learns from this clip "
        "ALONE; topic_relevance — 0.0 to 1.0, how directly it teaches the viewer's topic; "
        "self_contained — true only when it makes sense without omitted context; "
        "difficulty — 0.0 to 1.0, the prior knowledge the clip ASSUMES (0.1: no background, "
        "first exposure; 0.5: comfortable with the basics; 0.9: graduate/expert material). "
        "When supported by the clip, also return a grounded summary, two to four distinct "
        "takeaways, a topic-specific match_reason, and an assessment with prompt, exactly four "
        "options, correct_index, explanation, and an exact evidence_quote from the clip. "
        "Rules: (1) a clip must START at the beginning of the idea and END at its end — "
        "never mid-thought; (2) contextual overlap is allowed when two complete ideas share "
        "setup; (3) go in chronological order; (4) prefer complete clips of 20-90 seconds. "
        "Split longer sections into complete subtopics. A complete clip may be up to 180 "
        "seconds, but never force or truncate a longer section; (5) line indices range from 0 to "
        f"{n - 1} — never exceed {n - 1}."
    )
    user = (
        f"Transcript ({n} lines, each formatted `[index] MM:SS text`):\n\n" + lines +
        "\n\nReturn every substantive teaching clip as {title, start_line, end_line, "
        "start_quote, end_quote, reason, facet, kind, informativeness, topic_relevance, "
        "self_contained, difficulty, summary, takeaways, match_reason, assessment}."
    )
    return system, user


def _enrichment_prompts(clips: list[dict], topic: str) -> tuple[str, str]:
    system = (
        "Ground learning details only in the supplied accepted clip excerpts. Do not add facts "
        "from titles, outside knowledge, or another clip. Each assessment must have exactly one "
        "correct option and an exact evidence quote from its own excerpt. Do not provide "
        "chain-of-thought."
    )
    blocks = []
    for clip in clips:
        blocks.append(
            f"<clip id={clip['_clip_id']!r}>\n{clip['_clip_text']}\n</clip>"
        )
    user = (
        f"Viewer topic: {topic.strip() or '(none)'}.\n\n"
        + "\n\n".join(blocks)
        + "\n\nBased on the preceding accepted clip excerpts, return one item for every clip_id "
          "with a grounded 1-2 sentence summary, 2-4 distinct takeaways, a topic-specific "
          "match_reason, and a four-option assessment whose evidence_quote is copied exactly "
          "from that clip."
    )
    return system, user


# ---------------------------------------------------------------------------
# Transcript alignment and validation

def _mmss(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


def _toks(text: str) -> list[str]:
    return [match.group(0).lower() for match in _WORD_RE.finditer(text or "")]


def _contains_quote(text: str, quote: str) -> bool:
    haystack, needle = _toks(text), _toks(quote)
    if not needle or len(needle) > len(haystack):
        return False
    return any(haystack[i:i + len(needle)] == needle
               for i in range(len(haystack) - len(needle) + 1))


def _locate_quote_match(
    words: list[dict], quote: str, lo_t: float, hi_t: float, want: str,
) -> tuple[float, float] | None:
    """Return ``(boundary_time, score)`` for a high-confidence word alignment."""
    quote_tokens = _toks(quote)
    if not quote_tokens or not words:
        return None
    qn = min(len(quote_tokens), 6)
    target_tokens = quote_tokens[:qn] if want == "start" else quote_tokens[-qn:]
    target = " ".join(target_tokens)
    timed: list[tuple[str, float, float]] = []
    for word in words:
        try:
            start = float(word.get("start", 0.0))
            end = float(word.get("end", start))
        except (TypeError, ValueError):
            continue
        if not (lo_t - 1e-6 <= start <= hi_t + 1e-6):
            continue
        tokens = _toks(str(word.get("word") or ""))
        if tokens:
            timed.append((tokens[0], start, end))
    if len(timed) < qn:
        return None

    try:
        from rapidfuzz import fuzz

        score_fn = lambda a, b: float(fuzz.ratio(a, b))
    except Exception:  # pragma: no cover - rapidfuzz is a required dependency
        from difflib import SequenceMatcher

        score_fn = lambda a, b: 100.0 * SequenceMatcher(None, a, b).ratio()

    best: tuple[float, float] | None = None
    indices = range(len(timed) - qn + 1)
    if want == "end":
        indices = reversed(list(indices))
    for i in indices:
        window = " ".join(token for token, _start, _end in timed[i:i + qn])
        score = score_fn(target, window)
        boundary = timed[i][1] if want == "start" else timed[i + qn - 1][2]
        if best is None or score > best[1]:
            best = (boundary, score)
    return best if best is not None and best[1] >= 85.0 else None


def _locate_quote(words: list[dict], quote: str, lo_t: float, hi_t: float,
                  want: str) -> Optional[float]:
    match = _locate_quote_match(words, quote, lo_t, hi_t, want)
    return match[0] if match else None


def _caption_gap_boundary(
    segments: list[dict], semantic_time: float, *, direction: str,
    min_gap_s: float = 0.25, max_move_s: float = 2.0,
) -> float:
    """Move outward to the nearest qualifying caption-gap midpoint."""
    candidates: list[float] = []
    for left, right in zip(segments, segments[1:]):
        gap_start = float(left.get("end", 0.0))
        gap_end = float(right.get("start", gap_start))
        if gap_end - gap_start < min_gap_s:
            continue
        midpoint = (gap_start + gap_end) / 2.0
        delta = semantic_time - midpoint if direction == "start" else midpoint - semantic_time
        if -1e-9 <= delta <= max_move_s + 1e-9:
            candidates.append(midpoint)
    if not candidates:
        return semantic_time
    return max(candidates) if direction == "start" else min(candidates)


def _near_duplicate(a: dict, b: dict, threshold: float = _DUPLICATE_OVERLAP) -> bool:
    overlap = min(float(a["end"]), float(b["end"])) - max(float(a["start"]), float(b["start"]))
    if overlap <= 0:
        return False
    shorter = min(float(a["end"]) - float(a["start"]),
                  float(b["end"]) - float(b["start"]))
    return shorter > 0 and overlap / shorter >= threshold


def _content_tokens(text: str) -> set[str]:
    stop = {
        "about", "after", "again", "also", "an", "and", "are", "as", "at", "be",
        "because", "been", "before", "being", "but", "by", "can", "could", "did", "do",
        "does", "doing", "for", "from", "had", "has", "have", "having", "he", "her",
        "here", "him", "his", "how", "if", "in", "into", "is", "it", "its", "just",
        "may", "me", "more", "most", "my", "no", "nor", "not", "now", "of", "off",
        "on", "only", "or", "our", "she", "should", "so", "some", "such", "than",
        "that", "the", "their", "them", "then", "there", "these", "they", "this",
        "those", "through", "to", "too", "us", "very", "was", "we", "were", "what",
        "when", "where", "which", "who", "why", "will", "with", "would", "yes", "you",
        "your",
    }

    def stem(token: str) -> str:
        if token.endswith("ies") and len(token) > 4:
            return token[:-3] + "y"
        if token.endswith("ing") and len(token) > 5:
            return token[:-3]
        if token.endswith("ed") and len(token) > 4:
            return token[:-2]
        if token.endswith("es") and len(token) > 4:
            return token[:-2]
        if token.endswith("s") and len(token) > 3:
            return token[:-1]
        return token

    return {
        stem(token) for token in _toks(text)
        if (len(token) >= 2 or token in {"c", "r"}) and token not in stop
    }


def _timed_clip_text(words: list[dict], start: float, end: float) -> str:
    tokens: list[str] = []
    for word in words:
        try:
            word_start = float(word.get("start", 0.0))
            word_end = float(word.get("end", word_start))
        except (TypeError, ValueError):
            continue
        if min(word_end, end) - max(word_start, start) <= 0.0:
            continue
        text = str(word.get("word") or "").strip()
        if text:
            tokens.append(text)
    return " ".join(tokens).strip()


def _text_has_grounding(text: str, transcript_text: str) -> bool:
    source = _content_tokens(transcript_text)
    generated = _content_tokens(text)
    if not source or not generated:
        return False
    shared = source & generated
    return len(shared) >= min(2, len(generated))


def _validated_assessment(value: object, *, grounding_text: str) -> dict | None:
    """Validate assessment evidence only against the real clip transcript.

    ``evidence_quote`` is consumed as an internal guard and intentionally omitted
    from the stored dictionary to preserve the existing optional assessment shape.
    """
    try:
        draft = value if isinstance(value, _AssessmentDraft) else _AssessmentDraft.model_validate(value)
    except (ValidationError, TypeError, ValueError):
        return None
    if not _contains_quote(grounding_text, draft.evidence_quote):
        return None
    evidence_tokens = _content_tokens(draft.evidence_quote)
    if len(evidence_tokens) < 2:
        return None
    answer_support = _content_tokens(
        f"{draft.options[draft.correct_index]} {draft.explanation}"
    )
    if not evidence_tokens & answer_support:
        return None
    return {
        "prompt": draft.prompt,
        "options": list(draft.options),
        "correct_index": draft.correct_index,
        "explanation": draft.explanation,
    }


@dataclass
class _Conversion:
    clips: list[dict] = field(default_factory=list)
    proposed_count: int = 0
    rejected_reasons: list[str] = field(default_factory=list)
    enrichment_errors: list[str] = field(default_factory=list)
    near_duplicate: bool = False
    non_chronological: bool = False
    medium_uncertainty: bool = False
    score_below_green: bool = False
    long_clip: bool = False

    @property
    def accepted_count(self) -> int:
        return len(self.clips)


def _setting_bool(settings: dict, key: str, default: bool) -> bool:
    value = settings.get(key)
    return default if value is None else bool(value)


def _strict_score(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    score = float(value)
    return score if 0.0 <= score <= 1.0 else None


def _learning_details(topic_obj: object, clip_text: str, topic: str) -> tuple[dict, list[str]]:
    errors: list[str] = []
    details = {"summary": "", "takeaways": [], "match_reason": "", "assessment": None}
    summary = " ".join(str(getattr(topic_obj, "summary", "") or "").split())
    takeaways = [" ".join(str(item).split()) for item in
                 (getattr(topic_obj, "takeaways", None) or []) if str(item).strip()]
    match_reason = " ".join(str(getattr(topic_obj, "match_reason", "") or "").split())
    if not summary or not _text_has_grounding(summary, clip_text):
        errors.append("summary_not_grounded")
    else:
        details["summary"] = summary
    if not 2 <= len(takeaways) <= 4 or len({item.casefold() for item in takeaways}) != len(takeaways):
        errors.append("takeaways_invalid")
    elif any(not _text_has_grounding(item, clip_text) for item in takeaways):
        errors.append("takeaways_not_grounded")
    else:
        details["takeaways"] = takeaways
    if not match_reason or not _text_has_grounding(match_reason, clip_text):
        errors.append("match_reason_not_grounded")
    elif topic.strip() and not (_content_tokens(topic) & _content_tokens(match_reason)):
        errors.append("match_reason_not_topic_specific")
    else:
        details["match_reason"] = match_reason
    assessment = _validated_assessment(
        getattr(topic_obj, "assessment", None), grounding_text=clip_text,
    )
    if assessment is None:
        errors.append("assessment_invalid")
    else:
        details["assessment"] = assessment
    return details, errors


def _plan_to_report(
    plan: _Plan | _BoundaryPlan | _LegacyPlan | _ProductionPlan,
    segments: list[dict],
    words: list[dict],
    settings: dict,
    *,
    topic: str = "",
    require_enrichment: bool = False,
) -> _Conversion:
    report = _Conversion(proposed_count=len(plan.topics))
    n = len(segments)
    if not n:
        report.rejected_reasons.append("missing_segments")
        return report

    fine = _setting_bool(settings, "segment_fine_snap", config.SEGMENT_FINE_SNAP)
    info_setting = settings.get("segment_informativeness_min")
    relevance_setting = settings.get("segment_topic_relevance_min")
    info_floor = max(
        _MIN_SCORE,
        float(config.SEGMENT_INFORMATIVENESS_MIN if info_setting is None else info_setting),
    )
    relevance_floor = max(
        _MIN_SCORE,
        float(config.SEGMENT_TOPIC_RELEVANCE_MIN
              if relevance_setting is None else relevance_setting),
    )
    previous_start = -1
    raw: list[dict] = []

    for index, proposal in enumerate(plan.topics):
        prefix = f"proposal_{index}"
        if isinstance(proposal, _ProductionTopic):
            kind = str(proposal.kind or "").strip().lower()
            if kind not in {"content", "educational"}:
                report.rejected_reasons.append(f"{prefix}:not_educational")
                continue

        a, b = proposal.start_line, proposal.end_line
        if (isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, int)
                or not isinstance(b, int) or a < 0 or b < 0 or a >= n or b >= n or a > b):
            report.rejected_reasons.append(f"{prefix}:bad_index")
            continue
        if a < previous_start:
            report.non_chronological = True
            report.rejected_reasons.append(f"{prefix}:non_chronological")
            continue
        previous_start = a

        start_quote = str(proposal.start_quote or "").strip()
        end_quote = str(proposal.end_quote or "").strip()
        if not _contains_quote(str(segments[a].get("text") or ""), start_quote):
            report.rejected_reasons.append(f"{prefix}:bad_start_quote")
            continue
        if not _contains_quote(str(segments[b].get("text") or ""), end_quote):
            report.rejected_reasons.append(f"{prefix}:bad_end_quote")
            continue
        if not words:
            report.rejected_reasons.append(f"{prefix}:missing_word_timestamps")
            continue

        segment_start = float(segments[a].get("start", 0.0))
        segment_end = float(segments[b].get("end", segment_start))
        start_match = _locate_quote_match(
            words, start_quote, segment_start, float(segments[a].get("end", segment_start)), "start",
        )
        end_match = _locate_quote_match(
            words, end_quote, float(segments[b].get("start", segment_end)), segment_end, "end",
        )
        if start_match is None:
            report.rejected_reasons.append(f"{prefix}:start_quote_unaligned")
            continue
        if end_match is None:
            report.rejected_reasons.append(f"{prefix}:end_quote_unaligned")
            continue

        info = _strict_score(proposal.informativeness)
        relevance = _strict_score(proposal.topic_relevance)
        difficulty = _strict_score(proposal.difficulty)
        if info is None or relevance is None or difficulty is None:
            report.rejected_reasons.append(f"{prefix}:score_out_of_range")
            continue
        if info < info_floor or relevance < relevance_floor:
            report.rejected_reasons.append(f"{prefix}:quality_below_floor")
            continue
        if proposal.self_contained is not True:
            report.rejected_reasons.append(f"{prefix}:not_self_contained")
            continue
        uncertainty = str(getattr(proposal, "uncertainty", "low") or "low")
        uncertainty_reasons = [str(getattr(reason, "value", reason))
                               for reason in (getattr(proposal, "uncertainty_reasons", None) or [])]
        if uncertainty == "high":
            report.rejected_reasons.append(f"{prefix}:high_uncertainty")
            continue

        start = start_match[0] if fine else segment_start
        end = end_match[0] if fine else segment_end
        if end <= start:
            report.rejected_reasons.append(f"{prefix}:reversed_aligned_boundary")
            continue
        start = _caption_gap_boundary(segments, start, direction="start")
        end = _caption_gap_boundary(segments, end, direction="end")
        start, end = round(start, 3), round(end, 3)
        duration = round(end - start, 3)
        if duration < _MIN_CLIP_S or duration > _MAX_CLIP_S:
            report.rejected_reasons.append(f"{prefix}:invalid_duration")
            continue

        clip_text = _timed_clip_text(words, start, end)
        if not clip_text:
            report.rejected_reasons.append(f"{prefix}:empty_aligned_transcript")
            continue
        clip_id = f"clip-{index + 1:03d}-{a}-{b}"
        clip = {
            "start": start,
            "end": end,
            "title": str(proposal.title or "").strip(),
            "facet": str(proposal.facet or "").strip(),
            "reason": str(proposal.reason or "").strip(),
            "kind": "educational",
            "informativeness": info,
            "topic_relevance": relevance,
            "self_contained": True,
            "difficulty": difficulty,
            "_uncertainty": uncertainty,
            "_uncertainty_reasons": uncertainty_reasons,
            "_start_line": a,
            "_end_line": b,
            "_clip_id": clip_id,
            "_clip_text": clip_text,
            "summary": "",
            "takeaways": [],
            "match_reason": "",
            "assessment": None,
        }
        if hasattr(proposal, "summary"):
            details, errors = _learning_details(proposal, clip_text, topic)
            clip.update(details)
            if errors:
                report.enrichment_errors.extend(f"{clip_id}:{error}" for error in errors)
        elif require_enrichment:
            report.enrichment_errors.append(f"{clip_id}:missing_enrichment")
        raw.append(clip)

    # Detect duplicates before removing them so classification cannot turn green by repair.
    report.medium_uncertainty = any(
        clip.get("_uncertainty") == "medium" for clip in raw
    )
    report.score_below_green = any(
        min(float(clip["informativeness"]), float(clip["topic_relevance"])) < _GREEN_SCORE
        for clip in raw
    )
    report.long_clip = any(
        float(clip["end"]) - float(clip["start"]) >= _UNCERTAIN_DURATION_S
        for clip in raw
    )
    for i, candidate in enumerate(raw):
        if any(_near_duplicate(candidate, other) for other in raw[i + 1:]):
            report.near_duplicate = True
            break

    quality_order = sorted(
        raw,
        key=lambda clip: (
            clip["informativeness"] + clip["topic_relevance"],
            -(clip["end"] - clip["start"]),
        ),
        reverse=True,
    )
    kept: list[dict] = []
    for candidate in quality_order:
        if not any(_near_duplicate(candidate, prior) for prior in kept):
            kept.append(candidate)
    kept.sort(key=lambda clip: (clip["start"], clip["end"]))
    configured_limit = settings.get("max_clips")
    limit = config.SEGMENT_MAX_CLIPS if configured_limit is None else int(configured_limit)
    limit = max(0, min(_MAX_CLIPS, limit))
    report.clips = kept[:limit]
    for index, clip in enumerate(report.clips):
        clip["sequence_index"] = index + 1
    return report


def _public_clips(clips: list[dict]) -> list[dict]:
    return [{key: value for key, value in clip.items() if not key.startswith("_")}
            for clip in clips]


def _plan_to_clips(plan: _Plan | _BoundaryPlan | _LegacyPlan | _ProductionPlan,
                   segments: list[dict],
                   words: list[dict], settings: dict) -> list[dict]:
    """Compatibility helper used by focused conversion tests."""
    return _public_clips(_plan_to_report(plan, segments, words, settings).clips)


@dataclass(frozen=True)
class _Classification:
    status: Literal["green", "uncertain", "invalid"]
    reasons: tuple[str, ...]


def _transcript_duration(segments: list[dict]) -> float:
    if not segments:
        return 0.0
    starts = [float(segment.get("start", 0.0)) for segment in segments]
    ends = [float(segment.get("end", start)) for segment, start in zip(segments, starts)]
    return max(ends) - min(starts)


def _classify_flash(report: _Conversion, segments: list[dict], topic: str,
                    *, enrichment_required: bool) -> _Classification:
    invalid: list[str] = []
    uncertain: list[str] = []
    if report.rejected_reasons:
        invalid.extend(report.rejected_reasons)
    if report.non_chronological:
        invalid.append("non_chronological")
    if report.proposed_count and report.accepted_count == 0:
        invalid.append("all_proposals_rejected")
    if report.accepted_count == 0 and _transcript_duration(segments) >= 120.0:
        invalid.append("zero_clips_long_transcript")
    if enrichment_required and report.enrichment_errors:
        invalid.extend(report.enrichment_errors)
    if invalid:
        return _Classification("invalid", tuple(dict.fromkeys(invalid)))

    if report.medium_uncertainty or any(
        clip.get("_uncertainty") == "medium" for clip in report.clips
    ):
        uncertain.append("medium_uncertainty")
    if report.score_below_green or any(
        min(float(clip["informativeness"]), float(clip["topic_relevance"])) < _GREEN_SCORE
        for clip in report.clips
    ):
        uncertain.append("quality_score_below_green")
    if report.long_clip or any(
        float(clip["end"]) - float(clip["start"]) >= _UNCERTAIN_DURATION_S
        for clip in report.clips
    ):
        uncertain.append("long_clip")
    if report.near_duplicate:
        uncertain.append("near_duplicate")
    if topic.strip() and report.clips:
        topic_tokens = _content_tokens(topic)
        clip_tokens = set().union(*(_content_tokens(clip["_clip_text"]) for clip in report.clips))
        if topic_tokens and not topic_tokens & clip_tokens:
            uncertain.append("zero_direct_topic_support")
    if uncertain:
        return _Classification("uncertain", tuple(dict.fromkeys(uncertain)))
    return _Classification("green", ())


# ---------------------------------------------------------------------------
# Gemini calls, enrichment, routing, and telemetry

@dataclass
class SegmentResult:
    clips: list[dict]
    notes: str
    route: str
    classification: str
    classification_reasons: list[str] = field(default_factory=list)
    fallback_reasons: list[str] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)
    proposed_count: int = 0
    accepted_count: int = 0
    error: str | None = None
    flash_configuration_error: str | None = None


class _SchemaResponseError(RuntimeError):
    def __init__(self, message: str, telemetry: object):
        super().__init__(message)
        self.telemetry = telemetry


def _telemetry_dict(value: object) -> dict:
    if value is None:
        return {}
    if hasattr(value, "as_dict"):
        return dict(value.as_dict())
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return dict(value)
    return {"value": str(value)}


def _call_model(
    system: str,
    user: str,
    schema: type[BaseModel],
    *,
    model: str,
    thinking_level: str,
    max_output_tokens: int,
    timeout_s: float,
    deadline_monotonic: float,
    operation: str,
    prompt_version: str,
    cancelled: CancelledCb,
) -> tuple[BaseModel, dict]:
    from ..gemini_client import generate_json_v3

    result = generate_json_v3(
        system,
        user,
        schema,
        model=model,
        thinking_level=thinking_level,
        max_output_tokens=max_output_tokens,
        timeout_s=timeout_s,
        deadline_monotonic=deadline_monotonic,
        operation=operation,
        prompt_version=prompt_version,
        max_retries=1,
        cancelled=cancelled,
    )
    try:
        parsed = schema.model_validate_json(result.text.strip())
    except (ValidationError, ValueError) as exc:
        raise _SchemaResponseError(
            f"invalid {schema.__name__} response: {exc}", result.telemetry,
        ) from exc
    return parsed, _telemetry_dict(result.telemetry)


def _exception_telemetry(exc: Exception) -> dict:
    return _telemetry_dict(getattr(exc, "telemetry", None))


def _model_cost(call: dict) -> float:
    model = str(call.get("model") or "").lower()
    rates = _PRICING_PER_MILLION["flash" if "flash" in model else "pro"]
    prompt = int(call.get("prompt_tokens") or call.get("prompt_token_count") or 0)
    candidate = int(
        call.get("candidate_tokens") or call.get("candidates_token_count") or 0
    )
    thought = int(call.get("thought_tokens") or call.get("thoughts_token_count") or 0)
    return (prompt * rates["input"] + (candidate + thought) * rates["output"]) / 1_000_000.0


def _emit(sink: Optional[Callable[[dict], None]], event: str, **fields) -> None:
    payload = {"event": event, **fields}
    log.info("segment_event %s", json.dumps(payload, sort_keys=True, default=str))
    if sink is not None:
        try:
            sink(payload)
        except Exception:  # telemetry must never fail segmentation
            log.warning("segment telemetry sink failed", exc_info=True)


def _cancel_requested(cancelled: object) -> bool:
    if cancelled is None:
        return False
    if callable(cancelled):
        return bool(cancelled())
    is_set = getattr(cancelled, "is_set", None)
    return bool(is_set()) if callable(is_set) else bool(cancelled)


def _lines(segments: list[dict]) -> str:
    return "\n".join(
        f"[{index}] {_mmss(segment.get('start', 0.0))} "
        f"{str(segment.get('text') or '').strip()}"
        for index, segment in enumerate(segments)
    )


def _run_selection_profile(
    profile: str,
    transcript: dict,
    topic: str,
    settings: dict,
    *,
    deadline: float,
    cancelled: CancelledCb,
) -> tuple[_Conversion, _Classification, list[dict]]:
    segments = transcript.get("segments") or []
    words = transcript.get("words") or []
    rendered = _lines(segments)

    if profile == PRODUCTION_PRO_PROFILE:
        system, user = _legacy_prompts(rendered, len(segments), topic)
        schema: type[BaseModel] = _LegacyPlan
        model = config.SEGMENT_PRO_MODEL
        level, cap, timeout = "high", _SELECTION_OUTPUT_TOKENS, _PRO_TIMEOUT_S
        operation = "pro_authoritative"
    elif profile == CORRECTED_PRO_PROFILE:
        system, user = _prompts(rendered, len(segments), topic)
        schema = _Plan
        model = config.SEGMENT_PRO_MODEL
        level, cap, timeout = "high", _SELECTION_OUTPUT_TOKENS, _PRO_TIMEOUT_S
        operation = "pro_fallback"
    elif profile == FLASH_SINGLE_PROFILE:
        system, user = _prompts(rendered, len(segments), topic)
        schema = _Plan
        model = config.SEGMENT_FLASH_MODEL
        level, cap, timeout = "medium", _SELECTION_OUTPUT_TOKENS, _FLASH_SINGLE_TIMEOUT_S
        operation = "flash_single_candidate"
    elif profile == FLASH_SPLIT_PROFILE:
        system, user = _boundary_prompts(rendered, len(segments), topic)
        schema = _BoundaryPlan
        model = config.SEGMENT_FLASH_MODEL
        level, cap, timeout = "medium", _BOUNDARY_OUTPUT_TOKENS, _FLASH_BOUNDARY_TIMEOUT_S
        operation = "flash_boundary_selector"
    else:
        raise ValueError(f"unknown segmentation profile: {profile}")

    parsed, call = _call_model(
        system,
        user,
        schema,
        model=model,
        thinking_level=level,
        max_output_tokens=cap,
        timeout_s=timeout,
        deadline_monotonic=deadline,
        operation=operation,
        prompt_version=profile,
        cancelled=cancelled,
    )
    require_enrichment = profile in {CORRECTED_PRO_PROFILE, FLASH_SINGLE_PROFILE}
    report = _plan_to_report(
        parsed, segments, words, settings, topic=topic, require_enrichment=require_enrichment,
    )
    if profile.startswith("flash_"):
        classification = _classify_flash(
            report, segments, topic, enrichment_required=(profile == FLASH_SINGLE_PROFILE),
        )
    else:
        classification = _Classification("green" if report.clips else "invalid", ())
    return report, classification, [call]


def _apply_enrichment(clips: list[dict], plan: _EnrichmentPlan, topic: str) -> list[str]:
    by_id: dict[str, _EnrichmentItem] = {}
    duplicate_ids: set[str] = set()
    for item in plan.items:
        if item.clip_id in by_id:
            duplicate_ids.add(item.clip_id)
        by_id[item.clip_id] = item
    errors: list[str] = []
    expected = {clip["_clip_id"] for clip in clips}
    if set(by_id) - expected:
        errors.extend(f"unknown_clip_id:{clip_id}" for clip_id in sorted(set(by_id) - expected))
    errors.extend(f"duplicate_clip_id:{clip_id}" for clip_id in sorted(duplicate_ids))
    for clip in clips:
        item = by_id.get(clip["_clip_id"])
        if item is None:
            errors.append(f"missing_clip_id:{clip['_clip_id']}")
            continue
        details, item_errors = _learning_details(item, clip["_clip_text"], topic)
        clip.update(details)
        if item_errors:
            errors.extend(f"{clip['_clip_id']}:{error}" for error in item_errors)
            continue
    return errors


def _invalid_enrichment_clip_ids(errors: list[str], clips: list[dict]) -> set[str]:
    ids = {clip["_clip_id"] for clip in clips}
    invalid: set[str] = set()
    for error in errors:
        for clip_id in ids:
            if clip_id in error:
                invalid.add(clip_id)
    if any(error.startswith(("unknown_clip_id:", "duplicate_clip_id:")) for error in errors):
        invalid.update(ids)
    return invalid


def _enrich_split(
    clips: list[dict],
    topic: str,
    *,
    deadline: float,
    cancelled: CancelledCb,
) -> tuple[list[dict], list[dict], list[str], str | None]:
    calls: list[dict] = []
    fallback_reasons: list[str] = []
    flash_configuration_error: str | None = None
    system, user = _enrichment_prompts(clips, topic)
    try:
        plan, call = _call_model(
            system,
            user,
            _EnrichmentPlan,
            model=config.SEGMENT_FLASH_MODEL,
            thinking_level="low",
            max_output_tokens=_ENRICH_OUTPUT_TOKENS,
            timeout_s=_FLASH_ENRICH_TIMEOUT_S,
            deadline_monotonic=deadline,
            operation="flash_grounded_enrichment",
            prompt_version=FLASH_SPLIT_PROFILE,
            cancelled=cancelled,
        )
        calls.append(call)
        errors = _apply_enrichment(clips, plan, topic)
    except Exception as exc:  # schema/transport failure makes every enrichment item invalid
        telemetry = _exception_telemetry(exc)
        if telemetry:
            calls.append(telemetry)
        error = f"{type(exc).__name__}: {exc}"
        if _flash_configuration_failure(error):
            flash_configuration_error = error
        errors = [f"{clip['_clip_id']}:flash_enrichment_failure" for clip in clips]

    invalid_ids = _invalid_enrichment_clip_ids(errors, clips)
    if not invalid_ids:
        return clips, calls, fallback_reasons, flash_configuration_error

    fallback_reasons.extend(f"invalid_enrichment:{clip_id}" for clip_id in sorted(invalid_ids))
    retry_clips = [clip for clip in clips if clip["_clip_id"] in invalid_ids]
    # Clear invalid drafts before the Pro retry; if Pro also fails the valid clip survives
    # with the existing optional learning-detail behavior.
    for clip in retry_clips:
        clip.update({"summary": "", "takeaways": [], "match_reason": "", "assessment": None})
    system, user = _enrichment_prompts(retry_clips, topic)
    try:
        plan, call = _call_model(
            system,
            user,
            _EnrichmentPlan,
            model=config.SEGMENT_PRO_MODEL,
            thinking_level="high",
            max_output_tokens=_ENRICH_OUTPUT_TOKENS,
            timeout_s=_PRO_TIMEOUT_S,
            deadline_monotonic=deadline,
            operation="pro_enrichment_fallback",
            prompt_version=CORRECTED_PRO_PROFILE,
            cancelled=cancelled,
        )
        calls.append(call)
        pro_errors = _apply_enrichment(retry_clips, plan, topic)
        if pro_errors:
            fallback_reasons.append("pro_enrichment_failure")
    except Exception as exc:
        telemetry = _exception_telemetry(exc)
        if telemetry:
            calls.append(telemetry)
        fallback_reasons.append("pro_enrichment_failure")
    return clips, calls, fallback_reasons, flash_configuration_error


def run_segment_profile(
    transcript: dict,
    settings: dict,
    profile: str,
    *,
    topic: str = "",
    deadline_monotonic: float | None = None,
    cancelled: CancelledCb = None,
) -> SegmentResult:
    """Run one immutable benchmark profile without production routing."""
    segments = transcript.get("segments") or []
    if not segments:
        return SegmentResult([], "No transcript segments to segment.", profile, "invalid",
                             ["missing_segments"], proposed_count=0, accepted_count=0)
    deadline = deadline_monotonic or (time.monotonic() + _TOTAL_DEADLINE_S)
    try:
        report, classification, calls = _run_selection_profile(
            profile, transcript, topic, settings, deadline=deadline, cancelled=cancelled,
        )
        fallback_reasons: list[str] = []
        flash_configuration_error: str | None = None
        if (profile == FLASH_SPLIT_PROFILE and classification.status == "green"
                and report.clips):
            (report.clips, enrichment_calls, fallback_reasons,
             flash_configuration_error) = _enrich_split(
                 report.clips, topic, deadline=deadline, cancelled=cancelled,
             )
            calls.extend(enrichment_calls)
        clips = _public_clips(report.clips)
        notes = f"{len(clips)} topic clip(s) from {len(segments)} transcript segments."
        return SegmentResult(
            clips,
            notes,
            profile,
            classification.status,
            list(classification.reasons),
            fallback_reasons,
            calls,
            report.proposed_count,
            len(clips),
            flash_configuration_error=flash_configuration_error,
        )
    except Exception as exc:  # callers decide whether an invalid profile should fall back
        call = _exception_telemetry(exc)
        calls = [call] if call else []
        return SegmentResult(
            [],
            "Segmentation model call failed.",
            profile,
            "invalid",
            [f"request_failure:{type(exc).__name__}"],
            calls=calls,
            error=f"{type(exc).__name__}: {exc}",
        )


def _hybrid_selected(video_id: str, percent: float) -> bool:
    if not video_id or percent <= 0.0:
        return False
    if percent >= 100.0:
        return True
    bucket = int.from_bytes(hashlib.sha256(video_id.encode("utf-8")).digest()[:8], "big") % 10_000
    return bucket < int(percent * 100)


def _flash_configuration_failure(error: str | None) -> bool:
    text = str(error or "").casefold()
    return bool(text) and any(marker in text for marker in (
        "status 400", "status 401", "status 403", "status 404",
        "400 invalid_argument", "401 unauthenticated", "403 permission_denied",
        "404 not_found", "invalid_argument", "permission_denied", "not_found",
        "api key", "permission", "model not found", "model is not found",
        "unsupported model",
        "requires an explicit gemini 3 model", "invalid model",
    ))


def _disable_flash(reason: str) -> None:
    global _flash_disabled_reason
    with _flash_disable_lock:
        if _flash_disabled_reason is None:
            _flash_disabled_reason = reason


def _flash_disable_reason() -> str | None:
    with _flash_disable_lock:
        return _flash_disabled_reason


def _authoritative_pro(transcript: dict, settings: dict, topic: str, deadline: float,
                       cancelled: CancelledCb, *, fallback: bool = False) -> SegmentResult:
    profile = AUTHORITATIVE_PRO_PROFILE
    result = run_segment_profile(
        transcript, settings, profile, topic=topic,
        deadline_monotonic=deadline, cancelled=cancelled,
    )
    operation = "pro_fallback" if fallback else "pro_authoritative"
    for call in result.calls:
        call["operation"] = operation
    return result


def segment_clips_detailed(
    transcript: dict,
    settings: dict,
    *,
    topic: str = "",
    video_id: str = "",
    progress: ProgressCb = None,
    routing_mode: str | None = None,
) -> SegmentResult:
    segments = transcript.get("segments") or []
    if not segments:
        return SegmentResult([], "No transcript segments to segment.", "none", "invalid",
                             ["missing_segments"])

    configured_mode = str(routing_mode or config.SEGMENT_ROUTING_MODE).lower()
    mode = configured_mode
    if mode not in {"pro_only", "shadow", "hybrid"}:
        mode = "pro_only"
    disabled_reason = _flash_disable_reason()
    if disabled_reason is not None and mode in {"shadow", "hybrid"}:
        mode = "pro_only"
    percent = float(config.SEGMENT_HYBRID_PERCENT)
    selected = mode == "hybrid" and _hybrid_selected(video_id, percent)
    route = "flash_first" if selected else "pro_authoritative"
    sink = settings.get("_segment_telemetry") if isinstance(settings, dict) else None
    cancelled = settings.get("_segment_cancelled") if isinstance(settings, dict) else None
    deadline = time.monotonic() + _TOTAL_DEADLINE_S
    prompt_version = (PRODUCTION_FLASH_PROFILE
                      if selected or mode == "shadow"
                      else AUTHORITATIVE_PRO_PROFILE)
    _emit(sink, "route_selected", video_id=video_id or None, mode=mode,
          configured_mode=configured_mode, route=route, prompt_version=prompt_version,
          hybrid_percent=percent, flash_disabled_reason=disabled_reason)
    if progress and not _cancel_requested(cancelled):
        progress(0.1, "Understanding the transcript…")

    result: SegmentResult
    if mode == "shadow":
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="segment-shadow")
        flash_future = pool.submit(
            run_segment_profile, transcript, settings, PRODUCTION_FLASH_PROFILE,
            topic=topic, deadline_monotonic=deadline, cancelled=cancelled,
        )

        def finish_shadow(future) -> None:
            try:
                shadow = future.result()
            except Exception as exc:  # defensive: run_segment_profile normally captures errors
                shadow = SegmentResult(
                    [], "Shadow segmentation failed.", PRODUCTION_FLASH_PROFILE,
                    "invalid", [f"request_failure:{type(exc).__name__}"],
                    error=f"{type(exc).__name__}: {exc}",
                )
            for call in shadow.calls:
                _emit(sink, "model_call", video_id=video_id or None, shadow=True, **call)
            for reason in shadow.fallback_reasons:
                _emit(sink, "pro_fallback", video_id=video_id or None,
                      shadow=True, reason=reason)
            shadow_cost = sum(_model_cost(call) for call in shadow.calls)
            shadow_accepted = shadow.accepted_count
            _emit(
                sink,
                "shadow_comparison",
                video_id=video_id or None,
                classification=shadow.classification,
                reasons=shadow.classification_reasons,
                proposed_count=shadow.proposed_count,
                accepted_count=shadow_accepted,
                fallback_reasons=shadow.fallback_reasons,
                fallback_rate=1.0 if shadow.fallback_reasons else 0.0,
                pricing_version=_PRICING_VERSION,
                estimated_cost_usd=round(shadow_cost, 8),
                cost_per_accepted_clip_usd=(
                    round(shadow_cost / shadow_accepted, 8) if shadow_accepted else None
                ),
                error=shadow.error,
            )
            flash_error = shadow.flash_configuration_error or shadow.error
            if _flash_configuration_failure(flash_error):
                _disable_flash(str(flash_error))
                _emit(sink, "route_rollback", video_id=video_id or None,
                      reason="flash_model_access_or_configuration_failure")

        flash_future.add_done_callback(finish_shadow)
        pool.shutdown(wait=False)
        result = _authoritative_pro(
            transcript, settings, topic, deadline, cancelled,
        )
        result.route = "shadow_pro_authoritative"
    elif selected:
        flash = run_segment_profile(
            transcript, settings, PRODUCTION_FLASH_PROFILE, topic=topic,
            deadline_monotonic=deadline, cancelled=cancelled,
        )
        _emit(sink, "flash_classified", video_id=video_id or None,
              classification=flash.classification, reasons=flash.classification_reasons,
              proposed_count=flash.proposed_count, accepted_count=flash.accepted_count)
        flash_error = flash.flash_configuration_error or flash.error
        if _flash_configuration_failure(flash_error):
            _disable_flash(str(flash_error))
            _emit(sink, "route_rollback", video_id=video_id or None,
                  reason="flash_model_access_or_configuration_failure")
        accept_partial_flash = bool(
            isinstance(settings, dict)
            and settings.get("segment_accept_partial_flash")
            and flash.clips
            and not flash.error
        )
        if (flash.classification == "green" and not flash.error) or accept_partial_flash:
            result = flash
            result.route = "hybrid_flash"
        else:
            fallback_reasons = list(flash.classification_reasons) or ["flash_request_failure"]
            for reason in fallback_reasons:
                _emit(sink, "pro_fallback", video_id=video_id or None, reason=reason)
            pro = _authoritative_pro(
                transcript, settings, topic, deadline, cancelled, fallback=True,
            )
            # Never expose uncertain/invalid Flash when Pro fails.
            result = pro
            result.route = "hybrid_pro_fallback"
            result.fallback_reasons = fallback_reasons
            result.calls = flash.calls + pro.calls
    else:
        result = _authoritative_pro(
            transcript, settings, topic, deadline, cancelled,
        )
        result.route = "hybrid_control_pro" if mode == "hybrid" else "pro_only"

    cancelled_now = _cancel_requested(cancelled)
    if progress and not cancelled_now:
        progress(0.85, "Placing clip boundaries…")
    total_cost = sum(_model_cost(call) for call in result.calls)
    accepted = len(result.clips)
    fallback_rate = 1.0 if result.fallback_reasons else 0.0
    for call in result.calls:
        _emit(sink, "model_call", video_id=video_id or None, **call)
    _emit(
        sink,
        "segment_completed" if not result.error else "segment_error",
        video_id=video_id or None,
        route=result.route,
        classification=result.classification,
        classification_reasons=result.classification_reasons,
        fallback_reasons=result.fallback_reasons,
        proposed_count=result.proposed_count,
        accepted_count=accepted,
        zero_output=(accepted == 0),
        fallback_rate=fallback_rate,
        pricing_version=_PRICING_VERSION,
        estimated_cost_usd=round(total_cost, 8),
        cost_per_accepted_clip_usd=(round(total_cost / accepted, 8) if accepted else None),
        error=result.error,
    )
    if progress and not _cancel_requested(cancelled):
        progress(1.0, f"{accepted} clip(s) ready")
    return result


def segment_clips(
    transcript: dict,
    settings: dict,
    progress: ProgressCb = None,
    topic: str = "",
    video_id: str = "",
) -> tuple[list[dict], str]:
    """Return guarded educational clips while preserving the existing public tuple."""
    result = segment_clips_detailed(
        transcript,
        settings,
        topic=topic,
        video_id=video_id,
        progress=progress,
    )
    return result.clips, result.notes
