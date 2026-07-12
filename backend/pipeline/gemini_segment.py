"""Guarded Gemini educational clip segmentation.

Production starts with a guarded Flash-first canary. Hybrid mode admits only
deterministic ``green`` Gemini 3.5 Flash results and lets the generation-wide
confidence gate decide whether an uncertain request may use the single Pro
fallback. Shadow and Pro-only modes remain available as explicit overrides.

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
PRO_BOUNDARY_PROFILE = "pro_boundary_v1"
# Production Flash performs only the compact, quality-critical boundary choice.
PRODUCTION_FLASH_PROFILE = FLASH_SPLIT_PROFILE
# Authoritative and fallback Pro routes use the same compact boundary contract.
# Legacy profiles remain readable for old cache/test compatibility only.
AUTHORITATIVE_PRO_PROFILE = PRO_BOUNDARY_PROFILE
SEGMENT_PROFILES = (
    PRODUCTION_PRO_PROFILE,
    CORRECTED_PRO_PROFILE,
    FLASH_SINGLE_PROFILE,
    FLASH_SPLIT_PROFILE,
    PRO_BOUNDARY_PROFILE,
)

_TOTAL_DEADLINE_S = 150.0
_FLASH_SINGLE_TIMEOUT_S = 45.0
_FLASH_BOUNDARY_TIMEOUT_S = 45.0
_FLASH_REPAIR_TIMEOUT_S = 20.0
_FLASH_ENRICH_TIMEOUT_S = 25.0
_PRO_TIMEOUT_S = 90.0
_SELECTION_OUTPUT_TOKENS = 24_576
_BOUNDARY_OUTPUT_TOKENS = 4_096
_BOUNDARY_REPAIR_OUTPUT_TOKENS = 1_024
_ENRICH_OUTPUT_TOKENS = 2_048
_MIN_CLIP_S = 1.0
_MAX_CLIP_S = 180.0
_UNCERTAIN_DURATION_S = 150.0
_GREEN_SCORE = 0.75
_MAX_CLIPS = 40
_PRODUCTION_MAX_CANDIDATES = 8
_DUPLICATE_OVERLAP = 0.8
_CONTEXT_CUE_LIMIT = 2
_CONTEXT_WINDOW_S = 30.0
_BOUNDARY_PAD_S = 0.3
_LONG_RANGE_REPAIR_MIN_S = 20.0
_LONG_RANGE_REPAIR_TARGET_S = 75.0
_LONG_RANGE_REPAIR_MAX_S = 150.0
_LONG_RANGE_REPAIR_START_SCAN_S = 30.0
_REPAIR_NEIGHBOR_CUES = 2
_BOUNDARY_REPAIR_PROMPT_VERSION = "boundary_repair_v1"
_CARD_ENRICHMENT_PROMPT_VERSION = "accepted_clip_enrichment_v1"

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
    candidate_id: _NonBlank
    start_line: int = Field(ge=0, strict=True)
    end_line: int = Field(ge=0, strict=True)
    start_quote: _NonBlank
    end_quote: _NonBlank
    title: _NonBlank
    learning_objective: _NonBlank
    facet: _NonBlank
    reason: _NonBlank
    informativeness: float = Field(ge=0.0, le=1.0, strict=True)
    topic_relevance: float = Field(ge=0.0, le=1.0, strict=True)
    educational_importance: float = Field(ge=0.0, le=1.0, strict=True)
    difficulty: float = Field(ge=0.0, le=1.0, strict=True)
    self_contained: bool = Field(strict=True)
    is_standalone: bool = Field(strict=True)
    prerequisite_candidate_ids: list[_NonBlank] = Field(max_length=8)
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


class _BoundaryRepairItem(_StrictModel):
    candidate_id: _NonBlank
    start_line: int = Field(ge=0, strict=True)
    end_line: int = Field(ge=0, strict=True)
    start_quote: _NonBlank
    end_quote: _NonBlank


class _BoundaryRepairPlan(_StrictModel):
    items: list[_BoundaryRepairItem] = Field(max_length=_PRODUCTION_MAX_CANDIDATES)


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


class _CardEnrichmentItem(_StrictModel):
    clip_id: _NonBlank
    summary: _NonBlank
    takeaways: list[_NonBlank] = Field(min_length=2, max_length=4)
    match_reason: _NonBlank


class _CardEnrichmentPlan(_StrictModel):
    items: list[_CardEnrichmentItem] = Field(max_length=3)


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
- Prefer fewer clips to forcing an incomplete idea.
- Contextual overlap is allowed only when both clips remain independently complete.
- Copy exact transcript line IDs and exact opening/closing quotes.
- A complete clip may be 1 to 180 seconds; prefer focused 20 to 90 second units.
- Include the setup or question needed to understand the teaching and the conclusion,
  answer, or worked result that completes it.
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
        "topic, and make each learning objective name the relevant idea."
    )


def _selection_fields(*, enriched: bool) -> str:
    fields = (
        "candidate_id, start_line, end_line, start_quote, end_quote, title, "
        "learning_objective, facet, reason, informativeness, topic_relevance, "
        "educational_importance, difficulty, self_contained, is_standalone, "
        "prerequisite_candidate_ids, uncertainty, uncertainty_reasons"
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


def _boundary_prompts(
    lines: str,
    n: int,
    topic: str = "",
    *,
    max_candidates: int = _PRODUCTION_MAX_CANDIDATES,
) -> tuple[str, str]:
    system = (
        "You select self-contained educational clip boundaries from timestamped transcripts.\n\n"
        + _POLICY_AND_EXAMPLES
    )
    user = (
        f"{_topic_rule(topic)}\nLine IDs must be between 0 and {n - 1}.\n\n"
        f"Transcript ({n} lines, formatted `[index] MM:SS text`):\n{lines}\n\n"
        "Choose the strongest educational moments globally from anywhere in the transcript. "
        "Do not favor the beginning and do not return every chronological section. Rank the "
        "strongest, most educational, self-contained moments first; sequencing happens later. "
        "For every moment, verify the displayed start/end timestamps before returning it: the "
        "selected source span must be at most 150 seconds and should usually be 20 to 90 seconds. "
        "The shorter limit leaves room for deterministic context repair while keeping the final "
        "clip inside the hard 180-second envelope. If a section is longer, choose one smaller "
        "complete sub-explanation or omit it; never return the whole long section. "
        f"Return at most {max(1, min(_PRODUCTION_MAX_CANDIDATES, int(max_candidates)))} "
        "moments, and only when boundaries and context are low-uncertainty. "
        "Use a unique candidate_id for every moment. A non-standalone moment must list the "
        "candidate_id(s) that provide its required context; a standalone moment must list no "
        "prerequisites. "
        f"Every item must contain {_selection_fields(enriched=False)}. Learning details and "
        "assessments are generated later, so do not include them."
    )
    return system, user


def _boundary_repair_prompts(
    candidates: list["_BoundaryRepairCandidate"],
    segments: list[dict],
    topic: str,
) -> tuple[str, str, dict[str, tuple[set[int], set[int]]]]:
    """Render only the neighboring cue windows needed to repair dirty edges."""
    system = (
        "You repair transcript-cue boundaries for already selected educational moments. "
        "Use only the displayed neighboring cues. Return at most one item per candidate, "
        "omit a candidate when no clean self-contained boundary exists, and copy each edge "
        "quote exactly from its selected cue. Do not summarize, enrich, or add assessments."
    )
    blocks: list[str] = []
    allowed: dict[str, tuple[set[int], set[int]]] = {}
    n = len(segments)
    for candidate in candidates:
        start_lines = set(range(
            max(0, candidate.start_line - _REPAIR_NEIGHBOR_CUES),
            min(n, candidate.start_line + _REPAIR_NEIGHBOR_CUES + 1),
        ))
        end_lines = set(range(
            max(0, candidate.end_line - _REPAIR_NEIGHBOR_CUES),
            min(n, candidate.end_line + _REPAIR_NEIGHBOR_CUES + 1),
        ))
        allowed[candidate.candidate_id] = (start_lines, end_lines)

        def render(indices: set[int]) -> str:
            return "\n".join(
                f"[{index}] {_mmss(segments[index].get('start', 0.0))} "
                f"{str(segments[index].get('text') or '').strip()}"
                for index in sorted(indices)
            )

        blocks.append(
            f"<candidate id={candidate.candidate_id!r} failed_check={candidate.reason!r}>\n"
            f"title: {candidate.proposal.title}\n"
            f"learning objective: {candidate.proposal.reason}\n"
            f"original cue range: {candidate.proposal.start_line}-{candidate.proposal.end_line}\n"
            f"allowed start_line IDs: {sorted(start_lines)}\n"
            f"<start_neighbors>\n{render(start_lines)}\n</start_neighbors>\n"
            f"allowed end_line IDs: {sorted(end_lines)}\n"
            f"<end_neighbors>\n{render(end_lines)}\n</end_neighbors>\n"
            "</candidate>"
        )
    user = (
        f"Viewer topic: {topic.strip() or '(none)'}.\n\n"
        + "\n\n".join(blocks)
        + "\n\nRepair only the preceding candidates. For each safe repair return "
          "candidate_id, start_line, end_line, start_quote, and end_quote. Each line ID "
          "must come from that candidate's corresponding allowed list."
    )
    return system, user, allowed


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


def _card_enrichment_prompts(items: list[dict], topic: str) -> tuple[str, str]:
    system = (
        "Enrich already accepted educational clips using only each supplied transcript "
        "excerpt. Do not create quizzes, assessments, outside facts, or chain-of-thought."
    )
    blocks = []
    for item in items[:3]:
        blocks.append(
            f"<clip id={str(item.get('clip_id') or '')!r}>\n"
            f"Title: {str(item.get('title') or '').strip()}\n"
            f"Learning objective: {str(item.get('learning_objective') or '').strip()}\n"
            f"Transcript: {str(item.get('text') or '').strip()}\n"
            "</clip>"
        )
    user = (
        f"Viewer topic: {topic.strip() or '(none)'}.\n\n"
        + "\n\n".join(blocks)
        + "\n\nReturn one item per clip_id with a grounded 1-2 sentence summary, "
          "2-4 distinct grounded takeaways, and a short topic-specific match_reason."
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


def _guard_text(text: str, *, ignore_caption_case: bool) -> str:
    """Remove only the unreliable lowercase signal from auto-caption guards."""
    normalized = str(text or "").strip()
    if ignore_caption_case and normalized[:1].islower():
        normalized = normalized[:1].upper() + normalized[1:]
    return normalized


def _cue_opens_mid_thought(text: str, *, ignore_caption_case: bool) -> bool:
    from .discourse import opens_mid_thought

    return opens_mid_thought(
        _guard_text(text, ignore_caption_case=ignore_caption_case)
    )


def _cue_has_weak_end(
    text: str,
    next_text: str,
    *,
    ignore_caption_case: bool,
) -> bool:
    """Use the existing weak-end guard and continuation onset as cue evidence."""
    from .refine import _is_weak_end
    from .sentences import Sentence, classify_terminator

    guarded = _guard_text(text, ignore_caption_case=ignore_caption_case)
    terminator = classify_terminator(guarded)
    sentence = Sentence(
        idx=0,
        text=guarded,
        start=0.0,
        end=1.0,
        terminator=terminator,
        ends_with_period=bool(terminator),
        word_start_idx=0,
        word_end_idx=0,
        align_confidence=1.0,
    )
    if _is_weak_end(sentence):
        return True
    if terminator or not next_text:
        return False
    # Supadata's fixed-size cues frequently split an ordinary phrase without
    # punctuation or an explicit conjunction. Long bare edges and gerund-led
    # constructions are therefore uncertain even when the next cue starts with
    # a capitalized noun phrase (for example, "by substituting / the numbers").
    raw_words = _toks(guarded)
    explicit_closure_words = {
        "answer", "answered", "complete", "completed", "completely", "conclusion",
        "end", "ended", "final", "finished", "finishes", "result", "solved",
    }
    if explicit_closure_words.intersection(raw_words[-5:]):
        return False
    next_words = _toks(next_text)
    if (
        len(raw_words) >= 2
        and raw_words[-1].endswith("ing")
        and len(raw_words[-1]) > 5
        and raw_words[-2]
        in {"after", "are", "before", "being", "by", "is", "was", "were", "while"}
    ):
        return True
    if len(next_words) >= 2 and next_words[1] in {
        "that", "when", "where", "which", "whose",
    }:
        return True
    if not _cue_opens_mid_thought(
        next_text, ignore_caption_case=ignore_caption_case
    ):
        return False
    # A bare auto-caption edge needs expansion only when the following cue has
    # an explicit dependency signal. The onset guard's lowercase/short-text
    # fallbacks are intentionally insufficient by themselves here.
    from .discourse import ANAPHORS, CONTEXT_DEP_HEADS, CONTINUATION_MARKERS

    words = next_words
    if not words:
        return False
    return bool(
        words[0] in CONTINUATION_MARKERS
        or words[0] in ANAPHORS
        or (len(words) > 1 and words[0] == "the" and words[1] in CONTEXT_DEP_HEADS)
    )


def _cue_boundary_confidence(text: str, *, ignore_caption_case: bool) -> float:
    from .sentences import classify_terminator

    guarded = _guard_text(text, ignore_caption_case=ignore_caption_case)
    return 1.0 if classify_terminator(guarded) else 0.90


def _close_cue_context(
    segments: list[dict],
    start_line: int,
    end_line: int,
    *,
    ignore_caption_case: bool,
    cue_limit: int = _CONTEXT_CUE_LIMIT,
) -> tuple[int, int, str | None]:
    """Expand dirty edges by at most two cues and thirty seconds per side."""
    cue_limit = max(0, min(_CONTEXT_CUE_LIMIT, int(cue_limit)))
    original_start = start_line
    original_end = end_line
    for _ in range(cue_limit):
        if not _cue_opens_mid_thought(
            str(segments[start_line].get("text") or ""),
            ignore_caption_case=ignore_caption_case,
        ):
            break
        candidate = start_line - 1
        if candidate < 0:
            break
        movement = (
            float(segments[original_start].get("start", 0.0))
            - float(segments[candidate].get("start", 0.0))
        )
        if movement > _CONTEXT_WINDOW_S + 1e-9:
            break
        start_line = candidate
    if _cue_opens_mid_thought(
        str(segments[start_line].get("text") or ""),
        ignore_caption_case=ignore_caption_case,
    ):
        return start_line, end_line, "unresolved_weak_start"

    for _ in range(cue_limit):
        next_text = (
            str(segments[end_line + 1].get("text") or "")
            if end_line + 1 < len(segments)
            else ""
        )
        if not _cue_has_weak_end(
            str(segments[end_line].get("text") or ""),
            next_text,
            ignore_caption_case=ignore_caption_case,
        ):
            break
        candidate = end_line + 1
        if candidate >= len(segments):
            break
        movement = (
            float(segments[candidate].get("end", 0.0))
            - float(segments[original_end].get("end", 0.0))
        )
        if movement > _CONTEXT_WINDOW_S + 1e-9:
            break
        end_line = candidate
    next_text = (
        str(segments[end_line + 1].get("text") or "")
        if end_line + 1 < len(segments)
        else ""
    )
    if _cue_has_weak_end(
        str(segments[end_line].get("text") or ""),
        next_text,
        ignore_caption_case=ignore_caption_case,
    ):
        return start_line, end_line, "unresolved_weak_end"
    return start_line, end_line, None


def _padded_cue_bounds(
    segments: list[dict], start_line: int, end_line: int,
) -> tuple[float, float]:
    """Add 300 ms room without crossing the midpoint to adjacent speech."""
    start = float(segments[start_line].get("start", 0.0))
    end = float(segments[end_line].get("end", start))
    if start_line > 0:
        previous_end = float(segments[start_line - 1].get("end", start))
        if previous_end <= start:
            start = max(start - _BOUNDARY_PAD_S, (previous_end + start) / 2.0)
    else:
        start = max(0.0, start - _BOUNDARY_PAD_S)
    if end_line + 1 < len(segments):
        next_start = float(segments[end_line + 1].get("start", end))
        if next_start >= end:
            end = min(end + _BOUNDARY_PAD_S, (end + next_start) / 2.0)
    else:
        end = min(end + _BOUNDARY_PAD_S, float(segments[-1].get("end", end)))
    return start, end


def _repair_oversized_cue_range(
    segments: list[dict],
    start_line: int,
    end_line: int,
    *,
    ignore_caption_case: bool,
    anchor_text: str = "",
) -> tuple[int, int] | None:
    """Choose a complete cue-level subunit without making another model call."""
    anchor_tokens = _content_tokens(anchor_text)
    original_start = float(segments[start_line].get("start", 0.0))
    candidates: list[tuple[float, float, float, float, int, int]] = []
    for candidate_start in range(start_line, end_line + 1):
        candidate_start_time = float(
            segments[candidate_start].get("start", original_start)
        )
        start_shift = candidate_start_time - original_start
        if start_shift > _LONG_RANGE_REPAIR_START_SCAN_S:
            break
        if _cue_opens_mid_thought(
            str(segments[candidate_start].get("text") or ""),
            ignore_caption_case=ignore_caption_case,
        ):
            continue
        opening_segments = segments[
            candidate_start:min(end_line + 1, candidate_start + 3)
        ]
        anchor_overlap = sum(
            (3 - offset)
            * len(
                anchor_tokens
                & _content_tokens(str(segment.get("text") or ""))
            )
            for offset, segment in enumerate(opening_segments)
        )
        for candidate_end in range(candidate_start, end_line + 1):
            start, end = _padded_cue_bounds(
                segments, candidate_start, candidate_end
            )
            duration = end - start
            if duration > _LONG_RANGE_REPAIR_MAX_S:
                break
            if duration < _LONG_RANGE_REPAIR_MIN_S:
                continue
            next_text = (
                str(segments[candidate_end + 1].get("text") or "")
                if candidate_end + 1 < len(segments)
                else ""
            )
            if _cue_has_weak_end(
                str(segments[candidate_end].get("text") or ""),
                next_text,
                ignore_caption_case=ignore_caption_case,
            ):
                continue
            candidates.append((
                -float(anchor_overlap),
                start_shift,
                abs(duration - _LONG_RANGE_REPAIR_TARGET_S),
                -duration,
                candidate_start,
                candidate_end,
            ))
    if not candidates:
        return None
    best = min(candidates)
    return best[4], best[5]


def _cue_clip_text(segments: list[dict], start_line: int, end_line: int) -> str:
    return " ".join(
        str(segment.get("text") or "").strip()
        for segment in segments[start_line:end_line + 1]
        if str(segment.get("text") or "").strip()
    ).strip()


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
    repair_candidates: list["_BoundaryRepairCandidate"] = field(default_factory=list)

    @property
    def accepted_count(self) -> int:
        return len(self.clips)


@dataclass(frozen=True)
class _BoundaryRepairCandidate:
    candidate_id: str
    prefix: str
    proposal: _BoundaryTopic
    start_line: int
    end_line: int
    reason: str


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


def _configured_clip_limit(settings: dict) -> int:
    configured = settings.get("max_clips")
    limit = config.SEGMENT_MAX_CLIPS if configured is None else int(configured)
    return max(0, min(_MAX_CLIPS, limit))


def _finalize_clips(clips: list[dict], settings: dict) -> list[dict]:
    """Dedupe and limit while candidates remain quality-ranked."""
    quality_order = sorted(
        clips,
        key=lambda clip: (
            (
                0.45 * float(clip["topic_relevance"])
                + 0.35 * float(clip["educational_importance"])
                + 0.20 * float(clip["informativeness"])
            )
            - (0.08 if clip.get("uncertainty") == "medium" else 0.0),
            -(clip["end"] - clip["start"]),
        ),
        reverse=True,
    )
    kept: list[dict] = []
    for candidate in quality_order:
        if not any(_near_duplicate(candidate, prior) for prior in kept):
            kept.append(candidate)
    limit = _configured_clip_limit(settings)
    by_candidate_id = {
        str(clip.get("selection_candidate_id") or ""): clip
        for clip in kept
        if str(clip.get("selection_candidate_id") or "")
    }

    def prerequisite_closure(
        candidate_id: str,
        trail: set[str],
    ) -> list[dict] | None:
        if candidate_id in trail:
            return None
        clip = by_candidate_id.get(candidate_id)
        if clip is None:
            return None
        closure: list[dict] = []
        for prerequisite in clip.get("prerequisite_ids") or []:
            prerequisite_items = prerequisite_closure(
                str(prerequisite),
                {*trail, candidate_id},
            )
            if prerequisite_items is None:
                return None
            closure.extend(prerequisite_items)
        closure.append(clip)
        deduped: list[dict] = []
        seen_ids: set[str] = set()
        for item in closure:
            item_id = str(item.get("selection_candidate_id") or "")
            if item_id and item_id not in seen_ids:
                seen_ids.add(item_id)
                deduped.append(item)
        return deduped

    selected: list[dict] = []
    selected_ids: set[str] = set()
    for candidate in kept:
        candidate_id = str(candidate.get("selection_candidate_id") or "")
        bundle = prerequisite_closure(candidate_id, set()) if candidate_id else [candidate]
        if bundle is None:
            continue
        additions = [
            item
            for item in bundle
            if str(item.get("selection_candidate_id") or "") not in selected_ids
        ]
        if len(selected) + len(additions) > limit:
            continue
        for item in additions:
            selected.append(item)
            item_id = str(item.get("selection_candidate_id") or "")
            if item_id:
                selected_ids.add(item_id)
        if len(selected) >= limit:
            break
    selected.sort(key=lambda clip: (clip["start"], clip["end"]))
    for index, clip in enumerate(selected):
        clip["sequence_index"] = index + 1
    return selected


def _drop_unmet_prerequisite_clips(report: _Conversion) -> None:
    """Fail closed on unknown or cyclic selector dependencies before shipping."""
    by_id = {
        str(clip.get("selection_candidate_id") or ""): clip
        for clip in report.clips
        if str(clip.get("selection_candidate_id") or "")
    }
    resolved = {
        candidate_id
        for candidate_id, clip in by_id.items()
        if bool(clip.get("is_standalone")) and not clip.get("prerequisite_ids")
    }
    changed = True
    while changed:
        changed = False
        for candidate_id, clip in by_id.items():
            if candidate_id in resolved:
                continue
            prerequisites = {
                str(value)
                for value in (clip.get("prerequisite_ids") or [])
                if str(value)
            }
            if prerequisites and prerequisites.issubset(resolved):
                resolved.add(candidate_id)
                changed = True
    if len(resolved) == len(by_id):
        return
    removed = set(by_id) - resolved
    report.clips = [
        clip
        for clip in report.clips
        if str(clip.get("selection_candidate_id") or "") in resolved
    ]
    for candidate_id in sorted(removed):
        report.rejected_reasons.append(
            f"candidate_{candidate_id}:unmet_or_cyclic_prerequisite"
        )
    for index, clip in enumerate(report.clips):
        clip["sequence_index"] = index + 1


def _plan_to_report(
    plan: _Plan | _BoundaryPlan | _LegacyPlan | _ProductionPlan,
    segments: list[dict],
    words: list[dict],
    settings: dict,
    *,
    topic: str = "",
    require_enrichment: bool = False,
    context_cue_limit: int = _CONTEXT_CUE_LIMIT,
) -> _Conversion:
    report = _Conversion(proposed_count=len(plan.topics))
    n = len(segments)
    if not n:
        report.rejected_reasons.append("missing_segments")
        return report

    ignore_caption_case = bool(settings.get("_segment_ignore_caption_case", True))
    raw: list[dict] = []
    seen_candidate_ids: set[str] = set()

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
        start_quote = str(proposal.start_quote or "").strip()
        end_quote = str(proposal.end_quote or "").strip()
        if not _contains_quote(str(segments[a].get("text") or ""), start_quote):
            report.rejected_reasons.append(f"{prefix}:bad_start_quote")
            continue
        if not _contains_quote(str(segments[b].get("text") or ""), end_quote):
            report.rejected_reasons.append(f"{prefix}:bad_end_quote")
            continue
        info = _strict_score(proposal.informativeness)
        relevance = _strict_score(proposal.topic_relevance)
        raw_importance = getattr(proposal, "educational_importance", None)
        importance = (
            _strict_score(raw_importance)
            if raw_importance is not None
            else (
                round((float(info) + float(relevance)) / 2.0, 3)
                if info is not None and relevance is not None
                else None
            )
        )
        difficulty = _strict_score(proposal.difficulty)
        if info is None or relevance is None or importance is None or difficulty is None:
            report.rejected_reasons.append(f"{prefix}:score_out_of_range")
            continue
        if proposal.self_contained is not True:
            report.rejected_reasons.append(f"{prefix}:not_self_contained")
            continue
        candidate_id = " ".join(
            str(
                getattr(proposal, "candidate_id", "")
                or f"clip-{index + 1:03d}-{proposal.start_line}-{proposal.end_line}"
            ).split()
        )
        if candidate_id in seen_candidate_ids:
            report.rejected_reasons.append(f"{prefix}:duplicate_candidate_id")
            continue
        seen_candidate_ids.add(candidate_id)
        prerequisites = list(dict.fromkeys(
            " ".join(str(value or "").split())
            for value in (getattr(proposal, "prerequisite_candidate_ids", None) or [])
            if " ".join(str(value or "").split())
        ))
        is_standalone = bool(
            getattr(proposal, "is_standalone", proposal.self_contained)
        )
        if (is_standalone and prerequisites) or (not is_standalone and not prerequisites):
            report.rejected_reasons.append(f"{prefix}:inconsistent_prerequisites")
            continue
        uncertainty = str(getattr(proposal, "uncertainty", "low") or "low")
        uncertainty_reasons = [str(getattr(reason, "value", reason))
                               for reason in (getattr(proposal, "uncertainty_reasons", None) or [])]
        if uncertainty == "high":
            report.rejected_reasons.append(f"{prefix}:{uncertainty}_uncertainty")
            continue

        a, b, closure_error = _close_cue_context(
            segments,
            a,
            b,
            ignore_caption_case=ignore_caption_case,
            cue_limit=context_cue_limit,
        )
        if closure_error:
            report.rejected_reasons.append(f"{prefix}:{closure_error}")
            if isinstance(proposal, _BoundaryTopic):
                report.repair_candidates.append(_BoundaryRepairCandidate(
                    candidate_id=candidate_id,
                    prefix=prefix,
                    proposal=proposal,
                    start_line=a,
                    end_line=b,
                    reason=closure_error,
                ))
            continue
        start, end = _padded_cue_bounds(segments, a, b)
        if end <= start:
            report.rejected_reasons.append(f"{prefix}:reversed_cue_boundary")
            continue
        start, end = round(start, 3), round(end, 3)
        duration = round(end - start, 3)
        if duration > _MAX_CLIP_S:
            repaired_range = _repair_oversized_cue_range(
                segments,
                a,
                b,
                ignore_caption_case=ignore_caption_case,
                anchor_text=" ".join(
                    str(value or "")
                    for value in (
                        proposal.title,
                        getattr(proposal, "learning_objective", ""),
                        proposal.facet,
                        proposal.reason,
                    )
                ),
            )
            if repaired_range is not None:
                a, b = repaired_range
                start, end = _padded_cue_bounds(segments, a, b)
                start, end = round(start, 3), round(end, 3)
                duration = round(end - start, 3)
        if duration < _MIN_CLIP_S or duration > _MAX_CLIP_S:
            report.rejected_reasons.append(f"{prefix}:invalid_duration")
            continue

        clip_text = _cue_clip_text(segments, a, b)
        if not clip_text:
            report.rejected_reasons.append(f"{prefix}:empty_cue_transcript")
            continue
        cue_ids = [
            str(segments[line].get("cue_id") or f"cue-{line}")
            for line in range(a, b + 1)
        ]
        clip_id = f"clip-{index + 1:03d}-{a}-{b}"
        clip = {
            "start": start,
            "end": end,
            "title": str(proposal.title or "").strip(),
            "learning_objective": str(
                getattr(proposal, "learning_objective", "")
                or proposal.reason
                or proposal.title
            ).strip(),
            "facet": str(proposal.facet or "").strip(),
            "reason": str(proposal.reason or "").strip(),
            "kind": "educational",
            "informativeness": info,
            "topic_relevance": relevance,
            "self_contained": True,
            "difficulty": difficulty,
            "educational_importance": importance,
            "boundary_confidence": _cue_boundary_confidence(
                str(segments[b].get("text") or ""),
                ignore_caption_case=ignore_caption_case,
            ),
            "is_standalone": is_standalone,
            "chain_id": "",
            "chain_position": 0,
            "prerequisite_ids": prerequisites,
            "cue_ids": cue_ids,
            "start_cue_id": cue_ids[0],
            "end_cue_id": cue_ids[-1],
            "selection_candidate_id": candidate_id,
            "uncertainty": uncertainty,
            "uncertainty_reasons": uncertainty_reasons,
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

    by_candidate_id = {
        str(clip["selection_candidate_id"]): clip for clip in raw
    }
    depended_on = {
        prerequisite
        for clip in raw
        for prerequisite in (clip.get("prerequisite_ids") or [])
    }

    def chain_root_and_depth(candidate_id: str, trail: set[str]) -> tuple[str, int]:
        if candidate_id in trail:
            return candidate_id, 0
        clip = by_candidate_id.get(candidate_id)
        prerequisites = list((clip or {}).get("prerequisite_ids") or [])
        if not prerequisites:
            return candidate_id, 0
        roots_and_depths = [
            chain_root_and_depth(prerequisite, {*trail, candidate_id})
            for prerequisite in prerequisites
        ]
        root = sorted(value[0] for value in roots_and_depths)[0]
        return root, max(value[1] for value in roots_and_depths) + 1

    for clip in raw:
        candidate_id = str(clip["selection_candidate_id"])
        if clip.get("prerequisite_ids") or candidate_id in depended_on:
            root, depth = chain_root_and_depth(candidate_id, set())
            clip["chain_id"] = f"chain:{root}"
            clip["chain_position"] = depth

    # Detect duplicates before removing them so classification cannot turn green by repair.
    report.medium_uncertainty = any(
        clip.get("uncertainty") == "medium" for clip in raw
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

    report.clips = _finalize_clips(raw, settings)
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
    del segments, topic, enrichment_required
    if report.accepted_count:
        return _Classification("green", ())
    reasons = list(report.rejected_reasons) or ["zero_valid_candidates"]
    return _Classification("invalid", tuple(dict.fromkeys(reasons)))


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
    rejection_reasons: list[str] = field(default_factory=list)


class _SchemaResponseError(RuntimeError):
    def __init__(self, message: str, telemetry: object):
        super().__init__(message)
        self.telemetry = telemetry


class _ModelCallError(RuntimeError):
    def __init__(self, message: str, telemetry: dict):
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
    budget_reserve: Optional[Callable[..., object]] = None,
) -> tuple[BaseModel, dict]:
    from ..gemini_client import generate_json_v3

    prompt_text = f"{system}\n\n{user}"
    reservation: dict[str, object] = {}
    if callable(budget_reserve):
        reserved = budget_reserve(
            operation=operation,
            model=model,
            max_output_tokens=max_output_tokens,
            prompt_text=prompt_text,
            estimated_input_tokens=max(1, (len(prompt_text) + 3) // 4),
        )
        if isinstance(reserved, dict):
            reservation = dict(reserved)
    try:
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
            max_retries=0,
            cancelled=cancelled,
        )
    except Exception as exc:
        if _cancel_requested(cancelled):
            raise
        raise _ModelCallError(
            f"{type(exc).__name__}: {exc}",
            {
                "model": model,
                "operation": operation,
                "prompt_version": prompt_version,
                "thinking_level": thinking_level,
                "retries": 0,
                "error_type": type(exc).__name__,
                "dispatched": True,
                **reservation,
            },
        ) from exc
    telemetry = _telemetry_dict(result.telemetry)
    for key, value in reservation.items():
        telemetry.setdefault(key, value)
    telemetry.setdefault("dispatched", True)
    try:
        parsed = schema.model_validate_json(result.text.strip())
    except (ValidationError, ValueError) as exc:
        raise _SchemaResponseError(
            f"invalid {schema.__name__} response: {exc}", telemetry,
        ) from exc
    return parsed, telemetry


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


def _repair_failed_boundaries(
    report: _Conversion,
    segments: list[dict],
    words: list[dict],
    topic: str,
    settings: dict,
    *,
    deadline: float,
    cancelled: CancelledCb,
) -> list[dict]:
    """Run one localized Flash batch and merge only independently valid repairs."""
    candidates = report.repair_candidates[:_PRODUCTION_MAX_CANDIDATES]
    if not candidates or report.accepted_count >= _configured_clip_limit(settings):
        return []

    system, user, allowed = _boundary_repair_prompts(candidates, segments, topic)
    sink = settings.get("_segment_telemetry")
    try:
        plan, call = _call_model(
            system,
            user,
            _BoundaryRepairPlan,
            model=config.SEGMENT_FLASH_MODEL,
            thinking_level="low",
            max_output_tokens=_BOUNDARY_REPAIR_OUTPUT_TOKENS,
            timeout_s=_FLASH_REPAIR_TIMEOUT_S,
            deadline_monotonic=deadline,
            operation="flash_boundary_repair",
            prompt_version=_BOUNDARY_REPAIR_PROMPT_VERSION,
            cancelled=cancelled,
            budget_reserve=settings.get("_segment_budget_reserve"),
        )
        calls = [call]
    except Exception as exc:
        telemetry = _exception_telemetry(exc)
        calls = [telemetry] if telemetry else []
        report.rejected_reasons.append(
            f"boundary_repair_request_failure:{type(exc).__name__}"
        )
        _emit(
            sink,
            "boundary_repair",
            attempted_count=len(candidates),
            accepted_count=0,
            reason=f"request_failure:{type(exc).__name__}",
        )
        return calls

    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    seen: set[str] = set()
    repaired: list[dict] = []
    for item in plan.items:
        candidate = by_id.get(item.candidate_id)
        if candidate is None:
            report.rejected_reasons.append(
                f"boundary_repair:unknown_candidate_id:{item.candidate_id}"
            )
            continue
        if item.candidate_id in seen:
            report.rejected_reasons.append(f"{candidate.prefix}:duplicate_repair")
            continue
        seen.add(item.candidate_id)
        allowed_starts, allowed_ends = allowed[item.candidate_id]
        if item.start_line not in allowed_starts or item.end_line not in allowed_ends:
            report.rejected_reasons.append(f"{candidate.prefix}:repair_outside_neighbors")
            continue
        if item.end_line < item.start_line:
            report.rejected_reasons.append(f"{candidate.prefix}:repair_reversed_range")
            continue

        repaired_proposal = candidate.proposal.model_copy(update={
            "start_line": item.start_line,
            "end_line": item.end_line,
            "start_quote": item.start_quote,
            "end_quote": item.end_quote,
        })
        repaired_report = _plan_to_report(
            _BoundaryPlan(topics=[repaired_proposal]),
            segments,
            words,
            settings,
            topic=topic,
            context_cue_limit=0,
        )
        if repaired_report.accepted_count != 1:
            reasons = repaired_report.rejected_reasons or ["invalid_boundary"]
            for reason in reasons:
                suffix = reason.split(":", 1)[-1]
                report.rejected_reasons.append(
                    f"{candidate.prefix}:repair_{suffix}"
                )
            continue

        clip = repaired_report.clips[0]
        clip["boundary_confidence"] = 0.85
        clip["selection_candidate_id"] = candidate.candidate_id
        clip["_clip_id"] = candidate.candidate_id
        repaired.append(clip)
        original_rejection = f"{candidate.prefix}:{candidate.reason}"
        if original_rejection in report.rejected_reasons:
            report.rejected_reasons.remove(original_rejection)

    for candidate in candidates:
        if candidate.candidate_id not in seen:
            report.rejected_reasons.append(f"{candidate.prefix}:repair_omitted")

    report.clips = _finalize_clips([*report.clips, *repaired], settings)
    _emit(
        sink,
        "boundary_repair",
        attempted_count=len(candidates),
        accepted_count=len(repaired),
        rejected_count=max(0, len(candidates) - len(repaired)),
    )
    return calls


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
        system, user = _boundary_prompts(
            rendered,
            len(segments),
            topic,
            max_candidates=min(
                _PRODUCTION_MAX_CANDIDATES,
                max(1, int(settings.get("max_clips") or _PRODUCTION_MAX_CANDIDATES)),
            ),
        )
        schema = _BoundaryPlan
        model = config.SEGMENT_FLASH_MODEL
        level, cap, timeout = "medium", _BOUNDARY_OUTPUT_TOKENS, _FLASH_BOUNDARY_TIMEOUT_S
        operation = "flash_boundary_selector"
    elif profile == PRO_BOUNDARY_PROFILE:
        system, user = _boundary_prompts(
            rendered,
            len(segments),
            topic,
            max_candidates=min(
                _PRODUCTION_MAX_CANDIDATES,
                max(1, int(settings.get("max_clips") or _PRODUCTION_MAX_CANDIDATES)),
            ),
        )
        schema = _BoundaryPlan
        model = config.SEGMENT_PRO_MODEL
        level, cap, timeout = "high", _BOUNDARY_OUTPUT_TOKENS, _PRO_TIMEOUT_S
        operation = "pro_fallback"
    else:
        raise ValueError(f"unknown segmentation profile: {profile}")

    if profile == FLASH_SPLIT_PROFILE:
        requested_level = str(
            settings.get("_segment_thinking_level") or level
        ).strip().lower()
        if requested_level in {"minimal", "low", "medium", "high"}:
            level = requested_level
    operation = str(settings.get("_segment_operation") or operation)
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
        budget_reserve=settings.get("_segment_budget_reserve"),
    )
    require_enrichment = profile in {CORRECTED_PRO_PROFILE, FLASH_SINGLE_PROFILE}
    conversion_settings = dict(settings)
    conversion_settings.setdefault(
        "_segment_ignore_caption_case",
        str(transcript.get("source") or "").casefold() == "supadata",
    )
    if profile in {PRODUCTION_FLASH_PROFILE, PRO_BOUNDARY_PROFILE}:
        configured_limit = conversion_settings.get("max_clips")
        conversion_settings["max_clips"] = min(
            _PRODUCTION_MAX_CANDIDATES,
            int(config.SEGMENT_MAX_CLIPS if configured_limit is None else configured_limit),
        )
    report = _plan_to_report(
        parsed,
        segments,
        words,
        conversion_settings,
        topic=topic,
        require_enrichment=require_enrichment,
    )
    calls = [call]
    if profile == FLASH_SPLIT_PROFILE and report.repair_candidates:
        calls.extend(_repair_failed_boundaries(
            report,
            segments,
            words,
            topic,
            conversion_settings,
            deadline=deadline,
            cancelled=cancelled,
        ))
    if profile in {FLASH_SPLIT_PROFILE, PRO_BOUNDARY_PROFILE}:
        _drop_unmet_prerequisite_clips(report)
    if profile.startswith("flash_"):
        classification = _classify_flash(
            report, segments, topic, enrichment_required=(profile == FLASH_SINGLE_PROFILE),
        )
    else:
        classification = _Classification(
            "green" if report.clips else "invalid",
            () if report.clips else tuple(dict.fromkeys(report.rejected_reasons)),
        )
    return report, classification, calls


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


def enrich_accepted_clips(
    items: list[dict],
    *,
    topic: str,
    settings: dict | None = None,
    deadline_monotonic: float | None = None,
    cancelled: CancelledCb = None,
) -> tuple[dict[str, dict], list[dict]]:
    """Enrich at most three persisted clips without affecting clip validity."""
    batch = [dict(item) for item in items[:3] if str(item.get("clip_id") or "").strip()]
    if not batch:
        return {}, []
    source_by_id = {
        str(item["clip_id"]): str(item.get("text") or "").strip()
        for item in batch
    }
    system, user = _card_enrichment_prompts(batch, topic)
    try:
        plan, call = _call_model(
            system,
            user,
            _CardEnrichmentPlan,
            model=config.SEGMENT_FLASH_MODEL,
            thinking_level="low",
            max_output_tokens=_ENRICH_OUTPUT_TOKENS,
            timeout_s=_FLASH_ENRICH_TIMEOUT_S,
            deadline_monotonic=(
                deadline_monotonic or (time.monotonic() + _FLASH_ENRICH_TIMEOUT_S)
            ),
            operation="flash_grounded_enrichment",
            prompt_version=_CARD_ENRICHMENT_PROMPT_VERSION,
            cancelled=cancelled,
            budget_reserve=(settings or {}).get("_segment_budget_reserve"),
        )
        calls = [call]
    except Exception as exc:
        telemetry = _exception_telemetry(exc)
        return {}, [telemetry] if telemetry else []

    enriched: dict[str, dict] = {}
    seen: set[str] = set()
    topic_tokens = _content_tokens(topic)
    for item in plan.items:
        clip_id = str(item.clip_id)
        grounding_text = source_by_id.get(clip_id, "")
        if not grounding_text or clip_id in seen:
            continue
        seen.add(clip_id)
        summary = " ".join(item.summary.split())
        takeaways = [" ".join(value.split()) for value in item.takeaways]
        match_reason = " ".join(item.match_reason.split())
        if not _text_has_grounding(summary, grounding_text):
            continue
        if any(not _text_has_grounding(value, grounding_text) for value in takeaways):
            continue
        if not _text_has_grounding(match_reason, grounding_text):
            continue
        if topic_tokens and not topic_tokens.intersection(_content_tokens(match_reason)):
            continue
        enriched[clip_id] = {
            "summary": summary[:700],
            "takeaways": takeaways[:4],
            "match_reason": match_reason[:700],
        }
    return enriched, calls


def _enrich_split(
    clips: list[dict],
    topic: str,
    settings: dict | None = None,
    *,
    deadline: float,
    cancelled: CancelledCb,
) -> tuple[list[dict], list[dict], list[str], str | None]:
    calls: list[dict] = []
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
            budget_reserve=(settings or {}).get("_segment_budget_reserve"),
        )
        calls.append(call)
        errors = _apply_enrichment(clips, plan, topic)
    except Exception as exc:  # schema/transport failure makes every enrichment item invalid
        telemetry = _exception_telemetry(exc)
        if telemetry:
            calls.append(telemetry)
        errors = [f"{clip['_clip_id']}:flash_enrichment_failure" for clip in clips]

    invalid_ids = _invalid_enrichment_clip_ids(errors, clips)
    for clip in clips:
        if clip["_clip_id"] in invalid_ids:
            clip.update({
                "summary": "",
                "takeaways": [],
                "match_reason": "",
                "assessment": None,
            })
    # Learning details are optional. A valid boundary selection never incurs a
    # slower Pro retry merely because enrichment was absent or malformed.
    return clips, calls, [], None


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
                and report.clips and bool(settings.get("segment_enrich_clips"))):
            (report.clips, enrichment_calls, fallback_reasons,
             flash_configuration_error) = _enrich_split(
                 report.clips,
                 topic,
                 settings,
                 deadline=deadline,
                 cancelled=cancelled,
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
            rejection_reasons=list(report.rejected_reasons),
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
            rejection_reasons=[f"request_failure:{type(exc).__name__}"],
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
    profile = PRO_BOUNDARY_PROFILE if fallback else AUTHORITATIVE_PRO_PROFILE
    operation = "pro_fallback" if fallback else "pro_authoritative"
    runtime_settings = dict(settings)
    runtime_settings["_segment_operation"] = operation
    result = run_segment_profile(
        transcript, runtime_settings, profile, topic=topic,
        deadline_monotonic=deadline, cancelled=cancelled,
    )
    for call in result.calls:
        call["operation"] = operation
    return result


def pro_boundary_fallback_detailed(
    transcript: dict,
    settings: dict,
    *,
    topic: str = "",
    video_id: str = "",
) -> SegmentResult:
    """Run the one aggregate, boundary-only Pro fallback on an existing transcript."""
    sink = settings.get("_segment_telemetry")
    cancelled = settings.get("_segment_cancelled")
    deadline = time.monotonic() + _TOTAL_DEADLINE_S
    configured_deadline = settings.get("deadline_monotonic")
    if configured_deadline is not None:
        try:
            deadline = min(deadline, float(configured_deadline))
        except (TypeError, ValueError, OverflowError):
            pass
    reasons = ["aggregate_initial_yield_below_three"]
    _emit(sink, "pro_fallback", video_id=video_id or None, reasons=reasons)
    result = _authoritative_pro(
        transcript,
        settings,
        topic,
        deadline,
        cancelled,
        fallback=True,
    )
    result.route = "aggregate_pro_fallback"
    result.fallback_reasons = reasons
    for call in result.calls:
        _emit(sink, "model_call", video_id=video_id or None, **call)
    accepted = len(result.clips)
    total_cost = sum(_model_cost(call) for call in result.calls)
    _emit(
        sink,
        "segment_completed" if not result.error else "segment_error",
        video_id=video_id or None,
        route=result.route,
        classification=result.classification,
        classification_reasons=result.classification_reasons,
        rejection_reasons=result.rejection_reasons,
        fallback_reasons=reasons,
        proposed_count=result.proposed_count,
        accepted_count=accepted,
        zero_output=(accepted == 0),
        fallback_rate=1.0,
        pricing_version=_PRICING_VERSION,
        estimated_cost_usd=round(total_cost, 8),
        cost_per_accepted_clip_usd=(
            round(total_cost / accepted, 8) if accepted else None
        ),
        error=result.error,
    )
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
    flash_only = configured_mode == "flash_only"
    mode = configured_mode
    if mode not in {"pro_only", "shadow", "hybrid", "flash_only"}:
        mode = "pro_only"
    disabled_reason = None if flash_only else _flash_disable_reason()
    if disabled_reason is not None and mode in {"shadow", "hybrid"}:
        mode = "pro_only"
    percent = float(config.SEGMENT_HYBRID_PERCENT)
    generation_context = (
        settings.get("generation_context") or settings.get("provider_context")
        if isinstance(settings, dict)
        else None
    )
    generation_hash_key = str(
        getattr(generation_context, "generation_id", "") or video_id
    )
    selected = flash_only or (
        mode == "hybrid" and _hybrid_selected(generation_hash_key, percent)
    )
    route = "flash_first" if selected else "pro_authoritative"
    sink = settings.get("_segment_telemetry") if isinstance(settings, dict) else None
    cancelled = settings.get("_segment_cancelled") if isinstance(settings, dict) else None
    deadline = time.monotonic() + _TOTAL_DEADLINE_S
    configured_deadline = settings.get("deadline_monotonic")
    if configured_deadline is not None:
        try:
            deadline = min(deadline, float(configured_deadline))
        except (TypeError, ValueError, OverflowError):
            pass
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
                rejection_reasons=shadow.rejection_reasons,
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
              rejection_reasons=flash.rejection_reasons,
              proposed_count=flash.proposed_count, accepted_count=flash.accepted_count)
        flash_error = flash.flash_configuration_error or flash.error
        if _flash_configuration_failure(flash_error):
            _disable_flash(str(flash_error))
            _emit(sink, "route_rollback", video_id=video_id or None,
                  reason="flash_model_access_or_configuration_failure")
        if flash.classification == "green" and not flash.error:
            result = flash
            result.route = "hybrid_flash"
        else:
            # Bootstrap's hard latency contract never dispatches Pro, even if a
            # caller forgets to install the normal generation-level fallback gate.
            fallback_allowed = not flash_only
            fallback_gate = (
                settings.get("_segment_pro_fallback_gate")
                if isinstance(settings, dict)
                else None
            )
            if fallback_allowed and callable(fallback_gate):
                try:
                    fallback_allowed = bool(
                        fallback_gate(
                            accepted_count=flash.accepted_count,
                            video_id=video_id,
                        )
                    )
                except Exception as exc:  # fail closed: a gate bug must not spend Pro
                    fallback_allowed = False
                    _emit(
                        sink,
                        "pro_fallback_deferred",
                        video_id=video_id or None,
                        reason=f"fallback_gate_error:{type(exc).__name__}",
                    )
            fallback_reasons = list(flash.classification_reasons) or ["flash_request_failure"]
            if fallback_allowed:
                _emit(
                    sink,
                    "pro_fallback",
                    video_id=video_id or None,
                    reasons=fallback_reasons,
                )
                pro = _authoritative_pro(
                    transcript, settings, topic, deadline, cancelled, fallback=True,
                )
                # Never expose uncertain/invalid Flash when Pro fails.
                result = pro
                result.route = "hybrid_pro_fallback"
                result.fallback_reasons = fallback_reasons
                result.calls = flash.calls + pro.calls
                result.rejection_reasons = [
                    *flash.rejection_reasons,
                    *pro.rejection_reasons,
                ]
            else:
                _emit(
                    sink,
                    "pro_fallback_deferred",
                    video_id=video_id or None,
                    reasons=fallback_reasons,
                )
                # Non-green output never ships while the aggregate gate waits for
                # the second video or chooses a later backfill for the one fallback.
                flash.clips = []
                flash.accepted_count = 0
                flash.fallback_reasons = []
                result = flash
                result.route = "hybrid_flash_deferred"
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
        rejection_reasons=result.rejection_reasons,
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
        routing_mode=(
            str(settings.get("_segment_routing_mode") or "").strip() or None
            if isinstance(settings, dict)
            else None
        ),
    )
    return result.clips, result.notes
