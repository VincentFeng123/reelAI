"""Guarded Gemini educational clip segmentation.

Production uses one low-thinking Flash call over the whole timestamped transcript,
then applies deterministic quality, context, grounding, filler, and deduplication
guards. Legacy routing and enrichment helpers remain available only for isolated
evaluation compatibility; the public production adapter never dispatches them.

The public contract stays ``segment_clips(...) -> (clips, notes)``.  Model names,
routing decisions, and call telemetry are logged internally and never added to a
clip, note, or API response.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
import unicodedata
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

_WORD_RE = re.compile(r"[^\W_]+(?:['\u2018\u2019\u02bc][^\W_]+)*", re.UNICODE)
_APOSTROPHES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u02bc": "'"})
_CandidateId = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1)
]
_BoundaryQuote = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1)
]
_ClipTitle = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1)
]
_LearningObjective = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1)
]
_Facet = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1)
]
_EvidenceQuote = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1)
]
_OptionalReason = Annotated[
    str, StringConstraints(strip_whitespace=True)
]
_NON_SPEECH_MARKER_PATTERN = (
    r"(?:\[\s*(?:(?:(?:theme|intro|outro|background)\s+)?music|applause|"
    r"laughter|cheering|inaudible)\s*\]|"
    r"\(\s*(?:(?:(?:theme|intro|outro|background)\s+)?music|applause|"
    r"laughter|cheering|inaudible)\s*\)|"
    r"[\u2669-\u266c]+)"
)
_NON_SPEECH_MARKER_RE = re.compile(_NON_SPEECH_MARKER_PATTERN, re.IGNORECASE)
_STRUCTURAL_FILLER_RE = re.compile(
    rf"(?:{_NON_SPEECH_MARKER_PATTERN}|\b(?:thanks? for watching|have a great day|see you next time|"
    r"like and subscribe|please subscribe|"
    r"subscribe to (?:this|the|my|our) channel|today'?s sponsor|"
    r"check out (?:our|the) video|"
    r"administrative (?:note|announcement)|course (?:administration|logistics)|"
    r"(?:a\s+)?(?:quick|brief|short) (?:aside|tangent)|"
    r"look at (?!this\b)(?!(?:the\s+)?(?:animation|chart|diagram|drawing|equation|"
    r"figure|graph|image|map|object|screen|shape|simulation|slide|table)\b)"
    r"(?:the\s+)?"
    r"(?![^.!?]{0,80}\b(?:after|as|because|before|by|how|if|that|the way|when|"
    r"where|whether|which|while|why)\b)"
    r"(?=(?:[^\W_]+(?:['’\-][^\W_]+)?\s*){1,5}[.!?])"
    r"[^.!?]{1,80}(?=[.!?](?:\s|$))|"
    r"(?<!\bare )(?<!\bis )(?<!\bwere )(?<!\byou are )(?<!\byou['’]re )"
    r"\bwelcome(?:\s+back)?(?:\s+to\s+(?:(?:this|the|my|our)\s+)?"
    r"(?:channel|video|lesson|course|show|episode|series)\b|\s+to\b|"
    r"(?=[!,.]|\s*$))|"
    r"in this (?:video|lesson|course) we(?:['’]ll| will)|"
    r"before we (?:begin|get started)|let(?:['’]?s| us) move on|"
    r"next we(?:['’]ll| will)|cover (?:that|this) in (?:this|the) course)\b|"
    r"(?:but\s+)?we(?:['’]ll| will)\s+(?:(?:talk|discuss|learn|say|cover|"
    r"explore|explain)\s+"
    r"(?:more\s+)?about\s+(?:that|this|it)|(?:talk|discuss)\s+about\s+"
    r"(?:that|this|it)\s+more|(?:discuss|cover|revisit|explore|explain)\s+"
    r"(?:that|this|it)(?:\s+more)?|(?:return|come\s+back)\s+to\s+"
    r"(?:that|this|it))\s+(?:next\s+time|later|in\s+(?:a|the)\s+"
    r"(?:next|future)\s+(?:video|lesson|section|episode))\b|"
    r"(?:^|[.!?]\s+)(?:(?:all right|alright|okay|ok|so|now|well|yeah)"
    r"\s*[,;:]?\s+)*(?:"
    r"let(?:['’]?s| us)\s+(?:begin|get started|start|dive in|delve)"
    r"\s*[.!?](?=\s|$)|"
    r"(?:cool|hey|fun fact|brilliant)\s*[!,.](?=\s|$)|"
    r"(?:oh\s*[,;:]?\s*)?(?:yeah\s*[,;:]?\s*)?by the way\b|"
    r"sponsored by\b|"
    r"we (?:made|have) (?:a|an|another|whole) video "
    r"(?:about|explaining|on)\b|my name is\b|"
    r"to (?:recap|summarize)(?:\s*[,;:]|\s*$)|"
    r"in summary(?:\s*[,;:]|\s+(?=(?:we|the|this|these|there|our|a|an)\b))|"
    r"we (?:are|'re) reaching (?:a|the) crossroad now\b|"
    r"we(?:['’]ve| have) already (?:done|covered|finished)\b.{0,80}))",
    re.IGNORECASE,
)
_INTERNAL_INTERRUPTION_MARKER_RE = re.compile(
    r"\b(?:today'?s sponsor|sponsored by|administrative (?:note|announcement)|"
    r"course (?:administration|logistics)|(?:a\s+)?(?:quick|brief|short) "
    r"(?:aside|tangent)|housekeeping)\b",
    re.IGNORECASE,
)
_VISUAL_DEPENDENCY_RE = re.compile(
    r"\b(?:as you can see|as shown (?:here|on (?:the )?screen)|"
    r"on (?:the )?screen|this (?:diagram|figure|chart|graph|image|slide|drawing)|"
    r"the (?:diagram|figure|chart|graph|image|slide) (?:shows|illustrates)|"
    r"look (?:here|at this|at (?:(?:this|the) )?(?:animation|chart|diagram|drawing|equation|"
    r"figure|graph|image|map|object|screen|shape|simulation|slide|table))|"
    r"over here|watch (?:this|what happens)|"
    r"I(?:['’]m| am)? (?:drawing|writing)|I(?:['’]ll| will) (?:draw|write))\b",
    re.IGNORECASE,
)
_DANGLING_TAIL_PREFIX_RE = re.compile(
    r"^\s*tail[.!?]\s+"
    r"(?:(?:[Uu]m+|[Uu]h+|[Ww]ell)[, ]+)?"
    r"(?:[Ii]t|[Tt]his|[Tt]hat|[Tt]hey|[Tt]hese|[Tt]hose|[Hh]e|[Ss]he)\b"
)
_OPENING_COMPARATIVE_FRAGMENT_RE = re.compile(
    r"^\s*(?:much\s+)?(?:more|less)\s+[a-z][a-z'-]*\?(?:\s+|$)",
    re.IGNORECASE,
)
_EXISTENTIAL_OPENING_RE = re.compile(
    r"^\s*there\s+(?:is|are|was|were)\s+"
    r"(?:(?:(?:a|an|no|some|many|several|multiple|numerous|few)|"
    r"(?:one|two|three|four|five|six|seven|eight|nine|ten)|\d+)\s+)?"
    r"(?P<tail>[^\W_][^\n.!?]*?)\s*$",
    re.IGNORECASE,
)
_EXISTENTIAL_UNRESOLVED_RE = re.compile(
    r"\b(?:this|these|those|they|them|their|theirs|he|him|his|she|her|"
    r"hers|it|its|mine|ours|yours|here|there|"
    r"such|same|former|latter|above|below|previous|following|earlier|more|"
    r"other|others|ones)\b",
    re.IGNORECASE,
)
_EXISTENTIAL_CONTEXTUAL_THAT_RE = re.compile(
    r"(?:^|\b(?:of|for|in|on|about|from|by|to|with|under|over)\s+)that\b",
    re.IGNORECASE,
)
_EXISTENTIAL_TERMINAL_REFERENCE_RE = re.compile(
    r"(?:\b(?:that|so)|\bthat\s+[a-z0-9][a-z0-9'-]*)\s*$",
    re.IGNORECASE,
)
_EXISTENTIAL_DEMONSTRATIVE_THAT_RE = re.compile(
    r"(?:\b(?:why|how|when|where|which|who|whose)\s+that\s+[a-z0-9]|"
    r"\bthat\s+(?:[a-z][a-z'-]*(?:tion|sion|ment|ness|ity|ence|ance|"
    r"ship|hood|ism|ure|age|acy|ics)|answer|cell|enzyme|example|gene|idea|"
    r"method|one|pathway|problem|protein|reaction|result|step|subject|theory|"
    r"thing|topic)\b)",
    re.IGNORECASE,
)
_EXISTENTIAL_BACK_REFERENCE_RE = re.compile(
    r"\b(?:mentioned|shown|discussed|described|introduced|seen|noted|defined|"
    r"explained)\s+(?:earlier|before|above|previously)\b",
    re.IGNORECASE,
)
_EXISTENTIAL_SCOPE_RE = re.compile(
    r"(?:\b(?:of|in|within|among|between|during|under|on|for|about|behind|"
    r"from|by|to|across|inside|outside)\s+(?:the\s+|a\s+|an\s+)?[a-z]|"
    r"\b(?:that|which|who|whose|where|when|why|how)\s+[a-z]|"
    r"\b(?:i|we|you)\s+(?:(?:can|could|will|would|should|do|did)\s+)?"
    r"[a-z][a-z'-]*\s+(?:the\s+|a\s+|an\s+)?[a-z0-9][a-z0-9'-]*|"
    r"\b[a-z][a-z'-]*\s+[a-z][a-z'-]*(?:ing|ed)\s+"
    r"(?:the\s+|a\s+|an\s+)?[a-z0-9]|"
    r",\s*(?:(?:namely|specifically|called)\s+)?[a-z0-9][a-z0-9'-]*"
    r"\s+(?:and|or)\s+[a-z0-9][a-z0-9'-]*|[;:]\s*[a-z0-9])",
    re.IGNORECASE,
)
_TERMINAL_CALLBACK_RE = re.compile(
    r"(?:^|[.!?]\s+)(?:look|think|go|turn|refer) back (?:at|to)\b[^.!?]*[.!?]?\s*$",
    re.IGNORECASE,
)
_TERMINAL_DANGLING_TRANSITION_RE = re.compile(
    r"(?:^|[.!?]\s+)(?:all\s+right\s*[,;:]?\s*)?"
    r"let(?:['’]?s|\s+us)\s*[.!?]?\s*$",
    re.IGNORECASE,
)
_TERMINAL_INCOMPLETE_SUBJECT_RE = re.compile(
    r"\b(?:i|we|you|he|she|it|they|this|that)['’](?:d|ll|m|re|s|ve)"
    r"[.!?]?\s*$",
    re.IGNORECASE,
)
_TERMINAL_BARE_SUBJECT_RE = re.compile(
    r"\b(?:and|as|because|but|if|or|since|so|that|though|unless|until|when|while)"
    r"\s+(?:i|we|you|he|she|it|they)\s*$",
    re.IGNORECASE,
)
_TERMINAL_DANGLING_ARTICLE_RE = re.compile(
    r"\b(?:a|an|the)\s*$",
    re.IGNORECASE,
)
_TERMINAL_DANGLING_DEGREE_RE = re.compile(
    r"\b(?:am|are|be|been|being|feels?|is|looks?|seems?|sounds?|was|were)"
    r"\s+(?:less|more|quite|rather|really|so|too|very)\s*$",
    re.IGNORECASE,
)
_TERMINAL_AMBIGUOUS_DEGREE_RE = re.compile(
    r"\b(?:am|are|be|been|being|feels?|is|looks?|seems?|sounds?|was|were)"
    r"\s+pretty\s*$",
    re.IGNORECASE,
)
_TRAILING_FORWARD_SETUP_RE = re.compile(
    r"(?:^|[.!?]\s+)(?:but\s+)?what happens if\b.*?\?\s*"
    r"(?:now\s*[,]?\s*)?we\s+can(?:not|['’]t)\b[^.!?]*[.!?]?\s*$",
    re.IGNORECASE | re.DOTALL,
)
_FORWARD_SOLUTION_CONTINUATION_RE = re.compile(
    r"^\s*(?:so\s+)?instead\b",
    re.IGNORECASE,
)
_TERMINAL_EXEMPLIFICATION_RE = re.compile(
    r"(?:^|[.!?;,]\s+)(?P<so>so\s*[,]?\s+)?"
    r"(?:for\s+(?:example|instance)|such\s+as)\s*[,]?\s*"
    r"(?P<tail>[^.!?]*)$",
    re.IGNORECASE,
)
_TERMINAL_STRANDED_PREPOSITION_RE = re.compile(
    r"\b(?:a|an|the|this|that|these|those|our|your|their)\s+"
    r"(?:[a-z][a-z'-]*\s+){1,4}(?:(?:that|which|who|whom)\s+)?"
    r"(?:i|we|you|they)\s+(?:can|could|will|would|should|may|might)\s+"
    r"(?:[a-z][a-z'-]*\s+){1,4}(?:from|with|to|for|about|on|at|by)"
    r"[.!?][\"')\]]*$",
    re.IGNORECASE,
)
_VAMPIRE_TOPIC_RE = re.compile(
    r"\b(?:vampir\w*|dracula|nosferatu)\b",
    re.IGNORECASE,
)
_VAMPIRE_PSEUDOSCIENCE_SIGNAL_RE = re.compile(
    r"\b(?:supernatural|lore|condition|virus|cross[- ]?wired|receptors?|"
    r"visual cortex|crucifix|dark entit\w*)\b",
    re.IGNORECASE,
)
_STANDALONE_QUESTION_HEADS = frozenset({
    "what", "how", "why", "where", "when", "which", "who", "whose", "whom",
    "is", "are", "can", "could", "would", "should", "does", "do", "did",
    "will", "was", "were", "has", "have", "had",
})
_QUESTION_PREFIXES = frozenset({
    "and", "but", "so", "now", "well", "okay", "ok", "alright",
})
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
_FLASH_BOUNDARY_TIMEOUT_S = 90.0
_FLASH_REPAIR_TIMEOUT_S = 20.0
_FLASH_ENRICH_TIMEOUT_S = 25.0
_PRO_TIMEOUT_S = 90.0
_SELECTION_OUTPUT_TOKENS = 24_576
_BOUNDARY_OUTPUT_TOKENS = 12_288
_BOUNDARY_REPAIR_OUTPUT_TOKENS = 1_024
_ENRICH_OUTPUT_TOKENS = 2_048
_MAX_CLIPS = 40
_GREEN_SCORE = 0.75
_DUPLICATE_OVERLAP = 0.8
_MAX_INTERNAL_FILLER_DURATION_S = 12.0
_MAX_INTERNAL_FILLER_WORDS = 32
_SECTION_RESET_GAP_S = 8.0
_BOUNDARY_PAD_S = 0.3
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
    evidence_quote: _NonBlank = Field(
        description=(
            "Exact consecutive transcript words copied from inside this clip; preserve "
            "the transcript spelling and never paraphrase or stitch text."
        )
    )

    @model_validator(mode="after")
    def _unique_options(self):
        normalized = {" ".join(option.split()).casefold() for option in self.options}
        if len(normalized) != 4:
            raise ValueError("assessment options must be distinct")
        if any("all of the above" in option.casefold() for option in self.options):
            raise ValueError("all-of-the-above options are not allowed")
        return self


class _BoundaryTopic(_StrictModel):
    candidate_id: _CandidateId
    start_line: int = Field(ge=0, strict=True)
    end_line: int = Field(ge=0, strict=True)
    start_quote: _BoundaryQuote = Field(
        description=(
            "Four to eight consecutive words copied exactly from the cited start line, "
            "beginning at the first required teaching words."
        )
    )
    end_quote: _BoundaryQuote = Field(
        description=(
            "Four to eight consecutive words copied exactly from the cited end line, "
            "ending at the complete teaching conclusion."
        )
    )
    title: _ClipTitle
    learning_objective: _LearningObjective
    facet: _Facet
    reason: _OptionalReason = ""
    informativeness: float = Field(ge=0.0, le=1.0, strict=True)
    topic_relevance: float = Field(ge=0.0, le=1.0, strict=True)
    educational_importance: float = Field(ge=0.0, le=1.0, strict=True)
    difficulty: float = Field(ge=0.0, le=1.0, strict=True)
    directly_teaches_topic: bool = Field(strict=True)
    substantive: bool = Field(strict=True)
    factually_grounded: bool = Field(strict=True)
    topic_evidence_quote: _EvidenceQuote = Field(
        description=(
            "Shortest five to twelve consecutive transcript words wholly between the "
            "chosen start and end quotes that prove relevance; never paraphrase or stitch."
        )
    )
    self_contained: bool = Field(strict=True)
    is_standalone: bool = Field(strict=True)
    prerequisite_candidate_ids: list[_CandidateId] = Field(default_factory=list, max_length=8)
    uncertainty: Literal["low", "medium", "high"] = "low"
    uncertainty_reasons: list[_UncertaintyReason] = Field(default_factory=list, max_length=6)

    @model_validator(mode="after")
    def _uncertainty_has_reason(self):
        if self.uncertainty != "low" and not self.uncertainty_reasons:
            raise ValueError("medium/high uncertainty requires a reason")
        return self


class _Topic(_BoundaryTopic):
    reason: _NonBlank
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
    items: list[_BoundaryRepairItem] = Field(max_length=_MAX_CLIPS)


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
- First understand the whole transcript and the viewer's exact request. Return only
  related, complete, substantive teaching units that make sense to a cold viewer hearing
  the clip without seeing the original video. Related includes the requested subject and
  clearly useful prerequisite or supporting facets, not merely adjacent material.
- Include every necessary setup or prerequisite through the explanation's natural
  conclusion. For a worked example, include the question or setup, reasoning, and answer.
- Give each candidate exactly one coherent learning objective. When adjacent speech teaches
  independent facets, return separate candidates for those facets instead of bundling them.
- Omit greetings, credentials, sponsors, administration, promos, transitions, previews,
  recaps, outros, atmospheric hooks, scene-setting, music/applause caption markers,
  colorful flourishes, audience banter, post-conclusion jokes, tangents, repeated
  restatements, and partial explanations.
- Keep starts and ends free of that filler. Never add filler or incomplete material at an
  opening or ending. A brief nonessential aside inside an otherwise
  valuable complete unit may remain when cutting around it would break the teaching arc;
  never discard the whole unit solely for a short internal interruption.
- Omit teaching that depends on a diagram, screen, gesture, drawing, or other missing
  visual context. Mark self_contained and is_standalone false for such material.
- Exhaustively enumerate every distinct related teaching unit, up to 40 per source. Prefer an empty
  slot to filler or an incomplete idea. Do not shorten a complete idea to fit a target
  length; clip duration is never a selection criterion.
- Keep distinct informational facets from the same source. Do not return two clips that
  teach the same learning objective in different words.
- Return every qualifying related unit, while scoring the densest, most useful, and most
  central units highest so the application can prioritize them within difficulty stages.
- Return a candidate only when informativeness, topic_relevance, and educational_importance
  are each at least 0.75 and the spoken unit satisfies every substantive, grounding,
  context, and filler rule.
- Copy exact transcript line IDs and exact opening/closing quotes. start_quote must be the
  first words a cold viewer needs to hear for this one teaching objective, after every
  atmospheric hook, scene-setting flourish, or opening joke. end_quote must be the last
  words of its complete conclusion, before audience banter, a next-topic setup, or a
  post-conclusion joke. Quotes may begin or end inside a coarse transcript line. Copy 4-8
  consecutive words exactly, preserve the transcript spelling, keep the quote wholly inside
  its cited edge line, and make it specific enough to occur only once there. Never
  paraphrase, correct, stitch, or cross transcript lines. topic_evidence_quote must be the
  shortest useful 5-12 consecutive words wholly between those chosen edges, copied with the
  same exactness.
- A worked example cannot end at its question, setup, or first substituted value. Either
  include its reasoning and answer through end_quote or end the candidate before that
  optional example begins.
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
    if re.search(
        r"\b(?:versus|vs\.?|compare(?:d|s|ing)?|comparison|contrast|difference between)\b|/",
        topic,
        flags=re.IGNORECASE,
    ):
        compound_rule = (
            "For a comparison request, deeply teaching either requested side is relevant, "
            "as is teaching their relationship. Return separate substantive units for either "
            "side; do not require every candidate to repeat or compare both sides. "
        )
    elif "," in topic or ";" in topic:
        compound_rule = (
            "When the topic lists multiple requested ideas, a span directly matches when "
            "it deeply teaches any one requested component. Require a relationship between "
            "components only when the viewer explicitly asks to compare, connect, relate, "
            "or apply them together. "
        )
    else:
        compound_rule = (
            "When the topic names multiple linked ideas, deeply teaching any one requested "
            "component or a useful prerequisite facet is relevant. The selected speech or "
            "its exact evidence quote must still anchor that facet to the named subject; a "
            "broad word used in an unrelated domain is not enough. "
        )
    return (
        f"The viewer is studying {topic!r}. Return only units that teach that topic or a "
        "clearly useful prerequisite/supporting facet, and make each learning objective name "
        "the relevant idea. Set directly_teaches_topic=true for either a direct unit or an "
        "explicitly topic-linked prerequisite/supporting unit. Set it false when the span "
        "merely names the subject, course, institution, or speaker, or belongs to an adjacent "
        "field without a useful connection to the request. "
        f"{compound_rule}"
        "When the topic requests "
        "identification, recognition, diagnosis, derivation, comparison, or application, "
        "include units that teach or perform that task for the named object as well as "
        "separate, explicitly topic-anchored prerequisite facets. A history or definition "
        "alone is not a direct match to the requested task; return it only as a separate "
        "facet when exact evidence anchors it to the named subject. Task fulfillment raises "
        "educational importance and centrality; it does not exclude a genuinely related, "
        "topic-anchored prerequisite facet. Shared vocabulary, a loose analogy, or general "
        "systems thinking alone is not a useful prerequisite. Include supporting material "
        "only when it is genuinely needed to understand or apply the exact requested topic."
        " Exclude fictional, supernatural, pseudoscientific, or invented mechanisms unless "
        "the viewer explicitly requested that fictional subject. Borrowing real academic "
        "terminology does not make an invented claim educational evidence."
    )


def _contains_unrequested_vampire_pseudoscience(text: str, topic: str) -> bool:
    requested = " ".join(str(topic or "").split())
    if _VAMPIRE_TOPIC_RE.search(requested):
        return False
    candidate = re.sub(r"\bvampire bats?\b", "", str(text or ""), flags=re.IGNORECASE)
    return bool(
        re.search(r"\bvampir\w*\b", candidate, flags=re.IGNORECASE)
        and _VAMPIRE_PSEUDOSCIENCE_SIGNAL_RE.search(candidate)
    )


def _learner_rule(level: str) -> str:
    normalized = " ".join(str(level or "").split()).casefold()
    if normalized not in {"beginner", "intermediate", "advanced"}:
        return ""
    return (
        f"The viewer's current level is {normalized}. Still return qualifying units at every "
        "difficulty. Difficulty is metadata, not an eligibility filter; the application "
        "organizes accepted units later."
    )


def _selection_fields(*, enriched: bool) -> str:
    fields = (
        "candidate_id (a short unique slug), start_line, end_line, start_quote and "
        "end_quote (each 4-8 exact consecutive words wholly inside its cited line, preserving "
        "transcript spelling and marking the first required teaching words and last complete "
        "conclusion, even inside a coarse line; never paraphrase, stitch, or cross lines), "
        "title (at most 12 words), "
        "learning_objective (at most 24 words), facet (at most 12 words), "
        "informativeness, topic_relevance, "
        "educational_importance, difficulty, directly_teaches_topic, substantive, "
        "factually_grounded, "
        "topic_evidence_quote (the shortest exact 5-12 consecutive-word quote copied wholly "
        "between the chosen edges that proves the clip teaches the topic; preserve spelling "
        "and never paraphrase or stitch), self_contained, is_standalone, "
        "prerequisite_candidate_ids (omit it or return []), uncertainty (omit for low), "
        "uncertainty_reasons (omit for low)"
    )
    if enriched:
        fields += (
            ", summary, takeaways (2-4 distinct grounded points), match_reason, and "
            "assessment {prompt (at most 16 words), exactly four distinct options "
            "(at most 8 words each), correct_index, explanation (one sentence, at most "
            "24 words), evidence_quote copied exactly from the selected clip}"
        )
    return fields


def _prompts(
    lines: str,
    n: int,
    topic: str = "",
    learner_level: str = "",
) -> tuple[str, str]:
    """Gemini 3.5 single-pass prompt: policy/examples, context, task last."""
    system = (
        "You select self-contained educational clips from timestamped transcripts.\n\n"
        + _POLICY_AND_EXAMPLES
    )
    learner_rule = _learner_rule(learner_level)
    learner_line = f"{learner_rule}\n" if learner_rule else ""
    user = (
        f"Line IDs must be between 0 and {n - 1}.\n\n"
        f"Transcript ({n} lines, formatted `[index] MM:SS text`):\n{lines}\n\n"
        f"Exact user request: {topic.strip() or '(all educational topics)'}\n"
        f"{_topic_rule(topic)}\n{learner_line}"
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
    learner_level: str = "",
) -> tuple[str, str]:
    system = (
        "You select self-contained educational clip boundaries from timestamped transcripts.\n\n"
        + _POLICY_AND_EXAMPLES
    )
    del learner_level
    user = (
        f"Transcript ({n} lines, formatted `[index] MM:SS text`; valid line IDs are "
        f"0 through {n - 1}):\n{lines}\n\n"
        f"Exact user request: {topic.strip() or '(all educational topics)'}\n"
        f"{_topic_rule(topic)}\n\n"
        "Task:\n"
        "1. Understand the whole transcript before selecting anything; scan the whole "
        "transcript from first to last.\n"
        "2. Map every distinct educational unit related to the exact request, including "
        "niche facts, useful prerequisite facets, examples, mechanisms, comparisons, and "
        "conclusions. Return every distinct qualifying moment, up to 40 for this source; "
        "do not stop after the first few units or at an arbitrary count below that cap.\n"
        "3. For every qualifying unit, verify its timestamps and choose the minimum complete "
        "span containing necessary setup, reasoning, and the natural conclusion, regardless "
        "of its duration. Give it exactly one learning objective. Split independent adjacent "
        "facets into separate candidates even when they share one coarse transcript line. "
        "Keep opening and ending edges clean, including generic lead-ins and bracketed "
        "non-speech markers. Split around a substantial "
        "interruption, but keep a brief internal aside when removing it would break an "
        "otherwise valuable complete arc. Omit only high-uncertainty boundaries; low or "
        "medium uncertainty is allowed. Omit material that requires an unseen visual.\n"
        "4. Score topic relevance, information density, educational value, and difficulty "
        "honestly. Return a unit only when topic_relevance, informativeness, and "
        "educational_importance are each at least 0.75. Difficulty is metadata, not an "
        "eligibility filter; "
        "it records prior knowledge only: 0.00-0.33 means "
        "beginner, 0.34-0.66 means intermediate, and 0.67-1.00 means advanced. Return units "
        "across that entire scale.\n"
        "5. Return every distinct qualifying unit. Set substantive and factually_grounded true "
        "only for academically sound teaching; course logistics and institutional framing are "
        "not teaching units. Each unit must be standalone, use a unique "
        "candidate_id, and list no prerequisite candidate IDs because required setup belongs "
        "inside its span.\n"
        f"Every item must contain {_selection_fields(enriched=False)}. Learning details and "
        "assessments are generated later. Do not include them, chain-of-thought, or hidden "
        "reasoning."
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
            "learning objective: "
            f"{getattr(candidate.proposal, 'learning_objective', '') or getattr(candidate.proposal, 'reason', '')}\n"
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
        "setup; (3) go in chronological order; (4) select the minimum complete span and never "
        "truncate required context or a conclusion; (5) line indices range from 0 to "
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
    return [
        unicodedata.normalize("NFKC", match.group(0))
        .translate(_APOSTROPHES)
        .casefold()
        for match in _WORD_RE.finditer(text or "")
    ]


def _contains_quote(text: str, quote: str) -> bool:
    haystack, needle = _toks(text), _toks(quote)
    if not needle or len(needle) > len(haystack):
        return False
    return any(haystack[i:i + len(needle)] == needle
               for i in range(len(haystack) - len(needle) + 1))


def _exact_boundary_quote(text: str, *, want: str) -> str:
    """Return an exact short quote from the retained transcript edge."""
    matches = list(_WORD_RE.finditer(text or ""))
    if not matches:
        return ""
    chosen = matches[:6] if want == "start" else matches[-6:]
    return (text or "")[chosen[0].start():chosen[-1].end()]


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
    from .discourse import first_lexical_character_index

    normalized = str(text or "").strip()
    lexical_index = first_lexical_character_index(normalized)
    if (
        ignore_caption_case
        and lexical_index is not None
        and normalized[lexical_index].islower()
    ):
        normalized = (
            normalized[:lexical_index]
            + normalized[lexical_index].upper()
            + normalized[lexical_index + 1:]
        )
    return normalized


def _cue_opens_mid_thought(text: str, *, ignore_caption_case: bool) -> bool:
    from .discourse import CONTEXT_DEP_HEADS, opens_mid_thought

    guarded = _guard_text(text, ignore_caption_case=ignore_caption_case)
    opening_clause = re.split(r"[.!?]", guarded, maxsplit=1)[0]
    existential = _EXISTENTIAL_OPENING_RE.fullmatch(opening_clause)
    if existential is not None:
        tail = existential.group("tail")
        tail_words = [word.casefold() for word in _WORD_RE.findall(tail)]
        that_count = sum(word == "that" for word in tail_words)
        has_contextual_definite = any(
            word == "the"
            and index + 1 < len(tail_words)
            and tail_words[index + 1] in CONTEXT_DEP_HEADS
            for index, word in enumerate(tail_words)
        )
        if (
            not _EXISTENTIAL_UNRESOLVED_RE.search(tail)
            and not _EXISTENTIAL_CONTEXTUAL_THAT_RE.search(tail)
            and not _EXISTENTIAL_TERMINAL_REFERENCE_RE.search(tail)
            and not _EXISTENTIAL_DEMONSTRATIVE_THAT_RE.search(tail)
            and that_count <= 1
            and not _EXISTENTIAL_BACK_REFERENCE_RE.search(tail)
            and not has_contextual_definite
            and _EXISTENTIAL_SCOPE_RE.search(tail)
        ):
            return False
    return opens_mid_thought(guarded)


def _cue_opens_mid_thought_at(
    segments: list[dict],
    index: int,
    *,
    ignore_caption_case: bool,
) -> bool:
    """Use the preceding cue to recover reliable lowercase-fragment evidence."""
    from .discourse import first_lexical_character_index

    text = str(segments[index].get("text") or "").strip()
    if _DANGLING_TAIL_PREFIX_RE.search(text):
        return True
    if _cue_opens_mid_thought(
        text, ignore_caption_case=ignore_caption_case
    ):
        return True
    if index > 0:
        previous_text = str(segments[index - 1].get("text") or "")
        if (
            not _cue_is_only_structural_filler(previous_text)
            and _cue_has_explicit_dangling_end(
                previous_text,
                text,
            )
        ):
            return True
    opening_terminator = re.search(r"[.!?]", text)
    lexical_index = first_lexical_character_index(text)
    starts_lowercase = bool(
        lexical_index is not None and text[lexical_index].islower()
    )
    if (
        not ignore_caption_case
        or index <= 0
        or not starts_lowercase
        or (
            opening_terminator is not None
            and opening_terminator.group(0) == "?"
        )
    ):
        return False

    previous_text = str(segments[index - 1].get("text") or "").strip()
    if re.search(r"[,;:\-—][\"')\]]*$", previous_text):
        return True
    words = _toks(text)
    if words and words[0] in {
        "by", "during", "from", "into", "of", "onto", "to", "while", "with",
        "without",
    }:
        return True

    from .sentences import classify_terminator

    previous_words = _toks(previous_text)
    explicit_closure_words = {
        "answer", "answered", "complete", "completed", "completely", "conclusion",
        "end", "ended", "final", "finished", "finishes", "result", "solved",
    }
    return bool(
        len(previous_words) >= 5
        and not classify_terminator(previous_text)
        and not explicit_closure_words.intersection(previous_words[-5:])
    )


def _cue_begins_standalone_question(text: str) -> bool:
    words = _toks(text)
    return bool(
        words
        and (
            words[0] in _STANDALONE_QUESTION_HEADS
            or (
                len(words) > 1
                and words[0] in _QUESTION_PREFIXES
                and words[1] in _STANDALONE_QUESTION_HEADS
            )
        )
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

    raw_text = str(text or "").strip()
    if _cue_has_explicit_dangling_end(raw_text, next_text):
        return True
    guarded = _guard_text(raw_text, ignore_caption_case=ignore_caption_case)
    terminator = classify_terminator(guarded)
    if terminator and _TERMINAL_STRANDED_PREPOSITION_RE.search(guarded):
        return False
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
        next_words
        and len(next_words[0]) > 5
        and next_words[0].endswith("ing")
    ):
        return True
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
    from .discourse import ANAPHORS, CONTEXT_DEP_HEADS, CONTINUATION_MARKERS, _AUX_VERB

    words = next_words
    if not words:
        return False
    if len(words) < 3 and words[0] in _AUX_VERB:
        return True
    return bool(
        words[0] in CONTINUATION_MARKERS
        or words[0] in ANAPHORS
        or (len(words) > 1 and words[0] == "the" and words[1] in CONTEXT_DEP_HEADS)
    )


def _cue_has_explicit_dangling_end(text: str, next_text: str) -> bool:
    """Detect syntax that cannot end a thought without broad caption guesses."""
    from .discourse import first_lexical_character_index

    raw_text = str(text or "").strip()
    next_lexical_index = first_lexical_character_index(next_text)
    ambiguous_degree_continues = bool(
        _TERMINAL_AMBIGUOUS_DEGREE_RE.search(raw_text)
        and next_lexical_index is not None
        and str(next_text)[next_lexical_index].islower()
    )
    return bool(
        _TERMINAL_CALLBACK_RE.search(raw_text)
        or _TERMINAL_DANGLING_TRANSITION_RE.search(raw_text)
        or _TERMINAL_INCOMPLETE_SUBJECT_RE.search(raw_text)
        or _TERMINAL_BARE_SUBJECT_RE.search(raw_text)
        or _TERMINAL_DANGLING_ARTICLE_RE.search(raw_text)
        or _TERMINAL_DANGLING_DEGREE_RE.search(raw_text)
        or ambiguous_degree_continues
        or _has_unfinished_exemplification_tail(raw_text)
        or re.search(r"[,;:\-—][\"')\]]*$", raw_text)
    )


def _has_unfinished_exemplification_tail(text: str) -> bool:
    """Detect a terminal example setup that never reaches its explanatory clause."""
    raw_text = str(text or "").strip()
    match = _TERMINAL_EXEMPLIFICATION_RE.search(raw_text)
    if match is None:
        return False
    tail_words = _toks(match.group("tail"))
    if not tail_words:
        return True
    if match.group("so") and len(tail_words) <= 3:
        return True
    return len(tail_words) <= 2 and not re.search(r"[.!?][\"')\]]*$", raw_text)


def _cue_boundary_confidence(text: str, *, ignore_caption_case: bool) -> float:
    from .sentences import classify_terminator

    guarded = _guard_text(text, ignore_caption_case=ignore_caption_case)
    return 1.0 if classify_terminator(guarded) else 0.90


def _trim_trailing_incomplete_suffix(
    segments: list[dict], start_line: int, end_line: int,
) -> int | None:
    """Trim a cue-aligned teaser/transition suffix, or reject if no clean prefix exists."""
    from .discourse import first_lexical_character_index
    from .sentences import classify_terminator

    following_text = (
        str(segments[end_line + 1].get("text") or "")
        if end_line + 1 < len(segments)
        else ""
    )
    solution_continues = bool(
        _FORWARD_SOLUTION_CONTINUATION_RE.search(following_text)
    )
    for line in range(end_line, start_line - 1, -1):
        suffix = _cue_clip_text(segments, line, end_line)
        dangling_transition = bool(
            _TERMINAL_DANGLING_TRANSITION_RE.search(suffix)
        )
        forward_setup = bool(
            solution_continues
            and _TRAILING_FORWARD_SETUP_RE.search(suffix)
        )
        dangling_degree = bool(_TERMINAL_DANGLING_DEGREE_RE.search(suffix))
        ambiguous_degree = _TERMINAL_AMBIGUOUS_DEGREE_RE.search(suffix)
        following_lexical_index = first_lexical_character_index(following_text)
        dangling_ambiguous_degree = bool(
            ambiguous_degree
            and following_lexical_index is not None
            and following_text[following_lexical_index].islower()
        )
        bare_subject = _TERMINAL_BARE_SUBJECT_RE.search(suffix)
        dangling_bare_subject = bool(
            bare_subject
            and re.search(r"[.!?]", suffix[:bare_subject.start()])
        )
        if not (
            dangling_transition
            or dangling_degree
            or dangling_ambiguous_degree
            or dangling_bare_subject
            or forward_setup
        ):
            continue
        previous_line = line - 1
        if previous_line < start_line:
            return end_line if forward_setup else None
        previous_text = str(segments[previous_line].get("text") or "").strip()
        if classify_terminator(previous_text):
            return previous_line
        return end_line if forward_setup else None
    return end_line


def _close_cue_context(
    segments: list[dict],
    start_line: int,
    end_line: int,
    *,
    ignore_caption_case: bool,
    cue_limit: int | None = None,
) -> tuple[int, int, str | None]:
    """Expand dirty edges until discourse closes or a real section edge is reached."""
    expansion_limit = (
        len(segments) if cue_limit is None else max(0, int(cue_limit))
    )

    def crosses_section_reset(left: int, right: int) -> bool:
        return (
            float(segments[right].get("start", 0.0))
            - float(segments[left].get("end", 0.0))
            >= _SECTION_RESET_GAP_S
        )

    following_text = (
        str(segments[end_line + 1].get("text") or "")
        if end_line + 1 < len(segments)
        else ""
    )
    forward_solution_needed = bool(
        _FORWARD_SOLUTION_CONTINUATION_RE.search(following_text)
        and any(
            _TRAILING_FORWARD_SETUP_RE.search(
                _cue_clip_text(segments, line, end_line)
            )
            for line in range(start_line, end_line + 1)
        )
    )
    trimmed_end = _trim_trailing_incomplete_suffix(
        segments, start_line, end_line
    )
    if trimmed_end is None:
        return start_line, end_line, "unresolved_weak_end"
    suffix_was_trimmed = trimmed_end < end_line
    end_line = trimmed_end
    force_end_clause_completion = bool(
        _TERMINAL_INCOMPLETE_SUBJECT_RE.search(
            str(segments[end_line].get("text") or "").strip()
        )
    )
    from .sentences import classify_terminator

    force_start_question_setup = bool(
        start_line > 0
        and _OPENING_COMPARATIVE_FRAGMENT_RE.search(
            str(segments[start_line].get("text") or "").strip()
        )
        and not classify_terminator(
            str(segments[start_line - 1].get("text") or "").strip()
        )
    )
    if _DANGLING_TAIL_PREFIX_RE.search(
        str(segments[start_line].get("text") or "").strip()
    ):
        for candidate in range(start_line + 1, end_line + 1):
            if crosses_section_reset(candidate - 1, candidate) and not _cue_opens_mid_thought_at(
                segments,
                candidate,
                ignore_caption_case=ignore_caption_case,
            ):
                start_line = candidate
                break
    selected_start_line = start_line
    from .discourse import _has_unresolved_opening_back_reference

    selected_start_text = str(segments[selected_start_line].get("text") or "")
    opening_reference_requires_context = _has_unresolved_opening_back_reference(
        selected_start_text
    )

    def opening_reference_is_unresolved() -> bool:
        if not opening_reference_requires_context:
            return False
        prior_text = (
            _cue_clip_text(segments, start_line, selected_start_line - 1)
            if start_line < selected_start_line
            else ""
        )
        return _has_unresolved_opening_back_reference(
            selected_start_text,
            prior_text=prior_text,
        )

    original_end = end_line
    start_expansions = 0
    while start_expansions < expansion_limit:
        if force_start_question_setup and _cue_begins_standalone_question(
            str(segments[start_line].get("text") or "")
        ):
            force_start_question_setup = False
            break
        if (
            not force_start_question_setup
            and not _cue_opens_mid_thought_at(
                segments,
                start_line,
                ignore_caption_case=ignore_caption_case,
            )
            and not opening_reference_is_unresolved()
        ):
            break
        candidate = start_line - 1
        if candidate < 0:
            break
        if crosses_section_reset(candidate, start_line):
            break
        start_line = candidate
        start_expansions += 1
    if force_start_question_setup or _cue_opens_mid_thought_at(
        segments,
        start_line,
        ignore_caption_case=ignore_caption_case,
    ) or opening_reference_is_unresolved():
        return start_line, end_line, "unresolved_weak_start"

    consumed_end_cues = 0
    if forward_solution_needed and not suffix_was_trimmed:
        candidate = end_line + 1
        if candidate >= len(segments) or expansion_limit <= 0:
            return start_line, end_line, "unresolved_weak_end"
        if crosses_section_reset(end_line, candidate):
            return start_line, end_line, "unresolved_weak_end"
        end_line = candidate
        consumed_end_cues = 1

    end_cue_limit = (
        0 if suffix_was_trimmed else max(0, expansion_limit - consumed_end_cues)
    )
    end_expansions = 0
    while end_expansions < end_cue_limit:
        current_end_text = str(segments[end_line].get("text") or "")
        if force_end_clause_completion and classify_terminator(current_end_text):
            force_end_clause_completion = False
        next_text = (
            str(segments[end_line + 1].get("text") or "")
            if end_line + 1 < len(segments)
            else ""
        )
        if not force_end_clause_completion and not _cue_has_weak_end(
            current_end_text,
            next_text,
            ignore_caption_case=ignore_caption_case,
        ):
            break
        candidate = end_line + 1
        if candidate >= len(segments):
            break
        if crosses_section_reset(end_line, candidate):
            break
        end_line = candidate
        end_expansions += 1
    next_text = "" if suffix_was_trimmed else (
        str(segments[end_line + 1].get("text") or "")
        if end_line + 1 < len(segments)
        else ""
    )
    final_end_text = (
        _cue_clip_text(segments, original_end, end_line)
        if end_line > original_end
        else str(segments[end_line].get("text") or "")
    )
    if _cue_has_weak_end(
        final_end_text,
        next_text,
        ignore_caption_case=ignore_caption_case,
    ) or force_end_clause_completion:
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


def _cue_clip_text(segments: list[dict], start_line: int, end_line: int) -> str:
    return " ".join(
        str(segment.get("text") or "").strip()
        for segment in segments[start_line:end_line + 1]
        if str(segment.get("text") or "").strip()
    ).strip()


_FILLER_REMAINDER_WORDS = frozenset({
    "a", "an", "and", "back", "for", "from", "have", "hope", "i", "ll", "no",
    "our", "that", "the", "thanks", "thank", "to", "today", "we", "we'll", "will", "you",
    "your",
})


def _structural_filler_matches(text: str) -> list[re.Match[str]]:
    return list(_STRUCTURAL_FILLER_RE.finditer(str(text or "")))


def _structural_match_is_edge(
    text: str,
    match: re.Match[str],
    *,
    want: str,
) -> bool:
    before = text[:match.start()]
    after = text[match.end():]
    is_non_speech_marker = bool(
        _NON_SPEECH_MARKER_RE.fullmatch(match.group(0))
    )
    if want == "start":
        # A caption marker is removable only when it actually precedes speech.
        # This keeps a brief marker between teaching clauses inside the clip.
        clean_prefix = (
            not _WORD_RE.search(before)
            if is_non_speech_marker
            else (
                not re.search(r"[.!?]", before)
                and len(_toks(before)) <= 5
            )
        )
        return bool(clean_prefix and _WORD_RE.search(after))
    return bool(_WORD_RE.search(before) and not _WORD_RE.search(after))


def _cue_is_only_structural_filler(text: str) -> bool:
    matches = _structural_filler_matches(text)
    if not matches:
        return False
    remainder = list(str(text or ""))
    for match in matches:
        remainder[match.start():match.end()] = " " * (match.end() - match.start())
    return set(_toks("".join(remainder))).issubset(_FILLER_REMAINDER_WORDS)


def _quote_character_spans(text: str, quote: str) -> list[tuple[int, int]]:
    text_matches = list(_WORD_RE.finditer(str(text or "")))
    quote_tokens = _toks(quote)
    if not quote_tokens or len(quote_tokens) > len(text_matches):
        return []
    text_tokens = [
        unicodedata.normalize("NFKC", match.group(0))
        .translate(_APOSTROPHES)
        .casefold()
        for match in text_matches
    ]
    spans: list[tuple[int, int]] = []
    for index in range(len(text_tokens) - len(quote_tokens) + 1):
        if text_tokens[index:index + len(quote_tokens)] == quote_tokens:
            spans.append((
                text_matches[index].start(),
                text_matches[index + len(quote_tokens) - 1].end(),
            ))
    return spans


def _quote_character_span(text: str, quote: str) -> tuple[int, int] | None:
    spans = _quote_character_spans(text, quote)
    return spans[0] if spans else None


def _literal_source_quote(
    text: str,
    quote: str,
    span: tuple[int, int],
) -> str:
    """Return the matched source spelling, including exact requested punctuation."""
    source = str(text or "")
    quote_matches = list(_WORD_RE.finditer(str(quote or "")))
    start, end = span
    if quote_matches:
        prefix = str(quote or "")[:quote_matches[0].start()]
        suffix = str(quote or "")[quote_matches[-1].end():]
        if prefix and start >= len(prefix) and source[start - len(prefix):start] == prefix:
            start -= len(prefix)
        if suffix and source[end:end + len(suffix)] == suffix:
            end += len(suffix)
    return source[start:end]


def _semantic_edge_quote(
    text: str,
    quote: str,
    *,
    want: str,
) -> tuple[tuple[int, int] | None, bool, str | None]:
    """Ground one semantic edge without inventing a timestamp.

    The edge occurrence nearest the physical cue edge is authoritative when the quote
    already begins/ends the cue. A quote that excludes real words requires projection and
    therefore must have exactly one normalized occurrence in that cue.
    """
    word_matches = list(_WORD_RE.finditer(str(text or "")))
    spans = _quote_character_spans(text, quote)
    if not word_matches or not spans:
        return None, False, "ungrounded_boundary_quote"
    span = spans[0] if want == "start" else spans[-1]
    projected = bool(
        _WORD_RE.search(text[:span[0]])
        if want == "start"
        else _WORD_RE.search(text[span[1]:])
    )
    if projected and len(spans) != 1:
        return None, True, f"ambiguous_{want}_quote"
    return span, projected, None


def _expanded_context_edge_quote(
    text: str,
    *,
    want: str,
) -> tuple[str, str | None]:
    """Choose an exact edge quote after safely removable filler-only sentences."""
    raw_text = str(text or "")
    sentence_spans = [
        (match.start(), match.end())
        for match in re.finditer(r"[^.!?]+(?:[.!?]+|$)", raw_text)
        if _WORD_RE.search(match.group(0))
    ]
    if not sentence_spans:
        return "", "empty_expanded_context_edge"

    retained_left = 0
    retained_right = len(raw_text)
    ordered_spans = sentence_spans if want == "start" else list(reversed(sentence_spans))
    for left, right in ordered_spans:
        sentence = raw_text[left:right]
        matches = _structural_filler_matches(sentence)
        if not matches:
            break
        if _cue_is_only_structural_filler(sentence):
            if want == "start":
                retained_left = right
            else:
                retained_right = left
            continue

        edge_matches = [
            match
            for match in matches
            if _structural_match_is_edge(sentence, match, want=want)
        ]
        if edge_matches:
            inline_boundary_applied = False
            if want == "start":
                match = max(edge_matches, key=lambda value: value.end())
                separator_pattern = (
                    r"(?:\s+|\s*[,;:—-]\s*)"
                    if _NON_SPEECH_MARKER_RE.fullmatch(match.group(0))
                    else r"\s*[,;:—-]\s*"
                )
                separator = re.match(separator_pattern, sentence[match.end():])
                if separator is not None:
                    retained_left = left + match.end() + separator.end()
                    inline_boundary_applied = True
            else:
                match = min(edge_matches, key=lambda value: value.start())
                separator_pattern = (
                    r"(?:\s+|[,;:—-]\s*)$"
                    if _NON_SPEECH_MARKER_RE.fullmatch(match.group(0))
                    else r"[,;:—-]\s*$"
                )
                separator = re.search(separator_pattern, sentence[:match.start()])
                if separator is not None:
                    retained_right = left + separator.start()
                    inline_boundary_applied = True
            if inline_boundary_applied:
                break
            return "", "unresolved_expanded_edge_filler"
        break

    retained_words = [
        match
        for match in _WORD_RE.finditer(raw_text)
        if retained_left <= match.start() and match.end() <= retained_right
    ]
    if not retained_words:
        return "", "empty_expanded_context_edge"
    chosen = retained_words[:6] if want == "start" else retained_words[-6:]
    return raw_text[chosen[0].start():chosen[-1].end()], None


def _edge_has_unresolved_structural_filler(
    text: str,
    quote_span: tuple[int, int],
    *,
    want: str,
) -> bool:
    """Reject structural filler that the semantic quote still leaves on an edge."""
    for match in _structural_filler_matches(text):
        is_non_speech_marker = bool(
            _NON_SPEECH_MARKER_RE.fullmatch(match.group(0))
        )
        if (
            want == "start"
            and is_non_speech_marker
            and quote_span[0] < match.end()
        ):
            selected_prefix = (
                text[quote_span[0]:match.start()]
                if quote_span[0] < match.start()
                else ""
            )
            if (
                not _WORD_RE.search(selected_prefix)
                or _cue_is_only_structural_filler(selected_prefix)
            ):
                return True
        is_edge_match = _structural_match_is_edge(text, match, want=want)
        if want == "start" and is_edge_match and quote_span[0] < match.end():
            return True
        if want == "end" and is_edge_match and quote_span[1] > match.start():
            return True
    return False


def _replace_structural_edge_quote(
    text: str,
    quote: str,
    *,
    want: str,
) -> tuple[str, bool, str | None]:
    """Replace a model quote that includes removable filler at a cue edge."""
    if _cue_is_only_structural_filler(text):
        # The existing cue-level pass can remove this cue without projecting a
        # boundary inside it. Same-cue repair is only for mixed teaching/filler.
        return quote, False, None
    quote_span, _projected, error = _semantic_edge_quote(text, quote, want=want)
    if error or quote_span is None:
        # Preserve the normal grounding/ambiguity failure emitted downstream.
        return quote, False, None
    if not _edge_has_unresolved_structural_filler(text, quote_span, want=want):
        return quote, False, None

    replacement, error = _expanded_context_edge_quote(text, want=want)
    if error:
        return "", False, "unresolved_edge_filler"
    replacement_span, projected, error = _semantic_edge_quote(
        text, replacement, want=want
    )
    if error or replacement_span is None or not projected:
        return "", False, error or "unresolved_edge_filler"
    if _edge_has_unresolved_structural_filler(
        text, replacement_span, want=want
    ):
        return "", False, "unresolved_edge_filler"
    return replacement, True, None


def _semantic_clip_slice(
    segments: list[dict],
    start_line: int,
    end_line: int,
    *,
    start_span: tuple[int, int] | None,
    end_span: tuple[int, int] | None,
) -> tuple[str, dict[str, tuple[int, int]]]:
    """Return transcript speech between projected edge quotes, including internal asides."""
    parts: list[str] = []
    spans_by_cue: dict[str, tuple[int, int]] = {}
    for line in range(start_line, end_line + 1):
        text = str(segments[line].get("text") or "")
        left = start_span[0] if line == start_line and start_span is not None else 0
        right = end_span[1] if line == end_line and end_span is not None else len(text)
        if right <= left:
            return "", {}
        cue_id = str(segments[line].get("cue_id") or f"cue-{line}")
        spans_by_cue[cue_id] = (left, right)
        selected = text[left:right].strip()
        if selected:
            parts.append(selected)
    return " ".join(parts).strip(), spans_by_cue


def _trim_structural_filler_edges(
    segments: list[dict], start_line: int, end_line: int,
    *,
    ignore_caption_case: bool,
) -> tuple[int, int] | None:
    """Trim contiguous edge filler without discarding teaching around an aside."""
    filler_lines = {
        line
        for line in range(start_line, end_line + 1)
        if _cue_is_only_structural_filler(str(segments[line].get("text") or ""))
    }
    if not filler_lines:
        return start_line, end_line

    trimmed_start = start_line
    while trimmed_start <= end_line and trimmed_start in filler_lines:
        trimmed_start += 1
    trimmed_end = end_line
    while trimmed_end >= trimmed_start and trimmed_end in filler_lines:
        trimmed_end -= 1
    if trimmed_start > trimmed_end:
        return None
    if _cue_opens_mid_thought_at(
        segments, trimmed_start, ignore_caption_case=ignore_caption_case,
    ):
        return None
    trailing_text = str(segments[trimmed_end].get("text") or "").strip()
    if _cue_has_weak_end(
        trailing_text,
        "",
        ignore_caption_case=ignore_caption_case,
    ):
        return None
    return trimmed_start, trimmed_end


def _internal_structural_filler_reason(
    segments: list[dict], start_line: int, end_line: int,
) -> str | None:
    """Tolerate brief internal interruptions, but reject substantial filler."""
    filler_lines = [
        line
        for line in range(start_line + 1, end_line)
        if (
            _cue_is_only_structural_filler(
                str(segments[line].get("text") or "")
            )
            or _INTERNAL_INTERRUPTION_MARKER_RE.search(
                str(segments[line].get("text") or "")
            )
        )
    ]
    if not filler_lines:
        return None

    duration = 0.0
    word_count = 0
    for line in filler_lines:
        segment = segments[line]
        try:
            start = float(segment.get("start") or 0.0)
            end = float(segment.get("end") or start)
        except (TypeError, ValueError, OverflowError):
            return "long_internal_filler_block"
        if not math.isfinite(start) or not math.isfinite(end) or end < start:
            return "long_internal_filler_block"
        duration += end - start
        word_count += len(_toks(str(segment.get("text") or "")))
    if (
        duration > _MAX_INTERNAL_FILLER_DURATION_S
        or word_count > _MAX_INTERNAL_FILLER_WORDS
    ):
        return "long_internal_filler_block"
    return None


def _same_cue_internal_filler_reason(text: str) -> str | None:
    """Apply the existing filler budget to interruption blocks inside coarse cues."""
    raw_text = str(text or "")
    blocks: set[tuple[int, int]] = set()
    for match in _INTERNAL_INTERRUPTION_MARKER_RE.finditer(raw_text):
        if not _WORD_RE.search(raw_text[:match.start()]):
            continue
        if not _WORD_RE.search(raw_text[match.end():]):
            continue
        previous = max(
            raw_text.rfind(".", 0, match.start()),
            raw_text.rfind("!", 0, match.start()),
            raw_text.rfind("?", 0, match.start()),
        )
        following = re.search(r"[.!?]", raw_text[match.end():])
        right = (
            match.end() + following.end()
            if following is not None
            else len(raw_text)
        )
        blocks.add((previous + 1, right))
    if sum(
        len(_toks(raw_text[left:right]))
        for left, right in blocks
    ) > _MAX_INTERNAL_FILLER_WORDS:
        return "long_internal_filler_block"
    return None


def _clip_requires_visual_context(text: str) -> bool:
    return bool(_VISUAL_DEPENDENCY_RE.search(str(text or "")))


def _near_duplicate(a: dict, b: dict, threshold: float = _DUPLICATE_OVERLAP) -> bool:
    a_cue_ids = {
        str(value).strip()
        for value in (a.get("cue_ids") or [])
        if str(value).strip()
    }
    b_cue_ids = {
        str(value).strip()
        for value in (b.get("cue_ids") or [])
        if str(value).strip()
    }
    if a_cue_ids and b_cue_ids:
        shared = a_cue_ids & b_cue_ids
        if not shared:
            return False
        a_semantic_spans = a.get("_semantic_spans_by_cue")
        b_semantic_spans = b.get("_semantic_spans_by_cue")
        if not isinstance(a_semantic_spans, dict) or not isinstance(
            b_semantic_spans, dict
        ):
            return True
        for cue_id in shared:
            a_span = a_semantic_spans.get(cue_id)
            b_span = b_semantic_spans.get(cue_id)
            if (
                isinstance(a_span, tuple)
                and len(a_span) == 2
                and isinstance(b_span, tuple)
                and len(b_span) == 2
                and max(a_span[0], b_span[0]) < min(a_span[1], b_span[1])
            ):
                return True
        return False

    overlap = min(float(a["end"]), float(b["end"])) - max(float(a["start"]), float(b["start"]))
    if overlap <= 0:
        return False
    shorter = min(float(a["end"]) - float(a["start"]),
                  float(b["end"]) - float(b["start"]))
    return shorter > 0 and overlap / shorter >= threshold


def _semantic_restatement(
    a: dict,
    b: dict,
    threshold: float = _DUPLICATE_OVERLAP,
) -> bool:
    """Match reworded copies by their normalized objective and facet."""
    generic = {
        "complete", "concept", "example", "explain", "idea", "learn",
        "lesson", "point", "teach", "understand", "work",
    }
    a_tokens = _content_tokens(
        f"{a.get('learning_objective', '')} {a.get('facet', '')}"
    ) - generic
    b_tokens = _content_tokens(
        f"{b.get('learning_objective', '')} {b.get('facet', '')}"
    ) - generic
    smaller = min(len(a_tokens), len(b_tokens))
    if smaller < 2:
        return False
    shared = len(a_tokens & b_tokens)
    return shared >= 2 and shared / smaller >= threshold


def _duplicates(a: dict, b: dict) -> bool:
    return _near_duplicate(a, b) or _semantic_restatement(a, b)


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


def _range_repair_lost_support(
    text: str,
    *,
    original_text: str,
    retained_text: str,
) -> bool:
    generated_tokens = _content_tokens(text)
    original_support = generated_tokens & _content_tokens(original_text)
    retained_support = generated_tokens & _content_tokens(retained_text)
    return bool(original_support - retained_support)


def _objective_after_range_repair(
    objective: str,
    *,
    original_text: str,
    retained_text: str,
    evidence_quote: str,
    require_grounding: bool = False,
) -> str:
    """Remove claims whose only transcript support was cut by range repair."""
    lost_support = _range_repair_lost_support(
        objective,
        original_text=original_text,
        retained_text=retained_text,
    )
    if not lost_support and not (
        require_grounding and not _text_has_grounding(objective, retained_text)
    ):
        return objective
    grounded = " ".join(str(evidence_quote or "").split()).rstrip(" .!?;:")
    if not grounded:
        return objective
    return f"Understand this transcript-grounded point: {grounded}."


def _title_after_range_repair(
    title: str,
    *,
    original_text: str,
    retained_text: str,
    evidence_quote: str,
    require_grounding: bool = False,
) -> str:
    """Keep repaired clip labels within the teaching claim that remains."""
    lost_support = _range_repair_lost_support(
        title,
        original_text=original_text,
        retained_text=retained_text,
    )
    if not lost_support and not (
        require_grounding and not _text_has_grounding(title, retained_text)
    ):
        return title
    grounded_words = " ".join(str(evidence_quote or "").split()).split()
    if not grounded_words:
        return title
    grounded = " ".join(grounded_words[:10]).rstrip(" .!?;:")
    return grounded[:80]


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


def _configured_clip_limit(settings: dict) -> int | None:
    configured = settings.get("max_clips")
    return None if configured is None else max(0, int(configured))


def _quality_order(clip: dict) -> tuple[float, float, float]:
    scores = (
        float(clip.get("informativeness") or 0.0),
        float(clip.get("topic_relevance") or 0.0),
        float(clip.get("educational_importance") or 0.0),
    )
    return min(scores), sum(scores) / len(scores), scores[1]


def _difficulty_stage(difficulty: object) -> int:
    score = float(difficulty or 0.0)
    return 0 if score < 0.34 else 1 if score < 0.67 else 2


def _finalize_clips(clips: list[dict], settings: dict) -> list[dict]:
    """Keep the strongest restatement and stage every qualifying candidate."""
    quality_order = sorted(
        clips,
        key=lambda clip: (
            -_quality_order(clip)[0],
            -_quality_order(clip)[1],
            -_quality_order(clip)[2],
            float(clip["start"]),
            float(clip["end"]),
            int(clip.get("_proposal_index") or 0),
            str(clip.get("selection_candidate_id") or ""),
        ),
    )
    kept: list[dict] = []
    for candidate in quality_order:
        if not any(_duplicates(candidate, prior) for prior in kept):
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
        if limit is not None and len(selected) + len(additions) > limit:
            continue
        for item in additions:
            selected.append(item)
            item_id = str(item.get("selection_candidate_id") or "")
            if item_id:
                selected_ids.add(item_id)
        if limit is not None and len(selected) >= limit:
            break
    selected.sort(
        key=lambda clip: (
            _difficulty_stage(clip.get("difficulty")),
            -_quality_order(clip)[0],
            -_quality_order(clip)[1],
            -_quality_order(clip)[2],
            float(clip["start"]),
            float(clip["end"]),
            int(clip.get("_proposal_index") or 0),
            str(clip.get("selection_candidate_id") or ""),
        )
    )
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
    context_cue_limit: int | None = None,
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
        proposed_start, proposed_end = a, b
        start_quote = str(proposal.start_quote or "").strip()
        end_quote = str(proposal.end_quote or "").strip()
        start_text = str(segments[a].get("text") or "").strip()
        end_text = str(segments[b].get("text") or "").strip()
        quote_repaired = False
        if not _contains_quote(start_text, start_quote):
            matching_lines = [
                line
                for line in range(proposed_start, proposed_end + 1)
                if _contains_quote(
                    str(segments[line].get("text") or ""), start_quote
                )
            ]
            if len(matching_lines) != 1:
                report.rejected_reasons.append(f"{prefix}:bad_start_quote")
                continue
            anchored_line = matching_lines[0]
            if any(
                not _cue_is_only_structural_filler(
                    str(segments[line].get("text") or "")
                )
                for line in range(proposed_start, anchored_line)
            ):
                report.rejected_reasons.append(f"{prefix}:bad_start_quote")
                continue
            a = anchored_line
            start_text = str(segments[a].get("text") or "").strip()
            quote_repaired = True
        if not _contains_quote(end_text, end_quote):
            matching_lines = [
                line
                for line in range(proposed_start, proposed_end + 1)
                if _contains_quote(
                    str(segments[line].get("text") or ""), end_quote
                )
            ]
            if len(matching_lines) != 1:
                report.rejected_reasons.append(f"{prefix}:bad_end_quote")
                continue
            anchored_line = matching_lines[0]
            if any(
                not _cue_is_only_structural_filler(
                    str(segments[line].get("text") or "")
                )
                for line in range(anchored_line + 1, proposed_end + 1)
            ):
                report.rejected_reasons.append(f"{prefix}:bad_end_quote")
                continue
            b = anchored_line
            end_text = str(segments[b].get("text") or "").strip()
            quote_repaired = True
        if a > b:
            report.rejected_reasons.append(f"{prefix}:reversed_quote_order")
            continue
        selected_start_before_context = a
        selected_end_before_context = b
        context_repair_source_text = _cue_clip_text(
            segments, a, min(n - 1, b + 1)
        )
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
        below_green = next(
            (
                name
                for name, score in (
                    ("informativeness", info),
                    ("topic_relevance", relevance),
                    ("educational_importance", importance),
                )
                if score < _GREEN_SCORE
            ),
            None,
        )
        if below_green is not None:
            report.rejected_reasons.append(f"{prefix}:{below_green}_below_green")
            continue
        if proposal.self_contained is not True:
            report.rejected_reasons.append(f"{prefix}:not_self_contained")
            continue
        if isinstance(proposal, _BoundaryTopic):
            if proposal.directly_teaches_topic is not True:
                report.rejected_reasons.append(f"{prefix}:does_not_directly_teach_topic")
                continue
            if proposal.substantive is not True:
                report.rejected_reasons.append(f"{prefix}:not_substantive")
                continue
            if proposal.factually_grounded is not True:
                report.rejected_reasons.append(f"{prefix}:not_factually_grounded")
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
        if not is_standalone or prerequisites:
            report.rejected_reasons.append(f"{prefix}:not_standalone")
            continue
        uncertainty = str(getattr(proposal, "uncertainty", "low") or "low")
        uncertainty_reasons = [str(getattr(reason, "value", reason))
                               for reason in (getattr(proposal, "uncertainty_reasons", None) or [])]
        if uncertainty == "high":
            report.rejected_reasons.append(f"{prefix}:{uncertainty}_uncertainty")
            continue

        start_quote, repaired_start_edge, edge_error = _replace_structural_edge_quote(
            start_text,
            start_quote,
            want="start",
        )
        if edge_error:
            report.rejected_reasons.append(f"{prefix}:{edge_error}")
            continue
        end_quote, repaired_end_edge, edge_error = _replace_structural_edge_quote(
            end_text,
            end_quote,
            want="end",
        )
        if edge_error:
            report.rejected_reasons.append(f"{prefix}:{edge_error}")
            continue
        quote_repaired = quote_repaired or repaired_start_edge or repaired_end_edge

        # Run discourse closure against the teaching slice the model selected, not
        # against hook/joke text that is deliberately outside its exact edge quotes.
        closure_segments = segments
        preliminary_start_spans = _quote_character_spans(start_text, start_quote)
        preliminary_end_spans = _quote_character_spans(end_text, end_quote)
        if len(preliminary_start_spans) == 1 or len(preliminary_end_spans) == 1:
            closure_segments = [dict(segment) for segment in segments]
            start_left = (
                preliminary_start_spans[0][0]
                if len(preliminary_start_spans) == 1
                else 0
            )
            end_right = (
                preliminary_end_spans[0][1]
                if len(preliminary_end_spans) == 1
                else len(end_text)
            )
            if a == b:
                if start_left < end_right:
                    closure_segments[a]["text"] = start_text[start_left:end_right]
            else:
                closure_segments[a]["text"] = start_text[start_left:]
                closure_segments[b]["text"] = end_text[:end_right]

        a, b, closure_error = _close_cue_context(
            closure_segments,
            a,
            b,
            ignore_caption_case=ignore_caption_case,
            cue_limit=context_cue_limit,
        )
        if closure_error:
            report.rejected_reasons.append(f"{prefix}:{closure_error}")
            continue

        filler_trim = _trim_structural_filler_edges(
            segments,
            a,
            b,
            ignore_caption_case=ignore_caption_case,
        )
        if filler_trim is None:
            report.rejected_reasons.append(f"{prefix}:contains_filler")
            continue
        a, b = filler_trim
        internal_filler_reason = _internal_structural_filler_reason(segments, a, b)
        if internal_filler_reason:
            report.rejected_reasons.append(f"{prefix}:{internal_filler_reason}")
            continue
        context_was_trimmed = b < selected_end_before_context
        start, end = _padded_cue_bounds(segments, a, b)
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            report.rejected_reasons.append(f"{prefix}:reversed_cue_boundary")
            continue
        start, end = round(start, 3), round(end, 3)

        full_clip_text = _cue_clip_text(segments, a, b)
        if not full_clip_text:
            report.rejected_reasons.append(f"{prefix}:empty_cue_transcript")
            continue
        if not _contains_quote(full_clip_text, start_quote):
            if type(proposal) is _BoundaryTopic:
                report.rejected_reasons.append(f"{prefix}:ungrounded_boundary_quote")
                continue
            start_quote = _exact_boundary_quote(full_clip_text, want="start")
            quote_repaired = True
        if not _contains_quote(full_clip_text, end_quote):
            if type(proposal) is _BoundaryTopic:
                report.rejected_reasons.append(f"{prefix}:ungrounded_boundary_quote")
                continue
            end_quote = _exact_boundary_quote(full_clip_text, want="end")
            quote_repaired = True
        if not start_quote or not end_quote:
            report.rejected_reasons.append(f"{prefix}:ungrounded_boundary_quote")
            continue

        start_span: tuple[int, int] | None = None
        end_span: tuple[int, int] | None = None
        if a != selected_start_before_context:
            start_quote, edge_error = _expanded_context_edge_quote(
                str(segments[a].get("text") or ""), want="start"
            )
            if edge_error:
                report.rejected_reasons.append(f"{prefix}:{edge_error}")
                continue
            quote_repaired = True
        start_span, start_projected, edge_error = _semantic_edge_quote(
            str(segments[a].get("text") or ""), start_quote, want="start"
        )
        if edge_error:
            report.rejected_reasons.append(f"{prefix}:{edge_error}")
            continue
        assert start_span is not None
        start_text = str(segments[a].get("text") or "")
        start_quote = _literal_source_quote(start_text, start_quote, start_span)
        if _edge_has_unresolved_structural_filler(
            str(segments[a].get("text") or ""), start_span, want="start"
        ):
            report.rejected_reasons.append(f"{prefix}:unresolved_edge_filler")
            continue

        if b != selected_end_before_context:
            end_quote, edge_error = _expanded_context_edge_quote(
                str(segments[b].get("text") or ""), want="end"
            )
            if edge_error:
                report.rejected_reasons.append(f"{prefix}:{edge_error}")
                continue
            quote_repaired = True
        end_span, end_projected, edge_error = _semantic_edge_quote(
            str(segments[b].get("text") or ""), end_quote, want="end"
        )
        if edge_error:
            report.rejected_reasons.append(f"{prefix}:{edge_error}")
            continue
        assert end_span is not None
        end_text = str(segments[b].get("text") or "")
        end_quote = _literal_source_quote(end_text, end_quote, end_span)
        if _edge_has_unresolved_structural_filler(
            str(segments[b].get("text") or ""), end_span, want="end"
        ):
            report.rejected_reasons.append(f"{prefix}:unresolved_edge_filler")
            continue

        clip_text, semantic_spans_by_cue = _semantic_clip_slice(
            segments,
            a,
            b,
            start_span=start_span if start_projected else None,
            end_span=end_span if end_projected else None,
        )
        if not clip_text:
            report.rejected_reasons.append(f"{prefix}:reversed_semantic_boundary")
            continue
        if _contains_unrequested_vampire_pseudoscience(clip_text, topic):
            report.rejected_reasons.append(f"{prefix}:fictional_framing")
            continue
        same_cue_filler_reason = _same_cue_internal_filler_reason(clip_text)
        if same_cue_filler_reason:
            report.rejected_reasons.append(f"{prefix}:{same_cue_filler_reason}")
            continue
        if _clip_requires_visual_context(clip_text):
            report.rejected_reasons.append(f"{prefix}:requires_visual_context")
            continue
        topic_evidence_quote = " ".join(
            str(getattr(proposal, "topic_evidence_quote", "") or "").split()
        )
        if isinstance(proposal, _BoundaryTopic):
            evidence_word_count = len(_toks(topic_evidence_quote))
            if evidence_word_count < 5 or evidence_word_count > 40:
                report.rejected_reasons.append(f"{prefix}:invalid_topic_evidence_quote_length")
                continue
            evidence_span = _quote_character_span(clip_text, topic_evidence_quote)
            if evidence_span is None:
                report.rejected_reasons.append(f"{prefix}:ungrounded_topic_evidence_quote")
                continue
            topic_evidence_quote = _literal_source_quote(
                clip_text,
                topic_evidence_quote,
                evidence_span,
            )
        learning_objective = str(
            getattr(proposal, "learning_objective", "")
            or getattr(proposal, "reason", "")
            or proposal.title
        ).strip()
        clip_title = str(proposal.title or "").strip()
        clip_facet = str(proposal.facet or "").strip()
        clip_reason = str(
            getattr(proposal, "reason", "") or learning_objective
        ).strip()
        if context_was_trimmed:
            repair_source_text = context_repair_source_text
            require_grounding = context_was_trimmed
            learning_objective = _objective_after_range_repair(
                learning_objective,
                original_text=repair_source_text,
                retained_text=clip_text,
                evidence_quote=topic_evidence_quote,
                require_grounding=require_grounding,
            )
            clip_title = _title_after_range_repair(
                clip_title,
                original_text=repair_source_text,
                retained_text=clip_text,
                evidence_quote=topic_evidence_quote,
                require_grounding=require_grounding,
            )
            clip_facet = _title_after_range_repair(
                clip_facet,
                original_text=repair_source_text,
                retained_text=clip_text,
                evidence_quote=topic_evidence_quote,
                require_grounding=require_grounding,
            )
            clip_reason = _objective_after_range_repair(
                clip_reason,
                original_text=repair_source_text,
                retained_text=clip_text,
                evidence_quote=topic_evidence_quote,
                require_grounding=require_grounding,
            )
        cue_ids = [
            str(segments[line].get("cue_id") or f"cue-{line}")
            for line in range(a, b + 1)
        ]
        edge_projection: dict[str, dict[str, object]] = {}
        if start_projected:
            edge_projection["start"] = {
                "required": True,
                "cue_id": cue_ids[0],
                "quote": start_quote,
            }
        if end_projected:
            edge_projection["end"] = {
                "required": True,
                "cue_id": cue_ids[-1],
                "quote": end_quote,
            }
        clip_id = f"clip-{index + 1:03d}-{a}-{b}"
        clip = {
            "start": start,
            "end": end,
            "start_quote": start_quote,
            "end_quote": end_quote,
            "title": clip_title,
            "learning_objective": learning_objective,
            "facet": clip_facet,
            "reason": clip_reason,
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
            "_proposal_index": index,
            "_semantic_spans_by_cue": semantic_spans_by_cue,
            "_quote_repaired": quote_repaired,
            "directly_teaches_topic": bool(
                getattr(proposal, "directly_teaches_topic", True)
            ),
            "substantive": bool(getattr(proposal, "substantive", True)),
            "factually_grounded": bool(getattr(proposal, "factually_grounded", True)),
            "topic_evidence_quote": topic_evidence_quote,
            "summary": "",
            "takeaways": [],
            "match_reason": "",
            "assessment": None,
        }
        if edge_projection:
            clip["edge_projection"] = edge_projection
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
        min(
            float(clip["informativeness"]),
            float(clip["topic_relevance"]),
            float(clip["educational_importance"]),
        )
        < _GREEN_SCORE
        for clip in raw
    )
    for i, candidate in enumerate(raw):
        if any(_duplicates(candidate, other) for other in raw[i + 1:]):
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
        if report.score_below_green:
            return _Classification("invalid", ("quality_score_below_green",))
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


def _validate_model_response(
    schema: type[BaseModel], text: str,
) -> tuple[BaseModel, list[str]]:
    """Validate one response, salvaging valid boundary candidates independently."""
    if schema is not _BoundaryPlan:
        return schema.model_validate_json(text), []

    payload = json.loads(text)
    if (
        not isinstance(payload, dict)
        or set(payload) != {"topics"}
        or not isinstance(payload.get("topics"), list)
    ):
        return schema.model_validate(payload), []

    topics: list[_BoundaryTopic] = []
    rejection_reasons: list[str] = []
    for index, raw_topic in enumerate(payload["topics"]):
        try:
            topics.append(_BoundaryTopic.model_validate(raw_topic))
        except ValidationError as exc:
            first_error = exc.errors(include_url=False)[0]
            location = ".".join(str(part) for part in first_error.get("loc", ()))
            rejection_reasons.append(
                f"proposal_{index}:schema_invalid:{location or 'item'}:"
                f"{first_error.get('type') or 'validation_error'}"
            )
    return _BoundaryPlan(topics=topics), rejection_reasons


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
    max_retries: int = 1,
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
            max_retries=max_retries,
            cancelled=cancelled,
        )
    except Exception as exc:
        if _cancel_requested(cancelled):
            raise
        provider_telemetry = _telemetry_dict(getattr(exc, "telemetry", None))
        raise _ModelCallError(
            f"{type(exc).__name__}: {exc}",
            {
                "model": model,
                "operation": operation,
                "prompt_version": prompt_version,
                "thinking_level": thinking_level,
                **provider_telemetry,
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
        parsed, schema_rejections = _validate_model_response(
            schema, result.text.strip(),
        )
    except (ValidationError, ValueError) as exc:
        raise _SchemaResponseError(
            f"invalid {schema.__name__} response: {exc}", telemetry,
        ) from exc
    if schema_rejections:
        telemetry["schema_rejected_count"] = len(schema_rejections)
        telemetry["schema_rejection_reasons"] = schema_rejections
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
    candidates = report.repair_candidates[:_MAX_CLIPS]
    configured_limit = _configured_clip_limit(settings)
    if (
        not candidates
        or (
            configured_limit is not None
            and report.accepted_count >= configured_limit
        )
    ):
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
    learner_level = str(
        settings.get("_knowledge_level")
        or settings.get("knowledge_level")
        or settings.get("learner_level")
        or ""
    )
    if profile == PRODUCTION_PRO_PROFILE:
        system, user = _legacy_prompts(rendered, len(segments), topic)
        schema: type[BaseModel] = _LegacyPlan
        model = config.SEGMENT_PRO_MODEL
        level, cap, timeout = "high", _SELECTION_OUTPUT_TOKENS, _PRO_TIMEOUT_S
        operation = "pro_authoritative"
    elif profile == CORRECTED_PRO_PROFILE:
        system, user = _prompts(
            rendered, len(segments), topic, learner_level=learner_level,
        )
        schema = _Plan
        model = config.SEGMENT_PRO_MODEL
        level, cap, timeout = "high", _SELECTION_OUTPUT_TOKENS, _PRO_TIMEOUT_S
        operation = "pro_fallback"
    elif profile == FLASH_SINGLE_PROFILE:
        system, user = _prompts(
            rendered, len(segments), topic, learner_level=learner_level,
        )
        schema = _Plan
        model = config.SEGMENT_FLASH_MODEL
        level, cap, timeout = "medium", _SELECTION_OUTPUT_TOKENS, _FLASH_SINGLE_TIMEOUT_S
        operation = "flash_single_candidate"
    elif profile == FLASH_SPLIT_PROFILE:
        system, user = _boundary_prompts(
            rendered,
            len(segments),
            topic,
            learner_level=learner_level,
        )
        schema = _BoundaryPlan
        model = config.SEGMENT_FLASH_MODEL
        level, cap, timeout = "low", _BOUNDARY_OUTPUT_TOKENS, _FLASH_BOUNDARY_TIMEOUT_S
        operation = "flash_boundary_selector"
    elif profile == PRO_BOUNDARY_PROFILE:
        system, user = _boundary_prompts(
            rendered,
            len(segments),
            topic,
            learner_level=learner_level,
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
        if requested_level in {"minimal", "low"}:
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
        # Keep the production selector to one logical, once-reserved call while
        # allowing one same-model retry for transient provider failures.
        max_retries=1,
    )
    require_enrichment = profile in {CORRECTED_PRO_PROFILE, FLASH_SINGLE_PROFILE}
    conversion_settings = dict(settings)
    conversion_settings.setdefault(
        "_segment_ignore_caption_case",
        str(transcript.get("source") or "").casefold() == "supadata",
    )
    if profile in {PRODUCTION_FLASH_PROFILE, PRO_BOUNDARY_PROFILE}:
        # Cache and persist the complete selector result. Public request ceilings
        # apply only when a feed inventory is surfaced, never during selection.
        conversion_settings.pop("max_clips", None)
    report = _plan_to_report(
        parsed,
        segments,
        words,
        conversion_settings,
        topic=topic,
        require_enrichment=require_enrichment,
    )
    schema_rejections = call.get("schema_rejection_reasons")
    if isinstance(schema_rejections, list):
        clean_rejections = [
            str(reason) for reason in schema_rejections if str(reason).strip()
        ]
        report.proposed_count += len(clean_rejections)
        report.rejected_reasons = clean_rejections + report.rejected_reasons
    calls = [call]
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
        error_type = str(call.get("error_type") or type(exc).__name__)
        failure_reason = f"request_failure:{error_type}"
        return SegmentResult(
            [],
            "Segmentation model call failed.",
            profile,
            "invalid",
            [failure_reason],
            calls=calls,
            error=f"{type(exc).__name__}: {exc}",
            rejection_reasons=[failure_reason],
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

    # Production defaults to the single Flash selector. Explicit alternate modes
    # remain available only to legacy evaluation callers.
    configured_mode = str(routing_mode or "flash_only").lower()
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
        routing_mode="flash_only",
    )
    if result.error:
        # A provider/schema/transport failure is not a valid empty selection.
        # Raising prevents the caller from persisting a poisoned empty cache
        # entry while successful zero-match responses remain cacheable.
        raise RuntimeError("segmentation provider call failed")
    return result.clips, result.notes
