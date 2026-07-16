"""
Pydantic schemas for the ingestion pipeline.

Style mirrors `app/models.py` — plain BaseModel, no ConfigDict, no aliases, snake_case.
The public response type for `POST /api/ingest/url` is `ReelOutWithAttribution`, which is
a minimal superset of the existing `ReelOut` adding an optional `source_attribution` field
plus a `terms_notice`. The iOS client's `Reel` decoder tolerates extra fields today, so this
is forward-compatible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from ..models import ReelOut


PlatformLiteral = Literal["yt"]


@dataclass(frozen=True)
class YouTubeSourceRef:
    source_id: str
    source_url: str
    playback_url: str
    platform: PlatformLiteral = "yt"


class IngestRequest(BaseModel):
    source_url: str = Field(min_length=1, max_length=2000)
    material_id: str | None = Field(default=None, max_length=240)
    concept_id: str | None = Field(default=None, max_length=240)
    target_clip_duration_sec: int | None = None
    target_clip_duration_min_sec: int | None = None
    target_clip_duration_max_sec: int | None = None
    language: str = Field(default="en", min_length=2, max_length=8)
    multi_platform_search: bool = False

class IngestFeedRequest(BaseModel):
    feed_url: str = Field(min_length=1, max_length=2000)
    max_items: int = Field(default=6, ge=1, le=20)
    material_id: str | None = Field(default=None, max_length=240)
    concept_id: str | None = Field(default=None, max_length=240)
    target_clip_duration_sec: int | None = None
    target_clip_duration_min_sec: int | None = None
    target_clip_duration_max_sec: int | None = None
    language: str = Field(default="en", min_length=2, max_length=8)
    multi_platform_search: bool = False


WordSourceLiteral = Literal[
    "native_caption",
    "legacy",
]


class IngestTranscriptWord(BaseModel):
    """
    Legacy word-level timestamp retained only for stored-payload compatibility.

    The active clipping path cuts directly on timestamped transcript cue
    boundaries and does not synthesize word timestamps.
    """

    start: float = Field(ge=0.0)
    end: float = Field(ge=0.0)
    text: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_time_order(self) -> "IngestTranscriptWord":
        if self.end < self.start:
            object.__setattr__(self, "end", self.start + 0.005)
        return self


class IngestTranscriptCue(BaseModel):
    cue_id: str = ""
    start: float = Field(ge=0.0)
    end: float = Field(ge=0.0)
    text: str
    # Additive fields for Phase A.1. Existing persisted rows deserialize with empty
    # `words` and `word_source="legacy"`, letting the ClipBoundaryEngine fall back
    # to cue-level sentence picks without error.
    words: list[IngestTranscriptWord] = Field(default_factory=list)
    word_source: WordSourceLiteral = "legacy"

    @model_validator(mode="after")
    def validate_time_order(self) -> "IngestTranscriptCue":
        if self.end < self.start:
            # Pad malformed legacy cues so downstream compatibility reads remain safe.
            object.__setattr__(self, "end", self.start + 0.01)
        return self

    def word_window(self, t0: float, t1: float) -> list[IngestTranscriptWord]:
        """Return words overlapping the [t0, t1] interval — linear scan, cues are short."""
        if not self.words:
            return []
        out: list[IngestTranscriptWord] = []
        for w in self.words:
            if w.end < t0:
                continue
            if w.start > t1:
                break
            out.append(w)
        return out


class IngestSegment(BaseModel):
    t_start: float = Field(ge=0.0)
    t_end: float = Field(ge=0.0)
    text: str
    score: float = 1.0


class IngestMetadata(BaseModel):
    platform: PlatformLiteral
    source_id: str
    source_url: str
    playback_url: str
    title: str = ""
    description: str = ""
    author_handle: str = ""
    author_name: str = ""
    author_url: str = ""
    channel_id: str = ""
    duration_sec: float | None = None
    thumbnail_url: str = ""
    upload_date_iso: str | None = None
    view_count: int | None = None
    like_count: int | None = None
    comment_count: int | None = None
    repost_count: int | None = None
    hashtags: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    audio_title: str = ""
    audio_artist: str = ""
    language: str = ""
    location: str = ""
    is_private: bool = False
    is_live: bool = False


class ReelOutWithAttribution(ReelOut):
    """
    Superset of the existing ReelOut with an optional attribution line the iOS UI can render.
    `app/models.py` stays untouched per the plan — this subclass lives only in the ingestion
    package and only the new `/api/ingest/*` endpoints return it.
    """

    source_attribution: str | None = None
    selection_contract_version: str | None = None
    boundary_status: str = ""
    acoustic_verified: bool = False
    boundary_confidence: float | None = None
    is_standalone: bool = True
    chain_id: str = ""
    chain_position: float = 0.0
    selection_candidate_id: str = ""
    prerequisite_ids: list[str] = Field(default_factory=list)
    selection_quality_floor: float | None = Field(default=None, exclude=True)
    selection_quality_mean: float | None = Field(default=None, exclude=True)
    selection_topic_relevance: float | None = Field(default=None, exclude=True)
    selection_source_rank: int = Field(default=0, exclude=True)
    selection_intent_role: str = Field(default="primary", exclude=True)
    selection_intent_coverage: float = Field(default=1.0, exclude=True)


class IngestResult(BaseModel):
    reel: ReelOutWithAttribution
    reels: list[ReelOutWithAttribution] = Field(default_factory=list)
    metadata: IngestMetadata
    terms_notice: str
    trace_id: str


class IngestFeedItem(BaseModel):
    source_url: str
    status: Literal["ok", "error", "skipped"]
    reel: ReelOutWithAttribution | None = None
    metadata: IngestMetadata | None = None
    error: str | None = None


class IngestFeedResult(BaseModel):
    feed_url: str
    total_resolved: int
    succeeded: int
    failed: int
    items: list[IngestFeedItem]
    terms_notice: str
    trace_id: str


# --------------------------------------------------------------------- #
# Topic-based multi-platform search (POST /api/ingest/search)
# --------------------------------------------------------------------- #


class IngestSearchRequest(BaseModel):
    """
    Topic-based search, currently YouTube-only.

    Discovery and ingestion are YouTube-only; source-native captions are optional.
    """

    query: str = Field(min_length=1, max_length=500)
    platforms: list[PlatformLiteral] = Field(default_factory=lambda: ["yt"])
    max_per_platform: int = Field(default=5, ge=1, le=15)
    material_id: str | None = Field(default=None, max_length=240)
    concept_id: str | None = Field(default=None, max_length=240)
    target_clip_duration_sec: int | None = None
    target_clip_duration_min_sec: int | None = None
    target_clip_duration_max_sec: int | None = None
    language: str = Field(default="en", min_length=2, max_length=8)
    # Bare source_id strings (no prefix) of reels the client already has. The server
    # skips any resolved URL whose extracted source_id is in this set. This is the
    # pagination/infinite-scroll mechanism — pass every reel you've already seen.
    exclude_video_ids: list[str] = Field(default_factory=list)
    multi_platform_search: bool = False

    @model_validator(mode="after")
    def validate_youtube_search(self) -> "IngestSearchRequest":
        self.query = " ".join(self.query.split())
        if not self.query:
            raise ValueError("query must contain non-whitespace text.")
        if set(self.platforms) != {"yt"}:
            raise ValueError("Only YouTube search is supported; platforms must be ['yt'].")
        return self


class IngestSearchItem(BaseModel):
    platform: PlatformLiteral
    source_url: str
    status: Literal["ok", "error", "skipped", "rate_limited"]
    reel: ReelOutWithAttribution | None = None
    metadata: IngestMetadata | None = None
    error: str | None = None


class IngestSearchResult(BaseModel):
    query: str
    material_id: str
    platforms: list[PlatformLiteral]
    per_platform_resolved: dict[str, int] = Field(default_factory=dict)
    per_platform_succeeded: dict[str, int] = Field(default_factory=dict)
    per_platform_failed: dict[str, int] = Field(default_factory=dict)
    per_platform_errors: dict[str, str] = Field(default_factory=dict)
    total_resolved: int
    succeeded: int
    failed: int
    items: list[IngestSearchItem]
    terms_notice: str
    trace_id: str


# --------------------------------------------------------------------- #
# Topic-aware multi-reel cut (POST /api/ingest/topic-cut)
# --------------------------------------------------------------------- #


class IngestTopicCutRequest(BaseModel):
    """
    Request payload for `POST /api/ingest/topic-cut`.

    Unlike the legacy `/api/ingest/url`, this endpoint cuts a long-form video
    into MULTIPLE per-topic reels (one per topic the creator introduces and
    transitions away from). The clip-duration knobs are intentionally absent —
    the topic boundaries determine the clip lengths, not a fixed budget.

    `material_id` and `concept_id` are optional. When omitted the pipeline
    routes the resulting reels under the existing ingestion sentinel material
    so they're browsable via `/api/feed?material_id=ingest-scratch`.
    """

    source_url: str = Field(min_length=1, max_length=2000)
    material_id: str | None = Field(default=None, max_length=240)
    concept_id: str | None = Field(default=None, max_length=240)
    language: str = Field(default="en", min_length=2, max_length=8)
    multi_platform_search: bool = False
    use_llm: bool = Field(
        default=True,
        description=(
            "If False, skip the LLM topic-segmentation pass and use the "
            "lexical-novelty heuristic only. Useful for offline runs / tests."
        ),
    )
    query: str | None = Field(
        default=None,
        max_length=500,
        description=(
            "When provided, only topic reels relevant to this search query "
            "are returned. Uses token-overlap scoring on labels, summaries, "
            "and transcript text."
        ),
    )

    @model_validator(mode="after")
    def validate_supported_mode(self) -> "IngestTopicCutRequest":
        if not self.use_llm:
            raise ValueError(
                "use_llm=false is unsupported; timestamped-cue Gemini segmentation is required."
            )
        if self.query is not None:
            self.query = " ".join(self.query.split())
            if not self.query:
                raise ValueError("query must contain non-whitespace text when provided.")
        return self


class IngestTopicCutResult(BaseModel):
    """
    Response payload for `POST /api/ingest/topic-cut`.

    For YouTube Shorts, `reels` is empty and `is_short` is True — the caller
    should leave the original video untouched per the topic-cut contract.

    For long-form videos, `reels` carries one entry per topic the cutter
    identified. Each entry decodes cleanly into the iOS `Reel` struct via the
    existing decoder, so no client-side schema changes are required.
    """

    source_url: str
    video_id: str
    is_short: bool
    classification_reason: str
    duration_sec: float
    reel_count: int
    reels: list[ReelOutWithAttribution]
    metadata: IngestMetadata | None = None
    terms_notice: str
    trace_id: str


__all__ = [
    "PlatformLiteral",
    "YouTubeSourceRef",
    "IngestRequest",
    "IngestFeedRequest",
    "IngestSearchRequest",
    "IngestTopicCutRequest",
    "IngestTranscriptCue",
    "IngestTranscriptWord",
    "WordSourceLiteral",
    "IngestSegment",
    "IngestMetadata",
    "ReelOutWithAttribution",
    "IngestResult",
    "IngestFeedItem",
    "IngestFeedResult",
    "IngestSearchItem",
    "IngestSearchResult",
    "IngestTopicCutResult",
]
