"""
Pydantic schemas for the ingestion pipeline.

Style mirrors `app/models.py` — plain BaseModel, no ConfigDict, no aliases, snake_case.
The public response type for `POST /api/ingest/url` is `ReelOutWithAttribution`, which is
a minimal superset of the existing `ReelOut` adding an optional `source_attribution` field
plus a `terms_notice`. The iOS client's `Reel` decoder tolerates extra fields today, so this
is forward-compatible.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from ..models import ReelOut


PlatformLiteral = Literal["yt", "ig", "tt"]


class IngestRequest(BaseModel):
    source_url: str = Field(min_length=1, max_length=2000)
    material_id: str | None = Field(default=None, max_length=240)
    concept_id: str | None = Field(default=None, max_length=240)
    target_clip_duration_sec: int = Field(default=45, ge=15, le=180)
    target_clip_duration_min_sec: int = Field(default=15, ge=15, le=180)
    target_clip_duration_max_sec: int = Field(default=60, ge=15, le=180)
    language: str = Field(default="en", min_length=2, max_length=8)

    @model_validator(mode="after")
    def validate_duration_bounds(self) -> "IngestRequest":
        if self.target_clip_duration_max_sec < self.target_clip_duration_min_sec:
            raise ValueError("target_clip_duration_max_sec must be >= target_clip_duration_min_sec.")
        if not (
            self.target_clip_duration_min_sec
            <= self.target_clip_duration_sec
            <= self.target_clip_duration_max_sec
        ):
            raise ValueError("target_clip_duration_sec must fall within the min/max range.")
        return self


class IngestFeedRequest(BaseModel):
    feed_url: str = Field(min_length=1, max_length=2000)
    max_items: int = Field(default=6, ge=1, le=20)
    material_id: str | None = Field(default=None, max_length=240)
    concept_id: str | None = Field(default=None, max_length=240)
    target_clip_duration_sec: int = Field(default=45, ge=15, le=180)
    target_clip_duration_min_sec: int = Field(default=15, ge=15, le=180)
    target_clip_duration_max_sec: int = Field(default=60, ge=15, le=180)
    language: str = Field(default="en", min_length=2, max_length=8)


class IngestTranscriptCue(BaseModel):
    start: float = Field(ge=0.0)
    end: float = Field(ge=0.0)
    text: str

    @model_validator(mode="after")
    def validate_time_order(self) -> "IngestTranscriptCue":
        if self.end < self.start:
            # Whisper occasionally emits a 0-duration cue; pad it by 10ms so downstream math doesn't divide by zero.
            object.__setattr__(self, "end", self.start + 0.01)
        return self


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


class IngestResult(BaseModel):
    reel: ReelOutWithAttribution
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

    The adapter still supports Instagram + TikTok but the default is YouTube
    because IG/TT robots.txt explicitly disallows bots (User-Agent: ReelAIBot
    in the good-faith crawler config at ingestion/__init__.py) so those
    platforms would bounce at the robots.txt check anyway. Callers who want
    IG/TT can still opt in by passing `platforms` explicitly and accepting
    the legal/ToS posture.

    Each resolved URL is processed through the same ingest_url pipeline, so
    search results get Whisper fallback, silence-aware cuts, and full metadata
    just like a manually pasted URL.
    """

    query: str = Field(min_length=1, max_length=500)
    platforms: list[PlatformLiteral] = Field(default_factory=lambda: ["yt"])
    max_per_platform: int = Field(default=5, ge=1, le=15)
    material_id: str | None = Field(default=None, max_length=240)
    concept_id: str | None = Field(default=None, max_length=240)
    target_clip_duration_sec: int = Field(default=45, ge=15, le=180)
    target_clip_duration_min_sec: int = Field(default=15, ge=15, le=180)
    target_clip_duration_max_sec: int = Field(default=60, ge=15, le=180)
    language: str = Field(default="en", min_length=2, max_length=8)
    # Bare source_id strings (no prefix) of reels the client already has. The server
    # skips any resolved URL whose extracted source_id is in this set. This is the
    # pagination/infinite-scroll mechanism — pass every reel you've already seen.
    exclude_video_ids: list[str] = Field(default_factory=list)


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
    "IngestRequest",
    "IngestFeedRequest",
    "IngestSearchRequest",
    "IngestTopicCutRequest",
    "IngestTranscriptCue",
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
