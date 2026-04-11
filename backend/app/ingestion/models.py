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
    Topic-based search across YouTube + Instagram + TikTok.

    Each platform is resolved independently (via yt-dlp's native search extractors)
    then every resolved URL is processed through the same ingest_url pipeline, so
    search results get Whisper fallback, silence-aware cuts, and full metadata just
    like a manually pasted URL.
    """

    query: str = Field(min_length=1, max_length=500)
    platforms: list[PlatformLiteral] = Field(default_factory=lambda: ["yt", "ig", "tt"])
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


__all__ = [
    "PlatformLiteral",
    "IngestRequest",
    "IngestFeedRequest",
    "IngestSearchRequest",
    "IngestTranscriptCue",
    "IngestSegment",
    "IngestMetadata",
    "ReelOutWithAttribution",
    "IngestResult",
    "IngestFeedItem",
    "IngestFeedResult",
    "IngestSearchItem",
    "IngestSearchResult",
]
