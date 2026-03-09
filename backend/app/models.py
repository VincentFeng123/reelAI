from typing import Literal

from pydantic import BaseModel, Field


class ConceptOut(BaseModel):
    id: str
    title: str
    keywords: list[str]
    summary: str


class MaterialResponse(BaseModel):
    material_id: str
    extracted_concepts: list[ConceptOut]


class ReelsGenerateRequest(BaseModel):
    material_id: str
    concept_id: str | None = None
    num_reels: int = Field(default=8, ge=1, le=30)
    creative_commons_only: bool = False
    generation_mode: Literal["slow", "fast"] = "fast"
    min_relevance: float | None = Field(default=None, ge=-1.0, le=1.2)
    video_pool_mode: Literal["short-first", "balanced", "long-form"] = "short-first"
    preferred_video_duration: Literal["any", "short", "medium", "long"] = "any"
    target_clip_duration_sec: int = Field(default=55, ge=15, le=180)
    target_clip_duration_min_sec: int | None = Field(default=None, ge=15, le=180)
    target_clip_duration_max_sec: int | None = Field(default=None, ge=15, le=180)


class CaptionCue(BaseModel):
    start: float
    end: float
    text: str


class ReelOut(BaseModel):
    reel_id: str
    concept_id: str
    concept_title: str
    video_title: str = ""
    video_description: str = ""
    ai_summary: str = ""
    video_url: str
    t_start: float
    t_end: float
    transcript_snippet: str
    takeaways: list[str]
    captions: list[CaptionCue] = Field(default_factory=list)
    score: float
    relevance_score: float | None = None
    discovery_score: float | None = None
    clipability_score: float | None = None
    query_strategy: str = ""
    retrieval_stage: str = ""
    source_surface: str = ""
    matched_terms: list[str] = Field(default_factory=list)
    relevance_reason: str = ""
    concept_position: int | None = None
    total_concepts: int | None = None
    video_duration_sec: int | None = None
    clip_duration_sec: float | None = None


class ReelsGenerateResponse(BaseModel):
    reels: list[ReelOut]


class ReelsCanGenerateResponse(BaseModel):
    can_generate: bool
    blocked_by_settings: bool = False
    estimated_success_rate: float = 0.0
    total_probed: int = 0
    passed_all_filters: int = 0
    primary_bottleneck: str = ""
    message: str = ""


class ReelsCanGenerateAnyRequest(BaseModel):
    material_ids: list[str] = Field(default_factory=list)
    creative_commons_only: bool = False
    generation_mode: Literal["slow", "fast"] = "fast"
    min_relevance: float | None = Field(default=None, ge=-1.0, le=1.2)
    video_pool_mode: Literal["short-first", "balanced", "long-form"] = "short-first"
    preferred_video_duration: Literal["any", "short", "medium", "long"] = "any"
    target_clip_duration_sec: int = Field(default=55, ge=15, le=180)
    target_clip_duration_min_sec: int | None = Field(default=None, ge=15, le=180)
    target_clip_duration_max_sec: int | None = Field(default=None, ge=15, le=180)


class ReelsCanGenerateAnyResponse(BaseModel):
    can_generate_any: bool
    topics_checked: int = 0
    topics_can_generate: int = 0
    blocked_by_settings_topics: int = 0
    no_source_topics: int = 0
    estimated_success_rate: float = 0.0
    total_probed: int = 0
    passed_all_filters: int = 0
    primary_bottleneck: str = ""
    message: str = ""


class FeedResponse(BaseModel):
    page: int
    limit: int
    total: int
    reels: list[ReelOut]


class FeedbackRequest(BaseModel):
    reel_id: str
    helpful: bool = False
    confusing: bool = False
    rating: int | None = Field(default=None, ge=1, le=5)
    saved: bool = False


class FeedbackResponse(BaseModel):
    status: str
    reel_id: str


class ChatMessageIn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    topic: str | None = None
    text: str | None = None
    history: list[ChatMessageIn] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str


class CommunityReelIn(BaseModel):
    platform: Literal["youtube", "instagram", "tiktok"]
    source_url: str = Field(min_length=1)
    embed_url: str = Field(min_length=1)


class CommunityReelOut(BaseModel):
    id: str
    platform: Literal["youtube", "instagram", "tiktok"]
    source_url: str
    embed_url: str


class CommunitySetCreateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=140)
    description: str = Field(min_length=18, max_length=2000)
    tags: list[str] = Field(default_factory=list)
    reels: list[CommunityReelIn] = Field(default_factory=list)
    thumbnail_url: str = Field(min_length=1)
    curator: str | None = None


class CommunitySetOut(BaseModel):
    id: str
    title: str
    description: str
    tags: list[str] = Field(default_factory=list)
    reels: list[CommunityReelOut] = Field(default_factory=list)
    reel_count: int
    curator: str
    likes: int
    learners: int
    updated_label: str
    thumbnail_url: str
    featured: bool


class CommunitySetsResponse(BaseModel):
    sets: list[CommunitySetOut] = Field(default_factory=list)
