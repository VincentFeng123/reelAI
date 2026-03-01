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
    generation_mode: Literal["slow", "fast"] = "slow"


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
    concept_position: int | None = None
    total_concepts: int | None = None


class ReelsGenerateResponse(BaseModel):
    reels: list[ReelOut]


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
