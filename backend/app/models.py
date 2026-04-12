from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ConceptOut(BaseModel):
    id: str
    title: str
    keywords: list[str]
    summary: str


class MaterialResponse(BaseModel):
    material_id: str
    extracted_concepts: list[ConceptOut]


def _validate_clip_duration_bounds(
    target: int | None,
    min_sec: int | None,
    max_sec: int | None,
) -> tuple[int | None, int | None, int | None]:
    """
    Ensure min <= target <= max. Swap min/max if inverted and clamp the target
    into the range. main._resolve_target_clip_duration_bounds does the final
    hard clamp; this just rejects obviously nonsensical combinations early so
    we don't propagate `min=180, max=15` through the retrieval pipeline.
    """
    if min_sec is not None and max_sec is not None and min_sec > max_sec:
        min_sec, max_sec = max_sec, min_sec
    if target is not None:
        if min_sec is not None and target < min_sec:
            target = min_sec
        if max_sec is not None and target > max_sec:
            target = max_sec
    return target, min_sec, max_sec


class ReelsGenerateRequest(BaseModel):
    material_id: str = Field(min_length=1, max_length=128)
    concept_id: str | None = Field(default=None, max_length=128)
    num_reels: int = Field(default=8, ge=1, le=60)
    exclude_video_ids: list[str] = Field(default_factory=list, max_length=500)
    creative_commons_only: bool = False
    generation_mode: Literal["slow", "fast"] = "fast"
    min_relevance: float | None = Field(default=None, ge=-1.0, le=1.2)
    video_pool_mode: Literal["short-first", "balanced", "long-form"] = "short-first"
    preferred_video_duration: Literal["any", "short", "medium", "long"] = "any"
    target_clip_duration_sec: int = Field(default=55, ge=15, le=180)
    target_clip_duration_min_sec: int | None = Field(default=None, ge=15, le=180)
    target_clip_duration_max_sec: int | None = Field(default=None, ge=15, le=180)

    @model_validator(mode="after")
    def _validate_clip_bounds(self) -> "ReelsGenerateRequest":
        target, min_sec, max_sec = _validate_clip_duration_bounds(
            self.target_clip_duration_sec,
            self.target_clip_duration_min_sec,
            self.target_clip_duration_max_sec,
        )
        self.target_clip_duration_sec = target or 55
        self.target_clip_duration_min_sec = min_sec
        self.target_clip_duration_max_sec = max_sec
        return self


class CaptionCue(BaseModel):
    start: float
    end: float
    text: str


class ReelOut(BaseModel):
    reel_id: str
    material_id: str
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
    generation_id: str | None = None
    response_profile: str | None = None
    refinement_job_id: str | None = None
    refinement_status: str | None = None


class ReelsCanGenerateResponse(BaseModel):
    can_generate: bool
    blocked_by_settings: bool = False
    estimated_success_rate: float = 0.0
    total_probed: int = 0
    passed_all_filters: int = 0
    primary_bottleneck: str = ""
    message: str = ""


class ReelsCanGenerateAnyRequest(BaseModel):
    material_ids: list[str] = Field(default_factory=list, max_length=50)
    creative_commons_only: bool = False
    generation_mode: Literal["slow", "fast"] = "fast"
    min_relevance: float | None = Field(default=None, ge=-1.0, le=1.2)
    video_pool_mode: Literal["short-first", "balanced", "long-form"] = "short-first"
    preferred_video_duration: Literal["any", "short", "medium", "long"] = "any"
    target_clip_duration_sec: int = Field(default=55, ge=15, le=180)
    target_clip_duration_min_sec: int | None = Field(default=None, ge=15, le=180)
    target_clip_duration_max_sec: int | None = Field(default=None, ge=15, le=180)

    @model_validator(mode="after")
    def _validate_clip_bounds(self) -> "ReelsCanGenerateAnyRequest":
        target, min_sec, max_sec = _validate_clip_duration_bounds(
            self.target_clip_duration_sec,
            self.target_clip_duration_min_sec,
            self.target_clip_duration_max_sec,
        )
        self.target_clip_duration_sec = target or 55
        self.target_clip_duration_min_sec = min_sec
        self.target_clip_duration_max_sec = max_sec
        return self


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
    generation_id: str | None = None
    response_profile: str | None = None
    refinement_job_id: str | None = None
    refinement_status: str | None = None


class RefinementStatusResponse(BaseModel):
    job_id: str
    status: str
    material_id: str
    request_key: str
    source_generation_id: str
    result_generation_id: str | None = None
    active_generation_id: str | None = None
    completed_at: str | None = None
    error: str | None = None


class FeedbackRequest(BaseModel):
    reel_id: str = Field(min_length=1, max_length=128)
    helpful: bool = False
    confusing: bool = False
    rating: int | None = Field(default=None, ge=1, le=5)
    saved: bool = False


class FeedbackResponse(BaseModel):
    status: str
    reel_id: str


class ChatMessageIn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str
    topic: str | None = None
    text: str | None = None
    history: list[ChatMessageIn] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str


class CommunityReelIn(BaseModel):
    id: str | None = Field(default=None, min_length=1, max_length=120)
    platform: Literal["youtube", "instagram", "tiktok"]
    source_url: str = Field(min_length=1)
    embed_url: str = Field(min_length=1)
    t_start_sec: float | None = Field(default=None, ge=0)
    t_end_sec: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_clip_range(self) -> "CommunityReelIn":
        if self.t_end_sec is not None:
            start = self.t_start_sec if self.t_start_sec is not None else 0.0
            if self.t_end_sec <= start:
                raise ValueError("t_end_sec must be greater than t_start_sec.")
        return self


class CommunityReelOut(BaseModel):
    id: str
    platform: Literal["youtube", "instagram", "tiktok"]
    source_url: str
    embed_url: str
    t_start_sec: float | None = None
    t_end_sec: float | None = None


class CommunitySetCreateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=140)
    description: str = Field(min_length=18, max_length=2000)
    tags: list[str] = Field(default_factory=list)
    reels: list[CommunityReelIn] = Field(default_factory=list)
    thumbnail_url: str = Field(min_length=1)
    curator: str | None = None


class CommunitySetUpdateRequest(BaseModel):
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
    # Aggregate dislike count. Defaulted so old clients that decode this
    # model without the new field still work and new rows start at 0.
    dislikes: int = 0
    # Whether the authenticated viewer has liked / disliked this set.
    # Both default to False for anonymous listings; mutual exclusion is
    # enforced on write — at most one can be true for a given (user, set).
    viewer_liked: bool = False
    viewer_disliked: bool = False
    learners: int
    updated_label: str
    updated_at: str | None = None
    created_at: str | None = None
    thumbnail_url: str
    featured: bool


class CommunitySetsResponse(BaseModel):
    sets: list[CommunitySetOut] = Field(default_factory=list)


class CommunitySetFeedbackRequest(BaseModel):
    # The user's intended vote state *after* this call. To un-vote a
    # set, send `liked=False, disliked=False`. The server enforces
    # mutual exclusion — sending both as True is a 400.
    liked: bool = False
    disliked: bool = False


class CommunitySetFeedbackResponse(BaseModel):
    status: str
    set_id: str
    likes: int
    dislikes: int
    viewer_liked: bool
    viewer_disliked: bool


class CommunitySendSignupVerificationRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    username: str | None = Field(default=None, min_length=1, max_length=32)


class CommunitySendSignupVerificationResponse(BaseModel):
    email: str
    verification_required: bool = True
    verified: bool = False
    verification_code_debug: str | None = None


class CommunityVerifySignupEmailRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    code: str = Field(min_length=4, max_length=16)


class CommunityVerifySignupEmailResponse(BaseModel):
    email: str
    verified: bool = True


class CommunityAuthRegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=8, max_length=128)


class CommunityAuthLoginRequest(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=8, max_length=128)


class CommunityAccountOut(BaseModel):
    id: str
    username: str
    email: str | None = None
    is_verified: bool = True


class CommunityAuthSessionResponse(BaseModel):
    account: CommunityAccountOut
    session_token: str
    claimed_legacy_sets: int = 0
    verification_required: bool = False
    verification_code_debug: str | None = None


class CommunityChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=8, max_length=128)
    new_password: str = Field(min_length=8, max_length=128)


class CommunityDeleteAccountRequest(BaseModel):
    current_password: str = Field(min_length=8, max_length=128)


class CommunityChangeEmailRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    current_password: str = Field(min_length=8, max_length=128)


class CommunityVerifyAccountRequest(BaseModel):
    code: str = Field(min_length=4, max_length=16)


class CommunityAuthMeResponse(BaseModel):
    account: CommunityAccountOut


class CommunityVerifyAccountResponse(BaseModel):
    account: CommunityAccountOut
    claimed_legacy_sets: int = 0


class CommunityResendVerificationResponse(BaseModel):
    account: CommunityAccountOut
    verification_code_debug: str | None = None


class CommunityChangeEmailResponse(BaseModel):
    account: CommunityAccountOut
    verification_code_debug: str | None = None


class CommunityHistoryItemIn(BaseModel):
    material_id: str = Field(min_length=1, max_length=240)
    title: str = Field(min_length=1, max_length=200)
    updated_at: int = Field(ge=0)
    starred: bool = False
    generation_mode: Literal["slow", "fast"] = "fast"
    source: Literal["search", "community"] = "search"
    feed_query: str | None = Field(default=None, max_length=4000)
    active_index: int | None = Field(default=None, ge=0)
    active_reel_id: str | None = Field(default=None, max_length=400)


class CommunityHistoryItemOut(BaseModel):
    material_id: str
    title: str
    updated_at: int
    starred: bool = False
    generation_mode: Literal["slow", "fast"] = "fast"
    source: Literal["search", "community"] = "search"
    feed_query: str | None = None
    active_index: int | None = None
    active_reel_id: str | None = None


class CommunityHistoryReplaceRequest(BaseModel):
    items: list[CommunityHistoryItemIn] = Field(default_factory=list)


class CommunityHistoryResponse(BaseModel):
    items: list[CommunityHistoryItemOut] = Field(default_factory=list)


class CommunitySettingsPayload(BaseModel):
    generation_mode: Literal["slow", "fast"] = "fast"
    default_input_mode: Literal["topic", "source", "file"] = "source"
    min_relevance_threshold: float = Field(default=0.3, ge=0.0, le=0.6)
    start_muted: bool = True
    video_pool_mode: Literal["short-first", "balanced", "long-form"] = "short-first"
    preferred_video_duration: Literal["any", "short", "medium", "long"] = "any"
    target_clip_duration_sec: int = Field(default=55, ge=15, le=180)
    target_clip_duration_min_sec: int = Field(default=20, ge=15, le=180)
    target_clip_duration_max_sec: int = Field(default=55, ge=15, le=180)
    autoplay_next_reel: bool = False

    @model_validator(mode="after")
    def validate_clip_duration_bounds(self) -> "CommunitySettingsPayload":
        if self.target_clip_duration_max_sec <= self.target_clip_duration_min_sec:
            raise ValueError("target_clip_duration_max_sec must be greater than target_clip_duration_min_sec.")
        if self.target_clip_duration_max_sec - self.target_clip_duration_min_sec < 15:
            raise ValueError("target clip duration range must be at least 15 seconds wide.")
        if not self.target_clip_duration_min_sec <= self.target_clip_duration_sec <= self.target_clip_duration_max_sec:
            raise ValueError("target_clip_duration_sec must fall within the configured min/max range.")
        return self


class CommunitySettingsResponse(CommunitySettingsPayload):
    pass


class CommunityStarredSetsPayload(BaseModel):
    set_ids: list[str] = []


class CommunityStarredSetsResponse(BaseModel):
    set_ids: list[str] = []


class CommunityFeedSnapshotPayload(BaseModel):
    snapshot: dict = {}


class CommunityFeedSnapshotsResponse(BaseModel):
    snapshots: dict[str, dict] = {}


class CommunityDraftPayload(BaseModel):
    draft: dict = {}


class CommunityDraftsResponse(BaseModel):
    drafts: dict[str, dict] = {}


class CommunityMaterialSeedsPayload(BaseModel):
    seeds: dict[str, dict] = {}


class CommunityMaterialSeedsResponse(BaseModel):
    seeds: dict[str, dict] = {}


class CommunityMaterialGroupsPayload(BaseModel):
    groups: dict[str, dict] = {}


class CommunityMaterialGroupsResponse(BaseModel):
    groups: dict[str, dict] = {}
