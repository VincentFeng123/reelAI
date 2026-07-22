from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .clip_engine.metadata import extract_video_id as extract_youtube_video_id


class ConceptOut(BaseModel):
    id: str
    title: str
    keywords: list[str]
    summary: str


class MaterialResponse(BaseModel):
    material_id: str
    extracted_concepts: list[ConceptOut]


class BillingPlanOut(BaseModel):
    code: Literal["free", "plus", "pro"]
    name: str
    monthly_price_cents: int
    daily_limit: int


class BillingPlansResponse(BaseModel):
    plans: list[BillingPlanOut]


class BillingSubscriptionOut(BaseModel):
    provider: Literal["stripe"]
    plan: Literal["plus", "pro"]
    status: str
    current_period_end: str | None = None
    cancel_at_period_end: bool = False
    product_id: str


class BillingStatusResponse(BaseModel):
    plan: Literal["free", "plus", "pro"]
    daily_limit: int
    used_searches: int
    remaining_searches: int
    reset_at: str
    subscriptions: list[BillingSubscriptionOut] = Field(default_factory=list)


class BillingCheckoutRequest(BaseModel):
    plan: Literal["plus", "pro"]


class BillingRedirectResponse(BaseModel):
    url: str


class ReelsGenerateRequest(BaseModel):
    material_id: str = Field(min_length=1, max_length=128)
    concept_id: str | None = Field(default=None, max_length=128)
    num_reels: int = Field(default=20, ge=1, le=300)
    exclude_video_ids: list[str] = Field(default_factory=list, max_length=500)
    continuation_token: str | None = Field(default=None, max_length=128)
    creative_commons_only: bool = False
    generation_mode: Literal["slow", "fast"] = "slow"
    min_relevance: float | None = Field(default=None, ge=-1.0, le=1.2)
    preferred_video_duration: Literal["any", "short", "medium", "long"] = "any"
    # Deprecated compatibility fields. The relevance/silence pipeline ignores them.
    target_clip_duration_sec: int | None = None
    target_clip_duration_min_sec: int | None = None
    target_clip_duration_max_sec: int | None = None
    multi_platform_search: bool = False


class CaptionCue(BaseModel):
    start: float
    end: float
    text: str


class ReelOut(BaseModel):
    reel_id: str
    material_id: str
    concept_id: str
    concept_title: str
    video_id: str = ""
    video_title: str = ""
    video_description: str = ""
    channel_name: str = ""
    ai_summary: str = ""
    video_url: str
    t_start: float
    t_end: float
    transcript_snippet: str
    takeaways: list[str]
    captions: list[CaptionCue] = Field(default_factory=list)
    score: float
    # Deterministic transcript/query similarity used by retrieval shaping.
    relevance_score: float | None = None
    # Authoritative selector score used by the current topic-relevance hard gate.
    topic_relevance: float | None = None
    discovery_score: float | None = None
    clipability_score: float | None = None
    query_strategy: str = ""
    retrieval_stage: str = ""
    source_surface: str = ""
    matched_terms: list[str] = Field(default_factory=list)
    relevance_reason: str = ""
    match_reason: str = ""
    concept_position: int | None = None
    total_concepts: int | None = None
    video_duration_sec: int | None = None
    clip_duration_sec: float | None = None
    difficulty: float = 0.5
    informativeness: float = 0.6
    duration_preference_met: bool = True
    duration_fit: Literal["in_range", "shorter", "longer"] = "in_range"
    model_used: str | None = None
    quality_degraded: bool = False
    selected_cue_ids: list[str] = Field(default_factory=list)
    selection_contract_version: str | None = None

    @model_validator(mode="after")
    def populate_youtube_video_id(self) -> "ReelOut":
        if not self.video_id:
            source_id = extract_youtube_video_id(self.video_url)
            if source_id:
                object.__setattr__(self, "video_id", source_id)
        return self


class ReelsGenerateResponse(BaseModel):
    reels: list[ReelOut]
    generation_id: str | None = None
    response_profile: str | None = None
    batch_id: str | None = None
    batch_size: int = 0
    continuation_token: str | None = None
    terminal_status: Literal["completed", "partial", "exhausted"] | None = None
    reconciliation_tail_reel_ids: list[str] | None = None


class GenerationJobQueuedResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running"]
    status_url: str
    stream_url: str


class GenerationJobStatusResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "partial", "exhausted", "failed", "cancelled"]
    phase: str = ""
    progress: float = 0.0
    attempt_count: int = 0
    max_attempts: int = 3
    lease_expires_at: str | None = None
    heartbeat_at: str | None = None
    deadline_at: str | None = None
    material_id: str
    request_key: str
    result_generation_id: str | None = None
    model_used: str | None = None
    quality_degraded: bool = False
    usage: dict = Field(default_factory=dict)
    error: dict | None = None
    reels: list[ReelOut] = Field(default_factory=list)
    reconciliation_tail_reel_ids: list[str] | None = None
    created_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None


class ReelsCanGenerateResponse(BaseModel):
    availability: Literal["available", "unavailable", "unknown"]
    evidence_source: Literal["cache", "provider", "none"]
    evidence_age_sec: float | None = None
    candidate_count: int = 0
    filters_applied: dict = Field(default_factory=dict)
    message: str = ""


class ReelsCanGenerateAnyRequest(BaseModel):
    material_ids: list[str] = Field(default_factory=list, max_length=50)
    creative_commons_only: bool = False
    generation_mode: Literal["slow", "fast"] = "slow"
    min_relevance: float | None = Field(default=None, ge=-1.0, le=1.2)
    preferred_video_duration: Literal["any", "short", "medium", "long"] = "any"
    target_clip_duration_sec: int | None = None
    target_clip_duration_min_sec: int | None = None
    target_clip_duration_max_sec: int | None = None
    multi_platform_search: bool = False


class ReelsCanGenerateAnyResponse(BaseModel):
    availability: Literal["available", "unavailable", "unknown"]
    evidence_source: Literal["cache", "provider", "none"]
    evidence_age_sec: float | None = None
    candidate_count: int = 0
    filters_applied: dict = Field(default_factory=dict)
    materials_checked: int = 0
    message: str = ""


class MaterialLevelUpdateRequest(BaseModel):
    knowledge_level: Literal["beginner", "intermediate", "advanced"]


class FeedResponse(BaseModel):
    page: int
    limit: int
    total: int
    reels: list[ReelOut]
    generation_id: str | None = None
    response_profile: str | None = None
    generation_job_id: str | None = None
    generation_job_status: str | None = None
    continuation_token: str | None = None
    # The durable worker profile used for this request.
    effective_generation_mode: str | None = None
    # Retained for response compatibility; Railway workers do not override it.
    generation_mode_overridden: bool = False
    knowledge_level: str | None = None
    effective_level_target: float | None = None


class FeedbackRequest(BaseModel):
    reel_id: str = Field(min_length=1, max_length=128)
    helpful: bool = False
    confusing: bool = False
    rating: int | None = Field(default=None, ge=1, le=5)
    saved: bool = False

    @model_validator(mode="after")
    def validate_learning_signal(self) -> "FeedbackRequest":
        if self.helpful and self.confusing:
            raise ValueError("Feedback cannot be both helpful and confusing.")
        return self


class FeedbackResponse(BaseModel):
    status: str
    reel_id: str


class ReelProgressRequest(BaseModel):
    max_fraction: float = Field(ge=0.0, le=1.0)


class ReelProgressResponse(BaseModel):
    reel_id: str
    completed: bool
    newly_completed: bool
    assessment_ready: bool
    information_units: float
    readiness_threshold: float


class ReelScrollResponse(BaseModel):
    reel_id: str
    material_id: str
    newly_scrolled: bool
    assessment_ready: bool
    scroll_count: int
    cadence_target: int


class AssessmentQuestionOut(BaseModel):
    id: str
    reel_id: str
    concept_id: str
    concept_title: str
    prompt: str
    options: list[str]


class AssessmentSessionOut(BaseModel):
    id: str
    material_id: str
    status: Literal["pending", "completed", "snoozed"]
    current_index: int
    question_count: int
    answered_count: int
    questions: list[AssessmentQuestionOut]
    score: float | None = None
    understood_concepts: list[str] = Field(default_factory=list)
    revisit_concepts: list[str] = Field(default_factory=list)
    recent_accuracy: float | None = None
    rolling_accuracy: float | None = None


class AssessmentWrapperResponse(BaseModel):
    status: str
    assessment_ready: bool
    session: AssessmentSessionOut | None = None
    recent_accuracy: float | None = None
    rolling_accuracy: float | None = None


class AssessmentNextRequest(BaseModel):
    material_id: str = Field(min_length=1, max_length=240)


class AssessmentAnswerRequest(BaseModel):
    question_id: str = Field(min_length=1, max_length=160)
    choice_index: int = Field(ge=0, le=3)


class AssessmentAnswerResponse(BaseModel):
    correct: bool
    correct_index: int
    explanation: str
    session: AssessmentSessionOut


class AssessmentSnoozeResponse(BaseModel):
    status: str
    assessment_ready: bool


class ChatMessageIn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str
    topic: str | None = None
    text: str | None = None
    history: list[ChatMessageIn] = Field(default_factory=list)
    reel_summary: str | None = None
    video_title: str | None = None
    video_description: str | None = None
    transcript_snippet: str | None = None


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


class CommunitySetsDeleteRequest(BaseModel):
    set_ids: list[str] = Field(min_length=1, max_length=120)


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


class CommunityHistoryRecall(BaseModel):
    recent_score: int | None = Field(default=None, ge=0)
    recent_question_count: int | None = Field(default=None, ge=0)
    recent_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    rolling_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    understood_concepts: list[str] = Field(default_factory=list)
    revisit_concepts: list[str] = Field(default_factory=list)
    completed_at: str | None = None


class CommunityHistoryItemIn(BaseModel):
    material_id: str = Field(min_length=1, max_length=240)
    title: str = Field(min_length=1, max_length=200)
    updated_at: int = Field(ge=0)
    starred: bool = False
    generation_mode: Literal["slow", "fast"] = "slow"
    source: Literal["search", "community"] = "search"
    feed_query: str | None = Field(default=None, max_length=4000)
    active_index: int | None = Field(default=None, ge=0)
    active_reel_id: str | None = Field(default=None, max_length=400)
    recall: CommunityHistoryRecall | None = None
    recent_recall_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    rolling_recall_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)


class CommunityHistoryItemOut(BaseModel):
    material_id: str
    title: str
    updated_at: int
    starred: bool = False
    generation_mode: Literal["slow", "fast"] = "slow"
    source: Literal["search", "community"] = "search"
    feed_query: str | None = None
    active_index: int | None = None
    active_reel_id: str | None = None
    recall: CommunityHistoryRecall | None = None
    recent_recall_accuracy: float | None = None
    rolling_recall_accuracy: float | None = None


class CommunityHistoryReplaceRequest(BaseModel):
    items: list[CommunityHistoryItemIn] = Field(default_factory=list)


class CommunityHistoryResponse(BaseModel):
    items: list[CommunityHistoryItemOut] = Field(default_factory=list)


class CommunitySettingsPayload(BaseModel):
    generation_mode: Literal["slow", "fast"] = "slow"
    default_input_mode: Literal["topic", "source", "file"] = "source"
    min_relevance_threshold: float = Field(default=0.3, ge=0.0, le=0.6)
    start_muted: bool = True
    creative_commons_only: bool = False
    preferred_video_duration: Literal["any", "short", "medium", "long"] = "any"
    # Deprecated compatibility fields. Clip generation ignores them.
    target_clip_duration_sec: int | None = None
    target_clip_duration_min_sec: int | None = None
    target_clip_duration_max_sec: int | None = None
    autoplay_next_reel: bool = False


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
