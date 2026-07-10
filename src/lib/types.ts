export type Concept = {
  id: string;
  title: string;
  keywords: string[];
  summary: string;
};

export type MaterialResponse = {
  material_id: string;
  extracted_concepts: Concept[];
};

export type Reel = {
  reel_id: string;
  material_id: string;
  concept_id: string;
  concept_title: string;
  video_title?: string;
  video_description?: string;
  channel_name?: string;
  published_at?: string | null;
  view_count?: number | null;
  video_description_truncated?: boolean;
  ai_summary?: string;
  ai_summary_truncated?: boolean;
  video_url: string;
  t_start: number;
  t_end: number;
  transcript_snippet: string;
  transcript_snippet_truncated?: boolean;
  takeaways: string[];
  captions?: CaptionCue[];
  score: number;
  relevance_score?: number;
  matched_terms?: string[];
  relevance_reason?: string;
  match_reason?: string;
  informativeness?: number;
  concept_position?: number;
  total_concepts?: number;
  video_duration_sec?: number;
  clip_duration_sec?: number;
  difficulty?: number;
  model_used?: string | null;
  quality_degraded?: boolean;
  duration_preference_met?: boolean;
  duration_fit?: "in_range" | "shorter" | "longer";
  selected_cue_ids?: string[];
  community_has_explicit_end?: boolean;
  // Optional attribution line populated by POST /api/ingest/url. Older responses
  // (from /api/feed, /api/reels/generate) do not include this field.
  source_attribution?: string | null;
};

export type CaptionCue = {
  cue_id?: string | number;
  start: number;
  end: number;
  text: string;
};

export type ReelsGenerateResponse = {
  reels: Reel[];
  generation_id?: string | null;
  response_profile?: string | null;
  model_used?: string | null;
  quality_degraded?: boolean;
};

export type GenerationJobStatus =
  | "queued"
  | "running"
  | "completed"
  | "partial"
  | "exhausted"
  | "failed"
  | "cancelled";

export type GenerationTerminalStatus = Exclude<GenerationJobStatus, "queued" | "running">;

export type TypedApiError = {
  code: string;
  message: string;
  provider?: string | null;
  retry_after_sec?: number | null;
  details?: Record<string, unknown> | null;
};

export type GenerationQueuedResponse = {
  job_id: string;
  status: "queued" | "running";
  status_url: string;
  stream_url: string;
};

export type ReelsGenerateSubmission = ReelsGenerateResponse | GenerationQueuedResponse;

type GenerationStreamEventBase = {
  job_id: string;
  seq: number;
  timestamp: string;
};

export type ReelsGenerateStreamEvent = GenerationStreamEventBase & (
  | {
      type: "candidate";
      payload: { reel: Reel; provisional: true };
    }
  | {
      type: "final";
      payload: {
        reels: Reel[];
        generation_id?: string | null;
        authoritative: true;
      };
    }
  | {
      type: "terminal";
      payload: {
        status: GenerationTerminalStatus;
        result_generation_id?: string | null;
        error?: TypedApiError | null;
      };
    }
);

export type GenerationJobStatusResponse = {
  job_id: string;
  status: GenerationJobStatus;
  phase?: string | null;
  progress?: number | null;
  attempt_count?: number | null;
  max_attempts?: number | null;
  lease_expires_at?: string | null;
  heartbeat_at?: string | null;
  deadline_at?: string | null;
  material_id: string;
  request_key: string;
  result_generation_id?: string | null;
  model_used?: string | null;
  quality_degraded?: boolean;
  usage?: Record<string, unknown> | null;
  error?: TypedApiError | null;
  reels?: Reel[];
};

export type GenerationJobCancelResponse = GenerationJobStatusResponse;

export type AvailabilityEvidence = {
  availability: "available" | "unavailable" | "unknown";
  evidence_source: "cache" | "provider" | "none" | string;
  evidence_age_sec: number | null;
  candidate_count: number;
  filters_applied: Record<string, unknown> | string[];
  message?: string;
};

export type ReelsCanGenerateResponse = AvailabilityEvidence & {
  can_generate?: boolean;
};

export type ReelsCanGenerateAnyResponse = AvailabilityEvidence & {
  can_generate_any?: boolean;
  materials_checked: number;
};

export type FeedResponse = {
  page: number;
  limit: number;
  total: number;
  reels: Reel[];
  generation_id?: string | null;
  response_profile?: string | null;
  generation_job_id?: string | null;
  generation_job_status?: GenerationJobStatus | null;
  knowledge_level?: string | null;
  effective_level_target?: number | null;
};

export type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

export type ChatResponse = {
  answer: string;
};

export type ReelProgressResponse = {
  reel_id: string;
  completed: boolean;
  newly_completed: boolean;
  assessment_ready: boolean;
  information_units: number;
  readiness_threshold: number;
};

export type AssessmentQuestion = {
  id: string;
  reel_id: string;
  concept_id: string;
  concept_title: string;
  prompt: string;
  options: string[];
};

export type AssessmentSession = {
  id: string;
  material_id: string;
  status: string;
  current_index: number;
  question_count: number;
  answered_count: number;
  questions: AssessmentQuestion[];
  score?: number | null;
  understood_concepts: string[];
  revisit_concepts: string[];
  recent_accuracy?: number | null;
  rolling_accuracy?: number | null;
};

export type AssessmentStatusResponse = {
  status: string;
  assessment_ready: boolean;
  session?: AssessmentSession | null;
  recent_accuracy?: number | null;
  rolling_accuracy?: number | null;
};

export type AssessmentAnswerResponse = {
  correct: boolean;
  correct_index: number;
  explanation: string;
  session: AssessmentSession;
};

export type AssessmentSnoozeResponse = {
  status: string;
  assessment_ready: false;
};

// POST /api/ingest/url — native-caption YouTube ingestion.

export type IngestUrlRequest = {
  source_url: string;
  material_id?: string | null;
  concept_id?: string | null;
  target_clip_duration_sec?: number;
  target_clip_duration_min_sec?: number;
  target_clip_duration_max_sec?: number;
  language?: string;
};

/** Mirror of `backend/app/ingestion/models.py:IngestMetadata`. All fields optional. */
export type IngestMetadata = {
  platform: "yt";
  source_id: string;
  source_url: string;
  playback_url: string;
  title?: string;
  description?: string;
  author_handle?: string;
  author_name?: string;
  author_url?: string;
  duration_sec?: number | null;
  thumbnail_url?: string;
  upload_date_iso?: string | null;
  view_count?: number | null;
  like_count?: number | null;
  comment_count?: number | null;
  repost_count?: number | null;
  hashtags?: string[];
  categories?: string[];
  audio_title?: string;
  audio_artist?: string;
  language?: string;
  location?: string;
  is_private?: boolean;
  is_live?: boolean;
};

export type IngestResult = {
  reel: Reel;
  metadata: IngestMetadata;
  terms_notice: string;
  trace_id: string;
};

// POST /api/ingest/search — topic-based YouTube discovery.

export type IngestSearchRequest = {
  query: string;
  max_per_platform?: number;
  material_id?: string | null;
  concept_id?: string | null;
  target_clip_duration_sec?: number;
  target_clip_duration_min_sec?: number;
  target_clip_duration_max_sec?: number;
  language?: string;
  /** Bare source_id strings of reels the client already has — pagination for infinite scroll. */
  exclude_video_ids?: string[];
};

export type IngestSearchItem = {
  platform: "yt";
  source_url: string;
  status: "ok" | "error" | "skipped" | "rate_limited";
  reel?: Reel | null;
  metadata?: IngestMetadata | null;
  error?: string | null;
};

export type IngestSearchResult = {
  query: string;
  material_id: string;
  platforms: ["yt"] | "yt"[];
  per_platform_resolved: Record<string, number>;
  per_platform_succeeded: Record<string, number>;
  per_platform_failed: Record<string, number>;
  per_platform_errors: Record<string, string>;
  total_resolved: number;
  succeeded: number;
  failed: number;
  items: IngestSearchItem[];
  terms_notice: string;
  trace_id: string;
};

export type CommunityReelPlatform = "youtube" | "instagram" | "tiktok";

export type CommunityReelEmbed = {
  id: string;
  platform: CommunityReelPlatform;
  sourceUrl: string;
  embedUrl: string;
  tStartSec?: number;
  tEndSec?: number;
};

export type CommunitySet = {
  id: string;
  title: string;
  description: string;
  tags: string[];
  reels: CommunityReelEmbed[];
  reelCount: number;
  curator: string;
  likes: number;
  learners: number;
  updatedLabel: string;
  updatedAt?: string | null;
  createdAt?: string | null;
  thumbnailUrl: string;
  featured: boolean;
};

export type CommunityAccount = {
  id: string;
  username: string;
  email?: string | null;
  isVerified: boolean;
};

export type CommunityAuthSession = {
  account: CommunityAccount;
  sessionToken: string;
  claimedLegacySets: number;
  verificationRequired: boolean;
  verificationCodeDebug?: string | null;
};
