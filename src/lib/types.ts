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
  concept_position?: number;
  total_concepts?: number;
  video_duration_sec?: number;
  clip_duration_sec?: number;
  community_has_explicit_end?: boolean;
  // Optional attribution line populated by POST /api/ingest/url. Older responses
  // (from /api/feed, /api/reels/generate) do not include this field.
  source_attribution?: string | null;
};

export type CaptionCue = {
  start: number;
  end: number;
  text: string;
};

export type ReelsGenerateResponse = {
  reels: Reel[];
  generation_id?: string | null;
  response_profile?: string | null;
  refinement_job_id?: string | null;
  refinement_status?: string | null;
};

export type ReelsGenerateStreamEvent =
  | {
      type: "reel";
      reel: Reel;
    }
  | {
      type: "done";
      response: ReelsGenerateResponse;
    }
  | {
      type: "error";
      detail: string;
      status_code?: number;
    };

export type ReelsCanGenerateResponse = {
  can_generate: boolean;
  blocked_by_settings: boolean;
  estimated_success_rate: number;
  total_probed: number;
  passed_all_filters: number;
  primary_bottleneck: string;
  message: string;
};

export type ReelsCanGenerateAnyResponse = {
  can_generate_any: boolean;
  topics_checked: number;
  topics_can_generate: number;
  blocked_by_settings_topics: number;
  no_source_topics: number;
  estimated_success_rate: number;
  total_probed: number;
  passed_all_filters: number;
  primary_bottleneck: string;
  message: string;
};

export type FeedResponse = {
  page: number;
  limit: number;
  total: number;
  reels: Reel[];
  generation_id?: string | null;
  response_profile?: string | null;
  refinement_job_id?: string | null;
  refinement_status?: string | null;
};

export type RefinementStatusResponse = {
  job_id: string;
  status: string;
  material_id: string;
  request_key: string;
  source_generation_id: string;
  result_generation_id?: string | null;
  active_generation_id?: string | null;
  completed_at?: string | null;
  error?: string | null;
};

export type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

export type ChatResponse = {
  answer: string;
};

// POST /api/ingest/url — backend reel ingestion pipeline (YouTube / Instagram / TikTok).

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
  platform: "yt" | "ig" | "tt";
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

// POST /api/ingest/search — topic-based multi-platform discovery.

export type IngestSearchRequest = {
  query: string;
  platforms?: Array<"yt" | "ig" | "tt">;
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
  platform: "yt" | "ig" | "tt";
  source_url: string;
  status: "ok" | "error" | "skipped" | "rate_limited";
  reel?: Reel | null;
  metadata?: IngestMetadata | null;
  error?: string | null;
};

export type IngestSearchResult = {
  query: string;
  material_id: string;
  platforms: Array<"yt" | "ig" | "tt">;
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
