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
  concept_id: string;
  concept_title: string;
  video_title?: string;
  video_description?: string;
  ai_summary?: string;
  video_url: string;
  t_start: number;
  t_end: number;
  transcript_snippet: string;
  takeaways: string[];
  captions?: CaptionCue[];
  score: number;
  relevance_score?: number;
  matched_terms?: string[];
  relevance_reason?: string;
  concept_position?: number;
  total_concepts?: number;
};

export type CaptionCue = {
  start: number;
  end: number;
  text: string;
};

export type ReelsGenerateResponse = {
  reels: Reel[];
};

export type FeedResponse = {
  page: number;
  limit: number;
  total: number;
  reels: Reel[];
};

export type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

export type ChatResponse = {
  answer: string;
};

export type CommunityReelPlatform = "youtube" | "instagram" | "tiktok";

export type CommunityReelEmbed = {
  id: string;
  platform: CommunityReelPlatform;
  sourceUrl: string;
  embedUrl: string;
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
  thumbnailUrl: string;
  featured: boolean;
};
