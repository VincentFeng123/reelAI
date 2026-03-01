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
