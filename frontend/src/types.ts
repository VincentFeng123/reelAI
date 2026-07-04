export interface Clip {
  n: number;
  video_id: string;
  facet: string;
  reason: string;
  start: number;
  end: number;
  duration: number;
  embed_url: string;
  path: string | null;
  // structure-first extras (optional; present when analysis_profile="full")
  role?: string;
  title?: string;
  context_card?: string;
  sequence_index?: number;
  prerequisite_clips?: number[];
  // FE2 quality signals (surfaced by _build_embed_clips)
  notes?: string[];
  final_quality?: number | null;
  warnings?: string[];
  ship_flagged?: boolean;
}

export type JobStatus = "pending" | "running" | "done" | "error" | "cancelled";

export interface JobSnapshot {
  job_id: string;
  status: JobStatus;
  stage: string;
  pct: number;
  message: string;
  title: string;
  video_id: string | null;
  clips: Clip[];
  notes: string;
  error: string | null;
}

export interface Settings {
  allow_question_exclaim_ends: boolean;
  mmr_lambda: number;
  export_resolution: number;
}

export type Phase = "input" | "processing" | "results" | "error";
