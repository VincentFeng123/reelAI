import type {
  ChatMessage,
  ChatResponse,
  FeedResponse,
  MaterialResponse,
  ReelsGenerateResponse,
} from "@/lib/types";

const API_BASE = (
  process.env.NEXT_PUBLIC_API_BASE || (process.env.NODE_ENV === "development" ? "http://127.0.0.1:8000" : "")
).replace(/\/$/, "");
const BACKEND_DOWN_ERROR = API_BASE
  ? `Cannot reach backend at ${API_BASE}. Make sure the backend server is running.`
  : "Cannot reach backend. Check your deployment and API routes.";

export async function uploadMaterial(params: {
  text?: string;
  file?: File;
  subjectTag?: string;
}): Promise<MaterialResponse> {
  const form = new FormData();
  if (params.text) {
    form.append("text", params.text);
  }
  if (params.file) {
    form.append("file", params.file);
  }
  if (params.subjectTag) {
    form.append("subject_tag", params.subjectTag);
  }

  const res = await safeFetch(`${API_BASE}/api/material`, {
    method: "POST",
    body: form,
  });
  return res.json();
}

export async function generateReels(params: {
  materialId: string;
  numReels?: number;
  conceptId?: string;
}): Promise<ReelsGenerateResponse> {
  const res = await safeFetch(`${API_BASE}/api/reels/generate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      material_id: params.materialId,
      concept_id: params.conceptId,
      num_reels: params.numReels ?? 7,
      creative_commons_only: false,
    }),
  });

  return res.json();
}

export async function fetchFeed(params: {
  materialId: string;
  page: number;
  limit: number;
  prefetch?: number;
  autofill?: boolean;
}): Promise<FeedResponse> {
  const query = new URLSearchParams({
    material_id: params.materialId,
    page: String(params.page),
    limit: String(params.limit),
    autofill: String(params.autofill ?? true),
    prefetch: String(params.prefetch ?? 7),
    creative_commons_only: "false",
  });

  const res = await safeFetch(`${API_BASE}/api/feed?${query}`, {
    cache: "no-store",
  });

  return res.json();
}

export async function sendFeedback(params: {
  reelId: string;
  helpful?: boolean;
  confusing?: boolean;
  rating?: number;
  saved?: boolean;
}): Promise<void> {
  await safeFetch(`${API_BASE}/api/reels/feedback`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      reel_id: params.reelId,
      helpful: Boolean(params.helpful),
      confusing: Boolean(params.confusing),
      rating: params.rating,
      saved: Boolean(params.saved),
    }),
  });
}

export async function askStudyChat(params: {
  message: string;
  topic?: string;
  text?: string;
  history?: ChatMessage[];
}): Promise<ChatResponse> {
  const res = await safeFetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      message: params.message,
      topic: params.topic,
      text: params.text,
      history: params.history ?? [],
    }),
  });
  return res.json();
}

async function safeFetch(url: string, init?: RequestInit): Promise<Response> {
  let response: Response;
  try {
    response = await fetch(url, init);
  } catch {
    throw new Error(BACKEND_DOWN_ERROR);
  }

  if (!response.ok) {
    throw new Error(await safeError(response));
  }
  return response;
}

async function safeError(response: Response): Promise<string> {
  try {
    const json = await response.json();
    return json?.detail || json?.message || `Request failed (${response.status})`;
  } catch {
    return `Request failed (${response.status})`;
  }
}
