import type {
  ChatMessage,
  ChatResponse,
  CommunitySet,
  CommunityReelPlatform,
  FeedResponse,
  MaterialResponse,
  ReelsCanGenerateAnyResponse,
  ReelsCanGenerateResponse,
  ReelsGenerateResponse,
} from "@/lib/types";
import type { PreferredVideoDuration, VideoPoolMode } from "@/lib/settings";

const RAW_API_BASE = (
  process.env.NEXT_PUBLIC_API_BASE || (process.env.NODE_ENV === "development" ? "http://127.0.0.1:8000" : "")
).replace(/\/$/, "");
const DEFAULT_REQUEST_TIMEOUT_MS = 30_000;
const FEED_REQUEST_TIMEOUT_MS = 45_000;
const GENERATE_REQUEST_TIMEOUT_MS = 90_000;
const BACKEND_DOWN_ERROR = RAW_API_BASE
  ? `Cannot reach backend at ${RAW_API_BASE}. Make sure the backend server is running.`
  : "Cannot reach backend. Check your deployment and API routes.";

function apiUrl(path: string): string {
  const cleanPath = path.startsWith("/") ? path : `/${path}`;
  if (!RAW_API_BASE) {
    return `/api${cleanPath}`;
  }
  const cleanBase = RAW_API_BASE.replace(/\/$/, "");
  if (cleanBase.endsWith("/api")) {
    return `${cleanBase}${cleanPath}`;
  }
  return `${cleanBase}/api${cleanPath}`;
}

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

  const res = await safeFetch(apiUrl("/material"), {
    method: "POST",
    body: form,
  });
  return res.json();
}

export async function generateReels(params: {
  materialId: string;
  numReels?: number;
  conceptId?: string;
  generationMode?: "slow" | "fast";
  minRelevance?: number;
  videoPoolMode?: VideoPoolMode;
  preferredVideoDuration?: PreferredVideoDuration;
  targetClipDurationSec?: number;
  targetClipDurationMinSec?: number;
  targetClipDurationMaxSec?: number;
}): Promise<ReelsGenerateResponse> {
  const res = await safeFetch(apiUrl("/reels/generate"), {
    method: "POST",
    timeoutMs: GENERATE_REQUEST_TIMEOUT_MS,
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      material_id: params.materialId,
      concept_id: params.conceptId,
      num_reels: params.numReels ?? 7,
      creative_commons_only: false,
      generation_mode: params.generationMode ?? "slow",
      min_relevance: Number.isFinite(params.minRelevance) ? params.minRelevance : undefined,
      video_pool_mode: params.videoPoolMode ?? "short-first",
      preferred_video_duration: params.preferredVideoDuration ?? "any",
      target_clip_duration_sec: Number.isFinite(params.targetClipDurationSec) ? Math.round(params.targetClipDurationSec as number) : undefined,
      target_clip_duration_min_sec: Number.isFinite(params.targetClipDurationMinSec)
        ? Math.round(params.targetClipDurationMinSec as number)
        : undefined,
      target_clip_duration_max_sec: Number.isFinite(params.targetClipDurationMaxSec)
        ? Math.round(params.targetClipDurationMaxSec as number)
        : undefined,
    }),
  });

  return res.json();
}

export async function checkReelsCanGenerate(params: {
  materialId: string;
  conceptId?: string;
  generationMode?: "slow" | "fast";
  minRelevance?: number;
  videoPoolMode?: VideoPoolMode;
  preferredVideoDuration?: PreferredVideoDuration;
  targetClipDurationSec?: number;
  targetClipDurationMinSec?: number;
  targetClipDurationMaxSec?: number;
}): Promise<ReelsCanGenerateResponse> {
  const res = await safeFetch(apiUrl("/reels/can-generate"), {
    method: "POST",
    timeoutMs: FEED_REQUEST_TIMEOUT_MS,
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      material_id: params.materialId,
      concept_id: params.conceptId,
      num_reels: 1,
      creative_commons_only: false,
      generation_mode: params.generationMode ?? "slow",
      min_relevance: Number.isFinite(params.minRelevance) ? params.minRelevance : undefined,
      video_pool_mode: params.videoPoolMode ?? "short-first",
      preferred_video_duration: params.preferredVideoDuration ?? "any",
      target_clip_duration_sec: Number.isFinite(params.targetClipDurationSec) ? Math.round(params.targetClipDurationSec as number) : undefined,
      target_clip_duration_min_sec: Number.isFinite(params.targetClipDurationMinSec)
        ? Math.round(params.targetClipDurationMinSec as number)
        : undefined,
      target_clip_duration_max_sec: Number.isFinite(params.targetClipDurationMaxSec)
        ? Math.round(params.targetClipDurationMaxSec as number)
        : undefined,
    }),
  });
  return res.json();
}

export async function checkReelsCanGenerateAny(params: {
  materialIds?: string[];
  generationMode?: "slow" | "fast";
  minRelevance?: number;
  videoPoolMode?: VideoPoolMode;
  preferredVideoDuration?: PreferredVideoDuration;
  targetClipDurationSec?: number;
  targetClipDurationMinSec?: number;
  targetClipDurationMaxSec?: number;
}): Promise<ReelsCanGenerateAnyResponse> {
  const materialIds = Array.isArray(params.materialIds)
    ? params.materialIds.map((id) => String(id || "").trim()).filter(Boolean)
    : [];
  const res = await safeFetch(apiUrl("/reels/can-generate-any"), {
    method: "POST",
    timeoutMs: FEED_REQUEST_TIMEOUT_MS,
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      material_ids: materialIds,
      creative_commons_only: false,
      generation_mode: params.generationMode ?? "slow",
      min_relevance: Number.isFinite(params.minRelevance) ? params.minRelevance : undefined,
      video_pool_mode: params.videoPoolMode ?? "short-first",
      preferred_video_duration: params.preferredVideoDuration ?? "any",
      target_clip_duration_sec: Number.isFinite(params.targetClipDurationSec) ? Math.round(params.targetClipDurationSec as number) : undefined,
      target_clip_duration_min_sec: Number.isFinite(params.targetClipDurationMinSec)
        ? Math.round(params.targetClipDurationMinSec as number)
        : undefined,
      target_clip_duration_max_sec: Number.isFinite(params.targetClipDurationMaxSec)
        ? Math.round(params.targetClipDurationMaxSec as number)
        : undefined,
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
  generationMode?: "slow" | "fast";
  minRelevance?: number;
  videoPoolMode?: VideoPoolMode;
  preferredVideoDuration?: PreferredVideoDuration;
  targetClipDurationSec?: number;
  targetClipDurationMinSec?: number;
  targetClipDurationMaxSec?: number;
}): Promise<FeedResponse> {
  const query = new URLSearchParams({
    material_id: params.materialId,
    page: String(params.page),
    limit: String(params.limit),
    autofill: String(params.autofill ?? true),
    prefetch: String(params.prefetch ?? 7),
    creative_commons_only: "false",
    generation_mode: params.generationMode ?? "slow",
    video_pool_mode: params.videoPoolMode ?? "short-first",
    preferred_video_duration: params.preferredVideoDuration ?? "any",
    target_clip_duration_sec: String(
      Number.isFinite(params.targetClipDurationSec) ? Math.round(params.targetClipDurationSec as number) : 55,
    ),
  });
  if (Number.isFinite(params.targetClipDurationMinSec)) {
    query.set("target_clip_duration_min_sec", String(Math.round(params.targetClipDurationMinSec as number)));
  }
  if (Number.isFinite(params.targetClipDurationMaxSec)) {
    query.set("target_clip_duration_max_sec", String(Math.round(params.targetClipDurationMaxSec as number)));
  }
  if (Number.isFinite(params.minRelevance)) {
    query.set("min_relevance", String(params.minRelevance));
  }

  const res = await safeFetch(`${apiUrl("/feed")}?${query}`, {
    cache: "no-store",
    timeoutMs: FEED_REQUEST_TIMEOUT_MS,
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
  await safeFetch(apiUrl("/reels/feedback"), {
    method: "POST",
    timeoutMs: DEFAULT_REQUEST_TIMEOUT_MS,
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
  const res = await safeFetch(apiUrl("/chat"), {
    method: "POST",
    timeoutMs: DEFAULT_REQUEST_TIMEOUT_MS,
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

type CommunitySetApi = {
  id: string;
  title: string;
  description: string;
  tags: string[];
  reels: Array<{
    id: string;
    platform: CommunityReelPlatform;
    source_url: string;
    embed_url: string;
  }>;
  reel_count: number;
  curator: string;
  likes: number;
  learners: number;
  updated_label: string;
  thumbnail_url: string;
  featured: boolean;
};

function normalizeCommunitySet(raw: unknown): CommunitySet | null {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    return null;
  }
  const row = raw as Partial<CommunitySetApi>;
  const title = typeof row.title === "string" ? row.title.trim() : "";
  const description = typeof row.description === "string" ? row.description.trim() : "";
  if (!title || !description) {
    return null;
  }
  const reels = Array.isArray(row.reels)
    ? row.reels
        .map((reel) => {
          if (!reel || typeof reel !== "object") {
            return null;
          }
          const platform = reel.platform;
          if (platform !== "youtube" && platform !== "instagram" && platform !== "tiktok") {
            return null;
          }
          const sourceUrl = typeof reel.source_url === "string" ? reel.source_url.trim() : "";
          const embedUrl = typeof reel.embed_url === "string" ? reel.embed_url.trim() : "";
          if (!sourceUrl || !embedUrl) {
            return null;
          }
          return {
            id: typeof reel.id === "string" && reel.id.trim() ? reel.id.trim() : `community-reel-${Math.random().toString(36).slice(2, 10)}`,
            platform,
            sourceUrl,
            embedUrl,
          };
        })
        .filter(Boolean) as CommunitySet["reels"]
    : [];

  const tags = Array.isArray(row.tags)
    ? row.tags.map((tag) => String(tag || "").trim().toLowerCase()).filter(Boolean).slice(0, 6)
    : [];

  return {
    id: typeof row.id === "string" && row.id.trim() ? row.id.trim() : `community-set-${Math.random().toString(36).slice(2, 10)}`,
    title,
    description,
    tags,
    reels,
    reelCount: Math.max(reels.length, Number(row.reel_count) || 0),
    curator: typeof row.curator === "string" && row.curator.trim() ? row.curator.trim() : "Community member",
    likes: Math.max(0, Math.floor(Number(row.likes) || 0)),
    learners: Math.max(0, Math.floor(Number(row.learners) || 0)),
    updatedLabel: typeof row.updated_label === "string" && row.updated_label.trim() ? row.updated_label.trim() : "Updated just now",
    thumbnailUrl: typeof row.thumbnail_url === "string" && row.thumbnail_url.trim() ? row.thumbnail_url.trim() : "/images/community/ai-systems.svg",
    featured: Boolean(row.featured),
  };
}

export async function fetchCommunitySets(): Promise<CommunitySet[]> {
  const res = await safeFetch(apiUrl("/community/sets"), {
    cache: "no-store",
    timeoutMs: DEFAULT_REQUEST_TIMEOUT_MS,
  });
  const json = await res.json();
  const rows = Array.isArray(json?.sets) ? json.sets : [];
  return rows.map(normalizeCommunitySet).filter(Boolean) as CommunitySet[];
}

export async function createCommunitySet(params: {
  title: string;
  description: string;
  tags: string[];
  reels: Array<{
    platform: CommunityReelPlatform;
    sourceUrl: string;
    embedUrl: string;
  }>;
  thumbnailUrl: string;
  curator?: string;
}): Promise<CommunitySet> {
  const res = await safeFetch(apiUrl("/community/sets"), {
    method: "POST",
    timeoutMs: DEFAULT_REQUEST_TIMEOUT_MS,
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      title: params.title,
      description: params.description,
      tags: params.tags,
      reels: params.reels.map((reel) => ({
        platform: reel.platform,
        source_url: reel.sourceUrl,
        embed_url: reel.embedUrl,
      })),
      thumbnail_url: params.thumbnailUrl,
      curator: params.curator,
    }),
  });

  const created = normalizeCommunitySet(await res.json());
  if (!created) {
    throw new Error("Backend returned an invalid community set payload.");
  }
  return created;
}

type SafeFetchInit = RequestInit & {
  timeoutMs?: number;
};

async function safeFetch(url: string, init?: SafeFetchInit): Promise<Response> {
  const timeoutMs = Math.max(1_000, Number(init?.timeoutMs) || DEFAULT_REQUEST_TIMEOUT_MS);
  const upstreamSignal = init?.signal;
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  if (upstreamSignal) {
    if (upstreamSignal.aborted) {
      controller.abort();
    } else {
      upstreamSignal.addEventListener("abort", () => controller.abort(), { once: true });
    }
  }

  let response: Response;
  try {
    const requestInit: RequestInit = {
      ...init,
      signal: controller.signal,
    };
    delete (requestInit as SafeFetchInit).timeoutMs;
    response = await fetch(url, requestInit);
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") {
      throw new Error(`Request timed out after ${Math.round(timeoutMs / 1000)}s.`);
    }
    throw new Error(BACKEND_DOWN_ERROR);
  } finally {
    clearTimeout(timeoutId);
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
