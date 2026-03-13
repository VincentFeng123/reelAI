import type {
  ChatMessage,
  ChatResponse,
  CommunityAccount,
  CommunityAuthSession,
  CommunitySet,
  CommunityReelPlatform,
  FeedResponse,
  MaterialResponse,
  ReelsCanGenerateAnyResponse,
  ReelsCanGenerateResponse,
  ReelsGenerateResponse,
} from "@/lib/types";
import type { StoredHistoryItem } from "@/lib/historyStorage";
import type { PreferredVideoDuration, VideoPoolMode } from "@/lib/settings";

const RAW_API_BASE = (
  process.env.NEXT_PUBLIC_API_BASE || (process.env.NODE_ENV === "development" ? "http://127.0.0.1:8000" : "")
).replace(/\/$/, "");
const BACKEND_DOWN_ERROR = RAW_API_BASE
  ? `Cannot reach backend at ${RAW_API_BASE}. Make sure the backend server is running.`
  : "Cannot reach backend. Check your deployment and API routes.";
const COMMUNITY_OWNER_KEY_STORAGE_KEY = "studyreels-community-owner-key";
export const COMMUNITY_OWNED_SET_IDS_STORAGE_KEY = "studyreels-community-owned-set-ids";
const COMMUNITY_ACCOUNT_STORAGE_KEY = "studyreels-community-account";
const COMMUNITY_SESSION_TOKEN_STORAGE_KEY = "studyreels-community-session-token";
const COMMUNITY_OWNER_HEADER = "X-StudyReels-Owner-Key";
const COMMUNITY_SESSION_HEADER = "X-StudyReels-Session-Token";
export const COMMUNITY_AUTH_CHANGED_EVENT = "studyreels-community-auth-changed";
let communityOwnerKeyMemoryFallback: string | null = null;

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

function normalizeOwnedCommunitySetIds(raw: unknown): string[] {
  if (!Array.isArray(raw)) {
    return [];
  }
  const seen = new Set<string>();
  const normalized: string[] = [];
  for (const value of raw) {
    const id = String(value || "").trim();
    if (!id || seen.has(id)) {
      continue;
    }
    seen.add(id);
    normalized.push(id);
  }
  return normalized;
}

function createCommunityOwnerKey(): string {
  if (typeof crypto !== "undefined") {
    if (typeof crypto.randomUUID === "function") {
      return `${crypto.randomUUID()}-${crypto.randomUUID()}`;
    }
    if (typeof crypto.getRandomValues === "function") {
      const bytes = new Uint8Array(24);
      crypto.getRandomValues(bytes);
      return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
    }
  }
  // crypto is unavailable — generate a best-effort key from multiple entropy
  // sources. This path only runs in very old or non-browser environments.
  const segments: string[] = [];
  for (let i = 0; i < 6; i++) {
    segments.push(Math.random().toString(36).slice(2, 10));
  }
  return segments.join("-");
}

function normalizeStoredCommunityOwnerKey(raw: string | null): string | null {
  const ownerKey = (raw || "").trim();
  return ownerKey.length >= 24 ? ownerKey : null;
}

function persistCommunityOwnerKey(ownerKey: string): void {
  communityOwnerKeyMemoryFallback = ownerKey;
  if (typeof window === "undefined") {
    return;
  }
  try {
    window.sessionStorage.setItem(COMMUNITY_OWNER_KEY_STORAGE_KEY, ownerKey);
    window.localStorage.removeItem(COMMUNITY_OWNER_KEY_STORAGE_KEY);
  } catch {
    // Ignore storage failures and keep the in-memory fallback.
  }
}

function clearCommunityOwnerKey(): void {
  communityOwnerKeyMemoryFallback = null;
  if (typeof window === "undefined") {
    return;
  }
  try {
    window.sessionStorage.removeItem(COMMUNITY_OWNER_KEY_STORAGE_KEY);
  } catch {
    // Ignore storage failures during logout/session teardown.
  }
  try {
    window.localStorage.removeItem(COMMUNITY_OWNER_KEY_STORAGE_KEY);
  } catch {
    // Ignore storage failures during logout/session teardown.
  }
}

function getCommunityOwnerKey(): string | null {
  if (typeof window === "undefined") {
    return communityOwnerKeyMemoryFallback;
  }
  try {
    const existingSessionKey = normalizeStoredCommunityOwnerKey(
      window.sessionStorage.getItem(COMMUNITY_OWNER_KEY_STORAGE_KEY),
    );
    if (existingSessionKey) {
      communityOwnerKeyMemoryFallback = existingSessionKey;
      return existingSessionKey;
    }
    const legacyLocalKey = normalizeStoredCommunityOwnerKey(
      window.localStorage.getItem(COMMUNITY_OWNER_KEY_STORAGE_KEY),
    );
    if (legacyLocalKey) {
      persistCommunityOwnerKey(legacyLocalKey);
      return legacyLocalKey;
    }
    const next = communityOwnerKeyMemoryFallback || createCommunityOwnerKey();
    persistCommunityOwnerKey(next);
    return next;
  } catch {
    if (communityOwnerKeyMemoryFallback) {
      return communityOwnerKeyMemoryFallback;
    }
    const next = createCommunityOwnerKey();
    communityOwnerKeyMemoryFallback = next;
    return next;
  }
}

function communityOwnerHeaders(): HeadersInit {
  const ownerKey = getCommunityOwnerKey();
  return ownerKey ? { [COMMUNITY_OWNER_HEADER]: ownerKey } : {};
}

function communitySessionHeaders(): HeadersInit {
  const session = readCommunityAuthSession();
  return session?.sessionToken ? { [COMMUNITY_SESSION_HEADER]: session.sessionToken } : {};
}

function communityRequestHeaders(): HeadersInit {
  return {
    ...communityOwnerHeaders(),
    ...communitySessionHeaders(),
  };
}

function normalizeCommunityAccount(raw: unknown): CommunityAccount | null {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    return null;
  }
  const row = raw as Partial<CommunityAccount> & { is_verified?: unknown; isVerified?: unknown; email?: unknown };
  const id = typeof row.id === "string" ? row.id.trim() : "";
  const username = typeof row.username === "string" ? row.username.trim() : "";
  if (!id || !username) {
    return null;
  }
  const email = typeof row.email === "string" && row.email.trim() ? row.email.trim() : null;
  const isVerifiedRaw = typeof row.isVerified === "boolean" ? row.isVerified : row.is_verified;
  return {
    id,
    username,
    email,
    isVerified: typeof isVerifiedRaw === "boolean" ? isVerifiedRaw : true,
  };
}

type StoredCommunityAuthSession = {
  account: CommunityAccount;
  sessionToken: string;
};

function saveCommunityAuthSessionStorage(session: StoredCommunityAuthSession | null): void {
  if (typeof window === "undefined") {
    return;
  }
  try {
    // Migrate any legacy localStorage token to sessionStorage on first access.
    const legacyToken = window.localStorage.getItem(COMMUNITY_SESSION_TOKEN_STORAGE_KEY);
    if (legacyToken) {
      window.localStorage.removeItem(COMMUNITY_SESSION_TOKEN_STORAGE_KEY);
    }

    if (!session) {
      window.localStorage.removeItem(COMMUNITY_ACCOUNT_STORAGE_KEY);
      window.sessionStorage.removeItem(COMMUNITY_SESSION_TOKEN_STORAGE_KEY);
      return;
    }
    // Account metadata (non-secret) stays in localStorage for cross-tab display.
    // The session token goes in sessionStorage to limit XSS blast radius — an
    // attacker would need code running in the same tab to read it.
    window.localStorage.setItem(COMMUNITY_ACCOUNT_STORAGE_KEY, JSON.stringify(session.account));
    window.sessionStorage.setItem(COMMUNITY_SESSION_TOKEN_STORAGE_KEY, session.sessionToken);
  } catch {
    // Ignore storage failures and keep the in-memory response path usable.
  } finally {
    window.dispatchEvent(new Event(COMMUNITY_AUTH_CHANGED_EVENT));
  }
}

export function clearCommunityAuthSession(): void {
  clearCommunityOwnerKey();
  saveCommunityAuthSessionStorage(null);
}

export function readCommunityAuthSession(): CommunityAuthSession | null {
  if (typeof window === "undefined") {
    return null;
  }
  try {
    const account = normalizeCommunityAccount(JSON.parse(window.localStorage.getItem(COMMUNITY_ACCOUNT_STORAGE_KEY) || "null"));
    // Read token from sessionStorage (preferred) with localStorage migration fallback.
    let sessionToken = (window.sessionStorage.getItem(COMMUNITY_SESSION_TOKEN_STORAGE_KEY) || "").trim();
    if (sessionToken.length < 24) {
      // Migrate legacy localStorage token if present.
      const legacyToken = (window.localStorage.getItem(COMMUNITY_SESSION_TOKEN_STORAGE_KEY) || "").trim();
      if (legacyToken.length >= 24) {
        sessionToken = legacyToken;
        window.sessionStorage.setItem(COMMUNITY_SESSION_TOKEN_STORAGE_KEY, sessionToken);
        window.localStorage.removeItem(COMMUNITY_SESSION_TOKEN_STORAGE_KEY);
      }
    }
    if (!account || sessionToken.length < 24) {
      return null;
    }
    return {
      account,
      sessionToken,
      claimedLegacySets: 0,
      // Derive verification state from the persisted account instead of
      // hardcoding false — prevents a stale "verified" flash after email change
      // or before the first /auth/me round-trip refreshes the UI.
      verificationRequired: account.isVerified === false,
      verificationCodeDebug: null,
    };
  } catch {
    return null;
  }
}

export function readOwnedCommunitySetIds(): string[] {
  if (typeof window === "undefined") {
    return [];
  }
  try {
    return normalizeOwnedCommunitySetIds(JSON.parse(window.localStorage.getItem(COMMUNITY_OWNED_SET_IDS_STORAGE_KEY) || "[]"));
  } catch {
    return [];
  }
}

export function saveOwnedCommunitySetIds(nextIds: string[]): string[] {
  const normalized = normalizeOwnedCommunitySetIds(nextIds);
  if (typeof window !== "undefined") {
    try {
      window.localStorage.setItem(COMMUNITY_OWNED_SET_IDS_STORAGE_KEY, JSON.stringify(normalized));
    } catch {
      // Ignore storage failures and keep in-memory state.
    }
  }
  return normalized;
}

type CommunityAuthResponseApi = {
  account?: unknown;
  session_token?: unknown;
  claimed_legacy_sets?: unknown;
  verification_required?: unknown;
  verification_code_debug?: unknown;
};

function normalizeCommunityAuthSession(raw: unknown): CommunityAuthSession | null {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    return null;
  }
  const row = raw as CommunityAuthResponseApi;
  const account = normalizeCommunityAccount(row.account);
  const sessionToken = typeof row.session_token === "string" ? row.session_token.trim() : "";
  if (!account || sessionToken.length < 24) {
    return null;
  }
  return {
    account,
    sessionToken,
    claimedLegacySets: Math.max(0, Math.floor(Number(row.claimed_legacy_sets) || 0)),
    verificationRequired: Boolean(row.verification_required),
    verificationCodeDebug:
      typeof row.verification_code_debug === "string" && row.verification_code_debug.trim()
        ? row.verification_code_debug.trim()
        : null,
  };
}

async function parseJsonResponse<T>(response: Response): Promise<T> {
  try {
    return await response.json() as T;
  } catch {
    throw new Error("Backend returned an invalid JSON response.");
  }
}

export async function registerCommunityAccount(params: {
  username: string;
  email: string;
  password: string;
}): Promise<CommunityAuthSession> {
  const res = await safeFetch(apiUrl("/community/auth/register"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...communityOwnerHeaders(),
    },
    body: JSON.stringify({
      username: params.username,
      email: params.email,
      password: params.password,
    }),
  });
  const session = normalizeCommunityAuthSession(await parseJsonResponse<unknown>(res));
  if (!session) {
    throw new Error("Backend returned an invalid auth response.");
  }
  saveCommunityAuthSessionStorage({
    account: session.account,
    sessionToken: session.sessionToken,
  });
  return session;
}

export async function loginCommunityAccount(params: {
  username: string;
  password: string;
}): Promise<CommunityAuthSession> {
  const res = await safeFetch(apiUrl("/community/auth/login"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...communityOwnerHeaders(),
    },
    body: JSON.stringify({
      username: params.username,
      password: params.password,
    }),
  });
  const session = normalizeCommunityAuthSession(await parseJsonResponse<unknown>(res));
  if (!session) {
    throw new Error("Backend returned an invalid auth response.");
  }
  saveCommunityAuthSessionStorage({
    account: session.account,
    sessionToken: session.sessionToken,
  });
  return session;
}

export async function verifyCommunityAccount(params: {
  code: string;
}): Promise<{ account: CommunityAccount; claimedLegacySets: number }> {
  const res = await safeFetch(apiUrl("/community/auth/verify"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...communityRequestHeaders(),
    },
    body: JSON.stringify({
      code: params.code,
    }),
  });
  const json = await parseJsonResponse<{ account?: unknown; claimed_legacy_sets?: unknown }>(res);
  const account = normalizeCommunityAccount(json?.account);
  if (!account) {
    throw new Error("Backend returned an invalid verification response.");
  }
  const stored = readCommunityAuthSession();
  if (!stored?.sessionToken) {
    throw new Error("Session expired. Sign in again.");
  }
  saveCommunityAuthSessionStorage({
    account,
    sessionToken: stored.sessionToken,
  });
  return {
    account,
    claimedLegacySets: Math.max(0, Math.floor(Number(json?.claimed_legacy_sets) || 0)),
  };
}

export async function resendCommunityVerification(): Promise<{ account: CommunityAccount; verificationCodeDebug: string | null }> {
  const res = await safeFetch(apiUrl("/community/auth/resend-verification"), {
    method: "POST",
    headers: {
      ...communityRequestHeaders(),
    },
  });
  const json = await parseJsonResponse<{ account?: unknown; verification_code_debug?: unknown }>(res);
  const account = normalizeCommunityAccount(json?.account);
  if (!account) {
    throw new Error("Backend returned an invalid verification response.");
  }
  const stored = readCommunityAuthSession();
  if (!stored?.sessionToken) {
    throw new Error("Session expired. Sign in again.");
  }
  saveCommunityAuthSessionStorage({
    account,
    sessionToken: stored.sessionToken,
  });
  return {
    account,
    verificationCodeDebug:
      typeof json?.verification_code_debug === "string" && json.verification_code_debug.trim()
        ? json.verification_code_debug.trim()
        : null,
  };
}

export async function changeCommunityVerificationEmail(params: {
  email: string;
  currentPassword: string;
}): Promise<{ account: CommunityAccount; verificationCodeDebug: string | null }> {
  const res = await safeFetch(apiUrl("/community/auth/change-email"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...communityRequestHeaders(),
    },
    body: JSON.stringify({
      email: params.email,
      current_password: params.currentPassword,
    }),
  });
  const json = await parseJsonResponse<{ account?: unknown; verification_code_debug?: unknown }>(res);
  const account = normalizeCommunityAccount(json?.account);
  if (!account) {
    throw new Error("Backend returned an invalid email-change response.");
  }
  const stored = readCommunityAuthSession();
  if (!stored?.sessionToken) {
    throw new Error("Session expired. Sign in again.");
  }
  saveCommunityAuthSessionStorage({
    account,
    sessionToken: stored.sessionToken,
  });
  return {
    account,
    verificationCodeDebug:
      typeof json?.verification_code_debug === "string" && json.verification_code_debug.trim()
        ? json.verification_code_debug.trim()
        : null,
  };
}

export async function fetchCommunityAccount(): Promise<CommunityAccount | null> {
  const stored = readCommunityAuthSession();
  if (!stored?.sessionToken) {
    return null;
  }
  try {
    const res = await safeFetch(apiUrl("/community/auth/me"), {
      cache: "no-store",
      headers: { ...communitySessionHeaders() },
    });
    const json = await parseJsonResponse<{ account?: unknown }>(res);
    const account = normalizeCommunityAccount(json?.account);
    if (!account) {
      clearCommunityAuthSession();
      return null;
    }
    saveCommunityAuthSessionStorage({
      account,
      sessionToken: stored.sessionToken,
    });
    return account;
  } catch (error) {
    if (isSessionExpiredError(error)) {
      clearCommunityAuthSession();
      return null;
    }
    throw error;
  }
}

export async function logoutCommunityAccount(): Promise<void> {
  const stored = readCommunityAuthSession();
  try {
    if (stored?.sessionToken) {
      await safeFetch(apiUrl("/community/auth/logout"), {
        method: "POST",
        headers: { ...communitySessionHeaders() },
      });
    }
  } finally {
    clearCommunityAuthSession();
  }
}

export async function changeCommunityPassword(params: {
  currentPassword: string;
  newPassword: string;
}): Promise<void> {
  await safeFetch(apiUrl("/community/auth/change-password"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...communitySessionHeaders(),
    },
    body: JSON.stringify({
      current_password: params.currentPassword,
      new_password: params.newPassword,
    }),
  });
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
  return parseJsonResponse<MaterialResponse>(res);
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
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      material_id: params.materialId,
      concept_id: params.conceptId,
      num_reels: params.numReels ?? 7,
      creative_commons_only: false,
      generation_mode: params.generationMode ?? "fast",
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

  return parseJsonResponse<ReelsGenerateResponse>(res);
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
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      material_id: params.materialId,
      concept_id: params.conceptId,
      num_reels: 1,
      creative_commons_only: false,
      generation_mode: params.generationMode ?? "fast",
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
  return parseJsonResponse<ReelsCanGenerateResponse>(res);
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
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      material_ids: materialIds,
      creative_commons_only: false,
      generation_mode: params.generationMode ?? "fast",
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
  return parseJsonResponse<ReelsCanGenerateAnyResponse>(res);
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
    generation_mode: params.generationMode ?? "fast",
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
  });

  return parseJsonResponse<FeedResponse>(res);
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
  return parseJsonResponse<ChatResponse>(res);
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
    t_start_sec?: number;
    t_end_sec?: number;
  }>;
  reel_count: number;
  curator: string;
  likes: number;
  learners: number;
  updated_label: string;
  updated_at?: string | null;
  created_at?: string | null;
  thumbnail_url: string;
  featured: boolean;
};

type CommunityHistoryItemApi = {
  material_id: string;
  title: string;
  updated_at: number;
  starred?: boolean;
  generation_mode?: "slow" | "fast";
  source?: "search" | "community";
  feed_query?: string | null;
};

let communityHistorySyncVersion = 0;
let communityHistorySyncChain: Promise<void> = Promise.resolve();

function parseApiTimestampMs(value: unknown): number | null {
  if (typeof value === "number") {
    return Number.isFinite(value) && value > 0 ? value : null;
  }
  if (typeof value !== "string") {
    return null;
  }
  const trimmed = value.trim();
  if (!trimmed) {
    return null;
  }
  if (/^\d+$/.test(trimmed)) {
    const numeric = Number(trimmed);
    if (Number.isFinite(numeric) && numeric > 0) {
      return numeric >= 1_000_000_000_000 ? numeric : numeric * 1000;
    }
  }
  const directParsed = Date.parse(trimmed);
  if (Number.isFinite(directParsed)) {
    return directParsed;
  }
  const normalized = trimmed.replace(" ", "T");
  let candidate = normalized;
  if (/[+-]\d{4}$/.test(candidate)) {
    candidate = `${candidate.slice(0, -5)}${candidate.slice(-5, -2)}:${candidate.slice(-2)}`;
  } else if (/[+-]\d{2}$/.test(candidate)) {
    candidate = `${candidate}:00`;
  } else if (!/(?:z|[+-]\d{2}:\d{2})$/i.test(candidate) && /\d{2}:\d{2}:\d{2}/.test(candidate)) {
    candidate = `${candidate}Z`;
  }
  const normalizedParsed = Date.parse(candidate);
  return Number.isFinite(normalizedParsed) ? normalizedParsed : null;
}

function isRelativeNowUpdatedLabel(value: unknown): boolean {
  if (typeof value !== "string") {
    return false;
  }
  const raw = value
    .trim()
    .replace(/^last\s+edited\s*:\s*/i, "")
    .replace(/^updated\s*[:\-]?\s*/i, "")
    .trim()
    .toLowerCase();
  return raw === "today" || raw === "just now" || raw === "less than 1 minute ago";
}

function inferUpdatedAtIsoFromLabel(value: unknown): string | null {
  if (typeof value !== "string") {
    return null;
  }
  const raw = value
    .trim()
    .replace(/^last\s+edited\s*:\s*/i, "")
    .replace(/^updated\s*[:\-]?\s*/i, "")
    .trim()
    .toLowerCase();
  if (!raw) {
    return null;
  }
  const nowMs = Date.now();
  if (raw === "today" || raw === "just now" || raw === "less than 1 minute ago") {
    return null;
  }
  if (raw === "yesterday") {
    return new Date(nowMs - 24 * 60 * 60 * 1000).toISOString();
  }
  const match = raw.match(/^(\d+|an?|one)\s+(second|minute|hour|day|week|month|year)s?(?:\s+ago)?$/i);
  if (!match) {
    return null;
  }
  const amountToken = match[1].toLowerCase();
  const amount = amountToken === "a" || amountToken === "an" || amountToken === "one" ? 1 : Number(amountToken);
  if (!Number.isFinite(amount) || amount <= 0) {
    return null;
  }
  const unit = match[2].toLowerCase();
  const unitMs =
    unit === "second"
      ? 1000
      : unit === "minute"
        ? 60 * 1000
        : unit === "hour"
          ? 60 * 60 * 1000
          : unit === "day"
            ? 24 * 60 * 60 * 1000
            : unit === "week"
              ? 7 * 24 * 60 * 60 * 1000
              : unit === "month"
                ? 30 * 24 * 60 * 60 * 1000
                : 365 * 24 * 60 * 60 * 1000;
  return new Date(nowMs - amount * unitMs).toISOString();
}

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
          const tStartRaw = Number(reel.t_start_sec);
          const tEndRaw = Number(reel.t_end_sec);
          const hasStart = Number.isFinite(tStartRaw) && tStartRaw >= 0;
          const hasEnd = Number.isFinite(tEndRaw) && tEndRaw > 0;
          const tStartSec = hasStart ? tStartRaw : undefined;
          const tEndSec = hasEnd && (!hasStart || tEndRaw > tStartRaw) ? tEndRaw : undefined;
          return {
            id: typeof reel.id === "string" && reel.id.trim() ? reel.id.trim() : `community-reel-${Math.random().toString(36).slice(2, 10)}`,
            platform,
            sourceUrl,
            embedUrl,
            tStartSec,
            tEndSec,
          };
        })
        .filter(Boolean) as CommunitySet["reels"]
    : [];

  const tags = Array.isArray(row.tags)
    ? row.tags.map((tag) => String(tag || "").trim().toLowerCase()).filter(Boolean).slice(0, 6)
    : [];
  const createdAtRaw = typeof row.created_at === "string" ? row.created_at.trim() : row.created_at;
  const createdAtMs = parseApiTimestampMs(createdAtRaw);
  const createdAt = createdAtMs != null ? new Date(createdAtMs).toISOString() : null;
  const updatedAtRaw = typeof row.updated_at === "string" ? row.updated_at.trim() : row.updated_at;
  const updatedAtMs = parseApiTimestampMs(updatedAtRaw);
  const updatedAt = updatedAtMs != null
    ? new Date(updatedAtMs).toISOString()
    : createdAt ?? inferUpdatedAtIsoFromLabel(row.updated_label);
  const rawUpdatedLabel = typeof row.updated_label === "string" && row.updated_label.trim() ? row.updated_label.trim() : "Last Edited: unknown";
  const safeUpdatedLabel = updatedAt == null && isRelativeNowUpdatedLabel(rawUpdatedLabel) ? "Last Edited: unknown" : rawUpdatedLabel;

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
    updatedLabel: safeUpdatedLabel,
    updatedAt,
    createdAt,
    thumbnailUrl: typeof row.thumbnail_url === "string" && row.thumbnail_url.trim() ? row.thumbnail_url.trim() : "/images/community/ai-systems.svg",
    featured: Boolean(row.featured),
  };
}

function normalizeStoredHistoryItem(raw: unknown): StoredHistoryItem | null {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    return null;
  }
  const row = raw as Partial<CommunityHistoryItemApi>;
  const materialId = typeof row.material_id === "string" ? row.material_id.trim() : "";
  const title = typeof row.title === "string" ? row.title.trim() : "";
  if (!materialId || !title) {
    return null;
  }
  return {
    materialId,
    title,
    updatedAt: Math.max(0, Math.floor(Number(row.updated_at) || 0)),
    starred: Boolean(row.starred),
    generationMode: row.generation_mode === "slow" ? "slow" : "fast",
    source: row.source === "community" ? "community" : "search",
    feedQuery: typeof row.feed_query === "string" && row.feed_query.trim() ? row.feed_query.trim() : undefined,
  };
}

export async function fetchCommunityHistory(): Promise<StoredHistoryItem[]> {
  const stored = readCommunityAuthSession();
  if (!stored?.sessionToken) {
    return [];
  }
  const res = await safeFetch(apiUrl("/community/history"), {
    cache: "no-store",
    headers: { ...communitySessionHeaders() },
  });
  const json = await parseJsonResponse<{ items?: unknown[] }>(res);
  const rows = Array.isArray(json?.items) ? json.items : [];
  return rows.map(normalizeStoredHistoryItem).filter(Boolean) as StoredHistoryItem[];
}

export async function replaceCommunityHistory(items: StoredHistoryItem[]): Promise<StoredHistoryItem[]> {
  const stored = readCommunityAuthSession();
  if (!stored?.sessionToken) {
    return items;
  }
  const res = await safeFetch(apiUrl("/community/history"), {
    method: "PUT",
    headers: {
      "Content-Type": "application/json",
      ...communitySessionHeaders(),
    },
    body: JSON.stringify({
      items: items.map((item) => ({
        material_id: item.materialId,
        title: item.title,
        updated_at: item.updatedAt,
        starred: item.starred,
        generation_mode: item.generationMode,
        source: item.source,
        feed_query: item.feedQuery ?? null,
      })),
    }),
  });
  const json = await parseJsonResponse<{ items?: unknown[] }>(res);
  const rows = Array.isArray(json?.items) ? json.items : [];
  return rows.map(normalizeStoredHistoryItem).filter(Boolean) as StoredHistoryItem[];
}

export function queueCommunityHistorySync(items: StoredHistoryItem[]): Promise<void> {
  const snapshot = items.map((item) => ({ ...item }));
  const syncVersion = communityHistorySyncVersion + 1;
  communityHistorySyncVersion = syncVersion;
  communityHistorySyncChain = communityHistorySyncChain
    .catch(() => {
      // Keep the queue alive after a prior failed sync.
    })
    .then(async () => {
      if (syncVersion !== communityHistorySyncVersion) {
        return;
      }
      await replaceCommunityHistory(snapshot);
    });
  return communityHistorySyncChain;
}

export async function fetchCommunitySets(): Promise<CommunitySet[]> {
  const res = await safeFetch(apiUrl("/community/sets"), {
    cache: "no-store",
  });
  const json = await parseJsonResponse<{ sets?: unknown[] }>(res);
  const rows = Array.isArray(json?.sets) ? json.sets : [];
  return rows.map(normalizeCommunitySet).filter(Boolean) as CommunitySet[];
}

export async function fetchOwnedCommunitySets(): Promise<CommunitySet[]> {
  const res = await safeFetch(apiUrl("/community/sets/mine"), {
    cache: "no-store",
    headers: { ...communityRequestHeaders() },
  });
  const json = await parseJsonResponse<{ sets?: unknown[] }>(res);
  const rows = Array.isArray(json?.sets) ? json.sets : [];
  return rows.map(normalizeCommunitySet).filter(Boolean) as CommunitySet[];
}

export async function createCommunitySet(params: {
  title: string;
  description: string;
  tags: string[];
  reels: Array<{
    id?: string;
    platform: CommunityReelPlatform;
    sourceUrl: string;
    embedUrl: string;
    tStartSec?: number;
    tEndSec?: number;
  }>;
  thumbnailUrl: string;
  curator?: string;
}): Promise<CommunitySet> {
  const res = await safeFetch(apiUrl("/community/sets"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...communityRequestHeaders(),
    },
    body: JSON.stringify({
      title: params.title,
      description: params.description,
      tags: params.tags,
      reels: params.reels.map((reel) => ({
        platform: reel.platform,
        source_url: reel.sourceUrl,
        embed_url: reel.embedUrl,
        ...(Number.isFinite(reel.tStartSec) ? { t_start_sec: Number(reel.tStartSec) } : {}),
        ...(Number.isFinite(reel.tEndSec) ? { t_end_sec: Number(reel.tEndSec) } : {}),
      })),
      thumbnail_url: params.thumbnailUrl,
      curator: params.curator,
    }),
  });

  const created = normalizeCommunitySet(await parseJsonResponse<unknown>(res));
  if (!created) {
    throw new Error("Backend returned an invalid community set payload.");
  }
  return created;
}

export async function updateCommunitySet(params: {
  setId: string;
  title: string;
  description: string;
  tags: string[];
  reels: Array<{
    id?: string;
    platform: CommunityReelPlatform;
    sourceUrl: string;
    embedUrl: string;
    tStartSec?: number;
    tEndSec?: number;
  }>;
  thumbnailUrl: string;
  curator?: string;
}): Promise<CommunitySet> {
  const setId = params.setId.trim();
  if (!setId) {
    throw new Error("setId is required.");
  }

  const res = await safeFetch(apiUrl(`/community/sets/${encodeURIComponent(setId)}`), {
    method: "PUT",
    headers: {
      "Content-Type": "application/json",
      ...communityRequestHeaders(),
    },
    body: JSON.stringify({
      title: params.title,
      description: params.description,
      tags: params.tags,
      reels: params.reels.map((reel) => ({
        ...(typeof reel.id === "string" && reel.id.trim() ? { id: reel.id.trim() } : {}),
        platform: reel.platform,
        source_url: reel.sourceUrl,
        embed_url: reel.embedUrl,
        ...(Number.isFinite(reel.tStartSec) ? { t_start_sec: Number(reel.tStartSec) } : {}),
        ...(Number.isFinite(reel.tEndSec) ? { t_end_sec: Number(reel.tEndSec) } : {}),
      })),
      thumbnail_url: params.thumbnailUrl,
      curator: params.curator,
    }),
  });

  const updated = normalizeCommunitySet(await parseJsonResponse<unknown>(res));
  if (!updated) {
    throw new Error("Backend returned an invalid community set payload.");
  }
  return updated;
}

export async function deleteCommunitySet(params: { setId: string }): Promise<void> {
  const setId = params.setId.trim();
  if (!setId) {
    throw new Error("setId is required.");
  }
  const encodedSetId = encodeURIComponent(setId);
  try {
    await safeFetch(apiUrl(`/community/sets/${encodedSetId}`), {
      method: "DELETE",
      headers: communityRequestHeaders(),
    });
    return;
  } catch (error) {
    const message = error instanceof Error ? error.message : "";
    const isMethodNotAllowed = /method not allowed/i.test(message) || message.includes("(405)");
    if (!isMethodNotAllowed) {
      throw error;
    }
  }

  await safeFetch(apiUrl(`/community/sets/${encodedSetId}/delete`), {
    method: "POST",
    headers: communityRequestHeaders(),
  });
}

export async function fetchCommunityReelDuration(params: {
  sourceUrl: string;
}): Promise<number | null> {
  const sourceUrl = params.sourceUrl.trim();
  if (!sourceUrl) {
    return null;
  }
  const query = new URLSearchParams({ source_url: sourceUrl });
  const res = await safeFetch(`${apiUrl("/community/reels/duration")}?${query.toString()}`, {
    cache: "no-store",
  });
  const json = await parseJsonResponse<{ duration_sec?: unknown }>(res);
  const durationRaw = Number(json?.duration_sec);
  if (!Number.isFinite(durationRaw) || durationRaw <= 0) {
    return null;
  }
  return durationRaw;
}

type SafeFetchInit = RequestInit & {
  timeoutMs?: number;
};

async function safeFetch(url: string, init?: SafeFetchInit): Promise<Response> {
  const timeoutRaw = Number(init?.timeoutMs);
  const timeoutMs = Number.isFinite(timeoutRaw) && timeoutRaw > 0 ? Math.max(1_000, timeoutRaw) : null;
  const upstreamSignal = init?.signal;
  const controller = new AbortController();
  const timeoutId = timeoutMs ? setTimeout(() => controller.abort(), timeoutMs) : null;
  const onUpstreamAbort = () => controller.abort();

  if (upstreamSignal) {
    if (upstreamSignal.aborted) {
      controller.abort();
    } else {
      upstreamSignal.addEventListener("abort", onUpstreamAbort, { once: true });
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
      throw new Error("Request was interrupted.");
    }
    throw new Error(BACKEND_DOWN_ERROR);
  } finally {
    if (timeoutId) {
      clearTimeout(timeoutId);
    }
    if (upstreamSignal) {
      upstreamSignal.removeEventListener("abort", onUpstreamAbort);
    }
  }

  if (!response.ok) {
    throw await buildApiError(response);
  }
  return response;
}

class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function buildApiError(response: Response): Promise<ApiError> {
  let message: string;
  try {
    const json = await response.json();
    message = json?.detail || json?.message || `Request failed (${response.status})`;
  } catch {
    message = `Request failed (${response.status})`;
  }
  return new ApiError(message, response.status);
}

export function isSessionExpiredError(error: unknown): boolean {
  if (error instanceof ApiError && error.status === 401) {
    return true;
  }
  if (error instanceof Error) {
    return /session expired|sign in/i.test(error.message);
  }
  return false;
}
