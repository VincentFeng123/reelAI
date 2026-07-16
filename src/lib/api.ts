import type {
  AssessmentAnswerResponse,
  AssessmentSnoozeResponse,
  AssessmentStatusResponse,
  ChatMessage,
  ChatResponse,
  CommunityAccount,
  CommunityAuthSession,
  CommunitySet,
  CommunityReelPlatform,
  FeedResponse,
  GenerationBatchTerminalStatus,
  GenerationJobCancelResponse,
  GenerationJobStatusResponse,
  GenerationQueuedResponse,
  GenerationTerminalStatus,
  IngestResult,
  IngestSearchRequest,
  IngestSearchResult,
  IngestUrlRequest,
  MaterialResponse,
  Reel,
  ReelProgressResponse,
  ReelScrollResponse,
  ReelsCanGenerateAnyResponse,
  ReelsCanGenerateResponse,
  ReelsGenerateResponse,
  ReelsGenerateSubmission,
  ReelsGenerateStreamEvent,
  TypedApiError,
} from "@/lib/types";
import type { StoredHistoryItem } from "@/lib/historyStorage";
import {
  normalizeStudyReelsSettings,
  readStudyReelsSettings,
  setActiveStudyReelsSettingsScope,
  type PreferredVideoDuration,
  type StudyReelsSettings,
} from "@/lib/settings";

const RAW_API_BASE = (process.env.NEXT_PUBLIC_API_BASE || "").replace(/\/$/, "");
const RAW_DEPLOYED_API_BASE = (process.env.NEXT_PUBLIC_DEPLOYED_API_BASE || "").replace(/\/$/, "");
const DEFAULT_DEPLOYED_API_BASE = "https://reelai-production.up.railway.app";

let _cachedApiBase: string | null = null;
function resolveApiBase(): string {
  if (_cachedApiBase !== null) {
    return _cachedApiBase;
  }
  let base: string;
  if (RAW_API_BASE) {
    base = RAW_API_BASE;
  } else if (process.env.NODE_ENV === "development") {
    base = "http://127.0.0.1:8000";
  } else if (typeof window !== "undefined") {
    const hostname = window.location.hostname.trim().toLowerCase();
    if (hostname && hostname !== "localhost" && hostname !== "127.0.0.1") {
      base = RAW_DEPLOYED_API_BASE || DEFAULT_DEPLOYED_API_BASE;
    } else {
      base = RAW_DEPLOYED_API_BASE;
    }
  } else {
    base = RAW_DEPLOYED_API_BASE;
  }
  _cachedApiBase = base.replace(/\/$/, "");
  return _cachedApiBase;
}

function getResolvedApiBase(): string {
  return resolveApiBase();
}
function getBackendDownError(): string {
  const base = getResolvedApiBase();
  return base
    ? `Cannot reach backend at ${base}. Make sure the backend server is running.`
    : "Cannot reach backend. Check your deployment and API routes.";
}
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
  const base = getResolvedApiBase();
  if (!base) {
    return `/api${cleanPath}`;
  }
  const cleanBase = base.replace(/\/$/, "");
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
  // WARNING: Math.random() is not cryptographically secure.
  if (typeof console !== "undefined" && console.warn) {
    console.warn("StudyReels: crypto API unavailable, falling back to Math.random for owner key generation. This is not cryptographically secure.");
  }
  const segments: string[] = [];
  for (let i = 0; i < 8; i++) {
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
    window.localStorage.setItem(COMMUNITY_OWNER_KEY_STORAGE_KEY, ownerKey);
    window.sessionStorage.removeItem(COMMUNITY_OWNER_KEY_STORAGE_KEY);
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
    const existingLocalKey = normalizeStoredCommunityOwnerKey(
      window.localStorage.getItem(COMMUNITY_OWNER_KEY_STORAGE_KEY),
    );
    if (existingLocalKey) {
      communityOwnerKeyMemoryFallback = existingLocalKey;
      return existingLocalKey;
    }
    const legacySessionKey = normalizeStoredCommunityOwnerKey(
      window.sessionStorage.getItem(COMMUNITY_OWNER_KEY_STORAGE_KEY),
    );
    if (legacySessionKey) {
      persistCommunityOwnerKey(legacySessionKey);
      return legacySessionKey;
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

function readCommunitySessionToken(): string {
  if (typeof window === "undefined") {
    return "";
  }
  try {
    let sessionToken = (window.sessionStorage.getItem(COMMUNITY_SESSION_TOKEN_STORAGE_KEY) || "").trim();
    if (sessionToken.length < 24) {
      const legacyToken = (window.localStorage.getItem(COMMUNITY_SESSION_TOKEN_STORAGE_KEY) || "").trim();
      if (legacyToken.length >= 24) {
        sessionToken = legacyToken;
        window.sessionStorage.setItem(COMMUNITY_SESSION_TOKEN_STORAGE_KEY, sessionToken);
        window.localStorage.removeItem(COMMUNITY_SESSION_TOKEN_STORAGE_KEY);
      }
    }
    return sessionToken;
  } catch {
    return "";
  }
}

function communitySessionHeaders(): HeadersInit {
  const sessionToken = readCommunitySessionToken();
  return sessionToken ? { [COMMUNITY_SESSION_HEADER]: sessionToken } : {};
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
  account: CommunityAccount | null;
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
      setActiveStudyReelsSettingsScope(null);
      return;
    }
    // Account metadata (non-secret) stays in localStorage for cross-tab display.
    // The session token goes in sessionStorage to limit XSS blast radius — an
    // attacker would need code running in the same tab to read it.
    if (session.account) {
      window.localStorage.setItem(COMMUNITY_ACCOUNT_STORAGE_KEY, JSON.stringify(session.account));
    } else {
      window.localStorage.removeItem(COMMUNITY_ACCOUNT_STORAGE_KEY);
    }
    window.sessionStorage.setItem(COMMUNITY_SESSION_TOKEN_STORAGE_KEY, session.sessionToken);
    setActiveStudyReelsSettingsScope(session.account?.id ?? null);
  } catch {
    // Ignore storage failures and keep the in-memory response path usable.
    setActiveStudyReelsSettingsScope(session?.account?.id ?? null);
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
    const sessionToken = readCommunitySessionToken();
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

function clearOwnedCommunitySetIds(): void {
  if (typeof window === "undefined") {
    return;
  }
  try {
    window.localStorage.removeItem(COMMUNITY_OWNED_SET_IDS_STORAGE_KEY);
  } catch {
    // Ignore storage failures during account teardown.
  }
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
  } catch (error) {
    if (isTransportError(error) || isRequestInterruptedError(error)) {
      throw error;
    }
    throw new Error("Backend returned an invalid JSON response.");
  }
}

export async function sendCommunitySignupVerification(params: {
  email: string;
  username?: string;
}): Promise<{ email: string; verified: boolean; verificationRequired: boolean; verificationCodeDebug: string | null }> {
  const res = await safeFetch(apiUrl("/community/auth/send-signup-verification"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...communityOwnerHeaders(),
    },
    body: JSON.stringify({
      email: params.email,
      username: params.username,
    }),
  });
  const json = await parseJsonResponse<{
    email?: unknown;
    verified?: unknown;
    verification_required?: unknown;
    verification_code_debug?: unknown;
  }>(res);
  const email = typeof json?.email === "string" && json.email.trim() ? json.email.trim() : "";
  if (!email) {
    throw new Error("Backend returned an invalid verification response.");
  }
  return {
    email,
    verified: Boolean(json?.verified),
    verificationRequired: Boolean(json?.verification_required),
    verificationCodeDebug:
      typeof json?.verification_code_debug === "string" && json.verification_code_debug.trim()
        ? json.verification_code_debug.trim()
        : null,
  };
}

export async function verifyCommunitySignupEmail(params: {
  email: string;
  code: string;
}): Promise<{ email: string; verified: boolean }> {
  const res = await safeFetch(apiUrl("/community/auth/verify-signup-email"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...communityOwnerHeaders(),
    },
    body: JSON.stringify({
      email: params.email,
      code: params.code,
    }),
  });
  const json = await parseJsonResponse<{ email?: unknown; verified?: unknown }>(res);
  const email = typeof json?.email === "string" && json.email.trim() ? json.email.trim() : "";
  if (!email) {
    throw new Error("Backend returned an invalid verification response.");
  }
  return {
    email,
    verified: Boolean(json?.verified),
  };
}

export async function registerCommunityAccount(params: {
  username: string;
  email: string;
  password: string;
  persistAccountWhenUnverified?: boolean;
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
    account:
      params.persistAccountWhenUnverified === false && (session.verificationRequired || session.account.isVerified === false)
        ? null
        : session.account,
    sessionToken: session.sessionToken,
  });
  return session;
}

export async function loginCommunityAccount(params: {
  username: string;
  password: string;
  persistAccountWhenUnverified?: boolean;
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
    account:
      params.persistAccountWhenUnverified === false && (session.verificationRequired || session.account.isVerified === false)
        ? null
        : session.account,
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
  const sessionToken = readCommunitySessionToken();
  if (sessionToken.length < 24) {
    throw new Error("Session expired. Sign in again.");
  }
  saveCommunityAuthSessionStorage({
    account,
    sessionToken,
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
  const sessionToken = readCommunitySessionToken();
  if (sessionToken.length < 24) {
    throw new Error("Session expired. Sign in again.");
  }
  saveCommunityAuthSessionStorage({
    account,
    sessionToken,
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
  const sessionToken = readCommunitySessionToken();
  if (sessionToken.length < 24) {
    throw new Error("Session expired. Sign in again.");
  }
  saveCommunityAuthSessionStorage({
    account,
    sessionToken,
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

export async function deleteCommunityAccount(params: {
  currentPassword: string;
}): Promise<void> {
  try {
    await safeFetch(apiUrl("/community/auth/delete-account"), {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...communitySessionHeaders(),
      },
      body: JSON.stringify({
        current_password: params.currentPassword,
      }),
    });
  } catch (error) {
    if (error instanceof Error && /session expired|sign in/i.test(error.message)) {
      clearOwnedCommunitySetIds();
      clearCommunityAuthSession();
    }
    throw error;
  }
  clearOwnedCommunitySetIds();
  clearCommunityAuthSession();
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
  knowledgeLevel?: "beginner" | "intermediate" | "advanced";
  signal?: AbortSignal;
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
  if (params.knowledgeLevel) {
    form.append("knowledge_level", params.knowledgeLevel);
  }

  const res = await safeFetch(apiUrl("/material"), {
    method: "POST",
    headers: {
      ...communityRequestHeaders(),
    },
    body: form,
    signal: params.signal,
  });
  return parseJsonResponse<MaterialResponse>(res);
}

export async function updateMaterialLevel(params: {
  materialId: string;
  knowledgeLevel: "beginner" | "intermediate" | "advanced";
}): Promise<{ knowledge_level: string; effective_level_target: number }> {
  const res = await safeFetch(apiUrl(`/materials/${encodeURIComponent(params.materialId)}/level`), {
    method: "PATCH",
    headers: {
      "Content-Type": "application/json",
      ...communityRequestHeaders(),
    },
    body: JSON.stringify({ knowledge_level: params.knowledgeLevel }),
  });
  return parseJsonResponse<{ knowledge_level: string; effective_level_target: number }>(res);
}

type GenerateReelsParams = {
  materialId: string;
  numReels?: number;
  conceptId?: string;
  continuationToken?: string;
  generationMode?: "slow" | "fast";
  minRelevance?: number;
  creativeCommonsOnly?: boolean;
  preferredVideoDuration?: PreferredVideoDuration;
  targetClipDurationSec?: number;
  targetClipDurationMinSec?: number;
  targetClipDurationMaxSec?: number;
  signal?: AbortSignal;
};

// Max number of IDs we send on any request. After a long session the client's
// seen-source-video list can grow to hundreds; joining all of them into a
// comma-separated query string risks a Request-URI-Too-Large response from some
// proxies. Keep the most recent ones (they're the likeliest duplicates).
const MAX_EXCLUDED_VIDEO_IDS = 250;
const MAX_EXCLUDED_REEL_IDS = 200;
// Any "video_id" longer than this is almost certainly a full URL (non-YouTube
// community reel). The backend can't match on URLs, so skip them.
const VIDEO_ID_MAX_LEN = 32;
const VIDEO_ID_RE = /^[A-Za-z0-9_-]+$/;

function normalizeVideoIdList(values: string[] | undefined): string[] {
  if (!Array.isArray(values)) {
    return [];
  }
  const seen = new Set<string>();
  const normalized: string[] = [];
  for (const rawValue of values) {
    const clean = String(rawValue || "").trim();
    if (!clean || clean.length > VIDEO_ID_MAX_LEN || !VIDEO_ID_RE.test(clean)) {
      continue;
    }
    if (seen.has(clean)) {
      continue;
    }
    seen.add(clean);
    normalized.push(clean);
  }
  return normalized.length > MAX_EXCLUDED_VIDEO_IDS
    ? normalized.slice(normalized.length - MAX_EXCLUDED_VIDEO_IDS)
    : normalized;
}

function normalizeReelIdList(values: string[] | undefined): string[] {
  if (!Array.isArray(values)) {
    return [];
  }
  const seen = new Set<string>();
  const normalized: string[] = [];
  for (const rawValue of values) {
    const clean = String(rawValue || "").trim();
    if (!clean || clean.includes(",") || seen.has(clean)) {
      continue;
    }
    seen.add(clean);
    normalized.push(clean);
  }
  return normalized.length > MAX_EXCLUDED_REEL_IDS
    ? normalized.slice(normalized.length - MAX_EXCLUDED_REEL_IDS)
    : normalized;
}

function generationModeCeiling(_mode: "slow" | "fast" | undefined): number {
  return 9;
}

function requestedReelCount(params: Pick<GenerateReelsParams, "generationMode" | "numReels">): number {
  const ceiling = generationModeCeiling(params.generationMode);
  const requested = params.numReels;
  if (typeof requested !== "number" || !Number.isFinite(requested)) {
    return ceiling;
  }
  return Math.min(ceiling, Math.max(1, Math.trunc(requested)));
}

function buildGenerateReelsRequestBody(params: GenerateReelsParams): Record<string, unknown> {
  return {
    material_id: params.materialId,
    concept_id: params.conceptId,
    num_reels: requestedReelCount(params),
    continuation_token: String(params.continuationToken || "").trim() || undefined,
    creative_commons_only: params.creativeCommonsOnly === true,
    generation_mode: params.generationMode ?? "slow",
    min_relevance: Number.isFinite(params.minRelevance) ? params.minRelevance : undefined,
    preferred_video_duration: params.preferredVideoDuration ?? "any",
  };
}

export async function generateReels(params: GenerateReelsParams): Promise<ReelsGenerateSubmission> {
  const res = await safeFetch(apiUrl("/reels/generate"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      // Backend requires an owner-key identity header on reels endpoints to
      // rate-limit per-device and block anonymous API scraping.
      ...communityRequestHeaders(),
    },
    body: JSON.stringify(buildGenerateReelsRequestBody(params)),
    signal: params.signal,
    timeoutMs: 30_000,
    keepSignalAliveThroughBody: true,
  });

  return parseJsonResponse<ReelsGenerateSubmission>(res);
}

// Maps the web app's settings-friendly camelCase to the snake_case the backend expects.
/**
 * POST /api/ingest/url — ingests a canonical YouTube video URL using hosted
 * timestamped transcript cues. Native captions are preferred but
 * optional; no media download or local Whisper runtime is used.
 */
type IngestUrlParams = {
  sourceUrl: string;
  materialId?: string | null;
  conceptId?: string | null;
  targetClipDurationSec?: number;
  targetClipDurationMinSec?: number;
  targetClipDurationMaxSec?: number;
  language?: string;
  signal?: AbortSignal;
};

type IngestSearchParams = {
  query: string;
  maxPerPlatform?: number;
  materialId?: string | null;
  conceptId?: string | null;
  targetClipDurationSec?: number;
  targetClipDurationMinSec?: number;
  targetClipDurationMaxSec?: number;
  language?: string;
  excludeVideoIds?: string[];
  signal?: AbortSignal;
};

/**
 * POST /api/ingest/search — YouTube-only discovery using hosted timestamped
 * transcript cues. Native captions are preferred but optional; no media download
 * or local Whisper runtime is used.
 */
export async function ingestSearch(params: IngestSearchParams): Promise<IngestSearchResult> {
  const body: IngestSearchRequest = {
    query: params.query,
    max_per_platform: Number.isFinite(params.maxPerPlatform)
      ? Math.round(params.maxPerPlatform as number)
      : undefined,
    material_id: params.materialId ?? undefined,
    concept_id: params.conceptId ?? undefined,
    language: params.language,
    exclude_video_ids: params.excludeVideoIds,
  };

  const res = await safeFetch(apiUrl("/ingest/search"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
    signal: params.signal,
    timeoutMs: 300_000,
    keepSignalAliveThroughBody: true,
  });

  return parseJsonResponse<IngestSearchResult>(res);
}

export async function ingestUrl(params: IngestUrlParams): Promise<IngestResult> {
  const body: IngestUrlRequest = {
    source_url: params.sourceUrl,
    material_id: params.materialId ?? undefined,
    concept_id: params.conceptId ?? undefined,
    language: params.language,
  };

  const res = await safeFetch(apiUrl("/ingest/url"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
    signal: params.signal,
    timeoutMs: 180_000,
    keepSignalAliveThroughBody: true,
  });

  return parseJsonResponse<IngestResult>(res);
}

function isQueuedGeneration(value: ReelsGenerateSubmission): value is GenerationQueuedResponse {
  return "job_id" in value && typeof value.job_id === "string" && value.job_id.trim().length > 0;
}

function finalResponseFromEvent(event: Extract<ReelsGenerateStreamEvent, { type: "final" }>): ReelsGenerateResponse {
  return {
    reels: Array.isArray(event.payload.reels) ? event.payload.reels : [],
    generation_id: event.payload.generation_id,
    response_profile: "unified",
  };
}

function terminalResponseFromStatus(status: GenerationJobStatusResponse): ReelsGenerateResponse | null {
  if (!Array.isArray(status.reels)) {
    return null;
  }
  return {
    reels: status.reels,
    generation_id: status.result_generation_id,
    response_profile: "unified",
    model_used: status.model_used,
    quality_degraded: status.quality_degraded,
  };
}

function withDurableBatchMetadata(
  response: ReelsGenerateResponse,
  jobId: string,
  terminalStatus: GenerationBatchTerminalStatus,
): ReelsGenerateResponse {
  return {
    ...response,
    batch_id: jobId,
    batch_size: response.reels.length,
    continuation_token: jobId,
    terminal_status: terminalStatus,
  };
}

function throwTerminalGenerationError(error: TypedApiError | null | undefined, status: string): never {
  const fallbackMessage = status === "cancelled" ? "Generation was cancelled." : `Generation ended with status ${status}.`;
  throw new ApiError(error?.message || fallbackMessage, 409, error?.code || `generation_${status}`, error ?? null);
}

async function waitForReconnect(signal?: AbortSignal): Promise<void> {
  if (signal?.aborted) {
    throw new Error("Request was interrupted.");
  }
  await new Promise<void>((resolve, reject) => {
    const timer = window.setTimeout(resolve, 400);
    const onAbort = () => {
      window.clearTimeout(timer);
      reject(new Error("Request was interrupted."));
    };
    signal?.addEventListener("abort", onAbort, { once: true });
    window.setTimeout(() => signal?.removeEventListener("abort", onAbort), 401);
  });
}

export async function fetchGenerationStatus(
  jobId: string,
  options?: { signal?: AbortSignal; timeoutMs?: number },
): Promise<GenerationJobStatusResponse> {
  const res = await safeFetch(apiUrl(`/reels/generation-status/${encodeURIComponent(jobId)}`), {
    cache: "no-store",
    headers: { ...communityRequestHeaders() },
    signal: options?.signal,
    timeoutMs: options?.timeoutMs ?? 30_000,
    keepSignalAliveThroughBody: true,
  });
  return parseJsonResponse<GenerationJobStatusResponse>(res);
}

export async function cancelGenerationJob(
  jobId: string,
  options?: { signal?: AbortSignal },
): Promise<GenerationJobCancelResponse> {
  const res = await safeFetch(apiUrl(`/reels/generation-jobs/${encodeURIComponent(jobId)}/cancel`), {
    method: "POST",
    headers: { ...communityRequestHeaders() },
    signal: options?.signal,
  });
  return parseJsonResponse<GenerationJobCancelResponse>(res);
}

async function consumeGenerationJob(
  job: GenerationQueuedResponse,
  options: {
    signal?: AbortSignal;
    onCandidate?: (reel: Reel) => void;
    onReconnect?: (consecutiveIdleWindows: number) => void;
    onActivity?: () => void;
    onTerminal?: (status: GenerationTerminalStatus) => void;
    idleTimeoutMs?: number;
  },
): Promise<ReelsGenerateResponse> {
  let afterSeq = 0;
  let finalResponse: ReelsGenerateResponse | null = null;
  let consecutiveIdleWindows = 0;
  const idleTimeoutMs = Math.max(10, options.idleTimeoutMs ?? 35_000);
  // Backend permits an eight-minute queue window plus a one-hour quality-first
  // execution window. Keep one minute of transport slack, but stay finite.
  const deadline = Date.now() + 69 * 60_000;

  while (Date.now() < deadline) {
    const remainingMs = Math.max(1_000, deadline - Date.now());
    let terminalStatus: string | null = null;
    let terminalError: TypedApiError | null | undefined;
    try {
      const query = new URLSearchParams({ after_seq: String(afterSeq) });
      const res = await safeFetch(
        `${apiUrl(`/reels/generation-stream/${encodeURIComponent(job.job_id)}`)}?${query.toString()}`,
        {
          cache: "no-store",
          headers: {
            Accept: "application/x-ndjson",
            ...communityRequestHeaders(),
          },
          signal: options.signal,
          timeoutMs: Math.min(540_000, remainingMs),
          keepSignalAliveThroughBody: true,
        },
      );
      if (!res.body) {
        throw new TransportError("Generation stream returned no response body.");
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      const processLine = (line: string) => {
        const trimmed = line.trim();
        if (!trimmed) {
          return;
        }
        const event = JSON.parse(trimmed) as ReelsGenerateStreamEvent;
        if (event.job_id !== job.job_id || !Number.isInteger(event.seq) || event.seq <= afterSeq) {
          return;
        }
        consecutiveIdleWindows = 0;
        options.onActivity?.();
        afterSeq = event.seq;
        if (event.type === "candidate") {
          options.onCandidate?.(event.payload.reel);
        } else if (event.type === "final") {
          finalResponse = finalResponseFromEvent(event);
        } else {
          terminalStatus = event.payload.status;
          terminalError = event.payload.error;
        }
      };

      let readerDone = false;
      try {
        while (!terminalStatus) {
          let idleTimer: ReturnType<typeof setTimeout> | null = null;
          const idleTimeout = new Promise<never>((_, reject) => {
            idleTimer = setTimeout(() => {
              reject(new TransportError("Generation stream was idle; reconnecting to the durable job."));
            }, idleTimeoutMs);
          });
          let chunk: ReadableStreamReadResult<Uint8Array>;
          try {
            chunk = await Promise.race([reader.read(), idleTimeout]);
          } finally {
            if (idleTimer) {
              clearTimeout(idleTimer);
            }
          }
          const { done, value } = chunk;
          if (done) {
            readerDone = true;
            break;
          }
          buffer += decoder.decode(value, { stream: true });
          let newlineIndex = buffer.indexOf("\n");
          while (newlineIndex >= 0) {
            processLine(buffer.slice(0, newlineIndex));
            buffer = buffer.slice(newlineIndex + 1);
            newlineIndex = buffer.indexOf("\n");
          }
        }
        buffer += decoder.decode();
        if (buffer.trim()) {
          processLine(buffer);
        }
      } finally {
        if (!readerDone) {
          try {
            await reader.cancel();
          } catch {
            // The underlying request may already have failed or been aborted.
          }
        }
        reader.releaseLock();
      }
    } catch (error) {
      if (isRequestInterruptedError(error) || options.signal?.aborted) {
        throw error;
      }
      if (!isTransportError(error)) {
        throw error;
      }
      if (error instanceof TransportError && /stream was idle/i.test(error.message)) {
        consecutiveIdleWindows += 1;
        options.onReconnect?.(consecutiveIdleWindows);
      }
    }

    if (terminalStatus) {
      options.onTerminal?.(terminalStatus as GenerationTerminalStatus);
      if (terminalStatus === "failed" || terminalStatus === "cancelled") {
        throwTerminalGenerationError(terminalError, terminalStatus);
      }
      if (
        finalResponse
        && (terminalStatus === "completed" || terminalStatus === "partial" || terminalStatus === "exhausted")
      ) {
        return withDurableBatchMetadata(finalResponse, job.job_id, terminalStatus);
      }
    }

    let status: GenerationJobStatusResponse;
    try {
      status = await fetchGenerationStatus(job.job_id, {
        signal: options.signal,
        timeoutMs: Math.min(30_000, Math.max(1_000, deadline - Date.now())),
      });
    } catch (error) {
      if (isRequestInterruptedError(error) || options.signal?.aborted) {
        throw error;
      }
      if (!isTransportError(error)) {
        throw error;
      }
      await waitForReconnect(options.signal);
      continue;
    }
    if (status.status === "failed" || status.status === "cancelled") {
      options.onTerminal?.(status.status);
      throwTerminalGenerationError(status.error, status.status);
    }
    if (status.status === "completed" || status.status === "partial" || status.status === "exhausted") {
      options.onTerminal?.(status.status);
      const response = finalResponse ?? terminalResponseFromStatus(status);
      if (response) {
        return withDurableBatchMetadata(response, job.job_id, status.status);
      }
      throw new ApiError("Generation completed without a final inventory.", 502, "generation_missing_final");
    }
    await waitForReconnect(options.signal);
  }

  throw new TransportError("Generation did not finish before the job deadline.");
}

export async function generateReelsStream(
  params: GenerateReelsParams & {
    signal?: AbortSignal;
    onCandidate?: (reel: Reel) => void;
    generationJobId?: string | null;
    onReconnect?: (consecutiveIdleWindows: number) => void;
    onActivity?: () => void;
    onTerminal?: (status: GenerationTerminalStatus) => void;
    idleTimeoutMs?: number;
  },
): Promise<ReelsGenerateResponse> {
  const resumeJobId = String(params.generationJobId || "").trim();
  const submission: ReelsGenerateSubmission = resumeJobId
    ? {
        job_id: resumeJobId,
        status: "running",
        status_url: `/reels/generation-status/${encodeURIComponent(resumeJobId)}`,
        stream_url: `/reels/generation-stream/${encodeURIComponent(resumeJobId)}`,
      }
    : await generateReels(params);
  if (!isQueuedGeneration(submission)) {
    return submission;
  }
  return consumeGenerationJob(submission, {
    signal: params.signal,
    onCandidate: params.onCandidate,
    onReconnect: params.onReconnect,
    onActivity: params.onActivity,
    onTerminal: params.onTerminal,
    idleTimeoutMs: params.idleTimeoutMs,
  });
}

export async function checkReelsCanGenerate(params: {
  materialId: string;
  conceptId?: string;
  generationMode?: "slow" | "fast";
  minRelevance?: number;
  creativeCommonsOnly?: boolean;
  preferredVideoDuration?: PreferredVideoDuration;
  targetClipDurationSec?: number;
  targetClipDurationMinSec?: number;
  targetClipDurationMaxSec?: number;
}): Promise<ReelsCanGenerateResponse> {
  const res = await safeFetch(apiUrl("/reels/can-generate"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...communityRequestHeaders(),
    },
    body: JSON.stringify({
      material_id: params.materialId,
      concept_id: params.conceptId,
      num_reels: generationModeCeiling(params.generationMode),
      creative_commons_only: params.creativeCommonsOnly === true,
      generation_mode: params.generationMode ?? "slow",
      min_relevance: Number.isFinite(params.minRelevance) ? params.minRelevance : undefined,
      preferred_video_duration: params.preferredVideoDuration ?? "any",
    }),
  });
  return parseJsonResponse<ReelsCanGenerateResponse>(res);
}

export async function checkReelsCanGenerateAny(params: {
  materialIds?: string[];
  generationMode?: "slow" | "fast";
  minRelevance?: number;
  creativeCommonsOnly?: boolean;
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
      ...communityRequestHeaders(),
    },
    body: JSON.stringify({
      material_ids: materialIds,
      creative_commons_only: params.creativeCommonsOnly === true,
      generation_mode: params.generationMode ?? "slow",
      min_relevance: Number.isFinite(params.minRelevance) ? params.minRelevance : undefined,
      preferred_video_duration: params.preferredVideoDuration ?? "any",
    }),
  });
  return parseJsonResponse<ReelsCanGenerateAnyResponse>(res);
}

export async function fetchFeed(params: {
  materialId: string;
  page: number;
  limit: number;
  excludeVideoIds?: string[];
  excludeReelIds?: string[];
  prefetch?: number;
  autofill?: boolean;
  generationMode?: "slow" | "fast";
  minRelevance?: number;
  creativeCommonsOnly?: boolean;
  preferredVideoDuration?: PreferredVideoDuration;
  targetClipDurationSec?: number;
  targetClipDurationMinSec?: number;
  targetClipDurationMaxSec?: number;
  signal?: AbortSignal;
}): Promise<FeedResponse> {
  const query = new URLSearchParams({
    material_id: params.materialId,
    page: String(params.page),
    limit: String(params.limit),
    autofill: String(params.autofill ?? true),
    prefetch: String(params.prefetch ?? 9),
    creative_commons_only: String(params.creativeCommonsOnly === true),
    generation_mode: params.generationMode ?? "slow",
    preferred_video_duration: params.preferredVideoDuration ?? "any",
  });
  if (Number.isFinite(params.minRelevance)) {
    query.set("min_relevance", String(params.minRelevance));
  }
  const excludeVideoIds = normalizeVideoIdList(params.excludeVideoIds);
  if (excludeVideoIds.length > 0) {
    query.set("exclude_video_ids", excludeVideoIds.join(","));
  }
  const excludeReelIds = normalizeReelIdList(params.excludeReelIds);
  if (excludeReelIds.length > 0) {
    query.set("exclude_reel_ids", excludeReelIds.join(","));
  }

  const res = await safeFetch(`${apiUrl("/feed")}?${query}`, {
    cache: "no-store",
    headers: {
      ...communityRequestHeaders(),
    },
    signal: params.signal,
    timeoutMs: 300_000,
    keepSignalAliveThroughBody: true,
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
      ...communityRequestHeaders(),
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
  signal?: AbortSignal;
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
    signal: params.signal,
  });
  return parseJsonResponse<ChatResponse>(res);
}

export async function reportReelProgress(params: {
  reelId: string;
  maxFraction: number;
  signal?: AbortSignal;
}): Promise<ReelProgressResponse> {
  const maxFraction = Math.max(0, Math.min(1, Number(params.maxFraction) || 0));
  const res = await safeFetch(apiUrl(`/reels/${encodeURIComponent(params.reelId)}/progress`), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...communityRequestHeaders(),
    },
    body: JSON.stringify({ max_fraction: maxFraction }),
    signal: params.signal,
  });
  return parseJsonResponse<ReelProgressResponse>(res);
}

export async function reportReelScroll(params: {
  reelId: string;
  signal?: AbortSignal;
}): Promise<ReelScrollResponse> {
  const res = await safeFetch(apiUrl(`/reels/${encodeURIComponent(params.reelId)}/scroll`), {
    method: "POST",
    headers: {
      ...communityRequestHeaders(),
    },
    signal: params.signal,
  });
  return parseJsonResponse<ReelScrollResponse>(res);
}

export async function fetchPendingAssessment(params: {
  materialId: string;
  signal?: AbortSignal;
}): Promise<AssessmentStatusResponse> {
  const query = new URLSearchParams({ material_id: params.materialId });
  const res = await safeFetch(`${apiUrl("/assessments/pending")}?${query.toString()}`, {
    cache: "no-store",
    headers: {
      ...communityRequestHeaders(),
    },
    signal: params.signal,
  });
  return parseJsonResponse<AssessmentStatusResponse>(res);
}

export async function startNextAssessment(params: {
  materialId: string;
  signal?: AbortSignal;
}): Promise<AssessmentStatusResponse> {
  const res = await safeFetch(apiUrl("/assessments/next"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...communityRequestHeaders(),
    },
    body: JSON.stringify({ material_id: params.materialId }),
    signal: params.signal,
  });
  return parseJsonResponse<AssessmentStatusResponse>(res);
}

export async function answerAssessmentQuestion(params: {
  sessionId: string;
  questionId: string;
  choiceIndex: number;
  signal?: AbortSignal;
}): Promise<AssessmentAnswerResponse> {
  const res = await safeFetch(apiUrl(`/assessments/${encodeURIComponent(params.sessionId)}/answer`), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...communityRequestHeaders(),
    },
    body: JSON.stringify({
      question_id: params.questionId,
      choice_index: params.choiceIndex,
    }),
    signal: params.signal,
  });
  return parseJsonResponse<AssessmentAnswerResponse>(res);
}

export async function snoozeAssessment(params: {
  sessionId: string;
  signal?: AbortSignal;
}): Promise<AssessmentSnoozeResponse> {
  const res = await safeFetch(apiUrl(`/assessments/${encodeURIComponent(params.sessionId)}/snooze`), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...communityRequestHeaders(),
    },
    signal: params.signal,
  });
  return parseJsonResponse<AssessmentSnoozeResponse>(res);
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
  active_index?: number | null;
  active_reel_id?: string | null;
  recall?: {
    recent_score?: number | null;
    recent_question_count?: number | null;
    recent_accuracy?: number | null;
    rolling_accuracy?: number | null;
    understood_concepts?: string[] | null;
    revisit_concepts?: string[] | null;
    completed_at?: string | null;
  } | null;
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
  const activeIndex = Number(row.active_index);
  const recallRow = row.recall && typeof row.recall === "object" ? row.recall : null;
  const recentScore = Number(recallRow?.recent_score);
  const recentQuestionCount = Number(recallRow?.recent_question_count);
  const recentAccuracy = Number(recallRow?.recent_accuracy);
  const rollingAccuracy = Number(recallRow?.rolling_accuracy);
  const completedAt = recallRow?.completed_at ? Date.parse(recallRow.completed_at) : Number.NaN;
  return {
    materialId,
    title,
    updatedAt: Math.max(0, Math.floor(Number(row.updated_at) || 0)),
    starred: Boolean(row.starred),
    generationMode: row.generation_mode === "fast" ? "fast" : "slow",
    source: row.source === "community" ? "community" : "search",
    feedQuery: typeof row.feed_query === "string" && row.feed_query.trim() ? row.feed_query.trim() : undefined,
    activeIndex: Number.isFinite(activeIndex) && activeIndex >= 0 ? Math.floor(activeIndex) : undefined,
    activeReelId: typeof row.active_reel_id === "string" && row.active_reel_id.trim() ? row.active_reel_id.trim() : undefined,
    recall: recallRow
      ? {
          recentScore: Number.isFinite(recentScore) ? Math.max(0, recentScore) : undefined,
          recentQuestionCount: Number.isFinite(recentQuestionCount) ? Math.max(0, Math.floor(recentQuestionCount)) : undefined,
          recentAccuracy: Number.isFinite(recentAccuracy) ? Math.max(0, Math.min(1, recentAccuracy)) : undefined,
          rollingAccuracy: Number.isFinite(rollingAccuracy) ? Math.max(0, Math.min(1, rollingAccuracy)) : undefined,
          understoodConcepts: Array.isArray(recallRow.understood_concepts)
            ? recallRow.understood_concepts.map((value) => String(value || "").trim()).filter(Boolean).slice(0, 12)
            : [],
          revisitConcepts: Array.isArray(recallRow.revisit_concepts)
            ? recallRow.revisit_concepts.map((value) => String(value || "").trim()).filter(Boolean).slice(0, 12)
            : [],
          completedAt: Number.isFinite(completedAt) && completedAt > 0 ? Math.floor(completedAt) : undefined,
        }
      : undefined,
  };
}

function normalizeCommunitySettings(raw: unknown): StudyReelsSettings {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    return normalizeStudyReelsSettings({});
  }
  const row = raw as Record<string, unknown>;
  const localSettings = readStudyReelsSettings();
  return normalizeStudyReelsSettings({
    generationMode: row.generation_mode as string | null | undefined,
    defaultInputMode: row.default_input_mode as string | null | undefined,
    minRelevanceThreshold: row.min_relevance_threshold as string | number | null | undefined,
    startMuted: row.start_muted as string | boolean | null | undefined,
    autoplayNextReel: (row.autoplay_next_reel as string | boolean | null | undefined)
      ?? (row.autoplayNextReel as string | boolean | null | undefined)
      ?? localSettings.autoplayNextReel,
    creativeCommonsOnly: row.creative_commons_only as string | boolean | null | undefined,
    preferredVideoDuration: row.preferred_video_duration as string | null | undefined,
    targetClipDurationSec: row.target_clip_duration_sec as string | number | null | undefined,
    targetClipDurationMinSec: row.target_clip_duration_min_sec as string | number | null | undefined,
    targetClipDurationMaxSec: row.target_clip_duration_max_sec as string | number | null | undefined,
  });
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
        active_index: typeof item.activeIndex === "number" && Number.isFinite(item.activeIndex)
          ? Math.max(0, Math.floor(item.activeIndex))
          : null,
        active_reel_id: item.activeReelId ?? null,
        recall: item.recall
          ? {
              recent_score: item.recall.recentScore ?? null,
              recent_question_count: item.recall.recentQuestionCount ?? null,
              recent_accuracy: item.recall.recentAccuracy ?? null,
              rolling_accuracy: item.recall.rollingAccuracy ?? null,
              understood_concepts: item.recall.understoodConcepts,
              revisit_concepts: item.recall.revisitConcepts,
              completed_at: item.recall.completedAt ? new Date(item.recall.completedAt).toISOString() : null,
            }
          : null,
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

export async function fetchCommunitySettings(): Promise<StudyReelsSettings | null> {
  const stored = readCommunityAuthSession();
  if (!stored?.sessionToken) {
    return null;
  }
  const res = await safeFetch(apiUrl("/community/settings"), {
    cache: "no-store",
    headers: { ...communitySessionHeaders() },
  });
  return normalizeCommunitySettings(await parseJsonResponse<unknown>(res));
}

export async function replaceCommunitySettings(settings: StudyReelsSettings): Promise<StudyReelsSettings> {
  const stored = readCommunityAuthSession();
  if (!stored?.sessionToken) {
    return settings;
  }
  const res = await safeFetch(apiUrl("/community/settings"), {
    method: "PUT",
    headers: {
      "Content-Type": "application/json",
      ...communitySessionHeaders(),
    },
    body: JSON.stringify({
      generation_mode: settings.generationMode,
      default_input_mode: settings.defaultInputMode,
      min_relevance_threshold: settings.minRelevanceThreshold,
      start_muted: settings.startMuted,
      creative_commons_only: settings.creativeCommonsOnly,
      preferred_video_duration: settings.preferredVideoDuration,
    }),
  });
  return normalizeCommunitySettings(await parseJsonResponse<unknown>(res));
}

let communitySettingsSyncVersion = 0;
let communitySettingsSyncChain: Promise<void> = Promise.resolve();

export function queueCommunitySettingsSync(settings: StudyReelsSettings): Promise<void> {
  const snapshot = { ...settings };
  const syncVersion = communitySettingsSyncVersion + 1;
  communitySettingsSyncVersion = syncVersion;
  communitySettingsSyncChain = communitySettingsSyncChain
    .catch(() => {
      // Keep the queue alive after a prior failed sync.
    })
    .then(async () => {
      if (syncVersion !== communitySettingsSyncVersion) {
        return;
      }
      await replaceCommunitySettings(snapshot);
    });
  return communitySettingsSyncChain;
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
  keepSignalAliveThroughBody?: boolean;
};

async function safeFetch(url: string, init?: SafeFetchInit): Promise<Response> {
  const timeoutRaw = Number(init?.timeoutMs);
  const timeoutMs = Number.isFinite(timeoutRaw) && timeoutRaw > 0 ? Math.max(1_000, timeoutRaw) : null;
  const upstreamSignal = init?.signal;
  const controller = new AbortController();
  const keepSignalAliveThroughBody = init?.keepSignalAliveThroughBody === true;
  let didTimeout = false;
  const timeoutId = timeoutMs
    ? setTimeout(() => {
      didTimeout = true;
      controller.abort();
    }, timeoutMs)
    : null;
  const onUpstreamAbort = () => controller.abort();
  let cleanedUp = false;
  const cleanup = () => {
    if (cleanedUp) {
      return;
    }
    cleanedUp = true;
    if (timeoutId) {
      clearTimeout(timeoutId);
    }
    if (upstreamSignal) {
      upstreamSignal.removeEventListener("abort", onUpstreamAbort);
    }
  };
  const abortError = (): Error => didTimeout
    ? new TransportError("Request timed out before the backend response completed.")
    : new Error("Request was interrupted.");

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
    delete (requestInit as SafeFetchInit).keepSignalAliveThroughBody;
    response = await fetch(url, requestInit);
  } catch (error) {
    cleanup();
    if (error instanceof Error && error.name === "AbortError") {
      throw abortError();
    }
    throw new TransportError(getBackendDownError());
  }

  if (!keepSignalAliveThroughBody || !response.body) {
    cleanup();
    if (!response.ok) {
      throw await buildApiError(response);
    }
    return response;
  }

  const sourceReader = response.body.getReader();
  let sourceReaderReleased = false;
  const releaseSourceReader = () => {
    if (sourceReaderReleased) {
      return;
    }
    sourceReaderReleased = true;
    sourceReader.releaseLock();
  };
  const body = new ReadableStream<Uint8Array>({
    async pull(streamController) {
      try {
        const { done, value } = await sourceReader.read();
        if (done) {
          releaseSourceReader();
          cleanup();
          streamController.close();
          return;
        }
        streamController.enqueue(value);
      } catch (error) {
        releaseSourceReader();
        cleanup();
        if (error instanceof Error && error.name === "AbortError") {
          streamController.error(abortError());
          return;
        }
        streamController.error(new TransportError(getBackendDownError()));
      }
    },
    async cancel(reason) {
      controller.abort();
      try {
        await sourceReader.cancel(reason);
      } catch {
        // The source may already be errored by the request abort.
      } finally {
        releaseSourceReader();
        cleanup();
      }
    },
  });

  const wrappedResponse = new Response(body, {
    status: response.status,
    statusText: response.statusText,
    headers: response.headers,
  });
  if (!wrappedResponse.ok) {
    throw await buildApiError(wrappedResponse);
  }
  return wrappedResponse;
}

export function isRequestInterruptedError(error: unknown): boolean {
  return error instanceof Error && error.message === "Request was interrupted.";
}

class TransportError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "TransportError";
  }
}

export function isTransportError(error: unknown): boolean {
  return error instanceof TransportError;
}

export class ApiError extends Error {
  status: number;
  code: string | null;
  payload: TypedApiError | null;
  constructor(message: string, status: number, code: string | null = null, payload: TypedApiError | null = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.payload = payload;
  }
}

async function buildApiError(response: Response): Promise<ApiError> {
  let message = `Request failed (${response.status})`;
  let typedError: TypedApiError | null = null;
  try {
    const json = await response.json();
    const detail = json?.detail ?? json?.error ?? json;
    if (typeof detail === "string") {
      message = detail;
    } else if (detail && typeof detail === "object" && !Array.isArray(detail)) {
      const row = detail as Record<string, unknown>;
      const code = typeof row.code === "string" ? row.code.trim() : "";
      const typedMessage = typeof row.message === "string" ? row.message.trim() : "";
      if (code && typedMessage) {
        const retryAfter = Number(row.retry_after_sec);
        typedError = {
          code,
          message: typedMessage,
          provider: typeof row.provider === "string" ? row.provider : null,
          retry_after_sec: Number.isFinite(retryAfter) ? retryAfter : null,
          details: row.details && typeof row.details === "object" && !Array.isArray(row.details)
            ? row.details as Record<string, unknown>
            : null,
        };
        message = typedMessage;
      } else if (typeof json?.message === "string" && json.message.trim()) {
        message = json.message.trim();
      }
    }
  } catch (error) {
    if (isTransportError(error) || isRequestInterruptedError(error)) {
      throw error;
    }
    // Keep the generic status message when the body is absent or malformed.
  }
  return new ApiError(message, response.status, typedError?.code ?? null, typedError);
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
