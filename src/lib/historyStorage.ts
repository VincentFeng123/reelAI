"use client";

export type StoredHistoryGenerationMode = "slow" | "fast";
export type StoredHistorySource = "search" | "community";

export type StoredHistoryItem = {
  materialId: string;
  title: string;
  updatedAt: number;
  starred: boolean;
  generationMode: StoredHistoryGenerationMode;
  source: StoredHistorySource;
  feedQuery?: string;
  activeIndex?: number;
  activeReelId?: string;
};

export const HISTORY_STORAGE_KEY = "studyreels-material-history";
export const LEGACY_TOPIC_HISTORY_STORAGE_KEY = "studyreels-reel-topic-history";

const HISTORY_SCOPE_STORAGE_PREFIX = "studyreels-material-history-scope";

function normalizeHistoryAccountId(accountId: string | null | undefined): string | null {
  const normalized = String(accountId || "").trim();
  return normalized || null;
}

function historyScopeStorageKey(accountId: string | null | undefined): string {
  return `${HISTORY_SCOPE_STORAGE_PREFIX}:${normalizeHistoryAccountId(accountId) ?? "guest"}`;
}

function seedGuestHistoryScopeFromActiveHistory(): string | null {
  if (typeof window === "undefined") {
    return null;
  }
  try {
    const legacyRaw = window.localStorage.getItem(HISTORY_STORAGE_KEY);
    if (legacyRaw === null) {
      return null;
    }
    window.localStorage.setItem(historyScopeStorageKey(null), legacyRaw);
    return legacyRaw;
  } catch {
    return null;
  }
}

export function normalizeStoredHistoryItem(raw: unknown): StoredHistoryItem | null {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    return null;
  }
  const row = raw as Record<string, unknown>;
  const materialId = String(row.materialId || "").trim();
  const title = String(row.title || "").trim();
  if (!materialId || !title) {
    return null;
  }
  const activeIndexRaw = Number(row.activeIndex);
  const activeIndex = Number.isFinite(activeIndexRaw) && activeIndexRaw >= 0 ? Math.floor(activeIndexRaw) : undefined;
  const activeReelId = typeof row.activeReelId === "string" && row.activeReelId.trim() ? row.activeReelId.trim() : undefined;
  return {
    materialId,
    title: title || "New Study Session",
    updatedAt: Math.max(0, Math.floor(Number(row.updatedAt) || 0)),
    starred: Boolean(row.starred),
    generationMode: row.generationMode === "slow" ? "slow" : "fast",
    source: row.source === "community" ? "community" : "search",
    feedQuery: typeof row.feedQuery === "string" && row.feedQuery.trim() ? row.feedQuery.trim() : undefined,
    activeIndex,
    activeReelId,
  };
}

export function normalizeStoredHistoryItems(raw: unknown): StoredHistoryItem[] {
  if (!Array.isArray(raw)) {
    return [];
  }
  return raw.map(normalizeStoredHistoryItem).filter(Boolean) as StoredHistoryItem[];
}

export function readScopedHistorySnapshot(accountId: string | null | undefined): string | null {
  if (typeof window === "undefined") {
    return null;
  }
  try {
    const normalizedAccountId = normalizeHistoryAccountId(accountId);
    const scopedRaw = window.localStorage.getItem(historyScopeStorageKey(normalizedAccountId));
    if (scopedRaw !== null) {
      return scopedRaw;
    }
    if (normalizedAccountId) {
      return null;
    }
    return seedGuestHistoryScopeFromActiveHistory();
  } catch {
    return null;
  }
}

export function writeScopedHistorySnapshot(accountId: string | null | undefined, rawHistory: string | null): void {
  if (typeof window === "undefined") {
    return;
  }
  try {
    const scopedKey = historyScopeStorageKey(accountId);
    if (rawHistory === null) {
      window.localStorage.removeItem(scopedKey);
      window.localStorage.removeItem(HISTORY_STORAGE_KEY);
      return;
    }
    window.localStorage.setItem(scopedKey, rawHistory);
    window.localStorage.setItem(HISTORY_STORAGE_KEY, rawHistory);
  } catch {
    // Ignore storage failures so the rest of the UI remains usable.
  }
}
