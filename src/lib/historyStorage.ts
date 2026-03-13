"use client";

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
