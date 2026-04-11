"use client";

export type GenerationMode = "slow" | "fast";
export type SearchInputMode = "topic" | "source" | "file" | "url";
export type VideoPoolMode = "short-first" | "balanced" | "long-form";
export type PreferredVideoDuration = "any" | "short" | "medium" | "long";

export type StudyReelsSettings = {
  generationMode: GenerationMode;
  defaultInputMode: SearchInputMode;
  minRelevanceThreshold: number;
  startMuted: boolean;
  autoplayNextReel: boolean;
  videoPoolMode: VideoPoolMode;
  preferredVideoDuration: PreferredVideoDuration;
  targetClipDurationSec: number;
  targetClipDurationMinSec: number;
  targetClipDurationMaxSec: number;
  /**
   * When true, topic/text/file submits go through /api/ingest/search (real scraping
   * of YouTube + Instagram + TikTok). When false, they fall back to the legacy
   * /api/material → YouTube-only pipeline. Local-only, persisted client-side.
   */
  multiPlatformSearch: boolean;
};

export const GENERATION_MODE_STORAGE_KEY = "studyreels-generation-mode";
export const SEARCH_INPUT_MODE_STORAGE_KEY = "studyreels-search-input-mode";
export const MIN_RELEVANCE_STORAGE_KEY = "studyreels-min-relevance-threshold";
export const MUTED_STORAGE_KEY = "studyreels-muted";
export const VIDEO_POOL_MODE_STORAGE_KEY = "studyreels-video-pool-mode";
export const PREFERRED_VIDEO_DURATION_STORAGE_KEY = "studyreels-preferred-video-duration";
export const TARGET_CLIP_DURATION_STORAGE_KEY = "studyreels-target-clip-duration-sec";
export const TARGET_CLIP_DURATION_MIN_STORAGE_KEY = "studyreels-target-clip-duration-min-sec";
export const TARGET_CLIP_DURATION_MAX_STORAGE_KEY = "studyreels-target-clip-duration-max-sec";
export const SETTINGS_UPDATED_EVENT = "studyreels-settings-updated";

const SETTINGS_SCOPE_STORAGE_PREFIX = "studyreels-settings-scope";
const ACTIVE_SETTINGS_SCOPE_STORAGE_KEY = "studyreels-settings-active-account";

export const MIN_RELEVANCE_MIN = 0.0;
export const MIN_RELEVANCE_MAX = 0.6;
export const MIN_RELEVANCE = MIN_RELEVANCE_MIN;
export const MAX_RELEVANCE = MIN_RELEVANCE_MAX;
export const TARGET_CLIP_DURATION_MIN = 15;
export const TARGET_CLIP_DURATION_MAX = 180;
export const TARGET_CLIP_DURATION_MIN_GAP = 15;

export const DEFAULT_STUDY_REELS_SETTINGS: StudyReelsSettings = {
  generationMode: "fast",
  defaultInputMode: "source",
  minRelevanceThreshold: 0.3,
  startMuted: true,
  autoplayNextReel: false,
  videoPoolMode: "short-first",
  preferredVideoDuration: "any",
  targetClipDurationSec: 55,
  targetClipDurationMinSec: 20,
  targetClipDurationMaxSec: 55,
  // Legacy YouTube HTML-scraping flow only. The /api/ingest/search path
  // (yt-dlp + Instagram/TikTok) gets bounced by robots.txt and YouTube's
  // cloud-IP bot detection, so the submit flow in UploadPanel ignores this
  // and always routes through /api/material + /api/reels/generate-stream.
  multiPlatformSearch: false,
};

type StudyReelsSettingsInput = {
  generationMode?: GenerationMode | string | null;
  defaultInputMode?: SearchInputMode | string | null;
  minRelevanceThreshold?: number | string | null;
  startMuted?: boolean | string | null;
  autoplayNextReel?: boolean | string | null;
  videoPoolMode?: VideoPoolMode | string | null;
  preferredVideoDuration?: PreferredVideoDuration | string | null;
  targetClipDurationSec?: number | string | null;
  targetClipDurationMinSec?: number | string | null;
  targetClipDurationMaxSec?: number | string | null;
  multiPlatformSearch?: boolean | string | null;
};

let activeSettingsScopeMemoryFallback: string | null = null;

function toGenerationMode(value: string | null | undefined): GenerationMode {
  return value === "slow" ? "slow" : "fast";
}

function toInputMode(value: string | null | undefined): SearchInputMode {
  // `url` is intentionally excluded — it's a legacy/internal value that the settings
  // picker no longer exposes. Any persisted `url` default falls back to `source`.
  if (value === "topic" || value === "file") {
    return value;
  }
  return "source";
}

function toVideoPoolMode(value: string | null | undefined): VideoPoolMode {
  if (value === "balanced" || value === "long-form") {
    return value;
  }
  return "short-first";
}

function toPreferredVideoDuration(value: string | null | undefined): PreferredVideoDuration {
  if (value === "short" || value === "medium" || value === "long") {
    return value;
  }
  return "any";
}

function clampNumber(value: unknown, min: number, max: number, fallback: number): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return Math.max(min, Math.min(max, parsed));
}

function defaultClipDurationBounds(targetClipDurationSec: number): { min: number; max: number } {
  const min = Math.max(TARGET_CLIP_DURATION_MIN, Math.round(targetClipDurationSec * 0.35));
  const max = Math.max(min + TARGET_CLIP_DURATION_MIN_GAP, Math.round(targetClipDurationSec));
  return {
    min,
    max: Math.min(TARGET_CLIP_DURATION_MAX, max),
  };
}

function enforceClipDurationGap(minSec: number, maxSec: number): { min: number; max: number } {
  let min = Math.max(TARGET_CLIP_DURATION_MIN, Math.min(TARGET_CLIP_DURATION_MAX, Math.round(minSec)));
  let max = Math.max(TARGET_CLIP_DURATION_MIN, Math.min(TARGET_CLIP_DURATION_MAX, Math.round(maxSec)));
  if (min > max) {
    [min, max] = [max, min];
  }
  if (max - min >= TARGET_CLIP_DURATION_MIN_GAP) {
    return { min, max };
  }

  const expandedMax = Math.min(TARGET_CLIP_DURATION_MAX, min + TARGET_CLIP_DURATION_MIN_GAP);
  if (expandedMax - min >= TARGET_CLIP_DURATION_MIN_GAP) {
    return { min, max: expandedMax };
  }

  const loweredMin = Math.max(TARGET_CLIP_DURATION_MIN, max - TARGET_CLIP_DURATION_MIN_GAP);
  return { min: loweredMin, max };
}

function normalizeSettingsAccountId(accountId: string | null | undefined): string | null {
  const normalized = String(accountId || "").trim();
  return normalized || null;
}

function settingsScopeStorageKey(accountId: string | null | undefined): string {
  return `${SETTINGS_SCOPE_STORAGE_PREFIX}:${normalizeSettingsAccountId(accountId) ?? "guest"}`;
}

function hasLegacySettingsSnapshot(): boolean {
  if (typeof window === "undefined") {
    return false;
  }
  return [
    GENERATION_MODE_STORAGE_KEY,
    SEARCH_INPUT_MODE_STORAGE_KEY,
    MIN_RELEVANCE_STORAGE_KEY,
    MUTED_STORAGE_KEY,
    VIDEO_POOL_MODE_STORAGE_KEY,
    PREFERRED_VIDEO_DURATION_STORAGE_KEY,
    TARGET_CLIP_DURATION_STORAGE_KEY,
    TARGET_CLIP_DURATION_MIN_STORAGE_KEY,
    TARGET_CLIP_DURATION_MAX_STORAGE_KEY,
  ].some((key) => window.localStorage.getItem(key) !== null);
}

function readLegacyStudyReelsSettings(): StudyReelsSettings {
  if (typeof window === "undefined") {
    return DEFAULT_STUDY_REELS_SETTINGS;
  }
  return normalizeStudyReelsSettings({
    generationMode: (window.localStorage.getItem(GENERATION_MODE_STORAGE_KEY) || undefined) as GenerationMode | undefined,
    defaultInputMode: (window.localStorage.getItem(SEARCH_INPUT_MODE_STORAGE_KEY) || undefined) as SearchInputMode | undefined,
    minRelevanceThreshold: window.localStorage.getItem(MIN_RELEVANCE_STORAGE_KEY),
    startMuted: window.localStorage.getItem(MUTED_STORAGE_KEY) !== "0",
    videoPoolMode: (window.localStorage.getItem(VIDEO_POOL_MODE_STORAGE_KEY) || undefined) as VideoPoolMode | undefined,
    preferredVideoDuration: (window.localStorage.getItem(PREFERRED_VIDEO_DURATION_STORAGE_KEY) || undefined) as PreferredVideoDuration | undefined,
    targetClipDurationSec: window.localStorage.getItem(TARGET_CLIP_DURATION_STORAGE_KEY),
    targetClipDurationMinSec: window.localStorage.getItem(TARGET_CLIP_DURATION_MIN_STORAGE_KEY),
    targetClipDurationMaxSec: window.localStorage.getItem(TARGET_CLIP_DURATION_MAX_STORAGE_KEY),
  });
}

function parseScopedStudyReelsSettingsSnapshot(raw: string | null): StudyReelsSettings | null {
  if (!raw) {
    return null;
  }
  try {
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return null;
    }
    const row = parsed as Record<string, unknown>;
    return normalizeStudyReelsSettings({
      generationMode: row.generationMode as string | null | undefined,
      defaultInputMode: row.defaultInputMode as string | null | undefined,
      minRelevanceThreshold: row.minRelevanceThreshold as string | number | null | undefined,
      startMuted: row.startMuted as string | boolean | null | undefined,
      autoplayNextReel: row.autoplayNextReel as string | boolean | null | undefined,
      videoPoolMode: row.videoPoolMode as string | null | undefined,
      preferredVideoDuration: row.preferredVideoDuration as string | null | undefined,
      targetClipDurationSec: row.targetClipDurationSec as string | number | null | undefined,
      targetClipDurationMinSec: row.targetClipDurationMinSec as string | number | null | undefined,
      targetClipDurationMaxSec: row.targetClipDurationMaxSec as string | number | null | undefined,
      multiPlatformSearch: row.multiPlatformSearch as string | boolean | null | undefined,
    });
  } catch {
    return null;
  }
}

function serializeScopedStudyReelsSettingsSnapshot(settings: StudyReelsSettings): string {
  return JSON.stringify(settings);
}

function dispatchSettingsUpdated(settings: StudyReelsSettings): void {
  if (typeof window === "undefined") {
    return;
  }
  window.dispatchEvent(new CustomEvent(SETTINGS_UPDATED_EVENT, { detail: settings }));
}

function seedGuestScopedSettingsFromLegacy(): StudyReelsSettings {
  const seeded = hasLegacySettingsSnapshot() ? readLegacyStudyReelsSettings() : DEFAULT_STUDY_REELS_SETTINGS;
  try {
    if (typeof window !== "undefined") {
      window.localStorage.setItem(settingsScopeStorageKey(null), serializeScopedStudyReelsSettingsSnapshot(seeded));
    }
  } catch {
    // Ignore storage failures and fall back to the normalized settings object.
  }
  return seeded;
}

function readStoredActiveSettingsScopeAccountId(): string | null {
  if (typeof window === "undefined") {
    return activeSettingsScopeMemoryFallback;
  }
  try {
    const stored = normalizeSettingsAccountId(window.sessionStorage.getItem(ACTIVE_SETTINGS_SCOPE_STORAGE_KEY));
    activeSettingsScopeMemoryFallback = stored;
    return stored;
  } catch {
    return activeSettingsScopeMemoryFallback;
  }
}

export function normalizeStudyReelsSettings(raw: StudyReelsSettingsInput): StudyReelsSettings {
  const minRelevanceThreshold = clampNumber(
    raw.minRelevanceThreshold,
    MIN_RELEVANCE_MIN,
    MIN_RELEVANCE_MAX,
    DEFAULT_STUDY_REELS_SETTINGS.minRelevanceThreshold,
  );
  const targetClipDurationSec = clampNumber(
    raw.targetClipDurationSec,
    TARGET_CLIP_DURATION_MIN,
    TARGET_CLIP_DURATION_MAX,
    DEFAULT_STUDY_REELS_SETTINGS.targetClipDurationSec,
  );
  const fallbackBounds = defaultClipDurationBounds(Math.round(targetClipDurationSec));
  let targetClipDurationMinSec = Math.round(
    clampNumber(
      raw.targetClipDurationMinSec,
      TARGET_CLIP_DURATION_MIN,
      TARGET_CLIP_DURATION_MAX,
      fallbackBounds.min,
    ),
  );
  let targetClipDurationMaxSec = Math.round(
    clampNumber(
      raw.targetClipDurationMaxSec,
      TARGET_CLIP_DURATION_MIN,
      TARGET_CLIP_DURATION_MAX,
      fallbackBounds.max,
    ),
  );
  const adjustedBounds = enforceClipDurationGap(targetClipDurationMinSec, targetClipDurationMaxSec);
  targetClipDurationMinSec = adjustedBounds.min;
  targetClipDurationMaxSec = adjustedBounds.max;
  const hasExplicitTarget = raw.targetClipDurationSec !== null && raw.targetClipDurationSec !== undefined && raw.targetClipDurationSec !== "";
  const midpointTarget = Math.round((targetClipDurationMinSec + targetClipDurationMaxSec) / 2);
  const normalizedTarget = hasExplicitTarget
    ? Math.round(clampNumber(targetClipDurationSec, targetClipDurationMinSec, targetClipDurationMaxSec, midpointTarget))
    : midpointTarget;

  // Force-override any stored value (true or false) back to false. The
  // multi-platform /api/ingest/search path is broken on cloud IPs (YouTube
  // bot detection + IG/TT robots.txt), so we strand existing users on the
  // legacy flow regardless of their old persisted preference. The setting
  // is still surfaced in SettingsPanel as a toggle but has no effect on
  // the submit routing in UploadPanel.
  const multiPlatformSearch = false;

  return {
    generationMode: toGenerationMode(raw.generationMode),
    defaultInputMode: toInputMode(raw.defaultInputMode),
    minRelevanceThreshold: Number(minRelevanceThreshold.toFixed(2)),
    startMuted: raw.startMuted !== false && raw.startMuted !== "0" && raw.startMuted !== "false",
    autoplayNextReel: raw.autoplayNextReel !== false && raw.autoplayNextReel !== "0" && raw.autoplayNextReel !== "false",
    videoPoolMode: toVideoPoolMode(raw.videoPoolMode),
    preferredVideoDuration: toPreferredVideoDuration(raw.preferredVideoDuration),
    targetClipDurationSec: normalizedTarget,
    targetClipDurationMinSec,
    targetClipDurationMaxSec,
    multiPlatformSearch,
  };
}

export function readScopedStudyReelsSettings(accountId: string | null | undefined): StudyReelsSettings {
  if (typeof window === "undefined") {
    return DEFAULT_STUDY_REELS_SETTINGS;
  }
  try {
    const normalizedAccountId = normalizeSettingsAccountId(accountId);
    const scoped = parseScopedStudyReelsSettingsSnapshot(window.localStorage.getItem(settingsScopeStorageKey(normalizedAccountId)));
    if (scoped) {
      return scoped;
    }
    if (normalizedAccountId) {
      return DEFAULT_STUDY_REELS_SETTINGS;
    }
    return seedGuestScopedSettingsFromLegacy();
  } catch {
    return DEFAULT_STUDY_REELS_SETTINGS;
  }
}

export function writeScopedStudyReelsSettings(
  accountId: string | null | undefined,
  raw: StudyReelsSettingsInput,
  options?: { dispatch?: boolean },
): StudyReelsSettings {
  const normalized = normalizeStudyReelsSettings(raw);
  if (typeof window !== "undefined") {
    try {
      window.localStorage.setItem(
        settingsScopeStorageKey(accountId),
        serializeScopedStudyReelsSettingsSnapshot(normalized),
      );
    } catch {
      // Ignore storage failures and keep the normalized settings in memory.
    }
    if ((options?.dispatch ?? true) && normalizeSettingsAccountId(accountId) === readStoredActiveSettingsScopeAccountId()) {
      dispatchSettingsUpdated(normalized);
    }
  }
  return normalized;
}

export function readActiveStudyReelsSettingsScopeAccountId(): string | null {
  return readStoredActiveSettingsScopeAccountId();
}

export function setActiveStudyReelsSettingsScope(
  accountId: string | null | undefined,
  options?: { settings?: StudyReelsSettingsInput; dispatch?: boolean },
): StudyReelsSettings {
  const normalizedAccountId = normalizeSettingsAccountId(accountId);
  activeSettingsScopeMemoryFallback = normalizedAccountId;
  if (typeof window !== "undefined") {
    try {
      if (normalizedAccountId) {
        window.sessionStorage.setItem(ACTIVE_SETTINGS_SCOPE_STORAGE_KEY, normalizedAccountId);
      } else {
        window.sessionStorage.removeItem(ACTIVE_SETTINGS_SCOPE_STORAGE_KEY);
      }
    } catch {
      // Ignore storage failures and fall back to the in-memory scope.
    }
  }
  const next = options?.settings
    ? writeScopedStudyReelsSettings(normalizedAccountId, options.settings, { dispatch: false })
    : readScopedStudyReelsSettings(normalizedAccountId);
  if (options?.dispatch ?? true) {
    dispatchSettingsUpdated(next);
  }
  return next;
}

export function readStudyReelsSettings(): StudyReelsSettings {
  return readScopedStudyReelsSettings(readStoredActiveSettingsScopeAccountId());
}

export function saveStudyReelsSettings(raw: StudyReelsSettingsInput): StudyReelsSettings {
  return writeScopedStudyReelsSettings(readStoredActiveSettingsScopeAccountId(), raw);
}

export function subscribeToStudyReelsSettings(onChange: (settings: StudyReelsSettings) => void): () => void {
  if (typeof window === "undefined") {
    return () => {};
  }

  const emit = (next?: StudyReelsSettings) => {
    onChange(next ?? readStudyReelsSettings());
  };

  const onEvent = (event: Event) => {
    if (event instanceof CustomEvent) {
      const detail = event.detail as StudyReelsSettingsInput | undefined;
      if (detail && typeof detail === "object") {
        emit(normalizeStudyReelsSettings(detail));
        return;
      }
    }
    emit();
  };

  const onStorage = (event: StorageEvent) => {
    if (event.storageArea !== window.localStorage) {
      return;
    }
    const activeScopeKey = settingsScopeStorageKey(readStoredActiveSettingsScopeAccountId());
    if (event.key && event.key !== activeScopeKey) {
      return;
    }
    emit();
  };

  window.addEventListener(SETTINGS_UPDATED_EVENT, onEvent as EventListener);
  window.addEventListener("storage", onStorage);

  return () => {
    window.removeEventListener(SETTINGS_UPDATED_EVENT, onEvent as EventListener);
    window.removeEventListener("storage", onStorage);
  };
}
