export type GenerationMode = "slow" | "fast";
export type SearchInputMode = "topic" | "source" | "file";
export type VideoPoolMode = "short-first" | "balanced" | "long-form";
export type PreferredVideoDuration = "any" | "short" | "medium" | "long";

export type StudyReelsSettings = {
  generationMode: GenerationMode;
  defaultInputMode: SearchInputMode;
  minRelevanceThreshold: number;
  startMuted: boolean;
  videoPoolMode: VideoPoolMode;
  preferredVideoDuration: PreferredVideoDuration;
  targetClipDurationSec: number;
  targetClipDurationMinSec: number;
  targetClipDurationMaxSec: number;
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

export const MIN_RELEVANCE_MIN = 0.0;
export const MIN_RELEVANCE_MAX = 0.6;
export const MIN_RELEVANCE = MIN_RELEVANCE_MIN;
export const MAX_RELEVANCE = MIN_RELEVANCE_MAX;
export const TARGET_CLIP_DURATION_MIN = 15;
export const TARGET_CLIP_DURATION_MAX = 180;
export const TARGET_CLIP_DURATION_MIN_GAP = 15;

export const DEFAULT_STUDY_REELS_SETTINGS: StudyReelsSettings = {
  generationMode: "slow",
  defaultInputMode: "source",
  minRelevanceThreshold: 0.3,
  startMuted: true,
  videoPoolMode: "short-first",
  preferredVideoDuration: "any",
  targetClipDurationSec: 55,
  targetClipDurationMinSec: 20,
  targetClipDurationMaxSec: 55,
};

type StudyReelsSettingsInput = {
  generationMode?: GenerationMode | string | null;
  defaultInputMode?: SearchInputMode | string | null;
  minRelevanceThreshold?: number | string | null;
  startMuted?: boolean | string | null;
  videoPoolMode?: VideoPoolMode | string | null;
  preferredVideoDuration?: PreferredVideoDuration | string | null;
  targetClipDurationSec?: number | string | null;
  targetClipDurationMinSec?: number | string | null;
  targetClipDurationMaxSec?: number | string | null;
};

function toGenerationMode(value: string | null | undefined): GenerationMode {
  return value === "fast" ? "fast" : "slow";
}

function toInputMode(value: string | null | undefined): SearchInputMode {
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

  return {
    generationMode: toGenerationMode(raw.generationMode),
    defaultInputMode: toInputMode(raw.defaultInputMode),
    minRelevanceThreshold: Number(minRelevanceThreshold.toFixed(2)),
    startMuted: raw.startMuted !== false && raw.startMuted !== "0" && raw.startMuted !== "false",
    videoPoolMode: toVideoPoolMode(raw.videoPoolMode),
    preferredVideoDuration: toPreferredVideoDuration(raw.preferredVideoDuration),
    targetClipDurationSec: normalizedTarget,
    targetClipDurationMinSec,
    targetClipDurationMaxSec,
  };
}

export function readStudyReelsSettings(): StudyReelsSettings {
  if (typeof window === "undefined") {
    return DEFAULT_STUDY_REELS_SETTINGS;
  }
  const fromStorage = normalizeStudyReelsSettings({
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
  return fromStorage;
}

export function saveStudyReelsSettings(raw: StudyReelsSettingsInput): StudyReelsSettings {
  const next = normalizeStudyReelsSettings(raw);
  if (typeof window === "undefined") {
    return next;
  }
  window.localStorage.setItem(GENERATION_MODE_STORAGE_KEY, next.generationMode);
  window.localStorage.setItem(SEARCH_INPUT_MODE_STORAGE_KEY, next.defaultInputMode);
  window.localStorage.setItem(MIN_RELEVANCE_STORAGE_KEY, next.minRelevanceThreshold.toFixed(2));
  window.localStorage.setItem(MUTED_STORAGE_KEY, next.startMuted ? "1" : "0");
  window.localStorage.setItem(VIDEO_POOL_MODE_STORAGE_KEY, next.videoPoolMode);
  window.localStorage.setItem(PREFERRED_VIDEO_DURATION_STORAGE_KEY, next.preferredVideoDuration);
  window.localStorage.setItem(TARGET_CLIP_DURATION_STORAGE_KEY, String(next.targetClipDurationSec));
  window.localStorage.setItem(TARGET_CLIP_DURATION_MIN_STORAGE_KEY, String(next.targetClipDurationMinSec));
  window.localStorage.setItem(TARGET_CLIP_DURATION_MAX_STORAGE_KEY, String(next.targetClipDurationMaxSec));
  window.dispatchEvent(new CustomEvent(SETTINGS_UPDATED_EVENT, { detail: next }));
  return next;
}

export function subscribeToStudyReelsSettings(onChange: (settings: StudyReelsSettings) => void): () => void {
  if (typeof window === "undefined") {
    return () => {};
  }

  const onEvent = (event: Event) => {
    if (event instanceof CustomEvent) {
      const detail = event.detail as StudyReelsSettingsInput | undefined;
      if (detail && typeof detail === "object") {
        onChange(normalizeStudyReelsSettings(detail));
        return;
      }
    }
    onChange(readStudyReelsSettings());
  };

  const onStorage = (event: StorageEvent) => {
    if (event.storageArea !== window.localStorage) {
      return;
    }
    if (
      event.key
      && ![
        GENERATION_MODE_STORAGE_KEY,
        SEARCH_INPUT_MODE_STORAGE_KEY,
        MIN_RELEVANCE_STORAGE_KEY,
        MUTED_STORAGE_KEY,
        VIDEO_POOL_MODE_STORAGE_KEY,
        PREFERRED_VIDEO_DURATION_STORAGE_KEY,
        TARGET_CLIP_DURATION_STORAGE_KEY,
        TARGET_CLIP_DURATION_MIN_STORAGE_KEY,
        TARGET_CLIP_DURATION_MAX_STORAGE_KEY,
      ].includes(event.key)
    ) {
      return;
    }
    onChange(readStudyReelsSettings());
  };

  window.addEventListener(SETTINGS_UPDATED_EVENT, onEvent as EventListener);
  window.addEventListener("storage", onStorage);

  return () => {
    window.removeEventListener(SETTINGS_UPDATED_EVENT, onEvent as EventListener);
    window.removeEventListener("storage", onStorage);
  };
}
