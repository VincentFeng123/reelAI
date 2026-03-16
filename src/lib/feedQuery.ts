import { normalizeStudyReelsSettings, type GenerationMode, type StudyReelsSettings } from "@/lib/settings";

export type SearchFeedQuerySettings = Pick<
  StudyReelsSettings,
  | "minRelevanceThreshold"
  | "startMuted"
  | "videoPoolMode"
  | "preferredVideoDuration"
  | "targetClipDurationSec"
  | "targetClipDurationMinSec"
  | "targetClipDurationMaxSec"
>;

export type SearchFeedQuerySettingsOverride = Partial<SearchFeedQuerySettings>;

function parseQueryNumber(value: string | null): number | undefined {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function parseQueryBoolean(value: string | null): boolean | undefined {
  const normalized = String(value || "").trim().toLowerCase();
  if (normalized === "1" || normalized === "true") {
    return true;
  }
  if (normalized === "0" || normalized === "false") {
    return false;
  }
  return undefined;
}

export function applySearchFeedSettingsToParams(
  params: URLSearchParams,
  settings: SearchFeedQuerySettings,
): URLSearchParams {
  params.set("min_relevance", String(Number(settings.minRelevanceThreshold.toFixed(2))));
  params.set("video_pool_mode", settings.videoPoolMode);
  params.set("preferred_video_duration", settings.preferredVideoDuration);
  params.set("target_clip_duration_sec", String(Math.round(settings.targetClipDurationSec)));
  params.set("target_clip_duration_min_sec", String(Math.round(settings.targetClipDurationMinSec)));
  params.set("target_clip_duration_max_sec", String(Math.round(settings.targetClipDurationMaxSec)));
  params.set("start_muted", settings.startMuted ? "1" : "0");
  return params;
}

export function buildSearchFeedQuery(params: {
  materialId: string;
  generationMode: GenerationMode;
  returnTab?: string;
  returnCommunitySetId?: string | null;
  settings: SearchFeedQuerySettings;
}): string {
  const query = new URLSearchParams();
  query.set("material_id", params.materialId);
  query.set("generation_mode", params.generationMode);
  if (params.returnTab?.trim()) {
    query.set("return_tab", params.returnTab.trim());
  }
  if (params.returnCommunitySetId?.trim()) {
    query.set("return_community_set_id", params.returnCommunitySetId.trim());
  }
  applySearchFeedSettingsToParams(query, params.settings);
  return query.toString();
}

export function readSearchFeedQuerySettings(getParam: (key: string) => string | null): SearchFeedQuerySettingsOverride | null {
  const settings: SearchFeedQuerySettingsOverride = {
    minRelevanceThreshold: parseQueryNumber(getParam("min_relevance")),
    startMuted: parseQueryBoolean(getParam("start_muted")),
    videoPoolMode: (getParam("video_pool_mode") || undefined) as SearchFeedQuerySettings["videoPoolMode"] | undefined,
    preferredVideoDuration: (
      getParam("preferred_video_duration") || undefined
    ) as SearchFeedQuerySettings["preferredVideoDuration"] | undefined,
    targetClipDurationSec: parseQueryNumber(getParam("target_clip_duration_sec")),
    targetClipDurationMinSec: parseQueryNumber(getParam("target_clip_duration_min_sec")),
    targetClipDurationMaxSec: parseQueryNumber(getParam("target_clip_duration_max_sec")),
  };
  return Object.values(settings).some((value) => value !== undefined) ? settings : null;
}

export function mergeSearchFeedQuerySettings(
  base: StudyReelsSettings,
  override: SearchFeedQuerySettingsOverride | null | undefined,
): StudyReelsSettings {
  if (!override) {
    return base;
  }
  return normalizeStudyReelsSettings({
    generationMode: base.generationMode,
    defaultInputMode: base.defaultInputMode,
    minRelevanceThreshold: override.minRelevanceThreshold ?? base.minRelevanceThreshold,
    startMuted: override.startMuted ?? base.startMuted,
    videoPoolMode: override.videoPoolMode ?? base.videoPoolMode,
    preferredVideoDuration: override.preferredVideoDuration ?? base.preferredVideoDuration,
    targetClipDurationSec: override.targetClipDurationSec ?? base.targetClipDurationSec,
    targetClipDurationMinSec: override.targetClipDurationMinSec ?? base.targetClipDurationMinSec,
    targetClipDurationMaxSec: override.targetClipDurationMaxSec ?? base.targetClipDurationMaxSec,
  });
}
