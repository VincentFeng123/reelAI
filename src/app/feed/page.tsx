"use client";

import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

import { LoadingFlappyMiniGame } from "@/components/LoadingFlappyMiniGame";
import { ReelCard } from "@/components/ReelCard";
import { askStudyChat, fetchFeed, generateReels, sendFeedback, uploadMaterial } from "@/lib/api";
import {
  type GenerationMode,
  type PreferredVideoDuration,
  type VideoPoolMode,
  GENERATION_MODE_STORAGE_KEY,
  MUTED_STORAGE_KEY,
  readStudyReelsSettings,
} from "@/lib/settings";
import type { ChatMessage, Reel } from "@/lib/types";

const PAGE_SIZE = 5;
const REEL_SNAP_DURATION_MS = 300;
const POST_SNAP_COOLDOWN_MS = 30;
const WHEEL_GESTURE_RELEASE_MS = 220;
const WHEEL_DELTA_THRESHOLD = 110;
const WHEEL_REARM_DELTA_THRESHOLD = 8;
const TOUCH_GESTURE_COOLDOWN_MS = 30;
const RIGHT_PANEL_MIN_PX = 300;
const LEFT_PANEL_MIN_PX = 380;
const RIGHT_SPLIT_BAR_PX = 14;
const RIGHT_TOP_MIN_PX = 220;
const RIGHT_BOTTOM_MIN_PX = 180;
const MOBILE_DETAILS_CLOSE_MS = 240;
const MATERIAL_SEEDS_STORAGE_KEY = "studyreels-material-seeds";
const MATERIAL_GROUPS_STORAGE_KEY = "studyreels-material-groups";
const FEED_PROGRESS_STORAGE_KEY = "studyreels-feed-progress";
const FEED_SESSION_STORAGE_KEY = "studyreels-feed-sessions";
const HISTORY_STORAGE_KEY = "studyreels-material-history";
const MAX_SAVED_FEED_PROGRESS = 240;
const MAX_SAVED_FEED_SESSIONS = 24;
const MAX_REELS_PER_FEED_SESSION = 80;
const MAX_EMPTY_GENERATION_STREAK = 10;
const COMMUNITY_SET_FEED_HANDOFF_PREFIX = "studyreels-community-feed-handoff-";
type FeedbackAction = "helpful" | "confusing" | "save";

type FeedTuningSettings = {
  minRelevance: number;
  videoPoolMode: VideoPoolMode;
  preferredVideoDuration: PreferredVideoDuration;
  targetClipDurationSec: number;
  targetClipDurationMinSec: number;
  targetClipDurationMaxSec: number;
};

type ReelFeedbackState = {
  helpful?: boolean;
  confusing?: boolean;
  saved?: boolean;
  rating?: number;
};

type MaterialSeed = {
  topic?: string;
  text?: string;
  title?: string;
  updatedAt?: number;
};

type MaterialGroup = {
  materialIds: string[];
  title?: string;
  updatedAt?: number;
};

type FeedProgressEntry = {
  index: number;
  reelId?: string;
  updatedAt: number;
};

type FeedSessionSnapshot = {
  reels: Reel[];
  page: number;
  total: number;
  canRequestMore: boolean;
  generationMode: GenerationMode;
  mutedPreference: boolean;
  captionsPreference: boolean;
  activeIndex: number;
  activeReelId?: string;
  updatedAt: number;
};

type CommunityFeedHandoffPayload = {
  setId: string;
  setTitle: string;
  selectedReelId?: string;
  reels: Array<{
    id: string;
    platform: string;
    sourceUrl: string;
    embedUrl: string;
    tStartSec?: number;
    tEndSec?: number;
  }>;
};

function parseMaterialSeeds(raw: string | null): Record<string, MaterialSeed> {
  if (!raw) {
    return {};
  }
  try {
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return {};
    }
    return parsed as Record<string, MaterialSeed>;
  } catch {
    return {};
  }
}

function parseMaterialGroups(raw: string | null): Record<string, MaterialGroup> {
  if (!raw) {
    return {};
  }
  try {
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return {};
    }
    const result: Record<string, MaterialGroup> = {};
    for (const [materialId, value] of Object.entries(parsed as Record<string, unknown>)) {
      if (!materialId || typeof materialId !== "string") {
        continue;
      }
      if (!value || typeof value !== "object" || Array.isArray(value)) {
        continue;
      }
      const row = value as Record<string, unknown>;
      const materialIds = Array.isArray(row.materialIds)
        ? Array.from(new Set(row.materialIds.map((id) => String(id || "").trim()).filter(Boolean)))
        : [];
      if (materialIds.length === 0) {
        continue;
      }
      result[materialId] = {
        materialIds,
        title: typeof row.title === "string" && row.title.trim() ? row.title.trim() : undefined,
        updatedAt: Number(row.updatedAt) || 0,
      };
    }
    return result;
  } catch {
    return {};
  }
}

function parseFeedProgress(raw: string | null): Record<string, FeedProgressEntry> {
  if (!raw) {
    return {};
  }
  try {
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return {};
    }
    const result: Record<string, FeedProgressEntry> = {};
    for (const [materialId, value] of Object.entries(parsed as Record<string, unknown>)) {
      if (!materialId || typeof materialId !== "string") {
        continue;
      }
      if (!value || typeof value !== "object" || Array.isArray(value)) {
        continue;
      }
      const row = value as Record<string, unknown>;
      const index = Number(row.index);
      if (!Number.isFinite(index) || index < 0) {
        continue;
      }
      result[materialId] = {
        index: Math.floor(index),
        reelId: typeof row.reelId === "string" && row.reelId.trim() ? row.reelId.trim() : undefined,
        updatedAt: Number(row.updatedAt) || 0,
      };
    }
    return result;
  } catch {
    return {};
  }
}

function parseFeedSessions(raw: string | null): Record<string, FeedSessionSnapshot> {
  if (!raw) {
    return {};
  }
  try {
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return {};
    }
    const result: Record<string, FeedSessionSnapshot> = {};
    for (const [materialId, value] of Object.entries(parsed as Record<string, unknown>)) {
      if (!materialId || typeof materialId !== "string") {
        continue;
      }
      if (!value || typeof value !== "object" || Array.isArray(value)) {
        continue;
      }
      const row = value as Record<string, unknown>;
      const reels = Array.isArray(row.reels)
        ? row.reels
            .filter((item) => {
              if (!item || typeof item !== "object" || Array.isArray(item)) {
                return false;
              }
              const reel = item as Record<string, unknown>;
              return typeof reel.reel_id === "string" && reel.reel_id.trim() && typeof reel.video_url === "string";
            })
            .map((item) => item as Reel)
            .slice(-MAX_REELS_PER_FEED_SESSION)
        : [];
      const page = Math.max(1, Math.floor(Number(row.page) || 1));
      const total = Math.max(reels.length, Math.floor(Number(row.total) || reels.length));
      const activeIndex = Math.max(0, Math.floor(Number(row.activeIndex) || 0));
      result[materialId] = {
        reels,
        page,
        total,
        canRequestMore: row.canRequestMore !== false,
        generationMode: row.generationMode === "slow" ? "slow" : "fast",
        mutedPreference: row.mutedPreference !== false,
        captionsPreference: Boolean(row.captionsPreference),
        activeIndex,
        activeReelId: typeof row.activeReelId === "string" && row.activeReelId.trim() ? row.activeReelId.trim() : undefined,
        updatedAt: Number(row.updatedAt) || 0,
      };
    }
    return result;
  } catch {
    return {};
  }
}

function parseCommunityFeedHandoff(raw: string | null): CommunityFeedHandoffPayload | null {
  if (!raw) {
    return null;
  }
  try {
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return null;
    }
    const row = parsed as Record<string, unknown>;
    const setId = typeof row.setId === "string" ? row.setId.trim() : "";
    const setTitle = typeof row.setTitle === "string" ? row.setTitle.trim() : "";
    const selectedReelId = typeof row.selectedReelId === "string" ? row.selectedReelId.trim() : "";
    const reelsRaw = Array.isArray(row.reels) ? row.reels : [];
    const reels = reelsRaw
      .map((entry) => {
        if (!entry || typeof entry !== "object" || Array.isArray(entry)) {
          return null;
        }
        const reel = entry as Record<string, unknown>;
        const id = typeof reel.id === "string" ? reel.id.trim() : "";
        const platform = typeof reel.platform === "string" ? reel.platform.trim().toLowerCase() : "";
        const sourceUrl = typeof reel.sourceUrl === "string" ? reel.sourceUrl.trim() : "";
        const embedUrl = typeof reel.embedUrl === "string" ? reel.embedUrl.trim() : "";
        if (!id || !platform || !sourceUrl || !embedUrl) {
          return null;
        }
        const tStartRaw = Number(reel.tStartSec);
        const tEndRaw = Number(reel.tEndSec);
        const tStartSec = Number.isFinite(tStartRaw) && tStartRaw >= 0 ? tStartRaw : undefined;
        const tEndSec = Number.isFinite(tEndRaw) && tEndRaw > 0 ? tEndRaw : undefined;
        return { id, platform, sourceUrl, embedUrl, tStartSec, tEndSec };
      })
      .filter(Boolean) as CommunityFeedHandoffPayload["reels"];

    if (!setId || reels.length === 0) {
      return null;
    }
    return {
      setId,
      setTitle: setTitle || "Community Reel",
      selectedReelId: selectedReelId || undefined,
      reels,
    };
  } catch {
    return null;
  }
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function sanitizeAbsoluteUrl(value: string): string {
  const trimmed = value.trim();
  if (!trimmed) {
    return "";
  }
  try {
    const parsed = new URL(trimmed);
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
      return "";
    }
    parsed.hash = "";
    return parsed.toString();
  } catch {
    return "";
  }
}

function getCommunityPlatformLabel(value: string): string {
  const normalized = value.trim().toLowerCase();
  if (normalized === "youtube") {
    return "YouTube";
  }
  if (normalized === "instagram") {
    return "Instagram";
  }
  if (normalized === "tiktok") {
    return "TikTok";
  }
  return "Community";
}

function buildCommunityFeedReelId(setId: string, reelId: string): string {
  return `community:${encodeURIComponent(setId)}:${encodeURIComponent(reelId)}`;
}

function normalizeClipRange(startRaw: unknown, endRaw: unknown): { start: number; end: number; hasExplicitEnd: boolean } {
  const parsedStart = Number(startRaw);
  const parsedEnd = Number(endRaw);
  const hasStart = Number.isFinite(parsedStart) && parsedStart >= 0;
  const hasEnd = Number.isFinite(parsedEnd) && parsedEnd > 0;
  const start = hasStart ? parsedStart : 0;
  const end = hasEnd && parsedEnd > start ? parsedEnd : start + 180;
  const hasExplicitEnd = hasEnd && parsedEnd > start;
  return { start, end, hasExplicitEnd };
}

function buildCommunityFeedReel(params: {
  setId: string;
  setTitle: string;
  reelId: string;
  platform: string;
  reelUrl: string;
  sourceUrl: string;
  tStartSec?: unknown;
  tEndSec?: unknown;
}): Reel | null {
  const videoUrl = sanitizeAbsoluteUrl(params.reelUrl) || sanitizeAbsoluteUrl(params.sourceUrl);
  if (!videoUrl) {
    return null;
  }
  const sourceUrl = sanitizeAbsoluteUrl(params.sourceUrl);
  const safeSetId = params.setId.trim() || "community-set";
  const normalizedReelId = params.reelId.trim() || videoUrl;
  const title = params.setTitle.trim() || "Community Reel";
  const platformLabel = getCommunityPlatformLabel(params.platform);
  const sourceSnippet = sourceUrl || "Opened from Community Reels.";
  const clipRange = normalizeClipRange(params.tStartSec, params.tEndSec);
  const clipDurationSec = clipRange.hasExplicitEnd ? Math.max(0, clipRange.end - clipRange.start) : undefined;

  return {
    reel_id: buildCommunityFeedReelId(safeSetId, normalizedReelId),
    concept_id: safeSetId,
    concept_title: title,
    video_title: `${title} (${platformLabel})`,
    video_description: sourceUrl ? `Source: ${sourceUrl}` : `${platformLabel} reel`,
    ai_summary: "Imported from Community Reels.",
    video_url: videoUrl,
    t_start: clipRange.start,
    t_end: clipRange.end,
    ...(Number.isFinite(clipDurationSec) && Number(clipDurationSec) > 0 ? { clip_duration_sec: clipDurationSec } : {}),
    community_has_explicit_end: clipRange.hasExplicitEnd,
    transcript_snippet: sourceSnippet,
    takeaways: [],
    score: 1,
    relevance_score: 1,
    matched_terms: [],
    relevance_reason: "Opened from a community set.",
  };
}

function buildCommunityPreviewReel(params: {
  setId: string;
  setTitle: string;
  reelId: string;
  platform: string;
  reelUrl: string;
  sourceUrl: string;
  tStartSec?: string;
  tEndSec?: string;
}): Reel | null {
  return buildCommunityFeedReel({
    setId: params.setId,
    setTitle: params.setTitle,
    reelId: params.reelId,
    platform: params.platform,
    reelUrl: params.reelUrl,
    sourceUrl: params.sourceUrl,
    tStartSec: params.tStartSec,
    tEndSec: params.tEndSec,
  });
}

function buildCommunityFeedReelsFromHandoff(payload: CommunityFeedHandoffPayload): Reel[] {
  return payload.reels
    .map((reel) =>
      buildCommunityFeedReel({
        setId: payload.setId,
        setTitle: payload.setTitle,
        reelId: reel.id,
        platform: reel.platform,
        reelUrl: reel.embedUrl || reel.sourceUrl,
        sourceUrl: reel.sourceUrl,
        tStartSec: reel.tStartSec,
        tEndSec: reel.tEndSec,
      }),
    )
    .filter(Boolean) as Reel[];
}

function FeedPageInner() {
  const params = useSearchParams();
  const router = useRouter();
  const materialId = params.get("material_id") || "";
  const generationModeParam = params.get("generation_mode");
  const communitySetIdParam = params.get("community_set_id") || "";
  const communitySetTitleParam = params.get("community_set_title") || "";
  const communityReelIdParam = params.get("community_reel_id") || "";
  const communityReelPlatformParam = params.get("community_reel_platform") || "";
  const communityReelUrlParam = params.get("community_reel_url") || "";
  const communityReelSourceUrlParam = params.get("community_reel_source_url") || "";
  const communityStartSecParam = params.get("community_t_start_sec") || "";
  const communityEndSecParam = params.get("community_t_end_sec") || "";
  const communityHandoffIdParam = params.get("community_handoff_id") || "";
  const returnTabParam = params.get("return_tab") || "";
  const returnCommunitySetIdParam = params.get("return_community_set_id") || "";

  const communityPreviewReel = useMemo(
    () =>
      buildCommunityPreviewReel({
        setId: communitySetIdParam,
        setTitle: communitySetTitleParam,
        reelId: communityReelIdParam,
        platform: communityReelPlatformParam,
        reelUrl: communityReelUrlParam,
        sourceUrl: communityReelSourceUrlParam,
        tStartSec: communityStartSecParam,
        tEndSec: communityEndSecParam,
      }),
    [
      communityEndSecParam,
      communityReelIdParam,
      communityReelPlatformParam,
      communityReelSourceUrlParam,
      communityReelUrlParam,
      communityStartSecParam,
      communitySetIdParam,
      communitySetTitleParam,
    ],
  );

  const [reels, setReels] = useState<Reel[]>([]);
  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [invalidCommunityHandoff, setInvalidCommunityHandoff] = useState(false);
  const [generatingMore, setGeneratingMore] = useState(false);
  const [bootstrappingFirstReels, setBootstrappingFirstReels] = useState(false);
  const [canRequestMore, setCanRequestMore] = useState(true);
  const [activeIndex, setActiveIndex] = useState(0);
  const [feedbackByReel, setFeedbackByReel] = useState<Record<string, ReelFeedbackState>>({});
  const [pendingAction, setPendingAction] = useState<FeedbackAction | null>(null);
  const [mobileDetailsOpen, setMobileDetailsOpen] = useState(false);
  const [mobileDetailsClosing, setMobileDetailsClosing] = useState(false);
  const [mutedPreference, setMutedPreference] = useState(true);
  const [chatByReel, setChatByReel] = useState<Record<string, ChatMessage[]>>({});
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);
  const [rightPanelWidthPx, setRightPanelWidthPx] = useState(360);
  const [rightTopRatio, setRightTopRatio] = useState(0.62);
  const [generationMode, setGenerationMode] = useState<GenerationMode>("fast");
  const [captionsPreference, setCaptionsPreference] = useState(false);
  const [sessionHydrated, setSessionHydrated] = useState(false);

  const feedViewportRef = useRef<HTMLDivElement | null>(null);
  const isFetchingRef = useRef(false);
  const isGeneratingRef = useRef(false);
  const activeIndexRef = useRef(0);
  const stepLockUntilRef = useRef(0);
  const isTransitioningRef = useRef(false);
  const transitionUnlockTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const wheelAccumRef = useRef(0);
  const wheelGestureConsumedRef = useRef(false);
  const wheelReadyToRearmRef = useRef(false);
  const wheelGestureReleaseTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const touchStartYRef = useRef<number | null>(null);
  const emptyGenerateStreakRef = useRef(0);
  const bootstrapAttemptedRef = useRef(false);
  const desktopShellRef = useRef<HTMLDivElement | null>(null);
  const rightColumnRef = useRef<HTMLDivElement | null>(null);
  const dragModeRef = useRef<"lr" | "tb" | null>(null);
  const isRecoveringMissingMaterialRef = useRef(false);
  const recoveryAttemptedIdsRef = useRef<Set<string>>(new Set());
  const pendingResumeRef = useRef<FeedProgressEntry | null>(null);
  const resumeAppliedRef = useRef(false);
  const resumeLoadingRef = useRef(false);
  const isFastTopUpRef = useRef(false);
  const mobileDetailsCloseTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mutedRestoredFromSnapshotRef = useRef(false);
  const hydratedMaterialIdRef = useRef<string | null>(null);
  const materialIdsForFeedRef = useRef<string[]>([]);

  const normalizeClipKeyTime = useCallback((value: unknown): string => {
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) {
      return "0.00";
    }
    return (Math.round(parsed * 100) / 100).toFixed(2);
  }, []);

  const reelClipKey = useCallback((reel: Reel): string => {
    const match = reel.video_url.match(/\/embed\/([^?&/]+)/);
    const videoId = match?.[1] || reel.video_url;
    const start = normalizeClipKeyTime(reel.t_start);
    const end = normalizeClipKeyTime(reel.t_end);
    return `${videoId}:${start}:${end}`;
  }, [normalizeClipKeyTime]);

  const dedupeByIdentity = useCallback(
    (rows: Reel[], existing: Reel[] = []): Reel[] => {
      const seenClipKeys = new Set<string>();
      const seenReelIds = new Set<string>();
      const deduped: Reel[] = [];

      const pushIfUnique = (reel: Reel) => {
        const reelId = String(reel.reel_id || "").trim();
        const clipKey = reelClipKey(reel);
        if (reelId && seenReelIds.has(reelId)) {
          return;
        }
        if (seenClipKeys.has(clipKey)) {
          return;
        }
        if (reelId) {
          seenReelIds.add(reelId);
        }
        seenClipKeys.add(clipKey);
        deduped.push(reel);
      };

      for (const reel of existing) {
        pushIfUnique(reel);
      }
      for (const reel of rows) {
        pushIfUnique(reel);
      }
      return deduped;
    },
    [reelClipKey],
  );

  const interleaveReelBatches = useCallback((batches: Reel[][]): Reel[] => {
    if (batches.length <= 1) {
      return batches[0] ? [...batches[0]] : [];
    }
    const queues = batches.map((batch) => [...batch]);
    const merged: Reel[] = [];
    let added = true;
    while (added) {
      added = false;
      for (const queue of queues) {
        const next = queue.shift();
        if (!next) {
          continue;
        }
        merged.push(next);
        added = true;
      }
    }
    return merged;
  }, []);

  const getFeedMaterialIds = useCallback((): string[] => {
    const ids = materialIdsForFeedRef.current
      .map((id) => id.trim())
      .filter(Boolean);
    if (ids.length > 0) {
      return ids;
    }
    return materialId ? [materialId] : [];
  }, [materialId]);

  const getFeedTuningSettings = useCallback((): FeedTuningSettings => {
    const settings = readStudyReelsSettings();
    return {
      minRelevance: settings.minRelevanceThreshold,
      videoPoolMode: settings.videoPoolMode,
      preferredVideoDuration: settings.preferredVideoDuration,
      targetClipDurationSec: settings.targetClipDurationSec,
      targetClipDurationMinSec: settings.targetClipDurationMinSec,
      targetClipDurationMaxSec: settings.targetClipDurationMaxSec,
    };
  }, []);

  const hasMore = reels.length < total;
  const isFastGeneration = generationMode === "fast";

  useEffect(() => {
    if (generationModeParam === "fast" || generationModeParam === "slow") {
      setGenerationMode(generationModeParam);
      return;
    }
    if (typeof window === "undefined") {
      return;
    }
    const savedMode = window.localStorage.getItem(GENERATION_MODE_STORAGE_KEY);
    if (savedMode === "fast" || savedMode === "slow") {
      setGenerationMode(savedMode);
    }
  }, [generationModeParam]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    if (!materialId) {
      return;
    }
    try {
      const rawHistory = window.localStorage.getItem(HISTORY_STORAGE_KEY);
      if (!rawHistory) {
        return;
      }
      const parsed = JSON.parse(rawHistory);
      if (!Array.isArray(parsed)) {
        return;
      }
      let didChange = false;
      const updated = parsed.map((item) => {
        if (!item || typeof item !== "object") {
          return item;
        }
        const row = item as Record<string, unknown>;
        if (String(row.materialId || "") !== materialId) {
          return item;
        }
        if (row.generationMode === generationMode) {
          return item;
        }
        didChange = true;
        return {
          ...row,
          generationMode,
        };
      });
      if (didChange) {
        window.localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(updated));
      }
    } catch {
      // Ignore malformed history payloads and keep feed mode persistence functional.
    }
  }, [generationMode, materialId]);

  const setGenerationModeWithUrlSync = useCallback(
    (nextMode: GenerationMode) => {
      if (nextMode === generationMode) {
        return;
      }
      setGenerationMode(nextMode);
      if (!materialId) {
        return;
      }
      const nextParams = new URLSearchParams(params.toString());
      nextParams.set("material_id", materialId);
      nextParams.set("generation_mode", nextMode);
      router.replace(`/feed?${nextParams.toString()}`, { scroll: false });
    },
    [generationMode, materialId, params, router],
  );

  const recoverMissingMaterial = useCallback(
    async (missingMaterialId: string): Promise<boolean> => {
      if (typeof window === "undefined" || isRecoveringMissingMaterialRef.current) {
        return false;
      }

      const seeds = parseMaterialSeeds(window.localStorage.getItem(MATERIAL_SEEDS_STORAGE_KEY));
      const seed = seeds[missingMaterialId];
      const topic = String(seed?.topic || "").trim();
      const text = String(seed?.text || "").trim();
      if (!topic && !text) {
        return false;
      }

      isRecoveringMissingMaterialRef.current = true;
      setError("Session expired on server. Rebuilding your material...");
      try {
        const rebuilt = await uploadMaterial({
          subjectTag: topic || undefined,
          text: text || undefined,
        });
        const rebuiltId = rebuilt.material_id;
        seeds[rebuiltId] = {
          ...seed,
          topic: topic || undefined,
          text: text || undefined,
          updatedAt: Date.now(),
        };
        delete seeds[missingMaterialId];
        window.localStorage.setItem(MATERIAL_SEEDS_STORAGE_KEY, JSON.stringify(seeds));
        const groups = parseMaterialGroups(window.localStorage.getItem(MATERIAL_GROUPS_STORAGE_KEY));
        const nextGroups: Record<string, MaterialGroup> = {};
        for (const [groupId, group] of Object.entries(groups)) {
          const nextMaterialIds = Array.from(
            new Set(group.materialIds.map((id) => (id === missingMaterialId ? rebuiltId : id)).filter(Boolean)),
          );
          if (nextMaterialIds.length === 0) {
            continue;
          }
          const nextGroupId = groupId === missingMaterialId ? rebuiltId : groupId;
          nextGroups[nextGroupId] = {
            ...group,
            materialIds: nextMaterialIds,
            updatedAt: Date.now(),
          };
        }
        window.localStorage.setItem(MATERIAL_GROUPS_STORAGE_KEY, JSON.stringify(nextGroups));

        const nextParams = new URLSearchParams(params.toString());
        nextParams.set("material_id", rebuiltId);
        router.replace(`/feed?${nextParams.toString()}`);
        return true;
      } catch (e) {
        setError(e instanceof Error ? e.message : "Could not rebuild material.");
        return false;
      } finally {
        isRecoveringMissingMaterialRef.current = false;
      }
    },
    [params, router],
  );

  const loadPage = useCallback(
    async (targetPage: number, options?: { autofill?: boolean }) => {
      const feedMaterialIds = getFeedMaterialIds();
      if (feedMaterialIds.length === 0 || isFetchingRef.current) {
        return;
      }
      isFetchingRef.current = true;
      setError(null);

      try {
        const tuning = getFeedTuningSettings();
        const rows = await Promise.all(
          feedMaterialIds.map(async (id) => {
            try {
              const data = await fetchFeed({
                materialId: id,
                page: targetPage,
                limit: PAGE_SIZE,
                autofill: options?.autofill ?? true,
                prefetch: generationMode === "fast" ? 18 : 12,
                generationMode,
                minRelevance: tuning.minRelevance,
                videoPoolMode: tuning.videoPoolMode,
                preferredVideoDuration: tuning.preferredVideoDuration,
                targetClipDurationSec: tuning.targetClipDurationSec,
                targetClipDurationMinSec: tuning.targetClipDurationMinSec,
                targetClipDurationMaxSec: tuning.targetClipDurationMaxSec,
              });
              return { materialId: id, data, error: null };
            } catch (error) {
              return { materialId: id, data: null, error };
            }
          }),
        );

        const successful = rows.filter((row) => row.data);
        if (successful.length === 0) {
          if (targetPage === 1) {
            const firstError = rows[0]?.error;
            const message = firstError instanceof Error ? firstError.message : "Feed failed to load";
            if (
              feedMaterialIds.length === 1 &&
              /material_id not found/i.test(message) &&
              !recoveryAttemptedIdsRef.current.has(materialId)
            ) {
              recoveryAttemptedIdsRef.current.add(materialId);
              const recovered = await recoverMissingMaterial(materialId);
              if (recovered) {
                return;
              }
            }
            setError(message);
          }
          return;
        }

        const fetchedReels = dedupeByIdentity(interleaveReelBatches(successful.map((row) => row.data!.reels)));
        const fetchedTotal = successful.reduce((sum, row) => sum + Math.max(0, Number(row.data!.total) || 0), 0);

        setPage(targetPage);
        if (targetPage === 1) {
          setTotal(Math.max(fetchedTotal, fetchedReels.length));
        } else {
          setTotal((prevTotal) => Math.max(prevTotal, fetchedTotal, fetchedReels.length));
        }

        if (targetPage === 1) {
          setReels(fetchedReels);
        } else {
          setReels((prev) => dedupeByIdentity(fetchedReels, prev));
        }

        if (successful.length < rows.length) {
          const failedIds = rows.filter((row) => !row.data).map((row) => row.materialId);
          console.warn("Some topic feeds failed to load:", failedIds);
        }
      } catch (e) {
        if (targetPage === 1) {
          setError(e instanceof Error ? e.message : "Feed failed to load");
        }
      } finally {
        setLoading(false);
        isFetchingRef.current = false;
      }
    },
    [dedupeByIdentity, generationMode, getFeedMaterialIds, getFeedTuningSettings, interleaveReelBatches, materialId, recoverMissingMaterial],
  );

  const requestMore = useCallback(async (options?: { surfaceError?: boolean }): Promise<Reel[]> => {
    const feedMaterialIds = getFeedMaterialIds();
    if (feedMaterialIds.length === 0 || isGeneratingRef.current || !canRequestMore) {
      return [];
    }
    const tuning = getFeedTuningSettings();
    const batchSize = isFastGeneration ? 10 : 14;
    const perTopicBatch = Math.max(1, Math.ceil(batchSize / feedMaterialIds.length));
    isGeneratingRef.current = true;
    setGeneratingMore(true);
    if (options?.surfaceError) {
      setError(null);
    }
    try {
      const generatedRows = await Promise.all(
        feedMaterialIds.map(async (id) => {
          try {
            return await generateReels({
              materialId: id,
              numReels: perTopicBatch,
              generationMode,
              minRelevance: tuning.minRelevance,
              videoPoolMode: tuning.videoPoolMode,
              preferredVideoDuration: tuning.preferredVideoDuration,
              targetClipDurationSec: tuning.targetClipDurationSec,
              targetClipDurationMinSec: tuning.targetClipDurationMinSec,
              targetClipDurationMaxSec: tuning.targetClipDurationMaxSec,
            });
          } catch (e) {
            console.warn(`Background reel generation failed for topic material ${id}:`, e);
            return null;
          }
        }),
      );
      const generated = dedupeByIdentity(interleaveReelBatches(generatedRows.map((row) => row?.reels ?? [])));
      if (generated.length === 0) {
        emptyGenerateStreakRef.current += 1;
        if (emptyGenerateStreakRef.current >= MAX_EMPTY_GENERATION_STREAK) {
          setCanRequestMore(false);
        }
        if (options?.surfaceError) {
          setError("No reels were generated yet. Try a broader topic or add more source text.");
        }
        return [];
      }
      emptyGenerateStreakRef.current = 0;
      return generated;
    } catch (e) {
      console.warn("Background reel generation failed:", e);
      emptyGenerateStreakRef.current += 1;
      if (emptyGenerateStreakRef.current >= MAX_EMPTY_GENERATION_STREAK) {
        setCanRequestMore(false);
      }
      if (options?.surfaceError) {
        setError(e instanceof Error ? e.message : "Could not generate reels right now.");
      }
      return [];
    } finally {
      setGeneratingMore(false);
      isGeneratingRef.current = false;
    }
  }, [canRequestMore, dedupeByIdentity, generationMode, getFeedMaterialIds, getFeedTuningSettings, interleaveReelBatches, isFastGeneration]);

  useEffect(() => {
    mutedRestoredFromSnapshotRef.current = false;
    if (!materialId) {
      materialIdsForFeedRef.current = [];
      hydratedMaterialIdRef.current = null;
      setSessionHydrated(false);
      pendingResumeRef.current = null;
      resumeAppliedRef.current = false;
      resumeLoadingRef.current = false;
      let communityRows = communityPreviewReel ? [communityPreviewReel] : [];
      let preferredCommunityReelId = communityPreviewReel?.reel_id || "";
      let handoffMissing = false;
      if (typeof window !== "undefined" && communityHandoffIdParam) {
        const storageKey = `${COMMUNITY_SET_FEED_HANDOFF_PREFIX}${communityHandoffIdParam}`;
        const handoffPayload = parseCommunityFeedHandoff(window.sessionStorage.getItem(storageKey));
        if (handoffPayload) {
          const handoffRows = buildCommunityFeedReelsFromHandoff(handoffPayload);
          if (handoffRows.length > 0) {
            communityRows = handoffRows;
            const selectedReelId = handoffPayload.selectedReelId?.trim();
            preferredCommunityReelId =
              selectedReelId
                ? buildCommunityFeedReelId(handoffPayload.setId, selectedReelId)
                : handoffRows[0].reel_id;
          } else if (!communityPreviewReel) {
            handoffMissing = true;
          }
        } else if (!communityPreviewReel) {
          handoffMissing = true;
        }
        window.sessionStorage.removeItem(storageKey);
      }
      setInvalidCommunityHandoff(handoffMissing && communityRows.length === 0);
      setReels(communityRows);
      setPage(1);
      setTotal(communityRows.length);
      setCanRequestMore(false);
      if (communityRows.length === 0) {
        setActiveIndex(0);
      } else if (preferredCommunityReelId) {
        const preferredIndex = communityRows.findIndex((reel) => reel.reel_id === preferredCommunityReelId);
        setActiveIndex(preferredIndex >= 0 ? preferredIndex : 0);
      } else {
        setActiveIndex(0);
      }
      setFeedbackByReel({});
      setPendingAction(null);
      setChatByReel({});
      setChatInput("");
      setChatLoading(false);
      setChatError(null);
      setError(null);
      setMobileDetailsOpen(false);
      setBootstrappingFirstReels(false);
      setGeneratingMore(false);
      emptyGenerateStreakRef.current = 0;
      bootstrapAttemptedRef.current = false;
      setLoading(false);
      return;
    }
    setInvalidCommunityHandoff(false);
    if (hydratedMaterialIdRef.current === materialId) {
      return;
    }
    hydratedMaterialIdRef.current = materialId;
    setSessionHydrated(false);
    let resumeTarget: FeedProgressEntry | null = null;
    let restoredSession: FeedSessionSnapshot | null = null;
    let feedMaterialIds = [materialId];
    if (typeof window !== "undefined") {
      const allGroups = parseMaterialGroups(window.localStorage.getItem(MATERIAL_GROUPS_STORAGE_KEY));
      const groupedMaterialIds = allGroups[materialId]?.materialIds ?? [];
      feedMaterialIds = Array.from(new Set([materialId, ...groupedMaterialIds].map((id) => id.trim()).filter(Boolean)));
      const allProgress = parseFeedProgress(window.localStorage.getItem(FEED_PROGRESS_STORAGE_KEY));
      resumeTarget = allProgress[materialId] ?? null;
      const allSessions = parseFeedSessions(window.localStorage.getItem(FEED_SESSION_STORAGE_KEY));
      restoredSession = allSessions[materialId] ?? null;
    }
    materialIdsForFeedRef.current = Array.from(new Set(feedMaterialIds));
    pendingResumeRef.current = resumeTarget;
    resumeAppliedRef.current = false;
    resumeLoadingRef.current = false;
    recoveryAttemptedIdsRef.current.clear();
    setLoading(!restoredSession || restoredSession.reels.length === 0);
    setReels([]);
    setPage(1);
    setTotal(0);
    setCanRequestMore(true);
    setActiveIndex(0);
    setFeedbackByReel({});
    setMobileDetailsOpen(false);
    setBootstrappingFirstReels(false);
    emptyGenerateStreakRef.current = 0;
    bootstrapAttemptedRef.current = false;
    if (restoredSession) {
      const restoredReels = dedupeByIdentity(restoredSession.reels).slice(-MAX_REELS_PER_FEED_SESSION);
      const restoredIndex = restoredReels.length > 0 ? clamp(restoredSession.activeIndex, 0, restoredReels.length - 1) : 0;
      const restoredReelId = restoredReels[restoredIndex]?.reel_id;
      setReels(restoredReels);
      setPage(restoredSession.page);
      setTotal(Math.max(restoredSession.total, restoredReels.length));
      setCanRequestMore(restoredSession.canRequestMore);
      const modeFromQuery =
        generationModeParam === "fast" || generationModeParam === "slow" ? generationModeParam : null;
      setGenerationMode(modeFromQuery ?? restoredSession.generationMode);
      setMutedPreference(restoredSession.mutedPreference);
      setCaptionsPreference(restoredSession.captionsPreference);
      mutedRestoredFromSnapshotRef.current = true;
      if (restoredReels.length > 0) {
        const snapshotResume: FeedProgressEntry = {
          index: restoredIndex,
          reelId: restoredSession.activeReelId ?? restoredReelId,
          updatedAt: restoredSession.updatedAt || Date.now(),
        };
        if (!resumeTarget || snapshotResume.updatedAt >= (resumeTarget.updatedAt || 0)) {
          pendingResumeRef.current = snapshotResume;
        }
      }
      setSessionHydrated(true);
    }
    if (!restoredSession || restoredSession.reels.length === 0) {
      void loadPage(1, { autofill: true });
    }
  }, [communityHandoffIdParam, communityPreviewReel, dedupeByIdentity, generationModeParam, materialId, loadPage]);

  useEffect(() => {
    if (!materialId || sessionHydrated || loading) {
      return;
    }
    setSessionHydrated(true);
  }, [loading, materialId, sessionHydrated]);

  const appendGeneratedReels = useCallback(
    (generated: Reel[]) => {
      if (!generated.length) {
        return;
      }
      setReels((prev) => {
        const merged = dedupeByIdentity(generated, prev);
        const added = merged.length - prev.length;
        if (added > 0) {
          setTotal((prevTotal) => Math.max(prevTotal, merged.length));
        }
        return merged;
      });
    },
    [dedupeByIdentity],
  );

  const runFastTopUp = useCallback(async () => {
    const feedMaterialIds = getFeedMaterialIds();
    if (feedMaterialIds.length === 0 || generationMode !== "fast" || !canRequestMore || isFastTopUpRef.current || isGeneratingRef.current) {
      return;
    }
    const tuning = getFeedTuningSettings();
    const perTopicBatch = Math.max(1, Math.ceil(12 / feedMaterialIds.length));
    isFastTopUpRef.current = true;
    try {
      const generatedRows = await Promise.all(
        feedMaterialIds.map(async (id) => {
          try {
            return await generateReels({
              materialId: id,
              numReels: perTopicBatch,
              generationMode,
              minRelevance: tuning.minRelevance,
              videoPoolMode: tuning.videoPoolMode,
              preferredVideoDuration: tuning.preferredVideoDuration,
              targetClipDurationSec: tuning.targetClipDurationSec,
              targetClipDurationMinSec: tuning.targetClipDurationMinSec,
              targetClipDurationMaxSec: tuning.targetClipDurationMaxSec,
            });
          } catch (e) {
            console.warn(`Fast mode background top-up failed for topic material ${id}:`, e);
            return null;
          }
        }),
      );
      const generated = dedupeByIdentity(interleaveReelBatches(generatedRows.map((row) => row?.reels ?? [])));
      if (generated.length > 0) {
        emptyGenerateStreakRef.current = 0;
        appendGeneratedReels(generated);
      }
    } catch (e) {
      console.warn("Fast mode background top-up failed:", e);
    } finally {
      isFastTopUpRef.current = false;
    }
  }, [appendGeneratedReels, canRequestMore, dedupeByIdentity, generationMode, getFeedMaterialIds, getFeedTuningSettings, interleaveReelBatches]);

  const bootstrapFirstReels = useCallback(
    async (manual = false) => {
      if (!materialId || isGeneratingRef.current || !canRequestMore) {
        return;
      }
      setBootstrappingFirstReels(true);
      try {
        const generated = await requestMore({ surfaceError: manual });
        appendGeneratedReels(generated);
        if (generationMode === "fast" && generated.length > 0) {
          void runFastTopUp();
        }
      } finally {
        setBootstrappingFirstReels(false);
      }
    },
    [appendGeneratedReels, canRequestMore, generationMode, materialId, requestMore, runFastTopUp],
  );

  useEffect(() => {
    if (!materialId || loading || reels.length > 0 || bootstrapAttemptedRef.current) {
      return;
    }
    bootstrapAttemptedRef.current = true;
    void bootstrapFirstReels(false);
  }, [bootstrapFirstReels, loading, materialId, reels.length]);

  const maybeLoadMore = useCallback(() => {
    if (isFetchingRef.current) {
      return;
    }
    if (hasMore) {
      loadPage(page + 1, { autofill: true });
      return;
    }
    if (canRequestMore && !isGeneratingRef.current) {
      void (async () => {
        const generated = await requestMore();
        appendGeneratedReels(generated);
        if (generationMode === "fast" && generated.length > 0) {
          void runFastTopUp();
        }
      })();
    }
  }, [appendGeneratedReels, canRequestMore, generationMode, hasMore, loadPage, page, requestMore, runFastTopUp]);

  const shouldBlockDownwardAtEnd = useCallback(
    (direction: 1 | -1): boolean => {
      if (direction <= 0 || reels.length === 0) {
        return false;
      }
      if (activeIndexRef.current < reels.length - 1) {
        return false;
      }
      maybeLoadMore();
      wheelGestureConsumedRef.current = false;
      wheelReadyToRearmRef.current = false;
      wheelAccumRef.current = 0;
      return true;
    },
    [maybeLoadMore, reels.length],
  );

  const maybeResumeProgress = useCallback(() => {
    if (!materialId || resumeAppliedRef.current) {
      return;
    }
    const target = pendingResumeRef.current;
    if (!target) {
      resumeAppliedRef.current = true;
      return;
    }
    if (reels.length === 0) {
      return;
    }

    if (target.reelId) {
      const reelIndex = reels.findIndex((reel) => reel.reel_id === target.reelId);
      if (reelIndex >= 0) {
        setActiveIndex((prev) => (prev === reelIndex ? prev : reelIndex));
        resumeAppliedRef.current = true;
        return;
      }
    }

    const targetIndex = Math.max(0, Math.floor(target.index || 0));
    if (targetIndex < reels.length) {
      setActiveIndex((prev) => (prev === targetIndex ? prev : targetIndex));
      resumeAppliedRef.current = true;
      return;
    }

    if (resumeLoadingRef.current) {
      return;
    }

    if (hasMore) {
      resumeLoadingRef.current = true;
      void loadPage(page + 1, { autofill: true }).finally(() => {
        resumeLoadingRef.current = false;
      });
      return;
    }

    if (canRequestMore) {
      resumeLoadingRef.current = true;
      void (async () => {
        const generated = await requestMore();
        appendGeneratedReels(generated);
        if (generationMode === "fast" && generated.length > 0) {
          void runFastTopUp();
        }
      })().finally(() => {
        resumeLoadingRef.current = false;
      });
      return;
    }

    setActiveIndex((prev) => (reels.length > 0 ? Math.min(prev, reels.length - 1) : 0));
    resumeAppliedRef.current = true;
  }, [appendGeneratedReels, canRequestMore, generationMode, hasMore, loadPage, materialId, page, reels, requestMore, runFastTopUp]);

  useEffect(() => {
    maybeResumeProgress();
  }, [maybeResumeProgress]);

  useEffect(() => {
    setActiveIndex((prev) => {
      if (reels.length === 0) {
        return 0;
      }
      return Math.min(prev, reels.length - 1);
    });
  }, [reels.length]);

  useEffect(() => {
    activeIndexRef.current = activeIndex;
  }, [activeIndex]);

  useEffect(() => {
    if (typeof window === "undefined" || !materialId || reels.length === 0) {
      return;
    }
    const index = clamp(activeIndex, 0, reels.length - 1);
    const activeReel = reels[index];
    if (!activeReel?.reel_id) {
      return;
    }
    const allProgress = parseFeedProgress(window.localStorage.getItem(FEED_PROGRESS_STORAGE_KEY));
    allProgress[materialId] = {
      index,
      reelId: activeReel.reel_id,
      updatedAt: Date.now(),
    };
    const ordered = Object.entries(allProgress)
      .sort((a, b) => (b[1].updatedAt || 0) - (a[1].updatedAt || 0))
      .slice(0, MAX_SAVED_FEED_PROGRESS);
    window.localStorage.setItem(FEED_PROGRESS_STORAGE_KEY, JSON.stringify(Object.fromEntries(ordered)));
  }, [activeIndex, materialId, reels]);

  useEffect(() => {
    return () => {
      if (transitionUnlockTimerRef.current) {
        clearTimeout(transitionUnlockTimerRef.current);
        transitionUnlockTimerRef.current = null;
      }
      if (wheelGestureReleaseTimerRef.current) {
        clearTimeout(wheelGestureReleaseTimerRef.current);
        wheelGestureReleaseTimerRef.current = null;
      }
      if (mobileDetailsCloseTimerRef.current) {
        clearTimeout(mobileDetailsCloseTimerRef.current);
        mobileDetailsCloseTimerRef.current = null;
      }
      wheelReadyToRearmRef.current = false;
    };
  }, []);

  useEffect(() => {
    if (mobileDetailsCloseTimerRef.current) {
      clearTimeout(mobileDetailsCloseTimerRef.current);
      mobileDetailsCloseTimerRef.current = null;
    }
    setMobileDetailsClosing(false);
    setMobileDetailsOpen(false);
    setChatInput("");
    setChatError(null);
  }, [activeIndex]);

  const openMobileDetails = useCallback(() => {
    if (mobileDetailsCloseTimerRef.current) {
      clearTimeout(mobileDetailsCloseTimerRef.current);
      mobileDetailsCloseTimerRef.current = null;
    }
    setMobileDetailsClosing(false);
    setMobileDetailsOpen(true);
  }, []);

  const closeMobileDetails = useCallback(() => {
    if (!mobileDetailsOpen || mobileDetailsClosing) {
      return;
    }
    setMobileDetailsClosing(true);
    if (mobileDetailsCloseTimerRef.current) {
      clearTimeout(mobileDetailsCloseTimerRef.current);
    }
    mobileDetailsCloseTimerRef.current = setTimeout(() => {
      setMobileDetailsOpen(false);
      setMobileDetailsClosing(false);
      mobileDetailsCloseTimerRef.current = null;
    }, MOBILE_DETAILS_CLOSE_MS);
  }, [mobileDetailsClosing, mobileDetailsOpen]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    if (mutedRestoredFromSnapshotRef.current) {
      return;
    }
    const saved = window.localStorage.getItem(MUTED_STORAGE_KEY);
    if (saved === "0") {
      setMutedPreference(false);
    } else if (saved === "1") {
      setMutedPreference(true);
    }
  }, []);

  useEffect(() => {
    if (typeof window === "undefined" || !materialId || !sessionHydrated) {
      return;
    }
    const dedupedReels = dedupeByIdentity(reels).slice(-MAX_REELS_PER_FEED_SESSION);
    const index = dedupedReels.length > 0 ? clamp(activeIndex, 0, dedupedReels.length - 1) : 0;
    const activeReelId = dedupedReels[index]?.reel_id;
    const allSessions = parseFeedSessions(window.localStorage.getItem(FEED_SESSION_STORAGE_KEY));
    allSessions[materialId] = {
      reels: dedupedReels,
      page: Math.max(1, Math.floor(page || 1)),
      total: Math.max(total, dedupedReels.length),
      canRequestMore,
      generationMode,
      mutedPreference,
      captionsPreference,
      activeIndex: index,
      activeReelId,
      updatedAt: Date.now(),
    };
    const ordered = Object.entries(allSessions)
      .sort((a, b) => (b[1].updatedAt || 0) - (a[1].updatedAt || 0))
      .slice(0, MAX_SAVED_FEED_SESSIONS);
    window.localStorage.setItem(FEED_SESSION_STORAGE_KEY, JSON.stringify(Object.fromEntries(ordered)));
  }, [
    activeIndex,
    canRequestMore,
    captionsPreference,
    dedupeByIdentity,
    generationMode,
    materialId,
    mutedPreference,
    page,
    reels,
    sessionHydrated,
    total,
  ]);

  const stopDragging = useCallback(() => {
    dragModeRef.current = null;
    if (typeof document !== "undefined") {
      document.body.style.userSelect = "";
      document.body.style.cursor = "";
    }
  }, []);

  useEffect(() => {
    const onPointerMove = (event: PointerEvent) => {
      const mode = dragModeRef.current;
      if (!mode) {
        return;
      }

      if (mode === "lr") {
        const shell = desktopShellRef.current;
        if (!shell) {
          return;
        }
        const rect = shell.getBoundingClientRect();
        const pointerX = event.clientX - rect.left;
        const nextRight = clamp(rect.width - pointerX, RIGHT_PANEL_MIN_PX, rect.width - LEFT_PANEL_MIN_PX);
        setRightPanelWidthPx(nextRight);
        return;
      }

      const rightCol = rightColumnRef.current;
      if (!rightCol) {
        return;
      }
      const rect = rightCol.getBoundingClientRect();
      const available = rect.height - RIGHT_SPLIT_BAR_PX;
      if (available <= RIGHT_TOP_MIN_PX + RIGHT_BOTTOM_MIN_PX) {
        return;
      }
      const pointerY = event.clientY - rect.top;
      const topPx = clamp(pointerY, RIGHT_TOP_MIN_PX, available - RIGHT_BOTTOM_MIN_PX);
      setRightTopRatio(topPx / available);
    };

    const onPointerUp = () => {
      stopDragging();
    };

    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);
    window.addEventListener("pointercancel", onPointerUp);
    return () => {
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", onPointerUp);
      window.removeEventListener("pointercancel", onPointerUp);
    };
  }, [stopDragging]);

  const onStartLeftRightResize = useCallback((event: React.PointerEvent<HTMLButtonElement>) => {
    if (event.button !== 0) {
      return;
    }
    event.preventDefault();
    dragModeRef.current = "lr";
    document.body.style.userSelect = "none";
    document.body.style.cursor = "col-resize";
  }, []);

  const onStartTopBottomResize = useCallback((event: React.PointerEvent<HTMLButtonElement>) => {
    if (event.button !== 0) {
      return;
    }
    event.preventDefault();
    dragModeRef.current = "tb";
    document.body.style.userSelect = "none";
    document.body.style.cursor = "row-resize";
  }, []);

  useEffect(() => {
    return () => {
      stopDragging();
    };
  }, [stopDragging]);

  const resetFeedScrollPosition = useCallback(() => {
    const viewport = feedViewportRef.current;
    if (!viewport) {
      return;
    }
    if (viewport.scrollTop !== 0) {
      viewport.scrollTop = 0;
    }
    if (viewport.scrollLeft !== 0) {
      viewport.scrollLeft = 0;
    }
  }, []);

  const beginSnapTransitionLock = useCallback(() => {
    isTransitioningRef.current = true;
    if (transitionUnlockTimerRef.current) {
      clearTimeout(transitionUnlockTimerRef.current);
    }
    transitionUnlockTimerRef.current = setTimeout(() => {
      isTransitioningRef.current = false;
      stepLockUntilRef.current = Date.now() + POST_SNAP_COOLDOWN_MS;
      transitionUnlockTimerRef.current = null;
    }, REEL_SNAP_DURATION_MS);
  }, []);

  const jumpOneReel = useCallback(
    (direction: 1 | -1) => {
      if (reels.length === 0) {
        return;
      }
      const current = activeIndexRef.current;
      const next = Math.max(0, Math.min(reels.length - 1, current + direction));
      if (next === current) {
        if (direction > 0) {
          maybeLoadMore();
        }
        return;
      }

      beginSnapTransitionLock();
      setActiveIndex(next);
      if (next >= Math.max(reels.length - 2, 0)) {
        maybeLoadMore();
      }
    },
    [beginSnapTransitionLock, maybeLoadMore, reels.length],
  );

  const isControlTarget = useCallback((target: EventTarget | null): boolean => {
    if (!(target instanceof Element)) {
      return false;
    }
    return Boolean(target.closest("[data-reel-control='true']"));
  }, []);

  const handleFeedWheelNative = useCallback(
    (event: WheelEvent) => {
      if (reels.length === 0) {
        return;
      }
      if (isControlTarget(event.target)) {
        return;
      }
      event.preventDefault();
      resetFeedScrollPosition();

      if (isTransitioningRef.current) {
        return;
      }
      const now = Date.now();
      if (now < stepLockUntilRef.current) {
        return;
      }

      const dominantDelta = Math.abs(event.deltaY) >= Math.abs(event.deltaX) ? event.deltaY : event.deltaX;
      if (dominantDelta === 0) {
        return;
      }
      const normalizedDelta = Math.max(-100, Math.min(100, dominantDelta));

      if (wheelGestureReleaseTimerRef.current) {
        clearTimeout(wheelGestureReleaseTimerRef.current);
      }
      wheelGestureReleaseTimerRef.current = setTimeout(() => {
        wheelAccumRef.current = 0;
        wheelReadyToRearmRef.current = true;
        wheelGestureReleaseTimerRef.current = null;
      }, WHEEL_GESTURE_RELEASE_MS);

      if (wheelGestureConsumedRef.current) {
        if (wheelReadyToRearmRef.current) {
          // Re-arm only when wheel input has truly settled (tiny delta).
          if (Math.abs(normalizedDelta) > WHEEL_REARM_DELTA_THRESHOLD) {
            return;
          }
          wheelGestureConsumedRef.current = false;
          wheelReadyToRearmRef.current = false;
          wheelAccumRef.current = 0;
        }
        return;
      }

      if (wheelAccumRef.current !== 0 && Math.sign(wheelAccumRef.current) !== Math.sign(normalizedDelta)) {
        wheelAccumRef.current = normalizedDelta;
      } else {
        wheelAccumRef.current += normalizedDelta;
      }

      if (Math.abs(wheelAccumRef.current) < WHEEL_DELTA_THRESHOLD) {
        return;
      }

      const direction: 1 | -1 = wheelAccumRef.current > 0 ? 1 : -1;
      wheelAccumRef.current = 0;
      if (shouldBlockDownwardAtEnd(direction)) {
        return;
      }
      wheelGestureConsumedRef.current = true;
      jumpOneReel(direction);
    },
    [isControlTarget, jumpOneReel, reels.length, resetFeedScrollPosition, shouldBlockDownwardAtEnd],
  );

  useEffect(() => {
    const viewport = feedViewportRef.current;
    if (!viewport) {
      return;
    }
    const onWheel = (event: WheelEvent) => {
      handleFeedWheelNative(event);
    };
    viewport.addEventListener("wheel", onWheel, { passive: false });
    return () => {
      viewport.removeEventListener("wheel", onWheel);
    };
  }, [handleFeedWheelNative]);

  const onFeedTouchStart = useCallback(
    (event: React.TouchEvent<HTMLDivElement>) => {
      if (isControlTarget(event.target)) {
        touchStartYRef.current = null;
        return;
      }
      touchStartYRef.current = event.touches[0]?.clientY ?? null;
    },
    [isControlTarget],
  );

  const onFeedTouchMove = useCallback(
    (event: React.TouchEvent<HTMLDivElement>) => {
      if (isControlTarget(event.target)) {
        return;
      }
      event.preventDefault();
    },
    [isControlTarget],
  );

  const onFeedTouchEnd = useCallback(
    (event: React.TouchEvent<HTMLDivElement>) => {
      if (isControlTarget(event.target)) {
        touchStartYRef.current = null;
        return;
      }
      const startY = touchStartYRef.current;
      touchStartYRef.current = null;
      if (startY == null || reels.length === 0) {
        return;
      }

      const endY = event.changedTouches[0]?.clientY ?? startY;
      const deltaY = startY - endY;
      if (Math.abs(deltaY) < 28) {
        return;
      }
      const direction: 1 | -1 = deltaY > 0 ? 1 : -1;
      if (shouldBlockDownwardAtEnd(direction)) {
        return;
      }
      if (isTransitioningRef.current) {
        return;
      }

      const now = Date.now();
      if (now < stepLockUntilRef.current) {
        return;
      }
      stepLockUntilRef.current = now + TOUCH_GESTURE_COOLDOWN_MS;
      jumpOneReel(direction);
    },
    [isControlTarget, jumpOneReel, reels.length, shouldBlockDownwardAtEnd],
  );

  const activeReel = reels[activeIndex] ?? null;
  const atLastLoadedReel = reels.length > 0 && activeIndex >= reels.length - 1;
  const noMoreReelsAvailable =
    atLastLoadedReel &&
    reels.length > 0 &&
    !hasMore &&
    !canRequestMore &&
    !generatingMore &&
    !bootstrappingFirstReels &&
    !loading;
  const waitingForMoreReels =
    atLastLoadedReel &&
    reels.length > 0 &&
    !noMoreReelsAvailable &&
    (hasMore || generatingMore || bootstrappingFirstReels || canRequestMore);
  const activeVideoDescription = useMemo(() => {
    if (!activeReel) {
      return "";
    }
    const description = activeReel.video_description?.trim();
    if (description) {
      return description;
    }
    const snippet = activeReel.transcript_snippet?.trim();
    if (snippet) {
      return snippet;
    }
    return "No video description available for this reel.";
  }, [activeReel]);

  const activeAiSummary = useMemo(() => {
    if (!activeReel) {
      return "";
    }
    const aiSummary = activeReel.ai_summary?.trim();
    if (aiSummary) {
      return aiSummary;
    }
    const takeawaySummary = activeReel.takeaways
      .map((point) => point.trim())
      .filter(Boolean)
      .slice(0, 3)
      .join(" ");
    if (takeawaySummary) {
      return takeawaySummary;
    }
    const snippet = activeReel.transcript_snippet?.trim();
    if (snippet) {
      return snippet;
    }
    return "No AI summary available for this reel.";
  }, [activeReel]);

  const activeRelevanceReason = useMemo(() => {
    if (!activeReel) {
      return "";
    }
    const reason = activeReel.relevance_reason?.trim();
    if (reason) {
      return reason;
    }
    const terms = (activeReel.matched_terms ?? []).map((term) => term.trim()).filter(Boolean).slice(0, 5);
    if (terms.length > 0) {
      return `Matched terms from your material: ${terms.join(", ")}.`;
    }
    return "This reel was selected using semantic and keyword overlap with your uploaded material.";
  }, [activeReel]);

  const activeFeedback = useMemo(() => {
    if (!activeReel) {
      return {};
    }
    return feedbackByReel[activeReel.reel_id] ?? {};
  }, [activeReel, feedbackByReel]);

  const activeChatMessages = useMemo(() => {
    if (!activeReel) {
      return [];
    }
    return (
      chatByReel[activeReel.reel_id] ?? [
        {
          role: "assistant",
          content: "Ask me about this reel and I will break it down simply.",
        },
      ]
    );
  }, [activeReel, chatByReel]);

  const sendActiveChat = useCallback(async () => {
    if (!activeReel) {
      return;
    }
    const message = chatInput.trim();
    if (!message || chatLoading) {
      return;
    }

    setChatError(null);
    setChatLoading(true);

    const baseHistory =
      chatByReel[activeReel.reel_id] ?? [
        {
          role: "assistant" as const,
          content: "Ask me about this reel and I will break it down simply.",
        },
      ];
    const nextHistory: ChatMessage[] = [...baseHistory, { role: "user", content: message }];
    setChatByReel((prev) => ({ ...prev, [activeReel.reel_id]: nextHistory }));
    setChatInput("");

    try {
      const contextText = [
        activeReel.video_description,
        activeReel.ai_summary,
        activeReel.relevance_reason,
        ...(activeReel.matched_terms ?? []),
        activeReel.transcript_snippet,
        ...activeReel.takeaways,
      ]
        .filter(Boolean)
        .join("\n");
      const result = await askStudyChat({
        message,
        topic: activeReel.concept_title,
        text: contextText,
        history: nextHistory.slice(-8),
      });
      setChatByReel((prev) => {
        const current = prev[activeReel.reel_id] ?? nextHistory;
        return {
          ...prev,
          [activeReel.reel_id]: [...current, { role: "assistant", content: result.answer }],
        };
      });
    } catch (e) {
      setChatError(e instanceof Error ? e.message : "Chat failed");
      setChatByReel((prev) => {
        const current = prev[activeReel.reel_id] ?? nextHistory;
        return {
          ...prev,
          [activeReel.reel_id]: [...current, { role: "assistant", content: "I could not reply right now. Try again." }],
        };
      });
    } finally {
      setChatLoading(false);
    }
  }, [activeReel, chatByReel, chatInput, chatLoading]);

  const submitActiveFeedback = useCallback(
    async (action: FeedbackAction) => {
      if (!activeReel) {
        return;
      }

      const current = activeFeedback;
      let payload: ReelFeedbackState;
      if (action === "helpful") {
        const nextHelpful = !Boolean(current.helpful);
        payload = nextHelpful
          ? { helpful: true, confusing: false, rating: 5 }
          : { helpful: false, confusing: false, rating: 3 };
      } else if (action === "confusing") {
        const nextConfusing = !Boolean(current.confusing);
        payload = nextConfusing
          ? { helpful: false, confusing: true, rating: 2 }
          : { helpful: false, confusing: false, rating: 3 };
      } else {
        const nextSaved = !Boolean(current.saved);
        payload = { saved: nextSaved };
      }

      setPendingAction(action);
      setError(null);
      try {
        await sendFeedback({ reelId: activeReel.reel_id, ...payload });
        setFeedbackByReel((prev) => ({
          ...prev,
          [activeReel.reel_id]: {
            ...prev[activeReel.reel_id],
            ...payload,
          },
        }));
      } catch (e) {
        setError(e instanceof Error ? e.message : "Could not save feedback");
      } finally {
        setPendingAction(null);
      }
    },
    [activeFeedback, activeReel],
  );

  const renderGenerationModeToggle = (className?: string) => (
    <div
      role="group"
      aria-label="Generation mode"
      className={[
        "relative grid h-10 w-[128px] grid-cols-2 items-center overflow-hidden rounded-2xl border border-white/25 bg-white/[0.06] p-1 text-[10px] font-semibold uppercase tracking-[0.06em] text-white backdrop-blur-lg",
        className,
      ]
        .filter(Boolean)
        .join(" ")}
    >
      <span aria-hidden="true" className="pointer-events-none absolute inset-0 bg-black/45" />
      <span
        aria-hidden="true"
        className={`pointer-events-none absolute bottom-1 left-1 top-1 z-10 w-[calc(50%-4px)] rounded-xl bg-white transition-transform duration-300 ease-out ${
          generationMode === "fast" ? "translate-x-full" : "translate-x-0"
        }`}
      />
      <button
        type="button"
        onClick={() => setGenerationModeWithUrlSync("slow")}
        className={`relative z-10 rounded-xl px-2 py-1 transition-colors ${generationMode === "slow" ? "text-black" : "text-white/82"}`}
        aria-pressed={generationMode === "slow"}
      >
        Slow
      </button>
      <button
        type="button"
        onClick={() => setGenerationModeWithUrlSync("fast")}
        className={`relative z-10 rounded-xl px-2 py-1 transition-colors ${generationMode === "fast" ? "text-black" : "text-white/82"}`}
        aria-pressed={generationMode === "fast"}
      >
        Fast
      </button>
    </div>
  );
  const renderMobileFeedbackButton = (action: FeedbackAction, label: string, iconClass: string, active: boolean) => (
    <button
      type="button"
      onClick={() => submitActiveFeedback(action)}
      className={`relative grid h-10 w-10 place-items-center overflow-hidden rounded-2xl border-[0.8px] text-sm transition backdrop-blur-lg ${
        active ? "border-white/70 text-white" : "border-white/35 text-white/90 hover:border-white/55"
      }`}
      disabled={pendingAction !== null}
      aria-label={label}
      title={label}
    >
      <span aria-hidden="true" className={`pointer-events-none absolute inset-0 ${active ? "bg-black/55" : "bg-black/45"}`} />
      {pendingAction === action ? (
        <i className="fa-solid fa-spinner fa-spin relative z-10" aria-hidden="true" />
      ) : (
        <i className={`fa-solid ${iconClass} relative z-10`} aria-hidden="true" />
      )}
    </button>
  );
  const rightTopPercent = Math.round(rightTopRatio * 1000) / 10;
  const feedFallbackPath = useMemo(() => {
    if (returnTabParam === "search") {
      return "/?tab=search";
    }
    const returnSetId = (returnCommunitySetIdParam.trim() || communitySetIdParam.trim());
    if (returnTabParam === "community" || returnSetId || communityPreviewReel || communityHandoffIdParam) {
      const nextParams = new URLSearchParams();
      nextParams.set("tab", "community");
      if (returnSetId) {
        nextParams.set("community_set_id", returnSetId);
      }
      return `/?${nextParams.toString()}`;
    }
    return "/?tab=search";
  }, [communityHandoffIdParam, communityPreviewReel, communitySetIdParam, returnCommunitySetIdParam, returnTabParam]);

  const navigateBackToPreviousPage = useCallback(() => {
    if (returnTabParam === "search" || returnTabParam === "community") {
      router.push(feedFallbackPath);
      return;
    }
    if (typeof window !== "undefined") {
      const hasHistory = window.history.length > 1;
      let hasSameOriginReferrer = false;
      try {
        if (!document.referrer) {
          hasSameOriginReferrer = true;
        } else {
          hasSameOriginReferrer = new URL(document.referrer).origin === window.location.origin;
        }
      } catch {
        hasSameOriginReferrer = false;
      }
      if (hasHistory && hasSameOriginReferrer) {
        router.back();
        return;
      }
    }
    router.push(feedFallbackPath);
  }, [feedFallbackPath, returnTabParam, router]);

  if (!materialId && !communityPreviewReel && invalidCommunityHandoff) {
    return (
      <main className="fixed inset-0 px-6 md:inset-4">
        <div className="flex h-full items-center justify-center">
          <div className="rounded-3xl border border-white/25 bg-black/60 p-6 text-center text-white backdrop-blur-sm">
            <p className="text-sm">Community reel preview expired. Reopen this set from the Community tab.</p>
            <button
              className="mt-4 rounded-2xl border border-white/25 bg-white px-4 py-2 text-xs font-semibold text-black"
              onClick={navigateBackToPreviousPage}
            >
              Back to Community
            </button>
          </div>
        </div>
      </main>
    );
  }

  if (!materialId && !communityPreviewReel && !communityHandoffIdParam) {
    return (
      <main className="fixed inset-0 px-6 md:inset-4">
        <div className="flex h-full items-center justify-center">
          <div className="rounded-3xl border border-white/25 bg-black/60 p-6 text-center text-white backdrop-blur-sm">
            <p className="text-sm">Missing material_id or community reel selection.</p>
            <button
              className="mt-4 rounded-2xl border border-white/25 bg-white px-4 py-2 text-xs font-semibold text-black"
              onClick={navigateBackToPreviousPage}
            >
              Back to Upload
            </button>
          </div>
        </div>
      </main>
    );
  }

  return (
    <main className="fixed inset-0 overflow-visible md:inset-4 md:overflow-hidden">
      <button
        type="button"
        onClick={navigateBackToPreviousPage}
        aria-label="Back to main page"
        className="absolute left-3 top-3 z-[9999] grid h-9 w-9 place-items-center rounded-xl border border-white/20 bg-black/50 text-white shadow-[0_8px_24px_rgba(0,0,0,0.35)] backdrop-blur-md transition hover:bg-white/12"
      >
        <i className="fa-solid fa-arrow-left text-xs" aria-hidden="true" />
      </button>
      <div className="absolute right-3 top-3 z-[9999] lg:hidden">
        {renderGenerationModeToggle("shadow-[0_8px_24px_rgba(0,0,0,0.35)]")}
      </div>
      {error ? (
        <div className="absolute left-0 right-0 top-3 z-[2147483647] mx-auto w-fit">
          <div className="relative overflow-hidden rounded-xl border border-gray-300/45 bg-white/10 px-4 py-2 text-xs text-white shadow-[0_12px_28px_rgba(0,0,0,0.35)] backdrop-blur-xl backdrop-saturate-150">
            <div aria-hidden="true" className="pointer-events-none absolute inset-0 bg-black/45" />
            <span className="relative">{error}</span>
          </div>
        </div>
      ) : null}

      <div ref={desktopShellRef} className="h-full min-h-[100dvh] md:min-h-0 lg:flex">
        <section className="relative h-[100dvh] min-h-[100dvh] md:h-full md:min-h-0 lg:min-w-0 lg:flex-1">
          <div className="absolute right-3 top-3 z-30 hidden lg:block">
            {renderGenerationModeToggle("shadow-[0_8px_24px_rgba(0,0,0,0.35)]")}
          </div>
          {activeReel && !mobileDetailsOpen ? (
            <div className="absolute right-3 top-1/2 z-30 flex -translate-y-1/2 flex-col gap-2">
              {renderMobileFeedbackButton("helpful", "Helpful", "fa-thumbs-up", Boolean(activeFeedback.helpful))}
              {renderMobileFeedbackButton("confusing", "Confusing", "fa-circle-question", Boolean(activeFeedback.confusing))}
              {renderMobileFeedbackButton("save", "Save", "fa-bookmark", Boolean(activeFeedback.saved))}
            </div>
          ) : null}
          <div
            ref={feedViewportRef}
            onTouchStart={onFeedTouchStart}
            onTouchMove={onFeedTouchMove}
            onTouchEnd={onFeedTouchEnd}
            className="reel-scroll m-4 h-[calc(100dvh-2rem)] min-h-[calc(100dvh-2rem)] overflow-hidden rounded-3xl overscroll-none touch-none md:m-0 md:h-full md:min-h-0 lg:h-full lg:min-h-0"
          >
            <div
              className="flex h-full flex-col transition-transform duration-300 ease-out"
              style={{ transform: `translate3d(0, -${activeIndex * 100}%, 0)` }}
            >
              {reels.map((reel, index) => (
                <div key={reel.reel_id} className="h-[calc(100dvh-2rem)] shrink-0 grow-0 basis-full md:h-full lg:h-full">
                  <ReelCard
                    reel={reel}
                    isActive={index === activeIndex}
                    mutedPreference={mutedPreference}
                    onMutedPreferenceChange={setMutedPreference}
                    captionsEnabled={captionsPreference}
                    onCaptionsEnabledChange={setCaptionsPreference}
                    onOpenContent={index === activeIndex ? openMobileDetails : undefined}
                  />
                </div>
              ))}
            </div>
            {reels.length === 0 ? (
              <div className="absolute inset-0 grid place-items-center p-6">
                <div className="max-w-sm rounded-3xl border border-white/20 bg-black/68 px-5 py-4 text-center text-white backdrop-blur">
                  {loading || bootstrappingFirstReels || generatingMore ? (
                    <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-white/62">Loading feed</p>
                  ) : null}
                  <p className="text-sm font-semibold">
                    {loading || bootstrappingFirstReels || generatingMore ? "Preparing your first reels..." : "No reels yet"}
                  </p>
                  <p className="mt-2 text-xs text-white/72">
                    {loading || bootstrappingFirstReels || generatingMore
                      ? "This can take a little while on first generation."
                      : "Try generating again, or adjust your topic/material for broader matches."}
                  </p>
                  {!loading && !bootstrappingFirstReels && !generatingMore ? (
                    <button
                      type="button"
                      onClick={() => void bootstrapFirstReels(true)}
                      className="mt-3 rounded-xl border border-white/25 bg-white px-3.5 py-2 text-xs font-semibold text-black"
                    >
                      Generate Reels
                    </button>
                  ) : null}
                  {loading || bootstrappingFirstReels || generatingMore ? <LoadingFlappyMiniGame /> : null}
                </div>
              </div>
            ) : null}
            {reels.length > 0 && waitingForMoreReels ? (
              <div className="pointer-events-none absolute inset-x-0 bottom-4 z-20 flex justify-center px-4">
                <div className="rounded-full border border-white/20 bg-black/72 px-4 py-2 text-[11px] font-semibold uppercase tracking-[0.12em] text-white/80 backdrop-blur-sm">
                  Finding more reels...
                </div>
              </div>
            ) : null}
            {reels.length > 0 && noMoreReelsAvailable ? (
              <div className="pointer-events-none absolute inset-x-0 bottom-4 z-20 flex justify-center px-4">
                <div className="rounded-full border border-white/20 bg-black/72 px-4 py-2 text-[11px] font-semibold uppercase tracking-[0.12em] text-white/78 backdrop-blur-sm">
                  No more reels found for this topic.
                </div>
              </div>
            ) : null}
          </div>

          {mobileDetailsOpen && activeReel ? (
            <div className="absolute inset-0 z-30 lg:hidden">
              <button
                aria-label="Close content panel"
                className="absolute inset-0 bg-black/55"
                onClick={closeMobileDetails}
              />
              <div
                className={`absolute inset-x-4 bottom-8 max-h-[80svh] touch-pan-y overflow-y-auto overscroll-y-contain rounded-3xl border border-white/20 bg-black/92 px-4 pt-4 pb-4 text-white backdrop-blur [-webkit-overflow-scrolling:touch] ${
                  mobileDetailsClosing ? "animate-mobile-sheet-out" : "animate-mobile-sheet-in"
                }`}
              >
                <button
                  type="button"
                  onClick={closeMobileDetails}
                  aria-label="Close content panel"
                  className="absolute right-3 top-3 grid h-8 w-8 place-items-center text-base text-white/85 transition hover:text-white"
                >
                  <i className="fa-solid fa-xmark" aria-hidden="true" />
                </button>
                <h2 className="pr-8 text-xl font-bold leading-tight">{activeReel.concept_title}</h2>

                <div className="mt-3 min-w-0 rounded-2xl border border-white/20 bg-black/55 p-3 text-sm text-white/90">
                  <p className="mb-1 text-[10px] font-semibold uppercase tracking-[0.1em] text-white/60">Video Description</p>
                  <p className="whitespace-pre-line break-words [overflow-wrap:anywhere]">{activeVideoDescription}</p>
                </div>

                <div className="mt-3 min-w-0 rounded-2xl border border-white/20 bg-black/55 p-3 text-sm text-white/90">
                  <p className="mb-1 text-[10px] font-semibold uppercase tracking-[0.1em] text-white/60">AI Summary</p>
                  <p className="break-words [overflow-wrap:anywhere]">{activeAiSummary}</p>
                </div>

                <div className="mt-3 min-w-0 rounded-2xl border border-white/20 bg-black/55 p-3 text-sm text-white/90">
                  <div className="mb-1 flex items-center justify-between gap-2">
                    <p className="text-[10px] font-semibold uppercase tracking-[0.1em] text-white/60">Why This Matches</p>
                    {typeof activeReel.relevance_score === "number" ? (
                      <span className="text-[10px] font-semibold uppercase tracking-[0.08em] text-white/65">
                        Score {activeReel.relevance_score.toFixed(2)}
                      </span>
                    ) : null}
                  </div>
                  <p className="break-words [overflow-wrap:anywhere]">{activeRelevanceReason}</p>
                  {(activeReel.matched_terms?.length ?? 0) > 0 ? (
                    <div className="mt-2 flex flex-wrap gap-1.5">
                      {(activeReel.matched_terms ?? []).slice(0, 6).map((term) => (
                        <span
                          key={`mobile-match-${activeReel.reel_id}-${term}`}
                          className="rounded-full border border-white/20 bg-black/60 px-2 py-0.5 text-[10px] text-white/78"
                        >
                          {term}
                        </span>
                      ))}
                    </div>
                  ) : null}
                </div>

              </div>
            </div>
          ) : null}
        </section>

        <button
          type="button"
          aria-label="Resize panels"
          onPointerDown={onStartLeftRightResize}
          className="hidden w-3 cursor-col-resize touch-none select-none items-center justify-center lg:flex"
        >
          <span className="h-20 w-px rounded-full bg-white/35" />
        </button>

        <div
          ref={rightColumnRef}
          className="hidden h-full min-h-0 flex-none lg:grid"
          style={{
            width: `${Math.round(rightPanelWidthPx)}px`,
            minWidth: `${RIGHT_PANEL_MIN_PX}px`,
            maxWidth: "52vw",
            gridTemplateRows: `${rightTopPercent}% ${RIGHT_SPLIT_BAR_PX}px minmax(0, 1fr)`,
          }}
        >
          <aside className="min-h-0 min-w-0 overflow-y-auto rounded-3xl border border-white/20 bg-black/72 px-5 pt-5 pb-2 text-white">
            {!activeReel ? (
              <div className="flex h-full items-center justify-center text-sm text-white/80">Loading reel details...</div>
            ) : (
              <div className="flex min-h-full flex-col pb-2">
                <h2 className="text-2xl font-bold leading-tight">{activeReel.concept_title}</h2>

                <div className="mt-3 min-w-0 rounded-2xl border border-white/20 bg-black/55 p-3 text-sm text-white/90">
                  <p className="mb-1 text-[10px] font-semibold uppercase tracking-[0.1em] text-white/60">Video Description</p>
                  <p className="whitespace-pre-line break-words [overflow-wrap:anywhere]">{activeVideoDescription}</p>
                </div>

                <div className="mt-3 min-w-0 rounded-2xl border border-white/20 bg-black/55 p-3 text-sm text-white/90">
                  <p className="mb-1 text-[10px] font-semibold uppercase tracking-[0.1em] text-white/60">AI Summary</p>
                  <p className="break-words [overflow-wrap:anywhere]">{activeAiSummary}</p>
                </div>

                <div className="mt-3 mb-0 min-w-0 rounded-2xl border border-white/20 bg-black/55 p-3 text-sm text-white/90">
                  <div className="mb-1 flex items-center justify-between gap-2">
                    <p className="text-[10px] font-semibold uppercase tracking-[0.1em] text-white/60">Why This Matches</p>
                    {typeof activeReel.relevance_score === "number" ? (
                      <span className="text-[10px] font-semibold uppercase tracking-[0.08em] text-white/65">
                        Score {activeReel.relevance_score.toFixed(2)}
                      </span>
                    ) : null}
                  </div>
                  <p className="break-words [overflow-wrap:anywhere]">{activeRelevanceReason}</p>
                  {(activeReel.matched_terms?.length ?? 0) > 0 ? (
                    <div className="mt-2 flex flex-wrap gap-1.5">
                      {(activeReel.matched_terms ?? []).slice(0, 6).map((term) => (
                        <span
                          key={`desktop-match-${activeReel.reel_id}-${term}`}
                          className="rounded-full border border-white/20 bg-black/60 px-2 py-0.5 text-[10px] text-white/78"
                        >
                          {term}
                        </span>
                      ))}
                    </div>
                  ) : null}
                </div>
                <div aria-hidden="true" className="h-1 shrink-0" />

              </div>
            )}
          </aside>

          <button
            type="button"
            aria-label="Resize right panels"
            onPointerDown={onStartTopBottomResize}
            className="group grid h-[14px] cursor-row-resize touch-none select-none place-items-center bg-transparent"
          >
            <span className="h-px w-20 rounded-full bg-white/35 transition group-hover:bg-white/55" />
          </button>

          <section className="flex min-h-0 min-w-0 flex-col rounded-3xl border border-white/20 bg-black/62 px-4 pt-4 pb-0 text-white">
            <div className="mb-2 flex items-center justify-between">
              <p className="text-xs font-semibold uppercase tracking-[0.11em] text-white/75">AI Chat</p>
            </div>

            {!activeReel ? (
              <div className="flex h-full items-center justify-center text-sm text-white/70">Open a reel to chat.</div>
            ) : (
              <>
                <div className="min-h-0 flex-1 space-y-2 overflow-y-auto rounded-2xl bg-black/45 p-2.5">
                  {activeChatMessages.map((msg, idx) => (
                    <div
                      key={`${activeReel.reel_id}-chat-${idx}`}
                      className={`rounded-xl px-3 py-2 text-sm leading-relaxed break-words [overflow-wrap:anywhere] ${
                        msg.role === "user"
                          ? "ml-auto w-fit max-w-[85%] bg-[#1b1b1b] text-right text-white"
                          : "mr-8 bg-white/10 text-white/92"
                      }`}
                    >
                      {msg.content}
                    </div>
                  ))}
                  {chatLoading ? (
                    <div className="mr-8 rounded-xl bg-white/10 px-3 py-2 text-sm text-white/76">Thinking...</div>
                  ) : null}
                </div>

                <div className="mt-auto mb-4 flex items-center gap-2">
                  <input
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        e.preventDefault();
                        void sendActiveChat();
                      }
                    }}
                    placeholder="Ask about this reel..."
                    className="h-9 flex-1 rounded-xl border border-white/20 bg-black/55 px-3 text-sm text-white outline-none placeholder:text-white/45 focus:border-white/45"
                  />
                  <button
                    type="button"
                    onClick={() => void sendActiveChat()}
                    disabled={chatLoading || !chatInput.trim()}
                    aria-label="Send message"
                    className="grid h-9 w-9 place-items-center rounded-xl border border-white/25 bg-white/12 text-white transition hover:bg-white/18 disabled:opacity-45"
                  >
                    <i className="fa-solid fa-arrow-up text-xs" aria-hidden="true" />
                  </button>
                </div>
                {chatError ? <p className="mb-4 text-xs text-white/65">{chatError}</p> : null}
              </>
            )}
          </section>
        </div>
      </div>
    </main>
  );
}

export default function FeedPage() {
  return (
    <Suspense
      fallback={
        <main className="fixed inset-0 flex items-center justify-center text-sm text-white md:inset-4">
          Loading feed...
        </main>
      }
    >
      <FeedPageInner />
    </Suspense>
  );
}
