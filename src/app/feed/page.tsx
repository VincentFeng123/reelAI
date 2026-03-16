"use client";

import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

import { FullscreenLoadingScreen } from "@/components/FullscreenLoadingScreen";
import { ReelCard } from "@/components/ReelCard";
import {
  COMMUNITY_AUTH_CHANGED_EVENT,
  askStudyChat,
  clearCommunityAuthSession,
  fetchCommunitySettings,
  fetchFeed,
  fetchRefinementStatus,
  generateReelsStream,
  isRequestInterruptedError,
  isSessionExpiredError,
  queueCommunityHistorySync,
  readCommunityAuthSession,
  sendFeedback,
  uploadMaterial,
} from "@/lib/api";
import { applySearchFeedSettingsToParams, mergeSearchFeedQuerySettings, readSearchFeedQuerySettings } from "@/lib/feedQuery";
import {
  HISTORY_STORAGE_KEY,
  normalizeStoredHistoryItems as normalizeHistoryStorageItems,
  type StoredHistoryItem,
  writeScopedHistorySnapshot,
} from "@/lib/historyStorage";
import { useLoadingScreenGate } from "@/lib/useLoadingScreenGate";
import {
  type GenerationMode,
  type PreferredVideoDuration,
  type StudyReelsSettings,
  type VideoPoolMode,
  readStudyReelsSettings,
  setActiveStudyReelsSettingsScope,
} from "@/lib/settings";
import type { ChatMessage, Reel } from "@/lib/types";

const PAGE_SIZE = 5;
const INITIAL_FAST_PREFETCH = 6;
const INITIAL_SLOW_PREFETCH = 4;
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
const MAX_SAVED_FEED_PROGRESS = 240;
const MAX_SAVED_FEED_SESSIONS = 24;
const MAX_HISTORY_ITEMS = 120;
const MAX_REELS_PER_FEED_SESSION = 80;
const COMPACT_REELS_PER_FEED_SESSION = 48;
const MINIMAL_REELS_PER_FEED_SESSION = 20;
const MAX_EMPTY_GENERATION_STREAK = 10;
const REFINEMENT_POLL_INTERVAL_MS = 3000;
const COMMUNITY_SET_FEED_HANDOFF_PREFIX = "studyreels-community-feed-handoff-";
const DESCRIPTION_PREVIEW_CHAR_LIMIT = 180;
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

type FeedSearchScope = {
  key: string;
  seq: number;
  controller: AbortController;
};

type ExpandableTextProps = {
  text: string;
  expanded: boolean;
  onToggle: () => void;
  className?: string;
  previewChars?: number;
  forceExpandable?: boolean;
  loading?: boolean;
};

function buildExpandablePreview(text: string, previewChars: number): { preview: string; truncated: boolean } {
  const normalized = text.trim();
  if (normalized.length <= previewChars) {
    return { preview: normalized, truncated: false };
  }
  const sliced = normalized.slice(0, previewChars);
  const lastBoundary = Math.max(sliced.lastIndexOf(" "), sliced.lastIndexOf("\n"));
  const preview = (lastBoundary >= Math.floor(previewChars * 0.6) ? sliced.slice(0, lastBoundary) : sliced).trimEnd();
  return { preview, truncated: true };
}

function ExpandableText({
  text,
  expanded,
  onToggle,
  className,
  previewChars = DESCRIPTION_PREVIEW_CHAR_LIMIT,
  forceExpandable = false,
  loading = false,
}: ExpandableTextProps) {
  const { preview, truncated } = useMemo(() => buildExpandablePreview(text, previewChars), [previewChars, text]);
  const canExpand = truncated || forceExpandable;
  const displayedText = truncated && !expanded ? `${preview}...` : text;

  return (
    <p className={className}>
      {displayedText}
      {canExpand ? (
        <>
          {" "}
          <button
            type="button"
            onClick={onToggle}
            disabled={loading}
            aria-busy={loading}
            className="inline font-semibold text-white underline decoration-white/40 underline-offset-2 transition hover:text-white/82 hover:decoration-white/70 disabled:cursor-wait disabled:text-white/60 disabled:decoration-white/25"
          >
            {loading ? "Loading full text..." : expanded ? "View less" : "View more"}
          </button>
        </>
      ) : null}
    </p>
  );
}

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

function safeLocalStorageSetItem(key: string, value: string): boolean {
  if (typeof window === "undefined") {
    return false;
  }
  try {
    window.localStorage.setItem(key, value);
    return true;
  } catch {
    return false;
  }
}

function trimStoredText(value: string | undefined, maxChars: number): string | undefined {
  const normalized = String(value || "").trim();
  if (!normalized) {
    return undefined;
  }
  return normalized.length <= maxChars ? normalized : `${normalized.slice(0, Math.max(0, maxChars - 1)).trimEnd()}…`;
}

function compactStoredText(value: string | undefined, maxChars: number): { text?: string; truncated: boolean } {
  const text = trimStoredText(value, maxChars);
  return {
    text,
    truncated: Boolean(text) && text !== String(value || "").trim(),
  };
}

function looksLikeCompactedStoredText(value: string | undefined, maxChars: number): boolean {
  const normalized = String(value || "").trim();
  if (!normalized || !normalized.endsWith("…")) {
    return false;
  }
  return normalized.length >= Math.floor(maxChars * 0.95) && normalized.length <= maxChars;
}

function hasCompactedDescriptionText(reel: Reel): boolean {
  const description = reel.video_description?.trim();
  if (description) {
    return Boolean(reel.video_description_truncated) || looksLikeCompactedStoredText(description, 220) || looksLikeCompactedStoredText(description, 360);
  }
  const snippet = reel.transcript_snippet?.trim();
  if (snippet) {
    return Boolean(reel.transcript_snippet_truncated) || looksLikeCompactedStoredText(snippet, 220) || looksLikeCompactedStoredText(snippet, 360);
  }
  return false;
}

function selectStoredSessionWindow(reels: Reel[], activeIndex: number, maxReels: number): { reels: Reel[]; activeIndex: number } {
  if (reels.length <= maxReels) {
    return { reels, activeIndex: clamp(activeIndex, 0, Math.max(0, reels.length - 1)) };
  }
  const clampedActiveIndex = clamp(activeIndex, 0, reels.length - 1);
  const leadingCount = Math.floor(maxReels / 2);
  const start = clamp(clampedActiveIndex - leadingCount, 0, reels.length - maxReels);
  const nextReels = reels.slice(start, start + maxReels);
  return {
    reels: nextReels,
    activeIndex: clamp(clampedActiveIndex - start, 0, Math.max(0, nextReels.length - 1)),
  };
}

function compactStoredReel(reel: Reel, mode: "compact" | "minimal"): Reel {
  const isMinimal = mode === "minimal";
  const takeawayLimit = isMinimal ? 2 : 3;
  const takeawayCharLimit = isMinimal ? 120 : 180;
  const matchedTermLimit = isMinimal ? 4 : 6;
  const videoDescription = compactStoredText(reel.video_description, isMinimal ? 220 : 360);
  const aiSummary = compactStoredText(reel.ai_summary, isMinimal ? 220 : 360);
  const transcriptSnippet = compactStoredText(reel.transcript_snippet, isMinimal ? 220 : 360);
  return {
    ...reel,
    video_title: trimStoredText(reel.video_title, isMinimal ? 140 : 200),
    video_description: videoDescription.text,
    video_description_truncated: videoDescription.truncated || undefined,
    ai_summary: aiSummary.text,
    ai_summary_truncated: aiSummary.truncated || undefined,
    transcript_snippet: transcriptSnippet.text || "",
    transcript_snippet_truncated: transcriptSnippet.truncated || undefined,
    takeaways: (reel.takeaways ?? []).map((item) => trimStoredText(item, takeawayCharLimit) || "").filter(Boolean).slice(0, takeawayLimit),
    matched_terms: (reel.matched_terms ?? []).map((term) => trimStoredText(term, 48) || "").filter(Boolean).slice(0, matchedTermLimit),
    captions: undefined,
  };
}

function compactFeedSessionSnapshot(snapshot: FeedSessionSnapshot, mode: "compact" | "minimal"): FeedSessionSnapshot {
  const { reels, activeIndex } = selectStoredSessionWindow(
    snapshot.reels,
    snapshot.activeIndex,
    mode === "minimal" ? MINIMAL_REELS_PER_FEED_SESSION : COMPACT_REELS_PER_FEED_SESSION,
  );
  const compactedReels = reels.map((reel) => compactStoredReel(reel, mode));
  const nextActiveIndex = compactedReels.length > 0 ? clamp(activeIndex, 0, compactedReels.length - 1) : 0;
  return {
    ...snapshot,
    reels: compactedReels,
    activeIndex: nextActiveIndex,
    activeReelId: compactedReels[nextActiveIndex]?.reel_id,
  };
}

function persistFeedProgressSnapshot(materialId: string, entry: FeedProgressEntry): void {
  if (typeof window === "undefined" || !materialId) {
    return;
  }
  try {
    const allProgress = parseFeedProgress(window.localStorage.getItem(FEED_PROGRESS_STORAGE_KEY));
    allProgress[materialId] = entry;
    const ordered = Object.entries(allProgress)
      .sort((a, b) => (b[1].updatedAt || 0) - (a[1].updatedAt || 0))
      .slice(0, MAX_SAVED_FEED_PROGRESS);
    const attempts = [
      ordered,
      ordered.slice(0, Math.min(96, ordered.length)),
      ordered.slice(0, Math.min(24, ordered.length)),
      ordered.slice(0, 1),
    ];
    for (const candidate of attempts) {
      if (safeLocalStorageSetItem(FEED_PROGRESS_STORAGE_KEY, JSON.stringify(Object.fromEntries(candidate)))) {
        return;
      }
    }
  } catch {
    // Keep feed interaction usable even if browser storage is unavailable.
  }
}

function persistFeedSessionSnapshot(materialId: string, snapshot: FeedSessionSnapshot): void {
  if (typeof window === "undefined" || !materialId) {
    return;
  }
  try {
    const allSessions = parseFeedSessions(window.localStorage.getItem(FEED_SESSION_STORAGE_KEY));
    allSessions[materialId] = snapshot;
    const ordered = Object.entries(allSessions)
      .sort((a, b) => (b[1].updatedAt || 0) - (a[1].updatedAt || 0))
      .slice(0, MAX_SAVED_FEED_SESSIONS);
    // Retry with fewer and smaller snapshots so feed resume stays best-effort instead of crashing the page.
    const attempts: Array<Array<[string, FeedSessionSnapshot]>> = [
      ordered,
      ordered.slice(0, Math.min(12, ordered.length)),
      ordered
        .slice(0, Math.min(8, ordered.length))
        .map(([id, value]) => [id, id === materialId ? value : compactFeedSessionSnapshot(value, "compact")]),
      ordered
        .slice(0, Math.min(4, ordered.length))
        .map(([id, value]) => [id, compactFeedSessionSnapshot(value, id === materialId ? "compact" : "minimal")]),
      [[materialId, compactFeedSessionSnapshot(snapshot, "compact")]],
      [[materialId, compactFeedSessionSnapshot(snapshot, "minimal")]],
    ];
    for (const candidate of attempts) {
      if (safeLocalStorageSetItem(FEED_SESSION_STORAGE_KEY, JSON.stringify(Object.fromEntries(candidate)))) {
        return;
      }
    }
  } catch {
    // Keep feed interaction usable even if browser storage is unavailable.
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
    material_id: safeSetId,
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
  const feedRouteKey = params.toString();
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
  const searchFeedSettingsOverride = useMemo(
    () => readSearchFeedQuerySettings((key) => params.get(key)),
    [params],
  );

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
  const [initialFeedScreenReady, setInitialFeedScreenReady] = useState(false);
  const [authAccountId, setAuthAccountId] = useState<string | null>(null);
  const [authScopeHydrated, setAuthScopeHydrated] = useState(false);
  const [settingsScopeReady, setSettingsScopeReady] = useState(false);

  const feedViewportRef = useRef<HTMLDivElement | null>(null);
  const isFetchingRef = useRef(false);
  const isGeneratingRef = useRef(false);
  const activeIndexRef = useRef(0);
  const reelsRef = useRef<Reel[]>([]);
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
  const settingsLoadSequenceRef = useRef(0);
  const pendingRefinementJobsRef = useRef<Map<string, string>>(new Map());
  const isRefreshingRefinementRef = useRef(false);
  const pendingHistorySyncRef = useRef<StoredHistoryItem[] | null>(null);
  const historySyncTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const activeSearchScopeRef = useRef<FeedSearchScope>({
    key: "",
    seq: 0,
    controller: new AbortController(),
  });

  const abortActiveSearchScope = useCallback(() => {
    const scope = activeSearchScopeRef.current;
    if (!scope.controller.signal.aborted) {
      scope.controller.abort();
    }
  }, []);

  const resetActiveSearchRequestState = useCallback(() => {
    isFetchingRef.current = false;
    isGeneratingRef.current = false;
    isFastTopUpRef.current = false;
    isRefreshingRefinementRef.current = false;
    isRecoveringMissingMaterialRef.current = false;
    resumeLoadingRef.current = false;
    pendingRefinementJobsRef.current.clear();
    setGeneratingMore(false);
    setBootstrappingFirstReels(false);
  }, []);

  const isSearchScopeActive = useCallback((scope: Pick<FeedSearchScope, "key" | "seq">): boolean => {
    const current = activeSearchScopeRef.current;
    return current.key === scope.key && current.seq === scope.seq && !current.controller.signal.aborted;
  }, []);

  useEffect(() => {
    abortActiveSearchScope();
    const previous = activeSearchScopeRef.current;
    activeSearchScopeRef.current = {
      key: feedRouteKey,
      seq: previous.seq + 1,
      controller: new AbortController(),
    };
    resetActiveSearchRequestState();
    return () => {
      const current = activeSearchScopeRef.current;
      if (current.key === feedRouteKey) {
        current.controller.abort();
      }
    };
  }, [abortActiveSearchScope, feedRouteKey, resetActiveSearchRequestState]);

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

  const mergeFeedSettingsSnapshot = useCallback(
    (settings: StudyReelsSettings): StudyReelsSettings => mergeSearchFeedQuerySettings(settings, searchFeedSettingsOverride),
    [searchFeedSettingsOverride],
  );

  const getFeedTuningSettings = useCallback((): FeedTuningSettings => {
    const settings = mergeFeedSettingsSnapshot(readStudyReelsSettings());
    return {
      minRelevance: settings.minRelevanceThreshold,
      videoPoolMode: settings.videoPoolMode,
      preferredVideoDuration: settings.preferredVideoDuration,
      targetClipDurationSec: settings.targetClipDurationSec,
      targetClipDurationMinSec: settings.targetClipDurationMinSec,
      targetClipDurationMaxSec: settings.targetClipDurationMaxSec,
    };
  }, [mergeFeedSettingsSnapshot]);

  const hasMore = reels.length < total;
  const isFastGeneration = generationMode === "fast";
  const hasExplicitGenerationModeParam = generationModeParam === "fast" || generationModeParam === "slow";

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const syncAuthAccountId = () => {
      setAuthAccountId(readCommunityAuthSession()?.account?.id?.trim() || null);
      setAuthScopeHydrated(true);
    };
    syncAuthAccountId();
    window.addEventListener(COMMUNITY_AUTH_CHANGED_EVENT, syncAuthAccountId);
    return () => {
      window.removeEventListener(COMMUNITY_AUTH_CHANGED_EVENT, syncAuthAccountId);
    };
  }, []);

  useEffect(() => {
    if (generationModeParam === "fast" || generationModeParam === "slow") {
      setGenerationMode(generationModeParam);
    }
  }, [generationModeParam]);

  useEffect(() => {
    if (!authScopeHydrated) {
      return;
    }
    if (typeof window === "undefined") {
      setSettingsScopeReady(true);
      return;
    }
    let cancelled = false;
    const loadSequence = settingsLoadSequenceRef.current + 1;
    settingsLoadSequenceRef.current = loadSequence;
    setSettingsScopeReady(false);

    const applySettingsSnapshot = (settingsSnapshot: ReturnType<typeof readStudyReelsSettings>) => {
      const mergedSettings = mergeFeedSettingsSnapshot(settingsSnapshot);
      if (!hasExplicitGenerationModeParam) {
        setGenerationMode(mergedSettings.generationMode);
      }
      if (!mutedRestoredFromSnapshotRef.current) {
        setMutedPreference(mergedSettings.startMuted);
      }
    };

    const localSettings = setActiveStudyReelsSettingsScope(authAccountId);
    applySettingsSnapshot(localSettings);

    if (!authAccountId) {
      setSettingsScopeReady(true);
      return;
    }

    void (async () => {
      try {
        const remoteSettings = await fetchCommunitySettings();
        if (
          cancelled
          || settingsLoadSequenceRef.current !== loadSequence
          || !remoteSettings
          || readCommunityAuthSession()?.account?.id?.trim() !== authAccountId
        ) {
          return;
        }
        const activeSettings = setActiveStudyReelsSettingsScope(authAccountId, { settings: remoteSettings });
        applySettingsSnapshot(activeSettings);
      } catch (error) {
        if (cancelled || settingsLoadSequenceRef.current !== loadSequence) {
          return;
        }
        if (isSessionExpiredError(error)) {
          clearCommunityAuthSession();
          setAuthAccountId(null);
        }
      } finally {
        if (!cancelled && settingsLoadSequenceRef.current === loadSequence) {
          setSettingsScopeReady(true);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [authAccountId, authScopeHydrated, hasExplicitGenerationModeParam, mergeFeedSettingsSnapshot]);

  const clearHistorySyncTimer = useCallback(() => {
    if (historySyncTimerRef.current) {
      clearTimeout(historySyncTimerRef.current);
      historySyncTimerRef.current = null;
    }
  }, []);

  const flushPendingHistorySync = useCallback(() => {
    clearHistorySyncTimer();
    const pending = pendingHistorySyncRef.current;
    pendingHistorySyncRef.current = null;
    if (!pending || pending.length === 0) {
      return;
    }
    void queueCommunityHistorySync(pending).catch(() => {
      // Keep the local history state even if cross-device sync fails.
    });
  }, [clearHistorySyncTimer]);

  const scheduleRemoteHistorySync = useCallback((items: StoredHistoryItem[]) => {
    const accountId = authAccountId || readCommunityAuthSession()?.account?.id?.trim() || null;
    if (!accountId) {
      return;
    }
    pendingHistorySyncRef.current = items.map((item) => ({ ...item }));
    clearHistorySyncTimer();
    historySyncTimerRef.current = setTimeout(() => {
      const snapshot = pendingHistorySyncRef.current;
      pendingHistorySyncRef.current = null;
      historySyncTimerRef.current = null;
      if (!snapshot || snapshot.length === 0) {
        return;
      }
      void queueCommunityHistorySync(snapshot).catch(() => {
        // Keep the local history state even if cross-device sync fails.
      });
    }, 900);
  }, [authAccountId, clearHistorySyncTimer]);

  const buildPersistedSearchFeedQuery = useCallback(() => {
    if (!materialId) {
      return "";
    }
    const nextParams = new URLSearchParams(params.toString());
    nextParams.set("material_id", materialId);
    nextParams.set("generation_mode", generationMode);
    applySearchFeedSettingsToParams(nextParams, mergeFeedSettingsSnapshot(readStudyReelsSettings()));
    return nextParams.toString();
  }, [generationMode, materialId, mergeFeedSettingsSnapshot, params]);

  const persistCurrentSearchHistoryEntry = useCallback((options?: {
    syncRemote?: boolean;
    touchUpdatedAt?: boolean;
    reels?: Reel[];
    activeIndex?: number;
  }) => {
    if (typeof window === "undefined" || !materialId) {
      return;
    }
    try {
      const rawHistory = window.localStorage.getItem(HISTORY_STORAGE_KEY);
      const historyItems = rawHistory ? normalizeHistoryStorageItems(JSON.parse(rawHistory)) : [];
      const existing = historyItems.find((item) => item.materialId === materialId);
      const seeds = parseMaterialSeeds(window.localStorage.getItem(MATERIAL_SEEDS_STORAGE_KEY));
      const groups = parseMaterialGroups(window.localStorage.getItem(MATERIAL_GROUPS_STORAGE_KEY));
      const currentReels = options?.reels ?? reelsRef.current;
      const activeIndexValue = options?.activeIndex ?? activeIndexRef.current;
      const currentIndex = currentReels.length > 0 ? clamp(activeIndexValue, 0, currentReels.length - 1) : undefined;
      const currentReelId = currentIndex !== undefined ? currentReels[currentIndex]?.reel_id : undefined;
      const nextEntry: StoredHistoryItem = {
        materialId,
        title: existing?.title || groups[materialId]?.title?.trim() || seeds[materialId]?.title?.trim() || "New Study Session",
        updatedAt: options?.touchUpdatedAt === false ? existing?.updatedAt ?? Date.now() : Date.now(),
        starred: existing?.starred ?? false,
        generationMode,
        source: existing?.source ?? "search",
        feedQuery: buildPersistedSearchFeedQuery() || existing?.feedQuery,
        activeIndex: currentIndex ?? existing?.activeIndex,
        activeReelId: currentReelId || existing?.activeReelId,
      };
      const didChange =
        !existing
        || existing.title !== nextEntry.title
        || existing.updatedAt !== nextEntry.updatedAt
        || existing.starred !== nextEntry.starred
        || existing.generationMode !== nextEntry.generationMode
        || existing.source !== nextEntry.source
        || existing.feedQuery !== nextEntry.feedQuery
        || existing.activeIndex !== nextEntry.activeIndex
        || existing.activeReelId !== nextEntry.activeReelId;
      if (!didChange) {
        return;
      }
      const nextHistory = existing
        ? historyItems.map((item) => (item.materialId === materialId ? nextEntry : item))
        : [nextEntry, ...historyItems].slice(0, MAX_HISTORY_ITEMS);
      const accountId = authAccountId || readCommunityAuthSession()?.account?.id?.trim() || null;
      writeScopedHistorySnapshot(accountId, JSON.stringify(nextHistory));
      if (options?.syncRemote !== false) {
        scheduleRemoteHistorySync(nextHistory);
      }
    } catch {
      // Ignore malformed history payloads and keep feed mode persistence functional.
    }
  }, [authAccountId, buildPersistedSearchFeedQuery, generationMode, materialId, scheduleRemoteHistorySync]);

  useEffect(() => {
    persistCurrentSearchHistoryEntry();
  }, [feedRouteKey, generationMode, materialId, persistCurrentSearchHistoryEntry]);

  useEffect(() => {
    if (!materialId || reels.length === 0) {
      return;
    }
    persistCurrentSearchHistoryEntry({ syncRemote: false, touchUpdatedAt: false, reels, activeIndex });
  }, [activeIndex, materialId, persistCurrentSearchHistoryEntry, reels, reels.length]);

  useEffect(() => () => {
    flushPendingHistorySync();
  }, [flushPendingHistorySync]);

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
      applySearchFeedSettingsToParams(nextParams, mergeFeedSettingsSnapshot(readStudyReelsSettings()));
      router.replace(`/feed?${nextParams.toString()}`, { scroll: false });
    },
    [generationMode, materialId, mergeFeedSettingsSnapshot, params, router],
  );

  const recoverMissingMaterial = useCallback(
    async (missingMaterialId: string): Promise<boolean> => {
      if (typeof window === "undefined" || isRecoveringMissingMaterialRef.current) {
        return false;
      }
      const searchScope = activeSearchScopeRef.current;

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
          signal: searchScope.controller.signal,
        });
        if (!isSearchScopeActive(searchScope)) {
          return false;
        }
        const rebuiltId = rebuilt.material_id;
        seeds[rebuiltId] = {
          ...seed,
          topic: topic || undefined,
          text: text || undefined,
          updatedAt: Date.now(),
        };
        delete seeds[missingMaterialId];
        safeLocalStorageSetItem(MATERIAL_SEEDS_STORAGE_KEY, JSON.stringify(seeds));
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
        safeLocalStorageSetItem(MATERIAL_GROUPS_STORAGE_KEY, JSON.stringify(nextGroups));

        const nextParams = new URLSearchParams(params.toString());
        nextParams.set("material_id", rebuiltId);
        router.replace(`/feed?${nextParams.toString()}`);
        return true;
      } catch (e) {
        if (!isSearchScopeActive(searchScope) || isRequestInterruptedError(e)) {
          return false;
        }
        setError(e instanceof Error ? e.message : "Could not rebuild material.");
        return false;
      } finally {
        if (isSearchScopeActive(searchScope)) {
          isRecoveringMissingMaterialRef.current = false;
        }
      }
    },
    [isSearchScopeActive, params, router],
  );

  const registerRefinementJob = useCallback(
    (materialIdValue: string, payload?: { refinement_job_id?: string | null; refinement_status?: string | null }) => {
      const materialIdKey = String(materialIdValue || "").trim();
      if (!materialIdKey) {
        return;
      }
      const jobId = String(payload?.refinement_job_id || "").trim();
      const status = String(payload?.refinement_status || "").trim().toLowerCase();
      if (!jobId || status === "failed" || status === "completed" || status === "superseded") {
        pendingRefinementJobsRef.current.delete(materialIdKey);
        return;
      }
      pendingRefinementJobsRef.current.set(materialIdKey, jobId);
    },
    [],
  );

  const hasPendingRefinementForFeed = useCallback((): boolean => {
    const feedMaterialIds = getFeedMaterialIds();
    if (feedMaterialIds.length === 0) {
      return false;
    }
    return feedMaterialIds.some((id) => pendingRefinementJobsRef.current.has(String(id || "").trim()));
  }, [getFeedMaterialIds]);

  const countReelsForMaterial = useCallback(
    (materialIdValue: string): number => {
      const materialIdKey = String(materialIdValue || "").trim();
      if (!materialIdKey) {
        return 0;
      }
      const singleFeedMaterialId = getFeedMaterialIds().length === 1 ? materialIdKey : "";
      return reels.reduce((count, reel) => {
        const reelMaterialId = String(reel.material_id || singleFeedMaterialId).trim();
        return reelMaterialId === materialIdKey ? count + 1 : count;
      }, 0);
    },
    [getFeedMaterialIds, reels],
  );

  const feedNeedsBootstrapTopUp = useCallback((): boolean => {
    const feedMaterialIds = getFeedMaterialIds();
    if (feedMaterialIds.length === 0) {
      return false;
    }
    const minimumPerTopic = generationMode === "fast" ? 3 : 5;
    return feedMaterialIds.some((id) => countReelsForMaterial(id) < minimumPerTopic);
  }, [countReelsForMaterial, generationMode, getFeedMaterialIds]);

  const shouldBlockOnPendingRefinement = useCallback((): boolean => {
    return hasPendingRefinementForFeed() && !feedNeedsBootstrapTopUp();
  }, [feedNeedsBootstrapTopUp, hasPendingRefinementForFeed]);

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

  const loadPage = useCallback(
    async (targetPage: number, options?: { autofill?: boolean }) => {
      const feedMaterialIds = getFeedMaterialIds();
      if (!settingsScopeReady || feedMaterialIds.length === 0 || isFetchingRef.current) {
        return;
      }
      const searchScope = activeSearchScopeRef.current;
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
                prefetch: generationMode === "fast" ? INITIAL_FAST_PREFETCH : INITIAL_SLOW_PREFETCH,
                generationMode,
                minRelevance: tuning.minRelevance,
                videoPoolMode: tuning.videoPoolMode,
                preferredVideoDuration: tuning.preferredVideoDuration,
                targetClipDurationSec: tuning.targetClipDurationSec,
                targetClipDurationMinSec: tuning.targetClipDurationMinSec,
                targetClipDurationMaxSec: tuning.targetClipDurationMaxSec,
                signal: searchScope.controller.signal,
              });
              return { materialId: id, data, error: null };
            } catch (error) {
              return { materialId: id, data: null, error };
            }
          }),
        );
        if (!isSearchScopeActive(searchScope)) {
          return;
        }

        const successful = rows.filter((row) => row.data);
        successful.forEach((row) => registerRefinementJob(row.materialId, row.data ?? undefined));
        if (successful.length === 0) {
          if (targetPage === 1) {
            const firstError = rows[0]?.error;
            if (isRequestInterruptedError(firstError)) {
              return;
            }
            const message = firstError instanceof Error ? firstError.message : "Feed failed to load";
            const missingMaterialId = String(rows[0]?.materialId || materialId || "").trim();
            if (
              feedMaterialIds.length === 1 &&
              /material_id not found/i.test(message) &&
              missingMaterialId &&
              !recoveryAttemptedIdsRef.current.has(missingMaterialId)
            ) {
              recoveryAttemptedIdsRef.current.add(missingMaterialId);
              const recovered = await recoverMissingMaterial(missingMaterialId);
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
        if (!isSearchScopeActive(searchScope) || isRequestInterruptedError(e)) {
          return;
        }
        if (targetPage === 1) {
          setError(e instanceof Error ? e.message : "Feed failed to load");
        }
      } finally {
        if (isSearchScopeActive(searchScope)) {
          setLoading(false);
          isFetchingRef.current = false;
        }
      }
    },
    [dedupeByIdentity, generationMode, getFeedMaterialIds, getFeedTuningSettings, interleaveReelBatches, isSearchScopeActive, materialId, recoverMissingMaterial, registerRefinementJob, settingsScopeReady],
  );

  const requestMore = useCallback(async (options?: { surfaceError?: boolean }): Promise<Reel[]> => {
    const feedMaterialIds = getFeedMaterialIds();
    if (!settingsScopeReady || feedMaterialIds.length === 0 || isGeneratingRef.current || !canRequestMore) {
      return [];
    }
    const searchScope = activeSearchScopeRef.current;
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
          const streamedReels: Reel[] = [];
          const currentCount = countReelsForMaterial(id);
          const initialTarget = generationMode === "fast" ? 3 : 5;
          // Keep extending the requested total upward so broad topics do not stall at a client-side cap.
          const targetTotal = currentCount > 0 ? currentCount + perTopicBatch : initialTarget;
          try {
            const data = await generateReelsStream({
              materialId: id,
              numReels: targetTotal,
              generationMode,
              minRelevance: tuning.minRelevance,
              videoPoolMode: tuning.videoPoolMode,
              preferredVideoDuration: tuning.preferredVideoDuration,
              targetClipDurationSec: tuning.targetClipDurationSec,
              targetClipDurationMinSec: tuning.targetClipDurationMinSec,
              targetClipDurationMaxSec: tuning.targetClipDurationMaxSec,
              signal: searchScope.controller.signal,
              onReel: (reel) => {
                if (!isSearchScopeActive(searchScope)) {
                  return;
                }
                streamedReels.push(reel);
                appendGeneratedReels([reel]);
              },
            });
            return { materialId: id, data, streamedReels, error: null };
          } catch (e) {
            if (!isRequestInterruptedError(e)) {
              console.warn(`Background reel generation failed for topic material ${id}:`, e);
            }
            return { materialId: id, data: null, streamedReels, error: e };
          }
        }),
      );
      if (!isSearchScopeActive(searchScope)) {
        return [];
      }
      const firstFailedRow = generatedRows.find((row) => row?.error);
      const firstFailureMessage =
        firstFailedRow?.error instanceof Error ? firstFailedRow.error.message : "";
      if (isRequestInterruptedError(firstFailedRow?.error)) {
        return [];
      }
      const missingMaterialId = String(firstFailedRow?.materialId || materialId || "").trim();
      if (
        feedMaterialIds.length === 1 &&
        /material_id not found/i.test(firstFailureMessage) &&
        missingMaterialId
      ) {
        if (!recoveryAttemptedIdsRef.current.has(missingMaterialId)) {
          recoveryAttemptedIdsRef.current.add(missingMaterialId);
          const recovered = await recoverMissingMaterial(missingMaterialId);
          if (recovered) {
            return [];
          }
        }
        return [];
      }
      generatedRows.forEach((row) => {
        if (row?.data) {
          registerRefinementJob(row.materialId, row.data);
        }
      });
      const generated = dedupeByIdentity(
        interleaveReelBatches(generatedRows.map((row) => row.data?.reels ?? row.streamedReels ?? [])),
      );
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
      if (!isSearchScopeActive(searchScope) || isRequestInterruptedError(e)) {
        return [];
      }
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
      if (isSearchScopeActive(searchScope)) {
        setGeneratingMore(false);
        isGeneratingRef.current = false;
      }
    }
  }, [appendGeneratedReels, canRequestMore, countReelsForMaterial, dedupeByIdentity, generationMode, getFeedMaterialIds, getFeedTuningSettings, interleaveReelBatches, isFastGeneration, isSearchScopeActive, materialId, recoverMissingMaterial, registerRefinementJob, settingsScopeReady]);

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
      pendingRefinementJobsRef.current.clear();
      setLoading(false);
      return;
    }
    if (!settingsScopeReady) {
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
      try {
        const rawHistory = window.localStorage.getItem(HISTORY_STORAGE_KEY);
        if (rawHistory) {
          const historyEntry = normalizeHistoryStorageItems(JSON.parse(rawHistory)).find((item) => item.materialId === materialId);
          if (historyEntry && (historyEntry.activeReelId || historyEntry.activeIndex !== undefined)) {
            const historyResume: FeedProgressEntry = {
              index: Math.max(0, Math.floor(historyEntry.activeIndex || 0)),
              reelId: historyEntry.activeReelId,
              updatedAt: historyEntry.updatedAt || 0,
            };
            if (!resumeTarget || historyResume.updatedAt >= (resumeTarget.updatedAt || 0)) {
              resumeTarget = historyResume;
            }
          }
        }
      } catch {
        // Ignore malformed history payloads and keep feed resume best-effort.
      }
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
    pendingRefinementJobsRef.current.clear();
    if (restoredSession) {
      const allowedMaterialIds = new Set(feedMaterialIds.map((id) => String(id || "").trim()).filter(Boolean));
      const singleFeedMaterialId = feedMaterialIds.length === 1 ? feedMaterialIds[0] : "";
      const restoredReels = dedupeByIdentity(
        restoredSession.reels.filter((reel) => {
          const reelMaterialId = String(reel.material_id || singleFeedMaterialId).trim();
          return !reelMaterialId || allowedMaterialIds.has(reelMaterialId);
        }),
      ).slice(-MAX_REELS_PER_FEED_SESSION);
      const restoredIndex = restoredReels.length > 0 ? clamp(restoredSession.activeIndex, 0, restoredReels.length - 1) : 0;
      const restoredReelId = restoredReels[restoredIndex]?.reel_id;
      const modeFromQuery =
        generationModeParam === "fast" || generationModeParam === "slow" ? generationModeParam : null;
      const restoredGenerationMode = modeFromQuery ?? restoredSession.generationMode;
      const minimumPerTopic = restoredGenerationMode === "fast" ? 3 : 5;
      const restoredCounts = new Map<string, number>();
      for (const reel of restoredReels) {
        const reelMaterialId = String(reel.material_id || singleFeedMaterialId).trim();
        if (!reelMaterialId) {
          continue;
        }
        restoredCounts.set(reelMaterialId, (restoredCounts.get(reelMaterialId) ?? 0) + 1);
      }
      const restoredSessionUnderfilled = feedMaterialIds.some(
        (id) => (restoredCounts.get(String(id || "").trim()) ?? 0) < minimumPerTopic,
      );
      setReels(restoredReels);
      setPage(restoredSession.page);
      setTotal(Math.max(restoredSession.total, restoredReels.length));
      setCanRequestMore(restoredSessionUnderfilled || restoredSession.canRequestMore);
      setGenerationMode(restoredGenerationMode);
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
  }, [communityHandoffIdParam, communityPreviewReel, dedupeByIdentity, generationModeParam, materialId, loadPage, settingsScopeReady]);

  useEffect(() => {
    if (!materialId || sessionHydrated || loading) {
      return;
    }
    setSessionHydrated(true);
  }, [loading, materialId, sessionHydrated]);

  const refreshRefinedFeed = useCallback(async () => {
    const feedMaterialIds = getFeedMaterialIds();
    if (!settingsScopeReady || feedMaterialIds.length === 0 || isRefreshingRefinementRef.current) {
      return;
    }
    const searchScope = activeSearchScopeRef.current;
    isRefreshingRefinementRef.current = true;
    try {
      const tuning = getFeedTuningSettings();
      const rows = await Promise.all(
        feedMaterialIds.map(async (id) => {
          try {
            const pageLimit = 25;
            const collected: Reel[] = [];
            let pageToLoad = 1;
            let totalForMaterial = 0;
            let latestData: Awaited<ReturnType<typeof fetchFeed>> | null = null;

            while (pageToLoad === 1 || collected.length < totalForMaterial) {
              if (!isSearchScopeActive(searchScope)) {
                return null;
              }
              const data = await fetchFeed({
                materialId: id,
                page: pageToLoad,
                limit: pageLimit,
                autofill: false,
                prefetch: 0,
                generationMode,
                minRelevance: tuning.minRelevance,
                videoPoolMode: tuning.videoPoolMode,
                preferredVideoDuration: tuning.preferredVideoDuration,
                targetClipDurationSec: tuning.targetClipDurationSec,
                targetClipDurationMinSec: tuning.targetClipDurationMinSec,
                targetClipDurationMaxSec: tuning.targetClipDurationMaxSec,
                signal: searchScope.controller.signal,
              });
              if (!isSearchScopeActive(searchScope)) {
                return null;
              }
              latestData = data;
              totalForMaterial = Math.max(0, Number(data.total) || 0);
              const merged = dedupeByIdentity(data.reels, collected);
              collected.splice(0, collected.length, ...merged);
              if (data.reels.length === 0 || collected.length >= totalForMaterial) {
                break;
              }
              pageToLoad += 1;
            }

            if (!latestData) {
              return null;
            }
            return {
              materialId: id,
              data: {
                ...latestData,
                page: 1,
                limit: pageLimit,
                total: Math.max(totalForMaterial, collected.length),
                reels: collected,
              },
            };
          } catch (error) {
            if (!isRequestInterruptedError(error)) {
              console.warn(`Refined feed refresh failed for topic material ${id}:`, error);
            }
            return null;
          }
        }),
      );
      if (!isSearchScopeActive(searchScope)) {
        return;
      }
      const successful = rows.filter((row): row is { materialId: string; data: Awaited<ReturnType<typeof fetchFeed>> } => Boolean(row?.data));
      if (successful.length === 0) {
        return;
      }
      successful.forEach((row) => registerRefinementJob(row.materialId, row.data));
      const refreshedReels = dedupeByIdentity(interleaveReelBatches(successful.map((row) => row.data.reels)));
      const refreshedTotal = successful.reduce((sum, row) => sum + Math.max(0, Number(row.data.total) || 0), 0);
      const currentReels = reelsRef.current;
      const currentReelId = currentReels[activeIndexRef.current]?.reel_id;
      const mergedReels = dedupeByIdentity(refreshedReels, currentReels);

      setReels((prev) => dedupeByIdentity(refreshedReels, prev));
      setTotal((prevTotal) => Math.max(prevTotal, refreshedTotal, mergedReels.length));
      setPage((prevPage) => Math.max(prevPage, Math.max(1, Math.ceil(mergedReels.length / PAGE_SIZE))));
      if (mergedReels.length === 0) {
        setActiveIndex(0);
      } else if (currentReelId) {
        const nextIndex = mergedReels.findIndex((reel) => reel.reel_id === currentReelId);
        setActiveIndex(nextIndex >= 0 ? nextIndex : Math.min(activeIndexRef.current, mergedReels.length - 1));
      } else {
        setActiveIndex(Math.min(activeIndexRef.current, mergedReels.length - 1));
      }
    } finally {
      if (isSearchScopeActive(searchScope)) {
        isRefreshingRefinementRef.current = false;
      }
    }
  }, [dedupeByIdentity, generationMode, getFeedMaterialIds, getFeedTuningSettings, interleaveReelBatches, isSearchScopeActive, registerRefinementJob, settingsScopeReady]);

  const runFastTopUp = useCallback(async () => {
    const feedMaterialIds = getFeedMaterialIds();
    if (
      !settingsScopeReady
      || feedMaterialIds.length === 0
      || generationMode !== "fast"
      || !canRequestMore
      || isFastTopUpRef.current
      || isGeneratingRef.current
      || shouldBlockOnPendingRefinement()
    ) {
      return;
    }
    const searchScope = activeSearchScopeRef.current;
    const tuning = getFeedTuningSettings();
    const perTopicBatch = Math.max(1, Math.ceil(12 / feedMaterialIds.length));
    isFastTopUpRef.current = true;
    try {
      const generatedRows = await Promise.all(
        feedMaterialIds.map(async (id) => {
          const streamedReels: Reel[] = [];
          const currentCount = countReelsForMaterial(id);
          const targetTotal = currentCount + perTopicBatch;
          try {
            const data = await generateReelsStream({
              materialId: id,
              numReels: targetTotal,
              generationMode,
              minRelevance: tuning.minRelevance,
              videoPoolMode: tuning.videoPoolMode,
              preferredVideoDuration: tuning.preferredVideoDuration,
              targetClipDurationSec: tuning.targetClipDurationSec,
              targetClipDurationMinSec: tuning.targetClipDurationMinSec,
              targetClipDurationMaxSec: tuning.targetClipDurationMaxSec,
              signal: searchScope.controller.signal,
              onReel: (reel) => {
                if (!isSearchScopeActive(searchScope)) {
                  return;
                }
                streamedReels.push(reel);
                appendGeneratedReels([reel]);
              },
            });
            return { materialId: id, data, streamedReels };
          } catch (e) {
            if (!isRequestInterruptedError(e)) {
              console.warn(`Fast mode background top-up failed for topic material ${id}:`, e);
            }
            return null;
          }
        }),
      );
      if (!isSearchScopeActive(searchScope)) {
        return;
      }
      generatedRows.forEach((row) => {
        if (row?.data) {
          registerRefinementJob(row.materialId, row.data);
        }
      });
      const generated = dedupeByIdentity(
        interleaveReelBatches(generatedRows.map((row) => row?.data.reels ?? row?.streamedReels ?? [])),
      );
      if (generated.length > 0) {
        emptyGenerateStreakRef.current = 0;
      }
    } catch (e) {
      if (!isSearchScopeActive(searchScope) || isRequestInterruptedError(e)) {
        return;
      }
      console.warn("Fast mode background top-up failed:", e);
    } finally {
      if (isSearchScopeActive(searchScope)) {
        isFastTopUpRef.current = false;
      }
    }
  }, [appendGeneratedReels, canRequestMore, countReelsForMaterial, dedupeByIdentity, generationMode, getFeedMaterialIds, getFeedTuningSettings, interleaveReelBatches, isSearchScopeActive, registerRefinementJob, settingsScopeReady, shouldBlockOnPendingRefinement]);

  useEffect(() => {
    if (!settingsScopeReady || getFeedMaterialIds().length === 0) {
      return;
    }
    let cancelled = false;
    const pollRefinements = async () => {
      const searchScope = activeSearchScopeRef.current;
      const pendingEntries = Array.from(pendingRefinementJobsRef.current.entries());
      if (pendingEntries.length === 0 || isRefreshingRefinementRef.current) {
        return;
      }
      let shouldRefresh = false;
      await Promise.all(
        pendingEntries.map(async ([materialIdKey, jobId]) => {
          try {
            const status = await fetchRefinementStatus(jobId, { signal: searchScope.controller.signal });
            if (!isSearchScopeActive(searchScope)) {
              return;
            }
            if (status.status === "failed" || status.status === "superseded" || status.status === "completed") {
              pendingRefinementJobsRef.current.delete(materialIdKey);
            }
            if (
              status.status === "completed"
              && status.result_generation_id
              && status.result_generation_id === status.active_generation_id
            ) {
              shouldRefresh = true;
            }
          } catch (error) {
            if (!isSearchScopeActive(searchScope) || isRequestInterruptedError(error)) {
              return;
            }
            console.warn(`Refinement status polling failed for material ${materialIdKey}:`, error);
          }
        }),
      );
      if (!cancelled && shouldRefresh && isSearchScopeActive(searchScope)) {
        await refreshRefinedFeed();
      }
    };

    void pollRefinements();
    const timer = window.setInterval(() => {
      void pollRefinements();
    }, REFINEMENT_POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [getFeedMaterialIds, isSearchScopeActive, refreshRefinedFeed, settingsScopeReady]);

  const bootstrapFirstReels = useCallback(
    async (manual = false) => {
      if (!materialId || isGeneratingRef.current || !canRequestMore) {
        return;
      }
      const searchScope = activeSearchScopeRef.current;
      setBootstrappingFirstReels(true);
      try {
        const generated = await requestMore({ surfaceError: manual });
        if (isSearchScopeActive(searchScope) && generationMode === "fast" && generated.length > 0) {
          void runFastTopUp();
        }
      } finally {
        if (isSearchScopeActive(searchScope)) {
          setBootstrappingFirstReels(false);
        }
      }
    },
    [canRequestMore, generationMode, isSearchScopeActive, materialId, requestMore, runFastTopUp],
  );

  useEffect(() => {
    if (
      !materialId
      || loading
      || bootstrappingFirstReels
      || bootstrapAttemptedRef.current
      || shouldBlockOnPendingRefinement()
      || !feedNeedsBootstrapTopUp()
    ) {
      return;
    }
    bootstrapAttemptedRef.current = true;
    void bootstrapFirstReels(false);
  }, [bootstrapFirstReels, bootstrappingFirstReels, feedNeedsBootstrapTopUp, loading, materialId, shouldBlockOnPendingRefinement]);

  const maybeLoadMore = useCallback(() => {
    if (isFetchingRef.current) {
      return;
    }
    if (hasMore) {
      loadPage(page + 1, { autofill: true });
      return;
    }
    if (canRequestMore && !isGeneratingRef.current) {
      if (shouldBlockOnPendingRefinement()) {
        return;
      }
      void (async () => {
        const generated = await requestMore();
        if (generationMode === "fast" && generated.length > 0) {
          void runFastTopUp();
        }
      })();
    }
  }, [canRequestMore, generationMode, hasMore, loadPage, page, requestMore, runFastTopUp, shouldBlockOnPendingRefinement]);

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
      if (shouldBlockOnPendingRefinement()) {
        return;
      }
      resumeLoadingRef.current = true;
      void (async () => {
        const generated = await requestMore();
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
  }, [canRequestMore, generationMode, hasMore, loadPage, materialId, page, reels, requestMore, runFastTopUp, shouldBlockOnPendingRefinement]);

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
    reelsRef.current = reels;
  }, [reels]);

  useEffect(() => {
    if (typeof window === "undefined" || !materialId || reels.length === 0) {
      return;
    }
    const index = clamp(activeIndex, 0, reels.length - 1);
    const activeReel = reels[index];
    if (!activeReel?.reel_id) {
      return;
    }
    persistFeedProgressSnapshot(materialId, {
      index,
      reelId: activeReel.reel_id,
      updatedAt: Date.now(),
    });
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
    if (typeof window === "undefined" || !materialId || !sessionHydrated) {
      return;
    }
    const allowedMaterialIds = new Set(getFeedMaterialIds().map((id) => String(id || "").trim()).filter(Boolean));
    const singleFeedMaterialId = allowedMaterialIds.size === 1 ? Array.from(allowedMaterialIds)[0] : "";
    const dedupedReels = dedupeByIdentity(
      reels.filter((reel) => {
        const reelMaterialId = String(reel.material_id || singleFeedMaterialId).trim();
        return !reelMaterialId || allowedMaterialIds.has(reelMaterialId);
      }),
    ).slice(-MAX_REELS_PER_FEED_SESSION);
    const index = dedupedReels.length > 0 ? clamp(activeIndex, 0, dedupedReels.length - 1) : 0;
    const activeReelId = dedupedReels[index]?.reel_id;
    persistFeedSessionSnapshot(materialId, {
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
    });
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
  const [descriptionExpanded, setDescriptionExpanded] = useState(false);
  const [descriptionHydrating, setDescriptionHydrating] = useState(false);
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
  const shouldKeepFeedEntryLoading = loading || (reels.length === 0 && bootstrappingFirstReels);
  const showLoadingScreen = useLoadingScreenGate(initialFeedScreenReady, {
    minimumVisibleMs: 1000,
  });

  useEffect(() => {
    setInitialFeedScreenReady(false);
  }, [feedRouteKey]);

  useEffect(() => {
    if (!shouldKeepFeedEntryLoading) {
      setInitialFeedScreenReady(true);
    }
  }, [shouldKeepFeedEntryLoading]);

  useEffect(() => {
    setDescriptionExpanded(false);
    setDescriptionHydrating(false);
  }, [activeReel?.reel_id]);

  const activeVideoDescription = useMemo(() => {
    if (!activeReel) {
      return { text: "", compacted: false };
    }
    const description = activeReel.video_description?.trim();
    if (description) {
      return { text: description, compacted: hasCompactedDescriptionText(activeReel) };
    }
    const snippet = activeReel.transcript_snippet?.trim();
    if (snippet) {
      return { text: snippet, compacted: hasCompactedDescriptionText(activeReel) };
    }
    return { text: "No video description available for this reel.", compacted: false };
  }, [activeReel]);

  const toggleDescriptionExpanded = useCallback(async () => {
    if (!descriptionExpanded && activeVideoDescription.compacted) {
      setDescriptionHydrating(true);
      try {
        await refreshRefinedFeed();
      } finally {
        setDescriptionHydrating(false);
      }
    }
    setDescriptionExpanded((prev) => !prev);
  }, [activeVideoDescription.compacted, descriptionExpanded, refreshRefinedFeed]);

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
    abortActiveSearchScope();
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
  }, [abortActiveSearchScope, feedFallbackPath, returnTabParam, router]);

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

  if (showLoadingScreen) {
    return <FullscreenLoadingScreen />;
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
                  <ExpandableText
                    text={activeVideoDescription.text}
                    expanded={descriptionExpanded}
                    onToggle={() => {
                      void toggleDescriptionExpanded();
                    }}
                    className="whitespace-pre-line break-words [overflow-wrap:anywhere]"
                    previewChars={DESCRIPTION_PREVIEW_CHAR_LIMIT}
                    forceExpandable={activeVideoDescription.compacted}
                    loading={descriptionHydrating}
                  />
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
                  <ExpandableText
                    text={activeVideoDescription.text}
                    expanded={descriptionExpanded}
                    onToggle={() => {
                      void toggleDescriptionExpanded();
                    }}
                    className="whitespace-pre-line break-words [overflow-wrap:anywhere]"
                    previewChars={DESCRIPTION_PREVIEW_CHAR_LIMIT}
                    forceExpandable={activeVideoDescription.compacted}
                    loading={descriptionHydrating}
                  />
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
      fallback={<FullscreenLoadingScreen />}
    >
      <FeedPageInner />
    </Suspense>
  );
}
