"use client";

import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

import { FullscreenLoadingScreen } from "@/components/FullscreenLoadingScreen";
import { RecallCheck, type RecallAnswerReveal } from "@/components/RecallCheck";
import { ReelCard } from "@/components/ReelCard";
import {
  COMMUNITY_AUTH_CHANGED_EVENT,
  answerAssessmentQuestion,
  askStudyChat,
  clearCommunityAuthSession,
  fetchCommunitySettings,
  fetchFeed,
  fetchPendingAssessment,
  generateReelsStream,
  isRequestInterruptedError,
  isSessionExpiredError,
  isTransportError,
  queueCommunityHistorySync,
  readCommunityAuthSession,
  reportReelProgress,
  reportReelScroll,
  sendFeedback,
  snoozeAssessment,
  startNextAssessment,
  uploadMaterial,
} from "@/lib/api";
import { applySearchFeedSettingsToParams, mergeSearchFeedQuerySettings, readSearchFeedQuerySettings } from "@/lib/feedQuery";
import {
  HISTORY_STORAGE_KEY,
  normalizeStoredHistoryItems as normalizeHistoryStorageItems,
  type StoredRecallSummary,
  type StoredHistoryItem,
  writeScopedHistorySnapshot,
} from "@/lib/historyStorage";
import { useLoadingScreenGate } from "@/lib/useLoadingScreenGate";
import {
  type GenerationMode,
  type PreferredVideoDuration,
  type StudyReelsSettings,
  readStudyReelsSettings,
  setActiveStudyReelsSettingsScope,
} from "@/lib/settings";
import type {
  AssessmentSession,
  AssessmentStatusResponse,
  ChatMessage,
  GenerationJobStatus,
  GenerationTerminalStatus,
  Reel,
} from "@/lib/types";

const PAGE_SIZE = 5;
const FAST_READY_RESERVOIR_TARGET = 8;
const SLOW_READY_RESERVOIR_TARGET = 12;
const READY_RESERVOIR_REFILL_THRESHOLD = 4;

function readyReservoirTarget(mode: GenerationMode): number {
  return mode === "fast" ? FAST_READY_RESERVOIR_TARGET : SLOW_READY_RESERVOIR_TARGET;
}
const REEL_SNAP_DURATION_MS = 300;
const POST_SNAP_COOLDOWN_MS = 30;
const WHEEL_GESTURE_RELEASE_MS = 220;
const WHEEL_DELTA_THRESHOLD = 110;
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
const MAX_REELS_PER_FEED_SESSION = 300;
const COMPACT_REELS_PER_FEED_SESSION = 48;
const MINIMAL_REELS_PER_FEED_SESSION = 20;
const RECOVERY_REQUEST_IDLE_TIMEOUT_MS = 18_000;
const GENERATION_STREAM_IDLE_TIMEOUT_MS = 35_000;
const GENERATION_EXHAUSTED_COOLDOWN_MS = 60_000;
const COMMUNITY_SET_FEED_HANDOFF_PREFIX = "studyreels-community-feed-handoff-";
const DESCRIPTION_PREVIEW_CHAR_LIMIT = 180;
const FEED_PLAYBACK_RATE_OPTIONS = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2] as const;
type FeedbackAction = "helpful" | "confusing" | "save";
type FeedRecoveryPhase = "idle" | "fetching-page" | "generating";

type FeedTuningSettings = {
  minRelevance: number;
  creativeCommonsOnly: boolean;
  preferredVideoDuration: PreferredVideoDuration;
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
  feedbackByReel: Record<string, ReelFeedbackState>;
  adaptiveExcludeReelIds: string[];
  page: number;
  total: number;
  canRequestMore: boolean;
  generationMode: GenerationMode;
  mutedPreference: boolean;
  autoplayEnabled: boolean;
  playbackRate: number;
  activeIndex: number;
  watchedFrontierIndex?: number;
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

type ActiveRecoveryRequest = {
  key: string;
  seq: number;
  phase: Extract<FeedRecoveryPhase, "fetching-page" | "generating">;
};

type ActiveGenerationJob = {
  jobId: string;
  status: Extract<GenerationJobStatus, "queued" | "running">;
};

type SessionMergeResult = {
  reels: Reel[];
  addedReels: Reel[];
  addedCount: number;
  updatedCount: number;
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

function GenerationStageStatus({
  ready,
  reconnecting = false,
  variant = "bar",
}: {
  ready: number;
  reconnecting?: boolean;
  variant?: "bar" | "center";
}) {
  const label = reconnecting
    ? "Reconnecting to the same clip job..."
    : ready > 0
      ? `${ready} ready · improving the rest`
      : "Finding the first clips";
  if (variant === "center") {
    return (
      <div role="status" aria-live="polite" className="w-full">
        <div className="relative h-1 overflow-hidden rounded-full bg-white/10">
          <div className="animate-progress-shimmer absolute inset-y-0 w-1/3 bg-gradient-to-r from-transparent via-white/60 to-transparent" />
        </div>
        <p className="mt-3 text-sm font-semibold">{label}</p>
        <p className="mt-1 text-xs text-white/72">Verified clips appear here as soon as they are ready.</p>
      </div>
    );
  }
  return (
    <div className="pointer-events-none absolute inset-x-0 top-0 z-[9998]" role="status" aria-live="polite">
      <div className="relative h-1 overflow-hidden bg-white/10">
        <div className="animate-progress-shimmer absolute inset-y-0 w-1/3 bg-gradient-to-r from-transparent via-white/60 to-transparent" />
      </div>
      <div className="flex justify-center py-1.5">
        <span className="rounded-full bg-black/56 px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.12em] text-white/80 backdrop-blur-sm">
          {label}
        </span>
      </div>
    </div>
  );
}

function detailSentences(value: string | undefined, limit = 4): string[] {
  const normalized = String(value || "").replace(/\s+/g, " ").trim();
  if (!normalized) {
    return [];
  }
  const chunks = normalized.match(/[^.!?]+[.!?]+|[^.!?]+$/g) ?? [normalized];
  const seen = new Set<string>();
  const sentences: string[] = [];
  for (const chunk of chunks) {
    const sentence = chunk.trim();
    const key = sentence.toLowerCase();
    if (!sentence || seen.has(key)) {
      continue;
    }
    seen.add(key);
    sentences.push(sentence);
    if (sentences.length >= limit) {
      break;
    }
  }
  return sentences;
}

function buildReelDetailContent(reel: Reel): { summary: string; takeaways: string[]; reason: string } {
  const transcriptSentences = detailSentences(reel.transcript_snippet, 6);
  const descriptionSentences = detailSentences(reel.video_description, 3);
  const summary = reel.ai_summary?.trim()
    || transcriptSentences.slice(0, 2).join(" ")
    || descriptionSentences.slice(0, 2).join(" ")
    || `This clip introduces ${reel.concept_title}.`;

  const takeaways: string[] = [];
  const seen = new Set<string>();
  for (const item of [...(reel.takeaways ?? []), ...transcriptSentences]) {
    const normalized = String(item || "").replace(/\s+/g, " ").trim();
    const key = normalized.toLowerCase();
    if (!normalized || seen.has(key) || normalized.toLowerCase() === summary.toLowerCase()) {
      continue;
    }
    seen.add(key);
    takeaways.push(normalized);
    if (takeaways.length >= 4) {
      break;
    }
  }
  if (takeaways.length === 0) {
    takeaways.push(summary);
  }

  const explicitReason = reel.match_reason?.trim() || reel.relevance_reason?.trim();
  const terms = (reel.matched_terms ?? []).map((term) => term.trim()).filter(Boolean).slice(0, 5);
  const transcriptEvidence = transcriptSentences[0]?.replace(/\s+/g, " ").trim();
  const reason = explicitReason
    || (terms.length > 0
      ? `This clip connects to ${reel.concept_title} through ${terms.join(", ")}.`
      : transcriptEvidence
        ? `The transcript directly supports ${reel.concept_title}: ${transcriptEvidence.slice(0, 180)}`
        : `This clip was selected because it explains ${reel.concept_title}, a concept in this study session.`);

  return { summary, takeaways, reason };
}

function withAssessmentAccuracy(
  session: AssessmentSession,
  response: Pick<AssessmentStatusResponse, "recent_accuracy" | "rolling_accuracy">,
): AssessmentSession {
  return {
    ...session,
    recent_accuracy: session.recent_accuracy ?? response.recent_accuracy,
    rolling_accuracy: session.rolling_accuracy ?? response.rolling_accuracy,
  };
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
      const watchedFrontierIndex = Math.max(
        activeIndex,
        Math.floor(Number(row.watchedFrontierIndex) || activeIndex),
      );
      const feedbackByReel: Record<string, ReelFeedbackState> = {};
      if (row.feedbackByReel && typeof row.feedbackByReel === "object" && !Array.isArray(row.feedbackByReel)) {
        for (const [reelId, rawFeedback] of Object.entries(row.feedbackByReel as Record<string, unknown>)) {
          if (!reelId || !rawFeedback || typeof rawFeedback !== "object" || Array.isArray(rawFeedback)) {
            continue;
          }
          const feedback = rawFeedback as Record<string, unknown>;
          feedbackByReel[reelId] = {
            helpful: feedback.helpful === true,
            confusing: feedback.confusing === true,
            saved: feedback.saved === true,
            rating: Number.isFinite(Number(feedback.rating)) ? Number(feedback.rating) : undefined,
          };
        }
      }
      const adaptiveExcludeReelIds = Array.isArray(row.adaptiveExcludeReelIds)
        ? row.adaptiveExcludeReelIds
            .map((value) => String(value || "").trim())
            .filter(Boolean)
            .slice(-200)
        : [];
      result[materialId] = {
        reels,
        feedbackByReel,
        adaptiveExcludeReelIds,
        page,
        total,
        canRequestMore: row.canRequestMore !== false,
        generationMode: row.generationMode === "fast" ? "fast" : "slow",
        mutedPreference: row.mutedPreference !== false,
        autoplayEnabled: row.autoplayEnabled === true || row.autoplayEnabled === "1" || row.autoplayEnabled === "true",
        playbackRate: normalizeFeedPlaybackRate(row.playbackRate),
        activeIndex,
        watchedFrontierIndex,
        activeReelId: typeof row.activeReelId === "string" && row.activeReelId.trim() ? row.activeReelId.trim() : undefined,
        updatedAt: Number(row.updatedAt) || 0,
      };
    }
    return result;
  } catch {
    return {};
  }
}

function normalizeFeedPlaybackRate(value: unknown): number {
  const parsed = Number(value);
  return FEED_PLAYBACK_RATE_OPTIONS.includes(parsed as (typeof FEED_PLAYBACK_RATE_OPTIONS)[number]) ? parsed : 1;
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

/** Format a caption timestamp as `M:SS` or `H:MM:SS`. Mirrors iOS FeedView helper. */
function formatCaptionTimestamp(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) {
    return "0:00";
  }
  const total = Math.round(seconds);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  if (hours > 0) {
    return `${hours}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
  }
  return `${minutes}:${String(secs).padStart(2, "0")}`;
}

function selectStoredSessionWindow(
  reels: Reel[],
  activeIndex: number,
  maxReels: number,
): { reels: Reel[]; activeIndex: number; startIndex: number } {
  if (reels.length <= maxReels) {
    return {
      reels,
      activeIndex: clamp(activeIndex, 0, Math.max(0, reels.length - 1)),
      startIndex: 0,
    };
  }
  const clampedActiveIndex = clamp(activeIndex, 0, reels.length - 1);
  const leadingCount = Math.floor(maxReels / 2);
  const start = clamp(clampedActiveIndex - leadingCount, 0, reels.length - maxReels);
  const nextReels = reels.slice(start, start + maxReels);
  return {
    reels: nextReels,
    activeIndex: clamp(clampedActiveIndex - start, 0, Math.max(0, nextReels.length - 1)),
    startIndex: start,
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
  const { reels, activeIndex, startIndex } = selectStoredSessionWindow(
    snapshot.reels,
    snapshot.activeIndex,
    mode === "minimal" ? MINIMAL_REELS_PER_FEED_SESSION : COMPACT_REELS_PER_FEED_SESSION,
  );
  const compactedReels = reels.map((reel) => compactStoredReel(reel, mode));
  const storedReelIds = new Set(compactedReels.map((reel) => reel.reel_id));
  const feedbackByReel = Object.fromEntries(
    Object.entries(snapshot.feedbackByReel).filter(([reelId]) => storedReelIds.has(reelId)),
  );
  const nextActiveIndex = compactedReels.length > 0 ? clamp(activeIndex, 0, compactedReels.length - 1) : 0;
  const nextWatchedFrontierIndex = compactedReels.length > 0
    ? clamp(
        Math.max(snapshot.activeIndex, snapshot.watchedFrontierIndex ?? snapshot.activeIndex) - startIndex,
        nextActiveIndex,
        compactedReels.length - 1,
      )
    : 0;
  return {
    ...snapshot,
    reels: compactedReels,
    feedbackByReel,
    activeIndex: nextActiveIndex,
    watchedFrontierIndex: nextWatchedFrontierIndex,
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

function sourceVideoKeyFromUrl(value: string): string {
  const trimmed = String(value || "").trim();
  if (!trimmed) {
    return "";
  }
  try {
    const parsed = new URL(trimmed);
    if (parsed.pathname.startsWith("/embed/")) {
      return parsed.pathname.split("/")[2] ?? "";
    }
    if (parsed.pathname.startsWith("/shorts/")) {
      return parsed.pathname.split("/")[2] ?? "";
    }
    if (parsed.hostname.includes("youtu.be")) {
      return parsed.pathname.replace("/", "") || "";
    }
    const youtubeId = parsed.searchParams.get("v");
    if (youtubeId?.trim()) {
      return youtubeId.trim();
    }
    parsed.search = "";
    parsed.hash = "";
    return parsed.toString();
  } catch {
    return trimmed;
  }
}

function preferNonEmptyString(nextValue: string | undefined, currentValue: string | undefined): string | undefined {
  return typeof nextValue === "string" && nextValue.trim() ? nextValue : currentValue;
}

function mergeReelMetadata(current: Reel, next: Reel): Reel {
  return {
    ...current,
    ...next,
    video_title: preferNonEmptyString(next.video_title, current.video_title),
    video_description: preferNonEmptyString(next.video_description, current.video_description),
    ai_summary: preferNonEmptyString(next.ai_summary, current.ai_summary),
    transcript_snippet: preferNonEmptyString(next.transcript_snippet, current.transcript_snippet) ?? "",
    relevance_reason: preferNonEmptyString(next.relevance_reason, current.relevance_reason),
    match_reason: preferNonEmptyString(next.match_reason, current.match_reason),
    takeaways: next.takeaways?.length ? next.takeaways : current.takeaways,
    matched_terms: next.matched_terms?.length ? next.matched_terms : current.matched_terms,
    captions: next.captions?.length ? next.captions : current.captions,
    video_description_truncated: next.video_description_truncated ?? current.video_description_truncated,
    ai_summary_truncated: next.ai_summary_truncated ?? current.ai_summary_truncated,
    transcript_snippet_truncated: next.transcript_snippet_truncated ?? current.transcript_snippet_truncated,
    relevance_score: Number.isFinite(next.relevance_score) ? next.relevance_score : current.relevance_score,
    informativeness: Number.isFinite(next.informativeness) ? next.informativeness : current.informativeness,
    clip_duration_sec: Number.isFinite(next.clip_duration_sec) ? next.clip_duration_sec : current.clip_duration_sec,
    video_duration_sec: Number.isFinite(next.video_duration_sec) ? next.video_duration_sec : current.video_duration_sec,
  };
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
  // Ingest-only sentinel materials have reels persisted by /api/ingest/search or
  // /api/ingest/url and primed into the feed session snapshot in
  // `UploadPanel.primeFeedSessionSnapshot`. They do not have an independently
  // searchable material record, so bootstrap and load-more stay disabled.
  const isIngestMaterial = useMemo(() => {
    const id = (materialId || "").trim();
    return id.startsWith("ingest-search:") || id === "ingest-scratch";
  }, [materialId]);
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
  const [autoplayEnabled, setAutoplayEnabled] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [chatByReel, setChatByReel] = useState<Record<string, ChatMessage[]>>({});
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);
  const [chatPanelOpen, setChatPanelOpen] = useState(true);
  const [assessmentSession, setAssessmentSession] = useState<AssessmentSession | null>(null);
  const [assessmentQuestionIndex, setAssessmentQuestionIndex] = useState(0);
  const [assessmentAnswerReveal, setAssessmentAnswerReveal] = useState<RecallAnswerReveal | null>(null);
  const [assessmentAnswering, setAssessmentAnswering] = useState(false);
  const [assessmentResultsVisible, setAssessmentResultsVisible] = useState(false);
  const [assessmentPreparingFeed, setAssessmentPreparingFeed] = useState(false);
  const [assessmentSnoozing, setAssessmentSnoozing] = useState(false);
  const [assessmentError, setAssessmentError] = useState<string | null>(null);
  const [assessmentBootstrapPending, setAssessmentBootstrapPending] = useState(true);
  const [assessmentGatePending, setAssessmentGatePending] = useState(false);
  const [assessmentAdvanceAfterClose, setAssessmentAdvanceAfterClose] = useState(false);
  const [rightPanelWidthPx, setRightPanelWidthPx] = useState(360);
  const [rightTopRatio, setRightTopRatio] = useState(0.62);
  const [generationMode, setGenerationMode] = useState<GenerationMode>("slow");
  const [knowledgeLevel, setKnowledgeLevel] = useState<string | null>(null);
  const [sessionHydrated, setSessionHydrated] = useState(false);
  const [initialFeedScreenReady, setInitialFeedScreenReady] = useState(false);
  const [authAccountId, setAuthAccountId] = useState<string | null>(null);
  const [authScopeHydrated, setAuthScopeHydrated] = useState(false);
  const [settingsScopeReady, setSettingsScopeReady] = useState(false);
  const [recoveryPhase, setRecoveryPhase] = useState<FeedRecoveryPhase>("idle");
  const [feedPagesExhausted, setFeedPagesExhausted] = useState(false);
  const [generationProgress, setGenerationProgress] = useState<{ received: number; reconnecting: boolean } | null>(null);
  const [pendingTailAdvance, setPendingTailAdvance] = useState(false);

  const feedViewportRef = useRef<HTMLDivElement | null>(null);
  const isFetchingRef = useRef(false);
  const isGeneratingRef = useRef(false);
  const activeIndexRef = useRef(0);
  const watchedFrontierIndexRef = useRef(0);
  const reelsRef = useRef<Reel[]>([]);
  const pendingAutoplayAdvanceRef = useRef(false);
  const stepLockUntilRef = useRef(0);
  const isTransitioningRef = useRef(false);
  const transitionUnlockTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const wheelAccumRef = useRef(0);
  const wheelGestureConsumedRef = useRef(false);
  const wheelReadyToRearmRef = useRef(false);
  const wheelGestureReleaseTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const touchStartYRef = useRef<number | null>(null);
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
  const autoplayRestoredFromSnapshotRef = useRef(false);
  const hydratedMaterialIdRef = useRef<string | null>(null);
  const materialIdsForFeedRef = useRef<string[]>([]);
  const knowledgeLevelByMaterialRef = useRef<Map<string, string>>(new Map());
  const settingsLoadSequenceRef = useRef(0);
  const adaptiveExcludeReelIdsRef = useRef<string[]>([]);
  const isRefreshingFeedRef = useRef(false);
  const pendingHistorySyncRef = useRef<StoredHistoryItem[] | null>(null);
  const historySyncTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const progressClearTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const generationJobByMaterialRef = useRef<Map<string, ActiveGenerationJob>>(new Map());
  const generationConsumerByMaterialRef = useRef<Map<string, { jobId: string; token: symbol }>>(new Map());
  const generationBatchTokensRef = useRef<Set<symbol>>(new Set());
  const generationExhaustedUntilRef = useRef<Map<string, number>>(new Map());
  const generationExhaustedTimerRef = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map());
  const recoveryRequestIdleTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const visibleTransportErrorRef = useRef(false);
  const transportFailureStreakRef = useRef(0);
  const activeRecoveryRequestRef = useRef<ActiveRecoveryRequest | null>(null);
  const chatAbortControllerRef = useRef<AbortController | null>(null);
  const assessmentProgressMaxRef = useRef<Map<string, number>>(new Map());
  const reportedForwardScrollKeysRef = useRef<Set<string>>(new Set());
  const assessmentStartRequestRef = useRef<{ key: string; seq: number; reelId: string } | null>(null);
  const activeSearchScopeRef = useRef<FeedSearchScope>({
    key: "",
    seq: 0,
    controller: new AbortController(),
  });

  const syncGenerationLockState = useCallback(() => {
    const active = generationBatchTokensRef.current.size > 0 || generationConsumerByMaterialRef.current.size > 0;
    isGeneratingRef.current = active;
    setGeneratingMore(active);
    setRecoveryPhase(active ? "generating" : "idle");
  }, []);

  const claimGenerationConsumer = useCallback((materialIdValue: string, jobId: string): symbol | null => {
    if (generationConsumerByMaterialRef.current.has(materialIdValue)) {
      return null;
    }
    const token = Symbol(jobId || materialIdValue);
    generationConsumerByMaterialRef.current.set(materialIdValue, { jobId, token });
    syncGenerationLockState();
    return token;
  }, [syncGenerationLockState]);

  const releaseGenerationConsumer = useCallback((materialIdValue: string, token: symbol) => {
    if (generationConsumerByMaterialRef.current.get(materialIdValue)?.token === token) {
      generationConsumerByMaterialRef.current.delete(materialIdValue);
    }
    syncGenerationLockState();
  }, [syncGenerationLockState]);

  const abortActiveSearchScope = useCallback(() => {
    const scope = activeSearchScopeRef.current;
    if (!scope.controller.signal.aborted) {
      scope.controller.abort();
    }
  }, []);

  const abortActiveChat = useCallback(() => {
    chatAbortControllerRef.current?.abort();
    chatAbortControllerRef.current = null;
  }, []);

  const resetActiveSearchRequestState = useCallback(() => {
    isFetchingRef.current = false;
    isGeneratingRef.current = false;
    isFastTopUpRef.current = false;
    isRefreshingFeedRef.current = false;
    isRecoveringMissingMaterialRef.current = false;
    generationConsumerByMaterialRef.current.clear();
    generationBatchTokensRef.current.clear();
    resumeLoadingRef.current = false;
    activeRecoveryRequestRef.current = null;
    if (recoveryRequestIdleTimerRef.current) {
      clearTimeout(recoveryRequestIdleTimerRef.current);
      recoveryRequestIdleTimerRef.current = null;
    }
    setGeneratingMore(false);
    setBootstrappingFirstReels(false);
    setRecoveryPhase("idle");
  }, []);

  const isSearchScopeActive = useCallback((scope: Pick<FeedSearchScope, "key" | "seq">): boolean => {
    const current = activeSearchScopeRef.current;
    return current.key === scope.key && current.seq === scope.seq && !current.controller.signal.aborted;
  }, []);

  const clearRecoveryRequestIdleTimer = useCallback(() => {
    if (recoveryRequestIdleTimerRef.current) {
      clearTimeout(recoveryRequestIdleTimerRef.current);
      recoveryRequestIdleTimerRef.current = null;
    }
  }, []);

  const renewActiveSearchScope = useCallback(() => {
    abortActiveSearchScope();
    clearRecoveryRequestIdleTimer();
    activeRecoveryRequestRef.current = null;
    const previous = activeSearchScopeRef.current;
    activeSearchScopeRef.current = {
      key: feedRouteKey,
      seq: previous.seq + 1,
      controller: new AbortController(),
    };
    resetActiveSearchRequestState();
  }, [abortActiveSearchScope, clearRecoveryRequestIdleTimer, feedRouteKey, resetActiveSearchRequestState]);

  useEffect(() => {
    renewActiveSearchScope();
    return () => {
      abortActiveChat();
      clearRecoveryRequestIdleTimer();
      const current = activeSearchScopeRef.current;
      if (current.key === feedRouteKey) {
        current.controller.abort();
      }
    };
  }, [abortActiveChat, clearRecoveryRequestIdleTimer, feedRouteKey, renewActiveSearchScope]);

  useEffect(() => {
    const abortPageRequests = () => {
      abortActiveChat();
      abortActiveSearchScope();
    };
    window.addEventListener("beforeunload", abortPageRequests);
    window.addEventListener("pagehide", abortPageRequests);
    window.addEventListener("unload", abortPageRequests);
    return () => {
      window.removeEventListener("beforeunload", abortPageRequests);
      window.removeEventListener("pagehide", abortPageRequests);
      window.removeEventListener("unload", abortPageRequests);
    };
  }, [abortActiveChat, abortActiveSearchScope]);

  const normalizeClipKeyTime = useCallback((value: unknown): string => {
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) {
      return "0.00";
    }
    return (Math.round(parsed * 1000) / 1000).toFixed(3);
  }, []);

  const reelClipKey = useCallback((reel: Reel): string => {
    const videoId = sourceVideoKeyFromUrl(reel.video_url) || reel.video_url;
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

  const mergeSessionReels = useCallback(
    (rows: Reel[], existing: Reel[] = []): SessionMergeResult => {
      const nextRows = dedupeByIdentity(rows);
      const merged = existing.map((reel) => ({ ...reel }));
      const addedReels: Reel[] = [];
      const clipIndexByKey = new Map<string, number>();
      const reelIndexById = new Map<string, number>();

      const registerReel = (reel: Reel, index: number) => {
        const clipKey = reelClipKey(reel);
        const reelId = String(reel.reel_id || "").trim();
        clipIndexByKey.set(clipKey, index);
        if (reelId) {
          reelIndexById.set(reelId, index);
        }
      };

      merged.forEach(registerReel);

      let updatedCount = 0;
      for (const reel of nextRows) {
        const clipKey = reelClipKey(reel);
        const reelId = String(reel.reel_id || "").trim();
        const existingIndex =
          clipIndexByKey.get(clipKey)
          ?? (reelId ? reelIndexById.get(reelId) : undefined);
        if (existingIndex != null) {
          const current = merged[existingIndex];
          const updated = mergeReelMetadata(current, reel);
          merged[existingIndex] = updated;
          registerReel(updated, existingIndex);
          updatedCount += 1;
          continue;
        }
        merged.push(reel);
        addedReels.push(reel);
        registerReel(reel, merged.length - 1);
      }

      return {
        reels: merged,
        addedReels,
        addedCount: addedReels.length,
        updatedCount,
      };
    },
    [dedupeByIdentity, reelClipKey],
  );

  const updateSessionReels = useCallback((nextReels: Reel[]) => {
    reelsRef.current = nextReels;
    setReels(nextReels);
  }, []);

  const orderReelsByDifficulty = useCallback((rows: Reel[]): Reel[] => {
    return rows
      .map((reel, index) => ({
        reel,
        index,
        difficulty: typeof reel.difficulty === "number" && Number.isFinite(reel.difficulty)
          ? Math.min(1, Math.max(0, reel.difficulty))
          : 0.5,
      }))
      .sort((left, right) => left.difficulty - right.difficulty || left.index - right.index)
      .map(({ reel }) => reel);
  }, []);

  const mergeReelBatchesByDifficulty = useCallback(
    (batches: Reel[][]): Reel[] => orderReelsByDifficulty(batches.flatMap((batch) => batch)),
    [orderReelsByDifficulty],
  );

  const getFeedMaterialIds = useCallback((): string[] => {
    const ids = materialIdsForFeedRef.current
      .map((id) => id.trim())
      .filter(Boolean);
    if (ids.length > 0) {
      return ids;
    }
    return materialId ? [materialId] : [];
  }, [materialId]);

  const clearGenerationTracking = useCallback(() => {
    generationJobByMaterialRef.current.clear();
    generationConsumerByMaterialRef.current.clear();
    generationBatchTokensRef.current.clear();
    generationExhaustedUntilRef.current.clear();
    for (const timer of generationExhaustedTimerRef.current.values()) {
      clearTimeout(timer);
    }
    generationExhaustedTimerRef.current.clear();
    setPendingTailAdvance(false);
  }, []);

  const isGenerationCoolingDown = useCallback((materialIdValue: string): boolean => {
    const until = generationExhaustedUntilRef.current.get(materialIdValue) ?? 0;
    if (until > Date.now()) {
      return true;
    }
    generationExhaustedUntilRef.current.delete(materialIdValue);
    return false;
  }, []);

  const coolDownExhaustedMaterial = useCallback((materialIdValue: string) => {
    const id = String(materialIdValue || "").trim();
    if (!id) {
      return;
    }
    generationExhaustedUntilRef.current.set(id, Date.now() + GENERATION_EXHAUSTED_COOLDOWN_MS);
    const existingTimer = generationExhaustedTimerRef.current.get(id);
    if (existingTimer) {
      clearTimeout(existingTimer);
    }
    generationExhaustedTimerRef.current.set(id, setTimeout(() => {
      generationExhaustedTimerRef.current.delete(id);
      generationExhaustedUntilRef.current.delete(id);
      if (!isIngestMaterial) {
        setCanRequestMore(true);
      }
    }, GENERATION_EXHAUSTED_COOLDOWN_MS));
    const materialIds = getFeedMaterialIds();
    if (materialIds.length > 0 && materialIds.every((materialIdKey) => isGenerationCoolingDown(materialIdKey))) {
      setCanRequestMore(false);
    }
  }, [getFeedMaterialIds, isGenerationCoolingDown, isIngestMaterial]);

  const noteGenerationTerminal = useCallback((materialIdValue: string, status: GenerationTerminalStatus) => {
    generationJobByMaterialRef.current.delete(materialIdValue);
    if (status === "exhausted") {
      coolDownExhaustedMaterial(materialIdValue);
    }
  }, [coolDownExhaustedMaterial]);

  const rememberFeedGenerationJob = useCallback((
    materialIdValue: string,
    response: Awaited<ReturnType<typeof fetchFeed>>,
  ) => {
    const jobId = String(response.generation_job_id || "").trim();
    const status = response.generation_job_status;
    if (jobId && (status == null || status === "queued" || status === "running")) {
      generationJobByMaterialRef.current.set(materialIdValue, { jobId, status: status ?? "running" });
      return;
    }
    if (status && status !== "queued" && status !== "running") {
      noteGenerationTerminal(materialIdValue, status);
    }
  }, [noteGenerationTerminal]);

  useEffect(() => () => {
    for (const timer of generationExhaustedTimerRef.current.values()) {
      clearTimeout(timer);
    }
    generationExhaustedTimerRef.current.clear();
  }, []);

  useEffect(() => {
    clearGenerationTracking();
  }, [clearGenerationTracking, feedRouteKey]);

  const mergeFeedSettingsSnapshot = useCallback(
    (settings: StudyReelsSettings): StudyReelsSettings => mergeSearchFeedQuerySettings(settings, searchFeedSettingsOverride),
    [searchFeedSettingsOverride],
  );

  const getFeedTuningSettings = useCallback((): FeedTuningSettings => {
    const settings = mergeFeedSettingsSnapshot(readStudyReelsSettings());
    return {
      minRelevance: settings.minRelevanceThreshold,
      creativeCommonsOnly: settings.creativeCommonsOnly,
      preferredVideoDuration: settings.preferredVideoDuration,
    };
  }, [mergeFeedSettingsSnapshot]);

  const setVisibleFeedError = useCallback((message: string | null, options?: { transport?: boolean }) => {
    visibleTransportErrorRef.current = Boolean(message && options?.transport);
    setError(message);
  }, []);

  const clearRecoveredTransportError = useCallback(() => {
    transportFailureStreakRef.current = 0;
    if (visibleTransportErrorRef.current) {
      visibleTransportErrorRef.current = false;
      setError(null);
    }
  }, []);

  const noteFeedTransportFailure = useCallback((failure: unknown, options?: { forceVisible?: boolean }) => {
    const message = failure instanceof Error ? failure.message : "Cannot reach backend right now.";
    transportFailureStreakRef.current += 1;
    const shouldSurface = Boolean(options?.forceVisible)
      || transportFailureStreakRef.current >= 2
      || reelsRef.current.length === 0;
    if (shouldSurface) {
      setVisibleFeedError(message, { transport: true });
    }
    return shouldSurface;
  }, [setVisibleFeedError]);

  const noteFeedFailure = useCallback((failure: unknown) => {
    transportFailureStreakRef.current = 0;
    setVisibleFeedError(failure instanceof Error ? failure.message : "Feed failed to load");
  }, [setVisibleFeedError]);

  const markPagedFeedExhausted = useCallback(() => {
    const visibleCount = reelsRef.current.length;
    setFeedPagesExhausted(true);
    setTotal((prevTotal) => {
      if (visibleCount <= 0) {
        return 0;
      }
      if (prevTotal <= 0) {
        return visibleCount;
      }
      return Math.min(prevTotal, visibleCount);
    });
    if (visibleCount > 0) {
      setPage((prevPage) => Math.min(prevPage, Math.max(1, Math.ceil(visibleCount / PAGE_SIZE))));
    }
  }, []);

  const markRecoveryProgress = useCallback((addedCount: number) => {
    if (addedCount > 0) {
      setFeedPagesExhausted(false);
      clearRecoveredTransportError();
    }
  }, [clearRecoveredTransportError]);

  const finishActiveRecoveryRequest = useCallback((scope: Pick<FeedSearchScope, "key" | "seq">) => {
    const current = activeRecoveryRequestRef.current;
    if (!current || current.key !== scope.key || current.seq !== scope.seq) {
      return;
    }
    activeRecoveryRequestRef.current = null;
    clearRecoveryRequestIdleTimer();
  }, [clearRecoveryRequestIdleTimer]);

  const armActiveRecoveryRequest = useCallback((
    scope: Pick<FeedSearchScope, "key" | "seq">,
    phase: Extract<FeedRecoveryPhase, "fetching-page" | "generating">,
  ) => {
    activeRecoveryRequestRef.current = { key: scope.key, seq: scope.seq, phase };
    clearRecoveryRequestIdleTimer();
    // Durable generation uses a separate non-destructive 35-second reader
    // watchdog that status-polls and reconnects to the same persisted job.
    if (phase === "generating") {
      return;
    }
    recoveryRequestIdleTimerRef.current = setTimeout(() => {
      const current = activeRecoveryRequestRef.current;
      if (!current || current.key !== scope.key || current.seq !== scope.seq) {
        return;
      }
      if (!isSearchScopeActive(scope)) {
        finishActiveRecoveryRequest(scope);
        return;
      }
      if (!isFetchingRef.current && !isGeneratingRef.current) {
        finishActiveRecoveryRequest(scope);
        return;
      }
      console.warn("Recovery request stalled; restarting search scope.", current);
      if (current.phase === "fetching-page" && reelsRef.current.length > 0) {
        markPagedFeedExhausted();
      }
      activeRecoveryRequestRef.current = null;
      clearRecoveryRequestIdleTimer();
      markRecoveryProgress(0);
      if (current.phase !== "fetching-page" || reelsRef.current.length === 0) {
        setFeedPagesExhausted(false);
      }
      renewActiveSearchScope();
    }, RECOVERY_REQUEST_IDLE_TIMEOUT_MS);
  }, [
    clearRecoveryRequestIdleTimer,
    finishActiveRecoveryRequest,
    isSearchScopeActive,
    markPagedFeedExhausted,
    markRecoveryProgress,
    renewActiveSearchScope,
  ]);

  const hasMore = !feedPagesExhausted && reels.length < total;
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
      if (!autoplayRestoredFromSnapshotRef.current) {
        setAutoplayEnabled(mergedSettings.autoplayNextReel);
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
    recall?: StoredRecallSummary;
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
        recall: options?.recall ?? existing?.recall,
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
        || existing.activeReelId !== nextEntry.activeReelId
        || JSON.stringify(existing.recall ?? null) !== JSON.stringify(nextEntry.recall ?? null);
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
      setVisibleFeedError("Session expired on server. Rebuilding your material...");
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
        setVisibleFeedError(e instanceof Error ? e.message : "Could not rebuild material.");
        return false;
      } finally {
        if (isSearchScopeActive(searchScope)) {
          isRecoveringMissingMaterialRef.current = false;
        }
      }
    },
    [isSearchScopeActive, params, router, setVisibleFeedError],
  );

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
    const unseenReadyCount = Math.max(0, reels.length - activeIndex - 1);
    return reels.length === 0 || unseenReadyCount <= READY_RESERVOIR_REFILL_THRESHOLD;
  }, [activeIndex, getFeedMaterialIds, reels.length]);

  const appendGeneratedReels = useCallback(
    (generated: Reel[]): SessionMergeResult => {
      if (!generated.length) {
        return {
          reels: reelsRef.current,
          addedReels: [],
          addedCount: 0,
          updatedCount: 0,
        };
      }
      const merged = mergeSessionReels(generated, reelsRef.current);
      if (merged.addedCount > 0 || merged.updatedCount > 0) {
        updateSessionReels(merged.reels);
        setTotal((prevTotal) => Math.max(prevTotal, merged.reels.length));
      }
      return merged;
    },
    [mergeSessionReels, updateSessionReels],
  );

  const reconcileGeneratedReels = useCallback(
    (
      _provisional: Reel[],
      finalInventory: Reel[],
      options: { preserveUnmatchedUnseen: boolean },
    ): SessionMergeResult => {
      const currentRows = reelsRef.current;
      const lockedPrefixLength = Math.min(
        currentRows.length,
        Math.max(activeIndexRef.current, watchedFrontierIndexRef.current) + 1,
      );
      const authoritativeById = new Map(
        finalInventory
          .map((reel) => [String(reel.reel_id || "").trim(), reel] as const)
          .filter(([reelId]) => Boolean(reelId)),
      );
      const authoritativeByClip = new Map(finalInventory.map((reel) => [reelClipKey(reel), reel] as const));
      const consumedAuthoritative = new Set<Reel>();
      const lockedPrefix = currentRows.slice(0, lockedPrefixLength).map((reel) => {
        const reelId = String(reel.reel_id || "").trim();
        const authoritative = (reelId ? authoritativeById.get(reelId) : undefined)
          ?? authoritativeByClip.get(reelClipKey(reel));
        if (!authoritative) {
          return reel;
        }
        consumedAuthoritative.add(authoritative);
        return mergeReelMetadata(reel, authoritative);
      });
      const stableUnseenRows = currentRows.slice(lockedPrefixLength);
      const provisionalReelIds = new Set(
        _provisional
          .map((reel) => String(reel.reel_id || "").trim())
          .filter(Boolean),
      );
      const provisionalClipKeys = new Set(_provisional.map((reel) => reelClipKey(reel)));
      const authoritativeUnseen = finalInventory.filter((reel) => !consumedAuthoritative.has(reel));
      const stableUnseenById = new Map(
        stableUnseenRows
          .map((reel) => [String(reel.reel_id || "").trim(), reel] as const)
          .filter(([reelId]) => Boolean(reelId)),
      );
      const stableUnseenByClip = new Map(stableUnseenRows.map((reel) => [reelClipKey(reel), reel] as const));
      const consumedStableUnseen = new Set<Reel>();
      const authoritativeTail = authoritativeUnseen.map((reel) => {
        const reelId = String(reel.reel_id || "").trim();
        const stable = (reelId ? stableUnseenById.get(reelId) : undefined)
          ?? stableUnseenByClip.get(reelClipKey(reel));
        if (!stable) {
          return reel;
        }
        consumedStableUnseen.add(stable);
        return mergeReelMetadata(stable, reel);
      });
      // Retain pre-existing unseen inventory during stream settlement. A
      // provisional candidate that is absent from the authoritative final was
      // intentionally removed; current-contract candidates already appear in
      // that final inventory and are merged above.
      // Restored snapshots are only a cache, so their unmatched unseen rows
      // must be removed when page one supplies the authoritative inventory.
      if (options.preserveUnmatchedUnseen) {
        authoritativeTail.push(...stableUnseenRows.filter((reel) => {
          if (consumedStableUnseen.has(reel)) {
            return false;
          }
          const reelId = String(reel.reel_id || "").trim();
          return !(reelId && provisionalReelIds.has(reelId))
            && !provisionalClipKeys.has(reelClipKey(reel));
        }));
      }
      const orderedTail = orderReelsByDifficulty(authoritativeTail);
      const reordered = dedupeByIdentity([...lockedPrefix, ...orderedTail]);
      const previousIdentity = new Set(currentRows.map((reel) => `${String(reel.reel_id || "").trim()}|${reelClipKey(reel)}`));
      const addedReels = reordered.filter(
        (reel) => !previousIdentity.has(`${String(reel.reel_id || "").trim()}|${reelClipKey(reel)}`),
      );
      const merged: SessionMergeResult = {
        reels: reordered,
        addedReels,
        addedCount: addedReels.length,
        updatedCount: Math.max(0, reordered.length - addedReels.length),
      };
      updateSessionReels(reordered);
      setTotal((prevTotal) => Math.max(prevTotal, reordered.length));
      return merged;
    },
    [dedupeByIdentity, orderReelsByDifficulty, reelClipKey, updateSessionReels],
  );

  const consumeFeedGenerationJob = useCallback((
    materialIdValue: string,
    response: Awaited<ReturnType<typeof fetchFeed>>,
    searchScope: FeedSearchScope,
  ): Promise<void> | null => {
    const jobId = String(response.generation_job_id || "").trim();
    const status = response.generation_job_status;
    if (!jobId || (status != null && status !== "queued" && status !== "running")) {
      return null;
    }
    const token = claimGenerationConsumer(materialIdValue, jobId);
    if (!token) {
      return null;
    }

    const streamedReels: Reel[] = [];
    return (async () => {
      try {
        const data = await generateReelsStream({
          materialId: materialIdValue,
          generationJobId: jobId,
          signal: searchScope.controller.signal,
          idleTimeoutMs: GENERATION_STREAM_IDLE_TIMEOUT_MS,
          onTerminal: (terminalStatus) => {
            if (isSearchScopeActive(searchScope)) {
              noteGenerationTerminal(materialIdValue, terminalStatus);
            }
          },
          onCandidate: (reel) => {
            if (!isSearchScopeActive(searchScope)) {
              return;
            }
            const appended = appendGeneratedReels([reel]);
            if (appended.addedCount > 0) {
              streamedReels.push(...appended.addedReels);
              markRecoveryProgress(appended.addedCount);
            }
          },
        });
        if (!isSearchScopeActive(searchScope)) {
          return;
        }
        reconcileGeneratedReels(streamedReels, data.reels, { preserveUnmatchedUnseen: true });
      } catch (error) {
        if (isSearchScopeActive(searchScope) && !isRequestInterruptedError(error)) {
          console.warn(`Feed generation stream failed for topic material ${materialIdValue}:`, error);
        }
      } finally {
        if (isSearchScopeActive(searchScope)) {
          releaseGenerationConsumer(materialIdValue, token);
        }
      }
    })();
  }, [
    appendGeneratedReels,
    isSearchScopeActive,
    markRecoveryProgress,
    noteGenerationTerminal,
    claimGenerationConsumer,
    reconcileGeneratedReels,
    releaseGenerationConsumer,
  ]);

  const loadPage = useCallback(
    async (
      targetPage: number,
      options?: { autofill?: boolean; preserveSession?: boolean; generationMode?: GenerationMode },
    ): Promise<{ addedCount: number; exhausted: boolean }> => {
      const feedMaterialIds = getFeedMaterialIds();
      if (!settingsScopeReady || feedMaterialIds.length === 0 || isFetchingRef.current) {
        return { addedCount: 0, exhausted: feedPagesExhausted };
      }
      const searchScope = activeSearchScopeRef.current;
      isFetchingRef.current = true;
      setRecoveryPhase("fetching-page");
      armActiveRecoveryRequest(searchScope, "fetching-page");

      try {
        const tuning = getFeedTuningSettings();
        const requestGenerationMode = options?.generationMode ?? generationMode;
        const requestLimit = adaptiveExcludeReelIdsRef.current.length > 0 ? 25 : PAGE_SIZE;
        const rows = await Promise.all(
          feedMaterialIds.map(async (id) => {
            try {
              const data = await fetchFeed({
                materialId: id,
                page: targetPage,
                limit: requestLimit,
                excludeReelIds: adaptiveExcludeReelIdsRef.current,
                autofill: options?.autofill ?? true,
                prefetch: readyReservoirTarget(requestGenerationMode),
                generationMode: requestGenerationMode,
                minRelevance: tuning.minRelevance,
                creativeCommonsOnly: tuning.creativeCommonsOnly,
                preferredVideoDuration: tuning.preferredVideoDuration,
                signal: searchScope.controller.signal,
              });
              return { materialId: id, data, error: null };
            } catch (error) {
              return { materialId: id, data: null, error };
            }
          }),
        );
        if (!isSearchScopeActive(searchScope)) {
          return { addedCount: 0, exhausted: feedPagesExhausted };
        }

        const successful = rows.filter((row) => row.data);
        if (successful.length === 0) {
          markRecoveryProgress(0);
          if (targetPage > 1 && reelsRef.current.length > 0) {
            markPagedFeedExhausted();
          }
          const firstError = rows[0]?.error;
          if (isTransportError(firstError)) {
            noteFeedTransportFailure(firstError, {
              forceVisible: targetPage === 1 && reelsRef.current.length === 0,
            });
          }
          if (targetPage === 1) {
            if (isRequestInterruptedError(firstError)) {
              return { addedCount: 0, exhausted: feedPagesExhausted };
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
                return { addedCount: 0, exhausted: false };
              }
            }
            if (!isTransportError(firstError)) {
              noteFeedFailure(firstError ?? message);
            }
          }
          setRecoveryPhase("idle");
          return { addedCount: 0, exhausted: feedPagesExhausted };
        }

        for (const row of successful) {
          rememberFeedGenerationJob(row.materialId, row.data!);
        }

        const fetchedReels = dedupeByIdentity(mergeReelBatchesByDifficulty(successful.map((row) => row.data!.reels)));
        const fetchedTotal = successful.reduce((sum, row) => sum + Math.max(0, Number(row.data!.total) || 0), 0);
        const merged = targetPage === 1
          ? options?.preserveSession
            ? reconcileGeneratedReels([], fetchedReels, { preserveUnmatchedUnseen: false })
            : mergeSessionReels(fetchedReels)
          : mergeSessionReels(fetchedReels, reelsRef.current);
        const exhausted = successful.every((row) => {
          const rowTotal = Math.max(0, Number(row.data!.total) || 0);
          return row.data!.reels.length === 0 || targetPage * requestLimit >= rowTotal;
        });

        setPage(targetPage);
        if (targetPage === 1) {
          setTotal(Math.max(fetchedTotal, merged.reels.length));
          for (const row of successful) {
            const rowLevel = String(row.data?.knowledge_level || "").trim();
            if (rowLevel) {
              knowledgeLevelByMaterialRef.current.set(row.materialId, rowLevel);
            }
          }
          const firstLevel = knowledgeLevelByMaterialRef.current.get(feedMaterialIds[0]);
          if (firstLevel && typeof firstLevel === "string") {
            setKnowledgeLevel(firstLevel);
          } else {
            setKnowledgeLevel(null);
          }
        } else {
          setTotal((prevTotal) => Math.max(prevTotal, fetchedTotal, merged.reels.length));
        }
        if (merged.addedCount > 0 || merged.updatedCount > 0 || targetPage === 1) {
          updateSessionReels(merged.reels);
        }
        markRecoveryProgress(merged.addedCount);
        if (targetPage > 1 && merged.addedCount === 0 && exhausted) {
          markPagedFeedExhausted();
        } else {
          setFeedPagesExhausted(exhausted);
        }
        clearRecoveredTransportError();

        if (successful.length < rows.length) {
          const failedIds = rows.filter((row) => !row.data).map((row) => row.materialId);
          console.warn("Some topic feeds failed to load:", failedIds);
        }
        setRecoveryPhase("idle");
        for (const row of successful) {
          void consumeFeedGenerationJob(
            row.materialId,
            row.data!,
            searchScope,
          );
        }
        syncGenerationLockState();
        return { addedCount: merged.addedCount, exhausted };
      } catch (e) {
        if (!isSearchScopeActive(searchScope) || isRequestInterruptedError(e)) {
          return { addedCount: 0, exhausted: feedPagesExhausted };
        }
        markRecoveryProgress(0);
        if (targetPage > 1 && reelsRef.current.length > 0) {
          markPagedFeedExhausted();
        }
        if (isTransportError(e)) {
          noteFeedTransportFailure(e, {
            forceVisible: targetPage === 1 && reelsRef.current.length === 0,
          });
        } else if (targetPage === 1) {
          noteFeedFailure(e);
        }
        setRecoveryPhase("idle");
        return { addedCount: 0, exhausted: feedPagesExhausted };
      } finally {
        if (isSearchScopeActive(searchScope)) {
          finishActiveRecoveryRequest(searchScope);
          setLoading(false);
          isFetchingRef.current = false;
        }
      }
    },
    [
      armActiveRecoveryRequest,
      clearRecoveredTransportError,
      consumeFeedGenerationJob,
      dedupeByIdentity,
      feedPagesExhausted,
      generationMode,
      getFeedMaterialIds,
      getFeedTuningSettings,
      mergeReelBatchesByDifficulty,
      isSearchScopeActive,
      markRecoveryProgress,
      materialId,
      mergeSessionReels,
      markPagedFeedExhausted,
      noteFeedFailure,
      noteFeedTransportFailure,
      recoverMissingMaterial,
      reconcileGeneratedReels,
      rememberFeedGenerationJob,
      settingsScopeReady,
      syncGenerationLockState,
      updateSessionReels,
      finishActiveRecoveryRequest,
    ],
  );

  const requestMore = useCallback(async (options?: { surfaceError?: boolean }): Promise<Reel[]> => {
    const allFeedMaterialIds = getFeedMaterialIds();
    if (!settingsScopeReady || allFeedMaterialIds.length === 0 || isGeneratingRef.current || !canRequestMore) {
      return [];
    }
    // Ingest-search / ingest-scratch reels are already primed into the feed
    // session by UploadPanel.tsx. Bail out before submitting a generation job.
    if (isIngestMaterial) {
      setCanRequestMore(false);
      setFeedPagesExhausted(true);
      return [];
    }
    const feedMaterialIds = allFeedMaterialIds.filter((id) => !isGenerationCoolingDown(id));
    if (feedMaterialIds.length === 0) {
      setCanRequestMore(false);
      return [];
    }
    const searchScope = activeSearchScopeRef.current;
    const tuning = getFeedTuningSettings();
    const unseenReadyCount = Math.max(0, reelsRef.current.length - activeIndexRef.current - 1);
    const requestedReadyCount = Math.max(1, readyReservoirTarget(generationMode) - unseenReadyCount);
    const perTopicBatch = Math.max(1, Math.ceil(requestedReadyCount / feedMaterialIds.length));
    const generationBatchToken = Symbol("request-more");
    generationBatchTokensRef.current.add(generationBatchToken);
    syncGenerationLockState();
    armActiveRecoveryRequest(searchScope, "generating");
    if (progressClearTimerRef.current) {
      clearTimeout(progressClearTimerRef.current);
      progressClearTimerRef.current = null;
    }
    setGenerationProgress({ received: 0, reconnecting: false });
    let progressErrored = false;
    try {
      const generatedRows = await Promise.all(
        feedMaterialIds.map(async (id) => {
          const streamedReels: Reel[] = [];
          const currentCount = countReelsForMaterial(id);
          const targetTotal = Math.min(readyReservoirTarget(generationMode), currentCount + perTopicBatch);
          const activeGenerationJob = generationJobByMaterialRef.current.get(id);
          const consumerToken = claimGenerationConsumer(id, activeGenerationJob?.jobId || "new-generation");
          if (!consumerToken) {
            return { materialId: id, data: null, streamedReels, error: null };
          }
          try {
            const data = await generateReelsStream({
              materialId: id,
              numReels: targetTotal,
              generationJobId: activeGenerationJob?.jobId,
              generationMode,
              minRelevance: tuning.minRelevance,
              creativeCommonsOnly: tuning.creativeCommonsOnly,
              preferredVideoDuration: tuning.preferredVideoDuration,
              signal: searchScope.controller.signal,
              idleTimeoutMs: GENERATION_STREAM_IDLE_TIMEOUT_MS,
              onActivity: () => {
                if (isSearchScopeActive(searchScope)) {
                  setGenerationProgress((prev) => (prev ? { ...prev, reconnecting: false } : prev));
                }
              },
              onReconnect: (consecutiveIdleWindows) => {
                if (isSearchScopeActive(searchScope) && consecutiveIdleWindows >= 2) {
                  setGenerationProgress((prev) => (prev ? { ...prev, reconnecting: true } : prev));
                }
              },
              onTerminal: (status) => {
                if (isSearchScopeActive(searchScope)) {
                  noteGenerationTerminal(id, status);
                }
              },
              onCandidate: (reel) => {
                if (!isSearchScopeActive(searchScope)) {
                  return;
                }
                setGenerationProgress((prev) => (prev ? { ...prev, received: prev.received + 1 } : null));
                const appended = appendGeneratedReels([reel]);
                if (appended.addedCount > 0) {
                  streamedReels.push(...appended.addedReels);
                  armActiveRecoveryRequest(searchScope, "generating");
                }
              },
            });
            if (!isSearchScopeActive(searchScope)) {
              return { materialId: id, data: null, streamedReels, error: null };
            }
            reconcileGeneratedReels(streamedReels, data.reels, { preserveUnmatchedUnseen: true });
            armActiveRecoveryRequest(searchScope, "generating");
            return { materialId: id, data, streamedReels: dedupeByIdentity(data.reels), error: null };
          } catch (e) {
            if (!isRequestInterruptedError(e)) {
              console.warn(`Background reel generation failed for topic material ${id}:`, e);
            }
            return { materialId: id, data: null, streamedReels, error: e };
          } finally {
            if (isSearchScopeActive(searchScope)) {
              releaseGenerationConsumer(id, consumerToken);
            }
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
      const generated = dedupeByIdentity(
        mergeReelBatchesByDifficulty(generatedRows.map((row) => row.streamedReels ?? [])),
      );
      if (generated.length === 0) {
        markRecoveryProgress(0);
        if (isTransportError(firstFailedRow?.error)) {
          noteFeedTransportFailure(firstFailedRow?.error, { forceVisible: Boolean(options?.surfaceError) });
        } else if (options?.surfaceError) {
          if (firstFailedRow?.error) {
            noteFeedFailure(firstFailedRow.error);
          } else {
            noteFeedFailure("Still searching for fresh reels. No new source videos yet.");
          }
        }
        setRecoveryPhase("idle");
        return [];
      }
      markRecoveryProgress(generated.length);
      setFeedPagesExhausted(false);
      clearRecoveredTransportError();
      setRecoveryPhase("idle");
      return generated;
    } catch (e) {
      if (!isSearchScopeActive(searchScope) || isRequestInterruptedError(e)) {
        return [];
      }
      progressErrored = true;
      console.warn("Background reel generation failed:", e);
      markRecoveryProgress(0);
      if (isTransportError(e)) {
        noteFeedTransportFailure(e, { forceVisible: Boolean(options?.surfaceError) });
      } else if (options?.surfaceError) {
        noteFeedFailure(e instanceof Error ? e.message : "Could not generate reels right now.");
      }
      setRecoveryPhase("idle");
      return [];
    } finally {
      if (isSearchScopeActive(searchScope)) {
        generationBatchTokensRef.current.delete(generationBatchToken);
        syncGenerationLockState();
        finishActiveRecoveryRequest(searchScope);
        if (progressErrored && !isGeneratingRef.current) {
          setGenerationProgress(null);
        } else if (!isGeneratingRef.current) {
          progressClearTimerRef.current = setTimeout(() => {
            progressClearTimerRef.current = null;
            setGenerationProgress(null);
          }, 800);
        }
      } else {
        setGenerationProgress(null);
      }
    }
  }, [
    appendGeneratedReels,
    armActiveRecoveryRequest,
    canRequestMore,
    claimGenerationConsumer,
    clearRecoveredTransportError,
    countReelsForMaterial,
    dedupeByIdentity,
    generationMode,
    getFeedMaterialIds,
    getFeedTuningSettings,
    mergeReelBatchesByDifficulty,
    isIngestMaterial,
    isGenerationCoolingDown,
    isSearchScopeActive,
    markRecoveryProgress,
    materialId,
    noteFeedFailure,
    noteFeedTransportFailure,
    noteGenerationTerminal,
    recoverMissingMaterial,
    reconcileGeneratedReels,
    releaseGenerationConsumer,
    settingsScopeReady,
    syncGenerationLockState,
    finishActiveRecoveryRequest,
  ]);

  useEffect(() => {
    mutedRestoredFromSnapshotRef.current = false;
    autoplayRestoredFromSnapshotRef.current = false;
    if (!materialId) {
      watchedFrontierIndexRef.current = 0;
      materialIdsForFeedRef.current = [];
      knowledgeLevelByMaterialRef.current.clear();
      adaptiveExcludeReelIdsRef.current = [];
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
      updateSessionReels(communityRows);
      setPage(1);
      setTotal(communityRows.length);
      setCanRequestMore(false);
      setFeedPagesExhausted(true);
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
      setVisibleFeedError(null);
      setMobileDetailsOpen(false);
      setBootstrappingFirstReels(false);
      setGeneratingMore(false);
      setKnowledgeLevel(null);
      bootstrapAttemptedRef.current = false;
      pendingAutoplayAdvanceRef.current = false;
      setPendingTailAdvance(false);
      setRecoveryPhase("idle");
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
    knowledgeLevelByMaterialRef.current.clear();
    adaptiveExcludeReelIdsRef.current = [];
    setSessionHydrated(false);
    setKnowledgeLevel(null);
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
    pendingAutoplayAdvanceRef.current = false;
    recoveryAttemptedIdsRef.current.clear();
    setLoading(!restoredSession || restoredSession.reels.length === 0);
    updateSessionReels([]);
    setPage(1);
    setTotal(0);
    setCanRequestMore(true);
    setFeedPagesExhausted(false);
    setActiveIndex(0);
    activeIndexRef.current = 0;
    watchedFrontierIndexRef.current = 0;
    setFeedbackByReel({});
    setMobileDetailsOpen(false);
    setBootstrappingFirstReels(false);
    bootstrapAttemptedRef.current = false;
    let restoredGenerationMode: GenerationMode =
      generationModeParam === "fast" || generationModeParam === "slow" ? generationModeParam : "slow";
    if (restoredSession) {
      const allowedMaterialIds = new Set(feedMaterialIds.map((id) => String(id || "").trim()).filter(Boolean));
      const singleFeedMaterialId = feedMaterialIds.length === 1 ? feedMaterialIds[0] : "";
      const restoredReels = mergeSessionReels(
        restoredSession.reels.filter((reel) => {
          const reelMaterialId = String(reel.material_id || singleFeedMaterialId).trim();
          return !reelMaterialId || allowedMaterialIds.has(reelMaterialId);
        }),
      ).reels.slice(-MAX_REELS_PER_FEED_SESSION);
      const restoredIndex = restoredReels.length > 0 ? clamp(restoredSession.activeIndex, 0, restoredReels.length - 1) : 0;
      const restoredReelId = restoredReels[restoredIndex]?.reel_id;
      const modeFromQuery =
        generationModeParam === "fast" || generationModeParam === "slow" ? generationModeParam : null;
      restoredGenerationMode = modeFromQuery ?? restoredSession.generationMode;
      updateSessionReels(restoredReels);
      setActiveIndex(restoredIndex);
      activeIndexRef.current = restoredIndex;
      watchedFrontierIndexRef.current = restoredReels.length > 0
        ? clamp(
            Math.max(restoredIndex, restoredSession.watchedFrontierIndex ?? restoredIndex),
            restoredIndex,
            restoredReels.length - 1,
          )
        : 0;
      setFeedbackByReel(restoredSession.feedbackByReel);
      adaptiveExcludeReelIdsRef.current = restoredSession.adaptiveExcludeReelIds;
      setPage(restoredSession.page);
      setTotal(Math.max(restoredSession.total, restoredReels.length));
      setCanRequestMore(true);
      setGenerationMode(restoredGenerationMode);
      setMutedPreference(restoredSession.mutedPreference);
      setAutoplayEnabled(restoredSession.autoplayEnabled);
      setPlaybackRate(normalizeFeedPlaybackRate(restoredSession.playbackRate));
      mutedRestoredFromSnapshotRef.current = true;
      autoplayRestoredFromSnapshotRef.current = true;
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
    // Ingest materials: never fall through to /api/feed — the primed snapshot IS the
    // whole feed. If hydration somehow produced no reels, show the empty state; we
    // can't recover from an ingest-search material through the legacy API.
    const currentIsIngest =
      materialId.startsWith("ingest-search:") || materialId === "ingest-scratch";
    if (currentIsIngest) {
      setCanRequestMore(false);
      setFeedPagesExhausted(true);
      if (!restoredSession || restoredSession.reels.length === 0) {
        setLoading(false);
        setSessionHydrated(true);
      }
      return;
    }
    const hasRestoredReels = Boolean(restoredSession?.reels.length);
    if (hasRestoredReels) {
      // A local snapshot makes playback instant, but it is not authoritative:
      // the durable job may have streamed or finalized more clips while this
      // page was away. Reconcile page one before deciding whether to generate.
      setBootstrappingFirstReels(true);
      const restoredSearchScope = activeSearchScopeRef.current;
      void loadPage(1, {
        autofill: true,
        preserveSession: true,
        generationMode: restoredGenerationMode,
      }).finally(() => {
        if (
          hydratedMaterialIdRef.current === materialId
          && isSearchScopeActive(restoredSearchScope)
        ) {
          setBootstrappingFirstReels(false);
        }
      });
    } else {
      void loadPage(1, { autofill: true });
    }
  }, [communityHandoffIdParam, communityPreviewReel, generationModeParam, isSearchScopeActive, loadPage, materialId, mergeSessionReels, settingsScopeReady, setVisibleFeedError, updateSessionReels]);

  useEffect(() => {
    if (!materialId || sessionHydrated || loading) {
      return;
    }
    setSessionHydrated(true);
  }, [loading, materialId, sessionHydrated]);

  useEffect(() => {
    assessmentProgressMaxRef.current.clear();
    reportedForwardScrollKeysRef.current.clear();
    assessmentStartRequestRef.current = null;
    setAssessmentSession(null);
    setAssessmentQuestionIndex(0);
    setAssessmentAnswerReveal(null);
    setAssessmentAnswering(false);
    setAssessmentResultsVisible(false);
    setAssessmentPreparingFeed(false);
    setAssessmentSnoozing(false);
    setAssessmentError(null);
    setAssessmentGatePending(false);
    setAssessmentAdvanceAfterClose(false);
    setAssessmentBootstrapPending(Boolean(materialId));
  }, [feedRouteKey, materialId]);

  useEffect(() => {
    if (!materialId) {
      setAssessmentBootstrapPending(false);
      return;
    }
    if (!settingsScopeReady || !sessionHydrated) {
      return;
    }
    const searchScope = activeSearchScopeRef.current;
    let cancelled = false;
    setAssessmentBootstrapPending(true);

    void (async () => {
      try {
        const materialIds = getFeedMaterialIds();
        for (const pendingMaterialId of materialIds) {
          const response = await fetchPendingAssessment({
            materialId: pendingMaterialId,
            signal: searchScope.controller.signal,
          });
          if (cancelled || !isSearchScopeActive(searchScope)) {
            return;
          }
          let assessmentResponse = response;
          if (!assessmentResponse.session && assessmentResponse.assessment_ready) {
            assessmentResponse = await startNextAssessment({
              materialId: pendingMaterialId,
              signal: searchScope.controller.signal,
            });
            if (cancelled || !isSearchScopeActive(searchScope)) {
              return;
            }
          }
          const pendingSession = assessmentResponse.session;
          if (
            pendingSession
            && pendingSession.questions.length > 0
            && pendingSession.answered_count < pendingSession.question_count
          ) {
            const nextSession = withAssessmentAccuracy(pendingSession, assessmentResponse);
            setAssessmentSession(nextSession);
            setAssessmentQuestionIndex(clamp(nextSession.current_index, 0, nextSession.questions.length - 1));
            setAssessmentAnswerReveal(null);
            setAssessmentResultsVisible(false);
            setAssessmentAdvanceAfterClose(false);
            break;
          }
        }
      } catch (error) {
        if (!cancelled && isSearchScopeActive(searchScope) && !isRequestInterruptedError(error)) {
          console.warn("Could not resume pending recall check:", error);
        }
      } finally {
        if (!cancelled && isSearchScopeActive(searchScope)) {
          setAssessmentBootstrapPending(false);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [getFeedMaterialIds, isSearchScopeActive, materialId, sessionHydrated, settingsScopeReady]);

  const refreshFeedInventory = useCallback(async () => {
    const feedMaterialIds = getFeedMaterialIds();
    if (!settingsScopeReady || feedMaterialIds.length === 0 || isRefreshingFeedRef.current) {
      return;
    }
    if (adaptiveExcludeReelIdsRef.current.length > 0) {
      return;
    }
    const searchScope = activeSearchScopeRef.current;
    isRefreshingFeedRef.current = true;
    setRecoveryPhase("fetching-page");
    armActiveRecoveryRequest(searchScope, "fetching-page");
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
                creativeCommonsOnly: tuning.creativeCommonsOnly,
                preferredVideoDuration: tuning.preferredVideoDuration,
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
              console.warn(`Feed refresh failed for topic material ${id}:`, error);
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
        markRecoveryProgress(0);
        setRecoveryPhase("idle");
        return;
      }
      const refreshedReels = dedupeByIdentity(mergeReelBatchesByDifficulty(successful.map((row) => row.data.reels)));
      const refreshedTotal = successful.reduce((sum, row) => sum + Math.max(0, Number(row.data.total) || 0), 0);
      const currentReels = reelsRef.current;
      const currentReelId = currentReels[activeIndexRef.current]?.reel_id;
      const mergedReels = mergeSessionReels(refreshedReels, currentReels).reels;

      updateSessionReels(mergedReels);
      setTotal((prevTotal) => Math.max(prevTotal, refreshedTotal, mergedReels.length));
      setPage((prevPage) => Math.max(prevPage, Math.max(1, Math.ceil(mergedReels.length / PAGE_SIZE))));
      clearRecoveredTransportError();
      if (mergedReels.length > currentReels.length) {
        markRecoveryProgress(mergedReels.length - currentReels.length);
      }
      if (mergedReels.length === 0) {
        setActiveIndex(0);
      } else if (currentReelId) {
        const nextIndex = mergedReels.findIndex((reel) => reel.reel_id === currentReelId);
        setActiveIndex(nextIndex >= 0 ? nextIndex : Math.min(activeIndexRef.current, mergedReels.length - 1));
      } else {
        setActiveIndex(Math.min(activeIndexRef.current, mergedReels.length - 1));
      }
      setRecoveryPhase("idle");
    } catch (error) {
      if (!isSearchScopeActive(searchScope) || isRequestInterruptedError(error)) {
        return;
      }
      markRecoveryProgress(0);
      if (isTransportError(error)) {
        noteFeedTransportFailure(error);
      }
      setRecoveryPhase("idle");
    } finally {
      if (isSearchScopeActive(searchScope)) {
        finishActiveRecoveryRequest(searchScope);
        isRefreshingFeedRef.current = false;
      }
    }
  }, [
    armActiveRecoveryRequest,
    clearRecoveredTransportError,
    dedupeByIdentity,
    generationMode,
    getFeedMaterialIds,
    getFeedTuningSettings,
    mergeReelBatchesByDifficulty,
    isSearchScopeActive,
    markRecoveryProgress,
    mergeSessionReels,
    noteFeedTransportFailure,
    settingsScopeReady,
    updateSessionReels,
    finishActiveRecoveryRequest,
  ]);

  const runFastTopUp = useCallback(async () => {
    const feedMaterialIds = getFeedMaterialIds().filter((id) => !isGenerationCoolingDown(id));
    if (
      !settingsScopeReady
      || feedMaterialIds.length === 0
      || generationMode !== "fast"
      || !canRequestMore
      || isFastTopUpRef.current
      || isGeneratingRef.current
    ) {
      return;
    }
    const searchScope = activeSearchScopeRef.current;
    const tuning = getFeedTuningSettings();
    const unseenReadyCount = Math.max(0, reelsRef.current.length - activeIndexRef.current - 1);
    const perTopicBatch = Math.max(
      1,
      Math.ceil(Math.max(1, readyReservoirTarget("fast") - unseenReadyCount) / feedMaterialIds.length),
    );
    isFastTopUpRef.current = true;
    const generationBatchToken = Symbol("fast-top-up");
    generationBatchTokensRef.current.add(generationBatchToken);
    syncGenerationLockState();
    try {
      const generatedRows = await Promise.all(
        feedMaterialIds.map(async (id) => {
          const streamedReels: Reel[] = [];
          const currentCount = countReelsForMaterial(id);
          const targetTotal = Math.min(readyReservoirTarget("fast"), currentCount + perTopicBatch);
          const activeGenerationJob = generationJobByMaterialRef.current.get(id);
          const consumerToken = claimGenerationConsumer(id, activeGenerationJob?.jobId || "new-fast-generation");
          if (!consumerToken) {
            return null;
          }
          try {
            const data = await generateReelsStream({
              materialId: id,
              numReels: targetTotal,
              generationJobId: activeGenerationJob?.jobId,
              generationMode,
              minRelevance: tuning.minRelevance,
              creativeCommonsOnly: tuning.creativeCommonsOnly,
              preferredVideoDuration: tuning.preferredVideoDuration,
              signal: searchScope.controller.signal,
              idleTimeoutMs: GENERATION_STREAM_IDLE_TIMEOUT_MS,
              onTerminal: (status) => {
                if (isSearchScopeActive(searchScope)) {
                  noteGenerationTerminal(id, status);
                }
              },
              onCandidate: (reel) => {
                if (!isSearchScopeActive(searchScope)) {
                  return;
                }
                const appended = appendGeneratedReels([reel]);
                if (appended.addedCount > 0) {
                  streamedReels.push(...appended.addedReels);
                }
              },
            });
            if (!isSearchScopeActive(searchScope)) {
              return null;
            }
            reconcileGeneratedReels(streamedReels, data.reels, { preserveUnmatchedUnseen: true });
            return { materialId: id, data, streamedReels: dedupeByIdentity(data.reels) };
          } catch (e) {
            if (!isRequestInterruptedError(e)) {
              console.warn(`Fast mode background top-up failed for topic material ${id}:`, e);
            }
            return null;
          } finally {
            if (isSearchScopeActive(searchScope)) {
              releaseGenerationConsumer(id, consumerToken);
            }
          }
        }),
      );
      if (!isSearchScopeActive(searchScope)) {
        return;
      }
      const generated = dedupeByIdentity(
        mergeReelBatchesByDifficulty(generatedRows.map((row) => row?.streamedReels ?? [])),
      );
      if (generated.length > 0) {
        markRecoveryProgress(generated.length);
        setFeedPagesExhausted(false);
      }
    } catch (e) {
      if (!isSearchScopeActive(searchScope) || isRequestInterruptedError(e)) {
        return;
      }
      console.warn("Fast mode background top-up failed:", e);
    } finally {
      if (isSearchScopeActive(searchScope)) {
        isFastTopUpRef.current = false;
        generationBatchTokensRef.current.delete(generationBatchToken);
        syncGenerationLockState();
      }
    }
  }, [
    appendGeneratedReels,
    canRequestMore,
    claimGenerationConsumer,
    countReelsForMaterial,
    dedupeByIdentity,
    generationMode,
    getFeedMaterialIds,
    getFeedTuningSettings,
    mergeReelBatchesByDifficulty,
    isGenerationCoolingDown,
    isSearchScopeActive,
    markRecoveryProgress,
    noteGenerationTerminal,
    reconcileGeneratedReels,
    releaseGenerationConsumer,
    settingsScopeReady,
    syncGenerationLockState,
  ]);

  const bootstrapFirstReels = useCallback(
    async (manual = false) => {
      if (!materialId || isGeneratingRef.current || !canRequestMore) {
        return;
      }
      // Ingest materials: feed is pre-populated via the primed session snapshot.
      // Do not trigger any legacy generate calls — they cannot service ingest
      // sentinel materials and return "material_id not found".
      if (isIngestMaterial) {
        setBootstrappingFirstReels(false);
        setCanRequestMore(false);
        setFeedPagesExhausted(true);
        return;
      }
      const searchScope = activeSearchScopeRef.current;
      const reservoirTarget = readyReservoirTarget(generationMode);
      setBootstrappingFirstReels(true);
      try {
        // A restored/current session can already contain rows from later
        // durable pages. Walk every still-available persisted page before
        // asking the backend to generate a larger inventory; a duplicate-only
        // page still advances us toward the next authoritative page.
        let nextPersistedPage = page + 1;
        let persistedPagesExhausted = !hasMore;
        const lastPersistedPage = Math.max(page, Math.ceil(total / PAGE_SIZE));
        while (
          !persistedPagesExhausted
          && nextPersistedPage <= lastPersistedPage
          && Math.max(0, reelsRef.current.length - activeIndexRef.current - 1) < reservoirTarget
        ) {
          const persisted = await loadPage(nextPersistedPage, {
            autofill: false,
            generationMode,
          });
          if (!isSearchScopeActive(searchScope)) {
            return;
          }
          persistedPagesExhausted = persisted.exhausted;
          nextPersistedPage += 1;
        }
        if (Math.max(0, reelsRef.current.length - activeIndexRef.current - 1) >= reservoirTarget) {
          return;
        }
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
    [canRequestMore, generationMode, hasMore, isIngestMaterial, isSearchScopeActive, loadPage, materialId, page, requestMore, runFastTopUp, total],
  );

  useEffect(() => {
    if (
      !materialId
      || loading
      || bootstrappingFirstReels
      || bootstrapAttemptedRef.current
      || !feedNeedsBootstrapTopUp()
    ) {
      return;
    }
    // Ingest materials never enter the legacy bootstrap loop — primed snapshot only.
    if (isIngestMaterial) {
      bootstrapAttemptedRef.current = true;
      return;
    }
    bootstrapAttemptedRef.current = true;
    void bootstrapFirstReels(false);
  }, [bootstrapFirstReels, bootstrappingFirstReels, feedNeedsBootstrapTopUp, isIngestMaterial, loading, materialId]);

  const maybeLoadMore = useCallback(() => {
    // For ingest-only sentinels, the primed session snapshot is the whole feed;
    // no durable generation or paginated feed request is valid.
    if (isIngestMaterial) {
      return;
    }
    if (hasMore && !isFetchingRef.current) {
      loadPage(page + 1, { autofill: true });
      return;
    }
    if (canRequestMore && !isGeneratingRef.current && feedNeedsBootstrapTopUp()) {
      void (async () => {
        const generated = await requestMore();
        if (generationMode === "fast" && generated.length > 0) {
          void runFastTopUp();
        }
      })();
    }
  }, [canRequestMore, feedNeedsBootstrapTopUp, generationMode, hasMore, isIngestMaterial, loadPage, page, requestMore, runFastTopUp]);

  const shouldBlockDownwardAtEnd = useCallback(
    (direction: 1 | -1): boolean => {
      if (direction <= 0 || reels.length === 0) {
        return false;
      }
      if (activeIndexRef.current < reels.length - 1) {
        return false;
      }
      pendingAutoplayAdvanceRef.current = true;
      setPendingTailAdvance(true);
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
  }, [canRequestMore, generationMode, hasMore, loadPage, materialId, page, reels, requestMore, runFastTopUp]);

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
    watchedFrontierIndexRef.current = Math.max(watchedFrontierIndexRef.current, activeIndex);
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
    abortActiveChat();
    if (mobileDetailsCloseTimerRef.current) {
      clearTimeout(mobileDetailsCloseTimerRef.current);
      mobileDetailsCloseTimerRef.current = null;
    }
    setMobileDetailsClosing(false);
    setMobileDetailsOpen(false);
    setChatInput("");
    setChatLoading(false);
    setChatError(null);
  }, [abortActiveChat, activeIndex]);

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
    abortActiveChat();
    setChatLoading(false);
    setMobileDetailsClosing(true);
    if (mobileDetailsCloseTimerRef.current) {
      clearTimeout(mobileDetailsCloseTimerRef.current);
    }
    mobileDetailsCloseTimerRef.current = setTimeout(() => {
      setMobileDetailsOpen(false);
      setMobileDetailsClosing(false);
      mobileDetailsCloseTimerRef.current = null;
    }, MOBILE_DETAILS_CLOSE_MS);
  }, [abortActiveChat, mobileDetailsClosing, mobileDetailsOpen]);

  const closeActiveChat = useCallback(() => {
    abortActiveChat();
    setChatLoading(false);
    setChatError(null);
    setChatPanelOpen(false);
  }, [abortActiveChat]);

  useEffect(() => {
    if (typeof window === "undefined" || !materialId || !sessionHydrated) {
      return;
    }
    const allowedMaterialIds = new Set(getFeedMaterialIds().map((id) => String(id || "").trim()).filter(Boolean));
    const singleFeedMaterialId = allowedMaterialIds.size === 1 ? Array.from(allowedMaterialIds)[0] : "";
    const dedupedReels = mergeSessionReels(
      reels.filter((reel) => {
        const reelMaterialId = String(reel.material_id || singleFeedMaterialId).trim();
        return !reelMaterialId || allowedMaterialIds.has(reelMaterialId);
      }),
    ).reels.slice(-MAX_REELS_PER_FEED_SESSION);
    const index = dedupedReels.length > 0 ? clamp(activeIndex, 0, dedupedReels.length - 1) : 0;
    const activeReelId = dedupedReels[index]?.reel_id;
    persistFeedSessionSnapshot(materialId, {
      reels: dedupedReels,
      feedbackByReel,
      adaptiveExcludeReelIds: adaptiveExcludeReelIdsRef.current,
      page: Math.max(1, Math.floor(page || 1)),
      total: Math.max(total, dedupedReels.length),
      canRequestMore,
      generationMode,
      mutedPreference,
      autoplayEnabled,
      playbackRate,
      activeIndex: index,
      watchedFrontierIndex: dedupedReels.length > 0
        ? clamp(
            Math.max(index, watchedFrontierIndexRef.current),
            index,
            dedupedReels.length - 1,
          )
        : 0,
      activeReelId,
      updatedAt: Date.now(),
    });
  }, [
    activeIndex,
    autoplayEnabled,
    canRequestMore,
    feedbackByReel,
    generationMode,
    mergeSessionReels,
    materialId,
    mutedPreference,
    page,
    playbackRate,
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

  const commitOneReelMove = useCallback(
    (direction: 1 | -1) => {
      pendingAutoplayAdvanceRef.current = false;
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
      activeIndexRef.current = next;
      setActiveIndex(next);
      if (reels.length - next - 1 <= READY_RESERVOIR_REFILL_THRESHOLD) {
        maybeLoadMore();
      }
    },
    [beginSnapTransitionLock, maybeLoadMore, reels.length],
  );

  const jumpOneReel = useCallback(
    (direction: 1 | -1) => {
      if (
        assessmentStartRequestRef.current
        || assessmentBootstrapPending
        || assessmentGatePending
        || assessmentSession
      ) {
        return;
      }
      if (direction < 0) {
        commitOneReelMove(direction);
        return;
      }

      const currentReels = reelsRef.current;
      const currentIndex = activeIndexRef.current;
      const nextIndex = Math.min(currentReels.length - 1, currentIndex + 1);
      if (currentReels.length === 0 || nextIndex <= currentIndex) {
        maybeLoadMore();
        return;
      }

      const outgoingReel = currentReels[currentIndex];
      const reelId = String(outgoingReel?.reel_id || "").trim();
      const reelMaterialId = String(outgoingReel?.material_id || materialId || "").trim();
      commitOneReelMove(1);
      if (!reelId || !reelMaterialId || reelId.startsWith("community:")) {
        return;
      }

      const searchScope = activeSearchScopeRef.current;
      const reportKey = `${searchScope.key}:${searchScope.seq}:${reelId}`;
      if (reportedForwardScrollKeysRef.current.has(reportKey)) {
        return;
      }
      reportedForwardScrollKeysRef.current.add(reportKey);

      void (async () => {
        try {
          const scroll = await reportReelScroll({
            reelId,
            signal: searchScope.controller.signal,
          });
          if (!isSearchScopeActive(searchScope) || !scroll.assessment_ready || assessmentStartRequestRef.current) {
            return;
          }

          const assessmentRequest = { key: searchScope.key, seq: searchScope.seq, reelId };
          assessmentStartRequestRef.current = assessmentRequest;
          setAssessmentGatePending(true);
          setAssessmentError(null);
          let openedAssessment = false;
          try {
            const response = await startNextAssessment({
              materialId: String(scroll.material_id || reelMaterialId).trim(),
              signal: searchScope.controller.signal,
            });
            if (!isSearchScopeActive(searchScope)) {
              return;
            }
            if (!response.session || response.session.questions.length === 0) {
              return;
            }

            const nextSession = withAssessmentAccuracy(response.session, response);
            openedAssessment = true;
            setAssessmentSession(nextSession);
            setAssessmentQuestionIndex(clamp(nextSession.current_index, 0, nextSession.questions.length - 1));
            setAssessmentAnswerReveal(null);
            setAssessmentResultsVisible(nextSession.answered_count >= nextSession.question_count);
            setAssessmentAdvanceAfterClose(false);
            setAssessmentGatePending(false);
            setAssessmentError(null);
          } catch (error) {
            if (isSearchScopeActive(searchScope) && !isRequestInterruptedError(error)) {
              console.warn("Could not start recall check:", error);
            }
          } finally {
            if (!openedAssessment && assessmentStartRequestRef.current === assessmentRequest) {
              assessmentStartRequestRef.current = null;
              if (isSearchScopeActive(searchScope)) {
                setAssessmentGatePending(false);
              }
            }
          }
        } catch (error) {
          reportedForwardScrollKeysRef.current.delete(reportKey);
          if (isSearchScopeActive(searchScope) && !isRequestInterruptedError(error)) {
            console.warn(`Could not record forward scroll for ${reelId}:`, error);
          }
        }
      })();
    },
    [
      assessmentBootstrapPending,
      assessmentGatePending,
      assessmentSession,
      commitOneReelMove,
      isSearchScopeActive,
      materialId,
      maybeLoadMore,
    ],
  );

  const requestAutoplayAdvance = useCallback(() => {
    if (reels.length === 0) {
      return;
    }
    if (activeIndexRef.current < reels.length - 1) {
      jumpOneReel(1);
      return;
    }
    pendingAutoplayAdvanceRef.current = true;
    setPendingTailAdvance(true);
    maybeLoadMore();
  }, [jumpOneReel, maybeLoadMore, reels.length]);

  useEffect(() => {
    if (!pendingAutoplayAdvanceRef.current) {
      return;
    }
    if (activeIndex >= reels.length - 1) {
      if (!hasMore && !canRequestMore) {
        pendingAutoplayAdvanceRef.current = false;
        setPendingTailAdvance(false);
      }
      return;
    }
    pendingAutoplayAdvanceRef.current = false;
    setPendingTailAdvance(false);
    jumpOneReel(1);
  }, [activeIndex, canRequestMore, hasMore, jumpOneReel, reels.length]);

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

      if (wheelGestureReleaseTimerRef.current) {
        clearTimeout(wheelGestureReleaseTimerRef.current);
      }
      wheelGestureReleaseTimerRef.current = setTimeout(() => {
        wheelAccumRef.current = 0;
        wheelReadyToRearmRef.current = wheelGestureConsumedRef.current;
        wheelGestureReleaseTimerRef.current = null;
      }, WHEEL_GESTURE_RELEASE_MS);

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

      if (wheelGestureConsumedRef.current) {
        if (!wheelReadyToRearmRef.current) {
          return;
        }
        wheelGestureConsumedRef.current = false;
        wheelReadyToRearmRef.current = false;
        wheelAccumRef.current = 0;
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
  const assessmentPlaybackBlocked = assessmentBootstrapPending || assessmentGatePending || Boolean(assessmentSession);
  const [descriptionExpanded, setDescriptionExpanded] = useState(false);
  const [descriptionHydrating, setDescriptionHydrating] = useState(false);
  const atEndOfVisibleReels = reels.length > 0 && activeIndex >= reels.length - 1;
  const activeRecoveryRequest = recoveryPhase === "fetching-page" || recoveryPhase === "generating";
  const noMoreReelsAvailable =
    reels.length > 0 &&
    !hasMore &&
    !canRequestMore &&
    !activeRecoveryRequest &&
    !loading;
  const activelyFindingMoreReels = atEndOfVisibleReels && !noMoreReelsAvailable && activeRecoveryRequest;
  const shouldKeepFeedEntryLoading = loading || (reels.length === 0 && bootstrappingFirstReels);
  const showLoadingScreen = useLoadingScreenGate(initialFeedScreenReady, {
    minimumVisibleMs: 250,
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
        await refreshFeedInventory();
      } finally {
        setDescriptionHydrating(false);
      }
    }
    setDescriptionExpanded((prev) => !prev);
  }, [activeVideoDescription.compacted, descriptionExpanded, refreshFeedInventory]);

  const activeReelDetails = useMemo(
    () => (activeReel ? buildReelDetailContent(activeReel) : { summary: "", takeaways: [], reason: "" }),
    [activeReel],
  );

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

    abortActiveChat();
    const controller = new AbortController();
    chatAbortControllerRef.current = controller;
    const reelId = activeReel.reel_id;

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
        activeReel.match_reason,
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
        signal: controller.signal,
      });
      if (controller.signal.aborted || chatAbortControllerRef.current !== controller) {
        return;
      }
      setChatByReel((prev) => {
        const current = prev[reelId] ?? nextHistory;
        return {
          ...prev,
          [reelId]: [...current, { role: "assistant", content: result.answer }],
        };
      });
    } catch (e) {
      if (controller.signal.aborted || isRequestInterruptedError(e) || chatAbortControllerRef.current !== controller) {
        return;
      }
      setChatError(e instanceof Error ? e.message : "Chat failed");
      setChatByReel((prev) => {
        const current = prev[reelId] ?? nextHistory;
        return {
          ...prev,
          [reelId]: [...current, { role: "assistant", content: "I could not reply right now. Try again." }],
        };
      });
    } finally {
      if (chatAbortControllerRef.current === controller) {
        chatAbortControllerRef.current = null;
        setChatLoading(false);
      }
    }
  }, [abortActiveChat, activeReel, chatByReel, chatInput, chatLoading]);

  const rerankUnseenTail = useCallback(async () => {
    const feedMaterialIds = getFeedMaterialIds();
    const currentReels = reelsRef.current;
    if (!settingsScopeReady || feedMaterialIds.length === 0 || currentReels.length === 0) {
      return;
    }
    const searchScope = activeSearchScopeRef.current;

    const currentIndex = clamp(activeIndexRef.current, 0, currentReels.length - 1);
    const preservedThroughIndex = clamp(
      Math.max(currentIndex, watchedFrontierIndexRef.current),
      currentIndex,
      currentReels.length - 1,
    );
    const currentActiveReelId = currentReels[currentIndex]?.reel_id;
    const watchedPrefix = currentReels.slice(0, preservedThroughIndex + 1);
    const excludeReelIds = watchedPrefix.map((reel) => String(reel.reel_id || "").trim()).filter(Boolean);
    const tailCapacity = Math.max(0, MAX_REELS_PER_FEED_SESSION - watchedPrefix.length);
    if (tailCapacity === 0) {
      adaptiveExcludeReelIdsRef.current = excludeReelIds.slice(-200);
      return;
    }
    const unseenCount = Math.max(0, currentReels.length - watchedPrefix.length);
    const pagesToFetch = Math.max(1, Math.ceil(Math.min(tailCapacity, Math.max(25, unseenCount)) / 25));
    const tuning = getFeedTuningSettings();
    const responses = await Promise.all(
      feedMaterialIds.map(async (id) => {
        try {
          const pages: Array<Awaited<ReturnType<typeof fetchFeed>>> = [];
          for (let pageNumber = 1; pageNumber <= pagesToFetch; pageNumber += 1) {
            if (!isSearchScopeActive(searchScope)) {
              return null;
            }
            pages.push(await fetchFeed({
              materialId: id,
              page: pageNumber,
              limit: 25,
              excludeReelIds,
              autofill: false,
              prefetch: 0,
              generationMode,
              minRelevance: tuning.minRelevance,
              creativeCommonsOnly: tuning.creativeCommonsOnly,
              preferredVideoDuration: tuning.preferredVideoDuration,
              signal: searchScope.controller.signal,
            }));
          }
          return pages;
        } catch {
          return null;
        }
      }),
    );
    if (!isSearchScopeActive(searchScope) || responses.some((response) => response === null)) {
      return;
    }
    const successful = responses as Array<Array<Awaited<ReturnType<typeof fetchFeed>>>>;
    const nextLevelsByMaterial = new Map<string, string>();
    for (let index = 0; index < successful.length; index += 1) {
      const nextLevel = String(successful[index]?.[0]?.knowledge_level || "").trim();
      if (nextLevel) {
        nextLevelsByMaterial.set(feedMaterialIds[index], nextLevel);
      }
    }
    const levelChanged = Array.from(nextLevelsByMaterial).some(([id, nextLevel]) => {
      const previousLevel = knowledgeLevelByMaterialRef.current.get(id);
      return previousLevel === undefined || previousLevel !== nextLevel;
    });
    for (const [id, nextLevel] of nextLevelsByMaterial) {
      knowledgeLevelByMaterialRef.current.set(id, nextLevel);
    }
    if (levelChanged) {
      // A restored session has no trustworthy level baseline, and any member of
      // a grouped feed can advance independently. Stop every old-level stream
      // before applying the newly ranked inventory.
      clearGenerationTracking();
      setAssessmentPreparingFeed(false);
      renewActiveSearchScope();
    }

    const rankedTail = dedupeByIdentity(mergeReelBatchesByDifficulty(
      successful.map((pages) => dedupeByIdentity(pages.flatMap((response) => response.reels))),
    ));
    const nextReels = dedupeByIdentity(rankedTail, watchedPrefix).slice(0, MAX_REELS_PER_FEED_SESSION);
    adaptiveExcludeReelIdsRef.current = excludeReelIds.slice(-200);
    updateSessionReels(nextReels);
    const nextActiveIndex = currentActiveReelId
      ? nextReels.findIndex((reel) => reel.reel_id === currentActiveReelId)
      : -1;
    activeIndexRef.current = nextActiveIndex >= 0 ? nextActiveIndex : currentIndex;
    setActiveIndex(activeIndexRef.current);
    setTotal(Math.max(
      nextReels.length,
      watchedPrefix.length + successful.reduce((sum, pages) => {
        const lastResponse = pages[pages.length - 1];
        return sum + Math.max(0, Number(lastResponse?.total) || 0);
      }, 0),
    ));
    setPage(pagesToFetch);
    setCanRequestMore(true);
    setFeedPagesExhausted(false);
    const primaryLevel = nextLevelsByMaterial.get(feedMaterialIds[0]);
    if (primaryLevel) {
      setKnowledgeLevel(primaryLevel);
    }
  }, [
    clearGenerationTracking,
    dedupeByIdentity,
    generationMode,
    getFeedMaterialIds,
    getFeedTuningSettings,
    mergeReelBatchesByDifficulty,
    isSearchScopeActive,
    renewActiveSearchScope,
    settingsScopeReady,
    updateSessionReels,
  ]);

  const reportActiveReelProgress = useCallback((reel: Reel, maxFraction: number, _naturalEnd: boolean) => {
    const reelId = String(reel.reel_id || "").trim();
    const progressMaterialId = String(reel.material_id || materialId || "").trim();
    if (!reelId || !progressMaterialId || reelId.startsWith("community:")) {
      return;
    }
    const normalizedFraction = clamp(maxFraction, 0, 1);
    const previousFraction = assessmentProgressMaxRef.current.get(reelId) ?? 0;
    if (normalizedFraction <= previousFraction || normalizedFraction < 0.8) {
      return;
    }
    assessmentProgressMaxRef.current.set(reelId, normalizedFraction);
    const searchScope = activeSearchScopeRef.current;
    void reportReelProgress({
      reelId,
      maxFraction: normalizedFraction,
      signal: searchScope.controller.signal,
    }).catch((error) => {
      if (isSearchScopeActive(searchScope) && !isRequestInterruptedError(error)) {
        console.warn(`Could not save reel progress for ${reelId}:`, error);
      }
    });
  }, [isSearchScopeActive, materialId]);

  const persistAssessmentRecall = useCallback((session: AssessmentSession) => {
    const questionCount = Math.max(0, session.question_count);
    const scoreAccuracy = clamp(Number(session.score) || 0, 0, 1);
    const correctCount = Math.round(scoreAccuracy * questionCount);
    const recentAccuracy = Number.isFinite(session.recent_accuracy)
      ? clamp(Number(session.recent_accuracy), 0, 1)
      : questionCount > 0
        ? scoreAccuracy
        : undefined;
    persistCurrentSearchHistoryEntry({
      recall: {
        recentScore: correctCount,
        recentQuestionCount: questionCount,
        recentAccuracy,
        rollingAccuracy: Number.isFinite(session.rolling_accuracy)
          ? clamp(Number(session.rolling_accuracy), 0, 1)
          : undefined,
        understoodConcepts: session.understood_concepts,
        revisitConcepts: session.revisit_concepts,
        completedAt: Date.now(),
      },
    });
  }, [persistCurrentSearchHistoryEntry]);

  const submitAssessmentAnswer = useCallback(async (choiceIndex: number) => {
    if (!assessmentSession || assessmentAnswering || assessmentAnswerReveal) {
      return;
    }
    const question = assessmentSession.questions[assessmentQuestionIndex];
    if (!question || question.options.length !== 4 || choiceIndex < 0 || choiceIndex >= 4) {
      setAssessmentError("This question is unavailable. Choose Later to continue learning.");
      return;
    }
    const searchScope = activeSearchScopeRef.current;
    setAssessmentAnswering(true);
    setAssessmentError(null);
    try {
      const response = await answerAssessmentQuestion({
        sessionId: assessmentSession.id,
        questionId: question.id,
        choiceIndex,
        signal: searchScope.controller.signal,
      });
      if (!isSearchScopeActive(searchScope) || response.session.id !== assessmentSession.id) {
        return;
      }
      const nextSession: AssessmentSession = {
        ...response.session,
        recent_accuracy: response.session.recent_accuracy ?? assessmentSession.recent_accuracy,
        rolling_accuracy: response.session.rolling_accuracy ?? assessmentSession.rolling_accuracy,
      };
      setAssessmentSession(nextSession);
      setAssessmentAnswerReveal({
        questionId: question.id,
        choiceIndex,
        correct: response.correct,
        correctIndex: response.correct_index,
        explanation: response.explanation,
      });
    } catch (error) {
      if (isSearchScopeActive(searchScope) && !isRequestInterruptedError(error)) {
        setAssessmentError(error instanceof Error ? error.message : "Could not save that answer.");
      }
    } finally {
      if (isSearchScopeActive(searchScope)) {
        setAssessmentAnswering(false);
      }
    }
  }, [assessmentAnswerReveal, assessmentAnswering, assessmentQuestionIndex, assessmentSession, isSearchScopeActive]);

  const showNextAssessmentQuestion = useCallback(() => {
    if (!assessmentSession || !assessmentAnswerReveal) {
      return;
    }
    if (assessmentSession.answered_count >= assessmentSession.question_count) {
      setAssessmentResultsVisible(true);
      persistAssessmentRecall(assessmentSession);
      setAssessmentPreparingFeed(true);
      const searchScope = activeSearchScopeRef.current;
      void rerankUnseenTail().finally(() => {
        if (isSearchScopeActive(searchScope)) {
          setAssessmentPreparingFeed(false);
        }
      });
      return;
    }
    const nextIndex = clamp(
      Math.max(assessmentQuestionIndex + 1, assessmentSession.current_index),
      0,
      assessmentSession.questions.length - 1,
    );
    setAssessmentQuestionIndex(nextIndex);
    setAssessmentAnswerReveal(null);
    setAssessmentError(null);
  }, [assessmentAnswerReveal, assessmentQuestionIndex, assessmentSession, isSearchScopeActive, persistAssessmentRecall, rerankUnseenTail]);

  const closeAssessmentAndContinue = useCallback(() => {
    const shouldAdvance = assessmentAdvanceAfterClose;
    assessmentStartRequestRef.current = null;
    setAssessmentSession(null);
    setAssessmentQuestionIndex(0);
    setAssessmentAnswerReveal(null);
    setAssessmentAnswering(false);
    setAssessmentResultsVisible(false);
    setAssessmentPreparingFeed(false);
    setAssessmentSnoozing(false);
    setAssessmentError(null);
    setAssessmentGatePending(false);
    setAssessmentAdvanceAfterClose(false);
    if (shouldAdvance) {
      commitOneReelMove(1);
    }
  }, [assessmentAdvanceAfterClose, commitOneReelMove]);

  const snoozeActiveAssessment = useCallback(async () => {
    if (!assessmentSession || assessmentSnoozing) {
      return;
    }
    const searchScope = activeSearchScopeRef.current;
    setAssessmentSnoozing(true);
    setAssessmentError(null);
    try {
      await snoozeAssessment({
        sessionId: assessmentSession.id,
        signal: searchScope.controller.signal,
      });
      if (isSearchScopeActive(searchScope)) {
        closeAssessmentAndContinue();
      }
    } catch (error) {
      if (isSearchScopeActive(searchScope) && !isRequestInterruptedError(error)) {
        setAssessmentError(error instanceof Error ? error.message : "Could not snooze this check.");
        setAssessmentSnoozing(false);
      }
    }
  }, [assessmentSession, assessmentSnoozing, closeAssessmentAndContinue, isSearchScopeActive]);

  const handleActivePlaybackProgress = useCallback((maxFraction: number, naturalEnd: boolean) => {
    if (activeReel) {
      reportActiveReelProgress(activeReel, maxFraction, naturalEnd);
    }
  }, [activeReel, reportActiveReelProgress]);

  const submitActiveFeedback = useCallback(
    async (action: FeedbackAction) => {
      if (!activeReel) {
        return;
      }

      const current = activeFeedback;
      let payload: ReelFeedbackState;
      if (action === "helpful") {
        const nextHelpful = !Boolean(current.helpful);
        payload = {
          helpful: nextHelpful,
          confusing: false,
          rating: nextHelpful ? 5 : 3,
          saved: Boolean(current.saved),
        };
      } else if (action === "confusing") {
        const nextConfusing = !Boolean(current.confusing);
        payload = {
          helpful: false,
          confusing: nextConfusing,
          rating: nextConfusing ? 2 : 3,
          saved: Boolean(current.saved),
        };
      } else {
        const nextSaved = !Boolean(current.saved);
        payload = {
          helpful: Boolean(current.helpful),
          confusing: Boolean(current.confusing),
          rating: current.helpful ? 5 : current.confusing ? 2 : 3,
          saved: nextSaved,
        };
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
        await rerankUnseenTail();
      } catch (e) {
        setError(e instanceof Error ? e.message : "Could not save feedback");
      } finally {
        setPendingAction(null);
      }
    },
    [activeFeedback, activeReel, rerankUnseenTail],
  );

  const renderMobileFeedbackButton = (action: FeedbackAction, label: string, iconClass: string, active: boolean) => (
    <button
      type="button"
      onClick={() => submitActiveFeedback(action)}
      className={`relative grid h-10 w-10 place-items-center overflow-hidden rounded-2xl border-[0.8px] text-sm transition-colors duration-200 disabled:pointer-events-none backdrop-blur-lg ${
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
    abortActiveChat();
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
  }, [abortActiveChat, abortActiveSearchScope, feedFallbackPath, returnTabParam, router]);

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
      {error ? (
        <div className="absolute left-0 right-0 top-3 z-[2147483647] mx-auto w-fit">
          <div className="relative overflow-hidden rounded-xl border border-gray-300/45 bg-white/10 px-4 py-2 text-xs text-white shadow-[0_12px_28px_rgba(0,0,0,0.35)] backdrop-blur-xl backdrop-saturate-150">
            <div aria-hidden="true" className="pointer-events-none absolute inset-0 bg-black/45" />
            <span className="relative">{error}</span>
          </div>
        </div>
      ) : null}

      {assessmentSession ? (
        <RecallCheck
          session={assessmentSession}
          questionIndex={assessmentQuestionIndex}
          answerReveal={assessmentAnswerReveal}
          answering={assessmentAnswering}
          showResults={assessmentResultsVisible}
          preparingFeed={assessmentPreparingFeed}
          snoozing={assessmentSnoozing}
          error={assessmentError}
          onAnswer={submitAssessmentAnswer}
          onNextQuestion={showNextAssessmentQuestion}
          onLater={snoozeActiveAssessment}
          onContinue={closeAssessmentAndContinue}
        />
      ) : null}

      {(assessmentBootstrapPending || assessmentGatePending) && !assessmentSession ? (
        <div className="fixed inset-0 z-[10000] grid place-items-center bg-black/86 px-6 text-white backdrop-blur-xl" role="status" aria-live="polite">
          <div className="rounded-full border border-white/16 bg-white/[0.05] px-5 py-3 text-xs font-semibold uppercase tracking-[0.14em] text-white/72">
            {assessmentGatePending ? "Preparing recall check..." : "Loading learning progress..."}
          </div>
        </div>
      ) : null}

      <div ref={desktopShellRef} className="h-full min-h-[100dvh] md:min-h-0 lg:flex">
        <section className="relative h-[100dvh] min-h-[100dvh] md:h-full md:min-h-0 lg:min-w-0 lg:flex-1">
          {activeReel && !mobileDetailsOpen ? (
            <div className="absolute right-3 top-1/2 z-30 flex -translate-y-1/2 flex-col gap-2">
              {renderMobileFeedbackButton("helpful", "Got it", "fa-thumbs-up", Boolean(activeFeedback.helpful))}
              {renderMobileFeedbackButton("confusing", "Need help", "fa-circle-question", Boolean(activeFeedback.confusing))}
              {renderMobileFeedbackButton("save", "Save", "fa-bookmark", Boolean(activeFeedback.saved))}
            </div>
          ) : null}
          {generationProgress !== null && reels.length > 0 ? (
            <GenerationStageStatus ready={reels.length} reconnecting={generationProgress.reconnecting} />
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
                  {Math.abs(index - activeIndex) <= 1 ? (
                    <ReelCard
                      reel={reel}
                      isActive={index === activeIndex && !assessmentPlaybackBlocked}
                      mutedPreference={mutedPreference}
                      onMutedPreferenceChange={setMutedPreference}
                      autoplayEnabled={autoplayEnabled}
                      onAutoplayEnabledChange={setAutoplayEnabled}
                      playbackRate={playbackRate}
                      onPlaybackRateChange={setPlaybackRate}
                      onRequestNextReel={index === activeIndex ? requestAutoplayAdvance : undefined}
                      onPlaybackProgress={index === activeIndex ? handleActivePlaybackProgress : undefined}
                      onOpenContent={index === activeIndex ? openMobileDetails : undefined}
                    />
                  ) : null}
                </div>
              ))}
            </div>
            {reels.length === 0 ? (
              <div className="absolute inset-0 grid place-items-center p-6">
                <div className="max-w-sm rounded-3xl border border-white/20 bg-black/68 px-5 py-4 text-center text-white backdrop-blur">
                  {loading || bootstrappingFirstReels || generatingMore ? (
                    <GenerationStageStatus
                      ready={reels.length}
                      reconnecting={generationProgress?.reconnecting ?? false}
                      variant="center"
                    />
                  ) : (
                    <>
                      <p className="text-sm font-semibold">No reels yet</p>
                      <p className="mt-2 text-xs text-white/72">
                        Try generating again, or adjust your topic/material for broader matches.
                      </p>
                      <button
                        type="button"
                        onClick={() => void bootstrapFirstReels(true)}
                        className="mt-3 rounded-xl border border-white/25 bg-white px-3.5 py-2 text-xs font-semibold text-black"
                      >
                        Generate Reels
                      </button>
                    </>
                  )}
                </div>
              </div>
            ) : null}
            {reels.length > 0 && (activelyFindingMoreReels || pendingTailAdvance) ? (
              <div className="pointer-events-none absolute inset-x-0 bottom-4 z-20 flex justify-center px-4">
                <div className="max-w-xs rounded-2xl border border-white/20 bg-black/80 px-5 py-3 text-center text-white shadow-[0_12px_30px_rgba(0,0,0,0.35)] backdrop-blur-sm">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.12em] text-white/85">
                    {generationProgress?.reconnecting ? "Reconnecting to the same clip job..." : "Finding the next verified clip..."}
                  </p>
                  <p className="mt-1 text-[10px] text-white/60">Your next swipe will continue automatically.</p>
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
                  <p className="mb-1 text-[10px] font-semibold uppercase tracking-[0.1em] text-white/60">AI Summary</p>
                  <p className="break-words leading-snug [overflow-wrap:anywhere]">{activeReelDetails.summary}</p>
                </div>

                <div className="mt-3 min-w-0 rounded-2xl border border-white/20 bg-black/55 p-3 text-sm text-white/90">
                  <p className="mb-2 text-[10px] font-semibold uppercase tracking-[0.1em] text-white/60">Key Takeaways</p>
                  <ul className="space-y-1.5">
                    {activeReelDetails.takeaways.map((takeaway) => (
                      <li key={`mobile-takeaway-${activeReel.reel_id}-${takeaway}`} className="flex items-start gap-2">
                        <span aria-hidden="true" className="mt-[0.45rem] h-1 w-1 shrink-0 rounded-full bg-white/60" />
                        <span className="break-words [overflow-wrap:anywhere]">{takeaway}</span>
                      </li>
                    ))}
                  </ul>
                </div>

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
                  <p className="break-words [overflow-wrap:anywhere]">{activeReelDetails.reason}</p>
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

                {(activeReel.captions && activeReel.captions.length > 0) || (activeReel.transcript_snippet && activeReel.transcript_snippet.trim()) ? (
                  <div className="mt-3 min-w-0 rounded-2xl border border-white/20 bg-black/55 p-3 text-sm text-white/90">
                    <div className="mb-1 flex items-center justify-between gap-2">
                      <p className="text-[10px] font-semibold uppercase tracking-[0.1em] text-white/60">Transcript</p>
                      {activeReel.captions && activeReel.captions.length > 0 ? (
                        <span className="text-[10px] font-semibold uppercase tracking-[0.08em] text-white/60">
                          {activeReel.captions.length} cues
                        </span>
                      ) : null}
                    </div>
                    {activeReel.captions && activeReel.captions.length > 0 ? (
                      <div className="max-h-64 overflow-y-auto pr-1">
                        <ul className="flex flex-col gap-1.5">
                          {activeReel.captions.map((cue, idx) => (
                            <li
                              key={`mobile-cue-${activeReel.reel_id}-${idx}`}
                              className="flex items-start gap-2"
                            >
                              <span className="w-11 shrink-0 font-mono text-[10px] font-semibold leading-snug text-white/55">
                                {formatCaptionTimestamp(cue.start)}
                              </span>
                              <span className="flex-1 break-words leading-snug [overflow-wrap:anywhere]">
                                {cue.text}
                              </span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    ) : (
                      <p className="break-words leading-snug [overflow-wrap:anywhere]">
                        {activeReel.transcript_snippet}
                      </p>
                    )}
                    {activeReel.source_attribution ? (
                      <p className="mt-2 text-[10px] text-white/55">Source: {activeReel.source_attribution}</p>
                    ) : null}
                  </div>
                ) : null}

              </div>
            </div>
          ) : null}
        </section>

        <button
          type="button"
          aria-label="Resize panels"
          onPointerDown={onStartLeftRightResize}
          className="group hidden w-3 cursor-col-resize touch-none select-none items-center justify-center bg-transparent lg:flex"
        >
          <span className="h-20 w-px rounded-full bg-white/35 transition group-hover:bg-white/55 group-focus-visible:bg-white/55" />
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
                  <p className="mb-1 text-[10px] font-semibold uppercase tracking-[0.1em] text-white/60">AI Summary</p>
                  <p className="break-words leading-snug [overflow-wrap:anywhere]">{activeReelDetails.summary}</p>
                </div>

                <div className="mt-3 min-w-0 rounded-2xl border border-white/20 bg-black/55 p-3 text-sm text-white/90">
                  <p className="mb-2 text-[10px] font-semibold uppercase tracking-[0.1em] text-white/60">Key Takeaways</p>
                  <ul className="space-y-1.5">
                    {activeReelDetails.takeaways.map((takeaway) => (
                      <li key={`desktop-takeaway-${activeReel.reel_id}-${takeaway}`} className="flex items-start gap-2">
                        <span aria-hidden="true" className="mt-[0.45rem] h-1 w-1 shrink-0 rounded-full bg-white/60" />
                        <span className="break-words [overflow-wrap:anywhere]">{takeaway}</span>
                      </li>
                    ))}
                  </ul>
                </div>

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
                  <p className="break-words [overflow-wrap:anywhere]">{activeReelDetails.reason}</p>
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

                {(activeReel.captions && activeReel.captions.length > 0) || (activeReel.transcript_snippet && activeReel.transcript_snippet.trim()) ? (
                  <div className="mt-3 mb-0 min-w-0 rounded-2xl border border-white/20 bg-black/55 p-3 text-sm text-white/90">
                    <div className="mb-1 flex items-center justify-between gap-2">
                      <p className="text-[10px] font-semibold uppercase tracking-[0.1em] text-white/60">Transcript</p>
                      {activeReel.captions && activeReel.captions.length > 0 ? (
                        <span className="text-[10px] font-semibold uppercase tracking-[0.08em] text-white/60">
                          {activeReel.captions.length} cues
                        </span>
                      ) : null}
                    </div>
                    {activeReel.captions && activeReel.captions.length > 0 ? (
                      <div className="max-h-72 overflow-y-auto pr-1">
                        <ul className="flex flex-col gap-1.5">
                          {activeReel.captions.map((cue, idx) => (
                            <li
                              key={`desktop-cue-${activeReel.reel_id}-${idx}`}
                              className="flex items-start gap-2"
                            >
                              <span className="w-11 shrink-0 font-mono text-[10px] font-semibold leading-snug text-white/55">
                                {formatCaptionTimestamp(cue.start)}
                              </span>
                              <span className="flex-1 break-words leading-snug [overflow-wrap:anywhere]">
                                {cue.text}
                              </span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    ) : (
                      <p className="break-words leading-snug [overflow-wrap:anywhere]">
                        {activeReel.transcript_snippet}
                      </p>
                    )}
                    {activeReel.source_attribution ? (
                      <p className="mt-2 text-[10px] text-white/55">Source: {activeReel.source_attribution}</p>
                    ) : null}
                  </div>
                ) : null}
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
              {chatPanelOpen ? (
                <button
                  type="button"
                  onClick={closeActiveChat}
                  aria-label="Close AI chat"
                  className="grid h-7 w-7 place-items-center rounded-lg text-white/55 transition hover:bg-white/10 hover:text-white"
                >
                  <i className="fa-solid fa-xmark text-xs" aria-hidden="true" />
                </button>
              ) : null}
            </div>

            {!chatPanelOpen ? (
              <div className="grid min-h-0 flex-1 place-items-center">
                <button
                  type="button"
                  onClick={() => setChatPanelOpen(true)}
                  className="rounded-full border border-white/18 bg-white/[0.05] px-4 py-2 text-xs font-semibold text-white/72 transition hover:border-white/32 hover:text-white"
                >
                  Open AI Chat
                </button>
              </div>
            ) : !activeReel ? (
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
