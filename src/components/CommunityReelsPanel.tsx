"use client";

import { type ChangeEvent, type DragEvent, type FormEvent, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";

import {
  COMMUNITY_AUTH_CHANGED_EVENT,
  COMMUNITY_OWNED_SET_IDS_STORAGE_KEY,
  expireCommunityAuthSession,
  createCommunitySet,
  deleteCommunitySet,
  deleteCommunitySets,
  fetchCommunityAccount,
  fetchCommunityReelDuration,
  fetchCommunitySets,
  fetchOwnedCommunitySets,
  isSessionExpiredError,
  readCommunityAuthSession,
  saveOwnedCommunitySetIds,
  updateCommunitySet,
} from "@/lib/api";
import { safeStorageRemoveItem, safeStorageSetItem } from "@/lib/browserStorage";
import type { CommunityAccount } from "@/lib/types";
import { ViewportModalPortal } from "@/components/ViewportModalPortal";
import { FadePresence } from "@/components/FadePresence";
import {
  LOCAL_DEMO_ACCOUNT,
  LOCAL_DEMO_AVAILABLE,
  LOCAL_DEMO_COMMUNITY_SETS,
} from "@/lib/localDemo";
import { loadYouTubeIframeApi } from "@/lib/youtubeIframeApi";

const COMMUNITY_SETS_STORAGE_KEY = "studyreels-community-sets";
const COMMUNITY_CREATE_DRAFT_STORAGE_KEY = "studyreels-community-create-draft";
const COMMUNITY_EDIT_DRAFT_PREFIX = "studyreels-community-edit-draft-";
const LOCAL_DEMO_SETS_STORAGE_KEY = "studyreels-local-demo-community-sets";
const LOCAL_DEMO_STARRED_SET_IDS_STORAGE_KEY = "studyreels-local-demo-community-starred-set-ids";
const LOCAL_DEMO_CREATE_DRAFT_STORAGE_KEY = "studyreels-local-demo-community-create-draft";
const LOCAL_DEMO_EDIT_DRAFT_PREFIX = "studyreels-local-demo-community-edit-draft-";
const MAX_USER_SETS = 120;
const FALLBACK_THUMBNAIL_URL = "/images/community/ai-systems.svg";
const SUPPORTED_PLATFORMS_LABEL = "YouTube, Instagram, TikTok";
const MIN_SET_DESCRIPTION_LENGTH = 18;
const MAX_SET_TAGS = 6;
const FEATURED_CAROUSEL_INTERVAL_MS = 5200;
const FEATURED_CAROUSEL_TRANSITION_MS = 520;
const FEATURED_CAROUSEL_PAUSE_MS = 200;
const FEATURED_CAROUSEL_CONTENT_MIN_HEIGHT_FALLBACK = 250;
const FEATURED_CAROUSEL_CONTENT_MIN_HEIGHT_TOUCH_FALLBACK = 220;
const FEATURED_CAROUSEL_BUTTON_BOTTOM_MARGIN_PX = 14;
const FEATURED_CAROUSEL_IMAGE_BOTTOM_MARGIN_PX = 14;
const DIRECTORY_DETAIL_TRANSITION_MS = 440;
const COMMUNITY_SET_FEED_HANDOFF_PREFIX = "studyreels-community-feed-handoff-";
const COMMUNITY_SET_RETURN_SNAPSHOT_PREFIX = "studyreels-community-return-set-";
const COMMUNITY_STARRED_SET_IDS_STORAGE_KEY = "studyreels-community-starred-set-ids";
const COMMUNITY_CREATE_DRAFT_CONTEXT_KEY = "create";
const CLIP_SLIDER_MIN_GAP_SEC = 0.1;
const CLIP_SLIDER_STEP_SEC = 0.1;
const MAX_THUMBNAIL_FILE_BYTES = 1_500_000;
const YOUTUBE_DURATION_POLL_INTERVAL_MS = 220;
const YOUTUBE_DURATION_TIMEOUT_MS = 8_000;
const LAST_EDITED_REFRESH_INTERVAL_MS = 60_000;
const SECOND_MS = 1_000;
const MINUTE_MS = 60 * SECOND_MS;
const HOUR_MS = 60 * MINUTE_MS;
const DAY_MS = 24 * HOUR_MS;
const WEEK_MS = 7 * DAY_MS;
const MONTH_MS = 30 * DAY_MS;
const YEAR_MS = 365 * DAY_MS;
type ReelPlatform = "youtube" | "instagram" | "tiktok";

type CommunityReelEmbed = {
  id: string;
  platform: ReelPlatform;
  sourceUrl: string;
  embedUrl: string;
  tStartSec?: number;
  tEndSec?: number;
};

type DraftReelInput = {
  id: string;
  communityReelId?: string;
  value: string;
  tStartSec: string;
  tEndSec: string;
};

type ParsedDraftReel = {
  id: string;
  communityReelId?: string;
  value: string;
  tStartSec: string;
  tEndSec: string;
  clipStartSec: number | null;
  clipEndSec: number | null;
  hasClipRangeError: boolean;
  parsed: Omit<CommunityReelEmbed, "id"> | null;
};

type ReelDurationState = {
  sourceUrl: string;
  durationSec: number | null;
  loading: boolean;
};

type YouTubeDurationPlayer = {
  getDuration?: () => number;
  cueVideoById?: (videoId: string) => void;
  destroy?: () => void;
};

type YouTubeIframeApi = {
  Player: new (
    element: HTMLElement | string,
    options: {
      height?: string | number;
      width?: string | number;
      videoId?: string;
      playerVars?: Record<string, string | number>;
      events?: {
        onReady?: (event: { target: YouTubeDurationPlayer }) => void;
        onStateChange?: (event: { target: YouTubeDurationPlayer }) => void;
        onError?: () => void;
      };
    },
  ) => YouTubeDurationPlayer;
};

type YouTubeIframeWindow = Window & typeof globalThis & {
  YT?: YouTubeIframeApi;
  onYouTubeIframeAPIReady?: () => void;
};

type CommunitySet = {
  id: string;
  title: string;
  description: string;
  tags: string[];
  reels: CommunityReelEmbed[];
  reelCount: number;
  curator: string;
  likes: number;
  learners: number;
  updatedLabel: string;
  updatedAt?: string | null;
  createdAt?: string | null;
  thumbnailUrl: string;
  featured: boolean;
};

type StoredSetDraft = {
  title: string;
  description: string;
  tags: string;
  thumbnailPreview: string;
  thumbnailFileName: string;
  reelInputs: Array<{
    communityReelId?: string;
    value: string;
    tStartSec: string;
    tEndSec: string;
  }>;
};

type PublishResultModalState = {
  status: "success" | "error";
  title: string;
  message: string;
  label?: string;
  thumbnailUrl?: string;
  thumbnailAlt?: string;
};

type DeleteSetConfirmModalState = {
  setIds: string[];
  title: string;
};

type UnsavedDraftExitModalState = {
  action: "back-to-grid";
};

type DraftActionConfirmModalState = {
  action: "save-progress" | "clear-progress" | "save-set-changes";
  label: string;
  title: string;
  message: string;
  confirmLabel: string;
};

export type CommunityDraftExitActions = {
  saveDraftProgress: () => boolean;
  discardDraftChanges: () => void;
};

let draftRowCounter = 0;

function createDraftReelRow(value = "", tStartSec = "0", tEndSec = "", communityReelId?: string): DraftReelInput {
  draftRowCounter += 1;
  return {
    id: `draft-reel-${draftRowCounter}`,
    ...(typeof communityReelId === "string" && communityReelId.trim() ? { communityReelId: communityReelId.trim() } : {}),
    value,
    tStartSec,
    tEndSec,
  };
}

const PLATFORM_LABEL: Record<ReelPlatform, string> = {
  youtube: "YouTube",
  instagram: "Instagram",
  tiktok: "TikTok",
};

const PLATFORM_ICON: Record<ReelPlatform, string> = {
  youtube: "fa-brands fa-youtube",
  instagram: "fa-brands fa-instagram",
  tiktok: "fa-brands fa-tiktok",
};

function daysAgoToIso(daysAgo: number): string {
  return new Date(Date.now() - Math.max(0, daysAgo) * DAY_MS).toISOString();
}

const FEATURED_SETS: CommunitySet[] = [
  {
    id: "featured-kinematics-visuals",
    title: "Kinematics Visual Drills",
    description: "Short clips that connect equations of motion to intuitive motion sketches and worked examples.",
    tags: ["physics", "motion", "problem solving"],
    reels: [],
    reelCount: 34,
    curator: "Dr. Ramos",
    likes: 2840,
    learners: 12100,
    updatedLabel: "Last Edited: 2 days ago",
    updatedAt: daysAgoToIso(2),
    thumbnailUrl: "/images/community/physics-grid.svg",
    featured: true,
  },
  {
    id: "featured-cell-bio",
    title: "Cell Biology Core",
    description: "A sequence from membrane structure to signaling pathways with high-yield recap reels.",
    tags: ["biology", "cell", "exam prep"],
    reels: [],
    reelCount: 27,
    curator: "MedSchool Crew",
    likes: 1970,
    learners: 8700,
    updatedLabel: "Last Edited: 1 day ago",
    updatedAt: daysAgoToIso(1),
    thumbnailUrl: "/images/community/bio-lab.svg",
    featured: true,
  },
  {
    id: "featured-calc-derivatives",
    title: "Derivatives in Context",
    description: "From slope intuition to optimization and related rates with compact walkthrough reels.",
    tags: ["calculus", "derivatives", "math"],
    reels: [],
    reelCount: 31,
    curator: "Math Forge",
    likes: 2230,
    learners: 9400,
    updatedLabel: "Last Edited: 4 days ago",
    updatedAt: daysAgoToIso(4),
    thumbnailUrl: "/images/community/calculus-flow.svg",
    featured: true,
  },
];

const COMMUNITY_LIBRARY_SETS: CommunitySet[] = [
  {
    id: "community-world-history",
    title: "World History in Turning Points",
    description: "Ten key transitions from empire to modern states with source-backed clips.",
    tags: ["history", "timeline"],
    reels: [],
    reelCount: 22,
    curator: "Timeline Lab",
    likes: 960,
    learners: 4100,
    updatedLabel: "Last Edited: 6 days ago",
    updatedAt: daysAgoToIso(6),
    thumbnailUrl: "/images/community/civics-debate.svg",
    featured: false,
  },
  {
    id: "community-spanish-conversation",
    title: "Spanish Conversation Starters",
    description: "Pattern-first conversational mini reels for greetings, requests, and follow-ups.",
    tags: ["language", "spanish", "conversation"],
    reels: [],
    reelCount: 18,
    curator: "Lingua Spark",
    likes: 1180,
    learners: 5300,
    updatedLabel: "Last Edited: 3 days ago",
    updatedAt: daysAgoToIso(3),
    thumbnailUrl: "/images/community/language-story.svg",
    featured: false,
  },
  {
    id: "community-ml-foundations",
    title: "ML Foundations Fast Track",
    description: "Core probability, model intuition, and overfitting cues in under 20 reels.",
    tags: ["machine learning", "statistics"],
    reels: [],
    reelCount: 19,
    curator: "Data Guild",
    likes: 1450,
    learners: 6200,
    updatedLabel: "Last Edited: 1 week ago",
    updatedAt: daysAgoToIso(7),
    thumbnailUrl: "/images/community/ai-systems.svg",
    featured: false,
  },
];

const DEFAULT_COMMUNITY_SETS: CommunitySet[] = [...FEATURED_SETS, ...COMMUNITY_LIBRARY_SETS];

function formatCompact(value: number): string {
  if (value >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(1).replace(/\.0$/, "")}M`;
  }
  if (value >= 1_000) {
    return `${(value / 1_000).toFixed(1).replace(/\.0$/, "")}K`;
  }
  return String(value);
}

function formatCommunityPlatformSummary(reels: CommunityReelEmbed[]): string {
  const counts = reels.reduce<Record<ReelPlatform, number>>(
    (acc, reel) => {
      acc[reel.platform] += 1;
      return acc;
    },
    { youtube: 0, instagram: 0, tiktok: 0 },
  );
  return (Object.entries(counts) as Array<[ReelPlatform, number]>)
    .filter(([, count]) => count > 0)
    .map(([platform, count]) => `${count} ${PLATFORM_LABEL[platform]} ${count === 1 ? "reel" : "reels"}`)
    .join(", ");
}

function buildCommunitySetInformationParagraphs(set: CommunitySet, curatorLabel?: string): string[] {
  const availableReelCount = set.reels.length;
  const listedReelCount = Math.max(set.reelCount, availableReelCount);
  const platformSummary = formatCommunityPlatformSummary(set.reels);
  const resolvedCuratorLabel = typeof curatorLabel === "string" && curatorLabel.trim() ? curatorLabel.trim() : set.curator;
  const clippedReelCount = set.reels.filter((reel) => {
    const start = Number(reel.tStartSec ?? 0);
    const end = Number(reel.tEndSec);
    return Number.isFinite(end) && end > start;
  }).length;
  const tagSummary = set.tags.length > 0
    ? `Focus tags: ${set.tags.map((tag) => `#${tag}`).join(", ")}.`
    : "No topic tags were added to this set yet.";
  const availabilitySummary = availableReelCount > 0
    ? `This detail view currently includes ${availableReelCount} playable ${availableReelCount === 1 ? "reel" : "reels"}${platformSummary ? ` across ${platformSummary}` : ""}.`
    : listedReelCount > 0
      ? `This collection is listed as a ${listedReelCount}-reel study path, but playable reel embeds have not been attached to the detail view yet.`
      : "Playable reel embeds have not been attached to the detail view yet.";
  const clipSummary = availableReelCount === 0
    ? ""
    : clippedReelCount > 0
      ? `${clippedReelCount} ${clippedReelCount === 1 ? "reel uses" : "reels use"} explicit clip ranges for faster review.`
      : "These reels currently open on the original source without saved clip endpoints.";

  return [
    `${resolvedCuratorLabel} curated this set for ${formatCompact(set.learners)} learners, and it has collected ${formatCompact(set.likes)} likes so far.`,
    availabilitySummary,
    `${tagSummary}${clipSummary ? ` ${clipSummary}` : ""}`,
  ];
}

function matchesSetSearchQuery(set: CommunitySet, normalizedQuery: string): boolean {
  if (set.title.toLowerCase().includes(normalizedQuery)) {
    return true;
  }
  if (set.description.toLowerCase().includes(normalizedQuery)) {
    return true;
  }
  if (set.curator.toLowerCase().includes(normalizedQuery)) {
    return true;
  }
  if (set.tags.some((tag) => tag.toLowerCase().includes(normalizedQuery))) {
    return true;
  }
  return set.reels.some((reel) => PLATFORM_LABEL[reel.platform].toLowerCase().includes(normalizedQuery));
}

function parseTimestampMs(value: unknown): number | null {
  if (typeof value === "number") {
    if (Number.isFinite(value) && value > 0) {
      return value;
    }
    return null;
  }
  if (typeof value === "string") {
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
      // Treat DB timestamps without timezone as UTC to avoid local-time drift.
      candidate = `${candidate}Z`;
    }
    const normalizedParsed = Date.parse(candidate);
    return Number.isFinite(normalizedParsed) ? normalizedParsed : null;
  }
  return null;
}

function normalizeUpdatedAt(value: unknown): string | null {
  const parsedMs = parseTimestampMs(value);
  if (parsedMs == null) {
    return null;
  }
  return new Date(parsedMs).toISOString();
}

function normalizeLastEditedRaw(value: string): string {
  return value
    .trim()
    .replace(/^last\s+edited\s*:\s*/i, "")
    .replace(/^updated\s*[:\-]?\s*/i, "")
    .trim();
}

function isRelativeNowLastEditedLabel(value: string): boolean {
  const normalized = normalizeLastEditedRaw(value).toLowerCase();
  return normalized === "today" || normalized === "just now" || normalized === "less than 1 minute ago";
}

function inferUpdatedAtFromLastEditedLabel(value: string, nowMs: number): string | null {
  const normalizedRaw = normalizeLastEditedRaw(value);
  if (!normalizedRaw) {
    return null;
  }
  const normalized = normalizedRaw.toLowerCase();
  if (normalized === "today" || normalized === "just now" || normalized === "less than 1 minute ago") {
    return null;
  }
  if (normalized === "yesterday") {
    return new Date(nowMs - DAY_MS).toISOString();
  }
  const match = normalized.match(/^(\d+|an?|one)\s+(second|minute|hour|day|week|month|year)s?(?:\s+ago)?$/i);
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
      ? SECOND_MS
      : unit === "minute"
        ? MINUTE_MS
        : unit === "hour"
          ? HOUR_MS
          : unit === "day"
            ? DAY_MS
            : unit === "week"
              ? WEEK_MS
              : unit === "month"
                ? MONTH_MS
                : YEAR_MS;
  return new Date(nowMs - amount * unitMs).toISOString();
}

function normalizeLastEditedLabel(value: string): string {
  const normalizedRaw = normalizeLastEditedRaw(value);
  if (!normalizedRaw) {
    return "Last Edited: unknown";
  }
  const normalized = normalizedRaw.toLowerCase();
  if (normalized === "today") {
    return "Last Edited: today";
  }
  if (normalized === "just now") {
    return "Last Edited: just now";
  }
  if (normalized === "yesterday") {
    return "Last Edited: 1 day ago";
  }
  if (/^(\d+|an?|one)\s+(second|minute|hour|day|week|month|year)s?$/i.test(normalizedRaw)) {
    return `Last Edited: ${normalizedRaw} ago`;
  }
  if (/ago$/i.test(normalizedRaw)) {
    return `Last Edited: ${normalizedRaw}`;
  }
  return `Last Edited: ${normalizedRaw}`;
}

function formatRelativeElapsed(elapsedMs: number): string {
  const safeElapsedMs = Math.max(0, elapsedMs);
  const minutes = Math.floor(safeElapsedMs / MINUTE_MS);
  if (minutes < 1) {
    return "less than 1 minute ago";
  }
  if (minutes < 60) {
    return `${minutes} minute${minutes === 1 ? "" : "s"} ago`;
  }
  const hours = Math.floor(safeElapsedMs / HOUR_MS);
  if (hours < 24) {
    return `${hours} hour${hours === 1 ? "" : "s"} ago`;
  }
  const days = Math.floor(safeElapsedMs / DAY_MS);
  if (days < 7) {
    return `${days} day${days === 1 ? "" : "s"} ago`;
  }
  const weeks = Math.floor(safeElapsedMs / WEEK_MS);
  if (days < 30) {
    return `${weeks} week${weeks === 1 ? "" : "s"} ago`;
  }
  const months = Math.floor(safeElapsedMs / MONTH_MS);
  if (days < 365) {
    return `${months} month${months === 1 ? "" : "s"} ago`;
  }
  const years = Math.floor(safeElapsedMs / YEAR_MS);
  return `${years} year${years === 1 ? "" : "s"} ago`;
}

function formatModifiedLabel(set: Pick<CommunitySet, "updatedAt" | "updatedLabel">, nowMs: number): string {
  const updatedMs = parseTimestampMs(set.updatedAt);
  if (updatedMs != null) {
    const nowDate = new Date(nowMs);
    const updatedDate = new Date(updatedMs);
    const todayStartMs = new Date(nowDate.getFullYear(), nowDate.getMonth(), nowDate.getDate()).getTime();
    const updatedDayStartMs = new Date(updatedDate.getFullYear(), updatedDate.getMonth(), updatedDate.getDate()).getTime();
    const calendarDayDelta = Math.max(0, Math.round((todayStartMs - updatedDayStartMs) / DAY_MS));
    if (calendarDayDelta === 0) {
      return "Today";
    }
    if (calendarDayDelta === 1) {
      return "Yesterday";
    }
    if (calendarDayDelta < 7) {
      return `${calendarDayDelta} days ago`;
    }
    return updatedDate.toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
      ...(updatedDate.getFullYear() === nowDate.getFullYear() ? {} : { year: "numeric" }),
    });
  }

  const relativeLabel = normalizeLastEditedLabel(set.updatedLabel).replace(/^Last Edited:\s*/i, "");
  if (/^(today|just now|less than 1 minute ago)$/i.test(relativeLabel)) {
    return "Today";
  }
  if (/^(yesterday|1 day ago)$/i.test(relativeLabel)) {
    return "Yesterday";
  }
  return relativeLabel;
}

function parseAllTags(value: string): string[] {
  return Array.from(
    new Set(
      value
        .split(",")
        .map((part) => part.trim().toLowerCase())
        .filter(Boolean),
    ),
  );
}

function parseTags(value: string): string[] {
  return parseAllTags(value).slice(0, MAX_SET_TAGS);
}

function toAbsoluteUrl(value: string): URL | null {
  const trimmed = value.trim();
  if (!trimmed) {
    return null;
  }
  const maybeAbsolute = /^https?:\/\//i.test(trimmed) ? trimmed : `https://${trimmed}`;
  try {
    return new URL(maybeAbsolute);
  } catch {
    return null;
  }
}

function extractYouTubeVideoId(url: URL): string | null {
  const host = url.hostname.toLowerCase();
  if (host.includes("youtu.be")) {
    const id = url.pathname.split("/").filter(Boolean)[0];
    if (id) {
      return id;
    }
  }
  if (host.includes("youtube.com")) {
    if (url.pathname === "/watch") {
      const id = url.searchParams.get("v");
      if (id) {
        return id;
      }
    }
    if (url.pathname.startsWith("/shorts/")) {
      const id = url.pathname.split("/")[2];
      if (id) {
        return id;
      }
    }
    if (url.pathname.startsWith("/embed/")) {
      const id = url.pathname.split("/")[2];
      if (id) {
        return id;
      }
    }
  }
  return null;
}

function parseDetectedDuration(value: unknown): number | null {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= CLIP_SLIDER_MIN_GAP_SEC) {
    return null;
  }
  return parsed;
}

async function detectYouTubeDurationWithIframeApi(sourceUrl: string): Promise<number | null> {
  if (typeof window === "undefined" || typeof document === "undefined") {
    return null;
  }
  const parsedUrl = toAbsoluteUrl(sourceUrl);
  if (!parsedUrl) {
    return null;
  }
  const videoId = extractYouTubeVideoId(parsedUrl);
  if (!videoId) {
    return null;
  }

  try {
    await loadYouTubeIframeApi(YOUTUBE_DURATION_TIMEOUT_MS);
  } catch {
    return null;
  }

  const globalWindow = window as YouTubeIframeWindow;
  const api = globalWindow.YT;
  if (!api?.Player) {
    return null;
  }

  return await new Promise<number | null>((resolve) => {
    const host = document.createElement("div");
    host.style.position = "fixed";
    host.style.left = "-9999px";
    host.style.top = "-9999px";
    host.style.width = "1px";
    host.style.height = "1px";
    host.style.opacity = "0";
    host.setAttribute("aria-hidden", "true");
    document.body.appendChild(host);

    let settled = false;
    let player: YouTubeDurationPlayer | null = null;
    let pollTimer: ReturnType<typeof setInterval> | null = null;
    let timeoutTimer: ReturnType<typeof setTimeout> | null = null;

    const cleanup = (value: number | null) => {
      if (settled) {
        return;
      }
      settled = true;
      if (pollTimer !== null) {
        clearInterval(pollTimer);
      }
      if (timeoutTimer !== null) {
        clearTimeout(timeoutTimer);
      }
      try {
        player?.destroy?.();
      } catch {
        // Ignore player teardown failures.
      }
      host.remove();
      resolve(value);
    };

    const maybeResolveDuration = () => {
      const duration = parseDetectedDuration(player?.getDuration?.());
      if (duration !== null) {
        cleanup(duration);
      }
    };

    timeoutTimer = setTimeout(() => cleanup(null), YOUTUBE_DURATION_TIMEOUT_MS);

    try {
      player = new api.Player(host, {
        height: 1,
        width: 1,
        videoId,
        playerVars: {
          autoplay: 0,
          controls: 0,
          rel: 0,
          fs: 0,
          playsinline: 1,
          origin: window.location.origin,
        },
        events: {
          onReady: (event) => {
            player = event.target;
            try {
              event.target.cueVideoById?.(videoId);
            } catch {
              // cueVideoById is best-effort; polling still handles detection if unsupported.
            }
            maybeResolveDuration();
            pollTimer = setInterval(maybeResolveDuration, YOUTUBE_DURATION_POLL_INTERVAL_MS);
          },
          onStateChange: (event) => {
            player = event.target;
            maybeResolveDuration();
          },
          onError: () => {
            cleanup(null);
          },
        },
      });
    } catch {
      cleanup(null);
    }
  });
}

function parseReelUrl(input: string): Omit<CommunityReelEmbed, "id"> | null {
  const url = toAbsoluteUrl(input);
  if (!url) {
    return null;
  }
  const host = url.hostname.toLowerCase();

  if (host.includes("youtube.com") || host.includes("youtu.be")) {
    const videoId = extractYouTubeVideoId(url);
    if (!videoId || !/^[A-Za-z0-9_-]{6,}$/.test(videoId)) {
      return null;
    }
    return {
      platform: "youtube",
      sourceUrl: url.toString(),
      embedUrl: `https://www.youtube.com/embed/${videoId}`,
    };
  }

  if (host.includes("instagram.com")) {
    const match = url.pathname.match(/^\/(reel|p|tv)\/([A-Za-z0-9_-]+)/);
    if (!match) {
      return null;
    }
    const kind = match[1];
    const code = match[2];
    return {
      platform: "instagram",
      sourceUrl: url.toString(),
      embedUrl: `https://www.instagram.com/${kind}/${code}/embed`,
    };
  }

  if (host.includes("tiktok.com")) {
    const match = url.pathname.match(/\/video\/(\d+)/);
    if (!match) {
      return null;
    }
    const videoId = match[1];
    return {
      platform: "tiktok",
      sourceUrl: url.toString(),
      embedUrl: `https://www.tiktok.com/embed/v2/${videoId}`,
    };
  }

  return null;
}

function parseStoredReels(raw: unknown): CommunityReelEmbed[] {
  if (!Array.isArray(raw)) {
    return [];
  }
  const parsed: CommunityReelEmbed[] = [];
  for (const [index, entry] of raw.entries()) {
    if (!entry || typeof entry !== "object" || Array.isArray(entry)) {
      continue;
    }
    const row = entry as Record<string, unknown>;
    const platform = row.platform;
    const sourceUrl = typeof row.sourceUrl === "string" ? row.sourceUrl.trim() : "";
    const embedUrl = typeof row.embedUrl === "string" ? row.embedUrl.trim() : "";
    const tStartRaw = Number(row.tStartSec);
    const tEndRaw = Number(row.tEndSec);
    const tStartSec = Number.isFinite(tStartRaw) && tStartRaw >= 0 ? tStartRaw : undefined;
    const tEndSec = Number.isFinite(tEndRaw) && tEndRaw > 0 && (tStartSec == null || tEndRaw > tStartRaw) ? tEndRaw : undefined;
    if (!sourceUrl || !embedUrl) {
      continue;
    }
    if (platform !== "youtube" && platform !== "instagram" && platform !== "tiktok") {
      continue;
    }
    parsed.push({
      id: typeof row.id === "string" && row.id.trim() ? row.id.trim() : `stored-reel-${index}`,
      platform,
      sourceUrl,
      embedUrl,
      tStartSec,
      tEndSec,
    });
  }
  return parsed;
}

function parseStoredSets(raw: string | null): CommunitySet[] {
  if (!raw) {
    return [];
  }
  try {
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) {
      return [];
    }
    const nowMs = Date.now();
    return parsed
      .filter((item) => item && typeof item === "object" && !Array.isArray(item))
      .map((item) => item as Partial<CommunitySet>)
      .filter(
        (item) =>
          typeof item.id === "string" &&
          Boolean(item.id.trim()) &&
          typeof item.title === "string" &&
          Boolean(item.title.trim()) &&
          typeof item.description === "string",
      )
      .map((item) => {
        const reels = parseStoredReels(item.reels);
        const reelCount = Math.max(reels.length, Math.max(0, Math.floor(Number(item.reelCount) || 0)));
        const fallbackUpdatedLabel =
          typeof item.updatedLabel === "string" && item.updatedLabel.trim() ? item.updatedLabel.trim() : "Last Edited: unknown";
        const normalizedCreatedAt = normalizeUpdatedAt(item.createdAt);
        const normalizedUpdatedAt =
          normalizeUpdatedAt(item.updatedAt) ?? normalizedCreatedAt ?? inferUpdatedAtFromLastEditedLabel(fallbackUpdatedLabel, nowMs);
        const normalizedUpdatedLabel =
          normalizedUpdatedAt == null && isRelativeNowLastEditedLabel(fallbackUpdatedLabel)
            ? "Last Edited: unknown"
            : normalizeLastEditedLabel(fallbackUpdatedLabel);
        return {
          id: item.id!.trim(),
          title: item.title!.trim(),
          description: item.description!.trim(),
          tags: Array.isArray(item.tags)
            ? item.tags
                .map((tag) => String(tag || "").trim().toLowerCase())
                .filter(Boolean)
                .slice(0, MAX_SET_TAGS)
            : [],
          reels,
          reelCount,
          curator: typeof item.curator === "string" && item.curator.trim() ? item.curator.trim() : "Community member",
          likes: Math.max(0, Math.floor(Number(item.likes) || 0)),
          learners: Math.max(0, Math.floor(Number(item.learners) || 0)),
          updatedLabel: normalizedUpdatedLabel,
          updatedAt: normalizedUpdatedAt,
          createdAt: normalizedCreatedAt,
          thumbnailUrl: typeof item.thumbnailUrl === "string" && item.thumbnailUrl.trim() ? item.thumbnailUrl.trim() : FALLBACK_THUMBNAIL_URL,
          featured: false,
        } as CommunitySet;
      })
      .slice(0, MAX_USER_SETS);
  } catch {
    return [];
  }
}

function parseStoredSetSnapshot(raw: string | null): CommunitySet | null {
  if (!raw) {
    return null;
  }
  try {
    const parsed = JSON.parse(raw);
    const normalized = parseStoredSets(JSON.stringify([parsed]));
    return normalized[0] ?? null;
  } catch {
    return null;
  }
}

function parseStoredSetDraft(raw: string | null): StoredSetDraft | null {
  if (!raw) {
    return null;
  }
  try {
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return null;
    }
    const row = parsed as Record<string, unknown>;
    const reelInputsRaw = Array.isArray(row.reelInputs) ? row.reelInputs : [];
    const reelInputs = reelInputsRaw
      .map((entry) => {
        if (!entry || typeof entry !== "object" || Array.isArray(entry)) {
          return null;
        }
        const reelRow = entry as Record<string, unknown>;
        return {
          ...(typeof reelRow.communityReelId === "string" && reelRow.communityReelId.trim()
            ? { communityReelId: reelRow.communityReelId.trim() }
            : {}),
          value: typeof reelRow.value === "string" ? reelRow.value : "",
          tStartSec: typeof reelRow.tStartSec === "string" ? reelRow.tStartSec : "0",
          tEndSec: typeof reelRow.tEndSec === "string" ? reelRow.tEndSec : "",
        };
      })
      .filter(Boolean) as StoredSetDraft["reelInputs"];
    return {
      title: typeof row.title === "string" ? row.title : "",
      description: typeof row.description === "string" ? row.description : "",
      tags: typeof row.tags === "string" ? row.tags : "",
      thumbnailPreview: typeof row.thumbnailPreview === "string" ? row.thumbnailPreview : "",
      thumbnailFileName: typeof row.thumbnailFileName === "string" ? row.thumbnailFileName : "",
      reelInputs: reelInputs.length > 0 ? reelInputs : [{ value: "", tStartSec: "0", tEndSec: "" }],
    };
  } catch {
    return null;
  }
}

let localDemoSetSequence = 0;

function buildLocalDemoSet(params: {
  existingSet?: CommunitySet | null;
  setId?: string;
  title: string;
  description: string;
  tags: string[];
  reels: Array<Omit<CommunityReelEmbed, "id"> & { id?: string }>;
  thumbnailUrl: string;
}): CommunitySet {
  const now = new Date().toISOString();
  localDemoSetSequence += 1;
  const id = params.existingSet?.id
    ?? params.setId?.trim()
    ?? `local-demo-set-created-${Date.now()}-${localDemoSetSequence}`;
  const reels = params.reels.map((reel, index) => ({
    ...reel,
    id: reel.id?.trim() || `${id}-reel-${index + 1}`,
  }));
  return {
    id,
    title: params.title,
    description: params.description,
    tags: params.tags,
    reels,
    reelCount: reels.length,
    curator: LOCAL_DEMO_ACCOUNT.username,
    likes: params.existingSet?.likes ?? 0,
    learners: params.existingSet?.learners ?? 0,
    updatedLabel: "Last Edited: just now",
    updatedAt: now,
    createdAt: params.existingSet?.createdAt ?? now,
    thumbnailUrl: params.thumbnailUrl,
    featured: params.existingSet?.featured ?? false,
  };
}

function draftRowsFromReels(reels: CommunityReelEmbed[]): DraftReelInput[] {
  if (reels.length === 0) {
    return [createDraftReelRow()];
  }
  return reels.map((reel) => {
    const start = Number.isFinite(reel.tStartSec) ? Number(reel.tStartSec) : 0;
    const endCandidate = Number(reel.tEndSec);
    const hasExplicitEnd = Number.isFinite(endCandidate) && endCandidate > start;
    return createDraftReelRow(
      reel.sourceUrl,
      formatClipSecondsInputValue(start),
      hasExplicitEnd ? formatClipSecondsInputValue(endCandidate) : "",
      reel.id,
    );
  });
}

function getSetReelCount(set: CommunitySet): number {
  return set.reels.length > 0 ? set.reels.length : set.reelCount;
}

function summarizePlatforms(reels: CommunityReelEmbed[]): ReelPlatform[] {
  return Array.from(new Set(reels.map((reel) => reel.platform)));
}

function toTitleCase(value: string): string {
  return value
    .split(/[\s_-]+/)
    .filter(Boolean)
    .map((token) => token[0].toUpperCase() + token.slice(1).toLowerCase())
    .join(" ");
}

function getSetIconClass(set: CommunitySet): string {
  const platforms = summarizePlatforms(set.reels);
  if (platforms.includes("youtube")) {
    return "fa-brands fa-youtube";
  }
  if (platforms.includes("instagram")) {
    return "fa-brands fa-instagram";
  }
  if (platforms.includes("tiktok")) {
    return "fa-brands fa-tiktok";
  }
  return "fa-solid fa-layer-group";
}

function parseClipSecondsInput(value: string): number | null {
  const trimmed = value.trim();
  if (!trimmed) {
    return null;
  }
  const parsed = Number(trimmed);
  if (!Number.isFinite(parsed) || parsed < 0) {
    return null;
  }
  return parsed;
}

function formatClipSecondsInputValue(value: number): string {
  if (!Number.isFinite(value)) {
    return "0";
  }
  const rounded = Math.round(value * 10) / 10;
  return rounded.toFixed(1).replace(/\.0$/, "");
}

function formatClipRangeLabel(reel: CommunityReelEmbed): string | null {
  const start = Number(reel.tStartSec);
  const end = Number(reel.tEndSec);
  if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) {
    return null;
  }
  return `${start.toFixed(1).replace(/\.0$/, "")}s - ${end.toFixed(1).replace(/\.0$/, "")}s`;
}

type CommunityReelsPanelMode = "community" | "create" | "edit";

type CommunityReelsPanelProps = {
  mode?: CommunityReelsPanelMode;
  demoMode?: boolean;
  isVisible?: boolean;
  onDetailOpenChange?: (isOpen: boolean) => void;
  onActiveCommunitySetChange?: (setId: string | null) => void;
  initialOpenSetId?: string | null;
  communityResetSignal?: number;
  onDraftUnsavedChangesChange?: (hasUnsavedChanges: boolean) => void;
  onDraftExitActionsChange?: (actions: CommunityDraftExitActions | null) => void;
  onOpenCommunityReelInFeed?: (payload: { setId: string; setTitle: string; selectedReelId: string; feedQuery: string }) => void;
};

type FeaturedTransitionStage = "idle" | "exiting" | "pause" | "entering";

export function CommunityReelsPanel({
  mode = "community",
  demoMode = false,
  isVisible = true,
  onDetailOpenChange,
  onActiveCommunitySetChange,
  initialOpenSetId = null,
  communityResetSignal = 0,
  onDraftUnsavedChangesChange,
  onDraftExitActionsChange,
  onOpenCommunityReelInFeed,
}: CommunityReelsPanelProps) {
  const router = useRouter();
  const localDemoEnabled = demoMode && LOCAL_DEMO_AVAILABLE;
  const starredSetsStorageKey = localDemoEnabled
    ? LOCAL_DEMO_STARRED_SET_IDS_STORAGE_KEY
    : COMMUNITY_STARRED_SET_IDS_STORAGE_KEY;
  const createDraftStorageKey = localDemoEnabled
    ? LOCAL_DEMO_CREATE_DRAFT_STORAGE_KEY
    : COMMUNITY_CREATE_DRAFT_STORAGE_KEY;
  const editDraftStoragePrefix = localDemoEnabled
    ? LOCAL_DEMO_EDIT_DRAFT_PREFIX
    : COMMUNITY_EDIT_DRAFT_PREFIX;
  const [activeCommunityCategory, setActiveCommunityCategory] = useState("Featured");
  const [activeFeaturedIndex, setActiveFeaturedIndex] = useState(0);
  const [leavingFeaturedIndex, setLeavingFeaturedIndex] = useState<number | null>(null);
  const [pendingFeaturedIndex, setPendingFeaturedIndex] = useState<number | null>(null);
  const [featuredTransitionStage, setFeaturedTransitionStage] = useState<FeaturedTransitionStage>("idle");
  const [featuredCarouselContentHeight, setFeaturedCarouselContentHeight] = useState(FEATURED_CAROUSEL_CONTENT_MIN_HEIGHT_FALLBACK);
  const [selectedDirectorySet, setSelectedDirectorySet] = useState<CommunitySet | null>(null);
  const [isDirectoryDetailOpen, setIsDirectoryDetailOpen] = useState(false);
  const [communityQuery, setCommunityQuery] = useState("");
  const [yourSetsQuery, setYourSetsQuery] = useState("");
  const [yourSetsModifiedSortDirection, setYourSetsModifiedSortDirection] = useState<"newest" | "oldest">("newest");
  const [isCompactCommunitySearchOpen, setIsCompactCommunitySearchOpen] = useState(false);
  const [isCompactYourSetsSearchOpen, setIsCompactYourSetsSearchOpen] = useState(false);
  const [setTitle, setSetTitle] = useState("");
  const [setDescription, setSetDescription] = useState("");
  const [setTags, setSetTags] = useState("");
  const [thumbnailPreview, setThumbnailPreview] = useState("");
  const [detailCarouselIndex, setDetailCarouselIndex] = useState(0);
  const [selectedDetailReelId, setSelectedDetailReelId] = useState<string | null>(null);
  const [reelInputs, setReelInputs] = useState<DraftReelInput[]>(() => [createDraftReelRow()]);
  const [reelDurationByRow, setReelDurationByRow] = useState<Record<string, ReelDurationState>>({});
  const [, setCreateError] = useState<string | null>(null);
  const [createSuccess, setCreateSuccess] = useState<string | null>(null);
  const [tagLimitError, setTagLimitError] = useState(false);
  const [publishResultModal, setPublishResultModal] = useState<PublishResultModalState | null>(null);
  const [deleteSetConfirmModal, setDeleteSetConfirmModal] = useState<DeleteSetConfirmModalState | null>(null);
  const [unsavedDraftExitModal, setUnsavedDraftExitModal] = useState<UnsavedDraftExitModalState | null>(null);
  const [draftActionConfirmModal, setDraftActionConfirmModal] = useState<DraftActionConfirmModalState | null>(null);
  const [isPostingSet, setIsPostingSet] = useState(false);
  const [activeEditSetId, setActiveEditSetId] = useState<string | null>(null);
  const [isEditSetEditorOpen, setIsEditSetEditorOpen] = useState(false);
  const [isCreateSetEditorOpen, setIsCreateSetEditorOpen] = useState(false);
  const [starredSetIds, setStarredSetIds] = useState<string[]>([]);
  const [starredSetsHydrated, setStarredSetsHydrated] = useState(false);
  const [activeSetActionsMenuId, setActiveSetActionsMenuId] = useState<string | null>(null);
  const [selectedEditableSetIds, setSelectedEditableSetIds] = useState<string[]>([]);
  const [deletingSetIds, setDeletingSetIds] = useState<string[]>([]);
  const [publicSets, setPublicSets] = useState<CommunitySet[]>([]);
  const [ownedSets, setOwnedSets] = useState<CommunitySet[]>(() => (
    localDemoEnabled ? [...LOCAL_DEMO_COMMUNITY_SETS] : []
  ));
  const [ownedSetIds, setOwnedSetIds] = useState<string[]>(() => (
    localDemoEnabled ? LOCAL_DEMO_COMMUNITY_SETS.map((set) => set.id) : []
  ));
  const [communityAccount, setCommunityAccount] = useState<CommunityAccount | null>(
    localDemoEnabled ? LOCAL_DEMO_ACCOUNT : null,
  );
  const [authBusy, setAuthBusy] = useState(false);
  const [authHydrated, setAuthHydrated] = useState(localDemoEnabled);
  const [storageHydrated, setStorageHydrated] = useState(false);
  const [relativeTimeNowMs, setRelativeTimeNowMs] = useState(() => Date.now());
  const [skipDetailTransitionOnce, setSkipDetailTransitionOnce] = useState(false);
  const [isInitialDetailRestorePending, setIsInitialDetailRestorePending] = useState(
    () => mode === "community" && Boolean(initialOpenSetId?.trim()),
  );
  const [isThumbnailDragOver, setIsThumbnailDragOver] = useState(false);
  const [thumbnailFileName, setThumbnailFileName] = useState("");
  const directoryViewRef = useRef<HTMLDivElement | null>(null);
  const detailContentScrollRef = useRef<HTMLDivElement | null>(null);
  const detailBackButtonRef = useRef<HTMLButtonElement | null>(null);
  const detailReturnFocusRef = useRef<HTMLElement | null>(null);
  const communityScrollRef = useRef<HTMLDivElement | null>(null);
  const compactCommunitySearchInputRef = useRef<HTMLInputElement | null>(null);
  const compactYourSetsSearchInputRef = useRef<HTMLInputElement | null>(null);
  const activeFeaturedSlideRef = useRef<HTMLDivElement | null>(null);
  const directoryDetailCloseTimerRef = useRef<number | null>(null);
  const reelDurationCacheRef = useRef<Record<string, number | null>>({});
  const consumedInitialSetIdRef = useRef<string | null>(null);
  const loadedEditSetIdRef = useRef<string | null>(null);
  const lastCommunityResetSignalRef = useRef(communityResetSignal);
  const draftBaselinesByContextRef = useRef<Record<string, string>>({});
  const ownedSetsAccountIdRef = useRef<string | null>(null);
  const [draftBaselineVersion, setDraftBaselineVersion] = useState(0);

  const clearOwnedCommunityState = useCallback(() => {
    setOwnedSets([]);
    setOwnedSetIds([]);
    setSelectedEditableSetIds([]);
    saveOwnedCommunitySetIds([]);
  }, []);

  const refreshOwnedCommunitySets = useCallback(async () => {
    const remoteOwnedSets = await fetchOwnedCommunitySets();
    const nextOwnedIds = remoteOwnedSets.map((set) => set.id);
    setOwnedSets(remoteOwnedSets.slice(0, MAX_USER_SETS));
    setOwnedSetIds(saveOwnedCommunitySetIds(nextOwnedIds));
    return remoteOwnedSets;
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    let cancelled = false;
    let authSyncSequence = 0;
    const starredSetIdsRaw = window.localStorage.getItem(starredSetsStorageKey);
    if (starredSetIdsRaw) {
      try {
        const parsed = JSON.parse(starredSetIdsRaw);
        if (Array.isArray(parsed)) {
          setStarredSetIds(
            parsed
              .map((value) => String(value || "").trim())
              .filter(Boolean),
          );
        }
      } catch {
        // Ignore malformed starred set storage values.
      }
    } else {
      setStarredSetIds([]);
    }
    setStarredSetsHydrated(true);

    if (localDemoEnabled) {
      const storedDemoSetsRaw = window.localStorage.getItem(LOCAL_DEMO_SETS_STORAGE_KEY);
      const demoSets = storedDemoSetsRaw === null
        ? [...LOCAL_DEMO_COMMUNITY_SETS]
        : parseStoredSets(storedDemoSetsRaw);
      setCommunityAccount(LOCAL_DEMO_ACCOUNT);
      setOwnedSets(demoSets);
      setOwnedSetIds(demoSets.map((set) => set.id));
      setPublicSets([]);
      ownedSetsAccountIdRef.current = LOCAL_DEMO_ACCOUNT.id;
      setAuthHydrated(true);
      setStorageHydrated(true);
      return () => {
        cancelled = true;
        authSyncSequence += 1;
      };
    }

    setAuthHydrated(false);
    safeStorageRemoveItem(window.localStorage, COMMUNITY_SETS_STORAGE_KEY);
    setStorageHydrated(true);

    const syncCommunityAuthState = async (options?: { validateSession?: boolean }) => {
      const syncSequence = authSyncSequence + 1;
      authSyncSequence = syncSequence;
      const storedSession = readCommunityAuthSession();
      const storedAccount = storedSession?.account ?? null;
      setCommunityAccount(storedAccount);
      if (!storedSession?.sessionToken) {
        ownedSetsAccountIdRef.current = null;
        clearOwnedCommunityState();
        if (!cancelled) {
          setAuthHydrated(true);
        }
        return;
      }
      if (ownedSetsAccountIdRef.current && ownedSetsAccountIdRef.current !== (storedAccount?.id ?? null)) {
        clearOwnedCommunityState();
      }
      try {
        const account = options?.validateSession ? await fetchCommunityAccount() : storedAccount;
        if (cancelled || authSyncSequence !== syncSequence) {
          return;
        }
        if (!account) {
          setCommunityAccount(null);
          ownedSetsAccountIdRef.current = null;
          clearOwnedCommunityState();
          return;
        }
        setCommunityAccount(account);
        if (account.isVerified) {
          await refreshOwnedCommunitySets();
          ownedSetsAccountIdRef.current = account.id;
        } else {
          ownedSetsAccountIdRef.current = account.id;
          clearOwnedCommunityState();
        }
      } catch (error) {
        if (cancelled || authSyncSequence != syncSequence) {
          return;
        }
        if (isSessionExpiredError(error)) {
          expireCommunityAuthSession();
          setCommunityAccount(null);
          ownedSetsAccountIdRef.current = null;
          clearOwnedCommunityState();
        }
      } finally {
        if (!cancelled && authSyncSequence === syncSequence) {
          setAuthHydrated(true);
        }
      }
    };

    void (async () => {
      try {
        const remoteSets = await fetchCommunitySets();
        if (cancelled) {
          return;
        }
        setPublicSets(remoteSets.slice(0, MAX_USER_SETS));
      } catch {
        // Keep the public list empty if the backend is unavailable.
      }
    })();
    void syncCommunityAuthState({ validateSession: true });

    const onStorage = (event: StorageEvent) => {
      if (event.storageArea !== window.localStorage) {
        return;
      }
      if (
        event.key
        && event.key !== "studyreels-community-account"
        && event.key !== "studyreels-community-session-token"
        && event.key !== COMMUNITY_OWNED_SET_IDS_STORAGE_KEY
      ) {
        return;
      }
      void syncCommunityAuthState();
    };
    const onAuthChanged = () => {
      void syncCommunityAuthState();
    };
    window.addEventListener("storage", onStorage);
    window.addEventListener(COMMUNITY_AUTH_CHANGED_EVENT, onAuthChanged);

    return () => {
      cancelled = true;
      authSyncSequence += 1;
      window.removeEventListener("storage", onStorage);
      window.removeEventListener(COMMUNITY_AUTH_CHANGED_EVENT, onAuthChanged);
    };
  }, [clearOwnedCommunityState, localDemoEnabled, refreshOwnedCommunitySets, starredSetsStorageKey]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const timer = window.setInterval(() => {
      setRelativeTimeNowMs(Date.now());
    }, LAST_EDITED_REFRESH_INTERVAL_MS);
    return () => {
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    if (typeof window === "undefined" || !storageHydrated) {
      return;
    }
    if (localDemoEnabled) {
      safeStorageSetItem(window.localStorage, LOCAL_DEMO_SETS_STORAGE_KEY, JSON.stringify(ownedSets));
      return;
    }
    saveOwnedCommunitySetIds(ownedSetIds);
  }, [localDemoEnabled, ownedSetIds, ownedSets, storageHydrated]);

  useEffect(() => {
    if (typeof window === "undefined" || !starredSetsHydrated) {
      return;
    }
    safeStorageSetItem(window.localStorage, starredSetsStorageKey, JSON.stringify(starredSetIds));
  }, [starredSetIds, starredSetsHydrated, starredSetsStorageKey]);

  useEffect(() => {
    if (!starredSetsHydrated) {
      return;
    }
    const editableSetIdSet = new Set(ownedSetIds);
    setStarredSetIds((prev) => {
      const next = prev.filter((setId) => editableSetIdSet.has(setId));
      if (next.length === prev.length) {
        return prev;
      }
      return next;
    });
  }, [ownedSetIds, starredSetsHydrated]);

  useEffect(() => {
    if (!activeSetActionsMenuId) {
      return;
    }
    const onPointerDown = (event: PointerEvent) => {
      if (!(event.target instanceof Element)) {
        return;
      }
      if (event.target.closest("[data-your-set-actions='true']")) {
        return;
      }
      setActiveSetActionsMenuId(null);
    };
    window.addEventListener("pointerdown", onPointerDown);
    return () => {
      window.removeEventListener("pointerdown", onPointerDown);
    };
  }, [activeSetActionsMenuId]);

  const clearDirectoryDetailCloseTimer = useCallback(() => {
    if (directoryDetailCloseTimerRef.current !== null) {
      window.clearTimeout(directoryDetailCloseTimerRef.current);
      directoryDetailCloseTimerRef.current = null;
    }
  }, []);

  useEffect(() => {
    return () => {
      clearDirectoryDetailCloseTimer();
    };
  }, [clearDirectoryDetailCloseTimer]);

  const isYourSetsMode = mode === "edit";
  const isStandaloneCreateMode = mode === "create";
  const isFormEditMode = isYourSetsMode && isEditSetEditorOpen;
  const isFormCreateMode = isStandaloneCreateMode || (isYourSetsMode && isCreateSetEditorOpen);
  const shouldShowEditSetGrid = isYourSetsMode && !isEditSetEditorOpen && !isCreateSetEditorOpen;
  const shouldShowEditSetForm = isFormEditMode || isFormCreateMode;
  const requiresCommunityAuth = isYourSetsMode || isStandaloneCreateMode;
  const isCommunityAuthReady = !requiresCommunityAuth || authHydrated;
  const needsCommunityAuth = requiresCommunityAuth && !communityAccount;
  const needsCommunityVerification = requiresCommunityAuth && communityAccount?.isVerified === false;
  const canManageYourSets = communityAccount?.isVerified === true;
  const shouldShowYourSetsSearch = isYourSetsMode && shouldShowEditSetGrid && canManageYourSets;
  const normalizedInitialOpenSetId = initialOpenSetId?.trim() || "";
  const shouldSuppressDirectoryDuringRestore =
    mode === "community" && isVisible && Boolean(normalizedInitialOpenSetId) && isInitialDetailRestorePending;
  const ownedSetIdSet = useMemo(() => new Set(ownedSetIds), [ownedSetIds]);
  const editableSets = useMemo(() => ownedSets.filter((set) => ownedSetIdSet.has(set.id)), [ownedSetIdSet, ownedSets]);
  const allSets = useMemo(() => {
    const merged: CommunitySet[] = [];
    const seen = new Set<string>();
    for (const bucket of [editableSets, publicSets, DEFAULT_COMMUNITY_SETS]) {
      for (const set of bucket) {
        if (seen.has(set.id)) {
          continue;
        }
        seen.add(set.id);
        merged.push(set);
      }
    }
    return merged;
  }, [editableSets, publicSets]);
  const starredSetIdSet = useMemo(() => new Set(starredSetIds), [starredSetIds]);
  const selectedEditableSetIdSet = useMemo(() => new Set(selectedEditableSetIds), [selectedEditableSetIds]);
  const deletingSetIdSet = useMemo(() => new Set(deletingSetIds), [deletingSetIds]);
  const isDeletingSets = deletingSetIds.length > 0;
  const filteredEditableSets = useMemo(() => {
    const normalized = yourSetsQuery.trim().toLowerCase();
    const filtered = normalized
      ? editableSets.filter((set) => matchesSetSearchQuery(set, normalized))
      : [...editableSets];
    return filtered.sort((left, right) => {
      const leftUpdatedMs = parseTimestampMs(left.updatedAt);
      const rightUpdatedMs = parseTimestampMs(right.updatedAt);
      if (leftUpdatedMs == null && rightUpdatedMs != null) {
        return 1;
      }
      if (leftUpdatedMs != null && rightUpdatedMs == null) {
        return -1;
      }
      if (leftUpdatedMs != null && rightUpdatedMs != null && leftUpdatedMs !== rightUpdatedMs) {
        return yourSetsModifiedSortDirection === "newest"
          ? rightUpdatedMs - leftUpdatedMs
          : leftUpdatedMs - rightUpdatedMs;
      }
      return left.title.localeCompare(right.title);
    });
  }, [editableSets, yourSetsModifiedSortDirection, yourSetsQuery]);
  const activeEditableSet = useMemo(
    () => editableSets.find((set) => set.id === activeEditSetId) ?? null,
    [activeEditSetId, editableSets],
  );
  const featuredCarouselSets = useMemo(() => FEATURED_SETS.slice(0, 3), []);
  const detailCarouselReels = useMemo(() => selectedDirectorySet?.reels ?? [], [selectedDirectorySet]);

  useEffect(() => {
    if (!shouldShowEditSetGrid || !canManageYourSets) {
      setSelectedEditableSetIds((prev) => (prev.length > 0 ? [] : prev));
      return;
    }
    setSelectedEditableSetIds((prev) => {
      const next = prev.filter((setId) => ownedSetIdSet.has(setId));
      return next.length === prev.length ? prev : next;
    });
  }, [canManageYourSets, ownedSetIdSet, shouldShowEditSetGrid]);

  useEffect(() => {
    if (mode === "edit") {
      setIsEditSetEditorOpen(false);
      setIsCreateSetEditorOpen(false);
      setCreateError(null);
      setCreateSuccess(null);
      return;
    }
    setIsEditSetEditorOpen(false);
    setIsCreateSetEditorOpen(false);
  }, [mode]);

  const filteredDirectorySets = useMemo(() => {
    const normalized = communityQuery.trim().toLowerCase();
    if (!normalized) {
      return allSets;
    }
    return allSets.filter((set) => matchesSetSearchQuery(set, normalized));
  }, [allSets, communityQuery]);

  const communityCategories = useMemo(() => {
    const tagCounts = new Map<string, number>();
    for (const set of allSets) {
      for (const tag of set.tags) {
        const normalized = tag.trim().toLowerCase();
        if (!normalized) {
          continue;
        }
        tagCounts.set(normalized, (tagCounts.get(normalized) ?? 0) + 1);
      }
    }
    const topTags = [...tagCounts.entries()]
      .sort((a, b) => b[1] - a[1])
      .map(([tag]) => toTitleCase(tag))
      .filter(Boolean)
      .slice(0, 3);
    return ["Featured", ...topTags];
  }, [allSets]);

  useEffect(() => {
    if (!communityCategories.includes(activeCommunityCategory)) {
      setActiveCommunityCategory(communityCategories[0] ?? "Featured");
    }
  }, [activeCommunityCategory, communityCategories]);

  const categoryFilteredSets = useMemo(() => {
    const hasQuery = communityQuery.trim().length > 0;
    if (hasQuery) {
      return filteredDirectorySets;
    }

    if (activeCommunityCategory === "Featured") {
      const featuredFirst = filteredDirectorySets.filter((set) => set.featured);
      const others = filteredDirectorySets.filter((set) => !set.featured);
      return [...featuredFirst, ...others];
    }

    const normalizedCategory = activeCommunityCategory.trim().toLowerCase();
    return filteredDirectorySets.filter((set) => {
      if (set.title.toLowerCase().includes(normalizedCategory)) {
        return true;
      }
      if (set.description.toLowerCase().includes(normalizedCategory)) {
        return true;
      }
      return set.tags.some((tag) => tag.toLowerCase().includes(normalizedCategory));
    });
  }, [activeCommunityCategory, communityQuery, filteredDirectorySets]);

  useEffect(() => {
    if (featuredCarouselSets.length === 0) {
      setActiveFeaturedIndex(0);
      setLeavingFeaturedIndex(null);
      setPendingFeaturedIndex(null);
      setFeaturedTransitionStage("idle");
      return;
    }
    setActiveFeaturedIndex((prev) => {
      if (prev < featuredCarouselSets.length) {
        return prev;
      }
      return 0;
    });
    setLeavingFeaturedIndex(null);
    setPendingFeaturedIndex(null);
    setFeaturedTransitionStage("idle");
  }, [featuredCarouselSets.length]);

  const startFeaturedTransition = useCallback((nextIndex: number) => {
    if (featuredCarouselSets.length <= 1 || featuredTransitionStage !== "idle") {
      return;
    }
    const setCount = featuredCarouselSets.length;
    const normalized = ((nextIndex % setCount) + setCount) % setCount;
    if (normalized === activeFeaturedIndex) {
      return;
    }
    setLeavingFeaturedIndex(activeFeaturedIndex);
    setPendingFeaturedIndex(normalized);
    setFeaturedTransitionStage("exiting");
  }, [activeFeaturedIndex, featuredCarouselSets.length, featuredTransitionStage]);

  const goToFeaturedSet = useCallback(
    (nextIndex: number) => {
      startFeaturedTransition(nextIndex);
    },
    [startFeaturedTransition],
  );

  const goToNextFeaturedSet = useCallback(() => {
    startFeaturedTransition(activeFeaturedIndex + 1);
  }, [activeFeaturedIndex, startFeaturedTransition]);

  useEffect(() => {
    if (mode !== "community" || isDirectoryDetailOpen || featuredCarouselSets.length <= 1) {
      return;
    }
    const intervalId = window.setInterval(() => {
      goToNextFeaturedSet();
    }, FEATURED_CAROUSEL_INTERVAL_MS);
    return () => {
      window.clearInterval(intervalId);
    };
  }, [featuredCarouselSets.length, goToNextFeaturedSet, isDirectoryDetailOpen, mode]);

  useEffect(() => {
    if (featuredTransitionStage !== "exiting") {
      return;
    }
    const timeoutId = window.setTimeout(() => {
      setLeavingFeaturedIndex(null);
      setFeaturedTransitionStage("pause");
    }, FEATURED_CAROUSEL_TRANSITION_MS);
    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [featuredTransitionStage]);

  useEffect(() => {
    if (featuredTransitionStage !== "pause") {
      return;
    }
    const timeoutId = window.setTimeout(() => {
      if (pendingFeaturedIndex === null) {
        setFeaturedTransitionStage("idle");
        return;
      }
      setActiveFeaturedIndex(pendingFeaturedIndex);
      setPendingFeaturedIndex(null);
      setFeaturedTransitionStage("entering");
    }, FEATURED_CAROUSEL_PAUSE_MS);
    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [featuredTransitionStage, pendingFeaturedIndex]);

  useEffect(() => {
    if (featuredTransitionStage !== "entering") {
      return;
    }
    const timeoutId = window.setTimeout(() => {
      setFeaturedTransitionStage("idle");
    }, FEATURED_CAROUSEL_TRANSITION_MS);
    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [featuredTransitionStage]);

  const isCommunitySearchActive = communityQuery.trim().length > 0;
  const isYourSetsSearchActive = yourSetsQuery.trim().length > 0;
  const isCompactCommunitySearchExpanded = isCompactCommunitySearchOpen || isCommunitySearchActive;
  const isCompactYourSetsSearchExpanded = isCompactYourSetsSearchOpen || isYourSetsSearchActive;

  useEffect(() => {
    if (mode !== "community" || isCommunitySearchActive || featuredCarouselSets.length === 0) {
      return;
    }
    const measure = () => {
      const activeSlide = activeFeaturedSlideRef.current;
      if (!activeSlide) {
        return;
      }
      const slideRect = activeSlide.getBoundingClientRect();
      const imageTarget = activeSlide.querySelector<HTMLElement>("[data-featured-image-target]");
      const ctaButton = activeSlide.querySelector<HTMLButtonElement>("[data-featured-view-set-button]");
      const imageBottom =
        imageTarget && imageTarget.offsetParent !== null
          ? imageTarget.getBoundingClientRect().bottom - slideRect.top + FEATURED_CAROUSEL_IMAGE_BOTTOM_MARGIN_PX
          : 0;
      const buttonBottom =
        ctaButton && ctaButton.offsetParent !== null
          ? ctaButton.getBoundingClientRect().bottom - slideRect.top + FEATURED_CAROUSEL_BUTTON_BOTTOM_MARGIN_PX
          : 0;
      const measuredHeight = Math.ceil(activeSlide.scrollHeight);
      const nextHeight = imageBottom > 0
        ? Math.max(FEATURED_CAROUSEL_CONTENT_MIN_HEIGHT_FALLBACK, Math.ceil(imageBottom))
        : buttonBottom > 0
          ? Math.max(FEATURED_CAROUSEL_CONTENT_MIN_HEIGHT_TOUCH_FALLBACK, Math.ceil(buttonBottom))
          : Math.max(FEATURED_CAROUSEL_CONTENT_MIN_HEIGHT_FALLBACK, measuredHeight);
      setFeaturedCarouselContentHeight((prev) => (prev === nextHeight ? prev : nextHeight));
    };
    measure();
    const rafId = window.requestAnimationFrame(measure);
    window.addEventListener("resize", measure);

    const resizeObserver = typeof ResizeObserver !== "undefined" && activeFeaturedSlideRef.current ? new ResizeObserver(measure) : null;
    if (resizeObserver && activeFeaturedSlideRef.current) {
      resizeObserver.observe(activeFeaturedSlideRef.current);
    }

    return () => {
      window.cancelAnimationFrame(rafId);
      window.removeEventListener("resize", measure);
      resizeObserver?.disconnect();
    };
  }, [activeFeaturedIndex, featuredCarouselSets.length, featuredTransitionStage, isCommunitySearchActive, mode]);

  const directorySets = categoryFilteredSets;
  const detailCarouselCount = detailCarouselReels.length;
  const maxDetailCarouselIndex = Math.max(0, detailCarouselCount - 1);
  const activeDetailCarouselReel = detailCarouselReels[detailCarouselIndex] ?? null;
  const selectedDirectorySetCuratorLabel = useMemo(() => {
    if (!selectedDirectorySet) {
      return "Community member";
    }
    const rawCurator = selectedDirectorySet.curator.trim();
    if (!rawCurator) {
      return "Community member";
    }
    if (rawCurator.toLowerCase() === "you") {
      const username = communityAccount?.username?.trim();
      return username || rawCurator;
    }
    return rawCurator;
  }, [communityAccount?.username, selectedDirectorySet]);

  const goToPreviousDetailCarousel = useCallback(() => {
    setDetailCarouselIndex((prev) => {
      if (detailCarouselCount <= 1) {
        return 0;
      }
      return (prev - 1 + detailCarouselCount) % detailCarouselCount;
    });
  }, [detailCarouselCount]);

  const goToNextDetailCarousel = useCallback(() => {
    setDetailCarouselIndex((prev) => {
      if (detailCarouselCount <= 1) {
        return 0;
      }
      return (prev + 1) % detailCarouselCount;
    });
  }, [detailCarouselCount]);

  const onCommunityCategoryChange = useCallback((category: string) => {
    if (category === activeCommunityCategory) {
      return;
    }
    const previousScrollTop = communityScrollRef.current?.scrollTop ?? 0;
    setActiveCommunityCategory(category);
    if (typeof window === "undefined") {
      return;
    }
    window.requestAnimationFrame(() => {
      const container = communityScrollRef.current;
      if (!container) {
        return;
      }
      container.scrollTop = Math.min(previousScrollTop, container.scrollHeight - container.clientHeight);
    });
  }, [activeCommunityCategory]);

  const formDraftContextKey = useMemo(() => {
    if (isFormEditMode && activeEditableSet) {
      return `edit:${activeEditableSet.id}`;
    }
    if (isFormCreateMode) {
      return COMMUNITY_CREATE_DRAFT_CONTEXT_KEY;
    }
    return null;
  }, [activeEditableSet, isFormCreateMode, isFormEditMode]);

  const normalizeStoredDraft = useCallback((draft: StoredSetDraft): StoredSetDraft => {
    const reelInputs = Array.isArray(draft.reelInputs) && draft.reelInputs.length > 0
      ? draft.reelInputs.map((row) => ({
          ...(typeof row.communityReelId === "string" && row.communityReelId.trim()
            ? { communityReelId: row.communityReelId.trim() }
            : {}),
          value: String(row.value ?? ""),
          tStartSec: String(row.tStartSec ?? "0"),
          tEndSec: String(row.tEndSec ?? ""),
        }))
      : [{ value: "", tStartSec: "0", tEndSec: "" }];
    return {
      title: String(draft.title ?? ""),
      description: String(draft.description ?? ""),
      tags: String(draft.tags ?? ""),
      thumbnailPreview: String(draft.thumbnailPreview ?? ""),
      thumbnailFileName: String(draft.thumbnailFileName ?? ""),
      reelInputs,
    };
  }, []);

  const setDraftBaselineForContext = useCallback((contextKey: string, draft: StoredSetDraft) => {
    const normalizedDraft = normalizeStoredDraft(draft);
    draftBaselinesByContextRef.current[contextKey] = JSON.stringify(normalizedDraft);
    setDraftBaselineVersion((prev) => prev + 1);
  }, [normalizeStoredDraft]);

  const buildCurrentDraftPayload = useCallback((): StoredSetDraft => {
    return normalizeStoredDraft({
      title: setTitle,
      description: setDescription,
      tags: setTags,
      thumbnailPreview,
      thumbnailFileName,
      reelInputs: reelInputs.map((row) => ({
        ...(typeof row.communityReelId === "string" && row.communityReelId.trim()
          ? { communityReelId: row.communityReelId.trim() }
          : {}),
        value: row.value,
        tStartSec: row.tStartSec,
        tEndSec: row.tEndSec,
      })),
    });
  }, [normalizeStoredDraft, reelInputs, setDescription, setTags, setTitle, thumbnailFileName, thumbnailPreview]);

  const buildCanonicalDraftForUnsavedComparison = useCallback((draft: StoredSetDraft): StoredSetDraft => {
    const normalizedDraft = normalizeStoredDraft(draft);
    const normalizedReelInputs = normalizedDraft.reelInputs.map((row) => {
      const normalizedValue = row.value;
      const sourceUrl = normalizedValue.trim();
      const durationState = Object.values(reelDurationByRow).find((entry) => entry.sourceUrl === sourceUrl);
      const durationSec = durationState?.durationSec;
      const hasDetectedDuration = Number.isFinite(durationSec) && Number(durationSec) > CLIP_SLIDER_MIN_GAP_SEC;
      const maxSec = hasDetectedDuration ? Math.max(CLIP_SLIDER_MIN_GAP_SEC * 2, Number(durationSec)) : null;
      let nextStart = parseClipSecondsInput(row.tStartSec) ?? 0;
      let nextEnd = parseClipSecondsInput(row.tEndSec);
      if (maxSec !== null) {
        nextStart = Math.min(Math.max(0, nextStart), maxSec - CLIP_SLIDER_MIN_GAP_SEC);
        if (nextEnd !== null) {
          nextEnd = Math.min(maxSec, Math.max(nextStart + CLIP_SLIDER_MIN_GAP_SEC, nextEnd));
        }
      }
      return {
        ...(typeof row.communityReelId === "string" && row.communityReelId.trim()
          ? { communityReelId: row.communityReelId.trim() }
          : {}),
        value: normalizedValue,
        tStartSec: formatClipSecondsInputValue(nextStart),
        tEndSec: nextEnd === null ? "" : formatClipSecondsInputValue(nextEnd),
      };
    });
    return {
      ...normalizedDraft,
      reelInputs: normalizedReelInputs,
    };
  }, [normalizeStoredDraft, reelDurationByRow]);

  const applyDraftToForm = useCallback((draft: StoredSetDraft, contextKey?: string) => {
    const normalizedDraft = normalizeStoredDraft(draft);
    setSetTitle(normalizedDraft.title);
    setSetDescription(normalizedDraft.description);
    setSetTags(normalizedDraft.tags);
    setThumbnailPreview(normalizedDraft.thumbnailPreview);
    setThumbnailFileName(normalizedDraft.thumbnailFileName);
    const nextRows = normalizedDraft.reelInputs.length > 0
      ? normalizedDraft.reelInputs.map((row) => createDraftReelRow(
        row.value,
        row.tStartSec,
        row.tEndSec,
        row.communityReelId,
      ))
      : [createDraftReelRow()];
    setReelInputs(nextRows);
    setReelDurationByRow({});
    reelDurationCacheRef.current = {};
    setCreateError(null);
    setCreateSuccess(null);
    if (contextKey) {
      setDraftBaselineForContext(contextKey, normalizedDraft);
    }
  }, [normalizeStoredDraft, setDraftBaselineForContext]);

  const applySetToForm = useCallback((set: CommunitySet) => {
    const nextDraft = normalizeStoredDraft({
      title: set.title,
      description: set.description,
      tags: set.tags.join(", "),
      thumbnailPreview: set.thumbnailUrl || "",
      thumbnailFileName: set.thumbnailUrl ? "Current thumbnail" : "",
      reelInputs: draftRowsFromReels(set.reels).map((row) => ({
        ...(typeof row.communityReelId === "string" && row.communityReelId.trim()
          ? { communityReelId: row.communityReelId.trim() }
          : {}),
        value: row.value,
        tStartSec: row.tStartSec,
        tEndSec: row.tEndSec,
      })),
    });
    setSetTitle(nextDraft.title);
    setSetDescription(nextDraft.description);
    setSetTags(nextDraft.tags);
    setThumbnailPreview(nextDraft.thumbnailPreview);
    setThumbnailFileName(nextDraft.thumbnailFileName);
    setReelInputs(nextDraft.reelInputs.map((row) => createDraftReelRow(
      row.value,
      row.tStartSec,
      row.tEndSec,
      row.communityReelId,
    )));
    setReelDurationByRow({});
    reelDurationCacheRef.current = {};
    setCreateError(null);
    setCreateSuccess(null);
    setDraftBaselineForContext(`edit:${set.id}`, nextDraft);
  }, [normalizeStoredDraft, setDraftBaselineForContext]);

  const clearCreateSetDraftProgress = useCallback(() => {
    if (typeof window === "undefined") {
      return;
    }
    window.localStorage.removeItem(createDraftStorageKey);
  }, [createDraftStorageKey]);

  const resetCreateSetForm = useCallback(() => {
    const nextDraft = normalizeStoredDraft({
      title: "",
      description: "",
      tags: "",
      thumbnailPreview: "",
      thumbnailFileName: "",
      reelInputs: [{ value: "", tStartSec: "0", tEndSec: "" }],
    });
    setSetTitle(nextDraft.title);
    setSetDescription(nextDraft.description);
    setSetTags(nextDraft.tags);
    setThumbnailPreview(nextDraft.thumbnailPreview);
    setThumbnailFileName(nextDraft.thumbnailFileName);
    setReelInputs(nextDraft.reelInputs.map((row) => createDraftReelRow(
      row.value,
      row.tStartSec,
      row.tEndSec,
      row.communityReelId,
    )));
    setReelDurationByRow({});
    reelDurationCacheRef.current = {};
    setCreateError(null);
    setCreateSuccess(null);
    setDraftBaselineForContext(COMMUNITY_CREATE_DRAFT_CONTEXT_KEY, nextDraft);
  }, [normalizeStoredDraft, setDraftBaselineForContext]);

  const currentDraftPayload = useMemo(() => buildCurrentDraftPayload(), [buildCurrentDraftPayload]);
  const hasUnsavedDraftChanges = useMemo(() => {
    if (!formDraftContextKey || !shouldShowEditSetForm) {
      return false;
    }
    const baseline = draftBaselinesByContextRef.current[formDraftContextKey];
    if (typeof baseline !== "string") {
      return false;
    }
    const baselineDraft = parseStoredSetDraft(baseline);
    if (!baselineDraft) {
      return false;
    }
    const baselineSerialized = JSON.stringify(buildCanonicalDraftForUnsavedComparison(baselineDraft));
    const currentSerialized = JSON.stringify(buildCanonicalDraftForUnsavedComparison(currentDraftPayload));
    return baselineSerialized !== currentSerialized;
  }, [buildCanonicalDraftForUnsavedComparison, currentDraftPayload, draftBaselineVersion, formDraftContextKey, shouldShowEditSetForm]);

  useEffect(() => {
    if (mode !== "create" || typeof window === "undefined") {
      return;
    }
    const storedDraft = parseStoredSetDraft(window.localStorage.getItem(createDraftStorageKey));
    if (storedDraft) {
      applyDraftToForm(storedDraft, COMMUNITY_CREATE_DRAFT_CONTEXT_KEY);
      return;
    }
    resetCreateSetForm();
  }, [applyDraftToForm, createDraftStorageKey, mode, resetCreateSetForm]);

  useEffect(() => {
    if (!requiresCommunityAuth || canManageYourSets) {
      return;
    }
    setIsEditSetEditorOpen(false);
    setIsCreateSetEditorOpen(false);
  }, [canManageYourSets, requiresCommunityAuth]);

  useEffect(() => {
    if (!isYourSetsMode) {
      loadedEditSetIdRef.current = null;
      setActiveEditSetId(null);
      return;
    }
    setActiveEditSetId((prev) => {
      if (prev && editableSets.some((set) => set.id === prev)) {
        return prev;
      }
      return null;
    });
  }, [editableSets, isYourSetsMode]);

  useEffect(() => {
    if (!isYourSetsMode) {
      return;
    }
    if (isEditSetEditorOpen && !activeEditableSet) {
      setIsEditSetEditorOpen(false);
    }
  }, [activeEditableSet, isYourSetsMode, isEditSetEditorOpen]);

  useEffect(() => {
    if (!isYourSetsMode || !isEditSetEditorOpen || typeof window === "undefined" || !activeEditSetId) {
      return;
    }
    if (loadedEditSetIdRef.current === activeEditSetId) {
      return;
    }
    const selectedSet = editableSets.find((set) => set.id === activeEditSetId);
    if (!selectedSet) {
      return;
    }
    const storedDraft = parseStoredSetDraft(window.localStorage.getItem(`${editDraftStoragePrefix}${activeEditSetId}`));
    if (storedDraft) {
      applyDraftToForm(storedDraft, `edit:${activeEditSetId}`);
    } else {
      applySetToForm(selectedSet);
    }
    loadedEditSetIdRef.current = activeEditSetId;
  }, [activeEditSetId, applyDraftToForm, applySetToForm, editDraftStoragePrefix, editableSets, isEditSetEditorOpen, isYourSetsMode]);

  const parsedDraftReels = useMemo<ParsedDraftReel[]>(
    () =>
      reelInputs.map((row) => {
        const trimmed = row.value.trim();
        const clipStartSec = parseClipSecondsInput(row.tStartSec);
        const clipEndSec = parseClipSecondsInput(row.tEndSec);
        const hasClipRangeError =
          clipStartSec === null || (clipEndSec !== null && clipEndSec <= clipStartSec);
        if (!trimmed) {
          return {
            id: row.id,
            communityReelId: row.communityReelId,
            value: row.value,
            tStartSec: row.tStartSec,
            tEndSec: row.tEndSec,
            clipStartSec,
            clipEndSec,
            hasClipRangeError,
            parsed: null,
          };
        }
        return {
          id: row.id,
          communityReelId: row.communityReelId,
          value: row.value,
          tStartSec: row.tStartSec,
          tEndSec: row.tEndSec,
          clipStartSec,
          clipEndSec,
          hasClipRangeError,
          parsed: parseReelUrl(trimmed),
        };
      }),
    [reelInputs],
  );

  useEffect(() => {
    if (mode !== "create" && mode !== "edit") {
      return;
    }
    let cancelled = false;
    const activeRows = parsedDraftReels.filter((row) => row.parsed && row.value.trim());

    setReelDurationByRow((prev) => {
      const next: Record<string, ReelDurationState> = {};
      for (const row of activeRows) {
        const sourceUrl = row.parsed!.sourceUrl;
        const existing = prev[row.id];
        if (existing && existing.sourceUrl === sourceUrl) {
          next[row.id] = existing;
          continue;
        }
        if (Object.prototype.hasOwnProperty.call(reelDurationCacheRef.current, sourceUrl)) {
          next[row.id] = {
            sourceUrl,
            durationSec: reelDurationCacheRef.current[sourceUrl],
            loading: false,
          };
          continue;
        }
        next[row.id] = {
          sourceUrl,
          durationSec: null,
          loading: true,
        };
      }
      return next;
    });

    for (const row of activeRows) {
      const sourceUrl = row.parsed!.sourceUrl;
      if (Object.prototype.hasOwnProperty.call(reelDurationCacheRef.current, sourceUrl)) {
        continue;
      }
      void (async () => {
        let durationSec: number | null = null;
        if (!localDemoEnabled) {
          try {
            durationSec = await fetchCommunityReelDuration({ sourceUrl });
          } catch {
            durationSec = null;
          }
        }
        if (row.parsed?.platform === "youtube") {
          const iframeDuration = await detectYouTubeDurationWithIframeApi(sourceUrl);
          if (iframeDuration !== null) {
            durationSec = durationSec === null ? iframeDuration : Math.max(durationSec, iframeDuration);
          }
        }
        reelDurationCacheRef.current[sourceUrl] = durationSec;
        if (cancelled) {
          return;
        }
        setReelDurationByRow((prev) => {
          const existing = prev[row.id];
          if (!existing || existing.sourceUrl !== sourceUrl) {
            return prev;
          }
          if (!existing.loading && existing.durationSec === durationSec) {
            return prev;
          }
          return {
            ...prev,
            [row.id]: {
              sourceUrl,
              durationSec,
              loading: false,
            },
          };
        });
      })();
    }

    return () => {
      cancelled = true;
    };
  }, [localDemoEnabled, mode, parsedDraftReels]);

  useEffect(() => {
    if (mode !== "create" && mode !== "edit") {
      return;
    }
    setReelInputs((prev) => {
      let changed = false;
      const next = prev.map((row) => {
        const durationSec = reelDurationByRow[row.id]?.durationSec;
        if (!Number.isFinite(durationSec) || Number(durationSec) <= CLIP_SLIDER_MIN_GAP_SEC) {
          return row;
        }
        const sliderMaxSec = Math.max(CLIP_SLIDER_MIN_GAP_SEC * 2, Number(durationSec));
        const currentStart = parseClipSecondsInput(row.tStartSec) ?? 0;
        const parsedEnd = parseClipSecondsInput(row.tEndSec);
        const hasExplicitEnd = parsedEnd !== null;
        const nextStart = Math.min(Math.max(0, currentStart), sliderMaxSec - CLIP_SLIDER_MIN_GAP_SEC);
        const formattedStart = formatClipSecondsInputValue(nextStart);
        const formattedEnd = hasExplicitEnd
          ? formatClipSecondsInputValue(
              Math.min(sliderMaxSec, Math.max(nextStart + CLIP_SLIDER_MIN_GAP_SEC, parsedEnd)),
            )
          : "";
        if (formattedStart === row.tStartSec && formattedEnd === row.tEndSec) {
          return row;
        }
        changed = true;
        return {
          ...row,
          tStartSec: formattedStart,
          tEndSec: formattedEnd,
        };
      });
      return changed ? next : prev;
    });
  }, [mode, reelDurationByRow]);

  const validDraftReelCount = useMemo(
    () =>
      parsedDraftReels.filter(
        (row) => row.value.trim() && row.parsed !== null && !row.hasClipRangeError,
      ).length,
    [parsedDraftReels],
  );
  const nonEmptyDraftReelCount = useMemo(
    () => parsedDraftReels.filter((row) => row.value.trim()).length,
    [parsedDraftReels],
  );
  const invalidDraftReelCount = Math.max(0, nonEmptyDraftReelCount - validDraftReelCount);
  const normalizedSetTitle = setTitle.trim();
  const normalizedSetDescription = setDescription.trim();
  const descriptionCharsRemaining = Math.max(0, MIN_SET_DESCRIPTION_LENGTH - normalizedSetDescription.length);
  const descriptionHasTooFewChars = descriptionCharsRemaining > 0;
  const shouldShowDescriptionError = Boolean(normalizedSetDescription) && descriptionHasTooFewChars;
  const parsedSetTags = useMemo(() => parseTags(setTags), [setTags]);
  const hasMaxTags = parsedSetTags.length >= MAX_SET_TAGS;
  const requiredCompletionCount =
    (normalizedSetTitle ? 1 : 0) +
    (normalizedSetDescription.length >= MIN_SET_DESCRIPTION_LENGTH ? 1 : 0) +
    (thumbnailPreview ? 1 : 0) +
    (validDraftReelCount > 0 && invalidDraftReelCount === 0 ? 1 : 0);
  const canPostSet = requiredCompletionCount === 4 && !isPostingSet && (!isFormEditMode || Boolean(activeEditableSet));
  const remainingPreviewRequirements = useMemo(() => {
    const items: string[] = [];
    if (!normalizedSetTitle) {
      items.push("Add a set name");
    }
    if (normalizedSetDescription.length < MIN_SET_DESCRIPTION_LENGTH) {
      items.push(`Write a description (${MIN_SET_DESCRIPTION_LENGTH}+ characters)`);
    }
    if (!thumbnailPreview) {
      items.push("Upload a thumbnail image");
    }
    if (nonEmptyDraftReelCount === 0) {
      items.push("Add at least one reel link");
    } else if (invalidDraftReelCount > 0) {
      items.push("Fix invalid reel links or clip ranges");
    }
    return items;
  }, [
    invalidDraftReelCount,
    nonEmptyDraftReelCount,
    normalizedSetDescription.length,
    normalizedSetTitle,
    thumbnailPreview,
    validDraftReelCount,
  ]);

  const onSetTagsChange = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    const nextValue = event.target.value;
    const allNextTags = parseAllTags(nextValue);
    if (allNextTags.length > MAX_SET_TAGS) {
      setTagLimitError(true);
      return;
    }
    setTagLimitError(false);
    setSetTags(nextValue);
  }, []);

  const onRemoveSetTag = useCallback((tagToRemove: string) => {
    const remainingTags = parseTags(setTags).filter((tag) => tag !== tagToRemove);
    setTagLimitError(false);
    setSetTags(remainingTags.join(", "));
  }, [setTags]);

  const applyThumbnailFile = useCallback((file: File | null | undefined) => {
    if (!file) {
      return;
    }
    if (!file.type.startsWith("image/")) {
      setCreateSuccess(null);
      setPublishResultModal({
        status: "error",
        label: isFormEditMode ? "Save Set Changes" : "Post Community Set",
        title: "Invalid Thumbnail",
        message: "Thumbnail must be an image file.",
      });
      return;
    }
    if (file.size > MAX_THUMBNAIL_FILE_BYTES) {
      setCreateSuccess(null);
      setPublishResultModal({
        status: "error",
        label: isFormEditMode ? "Save Set Changes" : "Post Community Set",
        title: "Thumbnail Too Large",
        message: "Thumbnail image must be 1.5 MB or smaller.",
      });
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === "string") {
        setThumbnailPreview(reader.result);
        setThumbnailFileName(file.name);
        setCreateError(null);
      }
    };
    reader.readAsDataURL(file);
  }, [isFormEditMode]);

  const onThumbnailFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    applyThumbnailFile(event.target.files?.[0]);
    event.target.value = "";
  };

  const onThumbnailDragEnter = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setIsThumbnailDragOver(true);
  };

  const onThumbnailDragOver = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    event.stopPropagation();
    event.dataTransfer.dropEffect = "copy";
    setIsThumbnailDragOver(true);
  };

  const onThumbnailDragLeave = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    event.stopPropagation();
    const nextTarget = event.relatedTarget as Node | null;
    if (nextTarget && event.currentTarget.contains(nextTarget)) {
      return;
    }
    setIsThumbnailDragOver(false);
  };

  const onThumbnailDrop = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setIsThumbnailDragOver(false);
    applyThumbnailFile(event.dataTransfer.files?.[0]);
  };

  const submitSet = useCallback(async (): Promise<boolean> => {
    const actionLabel = isFormEditMode ? "Save Set Changes" : "Post Community Set";
    const title = setTitle.trim();
    const description = setDescription.trim();
    const editSetId = activeEditableSet?.id?.trim() || activeEditSetId?.trim() || "";
    if (isFormEditMode && !editSetId) {
      setCreateSuccess(null);
      setPublishResultModal({
        status: "error",
        label: actionLabel,
        title: "No Set Selected",
        message: "Pick a set to edit first.",
      });
      return false;
    }
    setCreateSuccess(null);
    if (!title) {
      setPublishResultModal({
        status: "error",
        label: actionLabel,
        title: "Missing Set Name",
        message: "Set name is required.",
      });
      return false;
    }
    if (description.length < MIN_SET_DESCRIPTION_LENGTH) {
      setPublishResultModal({
        status: "error",
        label: actionLabel,
        title: "Description Too Short",
        message: `Description must be at least ${MIN_SET_DESCRIPTION_LENGTH} characters.`,
      });
      return false;
    }
    if (!thumbnailPreview) {
      setPublishResultModal({
        status: "error",
        label: actionLabel,
        title: "Missing Thumbnail",
        message: "Add a thumbnail image before posting.",
      });
      return false;
    }

    const nonEmptyRows = parsedDraftReels.filter((row) => row.value.trim());
    if (nonEmptyRows.length === 0) {
      setPublishResultModal({
        status: "error",
        label: actionLabel,
        title: "Missing Reels",
        message: "Add at least one reel URL to post this set.",
      });
      return false;
    }
    const firstInvalid = nonEmptyRows.find(
      (row) => row.parsed === null || row.hasClipRangeError,
    );
    if (firstInvalid) {
      if (firstInvalid.parsed === null) {
        setPublishResultModal({
          status: "error",
          label: actionLabel,
          title: "Invalid Reel Link",
          message: `One or more reel links are invalid. Supported: ${SUPPORTED_PLATFORMS_LABEL}.`,
        });
      } else {
        setPublishResultModal({
          status: "error",
          label: actionLabel,
          title: "Invalid Clip Range",
          message: "Each reel needs a valid clip range (start >= 0 and optional end > start).",
        });
      }
      return false;
    }

    const parsedReels = nonEmptyRows.map((row) => {
      const parsed = row.parsed!;
      return {
        ...(typeof row.communityReelId === "string" && row.communityReelId.trim()
          ? { id: row.communityReelId.trim() }
          : {}),
        platform: parsed.platform,
        sourceUrl: parsed.sourceUrl,
        embedUrl: parsed.embedUrl,
        tStartSec: row.clipStartSec ?? undefined,
        tEndSec: row.clipEndSec ?? undefined,
      };
    });

    const tags = parseTags(setTags);
    setPublishResultModal(null);
    setIsPostingSet(true);
    try {
      if (isFormEditMode && editSetId) {
        const updatedSet = localDemoEnabled
          ? buildLocalDemoSet({
              existingSet: activeEditableSet,
              setId: editSetId,
              title,
              description,
              tags,
              reels: parsedReels,
              thumbnailUrl: thumbnailPreview,
            })
          : await updateCommunitySet({
              setId: editSetId,
              title,
              description,
              tags,
              reels: parsedReels,
              thumbnailUrl: thumbnailPreview,
              curator: communityAccount?.username || activeEditableSet?.curator || "Community member",
            });
        setOwnedSets((prev) => [updatedSet, ...prev.filter((item) => item.id !== updatedSet.id)].slice(0, MAX_USER_SETS));
        if (typeof window !== "undefined") {
          window.localStorage.removeItem(`${editDraftStoragePrefix}${updatedSet.id}`);
        }
        setDraftBaselineForContext(`edit:${updatedSet.id}`, buildCurrentDraftPayload());
        setCreateError(null);
        setCreateSuccess(`Saved changes to "${updatedSet.title}".`);
        return true;
      } else {
        const createdSet = localDemoEnabled
          ? buildLocalDemoSet({
              title,
              description,
              tags,
              reels: parsedReels,
              thumbnailUrl: thumbnailPreview,
            })
          : await createCommunitySet({
              title,
              description,
              tags,
              reels: parsedReels,
              thumbnailUrl: thumbnailPreview,
              curator: communityAccount?.username || "Community member",
            });
        setOwnedSetIds((prev) => (prev.includes(createdSet.id) ? prev : [createdSet.id, ...prev]));
        setOwnedSets((prev) => [createdSet, ...prev.filter((item) => item.id !== createdSet.id)].slice(0, MAX_USER_SETS));
        clearCreateSetDraftProgress();
        resetCreateSetForm();
        setPublishResultModal({
          status: "success",
          title: "Saved to Your Sets",
          message: `"${createdSet.title}" was saved with ${createdSet.reels.length} reel${createdSet.reels.length === 1 ? "" : "s"}.`,
          thumbnailUrl: createdSet.thumbnailUrl || thumbnailPreview || undefined,
          thumbnailAlt: `${createdSet.title} thumbnail`,
        });
        return true;
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : isFormEditMode ? "Could not update community set." : "Could not post community set.";
      if (!localDemoEnabled && isSessionExpiredError(error)) {
        expireCommunityAuthSession();
        setCommunityAccount(null);
        ownedSetsAccountIdRef.current = null;
        clearOwnedCommunityState();
      }
      setCreateError(null);
      setPublishResultModal({
        status: "error",
        label: actionLabel,
        title: isFormEditMode ? "Save Failed" : "Publish Failed",
        message,
      });
      return false;
    } finally {
      setIsPostingSet(false);
    }
  }, [
    activeEditSetId,
    activeEditableSet,
    buildCurrentDraftPayload,
    clearCreateSetDraftProgress,
    communityAccount,
    editDraftStoragePrefix,
    isFormEditMode,
    localDemoEnabled,
    parsedDraftReels,
    resetCreateSetForm,
    setDraftBaselineForContext,
    setTags,
    setDescription,
    setTitle,
    thumbnailPreview,
    clearOwnedCommunityState,
    setOwnedSets,
  ]);

  const onCreateSet = useCallback((event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (isFormEditMode) {
      const targetTitle = activeEditableSet?.title?.trim() || "this set";
      setDraftActionConfirmModal({
        action: "save-set-changes",
        label: "Update Set",
        title: `Save changes to "${targetTitle}"?`,
        message: "Your edits will replace the currently published version of this set.",
        confirmLabel: "Save Set Changes",
      });
      return;
    }
    void submitSet();
  }, [activeEditableSet, isFormEditMode, submitSet]);

  const persistDraftPayload = useCallback((storageKey: string, draftPayload: StoredSetDraft): { ok: boolean; warning?: string; error?: string } => {
    if (typeof window === "undefined") {
      return { ok: false, error: "Draft storage is unavailable in this environment." };
    }
    if (safeStorageSetItem(window.localStorage, storageKey, JSON.stringify(draftPayload))) {
      return { ok: true };
    }
    // If storage quota is tight (large thumbnail previews), retry with a compact draft.
    const compactDraft: StoredSetDraft = {
      ...draftPayload,
      thumbnailPreview: "",
      thumbnailFileName: draftPayload.thumbnailFileName || "",
    };
    if (safeStorageSetItem(window.localStorage, storageKey, JSON.stringify(compactDraft))) {
      return {
        ok: true,
        warning: "Draft saved without thumbnail preview due to browser storage limits.",
      };
    }
    return { ok: false, error: "Could not save draft progress." };
  }, []);

  const saveCurrentDraftProgress = useCallback((options?: { showSuccessMessage?: boolean }): boolean => {
    if (typeof window === "undefined") {
      return false;
    }
    const showSuccessMessage = options?.showSuccessMessage ?? true;
    const draftPayload = buildCurrentDraftPayload();
    if (isFormEditMode) {
      const editSetId = activeEditableSet?.id?.trim() || activeEditSetId?.trim() || "";
      if (!editSetId) {
        setCreateSuccess(null);
        setPublishResultModal({
          status: "error",
          label: "Save Progress",
          title: "Save Failed",
          message: "Pick a set to edit before saving draft progress.",
        });
        return false;
      }
      const persistResult = persistDraftPayload(`${editDraftStoragePrefix}${editSetId}`, draftPayload);
      if (!persistResult.ok) {
        setCreateSuccess(null);
        setPublishResultModal({
          status: "error",
          label: "Save Progress",
          title: "Save Failed",
          message: persistResult.error || "Could not save draft progress.",
        });
        return false;
      }
      setDraftBaselineForContext(`edit:${editSetId}`, draftPayload);
      setCreateError(null);
      if (showSuccessMessage) {
        setCreateSuccess(persistResult.warning || "Draft progress saved for this set.");
      } else {
        setCreateSuccess(null);
      }
      return true;
    }
    const persistResult = persistDraftPayload(createDraftStorageKey, draftPayload);
    if (!persistResult.ok) {
      setCreateSuccess(null);
      setPublishResultModal({
        status: "error",
        label: "Save Progress",
        title: "Save Failed",
        message: persistResult.error || "Could not save draft progress.",
      });
      return false;
    }
    setDraftBaselineForContext(COMMUNITY_CREATE_DRAFT_CONTEXT_KEY, draftPayload);
    setCreateError(null);
    if (showSuccessMessage) {
      setCreateSuccess(persistResult.warning || "Draft progress saved.");
    } else {
      setCreateSuccess(null);
    }
    return true;
  }, [activeEditSetId, activeEditableSet, buildCurrentDraftPayload, createDraftStorageKey, editDraftStoragePrefix, isFormEditMode, persistDraftPayload, setDraftBaselineForContext]);

  const discardCurrentDraftChanges = useCallback(() => {
    if (!formDraftContextKey) {
      return;
    }
    const baselineRaw = draftBaselinesByContextRef.current[formDraftContextKey];
    const baselineDraft = parseStoredSetDraft(baselineRaw ?? null);
    if (baselineDraft) {
      applyDraftToForm(baselineDraft, formDraftContextKey);
      return;
    }
    if (formDraftContextKey === COMMUNITY_CREATE_DRAFT_CONTEXT_KEY) {
      resetCreateSetForm();
      return;
    }
    if (activeEditableSet) {
      applySetToForm(activeEditableSet);
      return;
    }
    setCreateError(null);
    setCreateSuccess(null);
  }, [activeEditableSet, applyDraftToForm, applySetToForm, formDraftContextKey, resetCreateSetForm]);

  const clearCreateSetProgress = useCallback(() => {
    try {
      if (typeof window !== "undefined") {
        window.localStorage.removeItem(createDraftStorageKey);
      }
      resetCreateSetForm();
      setCreateSuccess("Create set progress cleared.");
      return true;
    } catch (error) {
      const message = error instanceof Error ? error.message : "Could not clear create set progress.";
      setPublishResultModal({
        status: "error",
        label: "Clear Progress",
        title: "Clear Failed",
        message,
      });
      return false;
    }
  }, [createDraftStorageKey, resetCreateSetForm]);

  const onSaveDraftProgress = useCallback(() => {
    setDraftActionConfirmModal({
      action: "save-progress",
      label: "Save Draft",
      title: "Save your create-set draft?",
      message: "Your current fields and reel links will be stored so you can continue later.",
      confirmLabel: "Save Progress",
    });
  }, []);

  const onClearCreateProgress = useCallback(() => {
    setDraftActionConfirmModal({
      action: "clear-progress",
      label: "Clear Draft",
      title: "Clear all create-set progress?",
      message: "This resets the current create form and removes the saved draft from local storage.",
      confirmLabel: "Clear Progress",
    });
  }, []);

  const closeDraftActionConfirmModal = useCallback(() => {
    if (isPostingSet) {
      return;
    }
    setDraftActionConfirmModal(null);
  }, [isPostingSet]);

  const confirmDraftAction = useCallback(() => {
    if (!draftActionConfirmModal) {
      return;
    }
    const action = draftActionConfirmModal.action;
    setDraftActionConfirmModal(null);
    if (action === "save-progress") {
      saveCurrentDraftProgress({ showSuccessMessage: true });
      return;
    }
    if (action === "clear-progress") {
      clearCreateSetProgress();
      return;
    }
    void submitSet();
  }, [clearCreateSetProgress, draftActionConfirmModal, saveCurrentDraftProgress, submitSet]);

  const addReelInputRow = () => {
    setReelInputs((prev) => [...prev, createDraftReelRow()]);
  };

  const onOpenCreateSetFromGrid = useCallback(() => {
    if (!canManageYourSets) {
      return;
    }
    setIsEditSetEditorOpen(false);
    setIsCreateSetEditorOpen(true);
    setActiveEditSetId(null);
    loadedEditSetIdRef.current = null;
    if (typeof window !== "undefined") {
      const storedDraft = parseStoredSetDraft(window.localStorage.getItem(createDraftStorageKey));
      if (storedDraft) {
        applyDraftToForm(storedDraft, COMMUNITY_CREATE_DRAFT_CONTEXT_KEY);
      } else {
        resetCreateSetForm();
      }
    } else {
      resetCreateSetForm();
    }
    setCreateError(null);
    setCreateSuccess(null);
  }, [applyDraftToForm, canManageYourSets, createDraftStorageKey, resetCreateSetForm]);

  useEffect(() => {
    onDraftUnsavedChangesChange?.(hasUnsavedDraftChanges);
  }, [hasUnsavedDraftChanges, onDraftUnsavedChangesChange]);

  useEffect(() => {
    if (!onDraftExitActionsChange) {
      return;
    }
    if (!shouldShowEditSetForm) {
      onDraftExitActionsChange(null);
      return;
    }
    onDraftExitActionsChange({
      saveDraftProgress: () => saveCurrentDraftProgress({ showSuccessMessage: false }),
      discardDraftChanges: discardCurrentDraftChanges,
    });
    return () => {
      onDraftExitActionsChange(null);
    };
  }, [discardCurrentDraftChanges, onDraftExitActionsChange, saveCurrentDraftProgress, shouldShowEditSetForm]);

  const onOpenEditableSet = useCallback((setId: string) => {
    const normalized = setId.trim();
    if (!normalized) {
      return;
    }
    loadedEditSetIdRef.current = null;
    setActiveEditSetId(normalized);
    setIsEditSetEditorOpen(true);
    setIsCreateSetEditorOpen(false);
    setCreateError(null);
    setCreateSuccess(null);
  }, []);

  const backToEditSetGrid = useCallback(() => {
    setIsEditSetEditorOpen(false);
    setIsCreateSetEditorOpen(false);
    setCreateError(null);
    setCreateSuccess(null);
  }, []);

  const onBackToEditSetGrid = useCallback(() => {
    if (hasUnsavedDraftChanges) {
      setUnsavedDraftExitModal({ action: "back-to-grid" });
      return;
    }
    backToEditSetGrid();
  }, [backToEditSetGrid, hasUnsavedDraftChanges]);

  const closeUnsavedDraftExitModal = useCallback(() => {
    setUnsavedDraftExitModal(null);
  }, []);

  const confirmUnsavedDraftExitSave = useCallback(async () => {
    if (!unsavedDraftExitModal) {
      return;
    }
    const pendingAction = unsavedDraftExitModal.action;
    setUnsavedDraftExitModal(null);
    const didSave = isFormEditMode
      ? await submitSet()
      : saveCurrentDraftProgress({ showSuccessMessage: false });
    if (!didSave) {
      return;
    }
    if (pendingAction === "back-to-grid") {
      backToEditSetGrid();
    }
  }, [backToEditSetGrid, isFormEditMode, saveCurrentDraftProgress, submitSet, unsavedDraftExitModal]);

  const confirmUnsavedDraftExitDiscard = useCallback(() => {
    if (!unsavedDraftExitModal) {
      return;
    }
    const pendingAction = unsavedDraftExitModal.action;
    discardCurrentDraftChanges();
    setUnsavedDraftExitModal(null);
    if (pendingAction === "back-to-grid") {
      backToEditSetGrid();
    }
  }, [backToEditSetGrid, discardCurrentDraftChanges, unsavedDraftExitModal]);

  useEffect(() => {
    if (shouldShowEditSetForm || !unsavedDraftExitModal) {
      return;
    }
    setUnsavedDraftExitModal(null);
  }, [shouldShowEditSetForm, unsavedDraftExitModal]);

  useEffect(() => {
    if (shouldShowEditSetForm || !draftActionConfirmModal) {
      return;
    }
    setDraftActionConfirmModal(null);
  }, [draftActionConfirmModal, shouldShowEditSetForm]);

  const onToggleEditableSetSelection = useCallback((setId: string) => {
    const normalized = setId.trim();
    if (!normalized || isDeletingSets) {
      return;
    }
    setSelectedEditableSetIds((prev) => (
      prev.includes(normalized)
        ? prev.filter((id) => id !== normalized)
        : [...prev, normalized]
    ));
    setActiveSetActionsMenuId(null);
  }, [isDeletingSets]);

  const onToggleSetStar = useCallback((setId: string) => {
    const normalized = setId.trim();
    if (!normalized) {
      return;
    }
    setStarredSetIds((prev) => {
      if (prev.includes(normalized)) {
        return prev.filter((id) => id !== normalized);
      }
      return [normalized, ...prev];
    });
    setActiveSetActionsMenuId(null);
  }, []);

  const onRequestDeleteEditableSet = useCallback((setId: string) => {
    const normalized = setId.trim();
    if (!normalized || isDeletingSets) {
      return;
    }
    const target = editableSets.find((set) => set.id === normalized);
    const targetTitle = target?.title?.trim() || "this set";
    setActiveSetActionsMenuId(null);
    setDeleteSetConfirmModal({ setIds: [normalized], title: targetTitle });
  }, [editableSets, isDeletingSets]);

  const onRequestDeleteSelectedEditableSets = useCallback(() => {
    if (selectedEditableSetIds.length === 0 || isDeletingSets) {
      return;
    }
    const editableSetIds = new Set(editableSets.map((set) => set.id));
    const targetIds = selectedEditableSetIds.filter((setId) => editableSetIds.has(setId));
    if (targetIds.length === 0) {
      setSelectedEditableSetIds([]);
      return;
    }
    const singleTarget = targetIds.length === 1
      ? editableSets.find((set) => set.id === targetIds[0])
      : null;
    setActiveSetActionsMenuId(null);
    setDeleteSetConfirmModal({
      setIds: targetIds,
      title: singleTarget?.title?.trim() || `${targetIds.length} selected sets`,
    });
  }, [editableSets, isDeletingSets, selectedEditableSetIds]);

  const closeDeleteSetConfirmModal = useCallback(() => {
    if (isDeletingSets) {
      return;
    }
    setDeleteSetConfirmModal(null);
  }, [isDeletingSets]);

  const onDeleteEditableSets = useCallback(async (setIds: string[]) => {
    const normalizedSetIds = [...new Set(setIds.map((setId) => setId.trim()).filter(Boolean))]
      .filter((setId) => editableSets.some((set) => set.id === setId));
    if (normalizedSetIds.length === 0 || isDeletingSets) {
      return;
    }
    const deletedSetIdSet = new Set(normalizedSetIds);
    const singleTarget = normalizedSetIds.length === 1
      ? editableSets.find((set) => set.id === normalizedSetIds[0])
      : null;
    setDeleteSetConfirmModal(null);
    setPublishResultModal(null);
    setDeletingSetIds(normalizedSetIds);
    try {
      if (!localDemoEnabled) {
        if (normalizedSetIds.length === 1) {
          await deleteCommunitySet({ setId: normalizedSetIds[0] });
        } else {
          await deleteCommunitySets({ setIds: normalizedSetIds });
        }
      }
      setOwnedSets((prev) => prev.filter((set) => !deletedSetIdSet.has(set.id)));
      setOwnedSetIds((prev) => prev.filter((id) => !deletedSetIdSet.has(id)));
      setStarredSetIds((prev) => prev.filter((id) => !deletedSetIdSet.has(id)));
      setSelectedEditableSetIds((prev) => prev.filter((id) => !deletedSetIdSet.has(id)));
      if (typeof window !== "undefined") {
        for (const setId of normalizedSetIds) {
          window.localStorage.removeItem(`${editDraftStoragePrefix}${setId}`);
        }
      }
      if (activeEditSetId && deletedSetIdSet.has(activeEditSetId)) {
        loadedEditSetIdRef.current = null;
        setActiveEditSetId(null);
        setIsEditSetEditorOpen(false);
      }
      setCreateError(null);
      setCreateSuccess(
        normalizedSetIds.length === 1
          ? `Deleted "${singleTarget?.title || "this set"}".`
          : `Deleted ${normalizedSetIds.length} sets.`,
      );
    } catch (error) {
      const message = error instanceof Error
        ? error.message
        : normalizedSetIds.length === 1
          ? "Could not delete this set."
          : "Could not delete the selected sets.";
      if (!localDemoEnabled && isSessionExpiredError(error)) {
        expireCommunityAuthSession();
        setCommunityAccount(null);
        ownedSetsAccountIdRef.current = null;
        clearOwnedCommunityState();
      }
      setPublishResultModal({
        status: "error",
        label: "Your Sets",
        title: "Delete Failed",
        message,
      });
    } finally {
      setDeletingSetIds([]);
      setActiveSetActionsMenuId(null);
    }
  }, [activeEditSetId, clearOwnedCommunityState, editDraftStoragePrefix, editableSets, isDeletingSets, localDemoEnabled]);

  const removeReelInputRow = (rowId: string) => {
    setReelInputs((prev) => {
      const next = prev.filter((row) => row.id !== rowId);
      return next.length > 0 ? next : [createDraftReelRow()];
    });
    setReelDurationByRow((prev) => {
      if (!(rowId in prev)) {
        return prev;
      }
      const next = { ...prev };
      delete next[rowId];
      return next;
    });
  };

  const updateReelInputRow = (rowId: string, value: string) => {
    setReelInputs((prev) => prev.map((row) => (row.id === rowId ? { ...row, value } : row)));
  };

  const updateReelClipStartFromSlider = useCallback((rowId: string, value: number, sliderMaxSec: number) => {
    const normalizedMax = Math.max(CLIP_SLIDER_MIN_GAP_SEC * 2, sliderMaxSec);
    setReelInputs((prev) =>
      prev.map((row) => {
        if (row.id !== rowId) {
          return row;
        }
        const parsedEnd = parseClipSecondsInput(row.tEndSec);
        const hasExplicitEnd = parsedEnd !== null;
        const nextStart = Math.min(Math.max(0, value), normalizedMax - CLIP_SLIDER_MIN_GAP_SEC);
        const nextEnd = hasExplicitEnd
          ? Math.min(normalizedMax, Math.max(nextStart + CLIP_SLIDER_MIN_GAP_SEC, parsedEnd))
          : null;
        return {
          ...row,
          tStartSec: formatClipSecondsInputValue(nextStart),
          tEndSec: nextEnd === null ? "" : formatClipSecondsInputValue(nextEnd),
        };
      }),
    );
  }, []);

  const updateReelClipEndFromSlider = useCallback((rowId: string, value: number, sliderMaxSec: number) => {
    const normalizedMax = Math.max(CLIP_SLIDER_MIN_GAP_SEC * 2, sliderMaxSec);
    setReelInputs((prev) =>
      prev.map((row) => {
        if (row.id !== rowId) {
          return row;
        }
        const currentStart = parseClipSecondsInput(row.tStartSec) ?? 0;
        const nextStart = Math.min(Math.max(0, currentStart), normalizedMax - CLIP_SLIDER_MIN_GAP_SEC);
        const nextEnd = Math.min(normalizedMax, Math.max(nextStart + CLIP_SLIDER_MIN_GAP_SEC, value));
        return {
          ...row,
          tStartSec: formatClipSecondsInputValue(nextStart),
          tEndSec: formatClipSecondsInputValue(nextEnd),
        };
      }),
    );
  }, []);

  const closeDirectorySetModal = useCallback(() => {
    if (!selectedDirectorySet) {
      return;
    }
    const returnFocusTarget = detailReturnFocusRef.current;
    const closingSetId = selectedDirectorySet.id;
    clearDirectoryDetailCloseTimer();
    setIsDirectoryDetailOpen(false);
    directoryDetailCloseTimerRef.current = window.setTimeout(() => {
      setSelectedDirectorySet(null);
      directoryDetailCloseTimerRef.current = null;
      const fallbackTarget = document.querySelector<HTMLElement>(
        `[data-community-set-row][data-community-set-id="${CSS.escape(closingSetId)}"]`,
      );
      const focusTarget = returnFocusTarget?.isConnected ? returnFocusTarget : fallbackTarget;
      focusTarget?.focus({ preventScroll: true });
      detailReturnFocusRef.current = null;
    }, DIRECTORY_DETAIL_TRANSITION_MS);
  }, [clearDirectoryDetailCloseTimer, selectedDirectorySet]);

  useEffect(() => {
    if (mode === "community" && isVisible) {
      return;
    }
    clearDirectoryDetailCloseTimer();
    setSkipDetailTransitionOnce(false);
    setIsDirectoryDetailOpen(false);
    setSelectedDirectorySet(null);
    setSelectedDetailReelId(null);
    setDetailCarouselIndex(0);
    detailReturnFocusRef.current = null;
    consumedInitialSetIdRef.current = null;
    setIsInitialDetailRestorePending(false);
  }, [clearDirectoryDetailCloseTimer, isVisible, mode]);

  useEffect(() => {
    if (mode === "community" && isVisible && normalizedInitialOpenSetId) {
      setIsInitialDetailRestorePending(true);
      return;
    }
    setIsInitialDetailRestorePending(false);
  }, [isVisible, mode, normalizedInitialOpenSetId]);

  useEffect(() => {
    if (!normalizedInitialOpenSetId || mode !== "community" || !isVisible) {
      setIsInitialDetailRestorePending(false);
      return;
    }
    if (isDirectoryDetailOpen && selectedDirectorySet?.id === normalizedInitialOpenSetId) {
      setIsInitialDetailRestorePending(false);
      return;
    }
    if (!storageHydrated) {
      return;
    }
    const targetExists = allSets.some((set) => set.id === normalizedInitialOpenSetId);
    if (!targetExists) {
      setIsInitialDetailRestorePending(false);
    }
  }, [allSets, isDirectoryDetailOpen, isVisible, mode, normalizedInitialOpenSetId, selectedDirectorySet?.id, storageHydrated]);

  useEffect(() => {
    if (lastCommunityResetSignalRef.current === communityResetSignal) {
      return;
    }
    lastCommunityResetSignalRef.current = communityResetSignal;
    if (mode !== "community" || !isVisible) {
      return;
    }
    clearDirectoryDetailCloseTimer();
    setSkipDetailTransitionOnce(false);
    setIsDirectoryDetailOpen(false);
    setSelectedDirectorySet(null);
    setSelectedDetailReelId(null);
    setDetailCarouselIndex(0);
    detailReturnFocusRef.current = null;
    // Allow the follow-up initial-open effect to restore the targeted set detail
    // after returning from the feed back button.
    consumedInitialSetIdRef.current = null;
  }, [clearDirectoryDetailCloseTimer, communityResetSignal, initialOpenSetId, isVisible, mode]);

  const openDirectorySet = useCallback(
    (set: CommunitySet, options?: { immediate?: boolean; skipTransition?: boolean }) => {
      clearDirectoryDetailCloseTimer();
      const activeElement = document.activeElement;
      detailReturnFocusRef.current =
        activeElement instanceof HTMLElement && directoryViewRef.current?.contains(activeElement) ? activeElement : null;
      setSelectedDirectorySet(set);
      if (options?.skipTransition) {
        setSkipDetailTransitionOnce(true);
      }
      if (options?.immediate || typeof window === "undefined") {
        setIsDirectoryDetailOpen(true);
        return;
      }
      window.requestAnimationFrame(() => {
        setIsDirectoryDetailOpen(true);
      });
    },
    [clearDirectoryDetailCloseTimer],
  );

  useLayoutEffect(() => {
    if (mode !== "community" || !isVisible) {
      return;
    }
    const targetSetId = initialOpenSetId?.trim() || "";
    if (!targetSetId || consumedInitialSetIdRef.current === targetSetId) {
      return;
    }
    if (typeof window === "undefined") {
      return;
    }
    const snapshot = parseStoredSetSnapshot(window.sessionStorage.getItem(`${COMMUNITY_SET_RETURN_SNAPSHOT_PREFIX}${targetSetId}`));
    if (!snapshot) {
      return;
    }
    setSkipDetailTransitionOnce(true);
    clearDirectoryDetailCloseTimer();
    setSelectedDirectorySet(snapshot);
    setIsDirectoryDetailOpen(true);
    setIsInitialDetailRestorePending(false);
    consumedInitialSetIdRef.current = targetSetId;
  }, [clearDirectoryDetailCloseTimer, initialOpenSetId, isVisible, mode]);

  useLayoutEffect(() => {
    if (mode !== "community" || !isVisible) {
      return;
    }
    const targetSetId = initialOpenSetId?.trim() || "";
    if (!targetSetId) {
      consumedInitialSetIdRef.current = null;
      return;
    }
    if (consumedInitialSetIdRef.current === targetSetId) {
      return;
    }
    const targetSet = allSets.find((set) => set.id === targetSetId);
    if (!targetSet) {
      return;
    }
    openDirectorySet(targetSet, { immediate: true, skipTransition: true });
    setIsInitialDetailRestorePending(false);
    consumedInitialSetIdRef.current = targetSetId;
  }, [allSets, initialOpenSetId, isVisible, mode, openDirectorySet]);

  useEffect(() => {
    if (!skipDetailTransitionOnce || !isDirectoryDetailOpen) {
      return;
    }
    if (typeof window === "undefined") {
      setSkipDetailTransitionOnce(false);
      return;
    }
    const rafId = window.requestAnimationFrame(() => {
      setSkipDetailTransitionOnce(false);
    });
    return () => {
      window.cancelAnimationFrame(rafId);
    };
  }, [isDirectoryDetailOpen, skipDetailTransitionOnce]);

  useEffect(() => {
    if (!isDirectoryDetailOpen) {
      return;
    }
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        closeDirectorySetModal();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [closeDirectorySetModal, isDirectoryDetailOpen]);

  useEffect(() => {
    if (mode !== "community" || !isVisible || !isDirectoryDetailOpen || !selectedDirectorySet) {
      return;
    }
    if (detailContentScrollRef.current) {
      detailContentScrollRef.current.scrollTop = 0;
    }
    const frameId = window.requestAnimationFrame(() => {
      detailBackButtonRef.current?.focus({ preventScroll: true });
    });
    return () => {
      window.cancelAnimationFrame(frameId);
    };
  }, [isDirectoryDetailOpen, isVisible, mode, selectedDirectorySet?.id]);

  useEffect(() => {
    if (!onDetailOpenChange) {
      return;
    }
    onDetailOpenChange(mode === "community" && isVisible && isDirectoryDetailOpen);
  }, [isDirectoryDetailOpen, isVisible, mode, onDetailOpenChange]);

  useEffect(() => {
    if (!onActiveCommunitySetChange) {
      return;
    }
    const nextSetId =
      mode === "community" && isVisible
        ? isDirectoryDetailOpen && selectedDirectorySet
          ? selectedDirectorySet.id
          : isInitialDetailRestorePending && normalizedInitialOpenSetId
            ? normalizedInitialOpenSetId
            : null
        : null;
    onActiveCommunitySetChange(nextSetId);
  }, [
    isDirectoryDetailOpen,
    isInitialDetailRestorePending,
    isVisible,
    mode,
    normalizedInitialOpenSetId,
    onActiveCommunitySetChange,
    selectedDirectorySet,
  ]);

  useEffect(() => {
    setDetailCarouselIndex(0);
  }, [selectedDirectorySet?.id]);

  useEffect(() => {
    if (!selectedDirectorySet || selectedDirectorySet.reels.length === 0) {
      setSelectedDetailReelId(null);
      return;
    }
    const boundedIndex = Math.min(detailCarouselIndex, selectedDirectorySet.reels.length - 1);
    const reel = selectedDirectorySet.reels[boundedIndex];
    if (!reel) {
      setSelectedDetailReelId(null);
      return;
    }
    setSelectedDetailReelId((prev) => (prev === reel.id ? prev : reel.id));
  }, [detailCarouselIndex, selectedDirectorySet]);

  useEffect(() => {
    setDetailCarouselIndex((prev) => Math.min(prev, maxDetailCarouselIndex));
  }, [maxDetailCarouselIndex]);

  const selectedDirectorySetHasReels = (selectedDirectorySet?.reels.length ?? 0) > 0;

  const openCommunityReelInFeed = useCallback(
    (set: CommunitySet, reel: CommunityReelEmbed) => {
      const nextParams = new URLSearchParams({
        community_set_id: set.id,
        community_set_title: set.title,
        community_reel_id: reel.id,
        community_reel_platform: reel.platform,
        community_reel_url: reel.embedUrl || reel.sourceUrl,
        community_reel_source_url: reel.sourceUrl,
        return_tab: "community",
        return_community_set_id: set.id,
      });
      if (localDemoEnabled) {
        nextParams.set("return_demo", "account");
      }
      if (Number.isFinite(reel.tStartSec) && Number(reel.tStartSec) >= 0) {
        nextParams.set("community_t_start_sec", String(reel.tStartSec));
      }
      if (Number.isFinite(reel.tEndSec) && Number(reel.tEndSec) > 0) {
        nextParams.set("community_t_end_sec", String(reel.tEndSec));
      }
      if (typeof window !== "undefined") {
        try {
          window.sessionStorage.setItem(`${COMMUNITY_SET_RETURN_SNAPSHOT_PREFIX}${set.id}`, JSON.stringify(set));
        } catch {
          // Ignore snapshot persistence failures.
        }
        const handoffId = `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
        try {
          window.sessionStorage.setItem(
            `${COMMUNITY_SET_FEED_HANDOFF_PREFIX}${handoffId}`,
            JSON.stringify({
              setId: set.id,
              setTitle: set.title,
              selectedReelId: reel.id,
              reels: set.reels.map((row) => ({
                id: row.id,
                platform: row.platform,
                sourceUrl: row.sourceUrl,
                embedUrl: row.embedUrl,
                tStartSec: row.tStartSec,
                tEndSec: row.tEndSec,
              })),
            }),
          );
          nextParams.set("community_handoff_id", handoffId);
        } catch {
          // Ignore storage failures and fall back to URL payload only.
        }
      }
      const feedQuery = nextParams.toString();
      onOpenCommunityReelInFeed?.({
        setId: set.id,
        setTitle: set.title,
        selectedReelId: reel.id,
        feedQuery,
      });
      router.push(`/feed?${feedQuery}`);
    },
    [localDemoEnabled, onOpenCommunityReelInFeed, router],
  );

  const openSelectedSetReelsInFeed = useCallback(() => {
    if (!selectedDirectorySet || selectedDirectorySet.reels.length === 0) {
      return;
    }
    const selectedReel =
      selectedDirectorySet.reels.find((reel) => reel.id === selectedDetailReelId) ?? selectedDirectorySet.reels[0];
    openCommunityReelInFeed(selectedDirectorySet, selectedReel);
  }, [openCommunityReelInFeed, selectedDetailReelId, selectedDirectorySet]);

  return (
    <div
      className={`relative flex h-full min-h-0 flex-col overflow-hidden text-white ${
        mode === "community"
          ? "px-3 pb-0 pt-14 sm:px-5 sm:pb-0 md:px-7 md:pb-0 md:pt-20 lg:px-8 lg:pt-7 lg:pb-0"
          : "px-3 pb-0 pt-14 sm:px-5 sm:pb-0 md:px-7 md:pb-0 md:pt-20 lg:px-8 lg:pt-7 lg:pb-0"
      }`}
    >
      {mode === "community" ? (
        <div className="relative mx-auto min-h-0 w-full flex-1 overflow-hidden lg:w-11/12 xl:w-4/5 2xl:w-full 2xl:max-w-5xl">
          <div
            ref={directoryViewRef}
            className={`absolute inset-0 flex min-h-0 flex-col transition-opacity duration-[440ms] ease-[cubic-bezier(0.22,1,0.36,1)] ${
              isDirectoryDetailOpen || shouldSuppressDirectoryDuringRestore ? "opacity-0 pointer-events-none" : "opacity-100"
            }`}
            aria-hidden={isDirectoryDetailOpen || shouldSuppressDirectoryDuringRestore}
            inert={isDirectoryDetailOpen || shouldSuppressDirectoryDuringRestore}
          >
            <div data-top-chrome="community-directory" className="top-nav-fade absolute inset-x-0 top-0 z-20 w-full shrink-0">
              <div className="relative flex min-h-12 w-full items-center gap-3 px-1 sm:px-2 md:px-3">
                <div className="flex min-w-0 flex-1 items-center pl-4 sm:pl-5 lg:pl-0">
                  <h2 className="truncate text-xl font-semibold tracking-tight text-white sm:text-2xl md:text-[1.9rem]">Community Sets</h2>
                </div>
                <div
                  data-compact-search="community"
                  data-compact-search-expanded={isCompactCommunitySearchExpanded ? "true" : "false"}
                  className={`absolute right-1 top-1/2 z-10 h-10 shrink-0 -translate-y-1/2 overflow-hidden transition-[width] duration-[440ms] ease-in-out motion-reduce:transition-none sm:right-2 md:right-3 xl:hidden ${
                    isCompactCommunitySearchExpanded ? "w-[clamp(7.5rem,48vw,17rem)]" : "w-10"
                  }`}
                >
                  <div
                    aria-hidden={!isCompactCommunitySearchExpanded}
                    className={`absolute inset-0 transition-opacity duration-300 motion-reduce:transition-none ${
                      isCompactCommunitySearchExpanded ? "pointer-events-auto opacity-100" : "pointer-events-none opacity-0"
                    }`}
                  >
                    <input
                      ref={compactCommunitySearchInputRef}
                      value={communityQuery}
                      onChange={(event) => setCommunityQuery(event.target.value)}
                      onKeyDown={(event) => {
                        if (event.key === "Escape") {
                          setCommunityQuery("");
                          setIsCompactCommunitySearchOpen(false);
                        }
                      }}
                      tabIndex={isCompactCommunitySearchExpanded ? 0 : -1}
                      aria-label="Search community sets"
                      placeholder="Search sets"
                      className="h-10 w-full rounded-full bg-white/[0.08] pl-4 pr-9 text-xs text-white outline-none backdrop-blur-[18px] backdrop-saturate-150 placeholder:text-white/35 focus:bg-white/[0.12]"
                    />
                    <button
                      type="button"
                      onClick={() => {
                        setCommunityQuery("");
                        setIsCompactCommunitySearchOpen(false);
                      }}
                      tabIndex={isCompactCommunitySearchExpanded ? 0 : -1}
                      aria-label="Close community search"
                      className="absolute right-1 top-1/2 grid h-8 w-8 -translate-y-1/2 place-items-center rounded-full text-white/55 transition-colors hover:bg-white/[0.07] hover:text-white"
                    >
                      <i className="fa-solid fa-xmark text-xs" aria-hidden="true" />
                    </button>
                  </div>
                  <button
                    type="button"
                    onClick={() => {
                      setIsCompactCommunitySearchOpen(true);
                      window.requestAnimationFrame(() => compactCommunitySearchInputRef.current?.focus());
                    }}
                    disabled={isCompactCommunitySearchExpanded}
                    aria-hidden={isCompactCommunitySearchExpanded}
                    aria-label="Open community search"
                    className={`absolute inset-0 grid h-10 w-10 place-items-center rounded-full bg-transparent text-white/72 transition-[background-color,color,opacity] duration-300 hover:bg-white/[0.07] hover:text-white motion-reduce:transition-none ${
                      isCompactCommunitySearchExpanded ? "pointer-events-none opacity-0" : "pointer-events-auto opacity-100"
                    }`}
                  >
                    <svg
                      data-community-search-icon
                      aria-hidden="true"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="1.5"
                      strokeLinecap="round"
                      className="h-[18px] w-[18px]"
                    >
                      <circle cx="10.5" cy="10.5" r="6.25" />
                      <path d="m15.25 15.25 4.25 4.25" />
                    </svg>
                  </button>
                </div>
                <label className="hidden w-[23rem] shrink-0 xl:block">
                  <div className="relative">
                    <i
                      data-community-search-icon
                      className="fa-solid fa-magnifying-glass pointer-events-none absolute left-4 top-1/2 z-10 -translate-y-1/2 text-sm text-white/45"
                      aria-hidden="true"
                    />
                    <input
                      value={communityQuery}
                      onChange={(event) => setCommunityQuery(event.target.value)}
                      placeholder="Search community sets"
                      className="h-12 w-full rounded-[1.75rem] bg-white/[0.08] pl-12 pr-4 text-sm text-white outline-none backdrop-blur-[18px] backdrop-saturate-150 placeholder:text-white/35 focus:bg-white/[0.12]"
                    />
                  </div>
                </label>
              </div>
            </div>

            <div className="min-h-0 flex-1 overflow-hidden">
              <div ref={communityScrollRef} className="balanced-scroll-gutter h-full min-h-0 space-y-4 overflow-y-auto pb-6 pt-[4.5rem] md:space-y-5 md:pb-8 md:pt-20 lg:pb-10">
            {!isCommunitySearchActive && featuredCarouselSets.length > 0 ? (
              <section
                data-community-featured-carousel
                className="group/featured relative overflow-hidden rounded-[14px] bg-white/[0.07] p-3 pb-9 backdrop-blur-[18px] backdrop-saturate-150 sm:p-4 sm:pb-10"
              >
                {featuredCarouselSets.length > 1 ? (
                  <>
                    <button
                      type="button"
                      aria-label="Next featured set"
                      onClick={goToNextFeaturedSet}
                      className="community-featured-next-control absolute right-3 top-3 z-30 grid h-8 w-8 place-items-center rounded-full bg-black/45 text-white/82 transition-[background-color,color,opacity] duration-300 motion-reduce:transition-none hover:bg-white/[0.07] hover:text-white sm:right-4 sm:top-4 md:pointer-events-none md:top-1/2 md:-translate-y-1/2 md:opacity-0 md:group-hover/featured:pointer-events-auto md:group-hover/featured:opacity-100 md:group-focus-within/featured:pointer-events-auto md:group-focus-within/featured:opacity-100"
                    >
                      <i className="fa-solid fa-chevron-right text-[10px]" aria-hidden="true" />
                    </button>
                  </>
                ) : null}

                <div className="relative z-10 overflow-hidden" style={{ minHeight: `${featuredCarouselContentHeight}px` }}>
                  {featuredCarouselSets.map((set, index) => {
                    const isLeaving = featuredTransitionStage === "exiting" && index === leavingFeaturedIndex;
                    const isActive = index === activeFeaturedIndex && featuredTransitionStage !== "exiting" && featuredTransitionStage !== "pause";
                    if (!isActive && !isLeaving) {
                      return null;
                    }
                    const motionClass = isLeaving
                      ? "animate-featured-fade-exit"
                      : featuredTransitionStage === "entering"
                        ? "animate-featured-fade-enter"
                        : "opacity-100";
                    return (
                      <article
                        key={`${set.id}-${isLeaving ? "leaving" : "active"}`}
                        className={`absolute inset-0 ${isLeaving ? "z-10 pointer-events-none" : "z-20"} ${motionClass}`}
                      >
                        <div
                          ref={isLeaving ? null : activeFeaturedSlideRef}
                          className="community-featured-grid grid min-w-0 gap-4 sm:gap-5 md:items-center"
                        >
                          <div className="flex w-full min-w-0 max-w-2xl flex-col items-start text-left">
                            <p className="inline-flex rounded-full bg-white/12 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.1em] text-white/88">
                              Featured Set
                            </p>
                            <h3 className="mt-3 text-xl font-semibold leading-[1.12] text-white sm:text-2xl md:text-3xl">{set.title}</h3>
                            <p className="mt-3 w-full max-w-xl break-words text-sm leading-relaxed text-white/80 md:text-base">{set.description}</p>

                            <div className="mt-4 flex flex-wrap items-center gap-1.5 text-[9px] font-semibold uppercase tracking-[0.08em] text-white/72 sm:text-[10px]">
                              <span className="rounded-full bg-black/30 px-2 py-1">{getSetReelCount(set)} reels</span>
                              <span className="rounded-full bg-black/30 px-2 py-1">{formatCompact(set.learners)} learners</span>
                              <span className="rounded-full bg-black/30 px-2 py-1">{formatCompact(set.likes)} likes</span>
                            </div>

                            <button
                              type="button"
                              data-featured-view-set-button
                              onClick={() => openDirectorySet(set)}
                              className="mt-5 inline-flex w-auto items-center justify-center rounded-full bg-white px-6 py-2 text-xs font-semibold text-black transition hover:bg-[#f1eee5]"
                            >
                              View Set
                            </button>
                          </div>

                          <div className="relative hidden h-full min-h-[250px] md:block">
                            <div
                              data-featured-image-target
                              className="community-featured-image-frame absolute bottom-0 right-0 overflow-hidden rounded-[10px] bg-black/30"
                            >
                              <img
                                src={set.thumbnailUrl || FALLBACK_THUMBNAIL_URL}
                                alt={`${set.title} cover`}
                                className="community-featured-image w-full object-contain"
                              />
                            </div>
                          </div>
                        </div>
                      </article>
                    );
                  })}
                </div>

                {featuredCarouselSets.length > 1 ? (
                  <div className="absolute bottom-3 left-1/2 z-20 flex -translate-x-1/2 items-center justify-center gap-1.5 sm:bottom-4">
                    {featuredCarouselSets.map((set, index) => (
                      <button
                        key={`featured-dot-${set.id}`}
                        type="button"
                        aria-label={`Go to featured set ${index + 1}`}
                        onClick={() => goToFeaturedSet(index)}
                        className={`h-1.5 rounded-full ${
                          index === activeFeaturedIndex ? "w-5 bg-white" : "w-1.5 bg-white/45 hover:bg-white/70"
                        }`}
                      />
                    ))}
                  </div>
                ) : null}
              </section>
            ) : null}

            {!isCommunitySearchActive ? (
            <section className="flex items-center gap-2 overflow-x-auto pb-1">
              {communityCategories.map((category) => (
                <button
                  key={category}
                  type="button"
                  onClick={() => onCommunityCategoryChange(category)}
                  className={`whitespace-nowrap rounded-full px-3 py-1.5 text-xs font-medium sm:px-4 sm:py-2 sm:text-sm ${
                    activeCommunityCategory === category
                      ? "bg-white text-black"
                      : "bg-black/30 text-white/75 transition-colors hover:bg-white/[0.07] hover:text-white"
                  }`}
                >
                  {category}
                </button>
              ))}
            </section>
            ) : null}

            <section data-community-directory className="relative">
              <div className="relative z-10">
              <div className="mb-1.5 flex items-center justify-between px-1">
                <p className="text-[11px] font-semibold uppercase tracking-[0.1em] text-white/60">
                  {isCommunitySearchActive ? "Search Results" : "Community Directory"}
                </p>
                <p className="text-[10px] font-semibold uppercase tracking-[0.09em] text-white/45">{directorySets.length} sets</p>
              </div>
              {directorySets.length === 0 ? (
                <p className="px-1 py-4 text-sm text-white/65">
                  {isCommunitySearchActive ? "No sets matched your search." : "No sets matched that search."}
                </p>
              ) : (
                <div className={isCommunitySearchActive ? "flex flex-col gap-1.5" : "grid gap-1.5 md:grid-cols-2 md:gap-x-3 md:gap-y-2 lg:gap-x-5"}>
                  {directorySets.map((set) => {
                    const reelCount = getSetReelCount(set);
                    return (
                      <button
                        type="button"
                        key={set.id}
                        data-community-set-row
                        data-community-set-id={set.id}
                        onClick={() => openDirectorySet(set)}
                        className="group relative flex min-h-14 w-full items-center gap-2 rounded-lg bg-transparent px-2 py-1.5 text-left transition-colors hover:bg-white/[0.07] focus-visible:bg-white/[0.07] sm:gap-2.5 sm:rounded-xl"
                      >
                        <span className="grid h-11 w-11 shrink-0 overflow-hidden rounded-lg bg-black/30 text-white/82 transition-colors duration-200 group-hover:text-white">
                          {set.thumbnailUrl ? (
                            <img src={set.thumbnailUrl} alt="" aria-hidden="true" className="h-full w-full object-cover" />
                          ) : (
                            <i className="fa-regular fa-square text-sm sm:text-base" aria-hidden="true" />
                          )}
                        </span>
                        <div className="min-w-0 flex-1 text-left">
                          <p className="w-full truncate text-sm font-medium text-white transition-colors duration-200 group-hover:text-white">{set.title}</p>
                          <p className="mt-0.5 hidden w-full truncate text-xs text-white/58 transition-colors duration-200 group-hover:text-white/78 2xl:block">{set.description}</p>
                          <span className="mt-0.5 inline-flex text-[9px] font-semibold uppercase tracking-[0.08em] text-white/50 transition-colors duration-200 group-hover:text-white/75">
                            {reelCount} reels
                          </span>
                        </div>
                        <span
                          className="grid h-6 w-6 shrink-0 place-items-center self-center rounded-full text-white/50 transition-colors duration-200 group-hover:text-white/80"
                          aria-hidden="true"
                        >
                          <i className="fa-solid fa-chevron-right text-[10px]" />
                        </span>
                      </button>
                    );
                  })}
                </div>
              )}
              </div>
            </section>
              </div>
            </div>
          </div>

          <section
            role="region"
            aria-labelledby={selectedDirectorySet ? "community-set-detail-title" : undefined}
            className={`absolute inset-0 flex min-h-0 flex-col ${skipDetailTransitionOnce ? "" : "transition-opacity duration-[440ms] ease-[cubic-bezier(0.22,1,0.36,1)]"} ${
              isDirectoryDetailOpen ? "opacity-100" : "opacity-0 pointer-events-none"
            }`}
            aria-hidden={!isDirectoryDetailOpen}
            inert={!isDirectoryDetailOpen}
          >
            {selectedDirectorySet ? (
              <div
                ref={detailContentScrollRef}
                className="balanced-scroll-gutter min-h-0 flex-1 overflow-y-auto"
              >
                <div data-community-detail-view className="mx-auto w-full max-w-4xl pb-14 md:pb-20">
                  <div
                    data-top-chrome="community-detail-back"
                    className="top-nav-fade sticky top-0 z-20 w-full px-1 pb-5 pt-1 sm:px-2 md:px-3 md:pb-6"
                  >
                    <button
                      ref={detailBackButtonRef}
                      type="button"
                      onClick={closeDirectorySetModal}
                      className="inline-flex items-center gap-2 rounded-lg px-2 py-2 text-sm font-medium text-white/82 transition-colors hover:bg-white/[0.07] hover:text-white"
                    >
                      <i className="fa-solid fa-chevron-left text-[11px]" aria-hidden="true" />
                      Community Sets
                    </button>
                  </div>

                  <header
                    data-community-detail-header
                    className="pt-8 sm:pt-10 md:pt-12"
                  >
                    <div className="px-3 sm:px-5 md:px-8">
                      <span className="grid h-16 w-16 shrink-0 overflow-hidden rounded-[14px] bg-[#151515] text-white/90 sm:h-[72px] sm:w-[72px]">
                        {selectedDirectorySet.thumbnailUrl ? (
                          <img src={selectedDirectorySet.thumbnailUrl} alt="" aria-hidden="true" className="h-full w-full object-cover" />
                        ) : (
                          <i className={`${getSetIconClass(selectedDirectorySet)} m-auto text-xl`} aria-hidden="true" />
                        )}
                      </span>

                      <div className="mt-7 flex flex-col gap-5 sm:mt-8 md:flex-row md:items-end md:justify-between">
                        <div className="min-w-0 max-w-2xl">
                          <p className="text-[10px] font-semibold uppercase tracking-[0.12em] text-white/52">Community Set</p>
                          <h3 id="community-set-detail-title" className="mt-2 break-words text-[2rem] font-semibold leading-[1.08] tracking-[-0.025em] text-white sm:text-[2.45rem]">
                            {selectedDirectorySet.title}
                          </h3>
                          <p className="mt-3 break-words text-sm leading-relaxed text-white/68 sm:text-base">{selectedDirectorySet.description}</p>
                        </div>

                        <button
                          type="button"
                          onClick={openSelectedSetReelsInFeed}
                          disabled={!selectedDirectorySetHasReels}
                          className={`inline-flex h-10 shrink-0 items-center justify-center self-start rounded-full px-5 text-sm font-semibold transition-colors duration-300 md:self-end ${
                            selectedDirectorySetHasReels
                              ? "bg-white text-black hover:bg-white/[0.88]"
                              : "cursor-not-allowed bg-white/[0.32] text-black/60"
                          }`}
                        >
                          View Reels
                        </button>
                      </div>

                      <div className="mt-5 flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1 break-words text-xs text-white/50 sm:text-[13px]">
                        <span>{getSetReelCount(selectedDirectorySet)} reels</span>
                        <span aria-hidden="true">·</span>
                        <span>{formatCompact(selectedDirectorySet.learners)} learners</span>
                        <span aria-hidden="true">·</span>
                        <span>{formatCompact(selectedDirectorySet.likes)} likes</span>
                        <span aria-hidden="true">·</span>
                        <span>Curated by {selectedDirectorySetCuratorLabel}</span>
                      </div>
                    </div>
                  </header>

                  <div className="px-3 sm:px-5 md:px-8">
                    <section data-community-detail-media className="mt-10 overflow-hidden rounded-[14px] bg-[#151515] sm:mt-12">
                      {activeDetailCarouselReel ? (
                        <>
                          <div className="relative aspect-video w-full overflow-hidden bg-black">
                            <iframe
                              src={activeDetailCarouselReel.embedUrl}
                              title={`${selectedDirectorySet.title} reel preview`}
                              className="absolute inset-0 h-full w-full border-0"
                              loading="lazy"
                              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                              allowFullScreen
                            />
                            {detailCarouselCount > 1 ? (
                              <div className="absolute bottom-3 right-3 flex items-center gap-1.5">
                                <button
                                  type="button"
                                  onClick={goToPreviousDetailCarousel}
                                  aria-label="Previous reel"
                                  className="grid h-9 w-9 place-items-center rounded-full bg-black/[0.68] text-white/82 backdrop-blur-md transition-colors hover:bg-white/[0.07] hover:text-white"
                                >
                                  <i className="fa-solid fa-chevron-left text-[10px]" aria-hidden="true" />
                                </button>
                                <button
                                  type="button"
                                  onClick={goToNextDetailCarousel}
                                  aria-label="Next reel"
                                  className="grid h-9 w-9 place-items-center rounded-full bg-black/[0.68] text-white/82 backdrop-blur-md transition-colors hover:bg-white/[0.07] hover:text-white"
                                >
                                  <i className="fa-solid fa-chevron-right text-[10px]" aria-hidden="true" />
                                </button>
                              </div>
                            ) : null}
                          </div>
                          <div className="flex items-center justify-between gap-2 px-4 py-3 text-[10px] font-semibold uppercase tracking-[0.08em] text-white/55">
                            <span className="inline-flex items-center gap-1.5">
                              <i className={PLATFORM_ICON[activeDetailCarouselReel.platform]} aria-hidden="true" />
                              {PLATFORM_LABEL[activeDetailCarouselReel.platform]}
                            </span>
                            {formatClipRangeLabel(activeDetailCarouselReel) ? (
                              <span>Clip {formatClipRangeLabel(activeDetailCarouselReel)}</span>
                            ) : (
                              <span>{detailCarouselIndex + 1} / {detailCarouselCount}</span>
                            )}
                          </div>
                        </>
                      ) : (
                        <div className="relative aspect-video w-full overflow-hidden">
                          <img
                            src={selectedDirectorySet.thumbnailUrl || FALLBACK_THUMBNAIL_URL}
                            alt={`${selectedDirectorySet.title} cover`}
                            className="h-full w-full object-cover"
                          />
                        </div>
                      )}
                    </section>

                    <section data-community-detail-about className="mt-9 sm:mt-11">
                      <h4 className="text-xl font-semibold tracking-[-0.015em] text-white">About this set</h4>
                      <div className="mt-4 max-w-3xl space-y-4">
                        {buildCommunitySetInformationParagraphs(selectedDirectorySet, selectedDirectorySetCuratorLabel).map((paragraph, index) => (
                          <p key={`${selectedDirectorySet.id}-detail-info-${index}`} className="break-words text-sm leading-7 text-white/66 sm:text-[15px]">
                            {paragraph}
                          </p>
                        ))}
                      </div>
                    </section>
                  </div>
                </div>
              </div>
            ) : null}
          </section>
        </div>
      ) : (
        <>
          <div className="relative mx-auto flex min-h-0 w-full flex-1 flex-col lg:w-11/12 xl:w-4/5 2xl:w-full 2xl:max-w-5xl">
            <div data-top-chrome="community-management" className="top-nav-fade absolute inset-x-0 top-0 z-20 w-full shrink-0">
              <div className="relative flex min-h-12 w-full items-center gap-3 px-1 sm:px-2 md:px-3">
                <div
                  className={
                    isYourSetsMode && shouldShowEditSetGrid
                      ? "flex min-w-0 flex-1 items-center pl-4 sm:pl-5 lg:pl-0"
                      : "flex min-w-0 flex-1 items-center justify-center pl-4 sm:pl-5 lg:justify-start lg:pl-0"
                  }
                >
                  <div
                    className={`flex min-w-0 items-center gap-2 ${
                      isYourSetsMode && shouldShowEditSetForm ? "lg:pl-11" : ""
                    }`}
                  >
                    {isYourSetsMode && shouldShowEditSetForm ? (
                      <button
                        type="button"
                        onClick={onBackToEditSetGrid}
                        data-create-set-back="true"
                        aria-label="Back to all sets"
                        className="absolute left-1 sm:left-2 md:left-1.5 lg:left-0.5 inline-flex h-9 w-9 items-center justify-center rounded-xl text-white/70 transition-colors hover:bg-white/[0.07] hover:text-white focus-visible:bg-white/[0.09]"
                      >
                        <i className="fa-solid fa-chevron-left text-[11px]" aria-hidden="true" />
                      </button>
                    ) : null}
                    <h2 className="truncate text-xl font-semibold tracking-tight text-white sm:text-2xl md:text-[1.9rem]">
                      {isYourSetsMode
                        ? shouldShowEditSetForm
                          ? isFormEditMode
                            ? `Editing "${activeEditableSet?.title ?? "Set"}"`
                            : "Create Set"
                          : "Your Sets"
                        : "Create Set"}
                    </h2>
                  </div>
                </div>
                {shouldShowYourSetsSearch ? (
                  <>
                    <div
                      data-compact-search="your-sets"
                      data-compact-search-expanded={isCompactYourSetsSearchExpanded ? "true" : "false"}
                      className={`absolute right-1 top-1/2 z-10 h-10 shrink-0 -translate-y-1/2 overflow-hidden transition-[width] duration-[440ms] ease-in-out motion-reduce:transition-none sm:right-2 md:right-3 xl:hidden ${
                        isCompactYourSetsSearchExpanded ? "w-[clamp(7.5rem,48vw,17rem)]" : "w-10"
                      }`}
                    >
                      <div
                        aria-hidden={!isCompactYourSetsSearchExpanded}
                        className={`absolute inset-0 transition-opacity duration-300 motion-reduce:transition-none ${
                          isCompactYourSetsSearchExpanded ? "pointer-events-auto opacity-100" : "pointer-events-none opacity-0"
                        }`}
                      >
                        <input
                          ref={compactYourSetsSearchInputRef}
                          value={yourSetsQuery}
                          onChange={(event) => setYourSetsQuery(event.target.value)}
                          onKeyDown={(event) => {
                            if (event.key === "Escape") {
                              setYourSetsQuery("");
                              setIsCompactYourSetsSearchOpen(false);
                            }
                          }}
                          tabIndex={isCompactYourSetsSearchExpanded ? 0 : -1}
                          aria-label="Search your sets"
                          placeholder="Search sets"
                          className="h-10 w-full rounded-full bg-white/[0.08] pl-4 pr-9 text-xs text-white outline-none backdrop-blur-[18px] backdrop-saturate-150 placeholder:text-white/35 focus:bg-white/[0.12]"
                        />
                        <button
                          type="button"
                          onClick={() => {
                            setYourSetsQuery("");
                            setIsCompactYourSetsSearchOpen(false);
                          }}
                          tabIndex={isCompactYourSetsSearchExpanded ? 0 : -1}
                          aria-label="Close your sets search"
                          className="absolute right-1 top-1/2 grid h-8 w-8 -translate-y-1/2 place-items-center rounded-full text-white/55 transition-colors hover:bg-white/[0.07] hover:text-white"
                        >
                          <i className="fa-solid fa-xmark text-xs" aria-hidden="true" />
                        </button>
                      </div>
                      <button
                        type="button"
                        onClick={() => {
                          setIsCompactYourSetsSearchOpen(true);
                          window.requestAnimationFrame(() => compactYourSetsSearchInputRef.current?.focus());
                        }}
                        disabled={isCompactYourSetsSearchExpanded}
                        aria-hidden={isCompactYourSetsSearchExpanded}
                        aria-label="Open your sets search"
                        className={`absolute inset-0 grid h-10 w-10 place-items-center rounded-full bg-transparent text-white/72 transition-[background-color,color,opacity] duration-300 hover:bg-white/[0.07] hover:text-white motion-reduce:transition-none ${
                          isCompactYourSetsSearchExpanded ? "pointer-events-none opacity-0" : "pointer-events-auto opacity-100"
                        }`}
                      >
                        <svg
                          data-your-sets-search-icon
                          aria-hidden="true"
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="1.5"
                          strokeLinecap="round"
                          className="h-[18px] w-[18px]"
                        >
                          <circle cx="10.5" cy="10.5" r="6.25" />
                          <path d="m15.25 15.25 4.25 4.25" />
                        </svg>
                      </button>
                    </div>
                    <label className="hidden w-[23rem] shrink-0 xl:block">
                      <div className="relative">
                        <svg
                          data-your-sets-search-icon
                          aria-hidden="true"
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="1.5"
                          strokeLinecap="round"
                          className="pointer-events-none absolute left-4 top-1/2 z-10 h-[18px] w-[18px] -translate-y-1/2 text-white/55"
                        >
                          <circle cx="10.5" cy="10.5" r="6.25" />
                          <path d="m15.25 15.25 4.25 4.25" />
                        </svg>
                        <input
                          value={yourSetsQuery}
                          onChange={(event) => setYourSetsQuery(event.target.value)}
                          placeholder="Search your sets"
                          className="h-12 w-full rounded-[1.75rem] bg-white/[0.08] pl-12 pr-4 text-sm text-white outline-none backdrop-blur-[18px] backdrop-saturate-150 placeholder:text-white/35 focus:bg-white/[0.12]"
                        />
                      </div>
                    </label>
                  </>
                ) : null}
              </div>
            </div>

            <div className="min-h-0 flex-1 overflow-hidden md:-mx-1.5 lg:-mx-2.5">
              <div
                className={`balanced-scroll-gutter h-full min-h-0 overflow-y-auto pb-0 ${
                  shouldShowEditSetGrid ? "pt-[5.5rem] md:pt-24" : "pt-[3.75rem] md:pt-16"
                }`}
              >
                {!isCommunityAuthReady ? (
                  <section className="rounded-3xl px-1 pt-1 pb-2 sm:px-2 sm:pt-2 sm:pb-3 md:px-3 md:pt-3 md:pb-4">
                    <div className="relative overflow-hidden rounded-2xl bg-white/[0.07] p-5 backdrop-blur-[18px] backdrop-saturate-150">
                      <div className="relative z-10">
                        <p className="text-[11px] font-semibold uppercase tracking-[0.11em] text-white/70">Account</p>
                        <p className="mt-3 text-sm text-white/72">Loading your community account…</p>
                      </div>
                    </div>
                  </section>
                ) : needsCommunityAuth ? (
                  <section className="rounded-3xl px-1 pt-1 pb-2 sm:px-2 sm:pt-2 sm:pb-3 md:px-3 md:pt-3 md:pb-4">
                    <div className="relative overflow-hidden rounded-2xl bg-white/[0.07] p-4 pb-5 backdrop-blur-[18px] backdrop-saturate-150 sm:p-5 sm:pb-6">
                      <div className="relative z-10 max-w-xl">
                        <div className="flex flex-wrap items-center gap-2">
                          <p className="text-[11px] font-semibold uppercase tracking-[0.11em] text-white/70">Your Sets</p>
                          <span className="rounded-full border border-[#2b2b2b] bg-black/45 px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.09em] text-white/65">
                            Account Required
                          </span>
                        </div>
                        <h3 className="mt-10 text-xl font-semibold text-white">Sign in to manage your sets</h3>
                      </div>
                    </div>
                  </section>
                ) : needsCommunityVerification ? (
                  <section className="rounded-3xl px-1 pt-1 pb-2 sm:px-2 sm:pt-2 sm:pb-3 md:px-3 md:pt-3 md:pb-4">
                    <div className="relative overflow-hidden rounded-2xl bg-white/[0.07] p-4 pb-5 backdrop-blur-[18px] backdrop-saturate-150 sm:p-5 sm:pb-6">
                      <div className="relative z-10 max-w-xl">
                        <div className="flex flex-wrap items-center gap-2">
                          <p className="text-[11px] font-semibold uppercase tracking-[0.11em] text-white/70">Your Sets</p>
                          <span className="rounded-full border border-[#2b2b2b] bg-black/45 px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.09em] text-white/65">
                            Verification Required
                          </span>
                        </div>
                        <h3 className="mt-3 text-xl font-semibold text-white">Verify your email to manage Your Sets</h3>
                        <p className="mt-3 text-sm leading-6 text-white/72">
                          {communityAccount?.email
                            ? `Finish verifying ${communityAccount.email} before creating, editing, or viewing your sets.`
                            : "Finish verifying your account before creating, editing, or viewing your sets."}
                        </p>
                        <div className="mt-5 flex flex-wrap items-center gap-3">
                          <button
                            type="button"
                            onClick={() => router.push("/account?return_tab=edit")}
                            className="inline-flex h-11 min-w-[12rem] items-center justify-center rounded-xl border border-[#2b2b2b] bg-black/55 px-4 text-sm font-semibold text-white transition hover:bg-white hover:text-black"
                          >
                            Verify Account
                          </button>
                        </div>
                      </div>
                    </div>
                  </section>
                ) : (
                  <>
                {shouldShowEditSetGrid ? (
                  <section className="px-0 pb-1">
                    <div className="relative pb-2">
                      <div className="relative z-10">
                        {filteredEditableSets.length > 0 ? (
                          <div
                            data-your-sets-list
                            className="pt-0"
                          >
                            <div
                              data-your-sets-list-header
                              className="grid grid-cols-[1.75rem_minmax(0,1fr)_7rem_3rem] items-center px-2 pb-1 text-sm text-white/62 sm:grid-cols-[1.75rem_minmax(0,1fr)_10rem_3.25rem]"
                            >
                              <div
                                data-your-sets-name-action
                                className={`relative col-start-2 min-w-0 transition-[height] duration-300 ease-out motion-reduce:transition-none ${
                                  selectedEditableSetIds.length > 0 ? "h-7" : "h-5"
                                }`}
                              >
                                <span
                                  aria-hidden={selectedEditableSetIds.length > 0}
                                  className={`absolute inset-y-0 left-0 flex min-w-0 max-w-full items-center truncate transition-[opacity,transform] duration-200 ease-out motion-reduce:transition-none ${
                                    selectedEditableSetIds.length > 0
                                      ? "pointer-events-none -translate-y-1 opacity-0"
                                      : "translate-y-0 opacity-100"
                                  }`}
                                >
                                  Name
                                </span>
                                <button
                                  type="button"
                                  data-your-sets-delete-selected
                                  onClick={onRequestDeleteSelectedEditableSets}
                                  disabled={isDeletingSets || selectedEditableSetIds.length === 0}
                                  tabIndex={selectedEditableSetIds.length > 0 ? 0 : -1}
                                  aria-hidden={selectedEditableSetIds.length === 0}
                                  aria-label={`Delete ${selectedEditableSetIds.length} selected ${selectedEditableSetIds.length === 1 ? "set" : "sets"}`}
                                  className={`absolute left-0 top-0 inline-flex h-7 w-fit min-w-0 items-center justify-center overflow-hidden rounded-full border border-red-400/55 px-3 font-medium text-red-300 transition-[transform,opacity,background-color,border-color,color] [transition-duration:220ms,140ms,150ms,150ms,150ms] ease-out hover:border-red-300 hover:bg-white/[0.07] hover:text-red-200 focus-visible:border-red-300 focus-visible:text-red-200 focus-visible:outline-none motion-reduce:transition-none disabled:cursor-not-allowed ${
                                    selectedEditableSetIds.length > 0
                                      ? "translate-y-0 opacity-100 [transition-delay:0ms,220ms,0ms,0ms,0ms] disabled:opacity-50"
                                      : "pointer-events-none translate-y-2 opacity-0 [transition-delay:0ms]"
                                  }`}
                                >
                                  <span className="truncate">Delete</span>
                                </button>
                              </div>
                              <button
                                type="button"
                                data-your-sets-modified-sort
                                onClick={() => {
                                  setYourSetsModifiedSortDirection((current) => current === "newest" ? "oldest" : "newest");
                                }}
                                aria-label={`Sort by modified date, currently ${yourSetsModifiedSortDirection} first`}
                                title={`Sort by modified date (${yourSetsModifiedSortDirection} first)`}
                                className="col-start-3 inline-flex min-w-0 max-w-full items-center gap-1 overflow-hidden font-normal text-white/62 transition-colors hover:text-white focus-visible:text-white focus-visible:outline-none"
                              >
                                <span className="truncate">Modified</span>
                                <svg
                                  data-your-sets-sort-arrow
                                  viewBox="0 0 16 16"
                                  aria-hidden="true"
                                  className={`h-3.5 w-3.5 fill-none stroke-current transition-transform duration-200 ${
                                    yourSetsModifiedSortDirection === "oldest" ? "rotate-180" : "rotate-0"
                                  }`}
                                  strokeWidth="1.5"
                                  strokeLinecap="round"
                                  strokeLinejoin="round"
                                >
                                  <path d="m4.5 6.5 3.5 3.5 3.5-3.5" />
                                </svg>
                              </button>
                            </div>
                            <ul className="space-y-1">
                              {filteredEditableSets.map((set) => {
                                const isStarred = starredSetIdSet.has(set.id);
                                const isSelected = selectedEditableSetIdSet.has(set.id);
                                const isDeleting = deletingSetIdSet.has(set.id);
                                const isActionsMenuOpen = activeSetActionsMenuId === set.id;
                                const modifiedLabel = formatModifiedLabel(set, relativeTimeNowMs);
                                return (
                                  <li
                                    key={`edit-set-list-${set.id}`}
                                    data-your-set-row
                                    className={`group relative grid min-h-[3rem] grid-cols-[1.75rem_minmax(0,1fr)_7rem_3rem] items-stretch bg-transparent text-left sm:min-h-[3.25rem] sm:grid-cols-[1.75rem_minmax(0,1fr)_10rem_3.25rem] ${
                                      isActionsMenuOpen ? "z-40" : "z-0"
                                    }`}
                                  >
                                    <span
                                      data-your-set-surface
                                      aria-hidden="true"
                                      className="pointer-events-none absolute bottom-0 left-7 right-0 top-0 rounded-[12px] bg-transparent transition-colors duration-300 group-hover:bg-white/[0.07] motion-reduce:transition-none"
                                    />
                                    <button
                                      type="button"
                                      role="checkbox"
                                      data-your-set-checkbox
                                      data-selected={isSelected ? "true" : "false"}
                                      aria-checked={isSelected}
                                      aria-label={`${isSelected ? "Deselect" : "Select"} ${set.title}`}
                                      disabled={isDeletingSets}
                                      onClick={(event) => {
                                        event.preventDefault();
                                        event.stopPropagation();
                                        onToggleEditableSetSelection(set.id);
                                      }}
                                      className={`relative z-20 col-start-1 row-start-1 grid place-items-center rounded-l-[12px] transition-opacity duration-200 focus-visible:opacity-100 focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-40 motion-reduce:transition-none ${
                                        isSelected
                                          ? "opacity-100"
                                          : "opacity-0 group-hover:opacity-100 group-focus-within:opacity-100"
                                      }`}
                                    >
                                      <span
                                        aria-hidden="true"
                                        className={`grid h-4 w-4 place-items-center rounded-[4px] border transition-colors duration-200 motion-reduce:transition-none ${
                                          isSelected
                                            ? "border-white bg-white text-black"
                                            : "border-white/45 bg-black/30 text-transparent"
                                        }`}
                                      >
                                        <svg viewBox="0 0 16 16" className="h-3 w-3 fill-none stroke-current" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                          <path d="m3.25 8.25 3 3 6.5-7" />
                                        </svg>
                                      </span>
                                    </button>
                                    <button
                                      type="button"
                                      onClick={() => onOpenEditableSet(set.id)}
                                      aria-label={`Open ${set.title}`}
                                      className="relative z-10 col-start-2 col-span-2 grid min-w-0 grid-cols-[minmax(0,1fr)_7rem] items-center rounded-l-[12px] px-2 py-1.5 text-left outline-none focus-visible:outline focus-visible:outline-1 focus-visible:outline-offset-[-2px] focus-visible:outline-white/70 sm:grid-cols-[minmax(0,1fr)_10rem]"
                                    >
                                      <span className="flex min-w-0 items-center gap-3 overflow-hidden">
                                        <span
                                          data-your-set-thumbnail
                                          className="relative h-9 w-9 shrink-0 overflow-hidden rounded-[8px] bg-black sm:h-10 sm:w-10"
                                        >
                                          <img
                                            src={set.thumbnailUrl || FALLBACK_THUMBNAIL_URL}
                                            alt=""
                                            aria-hidden="true"
                                            loading="lazy"
                                            className="h-full w-full object-cover"
                                          />
                                        </span>
                                        <span className="flex min-w-0 flex-1 items-center gap-2 overflow-hidden">
                                          <span
                                            data-your-set-title
                                            title={set.title}
                                            className="min-w-0 flex-1 truncate text-sm font-medium text-white sm:text-[15px]"
                                          >
                                            {set.title}
                                          </span>
                                          {isStarred ? <i className="fa-solid fa-star shrink-0 text-[10px] text-white/80" aria-hidden="true" /> : null}
                                        </span>
                                      </span>
                                      <span
                                        data-your-set-modified
                                        title={modifiedLabel}
                                        className="min-w-0 truncate pr-2 text-sm text-white/62"
                                      >
                                        {modifiedLabel}
                                      </span>
                                    </button>
                                    <div
                                      data-your-set-actions="true"
                                      className="relative z-30 col-start-4 flex items-center justify-center"
                                    >
                                      <button
                                        type="button"
                                        onClick={(event) => {
                                          event.preventDefault();
                                          event.stopPropagation();
                                          setActiveSetActionsMenuId((prev) => (prev === set.id ? null : set.id));
                                        }}
                                        aria-label={`Actions for ${set.title}`}
                                        aria-haspopup="menu"
                                        aria-expanded={isActionsMenuOpen}
                                        aria-controls={`your-set-actions-menu-${set.id}`}
                                        className="grid h-9 w-9 place-items-center rounded-lg text-white/58 transition-colors hover:bg-white/[0.07] hover:text-white"
                                      >
                                        <svg
                                          data-your-set-more-icon
                                          viewBox="0 0 24 24"
                                          aria-hidden="true"
                                          className="h-5 w-5 shrink-0 fill-none stroke-current"
                                          strokeWidth="1.5"
                                          strokeLinecap="round"
                                          strokeLinejoin="round"
                                        >
                                          <circle cx="6" cy="12" r="1.15" fill="currentColor" stroke="none" />
                                          <circle cx="12" cy="12" r="1.15" fill="currentColor" stroke="none" />
                                          <circle cx="18" cy="12" r="1.15" fill="currentColor" stroke="none" />
                                        </svg>
                                      </button>
                                      <div
                                        aria-hidden={!isActionsMenuOpen}
                                        className={`absolute right-2 top-[calc(50%+1.5rem)] z-50 w-44 transition-opacity duration-300 motion-reduce:transition-none ${
                                          isActionsMenuOpen
                                            ? "pointer-events-auto opacity-100"
                                            : "pointer-events-none opacity-0"
                                        }`}
                                      >
                                        <div
                                          id={`your-set-actions-menu-${set.id}`}
                                          role="menu"
                                          className="overflow-hidden rounded-xl bg-[#202020] p-1.5"
                                        >
                                          <button
                                            type="button"
                                            role="menuitem"
                                            tabIndex={isActionsMenuOpen ? 0 : -1}
                                            onClick={() => {
                                              onOpenEditableSet(set.id);
                                              setActiveSetActionsMenuId(null);
                                            }}
                                            className="flex w-full items-center gap-2 rounded-lg px-2.5 py-2 text-left text-xs text-white/90 transition-colors hover:bg-white/[0.07]"
                                          >
                                            <i className="fa-solid fa-pen-to-square text-[11px] text-white/80" aria-hidden="true" />
                                            Edit
                                          </button>
                                          <button
                                            type="button"
                                            role="menuitem"
                                            tabIndex={isActionsMenuOpen ? 0 : -1}
                                            onClick={() => onToggleSetStar(set.id)}
                                            className="mt-0.5 flex w-full items-center gap-2 rounded-lg px-2.5 py-2 text-left text-xs text-white/90 transition-colors hover:bg-white/[0.07]"
                                          >
                                            <i className={`fa-${isStarred ? "solid" : "regular"} fa-star text-[11px] text-white/80`} aria-hidden="true" />
                                            {isStarred ? "Unstar" : "Star"}
                                          </button>
                                          <button
                                            type="button"
                                            role="menuitem"
                                            tabIndex={isActionsMenuOpen ? 0 : -1}
                                            onClick={() => {
                                              onRequestDeleteEditableSet(set.id);
                                            }}
                                            disabled={isDeleting}
                                            className="mt-0.5 flex w-full items-center gap-2 rounded-lg px-2.5 py-2 text-left text-xs text-white/90 transition-colors hover:bg-white/[0.07] disabled:cursor-not-allowed disabled:opacity-60"
                                          >
                                            <i className="fa-regular fa-trash-can text-[11px] text-white/80" aria-hidden="true" />
                                            {isDeleting ? "Deleting..." : "Delete"}
                                          </button>
                                        </div>
                                      </div>
                                    </div>
                                  </li>
                                );
                              })}
                            </ul>
                          </div>
                        ) : isYourSetsSearchActive ? (
                          <p data-your-sets-empty-state className="mt-4 pl-5 text-sm text-white/66 sm:pl-7 md:pl-[2.375rem] lg:pl-[1.375rem]">
                            No sets matched your search.
                          </p>
                        ) : (
                          <p data-your-sets-empty-state className="mt-4 pl-5 text-sm text-white/66 sm:pl-7 md:pl-[2.375rem] lg:pl-[1.375rem]">
                            No sets yet. Use the plus button to publish your first one.
                          </p>
                        )}
                      </div>
                    </div>
                  </section>
                ) : null}
                {shouldShowEditSetForm ? (
                <section data-create-set-view="true" className="px-1 pb-6 pt-2 sm:px-2 md:px-3 md:pb-8">
                <div className="grid gap-8 lg:grid-cols-[minmax(0,1.3fr)_minmax(0,0.9fr)] lg:items-start lg:gap-10">
                  <form onSubmit={onCreateSet} className="space-y-8">
                    <div className="space-y-5">
                      <div>
                        <h3 className="text-base font-semibold text-white">Set details</h3>
                        <p className="mt-1 text-xs leading-5 text-white/48">Give learners a clear, useful preview of what this set covers.</p>
                      </div>
                      <label className="block">
                        <span className="mb-2 flex items-center justify-between gap-2 text-xs font-medium text-white/62">
                          <span>Name</span>
                          <span className="text-[10px] text-white/45">{normalizedSetTitle.length}/70</span>
                        </span>
                        <input
                          value={setTitle}
                          onChange={(event) => setSetTitle(event.target.value)}
                          maxLength={70}
                          placeholder="Example: Organic Chemistry Reactions"
                          className="h-11 w-full rounded-xl bg-white/[0.08] px-3 text-sm text-white outline-none placeholder:text-white/32 transition-colors focus:bg-white/[0.12]"
                        />
                      </label>

                      <label className="block">
                        <span className="mb-2 flex items-center justify-between gap-2 text-xs font-medium text-white/62">
                          <span>Description</span>
                          <span className={`text-[10px] ${shouldShowDescriptionError ? "text-[#ff9b9b]" : descriptionHasTooFewChars ? "text-white/45" : "text-[#9ef8cb]"}`}>
                            {normalizedSetDescription.length} / {MIN_SET_DESCRIPTION_LENGTH} min
                          </span>
                        </span>
                        <textarea
                          value={setDescription}
                          onChange={(event) => setSetDescription(event.target.value)}
                          placeholder="What does this set cover and who is it for?"
                          aria-invalid={shouldShowDescriptionError}
                          aria-describedby={shouldShowDescriptionError ? "community-set-description-error" : undefined}
                          className={`h-24 w-full resize-none rounded-xl px-3 py-2.5 text-sm text-white outline-none placeholder:text-white/32 transition-colors ${
                            shouldShowDescriptionError ? "bg-red-400/[0.09] focus:bg-red-400/[0.13]" : "bg-white/[0.08] focus:bg-white/[0.12]"
                          }`}
                        />
                        {shouldShowDescriptionError ? (
                          <p id="community-set-description-error" className="mt-1.5 text-[11px] text-[#ff9b9b]">
                            Description must be at least {MIN_SET_DESCRIPTION_LENGTH} characters. Add {descriptionCharsRemaining} more
                            {descriptionCharsRemaining === 1 ? " character." : " characters."}
                          </p>
                        ) : null}
                      </label>

                      <label className="block">
                        <span className="mb-2 flex items-center justify-between gap-2 text-xs font-medium text-white/62">
                          <span>Tags</span>
                          <span className={`text-[10px] ${tagLimitError ? "text-[#ff9b9b]" : "text-white/45"}`}>
                            {parsedSetTags.length}/{MAX_SET_TAGS}
                          </span>
                        </span>
                        <input
                          value={setTags}
                          onChange={onSetTagsChange}
                          placeholder={hasMaxTags ? "Max tags reached. Edit or remove one to add another." : "chemistry, reaction mechanisms, exam prep"}
                          aria-invalid={tagLimitError}
                          aria-describedby="community-set-tags-help"
                          className={`h-11 w-full rounded-xl px-3 text-sm text-white outline-none placeholder:text-white/32 transition-colors ${
                            tagLimitError ? "bg-red-400/[0.09] focus:bg-red-400/[0.13]" : "bg-white/[0.08] focus:bg-white/[0.12]"
                          }`}
                        />
                        <p id="community-set-tags-help" className={`mt-1.5 text-[11px] ${tagLimitError ? "text-[#ff9b9b]" : "text-white/42"}`}>
                          {tagLimitError
                            ? `You can add up to ${MAX_SET_TAGS} tags. Remove one to add another.`
                            : hasMaxTags
                              ? `Max tags reached (${MAX_SET_TAGS}). Edit existing tags or remove one to add another.`
                              : "Add commas to add new tags."}
                        </p>
                        {parsedSetTags.length > 0 ? (
                          <div className="mt-2 flex flex-wrap gap-1.5">
                            {parsedSetTags.map((tag) => (
                              <button
                                key={`create-tag-${tag}`}
                                type="button"
                                onClick={() => onRemoveSetTag(tag)}
                                className="inline-flex items-center gap-1 rounded-full bg-white/[0.09] px-2.5 py-1 text-[10px] font-medium text-white/72 transition-colors hover:bg-white/[0.07] hover:text-white"
                                aria-label={`Remove tag ${tag}`}
                              >
                                <span>#{tag}</span>
                                <i className="fa-solid fa-xmark text-[8px]" aria-hidden="true" />
                              </button>
                            ))}
                          </div>
                        ) : null}
                      </label>

                      <div>
                        <p className="mb-2 block text-xs font-medium text-white/62">Cover image</p>
                        <input id="community-set-thumbnail" type="file" accept="image/*" className="peer sr-only" onChange={onThumbnailFileChange} />
                        <label
                          htmlFor="community-set-thumbnail"
                          onDragEnter={onThumbnailDragEnter}
                          onDragOver={onThumbnailDragOver}
                          onDragLeave={onThumbnailDragLeave}
                          onDrop={onThumbnailDrop}
                          className={`group relative block h-[190px] w-full cursor-pointer overflow-hidden rounded-xl transition-colors peer-focus-visible:bg-white/[0.12] ${
                            isThumbnailDragOver ? "bg-white/[0.14]" : "bg-white/[0.055] hover:bg-white/[0.07]"
                          } sm:h-[220px]`}
                        >
                          {thumbnailPreview ? (
                            <img
                              src={thumbnailPreview}
                              alt="Set thumbnail preview"
                              className="h-full w-full object-cover"
                            />
                          ) : (
                            <div className="grid h-full w-full place-items-center text-white/48 transition-colors group-hover:text-white/68">
                              <i className="fa-regular fa-image -translate-y-5 text-lg" aria-hidden="true" />
                            </div>
                          )}
                          <span className={`absolute inset-0 grid place-items-center text-white/85 ${thumbnailPreview ? "bg-black/45" : ""}`}>
                            <span className="flex max-w-[90%] translate-y-5 flex-col items-center text-center">
                              <span className={`truncate text-sm font-medium ${thumbnailPreview ? "text-white" : "text-white/78"}`}>
                                {thumbnailPreview ? thumbnailFileName || "Image selected" : "Drop an image here"}
                              </span>
                              <span className="mt-1 text-xs text-white/58">
                                {thumbnailPreview ? "Click to replace" : "or click to browse · PNG, JPG, WEBP"}
                              </span>
                            </span>
                          </span>
                        </label>
                        <p className="mt-2 text-[11px] text-white/42">A vertical image works best in mobile previews.</p>
                        {thumbnailPreview ? (
                          <button
                            type="button"
                            onClick={() => {
                              setThumbnailPreview("");
                              setThumbnailFileName("");
                            }}
                            className="mt-2 inline-flex h-8 items-center gap-1.5 rounded-lg px-2 text-xs font-medium text-white/52 transition-colors hover:bg-white/[0.07] hover:text-white"
                          >
                            <i className="fa-solid fa-trash text-[9px]" aria-hidden="true" />
                            Remove
                          </button>
                        ) : null}
                      </div>
                    </div>

                    <div className="space-y-4">
                      <div className="flex flex-wrap items-end justify-between gap-3">
                        <div>
                          <h3 className="text-base font-semibold text-white">Reels</h3>
                          <p className="mt-1 text-xs leading-5 text-white/48">Paste links from {SUPPORTED_PLATFORMS_LABEL}. Add at least one to publish.</p>
                        </div>
                        <p className={`text-xs ${invalidDraftReelCount > 0 ? "text-[#ff9b9b]" : "text-white/45"}`}>
                          {validDraftReelCount} ready{invalidDraftReelCount > 0 ? ` · ${invalidDraftReelCount} needs attention` : ""}
                        </p>
                      </div>
                      <div className="space-y-5">
                        {parsedDraftReels.map((row, index) => {
                          const hasInput = Boolean(row.value.trim());
                          const hasValidEmbed = row.parsed !== null;
                          const hasValidRange = !row.hasClipRangeError;
                          const durationState = reelDurationByRow[row.id];
                          const detectedDurationSec = durationState?.durationSec;
                          const hasDetectedDuration =
                            Number.isFinite(detectedDurationSec) && Number(detectedDurationSec) > CLIP_SLIDER_MIN_GAP_SEC;
                          const sliderMaxSec = hasDetectedDuration ? Number(detectedDurationSec) : null;
                          const sliderStartSec =
                            sliderMaxSec === null
                              ? 0
                              : Math.min(
                                  Math.max(0, row.clipStartSec ?? 0),
                                  Math.max(0, sliderMaxSec - CLIP_SLIDER_MIN_GAP_SEC),
                                );
                          const sliderEndSec =
                            sliderMaxSec === null
                              ? 0
                              : Math.min(
                                  sliderMaxSec,
                                  Math.max(
                                    sliderStartSec + CLIP_SLIDER_MIN_GAP_SEC,
                                    row.clipEndSec ?? sliderMaxSec,
                                  ),
                                );
                          const sliderStartPercent = sliderMaxSec === null ? 0 : (sliderStartSec / sliderMaxSec) * 100;
                          const sliderEndPercent = sliderMaxSec === null ? 0 : (sliderEndSec / sliderMaxSec) * 100;
                          const shouldShowSlider = hasInput && hasValidEmbed && sliderMaxSec !== null;
                          const isDurationLoading = hasInput && hasValidEmbed && (durationState?.loading ?? true);
                          return (
                            <div key={row.id} data-create-set-reel-row="true">
                              <div className="flex flex-col items-stretch gap-2 sm:flex-row sm:items-center sm:gap-3">
                                <input
                                  value={row.value}
                                  onChange={(event) => updateReelInputRow(row.id, event.target.value)}
                                  placeholder="Paste YouTube, Instagram reel, or TikTok URL"
                                  className="h-11 w-full rounded-xl bg-white/[0.08] px-3 text-sm text-white outline-none placeholder:text-white/32 transition-colors focus:bg-white/[0.12]"
                                />
                                <button
                                  type="button"
                                  onClick={() => removeReelInputRow(row.id)}
                                  className="inline-flex h-10 w-full shrink-0 items-center justify-center gap-1 rounded-xl bg-white/[0.055] text-white/52 transition-colors hover:bg-white/[0.07] hover:text-white sm:grid sm:h-11 sm:w-11 sm:place-items-center"
                                  aria-label={`Remove reel input ${index + 1}`}
                                >
                                  <i className="fa-solid fa-xmark text-xs" aria-hidden="true" />
                                  <span className="text-[10px] font-semibold uppercase tracking-[0.08em] sm:hidden">Remove</span>
                                </button>
                              </div>

                              {shouldShowSlider ? (
                                <div className="mt-2">
                                  <div className="flex items-center justify-between gap-2 text-[10px] font-semibold uppercase tracking-[0.08em] text-white/58">
                                    <span>Clip Range</span>
                                    <span>
                                      {formatClipSecondsInputValue(sliderStartSec)}s - {formatClipSecondsInputValue(sliderEndSec)}s
                                    </span>
                                  </div>
                                  <div className="relative mt-1.5 h-8">
                                    <div className="pointer-events-none absolute left-0 right-0 top-1/2 h-1.5 -translate-y-1/2 rounded-full bg-white/25" />
                                    <div
                                      className="pointer-events-none absolute top-1/2 h-1.5 -translate-y-1/2 rounded-full bg-white"
                                      style={{
                                        left: `${sliderStartPercent}%`,
                                        width: `${Math.max(0, sliderEndPercent - sliderStartPercent)}%`,
                                      }}
                                    />
                                    <input
                                      type="range"
                                      min={0}
                                      max={sliderMaxSec}
                                      step={CLIP_SLIDER_STEP_SEC}
                                      value={sliderStartSec}
                                      onChange={(event) => updateReelClipStartFromSlider(row.id, Number(event.target.value), sliderMaxSec)}
                                      className="dual-range-input absolute inset-0 h-8 w-full"
                                      aria-label={`Clip start for reel ${index + 1}`}
                                    />
                                    <input
                                      type="range"
                                      min={0}
                                      max={sliderMaxSec}
                                      step={CLIP_SLIDER_STEP_SEC}
                                      value={sliderEndSec}
                                      onChange={(event) => updateReelClipEndFromSlider(row.id, Number(event.target.value), sliderMaxSec)}
                                      className="dual-range-input absolute inset-0 h-8 w-full"
                                      aria-label={`Clip end for reel ${index + 1}`}
                                    />
                                  </div>
                                  <div className="mt-1 flex items-center justify-between text-[10px] text-white/50">
                                    <span>0s</span>
                                    <span>{formatClipSecondsInputValue(sliderMaxSec)}s</span>
                                  </div>
                                </div>
                              ) : null}
                              {isDurationLoading ? (
                                <p className="mt-2 text-[10px] text-white/55">Detecting video length...</p>
                              ) : null}
                              {hasInput && hasValidEmbed && !isDurationLoading && !hasDetectedDuration ? (
                                <p className="mt-2 text-[10px] text-[#ffb4b4]">
                                  Could not detect video length for this link yet. You can still post it, but trim sliders stay disabled until detection succeeds.
                                </p>
                              ) : null}

                              {hasInput && !hasValidEmbed ? (
                                <p className="mt-2 text-[11px] text-[#ffb4b4]">Invalid URL. Supported: {SUPPORTED_PLATFORMS_LABEL}.</p>
                              ) : null}
                              {hasInput && hasValidEmbed && !hasValidRange ? (
                                <p className="mt-2 text-[11px] text-[#ffb4b4]">Invalid clip range. If set, end must be greater than start.</p>
                              ) : null}

                              {row.parsed ? (
                                <div className="mt-3 overflow-hidden rounded-xl bg-black/30">
                                  <div className="flex items-center gap-1.5 px-2 py-1.5 text-[10px] font-semibold uppercase tracking-[0.08em] text-white/70">
                                    <i className={PLATFORM_ICON[row.parsed.platform]} aria-hidden="true" />
                                    {PLATFORM_LABEL[row.parsed.platform]} embed
                                    {hasValidRange ? (
                                      <span className="ml-1 text-white/55">
                                        {row.clipEndSec !== null
                                          ? `(${row.clipStartSec?.toFixed(1).replace(/\.0$/, "")}s - ${row.clipEndSec?.toFixed(1).replace(/\.0$/, "")}s)`
                                          : `(${row.clipStartSec?.toFixed(1).replace(/\.0$/, "")}s - full video)`}
                                      </span>
                                    ) : null}
                                  </div>
                                  <iframe
                                    data-create-set-reel-preview="true"
                                    src={row.parsed.embedUrl}
                                    title={`${PLATFORM_LABEL[row.parsed.platform]} reel preview`}
                                    className="h-[320px] w-full border-0 sm:h-[360px]"
                                    loading="lazy"
                                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                                    allowFullScreen
                                  />
                                </div>
                              ) : null}
                              {index < parsedDraftReels.length - 1 ? (
                                <div
                                  data-create-set-reel-divider="true"
                                  aria-hidden="true"
                                  className="mt-5 h-px bg-white/[0.08]"
                                />
                              ) : null}
                            </div>
                          );
                        })}
                      </div>
                      <button
                        type="button"
                        onClick={addReelInputRow}
                        className="inline-flex h-9 items-center gap-1.5 rounded-lg bg-white/[0.07] px-3 text-xs font-medium text-white/72 transition-colors hover:bg-white/[0.07] hover:text-white"
                      >
                        <i className="fa-solid fa-plus text-[9px]" aria-hidden="true" />
                        Add another reel
                      </button>
                    </div>

                    {isFormEditMode ? (
                      <div className="grid gap-2">
                        <button
                          type="submit"
                          disabled={!canPostSet}
                          className={`inline-flex h-11 w-full items-center justify-center rounded-xl px-4 text-sm font-semibold transition-colors ${
                            canPostSet
                              ? "bg-white text-black hover:bg-white/90"
                              : "cursor-not-allowed bg-white/[0.09] text-white/32"
                          }`}
                        >
                          {isPostingSet ? "Saving..." : "Save Set Changes"}
                        </button>
                      </div>
                    ) : (
                      <div data-create-set-actions="true" className="grid gap-2 sm:grid-cols-[auto_auto_minmax(12rem,1fr)]">
                        <button
                          type="button"
                          onClick={onClearCreateProgress}
                          className="inline-flex h-11 w-full items-center justify-center rounded-xl px-4 text-sm font-medium text-white/52 transition-colors hover:bg-white/[0.07] hover:text-white"
                        >
                          Clear
                        </button>
                        <button
                          type="button"
                          onClick={onSaveDraftProgress}
                          className="inline-flex h-11 w-full items-center justify-center rounded-xl bg-white/[0.08] px-4 text-sm font-medium text-white/82 transition-colors hover:bg-white/[0.07] hover:text-white"
                        >
                          Save draft
                        </button>
                        <button
                          type="submit"
                          disabled={!canPostSet}
                          className={`inline-flex h-11 w-full items-center justify-center rounded-xl px-5 text-sm font-semibold transition-colors ${
                            canPostSet
                              ? "bg-white text-black hover:bg-white/90"
                              : "cursor-not-allowed bg-white/[0.09] text-white/32"
                          }`}
                        >
                          {isPostingSet ? "Posting..." : "Post set"}
                        </button>
                      </div>
                    )}
                  </form>

                  <aside className="rounded-2xl bg-white/[0.035] p-4 sm:p-5 lg:sticky lg:top-3">
                    <div className="flex items-center justify-between gap-3">
                      <p className="text-sm font-medium text-white/72">Preview</p>
                      <span className={`text-xs font-medium ${canPostSet ? "text-[#9ef8cb]" : "text-white/42"}`}>
                        {canPostSet ? (isFormEditMode ? "Ready to save" : "Ready to post") : "Draft"}
                      </span>
                    </div>
                    <div className="mt-3 overflow-hidden rounded-xl bg-white/[0.055]">
                      {thumbnailPreview ? (
                        <img
                          src={thumbnailPreview}
                          alt="Draft set cover"
                          className="h-[200px] w-full object-cover"
                        />
                      ) : (
                        <div className="grid h-[200px] w-full place-items-center text-white/42">
                          <i className="fa-regular fa-image text-lg" aria-hidden="true" />
                        </div>
                      )}
                    </div>
                    <h3 className="mt-4 text-lg font-semibold leading-tight text-white">
                      {normalizedSetTitle || "Your set title"}
                    </h3>
                    <p className="mt-2 text-sm leading-relaxed text-white/64">
                      {normalizedSetDescription || "Add a description to show what learners will get from this set."}
                    </p>
                    {parsedSetTags.length > 0 ? (
                      <div className="mt-3 flex flex-wrap gap-1.5">
                        {parsedSetTags.map((tag) => (
                          <span key={`preview-tag-${tag}`} className="rounded-full bg-white/[0.09] px-2.5 py-1 text-[10px] font-medium text-white/68">
                            #{tag}
                          </span>
                        ))}
                      </div>
                    ) : null}
                    <div className="mt-5 rounded-xl bg-white/[0.045] px-3.5 py-3.5">
                      <div className="flex items-center justify-between gap-3">
                        <p className="text-xs font-medium text-white/68">
                          {remainingPreviewRequirements.length > 0
                            ? isFormEditMode ? "Before you save" : "Before you post"
                            : "Ready when you are"}
                        </p>
                        <span className="text-[11px] text-white/42">{requiredCompletionCount}/4</span>
                      </div>
                      {remainingPreviewRequirements.length > 0 ? (
                        <ul className="mt-3 space-y-2">
                          {remainingPreviewRequirements.map((item) => (
                            <li key={`remaining-requirement-${item}`} className="flex items-start gap-2.5 text-xs leading-5 text-white/72">
                              <i className="fa-regular fa-circle mt-[6px] text-[7px] text-white/38" aria-hidden="true" />
                              <span>{item}</span>
                            </li>
                          ))}
                        </ul>
                      ) : (
                        <p className="mt-2 text-xs text-[#9ef8cb]">All required details are complete.</p>
                      )}
                    </div>
                  </aside>
                </div>
              </section>
                ) : null}
                  </>
                )}
            </div>
          </div>
          </div>
          {shouldShowEditSetGrid && canManageYourSets ? (
            <button
              type="button"
              onClick={onOpenCreateSetFromGrid}
              data-floating-create-set
              aria-label="Create set"
              title="Create set"
              className="absolute bottom-[max(40px,env(safe-area-inset-bottom))] right-[max(40px,env(safe-area-inset-right))] z-30 grid h-14 w-14 place-items-center rounded-full bg-white text-black transition-colors duration-300 hover:bg-white/90"
            >
              <svg viewBox="0 0 24 24" aria-hidden="true" className="h-6 w-6 fill-none stroke-current" strokeWidth="1.5" strokeLinecap="round">
                <path d="M12 5v14M5 12h14" />
              </svg>
            </button>
          ) : null}
      </>
    )}
      <FadePresence show={Boolean(draftActionConfirmModal)}>
        {(modalVisible) => draftActionConfirmModal ? (
        <ViewportModalPortal>
          <div
            className={`fixed inset-0 z-[128] flex items-center justify-center overflow-y-auto bg-black/70 px-4 py-6 transition-opacity duration-300 motion-reduce:transition-none ${
              modalVisible ? "opacity-100" : "opacity-0"
            }`}
            role="presentation"
            onClick={closeDraftActionConfirmModal}
          >
            <div
              role="dialog"
              aria-modal="true"
              aria-label="Draft action confirmation"
              className="w-full max-w-xl rounded-[14px] bg-[#202020] p-5 text-white md:p-6"
              onClick={(event) => event.stopPropagation()}
            >
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="text-[11px] font-semibold uppercase tracking-[0.12em] text-white/65">{draftActionConfirmModal.label}</p>
                  <h3 className="mt-2 text-lg font-semibold text-white">{draftActionConfirmModal.title}</h3>
                </div>
                <button
                  type="button"
                  onClick={closeDraftActionConfirmModal}
                  aria-label="Close"
                  disabled={isPostingSet}
                  className="inline-flex h-8 w-8 items-center justify-center rounded-lg text-white/80 transition-colors hover:bg-white/[0.07] hover:text-white focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-55"
                >
                  <svg viewBox="0 0 20 20" aria-hidden="true" className="h-4 w-4 fill-none stroke-current stroke-2">
                    <path d="M5 5L15 15M15 5L5 15" strokeLinecap="round" />
                  </svg>
                </button>
              </div>
              <p className="mt-4 rounded-2xl px-4 py-3 text-sm text-white/88">
                {draftActionConfirmModal.message}
              </p>
              <div className="mt-4 flex flex-wrap items-center justify-end gap-2">
                <button
                  type="button"
                  onClick={closeDraftActionConfirmModal}
                  disabled={isPostingSet}
                  className="inline-flex min-w-[9rem] items-center justify-center whitespace-nowrap rounded-xl bg-black/35 px-5 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-white/[0.07] disabled:cursor-not-allowed disabled:opacity-55"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={confirmDraftAction}
                  disabled={isPostingSet}
                  className="inline-flex min-w-[9rem] items-center justify-center whitespace-nowrap rounded-xl bg-white px-5 py-2.5 text-sm font-semibold text-black transition-colors hover:bg-white/90 disabled:cursor-not-allowed disabled:opacity-70"
                >
                  {isPostingSet && draftActionConfirmModal.action === "save-set-changes"
                    ? "Saving..."
                    : draftActionConfirmModal.confirmLabel}
                </button>
              </div>
            </div>
          </div>
        </ViewportModalPortal>
        ) : null}
      </FadePresence>
      <FadePresence show={Boolean(unsavedDraftExitModal)}>
        {(modalVisible) => unsavedDraftExitModal ? (
        <ViewportModalPortal>
          <div
            className={`fixed inset-0 z-[127] flex items-center justify-center overflow-y-auto bg-black/70 px-4 py-6 transition-opacity duration-300 motion-reduce:transition-none ${
              modalVisible ? "opacity-100" : "opacity-0"
            }`}
            role="presentation"
            onClick={closeUnsavedDraftExitModal}
          >
            <div
              role="dialog"
              aria-modal="true"
              aria-label="Unsaved set draft changes"
              className="w-full max-w-xl rounded-[14px] bg-[#202020] p-5 text-white md:p-6"
              onClick={(event) => event.stopPropagation()}
            >
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="text-[11px] font-semibold uppercase tracking-[0.12em] text-white/65">Unsaved changes</p>
                  <h3 className="mt-2 text-lg font-semibold text-white">Save set changes before leaving?</h3>
                </div>
                <button
                  type="button"
                  onClick={closeUnsavedDraftExitModal}
                  aria-label="Close"
                  className="inline-flex h-8 w-8 items-center justify-center rounded-lg text-white/80 transition-colors hover:bg-white/[0.07] hover:text-white focus-visible:outline-none"
                >
                  <svg viewBox="0 0 20 20" aria-hidden="true" className="h-4 w-4 fill-none stroke-current stroke-2">
                    <path d="M5 5L15 15M15 5L5 15" strokeLinecap="round" />
                  </svg>
                </button>
              </div>
              <p className="mt-4 rounded-2xl px-4 py-3 text-sm text-white/88">
                {isFormEditMode
                  ? "Save to update this set, or discard these edits and continue."
                  : "Save to keep your draft progress, or discard these edits and continue."}
              </p>
              <div className="mt-4 flex flex-wrap items-center justify-end gap-2">
                <button
                  type="button"
                  onClick={confirmUnsavedDraftExitDiscard}
                  disabled={isPostingSet}
                  className="inline-flex min-w-[9rem] items-center justify-center whitespace-nowrap rounded-xl bg-black/35 px-5 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-white/[0.07] disabled:cursor-not-allowed disabled:opacity-60"
                >
                  Discard
                </button>
                <button
                  type="button"
                  onClick={confirmUnsavedDraftExitSave}
                  disabled={isPostingSet}
                  className="inline-flex min-w-[9rem] items-center justify-center whitespace-nowrap rounded-xl bg-white px-5 py-2.5 text-sm font-semibold text-black transition-colors hover:bg-white/90 disabled:cursor-not-allowed disabled:opacity-70"
                >
                  {isPostingSet ? "Saving..." : "Save"}
                </button>
              </div>
            </div>
          </div>
        </ViewportModalPortal>
        ) : null}
      </FadePresence>
      <FadePresence show={Boolean(deleteSetConfirmModal)}>
        {(modalVisible) => deleteSetConfirmModal ? (
        <ViewportModalPortal>
          <div
            className={`fixed inset-0 z-[126] flex items-center justify-center overflow-y-auto bg-black/70 px-4 py-6 transition-opacity duration-300 motion-reduce:transition-none ${
              modalVisible ? "opacity-100" : "opacity-0"
            }`}
            role="presentation"
            onClick={closeDeleteSetConfirmModal}
          >
            <div
              role="dialog"
              aria-modal="true"
              aria-label={deleteSetConfirmModal.setIds.length === 1 ? "Delete set confirmation" : "Delete sets confirmation"}
              className="w-full max-w-xl rounded-[14px] bg-[#202020] p-5 text-white md:p-6"
              onClick={(event) => event.stopPropagation()}
            >
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="text-[11px] font-semibold uppercase tracking-[0.12em] text-white/65">
                    {deleteSetConfirmModal.setIds.length === 1 ? "Delete Set" : "Delete Sets"}
                  </p>
                  <h3 className="mt-2 text-lg font-semibold text-white">
                    {deleteSetConfirmModal.setIds.length === 1
                      ? `Delete "${deleteSetConfirmModal.title}"?`
                      : `Delete ${deleteSetConfirmModal.setIds.length} selected sets?`}
                  </h3>
                </div>
                <button
                  type="button"
                  onClick={closeDeleteSetConfirmModal}
                  aria-label="Close"
                  disabled={isDeletingSets}
                  className="inline-flex h-8 w-8 items-center justify-center rounded-lg text-white/80 transition-colors hover:bg-white/[0.07] hover:text-white focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-55"
                >
                  <svg viewBox="0 0 20 20" aria-hidden="true" className="h-4 w-4 fill-none stroke-current stroke-2">
                    <path d="M5 5L15 15M15 5L5 15" strokeLinecap="round" />
                  </svg>
                </button>
              </div>
              <p className="mt-4 rounded-2xl px-4 py-3 text-sm text-white/88">
                This action cannot be undone.
              </p>
              <div className="mt-4 flex flex-wrap items-center justify-end gap-2">
                <button
                  type="button"
                  onClick={closeDeleteSetConfirmModal}
                  disabled={isDeletingSets}
                  className="inline-flex min-w-[9rem] items-center justify-center whitespace-nowrap rounded-xl bg-black/35 px-5 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-white/[0.07] disabled:cursor-not-allowed disabled:opacity-55"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={() => {
                    void onDeleteEditableSets(deleteSetConfirmModal.setIds);
                  }}
                  disabled={isDeletingSets}
                  className="inline-flex min-w-[9rem] items-center justify-center whitespace-nowrap rounded-xl bg-white px-5 py-2.5 text-sm font-semibold text-black transition-colors hover:bg-white/90 disabled:cursor-not-allowed disabled:opacity-70"
                >
                  {isDeletingSets ? "Deleting..." : "Delete"}
                </button>
              </div>
            </div>
          </div>
        </ViewportModalPortal>
        ) : null}
      </FadePresence>
      <FadePresence show={Boolean(publishResultModal)}>
        {(modalVisible) => publishResultModal ? (
        <ViewportModalPortal>
          <div
            className={`fixed inset-0 z-[125] flex items-center justify-center overflow-y-auto bg-black/70 px-4 py-6 transition-opacity duration-300 motion-reduce:transition-none ${
              modalVisible ? "opacity-100" : "opacity-0"
            }`}
            role="presentation"
            onClick={() => setPublishResultModal(null)}
          >
            <div
              role="dialog"
              aria-modal="true"
              aria-label="Publish result"
              className="w-full max-w-xl rounded-[14px] bg-[#202020] p-5 text-white backdrop-blur-2xl md:p-6"
              onClick={(event) => event.stopPropagation()}
            >
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="text-[11px] font-semibold uppercase tracking-[0.12em] text-white/65">{publishResultModal.label ?? "Post Set"}</p>
                  <h3 className="mt-2 text-lg font-semibold text-white">{publishResultModal.title}</h3>
                </div>
                <button
                  type="button"
                  onClick={() => setPublishResultModal(null)}
                  aria-label="Close publish result"
                  className="inline-flex h-8 w-8 items-center justify-center rounded-lg text-white/80 transition-colors hover:bg-white/[0.07] hover:text-white focus-visible:outline-none"
                >
                  <svg viewBox="0 0 20 20" aria-hidden="true" className="h-4 w-4 fill-none stroke-current stroke-2">
                    <path d="M5 5L15 15M15 5L5 15" strokeLinecap="round" />
                  </svg>
                </button>
              </div>
              <p className="mt-4 rounded-2xl px-4 py-3 text-sm text-white/88">
                {publishResultModal.message}
              </p>
              {publishResultModal.status === "success" && publishResultModal.thumbnailUrl ? (
                <div className="mt-4 overflow-hidden rounded-2xl border border-white/15 bg-black/45">
                  <img
                    src={publishResultModal.thumbnailUrl}
                    alt={publishResultModal.thumbnailAlt || "Published set thumbnail"}
                    className="h-52 w-full object-cover"
                  />
                </div>
              ) : null}
              <div className="mt-4 flex items-center justify-end">
                <button
                  type="button"
                  onClick={() => setPublishResultModal(null)}
                  className={`inline-flex min-w-[8rem] items-center justify-center whitespace-nowrap rounded-xl px-5 py-2.5 text-sm font-semibold transition-colors ${
                    publishResultModal.status === "success"
                      ? "bg-white text-black hover:bg-white/90"
                      : "bg-black/35 text-white hover:bg-white/[0.07]"
                  }`}
                >
                  OK
                </button>
              </div>
            </div>
          </div>
        </ViewportModalPortal>
        ) : null}
      </FadePresence>
    </div>
  );
}
