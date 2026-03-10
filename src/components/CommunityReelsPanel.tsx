"use client";

import { type ChangeEvent, type DragEvent, type FormEvent, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useRouter } from "next/navigation";

import {
  createCommunitySet,
  deleteCommunitySet,
  fetchCommunityReelDuration,
  fetchCommunitySets,
  readOwnedCommunitySetIds,
  saveOwnedCommunitySetIds,
  updateCommunitySet,
} from "@/lib/api";
import { loadYouTubeIframeApi } from "@/lib/youtubeIframeApi";

const COMMUNITY_SETS_STORAGE_KEY = "studyreels-community-sets";
const COMMUNITY_CREATE_DRAFT_STORAGE_KEY = "studyreels-community-create-draft";
const COMMUNITY_EDIT_DRAFT_PREFIX = "studyreels-community-edit-draft-";
const USER_CREATED_SET_ID_PREFIX = "user-set-";
const MAX_USER_SETS = 120;
const FALLBACK_THUMBNAIL_URL = "/images/community/ai-systems.svg";
const SUPPORTED_PLATFORMS_LABEL = "YouTube, Instagram, TikTok";
const MIN_SET_DESCRIPTION_LENGTH = 18;
const MAX_SET_TAGS = 6;
const FEATURED_CAROUSEL_INTERVAL_MS = 5200;
const FEATURED_CAROUSEL_TRANSITION_MS = 520;
const FEATURED_CAROUSEL_PAUSE_MS = 200;
const FEATURED_CAROUSEL_CONTENT_MIN_HEIGHT_FALLBACK = 410;
const FEATURED_CAROUSEL_CONTENT_MIN_HEIGHT_TOUCH_FALLBACK = 280;
const FEATURED_CAROUSEL_BUTTON_BOTTOM_MARGIN_PX = 18;
const FEATURED_CAROUSEL_IMAGE_BOTTOM_MARGIN_PX = 18;
const DIRECTORY_DETAIL_TRANSITION_MS = 440;
const DETAIL_CONTENT_TOP_PADDING_FALLBACK = 420;
const DETAIL_CONTENT_TOP_PADDING_GUTTER = 16;
const DETAIL_CONTENT_TOP_PADDING_UPSHIFT_PX = 56;
const DETAIL_REEL_CAROUSEL_INTERVAL_MS = 5200;
const DETAIL_BANNER_LEFT_EXPANSION_PX = 10;
const DETAIL_BANNER_LEFT_INSET_PX = 8;
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
  value: string;
  tStartSec: string;
  tEndSec: string;
};

type ParsedDraftReel = {
  id: string;
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
  setId: string;
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

function createDraftReelRow(value = "", tStartSec = "0", tEndSec = ""): DraftReelInput {
  draftRowCounter += 1;
  return {
    id: `draft-reel-${draftRowCounter}`,
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

const DETAIL_LOREM_PARAGRAPHS = [
  "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Integer luctus, lorem ut porta vehicula, lectus lectus viverra mi, id faucibus turpis est eget augue.",
  "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Velit euismod in pellentesque massa placerat duis ultricies lacus sed turpis tincidunt id aliquet.",
  "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Egestas integer eget aliquet nibh praesent tristique magna sit.",
  "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Cras semper auctor neque vitae tempus quam pellentesque nec nam aliquam.",
  "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. Convallis aenean et tortor at risus viverra adipiscing at in tellus.",
  "Nunc consequat interdum varius sit amet mattis vulputate enim nulla aliquet porttitor lacus luctus accumsan tortor posuere ac ut consequat semper viverra nam libero justo.",
  "Purus gravida quis blandit turpis cursus in hac habitasse platea dictumst quisque sagittis purus sit amet volutpat consequat mauris nunc congue nisi vitae suscipit tellus.",
  "Aliquet sagittis id consectetur purus ut faucibus pulvinar elementum integer enim neque volutpat ac tincidunt vitae semper quis lectus nulla at volutpat diam ut venenatis.",
  "Pharetra pharetra massa massa ultricies mi quis hendrerit dolor magna eget est lorem ipsum dolor sit amet consectetur adipiscing elit pellentesque habitant morbi tristique.",
  "Amet dictum sit amet justo donec enim diam vulputate ut pharetra sit amet aliquam id diam maecenas ultricies mi eget mauris pharetra et ultrices neque ornare aenean.",
];

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

function formatLastEditedLabel(set: Pick<CommunitySet, "updatedAt" | "updatedLabel">, nowMs: number): string {
  const updatedMs = parseTimestampMs(set.updatedAt);
  if (updatedMs != null) {
    return `Last Edited: ${formatRelativeElapsed(nowMs - updatedMs)}`;
  }
  return normalizeLastEditedLabel(set.updatedLabel);
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
  isVisible?: boolean;
  onDetailOpenChange?: (isOpen: boolean) => void;
  initialOpenSetId?: string | null;
  communityResetSignal?: number;
  onDraftUnsavedChangesChange?: (hasUnsavedChanges: boolean) => void;
  onDraftExitActionsChange?: (actions: CommunityDraftExitActions | null) => void;
  onOpenCommunityReelInFeed?: (payload: { setId: string; setTitle: string; selectedReelId: string; feedQuery: string }) => void;
};

type FeaturedTransitionStage = "idle" | "exiting" | "pause" | "entering";

export function CommunityReelsPanel({
  mode = "community",
  isVisible = true,
  onDetailOpenChange,
  initialOpenSetId = null,
  communityResetSignal = 0,
  onDraftUnsavedChangesChange,
  onDraftExitActionsChange,
  onOpenCommunityReelInFeed,
}: CommunityReelsPanelProps) {
  const router = useRouter();
  const [activeCommunityCategory, setActiveCommunityCategory] = useState("Featured");
  const [activeFeaturedIndex, setActiveFeaturedIndex] = useState(0);
  const [leavingFeaturedIndex, setLeavingFeaturedIndex] = useState<number | null>(null);
  const [pendingFeaturedIndex, setPendingFeaturedIndex] = useState<number | null>(null);
  const [featuredTransitionStage, setFeaturedTransitionStage] = useState<FeaturedTransitionStage>("idle");
  const [featuredTransitionDirection, setFeaturedTransitionDirection] = useState<1 | -1>(1);
  const [featuredCarouselContentHeight, setFeaturedCarouselContentHeight] = useState(FEATURED_CAROUSEL_CONTENT_MIN_HEIGHT_FALLBACK);
  const [selectedDirectorySet, setSelectedDirectorySet] = useState<CommunitySet | null>(null);
  const [isDirectoryDetailOpen, setIsDirectoryDetailOpen] = useState(false);
  const [query, setQuery] = useState("");
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
  const [deletingSetId, setDeletingSetId] = useState<string | null>(null);
  const [userSets, setUserSets] = useState<CommunitySet[]>([]);
  const [ownedSetIds, setOwnedSetIds] = useState<string[]>([]);
  const [storageHydrated, setStorageHydrated] = useState(false);
  const [portalReady, setPortalReady] = useState(false);
  const [relativeTimeNowMs, setRelativeTimeNowMs] = useState(() => Date.now());
  const [detailBannerLeft, setDetailBannerLeft] = useState(0);
  const [detailBannerRight, setDetailBannerRight] = useState(0);
  const [detailBannerHeight, setDetailBannerHeight] = useState(0);
  const [isDetailBannerCompact, setIsDetailBannerCompact] = useState(false);
  const [skipDetailTransitionOnce, setSkipDetailTransitionOnce] = useState(false);
  const [isThumbnailDragOver, setIsThumbnailDragOver] = useState(false);
  const [thumbnailFileName, setThumbnailFileName] = useState("");
  const panelRootRef = useRef<HTMLDivElement | null>(null);
  const detailBannerRef = useRef<HTMLDivElement | null>(null);
  const detailContentScrollRef = useRef<HTMLDivElement | null>(null);
  const communityScrollRef = useRef<HTMLDivElement | null>(null);
  const activeFeaturedSlideRef = useRef<HTMLDivElement | null>(null);
  const directoryDetailCloseTimerRef = useRef<number | null>(null);
  const reelDurationCacheRef = useRef<Record<string, number | null>>({});
  const consumedInitialSetIdRef = useRef<string | null>(null);
  const loadedEditSetIdRef = useRef<string | null>(null);
  const lastCommunityResetSignalRef = useRef(communityResetSignal);
  const draftBaselinesByContextRef = useRef<Record<string, string>>({});
  const [draftBaselineVersion, setDraftBaselineVersion] = useState(0);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    let cancelled = false;
    setPortalReady(true);
    const localSets = parseStoredSets(window.localStorage.getItem(COMMUNITY_SETS_STORAGE_KEY));
    const seededOwnedSetIds = saveOwnedCommunitySetIds([
      ...readOwnedCommunitySetIds(),
      ...localSets
        .filter((set) => {
          const id = set.id.trim();
          const curator = set.curator.trim().toLowerCase();
          return id.startsWith(USER_CREATED_SET_ID_PREFIX) || curator === "you";
        })
        .map((set) => set.id),
    ]);
    setUserSets(localSets);
    setOwnedSetIds(seededOwnedSetIds);
    const starredSetIdsRaw = window.localStorage.getItem(COMMUNITY_STARRED_SET_IDS_STORAGE_KEY);
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
    }
    setStarredSetsHydrated(true);
    setStorageHydrated(true);
    void (async () => {
      try {
        const remoteSets = await fetchCommunitySets();
        if (cancelled) {
          return;
        }
        setUserSets(remoteSets.slice(0, MAX_USER_SETS));
      } catch {
        // Keep local cache fallback if backend is unavailable.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

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
    window.localStorage.setItem(COMMUNITY_SETS_STORAGE_KEY, JSON.stringify(userSets.slice(0, MAX_USER_SETS)));
  }, [storageHydrated, userSets]);

  useEffect(() => {
    if (typeof window === "undefined" || !storageHydrated) {
      return;
    }
    saveOwnedCommunitySetIds(ownedSetIds);
  }, [ownedSetIds, storageHydrated]);

  useEffect(() => {
    if (typeof window === "undefined" || !starredSetsHydrated) {
      return;
    }
    window.localStorage.setItem(COMMUNITY_STARRED_SET_IDS_STORAGE_KEY, JSON.stringify(starredSetIds));
  }, [starredSetIds, starredSetsHydrated]);

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

  const updateDetailBannerGeometry = useCallback(() => {
    if (!panelRootRef.current) {
      return;
    }
    const panelRect = panelRootRef.current.getBoundingClientRect();
    if (panelRect.width < 40 || panelRect.height < 40) {
      // Skip geometry updates while the panel is hidden/off-layout.
      return;
    }
    const nextLeft = Math.max(0, Math.round(panelRect.left) - DETAIL_BANNER_LEFT_EXPANSION_PX);
    setDetailBannerLeft((prev) => (prev === nextLeft ? prev : nextLeft));
    const nextRight = Math.max(0, Math.round(window.innerWidth - panelRect.right) - DETAIL_BANNER_LEFT_EXPANSION_PX);
    setDetailBannerRight((prev) => (prev === nextRight ? prev : nextRight));

    if (!detailBannerRef.current) {
      return;
    }
    const nextHeight = Math.round(detailBannerRef.current.getBoundingClientRect().height);
    setDetailBannerHeight((prev) => (prev === nextHeight ? prev : nextHeight));
  }, []);

  const isYourSetsMode = mode === "edit";
  const isStandaloneCreateMode = mode === "create";
  const isFormEditMode = isYourSetsMode && isEditSetEditorOpen;
  const isFormCreateMode = isStandaloneCreateMode || (isYourSetsMode && isCreateSetEditorOpen);
  const shouldShowEditSetGrid = isYourSetsMode && !isEditSetEditorOpen && !isCreateSetEditorOpen;
  const shouldShowEditSetForm = isFormEditMode || isFormCreateMode;
  const allSets = useMemo(() => [...userSets, ...DEFAULT_COMMUNITY_SETS], [userSets]);
  const ownedSetIdSet = useMemo(() => new Set(ownedSetIds), [ownedSetIds]);
  const editableSets = useMemo(
    () => userSets.filter((set) => ownedSetIdSet.has(set.id)),
    [ownedSetIdSet, userSets],
  );
  const starredSetIdSet = useMemo(() => new Set(starredSetIds), [starredSetIds]);
  const orderedEditableSets = useMemo(() => {
    const starred: CommunitySet[] = [];
    const regular: CommunitySet[] = [];
    for (const set of editableSets) {
      if (starredSetIdSet.has(set.id)) {
        starred.push(set);
      } else {
        regular.push(set);
      }
    }
    return [...starred, ...regular];
  }, [editableSets, starredSetIdSet]);
  const activeEditableSet = useMemo(
    () => editableSets.find((set) => set.id === activeEditSetId) ?? null,
    [activeEditSetId, editableSets],
  );
  const featuredCarouselSets = useMemo(() => FEATURED_SETS.slice(0, 3), []);
  const detailCarouselReels = useMemo(() => selectedDirectorySet?.reels ?? [], [selectedDirectorySet]);

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
    const normalized = query.trim().toLowerCase();
    if (!normalized) {
      return allSets;
    }
    return allSets.filter((set) => {
      if (set.title.toLowerCase().includes(normalized)) {
        return true;
      }
      if (set.description.toLowerCase().includes(normalized)) {
        return true;
      }
      if (set.curator.toLowerCase().includes(normalized)) {
        return true;
      }
      if (set.tags.some((tag) => tag.includes(normalized))) {
        return true;
      }
      return set.reels.some((reel) => PLATFORM_LABEL[reel.platform].toLowerCase().includes(normalized));
    });
  }, [allSets, query]);

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
    const hasQuery = query.trim().length > 0;
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
  }, [activeCommunityCategory, filteredDirectorySets, query]);

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
    const forwardSteps = (normalized - activeFeaturedIndex + setCount) % setCount;
    const backwardSteps = (activeFeaturedIndex - normalized + setCount) % setCount;
    setFeaturedTransitionDirection(forwardSteps <= backwardSteps ? 1 : -1);
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
    if (mode !== "community" || featuredCarouselSets.length <= 1) {
      return;
    }
    const intervalId = window.setInterval(() => {
      goToNextFeaturedSet();
    }, FEATURED_CAROUSEL_INTERVAL_MS);
    return () => {
      window.clearInterval(intervalId);
    };
  }, [featuredCarouselSets.length, goToNextFeaturedSet, mode]);

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

  const isSearchActive = query.trim().length > 0;

  useEffect(() => {
    if (mode !== "community" || isSearchActive || featuredCarouselSets.length === 0) {
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
  }, [activeFeaturedIndex, featuredCarouselSets.length, featuredTransitionStage, isSearchActive, mode]);

  const directorySets = categoryFilteredSets;
  const detailCarouselCount = detailCarouselReels.length;
  const maxDetailCarouselIndex = Math.max(0, detailCarouselCount - 1);
  const activeDetailCarouselReel = detailCarouselReels[detailCarouselIndex] ?? null;

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
      ? normalizedDraft.reelInputs.map((row) => createDraftReelRow(row.value, row.tStartSec, row.tEndSec))
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
    setReelInputs(nextDraft.reelInputs.map((row) => createDraftReelRow(row.value, row.tStartSec, row.tEndSec)));
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
    window.localStorage.removeItem(COMMUNITY_CREATE_DRAFT_STORAGE_KEY);
  }, []);

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
    setReelInputs(nextDraft.reelInputs.map((row) => createDraftReelRow(row.value, row.tStartSec, row.tEndSec)));
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
    const storedDraft = parseStoredSetDraft(window.localStorage.getItem(COMMUNITY_CREATE_DRAFT_STORAGE_KEY));
    if (storedDraft) {
      applyDraftToForm(storedDraft, COMMUNITY_CREATE_DRAFT_CONTEXT_KEY);
      return;
    }
    resetCreateSetForm();
  }, [applyDraftToForm, mode, resetCreateSetForm]);

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
    const storedDraft = parseStoredSetDraft(window.localStorage.getItem(`${COMMUNITY_EDIT_DRAFT_PREFIX}${activeEditSetId}`));
    if (storedDraft) {
      applyDraftToForm(storedDraft, `edit:${activeEditSetId}`);
    } else {
      applySetToForm(selectedSet);
    }
    loadedEditSetIdRef.current = activeEditSetId;
  }, [activeEditSetId, applyDraftToForm, applySetToForm, editableSets, isEditSetEditorOpen, isYourSetsMode]);

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
        try {
          durationSec = await fetchCommunityReelDuration({ sourceUrl });
        } catch {
          durationSec = null;
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
  }, [mode, parsedDraftReels]);

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

  const hasDetectedDurationForRow = useCallback(
    (rowId: string) => {
      const durationSec = reelDurationByRow[rowId]?.durationSec;
      return Number.isFinite(durationSec) && Number(durationSec) > CLIP_SLIDER_MIN_GAP_SEC;
    },
    [reelDurationByRow],
  );

  const validDraftReelCount = useMemo(
    () =>
      parsedDraftReels.filter(
        (row) => row.value.trim() && row.parsed !== null && !row.hasClipRangeError && hasDetectedDurationForRow(row.id),
      ).length,
    [hasDetectedDurationForRow, parsedDraftReels],
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
  const parsedSetTags = useMemo(() => parseTags(setTags), [setTags]);
  const hasMaxTags = parsedSetTags.length >= MAX_SET_TAGS;
  const requiredCompletionCount =
    (normalizedSetTitle ? 1 : 0) +
    (normalizedSetDescription.length >= MIN_SET_DESCRIPTION_LENGTH ? 1 : 0) +
    (thumbnailPreview ? 1 : 0) +
    (validDraftReelCount > 0 && invalidDraftReelCount === 0 ? 1 : 0);
  const completionPercent = Math.round((requiredCompletionCount / 4) * 100);
  const progressPercent = Math.min(100, Math.max(0, completionPercent));
  const progressRadius = 34;
  const progressCircumference = 2 * Math.PI * progressRadius;
  const progressOffset = progressCircumference * (1 - progressPercent / 100);
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
    } else if (validDraftReelCount === 0) {
      items.push("Wait for reel duration detection");
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
      (row) => row.parsed === null || row.hasClipRangeError || !hasDetectedDurationForRow(row.id),
    );
    if (firstInvalid) {
      if (firstInvalid.parsed === null) {
        setPublishResultModal({
          status: "error",
          label: actionLabel,
          title: "Invalid Reel Link",
          message: `One or more reel links are invalid. Supported: ${SUPPORTED_PLATFORMS_LABEL}.`,
        });
      } else if (!hasDetectedDurationForRow(firstInvalid.id)) {
        setPublishResultModal({
          status: "error",
          label: actionLabel,
          title: "Duration Not Ready",
          message: "Wait until each valid reel's video length is detected before posting.",
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
        const updatedSet = await updateCommunitySet({
          setId: editSetId,
          title,
          description,
          tags,
          reels: parsedReels,
          thumbnailUrl: thumbnailPreview,
          curator: activeEditableSet?.curator || "You",
        });
        setUserSets((prev) => [updatedSet, ...prev.filter((item) => item.id !== updatedSet.id)].slice(0, MAX_USER_SETS));
        if (typeof window !== "undefined") {
          window.localStorage.removeItem(`${COMMUNITY_EDIT_DRAFT_PREFIX}${updatedSet.id}`);
        }
        setDraftBaselineForContext(`edit:${updatedSet.id}`, buildCurrentDraftPayload());
        setCreateError(null);
        setCreateSuccess(`Saved changes to "${updatedSet.title}".`);
        return true;
      } else {
        const createdSet = await createCommunitySet({
          title,
          description,
          tags,
          reels: parsedReels,
          thumbnailUrl: thumbnailPreview,
          curator: "You",
        });
        setOwnedSetIds((prev) => (prev.includes(createdSet.id) ? prev : [createdSet.id, ...prev]));
        setUserSets((prev) => [createdSet, ...prev.filter((item) => item.id !== createdSet.id)].slice(0, MAX_USER_SETS));
        clearCreateSetDraftProgress();
        resetCreateSetForm();
        setPublishResultModal({
          status: "success",
          title: "Published Successfully",
          message: `"${createdSet.title}" is now live with ${createdSet.reels.length} reel${createdSet.reels.length === 1 ? "" : "s"}.`,
          thumbnailUrl: createdSet.thumbnailUrl || thumbnailPreview || undefined,
          thumbnailAlt: `${createdSet.title} thumbnail`,
        });
        return true;
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : isFormEditMode ? "Could not update community set." : "Could not post community set.";
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
    hasDetectedDurationForRow,
    isFormEditMode,
    parsedDraftReels,
    resetCreateSetForm,
    setDraftBaselineForContext,
    setTags,
    setDescription,
    setTitle,
    thumbnailPreview,
    setUserSets,
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
    try {
      window.localStorage.setItem(storageKey, JSON.stringify(draftPayload));
      return { ok: true };
    } catch {
      // If storage quota is tight (large thumbnail previews), retry with a compact draft.
      const compactDraft: StoredSetDraft = {
        ...draftPayload,
        thumbnailPreview: "",
        thumbnailFileName: draftPayload.thumbnailFileName || "",
      };
      try {
        window.localStorage.setItem(storageKey, JSON.stringify(compactDraft));
        return {
          ok: true,
          warning: "Draft saved without thumbnail preview due browser storage limits.",
        };
      } catch (error) {
        const message = error instanceof Error ? error.message : "Could not save draft progress.";
        return { ok: false, error: message };
      }
    }
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
      const persistResult = persistDraftPayload(`${COMMUNITY_EDIT_DRAFT_PREFIX}${editSetId}`, draftPayload);
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
    const persistResult = persistDraftPayload(COMMUNITY_CREATE_DRAFT_STORAGE_KEY, draftPayload);
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
  }, [activeEditSetId, activeEditableSet, buildCurrentDraftPayload, isFormEditMode, persistDraftPayload, setDraftBaselineForContext]);

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
        window.localStorage.removeItem(COMMUNITY_CREATE_DRAFT_STORAGE_KEY);
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
  }, [resetCreateSetForm]);

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
    setIsEditSetEditorOpen(false);
    setIsCreateSetEditorOpen(true);
    setActiveEditSetId(null);
    loadedEditSetIdRef.current = null;
    if (typeof window !== "undefined") {
      const storedDraft = parseStoredSetDraft(window.localStorage.getItem(COMMUNITY_CREATE_DRAFT_STORAGE_KEY));
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
  }, [applyDraftToForm, resetCreateSetForm]);

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
    const didSave = isFormEditMode
      ? await submitSet()
      : saveCurrentDraftProgress({ showSuccessMessage: false });
    if (!didSave) {
      return;
    }
    const pendingAction = unsavedDraftExitModal.action;
    setUnsavedDraftExitModal(null);
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
    if (!normalized || deletingSetId) {
      return;
    }
    const target = editableSets.find((set) => set.id === normalized);
    const targetTitle = target?.title?.trim() || "this set";
    setActiveSetActionsMenuId(null);
    setDeleteSetConfirmModal({ setId: normalized, title: targetTitle });
  }, [deletingSetId, editableSets]);

  const closeDeleteSetConfirmModal = useCallback(() => {
    if (deletingSetId) {
      return;
    }
    setDeleteSetConfirmModal(null);
  }, [deletingSetId]);

  const onDeleteEditableSet = useCallback(async (setId: string) => {
    const normalized = setId.trim();
    if (!normalized || deletingSetId) {
      return;
    }
    const target = editableSets.find((set) => set.id === normalized);
    const targetTitle = target?.title || "this set";
    setDeleteSetConfirmModal(null);
    setPublishResultModal(null);
    setDeletingSetId(normalized);
    try {
      await deleteCommunitySet({ setId: normalized });
      setUserSets((prev) => prev.filter((set) => set.id !== normalized));
      setOwnedSetIds((prev) => prev.filter((id) => id !== normalized));
      setStarredSetIds((prev) => prev.filter((id) => id !== normalized));
      if (typeof window !== "undefined") {
        window.localStorage.removeItem(`${COMMUNITY_EDIT_DRAFT_PREFIX}${normalized}`);
      }
      if (activeEditSetId === normalized) {
        loadedEditSetIdRef.current = null;
        setActiveEditSetId(null);
        setIsEditSetEditorOpen(false);
      }
      setCreateError(null);
      setCreateSuccess(`Deleted "${targetTitle}".`);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Could not delete this set.";
      setPublishResultModal({
        status: "error",
        label: "Your Sets",
        title: "Delete Failed",
        message,
      });
    } finally {
      setDeletingSetId(null);
      setActiveSetActionsMenuId(null);
    }
  }, [activeEditSetId, deletingSetId, editableSets]);

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
    clearDirectoryDetailCloseTimer();
    setIsDetailBannerCompact(false);
    setIsDirectoryDetailOpen(false);
    directoryDetailCloseTimerRef.current = window.setTimeout(() => {
      setSelectedDirectorySet(null);
      directoryDetailCloseTimerRef.current = null;
    }, DIRECTORY_DETAIL_TRANSITION_MS);
  }, [clearDirectoryDetailCloseTimer, selectedDirectorySet]);

  useEffect(() => {
    if (mode === "community" && isVisible) {
      return;
    }
    clearDirectoryDetailCloseTimer();
    setIsDetailBannerCompact(false);
    setSkipDetailTransitionOnce(false);
    setIsDirectoryDetailOpen(false);
    setSelectedDirectorySet(null);
    setSelectedDetailReelId(null);
    setDetailCarouselIndex(0);
    consumedInitialSetIdRef.current = null;
  }, [clearDirectoryDetailCloseTimer, isVisible, mode]);

  useEffect(() => {
    if (lastCommunityResetSignalRef.current === communityResetSignal) {
      return;
    }
    lastCommunityResetSignalRef.current = communityResetSignal;
    if (mode !== "community" || !isVisible) {
      return;
    }
    clearDirectoryDetailCloseTimer();
    setIsDetailBannerCompact(false);
    setSkipDetailTransitionOnce(false);
    setIsDirectoryDetailOpen(false);
    setSelectedDirectorySet(null);
    setSelectedDetailReelId(null);
    setDetailCarouselIndex(0);
    consumedInitialSetIdRef.current = initialOpenSetId?.trim() || null;
  }, [clearDirectoryDetailCloseTimer, communityResetSignal, initialOpenSetId, isVisible, mode]);

  const openDirectorySet = useCallback(
    (set: CommunitySet, options?: { immediate?: boolean; skipTransition?: boolean }) => {
      clearDirectoryDetailCloseTimer();
      updateDetailBannerGeometry();
      setSelectedDirectorySet(set);
      setIsDetailBannerCompact(false);
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
    [clearDirectoryDetailCloseTimer, updateDetailBannerGeometry],
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
    setIsDetailBannerCompact(false);
    setIsDirectoryDetailOpen(true);
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
    if (!onDetailOpenChange) {
      return;
    }
    onDetailOpenChange(mode === "community" && isVisible && isDirectoryDetailOpen);
  }, [isDirectoryDetailOpen, isVisible, mode, onDetailOpenChange]);

  useEffect(() => {
    if (!selectedDirectorySet) {
      setIsDetailBannerCompact(false);
    }
  }, [selectedDirectorySet]);

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

  useEffect(() => {
    if (mode !== "community" || !isVisible || !isDirectoryDetailOpen || detailCarouselReels.length <= 1) {
      return;
    }
    const intervalId = window.setInterval(() => {
      setDetailCarouselIndex((prev) => (prev + 1) % detailCarouselReels.length);
    }, DETAIL_REEL_CAROUSEL_INTERVAL_MS);
    return () => {
      window.clearInterval(intervalId);
    };
  }, [detailCarouselReels.length, isDirectoryDetailOpen, isVisible, mode]);

  useEffect(() => {
    if (mode !== "community" || !isDirectoryDetailOpen || !selectedDirectorySet) {
      return;
    }
    const el = detailContentScrollRef.current;
    if (!el) {
      return;
    }
    const onScroll = () => {
      setIsDetailBannerCompact((prev) => {
        const next = el.scrollTop > 0;
        return prev === next ? prev : next;
      });
    };
    onScroll();
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      el.removeEventListener("scroll", onScroll);
    };
  }, [isDirectoryDetailOpen, mode, selectedDirectorySet]);

  useEffect(() => {
    if (mode !== "community" || !isVisible || !portalReady || !selectedDirectorySet) {
      return;
    }
    const update = () => {
      updateDetailBannerGeometry();
    };
    update();
    const rafId = window.requestAnimationFrame(update);
    window.addEventListener("resize", update);

    const resizeObserver = typeof ResizeObserver !== "undefined" ? new ResizeObserver(update) : null;
    if (resizeObserver && panelRootRef.current) {
      resizeObserver.observe(panelRootRef.current);
    }
    if (resizeObserver && detailBannerRef.current) {
      resizeObserver.observe(detailBannerRef.current);
    }

    return () => {
      window.cancelAnimationFrame(rafId);
      window.removeEventListener("resize", update);
      resizeObserver?.disconnect();
    };
  }, [isDirectoryDetailOpen, isVisible, mode, portalReady, selectedDirectorySet, updateDetailBannerGeometry]);

  const detailContentTopPadding = Math.max(
    DETAIL_CONTENT_TOP_PADDING_FALLBACK,
    detailBannerHeight + DETAIL_CONTENT_TOP_PADDING_GUTTER - DETAIL_CONTENT_TOP_PADDING_UPSHIFT_PX,
  );
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
    [onOpenCommunityReelInFeed, router],
  );

  const openSelectedSetReelsInFeed = useCallback(() => {
    if (!selectedDirectorySet || selectedDirectorySet.reels.length === 0) {
      return;
    }
    const selectedReel =
      selectedDirectorySet.reels.find((reel) => reel.id === selectedDetailReelId) ?? selectedDirectorySet.reels[0];
    openCommunityReelInFeed(selectedDirectorySet, selectedReel);
  }, [openCommunityReelInFeed, selectedDetailReelId, selectedDirectorySet]);

  const detailBannerPortal =
    mode === "community" && isVisible && portalReady && selectedDirectorySet
      ? createPortal(
        <div
          ref={detailBannerRef}
          className={`pointer-events-none fixed top-0 z-[96] overflow-hidden ${
            isDetailBannerCompact ? "bg-transparent backdrop-blur-[10px]" : "bg-white/[0.04] backdrop-blur-[4px]"
          } transition-[opacity,backdrop-filter] duration-[560ms] ease-[cubic-bezier(0.25,0.1,0.25,1)] ${
            isDirectoryDetailOpen ? "opacity-100" : "opacity-0 pointer-events-none"
          }`}
            style={{
              left: `${detailBannerLeft + DETAIL_BANNER_LEFT_INSET_PX}px`,
              right: `${detailBannerRight}px`,
            }}
          >
            <div
              className={`relative z-10 transition-[padding] duration-[840ms] ease-[cubic-bezier(0.2,0.85,0.25,1)] ${
                isDetailBannerCompact
                  ? "px-4 pt-4 sm:px-6 sm:pt-4 md:px-7"
                  : "px-4 pt-[calc(max(env(safe-area-inset-top),0px)+24px)] sm:px-6 sm:pt-[calc(max(env(safe-area-inset-top),0px)+28px)] md:px-7 md:pt-[calc(max(env(safe-area-inset-top),0px)+34px)]"
              }`}
            >
              <div
                className={`overflow-hidden will-change-[max-height,transform,opacity] transition-[max-height,opacity,transform,padding] duration-[840ms] ease-[cubic-bezier(0.2,0.85,0.25,1)] ${
                  isDetailBannerCompact ? "max-h-0 -translate-y-3 opacity-0 pb-0" : "max-h-[920px] translate-y-0 opacity-100 pb-12 sm:pb-14 md:pb-16"
                }`}
              >
                <button
                  type="button"
                  onClick={closeDirectorySetModal}
                  className="pointer-events-auto mt-1 inline-flex items-center gap-2 rounded-full px-2.5 py-2 text-[13px] font-semibold text-white/90 transition hover:text-white sm:text-sm"
                >
                  <i className="fa-solid fa-chevron-left text-[11px] sm:text-xs" aria-hidden="true" />
                  Community Sets
                </button>

                <div className="mt-12 flex flex-col gap-5 sm:mt-14 sm:gap-6 md:mt-16">
                  <span className="grid h-12 w-12 shrink-0 overflow-hidden rounded-2xl bg-black/28 text-white/90 sm:h-14 sm:w-14">
                    {selectedDirectorySet.thumbnailUrl ? (
                      <img src={selectedDirectorySet.thumbnailUrl} alt="" aria-hidden="true" className="h-full w-full object-cover" />
                    ) : (
                      <i className={`${getSetIconClass(selectedDirectorySet)} text-lg sm:text-xl`} aria-hidden="true" />
                    )}
                  </span>

                  <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
                    <div className="min-w-0">
                      <p className="text-[10px] font-semibold uppercase tracking-[0.1em] text-white/70">Community Set</p>
                      <h3 className="mt-1 text-[1.9rem] font-semibold leading-[1.08] text-white sm:text-[2.2rem] md:text-[2.8rem]">{selectedDirectorySet.title}</h3>
                      <p className="mt-2 max-w-3xl text-sm leading-relaxed text-white/85 sm:text-[0.98rem] md:text-[1.05rem]">{selectedDirectorySet.description}</p>
                    </div>

                    <button
                      type="button"
                      onClick={openSelectedSetReelsInFeed}
                      disabled={!selectedDirectorySetHasReels}
                      className={`pointer-events-auto inline-flex h-10 items-center justify-center self-start rounded-full px-5 text-sm font-semibold transition md:self-center ${
                        selectedDirectorySetHasReels
                          ? "bg-white text-[#06233a] hover:bg-[#d9eefb]"
                          : "cursor-not-allowed bg-white/45 text-[#06233a]/70"
                      }`}
                    >
                      View Reels
                    </button>
                  </div>
                </div>

                <div className="mt-4 flex flex-wrap items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.08em] text-white/80 sm:mt-5 sm:text-[11px]">
                  <span className="rounded-full bg-black/28 px-2.5 py-1">{getSetReelCount(selectedDirectorySet)} reels</span>
                  <span className="rounded-full bg-black/28 px-2.5 py-1">{formatCompact(selectedDirectorySet.learners)} learners</span>
                  <span className="rounded-full bg-black/28 px-2.5 py-1">{formatCompact(selectedDirectorySet.likes)} likes</span>
                  <span className="rounded-full bg-black/28 px-2.5 py-1">Curated by {selectedDirectorySet.curator}</span>
                </div>
              </div>

              <div
                className={`overflow-hidden transition-[max-height,opacity,transform,backdrop-filter] duration-[640ms] ease-[cubic-bezier(0.2,0.85,0.25,1)] ${
                  isDetailBannerCompact
                    ? "pointer-events-auto h-16 max-h-16 translate-y-0 rounded-2xl border border-[#8d8d8d]/35 bg-black/30 opacity-100 backdrop-blur-[12px] sm:h-20 sm:max-h-20"
                    : "pointer-events-none h-0 max-h-0 -translate-y-2 opacity-0"
                }`}
              >
                <div className="mx-auto flex h-full w-full max-w-none items-center justify-between gap-3 px-3 sm:px-4">
                  <div className="ml-1 min-w-0 max-w-[62%] flex items-center gap-2.5 sm:ml-2">
                    <span className="grid h-9 w-9 shrink-0 overflow-hidden rounded-xl bg-black/28 text-white/90">
                      {selectedDirectorySet.thumbnailUrl ? (
                        <img src={selectedDirectorySet.thumbnailUrl} alt="" aria-hidden="true" className="h-full w-full object-cover" />
                      ) : (
                        <i className={`${getSetIconClass(selectedDirectorySet)} text-sm`} aria-hidden="true" />
                      )}
                    </span>
                    <p className="truncate text-[0.96rem] font-semibold text-white sm:text-[1.02rem]">{selectedDirectorySet.title}</p>
                  </div>
                  <button
                    type="button"
                    onClick={openSelectedSetReelsInFeed}
                    disabled={!selectedDirectorySetHasReels}
                    className={`pointer-events-auto mr-2 inline-flex h-9 shrink-0 items-center justify-center rounded-full px-4 text-xs font-semibold transition sm:mr-3 sm:px-5 ${
                      selectedDirectorySetHasReels
                        ? "bg-white text-[#06233a] hover:bg-[#d9eefb]"
                        : "cursor-not-allowed bg-white/45 text-[#06233a]/70"
                    }`}
                  >
                    View Reels
                  </button>
                </div>
              </div>
            </div>
          </div>,
          document.body,
        )
      : null;

  return (
    <div
      ref={panelRootRef}
      className={`flex h-full min-h-0 flex-col overflow-hidden text-white ${
        mode === "community"
          ? "px-3 pb-0 pt-14 sm:px-5 sm:pb-0 md:px-7 md:pb-0 md:pt-20 lg:px-8 lg:pt-7 lg:pb-0"
          : "px-3 pb-0 pt-14 sm:px-5 sm:pb-0 md:px-7 md:pb-0 md:pt-20 lg:px-8 lg:pt-7 lg:pb-0"
      }`}
    >
      {mode === "community" ? (
        <div className="relative min-h-0 flex-1 overflow-hidden">
          <div
            className={`absolute inset-0 flex min-h-0 flex-col transition-opacity duration-[440ms] ease-[cubic-bezier(0.22,1,0.36,1)] ${
              isDirectoryDetailOpen ? "opacity-0 pointer-events-none" : "opacity-100"
            }`}
            aria-hidden={isDirectoryDetailOpen}
          >
            <div className="shrink-0">
              <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between md:gap-4">
                <div className="w-full pl-5 sm:pl-6 md:w-auto md:pl-8 lg:pl-3">
                  <div className="flex items-center justify-center gap-2 md:justify-start">
                    <h2 className="text-xl font-semibold tracking-tight text-white sm:text-2xl md:text-[1.9rem]">Community Sets</h2>
                    <span className="rounded-full border border-white/20 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.09em] text-white/55">Beta</span>
                  </div>
                </div>
                <label className="mx-auto block w-[calc(100%-0.25rem)] self-stretch md:mx-0 md:mr-5 md:w-[20.5rem] md:self-auto lg:mr-3 lg:w-[23rem]">
                  <div className="relative">
                    <i className="fa-solid fa-magnifying-glass pointer-events-none absolute left-4 top-1/2 -translate-y-1/2 text-sm text-white/45" />
                    <input
                      value={query}
                      onChange={(event) => setQuery(event.target.value)}
                      placeholder="Search community sets"
                      className="h-11 w-full rounded-xl border border-white/20 bg-black/35 pl-11 pr-4 text-sm text-white outline-none placeholder:text-white/45 focus:border-white/40 sm:h-12 sm:pl-12"
                    />
                  </div>
                </label>
              </div>
            </div>

            <div className="mt-3 min-h-0 flex-1 overflow-hidden md:mt-4">
              <div ref={communityScrollRef} className="balanced-scroll-gutter h-full min-h-0 space-y-4 overflow-y-auto pb-6 md:space-y-5 md:pb-8 lg:pb-10">
            {!isSearchActive && featuredCarouselSets.length > 0 ? (
              <section className="group/featured relative overflow-hidden rounded-[1.5rem] border border-[#2b2b2b] bg-transparent p-4 pb-12 backdrop-blur-[4px] max-[380px]:pb-10 sm:rounded-[2rem] sm:p-5 sm:pb-14 md:p-7 md:pb-16 lg:p-8">
                <div className="pointer-events-none absolute inset-0 bg-white/[0.04]" />

                {featuredCarouselSets.length > 1 ? (
                  <>
                    <button
                      type="button"
                      aria-label="Next featured set"
                      onClick={goToNextFeaturedSet}
                      className="absolute right-4 top-4 z-30 grid h-9 w-9 place-items-center rounded-full bg-black/45 text-white/82 transition-all duration-200 hover:bg-black/60 hover:text-white md:pointer-events-none md:right-6 md:top-1/2 md:-translate-y-1/2 md:opacity-0 md:group-hover/featured:pointer-events-auto md:group-hover/featured:opacity-100 md:group-focus-within/featured:pointer-events-auto md:group-focus-within/featured:opacity-100"
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
                      ? featuredTransitionDirection === 1
                        ? "animate-featured-fade-exit animate-featured-slide-exit-forward"
                        : "animate-featured-fade-exit animate-featured-slide-exit-backward"
                      : featuredTransitionStage === "entering"
                        ? featuredTransitionDirection === 1
                          ? "animate-featured-fade-enter animate-featured-slide-enter-forward"
                          : "animate-featured-fade-enter animate-featured-slide-enter-backward"
                        : "opacity-100";
                    return (
                      <article
                        key={`${set.id}-${isLeaving ? "leaving" : "active"}`}
                        className={`absolute inset-0 ${isLeaving ? "z-10 pointer-events-none" : "z-20"} ${motionClass}`}
                      >
                        <div
                          ref={isLeaving ? null : activeFeaturedSlideRef}
                          className="grid min-h-[285px] gap-5 max-[380px]:gap-4 sm:min-h-[320px] sm:gap-7 md:min-h-[410px] md:grid-cols-[minmax(0,1fr)_minmax(0,1.05fr)] md:items-center"
                        >
                          <div className="flex max-w-2xl flex-col items-start text-left md:pl-2 lg:pl-8">
                            <p className="inline-flex rounded-full bg-white/12 px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.1em] text-white/88">
                              Featured Set
                            </p>
                            <h3 className="mt-5 text-[1.65rem] font-semibold leading-[1.12] text-white max-[380px]:mt-3 sm:mt-4 sm:text-[2rem] md:text-[2.6rem] lg:text-[3.05rem]">{set.title}</h3>
                            <p className="mt-5 max-w-xl text-[0.95rem] leading-relaxed text-white/84 max-[380px]:mt-3 sm:mt-4 sm:text-[1.02rem] md:text-[1.08rem] lg:text-[1.18rem]">{set.description}</p>

                            <div className="mt-7 flex flex-wrap items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.08em] text-white/78 max-[380px]:mt-5 sm:mt-5 sm:gap-2.5 sm:text-[11px]">
                              <span className="rounded-full bg-black/30 px-2 py-1">{getSetReelCount(set)} reels</span>
                              <span className="rounded-full bg-black/30 px-2 py-1">{formatCompact(set.learners)} learners</span>
                              <span className="rounded-full bg-black/30 px-2 py-1">{formatCompact(set.likes)} likes</span>
                            </div>

                            <button
                              type="button"
                              data-featured-view-set-button
                              onClick={() => openDirectorySet(set)}
                              className="mt-8 inline-flex w-full items-center justify-center rounded-full bg-white px-6 py-2.5 text-sm font-semibold text-black transition max-[380px]:mt-6 hover:bg-[#f1eee5] sm:mt-7 sm:w-auto sm:px-8 sm:py-3"
                            >
                              View Set
                            </button>
                          </div>

                          <div className="relative hidden h-full min-h-[320px] md:block">
                            <div className="absolute right-8 top-2 z-20 max-w-[78%] rounded-full border border-white/35 bg-white/20 px-3 py-1.5 text-xs font-semibold text-white/92 backdrop-blur-xl lg:right-16 lg:max-w-[72%] lg:px-4 lg:py-2 lg:text-sm">
                              @{set.curator} trending this week
                            </div>
                            <div
                              data-featured-image-target
                              className="absolute bottom-0 right-8 w-[84%] overflow-hidden rounded-[1.4rem] border border-white/25 bg-black/30 lg:right-12 lg:w-[80%] lg:rounded-[1.7rem]"
                            >
                              <img
                                src={set.thumbnailUrl || FALLBACK_THUMBNAIL_URL}
                                alt={`${set.title} cover`}
                                className="h-[280px] w-full object-contain md:h-[310px] lg:h-[330px]"
                              />
                            </div>
                          </div>
                        </div>
                      </article>
                    );
                  })}
                </div>

                {featuredCarouselSets.length > 1 ? (
                  <div className="absolute bottom-4 left-1/2 z-20 flex -translate-x-1/2 items-center justify-center gap-2 md:bottom-5">
                    {featuredCarouselSets.map((set, index) => (
                      <button
                        key={`featured-dot-${set.id}`}
                        type="button"
                        aria-label={`Go to featured set ${index + 1}`}
                        onClick={() => goToFeaturedSet(index)}
                        className={`h-2 rounded-full transition-all ${
                          index === activeFeaturedIndex ? "w-6 bg-white" : "w-2 bg-white/45 hover:bg-white/70"
                        }`}
                      />
                    ))}
                  </div>
                ) : null}
              </section>
            ) : null}

            {!isSearchActive ? (
            <section className="flex items-center gap-2 overflow-x-auto pb-1">
              {communityCategories.map((category) => (
                <button
                  key={category}
                  type="button"
                  onClick={() => onCommunityCategoryChange(category)}
                  className={`whitespace-nowrap rounded-full px-3 py-1.5 text-xs font-medium transition-all duration-200 ease-out sm:px-4 sm:py-2 sm:text-sm ${
                    activeCommunityCategory === category
                      ? "bg-white text-black"
                      : "bg-black/30 text-white/75 hover:bg-white/20 hover:text-white hover:backdrop-blur-sm"
                  }`}
                >
                  {category}
                </button>
              ))}
            </section>
            ) : null}

            <section className="relative overflow-hidden rounded-2xl bg-transparent px-3 pt-3 pb-3 backdrop-blur-[3px] sm:px-4 sm:pt-3.5 sm:pb-4">
              <div className="pointer-events-none absolute inset-0 bg-white/[0.04]" />
              <div className="relative z-10">
              <div className="mb-2 flex items-center justify-between">
                <p className="text-[11px] font-semibold uppercase tracking-[0.1em] text-white/60">
                  {isSearchActive ? "Search Results" : "Community Directory"}
                </p>
                <p className="text-[10px] font-semibold uppercase tracking-[0.09em] text-white/45">{directorySets.length} sets</p>
              </div>
              {directorySets.length === 0 ? (
                <p className="px-3 py-4 text-sm text-white/65">
                  {isSearchActive ? "No sets matched your search." : "No sets matched that search."}
                </p>
              ) : (
                <div className={isSearchActive ? "flex flex-col gap-2.5" : "grid gap-2.5 md:grid-cols-2 md:gap-x-4 md:gap-y-3 lg:gap-x-10"}>
                  {directorySets.map((set) => {
                    const reelCount = getSetReelCount(set);
                    return (
                      <button
                        type="button"
                        key={set.id}
                        onClick={() => openDirectorySet(set)}
                        className="group relative flex w-full items-center gap-2.5 rounded-xl bg-[#1c1c1c] px-3 py-3 text-left transition-all duration-200 ease-out hover:bg-[#121212] sm:gap-3 sm:rounded-2xl"
                      >
                        <span className="grid h-[3.75rem] w-[3.75rem] shrink-0 overflow-hidden rounded-lg bg-black/30 text-white/82 transition-colors duration-200 group-hover:text-white sm:h-16 sm:w-16 sm:rounded-xl">
                          {set.thumbnailUrl ? (
                            <img src={set.thumbnailUrl} alt="" aria-hidden="true" className="h-full w-full object-cover" />
                          ) : (
                            <i className="fa-regular fa-square text-sm sm:text-base" aria-hidden="true" />
                          )}
                        </span>
                        <div className="min-w-0 flex-1 text-left">
                          <p className="w-full truncate text-[0.97rem] font-medium text-white transition-colors duration-200 group-hover:text-white sm:text-[1.02rem]">{set.title}</p>
                          <p className="mt-0.5 hidden w-full truncate text-sm text-white/58 transition-colors duration-200 group-hover:text-white/78 lg:block">{set.description}</p>
                          <span className="mt-1 inline-flex rounded-full bg-white/10 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.08em] text-white/62 transition-colors duration-200 group-hover:text-white/80">
                            {reelCount} reels
                          </span>
                        </div>
                        <span
                          className="grid h-7 w-7 shrink-0 place-items-center self-center rounded-full text-white/58 transition-colors duration-200 group-hover:text-white/80 sm:h-8 sm:w-8"
                          aria-hidden="true"
                        >
                          <i className="fa-solid fa-chevron-right text-[11px]" />
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
            role="dialog"
            aria-modal="true"
            aria-label={selectedDirectorySet ? `${selectedDirectorySet.title} details` : "Community set details"}
            className={`absolute inset-0 flex min-h-0 flex-col ${skipDetailTransitionOnce ? "" : "transition-opacity duration-[440ms] ease-[cubic-bezier(0.22,1,0.36,1)]"} ${
              isDirectoryDetailOpen ? "opacity-100" : "opacity-0 pointer-events-none"
            }`}
          >
            {selectedDirectorySet ? (
              <div
                ref={detailContentScrollRef}
                className="balanced-scroll-gutter min-h-0 flex-1 overflow-y-auto pb-6 md:pb-8 lg:pb-10"
                style={{ paddingTop: detailContentTopPadding }}
              >
                <div className="px-1 sm:px-2 md:px-3">
                  <section className="rounded-2xl px-4 py-4 sm:px-5 sm:py-5">
                    <div className="flex items-center justify-between gap-3">
                      <p className="text-[11px] font-semibold uppercase tracking-[0.1em] text-white/65">Reel Preview</p>
                      <div className="flex items-center gap-1.5">
                        <button
                          type="button"
                          onClick={goToPreviousDetailCarousel}
                          disabled={detailCarouselCount <= 1}
                          aria-label="Previous reel"
                          className="grid h-8 w-8 place-items-center rounded-full bg-black/45 text-white/80 transition hover:bg-black/60 hover:text-white disabled:cursor-not-allowed disabled:opacity-35"
                        >
                          <i className="fa-solid fa-chevron-left text-[10px]" aria-hidden="true" />
                        </button>
                        <button
                          type="button"
                          onClick={goToNextDetailCarousel}
                          disabled={detailCarouselCount <= 1}
                          aria-label="Next reel"
                          className="grid h-8 w-8 place-items-center rounded-full bg-black/45 text-white/80 transition hover:bg-black/60 hover:text-white disabled:cursor-not-allowed disabled:opacity-35"
                        >
                          <i className="fa-solid fa-chevron-right text-[10px]" aria-hidden="true" />
                        </button>
                      </div>
                    </div>

                    <div className="mt-3">
                      {activeDetailCarouselReel ? (
                        <>
                          <iframe
                            src={activeDetailCarouselReel.embedUrl}
                            title={`${selectedDirectorySet.title} reel preview`}
                            className="h-[270px] w-full border-0 sm:h-[360px] md:h-[440px]"
                            loading="lazy"
                            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                            allowFullScreen
                          />
                          <div className="mt-2 flex items-center justify-between gap-2 text-[10px] font-semibold uppercase tracking-[0.08em] text-white/62">
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
                        <p className="rounded-xl bg-black/30 px-3 py-6 text-sm text-white/65">No reels uploaded for this set yet.</p>
                      )}
                    </div>
                  </section>

                  <section className="mt-4 rounded-2xl px-4 py-4 sm:px-5 sm:py-5">
                    <p className="text-[11px] font-semibold uppercase tracking-[0.1em] text-white/65">Information</p>
                    {selectedDirectorySet.tags.length > 0 ? (
                      <div className="mt-3 flex flex-wrap gap-2">
                        {selectedDirectorySet.tags.map((tag) => (
                          <span key={`${selectedDirectorySet.id}-tag-${tag}`} className="rounded-full bg-white/10 px-2.5 py-1 text-[11px] text-white/72">
                            #{tag}
                          </span>
                        ))}
                      </div>
                    ) : null}
                    <div className="mt-3 space-y-3">
                      {DETAIL_LOREM_PARAGRAPHS.map((paragraph, index) => (
                        <p key={`${selectedDirectorySet.id}-detail-lorem-${index}`} className="text-sm leading-relaxed text-white/78">
                          {paragraph}
                        </p>
                      ))}
                    </div>
                  </section>

                </div>
              </div>
            ) : null}
          </section>
        </div>
      ) : (
        <>
          <div className="flex min-h-0 flex-1 flex-col pt-1 md:pt-2">
            <div className="shrink-0">
              <div
                className={`flex gap-3 md:-mx-2 md:flex-row md:items-center md:justify-between md:gap-4 lg:-mx-3 ${
                  isYourSetsMode && shouldShowEditSetGrid ? "flex-row items-center justify-between" : "flex-col"
                }`}
              >
                <div
                  className={
                    isYourSetsMode && shouldShowEditSetGrid
                      ? "min-w-0 flex-1 pl-2 md:w-auto md:pl-2 lg:pl-2"
                      : "w-full pl-5 sm:pl-6 md:w-auto md:pl-6 lg:pl-2"
                  }
                >
                  <div
                    className={`flex items-center gap-2 md:justify-start ${
                      isYourSetsMode && shouldShowEditSetGrid ? "justify-start" : "justify-center"
                    }`}
                  >
                    {isYourSetsMode && shouldShowEditSetForm ? (
                      <button
                        type="button"
                        onClick={onBackToEditSetGrid}
                        aria-label="Back to all sets"
                        className="inline-flex h-8 w-8 items-center justify-center rounded-xl border border-[#2b2b2b] text-white/80 transition hover:bg-white/10 hover:text-white"
                      >
                        <i className="fa-solid fa-chevron-left text-[11px]" aria-hidden="true" />
                      </button>
                    ) : null}
                    <h2 className="text-xl font-semibold tracking-tight text-white sm:text-2xl md:text-[1.9rem]">
                      {isYourSetsMode
                        ? shouldShowEditSetForm
                          ? isFormEditMode
                            ? `Editing "${activeEditableSet?.title ?? "Set"}"`
                            : "Create Set"
                          : "Your Sets"
                        : "Create Set"}
                    </h2>
                    <span className="rounded-full border border-[#2b2b2b] px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.09em] text-white/55">Beta</span>
                  </div>
                </div>
                {isYourSetsMode && shouldShowEditSetGrid ? (
                  <div className="flex items-center justify-center pr-2 sm:pr-2 md:justify-end md:pr-2 lg:pr-2">
                    <button
                      type="button"
                      onClick={onOpenCreateSetFromGrid}
                      className="inline-flex h-10 min-w-[8.25rem] items-center justify-center rounded-xl border border-[#2b2b2b] bg-black/35 px-4 text-xs font-semibold uppercase tracking-[0.08em] text-white/80 backdrop-blur-md transition hover:bg-white/10 hover:text-white"
                    >
                      Create Set
                    </button>
                  </div>
                ) : null}
              </div>
            </div>

            <div className="mt-3 min-h-0 flex-1 overflow-hidden md:-mx-4 md:mt-4 lg:-mx-5">
              <div className="balanced-scroll-gutter h-full min-h-0 overflow-y-auto pb-0">
                {shouldShowEditSetGrid ? (
                  <section className="rounded-3xl px-1 pt-1 pb-2 sm:px-2 sm:pt-2 sm:pb-3 md:px-3 md:pt-3 md:pb-4">
                    <div className="relative overflow-hidden rounded-2xl border border-[#2b2b2b] bg-transparent p-4 pb-5 backdrop-blur-[4px] sm:p-5 sm:pb-6">
                      <div className="pointer-events-none absolute inset-0 bg-white/[0.04]" />
                      <div className="relative z-10">
                        <div className="flex flex-wrap items-center justify-between gap-3">
                          <div>
                            <p className="text-[11px] font-semibold uppercase tracking-[0.11em] text-white/70">Your Created Sets</p>
                          </div>
                          <div className="flex items-center">
                            <span className="rounded-full border border-[#2b2b2b] bg-black/45 px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.09em] text-white/72">
                              {editableSets.length} set{editableSets.length === 1 ? "" : "s"}
                            </span>
                          </div>
                        </div>
                        {orderedEditableSets.length > 0 ? (
                          <div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
                            {orderedEditableSets.map((set) => {
                              const reelCount = getSetReelCount(set);
                              const isStarred = starredSetIdSet.has(set.id);
                              const isDeleting = deletingSetId === set.id;
                              const isActionsMenuOpen = activeSetActionsMenuId === set.id;
                              return (
                                <div
                                  key={`edit-set-grid-${set.id}`}
                                  className="group relative overflow-hidden rounded-2xl border border-[#2b2b2b] bg-black/35 text-left transition hover:border-white/35 hover:bg-black/55"
                                >
                                  <div
                                    data-your-set-actions="true"
                                    className={`absolute right-2 top-2 z-20 transition-opacity ${
                                      isActionsMenuOpen
                                        ? "opacity-100 pointer-events-auto"
                                        : "opacity-0 pointer-events-none group-hover:opacity-100 group-hover:pointer-events-auto group-focus-within:opacity-100 group-focus-within:pointer-events-auto"
                                    }`}
                                  >
                                    <button
                                      type="button"
                                      onClick={(event) => {
                                        event.preventDefault();
                                        event.stopPropagation();
                                        setActiveSetActionsMenuId((prev) => (prev === set.id ? null : set.id));
                                      }}
                                      aria-label={`Actions for ${set.title}`}
                                      className="inline-flex h-7 w-7 items-center justify-center rounded-lg border border-[#2b2b2b] bg-black text-white/80 transition hover:bg-black hover:text-white"
                                    >
                                      <i className="fa-solid fa-ellipsis text-xs" aria-hidden="true" />
                                    </button>
                                    {isActionsMenuOpen ? (
                                      <div className="absolute right-0 top-full mt-1 w-36">
                                        <div className="relative rounded-xl border border-[#2b2b2b] bg-black p-1">
                                          <button
                                            type="button"
                                            onClick={() => {
                                              onOpenEditableSet(set.id);
                                              setActiveSetActionsMenuId(null);
                                            }}
                                            className="flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left text-xs text-white/90 transition hover:bg-white/10"
                                          >
                                            <i className="fa-solid fa-pen-to-square text-[11px] text-white/80" aria-hidden="true" />
                                            Edit
                                          </button>
                                          <button
                                            type="button"
                                            onClick={() => onToggleSetStar(set.id)}
                                            className="mt-0.5 flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left text-xs text-white/90 transition hover:bg-white/10"
                                          >
                                            <i className={`fa-${isStarred ? "solid" : "regular"} fa-star text-[11px] text-white/80`} aria-hidden="true" />
                                            {isStarred ? "Unstar" : "Star"}
                                          </button>
                                          <button
                                            type="button"
                                            onClick={() => {
                                              onRequestDeleteEditableSet(set.id);
                                            }}
                                            disabled={isDeleting}
                                            className="mt-0.5 flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left text-xs text-white/90 transition hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-60"
                                          >
                                            <i className="fa-regular fa-trash-can text-[11px] text-white/80" aria-hidden="true" />
                                            {isDeleting ? "Deleting..." : "Delete"}
                                          </button>
                                        </div>
                                      </div>
                                    ) : null}
                                  </div>
                                  <button
                                    type="button"
                                    onClick={() => onOpenEditableSet(set.id)}
                                    className="w-full text-left"
                                  >
                                    <div className="h-32 w-full overflow-hidden bg-black/45">
                                      {set.thumbnailUrl ? (
                                        <img src={set.thumbnailUrl} alt="" aria-hidden="true" className="h-full w-full object-cover transition duration-300 group-hover:scale-[1.03]" />
                                      ) : (
                                        <div className="grid h-full w-full place-items-center text-white/68">
                                          <i className={`${getSetIconClass(set)} text-base`} aria-hidden="true" />
                                        </div>
                                      )}
                                    </div>
                                    <div className="space-y-2 px-3 py-3">
                                      <div className="flex items-center gap-1.5">
                                        {isStarred ? <i className="fa-solid fa-star shrink-0 text-[11px] text-white" aria-hidden="true" /> : null}
                                        <p className="truncate text-sm font-semibold text-white">{set.title}</p>
                                      </div>
                                      <p className="line-clamp-2 text-xs leading-relaxed text-white/62">{set.description}</p>
                                      <div className="flex items-center justify-between text-[10px] font-semibold uppercase tracking-[0.08em] text-white/58">
                                        <span>{reelCount} reels</span>
                                        <span>{formatLastEditedLabel(set, relativeTimeNowMs)}</span>
                                      </div>
                                    </div>
                                  </button>
                                </div>
                              );
                            })}
                          </div>
                        ) : (
                          <p className="mt-4 text-sm text-white/66">
                            No sets yet. Use Create Set to publish your first one.
                          </p>
                        )}
                      </div>
                    </div>
                  </section>
                ) : null}
                {shouldShowEditSetForm ? (
                <section className="rounded-3xl px-1 pt-1 pb-2 sm:px-2 sm:pt-2 sm:pb-3 md:px-3 md:pt-3 md:pb-4">
                <div className="grid gap-4 lg:grid-cols-[minmax(0,1.3fr)_minmax(0,0.9fr)] lg:items-start">
                  <form onSubmit={onCreateSet} className="space-y-4 md:space-y-5">
                    <div className="relative overflow-hidden rounded-2xl border border-[#2b2b2b] bg-transparent p-4 pb-5 backdrop-blur-[4px] sm:p-5 sm:pb-6">
                      <div className="pointer-events-none absolute inset-0 bg-white/[0.04]" />
                      <div className="relative z-10 space-y-5">
                      <label className="block">
                        <span className="mb-2 flex items-center justify-between gap-2 text-xs text-white/72">
                          <span>Set Name</span>
                          <span className="text-[10px] text-white/45">{normalizedSetTitle.length}/70</span>
                        </span>
                        <input
                          value={setTitle}
                          onChange={(event) => setSetTitle(event.target.value)}
                          maxLength={70}
                          placeholder="Example: Organic Chemistry Reactions"
                          className="h-11 w-full rounded-xl border border-[#2b2b2b] bg-black/55 px-3 text-sm text-white outline-none placeholder:text-white/40 transition-colors focus:border-[#2b2b2b]"
                        />
                      </label>

                      <label className="block">
                        <span className="mb-2 flex items-center justify-between gap-2 text-xs text-white/72">
                          <span>Description</span>
                          <span className={`text-[10px] ${descriptionHasTooFewChars ? "text-[#ff8f8f]" : "text-[#9ef8cb]"}`}>
                            {normalizedSetDescription.length} / {MIN_SET_DESCRIPTION_LENGTH} min
                          </span>
                        </span>
                        <textarea
                          value={setDescription}
                          onChange={(event) => setSetDescription(event.target.value)}
                          placeholder="What does this set cover and who is it for?"
                          className={`h-24 w-full resize-none rounded-xl border bg-black/55 px-3 py-2 text-sm text-white outline-none placeholder:text-white/40 transition-colors md:h-24 ${
                            descriptionHasTooFewChars ? "border-[#ff8f8f]/70 focus:border-[#ff8f8f]" : "border-[#2b2b2b] focus:border-[#2b2b2b]"
                          }`}
                        />
                        {descriptionHasTooFewChars ? (
                          <p className="mt-1.5 text-[11px] text-[#ff8f8f]">
                            Description must be at least {MIN_SET_DESCRIPTION_LENGTH} characters. Add {descriptionCharsRemaining} more
                            {descriptionCharsRemaining === 1 ? " character." : " characters."}
                          </p>
                        ) : null}
                      </label>

                      <label className="block">
                        <span className="mb-2 flex items-center justify-between gap-2 text-xs text-white/72">
                          <span>Tags</span>
                          <span className={`text-[10px] ${tagLimitError ? "text-[#ff8f8f]" : "text-white/45"}`}>
                            {parsedSetTags.length}/{MAX_SET_TAGS}
                          </span>
                        </span>
                        <input
                          value={setTags}
                          onChange={onSetTagsChange}
                          placeholder={hasMaxTags ? "Max tags reached. Edit or remove one to add another." : "chemistry, reaction mechanisms, exam prep"}
                          className={`h-11 w-full rounded-xl border bg-black/55 px-3 text-sm text-white outline-none placeholder:text-white/40 transition-colors ${
                            tagLimitError
                              ? "border-[#ff8f8f]/70 focus:border-[#ff8f8f]"
                              : "border-[#2b2b2b] focus:border-[#2b2b2b]"
                          }`}
                        />
                        <p className={`mt-1.5 text-[11px] ${tagLimitError ? "text-[#ff8f8f]" : "text-zinc-400"}`}>
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
                                className="inline-flex items-center gap-1 rounded-full border border-[#2b2b2b] bg-white/6 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.08em] text-white/75 transition hover:bg-white/10 hover:text-white"
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
                        <p className="mb-2 block text-xs text-white/72">Thumbnail Image</p>
                        <input id="community-set-thumbnail" type="file" accept="image/*" className="sr-only" onChange={onThumbnailFileChange} />
                        <label
                          htmlFor="community-set-thumbnail"
                          onDragEnter={onThumbnailDragEnter}
                          onDragOver={onThumbnailDragOver}
                          onDragLeave={onThumbnailDragLeave}
                          onDrop={onThumbnailDrop}
                          className={`group relative block h-[220px] w-full cursor-pointer overflow-hidden rounded-xl border border-dashed ${
                            isThumbnailDragOver ? "border-white/60 bg-white/10" : "border-[#2b2b2b] bg-black/55"
                          } sm:h-[250px]`}
                        >
                          {thumbnailPreview ? (
                            <img
                              src={thumbnailPreview}
                              alt="Set thumbnail preview"
                              className="h-full w-full object-cover transition opacity-100"
                            />
                          ) : (
                            <div className="grid h-full w-full place-items-center bg-[linear-gradient(145deg,rgba(255,255,255,0.16),rgba(255,255,255,0.04))] text-white/70 transition group-hover:text-white/85">
                              <i className="fa-regular fa-image -translate-y-6 text-lg sm:-translate-y-7" aria-hidden="true" />
                            </div>
                          )}
                          <span className="absolute inset-0 grid place-items-center bg-black/35 text-white/85">
                            <span className="flex max-w-[90%] translate-y-5 flex-col items-center text-center sm:translate-y-6">
                              <span className={`truncate text-sm font-semibold ${thumbnailPreview ? "text-white" : "text-white/85"}`}>
                                {thumbnailPreview ? thumbnailFileName || "Image selected" : "Drag and drop your image here"}
                              </span>
                              <span className="mt-1 text-xs text-white/58">
                                {thumbnailPreview ? "Click to replace image" : "Or click to browse (PNG, JPG, WEBP)"}
                              </span>
                            </span>
                          </span>
                        </label>
                        <p className="mt-2 text-[11px] text-zinc-400">Use a vertical image for better mobile previews.</p>
                        {thumbnailPreview ? (
                          <button
                            type="button"
                            onClick={() => {
                              setThumbnailPreview("");
                              setThumbnailFileName("");
                            }}
                            className="mt-2 inline-flex items-center gap-1 rounded-lg border border-[#2b2b2b] bg-black/45 px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.08em] text-white/72 transition hover:bg-white/10 hover:text-white"
                          >
                            <i className="fa-solid fa-trash text-[9px]" aria-hidden="true" />
                            Remove
                          </button>
                        ) : null}
                      </div>
                    </div>
                    </div>

                    <div className="relative min-h-0 overflow-hidden rounded-2xl border border-[#2b2b2b] bg-transparent p-3.5 pb-4 backdrop-blur-[4px] sm:p-4 sm:pb-5">
                      <div className="pointer-events-none absolute inset-0 bg-white/[0.04]" />
                      <div className="relative z-10">
                      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                        <p className="text-[10px] font-semibold uppercase tracking-[0.09em] text-white/68">Embed Reels ({SUPPORTED_PLATFORMS_LABEL})</p>
                        <p className="text-[10px] font-semibold uppercase tracking-[0.08em] text-white/55">
                          {validDraftReelCount} valid / {invalidDraftReelCount} invalid
                        </p>
                      </div>
                      <div className="balanced-scroll-gutter max-h-[320px] space-y-3 overflow-y-auto">
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
                            <div key={row.id} className="rounded-xl p-3">
                              <div className="flex flex-col items-stretch gap-2 sm:flex-row sm:items-center sm:gap-3">
                                <input
                                  value={row.value}
                                  onChange={(event) => updateReelInputRow(row.id, event.target.value)}
                                  placeholder="Paste YouTube, Instagram reel, or TikTok URL"
                                  className="h-10 w-full rounded-lg bg-black/60 px-2.5 text-xs text-white outline-none placeholder:text-white/40 sm:h-9"
                                />
                                <button
                                  type="button"
                                  onClick={() => removeReelInputRow(row.id)}
                                  className="inline-flex h-9 w-full shrink-0 items-center justify-center gap-1 rounded-lg bg-black/55 text-white/72 transition hover:bg-white/10 hover:text-white sm:grid sm:h-8 sm:w-8 sm:place-items-center"
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
                                  Could not detect video length for this link yet. Slider appears after detection.
                                </p>
                              ) : null}

                              {hasInput && !hasValidEmbed ? (
                                <p className="mt-2 text-[11px] text-[#ffb4b4]">Invalid URL. Supported: {SUPPORTED_PLATFORMS_LABEL}.</p>
                              ) : null}
                              {hasInput && hasValidEmbed && !hasValidRange ? (
                                <p className="mt-2 text-[11px] text-[#ffb4b4]">Invalid clip range. If set, end must be greater than start.</p>
                              ) : null}

                              {row.parsed ? (
                                <div className="mt-3 overflow-hidden rounded-lg">
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
                                    src={row.parsed.embedUrl}
                                    title={`${PLATFORM_LABEL[row.parsed.platform]} reel preview`}
                                    className="h-[180px] w-full border-0 sm:h-[160px]"
                                    loading="lazy"
                                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                                    allowFullScreen
                                  />
                                </div>
                              ) : null}
                            </div>
                          );
                        })}
                      </div>
                      <button
                        type="button"
                        onClick={addReelInputRow}
                        className="mt-3 inline-flex items-center gap-1 px-1 py-1 text-[10px] font-semibold uppercase tracking-[0.08em] text-white/72 transition hover:text-white"
                      >
                        <i className="fa-solid fa-plus text-[9px]" aria-hidden="true" />
                        Add Reels
                      </button>
                    </div>
                    </div>

                    {isFormEditMode ? (
                      <div className="grid gap-2">
                        <button
                          type="submit"
                          disabled={!canPostSet}
                          className={`inline-flex h-11 w-full items-center justify-center rounded-xl border px-4 text-sm font-semibold transition ${
                            canPostSet
                              ? "border-[#2b2b2b] bg-black/55 text-white hover:bg-white hover:text-black"
                              : "cursor-not-allowed border-[#2b2b2b] bg-black/35 text-white/45"
                          }`}
                        >
                          {isPostingSet ? "Saving..." : "Save Set Changes"}
                        </button>
                      </div>
                    ) : (
                      <div className="grid gap-2 sm:grid-cols-3">
                        <button
                          type="button"
                          onClick={onClearCreateProgress}
                          className="inline-flex h-11 w-full items-center justify-center rounded-xl border border-[#2b2b2b] bg-black/35 px-4 text-sm font-semibold text-white/80 transition hover:bg-white/10 hover:text-white"
                        >
                          Clear Progress
                        </button>
                        <button
                          type="button"
                          onClick={onSaveDraftProgress}
                          className="inline-flex h-11 w-full items-center justify-center rounded-xl border border-[#2b2b2b] bg-black/35 px-4 text-sm font-semibold text-white/80 transition hover:bg-white/10 hover:text-white"
                        >
                          Save Progress
                        </button>
                        <button
                          type="submit"
                          disabled={!canPostSet}
                          className={`inline-flex h-11 w-full items-center justify-center rounded-xl border px-4 text-sm font-semibold transition ${
                            canPostSet
                              ? "border-[#2b2b2b] bg-black/55 text-white hover:bg-white hover:text-black"
                              : "cursor-not-allowed border-[#2b2b2b] bg-black/35 text-white/45"
                          }`}
                        >
                          {isPostingSet ? "Posting..." : "Post Community Set"}
                        </button>
                      </div>
                    )}
                  </form>

                  <aside className="relative overflow-hidden rounded-2xl border border-[#2b2b2b] bg-transparent p-4 pb-5 backdrop-blur-[4px] sm:p-5 sm:pb-6 lg:sticky lg:top-3">
                    <div className="pointer-events-none absolute inset-0 bg-white/[0.04]" />
                    <div className="relative z-10">
                      <p className="text-[10px] font-semibold uppercase tracking-[0.1em] text-white/62">Live Preview</p>
                      <div className="mt-3 overflow-hidden rounded-xl border border-[#2b2b2b] bg-black/45">
                        {thumbnailPreview ? (
                          <img
                            src={thumbnailPreview}
                            alt="Draft set cover"
                            className="h-[220px] w-full object-cover"
                          />
                        ) : (
                          <div className="grid h-[220px] w-full place-items-center bg-[linear-gradient(145deg,rgba(255,255,255,0.16),rgba(255,255,255,0.04))] text-white/70">
                            <i className="fa-regular fa-image text-lg" aria-hidden="true" />
                          </div>
                        )}
                      </div>
                      <h3 className="mt-3 text-lg font-semibold leading-tight text-white">
                        {normalizedSetTitle || "Your set title"}
                      </h3>
                      <p className="mt-2 text-sm leading-relaxed text-white/70">
                        {normalizedSetDescription || "Add a description to show what learners will get from this set."}
                      </p>
                      <div className="mt-3 flex flex-wrap gap-1.5">
                        {parsedSetTags.length > 0
                          ? parsedSetTags.map((tag) => (
                            <span key={`preview-tag-${tag}`} className="rounded-full bg-white/10 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.08em] text-white/72">
                              #{tag}
                            </span>
                          ))
                          : null}
                      </div>
                      <div className="mt-3 rounded-xl border border-[#2b2b2b] bg-black/35 px-3 py-3">
                        <div className="grid grid-cols-[auto_minmax(0,1fr)] items-start gap-4">
                          <div className="flex flex-col items-center">
                            <div className="relative h-[86px] w-[86px]">
                              <svg viewBox="0 0 88 88" className="h-full w-full -rotate-90" aria-hidden="true">
                                <circle cx="44" cy="44" r={progressRadius} stroke="rgba(255,255,255,0.14)" strokeWidth="7" fill="none" />
                                <circle
                                  cx="44"
                                  cy="44"
                                  r={progressRadius}
                                  stroke="#ffffff"
                                  strokeWidth="7"
                                  strokeLinecap="round"
                                  fill="none"
                                  strokeDasharray={progressCircumference}
                                  strokeDashoffset={progressOffset}
                                  style={{
                                    transition: "stroke-dashoffset 420ms cubic-bezier(0.22, 1, 0.36, 1)",
                                  }}
                                />
                              </svg>
                              <div className="absolute inset-0 flex flex-col items-center justify-center">
                                <span className="text-base font-semibold leading-none text-white">{progressPercent}%</span>
                              </div>
                            </div>
                            <p className="mt-2 text-center text-[10px] font-semibold uppercase tracking-[0.08em] text-white/62">
                              {requiredCompletionCount}/4 done
                            </p>
                          </div>
                          <div>
                            <p className="text-[10px] font-semibold uppercase tracking-[0.08em] text-white/58">
                              {isFormEditMode ? "Still needed to save" : "Still needed to post"}
                            </p>
                            {remainingPreviewRequirements.length > 0 ? (
                              <ul className="mt-2 space-y-1.5">
                                {remainingPreviewRequirements.map((item) => (
                                  <li key={`remaining-requirement-${item}`} className="flex items-start gap-2 text-xs text-white/80">
                                    <i className="fa-regular fa-circle text-[8px] text-white/55 mt-[4px]" aria-hidden="true" />
                                    <span>{item}</span>
                                  </li>
                                ))}
                              </ul>
                            ) : (
                              <p className="mt-2 text-xs text-[#9ef8cb]">All required items completed.</p>
                            )}
                          </div>
                        </div>
                      </div>
                      <div className="mt-3 grid grid-cols-2 gap-2">
                        <div className="rounded-lg border border-[#2b2b2b] bg-black/35 px-2.5 py-2">
                          <p className="text-[10px] uppercase tracking-[0.08em] text-white/52">Reels</p>
                          <p className="mt-1 text-sm font-semibold text-white">{validDraftReelCount}</p>
                        </div>
                        <div className="rounded-lg border border-[#2b2b2b] bg-black/35 px-2.5 py-2">
                          <p className="text-[10px] uppercase tracking-[0.08em] text-white/52">Status</p>
                          <p className={`mt-1 text-sm font-semibold ${canPostSet ? "text-[#9ef8cb]" : "text-white/76"}`}>
                            {canPostSet ? (isFormEditMode ? "Ready to save" : "Ready to post") : "Draft"}
                          </p>
                        </div>
                      </div>
                    </div>
                  </aside>
                </div>
              </section>
                ) : null}
            </div>
          </div>
          </div>
      </>
    )}
      {detailBannerPortal}
      {draftActionConfirmModal ? (
        <div
          className="fixed inset-0 z-[128] flex items-center justify-center bg-black/70 px-4 py-6 backdrop-blur-[2px] transition-opacity duration-200 ease-out opacity-100"
          role="presentation"
          onClick={closeDraftActionConfirmModal}
        >
          <div
            role="dialog"
            aria-modal="true"
            aria-label="Draft action confirmation"
            className="w-full max-w-xl rounded-3xl border border-white/25 bg-black p-5 text-white shadow-[0_18px_80px_rgba(0,0,0,0.5)] backdrop-blur-2xl transition-opacity duration-200 ease-out opacity-100 md:p-6"
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
                className="inline-flex h-8 w-8 items-center justify-center text-white/80 transition-colors hover:text-white focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-55"
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
                className="inline-flex min-w-[9rem] items-center justify-center whitespace-nowrap rounded-xl bg-black/35 px-5 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-55"
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
      ) : null}
      {unsavedDraftExitModal ? (
        <div
          className="fixed inset-0 z-[127] flex items-center justify-center bg-black/70 px-4 py-6 backdrop-blur-[2px] transition-opacity duration-200 ease-out opacity-100"
          role="presentation"
          onClick={closeUnsavedDraftExitModal}
        >
          <div
            role="dialog"
            aria-modal="true"
            aria-label="Unsaved set draft changes"
            className="w-full max-w-xl rounded-3xl border border-white/25 bg-black p-5 text-white shadow-[0_18px_80px_rgba(0,0,0,0.5)] backdrop-blur-2xl transition-opacity duration-200 ease-out opacity-100 md:p-6"
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
                className="inline-flex h-8 w-8 items-center justify-center text-white/80 transition-colors hover:text-white focus-visible:outline-none"
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
                className="inline-flex min-w-[9rem] items-center justify-center whitespace-nowrap rounded-xl bg-black/35 px-5 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-60"
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
      ) : null}
      {deleteSetConfirmModal ? (
        <div
          className="fixed inset-0 z-[126] flex items-center justify-center bg-black/70 px-4 py-6 backdrop-blur-[2px] transition-opacity duration-200 ease-out opacity-100"
          role="presentation"
          onClick={closeDeleteSetConfirmModal}
        >
          <div
            role="dialog"
            aria-modal="true"
            aria-label="Delete set confirmation"
            className="w-full max-w-xl rounded-3xl border border-white/25 bg-black p-5 text-white shadow-[0_18px_80px_rgba(0,0,0,0.5)] backdrop-blur-2xl transition-opacity duration-200 ease-out opacity-100 md:p-6"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="text-[11px] font-semibold uppercase tracking-[0.12em] text-white/65">Delete Set</p>
                <h3 className="mt-2 text-lg font-semibold text-white">Delete "{deleteSetConfirmModal.title}"?</h3>
              </div>
              <button
                type="button"
                onClick={closeDeleteSetConfirmModal}
                aria-label="Close"
                disabled={Boolean(deletingSetId)}
                className="inline-flex h-8 w-8 items-center justify-center text-white/80 transition-colors hover:text-white focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-55"
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
                disabled={Boolean(deletingSetId)}
                className="inline-flex min-w-[9rem] items-center justify-center whitespace-nowrap rounded-xl bg-black/35 px-5 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-55"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={() => {
                  void onDeleteEditableSet(deleteSetConfirmModal.setId);
                }}
                disabled={Boolean(deletingSetId)}
                className="inline-flex min-w-[9rem] items-center justify-center whitespace-nowrap rounded-xl bg-white px-5 py-2.5 text-sm font-semibold text-black transition-colors hover:bg-white/90 disabled:cursor-not-allowed disabled:opacity-70"
              >
                {deletingSetId === deleteSetConfirmModal.setId ? "Deleting..." : "Delete"}
              </button>
            </div>
          </div>
        </div>
      ) : null}
      {publishResultModal ? (
        <div
          className="fixed inset-0 z-[125] flex items-center justify-center bg-black/70 px-4 py-6 backdrop-blur-[2px]"
          role="presentation"
          onClick={() => setPublishResultModal(null)}
        >
          <div
            role="dialog"
            aria-modal="true"
            aria-label="Publish result"
            className="w-full max-w-xl rounded-3xl border border-white/25 bg-black p-5 text-white shadow-[0_18px_80px_rgba(0,0,0,0.5)] backdrop-blur-2xl md:p-6"
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
                className="inline-flex h-8 w-8 items-center justify-center text-white/80 transition-colors hover:text-white focus-visible:outline-none"
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
                    : "bg-black/35 text-white hover:bg-white/10"
                }`}
              >
                OK
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
