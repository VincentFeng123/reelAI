"use client";

import { type FormEvent, type KeyboardEvent, type RefObject, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";

import { BillingGateDialog, type BillingGateReason } from "@/components/BillingGateDialog";
import { CustomSelect } from "@/components/CustomSelect";
import {
  ingestUrl,
  isDailySearchLimitError,
  isRequestInterruptedError,
  isVerifiedAccountRequiredError,
  uploadMaterial,
} from "@/lib/api";
import { safeStorageSetItem } from "@/lib/browserStorage";
import { buildSearchFeedQuery } from "@/lib/feedQuery";
import { type GenerationMode, readStudyReelsSettings } from "@/lib/settings";
import { requestBillingStatusRefresh } from "@/lib/useBillingStatus";
import { CURRENT_SELECTION_CONTRACT_VERSION, type BillingStatus, type CommunityAccount, type Reel } from "@/lib/types";

const MATERIAL_SEEDS_STORAGE_KEY = "studyreels-material-seeds";
const MATERIAL_GROUPS_STORAGE_KEY = "studyreels-material-groups";
const MAX_MATERIAL_SEEDS = 120;
const MAX_MATERIAL_GROUPS = 80;
const MAX_SEED_TEXT_CHARS = 16000;
const INGEST_SENTINEL_MATERIAL_ID = "ingest-scratch";
// Must stay in sync with FEED_SESSION_STORAGE_KEY in src/app/feed/page.tsx — both
// read/write the same localStorage key. Priming the session snapshot here lets the
// feed page hydrate with ingested reels on mount instead of submitting a durable
// generation job for an ingest-search/ingest-scratch sentinel material, which has
// no independently searchable material record.
const FEED_SESSION_STORAGE_KEY = "studyreels-feed-sessions";
type KnowledgeLevel = "beginner" | "intermediate" | "advanced";
type HomeIdeaIconName =
  | "fa-building"
  | "fa-chart-bar"
  | "fa-chess-knight"
  | "fa-clipboard"
  | "fa-clock"
  | "fa-cloud"
  | "fa-comments"
  | "fa-compass"
  | "fa-file-code"
  | "fa-file-lines"
  | "fa-flag"
  | "fa-headphones"
  | "fa-hospital"
  | "fa-keyboard"
  | "fa-lightbulb"
  | "fa-map"
  | "fa-money-bill-1"
  | "fa-moon"
  | "fa-newspaper"
  | "fa-paper-plane"
  | "fa-rectangle-list"
  | "fa-star"
  | "fa-sun";

const HOME_IDEAS_PER_PAGE = 4;
const HOME_IDEA_ROTATION_MS = 5 * 60 * 1_000;
const HOME_IDEA_OFFSET_STORAGE_KEY = "studyreels-home-idea-offset";

export const HOME_IDEAS: ReadonlyArray<{ title: string; icon: HomeIdeaIconName }> = [
  { title: "Learn calculus basics", icon: "fa-chart-bar" },
  { title: "How do black holes work?", icon: "fa-moon" },
  { title: "Intro to Python", icon: "fa-file-code" },
  { title: "The French Revolution", icon: "fa-flag" },
  { title: "Understand supply and demand", icon: "fa-chart-bar" },
  { title: "How does DNA store information?", icon: "fa-file-lines" },
  { title: "Practice conversational Spanish", icon: "fa-comments" },
  { title: "Explore the psychology of habits", icon: "fa-lightbulb" },
  { title: "Explain quantum entanglement", icon: "fa-star" },
  { title: "Learn SQL joins visually", icon: "fa-file-code" },
  { title: "How do vaccines train immunity?", icon: "fa-hospital" },
  { title: "Why did the Roman Empire fall?", icon: "fa-building" },
  { title: "Master fractions step by step", icon: "fa-chart-bar" },
  { title: "How does the internet work?", icon: "fa-compass" },
  { title: "Learn to read a balance sheet", icon: "fa-newspaper" },
  { title: "Introduction to music theory", icon: "fa-headphones" },
  { title: "What drives climate change?", icon: "fa-cloud" },
  { title: "Learn linear algebra basics", icon: "fa-rectangle-list" },
  { title: "How do neural networks learn?", icon: "fa-lightbulb" },
  { title: "What sparked the Industrial Revolution?", icon: "fa-clock" },
  { title: "Practice probability with examples", icon: "fa-chart-bar" },
  { title: "Why does gravity slow time?", icon: "fa-moon" },
  { title: "Build a simple JavaScript app", icon: "fa-keyboard" },
  { title: "Understand natural selection", icon: "fa-compass" },
  { title: "Learn the scientific method", icon: "fa-clipboard" },
  { title: "What causes inflation?", icon: "fa-money-bill-1" },
  { title: "Explore ancient Egyptian history", icon: "fa-building" },
  { title: "How does human memory work?", icon: "fa-lightbulb" },
  { title: "Solve quadratic equations", icon: "fa-file-lines" },
  { title: "Learn basic statistics", icon: "fa-chart-bar" },
  { title: "How do airplanes stay in the air?", icon: "fa-paper-plane" },
  { title: "Introduction to philosophy", icon: "fa-comments" },
  { title: "How does photosynthesis work?", icon: "fa-sun" },
  { title: "Learn the basics of chess strategy", icon: "fa-chess-knight" },
  { title: "How do maps represent the world?", icon: "fa-map" },
  { title: "What makes a star explode?", icon: "fa-star" },
];

export function homeGreetingForHour(hour: number, username?: string | null): string {
  const base = hour >= 5 && hour < 12
    ? "Good morning"
    : hour >= 12 && hour < 17
      ? "Good afternoon"
      : "Good evening";
  const name = username?.trim();
  return name ? `${base}, ${name}` : base;
}

function HomeIdeaIcon({ name }: { name: HomeIdeaIconName }) {
  return (
    <i
      aria-hidden="true"
      className={`fa-regular ${name} w-[18px] shrink-0 text-center text-[15px] font-normal leading-none`}
    />
  );
}

export type UnifiedComposerRoute = "file" | "url" | "source" | "topic";

const KNOWLEDGE_LEVEL_OPTIONS: Array<{ value: KnowledgeLevel; label: string }> = [
  { value: "beginner", label: "Beginner" },
  { value: "intermediate", label: "Intermediate" },
  { value: "advanced", label: "Advanced" },
];

const INGEST_URL_HOST_ALLOWLIST = new Set([
  "youtube.com",
  "www.youtube.com",
  "m.youtube.com",
  "youtu.be",
  "music.youtube.com",
]);
const YOUTUBE_VIDEO_ID_RE = /^[A-Za-z0-9_-]{11}$/;

/**
 * Lightweight client-side sanity check so we don't POST every keystroke to the backend.
 * The backend still does the authoritative host check via the yt-dlp adapter.
 */
export function isLikelyIngestUrl(raw: string): boolean {
  const trimmed = raw.trim();
  if (!trimmed) {
    return false;
  }
  const candidate = /^https?:\/\//i.test(trimmed) ? trimmed : `https://${trimmed}`;
  try {
    const parsed = new URL(candidate);
    const host = parsed.hostname.toLowerCase();
    if (!INGEST_URL_HOST_ALLOWLIST.has(host)) {
      return false;
    }
    if (host === "youtu.be") {
      return YOUTUBE_VIDEO_ID_RE.test(parsed.pathname.split("/").filter(Boolean)[0] || "");
    }
    const parts = parsed.pathname.split("/").filter(Boolean);
    if (parsed.pathname === "/watch") {
      return YOUTUBE_VIDEO_ID_RE.test(parsed.searchParams.get("v") || "");
    }
    if (parsed.pathname === "/playlist") {
      return Boolean(parsed.searchParams.get("list")?.trim());
    }
    if (["shorts", "embed", "live"].includes(parts[0] || "")) {
      return YOUTUBE_VIDEO_ID_RE.test(parts[1] || "");
    }
    if (parts[0]?.startsWith("@")) {
      return parts[0].length > 1;
    }
    return ["channel", "c", "user"].includes(parts[0] || "") && Boolean(parts[1]?.trim());
  } catch {
    return false;
  }
}

export function resolveUnifiedComposerRoute(params: {
  attachment?: { name: string } | null;
  prompt: string;
}): UnifiedComposerRoute {
  if (params.attachment) {
    return "file";
  }
  const prompt = params.prompt.trim();
  if (isLikelyIngestUrl(prompt)) {
    return "url";
  }
  if (/[\r\n]/.test(params.prompt) || prompt.length > 80) {
    return "source";
  }
  return "topic";
}

type MaterialSeed = {
  topic?: string;
  text?: string;
  knowledgeLevel?: KnowledgeLevel;
  title: string;
  updatedAt: number;
};

type MaterialGroup = {
  materialIds: string[];
  title?: string;
  updatedAt: number;
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
    const normalized: Record<string, MaterialSeed> = {};
    for (const [id, value] of Object.entries(parsed as Record<string, unknown>)) {
      if (!id || typeof id !== "string") {
        continue;
      }
      if (!value || typeof value !== "object" || Array.isArray(value)) {
        continue;
      }
      const seed = value as Record<string, unknown>;
      const title = String(seed.title || "").trim();
      const storedKnowledgeLevel = String(seed.knowledgeLevel || "").trim();
      const knowledgeLevel = storedKnowledgeLevel === "beginner"
        || storedKnowledgeLevel === "intermediate"
        || storedKnowledgeLevel === "advanced"
        ? storedKnowledgeLevel
        : undefined;
      if (!title) {
        continue;
      }
      normalized[id] = {
        topic: String(seed.topic || "").trim() || undefined,
        text: String(seed.text || "").trim() || undefined,
        knowledgeLevel,
        title,
        updatedAt: Number(seed.updatedAt) || 0,
      };
    }
    return normalized;
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
    const normalized: Record<string, MaterialGroup> = {};
    for (const [id, value] of Object.entries(parsed as Record<string, unknown>)) {
      if (!id || typeof id !== "string") {
        continue;
      }
      if (!value || typeof value !== "object" || Array.isArray(value)) {
        continue;
      }
      const group = value as Record<string, unknown>;
      const materialIds = Array.isArray(group.materialIds)
        ? Array.from(
            new Set(
              group.materialIds
                .map((row) => String(row || "").trim())
                .filter(Boolean),
            ),
          )
        : [];
      if (materialIds.length === 0) {
        continue;
      }
      normalized[id] = {
        materialIds,
        title: typeof group.title === "string" && group.title.trim() ? group.title.trim() : undefined,
        updatedAt: Number(group.updatedAt) || 0,
      };
    }
    return normalized;
  } catch {
    return {};
  }
}

type UploadPanelProps = {
  active?: boolean;
  account?: CommunityAccount | null;
  billingStatus?: BillingStatus | null;
  demoMode?: boolean;
  onMaterialCreated?: (params: {
    materialId: string;
    title: string;
    topic?: string;
    generationMode: GenerationMode;
    feedQuery: string;
  }) => void | Promise<void>;
  onScrollOffsetChange?: (isOffset: boolean) => void;
  onScrollGesture?: () => void;
  onScrollabilityChange?: (isScrollable: boolean) => void;
  heroTitleRef?: RefObject<HTMLHeadingElement | null>;
};

/**
 * Minimal shape of the feed-page's `FeedSessionSnapshot` stored under
 * `FEED_SESSION_STORAGE_KEY`. Intentionally duplicated here rather than imported
 * from `src/app/feed/page.tsx` (which is a Next.js page module). Field names must
 * exactly match `feed/page.tsx` or hydration will silently fail.
 */
type PrimedFeedSessionSnapshot = {
  selectionContractVersion: string;
  reels: Reel[];
  page: number;
  total: number;
  canRequestMore: boolean;
  generationMode: GenerationMode;
  mutedPreference: boolean;
  autoplayEnabled: boolean;
  playbackRate: number;
  activeIndex: number;
  activeReelId?: string;
  updatedAt: number;
};

/**
 * Write a snapshot into localStorage under the feed page's session key so that
 * `feed/page.tsx`'s hydration on mount picks up the primed reels and skips the
 * legacy generate path. Robust against localStorage quota failures (they're
 * silently ignored — the feed page will fall through to its normal bootstrap
 * flow, which our `isIngestMaterial` short-circuit in the feed page also handles).
 */
function primeFeedSessionSnapshot(
  materialId: string,
  reels: Reel[],
  activeReelId: string | undefined,
  settings: ReturnType<typeof readStudyReelsSettings>,
): void {
  if (typeof window === "undefined" || reels.length === 0) {
    return;
  }
  let activeIndex = 0;
  if (activeReelId) {
    const found = reels.findIndex((r) => r.reel_id === activeReelId);
    if (found >= 0) {
      activeIndex = found;
    }
  }
  const snapshot: PrimedFeedSessionSnapshot = {
    selectionContractVersion: CURRENT_SELECTION_CONTRACT_VERSION,
    reels,
    page: 1,
    total: reels.length,
    canRequestMore: false,
    generationMode: settings.generationMode,
    mutedPreference: settings.startMuted,
    autoplayEnabled: settings.autoplayNextReel,
    playbackRate: 1,
    activeIndex,
    activeReelId,
    updatedAt: Date.now(),
  };
  try {
    const raw = window.localStorage.getItem(FEED_SESSION_STORAGE_KEY);
    let existing: Record<string, unknown> = {};
    if (raw) {
      try {
        const parsed = JSON.parse(raw);
        if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
          existing = parsed as Record<string, unknown>;
        }
      } catch {
        existing = {};
      }
    }
    existing[materialId] = snapshot;
    safeStorageSetItem(window.localStorage, FEED_SESSION_STORAGE_KEY, JSON.stringify(existing));
  } catch {
    // Ignore storage failures — the feed page will still handle ingest materials
    // correctly via its isIngestMaterial short-circuit.
  }
}

function buildMaterialTitle(params: { topic: string; text: string; fileName: string }): string {
  const topic = params.topic.trim();
  if (topic) {
    return topic;
  }
  if (params.fileName.trim()) {
    return params.fileName.trim();
  }
  const snippet = params.text.trim().replace(/\s+/g, " ").slice(0, 58);
  if (snippet) {
    return snippet;
  }
  return "New Study Session";
}

export function UploadPanel({ active = true, account, billingStatus = null, demoMode = false, onMaterialCreated, onScrollOffsetChange, onScrollGesture, onScrollabilityChange, heroTitleRef }: UploadPanelProps) {
  const router = useRouter();
  const touchStartYRef = useRef<number | null>(null);
  const formRef = useRef<HTMLFormElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const didRotateHomeIdeasOnMountRef = useRef(false);
  const [prompt, setPrompt] = useState("");
  const [file, setFile] = useState<File | undefined>();
  const [knowledgeLevel, setKnowledgeLevel] = useState<KnowledgeLevel>("beginner");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [billingGate, setBillingGate] = useState<{ reason: BillingGateReason; requiredSearches: number } | null>(null);
  const [localHour, setLocalHour] = useState<number | null>(null);
  const [homeIdeaOffset, setHomeIdeaOffset] = useState(0);
  const abortControllerRef = useRef<AbortController | null>(null);
  const selectedFileName = file?.name ?? "";
  // The parent hydrates this account after mount. Avoid reading localStorage during
  // render, which would make the first browser render differ from the server HTML.
  const verifiedAccount = account?.isVerified ? account : null;
  const searchCost = 1;
  const composerRoute = useMemo(
    () => resolveUnifiedComposerRoute({ attachment: file, prompt }),
    [file, prompt],
  );
  const visibleHomeIdeas = useMemo(
    () => HOME_IDEAS.slice(homeIdeaOffset, homeIdeaOffset + HOME_IDEAS_PER_PAGE),
    [homeIdeaOffset],
  );
  const advanceHomeIdeas = useCallback(() => {
    setHomeIdeaOffset((offset) => {
      const nextOffset = (offset + HOME_IDEAS_PER_PAGE) % HOME_IDEAS.length;
      safeStorageSetItem(window.sessionStorage, HOME_IDEA_OFFSET_STORAGE_KEY, String(nextOffset));
      return nextOffset;
    });
  }, []);

  const disabled = loading || (!file && !prompt.trim());

  const resizeTextarea = useCallback(() => {
    const element = textareaRef.current;
    if (!element) {
      return;
    }
    element.style.height = "auto";
    // Three 28px lines plus the textarea's vertical padding.
    const nextHeight = Math.min(element.scrollHeight, 92);
    element.style.height = `${Math.max(nextHeight, 72)}px`;
    element.style.overflowY = element.scrollHeight > 92 ? "auto" : "hidden";
  }, []);

  useEffect(() => {
    resizeTextarea();
  }, [prompt, resizeTextarea]);

  useEffect(() => {
    const frame = window.requestAnimationFrame(() => setLocalHour(new Date().getHours()));
    return () => window.cancelAnimationFrame(frame);
  }, []);

  useEffect(() => {
    if (didRotateHomeIdeasOnMountRef.current) {
      return;
    }
    didRotateHomeIdeasOnMountRef.current = true;
    let previousOffset = 0;
    try {
      const storedOffset = Number(window.sessionStorage.getItem(HOME_IDEA_OFFSET_STORAGE_KEY));
      if (
        Number.isInteger(storedOffset)
        && storedOffset >= 0
        && storedOffset < HOME_IDEAS.length
        && storedOffset % HOME_IDEAS_PER_PAGE === 0
      ) {
        previousOffset = storedOffset;
      }
    } catch {
      // Storage can be unavailable in privacy-restricted browser contexts.
    }
    const refreshOffset = (previousOffset + HOME_IDEAS_PER_PAGE) % HOME_IDEAS.length;
    setHomeIdeaOffset(refreshOffset);
    safeStorageSetItem(window.sessionStorage, HOME_IDEA_OFFSET_STORAGE_KEY, String(refreshOffset));
  }, []);

  useEffect(() => {
    if (!active || prompt.trim()) {
      return;
    }
    const interval = window.setInterval(advanceHomeIdeas, HOME_IDEA_ROTATION_MS);
    return () => window.clearInterval(interval);
  }, [active, advanceHomeIdeas, prompt]);

  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort();
    };
  }, []);

  useEffect(() => {
    if (!active) {
      abortControllerRef.current?.abort();
    }
  }, [active]);

  useEffect(() => {
    const abortActiveRequest = () => abortControllerRef.current?.abort();
    window.addEventListener("beforeunload", abortActiveRequest);
    window.addEventListener("pagehide", abortActiveRequest);
    window.addEventListener("unload", abortActiveRequest);
    return () => {
      window.removeEventListener("beforeunload", abortActiveRequest);
      window.removeEventListener("pagehide", abortActiveRequest);
      window.removeEventListener("unload", abortActiveRequest);
    };
  }, []);

  const onSubmit = useCallback(async (event: FormEvent) => {
    event.preventDefault();
    if (!active || loading || (!file && !prompt.trim())) {
      return;
    }
    if (demoMode) {
      router.push("/feed?demo=player&return_tab=search");
      return;
    }
    if (!verifiedAccount?.isVerified) {
      setBillingGate({ reason: "sign_in", requiredSearches: searchCost });
      return;
    }
    if (billingStatus && billingStatus.remaining_searches < searchCost) {
      setBillingGate({ reason: "quota", requiredSearches: searchCost });
      return;
    }
    abortControllerRef.current?.abort();
    const controller = new AbortController();
    abortControllerRef.current = controller;
    setError(null);
    setLoading(true);

    try {
      const activeSettings = readStudyReelsSettings();
      const generationModeForSearch = activeSettings.generationMode;

      // Topic / text / file submits create material records; uncached reel
      // generation is owned by the durable backend job lifecycle.

      // URL ingest path diverges completely from the material-upload path: we call
      // /api/ingest/url instead of /api/material, then land on the feed scoped to
      // the `ingest-scratch` sentinel material so prior ingests form a scrollable feed.
      if (composerRoute === "url") {
        const trimmed = prompt.trim();
        const normalized = /^https?:\/\//i.test(trimmed) ? trimmed : `https://${trimmed}`;
        const result = await ingestUrl({
          sourceUrl: normalized,
          signal: controller.signal,
        });
        if (controller.signal.aborted) {
          return;
        }
        const ingestedReels = Array.isArray(result.reels) && result.reels.length > 0
          ? result.reels
          : [result.reel];
        const ingestedReel = ingestedReels[0];
        const ingestedMetadata = result.metadata;
        const ingestMaterialId = ingestedReel.material_id || INGEST_SENTINEL_MATERIAL_ID;
        const ingestTitle =
          (ingestedReel.video_title?.trim() ||
            ingestedMetadata.title?.trim() ||
            (ingestedMetadata.author_handle ? `@${ingestedMetadata.author_handle}` : "") ||
            "Ingested reel").slice(0, 58);

        // Prime every verified clip returned by the URL adapter. Older backends
        // expose only `reel`, which is normalized into the fallback array above.
        primeFeedSessionSnapshot(ingestMaterialId, ingestedReels, ingestedReel.reel_id, activeSettings);

        if (typeof window !== "undefined") {
          const seeds = parseMaterialSeeds(window.localStorage.getItem(MATERIAL_SEEDS_STORAGE_KEY));
          seeds[ingestMaterialId] = {
            topic: undefined,
            text: undefined,
            title: ingestTitle,
            updatedAt: Date.now(),
          };
          const ordered = Object.entries(seeds)
            .sort((a, b) => (b[1].updatedAt || 0) - (a[1].updatedAt || 0))
            .slice(0, MAX_MATERIAL_SEEDS);
          safeStorageSetItem(window.localStorage, MATERIAL_SEEDS_STORAGE_KEY, JSON.stringify(Object.fromEntries(ordered)));
        }

        if (onMaterialCreated) {
          const feedQuery = buildSearchFeedQuery({
            materialId: ingestMaterialId,
            generationMode: generationModeForSearch,
            returnTab: "search",
            settings: activeSettings,
          });
          await onMaterialCreated({
            materialId: ingestMaterialId,
            title: ingestTitle,
            topic: undefined,
            generationMode: generationModeForSearch,
            feedQuery,
          });
          if (controller.signal.aborted) {
            return;
          }
        }

        const ingestFeedQuery = buildSearchFeedQuery({
          materialId: ingestMaterialId,
          generationMode: generationModeForSearch,
          returnTab: "search",
          settings: activeSettings,
        });
        router.push(`/feed?${ingestFeedQuery}&active_reel_id=${encodeURIComponent(ingestedReel.reel_id)}`);
        requestBillingStatusRefresh();
        setPrompt("");
        return;
      }

      const promptValue = prompt.trim();
      const topicValue = composerRoute === "topic" ? promptValue : "";
      const textValue = composerRoute === "source" || composerRoute === "file" ? promptValue : "";
      const fileValue = composerRoute === "file" ? file : undefined;
      const title = buildMaterialTitle({
        topic: topicValue,
        text: textValue,
        fileName: fileValue?.name ?? "",
      });
      let materialIds: string[] = [];
      const material = await uploadMaterial({
        text: textValue || undefined,
        file: fileValue,
        subjectTag: topicValue || undefined,
        knowledgeLevel,
        signal: controller.signal,
      });
      if (controller.signal.aborted) {
        return;
      }
      materialIds = [material.material_id];

      const primaryMaterialId = materialIds[0];
      if (!primaryMaterialId) {
        throw new Error("Material creation failed.");
      }

      if (typeof window !== "undefined") {
        const seeds = parseMaterialSeeds(window.localStorage.getItem(MATERIAL_SEEDS_STORAGE_KEY));
        const now = Date.now();
        seeds[primaryMaterialId] = {
          topic: topicValue || undefined,
          text: textValue ? textValue.slice(0, MAX_SEED_TEXT_CHARS) : undefined,
          knowledgeLevel,
          title,
          updatedAt: now,
        };
        const ordered = Object.entries(seeds)
          .sort((a, b) => (b[1].updatedAt || 0) - (a[1].updatedAt || 0))
          .slice(0, MAX_MATERIAL_SEEDS);
        safeStorageSetItem(window.localStorage, MATERIAL_SEEDS_STORAGE_KEY, JSON.stringify(Object.fromEntries(ordered)));

        const groups = parseMaterialGroups(window.localStorage.getItem(MATERIAL_GROUPS_STORAGE_KEY));
        delete groups[primaryMaterialId];
        const orderedGroups = Object.entries(groups)
          .sort((a, b) => (b[1].updatedAt || 0) - (a[1].updatedAt || 0))
          .slice(0, MAX_MATERIAL_GROUPS);
        safeStorageSetItem(window.localStorage, MATERIAL_GROUPS_STORAGE_KEY, JSON.stringify(Object.fromEntries(orderedGroups)));
      }
      if (onMaterialCreated) {
        const feedQuery = buildSearchFeedQuery({
          materialId: primaryMaterialId,
          generationMode: generationModeForSearch,
          returnTab: "search",
          settings: activeSettings,
        });
        await onMaterialCreated({
          materialId: primaryMaterialId,
          title,
          topic: topicValue || undefined,
          generationMode: generationModeForSearch,
          feedQuery,
        });
        if (controller.signal.aborted) {
          return;
        }
      }
      const nextQuery = buildSearchFeedQuery({
        materialId: primaryMaterialId,
        generationMode: generationModeForSearch,
        returnTab: "search",
        settings: activeSettings,
      });
      router.push(`/feed?${nextQuery}`);
      requestBillingStatusRefresh();
      setPrompt("");
      setFile(undefined);
    } catch (e) {
      if (isRequestInterruptedError(e) || (e instanceof DOMException && e.name === "AbortError")) {
        return;
      }
      if (isDailySearchLimitError(e)) {
        setBillingGate({ reason: "quota", requiredSearches: searchCost });
        requestBillingStatusRefresh();
        return;
      }
      if (isVerifiedAccountRequiredError(e)) {
        setBillingGate({ reason: "sign_in", requiredSearches: searchCost });
        return;
      }
      setError(e instanceof Error ? e.message : "Something failed");
    } finally {
      if (abortControllerRef.current === controller) {
        abortControllerRef.current = null;
        setLoading(false);
      }
    }
  }, [active, billingStatus, composerRoute, demoMode, file, knowledgeLevel, loading, onMaterialCreated, prompt, router, searchCost, verifiedAccount]);

  const reportScrollability = useCallback(() => {
    const element = formRef.current;
    if (!element) {
      return;
    }
    const isScrollable = element.scrollHeight - element.clientHeight > 1;
    onScrollabilityChange?.(isScrollable);
    if (!isScrollable) {
      onScrollOffsetChange?.(false);
    }
  }, [onScrollOffsetChange, onScrollabilityChange]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    reportScrollability();
    const element = formRef.current;
    if (!element) {
      return;
    }
    const onResize = () => {
      reportScrollability();
    };
    window.addEventListener("resize", onResize);
    const observer = typeof ResizeObserver !== "undefined" ? new ResizeObserver(onResize) : null;
    observer?.observe(element);
    return () => {
      window.removeEventListener("resize", onResize);
      observer?.disconnect();
    };
  }, [reportScrollability]);

  useEffect(() => {
    reportScrollability();
  }, [error, loading, prompt, reportScrollability, selectedFileName]);

  const onPromptKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (
      event.key !== "Enter"
      || event.shiftKey
      || event.nativeEvent.isComposing
      || event.nativeEvent.keyCode === 229
    ) {
      return;
    }
    if (typeof window !== "undefined" && window.matchMedia("(pointer: coarse)").matches) {
      return;
    }
    event.preventDefault();
    if (disabled) {
      return;
    }
    formRef.current?.requestSubmit();
  };

  const compactStatus = loading
    ? "Preparing your reels…"
    : error
      ? error
      : !verifiedAccount?.isVerified
        ? "Sign in with a verified account to search."
        : billingStatus && billingStatus.remaining_searches <= Math.min(3, billingStatus.daily_limit)
          ? `${billingStatus.remaining_searches} search${billingStatus.remaining_searches === 1 ? "" : "es"} left today.`
          : null;
  const showBottomSignInStatus = !loading && !error && !verifiedAccount?.isVerified;

  return (
    <form
      ref={formRef}
      onSubmit={onSubmit}
      onWheelCapture={(event) => {
        const isScrollable = event.currentTarget.scrollHeight - event.currentTarget.clientHeight > 1;
        if (isScrollable && event.deltaY > 0) {
          onScrollGesture?.();
        }
      }}
      onTouchStartCapture={(event) => {
        const nextTouch = event.touches.item(0);
        touchStartYRef.current = nextTouch ? nextTouch.clientY : null;
      }}
      onTouchMoveCapture={(event) => {
        const startY = touchStartYRef.current;
        const nextTouch = event.touches.item(0);
        if (startY === null || !nextTouch) {
          return;
        }
        const isScrollable = event.currentTarget.scrollHeight - event.currentTarget.clientHeight > 1;
        if (isScrollable && startY - nextTouch.clientY > 0) {
          onScrollGesture?.();
        }
      }}
      onTouchEndCapture={() => {
        touchStartYRef.current = null;
      }}
      onScrollCapture={(event) => {
        const isScrollable = event.currentTarget.scrollHeight - event.currentTarget.clientHeight > 1;
        onScrollOffsetChange?.(isScrollable && event.currentTarget.scrollTop > 0);
      }}
      onScroll={(event) => {
        const isScrollable = event.currentTarget.scrollHeight - event.currentTarget.clientHeight > 1;
        onScrollOffsetChange?.(isScrollable && event.currentTarget.scrollTop > 0);
      }}
      className="relative mx-auto flex h-full w-full max-w-[960px] flex-col items-center justify-center overflow-y-auto px-4 py-20 sm:px-8"
    >
      <header
        className={`mb-8 text-center transition-opacity duration-[420ms] ease-in-out motion-reduce:transition-none sm:mb-10 ${
          localHour === null ? "opacity-0" : "opacity-100"
        }`}
      >
        <h1 ref={heroTitleRef} className="text-[clamp(1.65rem,5vw,2rem)] font-semibold tracking-[-0.035em] text-white">
          {homeGreetingForHour(localHour ?? 12, account?.username)}
        </h1>
        <p className="mt-2 text-sm text-white/50 sm:text-[15px]">What can I help you learn today?</p>
      </header>

      <input
        ref={fileInputRef}
        id="material-file"
        className="hidden"
        tabIndex={-1}
        aria-hidden="true"
        type="file"
        accept=".pdf,.docx,.txt,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,text/plain"
        onChange={(event) => {
          setFile(event.target.files?.[0]);
          setError(null);
        }}
      />

      <div className="w-full max-w-[680px]">
        <div className="rounded-[28px] border-[0.5px] border-[#3a3a3a] bg-[#242424] px-3 py-3">
          {file ? (
            <div className="mb-2 flex px-1 pt-1">
              <div className="flex min-w-0 max-w-full items-center gap-2 rounded-2xl bg-[#343434] py-2 pl-3 pr-2 text-sm text-white">
                <i className="fa-regular fa-file-lines shrink-0 text-white/70" aria-hidden="true" />
                <span className="max-w-[15rem] truncate sm:max-w-[26rem]">{selectedFileName}</span>
                <button
                  type="button"
                  aria-label={`Remove ${selectedFileName}`}
                  onClick={() => {
                    setFile(undefined);
                    if (fileInputRef.current) {
                      fileInputRef.current.value = "";
                    }
                  }}
                  className="grid h-7 w-7 shrink-0 place-items-center rounded-full text-white/65 transition-colors hover:bg-white/[0.07] hover:text-white focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-white"
                >
                  <i className="fa-solid fa-xmark text-xs" aria-hidden="true" />
                </button>
              </div>
            </div>
          ) : null}

          <label htmlFor="reelai-composer" className="sr-only">Ask ReelAI</label>
          <textarea
            ref={textareaRef}
            id="reelai-composer"
            rows={1}
            value={prompt}
            onChange={(event) => {
              setPrompt(event.target.value);
              setError(null);
            }}
            onKeyDown={onPromptKeyDown}
            placeholder={file ? "Add instructions (optional)" : "Ask ReelAI"}
            aria-describedby={compactStatus ? "composer-status" : undefined}
            className="block min-h-[72px] max-h-[92px] w-full resize-none overflow-y-hidden bg-transparent px-1 py-1 text-[17px] leading-7 text-white outline-none placeholder:text-white/45"
          />

          <div className="mt-2 flex items-center gap-2">
            <button
              type="button"
              aria-label="Attach a PDF, DOCX, or TXT file"
              onClick={() => fileInputRef.current?.click()}
              className="grid h-9 w-9 shrink-0 place-items-center rounded-full text-white/80 transition-colors hover:bg-white/[0.07] hover:text-white focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-white"
            >
              <i className="fa-solid fa-plus" aria-hidden="true" />
            </button>

            <CustomSelect
              name="knowledge-level"
              label="Knowledge level"
              value={knowledgeLevel}
              options={KNOWLEDGE_LEVEL_OPTIONS}
              onChange={setKnowledgeLevel}
              className="w-fit min-w-0"
              buttonClassName="h-9 max-w-[10.5rem] rounded-full bg-transparent px-3 text-xs font-medium text-white/70 hover:bg-white/[0.07] hover:text-white sm:text-sm"
              menuClassName="w-[200%] max-w-[calc(100vw-5rem)]"
              showSelectedCheck
            />

            <span className="min-w-0 flex-1" />
            <button
              type="submit"
              disabled={disabled}
              aria-label={loading ? "Starting search" : "Send"}
              className="grid h-9 w-9 shrink-0 place-items-center rounded-full bg-white text-black transition-opacity duration-300 motion-reduce:transition-none hover:bg-white/90 disabled:cursor-not-allowed disabled:bg-white/20 disabled:text-white/35 disabled:opacity-70 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white"
            >
              <i className={`fa-solid ${loading ? "fa-spinner fa-spin" : "fa-arrow-up"}`} aria-hidden="true" />
            </button>
          </div>
        </div>

        {compactStatus && !showBottomSignInStatus ? (
          <div className="px-2 pt-2">
            <p
              id="composer-status"
              role={error ? "alert" : "status"}
              className={`text-xs ${error ? "text-red-300" : "text-white/50"}`}
            >
              {compactStatus}
              {billingStatus
                && verifiedAccount?.isVerified
                && billingStatus.remaining_searches <= Math.min(3, billingStatus.daily_limit)
                ? <> <span>Uses 1 search</span></>
                : null}
            </p>
          </div>
        ) : null}

        <div
          data-home-suggestions="true"
          data-home-suggestion-offset={homeIdeaOffset}
          className="mt-1 space-y-0"
        >
          {visibleHomeIdeas.map((idea) => (
            <button
              key={idea.title}
              type="button"
              onClick={() => {
                setPrompt(idea.title);
                setError(null);
                window.requestAnimationFrame(() => textareaRef.current?.focus());
              }}
              className="flex min-h-11 w-full items-center gap-3.5 rounded-xl px-3 py-2 text-left text-[13px] text-white/60 transition-colors hover:bg-white/[0.07] hover:text-white/85 motion-reduce:transition-none sm:text-[14px]"
            >
              <HomeIdeaIcon name={idea.icon} />
              <span>{idea.title}</span>
            </button>
          ))}
        </div>
      </div>
      {showBottomSignInStatus ? (
        <p
          id="composer-status"
          role="status"
          className="absolute bottom-[calc(max(env(safe-area-inset-bottom),0px)+18px)] left-1/2 w-[calc(100%-2rem)] -translate-x-1/2 text-center text-xs text-white/45"
        >
          Sign in with a verified account to search.
        </p>
      ) : null}
      {billingGate ? (
        <BillingGateDialog
          reason={billingGate.reason}
          account={verifiedAccount}
          requiredSearches={billingGate.requiredSearches}
          onClose={() => setBillingGate(null)}
        />
      ) : null}
    </form>
  );
}
