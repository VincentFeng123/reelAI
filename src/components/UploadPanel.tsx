"use client";

import { type DragEvent, type FormEvent, type RefObject, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";

import { ingestSearch, ingestUrl, uploadMaterial } from "@/lib/api";
import { safeStorageSetItem } from "@/lib/browserStorage";
import { buildSearchFeedQuery } from "@/lib/feedQuery";
import { type GenerationMode, type SearchInputMode, readStudyReelsSettings, subscribeToStudyReelsSettings } from "@/lib/settings";
import type { Reel } from "@/lib/types";

const MATERIAL_SEEDS_STORAGE_KEY = "studyreels-material-seeds";
const MATERIAL_GROUPS_STORAGE_KEY = "studyreels-material-groups";
const MAX_MATERIAL_SEEDS = 120;
const MAX_MATERIAL_GROUPS = 80;
const MAX_SEED_TEXT_CHARS = 16000;
const INGEST_SENTINEL_MATERIAL_ID = "ingest-scratch";
// Must stay in sync with FEED_SESSION_STORAGE_KEY in src/app/feed/page.tsx — both
// read/write the same localStorage key. Priming the session snapshot here lets the
// feed page hydrate with ingested reels on mount instead of calling the legacy
// /api/reels/generate-stream path on an ingest-search/ingest-scratch sentinel
// material (which cannot be served by the old pipeline and returns "not found").
const FEED_SESSION_STORAGE_KEY = "studyreels-feed-sessions";
type InputMode = SearchInputMode;

const INPUT_MODE_OPTIONS: Array<{ value: InputMode; label: string }> = [
  { value: "topic", label: "Topic" },
  { value: "source", label: "Text" },
  { value: "file", label: "File Upload" },
  { value: "url", label: "Reel URL" },
];

const INGEST_URL_HOST_ALLOWLIST = new Set([
  "youtube.com",
  "www.youtube.com",
  "m.youtube.com",
  "youtu.be",
  "music.youtube.com",
  "instagram.com",
  "www.instagram.com",
  "tiktok.com",
  "www.tiktok.com",
  "vm.tiktok.com",
  "m.tiktok.com",
]);

/**
 * Lightweight client-side sanity check so we don't POST every keystroke to the backend.
 * The backend still does the authoritative host check via the yt-dlp adapter.
 */
function isLikelyIngestUrl(raw: string): boolean {
  const trimmed = raw.trim();
  if (!trimmed) {
    return false;
  }
  const candidate = /^https?:\/\//i.test(trimmed) ? trimmed : `https://${trimmed}`;
  try {
    const parsed = new URL(candidate);
    return INGEST_URL_HOST_ALLOWLIST.has(parsed.hostname.toLowerCase());
  } catch {
    return false;
  }
}

type MaterialSeed = {
  topic?: string;
  text?: string;
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
      if (!title) {
        continue;
      }
      normalized[id] = {
        topic: String(seed.topic || "").trim() || undefined,
        text: String(seed.text || "").trim() || undefined,
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

export function UploadPanel({ onMaterialCreated, onScrollOffsetChange, onScrollGesture, onScrollabilityChange, heroTitleRef }: UploadPanelProps) {
  const router = useRouter();
  const touchStartYRef = useRef<number | null>(null);
  const formRef = useRef<HTMLFormElement | null>(null);
  const [topics, setTopics] = useState<string[]>([""]);
  const [text, setText] = useState("");
  const [file, setFile] = useState<File | undefined>();
  const [reelUrl, setReelUrl] = useState("");
  const [inputMode, setInputMode] = useState<InputMode>("source");
  const [isDraggingFile, setIsDraggingFile] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [generationMode, setGenerationMode] = useState<GenerationMode>("fast");
  const selectedFileName = file?.name ?? "";

  useEffect(() => {
    const saved = readStudyReelsSettings();
    setGenerationMode(saved.generationMode);
    setInputMode(saved.defaultInputMode);
    return subscribeToStudyReelsSettings((next) => {
      setGenerationMode(next.generationMode);
      setInputMode(next.defaultInputMode);
    });
  }, []);

  const disabled = useMemo(() => {
    if (loading) {
      return true;
    }
    if (inputMode === "topic") {
      return !topics.some((t) => t.trim());
    }
    if (inputMode === "source") {
      return !text.trim();
    }
    if (inputMode === "url") {
      return !isLikelyIngestUrl(reelUrl);
    }
    return !file;
  }, [file, inputMode, loading, reelUrl, text, topics]);

  const onSubmit = useCallback(async (event: FormEvent) => {
    event.preventDefault();
    setError(null);
    setLoading(true);

    try {
      const activeSettings = readStudyReelsSettings();
      const generationModeForSearch = activeSettings.generationMode;
      const useMultiPlatformSearch = activeSettings.multiPlatformSearch;

      // Shared helper: navigate to the feed scoped to the ingestSearch-returned
      // material and persist a local seed so the history panel shows the entry.
      const launchMultiPlatformFeed = async (params: {
        query: string;
        title: string;
        topic?: string;
        firstReelId?: string;
        materialId: string;
      }) => {
        if (typeof window !== "undefined") {
          const seeds = parseMaterialSeeds(window.localStorage.getItem(MATERIAL_SEEDS_STORAGE_KEY));
          seeds[params.materialId] = {
            topic: params.topic,
            text: undefined,
            title: params.title,
            updatedAt: Date.now(),
          };
          const ordered = Object.entries(seeds)
            .sort((a, b) => (b[1].updatedAt || 0) - (a[1].updatedAt || 0))
            .slice(0, MAX_MATERIAL_SEEDS);
          safeStorageSetItem(window.localStorage, MATERIAL_SEEDS_STORAGE_KEY, JSON.stringify(Object.fromEntries(ordered)));
        }
        if (onMaterialCreated) {
          const feedQuery = buildSearchFeedQuery({
            materialId: params.materialId,
            generationMode: generationModeForSearch,
            returnTab: "search",
            settings: activeSettings,
          });
          await onMaterialCreated({
            materialId: params.materialId,
            title: params.title,
            topic: params.topic,
            generationMode: generationModeForSearch,
            feedQuery,
          });
        }
        const searchFeedQuery = buildSearchFeedQuery({
          materialId: params.materialId,
          generationMode: generationModeForSearch,
          returnTab: "search",
          settings: activeSettings,
        });
        const suffix = params.firstReelId
          ? `&active_reel_id=${encodeURIComponent(params.firstReelId)}`
          : "";
        router.push(`/feed?${searchFeedQuery}${suffix}`);
      };

      // Topic / text / file with multi-platform search ON: route through
      // /api/ingest/search so we get real scraping across YouTube, Instagram, and
      // TikTok. Topic is the simplest (topic text IS the query). Text and file need
      // a two-step flow: upload via /api/material first so the backend's LLM concept
      // extractor can turn prose/files into useful search terms, then call ingestSearch
      // with the top concept titles joined as the query.
      // Helper: extract successful reels from an IngestSearchResult into a flat Reel[].
      const collectSuccessfulReels = (items: Array<{ status: string; reel?: Reel | null | undefined }>): Reel[] => {
        const out: Reel[] = [];
        for (const item of items) {
          if (item.status === "ok" && item.reel) {
            out.push(item.reel);
          }
        }
        return out;
      };

      if (useMultiPlatformSearch && inputMode === "topic") {
        const topicList = topics.map((t) => t.trim()).filter(Boolean);
        const combinedQuery = topicList.join(" ").trim();
        if (!combinedQuery) {
          throw new Error("Enter a topic before searching.");
        }
        const searchResult = await ingestSearch({
          query: combinedQuery,
          platforms: ["yt", "ig", "tt"],
          maxPerPlatform: 4,
          targetClipDurationSec: activeSettings.targetClipDurationSec,
          targetClipDurationMinSec: activeSettings.targetClipDurationMinSec,
          targetClipDurationMaxSec: activeSettings.targetClipDurationMaxSec,
        });
        const successfulReels = collectSuccessfulReels(searchResult.items);
        if (successfulReels.length === 0) {
          throw new Error(`No reels found for “${combinedQuery}”. Try a different topic.`);
        }
        const materialId = searchResult.material_id || "ingest-search:unknown";
        const firstReelId = successfulReels[0]?.reel_id;
        primeFeedSessionSnapshot(materialId, successfulReels, firstReelId, activeSettings);
        await launchMultiPlatformFeed({
          query: combinedQuery,
          title: combinedQuery.slice(0, 58),
          topic: combinedQuery,
          firstReelId,
          materialId,
        });
        setTopics([""]);
        return;
      }

      if (useMultiPlatformSearch && inputMode === "source") {
        const trimmedText = text.trim();
        if (!trimmedText) {
          throw new Error("Enter some text before searching.");
        }
        const material = await uploadMaterial({ text: trimmedText });
        const conceptTitles = (material.extracted_concepts ?? [])
          .slice(0, 3)
          .map((c) => (c?.title ?? "").trim())
          .filter(Boolean);
        const query = conceptTitles.length > 0
          ? conceptTitles.join(" ")
          : trimmedText.replace(/\s+/g, " ").slice(0, 180);
        const searchResult = await ingestSearch({
          query,
          platforms: ["yt", "ig", "tt"],
          maxPerPlatform: 4,
          targetClipDurationSec: activeSettings.targetClipDurationSec,
          targetClipDurationMinSec: activeSettings.targetClipDurationMinSec,
          targetClipDurationMaxSec: activeSettings.targetClipDurationMaxSec,
        });
        const successfulReels = collectSuccessfulReels(searchResult.items);
        if (successfulReels.length === 0) {
          throw new Error("No reels found for your text. Try a different source.");
        }
        const materialId = searchResult.material_id || "ingest-search:unknown";
        const firstReelId = successfulReels[0]?.reel_id;
        const title = trimmedText.replace(/\s+/g, " ").slice(0, 58);
        primeFeedSessionSnapshot(materialId, successfulReels, firstReelId, activeSettings);
        await launchMultiPlatformFeed({
          query,
          title: title || "Study Session",
          firstReelId,
          materialId,
        });
        setText("");
        return;
      }

      if (useMultiPlatformSearch && inputMode === "file") {
        if (!file) {
          throw new Error("Pick a file before searching.");
        }
        const material = await uploadMaterial({ file });
        const conceptTitles = (material.extracted_concepts ?? [])
          .slice(0, 3)
          .map((c) => (c?.title ?? "").trim())
          .filter(Boolean);
        const fileName = file.name;
        const query = conceptTitles.length > 0
          ? conceptTitles.join(" ")
          : fileName.replace(/\.[^.]+$/, "").replace(/[_\-]+/g, " ").slice(0, 180);
        const searchResult = await ingestSearch({
          query,
          platforms: ["yt", "ig", "tt"],
          maxPerPlatform: 4,
          targetClipDurationSec: activeSettings.targetClipDurationSec,
          targetClipDurationMinSec: activeSettings.targetClipDurationMinSec,
          targetClipDurationMaxSec: activeSettings.targetClipDurationMaxSec,
        });
        const successfulReels = collectSuccessfulReels(searchResult.items);
        if (successfulReels.length === 0) {
          throw new Error("No reels found for your file. Try a different document.");
        }
        const materialId = searchResult.material_id || "ingest-search:unknown";
        const firstReelId = successfulReels[0]?.reel_id;
        primeFeedSessionSnapshot(materialId, successfulReels, firstReelId, activeSettings);
        await launchMultiPlatformFeed({
          query,
          title: fileName.slice(0, 58),
          firstReelId,
          materialId,
        });
        setFile(undefined);
        return;
      }

      // URL ingest path diverges completely from the material-upload path: we call
      // /api/ingest/url instead of /api/material, then land on the feed scoped to
      // the `ingest-scratch` sentinel material so prior ingests form a scrollable feed.
      if (inputMode === "url") {
        const trimmed = reelUrl.trim();
        const normalized = /^https?:\/\//i.test(trimmed) ? trimmed : `https://${trimmed}`;
        const result = await ingestUrl({
          sourceUrl: normalized,
          targetClipDurationSec: activeSettings.targetClipDurationSec,
          targetClipDurationMinSec: activeSettings.targetClipDurationMinSec,
          targetClipDurationMaxSec: activeSettings.targetClipDurationMaxSec,
        });
        const ingestedReel = result.reel;
        const ingestedMetadata = result.metadata;
        const ingestMaterialId = ingestedReel.material_id || INGEST_SENTINEL_MATERIAL_ID;
        const ingestTitle =
          (ingestedReel.video_title?.trim() ||
            ingestedMetadata.title?.trim() ||
            (ingestedMetadata.author_handle ? `@${ingestedMetadata.author_handle}` : "") ||
            "Ingested reel").slice(0, 58);

        // Prime the feed snapshot with the single ingested reel so the feed page's
        // bootstrap hydrates with it and skips the legacy generate path.
        primeFeedSessionSnapshot(ingestMaterialId, [ingestedReel], ingestedReel.reel_id, activeSettings);

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
        }

        const ingestFeedQuery = buildSearchFeedQuery({
          materialId: ingestMaterialId,
          generationMode: generationModeForSearch,
          returnTab: "search",
          settings: activeSettings,
        });
        router.push(`/feed?${ingestFeedQuery}&active_reel_id=${encodeURIComponent(ingestedReel.reel_id)}`);
        setReelUrl("");
        return;
      }

      const topicList = inputMode === "topic" ? topics.map((t) => t.trim()).filter(Boolean) : [];
      const topicValue = topicList.join(", ");
      const textValue = inputMode === "source" ? text.trim() : "";
      const fileValue = inputMode === "file" ? file : undefined;
      const title = buildMaterialTitle({
        topic: topicValue,
        text: textValue,
        fileName: fileValue?.name ?? "",
      });
      let materialIds: string[] = [];
      if (inputMode === "topic" && topicList.length > 1) {
        const uploads = await Promise.all(
          topicList.map(async (topic) =>
            uploadMaterial({
              subjectTag: topic,
            }),
          ),
        );
        materialIds = uploads.map((row) => row.material_id).filter(Boolean);
      } else {
        const material = await uploadMaterial({
          text: textValue || undefined,
          file: fileValue,
          subjectTag: topicValue || undefined,
        });
        materialIds = [material.material_id];
      }

      const primaryMaterialId = materialIds[0];
      if (!primaryMaterialId) {
        throw new Error("Material creation failed.");
      }

      if (typeof window !== "undefined") {
        const seeds = parseMaterialSeeds(window.localStorage.getItem(MATERIAL_SEEDS_STORAGE_KEY));
        const now = Date.now();
        if (inputMode === "topic" && topicList.length > 1) {
          materialIds.forEach((id, index) => {
            const topic = topicList[index]?.trim();
            if (!id || !topic) {
              return;
            }
            seeds[id] = {
              topic,
              text: undefined,
              title: topic,
              updatedAt: now - index,
            };
          });
        } else {
          seeds[primaryMaterialId] = {
            topic: topicValue || undefined,
            text: textValue ? textValue.slice(0, MAX_SEED_TEXT_CHARS) : undefined,
            title,
            updatedAt: now,
          };
        }
        const ordered = Object.entries(seeds)
          .sort((a, b) => (b[1].updatedAt || 0) - (a[1].updatedAt || 0))
          .slice(0, MAX_MATERIAL_SEEDS);
        safeStorageSetItem(window.localStorage, MATERIAL_SEEDS_STORAGE_KEY, JSON.stringify(Object.fromEntries(ordered)));

        const groups = parseMaterialGroups(window.localStorage.getItem(MATERIAL_GROUPS_STORAGE_KEY));
        if (inputMode === "topic" && topicList.length > 1) {
          groups[primaryMaterialId] = {
            materialIds,
            title,
            updatedAt: now,
          };
        } else {
          delete groups[primaryMaterialId];
        }
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
      }
      const nextQuery = buildSearchFeedQuery({
        materialId: primaryMaterialId,
        generationMode: generationModeForSearch,
        returnTab: "search",
        settings: activeSettings,
      });
      router.push(`/feed?${nextQuery}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something failed");
    } finally {
      setLoading(false);
    }
  }, [file, inputMode, onMaterialCreated, reelUrl, router, text, topics]);

  const onFileDrop = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    setIsDraggingFile(false);
    const dropped = event.dataTransfer.files?.[0];
    if (dropped) {
      setFile(dropped);
    }
  };

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
  }, [inputMode, reportScrollability, topics.length, text, selectedFileName, error]);

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
      className="mx-auto flex h-full w-full flex-col justify-center overflow-x-visible overflow-y-auto px-6 py-6 md:overflow-x-visible md:overflow-y-hidden md:px-10 md:py-8 lg:max-w-[1040px] lg:px-5"
    >
      <header className="relative mb-4 overflow-visible text-center">
        <img
          src="/logo.png"
          alt="StudyReels logo"
          className="relative z-20 mx-auto hidden h-4 w-[4.75rem] max-w-[26vw] translate-y-16 object-cover opacity-70 md:block"
        />
        <div className="mt-8 md:mt-20">
          <h1
            ref={heroTitleRef}
            className="relative z-[1] inline-block overflow-visible px-[0.12em] py-[0.12em] text-[clamp(2.9rem,10.4vw,7.7rem)] font-black leading-[0.96] tracking-tight"
          >
            <span
              aria-hidden="true"
              className="pointer-events-none absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 whitespace-nowrap px-[0.04em] py-[0.08em] text-white/30 blur-[12px] opacity-45"
            >
              Study Reels
            </span>
            <span className="relative whitespace-nowrap text-white/[0.56] [text-shadow:0_0_18px_rgba(255,255,255,0.08),0_0_44px_rgba(255,255,255,0.03)]">
              Study Reels
            </span>
          </h1>
          <p className="relative z-20 mt-5 text-sm text-white/68">Pick a mode, add your material, and start your short study feed.</p>
        </div>
      </header>

      <input
        id="material-file"
        className="sr-only"
        type="file"
        accept=".pdf,.docx,.txt"
        onChange={(e) => setFile(e.target.files?.[0])}
      />

      <div className="relative z-20 mt-8 max-w-[300px] md:mt-2 md:max-w-[430px]">
        <p className="mb-2 text-xs font-semibold uppercase tracking-[0.12em] text-white/70">Input Mode</p>
        <div
          role="tablist"
          aria-label="Select input mode"
          className="relative grid w-full grid-cols-4 rounded-2xl border border-white/15 bg-white/[0.08] p-1 backdrop-blur-[18px] backdrop-saturate-150"
        >
          <span
            aria-hidden="true"
            className="pointer-events-none absolute bottom-1 left-1 top-1 w-[calc((100%-8px)/4)] rounded-xl bg-white transition-transform duration-300 ease-out"
            style={{
              transform: `translateX(${INPUT_MODE_OPTIONS.findIndex((option) => option.value === inputMode) * 100}%)`,
            }}
          />
          {INPUT_MODE_OPTIONS.map((option) => (
            <button
              key={option.value}
              role="tab"
              type="button"
              aria-selected={inputMode === option.value}
              onClick={() => {
                setInputMode(option.value);
                setError(null);
              }}
              className={`relative z-10 rounded-xl px-1.5 py-1.5 text-[10px] font-semibold uppercase tracking-[0.05em] transition-colors md:px-2.5 md:py-2 md:text-[11px] ${
                inputMode === option.value ? "text-black" : "text-white/80 hover:text-white"
              }`}
            >
              {option.label}
            </button>
          ))}
        </div>
      </div>

      <div className="relative z-20 mt-6 h-[160px] min-h-[160px] md:mt-4 md:h-[175px] md:min-h-[175px]">
        {inputMode === "topic" ? (
          <>
            <label className="mb-2 block text-xs font-semibold uppercase tracking-[0.12em] text-white/70">Topics</label>
            <div className="h-full min-h-[160px] md:min-h-[175px] flex flex-col gap-2 overflow-y-auto">
              {topics.map((t, i) => (
                <div key={i} className="pr-1">
                  <div className="relative w-full rounded-2xl border border-white/15 bg-white/[0.08] backdrop-blur-[18px] backdrop-saturate-150 transition-colors duration-200 focus-within:bg-white/[0.12]">
                    <input
                      className="h-12 w-full rounded-2xl border-0 bg-transparent px-4 pr-11 text-sm text-white outline-none placeholder:text-white/40"
                      placeholder={i === 0 ? "e.g. linear regression" : "e.g. another topic"}
                      value={t}
                      onChange={(e) => {
                        const next = [...topics];
                        next[i] = e.target.value;
                        setTopics(next);
                      }}
                    />
                    {topics.length > 1 ? (
                      <button
                        type="button"
                        onClick={() => setTopics(topics.filter((_, j) => j !== i))}
                        className="absolute right-3 top-1/2 -translate-y-1/2 text-white/60 transition-colors duration-200 hover:text-white"
                        aria-label="Remove topic"
                      >
                        <i className="fa-solid fa-xmark text-xs" aria-hidden="true" />
                      </button>
                    ) : null}
                  </div>
                </div>
              ))}
              <button
                type="button"
                onClick={() => setTopics([...topics, ""])}
                className="mt-1 flex w-fit items-center gap-1.5 rounded-xl px-3 py-1.5 text-xs font-semibold text-white/60 transition hover:text-white/90"
              >
                <i className="fa-solid fa-plus text-[10px]" aria-hidden="true" />
                Add topic
              </button>
            </div>
          </>
        ) : null}

        {inputMode === "source" ? (
          <>
            <label className="mb-2 block text-xs font-semibold uppercase tracking-[0.12em] text-white/70">Text</label>
            <div className="h-full rounded-2xl border border-white/15 bg-white/[0.08] backdrop-blur-[18px] backdrop-saturate-150 transition-colors duration-200 focus-within:bg-white/[0.12]">
              <textarea
                className="h-full min-h-[160px] w-full resize-none overflow-y-auto rounded-2xl border-0 bg-transparent p-5 text-sm leading-relaxed text-white outline-none placeholder:text-white/40 md:min-h-[175px]"
                placeholder="Paste notes, textbook text, or any material here..."
                value={text}
                onChange={(e) => setText(e.target.value)}
              />
            </div>
          </>
        ) : null}

        {inputMode === "file" ? (
          <>
            <label className="mb-2 block text-xs font-semibold uppercase tracking-[0.12em] text-white/70">File Upload</label>
            <label
              htmlFor="material-file"
              onDragOver={(event) => {
                event.preventDefault();
                setIsDraggingFile(true);
              }}
              onDragLeave={() => setIsDraggingFile(false)}
              onDrop={onFileDrop}
              className={`flex h-full min-h-[160px] w-full cursor-pointer flex-col items-center justify-center rounded-2xl border border-dashed bg-white/[0.08] p-6 text-center text-white outline-none backdrop-blur-[18px] backdrop-saturate-150 transition-colors duration-200 md:min-h-[175px] ${
                isDraggingFile ? "border-white/30 bg-white/[0.12]" : "border-white/15"
              }`}
            >
              <span className="grid h-12 w-12 place-items-center rounded-full border border-white/15 bg-white/[0.08] text-white/85 backdrop-blur-[18px] backdrop-saturate-150">
                <i className="fa-solid fa-arrow-up-from-bracket text-base" aria-hidden="true" />
              </span>
              <p className={`mt-4 max-w-[90%] truncate text-sm font-semibold ${selectedFileName ? "text-white" : "text-white/85"}`}>
                {selectedFileName || "Drag and drop your file here"}
              </p>
              <p className="mt-1 text-xs text-white/58">{selectedFileName ? "Click to replace file" : "Or click to browse (PDF, DOCX, TXT)"}</p>
            </label>
          </>
        ) : null}

        {inputMode === "url" ? (
          <>
            <label className="mb-2 block text-xs font-semibold uppercase tracking-[0.12em] text-white/70">Reel URL</label>
            <div className="h-full flex flex-col gap-3">
              <div className="rounded-2xl border border-white/15 bg-white/[0.08] backdrop-blur-[18px] backdrop-saturate-150 transition-colors duration-200 focus-within:bg-white/[0.12]">
                <input
                  type="url"
                  inputMode="url"
                  autoComplete="off"
                  autoCapitalize="off"
                  spellCheck={false}
                  className="h-12 w-full rounded-2xl border-0 bg-transparent px-4 text-sm text-white outline-none placeholder:text-white/40"
                  placeholder="Paste an Instagram, TikTok, or YouTube URL"
                  value={reelUrl}
                  onChange={(e) => setReelUrl(e.target.value)}
                />
              </div>
              <p className="px-1 text-[11px] leading-snug text-white/55">
                Downloads, transcribes (Whisper fallback if needed), and extracts a clip with full metadata. Public reels only.
              </p>
            </div>
          </>
        ) : null}
      </div>

      <div className="relative z-20 mt-6 shrink-0 flex flex-col gap-2 md:mt-6">
        <p className="min-h-5 text-sm text-white/80">{error ?? ""}</p>
        <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
          <div className="w-full md:max-w-[220px]">
            <label className="mb-2 block text-xs font-semibold uppercase tracking-[0.12em] text-white/70">Generation Speed</label>
            <div className="relative grid h-12 grid-cols-2 items-center rounded-2xl border border-white/15 bg-white/[0.08] p-1 backdrop-blur-[18px] backdrop-saturate-150">
              <span
                aria-hidden="true"
                className={`pointer-events-none absolute bottom-1 left-1 top-1 w-[calc(50%-4px)] rounded-xl bg-white transition-transform duration-300 ease-out ${
                  generationMode === "fast" ? "translate-x-full" : "translate-x-0"
                }`}
              />
              <button
                type="button"
                onClick={() => setGenerationMode("slow")}
                className={`relative z-10 rounded-xl px-2 py-2 text-[11px] font-semibold uppercase tracking-[0.04em] transition-colors duration-200 ${
                  generationMode === "slow" ? "text-black" : "text-white/75 hover:text-white"
                }`}
                aria-pressed={generationMode === "slow"}
              >
                Slow
              </button>
              <button
                type="button"
                onClick={() => setGenerationMode("fast")}
                className={`relative z-10 rounded-xl px-2 py-2 text-[11px] font-semibold uppercase tracking-[0.04em] transition-colors duration-200 ${
                  generationMode === "fast" ? "text-black" : "text-white/75 hover:text-white"
                }`}
                aria-pressed={generationMode === "fast"}
              >
                Fast
              </button>
            </div>
          </div>

          <button
            type="submit"
            disabled={disabled}
            className="inline-flex w-full items-center justify-center rounded-2xl border border-white/30 bg-white px-7 py-3 text-sm font-bold text-black transition-colors hover:bg-white/92 disabled:cursor-not-allowed disabled:opacity-60 md:w-[12rem]"
          >
            <span className="inline-flex w-[9.5rem] items-center justify-center text-center">
              {loading ? "Starting..." : "Start Learning"}
            </span>
          </button>
        </div>
      </div>
    </form>
  );
}
