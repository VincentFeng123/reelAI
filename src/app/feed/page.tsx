"use client";

import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

import { LoadingFlappyMiniGame } from "@/components/LoadingFlappyMiniGame";
import { ReelCard } from "@/components/ReelCard";
import { askStudyChat, fetchFeed, generateReels, sendFeedback, uploadMaterial } from "@/lib/api";
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
const MATERIAL_SEEDS_STORAGE_KEY = "studyreels-material-seeds";
const FEED_PROGRESS_STORAGE_KEY = "studyreels-feed-progress";
const GENERATION_MODE_STORAGE_KEY = "studyreels-generation-mode";
const HISTORY_STORAGE_KEY = "studyreels-material-history";
const MAX_SAVED_FEED_PROGRESS = 240;
type FeedbackAction = "helpful" | "confusing" | "save";
type GenerationMode = "slow" | "fast";

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

type FeedProgressEntry = {
  index: number;
  reelId?: string;
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
    return parsed as Record<string, MaterialSeed>;
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

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function FeedPageInner() {
  const params = useSearchParams();
  const router = useRouter();
  const materialId = params.get("material_id") || "";
  const generationModeParam = params.get("generation_mode");

  const [reels, setReels] = useState<Reel[]>([]);
  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [generatingMore, setGeneratingMore] = useState(false);
  const [bootstrappingFirstReels, setBootstrappingFirstReels] = useState(false);
  const [canRequestMore, setCanRequestMore] = useState(true);
  const [activeIndex, setActiveIndex] = useState(0);
  const [feedbackByReel, setFeedbackByReel] = useState<Record<string, ReelFeedbackState>>({});
  const [pendingAction, setPendingAction] = useState<FeedbackAction | null>(null);
  const [mobileDetailsOpen, setMobileDetailsOpen] = useState(false);
  const [mutedPreference, setMutedPreference] = useState(true);
  const [chatByReel, setChatByReel] = useState<Record<string, ChatMessage[]>>({});
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);
  const [rightPanelWidthPx, setRightPanelWidthPx] = useState(360);
  const [rightTopRatio, setRightTopRatio] = useState(0.62);
  const [generationMode, setGenerationMode] = useState<GenerationMode>("slow");

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

  const reelClipKey = useCallback((reel: Reel): string => {
    const match = reel.video_url.match(/\/embed\/([^?&/]+)/);
    const videoId = match?.[1] || reel.video_url;
    const start = Math.floor(Number(reel.t_start) || 0);
    const end = Math.floor(Number(reel.t_end) || 0);
    return `${videoId}:${start}:${end}`;
  }, []);

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
    window.localStorage.setItem(GENERATION_MODE_STORAGE_KEY, generationMode);
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

  useEffect(() => {
    if (!materialId) {
      return;
    }
    if (generationModeParam === generationMode) {
      return;
    }
    const nextParams = new URLSearchParams(params.toString());
    nextParams.set("material_id", materialId);
    nextParams.set("generation_mode", generationMode);
    router.replace(`/feed?${nextParams.toString()}`, { scroll: false });
  }, [generationMode, generationModeParam, materialId, params, router]);

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
        router.replace(`/feed?material_id=${rebuiltId}`);
        return true;
      } catch (e) {
        setError(e instanceof Error ? e.message : "Could not rebuild material.");
        return false;
      } finally {
        isRecoveringMissingMaterialRef.current = false;
      }
    },
    [router],
  );

  const loadPage = useCallback(
    async (targetPage: number, options?: { autofill?: boolean }) => {
      if (!materialId || isFetchingRef.current) {
        return;
      }
      isFetchingRef.current = true;
      setError(null);

      try {
        const data = await fetchFeed({
          materialId,
          page: targetPage,
          limit: PAGE_SIZE,
          autofill: options?.autofill ?? false,
          prefetch: 7,
          generationMode,
        });

        setTotal(data.total);
        setPage(targetPage);

        if (targetPage === 1) {
          const deduped = dedupeByIdentity(data.reels);
          setReels(deduped);
          setTotal(Math.max(data.total, deduped.length));
        } else {
          setReels((prev) => dedupeByIdentity(data.reels, prev));
        }
      } catch (e) {
        if (targetPage === 1) {
          const message = e instanceof Error ? e.message : "Feed failed to load";
          if (/material_id not found/i.test(message) && !recoveryAttemptedIdsRef.current.has(materialId)) {
            recoveryAttemptedIdsRef.current.add(materialId);
            const recovered = await recoverMissingMaterial(materialId);
            if (recovered) {
              return;
            }
          }
          setError(message);
        }
      } finally {
        setLoading(false);
        isFetchingRef.current = false;
      }
    },
    [dedupeByIdentity, generationMode, materialId, recoverMissingMaterial],
  );

  const requestMore = useCallback(async (options?: { surfaceError?: boolean }): Promise<Reel[]> => {
    if (!materialId || isGeneratingRef.current || !canRequestMore) {
      return [];
    }
    const batchSize = isFastGeneration ? 2 : 7;
    isGeneratingRef.current = true;
    setGeneratingMore(true);
    if (options?.surfaceError) {
      setError(null);
    }
    try {
      const generated = await generateReels({
        materialId,
        numReels: batchSize,
        generationMode,
      });
      if (generated.reels.length === 0) {
        emptyGenerateStreakRef.current += 1;
        if (emptyGenerateStreakRef.current >= 4) {
          setCanRequestMore(false);
        }
        if (options?.surfaceError) {
          setError("No reels were generated yet. Try a broader topic or add more source text.");
        }
        return [];
      }
      emptyGenerateStreakRef.current = 0;
      return generated.reels;
    } catch (e) {
      console.warn("Background reel generation failed:", e);
      emptyGenerateStreakRef.current += 1;
      if (emptyGenerateStreakRef.current >= 3) {
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
  }, [canRequestMore, generationMode, isFastGeneration, materialId]);

  useEffect(() => {
    if (!materialId) {
      pendingResumeRef.current = null;
      resumeAppliedRef.current = false;
      resumeLoadingRef.current = false;
      setLoading(false);
      return;
    }
    if (typeof window !== "undefined") {
      const allProgress = parseFeedProgress(window.localStorage.getItem(FEED_PROGRESS_STORAGE_KEY));
      pendingResumeRef.current = allProgress[materialId] ?? null;
    } else {
      pendingResumeRef.current = null;
    }
    resumeAppliedRef.current = false;
    resumeLoadingRef.current = false;
    recoveryAttemptedIdsRef.current.clear();
    setLoading(true);
    setCanRequestMore(true);
    setActiveIndex(0);
    setFeedbackByReel({});
    setMobileDetailsOpen(false);
    setBootstrappingFirstReels(false);
    emptyGenerateStreakRef.current = 0;
    bootstrapAttemptedRef.current = false;
    loadPage(1, { autofill: false });
  }, [materialId]);

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
    if (
      !materialId ||
      generationMode !== "fast" ||
      !canRequestMore ||
      isFastTopUpRef.current ||
      isGeneratingRef.current
    ) {
      return;
    }
    isFastTopUpRef.current = true;
    try {
      const generated = await generateReels({
        materialId,
        numReels: 5,
        generationMode,
      });
      if (generated.reels.length > 0) {
        emptyGenerateStreakRef.current = 0;
        appendGeneratedReels(generated.reels);
      }
    } catch (e) {
      console.warn("Fast mode background top-up failed:", e);
    } finally {
      isFastTopUpRef.current = false;
    }
  }, [appendGeneratedReels, canRequestMore, generationMode, materialId]);

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
      loadPage(page + 1, { autofill: false });
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
      void loadPage(page + 1, { autofill: false }).finally(() => {
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
      wheelReadyToRearmRef.current = false;
    };
  }, []);

  useEffect(() => {
    setMobileDetailsOpen(false);
    setChatInput("");
    setChatError(null);
  }, [activeIndex]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const saved = window.localStorage.getItem("studyreels-muted");
    if (saved === "0") {
      setMutedPreference(false);
    } else if (saved === "1") {
      setMutedPreference(true);
    }
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    window.localStorage.setItem("studyreels-muted", mutedPreference ? "1" : "0");
  }, [mutedPreference]);

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
  const activeReelPosition = activeReel ? activeIndex + 1 : 0;
  const loadedReelCount = reels.length;
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

  const actionButtonClass = (active: boolean) =>
    `grid h-10 w-10 place-items-center rounded-2xl border text-sm transition ${
      active ? "border-white bg-white text-black" : "border-white/30 bg-transparent text-white hover:bg-white/10"
    }`;

  const renderFeedbackIconButton = (
    action: FeedbackAction,
    label: string,
    iconClass: string,
    active: boolean,
  ) => (
    <div className="group relative">
      <button
        onClick={() => submitActiveFeedback(action)}
        className={actionButtonClass(active)}
        disabled={pendingAction !== null}
        aria-label={label}
      >
        {pendingAction === action ? (
          <i className="fa-solid fa-spinner fa-spin" aria-hidden="true" />
        ) : (
          <i className={`fa-solid ${iconClass}`} aria-hidden="true" />
        )}
      </button>
      <span className="pointer-events-none absolute bottom-full left-1/2 mb-2 -translate-x-1/2 translate-y-1 whitespace-nowrap rounded-md border border-white/20 bg-black/85 px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.08em] text-white opacity-0 transition-all duration-150 delay-0 group-hover:translate-y-0 group-hover:opacity-100 group-hover:delay-500">
        {label}
      </span>
    </div>
  );
  const renderGenerationModeToggle = (className?: string) => (
    <div
      role="group"
      aria-label="Generation mode"
      className={[
        "relative grid h-10 w-[128px] grid-cols-2 items-center rounded-2xl border border-white/25 bg-black/60 p-1 text-[10px] font-semibold uppercase tracking-[0.06em] text-white backdrop-blur-md",
        className,
      ]
        .filter(Boolean)
        .join(" ")}
    >
      <span
        aria-hidden="true"
        className={`pointer-events-none absolute bottom-1 left-1 top-1 w-[calc(50%-4px)] rounded-xl bg-white transition-transform duration-300 ease-out ${
          generationMode === "fast" ? "translate-x-full" : "translate-x-0"
        }`}
      />
      <button
        type="button"
        onClick={() => setGenerationMode("slow")}
        className={`relative z-10 rounded-xl px-2 py-1 transition-colors ${generationMode === "slow" ? "text-black" : "text-white/82"}`}
        aria-pressed={generationMode === "slow"}
      >
        Slow
      </button>
      <button
        type="button"
        onClick={() => setGenerationMode("fast")}
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
      className={`grid h-10 w-10 place-items-center rounded-2xl border text-sm transition backdrop-blur-md ${
        active ? "border-white bg-white text-black" : "border-white/28 bg-black/60 text-white/92 hover:bg-black/70"
      }`}
      disabled={pendingAction !== null}
      aria-label={label}
      title={label}
    >
      {pendingAction === action ? (
        <i className="fa-solid fa-spinner fa-spin" aria-hidden="true" />
      ) : (
        <i className={`fa-solid ${iconClass}`} aria-hidden="true" />
      )}
    </button>
  );
  const rightTopPercent = Math.round(rightTopRatio * 1000) / 10;

  if (!materialId) {
    return (
      <main className="fixed inset-0 px-6 md:inset-4">
        <div className="flex h-full items-center justify-center">
          <div className="rounded-3xl border border-white/25 bg-black/60 p-6 text-center text-white backdrop-blur-sm">
            <p className="text-sm">Missing material_id.</p>
            <button
              className="mt-4 rounded-2xl border border-white/25 bg-white px-4 py-2 text-xs font-semibold text-black"
              onClick={() => router.push("/")}
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
        onClick={() => router.push("/")}
        aria-label="Back to main page"
        className="absolute left-3 top-3 z-[9999] grid h-9 w-9 place-items-center rounded-xl border border-white/20 bg-black/50 text-white shadow-[0_8px_24px_rgba(0,0,0,0.35)] backdrop-blur-md transition hover:bg-white/12"
      >
        <i className="fa-solid fa-arrow-left text-xs" aria-hidden="true" />
      </button>
      <div className="absolute right-3 top-3 z-[9999] lg:hidden">
        {renderGenerationModeToggle("shadow-[0_8px_24px_rgba(0,0,0,0.35)]")}
      </div>
      {error ? (
        <div className="absolute left-0 right-0 top-3 z-30 mx-auto w-fit rounded-full border border-white/25 bg-black/75 px-4 py-2 text-xs text-white">
          {error}
        </div>
      ) : null}

      <div ref={desktopShellRef} className="h-full min-h-[100dvh] md:min-h-0 lg:flex">
        <section className="relative h-[100dvh] min-h-[100dvh] md:h-full md:min-h-0 lg:min-w-0 lg:flex-1">
          {activeReel && !mobileDetailsOpen ? (
            <div className="absolute right-3 top-1/2 z-30 flex -translate-y-1/2 flex-col gap-2 rounded-2xl border border-white/20 bg-black/55 p-2 shadow-[0_10px_28px_rgba(0,0,0,0.38)] backdrop-blur-md lg:hidden">
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
            className="reel-scroll h-[100dvh] min-h-[100dvh] overflow-hidden rounded-none overscroll-none touch-none md:h-full md:min-h-0 lg:h-full lg:min-h-0 lg:rounded-3xl"
          >
            <div
              className="flex h-full flex-col transition-transform duration-300 ease-out"
              style={{ transform: `translate3d(0, -${activeIndex * 100}%, 0)` }}
            >
              {reels.map((reel, index) => (
                <div key={reel.reel_id} className="h-[100dvh] shrink-0 grow-0 basis-full md:h-full lg:h-full">
                  <ReelCard
                    reel={reel}
                    isActive={index === activeIndex}
                    mutedPreference={mutedPreference}
                    onMutedPreferenceChange={setMutedPreference}
                    onOpenContent={index === activeIndex ? () => setMobileDetailsOpen(true) : undefined}
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
                onClick={() => setMobileDetailsOpen(false)}
              />
              <div className="absolute inset-x-0 bottom-0 max-h-[74svh] overflow-y-auto rounded-t-3xl border border-white/20 bg-black/92 px-4 pt-4 pb-0 text-white backdrop-blur">
                <div className="flex items-center justify-between gap-3">
                  <div className="inline-flex w-fit rounded-full border border-white/25 bg-black/70 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.12em] text-white">
                    Reel {activeReelPosition}/{loadedReelCount} loaded
                  </div>
                  <button
                    onClick={() => setMobileDetailsOpen(false)}
                    className="rounded-xl border border-white/25 bg-black/60 px-3 py-1.5 text-xs font-semibold text-white"
                  >
                    Close
                  </button>
                </div>

                <h2 className="mt-3 text-xl font-bold leading-tight">{activeReel.concept_title}</h2>

                <div className="mt-3 min-w-0 rounded-2xl border border-white/20 bg-black/55 p-3 text-sm text-white/90">
                  <p className="mb-1 text-[10px] font-semibold uppercase tracking-[0.1em] text-white/60">Video Description</p>
                  <p className="whitespace-pre-line break-words [overflow-wrap:anywhere]">{activeVideoDescription}</p>
                </div>

                <div className="mt-3 min-w-0 rounded-2xl border border-white/20 bg-black/55 p-3 text-sm text-white/90">
                  <p className="mb-1 text-[10px] font-semibold uppercase tracking-[0.1em] text-white/60">AI Summary</p>
                  <p className="break-words [overflow-wrap:anywhere]">{activeAiSummary}</p>
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
          <aside className="min-h-0 min-w-0 overflow-y-auto rounded-3xl border border-white/20 bg-black/72 px-5 pt-5 pb-0 text-white">
            {!activeReel ? (
              <div className="flex h-full items-center justify-center text-sm text-white/80">Loading reel details...</div>
            ) : (
              <div className="flex h-full flex-col">
                <h2 className="text-2xl font-bold leading-tight">{activeReel.concept_title}</h2>

                <div className="mt-3 min-w-0 rounded-2xl border border-white/20 bg-black/55 p-3 text-sm text-white/90">
                  <p className="mb-1 text-[10px] font-semibold uppercase tracking-[0.1em] text-white/60">Video Description</p>
                  <p className="whitespace-pre-line break-words [overflow-wrap:anywhere]">{activeVideoDescription}</p>
                </div>

                <div className="mt-3 min-w-0 rounded-2xl border border-white/20 bg-black/55 p-3 text-sm text-white/90">
                  <p className="mb-1 text-[10px] font-semibold uppercase tracking-[0.1em] text-white/60">AI Summary</p>
                  <p className="break-words [overflow-wrap:anywhere]">{activeAiSummary}</p>
                </div>

                <div className="mt-auto pt-4 mb-[17px]">
                  <div className="flex flex-wrap items-center gap-2">
                    {renderFeedbackIconButton("helpful", "Helpful", "fa-thumbs-up", Boolean(activeFeedback.helpful))}
                    {renderFeedbackIconButton("confusing", "Confusing", "fa-circle-question", Boolean(activeFeedback.confusing))}
                    {renderFeedbackIconButton("save", "Save", "fa-bookmark", Boolean(activeFeedback.saved))}
                    {renderGenerationModeToggle()}
                  </div>
                </div>
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
                        msg.role === "user" ? "ml-8 bg-black/70 text-white" : "mr-8 bg-white/10 text-white/92"
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
