"use client";

import { type MouseEvent as ReactMouseEvent, type UIEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";

import { CommunityReelsPanel } from "@/components/CommunityReelsPanel";
import { UploadPanel } from "@/components/UploadPanel";
import { VolumetricLightBackground } from "@/components/VolumetricLightBackground";

const HISTORY_STORAGE_KEY = "studyreels-material-history";
const LEGACY_TOPIC_HISTORY_STORAGE_KEY = "studyreels-reel-topic-history";
const MATERIAL_SEEDS_STORAGE_KEY = "studyreels-material-seeds";
const MATERIAL_GROUPS_STORAGE_KEY = "studyreels-material-groups";
const FEED_PROGRESS_STORAGE_KEY = "studyreels-feed-progress";
const FEED_SESSION_STORAGE_KEY = "studyreels-feed-sessions";
const MAX_HISTORY_ITEMS = 120;
const MOBILE_SIDEBAR_CLOSE_MS = 260;
const TOP_CHROME_GESTURE_WINDOW_MS = 220;
const SIDEBAR_INFO_TOOLTIP_DELAY_MS = 1000;
const SIDEBAR_INFO_TOOLTIP_VISIBLE_MS = 2200;
const SIDEBAR_INFO_TOOLTIP_FADE_MS = 180;
const SIDEBAR_INFO_TOOLTIP_ANIMATE_IN_MS = 24;
type GenerationMode = "slow" | "fast";
type SidebarTab = "search" | "community" | "create";

type HistoryItem = {
  materialId: string;
  title: string;
  updatedAt: number;
  starred: boolean;
  generationMode: GenerationMode;
};

function parseMaterialHistory(raw: string | null): HistoryItem[] {
  if (!raw) {
    return [];
  }
  try {
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) {
      return [];
    }
    return parsed
      .filter((item) => item && typeof item.materialId === "string" && typeof item.title === "string")
      .map((item): HistoryItem => ({
        materialId: String(item.materialId),
        title: String(item.title).trim() || "New Study Session",
        updatedAt: Number(item.updatedAt) || 0,
        starred: Boolean(item.starred),
        generationMode: item.generationMode === "fast" ? "fast" : "slow",
      }))
      .slice(0, MAX_HISTORY_ITEMS);
  } catch {
    return [];
  }
}

function parseLegacyTopicHistory(raw: string | null): HistoryItem[] {
  if (!raw) {
    return [];
  }
  try {
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) {
      return [];
    }
    return parsed
      .filter(
        (item) =>
          item &&
          typeof item.materialId === "string" &&
          typeof item.topic === "string" &&
          String(item.materialId).trim() &&
          String(item.topic).trim(),
      )
      .map((item): HistoryItem => ({
        materialId: String(item.materialId),
        title: String(item.topic).trim(),
        updatedAt: Number(item.updatedAt) || 0,
        starred: false,
        generationMode: "slow",
      }))
      .slice(0, MAX_HISTORY_ITEMS);
  } catch {
    return [];
  }
}

function mergeHistory(primary: HistoryItem[], secondary: HistoryItem[]): HistoryItem[] {
  const map = new Map<string, HistoryItem>();
  for (const item of [...primary, ...secondary]) {
    const existing = map.get(item.materialId);
    if (!existing) {
      map.set(item.materialId, item);
      continue;
    }
    if (item.updatedAt > existing.updatedAt) {
      map.set(item.materialId, {
        ...item,
        starred: item.starred || existing.starred,
        generationMode: item.generationMode || existing.generationMode || "slow",
      });
      continue;
    }
    map.set(item.materialId, {
      ...existing,
      starred: existing.starred || item.starred,
      generationMode: existing.generationMode || item.generationMode || "slow",
    });
  }
  return [...map.values()].sort((a, b) => b.updatedAt - a.updatedAt).slice(0, MAX_HISTORY_ITEMS);
}

export default function HomePage() {
  const router = useRouter();
  const [historyQuery, setHistoryQuery] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  const [mobileSidebarClosing, setMobileSidebarClosing] = useState(false);
  const [topChromeOffset, setTopChromeOffset] = useState(false);
  const [topChromeGestureActive, setTopChromeGestureActive] = useState(false);
  const [activeHistoryMenuId, setActiveHistoryMenuId] = useState<string | null>(null);
  const [activeSidebarTab, setActiveSidebarTab] = useState<SidebarTab>("search");
  const [communityDetailOpen, setCommunityDetailOpen] = useState(false);
  const [sidebarInfoTooltip, setSidebarInfoTooltip] = useState<{ text: string; left: number; top: number; visible: boolean; align: "left" | "right" } | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const historyRef = useRef<HistoryItem[]>([]);
  const mobileSidebarCloseTimerRef = useRef<number | null>(null);
  const sidebarInfoTooltipShowTimerRef = useRef<number | null>(null);
  const sidebarInfoTooltipHideTimerRef = useRef<number | null>(null);
  const sidebarInfoTooltipDismissTimerRef = useRef<number | null>(null);
  const sidebarInfoTooltipAnimateInTimerRef = useRef<number | null>(null);
  const topChromeGestureTimerRef = useRef<number | null>(null);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const base = parseMaterialHistory(window.localStorage.getItem(HISTORY_STORAGE_KEY));
    const legacy = parseLegacyTopicHistory(window.localStorage.getItem(LEGACY_TOPIC_HISTORY_STORAGE_KEY));
    const merged = mergeHistory(base, legacy);
    historyRef.current = merged;
    setHistory(merged);
    window.localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(merged));
  }, []);

  const historySorted = useMemo(
    () =>
      [...history].sort((a, b) => {
        if (a.starred !== b.starred) {
          return Number(b.starred) - Number(a.starred);
        }
        return b.updatedAt - a.updatedAt;
      }),
    [history],
  );
  const filteredHistory = useMemo(() => {
    const query = historyQuery.trim().toLowerCase();
    if (!query) {
      return historySorted;
    }
    return historySorted.filter((item) => item.title.toLowerCase().includes(query));
  }, [historyQuery, historySorted]);

  const persistHistory = useCallback((next: HistoryItem[]) => {
    historyRef.current = next;
    setHistory(next);
    if (typeof window !== "undefined") {
      window.localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(next));
    }
  }, []);

  const upsertHistory = useCallback(
    (entry: { materialId: string; title: string; updatedAt: number; starred?: boolean; generationMode?: GenerationMode }) => {
      const existing = historyRef.current.find((item) => item.materialId === entry.materialId);
      const merged: HistoryItem = {
        materialId: entry.materialId,
        title: entry.title,
        updatedAt: entry.updatedAt,
        starred: entry.starred ?? existing?.starred ?? false,
        generationMode: entry.generationMode ?? existing?.generationMode ?? "slow",
      };
      const next = [merged, ...historyRef.current.filter((item) => item.materialId !== merged.materialId)].slice(0, MAX_HISTORY_ITEMS);
      persistHistory(next);
    },
    [persistHistory],
  );

  const clearAllHistory = useCallback(() => {
    historyRef.current = [];
    setHistory([]);
    setHistoryQuery("");
    setError(null);
    setActiveHistoryMenuId(null);
    if (typeof window !== "undefined") {
      window.localStorage.removeItem(HISTORY_STORAGE_KEY);
      window.localStorage.removeItem(LEGACY_TOPIC_HISTORY_STORAGE_KEY);
      window.localStorage.removeItem(MATERIAL_SEEDS_STORAGE_KEY);
      window.localStorage.removeItem(MATERIAL_GROUPS_STORAGE_KEY);
      window.localStorage.removeItem(FEED_PROGRESS_STORAGE_KEY);
      window.localStorage.removeItem(FEED_SESSION_STORAGE_KEY);
    }
  }, []);

  const clearMobileSidebarCloseTimer = useCallback(() => {
    if (mobileSidebarCloseTimerRef.current !== null) {
      window.clearTimeout(mobileSidebarCloseTimerRef.current);
      mobileSidebarCloseTimerRef.current = null;
    }
  }, []);

  const clearTopChromeGestureTimer = useCallback(() => {
    if (topChromeGestureTimerRef.current !== null) {
      window.clearTimeout(topChromeGestureTimerRef.current);
      topChromeGestureTimerRef.current = null;
    }
  }, []);

  const triggerTopChromeGesture = useCallback(() => {
    if (typeof window === "undefined") {
      return;
    }
    clearTopChromeGestureTimer();
    setTopChromeGestureActive(true);
    topChromeGestureTimerRef.current = window.setTimeout(() => {
      setTopChromeGestureActive(false);
      topChromeGestureTimerRef.current = null;
    }, TOP_CHROME_GESTURE_WINDOW_MS);
  }, [clearTopChromeGestureTimer]);

  const clearSidebarInfoTooltipTimers = useCallback(() => {
    if (sidebarInfoTooltipShowTimerRef.current !== null) {
      window.clearTimeout(sidebarInfoTooltipShowTimerRef.current);
      sidebarInfoTooltipShowTimerRef.current = null;
    }
    if (sidebarInfoTooltipHideTimerRef.current !== null) {
      window.clearTimeout(sidebarInfoTooltipHideTimerRef.current);
      sidebarInfoTooltipHideTimerRef.current = null;
    }
    if (sidebarInfoTooltipDismissTimerRef.current !== null) {
      window.clearTimeout(sidebarInfoTooltipDismissTimerRef.current);
      sidebarInfoTooltipDismissTimerRef.current = null;
    }
    if (sidebarInfoTooltipAnimateInTimerRef.current !== null) {
      window.clearTimeout(sidebarInfoTooltipAnimateInTimerRef.current);
      sidebarInfoTooltipAnimateInTimerRef.current = null;
    }
  }, []);

  const shouldDisableSidebarTooltips = useCallback(() => {
    if (typeof window === "undefined") {
      return false;
    }
    return window.matchMedia("(max-width: 1023px), (hover: none), (pointer: coarse)").matches;
  }, []);

  const onSidebarInfoHoverStart = useCallback(
    (event: ReactMouseEvent<HTMLElement>, text: string, align: "left" | "right" = "left") => {
      if (typeof window === "undefined") {
        return;
      }
      if (shouldDisableSidebarTooltips()) {
        clearSidebarInfoTooltipTimers();
        setSidebarInfoTooltip(null);
        return;
      }
      const rect = event.currentTarget.getBoundingClientRect();
      clearSidebarInfoTooltipTimers();
      setSidebarInfoTooltip(null);
      sidebarInfoTooltipShowTimerRef.current = window.setTimeout(() => {
        const left = align === "right"
          ? Math.min(window.innerWidth - 12, Math.max(232, rect.right))
          : Math.min(window.innerWidth - 232, Math.max(12, rect.left));
        const top = Math.min(window.innerHeight - 12, rect.bottom + 8);
        setSidebarInfoTooltip({ text, left, top, visible: false, align });
        sidebarInfoTooltipAnimateInTimerRef.current = window.setTimeout(() => {
          setSidebarInfoTooltip((prev) => (prev ? { ...prev, visible: true } : prev));
          sidebarInfoTooltipAnimateInTimerRef.current = null;
        }, SIDEBAR_INFO_TOOLTIP_ANIMATE_IN_MS);
        sidebarInfoTooltipShowTimerRef.current = null;
        sidebarInfoTooltipHideTimerRef.current = window.setTimeout(() => {
          setSidebarInfoTooltip((prev) => (prev ? { ...prev, visible: false } : prev));
          sidebarInfoTooltipDismissTimerRef.current = window.setTimeout(() => {
            setSidebarInfoTooltip(null);
            sidebarInfoTooltipDismissTimerRef.current = null;
          }, SIDEBAR_INFO_TOOLTIP_FADE_MS);
          sidebarInfoTooltipHideTimerRef.current = null;
        }, SIDEBAR_INFO_TOOLTIP_VISIBLE_MS);
      }, SIDEBAR_INFO_TOOLTIP_DELAY_MS);
    },
    [clearSidebarInfoTooltipTimers, shouldDisableSidebarTooltips],
  );

  const onSidebarInfoHoverEnd = useCallback(() => {
    if (shouldDisableSidebarTooltips()) {
      clearSidebarInfoTooltipTimers();
      setSidebarInfoTooltip(null);
      return;
    }
    clearSidebarInfoTooltipTimers();
    setSidebarInfoTooltip((prev) => (prev ? { ...prev, visible: false } : prev));
    if (typeof window === "undefined") {
      setSidebarInfoTooltip(null);
      return;
    }
    sidebarInfoTooltipDismissTimerRef.current = window.setTimeout(() => {
      setSidebarInfoTooltip(null);
      sidebarInfoTooltipDismissTimerRef.current = null;
    }, SIDEBAR_INFO_TOOLTIP_FADE_MS);
  }, [clearSidebarInfoTooltipTimers, shouldDisableSidebarTooltips]);

  const openMobileSidebar = useCallback(() => {
    clearMobileSidebarCloseTimer();
    setMobileSidebarClosing(false);
    setMobileSidebarOpen(true);
  }, [clearMobileSidebarCloseTimer]);

  const closeMobileSidebar = useCallback(() => {
    if (!mobileSidebarOpen || mobileSidebarClosing) {
      return;
    }
    clearMobileSidebarCloseTimer();
    setMobileSidebarClosing(true);
    mobileSidebarCloseTimerRef.current = window.setTimeout(() => {
      setMobileSidebarOpen(false);
      setMobileSidebarClosing(false);
      mobileSidebarCloseTimerRef.current = null;
    }, MOBILE_SIDEBAR_CLOSE_MS);
  }, [clearMobileSidebarCloseTimer, mobileSidebarClosing, mobileSidebarOpen]);

  const forceCloseMobileSidebar = useCallback(() => {
    clearMobileSidebarCloseTimer();
    setMobileSidebarClosing(false);
    setMobileSidebarOpen(false);
  }, [clearMobileSidebarCloseTimer]);

  const startNewSearch = useCallback(() => {
    setActiveSidebarTab("search");
    setHistoryQuery("");
    setError(null);
    setActiveHistoryMenuId(null);
    forceCloseMobileSidebar();
  }, [forceCloseMobileSidebar]);

  const isSidebarInteractiveTarget = useCallback((target: EventTarget | null): boolean => {
    if (!(target instanceof Element)) {
      return false;
    }
    return Boolean(
      target.closest(
        "button, a, input, textarea, select, label, [role='button'], [data-history-actions='true'], [contenteditable='true']",
      ),
    );
  }, []);

  useEffect(() => {
    return () => {
      clearMobileSidebarCloseTimer();
      clearSidebarInfoTooltipTimers();
      clearTopChromeGestureTimer();
    };
  }, [clearMobileSidebarCloseTimer, clearSidebarInfoTooltipTimers, clearTopChromeGestureTimer]);

  const onMainScroll = useCallback((event: UIEvent<HTMLElement>) => {
    const isOffset = event.currentTarget.scrollTop > 0;
    setTopChromeOffset((prev) => (prev === isOffset ? prev : isOffset));
  }, []);

  const onSearchPanelScrollOffsetChange = useCallback((isOffset: boolean) => {
    if (activeSidebarTab !== "search") {
      return;
    }
    setTopChromeOffset((prev) => (prev === isOffset ? prev : isOffset));
    if (!isOffset) {
      clearTopChromeGestureTimer();
      setTopChromeGestureActive(false);
    }
  }, [activeSidebarTab, clearTopChromeGestureTimer]);

  useEffect(() => {
    if (activeSidebarTab === "search") {
      setTopChromeOffset(false);
      setTopChromeGestureActive(false);
    }
  }, [activeSidebarTab]);

  const openMaterialFeed = useCallback(
    (materialId: string) => {
      const existing = historyRef.current.find((item) => item.materialId === materialId);
      if (existing) {
        upsertHistory({ ...existing, updatedAt: Date.now() });
      }
      setActiveHistoryMenuId(null);
      forceCloseMobileSidebar();
      const mode = existing?.generationMode ?? "slow";
      router.push(`/feed?material_id=${materialId}&generation_mode=${mode}`);
    },
    [forceCloseMobileSidebar, router, upsertHistory],
  );

  const toggleHistoryStar = useCallback(
    (materialId: string) => {
      const existing = historyRef.current.find((item) => item.materialId === materialId);
      if (!existing) {
        return;
      }
      upsertHistory({
        ...existing,
        starred: !existing.starred,
        updatedAt: existing.updatedAt,
      });
      setActiveHistoryMenuId(null);
    },
    [upsertHistory],
  );

  const deleteHistoryItem = useCallback(
    (materialId: string) => {
      const next = historyRef.current.filter((item) => item.materialId !== materialId);
      persistHistory(next);
      setActiveHistoryMenuId((prev) => (prev === materialId ? null : prev));
    },
    [persistHistory],
  );

  useEffect(() => {
    if (!activeHistoryMenuId) {
      return;
    }
    const onPointerDown = (event: PointerEvent) => {
      if (!(event.target instanceof Element)) {
        return;
      }
      if (event.target.closest("[data-history-actions='true']")) {
        return;
      }
      setActiveHistoryMenuId(null);
    };
    window.addEventListener("pointerdown", onPointerDown);
    return () => {
      window.removeEventListener("pointerdown", onPointerDown);
    };
  }, [activeHistoryMenuId]);

  const onUploadMaterialCreated = useCallback(
    async (params: { materialId: string; title: string; topic?: string; generationMode: GenerationMode }) => {
      const nextTitle = params.title?.trim() || params.topic?.trim() || "New Study Session";
      upsertHistory({
        materialId: params.materialId,
        title: nextTitle,
        updatedAt: Date.now(),
        generationMode: params.generationMode,
      });
    },
    [upsertHistory],
  );

  const switchSidebarTab = useCallback(
    (tab: SidebarTab) => {
      setActiveSidebarTab(tab);
      setActiveHistoryMenuId(null);
      setError(null);
      forceCloseMobileSidebar();
    },
    [forceCloseMobileSidebar],
  );
  const isCommunityPanel = activeSidebarTab === "community" || activeSidebarTab === "create";
  const hideMobileTopControls = isCommunityPanel && communityDetailOpen;
  const [lastCommunityPanelMode, setLastCommunityPanelMode] = useState<"community" | "create">("community");

  useEffect(() => {
    if (activeSidebarTab === "community" || activeSidebarTab === "create") {
      setLastCommunityPanelMode(activeSidebarTab);
    }
  }, [activeSidebarTab]);

  const communityPanelMode = activeSidebarTab === "search"
    ? lastCommunityPanelMode
    : activeSidebarTab === "create"
      ? "create"
      : "community";

  const sidebarPanelContent = (
    <>
      <div className="mt-2 flex items-center justify-end gap-2 lg:mt-0 lg:justify-between">
        <span
          aria-hidden="true"
          className="hidden h-8 w-8 -translate-x-2 items-center justify-center text-xl font-black leading-none tracking-tight text-white/58 lg:inline-flex"
        >
          R
        </span>
        <div
          className="group relative"
          onMouseEnter={(event) => onSidebarInfoHoverStart(event, "Start a new search", "right")}
          onMouseLeave={onSidebarInfoHoverEnd}
        >
          <button
            type="button"
            onClick={startNewSearch}
            aria-label="Start new search"
            className="grid h-8 w-8 place-items-center rounded-xl border border-white/15 bg-transparent text-sm font-semibold text-white/90 transition-colors duration-200 hover:bg-white/10 hover:text-white"
          >
            +
          </button>
        </div>
      </div>

      <div
        className="group relative mt-3"
        onMouseEnter={(event) => onSidebarInfoHoverStart(event, "Start a new chat session")}
        onMouseLeave={onSidebarInfoHoverEnd}
      >
        <button
          type="button"
          onClick={startNewSearch}
          className={`h-9 w-full rounded-xl border bg-transparent px-2.5 text-left text-xs transition-colors duration-200 ${
            activeSidebarTab === "search" ? "border-white bg-white text-black" : "border-white/15 text-white/85 hover:bg-white/10 hover:text-white"
          }`}
        >
          <div className="flex h-full items-center justify-between gap-1.5">
            <p className="truncate font-semibold leading-none">Search</p>
            <i
              className={`fa-solid fa-magnifying-glass text-[11px] ${
                activeSidebarTab === "search" ? "text-black/80" : "text-white/74 transition-colors duration-200 group-hover:text-white"
              }`}
              aria-hidden="true"
            />
          </div>
        </button>
      </div>

      <div
        className="group relative mt-2"
        onMouseEnter={(event) => onSidebarInfoHoverStart(event, "Curated reel sets from the community")}
        onMouseLeave={onSidebarInfoHoverEnd}
      >
        <button
          type="button"
          onClick={() => switchSidebarTab("community")}
          className={`h-9 w-full rounded-xl border bg-transparent px-2.5 text-left text-xs transition-colors duration-200 ${
            activeSidebarTab === "community"
              ? "border-white bg-white text-black"
              : "border-white/15 text-white/85 hover:bg-white/10 hover:text-white"
          }`}
        >
          <div className="flex h-full items-center justify-between gap-1.5">
            <p className="truncate font-semibold leading-none">Community Reels</p>
            <i
              className={`fa-solid fa-users text-[11px] ${
                activeSidebarTab === "community" ? "text-black/80" : "text-white/74 transition-colors duration-200 group-hover:text-white"
              }`}
              aria-hidden="true"
            />
          </div>
        </button>
      </div>

      <div
        className="group relative mt-2"
        onMouseEnter={(event) => onSidebarInfoHoverStart(event, "Build and publish your own community reel set")}
        onMouseLeave={onSidebarInfoHoverEnd}
      >
        <button
          type="button"
          onClick={() => switchSidebarTab("create")}
          className={`h-9 w-full rounded-xl border bg-transparent px-2.5 text-left text-xs transition-colors duration-200 ${
            activeSidebarTab === "create" ? "border-white bg-white text-black" : "border-white/15 text-white/85 hover:bg-white/10 hover:text-white"
          }`}
        >
          <div className="flex h-full items-center justify-between gap-1.5">
            <p className="truncate font-semibold leading-none">Create Set</p>
            <i
              className={`fa-solid fa-plus text-[11px] ${
                activeSidebarTab === "create" ? "text-black/80" : "text-white/74 transition-colors duration-200 group-hover:text-white"
              }`}
              aria-hidden="true"
            />
          </div>
        </button>
      </div>

      <div className="mt-6">
        <input
          value={historyQuery}
          onChange={(event) => setHistoryQuery(event.target.value)}
          placeholder="Search history..."
          className="h-9 w-full rounded-xl border border-white/20 bg-black/55 px-3 text-sm text-white outline-none placeholder:text-white/45 focus:border-white/45"
        />
      </div>

      <div className="mt-4 min-h-0 pr-1 lg:flex-1 lg:overflow-y-auto">
        <div className="mb-2 flex items-center justify-between">
          <p className="text-[10px] font-semibold uppercase tracking-[0.1em] text-white/60">History</p>
          <button
            type="button"
            onClick={clearAllHistory}
            className="text-[10px] font-semibold uppercase tracking-[0.1em] text-white/60 transition hover:text-white"
          >
            Clear History
          </button>
        </div>
        <div className="relative pb-1">
          <div className="relative space-y-1.5">
            {filteredHistory.length === 0 ? (
              <p className="text-xs text-white/42">No history yet.</p>
            ) : (
              filteredHistory.map((entry) => (
                <div
                  key={`history-${entry.materialId}`}
                  className="group relative"
                >
                  <button
                    type="button"
                    onClick={() => openMaterialFeed(entry.materialId)}
                    className="h-9 w-full rounded-xl border border-white/15 bg-black/45 px-2.5 pr-10 text-left text-xs text-white/85 transition-colors duration-200 hover:bg-white/10 hover:text-white"
                  >
                    <div className="flex items-center gap-1.5">
                      {entry.starred ? <i className="fa-solid fa-star text-[10px] text-white/75 transition-colors group-hover:text-white" aria-hidden="true" /> : null}
                      <p className="truncate font-semibold leading-none transition-colors group-hover:text-white">{entry.title}</p>
                      <span className="shrink-0 rounded-md bg-black/55 px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-[0.08em] text-white/68 transition-colors group-hover:text-white">
                        {entry.generationMode}
                      </span>
                    </div>
                  </button>

                  <div data-history-actions="true" className="absolute right-1.5 top-1.5 z-20">
                    <button
                      type="button"
                      aria-label="History item actions"
                      onClick={(event) => {
                        event.stopPropagation();
                        setActiveHistoryMenuId((prev) => (prev === entry.materialId ? null : entry.materialId));
                      }}
                      className="grid h-6 w-6 place-items-center rounded-md text-white/70 opacity-100 transition hover:bg-white/10 hover:text-white lg:opacity-0 lg:group-hover:opacity-100 lg:focus-within:opacity-100"
                    >
                      <i className="fa-solid fa-ellipsis text-[11px]" aria-hidden="true" />
                    </button>

                    {activeHistoryMenuId === entry.materialId ? (
                      <div className="absolute right-0 top-full z-30 mt-1 w-36">
                        <div className="absolute inset-0 rounded-xl bg-black/32 backdrop-blur-md" />
                        <div className="relative rounded-xl border border-white/20 bg-black/62 p-1 shadow-lg">
                          <button
                            type="button"
                            onClick={(event) => {
                              event.stopPropagation();
                              toggleHistoryStar(entry.materialId);
                            }}
                            className="flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left text-xs text-white/90 transition hover:bg-white/10"
                          >
                            <i
                              className={`fa-${entry.starred ? "solid" : "regular"} fa-star text-[11px] text-white/80`}
                              aria-hidden="true"
                            />
                            {entry.starred ? "Unstar" : "Star"}
                          </button>
                          <button
                            type="button"
                            onClick={(event) => {
                              event.stopPropagation();
                              deleteHistoryItem(entry.materialId);
                            }}
                            className="mt-0.5 flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left text-xs text-white/90 transition hover:bg-white/10"
                          >
                            <i className="fa-regular fa-trash-can text-[11px] text-white/80" aria-hidden="true" />
                            Delete
                          </button>
                        </div>
                      </div>
                    ) : null}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </>
  );

  return (
    <main onScroll={onMainScroll} className="home-hero-shell fixed inset-0 h-[100dvh] overflow-x-hidden overflow-y-auto lg:overflow-hidden">
      {activeSidebarTab === "search" ? (
        <div className="absolute -top-[28%] -right-[0%] bottom-0 z-[2] w-[100%]">
          <VolumetricLightBackground />
        </div>
      ) : null}
      {isCommunityPanel ? (
        <div className="pointer-events-none absolute inset-0 z-[3] overflow-hidden">
          <div className="absolute inset-0 bg-black/30" />
          <img
            src="/images/community/80543.jpg"
            alt=""
            aria-hidden="true"
            className="absolute bottom-0 right-0 h-auto w-[82vw] max-w-none rotate-180 object-contain sm:w-[74vw] md:w-[66vw] lg:w-[58vw]"
            style={{
              opacity: 0.15,
              WebkitMaskImage: "linear-gradient(to right, rgba(0, 0, 0, 1) 0%, rgba(0, 0, 0, 1) 90%, rgba(0, 0, 0, 0) 100%)",
              maskImage: "linear-gradient(to right, rgba(0, 0, 0, 1) 0%, rgba(0, 0, 0, 1) 90%, rgba(0, 0, 0, 0) 100%)",
              WebkitMaskRepeat: "no-repeat",
              maskRepeat: "no-repeat",
              WebkitMaskSize: "100% 100%",
              maskSize: "100% 100%",
            }}
          />
        </div>
      ) : null}
      {error ? (
        <div className="absolute left-0 right-0 top-3 z-30 mx-auto w-fit rounded-full border border-white/25 bg-black/80 px-4 py-2 text-xs text-white">
          {error}
        </div>
      ) : null}
      {sidebarInfoTooltip && !shouldDisableSidebarTooltips() ? (
        <div
          className={`pointer-events-none fixed z-[90] max-w-[220px] rounded-lg border border-white/15 bg-black/95 px-2 py-1 text-left text-[10px] text-white/92 shadow-[0_12px_30px_rgba(0,0,0,0.5)] backdrop-blur-sm transition-[opacity,transform] duration-[220ms] ease-out will-change-[transform,opacity] ${
            sidebarInfoTooltip.visible ? "translate-y-0 opacity-100" : "-translate-y-1.5 opacity-0"
          } ${sidebarInfoTooltip.align === "right" ? "origin-top-right -translate-x-full" : "origin-top-left"}`}
          style={{ left: `${sidebarInfoTooltip.left}px`, top: `${sidebarInfoTooltip.top}px` }}
        >
          {sidebarInfoTooltip.text}
        </div>
      ) : null}

      <div
        aria-hidden="true"
        className={`pointer-events-none fixed inset-x-0 top-0 z-[68] h-[calc(max(env(safe-area-inset-top),0px)+68px)] ${
          activeSidebarTab === "search" ? "transition-none" : "transition-opacity duration-150"
        } lg:hidden ${
          (topChromeOffset || topChromeGestureActive) && !mobileSidebarOpen ? "opacity-100" : "opacity-0"
        }`}
      >
        <div
          className={`h-full w-full ${
            activeSidebarTab === "community"
              ? "bg-black"
              : activeSidebarTab === "search"
                ? "relative overflow-hidden bg-white/[0.05] backdrop-blur-[10px] backdrop-saturate-160"
                : "bg-black/28 backdrop-blur-[28px] backdrop-saturate-180"
          }`}
        >
          {activeSidebarTab === "search" ? <div className="absolute inset-0 bg-black/45" /> : null}
        </div>
      </div>

      <button
        type="button"
        onClick={openMobileSidebar}
        aria-label="Open topic menu"
        style={{
          left: "calc(max(env(safe-area-inset-left), 0px) + 10px)",
          top: "calc(max(env(safe-area-inset-top), 0px) + 10px)",
        }}
        className={`fixed z-[110] grid h-10 w-10 place-items-center text-white/90 transition-opacity hover:text-white md:left-7 md:top-7 lg:hidden ${
          mobileSidebarOpen || hideMobileTopControls ? "pointer-events-none opacity-0" : "opacity-100"
        }`}
      >
        <i className="fa-solid fa-bars text-base" aria-hidden="true" />
      </button>
      <div
        aria-hidden="true"
        style={{
          top: "calc(max(env(safe-area-inset-top), 0px) + 10px)",
        }}
        className={`pointer-events-none fixed left-1/2 z-[110] flex h-10 -translate-x-1/2 items-center transition-opacity lg:hidden ${
          mobileSidebarOpen || hideMobileTopControls ? "opacity-0" : "opacity-100"
        }`}
      >
        <img src="/logo.png" alt="" className="h-auto w-[5rem] object-contain opacity-70 md:w-[5.5rem]" />
      </div>

      {mobileSidebarOpen ? (
        <div className="fixed left-0 top-0 z-50 h-[100dvh] w-screen lg:hidden">
          <button
            type="button"
            aria-label="Close topic menu overlay"
            onClick={closeMobileSidebar}
            className={`absolute inset-0 bg-black/70 ${mobileSidebarClosing ? "animate-mobile-overlay-out" : "animate-mobile-overlay-in"}`}
          />
          <aside
            onClick={(event) => {
              if (isSidebarInteractiveTarget(event.target)) {
                return;
              }
              closeMobileSidebar();
            }}
            className={`absolute bottom-4 left-4 top-6 w-[min(24rem,calc(100vw-2rem))] rounded-3xl bg-black/30 px-3 pb-3 pt-3 text-white shadow-[0_0_40px_rgba(0,0,0,0.45)] backdrop-blur-[26px] backdrop-saturate-150 ${
              mobileSidebarClosing ? "animate-mobile-sidenav-out" : "animate-mobile-sidenav-in"
            }`}
          >
            <div className="flex h-full min-h-0 flex-col">
              <div className="-mx-3 shrink-0 px-3 pb-2 pt-0.5">
                <div className="relative flex h-10 items-center justify-between overflow-hidden rounded-2xl px-0.5">
                  <span
                    aria-hidden="true"
                    className="relative z-10 inline-flex h-10 w-10 -translate-x-2 items-center justify-center text-2xl font-black leading-none tracking-tight text-white/58"
                  >
                    R
                  </span>
                  <button
                    type="button"
                    onClick={closeMobileSidebar}
                    aria-label="Close topic menu"
                    className="relative z-10 mr-1 -translate-y-0.5 p-1 text-white/80 transition hover:text-white"
                  >
                    <i className="fa-solid fa-xmark text-base" aria-hidden="true" />
                  </button>
                </div>
              </div>
              <div className="min-h-0 flex-1 overflow-y-auto">
                <div className="rounded-2xl">
                  {sidebarPanelContent}
                </div>
              </div>
            </div>
          </aside>
        </div>
      ) : null}

      <div className="relative z-20 mx-auto h-full min-h-0 w-full max-w-[1680px] lg:grid lg:grid-cols-[280px_1px_minmax(0,1fr)]">
        <aside className="relative z-20 hidden min-h-0 flex-col rounded-3xl bg-black/72 px-3 pt-3 pb-2 text-white lg:mt-8 lg:mb-2 lg:flex lg:px-5">
          {sidebarPanelContent}
        </aside>

        <div className="relative z-20 hidden h-full items-center justify-center lg:flex lg:translate-x-3">
          <span className="h-[80%] w-px rounded-full bg-white/20" />
        </div>

        <section
          className={`relative z-20 h-full min-h-0 w-full overflow-hidden rounded-3xl ${
            isCommunityPanel ? "bg-transparent" : "bg-black/62"
          } lg:my-2 lg:justify-self-end ${
            activeSidebarTab === "community"
              ? "translate-x-0 md:translate-x-1 lg:translate-x-2 lg:w-[99%]"
              : "lg:w-[97%]"
          }`}
        >
          <div className={activeSidebarTab === "search" ? "h-full min-h-0" : "hidden h-full min-h-0"}>
            <UploadPanel
              onMaterialCreated={onUploadMaterialCreated}
              onScrollOffsetChange={onSearchPanelScrollOffsetChange}
              onScrollGesture={triggerTopChromeGesture}
            />
          </div>
          <div className={activeSidebarTab === "search" ? "hidden h-full min-h-0" : "h-full min-h-0"}>
            <CommunityReelsPanel
              mode={communityPanelMode}
              isVisible={activeSidebarTab !== "search"}
              onDetailOpenChange={setCommunityDetailOpen}
            />
          </div>
        </section>
      </div>
    </main>
  );
}
