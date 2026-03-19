"use client";

import { Suspense, type MouseEvent as ReactMouseEvent, type UIEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

import { type CommunityDraftExitActions, CommunityReelsPanel } from "@/components/CommunityReelsPanel";
import { FullscreenLoadingScreen } from "@/components/FullscreenLoadingScreen";
import { ViewportModalPortal } from "@/components/ViewportModalPortal";
import {
  COMMUNITY_AUTH_CHANGED_EVENT,
  clearCommunityAuthSession,
  fetchCommunityAccount,
  fetchCommunityHistory,
  fetchCommunitySettings,
  isSessionExpiredError,
  logoutCommunityAccount,
  queueCommunityHistorySync,
  queueCommunitySettingsSync,
  readCommunityAuthSession,
} from "@/lib/api";
import { safeStorageRemoveItem, safeStorageSetItem } from "@/lib/browserStorage";
import {
  LEGACY_TOPIC_HISTORY_STORAGE_KEY,
  normalizeStoredHistoryItems,
  readScopedHistorySnapshot,
  type StoredHistoryItem,
  writeScopedHistorySnapshot,
} from "@/lib/historyStorage";
import type { CommunityAccount } from "@/lib/types";
import {
  type SettingsAvailabilityModalSnapshot,
  type SettingsAvailabilityState,
  type SettingsPanelHandle,
  SettingsPanel,
} from "@/components/SettingsPanel";
import { UploadPanel } from "@/components/UploadPanel";
import { VolumetricLightBackground } from "@/components/VolumetricLightBackground";
import { setActiveStudyReelsSettingsScope, type StudyReelsSettings } from "@/lib/settings";
import { useLoadingScreenGate } from "@/lib/useLoadingScreenGate";

const MATERIAL_SEEDS_STORAGE_KEY = "studyreels-material-seeds";
const MATERIAL_GROUPS_STORAGE_KEY = "studyreels-material-groups";
const FEED_PROGRESS_STORAGE_KEY = "studyreels-feed-progress";
const FEED_SESSION_STORAGE_KEY = "studyreels-feed-sessions";
const ACTIVE_SIDEBAR_TAB_SESSION_KEY = "studyreels-active-sidebar-tab";
const ACTIVE_COMMUNITY_SET_ID_SESSION_KEY = "studyreels-active-community-set-id";
const MAX_HISTORY_ITEMS = 120;
const MOBILE_SIDEBAR_CLOSE_MS = 260;
const TOP_CHROME_GESTURE_WINDOW_MS = 220;
const SIDEBAR_INFO_TOOLTIP_DELAY_MS = 1000;
const SIDEBAR_INFO_TOOLTIP_VISIBLE_MS = 2200;
const SIDEBAR_INFO_TOOLTIP_FADE_MS = 180;
const SIDEBAR_INFO_TOOLTIP_ANIMATE_IN_MS = 24;
type GenerationMode = StoredHistoryItem["generationMode"];
type HistorySource = StoredHistoryItem["source"];
type SidebarTab = "search" | "history" | "community" | "create" | "edit" | "settings";
type SidebarSwitchIntent = {
  tab: SidebarTab;
  clearHistoryQuery?: boolean;
  resetCommunityView?: boolean;
};
type HistoryInfoField = {
  label: string;
  value: string;
};
type HistoryInfoSection = {
  title: string;
  fields: HistoryInfoField[];
};

const DEFAULT_SETTINGS_AVAILABILITY_STATE: SettingsAvailabilityState = {
  status: "checking",
  message: "Estimating success rate from configuration heuristics...",
  limitingFactors: [],
};

type HistoryItem = StoredHistoryItem;

function normalizeSidebarTab(value: string | null): SidebarTab | null {
  if (value === "create" || value === "edit") {
    return "edit";
  }
  if (value === "search" || value === "history" || value === "community" || value === "settings") {
    return value;
  }
  return null;
}

function parseCommunitySetIdFromHistoryMaterialId(materialId: string): string | null {
  const prefix = "community:";
  if (!materialId.startsWith(prefix)) {
    return null;
  }
  const raw = materialId.slice(prefix.length).trim();
  if (!raw) {
    return null;
  }
  try {
    return decodeURIComponent(raw);
  } catch {
    return raw;
  }
}

function parseMaterialHistory(raw: string | null): HistoryItem[] {
  if (!raw) {
    return [];
  }
  try {
    return normalizeStoredHistoryItems(JSON.parse(raw)).slice(0, MAX_HISTORY_ITEMS);
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
        generationMode: "fast",
        source: "search",
        feedQuery: undefined,
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
        generationMode: item.generationMode || existing.generationMode || "fast",
        source: item.source || existing.source || "search",
        feedQuery: item.feedQuery ?? existing.feedQuery,
        activeIndex: item.activeIndex ?? existing.activeIndex,
        activeReelId: item.activeReelId ?? existing.activeReelId,
      });
      continue;
    }
    map.set(item.materialId, {
      ...existing,
      starred: existing.starred || item.starred,
      generationMode: existing.generationMode || item.generationMode || "fast",
      source: existing.source || item.source || "search",
      feedQuery: existing.feedQuery ?? item.feedQuery,
      activeIndex: existing.activeIndex ?? item.activeIndex,
      activeReelId: existing.activeReelId ?? item.activeReelId,
    });
  }
  return [...map.values()].sort((a, b) => b.updatedAt - a.updatedAt).slice(0, MAX_HISTORY_ITEMS);
}

function pushHistoryInfoField(fields: HistoryInfoField[], label: string, value: string | null | undefined): void {
  const normalized = typeof value === "string" ? value.trim() : "";
  if (!normalized) {
    return;
  }
  fields.push({ label, value: normalized });
}

function formatHistoryInfoToken(value: string | null | undefined): string | null {
  const normalized = String(value || "").trim();
  if (!normalized) {
    return null;
  }
  return normalized
    .split(/[-_]/)
    .filter(Boolean)
    .map((part) => {
      const lower = part.toLowerCase();
      return `${lower.charAt(0).toUpperCase()}${lower.slice(1)}`;
    })
    .join(" ");
}

function formatHistoryInfoBoolean(value: boolean): string {
  return value ? "Yes" : "No";
}

function formatHistoryInfoBooleanQuery(value: string | null): string | null {
  const normalized = String(value || "").trim().toLowerCase();
  if (!normalized) {
    return null;
  }
  if (normalized === "1" || normalized === "true") {
    return "Yes";
  }
  if (normalized === "0" || normalized === "false") {
    return "No";
  }
  return null;
}

function formatHistoryInfoDate(timestamp: number): string {
  if (!Number.isFinite(timestamp) || timestamp <= 0) {
    return "Unknown";
  }
  try {
    return new Date(timestamp).toLocaleString(undefined, {
      dateStyle: "medium",
      timeStyle: "short",
    });
  } catch {
    return new Date(timestamp).toLocaleString();
  }
}

function formatHistoryInfoSeconds(value: string | null | undefined): string | null {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < 0) {
    return null;
  }
  return `${Math.round(parsed)} sec`;
}

function formatHistoryInfoClipRange(minValue: string | null, maxValue: string | null): string | null {
  const minLabel = formatHistoryInfoSeconds(minValue);
  const maxLabel = formatHistoryInfoSeconds(maxValue);
  if (!minLabel && !maxLabel) {
    return null;
  }
  if (minLabel && maxLabel) {
    return minLabel === maxLabel ? minLabel : `${minLabel} to ${maxLabel}`;
  }
  return minLabel || maxLabel;
}

function formatHistoryInfoStrictness(value: string | null | undefined): string | null {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < 0) {
    return null;
  }
  return `${parsed.toFixed(2)} minimum relevance`;
}

function formatHistoryInfoReturnTab(value: string | null | undefined): string | null {
  const normalized = String(value || "").trim().toLowerCase();
  if (!normalized) {
    return null;
  }
  if (normalized === "create" || normalized === "edit") {
    return "Your Sets";
  }
  if (normalized === "history") {
    return "Search History";
  }
  return formatHistoryInfoToken(normalized);
}

function parseHistoryInfoQuery(feedQuery: string | undefined): URLSearchParams | null {
  const normalized = typeof feedQuery === "string" ? feedQuery.trim() : "";
  if (!normalized) {
    return null;
  }
  return new URLSearchParams(normalized);
}

function buildHistoryInfoSections(item: HistoryItem): HistoryInfoSection[] {
  const params = parseHistoryInfoQuery(item.feedQuery);
  const summaryFields: HistoryInfoField[] = [];
  const searchFields: HistoryInfoField[] = [];
  const playbackFields: HistoryInfoField[] = [];

  pushHistoryInfoField(summaryFields, "Source", item.source === "community" ? "Community" : "Search");
  pushHistoryInfoField(summaryFields, "Generation mode", item.generationMode === "slow" ? "Slow" : "Fast");
  pushHistoryInfoField(summaryFields, "Starred", formatHistoryInfoBoolean(item.starred));
  pushHistoryInfoField(summaryFields, "Last updated", formatHistoryInfoDate(item.updatedAt));
  pushHistoryInfoField(summaryFields, "Material ID", item.materialId);

  const communitySetId = params?.get("community_set_id") || parseCommunitySetIdFromHistoryMaterialId(item.materialId);
  pushHistoryInfoField(summaryFields, "Community set ID", communitySetId);
  pushHistoryInfoField(summaryFields, "Return tab", formatHistoryInfoReturnTab(params?.get("return_tab")));
  pushHistoryInfoField(summaryFields, "Return set ID", params?.get("return_community_set_id"));

  pushHistoryInfoField(searchFields, "Strictness", formatHistoryInfoStrictness(params?.get("min_relevance")));
  pushHistoryInfoField(searchFields, "Target clip length", formatHistoryInfoSeconds(params?.get("target_clip_duration_sec")));
  pushHistoryInfoField(
    searchFields,
    "Clip length range",
    formatHistoryInfoClipRange(params?.get("target_clip_duration_min_sec") || null, params?.get("target_clip_duration_max_sec") || null),
  );
  pushHistoryInfoField(searchFields, "Video pool", formatHistoryInfoToken(params?.get("video_pool_mode")));
  pushHistoryInfoField(searchFields, "Preferred duration", formatHistoryInfoToken(params?.get("preferred_video_duration")));
  pushHistoryInfoField(searchFields, "Start muted", formatHistoryInfoBooleanQuery(params?.get("start_muted") || null));
  pushHistoryInfoField(searchFields, "Creative Commons only", formatHistoryInfoBooleanQuery(params?.get("creative_commons_only") || null));

  if (typeof item.activeIndex === "number" && Number.isFinite(item.activeIndex) && item.activeIndex >= 0) {
    pushHistoryInfoField(playbackFields, "Resume reel", `#${item.activeIndex + 1}`);
  }
  pushHistoryInfoField(playbackFields, "Active reel ID", item.activeReelId);
  pushHistoryInfoField(playbackFields, "Community reel ID", params?.get("community_reel_id"));
  pushHistoryInfoField(playbackFields, "Platform", formatHistoryInfoToken(params?.get("community_reel_platform")));
  pushHistoryInfoField(playbackFields, "Clip start", formatHistoryInfoSeconds(params?.get("community_t_start_sec")));
  pushHistoryInfoField(playbackFields, "Clip end", formatHistoryInfoSeconds(params?.get("community_t_end_sec")));

  return [
    { title: "Session details", fields: summaryFields },
    { title: "Search settings", fields: searchFields },
    { title: "Playback", fields: playbackFields },
  ].filter((section) => section.fields.length > 0);
}

function HomePageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const forcedSidebarTab = useMemo(() => normalizeSidebarTab(searchParams.get("tab")), [searchParams]);
  const forcedCommunitySetId = useMemo(() => {
    const fromCommunitySet = searchParams.get("community_set_id");
    const fromReturnSet = searchParams.get("return_community_set_id");
    const raw = fromCommunitySet || fromReturnSet;
    if (!raw) {
      return null;
    }
    const trimmed = raw.trim();
    return trimmed || null;
  }, [searchParams]);
  const [historyQuery, setHistoryQuery] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  const [mobileSidebarClosing, setMobileSidebarClosing] = useState(false);
  const [topChromeOffset, setTopChromeOffset] = useState(false);
  const [topChromeGestureActive, setTopChromeGestureActive] = useState(false);
  const [searchPanelScrollable, setSearchPanelScrollable] = useState(false);
  const [activeSidebarHistoryMenuId, setActiveSidebarHistoryMenuId] = useState<string | null>(null);
  const [activeHistoryPageMenuId, setActiveHistoryPageMenuId] = useState<string | null>(null);
  const [selectedHistoryInfoId, setSelectedHistoryInfoId] = useState<string | null>(null);
  const [accountMenuOpen, setAccountMenuOpen] = useState(false);
  const [activeSidebarTab, setActiveSidebarTab] = useState<SidebarTab>("search");
  const [sidebarTabHydrated, setSidebarTabHydrated] = useState(false);
  const [activeCommunitySetId, setActiveCommunitySetId] = useState<string | null>(null);
  const [communityAccount, setCommunityAccount] = useState<CommunityAccount | null>(null);
  const [hasUnsavedSettingsChanges, setHasUnsavedSettingsChanges] = useState(false);
  const [settingsModalView, setSettingsModalView] = useState<null | "unsaved" | "availability">(null);
  const [settingsAvailabilityModalSnapshot, setSettingsAvailabilityModalSnapshot] = useState<SettingsAvailabilityModalSnapshot | null>(null);
  const [hasUnsavedCommunityDraftChanges, setHasUnsavedCommunityDraftChanges] = useState(false);
  const [showUnsavedCommunityDraftModal, setShowUnsavedCommunityDraftModal] = useState(false);
  const [pendingSidebarSwitchIntent, setPendingSidebarSwitchIntent] = useState<SidebarSwitchIntent | null>(null);
  const [pendingCommunityDraftSwitchIntent, setPendingCommunityDraftSwitchIntent] = useState<SidebarSwitchIntent | null>(null);
  const [pendingSaveSwitchUntilHeuristicClose, setPendingSaveSwitchUntilHeuristicClose] = useState(false);
  const [communityDetailOpen, setCommunityDetailOpen] = useState(false);
  const [communityResetSignal, setCommunityResetSignal] = useState(0);
  const [sidebarInfoTooltip, setSidebarInfoTooltip] = useState<{ text: string; left: number; top: number; visible: boolean; align: "left" | "right" } | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const historyRef = useRef<HistoryItem[]>([]);
  const settingsPanelRef = useRef<SettingsPanelHandle | null>(null);
  const communityDraftExitActionsRef = useRef<CommunityDraftExitActions | null>(null);
  const hasProcessedInitialSidebarHydrationRef = useRef(false);
  const mobileSidebarCloseTimerRef = useRef<number | null>(null);
  const sidebarInfoTooltipShowTimerRef = useRef<number | null>(null);
  const sidebarInfoTooltipHideTimerRef = useRef<number | null>(null);
  const sidebarInfoTooltipDismissTimerRef = useRef<number | null>(null);
  const sidebarInfoTooltipAnimateInTimerRef = useRef<number | null>(null);
  const topChromeGestureTimerRef = useRef<number | null>(null);
  const historyMutationVersionRef = useRef(0);
  const historyLoadSequenceRef = useRef(0);
  const settingsLoadSequenceRef = useRef(0);

  const resolveHistoryAccountId = useCallback(() => {
    const activeAccountId = communityAccount?.id?.trim();
    if (activeAccountId) {
      return activeAccountId;
    }
    return readCommunityAuthSession()?.account?.id?.trim() || null;
  }, [communityAccount?.id]);

  const setHistorySnapshot = useCallback((next: HistoryItem[], options?: { accountId?: string | null }) => {
    const scopedAccountId = options?.accountId !== undefined ? options.accountId : resolveHistoryAccountId();
    historyRef.current = next;
    setHistory(next);
    writeScopedHistorySnapshot(scopedAccountId, JSON.stringify(next));
  }, [resolveHistoryAccountId]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const scopedAccountId = resolveHistoryAccountId();
    const base = parseMaterialHistory(readScopedHistorySnapshot(scopedAccountId));
    const legacy = scopedAccountId ? [] : parseLegacyTopicHistory(window.localStorage.getItem(LEGACY_TOPIC_HISTORY_STORAGE_KEY));
    const merged = mergeHistory(base, legacy);
    setHistorySnapshot(merged, { accountId: scopedAccountId });

    if (!scopedAccountId) {
      return;
    }

    let cancelled = false;
    const loadSequence = historyLoadSequenceRef.current + 1;
    historyLoadSequenceRef.current = loadSequence;
    const mutationVersionAtLoadStart = historyMutationVersionRef.current;

    void (async () => {
      try {
        const remoteHistory = await fetchCommunityHistory();
        if (
          cancelled
          || historyLoadSequenceRef.current !== loadSequence
          || historyMutationVersionRef.current !== mutationVersionAtLoadStart
        ) {
          return;
        }
        setHistorySnapshot(remoteHistory, { accountId: scopedAccountId });
      } catch (error) {
        if (cancelled || historyLoadSequenceRef.current !== loadSequence) {
          return;
        }
        if (isSessionExpiredError(error)) {
          clearCommunityAuthSession();
          setCommunityAccount(null);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [resolveHistoryAccountId, setHistorySnapshot]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const scopedAccountId = resolveHistoryAccountId();
    setActiveStudyReelsSettingsScope(scopedAccountId);

    if (!scopedAccountId) {
      return;
    }

    let cancelled = false;
    const loadSequence = settingsLoadSequenceRef.current + 1;
    settingsLoadSequenceRef.current = loadSequence;

    void (async () => {
      try {
        const remoteSettings = await fetchCommunitySettings();
        if (
          cancelled
          || settingsLoadSequenceRef.current !== loadSequence
          || resolveHistoryAccountId() !== scopedAccountId
          || !remoteSettings
        ) {
          return;
        }
        setActiveStudyReelsSettingsScope(scopedAccountId, { settings: remoteSettings });
      } catch (error) {
        if (cancelled || settingsLoadSequenceRef.current !== loadSequence) {
          return;
        }
        if (isSessionExpiredError(error)) {
          clearCommunityAuthSession();
          setCommunityAccount(null);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [resolveHistoryAccountId]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    let cancelled = false;
    const syncAccountFromStorage = () => {
      const stored = readCommunityAuthSession();
      if (!cancelled) {
        setCommunityAccount(stored?.account ?? null);
      }
    };
    const validateStoredAccount = async () => {
      const stored = readCommunityAuthSession();
      if (!stored?.sessionToken) {
        if (!cancelled) {
          setCommunityAccount(null);
        }
        return;
      }
      try {
        const account = await fetchCommunityAccount();
        if (!cancelled) {
          setCommunityAccount(account);
        }
      } catch {
        syncAccountFromStorage();
      }
    };
    const onStorage = (event: StorageEvent) => {
      if (event.storageArea !== window.localStorage) {
        return;
      }
      if (
        event.key
        && event.key !== "studyreels-community-account"
        && event.key !== "studyreels-community-session-token"
      ) {
        return;
      }
      syncAccountFromStorage();
    };
    const onAuthChanged = () => {
      syncAccountFromStorage();
    };
    syncAccountFromStorage();
    void validateStoredAccount();
    window.addEventListener("storage", onStorage);
    window.addEventListener(COMMUNITY_AUTH_CHANGED_EVENT, onAuthChanged);
    return () => {
      cancelled = true;
      window.removeEventListener("storage", onStorage);
      window.removeEventListener(COMMUNITY_AUTH_CHANGED_EVENT, onAuthChanged);
    };
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const savedTab = normalizeSidebarTab(window.sessionStorage.getItem(ACTIVE_SIDEBAR_TAB_SESSION_KEY));
    const savedCommunitySetIdRaw = window.sessionStorage.getItem(ACTIVE_COMMUNITY_SET_ID_SESSION_KEY);
    const savedCommunitySetId = savedCommunitySetIdRaw?.trim() || null;
    const nextTab = savedTab ?? forcedSidebarTab ?? (forcedCommunitySetId ? "community" : "search");
    setActiveSidebarTab(nextTab);
    setActiveCommunitySetId(savedCommunitySetId ?? (!savedTab ? forcedCommunitySetId : null));
    setSidebarTabHydrated(true);
  }, [forcedCommunitySetId, forcedSidebarTab]);

  useEffect(() => {
    if (!sidebarTabHydrated) {
      return;
    }
    if (!hasProcessedInitialSidebarHydrationRef.current) {
      hasProcessedInitialSidebarHydrationRef.current = true;
      return;
    }
    if (!forcedSidebarTab && !forcedCommunitySetId) {
      return;
    }
    if (forcedCommunitySetId) {
      setActiveCommunitySetId(forcedCommunitySetId);
      setActiveSidebarTab(forcedSidebarTab ?? "community");
      return;
    }
    if (forcedSidebarTab) {
      setActiveSidebarTab(forcedSidebarTab);
    }
  }, [forcedCommunitySetId, forcedSidebarTab, sidebarTabHydrated]);

  useEffect(() => {
    if (typeof window === "undefined" || !sidebarTabHydrated) {
      return;
    }
    safeStorageSetItem(window.sessionStorage, ACTIVE_SIDEBAR_TAB_SESSION_KEY, activeSidebarTab);
  }, [activeSidebarTab, sidebarTabHydrated]);

  useEffect(() => {
    if (typeof window === "undefined" || !sidebarTabHydrated) {
      return;
    }
    if (activeCommunitySetId) {
      safeStorageSetItem(window.sessionStorage, ACTIVE_COMMUNITY_SET_ID_SESSION_KEY, activeCommunitySetId);
      return;
    }
    safeStorageRemoveItem(window.sessionStorage, ACTIVE_COMMUNITY_SET_ID_SESSION_KEY);
  }, [activeCommunitySetId, sidebarTabHydrated]);

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
  const selectedHistoryInfoItem = useMemo(
    () => (selectedHistoryInfoId ? history.find((item) => item.materialId === selectedHistoryInfoId) ?? null : null),
    [history, selectedHistoryInfoId],
  );
  const selectedHistoryInfoSections = useMemo(
    () => (selectedHistoryInfoItem ? buildHistoryInfoSections(selectedHistoryInfoItem) : []),
    [selectedHistoryInfoItem],
  );
  const selectedHistoryInfoQuery = useMemo(
    () => (selectedHistoryInfoItem?.feedQuery?.trim() ? selectedHistoryInfoItem.feedQuery.trim() : null),
    [selectedHistoryInfoItem],
  );

  const persistHistory = useCallback((next: HistoryItem[]) => {
    const scopedAccountId = resolveHistoryAccountId();
    historyMutationVersionRef.current += 1;
    setHistorySnapshot(next, { accountId: scopedAccountId });
    if (!scopedAccountId) {
      return;
    }
    void queueCommunityHistorySync(next).catch((error) => {
      if (isSessionExpiredError(error)) {
        clearCommunityAuthSession();
        setCommunityAccount(null);
      }
    });
  }, [resolveHistoryAccountId, setHistorySnapshot]);

  const syncSavedSettings = useCallback((settings: StudyReelsSettings) => {
    const scopedAccountId = resolveHistoryAccountId();
    if (!scopedAccountId) {
      return;
    }
    void queueCommunitySettingsSync(settings).catch((error) => {
      if (isSessionExpiredError(error)) {
        clearCommunityAuthSession();
        setCommunityAccount(null);
      }
    });
  }, [resolveHistoryAccountId]);

  const upsertHistory = useCallback(
    (entry: {
      materialId: string;
      title: string;
      updatedAt: number;
      starred?: boolean;
      generationMode?: GenerationMode;
      source?: HistorySource;
      feedQuery?: string;
      activeIndex?: number;
      activeReelId?: string;
    }) => {
      const existing = historyRef.current.find((item) => item.materialId === entry.materialId);
      const merged: HistoryItem = {
        materialId: entry.materialId,
        title: entry.title,
        updatedAt: entry.updatedAt,
        starred: entry.starred ?? existing?.starred ?? false,
        generationMode: entry.generationMode ?? existing?.generationMode ?? "fast",
        source: entry.source ?? existing?.source ?? "search",
        feedQuery: entry.feedQuery ?? existing?.feedQuery,
        activeIndex: entry.activeIndex ?? existing?.activeIndex,
        activeReelId: entry.activeReelId ?? existing?.activeReelId,
      };
      const next = [merged, ...historyRef.current.filter((item) => item.materialId !== merged.materialId)].slice(0, MAX_HISTORY_ITEMS);
      persistHistory(next);
    },
    [persistHistory],
  );

  const closeHistoryMenus = useCallback(() => {
    setActiveSidebarHistoryMenuId(null);
    setActiveHistoryPageMenuId(null);
  }, []);

  const clearAllHistory = useCallback(() => {
    const scopedAccountId = resolveHistoryAccountId();
    historyMutationVersionRef.current += 1;
    historyRef.current = [];
    setHistory([]);
    setHistoryQuery("");
    setError(null);
    closeHistoryMenus();
    setSelectedHistoryInfoId(null);
    if (typeof window !== "undefined") {
      if (scopedAccountId) {
        writeScopedHistorySnapshot(scopedAccountId, JSON.stringify([]));
        void queueCommunityHistorySync([]).catch((error) => {
          if (isSessionExpiredError(error)) {
            clearCommunityAuthSession();
            setCommunityAccount(null);
          }
        });
      } else {
        writeScopedHistorySnapshot(scopedAccountId, null);
      }
      if (!scopedAccountId) {
        window.localStorage.removeItem(LEGACY_TOPIC_HISTORY_STORAGE_KEY);
      }
      window.localStorage.removeItem(MATERIAL_SEEDS_STORAGE_KEY);
      window.localStorage.removeItem(MATERIAL_GROUPS_STORAGE_KEY);
      window.localStorage.removeItem(FEED_PROGRESS_STORAGE_KEY);
      window.localStorage.removeItem(FEED_SESSION_STORAGE_KEY);
    }
  }, [closeHistoryMenus, resolveHistoryAccountId]);

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
    if (activeSidebarTab === "search" && !searchPanelScrollable) {
      return;
    }
    clearTopChromeGestureTimer();
    setTopChromeGestureActive(true);
    topChromeGestureTimerRef.current = window.setTimeout(() => {
      setTopChromeGestureActive(false);
      topChromeGestureTimerRef.current = null;
    }, TOP_CHROME_GESTURE_WINDOW_MS);
  }, [activeSidebarTab, clearTopChromeGestureTimer, searchPanelScrollable]);

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
      const target = event.currentTarget;
      if (shouldDisableSidebarTooltips()) {
        clearSidebarInfoTooltipTimers();
        delete target.dataset.tooltipHoverLocked;
        setSidebarInfoTooltip(null);
        return;
      }
      if (target.dataset.tooltipHoverLocked === "true") {
        return;
      }
      target.dataset.tooltipHoverLocked = "true";
      const rect = target.getBoundingClientRect();
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

  const onSidebarInfoHoverEnd = useCallback((event: ReactMouseEvent<HTMLElement>) => {
    delete event.currentTarget.dataset.tooltipHoverLocked;
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
    setAccountMenuOpen(false);
    setMobileSidebarClosing(true);
    mobileSidebarCloseTimerRef.current = window.setTimeout(() => {
      setMobileSidebarOpen(false);
      setMobileSidebarClosing(false);
      mobileSidebarCloseTimerRef.current = null;
    }, MOBILE_SIDEBAR_CLOSE_MS);
  }, [clearMobileSidebarCloseTimer, mobileSidebarClosing, mobileSidebarOpen]);

  const forceCloseMobileSidebar = useCallback(() => {
    clearMobileSidebarCloseTimer();
    setAccountMenuOpen(false);
    setMobileSidebarClosing(false);
    setMobileSidebarOpen(false);
  }, [clearMobileSidebarCloseTimer]);

  const clearCommunitySelectionFromUrl = useCallback(() => {
    if (typeof window === "undefined") {
      return;
    }
    try {
      const nextUrl = new URL(window.location.href);
      const keys = Array.from(nextUrl.searchParams.keys());
      for (const key of keys) {
        if (key.startsWith("community_")) {
          nextUrl.searchParams.delete(key);
        }
      }
      const nextSearch = nextUrl.searchParams.toString();
      const nextPath = nextSearch ? `${nextUrl.pathname}?${nextSearch}` : nextUrl.pathname;
      window.history.replaceState(window.history.state, "", `${nextPath}${nextUrl.hash}`);
    } catch {
      // Ignore URL rewrite failures.
    }
  }, []);

  const applySidebarSwitchIntent = useCallback((intent: SidebarSwitchIntent) => {
    setActiveSidebarTab(intent.tab);
    if (intent.tab === "community" && intent.resetCommunityView && activeSidebarTab === "community") {
      setCommunityResetSignal((prev) => prev + 1);
      clearCommunitySelectionFromUrl();
    }
    setError(null);
    closeHistoryMenus();
    setSelectedHistoryInfoId(null);
    setAccountMenuOpen(false);
    if (intent.clearHistoryQuery) {
      setHistoryQuery("");
    }
    forceCloseMobileSidebar();
  }, [activeSidebarTab, clearCommunitySelectionFromUrl, closeHistoryMenus, forceCloseMobileSidebar]);

  const requestSidebarSwitch = useCallback(
    (intent: SidebarSwitchIntent) => {
      const hasPendingUnsavedSettings = hasUnsavedSettingsChanges || Boolean(settingsPanelRef.current?.hasUnsavedChanges());
      if (activeSidebarTab === "settings" && intent.tab !== "settings" && hasPendingUnsavedSettings) {
        setPendingSidebarSwitchIntent(intent);
        setSettingsModalView("unsaved");
        return;
      }
      const leavingCommunityDraftForm = (activeSidebarTab === "create" || activeSidebarTab === "edit")
        && intent.tab !== activeSidebarTab
        && hasUnsavedCommunityDraftChanges;
      if (leavingCommunityDraftForm) {
        setPendingCommunityDraftSwitchIntent(intent);
        setShowUnsavedCommunityDraftModal(true);
        return;
      }
      applySidebarSwitchIntent(intent);
    },
    [activeSidebarTab, applySidebarSwitchIntent, hasUnsavedCommunityDraftChanges, hasUnsavedSettingsChanges],
  );

  const startNewSearch = useCallback(() => {
    requestSidebarSwitch({ tab: "search", clearHistoryQuery: true });
  }, [requestSidebarSwitch]);

  const openAccountScreen = useCallback(() => {
    closeHistoryMenus();
    setSelectedHistoryInfoId(null);
    setAccountMenuOpen(false);
    forceCloseMobileSidebar();
    const returnTab = activeSidebarTab === "create" ? "edit" : activeSidebarTab;
    router.push(`/account?return_tab=${returnTab}`);
  }, [activeSidebarTab, closeHistoryMenus, forceCloseMobileSidebar, router]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const hasPendingUnsavedSettings = activeSidebarTab === "settings"
      && (hasUnsavedSettingsChanges || Boolean(settingsPanelRef.current?.hasUnsavedChanges()));
    const hasPendingUnsavedCommunityDraft = (activeSidebarTab === "create" || activeSidebarTab === "edit")
      && hasUnsavedCommunityDraftChanges;
    if (!hasPendingUnsavedSettings && !hasPendingUnsavedCommunityDraft) {
      return;
    }
    const onBeforeUnload = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", onBeforeUnload);
    return () => {
      window.removeEventListener("beforeunload", onBeforeUnload);
    };
  }, [activeSidebarTab, hasUnsavedCommunityDraftChanges, hasUnsavedSettingsChanges]);

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

  const onSearchPanelScrollabilityChange = useCallback((isScrollable: boolean) => {
    setSearchPanelScrollable((prev) => (prev === isScrollable ? prev : isScrollable));
    if (!isScrollable) {
      setTopChromeOffset(false);
      clearTopChromeGestureTimer();
      setTopChromeGestureActive(false);
    }
  }, [clearTopChromeGestureTimer]);

  useEffect(() => {
    if (activeSidebarTab === "search") {
      setTopChromeOffset(false);
      setTopChromeGestureActive(false);
    }
  }, [activeSidebarTab]);

  useEffect(() => {
    if (activeSidebarTab !== "community") {
      return;
    }
    setCommunityResetSignal((prev) => prev + 1);
    if (!forcedCommunitySetId) {
      clearCommunitySelectionFromUrl();
    }
  }, [activeSidebarTab, clearCommunitySelectionFromUrl, forcedCommunitySetId]);

  const visibleSidebarTab = sidebarTabHydrated ? activeSidebarTab : null;
  const shouldShowTopChromeStrip = sidebarTabHydrated
    && !mobileSidebarOpen
    && (
      visibleSidebarTab === "settings"
        ? true
        : visibleSidebarTab === "search"
        ? searchPanelScrollable && (topChromeOffset || topChromeGestureActive)
        : topChromeOffset
    );
  // Keep settings content position stable even when the mobile sidebar temporarily hides the top chrome.
  const shouldInsetSettingsForTopChrome = visibleSidebarTab === "settings";

  const openMaterialFeed = useCallback(
    (materialId: string) => {
      const existing = historyRef.current.find((item) => item.materialId === materialId);
      if (existing) {
        upsertHistory({ ...existing, updatedAt: Date.now() });
      }
      closeHistoryMenus();
      setSelectedHistoryInfoId(null);
      forceCloseMobileSidebar();
      const savedFeedQuery = (existing?.feedQuery || "").trim();
      if (savedFeedQuery) {
        router.push(`/feed?${savedFeedQuery}`);
        return;
      }
      if (existing?.source === "community") {
        const fallbackSetId = parseCommunitySetIdFromHistoryMaterialId(materialId);
        if (fallbackSetId) {
          const fallbackParams = new URLSearchParams({
            tab: "community",
            community_set_id: fallbackSetId,
          });
          router.push(`/?${fallbackParams.toString()}`);
          return;
        }
      }
      const mode = existing?.generationMode ?? "fast";
      const returnTab = activeSidebarTab === "history"
        ? "history"
        : activeSidebarTab === "community" || activeSidebarTab === "create" || activeSidebarTab === "edit"
          ? "community"
          : "search";
      const nextParams = new URLSearchParams({
        material_id: materialId,
        generation_mode: mode,
        return_tab: returnTab,
      });
      router.push(`/feed?${nextParams.toString()}`);
    },
    [activeSidebarTab, closeHistoryMenus, forceCloseMobileSidebar, router, upsertHistory],
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
      closeHistoryMenus();
    },
    [closeHistoryMenus, upsertHistory],
  );

  const deleteHistoryItem = useCallback(
    (materialId: string) => {
      const next = historyRef.current.filter((item) => item.materialId !== materialId);
      persistHistory(next);
      closeHistoryMenus();
      setSelectedHistoryInfoId((prev) => (prev === materialId ? null : prev));
    },
    [closeHistoryMenus, persistHistory],
  );

  useEffect(() => {
    if (!selectedHistoryInfoId) {
      return;
    }
    if (!selectedHistoryInfoItem) {
      setSelectedHistoryInfoId(null);
    }
  }, [selectedHistoryInfoId, selectedHistoryInfoItem]);

  useEffect(() => {
    if (!activeSidebarHistoryMenuId && !activeHistoryPageMenuId) {
      return;
    }
    const onPointerDown = (event: PointerEvent) => {
      if (!(event.target instanceof Element)) {
        return;
      }
      if (event.target.closest("[data-history-actions='true']")) {
        return;
      }
      closeHistoryMenus();
    };
    window.addEventListener("pointerdown", onPointerDown);
    return () => {
      window.removeEventListener("pointerdown", onPointerDown);
    };
  }, [activeHistoryPageMenuId, activeSidebarHistoryMenuId, closeHistoryMenus]);

  useEffect(() => {
    if (!selectedHistoryInfoItem) {
      return;
    }
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setSelectedHistoryInfoId(null);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [selectedHistoryInfoItem]);

  useEffect(() => {
    if (!accountMenuOpen) {
      return;
    }
    const onPointerDown = (event: PointerEvent) => {
      if (!(event.target instanceof Element)) {
        return;
      }
      if (event.target.closest("[data-account-actions='true']")) {
        return;
      }
      setAccountMenuOpen(false);
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setAccountMenuOpen(false);
      }
    };
    window.addEventListener("pointerdown", onPointerDown);
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("pointerdown", onPointerDown);
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [accountMenuOpen]);

  const onUploadMaterialCreated = useCallback(
    async (params: { materialId: string; title: string; topic?: string; generationMode: GenerationMode; feedQuery: string }) => {
      const nextTitle = params.title?.trim() || params.topic?.trim() || "New Study Session";
      upsertHistory({
        materialId: params.materialId,
        title: nextTitle,
        updatedAt: Date.now(),
        generationMode: params.generationMode,
        source: "search",
        feedQuery: params.feedQuery.trim() || undefined,
      });
    },
    [upsertHistory],
  );

  const onCommunityReelFeedOpened = useCallback(
    (payload: { setId: string; setTitle: string; selectedReelId: string; feedQuery: string }) => {
      const normalizedSetId = payload.setId.trim();
      if (!normalizedSetId) {
        return;
      }
      const title = payload.setTitle.trim() || "Community Reel Set";
      upsertHistory({
        materialId: `community:${encodeURIComponent(normalizedSetId)}`,
        title,
        updatedAt: Date.now(),
        generationMode: "fast",
        source: "community",
        feedQuery: payload.feedQuery.trim() || undefined,
        activeReelId: payload.selectedReelId.trim() || undefined,
      });
    },
    [upsertHistory],
  );

  const switchSidebarTab = useCallback(
    (tab: SidebarTab, options?: { resetCommunityView?: boolean }) => {
      requestSidebarSwitch({ tab, resetCommunityView: options?.resetCommunityView });
    },
    [requestSidebarSwitch],
  );
  const openYourSetsFromAccountMenu = useCallback(() => {
    setAccountMenuOpen(false);
    switchSidebarTab("edit");
  }, [switchSidebarTab]);
  const onAccountMenuSignOut = useCallback(async () => {
    setAccountMenuOpen(false);
    try {
      await logoutCommunityAccount();
    } finally {
      setCommunityAccount(null);
    }
  }, []);
  const onCommunityDraftExitActionsChange = useCallback((actions: CommunityDraftExitActions | null) => {
    // Unsaved draft state is owned by the panel's explicit dirty-state callback.
    // Clearing it here would incorrectly mark the form clean during effect cleanups.
    communityDraftExitActionsRef.current = actions;
  }, []);
  const closeSettingsModal = useCallback((source: "close-button" | "backdrop" = "close-button") => {
    if (settingsModalView === "availability") {
      settingsPanelRef.current?.dismissAvailabilityModal(source);
      return;
    }
    setSettingsModalView(null);
    setPendingSidebarSwitchIntent(null);
    setPendingSaveSwitchUntilHeuristicClose(false);
  }, [settingsModalView]);
  const closeUnsavedCommunityDraftModal = useCallback(() => {
    setShowUnsavedCommunityDraftModal(false);
    setPendingCommunityDraftSwitchIntent(null);
  }, []);
  const discardSettingsAndContinue = useCallback(() => {
    const intent = pendingSidebarSwitchIntent;
    if (!intent) {
      closeSettingsModal();
      return;
    }
    settingsPanelRef.current?.discardUnsavedChanges();
    setHasUnsavedSettingsChanges(false);
    setSettingsModalView(null);
    setPendingSidebarSwitchIntent(null);
    setPendingSaveSwitchUntilHeuristicClose(false);
    applySidebarSwitchIntent(intent);
  }, [applySidebarSwitchIntent, closeSettingsModal, pendingSidebarSwitchIntent]);
  const saveSettingsAndContinue = useCallback(() => {
    const intent = pendingSidebarSwitchIntent;
    if (!intent) {
      closeSettingsModal();
      return;
    }
    const settingsPanel = settingsPanelRef.current;
    if (!settingsPanel) {
      setHasUnsavedSettingsChanges(false);
      setSettingsModalView(null);
      setPendingSidebarSwitchIntent(null);
      setPendingSaveSwitchUntilHeuristicClose(false);
      applySidebarSwitchIntent(intent);
      return;
    }
    setSettingsModalView("availability");
    setPendingSaveSwitchUntilHeuristicClose(true);
    settingsPanel.savePreferences();
    setHasUnsavedSettingsChanges(false);
  }, [applySidebarSwitchIntent, closeSettingsModal, pendingSidebarSwitchIntent]);
  const discardCommunityDraftAndContinue = useCallback(() => {
    const intent = pendingCommunityDraftSwitchIntent;
    if (!intent) {
      closeUnsavedCommunityDraftModal();
      return;
    }
    communityDraftExitActionsRef.current?.discardDraftChanges();
    setHasUnsavedCommunityDraftChanges(false);
    setShowUnsavedCommunityDraftModal(false);
    setPendingCommunityDraftSwitchIntent(null);
    applySidebarSwitchIntent(intent);
  }, [applySidebarSwitchIntent, closeUnsavedCommunityDraftModal, pendingCommunityDraftSwitchIntent]);
  const saveCommunityDraftAndContinue = useCallback(() => {
    const intent = pendingCommunityDraftSwitchIntent;
    if (!intent) {
      closeUnsavedCommunityDraftModal();
      return;
    }
    setShowUnsavedCommunityDraftModal(false);
    setPendingCommunityDraftSwitchIntent(null);
    const didSave = communityDraftExitActionsRef.current?.saveDraftProgress() ?? true;
    if (!didSave) {
      return;
    }
    setHasUnsavedCommunityDraftChanges(false);
    applySidebarSwitchIntent(intent);
  }, [applySidebarSwitchIntent, closeUnsavedCommunityDraftModal, pendingCommunityDraftSwitchIntent]);
  const onSettingsAvailabilityModalClose = useCallback(
    (source: "close-button" | "backdrop") => {
      setSettingsModalView(null);
      if (!pendingSaveSwitchUntilHeuristicClose) {
        return;
      }
      const intent = pendingSidebarSwitchIntent;
      setPendingSaveSwitchUntilHeuristicClose(false);
      setPendingSidebarSwitchIntent(null);
      if (source !== "close-button" || !intent) {
        return;
      }
      applySidebarSwitchIntent(intent);
    },
    [applySidebarSwitchIntent, pendingSaveSwitchUntilHeuristicClose, pendingSidebarSwitchIntent],
  );
  const isCommunityPanel = visibleSidebarTab === "community" || visibleSidebarTab === "create" || visibleSidebarTab === "edit";
  const usesCommunityShell = isCommunityPanel || visibleSidebarTab === "history";
  const hasCommunityBackdrop = usesCommunityShell || visibleSidebarTab === "settings";
  const hideMobileTopControls = isCommunityPanel && communityDetailOpen;
  const [lastCommunityPanelMode, setLastCommunityPanelMode] = useState<"community" | "create" | "edit">("community");

  useEffect(() => {
    if (activeSidebarTab === "community" || activeSidebarTab === "create" || activeSidebarTab === "edit") {
      setLastCommunityPanelMode(activeSidebarTab);
    }
  }, [activeSidebarTab]);

  const communityPanelMode = visibleSidebarTab === "search"
    ? lastCommunityPanelMode
    : visibleSidebarTab === "create"
      ? "create"
      : visibleSidebarTab === "edit"
        ? "edit"
        : "community";
  const activeSettingsAvailabilityState = settingsAvailabilityModalSnapshot?.state ?? DEFAULT_SETTINGS_AVAILABILITY_STATE;
  const showLoadingScreen = useLoadingScreenGate(sidebarTabHydrated, { minimumVisibleMs: 2000 });

  if (showLoadingScreen) {
    return <FullscreenLoadingScreen />;
  }

  const historyPanelContent = (
    <div className={visibleSidebarTab === "history" ? "flex h-full min-h-0 flex-col" : "hidden h-full min-h-0"}>
      <div className="flex h-full min-h-0 flex-col px-5 pb-6 pt-[calc(max(env(safe-area-inset-top),0px)+5rem)] sm:px-6 md:px-8 lg:px-10 lg:pt-12">
        <div className="shrink-0 text-center">
          <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-white/52">Library</p>
          <h2 className="mt-3 text-3xl font-semibold tracking-tight text-white sm:text-[2.5rem]">Search History</h2>
          <p className="mt-3 text-sm text-white/58">Search your saved sessions and reopen them from here.</p>
        </div>

        <div className="mx-auto mt-8 w-full max-w-2xl shrink-0">
          <div className="relative">
            <i className="fa-solid fa-magnifying-glass pointer-events-none absolute left-4 top-1/2 -translate-y-1/2 text-sm text-white/38" aria-hidden="true" />
            <input
              value={historyQuery}
              onChange={(event) => setHistoryQuery(event.target.value)}
              placeholder="Search history..."
              className="h-14 w-full rounded-[1.75rem] bg-white/[0.08] pl-12 pr-4 text-base text-white outline-none backdrop-blur-[18px] backdrop-saturate-150 transition-colors duration-200 placeholder:text-white/35 focus:bg-white/[0.12]"
            />
          </div>

          <div className="mt-3 flex items-center justify-between gap-3 text-xs text-white/52">
            <p>
              {historyQuery.trim()
                ? `${filteredHistory.length} ${filteredHistory.length === 1 ? "result" : "results"}`
                : `${historySorted.length} saved ${historySorted.length === 1 ? "session" : "sessions"}`}
            </p>
            {historySorted.length > 0 ? (
              <button
                type="button"
                onClick={clearAllHistory}
                aria-label="Clear history"
                className="grid h-7 w-7 place-items-center rounded-full text-white/48 transition hover:bg-white/10 hover:text-white"
              >
                <i className="fa-solid fa-trash-can text-[11px]" aria-hidden="true" />
              </button>
            ) : null}
          </div>
        </div>

        <div className="mx-auto mt-8 min-h-0 w-full max-w-4xl flex-1 overflow-y-auto pb-2">
          {historySorted.length === 0 ? (
            <div className="grid min-h-full place-items-center px-4 py-12">
              <div className="max-w-md text-center">
                <p className="text-lg font-semibold text-white">No search history yet.</p>
                <p className="mt-3 text-sm leading-6 text-white/58">
                  Start a search or open a community reel set, and your recent sessions will show up here.
                </p>
              </div>
            </div>
          ) : filteredHistory.length === 0 ? (
            <div className="grid min-h-full place-items-center px-4 py-12">
              <div className="max-w-md text-center">
                <p className="text-lg font-semibold text-white">No matching history items.</p>
                <p className="mt-3 text-sm leading-6 text-white/58">Try a different title or clear the search box.</p>
              </div>
            </div>
          ) : (
            <div className="space-y-3 pb-4">
              {filteredHistory.map((entry) => (
                <div
                  key={`history-${entry.materialId}`}
                  className={`group relative ${activeHistoryPageMenuId === entry.materialId ? "z-40" : "z-0"}`}
                >
                  <button
                    type="button"
                    onClick={() => openMaterialFeed(entry.materialId)}
                    className="w-full rounded-[1.75rem] bg-white/[0.07] px-4 py-4 pr-14 text-left backdrop-blur-[18px] backdrop-saturate-150 transition-colors duration-200 hover:bg-white/[0.11] sm:px-5 sm:py-5"
                  >
                    <div className="flex items-start gap-3">
                      <span className="mt-0.5 grid h-10 w-10 shrink-0 place-items-center rounded-2xl bg-black/30 text-white/78 transition-colors duration-200 group-hover:text-white">
                        <i className={`fa-solid ${entry.source === "community" ? "fa-users" : "fa-magnifying-glass"} text-sm`} aria-hidden="true" />
                      </span>
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2">
                          {entry.starred ? <i className="fa-solid fa-star text-[11px] text-white/72 transition-colors group-hover:text-white" aria-hidden="true" /> : null}
                          <p className="truncate text-sm font-semibold text-white sm:text-base">{entry.title}</p>
                          <span
                            className={`shrink-0 rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.08em] ${
                              entry.source === "community"
                                ? "bg-white/16 text-white/88"
                                : "bg-black/45 text-white/62"
                            }`}
                          >
                            {entry.source === "community" ? "Community" : entry.generationMode}
                          </span>
                        </div>
                        <p className="mt-2 text-xs text-white/48">{formatHistoryInfoDate(entry.updatedAt)}</p>
                      </div>
                    </div>
                  </button>

                  <div data-history-actions="true" className="absolute right-3 top-3 z-30">
                    <button
                      type="button"
                      aria-label="History item actions"
                      onClick={(event) => {
                        event.stopPropagation();
                        setActiveSidebarHistoryMenuId(null);
                        setActiveHistoryPageMenuId((prev) => (prev === entry.materialId ? null : entry.materialId));
                      }}
                      data-force-visible={activeHistoryPageMenuId === entry.materialId ? "true" : undefined}
                      className="grid h-8 w-8 place-items-center rounded-full text-white/62 transition hover:bg-white/10 hover:text-white"
                    >
                      <i className="fa-solid fa-ellipsis text-[12px]" aria-hidden="true" />
                    </button>

                    <div
                      className={`absolute right-0 top-full z-50 mt-2 w-44 transition-opacity duration-180 ${
                        activeHistoryPageMenuId === entry.materialId
                          ? "pointer-events-auto opacity-100"
                          : "pointer-events-none opacity-0"
                      }`}
                    >
                      <div
                        role="menu"
                        className="overflow-hidden rounded-2xl bg-black p-1.5 shadow-[0_20px_48px_rgba(0,0,0,0.45)]"
                      >
                        <button
                          type="button"
                          onClick={(event) => {
                            event.stopPropagation();
                            setSelectedHistoryInfoId(entry.materialId);
                            closeHistoryMenus();
                          }}
                          className="flex w-full items-center gap-2.5 rounded-xl px-3 py-2 text-left text-xs text-white/90 transition hover:bg-white/10"
                        >
                          <i className="fa-regular fa-circle-info text-[11px] text-white/80" aria-hidden="true" />
                          More information
                        </button>
                        <button
                          type="button"
                          onClick={(event) => {
                            event.stopPropagation();
                            toggleHistoryStar(entry.materialId);
                          }}
                          className="mt-1 flex w-full items-center gap-2.5 rounded-xl px-3 py-2 text-left text-xs text-white/90 transition hover:bg-white/10"
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
                          className="mt-1 flex w-full items-center gap-2.5 rounded-xl px-3 py-2 text-left text-xs text-white/90 transition hover:bg-white/10"
                        >
                          <i className="fa-regular fa-trash-can text-[11px] text-white/80" aria-hidden="true" />
                          Delete
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );

  const sidebarPanelContent = (
    <div className="flex h-full min-h-0 flex-col lg:pt-2">
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
            className="grid h-8 w-8 place-items-center rounded-xl bg-transparent text-sm font-semibold text-white/90 transition-colors duration-200 hover:bg-white/10 hover:text-white"
          >
            <span className="translate-x-[0.5px] -translate-y-[0.5px] leading-none">+</span>
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
          className={`h-9 w-full rounded-xl bg-transparent px-2.5 text-left text-xs transition-colors duration-200 ${
            visibleSidebarTab === "search" ? "bg-white text-black" : "text-white/85 hover:bg-white/10 hover:text-white"
          }`}
        >
          <div className="flex h-full items-center justify-between gap-1.5">
            <p className="truncate font-semibold leading-none">Search</p>
            <i
              className={`fa-solid fa-magnifying-glass text-[11px] ${
                visibleSidebarTab === "search" ? "text-black/80" : "text-white/74 transition-colors duration-200 group-hover:text-white"
              }`}
              aria-hidden="true"
            />
          </div>
        </button>
      </div>

      <div
        className="group relative mt-2"
        onMouseEnter={(event) => onSidebarInfoHoverStart(event, "Browse your saved search sessions")}
        onMouseLeave={onSidebarInfoHoverEnd}
      >
        <button
          type="button"
          onClick={() => switchSidebarTab("history")}
          className={`h-9 w-full rounded-xl bg-transparent px-2.5 text-left text-xs transition-colors duration-200 ${
            visibleSidebarTab === "history" ? "bg-white text-black" : "text-white/85 hover:bg-white/10 hover:text-white"
          }`}
        >
          <div className="flex h-full items-center justify-between gap-1.5">
            <p className="truncate font-semibold leading-none">Search History</p>
            <i
              className={`fa-solid fa-clock-rotate-left text-[11px] ${
                visibleSidebarTab === "history" ? "text-black/80" : "text-white/74 transition-colors duration-200 group-hover:text-white"
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
          onClick={() => switchSidebarTab("community", { resetCommunityView: true })}
          className={`h-9 w-full rounded-xl bg-transparent px-2.5 text-left text-xs transition-colors duration-200 ${
            visibleSidebarTab === "community"
              ? "bg-white text-black"
              : "text-white/85 hover:bg-white/10 hover:text-white"
          }`}
        >
          <div className="flex h-full items-center justify-between gap-1.5">
            <p className="truncate font-semibold leading-none">Community Reels</p>
            <i
              className={`fa-solid fa-users text-[11px] ${
                visibleSidebarTab === "community" ? "text-black/80" : "text-white/74 transition-colors duration-200 group-hover:text-white"
              }`}
              aria-hidden="true"
            />
          </div>
        </button>
      </div>

      <div
        className="group relative mt-2"
        onMouseEnter={(event) => onSidebarInfoHoverStart(event, "Create and edit your community sets")}
        onMouseLeave={onSidebarInfoHoverEnd}
      >
        <button
          type="button"
          onClick={() => switchSidebarTab("edit")}
          className={`h-9 w-full rounded-xl bg-transparent px-2.5 text-left text-xs transition-colors duration-200 ${
            visibleSidebarTab === "edit" || visibleSidebarTab === "create"
              ? "bg-white text-black"
              : "text-white/85 hover:bg-white/10 hover:text-white"
          }`}
        >
          <div className="flex h-full items-center justify-between gap-1.5">
            <p className="truncate font-semibold leading-none">Your Sets</p>
            <i
              className={`fa-solid fa-pen-to-square text-[11px] ${
                visibleSidebarTab === "edit" || visibleSidebarTab === "create"
                  ? "text-black/80"
                  : "text-white/74 transition-colors duration-200 group-hover:text-white"
              }`}
              aria-hidden="true"
            />
          </div>
        </button>
      </div>

      <div className="mt-4 min-h-0 flex-1 overflow-y-auto px-2.5 lg:flex-[0.95]">
        <div className="mb-2 flex items-center justify-between">
          <p className="text-[10px] font-semibold uppercase tracking-[0.1em] text-white/60">History</p>
          <button
            type="button"
            onClick={clearAllHistory}
            aria-label="Clear history"
            className="grid h-7 w-7 translate-x-1.5 place-items-center rounded-full text-white/60 transition hover:bg-white/10 hover:text-white"
          >
            <i className="fa-solid fa-trash-can text-[11px]" aria-hidden="true" />
          </button>
        </div>
        <div className="relative pb-1">
          <div className="relative space-y-1.5">
            {historySorted.length === 0 ? (
              <p className="text-xs text-zinc-400">No history yet.</p>
            ) : (
              historySorted.map((entry) => (
                <div
                  key={`history-${entry.materialId}`}
                  className="group relative"
                >
                  <button
                    type="button"
                    onClick={() => openMaterialFeed(entry.materialId)}
                    className="h-9 w-full rounded-xl bg-transparent px-2.5 pr-10 text-left text-xs text-white/85 transition-colors duration-200 hover:bg-white/10 hover:text-white"
                  >
                    <div className="flex items-center gap-1.5">
                      {entry.starred ? <i className="fa-solid fa-star text-[10px] text-white/75 transition-colors group-hover:text-white" aria-hidden="true" /> : null}
                      <p className="truncate font-semibold leading-none transition-colors group-hover:text-white">{entry.title}</p>
                      {entry.source === "community" ? (
                        <span className="shrink-0 rounded-md bg-white/16 px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-[0.08em] text-white/92 transition-colors group-hover:bg-white/22 group-hover:text-white">
                          Community
                        </span>
                      ) : (
                        <span className="shrink-0 rounded-md bg-black/55 px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-[0.08em] text-white/68 transition-colors group-hover:text-white">
                          {entry.generationMode}
                        </span>
                      )}
                    </div>
                  </button>

                  <div data-history-actions="true" className="absolute right-1.5 top-1.5 z-20">
                    <button
                      type="button"
                      aria-label="History item actions"
                      onClick={(event) => {
                        event.stopPropagation();
                        setActiveHistoryPageMenuId(null);
                        setActiveSidebarHistoryMenuId((prev) => (prev === entry.materialId ? null : entry.materialId));
                      }}
                      data-force-visible={activeSidebarHistoryMenuId === entry.materialId ? "true" : undefined}
                      className="reveal-on-desktop-hover grid h-6 w-6 place-items-center rounded-md text-white/70 transition hover:bg-white/10 hover:text-white"
                    >
                      <i className="fa-solid fa-ellipsis text-[11px]" aria-hidden="true" />
                    </button>

                    <div
                      className={`absolute right-0 top-full z-30 mt-1 w-44 transition-opacity duration-180 ${
                        activeSidebarHistoryMenuId === entry.materialId
                          ? "pointer-events-auto opacity-100"
                          : "pointer-events-none opacity-0"
                      }`}
                    >
                      <div
                        role="menu"
                        className="overflow-hidden rounded-2xl bg-black p-1.5 shadow-[0_20px_48px_rgba(0,0,0,0.45)]"
                      >
                        <button
                          type="button"
                          onClick={(event) => {
                            event.stopPropagation();
                            setSelectedHistoryInfoId(entry.materialId);
                            closeHistoryMenus();
                          }}
                          className="flex w-full items-center gap-2.5 rounded-xl px-3 py-2 text-left text-xs text-white/90 transition hover:bg-white/10"
                        >
                          <i className="fa-regular fa-circle-info text-[11px] text-white/80" aria-hidden="true" />
                          More information
                        </button>
                        <button
                          type="button"
                          onClick={(event) => {
                            event.stopPropagation();
                            toggleHistoryStar(entry.materialId);
                          }}
                          className="mt-1 flex w-full items-center gap-2.5 rounded-xl px-3 py-2 text-left text-xs text-white/90 transition hover:bg-white/10"
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
                          className="mt-1 flex w-full items-center gap-2.5 rounded-xl px-3 py-2 text-left text-xs text-white/90 transition hover:bg-white/10"
                        >
                          <i className="fa-regular fa-trash-can text-[11px] text-white/80" aria-hidden="true" />
                          Delete
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      <div className="mb-0 mt-auto pt-4 lg:pt-0">
        <div className="group relative">
          <button
            type="button"
            onClick={() => switchSidebarTab("settings")}
            className={`h-9 w-full rounded-xl bg-transparent px-2.5 text-left text-xs transition-colors duration-200 ${
              visibleSidebarTab === "settings"
                ? "bg-white text-black"
                : "text-white/85 hover:bg-white/10 hover:text-white"
            }`}
          >
            <div className="flex h-full items-center justify-between gap-1.5">
              <p className="truncate font-semibold leading-none">Settings</p>
              <i
                className={`fa-solid fa-gear text-[11px] ${
                  visibleSidebarTab === "settings" ? "text-black/80" : "text-white/74 transition-colors duration-200 group-hover:text-white"
                }`}
                aria-hidden="true"
              />
            </div>
          </button>
        </div>
        <div
          data-account-actions="true"
          className="group relative mt-2"
        >
          {communityAccount ? (
            <div
              className={`absolute inset-x-0 bottom-full z-30 mb-2 ${
                accountMenuOpen ? "pointer-events-auto" : "pointer-events-none"
              }`}
            >
              <div
                role="menu"
                className={`overflow-hidden rounded-2xl bg-black p-1.5 shadow-[0_20px_48px_rgba(0,0,0,0.45)] transition duration-180 ${
                  accountMenuOpen ? "pointer-events-auto translate-y-0 opacity-100" : "pointer-events-none translate-y-1 opacity-0"
                }`}
              >
                <button
                  type="button"
                  onClick={openAccountScreen}
                  className="flex w-full items-center gap-2.5 rounded-xl px-3 py-2 text-left text-xs text-white/90 transition hover:bg-white/10"
                >
                  <i className="fa-regular fa-id-badge text-[11px] text-white/75" aria-hidden="true" />
                  Manage Account
                </button>
                <button
                  type="button"
                  onClick={openYourSetsFromAccountMenu}
                  className="mt-1 flex w-full items-center gap-2.5 rounded-xl px-3 py-2 text-left text-xs text-white/90 transition hover:bg-white/10"
                >
                  <i className="fa-regular fa-folder-open text-[11px] text-white/75" aria-hidden="true" />
                  Your Sets
                </button>
                <button
                  type="button"
                  onClick={() => {
                    void onAccountMenuSignOut();
                  }}
                  className="mt-1 flex w-full items-center gap-2.5 rounded-xl px-3 py-2 text-left text-xs text-white/90 transition hover:bg-white/10"
                >
                  <i className="fa-solid fa-right-from-bracket text-[11px] text-white/75" aria-hidden="true" />
                  Log Out
                </button>
              </div>
            </div>
          ) : null}
          <button
            type="button"
            onClick={() => {
              if (communityAccount) {
                closeHistoryMenus();
                setAccountMenuOpen((prev) => !prev);
                return;
              }
              openAccountScreen();
            }}
            aria-haspopup={communityAccount ? "menu" : undefined}
            aria-expanded={communityAccount ? accountMenuOpen : undefined}
            className="h-10 w-full rounded-xl bg-transparent px-2.5 text-left text-xs text-white/85 transition-colors duration-200 hover:bg-white/10 hover:text-white"
          >
            <div className="flex h-full items-center justify-between gap-1.5">
              <p className="truncate font-semibold leading-none">
                {communityAccount ? `@${communityAccount.username}` : "Login / Sign Up"}
              </p>
              <i
                className={`text-[11px] ${
                  communityAccount
                    ? `fa-solid ${accountMenuOpen ? "fa-chevron-up" : "fa-user"} text-white/74 transition-colors duration-200 group-hover:text-white`
                    : "fa-solid fa-right-to-bracket text-white/74 transition-colors duration-200 group-hover:text-white"
                }`}
                aria-hidden="true"
              />
            </div>
          </button>
        </div>
      </div>
    </div>
  );

  return (
    <main onScroll={onMainScroll} className="home-hero-shell fixed inset-0 h-[100dvh] overflow-x-hidden overflow-y-auto lg:overflow-hidden">
      {visibleSidebarTab === "search" ? (
        <div className="absolute -top-[28%] -right-[0%] bottom-0 z-[2] w-[100%]">
          <VolumetricLightBackground />
        </div>
      ) : null}
      {hasCommunityBackdrop ? (
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
          className={`pointer-events-none fixed z-[90] max-w-[220px] rounded-lg bg-black/95 px-2 py-1 text-left text-[10px] text-white/92 shadow-[0_12px_30px_rgba(0,0,0,0.5)] backdrop-blur-sm transition-[opacity,transform] duration-[220ms] ease-out will-change-[transform,opacity] ${
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
          visibleSidebarTab === "search" ? "transition-none" : "transition-opacity duration-150"
        } lg:hidden ${
          shouldShowTopChromeStrip ? "opacity-100" : "opacity-0"
        }`}
      >
        <div
          className={`h-full w-full ${
            visibleSidebarTab === "community" || visibleSidebarTab === "edit" || visibleSidebarTab === "settings" || visibleSidebarTab === "history"
              ? "bg-black"
              : visibleSidebarTab === "search"
                ? "relative overflow-hidden bg-white/[0.05] backdrop-blur-[10px] backdrop-saturate-160"
                : "bg-black/28 backdrop-blur-[28px] backdrop-saturate-180"
          }`}
        >
          {visibleSidebarTab === "search" ? <div className="absolute inset-0 bg-black/45" /> : null}
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
            className={`absolute bottom-4 left-4 top-6 w-[min(24rem,calc(100vw-2rem))] rounded-3xl bg-white/[0.08] px-3 pb-3 pt-3 text-white shadow-[0_0_40px_rgba(0,0,0,0.32)] backdrop-blur-[26px] backdrop-saturate-150 ${
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
                <div className="h-full rounded-2xl">
                  {sidebarPanelContent}
                </div>
              </div>
            </div>
          </aside>
        </div>
      ) : null}

      <div className="relative z-20 mx-auto h-full min-h-0 w-full max-w-[1680px] lg:mx-0 lg:max-w-none lg:pl-4 lg:grid lg:grid-cols-[280px_minmax(0,1fr)]">
        <aside className="relative z-20 hidden min-h-0 flex-col rounded-3xl bg-white/[0.08] px-3 pt-3 pb-3 text-white shadow-[0_0_40px_rgba(0,0,0,0.24)] backdrop-blur-[18px] backdrop-saturate-150 lg:mt-6 lg:mb-6 lg:flex lg:w-[280px] lg:justify-self-start lg:px-5 lg:pb-5">
          {sidebarPanelContent}
        </aside>

        <section
          className={`relative z-20 h-full min-h-0 w-full overflow-hidden rounded-3xl ${
            hasCommunityBackdrop ? "bg-transparent" : "bg-black/62"
          } ${
            usesCommunityShell
              ? "lg:mt-2 lg:mb-0"
              : "lg:my-2"
          } lg:max-w-[1280px] lg:justify-self-center ${
            usesCommunityShell
              ? "translate-x-0 md:translate-x-1 lg:translate-x-0 lg:w-[99%]"
              : "lg:w-[97%]"
          }`}
        >
          <div className={visibleSidebarTab === "search" ? "h-full min-h-0" : "hidden h-full min-h-0"}>
            <UploadPanel
              onMaterialCreated={onUploadMaterialCreated}
              onScrollOffsetChange={onSearchPanelScrollOffsetChange}
              onScrollGesture={triggerTopChromeGesture}
              onScrollabilityChange={onSearchPanelScrollabilityChange}
            />
          </div>
          {historyPanelContent}
          <div
            className={
              visibleSidebarTab === "settings"
                ? `h-full min-h-0 ${shouldInsetSettingsForTopChrome ? "pt-[calc(max(env(safe-area-inset-top),0px)+0px)] -mt-[10px] md:mt-0 md:pt-[calc(max(env(safe-area-inset-top),0px)+30px)] lg:pt-0" : ""}`
                : "hidden h-full min-h-0"
            }
          >
            <SettingsPanel
              ref={settingsPanelRef}
              onClearSearchData={clearAllHistory}
              onSettingsSaved={syncSavedSettings}
              onUnsavedChangesChange={setHasUnsavedSettingsChanges}
              onAvailabilityModalClose={onSettingsAvailabilityModalClose}
              availabilityModalMode={settingsModalView === "availability" ? "inline" : "overlay"}
              onAvailabilityModalStateChange={setSettingsAvailabilityModalSnapshot}
            />
          </div>
          <div className={visibleSidebarTab === "community" || visibleSidebarTab === "create" || visibleSidebarTab === "edit" ? "h-full min-h-0" : "hidden h-full min-h-0"}>
            <CommunityReelsPanel
              mode={communityPanelMode}
              isVisible={visibleSidebarTab === "community" || visibleSidebarTab === "create" || visibleSidebarTab === "edit"}
              onDetailOpenChange={setCommunityDetailOpen}
              initialOpenSetId={activeCommunitySetId}
              communityResetSignal={communityResetSignal}
              onOpenCommunityReelInFeed={onCommunityReelFeedOpened}
              onDraftUnsavedChangesChange={setHasUnsavedCommunityDraftChanges}
              onDraftExitActionsChange={onCommunityDraftExitActionsChange}
              onActiveCommunitySetChange={setActiveCommunitySetId}
            />
          </div>
        </section>
      </div>
      {showUnsavedCommunityDraftModal ? (
        <ViewportModalPortal>
          <div
            className="fixed inset-0 z-[121] flex items-center justify-center overflow-y-auto bg-black/70 px-4 py-6 backdrop-blur-[2px] transition-opacity duration-200 ease-out opacity-100"
            role="presentation"
            onClick={closeUnsavedCommunityDraftModal}
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
                  onClick={closeUnsavedCommunityDraftModal}
                  aria-label="Close"
                  className="inline-flex h-8 w-8 items-center justify-center text-white/80 transition-colors hover:text-white focus-visible:outline-none"
                >
                  <svg viewBox="0 0 20 20" aria-hidden="true" className="h-4 w-4 fill-none stroke-current stroke-2">
                    <path d="M5 5L15 15M15 5L5 15" strokeLinecap="round" />
                  </svg>
                </button>
              </div>
              <p className="mt-4 rounded-2xl px-4 py-3 text-sm text-white/88">
                Save to keep your draft progress, or discard these edits and continue.
              </p>
              <div className="mt-4 flex flex-wrap items-center justify-end gap-2">
                <button
                  type="button"
                  onClick={discardCommunityDraftAndContinue}
                  className="inline-flex min-w-[9rem] items-center justify-center whitespace-nowrap rounded-xl bg-black/35 px-5 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-white/10"
                >
                  Discard
                </button>
                <button
                  type="button"
                  onClick={saveCommunityDraftAndContinue}
                  className="inline-flex min-w-[9rem] items-center justify-center whitespace-nowrap rounded-xl bg-white px-5 py-2.5 text-sm font-semibold text-black transition-colors hover:bg-white/90"
                >
                  Save
                </button>
              </div>
            </div>
          </div>
        </ViewportModalPortal>
      ) : null}
      {selectedHistoryInfoItem ? (
        <ViewportModalPortal>
          <div
            className="fixed inset-0 z-[122] flex items-center justify-center overflow-y-auto bg-black/70 px-4 py-6 backdrop-blur-[2px] transition-opacity duration-200 ease-out opacity-100"
            role="presentation"
            onClick={() => setSelectedHistoryInfoId(null)}
          >
            <div
              role="dialog"
              aria-modal="true"
              aria-label={`${selectedHistoryInfoItem.title} history information`}
              className="w-full max-w-2xl rounded-3xl border border-zinc-600/35 bg-black p-5 text-white shadow-[0_18px_80px_rgba(0,0,0,0.5)] backdrop-blur-2xl transition-opacity duration-200 ease-out opacity-100 md:p-6"
              onClick={(event) => event.stopPropagation()}
            >
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="text-[11px] font-semibold uppercase tracking-[0.12em] text-white/65">More information</p>
                  <h3 className="mt-2 text-lg font-semibold text-white">{selectedHistoryInfoItem.title}</h3>
                </div>
                <button
                  type="button"
                  onClick={() => setSelectedHistoryInfoId(null)}
                  aria-label="Close"
                  className="inline-flex h-8 w-8 items-center justify-center text-white/80 transition-colors hover:text-white focus-visible:outline-none"
                >
                  <svg viewBox="0 0 20 20" aria-hidden="true" className="h-4 w-4 fill-none stroke-current stroke-2">
                    <path d="M5 5L15 15M15 5L5 15" strokeLinecap="round" />
                  </svg>
                </button>
              </div>
              <div className="mt-4 max-h-[70vh] space-y-4 overflow-y-auto pr-1">
                {selectedHistoryInfoSections.map((section) => (
                  <section key={section.title}>
                    <p className="text-[11px] font-semibold uppercase tracking-[0.1em] text-white/58">{section.title}</p>
                    <div className="mt-2 grid gap-2 sm:grid-cols-2">
                      {section.fields.map((field) => (
                        <div
                          key={`${section.title}-${field.label}`}
                          className="rounded-2xl border border-zinc-800 bg-white/[0.04] px-3 py-3"
                        >
                          <p className="text-[10px] font-semibold uppercase tracking-[0.1em] text-white/52">{field.label}</p>
                          <p className="mt-1 break-words text-sm text-white/92">{field.value}</p>
                        </div>
                      ))}
                    </div>
                  </section>
                ))}
                {selectedHistoryInfoQuery ? (
                  <section>
                    <p className="text-[11px] font-semibold uppercase tracking-[0.1em] text-white/58">Saved feed query</p>
                    <div className="mt-2 rounded-2xl border border-zinc-800 bg-white/[0.04] px-3 py-3">
                      <p className="break-all font-mono text-[11px] leading-5 text-white/72">{selectedHistoryInfoQuery}</p>
                    </div>
                  </section>
                ) : (
                  <div className="rounded-2xl border border-zinc-800 bg-white/[0.04] px-4 py-3 text-sm text-white/72">
                    Detailed feed settings were not saved for this history item.
                  </div>
                )}
              </div>
              <div className="mt-4 flex items-center justify-end">
                <button
                  type="button"
                  onClick={() => setSelectedHistoryInfoId(null)}
                  className="inline-flex min-w-[8rem] items-center justify-center whitespace-nowrap rounded-xl bg-white px-5 py-2.5 text-sm font-semibold text-black transition-colors hover:bg-white/90"
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        </ViewportModalPortal>
      ) : null}
      {settingsModalView ? (
        <ViewportModalPortal>
          <div
            className="fixed inset-0 z-[120] flex items-center justify-center overflow-y-auto bg-black/70 px-4 py-6 backdrop-blur-[2px] transition-opacity duration-200 ease-out opacity-100"
            role="presentation"
            onClick={() => closeSettingsModal("backdrop")}
          >
            <div
              role="dialog"
              aria-modal="true"
              aria-label={settingsModalView === "availability" ? "Configuration success rate" : "Unsaved settings changes"}
              className="w-full max-w-xl rounded-3xl border border-white/25 bg-black p-5 text-white shadow-[0_18px_80px_rgba(0,0,0,0.5)] backdrop-blur-2xl transition-opacity duration-200 ease-out opacity-100 md:p-6"
              onClick={(event) => event.stopPropagation()}
            >
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="text-[11px] font-semibold uppercase tracking-[0.12em] text-white/65">
                    {settingsModalView === "availability" ? "Configuration check" : "Unsaved changes"}
                  </p>
                  <h3 className="mt-2 text-lg font-semibold text-white">
                    {settingsModalView === "availability"
                      ? activeSettingsAvailabilityState.status === "checking"
                        ? "Checking success rate..."
                        : "Success rate result"
                      : "Save settings before leaving?"}
                  </h3>
                </div>
                <button
                  type="button"
                  onClick={() => closeSettingsModal("close-button")}
                  aria-label="Close"
                  className="inline-flex h-8 w-8 items-center justify-center text-white/80 transition-colors hover:text-white focus-visible:outline-none"
                >
                  <svg viewBox="0 0 20 20" aria-hidden="true" className="h-4 w-4 fill-none stroke-current stroke-2">
                    <path d="M5 5L15 15M15 5L5 15" strokeLinecap="round" />
                  </svg>
                </button>
              </div>
              {settingsModalView === "availability" ? (
                <>
                  <div
                    className={`mt-4 rounded-2xl border px-4 py-3 text-sm ${
                      activeSettingsAvailabilityState.status === "ok"
                        ? "border-emerald-300/45 bg-emerald-500/14 text-emerald-100"
                        : activeSettingsAvailabilityState.status === "partial"
                        ? "border-sky-300/45 bg-sky-500/14 text-sky-100"
                        : activeSettingsAvailabilityState.status === "blocked"
                        ? "border-rose-300/45 bg-rose-500/16 text-rose-100"
                        : activeSettingsAvailabilityState.status === "none"
                        ? "border-amber-300/45 bg-amber-500/16 text-amber-100"
                        : activeSettingsAvailabilityState.status === "error"
                        ? "border-rose-300/45 bg-rose-500/16 text-rose-100"
                        : "border-white/24 bg-white/[0.06] text-white/88"
                    }`}
                  >
                    <p>{activeSettingsAvailabilityState.message}</p>
                    {activeSettingsAvailabilityState.limitingFactors.length > 0 ? (
                      <div className="mt-2 border-t border-white/20 pt-2 text-xs">
                        <p className="font-semibold">
                          {activeSettingsAvailabilityState.limitingFactors.length > 1 ? "Main limits:" : "Main limit:"}
                        </p>
                        <ul className="mt-1.5 space-y-1">
                          {activeSettingsAvailabilityState.limitingFactors.map((factor) => (
                            <li key={factor} className="flex items-start gap-1.5">
                              <span aria-hidden="true" className="leading-[1.2] opacity-80">•</span>
                              <span>{factor}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    ) : null}
                  </div>
                  <div className="mt-4 flex items-center justify-end">
                    <button
                      type="button"
                      onClick={() => closeSettingsModal("close-button")}
                      className="inline-flex min-w-[8rem] items-center justify-center whitespace-nowrap rounded-xl bg-white px-5 py-2.5 text-sm font-semibold text-black transition-colors hover:bg-white/90"
                    >
                      OK
                    </button>
                  </div>
                </>
              ) : (
                <>
                  <p className="mt-4 rounded-2xl px-4 py-3 text-sm text-white/88">
                    You changed settings. Save to apply them, or discard these edits and continue.
                  </p>
                  <div className="mt-4 flex flex-wrap items-center justify-end gap-2">
                    <button
                      type="button"
                      onClick={discardSettingsAndContinue}
                      className="inline-flex min-w-[9rem] items-center justify-center whitespace-nowrap rounded-xl bg-black/35 px-5 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-white/10"
                    >
                      Discard
                    </button>
                    <button
                      type="button"
                      onClick={saveSettingsAndContinue}
                      className="inline-flex min-w-[9rem] items-center justify-center whitespace-nowrap rounded-xl bg-white px-5 py-2.5 text-sm font-semibold text-black transition-colors hover:bg-white/90"
                    >
                      Save
                    </button>
                  </div>
                </>
              )}
            </div>
          </div>
        </ViewportModalPortal>
      ) : null}
    </main>
  );
}

export default function HomePage() {
  return (
    <Suspense fallback={<FullscreenLoadingScreen />}>
      <HomePageContent />
    </Suspense>
  );
}
