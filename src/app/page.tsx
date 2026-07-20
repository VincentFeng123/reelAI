"use client";

import {
  Suspense,
  type ReactNode,
  type RefObject,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useRouter, useSearchParams } from "next/navigation";

import { type CommunityDraftExitActions, CommunityReelsPanel } from "@/components/CommunityReelsPanel";
import { FullscreenLoadingScreen } from "@/components/FullscreenLoadingScreen";
import { ViewportModalPortal } from "@/components/ViewportModalPortal";
import {
  COMMUNITY_AUTH_CHANGED_EVENT,
  expireCommunityAuthSession,
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
  type SettingsPanelHandle,
  type SettingsSection,
  SettingsPanel,
} from "@/components/SettingsPanel";
import { UploadPanel } from "@/components/UploadPanel";
import {
  readStudyReelsSettings,
  saveStudyReelsSettings,
  setActiveStudyReelsSettingsScope,
  subscribeToStudyReelsSettings,
  type GenerationMode as SettingsGenerationMode,
  type StudyReelsSettings,
} from "@/lib/settings";
import { useBillingStatus } from "@/lib/useBillingStatus";
import { useLoadingScreenGate } from "@/lib/useLoadingScreenGate";
import {
  LOCAL_DEMO_ACCOUNT,
  LOCAL_DEMO_AVAILABLE,
  LOCAL_DEMO_BILLING_PLANS,
  LOCAL_DEMO_BILLING_STATUS,
  LOCAL_DEMO_HISTORY,
  isLocalDemoView,
} from "@/lib/localDemo";

const MATERIAL_SEEDS_STORAGE_KEY = "studyreels-material-seeds";
const MATERIAL_GROUPS_STORAGE_KEY = "studyreels-material-groups";
const FEED_PROGRESS_STORAGE_KEY = "studyreels-feed-progress";
const FEED_SESSION_STORAGE_KEY = "studyreels-feed-sessions";
const ACTIVE_SIDEBAR_TAB_SESSION_KEY = "studyreels-active-sidebar-tab";
const ACTIVE_COMMUNITY_SET_ID_SESSION_KEY = "studyreels-active-community-set-id";
const DESKTOP_SIDEBAR_COLLAPSED_SESSION_KEY = "studyreels-desktop-sidebar-collapsed";
const MAX_HISTORY_ITEMS = 120;
const UI_FADE_MS = 340;
const MOBILE_SIDEBAR_CLOSE_MS = UI_FADE_MS;
const DESKTOP_SIDEBAR_EXPANDED_PX = 272;
const DESKTOP_SIDEBAR_COLLAPSED_PX = 68;
const LARGE_CENTERED_MODAL_PANEL_CLASS = "flex h-[calc(100dvh-24px)] w-full flex-col sm:h-[min(600px,calc(100dvh-64px))] sm:max-w-[820px]";
type GenerationMode = StoredHistoryItem["generationMode"];
type HistorySource = StoredHistoryItem["source"];
type SidebarTab = "search" | "history" | "community" | "create" | "edit" | "settings";
type AccountReturnTab = "search" | "community" | "edit" | "settings";
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

type ShellIconName =
  | "search"
  | "panel"
  | "compose"
  | "community"
  | "sets"
  | "close"
  | "menu"
  | "chevron"
  | "settings"
  | "auth"
  | "chat"
  | "more"
  | "info"
  | "star"
  | "trash";

function ShellIcon({
  name,
  className = "h-[18px] w-[18px]",
  filled = false,
}: {
  name: ShellIconName;
  className?: string;
  filled?: boolean;
}) {
  let content: ReactNode;

  switch (name) {
    case "search":
      content = (
        <>
          <circle cx="10.8" cy="10.8" r="6.3" />
          <path d="m15.5 15.5 4 4" />
        </>
      );
      break;
    case "panel":
      content = (
        <>
          <rect x="3.25" y="4.25" width="17.5" height="15.5" rx="2.75" />
          <path d="M9 4.5v15" />
        </>
      );
      break;
    case "compose":
      content = (
        <>
          <path d="M12 5H7a3 3 0 0 0-3 3v9a3 3 0 0 0 3 3h9a3 3 0 0 0 3-3v-5" />
          <path d="m9.5 14.5.8-3 6.8-6.8a1.4 1.4 0 0 1 2 2l-6.8 6.8-2.8 1Z" />
        </>
      );
      break;
    case "community":
      content = (
        <>
          <circle cx="9" cy="9" r="3" />
          <circle cx="17" cy="10" r="2.25" />
          <path d="M3.8 19c.5-3 2.3-4.5 5.2-4.5s4.7 1.5 5.2 4.5M15 14.7c2.8-.5 4.6.8 5.2 3.3" />
        </>
      );
      break;
    case "sets":
      content = (
        <>
          <path d="M3.5 7.5h6l1.8 2h9.2v7.25A2.75 2.75 0 0 1 17.75 19.5H6.25a2.75 2.75 0 0 1-2.75-2.75V7.5Z" />
          <path d="M3.5 7.5V6.75A2.25 2.25 0 0 1 5.75 4.5h3.4l1.7 2" />
        </>
      );
      break;
    case "close":
      content = <path d="m6 6 12 12M18 6 6 18" />;
      break;
    case "menu":
      content = <path d="M4 8h16M4 16h16" />;
      break;
    case "chevron":
      content = <path d="m9 5 7 7-7 7" />;
      break;
    case "settings":
      content = (
        <>
          <path d="M4 7h7M15 7h5M4 17h5M13 17h7" />
          <circle cx="13" cy="7" r="2" />
          <circle cx="11" cy="17" r="2" />
        </>
      );
      break;
    case "auth":
      content = (
        <>
          <path d="M10 5H6.5A2.5 2.5 0 0 0 4 7.5v9A2.5 2.5 0 0 0 6.5 19H10" />
          <path d="m14 8 4 4-4 4M18 12H9" />
        </>
      );
      break;
    case "chat":
      content = <path d="M20 11.5a7.5 7.5 0 0 1-8 7.5 9 9 0 0 1-3.4-.7L4 20l1.5-4a7.5 7.5 0 1 1 14.5-4.5Z" />;
      break;
    case "more":
      content = (
        <>
          <circle cx="6" cy="12" r="1" fill="currentColor" stroke="none" />
          <circle cx="12" cy="12" r="1" fill="currentColor" stroke="none" />
          <circle cx="18" cy="12" r="1" fill="currentColor" stroke="none" />
        </>
      );
      break;
    case "info":
      content = (
        <>
          <circle cx="12" cy="12" r="8.5" />
          <path d="M12 10.5v5M12 7.5h.01" />
        </>
      );
      break;
    case "star":
      content = <path d="m12 3.8 2.55 5.15 5.7.83-4.12 4.02.97 5.68L12 16.8l-5.1 2.68.97-5.68-4.12-4.02 5.7-.83L12 3.8Z" />;
      break;
    case "trash":
      content = (
        <>
          <path d="M5 7h14M9 7V4.5h6V7M7 7l.75 12.5h8.5L17 7M10 10.5v5.5M14 10.5v5.5" />
        </>
      );
      break;
  }

  return (
    <svg
      viewBox="0 0 24 24"
      aria-hidden="true"
      className={`${className} shrink-0 ${filled ? "fill-current" : "fill-none"} stroke-current`}
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      {content}
    </svg>
  );
}

const MODAL_FOCUSABLE_SELECTOR = [
  "a[href]",
  "button:not([disabled])",
  "input:not([disabled])",
  "select:not([disabled])",
  "textarea:not([disabled])",
  "[tabindex]:not([tabindex='-1'])",
].join(",");

type ShellModalProps = {
  open: boolean;
  label: string;
  onRequestClose: () => void;
  children: ReactNode;
  panelClassName?: string;
  initialFocusRef?: RefObject<HTMLElement | null>;
};

function getModalReturnFocusTarget(): HTMLElement | null {
  if (typeof document === "undefined" || !(document.activeElement instanceof HTMLElement)) {
    return null;
  }
  const activeElement = document.activeElement;
  return activeElement
    .closest<HTMLElement>("[data-account-actions]")
    ?.querySelector<HTMLElement>('button[aria-haspopup="menu"]')
    ?? activeElement;
}

function ShellModal({ open, label, onRequestClose, children, panelClassName = "", initialFocusRef }: ShellModalProps) {
  const [isRendered, setIsRendered] = useState(open);
  const [isVisible, setIsVisible] = useState(false);
  const [dialogElement, setDialogElement] = useState<HTMLDivElement | null>(null);
  const renderedChildrenRef = useRef(children);
  const returnFocusRef = useRef<HTMLElement | null>(null);
  const requestCloseRef = useRef(onRequestClose);

  useLayoutEffect(() => {
    if (open) {
      renderedChildrenRef.current = children;
    }
  }, [children, open]);

  useEffect(() => {
    requestCloseRef.current = onRequestClose;
  }, [onRequestClose]);

  useEffect(() => {
    let frame = 0;
    let closeTimer = 0;
    if (open) {
      returnFocusRef.current = getModalReturnFocusTarget();
      setIsRendered(true);
      frame = window.requestAnimationFrame(() => setIsVisible(true));
    } else {
      setIsVisible(false);
      const closeDelay = window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : UI_FADE_MS;
      closeTimer = window.setTimeout(() => {
        setIsRendered(false);
        const returnFocus = returnFocusRef.current;
        window.requestAnimationFrame(() => {
          if (returnFocus?.isConnected) {
            returnFocus.focus();
          }
        });
      }, closeDelay);
    }
    return () => {
      window.cancelAnimationFrame(frame);
      window.clearTimeout(closeTimer);
    };
  }, [open]);

  useEffect(() => {
    const dialog = dialogElement;
    if (!dialog) {
      return;
    }
    const getFocusableElements = () => Array.from(dialog.querySelectorAll<HTMLElement>(MODAL_FOCUSABLE_SELECTOR))
      .filter((element) => (
        !element.hidden
        && element.getAttribute("aria-hidden") !== "true"
        && !element.closest("[inert], [aria-hidden='true']")
        && element.getClientRects().length > 0
      ));
    const requestedFocus = initialFocusRef?.current;
    const focusTarget = requestedFocus && getFocusableElements().includes(requestedFocus)
      ? requestedFocus
      : getFocusableElements()[0] ?? dialog;
    const frame = window.requestAnimationFrame(() => focusTarget?.focus());

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        requestCloseRef.current();
        return;
      }
      if (event.key !== "Tab" || !dialog) {
        return;
      }
      const focusable = getFocusableElements();
      if (focusable.length === 0) {
        event.preventDefault();
        dialog.focus();
        return;
      }
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };

    document.addEventListener("keydown", onKeyDown);
    return () => {
      window.cancelAnimationFrame(frame);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [dialogElement, initialFocusRef]);

  if (!isRendered) {
    return null;
  }

  return (
    <ViewportModalPortal>
      <div
        className={`fixed inset-0 z-[150] flex items-center justify-center bg-black/70 p-3 transition-opacity duration-300 motion-reduce:transition-none sm:p-6 ${
          isVisible ? "opacity-100" : "opacity-0"
        }`}
        role="presentation"
        onMouseDown={(event) => {
          if (event.target === event.currentTarget) {
            onRequestClose();
          }
        }}
      >
        <div
          ref={setDialogElement}
          role="dialog"
          aria-modal="true"
          aria-label={label}
          tabIndex={-1}
          className={`overflow-hidden rounded-[14px] bg-[#202020] text-white outline-none ${panelClassName}`}
        >
          {open ? children : renderedChildrenRef.current}
        </div>
      </div>
    </ViewportModalPortal>
  );
}

export function reelAIBrandLabel(plan: "free" | "plus" | "pro" | null | undefined): string {
  if (plan === "plus") {
    return "ReelAI Plus";
  }
  if (plan === "pro") {
    return "ReelAI Pro";
  }
  return "ReelAI";
}

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

function accountReturnTabForSidebar(tab: SidebarTab): AccountReturnTab {
  if (tab === "community") {
    return "community";
  }
  if (tab === "create" || tab === "edit") {
    return "edit";
  }
  return "search";
}

function buildAccountAuthPath(mode: "login" | "register", returnTab: AccountReturnTab): string {
  return `/account?mode=${mode}&return_tab=${returnTab}`;
}

function normalizeSettingsSection(value: string | null): SettingsSection | null {
  if (value === "search" || value === "playback" || value === "plan" || value === "data" || value === "account") {
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
        generationMode: "slow",
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
        generationMode: item.generationMode || existing.generationMode || "slow",
        source: item.source || existing.source || "search",
        feedQuery: item.feedQuery ?? existing.feedQuery,
        activeIndex: item.activeIndex ?? existing.activeIndex,
        activeReelId: item.activeReelId ?? existing.activeReelId,
        recall: item.recall ?? existing.recall,
      });
      continue;
    }
    map.set(item.materialId, {
      ...existing,
      starred: existing.starred || item.starred,
      generationMode: existing.generationMode || item.generationMode || "slow",
      source: existing.source || item.source || "search",
      feedQuery: existing.feedQuery ?? item.feedQuery,
      activeIndex: existing.activeIndex ?? item.activeIndex,
      activeReelId: existing.activeReelId ?? item.activeReelId,
      recall: existing.recall ?? item.recall,
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

function formatHistoryInfoAccuracy(value: number | undefined): string | null {
  if (!Number.isFinite(value)) {
    return null;
  }
  return `${Math.round(Math.max(0, Math.min(1, Number(value))) * 100)}%`;
}

function formatHistoryInfoSeconds(value: string | null | undefined): string | null {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < 0) {
    return null;
  }
  return `${Math.round(parsed)} sec`;
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
  const recallFields: HistoryInfoField[] = [];

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

  if (item.recall) {
    if (Number.isFinite(item.recall.recentScore) && Number.isFinite(item.recall.recentQuestionCount)) {
      pushHistoryInfoField(recallFields, "Latest score", `${item.recall.recentScore} of ${item.recall.recentQuestionCount}`);
    }
    pushHistoryInfoField(recallFields, "Recent accuracy", formatHistoryInfoAccuracy(item.recall.recentAccuracy));
    pushHistoryInfoField(recallFields, "Rolling accuracy", formatHistoryInfoAccuracy(item.recall.rollingAccuracy));
    pushHistoryInfoField(recallFields, "Concepts understood", item.recall.understoodConcepts.join(", "));
    pushHistoryInfoField(recallFields, "Concepts to revisit", item.recall.revisitConcepts.join(", "));
    if (item.recall.completedAt) {
      pushHistoryInfoField(recallFields, "Last recall check", formatHistoryInfoDate(item.recall.completedAt));
    }
  }

  return [
    { title: "Session details", fields: summaryFields },
    { title: "Recall", fields: recallFields },
    { title: "Search settings", fields: searchFields },
    { title: "Playback", fields: playbackFields },
  ].filter((section) => section.fields.length > 0);
}

function HomePageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const demoAccountEnabled = isLocalDemoView(searchParams.get("demo"), "account");
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
  const [activeSidebarHistoryMenuId, setActiveSidebarHistoryMenuId] = useState<string | null>(null);
  const [activeHistoryPageMenuId, setActiveHistoryPageMenuId] = useState<string | null>(null);
  const [selectedHistoryInfoId, setSelectedHistoryInfoId] = useState<string | null>(null);
  const [accountMenuOpen, setAccountMenuOpen] = useState(false);
  const [desktopSidebarCollapsed, setDesktopSidebarCollapsed] = useState(false);
  const [activeSidebarTab, setActiveSidebarTab] = useState<SidebarTab>("search");
  const [sidebarTabHydrated, setSidebarTabHydrated] = useState(false);
  const [activeCommunitySetId, setActiveCommunitySetId] = useState<string | null>(null);
  const [communityAccount, setCommunityAccount] = useState<CommunityAccount | null>(null);
  const [historySearchOpen, setHistorySearchOpen] = useState(false);
  const [settingsSection, setSettingsSection] = useState<SettingsSection | null>(null);
  const [settingsClosePrompt, setSettingsClosePrompt] = useState(false);
  const [pendingSettingsAuthMode, setPendingSettingsAuthMode] = useState<"login" | "register" | null>(null);
  const [generationMode, setGenerationMode] = useState<SettingsGenerationMode>("slow");
  const [composerResetKey, setComposerResetKey] = useState(0);
  const [hasUnsavedSettingsChanges, setHasUnsavedSettingsChanges] = useState(false);
  const [hasUnsavedCommunityDraftChanges, setHasUnsavedCommunityDraftChanges] = useState(false);
  const [showUnsavedCommunityDraftModal, setShowUnsavedCommunityDraftModal] = useState(false);
  const [pendingCommunityDraftSwitchIntent, setPendingCommunityDraftSwitchIntent] = useState<SidebarSwitchIntent | null>(null);
  const [communityDetailOpen, setCommunityDetailOpen] = useState(false);
  const [communityResetSignal, setCommunityResetSignal] = useState(0);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const historyRef = useRef<HistoryItem[]>([]);
  const searchHeroTitleRef = useRef<HTMLHeadingElement | null>(null);
  const mainContentRef = useRef<HTMLElement | null>(null);
  const settingsPanelRef = useRef<SettingsPanelHandle | null>(null);
  const settingsPromptFocusRef = useRef<HTMLButtonElement | null>(null);
  const settingsPromptReturnFocusRef = useRef<HTMLElement | null>(null);
  const communityDraftExitActionsRef = useRef<CommunityDraftExitActions | null>(null);
  const hasProcessedInitialSidebarHydrationRef = useRef(false);
  const mobileSidebarCloseTimerRef = useRef<number | null>(null);
  const mobileSidebarInitialFocusAppliedRef = useRef(false);
  const mobileSidebarTriggerRef = useRef<HTMLButtonElement | null>(null);
  const shellModalPresenceTimerRef = useRef<number | null>(null);
  const [shellModalPresenceActive, setShellModalPresenceActive] = useState(false);
  const historyMutationVersionRef = useRef(0);
  const historyLoadSequenceRef = useRef(0);
  const settingsLoadSequenceRef = useRef(0);
  const historySearchInputRef = useRef<HTMLInputElement | null>(null);

  const liveBilling = useBillingStatus(demoAccountEnabled ? null : communityAccount);
  const shellAccount = demoAccountEnabled ? LOCAL_DEMO_ACCOUNT : communityAccount;
  const billingStatus = demoAccountEnabled ? LOCAL_DEMO_BILLING_STATUS : liveBilling.status;
  const billingPlans = demoAccountEnabled ? LOCAL_DEMO_BILLING_PLANS : liveBilling.plans;
  const billingLoading = demoAccountEnabled ? false : liveBilling.loading;
  const billingError = demoAccountEnabled ? null : liveBilling.error;

  const resolveHistoryAccountId = useCallback(() => {
    if (demoAccountEnabled) {
      return LOCAL_DEMO_ACCOUNT.id;
    }
    const activeAccountId = communityAccount?.id?.trim();
    if (activeAccountId) {
      return activeAccountId;
    }
    return readCommunityAuthSession()?.account?.id?.trim() || null;
  }, [communityAccount?.id, demoAccountEnabled]);

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
    const storedHistory = parseMaterialHistory(readScopedHistorySnapshot(scopedAccountId));
    const base = demoAccountEnabled && storedHistory.length === 0
      ? LOCAL_DEMO_HISTORY
      : storedHistory;
    const legacy = scopedAccountId ? [] : parseLegacyTopicHistory(window.localStorage.getItem(LEGACY_TOPIC_HISTORY_STORAGE_KEY));
    const merged = mergeHistory(base, legacy);
    setHistorySnapshot(merged, { accountId: scopedAccountId });

    if (demoAccountEnabled || !scopedAccountId) {
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
          expireCommunityAuthSession();
          setCommunityAccount(null);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [demoAccountEnabled, resolveHistoryAccountId, setHistorySnapshot]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const scopedAccountId = resolveHistoryAccountId();
    setActiveStudyReelsSettingsScope(scopedAccountId);

    if (demoAccountEnabled || !scopedAccountId) {
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
          expireCommunityAuthSession();
          setCommunityAccount(null);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [demoAccountEnabled, resolveHistoryAccountId]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    if (demoAccountEnabled) {
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
  }, [demoAccountEnabled]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const savedTab = normalizeSidebarTab(window.sessionStorage.getItem(ACTIVE_SIDEBAR_TAB_SESSION_KEY));
    const savedCommunitySetIdRaw = window.sessionStorage.getItem(ACTIVE_COMMUNITY_SET_ID_SESSION_KEY);
    const savedCommunitySetId = savedCommunitySetIdRaw?.trim() || null;
    const nextTab = forcedSidebarTab ?? (forcedCommunitySetId ? "community" : savedTab ?? "search");
    setActiveSidebarTab(nextTab);
    setActiveCommunitySetId(forcedCommunitySetId ?? savedCommunitySetId);
    setSidebarTabHydrated(true);
  }, [forcedCommunitySetId, forcedSidebarTab]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    setDesktopSidebarCollapsed(window.sessionStorage.getItem(DESKTOP_SIDEBAR_COLLAPSED_SESSION_KEY) === "1");
  }, []);

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

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    safeStorageSetItem(window.sessionStorage, DESKTOP_SIDEBAR_COLLAPSED_SESSION_KEY, desktopSidebarCollapsed ? "1" : "0");
  }, [desktopSidebarCollapsed]);

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
  const historyRecent = useMemo(
    () => [...history].sort((a, b) => b.updatedAt - a.updatedAt),
    [history],
  );
  const filteredHistory = useMemo(() => {
    const query = historyQuery.trim().toLowerCase();
    if (!query) {
      return historyRecent;
    }
    return historyRecent.filter((item) => item.title.toLowerCase().includes(query));
  }, [historyQuery, historyRecent]);
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
  const shellModalLogicallyOpen = historySearchOpen
    || settingsSection !== null
    || showUnsavedCommunityDraftModal
    || selectedHistoryInfoItem !== null;

  useEffect(() => {
    if (shellModalPresenceTimerRef.current !== null) {
      window.clearTimeout(shellModalPresenceTimerRef.current);
      shellModalPresenceTimerRef.current = null;
    }
    if (shellModalLogicallyOpen) {
      setShellModalPresenceActive(true);
      return;
    }
    const fadeMs = window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : UI_FADE_MS;
    shellModalPresenceTimerRef.current = window.setTimeout(() => {
      setShellModalPresenceActive(false);
      shellModalPresenceTimerRef.current = null;
    }, fadeMs);
    return () => {
      if (shellModalPresenceTimerRef.current !== null) {
        window.clearTimeout(shellModalPresenceTimerRef.current);
        shellModalPresenceTimerRef.current = null;
      }
    };
  }, [shellModalLogicallyOpen]);

  const persistHistory = useCallback((next: HistoryItem[]) => {
    const scopedAccountId = resolveHistoryAccountId();
    historyMutationVersionRef.current += 1;
    setHistorySnapshot(next, { accountId: scopedAccountId });
    if (demoAccountEnabled || !scopedAccountId) {
      return;
    }
    void queueCommunityHistorySync(next).catch((error) => {
      if (isSessionExpiredError(error)) {
        expireCommunityAuthSession();
        setCommunityAccount(null);
      }
    });
  }, [demoAccountEnabled, resolveHistoryAccountId, setHistorySnapshot]);

  const syncSavedSettings = useCallback((settings: StudyReelsSettings) => {
    const scopedAccountId = resolveHistoryAccountId();
    if (demoAccountEnabled || !scopedAccountId) {
      return;
    }
    void queueCommunitySettingsSync(settings).catch((error) => {
      if (isSessionExpiredError(error)) {
        expireCommunityAuthSession();
        setCommunityAccount(null);
      }
    });
  }, [demoAccountEnabled, resolveHistoryAccountId]);

  useEffect(() => {
    setGenerationMode(readStudyReelsSettings().generationMode);
    return subscribeToStudyReelsSettings((settings) => {
      setGenerationMode(settings.generationMode);
    });
  }, [communityAccount?.id, demoAccountEnabled]);

  useEffect(() => {
    if (!sidebarTabHydrated) {
      return;
    }
    const requestedSection = normalizeSettingsSection(searchParams.get("settings"));
    if (requestedSection) {
      setSettingsSection(requestedSection);
      setActiveSidebarTab((current) => (current === "settings" ? "search" : current));
      return;
    }
    if (forcedSidebarTab === "settings") {
      const nextQuery = new URLSearchParams(searchParams.toString());
      nextQuery.delete("tab");
      nextQuery.set("settings", "search");
      router.replace(`/?${nextQuery.toString()}`);
      setSettingsSection("search");
      setActiveSidebarTab("search");
      return;
    }
    if (forcedSidebarTab === "history") {
      const nextQuery = new URLSearchParams(searchParams.toString());
      nextQuery.delete("tab");
      const query = nextQuery.toString();
      router.replace(query ? `/?${query}` : "/");
      setHistorySearchOpen(true);
      setActiveSidebarTab("search");
    }
  }, [forcedSidebarTab, router, searchParams, sidebarTabHydrated]);

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
      recall?: HistoryItem["recall"];
    }) => {
      const existing = historyRef.current.find((item) => item.materialId === entry.materialId);
      const merged: HistoryItem = {
        materialId: entry.materialId,
        title: entry.title,
        updatedAt: entry.updatedAt,
        starred: entry.starred ?? existing?.starred ?? false,
        generationMode: entry.generationMode ?? existing?.generationMode ?? "slow",
        source: entry.source ?? existing?.source ?? "search",
        feedQuery: entry.feedQuery ?? existing?.feedQuery,
        activeIndex: entry.activeIndex ?? existing?.activeIndex,
        activeReelId: entry.activeReelId ?? existing?.activeReelId,
        recall: entry.recall ?? existing?.recall,
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
    if (demoAccountEnabled) {
      writeScopedHistorySnapshot(scopedAccountId, JSON.stringify([]));
      return;
    }
    if (typeof window !== "undefined") {
      if (scopedAccountId) {
        writeScopedHistorySnapshot(scopedAccountId, JSON.stringify([]));
        void queueCommunityHistorySync([]).catch((error) => {
          if (isSessionExpiredError(error)) {
            expireCommunityAuthSession();
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
  }, [closeHistoryMenus, demoAccountEnabled, resolveHistoryAccountId]);

  const clearMobileSidebarCloseTimer = useCallback(() => {
    if (mobileSidebarCloseTimerRef.current !== null) {
      window.clearTimeout(mobileSidebarCloseTimerRef.current);
      mobileSidebarCloseTimerRef.current = null;
    }
  }, []);

  const openMobileSidebar = useCallback(() => {
    clearMobileSidebarCloseTimer();
    mobileSidebarInitialFocusAppliedRef.current = false;
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
      mobileSidebarInitialFocusAppliedRef.current = false;
      mobileSidebarCloseTimerRef.current = null;
      window.requestAnimationFrame(() => mobileSidebarTriggerRef.current?.focus());
    }, MOBILE_SIDEBAR_CLOSE_MS);
  }, [clearMobileSidebarCloseTimer, mobileSidebarClosing, mobileSidebarOpen]);

  useEffect(() => {
    if (
      !mobileSidebarOpen
      || mobileSidebarClosing
      || shellModalLogicallyOpen
      || shellModalPresenceActive
    ) {
      return;
    }
    const drawer = document.querySelector<HTMLElement>("[data-mobile-drawer]");
    if (!drawer) {
      return;
    }
    const getFocusableElements = () => Array.from(drawer.querySelectorAll<HTMLElement>(MODAL_FOCUSABLE_SELECTOR))
      .filter((element) => (
        !element.hidden
        && element.getAttribute("aria-hidden") !== "true"
        && !element.closest("[inert], [aria-hidden='true']")
        && element.getClientRects().length > 0
      ));
    const frame = mobileSidebarInitialFocusAppliedRef.current
      ? 0
      : window.requestAnimationFrame(() => {
        mobileSidebarInitialFocusAppliedRef.current = true;
        drawer.querySelector<HTMLElement>('button[aria-label="Close sidebar"]')?.focus();
      });
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        closeMobileSidebar();
        return;
      }
      if (event.key !== "Tab") {
        return;
      }
      const focusable = getFocusableElements();
      if (focusable.length === 0) {
        event.preventDefault();
        drawer.focus();
        return;
      }
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", onKeyDown);
    return () => {
      window.cancelAnimationFrame(frame);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [closeMobileSidebar, mobileSidebarClosing, mobileSidebarOpen, shellModalLogicallyOpen, shellModalPresenceActive]);

  const forceCloseMobileSidebar = useCallback(() => {
    clearMobileSidebarCloseTimer();
    setAccountMenuOpen(false);
    setMobileSidebarClosing(false);
    setMobileSidebarOpen(false);
    mobileSidebarInitialFocusAppliedRef.current = false;
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
    const shouldMoveFocusToContent = mobileSidebarOpen;
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
    if (shouldMoveFocusToContent) {
      window.requestAnimationFrame(() => mainContentRef.current?.focus());
    }
  }, [activeSidebarTab, clearCommunitySelectionFromUrl, closeHistoryMenus, forceCloseMobileSidebar, mobileSidebarOpen]);

  const requestSidebarSwitch = useCallback(
    (intent: SidebarSwitchIntent) => {
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
    [activeSidebarTab, applySidebarSwitchIntent, hasUnsavedCommunityDraftChanges],
  );

  const startNewSearch = useCallback(() => {
    setComposerResetKey((current) => current + 1);
    requestSidebarSwitch({ tab: "search", clearHistoryQuery: true });
  }, [requestSidebarSwitch]);

  const toggleDesktopSidebarCollapsed = useCallback(() => {
    closeHistoryMenus();
    setAccountMenuOpen(false);
    setDesktopSidebarCollapsed((prev) => !prev);
  }, [closeHistoryMenus]);

  const openAuthPage = useCallback((mode: "login" | "register" = "login") => {
    closeHistoryMenus();
    setSelectedHistoryInfoId(null);
    setAccountMenuOpen(false);
    router.push(buildAccountAuthPath(mode, accountReturnTabForSidebar(activeSidebarTab)));
  }, [activeSidebarTab, closeHistoryMenus, router]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const hasPendingUnsavedSettings = settingsSection !== null
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
  }, [activeSidebarTab, hasUnsavedCommunityDraftChanges, hasUnsavedSettingsChanges, settingsSection]);

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
    };
  }, [clearMobileSidebarCloseTimer]);

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
  const showGenerationSpeedToggle = visibleSidebarTab === "search";
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
      const mode = existing?.generationMode ?? "slow";
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
        generationMode: "slow",
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
  const sidebarPrimaryActions = [
    {
      key: "search",
      label: "New search",
      icon: "compose",
      isActive: visibleSidebarTab === "search",
      onClick: startNewSearch,
    },
    {
      key: "community",
      label: "Community Reels",
      icon: "community",
      isActive: visibleSidebarTab === "community",
      onClick: () => switchSidebarTab("community", { resetCommunityView: true }),
    },
    ...(shellAccount ? [{
      key: "edit",
      label: "Your Sets",
      icon: "sets" as const,
      isActive: visibleSidebarTab === "edit" || visibleSidebarTab === "create",
      onClick: () => switchSidebarTab("edit"),
    }] : []),
  ] satisfies ReadonlyArray<{
    key: string;
    label: string;
    icon: ShellIconName;
    isActive: boolean;
    onClick: () => void;
  }>;
  const exitDemoAccount = useCallback(() => {
    const nextQuery = new URLSearchParams(searchParams.toString());
    nextQuery.delete("demo");
    const query = nextQuery.toString();
    router.replace(query ? `/?${query}` : "/");
  }, [router, searchParams]);
  const onAccountMenuSignOut = useCallback(async () => {
    setAccountMenuOpen(false);
    if (demoAccountEnabled) {
      exitDemoAccount();
      return;
    }
    try {
      await logoutCommunityAccount();
    } finally {
      setCommunityAccount(null);
    }
  }, [demoAccountEnabled, exitDemoAccount]);
  const closeHistorySearch = useCallback(() => {
    setHistorySearchOpen(false);
    setHistoryQuery("");
  }, []);
  const closeHistoryInfo = useCallback(() => {
    setSelectedHistoryInfoId(null);
  }, []);
  const openSettings = useCallback((section: SettingsSection = "search") => {
    setAccountMenuOpen(false);
    setHistorySearchOpen(false);
    setSettingsClosePrompt(false);
    setPendingSettingsAuthMode(null);
    setSettingsSection(section);
    const nextQuery = new URLSearchParams(searchParams.toString());
    nextQuery.delete("tab");
    nextQuery.set("settings", section);
    router.replace(`/?${nextQuery.toString()}`);
  }, [router, searchParams]);
  const finishCloseSettings = useCallback((authMode?: "login" | "register") => {
    setSettingsSection(null);
    setSettingsClosePrompt(false);
    setPendingSettingsAuthMode(null);
    if (authMode) {
      const navigationDelay = window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : UI_FADE_MS;
      window.setTimeout(() => {
        router.push(buildAccountAuthPath(authMode, "settings"));
      }, navigationDelay);
      return;
    }
    const nextQuery = new URLSearchParams(searchParams.toString());
    nextQuery.delete("settings");
    const query = nextQuery.toString();
    router.replace(query ? `/?${query}` : "/");
  }, [router, searchParams]);
  const onSettingsSectionChange = useCallback((section: SettingsSection) => {
    setSettingsSection(section);
    const nextQuery = new URLSearchParams(searchParams.toString());
    nextQuery.delete("tab");
    nextQuery.set("settings", section);
    router.replace(`/?${nextQuery.toString()}`);
  }, [router, searchParams]);
  const showSettingsClosePrompt = useCallback((authMode: "login" | "register" | null) => {
    settingsPromptReturnFocusRef.current = document.activeElement instanceof HTMLElement
      ? document.activeElement
      : null;
    setPendingSettingsAuthMode(authMode);
    setSettingsClosePrompt(true);
  }, []);
  const keepEditingSettings = useCallback(() => {
    const returnFocus = settingsPromptReturnFocusRef.current;
    setSettingsClosePrompt(false);
    setPendingSettingsAuthMode(null);
    window.requestAnimationFrame(() => {
      if (returnFocus?.isConnected) {
        returnFocus.focus();
      }
    });
  }, []);
  const requestCloseSettings = useCallback(() => {
    setPendingSettingsAuthMode(null);
    const dirty = hasUnsavedSettingsChanges || Boolean(settingsPanelRef.current?.hasUnsavedChanges());
    if (dirty) {
      showSettingsClosePrompt(null);
      return;
    }
    finishCloseSettings();
  }, [finishCloseSettings, hasUnsavedSettingsChanges, showSettingsClosePrompt]);
  const requestOpenAuth = useCallback((mode: "login" | "register" | "verify" = "login") => {
    const authMode = mode === "register" ? "register" : "login";
    const dirty = hasUnsavedSettingsChanges || Boolean(settingsPanelRef.current?.hasUnsavedChanges());
    if (dirty) {
      showSettingsClosePrompt(authMode);
      return;
    }
    finishCloseSettings(authMode);
  }, [finishCloseSettings, hasUnsavedSettingsChanges, showSettingsClosePrompt]);
  const onSettingsModalRequestClose = useCallback(() => {
    if (settingsClosePrompt) {
      keepEditingSettings();
      return;
    }
    requestCloseSettings();
  }, [keepEditingSettings, requestCloseSettings, settingsClosePrompt]);
  useEffect(() => {
    if (!settingsClosePrompt) {
      return;
    }
    const frame = window.requestAnimationFrame(() => settingsPromptFocusRef.current?.focus());
    return () => window.cancelAnimationFrame(frame);
  }, [settingsClosePrompt]);
  const applyGenerationMode = useCallback((nextMode: SettingsGenerationMode) => {
    if (nextMode === generationMode) {
      return;
    }
    const saved = saveStudyReelsSettings({
      ...readStudyReelsSettings(),
      generationMode: nextMode,
    });
    setGenerationMode(saved.generationMode);
    syncSavedSettings(saved);
  }, [generationMode, syncSavedSettings]);
  const onCommunityDraftExitActionsChange = useCallback((actions: CommunityDraftExitActions | null) => {
    // Unsaved draft state is owned by the panel's explicit dirty-state callback.
    // Clearing it here would incorrectly mark the form clean during effect cleanups.
    communityDraftExitActionsRef.current = actions;
  }, []);
  const closeUnsavedCommunityDraftModal = useCallback(() => {
    setShowUnsavedCommunityDraftModal(false);
    setPendingCommunityDraftSwitchIntent(null);
  }, []);
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
  const isCommunityPanel = visibleSidebarTab === "community" || visibleSidebarTab === "create" || visibleSidebarTab === "edit";
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
  const desktopSidebarWidthPx = desktopSidebarCollapsed
    ? DESKTOP_SIDEBAR_COLLAPSED_PX
    : DESKTOP_SIDEBAR_EXPANDED_PX;
  const showLoadingScreen = useLoadingScreenGate(sidebarTabHydrated, { minimumVisibleMs: 250 });

  if (showLoadingScreen) {
    return <FullscreenLoadingScreen />;
  }

  const visibleBillingPlan = billingStatus?.plan;
  const brandLabel = reelAIBrandLabel(visibleBillingPlan);
  const accountPlanLabel = visibleBillingPlan === "pro"
    ? "Pro"
    : visibleBillingPlan === "plus"
      ? "Plus"
      : visibleBillingPlan === "free" || !shellAccount || !shellAccount.isVerified
        ? "Free"
        : billingError
          ? "Plan unavailable"
          : "Loading…";
  const accountInitial = (shellAccount?.username?.trim().charAt(0) || "R").toUpperCase();
  const showGuestAuthActions = !shellAccount
    && (visibleSidebarTab === "search" || visibleSidebarTab === "edit" || visibleSidebarTab === "create");
  const fastSlowToggle = (
    <div
      role="group"
      aria-label="Generation speed"
      className="relative flex h-9 w-[218px] items-center rounded-full bg-[#181818] text-sm font-medium text-zinc-400"
    >
      <span
        data-generation-speed-indicator
        aria-hidden="true"
        className={`absolute -left-0.5 -top-0.5 h-10 w-[calc(50%+2px)] rounded-full bg-[#242424] transition-transform duration-[360ms] ease-in-out motion-reduce:transition-none ${
          generationMode === "slow" ? "translate-x-full" : "translate-x-0"
        }`}
      />
      {(["fast", "slow"] as const).map((mode) => (
        <button
          key={mode}
          type="button"
          aria-pressed={generationMode === mode}
          onClick={() => applyGenerationMode(mode)}
          className={`relative z-10 h-9 flex-1 rounded-full capitalize transition-colors duration-300 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white ${
            generationMode === mode ? "text-white" : "hover:text-zinc-200"
          }`}
        >
          {mode}
        </button>
      ))}
    </div>
  );

  const renderAccountPopover = (collapsed = false) => (
    <div
      data-shell-account-popover={collapsed ? "collapsed" : "expanded"}
      role="menu"
      aria-hidden={!accountMenuOpen}
      inert={!accountMenuOpen}
      className={`absolute z-40 w-[256px] rounded-2xl bg-[#202020] p-2 transition-opacity duration-300 motion-reduce:transition-none ${
        collapsed ? "bottom-[calc(100%+24px)] left-[-6px]" : "bottom-[calc(100%+10px)] left-0"
      } ${
        accountMenuOpen ? "pointer-events-auto opacity-100" : "pointer-events-none opacity-0"
      }`}
    >
      {shellAccount ? (
        <button
          type="button"
          role="menuitem"
          onClick={() => openSettings("account")}
          className="flex w-full items-center gap-3 rounded-xl px-3 py-3 text-left transition-colors hover:bg-white/[0.07] focus-visible:outline focus-visible:outline-2 focus-visible:outline-white"
        >
          <span className="grid h-9 w-9 shrink-0 place-items-center rounded-full bg-emerald-500 text-sm font-semibold text-white">
            {accountInitial}
          </span>
          <span className="min-w-0">
            <span className="block truncate text-sm font-medium text-white">@{shellAccount.username}</span>
            <span className="mt-0.5 block text-xs text-zinc-400">{accountPlanLabel}</span>
          </span>
          <ShellIcon name="chevron" className="ml-auto h-3.5 w-3.5 text-zinc-400" />
        </button>
      ) : (
        <button
          type="button"
          role="menuitem"
          onClick={() => openAuthPage("login")}
          className="flex w-full items-center gap-3 rounded-xl px-3 py-3 text-left text-sm text-white transition-colors hover:bg-white/[0.07] focus-visible:outline focus-visible:outline-2 focus-visible:outline-white"
        >
          <ShellIcon name="auth" className="h-[18px] w-[18px] text-zinc-300" />
          Log in / Sign up
        </button>
      )}
      <button
        type="button"
        role="menuitem"
        onClick={() => openSettings("search")}
        className="mt-1 flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left text-sm text-white transition-colors hover:bg-white/[0.07] focus-visible:outline focus-visible:outline-2 focus-visible:outline-white"
      >
        <ShellIcon name="settings" className="h-[18px] w-[18px] text-zinc-300" />
        Settings
      </button>
      {LOCAL_DEMO_AVAILABLE && !demoAccountEnabled ? (
        <button
          type="button"
          role="menuitem"
          onClick={() => {
            setAccountMenuOpen(false);
            router.push("/?demo=account");
          }}
          className="mt-1 flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left text-sm text-white transition-colors hover:bg-white/[0.07] focus-visible:outline focus-visible:outline-2 focus-visible:outline-white"
        >
          <ShellIcon name="auth" className="h-[18px] w-[18px] text-zinc-300" />
          Use demo account
        </button>
      ) : null}
      {LOCAL_DEMO_AVAILABLE ? (
        <button
          type="button"
          role="menuitem"
          onClick={() => {
            setAccountMenuOpen(false);
            router.push("/feed?demo=player&return_tab=search");
          }}
          className="mt-1 flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left text-sm text-white transition-colors hover:bg-white/[0.07] focus-visible:outline focus-visible:outline-2 focus-visible:outline-white"
        >
          <ShellIcon name="chat" className="h-[18px] w-[18px] text-zinc-300" />
          Open demo reel player
        </button>
      ) : null}
      {LOCAL_DEMO_AVAILABLE ? (
        <button
          type="button"
          role="menuitem"
          onClick={() => {
            setAccountMenuOpen(false);
            router.push("/feed?demo=quiz&return_tab=search");
          }}
          className="mt-1 flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left text-sm text-white transition-colors hover:bg-white/[0.07] focus-visible:outline focus-visible:outline-2 focus-visible:outline-white"
        >
          <ShellIcon name="info" className="h-[18px] w-[18px] text-zinc-300" />
          Open demo quiz
        </button>
      ) : null}
      {shellAccount ? (
        <button
          type="button"
          role="menuitem"
          onClick={() => void onAccountMenuSignOut()}
          className="mt-1 flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left text-sm text-white transition-colors hover:bg-white/[0.07] focus-visible:outline focus-visible:outline-2 focus-visible:outline-white"
        >
          <ShellIcon name="auth" className="h-[18px] w-[18px] rotate-180 text-zinc-300" />
          {demoAccountEnabled ? "Exit demo" : "Log out"}
        </button>
      ) : null}
    </div>
  );

  const chatGPTSidebarContent = (
    <div className="flex h-full min-h-0 flex-col px-2 pb-[max(10px,env(safe-area-inset-bottom))] pt-[max(10px,env(safe-area-inset-top))] text-white">
      <div className="flex h-11 shrink-0 items-center gap-1 px-2">
        <p aria-label={brandLabel} data-reelai-wordmark="true" className="flex min-w-0 flex-1 items-baseline gap-1.5 truncate tracking-tight">
          <span aria-hidden="true" className="text-[24px] font-semibold">ReelAI</span>
          {visibleBillingPlan === "plus" || visibleBillingPlan === "pro" ? (
            <span aria-hidden="true" data-plan-suffix="true" className="text-[18px] font-normal text-white/78">
              {visibleBillingPlan === "plus" ? "Plus" : "Pro"}
            </span>
          ) : null}
        </p>
        <button
          type="button"
          onClick={() => {
            setHistoryQuery("");
            setHistorySearchOpen(true);
            setAccountMenuOpen(false);
          }}
          aria-label="Search history"
          className="grid h-9 w-9 place-items-center rounded-lg text-zinc-300 transition-colors hover:bg-white/[0.07] hover:text-white focus-visible:outline focus-visible:outline-2 focus-visible:outline-white"
        >
          <ShellIcon name="search" />
        </button>
        <button
          type="button"
          onClick={toggleDesktopSidebarCollapsed}
          aria-label="Collapse sidebar"
          className="hidden h-9 w-9 place-items-center rounded-lg text-zinc-300 transition-colors hover:bg-white/[0.07] hover:text-white focus-visible:outline focus-visible:outline-2 focus-visible:outline-white lg:grid"
        >
          <ShellIcon name="panel" />
        </button>
        <button
          type="button"
          onClick={closeMobileSidebar}
          aria-label="Close sidebar"
          className="grid h-9 w-9 place-items-center rounded-lg text-zinc-300 transition-colors hover:bg-white/[0.07] hover:text-white focus-visible:outline focus-visible:outline-2 focus-visible:outline-white lg:hidden"
        >
          <ShellIcon name="close" />
        </button>
      </div>

      <nav aria-label="Primary" className="mt-3 shrink-0 space-y-1">
        {sidebarPrimaryActions.map((action) => (
          <button
            key={action.key}
            type="button"
            onClick={action.onClick}
            className={`flex h-11 w-full items-center gap-3 rounded-xl px-3 text-left text-sm transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-white ${
              action.isActive ? "bg-[#2a2a2a] text-white" : "text-zinc-200 hover:bg-white/[0.07]"
            }`}
          >
            <ShellIcon name={action.icon} className="h-[18px] w-[18px]" />
            <span className="truncate">{action.label}</span>
          </button>
        ))}
      </nav>

      <div className="mt-5 min-h-0 flex-1 overflow-y-auto px-1">
        <p className="px-2 pb-2 text-xs font-semibold text-zinc-400">Recents</p>
        {historySorted.length === 0 ? (
          <p className="px-2 py-2 text-xs text-zinc-500">Your searches will appear here.</p>
        ) : (
          <div className="space-y-0.5">
            {historySorted.slice(0, 40).map((entry) => (
              <div key={`recent-${entry.materialId}`} className="group relative">
                <button
                  type="button"
                  onClick={() => openMaterialFeed(entry.materialId)}
                  className="flex h-10 w-full min-w-0 items-center gap-2 rounded-xl px-2.5 pr-10 text-left text-sm text-zinc-300 hover:bg-white/[0.07] hover:text-white focus-visible:outline focus-visible:outline-2 focus-visible:outline-white"
                >
                  {entry.starred ? <ShellIcon name="star" filled className="h-3.5 w-3.5 text-zinc-400" /> : null}
                  <span className="min-w-0 truncate">{entry.title}</span>
                </button>
                <div data-history-actions="true" className="absolute right-1.5 top-1.5">
                  <button
                    type="button"
                    aria-label={`Actions for ${entry.title}`}
                    data-force-visible={activeSidebarHistoryMenuId === entry.materialId ? "true" : undefined}
                    onClick={() => setActiveSidebarHistoryMenuId((current) => current === entry.materialId ? null : entry.materialId)}
                    className="reveal-on-desktop-hover grid h-7 w-7 place-items-center rounded-lg text-zinc-300 transition-colors hover:bg-white/[0.07] hover:text-white"
                  >
                    <ShellIcon name="more" className="h-4 w-4" />
                  </button>
                  <div
                    role="menu"
                    aria-hidden={activeSidebarHistoryMenuId !== entry.materialId}
                    inert={activeSidebarHistoryMenuId !== entry.materialId}
                    className={`absolute right-0 top-8 z-50 w-44 rounded-xl bg-[#202020] p-1.5 transition-opacity duration-300 motion-reduce:transition-none ${
                      activeSidebarHistoryMenuId === entry.materialId
                        ? "pointer-events-auto opacity-100"
                        : "pointer-events-none opacity-0"
                    }`}
                  >
                      <button type="button" role="menuitem" onClick={() => setSelectedHistoryInfoId(entry.materialId)} className="flex w-full items-center gap-2 rounded-lg px-2.5 py-2 text-left text-xs transition-colors hover:bg-white/[0.07]">
                        <ShellIcon name="info" className="h-4 w-4" /> More information
                      </button>
                      <button type="button" role="menuitem" onClick={() => toggleHistoryStar(entry.materialId)} className="mt-0.5 flex w-full items-center gap-2 rounded-lg px-2.5 py-2 text-left text-xs transition-colors hover:bg-white/[0.07]">
                        <ShellIcon name="star" className="h-4 w-4" filled={entry.starred} /> {entry.starred ? "Unstar" : "Star"}
                      </button>
                      <button type="button" role="menuitem" onClick={() => deleteHistoryItem(entry.materialId)} className="mt-0.5 flex w-full items-center gap-2 rounded-lg px-2.5 py-2 text-left text-xs text-red-300 transition-colors hover:bg-white/[0.07]">
                        <ShellIcon name="trash" className="h-4 w-4" /> Delete
                      </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <div data-account-actions="true" className="relative mt-2 shrink-0">
        {shellAccount ? (
          <>
            {renderAccountPopover()}
            <button
              type="button"
              onClick={() => setAccountMenuOpen((current) => !current)}
              aria-haspopup="menu"
              aria-expanded={accountMenuOpen}
              className="flex h-14 w-full items-center gap-3 rounded-xl px-2.5 text-left transition-colors hover:bg-white/[0.07] focus-visible:outline focus-visible:outline-2 focus-visible:outline-white"
            >
              <span className="grid h-9 w-9 shrink-0 place-items-center rounded-full bg-emerald-500 text-sm font-semibold text-white">
                {accountInitial}
              </span>
              <span className="min-w-0 flex-1">
                <span className="block truncate text-sm text-white">@{shellAccount.username}</span>
                <span className="mt-0.5 block text-xs text-zinc-500">{accountPlanLabel}</span>
              </span>
            </button>
          </>
        ) : (
          <button
            type="button"
            onClick={() => openAuthPage("login")}
            data-guest-sidebar-login="true"
            className="flex h-10 w-full items-center gap-2 rounded-lg px-2 text-left text-sm text-white transition-colors hover:bg-white/[0.07]"
          >
            <ShellIcon name="auth" className="h-[18px] w-[18px] text-zinc-300" />
            <span>Login</span>
          </button>
        )}
      </div>
    </div>
  );

  const chatGPTCollapsedSidebarContent = (
    <div className="flex h-full flex-col items-center gap-1 px-2 py-3 text-white">
      <button type="button" onClick={toggleDesktopSidebarCollapsed} aria-label="Expand sidebar" className="group relative grid h-10 w-10 place-items-center rounded-xl text-zinc-300 transition-colors hover:bg-white/[0.07] hover:text-white focus-visible:outline focus-visible:outline-2 focus-visible:outline-white">
        <img
          src="/reelai-mark-white-2.svg"
          alt=""
          aria-hidden="true"
          data-collapsed-sidebar-brand="true"
          className="absolute h-5 w-5 opacity-100 transition-opacity duration-150 ease-out group-hover:opacity-0 group-focus-visible:opacity-0 motion-reduce:transition-none"
        />
        <span
          aria-hidden="true"
          data-collapsed-sidebar-expand-icon="true"
          className="absolute inset-0 grid place-items-center opacity-0 transition-opacity duration-150 ease-out group-hover:opacity-100 group-focus-visible:opacity-100 motion-reduce:transition-none"
        >
          <ShellIcon name="panel" />
        </span>
      </button>
      <button type="button" onClick={() => setHistorySearchOpen(true)} aria-label="Search history" className="grid h-10 w-10 place-items-center rounded-xl text-zinc-300 transition-colors hover:bg-white/[0.07] hover:text-white focus-visible:outline focus-visible:outline-2 focus-visible:outline-white">
        <ShellIcon name="search" />
      </button>
      <div className="mt-2 flex flex-col gap-1">
        {sidebarPrimaryActions.map((action) => (
          <button key={action.key} type="button" onClick={action.onClick} aria-label={action.label} title={action.label} className={`grid h-10 w-10 place-items-center rounded-xl transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-white ${action.isActive ? "bg-[#2a2a2a] text-white" : "text-zinc-300 hover:bg-white/[0.07] hover:text-white"}`}>
            <ShellIcon name={action.icon} />
          </button>
        ))}
      </div>
      <div data-account-actions="true" className="relative mt-auto">
        {shellAccount ? (
          <>
            {renderAccountPopover(true)}
            <button type="button" onClick={() => setAccountMenuOpen((current) => !current)} aria-label={`@${shellAccount.username}`} aria-haspopup="menu" aria-expanded={accountMenuOpen} className="grid h-10 w-10 place-items-center rounded-full bg-emerald-500 text-sm font-semibold text-white focus-visible:outline focus-visible:outline-2 focus-visible:outline-white">
              {accountInitial}
            </button>
          </>
        ) : (
          <button type="button" onClick={() => openAuthPage("login")} aria-label="Login" className="grid h-10 w-10 place-items-center rounded-xl text-zinc-300 transition-colors hover:bg-white/[0.07] hover:text-white">
            <ShellIcon name="auth" />
          </button>
        )}
      </div>
    </div>
  );

  const shellIsInert = shellModalLogicallyOpen || shellModalPresenceActive;

  return (
    <main className="home-hero-shell fixed inset-0 h-[100dvh] overflow-hidden bg-black">
      <div
        aria-hidden={!error}
        className={`pointer-events-none absolute left-0 right-0 top-3 z-30 mx-auto w-fit rounded-full bg-[#2b2b2b] px-4 py-2 text-xs text-white transition-opacity duration-300 motion-reduce:transition-none ${
          error ? "opacity-100" : "opacity-0"
        }`}
      >
        {error}
      </div>
      <div
        aria-hidden="true"
        className={`top-nav-fade pointer-events-none fixed right-0 top-0 z-[109] hidden h-20 transition-[left,opacity] duration-300 motion-reduce:transition-none lg:block ${
          showGenerationSpeedToggle ? "opacity-100" : "opacity-0"
        }`}
        style={{ left: `calc(${desktopSidebarWidthPx}px + 2.5rem)` }}
      />
      <div
        aria-hidden={shellIsInert || mobileSidebarOpen || !showGuestAuthActions}
        inert={shellIsInert || mobileSidebarOpen || !showGuestAuthActions}
        style={{
          right: "calc(max(env(safe-area-inset-right), 0px) + 12px)",
          top: "calc(max(env(safe-area-inset-top), 0px) + 10px)",
        }}
        className={`fixed z-[112] flex h-10 items-center gap-2 transition-opacity duration-300 motion-reduce:transition-none ${
          showGuestAuthActions && !mobileSidebarOpen && !hideMobileTopControls
            ? "opacity-100"
            : "pointer-events-none opacity-0"
        }`}
        data-guest-auth-actions="true"
      >
        <button
          type="button"
          onClick={() => openAuthPage("login")}
          className="h-10 rounded-xl bg-[#2b2b2b] px-4 text-sm font-medium text-white transition-colors hover:bg-white/[0.07]"
        >
          Login
        </button>
        <button
          type="button"
          onClick={() => openAuthPage("register")}
          className="h-10 rounded-xl bg-white px-4 text-sm font-semibold text-black hover:bg-zinc-100"
        >
          Sign up
        </button>
      </div>
      <div
        aria-hidden={shellIsInert || !showGenerationSpeedToggle}
        inert={shellIsInert || !showGenerationSpeedToggle}
        className={`pointer-events-none fixed top-3 z-[112] hidden -translate-x-1/2 transition-[left,opacity] duration-300 ease-out motion-reduce:transition-none lg:block ${
          showGenerationSpeedToggle ? "opacity-100" : "opacity-0"
        }`}
        style={{ left: `calc(50% + ${desktopSidebarWidthPx / 2}px)` }}
      >
        <div className="pointer-events-auto">{fastSlowToggle}</div>
      </div>

      <button
        ref={mobileSidebarTriggerRef}
        type="button"
        data-mobile-sidebar-trigger="true"
        disabled={shellIsInert || mobileSidebarOpen}
        aria-hidden={shellIsInert || mobileSidebarOpen}
        inert={shellIsInert || mobileSidebarOpen}
        onClick={openMobileSidebar}
        aria-label="Open sidebar"
        style={{
          left: "calc(max(env(safe-area-inset-left), 0px) + 10px)",
          top: "calc(max(env(safe-area-inset-top), 0px) + 10px)",
        }}
        className={`fixed z-[110] grid h-10 w-10 place-items-center rounded-xl text-white/90 transition-colors duration-300 motion-reduce:transition-none hover:bg-white/[0.07] hover:text-white focus-visible:outline focus-visible:outline-2 focus-visible:outline-white md:left-7 md:top-7 lg:hidden ${
          mobileSidebarOpen || hideMobileTopControls ? "pointer-events-none opacity-0" : "opacity-100"
        }`}
      >
        <ShellIcon name="menu" className="h-5 w-5" />
      </button>
      <p
        aria-label={brandLabel}
        data-mobile-shell-brand="true"
        className={`fixed left-1/2 z-[110] flex h-10 max-w-[42vw] -translate-x-1/2 items-center gap-1.5 truncate text-[16px] tracking-tight text-white transition-opacity duration-300 motion-reduce:transition-none lg:hidden ${
          mobileSidebarOpen || hideMobileTopControls || shellIsInert
            ? "pointer-events-none opacity-0"
            : "opacity-100"
        }`}
        style={{ top: "calc(max(env(safe-area-inset-top), 0px) + 10px)" }}
      >
        <span aria-hidden="true" className="truncate font-semibold">ReelAI</span>
        {visibleBillingPlan === "plus" || visibleBillingPlan === "pro" ? (
          <span aria-hidden="true" className="shrink-0 font-normal text-white/78">
            {visibleBillingPlan === "plus" ? "Plus" : "Pro"}
          </span>
        ) : null}
      </p>
      <div
        aria-hidden={shellIsInert || mobileSidebarOpen || !showGenerationSpeedToggle}
        inert={shellIsInert || mobileSidebarOpen || !showGenerationSpeedToggle}
        style={{ top: "calc(max(env(safe-area-inset-top), 0px) + 62px)" }}
        className={`fixed left-1/2 z-[110] flex h-11 -translate-x-1/2 items-center transition-opacity duration-300 motion-reduce:transition-none lg:hidden ${
          showGenerationSpeedToggle && !mobileSidebarOpen && !hideMobileTopControls ? "opacity-100" : "pointer-events-none opacity-0"
        }`}
      >
        {fastSlowToggle}
      </div>

      {mobileSidebarOpen ? (
        <div aria-hidden={shellIsInert} inert={shellIsInert} className="fixed left-0 top-0 z-50 h-[100dvh] w-screen lg:hidden">
          <button
            type="button"
            aria-label="Close sidebar overlay"
            onClick={closeMobileSidebar}
            className={`absolute inset-0 bg-black/70 ${mobileSidebarClosing ? "animate-mobile-overlay-out" : "animate-mobile-overlay-in"}`}
          />
          <aside
            data-mobile-drawer="true"
            role="dialog"
            aria-modal="true"
            aria-label="Navigation drawer"
            tabIndex={-1}
            onClick={(event) => {
              if (isSidebarInteractiveTarget(event.target)) {
                return;
              }
              closeMobileSidebar();
            }}
            className={`absolute inset-y-0 left-0 w-[min(272px,86vw)] bg-black text-white ${
              mobileSidebarClosing ? "animate-mobile-sidenav-out" : "animate-mobile-sidenav-in"
            }`}
          >
            {chatGPTSidebarContent}
            <span aria-hidden="true" className="pointer-events-none absolute inset-y-0 right-0 w-px bg-white/[0.07]" />
          </aside>
        </div>
      ) : null}

      <div
        aria-hidden={shellIsInert || mobileSidebarOpen}
        inert={shellIsInert || mobileSidebarOpen}
        className="relative z-20 h-full min-h-0 w-full transition-[grid-template-columns] duration-300 ease-out motion-reduce:transition-none lg:grid"
        style={{ gridTemplateColumns: `minmax(0, ${desktopSidebarWidthPx}px) minmax(0, 1fr)` }}
      >
        <aside
          className="relative z-30 hidden min-h-0 bg-black text-white transition-[width] duration-300 ease-out motion-reduce:transition-none lg:flex"
          style={{ width: `${desktopSidebarWidthPx}px` }}
        >
          <div className="relative h-full min-h-0 w-full">
            <div
              aria-hidden={desktopSidebarCollapsed}
              inert={desktopSidebarCollapsed}
              className={`absolute inset-0 transition-opacity duration-300 motion-reduce:transition-none ${
                desktopSidebarCollapsed ? "pointer-events-none opacity-0" : "opacity-100"
              }`}
            >
              {chatGPTSidebarContent}
            </div>
            <div
              aria-hidden={!desktopSidebarCollapsed}
              inert={!desktopSidebarCollapsed}
              className={`absolute inset-0 transition-opacity duration-300 motion-reduce:transition-none ${
                desktopSidebarCollapsed ? "opacity-100" : "pointer-events-none opacity-0"
              }`}
            >
              {chatGPTCollapsedSidebarContent}
            </div>
          </div>
          <span aria-hidden="true" className="pointer-events-none absolute inset-y-0 right-0 w-px bg-white/[0.07]" />
        </aside>

        <section
          ref={mainContentRef}
          tabIndex={-1}
          aria-label={visibleSidebarTab === "community"
            ? "Community Reels"
            : visibleSidebarTab === "edit" || visibleSidebarTab === "create"
              ? "Your Sets"
              : "New search"}
          className="relative z-20 h-full min-h-0 w-full overflow-hidden bg-black"
        >
          <div
            aria-hidden={visibleSidebarTab !== "search"}
            inert={visibleSidebarTab !== "search"}
            className={`absolute inset-0 h-full min-h-0 transition-opacity duration-300 motion-reduce:transition-none ${
              visibleSidebarTab === "search" ? "opacity-100" : "pointer-events-none opacity-0"
            }`}
          >
            <UploadPanel
              key={composerResetKey}
              active={visibleSidebarTab === "search"}
              account={shellAccount}
              billingStatus={billingStatus}
              demoMode={demoAccountEnabled}
              onMaterialCreated={onUploadMaterialCreated}
              heroTitleRef={searchHeroTitleRef}
            />
          </div>
          <div
            aria-hidden={!isCommunityPanel}
            inert={!isCommunityPanel}
            className={`absolute inset-0 h-full min-h-0 transition-opacity duration-300 motion-reduce:transition-none ${
              isCommunityPanel ? "opacity-100" : "pointer-events-none opacity-0"
            }`}
          >
            <CommunityReelsPanel
              key={demoAccountEnabled ? "local-demo" : "community"}
              mode={communityPanelMode}
              demoMode={demoAccountEnabled}
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
      <ShellModal
          open={historySearchOpen}
          label="Search recent sessions"
          onRequestClose={closeHistorySearch}
          initialFocusRef={historySearchInputRef}
          panelClassName={LARGE_CENTERED_MODAL_PANEL_CLASS}
        >
          <div className="top-nav-fade top-nav-fade-charcoal sticky top-0 z-10 flex shrink-0 items-center gap-3 px-5 pb-4 pt-[max(20px,env(safe-area-inset-top))] sm:px-8 sm:pt-7">
            <ShellIcon name="search" className="h-[18px] w-[18px] text-zinc-400" />
            <input
              ref={historySearchInputRef}
              value={historyQuery}
              onChange={(event) => setHistoryQuery(event.target.value)}
              placeholder="Search..."
              aria-label="Search recent sessions"
              className="h-11 min-w-0 flex-1 bg-transparent text-base text-white outline-none placeholder:text-zinc-500 focus-visible:outline-none sm:text-lg"
            />
            <button
              type="button"
              onClick={closeHistorySearch}
              aria-label="Close search"
              className="grid h-9 w-9 place-items-center rounded-lg text-zinc-300 transition-colors hover:bg-white/[0.07] hover:text-white focus-visible:outline focus-visible:outline-2 focus-visible:outline-white"
            >
              <ShellIcon name="close" />
            </button>
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto px-4 pb-[max(24px,env(safe-area-inset-bottom))] sm:px-8">
            <p className="px-3 pb-3 pt-2 text-sm font-medium text-zinc-400">
              {historyQuery.trim() ? "Search results" : "Recent sessions"}
            </p>
            {filteredHistory.length === 0 ? (
              <div className="grid min-h-52 place-items-center px-6 text-center text-sm text-zinc-500">
                {historyRecent.length === 0 ? "Your recent searches will appear here." : "No sessions match that title."}
              </div>
            ) : (
              <div className="space-y-1">
                {filteredHistory.map((entry) => (
                  <button
                    key={`search-modal-${entry.materialId}`}
                    type="button"
                    onClick={() => {
                      closeHistorySearch();
                      const navigationDelay = window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : UI_FADE_MS;
                      window.setTimeout(() => openMaterialFeed(entry.materialId), navigationDelay);
                    }}
                    className="flex min-h-14 w-full items-center gap-4 rounded-xl px-3 py-3 text-left text-zinc-100 transition-colors hover:bg-white/[0.07] focus-visible:outline focus-visible:outline-2 focus-visible:outline-white"
                  >
                    <ShellIcon name="chat" className="h-[18px] w-[18px] text-zinc-300" />
                    {entry.starred ? <ShellIcon name="star" filled className="h-3.5 w-3.5 text-zinc-400" /> : null}
                    <span className="min-w-0 flex-1 truncate text-sm sm:text-base">{entry.title}</span>
                    <span className="shrink-0 text-xs text-zinc-500">{formatHistoryInfoDate(entry.updatedAt)}</span>
                  </button>
                ))}
              </div>
            )}
          </div>
      </ShellModal>
      <ShellModal
          open={settingsSection !== null}
          label="Settings"
          onRequestClose={onSettingsModalRequestClose}
          panelClassName={LARGE_CENTERED_MODAL_PANEL_CLASS}
        >
          <div className="relative h-full">
            <div
              aria-hidden={settingsClosePrompt}
              inert={settingsClosePrompt}
              className={`h-full transition-opacity duration-300 motion-reduce:transition-none ${
                settingsClosePrompt ? "pointer-events-none opacity-0" : "opacity-100"
              }`}
            >
              <SettingsPanel
                ref={settingsPanelRef}
                account={shellAccount}
                billingStatus={billingStatus}
                billingPlans={billingPlans}
                billingLoading={billingLoading}
                billingError={billingError}
                onBillingRefresh={demoAccountEnabled ? undefined : liveBilling.refresh}
                demoMode={demoAccountEnabled}
                initialSection={settingsSection ?? "search"}
                onSectionChange={onSettingsSectionChange}
                onClose={requestCloseSettings}
                onOpenAuth={requestOpenAuth}
                onAccountChange={setCommunityAccount}
                onClearSearchData={clearAllHistory}
                onSettingsSaved={syncSavedSettings}
                onUnsavedChangesChange={setHasUnsavedSettingsChanges}
                availabilityModalMode="inline"
              />
            </div>
            <div
              aria-hidden={!settingsClosePrompt}
              inert={!settingsClosePrompt}
              className={`absolute inset-0 z-10 flex h-full flex-col justify-between bg-[#202020] p-6 transition-opacity duration-300 motion-reduce:transition-none sm:p-8 ${
                settingsClosePrompt ? "opacity-100" : "pointer-events-none opacity-0"
              }`}
            >
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.12em] text-zinc-400">Unsaved changes</p>
                <h2 className="mt-3 text-2xl font-semibold tracking-tight text-white">Save settings before closing?</h2>
                <p className="mt-3 max-w-lg text-sm leading-6 text-zinc-400">
                  Save to apply your Search and Playback changes, discard them, or keep editing.
                </p>
              </div>
              <div className="mt-8 flex flex-col-reverse gap-2 sm:flex-row sm:justify-end">
                <button
                  ref={settingsPromptFocusRef}
                  type="button"
                  onClick={keepEditingSettings}
                  className="rounded-xl bg-white/[0.07] px-4 py-3 text-sm font-medium text-white transition-colors hover:bg-white/[0.07] focus-visible:outline focus-visible:outline-2 focus-visible:outline-white"
                >
                  Keep Editing
                </button>
                <button
                  type="button"
                  onClick={() => {
                    settingsPanelRef.current?.discardUnsavedChanges();
                    setHasUnsavedSettingsChanges(false);
                    finishCloseSettings(pendingSettingsAuthMode ?? undefined);
                  }}
                  className="rounded-xl bg-white/[0.07] px-4 py-3 text-sm font-medium text-white transition-colors hover:bg-white/[0.07] focus-visible:outline focus-visible:outline-2 focus-visible:outline-white"
                >
                  Discard
                </button>
                <button
                  type="button"
                  onClick={() => {
                    settingsPanelRef.current?.savePreferences();
                    setHasUnsavedSettingsChanges(false);
                    finishCloseSettings(pendingSettingsAuthMode ?? undefined);
                  }}
                  className="rounded-xl bg-white px-5 py-3 text-sm font-semibold text-black hover:bg-zinc-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white"
                >
                  Save
                </button>
              </div>
            </div>
          </div>
      </ShellModal>
      <ShellModal
          open={showUnsavedCommunityDraftModal}
          label="Unsaved set draft changes"
          onRequestClose={closeUnsavedCommunityDraftModal}
          panelClassName="w-full max-w-xl p-5 md:p-6"
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
                  className="inline-flex h-8 w-8 items-center justify-center rounded-lg text-white/80 transition-colors hover:bg-white/[0.07] hover:text-white focus-visible:outline-none"
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
                  className="inline-flex min-w-[9rem] items-center justify-center whitespace-nowrap rounded-xl bg-black/35 px-5 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-white/[0.07]"
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
      </ShellModal>
      <ShellModal
          open={selectedHistoryInfoItem !== null}
          label={`${selectedHistoryInfoItem?.title ?? "Search"} history information`}
          onRequestClose={closeHistoryInfo}
          panelClassName="flex max-h-[calc(100dvh-24px)] w-full max-w-2xl flex-col sm:max-h-[min(760px,calc(100dvh-48px))]"
        >
          {selectedHistoryInfoItem ? (
            <>
              <div className="flex shrink-0 items-start justify-between gap-4 px-5 pt-5 md:px-6 md:pt-6">
                <div>
                  <p className="text-[11px] font-semibold uppercase tracking-[0.12em] text-white/65">More information</p>
                  <h3 className="mt-2 text-lg font-semibold text-white">{selectedHistoryInfoItem.title}</h3>
                </div>
                <button
                  type="button"
                  onClick={closeHistoryInfo}
                  aria-label="Close"
                  className="inline-flex h-8 w-8 items-center justify-center rounded-lg text-white/80 transition-colors hover:bg-white/[0.07] hover:text-white focus-visible:outline-none"
                >
                  <svg viewBox="0 0 20 20" aria-hidden="true" className="h-4 w-4 fill-none stroke-current stroke-2">
                    <path d="M5 5L15 15M15 5L5 15" strokeLinecap="round" />
                  </svg>
                </button>
              </div>
              <div className="mt-4 min-h-0 flex-1 space-y-4 overflow-y-auto overscroll-contain px-5 pb-5 md:px-6 md:pb-6">
                {selectedHistoryInfoSections.map((section) => (
                  <section key={section.title}>
                    <p className="text-[11px] font-semibold uppercase tracking-[0.1em] text-white/58">{section.title}</p>
                    <div className="mt-2 grid gap-2 sm:grid-cols-2">
                      {section.fields.map((field) => (
                        <div
                          key={`${section.title}-${field.label}`}
                          className="rounded-2xl bg-white/[0.045] px-3 py-3"
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
                    <div className="mt-2 rounded-2xl bg-white/[0.045] px-3 py-3">
                      <p className="break-all font-mono text-[11px] leading-5 text-white/72">{selectedHistoryInfoQuery}</p>
                    </div>
                  </section>
                ) : (
                  <div className="rounded-2xl bg-white/[0.045] px-4 py-3 text-sm text-white/72">
                    Detailed feed settings were not saved for this history item.
                  </div>
                )}
              </div>
            </>
          ) : null}
      </ShellModal>
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
