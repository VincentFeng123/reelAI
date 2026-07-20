"use client";

import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useId,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";

import { BillingActions, activeBillingSubscription } from "@/components/BillingActions";
import { CustomSelect } from "@/components/CustomSelect";
import {
  changeCommunityPassword,
  deleteCommunityAccount,
  logoutCommunityAccount,
  readCommunityAuthSession,
  resendCommunityVerification,
  verifyCommunityAccount,
} from "@/lib/api";
import {
  DEFAULT_STUDY_REELS_SETTINGS,
  MAX_RELEVANCE,
  MIN_RELEVANCE,
  readStudyReelsSettings,
  saveStudyReelsSettings,
  subscribeToStudyReelsSettings,
  type PreferredVideoDuration,
  type StudyReelsSettings,
} from "@/lib/settings";
import type { BillingPlan, BillingStatus, CommunityAccount } from "@/lib/types";

export type SettingsSection = "search" | "playback" | "plan" | "data" | "account";

export type SettingsAvailabilityState = {
  status: "idle" | "checking" | "ok" | "partial" | "blocked" | "none" | "error";
  message: string;
  limitingFactors: string[];
};

// Kept as a compatibility export while the page shell moves the settings UI into a modal.
export type SettingsAvailabilityModalSnapshot = {
  mounted: boolean;
  visible: boolean;
  state: SettingsAvailabilityState;
};

export type SettingsPanelProps = {
  account?: CommunityAccount | null;
  billingStatus?: BillingStatus | null;
  billingPlans?: BillingPlan[];
  billingLoading?: boolean;
  billingError?: string | null;
  onBillingRefresh?: () => void | Promise<unknown>;
  demoMode?: boolean;
  initialSection?: SettingsSection;
  onSectionChange?: (section: SettingsSection) => void;
  onClose?: () => void;
  onOpenAuth?: (mode?: "login" | "register" | "verify") => void;
  onAccountChange?: (account: CommunityAccount | null) => void;
  onClearSearchData: () => void;
  onSettingsSaved?: (settings: StudyReelsSettings) => void;
  onUnsavedChangesChange?: (hasUnsavedChanges: boolean) => void;
  // Legacy shell hooks remain optional until old links have fully bridged to the modal.
  onAvailabilityModalClose?: (source: "close-button" | "backdrop") => void;
  availabilityModalMode?: "overlay" | "inline";
  onAvailabilityModalStateChange?: (snapshot: SettingsAvailabilityModalSnapshot | null) => void;
};

export type SettingsPanelHandle = {
  savePreferences: () => void;
  discardUnsavedChanges: () => void;
  hasUnsavedChanges: () => boolean;
  dismissAvailabilityModal: (source: "close-button" | "backdrop") => void;
};

type PreferenceDraft = Pick<
  StudyReelsSettings,
  | "minRelevanceThreshold"
  | "creativeCommonsOnly"
  | "preferredVideoDuration"
  | "startMuted"
  | "autoplayNextReel"
>;

const COMMUNITY_CACHE_KEYS = [
  "studyreels-community-sets",
  "studyreels-community-create-draft",
  "studyreels-community-starred-set-ids",
];

const SECTION_OPTIONS: ReadonlyArray<{
  id: SettingsSection;
  label: string;
  icon: "search" | "playback" | "plan" | "data" | "account";
}> = [
  { id: "search", label: "Search", icon: "search" },
  { id: "playback", label: "Playback", icon: "playback" },
  { id: "plan", label: "Plan & Usage", icon: "plan" },
  { id: "data", label: "Data Controls", icon: "data" },
  { id: "account", label: "Account", icon: "account" },
];

const DURATION_OPTIONS: ReadonlyArray<{ value: PreferredVideoDuration; label: string }> = [
  { value: "any", label: "Any length" },
  { value: "short", label: "Short" },
  { value: "medium", label: "Medium" },
  { value: "long", label: "Long" },
];

const RELEVANCE_STEP = 0.02;
const RELEVANCE_DIAL_START_DEG = -135;
const RELEVANCE_DIAL_END_DEG = 135;
const RELEVANCE_DIAL_SPAN_DEG = RELEVANCE_DIAL_END_DEG - RELEVANCE_DIAL_START_DEG;
const RELEVANCE_DIAL_SIZE_PX = 160;
const RELEVANCE_DIAL_CENTER_PX = RELEVANCE_DIAL_SIZE_PX / 2;
const RELEVANCE_DIAL_RADIUS_PX = 50;
const RELEVANCE_DIAL_CIRCUMFERENCE = 2 * Math.PI * RELEVANCE_DIAL_RADIUS_PX;
const RELEVANCE_DIAL_ARC_LENGTH = RELEVANCE_DIAL_CIRCUMFERENCE * (RELEVANCE_DIAL_SPAN_DEG / 360);
const ACCOUNT_VIEW_FADE_MS = 420;

function preferencesFromSettings(settings: StudyReelsSettings): PreferenceDraft {
  return {
    minRelevanceThreshold: settings.minRelevanceThreshold,
    creativeCommonsOnly: settings.creativeCommonsOnly,
    preferredVideoDuration: settings.preferredVideoDuration,
    startMuted: settings.startMuted,
    autoplayNextReel: settings.autoplayNextReel,
  };
}

function preferenceDraftsMatch(left: PreferenceDraft, right: PreferenceDraft): boolean {
  return (
    Number(left.minRelevanceThreshold.toFixed(2)) === Number(right.minRelevanceThreshold.toFixed(2))
    && left.creativeCommonsOnly === right.creativeCommonsOnly
    && left.preferredVideoDuration === right.preferredVideoDuration
    && left.startMuted === right.startMuted
    && left.autoplayNextReel === right.autoplayNextReel
  );
}

export function buildSettingsAvailabilityState(settings: StudyReelsSettings): SettingsAvailabilityState {
  const limitingFactors: string[] = [];
  if (settings.minRelevanceThreshold >= 0.45) {
    limitingFactors.push("a strict similarity threshold");
  }
  if (settings.preferredVideoDuration === "long") {
    limitingFactors.push("long source videos only");
  }
  if (settings.creativeCommonsOnly) {
    limitingFactors.push("Creative Commons licensing");
  }
  if (limitingFactors.length > 0) {
    return {
      status: "partial",
      message: "These filters may narrow source availability for some searches.",
      limitingFactors,
    };
  }
  return {
    status: "ok",
    message: "This configuration should work for most searches.",
    limitingFactors: [],
  };
}

function formatResetTime(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "the next daily reset";
  }
  return new Intl.DateTimeFormat(undefined, {
    hour: "numeric",
    minute: "2-digit",
    timeZone: "UTC",
    timeZoneName: "short",
  }).format(date);
}

function formatPlanName(plan: BillingStatus["plan"]): string {
  return plan === "free" ? "ReelAI" : `ReelAI ${plan.charAt(0).toUpperCase()}${plan.slice(1)}`;
}

function SectionIcon({ icon }: { icon: (typeof SECTION_OPTIONS)[number]["icon"] }) {
  if (icon === "search") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true" className="h-[18px] w-[18px] fill-none stroke-current stroke-[1.5]">
        <circle cx="11" cy="11" r="6.5" />
        <path d="m16 16 4 4" strokeLinecap="round" />
      </svg>
    );
  }
  if (icon === "playback") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true" className="h-[18px] w-[18px] fill-none stroke-current stroke-[1.5]">
        <rect x="3" y="5" width="18" height="14" rx="3" />
        <path d="m10 9 5 3-5 3V9Z" fill="currentColor" stroke="none" />
      </svg>
    );
  }
  if (icon === "plan") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true" className="h-[18px] w-[18px] fill-none stroke-current stroke-[1.5]">
        <rect x="3" y="5" width="18" height="14" rx="3" />
        <path d="M3 10h18M7 15h4" strokeLinecap="round" />
      </svg>
    );
  }
  if (icon === "data") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true" className="h-[18px] w-[18px] fill-none stroke-current stroke-[1.5]">
        <ellipse cx="12" cy="6" rx="7" ry="3" />
        <path d="M5 6v6c0 1.7 3.1 3 7 3s7-1.3 7-3V6M5 12v6c0 1.7 3.1 3 7 3s7-1.3 7-3v-6" />
      </svg>
    );
  }
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className="h-[18px] w-[18px] fill-none stroke-current stroke-[1.5]">
      <circle cx="12" cy="8" r="4" />
      <path d="M4.8 21a7.2 7.2 0 0 1 14.4 0" strokeLinecap="round" />
    </svg>
  );
}

function SettingsSwitch({
  checked,
  label,
  onChange,
}: {
  checked: boolean;
  label: string;
  onChange: (checked: boolean) => void;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-label={label}
      aria-checked={checked}
      onClick={() => onChange(!checked)}
      className={`relative inline-flex h-7 w-12 shrink-0 rounded-full transition-colors duration-300 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white ${
        checked ? "bg-white" : "bg-white/20"
      }`}
    >
      <span
        aria-hidden="true"
        className={`absolute top-[3px] h-[22px] w-[22px] rounded-full transition-transform duration-300 ease-out motion-reduce:transition-none ${
          checked ? "translate-x-[23px] bg-black" : "translate-x-[3px] bg-white"
        }`}
      />
    </button>
  );
}

function SimilarityDial({ value, onChange }: { value: number; onChange: (value: number) => void }) {
  const dialRef = useRef<HTMLDivElement | null>(null);
  const ratio = Math.max(0, Math.min(1, (value - MIN_RELEVANCE) / (MAX_RELEVANCE - MIN_RELEVANCE)));
  const angleDeg = RELEVANCE_DIAL_START_DEG + ratio * RELEVANCE_DIAL_SPAN_DEG;
  const angleRad = (angleDeg * Math.PI) / 180;
  const knobX = Math.cos(angleRad) * RELEVANCE_DIAL_RADIUS_PX;
  const knobY = Math.sin(angleRad) * RELEVANCE_DIAL_RADIUS_PX;
  const progressLength = RELEVANCE_DIAL_ARC_LENGTH * ratio;
  const bandLabel = value >= 0.45 ? "Strict" : value <= 0.18 ? "Loose" : "Balanced";

  const clampValue = useCallback((nextValue: number) => {
    const snapped = Math.round(nextValue / RELEVANCE_STEP) * RELEVANCE_STEP;
    return Number(Math.max(MIN_RELEVANCE, Math.min(MAX_RELEVANCE, snapped)).toFixed(2));
  }, []);

  const setFromPoint = useCallback((clientX: number, clientY: number) => {
    const element = dialRef.current;
    if (!element) {
      return;
    }
    const rect = element.getBoundingClientRect();
    const rawAngle = (Math.atan2(clientY - (rect.top + rect.height / 2), clientX - (rect.left + rect.width / 2)) * 180) / Math.PI;
    const boundedAngle = rawAngle > RELEVANCE_DIAL_END_DEG
      ? RELEVANCE_DIAL_END_DEG
      : rawAngle < RELEVANCE_DIAL_START_DEG
        ? RELEVANCE_DIAL_START_DEG
        : rawAngle;
    const nextRatio = (boundedAngle - RELEVANCE_DIAL_START_DEG) / RELEVANCE_DIAL_SPAN_DEG;
    onChange(clampValue(MIN_RELEVANCE + nextRatio * (MAX_RELEVANCE - MIN_RELEVANCE)));
  }, [clampValue, onChange]);

  const adjustValue = useCallback((delta: number) => {
    onChange(clampValue(value + delta));
  }, [clampValue, onChange, value]);

  const looseAngle = (RELEVANCE_DIAL_START_DEG * Math.PI) / 180;
  const strictAngle = (RELEVANCE_DIAL_END_DEG * Math.PI) / 180;
  const endpointRadius = RELEVANCE_DIAL_RADIUS_PX + 20;

  return (
    <div className="relative h-40 w-40 shrink-0" data-similarity-dial="true">
      <svg aria-hidden="true" className="absolute inset-0 h-full w-full" viewBox={`0 0 ${RELEVANCE_DIAL_SIZE_PX} ${RELEVANCE_DIAL_SIZE_PX}`}>
        <circle
          cx={RELEVANCE_DIAL_CENTER_PX}
          cy={RELEVANCE_DIAL_CENTER_PX}
          r={RELEVANCE_DIAL_RADIUS_PX}
          fill="none"
          stroke="rgba(255,255,255,0.17)"
          strokeWidth="7"
          strokeLinecap="round"
          strokeDasharray={`${RELEVANCE_DIAL_ARC_LENGTH} ${RELEVANCE_DIAL_CIRCUMFERENCE}`}
          transform={`rotate(225 ${RELEVANCE_DIAL_CENTER_PX} ${RELEVANCE_DIAL_CENTER_PX})`}
        />
        <circle
          cx={RELEVANCE_DIAL_CENTER_PX}
          cy={RELEVANCE_DIAL_CENTER_PX}
          r={RELEVANCE_DIAL_RADIUS_PX}
          fill="none"
          stroke="rgba(255,255,255,0.92)"
          strokeWidth="7"
          strokeLinecap="round"
          strokeDasharray={`${progressLength} ${RELEVANCE_DIAL_CIRCUMFERENCE}`}
          transform={`rotate(225 ${RELEVANCE_DIAL_CENTER_PX} ${RELEVANCE_DIAL_CENTER_PX})`}
        />
      </svg>
      <div
        ref={dialRef}
        role="slider"
        tabIndex={0}
        aria-label="Similarity threshold"
        aria-valuemin={MIN_RELEVANCE}
        aria-valuemax={MAX_RELEVANCE}
        aria-valuenow={Number(value.toFixed(2))}
        aria-valuetext={`${value.toFixed(2)} similarity threshold`}
        onPointerDown={(event) => {
          if (event.button !== 0) {
            return;
          }
          event.preventDefault();
          event.currentTarget.setPointerCapture(event.pointerId);
          setFromPoint(event.clientX, event.clientY);
        }}
        onPointerMove={(event) => {
          if (event.currentTarget.hasPointerCapture(event.pointerId)) {
            setFromPoint(event.clientX, event.clientY);
          }
        }}
        onPointerUp={(event) => {
          if (event.currentTarget.hasPointerCapture(event.pointerId)) {
            event.currentTarget.releasePointerCapture(event.pointerId);
          }
        }}
        onPointerCancel={(event) => {
          if (event.currentTarget.hasPointerCapture(event.pointerId)) {
            event.currentTarget.releasePointerCapture(event.pointerId);
          }
        }}
        onKeyDown={(event) => {
          if (event.key === "ArrowLeft" || event.key === "ArrowDown") {
            event.preventDefault();
            adjustValue(-RELEVANCE_STEP);
          } else if (event.key === "ArrowRight" || event.key === "ArrowUp") {
            event.preventDefault();
            adjustValue(RELEVANCE_STEP);
          } else if (event.key === "Home") {
            event.preventDefault();
            onChange(MIN_RELEVANCE);
          } else if (event.key === "End") {
            event.preventDefault();
            onChange(MAX_RELEVANCE);
          }
        }}
        className="absolute inset-0 cursor-grab rounded-full outline-none active:cursor-grabbing focus-visible:outline focus-visible:outline-1 focus-visible:outline-offset-1 focus-visible:outline-white/65"
        style={{ touchAction: "none" }}
      >
        <span
          aria-hidden="true"
          className="pointer-events-none absolute left-1/2 top-1/2 h-4 w-4 rounded-full bg-white"
          style={{ transform: `translate(calc(-50% + ${knobX}px), calc(-50% + ${knobY}px))` }}
        />
        <span
          aria-hidden="true"
          className="pointer-events-none absolute left-1/2 top-1/2 text-[10px] font-medium text-white/45"
          style={{ transform: `translate(calc(-50% + ${Math.cos(looseAngle) * endpointRadius - 11}px), calc(-50% + ${Math.sin(looseAngle) * endpointRadius}px))` }}
        >
          Loose
        </span>
        <span
          aria-hidden="true"
          className="pointer-events-none absolute left-1/2 top-1/2 text-[10px] font-medium text-white/45"
          style={{ transform: `translate(calc(-50% + ${Math.cos(strictAngle) * endpointRadius - 11}px), calc(-50% + ${Math.sin(strictAngle) * endpointRadius}px))` }}
        >
          Strict
        </span>
        <span aria-hidden="true" className="pointer-events-none absolute left-1/2 top-1/2 flex -translate-x-1/2 -translate-y-1/2 flex-col items-center">
          <span className="text-sm font-semibold tabular-nums text-white">{value.toFixed(2)}+</span>
          <span className="mt-0.5 text-[9px] font-medium uppercase tracking-[0.09em] text-white/48">{bandLabel}</span>
        </span>
      </div>
    </div>
  );
}

function SettingsRow({
  title,
  description,
  children,
}: {
  title: string;
  description?: string;
  children: ReactNode;
}) {
  return (
    <div className="flex min-h-[68px] items-center justify-between gap-4 rounded-2xl bg-white/[0.035] px-3 py-3 sm:px-4">
      <div className="min-w-0">
        <p className="text-[15px] font-medium text-white">{title}</p>
        {description ? <p className="mt-0.5 max-w-[34rem] text-xs leading-4 text-white/52">{description}</p> : null}
      </div>
      {children}
    </div>
  );
}

function PasswordField({
  label,
  value,
  onChange,
  autoComplete,
  placeholder,
  ariaLabel,
  className = "",
}: {
  label?: string;
  value: string;
  onChange: (value: string) => void;
  autoComplete: "current-password" | "new-password";
  placeholder?: string;
  ariaLabel?: string;
  className?: string;
}) {
  const inputId = useId();
  const [revealed, setRevealed] = useState(false);
  const accessibleName = ariaLabel || label || "Password";

  return (
    <div className={`grid w-full gap-1.5 ${className}`}>
      {label ? (
        <label htmlFor={inputId} className="text-xs font-medium text-white/62">
          {label}
        </label>
      ) : null}
      <div className="relative min-w-0">
        <input
          id={inputId}
          type={revealed ? "text" : "password"}
          value={value}
          onChange={(event) => onChange(event.target.value)}
          autoComplete={autoComplete}
          placeholder={placeholder}
          aria-label={label ? undefined : accessibleName}
          className="h-11 w-full rounded-xl bg-white/[0.08] px-3 pr-11 text-sm text-white outline-none placeholder:text-white/32"
        />
        <button
          type="button"
          onClick={() => setRevealed((current) => !current)}
          aria-label={`${revealed ? "Hide" : "Show"} ${accessibleName}`}
          aria-pressed={revealed}
          className="absolute right-1 top-1/2 grid h-9 w-9 -translate-y-1/2 place-items-center bg-transparent text-white/48 transition-colors hover:text-white/82"
        >
          <svg viewBox="0 0 24 24" aria-hidden="true" className="h-[18px] w-[18px] fill-none stroke-current stroke-[1.5]">
            <path d="M2.8 12s3.3-5.2 9.2-5.2S21.2 12 21.2 12s-3.3 5.2-9.2 5.2S2.8 12 2.8 12Z" strokeLinecap="round" strokeLinejoin="round" />
            <circle cx="12" cy="12" r="2.4" />
            {revealed ? <path d="m4 4 16 16" strokeLinecap="round" /> : null}
          </svg>
        </button>
      </div>
    </div>
  );
}

function SectionHeading({ id, title, detail }: { id: string; title: string; detail: string }) {
  return (
    <header className="mb-5">
      <h2 id={id} className="text-2xl font-semibold tracking-[-0.035em] text-white">{title}</h2>
      <p className="mt-1.5 max-w-2xl text-sm leading-5 text-white/52">{detail}</p>
    </header>
  );
}

export const SettingsPanel = forwardRef<SettingsPanelHandle, SettingsPanelProps>(function SettingsPanel(
  {
    account: accountProp,
    billingStatus = null,
    billingPlans = [],
    billingLoading = false,
    billingError = null,
    onBillingRefresh,
    demoMode = false,
    initialSection = "search",
    onSectionChange,
    onClose,
    onOpenAuth,
    onAccountChange,
    onClearSearchData,
    onSettingsSaved,
    onUnsavedChangesChange,
    onAvailabilityModalClose,
    onAvailabilityModalStateChange,
  },
  ref,
) {
  const initialSettings = useMemo(() => readStudyReelsSettings(), []);
  const [savedPreferences, setSavedPreferences] = useState<StudyReelsSettings>(initialSettings);
  const [draft, setDraft] = useState<PreferenceDraft>(() => preferencesFromSettings(initialSettings));
  const [settingsHydrated, setSettingsHydrated] = useState(false);
  const [activeSection, setActiveSection] = useState<SettingsSection>(initialSection);
  const [availabilityState, setAvailabilityState] = useState<SettingsAvailabilityState>({
    status: "idle",
    message: "",
    limitingFactors: [],
  });
  const [notice, setNotice] = useState<string | null>(null);
  const [billingActionError, setBillingActionError] = useState<string | null>(null);
  const [resolvedAccount, setResolvedAccount] = useState<CommunityAccount | null>(() => (
    accountProp === undefined ? readCommunityAuthSession()?.account ?? null : accountProp
  ));
  const [accountBusy, setAccountBusy] = useState<string | null>(null);
  const [accountError, setAccountError] = useState<string | null>(null);
  const [accountNotice, setAccountNotice] = useState<string | null>(null);
  const [accountView, setAccountView] = useState<"overview" | "password">("overview");
  const [accountContentVisible, setAccountContentVisible] = useState(true);
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmNewPassword, setConfirmNewPassword] = useState("");
  const [deletePassword, setDeletePassword] = useState("");
  const [deleteExpanded, setDeleteExpanded] = useState(false);
  const [verificationCode, setVerificationCode] = useState("");
  const [verificationCodeDebug, setVerificationCodeDebug] = useState<string | null>(null);
  const noticeTimerRef = useRef<number | null>(null);
  const accountViewTimerRef = useRef<number | null>(null);
  const accountViewFrameRef = useRef<number | null>(null);
  const dirtyRef = useRef(false);

  const hasUnsavedChanges = useMemo(
    () => !preferenceDraftsMatch(draft, preferencesFromSettings(savedPreferences)),
    [draft, savedPreferences],
  );
  dirtyRef.current = settingsHydrated && hasUnsavedChanges;

  const showNotice = useCallback((message: string) => {
    setNotice(message);
    if (typeof window === "undefined") {
      return;
    }
    if (noticeTimerRef.current !== null) {
      window.clearTimeout(noticeTimerRef.current);
    }
    noticeTimerRef.current = window.setTimeout(() => {
      noticeTimerRef.current = null;
      setNotice((current) => (current === message ? null : current));
    }, 2600);
  }, []);

  const updateAccount = useCallback((nextAccount: CommunityAccount | null) => {
    setResolvedAccount(nextAccount);
    onAccountChange?.(nextAccount);
  }, [onAccountChange]);

  const clearAccountViewTransition = useCallback(() => {
    if (typeof window === "undefined") {
      return;
    }
    if (accountViewTimerRef.current !== null) {
      window.clearTimeout(accountViewTimerRef.current);
      accountViewTimerRef.current = null;
    }
    if (accountViewFrameRef.current !== null) {
      window.cancelAnimationFrame(accountViewFrameRef.current);
      accountViewFrameRef.current = null;
    }
  }, []);

  const switchAccountView = useCallback((nextView: "overview" | "password") => {
    clearAccountViewTransition();
    if (nextView === "password") {
      setAccountError(null);
      setAccountNotice(null);
    }
    setAccountContentVisible(false);

    if (typeof window === "undefined") {
      setAccountView(nextView);
      setAccountContentVisible(true);
      return;
    }

    const delay = window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : ACCOUNT_VIEW_FADE_MS;
    accountViewTimerRef.current = window.setTimeout(() => {
      accountViewTimerRef.current = null;
      setAccountView(nextView);
      if (nextView === "overview") {
        setCurrentPassword("");
        setNewPassword("");
        setConfirmNewPassword("");
      }
      accountViewFrameRef.current = window.requestAnimationFrame(() => {
        accountViewFrameRef.current = null;
        setAccountContentVisible(true);
      });
    }, delay);
  }, [clearAccountViewTransition]);

  useEffect(() => {
    if (accountProp !== undefined) {
      setResolvedAccount(accountProp);
      return;
    }
    setResolvedAccount(readCommunityAuthSession()?.account ?? null);
  }, [accountProp]);

  useEffect(() => {
    setActiveSection(initialSection);
  }, [initialSection]);

  useEffect(() => {
    const stored = readStudyReelsSettings();
    setSavedPreferences(stored);
    setDraft(preferencesFromSettings(stored));
    setSettingsHydrated(true);
    return subscribeToStudyReelsSettings((next) => {
      setSavedPreferences(next);
      if (!dirtyRef.current) {
        setDraft(preferencesFromSettings(next));
      }
    });
  }, []);

  useEffect(() => {
    onUnsavedChangesChange?.(settingsHydrated && hasUnsavedChanges);
  }, [hasUnsavedChanges, onUnsavedChangesChange, settingsHydrated]);

  useEffect(() => {
    onAvailabilityModalStateChange?.(null);
  }, [onAvailabilityModalStateChange]);

  useEffect(() => () => {
    if (typeof window !== "undefined" && noticeTimerRef.current !== null) {
      window.clearTimeout(noticeTimerRef.current);
    }
    clearAccountViewTransition();
  }, [clearAccountViewTransition]);

  useEffect(() => {
    if (activeSection === "account") {
      return;
    }
    clearAccountViewTransition();
    setAccountView("overview");
    setAccountContentVisible(true);
    setCurrentPassword("");
    setNewPassword("");
    setConfirmNewPassword("");
  }, [activeSection, clearAccountViewTransition]);

  const selectSection = useCallback((section: SettingsSection) => {
    setActiveSection(section);
    onSectionChange?.(section);
  }, [onSectionChange]);

  const savePreferences = useCallback(() => {
    if (!settingsHydrated) {
      return;
    }
    // Hidden legacy values and the global Fast/Slow choice remain authoritative.
    const currentSettings = readStudyReelsSettings();
    const saved = saveStudyReelsSettings({
      ...currentSettings,
      minRelevanceThreshold: Number(draft.minRelevanceThreshold.toFixed(2)),
      creativeCommonsOnly: draft.creativeCommonsOnly,
      preferredVideoDuration: draft.preferredVideoDuration,
      startMuted: draft.startMuted,
      autoplayNextReel: draft.autoplayNextReel,
    });
    setSavedPreferences(saved);
    setDraft(preferencesFromSettings(saved));
    setAvailabilityState(buildSettingsAvailabilityState(saved));
    showNotice("Settings saved");
    onSettingsSaved?.(saved);
  }, [draft, onSettingsSaved, settingsHydrated, showNotice]);

  const discardUnsavedChanges = useCallback(() => {
    const current = readStudyReelsSettings();
    setSavedPreferences(current);
    setDraft(preferencesFromSettings(current));
  }, []);

  useImperativeHandle(ref, () => ({
    savePreferences,
    discardUnsavedChanges,
    hasUnsavedChanges: () => dirtyRef.current,
    dismissAvailabilityModal: (source) => onAvailabilityModalClose?.(source),
  }), [discardUnsavedChanges, onAvailabilityModalClose, savePreferences]);

  const resetDefaults = useCallback(() => {
    setDraft({
      minRelevanceThreshold: DEFAULT_STUDY_REELS_SETTINGS.minRelevanceThreshold,
      creativeCommonsOnly: DEFAULT_STUDY_REELS_SETTINGS.creativeCommonsOnly,
      preferredVideoDuration: DEFAULT_STUDY_REELS_SETTINGS.preferredVideoDuration,
      startMuted: DEFAULT_STUDY_REELS_SETTINGS.startMuted,
      autoplayNextReel: DEFAULT_STUDY_REELS_SETTINGS.autoplayNextReel,
    });
    setAvailabilityState({
      status: "idle",
      message: "Defaults loaded. Save to apply them.",
      limitingFactors: [],
    });
    showNotice("Defaults loaded — save to apply");
  }, [showNotice]);

  const clearSetCache = useCallback(() => {
    if (demoMode) {
      showNotice("Demo cache left unchanged");
      return;
    }
    if (typeof window !== "undefined") {
      for (const key of COMMUNITY_CACHE_KEYS) {
        window.localStorage.removeItem(key);
      }
      for (let index = window.localStorage.length - 1; index >= 0; index -= 1) {
        const key = window.localStorage.key(index);
        if (key?.startsWith("studyreels-community-edit-draft-") || key?.startsWith("studyreels-community-return-set-")) {
          window.localStorage.removeItem(key);
        }
      }
    }
    showNotice("Set cache cleared");
  }, [demoMode, showNotice]);

  const handlePasswordChange = useCallback(async () => {
    if (demoMode) {
      setAccountNotice("Password changes are disabled for the demo account.");
      setAccountError(null);
      return;
    }
    if (accountBusy) {
      return;
    }
    if (!currentPassword || !newPassword || !confirmNewPassword) {
      setAccountError("Complete all three password fields.");
      return;
    }
    if (newPassword.length < 8) {
      setAccountError("New password must be at least 8 characters.");
      return;
    }
    if (newPassword !== confirmNewPassword) {
      setAccountError("New passwords do not match.");
      return;
    }
    setAccountBusy("password");
    setAccountError(null);
    setAccountNotice(null);
    try {
      await changeCommunityPassword({ currentPassword, newPassword });
      setCurrentPassword("");
      setNewPassword("");
      setConfirmNewPassword("");
      setAccountNotice("Password updated.");
      switchAccountView("overview");
    } catch (error) {
      setAccountError(error instanceof Error ? error.message : "Could not change password.");
    } finally {
      setAccountBusy(null);
    }
  }, [accountBusy, confirmNewPassword, currentPassword, demoMode, newPassword, switchAccountView]);

  const handleSignOut = useCallback(async () => {
    if (demoMode) {
      setAccountNotice("Use Exit demo from the account menu to return to your normal session.");
      setAccountError(null);
      return;
    }
    if (accountBusy) {
      return;
    }
    setAccountBusy("signout");
    setAccountError(null);
    try {
      await logoutCommunityAccount();
      updateAccount(null);
    } catch (error) {
      setAccountError(error instanceof Error ? error.message : "Could not sign out.");
    } finally {
      setAccountBusy(null);
    }
  }, [accountBusy, demoMode, updateAccount]);

  const handleDeleteAccount = useCallback(async () => {
    if (demoMode) {
      setAccountNotice("Account deletion is disabled for the demo account.");
      setAccountError(null);
      return;
    }
    if (!deletePassword || accountBusy) {
      setAccountError("Enter your current password to delete this account.");
      return;
    }
    setAccountBusy("delete");
    setAccountError(null);
    try {
      await deleteCommunityAccount({ currentPassword: deletePassword });
      setDeletePassword("");
      updateAccount(null);
    } catch (error) {
      setAccountError(error instanceof Error ? error.message : "Could not delete this account.");
    } finally {
      setAccountBusy(null);
    }
  }, [accountBusy, deletePassword, demoMode, updateAccount]);

  const handleSendVerification = useCallback(async () => {
    if (demoMode) {
      setAccountNotice("The demo account is already verified.");
      setAccountError(null);
      return;
    }
    if (accountBusy) {
      return;
    }
    setAccountBusy("send-verification");
    setAccountError(null);
    setAccountNotice(null);
    try {
      const result = await resendCommunityVerification();
      updateAccount(result.account);
      setVerificationCodeDebug(result.verificationCodeDebug);
      setAccountNotice(`Verification code sent${result.account.email ? ` to ${result.account.email}` : ""}.`);
    } catch (error) {
      setAccountError(error instanceof Error ? error.message : "Could not send a verification code.");
    } finally {
      setAccountBusy(null);
    }
  }, [accountBusy, demoMode, updateAccount]);

  const handleVerify = useCallback(async () => {
    if (demoMode) {
      setAccountNotice("The demo account is already verified.");
      setAccountError(null);
      return;
    }
    if (!verificationCode.trim() || accountBusy) {
      setAccountError("Enter the verification code.");
      return;
    }
    setAccountBusy("verify");
    setAccountError(null);
    setAccountNotice(null);
    try {
      const result = await verifyCommunityAccount({ code: verificationCode.trim() });
      updateAccount(result.account);
      setVerificationCode("");
      setVerificationCodeDebug(null);
      setAccountNotice("Email verified.");
    } catch (error) {
      setAccountError(error instanceof Error ? error.message : "Could not verify this account.");
    } finally {
      setAccountBusy(null);
    }
  }, [accountBusy, demoMode, updateAccount, verificationCode]);

  const subscription = activeBillingSubscription(billingStatus);
  const percentUsed = !billingStatus || billingStatus.daily_limit <= 0
    ? 0
    : Math.min(100, Math.max(0, (billingStatus.used_searches / billingStatus.daily_limit) * 100));

  return (
    <div className="settings-surface flex h-full min-h-0 w-full flex-col overflow-hidden bg-[#202020] text-white md:flex-row">
      <aside className="hidden w-[180px] shrink-0 bg-[#191919] px-3 py-4 md:flex md:flex-col">
        {onClose ? (
          <button
            type="button"
            onClick={onClose}
            aria-label="Close settings"
            className="mb-4 inline-flex h-10 w-10 items-center justify-center rounded-xl text-white/72 transition-colors hover:bg-white/[0.07] hover:text-white focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white"
          >
            <svg viewBox="0 0 20 20" aria-hidden="true" className="h-4 w-4 fill-none stroke-current stroke-[1.5]">
              <path d="M4.5 4.5 15.5 15.5M15.5 4.5 4.5 15.5" strokeLinecap="round" />
            </svg>
          </button>
        ) : null}
        <nav aria-label="Settings categories" className="space-y-1">
          {SECTION_OPTIONS.map((section) => {
            const selected = activeSection === section.id;
            return (
              <button
                key={section.id}
                type="button"
                aria-current={selected ? "page" : undefined}
                onClick={() => selectSection(section.id)}
                className={`flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left text-sm transition focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white ${
                  selected ? "bg-white/[0.11] text-white" : "text-white/68 hover:bg-white/[0.07] hover:text-white"
                }`}
              >
                <SectionIcon icon={section.icon} />
                <span className="truncate">{section.label}</span>
              </button>
            );
          })}
        </nav>
      </aside>

      <div className="relative flex min-h-0 flex-1 flex-col">
        <div className="top-nav-fade top-nav-fade-charcoal sticky top-0 z-20 flex shrink-0 items-center gap-2 px-3 pb-2 pt-[max(0.75rem,env(safe-area-inset-top))] md:hidden">
          {onClose ? (
            <button
              type="button"
              onClick={onClose}
              aria-label="Close settings"
              className="inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-full text-white/72 transition-colors hover:bg-white/[0.07] hover:text-white focus-visible:outline focus-visible:outline-2 focus-visible:outline-white"
            >
              <svg viewBox="0 0 20 20" aria-hidden="true" className="h-4 w-4 fill-none stroke-current stroke-[1.5]">
                <path d="M4.5 4.5 15.5 15.5M15.5 4.5 4.5 15.5" strokeLinecap="round" />
              </svg>
            </button>
          ) : null}
          <nav aria-label="Settings categories" className="flex min-w-0 flex-1 snap-x gap-1 overflow-x-auto">
            {SECTION_OPTIONS.map((section) => (
              <button
                key={section.id}
                type="button"
                aria-current={activeSection === section.id ? "page" : undefined}
                onClick={() => selectSection(section.id)}
                className={`shrink-0 snap-start rounded-full px-3.5 py-2 text-xs font-medium transition focus-visible:outline focus-visible:outline-2 focus-visible:outline-white ${
                  activeSection === section.id ? "bg-white text-black" : "bg-white/[0.06] text-white/65"
                }`}
              >
                {section.label}
              </button>
            ))}
          </nav>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain px-3 pb-[calc(5.5rem+env(safe-area-inset-bottom))] pt-4 sm:px-5 md:px-6 md:pb-24 md:pt-5">
          <div className="mx-auto w-full max-w-[720px]">
            {activeSection === "search" ? (
              <section aria-labelledby="settings-search-title">
                <SectionHeading id="settings-search-title" title="Search" detail="Choose how ReelAI filters source videos. Changes apply after you save." />
                <div className="space-y-1.5">
                  <div className="flex items-center justify-between gap-3 rounded-2xl bg-white/[0.035] px-3 py-2 sm:px-4">
                    <div className="flex min-w-0 self-stretch flex-col py-1">
                      <p className="text-[15px] font-medium text-white">Similarity threshold</p>
                      <p className="my-auto max-w-[17rem] text-xs leading-4 text-white/52">Higher values keep results more closely related to your topic.</p>
                    </div>
                    <SimilarityDial
                      value={draft.minRelevanceThreshold}
                      onChange={(value) => setDraft((current) => ({
                        ...current,
                        minRelevanceThreshold: value,
                      }))}
                    />
                  </div>

                  <SettingsRow title="Creative Commons only" description="Only consider sources marked with a Creative Commons license.">
                    <SettingsSwitch
                      label="Creative Commons only"
                      checked={draft.creativeCommonsOnly}
                      onChange={(checked) => setDraft((current) => ({ ...current, creativeCommonsOnly: checked }))}
                    />
                  </SettingsRow>

                  <SettingsRow title="Source video length" description="Prefer a source length while keeping individual reels concise.">
                    <CustomSelect
                      label="Source video length"
                      value={draft.preferredVideoDuration}
                      options={DURATION_OPTIONS}
                      onChange={(value) => setDraft((current) => ({
                        ...current,
                        preferredVideoDuration: value,
                      }))}
                      className="shrink-0"
                      buttonClassName="h-9 min-w-[132px] rounded-full bg-white/[0.09] px-4 text-sm text-white/88 hover:bg-white/[0.07]"
                    />
                  </SettingsRow>
                </div>

                {availabilityState.message ? (
                  <div
                    aria-live="polite"
                    className={`mt-3 rounded-2xl px-3 py-2.5 text-sm leading-5 ${
                      availabilityState.status === "ok"
                        ? "bg-emerald-400/[0.09] text-emerald-100"
                        : availabilityState.status === "partial"
                          ? "bg-amber-300/[0.09] text-amber-100"
                          : "bg-white/[0.04] text-white/56"
                    }`}
                  >
                    <p>{availabilityState.message}</p>
                    {availabilityState.limitingFactors.length > 0 ? (
                      <p className="mt-1 text-xs opacity-75">Main limits: {availabilityState.limitingFactors.join(", ")}.</p>
                    ) : null}
                  </div>
                ) : null}
              </section>
            ) : null}

            {activeSection === "playback" ? (
              <section aria-labelledby="settings-playback-title">
                <SectionHeading id="settings-playback-title" title="Playback" detail="Set the default behavior for reels when a feed opens." />
                <div className="space-y-1.5">
                  <SettingsRow title="Start muted" description="Open each reel without audio until you turn sound on.">
                    <SettingsSwitch
                      label="Start reels muted"
                      checked={draft.startMuted}
                      onChange={(checked) => setDraft((current) => ({ ...current, startMuted: checked }))}
                    />
                  </SettingsRow>
                  <SettingsRow title="Autoplay next reel" description="Move to the next reel automatically when playback finishes.">
                    <SettingsSwitch
                      label="Autoplay next reel"
                      checked={draft.autoplayNextReel}
                      onChange={(checked) => setDraft((current) => ({ ...current, autoplayNextReel: checked }))}
                    />
                  </SettingsRow>
                </div>
              </section>
            ) : null}

            {activeSection === "plan" ? (
              <section aria-labelledby="settings-plan-title">
                <SectionHeading id="settings-plan-title" title="Plan & Usage" detail="Review today’s quota or manage your ReelAI subscription." />
                {!resolvedAccount ? (
                  <div className="rounded-2xl bg-white/[0.045] p-4">
                    <p className="text-lg font-medium text-white">Sign in to view your plan</p>
                    <p className="mt-2 text-sm leading-6 text-white/52">Your plan, daily usage, and Stripe subscription are tied to your ReelAI account.</p>
                    <div className="mt-5 flex flex-wrap gap-2">
                      <button
                        type="button"
                        onClick={() => onOpenAuth?.("login")}
                        className="rounded-xl bg-white px-4 py-2.5 text-sm font-semibold text-black transition hover:bg-white/88 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white"
                      >
                        Log in
                      </button>
                      <button
                        type="button"
                        onClick={() => onOpenAuth?.("register")}
                        className="rounded-xl bg-white/[0.08] px-4 py-2.5 text-sm font-medium text-white transition-colors hover:bg-white/[0.07] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white"
                      >
                        Sign up
                      </button>
                    </div>
                  </div>
                ) : !resolvedAccount.isVerified ? (
                  <div className="rounded-2xl bg-amber-300/[0.08] p-4 text-amber-100">
                    <p className="font-medium">Verify your email to use paid plans.</p>
                    <button type="button" onClick={() => selectSection("account")} className="mt-3 rounded-xl bg-white px-4 py-2 text-sm font-semibold text-black">
                      Open Account
                    </button>
                  </div>
                ) : billingStatus ? (
                  <div>
                  <div data-current-plan-card className="rounded-2xl bg-white/[0.045] p-4">
                    <div className="flex items-start justify-between gap-4">
                      <div>
                        <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-white/45">Current plan</p>
                        <p className="mt-2 text-2xl font-semibold tracking-[-0.035em] text-white">{formatPlanName(billingStatus.plan)}</p>
                      </div>
                      <span className="rounded-full bg-white/[0.09] px-3 py-1.5 text-xs font-medium text-white/80">
                        {billingStatus.remaining_searches} left today
                      </span>
                    </div>
                    <div className="mt-6 h-1.5 overflow-hidden rounded-full bg-white/[0.09]" aria-hidden="true">
                      <div className="h-full rounded-full bg-white" style={{ width: `${percentUsed}%` }} />
                    </div>
                    <p className="mt-2 text-xs text-white/50">
                      {billingStatus.used_searches} of {billingStatus.daily_limit} searches used · resets {formatResetTime(billingStatus.reset_at)}
                    </p>
                    {subscription?.cancel_at_period_end ? (
                      <p className="mt-3 rounded-xl bg-amber-300/[0.08] px-3 py-2.5 text-xs leading-5 text-amber-100">
                        Your paid access continues until {subscription.current_period_end
                          ? new Date(subscription.current_period_end).toLocaleDateString()
                          : "the end of the billing period"}.
                      </p>
                    ) : null}
                    {billingError ? (
                      <div className="mt-3 flex flex-col gap-3 rounded-xl bg-rose-300/[0.08] px-3 py-3 text-sm text-rose-100 sm:flex-row sm:items-center sm:justify-between" role="alert">
                        <p className="leading-5">{billingError}</p>
                        {onBillingRefresh ? (
                          <button
                            type="button"
                            onClick={() => void onBillingRefresh()}
                            disabled={billingLoading}
                            className="shrink-0 rounded-lg bg-white/[0.09] px-3 py-2 text-xs font-medium text-white disabled:opacity-45"
                          >
                            {billingLoading ? "Loading…" : "Try again"}
                          </button>
                        ) : null}
                      </div>
                    ) : null}
                  </div>
                  <div className="mt-5">
                    <BillingActions
                      status={billingStatus}
                      plans={billingPlans}
                      onError={setBillingActionError}
                      demoMode={demoMode}
                      presentation="pricing"
                    />
                  </div>
                  </div>
                ) : (
                  <div className="rounded-2xl bg-white/[0.045] p-4" aria-live="polite">
                    <p className="text-lg font-medium text-white">
                      {billingLoading ? "Loading your plan…" : "Plan details unavailable"}
                    </p>
                    <p className="mt-2 text-sm text-white/52">{billingError || "Try loading your plan again."}</p>
                    {onBillingRefresh ? (
                      <button
                        type="button"
                        onClick={() => void onBillingRefresh()}
                        disabled={billingLoading}
                        className="mt-4 rounded-xl bg-white/[0.09] px-4 py-2.5 text-sm font-medium text-white disabled:opacity-45"
                      >
                        {billingLoading ? "Loading…" : "Try again"}
                      </button>
                    ) : null}
                  </div>
                )}
                {billingActionError ? <p className="mt-3 text-sm text-[#ffb4b4]" role="alert">{billingActionError}</p> : null}
              </section>
            ) : null}

            {activeSection === "data" ? (
              <section aria-labelledby="settings-data-title">
                <SectionHeading id="settings-data-title" title="Data Controls" detail="Remove locally stored activity without changing your account or subscription." />
                <div className="space-y-1.5">
                  <SettingsRow title="Clear search data" description="Remove search history and saved search sessions from this account.">
                    <button
                      type="button"
                      onClick={() => {
                        onClearSearchData();
                        showNotice("Search data cleared");
                      }}
                      className="h-9 min-w-[96px] shrink-0 rounded-xl bg-white/[0.09] px-4 text-sm font-medium text-white transition-colors hover:bg-white/[0.07] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white"
                    >
                      Clear
                    </button>
                  </SettingsRow>
                  <SettingsRow title="Clear set cache" description="Remove cached community sets and unfinished local drafts on this device.">
                    <button
                      type="button"
                      onClick={clearSetCache}
                      className="h-9 min-w-[96px] shrink-0 rounded-xl bg-white/[0.09] px-4 text-sm font-medium text-white transition-colors hover:bg-white/[0.07] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white"
                    >
                      Clear
                    </button>
                  </SettingsRow>
                  <SettingsRow title="Reset defaults" description="Restore Search and Playback preferences. Save to apply the reset.">
                    <button
                      type="button"
                      onClick={resetDefaults}
                      className="h-9 min-w-[96px] shrink-0 rounded-xl bg-white/[0.09] px-4 text-sm font-medium text-white transition-colors hover:bg-white/[0.07] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white"
                    >
                      Reset
                    </button>
                  </SettingsRow>
                </div>
              </section>
            ) : null}

            {activeSection === "account" ? (
              <section
                aria-labelledby="settings-account-title"
                className={`transition-opacity duration-[420ms] ease-out motion-reduce:transition-none ${
                  accountContentVisible ? "opacity-100" : "opacity-0"
                }`}
                data-account-view={accountView}
              >
                {resolvedAccount && accountView === "password" ? (
                  <>
                    <SectionHeading
                      id="settings-account-title"
                      title="Update password"
                      detail="Confirm your current password, then choose a new one."
                    />
                    <form
                      className="w-full"
                      onSubmit={(event) => {
                        event.preventDefault();
                        void handlePasswordChange();
                      }}
                    >
                      <div className="grid gap-3">
                        <PasswordField
                          label="Current password"
                          value={currentPassword}
                          onChange={setCurrentPassword}
                          autoComplete="current-password"
                        />
                        <PasswordField
                          label="New password"
                          value={newPassword}
                          onChange={setNewPassword}
                          autoComplete="new-password"
                        />
                        <PasswordField
                          label="Confirm new password"
                          value={confirmNewPassword}
                          onChange={setConfirmNewPassword}
                          autoComplete="new-password"
                        />
                      </div>
                      <div className="mt-5 flex w-full flex-wrap justify-end gap-2">
                        <button
                          type="button"
                          onClick={() => {
                            setAccountError(null);
                            setAccountNotice(null);
                            switchAccountView("overview");
                          }}
                          disabled={Boolean(accountBusy)}
                          className="rounded-xl bg-white/[0.08] px-4 py-2.5 text-sm font-medium text-white disabled:opacity-45"
                        >
                          Cancel
                        </button>
                        <button
                          type="submit"
                          disabled={Boolean(accountBusy)}
                          className="rounded-xl bg-white px-4 py-2.5 text-sm font-semibold text-black disabled:opacity-45"
                        >
                          {accountBusy === "password" ? "Updating…" : "Update password"}
                        </button>
                      </div>
                    </form>
                  </>
                ) : (
                  <>
                    <SectionHeading id="settings-account-title" title="Account" detail="Manage your identity, security, and sign-in state." />
                    {!resolvedAccount ? (
                      <div className="rounded-2xl bg-white/[0.045] p-4">
                        <p className="text-lg font-medium text-white">You’re not signed in</p>
                        <p className="mt-2 text-sm leading-6 text-white/52">Sign in to sync history, settings, and sets across devices.</p>
                        <div className="mt-5 flex flex-wrap gap-2">
                          <button type="button" onClick={() => onOpenAuth?.("login")} className="rounded-xl bg-white px-4 py-2.5 text-sm font-semibold text-black">Log in</button>
                          <button type="button" onClick={() => onOpenAuth?.("register")} className="rounded-xl bg-white/[0.08] px-4 py-2.5 text-sm font-medium text-white">Sign up</button>
                        </div>
                      </div>
                    ) : (
                      <div className="space-y-3">
                        <div className="rounded-2xl bg-white/[0.045] p-4">
                          <div className="flex items-start gap-4">
                            <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-emerald-500 text-sm font-semibold text-white">
                              {resolvedAccount.username.trim().charAt(0).toUpperCase() || "R"}
                            </div>
                            <div className="min-w-0 flex-1">
                              <p className="truncate text-lg font-medium text-white">@{resolvedAccount.username}</p>
                              <p className="mt-1 truncate text-sm text-white/48">{resolvedAccount.email || "No email address"}</p>
                            </div>
                            <span className={`shrink-0 rounded-full px-2.5 py-1 text-[11px] font-medium ${
                              resolvedAccount.isVerified ? "bg-emerald-400/[0.12] text-emerald-100" : "bg-amber-300/[0.1] text-amber-100"
                            }`}>
                              {resolvedAccount.isVerified ? "Verified" : "Unverified"}
                            </span>
                          </div>
                          {!resolvedAccount.isVerified ? (
                            <div className="mt-5 rounded-2xl bg-white/[0.04] p-4">
                              <p className="text-sm text-white/68">Verify your email to unlock paid plans and keep your account secure.</p>
                              <div className="mt-3 flex flex-col gap-2 sm:flex-row">
                                <input
                                  value={verificationCode}
                                  onChange={(event) => setVerificationCode(event.target.value.replace(/\s/g, ""))}
                                  inputMode="numeric"
                                  autoComplete="one-time-code"
                                  placeholder="Verification code"
                                  className="h-11 min-w-0 flex-1 rounded-xl bg-white/[0.08] px-3 text-sm text-white placeholder:text-white/32 outline-none"
                                />
                                <button type="button" onClick={() => void handleVerify()} disabled={Boolean(accountBusy)} className="h-11 rounded-xl bg-white px-4 text-sm font-semibold text-black disabled:opacity-45">
                                  {accountBusy === "verify" ? "Verifying…" : "Verify"}
                                </button>
                                <button type="button" onClick={() => void handleSendVerification()} disabled={Boolean(accountBusy)} className="h-11 rounded-xl bg-white/[0.09] px-4 text-sm font-medium text-white disabled:opacity-45">
                                  {accountBusy === "send-verification" ? "Sending…" : "Send code"}
                                </button>
                              </div>
                              {verificationCodeDebug ? <p className="mt-2 text-xs text-amber-100/75">Local debug code: {verificationCodeDebug}</p> : null}
                            </div>
                          ) : null}
                        </div>

                        <div className="rounded-2xl bg-white/[0.035] p-4">
                          <h3 className="text-base font-medium text-white">Password</h3>
                          <p className="mt-1 text-xs leading-5 text-white/48">Change the password you use to sign in to ReelAI.</p>
                          <button
                            type="button"
                            onClick={() => switchAccountView("password")}
                            className="mt-3 rounded-xl bg-white px-4 py-2.5 text-sm font-semibold text-black"
                          >
                            Update password
                          </button>
                        </div>

                        <div className="rounded-2xl bg-white/[0.035] p-4">
                          <h3 className="text-base font-medium text-white">Session</h3>
                          <p className="mt-1 text-xs leading-5 text-white/48">Sign out of ReelAI on this device.</p>
                          <button type="button" onClick={() => void handleSignOut()} disabled={Boolean(accountBusy)} className="mt-3 rounded-xl bg-white/[0.09] px-4 py-2.5 text-sm font-medium text-white disabled:opacity-45">
                            {accountBusy === "signout" ? "Signing out…" : "Sign out"}
                          </button>
                        </div>

                        <div className="rounded-2xl bg-rose-400/[0.06] p-4">
                          <h3 className="text-base font-medium text-white">Delete account</h3>
                          <p className="mt-1 text-xs leading-5 text-white/48">Permanently delete this account, its sessions, history, settings, and sets.</p>
                          {!deleteExpanded ? (
                            <button type="button" onClick={() => setDeleteExpanded(true)} className="mt-3 rounded-xl bg-rose-400/[0.12] px-4 py-2.5 text-sm font-medium text-rose-100">
                              Delete account…
                            </button>
                          ) : (
                            <div className="mt-4 flex flex-col gap-2 sm:flex-row">
                              <PasswordField
                                value={deletePassword}
                                onChange={setDeletePassword}
                                autoComplete="current-password"
                                placeholder="Current password"
                                ariaLabel="Current password to confirm account deletion"
                                className="min-w-0 flex-1"
                              />
                              <button type="button" onClick={() => void handleDeleteAccount()} disabled={Boolean(accountBusy)} className="h-11 rounded-xl bg-rose-300 px-4 text-sm font-semibold text-rose-950 disabled:opacity-45">
                                {accountBusy === "delete" ? "Deleting…" : "Permanently delete"}
                              </button>
                              <button type="button" onClick={() => { setDeleteExpanded(false); setDeletePassword(""); }} className="h-11 rounded-xl bg-white/[0.08] px-4 text-sm font-medium text-white">
                                Cancel
                              </button>
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                  </>
                )}
                {accountError ? <p className="mt-3 text-sm text-[#ffb4b4]" role="alert">{accountError}</p> : null}
                {accountNotice ? <p className="mt-3 text-sm text-emerald-200" aria-live="polite">{accountNotice}</p> : null}
              </section>
            ) : null}

          </div>
        </div>

        <div
          className="pointer-events-none absolute inset-x-0 bottom-[max(0.75rem,env(safe-area-inset-bottom))] z-30 px-3 sm:px-5 md:bottom-6 md:px-6"
          data-settings-save-dock="true"
        >
          <div className="mx-auto flex w-full max-w-[720px] items-center justify-between gap-4">
              <p className="min-w-0 truncate text-xs text-white/48" aria-live="polite">{notice || (hasUnsavedChanges ? "Unsaved changes" : "")}</p>
              <button
                type="button"
                onClick={savePreferences}
                disabled={!settingsHydrated || !hasUnsavedChanges}
                className="pointer-events-auto min-w-[94px] rounded-xl bg-white px-5 py-2.5 text-sm font-semibold text-black transition hover:bg-white/88 disabled:cursor-not-allowed disabled:bg-white/[0.09] disabled:text-white/32"
              >
                Save
              </button>
          </div>
        </div>
      </div>
    </div>
  );
});
