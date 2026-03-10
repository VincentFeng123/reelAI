"use client";

import { forwardRef, useCallback, useEffect, useImperativeHandle, useMemo, useRef, useState } from "react";

import {
  type PreferredVideoDuration,
  type StudyReelsSettings,
  type VideoPoolMode,
  DEFAULT_STUDY_REELS_SETTINGS,
  MAX_RELEVANCE,
  MIN_RELEVANCE,
  TARGET_CLIP_DURATION_MAX,
  TARGET_CLIP_DURATION_MIN_GAP,
  TARGET_CLIP_DURATION_MIN,
  readStudyReelsSettings,
  saveStudyReelsSettings,
} from "@/lib/settings";

type SettingsPanelProps = {
  onClearSearchData: () => void;
  onUnsavedChangesChange?: (hasUnsavedChanges: boolean) => void;
  onAvailabilityModalClose?: (source: "close-button" | "backdrop") => void;
};

type SavedPreferences = StudyReelsSettings;
export type SettingsPanelHandle = {
  savePreferences: () => void;
  discardUnsavedChanges: () => void;
  hasUnsavedChanges: () => boolean;
};

const COMMUNITY_SETS_STORAGE_KEY = "studyreels-community-sets";
const RELEVANCE_STEP = 0.02;
const LOW_RELEVANCE_WARNING_THRESHOLD = 0.12;
const CLIP_DURATION_STEP = 5;
const CLIP_DURATION_MIN_GAP = TARGET_CLIP_DURATION_MIN_GAP;
const CLIP_DURATION_SPAN = TARGET_CLIP_DURATION_MAX - TARGET_CLIP_DURATION_MIN;
const RELEVANCE_DIAL_START_DEG = -135;
const RELEVANCE_DIAL_END_DEG = 135;
const RELEVANCE_DIAL_SPAN_DEG = RELEVANCE_DIAL_END_DEG - RELEVANCE_DIAL_START_DEG;
const RELEVANCE_DIAL_SIZE_PX = 176;
const RELEVANCE_DIAL_CENTER_PX = RELEVANCE_DIAL_SIZE_PX / 2;
const RELEVANCE_DIAL_RADIUS_PX = 54;
const RELEVANCE_DIAL_CIRCUMFERENCE = 2 * Math.PI * RELEVANCE_DIAL_RADIUS_PX;
const RELEVANCE_DIAL_ARC_LENGTH = RELEVANCE_DIAL_CIRCUMFERENCE * (RELEVANCE_DIAL_SPAN_DEG / 360);
const CLIP_DOT_LABEL_OFFSET_PX = 18;
const CLIP_DIAL_ENDPOINT_LABEL_RADIUS_PX = RELEVANCE_DIAL_RADIUS_PX + 20;
const CLIP_TRACK_ENDPOINT_LABEL_RADIUS_PX = RELEVANCE_DIAL_RADIUS_PX + 26;
const CLIP_ENDPOINT_LABEL_VERTICAL_PULL = 0.94;
const CLIP_ENDPOINT_LABEL_X_OFFSET_PX = -18;
const CLIP_ENDPOINT_LABEL_MIN_DISTANCE_PX = 34;
const CLIP_ENDPOINT_LABEL_MAX_PUSH_PX = 24;
const RELEVANCE_ENDPOINT_LABEL_X_OFFSET_PX = -20;

const VIDEO_POOL_OPTIONS: Array<{ value: VideoPoolMode; label: string }> = [
  { value: "short-first", label: "Short" },
  { value: "balanced", label: "Balanced" },
  { value: "long-form", label: "Long" },
];

const DURATION_OPTIONS: Array<{ value: PreferredVideoDuration; label: string }> = [
  { value: "any", label: "Any" },
  { value: "short", label: "Short" },
  { value: "medium", label: "Medium" },
  { value: "long", label: "Long" },
];

const durationSummaryLabel: Record<PreferredVideoDuration, string> = {
  any: "Any",
  short: "Short",
  medium: "Medium",
  long: "Long",
};

const poolSummaryLabel: Record<VideoPoolMode, string> = {
  "short-first": "Short",
  balanced: "Balanced",
  "long-form": "Long",
};

type AvailabilityState = {
  status: "idle" | "checking" | "ok" | "partial" | "blocked" | "none" | "error";
  message: string;
  limitingFactors: string[];
};

function SettingsInfoTooltip({ text }: { text: string }) {
  return (
    <span className="group relative inline-flex shrink-0">
      <span
        tabIndex={0}
        aria-label={text}
        className="inline-flex h-4 w-4 items-center justify-center rounded-full text-white/55 transition-colors hover:text-white focus-visible:text-white focus-visible:outline-none"
      >
        <i className="fa-solid fa-circle-info text-[11px]" aria-hidden="true" />
      </span>
      <span
        role="tooltip"
        className="pointer-events-none absolute left-1/2 top-full z-30 mt-1.5 w-max max-w-[220px] -translate-x-1/2 rounded-lg bg-black/90 px-3 py-2 text-[10px] font-medium leading-tight text-white/88 opacity-0 shadow-[0_10px_28px_rgba(0,0,0,0.45)] backdrop-blur-sm transition-opacity duration-150 group-hover:opacity-100 group-focus-within:opacity-100"
      >
        {text}
      </span>
    </span>
  );
}

function pushPointAwayFromPoint(
  baseX: number,
  baseY: number,
  otherX: number,
  otherY: number,
  minDistance: number,
  maxPush: number,
): { x: number; y: number } {
  const dx = baseX - otherX;
  const dy = baseY - otherY;
  const distance = Math.hypot(dx, dy);
  if (!Number.isFinite(distance) || distance === 0 || distance >= minDistance) {
    return { x: baseX, y: baseY };
  }
  const shortfall = minDistance - distance;
  const push = Math.min(maxPush, shortfall);
  return {
    x: baseX + (dx / distance) * push,
    y: baseY + (dy / distance) * push,
  };
}

function buildHeuristicAvailabilityState(settings: StudyReelsSettings): AvailabilityState {
  const relevanceRatio = Math.max(0, Math.min(1, settings.minRelevanceThreshold / MAX_RELEVANCE));
  const clipWidth = settings.targetClipDurationMaxSec - settings.targetClipDurationMinSec;
  const midpoint = (settings.targetClipDurationMinSec + settings.targetClipDurationMaxSec) / 2;
  let score = 78;

  score -= relevanceRatio * 34;
  if (settings.videoPoolMode === "short-first") {
    score += 8;
  } else if (settings.videoPoolMode === "balanced") {
    score += 3;
  } else {
    score -= 8;
  }

  if (settings.preferredVideoDuration === "any") {
    score += 9;
  } else if (settings.preferredVideoDuration === "medium") {
    score += 1;
  } else if (settings.preferredVideoDuration === "short") {
    score -= 3;
  } else {
    score -= 11;
  }

  if (clipWidth < 10) {
    score -= 20;
  } else if (clipWidth < 20) {
    score -= 12;
  } else if (clipWidth < 35) {
    score -= 6;
  } else {
    score += 4;
  }

  if (midpoint < 25) {
    score -= 7;
  } else if (midpoint > 115) {
    score -= 8;
  } else {
    score += 2;
  }

  if (settings.generationMode === "fast") {
    score -= 5;
  } else {
    score += 2;
  }

  const ratePct = Math.max(5, Math.min(96, Math.round(score)));
  const limitingFactors: string[] = [];
  if (settings.minRelevanceThreshold >= 0.45) {
    limitingFactors.push("high similarity threshold");
  }
  if (clipWidth < 20) {
    limitingFactors.push("narrow clip range");
  }
  if (settings.preferredVideoDuration === "long") {
    limitingFactors.push("long source preference");
  }
  if (settings.videoPoolMode === "long-form") {
    limitingFactors.push("long pool mode");
  }
  if (settings.generationMode === "fast") {
    limitingFactors.push("fast generation mode");
  }

  if (ratePct >= 78) {
    return {
      status: "ok",
      message: `Heuristic success estimate: ${ratePct}%.`,
      limitingFactors,
    };
  }
  if (ratePct >= 52) {
    return {
      status: "partial",
      message: `Heuristic success estimate: ${ratePct}%.`,
      limitingFactors,
    };
  }
  return {
    status: "blocked",
    message: `Heuristic success estimate: ${ratePct}%.`,
    limitingFactors,
  };
}

export const SettingsPanel = forwardRef<SettingsPanelHandle, SettingsPanelProps>(function SettingsPanel(
  { onClearSearchData, onUnsavedChangesChange, onAvailabilityModalClose }: SettingsPanelProps,
  ref,
) {
  const [minRelevanceThreshold, setMinRelevanceThreshold] = useState(DEFAULT_STUDY_REELS_SETTINGS.minRelevanceThreshold);
  const [startMuted, setStartMuted] = useState(DEFAULT_STUDY_REELS_SETTINGS.startMuted);
  const [videoPoolMode, setVideoPoolMode] = useState<VideoPoolMode>(DEFAULT_STUDY_REELS_SETTINGS.videoPoolMode);
  const [preferredVideoDuration, setPreferredVideoDuration] = useState<PreferredVideoDuration>(
    DEFAULT_STUDY_REELS_SETTINGS.preferredVideoDuration,
  );
  const [targetClipDurationMinSec, setTargetClipDurationMinSec] = useState(
    DEFAULT_STUDY_REELS_SETTINGS.targetClipDurationMinSec,
  );
  const [targetClipDurationMaxSec, setTargetClipDurationMaxSec] = useState(
    DEFAULT_STUDY_REELS_SETTINGS.targetClipDurationMaxSec,
  );
  const [savedPreferences, setSavedPreferences] = useState<SavedPreferences | null>(null);
  const [settingsHydrated, setSettingsHydrated] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [isAvailabilityModalMounted, setIsAvailabilityModalMounted] = useState(false);
  const [isAvailabilityModalVisible, setIsAvailabilityModalVisible] = useState(false);
  const [showLowRelevanceWarning, setShowLowRelevanceWarning] = useState(false);
  const [availabilityState, setAvailabilityState] = useState<AvailabilityState>({
    status: "idle",
    message: "Save settings to estimate success rate from configuration heuristics.",
    limitingFactors: [],
  });
  const availabilityRequestRef = useRef(0);
  const availabilityModalCloseTimerRef = useRef<number | null>(null);
  const availabilityModalOpenRafRef = useRef<number | null>(null);
  const availabilityHeuristicTimerRef = useRef<number | null>(null);
  const relevanceDialRef = useRef<HTMLDivElement | null>(null);
  const clipDialRef = useRef<HTMLDivElement | null>(null);
  const [isRelevanceDialDragging, setIsRelevanceDialDragging] = useState(false);
  const [clipDialDragHandle, setClipDialDragHandle] = useState<"min" | "max" | null>(null);

  const targetClipDurationSec = useMemo(
    () => Math.round((targetClipDurationMinSec + targetClipDurationMaxSec) / 2),
    [targetClipDurationMinSec, targetClipDurationMaxSec],
  );
  const relevanceDialRatio = useMemo(() => {
    if (MAX_RELEVANCE <= MIN_RELEVANCE) {
      return 0;
    }
    return (minRelevanceThreshold - MIN_RELEVANCE) / (MAX_RELEVANCE - MIN_RELEVANCE);
  }, [minRelevanceThreshold]);
  const relevanceDialAngleDeg = useMemo(
    () => RELEVANCE_DIAL_START_DEG + relevanceDialRatio * RELEVANCE_DIAL_SPAN_DEG,
    [relevanceDialRatio],
  );
  const relevanceDialAngleRad = useMemo(() => (relevanceDialAngleDeg * Math.PI) / 180, [relevanceDialAngleDeg]);
  const relevanceDialKnobX = useMemo(() => Math.cos(relevanceDialAngleRad) * RELEVANCE_DIAL_RADIUS_PX, [relevanceDialAngleRad]);
  const relevanceDialKnobY = useMemo(() => Math.sin(relevanceDialAngleRad) * RELEVANCE_DIAL_RADIUS_PX, [relevanceDialAngleRad]);
  const relevanceDialProgressLength = useMemo(
    () => Math.max(0, Math.min(RELEVANCE_DIAL_ARC_LENGTH, RELEVANCE_DIAL_ARC_LENGTH * relevanceDialRatio)),
    [relevanceDialRatio],
  );
  const relevanceBandLabel = useMemo(() => {
    if (minRelevanceThreshold >= 0.45) {
      return "Strict";
    }
    if (minRelevanceThreshold <= 0.18) {
      return "Loose";
    }
    return "Balanced";
  }, [minRelevanceThreshold]);
  const clampRelevanceThreshold = useCallback((value: number): number => {
    const snapped = Math.round(value / RELEVANCE_STEP) * RELEVANCE_STEP;
    return Number(Math.max(MIN_RELEVANCE, Math.min(MAX_RELEVANCE, snapped)).toFixed(2));
  }, []);
  const setRelevanceFromDialAngle = useCallback(
    (angleDeg: number) => {
      const boundedAngle = Math.max(RELEVANCE_DIAL_START_DEG, Math.min(RELEVANCE_DIAL_END_DEG, angleDeg));
      const ratio = (boundedAngle - RELEVANCE_DIAL_START_DEG) / RELEVANCE_DIAL_SPAN_DEG;
      const nextValue = MIN_RELEVANCE + ratio * (MAX_RELEVANCE - MIN_RELEVANCE);
      setMinRelevanceThreshold(clampRelevanceThreshold(nextValue));
    },
    [clampRelevanceThreshold],
  );
  const setRelevanceFromDialPoint = useCallback(
    (clientX: number, clientY: number) => {
      const element = relevanceDialRef.current;
      if (!element) {
        return;
      }
      const rect = element.getBoundingClientRect();
      const centerX = rect.left + rect.width / 2;
      const centerY = rect.top + rect.height / 2;
      const dx = clientX - centerX;
      const dy = clientY - centerY;
      const rawAngle = (Math.atan2(dy, dx) * 180) / Math.PI;
      if (rawAngle > RELEVANCE_DIAL_END_DEG) {
        setRelevanceFromDialAngle(RELEVANCE_DIAL_END_DEG);
        return;
      }
      if (rawAngle < RELEVANCE_DIAL_START_DEG) {
        setRelevanceFromDialAngle(RELEVANCE_DIAL_START_DEG);
        return;
      }
      setRelevanceFromDialAngle(rawAngle);
    },
    [setRelevanceFromDialAngle],
  );
  const adjustRelevanceDial = useCallback(
    (delta: number) => {
      setMinRelevanceThreshold((previous) => clampRelevanceThreshold(previous + delta));
    },
    [clampRelevanceThreshold],
  );
  const clipMinRatio = useMemo(() => {
    if (CLIP_DURATION_SPAN <= 0) {
      return 0;
    }
    return (targetClipDurationMinSec - TARGET_CLIP_DURATION_MIN) / CLIP_DURATION_SPAN;
  }, [targetClipDurationMinSec]);
  const clipMaxRatio = useMemo(() => {
    if (CLIP_DURATION_SPAN <= 0) {
      return 1;
    }
    return (targetClipDurationMaxSec - TARGET_CLIP_DURATION_MIN) / CLIP_DURATION_SPAN;
  }, [targetClipDurationMaxSec]);
  const clipMinAngleDeg = useMemo(
    () => RELEVANCE_DIAL_START_DEG + clipMinRatio * RELEVANCE_DIAL_SPAN_DEG,
    [clipMinRatio],
  );
  const clipMaxAngleDeg = useMemo(
    () => RELEVANCE_DIAL_START_DEG + clipMaxRatio * RELEVANCE_DIAL_SPAN_DEG,
    [clipMaxRatio],
  );
  const clipMinAngleRad = useMemo(() => (clipMinAngleDeg * Math.PI) / 180, [clipMinAngleDeg]);
  const clipMaxAngleRad = useMemo(() => (clipMaxAngleDeg * Math.PI) / 180, [clipMaxAngleDeg]);
  const clipMinKnobX = useMemo(() => Math.cos(clipMinAngleRad) * RELEVANCE_DIAL_RADIUS_PX, [clipMinAngleRad]);
  const clipMinKnobY = useMemo(() => Math.sin(clipMinAngleRad) * RELEVANCE_DIAL_RADIUS_PX, [clipMinAngleRad]);
  const clipMaxKnobX = useMemo(() => Math.cos(clipMaxAngleRad) * RELEVANCE_DIAL_RADIUS_PX, [clipMaxAngleRad]);
  const clipMaxKnobY = useMemo(() => Math.sin(clipMaxAngleRad) * RELEVANCE_DIAL_RADIUS_PX, [clipMaxAngleRad]);
  const clipMinUnitX = useMemo(() => clipMinKnobX / RELEVANCE_DIAL_RADIUS_PX, [clipMinKnobX]);
  const clipMinUnitY = useMemo(() => clipMinKnobY / RELEVANCE_DIAL_RADIUS_PX, [clipMinKnobY]);
  const clipMaxUnitX = useMemo(() => clipMaxKnobX / RELEVANCE_DIAL_RADIUS_PX, [clipMaxKnobX]);
  const clipMaxUnitY = useMemo(() => clipMaxKnobY / RELEVANCE_DIAL_RADIUS_PX, [clipMaxKnobY]);
  const clipMinInsideLabelStyle = useMemo(
    () => ({
      transform: `translate(calc(-50% + ${clipMinKnobX - clipMinUnitX * CLIP_DOT_LABEL_OFFSET_PX}px), calc(-50% + ${clipMinKnobY - clipMinUnitY * CLIP_DOT_LABEL_OFFSET_PX}px))`,
    }),
    [clipMinKnobX, clipMinKnobY, clipMinUnitX, clipMinUnitY],
  );
  const clipMinOutsideLabelStyle = useMemo(
    () => ({
      transform: `translate(calc(-50% + ${clipMinKnobX + clipMinUnitX * CLIP_DOT_LABEL_OFFSET_PX}px), calc(-50% + ${clipMinKnobY + clipMinUnitY * CLIP_DOT_LABEL_OFFSET_PX}px))`,
    }),
    [clipMinKnobX, clipMinKnobY, clipMinUnitX, clipMinUnitY],
  );
  const clipMaxInsideLabelStyle = useMemo(
    () => ({
      transform: `translate(calc(-50% + ${clipMaxKnobX - clipMaxUnitX * CLIP_DOT_LABEL_OFFSET_PX}px), calc(-50% + ${clipMaxKnobY - clipMaxUnitY * CLIP_DOT_LABEL_OFFSET_PX}px))`,
    }),
    [clipMaxKnobX, clipMaxKnobY, clipMaxUnitX, clipMaxUnitY],
  );
  const clipMaxOutsideLabelStyle = useMemo(
    () => ({
      transform: `translate(calc(-50% + ${clipMaxKnobX + clipMaxUnitX * CLIP_DOT_LABEL_OFFSET_PX}px), calc(-50% + ${clipMaxKnobY + clipMaxUnitY * CLIP_DOT_LABEL_OFFSET_PX}px))`,
    }),
    [clipMaxKnobX, clipMaxKnobY, clipMaxUnitX, clipMaxUnitY],
  );
  const clipMinOutsideLabelPointX = useMemo(() => clipMinKnobX + clipMinUnitX * CLIP_DOT_LABEL_OFFSET_PX, [clipMinKnobX, clipMinUnitX]);
  const clipMinOutsideLabelPointY = useMemo(() => clipMinKnobY + clipMinUnitY * CLIP_DOT_LABEL_OFFSET_PX, [clipMinKnobY, clipMinUnitY]);
  const clipMaxOutsideLabelPointX = useMemo(() => clipMaxKnobX + clipMaxUnitX * CLIP_DOT_LABEL_OFFSET_PX, [clipMaxKnobX, clipMaxUnitX]);
  const clipMaxOutsideLabelPointY = useMemo(() => clipMaxKnobY + clipMaxUnitY * CLIP_DOT_LABEL_OFFSET_PX, [clipMaxKnobY, clipMaxUnitY]);
  const clipDialStartAngleRad = useMemo(() => (RELEVANCE_DIAL_START_DEG * Math.PI) / 180, []);
  const clipDialEndAngleRad = useMemo(() => (RELEVANCE_DIAL_END_DEG * Math.PI) / 180, []);
  const clipDialStartBaseX = useMemo(
    () => Math.cos(clipDialStartAngleRad) * CLIP_TRACK_ENDPOINT_LABEL_RADIUS_PX + CLIP_ENDPOINT_LABEL_X_OFFSET_PX,
    [clipDialStartAngleRad],
  );
  const clipDialStartBaseY = useMemo(
    () => Math.sin(clipDialStartAngleRad) * CLIP_TRACK_ENDPOINT_LABEL_RADIUS_PX * CLIP_ENDPOINT_LABEL_VERTICAL_PULL,
    [clipDialStartAngleRad],
  );
  const clipDialEndBaseX = useMemo(
    () => Math.cos(clipDialEndAngleRad) * CLIP_TRACK_ENDPOINT_LABEL_RADIUS_PX + CLIP_ENDPOINT_LABEL_X_OFFSET_PX,
    [clipDialEndAngleRad],
  );
  const clipDialEndBaseY = useMemo(
    () => Math.sin(clipDialEndAngleRad) * CLIP_TRACK_ENDPOINT_LABEL_RADIUS_PX * CLIP_ENDPOINT_LABEL_VERTICAL_PULL,
    [clipDialEndAngleRad],
  );
  const clipDialStartLabelPoint = useMemo(
    () =>
      pushPointAwayFromPoint(
        clipDialStartBaseX,
        clipDialStartBaseY,
        clipMinOutsideLabelPointX,
        clipMinOutsideLabelPointY,
        CLIP_ENDPOINT_LABEL_MIN_DISTANCE_PX,
        CLIP_ENDPOINT_LABEL_MAX_PUSH_PX,
      ),
    [clipDialStartBaseX, clipDialStartBaseY, clipMinOutsideLabelPointX, clipMinOutsideLabelPointY],
  );
  const clipDialEndLabelPoint = useMemo(
    () =>
      pushPointAwayFromPoint(
        clipDialEndBaseX,
        clipDialEndBaseY,
        clipMaxOutsideLabelPointX,
        clipMaxOutsideLabelPointY,
        CLIP_ENDPOINT_LABEL_MIN_DISTANCE_PX,
        CLIP_ENDPOINT_LABEL_MAX_PUSH_PX,
      ),
    [clipDialEndBaseX, clipDialEndBaseY, clipMaxOutsideLabelPointX, clipMaxOutsideLabelPointY],
  );
  const clipDialStartLabelStyle = useMemo(
    () => ({
      transform: `translate(calc(-50% + ${clipDialStartLabelPoint.x}px), calc(-50% + ${clipDialStartLabelPoint.y}px))`,
    }),
    [clipDialStartLabelPoint.x, clipDialStartLabelPoint.y],
  );
  const clipDialEndLabelStyle = useMemo(
    () => ({
      transform: `translate(calc(-50% + ${clipDialEndLabelPoint.x}px), calc(-50% + ${clipDialEndLabelPoint.y}px))`,
    }),
    [clipDialEndLabelPoint.x, clipDialEndLabelPoint.y],
  );
  const relevanceLooseLabelStyle = useMemo(
    () => ({
      transform: `translate(calc(-50% + ${Math.cos(clipDialStartAngleRad) * CLIP_DIAL_ENDPOINT_LABEL_RADIUS_PX + RELEVANCE_ENDPOINT_LABEL_X_OFFSET_PX}px), calc(-50% + ${Math.sin(clipDialStartAngleRad) * CLIP_DIAL_ENDPOINT_LABEL_RADIUS_PX}px))`,
    }),
    [clipDialStartAngleRad],
  );
  const relevanceStrictLabelStyle = useMemo(
    () => ({
      transform: `translate(calc(-50% + ${Math.cos(clipDialEndAngleRad) * CLIP_DIAL_ENDPOINT_LABEL_RADIUS_PX + RELEVANCE_ENDPOINT_LABEL_X_OFFSET_PX}px), calc(-50% + ${Math.sin(clipDialEndAngleRad) * CLIP_DIAL_ENDPOINT_LABEL_RADIUS_PX}px))`,
    }),
    [clipDialEndAngleRad],
  );
  const clipDialSegmentLength = useMemo(
    () => Math.max(0, Math.min(RELEVANCE_DIAL_ARC_LENGTH, RELEVANCE_DIAL_ARC_LENGTH * (clipMaxRatio - clipMinRatio))),
    [clipMaxRatio, clipMinRatio],
  );
  const clipDialSegmentOffset = useMemo(() => -RELEVANCE_DIAL_ARC_LENGTH * clipMinRatio, [clipMinRatio]);
  const clipBandLabel = useMemo(() => {
    const clipWidth = targetClipDurationMaxSec - targetClipDurationMinSec;
    if (targetClipDurationMinSec >= 70) {
      return "Long";
    }
    if (clipWidth <= 18) {
      return "Tight";
    }
    if (targetClipDurationMaxSec <= 45) {
      return "Short";
    }
    return "Balanced";
  }, [targetClipDurationMaxSec, targetClipDurationMinSec]);
  const generationModeForChecks = savedPreferences?.generationMode ?? DEFAULT_STUDY_REELS_SETTINGS.generationMode;
  const defaultInputModeForSave = savedPreferences?.defaultInputMode ?? DEFAULT_STUDY_REELS_SETTINGS.defaultInputMode;

  const normalizeClipRange = useCallback((rawMin: number, rawMax: number): { min: number; max: number } => {
    const safeMinBound = TARGET_CLIP_DURATION_MIN;
    const safeMaxBound = TARGET_CLIP_DURATION_MAX;
    const gap = CLIP_DURATION_MIN_GAP;
    let min = Math.round(rawMin);
    let max = Math.round(rawMax);
    if (!Number.isFinite(min)) {
      min = safeMinBound;
    }
    if (!Number.isFinite(max)) {
      max = safeMaxBound;
    }
    min = Math.max(safeMinBound, Math.min(safeMaxBound, min));
    max = Math.max(safeMinBound, Math.min(safeMaxBound, max));
    if (max - min < gap) {
      if (rawMin <= rawMax) {
        max = Math.min(safeMaxBound, min + gap);
        min = Math.max(safeMinBound, max - gap);
      } else {
        min = Math.max(safeMinBound, max - gap);
        max = Math.min(safeMaxBound, min + gap);
      }
    }
    return { min, max };
  }, []);

  const updateClipRange = useCallback(
    (nextMin: number, nextMax: number) => {
      const normalized = normalizeClipRange(nextMin, nextMax);
      setTargetClipDurationMinSec(normalized.min);
      setTargetClipDurationMaxSec(normalized.max);
    },
    [normalizeClipRange],
  );
  const clampClipDurationValue = useCallback((value: number): number => {
    const snapped = Math.round(value / CLIP_DURATION_STEP) * CLIP_DURATION_STEP;
    return Math.max(TARGET_CLIP_DURATION_MIN, Math.min(TARGET_CLIP_DURATION_MAX, snapped));
  }, []);
  const updateClipRangeByHandle = useCallback(
    (handle: "min" | "max", nextValue: number) => {
      const safeValue = clampClipDurationValue(nextValue);
      const currentMin = clampClipDurationValue(targetClipDurationMinSec);
      const currentMax = clampClipDurationValue(targetClipDurationMaxSec);

      if (handle === "min") {
        let nextMin = safeValue;
        let nextMax = currentMax;
        const maxAllowed = nextMax - CLIP_DURATION_MIN_GAP;

        if (nextMin > maxAllowed) {
          const shift = nextMin - maxAllowed;
          nextMax = clampClipDurationValue(nextMax + shift);
          nextMin = Math.min(nextMin, nextMax - CLIP_DURATION_MIN_GAP);
        }

        nextMin = clampClipDurationValue(
          Math.max(TARGET_CLIP_DURATION_MIN, Math.min(nextMin, nextMax - CLIP_DURATION_MIN_GAP)),
        );
        setTargetClipDurationMinSec(nextMin);
        setTargetClipDurationMaxSec(nextMax);
        return;
      }

      let nextMin = currentMin;
      let nextMax = safeValue;
      const minAllowed = nextMin + CLIP_DURATION_MIN_GAP;

      if (nextMax < minAllowed) {
        const shift = minAllowed - nextMax;
        nextMin = clampClipDurationValue(nextMin - shift);
        nextMax = Math.max(nextMax, nextMin + CLIP_DURATION_MIN_GAP);
      }

      nextMax = clampClipDurationValue(
        Math.min(TARGET_CLIP_DURATION_MAX, Math.max(nextMax, nextMin + CLIP_DURATION_MIN_GAP)),
      );
      setTargetClipDurationMinSec(nextMin);
      setTargetClipDurationMaxSec(nextMax);
    },
    [clampClipDurationValue, targetClipDurationMaxSec, targetClipDurationMinSec],
  );
  const clipDialAngleFromPoint = useCallback((clientX: number, clientY: number): number | null => {
    const element = clipDialRef.current;
    if (!element) {
      return null;
    }
    const rect = element.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;
    const dx = clientX - centerX;
    const dy = clientY - centerY;
    return (Math.atan2(dy, dx) * 180) / Math.PI;
  }, []);
  const setClipHandleFromDialAngle = useCallback(
    (handle: "min" | "max", angleDeg: number) => {
      const boundedAngle = Math.max(RELEVANCE_DIAL_START_DEG, Math.min(RELEVANCE_DIAL_END_DEG, angleDeg));
      const ratio = (boundedAngle - RELEVANCE_DIAL_START_DEG) / RELEVANCE_DIAL_SPAN_DEG;
      const nextValue = TARGET_CLIP_DURATION_MIN + ratio * CLIP_DURATION_SPAN;
      updateClipRangeByHandle(handle, nextValue);
    },
    [updateClipRangeByHandle],
  );
  const setClipHandleFromDialPoint = useCallback(
    (handle: "min" | "max", clientX: number, clientY: number) => {
      const angle = clipDialAngleFromPoint(clientX, clientY);
      if (angle == null) {
        return;
      }
      setClipHandleFromDialAngle(handle, angle);
    },
    [clipDialAngleFromPoint, setClipHandleFromDialAngle],
  );
  const pickNearestClipHandle = useCallback(
    (clientX: number, clientY: number): "min" | "max" => {
      const element = clipDialRef.current;
      if (!element) {
        return "min";
      }
      const rect = element.getBoundingClientRect();
      const centerX = rect.left + rect.width / 2;
      const centerY = rect.top + rect.height / 2;
      const minX = centerX + clipMinKnobX;
      const minY = centerY + clipMinKnobY;
      const maxX = centerX + clipMaxKnobX;
      const maxY = centerY + clipMaxKnobY;
      const distMin = (clientX - minX) ** 2 + (clientY - minY) ** 2;
      const distMax = (clientX - maxX) ** 2 + (clientY - maxY) ** 2;
      return distMin <= distMax ? "min" : "max";
    },
    [clipMaxKnobX, clipMaxKnobY, clipMinKnobX, clipMinKnobY],
  );
  const adjustClipMinDial = useCallback(
    (delta: number) => {
      updateClipRangeByHandle("min", targetClipDurationMinSec + delta);
    },
    [targetClipDurationMinSec, updateClipRangeByHandle],
  );
  const adjustClipMaxDial = useCallback(
    (delta: number) => {
      updateClipRangeByHandle("max", targetClipDurationMaxSec + delta);
    },
    [targetClipDurationMaxSec, updateClipRangeByHandle],
  );

  useEffect(() => {
    const saved = readStudyReelsSettings();
    setMinRelevanceThreshold(saved.minRelevanceThreshold);
    setStartMuted(saved.startMuted);
    setVideoPoolMode(saved.videoPoolMode);
    setPreferredVideoDuration(saved.preferredVideoDuration);
    setTargetClipDurationMinSec(saved.targetClipDurationMinSec);
    setTargetClipDurationMaxSec(saved.targetClipDurationMaxSec);
    setSavedPreferences(saved);
    setSettingsHydrated(true);
  }, []);

  const runAvailabilityCheck = useCallback(async (settings: StudyReelsSettings) => {
    if (!settingsHydrated) {
      return;
    }
    const requestId = availabilityRequestRef.current + 1;
    availabilityRequestRef.current = requestId;
    setIsAvailabilityModalMounted(true);
    if (typeof window === "undefined") {
      setIsAvailabilityModalVisible(true);
    } else {
      if (availabilityModalCloseTimerRef.current !== null) {
        window.clearTimeout(availabilityModalCloseTimerRef.current);
        availabilityModalCloseTimerRef.current = null;
      }
      if (availabilityModalOpenRafRef.current !== null) {
        window.cancelAnimationFrame(availabilityModalOpenRafRef.current);
      }
      availabilityModalOpenRafRef.current = window.requestAnimationFrame(() => {
        setIsAvailabilityModalVisible(true);
      });
    }
    setAvailabilityState({
      status: "checking",
      message: "Estimating success rate from configuration heuristics...",
      limitingFactors: [],
    });
    setShowLowRelevanceWarning(settings.minRelevanceThreshold <= LOW_RELEVANCE_WARNING_THRESHOLD);

    if (typeof window === "undefined") {
      if (availabilityRequestRef.current !== requestId) {
        return;
      }
      setAvailabilityState(buildHeuristicAvailabilityState(settings));
      return;
    }

    if (availabilityHeuristicTimerRef.current !== null) {
      window.clearTimeout(availabilityHeuristicTimerRef.current);
    }
    availabilityHeuristicTimerRef.current = window.setTimeout(() => {
      if (availabilityRequestRef.current !== requestId) {
        return;
      }
      setAvailabilityState(buildHeuristicAvailabilityState(settings));
      availabilityHeuristicTimerRef.current = null;
    }, 260);
  }, [settingsHydrated]);

  const closeAvailabilityModal = useCallback((source: "close-button" | "backdrop") => {
    setIsAvailabilityModalVisible(false);
    if (typeof window === "undefined") {
      setIsAvailabilityModalMounted(false);
      onAvailabilityModalClose?.(source);
      return;
    }
    if (availabilityModalCloseTimerRef.current !== null) {
      window.clearTimeout(availabilityModalCloseTimerRef.current);
    }
    availabilityModalCloseTimerRef.current = window.setTimeout(() => {
      setIsAvailabilityModalMounted(false);
      availabilityModalCloseTimerRef.current = null;
    }, 220);
    onAvailabilityModalClose?.(source);
  }, [onAvailabilityModalClose]);

  useEffect(() => {
    return () => {
      if (typeof window === "undefined") {
        return;
      }
      if (availabilityModalCloseTimerRef.current !== null) {
        window.clearTimeout(availabilityModalCloseTimerRef.current);
      }
      if (availabilityModalOpenRafRef.current !== null) {
        window.cancelAnimationFrame(availabilityModalOpenRafRef.current);
      }
      if (availabilityHeuristicTimerRef.current !== null) {
        window.clearTimeout(availabilityHeuristicTimerRef.current);
      }
    };
  }, []);

  const showNotice = useCallback((message: string) => {
    setNotice(message);
    if (typeof window === "undefined") {
      return;
    }
    window.setTimeout(() => {
      setNotice((current) => (current === message ? null : current));
    }, 2200);
  }, []);

  const resetPreferences = () => {
    setMinRelevanceThreshold(DEFAULT_STUDY_REELS_SETTINGS.minRelevanceThreshold);
    setStartMuted(DEFAULT_STUDY_REELS_SETTINGS.startMuted);
    setVideoPoolMode(DEFAULT_STUDY_REELS_SETTINGS.videoPoolMode);
    setPreferredVideoDuration(DEFAULT_STUDY_REELS_SETTINGS.preferredVideoDuration);
    setTargetClipDurationMinSec(DEFAULT_STUDY_REELS_SETTINGS.targetClipDurationMinSec);
    setTargetClipDurationMaxSec(DEFAULT_STUDY_REELS_SETTINGS.targetClipDurationMaxSec);
    showNotice("Defaults loaded. Save to apply.");
  };

  const clearCommunitySetsCache = () => {
    if (typeof window !== "undefined") {
      window.localStorage.removeItem(COMMUNITY_SETS_STORAGE_KEY);
    }
    showNotice("Saved community set cache cleared.");
  };

  const settingsSummary = useMemo(() => {
    return `Match ${minRelevanceThreshold.toFixed(2)}+ · ${poolSummaryLabel[videoPoolMode]} feed · ${durationSummaryLabel[preferredVideoDuration]} source videos · ${targetClipDurationMinSec}-${targetClipDurationMaxSec}s clips · ${startMuted ? "Muted" : "Sound on"}`;
  }, [minRelevanceThreshold, preferredVideoDuration, startMuted, targetClipDurationMaxSec, targetClipDurationMinSec, videoPoolMode]);

  const hasUnsavedChanges = useMemo(() => {
    if (!savedPreferences) {
      return false;
    }
    const currentMinRelevance = Number(minRelevanceThreshold.toFixed(2));
    return (
      savedPreferences.minRelevanceThreshold !== currentMinRelevance
      || savedPreferences.startMuted !== startMuted
      || savedPreferences.videoPoolMode !== videoPoolMode
      || savedPreferences.preferredVideoDuration !== preferredVideoDuration
      || savedPreferences.targetClipDurationMinSec !== targetClipDurationMinSec
      || savedPreferences.targetClipDurationMaxSec !== targetClipDurationMaxSec
    );
  }, [
    minRelevanceThreshold,
    preferredVideoDuration,
    savedPreferences,
    startMuted,
    targetClipDurationMaxSec,
    targetClipDurationMinSec,
    videoPoolMode,
  ]);

  const savePreferences = useCallback(() => {
    if (!settingsHydrated) {
      return;
    }
    const saved = saveStudyReelsSettings({
      generationMode: generationModeForChecks,
      defaultInputMode: defaultInputModeForSave,
      minRelevanceThreshold,
      startMuted,
      videoPoolMode,
      preferredVideoDuration,
      targetClipDurationSec,
      targetClipDurationMinSec,
      targetClipDurationMaxSec,
    });
    setSavedPreferences(saved);
    showNotice("Settings saved and applied.");
    void runAvailabilityCheck(saved);
  }, [
    defaultInputModeForSave,
    generationModeForChecks,
    minRelevanceThreshold,
    preferredVideoDuration,
    runAvailabilityCheck,
    settingsHydrated,
    showNotice,
    startMuted,
    targetClipDurationMaxSec,
    targetClipDurationMinSec,
    targetClipDurationSec,
    videoPoolMode,
  ]);
  const discardUnsavedChanges = useCallback(() => {
    const saved = savedPreferences ?? readStudyReelsSettings();
    setMinRelevanceThreshold(saved.minRelevanceThreshold);
    setStartMuted(saved.startMuted);
    setVideoPoolMode(saved.videoPoolMode);
    setPreferredVideoDuration(saved.preferredVideoDuration);
    setTargetClipDurationMinSec(saved.targetClipDurationMinSec);
    setTargetClipDurationMaxSec(saved.targetClipDurationMaxSec);
    setSavedPreferences(saved);
  }, [savedPreferences]);

  useImperativeHandle(
    ref,
    () => ({
      savePreferences,
      discardUnsavedChanges,
      hasUnsavedChanges: () => settingsHydrated && hasUnsavedChanges,
    }),
    [discardUnsavedChanges, hasUnsavedChanges, savePreferences, settingsHydrated],
  );

  useEffect(() => {
    if (!onUnsavedChangesChange) {
      return;
    }
    onUnsavedChangesChange(settingsHydrated && hasUnsavedChanges);
  }, [hasUnsavedChanges, onUnsavedChangesChange, settingsHydrated]);

  return (
    <div className="flex h-full min-h-0 w-full justify-center overflow-y-auto px-6 pt-20 pb-6 md:px-10 md:pt-8 md:pb-10 lg:px-10">
      <div className="w-full max-w-[980px]">
        <header className="mb-6 md:mb-8">
          <h1 className="text-3xl font-semibold tracking-tight text-white md:text-4xl">Settings</h1>
          <p className="mt-2 text-sm text-white/70">Configure your default reel generation behavior and manage saved app data.</p>
        </header>

        <div className="rounded-3xl bg-white/[0.07] p-4 backdrop-blur-[4px] md:p-6">
          <p className="text-[11px] font-semibold uppercase tracking-[0.11em] text-white/62">Generation Defaults</p>

          <div className="mt-4 grid gap-4 md:grid-cols-2">
            <div className="rounded-2xl bg-white/[0.06] p-3.5 backdrop-blur-[4px] md:p-4">
              <div>
                <div className="flex items-center justify-between gap-3">
                  <div className="flex items-center gap-1.5">
                    <p className="text-sm font-semibold text-white/95">Similarity threshold</p>
                    <SettingsInfoTooltip text="Higher values keep results tightly related and filter out unrelated content." />
                  </div>
                  <span className="rounded-md bg-black/35 px-2 py-0.5 text-xs font-semibold text-white/90">
                    {minRelevanceThreshold.toFixed(2)}+
                  </span>
                </div>
                <div className="mt-3 flex justify-center">
                  <div className="relative h-44 w-44">
                    <svg
                      aria-hidden="true"
                      className="absolute inset-0 h-full w-full"
                      viewBox={`0 0 ${RELEVANCE_DIAL_SIZE_PX} ${RELEVANCE_DIAL_SIZE_PX}`}
                    >
                      <circle
                        cx={RELEVANCE_DIAL_CENTER_PX}
                        cy={RELEVANCE_DIAL_CENTER_PX}
                        r={RELEVANCE_DIAL_RADIUS_PX}
                        fill="none"
                        stroke="rgba(255,255,255,0.24)"
                        strokeWidth="9"
                        strokeLinecap="round"
                        strokeDasharray={`${RELEVANCE_DIAL_ARC_LENGTH} ${RELEVANCE_DIAL_CIRCUMFERENCE}`}
                        transform={`rotate(${225} ${RELEVANCE_DIAL_CENTER_PX} ${RELEVANCE_DIAL_CENTER_PX})`}
                      />
                      <circle
                        cx={RELEVANCE_DIAL_CENTER_PX}
                        cy={RELEVANCE_DIAL_CENTER_PX}
                        r={RELEVANCE_DIAL_RADIUS_PX}
                        fill="none"
                        stroke="rgba(255,255,255,0.95)"
                        strokeWidth="9"
                        strokeLinecap="round"
                        strokeDasharray={`${relevanceDialProgressLength} ${RELEVANCE_DIAL_CIRCUMFERENCE}`}
                        transform={`rotate(${225} ${RELEVANCE_DIAL_CENTER_PX} ${RELEVANCE_DIAL_CENTER_PX})`}
                      />
                    </svg>
                    <div
                      ref={relevanceDialRef}
                      role="slider"
                      tabIndex={0}
                      aria-label="Similarity threshold"
                      aria-valuemin={MIN_RELEVANCE}
                      aria-valuemax={MAX_RELEVANCE}
                      aria-valuenow={Number(minRelevanceThreshold.toFixed(2))}
                      aria-valuetext={`${minRelevanceThreshold.toFixed(2)} threshold`}
                      onPointerDown={(event) => {
                        event.preventDefault();
                        setIsRelevanceDialDragging(true);
                        event.currentTarget.setPointerCapture(event.pointerId);
                        setRelevanceFromDialPoint(event.clientX, event.clientY);
                      }}
                      onPointerMove={(event) => {
                        if (!isRelevanceDialDragging) {
                          return;
                        }
                        setRelevanceFromDialPoint(event.clientX, event.clientY);
                      }}
                      onPointerUp={(event) => {
                        setIsRelevanceDialDragging(false);
                        if (event.currentTarget.hasPointerCapture(event.pointerId)) {
                          event.currentTarget.releasePointerCapture(event.pointerId);
                        }
                      }}
                      onPointerCancel={(event) => {
                        setIsRelevanceDialDragging(false);
                        if (event.currentTarget.hasPointerCapture(event.pointerId)) {
                          event.currentTarget.releasePointerCapture(event.pointerId);
                        }
                      }}
                      onKeyDown={(event) => {
                        if (event.key === "ArrowLeft" || event.key === "ArrowDown") {
                          event.preventDefault();
                          adjustRelevanceDial(-RELEVANCE_STEP);
                          return;
                        }
                        if (event.key === "ArrowRight" || event.key === "ArrowUp") {
                          event.preventDefault();
                          adjustRelevanceDial(RELEVANCE_STEP);
                          return;
                        }
                        if (event.key === "PageDown") {
                          event.preventDefault();
                          adjustRelevanceDial(-RELEVANCE_STEP * 3);
                          return;
                        }
                        if (event.key === "PageUp") {
                          event.preventDefault();
                          adjustRelevanceDial(RELEVANCE_STEP * 3);
                          return;
                        }
                        if (event.key === "Home") {
                          event.preventDefault();
                          setMinRelevanceThreshold(MIN_RELEVANCE);
                          return;
                        }
                        if (event.key === "End") {
                          event.preventDefault();
                          setMinRelevanceThreshold(MAX_RELEVANCE);
                        }
                      }}
                      className={`absolute inset-0 rounded-full outline-none ${
                        isRelevanceDialDragging
                          ? "cursor-grabbing focus-visible:ring-0 focus-visible:ring-offset-0"
                          : "cursor-grab focus-visible:ring-2 focus-visible:ring-white/70 focus-visible:ring-offset-2 focus-visible:ring-offset-black/70"
                      }`}
                      style={{ touchAction: "none" }}
                    >
                      <span
                        aria-hidden="true"
                        className="pointer-events-none absolute left-1/2 top-1/2 h-4 w-4 rounded-full border border-black/35 bg-white shadow-[0_0_0_3px_rgba(255,255,255,0.2)]"
                        style={{
                          transform: `translate(calc(-50% + ${relevanceDialKnobX}px), calc(-50% + ${relevanceDialKnobY}px))`,
                        }}
                      />
                      <span
                        aria-hidden="true"
                        className="pointer-events-none absolute left-1/2 top-1/2 text-[11px] font-semibold text-white/65"
                        style={relevanceLooseLabelStyle}
                      >
                        Loose
                      </span>
                      <span
                        aria-hidden="true"
                        className="pointer-events-none absolute left-1/2 top-1/2 text-[11px] font-semibold text-white/65"
                        style={relevanceStrictLabelStyle}
                      >
                        Strict
                      </span>
                      <span aria-hidden="true" className="pointer-events-none absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 text-[11px] font-semibold uppercase tracking-[0.08em] text-white/82">
                        {relevanceBandLabel}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="rounded-2xl bg-white/[0.06] p-3.5 backdrop-blur-[4px] md:p-4">
              <div>
                <div className="flex items-center justify-between gap-3">
                  <div className="flex items-center gap-1.5">
                    <p className="text-sm font-semibold text-white/95">Clip length range</p>
                    <SettingsInfoTooltip text="Set hard minimum and maximum clip lengths generated for reels." />
                  </div>
                  <span className="rounded-md bg-black/35 px-2 py-0.5 text-xs font-semibold text-white/90">
                    {targetClipDurationMinSec}-{targetClipDurationMaxSec}s
                  </span>
                </div>
                <div className="mt-3 flex justify-center">
                  <div className="relative h-44 w-44">
                    <svg
                      aria-hidden="true"
                      className="absolute inset-0 h-full w-full"
                      viewBox={`0 0 ${RELEVANCE_DIAL_SIZE_PX} ${RELEVANCE_DIAL_SIZE_PX}`}
                    >
                      <circle
                        cx={RELEVANCE_DIAL_CENTER_PX}
                        cy={RELEVANCE_DIAL_CENTER_PX}
                        r={RELEVANCE_DIAL_RADIUS_PX}
                        fill="none"
                        stroke="rgba(255,255,255,0.24)"
                        strokeWidth="9"
                        strokeLinecap="round"
                        strokeDasharray={`${RELEVANCE_DIAL_ARC_LENGTH} ${RELEVANCE_DIAL_CIRCUMFERENCE}`}
                        transform={`rotate(${225} ${RELEVANCE_DIAL_CENTER_PX} ${RELEVANCE_DIAL_CENTER_PX})`}
                      />
                      <circle
                        cx={RELEVANCE_DIAL_CENTER_PX}
                        cy={RELEVANCE_DIAL_CENTER_PX}
                        r={RELEVANCE_DIAL_RADIUS_PX}
                        fill="none"
                        stroke="rgba(255,255,255,0.95)"
                        strokeWidth="9"
                        strokeLinecap="round"
                        strokeDasharray={`${clipDialSegmentLength} ${RELEVANCE_DIAL_CIRCUMFERENCE}`}
                        strokeDashoffset={clipDialSegmentOffset}
                        transform={`rotate(${225} ${RELEVANCE_DIAL_CENTER_PX} ${RELEVANCE_DIAL_CENTER_PX})`}
                      />
                    </svg>
                    <div
                      ref={clipDialRef}
                      role="group"
                      tabIndex={0}
                      aria-label="Clip length range dial"
                      onPointerDown={(event) => {
                        event.preventDefault();
                        const handle = pickNearestClipHandle(event.clientX, event.clientY);
                        setClipDialDragHandle(handle);
                        event.currentTarget.setPointerCapture(event.pointerId);
                        setClipHandleFromDialPoint(handle, event.clientX, event.clientY);
                      }}
                      onPointerMove={(event) => {
                        if (!clipDialDragHandle) {
                          return;
                        }
                        setClipHandleFromDialPoint(clipDialDragHandle, event.clientX, event.clientY);
                      }}
                      onPointerUp={(event) => {
                        setClipDialDragHandle(null);
                        if (event.currentTarget.hasPointerCapture(event.pointerId)) {
                          event.currentTarget.releasePointerCapture(event.pointerId);
                        }
                      }}
                      onPointerCancel={(event) => {
                        setClipDialDragHandle(null);
                        if (event.currentTarget.hasPointerCapture(event.pointerId)) {
                          event.currentTarget.releasePointerCapture(event.pointerId);
                        }
                      }}
                      onKeyDown={(event) => {
                        if (event.key === "ArrowLeft" || event.key === "ArrowDown") {
                          event.preventDefault();
                          adjustClipMinDial(-CLIP_DURATION_STEP);
                          return;
                        }
                        if (event.key === "ArrowRight" || event.key === "ArrowUp") {
                          event.preventDefault();
                          adjustClipMaxDial(CLIP_DURATION_STEP);
                          return;
                        }
                        if (event.key === "PageDown") {
                          event.preventDefault();
                          adjustClipMinDial(-CLIP_DURATION_STEP * 2);
                          adjustClipMaxDial(-CLIP_DURATION_STEP * 2);
                          return;
                        }
                        if (event.key === "PageUp") {
                          event.preventDefault();
                          adjustClipMinDial(CLIP_DURATION_STEP * 2);
                          adjustClipMaxDial(CLIP_DURATION_STEP * 2);
                          return;
                        }
                        if (event.key === "Home") {
                          event.preventDefault();
                          updateClipRange(TARGET_CLIP_DURATION_MIN, targetClipDurationMaxSec);
                          return;
                        }
                        if (event.key === "End") {
                          event.preventDefault();
                          updateClipRange(targetClipDurationMinSec, TARGET_CLIP_DURATION_MAX);
                        }
                      }}
                      className={`absolute inset-0 rounded-full outline-none focus:outline-none ${
                        clipDialDragHandle ? "cursor-grabbing" : "cursor-grab"
                      }`}
                      style={{ touchAction: "none" }}
                    >
                      <span
                        aria-hidden="true"
                        className="pointer-events-none absolute left-1/2 top-1/2 h-4 w-4 rounded-full border border-black/35 bg-white shadow-[0_0_0_3px_rgba(255,255,255,0.2)]"
                        style={{
                          transform: `translate(calc(-50% + ${clipMinKnobX}px), calc(-50% + ${clipMinKnobY}px))`,
                        }}
                      />
                      <span
                        aria-hidden="true"
                        className="pointer-events-none absolute left-1/2 top-1/2 h-4 w-4 rounded-full border border-black/35 bg-white shadow-[0_0_0_3px_rgba(255,255,255,0.2)]"
                        style={{
                          transform: `translate(calc(-50% + ${clipMaxKnobX}px), calc(-50% + ${clipMaxKnobY}px))`,
                        }}
                      />
                      <span
                        aria-hidden="true"
                        className="pointer-events-none absolute left-1/2 top-1/2 text-[12px] font-semibold uppercase tracking-[0.08em] text-white/70"
                        style={clipMinInsideLabelStyle}
                      >
                        -
                      </span>
                      <span
                        aria-hidden="true"
                        className="pointer-events-none absolute left-1/2 top-1/2 text-[10px] font-semibold text-white/88"
                        style={clipMinOutsideLabelStyle}
                      >
                        {targetClipDurationMinSec}s
                      </span>
                      <span
                        aria-hidden="true"
                        className="pointer-events-none absolute left-1/2 top-1/2 text-[12px] font-semibold uppercase tracking-[0.08em] text-white/70"
                        style={clipMaxInsideLabelStyle}
                      >
                        +
                      </span>
                      <span
                        aria-hidden="true"
                        className="pointer-events-none absolute left-1/2 top-1/2 text-[10px] font-semibold text-white/88"
                        style={clipMaxOutsideLabelStyle}
                      >
                        {targetClipDurationMaxSec}s
                      </span>
                      <span
                        aria-hidden="true"
                        className="pointer-events-none absolute left-1/2 top-1/2 text-[11px] font-semibold text-white/65"
                        style={clipDialStartLabelStyle}
                      >
                        {TARGET_CLIP_DURATION_MIN}s
                      </span>
                      <span
                        aria-hidden="true"
                        className="pointer-events-none absolute left-1/2 top-1/2 text-[11px] font-semibold text-white/65"
                        style={clipDialEndLabelStyle}
                      >
                        {TARGET_CLIP_DURATION_MAX}s
                      </span>
                      <span aria-hidden="true" className="pointer-events-none absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 text-[11px] font-semibold uppercase tracking-[0.08em] text-white/82">
                        {clipBandLabel}
                      </span>
                    </div>
                  </div>
                </div>
                <div className="mt-2 flex items-center justify-center">
                  <span className="text-[11px] font-semibold text-white/65">Target ~{targetClipDurationSec}s</span>
                </div>
              </div>
            </div>
          </div>

          <div className="mt-4 grid gap-4 md:grid-cols-2">
            <div className="rounded-2xl bg-white/[0.06] p-3.5 backdrop-blur-[4px] md:p-4">
              <div className="flex items-center gap-1.5">
                <p className="text-sm font-semibold text-white/95">Video pool mode</p>
                <SettingsInfoTooltip text="Control how aggressively long sources are included." />
              </div>
              <div className="relative mt-3 grid h-11 grid-cols-3 items-center rounded-2xl border border-white/20 bg-white/[0.08] p-1">
                <span
                  aria-hidden="true"
                  className="pointer-events-none absolute bottom-1 left-1 top-1 w-[calc((100%-8px)/3)] rounded-xl bg-white transition-transform duration-300 ease-out"
                  style={{
                    transform: `translateX(${VIDEO_POOL_OPTIONS.findIndex((option) => option.value === videoPoolMode) * 100}%)`,
                  }}
                />
                {VIDEO_POOL_OPTIONS.map((option) => (
                  <button
                    key={option.value}
                    type="button"
                    onClick={() => setVideoPoolMode(option.value)}
                    className={`relative z-10 rounded-xl px-1.5 py-2 text-[10px] font-semibold uppercase tracking-[0.04em] transition-colors duration-200 ${
                      videoPoolMode === option.value ? "text-black" : "text-white/75 hover:text-white"
                    }`}
                    aria-pressed={videoPoolMode === option.value}
                  >
                    {option.label}
                  </button>
                ))}
              </div>
            </div>

            <div className="rounded-2xl bg-white/[0.06] p-3.5 backdrop-blur-[4px] md:p-4">
              <div className="flex items-center gap-1.5">
                <p className="text-sm font-semibold text-white/95">Source video length</p>
                <SettingsInfoTooltip text="Prefer short clips, medium lessons, long lectures, or any." />
              </div>
              <div className="relative mt-3 grid h-11 grid-cols-4 items-center rounded-2xl border border-white/20 bg-white/[0.08] p-1">
                <span
                  aria-hidden="true"
                  className="pointer-events-none absolute bottom-1 left-1 top-1 w-[calc((100%-8px)/4)] rounded-xl bg-white transition-transform duration-300 ease-out"
                  style={{
                    transform: `translateX(${DURATION_OPTIONS.findIndex((option) => option.value === preferredVideoDuration) * 100}%)`,
                  }}
                />
                {DURATION_OPTIONS.map((option) => (
                  <button
                    key={option.value}
                    type="button"
                    onClick={() => setPreferredVideoDuration(option.value)}
                    className={`relative z-10 rounded-xl px-1.5 py-2 text-[10px] font-semibold uppercase tracking-[0.04em] transition-colors duration-200 ${
                      preferredVideoDuration === option.value ? "text-black" : "text-white/75 hover:text-white"
                    }`}
                    aria-pressed={preferredVideoDuration === option.value}
                  >
                    {option.label}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="mt-4 rounded-2xl bg-white/[0.06] p-3.5 backdrop-blur-[4px] md:p-4">
            <div className="flex items-center justify-between gap-4">
              <div className="flex items-center gap-1.5">
                <p className="text-sm font-semibold text-white/95">Start reels muted</p>
                <SettingsInfoTooltip text="Controls the default audio state when opening the feed." />
              </div>
              <button
                type="button"
                onClick={() => setStartMuted((prev) => !prev)}
                aria-pressed={startMuted}
                className={`relative inline-flex h-7 w-12 shrink-0 rounded-full border transition-colors duration-200 ${
                  startMuted ? "border-white bg-white" : "border-white/32 bg-black/50"
                }`}
              >
                <span
                  className={`absolute top-0.5 h-[22px] w-[22px] rounded-full transition-transform duration-200 ${
                    startMuted ? "translate-x-[24px] bg-black" : "translate-x-[2px] bg-white"
                  }`}
                />
              </button>
            </div>
          </div>

          <p className="mt-4 text-xs text-white/52">Current defaults: {settingsSummary}</p>
        </div>

        <div className="mt-4 rounded-3xl bg-white/[0.07] p-4 backdrop-blur-[4px] md:mt-5 md:p-6">
          <div className="flex items-center gap-1.5">
            <p className="text-[11px] font-semibold uppercase tracking-[0.11em] text-white/62">Utilities</p>
            <SettingsInfoTooltip text="Useful maintenance actions for search history and local cache." />
          </div>

          <div className="mt-4 grid gap-2 md:grid-cols-3">
            <button
              type="button"
              onClick={() => {
                onClearSearchData();
                showNotice("Search history and session data cleared.");
              }}
              className="rounded-xl border border-white/20 bg-black/36 px-3 py-2 text-xs font-semibold text-white/86 transition hover:bg-white/10"
            >
              Clear search data
            </button>
            <button
              type="button"
              onClick={clearCommunitySetsCache}
              className="rounded-xl border border-white/20 bg-black/36 px-3 py-2 text-xs font-semibold text-white/86 transition hover:bg-white/10"
            >
              Clear set cache
            </button>
            <button
              type="button"
              onClick={resetPreferences}
              className="rounded-xl border border-white/20 bg-black/36 px-3 py-2 text-xs font-semibold text-white/86 transition hover:bg-white/10"
            >
              Reset defaults
            </button>
          </div>
        </div>

        <div className="mt-2 flex items-end justify-between gap-3 pb-12">
          <p className="min-h-5 text-left text-xs text-white/72">{notice ?? ""}</p>
          <button
            type="button"
            onClick={savePreferences}
            disabled={!settingsHydrated || !hasUnsavedChanges}
            className="inline-flex min-w-[10rem] items-center justify-center whitespace-nowrap rounded-xl border border-white/24 bg-white px-7 py-3 text-sm font-semibold text-black transition-colors hover:bg-white/90 disabled:cursor-not-allowed disabled:opacity-50"
          >
            Save
          </button>
        </div>
      </div>
      {isAvailabilityModalMounted ? (
        <div
          className={`fixed inset-0 z-50 flex items-center justify-center bg-black/70 px-4 py-6 backdrop-blur-[2px] transition-opacity duration-200 ease-out ${
            isAvailabilityModalVisible ? "opacity-100" : "pointer-events-none opacity-0"
          }`}
          role="presentation"
          onClick={() => closeAvailabilityModal("backdrop")}
        >
          <div
            role="dialog"
            aria-modal="true"
            aria-label="Configuration success rate"
            className={`w-full max-w-xl rounded-3xl border border-white/25 bg-black p-5 text-white shadow-[0_18px_80px_rgba(0,0,0,0.5)] backdrop-blur-2xl transition-opacity duration-200 ease-out md:p-6 ${
              isAvailabilityModalVisible ? "opacity-100" : "opacity-0"
            }`}
            onClick={(event) => event.stopPropagation()}
          >
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="text-[11px] font-semibold uppercase tracking-[0.12em] text-white/65">Configuration check</p>
                <h3 className="mt-2 text-lg font-semibold text-white">
                  {availabilityState.status === "checking" ? "Checking success rate..." : "Success rate result"}
                </h3>
              </div>
              <button
                type="button"
                onClick={() => closeAvailabilityModal("close-button")}
                aria-label="Close"
                className="inline-flex h-8 w-8 items-center justify-center text-white/80 transition-colors hover:text-white focus-visible:outline-none"
              >
                <svg viewBox="0 0 20 20" aria-hidden="true" className="h-4 w-4 fill-none stroke-current stroke-2">
                  <path d="M5 5L15 15M15 5L5 15" strokeLinecap="round" />
                </svg>
              </button>
            </div>

            <div
              className={`mt-4 rounded-2xl border px-4 py-3 text-sm ${
                availabilityState.status === "ok"
                  ? "border-emerald-300/45 bg-emerald-500/14 text-emerald-100"
                  : availabilityState.status === "partial"
                  ? "border-sky-300/45 bg-sky-500/14 text-sky-100"
                  : availabilityState.status === "blocked"
                  ? "border-rose-300/45 bg-rose-500/16 text-rose-100"
                  : availabilityState.status === "none"
                  ? "border-amber-300/45 bg-amber-500/16 text-amber-100"
                  : availabilityState.status === "error"
                  ? "border-rose-300/45 bg-rose-500/16 text-rose-100"
                  : "border-white/24 bg-white/[0.06] text-white/88"
              }`}
            >
              <p>{availabilityState.message}</p>
              {availabilityState.limitingFactors.length > 0 ? (
                <div className="mt-2 border-t border-white/20 pt-2 text-xs">
                  <p className="font-semibold">
                  {availabilityState.limitingFactors.length > 1 ? "Main limits:" : "Main limit:"}
                  </p>
                  <ul className="mt-1.5 space-y-1">
                    {availabilityState.limitingFactors.map((factor) => (
                      <li key={factor} className="flex items-start gap-1.5">
                        <span aria-hidden="true" className="leading-[1.2] opacity-80">•</span>
                        <span className="leading-[1.2]">{factor}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              ) : null}
            </div>
            {showLowRelevanceWarning ? (
              <div className="mt-3 rounded-2xl border border-amber-300/45 bg-amber-500/16 px-4 py-3 text-sm text-amber-100">
                Warning: Videos unrelated to the topic may appear when similarity threshold is too low.
              </div>
            ) : null}
          </div>
        </div>
      ) : null}
    </div>
  );
});
