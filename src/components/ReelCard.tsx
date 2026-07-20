"use client";

import { type CSSProperties, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";

import type { Reel } from "@/lib/types";
import { loadYouTubeIframeApi } from "@/lib/youtubeIframeApi";

type Props = {
  reel: Reel;
  isActive: boolean;
  mutedPreference: boolean;
  onMutedPreferenceChange: (nextMuted: boolean) => void;
  autoplayEnabled: boolean;
  onAutoplayEnabledChange: (nextEnabled: boolean) => void;
  playbackRate: number;
  onPlaybackRateChange: (nextRate: number) => void;
  onRequestNextReel?: () => void;
  onPlaybackProgress?: (maxFraction: number, naturalEnd: boolean) => void;
  onOpenContent?: () => void;
};

type YouTubePlayer = {
  destroy: () => void;
  pauseVideo: () => void;
  playVideo: () => void;
  seekTo: (seconds: number, allowSeekAhead: boolean) => void;
  getCurrentTime: () => number;
  getAvailablePlaybackRates?: () => number[];
  getPlayerState?: () => number;
  mute: () => void;
  setPlaybackRate?: (rate: number) => void;
  unMute: () => void;
};

type VideoProvider = "youtube" | "external";

const PLAYER_REVEAL_DELAY_MS = 0;
const RESUME_MASK_MS = 480;
const AUTOPLAY_RETRY_DELAY_MS = 320;
const AUTOPLAY_MAX_RETRIES = 5;
const CLIP_END_POLL_INTERVAL_MS = 10;
const BOUNDARY_SEEK_PREROLL_SEC = 1;
const BOUNDARY_SEEK_CONFIRM_TOLERANCE_SEC = 0.25;
const BOUNDARY_SEEK_MAX_RETRIES = 2;
const BOUNDARY_SEEK_RETRY_GRACE_MS = 250;
const BOUNDARY_SEEK_ACCEPTABLE_OVERSHOOT_SEC = 1;
const BOUNDARY_ALIGNMENT_ERROR = "Could not align this clip to its exact start.";
const PLAYBACK_SPEED_OPTIONS = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2] as const;

function detectTouchLikeDevice(): boolean {
  if (typeof window === "undefined") {
    return false;
  }
  if (window.matchMedia?.("(pointer: coarse)").matches) {
    return true;
  }
  return navigator.maxTouchPoints > 0;
}

function detectMobilePhoneDevice(): boolean {
  if (typeof window === "undefined") {
    return false;
  }
  const ua = navigator.userAgent || "";
  const isIpadLike =
    /iPad/i.test(ua) ||
    (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1) ||
    /Macintosh/i.test(ua) && navigator.maxTouchPoints > 1;
  const isTabletUa = /Tablet|PlayBook|Silk/i.test(ua) || (/Android/i.test(ua) && !/Mobile/i.test(ua));
  const isPhoneUa = /iPhone|iPod|Android.+Mobile|Windows Phone|Mobile/i.test(ua);
  const narrowTouchViewport = (window.matchMedia?.("(max-width: 767px)").matches ?? false) && detectTouchLikeDevice();
  return !isIpadLike && !isTabletUa && (isPhoneUa || narrowTouchViewport);
}

function extractVideoId(urlValue: string): string | null {
  try {
    const url = new URL(urlValue);
    if (url.pathname.startsWith("/embed/")) {
      return url.pathname.split("/")[2] ?? null;
    }
    if (url.pathname.startsWith("/shorts/")) {
      return url.pathname.split("/")[2] ?? null;
    }
    const watch = url.searchParams.get("v");
    if (watch) {
      return watch;
    }
    if (url.hostname.includes("youtu.be")) {
      return url.pathname.replace("/", "") || null;
    }
  } catch {
    return null;
  }
  return null;
}

function detectVideoProvider(urlValue: string): VideoProvider {
  try {
    const url = new URL(urlValue);
    const host = url.hostname.toLowerCase();
    if (host.includes("youtube.com") || host.includes("youtu.be")) {
      return "youtube";
    }
  } catch {
    return "external";
  }
  return "external";
}

function normalizeExternalEmbedUrl(urlValue: string): string | null {
  try {
    const parsed = new URL(urlValue);
    if (parsed.protocol !== "https:" && parsed.protocol !== "http:") {
      return null;
    }
    if (parsed.hostname.toLowerCase().includes("clips.twitch.tv") && typeof window !== "undefined") {
      parsed.searchParams.set("parent", window.location.hostname);
    }
    return parsed.href;
  } catch {
    return null;
  }
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function hasReachedVerifiedClipEnd(playerTime: number, clipEnd: number): boolean {
  return Number.isFinite(playerTime) && playerTime + 0.01 >= clipEnd;
}

function hasReachedVerifiedClipStart(playerTime: number, clipStart: number): boolean {
  return Number.isFinite(playerTime) && playerTime >= clipStart;
}

function hasObservedBoundarySeek(playerTime: number, targetTime: number): boolean {
  const earliestExpectedTime = Math.max(
    0,
    targetTime - BOUNDARY_SEEK_PREROLL_SEC - BOUNDARY_SEEK_CONFIRM_TOLERANCE_SEC,
  );
  return Number.isFinite(playerTime)
    && playerTime >= earliestExpectedTime
    && playerTime <= targetTime + BOUNDARY_SEEK_CONFIRM_TOLERANCE_SEC;
}

function formatClock(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  const m = Math.floor(s / 60);
  const rem = s % 60;
  return `${m}:${String(rem).padStart(2, "0")}`;
}

function formatPlaybackRate(rate: number): string {
  return `${Number.isInteger(rate) ? rate.toFixed(0) : rate.toFixed(2).replace(/0$/, "")}x`;
}

export function ReelCard({
  reel,
  isActive,
  mutedPreference,
  onMutedPreferenceChange,
  autoplayEnabled,
  onAutoplayEnabledChange,
  playbackRate,
  onPlaybackRateChange,
  onRequestNextReel,
  onPlaybackProgress,
  onOpenContent,
}: Props) {
  const hostContainerRef = useRef<HTMLDivElement | null>(null);
  const playbackMenuRef = useRef<HTMLDivElement | null>(null);
  const playerRef = useRef<YouTubePlayer | null>(null);
  const progressTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const revealTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const resumeMaskTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const autoplayRetryTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const autoplayRetryCountRef = useRef(0);
  const autoplayEnabledRef = useRef(autoplayEnabled);
  const playbackRateRef = useRef(playbackRate);
  const isActiveRef = useRef(isActive);
  const didHandleClipEndRef = useRef(false);
  const didReportCompletionThresholdRef = useRef(false);
  const didUserInteractRef = useRef(false);
  const manualPauseRequestedRef = useRef(false);
  const isMutedRef = useRef(mutedPreference);
  const mutedPreferenceRef = useRef(mutedPreference);
  const isTouchLikeDeviceRef = useRef(false);
  const boundaryGateTargetRef = useRef<number | null>(null);
  const boundaryGateArmedRef = useRef(false);
  const boundaryGateAwaitingSeekRef = useRef(false);
  const boundarySeekRetryCountRef = useRef(0);
  const boundarySeekRetryAtRef = useRef<number | null>(null);

  const [isReady, setIsReady] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(mutedPreference);
  const [isSurfaceVisible, setIsSurfaceVisible] = useState(false);
  const [isResumeMaskVisible, setIsResumeMaskVisible] = useState(false);
  const [isBoundaryMaskVisible, setIsBoundaryMaskVisible] = useState(false);
  const [currentSec, setCurrentSec] = useState(0);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [isTouchLikeDevice, setIsTouchLikeDevice] = useState(false);
  const [isMobilePhoneDevice, setIsMobilePhoneDevice] = useState(false);
  const [isPlaybackMenuOpen, setIsPlaybackMenuOpen] = useState(false);

  const videoId = useMemo(() => extractVideoId(reel.video_url), [reel.video_url]);
  const videoProvider = useMemo(() => detectVideoProvider(reel.video_url), [reel.video_url]);
  const isYouTubeVideo = videoProvider === "youtube";
  const safeExternalUrl = useMemo(() => {
    if (isYouTubeVideo) return null;
    return normalizeExternalEmbedUrl(reel.video_url);
  }, [reel.video_url, isYouTubeVideo]);
  const clipStartRaw = Number(reel.t_start);
  const clipEndRaw = Number(reel.t_end);
  const clipStart = Number.isFinite(clipStartRaw) && clipStartRaw >= 0 ? clipStartRaw : 0;
  const clipEnd = Number.isFinite(clipEndRaw) && clipEndRaw > clipStart ? clipEndRaw : clipStart;
  const clipDuration = Math.max(0, clipEnd - clipStart);
  const activeCaption = useMemo(() => {
    const cues = reel.captions ?? [];
    if (!isActive || !isYouTubeVideo || cues.length === 0) {
      return null;
    }
    const cue = cues.find(
      (candidate) => currentSec >= Number(candidate.start) && currentSec < Number(candidate.end),
    );
    return cue?.text.trim() || null;
  }, [currentSec, isActive, isYouTubeVideo, reel.captions]);
  const progressPercent = clipDuration > 0 ? clamp((currentSec / clipDuration) * 100, 0, 100) : 0;
  const reelProgressStyle = { width: `${progressPercent}%` } as CSSProperties;
  const reelProgressDotStyle = { left: `${progressPercent}%` } as CSSProperties;
  const playbackRateIndex = Math.max(0, PLAYBACK_SPEED_OPTIONS.findIndex((rate) => rate === playbackRate));

  useEffect(() => {
    isMutedRef.current = isMuted;
  }, [isMuted]);

  useEffect(() => {
    mutedPreferenceRef.current = mutedPreference;
  }, [mutedPreference]);

  useEffect(() => {
    autoplayEnabledRef.current = autoplayEnabled;
  }, [autoplayEnabled]);

  useEffect(() => {
    isActiveRef.current = isActive;
  }, [isActive]);

  useLayoutEffect(() => {
    // Sync before paint so async player/state-change callbacks see the
    // updated rate on the same tick the prop changed. Plain useEffect
    // runs after paint and leaves handlers one tick stale.
    playbackRateRef.current = playbackRate;
  }, [playbackRate]);

  useEffect(() => {
    didHandleClipEndRef.current = false;
    didReportCompletionThresholdRef.current = false;
  }, [reel.reel_id]);

  useEffect(() => {
    setCurrentSec((prev) => clamp(prev, 0, clipDuration));
  }, [clipDuration]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const mediaQuery = window.matchMedia?.("(pointer: coarse)");
    const update = () => {
      const touchLike = detectTouchLikeDevice();
      isTouchLikeDeviceRef.current = touchLike;
      setIsTouchLikeDevice(touchLike);
      setIsMobilePhoneDevice(detectMobilePhoneDevice());
    };
    update();
    const onResize = () => update();
    window.addEventListener("resize", onResize);
    if (!mediaQuery) {
      return () => {
        window.removeEventListener("resize", onResize);
      };
    }
    const add = (mediaQuery as MediaQueryList & { addListener?: (listener: () => void) => void }).addListener;
    const remove = (mediaQuery as MediaQueryList & { removeListener?: (listener: () => void) => void }).removeListener;
    if (typeof mediaQuery.addEventListener === "function") {
      mediaQuery.addEventListener("change", update);
      return () => {
        mediaQuery.removeEventListener("change", update);
        window.removeEventListener("resize", onResize);
      };
    }
    if (typeof add === "function" && typeof remove === "function") {
      add.call(mediaQuery, update);
      return () => {
        remove.call(mediaQuery, update);
        window.removeEventListener("resize", onResize);
      };
    }
    return () => {
      window.removeEventListener("resize", onResize);
    };
  }, []);

  const stopProgressTimer = useCallback(() => {
    if (progressTimerRef.current) {
      clearInterval(progressTimerRef.current);
      progressTimerRef.current = null;
    }
  }, []);

  const destroyPlayerSafely = useCallback(() => {
    const player = playerRef.current;
    if (!player) {
      return;
    }
    playerRef.current = null;
    try {
      player.destroy();
    } catch (error) {
      console.warn("YouTube player destroy error:", error);
    } finally {
      const container = hostContainerRef.current;
      if (container) {
        try {
          container.replaceChildren();
        } catch (error) {
          console.warn("YouTube host clear error:", error);
        }
      }
    }
  }, []);

  const clearHostContainer = useCallback(() => {
    const container = hostContainerRef.current;
    if (!container) {
      return;
    }
    try {
      container.replaceChildren();
    } catch (error) {
      console.warn("YouTube host clear error:", error);
    }
  }, []);

  const syncProgress = useCallback(() => {
    const player = playerRef.current;
    if (!player) {
      return;
    }
    const now = clamp(player.getCurrentTime(), clipStart, clipEnd);
    const rel = clamp(now - clipStart, 0, clipDuration);
    setCurrentSec(rel);
  }, [clipDuration, clipEnd, clipStart]);

  const applyPlaybackRateToPlayer = useCallback((player: YouTubePlayer | null, nextRate: number) => {
    if (!player?.setPlaybackRate) {
      return;
    }
    const availableRates = player.getAvailablePlaybackRates?.();
    const resolvedRate =
      Array.isArray(availableRates) && availableRates.length > 0 && !availableRates.includes(nextRate)
        ? (availableRates.includes(1) ? 1 : availableRates[0] ?? nextRate)
        : nextRate;
    try {
      player.setPlaybackRate(resolvedRate);
    } catch {
      // Ignore unsupported playback rate updates from the iframe API.
    }
  }, []);

  const armBoundaryGate = useCallback((player: YouTubePlayer, targetTime: number) => {
    boundaryGateTargetRef.current = targetTime;
    boundaryGateArmedRef.current = true;
    boundarySeekRetryCountRef.current = 0;
    boundarySeekRetryAtRef.current = null;
    player.mute();
    isMutedRef.current = true;
    setIsMuted(true);
    setIsBoundaryMaskVisible(true);
  }, []);

  const seekToBoundary = useCallback((player: YouTubePlayer, targetTime: number) => {
    const decodeStart = Math.max(0, targetTime - BOUNDARY_SEEK_PREROLL_SEC);
    armBoundaryGate(player, targetTime);
    boundaryGateAwaitingSeekRef.current = true;
    player.seekTo(decodeStart, true);
  }, [armBoundaryGate]);

  const releaseBoundaryGate = useCallback((player: YouTubePlayer) => {
    boundaryGateTargetRef.current = null;
    boundaryGateArmedRef.current = false;
    boundaryGateAwaitingSeekRef.current = false;
    boundarySeekRetryCountRef.current = 0;
    boundarySeekRetryAtRef.current = null;
    setIsBoundaryMaskVisible(false);
    setIsResumeMaskVisible(false);
    setIsSurfaceVisible(true);
    const keepMuted =
      mutedPreferenceRef.current ||
      (isTouchLikeDeviceRef.current && !didUserInteractRef.current);
    if (keepMuted) {
      player.mute();
      isMutedRef.current = true;
      setIsMuted(true);
      return;
    }
    player.unMute();
    isMutedRef.current = false;
    setIsMuted(false);
  }, []);

  const coverBoundaryAndMute = useCallback((player: YouTubePlayer) => {
    boundaryGateTargetRef.current = null;
    boundaryGateArmedRef.current = true;
    boundaryGateAwaitingSeekRef.current = false;
    boundarySeekRetryCountRef.current = 0;
    boundarySeekRetryAtRef.current = null;
    player.mute();
    isMutedRef.current = true;
    setIsMuted(true);
    setIsBoundaryMaskVisible(true);
  }, []);

  const startProgressTimer = useCallback(() => {
    stopProgressTimer();
    progressTimerRef.current = setInterval(() => {
      const player = playerRef.current;
      if (!player) {
        return;
      }
      const playerTime = player.getCurrentTime();
      if (!Number.isFinite(playerTime)) {
        return;
      }
      if (boundaryGateArmedRef.current) {
        const gateTarget = boundaryGateTargetRef.current;
        if (
          gateTarget !== null &&
          boundaryGateAwaitingSeekRef.current
        ) {
          if (!hasObservedBoundarySeek(playerTime, gateTarget)) {
            player.mute();
            isMutedRef.current = true;
            setIsMuted(true);
            setIsBoundaryMaskVisible(true);
            const nowMs = Date.now();
            const lastRetryAt = boundarySeekRetryAtRef.current;
            const retryAgeMs = lastRetryAt === null ? Infinity : nowMs - lastRetryAt;
            if (
              boundarySeekRetryCountRef.current < BOUNDARY_SEEK_MAX_RETRIES
              && retryAgeMs >= BOUNDARY_SEEK_RETRY_GRACE_MS
            ) {
              boundarySeekRetryCountRef.current += 1;
              boundarySeekRetryAtRef.current = nowMs;
              player.seekTo(
                Math.max(0, gateTarget - BOUNDARY_SEEK_PREROLL_SEC),
                true,
              );
              return;
            }
            if (
              boundarySeekRetryCountRef.current < BOUNDARY_SEEK_MAX_RETRIES
              || retryAgeMs < BOUNDARY_SEEK_RETRY_GRACE_MS
            ) {
              return;
            }
            if (
              playerTime < gateTarget
              || playerTime > gateTarget + BOUNDARY_SEEK_ACCEPTABLE_OVERSHOOT_SEC
            ) {
              boundaryGateAwaitingSeekRef.current = false;
              manualPauseRequestedRef.current = true;
              player.pauseVideo();
              setIsPlaying(false);
              setLoadError(BOUNDARY_ALIGNMENT_ERROR);
              stopProgressTimer();
              return;
            }
          } else {
            boundarySeekRetryCountRef.current = 0;
            boundarySeekRetryAtRef.current = null;
          }
          boundaryGateAwaitingSeekRef.current = false;
        }
        if (
          gateTarget !== null &&
          hasReachedVerifiedClipStart(playerTime, gateTarget)
        ) {
          releaseBoundaryGate(player);
        } else {
          player.mute();
          isMutedRef.current = true;
          setIsMuted(true);
          setIsBoundaryMaskVisible(true);
          return;
        }
      }
      const now = clamp(playerTime, clipStart, clipEnd);
      const fraction = clipDuration > 0 ? clamp((now - clipStart) / clipDuration, 0, 1) : 0;
      if (fraction >= 0.8 && !didReportCompletionThresholdRef.current) {
        didReportCompletionThresholdRef.current = true;
        onPlaybackProgress?.(fraction, false);
      }
      // The iframe's integer `end` parameter is only a safety net. Enforce the
      // authoritative floating-point boundary ourselves, including at 2x.
      if (hasReachedVerifiedClipEnd(playerTime, clipEnd)) {
        onPlaybackProgress?.(1, true);
        if (!didHandleClipEndRef.current) {
          didHandleClipEndRef.current = true;
          manualPauseRequestedRef.current = true;
          coverBoundaryAndMute(player);
          player.pauseVideo();
          setIsPlaying(false);
          setCurrentSec(clipDuration);
          stopProgressTimer();
          if (autoplayEnabledRef.current && isActive && onRequestNextReel) {
            onRequestNextReel();
          }
        }
        return;
      }
      const rel = clamp(now - clipStart, 0, clipDuration);
      setCurrentSec(rel);
    }, CLIP_END_POLL_INTERVAL_MS);
  }, [
    clipDuration,
    clipEnd,
    clipStart,
    coverBoundaryAndMute,
    isActive,
    onPlaybackProgress,
    onRequestNextReel,
    releaseBoundaryGate,
    stopProgressTimer,
  ]);

  const retryBoundaryAlignment = useCallback(() => {
    const player = playerRef.current;
    if (!player) {
      return;
    }
    setLoadError(null);
    manualPauseRequestedRef.current = false;
    seekToBoundary(player, clipStart);
    applyPlaybackRateToPlayer(player, playbackRateRef.current);
    player.playVideo();
    setIsPlaying(true);
    startProgressTimer();
  }, [applyPlaybackRateToPlayer, clipStart, seekToBoundary, startProgressTimer]);

  useEffect(() => {
    if (!isYouTubeVideo || !isActive || !isReady || !isPlaying) {
      return;
    }
    startProgressTimer();
    return () => {
      stopProgressTimer();
    };
  }, [clipDuration, clipEnd, clipStart, isActive, isPlaying, isReady, isYouTubeVideo, startProgressTimer, stopProgressTimer]);

  const clearRevealTimer = useCallback(() => {
    if (revealTimerRef.current) {
      clearTimeout(revealTimerRef.current);
      revealTimerRef.current = null;
    }
  }, []);

  const clearResumeMaskTimer = useCallback(() => {
    if (resumeMaskTimerRef.current) {
      clearTimeout(resumeMaskTimerRef.current);
      resumeMaskTimerRef.current = null;
    }
  }, []);

  const clearAutoplayRetryTimer = useCallback(() => {
    if (autoplayRetryTimerRef.current) {
      clearTimeout(autoplayRetryTimerRef.current);
      autoplayRetryTimerRef.current = null;
    }
  }, []);

  const scheduleSurfaceReveal = useCallback(
    (delayMs: number) => {
      clearRevealTimer();
      setIsSurfaceVisible(false);
      revealTimerRef.current = setTimeout(() => {
        setIsSurfaceVisible(true);
        revealTimerRef.current = null;
      }, delayMs);
    },
    [clearRevealTimer],
  );

  const showResumeMask = useCallback(
    (durationMs: number) => {
      clearResumeMaskTimer();
      setIsResumeMaskVisible(true);
      resumeMaskTimerRef.current = setTimeout(() => {
        setIsResumeMaskVisible(false);
        resumeMaskTimerRef.current = null;
      }, durationMs);
    },
    [clearResumeMaskTimer],
  );

  useEffect(() => {
    return () => {
      stopProgressTimer();
      clearRevealTimer();
      clearResumeMaskTimer();
      clearAutoplayRetryTimer();
      destroyPlayerSafely();
    };
  }, [clearAutoplayRetryTimer, clearRevealTimer, clearResumeMaskTimer, destroyPlayerSafely, stopProgressTimer]);

  useEffect(() => {
    if (!isPlaybackMenuOpen) {
      return;
    }
    const onPointerDown = (event: PointerEvent) => {
      const target = event.target as Node | null;
      if (!target || playbackMenuRef.current?.contains(target)) {
        return;
      }
      setIsPlaybackMenuOpen(false);
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setIsPlaybackMenuOpen(false);
      }
    };
    window.addEventListener("pointerdown", onPointerDown);
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("pointerdown", onPointerDown);
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [isPlaybackMenuOpen]);

  useEffect(() => {
    if (!isActive) {
      setIsPlaybackMenuOpen(false);
    }
  }, [isActive]);

  useEffect(() => {
    stopProgressTimer();
    clearRevealTimer();
    clearResumeMaskTimer();
    clearAutoplayRetryTimer();
    autoplayRetryCountRef.current = 0;
    setCurrentSec(0);
    setIsPlaying(false);
    setIsReady(false);
    isMutedRef.current = mutedPreference;
    setIsMuted(mutedPreference);
    setIsSurfaceVisible(false);
    setIsResumeMaskVisible(false);
    boundaryGateTargetRef.current = isYouTubeVideo && isActive ? clipStart : null;
    boundaryGateArmedRef.current = isYouTubeVideo && isActive;
    boundaryGateAwaitingSeekRef.current = false;
    setIsBoundaryMaskVisible(isYouTubeVideo && isActive);
    setLoadError(null);
    didUserInteractRef.current = false;
    manualPauseRequestedRef.current = false;

    destroyPlayerSafely();

    if (!isActive) {
      return;
    }
    if (!isYouTubeVideo) {
      setIsReady(true);
      setIsPlaying(true);
      setCurrentSec(0);
      setIsSurfaceVisible(true);
      setIsResumeMaskVisible(false);
      setIsBoundaryMaskVisible(false);
      setLoadError(null);
      return;
    }
    if (!videoId) {
      setLoadError("Invalid YouTube clip URL");
      boundaryGateArmedRef.current = false;
      boundaryGateTargetRef.current = null;
      boundaryGateAwaitingSeekRef.current = false;
      setIsBoundaryMaskVisible(false);
      return;
    }
    if (!hostContainerRef.current) {
      return;
    }

    let cancelled = false;
    void loadYouTubeIframeApi()
      .then(() => {
        if (cancelled || !hostContainerRef.current) {
          return;
        }
        const yt = (window as any).YT;
        if (!yt?.Player) {
          setLoadError("YouTube player API unavailable");
          boundaryGateArmedRef.current = false;
          boundaryGateTargetRef.current = null;
          boundaryGateAwaitingSeekRef.current = false;
          setIsBoundaryMaskVisible(false);
          return;
        }
        clearHostContainer();
        const mountNode = document.createElement("div");
        mountNode.className = "h-full w-full";
        hostContainerRef.current.appendChild(mountNode);
        playerRef.current = new yt.Player(mountNode, {
          width: "100%",
          height: "100%",
          videoId,
          playerVars: {
            autoplay: 1,
            controls: 0,
            disablekb: 1,
            fs: 0,
            rel: 0,
            playsinline: 1,
            iv_load_policy: 3,
            start: Math.floor(clipStart),
            end: Math.ceil(clipEnd),
            mute: 1,
            enablejsapi: 1,
            origin: window.location.origin,
          },
          events: {
            onReady: (event: any) => {
              if (cancelled) {
                return;
              }
              applyPlaybackRateToPlayer(event.target as YouTubePlayer, playbackRateRef.current);
              const tryAutoplay = () => {
                if (cancelled || !isActive || didUserInteractRef.current) {
                  return;
                }
                seekToBoundary(event.target as YouTubePlayer, clipStart);
                event.target.playVideo();
              };
              const queueAutoplayRetry = () => {
                clearAutoplayRetryTimer();
                if (cancelled || !isActive || didUserInteractRef.current) {
                  return;
                }
                if (autoplayRetryCountRef.current >= AUTOPLAY_MAX_RETRIES) {
                  return;
                }
                autoplayRetryTimerRef.current = setTimeout(() => {
                  autoplayRetryTimerRef.current = null;
                  if (cancelled || !isActive || didUserInteractRef.current) {
                    return;
                  }
                  const playerStateValue =
                    typeof event.target.getPlayerState === "function" ? event.target.getPlayerState() : null;
                  if (playerStateValue === yt.PlayerState.PLAYING) {
                    return;
                  }
                  autoplayRetryCountRef.current += 1;
                  tryAutoplay();
                  queueAutoplayRetry();
                }, AUTOPLAY_RETRY_DELAY_MS);
              };
              autoplayRetryCountRef.current = 0;
              // Mobile browsers commonly block autoplay with sound; always start muted and retry.
              tryAutoplay();
              queueAutoplayRetry();
              setIsReady(true);
              setIsPlaying(false);
              setCurrentSec(0);
              setIsSurfaceVisible(true);
              scheduleSurfaceReveal(PLAYER_REVEAL_DELAY_MS);
            },
            onStateChange: (event: any) => {
              if (cancelled) {
                return;
              }
              const state = event.data;
              const playerState = yt.PlayerState;
              if (state === playerState.PLAYING) {
                clearAutoplayRetryTimer();
                if (
                  boundaryGateArmedRef.current &&
                  boundaryGateTargetRef.current === null
                ) {
                  manualPauseRequestedRef.current = true;
                  coverBoundaryAndMute(event.target as YouTubePlayer);
                  event.target.pauseVideo();
                  setIsPlaying(false);
                  stopProgressTimer();
                  return;
                }
                autoplayRetryCountRef.current = 0;
                didHandleClipEndRef.current = false;
                manualPauseRequestedRef.current = false;
                setIsPlaying(true);
                startProgressTimer();
              } else if (state === playerState.PAUSED) {
                setIsPlaying(false);
                setIsResumeMaskVisible(false);
                stopProgressTimer();
                syncProgress();
                if (
                  isActive
                  && !manualPauseRequestedRef.current
                  && !didUserInteractRef.current
                  && autoplayRetryCountRef.current < AUTOPLAY_MAX_RETRIES
                ) {
                  clearAutoplayRetryTimer();
                  autoplayRetryTimerRef.current = setTimeout(() => {
                    autoplayRetryTimerRef.current = null;
                    if (cancelled || !isActive || manualPauseRequestedRef.current) {
                      return;
                    }
                    autoplayRetryCountRef.current += 1;
                    seekToBoundary(event.target as YouTubePlayer, clipStart);
                    event.target.playVideo();
                  }, AUTOPLAY_RETRY_DELAY_MS);
                }
              } else if (state === playerState.ENDED) {
                clearAutoplayRetryTimer();
                onPlaybackProgress?.(1, true);
                if (!didHandleClipEndRef.current) {
                  didHandleClipEndRef.current = true;
                  manualPauseRequestedRef.current = true;
                  coverBoundaryAndMute(event.target as YouTubePlayer);
                  event.target.pauseVideo();
                  setIsPlaying(false);
                  setCurrentSec(clipDuration);
                  stopProgressTimer();
                  if (autoplayEnabledRef.current && isActive && onRequestNextReel) {
                    onRequestNextReel();
                  }
                }
                return;
              } else if ((state === playerState.UNSTARTED || state === playerState.CUED) && isActive) {
                // Retry autoplay for devices that initially report cued/unstarted.
                if (
                  boundaryGateArmedRef.current &&
                  boundaryGateTargetRef.current === null
                ) {
                  coverBoundaryAndMute(event.target as YouTubePlayer);
                  return;
                }
                manualPauseRequestedRef.current = false;
                if (autoplayRetryCountRef.current < AUTOPLAY_MAX_RETRIES) {
                  autoplayRetryCountRef.current += 1;
                }
                if (boundaryGateArmedRef.current || isMutedRef.current) {
                  event.target.mute();
                  setIsMuted(true);
                } else {
                  event.target.unMute();
                  setIsMuted(false);
                }
                event.target.playVideo();
              }
            },
            onError: () => {
              if (cancelled) {
                return;
              }
              clearAutoplayRetryTimer();
              setLoadError("Could not load this YouTube clip");
              setIsSurfaceVisible(false);
              setIsResumeMaskVisible(false);
              boundaryGateArmedRef.current = false;
              boundaryGateTargetRef.current = null;
              boundaryGateAwaitingSeekRef.current = false;
              setIsBoundaryMaskVisible(false);
              setIsPlaying(false);
              stopProgressTimer();
            },
          },
        });
      })
      .catch(() => {
        if (!cancelled) {
          setLoadError("Could not initialize the YouTube player");
          boundaryGateArmedRef.current = false;
          boundaryGateTargetRef.current = null;
          boundaryGateAwaitingSeekRef.current = false;
          setIsBoundaryMaskVisible(false);
        }
      });

    return () => {
      cancelled = true;
      stopProgressTimer();
      clearRevealTimer();
      clearResumeMaskTimer();
      clearAutoplayRetryTimer();
      destroyPlayerSafely();
      clearHostContainer();
    };
  }, [
    clearHostContainer,
    clearRevealTimer,
    clearResumeMaskTimer,
    clipDuration,
    clipEnd,
    clipStart,
    coverBoundaryAndMute,
    isActive,
    scheduleSurfaceReveal,
    seekToBoundary,
    showResumeMask,
    startProgressTimer,
    stopProgressTimer,
    syncProgress,
    videoId,
    isYouTubeVideo,
    onPlaybackProgress,
    onRequestNextReel,
    destroyPlayerSafely,
    clearAutoplayRetryTimer,
  ]);

  useEffect(() => {
    setIsMuted(mutedPreference);
    if (!isYouTubeVideo) {
      return;
    }
    const player = playerRef.current;
    if (!player || !isActive || !isReady) {
      return;
    }
    if (boundaryGateArmedRef.current) {
      player.mute();
      isMutedRef.current = true;
      setIsMuted(true);
      return;
    }
    if (isTouchLikeDevice && !didUserInteractRef.current) {
      player.mute();
      isMutedRef.current = true;
      setIsMuted(true);
      return;
    }
    if (mutedPreference) {
      player.mute();
      isMutedRef.current = true;
    } else {
      player.unMute();
      isMutedRef.current = false;
    }
  }, [isActive, isPlaying, isReady, mutedPreference, isTouchLikeDevice, isYouTubeVideo]);

  const togglePlayPause = useCallback(() => {
    if (!isYouTubeVideo) {
      return;
    }
    const player = playerRef.current;
    if (!player || !isReady) {
      return;
    }
    didUserInteractRef.current = true;
    clearAutoplayRetryTimer();
    if (
      !isPlaying &&
      !mutedPreference &&
      !boundaryGateArmedRef.current
    ) {
      player.unMute();
      isMutedRef.current = false;
      setIsMuted(false);
    }
    if (isPlaying) {
      manualPauseRequestedRef.current = true;
      player.pauseVideo();
      setIsPlaying(false);
      setIsResumeMaskVisible(false);
      stopProgressTimer();
      syncProgress();
      return;
    }
    if (clipDuration > 0 && currentSec >= clipDuration - 0.05) {
      didHandleClipEndRef.current = false;
      setCurrentSec(0);
      seekToBoundary(player, clipStart);
    }
    manualPauseRequestedRef.current = false;
    showResumeMask(RESUME_MASK_MS);
    scheduleSurfaceReveal(PLAYER_REVEAL_DELAY_MS);
    setIsSurfaceVisible(true);
    player.playVideo();
    setIsPlaying(true);
    startProgressTimer();
  }, [
    clearAutoplayRetryTimer,
    clipDuration,
    clipStart,
    currentSec,
    isPlaying,
    isReady,
    isYouTubeVideo,
    mutedPreference,
    scheduleSurfaceReveal,
    seekToBoundary,
    showResumeMask,
    startProgressTimer,
    stopProgressTimer,
    syncProgress,
  ]);

  const toggleMute = useCallback(() => {
    if (!isYouTubeVideo) {
      return;
    }
    const player = playerRef.current;
    if (!player || !isReady) {
      return;
    }
    didUserInteractRef.current = true;
    manualPauseRequestedRef.current = false;
    clearAutoplayRetryTimer();
    const nextMuted = !isMuted;
    mutedPreferenceRef.current = nextMuted;
    if (nextMuted) {
      player.mute();
      isMutedRef.current = true;
      setIsMuted(true);
    } else {
      if (boundaryGateArmedRef.current) {
        player.mute();
        isMutedRef.current = true;
        setIsMuted(true);
      } else {
        player.unMute();
        isMutedRef.current = false;
        setIsMuted(false);
      }
    }
    onMutedPreferenceChange(nextMuted);
  }, [clearAutoplayRetryTimer, isMuted, isReady, isYouTubeVideo, onMutedPreferenceChange]);

  useEffect(() => {
    if (!isYouTubeVideo) {
      return;
    }
    const player = playerRef.current;
    if (!player || !isReady) {
      return;
    }
    applyPlaybackRateToPlayer(player, playbackRate);
  }, [applyPlaybackRateToPlayer, isReady, isYouTubeVideo, playbackRate]);

  const toggleAutoplayPreference = useCallback(() => {
    const nextEnabled = !autoplayEnabled;
    autoplayEnabledRef.current = nextEnabled;
    onAutoplayEnabledChange(nextEnabled);
  }, [autoplayEnabled, onAutoplayEnabledChange]);

  const handlePlaybackRateChange = useCallback((nextRate: number) => {
    playbackRateRef.current = nextRate;
    applyPlaybackRateToPlayer(playerRef.current, nextRate);
    onPlaybackRateChange(nextRate);
  }, [applyPlaybackRateToPlayer, onPlaybackRateChange]);

  const handlePlaybackRateSliderChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const nextIndex = clamp(Math.round(Number(event.target.value)), 0, PLAYBACK_SPEED_OPTIONS.length - 1);
    const nextRate = PLAYBACK_SPEED_OPTIONS[nextIndex] ?? 1;
    handlePlaybackRateChange(nextRate);
  }, [handlePlaybackRateChange]);

  const onSeek = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      if (!isYouTubeVideo) {
        return;
      }
      const player = playerRef.current;
      if (!player || !isReady) {
        return;
      }
      const rel = clamp(Number(event.target.value), 0, clipDuration);
      didUserInteractRef.current = true;
      clearAutoplayRetryTimer();
      didHandleClipEndRef.current = false;
      setCurrentSec(rel);
      manualPauseRequestedRef.current = false;
      const targetTime = clipStart + rel;
      seekToBoundary(player, targetTime);
      if (!isPlaying && isActive) {
        showResumeMask(RESUME_MASK_MS);
        scheduleSurfaceReveal(PLAYER_REVEAL_DELAY_MS);
        setIsSurfaceVisible(true);
        player.playVideo();
        setIsPlaying(true);
        startProgressTimer();
      }
    },
    [
      clearAutoplayRetryTimer,
      clipDuration,
      clipStart,
      isActive,
      isPlaying,
      isReady,
      isYouTubeVideo,
      scheduleSurfaceReveal,
      seekToBoundary,
      showResumeMask,
      startProgressTimer,
    ],
  );

  const stopFeedGesturePropagation = useCallback((event: React.SyntheticEvent<HTMLElement>) => {
    event.stopPropagation();
  }, []);

  useEffect(() => {
    if (!isActive) {
      return;
    }
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.repeat) {
        return;
      }
      const isSpace = event.code === "Space" || event.key === " ";
      const isMute = event.code === "KeyM" || event.key.toLowerCase() === "m";
      if (!isSpace && !isMute) {
        return;
      }
      const target = event.target as HTMLElement | null;
      if (target) {
        const tag = target.tagName;
        if (
          target.isContentEditable ||
          tag === "INPUT" ||
          tag === "TEXTAREA" ||
          tag === "SELECT" ||
          tag === "BUTTON"
        ) {
          return;
        }
      }
      event.preventDefault();
      if (isSpace) {
        togglePlayPause();
        return;
      }
      if (isMute) {
        toggleMute();
        return;
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [isActive, toggleMute, togglePlayPause]);

  const hidePlayerSurface = isActive && isYouTubeVideo && (!isReady || !isSurfaceVisible || Boolean(loadError));
  const showTransitionMask = isActive && isYouTubeVideo && isResumeMaskVisible;
  const showBoundaryMask = isActive && isYouTubeVideo && isBoundaryMaskVisible;
  const canToggleFromSurface = isYouTubeVideo && isActive && isReady && !loadError;
  const surfaceAriaLabel = isPlaying ? "Pause clip" : "Play clip";
  const controlsEnabled = isReady && isActive && isYouTubeVideo;
  const controlButtonClass = (active: boolean) =>
    `grid h-11 w-11 place-items-center rounded-full text-base transition-colors duration-200 focus-visible:bg-white/[0.07] disabled:pointer-events-none disabled:text-white/35 lg:h-10 lg:w-10 ${
      active
        ? "bg-white/12 text-white"
        : "bg-transparent text-white/88 hover:bg-white/[0.07] hover:text-white"
    }`;
  const controlsChromeClass = isMobilePhoneDevice
    ? "rounded-2xl bg-black/70 px-3 py-2 backdrop-blur-md"
    : "px-0 py-0";

  return (
    <section className="relative h-full min-h-full w-full snap-start overflow-hidden rounded-none bg-black lg:rounded-[1.25rem]">
      {isActive ? (
        <div className="absolute inset-0 overflow-hidden">
          {isYouTubeVideo ? (
            <div
              ref={hostContainerRef}
              data-youtube-crop="true"
              className="pointer-events-none absolute inset-x-0 w-full"
              style={{ top: "-10%", height: "120%" }}
            />
          ) : safeExternalUrl ? (
            <iframe
              src={safeExternalUrl}
              title={reel.video_title || reel.concept_title || "Community reel"}
              className="absolute inset-0 h-full w-full"
              loading="eager"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
              sandbox="allow-scripts allow-popups allow-popups-to-escape-sandbox allow-presentation"
              allowFullScreen
            />
          ) : (
            <div className="flex h-full w-full items-center justify-center bg-black/70 text-xs uppercase tracking-[0.12em] text-white/55">
              Invalid video URL
            </div>
          )}
        </div>
      ) : (
        <div className="flex h-full w-full items-center justify-center bg-black/70 text-xs uppercase tracking-[0.12em] text-white/55">
          Scroll to play
        </div>
      )}

      <div
        className={`pointer-events-none absolute inset-0 z-10 bg-black ${
          hidePlayerSurface ? "opacity-100" : "opacity-0"
        }`}
      />

      <div
        className={`pointer-events-none absolute inset-0 z-[14] bg-black ${
          showTransitionMask ? "opacity-100" : "opacity-0"
        }`}
      />

      <div
        className={`pointer-events-none absolute inset-0 z-[14] bg-black ${
          showBoundaryMask ? "opacity-100" : "opacity-0"
        }`}
      />

      <div
        data-top-chrome="reel-player"
        className="top-nav-fade pointer-events-none absolute inset-x-0 top-0 z-[16] h-20"
      />

      <div
        data-bottom-chrome="reel-player"
        className="pointer-events-none absolute inset-x-0 bottom-0 z-[16] h-32 bg-gradient-to-t from-black via-black/75 to-transparent"
      />

      {isActive && isYouTubeVideo ? (
        <button
          type="button"
          aria-label={surfaceAriaLabel}
          onClick={togglePlayPause}
          disabled={!canToggleFromSurface}
          className="absolute inset-0 z-[15] cursor-default bg-transparent disabled:cursor-not-allowed"
        />
      ) : null}

      {isActive && isYouTubeVideo && isReady && !isPlaying && !loadError ? (
        <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center">
          <div className="grid h-14 w-14 place-items-center rounded-full bg-black/72 text-white/90 backdrop-blur-sm">
            <i className="fa-solid fa-play text-base" aria-hidden="true" />
          </div>
        </div>
      ) : null}

      {activeCaption ? (
        <div
          data-reel-caption="true"
          className="pointer-events-none absolute inset-x-5 bottom-24 z-[19] flex justify-center lg:bottom-28"
        >
          <span className="max-w-[92%] rounded-lg bg-black/80 px-3 py-1.5 text-center text-base font-semibold leading-snug text-white shadow-sm [text-shadow:0_1px_2px_rgba(0,0,0,0.9)] sm:text-lg lg:max-w-[88%] lg:text-xl">
            {activeCaption}
          </span>
        </div>
      ) : null}

      <div
        data-reel-control="true"
        onWheel={stopFeedGesturePropagation}
        onTouchStart={stopFeedGesturePropagation}
        onTouchMove={stopFeedGesturePropagation}
        onTouchEnd={stopFeedGesturePropagation}
        className="absolute inset-x-0 bottom-0 z-20 p-3"
      >
        <div className={controlsChromeClass}>
          <div className="mb-0 flex items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <div className="inline-flex h-9 items-center rounded-full bg-black/68 px-3 text-[10px] font-semibold uppercase tracking-[0.08em] text-white/92 backdrop-blur-sm">
                {isYouTubeVideo ? `${formatClock(currentSec)} / ${formatClock(clipDuration)}` : "Embedded Reel"}
              </div>
              {onOpenContent ? (
                <button
                  type="button"
                  data-reel-control="true"
                  onClick={onOpenContent}
                  className="inline-flex h-11 items-center rounded-full bg-black/68 px-3 text-[10px] font-semibold uppercase tracking-[0.1em] text-white/90 backdrop-blur-sm transition-colors hover:bg-white/[0.07] focus-visible:bg-white/[0.07] lg:hidden"
                >
                  Content
                </button>
              ) : null}
            </div>
            <div className="flex items-center gap-2">
              {isYouTubeVideo ? (
                <div ref={playbackMenuRef} className="relative flex items-center gap-2">
                  <button
                    type="button"
                    data-reel-control="true"
                    onClick={togglePlayPause}
                    className={controlButtonClass(Boolean(isPlaying))}
                    disabled={!controlsEnabled}
                    aria-label={isPlaying ? "Pause" : "Play"}
                    title={isPlaying ? "Pause" : "Play"}
                  >
                    <i className={`fa-solid ${isPlaying ? "fa-pause" : "fa-play"}`} aria-hidden="true" />
                  </button>
                  <button
                    type="button"
                    data-reel-control="true"
                    onClick={toggleMute}
                    className={controlButtonClass(!isMuted)}
                    disabled={!controlsEnabled}
                    aria-label={isMuted ? "Unmute" : "Mute"}
                  >
                    <i className={`fa-solid ${isMuted ? "fa-volume-xmark" : "fa-volume-high"}`} aria-hidden="true" />
                  </button>
                  <button
                    type="button"
                    data-reel-control="true"
                    onClick={() => setIsPlaybackMenuOpen((prev) => !prev)}
                    className={controlButtonClass(isPlaybackMenuOpen)}
                    aria-label="Playback settings"
                    aria-expanded={isPlaybackMenuOpen}
                    aria-haspopup="menu"
                    title="Playback settings"
                  >
                    <i className="fa-solid fa-gear" aria-hidden="true" />
                  </button>
                  <div
                    className={`absolute bottom-full right-0 z-30 mb-2 w-56 transition-opacity duration-300 motion-reduce:transition-none ${
                      isPlaybackMenuOpen
                        ? "pointer-events-auto opacity-100"
                        : "pointer-events-none opacity-0"
                    }`}
                  >
                    <div
                      role="menu"
                      className="overflow-hidden rounded-2xl bg-[#202020] p-1.5"
                    >
                      <button
                        type="button"
                        data-reel-control="true"
                        onClick={toggleAutoplayPreference}
                        aria-pressed={autoplayEnabled}
                        className="flex w-full items-center justify-between gap-3 rounded-xl px-3 py-2 text-left text-xs text-white/90 transition-colors hover:bg-white/[0.07]"
                      >
                        <span className="flex items-center gap-2.5">
                          <i className="fa-solid fa-play text-[11px] text-white/80" aria-hidden="true" />
                          Autoplay
                        </span>
                        <span
                          aria-hidden="true"
                          className={`relative inline-flex h-6 w-10 shrink-0 rounded-full transition-colors duration-300 ${
                            autoplayEnabled ? "bg-white" : "bg-white/24"
                          }`}
                        >
                          <span
                            className={`absolute top-0.5 h-[18px] w-[18px] rounded-full transition-transform duration-300 ease-out motion-reduce:transition-none ${
                              autoplayEnabled ? "translate-x-[20px] bg-black" : "translate-x-[2px] bg-white"
                            }`}
                          />
                        </span>
                      </button>
                      <div className="mt-1 rounded-xl px-3 py-2">
                        <div className="mb-2 flex items-center justify-between gap-2">
                          <div className="flex items-center gap-2.5 text-[10px] font-semibold uppercase tracking-[0.1em] text-white/58">
                            <i className="fa-solid fa-gauge-high text-[11px] text-white/72" aria-hidden="true" />
                            Speed
                          </div>
                          <span className="rounded-full bg-white px-2 py-0.5 text-[10px] font-semibold tracking-[0.04em] text-black">
                            {formatPlaybackRate(playbackRate)}
                          </span>
                        </div>
                        <div className="relative px-1 py-1">
                          <input
                            type="range"
                            min={0}
                            max={PLAYBACK_SPEED_OPTIONS.length - 1}
                            step={1}
                            value={playbackRateIndex}
                            onChange={handlePlaybackRateSliderChange}
                            className="playback-speed-range relative z-10 block h-4 w-full cursor-pointer"
                            aria-label="Playback speed"
                            aria-valuetext={formatPlaybackRate(playbackRate)}
                          />
                        </div>
                        <div className="mt-0.5 flex items-center justify-between px-1 text-[9px] font-semibold text-white/58">
                          <span>{PLAYBACK_SPEED_OPTIONS[0]}</span>
                          <span>{PLAYBACK_SPEED_OPTIONS[PLAYBACK_SPEED_OPTIONS.length - 1]}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <a
                  href={reel.video_url}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex h-10 items-center rounded-full bg-black/68 px-3 text-[10px] font-semibold uppercase tracking-[0.1em] text-white/90 backdrop-blur-sm transition-colors hover:bg-white/[0.07] focus-visible:bg-white/[0.07]"
                >
                  Open
                </a>
              )}
            </div>
          </div>

          {isYouTubeVideo ? (
            <div className="group/progress relative h-4 w-full leading-none">
              <div
                aria-hidden="true"
                className="pointer-events-none absolute inset-x-0 top-1/2 h-0.5 -translate-y-1/2 overflow-hidden rounded-full bg-[rgba(255,255,255,0.38)]"
              >
                <div className="h-full rounded-full bg-white" style={reelProgressStyle} />
              </div>
              <div
                aria-hidden="true"
                style={reelProgressDotStyle}
                className="pointer-events-none absolute top-1/2 z-[1] h-2 w-2 -translate-x-1/2 -translate-y-1/2 rounded-full bg-white"
              />
              <input
                data-reel-control="true"
                type="range"
                min={0}
                max={clipDuration}
                step={0.1}
                value={currentSec}
                onChange={onSeek}
                className="reel-range relative z-10 block h-4 w-full cursor-pointer disabled:opacity-40"
                disabled={!controlsEnabled}
              />
            </div>
          ) : (
            <p className="text-[10px] font-semibold uppercase tracking-[0.08em] text-white/60">Platform-managed playback</p>
          )}

          {loadError ? (
            <div className="mt-2 inline-flex items-center gap-2 rounded-full bg-black/76 px-3 py-1 text-xs text-white/78">
              <span>{loadError}</span>
              {loadError === BOUNDARY_ALIGNMENT_ERROR ? (
                <button
                  type="button"
                  data-reel-control="true"
                  onClick={retryBoundaryAlignment}
                  className="rounded-full bg-white/14 px-2 py-0.5 font-semibold text-white transition-colors hover:bg-white/[0.07]"
                >
                  Retry
                </button>
              ) : null}
            </div>
          ) : null}
        </div>
      </div>
    </section>
  );
}
