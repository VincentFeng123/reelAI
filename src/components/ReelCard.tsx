"use client";

import { type CSSProperties, useCallback, useEffect, useMemo, useRef, useState } from "react";

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
  onOpenContent?: () => void;
};

type YouTubePlayer = {
  destroy: () => void;
  pauseVideo: () => void;
  playVideo: () => void;
  seekTo: (seconds: number, allowSeekAhead: boolean) => void;
  getCurrentTime: () => number;
  getDuration?: () => number;
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

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
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
  const didUserInteractRef = useRef(false);
  const manualPauseRequestedRef = useRef(false);
  const isMutedRef = useRef(mutedPreference);

  const [isReady, setIsReady] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(mutedPreference);
  const [isSurfaceVisible, setIsSurfaceVisible] = useState(false);
  const [isResumeMaskVisible, setIsResumeMaskVisible] = useState(false);
  const [currentSec, setCurrentSec] = useState(0);
  const [detectedDurationSec, setDetectedDurationSec] = useState<number | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [isTouchLikeDevice, setIsTouchLikeDevice] = useState(false);
  const [isMobilePhoneDevice, setIsMobilePhoneDevice] = useState(false);
  const [isPlaybackMenuOpen, setIsPlaybackMenuOpen] = useState(false);

  const videoId = useMemo(() => extractVideoId(reel.video_url), [reel.video_url]);
  const videoProvider = useMemo(() => detectVideoProvider(reel.video_url), [reel.video_url]);
  const isYouTubeVideo = videoProvider === "youtube";
  const isCommunityImported = (reel.relevance_reason || "").trim() === "Opened from a community set.";
  const clipStartRaw = Number(reel.t_start);
  const clipEndRaw = Number(reel.t_end);
  const clipStart = Number.isFinite(clipStartRaw) && clipStartRaw >= 0 ? clipStartRaw : 0;
  const configuredClipEnd =
    Number.isFinite(clipEndRaw) && clipEndRaw > clipStart
      ? clipEndRaw
      : clipStart + 1;
  const configuredClipDuration = Math.max(0, configuredClipEnd - clipStart);
  const hasClipDurationMetadata = Number.isFinite(reel.clip_duration_sec) && Number(reel.clip_duration_sec) > 0;
  const hasExplicitEndFlag = reel.community_has_explicit_end === true;
  const looksLikeFallbackWindow = Math.abs(configuredClipDuration - 180) < 0.25;
  const hasCreatorClipRange =
    hasExplicitEndFlag ||
    hasClipDurationMetadata ||
    (isCommunityImported && configuredClipDuration > 0.5 && !looksLikeFallbackWindow);
  const detectedClipEnd =
    detectedDurationSec !== null && detectedDurationSec > clipStart
      ? detectedDurationSec
      : null;
  const shouldUseDetectedDurationAsMax = isCommunityImported && !hasCreatorClipRange;
  const shouldAutoExtendNearFullClip =
    isCommunityImported &&
    hasCreatorClipRange &&
    detectedClipEnd !== null &&
    clipStart <= 0.25 &&
    configuredClipEnd < detectedClipEnd &&
    configuredClipEnd >= detectedClipEnd * 0.93;
  const clipEnd =
    detectedClipEnd === null
      ? configuredClipEnd
      : shouldUseDetectedDurationAsMax || shouldAutoExtendNearFullClip
      ? detectedClipEnd
      : Math.max(clipStart + 1, Math.min(configuredClipEnd, detectedClipEnd));
  const clipDuration = Math.max(1, clipEnd - clipStart);
  const progressPercent = clipDuration > 0 ? clamp((currentSec / clipDuration) * 100, 0, 100) : 0;
  const reelProgressStyle = { width: `${progressPercent}%` } as CSSProperties;
  const reelProgressDotStyle = { left: `${progressPercent}%` } as CSSProperties;
  const playbackRateIndex = Math.max(0, PLAYBACK_SPEED_OPTIONS.findIndex((rate) => rate === playbackRate));

  useEffect(() => {
    isMutedRef.current = isMuted;
  }, [isMuted]);

  useEffect(() => {
    autoplayEnabledRef.current = autoplayEnabled;
  }, [autoplayEnabled]);

  useEffect(() => {
    playbackRateRef.current = playbackRate;
  }, [playbackRate]);

  useEffect(() => {
    setDetectedDurationSec(null);
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
      setIsTouchLikeDevice(detectTouchLikeDevice());
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

  const syncDetectedDurationFromPlayer = useCallback((player: YouTubePlayer | null) => {
    if (!isYouTubeVideo || !shouldUseDetectedDurationAsMax || detectedDurationSec !== null || !player) {
      return;
    }
    const raw = Number(player.getDuration?.());
    if (!Number.isFinite(raw) || raw <= 0) {
      return;
    }
    setDetectedDurationSec(raw);
  }, [detectedDurationSec, isYouTubeVideo, shouldUseDetectedDurationAsMax]);

  const syncProgress = useCallback(() => {
    const player = playerRef.current;
    if (!player) {
      return;
    }
    syncDetectedDurationFromPlayer(player);
    const now = clamp(player.getCurrentTime(), clipStart, clipEnd);
    const rel = clamp(now - clipStart, 0, clipDuration);
    setCurrentSec(rel);
  }, [clipDuration, clipEnd, clipStart, syncDetectedDurationFromPlayer]);

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

  const startProgressTimer = useCallback(() => {
    stopProgressTimer();
    progressTimerRef.current = setInterval(() => {
      const player = playerRef.current;
      if (!player) {
        return;
      }
      syncDetectedDurationFromPlayer(player);
      const now = clamp(player.getCurrentTime(), clipStart, clipEnd);
      if (now >= clipEnd - 0.15) {
        player.seekTo(clipStart, true);
        if (isActive) {
          player.playVideo();
        }
      }
      const rel = clamp(now - clipStart, 0, clipDuration);
      setCurrentSec(rel);
    }, 160);
  }, [clipDuration, clipEnd, clipStart, isActive, stopProgressTimer, syncDetectedDurationFromPlayer]);

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
    setIsMuted(mutedPreference);
    setIsSurfaceVisible(false);
    setIsResumeMaskVisible(false);
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
      setLoadError(null);
      return;
    }
    if (!videoId) {
      setLoadError("Invalid YouTube clip URL");
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
            autoplay: autoplayEnabledRef.current ? 1 : 0,
            controls: 0,
            disablekb: 1,
            fs: 0,
            rel: 0,
            playsinline: 1,
            iv_load_policy: 3,
            modestbranding: 1,
            start: clipStart,
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
              syncDetectedDurationFromPlayer(event.target as YouTubePlayer);
              const tryAutoplay = () => {
                if (cancelled || !isActive || !autoplayEnabledRef.current || didUserInteractRef.current) {
                  return;
                }
                event.target.mute();
                setIsMuted(true);
                event.target.seekTo(clipStart, true);
                event.target.playVideo();
              };
              const queueAutoplayRetry = () => {
                clearAutoplayRetryTimer();
                if (cancelled || !isActive || !autoplayEnabledRef.current || didUserInteractRef.current) {
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
              if (autoplayEnabledRef.current) {
                // Mobile browsers commonly block autoplay with sound; always start muted and retry.
                tryAutoplay();
                setIsMuted(true);
                queueAutoplayRetry();
              } else {
                clearAutoplayRetryTimer();
                if (mutedPreference) {
                  event.target.mute();
                  setIsMuted(true);
                } else {
                  event.target.unMute();
                  setIsMuted(false);
                }
              }
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
              syncDetectedDurationFromPlayer(event.target as YouTubePlayer);
              const state = event.data;
              const playerState = yt.PlayerState;
              const shouldResumePlayback = autoplayEnabledRef.current || didUserInteractRef.current;
              if (state === playerState.PLAYING) {
                clearAutoplayRetryTimer();
                autoplayRetryCountRef.current = 0;
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
                  && shouldResumePlayback
                  && !manualPauseRequestedRef.current
                  && autoplayRetryCountRef.current < AUTOPLAY_MAX_RETRIES
                ) {
                  clearAutoplayRetryTimer();
                  autoplayRetryTimerRef.current = setTimeout(() => {
                    autoplayRetryTimerRef.current = null;
                    if (cancelled || !isActive || manualPauseRequestedRef.current || !(autoplayEnabledRef.current || didUserInteractRef.current)) {
                      return;
                    }
                    autoplayRetryCountRef.current += 1;
                    if (isMutedRef.current) {
                      event.target.mute();
                      setIsMuted(true);
                    } else {
                      event.target.unMute();
                      setIsMuted(false);
                    }
                    event.target.seekTo(clipStart, true);
                    event.target.playVideo();
                  }, AUTOPLAY_RETRY_DELAY_MS);
                }
              } else if (state === playerState.ENDED) {
                clearAutoplayRetryTimer();
                if (!shouldResumePlayback) {
                  setIsPlaying(false);
                  setCurrentSec(0);
                  stopProgressTimer();
                  event.target.seekTo(clipStart, true);
                  return;
                }
                manualPauseRequestedRef.current = false;
                event.target.seekTo(clipStart, true);
                event.target.playVideo();
                setIsPlaying(true);
                setIsSurfaceVisible(true);
                setCurrentSec(0);
                scheduleSurfaceReveal(PLAYER_REVEAL_DELAY_MS);
                startProgressTimer();
              } else if ((state === playerState.UNSTARTED || state === playerState.CUED) && isActive && shouldResumePlayback) {
                // Retry autoplay for devices that initially report cued/unstarted.
                manualPauseRequestedRef.current = false;
                if (autoplayRetryCountRef.current < AUTOPLAY_MAX_RETRIES) {
                  autoplayRetryCountRef.current += 1;
                }
                if (isMutedRef.current) {
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
              setIsPlaying(false);
              stopProgressTimer();
            },
          },
        });
      })
      .catch(() => {
        if (!cancelled) {
          setLoadError("Could not initialize the YouTube player");
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
    clipStart,
    isActive,
    scheduleSurfaceReveal,
    showResumeMask,
    startProgressTimer,
    stopProgressTimer,
    syncProgress,
    syncDetectedDurationFromPlayer,
    videoId,
    isYouTubeVideo,
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
    if (isTouchLikeDevice && !didUserInteractRef.current) {
      player.mute();
      setIsMuted(true);
      return;
    }
    if (mutedPreference) {
      player.mute();
    } else {
      player.unMute();
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
    if (!isPlaying && !mutedPreference) {
      player.unMute();
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
    manualPauseRequestedRef.current = false;
    showResumeMask(RESUME_MASK_MS);
    scheduleSurfaceReveal(PLAYER_REVEAL_DELAY_MS);
    setIsSurfaceVisible(true);
    player.playVideo();
    setIsPlaying(true);
    startProgressTimer();
  }, [
    clearAutoplayRetryTimer,
    isPlaying,
    isReady,
    isYouTubeVideo,
    mutedPreference,
    scheduleSurfaceReveal,
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
    if (nextMuted) {
      player.mute();
      setIsMuted(true);
    } else {
      player.unMute();
      setIsMuted(false);
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
    if (!nextEnabled) {
      clearAutoplayRetryTimer();
      return;
    }
    if (!isYouTubeVideo) {
      return;
    }
    const player = playerRef.current;
    if (!player || !isReady || isPlaying || manualPauseRequestedRef.current) {
      return;
    }
    if (isMutedRef.current) {
      player.mute();
      setIsMuted(true);
    } else {
      player.unMute();
      setIsMuted(false);
    }
    didUserInteractRef.current = true;
    manualPauseRequestedRef.current = false;
    showResumeMask(RESUME_MASK_MS);
    scheduleSurfaceReveal(PLAYER_REVEAL_DELAY_MS);
    setIsSurfaceVisible(true);
    player.seekTo(clipStart + clamp(currentSec, 0, clipDuration), true);
    player.playVideo();
    setIsPlaying(true);
    startProgressTimer();
  }, [
    autoplayEnabled,
    clearAutoplayRetryTimer,
    clipDuration,
    clipStart,
    currentSec,
    isPlaying,
    isReady,
    isYouTubeVideo,
    onAutoplayEnabledChange,
    scheduleSurfaceReveal,
    showResumeMask,
    startProgressTimer,
  ]);

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
      setCurrentSec(rel);
      manualPauseRequestedRef.current = false;
      player.seekTo(clipStart + rel, true);
      if (!isPlaying && isActive) {
        showResumeMask(RESUME_MASK_MS);
        scheduleSurfaceReveal(PLAYER_REVEAL_DELAY_MS);
        setIsSurfaceVisible(true);
        player.playVideo();
        setIsPlaying(true);
        startProgressTimer();
      }
    },
    [clipDuration, clipStart, isActive, isPlaying, isReady, isYouTubeVideo, scheduleSurfaceReveal, showResumeMask, startProgressTimer],
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

  const hidePlayerSurface = isActive && isYouTubeVideo && (!isReady || !isPlaying || !isSurfaceVisible || Boolean(loadError));
  const showTransitionMask = isActive && isYouTubeVideo && isResumeMaskVisible;
  const canToggleFromSurface = isYouTubeVideo && isActive && isReady && !loadError;
  const surfaceAriaLabel = isPlaying ? "Pause clip" : "Play clip";
  const controlsEnabled = isReady && isActive && isYouTubeVideo;
  const controlButtonClass = (active: boolean) =>
    `grid h-9 w-9 place-items-center rounded-full text-base transition-colors duration-200 disabled:pointer-events-none disabled:text-white/35 ${
      active
        ? "bg-white/12 text-white"
        : "bg-transparent text-white/88 hover:bg-white/10 hover:text-white"
    }`;
  const controlsChromeClass = isMobilePhoneDevice
    ? "rounded-2xl border border-white/20 bg-black/70 px-3 py-2 shadow-[0_10px_26px_rgba(0,0,0,0.38)] backdrop-blur-md"
    : "px-0 py-0";

  return (
    <section className="relative h-full min-h-full w-full snap-start overflow-hidden rounded-none border-0 bg-transparent lg:rounded-3xl lg:border lg:border-white/20">
      {isActive ? (
        <div className="absolute inset-0 overflow-hidden">
          {isYouTubeVideo ? (
            <div
              ref={hostContainerRef}
              className="pointer-events-none absolute inset-0 h-full w-full"
            />
          ) : (
            <iframe
              src={reel.video_url}
              title={reel.video_title || reel.concept_title || "Community reel"}
            className="absolute inset-0 h-full w-full border-0"
            loading="eager"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
            sandbox="allow-scripts allow-same-origin allow-popups allow-popups-to-escape-sandbox allow-presentation"
            allowFullScreen
          />
        )}
        </div>
      ) : (
        <div className="flex h-full w-full items-center justify-center bg-black/70 text-xs uppercase tracking-[0.12em] text-white/55">
          Scroll to play
        </div>
      )}

      <div
        className={`pointer-events-none absolute inset-0 z-10 bg-black transition-opacity duration-200 ${
          hidePlayerSurface ? "opacity-95" : "opacity-0"
        }`}
      />

      <div
        className={`pointer-events-none absolute inset-0 z-[14] bg-black ${
          showTransitionMask ? "opacity-92" : "opacity-0"
        }`}
      />

      <div className="pointer-events-none absolute inset-x-0 top-0 z-[16] h-16 bg-black/95" />

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
          <div className="grid h-14 w-14 place-items-center rounded-full border border-white/28 bg-black/78 text-white/90">
            <i className="fa-solid fa-play text-base" aria-hidden="true" />
          </div>
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
          <div className="mb-2 flex items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <div className="inline-flex h-9 items-center rounded-full border border-white/30 bg-black/82 px-3 text-[10px] font-semibold uppercase tracking-[0.08em] text-white/92">
                {isYouTubeVideo ? `${formatClock(currentSec)} / ${formatClock(clipDuration)}` : "Embedded Reel"}
              </div>
              {onOpenContent ? (
                <button
                  type="button"
                  data-reel-control="true"
                  onClick={onOpenContent}
                  className="inline-flex h-8 items-center rounded-full border-[0.8px] border-white/30 px-3 text-[10px] font-semibold uppercase tracking-[0.1em] text-white/90 transition hover:bg-white/10 lg:hidden"
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
                    <i className="fa-solid fa-ellipsis" aria-hidden="true" />
                  </button>
                  <div
                    className={`absolute bottom-full right-0 z-30 mb-2 w-56 transition-opacity duration-180 ${
                      isPlaybackMenuOpen
                        ? "pointer-events-auto opacity-100"
                        : "pointer-events-none opacity-0"
                    }`}
                  >
                    <div
                      role="menu"
                      className="overflow-hidden rounded-2xl border border-white/15 bg-black p-1.5 shadow-[0_20px_48px_rgba(0,0,0,0.45)]"
                    >
                      <button
                        type="button"
                        data-reel-control="true"
                        onClick={toggleAutoplayPreference}
                        aria-pressed={autoplayEnabled}
                        className="flex w-full items-center justify-between gap-3 rounded-xl px-3 py-2 text-left text-xs text-white/90 transition hover:bg-white/10"
                      >
                        <span className="flex items-center gap-2.5">
                          <i className="fa-solid fa-play text-[11px] text-white/80" aria-hidden="true" />
                          Autoplay
                        </span>
                        <span
                          aria-hidden="true"
                          className={`relative inline-flex h-6 w-10 shrink-0 rounded-full border transition-colors duration-200 ${
                            autoplayEnabled ? "border-white bg-white" : "border-white/32 bg-black/50"
                          }`}
                        >
                          <span
                            className={`absolute top-0.5 h-[18px] w-[18px] rounded-full transition-transform duration-200 ${
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
                  className="inline-flex h-8 items-center rounded-full border-[0.8px] border-white/30 px-3 text-[10px] font-semibold uppercase tracking-[0.1em] text-white/90 transition hover:bg-white/10"
                >
                  Open
                </a>
              )}
            </div>
          </div>

          {isYouTubeVideo ? (
            <div className="group/progress relative w-full py-2">
              <div
                aria-hidden="true"
                className="pointer-events-none absolute inset-x-0 top-1/2 h-1.5 -translate-y-1/2 overflow-hidden rounded-full bg-[rgba(255,255,255,0.38)]"
              >
                <div className="h-full rounded-full bg-white" style={reelProgressStyle} />
              </div>
              <div
                aria-hidden="true"
                style={reelProgressDotStyle}
                className="pointer-events-none absolute top-1/2 z-[1] h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full bg-white shadow-[0_0_0_1px_rgba(0,0,0,0.28)] transition-transform duration-150 group-hover/progress:scale-150 group-focus-within/progress:scale-150"
              />
              <input
                data-reel-control="true"
                type="range"
                min={0}
                max={clipDuration}
                step={0.1}
                value={currentSec}
                onChange={onSeek}
                className="reel-range relative z-10 h-5 w-full cursor-pointer disabled:opacity-40"
                disabled={!controlsEnabled}
              />
            </div>
          ) : (
            <p className="text-[10px] font-semibold uppercase tracking-[0.08em] text-white/60">Platform-managed playback</p>
          )}

          {loadError ? <p className="mt-2 inline-flex rounded-full bg-black/76 px-3 py-1 text-xs text-white/78">{loadError}</p> : null}
        </div>
      </div>
    </section>
  );
}
