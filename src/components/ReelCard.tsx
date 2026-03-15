"use client";

import { type CSSProperties, useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { Reel } from "@/lib/types";
import { loadYouTubeIframeApi } from "@/lib/youtubeIframeApi";

type Props = {
  reel: Reel;
  isActive: boolean;
  mutedPreference: boolean;
  onMutedPreferenceChange: (nextMuted: boolean) => void;
  captionsEnabled: boolean;
  onCaptionsEnabledChange: (nextEnabled: boolean) => void;
  onOpenContent?: () => void;
};

type YouTubePlayer = {
  destroy: () => void;
  pauseVideo: () => void;
  playVideo: () => void;
  seekTo: (seconds: number, allowSeekAhead: boolean) => void;
  getCurrentTime: () => number;
  getDuration?: () => number;
  mute: () => void;
  unMute: () => void;
};

type VideoProvider = "youtube" | "external";

const PLAYER_REVEAL_DELAY_MS = 0;
const RESUME_MASK_MS = 480;
const AUTOPLAY_RETRY_DELAY_MS = 320;
const AUTOPLAY_MAX_RETRIES = 5;

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

export function ReelCard({
  reel,
  isActive,
  mutedPreference,
  onMutedPreferenceChange,
  captionsEnabled,
  onCaptionsEnabledChange,
  onOpenContent,
}: Props) {
  const hostContainerRef = useRef<HTMLDivElement | null>(null);
  const playerRef = useRef<YouTubePlayer | null>(null);
  const progressTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const revealTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const resumeMaskTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const autoplayRetryTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const autoplayRetryCountRef = useRef(0);
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
  const showCaptions = captionsEnabled;
  const captionCues = useMemo(
    () =>
      (reel.captions ?? [])
        .filter((cue) => Number.isFinite(cue.start) && Number.isFinite(cue.end) && Boolean(cue.text?.trim()))
        .sort((a, b) => a.start - b.start),
    [reel.captions],
  );

  useEffect(() => {
    isMutedRef.current = isMuted;
  }, [isMuted]);

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
            autoplay: 1,
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
              syncDetectedDurationFromPlayer(event.target as YouTubePlayer);
              const tryAutoplay = () => {
                if (cancelled || !isActive || didUserInteractRef.current) {
                  return;
                }
                event.target.mute();
                setIsMuted(true);
                event.target.seekTo(clipStart, true);
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
              // Mobile browsers commonly block autoplay with sound; always start muted and retry.
              autoplayRetryCountRef.current = 0;
              tryAutoplay();
              setIsMuted(true);
              setIsReady(true);
              setIsPlaying(false);
              setCurrentSec(0);
              setIsSurfaceVisible(true);
              scheduleSurfaceReveal(PLAYER_REVEAL_DELAY_MS);
              queueAutoplayRetry();
            },
            onStateChange: (event: any) => {
              if (cancelled) {
                return;
              }
              syncDetectedDurationFromPlayer(event.target as YouTubePlayer);
              const state = event.data;
              const playerState = yt.PlayerState;
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
                if (isActive && !manualPauseRequestedRef.current && autoplayRetryCountRef.current < AUTOPLAY_MAX_RETRIES) {
                  clearAutoplayRetryTimer();
                  autoplayRetryTimerRef.current = setTimeout(() => {
                    autoplayRetryTimerRef.current = null;
                    if (cancelled || !isActive || manualPauseRequestedRef.current) {
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
                manualPauseRequestedRef.current = false;
                event.target.seekTo(clipStart, true);
                event.target.playVideo();
                setIsPlaying(true);
                setIsSurfaceVisible(true);
                setCurrentSec(0);
                scheduleSurfaceReveal(PLAYER_REVEAL_DELAY_MS);
                startProgressTimer();
              } else if ((state === playerState.UNSTARTED || state === playerState.CUED) && isActive) {
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

  const toggleCaptions = useCallback(() => {
    onCaptionsEnabledChange(!captionsEnabled);
  }, [captionsEnabled, onCaptionsEnabledChange]);

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
      const isCaptions = event.code === "KeyC" || event.key.toLowerCase() === "c";
      if (!isSpace && !isMute && !isCaptions) {
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
      toggleCaptions();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [isActive, toggleCaptions, toggleMute, togglePlayPause]);

  const hidePlayerSurface = isActive && isYouTubeVideo && (!isReady || !isPlaying || !isSurfaceVisible || Boolean(loadError));
  const showTransitionMask = isActive && isYouTubeVideo && isResumeMaskVisible;
  const canToggleFromSurface = isYouTubeVideo && isActive && isReady && !loadError;
  const surfaceAriaLabel = isPlaying ? "Pause clip" : "Play clip";
  const controlsEnabled = isReady && isActive && isYouTubeVideo;
  const activeCaptionText = useMemo(() => {
    if (!showCaptions) {
      return "";
    }
    if (captionCues.length === 0) {
      return isMobilePhoneDevice ? "" : "No caption available.";
    }

    const now = clamp(currentSec, 0, clipDuration);
    const active = captionCues.find((cue) => now >= cue.start - 0.05 && now <= cue.end + 0.04);
    if (active) {
      return active.text;
    }
    for (let i = captionCues.length - 1; i >= 0; i -= 1) {
      const cue = captionCues[i];
      if (now > cue.end && now - cue.end <= 0.28) {
        return cue.text;
      }
    }
    return "";
  }, [captionCues, clipDuration, currentSec, isMobilePhoneDevice, showCaptions]);
  const captionClass = isMobilePhoneDevice
    ? "max-w-[92%] px-1 text-center text-[12px] font-medium leading-relaxed text-white/96 [text-shadow:0_1px_3px_rgba(0,0,0,0.9)]"
    : "max-w-[92%] rounded-xl border border-white/16 bg-black/72 px-3 py-2 text-center text-[12px] font-medium leading-relaxed text-white/96 backdrop-blur-sm";
  const controlButtonClass = (active: boolean) =>
    `grid h-9 w-9 place-items-center text-base transition ${
      active ? "text-white" : "text-white/88 hover:text-white"
    } disabled:text-white/35`;
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
          {isYouTubeVideo && showCaptions && activeCaptionText ? (
            <div className="mb-2 flex justify-center px-1">
              <p className={captionClass}>
                {activeCaptionText}
              </p>
            </div>
          ) : null}

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
                <>
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
                    onClick={toggleCaptions}
                    className={controlButtonClass(showCaptions)}
                    disabled={!controlsEnabled}
                    aria-label={showCaptions ? "Hide captions" : "Show captions"}
                    title={showCaptions ? "Hide captions" : "Show captions"}
                  >
                    <i className="fa-regular fa-closed-captioning" aria-hidden="true" />
                  </button>
                </>
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
            <div className="relative w-full">
              <div
                aria-hidden="true"
                className="pointer-events-none absolute inset-x-0 top-1/2 h-1.5 -translate-y-1/2 overflow-hidden rounded-full bg-white/32"
              >
                <div className="h-full rounded-full bg-white" style={reelProgressStyle} />
              </div>
              <input
                data-reel-control="true"
                type="range"
                min={0}
                max={clipDuration}
                step={0.1}
                value={currentSec}
                onChange={onSeek}
                className="reel-range relative z-10 h-1.5 w-full cursor-pointer disabled:opacity-40"
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
