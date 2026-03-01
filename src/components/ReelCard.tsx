"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { Reel } from "@/lib/types";

type Props = {
  reel: Reel;
  isActive: boolean;
  mutedPreference: boolean;
  onMutedPreferenceChange: (nextMuted: boolean) => void;
  onOpenContent?: () => void;
};

type YouTubePlayer = {
  destroy: () => void;
  pauseVideo: () => void;
  playVideo: () => void;
  seekTo: (seconds: number, allowSeekAhead: boolean) => void;
  getCurrentTime: () => number;
  mute: () => void;
  unMute: () => void;
};

const YOUTUBE_SCRIPT_ID = "studyreels-youtube-iframe-api";
const PLAYER_REVEAL_DELAY_MS = 0;
const RESUME_MASK_MS = 480;
let youtubeApiLoadPromise: Promise<void> | null = null;

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

function loadYouTubeIframeApi(): Promise<void> {
  if (typeof window === "undefined") {
    return Promise.resolve();
  }
  const readyYT = (window as any).YT;
  if (readyYT?.Player) {
    return Promise.resolve();
  }

  if (youtubeApiLoadPromise) {
    return youtubeApiLoadPromise;
  }

  youtubeApiLoadPromise = new Promise<void>((resolve, reject) => {
    const existing = document.getElementById(YOUTUBE_SCRIPT_ID) as HTMLScriptElement | null;
    const previousReady = (window as any).onYouTubeIframeAPIReady;
    (window as any).onYouTubeIframeAPIReady = () => {
      if (typeof previousReady === "function") {
        previousReady();
      }
      resolve();
    };

    if (existing) {
      return;
    }

    const script = document.createElement("script");
    script.id = YOUTUBE_SCRIPT_ID;
    script.src = "https://www.youtube.com/iframe_api";
    script.async = true;
    script.onerror = () => reject(new Error("Failed to load YouTube player API"));
    document.body.appendChild(script);
  });

  return youtubeApiLoadPromise;
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
  onOpenContent,
}: Props) {
  const hostContainerRef = useRef<HTMLDivElement | null>(null);
  const playerRef = useRef<YouTubePlayer | null>(null);
  const progressTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const revealTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const resumeMaskTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const didUserInteractRef = useRef(false);

  const [isReady, setIsReady] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(mutedPreference);
  const [isSurfaceVisible, setIsSurfaceVisible] = useState(false);
  const [isResumeMaskVisible, setIsResumeMaskVisible] = useState(false);
  const [showCaptions, setShowCaptions] = useState(false);
  const [currentSec, setCurrentSec] = useState(0);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [isTouchLikeDevice, setIsTouchLikeDevice] = useState(false);
  const [isMobilePhoneDevice, setIsMobilePhoneDevice] = useState(false);

  const videoId = useMemo(() => extractVideoId(reel.video_url), [reel.video_url]);
  const clipStart = Math.max(0, Math.floor(reel.t_start));
  const clipEnd = Math.max(clipStart + 1, Math.ceil(reel.t_end));
  const clipDuration = Math.max(1, clipEnd - clipStart);
  const captionCues = useMemo(
    () =>
      (reel.captions ?? [])
        .filter((cue) => Number.isFinite(cue.start) && Number.isFinite(cue.end) && Boolean(cue.text?.trim()))
        .sort((a, b) => a.start - b.start),
    [reel.captions],
  );

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

  const syncProgress = useCallback(() => {
    const player = playerRef.current;
    if (!player) {
      return;
    }
    const now = clamp(player.getCurrentTime(), clipStart, clipEnd);
    const rel = clamp(now - clipStart, 0, clipDuration);
    setCurrentSec(rel);
  }, [clipDuration, clipEnd, clipStart]);

  const startProgressTimer = useCallback(() => {
    stopProgressTimer();
    progressTimerRef.current = setInterval(() => {
      const player = playerRef.current;
      if (!player) {
        return;
      }
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
  }, [clipDuration, clipEnd, clipStart, isActive, stopProgressTimer]);

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
      destroyPlayerSafely();
    };
  }, [clearRevealTimer, clearResumeMaskTimer, destroyPlayerSafely, stopProgressTimer]);

  useEffect(() => {
    stopProgressTimer();
    clearRevealTimer();
    clearResumeMaskTimer();
    setCurrentSec(0);
    setIsPlaying(false);
    setIsReady(false);
    setIsMuted(mutedPreference);
    setIsSurfaceVisible(false);
    setIsResumeMaskVisible(false);
    setShowCaptions(false);
    setLoadError(null);
    didUserInteractRef.current = false;

    destroyPlayerSafely();

    if (!isActive) {
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
            end: clipEnd,
            mute: 1,
            enablejsapi: 1,
            origin: window.location.origin,
          },
          events: {
            onReady: (event: any) => {
              if (cancelled) {
                return;
              }
              // Mobile browsers commonly block autoplay with sound; start muted reliably.
              event.target.mute();
              event.target.seekTo(clipStart, true);
              event.target.playVideo();
              setIsMuted(true);
              setIsReady(true);
              setIsPlaying(true);
              setCurrentSec(0);
              setIsSurfaceVisible(true);
              scheduleSurfaceReveal(PLAYER_REVEAL_DELAY_MS);
              startProgressTimer();
            },
            onStateChange: (event: any) => {
              if (cancelled) {
                return;
              }
              const state = event.data;
              const playerState = yt.PlayerState;
              if (state === playerState.PLAYING) {
                setIsPlaying(true);
                startProgressTimer();
              } else if (state === playerState.PAUSED) {
                setIsPlaying(false);
                setIsResumeMaskVisible(false);
                stopProgressTimer();
                syncProgress();
              } else if (state === playerState.ENDED) {
                event.target.seekTo(clipStart, true);
                event.target.playVideo();
                setIsPlaying(true);
                setIsSurfaceVisible(true);
                setCurrentSec(0);
                scheduleSurfaceReveal(PLAYER_REVEAL_DELAY_MS);
                startProgressTimer();
              } else if ((state === playerState.UNSTARTED || state === playerState.CUED) && isActive) {
                // Retry autoplay for devices that initially report cued/unstarted.
                event.target.playVideo();
              }
            },
            onError: () => {
              if (cancelled) {
                return;
              }
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
      destroyPlayerSafely();
      clearHostContainer();
    };
  }, [
    clearHostContainer,
    clearRevealTimer,
    clearResumeMaskTimer,
    clipEnd,
    clipStart,
    isActive,
    scheduleSurfaceReveal,
    showResumeMask,
    startProgressTimer,
    stopProgressTimer,
    syncProgress,
    videoId,
    destroyPlayerSafely,
  ]);

  useEffect(() => {
    setIsMuted(mutedPreference);
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
  }, [isActive, isReady, mutedPreference, isTouchLikeDevice]);

  const togglePlayPause = useCallback(() => {
    const player = playerRef.current;
    if (!player || !isReady) {
      return;
    }
    didUserInteractRef.current = true;
    if (!isPlaying && !mutedPreference) {
      player.unMute();
      setIsMuted(false);
    }
    if (isPlaying) {
      player.pauseVideo();
      setIsPlaying(false);
      setIsResumeMaskVisible(false);
      stopProgressTimer();
      syncProgress();
      return;
    }
    showResumeMask(RESUME_MASK_MS);
    scheduleSurfaceReveal(PLAYER_REVEAL_DELAY_MS);
    setIsSurfaceVisible(true);
    player.playVideo();
    setIsPlaying(true);
    startProgressTimer();
  }, [isMuted, isPlaying, isReady, mutedPreference, scheduleSurfaceReveal, showResumeMask, startProgressTimer, stopProgressTimer, syncProgress]);

  const toggleMute = useCallback(() => {
    const player = playerRef.current;
    if (!player || !isReady) {
      return;
    }
    didUserInteractRef.current = true;
    const nextMuted = !isMuted;
    if (nextMuted) {
      player.mute();
      setIsMuted(true);
    } else {
      player.unMute();
      setIsMuted(false);
    }
    onMutedPreferenceChange(nextMuted);
  }, [isMuted, isReady, onMutedPreferenceChange]);

  const onSeek = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const player = playerRef.current;
      if (!player || !isReady) {
        return;
      }
      const rel = clamp(Number(event.target.value), 0, clipDuration);
      setCurrentSec(rel);
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
    [clipDuration, clipStart, isActive, isPlaying, isReady, scheduleSurfaceReveal, showResumeMask, startProgressTimer],
  );

  const stopFeedGesturePropagation = useCallback((event: React.SyntheticEvent<HTMLElement>) => {
    event.stopPropagation();
  }, []);

  const toggleCaptions = useCallback(() => {
    setShowCaptions((prev) => !prev);
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

  const hidePlayerSurface = isActive && (!isReady || !isPlaying || !isSurfaceVisible || Boolean(loadError));
  const showTransitionMask = isActive && isResumeMaskVisible;
  const canToggleFromSurface = isActive && isReady && !loadError;
  const surfaceAriaLabel = isPlaying ? "Pause clip" : "Play clip";
  const controlsEnabled = isReady && isActive;
  const activeCaptionText = useMemo(() => {
    if (!showCaptions) {
      return "";
    }
    const fallback = reel.transcript_snippet?.trim() || "No captions available for this reel.";
    if (captionCues.length === 0) {
      return fallback;
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
  }, [captionCues, clipDuration, currentSec, reel.transcript_snippet, showCaptions]);
  const controlButtonClass = (active: boolean) =>
    `grid h-9 w-9 place-items-center text-base transition ${
      active ? "text-white" : "text-white/88 hover:text-white"
    } disabled:text-white/35`;
  const controlsChromeClass = isMobilePhoneDevice
    ? "rounded-2xl border border-white/20 bg-black/70 px-3 py-2 shadow-[0_10px_26px_rgba(0,0,0,0.38)] backdrop-blur-md"
    : "px-0 py-0";

  return (
    <section className="relative h-full min-h-full w-full snap-start overflow-hidden rounded-3xl border border-white/20 bg-black/80">
      {isActive ? (
        <div className="absolute inset-0 overflow-hidden">
          <div
            ref={hostContainerRef}
            className="pointer-events-none absolute inset-x-0 -top-14 h-[calc(100%+56px)] w-full"
          />
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

      <div className="pointer-events-none absolute inset-x-0 top-0 z-[16] h-10 bg-black/95" />

      {isActive ? (
        <button
          type="button"
          aria-label={surfaceAriaLabel}
          onClick={togglePlayPause}
          disabled={!canToggleFromSurface}
          className="absolute inset-0 z-[15] cursor-pointer bg-transparent disabled:cursor-not-allowed"
        />
      ) : null}

      {isActive && isReady && !isPlaying && !loadError ? (
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
          {showCaptions && activeCaptionText ? (
            <div className="mb-2 flex justify-center px-1">
              <p className="max-w-[92%] rounded-xl bg-black/78 px-3 py-2 text-center text-[12px] font-medium leading-relaxed text-white/96">
                {activeCaptionText}
              </p>
            </div>
          ) : null}

          <div className="mb-2 flex items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <div className="inline-flex h-9 items-center rounded-full border border-white/30 bg-black/82 px-3 text-[10px] font-semibold uppercase tracking-[0.08em] text-white/92">
                {formatClock(currentSec)} / {formatClock(clipDuration)}
              </div>
              {onOpenContent ? (
                <button
                  type="button"
                  data-reel-control="true"
                  onClick={onOpenContent}
                  className="inline-flex h-8 items-center rounded-full border border-white/28 px-3 text-[10px] font-semibold uppercase tracking-[0.1em] text-white/92 transition hover:bg-white/10 lg:hidden"
                >
                  Content
                </button>
              ) : null}
            </div>
            <div className="flex items-center gap-2">
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
            </div>
          </div>

          <input
            data-reel-control="true"
            type="range"
            min={0}
            max={clipDuration}
            step={0.1}
            value={currentSec}
            onChange={onSeek}
            className="reel-range h-1.5 w-full cursor-pointer disabled:opacity-40"
            disabled={!controlsEnabled}
          />

          {loadError ? <p className="mt-2 inline-flex rounded-full bg-black/76 px-3 py-1 text-xs text-white/78">{loadError}</p> : null}
        </div>
      </div>
    </section>
  );
}
