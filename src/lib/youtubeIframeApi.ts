"use client";

const YOUTUBE_IFRAME_API_SCRIPT_ID = "studyreels-youtube-iframe-api";
const YOUTUBE_IFRAME_API_SRC = "https://www.youtube.com/iframe_api";
const DEFAULT_YOUTUBE_IFRAME_API_TIMEOUT_MS = 10_000;

type YouTubeIframeApiWindow = Window & typeof globalThis & {
  YT?: {
    Player?: unknown;
  };
  onYouTubeIframeAPIReady?: () => void;
};

let youtubeIframeApiLoadPromise: Promise<void> | null = null;

export function loadYouTubeIframeApi(timeoutMs = DEFAULT_YOUTUBE_IFRAME_API_TIMEOUT_MS): Promise<void> {
  if (typeof window === "undefined") {
    return Promise.resolve();
  }

  const globalWindow = window as YouTubeIframeApiWindow;
  if (globalWindow.YT?.Player) {
    return Promise.resolve();
  }
  if (youtubeIframeApiLoadPromise) {
    return youtubeIframeApiLoadPromise;
  }

  youtubeIframeApiLoadPromise = new Promise<void>((resolve, reject) => {
    const failedScript = document.getElementById(YOUTUBE_IFRAME_API_SCRIPT_ID) as HTMLScriptElement | null;
    if (failedScript?.dataset.loadState === "error") {
      failedScript.remove();
    }

    let settled = false;
    const settleResolve = () => {
      if (settled) {
        return;
      }
      settled = true;
      window.clearTimeout(timeoutId);
      const readyScript = document.getElementById(YOUTUBE_IFRAME_API_SCRIPT_ID) as HTMLScriptElement | null;
      if (readyScript) {
        readyScript.dataset.loadState = "ready";
      }
      resolve();
    };
    const settleReject = (error: Error) => {
      if (settled) {
        return;
      }
      settled = true;
      window.clearTimeout(timeoutId);
      const script = document.getElementById(YOUTUBE_IFRAME_API_SCRIPT_ID) as HTMLScriptElement | null;
      if (script && script.dataset.loadState !== "ready") {
        script.dataset.loadState = "error";
        script.remove();
      }
      reject(error);
    };
    const timeoutId = window.setTimeout(() => {
      youtubeIframeApiLoadPromise = null;
      settleReject(new Error("Timed out loading YouTube IFrame API."));
    }, timeoutMs);

    const currentScript = document.getElementById(YOUTUBE_IFRAME_API_SCRIPT_ID) as HTMLScriptElement | null;
    const previousReady = globalWindow.onYouTubeIframeAPIReady;
    globalWindow.onYouTubeIframeAPIReady = () => {
      if (typeof previousReady === "function") {
        previousReady();
      }
      settleResolve();
    };

    if (currentScript) {
      if (currentScript.dataset.loadState === "ready") {
        settleResolve();
      }
      return;
    }

    const script = document.createElement("script");
    script.id = YOUTUBE_IFRAME_API_SCRIPT_ID;
    script.dataset.loadState = "loading";
    script.src = YOUTUBE_IFRAME_API_SRC;
    script.async = true;
    script.onerror = () => {
      youtubeIframeApiLoadPromise = null;
      settleReject(new Error("Failed to load YouTube IFrame API."));
    };
    document.body.appendChild(script);
  }).catch((error) => {
    youtubeIframeApiLoadPromise = null;
    throw error;
  });

  return youtubeIframeApiLoadPromise;
}
