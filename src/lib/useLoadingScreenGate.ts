"use client";

import { useEffect, useRef, useState } from "react";

const BLUR_TEXT_DURATION_MS = 1500;
const BLUR_TEXT_STAGGER_MS = 200;
const LAST_LOADING_LETTER_INDEX = 5;
const FULL_LOADING_SEQUENCE_MS = BLUR_TEXT_DURATION_MS + (BLUR_TEXT_STAGGER_MS * LAST_LOADING_LETTER_INDEX);

type UseLoadingScreenGateOptions = {
  minimumVisibleMs?: number;
  completeAnimationCycle?: boolean;
};

function getNow() {
  return typeof performance === "undefined" ? Date.now() : performance.now();
}

export function useLoadingScreenGate(ready: boolean, options: UseLoadingScreenGateOptions = {}) {
  const { minimumVisibleMs = 0, completeAnimationCycle = false } = options;
  const mountedAtRef = useRef<number | null>(null);
  const [showLoadingScreen, setShowLoadingScreen] = useState(true);

  useEffect(() => {
    if (mountedAtRef.current === null) {
      mountedAtRef.current = getNow();
    }
    if (!ready) {
      setShowLoadingScreen(true);
      return;
    }
    const mountedAt = mountedAtRef.current;
    if (mountedAt === null) {
      return;
    }
    const elapsedMs = Math.max(0, getNow() - mountedAt);
    let remainingMs = Math.max(0, minimumVisibleMs - elapsedMs);
    if (completeAnimationCycle) {
      const targetElapsedMs = elapsedMs + remainingMs;
      const remainderMs = targetElapsedMs % FULL_LOADING_SEQUENCE_MS;
      remainingMs += remainderMs === 0 ? 0 : FULL_LOADING_SEQUENCE_MS - remainderMs;
    }
    const timer = window.setTimeout(() => {
      setShowLoadingScreen(false);
    }, remainingMs);
    return () => {
      window.clearTimeout(timer);
    };
  }, [completeAnimationCycle, minimumVisibleMs, ready]);

  return showLoadingScreen;
}
