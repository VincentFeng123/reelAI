"use client";

import { useEffect, useState } from "react";

const STAGE_HINTS = ["Finding videos…", "Cutting clips with AI…"];
const HINT_INTERVAL_MS = 6000;

interface GenerationProgressProps {
  received: number;
  requested: number;
  variant?: "bar" | "center";
}

export function GenerationProgress({ received, requested, variant = "bar" }: GenerationProgressProps) {
  const [hintIndex, setHintIndex] = useState(0);

  useEffect(() => {
    if (received > 0) return;
    const id = setInterval(() => {
      setHintIndex((i) => (i + 1) % STAGE_HINTS.length);
    }, HINT_INTERVAL_MS);
    return () => clearInterval(id);
  }, [received]);

  const isIndeterminate = received === 0;
  const pct = requested > 0 ? Math.min(1, received / requested) : 0;

  if (variant === "center") {
    return (
      <div role="status" aria-live="polite" className="w-full">
        <div className="relative h-1 overflow-hidden rounded-full bg-white/10">
          {isIndeterminate ? (
            <div className="animate-progress-shimmer absolute inset-y-0 w-1/3 bg-gradient-to-r from-transparent via-white/60 to-transparent" />
          ) : (
            <div
              className="absolute inset-y-0 left-0 bg-white/80 transition-all duration-500 ease-out"
              style={{ width: `${pct * 100}%` }}
            />
          )}
        </div>
        <p className="mt-3 text-sm font-semibold">
          {isIndeterminate ? STAGE_HINTS[hintIndex] : `${received} of ${requested} reels ready`}
        </p>
        <p className="mt-1 text-xs text-white/72">This can take a little while on first generation.</p>
      </div>
    );
  }

  return (
    <div
      className="pointer-events-none absolute inset-x-0 top-0 z-[9998]"
      role="status"
      aria-live="polite"
    >
      <div className="relative h-1 overflow-hidden bg-white/10">
        {isIndeterminate ? (
          <div className="animate-progress-shimmer absolute inset-y-0 w-1/3 bg-gradient-to-r from-transparent via-white/60 to-transparent" />
        ) : (
          <div
            className="absolute inset-y-0 left-0 bg-white/80 transition-all duration-500 ease-out"
            style={{ width: `${pct * 100}%` }}
          />
        )}
      </div>
      <div className="flex justify-center py-1.5">
        <span className="rounded-full bg-black/56 px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.12em] text-white/80 backdrop-blur-sm">
          {isIndeterminate
            ? STAGE_HINTS[hintIndex]
            : `${received} of ${requested} reels ready`}
        </span>
      </div>
    </div>
  );
}
