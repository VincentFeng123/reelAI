"use client";

import { useEffect } from "react";

export function GridGlowTracker() {
  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    const root = document.documentElement;
    const FOLLOW_EASE = 0.17;
    const SNAP_EPSILON = 0.35;
    let rafId = 0;
    let targetX = window.innerWidth * 0.5;
    let targetY = window.innerHeight * 0.5;
    let currentX = targetX;
    let currentY = targetY;

    const animate = () => {
      const dx = targetX - currentX;
      const dy = targetY - currentY;
      currentX += dx * FOLLOW_EASE;
      currentY += dy * FOLLOW_EASE;
      root.style.setProperty("--mouse-x", `${currentX}px`);
      root.style.setProperty("--mouse-y", `${currentY}px`);
      if (Math.abs(dx) > SNAP_EPSILON || Math.abs(dy) > SNAP_EPSILON) {
        rafId = window.requestAnimationFrame(animate);
        return;
      }
      currentX = targetX;
      currentY = targetY;
      root.style.setProperty("--mouse-x", `${currentX}px`);
      root.style.setProperty("--mouse-y", `${currentY}px`);
      rafId = 0;
    };

    const onMove = (event: MouseEvent) => {
      targetX = event.clientX;
      targetY = event.clientY;
      root.style.setProperty("--grid-glow-opacity", "1");
      if (rafId === 0) {
        rafId = window.requestAnimationFrame(animate);
      }
    };

    const onLeave = () => {
      root.style.setProperty("--grid-glow-opacity", "0");
    };

    window.addEventListener("mousemove", onMove, { passive: true });
    window.addEventListener("mouseleave", onLeave);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseleave", onLeave);
      if (rafId !== 0) {
        window.cancelAnimationFrame(rafId);
      }
    };
  }, []);

  return null;
}
