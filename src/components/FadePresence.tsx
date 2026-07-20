"use client";

import { type ReactNode, useEffect, useLayoutEffect, useRef, useState } from "react";

const DEFAULT_FADE_EXIT_MS = 340;

type FadePresenceProps = {
  show: boolean;
  children: (visible: boolean) => ReactNode;
  exitMs?: number;
};

export function FadePresence({ show, children, exitMs = DEFAULT_FADE_EXIT_MS }: FadePresenceProps) {
  const [isRendered, setIsRendered] = useState(show);
  const [isVisible, setIsVisible] = useState(false);
  const renderedChildrenRef = useRef(children);

  useLayoutEffect(() => {
    if (show) {
      renderedChildrenRef.current = children;
    }
  }, [children, show]);

  useEffect(() => {
    let frame = 0;
    let revealFrame = 0;
    let closeTimer = 0;
    if (show) {
      setIsRendered(true);
      frame = window.requestAnimationFrame(() => {
        revealFrame = window.requestAnimationFrame(() => setIsVisible(true));
      });
    } else {
      setIsVisible(false);
      const closeDelay = window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : exitMs;
      closeTimer = window.setTimeout(() => setIsRendered(false), closeDelay);
    }
    return () => {
      window.cancelAnimationFrame(frame);
      window.cancelAnimationFrame(revealFrame);
      window.clearTimeout(closeTimer);
    };
  }, [exitMs, show]);

  return isRendered ? (show ? children : renderedChildrenRef.current)(isVisible) : null;
}
