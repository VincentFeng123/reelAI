"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";

import { BillingActions } from "@/components/BillingActions";
import { ViewportModalPortal } from "@/components/ViewportModalPortal";
import { useBillingStatus } from "@/lib/useBillingStatus";
import type { CommunityAccount } from "@/lib/types";

export type BillingGateReason = "sign_in" | "quota";
const BILLING_MODAL_FADE_MS = 340;

type BillingGateDialogProps = {
  reason: BillingGateReason;
  account: CommunityAccount | null;
  requiredSearches?: number;
  onBillingAvailable?: () => void;
  onClose: () => void;
};

function formatReset(value: string | undefined): string {
  if (!value) {
    return "at the next UTC day";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "at the next UTC day";
  }
  return new Intl.DateTimeFormat(undefined, {
    weekday: "short",
    hour: "numeric",
    minute: "2-digit",
    timeZone: "UTC",
    timeZoneName: "short",
  }).format(date);
}

export function BillingGateDialog({ reason, account, requiredSearches = 1, onBillingAvailable, onClose }: BillingGateDialogProps) {
  const router = useRouter();
  const { status, plans, loading, error: loadError, refresh } = useBillingStatus(account);
  const [actionError, setActionError] = useState<string | null>(null);
  const [isVisible, setIsVisible] = useState(false);
  const dialogRef = useRef<HTMLElement | null>(null);
  const closeButtonRef = useRef<HTMLButtonElement | null>(null);
  const returnFocusRef = useRef<HTMLElement | null>(null);
  const closeTimerRef = useRef<number | null>(null);

  const requestClose = useCallback(() => {
    if (closeTimerRef.current !== null) {
      return;
    }
    setIsVisible(false);
    const closeDelay = window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : BILLING_MODAL_FADE_MS;
    closeTimerRef.current = window.setTimeout(onClose, closeDelay);
  }, [onClose]);

  const closeAfterBillingRefresh = useCallback(() => {
    if (!onBillingAvailable || closeTimerRef.current !== null) {
      return;
    }
    setIsVisible(false);
    const closeDelay = window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : BILLING_MODAL_FADE_MS;
    closeTimerRef.current = window.setTimeout(onBillingAvailable, closeDelay);
  }, [onBillingAvailable]);

  useEffect(() => {
    const frame = window.requestAnimationFrame(() => setIsVisible(true));
    return () => {
      window.cancelAnimationFrame(frame);
      if (closeTimerRef.current !== null) {
        window.clearTimeout(closeTimerRef.current);
      }
    };
  }, []);

  useEffect(() => {
    returnFocusRef.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    return () => returnFocusRef.current?.focus();
  }, []);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) {
      return;
    }
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        requestClose();
        return;
      }
      if (event.key !== "Tab") {
        return;
      }
      const focusable = Array.from(
        dialog.querySelectorAll<HTMLElement>(
          "button:not(:disabled), input:not(:disabled), [href], [tabindex]:not([tabindex='-1'])",
        ),
      ).filter((element) => element.offsetParent !== null);
      if (focusable.length === 0) {
        event.preventDefault();
        return;
      }
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (!focusable.includes(document.activeElement as HTMLElement)) {
        event.preventDefault();
        (event.shiftKey ? last : first).focus();
      } else if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    dialog.addEventListener("keydown", onKeyDown);
    return () => dialog.removeEventListener("keydown", onKeyDown);
  }, [requestClose]);

  useEffect(() => {
    const frame = window.requestAnimationFrame(() => closeButtonRef.current?.focus());
    return () => window.cancelAnimationFrame(frame);
  }, []);

  useEffect(() => {
    if (
      reason === "quota"
      && status
      && status.remaining_searches >= requiredSearches
    ) {
      closeAfterBillingRefresh();
    }
  }, [closeAfterBillingRefresh, reason, requiredSearches, status]);

  return (
    <ViewportModalPortal>
      <div
        className={`fixed inset-0 z-[2147483647] grid place-items-center bg-black/72 px-5 py-8 transition-opacity duration-300 motion-reduce:transition-none ${
          isVisible ? "opacity-100" : "opacity-0"
        }`}
        role="presentation"
        onMouseDown={requestClose}
      >
        <section
          ref={dialogRef}
          role="dialog"
          aria-modal="true"
          aria-labelledby="billing-gate-title"
          className="relative w-full max-w-[430px] overflow-hidden rounded-[14px] bg-[#202020] p-6 text-white sm:p-7"
          onMouseDown={(event) => event.stopPropagation()}
        >
          <div className="pointer-events-none absolute -right-20 -top-24 h-52 w-52 rounded-full bg-[#7d5cff]/20 blur-3xl" />
          <button
            ref={closeButtonRef}
            type="button"
            onClick={requestClose}
            aria-label="Close"
            className="absolute right-4 top-4 z-10 grid h-9 w-9 place-items-center rounded-full bg-white/[0.06] text-white/70 transition-colors hover:bg-white/[0.07] hover:text-white"
          >
            <i className="fa-solid fa-xmark text-xs" aria-hidden="true" />
          </button>

          <div className="relative">
            <div className="grid h-11 w-11 place-items-center rounded-2xl bg-white/[0.08] text-white">
              <i className={`fa-solid ${reason === "sign_in" ? "fa-user-lock" : "fa-bolt"}`} aria-hidden="true" />
            </div>
            <p className="mt-5 text-[10px] font-semibold uppercase tracking-[0.16em] text-white/55">Plan &amp; Usage</p>
            <h2 id="billing-gate-title" className="mt-2 text-3xl font-semibold tracking-[-0.045em]">
              {reason === "sign_in" ? "Sign in to start searching" : "Daily searches used"}
            </h2>
            <p className="mt-3 text-sm leading-6 text-white/68">
              {reason === "sign_in"
                ? "New searches use hosted providers, so they require a verified ReelAI account. Existing and cached reels stay available."
                : requiredSearches > 1
                  ? `This submission needs ${requiredSearches} searches. Your plan does not have enough remaining today.`
                  : `Your searches reset ${formatReset(status?.reset_at)}. Upgrade for more daily searches, or continue with saved reels.`}
            </p>

            {reason === "sign_in" ? (
              <button
                type="button"
                onClick={() => router.push("/account?return_tab=search")}
                className="mt-6 inline-flex h-12 w-full items-center justify-center rounded-[15px] bg-white px-5 text-sm font-bold text-black transition hover:bg-white/90"
              >
                Sign in or create account
              </button>
            ) : (
              <div className="mt-6">
                {loading && !status ? (
                  <div className="h-11 animate-pulse rounded-[14px] bg-white/10" />
                ) : status ? (
                  <BillingActions status={status} plans={plans} onError={setActionError} />
                ) : null}
              </div>
            )}

            {loadError ? (
              <div className="mt-4 flex items-start justify-between gap-3" aria-live="polite">
                <p className="text-xs leading-5 text-[#ffb4b4]">{loadError}</p>
                <button
                  type="button"
                  onClick={() => void refresh()}
                  disabled={loading}
                  className="shrink-0 rounded-full bg-white/[0.08] px-3 py-1.5 text-[11px] font-semibold text-white transition-colors hover:bg-white/[0.07] disabled:cursor-wait disabled:opacity-50"
                >
                  {loading ? "Trying…" : "Try again"}
                </button>
              </div>
            ) : null}
            {actionError ? <p className="mt-3 text-xs leading-5 text-[#ffb4b4]" aria-live="polite">{actionError}</p> : null}
          </div>
        </section>
      </div>
    </ViewportModalPortal>
  );
}
