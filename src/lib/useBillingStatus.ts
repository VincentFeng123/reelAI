"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import {
  ApiError,
  BILLING_STATUS_REFRESH_EVENT,
  COMMUNITY_AUTH_CHANGED_EVENT,
  fetchBillingPlans,
  fetchBillingStatus,
} from "@/lib/api";
import type { BillingPlan, BillingStatus, CommunityAccount } from "@/lib/types";

type BillingStatusState = {
  status: BillingStatus | null;
  plans: BillingPlan[];
  loading: boolean;
  error: string | null;
  refresh: () => Promise<BillingStatus | null>;
};

type ActiveBillingRefresh = {
  accountId: string;
  controller: AbortController;
  token: symbol;
  promise: Promise<BillingStatus | null>;
};

export const BILLING_ROLLOUT_UNAVAILABLE_MESSAGE =
  "Plans & usage are still being rolled out and aren't available yet. Please try again shortly.";

export function billingStatusErrorMessage(error: unknown): string {
  if (error instanceof ApiError && error.status === 404) {
    return BILLING_ROLLOUT_UNAVAILABLE_MESSAGE;
  }
  return error instanceof Error ? error.message : "Could not load plan usage.";
}

export function requestBillingStatusRefresh(): void {
  if (typeof window !== "undefined") {
    window.dispatchEvent(new Event(BILLING_STATUS_REFRESH_EVENT));
  }
}

export function useBillingStatus(account: CommunityAccount | null | undefined): BillingStatusState {
  const accountId = account?.isVerified ? account.id : "";
  const [status, setStatus] = useState<BillingStatus | null>(null);
  const [plans, setPlans] = useState<BillingPlan[]>([]);
  const [loading, setLoading] = useState(Boolean(accountId));
  const [error, setError] = useState<string | null>(null);
  const requestSequenceRef = useRef(0);
  const plansLoadedRef = useRef(false);
  const activeRefreshRef = useRef<ActiveBillingRefresh | null>(null);

  const refresh = useCallback((): Promise<BillingStatus | null> => {
    if (!accountId) {
      requestSequenceRef.current += 1;
      activeRefreshRef.current?.controller.abort();
      activeRefreshRef.current = null;
      setStatus(null);
      setLoading(false);
      setError(null);
      return Promise.resolve(null);
    }
    const activeRefresh = activeRefreshRef.current;
    if (activeRefresh?.accountId === accountId) {
      return activeRefresh.promise;
    }
    const requestSequence = requestSequenceRef.current + 1;
    requestSequenceRef.current = requestSequence;
    const controller = new AbortController();
    const token = Symbol(accountId);
    setLoading(true);
    setError(null);
    const promise = (async (): Promise<BillingStatus | null> => {
      try {
        const [nextStatus, nextPlans] = await Promise.all([
          fetchBillingStatus(controller.signal),
          plansLoadedRef.current
            ? Promise.resolve<BillingPlan[] | null>(null)
            : fetchBillingPlans(controller.signal),
        ]);
        if (requestSequence !== requestSequenceRef.current) {
          return null;
        }
        setStatus(nextStatus);
        if (nextPlans) {
          plansLoadedRef.current = true;
          setPlans(nextPlans);
        }
        return nextStatus;
      } catch (refreshError) {
        if (requestSequence === requestSequenceRef.current && !controller.signal.aborted) {
          setError(billingStatusErrorMessage(refreshError));
        }
        return null;
      } finally {
        if (activeRefreshRef.current?.token === token) {
          activeRefreshRef.current = null;
        }
        if (requestSequence === requestSequenceRef.current) {
          setLoading(false);
        }
      }
    })();
    activeRefreshRef.current = { accountId, controller, token, promise };
    return promise;
  }, [accountId]);

  useEffect(() => {
    requestSequenceRef.current += 1;
    activeRefreshRef.current?.controller.abort();
    activeRefreshRef.current = null;
    plansLoadedRef.current = false;
    setPlans([]);
    setStatus(null);
    void refresh();
    return () => {
      requestSequenceRef.current += 1;
      activeRefreshRef.current?.controller.abort();
      activeRefreshRef.current = null;
    };
  }, [accountId, refresh]);

  useEffect(() => {
    if (!accountId || typeof window === "undefined") {
      return;
    }
    const onRefresh = () => {
      void refresh();
    };
    const onVisibilityChange = () => {
      if (document.visibilityState === "visible") {
        void refresh();
      }
    };
    window.addEventListener("focus", onRefresh);
    window.addEventListener("pageshow", onRefresh);
    window.addEventListener(COMMUNITY_AUTH_CHANGED_EVENT, onRefresh);
    window.addEventListener(BILLING_STATUS_REFRESH_EVENT, onRefresh);
    document.addEventListener("visibilitychange", onVisibilityChange);
    return () => {
      window.removeEventListener("focus", onRefresh);
      window.removeEventListener("pageshow", onRefresh);
      window.removeEventListener(COMMUNITY_AUTH_CHANGED_EVENT, onRefresh);
      window.removeEventListener(BILLING_STATUS_REFRESH_EVENT, onRefresh);
      document.removeEventListener("visibilitychange", onVisibilityChange);
    };
  }, [accountId, refresh]);

  useEffect(() => {
    if (!accountId || typeof window === "undefined") {
      return;
    }
    const query = new URLSearchParams(window.location.search);
    const returnedFromCheckout = [query.get("billing"), query.get("checkout"), query.get("stripe")]
      .some((value) => value === "success" || value === "complete" || value === "return");
    if (!returnedFromCheckout) {
      return;
    }
    const intervals = [0, 1_200, 1_800, 3_000];
    let stopped = false;
    let timer: number | null = null;
    let index = 0;
    const poll = async () => {
      await refresh();
      if (stopped) {
        return;
      }
      index += 1;
      if (index < intervals.length) {
        timer = window.setTimeout(() => {
          void poll();
        }, intervals[index]);
      }
    };
    void poll();
    return () => {
      stopped = true;
      if (timer !== null) {
        window.clearTimeout(timer);
      }
    };
  }, [accountId, refresh]);

  return { status, plans, loading, error, refresh };
}
