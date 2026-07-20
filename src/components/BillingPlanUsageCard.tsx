"use client";

import { useState } from "react";

import { activeBillingSubscription, BillingActions } from "@/components/BillingActions";
import { useBillingStatus } from "@/lib/useBillingStatus";
import type { CommunityAccount } from "@/lib/types";

function formatResetTime(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "at the next UTC reset";
  }
  return new Intl.DateTimeFormat(undefined, {
    hour: "numeric",
    minute: "2-digit",
    timeZone: "UTC",
    timeZoneName: "short",
  }).format(date);
}

export function BillingPlanUsageCard({ account }: { account: CommunityAccount }) {
  const { status, plans, loading, error: loadError, refresh } = useBillingStatus(account);
  const [actionError, setActionError] = useState<string | null>(null);
  const subscription = activeBillingSubscription(status);
  const percentUsed = !status || status.daily_limit <= 0
    ? 0
    : Math.min(100, Math.max(0, (status.used_searches / status.daily_limit) * 100));

  return (
    <section className="mt-4 rounded-[22px] bg-white/[0.045] px-4 py-4" aria-labelledby="plan-usage-title">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p id="plan-usage-title" className="text-[10px] font-semibold uppercase tracking-[0.14em] text-white/65">Plan &amp; Usage</p>
          <p className="mt-1.5 text-xl font-semibold tracking-[-0.03em] text-white">
            {status ? `${status.plan.charAt(0).toUpperCase()}${status.plan.slice(1)}` : loading ? "Loading..." : "Unavailable"}
          </p>
        </div>
        {status ? (
          <div className="rounded-full bg-white/[0.08] px-3 py-1.5 text-[11px] font-semibold text-white">
            {status.remaining_searches} left
          </div>
        ) : null}
      </div>

      {status ? (
        <>
          <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-white/12" aria-hidden="true">
            <div className="h-full rounded-full bg-white" style={{ width: `${percentUsed}%` }} />
          </div>
          <p className="mt-2 text-xs leading-5 text-white/70">
            {status.used_searches} of {status.daily_limit} searches used · resets {formatResetTime(status.reset_at)}
          </p>
          {subscription ? (
            <p className="mt-1 text-[11px] leading-4 text-white/55">
              Managed securely with Stripe
              {subscription.cancel_at_period_end && subscription.current_period_end
                ? ` · ends ${new Date(subscription.current_period_end).toLocaleDateString()}`
                : ""}
            </p>
          ) : null}
          <div className="mt-3">
            <BillingActions status={status} plans={plans} onError={setActionError} />
          </div>
        </>
      ) : null}

      {loadError || actionError ? (
        <div className="mt-3 flex items-center justify-between gap-3" aria-live="polite">
          <p className="text-xs leading-5 text-[#ffb4b4]">{actionError || loadError}</p>
          {loadError ? (
            <button
              type="button"
              onClick={() => void refresh()}
              disabled={loading}
              className="shrink-0 rounded-full bg-white/[0.08] px-3 py-1.5 text-[11px] font-semibold text-white transition-colors hover:bg-white/[0.07] disabled:cursor-wait disabled:opacity-50"
            >
              {loading ? "Trying…" : "Try again"}
            </button>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}
