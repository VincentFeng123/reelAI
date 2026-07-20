"use client";

import { useState } from "react";

import { createStripeCheckout, createStripePortal } from "@/lib/api";
import type { BillingPlan, BillingPlanCode, BillingStatus, BillingSubscription } from "@/lib/types";

const INACTIVE_SUBSCRIPTION_STATUSES = new Set([
  "canceled",
  "cancelled",
  "expired",
  "inactive",
  "incomplete_expired",
  "refunded",
]);

export function activeBillingSubscription(status: BillingStatus | null): BillingSubscription | null {
  if (!status) {
    return null;
  }
  const manageableSubscriptions = status.subscriptions.filter((subscription) => (
    !INACTIVE_SUBSCRIPTION_STATUSES.has(subscription.status.trim().toLowerCase())
  ));
  return manageableSubscriptions.find((subscription) => subscription.plan === status.plan)
    ?? manageableSubscriptions[0]
    ?? null;
}

function formatPlanPrice(plan: BillingPlan): string {
  if (plan.monthly_price_cents <= 0) {
    return "Free";
  }
  return `$${(plan.monthly_price_cents / 100).toFixed(2)} USD/month`;
}

type BillingActionsProps = {
  status: BillingStatus | null;
  plans: BillingPlan[];
  onError?: (message: string | null) => void;
  demoMode?: boolean;
};

export function BillingActions({ status, plans, onError, demoMode = false }: BillingActionsProps) {
  const [busyAction, setBusyAction] = useState<string | null>(null);
  const activeSubscription = activeBillingSubscription(status);

  const navigateTo = (url: string) => {
    window.location.assign(url);
  };

  const openCheckout = async (plan: Exclude<BillingPlanCode, "free">) => {
    if (demoMode) {
      onError?.("Billing actions are disabled for the local demo account.");
      return;
    }
    if (busyAction) {
      return;
    }
    setBusyAction(plan);
    onError?.(null);
    try {
      navigateTo(await createStripeCheckout(plan));
    } catch (checkoutError) {
      onError?.(checkoutError instanceof Error ? checkoutError.message : "Could not open Stripe Checkout.");
      setBusyAction(null);
    }
  };

  const openPortal = async () => {
    if (demoMode) {
      onError?.("Billing actions are disabled for the local demo account.");
      return;
    }
    if (busyAction) {
      return;
    }
    setBusyAction("portal");
    onError?.(null);
    try {
      navigateTo(await createStripePortal());
    } catch (portalError) {
      onError?.(portalError instanceof Error ? portalError.message : "Could not open billing management.");
      setBusyAction(null);
    }
  };

  if (activeSubscription) {
    return (
      <div>
        <button
          type="button"
          onClick={() => void openPortal()}
          disabled={Boolean(busyAction)}
          className="inline-flex h-11 w-full items-center justify-center rounded-[14px] bg-white px-4 text-sm font-bold text-black transition hover:bg-white/90 disabled:cursor-wait disabled:opacity-60"
        >
          {busyAction === "portal" ? "Opening portal..." : "Manage subscription"}
        </button>
        <p className="mt-2 text-center text-[10px] font-medium text-white/50">U.S. Stripe subscription</p>
      </div>
    );
  }

  const paidPlans = plans.filter((plan): plan is BillingPlan & { code: Exclude<BillingPlanCode, "free"> } => plan.code !== "free");
  return (
    <div>
      <div className="grid grid-cols-2 gap-2">
        {paidPlans.map((plan) => (
          <button
            key={plan.code}
            type="button"
            onClick={() => void openCheckout(plan.code)}
            disabled={Boolean(busyAction)}
            className="rounded-[14px] bg-white px-3 py-2.5 text-left text-black transition hover:bg-white/90 disabled:cursor-wait disabled:opacity-60"
          >
            <span className="block text-sm font-bold">{busyAction === plan.code ? "Opening..." : plan.name}</span>
            <span className="mt-0.5 block text-[11px] font-medium text-black/60">
              {formatPlanPrice(plan)} · {plan.daily_limit}/day
            </span>
          </button>
        ))}
      </div>
      <p className="mt-2 text-center text-[10px] font-medium text-white/50">
        U.S. Stripe subscription · secure checkout hosted by Stripe
      </p>
    </div>
  );
}
