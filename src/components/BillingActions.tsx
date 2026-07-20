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

function formatPlanMonthlyAmount(plan: BillingPlan): string {
  return `$${(plan.monthly_price_cents / 100).toFixed(2)}`;
}

function planBenefits(plan: BillingPlan): string[] {
  return [
    `${plan.daily_limit} searches every day`,
    plan.code === "pro" ? "Highest daily search allowance" : "More daily searches than Free",
    "Search history and custom sets",
  ];
}

type BillingActionsProps = {
  status: BillingStatus | null;
  plans: BillingPlan[];
  onError?: (message: string | null) => void;
  demoMode?: boolean;
  presentation?: "compact" | "pricing";
};

export function BillingActions({ status, plans, onError, demoMode = false, presentation = "compact" }: BillingActionsProps) {
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

  const paidPlans = plans.filter((plan): plan is BillingPlan & { code: Exclude<BillingPlanCode, "free"> } => plan.code !== "free");

  if (presentation === "pricing") {
    return (
      <div data-billing-pricing="true">
        <div className="mb-3 flex items-end justify-between gap-4 px-1">
          <div>
            <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-white/45">Available plans</p>
            <p className="mt-1 text-sm text-white/58">Choose the daily allowance that fits your study routine.</p>
          </div>
          <p className="hidden shrink-0 text-[11px] text-white/38 sm:block">Monthly · USD</p>
        </div>

        <div data-billing-pricing-grid className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          {paidPlans.map((plan) => {
            const isCurrentPlan = status?.plan === plan.code;
            const opensPortal = Boolean(activeSubscription);
            const isOpening = opensPortal ? busyAction === "portal" : busyAction === plan.code;
            const actionLabel = isOpening
              ? "Opening Stripe…"
              : opensPortal
                ? isCurrentPlan
                  ? "Manage current plan"
                  : `Change to ${plan.name}`
                : `Purchase ${plan.name}`;

            return (
              <article
                key={plan.code}
                data-billing-plan-card={plan.code}
                className={`flex min-h-[21rem] flex-col rounded-[20px] p-5 ${
                  plan.code === "pro" ? "bg-white/[0.085]" : "bg-white/[0.05]"
                }`}
              >
                <div className="flex min-h-6 items-center justify-between gap-3">
                  <h3 className="text-lg font-semibold tracking-[-0.025em] text-white">{plan.name}</h3>
                  {isCurrentPlan ? (
                    <span className="rounded-full bg-white px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.08em] text-black">
                      Current
                    </span>
                  ) : plan.code === "pro" ? (
                    <span className="text-[10px] font-semibold uppercase tracking-[0.1em] text-white/48">Most access</span>
                  ) : null}
                </div>

                <div className="mt-5 flex items-end gap-2">
                  <p className="text-[2.5rem] font-semibold leading-none tracking-[-0.055em] text-white">
                    {formatPlanMonthlyAmount(plan)}
                  </p>
                  <p className="pb-0.5 text-xs leading-4 text-white/45">USD<br />per month</p>
                </div>

                <ul className="mt-7 space-y-3" aria-label={`${plan.name} plan benefits`}>
                  {planBenefits(plan).map((benefit) => (
                    <li key={benefit} className="flex items-start gap-2.5 text-sm leading-5 text-white/72">
                      <span className="mt-0.5 grid h-5 w-5 shrink-0 place-items-center rounded-full bg-white/[0.09] text-white/85" aria-hidden="true">
                        <svg viewBox="0 0 16 16" className="h-3 w-3 fill-none stroke-current" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                          <path d="m3.25 8.25 3 3 6.5-7" />
                        </svg>
                      </span>
                      <span>{benefit}</span>
                    </li>
                  ))}
                </ul>

                <div className="mt-auto pt-7">
                  <button
                    type="button"
                    data-stripe-plan-button={plan.code}
                    onClick={() => {
                      if (activeSubscription) {
                        void openPortal();
                      } else {
                        void openCheckout(plan.code);
                      }
                    }}
                    disabled={Boolean(busyAction)}
                    className="inline-flex h-11 w-full items-center justify-center rounded-[13px] bg-white px-4 text-sm font-bold text-black transition-colors hover:bg-white/88 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white disabled:cursor-wait disabled:opacity-55"
                  >
                    {actionLabel}
                  </button>
                </div>
              </article>
            );
          })}
        </div>
        <p className="mt-3 text-center text-[10px] font-medium text-white/42">
          Secure checkout and subscription management hosted by Stripe
        </p>
      </div>
    );
  }

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
