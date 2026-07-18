# ReelAI Stripe billing setup

This release uses one Stripe subscription catalog for `studyreels.app` and the
U.S.-only iOS app. It does not use StoreKit, App Store subscription products,
or Apple server notifications.

The iOS custom Plan & Usage screen calls the backend and opens Stripe-hosted
Checkout directly. Stripe sends the verified webhook to the FastAPI backend,
and only that webhook grants the entitlement. A Checkout success URL never
grants access by itself.

## Product catalog

Create these two recurring products in both Stripe test mode and Stripe live
mode. Test and live Price IDs are different.

| Field | ReelAI Plus | ReelAI Pro |
| --- | --- | --- |
| Product name | ReelAI Plus | ReelAI Pro |
| Description | 15 AI-powered searches per day across ReelAI on web and iPhone. | 50 AI-powered searches per day across ReelAI on web and iPhone. |
| Price | USD 4.99 | USD 19.99 |
| Billing period | Monthly | Monthly |
| Pricing model | Standard flat-rate recurring price | Standard flat-rate recurring price |
| Trial | None | None |
| Annual option | None | None |
| Usage-based billing | No | No |
| Internal lookup key (optional) | `reelai_plus_monthly_v1` | `reelai_pro_monthly_v1` |
| Daily search allowance | 15 | 50 |

The built-in Free plan is not a Stripe product:

- Name: Free
- Price: USD 0
- Allowance: 5 searches per UTC day

Use `REELAI` as the Stripe statement descriptor if it is available on the
account. Select the Stripe Tax category matching consumer software as a service
in the jurisdictions where ReelAI is registered; confirm the tax selection
with the business's tax adviser.

There are no trials, coupons, credit packs, lifetime purchases, annual prices,
or unlimited plan in this release.

Checkout v1 explicitly enables Stripe's `card` payment-method family. This
supports cards and card-backed wallets/Link without granting an entitlement
from an asynchronous ACH payment. Adding asynchronous payment methods later
requires handling their success and failure webhook lifecycle first.

## Stripe Dashboard configuration

1. Complete Stripe account activation, identity verification, business
   details, tax information, payout bank account, customer support details, and
   the public statement descriptor.
2. In test mode, create ReelAI Plus and ReelAI Pro with the values above.
3. Copy the test recurring Price IDs (they begin with `price_`).
4. Repeat the product and price setup in live mode and copy the live Price IDs.
5. Configure the Customer Portal:
   - allow cancellation;
   - allow switching between the Plus and Pro monthly prices;
   - apply plan switches immediately with proration;
   - end cancellations at the end of the current billing period;
   - show the business contact, privacy policy, and terms.
6. Create a webhook endpoint:
   - URL: `https://reelai-production.up.railway.app/api/billing/stripe/webhook`
   - events:
     - `checkout.session.completed`
     - `customer.subscription.created`
     - `customer.subscription.updated`
     - `customer.subscription.deleted`
     - `charge.refunded`
7. Copy the endpoint signing secret (it begins with `whsec_`).

Use the Railway URL above for launch. As of this implementation,
`https://studyreels.app/api/billing/*` returns Vercel 404 and must not be used
as the Stripe webhook target. If a same-origin proxy is deliberately deployed
and verified later, migrate the webhook endpoint rather than leaving two live
endpoints aimed at the same backend/database.

Never paste secret keys or webhook secrets into source code, client code,
screenshots, GitHub issues, or chat. Put them directly into Railway's encrypted
environment-variable settings.

## Railway variables

Production:

```dotenv
BILLING_ENFORCEMENT_ENABLED=0
BILLING_ENTITLEMENT_ENVIRONMENT=Production
BILLING_WEB_ORIGIN=https://studyreels.app
STRIPE_SECRET_KEY=sk_live_...
STRIPE_WEBHOOK_SECRET=whsec_...
STRIPE_PLUS_PRICE_ID=price_...
STRIPE_PRO_PRICE_ID=price_...
```

Stripe test mode on a staging deployment and staging database:

```dotenv
BILLING_ENFORCEMENT_ENABLED=0
BILLING_ENTITLEMENT_ENVIRONMENT=Sandbox
BILLING_WEB_ORIGIN=https://<staging-host>
STRIPE_SECRET_KEY=sk_test_...
STRIPE_WEBHOOK_SECRET=whsec_...
STRIPE_PLUS_PRICE_ID=price_...
STRIPE_PRO_PRICE_ID=price_...
```

The secret-key mode and `BILLING_ENTITLEMENT_ENVIRONMENT` must match. Test
customers and subscriptions are stored as Sandbox rows and cannot grant a
Production entitlement. Do not reuse a production database for destructive
Stripe test scenarios.

`BILLING_WEB_ORIGIN` must be an HTTPS origin without a path, query,
credentials, or fragment. Plain HTTP is accepted only for localhost
development.

## Backend behavior and public endpoints

- `GET /api/billing/plans` is public.
- `GET /api/billing/status` requires a verified ReelAI account.
- `POST /api/billing/stripe/checkout` accepts only `plus` or `pro`;
  clients cannot submit arbitrary Stripe Price IDs.
- `POST /api/billing/stripe/portal` creates a Stripe Customer Portal session.
- `POST /api/billing/stripe/webhook` verifies the raw body and
  `Stripe-Signature`, deduplicates provider events, and materializes the
  subscription.
- Account deletion expires open Checkout sessions and cancels all nonterminal
  Stripe subscriptions before deleting the ReelAI account.

Checkout binds the Stripe Customer and Subscription metadata to the ReelAI
account UUID. Open same-plan Checkout sessions are reused; choosing another
plan expires stale open sessions. Existing nonterminal subscriptions must be
managed through the Customer Portal.

The effective plan comes only from active, unexpired, webhook-verified Stripe
subscriptions in the configured environment. A stale, duplicate, out-of-order,
or mismatched-account event cannot transfer an entitlement.

## Search quota contract

- Free: 5 searches per UTC day.
- Plus: 15 searches per UTC day.
- Pro: 50 searches per UTC day.
- One initial material/topic submission or URL/search ingest consumes one
  search.
- Reopening history, cache hits, attaching to an active job, transport retries
  with the same idempotency key, continuations, and reuse of the same material
  operation do not consume another search.
- Capacity is reserved transactionally before provider work.
- A usable result consumes the reservation.
- Failed, cancelled, expired, or empty work refunds it.
- Existing per-minute, queue, and provider-cost safeguards still apply.

When exhausted, the backend returns HTTP 429, a `Retry-After` header, and a
typed body whose detail includes:

```json
{
  "code": "daily_search_limit_reached",
  "plan": "free",
  "limit": 5,
  "used": 5,
  "remaining": 0,
  "reset_at": "2026-07-19T00:00:00+00:00"
}
```

New provider-backed work requires a signed-in, verified ReelAI account.
Anonymous users may still read already-created or cached content.
This account requirement is independent of `BILLING_ENFORCEMENT_ENABLED`;
that flag disables quota rejection only. Confirm the deployed web and iOS
clients support verified sign-in before deploying this backend change.

Stripe currently supports scheduling Portal downgrades at period end only
between Prices on the same Product, while the Portal does not support two
Prices with the same monthly interval on one Product. Because Plus and Pro are
separate monthly Products, v1 uses immediate prorated plan switches. Exact
end-of-period tier downgrades would require a later custom Subscription
Schedule flow rather than a Dashboard-only Portal setting. See Stripe's
[Portal configuration limits](https://docs.stripe.com/customer-management/configure-portal).

## U.S.-only App Store setup

Do not create an App Store subscription group or any StoreKit products for this
Stripe-only build.

In App Store Connect:

1. Open **Pricing and Availability → App Availability**.
2. Choose **Specific Countries or Regions** and select only **United States**.
3. Supply the app version's description, screenshots, age rating, support URL,
   privacy policy URL, and the applicable terms/EULA.
4. In App Review notes, explain:
   - ReelAI is distributed only in the United States storefront;
   - Plus and Pro unlock daily AI-search allowances;
   - purchase and subscription management use Stripe on
     `https://studyreels.app`;
   - users sign in to the same ReelAI account on iOS and web;
   - the backend grants access only after a verified Stripe webhook.
5. Provide a working review account and precise steps to open **Plan & Usage**,
   start Checkout, return to the app, and manage/cancel through the Portal.
6. Ensure `https://studyreels.app/privacy`, the terms URL, and the support URL
   are live before submission.

As verified on 2026-07-18, `https://studyreels.app/privacy`,
`https://studyreels.app/terms`, and `https://studyreels.app/support` each return
404. These are launch blockers for App Review and Customer Portal setup. Create
real policy/support pages and real contact details before submission; this
guide intentionally does not invent legal language or a support identity.

Apple's current App Review Guidelines say the prohibition on buttons or links
to external purchasing mechanisms does not apply in the United States
storefront. App Review remains the final authority, and the rules can change:
review the current
[App Review Guidelines](https://developer.apple.com/app-store/review/guidelines/)
before every submission. Apple documents storefront selection under
[Manage availability for your app](https://developer.apple.com/help/app-store-connect/manage-your-apps-availability/manage-availability-for-your-app-on-the-app-store).

Restricting App Store availability does not geofence an already-installed app.
The server and product should still avoid presenting this Stripe-only iOS
purchase flow where it would violate local platform or payment rules.

## Rollout sequence

1. Merge and deploy the database migrations and billing code with
   `BILLING_ENFORCEMENT_ENABLED=0`.
2. Deploy the Vercel web app and Railway backend; first verify
   `https://reelai-production.up.railway.app/api/billing/plans`. Then configure
   and verify the intended web API base; do not treat the current
   `studyreels.app/api/billing/*` 404 as a billing response.
3. On a staging deployment/database, use Stripe test mode to verify:
   - Plus and Pro Checkout;
   - webhook signature rejection;
   - webhook delivery and entitlement polling;
   - same-plan Checkout reuse and cross-plan replacement;
   - Portal upgrade, downgrade, and cancellation;
   - full refunds;
   - account deletion with an open Checkout and active subscription.
4. Repeat a small smoke test with live-mode products and a real low-risk
   purchase/refund while enforcement remains off.
5. Submit the U.S.-only iOS build. Do not configure App Store subscriptions.
6. After approval, test web-to-iOS and iOS-to-web entitlement refresh using the
   same ReelAI account.
7. Set `BILLING_ENFORCEMENT_ENABLED=1` on Railway.
8. Monitor webhook failures, Checkout errors, quota refunds, successful-search
   cost by plan, and support requests. Roll back enforcement to `0` if quota
   settlement is unhealthy; subscriptions remain recorded.

Each quota reservation snapshots `plan_code`. Join
`search_quota_reservations` through `search_quota_reservation_jobs` to
`generation_provider_usage` for historical successful-search cost reporting,
including searches that fan out into multiple generation jobs and customers
who later change plans.

## Payment flow

Web and iOS use the same path:

```text
Customer → Stripe-hosted Checkout → ReelAI's Stripe balance → ReelAI's bank
                    ↓
             verified webhook
                    ↓
       Railway/Postgres entitlement
```

Stripe is the payment processor, not a marketplace recipient. Stripe deducts
its processing fees and any applicable adjustments before paying the available
balance to the ReelAI business's configured bank account.

## Final launch checklist

- [ ] Stripe business account activated
- [ ] Payout bank account and tax/business details complete
- [ ] Test Plus and Pro products/prices created
- [ ] Live Plus and Pro products/prices created
- [ ] Customer Portal configured
- [ ] Test and live webhook endpoints configured
- [ ] Railway secrets entered directly
- [ ] Vercel `RAILWAY_BACKEND_ORIGIN` points to the live backend
- [ ] Railway billing endpoints return JSON (not 404)
- [ ] Web app API base points at the deployed Railway backend
- [ ] Privacy policy, terms/EULA, and support pages live
- [ ] App Store availability restricted to United States
- [ ] App Review notes and review account prepared
- [ ] Stripe test-mode end-to-end matrix passed
- [ ] Live purchase/refund smoke test passed
- [ ] iOS/web cross-platform entitlement refresh passed
- [ ] iOS version approved
- [ ] Enforcement enabled only after all prior checks pass
