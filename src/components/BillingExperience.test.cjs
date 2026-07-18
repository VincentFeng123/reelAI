const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const uploadSource = fs.readFileSync(path.join(__dirname, "UploadPanel.tsx"), "utf8");
const accountSource = fs.readFileSync(path.join(__dirname, "CommunityAccountScreen.tsx"), "utf8");
const actionsSource = fs.readFileSync(path.join(__dirname, "BillingActions.tsx"), "utf8");
const gateSource = fs.readFileSync(path.join(__dirname, "BillingGateDialog.tsx"), "utf8");
const cardSource = fs.readFileSync(path.join(__dirname, "BillingPlanUsageCard.tsx"), "utf8");
const feedSource = fs.readFileSync(path.join(__dirname, "../app/feed/page.tsx"), "utf8");
const apiSource = fs.readFileSync(path.join(__dirname, "../lib/api.ts"), "utf8");
const hookSource = fs.readFileSync(path.join(__dirname, "../lib/useBillingStatus.ts"), "utf8");

test("new searches require a verified account and preflight one search", () => {
  assert.match(uploadSource, /if \(!verifiedAccount\?\.isVerified\)/);
  assert.match(uploadSource, /billingStatus\.remaining_searches < searchCost/);
  assert.match(uploadSource, /const searchCost = 1/);
  assert.match(uploadSource, /<span>Uses 1 search<\/span>/);
  assert.match(uploadSource, /isDailySearchLimitError\(e\)/);
  assert.match(uploadSource, /reason: "quota"/);
  assert.match(uploadSource, /requestBillingStatusRefresh\(\)/);
  assert.doesNotMatch(uploadSource, /readCommunityAuthSession|Add topic|Promise\.allSettled|partial_topics/);
});

test("account exposes custom plan usage and Stripe-only management", () => {
  assert.match(accountSource, /<BillingPlanUsageCard account=\{account\} \/>/);
  assert.match(accountSource, /active Stripe subscription is canceled automatically/);
  assert.match(actionsSource, /createStripeCheckout\(plan\)/);
  assert.match(actionsSource, /createStripePortal\(\)/);
  assert.match(actionsSource, /Manage subscription/);
  assert.match(actionsSource, /incomplete_expired/);
  assert.match(actionsSource, /USD\/month/);
  assert.match(actionsSource, /U\.S\. Stripe subscription/);
  assert.doesNotMatch(actionsSource, /apple|App Store|StoreKit/i);
  assert.doesNotMatch(cardSource, /apple|App Store|StoreKit/i);
  assert.match(apiSource, /JSON\.stringify\(\{ plan \}\)/);
});

test("billing refresh is single-flight, sequential, cleanup-safe, and the quota dialog receives focus", () => {
  assert.match(hookSource, /activeRefresh\?\.accountId === accountId[\s\S]*return activeRefresh\.promise/);
  assert.match(hookSource, /activeRefreshRef\.current\?\.controller\.abort\(\)/);
  assert.match(hookSource, /fetchBillingStatus\(controller\.signal\)/);
  assert.match(hookSource, /const intervals = \[0, 1_200, 1_800, 3_000\]/);
  assert.match(hookSource, /await refresh\(\)[\s\S]*window\.setTimeout/);
  assert.match(gateSource, /ref=\{closeButtonRef\}/);
  assert.match(gateSource, /closeButtonRef\.current\?\.focus\(\)/);
  assert.match(gateSource, /dialog\.querySelectorAll<HTMLElement>/);
  assert.match(gateSource, /returnFocusRef\.current\?\.focus\(\)/);
});

test("feed surfaces typed quota and verified-account failures as the billing dialog", () => {
  assert.match(feedSource, /if \(isDailySearchLimitError\(failure\)\)/);
  assert.match(feedSource, /setBillingGate\(\{ reason: "quota", requiredSearches: 1 \}\)/);
  assert.match(feedSource, /if \(isVerifiedAccountRequiredError\(failure\)\)/);
  assert.match(feedSource, /blockBillingWork\("quota"\)/);
  assert.match(feedSource, /billingWorkBlockedRef\.current/);
  assert.match(feedSource, /onBillingAvailable=\{resumeAfterBillingRefresh\}/);
  assert.match(feedSource, /<BillingGateDialog/);
});

test("an undeployed billing API shows rollout guidance with retry actions", () => {
  assert.match(hookSource, /error instanceof ApiError && error\.status === 404/);
  assert.match(hookSource, /Plans & usage are still being rolled out/);
  assert.match(cardSource, /onClick=\{\(\) => void refresh\(\)\}/);
  assert.match(gateSource, /onClick=\{\(\) => void refresh\(\)\}/);
  assert.match(cardSource, /Try again/);
  assert.match(gateSource, /Try again/);
});
