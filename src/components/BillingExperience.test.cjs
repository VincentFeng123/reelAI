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
const accountPageSource = fs.readFileSync(path.join(__dirname, "../app/account/page.tsx"), "utf8");
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
  assert.match(actionsSource, /if \(demoMode\) \{[\s\S]*Billing actions are disabled for the local demo account/);
});

test("legacy billing and signed-in account routes open the settings modal", () => {
  assert.match(accountPageSource, /requestedView === "billing"[\s\S]*router\.replace\("\/\?settings=plan"\)/);
  assert.match(accountPageSource, /if \(communityAccount\?\.isVerified\) \{[\s\S]*authCompleted \? postAuthTarget : "\/\?settings=account"/);
  assert.match(accountPageSource, /communityAccount\?\.isVerified[\s\S]*<FullscreenLoadingScreen/);
  assert.match(accountSource, /const AUTH_MODE_FADE_MS = 420/);
  assert.match(accountSource, /<FadePresence show=\{isVerificationModalOpen\}>/);
});

test("guest login and registration use the minimal dedicated account page", () => {
  assert.doesNotMatch(accountSource, /presentation\?: "page" \| "modal"|isModalPresentation|aria-label="Close account"/);
  assert.match(accountSource, /fixed inset-0 h-\[100dvh\] w-screen overflow-y-auto bg-black/);
  assert.match(accountSource, /max-w-\[380px\]/);
  assert.match(accountSource, /aria-label="ReelAI"[\s\S]*ReelAI/);
  assert.match(accountSource, /onAuthModeChange\?\.\(nextMode\)/);
  assert.match(accountPageSource, /searchParams\.get\("mode"\) === "register"/);
  assert.match(accountPageSource, /router\.replace\(query \? `\/account\?\$\{query\}` : "\/account"\)/);
  assert.match(accountPageSource, /returnTab === "settings"[\s\S]*"\/\?settings=account"/);
  assert.match(accountPageSource, /authCompleted \? postAuthTarget : "\/\?settings=account"/);
  assert.match(accountSource, /isVerificationModalOpen[\s\S]*event\.stopPropagation\(\)/);
  assert.match(accountSource, /z-\[170\]/);
  assert.match(accountSource, /event\.key === "Tab" && isVerificationModalOpen[\s\S]*dialog\.querySelectorAll<HTMLElement>/);
  assert.match(accountSource, /verificationDialogRef[\s\S]*tabIndex=\{-1\}/);
  assert.match(accountSource, /aria-hidden=\{isVerificationModalOpen\}[\s\S]*inert=\{isVerificationModalOpen\}/);
  assert.match(accountSource, /verificationReturnFocusRef[\s\S]*returnFocus\?\.isConnected[\s\S]*returnFocus\.focus\(\)/);
  assert.equal((accountSource.match(/restoreVerificationFocus\(\)/g) || []).length, 3);
  assert.match(accountSource, /loginCommunityAccount[\s\S]*onAccountChange\(result\.account\)/);
  assert.match(accountSource, /registerCommunityAccount/);
  assert.match(accountSource, /duration-\[420ms\] ease-in-out/);
  const modeTransitionSource = accountSource.slice(
    accountSource.indexOf("const switchAuthMode"),
    accountSource.indexOf("const verificationDigits"),
  );
  assert.match(modeTransitionSource, /setAuthModeTransitioning\(fadeMs > 0\)[\s\S]*setAuthMode\(nextMode\)[\s\S]*onAuthModeChange\?\.\(nextMode\)/);
  assert.doesNotMatch(modeTransitionSource, /setAuthContentVisible|requestAnimationFrame/);
  assert.match(accountSource, /\(\["login", "register"\] as const\)\.map/);
  assert.match(accountSource, /data-auth-mode=\{mode\}[\s\S]*aria-hidden=\{!isActiveMode\}[\s\S]*inert=\{!isActiveMode \|\| authModeTransitioning\}/);
  assert.match(accountSource, /col-start-1 row-start-1[\s\S]*transition-opacity duration-\[420ms\] ease-in-out[\s\S]*motion-reduce:transition-none/);
  assert.match(accountSource, /isActiveMode \? \(authModeTransitioning \? "pointer-events-none opacity-100" : "opacity-100"\) : "pointer-events-none opacity-0"/);
  assert.match(accountSource, /pendingAuthModeFocusRef\.current = nextMode/);
  assert.match(accountSource, /pendingAuthModeFocusRef\.current !== authMode[\s\S]*authModeFirstFieldRefs\.current\[authMode\]\?\.focus\(\)/);
  assert.match(accountSource, /authModeFirstFieldRefs\.current\[mode\] = node/);
  assert.match(accountSource, /bg-white[^"\n]*text-black/);
  const alternateAuthAction = accountSource.slice(
    accountSource.indexOf("Already have an account?"),
    accountSource.indexOf("</div>", accountSource.indexOf("Already have an account?")),
  );
  assert.match(alternateAuthAction, /font-semibold/);
  assert.doesNotMatch(alternateAuthAction, /bg-|rounded-/);
});

test("account back navigation uses the shared top gradient instead of a solid strip", () => {
  assert.match(
    accountSource,
    /data-top-chrome="account-back"[\s\S]{0,180}?className="top-nav-fade fixed inset-x-0 top-0/,
  );
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
  assert.match(gateSource, /setIsVisible\(false\)[\s\S]*prefers-reduced-motion: reduce[\s\S]*window\.setTimeout\(onClose, closeDelay\)/);
  assert.match(gateSource, /isVisible \? "opacity-100" : "opacity-0"/);
  assert.match(gateSource, /window\.setTimeout\(onBillingAvailable, closeDelay\)/);
  assert.doesNotMatch(gateSource, /shadow-/);
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
