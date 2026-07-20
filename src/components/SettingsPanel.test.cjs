const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const source = fs.readFileSync(path.join(__dirname, "SettingsPanel.tsx"), "utf8");
const globalStyles = fs.readFileSync(path.join(__dirname, "../app/globals.css"), "utf8");

test("settings expose the five modal sections and responsive category navigation", () => {
  assert.match(source, /export type SettingsSection = "search" \| "playback" \| "plan" \| "data" \| "account"/);
  assert.match(source, /Search/);
  assert.match(source, /Playback/);
  assert.match(source, /Plan & Usage/);
  assert.match(source, /Data Controls/);
  assert.match(source, /Account/);
  assert.match(source, /hidden w-\[180px\][\s\S]*md:flex/);
  assert.match(source, /md:hidden[\s\S]*overflow-x-auto/);
  assert.match(source, /stroke-\[1\.5\]/);
});

test("generation speed and deprecated settings are not exposed", () => {
  assert.doesNotMatch(source, /Generation speed/);
  assert.doesNotMatch(source, /GENERATION_MODE_OPTIONS/);
  assert.doesNotMatch(source, /Default input mode/);
  assert.doesNotMatch(source, /Transcript source/);
  assert.doesNotMatch(source, /Target clip duration/);
  assert.match(source, /Hidden legacy values and the global Fast\/Slow choice remain authoritative/);
  assert.match(source, /const currentSettings = readStudyReelsSettings\(\)/);
  assert.match(source, /saveStudyReelsSettings\(\{[\s\S]*\.\.\.currentSettings/);
});

test("search and playback preferences retain explicit save and dirty-state semantics", () => {
  assert.match(source, /Similarity threshold/);
  assert.match(source, /Creative Commons only/);
  assert.match(source, /Source video length/);
  assert.match(source, /Start muted/);
  assert.match(source, /Autoplay next reel/);
  assert.match(source, /savePreferences: \(\) => void/);
  assert.match(source, /discardUnsavedChanges: \(\) => void/);
  assert.match(source, /hasUnsavedChanges: \(\) => boolean/);
  assert.match(source, /disabled=\{!settingsHydrated \|\| !hasUnsavedChanges\}/);
  assert.match(source, /onUnsavedChangesChange\?\.\(settingsHydrated && hasUnsavedChanges\)/);
});

test("search settings restore the dial and use the custom rounded dropdown", () => {
  assert.match(source, /data-similarity-dial="true"/);
  assert.match(source, /role="slider"/);
  assert.match(source, /strokeDasharray=\{`\$\{progressLength\}/);
  assert.match(source, /setPointerCapture\(event\.pointerId\)/);
  assert.match(source, /touchAction: "none"/);
  assert.match(source, /<SimilarityDial[\s\S]*value=\{draft\.minRelevanceThreshold\}/);
  assert.match(source, /<CustomSelect[\s\S]*label="Source video length"[\s\S]*rounded-full/);
  assert.doesNotMatch(source, /<select/);
});

test("settings content uses compact padding and a narrower category rail", () => {
  assert.match(source, /hidden w-\[180px\][\s\S]*px-3 py-4/);
  assert.match(source, /min-h-\[68px\][\s\S]*px-3 py-3 sm:px-4/);
  assert.match(source, /overflow-y-auto overscroll-contain px-3 pb-\[calc\(5\.5rem\+env\(safe-area-inset-bottom\)\)\] pt-4 sm:px-5 md:px-6 md:pb-24 md:pt-5/);
  assert.doesNotMatch(source, /p-5 sm:p-6/);
});

test("settings close controls stay transparent until the shared hover tone and the mobile rail uses the top fade", () => {
  const closeButtonClasses = Array.from(source.matchAll(/aria-label="Close settings"\s+className="([^"]+)"/g), (match) => match[1]);
  assert.equal(closeButtonClasses.length, 2);
  closeButtonClasses.forEach((className) => {
    assert.doesNotMatch(className, /(?:^|\s)bg-white/);
    assert.match(className, /hover:bg-white\/\[0\.07\]/);
  });
  assert.match(source, /top-nav-fade top-nav-fade-charcoal sticky top-0[\s\S]*md:hidden/);
});

test("availability is inline and no nested settings modal remains", () => {
  assert.match(source, /buildSettingsAvailabilityState/);
  assert.doesNotMatch(source, /Save search settings to see their availability impact/);
  assert.match(source, /availabilityState\.message \? \(/);
  assert.match(source, /Main limits:/);
  assert.doesNotMatch(source, /ViewportModalPortal/);
  assert.doesNotMatch(source, /role="dialog"/);
  assert.match(source, /onAvailabilityModalStateChange\?\.\(null\)/);
});

test("similarity copy uses the dial height for deliberate vertical alignment", () => {
  assert.match(source, /flex min-w-0 self-stretch flex-col py-1/);
  assert.match(source, /my-auto max-w-\[17rem\]/);
});

test("plan, data, and account actions use existing application services", () => {
  assert.match(source, /<BillingActions[\s\S]*status=\{billingStatus\}[\s\S]*plans=\{billingPlans\}[\s\S]*demoMode=\{demoMode\}/);
  assert.match(source, /onClearSearchData\(\)/);
  assert.match(source, /clearSetCache/);
  assert.match(source, /resetDefaults/);
  assert.match(source, /changeCommunityPassword\(\{ currentPassword, newPassword \}\)/);
  assert.match(source, /deleteCommunityAccount\(\{ currentPassword: deletePassword \}\)/);
  assert.match(source, /logoutCommunityAccount\(\)/);
  assert.match(source, /resendCommunityVerification\(\)/);
  assert.match(source, /verifyCommunityAccount\(\{ code: verificationCode\.trim\(\) \}\)/);
  assert.match(source, /billingStatus \?[\s\S]*billingError \?[\s\S]*Try again/);
});

test("data control actions are wider with less vertical padding", () => {
  const dataControlsStart = source.indexOf('activeSection === "data"');
  const dataControls = source.slice(dataControlsStart, source.indexOf('activeSection === "account"', dataControlsStart));
  assert.equal((dataControls.match(/h-9 min-w-\[96px\]/g) || []).length, 3);
  assert.doesNotMatch(dataControls, /py-2\.5/);
});

test("password updates open a faded three-field account subview", () => {
  assert.match(source, /const ACCOUNT_VIEW_FADE_MS = 420/);
  assert.match(source, /accountView[\s\S]*"overview" \| "password"/);
  assert.match(source, /switchAccountView\("password"\)/);
  assert.match(source, /transition-opacity duration-\[420ms\][\s\S]*accountContentVisible \? "opacity-100" : "opacity-0"/);
  assert.match(source, /Current password[\s\S]*New password[\s\S]*Confirm new password/);
  assert.match(source, /function PasswordField/);
  assert.match(source, /type=\{revealed \? "text" : "password"\}/);
  assert.match(source, /aria-pressed=\{revealed\}/);
  assert.equal((source.match(/<PasswordField/g) || []).length, 4);
  assert.doesNotMatch(source, /Your new password must be at least eight characters/);
  assert.doesNotMatch(source, /max-w-\[30rem\]/);
  assert.match(source, /<form\s+className="w-full"/);
  assert.match(source, /className="h-11 w-full/);
  assert.match(source, /className="mt-5 flex w-full flex-wrap justify-end gap-2"/);
  assert.match(source, /newPassword !== confirmNewPassword/);
  assert.match(source, /New passwords do not match/);
  assert.match(source, /type="submit"[\s\S]*Update password/);
  assert.match(source, /switchAccountView\("overview"\)/);
  assert.match(source, /setConfirmNewPassword\(""\)/);
});

test("settings save floats above safely padded scrolling without a bottom strip", () => {
  assert.doesNotMatch(source, /data-settings-footer-cover="true"/);
  assert.match(source, /data-settings-save-dock="true"/);
  assert.match(source, /pointer-events-none absolute inset-x-0 bottom-\[max\(0\.75rem,env\(safe-area-inset-bottom\)\)\] z-30/);
  assert.match(source, /pointer-events-auto min-w-\[94px\][\s\S]*Save/);
  assert.match(source, /max-w-\[720px\][\s\S]*Save/);
  const saveDock = source.slice(source.indexOf('data-settings-save-dock="true"') - 250);
  assert.doesNotMatch(saveDock, /bg-\[#202020\]/);
});

test("demo settings keep destructive, auth, cache, and billing actions local", () => {
  assert.match(source, /demoMode\?: boolean/);
  assert.match(source, /Password changes are disabled for the demo account/);
  assert.match(source, /Use Exit demo from the account menu/);
  assert.match(source, /Account deletion is disabled for the demo account/);
  assert.match(source, /Demo cache left unchanged/);
});

test("settings use tonal borderless surfaces without intermittent white focus boxes", () => {
  assert.doesNotMatch(source, /border-white/);
  assert.doesNotMatch(source, /backdrop-blur-2xl/);
  assert.match(source, /bg-\[#202020\]/);
  assert.match(source, /bg-\[#191919\]/);
  assert.match(source, /settings-surface/);
  assert.match(globalStyles, /-webkit-tap-highlight-color: transparent/);
  assert.match(globalStyles, /:focus-visible[\s\S]*outline: none !important/);
  assert.match(globalStyles, /:focus-visible:not\(:disabled\)[\s\S]*opacity: 0\.82/);
  assert.match(source, /role="switch"/);
  assert.match(source, /transition-transform duration-300 ease-out[\s\S]*translate-x-\[23px\][\s\S]*translate-x-\[3px\]/);
});
