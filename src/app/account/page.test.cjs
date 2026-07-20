const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const source = fs.readFileSync(path.join(__dirname, "page.tsx"), "utf8");

test("account route selects login or registration without accepting arbitrary modes", () => {
  assert.match(source, /searchParams\.get\("mode"\) === "register" \? "register" : "login"/);
  assert.match(source, /initialAuthMode=\{requestedAuthMode\}/);
  assert.match(source, /onAuthModeChange=\{onAuthModeChange\}/);
});

test("account return targets are normalized and section-specific", () => {
  assert.match(source, /value === "search" \|\| value === "community" \|\| value === "edit" \|\| value === "settings"/);
  assert.match(source, /returnTab === "settings"[\s\S]*return "\/\?settings=account"/);
  assert.match(source, /if \(returnTab\) \{[\s\S]*return `\/\?tab=\$\{returnTab\}`/);
  assert.match(source, /return "\/"/);
  assert.match(source, /returnTab === "settings"[\s\S]*\? "\/\?settings=account"[\s\S]*`\/\?tab=\$\{returnTab\}`/);
});

test("verified bootstrap sessions open Settings while completed auth returns to its origin", () => {
  const bootstrap = source.slice(source.indexOf("const validateStoredAccount"), source.indexOf("const onStorage"));
  assert.match(bootstrap, /restoreCommunityAccountFromSessionToken\(\)/);
  assert.doesNotMatch(bootstrap, /if \(!stored\?\.sessionToken\)/);
  assert.match(source, /if \(communityAccount\?\.isVerified\) \{[\s\S]*authCompleted \? postAuthTarget : "\/\?settings=account"/);
  assert.match(source, /if \(nextAccount\?\.isVerified\) \{[\s\S]*setAuthCompleted\(true\)/);
});

test("unverified accounts remain on the page for the existing verification flow", () => {
  assert.match(source, /showLoadingScreen \|\| requestedView === "billing" \|\| communityAccount\?\.isVerified/);
  assert.match(source, /communityAccount\?\.isVerified && requestedView !== "billing" \? requestedView : "default"/);
  assert.match(source, /account=\{communityAccount\}/);
});

test("auth mode URL changes preserve only the normalized return target", () => {
  const modeCallback = source.slice(
    source.indexOf("const onAuthModeChange"),
    source.indexOf("if (showLoadingScreen"),
  );
  assert.match(modeCallback, /const nextQuery = new URLSearchParams\(\)/);
  assert.match(modeCallback, /mode === "register"[\s\S]*nextQuery\.set\("mode", "register"\)/);
  assert.match(modeCallback, /nextQuery\.set\("return_tab", returnTab\)/);
  assert.match(modeCallback, /router\.replace\(query \? `\/account\?\$\{query\}` : "\/account"\)/);
  assert.doesNotMatch(modeCallback, /new URLSearchParams\(searchParams\.toString\(\)\)/);
});

test("legacy billing links keep opening plan settings", () => {
  assert.match(source, /requestedView === "billing"[\s\S]*router\.replace\("\/\?settings=plan"\)/);
});
