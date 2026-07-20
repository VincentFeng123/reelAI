const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const componentSource = fs.readFileSync(
  path.join(__dirname, "FullscreenLoadingScreen.tsx"),
  "utf8",
);
const globalsSource = fs.readFileSync(
  path.join(__dirname, "../app/globals.css"),
  "utf8",
);

test("fullscreen loading matches the iOS staggered ReelAI blur wordmark", () => {
  assert.match(componentSource, /\["R", "E", "E", "L", "A", "I"\]/);
  assert.match(componentSource, /animationDelay: `\$\{index \* 0\.2\}s`/);
  assert.match(componentSource, /aria-label="Loading ReelAI"/);
  assert.match(globalsSource, /\.reelai-loading-screen[\s\S]*background: #080808/);
  assert.match(globalsSource, /\.reelai-loading-wordmark[\s\S]*gap: 10px[\s\S]*font-size: 24px[\s\S]*font-weight: 400/);
  assert.match(globalsSource, /\.reelai-loading-letter[\s\S]*reelai-loading-letter-blur 1\.5s linear infinite alternate/);
  assert.match(globalsSource, /@keyframes reelai-loading-letter-blur[\s\S]*filter: blur\(4px\)/);
  assert.match(globalsSource, /prefers-reduced-motion: reduce[\s\S]*\.reelai-loading-letter[\s\S]*animation: none/);
  assert.doesNotMatch(globalsSource, /\.reelai-loading-screen[\s\S]{0,500}background-image/);
});
