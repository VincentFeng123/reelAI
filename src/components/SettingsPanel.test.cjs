const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const source = fs.readFileSync(path.join(__dirname, "SettingsPanel.tsx"), "utf8");

test("generation speed is visible and fully participates in settings persistence", () => {
  assert.match(source, /Generation speed/);
  assert.match(source, /value: "fast", label: "Fast", detail: "2 sources · up to 9 clips"/);
  assert.match(source, /value: "slow", label: "Slow", detail: "3 sources · up to 9 clips"/);
  assert.match(source, /Both modes use the same relevance, context, and sentence-boundary refinement/);
  assert.match(source, /setGenerationMode\(saved\.generationMode\)/);
  assert.match(source, /setGenerationMode\(DEFAULT_STUDY_REELS_SETTINGS\.generationMode\)/);
  assert.match(source, /savedPreferences\.generationMode !== generationMode/);
  assert.match(source, /saveStudyReelsSettings\(\{[\s\S]*generationMode,/);
  assert.match(source, /onClick=\{\(\) => setGenerationMode\(option\.value\)\}/);
  assert.match(source, /role="group"[\s\S]*aria-label="Generation speed"/);
  assert.match(source, /aria-pressed=\{selected\}/);
  assert.doesNotMatch(source, /role="radio"/);
  assert.match(source, /generationMode === "fast" \? "Fast" : "Slow"/);
  assert.doesNotMatch(source, /generationModeForChecks/);
});
