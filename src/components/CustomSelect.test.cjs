const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const source = fs.readFileSync(path.join(__dirname, "CustomSelect.tsx"), "utf8");

test("custom select is an accessible opacity-only fading listbox", () => {
  assert.match(source, /<FadePresence show=\{open\}>/);
  assert.match(source, /role="listbox"/);
  assert.match(source, /role="option"/);
  assert.match(source, /aria-selected=\{selected\}/);
  assert.match(source, /aria-haspopup="listbox"/);
  assert.match(source, /aria-expanded=\{open\}/);
  assert.match(source, /inert=\{!visible\}/);
  assert.match(source, /transition-opacity duration-300/);
  assert.match(source, /visible \? "opacity-100" : "pointer-events-none opacity-0"/);
  assert.match(source, /role="listbox"[\s\S]{0,240}?bg-\[#202020\]/);
  assert.doesNotMatch(source, /shadow-|transition-transform|scale-/);
});

test("custom select supports keyboard navigation and outside dismissal", () => {
  assert.match(source, /event\.key === "ArrowDown"/);
  assert.match(source, /event\.key === "ArrowUp"/);
  assert.match(source, /event\.key === "Home"/);
  assert.match(source, /event\.key === "End"/);
  assert.match(source, /onKeyDownCapture=/);
  assert.match(source, /event\.key !== "Escape"/);
  assert.match(source, /event\.stopPropagation\(\)/);
  assert.match(source, /event\.key === "Tab"/);
  assert.match(source, /document\.addEventListener\("pointerdown", onPointerDown\)/);
  assert.match(source, /triggerRef\.current\?\.focus\(\)/);
});

test("custom select supports an opt-in wide menu with a far-edge selected check", () => {
  assert.match(source, /menuClassName\?: string/);
  assert.match(source, /showSelectedCheck\?: boolean/);
  assert.match(source, /showSelectedCheck && selected/);
  assert.match(source, /className="ml-auto shrink-0 pl-8"/);
});
