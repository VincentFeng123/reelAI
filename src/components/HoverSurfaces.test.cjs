const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const srcRoot = path.join(__dirname, "..");

function collectTsxFiles(directory) {
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const entryPath = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      return collectTsxFiles(entryPath);
    }
    return entry.isFile() && entry.name.endsWith(".tsx") ? [entryPath] : [];
  });
}

test("neutral hover surfaces use the sidenav hover tone", () => {
  const allowedSemanticHoverTones = new Set([
    "hover:bg-white",
    "hover:bg-white/70",
    "hover:bg-white/88",
    "hover:bg-white/90",
    "hover:bg-white/[0.88]",
    "hover:bg-zinc-100",
    "hover:bg-[#e8e8e8]",
    "hover:bg-[#f1eee5]",
    "hover:bg-[#7a2b33]",
    "hover:bg-[#a12831]",
  ]);
  const unexpected = [];

  for (const filePath of collectTsxFiles(srcRoot)) {
    const source = fs.readFileSync(filePath, "utf8");
    const hoverClasses = source.match(/(?<!group-)(?:enabled:)?hover:bg-[^\s"'`}]+/g) || [];
    for (const hoverClass of hoverClasses) {
      const normalized = hoverClass.replace(/^enabled:/, "");
      if (normalized !== "hover:bg-white/[0.07]" && !allowedSemanticHoverTones.has(normalized)) {
        unexpected.push(`${path.relative(srcRoot, filePath)}: ${hoverClass}`);
      }
    }
  }

  assert.deepEqual(unexpected, []);
});
