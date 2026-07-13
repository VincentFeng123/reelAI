const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const ts = require("typescript");

const filePath = path.join(__dirname, "page.tsx");
const source = fs.readFileSync(filePath, "utf8");
const sourceFile = ts.createSourceFile(
  filePath,
  source,
  ts.ScriptTarget.Latest,
  true,
  ts.ScriptKind.TSX,
);

test("learner level remains internal and has no manual feed control", () => {
  assert.match(source, /const \[knowledgeLevel, setKnowledgeLevel\] = useState/);
  assert.doesNotMatch(source, /const cycleLevel =/);
  assert.doesNotMatch(source, /updateMaterialLevel/);
  assert.doesNotMatch(source, /Change knowledge level/);
});

test("generation finals cannot reconcile after their search scope is invalidated", () => {
  const finalReconciliations = [];

  function visit(node) {
    if (
      ts.isExpressionStatement(node)
      && ts.isCallExpression(node.expression)
      && node.expression.expression.getText(sourceFile) === "reconcileGeneratedReels"
    ) {
      finalReconciliations.push(node);
    }
    ts.forEachChild(node, visit);
  }

  visit(sourceFile);
  assert.equal(finalReconciliations.length, 2, "both generation paths must remain covered");
  for (const reconciliation of finalReconciliations) {
    const block = reconciliation.parent;
    assert.ok(ts.isBlock(block), "final reconciliation must remain in a guarded block");
    const reconciliationIndex = block.statements.indexOf(reconciliation);
    const guard = block.statements[reconciliationIndex - 1];
    assert.ok(guard && ts.isIfStatement(guard), "final reconciliation must immediately follow a scope guard");
    assert.equal(
      guard.expression.getText(sourceFile),
      "!isSearchScopeActive(searchScope)",
      "the final inventory must check the active search scope",
    );
    assert.ok(ts.isBlock(guard.thenStatement));
    assert.ok(
      guard.thenStatement.statements.some((statement) => ts.isReturnStatement(statement)),
      "an invalidated generation must return before reconciliation",
    );
  }
});

test("feed generation keeps a twelve-clip unseen reservoir and resumes durable jobs", () => {
  assert.match(source, /const READY_RESERVOIR_TARGET = 12;/);
  assert.match(source, /const READY_RESERVOIR_REFILL_THRESHOLD = 4;/);
  assert.match(source, /rememberFeedGenerationJob\(row\.materialId, row\.data!\)/);
  assert.match(source, /generationJobId: activeGenerationJob\?\.jobId/);
  assert.match(source, /idleTimeoutMs: GENERATION_STREAM_IDLE_TIMEOUT_MS/);
  assert.match(source, /consecutiveIdleWindows >= 2/);
});

test("pagination remains available while background generation is active", () => {
  const maybeLoadMoreStart = source.indexOf("const maybeLoadMore = useCallback(() => {");
  const maybeLoadMoreEnd = source.indexOf("const shouldBlockDownwardAtEnd", maybeLoadMoreStart);
  assert.ok(maybeLoadMoreStart >= 0 && maybeLoadMoreEnd > maybeLoadMoreStart);
  const callbackText = source.slice(maybeLoadMoreStart, maybeLoadMoreEnd);
  const pageFetchIndex = callbackText.indexOf("if (hasMore && !isFetchingRef.current)");
  const generationGateIndex = callbackText.indexOf("!isGeneratingRef.current");
  assert.ok(pageFetchIndex >= 0, "available feed pages must be fetched first");
  assert.ok(generationGateIndex > pageFetchIndex, "active generation must not block persisted page fetching");
});

test("authoritative finals lock the watched prefix and reorder only the unseen tail", () => {
  assert.match(source, /const lockedPrefixLength = Math\.min\(currentRows\.length, activeIndexRef\.current \+ 1\)/);
  assert.match(source, /const lockedPrefix = currentRows\.slice\(0, lockedPrefixLength\)/);
  assert.match(source, /const stableUnseenRows = currentRows\.slice\(lockedPrefixLength\)/);
  assert.match(source, /const reordered = dedupeByIdentity\(\[\.\.\.lockedPrefix, \.\.\.authoritativeTail\]\)/);
});

test("a tail gesture is retained until the next reel arrives", () => {
  assert.match(source, /pendingAutoplayAdvanceRef\.current = true;\s+setPendingTailAdvance\(true\);\s+maybeLoadMore\(\)/);
  assert.match(source, /setPendingTailAdvance\(false\);\s+jumpOneReel\(1\)/);
  assert.match(source, /Your next swipe will continue automatically\./);
});

test("progress copy reports stages instead of a fabricated requested total", () => {
  assert.match(source, /Finding the first clips/);
  assert.match(source, /\$\{ready\} ready · improving the rest/);
  assert.doesNotMatch(source, /received: number; requested: number/);
  assert.doesNotMatch(source, /reels ready`/);
});

test("initial feed loads retain a level baseline for every grouped material", () => {
  const loadPageStart = source.indexOf("const loadPage = useCallback(");
  const loadPageEnd = source.indexOf("const requestMore = useCallback(", loadPageStart);
  assert.ok(loadPageStart >= 0 && loadPageEnd > loadPageStart);
  const callbackText = source.slice(loadPageStart, loadPageEnd);
  assert.match(callbackText, /for \(const row of successful\)/);
  assert.match(callbackText, /knowledgeLevelByMaterialRef\.current\.set\(row\.materialId, rowLevel\)/);
  assert.match(callbackText, /knowledgeLevelByMaterialRef\.current\.get\(feedMaterialIds\[0\]\)/);
});

test("restored and grouped level changes invalidate old generation before changing inventory", () => {
  const rerankStart = source.indexOf("const rerankUnseenTail = useCallback(async () => {");
  const rerankEnd = source.indexOf("const reportActiveReelProgress", rerankStart);
  assert.ok(rerankStart >= 0 && rerankEnd > rerankStart);
  const callbackText = source.slice(rerankStart, rerankEnd);
  const groupedLevelIndex = callbackText.indexOf("nextLevelsByMaterial.set(feedMaterialIds[index], nextLevel)");
  const restoredSessionGuard = callbackText.indexOf("previousLevel === undefined || previousLevel !== nextLevel");
  const clearIndex = callbackText.indexOf("clearGenerationTracking();", restoredSessionGuard);
  const renewIndex = callbackText.indexOf("renewActiveSearchScope();", restoredSessionGuard);
  const inventoryIndex = callbackText.indexOf("updateSessionReels(nextReels);", renewIndex);
  assert.ok(groupedLevelIndex >= 0, "every grouped material must be checked independently");
  assert.ok(restoredSessionGuard > groupedLevelIndex, "an absent restored-session baseline must count as stale");
  assert.ok(clearIndex > restoredSessionGuard, "old durable-job tracking must be cleared");
  assert.ok(renewIndex > clearIndex, "the old search scope must be invalidated");
  assert.ok(inventoryIndex > renewIndex, "the new inventory must only be applied after stale generation is aborted");
  assert.doesNotMatch(callbackText, /knowledgeLevel && nextLevel !== knowledgeLevel/);
  assert.doesNotMatch(source, /Change knowledge level/);
  assert.doesNotMatch(source, /auto-adjusting/);
});
