const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const ts = require("typescript");

const filePath = path.join(__dirname, "page.tsx");
const source = fs.readFileSync(filePath, "utf8");
const reelCardSource = fs.readFileSync(path.join(__dirname, "../../components/ReelCard.tsx"), "utf8");
const feedQuerySource = fs.readFileSync(path.join(__dirname, "../../lib/feedQuery.ts"), "utf8");
const typesSource = fs.readFileSync(path.join(__dirname, "../../lib/types.ts"), "utf8");
const sourceFile = ts.createSourceFile(
  filePath,
  source,
  ts.ScriptTarget.Latest,
  true,
  ts.ScriptKind.TSX,
);

function findVariableInitializer(name) {
  let initializer = null;
  function visit(node) {
    if (
      ts.isVariableDeclaration(node)
      && ts.isIdentifier(node.name)
      && node.name.text === name
    ) {
      initializer = node.initializer;
      return;
    }
    ts.forEachChild(node, visit);
  }
  visit(sourceFile);
  assert.ok(initializer, `expected ${name} declaration`);
  return initializer;
}

function compileUseCallback(name, bindings) {
  const initializer = findVariableInitializer(name);
  assert.ok(ts.isCallExpression(initializer), `${name} must remain a useCallback call`);
  const callback = initializer.arguments[0];
  assert.ok(
    callback && (ts.isArrowFunction(callback) || ts.isFunctionExpression(callback)),
    `${name} must have a callable first argument`,
  );
  const compiled = ts.transpile(
    `const callback = ${callback.getText(sourceFile)};`,
    { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS },
  );
  const names = Object.keys(bindings);
  const factory = new Function(...names, `${compiled}\nreturn callback;`);
  return factory(...names.map((key) => bindings[key]));
}

function compileFunctionDeclaration(name) {
  let declaration = null;
  function visit(node) {
    if (ts.isFunctionDeclaration(node) && node.name?.text === name) {
      declaration = node;
      return;
    }
    ts.forEachChild(node, visit);
  }
  visit(sourceFile);
  assert.ok(declaration, `expected ${name} function declaration`);
  const compiled = ts.transpile(
    declaration.getText(sourceFile),
    { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS },
  );
  return new Function(`${compiled}\nreturn ${name};`)();
}

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
  assert.equal(finalReconciliations.length, 3, "every generation path must remain covered");
  for (const reconciliation of finalReconciliations) {
    const reconciliationCall = reconciliation.expression;
    assert.equal(
      reconciliationCall.arguments[2]?.getText(sourceFile),
      "{ preserveUnmatchedUnseen: true }",
      "stream settlement must explicitly retain unmatched provisional rows",
    );
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

test("feed generation uses the shared fast and slow ready ceilings and resumes durable jobs", () => {
  assert.match(source, /const FAST_READY_RESERVOIR_TARGET = 8;/);
  assert.match(source, /const SLOW_READY_RESERVOIR_TARGET = 12;/);
  assert.match(source, /return mode === "fast" \? FAST_READY_RESERVOIR_TARGET : SLOW_READY_RESERVOIR_TARGET;/);
  assert.match(source, /prefetch: readyReservoirTarget\(requestGenerationMode\)/);
  assert.match(source, /Math\.min\(readyReservoirTarget\(generationMode\), currentCount \+ perTopicBatch\)/);
  assert.match(source, /Math\.min\(readyReservoirTarget\("fast"\), currentCount \+ perTopicBatch\)/);
  assert.match(source, /const READY_RESERVOIR_REFILL_THRESHOLD = 4;/);
  assert.match(source, /rememberFeedGenerationJob\(row\.materialId, row\.data!\)/);
  assert.match(source, /generationJobId: activeGenerationJob\?\.jobId/);
  assert.match(source, /idleTimeoutMs: GENERATION_STREAM_IDLE_TIMEOUT_MS/);
  assert.match(source, /consecutiveIdleWindows >= 2/);
});

test("feed-owned jobs start the shared stream immediately and only once per material", async () => {
  const generationConsumerByMaterialRef = { current: new Map() };
  const generationBatchTokensRef = { current: new Set() };
  const isGeneratingRef = { current: false };
  const streamCalls = [];
  const appended = [];
  const reconciled = [];
  const finishStreams = new Map();
  let generatingMore = false;
  let recoveryPhase = "idle";
  const syncGenerationLockState = compileUseCallback("syncGenerationLockState", {
    generationConsumerByMaterialRef,
    generationBatchTokensRef,
    isGeneratingRef,
    setGeneratingMore: (active) => {
      generatingMore = active;
    },
    setRecoveryPhase: (phase) => {
      recoveryPhase = phase;
    },
  });
  const claimGenerationConsumer = compileUseCallback("claimGenerationConsumer", {
    generationConsumerByMaterialRef,
    syncGenerationLockState,
  });
  const releaseGenerationConsumer = compileUseCallback("releaseGenerationConsumer", {
    generationConsumerByMaterialRef,
    syncGenerationLockState,
  });
  const callback = compileUseCallback("consumeFeedGenerationJob", {
    claimGenerationConsumer,
    releaseGenerationConsumer,
    GENERATION_STREAM_IDLE_TIMEOUT_MS: 35_000,
    generateReelsStream: async (params) => {
      streamCalls.push(params);
      params.onCandidate({ reel_id: `candidate-${params.materialId}` });
      await new Promise((resolve) => {
        finishStreams.set(params.materialId, resolve);
      });
      params.onTerminal("completed");
      return { reels: [{ reel_id: `final-${params.materialId}` }] };
    },
    isSearchScopeActive: () => true,
    noteGenerationTerminal: () => {},
    appendGeneratedReels: (rows) => {
      appended.push(...rows);
      return { reels: rows, addedReels: rows, addedCount: rows.length, updatedCount: 0 };
    },
    markRecoveryProgress: () => {},
    reconcileGeneratedReels: (...args) => reconciled.push(args),
    isRequestInterruptedError: () => false,
    console,
  });
  const searchScope = { key: "material", seq: 1, controller: new AbortController() };

  const first = callback(
    "material-a",
    { generation_job_id: "job-a", generation_job_status: "running" },
    searchScope,
  );
  const duplicate = callback(
    "material-a",
    { generation_job_id: "job-a", generation_job_status: "running" },
    searchScope,
  );
  const second = callback(
    "material-b",
    { generation_job_id: "job-b", generation_job_status: "queued" },
    searchScope,
  );

  assert.ok(first instanceof Promise);
  assert.ok(second instanceof Promise);
  assert.equal(duplicate, null);
  assert.deepEqual(streamCalls.map((call) => call.generationJobId), ["job-a", "job-b"]);
  assert.equal(isGeneratingRef.current, true);
  assert.equal(generatingMore, true);
  assert.equal(recoveryPhase, "generating");
  assert.deepEqual(appended.map((reel) => reel.reel_id), ["candidate-material-a", "candidate-material-b"]);

  finishStreams.get("material-a")();
  await first;
  assert.equal(generationConsumerByMaterialRef.current.size, 1);
  assert.equal(isGeneratingRef.current, true, "material B must retain the global lock after material A settles");

  const otherGeneration = Symbol("other-generation");
  generationBatchTokensRef.current.add(otherGeneration);
  finishStreams.get("material-b")();
  await second;
  assert.equal(generationConsumerByMaterialRef.current.size, 0);
  assert.equal(isGeneratingRef.current, true, "another generation batch must retain the global lock");

  generationBatchTokensRef.current.delete(otherGeneration);
  syncGenerationLockState();

  assert.equal(isGeneratingRef.current, false);
  assert.equal(generatingMore, false);
  assert.equal(recoveryPhase, "idle");
  assert.equal(reconciled.length, 2);
  assert.deepEqual(reconciled.map((args) => args[1][0].reel_id), ["final-material-a", "final-material-b"]);
  assert.ok(reconciled.every((args) => args[2].preserveUnmatchedUnseen === true));
});

test("feed starts returned generation jobs before yielding to bootstrap", () => {
  const loadPageStart = source.indexOf("const loadPage = useCallback(");
  const loadPageEnd = source.indexOf("const requestMore = useCallback(", loadPageStart);
  const loadPageText = source.slice(loadPageStart, loadPageEnd);
  const rememberIndex = loadPageText.indexOf("rememberFeedGenerationJob(row.materialId, row.data!)");
  const consumeIndex = loadPageText.indexOf("void consumeFeedGenerationJob(");
  assert.ok(rememberIndex >= 0);
  assert.ok(consumeIndex > rememberIndex, "the feed response must register then immediately consume its durable job");
  assert.doesNotMatch(loadPageText, /canStartFeedGenerationConsumers/);
});

test("the Reel contract explicitly retains v4 selection metadata", () => {
  assert.match(typesSource, /video_id\?: string \| null;/);
  assert.match(typesSource, /topic_relevance\?: number \| null;/);
  assert.match(typesSource, /selection_contract_version\?: string \| null;/);
});

test("restored reconciliation removes cached unseen rows and stream settlement drops rejected provisionals", () => {
  const currentRows = [
    { reel_id: "watched", video_url: "watched", video_title: "Watched" },
    { reel_id: "current", video_url: "current", video_title: "Cached current" },
    { reel_id: "prior-unseen", video_url: "prior-unseen", video_title: "Prior unseen" },
    { reel_id: "rejected-provisional", video_url: "rejected-provisional", video_title: "Rejected provisional" },
  ];
  const reelsRef = { current: currentRows };
  const activeIndexRef = { current: 1 };
  const watchedFrontierIndexRef = { current: 1 };
  const orderReelsByDifficulty = compileUseCallback("orderReelsByDifficulty", {});
  let renderedRows = currentRows;
  const callback = compileUseCallback("reconcileGeneratedReels", {
    reelsRef,
    activeIndexRef,
    watchedFrontierIndexRef,
    orderReelsByDifficulty,
    reelClipKey: (reel) => reel.video_url,
    dedupeByIdentity: (rows) => {
      const seen = new Set();
      return rows.filter((row) => !seen.has(row.reel_id) && seen.add(row.reel_id));
    },
    updateSessionReels: (rows) => {
      reelsRef.current = rows;
      renderedRows = rows;
    },
    setTotal: () => {},
  });

  callback(
    [],
    [
      {
        reel_id: "current",
        video_url: "current",
        video_title: "",
        captions: [],
        t_start: 10.125,
        t_end: 30.875,
      },
      { reel_id: "new-unseen", video_url: "new-unseen", video_title: "New unseen" },
    ],
    { preserveUnmatchedUnseen: false },
  );
  assert.deepEqual(renderedRows.map((row) => row.reel_id), ["watched", "current", "new-unseen"]);
  assert.equal(renderedRows[1].video_title, "");
  assert.deepEqual(renderedRows[1].captions, []);
  assert.equal(renderedRows[1].t_start, 10.125);
  assert.equal(renderedRows[1].t_end, 30.875);

  reelsRef.current = currentRows;
  callback(
    [{ reel_id: "rejected-provisional", video_url: "rejected-provisional" }],
    [
      { reel_id: "current", video_url: "current", video_title: "Authoritative current" },
      { reel_id: "new-unseen", video_url: "new-unseen", video_title: "New unseen" },
    ],
    { preserveUnmatchedUnseen: true },
  );
  assert.deepEqual(
    renderedRows.map((row) => row.reel_id),
    ["watched", "current", "new-unseen", "prior-unseen"],
  );
});

test("generation settlement freezes the watched frontier and orders only the unseen tail by difficulty", () => {
  const currentRows = [
    { reel_id: "watched", video_url: "watched", difficulty: 0.9 },
    { reel_id: "backtracked-current", video_url: "backtracked-current", difficulty: 0.8 },
    { reel_id: "watched-frontier", video_url: "watched-frontier", difficulty: 0.7 },
    { reel_id: "rejected-provisional", video_url: "rejected-provisional", difficulty: 0.2 },
  ];
  const reelsRef = { current: currentRows };
  const activeIndexRef = { current: 1 };
  const watchedFrontierIndexRef = { current: 2 };
  const orderReelsByDifficulty = compileUseCallback("orderReelsByDifficulty", {});
  let renderedRows = currentRows;
  const callback = compileUseCallback("reconcileGeneratedReels", {
    reelsRef,
    activeIndexRef,
    watchedFrontierIndexRef,
    orderReelsByDifficulty,
    reelClipKey: (reel) => reel.video_url,
    dedupeByIdentity: (rows) => {
      const seen = new Set();
      return rows.filter((row) => !seen.has(row.reel_id) && seen.add(row.reel_id));
    },
    updateSessionReels: (rows) => {
      reelsRef.current = rows;
      renderedRows = rows;
    },
    setTotal: () => {},
  });

  callback(
    [{ reel_id: "rejected-provisional", video_url: "rejected-provisional", difficulty: 0.2 }],
    [
      { reel_id: "watched-frontier", video_url: "watched-frontier", difficulty: 0.7, video_title: "Updated" },
      { reel_id: "hard", video_url: "hard", difficulty: 3 },
      { reel_id: "tie-first", video_url: "tie-first" },
      { reel_id: "easy", video_url: "easy", difficulty: -2 },
      { reel_id: "tie-second", video_url: "tie-second", difficulty: 0.5 },
    ],
    { preserveUnmatchedUnseen: true },
  );

  assert.deepEqual(renderedRows.map((row) => row.reel_id), [
    "watched",
    "backtracked-current",
    "watched-frontier",
    "easy",
    "tie-first",
    "tie-second",
    "hard",
  ]);
  assert.equal(renderedRows[2].video_title, "Updated");
});

test("grouped material batches use stable difficulty order instead of round robin", () => {
  const orderReelsByDifficulty = compileUseCallback("orderReelsByDifficulty", {});
  const mergeReelBatchesByDifficulty = compileUseCallback("mergeReelBatchesByDifficulty", {
    orderReelsByDifficulty,
  });

  const merged = mergeReelBatchesByDifficulty([
    [
      { reel_id: "hard", difficulty: 0.9 },
      { reel_id: "tie-first" },
    ],
    [
      { reel_id: "easy", difficulty: 0.1 },
      { reel_id: "tie-second", difficulty: Number.POSITIVE_INFINITY },
    ],
  ]);

  assert.deepEqual(merged.map((row) => row.reel_id), [
    "easy",
    "tie-first",
    "tie-second",
    "hard",
  ]);
});

test("a restored feed reconciles durable inventory with its restored mode and a stale-scope guard", () => {
  const loadPageStart = source.indexOf("const loadPage = useCallback(");
  const loadPageEnd = source.indexOf("const requestMore = useCallback(", loadPageStart);
  assert.ok(loadPageStart >= 0 && loadPageEnd > loadPageStart);
  const loadPageText = source.slice(loadPageStart, loadPageEnd);
  assert.match(loadPageText, /const requestGenerationMode = options\?\.generationMode \?\? generationMode/);
  assert.match(loadPageText, /generationMode: requestGenerationMode/);
  assert.match(
    loadPageText,
    /reconcileGeneratedReels\(\[\], fetchedReels, \{ preserveUnmatchedUnseen: false \}\)/,
  );

  const hydrationStart = source.indexOf("let restoredSession: FeedSessionSnapshot | null = null;");
  const hydrationEnd = source.indexOf("useEffect(() => {", hydrationStart);
  assert.ok(hydrationStart >= 0 && hydrationEnd > hydrationStart);
  const hydrationText = source.slice(hydrationStart, hydrationEnd);
  assert.match(hydrationText, /setBootstrappingFirstReels\(true\);/);
  assert.match(hydrationText, /generationMode: restoredGenerationMode/);
  assert.match(hydrationText, /hydratedMaterialIdRef\.current === materialId/);
  assert.match(hydrationText, /isSearchScopeActive\(restoredSearchScope\)/);
});

test("bootstrap consumes duplicate persisted pages before considering generation", async () => {
  const reelsRef = { current: Array.from({ length: 10 }, (_, index) => ({ reel_id: `r${index}` })) };
  const pageLoads = [];
  let generationRequests = 0;
  const searchScope = { key: "material", seq: 1 };
  const callback = compileUseCallback("bootstrapFirstReels", {
    materialId: "material",
    isGeneratingRef: { current: false },
    canRequestMore: true,
    isIngestMaterial: false,
    setBootstrappingFirstReels: () => {},
    setCanRequestMore: () => {},
    setFeedPagesExhausted: () => {},
    activeSearchScopeRef: { current: searchScope },
    page: 1,
    hasMore: true,
    total: 12,
    PAGE_SIZE: 5,
    reelsRef,
    activeIndexRef: { current: 0 },
    readyReservoirTarget: () => 12,
    generationMode: "slow",
    loadPage: async (targetPage, options) => {
      pageLoads.push({ targetPage, options });
      if (targetPage === 2) {
        return { addedCount: 0, exhausted: false };
      }
      reelsRef.current = [
        ...reelsRef.current,
        { reel_id: "r10" },
        { reel_id: "r11" },
        { reel_id: "r12" },
      ];
      return { addedCount: 3, exhausted: true };
    },
    isSearchScopeActive: (scope) => scope === searchScope,
    requestMore: async () => {
      generationRequests += 1;
      return [];
    },
    runFastTopUp: () => {},
  });

  await callback();

  assert.deepEqual(pageLoads.map((row) => row.targetPage), [2, 3]);
  assert.ok(pageLoads.every((row) => row.options.autofill === false));
  assert.equal(generationRequests, 0, "persisted inventory filled the reservoir");
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
  assert.match(source, /Math\.max\(activeIndexRef\.current, watchedFrontierIndexRef\.current\) \+ 1/);
  assert.match(source, /const lockedPrefix = currentRows\.slice\(0, lockedPrefixLength\)/);
  assert.match(source, /const stableUnseenRows = currentRows\.slice\(lockedPrefixLength\)/);
  assert.match(source, /const orderedTail = orderReelsByDifficulty\(authoritativeTail\)/);
  assert.match(source, /const reordered = dedupeByIdentity\(\[\.\.\.lockedPrefix, \.\.\.orderedTail\]\)/);
  assert.match(source, /watchedFrontierIndex: dedupedReels\.length > 0/);
});

test("same-source clips remain distinct when their authoritative float ranges differ", () => {
  const normalizeClipKeyTime = compileUseCallback("normalizeClipKeyTime", {});
  const reelClipKey = compileUseCallback("reelClipKey", {
    normalizeClipKeyTime,
    sourceVideoKeyFromUrl: () => "same-video",
  });
  const dedupeByIdentity = compileUseCallback("dedupeByIdentity", { reelClipKey });
  const rows = dedupeByIdentity([
    { reel_id: "facet-a", video_url: "https://youtu.be/same", t_start: 10.125, t_end: 31.875 },
    { reel_id: "facet-b", video_url: "https://youtu.be/same", t_start: 45.25, t_end: 80.5 },
    { reel_id: "duplicate-range", video_url: "https://youtu.be/same", t_start: 10.125, t_end: 31.875 },
  ]);
  assert.deepEqual(rows.map((reel) => reel.reel_id), ["facet-a", "facet-b"]);
});

test("web and iOS canonicalize the same supported YouTube source URL forms", () => {
  const sourceVideoKeyFromUrl = compileFunctionDeclaration("sourceVideoKeyFromUrl");
  const videoId = "dQw4w9WgXcQ";
  const supported = [
    `https://www.youtube.com/watch?v=${videoId}&list=PL123`,
    `https://m.youtube.com/watch?v=${videoId}`,
    `https://music.youtube.com/watch?v=${videoId}`,
    `https://www.youtube.com/shorts/${videoId}`,
    `https://www.youtube.com/embed/${videoId}`,
    `https://www.youtube.com/v/${videoId}`,
    `https://www.youtube.com/live/${videoId}`,
    `https://youtu.be/${videoId}?t=12`,
    `https://www.youtube-nocookie.com/embed/${videoId}`,
  ];

  assert.deepEqual(supported.map(sourceVideoKeyFromUrl), supported.map(() => videoId));
  assert.equal(
    sourceVideoKeyFromUrl(" https://example.com/embed/dQw4w9WgXcQ?view=full#part "),
    "https://example.com/embed/dQw4w9WgXcQ?view=full#part",
  );
  assert.equal(
    sourceVideoKeyFromUrl("https://www.youtube.com/live/not-valid?feature=share"),
    "https://www.youtube.com/live/not-valid?feature=share",
  );
});

test("legacy duration settings are readable but removed from newly written feed URLs", () => {
  assert.match(feedQuerySource, /getParam\("target_clip_duration_sec"\)/);
  assert.match(feedQuerySource, /params\.delete\("target_clip_duration_sec"\)/);
  assert.match(feedQuerySource, /params\.delete\("target_clip_duration_min_sec"\)/);
  assert.match(feedQuerySource, /params\.delete\("target_clip_duration_max_sec"\)/);
  assert.doesNotMatch(feedQuerySource, /params\.set\("target_clip_duration/);
});

test("YouTube playback honors float boundaries with a sub-50ms end watchdog", () => {
  assert.match(reelCardSource, /const CLIP_END_POLL_INTERVAL_MS = 20;/);
  assert.match(reelCardSource, /clipEndRaw > clipStart \? clipEndRaw : clipStart/);
  assert.match(reelCardSource, /if \(playerTime \+ 0\.01 >= clipEnd\)/);
  assert.match(
    reelCardSource,
    /if \(playerTime \+ 0\.01 >= clipEnd\) \{[\s\S]*?if \(!didHandleClipEndRef\.current\) \{[\s\S]*?player\.pauseVideo\(\);[\s\S]*?if \(autoplayEnabledRef\.current/,
  );
  assert.match(reelCardSource, /event\.target\.seekTo\(clipStart, true\)/);
  assert.doesNotMatch(reelCardSource, /clipStart \+ 1/);

  const endedHandler = reelCardSource.match(
    /else if \(state === playerState\.ENDED\) \{([\s\S]*?)\n\s*} else if \(/,
  )?.[1] || "";
  assert.match(endedHandler, /event\.target\.pauseVideo\(\)/);
  assert.match(endedHandler, /setCurrentSec\(clipDuration\)/);
  assert.doesNotMatch(endedHandler, /seekTo\(clipStart/);
  assert.doesNotMatch(endedHandler, /playVideo\(\)/);
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
