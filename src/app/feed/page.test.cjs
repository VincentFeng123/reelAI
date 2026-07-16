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

function compileFunctionDeclaration(name, bindings = {}) {
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
  const names = Object.keys(bindings);
  const factory = new Function(...names, `${compiled}\nreturn ${name};`);
  return factory(...names.map((key) => bindings[key]));
}

test("debounced history sync retains its originating community session", () => {
  const sessionA = { accountId: "account-a", sessionToken: `session-a-${"a".repeat(32)}` };
  const sessionB = { accountId: "account-b", sessionToken: `session-b-${"b".repeat(32)}` };
  const pendingHistorySyncRef = { current: null };
  const historySyncTimerRef = { current: null };
  const timerCallbacks = [];
  const syncCalls = [];
  let currentSession = sessionA;
  const clearHistorySyncTimer = () => {
    historySyncTimerRef.current = null;
  };
  const queueCommunityHistorySync = (items, sessionContext) => {
    syncCalls.push({ items, sessionContext });
    return Promise.resolve();
  };
  const scheduleRemoteHistorySync = compileUseCallback("scheduleRemoteHistorySync", {
    captureCommunitySessionContext: () => currentSession,
    pendingHistorySyncRef,
    clearHistorySyncTimer,
    historySyncTimerRef,
    setTimeout: (callback, delay) => {
      assert.equal(delay, 900);
      timerCallbacks.push(callback);
      return timerCallbacks.length;
    },
    queueCommunityHistorySync,
  });
  const flushPendingHistorySync = compileUseCallback("flushPendingHistorySync", {
    clearHistorySyncTimer,
    pendingHistorySyncRef,
    queueCommunityHistorySync,
  });
  const firstItems = [{
    materialId: "material-a",
    title: "Original title",
    updatedAt: 1,
    starred: false,
    generationMode: "slow",
    source: "search",
  }];

  scheduleRemoteHistorySync(firstItems);
  firstItems[0].title = "Mutated after scheduling";
  currentSession = sessionB;
  timerCallbacks.shift()();

  assert.equal(syncCalls.length, 1);
  assert.deepEqual(syncCalls[0].sessionContext, sessionA);
  assert.equal(syncCalls[0].items[0].title, "Original title");

  scheduleRemoteHistorySync([{ ...firstItems[0], materialId: "material-b" }]);
  currentSession = sessionA;
  flushPendingHistorySync();

  assert.equal(syncCalls.length, 2);
  assert.deepEqual(syncCalls[1].sessionContext, sessionB, "unmount flush must not recapture account A");
});

test("feed session restore accepts v35 and rejects v34, v33, or unversioned snapshots", () => {
  const currentContract = "quality_silence_v35";
  assert.match(
    typesSource,
    /CURRENT_SELECTION_CONTRACT_VERSION = "quality_silence_v35"/,
    "the client contract must remain explicit and shared",
  );
  const parseFeedSessions = compileFunctionDeclaration("parseFeedSessions", {
    CURRENT_SELECTION_CONTRACT_VERSION: currentContract,
    MAX_REELS_PER_FEED_SESSION: 300,
    normalizeFeedPlaybackRate: () => 1,
  });
  const snapshot = {
    reels: [{ reel_id: "reel-1", video_url: "https://www.youtube.com/watch?v=dQw4w9WgXcQ" }],
    feedbackByReel: {},
    adaptiveExcludeReelIds: [],
    page: 1,
    total: 1,
    canRequestMore: true,
    generationMode: "slow",
    mutedPreference: true,
    autoplayEnabled: true,
    playbackRate: 1,
    activeIndex: 0,
    updatedAt: 1,
  };

  const restored = parseFeedSessions(JSON.stringify({
    current: { ...snapshot, selectionContractVersion: currentContract },
    oldCurrent: { ...snapshot, selectionContractVersion: "quality_silence_v34" },
    predecessor: { ...snapshot, selectionContractVersion: "quality_silence_v33" },
    missing: snapshot,
  }));

  assert.deepEqual(Object.keys(restored), ["current"]);
  assert.equal(restored.current.selectionContractVersion, currentContract);

  const hydrationStart = source.indexOf("let restoredSession: FeedSessionSnapshot | null = null;");
  const hydrationEnd = source.indexOf("materialIdsForFeedRef.current", hydrationStart);
  const hydrationText = source.slice(hydrationStart, hydrationEnd);
  assert.match(hydrationText, /const allSessions = parseFeedSessions\(/);
  assert.match(hydrationText, /restoredSession = allSessions\[materialId\] \?\? null/);
});

test("feed snapshot persistence always stamps the current selection contract", () => {
  const writes = [];
  const currentContract = "quality_silence_v35";
  const persistFeedSessionSnapshot = compileFunctionDeclaration("persistFeedSessionSnapshot", {
    window: { localStorage: { getItem: () => null } },
    FEED_SESSION_STORAGE_KEY: "studyreels-feed-sessions",
    CURRENT_SELECTION_CONTRACT_VERSION: currentContract,
    MAX_SAVED_FEED_SESSIONS: 24,
    parseFeedSessions: () => ({}),
    compactFeedSessionSnapshot: (snapshot) => snapshot,
    safeLocalStorageSetItem: (key, value) => {
      writes.push({ key, value });
      return true;
    },
  });

  persistFeedSessionSnapshot("material-a", {
    selectionContractVersion: "quality_silence_v32",
    reels: [],
    feedbackByReel: {},
    adaptiveExcludeReelIds: [],
    page: 1,
    total: 0,
    canRequestMore: true,
    generationMode: "slow",
    mutedPreference: true,
    autoplayEnabled: true,
    playbackRate: 1,
    activeIndex: 0,
    updatedAt: 1,
  });

  assert.equal(writes.length, 1);
  assert.equal(writes[0].key, "studyreels-feed-sessions");
  const stored = JSON.parse(writes[0].value);
  assert.equal(stored["material-a"].selectionContractVersion, currentContract);
});

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
      ts.isCallExpression(node)
      && node.expression.getText(sourceFile) === "reconcileGeneratedReels"
      && node.arguments[2]?.getText(sourceFile) === "{ preserveUnmatchedUnseen: true }"
    ) {
      let statement = node;
      while (statement.parent && !ts.isBlock(statement.parent)) {
        statement = statement.parent;
      }
      finalReconciliations.push({ call: node, statement });
    }
    ts.forEachChild(node, visit);
  }

  visit(sourceFile);
  assert.equal(finalReconciliations.length, 2, "every generation path must remain covered");
  for (const reconciliation of finalReconciliations) {
    const reconciliationCall = reconciliation.call;
    assert.equal(
      reconciliationCall.arguments[2]?.getText(sourceFile),
      "{ preserveUnmatchedUnseen: true }",
      "stream settlement must explicitly retain unmatched provisional rows",
    );
    const block = reconciliation.statement.parent;
    assert.ok(ts.isBlock(block), "final reconciliation must remain in a guarded block");
    const reconciliationIndex = block.statements.indexOf(reconciliation.statement);
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

test("feed starts with three three-clip batches and resumes durable jobs", () => {
  assert.match(source, /const REEL_BATCH_SIZE = 3;/);
  assert.match(source, /const INITIAL_READY_BATCH_COUNT = 3;/);
  assert.match(source, /const INITIAL_READY_REEL_TARGET = REEL_BATCH_SIZE \* INITIAL_READY_BATCH_COUNT;/);
  assert.match(source, /const MAX_ZERO_GROWTH_CONTINUATIONS = 3;/);
  assert.match(source, /const MAX_GENERATION_ATTEMPTS_PER_FILL = 5;/);
  assert.match(source, /const PAGE_SIZE = INITIAL_READY_REEL_TARGET;/);
  assert.match(source, /return INITIAL_READY_REEL_TARGET;/);
  assert.match(source, /prefetch: readyReservoirTarget\(requestGenerationMode\)/);
  assert.match(source, /numReels: continuationToken/);
  assert.match(source, /existingMaterialReels \+ batchSize/);
  assert.match(source, /initialFill: true/);
  assert.match(source, /rememberFeedContinuationToken\(row\.materialId, row\.data!\)/);
  assert.match(source, /rememberFeedGenerationJob\(row\.materialId, row\.data!\)/);
  assert.match(source, /generationJobId: activeGenerationJob\?\.jobId/);
  assert.match(source, /continuationToken,/);
  assert.match(source, /idleTimeoutMs: GENERATION_STREAM_IDLE_TIMEOUT_MS/);
  assert.match(source, /consecutiveIdleWindows >= 2/);
  const requestMoreStart = source.indexOf("const requestMore = useCallback(");
  const requestMoreEnd = source.indexOf("\n\n  useEffect(() => {", requestMoreStart);
  const requestMoreText = source.slice(requestMoreStart, requestMoreEnd);
  assert.doesNotMatch(requestMoreText, /excludeVideoIds/);
  assert.doesNotMatch(requestMoreText, /markGenerationFinished/);
});

test("feed-owned jobs start the shared stream immediately and only once per material", async () => {
  const generationConsumerByMaterialRef = { current: new Map() };
  const generationBatchTokensRef = { current: new Set() };
  const isGeneratingRef = { current: false };
  const streamCalls = [];
  const appended = [];
  const reconciled = [];
  const settled = [];
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
      return {
        reels: [{ reel_id: `final-${params.materialId}` }],
        batch_id: `job-${params.materialId}`,
        batch_size: 1,
        continuation_token: `job-${params.materialId}`,
        terminal_status: "completed",
      };
    },
    isSearchScopeActive: () => true,
    noteGenerationTerminal: () => {},
    settleGenerationContinuation: (...args) => settled.push(args),
    appendGeneratedReels: (rows) => {
      appended.push(...rows);
      return { reels: rows, addedReels: rows, addedCount: rows.length, updatedCount: 0 };
    },
    markRecoveryProgress: () => {},
    clearPendingTailAdvance: () => {},
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
  assert.deepEqual(settled.map(([id, data]) => [id, data.continuation_token]), [
    ["material-a", "job-material-a"],
    ["material-b", "job-material-b"],
  ]);
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

test("feed-owned final inventory reconciles when no candidate event arrived", async () => {
  const generationConsumerByMaterialRef = { current: new Map() };
  const generationBatchTokensRef = { current: new Set() };
  const isGeneratingRef = { current: false };
  const reconciled = [];
  const syncGenerationLockState = compileUseCallback("syncGenerationLockState", {
    generationConsumerByMaterialRef,
    generationBatchTokensRef,
    isGeneratingRef,
    setGeneratingMore: () => {},
    setRecoveryPhase: () => {},
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
      params.onTerminal("partial");
      return {
        reels: [
          { reel_id: "final-a" },
          { reel_id: "final-b" },
        ],
        batch_id: "job-a",
        batch_size: 2,
        continuation_token: "job-a",
        terminal_status: "partial",
      };
    },
    isSearchScopeActive: () => true,
    noteGenerationTerminal: () => {},
    settleGenerationContinuation: () => {},
    appendGeneratedReels: () => {
      throw new Error("no provisional candidate should be required");
    },
    markRecoveryProgress: () => {},
    clearPendingTailAdvance: () => {},
    reconcileGeneratedReels: (...args) => {
      reconciled.push(args);
      return { reels: args[1], addedReels: args[1], addedCount: args[1].length, updatedCount: 0 };
    },
    isRequestInterruptedError: () => false,
    console: { warn: () => {} },
  });
  const searchScope = { key: "material", seq: 1, controller: new AbortController() };

  await callback(
    "material-a",
    { generation_job_id: "job-a", generation_job_status: "running" },
    searchScope,
  );

  assert.equal(reconciled.length, 1);
  assert.deepEqual(reconciled[0][0], []);
  assert.deepEqual(reconciled[0][1].map((reel) => reel.reel_id), ["final-a", "final-b"]);
  assert.deepEqual(reconciled[0][2], { preserveUnmatchedUnseen: true });
  assert.equal(isGeneratingRef.current, false);
});

test("a recovered generation stream failure clears pending tail advance", async () => {
  const consumers = new Map();
  const isGeneratingRef = { current: false };
  let clearedTailAdvance = 0;
  const callback = compileUseCallback("consumeFeedGenerationJob", {
    claimGenerationConsumer: (materialId, jobId) => {
      const token = Symbol(jobId);
      consumers.set(materialId, token);
      isGeneratingRef.current = true;
      return token;
    },
    releaseGenerationConsumer: (materialId, token) => {
      if (consumers.get(materialId) === token) {
        consumers.delete(materialId);
      }
      isGeneratingRef.current = consumers.size > 0;
    },
    isGeneratingRef,
    clearPendingTailAdvance: () => {
      clearedTailAdvance += 1;
    },
    GENERATION_STREAM_IDLE_TIMEOUT_MS: 35_000,
    generateReelsStream: async () => {
      throw new Error("generation stream idle timeout");
    },
    isSearchScopeActive: () => true,
    isRequestInterruptedError: () => false,
    noteGenerationTerminal: () => {},
    settleGenerationContinuation: () => {},
    appendGeneratedReels: () => ({ reels: [], addedReels: [], addedCount: 0, updatedCount: 0 }),
    markRecoveryProgress: () => {},
    reconcileGeneratedReels: () => ({ reels: [], addedReels: [], addedCount: 0, updatedCount: 0 }),
    console: { warn: () => {} },
  });

  await callback(
    "material-a",
    { generation_job_id: "recovered-job", generation_job_status: "running" },
    { key: "material", seq: 1, controller: new AbortController() },
  );

  assert.equal(consumers.size, 0);
  assert.equal(isGeneratingRef.current, false);
  assert.equal(clearedTailAdvance, 1);
});

test("failed empty generation reattaches through the authoritative feed", async () => {
  const searchScope = { key: "material", seq: 1, controller: new AbortController() };
  const isGeneratingRef = { current: false };
  const reelsRef = { current: [] };
  const generationBatchTokensRef = { current: new Set() };
  const generationConsumerByMaterialRef = { current: new Map() };
  const loadCalls = [];
  const surfacedErrors = [];
  let clearedTailAdvance = 0;
  let generationProgress = null;

  const callback = compileUseCallback("requestMore", {
    getFeedMaterialIds: () => ["material-a"],
    settingsScopeReady: true,
    isGeneratingRef,
    canRequestMore: true,
    isIngestMaterial: false,
    setCanRequestMore: () => {},
    setFeedPagesExhausted: () => {},
    isGenerationFinished: () => false,
    activeSearchScopeRef: { current: searchScope },
    getFeedTuningSettings: () => ({
      minRelevance: 0.3,
      creativeCommonsOnly: false,
      preferredVideoDuration: "any",
    }),
    reelsRef,
    activeIndexRef: { current: 0 },
    readyReservoirTarget: () => 8,
    REEL_BATCH_SIZE: 3,
    INITIAL_READY_REEL_TARGET: 9,
    generationMode: "fast",
    generationBatchTokensRef,
    syncGenerationLockState: () => {
      isGeneratingRef.current = generationBatchTokensRef.current.size > 0
        || generationConsumerByMaterialRef.current.size > 0;
    },
    armActiveRecoveryRequest: () => {},
    setVisibleFeedError: () => {},
    progressClearTimerRef: { current: null },
    setGenerationProgress: (next) => {
      generationProgress = typeof next === "function" ? next(generationProgress) : next;
    },
    countReelsForMaterial: () => 0,
    generationJobByMaterialRef: { current: new Map() },
    continuationTokenByMaterialRef: { current: new Map() },
    claimGenerationConsumer: () => Symbol("consumer"),
    GENERATION_STREAM_IDLE_TIMEOUT_MS: 35_000,
    generateReelsStream: async () => {
      throw new Error("Request timed out before the backend response completed.");
    },
    isSearchScopeActive: (scope) => scope === searchScope,
    noteGenerationTerminal: () => {},
    settleGenerationContinuation: () => {},
    appendGeneratedReels: () => ({ reels: [], addedReels: [], addedCount: 0, updatedCount: 0 }),
    dedupeByIdentity: (rows) => rows,
    reconcileGeneratedReels: () => {},
    isRequestInterruptedError: () => false,
    releaseGenerationConsumer: () => {},
    materialId: "material-a",
    recoveryAttemptedIdsRef: { current: new Set() },
    recoverMissingMaterial: async () => false,
    mergeReelBatchesInServerOrder: (batches) => batches.flat(),
    generationConsumerByMaterialRef,
    loadPage: async (targetPage, options) => {
      loadCalls.push({ targetPage, options });
      generationConsumerByMaterialRef.current.set("material-a", { jobId: "durable-job" });
      return { addedCount: 0, exhausted: false };
    },
    markRecoveryProgress: () => {},
    clearPendingTailAdvance: () => {
      clearedTailAdvance += 1;
    },
    isTransportError: () => true,
    noteFeedTransportFailure: (error) => surfacedErrors.push(error),
    noteFeedFailure: (error) => surfacedErrors.push(error),
    setRecoveryPhase: () => {},
    clearRecoveredTransportError: () => {},
    finishActiveRecoveryRequest: () => {},
    console: { warn: () => {} },
  });

  assert.deepEqual(await callback(), []);
  assert.deepEqual(loadCalls, [{
    targetPage: 1,
    options: {
      autofill: true,
      preserveSession: true,
      generationMode: "fast",
    },
  }]);
  assert.equal(generationConsumerByMaterialRef.current.has("material-a"), true);
  assert.deepEqual(surfacedErrors, [], "the recovered durable job must retain the loading state");
  assert.equal(isGeneratingRef.current, true);
  assert.equal(generationProgress, null, "the recovered consumer owns loading state after reattachment");
  assert.equal(clearedTailAdvance, 0, "an attached durable job must retain pending tail advance");
});

test("background zero-growth partial keeps an existing reel playable without a fatal error", async () => {
  const searchScope = { key: "material", seq: 1, controller: new AbortController() };
  const isGeneratingRef = { current: false };
  const reelsRef = {
    current: [{ reel_id: "verified-a", material_id: "material-a" }],
  };
  const generationBatchTokensRef = { current: new Set() };
  const generationConsumerByMaterialRef = { current: new Map() };
  const surfacedErrors = [];
  const callback = compileUseCallback("requestMore", {
    getFeedMaterialIds: () => ["material-a"],
    settingsScopeReady: true,
    isGeneratingRef,
    canRequestMore: true,
    isIngestMaterial: false,
    setCanRequestMore: () => {},
    setFeedPagesExhausted: () => {},
    isGenerationFinished: () => false,
    activeSearchScopeRef: { current: searchScope },
    getFeedTuningSettings: () => ({
      minRelevance: 0.3,
      creativeCommonsOnly: false,
      preferredVideoDuration: "any",
    }),
    reelsRef,
    REEL_BATCH_SIZE: 3,
    INITIAL_READY_REEL_TARGET: 9,
    generationMode: "fast",
    generationBatchTokensRef,
    syncGenerationLockState: () => {},
    armActiveRecoveryRequest: () => {},
    setVisibleFeedError: () => {},
    progressClearTimerRef: { current: null },
    setGenerationProgress: () => {},
    generationJobByMaterialRef: { current: new Map() },
    continuationTokenByMaterialRef: { current: new Map([["material-a", "previous-job"]]) },
    claimGenerationConsumer: () => Symbol("consumer"),
    GENERATION_STREAM_IDLE_TIMEOUT_MS: 35_000,
    generateReelsStream: async () => ({
      reels: [],
      batch_id: "next-job",
      batch_size: 0,
      continuation_token: "next-job",
      terminal_status: "partial",
    }),
    isSearchScopeActive: (scope) => scope === searchScope,
    noteGenerationTerminal: () => {},
    settleGenerationContinuation: () => {},
    appendGeneratedReels: () => ({ reels: reelsRef.current, addedReels: [], addedCount: 0, updatedCount: 0 }),
    dedupeByIdentity: (rows) => rows,
    reconcileGeneratedReels: () => ({ reels: reelsRef.current, addedReels: [], addedCount: 0, updatedCount: 0 }),
    isRequestInterruptedError: () => false,
    releaseGenerationConsumer: () => {},
    materialId: "material-a",
    recoveryAttemptedIdsRef: { current: new Set() },
    recoverMissingMaterial: async () => false,
    mergeReelBatchesInServerOrder: (batches) => batches.flat(),
    generationConsumerByMaterialRef,
    loadPage: async () => ({ addedCount: 0, exhausted: false }),
    markRecoveryProgress: () => {},
    clearPendingTailAdvance: () => {},
    isTransportError: () => false,
    noteFeedTransportFailure: (error) => surfacedErrors.push(error),
    noteFeedFailure: (error) => surfacedErrors.push(error),
    setRecoveryPhase: () => {},
    clearRecoveredTransportError: () => {},
    finishActiveRecoveryRequest: () => {},
    setTimeout: () => 1,
    console: { warn: () => {} },
  });

  assert.deepEqual(await callback({ surfaceError: false, requestedCount: 3 }), []);
  assert.deepEqual(surfacedErrors, []);

  assert.deepEqual(await callback({ surfaceError: true, requestedCount: 3 }), []);
  assert.deepEqual(surfacedErrors, ["No new clips arrived. Retry to continue searching."]);
  assert.equal(reelsRef.current.length, 1);
});

test("feed failure copy preserves an explicit string message", () => {
  const messages = [];
  const callback = compileUseCallback("noteFeedFailure", {
    transportFailureStreakRef: { current: 2 },
    setVisibleFeedError: (message) => messages.push(message),
  });

  callback("No new clips arrived. Retry to continue searching.");

  assert.deepEqual(messages, ["No new clips arrived. Retry to continue searching."]);
});

test("sequential terminal batches advance continuation and preserve a same-video sibling", async () => {
  const searchScope = { key: "material", seq: 1, controller: new AbortController() };
  const generationBatchTokensRef = { current: new Set() };
  const observedTerminals = [];
  const reconciled = [];
  const submittedParams = [];
  const continuationTokenByMaterialRef = { current: new Map() };
  const lastTerminalStatusByMaterialRef = { current: new Map() };
  const noteGenerationTerminal = (materialId, status) => observedTerminals.push({ materialId, status });
  const settleGenerationContinuation = compileUseCallback("settleGenerationContinuation", {
    continuationTokenByMaterialRef,
    lastTerminalStatusByMaterialRef,
    noteGenerationTerminal,
  });
  let batchNumber = 0;
  const callback = compileUseCallback("requestMore", {
    getFeedMaterialIds: () => ["material-a"],
    settingsScopeReady: true,
    isGeneratingRef: { current: false },
    canRequestMore: true,
    isIngestMaterial: false,
    setCanRequestMore: () => {},
    setFeedPagesExhausted: () => {},
    isGenerationFinished: () => false,
    activeSearchScopeRef: { current: searchScope },
    getFeedTuningSettings: () => ({
      minRelevance: 0.3,
      creativeCommonsOnly: false,
      preferredVideoDuration: "any",
    }),
    reelsRef: {
      current: [{
        reel_id: "existing",
        material_id: "material-a",
        video_id: "source-a",
        video_url: "https://youtube.com/watch?v=source-a",
        t_start: 0,
        t_end: 10,
      }],
    },
    activeIndexRef: { current: 0 },
    readyReservoirTarget: () => 8,
    REEL_BATCH_SIZE: 3,
    INITIAL_READY_REEL_TARGET: 9,
    generationMode: "fast",
    generationBatchTokensRef,
    syncGenerationLockState: () => {},
    armActiveRecoveryRequest: () => {},
    setVisibleFeedError: () => {},
    progressClearTimerRef: { current: null },
    setGenerationProgress: () => {},
    countReelsForMaterial: () => 0,
    generationJobByMaterialRef: { current: new Map() },
    continuationTokenByMaterialRef,
    claimGenerationConsumer: () => Symbol("consumer"),
    GENERATION_STREAM_IDLE_TIMEOUT_MS: 35_000,
    generateReelsStream: async (params) => {
      submittedParams.push(params);
      batchNumber += 1;
      return {
        reels: [{
          reel_id: `sibling-${batchNumber}`,
          material_id: "material-a",
          video_id: "source-a",
          video_url: "https://youtube.com/watch?v=source-a",
          t_start: batchNumber * 10,
          t_end: batchNumber * 10 + 10,
        }],
        batch_id: `job-${batchNumber}`,
        batch_size: 1,
        continuation_token: `job-${batchNumber}`,
        terminal_status: "completed",
      };
    },
    isSearchScopeActive: (scope) => scope === searchScope,
    noteGenerationTerminal,
    settleGenerationContinuation,
    appendGeneratedReels: () => {
      throw new Error("a direct cache hit has no candidate events");
    },
    dedupeByIdentity: (rows) => rows,
    reconcileGeneratedReels: (...args) => {
      reconciled.push(args);
      return { reels: args[1], addedReels: args[1], addedCount: args[1].length, updatedCount: 0 };
    },
    isRequestInterruptedError: () => false,
    releaseGenerationConsumer: () => {},
    materialId: "material-a",
    mergeReelBatchesInServerOrder: (batches) => batches.flat(),
    markRecoveryProgress: () => {},
    clearPendingTailAdvance: () => {},
    clearRecoveredTransportError: () => {},
    setRecoveryPhase: () => {},
    finishActiveRecoveryRequest: () => {},
    setTimeout: () => 1,
    console: { warn: () => {} },
  });

  const first = await callback();
  const second = await callback();

  assert.deepEqual(first.map((reel) => reel.reel_id), ["sibling-1"]);
  assert.deepEqual(second.map((reel) => reel.reel_id), ["sibling-2"]);
  assert.equal(submittedParams[0].continuationToken, undefined);
  assert.equal(submittedParams[1].continuationToken, "job-1");
  assert.equal(submittedParams[0].numReels, 4);
  assert.equal(submittedParams[1].numReels, 3);
  assert.equal(Object.hasOwn(submittedParams[0], "excludeVideoIds"), false);
  assert.equal(continuationTokenByMaterialRef.current.get("material-a"), "job-2");
  assert.deepEqual(observedTerminals, [
    { materialId: "material-a", status: "completed" },
    { materialId: "material-a", status: "completed" },
  ]);
  assert.equal(reconciled.length, 2);
});

test("only an authoritative exhausted terminal stops continuation", async () => {
  const generationJobByMaterialRef = {
    current: new Map([
      ["completed-material", { jobId: "completed-job" }],
      ["partial-material", { jobId: "partial-job" }],
      ["exhausted-material", { jobId: "exhausted-job" }],
      ["failed-material", { jobId: "failed-job" }],
      ["cancelled-material", { jobId: "cancelled-job" }],
    ]),
  };
  const generationFinishedRef = { current: new Set() };
  const canRequestMoreWrites = [];
  const lastTerminalStatusByMaterialRef = { current: new Map() };
  const isGenerationFinished = compileUseCallback("isGenerationFinished", {
    generationFinishedRef,
  });
  const markGenerationFinished = compileUseCallback("markGenerationFinished", {
    generationFinishedRef,
    getFeedMaterialIds: () => [
      "completed-material",
      "partial-material",
      "exhausted-material",
    ],
    isGenerationFinished,
    setCanRequestMore: (value) => canRequestMoreWrites.push(value),
  });
  const callback = compileUseCallback("noteGenerationTerminal", {
    generationJobByMaterialRef,
    lastTerminalStatusByMaterialRef,
    markGenerationFinished,
  });

  callback("completed-material", "completed");
  callback("partial-material", "partial");
  callback("exhausted-material", "exhausted");
  callback("failed-material", "failed");
  callback("cancelled-material", "cancelled");

  assert.equal(generationFinishedRef.current.has("completed-material"), false);
  assert.equal(generationFinishedRef.current.has("partial-material"), false);
  assert.equal(generationFinishedRef.current.has("exhausted-material"), true);
  assert.deepEqual(canRequestMoreWrites, []);
  assert.equal(generationJobByMaterialRef.current.size, 0);
  assert.equal(isGenerationFinished("failed-material"), false);
  assert.equal(lastTerminalStatusByMaterialRef.current.get("failed-material"), "failed");
  assert.equal(isGenerationFinished("cancelled-material"), false);
  assert.deepEqual(
    ["failed-material", "cancelled-material"].filter((id) => !isGenerationFinished(id)),
    ["failed-material", "cancelled-material"],
    "failed and cancelled jobs must remain eligible for an explicit retry",
  );

  assert.equal(isGenerationFinished("completed-material"), false);
  assert.equal(isGenerationFinished("partial-material"), false);
  assert.equal(isGenerationFinished("exhausted-material"), true);
  assert.doesNotMatch(source, /runFastTopUp/, "web must not eagerly quota-fill before the viewer reaches the tail");
});

test("terminal copy is hidden until the viewer reaches the last visible reel", () => {
  assert.match(
    source,
    /const noMoreReelsAvailable\s*=\s*atEndOfVisibleReels\s*&&\s*!hasMore\s*&&\s*!canRequestMore/,
  );
});

test("generation continuation is server-owned and does not exclude whole source videos", () => {
  const requestMoreStart = source.indexOf("const requestMore = useCallback(");
  const requestMoreEnd = source.indexOf("\n\n  useEffect(() => {", requestMoreStart);
  const callbackText = source.slice(requestMoreStart, requestMoreEnd);
  assert.match(callbackText, /const continuationToken = continuationTokenByMaterialRef\.current\.get\(id\)/);
  assert.match(callbackText, /continuationToken,/);
  assert.doesNotMatch(callbackText, /excludeVideoIds/);
});

test("a restored feed response seeds the next continuation token", () => {
  const continuationTokenByMaterialRef = { current: new Map([["material-a", "stale-token"]]) };
  const rememberFeedContinuationToken = compileUseCallback("rememberFeedContinuationToken", {
    continuationTokenByMaterialRef,
  });
  rememberFeedContinuationToken("material-a", { continuation_token: " restored-job " });
  assert.equal(continuationTokenByMaterialRef.current.get("material-a"), "restored-job");

  const loadPageStart = source.indexOf("const loadPage = useCallback(");
  const loadPageEnd = source.indexOf("const requestMore = useCallback(", loadPageStart);
  const loadPageText = source.slice(loadPageStart, loadPageEnd);
  const seedIndex = loadPageText.indexOf("rememberFeedContinuationToken(row.materialId, row.data!)");
  const consumeIndex = loadPageText.indexOf("void consumeFeedGenerationJob(");
  assert.ok(seedIndex >= 0, "page-one reload must seed the backend continuation token");
  assert.ok(consumeIndex > seedIndex, "the continuation token must be seeded before resuming its durable job");
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
  let renderedRows = currentRows;
  const callback = compileUseCallback("reconcileGeneratedReels", {
    reelsRef,
    activeIndexRef,
    watchedFrontierIndexRef,
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

test("generation settlement freezes the watched frontier and preserves authoritative unseen order", () => {
  const currentRows = [
    { reel_id: "watched", video_url: "watched", difficulty: 0.9 },
    { reel_id: "backtracked-current", video_url: "backtracked-current", difficulty: 0.8 },
    { reel_id: "watched-frontier", video_url: "watched-frontier", difficulty: 0.7 },
    { reel_id: "rejected-provisional", video_url: "rejected-provisional", difficulty: 0.2 },
  ];
  const reelsRef = { current: currentRows };
  const activeIndexRef = { current: 1 };
  const watchedFrontierIndexRef = { current: 2 };
  let renderedRows = currentRows;
  const callback = compileUseCallback("reconcileGeneratedReels", {
    reelsRef,
    activeIndexRef,
    watchedFrontierIndexRef,
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
    "hard",
    "tie-first",
    "easy",
    "tie-second",
  ]);
  assert.equal(renderedRows[2].video_title, "Updated");
});

test("grouped material batches preserve each authoritative server response order", () => {
  const mergeReelBatchesInServerOrder = compileUseCallback("mergeReelBatchesInServerOrder", {});

  const merged = mergeReelBatchesInServerOrder([
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
    "hard",
    "tie-first",
    "easy",
    "tie-second",
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
    /const allowServerAutofill = \(options\?\.autofill \?\? true\) && feedMaterialIds\.length === 1/,
  );
  assert.match(loadPageText, /autofill: allowServerAutofill/);
  assert.match(
    loadPageText,
    /reconcileGeneratedReels\(\[\], fetchedReels, \{ preserveUnmatchedUnseen: false \}\)/,
  );

  const hydrationStart = source.indexOf("let restoredSession: FeedSessionSnapshot | null = null;");
  const hydrationEnd = source.indexOf("useEffect(() => {", hydrationStart);
  assert.ok(hydrationStart >= 0 && hydrationEnd > hydrationStart);
  const hydrationText = source.slice(hydrationStart, hydrationEnd);
  assert.match(hydrationText, /setBootstrappingFirstReels\(true\);/);
  assert.match(hydrationText, /autofill: true,/);
  assert.match(hydrationText, /generationMode: restoredGenerationMode/);
  assert.doesNotMatch(hydrationText, /setCanRequestMore\(restoredSession\.canRequestMore\)/);
  assert.match(hydrationText, /current backend response[\s\S]*?setCanRequestMore\(true\)/);
  assert.match(hydrationText, /hydratedMaterialIdRef\.current === materialId/);
  assert.match(hydrationText, /isSearchScopeActive\(restoredSearchScope\)/);
});

test("a current backend job re-enables a stale snapshot search scope", () => {
  const generationJobByMaterialRef = { current: new Map() };
  const canRequestMoreWrites = [];
  const callback = compileUseCallback("rememberFeedGenerationJob", {
    generationJobByMaterialRef,
    setCanRequestMore: (value) => canRequestMoreWrites.push(value),
    noteGenerationTerminal: () => {
      throw new Error("an active job is not terminal");
    },
  });

  callback("material-a", {
    generation_job_id: "fresh-job",
    generation_job_status: "running",
  });

  assert.deepEqual(canRequestMoreWrites, [true]);
  assert.deepEqual(generationJobByMaterialRef.current.get("material-a"), {
    jobId: "fresh-job",
    status: "running",
  });
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
    INITIAL_READY_REEL_TARGET: 9,
    MAX_GENERATION_ATTEMPTS_PER_FILL: 5,
    MAX_ZERO_GROWTH_CONTINUATIONS: 3,
    REEL_BATCH_SIZE: 3,
    getFeedMaterialIds: () => ["material"],
    isGenerationFinished: () => false,
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
  });

  await callback();

  assert.deepEqual(pageLoads.map((row) => row.targetPage), [2, 3]);
  assert.ok(pageLoads.every((row) => row.options.autofill === false));
  assert.equal(generationRequests, 0, "persisted inventory filled the reservoir");
});

test("bootstrap continues partial initial inventory in bounded batches until nine are ready", async () => {
  const reelsRef = { current: [{ reel_id: "useful-a" }, { reel_id: "useful-b" }, { reel_id: "useful-c" }] };
  const searchScope = { key: "material", seq: 1 };
  const generationRequests = [];
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
    hasMore: false,
    total: 3,
    PAGE_SIZE: 9,
    reelsRef,
    activeIndexRef: { current: 0 },
    readyReservoirTarget: () => 9,
    INITIAL_READY_REEL_TARGET: 9,
    MAX_GENERATION_ATTEMPTS_PER_FILL: 5,
    MAX_ZERO_GROWTH_CONTINUATIONS: 3,
    REEL_BATCH_SIZE: 3,
    getFeedMaterialIds: () => ["material"],
    isGenerationFinished: () => false,
    generationMode: "fast",
    loadPage: async () => {
      throw new Error("an exhausted persisted inventory has no next page");
    },
    isSearchScopeActive: (scope) => scope === searchScope,
    requestMore: async (options) => {
      generationRequests.push(options);
      const nextIndex = reelsRef.current.length;
      reelsRef.current = [
        ...reelsRef.current,
        ...Array.from({ length: options.requestedCount }, (_, offset) => ({
          reel_id: `top-up-${nextIndex + offset}`,
        })),
      ];
      return reelsRef.current.slice(-options.requestedCount);
    },
  });

  await callback(false);

  assert.deepEqual(generationRequests.map((options) => options.requestedCount), [3, 3]);
  assert.equal(reelsRef.current.length, 9);
  const bootstrapEffectStart = source.indexOf("useEffect(() => {", source.indexOf("const bootstrapFirstReels"));
  const bootstrapEffectEnd = source.indexOf("const maybeLoadMore", bootstrapEffectStart);
  const bootstrapEffectText = source.slice(bootstrapEffectStart, bootstrapEffectEnd);
  assert.match(bootstrapEffectText, /startupReservoirComplete/);
  assert.match(bootstrapEffectText, /\|\| generatingMore/);
});

test("startup bounds repeated zero-growth partial continuations without discarding its usable reel", async () => {
  const reelsRef = { current: [{ reel_id: "strict-cached" }] };
  const searchScope = { key: "material", seq: 1 };
  let generationRequests = 0;
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
    hasMore: false,
    total: 1,
    PAGE_SIZE: 9,
    reelsRef,
    activeIndexRef: { current: 0 },
    readyReservoirTarget: () => 9,
    INITIAL_READY_REEL_TARGET: 9,
    MAX_GENERATION_ATTEMPTS_PER_FILL: 5,
    MAX_ZERO_GROWTH_CONTINUATIONS: 3,
    REEL_BATCH_SIZE: 3,
    getFeedMaterialIds: () => ["material"],
    isGenerationFinished: () => false,
    lastTerminalStatusByMaterialRef: { current: new Map([["material", "partial"]]) },
    generationMode: "fast",
    loadPage: async () => {
      throw new Error("there are no persisted pages");
    },
    isSearchScopeActive: (scope) => scope === searchScope,
    requestMore: async () => {
      generationRequests += 1;
      return [];
    },
  });

  await callback(false);

  assert.equal(generationRequests, 4);
  assert.equal(reelsRef.current.length, 1);
});

test("startup crosses zero-growth partials and keeps searching until nine reels are ready", async () => {
  const reelsRef = { current: [{ reel_id: "useful-initial" }] };
  const searchScope = { key: "material", seq: 1 };
  const statuses = new Map([["material", "partial"]]);
  const additionsByRequest = [0, 0, 3, 3, 2];
  const requestedCounts = [];
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
    hasMore: false,
    total: 1,
    PAGE_SIZE: 9,
    reelsRef,
    activeIndexRef: { current: 0 },
    readyReservoirTarget: () => 9,
    INITIAL_READY_REEL_TARGET: 9,
    MAX_GENERATION_ATTEMPTS_PER_FILL: 5,
    MAX_ZERO_GROWTH_CONTINUATIONS: 3,
    REEL_BATCH_SIZE: 3,
    getFeedMaterialIds: () => ["material"],
    isGenerationFinished: () => false,
    lastTerminalStatusByMaterialRef: { current: statuses },
    generationMode: "fast",
    loadPage: async () => {
      throw new Error("there are no persisted pages");
    },
    isSearchScopeActive: (scope) => scope === searchScope,
    requestMore: async (options) => {
      requestedCounts.push(options.requestedCount);
      const addedCount = additionsByRequest.shift() ?? 0;
      const currentCount = reelsRef.current.length;
      reelsRef.current = [
        ...reelsRef.current,
        ...Array.from({ length: addedCount }, (_, index) => ({
          reel_id: `fresh-${currentCount + index}`,
        })),
      ];
      return [];
    },
  });

  await callback(false);

  assert.deepEqual(requestedCounts, [3, 3, 3, 3, 2]);
  assert.equal(reelsRef.current.length, 9);
});

test("startup stops immediately after an authoritative exhausted continuation", async () => {
  const reelsRef = { current: [{ reel_id: "only-verified-reel" }] };
  const searchScope = { key: "material", seq: 1 };
  const statuses = new Map([["material", "partial"]]);
  let generationRequests = 0;
  let exhausted = false;
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
    hasMore: false,
    total: 1,
    PAGE_SIZE: 9,
    reelsRef,
    activeIndexRef: { current: 0 },
    readyReservoirTarget: () => 9,
    INITIAL_READY_REEL_TARGET: 9,
    MAX_GENERATION_ATTEMPTS_PER_FILL: 5,
    MAX_ZERO_GROWTH_CONTINUATIONS: 3,
    REEL_BATCH_SIZE: 3,
    getFeedMaterialIds: () => ["material"],
    isGenerationFinished: () => exhausted,
    lastTerminalStatusByMaterialRef: { current: statuses },
    generationMode: "fast",
    loadPage: async () => {
      throw new Error("there are no persisted pages");
    },
    isSearchScopeActive: (scope) => scope === searchScope,
    requestMore: async () => {
      generationRequests += 1;
      statuses.set("material", "exhausted");
      exhausted = true;
      return [];
    },
  });

  await callback(false);

  assert.equal(generationRequests, 1);
  assert.equal(reelsRef.current.length, 1);
});

test("persisted paging fills first, then an empty or short page falls through to the rolling batch", async () => {
  const pageLoads = [];
  const generationRequests = [];
  const isGeneratingRef = { current: false };
  const reelsRef = { current: Array.from({ length: 9 }, (_, index) => ({ reel_id: String(index) })) };
  const shared = {
    isIngestMaterial: false,
    isFetchingRef: { current: false },
    page: 1,
    reelsRef,
    canRequestMore: true,
    isGeneratingRef,
    REEL_BATCH_SIZE: 3,
    requestReadyBatch: async (requestedCount = 3) => {
      if (isGeneratingRef.current) {
        return 0;
      }
      isGeneratingRef.current = true;
      generationRequests.push(requestedCount);
      return 0;
    },
  };

  const pagePersistedRows = compileUseCallback("maybeLoadMore", {
    ...shared,
    hasMore: true,
    feedNeedsBootstrapTopUp: () => true,
    loadPage: async (targetPage, options) => {
      pageLoads.push({ targetPage, options });
      reelsRef.current = [
        ...reelsRef.current,
        { reel_id: "cached-9" },
        { reel_id: "cached-10" },
        { reel_id: "cached-11" },
      ];
      return { addedCount: 3, exhausted: false };
    },
  });
  await pagePersistedRows();
  assert.deepEqual(pageLoads, [{ targetPage: 2, options: { autofill: false } }]);
  assert.deepEqual(generationRequests, []);

  reelsRef.current = reelsRef.current.slice(0, 9);
  const emptyPersistedPage = compileUseCallback("maybeLoadMore", {
    ...shared,
    hasMore: true,
    feedNeedsBootstrapTopUp: () => true,
    loadPage: async () => ({ addedCount: 0, exhausted: true }),
  });
  await emptyPersistedPage();
  assert.deepEqual(generationRequests, [3], "an empty persisted page must fall through immediately");

  isGeneratingRef.current = false;
  reelsRef.current = reelsRef.current.slice(0, 9);
  const shortPersistedPage = compileUseCallback("maybeLoadMore", {
    ...shared,
    hasMore: true,
    feedNeedsBootstrapTopUp: () => true,
    loadPage: async () => {
      reelsRef.current = [...reelsRef.current, { reel_id: "one-cached-reel" }];
      return { addedCount: 1, exhausted: true };
    },
  });
  await shortPersistedPage();
  assert.deepEqual(generationRequests, [3, 2], "one cached reel leaves a two-reel logical batch shortfall");

  isGeneratingRef.current = false;
  const aboveThreshold = compileUseCallback("maybeLoadMore", {
    ...shared,
    hasMore: false,
    loadPage: async () => ({ addedCount: 0, exhausted: true }),
    feedNeedsBootstrapTopUp: () => false,
  });
  await aboveThreshold();
  assert.deepEqual(generationRequests, [3, 2]);

  const atThreshold = compileUseCallback("maybeLoadMore", {
    ...shared,
    hasMore: false,
    loadPage: async () => ({ addedCount: 0, exhausted: true }),
    feedNeedsBootstrapTopUp: () => true,
  });
  await Promise.all([atThreshold(), atThreshold()]);
  assert.deepEqual(generationRequests, [3, 2, 3], "an in-flight refill must not submit twice");
});

test("a one-reel partial continuation immediately fills the remaining two slots", async () => {
  const reelsRef = { current: Array.from({ length: 9 }, (_, index) => ({ reel_id: String(index) })) };
  const statuses = new Map();
  const requestedCounts = [];
  const callback = compileUseCallback("requestReadyBatch", {
    REEL_BATCH_SIZE: 3,
    MAX_ZERO_GROWTH_CONTINUATIONS: 3,
    MAX_GENERATION_ATTEMPTS_PER_FILL: 5,
    getFeedMaterialIds: () => ["material-a"],
    isGenerationFinished: () => false,
    isGeneratingRef: { current: false },
    reelsRef,
    lastTerminalStatusByMaterialRef: { current: statuses },
    requestMore: async ({ requestedCount }) => {
      requestedCounts.push(requestedCount);
      if (requestedCounts.length === 1) {
        reelsRef.current = [...reelsRef.current, { reel_id: "cached-sibling" }];
        statuses.set("material-a", "partial");
      } else {
        reelsRef.current = [
          ...reelsRef.current,
          { reel_id: "fresh-a" },
          { reel_id: "fresh-b" },
        ];
        statuses.set("material-a", "completed");
      }
      return [];
    },
  });

  assert.equal(await callback(), 3);
  assert.deepEqual(requestedCounts, [3, 2]);
  assert.equal(reelsRef.current.length, 12);
});

test("a ready-batch refill crosses two zero-growth partials before adding the next batch", async () => {
  const reelsRef = { current: Array.from({ length: 9 }, (_, index) => ({ reel_id: String(index) })) };
  const statuses = new Map([["material-a", "partial"]]);
  let generationRequests = 0;
  const callback = compileUseCallback("requestReadyBatch", {
    REEL_BATCH_SIZE: 3,
    MAX_ZERO_GROWTH_CONTINUATIONS: 3,
    MAX_GENERATION_ATTEMPTS_PER_FILL: 5,
    getFeedMaterialIds: () => ["material-a"],
    isGenerationFinished: () => false,
    isGeneratingRef: { current: false },
    reelsRef,
    lastTerminalStatusByMaterialRef: { current: statuses },
    requestMore: async () => {
      generationRequests += 1;
      if (generationRequests === 3) {
        reelsRef.current = [
          ...reelsRef.current,
          { reel_id: "fresh-a" },
          { reel_id: "fresh-b" },
          { reel_id: "fresh-c" },
        ];
      }
      return [];
    },
  });

  assert.equal(await callback(), 3);
  assert.equal(generationRequests, 3);
  assert.equal(reelsRef.current.length, 12);
});

test("a ready-batch refill bounds repeated zero-growth partials", async () => {
  const reelsRef = { current: Array.from({ length: 9 }, (_, index) => ({ reel_id: String(index) })) };
  let generationRequests = 0;
  const callback = compileUseCallback("requestReadyBatch", {
    REEL_BATCH_SIZE: 3,
    MAX_ZERO_GROWTH_CONTINUATIONS: 3,
    MAX_GENERATION_ATTEMPTS_PER_FILL: 5,
    getFeedMaterialIds: () => ["material-a"],
    isGenerationFinished: () => false,
    isGeneratingRef: { current: false },
    reelsRef,
    lastTerminalStatusByMaterialRef: { current: new Map([["material-a", "partial"]]) },
    requestMore: async () => {
      generationRequests += 1;
      return [];
    },
  });

  assert.equal(await callback(), 0);
  assert.equal(generationRequests, 4);
  assert.equal(reelsRef.current.length, 9);
});

test("ready-reservoir prefetch waits for batch two and keeps batch three buffered", () => {
  const shouldRefillReadyBuffer = compileFunctionDeclaration("shouldRefillReadyBuffer", {
    REEL_BATCH_SIZE: 3,
  });
  const reelsRef = { current: Array.from({ length: 9 }, (_, index) => ({ reel_id: String(index) })) };
  const activeIndexRef = { current: 0 };
  const callback = compileUseCallback("feedNeedsBootstrapTopUp", {
    getFeedMaterialIds: () => ["material-a"],
    reelsRef,
    activeIndexRef,
    shouldRefillReadyBuffer,
  });

  for (let index = 0; index <= 5; index += 1) {
    activeIndexRef.current = index;
    assert.equal(callback(), false, `index ${index} must not spend before batch two is complete`);
  }
  activeIndexRef.current = 6;
  assert.equal(callback(), true, "entering batch three must request exactly one replacement batch");
  activeIndexRef.current = 7;
  assert.equal(callback(), false, "the same buffered batch must not submit twice");
});

test("a settled duplicate-only batch clears the pending tail spinner and exposes retry", () => {
  const requestMoreStart = source.indexOf("const requestMore = useCallback(");
  const requestMoreEnd = source.indexOf("\n\n  useEffect(() => {", requestMoreStart);
  const callbackText = source.slice(requestMoreStart, requestMoreEnd);
  assert.match(callbackText, /const batchAddedReels = dedupeByIdentity\(\[\.\.\.streamedReels, \.\.\.reconciled\.addedReels\]\)/);
  assert.match(callbackText, /if \(generated\.length === 0\) \{[\s\S]*clearPendingTailAdvance\(\)/);
  assert.match(callbackText, /No new clips arrived\. Retry to continue searching\./);
  assert.match(source, /onClick=\{\(\) => void requestReadyBatch\(REEL_BATCH_SIZE, true\)\}/);
});

test("pagination remains available while background generation is active", () => {
  const maybeLoadMoreStart = source.indexOf("const maybeLoadMore = useCallback(async () => {");
  const maybeLoadMoreEnd = source.indexOf("const shouldBlockDownwardAtEnd", maybeLoadMoreStart);
  assert.ok(maybeLoadMoreStart >= 0 && maybeLoadMoreEnd > maybeLoadMoreStart);
  const callbackText = source.slice(maybeLoadMoreStart, maybeLoadMoreEnd);
  const pageFetchIndex = callbackText.indexOf("if (hasMore && !isFetchingRef.current)");
  const generationGateIndex = callbackText.indexOf("!isGeneratingRef.current");
  assert.ok(pageFetchIndex >= 0, "available feed pages must be fetched first");
  assert.ok(generationGateIndex > pageFetchIndex, "active generation must not block persisted page fetching");
});

test("authoritative finals lock the watched prefix and retain server order for the unseen tail", () => {
  assert.match(source, /Math\.max\(activeIndexRef\.current, watchedFrontierIndexRef\.current\) \+ 1/);
  assert.match(source, /const lockedPrefix = currentRows\.slice\(0, lockedPrefixLength\)/);
  assert.match(source, /const stableUnseenRows = currentRows\.slice\(lockedPrefixLength\)/);
  assert.match(source, /const reordered = dedupeByIdentity\(\[\.\.\.lockedPrefix, \.\.\.authoritativeTail\]\)/);
  assert.doesNotMatch(source, /orderReelsByDifficulty/);
  assert.match(source, /watchedFrontierIndex: dedupedReels\.length > 0/);
});

test("missing-material recovery preserves the level saved with the material seed", () => {
  const recoveryStart = source.indexOf("const recoverMissingMaterial = useCallback(");
  const recoveryEnd = source.indexOf("const feedNeedsBootstrapTopUp", recoveryStart);
  assert.ok(recoveryStart >= 0 && recoveryEnd > recoveryStart);
  const callbackText = source.slice(recoveryStart, recoveryEnd);
  assert.match(source, /knowledgeLevel\?: KnowledgeLevel;/);
  assert.match(callbackText, /knowledgeLevel: seed\?\.knowledgeLevel/);
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
  assert.match(reelCardSource, /if \(hasReachedVerifiedClipEnd\(playerTime, clipEnd\)\)/);
  assert.match(
    reelCardSource,
    /if \(hasReachedVerifiedClipEnd\(playerTime, clipEnd\)\) \{[\s\S]*?if \(!didHandleClipEndRef\.current\) \{[\s\S]*?player\.pauseVideo\(\);[\s\S]*?if \(autoplayEnabledRef\.current/,
  );
  const boundaryDeclaration = reelCardSource.match(
    /function hasReachedVerifiedClipEnd\([\s\S]*?\n}/,
  )?.[0];
  assert.ok(boundaryDeclaration);
  const compiledBoundary = ts.transpile(boundaryDeclaration, {
    target: ts.ScriptTarget.ES2022,
    module: ts.ModuleKind.CommonJS,
  });
  const hasReachedVerifiedClipEnd = new Function(
    `${compiledBoundary}\nreturn hasReachedVerifiedClipEnd;`,
  )();
  assert.equal(hasReachedVerifiedClipEnd(55.389, 55.4), false);
  assert.equal(hasReachedVerifiedClipEnd(55.391, 55.4), true);
  assert.equal(hasReachedVerifiedClipEnd(55.8, 55.4), true);
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

test("terminal tail gestures and autoplay never show a false search spinner", () => {
  const shared = {
    reels: [{ reel_id: "only" }],
    activeIndexRef: { current: 0 },
    hasMore: false,
    canRequestMore: false,
    isGeneratingRef: { current: false },
    isFetchingRef: { current: false },
    pendingAutoplayAdvanceRef: { current: false },
    setPendingTailAdvance: (value) => pendingStates.push(value),
    maybeLoadMore: () => {
      loadAttempts += 1;
    },
  };
  const pendingStates = [];
  let loadAttempts = 0;

  const gesture = compileUseCallback("shouldBlockDownwardAtEnd", {
    ...shared,
    wheelGestureConsumedRef: { current: true },
    wheelReadyToRearmRef: { current: true },
    wheelAccumRef: { current: 100 },
  });
  assert.equal(gesture(1), true);

  const autoplay = compileUseCallback("requestAutoplayAdvance", {
    ...shared,
    jumpOneReel: () => {
      throw new Error("terminal autoplay must not advance");
    },
  });
  autoplay();

  assert.equal(loadAttempts, 0);
  assert.deepEqual(pendingStates, [false, false]);
  assert.equal(shared.pendingAutoplayAdvanceRef.current, false);
});

test("a failed tail request clears its pending auto-advance spinner", () => {
  const requestMoreStart = source.indexOf("const requestMore = useCallback(");
  const requestMoreEnd = source.indexOf("\n\n  useEffect(() => {", requestMoreStart);
  assert.ok(requestMoreStart >= 0 && requestMoreEnd > requestMoreStart);
  const callbackText = source.slice(requestMoreStart, requestMoreEnd);
  assert.match(
    callbackText,
    /if \(generated\.length === 0\)[\s\S]*?clearPendingTailAdvance\(\);[\s\S]*?return \[\];/,
  );
  assert.match(
    callbackText,
    /catch \(e\)[\s\S]*?clearPendingTailAdvance\(\);[\s\S]*?return \[\];/,
  );
});

test("an already-running failed page fetch clears a later pending tail gesture", async () => {
  const searchScope = { key: "material", seq: 1, controller: new AbortController() };
  const isFetchingRef = { current: false };
  const isGeneratingRef = { current: false };
  const pendingAutoplayAdvanceRef = { current: false };
  let fetchStarted;
  const didStartFetch = new Promise((resolve) => {
    fetchStarted = resolve;
  });
  let rejectFetch;
  let clearedTailAdvance = 0;
  const callback = compileUseCallback("loadPage", {
    getFeedMaterialIds: () => ["material-a"],
    settingsScopeReady: true,
    isFetchingRef,
    activeSearchScopeRef: { current: searchScope },
    feedPagesExhausted: false,
    reelsRef: { current: [{ reel_id: "existing" }] },
    setRecoveryPhase: () => {},
    armActiveRecoveryRequest: () => {},
    getFeedTuningSettings: () => ({
      minRelevance: 0.75,
      creativeCommonsOnly: false,
      preferredVideoDuration: "any",
    }),
    generationMode: "fast",
    adaptiveExcludeReelIdsRef: { current: [] },
    PAGE_SIZE: 5,
    readyReservoirTarget: () => 8,
    fetchFeed: () => new Promise((resolve, reject) => {
      void resolve;
      rejectFetch = reject;
      fetchStarted();
    }),
    isSearchScopeActive: (scope) => scope === searchScope,
    markRecoveryProgress: () => {},
    markPagedFeedExhausted: () => {},
    isTransportError: () => false,
    noteFeedTransportFailure: () => {},
    noteFeedFailure: () => {},
    materialId: "material-a",
    finishActiveRecoveryRequest: () => {},
    setLoading: () => {},
    pendingAutoplayAdvanceRef,
    isGeneratingRef,
    clearPendingTailAdvance: () => {
      clearedTailAdvance += 1;
      pendingAutoplayAdvanceRef.current = false;
    },
  });

  const loadPromise = callback(2, { autofill: false });
  await didStartFetch;
  assert.equal(isFetchingRef.current, true);
  pendingAutoplayAdvanceRef.current = true;
  rejectFetch(new Error("feed page failed"));
  await loadPromise;

  assert.equal(isFetchingRef.current, false);
  assert.equal(clearedTailAdvance, 1);
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
