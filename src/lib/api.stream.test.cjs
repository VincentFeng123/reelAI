const assert = require("node:assert/strict");
const test = require("node:test");
const Module = require("node:module");
const path = require("node:path");

const originalResolveFilename = Module._resolveFilename;
Module._resolveFilename = function resolveFrontendAlias(request, parent, isMain, options) {
  const resolvedRequest = request.startsWith("@/")
    ? path.join(process.cwd(), "src", request.slice(2))
    : request;
  return originalResolveFilename.call(this, resolvedRequest, parent, isMain, options);
};
require("sucrase/register/ts");

const {
  checkReelsCanGenerate,
  clearCommunityAuthSession,
  expireCommunityAuthSession,
  fetchCommunityAccount,
  fetchCommunityHistory,
  fetchCommunitySettings,
  fetchFeed,
  fetchGenerationStatus,
  generateReels,
  generateReelsStream,
  queueCommunityHistorySync,
  queueCommunitySettingsSync,
  readCommunityAuthSession,
  reportReelScroll,
  startNextAssessment,
} = require("./api.ts");
const TEST_OPTIONS = { timeout: 1_000 };

function installCommunitySessionTestWindow() {
  const localValues = new Map();
  const sessionValues = new Map();
  const storage = (values) => ({
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, String(value)),
    removeItem: (key) => values.delete(key),
  });
  const accountKey = "studyreels-community-account";
  const sessionKey = "studyreels-community-session-token";
  const ownerKey = "studyreels-community-owner-key";
  const ownerValue = `owner-${"o".repeat(32)}`;
  const setSession = (id, token) => {
    localValues.set(accountKey, JSON.stringify({ id, username: id, isVerified: true }));
    sessionValues.set(sessionKey, token);
  };

  localValues.set(ownerKey, ownerValue);
  global.window = {
    location: { hostname: "localhost" },
    localStorage: storage(localValues),
    sessionStorage: storage(sessionValues),
    dispatchEvent: () => true,
  };
  return {
    localValues,
    sessionValues,
    accountKey,
    sessionKey,
    ownerKey,
    ownerValue,
    setSession,
  };
}

function queuedGenerationResponse() {
  return new Response(JSON.stringify({
    job_id: "job-stream-test",
    status: "queued",
    deduplicated: false,
  }), {
    status: 202,
    headers: { "Content-Type": "application/json" },
  });
}

test("scroll reporting uses the bodyless forward-scroll contract", TEST_OPTIONS, async () => {
  const originalFetch = global.fetch;
  const originalWindow = global.window;
  let requestUrl;
  let requestInit;
  const localValues = new Map();
  const sessionValues = new Map();
  const storage = (values) => ({
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, String(value)),
    removeItem: (key) => values.delete(key),
  });

  global.window = {
    location: { hostname: "localhost" },
    localStorage: storage(localValues),
    sessionStorage: storage(sessionValues),
  };
  global.fetch = async (url, init = {}) => {
    requestUrl = String(url);
    requestInit = init;
    return new Response(JSON.stringify({
      reel_id: "reel/scroll-test",
      material_id: "material-scroll-test",
      newly_scrolled: true,
      assessment_ready: false,
      scroll_count: 1,
      cadence_target: 4,
    }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  };

  try {
    const response = await reportReelScroll({ reelId: "reel/scroll-test" });
    assert.match(requestUrl, /\/api\/reels\/reel%2Fscroll-test\/scroll$/);
    assert.equal(requestInit.method, "POST");
    assert.equal(requestInit.body, undefined);
    assert.ok(new Headers(requestInit.headers).get("X-StudyReels-Owner-Key"));
    assert.equal(response.cadence_target, 4);
    assert.equal(response.material_id, "material-scroll-test");
  } finally {
    global.fetch = originalFetch;
    if (originalWindow === undefined) {
      delete global.window;
    } else {
      global.window = originalWindow;
    }
  }
});

test("scroll and assessment-next time out even after response headers", TEST_OPTIONS, async () => {
  const originalFetch = global.fetch;
  const originalWindow = global.window;
  const originalSetTimeout = global.setTimeout;
  const observedTimeouts = [];
  installCommunitySessionTestWindow();

  global.setTimeout = (callback, delay, ...args) => {
    observedTimeouts.push(delay);
    return originalSetTimeout(callback, 0, ...args);
  };
  global.fetch = async (_url, init = {}) => new Response(new ReadableStream({
    start(controller) {
      init.signal.addEventListener("abort", () => {
        const error = new Error("aborted");
        error.name = "AbortError";
        controller.error(error);
      }, { once: true });
    },
  }), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });

  try {
    await assert.rejects(
      reportReelScroll({ reelId: "reel-timeout" }),
      /timed out before the backend response completed/i,
    );
    await assert.rejects(
      startNextAssessment({ materialId: "material-timeout" }),
      /timed out before the backend response completed/i,
    );
    assert.deepEqual(observedTimeouts, [6_000, 8_000]);
  } finally {
    global.fetch = originalFetch;
    global.setTimeout = originalSetTimeout;
    if (originalWindow === undefined) {
      delete global.window;
    } else {
      global.window = originalWindow;
    }
  }
});

test("generation submission timeout remains active after response headers", TEST_OPTIONS, async () => {
  const originalFetch = global.fetch;
  const originalSetTimeout = global.setTimeout;
  let requestSignal;

  global.setTimeout = (callback, delay, ...args) => originalSetTimeout(
    callback,
    delay === 30_000 ? 20 : delay,
    ...args,
  );
  global.fetch = async (_url, init = {}) => {
    requestSignal = init.signal;
    const body = new ReadableStream({
      start(controller) {
        requestSignal.addEventListener("abort", () => {
          controller.error(new DOMException("The operation was aborted", "AbortError"));
        }, { once: true });
        originalSetTimeout(() => {
          if (requestSignal.aborted) {
            return;
          }
          controller.enqueue(new TextEncoder().encode(JSON.stringify({
            job_id: "job-stream-test",
            status: "queued",
            status_url: "/status",
            stream_url: "/stream",
          })));
          controller.close();
        }, 40);
      },
    });
    return new Response(body, {
      status: 202,
      headers: { "Content-Type": "application/json" },
    });
  };

  try {
    await assert.rejects(
      generateReels({ materialId: "material-stream-test" }),
      /Request timed out before the backend response completed\./,
    );
    assert.equal(requestSignal.aborted, true);
  } finally {
    global.fetch = originalFetch;
    global.setTimeout = originalSetTimeout;
  }
});

test("generation honors the requested count within the mode ceiling and omits deprecated duration preferences", TEST_OPTIONS, async () => {
  const originalFetch = global.fetch;
  const requests = [];
  global.fetch = async (url, init = {}) => {
    requests.push({ url: String(url), init });
    if (String(url).includes("/reels/generate")) {
      return new Response(JSON.stringify({
        reels: [],
        response_profile: "unified",
        batch_id: "server-batch",
        batch_size: 0,
        continuation_token: "server-batch",
        terminal_status: "completed",
      }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }
    return new Response(JSON.stringify({
      reels: [],
      total: 0,
      page: 1,
      limit: 5,
      continuation_token: "feed-batch",
    }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  };

  let firstGeneration;
  let feedResponse;
  try {
    firstGeneration = await generateReels({
      materialId: "material-duration-compat",
      generationMode: "fast",
      numReels: 2,
      continuationToken: " previous-batch ",
      targetClipDurationSec: 55,
      targetClipDurationMinSec: 20,
      targetClipDurationMaxSec: 90,
    });
    await generateReels({
      materialId: "material-duration-compat",
      generationMode: "slow",
      numReels: 2,
    });
    await generateReels({
      materialId: "material-duration-compat",
      generationMode: "fast",
      numReels: 99,
    });
    await checkReelsCanGenerate({
      materialId: "material-duration-compat",
      generationMode: "fast",
    });
    await checkReelsCanGenerate({
      materialId: "material-duration-compat",
      generationMode: "slow",
    });
    feedResponse = await fetchFeed({
      materialId: "material-duration-compat",
      page: 1,
      limit: 5,
      generationMode: "slow",
      targetClipDurationSec: 55,
      targetClipDurationMinSec: 20,
      targetClipDurationMaxSec: 90,
    });
  } finally {
    global.fetch = originalFetch;
  }

  const generationBody = JSON.parse(requests[0].init.body);
  const slowGenerationBody = JSON.parse(requests[1].init.body);
  const clampedGenerationBody = JSON.parse(requests[2].init.body);
  const fastAvailabilityBody = JSON.parse(requests[3].init.body);
  const slowAvailabilityBody = JSON.parse(requests[4].init.body);
  assert.equal(generationBody.num_reels, 2);
  assert.equal(generationBody.generation_mode, "fast");
  assert.equal(generationBody.continuation_token, "previous-batch");
  assert.equal(Object.hasOwn(generationBody, "exclude_video_ids"), false);
  assert.equal(slowGenerationBody.num_reels, 2);
  assert.equal(slowGenerationBody.generation_mode, "slow");
  assert.equal(clampedGenerationBody.num_reels, 9);
  assert.equal(fastAvailabilityBody.num_reels, 9);
  assert.equal(slowAvailabilityBody.num_reels, 9);
  assert.equal(Object.hasOwn(generationBody, "target_clip_duration_sec"), false);
  assert.equal(Object.hasOwn(generationBody, "target_clip_duration_min_sec"), false);
  assert.equal(Object.hasOwn(generationBody, "target_clip_duration_max_sec"), false);
  assert.doesNotMatch(requests[5].url, /target_clip_duration/);
  assert.equal(new URL(requests[5].url, "http://test").searchParams.get("prefetch"), "9");
  assert.equal(firstGeneration.batch_id, "server-batch");
  assert.equal(firstGeneration.batch_size, 0);
  assert.equal(firstGeneration.continuation_token, "server-batch");
  assert.equal(firstGeneration.terminal_status, "completed");
  assert.equal(feedResponse.continuation_token, "feed-batch");
});

test("generation status timeout remains active after response headers", TEST_OPTIONS, async () => {
  const originalFetch = global.fetch;
  const originalSetTimeout = global.setTimeout;
  let requestSignal;

  global.setTimeout = (callback, delay, ...args) => originalSetTimeout(
    callback,
    delay === 30_000 ? 20 : delay,
    ...args,
  );
  global.fetch = async (_url, init = {}) => {
    requestSignal = init.signal;
    const body = new ReadableStream({
      start(controller) {
        requestSignal.addEventListener("abort", () => {
          controller.error(new DOMException("The operation was aborted", "AbortError"));
        }, { once: true });
      },
    });
    return new Response(body, {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  };

  try {
    await assert.rejects(
      fetchGenerationStatus("job-stream-test"),
      /Request timed out before the backend response completed\./,
    );
    assert.equal(requestSignal.aborted, true);
  } finally {
    global.fetch = originalFetch;
    global.setTimeout = originalSetTimeout;
  }
});

test("feed caller abort remains active after response headers", TEST_OPTIONS, async () => {
  const originalFetch = global.fetch;
  const caller = new AbortController();
  let requestSignal;
  let resolveBodyReadStarted;
  const bodyReadStarted = new Promise((resolve) => {
    resolveBodyReadStarted = resolve;
  });

  global.fetch = async (_url, init = {}) => {
    requestSignal = init.signal;
    const body = new ReadableStream({
      start(controller) {
        requestSignal.addEventListener("abort", () => {
          controller.error(new DOMException("The operation was aborted", "AbortError"));
        }, { once: true });
      },
      pull() {
        resolveBodyReadStarted();
      },
    });
    return new Response(body, {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  };

  const request = fetchFeed({
    materialId: "material-stream-test",
    page: 1,
    limit: 7,
    signal: caller.signal,
  });
  try {
    await bodyReadStarted;
    caller.abort();
    await assert.rejects(request, /Request was interrupted\./);
    assert.equal(requestSignal.aborted, true);
  } finally {
    await request.catch(() => undefined);
    global.fetch = originalFetch;
  }
});

test("caller abort remains active while an idle generation body is being read", TEST_OPTIONS, async () => {
  const originalFetch = global.fetch;
  const caller = new AbortController();
  let streamBody;
  let streamController;
  let streamRequestSignal;
  let resolveBodyReadStarted;
  const bodyReadStarted = new Promise((resolve) => {
    resolveBodyReadStarted = resolve;
  });

  global.fetch = async (url, init = {}) => {
    if (String(url).includes("/reels/generate")) {
      return queuedGenerationResponse();
    }
    if (!String(url).includes("/reels/generation-stream/")) {
      throw new Error(`Unexpected request: ${url}`);
    }

    streamRequestSignal = init.signal;
    streamBody = new ReadableStream({
      start(controller) {
        streamController = controller;
        streamRequestSignal.addEventListener("abort", () => {
          controller.error(new DOMException("The operation was aborted", "AbortError"));
        }, { once: true });
      },
      pull() {
        resolveBodyReadStarted();
      },
    });
    return new Response(streamBody, {
      status: 200,
      headers: { "Content-Type": "application/x-ndjson" },
    });
  };

  const generation = generateReelsStream({
    materialId: "material-stream-test",
    signal: caller.signal,
  });

  try {
    await bodyReadStarted;
    caller.abort();
    const outcome = await Promise.race([
      generation.then(
        () => ({ status: "resolved" }),
        (error) => ({ status: "rejected", error }),
      ),
      new Promise((resolve) => setTimeout(() => resolve({ status: "pending" }), 100)),
    ]);

    assert.equal(outcome.status, "rejected");
    assert.equal(outcome.error?.message, "Request was interrupted.");
    assert.equal(streamRequestSignal.aborted, true);
  } finally {
    if (!streamRequestSignal?.aborted) {
      streamController?.error(new Error("Test cleanup"));
    }
    await generation.catch(() => undefined);
    global.fetch = originalFetch;
  }

  assert.equal(streamBody.locked, false);
});

test("generation timeout remains active after stream headers arrive", TEST_OPTIONS, async () => {
  const originalFetch = global.fetch;
  const originalSetTimeout = global.setTimeout;
  let streamBody;
  let streamRequestSignal;
  let resolveBodyReadStarted;
  const bodyReadStarted = new Promise((resolve) => {
    resolveBodyReadStarted = resolve;
  });

  global.setTimeout = (callback, delay, ...args) => originalSetTimeout(
    callback,
    delay === 540_000 ? 20 : delay,
    ...args,
  );
  global.fetch = async (url, init = {}) => {
    if (String(url).includes("/reels/generate")) {
      return queuedGenerationResponse();
    }
    if (String(url).includes("/reels/generation-status/")) {
      return new Response(JSON.stringify({
        job_id: "job-stream-test",
        status: "failed",
        material_id: "material-stream-test",
        request_key: "request-stream-test",
        error: { code: "test_terminal", message: "Status checked after stream timeout." },
      }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }
    if (!String(url).includes("/reels/generation-stream/")) {
      throw new Error(`Unexpected request: ${url}`);
    }

    streamRequestSignal = init.signal;
    streamBody = new ReadableStream({
      start(controller) {
        streamRequestSignal.addEventListener("abort", () => {
          controller.error(new DOMException("The operation was aborted", "AbortError"));
        }, { once: true });
      },
      pull() {
        resolveBodyReadStarted();
      },
    });
    return new Response(streamBody, {
      status: 200,
      headers: { "Content-Type": "application/x-ndjson" },
    });
  };

  const generation = generateReelsStream({ materialId: "material-stream-test" });
  try {
    await bodyReadStarted;
    await assert.rejects(generation, /Status checked after stream timeout\./);
    assert.equal(streamRequestSignal.aborted, true);
  } finally {
    global.fetch = originalFetch;
    global.setTimeout = originalSetTimeout;
  }

  assert.equal(streamBody.locked, false);
});

test("transient stream and status failures reconnect to the durable job", TEST_OPTIONS, async () => {
  const originalFetch = global.fetch;
  const originalWindow = global.window;
  let streamCalls = 0;
  let statusCalls = 0;

  global.window = {
    setTimeout(callback, delay, ...args) {
      return setTimeout(callback, delay === 400 || delay === 401 ? 1 : delay, ...args);
    },
    clearTimeout,
  };
  global.fetch = async (url) => {
    if (String(url).includes("/reels/generate")) {
      return queuedGenerationResponse();
    }
    if (String(url).includes("/reels/generation-status/")) {
      statusCalls += 1;
      throw new TypeError("temporary status transport failure");
    }
    if (!String(url).includes("/reels/generation-stream/")) {
      throw new Error(`Unexpected request: ${url}`);
    }
    streamCalls += 1;
    if (streamCalls === 1) {
      throw new TypeError("temporary stream transport failure");
    }
    const events = [
      {
        job_id: "job-stream-test",
        seq: 1,
        timestamp: "2026-07-10T00:00:00Z",
        type: "final",
        payload: { reels: [], authoritative: true },
      },
      {
        job_id: "job-stream-test",
        seq: 2,
        timestamp: "2026-07-10T00:00:01Z",
        type: "terminal",
        payload: { status: "completed" },
      },
    ];
    return new Response(
      `${events.map((event) => JSON.stringify(event)).join("\n")}\n`,
      { status: 200, headers: { "Content-Type": "application/x-ndjson" } },
    );
  };

  try {
    const response = await generateReelsStream({ materialId: "material-stream-test" });
    assert.deepEqual(response.reels, []);
    assert.equal(response.batch_id, "job-stream-test");
    assert.equal(response.batch_size, 0);
    assert.equal(response.continuation_token, "job-stream-test");
    assert.equal(response.terminal_status, "completed");
    assert.equal(streamCalls, 2);
    assert.equal(statusCalls, 1);
  } finally {
    global.fetch = originalFetch;
    if (originalWindow === undefined) {
      delete global.window;
    } else {
      global.window = originalWindow;
    }
  }
});

test("status fallback attaches durable continuation metadata to a terminal batch", TEST_OPTIONS, async () => {
  const originalFetch = global.fetch;
  let statusCalls = 0;

  global.fetch = async (url) => {
    if (String(url).includes("/reels/generate")) {
      return queuedGenerationResponse();
    }
    if (String(url).includes("/reels/generation-status/")) {
      statusCalls += 1;
      return new Response(JSON.stringify({
        job_id: "job-stream-test",
        status: "partial",
        material_id: "material-stream-test",
        request_key: "request-stream-test",
        result_generation_id: "generation-partial",
        reels: [{ reel_id: "fallback-reel" }],
      }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }
    if (!String(url).includes("/reels/generation-stream/")) {
      throw new Error(`Unexpected request: ${url}`);
    }
    const terminal = {
      job_id: "job-stream-test",
      seq: 1,
      timestamp: "2026-07-15T00:00:00Z",
      type: "terminal",
      payload: { status: "partial" },
    };
    return new Response(`${JSON.stringify(terminal)}\n`, {
      status: 200,
      headers: { "Content-Type": "application/x-ndjson" },
    });
  };

  try {
    const response = await generateReelsStream({ materialId: "material-stream-test" });
    assert.deepEqual(response.reels.map((reel) => reel.reel_id), ["fallback-reel"]);
    assert.equal(response.batch_id, "job-stream-test");
    assert.equal(response.batch_size, 1);
    assert.equal(response.continuation_token, "job-stream-test");
    assert.equal(response.terminal_status, "partial");
    assert.equal(statusCalls, 1);
  } finally {
    global.fetch = originalFetch;
  }
});

test("feed-owned durable jobs resume without a redundant generation POST", TEST_OPTIONS, async () => {
  const originalFetch = global.fetch;
  let generateCalls = 0;
  let streamCalls = 0;

  global.fetch = async (url) => {
    if (String(url).includes("/reels/generate")) {
      generateCalls += 1;
      throw new Error("resume must not submit a new generation job");
    }
    if (!String(url).includes("/reels/generation-stream/job-from-feed")) {
      throw new Error(`Unexpected request: ${url}`);
    }
    streamCalls += 1;
    const events = [
      {
        job_id: "job-from-feed",
        seq: 1,
        timestamp: "2026-07-10T00:00:00Z",
        type: "final",
        payload: { reels: [], authoritative: true },
      },
      {
        job_id: "job-from-feed",
        seq: 2,
        timestamp: "2026-07-10T00:00:01Z",
        type: "terminal",
        payload: { status: "completed" },
      },
    ];
    return new Response(
      `${events.map((event) => JSON.stringify(event)).join("\n")}\n`,
      { status: 200, headers: { "Content-Type": "application/x-ndjson" } },
    );
  };

  try {
    const response = await generateReelsStream({
      materialId: "material-stream-test",
      generationJobId: "job-from-feed",
    });
    assert.deepEqual(response.reels, []);
    assert.equal(generateCalls, 0);
    assert.equal(streamCalls, 1);
  } finally {
    global.fetch = originalFetch;
  }
});

test("idle streams reconnect to the same durable job without cancelling it", TEST_OPTIONS, async () => {
  const originalFetch = global.fetch;
  const originalWindow = global.window;
  let streamCalls = 0;
  let statusCalls = 0;
  let cancelCalls = 0;
  let durableJobCancelCalls = 0;
  const reconnectWindows = [];

  global.window = {
    setTimeout(callback, delay, ...args) {
      return setTimeout(callback, delay === 400 || delay === 401 ? 1 : delay, ...args);
    },
    clearTimeout,
  };
  global.fetch = async (url) => {
    if (String(url).includes("/generation-jobs/") && String(url).endsWith("/cancel")) {
      durableJobCancelCalls += 1;
      throw new Error("the idle watchdog must not cancel the durable job");
    }
    if (String(url).includes("/reels/generate")) {
      throw new Error("resume must not submit a new generation job");
    }
    if (String(url).includes("/reels/generation-status/")) {
      statusCalls += 1;
      return new Response(JSON.stringify({
        job_id: "job-idle-resume",
        status: "running",
        material_id: "material-stream-test",
        request_key: "request-stream-test",
      }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }
    if (!String(url).includes("/reels/generation-stream/job-idle-resume")) {
      throw new Error(`Unexpected request: ${url}`);
    }
    streamCalls += 1;
    if (streamCalls === 1) {
      return new Response(new ReadableStream({
        cancel() {
          cancelCalls += 1;
        },
      }), {
        status: 200,
        headers: { "Content-Type": "application/x-ndjson" },
      });
    }
    const events = [
      {
        job_id: "job-idle-resume",
        seq: 1,
        timestamp: "2026-07-10T00:00:00Z",
        type: "final",
        payload: { reels: [], authoritative: true },
      },
      {
        job_id: "job-idle-resume",
        seq: 2,
        timestamp: "2026-07-10T00:00:01Z",
        type: "terminal",
        payload: { status: "completed" },
      },
    ];
    return new Response(
      `${events.map((event) => JSON.stringify(event)).join("\n")}\n`,
      { status: 200, headers: { "Content-Type": "application/x-ndjson" } },
    );
  };

  try {
    const response = await generateReelsStream({
      materialId: "material-stream-test",
      generationJobId: "job-idle-resume",
      idleTimeoutMs: 20,
      onReconnect: (windowCount) => reconnectWindows.push(windowCount),
    });
    assert.deepEqual(response.reels, []);
    assert.equal(streamCalls, 2);
    assert.equal(statusCalls, 1);
    assert.equal(cancelCalls, 1);
    assert.equal(durableJobCancelCalls, 0);
    assert.deepEqual(reconnectWindows, [1]);
  } finally {
    global.fetch = originalFetch;
    if (originalWindow === undefined) {
      delete global.window;
    } else {
      global.window = originalWindow;
    }
  }
});

test("terminal stream events cancel and release an unfinished response body", TEST_OPTIONS, async () => {
  const originalFetch = global.fetch;
  const encoder = new TextEncoder();
  let cancelCount = 0;
  let streamBody;
  let streamRequestSignal;

  global.fetch = async (url, init = {}) => {
    if (String(url).includes("/reels/generate")) {
      return queuedGenerationResponse();
    }
    if (!String(url).includes("/reels/generation-stream/")) {
      throw new Error(`Unexpected request: ${url}`);
    }

    streamRequestSignal = init.signal;
    const events = [
      {
        job_id: "job-stream-test",
        seq: 1,
        timestamp: "2026-07-10T00:00:00Z",
        type: "final",
        payload: { reels: [], authoritative: true },
      },
      {
        job_id: "job-stream-test",
        seq: 2,
        timestamp: "2026-07-10T00:00:01Z",
        type: "terminal",
        payload: { status: "completed" },
      },
    ];
    streamBody = new ReadableStream({
      start(controller) {
        controller.enqueue(encoder.encode(`${events.map((event) => JSON.stringify(event)).join("\n")}\n`));
      },
      cancel() {
        cancelCount += 1;
      },
    });
    return new Response(streamBody, {
      status: 200,
      headers: { "Content-Type": "application/x-ndjson" },
    });
  };

  try {
    const response = await generateReelsStream({ materialId: "material-stream-test" });
    assert.deepEqual(response.reels, []);
    assert.equal(cancelCount, 1);
    assert.equal(streamRequestSignal.aborted, true);
  } finally {
    global.fetch = originalFetch;
  }

  assert.equal(streamBody.locked, false);
});

test("community reads never return or apply a superseded session response", TEST_OPTIONS, async (t) => {
  const tokenA = `session-a-${"a".repeat(32)}`;
  const tokenB = `session-b-${"b".repeat(32)}`;
  const scenarios = [
    {
      name: "account success",
      read: fetchCommunityAccount,
      response: { account: { id: "account-a", username: "alice", is_verified: true } },
      status: 200,
    },
    {
      name: "account 401",
      read: fetchCommunityAccount,
      response: { detail: "Session expired" },
      status: 401,
    },
    {
      name: "history success",
      read: fetchCommunityHistory,
      response: { items: [{ material_id: "material-a", title: "Account A", updated_at: 1 }] },
      status: 200,
    },
    {
      name: "history 401",
      read: fetchCommunityHistory,
      response: { detail: "Session expired" },
      status: 401,
    },
    {
      name: "settings success",
      read: fetchCommunitySettings,
      response: {
        generation_mode: "slow",
        default_input_mode: "topic",
        min_relevance_threshold: 0.3,
        start_muted: true,
        creative_commons_only: false,
        preferred_video_duration: "any",
      },
      status: 200,
    },
    {
      name: "settings 401",
      read: fetchCommunitySettings,
      response: { detail: "Session expired" },
      status: 401,
    },
  ];

  for (const scenario of scenarios) {
    await t.test(scenario.name, async () => {
      const originalFetch = global.fetch;
      const originalWindow = global.window;
      const state = installCommunitySessionTestWindow();
      const requests = [];
      let resolveRequest;
      state.setSession("account-a", tokenA);
      global.fetch = async (url, init = {}) => {
        requests.push({ url: String(url), init });
        return new Promise((resolve) => {
          resolveRequest = resolve;
        });
      };

      try {
        const pending = scenario.read();
        await new Promise((resolve) => setImmediate(resolve));
        assert.equal(requests.length, 1);

        state.setSession("account-b", tokenB);
        resolveRequest(new Response(JSON.stringify(scenario.response), {
          status: scenario.status,
          headers: { "Content-Type": "application/json" },
        }));

        await assert.rejects(pending, /Community session changed while the request was in flight/);
        assert.equal(new Headers(requests[0].init.headers).get("X-StudyReels-Session-Token"), tokenA);
        assert.equal(readCommunityAuthSession()?.account.id, "account-b");
        assert.equal(readCommunityAuthSession()?.sessionToken, tokenB);
        assert.equal(state.localValues.get(state.ownerKey), state.ownerValue);
      } finally {
        global.fetch = originalFetch;
        if (originalWindow === undefined) {
          delete global.window;
        } else {
          global.window = originalWindow;
        }
      }
    });
  }
});

test("community read expiry invalidates only login state", TEST_OPTIONS, async (t) => {
  const scenarios = [fetchCommunityAccount, fetchCommunityHistory, fetchCommunitySettings];

  for (const read of scenarios) {
    await t.test(read.name, async () => {
      const originalFetch = global.fetch;
      const originalWindow = global.window;
      const state = installCommunitySessionTestWindow();
      state.setSession("account-current", `session-current-${"c".repeat(32)}`);
      global.fetch = async () => new Response(JSON.stringify({ detail: "Session expired" }), {
        status: 401,
        headers: { "Content-Type": "application/json" },
      });

      try {
        await read().catch(() => null);
        assert.equal(readCommunityAuthSession(), null);
        assert.equal(state.localValues.has(state.accountKey), false);
        assert.equal(state.sessionValues.has(state.sessionKey), false);
        assert.equal(state.localValues.get(state.ownerKey), state.ownerValue);
      } finally {
        global.fetch = originalFetch;
        if (originalWindow === undefined) {
          delete global.window;
        } else {
          global.window = originalWindow;
        }
      }
    });
  }
});

test("queued history sync never crosses community sessions", TEST_OPTIONS, async () => {
  const originalFetch = global.fetch;
  const originalWindow = global.window;
  const localValues = new Map();
  const sessionValues = new Map();
  const storage = (values) => ({
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, String(value)),
    removeItem: (key) => values.delete(key),
  });
  const accountKey = "studyreels-community-account";
  const sessionKey = "studyreels-community-session-token";
  const ownerKey = "studyreels-community-owner-key";
  const ownerValue = `owner-${"x".repeat(32)}`;
  const tokenA = `session-a-${"a".repeat(32)}`;
  const tokenB = `session-b-${"b".repeat(32)}`;
  const setSession = (id, username, token) => {
    localValues.set(accountKey, JSON.stringify({ id, username, isVerified: true }));
    sessionValues.set(sessionKey, token);
  };
  const requests = [];
  let resolveFirstRequest;

  localValues.set(ownerKey, ownerValue);
  setSession("account-a", "alice", tokenA);
  global.window = {
    location: { hostname: "localhost" },
    localStorage: storage(localValues),
    sessionStorage: storage(sessionValues),
    dispatchEvent: () => true,
  };
  global.fetch = async (url, init = {}) => {
    requests.push({ url: String(url), init });
    if (requests.length === 1) {
      return new Promise((resolve) => {
        resolveFirstRequest = resolve;
      });
    }
    return new Response(JSON.stringify({ items: [] }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  };

  try {
    const first = queueCommunityHistorySync([{ materialId: "first", title: "First", updatedAt: 1 }]);
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(requests.length, 1);

    const queuedForA = queueCommunityHistorySync([{ materialId: "second", title: "Second", updatedAt: 2 }]);
    setSession("account-b", "bob", tokenB);
    resolveFirstRequest(new Response(JSON.stringify({ detail: "Session expired" }), {
      status: 401,
      headers: { "Content-Type": "application/json" },
    }));
    await Promise.all([first, queuedForA]);

    assert.equal(requests.length, 1, "queued account-A data must not dispatch after account B signs in");
    assert.equal(new Headers(requests[0].init.headers).get("X-StudyReels-Session-Token"), tokenA);
    assert.equal(readCommunityAuthSession()?.account.id, "account-b", "A's late 401 must not clear account B");
    assert.equal(localValues.get(ownerKey), ownerValue);

    await queueCommunityHistorySync([{ materialId: "third", title: "Third", updatedAt: 3 }]);
    assert.equal(requests.length, 2);
    assert.equal(new Headers(requests[1].init.headers).get("X-StudyReels-Session-Token"), tokenB);
  } finally {
    global.fetch = originalFetch;
    if (originalWindow === undefined) {
      delete global.window;
    } else {
      global.window = originalWindow;
    }
  }
});

test("queued settings sync never crosses community sessions", TEST_OPTIONS, async () => {
  const originalFetch = global.fetch;
  const originalWindow = global.window;
  const localValues = new Map();
  const sessionValues = new Map();
  const storage = (values) => ({
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, String(value)),
    removeItem: (key) => values.delete(key),
  });
  const accountKey = "studyreels-community-account";
  const sessionKey = "studyreels-community-session-token";
  const ownerKey = "studyreels-community-owner-key";
  const ownerValue = `owner-${"q".repeat(32)}`;
  const tokenA = `session-a-${"a".repeat(32)}`;
  const tokenB = `session-b-${"b".repeat(32)}`;
  const settings = {
    generationMode: "slow",
    defaultInputMode: "topic",
    minRelevanceThreshold: 0.3,
    startMuted: true,
    creativeCommonsOnly: false,
    preferredVideoDuration: "any",
  };
  const responseSettings = {
    generation_mode: "slow",
    default_input_mode: "topic",
    min_relevance_threshold: 0.3,
    start_muted: true,
    creative_commons_only: false,
    preferred_video_duration: "any",
  };
  const setSession = (id, token) => {
    localValues.set(accountKey, JSON.stringify({ id, username: id, isVerified: true }));
    sessionValues.set(sessionKey, token);
  };
  const requests = [];
  let resolveFirstRequest;

  localValues.set(ownerKey, ownerValue);
  setSession("account-a", tokenA);
  global.window = {
    location: { hostname: "localhost" },
    localStorage: storage(localValues),
    sessionStorage: storage(sessionValues),
    dispatchEvent: () => true,
  };
  global.fetch = async (url, init = {}) => {
    requests.push({ url: String(url), init });
    if (requests.length === 1) {
      return new Promise((resolve) => {
        resolveFirstRequest = resolve;
      });
    }
    return new Response(JSON.stringify(responseSettings), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  };

  try {
    const first = queueCommunitySettingsSync(settings);
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(requests.length, 1);

    const queuedForA = queueCommunitySettingsSync({ ...settings, startMuted: false });
    setSession("account-b", tokenB);
    resolveFirstRequest(new Response(JSON.stringify(responseSettings), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    }));
    await Promise.all([first, queuedForA]);

    assert.equal(requests.length, 1, "queued account-A settings must not dispatch after account B signs in");
    assert.equal(new Headers(requests[0].init.headers).get("X-StudyReels-Session-Token"), tokenA);

    await queueCommunitySettingsSync(settings);
    assert.equal(requests.length, 2);
    assert.equal(new Headers(requests[1].init.headers).get("X-StudyReels-Session-Token"), tokenB);
    assert.equal(localValues.get(ownerKey), ownerValue);
  } finally {
    global.fetch = originalFetch;
    if (originalWindow === undefined) {
      delete global.window;
    } else {
      global.window = originalWindow;
    }
  }
});

test("history 401 clears only the current community session", TEST_OPTIONS, async () => {
  const originalFetch = global.fetch;
  const originalWindow = global.window;
  const localValues = new Map();
  const sessionValues = new Map();
  const storage = (values) => ({
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, String(value)),
    removeItem: (key) => values.delete(key),
  });
  const accountKey = "studyreels-community-account";
  const sessionKey = "studyreels-community-session-token";
  const ownerKey = "studyreels-community-owner-key";
  const ownerValue = `owner-${"y".repeat(32)}`;

  localValues.set(accountKey, JSON.stringify({ id: "account-current", username: "current", isVerified: true }));
  localValues.set(ownerKey, ownerValue);
  sessionValues.set(sessionKey, `session-current-${"c".repeat(32)}`);
  global.window = {
    location: { hostname: "localhost" },
    localStorage: storage(localValues),
    sessionStorage: storage(sessionValues),
    dispatchEvent: () => true,
  };
  global.fetch = async () => new Response(JSON.stringify({ detail: "Session expired" }), {
    status: 401,
    headers: { "Content-Type": "application/json" },
  });

  try {
    await queueCommunityHistorySync([{ materialId: "current", title: "Current", updatedAt: 1 }]);
    assert.equal(readCommunityAuthSession(), null);
    assert.equal(localValues.has(accountKey), false);
    assert.equal(sessionValues.has(sessionKey), false);
    assert.equal(localValues.get(ownerKey), ownerValue, "device owner identity must survive stale login cleanup");
  } finally {
    global.fetch = originalFetch;
    if (originalWindow === undefined) {
      delete global.window;
    } else {
      global.window = originalWindow;
    }
  }
});

test("session expiry preserves the device owner while explicit logout clears it", TEST_OPTIONS, () => {
  const originalWindow = global.window;
  const localValues = new Map();
  const sessionValues = new Map();
  const storage = (values) => ({
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, String(value)),
    removeItem: (key) => values.delete(key),
  });
  const accountKey = "studyreels-community-account";
  const sessionKey = "studyreels-community-session-token";
  const ownerKey = "studyreels-community-owner-key";
  const ownerValue = `owner-${"z".repeat(32)}`;

  localValues.set(accountKey, JSON.stringify({ id: "account", username: "user" }));
  localValues.set(ownerKey, ownerValue);
  sessionValues.set(sessionKey, `session-${"s".repeat(32)}`);
  global.window = {
    localStorage: storage(localValues),
    sessionStorage: storage(sessionValues),
    dispatchEvent: () => true,
  };

  try {
    expireCommunityAuthSession();
    assert.equal(localValues.has(accountKey), false);
    assert.equal(sessionValues.has(sessionKey), false);
    assert.equal(localValues.get(ownerKey), ownerValue);

    clearCommunityAuthSession();
    assert.equal(localValues.has(ownerKey), false);
  } finally {
    if (originalWindow === undefined) {
      delete global.window;
    } else {
      global.window = originalWindow;
    }
  }
});
