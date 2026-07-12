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
  fetchFeed,
  fetchGenerationStatus,
  generateReels,
  generateReelsStream,
  reportReelScroll,
} = require("./api.ts");
const TEST_OPTIONS = { timeout: 1_000 };

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
