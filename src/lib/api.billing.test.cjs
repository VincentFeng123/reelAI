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
  createStripeCheckout,
  createStripePortal,
  fetchBillingPlans,
  fetchBillingStatus,
  ingestUrl,
  isDailySearchLimitError,
  uploadMaterial,
} = require("./api.ts");

function installBillingWindow() {
  const localValues = new Map([
    ["studyreels-community-account", JSON.stringify({ id: "account-billing", username: "billing", isVerified: true })],
  ]);
  const sessionValues = new Map([
    ["studyreels-community-session-token", `session-${"s".repeat(32)}`],
  ]);
  const storage = (values) => ({
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, String(value)),
    removeItem: (key) => values.delete(key),
  });
  global.window = {
    location: { hostname: "localhost" },
    localStorage: storage(localValues),
    sessionStorage: storage(sessionValues),
    dispatchEvent: () => true,
  };
}

test("billing plans and Stripe status preserve the public contract", async () => {
  const originalFetch = global.fetch;
  const originalWindow = global.window;
  installBillingWindow();
  const requests = [];
  global.fetch = async (url, init = {}) => {
    requests.push({ url: String(url), init });
    if (String(url).endsWith("/billing/plans")) {
      return Response.json({ plans: [
        { code: "free", name: "Free", monthly_price_cents: 0, daily_limit: 5 },
        { code: "plus", name: "Plus", monthly_price_cents: 499, daily_limit: 15 },
        { code: "pro", name: "Pro", monthly_price_cents: 1999, daily_limit: 50 },
      ] });
    }
    return Response.json({
      plan: "plus",
      daily_limit: 15,
      used_searches: 4,
      remaining_searches: 11,
      reset_at: "2026-07-19T00:00:00Z",
      subscriptions: [{
        provider: "stripe",
        plan: "plus",
        status: "active",
        current_period_end: "2026-08-18T00:00:00Z",
        cancel_at_period_end: false,
        product_id: "price_plus",
      }],
    });
  };

  try {
    const [plans, status] = await Promise.all([fetchBillingPlans(), fetchBillingStatus()]);
    assert.deepEqual(plans.map((plan) => [plan.code, plan.monthly_price_cents, plan.daily_limit]), [
      ["free", 0, 5],
      ["plus", 499, 15],
      ["pro", 1999, 50],
    ]);
    assert.equal(status.plan, "plus");
    assert.equal(status.remaining_searches, 11);
    assert.equal(status.subscriptions[0].provider, "stripe");
    const statusRequest = requests.find((request) => request.url.endsWith("/billing/status"));
    assert.ok(new Headers(statusRequest.init.headers).get("X-StudyReels-Session-Token"));
  } finally {
    global.fetch = originalFetch;
    if (originalWindow === undefined) delete global.window;
    else global.window = originalWindow;
  }
});

test("Stripe checkout and portal are server-created and authenticated", async () => {
  const originalFetch = global.fetch;
  const originalWindow = global.window;
  installBillingWindow();
  const requests = [];
  global.fetch = async (url, init = {}) => {
    requests.push({ url: String(url), init });
    return Response.json({
      url: String(url).endsWith("/billing/stripe/checkout")
        ? "https://checkout.stripe.com/c/pay/session"
        : "https://billing.stripe.com/p/session/portal",
    });
  };

  try {
    assert.equal(await createStripeCheckout("pro"), "https://checkout.stripe.com/c/pay/session");
    assert.equal(await createStripePortal(), "https://billing.stripe.com/p/session/portal");
    assert.deepEqual(JSON.parse(requests[0].init.body), { plan: "pro" });
    assert.equal(requests[0].init.method, "POST");
    assert.equal(requests[1].init.method, "POST");
    assert.ok(new Headers(requests[0].init.headers).get("X-StudyReels-Session-Token"));
    assert.equal(JSON.stringify(requests).includes("price_"), false, "the browser must never choose a Stripe Price ID");
  } finally {
    global.fetch = originalFetch;
    if (originalWindow === undefined) delete global.window;
    else global.window = originalWindow;
  }
});

test("Stripe redirects reject untrusted HTTPS hosts, credentials, and non-default ports", async () => {
  const originalFetch = global.fetch;
  const originalWindow = global.window;
  installBillingWindow();

  try {
    const checkoutUrls = [
      "https://evil.example/session",
      "https://checkout.stripe.com.evil.example/session",
      "https://user:secret@checkout.stripe.com/session",
      "https://checkout.stripe.com:8443/session",
    ];
    for (const url of checkoutUrls) {
      global.fetch = async () => Response.json({ url });
      await assert.rejects(createStripeCheckout("plus"), /invalid Stripe Checkout URL/);
    }

    const portalUrls = [
      "https://evil.example/session",
      "https://billing.stripe.com.evil.example/session",
      "https://user:secret@billing.stripe.com/session",
      "https://billing.stripe.com:8443/session",
    ];
    for (const url of portalUrls) {
      global.fetch = async () => Response.json({ url });
      await assert.rejects(createStripePortal(), /invalid Stripe Customer Portal URL/);
    }
  } finally {
    global.fetch = originalFetch;
    if (originalWindow === undefined) delete global.window;
    else global.window = originalWindow;
  }
});

test("daily quota 429 keeps typed top-level details and Retry-After", async () => {
  const originalFetch = global.fetch;
  const originalWindow = global.window;
  installBillingWindow();
  global.fetch = async () => Response.json({
    detail: {
      code: "daily_search_limit_reached",
      plan: "free",
      limit: 5,
      used: 5,
      remaining: 0,
      reset_at: "2026-07-19T00:00:00Z",
    },
  }, { status: 429, headers: { "Retry-After": "3600" } });

  try {
    await assert.rejects(fetchBillingStatus(), (error) => {
      assert.equal(isDailySearchLimitError(error), true);
      assert.equal(error.code, "daily_search_limit_reached");
      assert.equal(error.payload.retry_after_sec, 3600);
      assert.equal(error.payload.details.plan, "free");
      assert.equal(error.payload.details.limit, 5);
      assert.equal(error.payload.details.reset_at, "2026-07-19T00:00:00Z");
      return true;
    });
  } finally {
    global.fetch = originalFetch;
    if (originalWindow === undefined) delete global.window;
    else global.window = originalWindow;
  }
});

test("material and URL retries reuse durable idempotency keys", async () => {
  const originalFetch = global.fetch;
  const originalWindow = global.window;
  installBillingWindow();
  const requests = [];
  let requestCount = 0;
  global.fetch = async (url, init = {}) => {
    requestCount += 1;
    requests.push({ url: String(url), init });
    if (requestCount === 1 || requestCount === 4) {
      return Response.json({ detail: "temporary failure" }, { status: 503 });
    }
    if (String(url).endsWith("/material")) {
      return Response.json({ material_id: `material-${requestCount}`, extracted_concepts: [] });
    }
    return Response.json({ reel: {}, reels: [], metadata: {}, trace_id: "trace" });
  };

  try {
    await assert.rejects(uploadMaterial({ subjectTag: "linear regression" }));
    await uploadMaterial({ subjectTag: "linear regression" });
    await uploadMaterial({ subjectTag: "linear regression" });
    await assert.rejects(ingestUrl({ sourceUrl: "https://www.youtube.com/watch?v=abcdefghijk" }));
    await ingestUrl({ sourceUrl: "https://www.youtube.com/watch?v=abcdefghijk" });

    const keys = requests.map((request) => new Headers(request.init.headers).get("Idempotency-Key"));
    assert.ok(keys.every(Boolean));
    assert.equal(keys[0], keys[1], "material transport retry must reuse its key");
    assert.notEqual(keys[1], keys[2], "a completed material operation must release its pending key");
    assert.equal(keys[3], keys[4], "URL transport retry must reuse its key");
  } finally {
    global.fetch = originalFetch;
    if (originalWindow === undefined) delete global.window;
    else global.window = originalWindow;
  }
});
