const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const ts = require("typescript");

const filePath = path.join(__dirname, "localDemo.ts");
const source = fs.readFileSync(filePath, "utf8");

function loadModule(nodeEnv) {
  const previous = process.env.NODE_ENV;
  process.env.NODE_ENV = nodeEnv;
  try {
    const compiled = ts.transpile(source, {
      target: ts.ScriptTarget.ES2022,
      module: ts.ModuleKind.CommonJS,
    });
    const module = { exports: {} };
    new Function("module", "exports", compiled)(module, module.exports);
    return module.exports;
  } finally {
    if (previous === undefined) {
      delete process.env.NODE_ENV;
    } else {
      process.env.NODE_ENV = previous;
    }
  }
}

test("local demos are available only in development", () => {
  const development = loadModule("development");
  const production = loadModule("production");

  assert.equal(development.isLocalDemoView("account", "account"), true);
  assert.equal(development.isLocalDemoView("player", "player"), true);
  assert.equal(development.isLocalDemoView("quiz", "quiz"), true);
  assert.equal(development.isLocalDemoView("player", "account"), false);
  assert.equal(production.isLocalDemoView("account", "account"), false);
  assert.equal(production.LOCAL_DEMO_AVAILABLE, false);
});

test("demo fixtures provide a verified Pro identity, rich reels, a recall check, and useful owned sets", () => {
  const demo = loadModule("development");
  const reels = demo.LOCAL_DEMO_REELS;
  const assessment = demo.LOCAL_DEMO_ASSESSMENT_SESSION;
  const answers = demo.LOCAL_DEMO_ASSESSMENT_ANSWERS;
  const sets = demo.LOCAL_DEMO_COMMUNITY_SETS;

  assert.equal(demo.LOCAL_DEMO_ACCOUNT.isVerified, true);
  assert.equal(demo.LOCAL_DEMO_BILLING_STATUS.plan, "pro");
  assert.equal(demo.LOCAL_DEMO_HISTORY[0].feedQuery, "demo=player&return_tab=search");
  assert.equal(reels.length, 3);
  assert.equal(new Set(reels.map((reel) => reel.reel_id)).size, reels.length);
  for (const reel of reels) {
    assert.match(reel.reel_id, /^community:demo:/);
    assert.equal(new URL(reel.video_url).hostname, "www.youtube.com");
    assert.ok(reel.t_end > reel.t_start);
    assert.ok(reel.ai_summary.length > 40);
    assert.ok(reel.takeaways.length >= 3);
  }
  assert.equal(assessment.questions.length, assessment.question_count);
  assert.equal(assessment.question_count, 3);
  assert.equal(assessment.answered_count, 0);
  for (const question of assessment.questions) {
    assert.equal(question.options.length, 4);
    assert.ok(question.prompt.length > 30);
    assert.ok(question.reel_id.startsWith("community:demo:"));
    assert.ok(Number.isInteger(answers[question.id]?.correctIndex));
    assert.ok(answers[question.id].correctIndex >= 0 && answers[question.id].correctIndex < 4);
    assert.ok(answers[question.id].explanation.length > 40);
  }
  assert.equal(sets.length, 3);
  assert.equal(new Set(sets.map((set) => set.id)).size, sets.length);
  for (const set of sets) {
    assert.match(set.id, /^local-demo-set-/);
    assert.equal(set.curator, demo.LOCAL_DEMO_ACCOUNT.username);
    assert.equal(set.reelCount, set.reels.length);
    assert.ok(set.reels.length >= 2);
    assert.ok(set.description.length > 40);
    for (const reel of set.reels) {
      assert.equal(new URL(reel.sourceUrl).hostname, "www.youtube.com");
      assert.ok(reel.tEndSec > reel.tStartSec);
    }
  }
});
