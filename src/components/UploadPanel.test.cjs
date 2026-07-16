const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const ts = require("typescript");

test("topic, text, and file submissions use the selected learner level", () => {
  const filePath = path.join(__dirname, "UploadPanel.tsx");
  const source = fs.readFileSync(filePath, "utf8");
  const sourceFile = ts.createSourceFile(
    filePath,
    source,
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TSX,
  );
  let submitCallback;

  function visit(node) {
    if (
      ts.isVariableDeclaration(node)
      && ts.isIdentifier(node.name)
      && node.name.text === "onSubmit"
      && node.initializer
      && ts.isCallExpression(node.initializer)
      && node.initializer.expression.getText(sourceFile) === "useCallback"
    ) {
      submitCallback = node.initializer;
      return;
    }
    ts.forEachChild(node, visit);
  }

  visit(sourceFile);
  assert.ok(submitCallback, "UploadPanel onSubmit must remain discoverable");
  const callbackText = submitCallback.arguments[0].getText(sourceFile);
  assert.match(
    source,
    /const \[knowledgeLevel, setKnowledgeLevel\] = useState<KnowledgeLevel>\("beginner"\)/,
    "the visible selector must default to beginner",
  );
  assert.match(
    callbackText,
    /topicList\.map\([\s\S]*knowledgeLevel,/,
    "multi-topic upload must use the selected level",
  );
  assert.match(
    callbackText,
    /const material = await uploadMaterial\(\{[\s\S]*knowledgeLevel,/,
    "topic, text, and file uploads must all use the selected level",
  );
  const dependencies = submitCallback.arguments[1];
  assert.ok(
    dependencies && ts.isArrayLiteralExpression(dependencies),
    "onSubmit must declare its hook dependencies",
  );
  assert.ok(
    dependencies.elements.some((element) => element.getText(sourceFile) === "knowledgeLevel"),
    "onSubmit must refresh when the selected level changes",
  );
  assert.match(source, /type="radio"/);
  assert.match(source, /name="knowledge-level"/);
  assert.match(source, /knowledgeLevel\?: KnowledgeLevel;/);
  assert.match(callbackText, /knowledgeLevel,\s*title: topic,/);
  assert.match(callbackText, /knowledgeLevel,\s*title,/);
  assert.doesNotMatch(
    callbackText,
    /targetClipDuration/,
    "URL ingestion must not send deprecated clip-duration preferences",
  );
});

test("partial multi-topic uploads preserve each fulfilled topic and material pairing", () => {
  const filePath = path.join(__dirname, "UploadPanel.tsx");
  const source = fs.readFileSync(filePath, "utf8");
  const sourceFile = ts.createSourceFile(
    filePath,
    source,
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TSX,
  );
  let pairingFunction;
  function visit(node) {
    if (ts.isFunctionDeclaration(node) && node.name?.text === "pairFulfilledTopicMaterials") {
      pairingFunction = node;
      return;
    }
    ts.forEachChild(node, visit);
  }
  visit(sourceFile);
  assert.ok(pairingFunction, "the fulfilled topic/material pairing helper must remain discoverable");
  const compiled = ts.transpile(
    `${pairingFunction.getText(sourceFile)}\nmodule.exports = pairFulfilledTopicMaterials;`,
    { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS },
  );
  const module = { exports: {} };
  new Function("module", "exports", compiled)(module, module.exports);
  const pairFulfilledTopicMaterials = module.exports;
  const fulfilled = (materialId) => ({ status: "fulfilled", value: { material_id: materialId } });
  const rejected = { status: "rejected", reason: new Error("upload failed") };

  assert.deepEqual(
    pairFulfilledTopicMaterials(
      ["biology", "photosynthesis", "chain rule"],
      [fulfilled("material-biology"), rejected, fulfilled("material-chain")],
    ),
    [
      { topic: "biology", materialId: "material-biology" },
      { topic: "chain rule", materialId: "material-chain" },
    ],
  );
  assert.deepEqual(
    pairFulfilledTopicMaterials(
      ["biology", "photosynthesis", "chain rule"],
      [rejected, fulfilled("material-photo"), fulfilled("material-chain")],
    ),
    [
      { topic: "photosynthesis", materialId: "material-photo" },
      { topic: "chain rule", materialId: "material-chain" },
    ],
  );
  assert.match(source, /materialIds = fulfilledTopicMaterials\.map\(\(\{ materialId \}\) => materialId\)/);
  assert.match(source, /fulfilledTopicMaterials\.forEach\(\(\{ topic, materialId \}, index\) =>/);
});

test("URL ingestion primes every returned reel with a legacy single-reel fallback", () => {
  const filePath = path.join(__dirname, "UploadPanel.tsx");
  const source = fs.readFileSync(filePath, "utf8");

  assert.match(
    source,
    /Array\.isArray\(result\.reels\) && result\.reels\.length > 0[\s\S]*\? result\.reels[\s\S]*: \[result\.reel\]/,
    "new URL responses must use all reels while old responses keep working",
  );
  assert.match(
    source,
    /primeFeedSessionSnapshot\(ingestMaterialId, ingestedReels, ingestedReel\.reel_id, activeSettings\)/,
    "the feed snapshot must receive the complete verified URL inventory",
  );
});

test("URL ingestion primes feed snapshots with the current v38 selection contract", () => {
  const filePath = path.join(__dirname, "UploadPanel.tsx");
  const source = fs.readFileSync(filePath, "utf8");
  const sourceFile = ts.createSourceFile(
    filePath,
    source,
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TSX,
  );
  let declaration;
  function visit(node) {
    if (ts.isFunctionDeclaration(node) && node.name?.text === "primeFeedSessionSnapshot") {
      declaration = node;
      return;
    }
    ts.forEachChild(node, visit);
  }
  visit(sourceFile);
  assert.ok(declaration, "snapshot priming helper must remain discoverable");
  const compiled = ts.transpile(
    declaration.getText(sourceFile),
    { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS },
  );
  const writes = [];
  const localStorage = {
    getItem: () => JSON.stringify({ existing: { reels: [] } }),
    setItem: (key, value) => writes.push({ key, value }),
  };
  const primeFeedSessionSnapshot = new Function(
    "window",
    "CURRENT_SELECTION_CONTRACT_VERSION",
    "FEED_SESSION_STORAGE_KEY",
    "safeStorageSetItem",
    `${compiled}\nreturn primeFeedSessionSnapshot;`,
  )(
    { localStorage },
    "quality_silence_v38",
    "studyreels-feed-sessions",
    (storage, key, value) => {
      storage.setItem(key, value);
      return true;
    },
  );

  primeFeedSessionSnapshot(
    "ingest-search:abc",
    [
      { reel_id: "reel-1" },
      { reel_id: "reel-2" },
    ],
    "reel-2",
    { generationMode: "slow", startMuted: true, autoplayNextReel: true },
  );

  assert.equal(writes.length, 1);
  assert.equal(writes[0].key, "studyreels-feed-sessions");
  const stored = JSON.parse(writes[0].value);
  assert.equal(stored["ingest-search:abc"].selectionContractVersion, "quality_silence_v38");
  assert.equal(stored["ingest-search:abc"].activeIndex, 1);
  assert.deepEqual(stored.existing, { reels: [] });
});
