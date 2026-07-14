const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const ts = require("typescript");

test("topic submission starts at the hidden beginner level", () => {
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
    /const DEFAULT_KNOWLEDGE_LEVEL = "beginner" as const;/,
    "the hidden initial level must stay beginner",
  );
  assert.match(
    callbackText,
    /topicList\.map\([\s\S]*knowledgeLevel: DEFAULT_KNOWLEDGE_LEVEL/,
    "multi-topic upload must preserve the hidden starting level",
  );
  assert.match(
    callbackText,
    /knowledgeLevel: inputMode === "topic" \? DEFAULT_KNOWLEDGE_LEVEL : undefined/,
    "single-topic upload must preserve the hidden starting level",
  );
  const dependencies = submitCallback.arguments[1];
  assert.ok(
    dependencies && ts.isArrayLiteralExpression(dependencies),
    "onSubmit must declare its hook dependencies",
  );
  assert.ok(!source.includes('aria-label="How well do you know this topic?"'));
  assert.ok(!source.includes('role="radiogroup"'));
  assert.doesNotMatch(source, /setKnowledgeLevel/);
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
