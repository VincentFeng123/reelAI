const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const ts = require("typescript");

test("topic submission tracks the selected knowledge level", () => {
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
  assert.match(
    submitCallback.arguments[0].getText(sourceFile),
    /knowledgeLevel/,
    "topic upload must use the selected level",
  );
  const dependencies = submitCallback.arguments[1];
  assert.ok(
    dependencies && ts.isArrayLiteralExpression(dependencies),
    "onSubmit must declare its hook dependencies",
  );
  assert.ok(
    dependencies.elements.some((element) => element.getText(sourceFile) === "knowledgeLevel"),
    "knowledgeLevel must invalidate the memoized submit callback",
  );
});
