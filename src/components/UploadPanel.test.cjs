const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const ts = require("typescript");

const filePath = path.join(__dirname, "UploadPanel.tsx");
const source = fs.readFileSync(filePath, "utf8");
const sourceFile = ts.createSourceFile(
  filePath,
  source,
  ts.ScriptTarget.Latest,
  true,
  ts.ScriptKind.TSX,
);

function routeHelpers() {
  const names = new Set([
    "HOME_IDEAS",
    "INGEST_URL_HOST_ALLOWLIST",
    "YOUTUBE_VIDEO_ID_RE",
    "homeGreetingForHour",
    "isLikelyIngestUrl",
    "resolveUnifiedComposerRoute",
  ]);
  const declarations = sourceFile.statements
    .filter((statement) => {
      if (ts.isFunctionDeclaration(statement)) {
        return names.has(statement.name?.text);
      }
      if (ts.isVariableStatement(statement)) {
        return statement.declarationList.declarations.some(
          (declaration) => ts.isIdentifier(declaration.name) && names.has(declaration.name.text),
        );
      }
      return false;
    })
    .map((statement) => statement.getText(sourceFile).replace(/^export\s+/, ""))
    .join("\n");
  const compiled = ts.transpile(declarations, {
    target: ts.ScriptTarget.ES2022,
    module: ts.ModuleKind.None,
  });
  return new Function(`${compiled}\nreturn { HOME_IDEAS, homeGreetingForHour, isLikelyIngestUrl, resolveUnifiedComposerRoute };`)();
}

function submitCallbackText() {
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
  return {
    text: submitCallback.arguments[0].getText(sourceFile),
    dependencies: submitCallback.arguments[1],
  };
}

test("unified routing follows attachment, URL, multiline source text, then topic precedence", () => {
  const { resolveUnifiedComposerRoute } = routeHelpers();
  const youtubeUrl = "https://youtu.be/abcdefghijk";

  assert.equal(
    resolveUnifiedComposerRoute({ attachment: { name: "notes.pdf" }, prompt: youtubeUrl }),
    "file",
  );
  assert.equal(resolveUnifiedComposerRoute({ prompt: youtubeUrl }), "url");
  assert.equal(resolveUnifiedComposerRoute({ prompt: "first line\nsecond line" }), "source");
  assert.equal(resolveUnifiedComposerRoute({ prompt: "detailed one-line topic\n" }), "topic");
  assert.equal(resolveUnifiedComposerRoute({ prompt: "x".repeat(240) }), "topic");
  assert.equal(resolveUnifiedComposerRoute({ prompt: "x".repeat(80) }), "topic");
  assert.equal(resolveUnifiedComposerRoute({ prompt: "linear regression" }), "topic");
});

test("only supported, complete YouTube locations route to URL ingestion", () => {
  const { isLikelyIngestUrl, resolveUnifiedComposerRoute } = routeHelpers();

  assert.equal(isLikelyIngestUrl("youtube.com/watch?v=abcdefghijk"), true);
  assert.equal(isLikelyIngestUrl("https://www.youtube.com/shorts/abcdefghijk"), true);
  assert.equal(isLikelyIngestUrl("https://youtu.be/abcdefghijk?t=12"), true);
  assert.equal(isLikelyIngestUrl("https://vimeo.com/abcdefghijk"), false);
  assert.equal(isLikelyIngestUrl("https://youtube.com/watch?v=short"), false);
  assert.equal(
    resolveUnifiedComposerRoute({ prompt: "Watch https://youtu.be/abcdefghijk" }),
    "topic",
    "a URL embedded in a prompt must not bypass topic routing",
  );
});

test("home greeting follows local time and rotates four quiet learning actions from a broad icon-backed pool", () => {
  const { HOME_IDEAS, homeGreetingForHour } = routeHelpers();

  assert.equal(homeGreetingForHour(4, " Vincent "), "Good evening, Vincent");
  assert.equal(homeGreetingForHour(5, "Vincent"), "Good morning, Vincent");
  assert.equal(homeGreetingForHour(11, "Vincent"), "Good morning, Vincent");
  assert.equal(homeGreetingForHour(12, "Vincent"), "Good afternoon, Vincent");
  assert.equal(homeGreetingForHour(16, "Vincent"), "Good afternoon, Vincent");
  assert.equal(homeGreetingForHour(17, "Vincent"), "Good evening, Vincent");
  assert.equal(homeGreetingForHour(23, null), "Good evening");
  assert.ok(HOME_IDEAS.length >= 30);
  assert.equal(HOME_IDEAS.length % 4, 0);
  assert.equal(new Set(HOME_IDEAS.map(({ title }) => title)).size, HOME_IDEAS.length);
  assert.ok(HOME_IDEAS.every(({ icon }) => icon.startsWith("fa-")));
  assert.match(source, /What can I help you learn today\?/);
  assert.match(source, /new Date\(\)\.getHours\(\)/);
  assert.match(source, /data-home-suggestions="true"/);
  assert.match(source, /const HOME_IDEAS_PER_PAGE = 4/);
  assert.match(source, /const HOME_IDEA_ROTATION_MS = 5 \* 60 \* 1_000/);
  assert.match(source, /didRotateHomeIdeasOnMountRef\.current = true/);
  assert.match(source, /window\.sessionStorage\.getItem\(HOME_IDEA_OFFSET_STORAGE_KEY\)/);
  assert.match(source, /const refreshOffset = \(previousOffset \+ HOME_IDEAS_PER_PAGE\) % HOME_IDEAS\.length/);
  assert.match(source, /window\.setInterval\(advanceHomeIdeas, HOME_IDEA_ROTATION_MS\)/);
  assert.match(source, /visibleHomeIdeas\.map/);
  assert.match(source, /setPrompt\(idea\.title\)/);
  assert.match(source, /textareaRef\.current\?\.focus\(\)/);
  assert.match(source, /const showBottomSignInStatus = !loading && !error && !verifiedAccount\?\.isVerified/);
  assert.match(source, /compactStatus && !showBottomSignInStatus/);
  assert.match(source, /bottom-\[calc\(max\(env\(safe-area-inset-bottom\),0px\)\+18px\)\][^"\n]*left-1\/2[^"\n]*text-center/);
  assert.match(source, /data-home-suggestion-offset=\{homeIdeaOffset\}/);
  assert.match(source, /className=\{`fa-regular \$\{name\}[^`]*font-normal/);
  const homeIdeaIcon = source.slice(source.indexOf("function HomeIdeaIcon"), source.indexOf("export type UnifiedComposerRoute"));
  assert.match(homeIdeaIcon, /<i/);
  assert.doesNotMatch(homeIdeaIcon, /<svg|<text|<path/);
  assert.match(source, /min-h-11[^"\n]*gap-3\.5[^"\n]*py-2[^"\n]*text-\[13px\][^"\n]*sm:text-\[14px\]/);
  assert.doesNotMatch(source, /min-h-8 px-2 pt-2/);
  assert.doesNotMatch(source, /Ready when you are\./);
});

test("topic, source, and file submissions preserve one-search uploads and learner level", () => {
  const callback = submitCallbackText();

  assert.match(source, /const \[knowledgeLevel, setKnowledgeLevel\] = useState<KnowledgeLevel>\("beginner"\)/);
  assert.match(source, /name="knowledge-level"/);
  assert.match(callback.text, /const topicValue = composerRoute === "topic" \? promptValue : ""/);
  assert.match(callback.text, /const textValue = composerRoute === "source" \|\| composerRoute === "file" \? promptValue : ""/);
  assert.match(callback.text, /const fileValue = composerRoute === "file" \? file : undefined/);
  assert.match(callback.text, /const material = await uploadMaterial\(\{[\s\S]*knowledgeLevel,/);
  assert.match(callback.text, /knowledgeLevel,\s*title,/);
  assert.equal((callback.text.match(/await uploadMaterial\(/g) || []).length, 1);
  assert.doesNotMatch(source, /Add topic|Promise\.allSettled|partial_topics|topicOperations/);
  assert.doesNotMatch(callback.text, /targetClipDuration/);
  assert.ok(
    callback.dependencies
      && ts.isArrayLiteralExpression(callback.dependencies)
      && callback.dependencies.elements.some((element) => element.getText(sourceFile) === "knowledgeLevel"),
    "onSubmit must refresh when the selected level changes",
  );
});

test("composer is taller, narrower, subtly outlined, and retains native attachment controls", () => {
  assert.match(source, /<textarea[\s\S]*rows=\{1\}[\s\S]*min-h-\[72px\][\s\S]*max-h-\[92px\]/);
  assert.match(source, /Math\.min\(element\.scrollHeight, 92\)/);
  assert.match(source, /Math\.max\(nextHeight, 72\)/);
  assert.match(source, /max-w-\[680px\]/);
  assert.match(source, /border-\[0\.5px\] border-\[#3a3a3a\] bg-\[#242424\]/);
  assert.match(source, /bg-transparent px-1 py-1/);
  assert.match(source, /text-\[17px\] leading-7/);
  assert.match(source, /accept="\.pdf,\.docx,\.txt,/);
  assert.match(source, /className="hidden"[\s\S]*tabIndex=\{-1\}[\s\S]*aria-hidden="true"[\s\S]*type="file"/);
  assert.match(source, /aria-label="Attach a PDF, DOCX, or TXT file"/);
  assert.match(source, /aria-label=\{`Remove \$\{selectedFileName\}`\}/);
  assert.match(source, /aria-label=\{loading \? "Starting search" : "Send"\}/);
  assert.match(source, /rounded-full bg-white text-black/);
  assert.match(source, /<CustomSelect[\s\S]*label="Knowledge level"/);
  assert.match(source, /className="w-fit min-w-0"/);
  assert.match(source, /buttonClassName="[^"]*rounded-full bg-transparent[^"]*hover:text-white/);
  assert.match(source, /buttonClassName="[^"]*bg-transparent px-3[^"]*hover:bg-white\/\[0\.07\]/);
  assert.match(source, /buttonClassName="[^"]*hover:bg-white\/\[0\.07\][^"]*"/);
  assert.match(source, /menuClassName="w-\[200%\] max-w-\[calc\(100vw-5rem\)\]"/);
  assert.match(source, /showSelectedCheck/);
  assert.doesNotMatch(source, /<select/);
  assert.doesNotMatch(source, /Select input mode|defaultInputMode/);
});

test("billing is supplied by the shell and desktop Enter is IME-safe", () => {
  assert.match(source, /billingStatus\?: BillingStatus \| null/);
  assert.doesNotMatch(source, /useBillingStatus\(/);
  assert.match(source, /event\.shiftKey/);
  assert.match(source, /event\.nativeEvent\.isComposing/);
  assert.match(source, /event\.nativeEvent\.keyCode === 229/);
  assert.match(source, /window\.matchMedia\("\(pointer: coarse\)"\)\.matches/);
  assert.match(source, /formRef\.current\?\.requestSubmit\(\)/);
  assert.match(source, /!active \|\| loading \|\| \(!file && !prompt\.trim\(\)\)/);
  assert.match(source, /if \(demoMode\) \{[\s\S]*router\.push\("\/feed\?demo=player&return_tab=search"\)/);
});

test("file submissions retain optional composer instructions", () => {
  assert.match(source, /composerRoute === "source" \|\| composerRoute === "file" \? promptValue : ""/);
  assert.match(source, /text: textValue \|\| undefined,[\s\S]*file: fileValue/);
});

test("URL ingestion primes every returned reel with a legacy single-reel fallback", () => {
  assert.match(
    source,
    /Array\.isArray\(result\.reels\) && result\.reels\.length > 0[\s\S]*\? result\.reels[\s\S]*: \[result\.reel\]/,
  );
  assert.match(
    source,
    /primeFeedSessionSnapshot\(ingestMaterialId, ingestedReels, ingestedReel\.reel_id, activeSettings\)/,
  );
});

test("URL ingestion primes feed snapshots with the current selection contract", () => {
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
    [{ reel_id: "reel-1" }, { reel_id: "reel-2" }],
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
