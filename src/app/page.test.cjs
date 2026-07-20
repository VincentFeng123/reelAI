const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const source = fs.readFileSync(path.join(__dirname, "page.tsx"), "utf8");
const globalsSource = fs.readFileSync(path.join(__dirname, "globals.css"), "utf8");
const communitySource = fs.readFileSync(path.join(__dirname, "../components/CommunityReelsPanel.tsx"), "utf8");
const accountSource = fs.readFileSync(path.join(__dirname, "../components/CommunityAccountScreen.tsx"), "utf8");
const accountPageSource = fs.readFileSync(path.join(__dirname, "account/page.tsx"), "utf8");
const billingGateSource = fs.readFileSync(path.join(__dirname, "../components/BillingGateDialog.tsx"), "utf8");
const recallSource = fs.readFileSync(path.join(__dirname, "../components/RecallCheck.tsx"), "utf8");
const fadePresenceSource = fs.readFileSync(path.join(__dirname, "../components/FadePresence.tsx"), "utf8");
const customSelectSource = fs.readFileSync(path.join(__dirname, "../components/CustomSelect.tsx"), "utf8");
const reelCardSource = fs.readFileSync(path.join(__dirname, "../components/ReelCard.tsx"), "utf8");
const primaryActions = source.slice(
  source.indexOf("const sidebarPrimaryActions"),
  source.indexOf("const onAccountMenuSignOut"),
);

test("shell maps paid plan labels and leaves free or unavailable plans unqualified", () => {
  assert.match(source, /plan === "plus"[\s\S]*return "ReelAI Plus"/);
  assert.match(source, /plan === "pro"[\s\S]*return "ReelAI Pro"/);
  assert.match(source, /return "ReelAI"/);
  assert.match(source, /const visibleBillingPlan = billingStatus\?\.plan/);
  assert.match(source, /visibleBillingPlan === "free" \|\| !shellAccount \|\| !shellAccount\.isVerified/);
  assert.match(source, /billingError[\s\S]*"Plan unavailable"[\s\S]*"Loading…"/);
  assert.match(source, /data-reelai-wordmark="true"[\s\S]*text-\[24px\][\s\S]*font-semibold/);
  assert.match(source, /data-plan-suffix="true"[\s\S]*text-\[18px\][\s\S]*font-normal/);
  assert.match(source, /aria-label=\{brandLabel\}/);
  assert.doesNotMatch(source, /font-semibold[^"\n]*">\{brandLabel\}/);
});

test("local demo account is isolated from real auth, billing, and remote sync", () => {
  assert.match(source, /isLocalDemoView\(searchParams\.get\("demo"\), "account"\)/);
  assert.match(source, /useBillingStatus\(demoAccountEnabled \? null : communityAccount\)/);
  assert.match(source, /const shellAccount = demoAccountEnabled \? LOCAL_DEMO_ACCOUNT : communityAccount/);
  assert.match(source, /demoAccountEnabled \? LOCAL_DEMO_BILLING_STATUS : liveBilling\.status/);
  assert.match(source, /if \(demoAccountEnabled \|\| !scopedAccountId\) \{[\s\S]*return;/);
  assert.match(source, /router\.push\("\/\?demo=account"\)/);
  assert.match(source, /router\.push\("\/feed\?demo=player&return_tab=search"\)/);
  assert.match(source, /router\.push\("\/feed\?demo=quiz&return_tab=search"\)/);
  assert.match(source, /Open demo quiz/);
  assert.match(source, /demoMode=\{demoAccountEnabled\}/);
  assert.match(source, /demoAccountEnabled \? "Exit demo" : "Log out"/);
  assert.match(source, /<CommunityReelsPanel[\s\S]{0,220}?demoMode=\{demoAccountEnabled\}/);
  assert.match(communitySource, /const localDemoEnabled = demoMode && LOCAL_DEMO_AVAILABLE/);
  assert.match(communitySource, /if \(localDemoEnabled\) \{[\s\S]*setCommunityAccount\(LOCAL_DEMO_ACCOUNT\)[\s\S]*setOwnedSets\(demoSets\)[\s\S]*return \(\) =>/);
  assert.match(communitySource, /localDemoEnabled[\s\S]*\? buildLocalDemoSet\([\s\S]*: await updateCommunitySet\(/);
  assert.match(communitySource, /localDemoEnabled[\s\S]*\? buildLocalDemoSet\([\s\S]*: await createCommunitySet\(/);
  assert.match(communitySource, /if \(!localDemoEnabled\) \{[\s\S]{0,260}?await deleteCommunitySet/);
  assert.match(communitySource, /if \(!localDemoEnabled\) \{\s*try \{\s*durationSec = await fetchCommunityReelDuration/);
  assert.match(communitySource, /nextParams\.set\("return_demo", "account"\)/);
});

test("sidebar exposes only supported primary destinations", () => {
  assert.match(primaryActions, /label: "New search"/);
  assert.match(primaryActions, /label: "Community Reels"/);
  assert.match(primaryActions, /label: "Your Sets"/);
  assert.match(primaryActions, /label: "Community Reels"[\s\S]{0,260}?data-sidebar-icon="community-sets"[\s\S]{0,180}?fa-solid fa-users/);
  assert.match(primaryActions, /label: "Your Sets"[\s\S]{0,260}?data-sidebar-icon="your-sets"[\s\S]{0,180}?fa-solid fa-folder/);
  assert.match(primaryActions, /\.\.\.\(shellAccount \? \[\{/);
  assert.doesNotMatch(primaryActions, /Search History|label: "Settings"/);
  assert.match(source, /data-mobile-drawer="true"[\s\S]*bg-black text-white/);
  assert.match(source, /className="relative z-30 hidden min-h-0 bg-black text-white transition-\[width\]/);
  assert.match(source, /const DESKTOP_SIDEBAR_EXPANDED_PX = 272/);
  assert.match(source, /const DESKTOP_SIDEBAR_COLLAPSED_PX = 68/);
  assert.match(source, /desktopSidebarCollapsed[\s\S]*\? DESKTOP_SIDEBAR_COLLAPSED_PX[\s\S]*: DESKTOP_SIDEBAR_EXPANDED_PX/);
  assert.match(source, /transition-\[grid-template-columns\] duration-300 ease-out motion-reduce:transition-none lg:grid/);
  assert.match(source, /transition-\[width\] duration-300 ease-out motion-reduce:transition-none lg:flex/);
  assert.match(source, /const chatGPTCollapsedSidebarContent/);
  const collapsedSidebar = source.slice(
    source.indexOf("const chatGPTCollapsedSidebarContent"),
    source.indexOf("const shellIsInert"),
  );
  assert.match(collapsedSidebar, /src="\/reelai-mark-white-2\.svg"/);
  assert.match(collapsedSidebar, /data-collapsed-sidebar-brand="true"[\s\S]*h-5 w-5[\s\S]*opacity-100[\s\S]*transition-opacity duration-150[\s\S]*group-hover:opacity-0/);
  assert.match(collapsedSidebar, /data-collapsed-sidebar-expand-icon="true"[\s\S]*opacity-0[\s\S]*transition-opacity duration-150[\s\S]*group-hover:opacity-100/);
  assert.match(source, /style=\{\{ left: `calc\(\$\{desktopSidebarWidthPx\}px \+ 2\.5rem\)` \}\}/);
  assert.match(source, /transition-\[left,opacity\] duration-300 ease-out motion-reduce:transition-none lg:block/);
  assert.match(source, /w-\[min\(272px,86vw\)\]/);
  assert.match(source, /w-\[256px\][\s\S]*bottom-\[calc\(100%\+10px\)\] left-0/);
  assert.match(source, /action\.isActive \? "bg-\[#2a2a2a\] text-white"/);
  assert.doesNotMatch(source, /index === 0 && action\.isActive/);
  assert.match(source, /w-px bg-white\/\[0\.07\]/);
  assert.doesNotMatch(source, /bg-\[#171717\]/);
  assert.match(source, /onClick=\{\(\) => openAuthPage\("login"\)\}[\s\S]*<span>Login<\/span>/);
  assert.match(source, /data-guest-sidebar-login="true"[\s\S]*h-10[\s\S]*gap-2[\s\S]*px-2/);
});

test("collapsed account actions reuse the expanded account menu surface", () => {
  const accountPopover = source.slice(
    source.indexOf("const renderAccountPopover"),
    source.indexOf("const chatGPTSidebarContent"),
  );
  const collapsedSidebar = source.slice(
    source.indexOf("const chatGPTCollapsedSidebarContent"),
    source.indexOf("const shellIsInert"),
  );
  assert.match(accountPopover, /data-shell-account-popover=\{collapsed \? "collapsed" : "expanded"\}[\s\S]*w-\[256px\][\s\S]*bg-\[#202020\][\s\S]*transition-opacity duration-300/);
  assert.match(accountPopover, /collapsed \? "bottom-\[calc\(100%\+24px\)\] left-\[-6px\]" : "bottom-\[calc\(100%\+10px\)\] left-0"/);
  assert.match(source, /\{renderAccountPopover\(\)\}/);
  assert.match(collapsedSidebar, /\{renderAccountPopover\(true\)\}/);
  assert.doesNotMatch(collapsedSidebar, /left-\[calc\(100%\+10px\)\]/);
});

test("shell icons are thin and large centered overlays share one compact frame", () => {
  assert.match(source, /strokeWidth="1\.5"/);
  assert.match(source, /<ShellIcon name="search"/);
  assert.match(source, /<ShellIcon name="panel"/);
  assert.equal((source.match(/\{action\.icon\}/g) || []).length, 2);
  assert.doesNotMatch(source, /case "community"|case "sets"/);
  assert.doesNotMatch(source, /fa-magnifying-glass|fa-table-columns|fa-bars/);
  assert.match(source, /const LARGE_CENTERED_MODAL_PANEL_CLASS = "[^"]*sm:h-\[min\(600px,calc\(100dvh-64px\)\)\][^"]*sm:max-w-\[820px\]"/);
  assert.equal((source.match(/panelClassName=\{LARGE_CENTERED_MODAL_PANEL_CLASS\}/g) || []).length, 2);
  assert.doesNotMatch(source, /AUTH_MODAL_PANEL_CLASS/);
});

test("mobile drawer traps keyboard focus and hands primary navigation to content", () => {
  assert.match(source, /data-mobile-drawer="true"[\s\S]*role="dialog"[\s\S]*aria-modal="true"/);
  assert.match(source, /event\.key === "Escape"[\s\S]*closeMobileSidebar\(\)/);
  assert.match(source, /shouldMoveFocusToContent[\s\S]*mainContentRef\.current\?\.focus\(\)/);
  assert.match(source, /shellModalLogicallyOpen[\s\S]*shellModalPresenceActive/);
  assert.match(source, /mobileSidebarInitialFocusAppliedRef\.current/);
  assert.doesNotMatch(source, /activeElement\.closest\("\[data-mobile-drawer\]"\)/);
});

test("guest authentication uses a dedicated page and keeps Your Sets out of navigation", () => {
  assert.match(source, /const showGuestAuthActions = !shellAccount[\s\S]*visibleSidebarTab === "search"[\s\S]*visibleSidebarTab === "edit"[\s\S]*visibleSidebarTab === "create"/);
  assert.match(source, /data-guest-auth-actions="true"[\s\S]*openAuthPage\("login"\)[\s\S]*Login[\s\S]*openAuthPage\("register"\)[\s\S]*Sign up/);
  assert.match(source, /function buildAccountAuthPath[\s\S]*`\/account\?mode=\$\{mode\}&return_tab=\$\{returnTab\}`/);
  assert.match(source, /router\.push\(buildAccountAuthPath\(mode, accountReturnTabForSidebar\(activeSidebarTab\)\)\)/);
  assert.doesNotMatch(source, /authModalMode|AUTH_MODAL_PANEL_CLASS|presentation="modal"|label="ReelAI account"/);
  assert.doesNotMatch(accountSource, /presentation\?: "page" \| "modal"|isModalPresentation/);
  assert.match(accountSource, /fixed inset-0 h-\[100dvh\][\s\S]*max-w-\[380px\]/);
  assert.match(accountPageSource, /searchParams\.get\("mode"\) === "register"/);
  assert.match(accountPageSource, /onAuthModeChange=\{onAuthModeChange\}/);
});

test("all modal surfaces use the smaller shared radius", () => {
  assert.match(source, /overflow-hidden rounded-\[14px\] bg-\[#202020\]/);
  assert.match(accountSource, /max-w-md rounded-\[14px\] bg-\[#202020\]/);
  assert.match(billingGateSource, /max-w-\[430px\][^"\n]*rounded-\[14px\][^"\n]*bg-\[#202020\]/);
  const communityModalSurfaceClasses = Array.from(
    communitySource.matchAll(/role="dialog"[\s\S]{0,360}?className="([^"]*rounded-\[14px\][^"]*)"/g),
    (match) => match[1],
  );
  assert.equal(communityModalSurfaceClasses.length, 4);
  communityModalSurfaceClasses.forEach((className) => {
    assert.ok(className.includes("bg-[#202020]"));
    assert.doesNotMatch(className, /(?:^|\s)border(?:\s|$)|border-/);
  });
});

test("recall check is a compact borderless reel-width page with stacked surfaced choices", () => {
  assert.match(recallSource, /data-feed-item="recall-check"/);
  assert.match(recallSource, /role="region"/);
  assert.doesNotMatch(recallSource, /ViewportModalPortal|role="dialog"|aria-modal/);
  assert.doesNotMatch(recallSource, /\bborder(?:-[^\s"']+)?\b|h-px|fa-rotate|animate-spin/);
  assert.match(recallSource, /bg-transparent[^"\n]*text-white/);
  assert.match(recallSource, /max-w-\[29\.25rem\]/);
  assert.match(recallSource, /grid grid-cols-1 gap-2/);
  assert.match(recallSource, /bg-white\/\[0\.045\][^"\n]*enabled:hover:bg-white\/\[0\.07\]/);
  assert.match(recallSource, /data-recall-choice="true"/);
  assert.match(recallSource, /Recall Check<\/p>[\s\S]{0,160}?question\.concept_title/);
  assert.match(recallSource, /role="progressbar"[\s\S]{0,420}?h-0\.5/);
  assert.match(recallSource, /Question \{questionIndex \+ 1\} out of \{session\.question_count\}/);
  assert.doesNotMatch(recallSource, /Question \{questionIndex \+ 1\} of \{session\.question_count\} ·/);
  assert.match(recallSource, /correct\s*\? "bg-emerald-500\/20 text-emerald-100"/);
  assert.match(recallSource, /selected\s*\? "bg-red-500\/20 text-red-100"/);
});

test("expanded menus use the canonical modal surface without borders or shadows", () => {
  assert.match(source, /role="dialog"[\s\S]{0,220}?bg-\[#202020\]/);
  assert.equal((source.match(/role="menu"[\s\S]{0,320}?bg-\[#202020\]/g) || []).length, 2);
  assert.match(communitySource, /role="menu"\s*className="overflow-hidden rounded-xl bg-\[#202020\] p-1\.5"/);
  assert.match(customSelectSource, /role="listbox"[\s\S]{0,240}?rounded-2xl bg-\[#202020\] p-1\.5/);
  assert.match(reelCardSource, /role="menu"\s*className="overflow-hidden rounded-2xl bg-\[#202020\] p-1\.5"/);
  assert.doesNotMatch(communitySource, /role="menu"\s*className="[^"]*(?:border|shadow-)/);
  assert.doesNotMatch(customSelectSource, /role="listbox"[\s\S]{0,240}?(?:border|shadow-)/);
  assert.doesNotMatch(reelCardSource, /role="menu"\s*className="[^"]*(?:border|shadow-)/);
  assert.match(source, /role="menu"[\s\S]{0,900}?hover:bg-white\/\[0\.07\]/);
  assert.match(communitySource, /role="menu"[\s\S]{0,900}?hover:bg-white\/\[0\.07\]/);
  assert.match(customSelectSource, /role="listbox"[\s\S]{0,1800}?hover:bg-white\/\[0\.07\]/);
  assert.match(reelCardSource, /role="menu"[\s\S]{0,900}?hover:bg-white\/\[0\.07\]/);
});

test("the hamburger is two lines and only the mobile drawer slides from the left", () => {
  assert.match(source, /case "menu":[\s\S]*M4 8h16M4 16h16/);
  const sidenavKeyframes = globalsSource.slice(
    globalsSource.indexOf("@keyframes mobile-sidenav-in"),
    globalsSource.indexOf("@keyframes mobile-overlay-in"),
  );
  const overlayKeyframes = globalsSource.slice(
    globalsSource.indexOf("@keyframes mobile-overlay-in"),
    globalsSource.indexOf("@keyframes mobile-sheet-in"),
  );
  assert.match(sidenavKeyframes, /translateX\(-100%\)[\s\S]*translateX\(0\)/);
  assert.doesNotMatch(overlayKeyframes, /transform:/);
});

test("compact shell keeps plan-aware branding inline with the hamburger", () => {
  assert.match(source, /data-mobile-sidebar-trigger="true"/);
  assert.match(source, /data-mobile-shell-brand="true"/);
  assert.match(source, /aria-label=\{brandLabel\}[\s\S]{0,120}?data-mobile-shell-brand="true"/);
  assert.match(source, /data-mobile-shell-brand="true"[\s\S]{0,900}?visibleBillingPlan === "plus" \|\| visibleBillingPlan === "pro"/);
  assert.match(source, /visibleBillingPlan === "plus" \? "Plus" : "Pro"/);
  assert.match(source, /style=\{\{ top: "calc\(max\(env\(safe-area-inset-top\), 0px\) \+ 62px\)" \}\}/);
  assert.match(source, /data-mobile-sidebar-trigger="true"[\s\S]{0,900}?mobileSidebarOpen \|\| hideMobileTopControls[\s\S]{0,100}?"opacity-100"/);
  assert.match(source, /data-mobile-shell-brand="true"[\s\S]{0,500}?mobileSidebarOpen \|\| hideMobileTopControls \|\| shellIsInert[\s\S]{0,100}?"opacity-100"/);
});

test("history search is latest-first and title filtering is case-insensitive", () => {
  assert.match(source, /\[\.\.\.history\]\.sort\(\(a, b\) => b\.updatedAt - a\.updatedAt\)/);
  assert.match(source, /historyQuery\.trim\(\)\.toLowerCase\(\)/);
  assert.match(source, /item\.title\.toLowerCase\(\)\.includes\(query\)/);
  assert.match(source, /placeholder="Search\.\.\."/);
});

test("Fast and Slow persist immediately through the scoped settings system", () => {
  assert.match(source, /aria-label="Generation speed"/);
  assert.match(source, /\(\["fast", "slow"\] as const\)/);
  assert.match(source, /saveStudyReelsSettings\(\{[\s\S]*generationMode: nextMode/);
  assert.match(source, /syncSavedSettings\(saved\)/);
  assert.match(source, /const showGenerationSpeedToggle = visibleSidebarTab === "search"/);
  assert.match(source, /aria-hidden=\{shellIsInert \|\| !showGenerationSpeedToggle\}/);
  assert.match(source, /showGenerationSpeedToggle \? "opacity-100" : "opacity-0"/);
  assert.match(source, /showGenerationSpeedToggle && !mobileSidebarOpen && !hideMobileTopControls/);
  assert.match(source, /generationMode === "slow" \? "translate-x-full" : "translate-x-0"/);
  assert.match(source, /className="relative flex h-9 w-\[218px\][^"]*bg-\[#181818\]/);
  assert.match(source, /data-generation-speed-indicator/);
  assert.match(source, /-left-0\.5 -top-0\.5 h-10 w-\[calc\(50%\+2px\)\][^`]*bg-\[#242424\] transition-transform duration-\[360ms\] ease-in-out/);
});

test("shell surfaces use fade-only presence motion and no drop shadows", () => {
  assert.match(source, /const UI_FADE_MS = 340/);
  assert.match(source, /isVisible \? "opacity-100" : "opacity-0"/);
  assert.match(source, /setIsVisible\(false\)[\s\S]*setIsRendered\(false\)/);
  assert.match(source, /transition-opacity duration-300/);
  assert.doesNotMatch(source, /shadow-/);
  assert.match(globalsSource, /mobile-sidenav-in 320ms[\s\S]*mobile-sidenav-out 320ms/);
  assert.match(communitySource, /<FadePresence show=\{Boolean\(draftActionConfirmModal\)\}>/);
  assert.match(communitySource, /modalVisible \? "opacity-100" : "opacity-0"/);
  assert.doesNotMatch(communitySource, /animate-featured-slide|group-hover:scale|transition-\[max-height/);
  assert.match(fadePresenceSource, /const DEFAULT_FADE_EXIT_MS = 340/);
  assert.match(fadePresenceSource, /setIsVisible\(false\)[\s\S]*setIsRendered\(false\)/);
  assert.match(fadePresenceSource, /requestAnimationFrame\(\(\) => \{\s*revealFrame = window\.requestAnimationFrame\(\(\) => setIsVisible\(true\)\)/);
});

test("global selection defaults every surface off and opts in only text", () => {
  const readableTextStart = globalsSource.indexOf(":where(\n  p,");
  const editableTextStart = globalsSource.indexOf(":where(\n  input:not([type]),");
  const nonTextStart = globalsSource.indexOf(":where(\n  a,", editableTextStart);
  const mediaDragStart = globalsSource.indexOf(":where(img, picture, svg, video, audio, canvas, iframe, object, embed)");
  assert.ok(readableTextStart >= 0 && editableTextStart > readableTextStart);
  assert.ok(nonTextStart > editableTextStart && mediaDragStart > nonTextStart);

  const universalSelection = globalsSource.slice(globalsSource.indexOf("* {"), globalsSource.indexOf("::-webkit-scrollbar"));
  const readableTextSelection = globalsSource.slice(readableTextStart, editableTextStart);
  const editableTextSelection = globalsSource.slice(editableTextStart, nonTextStart);
  const nonTextSelection = globalsSource.slice(nonTextStart, mediaDragStart);
  assert.match(universalSelection, /-webkit-tap-highlight-color: transparent;[\s\S]*-webkit-user-select: none;[\s\S]*user-select: none;/);
  assert.match(readableTextSelection, /p,[\s\S]*h1,[\s\S]*blockquote,[\s\S]*code,[\s\S]*div:not\(:empty\):not\(:has\(\*\)\)[\s\S]*span:not\(:empty\):not\(:has\(\*\)\)/);
  assert.match(readableTextSelection, /-webkit-user-select: text;[\s\S]*user-select: text;/);
  assert.match(editableTextSelection, /input:not\(\[type\]\)[\s\S]*input\[type="text"\][\s\S]*textarea[\s\S]*\[role="textbox"\][\s\S]*contenteditable="plaintext-only"/);
  assert.match(editableTextSelection, /-webkit-user-select: text;[\s\S]*user-select: text;/);
  assert.match(nonTextSelection, /a \*,[\s\S]*button \*,[\s\S]*\[role="tab"\] \*,[\s\S]*\[role="menuitem"\] \*,[\s\S]*\[aria-hidden="true"\] \*,[\s\S]*picture \*,[\s\S]*svg \*/);
  assert.match(nonTextSelection, /-webkit-user-select: none;[\s\S]*user-select: none;/);
  assert.doesNotMatch(nonTextSelection, /focus-visible|outline|box-shadow/);
  assert.match(globalsSource.slice(mediaDragStart), /-webkit-user-drag: none/);
  assert.match(globalsSource, /::selection \{[\s\S]*background: #fff;[\s\S]*color: #000;/);
});

test("settings query links and legacy tabs bridge into overlays", () => {
  assert.match(source, /searchParams\.get\("settings"\)/);
  assert.match(source, /nextQuery\.set\("settings", section\)/);
  assert.match(source, /forcedSidebarTab === "settings"/);
  assert.match(source, /forcedSidebarTab === "history"/);
  assert.match(source, /nextQuery\.set\("settings", "search"\)/);
  assert.match(source, /const nextTab = forcedSidebarTab \?\? \(forcedCommunitySetId \? "community" : savedTab \?\? "search"\)/);
  assert.match(source, /setActiveCommunitySetId\(forcedCommunitySetId \?\? savedCommunitySetId\)/);
});

test("modal shell traps focus, closes with Escape, restores focus, and makes the shell inert", () => {
  assert.match(source, /event\.key === "Escape"/);
  assert.match(source, /event\.key !== "Tab"/);
  assert.match(source, /returnFocus\?\.isConnected[\s\S]*returnFocus\.focus\(\)/);
  assert.match(source, /aria-modal="true"/);
  assert.match(source, /inert=\{shellIsInert\}/);
  assert.match(source, /Keep Editing/);
  const closeButtonClasses = Array.from(source.matchAll(/aria-label="Close"\s+className="([^"]+)"/g), (match) => match[1]);
  assert.ok(closeButtonClasses.length >= 2);
  closeButtonClasses.forEach((className) => assert.match(className, /hover:bg-white\/\[0\.07\]/));
});

test("settings exit prompt preserves the draft and routes each auth mode", () => {
  assert.match(source, /aria-hidden=\{settingsClosePrompt\}[\s\S]*<SettingsPanel[\s\S]*aria-hidden=\{!settingsClosePrompt\}/);
  assert.match(source, /onOpenAuth=\{requestOpenAuth\}/);
  assert.match(source, /window\.setTimeout\(\(\) => \{[\s\S]*router\.push\(buildAccountAuthPath\(authMode, "settings"\)\)/);
  assert.match(source, /finishCloseSettings\(pendingSettingsAuthMode \?\? undefined\)/);
});

test("history stars render beside titles while search results stay latest-first", () => {
  assert.match(source, /historySorted\.slice\(0, 40\)\.map/);
  const historyTitle = source.slice(
    source.indexOf("onClick={() => openMaterialFeed(entry.materialId)}"),
    source.indexOf('<div data-history-actions="true"'),
  );
  assert.match(historyTitle, /entry\.starred/);
  assert.match(historyTitle, /<ShellIcon name="star" filled/);
  assert.match(source, /filteredHistory\.map[\s\S]*entry\.starred[\s\S]*<ShellIcon name="star" filled/);
});

test("history information scrolls to the modal edge without a blocking footer", () => {
  const infoModal = source.slice(
    source.indexOf("open={selectedHistoryInfoItem !== null}"),
    source.indexOf("</ShellModal>", source.indexOf("open={selectedHistoryInfoItem !== null}")),
  );
  assert.match(infoModal, /flex max-h-\[calc\(100dvh-24px\)\]/);
  assert.match(infoModal, /min-h-0 flex-1[\s\S]*overflow-y-auto overscroll-contain/);
  assert.doesNotMatch(infoModal, /max-h-\[70vh\]/);
  assert.doesNotMatch(infoModal, />\s*Close\s*</);
  assert.equal((infoModal.match(/onClick=\{closeHistoryInfo\}/g) || []).length, 1);
});

test("the seamless top gradient is shared by Community and Your Sets without a clipped content seam", () => {
  const topFadeCss = globalsSource.slice(
    globalsSource.indexOf(".top-nav-fade"),
    globalsSource.indexOf(".community-featured-grid"),
  );
  assert.match(topFadeCss, /\.top-nav-fade \{[\s\S]*background: transparent/);
  assert.match(topFadeCss, /\.top-nav-fade::before[\s\S]*inset: -3\.5rem -2\.5rem -5\.5rem[\s\S]*background: rgb\(var\(--top-nav-fade-rgb\)\)[\s\S]*backdrop-filter: blur\(8px\)[\s\S]*0\.995\) 18%[\s\S]*0\.96\) 32%[\s\S]*0\.86\) 46%[\s\S]*0\.68\) 60%[\s\S]*0\.46\) 72%[\s\S]*0\.28\) 82%[\s\S]*0\.13\) 90%[\s\S]*transparent 100%/);
  assert.doesNotMatch(topFadeCss, /background: linear-gradient|backdrop-filter: blur\(12px\)|backdrop-filter: blur\(22px\)|inset: -0\.75rem/);
  assert.doesNotMatch(globalsSource, /\.top-nav-fade::after/);
  assert.match(globalsSource, /\.top-nav-fade-charcoal[\s\S]*32 32 32/);
  assert.doesNotMatch(globalsSource, /\.top-nav-fade-community/);
  assert.doesNotMatch(topFadeCss, /box-shadow|border:/);
  assert.doesNotMatch(source, /top-nav-fade pointer-events-none fixed inset-x-0 top-0/);
  assert.match(source, /top-nav-fade top-nav-fade-charcoal sticky top-0/);
  assert.match(communitySource, /data-top-chrome="community-directory" className="top-nav-fade absolute inset-x-0 top-0 z-20 w-full shrink-0"/);
  assert.match(communitySource, /data-top-chrome="community-detail-back"[\s\S]{0,100}?className="top-nav-fade sticky top-0 z-20 w-full/);
  assert.match(communitySource, /data-top-chrome="community-management" className="top-nav-fade absolute inset-x-0 top-0 z-20 w-full shrink-0"/);
  assert.match(communitySource, /ref=\{communityScrollRef\} className="[^"]*pt-\[4\.5rem\][^"]*md:pt-20/);
  assert.match(communitySource, /shouldShowEditSetGrid \? "pt-\[5\.5rem\] md:pt-24" : "pt-\[3\.75rem\] md:pt-16"/);
  assert.doesNotMatch(communitySource, /data-community-detail-header[\s\S]{0,100}?top-nav-fade/);
  assert.doesNotMatch(communitySource, /className=\{shouldShowEditSetGrid \? "top-nav-fade/);
});

test("Community and Your Sets keep one-row headers with collapsible compact search", () => {
  assert.equal(
    (communitySource.match(/className="relative flex min-h-12 w-full items-center gap-3 px-1 sm:px-2 md:px-3"/g) || []).length,
    2,
  );
  assert.equal((communitySource.match(/data-compact-search-expanded=/g) || []).length, 2);
  assert.equal((communitySource.match(/transition-\[width\] duration-\[440ms\] ease-in-out/g) || []).length, 2);
  assert.equal((communitySource.match(/w-\[clamp\(7\.5rem,48vw,17rem\)\]/g) || []).length, 2);
  assert.equal((communitySource.match(/absolute right-1 top-1\/2 z-10 h-10 shrink-0 -translate-y-1\/2 overflow-hidden/g) || []).length, 2);
  assert.equal((communitySource.match(/flex min-w-0 flex-1 items-center pl-4 sm:pl-5 lg:pl-0/g) || []).length, 2);
  assert.match(
    communitySource,
    /isYourSetsMode && shouldShowEditSetGrid[\s\S]{0,160}?\? "flex min-w-0 flex-1 items-center pl-4 sm:pl-5 lg:pl-0"/,
  );
  assert.equal(
    (communitySource.match(/data-your-sets-empty-state className="mt-4 pl-5 text-sm text-white\/66 sm:pl-7 md:pl-\[2\.375rem\] lg:pl-\[1\.375rem\]"/g) || []).length,
    2,
  );
  assert.equal((communitySource.match(/className="hidden w-\[23rem\] shrink-0 xl:block"/g) || []).length, 2);
  assert.match(communitySource, /setIsCompactCommunitySearchOpen\(true\)[\s\S]{0,180}?compactCommunitySearchInputRef\.current\?\.focus\(\)/);
  assert.match(communitySource, /setIsCompactYourSetsSearchOpen\(true\)[\s\S]{0,180}?compactYourSetsSearchInputRef\.current\?\.focus\(\)/);
  assert.match(communitySource, /ref=\{compactCommunitySearchInputRef\}[\s\S]{0,1200}?pl-4 pr-9/);
  assert.match(communitySource, /ref=\{compactYourSetsSearchInputRef\}[\s\S]{0,1200}?pl-4 pr-9/);
  assert.equal((communitySource.match(/rounded-full bg-transparent text-white\/72 transition-\[background-color,color,opacity\]/g) || []).length, 2);
  assert.equal((communitySource.match(/data-community-search-icon/g) || []).length, 2);
  assert.equal((communitySource.match(/data-your-sets-search-icon/g) || []).length, 2);
  assert.match(communitySource, /className="pointer-events-none absolute left-4 top-1\/2 z-10 h-\[18px\] w-\[18px\] -translate-y-1\/2 text-white\/55"/);
  assert.doesNotMatch(communitySource, /data-top-chrome="community-directory"[\s\S]{0,220}?flex-col/);
  assert.doesNotMatch(communitySource, /data-top-chrome="community-management"[\s\S]{0,220}?flex-col/);
  assert.equal(
    (communitySource.match(/lg:w-11\/12 xl:w-4\/5 2xl:w-full 2xl:max-w-5xl/g) || []).length,
    2,
  );
});

test("community sets use a compact featured carousel and flat directory results", () => {
  assert.match(communitySource, /FEATURED_CAROUSEL_CONTENT_MIN_HEIGHT_FALLBACK = 250/);
  assert.match(communitySource, /mx-auto min-h-0 w-full flex-1 overflow-hidden lg:w-11\/12 xl:w-4\/5 2xl:w-full 2xl:max-w-5xl/);
  assert.match(communitySource, /data-community-featured-carousel/);
  assert.match(communitySource, /ref=\{communityScrollRef\} className="balanced-scroll-gutter h-full min-h-0 space-y-4 overflow-y-auto pb-6 pt-\[4\.5rem\] md:space-y-5 md:pb-8 md:pt-20 lg:pb-10"/);
  assert.match(communitySource, /data-community-search-icon/);
  assert.match(communitySource, /data-featured-image-target[\s\S]{0,220}?rounded-\[10px\]/);
  assert.doesNotMatch(communitySource, /trending this week/);

  const directory = communitySource.slice(
    communitySource.indexOf("data-community-directory"),
    communitySource.indexOf("</section>", communitySource.indexOf("data-community-directory")),
  );
  assert.doesNotMatch(directory, /backdrop-blur|rounded-2xl/);
  assert.match(directory, /data-community-set-row/);
  assert.match(directory, /bg-transparent[\s\S]*hover:bg-white\/\[0\.07\]/);
  assert.doesNotMatch(directory, /bg-\[#151515\]/);
  assert.match(directory, /h-11 w-11/);
});

test("Community and Your Sets omit Beta badges while owned sets use a compact sortable list", () => {
  assert.doesNotMatch(communitySource, />Beta<|>BETA</);
  const yourSetsListStart = communitySource.indexOf("{shouldShowEditSetGrid ? (");
  const yourSetsListEnd = communitySource.indexOf("{shouldShowEditSetForm ? (", yourSetsListStart);
  assert.ok(yourSetsListStart >= 0 && yourSetsListEnd > yourSetsListStart);
  const yourSetsList = communitySource.slice(yourSetsListStart, yourSetsListEnd);
  assert.match(yourSetsList, /<section className="px-0 pb-1">[\s\S]{0,100}?<div className="relative pb-2">/);
  assert.match(yourSetsList, /data-your-sets-list[\s\S]{0,240}?data-your-sets-list-header[\s\S]{0,320}?grid-cols-\[1\.75rem_minmax\(0,1fr\)_7rem_3rem\][\s\S]{0,5000}?data-your-sets-modified-sort/);
  assert.match(yourSetsList, /data-your-sets-modified-sort[\s\S]{0,500}?setYourSetsModifiedSortDirection[\s\S]{0,620}?font-normal[\s\S]{0,260}?<span className="truncate">Modified<\/span>[\s\S]{0,160}?data-your-sets-sort-arrow/);
  assert.doesNotMatch(yourSetsList, /data-your-sets-modified-sort[\s\S]{0,800}?(?:hover|focus-visible):font-semibold/);
  assert.match(yourSetsList, /yourSetsModifiedSortDirection === "oldest" \? "rotate-180" : "rotate-0"/);
  assert.match(communitySource, /useState<"newest" \| "oldest">\("newest"\)/);
  assert.match(communitySource, /parseTimestampMs\(left\.updatedAt\)[\s\S]{0,300}?parseTimestampMs\(right\.updatedAt\)[\s\S]{0,800}?yourSetsModifiedSortDirection === "newest"/);
  assert.match(yourSetsList, /grid-cols-\[1\.75rem_minmax\(0,1fr\)_7rem_3rem\][\s\S]{0,180}?sm:grid-cols-\[1\.75rem_minmax\(0,1fr\)_10rem_3\.25rem\]/);
  assert.match(yourSetsList, /<li[\s\S]{0,120}?data-your-set-row[\s\S]{0,240}?min-h-\[3rem\][\s\S]{0,180}?bg-transparent[\s\S]{0,140}?sm:min-h-\[3\.25rem\]/);
  assert.match(yourSetsList, /data-your-set-surface[\s\S]{0,220}?left-7 right-0[\s\S]{0,140}?rounded-\[12px\] bg-transparent[\s\S]{0,160}?group-hover:bg-white\/\[0\.07\]/);
  assert.match(yourSetsList, /col-start-2 col-span-2[\s\S]{0,180}?rounded-l-\[12px\] px-2 py-1\.5/);
  assert.match(yourSetsList, /data-your-set-thumbnail[\s\S]{0,260}?h-9 w-9[\s\S]{0,180}?sm:h-10 sm:w-10[\s\S]{0,420}?src=\{set\.thumbnailUrl \|\| FALLBACK_THUMBNAIL_URL\}[\s\S]{0,300}?object-cover/);
  assert.match(yourSetsList, /data-your-set-title[\s\S]{0,180}?title=\{set\.title\}[\s\S]{0,180}?min-w-0 flex-1 truncate/);
  assert.match(yourSetsList, /data-your-set-modified[\s\S]{0,180}?title=\{modifiedLabel\}[\s\S]{0,180}?min-w-0 truncate/);
  assert.match(yourSetsList, /onClick=\{\(\) => onOpenEditableSet\(set\.id\)\}[\s\S]{0,100}?aria-label=\{`Open \$\{set\.title\}`\}/);
  assert.match(yourSetsList, /formatModifiedLabel\(set, relativeTimeNowMs\)/);
  assert.match(yourSetsList, /data-your-set-actions="true"[\s\S]{0,900}?aria-haspopup="menu"[\s\S]{0,100}?aria-expanded=\{isActionsMenuOpen\}/);
  assert.match(yourSetsList, /data-your-set-more-icon[\s\S]{0,220}?h-5 w-5[\s\S]*<circle cx="6" cy="12" r="1\.15"[\s\S]*<circle cx="12" cy="12" r="1\.15"[\s\S]*<circle cx="18" cy="12" r="1\.15"/);
  assert.match(yourSetsList, /overflow-hidden rounded-xl bg-\[#202020\] p-1\.5/);
  assert.equal((yourSetsList.match(/role="menuitem"/g) || []).length, 3);
  assert.doesNotMatch(yourSetsList, /data-your-sets-grid|data-your-set-card|data-your-set-visual|data-your-set-image|data-your-set-blur-gradient|data-your-set-folder-icon|aspect-\[4\/3\]|data-inline-create-set|Your Created Sets|bg-\[#181818\]|\bp-4\b|\bpx-4\b|\bpy-3\b/);
  assert.match(communitySource, /shouldShowEditSetGrid && canManageYourSets[\s\S]{0,220}?data-floating-create-set[\s\S]{0,400}?bottom-\[max\(40px,env\(safe-area-inset-bottom\)\)\] right-\[max\(40px,env\(safe-area-inset-right\)\)\][\s\S]{0,160}?h-14 w-14[\s\S]{0,160}?bg-white text-black/);
  assert.doesNotMatch(communitySource, /data-floating-create-set[\s\S]{0,500}?(?:border|shadow)/);
});

test("Your Sets reveals stable custom selection controls and bulk deletes selected rows", () => {
  const yourSetsListStart = communitySource.indexOf("{shouldShowEditSetGrid ? (");
  const yourSetsListEnd = communitySource.indexOf("{shouldShowEditSetForm ? (", yourSetsListStart);
  const yourSetsList = communitySource.slice(yourSetsListStart, yourSetsListEnd);
  assert.match(communitySource, /const \[selectedEditableSetIds, setSelectedEditableSetIds\] = useState<string\[]>\(\[\]\)/);
  assert.match(communitySource, /setSelectedEditableSetIds\(\(prev\) => \([\s\S]{0,260}?prev\.includes\(normalized\)[\s\S]{0,220}?\[\.\.\.prev, normalized\]/);
  assert.match(yourSetsList, /data-your-sets-name-action[\s\S]{0,220}?transition-\[height\] duration-300 ease-out[\s\S]{0,180}?selectedEditableSetIds\.length > 0 \? "h-7" : "h-5"/);
  assert.match(yourSetsList, /aria-hidden=\{selectedEditableSetIds\.length > 0\}[\s\S]{0,360}?pointer-events-none -translate-y-1 opacity-0[\s\S]{0,180}?Name/);
  assert.match(yourSetsList, /data-your-sets-delete-selected[\s\S]{0,220}?onClick=\{onRequestDeleteSelectedEditableSets\}[\s\S]{0,260}?tabIndex=\{selectedEditableSetIds\.length > 0 \? 0 : -1\}[\s\S]{0,140}?aria-hidden=\{selectedEditableSetIds\.length === 0\}/);
  assert.match(yourSetsList, /transition-\[transform,opacity,background-color,border-color,color\][\s\S]{0,160}?\[transition-duration:220ms,140ms,150ms,150ms,150ms\][\s\S]{0,420}?translate-y-0 opacity-100 \[transition-delay:0ms,220ms,0ms,0ms,0ms\] disabled:opacity-50[\s\S]{0,160}?pointer-events-none translate-y-2 opacity-0 \[transition-delay:0ms\]/);
  const deleteSelectedStart = yourSetsList.indexOf("data-your-sets-delete-selected");
  const deleteSelectedEnd = yourSetsList.indexOf("</button>", deleteSelectedStart);
  const deleteSelectedButton = yourSetsList.slice(deleteSelectedStart, deleteSelectedEnd);
  assert.match(deleteSelectedButton, /rounded-full border border-red-400\/55[\s\S]{0,180}?text-red-300/);
  assert.match(deleteSelectedButton, /<span className="truncate">Delete<\/span>/);
  assert.equal((deleteSelectedButton.match(/<span/g) || []).length, 1);
  assert.match(yourSetsList, /data-your-set-checkbox[\s\S]{0,180}?data-selected=\{isSelected \? "true" : "false"\}[\s\S]{0,180}?aria-checked=\{isSelected\}/);
  assert.match(yourSetsList, /data-your-set-surface[\s\S]{0,220}?left-7 right-0/);
  assert.match(yourSetsList, /isSelected[\s\S]{0,180}?\? "opacity-100"[\s\S]{0,180}?: "opacity-0 group-hover:opacity-100 group-focus-within:opacity-100"/);
  assert.match(yourSetsList, /role="checkbox"/);
  assert.match(yourSetsList, /onToggleEditableSetSelection\(set\.id\)/);
  assert.match(yourSetsList, /col-start-2 col-span-2/);
  assert.match(yourSetsList, /data-your-set-actions="true"[\s\S]{0,180}?col-start-4/);
  assert.match(communitySource, /setDeleteSetConfirmModal\(\{[\s\S]{0,180}?setIds: targetIds/);
  assert.match(communitySource, /normalizedSetIds\.length === 1[\s\S]{0,180}?deleteCommunitySet[\s\S]{0,180}?deleteCommunitySets\(\{ setIds: normalizedSetIds \}\)/);
  assert.match(communitySource, /deleteSetConfirmModal\.setIds\.length === 1[\s\S]{0,500}?Delete \$\{deleteSetConfirmModal\.setIds\.length\} selected sets/);
  assert.match(communitySource, /onDeleteEditableSets\(deleteSetConfirmModal\.setIds\)/);
});

test("the set editor is borderless, flat, and keeps one clear completion guide", () => {
  const setEditorStart = communitySource.indexOf('data-create-set-view="true"');
  const setEditorEnd = communitySource.indexOf("</section>", setEditorStart);
  assert.ok(setEditorStart >= 0 && setEditorEnd > setEditorStart);
  const setEditor = communitySource.slice(setEditorStart, setEditorEnd);
  const positiveBorderUtilities = Array.from(
    setEditor.matchAll(/(?:^|[\s"'`])((?:[a-z-]+:)*border(?:-(?!0(?=[\s"'`}]))[^\s"'`}]+)?)(?=[\s"'`}])/gm),
    (match) => match[1],
  );

  assert.deepEqual(positiveBorderUtilities, []);
  assert.doesNotMatch(setEditor, /backdrop-blur|max-h-\[320px\]|<svg viewBox="0 0 88 88"/);
  assert.match(setEditor, /Set details/);
  assert.match(setEditor, /Paste links from \{SUPPORTED_PLATFORMS_LABEL\}/);
  assert.match(setEditor, /data-create-set-actions="true"/);
  assert.match(setEditor, /<div key=\{row\.id\} data-create-set-reel-row="true">/);
  assert.doesNotMatch(setEditor, /data-create-set-reel-row="true"[^>]*className/);
  assert.match(setEditor, /index < parsedDraftReels\.length - 1[\s\S]{0,220}?data-create-set-reel-divider="true"[\s\S]{0,140}?h-px bg-white\/\[0\.08\]/);
  assert.match(setEditor, /data-create-set-reel-preview="true"[\s\S]{0,220}?h-\[320px\][\s\S]{0,120}?sm:h-\[360px\]/);
  assert.match(setEditor, />\s*Clear\s*<\/[\s\S]{0,700}?bg-white\/\[0\.08\][\s\S]{0,700}?bg-white text-black/);
  assert.match(setEditor, /Save draft[\s\S]{0,700}?Post set/);
  assert.match(setEditor, /remainingPreviewRequirements\.length > 0/);
  assert.doesNotMatch(setEditor, />Reels<\/p>[\s\S]{0,500}?>Status<\/p>/);

  assert.match(communitySource, /data-create-set-back="true"[\s\S]{0,260}?absolute left-1[\s\S]{0,160}?sm:left-2 md:left-1\.5 lg:left-0\.5/);
  assert.match(communitySource, /isYourSetsMode && shouldShowEditSetForm \? "lg:pl-11" : ""/);
  assert.match(communitySource, /data-create-set-back="true"[\s\S]{0,300}?hover:bg-white\/\[0\.07\]/);
  assert.doesNotMatch(communitySource, /data-create-set-back="true"[\s\S]{0,260}?\bborder(?:\s|-(?!0\b))/);

  const dualRangeCss = globalsSource.slice(
    globalsSource.indexOf(".dual-range-input"),
    globalsSource.indexOf(".playback-speed-range"),
  );
  assert.equal((dualRangeCss.match(/border: 0/g) || []).length, 2);
  assert.equal((dualRangeCss.match(/box-shadow: none/g) || []).length, 2);
  assert.doesNotMatch(dualRangeCss, /border: [1-9]|box-shadow: 0 0/);
});

test("community set details use one centered page composition", () => {
  assert.doesNotMatch(communitySource, /detailBannerPortal|detailContentTopPadding|isDetailBannerCompact/);
  assert.match(communitySource, /data-community-detail-view[\s\S]*max-w-4xl/);
  assert.match(communitySource, /data-community-detail-header/);
  assert.match(communitySource, /data-community-detail-media/);
  assert.match(communitySource, /data-community-detail-about/);
  assert.match(communitySource, /top-nav-fade sticky top-0[\s\S]*Community Sets/);
  assert.match(communitySource, /activeDetailCarouselReel[\s\S]*iframe[\s\S]*selectedDirectorySet\.thumbnailUrl/);
  assert.match(communitySource, /onClick=\{openSelectedSetReelsInFeed\}/);
});

test("community set detail manages focus, scroll, and playable media without hidden controls", () => {
  assert.match(communitySource, /inert=\{isDirectoryDetailOpen \|\| shouldSuppressDirectoryDuringRestore\}/);
  assert.match(communitySource, /inert=\{!isDirectoryDetailOpen\}/);
  assert.match(communitySource, /detailContentScrollRef\.current\.scrollTop = 0/);
  assert.match(communitySource, /detailBackButtonRef\.current\?\.focus\(\{ preventScroll: true \}\)/);
  assert.match(communitySource, /focusTarget\?\.focus\(\{ preventScroll: true \}\)/);
  assert.doesNotMatch(communitySource, /DETAIL_REEL_CAROUSEL_INTERVAL_MS/);
  assert.match(communitySource, /data-community-set-id=\{set\.id\}/);
});

test("featured carousel uses a shorter frame with balanced perimeter spacing", () => {
  assert.match(communitySource, /data-community-featured-carousel[\s\S]{0,180}?p-3 pb-9[\s\S]{0,120}?sm:p-4 sm:pb-10/);
  assert.match(communitySource, /flex w-full min-w-0 max-w-2xl flex-col items-start text-left"/);
  assert.match(communitySource, /community-featured-image-frame absolute bottom-0 right-0/);
  assert.match(communitySource, /bottom-3[\s\S]{0,160}?sm:bottom-4/);
  assert.doesNotMatch(communitySource, /md:pl-6 lg:pl-8|right-6[\s\S]*lg:right-10/);
  assert.match(globalsSource, /@media \(min-width: 768px\) \{[\s\S]{0,120}?\.community-featured-grid \{[\s\S]{0,80}?min-height: 250px/);
  assert.match(globalsSource, /\.community-featured-image-frame \{[\s\S]*width: 82%/);
  assert.match(globalsSource, /@media \(hover: none\), \(pointer: coarse\)[\s\S]*community-featured-next-control/);
});
