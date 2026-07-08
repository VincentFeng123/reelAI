# Generation Progress Bar — Design (approved 2026-07-07)

## Goal
Users clicking "Start Learning" currently see nothing while reels generate (~30-60s before the
first reel streams in). Add a frontend-only progress indicator so they know generation is running
and how far along it is. Approach A (user-selected): no backend changes — drive the bar entirely
from the existing SSE reel events.

## Component — `src/components/GenerationProgress.tsx` (new)
Client component. Props: `{ received: number; requested: number }` (rendered only while active —
parent controls mounting). Thin fixed bar at the top of the feed viewport + short label.
Two modes:
- **Indeterminate** (`received === 0`): shimmer animation + status text rotating every ~6s through
  honest stage hints: "Finding videos…", "Cutting clips with AI…". (The frontend does not know the
  true backend stage — phrasing must not pretend otherwise.)
- **Determinate** (`received > 0`): fill = `received / requested`, label "N of M reels ready".
On unmount-worthy completion the parent snaps it to 100% briefly, then removes it (fade-out).
Match the feed page's existing styling idiom (Tailwind classes consistent with the dark theme).

## Wiring — `src/app/feed/page.tsx`
One new state: `generationProgress: { received: number; requested: number } | null`.
- **Start:** wherever the page enters the `"generating"` recovery phase / kicks off
  `generateReelsStream` (state at ~line 873; generating usages ~2673/3224), set
  `{ received: 0, requested: <the actual numReels passed, default 7 (api.ts:781)> }`.
- **Progress:** inside the existing `onReel` callback (api.ts:928-970 → feed page merge path via
  `mergeSessionReels` ~1047), increment `received`.
- **End:** when the stream resolves or errors, snap to complete (see edge cases) then clear to
  `null` after a short fade (~800ms).
- **Render:** `<GenerationProgress …/>` whenever `generationProgress !== null` — this covers both
  the initial generation and the "generating more" path (both stream through the same callback).

## Edge cases
- Backend returns FEWER than requested (`num_reels` is a cap): on stream end with
  `received < requested`, complete the bar — never leave it looking stuck.
- Stream error: clear the bar immediately; existing error UI owns the message.
- Zero reels produced: clear without a fake 100% celebration if `received === 0` on error; on a
  clean empty completion, complete-and-fade is fine.
- SSR-safe (client component; no window access at module scope). Timer for the rotating hints is
  cleaned up on unmount.

## Out of scope (explicitly)
Backend progress events (Option B — per-concept/stage SSE events) deferred; revisit after the live
smoke test shows whether the pre-first-reel indeterminate stretch feels too long.

## Verification
- `npm run typecheck` green.
- Live: in the hosted dev stack (frontend :3001 → backend :8001), click Start Learning and observe
  indeterminate → determinate fill as reels stream → complete/fade. (Manual — repo has no frontend
  test framework.)
