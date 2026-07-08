# Raw Practice Output — Design (user-approved 2026-07-08)

## Goal
The feed must play EXACTLY what the practice (Gemini) engine cuts — every topic clip, any length,
any position in the video — with no backend knob dropping or reshaping clips. The backend PROCESS
and its non-generation surfaces (community, auth, iOS endpoints, feed persistence, generations)
stay. This supersedes the duration-contract enforcement added by the final-review fix wave
(finding #2) — an explicit, user-decided product reversal, and it retires the fast/slow +
clip-duration settings as user-facing knobs.

## What currently reshapes engine output (all to be neutralized)
1. **ingest_topic duration gate** (`pipeline.py`: skip clips > target_max+8) — REMOVE. Persist
   whole-topic clips of any length. Log nothing-dropped (gate gone entirely).
2. **ingest_topic relevance drop** (`bridge.filter_by_query` drops clips with score <= floor) —
   STOP DROPPING on the material path: annotate `clip["score"]` (ordering signal for the feed)
   but persist ALL clips the engine returns. (Practice's gemini engine segments every topic and
   shows them all.) `ingest_search`/`ingest_url`/`ingest_topic_cut` keep their existing semantics
   — they are different endpoints; only the material path (`ingest_topic`) goes raw.
3. **Per-clip budget truncation** (`generate_reels` passes `max_reels=remaining`) — pass
   `max_reels=None`. The COST guardrails stay: `MATERIAL_MAX_VIDEOS_PER_CONCEPT=3`,
   `MATERIAL_GEN_MAX_VIDEOS=12` still bound Supadata/Gemini spend per generation; `num_reels`
   still shapes the RESPONSE page via `_finalize_generated_reels` but no longer stops persistence
   mid-video — the feed reads all persisted rows.
4. **Response-layer duration window** (`main.py` `_shape_request_page_reels`: default 19–55s
   filter + relaxed backfill; `_filter_reels_by_*duration*/clip_range*` on the probe/page paths) —
   neutralize for feed/page shaping so any clip length is served. Keep the probe's can_generate
   logic functional (it must not start failing because filters vanished — simplify honestly).
5. **Frontend knobs** — remove the fast/slow mode toggle and clip-duration settings controls from
   the webapp UI; stop sending `fast_mode` / `target_clip_duration_*` from the client (backend
   params stay accepted-but-inert for iOS compatibility). The ReelCard player must correctly play
   long clips (verify its clip-end watchdog and the 180s fallback-window heuristic don't fight
   multi-minute clips; adjust the heuristic if it misclassifies legit long clips).

## What does NOT change
- ReelOut contract, persistence (`generation_id` scoping), captions windowing (follows whatever
  [t_start, t_end] is — already length-agnostic), channel_name, streaming callbacks + progress
  loader, ranked_feed ordering/feedback, refinement/extension exclusions (cost fix), 429/502 error
  mapping, community/auth, dry_run/can-generate viability probing (minus duration filtering).
- iOS: params stay accepted; server-side behavior changes equally for iOS-generated content
  (acceptable per user direction — knobs are dead product-wide).

## Tests
- Update `test_clip_engine_material_topic.py`: the over-length-skip regression test now asserts
  the INVERSE (over-length clips persist); the `max_reels` mechanism tests stay (the param still
  works when passed — generate_reels just no longer passes it).
- Update/remove any `generate_reels`/page-shaping tests that assert duration filtering.
- clip_engine suite green (isolated `-p no:randomly`); typecheck green for the frontend change.

## Ledger note
Record in `.superpowers/sdd/progress.md` that final-review finding #2's enforcement is reversed by
explicit user decision (raw-practice-output), so the whole-branch review verdict's premise changes
knowingly, not accidentally.
