# Session summary — 2026-07-03 (Wave 2.5 review+gate → user pivot → Wave 3 + Wave 4)

One session that (1) reviewed + fixed + gated the already-implemented **Wave 2.5**, (2) took a
user's live-test verdict that reframed the whole project, and (3) implemented the headline
**Wave 3 + Wave 4** features. Test suite **599 → 656** (all offline). Everything was built with
reviewed implementer subagents + adversarial verification + mutation checks; the hard invariant
`unverified_kill = 0` held throughout. Living plan for the remaining work:
`docs/superpowers/plans/2026-07-03-wave-3-4.md`. Full ledger tail: `.superpowers/sdd/progress.md`.

---

## Part 0 — where the session started

Wave 2.5 (tasks W25-A..G, coverage & universality) had been **implemented** by prior work (suite
599) but W25-G shipped without its per-task review (its implementer died on a credits limit) and
there had been **no whole-change review and no post-wave live gate**. That was the entry point.

## Part 1 — Wave 2.5 whole-change review + fix + gate (599 → 601)

**Whole-change adversarial review** (7-seam Workflow: one reviewer per integration seam →
per-finding adversarial verification). **6 findings, 5 REFUTED with proofs** — the strongest being
that the "build-stage double-ledger" is *unreachable* (`compute_closure` always seeds the anchor,
so `build_candidate` never returns None for a re-selectable unit anchor); the `_clamped_range` +1
is intentional/tested; the arc NEUTRAL-crawl is documented pre-existing design with downstream
containment.

**1 CONFIRMED (minor), fixed:** the W25-F refund **superset-REPLACE** evicted shippable sliver
incumbents on the *judge hard core*, but the superset's `final_quality` wasn't gated until the
later `quality_floor` stage — so a below-floor superset could evict a sliver and then die at the
floor, shipping **nothing** for that span. Fix: hoisted `floor` above the refund loop + a new
`_refund_ship_worthy` gate (`assemble/__init__.py`). Added a **mutation-verified** regression test
(`test_refund_below_floor_superset_does_not_evict_shippable_incumbent`) — it fails without the fix.

**Test-quality audit** (separate agent): the Wave 2.5 safety invariants (unverified_kill=0,
atomicity bisection oracle, freshness staleness, inventory 0.60/item-duration, practice
preservation) are all mutation-resistant — EXCEPT one gap: `_excess_verified` (the atomicity
oracle's half that puts a verified over-inclusion on the known-BAD side) was **co-defended** in
every bisection test, so reverting it alone left them green. Closed with an isolating test
(`test_bisection_isolates_excess_verified_on_incomplete_prefix`, mutation-confirmed). Suite 601.

**Gate** (qP-9wwRrJbg): my `--freeze --runs 3` on the cached structure read recall **0.538** and
`forward_requires_edges = 1`. The ledger's `--rebuild --runs 3` (fresh unified-substrate structure)
reads recall **0.641 ± 0.097 — beating the 0.577 handoff baseline** — with `forward_requires_edges
= 0` (W25-B working on fresh structures) and the **marquee targets fixed**: items 3 (vectors) 0.96,
4 (notation) 1.00, 8 (UAM equations, the recurring casualty) 1.00, 22 (projectile) 1.00, 24
(relative-velocity) 1.00; item 16 (graph-build) 0.36 PARTIAL as predicted. Honest caveat:
comprehension 0.287 vs handoff 0.692 — substrate change (322→183 sentences) + coverage rising into
harder prerequisite-dependent content the harsh judge scores low (the SciTalk "coverage↑ / proxy↓"
non-alarming pattern). W25-G report augmented with the review outcome.

## Part 2 — the user pivot (the load-bearing finding)

The app was hosted locally (uvicorn, fresh code) and the **user tested it and judged the output
quality ACCEPTABLE.** This **resolves the judge-validity open question in the clipper's favor**: the
eval judge scored qP comprehension 0.23, but the human says the clips are fine → **the judge
over-flags**, exactly as the corruption-probe predicted. Quality is no longer the blocker.

**The real problem is LATENCY: ~20 min/video.** User direction: implement Wave 3 + Wave 4
(research-first), **then** a latency pass. So Wave 3/4 here targeted the remaining product-visible
gaps — not chasing judge scores.

## Part 3 — Wave 3 + 4 research (6-area Workflow)

Each area verified current tool state (Context7 for the Gemini video API, web for the ML libs) +
audited the existing code + checked Apple-Silicon feasibility → a synthesized plan. Raw findings:
`docs/audits/2026-07-03/wave-3-4-research.json`. **Feasibility-driven re-scoping (verified):**
- **DROP `torchaudio.forced_align`** — no torchaudio wheel for the installed torch 2.12.1;
  installing risks breaking sentence-transformers/TreeSeg; and it fixes the *wrong* error (both
  audits show `ends_on_period_rate=1.0` yet mid-clause ends → the residual is **segmentation, not
  timing**). Free text guards (BND1) cover the real error.
- **DROP `inaSpeechSegmenter`** — drags TensorFlow + Pyro4 into the shared venv; a one-call
  Gemini-audio music check gives equivalent discrimination at zero venv risk.
- **DEFER SaT/wtpsplit** (redundant with the existing LLM punctuation package) and the **parakeet
  default-flip** (opt-in only; no golden set to A/B).
- Much of the "new" work was a **delta, not a rewrite** — the frontend fields were already in the
  payload (just dropped by ClipCard); card-as-repair reused `generate_context_card` + `judge_clip`
  (which already injects a card); genre just needed the discarded yt-dlp metadata + one adapter.

## Part 4 — Wave 3 + 4 implementation (601 → 656; all invariant-safe)

Each task/batch: reviewed implementer subagent (offline tests) → my review (suite + no-Rejection
grep + invariant mutation-checks). Snapshots in `.backup/{pre-wave34-sprintA, post-wave34-sprintA,
wave34-headline-complete}-2026-07-03.tgz`.

**Sprint A — product-visible quick wins:**
- **FE1** — fixed the embed `ceil(end)+1` bleed (~1.7s into the next sentence on every clip);
  consolidated the two drifting copies into one shared `backend/embed.py` + tests.
- **FE2** — surfaced `final_quality` / `warnings` / `ship_flagged` into the clip payload.
- **GEN1** — surface the yt-dlp `categories`/`tags`/`artist`/`track` that `download()` was
  discarding + a supadata-mode `probe_metadata()` (one `extract_info(download=False)` call).
- **GEN2** — `EntertainmentAdapter` (single lenient `moment` contract, NOT in `_PROBLEM_ROLES`) +
  metadata→entertainment override. **The minimal fix so parody/music/comedy stops shipping 0
  clips.**
- **FE3** — `ClipCard` renders title, context card, "watch clip N first" chips, subtle quality
  badge, curated warning allowlist (internal warnings hidden).
- **FE4** — rebuilt the frontend into `backend/static` (typecheck + build clean).

**Sprint B — card-as-repair (top quality item, invariant-critical):**
- **CARD1** — `JUDGE_SYSTEM` now credits a provided context card (concepts it introduces satisfy
  `prerequisites_satisfied`; references it resolves satisfy `all_references_resolved`) — the card
  was already injected but ignored; also aligns the gate with eval.
- **CARD2/CARD3** — card-as-repair at **both** verified+confirmed kill gates (repair-terminal +
  post-snap/merge): when every confirmed reason is prereq/reference-family, generate a grounded
  card and re-judge the **same span**; ship if it clears → **purely accept-side** (converts a
  would-be Rejection into a ship, never creates one). Threaded the card into the re-judge seams.
- **CARD4** — invariant regression tests (UAM rescue positive at both gates; 3 negatives; the
  `unverified_kill=0` pin). I **independently mutation-checked** the `all→any` guard (fails the
  non-family test) and confirmed accept-side-only by reading both insertions.

**Sprint C — cheap boundary + card robustness:**
- **BND1** — free text-only boundary guards: `sentences_from_chunks` no longer fabricates
  `ends_with_period=True` on every caption chunk; `_snap_one` prefers a real terminator and a
  ≥3-word non-conjunction end — with a **safe fallback** (`_end_acceptor`) so fully-unpunctuated
  caption videos still ship clips (verified). Moves boundaries, never kills.
- **CARD5** — rescue-seeding from `missing_concept → introducer` when the prereq isn't in
  `referential` (still grounded → an ungroundable concept means no card and the kill stands) +
  first-sentence-names-subject suppression (conservative rapidfuzz).

**Sprint D (edge tier) — video judge:**
- **VID1** — `gemini_client.generate_json_video` + `video_part_inline` (verified against the
  installed google-genai 2.10.0 API).
- **VID2** — `edge_probe.py`: cut each survivor's first/last ~8s, one inline LOW-res video call,
  **advisory** warnings (`starts/ends_mid_sentence_audio`) — hooked after `refine_clip_boundaries`,
  never kills.
- **VID4** — config flags (`EDGE_PROBE_ENABLED` / `VIDEO_JUDGE_*`, **all default OFF** like
  DIARIZATION), eval columns, and the item-24 `visual_summary=""` fix. With flags off, behavior is
  byte-identical (the 622 pre-existing suite passed unchanged).

## Environment / process notes (carried from prior sessions, still true)
- **Gemini only** (author `gemini-2.5-flash`, judge `gemini-2.5-flash-lite`); keys in `clips/.env`;
  Groq key 401 and has **no video** input (the video judge is Gemini-only by construction).
- `clips/` is **NOT a git repo** — snapshot to `.backup/` before big changes; verification = full
  `pytest backend` + `compileall`.
- The dev server snapshots code at startup — **restart after changes**: `cd clips &&
  .venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000`. Frontend is prebuilt
  into `backend/static` — **rebuild** (`cd frontend && npm run build`) after any `.tsx` edit.
- **pytest `.pyc` caching bit once** during a mutation check — a reverted file kept "failing" until
  `__pycache__` was cleared. When a mutation check's post-restore run looks wrong, clear pycache.
- **Background long commands can be killed** (~10-15 min) and **stdout is block-buffered when
  redirected** (lost on kill) — run evals unbuffered (`python -u`) and **chunk** them; reuse
  already-fresh cached structures via `--freeze` to skip rebuilds.
- **zsh does not word-split unquoted `$VARS`** — a `run_eval $VIDS` passed all ids as ONE arg and
  silently skipped everything. Write multi-arg lists literally.

## Key numbers at session end
Suite **656 passing**, compileall clean, frontend typechecks + builds. Server live on current code
(`localhost:8000`). unverified_kill=0 preserved across all new paths. Wave 2.5 rebuild gate recall
**0.641** (beat 0.577 baseline). Wave 3/4 headline features shipped; follow-up tail documented.
