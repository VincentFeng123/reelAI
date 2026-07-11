# Session summary — 2026-07-01 → 07-02 (quality sprint on the structure-first clipper)

Five majors shipped, each spec'd → planned → built subagent-driven with per-task + whole-branch
reviews, and each **measured** with the harness built in step 1. Test suite grew **55 → 149**
(all offline, no LLM/network). No git repo — verification gates were full-pytest + compile
checkpoints; progress ledger at `.superpowers/sdd/progress.md`.

## 1. Trustworthy eval harness (`backend/eval/run_eval.py`)

- New flags: `--freeze` (reuse cached `work/<id>/structure.json`), `--freeze-specs` (assemble
  once, only re-judge → judge-only variance), `--runs N` (mean ± sample-std per metric),
  `--rebuild`. Aggregate computed over a **fixed video set** so std = run-to-run noise, not
  video-composition drift. Explicit ids never truncated; `select_videos` caps only defaults.
- **Finding that drove the whole session:** the structure REBUILD was the dominant eval noise
  (frozen → every metric ±0.000; rebuilt → n_clips flips 1↔2, comprehension 0.5↔1.0).
- A/B recipe: run baseline BEFORE a change, then same command after; `--freeze --runs 3` is the
  standard gate.

## 2. TreeSeg content map (`understand/treeseg.py`, `content_map.py`)

- Topic boundaries now **deterministic embedding divisive segmentation** (Ward-style bisecting +
  pause/discourse prior; chapters from the same tree; `all-MiniLM-L6-v2` lazy singleton). LLM
  only LABELS fixed segments (clamped by index — cannot re-partition; failure → keyword titles).
- `CONTENT_MAP_ENGINE=treeseg|llm` (legacy kept verbatim + auto-fallback `engine="llm-fallback"`,
  stderr-logged, `Structure.degraded+=["content_map"]`). **`SCHEMA_VERSION` 3→4** (all caches
  rebuilt once). `ContentMap.engine` field.
- Measured: partition byte-identical across processes; unresolved-ref & prereq-gap **0.72→0.28**;
  comprehension 0.44→0.22±0.19 (statistically inconclusive at 3 small vids); n_clips std
  unchanged (downstream LLM noise dominates once boundaries are fixed).

## 3. Judge rubric + CoT + 1-10 scale (G-Eval; `assemble/validate.py`)

- `JudgeVerdict`: `reasoning` FIRST field (google-genai preserves pydantic order → CoT generated
  before verdict; verified live), `score_10` 1-10 normalized to legacy 0-1 (`0.70 ≡ ≥7`),
  `error` (fallback-only, forced False on every successful parse). Anchored-bands rubric prompt;
  boolean gates + failure-kind vocabulary byte-preserved (repair-loop contract).
- **Honest outage policy (replaced silent understandable=True/0.7):** error verdict fails ALL
  gates, skips repair, never cached, completeness 0.5, grounding 0.6×, "unjudged" warning
  (−0.15), eval EXCLUDES it + reports `judge_error_rate`. Cross-model→authoring retry ladder
  preserved.
- Measured: determinism intact (±0.000); stricter at assembly (4→2 surviving clips, mean
  0.483→0.400, comprehension flat 0.333). Caveat: `--freeze-specs` re-assembles per invocation
  WITH the judge → old-vs-new isn't pure same-specs; needs cross-invocation spec persistence.

## 4. Clip integrity chain (robustness pkg 1; `assemble/integrity.py` + seams)

- `true_contents` keeps `unit_ids`/`referential` TRUTHFUL at every span change (build, repair
  expand, merge, post-snap). `merge_partb` metadata-aware union; merged spans **always
  re-judged** (success clears stale unjudged; failure → ledgered rejection; outage ships
  flagged). `validate_and_repair → (cand|None, Rejection|None)`.
- **6-stage drop ledger** (`repair/snap/dedupe/post_merge_judge/quality_floor/max_clips`):
  `assemble_clips → (specs, notes, rejections)`; CLI prints `[dropped/<stage>]` with judge
  kinds; eval gained `rejections_*` + `n_merged` columns. Cards: extractive fallback from
  referential summaries + penalized `missing_context_card` warning. `refine.py`/`boundary.py`
  keep/drop behavior FROZEN (fast path) — additive warnings only
  (`extended_for_min_duration`, `trimmed_start`; −0.10 each).
- **Measured: post-fix frozen eval fully deterministic (every metric ±0.000, 3 runs); the
  n_clips accounting closes exactly: 0.667 kept + 7.667 repair + 0.333 dedupe per video.**
  The session-opening mystery ("why does n_clips vary?") is ANSWERED: repair-stage judge
  rejections of borderline candidates (~7.7/video under the strict rubric judge) — now a
  visible tuning lever (`min_comprehension_score` / repair budget).

## 5. Quick wins (robustness pkg 3; 7 surgical fixes)

Precise-boundary failure returns judged clips unchanged; direction-safe Whisper picks
(`_pick_end` two-tier ≥rough preference, `_pick_start` strict return-rough, `keep_first` at
video start); atomic ffmpeg writes (`cut.finalize_output`, tmp→rename, size-checked — partial
mp4s can't poison caches); range-keyed export names `clip_{n}_{slug}_{start_ms}_{end_ms}.mp4`;
`_zipable_files` + 409 zip guard; relevance one-retry → `(rel, degraded)` + notes suffix
`" (topic filtering degraded — clips selected by role priority)"`; earliest-per-concept
`prerequisite_clips` (self-defined concepts never hinted; `sequence_clips(specs, graph,
units_by_id)`).

## Also produced

- **58-finding robustness audit** (6 parallel subsystem auditors, evidence-cited):
  `docs/clip-editing-robustness-audit.md` — pkg 1 + pkg 3 headers mark what shipped; the
  "Post-IC review backlog" section + pkg-2 items below are what remains.
- Specs/plans in `docs/superpowers/{specs,plans}/2026-07-0{1,2}-*.md`.

## Process record (what the review machinery caught)

- 4 plan-authoring bugs (fixture-vs-spec contradictions) caught by implementers/reviewers and
  adjudicated: spec governs, tests fixed (TreeSeg sec=30; judge calls==1 ladder; pick-start
  3.1 assertion; prereq [1] vs [1,2]).
- 2 genuine Criticals caught only by whole-branch reviews: IC post-snap metadata staleness
  (resurrected the card-restates-content bug); judge fallback booleans defaulting True
  (unjudged clips earning grounding credit).
- 1 false controller verification: hexdump byte-pattern checks split across dump lines —
  **verify glyphs/strings via Python `in` checks, never hexdump pipes.**
- Implementers twice "fixed" spec behavior to satisfy flawed tests (retry ladder; TreeSeg
  constant) — both reversed. Watch for this failure mode.

## Environment gotchas (verified this session)

- Gemini only (`GROQ_API_KEY` invalid/401); working models: gemini-2.5-flash (author),
  gemini-2.5-flash-lite (judge), gemini-2.5-pro (rate-limited). `gemini-2.0-flash` retired.
- `clips/` is NOT a git repo. Use `clips/.venv/bin/python`. Suite: 149 passed, fully offline.
- No golden files → gold metrics NaN. Eval uses first-8 cached videos by default; cheap trio:
  `uqwC41RDPyg NjvwWiCYLl4 yfajBIaDf1Q`.

---

# Pkg 2 plan — per-clip editing API + surfacing (NEXT; not yet spec'd)

The last big planned build. Goes through brainstorm → spec → plan → subagent-driven execution.
Intended scope (to be refined in brainstorming):

**A. Surfacing (make quality visible)**
- Clip payloads (`_build_embed_clips` / orchestrator) expose verdict summary, warnings,
  final_quality, and the run's rejections; frontend renders context cards + `prerequisite_clips`
  ("watch first") — both generated today but never shown — plus per-clip quality badges.

**B. Editing endpoints (make clips fixable — cheap now: structure cache + verdict cache +
rejection ledger already exist)**
- Nudge boundaries ±n seconds → re-snap → optional re-judge of the new span.
- Re-run judge/repair on one clip; drop/restore a clip (rejections make restore candidates
  visible); re-export single clip (range-keyed names already support this).
- Requires **clip identity** (stable clip_id) — also fixes the sequencing-renumber staleness
  (dangling `prerequisite_clips` after post-assembly drops).

**C. Persistence + lifecycle (infrastructure the endpoints need)**
- Persist job results to disk (restart survival; embed mode currently loses everything).
- Per-video locking (export tmp collision, concurrent job races); job cancel endpoint;
  orphan-file cleanup (old un-keyed exports, superseded clips).

**D. Accumulated review backlog to fold in**
- Post-refine emptiness guard (job can finish DONE with 0 clips + stale notes).
- Whisper-pass drops escape the ledger (`_resolve_overlaps` — ledger via before/after count).
- Shrink-aware metadata repair (trim/cap reduce span but unit_ids overclaim).
- Degraded/relevance flag on FAILURE-path notes; whisper-drift warning (display-only until
  boundary_score ordering changes); merged grounding_score recompute; `_pick_start` empty-pool
  rule; `_pick_end` middle-tier test.

**Pairs with roadmap #6 (human calibration set):** I build labeling tooling around
`eval/make_golden.py` skeletons while pkg 2 executes; human labels an
"understandable-in-isolation" set → anchors judge validity + unlocks gold metrics
(role_accuracy, anchor_recall, boundary_error, Pk/WindowDiff).

**After pkg 2:** roadmap #5 (OCR + Texo for the visual-heavy 0-clip class), #3 (Maverick coref —
deprioritized: TreeSeg already cut unresolved-ref 0.72→0.28), and the repair-yield tuning pass
(trade `min_comprehension_score`/repair budget vs clip yield using the rejections columns).

---

# Next-session prompt (copy-paste)

```
You are continuing quality work on the structure-first video clipper in clips/backend/
(Python/FastAPI; YouTube-embed clips via understand → TreeSeg segment → dependency graph →
context closure → G-Eval clip-only judge + repair → boundary snap → drop-reason ledger). All
LLM calls go through llm_json() (Gemini gemini-2.5-flash; judge on gemini-2.5-flash-lite).

READ FIRST: clips/docs/SESSION-2026-07-01-summary.md (everything shipped + pkg 2 scope),
then clips/RESEARCH.md (cited research), then clips/docs/clip-editing-robustness-audit.md
(finding backlog; pkg 1+3 headers = done).

Shipped — do NOT redo: trustworthy eval (--freeze/--freeze-specs/--runs N + rejections_*
columns), TreeSeg deterministic content map (CONTENT_MAP_ENGINE, SCHEMA_VERSION=4), G-Eval
judge (reasoning-first, score_10, honest error verdict, ship-but-flag), clip integrity chain
(true_contents, merge re-judge, 6-stage Rejection ledger, card fallback), 7 quick-win fixes.
Suite: 149 passed offline (.venv/bin/python -m pytest backend -q). Progress ledger:
clips/.superpowers/sdd/progress.md.

Environment gotchas: Gemini only (Groq key 401); gemini-2.0-flash retired; clips/ is NOT a
git repo (checkpoints = full pytest + compileall); no golden files (gold metrics NaN); eval
A/B recipe = run baseline BEFORE a change, then --freeze --runs 3 after; cheap video trio
uqwC41RDPyg NjvwWiCYLl4 yfajBIaDf1Q. Verify glyph/string edits with python 'in' checks,
never hexdump pipes. Watch for implementers "fixing" spec behavior to satisfy flawed tests —
spec governs; fix the test.

NEXT TASK: robustness pkg 2 — per-clip editing API + surfacing. Scope sections A-D in the
session summary (surfacing quality signals + cards in frontend; nudge/re-judge/drop/re-export
endpoints on a new stable clip_id; job persistence + per-video locking + cancel; fold in the
accumulated review backlog listed there). Process: brainstorm (scope questions: A-D in one
spec or split? clip_id design? persistence format?) → spec → plan → subagent-driven execution
with per-task reviews + whole-branch review, exactly as the ledger shows prior packages did.
Optional parallel track: roadmap #6 human calibration tooling around eval/make_golden.py.

KEY OPEN NUMBERS: comprehension 0.333 (flat, small-sample); repair gate rejects ~7.7
candidates/video (the yield-vs-rigor tuning lever); TreeSeg cut unresolved-ref/prereq-gap
0.72→0.28. The eval is deterministic under freeze — trust deltas only via the A/B recipe.

Start by reading the three docs, then ask which pkg-2 scope split I want.
```
