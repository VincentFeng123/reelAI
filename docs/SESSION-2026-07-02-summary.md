# Session summary — 2026-07-02 (audit → research → Waves 1+2 → coverage quick wins)

One session took the structure-first clipper from "architecture faithful, output not good" through
two verified implementation waves plus a coverage study driven by a real user test. Test suite
**149 → 449** (all offline). Everything below was built subagent-driven with per-task adversarial
reviews, whole-change reviews, and live A/B gates against frozen baselines. The living plan doc is
`docs/CLIP-QUALITY-2026-07-02-audit-and-fix-plan.md` (audit verdict + 4-wave plan + per-wave
outcome sections + Wave 2.5 spec) — read it first; this file is the session record + handoff.

## 1. Quality audit (21 agents) + fix-plan research (10 agents)

- **Verdict**: understanding layer good; clips die in selection → judge → repair. Kinematics
  lecture (dHjWVlfNraM) delivered ~1 of 5 reference outcomes; trio yield 0.667 clips/video with
  7.7 repair kills/video; judge hallucinated failure reasons (killed a near-perfect worked
  example on a factually false verdict) while passing mid-clause fragments with stale verdicts.
- Verified failure modes F1–F8 + 22-requirement design conformance + 11 ranked gaps, all
  adversarially verified with file:line evidence: `docs/audits/2026-07-02/*.json`.
- Research (5 angles, every top finding source-checked by a skeptic; 4-5/5 survived per angle):
  evidence-grounded judging, TPR-96%/TNR-<25% asymmetry, shrink repair, coverage selection,
  small-N calibration methodology, video-native judging feasibility. In the plan doc + audits dir.

## 2. Wave 1 — judge integrity (+ follow-up patch). SHIPPED, 267 tests

- `evidence_quote` on every FailureReason + deterministic containment check
  (`_verify_failure_reasons`); asymmetric 3-outcome gate (ACCEPT / ship-flagged
  `'unverified_judge_concerns'` −0.15 / REJECT only on verified **and** fresh-context-confirmed
  reasons via `confirm_kill`) — applied at **all** judge stages incl. post-merge/post-snap
  (follow-up patch closed that gap). `judged_text_hash` verdict cache + hash-triggered post-snap
  re-judge (stage `post_snap_judge`); closure/repair span budget clamped to the 240s ship cap.
- Calibration bootstrap: YouTube-chapters gold (Pk/WindowDiff/clip_straddle_rate un-NaN'd) +
  corruption-probe CLI (`backend/eval/judge_probe.py`, strict argparse).
- **Measured**: kinematics n_clips 4→5, comprehension 0.25→0.40, worked_example_completeness
  0→1.0; trio yield 0.667→1.111, repair kills 7.7→3.7, `unverified_kill` → **0.000 (standing
  tripwire)**; `phantom_verdict_rate` 0.227 on kinematics (phantom kills real + instrumented).
  Probe: judge TNR 0.20 on its own accepted clips; `reasoning_complete`/`result_complete` least
  valid gates. Trio comprehension 0.333→0.056 — HONEST shift: borderline clips now ship flagged
  instead of dying silently; judge-score validity is the open question (→ labeling).

## 3. Wave 2 — extraction rework. SHIPPED, 418 tests

- P1 contract-by-content with rebinding (contract bound to assembled span's roles, rebound after
  every mutation; whole-change review caught+fixed: calculation/derivation couldn't satisfy the
  result contract in base.py AND lecture.py's override). P2 repair: anchor judged at native size
  first, trim moves + noisy bisection (uses the text-hash cache for free revisits). P3
  **extraction plan** (user-directed design: LLM proposes what's worth extracting per video from
  the actual unit inventory; adapter = prior, not straitjacket; `ANCHOR_SELECTOR=plan` default,
  `priority` legacy path byte-equivalent, auto-fallback flagged) + deterministic `arcs.py`
  (setup→steps→result grammar, bypasses anchor roles) + topic quotas + `PLAN_ROLE_CAP=4`. P4
  dedupe prefers complete-arc/hard-gate-passing specs over score-inflated fragments +
  severed-pair linker. E1 labeling tooling: `/static/labeling/` page,
  `sample_for_labeling.py --collect`, `judge_calibration.py` (κ + bootstrap CI + per-kind
  precision, refuses n<10).
- **Measured**: trio 1.111→1.444; kinematics 5→6 clips incl. a COMPLETE merged worked example
  (the audit's severed bus problem); repair 3→1; straddle 0.40→0.167; plan fallbacks 0; tripwire
  0. Scorecard still 2/5: equations candidate now GENERATED but verified+confirmed-killed at
  post_snap_judge (missing_prerequisite — **card-as-repair in Wave 3 is the designed rescue**);
  practice clip absent; severed_pairs_linked=0 (conditions too narrow); n_trims=0 everywhere
  (trim routing never fires); uqw parody song 1→0 clips (honest kill of the stale garbled
  fragment; genre = Wave 4).

## 4. Coverage study — the qP-9wwRrJbg case (user test) → Wave 2.5 spec

User ran "AP Physics 1 - Unit 1 Review - Kinematics" (23.9 min, topic "kinematics") → **2 clips
vs a verified ideal of ~20** (26-item inventory =
`docs/audits/2026-07-02/qP-9wwRrJbg-coverage-gold.json`, now the standing coverage acceptance
test). Structure layer understood it (139 units); selection starved the judge. Root causes
(gap_analysis.json): stable-sort tie-breaking put ALL 12 anchors in the first 281s of 1432s;
practice_prompts mathematically unreachable under cap 12; dedupe collapsed 12 anchors→3 specs
with no refund (63 eligible units untouched); topic filter killed tangential gems; judge passed
multi-item mashes (no atomicity gate); requires-edges at half density (sparse
concepts_introduced + LLM edge pass prompt-forbidden from 'requires'). Judge kills ≈ 0 — the
judge was NOT the bottleneck here. Fresh verified research (saturation-stopped budgets, GMR
set-based selection, HiVid batched saliency, batch judging 30-82% cheaper, criteria injection,
CCQGen refund, CARE conformal omission bound, universal rubric cross-genre) → **Wave 2.5 spec in
the plan doc**.

## 5. Coverage quick wins — SHIPPED + GATED (451 tests). Acceptance test MISSED with a diagnosis.

Landed (both tasks reviewed+approved): content-scaled anchor budget (`compute_anchor_budget`:
floor 12, ceil(eligible/4)+4 if density high, ceiling `MAX_ANCHORS_CEIL=32`; explicit user dials
respected; flows to plan cap + quotas + max_clips), topic-spread greedy tie-break, PLAN_ROLE_CAP
in the legacy selector, relevance bypass (`topic_matches_subject`), dedupe **refund loop**
(≤`REFUND_ROUNDS=2`, residual re-selection through identical judge/integrity machinery), graph
nutrition (concepts_introduced nudge; LLM edge pass may emit 'requires'; no SCHEMA_VERSION bump).
New eval columns: anchor_budget / n_refund_rounds / n_refund_clips.

**Gate results**: qP-9wwRrJbg 2→7-9 clips (budget 23; **n_trims=4 — trim moves now fire**; 5 arcs
detected / 3 arc clips shipped; properly scoped roles incl. a misconception clip); kinematics
6→8 clips, chapter_coverage 0.938; trio stable 1.7±0.2; ALL tripwires clean.
On the STALE cached structure, recall was 23.1% (below the 31% baseline; zero clips after 836s)
— the gate diagnosed misbound topic labels (tail units shifted ~1 ASR run-on sentence, verified
against the transcript). **After `--rebuild` with Q2 graph nutrition, the diagnosis was
CONFIRMED and the acceptance test passes decisively: qP = 13-14 clips, comprehension 0.692,
mean_judge 0.731, ZERO flagged clips, chapter_coverage 0.917, 11 arcs detected / 4 arc clips,
refund loop shipped 4 clips over 2 rounds, INVENTORY RECALL 15/26 = 57.7%** — the back third of
the video finally ships (relative-motion definition + reverse-frame worked example + sig-figs
callback exam tip all present). Net journey for the user's video this session: **2 clips/~31% →
13-14 clips/57.7% recall/0.692 comprehension.**
New Wave 2.5 items from this: structure-cache freshness policy (old-code caches silently poison
selection); unit-boundary drift on ASR run-ons; remaining missed items as coverage targets
(vectors/scalars 3-4 — regressed vs pre-rebuild, investigate; UAM equations 8 — card-as-repair's
target; 205s graph-build example 16; projectile procedure/strategy 21-22; relative-velocity 24).
Also noted: assembly nondeterminism under identical structure (13 vs 14 clips — judge flicker on
novel text hashes); phantom_verdict_rate 0.265 on kinematics (rising, watch); uqw parody song
ships 0 (genre, Wave 4).

## Environment gotchas (verified this session; supersedes older notes)

- Gemini only (author `gemini-2.5-flash`, judge `gemini-2.5-flash-lite`); keys in `clips/.env`;
  Groq key still 401. `clips/` NOT a git repo — snapshots in `clips/.backup/` (pre-Wave-1 tarball
  exists; make a new one before big changes). Use `clips/.venv/bin/python`.
- **Workflow subagents must run eval commands FOREGROUND** (timeout 600000) — backgrounding +
  ending the turn kills the agent (bit us once; resumed via `resumeFromRunId` with cached
  results).
- Judge temp-0 nondeterminism is real under freeze (same video flips 0↔1 clips between
  invocations): the variance floor is NO LONGER 0.000 — always `--runs 3`, trust means±std.
- Dev server: `cd clips && .venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port
  8000` — snapshots code at startup; RESTART after every wave. Frontend prebuilt in
  backend/static/. Labeling page: `/static/labeling/` (run `sample_for_labeling.py --collect
  <ids>` first). Cheap cached test videos: dHjWVlfNraM (kinematics lecture), qP-9wwRrJbg (AP
  review + coverage gold), trio uqwC41RDPyg/NjvwWiCYLl4/yfajBIaDf1Q.
- Spec governs over stale tests; verify string edits via python `in` checks, never hexdump.

## Roadmap (order: verify quick wins → 2.5 → 3 → 4; labeling parallel anytime)

1. **Wave 2.5 — coverage & universality** (specced in plan doc): saturation-stopped per-video
   budget; threshold (τ) selection instead of top-K; HiVid-style batched unit-saliency pre-filter
   + batched first-pass judging (A/B verdict agreement first) + criteria injection; universal
   teachable-moment rubric blended with role priority; arc terminal-role broadening + minimum
   substance; atomicity gate + mash splitting (≤300s one-idea clips); clip-count dial
   (auto/int/query); CARE conformal omission bound (needs golden labels). FOLD IN Wave-2 tuning
   backlog: trim routing never fires; arc→ship attrition (3 detected → 1 shipped); severed-pair
   linker conditions; practice clips still not shipping standalone.
2. **Wave 3 — boundaries & cards**: card-as-repair + cards always-on generated BEFORE the gate +
   judge clip+card (rescues the equations-clip class); SaT sentence splitter on caption
   fallbacks; 2-of-3 boundary gate (terminator+SaT+pause≥250ms) + cut inside the pause; local
   parakeet-mlx ASR path; forced-alignment precise cuts (NOT WhisperX — 110ms err vs MFA 20ms).
3. **Wave 4 — video judge, genre, surfacing**: Gemini edge-probe (listen to first/last 8s,
   ~$0.0006/candidate) + render-audit judge on survivors (~1.5¢/clip, generateContent
   video_metadata offsets — NOT Interactions API); genre ensemble (inaSpeechSegmenter music
   fraction, yt-dlp categories/track/artist, transcript repetition, diarization turn stats) +
   entertainment adapter; reconciliation pass; frontend renders cards/watch-first/quality
   badges/flags; embed ceil(end)+1 bleed fix; eval passes real visual_summary.
4. **Human calibration (user's time, unlocks everything)**: ~30 labels → judge κ + per-kind
   trust weights + threshold re-fit; grow to 100-150 → PPI++ metrics + CARE omission bound.
   Expert ceiling on this criterion is κ≈0.51 — κ~0.5 is success.
5. Older infra backlog (pkg-2 B/C): per-clip nudge/re-judge/restore endpoints, stable clip_id,
   job persistence, per-video locking. Deprioritized: pyannote install, OCR fusion, Maverick.

## Key open numbers (post-Wave-2 gate)

Trio: n_clips 1.444±0.192, comprehension 0.037±0.064 (judge validity unresolved), repair
3.222±0.385, unverified_kill 0.000 (tripwire). Kinematics: 6 clips, scorecard 2/5 (definition +
complete worked example), chapter_coverage 0.875, phantom_verdict_rate 0.24. qP acceptance:
pre-quick-wins 2 clips / ~30% item coverage; target 10-20 clips, recall >>30%. Suite 449.

---

# Next-session prompt (copy-paste; also in NEXT_SESSION.md)

```
You are continuing quality work on the structure-first video clipper in clips/backend/
(Python/FastAPI; YouTube-embed clips via understand → TreeSeg segment → units/roles → dependency
graph → extraction plan + arcs + coverage quotas → context closure → contract-by-content →
quote-verified asymmetric clip-only judge (kills need verified+confirmed evidence, else
ship-flagged) + trim/grow repair → boundary snap → text-hash re-judge → refund loop → 7-stage
rejection ledger). All LLM calls via llm_json() (Gemini flash author / flash-lite judge, temp 0).

READ FIRST (in order): clips/docs/SESSION-2026-07-02-summary.md (session record + gotchas),
clips/docs/CLIP-QUALITY-2026-07-02-audit-and-fix-plan.md (living plan: audit → waves → outcomes →
Wave 2.5 spec), clips/.superpowers/sdd/progress.md (ledger tail), and skim
clips/docs/audits/2026-07-02/ (gap_analysis.json + qP-9wwRrJbg-coverage-gold.json matter most).

SHIPPED — do NOT redo: Wave 1 (quote-verified asymmetric judge gate at ALL kill stages,
confirm_kill, text-hash verdict cache + post_snap_judge re-judge, budget clamp, chapters gold,
judge probe), Wave 2 (contract-by-content w/ rebinding, native-first+trim repair, extraction plan
ANCHOR_SELECTOR=plan, arcs.py, quotas, dedupe arc-preference, severed-pair linker, labeling
tooling at /static/labeling/), coverage quick wins Q1+Q2 (content-scaled budget
compute_anchor_budget, topic-spread tie-break, legacy role cap, relevance bypass, refund loop,
graph nutrition) — suite was 449 passing at handoff.

FIRST TASK — verify the quick-wins gate (it may not have run): cd clips &&
.venv/bin/python -m pytest backend -q (expect ≥449); then live A/B:
run_eval uqwC41RDPyg NjvwWiCYLl4 yfajBIaDf1Q --freeze --runs 3;
run_eval dHjWVlfNraM --freeze; run_eval qP-9wwRrJbg --freeze; then a scratch mirror-dump for
qP computing inventory recall vs docs/audits/2026-07-02/qP-9wwRrJbg-coverage-gold.json (item
covered = ≥60% span overlap with a shipped clip). Baselines to beat: qP 2 clips/~30% coverage;
trio n_clips 1.444±0.192 repair 3.222; kinematics 6 clips scorecard 2/5. HARD TRIPWIRE:
unverified_kill must stay exactly 0. Report deltas honestly (comprehension may dip when flagged
clips ship — that is by design pending calibration).

THEN: Wave 2.5 (coverage & universality — full spec in the plan doc §"Coverage study") folding in
the Wave-2 tuning backlog (trim routing never fires: n_trims=0 everywhere; arc→ship attrition
3→1; severed-pair linker fired 0 times; practice clips never ship standalone). After 2.5: Wave 3
(card-as-repair + cards-before-gate — the designed rescue for the equations clip that keeps dying
verified+confirmed on missing_prerequisite; SaT boundaries; parakeet ASR; forced alignment). Then
Wave 4 (video edge-probe + render-audit judge, genre ensemble incl. music detection —
the uqw parody song currently ships 0 clips as "lecture"; frontend surfacing of
cards/watch-first/quality badges; embed +1s bleed). Process: spec → plan → subagent
implementation with per-task adversarial reviews + whole-change review + foreground live gate,
exactly as the ledger shows.

ASK THE USER EARLY: have they labeled clips at localhost:8000/static/labeling/? If yes, run
backend/eval/judge_calibration.py and use κ + per-kind precision to set kill-authority trust
weights and re-fit the 0.70 gate (this unblocks every "verified kill vs harsh judge" question —
currently the biggest unknown; probe says reasoning_complete/result_complete are the least valid
gates). If no, gently remind them: ~30 labels, page plays clips cold.

ENVIRONMENT GOTCHAS: Gemini only (Groq 401); clips/ NOT a git repo (snapshot to clips/.backup/
before big changes; verification = full pytest + compileall); use clips/.venv/bin/python; workflow
subagents must run evals FOREGROUND (timeout 600000) and must end by emitting their structured
output; judge temp-0 flicker is real — always --runs 3, variance floor is not 0; dev server
snapshots code at startup (restart after changes: cd clips && .venv/bin/python -m uvicorn
backend.main:app --host 127.0.0.1 --port 8000); spec governs over stale tests; verify string
edits with python 'in' checks, never hexdump.
```
