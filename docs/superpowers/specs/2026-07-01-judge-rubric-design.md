# Judge rubric + CoT + 1-10 scale (G-Eval) — design

**Date:** 2026-07-01 · **Status:** approved-pending-user-review · **Roadmap:** RESEARCH.md Tier-2
"Judge rubric+CoT, 1-10 scale" (G-Eval arXiv:2303.16634; scale finding arXiv:2405.01724).

## Problem

The clip-only judge (`assemble/validate.py`) is the measurement everything else is judged by, and
today's comprehension numbers are inconclusive partly because of it. Three research-backed gaps:
(1) no explicit rubric or reasoning step (G-Eval: rubric → CoT → form-filling lifts correlation,
Spearman 0.514); (2) a 0-1 float score (1-10 integer discriminates better, Kendall τ 0.428 vs
0.383; the float clusters at 0.7); (3) — audit finding — a judge API failure silently returns
`understandable=True, score=0.7`, shipping unjudged clips as top quality and inflating the eval
comprehension metric during outages.

## Decisions (locked with user)

1. **Approach A:** rubric + reasoning-first + 1-10 emitted, normalized to 0-1 internally — zero
   consumer churn. (Rejected: full G-Eval auto-CoT — criterion is fixed so generated steps are
   static, an extra LLM call for nothing; full 1-10 migration — churn, no measurement gain.)
2. **Judge outage policy: ship-but-flag.** Failed judge → `error=True` verdict (score 0, not
   understandable); clip ships with an `"unjudged"` warning; repair loop skipped; neutral
   completeness scoring; eval EXCLUDES unjudged clips and reports `judge_error_rate`.
3. Boolean gates and the `failure_reasons` kind vocabulary are UNCHANGED — the repair loop
   (`expand_candidate`) depends on them.

## Changes

### `backend/pipeline/assemble/validate.py`

- **`JudgeVerdict`:** add `reasoning: str = ""` as the FIRST declared field (google-genai
  preserves Pydantic declaration order in `response_schema`, so Gemini generates the CoT before
  the verdict fields — verify once empirically with a live call during implementation; if order
  is not honored, fall back to instructing reasoning-first in the prompt only). Add
  `score_10: int = 0` (emitted 1-10) and `error: bool = False` (never emitted by the LLM;
  set only by the fallback path). A `model_validator(mode="after")`: if `score_10 > 0`,
  `score = min(max(score_10, 1), 10) / 10.0`; else leave `score` as-is (legacy/back-compat).
- **`JUDGE_SYSTEM`:** rewritten as an explicit rubric, in this order:
  1. Criterion: self-containedness for a brand-new viewer (unchanged framing).
  2. Three numbered evaluation steps: (a) identify what the clip is about and why it matters;
     (b) hunt dangling references ("this/that/the previous equation") and assumed-but-never-
     introduced prerequisites; (c) if it is a worked problem, check statement/reasoning/result
     completeness.
  3. Anchored score bands: 1-2 incomprehensible without the source video; 3-4 major gaps (topic
     unclear OR key reference/prereq missing); 5-6 partially followable (topic clear, some gaps);
     7-8 fully understandable with minor rough edges; 9-10 flawlessly self-contained.
  4. "First write `reasoning`: 2-3 sentences applying the steps. Then set every boolean
     truthfully and `score_10` (1-10) per the bands."
  5. The existing boolean-gate explanations and the EXACT failure_reasons kind vocabulary,
     verbatim from the current prompt.
- **Fallback path** (`judge_clip` except-branch): return
  `JudgeVerdict(error=True, understandable=False, score=0.0, reasoning="judge unavailable")`
  instead of `JudgeVerdict(understandable=True, score=0.7)`. Keep the existing cross-model →
  authoring-model retry ladder above it.
- **`is_complete`:** returns False when `verdict.error` (guard added before the score check).
- **`validate_and_repair`:** on an error verdict, break the repair loop immediately (no
  expand/re-judge budget burn) and return the candidate as-is (ship-but-flag) — `cand.verdict`
  carries `error=True` downstream. The best-partial gate (topic/purpose/grounded/refs) treats an
  error verdict as NOT passing (those booleans are False on the fallback); the explicit error
  branch returns the candidate before that gate is consulted.

### `backend/pipeline/assemble/__init__.py` + `boundary_adapt.py` (warning plumbing)

- `snap_candidates` output dicts gain `"judge_error": bool(getattr(cand.verdict, "error", False))`.
- In assemble step 5 (where `boundary_score`/`final_quality` are computed), when
  `s["judge_error"]` append `"unjudged"` to `s["warnings"]` (creating the list if absent) BEFORE
  `boundary_score` reads it — the flag is user-visible and penalized like other warnings. The
  quality floor still applies: neutral 0.5 completeness means borderline unjudged clips can drop,
  which is intended (ship-but-flag, not ship-regardless).

### `backend/pipeline/assemble/scoring.py`

- `completeness_score(verdict, role, adapter)`: if `verdict.error`, return a neutral 0.5 (not
  top, not zero) so unjudged clips rank below judged-good ones and above judged-bad ones.

### `backend/eval/metrics.py`

- `comprehension(...)`: skip verdicts with `error=True` in both mean-score and rate; return an
  additional count so the caller can report coverage. New signature:
  `comprehension(specs, sentences, adapter, topic, threshold) -> tuple[mean, rate, n_judged, n_error]`
  (run_eval is the only caller; `judge_failures` gains the same skip-or-annotate treatment —
  annotate: include `error` in its per-clip tuple for --verbose visibility).

### `backend/eval/run_eval.py`

- `_measure`: unpack the new tuple; add `"judge_error_rate": n_error / (n_judged + n_error)`
  (0.0 when no clips) to the metrics dict. `comprehension_rate`/`mean_judge_score` now reflect
  judged clips only.

## Out of scope

Human calibration set (#6 — the only true validity measure; this change improves discrimination
and stability, not proven validity, and the spec says so). Token-probability score weighting from
the G-Eval paper (not exposed by Gemini structured output). Position-swap/pairwise bias
mitigations (judge scores clips absolutely, not pairwise). Repair-loop changes beyond the error
branch. The rest of the audit's integrity chain (robustness pkg 1, queued after this).

## Testing (offline, mocked `llm_json`)

New `backend/pipeline/assemble/tests/` (first tests for this package; punctuation/understand
convention):
1. Schema: `reasoning` is the first field in `JudgeVerdict.model_fields`; `score_10=8` →
   `score == 0.8`; `score_10=0` leaves explicit `score` untouched; `score_10` out of range clamps
   (12 → 1.0, min 1 → 0.1).
2. Prompt: JUDGE_SYSTEM contains all 9 canonical failure kinds, the five score bands, and a
   reasoning-first instruction.
3. Failure path: `llm_json` raising (both providers) → verdict `error=True, score=0.0,
   understandable=False`; `is_complete` False; `validate_and_repair` performs NO further judge
   calls after an error verdict (call-count assert), candidate kept, `"unjudged"` in warnings.
4. Scoring: `completeness_score` neutral 0.5 on error verdict.
5. Eval: `comprehension` excludes error verdicts (mixed list → mean/rate over judged only,
   counts correct); `judge_error_rate` computed in `_measure`.

## Verification / measurement (freeze harness)

1. One live `judge_clip` call: inspect raw response — `reasoning` generated before verdict fields
   and non-empty; `score_10` in 1-10.
2. A/B on frozen specs, judge-only variance: run the OLD-judge baseline (`--freeze-specs
   --runs 3`, 2-3 cached videos) BEFORE applying the prompt change, then the same command after —
   structures and specs are frozen, so the judge is the only variable; no old-prompt flag needs
   to be kept in the code. Compare: score distribution spread (expect wider than the
   0.7-clustered float), run-to-run flip rate (expect ≤ old), comprehension delta (report, don't
   over-read).
3. Full suite + compile clean; existing 88 tests unaffected (score normalization is additive).

## Risks

- Reasoning field lengthens judge outputs (more tokens/latency per call; JUDGE_WORKERS=4
  concurrency unchanged; flash-lite keeps it cheap).
- Score distribution will shift (bands ≠ old float habits): `JUDGE_MIN_SCORE=0.70` maps to
  score_10 ≥ 7 — the band text ("7-8 fully understandable") is written to match the existing
  threshold semantics deliberately.
- Cached verdicts in the assembly cache are per-run only (no cross-run persistence) — no
  migration needed.
