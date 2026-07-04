# Clip integrity chain (robustness pkg 1) — design

**Date:** 2026-07-01 · **Status:** approved-pending-user-review · **Source:** high/medium findings
from `docs/clip-editing-robustness-audit.md` (6-agent audit); package 1 of 3 agreed with user.

## Problem

The judge approves a clip's exact text, then the pipeline mutates it with no accounting:
1. **unit_ids lie** — closure inlines a far unit but only adds that unit's id; every unit
   physically between anchor and far unit is in the clip yet invisible to contract check,
   grounding, repair targeting, and cards (audit: high/M). Repair also never removes inlined
   ids from `referential`, so cards restate in-clip content (medium/S).
2. **Same-facet merge ships a never-judged span** carrying one side's verdict/unit_ids/card
   metadata (high/M).
3. **Five silent drop stages** — repair-None, snap-None, dedupe loser, quality floor, max_clips —
   discard candidates with no recorded reason; n_clips run-to-run variance is undiagnosable
   (medium/M; directly feeds the eval-noise problem measured earlier today).
4. **Cards fail silently** — `generate_context_card` returns "" on LLM failure or when the
   grounding filter zeroes it; closure's `truncated` flag never reaches the spec (high/M).
5. Smaller mutations (min-duration extension, start trims, whisper drift) change judged content
   with no warning `boundary_score` can see (high/M, partially).

## Decisions (locked with user)

1. **Scope: Core + cards.** Truthful unit_ids + mutation re-judge/warn + drop-reason ledger +
   card integrity. Sequencing/renumbering staleness and clip identity → pkg 2. (Judge-outage
   honesty already shipped with the judge-rubric work.)
2. **Hybrid re-judge policy.** Re-judge ALWAYS on same-facet merge (union span never judged;
   verdict cache keyed on unioned `frozenset(unit_ids)` makes it cheap/correct). Warn-only +
   metadata repair for extension/trim/whisper mutations.
3. **Ledger surface:** structured rejections returned by `assemble_clips` (third tuple element),
   printed by CLI, per-stage numeric counts in eval metrics. Job-artifact persistence → pkg 2.
4. **`refine.py` (shared with the legacy fast path) gets only ADDITIVE warning strings; its
   merge/dedupe behavior is untouched.** Part B gets its own metadata-aware dedupe.

## Changes

### New module `backend/pipeline/assemble/integrity.py` (pure, offline-testable)

- `true_contents(unit_ids: list[str], referential: list[tuple[str, str]], units: list[Unit],
  i_start: int, i_end: int) -> tuple[list[str], list[tuple[str, str]]]` — returns
  (unit_ids ∪ every unit whose sentence_range lies within [i_start, i_end], time-ordered;
  referential minus entries whose unit now lies inside the span). Idempotent.
- `merge_partb(a: dict, b: dict, units_by_id: dict, sentences) -> dict` — Part-B-aware merge of
  two overlapping same-facet specs: union span (min start/i_start, max end/i_end), union
  `unit_ids` + `referential` then `true_contents`, union warnings + `"merged_overlap"`,
  `judge_error` OR'd, `truncated` OR'd, all other keys from the higher-`final_quality` side;
  sets `"merged": True`.
- `@dataclass Rejection`: `cand_id, title, role, stage, reason, score (float|None),
  failure_kinds (list[str]), final_quality (float|None), start, end`. `stage` ∈
  `{"repair", "snap", "dedupe", "post_merge_judge", "quality_floor", "max_clips"}`.

### `candidates.py` / `validate.py` — truthful unit_ids at every span change

- `build_candidate`: after the span is computed (min/max over inline units), run `true_contents`
  and store the results on the Candidate.
- `expand_candidate`: the final `replace(...)` passes
  `unit_ids, referential = true_contents(new_ids, cand.referential, units, i0, i1)` — fixes both
  absorption and the stale-referential finding in one site.
- `validate_and_repair`: when the loop ends with a FAILING final verdict and returns None, the
  caller needs the last verdict to build the Rejection — change return to the candidate-or-None
  plus enough info: simplest contract-preserving option: on the None path, attach the last
  verdict + reason to a `rejection_info` dict the caller (assemble step 3) reads from the
  closure variable it already has (`cand.verdict` on the local candidate object it passed in is
  NOT visible after None-return) — therefore `validate_and_repair` returns
  `tuple[Optional[Candidate], Optional[Rejection]]`. Single caller (`assemble/__init__.py`)
  updated.

### `boundary_adapt.py` — Part-B dedupe + snap accounting

- `snap_candidates(cands, sentences, settings) -> tuple[list[dict], list[Rejection]]`:
  - `_snap_one` returning None → Rejection(stage="snap", reason="unsnappable (end<=start)").
  - Replace the `refine._dedupe` call with local `_dedupe_partb(specs, sentences, min_dur)`:
    same overlap/IoU decisions as `refine._dedupe` (reuse its helpers where importable without
    behavior change), but the loser of a keep-one decision → Rejection(stage="dedupe",
    reason="overlap loser to <winner cand_id>") and a same-facet merge goes through
    `merge_partb`.
  - `candidate_to_boundary_input` additionally carries `"truncated": cand.truncated`.
- Existing `"unjudged"` warning flow unchanged.

### `assemble/__init__.py` — re-judge hook + ledger assembly + signature

- Step 3 collects repair Rejections; step 4 unpacks `(specs, snap_rejections)`.
- NEW step 4b — hybrid re-judge: for each spec with `"merged": True`: recompute text from
  `sentence_start_idx..sentence_end_idx`, one `judge_clip` (the step-3 verdict cache dict +
  lock are passed down so the unioned-unit_ids key can hit), then refresh
  `completeness_score`/`final_quality`. Hard-core gate (topic ∧ purpose ∧ grounded ∧ refs, same
  as the best-partial gate) fails → Rejection(stage="post_merge_judge") and the spec is dropped.
  Judge outage (`error=True`) → ship-but-flag exactly like the rubric work (unjudged warning
  already flows via `judge_error`).
- `drop_weak` losers → Rejection(stage="quality_floor", reason="final_quality < floor");
  `max_clips` slice losers → Rejection(stage="max_clips").
- **Signature: `assemble_clips(...) -> tuple[list[dict], str, list[Rejection]]`.** Exactly three
  call sites (grep-verified): `orchestrator.py:202` (invoked through the thread wrapper — the
  unpack site), `cli.py:80`, `eval/run_eval.py:260`. CLI prints one line per rejection
  (`[dropped/<stage>] title (score=…, kinds=…)`). `notes` string unchanged.

### `refine.py` + `boundary.py` — additive warnings only (fast-path-safe)

- `_snap_one` min-duration extension loop: append `"extended_for_min_duration"` when it extends
  past the judged end.
- `_trim_start_after`: append `"trimmed_start"` when it moves a start.
- `boundary.py::_resolve_overlaps`: append `"trimmed_start"` on trim.
- `scoring.boundary_score` penalties: `+0.10 * ("extended_for_min_duration" in w)`,
  `+0.10 * ("trimmed_start" in w)`, `+0.10 * ("missing_context_card" in w)` (existing penalties
  and the `unjudged` 0.15 unchanged; clamp already in place).

### `context_card.py` — card integrity

- `generate_context_card(spec, units_by_id, adapter, topic)`: after the existing LLM +
  grounding-filter path, if the card is EMPTY and (`spec["referential"]` non-empty or
  `spec.get("truncated")`): extractive fallback — the FIRST referential entry's unit `summary`
  (closure emits referential in deterministic frontier order; verbatim, word-capped at
  `CONTEXT_CARD_MAX_WORDS`); grounded by construction. Empty/missing summary → try the next
  referential entry; none usable → warning path below. If
  the card is still empty after fallback → append `"missing_context_card"` to
  `spec["warnings"]`. Cards are generated after step 4b, so merged clips get cards from their
  UNION metadata (fixes the merged-card staleness by construction).
- Defensive (stale-referential belt-and-braces): skip any referential uid present in
  `spec["unit_ids"]` when building the card.

## Out of scope

Sequencing renumbering / dangling prerequisite_clips + clip identity (pkg 2); judging
post-snap-always (architecture C); job-artifact persistence of rejections (pkg 2); any
`refine._dedupe`/`_merge` behavior change (legacy fast path).

## Testing (offline; mocked `llm_json` where judging is involved)

`backend/pipeline/assemble/tests/test_integrity.py`:
1. `true_contents`: absorbs spanned units time-ordered; drops inside-referential; idempotent;
   leaves outside-referential alone.
2. `merge_partb`: unions span/unit_ids/referential/warnings; `judge_error`/`truncated` OR;
   winner metadata from higher final_quality; sets merged flag + warning.
3. `snap_candidates` rejections: unsnappable → stage=snap; overlap loser → stage=dedupe with
   winner reference; merged spec flagged.
4. Re-judge hook: merged spec re-judged once (mock counts calls; cache-hit path asserted by
   pre-seeding the cache with the unioned key); failing hard-core → stage=post_merge_judge
   rejection; outage → ship-but-flag.
5. Ledger: assemble-level accounting for quality_floor and max_clips stages (pure: feed specs
   through the step-5 logic).
6. `expand_candidate` replace passes true_contents (absorbed ids present; inlined referential
   removed).
7. Cards: empty-LLM → extractive fallback from referential summary; no referential + truncated
   False + empty → `missing_context_card` warning; referential inside unit_ids skipped.
8. Warnings: `boundary_score` penalties for the three new strings.
9. Regression: full suite (106) stays green; `validate_and_repair` tuple-return updated in its
   existing tests.

## Verification / measurement

1. Frozen A/B: `--freeze --runs 3` before/after on 3 cached videos — expect: rejections_* columns
   explain every n_clips delta (sum of kept + rejected constant per run), comprehension within
   noise, no new judge_error.
2. CLI smoke on a cached video: rejection lines print; merged clips (if any) show
   `merged_overlap`; cards non-empty or explicitly warned.

## Risks

- `_dedupe_partb` duplicating refine's overlap semantics risks drift — mitigate by reusing
  refine's small helpers (`_overlap_frac`/IoU math) where importable, and by a
  characterization test asserting identical keep/drop decisions on a non-merge fixture.
- Re-judge on merge changes clip survival (stricter) — measured, and rejections make it visible.
- `assemble_clips` signature change touches 3+ files — grep-verified in the plan.
