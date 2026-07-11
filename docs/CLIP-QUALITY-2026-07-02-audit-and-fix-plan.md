# Clip quality audit + fix plan — 2026-07-02

Produced by a 21-agent assessment workflow (live eval, kinematics end-to-end, clip-by-clip
inspection, 22-requirement design conformance, gap mining — every non-met claim adversarially
verified) plus a 10-agent research workflow (5 angles, every top finding source-checked by a
skeptic agent). Raw artifacts: `docs/audits/2026-07-02/` (per-agent JSON reports, full clip-text
dumps for all 4 audited videos, research findings with URLs + verification notes).

## Verdict

**The architecture is a faithful, working implementation of the structure-first design. The
output is not good yet.** The understanding layer (TreeSeg map, 61 well-labeled units on the
kinematics lecture, dependency edges) is solid; clips die in the selection → judge → repair
funnel. On the flagship kinematics use case the pipeline delivered ~1 of the 5 reference
outcomes. Production systems (OpusClip etc.) validate nothing — this pipeline's judge-gated
architecture is *ahead of the market in concept*; the failure is calibration and repair
mechanics, not design.

## Part 1 — Empirical state (measured this session)

### Frozen-trio eval baseline (`--freeze --runs 3`, all std = 0.000)
- comprehension_rate **0.333**, n_clips **0.667/video**, rejections_repair **7.667/video**
  (exact match to prior session; fully deterministic under freeze — judge noise did not reproduce).
- 23 of 25 candidate rejections at the repair stage; snap/quality_floor/max_clips reject nothing.
  The 6-stage ledger is functionally one judge+repair gate.
- carded_rate 0.000 everywhere; yfajBIaDf1Q ships 0 clips and its all-zero row silently drags the
  aggregate (should be NaN). Eval hardcodes `visual_summary=""` (metrics.py:40) — eval judging is
  blind even when perception exists.

### Kinematics end-to-end (dHjWVlfNraM, structure rebuilt, fresh assembly)
n_clips=4 (was 1 pre-redesign), comprehension 0.25, worked_example_completeness **0.0**.
- **Clip 1 (definition, 1.8–188s): GOOD** — meets its contract, judge 9/10, human agrees.
- Clip 2: merged span promises distance AND displacement answers, cuts before the second, ends
  semantically mid-sentence; shipped at final_quality **0.95** because the "claim" anchor role
  bypasses problem gates (judge said 4/10 — correctly).
- Clips 3+4: the two halves of ONE worked example (bus problem), severed with a 42s hole
  (the m→km→miles conversion) in neither clip, and `prerequisite_clips=[]` links them nowhere.
- **Wrongly killed:** bus part-(b) — a nearly ideal self-contained worked example (problem in its
  first sentence, givens, formula, substitution, result −28 mph with units + interpretation) —
  rejected at repair on a factually false `missing_problem_statement`. Both kinematic-equations
  candidates also died at repair.
- **Never candidates:** the practice→solution pair (temperature question). Six priority-0.70
  "claim" units crowded practice_prompt (0.52) out of MAX_ANCHORS=12. The particle worked example
  was unclippable because its results are labeled `calculation` — not an anchor role.

### Cross-video systemic failure modes (verified with file:line + full span text)
- **F1 Judge hallucinations are load-bearing**: killed clips citing "the previous equation"
  (phrase not in span), "that fog" as unresolved (introduced in-clip), "unclear subject" (subject
  verbatim in span) — while passing 9/10 clips that start AND end mid-clause, and a clip whose
  shipped text was never judged (min-duration extension added a garbled ASR power rule post-judge).
- **F2 Repair only grows**: 10–60s anchors inflated to 100–294s (some > the 240s ship cap — judged
  text that could never ship), then rejected for "jumps between topics". No trim move;
  `off_topic`/`other` verdicts produce zero repair targets and burn the budget after one call.
- **F3 No coverage/diversity in anchor selection** (flat priority sort, cap 12).
- **F4 Contract bound to anchor role**, so assembled worked problems skip problem gates
  (internal 0.95 vs judge 4/10 divergence).
- **F5 Judge validity unanchored**: eval/golden/ empty, all gold metrics NaN, same-family judge.
- **F6 Genre misdetection**: calculus parody song → "lecture" 0.9 (worked-problem gates killed the
  best verses); Tyson interview → "story" (crosstalk uncredited inside clips).
- **F7 Fabricated sentence boundaries**: caption-chunk fallback stamps `ends_with_period=True` on
  every chunk (sentences.py:392) → snapping is a no-op, mid-clause cuts score boundary=1.0.
- **F8 Cards never fire**: only generated when referential prereqs exist (zero shipped across the
  audit), generated AFTER the accept/reject gate, never judged — while eval measures clip+card.
- Dedupe tie-break dropped the corpus's best clip (complete 141s Tyson story arc lost to a
  judge-inflated 42s fragment that starts/ends mid-clause).

### Design conformance (22 requirements; all "non-met" verdicts adversarially confirmed)
Met: roles ontology (exact 20), units schema, dependency graph (10 relations), closure, contracts,
practice↔solution linkage, judge schema, bounded repair + ledger, grounded cards, adapters (11).
Partial: word timestamps synthesized on default supadata path; diarization coded but off (pyannote
not installed); treeseg map is 2-level (no subtopics); no pause/speaker/scene signals or pre-roll
at boundary time; judge never sees frames and **eval judge gets empty visual summary**; frontend
renders neither cards nor watch-first hints (data already in payload); golden dir empty → role
accuracy / anchor recall / boundary error permanently NaN.

## Part 2 — Fix plan (waves, with research-backed recipes)

Confidence key: sources fetched + verified by a skeptic agent; overclaims noted inline.

### Wave 1 — Judge integrity (attacks the 92% kill gate from both directions)
1. **Quote-then-verify failure reasons** (validate.py). Add `evidence_quote` to FailureReason;
   judge must copy exact span text evidencing each failure; verify post-hoc with normalized string
   containment; **unverifiable reasons cannot kill** (downgrade to warning, may still steer
   repair). Basis: Datadog production hallucination judge mandates per-claim quotes (F1 0.810
   HaluBench); requirement-hallucination is the documented driver of LLM over-rejection
   (arXiv:2603.00539). NOTE (verifier): the containment check itself is our sound-by-construction
   extension, not a measured pattern — track `phantom_verdict_rate` in eval to measure it.
   ~30 lines, zero extra LLM calls. FailureReason already has `reference_text` — check that too.
2. **Asymmetric 3-outcome gate**: ACCEPT (as today) / SHIP-WITH-FLAG (failures exist but none
   citation-verified) / REJECT (≥1 verified reason AND confirmed). Basis: judges measure TPR 96%
   vs TNR <25% — they over-flag massively (arXiv:2510.11822); kill mandates + deterministic
   validation gates kill 79–83% of false defect claims (Refute-or-Promote, arXiv:2604.19049).
3. **Fresh-context kill confirmation**: one NEW llm_json call per would-be rejection (~8/video,
   ≈$0.002 total) restating each failure as a neutral question against the transcript; only
   confirmed reasons count. Fresh context beats same-session self-check (CoVe arXiv:2309.11495,
   +23% F1; Song et al. p=0.008). Verifier note: prefer open-form restatement + quote request over
   strict yes/no (CoVe's own ablation favors it).
4. **Re-judge on any post-judge text change** (min-duration extension, trim, cap — not just
   merges), and **key the verdict cache on a hash of the judged text**, not frozenset(unit_ids).
5. **Clamp closure/repair span budget to the 240s ship cap** (config.py:213 vs :123 — one line).
6. **Do NOT build a Gemini panel**: 9-judge/7-family panels yield only ~2.2 effective votes
   (arXiv:2605.29800); a same-family second vote is nearly pure correlation. If a non-Google key
   ever works, add exactly one cross-family judge as a kill-veto.

### Wave 1-parallel — Calibration bootstrap (mostly human time; de-risks all tuning)
The strongest cross-angle warning (SciTalk, arXiv:2504.18805): an agentic loop optimized its own
scores for 5 iterations while human readability fell 3.8→2.2. Every threshold here is currently
tuned against an unvalidated proxy.
7. **YouTube chapters as free segmentation gold** (YTSeg recipe, arXiv:2402.17633): import
   creator-provided chapters via yt-dlp into eval/golden/, add Pk/WindowDiff to metrics.py —
   un-NaNs boundary gold with zero labeling. Bonus check: no clip should straddle a chapter.
8. **Corruption-based judge probe** (LLMBar pattern, arXiv:2310.07641): auto-generate known-bad
   variants of accepted clips (chop first/last sentence, remove antecedent unit via refers_to
   edges, splice off-topic unit) + paraphrase controls; measure per-gate TPR/TNR with zero human
   labels. Treat probe TPR as an upper bound.
9. **Hamel protocol for the human set** (hamel.dev/blog/posts/llm-judge): ~30 binary
   understandable-in-isolation labels + one-line critiques (label REJECTED candidates too — that's
   where the 7.7/video die), iterate the judge prompt to >90% agreement; then grow to ~100–150:
   50 random-stratified core (for κ + PPI) + targeted tranche (score band 4–7, judge-config
   disagreements, ≥10 labels per failure kind). Labeling protocol per decontextualization research
   (TACL 2021): binary FEASIBLE/INFEASIBLE + "what one line would you need to be told first?"
   (that answer is context-card gold). **Expert ceiling on this criterion is κ≈0.51 — κ~0.5 vs the
   judge is success; do not chase 0.8.** Decision rule: κ<0.4 replace judge, ≥0.6 trust and tune.
10. **Spend the labels**: re-fit the 0.70 gate by alignment sweep; per-failure-kind trust weights
    (kill authority only for kinds with precision ≥0.7 on ≥10 labels); use judge scores as ranks
    within a video, calibrate only the binary ship decision (arXiv:2406.12624). PPI++/EIF
    (arXiv:2601.05420) turns judge-on-everything + 100 human labels into unbiased comprehension
    estimates with valid CIs. Escalation hatch if κ stays low: cascade borderline verdicts to a
    stronger judge (arXiv:2407.18370); A/B a 0–5 scale variant (arXiv:2601.03444).

### Wave 2 — Repair rework + coverage (turns surviving candidates into the RIGHT clips)
11. **Judge the anchor at native size FIRST** — today _fill_contract inflates before the first
    judge call; the 10–60s anchors were never scored unexpanded. Then expand only as needed
    (cheapest-first). Bidirectional refinement is the norm in temporal grounding (TimeRefine,
    WACV 2026, arXiv:2412.09601); grow-only is an anomaly.
12. **Add trim moves** (`trim_head`/`trim_tail`, never the anchor or contract-required units).
    Route by verdict: unresolved-ref/missing-setup → grow toward antecedent; multi-topic/tail
    complaints → trim the offending edge. Search the trim lattice by noisy bisection between
    known-good (anchor) and known-bad (full span) — a noisy monotone oracle needs only
    O(log n / ε²) queries (Karp–Kleinberg); the frozenset verdict cache makes revisits free.
    Validate the monotonicity assumption on frozen structures first.
13. **Coverage-constrained anchor selection** (candidates.py): phase 1 take the best anchor from
    EACH content-map topic node; phase 2 fill remaining slots with saturation penalties
    (topic_decay^k · role_decay^k). This is relevance+coverage selection (AKS CVPR 2025
    arXiv:2502.21271; MDP3 arXiv:2501.02885) and quota-greedy is partition-matroid-principled.
    Add a chapter-coverage metric to eval.
14. **Worked-example ARC detection** (new understand/arcs.py): deterministic scan for
    (example_setup|problem_givens|practice_prompt) → worked_step/calculation+ → (result|solution),
    ≤2 interleaved non-arc units; arcs bypass `is_anchor_role` and enter selection as synthetic
    result-role anchors; verify with one MathNet-style call that returns only unit ids
    (arXiv:2604.18584). This is the only fix that recovers examples whose results are labeled
    `calculation`, and it feeds practice↔solution pairing across distance (shared concepts).
15. **Contract-by-content with rebinding** (contracts.py): choose the contract from roles present
    in the ASSEMBLED span (problem-shaped contracts dominate when setup+steps+answer are present),
    re-choose after every repair mutation, and compute final_quality under the same contract the
    judge used. Per-instance rubric evidence: HealthBench arXiv:2505.08775, RaR arXiv:2507.17746.
    Closes the 0.95-vs-4/10 divergence.
16. **Dedupe tie-break**: prefer complete arcs passing all hard gates over judge-score-inflated
    fragments; on containment, keep the span with greater contract/role coverage, not higher
    final_quality alone. Link or merge severed contract-completing adjacent clips (kinematics 3+4).

### Wave 3 — Boundaries + cards (what ships stops cutting mid-clause and cold-opening)
17. **Kill fabricated sentence ends**: run SaT (wtpsplit-lite, ~150ms/page CPU, punctuation-
    agnostic, EMNLP 2024) on caption/window fallback paths; `ends_with_period` only at SaT
    boundaries. Better long-term: **local ASR replacing supadata captions** — parakeet-tdt-0.6b-v3
    via parakeet-mlx (Apple Silicon, native punctuation + word timestamps, 6.34% WER, CC-BY;
    25 European languages — keep WhisperX as fallback). Auto-trigger: if <20% of caption chunks end
    in terminal punctuation, force the ASR path.
18. **2-of-3 boundary gate**: a clip-grade boundary needs 2 of {restored terminator, SaT boundary
    ±1 word, inter-word pause ≥0.25s} (speech punctuation restorers run ~0.62 F1 on spontaneous
    speech — one signal is never enough). Pause veto numbers: 88% of sentence-boundary pauses
    >250ms (ICPhS 2003); place the physical cut inside the pause (replaces fixed tail_pad_s).
    Cheap guards: reject end boundaries whose final sentence has <3 words or ends on a
    conjunction/preposition.
19. **Precise cuts by forced alignment, not WhisperX**: targeted CTC forced alignment (torchaudio
    forced_align, pin <2.9, or MFA CLI) on ±10s boundary windows with the KNOWN text. Measured
    word-boundary error: MFA ~20–25ms, MMS ~43–50ms, WhisperX ~110ms (arXiv:2606.18466).
20. **Cards always-on, generated BEFORE the gate**: every clip gets a one-line orientation card
    (suppress only when the clip's first sentence already names the subject); judge the clip+card
    (aligns gate with eval); card-as-repair for prerequisite/reference-only failures (accept
    card-completed clips without growing). Every production system ships an orientation device on
    every clip; ReelsEd's RCT (n=62, quiz 93.85% vs 79.72%) shipped a summary+label per reel.

### Wave 4 — Video-native judging + genre + surfacing
21. **Video render-audit at the final gate**: judge the actual rendered clip (or source URI +
    `video_metadata` start/end offsets via generateContent — NOT the Interactions API, which lacks
    it) with a video-capable Gemini. Cost ≈$0.015 per 240s clip with audio; run on survivors only
    (~$0.01–0.30/video). Adds gates text judging structurally cannot check:
    `starts_clean_audio`, `ends_clean_audio`, `visuals_referenced_are_visible`. MLLM judges beat
    text-only on video tasks (VideoJudge arXiv:2509.21451 — analogical evidence; VF-Eval warns
    they're imperfect, so keep programmatic gates). Cheaper variant first: **edge probe** — judge
    only the first/last ~8s (~$0.0006/candidate) for mid-sentence starts/ends; this alone catches
    the F7 blind spot by listening to the audio.
    **Guardrail confirmed**: do NOT replace the structure stages with a whole-video VLM pass —
    hour-scale timestamp grounding is still unreliable in mid-2026 (ExtremeWhenBench
    arXiv:2606.12300: monolithic 0.053 mIoU vs 0.354 retrieve-then-ground; 85% of failures are
    search failures). The transcript-anchored architecture stands.
22. **Genre detection ensemble** (precedence: metadata > audio > transcript stats > LLM):
    yt-dlp `categories/genre/track/artist` (free, already downloaded); inaSpeechSegmenter music
    fraction (singing tagged as music — catches the parody; MIREX-winning, production-grade);
    full-transcript repetition features + stratified head+middle+tail sampling (detect.py
    currently samples only head+tail); diarization turn stats for interview-vs-story (≥2 major
    speakers + ≥1.5 switches/min → interview). New entertainment adapter: single "moment"
    contract (hook/punchline/quotable), core verdict fields only — when unsure, gate less.
23. **Reconciliation pass** (one flash-lite call per candidate): cross-check global genre label vs
    the span's actual unit-role sequence; contract override + genre challenge on contradiction
    (pattern from arXiv:2509.16811, whose reconciliation pass was ablation-critical).
24. **Surfacing (pkg-2 scope A)**: render context cards, watch-first chips, and quality badges in
    the frontend (data already in payload; verdict/warnings/final_quality need one whitelist
    addition in _build_embed_clips). Fix the embed `ceil(end)+1` bleed. Eval fixes: pass real
    visual_summary; report 0-clip videos as NaN.

## Part 3 — Acceptance criteria

1. **Kinematics acceptance test** (rerun after each wave): target the 5-clip reference outcome —
   definition ✅ / equations / complete worked example / practice / linked solution. Currently 1/5.
2. **A/B discipline**: `--freeze --runs 3` before/after every change (variance floor measured at
   exactly 0.000 — any delta is real). Track rejections_repair (expect ↓ from 7.7), n_clips
   (↑ from 0.667), phantom_verdict_rate (new), verified-kill vs unverified-kill rates (new),
   chapter-coverage (new).
3. **Calibration gates**: judge–human κ ≥ 0.4 required before trusting any threshold tuning
   (κ≈0.5 is the realistic ceiling); block judge-prompt changes that reduce κ; corruption-probe
   per-gate TPR/TNR reported alongside.
4. **SciTalk rule**: any change where judge scores improve while proxy human-quality features
   degrade (span length ballooning, boundary regressions, card redundancy) is a stop signal.

## Wave 1 outcome (implemented + verified 2026-07-02, same day)

All of Wave 1 + the calibration-bootstrap code items shipped (267 tests, was 149; every task
adversarially reviewed; whole-change reviews clean). A follow-up closed the one substantive
verification finding: post_merge/post_snap kills now go through the same verified+confirmed gate.
Measured (live, `--freeze --runs 3` vs the pre-change baseline):
- **Kinematics: n_clips 4→5, comprehension 0.25→0.40, worked_example_completeness 0.0→1.0**,
  repair rejections 5→3. phantom_verdict_rate measured 0.227 (22.7% of judge failure citations
  failed quote verification — the phantom-kill surface is real and now instrumented).
- **Trio: n_clips 0.667→1.111, rejections_repair 7.667→3.667, unverified_kill → 0.000**
  (structural zero at all judge stages — keep as a regression tripwire).
- **Honest negative**: trio comprehension 0.333→0.056 — recovered clips ship flagged
  (`unverified_judge_concerns`) and low-scored rather than dying silently. The probe says the
  judge's own scores are the weak instrument here (TNR 0.20 on its own accepted clips;
  `reasoning_complete`/`result_complete` the least valid gates; chop-end corruptions outscore
  originals) — so treat the comprehension slide as unmeasured-quality, not confirmed-bad, until
  the human calibration labels arbitrate. Label repair-gate kills + the 4-7 score band first.
- NjvwWiCYLl4's baseline clip (the mid-clause 42s fragment) is now confirm-killed — plausibly a
  CORRECT kill; its genuinely best clip (the 141s story arc) is blocked on the Wave 2 dedupe
  tie-break, and its boundary problem on Wave 3.
- Judge nondeterminism at temperature 0 reappeared under freeze (same video diverges between
  solo and trio invocations) — the variance floor is no longer 0.000; use `--runs 3` stds.
- Design amendment (user-directed): Wave 2's anchor selection becomes an **extraction-plan
  step** — the model proposes what is worth extracting per video (which moments, what role,
  what completeness each needs) from the actual unit inventory, with the adapter as a prior
  rather than a hard rule; the deterministic layer enforces coverage/quotas/dedup/contracts.
  The labeling protocol gains a per-video "what would YOU clip?" question.

## Wave 2 outcome (implemented + gated 2026-07-02 night)

All six tasks shipped and reviewed (418 tests, was 267); the whole-change review caught one major
— calculation/derivation roles couldn't satisfy the worked-example contract (fixed in base.py AND
lecture.py's override). Gate vs post-Wave-1: trio yield 1.111→1.444 clips/video; kinematics 5→6
clips including a COMPLETE merged worked example (givens→steps→result, the audit's bus part-b);
repair kills 3→1; straddle 0.40→0.167; plan engine active (0 fallbacks); tripwire holds.
Honest negatives + open items: kinematics scorecard STILL 2/5 — the equations candidate is now
GENERATED (plan works) but dies verified+confirmed at post_snap_judge on missing_prerequisite/
reasoning/result (card-as-repair, Wave 3, is the designed rescue); no practice clip ships;
severed_pairs_linked=0 despite 3 arcs (linker conditions too narrow); n_trims=0 everywhere (trim
routing never fires — tuning item); kinematics comprehension 0.40→0.167 single-run (two flagged
clips drag the mean — judge validity remains THE open question; labeling tooling now live at
/static/labeling/). uqw parody song 1→0 clips: honest verified kill of the stale garbled
fragment (genre handling = Wave 4).

## Coverage study (2026-07-02 evening): the qP-9wwRrJbg case → Wave 2.5

User test: "AP Physics 1 - Unit 1 Review - Kinematics" (qP-9wwRrJbg, 23.9 min, topic query
"kinematics") shipped **2 clips**. Reference inventory (verified, saved as
`docs/audits/2026-07-02/qP-9wwRrJbg-coverage-gold.json`): **26 items, ideal ~20 clips**
(5 definitions, 6 concept/explanations, 4 real worked examples, 2 exam tips, 2 procedures,
1 interpretation…). The structure layer understood nearly all of it (139 units); selection
starved the judge. Full analysis: `docs/audits/2026-07-02/{inventory,gap_analysis}.json`.

Root causes (Wave-1 code, evidence in gap_analysis.json):
1. **Stable-sort front-loading**: the 27-way claim-priority tie broke by list order = time, so
   all 12 anchors sat in the FIRST 281s of a 1432s video. Nothing after 551s could ship.
2. All 12 practice_prompts mathematically unreachable under MAX_ANCHORS=12 (52 units outrank the
   best one) — though inventory shows they're 1-6s Socratic checks, NOT real practice items; the
   real value is 5 worked examples, 3 of which arcs.py cannot see (terminals labeled
   worked_step/claim/physical_interpretation, not result/solution).
3. **Dedupe never refunds the cap**: 12 anchors collapsed to 3 specs; 9 slots evaporated while 63
   eligible units sat untouched.
4. Topic filter: "kinematics" on an all-kinematics video is a no-op that still killed
   tangential-but-valuable items (sig-figs exam tip, the unit-conversion worked example).
5. Judge kills ≈ 0 in this run — and the judge PASSED two multi-item mashes (no atomicity gate).
6. Graph: 286 edges exist (earlier "1 edge" was a key-counting artifact) but requires-density is
   HALF the comparison video's — concepts_introduced sparse (20/139 units) and the LLM edge pass
   is prompt-forbidden from emitting 'requires' (dependencies.py:163-165).
Wave 2 projection (assessed against its actual code): 2 → ~8-12 clips with all 12 topics
covered, but the 12-cap makes the ideal 18-22 structurally impossible.

**Wave 2.5 — coverage package** (research verified, sources in
`docs/audits/2026-07-02/{research,verification}.json`):
1. Content-driven budget: saturation-stopped inventory pass (re-ask "important items not yet
   listed" until <~15% new; arXiv:2404.04068) sets budget = clamp(inventory, 8, 32); MAX_ANCHORS
   becomes a safety rail. Threshold (score ≥ τ) selection instead of top-K (GMR precedent).
2. **Refund loop** (CCQGen pattern): after dedupe, re-select from uncovered units until budget or
   residual exhausted — in reconstruction this alone turns 3 specs into ~10-12.
3. Cost stays flat via batching: HiVid-style batched unit-saliency pre-filter (10 units/call;
   m=2→m=10 cut costs ~6x) + batched first-pass judging (30-82% cheaper, ~3pp loss — A/B verdict
   agreement first; corrected numbers per verification.json) + criteria injection (free accuracy,
   arXiv:2604.13717) + deterministic pre-judge QC. 30 candidates can cost LESS than today's 12.
4. Universal-importance blend: genre-agnostic teachable-moment rubric score (Prompts-to-Summaries
   SOTA-unsupervised cross-genre) blended with role priority so taxonomy-invisible items
   (exam tips, misconceptions) can win slots. Schema-library + plan-proposed novel roles (ASEE).
5. Arc robustness: broaden terminal roles + minimum-substance filter (3 of 5 real examples
   invisible today; 4 of 5 detected "arcs" are 2-9s Socratic non-clips).
6. Atomicity: judge gains an over-inclusion/one-idea criterion; mash-splitter prefers ≤300s
   single-idea clips (engagement evidence: <5min clips, p<.05, n=152 observational).
7. Topic-filter semantics: query ≈ video topic → switch to "everything important" mode
   (relevance short-circuit); expose clip count as a NotebookLM-style dial (auto default).
8. Graph nutrition: prompt nudge so definition/equation units always name concepts_introduced;
   allow the LLM edge pass to emit 'requires'.
9. **Omission audit with a guarantee** (CARE, arXiv:2606.08969): conformally calibrated
   bounded-miss rate over ~100 labeled units (matches the planned golden set) — turns "did we
   miss something important?" into a monitored metric. New eval: inventory recall vs the
   qP coverage-gold file; chapter/node coverage; clips-per-ideal ratio.
Quick wins shippable before the full package: cap scaling by density/eligible-unit count,
tie-break by topic spread, per-role cap in the legacy selector, relevance bypass, refund loop.

## Artifact index

- `docs/audits/2026-07-02/eval_baseline.json` — full metric tables, per-video + aggregate.
- `docs/audits/2026-07-02/kinematics_e2e.json` — structure overview, all 4 clips with FULL span
  text/cards/verdicts, all 7 rejections, per-clip human judgment.
- `docs/audits/2026-07-02/trio_inspection.json` + `{uqwC41RDPyg,NjvwWiCYLl4,yfajBIaDf1Q}.json` —
  clip dumps and 12 cross-video systemic findings.
- `docs/audits/2026-07-02/design_conformance.json` — 22 requirements, file:line, verified.
- `docs/audits/2026-07-02/open_gaps.json` — 11 gaps ranked, top 7 adversarially verified.
- `docs/audits/2026-07-02/{judge-reliability,repair-boundaries,coverage-selection,calibration-methodology,field-scan}.json`
  — 39 research findings with URLs, adoption recipes, and skeptic verification notes; ~53 leads.
