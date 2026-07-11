# Next steps — 2026-07-03 (latency pass + Wave 3/4 follow-up tail)

Written at the end of the 2026-07-03 session (`docs/SESSION-2026-07-03-summary.md`). Priorities in
order. The #1 item is the user's explicit ask; the rest are documented follow-ups. Suite is **656**
passing at handoff; hard invariant **`unverified_kill = 0`** must stay.

---

## 1. LATENCY PASS — ~20 min/video → fast (THE priority; user-directed)

The user confirmed **output quality is acceptable** (the judge over-flags); latency is the product
blocker. **Do NOT chase judge scores.** Approach = profile → target the real sink → measure.

### 1a. Profile first (don't guess-optimize)
Instrument per-stage wall-clock. The orchestrator already emits SSE stage events
(`transcribing / punctuating / perceiving / segmenting / assembling / refining / cutting`,
`orchestrator.py`); add per-stage start/end timestamps to a log (or a `stats["stage_timings_ms"]`)
so ONE real run prints the breakdown. Run on a fresh (uncached) ~20-min video once, read the split.

### 1b. Levers, ranked (apply after profiling confirms the sink)
- **Multimodal perception is the prime suspect + it's OPTIONAL** — `perceive()` downloads the
  video + ffmpeg scene/keyframe extraction + Gemini-vision batches. It only feeds visual captions
  into judging/cards. **DECISION PENDING FROM USER: is the visual analysis worth keeping?** If not,
  gate it off by default (`analysis_profile=fast` or `MULTIMODAL=0`, both already exist in config)
  — likely the single biggest cut. (Zero quality regression per the user's "quality is fine"
  verdict, since their test HAD it on and they liked the output either way — confirm.)
- **Understanding runs SEQUENTIALLY** — `build_structure` does content_map → extract_units →
  dependencies in series (`understand/build.py`); `extract_units` is many LLM calls over ~180
  sentences. Parallelize independent chunks / overlap the sub-stages. **Zero quality change.**
- **Assembly concurrency is low** — `JUDGE_WORKERS = 4` (+ `PUNCT_WORKERS = 4`) is conservative for
  network-bound LLM calls. Bumping to ~8-12 is pure speed BUT **check the user's Gemini tier /
  RPM limits first** (free-tier flash-lite will 429 under high concurrency). Make it env-tunable;
  don't blindly raise it.
- **Structural (bigger wins, specced in the Wave-2.5b backlog):**
  - **Batched first-pass judging** — judge N units/candidates per call (HiVid/batch-judging;
    30-82% fewer calls) and pre-seed the verdict cache; A/B verdict parity via `judge_probe` first.
  - **Persistent disk verdict cache** — the in-memory cache dies with the job; a disk cache makes
    re-runs of the same video near-instant.
  - **Reuse cached structure on repeat videos** — the freshness-gated `load_structure` (W25-A)
    already skips understanding when a fresh cache exists; expose a "reuse analysis" affordance so
    a repeat video doesn't re-understand.
- Measure wall-clock A/B on one video after each change; the structure-rebuild is the dominant
  run-to-run noise, so time the stages, not just end-to-end.

### 1c. Watch-outs
- Gemini rate limits (429) under raised concurrency — back off / cap.
- Don't break the graceful-degrade path (`_run_full` → `_run_fast` on failure).
- Perception is cached per-video (`work/<id>/…`) — a second run of the same video already skips it;
  the 20 min is the FIRST (cold) run.

---

## 2. Wave 3/4 follow-up tail (default-OFF or golden-set-dependent — NOT yet built)

Full specs in `docs/superpowers/plans/2026-07-03-wave-3-4.md` + the research JSON. All must keep
`unverified_kill=0` (advisory/gate-less or accept-side only).
- **GEN5 — per-candidate reconciliation** (the *real* parody fix): GEN2 routes a parody to the
  lenient adapter, but a line the labeler tags `result` can still bind the generic result contract
  and hit problem gates. GEN5 cross-checks genre vs the span's role-sequence and overrides the
  contract to `moment`. Adds a per-candidate LLM call → **gate it on low genre confidence / tier
  conflict** to bound latency+cost. **Verify with a live parody run whether GEN2 alone already
  ships good clips before investing.**
- **GEN3 / GEN4** — genre ensemble depth: transcript-stats (repetition/rhyme) + head+**middle**+tail
  sampling (GEN3); a one-call Gemini-audio music tier (`audio_part` helper + `detect_audio.py`,
  GEN4). NOT inaSpeechSegmenter.
- **VID3 — render-audit (video tier 2)**: Files-API upload of the source once + per-survivor offset
  calls (`video_metadata` offsets), `VideoVerdict{visuals_referenced_are_visible,…}`, warning+dock
  advisory. `VIDEO_JUDGE_ENABLED`/`VIDEO_JUDGE_MODEL`/`VIDEO_MEDIA_RESOLUTION` config knobs are
  **already reserved** (default OFF). ~$0.01-0.30/video, survivors only.
- **BND2 — whisper re-timing**: repurpose the existing faster-whisper window to RE-TIME the
  LLM-chosen boundary + cut inside the following pause (replaces the WhisperX re-segmentation +
  fixed tail_pad). Needs audio download to succeed; measure vs BND1 on a golden set.
- **ASR1 — parakeet-mlx opt-in transcriber** + probe-first auto-trigger (<20% chunks terminal-
  punctuated). parakeet-mlx 0.5.2 installs clean on Apple Silicon (pure MLX, no torch). Keep
  `TRANSCRIBER` default `supadata`; defer the default flip until a golden-set A/B.

## 3. Wave 2.5 loose ends (follow-ups, not blockers)
- **Kinematics + trio live gate** — deferred this session (stopped to free resources for the user's
  live test). Run `run_eval dHjWVlfNraM uqwC41RDPyg NjvwWiCYLl4 yfajBIaDf1Q --rebuild --runs 3`
  (chunked, `python -u`) to complete the Wave 2.5 gate picture.
- **`forward_requires_edges = 1` on STALE caches** — it reads 0 on fresh rebuilds (W25-B works),
  1 on the cached qP structure. Confirm it's purely a stale-cache artifact (the freshness policy
  rebuilds it) vs a live LLM edge-direction violation.
- **Arc→ship attrition** — arcs are detected (2-4) but few ship as arc clips (`n_arc_clips_shipped`
  ~0-1). Investigate the dedupe/merge provenance path.

## 4. Human calibration (user's time — unlocks a lot)
The eval judge over-flags (confirmed by the human verdict). **~30 clip labels** at
`localhost:8000/static/labeling/` (page plays clips cold) → run `backend/eval/judge_calibration.py`
for judge κ + per-kind precision → set kill-authority trust weights + re-fit the 0.70 gate. Extend
the labeling pass to cover **boundary + genre** to build the small **golden set** that gates BND2 /
VID trust / the parakeet default-flip / CARE omission bound. Expert ceiling on this criterion is
κ≈0.51 — κ~0.5 is success.

## Process reminders (see the session summary for the full list)
Reviewed implementer subagents + adversarial verification + mutation checks; snapshot to `.backup/`
before big changes (clips/ is not git); restart uvicorn + rebuild frontend after edits; run evals
unbuffered + chunked (background commands can be killed ~10-15 min); write multi-arg shell lists
literally (zsh no word-split); clear `__pycache__` if a mutation check's restore looks wrong.
