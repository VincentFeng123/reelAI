# Next-session prompt — clipper LATENCY pass (updated 2026-07-03, supersedes the old prompt)

Paste the block below into a fresh Claude Code session. It assumes the repo at
`/Users/vincentfeng/Documents/practice/clips`. Full session record:
`docs/SESSION-2026-07-03-summary.md`. Next-steps plan: `docs/NEXT-STEPS-2026-07-03.md`.
Wave 3/4 plan: `docs/superpowers/plans/2026-07-03-wave-3-4.md`. Ledger: `.superpowers/sdd/progress.md`.

---

```
You are continuing work on a structure-first video clipper in clips/backend/ (Python/FastAPI + a
React SPA prebuilt into backend/static). It turns a YouTube video + topic into self-contained
YouTube-embed clips via: transcribe (supadata) → punctuation-restore → understand (TreeSeg content
map → units/roles → dependency graph, cached per video as a Structure) → assemble (extraction plan
→ arcs → coverage quotas → context closure → clip-only quote-verified asymmetric judge/repair →
boundary snap → context cards → chronological order). LLM = Gemini only (author gemini-2.5-flash,
judge gemini-2.5-flash-lite; keys in clips/.env; Groq 401 + no video). Suite = 656 tests, all
offline. clips/ is NOT a git repo. Use clips/.venv/bin/python.

READ FIRST (in order): docs/SESSION-2026-07-03-summary.md (what was just done — Wave 2.5 review+gate,
Wave 3, Wave 4), docs/NEXT-STEPS-2026-07-03.md (the plan — latency is priority #1), and skim
.superpowers/sdd/progress.md (ledger tail) + docs/superpowers/plans/2026-07-03-wave-3-4.md.

STATE: Waves 1, 2, 2.5, 3, 4 (headline features) are SHIPPED + reviewed. Quality is ACCEPTABLE per
the user's own live test (the eval judge over-flags — do NOT chase judge comprehension scores). The
app is live at localhost:8000 (restart uvicorn to pick up code changes; rebuild the frontend with
`cd frontend && npm run build` after any .tsx edit). HARD INVARIANT: unverified_kill = 0 exactly.

FIRST TASK — the LATENCY pass (~20 min/video is the product blocker; the user's explicit priority):
1. cd clips && .venv/bin/python -m pytest backend -q  (expect 656 green) — baseline.
2. PROFILE, don't guess: instrument per-stage wall-clock (the orchestrator emits SSE stages
   transcribing/punctuating/perceiving/segmenting/assembling/refining/cutting) so ONE real run of a
   fresh ~20-min video prints where the time goes. Report the split before optimizing.
3. Apply the ranked levers (docs/NEXT-STEPS-2026-07-03.md §1b): multimodal PERCEPTION is the prime
   suspect + optional (gate off via analysis_profile=fast / MULTIMODAL=0 — ASK THE USER whether the
   visual analysis is worth keeping; it's a quality/speed call that's theirs); UNDERSTANDING runs
   sequentially (parallelize content_map→units→deps, zero quality change); JUDGE_WORKERS/PUNCT_WORKERS
   =4 are low (bump, but CHECK the user's Gemini RPM/tier first — 429 risk); then structural wins
   (batched first-pass judging, persistent disk verdict cache, reuse cached structure on repeats).
   Measure wall-clock A/B on one video after each change. Keep the _run_full→_run_fast degrade path.

ASK THE USER EARLY: (a) is the multimodal VISUAL analysis worth keeping, or gate it off by default
for speed? (b) what's their Gemini tier / acceptable concurrency (for the JUDGE_WORKERS bump)?
(c) a target — "fast enough" = under 5 min? under 2? — it sets how aggressive to get.

THEN (Wave 3/4 follow-up tail, all default-OFF or golden-set-dependent — specs in the plan doc):
GEN5 reconciliation (the ROBUST parody fix — but verify with a live parody run whether GEN2 already
ships good clips first; gate the per-candidate call to bound latency), GEN3/GEN4 genre depth, VID3
render-audit (config knobs already reserved, default OFF), BND2 whisper re-timing, ASR1 parakeet
opt-in. Plus Wave-2.5 loose ends: kinematics+trio live gate (deferred), forward_requires_edges on
stale caches, arc→ship attrition. And nudge the user for ~30 clip labels (localhost:8000/static/
labeling/) → judge_calibration.py → trust weights + a golden set.

ENVIRONMENT GOTCHAS (verified this session): reviewed implementer subagents + adversarial verify +
mutation checks is the process; snapshot to .backup/ before big changes (clips/ not git; verify =
full pytest + compileall); restart uvicorn + rebuild frontend after edits; run evals UNBUFFERED
(python -u) and CHUNKED — background long commands can be killed ~10-15 min and redirected stdout is
block-buffered (lost on kill); reuse fresh cached structures via --freeze to skip rebuilds; zsh does
NOT word-split unquoted $VARS (write multi-arg run_eval lists literally); clear __pycache__ if a
mutation-check's post-restore run looks wrong; the video judge is Gemini-only (bypasses llm_json).
```
