# Educational Retrieval Bias — Implementation Report (task EDU)

## Status
COMPLETE — all tests green.

## Files changed
- `backend/app/clip_engine/config.py` — added `EXPAND_MODEL` (defaults `gemini-2.5-flash-lite`)
- `backend/app/clip_engine/expand.py` — rewrote `_SYSTEM`; extended `free_expand` templates; pinned Gemini call to `config.EXPAND_MODEL`
- `backend/app/clip_engine/rank.py` — added `_BOOST_PATTERNS`, `_PENALTY_PATTERNS`, `_edu_score()`; wired bounded `edu_score` into `score`; stored on video dict
- `backend/tests/clip_engine/test_expand.py` — 2 new tests (fallback templates; prompt-steering guard)
- `backend/tests/clip_engine/test_rank.py` — 1 new test (rank ordering, cross-match-count invariant, bounds, `edu_score` key)

## Test summary
- Baseline: 91 passed
- After implementation: 94 passed (+3)
- Command: `backend/.venv/bin/python -m pytest backend/tests/test_clip_engine_*.py backend/tests/clip_engine -q -p no:randomly`

## Expansion model finding
`GEMINI_MODEL` (default `gemini-2.5-flash`) was shared by both expansion and segmentation (`SEGMENT_MODEL` defaults to `GEMINI_MODEL`). Expansion was pinned to `EXPAND_MODEL` (new config var, default `gemini-2.5-flash-lite`) in `config.py`. The segmentation model (`SEGMENT_MODEL`) is unchanged. Override with `EXPAND_MODEL=<model>` env var if needed.

## Lever 1 — expand.py
- `_SYSTEM` rewritten to steer toward educational content (lectures, explainers, courses, tutorials, documentaries) and explicitly avoid entertainment phrasings (reactions, memes, compilations, etc.)
- `free_expand` now includes 5 educational templates: `{t} explained`, `{t} lecture`, `{t} tutorial`, `how {t} works`, `{t} course` (plus existing `{t} for beginners`). Dedup and corrected-first order preserved. JSON contract unchanged.

## Lever 2 — rank.py
- 15 boost patterns (word-boundary, case-insensitive): explained, explanation, lecture, course, tutorial, lesson, how…works, introduction/intro to, basics, fundamentals, professor, university, documentary, crash course, khan academy.
- 8 penalty patterns: reaction, prank, funny, meme/memes, top 10/top ten, compilation, challenge, vlog.
- `_edu_score = clamp(boost*1.0 - penalty*1.5, -3.0, +3.0)` — stored as `edu_score` on each video dict.
- Added to `score` only; sort key `(match_count, score, view_count)` structure unchanged — cross-match-count invariant preserved.

## Notes
- `:8001` controller needs a restart to serve the new expansion prompt and ranking logic.
- No frontend, VidScout/practice, or toggle changes made.
