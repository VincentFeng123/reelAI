# Educational Retrieval Bias — Design (user approved option B, 2026-07-08)

## Goal
Video retrieval should favor educational content (lectures, explainers, courses, tutorials,
documentaries) over entertainment. Always-on, no toggle (StudyReels is a study product; knobs were
deliberately removed in the raw-output pivot). Applies to the StudyReels clip_engine only —
VidScout/practice stays untouched as the reference. Zero NEW API calls: both levers ride existing
machinery.

## Lever 1 — steer the existing Gemini query expansion (`backend/app/clip_engine/expand.py`)
The engine already calls Gemini per search to expand a topic into queries (with a keyless
deterministic fallback). Changes:
- Rewrite `_SYSTEM` so expansion targets EDUCATIONAL YouTube results: spellcheck/correct the topic,
  infer the academic field/discipline, then produce queries that surface teaching content —
  phrasings like "X explained", "X lecture", "how X works", "X course", "X for students",
  field-qualified variants (e.g. ambiguous "jaguar" → "jaguar animal biology"). Explicitly instruct
  AVOIDING entertainment phrasings (reactions, memes, funny compilations). The JSON contract
  (`{"corrected", "queries"}`) and `_normalize` behavior are UNCHANGED.
- Extend the keyless fallback's template set with educational variants ("{t} explained",
  "{t} lecture", "{t} tutorial", "how {t} works", "{t} course") while keeping dedup/order rules.
- Model: if the expansion call's model is configurable and currently shares the segmentation
  model, pin expansion to the cheap tier (flash-lite class) via existing config conventions —
  do NOT change the segmentation model. If it already uses a cheap/lite model, leave it.

## Lever 2 — light educational ranking signal (`backend/app/clip_engine/rank.py`)
`merge_and_rank` currently scores `match_count*10 + log10(views) + rank_bonus` and sorts by
`(match_count, score, view_count)`. Add a bounded educational term into `score` ONLY (so it breaks
ties within the same match_count — deliberately light, never overriding query-match relevance):
- Boost markers (title + channel, case-insensitive, word-boundary): explained, explanation,
  lecture, course, tutorial, lesson, how ... works (regex), introduction/intro to, basics,
  fundamentals, professor, university, documentary, crash course, khan academy.
- Penalty markers: reaction, prank, funny, meme, top 10/top ten, compilation, challenge, vlog.
- Bounded contribution: cap total educational term to ±3.0 (e.g. +1.0 per distinct boost hit,
  −1.5 per distinct penalty hit, clamped). Store it on the video dict (e.g. `edu_score`) for
  debuggability.
- Keep the sort key structure unchanged.

## Error handling
Expansion already falls back keylessly on any Gemini failure — unchanged. Ranking term is pure
string matching — no failure modes beyond regex correctness.

## Tests (extend `backend/tests/clip_engine/` suites)
1. Fallback expansion includes educational templates; dedup/order/contract intact.
2. `_SYSTEM` prompt asserts educational steering exists (string containment — cheap guard).
3. Ranking: same match_count, "X explained — MIT lecture" outranks "X funny moments compilation";
   penalty capped; `edu_score` bounded ±3; sort key unchanged for differing match_counts
   (a penalized 2-match video still beats a boosted 1-match video).
4. Existing clip_engine suite stays green (91 baseline, isolated `-p no:randomly`).

## Out of scope
Gemini judge re-rank (option C); any VidScout/practice change; any frontend change; toggles.
