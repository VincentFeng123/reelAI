# Knowledge-Level Rating & Difficulty-Aware Feed — Design

**Date:** 2026-07-08
**Status:** Approved by user (brainstorming sections 1–3)
**Scope:** reelAI backend (`backend/`), webapp (`src/`), iOS app (`reelai/reelai/`), practice clipper back-port (`practice/clips/backend/`)

## Problem

Reels are served without regard to how well the user already knows the topic. A
beginner searching "linear algebra" gets the same videos and clip ordering as a
graduate student. The user wants (a) a rating of how well they know the topic
they're searching, and (b) the pipeline to search for and organize videos that
correspond to that knowledge level — with off-level clips kept at the back of
the feed and re-entering when the user's level rises.

## Decisions (user-approved)

1. **Rating source:** self-select at topic creation (Beginner / Intermediate /
   Advanced, default Beginner) + bounded auto-adjust from existing feedback.
2. **Where it acts:** BOTH search steering and output organization.
3. **Feed strictness:** soft boost + gentle progression. Nothing is ever hidden;
   off-level clips sink to the back and re-enter when the effective level
   reaches them (serve-time re-scoring, not persist-time).
4. **Surfaces:** webapp + iOS picker in this pass.
5. **Approach:** level on the material; per-clip difficulty scored inside the
   existing gemini_segment call; deterministic feedback drift. (Per-user
   cross-topic profiles and serve-time LLM re-ranking were considered and
   rejected for this pass.)

## Data model

`materials` migration (follow the existing `db.py` migration helpers):

| column | type | semantics |
| --- | --- | --- |
| `knowledge_level` | `TEXT NOT NULL DEFAULT 'beginner'` | `beginner \| intermediate \| advanced` |
| `level_adjustment` | `REAL NOT NULL DEFAULT 0.0` | auto-adjust drift, clamped to **[-0.35, +0.35]** |

`reels` migration:

| column | type | semantics |
| --- | --- | --- |
| `difficulty` | `REAL` (nullable) | engine-scored 0..1; `NULL` (legacy) treated as 0.5 at serve time |

Level → difficulty-scale mapping: beginner **0.15**, intermediate **0.50**,
advanced **0.85**. `effective_target = clamp01(level_value + level_adjustment)`.

## Auto-adjust rule (deterministic, no LLM)

On each feedback write (existing helpful/confusing/rating endpoint), for the
reel's material:

- Take the material's most recent **20** `reel_feedback` rows.
- If fewer than **5** rows: `level_adjustment = 0` (cold-start gate).
- Else `signal = 0.25*helpful_rate − 0.35*confusing_rate + 0.15*(avg_rating − 3)/2`
  (same shape as the existing `_concept_mastery`; `helpful_rate`/`confusing_rate`
  are fractions of the window's rows with that vote, `avg_rating ∈ [1,5]`, so
  `signal ∈ [−0.5, +0.4]`).
- `level_adjustment = clamp(−0.35, +0.35, signal)` — the clamp IS the bounding;
  direction is the contract: sustained *confusing* ⇒ drift down, sustained
  *helpful + high ratings* ⇒ drift up, never more than one level step.

## Pipeline behavior

### Search steering
- `clip_engine/expand.py expand_query(topic, n, level=None)`: one extra prompt
  line per level (beginner ⇒ "introduction to X", "X basics", "X for
  beginners"; advanced ⇒ "advanced X", "graduate X lecture", "X deep dive";
  intermediate ⇒ no extra line). `search.discover()` and
  `ingest_topic()` thread the level through; `ReelService.generate_reels`
  passes the material's level.
- `clip_engine/rank.py merge_and_rank(per_query, level=None)`: title-pattern
  difficulty bands exactly like the existing `_edu_score` convention —
  beginner band (`intro`, `introduction`, `basics`, `beginner`, `101`,
  `crash course`, `for dummies`), advanced band (`advanced`, `graduate`,
  `seminar`, `research`, `proof`, `lecture \d{2,3}`); +1.0 per hit matching the
  user's band, −1.0 per hit in the opposite band, clamped ±2.0, **added to
  `score` only** (the `(match_count, score, view_count)` sort-key structure is
  unchanged).

### Clip difficulty (no new LLM cost)
- `gemini_segment._Topic` gains `difficulty: float = 0.5` (0 = assumes no
  prior knowledge, 1 = expert). Prompt documents the scale with anchors.
  Normalized with the same mis-scale tolerance as informativeness
  (`_norm_informativeness`-style). Carried on clip dicts; **never gated** —
  difficulty is a signal, not a filter.
- Persisted into `reels.difficulty` via `_persist_engine_clip` →
  `upsert_reel_row`.
- **Back-port:** the updated `gemini_segment.py` is copied byte-identical to
  `practice/clips/backend/pipeline/gemini_segment.py` (the two clipper
  codebases stay the same code), plus the curation test file.

### Feed organization (serve-time, the "keep them in the back" contract)
- In `ranked_feed`, per clip:
  `score += 0.12 * (1 − 2*abs(difficulty − effective_target))`
  with `difficulty = 0.5` when NULL. Matched clips get up to +0.12; fully
  mismatched get down to −0.12. Off-level clips are **never excluded** by this
  mechanism.
- Because the bonus is computed per-request from the material's *current*
  effective target, level drift (or an explicit level change) automatically
  re-sorts the already-persisted library — back-of-feed advanced clips
  re-enter up front when the user reaches that level. The level bonus is
  NEVER baked into persisted `base_score`.
- Gentle progression: `score += 0.05 * (1 − difficulty) * max(0, 1 − (page_hint−1)/2)`
  — page 1 leans easier, gone by page 3.
- The ranked-feed **cache fingerprint incorporates the effective target** so a
  level change/drift invalidates cached orderings; `RANKED_FEED_CACHE_VERSION`
  bumps 6 → 7.

## API contract

- Generate/create-topic request: optional `knowledge_level` string field
  (absent ⇒ `beginner`; invalid values ⇒ 422).
- New endpoint: `PATCH /api/materials/{material_id}/level` with body
  `{"knowledge_level": "advanced"}` — updates the material AND resets
  `level_adjustment` to 0 (an explicit choice supersedes accumulated drift);
  returns the new effective target.
- Feed response: adds `knowledge_level` (the stored choice) and
  `effective_level_target` (float) at the response level so both clients can
  render "Beginner · auto-adjusting".

## UI

**Webapp:** three-chip selector (Beginner default) in the topic-creation flow,
included in the generate request; level pill on the feed page showing
"<Level> · auto-adjusting", tappable → PATCH → refetch feed.

**iOS:** segmented picker (3 levels) in the topic-creation screen, wired into
the request model; read-only level label on the feed. Post-creation level
changes on iOS ship in a later pass (the backend endpoint already exists).

## Error handling

- Unknown/invalid `knowledge_level` in requests → 422; absent → beginner.
- Model omits `difficulty` → pydantic default 0.5; mis-scaled values (0–10 /
  0–100) normalized, then clamped to [0, 1].
- Feedback endpoint failure to recompute adjustment is non-fatal (logged;
  adjustment stays at last value).
- Legacy reels (NULL difficulty) score neutrally — no feed churn on rollout.

## Testing

- `expand`: prompt contains the correct level line per level; no line for
  intermediate; level=None unchanged.
- `rank`: band scoring direction, clamp, sort-key structure unchanged.
- `gemini_segment`: difficulty parsed, normalized, carried; absent → 0.5;
  never gates a clip.
- DB: migrations idempotent; difficulty round-trips through persist → feed row.
- Adjustment rule: direction, ±0.35 bound, <5-feedback gate, 20-row window.
- `ranked_feed`: matched > mismatched at same base; NULL neutral; advanced clip
  rises when target rises (the re-enter contract); progression decays by page;
  fingerprint changes when effective target changes.
- API: generate accepts/validates `knowledge_level`; PATCH endpoint contract.
- Practice suite re-run after back-port; full isolated-suite battery as in the
  quality-overhaul session (`-p no:randomly`).

## Out of scope (explicitly)

- Per-user cross-topic knowledge profiles (possible phase 2 aggregating this
  data).
- Calibration quizzes.
- iOS post-creation level editing UI.
- Backfilling difficulty for legacy reels.
