# Task T3 Report — Wire IngestionPipeline into ReelService + `_concept_topic_query`

## What was built

### 1. `ReelService.__init__` — `backend/app/services/reels.py:1278`
Added `ingestion_pipeline=None` keyword parameter and `self.ingestion_pipeline = ingestion_pipeline` store.  Existing callers with positional/keyword `embedding_service` + `youtube_service` continue to work unchanged.

### 2. `main.py` reorder — `backend/app/main.py`
Moved `SERVERLESS_MODE` and `ingestion_pipeline = IngestionPipeline(...)` to be constructed **before** `reel_service`, then passed `ingestion_pipeline=ingestion_pipeline` into `ReelService(...)`.  Module-level name `ingestion_pipeline` is preserved (tests reference `main_module.ingestion_pipeline`).  Nothing between the old positions depended on the old order.

### 3. `_concept_topic_query` — `backend/app/services/reels.py` (after `_build_literal_query`)
New method on `ReelService`.  Mirrors the single-token-keyword logic of `_build_literal_query` without disambiguator/subject_tag:
- `_clean_query_text(title)` → empty → return `""`
- `len(normalize_terms([clean_title])) <= 1` → single-token → append first keyword from `_parse_keywords_json(concept_row.get("keywords_json"))` whose `_normalize_query_key` differs from the title's
- Return `_clean_query_text(" ".join(parts))`
No existing `generate_reels` behavior was changed.

### 4. New test file — `backend/tests/test_reels_concept_topic.py`
6 pure unit tests, no DB/network.  Uses `VERCEL=1` env var so `ReelService.__init__` sets `serverless_mode=True`.

## TDD RED → GREEN evidence

**RED** (first run — tests used "Osmosis" as the single-token example):
```
FAILED test_identical_keyword_skipped_uses_next
FAILED test_single_token_title_appends_first_differing_keyword
2 failed, 4 passed
```
Root cause: `normalize_terms(["Osmosis"])` returns `{'osmosi', 'osmosis'}` (len=2), so "Osmosis" does NOT qualify as single-token under this rule.  Used "ATP" (→ `{'atp'}`, len=1) instead.  Tests updated.

**GREEN** (second run):
```
6 passed in 0.44s
```

**Sanity run** (clip_engine + concept_topic):
```
12 passed in 7.19s
```

## Files changed
- `backend/app/services/reels.py` — `__init__` signature + store; new `_concept_topic_query` method (24 lines)
- `backend/app/main.py` — reorder: `SERVERLESS_MODE` → `ingestion_pipeline` → `reel_service(ingestion_pipeline=...)`
- `backend/tests/test_reels_concept_topic.py` — new file (6 tests)

## Self-review
- Purely additive: `generate_reels` not touched.
- `_concept_topic_query` reuses all four helpers the brief required (`_clean_query_text`, `_normalize_query_key`, `normalize_terms`, `_parse_keywords_json`) with no reimplementation.
- Tests correctly assert the actual rule behavior, not guessed output.

## Concerns
- **`normalize_terms` single-token semantics**: The rule fires for acronyms/short tokens (ATP, DNA, mRNA) but not for most natural-language single words (Osmosis, Diffusion) because the stemmer returns 2+ variants.  This is identical behavior to `_build_literal_query` — consistent, just non-obvious.  Worth documenting when T4 wires this into `generate_reels`.
- Pre-existing import errors in `run_mass_youtube_test.py` and `test_boundary_stress.py` (`from app.services...` instead of `from backend.app.services...`) are unrelated to T3.
