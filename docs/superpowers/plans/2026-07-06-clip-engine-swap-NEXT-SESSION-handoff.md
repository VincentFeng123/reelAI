# Clip-Engine Swap — NEXT-SESSION Handoff (Phase 3 + Phase 4)

**Written 2026-07-06.** Phases 1–2 are DONE and merged. This document is what the next session executes. Read it plus `.superpowers/sdd/progress.md` (the SDD ledger) to resume — you should not need to re-derive anything.

---

## 0. Where things stand

- **Branch:** `codex/high-quality-video-retrieval` @ `640f333` (Phase 1+2 fast-forward-merged here; feature branch `clip-engine-swap` deleted; **not pushed** to origin).
- **What's done:** `backend/app/clip_engine/` (Supadata search + expansion + rank; vendored gemini clipper; `run.py` runner; `bridge.py` translator + relevance filter; `metadata.py`). All four `/api/ingest/*` methods (`ingest_url/topic_cut/search/feed`) rewired to the engine, reusing `pipeline._persist_ingest()` so the `ReelOut` contract is byte-identical. iOS + web clients unchanged. **64 clip-engine tests green**; 12 old-path tests `@unittest.skip`'d.
- **Uncommitted WIP already in the tree (NOT ours):** `backend/app/services/reels.py` (+228), `segmenter.py` (+7), `clip_whisper_refine.py`, `llm_router.py` (−3, deletes the API-blackout comment → re-enables Groq Whisper), `backend/tests/test_reels_saliency.py` (+176), plus untracked `audit_*.csv` (user's reference artefacts — leave them). This is a pre-existing "saliency scoring on the legacy path" change. **See Phase 3 Task T0.**

## 1. Resume checklist (conventions)

- **Repo root:** `/Users/vincentfeng/Documents/reelai app/reelai/reelAI copy 2`. Run everything from here.
- **Tests:** `backend/.venv/bin/python -m pytest backend/tests -q` (only this venv has pytest + deps). Single file: `… -m pytest backend/tests/test_clip_engine_ingest_url.py -q`.
- **Imports:** `from backend.app.X import …` (never `from app…` — double-import breaks pydantic). Inside `clip_engine`, relative imports.
- **DB-integrated test harness template:** `backend/tests/test_clip_engine_ingest_url.py` — `TemporaryDirectory` + `os.environ["DATA_DIR"]`, `db_module._db_ready=False`, `get_settings.cache_clear()`, `main_module.settings=get_settings()`, `os.environ["REELAI_INGEST_SKIP_IMPORT_SWEEP"]="1"`, bump `main_module.ingestion_pipeline._rate_limiter = _PlatformRateLimiter(overrides={"yt": (1000, 60.0)})`, and mock `clip_engine_run.clip` + `clip_engine_meta.youtube_metadata` + `clip_engine_search.discover`. Real keys not needed (engine is mocked).
- **clip_engine config = env vars** (from gitignored `backend/.env`): `SUPADATA_API_KEY`, `GEMINI_API_KEY` (+`GOOGLE_API_KEY` fallback), `GROQ_API_KEY`, `GEMINI_MODEL`; knobs `CLIP_ENGINE`, `TRANSCRIBER`, `CLIP_SEARCH_MAX_VIDEOS`, `CLIP_SEARCH_BREADTH`. Phase 3 adds a new flag `REELS_CLIP_ENGINE` (default off).
- **Process:** continue subagent-driven-development (implement→review per task). Ledger: `.superpowers/sdd/progress.md`. Plans: `docs/superpowers/plans/`. Graph: `graphify-out/` (rebuild code graph after edits per project CLAUDE.md).

## 2. CONFIRM WITH THE USER before implementing Phase 3

These gate the work — resolve them first:

1. **Paid APIs / hosting (partly resolved):** The user already accepted **Supadata + Gemini** paid keys for the clip engine this session (the old "API blackout" is lifted *for the clip engine* — recorded in memory). STILL OPEN: (a) the uncommitted WIP re-enables **Groq Whisper** in `llm_router.py` — confirm that's wanted, and (b) the "~$0 hosting" preference conflicts with an always-on transcribe+Gemini backend — confirm the cost/latency budget for the material path (which fans out over many concepts × videos).
2. **Cutover strategy:** Phase 3 gates the new path behind `REELS_CLIP_ENGINE` (legacy `topic_cut` preserved for A/B + rollback) — this blocks Phase 4 deletion of `topic_cut.py` until the flag becomes the sole path. Confirm flag-gated (recommended, assumed here) vs hard-replace.
3. **Disposition of the reels.py saliency WIP** (T0): commit it as its own legacy-path commit, or is it meant to be superseded by the clip engine? It edits the exact functions Phase 3 bypasses.
4. **`num_reels` semantics:** `run.clip` returns MULTIPLE clips/video but ingest keeps ONE/video. For `generate_reels(num_reels=N)`, should N reels = N distinct videos (one clip each) or allow multiple clips per video? Drives the discover limit + selection loop.
5. **Concept relevance:** legacy filters segments by concept-embedding cosine; the engine takes only a topic string. Is topic-string relevance enough, or post-filter clips by concept embedding to keep feed `relevance_score` quality?

## 3. Phase 3 — rewire the study-material path (flag-gated)

**Seam:** `ReelService.generate_reels(material_id, concept_id, …, generation_id, on_reel_created)` at `backend/app/services/reels.py:1487` — the one method both `/api/reels/generate[-stream]` and `/api/feed` refinement converge on. Strategy: add a NEW branch at the top of `generate_reels` gated by `REELS_CLIP_ENGINE`, routing each extracted concept as a topic through the SAME building block `/api/ingest/search` uses (`clip_engine.search.discover` → `clip_engine.run.clip` → `pipeline._persist_ingest`). Preserve: reels-table row shape + `ReelOut` contract (via `_persist_ingest`), `ranked_feed` scoring, `generation_id`/`reel_generation_heads` tracking, refinement jobs, `reel_feedback`/`record_feedback`, the `on_reel_created` streaming callback, `dry_run` for can-generate, `FeedResponse` pagination. **Load-bearing gap:** `_persist_ingest → upsert_reel_row` hardcodes `generation_id=None`, but feed/refinement key on `generation_id` — thread it through (T1).

**Tasks (ordered, TDD; mirror the harness in §1):**

- **T0 (pre-req):** Land or park the uncommitted saliency WIP as its OWN commit *before* touching `reels.py`, so the Phase-3 diff stays isolated. Run `pytest backend/tests/test_reels_saliency.py -q` to confirm green. **Reconcile the `llm_router.py` Groq-Whisper re-enable with the memory blackout note (open q #1) before committing. Do NOT discard the WIP without user confirmation.**
- **T1:** Thread `generation_id` through the ingest persistence layer. Add `generation_id: str|None=None` to `upsert_reel_row` (`persistence.py:192`; replace hardcoded `row["generation_id"]=None` at ~212) and to `load_existing_reel` (`:152`; change WHERE `generation_id IS NULL OR = ""` at ~176 to `COALESCE(generation_id,'')=COALESCE(?,'')`). Add the param to `_persist_ingest` (`pipeline.py:974`) and pass through. **Test** `test_clip_engine_generation_id.py`: (a) reel persisted with `generation_id="gen-x"` has it in the row + `load_existing_reel` scoped to gen-x finds it; (b) `None` still round-trips (backward-compat; existing idempotency test still passes). Unique index is `(material_id, COALESCE(generation_id,''), video_id, t_start, t_end)` at `db.py:730` — same clip across two generations = two rows, by design.
- **T2:** Factor a reusable `ingest_topic(*, topic, material_id, concept_id, generation_id, exclude_video_ids, target_clip_duration_*, language, max_videos, on_reel_created=None, dry_run=False)` out of `ingest_search` (extract the discover→clip→bridge→`_persist_ingest` loop, `pipeline.py:807-865`), stamping `generation_id` and firing `on_reel_created` per persisted reel; returns `(reels, resolved_video_ids)`. Have `ingest_search` call it (no dup). **Test** `test_clip_engine_material_topic.py`.
- **T3:** Wire `IngestionPipeline` into `ReelService` (optional `ingestion_pipeline` param, default None→legacy; pass it in `main.py:~272-293`) + add `_concept_topic_query(concept_row)` (mirror `_build_literal_query` `reels.py:5348`; title + one keyword when title is single-token). **Test** `test_reels_concept_topic.py` (pure, no DB/network).
- **T4:** Add the flag-gated branch at the top of `generate_reels`: if `os.environ.get("REELS_CLIP_ENGINE")` and `self.ingestion_pipeline`: load the material's concepts (same query as `reels.py:1550`, scoped by `concept_id`); for each, call `ingest_topic(topic=_concept_topic_query(c), …, generation_id=generation_id, exclude_video_ids=accumulated, on_reel_created=on_reel_created)`, accumulating exclusions, stopping at `num_reels`; convert persisted `ReelOutWithAttribution`→ the legacy `ReelOut` dict shape; honor `dry_run` (T6). Leave the legacy `topic_cut` branch untouched below the guard. **Test** `test_clip_engine_generate_reels.py` (flag on → reels under active `generation_id`, `ranked_feed` returns them, `on_reel_created` fires per reel; flag off → legacy path, `discover` call count 0).
- **T5:** Verify feed + refinement + feedback end-to-end with the flag on (extend T4's test): `ranked_feed` ranks/paginates the clip-engine reels; refinement with a fresh `result_generation_id` persists + `_activate_generation` swaps `active_generation_id`; `record_feedback` shifts the ranked score. Bump `RANKED_FEED_CACHE_VERSION` (`db.py`) if stale cache could serve pre-flag rows. Full suite stays ≥64 green with flag on AND off.
- **T6:** `dry_run`/can-generate parity: in `ingest_topic`, `dry_run=True` → `discover` only (no `run.clip`/persist), return a viability count; feed it into `generate_reels`'s dry_run so `ReelsCanGenerateResponse` fields stay populated. **Test:** `dry_run=True` writes zero reel rows yet returns a non-empty probe.

## 4. Phase 4 — cleanup (only after Phase 3 verified where noted)

**SAFE NOW (dead on all live paths, independent of Phase 3):**
- Delete entire files: `backend/app/ingestion/adapters/yt_dlp_adapter.py`, `backend/app/ingestion/transcribe.py`.
- In `pipeline.py`: delete `_persist_topic_reels` (~410), `_ingest_url_with_topic_cut` (~646), `_preflight` (~194), the `_feed_executor` attr (~175). Re-verify `_pick_search_adapter` (~883) unused after T2, then delete.
- Remove now-dead `pipeline.py` imports that only served the deleted methods: `normalize_clip_window`, `pick_segments`, `silencedetect` (never referenced), plus `snippet_for_window`, `probe_duration`, `check_ffmpeg_available`, `map_info_dict_to_metadata`, `YtDlpAdapter`, `TempWorkspace`. Run `pytest backend/tests` after to confirm no import breakage.
- Delete the 12 `@unittest.skip("retired … clip-engine-swap")` tests: 7 in `test_ingestion_url.py` (~363,447,495,536,607,650,672), 5 in `test_ingestion_topic_cut.py` (~171,250,291,335,385). Coverage is in `test_clip_engine_*.py`.

**DO NOT DELETE until `REELS_CLIP_ENGINE` is the sole path (legacy `generate_reels` branch removed):** `services/topic_cut.py`, `clip_boundary.py`, `clip_whisper_refine.py`, `segment_features.py`, `importance_ranker.py`, `ingestion/segment.py` (`normalize_clip_window`/`snippet_for_window` still imported by `reels.py`).

**Dependencies — trim nothing now.** `faster-whisper`, `sentence-transformers`, `faiss-cpu`, `torch` all stay (clip_engine live and/or material path). `scikit-learn`, `spacy`, `en_core_web_lg`, `networkx` are used ONLY by the material-path `importance_ranker` — trim from `requirements.txt` + `pyproject.toml` ONLY after Phase 3 is verified AND the legacy branch is deleted (grep each for other importers first).

**Other:** No `backend/work` gitignore needed (old code used `tempfile.mkdtemp`). Leave the untracked `audit_*.csv` alone (user's reference artefacts).

## 5. Key rotation

`clip_engine` reads keys from `os.environ` (from gitignored `backend/.env`, `.gitignore:18`). Keys consumed: `SUPADATA_API_KEY` (search + default transcript), `GEMINI_API_KEY`/`GOOGLE_API_KEY` (segmentation/judge), `GROQ_API_KEY` (Whisper fallback). To rotate: edit the value in `backend/.env` (never commit) and **RESTART the backend** — `config.py` captures env into module constants at import (no hot reload). ⚠️ The ORIGINAL keys are still committed in `practice/clips/.env` (Supadata `sd_…`, Gemini `AQ…`) — rotate those at the provider and scrub if that repo is ever pushed. `GEMINI_API_KEY_2` in `backend/.env` is a manual spare (clip_engine only reads `GEMINI_API_KEY`/`GOOGLE_API_KEY`).

## 6. Recommended order next session

1. Confirm the §2 open questions with the user.
2. T0 (park/commit the WIP) → then Phase 3 T1–T6 (subagent-driven, per-task review).
3. Phase 4 "SAFE NOW" cleanup (can run in parallel with/after Phase 3 T1–T2 since it's independent).
4. Flip `REELS_CLIP_ENGINE` on by default (after real Supadata+Gemini smoke test) → then Phase 4 legacy deletion + dep trim.
5. Push / deploy when the user asks.
