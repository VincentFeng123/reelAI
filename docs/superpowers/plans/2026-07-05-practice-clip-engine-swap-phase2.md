# Practice Clip-Engine Swap — Phase 2 (REVISED) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]`.
> **This file SUPERSEDES Tasks 8–13 of `2026-07-05-practice-clip-engine-swap.md`.** Phase 1 (Tasks 1–7, the `clip_engine` library) is complete and merged. Task numbering continues at 8.

**Goal:** Wire the Phase-1 `clip_engine` (Supadata search + Gemini clipper) into the live `IngestionPipeline`, replacing the yt-dlp/Whisper/segment path, while preserving the `ReelOut` contract byte-for-byte so the iOS app and web frontend are unchanged.

**Key strategy (why this differs from the original sketch):** `IngestionPipeline` already has `_persist_ingest()` and `_persist_topic_reels()` which build AND persist `ReelOutWithAttribution` with the exact client contract — the canonical YouTube embed URL (`?start=&end=&modestbranding=1&rel=0&playsinline=1`), takeaways, AI summary, source attribution, and DB rows. So we **reuse `_persist_ingest`** by translating the practice engine's output into its inputs, rather than building a parallel adapter. This guarantees contract fidelity.

## Global Constraints

- **Reuse `_persist_ingest`, don't reinvent it.** For each surviving clip, call `self._persist_ingest(adapter_result=..., metadata=..., cues=..., chosen=IngestSegment(...), snippet=..., material_id=..., concept_id=..., clip_window=(start,end), target_max=...)`. It returns a `ReelOutWithAttribution` and writes the DB row. It reads only `adapter_result.platform`, `.source_id`, `.playback_url` — so a synthesized `AdapterResult(platform="yt", source_id=video_id, source_url=url, playback_url=<embed base>, video_path=Path("."), info_dict={})` is sufficient.
- **YouTube-only.** Non-YouTube URLs → `IngestUnsupportedSourceError`. `ingest_search`/`ingest_feed` only return YouTube. Drop the IG/TT adapter branches in the rewired methods.
- **Preserve query relevance.** For endpoints carrying a topic/query (`ingest_topic_cut.query`, per-concept material flow), filter/rank clips by token-overlap of the query against each clip's title + in-window transcript text. If the query is empty/None, keep all clips (topic-agnostic). Set the surviving clip's `IngestSegment.score` to its relevance score so feed ranking reflects it.
- **Contract preserved exactly.** Endpoints keep returning `IngestResult`/`IngestTopicCutResult`/`IngestSearchResult`/`IngestFeedResult`; response models, error mapping, `TERMS_NOTICE`, `trace_id`, serverless guard, and rate-limiter calls stay. Never rename/drop a `ReelOut` field.
- **Test & import convention:** run `backend/.venv/bin/python -m pytest <path> -v` from repo root; import `backend.app.X`; NEVER `from app...`; relative imports inside `clip_engine`. No dotenv in engine modules.
- **DB-integrated tests reuse the existing harness.** Phase 2 rewiring tests follow `backend/tests/test_ingestion_url.py`'s pattern: `unittest.TestCase`, `sys.path.insert(0, ROOT)`, `os.environ["REELAI_INGEST_SKIP_IMPORT_SWEEP"]="1"`, a `tempfile` DATA_DIR override + `db_module._db_ready = False` + `get_settings.cache_clear()` in `setUp`, then MOCK the `clip_engine` functions (`clip_engine.run.clip`, `clip_engine.search.discover`) instead of yt-dlp/ffmpeg/Whisper, and let `_persist_ingest` write to the temp DB. Read that file first and mirror its setUp/tearDown.
- **No heavy imports at pipeline import time.** The pipeline may `from ..clip_engine import run as clip_engine_run, search as clip_engine_search, bridge` at module top ONLY if those stay light (they do — heavy deps are lazy inside `run`). If import cost is a concern, import inside the methods.

---

## Task 8: Bridge translator + relevance filter (`clip_engine/bridge.py`)

**Files:**
- Create: `backend/app/clip_engine/bridge.py`
- Test: `backend/tests/clip_engine/test_bridge.py`

**Interfaces (Produces):**
- `to_cues(transcript: dict) -> list[IngestTranscriptCue]` — one cue per practice transcript segment (`{start,end,text}`), skipping empties.
- `to_metadata(video_id: str, meta: dict, source_url: str) -> IngestMetadata` — `platform="yt"`, `source_id=video_id`, `playback_url=f"https://www.youtube.com/embed/{video_id}"`, plus title/description/author_name/duration_sec/thumbnail_url/view_count from `meta` (a dict from Supadata search result or `metadata.youtube_metadata`).
- `synth_adapter_result(video_id: str, source_url: str) -> AdapterResult` — `AdapterResult(platform="yt", source_id=video_id, source_url=source_url, playback_url=f"https://www.youtube.com/embed/{video_id}", video_path=Path("."), info_dict={})`.
- `to_segment(clip: dict, transcript: dict) -> IngestSegment` — `t_start=clip["start"]`, `t_end=clip["end"]`, `text` = joined in-window transcript text (fallback to `clip.get("title","")`), `score=float(clip.get("score", 1.0))`.
- `window_text(transcript: dict, t0: float, t1: float) -> str` — joined text of segments overlapping [t0,t1].
- `relevance_score(clip: dict, transcript: dict, query: str) -> float` — token-overlap (Jaccard-ish) of `query` tokens vs the clip's title + in-window text; `1.0` when query is falsy.
- `filter_by_query(clips: list[dict], transcript: dict, query: str | None, *, floor: float = 0.0) -> list[dict]` — annotate each clip with `clip["score"] = relevance_score(...)`, drop clips at/below `floor` when query is truthy, sort by score desc; when query is falsy, return clips unchanged (all kept).

Imports: `from ..ingestion.models import IngestTranscriptCue, IngestMetadata, IngestSegment`; `from ..ingestion.adapters.base import AdapterResult`; `from pathlib import Path`. These are light (pydantic models + a dataclass) — safe at module top.

- [ ] **Step 1: Write failing tests** (pure, no DB) covering: `to_cues` maps segments and skips empties; `to_metadata` sets platform/source_id/playback_url + copies fields; `synth_adapter_result` yields the 3 fields `_persist_ingest` reads; `to_segment` computes in-window text + score; `relevance_score` returns 1.0 for empty query and a higher score for an on-topic clip than an off-topic one; `filter_by_query` keeps all when query is None and drops off-topic + sorts by score when query is set.

```python
# backend/tests/clip_engine/test_bridge.py  (illustrative — write full assertions)
from backend.app.clip_engine import bridge

TRANSCRIPT = {"segments": [
    {"start": 0.0, "end": 5.0, "text": "the chain rule in calculus"},
    {"start": 5.0, "end": 10.0, "text": "unrelated cooking tips"},
], "words": [], "duration": 600.0}


def test_to_cues_skips_empties():
    tx = {"segments": [{"start": 0, "end": 1, "text": "hi"}, {"start": 1, "end": 2, "text": ""}], "words": []}
    cues = bridge.to_cues(tx)
    assert [c.text for c in cues] == ["hi"]


def test_synth_adapter_result_fields():
    ar = bridge.synth_adapter_result("dQw4w9WgXcQ", "https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    assert ar.platform == "yt" and ar.source_id == "dQw4w9WgXcQ"
    assert ar.playback_url == "https://www.youtube.com/embed/dQw4w9WgXcQ"


def test_filter_by_query_keeps_all_when_no_query():
    clips = [{"start": 0.0, "end": 5.0, "title": "A"}, {"start": 5.0, "end": 10.0, "title": "B"}]
    assert bridge.filter_by_query(clips, TRANSCRIPT, None) == clips


def test_filter_by_query_ranks_on_topic_first():
    clips = [{"start": 5.0, "end": 10.0, "title": "cooking"}, {"start": 0.0, "end": 5.0, "title": "chain rule"}]
    out = bridge.filter_by_query(clips, TRANSCRIPT, "chain rule calculus", floor=0.0)
    assert out[0]["title"] == "chain rule"
    assert out[0]["score"] >= out[-1]["score"]
```

- [ ] **Step 2: Run to verify fail** — `backend/.venv/bin/python -m pytest backend/tests/clip_engine/test_bridge.py -v` → FAIL (module missing).
- [ ] **Step 3: Implement `bridge.py`** per the interfaces above. Keep `relevance_score` simple: lowercase-tokenize (`re.findall(r"[a-z0-9']+", ...)`), `overlap = |query∩clip| / max(1,|query|)`. No external deps.
- [ ] **Step 4: Run to verify pass** — same command → PASS; then full suite `backend/.venv/bin/python -m pytest backend/tests/clip_engine/ -v` (prior tests still green).
- [ ] **Step 5: Commit** — `git add backend/app/clip_engine/bridge.py backend/tests/clip_engine/test_bridge.py && git commit -m "feat(clip_engine): pipeline bridge translator + query relevance filter"`

---

## Task 9: Rewire `IngestionPipeline.ingest_url`

**Files:** Modify `backend/app/ingestion/pipeline.py` (`ingest_url` body); Test `backend/tests/test_clip_engine_ingest_url.py`

**Behavior:** Keep the signature + `set_trace_id`/`TERMS_NOTICE`/return shape. Replace the download→transcribe→segment body with:
1. `video_id = clip_engine_meta.extract_video_id(source_url)`; if None → `raise UnsupportedSourceError("Only YouTube URLs are supported.")` (maps to `IngestUnsupportedSourceError` at the endpoint — verify the endpoint's except list catches `UnsupportedSourceError`).
2. `engine_out = clip_engine_run.clip(source_url, topic=(concept_id or ""), settings={"language": language})` — wrap `UnsupportedURLError`→`UnsupportedSourceError`, `TranscriptError`→`TranscriptionError`, `ClipError`→`SegmentationError`.
3. `clips = engine_out["clips"]`; if empty → `raise SegmentationError("no on-topic clip")`. Pick the clip closest to `target_clip_duration_sec` (single-clip endpoint).
4. `meta = clip_engine_meta.youtube_metadata(video_id) or {}`; if no duration, use `engine_out["transcript"].get("duration")`.
5. Build inputs via `bridge`: `adapter_result = bridge.synth_adapter_result(video_id, source_url)`; `metadata = bridge.to_metadata(video_id, meta, source_url)`; `cues = bridge.to_cues(engine_out["transcript"])`; `chosen = bridge.to_segment(best_clip, engine_out["transcript"])`; `snippet = bridge.window_text(engine_out["transcript"], chosen.t_start, chosen.t_end)[:700]`.
6. `persisted = self._persist_ingest(adapter_result=adapter_result, metadata=metadata, cues=cues, chosen=chosen, snippet=snippet, material_id=material_id, concept_id=concept_id, clip_window=(chosen.t_start, chosen.t_end), target_max=int(target_clip_duration_max_sec))`.
7. `return IngestResult(reel=persisted, metadata=metadata, terms_notice=TERMS_NOTICE, trace_id=effective_trace)`.

Drop the `_preflight()`/ffmpeg/adapter/TempWorkspace machinery from this method (the engine path needs no local download/ffmpeg). Keep the rate-limiter acquire for `platform="yt"`.

- [ ] **Step 1: Write a failing DB-integrated test** mirroring `test_ingestion_url.py` setUp (temp DATA_DIR + `db_module._db_ready=False` + `get_settings.cache_clear()`), mocking `pipeline_module` engine references (`clip_engine_run.clip` returns a fake `{video_id, clips, transcript, notes}`; `youtube_metadata` returns `{"title":...,"duration_sec":...}`), then call `ingestion_pipeline.ingest_url(source_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ", material_id="m1", ...)` and assert `result.reel.video_url.startswith("https://www.youtube.com/embed/dQw4w9WgXcQ?start=")`, `result.metadata.platform=="yt"`, `result.reel.t_start`/`t_end` match the clip. Add a `test_ingest_url_rejects_non_youtube` asserting `IngestUnsupportedSourceError` for a vimeo URL. (Read `test_ingestion_url.py` for the exact DB setUp and the `ingestion_pipeline`/`app` handles.)
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Rewrite `ingest_url`** as above. Add the module-level engine imports (`from ..clip_engine import run as clip_engine_run, search as clip_engine_search, bridge, metadata as clip_engine_meta`).
- [ ] **Step 4: Run → PASS**; then `backend/.venv/bin/python -m pytest backend/tests/ -k "clip_engine or ingest" -v` to catch regressions in neighboring ingest tests.
- [ ] **Step 5: Commit** — `feat(ingest): route ingest_url through the clip engine (YouTube-only, reuse _persist_ingest)`

---

## Task 10: Rewire `IngestionPipeline.ingest_topic_cut`

**Files:** Modify `pipeline.py` (`ingest_topic_cut` body); Test `backend/tests/test_clip_engine_topic_cut.py`

**Behavior:** Keep signature/return (`IngestTopicCutResult`). Replace body with: extract video_id (YouTube-only), `engine_out = clip_engine_run.clip(source_url, topic=(query or ""), settings={"language": language})`, `kept = bridge.filter_by_query(engine_out["clips"], engine_out["transcript"], query)`. If no clips or duration < 60s and 0 clips → `is_short=True`, `reels=[]`. Else for EACH kept clip: build `chosen`/`cues`/`metadata`/`adapter_result` via bridge and call `self._persist_ingest(...)` (loop), collecting `ReelOutWithAttribution`. Return `IngestTopicCutResult(source_url, video_id, is_short, classification_reason=("short" if is_short else "long-form"), duration_sec, reel_count=len(reels), reels, metadata, terms_notice=TERMS_NOTICE, trace_id)`. `use_llm` is accepted but the gemini engine always uses the LLM; keep the param for signature compatibility (note it in a comment).

- [ ] **Step 1: Failing DB test** (mirror harness): mock `clip_engine_run.clip` returning 2 clips (one on-topic, one off-topic), call `ingest_topic_cut(query="chain rule", ...)`, assert only the on-topic reel(s) returned (relevance filter), `reel_count` matches, each reel's `video_url` is the embed with its own start/end.
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Rewrite `ingest_topic_cut`.**
- [ ] **Step 4: Run → PASS** + neighbor ingest tests.
- [ ] **Step 5: Commit** — `feat(ingest): route ingest_topic_cut through the clip engine + query relevance`

---

## Task 11: Rewire `IngestionPipeline.ingest_search`

**Files:** Modify `pipeline.py` (`ingest_search` body); Test `backend/tests/test_clip_engine_search.py`

**Behavior:** Keep signature/return (`IngestSearchResult`). Coerce `platforms` to `["yt"]`. `limit = min(max_per_platform, clip_engine_config.CLIP_SEARCH_MAX_VIDEOS)`. `disc = clip_engine_search.discover(query, limit=limit, exclude_video_ids=exclude_video_ids)`. For each `v` in `disc["videos"]` (bounded; a small thread pool is fine but sequential is acceptable): `engine_out = clip_engine_run.clip(v["url"], topic=query, settings={"language": language})`; if no clips → item status `"skipped"`; else pick best clip (target-duration-closest), `meta` from the search video dict (`title`/`channel`/`duration`/`thumbnail`/`view_count`) falling back to transcript duration, build bridge inputs, `self._persist_ingest(...)`, append `IngestSearchItem(platform="yt", source_url=v["url"], status="ok", reel=persisted, metadata=...)`. Per-video exceptions → `status="error"` (non-fatal). Build `IngestSearchResult` with `platforms=["yt"]`, per-platform counts, `terms_notice=TERMS_NOTICE + " Search is YouTube-only."`, `trace_id`.

- [ ] **Step 1: Failing DB test**: mock `clip_engine_search.discover` (1 YouTube video) + `clip_engine_run.clip` (1 clip), call `ingest_search(query="calc", platforms=["yt","ig","tt"], ...)`, assert `result.platforms == ["yt"]`, `succeeded == 1`, item reel `video_url` is the embed.
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Rewrite `ingest_search`.**
- [ ] **Step 4: Run → PASS** + neighbor tests.
- [ ] **Step 5: Commit** — `feat(ingest): route ingest_search through Supadata + clip engine (YouTube-only)`

---

## Task 12: Rewire `IngestionPipeline.ingest_feed`

**Files:** Modify `pipeline.py` (`ingest_feed` body) + add `resolve_feed_urls` to `clip_engine/metadata.py`; Test `backend/tests/test_clip_engine_feed.py`

**Behavior:** Add `clip_engine.metadata.resolve_feed_urls(feed_url, max_items) -> list[str]` (lazy yt-dlp `extract_flat`, YouTube-only, returns watch URLs; `[]` on failure). Rewire `ingest_feed` to resolve YouTube URLs then loop each through `clip_engine_run.clip` (best clip) → bridge → `self._persist_ingest` → `IngestFeedItem`. Build `IngestFeedResult`. Per-item failures non-fatal (`status="error"`).

- [ ] **Step 1: Failing test**: mock `resolve_feed_urls` (1 url) + `clip_engine_run.clip` (1 clip), assert `total_resolved==1`, `succeeded==1`, item `status=="ok"`. Plus a unit test for `resolve_feed_urls` with `yt_dlp` mocked.
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement `resolve_feed_urls` + rewrite `ingest_feed`.**
- [ ] **Step 4: Run → PASS** + neighbor tests.
- [ ] **Step 5: Commit** — `feat(ingest): route ingest_feed through the clip engine (YouTube-only)`

---

## Task 13: Deps, config keys, HTTP contract smoke test

**Files:** Modify `backend/requirements.txt`, `backend/pyproject.toml`, `backend/app/config.py`; Test `backend/tests/test_clip_engine_contract.py`

- [ ] **Step 1: Failing HTTP contract test** — `TestClient(app)` POST `/api/ingest/url` with the engine mocked (`pipeline_module.clip_engine_run.clip`) + temp DB, assert 200 and the response JSON has `reel`/`metadata`/`terms_notice`/`trace_id` and `reel` has `reel_id`/`video_url`/`t_start`/`t_end`/`captions`/`video_duration_sec`, and `video_url` starts with `https://www.youtube.com/embed/`. Ensure `SERVERLESS_MODE` is off in the test env (do NOT set `VERCEL`).
- [ ] **Step 2: Run → FAIL** (missing deps/config or wiring).
- [ ] **Step 3: Add deps** to `requirements.txt` + `pyproject.toml`: `google-genai>=1.0`, `tiktoken==0.8.0`, `rapidfuzz==3.10.1`. Add `Config` fields in `backend/app/config.py`: `clip_engine: str = "gemini"`, `supadata_api_key: str = ""`, `supadata_base: str = "https://api.supadata.ai/v1"`, `gemini_model: str = "gemini-2.5-flash"`, `segment_model: str = ""`, `clip_search_max_videos: int = 5`.
- [ ] **Step 4: Run → PASS**; then the full clip_engine suite + ingest tests: `backend/.venv/bin/python -m pytest backend/tests/ -k "clip_engine or ingest" -v`.
- [ ] **Step 5: Commit** — `feat(clip_engine): deps + config keys + HTTP contract smoke test`

---

## Self-Review (author checklist)
- Scraper port → Phase 1 Tasks 2–5; clipper → Phase 1 Tasks 6–7; translator+relevance → Task 8; endpoint rewiring reusing `_persist_ingest` → Tasks 9–12; deps/config/contract → Task 13.
- Contract fidelity comes from reusing `_persist_ingest` (canonical embed URL, takeaways, ai_summary, attribution, persistence) — verified by reading `pipeline.py:1247-1364`.
- Query relevance preserved via `bridge.filter_by_query` (Task 8), applied in topic-cut (Task 10) and available to the material flow (Phase 3).
- Open for Phase 3/4: material `/api/reels/*` + `/api/feed` refinement per concept; retire dead modules (`services/topic_cut.py`, `services/clip_boundary.py`, `app/ingestion/segment.py`, yt_dlp adapter search, IG/TT); gitignore `backend/work`; README + key rotation.
