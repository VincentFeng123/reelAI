# Design: Swap the backend scraper + clipper to the `practice/` engine

**Date:** 2026-07-05
**Status:** Draft for review
**Repo:** `reelai/reelAI copy 2` (FastAPI backend the iOS app + web frontend both consume)

## 1. Summary

Replace the website/iOS backend's video **scraping** (discovery/search) and **clipping**
(segment selection + boundaries) with the engine from the `practice/` folder
(VidScout + its Python FastAPI "clipper"). The client-facing `ReelOut` contract is
preserved exactly, so the **iOS app and the Next.js web frontend need zero changes**.

- **Scraper** today: `app/ingestion/adapters/yt_dlp_adapter.py` (yt-dlp search/download) and
  `services/youtube.py` (HTML/InnerTube scrape). → Replaced by VidScout's approach:
  topic → LLM/free expansion → **Supadata YouTube Search** → merge+rank (reimplemented in Python).
- **Clipper** today: `app/ingestion/segment.py`, `services/topic_cut.py`, `services/clip_boundary.py`
  (+ `reels.py` clipping). → Replaced by the practice clipper pipeline
  (`practice/clips/backend/`), defaulting to the **`gemini`** clip engine with
  `OUTPUT_MODE=embed`.
- **Transcripts** today: `youtube-transcript-api` → local `faster-whisper` fallback. →
  Replaced by **Supadata transcripts** (the practice default).

This deliberately **reintroduces paid APIs** (Supadata + Gemini), reversing the prior
"API blackout." The user approved this explicitly for the clip engine.

## 2. Goals / Non-goals

**Goals**
- Bring the practice folder's clipping quality into the live products, as-is (paid).
- Keep the `ReelOut` / `ReelOutWithAttribution` response shape byte-compatible → no client changes.
- Cover both entry paths: `/api/ingest/*` (topic/URL) **and** the legacy study-material
  flow (`/api/reels/generate`, `/api/reels/generate-stream`, `/api/feed`).
- Keep the backend a single FastAPI service on the existing Railway/Docker deploy.
- Keep the deployed image light: default to the `gemini` engine (no torch / no faster-whisper /
  no video download).

**Non-goals**
- No changes to iOS app or web frontend code (contract preserved).
- No changes to auth, community, chat, feedback, history, material parsing, or concept
  extraction — only the *search + clip* portion of the material flow is swapped.
- Instagram/TikTok clipping is **dropped** (practice clipper is YouTube-only).
- Not adopting VidScout's Node server or the practice clipper's job/SSE HTTP surface —
  we vendor the Python pipeline and call it in-process.
- The `full`/`topic` structure-first engine and multimodal/whisper features are **not** the
  default and are **not** required to be deployable in this project (may be left as
  lazy-imported, dev-only options).

## 3. Current-state facts (verified)

**Backend** = one FastAPI app `backend/app/main.py`, deployed to Railway via `Dockerfile`
(`uvicorn backend.app.main:app`, port 8000, ffmpeg installed). `api/*.py` are 34-byte Vercel
shims importing the same app. The Next.js frontend (`src/`) and the iOS app are pure HTTP clients.

**Endpoints to rewire:**
| Endpoint | main.py | Current backend call |
| --- | --- | --- |
| POST `/api/ingest/url` | 5849 | `ingestion_pipeline.ingest_url` → 1 clip |
| POST `/api/ingest/topic-cut` | 5902 | `ingest_topic_cut` → `topic_cut.cut_video_into_topic_reels` |
| POST `/api/ingest/search` | 5958 | `ingest_search` → yt-dlp search + fan-out ingest |
| POST `/api/ingest/feed` | 6016 | `ingest_feed` → resolve profile/hashtag/playlist |
| POST `/api/reels/generate` | 5762 | legacy `ReelService` (search + embed-ranked clip) |
| POST `/api/reels/generate-stream` | 6068 | SSE version of the above |
| GET `/api/feed` | 6696 | ranked feed from persisted reels + queues refinement |

**Client contract** = `ReelOut` (`backend/app/models.py:73-100`): `video_url` (YouTube embed
w/ `?start=&end=`), `t_start`, `t_end` (float sec), `captions: [{start,end,text}]`,
`transcript_snippet`, `takeaways`, `ai_summary`, plus scores. Ingest endpoints return
`ReelOutWithAttribution` (`ingestion/models.py:156`) inside `IngestResult` /
`IngestSearchResult` / `IngestTopicCutResult` (each with `metadata: IngestMetadata`,
`trace_id`). Persisted via `ingestion/persistence.py` keyed on `(material_id, video_id, t_start, t_end)`.

**Practice engine facts:**
- Input: `CreateJobReq{url, topic, settings}`, **YouTube-only** (`schemas.py` `YT_RE`).
- Config single-source: `practice/clips/backend/config.py`. Relevant defaults:
  `TRANSCRIBER=supadata`, `LLM_PROVIDER=gemini`, `OUTPUT_MODE=embed`, `CLIP_ENGINE=topic`
  (we override to `gemini`), `PRECISE_BOUNDARIES=1` (we set `0` for the light path),
  `SEGMENT_FINE_SNAP=1`.
- `gemini` engine = `pipeline/gemini_segment.py`: one Gemini pass over the Supadata transcript
  → `{title,start,end}` clips, fine-snapped to interpolated word times. Skips punctuation /
  structure / whisper / multimodal → **no torch, no download**.
- Async job model (`main.py` + `jobs.py` + `orchestrator.py`); we call the pipeline function
  directly and run it inline.
- **Supadata search is NOT in the clipper** — it lives in VidScout Node (`practice/lib/supadata.js`,
  `expand.js`, `rank.js`, `related.js`). That logic must be reimplemented in Python.

## 4. Architecture

Vendor + reimplement into the existing FastAPI backend. No new service. No client changes.

```
iOS app / web frontend  (UNCHANGED — same /api/* paths, same ReelOut JSON)
        │
        ▼
backend/app/main.py  endpoints (ingest/*, reels/*, feed)  ← rewired to call:
        │
        ▼
backend/app/clip_engine/            ← NEW package
   ├─ search.py        (Python port of VidScout: expand → Supadata search → rank)
   ├─ clipper/         (vendored practice/clips/backend pipeline, gemini path)
   │    ├─ config.py   (trimmed / env-driven; gemini+embed defaults)
   │    ├─ supadata_client.py, transcribe.py (Supadata transcript)
   │    ├─ gemini_segment.py, gemini_client.py, llm.py, sentences.py, embed.py
   │    └─ (understand/, assemble/, boundary.py, cut.py — vendored but lazy/optional)
   ├─ run.py           (inline runner: url+topic+settings → clip dicts; replaces jobs/SSE)
   └─ adapter.py       (clip dicts → ReelOut / ReelOutWithAttribution)
```

**Data flow, per entry point:**

- `POST /api/ingest/url` — validate YouTube URL → `run.clip(url, topic)` → take top clip →
  `adapter` → `IngestResult`. (Non-YouTube URL → 400 "YouTube only".)
- `POST /api/ingest/topic-cut` — `run.clip(url, topic)` → all clips → `IngestTopicCutResult`.
- `POST /api/ingest/search` — `search.discover(topic, exclude_video_ids)` (Supadata) →
  top-N YouTube videos → for each `run.clip(...)` (bounded pool) → flatten →
  `IngestSearchResult`.
- `POST /api/ingest/feed` — resolve list of YouTube URLs (keep yt-dlp `extract_flat` for
  playlist/channel resolution, YouTube-only) → per-URL `run.clip` → `IngestFeedResult`.
- `POST /api/reels/generate[-stream]` — material's extracted concepts (unchanged) → for each
  concept `search.discover(concept)` → best video → `run.clip` → `ReelOut[]`; stream reels as
  they finish (reuse the existing NDJSON/SSE plumbing). Persist via existing `persistence`.
- `GET /api/feed` — unchanged read/rank/paginate over persisted reels; its background
  "refinement" top-up calls the same new generate path.

**Sync vs async:** the practice pipeline is normally a background job. We run it **inline**
within the request, inside the client's existing timeouts (iOS: 180s ingest / 300s search /
900s generate). The `gemini` engine on one video is seconds, not minutes. `search` and
material `generate` bound the number of videos (config `CLIP_SEARCH_MAX_VIDEOS`, default ~5)
and use a small thread pool, matching today's 2-worker fan-out.

## 5. Component detail

### 5.1 Scraper port — `clip_engine/search.py`
Reimplement VidScout's Node scraper in Python:
- **Expand** (`lib/expand.js` + `lib/related.js`): topic → N diverse queries. Use Gemini
  (already a dep) for the LLM path; keep the free Wikipedia-opensearch + YouTube-autocomplete
  fallback so search still works without extra keys.
- **Search** (`lib/supadata.js`): `GET https://api.supadata.ai/v1/youtube/search` with
  `x-api-key`, sequential with 429 backoff, `sortBy/uploadDate/duration` filters, 1 credit/query.
- **Rank** (`lib/rank.js`): merge across queries, dedupe by video id, rank by match-count then
  `log10(views)`.
- **Dedup** against `exclude_video_ids` (for infinite scroll) — already part of the current
  ingest contract.
Output: ranked list of `{video_id, url, title, channel, views, ...}` YouTube videos.

### 5.2 Clipper vendor — `clip_engine/clipper/`
Copy `practice/clips/backend/` pipeline modules needed for the `gemini` + `embed` path:
`config.py`, `supadata_client.py`, `pipeline/transcribe.py`, `pipeline/gemini_segment.py`,
`gemini_client.py`, `llm.py`, `pipeline/sentences.py`, `embed.py`, `errors.py`, plus their
direct helpers. The heavier modules (`pipeline/understand/`, `pipeline/assemble/`,
`pipeline/boundary.py`, `pipeline/cut.py`, `pipeline/download.py`) are vendored but their
imports made **lazy** so the gemini path never imports torch / sentence-transformers /
faster-whisper / yt-dlp. `config.py` is adapted to read from the backend's env/`Config`
(no `.env` file load), with defaults `CLIP_ENGINE=gemini`, `OUTPUT_MODE=embed`,
`PRECISE_BOUNDARIES=0`, `TRANSCRIBER=supadata`, `LLM_PROVIDER=gemini`.

### 5.3 Inline runner — `clip_engine/run.py`
Replaces the FastAPI job/SSE surface (`main.py`, `jobs.py`, `orchestrator.py`) with a plain
function: `clip(url, topic, settings) -> list[ClipDict]`. Reuses `orchestrator.run_pipeline`'s
gemini branch logic (transcribe → gemini_segment → fine-snap → build embed clips) minus the
job registry and SSE. Progress for the streaming endpoints is surfaced via the existing
NDJSON/SSE emitters, not the clipper's SSE.

### 5.4 Adapter — `clip_engine/adapter.py`
Map a clip dict `{title, start, end, embed_url, ...}` + video metadata + transcript cues →
`ReelOut` / `ReelOutWithAttribution`:
- `video_url` = YouTube embed for `video_id` with `?start=floor(start)&end=ceil(end)`
  (matches current behavior and the iOS IFrame player).
- `t_start`/`t_end` = clip start/end. `video_duration_sec`, `clip_duration_sec` computed.
- `captions` = transcript cues within the window. `transcript_snippet`, `takeaways`,
  `ai_summary` from the clip/segment (gemini can produce a title/summary; `takeaways` may be
  best-effort or empty — the client tolerates missing fields).
- Scores default sensibly (the client only strictly needs `reel_id, video_url, t_start, t_end,
  video_duration_sec`).

### 5.5 Endpoint rewiring — `backend/app/main.py`
Point the seven handlers at the new package. Remove/retire the IG/TT branches. Keep request/
response models, auth, rate limiting, persistence, and error envelopes exactly as they are.

### 5.6 Config / deps / hosting
- **Deps to add** to `requirements.txt` for the light path: `google-genai>=1.0`, plus whatever
  `gemini_segment`/`sentences`/`supadata_client` import (`tiktoken`, `pysbd` already present,
  `numpy` present). **Not** added by default: `torch`, `sentence-transformers`, `faster-whisper`,
  `groq`, `zipstream-ng` (only needed by the non-default engines/features; gate behind an extra).
- **Env/keys:** `SUPADATA_API_KEY`, `GEMINI_API_KEY` on Railway. New tunables surfaced through
  `backend/app/config.py`: `CLIP_ENGINE` (default `gemini`), `CLIP_SEARCH_MAX_VIDEOS`,
  `SUPADATA_BASE`, `GEMINI_MODEL`, `SEGMENT_MODEL`.
- **Hosting:** no torch → image stays close to today's size; Supadata/Gemini are network calls.
  Note the Supadata credit model (~1 credit/expanded search query + ~1/transcript per video).

### 5.7 Security
`practice/clips/.env` has **real committed keys** (Supadata `sd_…`, Gemini `AQ…`). Rotate them,
delete from git history, and load only from Railway env. Do not copy that `.env` into the backend.

## 6. Testing strategy

- **Adapter unit tests**: clip dict + fixture transcript → assert `ReelOut` fields, embed URL
  `?start=&end=`, window clamping to `video_duration_sec`.
- **Scraper unit tests**: mock Supadata search responses → assert expansion, dedupe-by-id,
  ranking order, `exclude_video_ids` honored; free-path fallback when no LLM key.
- **Runner integration test** (recorded/mocked Supadata transcript + mocked Gemini) →
  `clip(url, topic)` returns well-formed clip dicts with monotonic, in-bounds windows.
- **Endpoint contract tests**: hit each rewired endpoint with a mocked engine → response
  JSON matches the current `ReelOut`/`Ingest*Result` schema the iOS decoder expects
  (snake_case, required fields present). A golden-schema compare guards the client contract.
- **YouTube-only guard**: IG/TT URL → 400; `/api/ingest/search` returns YouTube results only.
- Reuse the existing audit harness (`audit_*` / `audit_work/`) against a pinned corpus to
  compare clip quality before/after.

## 7. Phasing (incremental delivery)

1. **Phase 1 — foundation**: vendor clipper (gemini path) + `run.py` + `adapter.py` +
   `search.py`; unit/integration tests green. No endpoints changed yet.
2. **Phase 2 — ingest path**: rewire `/api/ingest/url`, `/topic-cut`, `/search`, `/feed`;
   drop IG/TT; contract tests green. This is the primary iOS create flow.
3. **Phase 3 — material path**: rewire `/api/reels/generate[-stream]` and the `/api/feed`
   refinement top-up to run each extracted concept through search+clip; persistence/ranking
   unchanged.
4. **Phase 4 — cleanup**: retire dead scraper/clipper modules that are now unreferenced
   (`yt_dlp_adapter` search, `topic_cut.py`, `clip_boundary.py`, legacy clip parts of
   `reels.py`, IG/TT adapters); update `requirements.txt`; rotate keys; update README.

Each phase is independently shippable and preserves the client contract.

## 8. Risks / open items

- **Latency**: `gemini` + Supadata per video is fast, but `/api/ingest/search` and material
  `generate` fan out over N videos — bound N and pool; lean on the streaming endpoints so the
  client sees reels as they land.
- **Supadata cost/limits**: search + transcript both bill; add per-request video caps and
  reuse the existing `transcript_cache`/`search_cache` tables to avoid re-billing.
- **`reels.py` blast radius** (Phase 3): 13k lines, woven into feed ranking + refinement.
  Mitigation: introduce a thin `generate_reels_for_concepts()` seam that the new engine backs,
  leaving ranking/persistence/feedback intact.
- **Gemini model availability**: practice uses `gemini-2.5-flash` / a pro-preview for topic
  selection. Confirm the exact model ids are enabled on the account; make them env-overridable.
- **Takeaways/summary richness**: the gemini-segment path yields less metadata than the `full`
  engine; acceptable since fields are optional, but note the UX diff vs the practice demo.
- **`vercel.json` serverless path**: `/api/ingest/*` is already disabled in serverless
  (`SERVERLESS_MODE`); keep that guard — the engine runs only on Railway.
