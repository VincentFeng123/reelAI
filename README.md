# ReelAI

ReelAI turns study materials into a learner-specific feed of transcript-grounded YouTube clips.

## Production architecture

- Vercel hosts only the Next.js application. `RAILWAY_BACKEND_ORIGIN` configures the same-origin `/api/*` rewrite.
- Railway hosts the durable FastAPI process and generation worker.
- PostgreSQL stores material data, generation jobs/events, provider usage, Supadata search evidence, and timestamped transcript artifacts.
- Supadata supplies YouTube search and hosted timestamped transcript cues, preferring native captions and generating a transcript when captions are unavailable. Instagram, TikTok, audio download, local Whisper, and synthetic word timing are not supported.
- Gemini selects self-contained clips on exact transcript-cue boundaries. `SEGMENT_FALLBACK_MODEL` is the only permitted fallback and is disabled when blank.

## Generation contract

`POST /api/reels/generate` returns either:

- `200` with an already-completed matching inventory; or
- `202` with `job_id`, `status_url`, and `stream_url`.

Jobs move through `queued → running → completed | partial | exhausted | failed | cancelled`. Railway workers lease jobs for 90 seconds, heartbeat every 15 seconds, make at most two lease attempts, and enforce an eight-minute deadline. Disconnecting a client only closes its subscription.

`GET /api/reels/generation-stream/{job_id}?after_seq=N` replays ordered NDJSON events:

- `candidate`: provisional reel;
- `final`: authoritative ordered/capped inventory;
- `terminal`: final status or typed error.

Every event includes `job_id`, a monotonic `seq`, and a timestamp. Status and cancellation are available at `/api/reels/generation-status/{job_id}` and `/api/reels/generation-jobs/{job_id}/cancel`.

## Retrieval and duration semantics

Only canonical YouTube video, playlist, and channel URLs are accepted by ingestion surfaces. Discovery always includes an unrestricted literal query, while duplicate and expanded searches may prefer HD sources. Creative Commons and source-duration filters map directly to Supadata when selected. Transcripts use `mode=auto`: native captions are used when available and Supadata generates hosted timestamped cues otherwise. The service does not download media or run local Whisper.

Clip duration is a preference. Valid self-contained clips are persisted inside the global 1–180 second safety envelope. Serving ranks requested-range clips first, then fills shortages with the nearest shorter/longer clips and reports `duration_preference_met` plus `duration_fit`.

Fast jobs allow three searches, three transcript requests, three segmentation calls, and one acquisition pass. Slow jobs begin with up to six searches/five videos and may make two continuation passes, with job-wide ceilings of 12 searches, 10 transcript requests, and 10 segmentation calls.

## Local development

```bash
cp backend/.env.example backend/.env
python3 -m venv backend/.venv
backend/.venv/bin/pip install -r backend/requirements.txt
backend/.venv/bin/uvicorn backend.app.main:app --reload --port 8000

cp .env.local.example .env.local
npm install
npm run dev
```

For production, set `DATABASE_URL`, `DATA_DIR=/data`, `SUPADATA_API_KEY`, `GEMINI_API_KEY`, `SEGMENT_MODEL`, and optionally `SEGMENT_FALLBACK_MODEL` on Railway. Set `RAILWAY_BACKEND_ORIGIN` on Vercel. Provider-backed live smoke tests are separately gated because they consume credits; mocked provider tests are the CI requirement.
