# ReelAI YouTube Clipping Workspace

This repository contains two distinct clipping workflows:

- The hosted ReelAI study application uses Supadata-hosted timestamped transcripts, Gemini selection, and timestamped YouTube embeds. It does not download media or run local Whisper.
- The local YouTube Topic-Clipper downloads media and uses local or Groq transcription plus ffmpeg to create physical clip files.

## Hosted ReelAI application

ReelAI turns study materials into a learner-specific feed of transcript-grounded YouTube clips.

### Production architecture

- Vercel hosts only the Next.js application. `RAILWAY_BACKEND_ORIGIN` configures the same-origin `/api/*` rewrite.
- Railway hosts the durable FastAPI process and generation worker.
- PostgreSQL stores material data, generation jobs/events, provider usage, Supadata search evidence, and timestamped transcript artifacts.
- Supadata supplies YouTube search and hosted timestamped transcript cues, preferring native captions and generating a transcript when captions are unavailable.
- Gemini selects self-contained clips on exact transcript-cue boundaries. `SEGMENT_FALLBACK_MODEL` is the only permitted fallback and is disabled when blank.

### Generation contract

`POST /api/reels/generate` returns either:

- `200` with an already-completed matching inventory; or
- `202` with `job_id`, `status_url`, and `stream_url`.

Jobs move through `queued → running → completed | partial | exhausted | failed | cancelled`. Railway workers lease jobs for 90 seconds, heartbeat every 15 seconds, make at most two lease attempts, and enforce an eight-minute deadline. Disconnecting a client only closes its subscription.

`GET /api/reels/generation-stream/{job_id}?after_seq=N` replays ordered NDJSON events:

- `candidate`: provisional reel;
- `final`: authoritative ordered/capped inventory;
- `terminal`: final status or typed error.

Every event includes `job_id`, a monotonic `seq`, and a timestamp. Status and cancellation are available at `/api/reels/generation-status/{job_id}` and `/api/reels/generation-jobs/{job_id}/cancel`.

### Retrieval and duration semantics

Only canonical YouTube video, playlist, and channel URLs are accepted by ingestion surfaces. Short topics search the exact user text first. Inputs longer than 12 words use a validated AI phrase summary as the practical first query while retaining the original text for final relevance checks. Remaining search calls use a validated AI one-word topic and its one-word synonyms; these non-primary requests prefer HD sources. Creative Commons and source-duration filters map directly to Supadata when selected. Transcripts use `mode=auto`: native captions are used when available and Supadata generates hosted timestamped cues otherwise.

Clip duration is a preference. Valid self-contained clips are persisted inside the global 1–180 second safety envelope. Serving ranks requested-range clips first, then fills shortages with the nearest shorter/longer clips and reports `duration_preference_met` plus `duration_fit`.

Fast jobs allow three searches, three transcript requests, three segmentation calls, and one acquisition pass. Slow jobs begin with up to six searches/five videos and may make two continuation passes, with job-wide ceilings of 12 searches, 10 transcript requests, and 10 segmentation calls.

### Local development

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

## Local YouTube Topic-Clipper

Paste a YouTube URL and topic to create separate local clips that start at sentence boundaries and end on complete thoughts.

The default local stack uses faster-whisper for word-level transcription and Gemini for selection. Downloading, sentence alignment, distinctness checks, and cutting run locally. Groq is also supported through environment variables.

### How it works

```text
URL + topic
  → yt-dlp downloads 720p video and a 16 kHz mono audio track
  → faster-whisper creates a punctuated, word-level transcript
  → sentence boundaries are aligned to millisecond timestamps
  → Gemini selects distinct, informative facets
  → segments snap to complete sentence boundaries
  → ffmpeg writes frame-accurate clips under output/<video_id>/
```

### Prerequisites

- macOS on Apple Silicon
- Python 3.12 and Node 18+
- ffmpeg: `brew install ffmpeg`
- A Gemini API key from <https://aistudio.google.com/apikey>

### Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Set GEMINI_API_KEY in .env.

cd frontend
npm install
npm run build
cd ..

uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000>. On a phone connected to the same network, open `http://<your-mac-LAN-ip>:8000`.

The first run downloads the configured faster-whisper model and local ranking models. CPU transcription time scales with video length; choose a smaller `WHISPER_MODEL` when lower latency matters.

To use Groq, set `TRANSCRIBER=groq`, `LLM_PROVIDER=groq`, and `GROQ_API_KEY` in `.env`.

### Development mode

```bash
# Terminal A
uvicorn backend.main:app --reload --port 8000

# Terminal B
cd frontend
npm run dev
```

### Settings

- Transcription provider
- Maximum source resolution
- Whether `?` and `!` may end clips
- Variety versus relevance

### Notes

- Use the local downloader only for content you are permitted to process.
- Intermediate files are cached under `work/<video_id>/`.
- Clips and their manifest are written under `output/<video_id>/`.
- Groq has upload-size and rate limits; long audio may be downsampled, chunked, and merged.

The local workflow keeps job state in memory. Multi-user hosting would additionally require a durable queue, database, object storage, and appropriate download infrastructure.
