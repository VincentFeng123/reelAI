# Local YouTube Topic-Clipper

Runs almost entirely on your Mac. Paste a YouTube URL + a **topic** (e.g. `momentum`) and
the AI decides how many clips to make, picks the most informative moments, and cuts them
into **separate** clips — each starting at a sentence start and **ending on a period**.

Default stack: **local faster-whisper** for transcription (word-level timestamps, no cloud)
+ **Gemini's free API** only for the AI selection step. Download, sentence alignment,
distinctness, and cutting are all local. (Groq is also supported via env vars — see below.)

## How it works

```
URL + topic
  → yt-dlp downloads 720p video + a 16 kHz mono audio track (cached per video id)
  → faster-whisper (local) → punctuated, word-level transcript
  → sentence index: split on real sentence boundaries (abbreviation/decimal guards),
    aligned to word-level millisecond times
  → AI selection: Gemini (structured JSON) picks distinct facets; a local cross-encoder
    pre-filters long videos; sentence-transformers MMR removes near-duplicates and
    decides the clip count from the content
  → snap each segment: start → sentence start, end → the nearest period
  → ffmpeg re-encodes a frame-accurate cut per clip → output/<video_id>/clip_N_<facet>.mp4
```

## Prerequisites

- macOS (Apple Silicon), **Python 3.12**, **Node 18+**
- **ffmpeg**: `brew install ffmpeg`
- A free **Gemini API key**: <https://aistudio.google.com/apikey> (no credit card)

## Setup

```bash
# 1. Backend venv + deps  (use Python 3.12 — torch/sentence-transformers wheels)
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure your key
cp .env.example .env
#   then edit .env and set GEMINI_API_KEY=...

# 3. Build the frontend (output lands in backend/static/)
cd frontend && npm install && npm run build && cd ..

# 4. Run
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000>. On your phone (same wifi) open `http://<your-mac-LAN-ip>:8000`.

> First run downloads, once: the faster-whisper model (~0.5 GB for `small`) and ~170 MB of
> CPU models (cross-encoder + MiniLM) into `~/.cache/huggingface`. CPU transcription of a
> 10-min video takes a few minutes; pick a smaller `WHISPER_MODEL` (e.g. `base`) to go faster.

### Using Groq instead (optional)
Set in `.env`: `TRANSCRIBER=groq`, `LLM_PROVIDER=groq`, `GROQ_API_KEY=gsk_...` (free key at
<https://console.groq.com>). Groq Whisper avoids the local model download and is much faster.

## Dev mode (hot reload)

```bash
# Terminal A — API
uvicorn backend.main:app --reload --port 8000
# Terminal B — UI with hot reload (proxies API to :8000)
cd frontend && npm run dev      # → http://localhost:5173
```

If port 8000 is busy, run uvicorn on another port (e.g. `--port 8008`) and update the
proxy targets in `frontend/vite.config.ts`.

## Settings (gear icon)

- **Transcription**: Groq (offline faster-whisper is stubbed for a future release)
- **Max resolution**: 480p / 720p
- **Allow `?`/`!` ends**: off by default (clips end only on a period `.`)
- **Variety ↔ Relevance**: MMR aggressiveness (lower = more distinct facets)

## Notes

- **Personal use only.** Downloading YouTube content may conflict with YouTube's Terms of
  Service. Use this for local, personal clipping of content you're allowed to use — it has
  no sharing/redistribution features by design.
- Intermediate files cache under `./work/<video_id>/`; clips are written to
  `./output/<video_id>/` (+ a `clips.json` manifest). Delete those folders to reclaim space.
- Groq's free tier has a 25 MB audio upload limit and per-minute rate limits; long videos
  are auto-downsampled to 16 kHz mono and, if still over 25 MB, chunked and merged.

## Future scaling (NOT needed for local use)

The MVP keeps all job state in memory and serves one machine. To host it for multiple users
you'd add a job queue (Redis + RQ/Celery) instead of the in-memory registry + asyncio tasks,
a database (Postgres) for job/clip persistence, object storage (S3) for outputs, and rotating
proxies for downloads. None of that is required to run this on your own Mac.
