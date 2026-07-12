# ReelAI FastAPI backend

The Python backend runs only as a durable Railway service. It is not deployed as Vercel Python functions.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Production requires PostgreSQL plus `SUPADATA_API_KEY` and `GEMINI_API_KEY`. The live iframe path reuses the practice fast engine: Supadata 180-character timed chunks, one whole-transcript Gemini pass, and quote-aligned iframe timestamps without punctuation restoration or local Whisper. Gemini 3.5 Flash runs first for every video; uncertain or invalid output falls back to Gemini 3.1 Pro. Native YouTube captions are optional because Supadata can generate a hosted timestamped transcript. No media is downloaded. Provider responses and model usage are persisted per generation, while successful search evidence is cached for six hours (empty results for 15 minutes) and validated timestamped transcript artifacts for 30 days.

Generation jobs are leased by the in-process worker with 15-second heartbeats, 90-second lease expiry, two maximum attempts, and a one-hour dead-job safety deadline. See the repository README for API and duration semantics.
