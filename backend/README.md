# ReelAI FastAPI backend

The Python backend runs only as a durable Railway service. It is not deployed as Vercel Python functions.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Production requires PostgreSQL plus `SUPADATA_API_KEY` and `GEMINI_API_KEY`. Supadata owns YouTube discovery and timestamped transcript retrieval. Each uncached inspected source gets at most one Gemini 3.5 Flash clip-selection call over the complete timestamped transcript; the YouTube video is not attached to Gemini. Gemini selects the exact first and last required words. The server may separately resolve an audio-only stream and widen those semantic boundaries outward to nearby quiet intervals, but it never contracts the selected speech and keeps a validated context-aligned boundary when acoustic verification is unavailable. Native YouTube captions may corroborate lexical timing but are not the semantic transcript. Provider responses and model usage are persisted per generation, while successful search evidence is cached for six hours (empty results for 15 minutes) and validated timestamped transcript artifacts for 30 days.

Generation jobs are leased by the in-process worker with 15-second heartbeats, 90-second lease expiry, two maximum attempts, and a one-hour dead-job safety deadline. See the repository README for API and duration semantics.
