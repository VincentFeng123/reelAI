# ReelAI FastAPI backend

The Python backend runs only as a durable Railway service. It is not deployed as Vercel Python functions.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Production requires PostgreSQL plus `SUPADATA_API_KEY` and `GEMINI_API_KEY`. Configure the primary segmentation model with `SEGMENT_MODEL`; `SEGMENT_FALLBACK_MODEL` is optional and is the only allowed model substitution. Native YouTube captions are mandatory. Provider responses and model usage are persisted per generation, while successful search evidence is cached for six hours (empty results for 15 minutes) and validated native transcript artifacts for 30 days.

Generation jobs are leased by the in-process worker with 15-second heartbeats, 90-second lease expiry, two maximum attempts, and an eight-minute deadline. See the repository README for API and duration semantics.
