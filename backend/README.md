# StudyReels Backend (FastAPI)

## Run

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Notes

- Uses SQLite + FAISS (with brute-force fallback if FAISS import fails).
- Uses timestamp-based YouTube embed playback; no local video download.
- Reel segments are forced to 20-60s and are selected from transcript-aligned timestamps.
- Searches both short-form and long-form videos, then cuts matching transcript windows.
- Caches YouTube search results, transcripts, embeddings, and LLM concept extraction in SQLite.
- If `OPENAI_API_KEY` is set, upload parsing is enhanced with GPT-generated concepts/objectives; otherwise it falls back to heuristics.

## Deploy on Vercel

Deploy the repo as a single project from root (`.`).

- Python entrypoint is `api/index.py`, which imports `backend.app.main:app`.
- Root `requirements.txt` includes `backend/requirements.txt` for dependency install.
- Set env vars in Vercel:
  - `APP_ENV=prod`
  - `FRONTEND_ORIGIN=https://<your-project-domain>`
  - `OPENAI_ENABLED=0` (set `1` only if you want to enable OpenAI calls)
  - `OPENAI_API_KEY=...`
  - `YOUTUBE_API_KEY=...`
- `DATA_DIR` defaults to `/tmp/studyreels-data` on Vercel.
