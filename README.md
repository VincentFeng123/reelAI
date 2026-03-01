# StudyReels MVP

StudyReels is a TikTok/Reels-style study feed that takes user material (PDF/DOCX/TXT/text), extracts concepts, finds matching YouTube videos, selects transcript-aligned 20-60s segments, and serves them in a vertical infinite feed.

- Retrieval strategy: search short-form videos first, then broaden to longer videos and cut transcript-matched segments.
- Feed strategy: bootstrap with ~7 reels and auto-generate more while the user scrolls.

The MVP uses timestamp-based YouTube embed playback (no local download of full YouTube videos).

## 1) High-level architecture

```text
[Next.js Frontend]
  - Upload material/text
  - Trigger reel generation
  - Vertical feed + feedback
        |
        v
[FastAPI Backend]
  /api/material
    -> parse file/text
    -> concept extraction + objectives
    -> chunk + embeddings
    -> persist material/concepts/chunks

  /api/reels/generate
    -> YouTube search API (cached)
    -> transcript fetch (cached)
    -> transcript chunking (15-30s blocks)
    -> embedding similarity (FAISS cosine)
    -> merge/trim to 20-60s segments
    -> persist reels

  /api/feed
    -> rank by base relevance + feedback signal

  /api/reels/feedback
    -> persist helpful/confusing/rating/saved
    -> affects future feed rank
        |
        v
[SQLite]
  materials, concepts, chunks, reels, feedback, caches

[Storage Layer]
  - Local filesystem (dev)
  - S3-compatible adapter (prod)
```

## 2) DB schema (SQLite)

Core tables:
- `materials(id, subject_tag, raw_text, source_type, source_path, created_at)`
- `concepts(id, material_id, title, keywords_json, summary, embedding_json, created_at)`
- `material_chunks(id, material_id, chunk_index, text, embedding_json, created_at)`
- `videos(id, title, channel_title, description, duration_sec, is_creative_commons, created_at)`
- `transcript_chunks(id, video_id, chunk_index, t_start, t_end, text, embedding_json, created_at)`
- `reels(id, material_id, concept_id, video_id, video_url, t_start, t_end, transcript_snippet, takeaways_json, base_score, created_at)`
- `reel_feedback(id, reel_id, helpful, confusing, rating, saved, created_at)`
- `search_cache(cache_key, response_json, created_at)`
- `transcript_cache(video_id, transcript_json, created_at)`
- `embedding_cache(text_hash, embedding_json, created_at)`

Defined in: `backend/app/db.py`.

## 3) API contract

### POST `/api/material`
Input: `multipart/form-data`
- `file` (optional): `.pdf | .docx | .txt`
- `text` (optional): raw text
- `subject_tag` (optional)

Example response:
```json
{
  "material_id": "cfd2276d-5685-4507-97fd-012fed6f9445",
  "extracted_concepts": [
    {
      "id": "e8f02cd7-fec9-4368-babc-a1f6ac13f6e7",
      "title": "Gradient Descent",
      "keywords": ["gradient descent", "learning rate", "loss function"],
      "summary": "Gradient descent updates parameters in the opposite direction of the gradient."
    }
  ]
}
```

### POST `/api/reels/generate`
Input JSON:
```json
{
  "material_id": "cfd2276d-5685-4507-97fd-012fed6f9445",
  "concept_id": null,
  "num_reels": 7,
  "creative_commons_only": false
}
```

Example response:
```json
{
  "reels": [
    {
      "reel_id": "04eec7b9-5c89-4ca6-95e4-35f2b74ae96f",
      "concept_id": "e8f02cd7-fec9-4368-babc-a1f6ac13f6e7",
      "concept_title": "Gradient Descent",
      "video_url": "https://www.youtube.com/embed/VIDEO_ID?start=120&end=156&autoplay=1&mute=1&playsinline=1&rel=0",
      "t_start": 120,
      "t_end": 156,
      "transcript_snippet": "...",
      "takeaways": ["...", "...", "..."],
      "score": 0.83
    }
  ]
}
```

### GET `/api/feed?material_id=...&page=1&limit=5&autofill=true&prefetch=7`
Example response:
```json
{
  "page": 1,
  "limit": 5,
  "total": 24,
  "reels": [
    {
      "reel_id": "04eec7b9-5c89-4ca6-95e4-35f2b74ae96f",
      "concept_id": "e8f02cd7-fec9-4368-babc-a1f6ac13f6e7",
      "concept_title": "Gradient Descent",
      "video_url": "https://www.youtube.com/embed/VIDEO_ID?start=120&end=156&autoplay=1&mute=1&playsinline=1&rel=0",
      "t_start": 120,
      "t_end": 156,
      "transcript_snippet": "...",
      "takeaways": ["...", "...", "..."],
      "score": 1.12,
      "concept_position": 3,
      "total_concepts": 12
    }
  ]
}
```

### POST `/api/reels/feedback`
Input JSON:
```json
{
  "reel_id": "04eec7b9-5c89-4ca6-95e4-35f2b74ae96f",
  "helpful": true,
  "confusing": false,
  "rating": 5,
  "saved": false
}
```

Response:
```json
{
  "status": "ok",
  "reel_id": "04eec7b9-5c89-4ca6-95e4-35f2b74ae96f"
}
```

## 4) Repo tree

```text
.
├── api
│   └── index.py                # Vercel Python entrypoint -> FastAPI app
├── requirements.txt            # Includes backend/requirements.txt for Vercel python runtime
├── README.md
├── backend
│   ├── .env.example
│   ├── requirements.txt
│   ├── README.md
│   └── app
│       ├── main.py
│       ├── config.py
│       ├── db.py
│       ├── models.py
│       └── services
│           ├── concepts.py
│           ├── embeddings.py
│           ├── parsers.py
│           ├── reels.py
│           ├── segmenter.py
│           ├── storage.py
│           ├── text_utils.py
│           ├── vector_search.py
│           └── youtube.py
├── package.json
├── next.config.ts
└── src
    ├── app
    │   ├── page.tsx
    │   ├── feed/page.tsx
    │   ├── layout.tsx
    │   └── globals.css
    ├── components
    │   ├── UploadPanel.tsx
    │   └── ReelCard.tsx
    └── lib
        ├── api.ts
        └── types.ts
```

## 5) Implementation steps (what the code does)

1. Ingest material via `/api/material` (text or file).
2. Parse text from PDF/DOCX/TXT.
3. Extract concepts/objectives heuristically from headings + frequent terms.
4. Embed concepts and material chunks (OpenAI embeddings; hash fallback if key missing).
5. For each concept, search YouTube (cached by query).
6. Fetch transcript (cached by video id).
7. Chunk transcript into 15-30s blocks, embed blocks, rank against concept with FAISS cosine.
8. Merge adjacent matched chunks and constrain final segment to 20-60s.
9. Build reel objects with timestamp-based embed URLs.
10. Feed ranking combines base similarity + feedback signal (`helpful/confusing/rating/saved`) at reel and concept level.

## 6) Core files to review

- Backend API entry: `backend/app/main.py`
- Retrieval and segment generation: `backend/app/services/reels.py`
- Segmenter logic: `backend/app/services/segmenter.py`
- YouTube search + transcript caching: `backend/app/services/youtube.py`
- Vercel API entrypoint: `api/index.py`
- Frontend feed page: `src/app/feed/page.tsx`
- Frontend reel UI: `src/components/ReelCard.tsx`

## 7) Run locally

### Backend
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# set OPENAI_API_KEY + YOUTUBE_API_KEY in .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend
```bash
cd /Users/vincentfeng/Documents/reelAI
npm install
cp .env.local.example .env.local
npm run dev
```

Open `http://localhost:3000`.

### Keep It Hosted In Background
From repo root:

```bash
./scripts/host-up.sh
./scripts/host-status.sh
./scripts/host-down.sh
```

- `host-up.sh` starts backend on `127.0.0.1:8000` and frontend on `127.0.0.1:3001`.
- Backend in `host-up.sh` runs in stable mode (no auto-reload watcher) for persistent hosting reliability.
- It clears `.next` before starting frontend to avoid stale chunk runtime errors.
- Logs are written to `.logs/backend.log` and `.logs/frontend.log`.

## 8) Deploy on Vercel (single project)

1. Import this GitHub repo into Vercel as a single project.
2. Set **Root Directory** to the repo root (`.`).
3. Add environment variables:
   - `APP_ENV=prod`
   - `FRONTEND_ORIGIN=https://<your-project-domain>`
   - `OPENAI_API_KEY=...`
   - `YOUTUBE_API_KEY=...`
   - Optional: S3 variables if using object storage.
4. Optional frontend variable:
   - `NEXT_PUBLIC_API_BASE` (leave unset for same-origin `/api`; set only if pointing to a different backend URL)
5. Deploy.

How this works in one project:
- Next.js serves the frontend from root (`src/...`).
- Vercel runs Python API at `api/index.py` and `api/[...path].py`, both reusing `backend/app/main.py`.
- Root `requirements.txt` pulls backend dependencies for the Python runtime.

### Verify

- Frontend loads: `https://<project-domain>`
- Backend health works: `https://<project-domain>/api/health`

## 9) Legal/ToS constraints and MVP choice

- This MVP does **not** download or redistribute full YouTube videos.
- Playback uses YouTube embed URLs with `start/end` timestamps.
- Creative Commons filtering is available via `creative_commons_only` input, using YouTube search license filter and license metadata check.

## 10) Known limitations and next improvements

Known limitations:
- Transcript availability is not guaranteed; fallback currently uses video description with low confidence.
- No user auth/multi-user separation yet.
- Ranking is heuristic and not personalized beyond immediate feedback.
- FAISS index is built at query-time from persisted vectors (no persistent ANN index file yet).
- Vercel serverless storage is ephemeral; SQLite data under `/tmp` is not durable across cold starts.

High-impact next steps:
- Add concept-level spaced repetition scheduling and mastery tracking.
- Add richer transcript sources (official caption APIs where available).
- Add async job queue for generation to reduce request latency.
- Add persistent FAISS indexes and background refresh.
- Add authentication + per-user saved reels + progress dashboard.
