# StudyReels Frontend (Next.js)

## Run

```bash
cd frontend
npm install
cp .env.local.example .env.local
npm run dev
```

Open `http://localhost:3000`.

## Deploy on Vercel

Deploy this folder as a separate Vercel project with root directory `frontend`.

- Set `NEXT_PUBLIC_API_BASE` to your backend Vercel URL, e.g.:
  `https://studyreels-backend.vercel.app`
