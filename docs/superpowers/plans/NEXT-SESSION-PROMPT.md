# Paste this to start the next session

Continue the reelAI **"practice clip-engine swap."** Phases 1–2 are DONE and merged; you're implementing **Phase 3** (study-material → reels rewire) then **Phase 4** (cleanup).

Repo root (run everything from here): `/Users/vincentfeng/Documents/reelai app/reelai/reelAI copy 2`. Branch: `codex/high-quality-video-retrieval` (not pushed).

**FIRST, read these two — they are self-contained, don't re-derive:**
- `docs/superpowers/plans/2026-07-06-clip-engine-swap-NEXT-SESSION-handoff.md` — the executable plan: the `reels.py:1487` seam, tasks **T0–T6**, the Phase-4 checklist, key rotation, and the **5 locked decisions** (no Groq → free local `faster-whisper`; **multiple clips per video**; practice-folder relevance, no concept-embedding post-filter; commit-then-supersede the pre-existing saliency WIP; **HARD-REPLACE** `generate_reels`, no feature flag). These are settled — don't re-ask.
- `.superpowers/sdd/progress.md` — the ledger of everything already done.

**How to work:** use `superpowers:subagent-driven-development` — a fresh implementer subagent per task + a per-task spec+quality review, TDD, frequent commits. Conventions: tests via `backend/.venv/bin/python -m pytest backend/tests -q` **from repo root**; import `from backend.app.X import …` (never `from app…`); DB-integrated test harness template = `backend/tests/test_clip_engine_ingest_url.py`; the clip engine reads keys from `os.environ` (gitignored `backend/.env`, engine is mocked in tests). After editing code files, rebuild the graphify code graph per the project `CLAUDE.md`.

**Order:**
1. **T0** — revert the `llm_router.py` Groq re-enable, then commit the pre-existing uncommitted `reels.py`/`segmenter.py`/`test_reels_saliency.py` saliency WIP as its own standalone commit.
2. **Phase 3 T1–T6** — thread `generation_id` through `_persist_ingest`→`upsert_reel_row`; add the multi-clip `ingest_topic` helper; wire `IngestionPipeline` into `ReelService`; **hard-replace** `generate_reels`' internals (delete the legacy `topic_cut` branch); verify feed + refinement + feedback; dry_run/can-generate parity. Retire the legacy `test_reels_*` tests that break (like the Phase-2 old-path tests).
3. **Phase 4** — "SAFE NOW" dead-code deletes now; after Phase 3 is verified, delete the legacy modules (`topic_cut.py`, `clip_boundary.py`, `importance_ranker.py`, …) + remove Groq (`groq_client.py`, `GROQ_API_KEY`, `groq` dep) + trim `scikit-learn`/`spacy`/`networkx`.

**Guardrails:** watch the latency/cost caveat (Supadata+Gemini per video over many concepts × videos) — add a per-generation video cap + light concurrency. Don't push or deploy without asking. The live Supadata+Gemini smoke test needs the real keys (already in `backend/.env`).

Start at **T0** and drive through the review gates; only stop to check in if a decision is genuinely ambiguous or a review finds a blocker.
