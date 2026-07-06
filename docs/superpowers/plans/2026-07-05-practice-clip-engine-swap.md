# Practice Clip-Engine Swap — Implementation Plan (Phases 1–2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the backend's video scraper + clipper with the `practice/` engine (Supadata search + Supadata transcript + Gemini `gemini`-segment clipping), preserving the `ReelOut` contract so the iOS app and web frontend are unchanged.

**Architecture:** Add a self-contained `backend/app/clip_engine/` package: a Python port of VidScout's Supadata search (expand → search → rank) and a vendored copy of the practice clipper's `gemini` path (Supadata transcript → one Gemini pass → embed clips), plus an adapter to `ReelOut`. Then rewire `IngestionPipeline`'s four methods to call it — the `/api/ingest/*` HTTP handlers, error mapping, and response models stay byte-identical.

**Tech Stack:** Python 3.12, FastAPI, Pydantic v2, `google-genai` (Gemini), Supadata REST API, `httpx`/`requests`, `pytest`. No `torch` / `faster-whisper` / `yt-dlp` download in the default path.

**Scope of THIS plan:** Phase 1 (engine package) + Phase 2 (rewire `/api/ingest/*`). Phase 3 (the study-material `/api/reels/generate[-stream]` + `/api/feed` refinement path through `reels.py`) and Phase 4 (dead-code cleanup, key rotation, README) are a **separate follow-up plan**, authored after this lands so its steps can reference the real, working engine.

## Global Constraints

- **Test & import convention (this repo).** Run tests with `backend/.venv/bin/python -m pytest <path> -v` **from the repo root** (`reelai/reelAI copy 2`). Import app modules as `backend.app.X` (e.g. `from backend.app.clip_engine.config import ...`, `from backend.app.ingestion.models import ...`). `backend/__init__.py` and `backend/app/__init__.py` exist, so `backend.app` is the ONE canonical package path — never `from app...` (that double-imports and breaks pydantic `isinstance`). Inside the `clip_engine` package, use relative imports (`from . import config`, `from ..models import CaptionCue`). The vendor source is at the absolute path `/Users/vincentfeng/Documents/reelai app/practice/clips/backend` (i.e. `../../practice/clips/backend` from repo root).
- **Runtime env keys.** `SUPADATA_API_KEY` / `GEMINI_API_KEY` / `GROQ_API_KEY` are in the gitignored `backend/.env`. `clip_engine/config.py` reads them from `os.environ` and stays a **pure reader (no dotenv load)** so the config tests that manipulate `os.environ` + reload behave deterministically. In prod (Railway) they're real env vars; for local server runs, `backend/.env` must be sourced/exported before launching (a Phase-4 launch-script concern, not this plan's tests). All engine tests are mocked — set `config.*` attributes directly (as the task tests do) or delete the env var + reload; no real keys needed. Note: the test shell must NOT have these keys exported, or the "missing key raises" test will not see them absent.
- **YouTube-only.** Non-YouTube URLs raise `IngestUnsupportedSourceError`; `ingest_search`/`ingest_feed` only ever return YouTube. Copy the URL regex from `practice/clips/backend/schemas.py` (`YT_RE`).
- **Preserve the client contract exactly.** Endpoints keep returning `IngestResult` / `IngestTopicCutResult` / `IngestSearchResult` / `IngestFeedResult` with `ReelOutWithAttribution` reels. Field names stay snake_case. Never rename or drop a `ReelOut` field.
- **Default engine config:** `CLIP_ENGINE=gemini`, `OUTPUT_MODE=embed`, `PRECISE_BOUNDARIES=0`, `TRANSCRIBER=supadata`, `LLM_PROVIDER=gemini`, `SEGMENT_FINE_SNAP=1`. No local ML, no video download.
- **No paid-key at import time.** Missing `SUPADATA_API_KEY` / `GEMINI_API_KEY` must fail at call time with a clear error, never at module import — module import must stay key-free so tests and serverless import cleanly.
- **Heavy imports are lazy.** `torch` / `faster_whisper` / `sentence_transformers` / `yt_dlp` must NOT be imported on the `gemini`/`embed` path. Guard any such import inside the function that needs it.
- **Serverless guard stays.** `/api/ingest/*` remain disabled when `SERVERLESS_MODE` (unchanged handler code).
- **Money guardrails:** every Supadata search/transcript call routes through the existing `search_cache` / `transcript_cache` tables; `CLIP_SEARCH_MAX_VIDEOS` (default 5) caps videos clipped per search request.

---

## File Structure

**New package `backend/app/clip_engine/`:**
- `__init__.py` — public exports: `discover`, `clip`, `to_reel_out`, `EngineError`.
- `config.py` — engine config bridged from `backend/app/config.py` + env; the `gemini`/`embed` defaults above.
- `errors.py` — `EngineError`, `SearchError`, `TranscriptError`, `ClipError`, `UnsupportedURLError`.
- `supadata_search.py` — Supadata YouTube Search client (port of `practice/lib/supadata.js`).
- `expand.py` — topic → `{corrected, queries[]}` (Gemini + free fallback; port of `practice/lib/expand.js` + the free path).
- `rank.py` — merge/dedupe/rank across queries (port of `practice/lib/rank.js`).
- `search.py` — `discover(topic, ...)` orchestrating expand → search → rank → exclude.
- `metadata.py` — `youtube_metadata(video_id)` via `yt_dlp.extract_info(download=False)` (lazy import) for single-URL flows.
- `run.py` — `clip(url, topic, settings)` inline runner (transcribe → `segment_clips` → embed clip dicts).
- `adapter.py` — `to_reel_out(...)` / `to_metadata(...)` mapping clip dicts → `ReelOutWithAttribution` / `IngestMetadata`.
- `clipper/` — vendored `practice/clips/backend/` gemini-path modules (see Task 6).

**Modified:**
- `backend/app/ingestion/pipeline.py` — reimplement `ingest_url` / `ingest_topic_cut` / `ingest_search` / `ingest_feed` bodies to call `clip_engine`.
- `backend/app/config.py` — add `clip_engine`, `supadata_api_key`, `supadata_base`, `gemini_model`, `segment_model`, `clip_search_max_videos`.
- `backend/requirements.txt` / `pyproject.toml` — add `google-genai>=1.0`, `tiktoken`, `rapidfuzz` (light-path deps only).

**Tests:** `backend/tests/clip_engine/test_*.py` mirroring each module.

---

## Task 1: Scaffold `clip_engine` package + config bridge

**Files:**
- Create: `backend/app/clip_engine/__init__.py`, `backend/app/clip_engine/config.py`, `backend/app/clip_engine/errors.py`
- Test: `backend/tests/clip_engine/test_config.py`

**Interfaces:**
- Produces: `clip_engine.config` module with attrs `SUPADATA_API_KEY`, `SUPADATA_BASE`, `SUPADATA_SEARCH_URL`, `GEMINI_MODEL`, `SEGMENT_MODEL`, `CLIP_ENGINE`, `OUTPUT_MODE`, `PRECISE_BOUNDARIES`, `SEGMENT_FINE_SNAP`, `SEGMENT_MIN_CLIP_S`, `SEGMENT_MAX_CLIPS`, `SEGMENT_MAX_OUTPUT_TOKENS`, `CLIP_SEARCH_MAX_VIDEOS`, `SEARCH_BREADTH`; helper `require_supadata_key() -> str`, `require_gemini_key() -> str`.
- Produces: `clip_engine.errors.EngineError` (base) and subclasses `SearchError`, `TranscriptError`, `ClipError`, `UnsupportedURLError`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/clip_engine/test_config.py
import importlib
import pytest


def test_defaults_are_gemini_embed(monkeypatch):
    monkeypatch.delenv("CLIP_ENGINE", raising=False)
    monkeypatch.delenv("PRECISE_BOUNDARIES", raising=False)
    cfg = importlib.reload(importlib.import_module("backend.app.clip_engine.config"))
    assert cfg.CLIP_ENGINE == "gemini"
    assert cfg.OUTPUT_MODE == "embed"
    assert cfg.PRECISE_BOUNDARIES is False
    assert cfg.SEGMENT_FINE_SNAP is True
    assert cfg.CLIP_SEARCH_MAX_VIDEOS == 5


def test_require_supadata_key_raises_when_missing(monkeypatch):
    monkeypatch.delenv("SUPADATA_API_KEY", raising=False)
    cfg = importlib.reload(importlib.import_module("backend.app.clip_engine.config"))
    from backend.app.clip_engine.errors import SearchError
    with pytest.raises(SearchError):
        cfg.require_supadata_key()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `backend/.venv/bin/python -m pytest backend/tests/clip_engine/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: app.clip_engine.config`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/app/clip_engine/errors.py
class EngineError(Exception):
    """Base error for the clip engine."""


class SearchError(EngineError):
    pass


class TranscriptError(EngineError):
    pass


class ClipError(EngineError):
    pass


class UnsupportedURLError(EngineError):
    pass
```

```python
# backend/app/clip_engine/config.py
"""Clip-engine config, bridged from env with practice-folder defaults.
Import stays key-free: missing keys raise only when a call needs them.
"""
from __future__ import annotations

import os

from .errors import SearchError, ClipError


def _flag(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() not in ("0", "false", "no", "")


SUPADATA_API_KEY = os.environ.get("SUPADATA_API_KEY", "")
SUPADATA_BASE = os.environ.get("SUPADATA_BASE", "https://api.supadata.ai/v1")
SUPADATA_SEARCH_URL = f"{SUPADATA_BASE}/youtube/search"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
SEGMENT_MODEL = os.environ.get("SEGMENT_MODEL", GEMINI_MODEL)

CLIP_ENGINE = os.environ.get("CLIP_ENGINE", "gemini")
OUTPUT_MODE = os.environ.get("OUTPUT_MODE", "embed")
PRECISE_BOUNDARIES = _flag("PRECISE_BOUNDARIES", False)
SEGMENT_FINE_SNAP = _flag("SEGMENT_FINE_SNAP", True)

SEGMENT_MIN_CLIP_S = float(os.environ.get("SEGMENT_MIN_CLIP_S", "15"))
SEGMENT_MAX_CLIPS = int(os.environ.get("SEGMENT_MAX_CLIPS", "40"))
SEGMENT_MAX_OUTPUT_TOKENS = int(os.environ.get("SEGMENT_MAX_OUTPUT_TOKENS", "24576"))
TAIL_PAD_S = float(os.environ.get("SEGMENT_TAIL_PAD_S", "0.15"))

CLIP_SEARCH_MAX_VIDEOS = int(os.environ.get("CLIP_SEARCH_MAX_VIDEOS", "5"))
SEARCH_BREADTH = int(os.environ.get("CLIP_SEARCH_BREADTH", "5"))


def require_supadata_key() -> str:
    if not SUPADATA_API_KEY:
        raise SearchError("SUPADATA_API_KEY is not set.")
    return SUPADATA_API_KEY


def require_gemini_key() -> str:
    if not GEMINI_API_KEY:
        raise ClipError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set.")
    return GEMINI_API_KEY
```

```python
# backend/app/clip_engine/__init__.py
from .errors import EngineError, SearchError, TranscriptError, ClipError, UnsupportedURLError

__all__ = ["EngineError", "SearchError", "TranscriptError", "ClipError", "UnsupportedURLError"]
```

Also create empty `backend/tests/clip_engine/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `backend/.venv/bin/python -m pytest backend/tests/clip_engine/test_config.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add backend/app/clip_engine/__init__.py backend/app/clip_engine/config.py backend/app/clip_engine/errors.py backend/tests/clip_engine/
git commit -m "feat(clip_engine): scaffold package + config bridge"
```

---

## Task 2: Supadata YouTube Search client

**Files:**
- Create: `backend/app/clip_engine/supadata_search.py`
- Test: `backend/tests/clip_engine/test_supadata_search.py`

**Interfaces:**
- Consumes: `config.SUPADATA_SEARCH_URL`, `config.require_supadata_key()`.
- Produces: `search_one(query: str, filters: dict | None = None) -> dict` returning `{"query": str, "videos": list[dict], "billed": int}` where each video is the raw Supadata result dict (`id`, `title`, `channel`, `thumbnail`, `duration`, `viewCount`, `uploadDate`). `search_all(queries: list[str], filters: dict | None = None) -> dict` returning `{"per_query": list[dict], "credits_used": int, "warning": str | None}`. Raises `SearchError` on non-2xx (except 429, retried).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/clip_engine/test_supadata_search.py
import pytest
from backend.app.clip_engine import supadata_search as ss
from backend.app.clip_engine.errors import SearchError


class _Resp:
    def __init__(self, status, payload, headers=None):
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}
    def json(self):
        return self._payload


def test_search_one_returns_videos(monkeypatch):
    monkeypatch.setattr(ss.config, "SUPADATA_API_KEY", "sd_test")
    calls = {}
    def fake_get(url, headers=None, params=None, timeout=None):
        calls["headers"] = headers
        return _Resp(200, {"results": [
            {"id": "abc", "title": "T", "type": "video"},
            {"id": "def", "type": "channel"},
        ]}, {"x-billable-requests": "1"})
    monkeypatch.setattr(ss.httpx, "get", fake_get)
    out = ss.search_one("calculus")
    assert calls["headers"]["x-api-key"] == "sd_test"
    assert [v["id"] for v in out["videos"]] == ["abc"]
    assert out["billed"] == 1


def test_search_all_aggregates_and_warns_on_402(monkeypatch):
    monkeypatch.setattr(ss.config, "SUPADATA_API_KEY", "sd_test")
    seq = iter([
        _Resp(200, {"results": [{"id": "a", "type": "video"}]}, {"x-billable-requests": "1"}),
        _Resp(402, {"message": "out of credits"}, {"x-billable-requests": "0"}),
    ])
    monkeypatch.setattr(ss.httpx, "get", lambda *a, **k: next(seq))
    monkeypatch.setattr(ss.time, "sleep", lambda *_: None)
    out = ss.search_all(["a", "b"])
    assert out["credits_used"] == 1
    assert "out of Supadata credits" in out["warning"]
    assert len(out["per_query"]) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `backend/.venv/bin/python -m pytest backend/tests/clip_engine/test_supadata_search.py -v`
Expected: FAIL — `ModuleNotFoundError: app.clip_engine.supadata_search`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/app/clip_engine/supadata_search.py
"""Supadata YouTube Search client — Python port of practice/lib/supadata.js.
One query = one page (~20 results) = 1 credit. Sequential with 429 backoff.
"""
from __future__ import annotations

import time

import httpx

from . import config
from .errors import SearchError

_MAX_RETRIES = 3


def search_one(query: str, filters: dict | None = None) -> dict:
    key = config.require_supadata_key()
    filters = filters or {}
    params = {"query": query, "type": "video"}
    if filters.get("sortBy") and filters["sortBy"] != "relevance":
        params["sortBy"] = filters["sortBy"]
    if filters.get("uploadDate") and filters["uploadDate"] != "all":
        params["uploadDate"] = filters["uploadDate"]
    if filters.get("duration") and filters["duration"] != "all":
        params["duration"] = filters["duration"]

    attempt = 0
    while True:
        r = httpx.get(config.SUPADATA_SEARCH_URL, headers={"x-api-key": key},
                      params=params, timeout=30.0)
        billed = int(r.headers.get("x-billable-requests") or 0)
        if r.status_code == 429 and attempt < _MAX_RETRIES:
            retry_after = float(r.headers.get("retry-after") or 0) or 1.2 * (attempt + 1)
            time.sleep(retry_after)
            attempt += 1
            continue
        if r.status_code >= 400:
            detail = ""
            try:
                detail = r.json().get("message", "")
            except Exception:
                detail = ""
            err = SearchError(f"Supadata {r.status_code}{': ' + detail if detail else ''}")
            err.status = r.status_code  # type: ignore[attr-defined]
            err.billed = billed  # type: ignore[attr-defined]
            raise err
        data = r.json()
        results = data.get("results") if isinstance(data, dict) else None
        results = results if isinstance(results, list) else []
        videos = [it for it in results if (it.get("type") == "video" if it.get("type") else True)]
        return {"query": query, "videos": videos, "billed": billed or 1}


def search_all(queries: list[str], filters: dict | None = None) -> dict:
    credits_used = 0
    per_query: list[dict] = []
    errors: list[dict] = []
    for i, q in enumerate(queries):
        try:
            res = search_one(q, filters)
            credits_used += res["billed"]
            per_query.append(res)
        except SearchError as e:
            credits_used += getattr(e, "billed", 0) or 0
            status = getattr(e, "status", None)
            errors.append({"query": q, "status": status, "message": str(e)})
            per_query.append({"query": q, "videos": [], "billed": getattr(e, "billed", 0) or 0,
                              "error": str(e), "status": status})
        if i < len(queries) - 1:
            time.sleep(0.25)

    warning = None
    if errors:
        out_of_credits = any(e["status"] == 402 for e in errors)
        rate_limited = any(e["status"] == 429 for e in errors)
        reason = (" (out of Supadata credits)" if out_of_credits
                  else " (rate limited)" if rate_limited else "")
        warning = f"{len(errors)} of {len(queries)} searches failed{reason}. Showing partial results."
    return {"per_query": per_query, "credits_used": credits_used, "warning": warning}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `backend/.venv/bin/python -m pytest backend/tests/clip_engine/test_supadata_search.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add backend/app/clip_engine/supadata_search.py backend/tests/clip_engine/test_supadata_search.py
git commit -m "feat(clip_engine): Supadata YouTube search client"
```

---

## Task 3: Topic expansion (`expand.py`)

**Files:**
- Create: `backend/app/clip_engine/expand.py`
- Test: `backend/tests/clip_engine/test_expand.py`

**Interfaces:**
- Consumes: `config.GEMINI_MODEL`, `config.require_gemini_key()`, and (lazily) `google.genai`.
- Produces: `expand_query(topic: str, n: int) -> dict` → `{"corrected": str, "queries": list[str], "provider_used": str}`. Queries are deduped (case-insensitive), corrected topic first, length ≤ n. Gemini failure or no key → deterministic free fallback `free_expand(topic, n)` which returns `[topic]` plus simple `"{topic} explained"`, `"{topic} tutorial"`, `"{topic} for beginners"` variants (no network) so search still works keyless.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/clip_engine/test_expand.py
from backend.app.clip_engine import expand


def test_free_fallback_when_no_key(monkeypatch):
    monkeypatch.setattr(expand.config, "GEMINI_API_KEY", "")
    out = expand.expand_query("calculus", 4)
    assert out["provider_used"] == "free"
    assert out["queries"][0] == "calculus"
    assert len(out["queries"]) <= 4
    assert len(set(q.lower() for q in out["queries"])) == len(out["queries"])  # deduped


def test_gemini_path_parses_json(monkeypatch):
    monkeypatch.setattr(expand.config, "GEMINI_API_KEY", "g_test")
    monkeypatch.setattr(
        expand, "_gemini_expand_raw",
        lambda system, user, model: '{"corrected": "calculus", "queries": ["calculus", "derivatives", "integrals"]}',
    )
    out = expand.expand_query("calculas", 5)
    assert out["provider_used"] == "gemini"
    assert out["corrected"] == "calculus"
    assert out["queries"][:2] == ["calculus", "derivatives"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `backend/.venv/bin/python -m pytest backend/tests/clip_engine/test_expand.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/app/clip_engine/expand.py
"""Topic → diverse YouTube search queries. Gemini when keyed, else a keyless
deterministic fallback. Port of practice/lib/expand.js (LLM path + free path).
"""
from __future__ import annotations

import json
import re

from . import config

_SYSTEM = (
    "You expand a user's search topic into a diverse set of YouTube search queries that "
    "maximize topical coverage. Spellcheck and correct the input, infer intent, then produce "
    "up to N distinct queries covering the corrected topic, close synonyms, important "
    "sub-topics, and phrase variants (\"X tutorial\", \"X explained\", \"X for beginners\"). "
    "Return ONLY strict JSON: {\"corrected\": \"...\", \"queries\": [\"q1\", ...]} with the "
    "corrected topic first in queries. No prose, no code fences."
)


def _user(topic: str, n: int) -> str:
    return (f"User topic: {json.dumps(topic)}\nN = {n}\n"
            f"Return JSON with \"corrected\" and \"queries\" (at most {n}, corrected first). JSON only.")


def _safe_json(text: str) -> dict | None:
    if not text:
        return None
    t = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", t, re.IGNORECASE)
    if m:
        t = m.group(1).strip()
    a, b = t.find("{"), t.rfind("}")
    if a == -1 or b == -1 or b < a:
        return None
    try:
        return json.loads(t[a:b + 1])
    except Exception:
        return None


def _normalize(corrected: str | None, queries, fallback: str, n: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    def push(q):
        if not q:
            return
        s = str(q).strip()
        if not s or s.lower() in seen:
            return
        seen.add(s.lower())
        out.append(s)

    push(corrected or fallback)
    for q in (queries or []):
        push(q)
    if not out:
        push(fallback)
    return out[:n]


def free_expand(topic: str, n: int) -> dict:
    variants = [topic, f"{topic} explained", f"{topic} tutorial", f"{topic} for beginners"]
    return {"corrected": topic, "queries": _normalize(topic, variants, topic, n),
            "provider_used": "free"}


def _gemini_expand_raw(system: str, user: str, model: str) -> str:
    from google import genai  # lazy import
    client = genai.Client(api_key=config.require_gemini_key())
    resp = client.models.generate_content(
        model=model,
        contents=[{"role": "user", "parts": [{"text": system + "\n\n" + user}]}],
        config={"response_mime_type": "application/json", "temperature": 0.2},
    )
    return getattr(resp, "text", "") or ""


def expand_query(topic: str, n: int) -> dict:
    topic = topic.strip()
    if not config.GEMINI_API_KEY:
        return free_expand(topic, n)
    try:
        raw = _gemini_expand_raw(_SYSTEM, _user(topic, n), config.GEMINI_MODEL)
        parsed = _safe_json(raw)
        if parsed:
            return {"corrected": parsed.get("corrected") or topic,
                    "queries": _normalize(parsed.get("corrected"), parsed.get("queries"), topic, n),
                    "provider_used": "gemini"}
    except Exception:
        pass
    return free_expand(topic, n)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `backend/.venv/bin/python -m pytest backend/tests/clip_engine/test_expand.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add backend/app/clip_engine/expand.py backend/tests/clip_engine/test_expand.py
git commit -m "feat(clip_engine): topic expansion (Gemini + keyless fallback)"
```

---

## Task 4: Merge + rank (`rank.py`)

**Files:**
- Create: `backend/app/clip_engine/rank.py`
- Test: `backend/tests/clip_engine/test_rank.py`

**Interfaces:**
- Produces: `merge_and_rank(per_query: list[dict]) -> list[dict]` → ranked list of `{"id", "title", "channel", "thumbnail", "duration", "view_count", "upload_date", "url", "match_count", "score", "matched_queries"}`, sorted by `match_count` desc, then `score` desc, then `view_count` desc. Port of `practice/lib/rank.js`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/clip_engine/test_rank.py
from backend.app.clip_engine.rank import merge_and_rank


def test_ranks_by_match_count_then_views():
    per_query = [
        {"query": "a", "videos": [{"id": "x", "title": "X", "viewCount": 100},
                                  {"id": "y", "title": "Y", "viewCount": 999}]},
        {"query": "b", "videos": [{"id": "x", "title": "X", "viewCount": 100}]},
    ]
    ranked = merge_and_rank(per_query)
    assert ranked[0]["id"] == "x"            # match_count 2 beats y's 1
    assert ranked[0]["match_count"] == 2
    assert ranked[0]["url"] == "https://www.youtube.com/watch?v=x"
    assert sorted(ranked[0]["matched_queries"]) == ["a", "b"]


def test_skips_videos_without_id():
    ranked = merge_and_rank([{"query": "a", "videos": [{"title": "no id"}]}])
    assert ranked == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `backend/.venv/bin/python -m pytest backend/tests/clip_engine/test_rank.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/app/clip_engine/rank.py
"""Merge results across expanded queries, dedupe by video id, rank.
Strongest signal = match_count (how many queries surfaced the video).
Port of practice/lib/rank.js.
"""
from __future__ import annotations

import math


def _channel_name(v: dict) -> str:
    ch = v.get("channel")
    if isinstance(ch, dict):
        return ch.get("name") or ch.get("title") or ""
    return ch or ""


def merge_and_rank(per_query: list[dict]) -> list[dict]:
    by_id: dict[str, dict] = {}
    for res in per_query or []:
        for rank, v in enumerate(res.get("videos") or []):
            vid = v.get("id")
            if not vid:
                continue
            entry = by_id.get(vid)
            if entry is None:
                vc = v.get("viewCount")
                entry = {
                    "id": vid,
                    "title": v.get("title") or "(untitled)",
                    "channel": _channel_name(v),
                    "thumbnail": v.get("thumbnail") or "",
                    "duration": v.get("duration") if isinstance(v.get("duration"), (int, float)) else None,
                    "view_count": vc if isinstance(vc, (int, float)) else (int(vc) if str(vc or "").isdigit() else 0),
                    "upload_date": v.get("uploadDate") or "",
                    "url": f"https://www.youtube.com/watch?v={vid}",
                    "match_count": 0,
                    "best_rank": rank,
                    "matched_queries": [],
                }
                by_id[vid] = entry
            entry["match_count"] += 1
            entry["best_rank"] = min(entry["best_rank"], rank)
            q = res.get("query")
            if q and q not in entry["matched_queries"]:
                entry["matched_queries"].append(q)

    items = list(by_id.values())
    for v in items:
        view_score = math.log10((v["view_count"] or 0) + 10)
        rank_score = 1 / (1 + v["best_rank"])
        v["score"] = v["match_count"] * 10 + view_score + rank_score * 2
    items.sort(key=lambda v: (v["match_count"], v["score"], v["view_count"]), reverse=True)
    for v in items:
        v.pop("best_rank", None)
    return items
```

- [ ] **Step 4: Run test to verify it passes**

Run: `backend/.venv/bin/python -m pytest backend/tests/clip_engine/test_rank.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add backend/app/clip_engine/rank.py backend/tests/clip_engine/test_rank.py
git commit -m "feat(clip_engine): merge + rank search results"
```

---

## Task 5: Discovery orchestrator (`search.py`)

**Files:**
- Create: `backend/app/clip_engine/search.py`
- Test: `backend/tests/clip_engine/test_search.py`

**Interfaces:**
- Consumes: `expand.expand_query`, `supadata_search.search_all`, `rank.merge_and_rank`, `config.SEARCH_BREADTH`.
- Produces: `discover(topic: str, limit: int, exclude_video_ids: list[str] | None = None, breadth: int | None = None) -> dict` → `{"corrected": str, "videos": list[dict], "credits_used": int, "warning": str | None}` where `videos` are ranked video dicts (from `merge_and_rank`) with excluded ids removed and truncated to `limit`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/clip_engine/test_search.py
from backend.app.clip_engine import search


def test_discover_excludes_and_limits(monkeypatch):
    monkeypatch.setattr(search.expand, "expand_query",
                        lambda t, n: {"corrected": "calc", "queries": ["calc"], "provider_used": "free"})
    monkeypatch.setattr(search.supadata_search, "search_all",
                        lambda queries, filters=None: {"per_query": [
                            {"query": "calc", "videos": [{"id": "a"}, {"id": "b"}, {"id": "c"}]}],
                            "credits_used": 1, "warning": None})
    out = search.discover("calc", limit=2, exclude_video_ids=["a"])
    assert [v["id"] for v in out["videos"]] == ["b", "c"]
    assert out["credits_used"] == 1
    assert out["corrected"] == "calc"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `backend/.venv/bin/python -m pytest backend/tests/clip_engine/test_search.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/app/clip_engine/search.py
"""Discovery: topic -> expand -> Supadata search -> rank -> exclude -> top N."""
from __future__ import annotations

from . import config, expand, rank, supadata_search


def discover(topic: str, limit: int, exclude_video_ids: list[str] | None = None,
             breadth: int | None = None) -> dict:
    n = max(1, breadth or config.SEARCH_BREADTH)
    expansion = expand.expand_query(topic, n)
    res = supadata_search.search_all(expansion["queries"])
    ranked = rank.merge_and_rank(res["per_query"])
    exclude = set(exclude_video_ids or [])
    videos = [v for v in ranked if v["id"] not in exclude][:limit]
    return {"corrected": expansion["corrected"], "videos": videos,
            "credits_used": res["credits_used"], "warning": res["warning"]}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `backend/.venv/bin/python -m pytest backend/tests/clip_engine/test_search.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/clip_engine/search.py backend/tests/clip_engine/test_search.py
git commit -m "feat(clip_engine): discovery orchestrator"
```

---

## Task 6: Vendor the practice clipper (gemini path)

**Files:**
- Create: `backend/app/clip_engine/clipper/` (vendored subset of `practice/clips/backend/`)
- Test: `backend/tests/clip_engine/test_clipper_import.py`

**Interfaces:**
- Produces (from vendored code, imports adjusted to the new package): `clipper.pipeline.gemini_segment.segment_clips(transcript: dict, settings: dict, progress=None) -> tuple[list[dict], str]`; `clipper.pipeline.transcribe.transcribe_supadata(...)`; `clipper.embed.embed_url(video_id, start, end) -> str`; `clipper.llm.llm_json(...)`.

- [ ] **Step 1: Copy the gemini-path modules**

Copy these files from `practice/clips/backend/` into `backend/app/clip_engine/clipper/`, preserving the sub-paths:
`__init__.py`, `config.py`, `errors.py`, `llm.py`, `embed.py`, `gemini_client.py`, `supadata_client.py`, `groq_client.py`,
`pipeline/__init__.py`, `pipeline/transcribe.py`, `pipeline/gemini_segment.py`, `pipeline/sentences.py`.
(`groq_client.py` is included because `transcribe.py` references it; its `groq` import is made lazy in Step 3 so the `groq` package is NOT required on the default path.)

```bash
# Run from the repo root: reelai/reelAI copy 2
SRC="/Users/vincentfeng/Documents/reelai app/practice/clips/backend"
DST="backend/app/clip_engine/clipper"
mkdir -p "$DST/pipeline"
for f in __init__.py config.py errors.py llm.py embed.py gemini_client.py supadata_client.py; do cp "$SRC/$f" "$DST/$f"; done
for f in __init__.py transcribe.py gemini_segment.py sentences.py; do cp "$SRC/pipeline/$f" "$DST/pipeline/$f"; done
```

- [ ] **Step 2: Rewrite the vendored `clipper/config.py`**

Replace the vendored `clipper/config.py` with a thin shim that re-exports the engine config so the vendored modules keep using `from .. import config` / `config.SEGMENT_MODEL` etc. unchanged:

```python
# backend/app/clip_engine/clipper/config.py
"""Vendored clipper config shim → delegates to app.clip_engine.config.
Adds the extra constants the vendored gemini path reads that aren't in the
engine config, with practice defaults. NO .env file load, NO paths/mkdir.
"""
from __future__ import annotations

from .. import config as _engine

SEGMENT_MODEL = _engine.SEGMENT_MODEL
SEGMENT_FINE_SNAP = _engine.SEGMENT_FINE_SNAP
SEGMENT_MIN_CLIP_S = _engine.SEGMENT_MIN_CLIP_S
SEGMENT_MAX_CLIPS = _engine.SEGMENT_MAX_CLIPS
SEGMENT_MAX_OUTPUT_TOKENS = _engine.SEGMENT_MAX_OUTPUT_TOKENS
GEMINI_MODEL = _engine.GEMINI_MODEL
GEMINI_API_KEY = _engine.GEMINI_API_KEY
LLM_PROVIDER = "gemini"
CHARS_PER_TOKEN = 4
DEFAULTS = {"tail_pad_s": _engine.TAIL_PAD_S, "lead_pad_s": 0.06}
# Supadata transcript
SUPADATA_API_KEY = _engine.SUPADATA_API_KEY
SUPADATA_BASE = _engine.SUPADATA_BASE
SUPADATA_CHUNK_SIZE = 180
```

- [ ] **Step 3: Make heavy imports lazy + write the import smoke test**

Import-graph facts (verified against the source) and the exact lazy-import edits required:
- `embed.py` imports only `math`; `gemini_segment.py` imports `config` (shim) + `..llm` and lazily `rapidfuzz` (inside `_locate_quote`); `llm.py` imports `config` + `pydantic` at module level and lazily imports `gemini_client` / `groq_client` / `.pipeline.select` *inside* functions (already guarded). So importing `gemini_segment` + `embed` (the Task 6 smoke test) pulls NO heavy deps — good, do not change these.
- `gemini_client.py` imports `from google import genai` at module level. It is only reached lazily via `llm.py`, so the smoke test never triggers it; `google-genai` is added to deps in Task 13. Leave it.
- **Required edit — `clipper/pipeline/transcribe.py`:** move the module-level `from ..groq_client import transcribe_audio` (≈line 17) INTO the Groq/whisper function that actually uses it, so importing `transcribe` does not require Groq. `faster_whisper` is already lazy in `_get_whisper` — confirm and leave.
- **Required edit — `clipper/groq_client.py`:** move the module-level `groq` SDK import (`import groq` / `from groq import ...`) INSIDE the functions that use it, so `import ...clipper.groq_client` succeeds without the `groq` package installed (the default `TRANSCRIBER=supadata` + `LLM_PROVIDER=gemini` path never calls Groq). Do NOT add `groq` to requirements.
- Do not otherwise modify the vendored logic. If any *other* module-level import pulls `torch` / `sentence_transformers` / `yt_dlp` on the imported path, make it lazy the same way and note it in your report.

```python
# backend/tests/clip_engine/test_clipper_import.py
import sys


def test_gemini_path_imports_without_torch():
    # Importing the gemini path must not pull torch / faster_whisper / yt_dlp.
    import backend.app.clip_engine.clipper.pipeline.gemini_segment as gs
    import backend.app.clip_engine.clipper.embed as embed
    assert callable(gs.segment_clips)
    assert embed.embed_url("vid123", 10.4, 42.9) == "https://www.youtube.com/embed/vid123?start=10&end=43&rel=0"
    assert "torch" not in sys.modules
    assert "faster_whisper" not in sys.modules
```

- [ ] **Step 4: Run test to verify it passes**

Run: `backend/.venv/bin/python -m pytest backend/tests/clip_engine/test_clipper_import.py -v`
Expected: PASS. (If it fails on a heavy import, trace the import chain and make that import lazy — do not add torch to requirements.)

- [ ] **Step 5: Commit**

```bash
git add backend/app/clip_engine/clipper backend/tests/clip_engine/test_clipper_import.py
git commit -m "feat(clip_engine): vendor practice clipper gemini path (lazy heavy imports)"
```

---

## Task 7: Inline clip runner (`run.py`)

**Files:**
- Create: `backend/app/clip_engine/run.py`
- Test: `backend/tests/clip_engine/test_run.py`

**Interfaces:**
- Consumes: `clipper.pipeline.transcribe.transcribe_supadata`, `clipper.pipeline.gemini_segment.segment_clips`, `clipper.embed.embed_url`, `metadata.extract_video_id`.
- Produces: `clip(url: str, topic: str, settings: dict | None = None) -> dict` → `{"video_id": str, "clips": list[dict], "transcript": dict, "notes": str}`. Each clip dict has `start`, `end`, `cut_end`, `title`, `facet`, `reason`, `sequence_index`, `embed_url`. Raises `UnsupportedURLError` (non-YouTube), `TranscriptError`, `ClipError`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/clip_engine/test_run.py
import pytest
from backend.app.clip_engine import run
from backend.app.clip_engine.errors import UnsupportedURLError


def test_rejects_non_youtube():
    with pytest.raises(UnsupportedURLError):
        run.clip("https://vimeo.com/123", "topic")


def test_clip_builds_embed_urls(monkeypatch):
    fake_tx = {"segments": [{"start": 0.0, "end": 5.0, "text": "hello world"}],
               "words": [], "duration": 5.0}
    monkeypatch.setattr(run, "_transcribe", lambda url, video_id, settings: fake_tx)
    monkeypatch.setattr(run.gemini_segment, "segment_clips",
                        lambda transcript, settings, progress=None: (
                            [{"start": 1.0, "end": 4.0, "cut_end": 4.15, "title": "Bit",
                              "facet": "concept", "reason": "", "sequence_index": 1}], "1 clip"))
    out = run.clip("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "topic")
    assert out["video_id"] == "dQw4w9WgXcQ"
    assert out["clips"][0]["embed_url"] == "https://www.youtube.com/embed/dQw4w9WgXcQ?start=1&end=4&rel=0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `backend/.venv/bin/python -m pytest backend/tests/clip_engine/test_run.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/app/clip_engine/run.py
"""Inline clip runner — replaces the practice clipper's job/SSE surface with a
plain function. Supadata transcript -> one Gemini pass -> embed clip specs.
"""
from __future__ import annotations

from . import config
from .clipper import embed
from .clipper.pipeline import gemini_segment
from .errors import ClipError, TranscriptError, UnsupportedURLError
from .metadata import extract_video_id


def _transcribe(url: str, video_id: str, settings: dict) -> dict:
    """Fetch a timestamped Supadata transcript as {segments, words, duration, ...}."""
    from .clipper.pipeline.transcribe import transcribe_supadata  # lazy
    try:
        return transcribe_supadata(url, video_id, settings)
    except Exception as exc:  # normalize to engine error
        raise TranscriptError(f"Supadata transcript failed for {video_id}: {exc}") from exc


def clip(url: str, topic: str, settings: dict | None = None) -> dict:
    video_id = extract_video_id(url)
    if not video_id:
        raise UnsupportedURLError(f"Not a recognized YouTube URL: {url}")
    settings = dict(settings or {})
    settings.setdefault("segment_model", config.SEGMENT_MODEL)
    settings.setdefault("segment_fine_snap", config.SEGMENT_FINE_SNAP)
    settings.setdefault("segment_min_clip_s", config.SEGMENT_MIN_CLIP_S)

    transcript = _transcribe(url, video_id, settings)
    if not (transcript.get("segments")):
        raise TranscriptError(f"Empty transcript for {video_id}")

    try:
        clips, notes = gemini_segment.segment_clips(transcript, settings)
    except Exception as exc:
        raise ClipError(f"Gemini segmentation failed for {video_id}: {exc}") from exc

    for c in clips:
        # Embed end uses c["end"] (the semantic clip end), NOT cut_end (which adds
        # tail_pad for the ffmpeg cut path we don't use in embed mode). This keeps the
        # embed window consistent with the adapter's t_end.
        c["embed_url"] = embed.embed_url(video_id, c["start"], c["end"])
    return {"video_id": video_id, "clips": clips, "transcript": transcript, "notes": notes}
```

Also add `extract_video_id` and `youtube_metadata` to `metadata.py`:

```python
# backend/app/clip_engine/metadata.py
"""YouTube id extraction + lightweight (no-download) metadata fetch."""
from __future__ import annotations

import re

_YT_ID = re.compile(
    r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/|youtube\.com/embed/|"
    r"youtube\.com/live/|m\.youtube\.com/watch\?v=)([A-Za-z0-9_-]{11})",
    re.IGNORECASE,
)


def extract_video_id(url: str) -> str | None:
    if not url:
        return None
    m = _YT_ID.search(url)
    return m.group(1) if m else None


def youtube_metadata(video_id: str) -> dict:
    """Title/author/duration/thumbnail via yt-dlp (metadata only, no download).
    Lazy import; returns {} on failure (callers fall back to transcript/embed data).
    """
    try:
        import yt_dlp  # lazy — not on the hot path when search already provides metadata
        opts = {"quiet": True, "skip_download": True, "extract_flat": False, "noplaylist": True}
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
        return {
            "title": info.get("title") or "",
            "author_name": info.get("uploader") or info.get("channel") or "",
            "author_url": info.get("uploader_url") or info.get("channel_url") or "",
            "duration_sec": float(info.get("duration")) if info.get("duration") else None,
            "thumbnail_url": info.get("thumbnail") or "",
            "view_count": info.get("view_count"),
            "upload_date_iso": info.get("upload_date") or None,
            "description": info.get("description") or "",
        }
    except Exception:
        return {}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `backend/.venv/bin/python -m pytest backend/tests/clip_engine/test_run.py -v`
Expected: PASS.

> **Confirmed signature (verified in the vendored code):** `transcribe_supadata(url: str, video_id: str, settings: dict, progress=None) -> dict` returning `{text, duration, words, segments, source, chunks}`. It caches to `config.WORK_DIR/video_id/transcript.json`, so the clipper config shim's `WORK_DIR` must be a writable path at runtime (NOT exercised by this task's mocked test). `_transcribe(url, video_id, settings)` passes the full watch URL, the id, and `settings` (which carries `language`). `segment_clips` needs `segments` (each `{start,end,text}`) + `words` (each `{word,start,end}`).

- [ ] **Step 5: Commit**

```bash
git add backend/app/clip_engine/run.py backend/app/clip_engine/metadata.py backend/tests/clip_engine/test_run.py
git commit -m "feat(clip_engine): inline clip runner + youtube metadata"
```

---

## Task 8: Adapter → `ReelOutWithAttribution`

**Files:**
- Create: `backend/app/clip_engine/adapter.py`
- Test: `backend/tests/clip_engine/test_adapter.py`

**Interfaces:**
- Consumes: `app.models.CaptionCue`, `app.ingestion.models.ReelOutWithAttribution`, `app.ingestion.models.IngestMetadata`.
- Produces:
  - `to_metadata(video_id: str, meta: dict, source_url: str) -> IngestMetadata` (platform `"yt"`).
  - `to_reel_out(clip: dict, transcript: dict, meta: dict, *, reel_id: str, material_id: str, concept_id: str, concept_title: str, video_id: str, source_url: str, position: int | None = None, total: int | None = None) -> ReelOutWithAttribution`.
  - `captions_for_window(transcript: dict, t0: float, t1: float) -> list[CaptionCue]`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/clip_engine/test_adapter.py
from backend.app.clip_engine import adapter


def test_to_reel_out_maps_core_fields():
    clip = {"start": 10.0, "end": 40.0, "cut_end": 40.15, "title": "Chain rule",
            "facet": "concept", "reason": "core idea", "sequence_index": 1,
            "embed_url": "https://www.youtube.com/embed/dQw4w9WgXcQ?start=10&end=41&rel=0"}
    transcript = {"segments": [
        {"start": 9.0, "end": 12.0, "text": "before"},
        {"start": 12.0, "end": 38.0, "text": "the chain rule says"},
        {"start": 41.0, "end": 45.0, "text": "after"}], "words": [], "duration": 600.0}
    meta = {"title": "Calculus 101", "author_name": "MathChan", "duration_sec": 600.0,
            "thumbnail_url": "http://t", "description": "d"}
    reel = adapter.to_reel_out(clip, transcript, meta, reel_id="r1", material_id="m1",
                               concept_id="c1", concept_title="Chain rule", video_id="dQw4w9WgXcQ",
                               source_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    assert reel.video_url == clip["embed_url"]
    assert reel.t_start == 10.0 and reel.t_end == 40.0
    assert reel.video_title == "Calculus 101"
    assert reel.channel_name == "MathChan"
    assert reel.video_duration_sec == 600
    assert reel.clip_duration_sec == 30.0
    assert any("chain rule" in c.text for c in reel.captions)
    assert reel.reel_id == "r1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `backend/.venv/bin/python -m pytest backend/tests/clip_engine/test_adapter.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/app/clip_engine/adapter.py
"""Map engine clip dicts -> the client-facing ReelOut contract."""
from __future__ import annotations

from ..models import CaptionCue
from ..ingestion.models import IngestMetadata, ReelOutWithAttribution


def captions_for_window(transcript: dict, t0: float, t1: float) -> list[CaptionCue]:
    cues: list[CaptionCue] = []
    for s in transcript.get("segments") or []:
        st, en = float(s.get("start", 0.0)), float(s.get("end", 0.0))
        if en < t0 or st > t1:
            continue
        text = (s.get("text") or "").strip()
        if text:
            cues.append(CaptionCue(start=st, end=en, text=text))
    return cues


def _snippet(cues: list[CaptionCue], limit: int = 400) -> str:
    joined = " ".join(c.text for c in cues).strip()
    return joined[:limit]


def to_metadata(video_id: str, meta: dict, source_url: str) -> IngestMetadata:
    playback = f"https://www.youtube.com/embed/{video_id}"
    return IngestMetadata(
        platform="yt", source_id=video_id, source_url=source_url, playback_url=playback,
        title=meta.get("title", ""), description=meta.get("description", ""),
        author_name=meta.get("author_name", ""), author_url=meta.get("author_url", ""),
        duration_sec=meta.get("duration_sec"), thumbnail_url=meta.get("thumbnail_url", ""),
        upload_date_iso=meta.get("upload_date_iso"),
        view_count=int(meta["view_count"]) if str(meta.get("view_count") or "").isdigit() else None,
    )


def to_reel_out(clip: dict, transcript: dict, meta: dict, *, reel_id: str, material_id: str,
                concept_id: str, concept_title: str, video_id: str, source_url: str,
                position: int | None = None, total: int | None = None) -> ReelOutWithAttribution:
    t_start = float(clip["start"])
    t_end = float(clip["end"])
    dur = meta.get("duration_sec")
    if dur:
        t_end = min(t_end, float(dur))
    cues = captions_for_window(transcript, t_start, t_end)
    author = meta.get("author_name", "")
    return ReelOutWithAttribution(
        reel_id=reel_id, material_id=material_id, concept_id=concept_id,
        concept_title=concept_title or (clip.get("title") or ""),
        video_title=meta.get("title", ""), video_description=meta.get("description", ""),
        channel_name=author, ai_summary=(clip.get("reason") or ""),
        video_url=clip["embed_url"], t_start=t_start, t_end=t_end,
        transcript_snippet=_snippet(cues), takeaways=[], captions=cues,
        score=1.0, source_surface="supadata_youtube",
        query_strategy="gemini_segment", retrieval_stage="clip_engine",
        concept_position=position, total_concepts=total,
        video_duration_sec=int(dur) if dur else None,
        clip_duration_sec=round(t_end - t_start, 3),
        source_attribution=(f"{meta.get('title','')} — {author}".strip(" —") or None),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `backend/.venv/bin/python -m pytest backend/tests/clip_engine/test_adapter.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/clip_engine/adapter.py backend/tests/clip_engine/test_adapter.py
git commit -m "feat(clip_engine): adapter to ReelOut contract"
```

---

## Task 9: Rewire `IngestionPipeline.ingest_url`

**Files:**
- Modify: `backend/app/ingestion/pipeline.py` (method `ingest_url`)
- Test: `backend/tests/clip_engine/test_pipeline_ingest_url.py`

**Interfaces:**
- Consumes: `clip_engine.run.clip`, `clip_engine.metadata.youtube_metadata`, `clip_engine.adapter`, existing `persistence.upsert_reel_row`, existing error types.
- Produces: `ingest_url(...) -> IngestResult` unchanged signature/return. Picks the single best clip (highest `sequence_index==1` i.e. first, or longest). Non-YouTube → `IngestUnsupportedSourceError`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/clip_engine/test_pipeline_ingest_url.py
import pytest


def test_ingest_url_returns_ingest_result(monkeypatch, ingestion_pipeline):
    from backend.app.clip_engine import run as engine_run
    monkeypatch.setattr(engine_run, "clip", lambda url, topic, settings=None: {
        "video_id": "dQw4w9WgXcQ",
        "clips": [{"start": 5.0, "end": 35.0, "cut_end": 35.1, "title": "Bit", "facet": "c",
                   "reason": "", "sequence_index": 1,
                   "embed_url": "https://www.youtube.com/embed/dQw4w9WgXcQ?start=5&end=35&rel=0"}],
        "transcript": {"segments": [{"start": 5.0, "end": 35.0, "text": "hi"}], "words": [], "duration": 300.0},
        "notes": "1 clip"})
    from backend.app.clip_engine import metadata
    monkeypatch.setattr(metadata, "youtube_metadata", lambda vid: {"title": "V", "duration_sec": 300.0})

    result = ingestion_pipeline.ingest_url(source_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                                           material_id="m1", concept_id=None,
                                           target_clip_duration_sec=45,
                                           target_clip_duration_min_sec=15,
                                           target_clip_duration_max_sec=60, language="en")
    assert result.reel.video_url.startswith("https://www.youtube.com/embed/dQw4w9WgXcQ")
    assert result.metadata.platform == "yt"
    assert result.reel.t_start == 5.0


def test_ingest_url_rejects_non_youtube(ingestion_pipeline):
    from backend.app.ingestion.errors import IngestUnsupportedSourceError
    with pytest.raises(IngestUnsupportedSourceError):
        ingestion_pipeline.ingest_url(source_url="https://vimeo.com/1", material_id=None,
                                      concept_id=None, target_clip_duration_sec=45,
                                      target_clip_duration_min_sec=15,
                                      target_clip_duration_max_sec=60, language="en")
```

Add a fixture in `backend/tests/clip_engine/conftest.py`:

```python
# backend/tests/clip_engine/conftest.py
import pytest


@pytest.fixture
def ingestion_pipeline():
    from backend.app.ingestion.pipeline import IngestionPipeline
    return IngestionPipeline()  # match the real constructor args used in main.py:288-293
```

- [ ] **Step 2: Run test to verify it fails**

Run: `backend/.venv/bin/python -m pytest backend/tests/clip_engine/test_pipeline_ingest_url.py -v`
Expected: FAIL (old body still runs yt-dlp/segment path or errors on the mock).

- [ ] **Step 3: Rewrite the `ingest_url` body**

Replace the body of `IngestionPipeline.ingest_url` with the engine path. Keep the method signature, the sentinel `material_id` default, `trace_id` generation, `terms_notice`, and `persistence.upsert_reel_row` exactly as the current method does them (read the current method first and reuse its helpers). Core:

```python
# inside IngestionPipeline.ingest_url(...)
from ..clip_engine import run as engine_run
from ..clip_engine import metadata as engine_meta
from ..clip_engine import adapter as engine_adapter
from ..clip_engine.errors import UnsupportedURLError, TranscriptError, ClipError, SearchError
from .errors import (IngestUnsupportedSourceError, IngestTranscriptionError,
                     IngestSegmentationError)

video_id = engine_meta.extract_video_id(source_url)
if not video_id:
    raise IngestUnsupportedSourceError("Only YouTube URLs are supported.")

try:
    engine_out = engine_run.clip(source_url, topic=(concept_id or ""), settings={"language": language})
except UnsupportedURLError as exc:
    raise IngestUnsupportedSourceError(str(exc)) from exc
except TranscriptError as exc:
    raise IngestTranscriptionError(str(exc)) from exc
except ClipError as exc:
    raise IngestSegmentationError(str(exc)) from exc

clips = engine_out["clips"]
if not clips:
    raise IngestSegmentationError("No on-topic clip found.")
# single-clip endpoint: prefer the target-duration-closest clip, else the first
target = target_clip_duration_sec
best = min(clips, key=lambda c: abs((c["end"] - c["start"]) - target))

meta = engine_meta.youtube_metadata(video_id) or {}
if not meta.get("duration_sec"):
    meta["duration_sec"] = engine_out["transcript"].get("duration")

resolved_material_id = material_id or self._scratch_material_id  # reuse existing sentinel
reel_id = self._new_reel_id(video_id, best["start"], best["end"])  # reuse existing id helper
reel = engine_adapter.to_reel_out(best, engine_out["transcript"], meta,
                                  reel_id=reel_id, material_id=resolved_material_id,
                                  concept_id=(concept_id or ""), concept_title="",
                                  video_id=video_id, source_url=source_url)
ingest_meta = engine_adapter.to_metadata(video_id, meta, source_url)
self._persist_reel(reel, ingest_meta)  # reuse existing persistence.upsert_reel_row wrapper

return IngestResult(reel=reel, metadata=ingest_meta,
                    terms_notice=self._terms_notice(), trace_id=self._new_trace_id())
```

> Reuse the pipeline's existing private helpers for the sentinel material id, reel-id generation, persistence, terms notice, and trace id. If those are inlined in the old body rather than helpers, extract them into small private methods as part of this task so this method stays readable.

- [ ] **Step 4: Run test to verify it passes**

Run: `backend/.venv/bin/python -m pytest backend/tests/clip_engine/test_pipeline_ingest_url.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add backend/app/ingestion/pipeline.py backend/tests/clip_engine/test_pipeline_ingest_url.py backend/tests/clip_engine/conftest.py
git commit -m "feat(ingest): route ingest_url through the clip engine (YouTube-only)"
```

---

## Task 10: Rewire `IngestionPipeline.ingest_topic_cut`

**Files:**
- Modify: `backend/app/ingestion/pipeline.py` (method `ingest_topic_cut`)
- Test: `backend/tests/clip_engine/test_pipeline_topic_cut.py`

**Interfaces:**
- Produces: `ingest_topic_cut(...) -> IngestTopicCutResult` — all engine clips become `reels`; `is_short` True when duration < 60s and 0 clips; `reel_count = len(reels)`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/clip_engine/test_pipeline_topic_cut.py
def test_topic_cut_returns_all_clips(monkeypatch, ingestion_pipeline):
    from backend.app.clip_engine import run as engine_run, metadata
    monkeypatch.setattr(engine_run, "clip", lambda url, topic, settings=None: {
        "video_id": "dQw4w9WgXcQ",
        "clips": [
            {"start": 0.0, "end": 30.0, "cut_end": 30.1, "title": "A", "facet": "c", "reason": "",
             "sequence_index": 1, "embed_url": "https://www.youtube.com/embed/dQw4w9WgXcQ?start=0&end=30&rel=0"},
            {"start": 31.0, "end": 70.0, "cut_end": 70.1, "title": "B", "facet": "c", "reason": "",
             "sequence_index": 2, "embed_url": "https://www.youtube.com/embed/dQw4w9WgXcQ?start=31&end=70&rel=0"}],
        "transcript": {"segments": [{"start": 0.0, "end": 70.0, "text": "x"}], "words": [], "duration": 600.0},
        "notes": "2"})
    monkeypatch.setattr(metadata, "youtube_metadata", lambda vid: {"title": "V", "duration_sec": 600.0})
    result = ingestion_pipeline.ingest_topic_cut(source_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                                                 material_id="m1", concept_id=None, language="en",
                                                 use_llm=True, query="topic")
    assert result.reel_count == 2
    assert result.is_short is False
    assert [r.t_start for r in result.reels] == [0.0, 31.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `backend/.venv/bin/python -m pytest backend/tests/clip_engine/test_pipeline_topic_cut.py -v`
Expected: FAIL.

- [ ] **Step 3: Rewrite the `ingest_topic_cut` body**

Mirror Task 9's error mapping and helpers, but keep ALL clips and build `IngestTopicCutResult`:

```python
# inside ingest_topic_cut(...) — after engine_run.clip(...) as in Task 9
duration = float(meta.get("duration_sec") or engine_out["transcript"].get("duration") or 0.0)
reels = [engine_adapter.to_reel_out(c, engine_out["transcript"], meta,
            reel_id=self._new_reel_id(video_id, c["start"], c["end"]),
            material_id=resolved_material_id, concept_id=(concept_id or ""),
            concept_title=(query or ""), video_id=video_id, source_url=source_url,
            position=i + 1, total=len(clips))
         for i, c in enumerate(clips)]
for r, im in ((r, ingest_meta) for r in reels):
    self._persist_reel(r, im)
is_short = duration and duration < 60.0 and not reels
return IngestTopicCutResult(source_url=source_url, video_id=video_id, is_short=bool(is_short),
                            classification_reason=("short" if is_short else "long-form"),
                            duration_sec=duration, reel_count=len(reels), reels=reels,
                            metadata=ingest_meta, terms_notice=self._terms_notice(),
                            trace_id=self._new_trace_id())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `backend/.venv/bin/python -m pytest backend/tests/clip_engine/test_pipeline_topic_cut.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/ingestion/pipeline.py backend/tests/clip_engine/test_pipeline_topic_cut.py
git commit -m "feat(ingest): route ingest_topic_cut through the clip engine"
```

---

## Task 11: Rewire `IngestionPipeline.ingest_search`

**Files:**
- Modify: `backend/app/ingestion/pipeline.py` (method `ingest_search`)
- Test: `backend/tests/clip_engine/test_pipeline_search.py`

**Interfaces:**
- Produces: `ingest_search(...) -> IngestSearchResult` — `clip_engine.search.discover(query, limit=min(max_per_platform, CLIP_SEARCH_MAX_VIDEOS), exclude_video_ids=...)` → per-video `engine_run.clip` (best clip each) → items. `platforms` is coerced to `["yt"]`; non-yt requests still return YouTube results with a `terms_notice` explaining YouTube-only.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/clip_engine/test_pipeline_search.py
def test_search_ingests_youtube_results(monkeypatch, ingestion_pipeline):
    from backend.app.clip_engine import search, run as engine_run, metadata
    monkeypatch.setattr(search, "discover", lambda topic, limit, exclude_video_ids=None, breadth=None: {
        "corrected": "calc", "credits_used": 1, "warning": None,
        "videos": [{"id": "9bZkp7q19f0", "url": "https://www.youtube.com/watch?v=9bZkp7q19f0", "title": "V1",
                    "channel": "Ch", "duration": 300, "view_count": 10, "thumbnail": "t", "upload_date": ""}]})
    monkeypatch.setattr(engine_run, "clip", lambda url, topic, settings=None: {
        "video_id": "9bZkp7q19f0", "clips": [{"start": 3.0, "end": 33.0, "cut_end": 33.1, "title": "X",
        "facet": "c", "reason": "", "sequence_index": 1,
        "embed_url": "https://www.youtube.com/embed/9bZkp7q19f0?start=3&end=33&rel=0"}],
        "transcript": {"segments": [{"start": 3.0, "end": 33.0, "text": "x"}], "words": [], "duration": 300.0},
        "notes": "1"})
    monkeypatch.setattr(metadata, "youtube_metadata", lambda vid: {})
    result = ingestion_pipeline.ingest_search(query="calc", platforms=["yt", "ig", "tt"],
                                              max_per_platform=5, material_id="m1", concept_id=None,
                                              target_clip_duration_sec=45, target_clip_duration_min_sec=15,
                                              target_clip_duration_max_sec=60, language="en",
                                              exclude_video_ids=[])
    assert result.platforms == ["yt"]
    assert result.succeeded == 1
    assert result.items[0].reel.video_url.startswith("https://www.youtube.com/embed/9bZkp7q19f0")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `backend/.venv/bin/python -m pytest backend/tests/clip_engine/test_pipeline_search.py -v`
Expected: FAIL.

- [ ] **Step 3: Rewrite the `ingest_search` body**

```python
# inside ingest_search(...)
from ..clip_engine import search as engine_search, run as engine_run
from ..clip_engine import metadata as engine_meta, adapter as engine_adapter, config as engine_config

resolved_material_id = material_id or self._scratch_material_id
limit = min(max_per_platform, engine_config.CLIP_SEARCH_MAX_VIDEOS)
disc = engine_search.discover(query, limit=limit, exclude_video_ids=exclude_video_ids)

items = []
succeeded = failed = 0
for v in disc["videos"]:
    vid = v["id"]
    try:
        out = engine_run.clip(v["url"], topic=query, settings={"language": language})
        if not out["clips"]:
            items.append(IngestSearchItem(platform="yt", source_url=v["url"], status="skipped"))
            continue
        best = min(out["clips"], key=lambda c: abs((c["end"] - c["start"]) - target_clip_duration_sec))
        meta = {"title": v.get("title", ""), "author_name": v.get("channel", ""),
                "duration_sec": v.get("duration") or out["transcript"].get("duration"),
                "thumbnail_url": v.get("thumbnail", ""), "view_count": v.get("view_count")}
        reel = engine_adapter.to_reel_out(best, out["transcript"], meta,
                    reel_id=self._new_reel_id(vid, best["start"], best["end"]),
                    material_id=resolved_material_id, concept_id=(concept_id or ""),
                    concept_title=query, video_id=vid, source_url=v["url"])
        im = engine_adapter.to_metadata(vid, meta, v["url"])
        self._persist_reel(reel, im)
        items.append(IngestSearchItem(platform="yt", source_url=v["url"], status="ok",
                                      reel=reel, metadata=im))
        succeeded += 1
    except Exception as exc:  # per-video failures are non-fatal
        failed += 1
        items.append(IngestSearchItem(platform="yt", source_url=v["url"], status="error",
                                      error=str(exc)))

return IngestSearchResult(query=query, material_id=resolved_material_id, platforms=["yt"],
    per_platform_resolved={"yt": len(disc["videos"])},
    per_platform_succeeded={"yt": succeeded}, per_platform_failed={"yt": failed},
    per_platform_errors={}, total_resolved=len(disc["videos"]), succeeded=succeeded,
    failed=failed, items=items,
    terms_notice=self._terms_notice() + " Search is YouTube-only.",
    trace_id=self._new_trace_id())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `backend/.venv/bin/python -m pytest backend/tests/clip_engine/test_pipeline_search.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/ingestion/pipeline.py backend/tests/clip_engine/test_pipeline_search.py
git commit -m "feat(ingest): route ingest_search through Supadata + clip engine (YouTube-only)"
```

---

## Task 12: Rewire `IngestionPipeline.ingest_feed`

**Files:**
- Modify: `backend/app/ingestion/pipeline.py` (method `ingest_feed`)
- Test: `backend/tests/clip_engine/test_pipeline_feed.py`

**Interfaces:**
- Produces: `ingest_feed(...) -> IngestFeedResult` — resolve the feed URL to YouTube video URLs via `metadata.resolve_feed_urls(feed_url, max_items)` (yt-dlp `extract_flat`, lazy import, YouTube-only), then clip each (best clip) into `IngestFeedItem`s.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/clip_engine/test_pipeline_feed.py
def test_feed_ingests_resolved_urls(monkeypatch, ingestion_pipeline):
    from backend.app.clip_engine import metadata, run as engine_run
    monkeypatch.setattr(metadata, "resolve_feed_urls",
                        lambda feed_url, max_items: ["https://www.youtube.com/watch?v=9bZkp7q19f0"])
    monkeypatch.setattr(engine_run, "clip", lambda url, topic, settings=None: {
        "video_id": "9bZkp7q19f0", "clips": [{"start": 1.0, "end": 20.0, "cut_end": 20.1, "title": "T",
        "facet": "c", "reason": "", "sequence_index": 1,
        "embed_url": "https://www.youtube.com/embed/9bZkp7q19f0?start=1&end=20&rel=0"}],
        "transcript": {"segments": [{"start": 1.0, "end": 20.0, "text": "x"}], "words": [], "duration": 120.0},
        "notes": "1"})
    monkeypatch.setattr(metadata, "youtube_metadata", lambda vid: {})
    result = ingestion_pipeline.ingest_feed(feed_url="https://www.youtube.com/@chan", max_items=6,
                                            material_id="m1", concept_id=None, target_clip_duration_sec=45,
                                            target_clip_duration_min_sec=15, target_clip_duration_max_sec=60,
                                            language="en")
    assert result.total_resolved == 1 and result.succeeded == 1
    assert result.items[0].status == "ok"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `backend/.venv/bin/python -m pytest backend/tests/clip_engine/test_pipeline_feed.py -v`
Expected: FAIL.

- [ ] **Step 3: Add `resolve_feed_urls` + rewrite `ingest_feed`**

Add to `backend/app/clip_engine/metadata.py`:

```python
def resolve_feed_urls(feed_url: str, max_items: int) -> list[str]:
    """Resolve a channel/playlist URL to individual YouTube watch URLs (no download)."""
    try:
        import yt_dlp  # lazy
        opts = {"quiet": True, "skip_download": True, "extract_flat": True,
                "playlistend": max_items}
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(feed_url, download=False)
        entries = info.get("entries") or []
        urls = []
        for e in entries[:max_items]:
            vid = e.get("id")
            if vid and extract_video_id(f"https://www.youtube.com/watch?v={vid}"):
                urls.append(f"https://www.youtube.com/watch?v={vid}")
        return urls
    except Exception:
        return []
```

Rewrite `ingest_feed` to loop resolved URLs through `engine_run.clip` (best clip → `IngestFeedItem`), building `IngestFeedResult` (mirror Task 11's per-item try/except and persistence).

- [ ] **Step 4: Run test to verify it passes**

Run: `backend/.venv/bin/python -m pytest backend/tests/clip_engine/test_pipeline_feed.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/ingestion/pipeline.py backend/app/clip_engine/metadata.py backend/tests/clip_engine/test_pipeline_feed.py
git commit -m "feat(ingest): route ingest_feed through the clip engine (YouTube-only)"
```

---

## Task 13: Dependencies, config keys, contract smoke test

**Files:**
- Modify: `backend/requirements.txt`, `backend/pyproject.toml`, `backend/app/config.py`
- Test: `backend/tests/clip_engine/test_contract_smoke.py`

**Interfaces:**
- Produces: pinned light-path deps; `Config` fields `clip_engine`, `supadata_api_key`, `supadata_base`, `gemini_model`, `segment_model`, `clip_search_max_videos`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/clip_engine/test_contract_smoke.py
from fastapi.testclient import TestClient


def test_ingest_url_response_schema(monkeypatch):
    # The engine is mocked; this asserts the HTTP layer still emits the ReelOut contract.
    from backend.app.clip_engine import run as engine_run, metadata
    monkeypatch.setattr(engine_run, "clip", lambda url, topic, settings=None: {
        "video_id": "dQw4w9WgXcQ", "notes": "1",
        "clips": [{"start": 5.0, "end": 35.0, "cut_end": 35.1, "title": "T", "facet": "c",
                   "reason": "", "sequence_index": 1,
                   "embed_url": "https://www.youtube.com/embed/dQw4w9WgXcQ?start=5&end=35&rel=0"}],
        "transcript": {"segments": [{"start": 5.0, "end": 35.0, "text": "x"}], "words": [], "duration": 300.0}})
    monkeypatch.setattr(metadata, "youtube_metadata", lambda vid: {"title": "V", "duration_sec": 300.0})
    from backend.app.main import app
    client = TestClient(app)
    r = client.post("/api/ingest/url", json={"source_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                                             "material_id": "m1"})
    assert r.status_code == 200
    body = r.json()
    for key in ("reel", "metadata", "terms_notice", "trace_id"):
        assert key in body
    reel = body["reel"]
    for key in ("reel_id", "video_url", "t_start", "t_end", "captions", "video_duration_sec"):
        assert key in reel
    assert reel["video_url"].startswith("https://www.youtube.com/embed/dQw4w9WgXcQ")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `backend/.venv/bin/python -m pytest backend/tests/clip_engine/test_contract_smoke.py -v`
Expected: FAIL if deps/config missing (e.g. `google-genai` import) — otherwise it should pass once wiring is complete. If it fails only on `SERVERLESS_MODE`, set `monkeypatch.setenv` to disable it in the test.

- [ ] **Step 3: Add deps + config**

Append to `backend/requirements.txt` (and mirror in `pyproject.toml` `dependencies`):

```
google-genai>=1.0
tiktoken==0.8.0
rapidfuzz==3.10.1
```

Add to the `Config` model in `backend/app/config.py` (match the existing pydantic-settings style):

```python
    clip_engine: str = "gemini"
    supadata_api_key: str = ""
    supadata_base: str = "https://api.supadata.ai/v1"
    gemini_model: str = "gemini-2.5-flash"
    segment_model: str = ""
    clip_search_max_videos: int = 5
```

- [ ] **Step 4: Run the full engine suite**

Run: `backend/.venv/bin/python -m pytest backend/tests/clip_engine/ -v`
Expected: PASS (all tasks' tests green).

- [ ] **Step 5: Commit**

```bash
git add backend/requirements.txt backend/pyproject.toml backend/app/config.py backend/tests/clip_engine/test_contract_smoke.py
git commit -m "feat(clip_engine): add deps, config keys, HTTP contract smoke test"
```

---

## Self-Review (author checklist — completed)

**Spec coverage:** scraper port → Tasks 2–5; clipper vendor (gemini/embed) → Tasks 6–7; adapter → Task 8; `/api/ingest/*` rewiring + YouTube-only → Tasks 9–12; deps/hosting/config → Task 13; contract preservation → Tasks 8 & 13. Sync-inline execution → Task 7 `run.clip`. Material path (`/api/reels/*`, `/api/feed`) and dead-code cleanup + key rotation are **explicitly deferred to the follow-up plan** (stated in the header).

**Placeholder scan:** every code step contains real code; the two "reuse existing helpers" notes (Tasks 9–10) point at concrete existing behaviors (sentinel material id, reel-id, persistence, terms notice, trace id) the implementer must read from the current `pipeline.py` — flagged, not vague. `transcribe_supadata`'s exact call signature is called out as a confirm-in-code note in Task 7.

**Type consistency:** clip dict keys (`start/end/cut_end/title/facet/reason/sequence_index/embed_url`) are produced in Task 7 and consumed identically in Tasks 8–12; `discover()` return shape (`corrected/videos/credits_used/warning`) defined in Task 5, consumed in Task 11; `to_reel_out(...)` signature defined in Task 8, called with the same kwargs in Tasks 9–12.

## Open follow-ups for the Phase 3–4 plan
- Route each extracted concept through `search.discover` + `run.clip` inside `reels.py` / `ReelService`, keeping feed ranking, refinement jobs, persistence, and feedback intact (thin `generate_reels_for_concepts()` seam).
- Retire now-dead modules: `services/topic_cut.py`, `services/clip_boundary.py`, `app/ingestion/segment.py`, `app/ingestion/adapters/yt_dlp_adapter.py` search paths, IG/TT adapters, legacy clip parts of `reels.py`.
- Rotate the committed Supadata/Gemini keys in `practice/clips/.env`; scrub from git history; set them only in Railway env.
- Update `README.md` to describe the Supadata+Gemini engine and the new env vars; confirm `Dockerfile` installs any yt-dlp/ffmpeg still needed for metadata/feed resolution.
