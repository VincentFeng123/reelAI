# Knowledge-Level Rating & Difficulty-Aware Feed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Users rate how well they know a topic (Beginner/Intermediate/Advanced, auto-adjusted by feedback); the pipeline searches for level-appropriate videos and the feed soft-boosts level-matched clips, keeping off-level clips at the back so they re-enter when the user's level rises.

**Architecture:** The level lives on the `materials` row and steers (a) Gemini query expansion, (b) video ranking, (c) per-clip difficulty scoring inside the existing gemini_segment call. Difficulty persists on reels; the feed computes the level-match bonus **at serve time** from the material's current effective target, so nothing is ever filtered out and level drift re-sorts the existing library. Spec: `docs/superpowers/specs/2026-07-08-knowledge-level-design.md`.

**Tech Stack:** FastAPI + SQLite/Postgres (backend), pydantic, pytest (`-p no:randomly`, run with `backend/.venv/bin/python -m pytest` from the repo root), Next.js/TypeScript (webapp), SwiftUI (iOS).

## Global Constraints

- Repo root: `/Users/vincentfeng/Documents/reelai app/reelai/reelAI copy 2` (all backend/webapp paths relative to it). iOS app: `/Users/vincentfeng/Documents/reelai app/reelai/reelai/`. Practice clipper: `/Users/vincentfeng/Documents/reelai app/practice/clips/`.
- Levels: `beginner | intermediate | advanced`; absent/None ⇒ `beginner`; invalid strings in API requests ⇒ 422.
- Level→target mapping: beginner **0.15**, intermediate **0.50**, advanced **0.85**. `level_adjustment` clamped to **[-0.35, +0.35]**. `effective_target = clamp01(level_value + level_adjustment)`.
- Difficulty is a SIGNAL, never a filter: no gate anywhere drops a clip for difficulty.
- The feed's level bonus is computed per-request; it must NEVER be baked into persisted `base_score`.
- `RANKED_FEED_CACHE_VERSION` bumps 6 → 7 exactly once (Task 8).
- `backend/app/clip_engine/clipper/pipeline/gemini_segment.py` must stay byte-identical with `practice/clips/backend/pipeline/gemini_segment.py` (Task 13 re-copies it).
- Tests: every backend test run uses `backend/.venv/bin/python -m pytest <file> -p no:randomly -q` from the repo root. Practice tests use `practice/clips/.venv/bin/python -m pytest` from `practice/clips/`.
- The 21 full-suite aggregate failures in community/medium test files are pre-existing order pollution — NOT regressions; judge suites in isolation.

---

### Task 1: DB migrations — `materials.knowledge_level`, `materials.level_adjustment`, `reels.difficulty`

**Files:**
- Modify: `backend/app/db.py` (schema DDL block near line 63 `CREATE TABLE IF NOT EXISTS materials`, sqlite migration helpers near line 678, postgres `ADD COLUMN IF NOT EXISTS` block near line 807)
- Test: `backend/tests/test_knowledge_level_migrations.py` (create)

**Interfaces:**
- Produces: columns `materials.knowledge_level TEXT NOT NULL DEFAULT 'beginner'`, `materials.level_adjustment REAL NOT NULL DEFAULT 0.0`, `reels.difficulty REAL` (nullable) — available to every later task after `get_conn()`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_knowledge_level_migrations.py
"""Columns for the knowledge-level feature exist after init and are idempotent."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class KnowledgeLevelMigrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self._prev = os.environ.get("DATA_DIR")
        os.environ["DATA_DIR"] = self.temp_dir.name
        from backend.app import db as db_module
        from backend.app.config import get_settings
        db_module._db_ready = False
        get_settings.cache_clear()
        self.db = db_module
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        from backend.app.config import get_settings
        if self._prev is None:
            os.environ.pop("DATA_DIR", None)
        else:
            os.environ["DATA_DIR"] = self._prev
        self.db._db_ready = False
        get_settings.cache_clear()

    def _cols(self, table: str) -> dict[str, str]:
        with self.db.get_conn() as conn:
            rows = self.db.fetch_all(conn, f"PRAGMA table_info({table})")
        return {r["name"]: str(r["type"]).upper() for r in rows}

    def test_materials_columns_exist_with_defaults(self) -> None:
        cols = self._cols("materials")
        self.assertIn("knowledge_level", cols)
        self.assertIn("level_adjustment", cols)
        with self.db.get_conn(transactional=True) as conn:
            conn.execute(
                "INSERT INTO materials (id, subject_tag, raw_text, source_type, source_path, created_at) "
                "VALUES ('m1', 'physics', 'Topic: physics', 'topic', NULL, '2026-07-08T00:00:00+00:00')"
            )
        with self.db.get_conn() as conn:
            row = self.db.fetch_one(conn, "SELECT knowledge_level, level_adjustment FROM materials WHERE id='m1'")
        self.assertEqual(row["knowledge_level"], "beginner")
        self.assertEqual(float(row["level_adjustment"]), 0.0)

    def test_reels_difficulty_column_exists_nullable(self) -> None:
        cols = self._cols("reels")
        self.assertIn("difficulty", cols)

    def test_init_is_idempotent(self) -> None:
        # Re-running init on an already-migrated DB must not raise.
        self.db._db_ready = False
        with self.db.get_conn() as conn:
            self.db.fetch_one(conn, "SELECT 1")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `backend/.venv/bin/python -m pytest backend/tests/test_knowledge_level_migrations.py -p no:randomly -q`
Expected: FAIL — `AssertionError: 'knowledge_level' not found in ...`

- [ ] **Step 3: Implement the migrations**

In `backend/app/db.py`:

(a) In the `CREATE TABLE IF NOT EXISTS materials` DDL (line ~63), add two columns after `subject_tag TEXT,`:

```sql
    knowledge_level TEXT NOT NULL DEFAULT 'beginner',
    level_adjustment REAL NOT NULL DEFAULT 0.0,
```

(b) In the `CREATE TABLE IF NOT EXISTS reels` DDL, add after the `base_score` (or equivalent scoring) column line:

```sql
    difficulty REAL,
```

(c) Add a sqlite migration helper next to `_migrate_reels_unique_clip_index_sqlite` (line ~714), following the same pragma-guarded pattern used by the existing sqlite migrators:

```python
def _migrate_knowledge_level_sqlite(conn: sqlite3.Connection) -> None:
    """Add knowledge-level columns to pre-existing DBs (sqlite lacks
    ADD COLUMN IF NOT EXISTS)."""
    material_cols = {r[1] for r in conn.execute("PRAGMA table_info(materials)").fetchall()}
    if "knowledge_level" not in material_cols:
        conn.execute(
            "ALTER TABLE materials ADD COLUMN knowledge_level TEXT NOT NULL DEFAULT 'beginner'"
        )
    if "level_adjustment" not in material_cols:
        conn.execute(
            "ALTER TABLE materials ADD COLUMN level_adjustment REAL NOT NULL DEFAULT 0.0"
        )
    reel_cols = {r[1] for r in conn.execute("PRAGMA table_info(reels)").fetchall()}
    if "difficulty" not in reel_cols:
        conn.execute("ALTER TABLE reels ADD COLUMN difficulty REAL")
```

Call it from the sqlite init path right after the existing `_migrate_reels_unique_clip_index_sqlite(conn)` call (search for that call site inside the init/migration runner).

(d) In the postgres migration block (line ~807, alongside the other `ADD COLUMN IF NOT EXISTS` statements):

```python
                cur.execute("ALTER TABLE materials ADD COLUMN IF NOT EXISTS knowledge_level TEXT NOT NULL DEFAULT 'beginner'")
                cur.execute("ALTER TABLE materials ADD COLUMN IF NOT EXISTS level_adjustment REAL NOT NULL DEFAULT 0.0")
                cur.execute("ALTER TABLE reels ADD COLUMN IF NOT EXISTS difficulty REAL")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `backend/.venv/bin/python -m pytest backend/tests/test_knowledge_level_migrations.py -p no:randomly -q`
Expected: `3 passed`

- [ ] **Step 5: Regression check — DB-touching suites still pass**

Run: `backend/.venv/bin/python -m pytest backend/tests/test_clip_engine_material_topic.py backend/tests/test_clip_engine_generate_reels.py -p no:randomly -q`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add backend/app/db.py backend/tests/test_knowledge_level_migrations.py
git commit -m "feat(db): knowledge_level + level_adjustment on materials, difficulty on reels"
```

---

### Task 2: Level helpers module

**Files:**
- Create: `backend/app/services/knowledge_level.py`
- Test: `backend/tests/test_knowledge_level_helpers.py` (create)

**Interfaces:**
- Produces (used by Tasks 7–10):
  - `KNOWLEDGE_LEVELS: tuple[str, ...] = ("beginner", "intermediate", "advanced")`
  - `LEVEL_VALUES: dict[str, float]` — beginner 0.15 / intermediate 0.50 / advanced 0.85
  - `normalize_knowledge_level(value: str | None) -> str` — lowercase/strip; None/empty → "beginner"; unknown → raises `ValueError`
  - `effective_level_target(level: str | None, adjustment: float | None) -> float` — clamp01(LEVEL_VALUES[normalized] + clamp(adjustment, -0.35, 0.35))
  - `ADJUSTMENT_BOUND: float = 0.35`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_knowledge_level_helpers.py
import sys
import unittest
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.services.knowledge_level import (  # noqa: E402
    LEVEL_VALUES,
    effective_level_target,
    normalize_knowledge_level,
)


class NormalizeTests(unittest.TestCase):
    def test_none_and_empty_default_to_beginner(self) -> None:
        self.assertEqual(normalize_knowledge_level(None), "beginner")
        self.assertEqual(normalize_knowledge_level("  "), "beginner")

    def test_case_and_whitespace_tolerant(self) -> None:
        self.assertEqual(normalize_knowledge_level(" Advanced "), "advanced")

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError):
            normalize_knowledge_level("expert")


class TargetTests(unittest.TestCase):
    def test_mapping(self) -> None:
        self.assertEqual(LEVEL_VALUES["beginner"], 0.15)
        self.assertEqual(LEVEL_VALUES["intermediate"], 0.50)
        self.assertEqual(LEVEL_VALUES["advanced"], 0.85)

    def test_adjustment_applied_and_clamped(self) -> None:
        self.assertAlmostEqual(effective_level_target("beginner", 0.2), 0.35)
        # adjustment beyond the bound is clamped to ±0.35
        self.assertAlmostEqual(effective_level_target("beginner", 9.0), 0.50)
        self.assertAlmostEqual(effective_level_target("advanced", -9.0), 0.50)

    def test_result_clamped_to_unit_interval(self) -> None:
        self.assertAlmostEqual(effective_level_target("advanced", 0.35), 1.0)

    def test_none_inputs(self) -> None:
        self.assertAlmostEqual(effective_level_target(None, None), 0.15)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `backend/.venv/bin/python -m pytest backend/tests/test_knowledge_level_helpers.py -p no:randomly -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.app.services.knowledge_level'`

- [ ] **Step 3: Write the module**

```python
# backend/app/services/knowledge_level.py
"""Knowledge-level semantics: level names, difficulty-scale mapping, and the
effective serving target. Spec: docs/superpowers/specs/2026-07-08-knowledge-level-design.md.

The level lives on the materials row (`knowledge_level`, `level_adjustment`);
everything here is pure — no DB, no LLM.
"""
from __future__ import annotations

KNOWLEDGE_LEVELS: tuple[str, ...] = ("beginner", "intermediate", "advanced")

# Positions on the same 0..1 scale the engine's per-clip `difficulty` uses.
LEVEL_VALUES: dict[str, float] = {
    "beginner": 0.15,
    "intermediate": 0.50,
    "advanced": 0.85,
}

# Auto-adjust drift can never exceed one level step; the user's explicit
# choice stays authoritative.
ADJUSTMENT_BOUND: float = 0.35


def normalize_knowledge_level(value: str | None) -> str:
    """Lowercased/stripped level name; absent -> 'beginner'; unknown -> ValueError."""
    cleaned = (value or "").strip().lower()
    if not cleaned:
        return "beginner"
    if cleaned not in KNOWLEDGE_LEVELS:
        raise ValueError(f"unknown knowledge_level: {value!r}")
    return cleaned


def effective_level_target(level: str | None, adjustment: float | None) -> float:
    """The difficulty the feed should aim at RIGHT NOW for this material."""
    base = LEVEL_VALUES[normalize_knowledge_level(level)]
    adj = max(-ADJUSTMENT_BOUND, min(ADJUSTMENT_BOUND, float(adjustment or 0.0)))
    return max(0.0, min(1.0, base + adj))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `backend/.venv/bin/python -m pytest backend/tests/test_knowledge_level_helpers.py -p no:randomly -q`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/knowledge_level.py backend/tests/test_knowledge_level_helpers.py
git commit -m "feat(level): knowledge-level helpers (mapping, normalize, effective target)"
```

---

### Task 3: Engine — per-clip `difficulty` from the same Gemini call

**Files:**
- Modify: `backend/app/clip_engine/clipper/pipeline/gemini_segment.py`
- Test: `backend/tests/clip_engine/test_gemini_segment_difficulty.py` (create)

**Interfaces:**
- Consumes: existing `_Topic`, `_plan_to_clips`, `_prompts`, `_norm_informativeness` in that file.
- Produces: every clip dict returned by `segment_clips` carries `"difficulty": float` in [0,1] (default 0.5 when the model omits it). NO gate uses it.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/clip_engine/test_gemini_segment_difficulty.py
"""Per-clip difficulty: parsed, normalized, carried — NEVER gates a clip."""
import pytest

from backend.app.clip_engine.clipper.pipeline.gemini_segment import (
    _Plan,
    _Topic,
    _plan_to_clips,
    _prompts,
)


def _segs(n: int, sec: float = 30.0) -> list[dict]:
    return [{"start": i * sec, "end": (i + 1) * sec, "text": f"line {i}"} for i in range(n)]


def _run(topics, n=10, settings=None):
    return _plan_to_clips(_Plan(topics=topics), _segs(n), [], settings or {})


def test_difficulty_carried_on_clip():
    t = _Topic(title="T", start_line=0, end_line=1, informativeness=0.9, difficulty=0.8)
    clips = _run([t])
    assert clips[0]["difficulty"] == pytest.approx(0.8)


def test_difficulty_defaults_to_half_when_omitted():
    t = _Topic(title="T", start_line=0, end_line=1, informativeness=0.9)
    clips = _run([t])
    assert clips[0]["difficulty"] == pytest.approx(0.5)


def test_misscaled_difficulty_normalized():
    t = _Topic(title="T", start_line=0, end_line=1, informativeness=0.9, difficulty=7)
    clips = _run([t])
    assert clips[0]["difficulty"] == pytest.approx(0.7)


def test_extreme_difficulty_never_gates():
    hard = _Topic(title="H", start_line=0, end_line=1, informativeness=0.9, difficulty=1.0)
    easy = _Topic(title="E", start_line=2, end_line=3, informativeness=0.9, difficulty=0.0)
    clips = _run([hard, easy])
    assert {c["title"] for c in clips} == {"H", "E"}


def test_prompt_documents_difficulty_scale():
    system, user = _prompts("[0] 00:00 hi", 1)
    assert "difficulty" in system
    assert "difficulty" in user
```

- [ ] **Step 2: Run test to verify it fails**

Run: `backend/.venv/bin/python -m pytest backend/tests/clip_engine/test_gemini_segment_difficulty.py -p no:randomly -q`
Expected: FAIL — `_Topic` has no field `difficulty` / KeyError `'difficulty'`

- [ ] **Step 3: Implement**

In `backend/app/clip_engine/clipper/pipeline/gemini_segment.py`:

(a) `_Topic` gains one field after `informativeness`:

```python
    difficulty: float = 0.5   # 0 = assumes no prior knowledge, 1 = expert-level
```

(b) In `_prompts`, extend the per-clip field list in the `system` string. Replace the fragment

```
"informativeness — 0.0 to 1.0, how much a motivated student learns from this clip "
"ALONE (0.9+: a complete idea taught well; ~0.5: partial value; <0.5: little value). "
```

with

```
"informativeness — 0.0 to 1.0, how much a motivated student learns from this clip "
"ALONE (0.9+: a complete idea taught well; ~0.5: partial value; <0.5: little value); "
"difficulty — 0.0 to 1.0, the prior knowledge the clip ASSUMES (0.1: no background, "
"first exposure; 0.5: comfortable with the basics; 0.9: graduate/expert material). "
```

and in the `user` string replace `informativeness}` with `informativeness, difficulty}` inside the returned-fields list.

(c) In `_plan_to_clips`, where the raw clip dict is appended (the `raw.append({...})` call), add after `"informativeness": info`:

```python
                    "difficulty": _norm_informativeness(tp.difficulty),
```

(`_norm_informativeness` is scale-clamping for any 0..1 score; reuse it rather than duplicating.)

- [ ] **Step 4: Run tests — new file plus the existing engine suites**

Run: `backend/.venv/bin/python -m pytest backend/tests/clip_engine -p no:randomly -q`
Expected: all pass (new 5 + existing)

- [ ] **Step 5: Commit**

```bash
git add backend/app/clip_engine/clipper/pipeline/gemini_segment.py backend/tests/clip_engine/test_gemini_segment_difficulty.py
git commit -m "feat(engine): per-clip difficulty score from the same segmentation call"
```

---

### Task 4: Persist difficulty on reels

**Files:**
- Modify: `backend/app/ingestion/persistence.py` (`upsert_reel_row`, line ~194)
- Modify: `backend/app/ingestion/pipeline.py` (`_persist_engine_clip` → `_persist_ingest` → `upsert_reel_row` chain)
- Test: extend `backend/tests/test_clip_engine_material_topic.py`

**Interfaces:**
- Consumes: clip dicts now carry `"difficulty"` (Task 3).
- Produces: `upsert_reel_row(..., difficulty: float | None = None)` writes `reels.difficulty`; `_persist_ingest(..., clip_difficulty: float | None = None)`.

- [ ] **Step 1: Write the failing test** (append to `backend/tests/test_clip_engine_material_topic.py`)

```python
class DifficultyPersistenceTests(IngestTopicTests):
    """Engine difficulty lands in reels.difficulty; absent -> NULL."""

    @staticmethod
    def _difficulty_engine_out(*_a, **_kw) -> dict:
        return {
            "video_id": "vidAAAAAAAA",
            "clips": [{
                "start": 30.0, "end": 75.0, "cut_end": 75.15,
                "title": "Scored", "facet": "", "reason": "",
                "informativeness": 0.9, "difficulty": 0.8,
                "sequence_index": 0,
                "embed_url": "https://www.youtube.com/embed/vidAAAAAAAA?start=30&end=75&rel=0",
            }],
            "transcript": {"segments": [
                {"start": 30.0, "end": 75.0, "text": "here we explain photosynthesis"}
            ], "words": [], "duration": 600.0},
            "notes": "",
        }

    def test_difficulty_round_trips_to_db(self) -> None:
        with _Patched() as (mock_search, mock_run):
            mock_search.discover.side_effect = (
                lambda topic, limit, exclude_video_ids=None, **kw: {
                    "corrected": topic, "videos": [VID_A],
                    "credits_used": 0, "warning": None,
                }
            )
            mock_run.clip.side_effect = self._difficulty_engine_out
            main_module.ingestion_pipeline.ingest_topic(
                topic=TOPIC, material_id="mat-diff", concept_id="con-diff",
                generation_id="gen-diff", max_videos=1,
            )
        with db_module.get_conn() as conn:
            row = db_module.fetch_all(
                conn, "SELECT difficulty FROM reels WHERE generation_id = ?", ("gen-diff",)
            )[0]
        self.assertAlmostEqual(float(row["difficulty"]), 0.8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `backend/.venv/bin/python -m pytest "backend/tests/test_clip_engine_material_topic.py::DifficultyPersistenceTests" -p no:randomly -q`
Expected: FAIL — difficulty is NULL (`TypeError: float() argument must be ... not 'NoneType'`)

- [ ] **Step 3: Implement**

(a) `backend/app/ingestion/persistence.py` — `upsert_reel_row` signature gains `difficulty: float | None = None` (after `generation_id`), and the `row = {...}` dict gains `"difficulty": difficulty,`.

(b) `backend/app/ingestion/pipeline.py`:
- `_persist_ingest` signature gains `clip_difficulty: float | None = None` (after `clip_title`); pass `difficulty=clip_difficulty` in its `upsert_reel_row(...)` call.
- `_persist_engine_clip` passes it from the clip dict in its `self._persist_ingest(...)` call:

```python
            clip_title=str(clip.get("title") or "").strip(),
            clip_difficulty=(
                None if clip.get("difficulty") is None else float(clip["difficulty"])
            ),
```

- [ ] **Step 4: Run tests**

Run: `backend/.venv/bin/python -m pytest backend/tests/test_clip_engine_material_topic.py backend/tests/test_ingestion_url.py -p no:randomly -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add backend/app/ingestion/persistence.py backend/app/ingestion/pipeline.py backend/tests/test_clip_engine_material_topic.py
git commit -m "feat(persist): reels.difficulty round-trip from engine clips"
```

---

### Task 5: Expansion steering by level

**Files:**
- Modify: `backend/app/clip_engine/expand.py`
- Test: extend `backend/tests/clip_engine/test_expand.py`

**Interfaces:**
- Produces: `expand_query(topic: str, n: int, level: str | None = None) -> dict` — level is a normalized level name or None; beginner/advanced add one steering line to the system prompt; intermediate/None add nothing.

- [ ] **Step 1: Write the failing test** (append to `backend/tests/clip_engine/test_expand.py`)

```python
def test_expand_level_steering_lines(monkeypatch):
    from backend.app.clip_engine import expand as ex
    seen = {}

    def fake_raw(system, user, model):
        seen["system"] = system
        return '{"corrected": "physics", "queries": ["physics"]}'

    monkeypatch.setattr(ex.config, "GEMINI_API_KEY", "k")
    monkeypatch.setattr(ex, "_gemini_expand_raw", fake_raw)

    ex.expand_query("physics", 3, level="beginner")
    assert "beginner" in seen["system"].lower()
    assert "introduction to" in seen["system"]

    ex.expand_query("physics", 3, level="advanced")
    assert "advanced" in seen["system"].lower()
    assert "graduate" in seen["system"]

    ex.expand_query("physics", 3, level="intermediate")
    assert "graduate" not in seen["system"] and "for beginners" not in seen["system"]

    ex.expand_query("physics", 3)  # None unchanged
    assert "graduate" not in seen["system"] and "for beginners" not in seen["system"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `backend/.venv/bin/python -m pytest backend/tests/clip_engine/test_expand.py -p no:randomly -q`
Expected: FAIL — `expand_query() got an unexpected keyword argument 'level'`

- [ ] **Step 3: Implement**

In `backend/app/clip_engine/expand.py`:

```python
_LEVEL_LINES = {
    "beginner": (
        " The viewer is a BEGINNER on this topic: prefer phrasings like "
        "'introduction to X', 'X basics', 'X for beginners', 'X crash course'; "
        "avoid graduate-level or research phrasings."
    ),
    "advanced": (
        " The viewer is ADVANCED on this topic: prefer phrasings like "
        "'advanced X', 'graduate X lecture', 'X deep dive', 'X seminar'; "
        "avoid 'for beginners' phrasings."
    ),
}


def expand_query(topic: str, n: int, level: str | None = None) -> dict:
    topic = topic.strip()
    system = _SYSTEM + _LEVEL_LINES.get((level or "").strip().lower(), "")
    if not config.GEMINI_API_KEY:
        return free_expand(topic, n)
    try:
        raw = _gemini_expand_raw(system, _user(topic, n), config.EXPAND_MODEL)
        parsed = _safe_json(raw)
        if parsed:
            return {"corrected": parsed.get("corrected") or topic,
                    "queries": _normalize(parsed.get("corrected"), parsed.get("queries"), topic, n),
                    "provider_used": "gemini"}
    except Exception:
        pass
    return free_expand(topic, n)
```

(This replaces the existing `expand_query` body; the only changes are the `level` parameter and `system` assembly — the `_SYSTEM` constant itself is untouched.)

- [ ] **Step 4: Run tests**

Run: `backend/.venv/bin/python -m pytest backend/tests/clip_engine/test_expand.py -p no:randomly -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add backend/app/clip_engine/expand.py backend/tests/clip_engine/test_expand.py
git commit -m "feat(expand): level-targeted query expansion (beginner/advanced steering lines)"
```

---

### Task 6: Video ranking level bands

**Files:**
- Modify: `backend/app/clip_engine/rank.py`
- Test: extend `backend/tests/clip_engine/test_rank.py`

**Interfaces:**
- Consumes: existing `_edu_score` convention (score-only adjustment; `(match_count, score, view_count)` sort keys unchanged).
- Produces: `merge_and_rank(per_query: list[dict], level: str | None = None) -> list[dict]`; internal `_level_score(v: dict, level: str | None) -> float` in [-2, +2].

- [ ] **Step 1: Write the failing test** (append to `backend/tests/clip_engine/test_rank.py`)

```python
def test_level_score_bands():
    from backend.app.clip_engine.rank import _level_score
    intro = {"title": "Introduction to Physics 101", "channel": ""}
    grad = {"title": "Graduate Physics Seminar", "channel": ""}
    assert _level_score(intro, "beginner") > 0
    assert _level_score(grad, "beginner") < 0
    assert _level_score(grad, "advanced") > 0
    assert _level_score(intro, "advanced") < 0
    assert _level_score(intro, None) == 0.0
    assert _level_score(intro, "intermediate") == 0.0


def test_level_score_clamped():
    from backend.app.clip_engine.rank import _level_score
    stacked = {"title": "intro introduction basics beginner 101 crash course", "channel": ""}
    assert _level_score(stacked, "beginner") <= 2.0


def test_merge_and_rank_level_reorders_within_match_band():
    from backend.app.clip_engine.rank import merge_and_rank
    per_query = [{"videos": [
        {"id": "adv", "title": "Graduate Physics Seminar", "viewCount": 100},
        {"id": "beg", "title": "Physics for Beginners", "viewCount": 100},
    ]}]
    ranked = merge_and_rank(per_query, level="beginner")
    assert [v["id"] for v in ranked][0] == "beg"
    ranked_none = merge_and_rank(per_query)  # level omitted -> original behavior
    assert len(ranked_none) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `backend/.venv/bin/python -m pytest backend/tests/clip_engine/test_rank.py -p no:randomly -q`
Expected: FAIL — `ImportError: cannot import name '_level_score'`

- [ ] **Step 3: Implement**

In `backend/app/clip_engine/rank.py`, after the `_edu_score` block:

```python
# -- Knowledge-level ranking signal ------------------------------------------ #
# +1.0 per distinct hit matching the viewer's band, -1.0 per hit in the
# opposite band, clamped to ±2.0. Added to `score` only — the
# (match_count, score, view_count) sort-key structure stays unchanged,
# same convention as _edu_score.

_BEGINNER_BAND = [
    re.compile(r'\bintro(?:duction)?\b', re.I),
    re.compile(r'\bbasics\b', re.I),
    re.compile(r'\bbeginners?\b', re.I),
    re.compile(r'\b101\b'),
    re.compile(r'\bcrash\s+course\b', re.I),
    re.compile(r'\bfor\s+dummies\b', re.I),
]

_ADVANCED_BAND = [
    re.compile(r'\badvanced\b', re.I),
    re.compile(r'\bgraduate\b', re.I),
    re.compile(r'\bseminar\b', re.I),
    re.compile(r'\bresearch\b', re.I),
    re.compile(r'\bproofs?\b', re.I),
    re.compile(r'\blecture\s+\d{2,3}\b', re.I),
]


def _level_score(v: dict, level: str | None) -> float:
    lvl = (level or "").strip().lower()
    if lvl == "beginner":
        match_band, opposite_band = _BEGINNER_BAND, _ADVANCED_BAND
    elif lvl == "advanced":
        match_band, opposite_band = _ADVANCED_BAND, _BEGINNER_BAND
    else:
        return 0.0
    text = f"{v.get('title', '')} {_channel_name(v)}"
    hits = sum(1.0 for p in match_band if p.search(text))
    misses = sum(1.0 for p in opposite_band if p.search(text))
    return max(-2.0, min(2.0, hits - misses))
```

Then in `merge_and_rank`: change the signature to `def merge_and_rank(per_query: list[dict], level: str | None = None) -> list[dict]:` and, at the point where `_edu_score(...)` is added into each video's `score`, also add `+ _level_score(v, level)` (identical placement convention — score only, sort keys untouched).

- [ ] **Step 4: Run tests**

Run: `backend/.venv/bin/python -m pytest backend/tests/clip_engine/test_rank.py -p no:randomly -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add backend/app/clip_engine/rank.py backend/tests/clip_engine/test_rank.py
git commit -m "feat(rank): level-band title signal added to video ranking score"
```

---

### Task 7: Thread the level through discover → ingest_topic → generate_reels

**Files:**
- Modify: `backend/app/clip_engine/search.py` (`discover`, line ~7)
- Modify: `backend/app/ingestion/pipeline.py` (`ingest_topic` signature + `discover` call + `_clip_and_filter` unchanged)
- Modify: `backend/app/services/reels.py` (`generate_reels` material fetch line ~1553 and `ingest_topic` call line ~1799)
- Test: extend `backend/tests/clip_engine/test_search.py` and `backend/tests/test_clip_engine_material_topic.py`

**Interfaces:**
- Produces: `discover(topic, limit, exclude_video_ids=None, breadth=None, level=None)`; `ingest_topic(..., knowledge_level: str | None = None)`; `generate_reels` reads `knowledge_level` from the material row and passes it down.

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/clip_engine/test_search.py`:

```python
def test_discover_threads_level_to_expand_and_rank(monkeypatch):
    from backend.app.clip_engine import search as s
    seen = {}
    monkeypatch.setattr(s.expand, "expand_query",
                        lambda topic, n, level=None: seen.setdefault("expand_level", level)
                        or {"corrected": topic, "queries": [topic]})
    monkeypatch.setattr(s.supadata_search, "search_all",
                        lambda qs: {"per_query": [], "credits_used": 0, "warning": None})
    monkeypatch.setattr(s.rank, "merge_and_rank",
                        lambda pq, level=None: seen.setdefault("rank_level", level) or [])
    s.discover("physics", limit=3, level="advanced")
    assert seen["expand_level"] == "advanced"
    assert seen["rank_level"] == "advanced"
```

Append to `backend/tests/test_clip_engine_material_topic.py` (inside a new class following the `_Patched` harness):

```python
class LevelThreadingTests(IngestTopicTests):
    def test_ingest_topic_passes_level_to_discover(self) -> None:
        with _Patched() as (mock_search, mock_run):
            mock_run.clip.side_effect = _clip_side_effect
            main_module.ingestion_pipeline.ingest_topic(
                topic=TOPIC, material_id="mat-lvl", concept_id="con-lvl",
                generation_id="gen-lvl", max_videos=1, knowledge_level="advanced",
            )
            _, kwargs = mock_search.discover.call_args
            self.assertEqual(kwargs.get("level"), "advanced")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `backend/.venv/bin/python -m pytest backend/tests/clip_engine/test_search.py "backend/tests/test_clip_engine_material_topic.py::LevelThreadingTests" -p no:randomly -q`
Expected: FAIL — unexpected keyword `level` / `knowledge_level`

- [ ] **Step 3: Implement**

(a) `backend/app/clip_engine/search.py`:

```python
def discover(topic: str, limit: int, exclude_video_ids: list[str] | None = None,
             breadth: int | None = None, level: str | None = None) -> dict:
    n = max(1, breadth or config.SEARCH_BREADTH)
    expansion = expand.expand_query(topic, n, level=level)
    res = supadata_search.search_all(expansion["queries"])
    ranked = rank.merge_and_rank(res["per_query"], level=level)
    exclude = set(exclude_video_ids or [])
    videos = [v for v in ranked if v["id"] not in exclude][:limit]
    return {"corrected": expansion["corrected"], "videos": videos,
            "credits_used": res["credits_used"], "warning": res["warning"]}
```

(b) `backend/app/ingestion/pipeline.py` — `ingest_topic` signature gains `knowledge_level: str | None = None` (after `language`), and the `clip_engine_search.discover(...)` call becomes:

```python
        disc = clip_engine_search.discover(
            topic, limit=limit, exclude_video_ids=bare_exclusions, level=knowledge_level
        )
```

(c) `backend/app/services/reels.py` — in `generate_reels`, the material fetch (line ~1553) becomes:

```python
        material = fetch_one(
            conn,
            "SELECT subject_tag, source_type, knowledge_level, level_adjustment FROM materials WHERE id = ?",
            (material_id,),
        )
        material_knowledge_level = str((material or {}).get("knowledge_level") or "beginner")
```

and the `self.ingestion_pipeline.ingest_topic(...)` call gains:

```python
                    knowledge_level=material_knowledge_level,
```

- [ ] **Step 4: Run tests**

Run: `backend/.venv/bin/python -m pytest backend/tests/clip_engine/test_search.py backend/tests/test_clip_engine_material_topic.py backend/tests/test_clip_engine_generate_reels.py -p no:randomly -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add backend/app/clip_engine/search.py backend/app/ingestion/pipeline.py backend/app/services/reels.py backend/tests/clip_engine/test_search.py backend/tests/test_clip_engine_material_topic.py
git commit -m "feat(level): thread knowledge level generate_reels -> ingest_topic -> discover"
```

---

### Task 8: Feed — serve-time level bonus, progression, cache keying (the "keep them in the back" contract)

**Files:**
- Modify: `backend/app/services/reels.py` (`ranked_feed` SELECT ~line 7975 + score block ~line 8218; `_ranked_feed_cache_key` line ~1343; `RANKED_FEED_CACHE_VERSION` line ~1275)
- Test: extend `backend/tests/test_clip_engine_generate_reels.py`

**Interfaces:**
- Consumes: `effective_level_target` (Task 2); `reels.difficulty` (Tasks 1/4).
- Produces: feed scoring includes the level bonus; `ranked_feed` exposes `self._last_effective_level_target: float` AND `self._last_knowledge_level: str` for the endpoint (Task 10); cache version 7.

- [ ] **Step 1: Write the failing test** (append to `backend/tests/test_clip_engine_generate_reels.py`; use the file's existing harness — patched engine, seeded material, `db_module`, `main_module`)

```python
class LevelAwareFeedTests(ClipEngineGenerateReelsTests):
    """Serve-time level scoring: matched clips first, off-level kept at the
    back, and a level change re-sorts WITHOUT regeneration."""

    def _seed_two_reels_with_difficulty(self) -> None:
        # Two persisted reels on the same material: one easy, one hard.
        with db_module.get_conn(transactional=True) as conn:
            for reel_id, vid, d in (("r-easy", "yt:videasy00001", 0.15),
                                    ("r-hard", "yt:vidhard00001", 0.85)):
                conn.execute(
                    "INSERT INTO videos (id, platform, source_id, title, channel_title, duration_sec, created_at) "
                    "VALUES (?, 'yt', ?, 'Photosynthesis', 'Chan', 600, '2026-07-08T00:00:00+00:00')",
                    (vid, vid.split(":", 1)[1]),
                )
                conn.execute(
                    "INSERT INTO reels (id, material_id, concept_id, video_id, video_url, t_start, t_end, "
                    "transcript_snippet, takeaways_json, base_score, generation_id, difficulty, created_at) "
                    "VALUES (?, ?, ?, ?, 'https://www.youtube.com/embed/x?start=0&end=30', 0, 30, "
                    "'photosynthesis explained', '[]', 0.8, 'gen-lvl', ?, '2026-07-08T00:00:00+00:00')",
                    (reel_id, MATERIAL_ID, CONCEPT_ID, vid, d),
                )

    def test_beginner_feed_puts_easy_first_but_keeps_hard(self) -> None:
        self._seed_two_reels_with_difficulty()
        with db_module.get_conn(transactional=True) as conn:
            conn.execute("UPDATE materials SET knowledge_level='beginner', level_adjustment=0 WHERE id=?",
                         (MATERIAL_ID,))
        with db_module.get_conn() as conn:
            feed = main_module.reel_service.ranked_feed(conn, material_id=MATERIAL_ID, generation_id="gen-lvl")
        ids = [r["reel_id"] for r in feed]
        self.assertEqual(ids[0], "r-easy")
        self.assertIn("r-hard", ids)          # NEVER hidden — waits at the back

    def test_level_change_resorts_without_regeneration(self) -> None:
        self._seed_two_reels_with_difficulty()
        with db_module.get_conn(transactional=True) as conn:
            conn.execute("UPDATE materials SET knowledge_level='advanced', level_adjustment=0 WHERE id=?",
                         (MATERIAL_ID,))
        with db_module.get_conn() as conn:
            feed = main_module.reel_service.ranked_feed(conn, material_id=MATERIAL_ID, generation_id="gen-lvl")
        self.assertEqual(feed[0]["reel_id"], "r-hard")   # the back-of-feed clip re-entered

    def test_cache_version_is_7(self) -> None:
        self.assertEqual(main_module.reel_service.RANKED_FEED_CACHE_VERSION, 7)
```

NOTE for the implementer: check the exact `reels`/`videos` INSERT column lists against `backend/app/db.py` DDL before running — adjust column names in the seed helper if they differ (e.g. `takeaways_json` vs `takeaways`). The assertion structure is the contract.

- [ ] **Step 2: Run test to verify it fails**

Run: `backend/.venv/bin/python -m pytest "backend/tests/test_clip_engine_generate_reels.py::LevelAwareFeedTests" -p no:randomly -q`
Expected: FAIL (no difficulty in feed scoring; cache version 6)

- [ ] **Step 3: Implement**

In `backend/app/services/reels.py`:

(a) Import at top (near the other service imports): `from .knowledge_level import effective_level_target`

(b) `RANKED_FEED_CACHE_VERSION = 7` (was 6) — also update the pinned test `test_ranked_feed_cache_version_is_6` in `backend/tests/test_clip_engine_feed_refine_feedback.py` to expect 7 (rename to `..._is_7`, message "level-aware scoring").

(c) In `ranked_feed`, where the material row is loaded (it already reads `subject_tag`/`source_type` for `strict_topic_only`), extend that SELECT with `knowledge_level, level_adjustment` and compute once:

```python
        level_target = effective_level_target(
            (material or {}).get("knowledge_level"),
            (material or {}).get("level_adjustment"),
        )
        self._last_effective_level_target = level_target
        self._last_knowledge_level = str((material or {}).get("knowledge_level") or "beginner")
```

(d) Add `r.difficulty` to the big reels SELECT column list (line ~7975) AND to its `GROUP BY` list.

(e) In the score block (line ~8218, the `score = (...)` expression), append two terms:

```python
                + 0.12 * (1.0 - 2.0 * abs(
                    (0.5 if row.get("difficulty") is None else float(row["difficulty"]))
                    - level_target
                ))
                + 0.05 * (1.0 - (0.5 if row.get("difficulty") is None else float(row["difficulty"])))
                    * max(0.0, 1.0 - (safe_page_hint - 1) / 2.0)
```

…where `safe_page_hint` is the page indicator already available in `ranked_feed`'s scope — **verify its actual variable name in `ranked_feed` (it may be `page` or `page_hint`) and use that**. If `ranked_feed` has no page concept in scope, apply the progression term only when the caller passes `page_hint=1` behavior — i.e. add a `page_hint: int = 1` parameter to `ranked_feed` defaulting to 1 and have the `/api/feed` endpoint pass its page number.

Also add `"difficulty"` to the `scored.append({...})` dict: `"difficulty": (None if row.get("difficulty") is None else float(row["difficulty"])),`.

(f) `_ranked_feed_cache_key` (line ~1343) gains a parameter `level_target: float = 0.5` and adds `"level_target": round(float(level_target), 3),` to the `payload` dict; both call sites (lines ~1459 and ~1498) pass `level_target=level_target`.

- [ ] **Step 4: Run tests**

Run: `backend/.venv/bin/python -m pytest backend/tests/test_clip_engine_generate_reels.py backend/tests/test_clip_engine_feed_refine_feedback.py -p no:randomly -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/reels.py backend/tests/test_clip_engine_generate_reels.py backend/tests/test_clip_engine_feed_refine_feedback.py
git commit -m "feat(feed): serve-time level bonus + page-1 progression; cache v7 keyed on level target"
```

---

### Task 9: Auto-adjust from feedback

**Files:**
- Modify: `backend/app/services/reels.py` (`record_feedback` + new `update_level_adjustment` method near `_concept_mastery`, line ~7055)
- Test: `backend/tests/test_level_auto_adjust.py` (create)

**Interfaces:**
- Consumes: `reel_feedback` rows (existing), `ADJUSTMENT_BOUND` (Task 2).
- Produces: `ReelService.update_level_adjustment(conn, material_id: str) -> float` — recomputes and persists `materials.level_adjustment`, returns the new value; called at the end of `record_feedback`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_level_auto_adjust.py
"""Auto-adjust: last-20 window, <5-row gate, direction, ±0.35 bound.
Uses a temp DB (same DATA_DIR pattern as test_knowledge_level_migrations)."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class LevelAutoAdjustTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self._prev = os.environ.get("DATA_DIR")
        os.environ["DATA_DIR"] = self.temp_dir.name
        from backend.app import db as db_module
        from backend.app.config import get_settings
        from backend.app.services.reels import ReelService
        db_module._db_ready = False
        get_settings.cache_clear()
        self.db = db_module
        self.svc = ReelService(embedding_service=None, youtube_service=None)
        self.addCleanup(self._restore)
        with self.db.get_conn(transactional=True) as conn:
            conn.execute(
                "INSERT INTO materials (id, subject_tag, raw_text, source_type, source_path, created_at) "
                "VALUES ('m1', 'physics', 'Topic: physics', 'topic', NULL, '2026-07-08T00:00:00+00:00')"
            )
            conn.execute(
                "INSERT INTO concepts (id, material_id, title, keywords_json, summary, embedding_json, created_at) "
                "VALUES ('c1', 'm1', 'Physics', '[]', '', NULL, '2026-07-08T00:00:00+00:00')"
            )
            conn.execute(
                "INSERT INTO videos (id, platform, source_id, title, channel_title, duration_sec, created_at) "
                "VALUES ('yt:v1', 'yt', 'v1', 'T', 'C', 600, '2026-07-08T00:00:00+00:00')"
            )
            conn.execute(
                "INSERT INTO reels (id, material_id, concept_id, video_id, video_url, t_start, t_end, "
                "transcript_snippet, takeaways_json, base_score, created_at) "
                "VALUES ('r1', 'm1', 'c1', 'yt:v1', 'u', 0, 30, 's', '[]', 1.0, '2026-07-08T00:00:00+00:00')"
            )

    def _restore(self) -> None:
        from backend.app.config import get_settings
        if self._prev is None:
            os.environ.pop("DATA_DIR", None)
        else:
            os.environ["DATA_DIR"] = self._prev
        self.db._db_ready = False
        get_settings.cache_clear()

    def _feedback(self, n: int, helpful: int, confusing: int, rating: int | None) -> None:
        with self.db.get_conn(transactional=True) as conn:
            for i in range(n):
                conn.execute(
                    "INSERT INTO reel_feedback (reel_id, helpful, confusing, rating, created_at) "
                    "VALUES ('r1', ?, ?, ?, ?)",
                    (helpful, confusing, rating, f"2026-07-08T00:00:{i:02d}+00:00"),
                )

    def _adj(self) -> float:
        with self.db.get_conn() as conn:
            adj = self.svc.update_level_adjustment(conn, "m1")
        return adj

    def test_gate_below_five_rows(self) -> None:
        self._feedback(4, helpful=1, confusing=0, rating=5)
        self.assertEqual(self._adj(), 0.0)

    def test_sustained_helpful_drifts_up(self) -> None:
        self._feedback(10, helpful=1, confusing=0, rating=5)
        self.assertGreater(self._adj(), 0.0)

    def test_sustained_confusing_drifts_down(self) -> None:
        self._feedback(10, helpful=0, confusing=1, rating=2)
        self.assertLess(self._adj(), 0.0)

    def test_bounded(self) -> None:
        self._feedback(20, helpful=1, confusing=0, rating=5)
        self.assertLessEqual(abs(self._adj()), 0.35)

    def test_persisted_on_material(self) -> None:
        self._feedback(10, helpful=1, confusing=0, rating=5)
        expected = self._adj()
        with self.db.get_conn() as conn:
            row = self.db.fetch_one(conn, "SELECT level_adjustment FROM materials WHERE id='m1'")
        self.assertAlmostEqual(float(row["level_adjustment"]), expected)


if __name__ == "__main__":
    unittest.main()
```

NOTE for the implementer: verify the `reel_feedback` INSERT column list against `backend/app/db.py` line ~201 (it may include an `id` or owner column) and adjust the seed helper. The assertion structure is the contract.

- [ ] **Step 2: Run test to verify it fails**

Run: `backend/.venv/bin/python -m pytest backend/tests/test_level_auto_adjust.py -p no:randomly -q`
Expected: FAIL — `AttributeError: 'ReelService' object has no attribute 'update_level_adjustment'`

- [ ] **Step 3: Implement**

In `backend/app/services/reels.py`, next to `_concept_mastery` (line ~7055):

```python
    LEVEL_FEEDBACK_WINDOW = 20
    LEVEL_FEEDBACK_MIN_ROWS = 5

    def update_level_adjustment(self, conn, material_id: str) -> float:
        """Recompute the material's level drift from its most recent feedback.

        signal = 0.25*helpful_rate - 0.35*confusing_rate + 0.15*(avg_rating-3)/2
        (same shape as _concept_mastery), clamped to ±ADJUSTMENT_BOUND.
        Fewer than LEVEL_FEEDBACK_MIN_ROWS rows -> 0 (cold-start gate)."""
        from .knowledge_level import ADJUSTMENT_BOUND

        rows = fetch_all(
            conn,
            """
            SELECT f.helpful, f.confusing, f.rating
            FROM reel_feedback f
            JOIN reels r ON r.id = f.reel_id
            WHERE r.material_id = ?
            ORDER BY f.created_at DESC
            LIMIT ?
            """,
            (material_id, self.LEVEL_FEEDBACK_WINDOW),
        )
        if len(rows) < self.LEVEL_FEEDBACK_MIN_ROWS:
            adjustment = 0.0
        else:
            n = len(rows)
            helpful_rate = sum(1 for r in rows if int(r["helpful"] or 0) > 0) / n
            confusing_rate = sum(1 for r in rows if int(r["confusing"] or 0) > 0) / n
            ratings = [float(r["rating"]) for r in rows if r["rating"] is not None]
            avg_rating = (sum(ratings) / len(ratings)) if ratings else 3.0
            signal = 0.25 * helpful_rate - 0.35 * confusing_rate + 0.15 * (avg_rating - 3.0) / 2.0
            adjustment = max(-ADJUSTMENT_BOUND, min(ADJUSTMENT_BOUND, signal))
        execute_modify(
            conn,
            "UPDATE materials SET level_adjustment = ? WHERE id = ?",
            (adjustment, material_id),
        )
        return adjustment
```

Then at the end of `record_feedback` (find `def record_feedback` in the same file), after the feedback row is written, add — non-fatal on error per the spec:

```python
        try:
            reel_row = fetch_one(conn, "SELECT material_id FROM reels WHERE id = ?", (reel_id,))
            if reel_row and reel_row.get("material_id"):
                self.update_level_adjustment(conn, str(reel_row["material_id"]))
        except Exception:
            logger.exception("level adjustment recompute failed for reel %s", reel_id)
```

(Adapt the variable name for the reel id to `record_feedback`'s actual parameter name.)

- [ ] **Step 4: Run tests**

Run: `backend/.venv/bin/python -m pytest backend/tests/test_level_auto_adjust.py backend/tests/test_clip_engine_feed_refine_feedback.py -p no:randomly -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/reels.py backend/tests/test_level_auto_adjust.py
git commit -m "feat(level): feedback-driven auto-adjust (bounded, gated, persisted)"
```

---

### Task 10: API — create-material level, PATCH level endpoint, feed response fields

**Files:**
- Modify: `backend/app/main.py` (`create_material` line ~5663; new PATCH endpoint near it; `/api/feed` endpoint line ~6730)
- Modify: `backend/app/models.py` (`FeedResponse` line ~206; new `MaterialLevelUpdateRequest`)
- Test: `backend/tests/test_knowledge_level_api.py` (create)

**Interfaces:**
- Consumes: `normalize_knowledge_level`, `effective_level_target` (Task 2); `ranked_feed._last_effective_level_target` (Task 8).
- Produces: `POST /api/material` accepts `knowledge_level` form field; `PATCH /api/materials/{material_id}/level` body `{"knowledge_level": "..."}` → `{"knowledge_level": ..., "effective_level_target": ...}` and resets `level_adjustment` to 0; `FeedResponse.knowledge_level: str | None` and `FeedResponse.effective_level_target: float | None`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_knowledge_level_api.py
"""API contract: create-material level field, PATCH level, feed fields.
FastAPI TestClient against a temp DB (pattern from test_clip_engine_contract)."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("REELAI_INGEST_SKIP_IMPORT_SWEEP", "1")


class KnowledgeLevelApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self._prev = os.environ.get("DATA_DIR")
        os.environ["DATA_DIR"] = self.temp_dir.name
        from backend.app import db as db_module
        from backend.app.config import get_settings
        db_module._db_ready = False
        get_settings.cache_clear()
        import backend.app.main as main_module
        main_module.settings = get_settings()
        from fastapi.testclient import TestClient
        self.db = db_module
        self.client = TestClient(main_module.app)
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        from backend.app.config import get_settings
        if self._prev is None:
            os.environ.pop("DATA_DIR", None)
        else:
            os.environ["DATA_DIR"] = self._prev
        self.db._db_ready = False
        get_settings.cache_clear()

    def test_create_material_stores_level(self) -> None:
        resp = self.client.post("/api/material",
                                data={"subject_tag": "physics", "knowledge_level": "advanced"})
        self.assertEqual(resp.status_code, 200, resp.text)
        material_id = resp.json()["material_id"]
        with self.db.get_conn() as conn:
            row = self.db.fetch_one(
                conn, "SELECT knowledge_level FROM materials WHERE id = ?", (material_id,))
        self.assertEqual(row["knowledge_level"], "advanced")

    def test_create_material_default_beginner_and_invalid_422(self) -> None:
        ok = self.client.post("/api/material", data={"subject_tag": "physics"})
        self.assertEqual(ok.status_code, 200)
        mid = ok.json()["material_id"]
        with self.db.get_conn() as conn:
            row = self.db.fetch_one(conn, "SELECT knowledge_level FROM materials WHERE id = ?", (mid,))
        self.assertEqual(row["knowledge_level"], "beginner")
        bad = self.client.post("/api/material",
                               data={"subject_tag": "physics", "knowledge_level": "expert"})
        self.assertEqual(bad.status_code, 422)

    def test_patch_level_updates_and_resets_adjustment(self) -> None:
        created = self.client.post("/api/material", data={"subject_tag": "physics"})
        mid = created.json()["material_id"]
        with self.db.get_conn(transactional=True) as conn:
            conn.execute("UPDATE materials SET level_adjustment = 0.3 WHERE id = ?", (mid,))
        resp = self.client.patch(f"/api/materials/{mid}/level",
                                 json={"knowledge_level": "advanced"})
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["knowledge_level"], "advanced")
        self.assertAlmostEqual(body["effective_level_target"], 0.85)
        with self.db.get_conn() as conn:
            row = self.db.fetch_one(
                conn, "SELECT knowledge_level, level_adjustment FROM materials WHERE id = ?", (mid,))
        self.assertEqual(row["knowledge_level"], "advanced")
        self.assertEqual(float(row["level_adjustment"]), 0.0)

    def test_patch_unknown_material_404_and_bad_level_422(self) -> None:
        self.assertEqual(
            self.client.patch("/api/materials/nope/level",
                              json={"knowledge_level": "advanced"}).status_code, 404)
        created = self.client.post("/api/material", data={"subject_tag": "physics"})
        mid = created.json()["material_id"]
        self.assertEqual(
            self.client.patch(f"/api/materials/{mid}/level",
                              json={"knowledge_level": "expert"}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
```

NOTE for the implementer: check `MaterialResponse`'s actual field name for the id (`material_id` vs `id`) in `backend/app/models.py` and adjust the test accesses. If `create_material` requires auth/rate-limit context, mirror how `backend/tests/test_clip_engine_contract.py` constructs its TestClient.

- [ ] **Step 2: Run test to verify it fails**

Run: `backend/.venv/bin/python -m pytest backend/tests/test_knowledge_level_api.py -p no:randomly -q`
Expected: FAIL (level not stored; PATCH route 404s)

- [ ] **Step 3: Implement**

(a) `backend/app/models.py`:

```python
class MaterialLevelUpdateRequest(BaseModel):
    knowledge_level: Literal["beginner", "intermediate", "advanced"]
```

and `FeedResponse` gains:

```python
    knowledge_level: str | None = None
    effective_level_target: float | None = None
```

(b) `backend/app/main.py` — `create_material` signature gains `knowledge_level: str | None = Form(default=None)`; validate right after the empty-input check:

```python
    from .services.knowledge_level import normalize_knowledge_level
    try:
        normalized_level = normalize_knowledge_level(knowledge_level)
    except ValueError:
        raise HTTPException(status_code=422, detail="knowledge_level must be beginner, intermediate, or advanced")
```

and add `"knowledge_level": normalized_level,` to the materials `upsert(...)` dict (line ~5673).

(c) New endpoint after `create_material`:

```python
@app.patch("/api/materials/{material_id}/level")
def update_material_level(material_id: str, payload: MaterialLevelUpdateRequest):
    from .services.knowledge_level import effective_level_target
    with get_conn(transactional=True) as conn:
        row = fetch_one(conn, "SELECT id FROM materials WHERE id = ? LIMIT 1", (material_id,))
        if not row:
            raise HTTPException(status_code=404, detail="material not found")
        # Explicit choice supersedes accumulated drift.
        execute_modify(
            conn,
            "UPDATE materials SET knowledge_level = ?, level_adjustment = 0.0 WHERE id = ?",
            (payload.knowledge_level, material_id),
        )
    return {
        "knowledge_level": payload.knowledge_level,
        "effective_level_target": effective_level_target(payload.knowledge_level, 0.0),
    }
```

(`MaterialLevelUpdateRequest` import joins the existing models import in main.py. Pydantic's `Literal` makes invalid levels a 422 automatically.)

(d) `/api/feed` endpoint (line ~6730): after the `ranked_feed(...)` call, populate the response fields from the service:

```python
        knowledge_level=getattr(reel_service, "_last_knowledge_level", None),
        effective_level_target=getattr(reel_service, "_last_effective_level_target", None),
```

(added to the `FeedResponse(...)` construction — locate it in the endpoint body).

- [ ] **Step 4: Run tests**

Run: `backend/.venv/bin/python -m pytest backend/tests/test_knowledge_level_api.py backend/tests/test_clip_engine_contract.py -p no:randomly -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add backend/app/main.py backend/app/models.py backend/tests/test_knowledge_level_api.py
git commit -m "feat(api): knowledge_level on create-material, PATCH level endpoint, feed fields"
```

---

### Task 11: Webapp UI — level chips at creation, level pill on feed

**Files:**
- Modify: `src/lib/api.ts` (`uploadMaterial` line ~705; new `updateMaterialLevel`; feed response type)
- Modify: `src/components/UploadPanel.tsx` (topic mode form, state near line ~265)
- Modify: `src/app/feed/page.tsx` (level pill)

**Interfaces:**
- Consumes: backend endpoints from Task 10.
- Produces: `uploadMaterial({ ..., knowledgeLevel?: "beginner"|"intermediate"|"advanced" })`; `updateMaterialLevel(materialId, level)`; the feed page renders and updates the level.

- [ ] **Step 1: API helpers** (`src/lib/api.ts`)

In `uploadMaterial`'s params type add `knowledgeLevel?: "beginner" | "intermediate" | "advanced";` and next to the existing `subjectTag` append:

```typescript
  if (params.knowledgeLevel) {
    form.append("knowledge_level", params.knowledgeLevel);
  }
```

Add below `uploadMaterial`:

```typescript
export async function updateMaterialLevel(params: {
  materialId: string;
  knowledgeLevel: "beginner" | "intermediate" | "advanced";
}): Promise<{ knowledge_level: string; effective_level_target: number }> {
  const res = await fetch(apiUrl(`/api/materials/${encodeURIComponent(params.materialId)}/level`), {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ knowledge_level: params.knowledgeLevel }),
  });
  if (!res.ok) throw await buildApiError(res);
  return res.json();
}
```

(Match the file's existing fetch/error helpers — `apiUrl` and `buildApiError` are already used throughout; if requests in this file attach auth headers via a shared helper, use that same helper.)

Extend the feed response type (where the `/api/feed` response is typed in this file) with `knowledge_level?: string | null; effective_level_target?: number | null;`.

- [ ] **Step 2: UploadPanel chips** (`src/components/UploadPanel.tsx`)

Add state near the `topics` state (line ~265):

```tsx
const [knowledgeLevel, setKnowledgeLevel] = useState<"beginner" | "intermediate" | "advanced">("beginner");
```

Render inside the topic-mode form (visible whenever `inputMode === "topic"`), styled with the file's existing chip/button classes:

```tsx
<div className="flex gap-2" role="radiogroup" aria-label="How well do you know this topic?">
  {(["beginner", "intermediate", "advanced"] as const).map((level) => (
    <button
      key={level}
      type="button"
      role="radio"
      aria-checked={knowledgeLevel === level}
      onClick={() => setKnowledgeLevel(level)}
      className={knowledgeLevel === level ? "chip chip-active" : "chip"}
    >
      {level.charAt(0).toUpperCase() + level.slice(1)}
    </button>
  ))}
</div>
```

(Replace `chip`/`chip-active` with the panel's actual selected/unselected button classes — copy them from the existing mode-selector buttons in the same file.) Pass `knowledgeLevel` into the `uploadMaterial({ ... })` call in the submit handler.

- [ ] **Step 3: Feed level pill** (`src/app/feed/page.tsx`)

Where the feed response lands in state, keep `knowledge_level` + `effective_level_target`. Render a pill near the feed header:

```tsx
{knowledgeLevel ? (
  <button
    type="button"
    onClick={cycleLevel}
    className="text-xs rounded-full border px-3 py-1 opacity-80 hover:opacity-100"
    title="Tap to change your level for this topic"
  >
    {knowledgeLevel.charAt(0).toUpperCase() + knowledgeLevel.slice(1)} · auto-adjusting
  </button>
) : null}
```

with:

```tsx
async function cycleLevel() {
  const order = ["beginner", "intermediate", "advanced"] as const;
  const next = order[(order.indexOf(knowledgeLevel as any) + 1) % order.length];
  const updated = await updateMaterialLevel({ materialId, knowledgeLevel: next });
  setKnowledgeLevel(updated.knowledge_level);
  await refetchFeed();   // the page's existing feed reload function
}
```

(Adapt `materialId`/`refetchFeed` to the page's actual state/props names.)

- [ ] **Step 4: Verify**

Run: `npm run build 2>&1 | tail -5` from the repo root (or `npx tsc --noEmit` if faster) — Expected: no type errors. Then manually: create a topic with "Advanced" selected, confirm the pill shows on the feed and tapping it re-orders the feed.

- [ ] **Step 5: Commit**

```bash
git add src/lib/api.ts src/components/UploadPanel.tsx src/app/feed/page.tsx
git commit -m "feat(web): knowledge-level chips at topic creation + live level pill on feed"
```

---

### Task 12: iOS — level picker at creation, level label on feed

**Files (all under `/Users/vincentfeng/Documents/reelai app/reelai/reelai/`):**
- Modify: `APIClient.swift` (the `/api/material` multipart builder, line ~482)
- Modify: `Models.swift` (feed response struct — find the struct decoding `/api/feed`)
- Modify: `CreateView.swift` (topic creation form + `startIngestSession`, line ~522)
- Modify: `FeedView.swift` (level label)

**Interfaces:**
- Consumes: backend contract from Task 10.
- Produces: creation requests carry `knowledge_level`; feed decodes `knowledge_level`.

- [ ] **Step 1: APIClient** — the function at APIClient.swift:482 that builds the `/api/material` multipart body gains a parameter `knowledgeLevel: String? = nil`, and where it appends the `subject_tag` form field, append equivalently:

```swift
if let knowledgeLevel, !knowledgeLevel.isEmpty {
    appendFormField(name: "knowledge_level", value: knowledgeLevel)
}
```

(Match the file's actual multipart append helper — copy the exact pattern used for `subject_tag` two lines above.)

- [ ] **Step 2: Models** — in `Models.swift`, the struct that decodes the `/api/feed` response gains:

```swift
let knowledgeLevel: String?
let effectiveLevelTarget: Double?
```

with coding keys `knowledge_level` / `effective_level_target` following the file's existing CodingKeys convention.

- [ ] **Step 3: CreateView** — add state + picker in the topic-creation form:

```swift
@State private var knowledgeLevel: String = "beginner"
```

```swift
Picker("How well do you know this topic?", selection: $knowledgeLevel) {
    Text("Beginner").tag("beginner")
    Text("Intermediate").tag("intermediate")
    Text("Advanced").tag("advanced")
}
.pickerStyle(.segmented)
```

placed under the topic text field, and pass `knowledgeLevel: knowledgeLevel` through `startIngestSession()`'s call into the APIClient material upload.

- [ ] **Step 4: FeedView** — where the feed response is available, render a small read-only label (following the view's existing caption styling):

```swift
if let level = feedResponse.knowledgeLevel {
    Text("\(level.capitalized) · auto-adjusting")
        .font(.caption2)
        .foregroundStyle(.secondary)
}
```

- [ ] **Step 5: Verify**

Build the app for the simulator (`xcodebuild -project ../reelai.xcodeproj -scheme reelai -destination 'platform=iOS Simulator,name=iPhone 16' build 2>&1 | tail -3` — adjust scheme/destination to the project's actual ones, or build in Xcode). Expected: build succeeds.

- [ ] **Step 6: Commit** (the iOS folder is outside the backend git repo — check `git -C "/Users/vincentfeng/Documents/reelai app/reelai/reelai" status` first; if it is not a repo, skip the commit and note it in the task report)

```bash
git add APIClient.swift Models.swift CreateView.swift FeedView.swift
git commit -m "feat(ios): knowledge-level picker at topic creation + feed level label"
```

---

### Task 13: Practice back-port — keep the two clippers byte-identical

**Files:**
- Copy: `backend/app/clip_engine/clipper/pipeline/gemini_segment.py` → `/Users/vincentfeng/Documents/reelai app/practice/clips/backend/pipeline/gemini_segment.py`
- Copy (import-adapted): `backend/tests/clip_engine/test_gemini_segment_difficulty.py` → `/Users/vincentfeng/Documents/reelai app/practice/clips/backend/pipeline/tests/test_gemini_segment_difficulty.py`

- [ ] **Step 1: Copy the engine file and verify byte-identity**

```bash
cp "backend/app/clip_engine/clipper/pipeline/gemini_segment.py" \
   "/Users/vincentfeng/Documents/reelai app/practice/clips/backend/pipeline/gemini_segment.py"
diff "backend/app/clip_engine/clipper/pipeline/gemini_segment.py" \
     "/Users/vincentfeng/Documents/reelai app/practice/clips/backend/pipeline/gemini_segment.py" && echo IDENTICAL
```

Expected: `IDENTICAL`

- [ ] **Step 2: Copy the difficulty tests with adapted imports**

```bash
sed -e 's/backend\.app\.clip_engine\.clipper\.pipeline/backend.pipeline/g' \
  "backend/tests/clip_engine/test_gemini_segment_difficulty.py" \
  > "/Users/vincentfeng/Documents/reelai app/practice/clips/backend/pipeline/tests/test_gemini_segment_difficulty.py"
```

- [ ] **Step 3: Run the practice suite**

Run: `cd "/Users/vincentfeng/Documents/reelai app/practice/clips" && .venv/bin/python -m pytest backend/pipeline/tests -p no:randomly -q`
Expected: all pass (existing + new difficulty tests)

- [ ] **Step 4: Commit** (practice folder — check `git -C "/Users/vincentfeng/Documents/reelai app/practice" status` first; if not a repo, skip and note)

---

### Task 14: Full verification battery + services restart

- [ ] **Step 1: Isolated backend suites**

Run from the repo root:

```bash
backend/.venv/bin/python -m pytest backend/tests/clip_engine \
  backend/tests/test_knowledge_level_migrations.py \
  backend/tests/test_knowledge_level_helpers.py \
  backend/tests/test_level_auto_adjust.py \
  backend/tests/test_knowledge_level_api.py \
  backend/tests/test_reels_concept_topic.py \
  backend/tests/test_feed_chain_integrity.py -p no:randomly -q
for f in test_clip_engine_material_topic test_clip_engine_generate_reels test_clip_engine_contract \
         test_clip_engine_topic_cut test_clip_engine_ingest_url test_clip_engine_can_generate \
         test_clip_engine_feed_refine_feedback test_ingestion_url test_clip_engine_search test_clip_engine_feed; do
  backend/.venv/bin/python -m pytest backend/tests/$f.py -p no:randomly -q | tail -1
done
```

Expected: all pass.

- [ ] **Step 2: Full aggregate — compare against the 21-failure baseline**

Run: `backend/.venv/bin/python -m pytest backend/tests -p no:randomly -q 2>&1 | tail -1`
Expected: failures only in the known community/medium files (order pollution); any NEW failing file = investigate before proceeding.

- [ ] **Step 3: Graphify rebuild** (project rule)

Run from `/Users/vincentfeng/Documents/reelai app`:
`python3 -c "from graphify.watch import _rebuild_code; from pathlib import Path; _rebuild_code(Path('.'))"`

- [ ] **Step 4: Restart the local backend so the feature is live**

```bash
kill $(lsof -ti :8001) 2>/dev/null; sleep 1
cd "/Users/vincentfeng/Documents/reelai app/reelai/reelAI copy 2" && bash -c 'set -a; source backend/.env; set +a; export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false KMP_DUPLICATE_LIB_OK=TRUE; nohup backend/.venv/bin/python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8001 > .logs/backend-8001.log 2>&1 &'
sleep 5 && curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8001/docs
```

Expected: `200`. (This backend process is agent-started in this session, so restarting it is permitted.)

- [ ] **Step 5: Manual smoke** — create a topic as "Beginner" in the webapp (localhost:3001), generate, confirm the feed leads with intro-style clips and the pill shows "Beginner · auto-adjusting"; tap the pill twice to Advanced and confirm the feed re-orders without regeneration.
