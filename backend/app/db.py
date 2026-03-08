import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterable

from .config import get_settings


SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS materials (
    id TEXT PRIMARY KEY,
    subject_tag TEXT,
    raw_text TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_path TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS concepts (
    id TEXT PRIMARY KEY,
    material_id TEXT NOT NULL,
    title TEXT NOT NULL,
    keywords_json TEXT NOT NULL,
    summary TEXT NOT NULL,
    embedding_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(material_id) REFERENCES materials(id)
);

CREATE INDEX IF NOT EXISTS idx_concepts_material_id ON concepts(material_id);

CREATE TABLE IF NOT EXISTS material_chunks (
    id TEXT PRIMARY KEY,
    material_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(material_id) REFERENCES materials(id)
);

CREATE INDEX IF NOT EXISTS idx_material_chunks_material_id ON material_chunks(material_id);

CREATE TABLE IF NOT EXISTS videos (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    channel_title TEXT,
    description TEXT,
    duration_sec INTEGER,
    view_count INTEGER DEFAULT 0,
    is_creative_commons INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS transcript_chunks (
    id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    t_start REAL NOT NULL,
    t_end REAL NOT NULL,
    text TEXT NOT NULL,
    embedding_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(video_id) REFERENCES videos(id)
);

CREATE INDEX IF NOT EXISTS idx_transcript_chunks_video_id ON transcript_chunks(video_id);

CREATE TABLE IF NOT EXISTS reels (
    id TEXT PRIMARY KEY,
    material_id TEXT NOT NULL,
    concept_id TEXT NOT NULL,
    video_id TEXT NOT NULL,
    video_url TEXT NOT NULL,
    t_start REAL NOT NULL,
    t_end REAL NOT NULL,
    transcript_snippet TEXT NOT NULL,
    takeaways_json TEXT NOT NULL,
    base_score REAL NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(material_id) REFERENCES materials(id),
    FOREIGN KEY(concept_id) REFERENCES concepts(id),
    FOREIGN KEY(video_id) REFERENCES videos(id)
);

CREATE INDEX IF NOT EXISTS idx_reels_material_id ON reels(material_id);
CREATE INDEX IF NOT EXISTS idx_reels_concept_id ON reels(concept_id);
-- Keep only one reel per exact (material, video, clip window) and enforce it going forward.
DELETE FROM reels
WHERE rowid NOT IN (
    SELECT MIN(rowid)
    FROM reels
    GROUP BY material_id, video_id, CAST(t_start AS INTEGER), CAST(t_end AS INTEGER)
);
DROP INDEX IF EXISTS idx_reels_material_video_unique;
CREATE UNIQUE INDEX IF NOT EXISTS idx_reels_material_video_clip_unique ON reels(material_id, video_id, t_start, t_end);

CREATE TABLE IF NOT EXISTS reel_feedback (
    id TEXT PRIMARY KEY,
    reel_id TEXT NOT NULL,
    helpful INTEGER NOT NULL DEFAULT 0,
    confusing INTEGER NOT NULL DEFAULT 0,
    rating INTEGER,
    saved INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY(reel_id) REFERENCES reels(id)
);

CREATE INDEX IF NOT EXISTS idx_reel_feedback_reel_id ON reel_feedback(reel_id);

CREATE TABLE IF NOT EXISTS retrieval_runs (
    id TEXT PRIMARY KEY,
    material_id TEXT NOT NULL,
    concept_id TEXT NOT NULL,
    concept_title TEXT NOT NULL,
    selected_video_id TEXT,
    failure_reason TEXT NOT NULL DEFAULT '',
    debug_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY(material_id) REFERENCES materials(id),
    FOREIGN KEY(concept_id) REFERENCES concepts(id)
);

CREATE INDEX IF NOT EXISTS idx_retrieval_runs_material_concept ON retrieval_runs(material_id, concept_id, created_at DESC);

CREATE TABLE IF NOT EXISTS retrieval_queries (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    query_text TEXT NOT NULL,
    strategy TEXT NOT NULL,
    stage TEXT NOT NULL,
    source_surface TEXT NOT NULL DEFAULT '',
    source_terms_json TEXT NOT NULL DEFAULT '[]',
    weight REAL NOT NULL DEFAULT 0.0,
    result_count INTEGER NOT NULL DEFAULT 0,
    kept_count INTEGER NOT NULL DEFAULT 0,
    position INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES retrieval_runs(id)
);

CREATE INDEX IF NOT EXISTS idx_retrieval_queries_run_stage ON retrieval_queries(run_id, stage, strategy);

CREATE TABLE IF NOT EXISTS retrieval_candidates (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    video_id TEXT NOT NULL,
    video_title TEXT NOT NULL DEFAULT '',
    channel_title TEXT NOT NULL DEFAULT '',
    strategy TEXT NOT NULL DEFAULT '',
    stage TEXT NOT NULL DEFAULT '',
    query_text TEXT NOT NULL DEFAULT '',
    source_surface TEXT NOT NULL DEFAULT '',
    discovery_score REAL NOT NULL DEFAULT 0.0,
    clipability_score REAL NOT NULL DEFAULT 0.0,
    final_score REAL NOT NULL DEFAULT 0.0,
    feature_json TEXT NOT NULL DEFAULT '{}',
    position INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES retrieval_runs(id)
);

CREATE INDEX IF NOT EXISTS idx_retrieval_candidates_run_position ON retrieval_candidates(run_id, position);

CREATE TABLE IF NOT EXISTS retrieval_outcomes (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    video_id TEXT NOT NULL,
    t_start REAL NOT NULL DEFAULT 0,
    t_end REAL NOT NULL DEFAULT 0,
    reason_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES retrieval_runs(id),
    FOREIGN KEY(video_id) REFERENCES videos(id)
);

CREATE INDEX IF NOT EXISTS idx_retrieval_outcomes_run ON retrieval_outcomes(run_id, created_at DESC);

CREATE TABLE IF NOT EXISTS community_sets (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    tags_json TEXT NOT NULL,
    reels_json TEXT NOT NULL,
    reel_count INTEGER NOT NULL DEFAULT 0,
    curator TEXT NOT NULL,
    likes INTEGER NOT NULL DEFAULT 0,
    learners INTEGER NOT NULL DEFAULT 1,
    updated_label TEXT NOT NULL,
    thumbnail_url TEXT NOT NULL,
    featured INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_community_sets_featured_created_at ON community_sets(featured DESC, created_at DESC);

CREATE TABLE IF NOT EXISTS search_cache (
    cache_key TEXT PRIMARY KEY,
    response_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS transcript_cache (
    video_id TEXT PRIMARY KEY,
    transcript_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS embedding_cache (
    text_hash TEXT PRIMARY KEY,
    embedding_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS llm_cache (
    cache_key TEXT PRIMARY KEY,
    response_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

_db_init_lock = threading.Lock()
_db_ready = False


def _db_path() -> str:
    settings = get_settings()
    os.makedirs(settings.data_dir, exist_ok=True)
    return os.path.join(settings.data_dir, "studyreels.db")


def init_db() -> None:
    global _db_ready
    with sqlite3.connect(_db_path()) as conn:
        conn.executescript(SCHEMA)
        # Lightweight schema migrations for existing local databases.
        try:
            conn.execute("ALTER TABLE videos ADD COLUMN view_count INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass
        conn.commit()
    _db_ready = True


def ensure_db_initialized() -> None:
    global _db_ready
    if _db_ready:
        return
    with _db_init_lock:
        if _db_ready:
            return
        init_db()


@contextmanager
def get_conn():
    ensure_db_initialized()
    conn = sqlite3.connect(_db_path(), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000;")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def upsert(conn: sqlite3.Connection, table: str, data: dict[str, Any], pk: str = "id") -> None:
    cols = list(data.keys())
    placeholders = ", ".join(["?"] * len(cols))
    assignments = ", ".join([f"{c}=excluded.{c}" for c in cols if c != pk])
    sql = (
        f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT({pk}) DO UPDATE SET {assignments}"
    )
    values = [data[c] for c in cols]
    for attempt in range(4):
        try:
            conn.execute(sql, values)
            return
        except sqlite3.OperationalError as exc:
            # WAL can transiently lock under concurrent writes; retry quickly.
            if "locked" not in str(exc).lower() or attempt >= 3:
                raise
            time.sleep(0.03 * (attempt + 1))


def fetch_all(conn: sqlite3.Connection, query: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
    rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def fetch_one(conn: sqlite3.Connection, query: str, params: Iterable[Any] = ()) -> dict[str, Any] | None:
    row = conn.execute(query, params).fetchone()
    return dict(row) if row else None


def dumps_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True)


def loads_json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default
