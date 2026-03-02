import json
import os
import sqlite3
import threading
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
    conn.execute(sql, [data[c] for c in cols])


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
