import json
import os
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterable

from .config import get_settings

try:
    import psycopg
    from psycopg import errors as pg_errors
    from psycopg.rows import dict_row
except Exception:  # pragma: no cover - optional dependency at runtime
    psycopg = None
    pg_errors = None
    dict_row = None


class DatabaseIntegrityError(Exception):
    pass


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
    owner_key_hash TEXT,
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
DEFAULT_SQLITE_BUSY_TIMEOUT_MS = 120000
_SAFE_SQL_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _db_path() -> str:
    settings = get_settings()
    os.makedirs(settings.data_dir, exist_ok=True)
    return os.path.join(settings.data_dir, "studyreels.db")


def _sqlite_busy_timeout_ms() -> int:
    raw = os.getenv("SQLITE_BUSY_TIMEOUT_MS", "").strip()
    if not raw:
        return DEFAULT_SQLITE_BUSY_TIMEOUT_MS
    try:
        parsed = int(raw)
    except ValueError:
        return DEFAULT_SQLITE_BUSY_TIMEOUT_MS
    return max(1000, min(600000, parsed))


def _database_url() -> str:
    settings = get_settings()
    raw = (settings.database_url or "").strip() or os.getenv("DATABASE_URL", "").strip()
    if raw.startswith("postgres://"):
        raw = "postgresql://" + raw[len("postgres://") :]
    return raw


def _is_postgres_configured() -> bool:
    url = _database_url()
    return url.startswith("postgresql://")


def _is_postgres_conn(conn: Any) -> bool:
    if psycopg is None:
        return False
    return conn.__class__.__module__.startswith("psycopg")


def _adapt_query_for_postgres(query: str) -> str:
    result: list[str] = []
    i = 0
    in_single_quote = False
    in_double_quote = False
    in_line_comment = False
    in_block_comment = False
    while i < len(query):
        char = query[i]
        nxt = query[i + 1] if i + 1 < len(query) else ""

        if in_line_comment:
            result.append(char)
            if char == "\n":
                in_line_comment = False
            i += 1
            continue

        if in_block_comment:
            result.append(char)
            if char == "*" and nxt == "/":
                result.append(nxt)
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue

        if not in_single_quote and not in_double_quote:
            if char == "-" and nxt == "-":
                result.append(char)
                result.append(nxt)
                in_line_comment = True
                i += 2
                continue
            if char == "/" and nxt == "*":
                result.append(char)
                result.append(nxt)
                in_block_comment = True
                i += 2
                continue
            if char == "?":
                result.append("%s")
                i += 1
                continue

        result.append(char)
        if char == "'" and not in_double_quote:
            if in_single_quote and nxt == "'":
                result.append(nxt)
                i += 2
                continue
            in_single_quote = not in_single_quote
        elif char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
        i += 1
    return "".join(result)


def _strip_sql_line_comments(statement: str) -> str:
    lines = [line for line in statement.splitlines() if not line.strip().startswith("--")]
    return "\n".join(lines).strip()


def _postgres_schema_statements() -> list[str]:
    statements: list[str] = []
    for chunk in SCHEMA.split(";"):
        stmt = _strip_sql_line_comments(chunk)
        if not stmt:
            continue
        upper = stmt.upper()
        if upper.startswith("PRAGMA "):
            continue
        if re.search(r"\bROWID\b", upper):
            continue
        statements.append(stmt)
    return statements


def _migrate_reel_feedback_uniqueness_sqlite(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        DELETE FROM reel_feedback
        WHERE EXISTS (
            SELECT 1
            FROM reel_feedback AS newer
            WHERE newer.reel_id = reel_feedback.reel_id
              AND (
                  newer.created_at > reel_feedback.created_at
                  OR (newer.created_at = reel_feedback.created_at AND newer.rowid > reel_feedback.rowid)
              )
        )
        """
    )
    conn.execute("DROP INDEX IF EXISTS idx_reel_feedback_reel_id")
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_reel_feedback_reel_id_unique ON reel_feedback(reel_id)")


def _migrate_reel_feedback_uniqueness_postgres(conn: Any) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            DELETE FROM reel_feedback AS older
            USING reel_feedback AS newer
            WHERE older.reel_id = newer.reel_id
              AND (
                  older.created_at < newer.created_at
                  OR (older.created_at = newer.created_at AND older.ctid < newer.ctid)
              )
            """
        )
        cur.execute("DROP INDEX IF EXISTS idx_reel_feedback_reel_id")
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_reel_feedback_reel_id_unique ON reel_feedback(reel_id)")


def _migrate_reels_unique_clip_index_sqlite(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        DELETE FROM reels
        WHERE rowid NOT IN (
            SELECT MIN(rowid)
            FROM reels
            GROUP BY material_id, video_id, t_start, t_end
        )
        """
    )
    conn.execute("DROP INDEX IF EXISTS idx_reels_material_video_unique")
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_reels_material_video_clip_unique ON reels(material_id, video_id, t_start, t_end)")


def _migrate_reels_unique_clip_index_postgres(conn: Any) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            DELETE FROM reels AS older
            USING reels AS newer
            WHERE older.material_id = newer.material_id
              AND older.video_id = newer.video_id
              AND older.t_start = newer.t_start
              AND older.t_end = newer.t_end
              AND older.ctid < newer.ctid
            """
        )
        cur.execute("DROP INDEX IF EXISTS idx_reels_material_video_unique")
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_reels_material_video_clip_unique ON reels(material_id, video_id, t_start, t_end)")


def init_db() -> None:
    global _db_ready
    if _is_postgres_configured():
        if psycopg is None:
            raise RuntimeError("DATABASE_URL is set to PostgreSQL, but psycopg is not installed.")
        with psycopg.connect(_database_url(), connect_timeout=10) as conn:
            with conn.cursor() as cur:
                for statement in _postgres_schema_statements():
                    cur.execute(statement)
                # Lightweight schema migration for older community_sets tables.
                cur.execute("ALTER TABLE community_sets ADD COLUMN IF NOT EXISTS updated_at TEXT")
                cur.execute("ALTER TABLE community_sets ADD COLUMN IF NOT EXISTS owner_key_hash TEXT")
                cur.execute("UPDATE community_sets SET updated_at = created_at WHERE updated_at IS NULL OR BTRIM(updated_at) = ''")
            _migrate_reels_unique_clip_index_postgres(conn)
            _migrate_reel_feedback_uniqueness_postgres(conn)
            conn.commit()
        _db_ready = True
        return

    with sqlite3.connect(_db_path()) as conn:
        conn.executescript(SCHEMA)
        # Lightweight schema migration for existing local databases.
        try:
            conn.execute("ALTER TABLE videos ADD COLUMN view_count INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE community_sets ADD COLUMN updated_at TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE community_sets ADD COLUMN owner_key_hash TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("UPDATE community_sets SET updated_at = created_at WHERE updated_at IS NULL OR TRIM(updated_at) = ''")
        except sqlite3.OperationalError:
            pass
        _migrate_reels_unique_clip_index_sqlite(conn)
        _migrate_reel_feedback_uniqueness_sqlite(conn)
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
def get_conn(*, transactional: bool = False):
    ensure_db_initialized()
    if _is_postgres_configured():
        if psycopg is None:
            raise RuntimeError("DATABASE_URL is set to PostgreSQL, but psycopg is not installed.")
        conn = psycopg.connect(_database_url(), connect_timeout=15)
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise
        else:
            conn.commit()
        finally:
            conn.close()
        return

    busy_timeout_ms = _sqlite_busy_timeout_ms()
    conn = sqlite3.connect(
        _db_path(),
        timeout=max(1.0, busy_timeout_ms / 1000.0),
        isolation_level="" if transactional else None,
    )
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {busy_timeout_ms};")
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    try:
        yield conn
    except Exception:
        if transactional:
            conn.rollback()
        raise
    else:
        if transactional:
            conn.commit()
    finally:
        conn.close()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_unique_violation(exc: Exception) -> bool:
    if pg_errors is not None and isinstance(exc, pg_errors.UniqueViolation):
        return True
    return getattr(exc, "sqlstate", "") == "23505"


def upsert(conn: Any, table: str, data: dict[str, Any], pk: str = "id") -> None:
    if not _SAFE_SQL_IDENTIFIER_RE.fullmatch(table):
        raise ValueError(f"Unsafe table name: {table!r}")
    if not _SAFE_SQL_IDENTIFIER_RE.fullmatch(pk):
        raise ValueError(f"Unsafe primary key name: {pk!r}")
    cols = list(data.keys())
    for col in cols:
        if not _SAFE_SQL_IDENTIFIER_RE.fullmatch(col):
            raise ValueError(f"Unsafe column name: {col!r}")
    placeholders = ", ".join(["?"] * len(cols))
    assignments = ", ".join([f"{c}=excluded.{c}" for c in cols if c != pk])
    if assignments:
        sql = (
            f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders}) "
            f"ON CONFLICT({pk}) DO UPDATE SET {assignments}"
        )
    else:
        sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders}) ON CONFLICT({pk}) DO NOTHING"

    values = [data[c] for c in cols]
    is_pg = _is_postgres_conn(conn)
    query = _adapt_query_for_postgres(sql) if is_pg else sql

    for attempt in range(8):
        try:
            if is_pg:
                with conn.cursor() as cur:
                    if getattr(conn, "autocommit", False):
                        cur.execute(query, values)
                    else:
                        cur.execute("SAVEPOINT studyreels_upsert")
                        try:
                            cur.execute(query, values)
                        except Exception:
                            cur.execute("ROLLBACK TO SAVEPOINT studyreels_upsert")
                            cur.execute("RELEASE SAVEPOINT studyreels_upsert")
                            raise
                        else:
                            cur.execute("RELEASE SAVEPOINT studyreels_upsert")
            else:
                conn.execute(query, values)
            return
        except sqlite3.OperationalError as exc:
            # WAL can transiently lock under concurrent writes; retry quickly.
            if "locked" not in str(exc).lower() or attempt >= 7:
                raise
            try:
                conn.rollback()
            except Exception:
                pass
            # Exponential-ish backoff helps during concurrent material ingestion.
            time.sleep(min(2.0, 0.05 * (2 ** attempt)))
        except Exception as exc:
            if is_pg and _is_unique_violation(exc):
                raise DatabaseIntegrityError(str(exc)) from exc
            raise


def fetch_all(conn: Any, query: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
    if _is_postgres_conn(conn):
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(_adapt_query_for_postgres(query), tuple(params))
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    rows = conn.execute(query, tuple(params)).fetchall()
    return [dict(r) for r in rows]


def fetch_one(conn: Any, query: str, params: Iterable[Any] = ()) -> dict[str, Any] | None:
    if _is_postgres_conn(conn):
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(_adapt_query_for_postgres(query), tuple(params))
            row = cur.fetchone()
        return dict(row) if row else None

    row = conn.execute(query, tuple(params)).fetchone()
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
