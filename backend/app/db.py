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

CREATE TABLE IF NOT EXISTS reel_generations (
    id TEXT PRIMARY KEY,
    material_id TEXT NOT NULL,
    concept_id TEXT,
    request_key TEXT NOT NULL,
    generation_mode TEXT NOT NULL DEFAULT 'fast',
    retrieval_profile TEXT NOT NULL DEFAULT 'bootstrap',
    status TEXT NOT NULL DEFAULT 'pending',
    source_generation_id TEXT,
    reel_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    activated_at TEXT,
    error_text TEXT,
    FOREIGN KEY(material_id) REFERENCES materials(id),
    FOREIGN KEY(concept_id) REFERENCES concepts(id)
);

CREATE INDEX IF NOT EXISTS idx_reel_generations_material_request_created
ON reel_generations(material_id, request_key, created_at DESC);

CREATE TABLE IF NOT EXISTS reel_generation_heads (
    id TEXT PRIMARY KEY,
    material_id TEXT NOT NULL,
    request_key TEXT NOT NULL,
    active_generation_id TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(material_id) REFERENCES materials(id)
);

CREATE INDEX IF NOT EXISTS idx_reel_generation_heads_active
ON reel_generation_heads(active_generation_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_reel_generation_heads_material_request
ON reel_generation_heads(material_id, request_key);

CREATE TABLE IF NOT EXISTS reel_generation_jobs (
    id TEXT PRIMARY KEY,
    material_id TEXT NOT NULL,
    concept_id TEXT,
    request_key TEXT NOT NULL,
    source_generation_id TEXT NOT NULL,
    result_generation_id TEXT,
    target_profile TEXT NOT NULL DEFAULT 'deep',
    request_params_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'queued',
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    error_text TEXT,
    FOREIGN KEY(material_id) REFERENCES materials(id),
    FOREIGN KEY(concept_id) REFERENCES concepts(id)
);

CREATE INDEX IF NOT EXISTS idx_reel_generation_jobs_material_request_status
ON reel_generation_jobs(material_id, request_key, status, created_at DESC);

CREATE TABLE IF NOT EXISTS reels (
    id TEXT PRIMARY KEY,
    generation_id TEXT,
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

CREATE TABLE IF NOT EXISTS request_frontier_entries (
    id TEXT PRIMARY KEY,
    material_id TEXT NOT NULL,
    request_key TEXT NOT NULL,
    family_key TEXT NOT NULL,
    stage TEXT NOT NULL DEFAULT '',
    query_text TEXT NOT NULL DEFAULT '',
    source_family TEXT NOT NULL DEFAULT '',
    seed_video_id TEXT NOT NULL DEFAULT '',
    seed_channel_id TEXT NOT NULL DEFAULT '',
    anchor_mode TEXT NOT NULL DEFAULT '',
    runs INTEGER NOT NULL DEFAULT 0,
    new_good_videos INTEGER NOT NULL DEFAULT 0,
    new_accepted_reels INTEGER NOT NULL DEFAULT 0,
    new_visible_reels INTEGER NOT NULL DEFAULT 0,
    duplicate_rate REAL NOT NULL DEFAULT 0.0,
    off_topic_rate REAL NOT NULL DEFAULT 0.0,
    last_run_at TEXT,
    cooldown_until TEXT,
    exhausted INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(material_id) REFERENCES materials(id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_request_frontier_entries_material_request_family
ON request_frontier_entries(material_id, request_key, family_key);

CREATE INDEX IF NOT EXISTS idx_request_frontier_entries_request_stage
ON request_frontier_entries(material_id, request_key, stage, exhausted, updated_at DESC);

CREATE TABLE IF NOT EXISTS request_video_mining_state (
    id TEXT PRIMARY KEY,
    material_id TEXT NOT NULL,
    request_key TEXT NOT NULL,
    video_id TEXT NOT NULL,
    mining_state TEXT NOT NULL DEFAULT 'unmined',
    quality_tier TEXT NOT NULL DEFAULT '',
    transcript_fetched INTEGER NOT NULL DEFAULT 0,
    windows_scanned INTEGER NOT NULL DEFAULT 0,
    clusters_mined INTEGER NOT NULL DEFAULT 0,
    accepted_clip_count INTEGER NOT NULL DEFAULT 0,
    visible_clip_count INTEGER NOT NULL DEFAULT 0,
    remaining_spans_json TEXT NOT NULL DEFAULT '[]',
    last_mined_at TEXT,
    exhausted INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(material_id) REFERENCES materials(id),
    FOREIGN KEY(video_id) REFERENCES videos(id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_request_video_mining_state_material_request_video
ON request_video_mining_state(material_id, request_key, video_id);

CREATE INDEX IF NOT EXISTS idx_request_video_mining_state_request_status
ON request_video_mining_state(material_id, request_key, mining_state, exhausted, updated_at DESC);

CREATE TABLE IF NOT EXISTS community_accounts (
    id TEXT PRIMARY KEY,
    username TEXT NOT NULL,
    username_normalized TEXT NOT NULL UNIQUE,
    email TEXT,
    email_normalized TEXT,
    password_hash TEXT NOT NULL,
    password_salt TEXT NOT NULL,
    verified_at TEXT,
    verification_code_hash TEXT,
    verification_expires_at TEXT,
    legacy_claim_owner_key_hash TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS community_signup_email_verifications (
    id TEXT PRIMARY KEY,
    owner_key_hash TEXT NOT NULL,
    email TEXT NOT NULL,
    email_normalized TEXT NOT NULL,
    verified_at TEXT,
    verification_code_hash TEXT,
    verification_expires_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_community_signup_email_verifications_owner_email
ON community_signup_email_verifications(owner_key_hash, email_normalized);

CREATE INDEX IF NOT EXISTS idx_community_signup_email_verifications_email_updated_at
ON community_signup_email_verifications(email_normalized, updated_at DESC);

CREATE TABLE IF NOT EXISTS community_sessions (
    id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL,
    token_hash TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    last_used_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    FOREIGN KEY(account_id) REFERENCES community_accounts(id)
);

CREATE INDEX IF NOT EXISTS idx_community_sessions_account_id ON community_sessions(account_id);

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
    owner_account_id TEXT,
    visibility TEXT NOT NULL DEFAULT 'public',
    featured INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_community_sets_featured_created_at ON community_sets(featured DESC, created_at DESC);

CREATE TABLE IF NOT EXISTS community_material_history (
    account_id TEXT NOT NULL,
    material_id TEXT NOT NULL,
    title TEXT NOT NULL,
    updated_at BIGINT NOT NULL,
    starred INTEGER NOT NULL DEFAULT 0,
    generation_mode TEXT NOT NULL DEFAULT 'fast',
    source TEXT NOT NULL DEFAULT 'search',
    feed_query TEXT,
    active_index INTEGER,
    active_reel_id TEXT,
    PRIMARY KEY(account_id, material_id),
    FOREIGN KEY(account_id) REFERENCES community_accounts(id)
);

CREATE INDEX IF NOT EXISTS idx_community_material_history_account_updated_at
ON community_material_history(account_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS community_account_settings (
    account_id TEXT PRIMARY KEY,
    generation_mode TEXT NOT NULL DEFAULT 'fast',
    default_input_mode TEXT NOT NULL DEFAULT 'source',
    min_relevance_threshold REAL NOT NULL DEFAULT 0.3,
    start_muted INTEGER NOT NULL DEFAULT 1,
    video_pool_mode TEXT NOT NULL DEFAULT 'short-first',
    preferred_video_duration TEXT NOT NULL DEFAULT 'any',
    target_clip_duration_sec INTEGER NOT NULL DEFAULT 55,
    target_clip_duration_min_sec INTEGER NOT NULL DEFAULT 20,
    target_clip_duration_max_sec INTEGER NOT NULL DEFAULT 55,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(account_id) REFERENCES community_accounts(id)
);

CREATE TABLE IF NOT EXISTS rate_limit_events (
    id TEXT PRIMARY KEY,
    rate_key TEXT NOT NULL,
    hit_at REAL NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_rate_limit_events_key_hit_at ON rate_limit_events(rate_key, hit_at);
CREATE INDEX IF NOT EXISTS idx_rate_limit_events_hit_at ON rate_limit_events(hit_at);

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

CREATE TABLE IF NOT EXISTS ranked_feed_cache (
    cache_key TEXT PRIMARY KEY,
    material_id TEXT NOT NULL,
    generation_id TEXT NOT NULL DEFAULT '',
    fast_mode INTEGER NOT NULL DEFAULT 0,
    source_fingerprint TEXT NOT NULL,
    response_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ranked_feed_cache_material_generation
ON ranked_feed_cache(material_id, generation_id, fast_mode);

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
LEGACY_COMMUNITY_OWNER_HASH = "__legacy_unowned__"
DEFAULT_COMMUNITY_VISIBILITY = "public"
PUBLIC_COMMUNITY_VISIBILITY = "public"


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
            GROUP BY material_id, COALESCE(generation_id, ''), video_id, t_start, t_end
        )
        """
    )
    conn.execute("DROP INDEX IF EXISTS idx_reels_material_video_unique")
    conn.execute("DROP INDEX IF EXISTS idx_reels_material_video_clip_unique")
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_reels_material_video_clip_unique
        ON reels(material_id, COALESCE(generation_id, ''), video_id, t_start, t_end)
        """
    )


def _migrate_reels_unique_clip_index_postgres(conn: Any) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            DELETE FROM reels AS older
            USING reels AS newer
            WHERE older.material_id = newer.material_id
              AND COALESCE(older.generation_id, '') = COALESCE(newer.generation_id, '')
              AND older.video_id = newer.video_id
              AND older.t_start = newer.t_start
              AND older.t_end = newer.t_end
              AND older.ctid < newer.ctid
            """
        )
        cur.execute("DROP INDEX IF EXISTS idx_reels_material_video_unique")
        cur.execute("DROP INDEX IF EXISTS idx_reels_material_video_clip_unique")
        cur.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_reels_material_video_clip_unique
            ON reels(material_id, COALESCE(generation_id, ''), video_id, t_start, t_end)
            """
        )


def _ensure_reels_generation_index_sqlite(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE INDEX IF NOT EXISTS idx_reels_generation_id ON reels(generation_id)")


def _ensure_reels_generation_index_postgres(conn: Any) -> None:
    with conn.cursor() as cur:
        cur.execute("CREATE INDEX IF NOT EXISTS idx_reels_generation_id ON reels(generation_id)")


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
                cur.execute("ALTER TABLE community_accounts ADD COLUMN IF NOT EXISTS email TEXT")
                cur.execute("ALTER TABLE community_accounts ADD COLUMN IF NOT EXISTS email_normalized TEXT")
                cur.execute("ALTER TABLE community_accounts ADD COLUMN IF NOT EXISTS verified_at TEXT")
                cur.execute("ALTER TABLE community_accounts ADD COLUMN IF NOT EXISTS verification_code_hash TEXT")
                cur.execute("ALTER TABLE community_accounts ADD COLUMN IF NOT EXISTS verification_expires_at TEXT")
                cur.execute("ALTER TABLE community_accounts ADD COLUMN IF NOT EXISTS legacy_claim_owner_key_hash TEXT")
                cur.execute("ALTER TABLE reels ADD COLUMN IF NOT EXISTS generation_id TEXT")
                cur.execute("ALTER TABLE reel_generation_jobs ADD COLUMN IF NOT EXISTS request_params_json TEXT NOT NULL DEFAULT '{}'")
                _ensure_reels_generation_index_postgres(conn)
                cur.execute(
                    "UPDATE community_accounts SET email = NULL WHERE email IS NOT NULL AND BTRIM(email) = ''"
                )
                cur.execute(
                    "UPDATE community_accounts SET email_normalized = NULL WHERE email_normalized IS NOT NULL AND BTRIM(email_normalized) = ''"
                )
                cur.execute(
                    """
                    UPDATE community_accounts
                    SET verified_at = created_at
                    WHERE (email IS NULL OR BTRIM(email) = '')
                      AND (verified_at IS NULL OR BTRIM(verified_at) = '')
                    """
                )
                cur.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS idx_community_accounts_email_normalized_unique ON community_accounts(email_normalized)"
                )
                cur.execute("ALTER TABLE community_sets ADD COLUMN IF NOT EXISTS updated_at TEXT")
                cur.execute("ALTER TABLE community_sets ADD COLUMN IF NOT EXISTS owner_key_hash TEXT")
                cur.execute("ALTER TABLE community_sets ADD COLUMN IF NOT EXISTS owner_account_id TEXT")
                cur.execute("ALTER TABLE community_sets ADD COLUMN IF NOT EXISTS visibility TEXT")
                cur.execute("ALTER TABLE community_material_history ADD COLUMN IF NOT EXISTS active_index INTEGER")
                cur.execute("ALTER TABLE community_material_history ADD COLUMN IF NOT EXISTS active_reel_id TEXT")
                cur.execute("UPDATE community_sets SET updated_at = created_at WHERE updated_at IS NULL OR BTRIM(updated_at) = ''")
                cur.execute(
                    """
                    UPDATE community_sets
                    SET owner_key_hash = %s
                    WHERE owner_key_hash IS NULL OR BTRIM(owner_key_hash) = ''
                    """,
                    (LEGACY_COMMUNITY_OWNER_HASH,),
                )
                cur.execute(
                    """
                    UPDATE community_sets
                    SET visibility = CASE
                        WHEN featured = 1 OR owner_key_hash = %s THEN %s
                        ELSE %s
                    END
                    WHERE visibility IS NULL OR BTRIM(visibility) = ''
                    """,
                    (LEGACY_COMMUNITY_OWNER_HASH, PUBLIC_COMMUNITY_VISIBILITY, DEFAULT_COMMUNITY_VISIBILITY),
                )
                cur.execute(
                    """
                    UPDATE community_sets
                    SET visibility = %s
                    WHERE owner_account_id IS NOT NULL
                      AND (
                          visibility IS NULL
                          OR BTRIM(visibility) = ''
                          OR visibility = 'private'
                      )
                    """,
                    (PUBLIC_COMMUNITY_VISIBILITY,),
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_community_sets_owner_account_updated_at ON community_sets(owner_account_id, updated_at DESC)"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_community_sets_visibility_featured_updated_at ON community_sets(visibility, featured DESC, updated_at DESC)"
                )
            _migrate_reels_unique_clip_index_postgres(conn)
            _migrate_reel_feedback_uniqueness_postgres(conn)
            conn.commit()
        _db_ready = True
        return

    conn = sqlite3.connect(_db_path())
    try:
        conn.executescript(SCHEMA)
        # Lightweight schema migration for existing local databases.
        try:
            conn.execute("ALTER TABLE videos ADD COLUMN view_count INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE community_accounts ADD COLUMN email TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE community_accounts ADD COLUMN email_normalized TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE community_accounts ADD COLUMN verified_at TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE community_accounts ADD COLUMN verification_code_hash TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE community_accounts ADD COLUMN verification_expires_at TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE community_accounts ADD COLUMN legacy_claim_owner_key_hash TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE reels ADD COLUMN generation_id TEXT")
        except sqlite3.OperationalError:
            pass
        _ensure_reels_generation_index_sqlite(conn)
        try:
            conn.execute("ALTER TABLE reel_generation_jobs ADD COLUMN request_params_json TEXT NOT NULL DEFAULT '{}'")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("UPDATE community_accounts SET email = NULL WHERE email IS NOT NULL AND TRIM(email) = ''")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute(
                "UPDATE community_accounts SET email_normalized = NULL WHERE email_normalized IS NOT NULL AND TRIM(email_normalized) = ''"
            )
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute(
                """
                UPDATE community_accounts
                SET verified_at = created_at
                WHERE (email IS NULL OR TRIM(email) = '')
                  AND (verified_at IS NULL OR TRIM(verified_at) = '')
                """
            )
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
            conn.execute("ALTER TABLE community_sets ADD COLUMN owner_account_id TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE community_sets ADD COLUMN visibility TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE community_material_history ADD COLUMN active_index INTEGER")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE community_material_history ADD COLUMN active_reel_id TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("UPDATE community_sets SET updated_at = created_at WHERE updated_at IS NULL OR TRIM(updated_at) = ''")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute(
                """
                UPDATE community_sets
                SET owner_key_hash = ?
                WHERE owner_key_hash IS NULL OR TRIM(owner_key_hash) = ''
                """,
                (LEGACY_COMMUNITY_OWNER_HASH,),
            )
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute(
                """
                UPDATE community_sets
                SET visibility = CASE
                    WHEN featured = 1 OR owner_key_hash = ? THEN ?
                    ELSE ?
                END
                WHERE visibility IS NULL OR TRIM(visibility) = ''
                """,
                (LEGACY_COMMUNITY_OWNER_HASH, PUBLIC_COMMUNITY_VISIBILITY, DEFAULT_COMMUNITY_VISIBILITY),
            )
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute(
                """
                UPDATE community_sets
                SET visibility = ?
                WHERE owner_account_id IS NOT NULL
                  AND (
                      visibility IS NULL
                      OR TRIM(visibility) = ''
                      OR visibility = 'private'
                  )
                """,
                (PUBLIC_COMMUNITY_VISIBILITY,),
            )
        except sqlite3.OperationalError:
            pass
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_community_accounts_email_normalized_unique ON community_accounts(email_normalized)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_community_sets_owner_account_updated_at ON community_sets(owner_account_id, updated_at DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_community_sets_visibility_featured_updated_at ON community_sets(visibility, featured DESC, updated_at DESC)"
        )
        _migrate_reels_unique_clip_index_sqlite(conn)
        _migrate_reel_feedback_uniqueness_sqlite(conn)
        conn.commit()
    finally:
        conn.close()
    _db_ready = True


def ensure_db_initialized() -> None:
    global _db_ready
    if _db_ready:
        return
    with _db_init_lock:
        if _db_ready:
            return
        init_db()


## ---------------------------------------------------------------------------
## Lightweight PostgreSQL connection pool (no external dependency required)
## ---------------------------------------------------------------------------
_pg_pool_lock = threading.Lock()
_pg_pool: list[Any] = []
_PG_POOL_MAX_IDLE = 6


def _pg_pool_acquire() -> Any:
    """Return a reusable PostgreSQL connection, or create a fresh one."""
    while True:
        conn: Any = None
        with _pg_pool_lock:
            if _pg_pool:
                conn = _pg_pool.pop()
        if conn is None:
            return psycopg.connect(_database_url(), connect_timeout=15, autocommit=True)
        # Validate before reuse — discard stale/broken connections.
        try:
            conn.execute("SELECT 1")
            return conn
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
            # Loop and try next pooled connection or create a new one.


def _pg_pool_release(conn: Any) -> None:
    """Return a connection to the pool (or close it if the pool is full)."""
    try:
        # Reset to a clean state: rollback any uncommitted work, set autocommit.
        if not conn.autocommit:
            conn.rollback()
            conn.autocommit = True
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        return
    with _pg_pool_lock:
        if len(_pg_pool) < _PG_POOL_MAX_IDLE:
            _pg_pool.append(conn)
        else:
            try:
                conn.close()
            except Exception:
                pass


@contextmanager
def get_conn(*, transactional: bool = False):
    ensure_db_initialized()
    if _is_postgres_configured():
        if psycopg is None:
            raise RuntimeError("DATABASE_URL is set to PostgreSQL, but psycopg is not installed.")
        conn = _pg_pool_acquire()
        conn.autocommit = not transactional
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
            _pg_pool_release(conn)
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


def insert(conn: Any, table: str, data: dict[str, Any]) -> None:
    """Plain INSERT — raises DatabaseIntegrityError on unique constraint violation."""
    if not _SAFE_SQL_IDENTIFIER_RE.fullmatch(table):
        raise ValueError(f"Unsafe table name: {table!r}")
    cols = list(data.keys())
    for col in cols:
        if not _SAFE_SQL_IDENTIFIER_RE.fullmatch(col):
            raise ValueError(f"Unsafe column name: {col!r}")
    placeholders = ", ".join(["?"] * len(cols))
    sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})"
    values = [data[c] for c in cols]
    is_pg = _is_postgres_conn(conn)
    query = _adapt_query_for_postgres(sql) if is_pg else sql
    try:
        if is_pg:
            with conn.cursor() as cur:
                cur.execute(query, values)
        else:
            conn.execute(query, values)
    except Exception as exc:
        if is_pg and _is_unique_violation(exc):
            raise DatabaseIntegrityError(str(exc)) from exc
        if isinstance(exc, sqlite3.IntegrityError):
            raise DatabaseIntegrityError(str(exc)) from exc
        raise


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


def execute_modify(conn: Any, query: str, params: Iterable[Any] = ()) -> int:
    """Execute a non-SELECT statement (INSERT/UPDATE/DELETE) and return the number of affected rows."""
    if _is_postgres_conn(conn):
        with conn.cursor() as cur:
            cur.execute(_adapt_query_for_postgres(query), tuple(params))
            return int(cur.rowcount or 0)

    cursor = conn.execute(query, tuple(params))
    return int(getattr(cursor, "rowcount", 0) or 0)


def dumps_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True)


def loads_json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default
