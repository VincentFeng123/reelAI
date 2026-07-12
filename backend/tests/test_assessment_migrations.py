"""Assessment/reel-content schema parity for SQLite and PostgreSQL."""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path

from backend.app import db as db_module
from backend.app.config import get_settings


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}


def test_fresh_sqlite_schema_contains_assessment_tables_and_private_keys() -> None:
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(db_module.SCHEMA)
        assert {"ai_summary", "match_reason", "informativeness"} <= _columns(conn, "reels")
        assert {
            "learner_reel_progress",
            "reel_assessment_questions",
            "assessment_sessions",
            "assessment_session_questions",
            "assessment_attempts",
            "assessment_concept_outcomes",
        } <= {
            str(row[0])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        assert {"correct_index", "explanation", "fingerprint"} <= _columns(
            conn, "reel_assessment_questions"
        )
        assert "scrolled_at" in _columns(conn, "learner_reel_progress")
    finally:
        conn.close()


def test_existing_sqlite_reels_and_history_are_migrated_idempotently() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        previous = os.environ.get("DATA_DIR")
        os.environ["DATA_DIR"] = tmp
        get_settings.cache_clear()
        db_module._db_ready = False
        try:
            path = Path(tmp) / "studyreels.db"
            conn = sqlite3.connect(path)
            conn.execute(
                "CREATE TABLE reels ("
                "id TEXT PRIMARY KEY, generation_id TEXT, material_id TEXT NOT NULL, "
                "concept_id TEXT NOT NULL, video_id TEXT NOT NULL, video_url TEXT NOT NULL, "
                "t_start REAL NOT NULL, t_end REAL NOT NULL, transcript_snippet TEXT NOT NULL, "
                "takeaways_json TEXT NOT NULL, base_score REAL NOT NULL, created_at TEXT NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE community_material_history ("
                "account_id TEXT NOT NULL, material_id TEXT NOT NULL, title TEXT NOT NULL, "
                "updated_at BIGINT NOT NULL, starred INTEGER NOT NULL DEFAULT 0, "
                "generation_mode TEXT NOT NULL DEFAULT 'slow', source TEXT NOT NULL DEFAULT 'search', "
                "feed_query TEXT, active_index INTEGER, active_reel_id TEXT, "
                "PRIMARY KEY(account_id, material_id))"
            )
            conn.execute(
                "CREATE TABLE learner_reel_progress ("
                "learner_id TEXT NOT NULL, reel_id TEXT NOT NULL, material_id TEXT NOT NULL, "
                "max_fraction REAL NOT NULL DEFAULT 0.0, completed_at TEXT, "
                "created_at TEXT NOT NULL, updated_at TEXT NOT NULL, "
                "PRIMARY KEY(learner_id, reel_id))"
            )
            conn.commit()
            conn.close()

            db_module.init_db()
            db_module._db_ready = False
            db_module.init_db()
            migrated = sqlite3.connect(path)
            try:
                assert {"ai_summary", "match_reason", "informativeness"} <= _columns(
                    migrated, "reels"
                )
                assert "recall_json" in _columns(migrated, "community_material_history")
                assert "scrolled_at" in _columns(migrated, "learner_reel_progress")
                indexes = {
                    str(row[1])
                    for row in migrated.execute("PRAGMA index_list(learner_reel_progress)")
                }
                assert "idx_learner_reel_progress_material_scrolled" in indexes
            finally:
                migrated.close()
        finally:
            if previous is None:
                os.environ.pop("DATA_DIR", None)
            else:
                os.environ["DATA_DIR"] = previous
            get_settings.cache_clear()
            db_module._db_ready = False


def test_postgres_schema_contains_matching_assessment_contract() -> None:
    sql = "\n".join(db_module._postgres_schema_statements())
    for table in (
        "learner_reel_progress",
        "reel_assessment_questions",
        "assessment_sessions",
        "assessment_session_questions",
        "assessment_attempts",
        "assessment_concept_outcomes",
    ):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in sql
    assert "WHERE status = 'pending'" in sql
    assert "correct_index INTEGER NOT NULL" in sql
    assert "scrolled_at TEXT" in sql
