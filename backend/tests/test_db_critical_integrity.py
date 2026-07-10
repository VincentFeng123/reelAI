from __future__ import annotations

import os
import sqlite3
import tempfile
from unittest import mock

import pytest

from backend.app import db
from backend.app.config import get_settings


class _UniqueViolation(Exception):
    sqlstate = "23505"


class _RecordingCursor:
    def __init__(self) -> None:
        self.statements: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, statement, values=None) -> None:
        self.statements.append(statement)
        if statement.startswith("INSERT INTO"):
            raise _UniqueViolation("duplicate")


class _TransactionalPostgresConnection:
    autocommit = False

    def __init__(self) -> None:
        self.recording_cursor = _RecordingCursor()

    def cursor(self):
        return self.recording_cursor


def test_insert_only_normalizes_unique_violations() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("CREATE TABLE parents (id TEXT PRIMARY KEY)")
    conn.execute(
        "CREATE TABLE children (id TEXT PRIMARY KEY, parent_id TEXT NOT NULL REFERENCES parents(id))"
    )

    db.insert(conn, "parents", {"id": "parent-1"})
    with pytest.raises(db.DatabaseIntegrityError):
        db.insert(conn, "parents", {"id": "parent-1"})

    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        db.insert(conn, "children", {"id": "child-1", "parent_id": "missing"})


def test_postgres_unique_insert_rolls_back_to_savepoint() -> None:
    conn = _TransactionalPostgresConnection()
    with mock.patch.object(db, "_is_postgres_conn", return_value=True):
        with pytest.raises(db.DatabaseIntegrityError, match="duplicate"):
            db.insert(conn, "materials", {"id": "material-1"})

    assert conn.recording_cursor.statements == [
        "SAVEPOINT studyreels_insert",
        "INSERT INTO materials (id) VALUES (%s)",
        "ROLLBACK TO SAVEPOINT studyreels_insert",
        "RELEASE SAVEPOINT studyreels_insert",
    ]


def test_reel_clip_uniqueness_is_scoped_to_concept() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE reels (
            id TEXT PRIMARY KEY,
            generation_id TEXT,
            material_id TEXT NOT NULL,
            concept_id TEXT NOT NULL,
            video_id TEXT NOT NULL,
            t_start REAL NOT NULL,
            t_end REAL NOT NULL
        )
        """
    )
    db._migrate_reels_unique_clip_index_sqlite(conn)

    values = ("generation-1", "material-1", "concept-1", "yt:video-1", 10.0, 20.0)
    conn.execute(
        "INSERT INTO reels VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("reel-1", *values),
    )
    conn.execute(
        "INSERT INTO reels VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("reel-2", "generation-1", "material-1", "concept-2", "yt:video-1", 10.0, 20.0),
    )
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
        conn.execute(
            "INSERT INTO reels VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("reel-3", *values),
        )


def test_runtime_sqlite_connections_enforce_foreign_keys() -> None:
    previous_data_dir = os.environ.get("DATA_DIR")
    with tempfile.TemporaryDirectory() as temp_dir:
        os.environ["DATA_DIR"] = temp_dir
        get_settings.cache_clear()
        db._db_ready = False
        try:
            db.init_db()
            with db.get_conn() as conn:
                enabled = conn.execute("PRAGMA foreign_keys").fetchone()[0]
                head_columns = {
                    row[1]
                    for row in conn.execute("PRAGMA table_info(reel_generation_heads)").fetchall()
                }
            assert enabled == 1
            assert "refinement_state_json" in head_columns

            db._db_ready = False
            db.init_db()
        finally:
            db._db_ready = False
            get_settings.cache_clear()
            if previous_data_dir is None:
                os.environ.pop("DATA_DIR", None)
            else:
                os.environ["DATA_DIR"] = previous_data_dir
