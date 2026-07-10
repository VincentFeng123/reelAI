"""Portable, dependency-safe source takedown tests."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

import pytest

from backend.app import db as db_module
from backend.app.ingestion.persistence import takedown_by_source_url


class _PostgresCursor:
    def __init__(self, database: sqlite3.Connection, *, dict_rows: bool = False) -> None:
        self._database = database
        self._dict_rows = dict_rows
        self._cursor: sqlite3.Cursor | None = None

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        if self._cursor is not None:
            self._cursor.close()

    def execute(self, query: str, params=()) -> None:
        self._cursor = self._database.execute(query.replace("%s", "?"), tuple(params))

    def fetchall(self):
        assert self._cursor is not None
        rows = self._cursor.fetchall()
        return [dict(row) for row in rows] if self._dict_rows else rows

    def fetchone(self):
        assert self._cursor is not None
        row = self._cursor.fetchone()
        return dict(row) if row is not None and self._dict_rows else row

    @property
    def rowcount(self) -> int:
        return int(self._cursor.rowcount if self._cursor is not None else 0)


class _PostgresConnection:
    """Exercise the psycopg branches of the shared DB helpers over SQLite."""

    __module__ = "psycopg.testing"

    def __init__(self, database: sqlite3.Connection) -> None:
        self._database = database

    def cursor(self, *, row_factory=None) -> _PostgresCursor:
        return _PostgresCursor(self._database, dict_rows=row_factory is not None)

    def commit(self) -> None:
        self._database.commit()

    def rollback(self) -> None:
        self._database.rollback()


def _seed_takedown_graph(conn: sqlite3.Connection, *, source_url: str, direct: bool) -> None:
    timestamp = "2026-07-09T00:00:00+00:00"
    conn.execute(
        "INSERT INTO materials (id, raw_text, source_type, created_at) VALUES ('m1', 'x', 'topic', ?)",
        (timestamp,),
    )
    conn.execute(
        "INSERT INTO concepts (id, material_id, title, keywords_json, summary, created_at) "
        "VALUES ('c1', 'm1', 'Concept', '[]', '', ?)",
        (timestamp,),
    )
    conn.execute("INSERT INTO videos (id, title, created_at) VALUES ('yt:dQw4w9WgXcQ', 'Target', ?)", (timestamp,))
    conn.execute("INSERT INTO videos (id, title, created_at) VALUES ('yt:M7lc1UVf-VE', 'Survivor', ?)", (timestamp,))
    for reel_id, video_id, video_url in (
        ("target", "yt:dQw4w9WgXcQ", source_url if direct else "https://www.youtube.com/embed/dQw4w9WgXcQ"),
        ("survivor", "yt:M7lc1UVf-VE", "https://www.youtube.com/watch?v=M7lc1UVf-VE"),
    ):
        conn.execute(
            "INSERT INTO reels "
            "(id, material_id, concept_id, video_id, video_url, t_start, t_end, "
            "transcript_snippet, takeaways_json, base_score, created_at) "
            "VALUES (?, 'm1', 'c1', ?, ?, 0, 60, 'Transcript', '[]', 1, ?)",
            (reel_id, video_id, video_url, timestamp),
        )
        conn.execute(
            "INSERT INTO learner_reel_progress "
            "(learner_id, reel_id, material_id, max_fraction, completed_at, created_at, updated_at) "
            "VALUES ('owner:learner', ?, 'm1', 1, ?, ?, ?)",
            (reel_id, timestamp, timestamp, timestamp),
        )
        conn.execute(
            "INSERT INTO reel_feedback (id, learner_id, reel_id, helpful, created_at) "
            "VALUES (?, 'owner:learner', ?, 1, ?)",
            (f"feedback-{reel_id}", reel_id, timestamp),
        )
        conn.execute(
            "INSERT INTO reel_assessment_questions "
            "(id, reel_id, fingerprint, prompt, options_json, correct_index, explanation, created_at) "
            "VALUES (?, ?, ?, 'Prompt?', '[\"A\",\"B\",\"C\",\"D\"]', 0, 'Because A.', ?)",
            (f"question-{reel_id}", reel_id, f"fingerprint-{reel_id}", timestamp),
        )
        metadata_url = source_url if reel_id == "target" and not direct else video_url
        conn.execute(
            "INSERT INTO llm_cache (cache_key, response_json, created_at) VALUES (?, ?, ?)",
            (f"ingest_meta:{reel_id}", json.dumps({"source_url": metadata_url}, separators=(",", ":")), timestamp),
        )

    conn.execute(
        "INSERT INTO assessment_sessions "
        "(id, learner_id, material_id, status, question_count, correct_count, created_at, updated_at, completed_at) "
        "VALUES ('session', 'owner:learner', 'm1', 'completed', 2, 1, ?, ?, ?)",
        (timestamp, timestamp, timestamp),
    )
    for position, reel_id in enumerate(("target", "survivor")):
        conn.execute(
            "INSERT INTO assessment_session_questions (session_id, question_id, position) "
            "VALUES ('session', ?, ?)",
            (f"question-{reel_id}", position),
        )
        conn.execute(
            "INSERT INTO assessment_attempts "
            "(id, learner_id, session_id, question_id, choice_index, is_correct, created_at) "
            "VALUES (?, 'owner:learner', 'session', ?, 0, ?, ?)",
            (f"attempt-{reel_id}", f"question-{reel_id}", 1 if position == 0 else 0, timestamp),
        )
    conn.execute(
        "INSERT INTO assessment_concept_outcomes "
        "(learner_id, session_id, material_id, concept_id, question_count, correct_count, "
        "accuracy, adjustment, source_reel_id, source_video_id, created_at) "
        "VALUES ('owner:learner', 'session', 'm1', 'c1', 2, 1, .5, 0, 'target', 'v1', ?)",
        (timestamp,),
    )
    conn.commit()


@pytest.mark.parametrize("dialect", ["sqlite", "postgres"])
@pytest.mark.parametrize("direct", [True, False], ids=["stored-url", "metadata-url"])
def test_takedown_cleans_assessment_dependencies_portably(
    monkeypatch: pytest.MonkeyPatch, dialect: str, direct: bool
) -> None:
    database = sqlite3.connect(":memory:")
    database.row_factory = sqlite3.Row
    database.executescript(db_module.SCHEMA)
    database.execute("PRAGMA foreign_keys = ON")
    stored_source_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&feature=share"
    takedown_url = "https://youtu.be/dQw4w9WgXcQ?t=12"
    _seed_takedown_graph(database, source_url=stored_source_url, direct=direct)

    connection: Any = database
    if dialect == "postgres":
        connection = _PostgresConnection(database)
        monkeypatch.setattr(
            db_module,
            "_is_postgres_conn",
            lambda value: isinstance(value, _PostgresConnection),
        )

    try:
        assert takedown_by_source_url(connection, takedown_url) == 1

        assert database.execute("SELECT COUNT(*) FROM reels WHERE id = 'target'").fetchone()[0] == 0
        assert database.execute("SELECT COUNT(*) FROM reels WHERE id = 'survivor'").fetchone()[0] == 1
        assert database.execute("SELECT COUNT(*) FROM assessment_sessions").fetchone()[0] == 0
        assert database.execute("SELECT COUNT(*) FROM assessment_session_questions").fetchone()[0] == 0
        assert database.execute("SELECT COUNT(*) FROM assessment_attempts").fetchone()[0] == 0
        assert database.execute("SELECT COUNT(*) FROM assessment_concept_outcomes").fetchone()[0] == 0
        assert database.execute(
            "SELECT COUNT(*) FROM reel_assessment_questions WHERE reel_id = 'target'"
        ).fetchone()[0] == 0
        assert database.execute(
            "SELECT COUNT(*) FROM reel_assessment_questions WHERE reel_id = 'survivor'"
        ).fetchone()[0] == 1
        assert database.execute(
            "SELECT COUNT(*) FROM learner_reel_progress WHERE reel_id = 'target'"
        ).fetchone()[0] == 0
        assert database.execute(
            "SELECT COUNT(*) FROM learner_reel_progress WHERE reel_id = 'survivor'"
        ).fetchone()[0] == 1
        assert database.execute(
            "SELECT COUNT(*) FROM reel_feedback WHERE reel_id = 'target'"
        ).fetchone()[0] == 0
        assert database.execute(
            "SELECT COUNT(*) FROM llm_cache WHERE cache_key = 'ingest_meta:target'"
        ).fetchone()[0] == 0
        assert database.execute(
            "SELECT COUNT(*) FROM llm_cache WHERE cache_key = 'ingest_meta:survivor'"
        ).fetchone()[0] == 1
        assert database.execute(
            "SELECT COUNT(*) FROM blocked_video_tombstones WHERE video_id = 'dQw4w9WgXcQ'"
        ).fetchone()[0] == 1
        assert database.execute(
            "SELECT COUNT(*) FROM videos WHERE id = 'yt:M7lc1UVf-VE'"
        ).fetchone()[0] == 1
    finally:
        database.close()
