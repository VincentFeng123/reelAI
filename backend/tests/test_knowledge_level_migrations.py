"""Columns for the knowledge-level feature exist after init and are idempotent."""
import os
import sqlite3
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

    def test_feedback_and_learner_progress_schema(self) -> None:
        feedback_cols = self._cols("reel_feedback")
        self.assertIn("learner_id", feedback_cols)
        self.assertIn("mastery_updated_at", feedback_cols)
        self.assertIn("updated_at", feedback_cols)
        progress_cols = self._cols("learner_material_progress")
        self.assertEqual(
            {
                "learner_id",
                "material_id",
                "selected_level",
                "global_adjustment",
                "difficulty_reset_at",
                "feedback_revision",
                "updated_at",
            },
            set(progress_cols),
        )
        with self.db.get_conn() as conn:
            indexes = self.db.fetch_all(conn, "PRAGMA index_list(reel_feedback)")
        composite = [row for row in indexes if row["name"] == "idx_reel_feedback_learner_reel_unique"]
        self.assertEqual(len(composite), 1)
        self.assertEqual(int(composite[0]["unique"]), 1)

    def test_sqlite_migrates_global_feedback_to_legacy_identity(self) -> None:
        # Initialize all unrelated tables, then replace reel_feedback with the
        # pre-migration globally-unique shape.
        path = Path(self.temp_dir.name) / "studyreels.db"
        conn = sqlite3.connect(path)
        conn.executescript(self.db.SCHEMA)
        conn.execute("DROP INDEX IF EXISTS idx_reel_feedback_reel_id")
        conn.execute("DROP INDEX IF EXISTS idx_reel_feedback_learner_reel_unique")
        conn.execute("DROP TABLE reel_feedback")
        conn.execute(
            "CREATE TABLE reel_feedback ("
            "id TEXT PRIMARY KEY, reel_id TEXT NOT NULL, helpful INTEGER NOT NULL DEFAULT 0, "
            "confusing INTEGER NOT NULL DEFAULT 0, rating INTEGER, saved INTEGER NOT NULL DEFAULT 0, "
            "created_at TEXT NOT NULL)"
        )
        conn.execute(
            "CREATE UNIQUE INDEX idx_reel_feedback_reel_id_unique ON reel_feedback(reel_id)"
        )
        conn.execute(
            "INSERT INTO reel_feedback (id, reel_id, helpful, confusing, saved, created_at) "
            "VALUES ('f1', 'r1', 1, 0, 0, '2026-07-08T00:00:00+00:00')"
        )
        conn.commit()
        conn.close()

        self.db._db_ready = False
        with self.db.get_conn(transactional=True) as migrated:
            row = self.db.fetch_one(migrated, "SELECT * FROM reel_feedback WHERE id = 'f1'")
            self.assertEqual(row["learner_id"], self.db.LEGACY_LEARNER_ID)
            self.assertEqual(row["mastery_updated_at"], row["created_at"])
            self.assertEqual(row["updated_at"], row["created_at"])
            migrated.execute(
                "INSERT INTO reel_feedback "
                "(id, learner_id, reel_id, helpful, confusing, saved, updated_at, created_at) "
                "VALUES ('f2', 'owner:a', 'r1', 0, 0, 0, 't2', 't2')"
            )
            migrated.execute(
                "INSERT INTO reel_feedback "
                "(id, learner_id, reel_id, helpful, confusing, saved, updated_at, created_at) "
                "VALUES ('f3', 'owner:b', 'r1', 0, 0, 0, 't3', 't3')"
            )
            with self.assertRaises(sqlite3.IntegrityError):
                migrated.execute(
                    "INSERT INTO reel_feedback "
                    "(id, learner_id, reel_id, helpful, confusing, saved, updated_at, created_at) "
                    "VALUES ('f4', 'owner:a', 'r1', 0, 0, 0, 't4', 't4')"
                )

    def test_postgres_feedback_migration_builds_composite_unique_index(self) -> None:
        calls: list[tuple[str, tuple | None]] = []

        class Cursor:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def execute(self, sql, params=None):
                calls.append((" ".join(str(sql).split()), params))

        class Connection:
            def cursor(self):
                return Cursor()

        self.db._migrate_reel_feedback_uniqueness_postgres(Connection())
        sql = "\n".join(statement for statement, _ in calls)
        self.assertIn("ADD COLUMN IF NOT EXISTS learner_id", sql)
        self.assertIn("ADD COLUMN IF NOT EXISTS mastery_updated_at", sql)
        self.assertIn(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_reel_feedback_learner_reel_unique ON reel_feedback(learner_id, reel_id)",
            sql,
        )

    def test_init_is_idempotent(self) -> None:
        # Re-running init on an already-migrated DB must not raise.
        self.db._db_ready = False
        with self.db.get_conn() as conn:
            self.db.fetch_one(conn, "SELECT 1")


if __name__ == "__main__":
    unittest.main()
