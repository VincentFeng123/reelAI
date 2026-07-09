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
