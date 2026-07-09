"""Auto-adjust: last-20 window, <5-row gate, direction, ±0.35 bound.
Uses a temp DB (same DATA_DIR pattern as test_knowledge_level_migrations).

Seed adaptation: reel_feedback has a UNIQUE constraint on reel_id (one row
per reel after the migration).  To build a feedback window of N we insert N
reels and one feedback row per reel, all tied to the same material.
"""
import os
import sys
import tempfile
import unittest
import uuid
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
            # videos: no platform/source_id columns; has title, channel_title, duration_sec
            conn.execute(
                "INSERT INTO videos (id, title, channel_title, duration_sec, created_at) "
                "VALUES ('yt:v1', 'T', 'C', 600, '2026-07-08T00:00:00+00:00')"
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
        """Insert n reels for m1 and one feedback row per reel.

        reel_feedback has a UNIQUE constraint on reel_id, so we model a window
        of N by creating N reels rather than N rows for a single reel.
        """
        with self.db.get_conn(transactional=True) as conn:
            for i in range(n):
                reel_id = f"r{i}"
                conn.execute(
                    "INSERT OR IGNORE INTO reels (id, material_id, concept_id, video_id, video_url, "
                    "t_start, t_end, transcript_snippet, takeaways_json, base_score, created_at) "
                    "VALUES (?, 'm1', 'c1', 'yt:v1', 'u', ?, ?, 's', '[]', 1.0, ?)",
                    (reel_id, i * 30, i * 30 + 30, f"2026-07-08T00:00:{i:02d}+00:00"),
                )
                conn.execute(
                    "INSERT OR IGNORE INTO reel_feedback (id, reel_id, helpful, confusing, rating, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (str(uuid.uuid4()), reel_id, helpful, confusing, rating,
                     f"2026-07-08T00:00:{i:02d}+00:00"),
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
