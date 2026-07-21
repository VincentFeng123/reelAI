"""Learner-scoped global difficulty adjustment semantics."""
import json
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
    LEARNER = "owner:test-learner"

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
            for concept_id, title in (("c1", "Forces"), ("c2", "Energy")):
                conn.execute(
                    "INSERT INTO concepts (id, material_id, title, keywords_json, summary, embedding_json, created_at) "
                    "VALUES (?, 'm1', ?, '[]', '', NULL, '2026-07-08T00:00:00+00:00')",
                    (concept_id, title),
                )
            for video_id in ("yt:v1", "yt:v2"):
                conn.execute(
                    "INSERT INTO videos (id, title, channel_title, duration_sec, created_at) "
                    "VALUES (?, 'T', 'C', 600, '2026-07-08T00:00:00+00:00')",
                    (video_id,),
                )
            self.svc.learner_progress(conn, "m1", self.LEARNER)
            conn.execute(
                "UPDATE learner_material_progress SET difficulty_reset_at = ? "
                "WHERE learner_id = ? AND material_id = 'm1'",
                ("2026-07-08T00:00:00+00:00", self.LEARNER),
            )

    def _restore(self) -> None:
        from backend.app.config import get_settings

        if self._prev is None:
            os.environ.pop("DATA_DIR", None)
        else:
            os.environ["DATA_DIR"] = self._prev
        self.db._db_ready = False
        get_settings.cache_clear()

    def _feedback(self, signals: list[tuple[str, bool, bool]]) -> None:
        with self.db.get_conn(transactional=True) as conn:
            for i, (concept_id, helpful, confusing) in enumerate(signals, start=1):
                reel_id = f"r{i}"
                video_id = "yt:v1" if concept_id == "c1" else "yt:v2"
                timestamp = f"2026-07-08T00:{i:02d}:00+00:00"
                conn.execute(
                    "INSERT INTO reels (id, material_id, concept_id, video_id, video_url, "
                    "t_start, t_end, transcript_snippet, takeaways_json, base_score, created_at) "
                    "VALUES (?, 'm1', ?, ?, 'u', ?, ?, 's', '[]', 1.0, ?)",
                    (reel_id, concept_id, video_id, i * 30, i * 30 + 20, timestamp),
                )
                conn.execute(
                    "INSERT INTO reel_feedback "
                    "(id, learner_id, reel_id, helpful, confusing, rating, saved, mastery_updated_at, updated_at, created_at) "
                    "VALUES (?, ?, ?, ?, ?, NULL, 0, ?, ?, ?)",
                    (
                        str(uuid.uuid4()),
                        self.LEARNER,
                        reel_id,
                        1 if helpful else 0,
                        1 if confusing else 0,
                        timestamp,
                        timestamp,
                        timestamp,
                    ),
                )

    def _adj(self) -> float:
        with self.db.get_conn(transactional=True) as conn:
            return self.svc.update_level_adjustment(conn, "m1", self.LEARNER)

    def test_requires_three_mastery_responses(self) -> None:
        self._feedback([("c1", True, False), ("c2", True, False)])
        self.assertEqual(self._adj(), 0.0)

    def test_requires_two_concepts(self) -> None:
        self._feedback([("c1", True, False)] * 3)
        self.assertEqual(self._adj(), 0.0)

    def test_uses_explicit_signed_weights_across_two_concepts(self) -> None:
        self._feedback(
            [("c1", True, False), ("c2", True, False), ("c1", False, True)]
        )
        self.assertAlmostEqual(self._adj(), 0.02)

    def test_uses_latest_twelve_signals(self) -> None:
        latest = [
            ("c1" if i % 2 else "c2", i < 6, i >= 6)
            for i in range(12)
        ]
        self._feedback([("c1", True, False), *latest])
        self.assertAlmostEqual(self._adj(), -0.12)

    def test_paged_scan_reaches_compatible_signals_behind_stale_history(self) -> None:
        from unittest.mock import patch

        from backend.app.services import reels as reels_module

        stale_context = json.dumps({
            "selection_authority": "gemini",
            "concept_family_contract_version": "concept_family_v2",
        })
        with self.db.get_conn(transactional=True) as conn:
            for index in range(30):
                reel_id = f"stale-{index}"
                timestamp = f"2026-07-10T00:{index:02d}:00+00:00"
                conn.execute(
                    "INSERT INTO reels (id, material_id, concept_id, video_id, "
                    "video_url, t_start, t_end, transcript_snippet, takeaways_json, "
                    "base_score, search_context_json, created_at) VALUES (?, 'm1', "
                    "'c1', 'yt:v1', 'u', ?, ?, 's', '[]', 1.0, ?, ?)",
                    (reel_id, index * 30, index * 30 + 20, stale_context, timestamp),
                )
                conn.execute(
                    "INSERT INTO reel_feedback (id, learner_id, reel_id, helpful, "
                    "confusing, rating, saved, mastery_updated_at, updated_at, created_at) "
                    "VALUES (?, ?, ?, 0, 1, NULL, 0, ?, ?, ?)",
                    (
                        str(uuid.uuid4()),
                        self.LEARNER,
                        reel_id,
                        timestamp,
                        timestamp,
                        timestamp,
                    ),
                )
            for index, concept_id in enumerate(("c1", "c2", "c1"), start=1):
                reel_id = f"compatible-{index}"
                video_id = "yt:v1" if concept_id == "c1" else "yt:v2"
                timestamp = f"2026-07-09T00:0{index}:00+00:00"
                conn.execute(
                    "INSERT INTO reels (id, material_id, concept_id, video_id, "
                    "video_url, t_start, t_end, transcript_snippet, takeaways_json, "
                    "base_score, created_at) VALUES (?, 'm1', ?, ?, 'u', ?, ?, "
                    "'s', '[]', 1.0, ?)",
                    (reel_id, concept_id, video_id, 1000 + index * 30,
                     1020 + index * 30, timestamp),
                )
                conn.execute(
                    "INSERT INTO reel_feedback (id, learner_id, reel_id, helpful, "
                    "confusing, rating, saved, mastery_updated_at, updated_at, created_at) "
                    "VALUES (?, ?, ?, 1, 0, NULL, 0, ?, ?, ?)",
                    (
                        str(uuid.uuid4()),
                        self.LEARNER,
                        reel_id,
                        timestamp,
                        timestamp,
                        timestamp,
                    ),
                )

            original_fetch_all = reels_module.fetch_all
            page_limits: list[int] = []

            def tracking_fetch_all(conn_arg, query, params=()):
                if "LIMIT ? OFFSET ?" in query:
                    page_limits.append(int(tuple(params)[-2]))
                return original_fetch_all(conn_arg, query, params)

            with patch.object(reels_module, "fetch_all", tracking_fetch_all):
                adjustment = self.svc.update_level_adjustment(
                    conn,
                    "m1",
                    self.LEARNER,
                )

        self.assertAlmostEqual(adjustment, 0.12)
        self.assertGreaterEqual(len(page_limits), 2)
        self.assertEqual(set(page_limits), {self.svc.GLOBAL_SIGNAL_SCAN_PAGE_SIZE})

    def test_paged_scan_caps_an_all_stale_history(self) -> None:
        from unittest.mock import patch

        from backend.app.services import reels as reels_module

        original_fetch_all = reels_module.fetch_all
        feedback_page_calls = 0
        stale_context = json.dumps({
            "selection_authority": "gemini",
            "concept_family_contract_version": "concept_family_v2",
        })

        def stale_history_fetch_all(conn_arg, query, params=()):
            nonlocal feedback_page_calls
            if "FROM reel_feedback f" in query and "LIMIT ? OFFSET ?" in query:
                feedback_page_calls += 1
                page_size = int(tuple(params)[-2])
                return [
                    {
                        "helpful": 0,
                        "confusing": 1,
                        "concept_id": "c1",
                        "event_at": "2099-01-01T00:00:00+00:00",
                        "search_context_json": stale_context,
                    }
                    for _ in range(page_size)
                ]
            return original_fetch_all(conn_arg, query, params)

        with self.db.get_conn(transactional=True) as conn:
            with patch.object(
                reels_module,
                "fetch_all",
                stale_history_fetch_all,
            ):
                adjustment = self.svc.update_level_adjustment(
                    conn,
                    "m1",
                    self.LEARNER,
                )

        self.assertEqual(adjustment, 0.0)
        self.assertEqual(
            feedback_page_calls,
            self.svc.GLOBAL_SIGNAL_SCAN_MAX_ROWS
            // self.svc.GLOBAL_SIGNAL_SCAN_PAGE_SIZE,
        )

    def test_clamps_global_drift_to_positive_point_two(self) -> None:
        self._feedback(
            [("c1" if i % 2 else "c2", True, False) for i in range(12)]
        )
        self.assertAlmostEqual(self._adj(), 0.20)

    def test_clamps_global_drift_to_negative_point_two(self) -> None:
        self._feedback(
            [("c1" if i % 2 else "c2", False, True) for i in range(12)]
        )
        self.assertAlmostEqual(self._adj(), -0.20)

    def test_persists_on_learner_progress_not_material(self) -> None:
        self._feedback([("c1", True, False), ("c2", True, False), ("c1", True, False)])
        expected = self._adj()
        with self.db.get_conn() as conn:
            material = self.db.fetch_one(conn, "SELECT level_adjustment FROM materials WHERE id='m1'")
            progress = self.db.fetch_one(
                conn,
                "SELECT global_adjustment FROM learner_material_progress "
                "WHERE learner_id = ? AND material_id = 'm1'",
                (self.LEARNER,),
            )
        self.assertEqual(float(material["level_adjustment"]), 0.0)
        self.assertAlmostEqual(float(progress["global_adjustment"]), expected)

    def test_manual_level_change_resets_drift_but_retains_coverage(self) -> None:
        self._feedback([("c1", True, False), ("c2", True, False), ("c1", True, False)])
        self.assertAlmostEqual(self._adj(), 0.12)
        with self.db.get_conn(transactional=True) as conn:
            progress = self.svc.set_learner_level(conn, "m1", self.LEARNER, "intermediate")
            self.assertEqual(progress["selected_level"], "intermediate")
            self.assertEqual(float(progress["global_adjustment"]), 0.0)
            self.assertEqual(self.svc.update_level_adjustment(conn, "m1", self.LEARNER), 0.0)
            coverage, adjustments, latest, target = self.svc._learner_adaptation_context(
                conn, "m1", self.LEARNER
            )
        self.assertEqual(coverage["c1"]["helpful"], 2.0)
        self.assertEqual(coverage["c2"]["helpful"], 1.0)
        self.assertEqual(adjustments, {})
        self.assertIsNone(latest)
        self.assertAlmostEqual(target, 0.50)


if __name__ == "__main__":
    unittest.main()
