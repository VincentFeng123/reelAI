"""Focused adaptive curriculum and learner-feedback contract tests."""
from __future__ import annotations

import sqlite3
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.db import SCHEMA, _migrate_reel_feedback_uniqueness_sqlite
from backend.app.db import now_iso
from backend.app.services.reels import ReelService


class AdaptiveCurriculumTests(unittest.TestCase):
    MATERIAL = "m1"
    LEARNER = "owner:learner-a"

    def setUp(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        _migrate_reel_feedback_uniqueness_sqlite(self.conn)
        self.svc = ReelService(embedding_service=None, youtube_service=None)
        self.conn.execute(
            "INSERT INTO materials "
            "(id, subject_tag, raw_text, source_type, source_path, knowledge_level, created_at) "
            "VALUES (?, 'physics', 'physics', 'topic', NULL, 'beginner', '2026-07-09T00:00:00+00:00')",
            (self.MATERIAL,),
        )
        for concept_id in ("c1", "c2"):
            self.conn.execute(
                "INSERT INTO concepts "
                "(id, material_id, title, keywords_json, summary, embedding_json, created_at) "
                "VALUES (?, ?, ?, '[]', '', NULL, '2026-07-09T00:00:00+00:00')",
                (concept_id, self.MATERIAL, concept_id),
            )
        for video_id in ("va", "vb", "vc"):
            self.conn.execute(
                "INSERT INTO videos (id, title, channel_title, duration_sec, created_at) "
                "VALUES (?, ?, 'channel', 600, '2026-07-09T00:00:00+00:00')",
                (video_id, video_id),
            )
        self.svc.learner_progress(self.conn, self.MATERIAL, self.LEARNER)

    def tearDown(self) -> None:
        self.conn.close()

    def _insert_reel(
        self,
        reel_id: str,
        concept_id: str,
        video_id: str,
        start: float,
        difficulty: float | None,
    ) -> None:
        self.conn.execute(
            "INSERT INTO reels "
            "(id, material_id, concept_id, video_id, video_url, t_start, t_end, "
            "transcript_snippet, takeaways_json, base_score, difficulty, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 'text', '[]', 1.0, ?, ?)",
            (
                reel_id,
                self.MATERIAL,
                concept_id,
                video_id,
                f"https://youtube.test/{video_id}",
                start,
                start + 20.0,
                difficulty,
                f"2026-07-09T00:{int(start) % 60:02d}:00+00:00",
            ),
        )

    @staticmethod
    def _item(
        reel_id: str,
        concept_id: str,
        video_id: str,
        start: float,
        score: float,
        difficulty: float | None,
    ) -> dict:
        return {
            "reel_id": reel_id,
            "concept_id": concept_id,
            "video_id": video_id,
            "t_start": start,
            "t_end": start + 20.0,
            "score": score,
            "difficulty": difficulty,
        }

    def test_selected_levels_start_at_all_three_targets(self) -> None:
        expected = {"beginner": 0.15, "intermediate": 0.50, "advanced": 0.85}
        for index, (level, target) in enumerate(expected.items(), start=2):
            material_id = f"m{index}"
            self.conn.execute(
                "INSERT INTO materials "
                "(id, subject_tag, raw_text, source_type, source_path, knowledge_level, created_at) "
                "VALUES (?, 'physics', 'physics', 'topic', NULL, ?, '2026-07-09T00:00:00+00:00')",
                (material_id, level),
            )
            progress = self.svc.learner_progress(self.conn, material_id, self.LEARNER)
            self.assertEqual(progress["selected_level"], level)
            self.assertAlmostEqual(
                self.svc._learner_adaptation_context(
                    self.conn, material_id, self.LEARNER
                )[3],
                target,
            )

    def test_sources_interleave_and_each_source_stays_chronological(self) -> None:
        items = [
            self._item("A2", "c1", "va", 60, 10, 0.2),
            self._item("B2", "c2", "vb", 60, 9, 0.2),
            self._item("A1", "c1", "va", 10, 5, 0.2),
            self._item("B1", "c2", "vb", 10, 4, 0.2),
        ]
        ordered = self.svc.adaptive_curriculum_order(
            self.conn, self.MATERIAL, self.LEARNER, items
        )
        self.assertEqual([row["reel_id"] for row in ordered], ["A1", "B1", "A2", "B2"])

    def test_unseen_tail_starts_from_another_source_when_available(self) -> None:
        items = [
            self._item("same-source", "c1", "va", 10, 10, 0.2),
            self._item("other-source", "c2", "vb", 10, 1, 0.2),
        ]
        ordered = self.svc.adaptive_curriculum_order(
            self.conn,
            self.MATERIAL,
            self.LEARNER,
            items,
            previous_video_id="va",
        )
        self.assertEqual(ordered[0]["reel_id"], "other-source")

    def test_got_it_puts_a_different_concept_next(self) -> None:
        self._insert_reel("watched", "c1", "va", 1, 0.5)
        self.svc.record_feedback(
            self.conn, "watched", helpful=True, confusing=False, rating=None,
            saved=False, learner_id=self.LEARNER,
        )
        items = [
            self._item("same", "c1", "vb", 10, 10, 0.6),
            self._item("different", "c2", "vc", 10, 1, 0.2),
        ]
        ordered = self.svc.adaptive_curriculum_order(
            self.conn, self.MATERIAL, self.LEARNER, items
        )
        self.assertEqual(ordered[0]["reel_id"], "different")
        adjustments = self.svc._learner_adaptation_context(
            self.conn, self.MATERIAL, self.LEARNER
        )[1]
        self.assertAlmostEqual(adjustments["c1"], 0.04)

    def test_need_help_prefers_easier_same_concept_from_other_source(self) -> None:
        self._insert_reel("watched", "c1", "va", 1, 0.8)
        self.svc.record_feedback(
            self.conn, "watched", helpful=False, confusing=True, rating=None,
            saved=False, learner_id=self.LEARNER,
        )
        items = [
            self._item("same-source", "c1", "va", 30, 10, 0.1),
            self._item("alternate-source", "c1", "vb", 10, 1, 0.35),
            self._item("other-concept", "c2", "vc", 10, 9, 0.1),
        ]
        ordered = self.svc.adaptive_curriculum_order(
            self.conn, self.MATERIAL, self.LEARNER, items
        )
        self.assertEqual(ordered[0]["reel_id"], "alternate-source")
        adjustments = self.svc._learner_adaptation_context(
            self.conn, self.MATERIAL, self.LEARNER
        )[1]
        self.assertAlmostEqual(adjustments["c1"], -0.06)

    def test_need_help_falls_back_to_same_source(self) -> None:
        self._insert_reel("watched", "c1", "va", 1, 0.8)
        self.svc.record_feedback(
            self.conn, "watched", helpful=False, confusing=True, rating=None,
            saved=False, learner_id=self.LEARNER,
        )
        items = [
            self._item("same-source", "c1", "va", 30, 1, 0.2),
            self._item("other-concept", "c2", "vc", 10, 10, 0.1),
        ]
        ordered = self.svc.adaptive_curriculum_order(
            self.conn, self.MATERIAL, self.LEARNER, items
        )
        self.assertEqual(ordered[0]["reel_id"], "same-source")

    def test_need_help_remediation_avoids_preserved_current_source(self) -> None:
        self._insert_reel("mastery", "c1", "va", 1, 0.8)
        self.svc.record_feedback(
            self.conn, "mastery", helpful=False, confusing=True, rating=None,
            saved=False, learner_id=self.LEARNER,
        )
        items = [
            self._item("current-source", "c1", "vb", 10, 10, 0.2),
            self._item("third-source", "c1", "vc", 10, 1, 0.3),
        ]
        ordered = self.svc.adaptive_curriculum_order(
            self.conn,
            self.MATERIAL,
            self.LEARNER,
            items,
            previous_video_id="vb",
        )
        self.assertEqual(ordered[0]["reel_id"], "third-source")

    def test_need_help_boundary_falls_back_to_mastery_source_before_repeating_current(self) -> None:
        self._insert_reel("mastery", "c1", "va", 1, 0.8)
        self.svc.record_feedback(
            self.conn, "mastery", helpful=False, confusing=True, rating=None,
            saved=False, learner_id=self.LEARNER,
        )
        items = [
            self._item("current-source", "c1", "vb", 10, 10, 0.2),
            self._item("mastery-source", "c1", "va", 30, 1, 0.3),
        ]
        ordered = self.svc.adaptive_curriculum_order(
            self.conn,
            self.MATERIAL,
            self.LEARNER,
            items,
            previous_video_id="vb",
        )
        self.assertEqual(ordered[0]["reel_id"], "mastery-source")

    def test_concept_adjustment_is_bounded_and_learners_are_isolated(self) -> None:
        other = "owner:learner-b"
        self.svc.learner_progress(self.conn, self.MATERIAL, other)
        for index in range(7):
            reel_id = f"feedback-{index}"
            self._insert_reel(reel_id, "c1", "va", 10 + index * 25, 0.5)
            self.svc.record_feedback(
                self.conn, reel_id, helpful=True, confusing=False, rating=None,
                saved=False, learner_id=self.LEARNER,
            )
            self.svc.record_feedback(
                self.conn, reel_id, helpful=False, confusing=True, rating=None,
                saved=False, learner_id=other,
            )
        first = self.svc._learner_adaptation_context(
            self.conn, self.MATERIAL, self.LEARNER
        )[1]
        second = self.svc._learner_adaptation_context(
            self.conn, self.MATERIAL, other
        )[1]
        self.assertEqual(first["c1"], 0.25)
        self.assertEqual(second["c1"], -0.25)
        rows = self.conn.execute(
            "SELECT learner_id, helpful, confusing FROM reel_feedback "
            "WHERE reel_id = 'feedback-0' ORDER BY learner_id"
        ).fetchall()
        self.assertEqual(len(rows), 2)

    def test_save_update_is_mastery_neutral_and_preserves_full_state(self) -> None:
        self._insert_reel("saved", "c1", "va", 1, None)
        first_revision = self.svc.record_feedback(
            self.conn, "saved", helpful=True, confusing=False, rating=None,
            saved=False, learner_id=self.LEARNER,
        )
        before = dict(
            self.conn.execute(
                "SELECT * FROM reel_feedback WHERE learner_id = ? AND reel_id = 'saved'",
                (self.LEARNER,),
            ).fetchone()
        )
        second_revision = self.svc.record_feedback(
            self.conn, "saved", helpful=True, confusing=False, rating=4,
            saved=True, learner_id=self.LEARNER,
        )
        after = dict(
            self.conn.execute(
                "SELECT * FROM reel_feedback WHERE learner_id = ? AND reel_id = 'saved'",
                (self.LEARNER,),
            ).fetchone()
        )
        self.assertEqual(after["mastery_updated_at"], before["mastery_updated_at"])
        self.assertEqual((after["helpful"], after["confusing"], after["rating"], after["saved"]), (1, 0, 4, 1))
        self.assertEqual(second_revision, first_revision + 1)
        self.assertEqual(self.svc._difficulty({"difficulty": None}), 0.5)

    def _insert_assessment_outcome(
        self,
        *,
        session_id: str,
        concept_id: str,
        adjustment: float,
        video_id: str = "va",
        difficulty: float = 0.8,
        learner_id: str | None = None,
    ) -> None:
        learner = learner_id or self.LEARNER
        timestamp = now_iso()
        self.conn.execute(
            "INSERT INTO assessment_sessions "
            "(id, learner_id, material_id, status, current_index, question_count, "
            "correct_count, information_units, readiness_threshold, created_at, updated_at, completed_at) "
            "VALUES (?, ?, ?, 'completed', 1, 1, ?, 3.5, 3.5, ?, ?, ?)",
            (
                session_id,
                learner,
                self.MATERIAL,
                1 if adjustment > 0 else 0,
                timestamp,
                timestamp,
                timestamp,
            ),
        )
        self.conn.execute(
            "INSERT INTO assessment_concept_outcomes "
            "(learner_id, session_id, material_id, concept_id, question_count, "
            "correct_count, accuracy, adjustment, source_reel_id, source_video_id, "
            "source_difficulty, created_at) "
            "VALUES (?, ?, ?, ?, 1, ?, ?, ?, NULL, ?, ?, ?)",
            (
                learner,
                session_id,
                self.MATERIAL,
                concept_id,
                1 if adjustment > 0 else 0,
                1.0 if adjustment > 0 else 0.0,
                adjustment,
                video_id,
                difficulty,
                timestamp,
            ),
        )

    def test_assessment_outcomes_adjust_concepts_with_combined_bound(self) -> None:
        self.conn.execute(
            "UPDATE learner_material_progress SET difficulty_reset_at = '' "
            "WHERE learner_id = ? AND material_id = ?",
            (self.LEARNER, self.MATERIAL),
        )
        self._insert_assessment_outcome(
            session_id="assessment-positive", concept_id="c1", adjustment=0.08
        )
        self._insert_assessment_outcome(
            session_id="assessment-negative", concept_id="c2", adjustment=-0.12
        )
        adjustments = self.svc._learner_adaptation_context(
            self.conn, self.MATERIAL, self.LEARNER
        )[1]
        self.assertAlmostEqual(adjustments["c1"], 0.08)
        self.assertAlmostEqual(adjustments["c2"], -0.12)

        for index in range(3):
            self._insert_assessment_outcome(
                session_id=f"assessment-bound-{index}",
                concept_id="c1",
                adjustment=0.08,
            )
        bounded = self.svc._learner_adaptation_context(
            self.conn, self.MATERIAL, self.LEARNER
        )[1]
        self.assertEqual(bounded["c1"], 0.25)

    def test_acquisition_orders_wrong_quiz_concept_before_right_quiz_concept(
        self,
    ) -> None:
        self._insert_assessment_outcome(
            session_id="acquisition-right",
            concept_id="c1",
            adjustment=0.08,
        )
        self._insert_assessment_outcome(
            session_id="acquisition-wrong",
            concept_id="c2",
            adjustment=-0.12,
        )
        concepts = [
            dict(row)
            for row in self.conn.execute(
                "SELECT * FROM concepts WHERE material_id = ? ORDER BY id",
                (self.MATERIAL,),
            ).fetchall()
        ]

        ordered = self.svc._order_concepts(
            self.conn,
            self.MATERIAL,
            concepts,
            self.LEARNER,
        )

        self.assertEqual([row["id"] for row in ordered], ["c2", "c1"])

    def test_incorrect_assessment_prefers_easier_alternative_source(self) -> None:
        self.conn.execute(
            "UPDATE learner_material_progress SET difficulty_reset_at = '' "
            "WHERE learner_id = ? AND material_id = ?",
            (self.LEARNER, self.MATERIAL),
        )
        self._insert_assessment_outcome(
            session_id="assessment-remediation",
            concept_id="c1",
            adjustment=-0.12,
            video_id="va",
            difficulty=0.8,
        )
        items = [
            self._item("same-source", "c1", "va", 30, 10, 0.2),
            self._item("alternate-source", "c1", "vb", 10, 1, 0.35),
            self._item("other-concept", "c2", "vc", 10, 9, 0.1),
        ]
        ordered = self.svc.adaptive_curriculum_order(
            self.conn, self.MATERIAL, self.LEARNER, items
        )
        self.assertEqual(ordered[0]["reel_id"], "alternate-source")

    def test_mixed_assessment_session_remediates_incorrect_concept(self) -> None:
        self.conn.execute(
            "UPDATE learner_material_progress SET difficulty_reset_at = '' "
            "WHERE learner_id = ? AND material_id = ?",
            (self.LEARNER, self.MATERIAL),
        )
        timestamp = now_iso()
        self.conn.execute(
            "INSERT INTO assessment_sessions "
            "(id, learner_id, material_id, status, current_index, question_count, "
            "correct_count, information_units, readiness_threshold, created_at, updated_at, completed_at) "
            "VALUES ('mixed-session', ?, ?, 'completed', 2, 2, 1, 3.5, 3.5, ?, ?, ?)",
            (self.LEARNER, self.MATERIAL, timestamp, timestamp, timestamp),
        )
        for concept_id, adjustment, video_id in (
            ("c1", 0.08, "va"),
            ("c2", -0.12, "vb"),
        ):
            self.conn.execute(
                "INSERT INTO assessment_concept_outcomes "
                "(learner_id, session_id, material_id, concept_id, question_count, "
                "correct_count, accuracy, adjustment, source_reel_id, source_video_id, "
                "source_difficulty, created_at) "
                "VALUES (?, 'mixed-session', ?, ?, 1, ?, ?, ?, NULL, ?, 0.8, ?)",
                (
                    self.LEARNER,
                    self.MATERIAL,
                    concept_id,
                    1 if adjustment > 0 else 0,
                    1.0 if adjustment > 0 else 0.0,
                    adjustment,
                    video_id,
                    timestamp,
                ),
            )
        items = [
            self._item("high-score-mastered", "c1", "va", 10, 10, 0.5),
            self._item("weak-alternate", "c2", "vc", 10, 1, 0.3),
        ]
        ordered = self.svc.adaptive_curriculum_order(
            self.conn, self.MATERIAL, self.LEARNER, items
        )
        self.assertEqual(ordered[0]["reel_id"], "weak-alternate")

    def test_all_weak_concepts_in_latest_session_queue_remediation(self) -> None:
        self.conn.execute(
            "UPDATE learner_material_progress SET difficulty_reset_at = '' "
            "WHERE learner_id = ? AND material_id = ?",
            (self.LEARNER, self.MATERIAL),
        )
        timestamp = now_iso()
        self.conn.execute(
            "INSERT INTO assessment_sessions "
            "(id, learner_id, material_id, status, current_index, question_count, "
            "correct_count, information_units, readiness_threshold, created_at, updated_at, completed_at) "
            "VALUES ('two-weak-session', ?, ?, 'completed', 2, 2, 0, 3.5, 3.5, ?, ?, ?)",
            (self.LEARNER, self.MATERIAL, timestamp, timestamp, timestamp),
        )
        for concept_id, video_id in (("c1", "va"), ("c2", "vb")):
            self.conn.execute(
                "INSERT INTO assessment_concept_outcomes "
                "(learner_id, session_id, material_id, concept_id, question_count, "
                "correct_count, accuracy, adjustment, source_reel_id, source_video_id, "
                "source_difficulty, created_at) "
                "VALUES (?, 'two-weak-session', ?, ?, 1, 0, 0.0, -0.12, NULL, ?, 0.8, ?)",
                (self.LEARNER, self.MATERIAL, concept_id, video_id, timestamp),
            )
        items = [
            self._item("high-score", "c1", "va", 10, 100, 0.9),
            self._item("alternate-c1", "c1", "vb", 10, 1, 0.3),
            self._item("alternate-c2", "c2", "vc", 10, 1, 0.3),
        ]
        ordered = self.svc.adaptive_curriculum_order(
            self.conn, self.MATERIAL, self.LEARNER, items
        )
        self.assertEqual(
            [row["reel_id"] for row in ordered[:2]],
            ["alternate-c1", "alternate-c2"],
        )

    def test_assessment_concept_is_one_global_mastery_response(self) -> None:
        self.conn.execute(
            "UPDATE learner_material_progress SET difficulty_reset_at = '' "
            "WHERE learner_id = ? AND material_id = ?",
            (self.LEARNER, self.MATERIAL),
        )
        self._insert_assessment_outcome(
            session_id="global-1", concept_id="c1", adjustment=0.08
        )
        self._insert_assessment_outcome(
            session_id="global-2", concept_id="c2", adjustment=0.08
        )
        self._insert_assessment_outcome(
            session_id="global-3", concept_id="c1", adjustment=0.08
        )
        adjustment = self.svc.update_level_adjustment(
            self.conn, self.MATERIAL, self.LEARNER
        )
        self.assertAlmostEqual(adjustment, 0.20)

    def test_first_quiz_signal_updates_global_drift_without_manual_minimum(self) -> None:
        self.conn.execute(
            "UPDATE learner_material_progress SET difficulty_reset_at = '' "
            "WHERE learner_id = ? AND material_id = ?",
            (self.LEARNER, self.MATERIAL),
        )
        self._insert_assessment_outcome(
            session_id="first-quiz", concept_id="c1", adjustment=-0.12
        )
        adjustment = self.svc.update_level_adjustment(
            self.conn, self.MATERIAL, self.LEARNER
        )
        self.assertAlmostEqual(adjustment, -0.12)


if __name__ == "__main__":
    unittest.main()
