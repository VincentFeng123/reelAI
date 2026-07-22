"""Focused adaptive curriculum and learner-feedback contract tests."""
from __future__ import annotations

import json
import sqlite3
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.db import SCHEMA, _migrate_reel_feedback_uniqueness_sqlite
from backend.app.db import now_iso
from backend.app.ingestion import persistence as persistence_module
from backend.app.ingestion.persistence import ensure_clip_concept
from backend.app.services import reels as reels_module
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

    def _set_family_context(
        self,
        *,
        reel_id: str,
        concept_id: str,
        family: str,
        aliases: list[str] | None = None,
        contract_version: str = "concept_family_v3",
        selection_authority: str = "gemini",
    ) -> dict:
        context = {
            "selection_contract_version": "quality_silence_v40",
            "selection_authority": selection_authority,
            "concept_family_contract_version": contract_version,
            "concept_family": family,
            "concept_aliases": [] if aliases is None else aliases,
        }
        self.conn.execute(
            "UPDATE reels SET search_context_json = ? WHERE id = ?",
            (json.dumps(context), reel_id),
        )
        persistence_module._record_concept_family_profile(
            self.conn,
            material_id=self.MATERIAL,
            concept_id=concept_id,
            search_context=context,
        )
        return context

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

    def test_fresh_learner_family_adaptation_adds_no_profile_query(self) -> None:
        items = [
            self._item("fresh-a", "c1", "va", 10, 10, 0.2),
            self._item("fresh-b", "c2", "vb", 10, 9, 0.2),
        ]
        with mock.patch.object(
            self.svc,
            "_persisted_concept_family_profiles",
            side_effect=AssertionError("fresh feeds must not load family profiles"),
        ):
            ordered = self.svc.adaptive_curriculum_order(
                self.conn,
                self.MATERIAL,
                self.LEARNER,
                items,
            )
        self.assertEqual([row["reel_id"] for row in ordered], ["fresh-a", "fresh-b"])

    def test_post_feedback_family_profile_query_is_one_row_per_relevant_concept(
        self,
    ) -> None:
        for concept_id, title in (
            ("c1", "formula and units"),
            ("c2", "force calculation and units"),
            ("other", "Newton's first-law inertia"),
        ):
            if concept_id in {"c1", "c2"}:
                self.conn.execute(
                    "UPDATE concepts SET title = ? WHERE id = ?",
                    (title, concept_id),
                )
            else:
                self.conn.execute(
                    "INSERT INTO concepts "
                    "(id, material_id, title, keywords_json, summary, "
                    "embedding_json, created_at) "
                    "VALUES (?, ?, ?, '[]', '', NULL, "
                    "'2026-07-09T00:00:01+00:00')",
                    (concept_id, self.MATERIAL, title),
                )
        family_context = {
            "selection_contract_version": "quality_silence_v40",
            "selection_authority": "gemini",
            "concept_family_contract_version": "concept_family_v3",
            "concept_family": "Newton's second law of motion",
            "concept_aliases": [],
            "irrelevant_payload": "x" * 512,
        }
        self.assertTrue(persistence_module.upsert_reel_row(
            self.conn,
            reel_id="family-signal",
            material_id=self.MATERIAL,
            concept_id="c1",
            video_id="va",
            video_url="https://youtube.test/va",
            t_start=1.0,
            t_end=21.0,
            transcript_snippet="signal",
            takeaways=[],
            difficulty=0.4,
            search_context=family_context,
        ))
        for index in range(400):
            start = 31.0 + index * 25.0
            self.assertTrue(persistence_module.upsert_reel_row(
                self.conn,
                reel_id=f"same-concept-history-{index}",
                material_id=self.MATERIAL,
                concept_id="c2",
                video_id="vb",
                video_url="https://youtube.test/vb",
                t_start=start,
                t_end=start + 20.0,
                transcript_snippet="same relevant concept",
                takeaways=[],
                difficulty=0.4,
                search_context=family_context,
            ))
        profile_counts = self.conn.execute(
            "SELECT concept_id, COUNT(*) AS profile_count "
            "FROM concept_family_profiles GROUP BY concept_id ORDER BY concept_id"
        ).fetchall()
        self.assertEqual(
            [(row["concept_id"], row["profile_count"]) for row in profile_counts],
            [("c1", 1), ("c2", 1)],
        )

        self.svc.record_feedback(
            self.conn,
            "family-signal",
            helpful=True,
            confusing=False,
            rating=None,
            saved=False,
            learner_id=self.LEARNER,
        )

        original_fetch_all = reels_module.fetch_all
        profile_reads: list[tuple[str, tuple[object, ...], int]] = []
        reel_json_profile_reads: list[str] = []

        def traced_fetch_all(conn, query, params=()):
            result = original_fetch_all(conn, query, params)
            normalized_query = " ".join(query.split())
            if "FROM concept_family_profiles" in normalized_query:
                profile_reads.append((query, tuple(params), len(result)))
            if (
                "SELECT concept_id, search_context_json FROM reels"
                in normalized_query
            ):
                reel_json_profile_reads.append(query)
            return result

        with mock.patch.object(
            reels_module,
            "fetch_all",
            side_effect=traced_fetch_all,
        ):
            ordered = self.svc.adaptive_curriculum_order(
                self.conn,
                self.MATERIAL,
                self.LEARNER,
                [
                    self._item("same-family", "c2", "vb", 80, 10, 0.4),
                    self._item("different", "other", "vc", 80, 1, 0.4),
                ],
            )

        self.assertEqual(ordered[0]["reel_id"], "different")
        self.assertEqual(reel_json_profile_reads, [])
        self.assertEqual(len(profile_reads), 1)
        query, params, row_count = profile_reads[0]
        self.assertIn("concept_id IN", " ".join(query.split()))
        self.assertEqual(set(params), {self.MATERIAL, "c1", "c2", "other"})
        self.assertEqual(row_count, 2)

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

    def test_adaptive_order_propagates_only_trusted_family_thumb_signals(self) -> None:
        family = {
            "action": "action-reaction pairs",
            "identify": "identifying action-reaction pairs",
            "gravity": "gravitational action-reaction pairs",
            "misconception": "action-reaction acceleration misconception",
        }
        for concept_id, title in family.items():
            self.conn.execute(
                "INSERT INTO concepts "
                "(id, material_id, title, keywords_json, summary, embedding_json, created_at) "
                "VALUES (?, ?, ?, '[]', '', NULL, '2026-07-09T00:00:01+00:00')",
                (concept_id, self.MATERIAL, title),
            )
        self._insert_reel("watched-family", "action", "va", 1, 0.8)
        for index, concept_id in enumerate(family):
            reel_id = (
                "watched-family"
                if concept_id == "action"
                else f"family-profile-{concept_id}"
            )
            if concept_id != "action":
                self._insert_reel(
                    reel_id,
                    concept_id,
                    ("vb", "vc", "vb")[index - 1],
                    30 + index * 20,
                    0.4,
                )
            self._set_family_context(
                reel_id=reel_id,
                concept_id=concept_id,
                family="Newton's third law of motion",
            )

        self.svc.record_feedback(
            self.conn,
            "watched-family",
            helpful=True,
            confusing=False,
            rating=None,
            saved=False,
            learner_id=self.LEARNER,
        )
        coverage, adjustments, _, _ = self.svc._learner_adaptation_context(
            self.conn, self.MATERIAL, self.LEARNER
        )
        self.assertEqual(
            coverage,
            {"action": {"helpful": 1.0, "confusing": 0.0}},
        )
        self.assertEqual(set(adjustments), {"action"})
        self.assertAlmostEqual(adjustments["action"], 0.04)

        helpful_order = self.svc.adaptive_curriculum_order(
            self.conn,
            self.MATERIAL,
            self.LEARNER,
            [
                self._item("family-repeat", "identify", "vb", 10, 10, 0.3),
                self._item("different", "c2", "vc", 10, 1, 0.3),
            ],
        )
        self.assertEqual(helpful_order[0]["reel_id"], "different")

        self.svc.record_feedback(
            self.conn,
            "watched-family",
            helpful=False,
            confusing=True,
            rating=None,
            saved=False,
            learner_id=self.LEARNER,
        )
        coverage, adjustments, _, _ = self.svc._learner_adaptation_context(
            self.conn, self.MATERIAL, self.LEARNER
        )
        self.assertEqual(
            coverage,
            {"action": {"helpful": 0.0, "confusing": 1.0}},
        )
        self.assertEqual(set(adjustments), {"action"})
        self.assertAlmostEqual(adjustments["action"], -0.06)

        confusing_order = self.svc.adaptive_curriculum_order(
            self.conn,
            self.MATERIAL,
            self.LEARNER,
            [
                self._item("family-remediation", "misconception", "vb", 10, 1, 0.3),
                self._item("different", "c2", "vc", 10, 10, 0.3),
            ],
        )
        self.assertEqual(confusing_order[0]["reel_id"], "family-remediation")

    def test_adaptive_order_propagates_trusted_family_quiz_outcomes(self) -> None:
        for concept_id, title in (
            ("quiz-source", "formula and units"),
            ("quiz-variant", "force calculation and units"),
        ):
            self.conn.execute(
                "INSERT INTO concepts "
                "(id, material_id, title, keywords_json, summary, embedding_json, created_at) "
                "VALUES (?, ?, ?, '[]', '', NULL, '2026-07-09T00:00:01+00:00')",
                (concept_id, self.MATERIAL, title),
            )
        for reel_id, concept_id, video_id, start in (
            ("quiz-source-reel", "quiz-source", "va", 1),
            ("quiz-variant-reel", "quiz-variant", "vb", 30),
        ):
            self._insert_reel(reel_id, concept_id, video_id, start, 0.4)
            self._set_family_context(
                reel_id=reel_id,
                concept_id=concept_id,
                family="Newton's second law of motion",
            )

        self._insert_assessment_outcome(
            session_id="quiz-correct",
            concept_id="quiz-source",
            adjustment=0.10,
            source_reel_id="quiz-source-reel",
        )
        correct_coverage, correct_adjustments, _, _ = (
            self.svc._learner_adaptation_context(
                self.conn,
                self.MATERIAL,
                self.LEARNER,
                propagate_concept_families=True,
                candidate_concept_ids={"quiz-variant"},
            )
        )
        self.assertEqual(correct_coverage["quiz-variant"]["helpful"], 1.0)
        self.assertGreater(correct_adjustments["quiz-variant"], 0.0)
        correct_order = self.svc.adaptive_curriculum_order(
            self.conn,
            self.MATERIAL,
            self.LEARNER,
            [
                self._item("quiz-family-repeat", "quiz-variant", "vb", 80, 10, 0.4),
                self._item("quiz-different", "c2", "vc", 80, 1, 0.4),
            ],
        )
        self.assertEqual(correct_order[0]["reel_id"], "quiz-different")

        self.conn.execute("DELETE FROM assessment_concept_outcomes")
        self.conn.execute("DELETE FROM assessment_sessions")
        self._insert_assessment_outcome(
            session_id="quiz-wrong",
            concept_id="quiz-source",
            adjustment=-0.10,
            source_reel_id="quiz-source-reel",
        )
        wrong_coverage, wrong_adjustments, _, _ = (
            self.svc._learner_adaptation_context(
                self.conn,
                self.MATERIAL,
                self.LEARNER,
                propagate_concept_families=True,
                candidate_concept_ids={"quiz-variant"},
            )
        )
        self.assertEqual(wrong_coverage["quiz-variant"]["confusing"], 1.0)
        self.assertLess(wrong_adjustments["quiz-variant"], 0.0)
        wrong_order = self.svc.adaptive_curriculum_order(
            self.conn,
            self.MATERIAL,
            self.LEARNER,
            [
                self._item("quiz-family-remediation", "quiz-variant", "vb", 80, 1, 0.3),
                self._item("quiz-different", "c2", "vc", 80, 10, 0.5),
            ],
        )
        self.assertEqual(wrong_order[0]["reel_id"], "quiz-family-remediation")

    def test_ai_canonical_family_profiles_do_not_merge_untrusted_titles(self) -> None:
        self.assertEqual(
            self.svc._concept_family_identity("Newton's 1st law"),
            self.svc._concept_family_identity("Newton's first law"),
        )
        self.assertEqual(
            self.svc._concept_family_identity("Kepler's 5th law"),
            self.svc._concept_family_identity("Kepler's fifth law"),
        )
        self.assertEqual(
            self.svc._concept_family_identity("Asimov's 0th law"),
            self.svc._concept_family_identity("Asimov's zeroth law"),
        )
        self.assertEqual(
            self.svc._concept_family_identity("Twenty-first Amendment"),
            self.svc._concept_family_identity("21st Amendment"),
        )

        concepts = [
            {"id": "first", "title": "Newton's first law"},
            {"id": "inertia", "title": "inertia and motion"},
            {"id": "second", "title": "Newton's second law overview"},
            {"id": "fma", "title": "force-mass-acceleration proportionality"},
            {"id": "thermo", "title": "first law of thermodynamics"},
        ]

        def profile(*values: str) -> set[str]:
            return {
                identity
                for identity in (
                    self.svc._concept_family_identity(value) for value in values
                )
                if identity
            }

        families = self.svc._concept_family_ids(
            concepts,
            {
                "first": profile("Newton's first law"),
                "inertia": profile("Newton's first law"),
                "second": profile("Newton's second law overview"),
                "fma": profile("force-mass-acceleration proportionality"),
                "thermo": profile("first law of thermodynamics"),
            },
        )
        self.assertIn("inertia", families["first"])
        self.assertIn("first", families["inertia"])
        self.assertNotIn("fma", families["second"])
        self.assertNotIn("second", families["fma"])
        self.assertNotIn("second", families["first"])
        self.assertNotIn("first", families["second"])
        self.assertNotIn("thermo", families["first"])
        self.assertNotIn("first", families["thermo"])
        self.assertEqual(self.svc._concept_family_identity("first law"), "")

        rollout_families = self.svc._concept_family_ids(
            [
                {"id": "legacy-first", "title": "Newton's first law"},
                {"id": "new-inertia", "title": "law of inertia"},
            ],
            {
                "new-inertia": profile("Newton's first law"),
            },
        )
        self.assertIn("new-inertia", rollout_families["legacy-first"])
        self.assertIn("legacy-first", rollout_families["new-inertia"])

        untrusted_families = self.svc._concept_family_ids(
            [
                {"id": "blood-a", "title": "blood type A"},
                {"id": "blood-b", "title": "blood type B"},
                {"id": "inertia", "title": "law of inertia"},
                {"id": "first", "title": "Newton's first law"},
            ],
            {},
        )
        self.assertEqual(untrusted_families["blood-a"], {"blood-a"})
        self.assertEqual(untrusted_families["blood-b"], {"blood-b"})
        self.assertEqual(untrusted_families["inertia"], {"inertia"})
        self.assertEqual(untrusted_families["first"], {"first"})

        comparison_families = self.svc._concept_family_ids(
            [
                {"id": "newton-members", "title": "Newton's first and second laws"},
                {"id": "newton-overview", "title": "Newton's laws comparison"},
                {"id": "world-members", "title": "World War I and II comparison"},
                {
                    "id": "world-overview",
                    "title": "World War I and II comparison overview",
                },
            ],
            {
                "newton-members": profile("Newtonian laws comparison"),
                "newton-overview": profile("Newtonian laws comparison"),
                "world-members": profile("World Wars comparison"),
                "world-overview": profile("World Wars comparison"),
            },
        )
        self.assertIn("newton-overview", comparison_families["newton-members"])
        self.assertIn("newton-members", comparison_families["newton-overview"])
        self.assertIn("world-overview", comparison_families["world-members"])
        self.assertIn("world-members", comparison_families["world-overview"])

        terminal_notation_families = self.svc._concept_family_ids(
            [
                {"id": "swift-nullable", "title": "Swift String?"},
                {"id": "swift-plain", "title": "Swift String"},
                {"id": "factorial", "title": "factorial n!"},
                {"id": "plain-n", "title": "factorial n"},
                {
                    "id": "typescript-bang",
                    "title": "TypeScript non-null assertion !",
                },
                {
                    "id": "typescript-plain",
                    "title": "TypeScript non-null assertion",
                },
            ],
            {
                "swift-nullable": profile("Swift String?"),
                "swift-plain": profile("Swift String"),
                "factorial": profile("factorial n!"),
                "plain-n": profile("factorial n"),
                "typescript-bang": profile("TypeScript non-null assertion !"),
                "typescript-plain": profile("TypeScript non-null assertion"),
            },
        )
        for marked, plain in (
            ("swift-nullable", "swift-plain"),
            ("factorial", "plain-n"),
            ("typescript-bang", "typescript-plain"),
        ):
            self.assertNotIn(plain, terminal_notation_families[marked])
            self.assertNotIn(marked, terminal_notation_families[plain])

    def test_connected_family_profile_uses_one_canonical_ai_identity(self) -> None:
        contexts = (
            ("Newton's first law", []),
            ("Newton's first law", []),
            ("Newton's first law", []),
        )
        for index, ((family, aliases), video_id) in enumerate(
            zip(contexts, ("va", "vb", "vc"))
        ):
            reel_id = f"connected-family-{index}"
            self._insert_reel(reel_id, "c1", video_id, index * 30 + 1, 0.4)
            self._set_family_context(
                reel_id=reel_id,
                concept_id="c1",
                family=family,
                aliases=aliases,
            )

        profiles = self.svc._persisted_concept_family_profiles(
            self.conn,
            self.MATERIAL,
        )
        self.assertEqual(
            profiles["c1"],
            {self.svc._concept_family_identity("Newton's first law")},
        )

        self._insert_reel("corrupt-alias-profile", "c2", "va", 120, 0.4)
        self._set_family_context(
            reel_id="corrupt-alias-profile",
            concept_id="c2",
            family="Newton's first law",
            aliases=["photosynthesis"],
        )
        profiles = self.svc._persisted_concept_family_profiles(
            self.conn,
            self.MATERIAL,
        )
        self.assertNotIn("c2", profiles)

    def test_previous_broad_family_contract_is_not_trusted_for_adaptation(
        self,
    ) -> None:
        self._insert_reel("stale-broad-family", "c1", "va", 1, 0.4)
        self.conn.execute(
            "UPDATE reels SET search_context_json = ? WHERE id = ?",
            (
                json.dumps({
                    "selection_contract_version": "quality_silence_v38",
                    "selection_authority": "gemini",
                    "concept_family_contract_version": "concept_family_v2",
                    "concept_family": "Newton's second law of motion",
                    "concept_aliases": [],
                }),
                "stale-broad-family",
            ),
        )

        profiles = self.svc._persisted_concept_family_profiles(
            self.conn,
            self.MATERIAL,
        )

        self.assertNotIn("c1", profiles)

    def test_stale_gemini_family_signals_are_excluded_from_v3_adaptation(
        self,
    ) -> None:
        self.conn.execute(
            "UPDATE learner_material_progress SET difficulty_reset_at = '' "
            "WHERE learner_id = ? AND material_id = ?",
            (self.LEARNER, self.MATERIAL),
        )
        for reel_id, concept_id, video_id, start in (
            ("stale-v2", "c1", "va", 1),
            ("current-v3", "c1", "vb", 30),
            ("legacy-unversioned", "c2", "vc", 60),
        ):
            self._insert_reel(reel_id, concept_id, video_id, start, 0.6)
        self.conn.execute(
            "UPDATE reels SET search_context_json = ? WHERE id = 'stale-v2'",
            (json.dumps({
                "selection_authority": "gemini",
                "concept_family_contract_version": "concept_family_v2",
                "concept_family": "Newton's second law of motion",
                "concept_aliases": [],
            }),),
        )
        self.conn.execute(
            "UPDATE reels SET search_context_json = ? WHERE id = 'current-v3'",
            (json.dumps({
                "selection_authority": "gemini",
                "concept_family_contract_version": "concept_family_v3",
                "concept_family": "force-mass-acceleration proportionality",
                "concept_aliases": [],
            }),),
        )
        for reel_id, helpful, confusing in (
            ("stale-v2", False, True),
            ("current-v3", True, False),
            ("legacy-unversioned", False, True),
        ):
            self.svc.record_feedback(
                self.conn,
                reel_id,
                helpful=helpful,
                confusing=confusing,
                rating=None,
                saved=False,
                learner_id=self.LEARNER,
            )
        self.conn.execute(
            "UPDATE reel_feedback SET mastery_updated_at = CASE reel_id "
            "WHEN 'current-v3' THEN '2026-07-20T00:00:00+00:00' "
            "WHEN 'legacy-unversioned' THEN '2026-07-21T00:00:00+00:00' "
            "ELSE '2099-01-01T00:00:00+00:00' END "
            "WHERE learner_id = ?",
            (self.LEARNER,),
        )
        self._insert_assessment_outcome(
            session_id="stale-v2-quiz",
            concept_id="c1",
            adjustment=-0.12,
            source_reel_id="stale-v2",
        )
        self._insert_assessment_outcome(
            session_id="current-v3-quiz",
            concept_id="c1",
            adjustment=0.08,
            source_reel_id="current-v3",
        )
        self._insert_assessment_outcome(
            session_id="legacy-unversioned-quiz",
            concept_id="c2",
            adjustment=-0.12,
            source_reel_id="legacy-unversioned",
        )
        self.conn.execute(
            "UPDATE assessment_concept_outcomes SET created_at = CASE session_id "
            "WHEN 'current-v3-quiz' THEN '2026-07-20T01:00:00+00:00' "
            "WHEN 'legacy-unversioned-quiz' THEN '2026-07-21T01:00:00+00:00' "
            "ELSE '2099-01-01T01:00:00+00:00' END "
            "WHERE learner_id = ?",
            (self.LEARNER,),
        )

        coverage, adjustments, latest, _ = self.svc._learner_adaptation_context(
            self.conn,
            self.MATERIAL,
            self.LEARNER,
        )

        self.assertEqual(
            coverage,
            {
                "c1": {"helpful": 2.0, "confusing": 0.0},
                "c2": {"helpful": 0.0, "confusing": 2.0},
            },
        )
        self.assertAlmostEqual(adjustments["c1"], 0.12)
        self.assertAlmostEqual(adjustments["c2"], -0.18)
        self.assertEqual((latest or {}).get("session_id"), "legacy-unversioned-quiz")
        self.assertEqual(
            [
                row["concept_id"]
                for row in (latest or {}).get("assessment_remediations") or []
            ],
            ["c2"],
        )
        self.assertAlmostEqual(
            self.svc.update_level_adjustment(
                self.conn,
                self.MATERIAL,
                self.LEARNER,
            ),
            -0.06,
        )

    def test_stale_reels_do_not_affect_acquisition_but_legacy_rows_still_do(
        self,
    ) -> None:
        self._insert_reel("stale-helpful", "c1", "va", 1, 0.4)
        self._insert_reel("current-count", "c2", "vb", 30, 0.4)
        self.conn.execute(
            "UPDATE reels SET search_context_json = ? WHERE id = 'stale-helpful'",
            (json.dumps({
                "selection_authority": "gemini",
                "concept_family_contract_version": "concept_family_v2",
                "concept_family": "Newton's second law of motion",
                "concept_aliases": [],
            }),),
        )
        self.conn.execute(
            "UPDATE reels SET search_context_json = ? WHERE id = 'current-count'",
            (json.dumps({
                "selection_authority": "gemini",
                "concept_family_contract_version": "concept_family_v3",
                "concept_family": "net force vector sum",
                "concept_aliases": [],
            }),),
        )
        self.svc.record_feedback(
            self.conn,
            "stale-helpful",
            helpful=True,
            confusing=False,
            rating=5,
            saved=False,
            learner_id=self.LEARNER,
        )
        self._insert_assessment_outcome(
            session_id="stale-helpful-quiz",
            concept_id="c1",
            adjustment=0.08,
            source_reel_id="stale-helpful",
        )
        concepts = [
            dict(row)
            for row in self.conn.execute(
                "SELECT * FROM concepts WHERE material_id = ? ORDER BY id",
                (self.MATERIAL,),
            ).fetchall()
        ]
        self.assertEqual(
            [row["id"] for row in self.svc._order_concepts(
                self.conn, self.MATERIAL, concepts, self.LEARNER
            )],
            ["c1", "c2"],
        )

        self._insert_reel("legacy-helpful", "c1", "vc", 60, 0.4)
        self.svc.record_feedback(
            self.conn,
            "legacy-helpful",
            helpful=True,
            confusing=False,
            rating=5,
            saved=False,
            learner_id=self.LEARNER,
        )
        self.assertEqual(
            [row["id"] for row in self.svc._order_concepts(
                self.conn, self.MATERIAL, concepts, self.LEARNER
            )],
            ["c2", "c1"],
        )

    def test_persisted_family_profiles_reject_conflicting_shared_facet(self) -> None:
        self.conn.execute("UPDATE concepts SET title = 'Newton laws' WHERE id = 'c1'")
        contexts = (
            {
                "selection_contract_version": "quality_silence_v39",
                "selection_authority": "gemini",
                "concept_family_contract_version": "concept_family_v3",
                "concept_family": "Newton's first law",
                "concept_aliases": [],
            },
            {
                "selection_contract_version": "quality_silence_v39",
                "selection_authority": "gemini",
                "concept_family_contract_version": "concept_family_v3",
                "concept_family": "Newton's second law",
                "concept_aliases": [],
            },
        )
        for index, (video_id, context) in enumerate(zip(("va", "vb"), contexts)):
            reel_id = f"conflicting-family-{index}"
            self._insert_reel(reel_id, "c1", video_id, index * 30 + 1, 0.4)
            self._set_family_context(
                reel_id=reel_id,
                concept_id="c1",
                family=str(context["concept_family"]),
            )

        profiles = self.svc._persisted_concept_family_profiles(
            self.conn,
            self.MATERIAL,
        )
        self.assertNotIn("c1", profiles)

    def test_family_scoped_concept_ids_isolate_conflicting_shared_facets(self) -> None:
        newton_id, _, _ = ensure_clip_concept(
            self.conn,
            material_id=self.MATERIAL,
            title="first law",
            semantic_identity="Newton's first law",
        )
        thermo_id, _, _ = ensure_clip_concept(
            self.conn,
            material_id=self.MATERIAL,
            title="first law",
            semantic_identity="first law of thermodynamics",
        )
        ordinal_variant_id, _, _ = ensure_clip_concept(
            self.conn,
            material_id=self.MATERIAL,
            title="law of inertia",
            semantic_identity="Newton's 1st law",
        )
        self.assertEqual(newton_id, ordinal_variant_id)
        self.assertNotEqual(newton_id, thermo_id)

        python_version_id = ensure_clip_concept(
            self.conn,
            material_id=self.MATERIAL,
            title="Python version 3.12 typing",
            semantic_identity="Python version 3.12 typing",
        )[0]
        python_decimal_id = ensure_clip_concept(
            self.conn,
            material_id=self.MATERIAL,
            title="Python 3.12 typing",
            semantic_identity="Python 3.12 typing",
        )[0]
        python_previous_id = ensure_clip_concept(
            self.conn,
            material_id=self.MATERIAL,
            title="Python 3.11 typing",
            semantic_identity="Python 3.11 typing",
        )[0]
        self.assertEqual(python_version_id, python_decimal_id)
        self.assertNotEqual(python_version_id, python_previous_id)

        language_ids = {
            language: ensure_clip_concept(
                self.conn,
                material_id=self.MATERIAL,
                title=f"{language} memory management",
                semantic_identity=f"{language} memory management",
            )[0]
            for language in ("C", "C++", "C#")
        }
        self.assertEqual(len(set(language_ids.values())), 3)
        operator_ids = {
            operator: ensure_clip_concept(
                self.conn,
                material_id=self.MATERIAL,
                title=f"JavaScript {operator} operator",
                semantic_identity=f"JavaScript {operator} operator",
            )[0]
            for operator in ("&&", "||", "??")
        }
        self.assertEqual(len(set(operator_ids.values())), 3)
        terminal_notation_pairs = (
            ("Swift nullable type String?", "Swift nullable type String"),
            ("factorial operation n!", "factorial operation n"),
            (
                "TypeScript non-null assertion !",
                "TypeScript non-null assertion",
            ),
        )
        for marked, plain in terminal_notation_pairs:
            marked_id = ensure_clip_concept(
                self.conn,
                material_id=self.MATERIAL,
                title=marked,
                semantic_identity=marked,
            )[0]
            plain_id = ensure_clip_concept(
                self.conn,
                material_id=self.MATERIAL,
                title=plain,
                semantic_identity=plain,
            )[0]
            self.assertNotEqual(marked_id, plain_id)

        for index, (concept_id, family, video_id) in enumerate((
            (newton_id, "Newton's first law", "va"),
            (
                thermo_id,
                "first law of thermodynamics",
                "vb",
            ),
        )):
            reel_id = f"isolated-family-{index}"
            self._insert_reel(reel_id, concept_id, video_id, index * 30 + 1, 0.4)
            self.conn.execute(
                "UPDATE reels SET search_context_json = ? WHERE id = ?",
                (
                    json.dumps({
                        "selection_contract_version": "quality_silence_v39",
                        "selection_authority": "gemini",
                        "concept_family_contract_version": "concept_family_v3",
                        "concept_family": family,
                        "concept_aliases": [],
                    }),
                    reel_id,
                ),
            )

        self.svc.record_feedback(
            self.conn,
            "isolated-family-0",
            helpful=True,
            confusing=False,
            rating=None,
            saved=False,
            learner_id=self.LEARNER,
        )
        coverage, adjustments, _, _ = self.svc._learner_adaptation_context(
            self.conn,
            self.MATERIAL,
            self.LEARNER,
        )
        self.assertEqual(
            coverage[newton_id],
            {"helpful": 1.0, "confusing": 0.0},
        )
        self.assertNotIn(thermo_id, coverage)
        self.assertAlmostEqual(adjustments[newton_id], 0.04)
        self.assertNotIn(thermo_id, adjustments)

    def test_case_sensitive_single_letter_concepts_isolate_thumb_and_quiz_signals(
        self,
    ) -> None:
        uppercase_id, uppercase_title, uppercase_key = ensure_clip_concept(
            self.conn,
            material_id=self.MATERIAL,
            title="C",
        )
        lowercase_id, lowercase_title, lowercase_key = ensure_clip_concept(
            self.conn,
            material_id=self.MATERIAL,
            title="c",
        )
        self.assertNotEqual(uppercase_id, lowercase_id)
        self.assertEqual((uppercase_title, uppercase_key), ("C", "C"))
        self.assertEqual((lowercase_title, lowercase_key), ("c", "c"))

        self._insert_reel("uppercase-C", uppercase_id, "va", 1, 0.4)
        self._insert_reel("lowercase-c", lowercase_id, "vb", 31, 0.4)
        self.svc.record_feedback(
            self.conn,
            "uppercase-C",
            helpful=True,
            confusing=False,
            rating=None,
            saved=False,
            learner_id=self.LEARNER,
        )
        self.svc.record_feedback(
            self.conn,
            "lowercase-c",
            helpful=False,
            confusing=True,
            rating=None,
            saved=False,
            learner_id=self.LEARNER,
        )
        coverage, adjustments, _, _ = self.svc._learner_adaptation_context(
            self.conn,
            self.MATERIAL,
            self.LEARNER,
        )
        self.assertEqual(
            coverage,
            {
                uppercase_id: {"helpful": 1.0, "confusing": 0.0},
                lowercase_id: {"helpful": 0.0, "confusing": 1.0},
            },
        )
        self.assertAlmostEqual(adjustments[uppercase_id], 0.04)
        self.assertAlmostEqual(adjustments[lowercase_id], -0.06)

        self.conn.execute(
            "UPDATE learner_material_progress SET difficulty_reset_at = '' "
            "WHERE learner_id = ? AND material_id = ?",
            (self.LEARNER, self.MATERIAL),
        )
        self._insert_assessment_outcome(
            session_id="lowercase-c-wrong-quiz",
            concept_id=lowercase_id,
            adjustment=-0.12,
            video_id="vb",
            difficulty=0.4,
        )
        _, combined, _, _ = self.svc._learner_adaptation_context(
            self.conn,
            self.MATERIAL,
            self.LEARNER,
        )
        self.assertAlmostEqual(combined[uppercase_id], 0.04)
        self.assertAlmostEqual(combined[lowercase_id], -0.18)

    def test_unicode_case_sensitive_symbols_isolate_persistence_and_adaptation(
        self,
    ) -> None:
        uppercase_id, uppercase_title, uppercase_key = ensure_clip_concept(
            self.conn,
            material_id=self.MATERIAL,
            title="Δ",
        )
        lowercase_id, lowercase_title, lowercase_key = ensure_clip_concept(
            self.conn,
            material_id=self.MATERIAL,
            title="δ",
        )
        self.assertNotEqual(uppercase_id, lowercase_id)
        self.assertEqual((uppercase_title, uppercase_key), ("Δ", "Δ"))
        self.assertEqual((lowercase_title, lowercase_key), ("δ", "δ"))

        self._insert_reel("uppercase-delta", uppercase_id, "va", 1, 0.4)
        self._insert_reel("lowercase-delta", lowercase_id, "vb", 31, 0.4)
        self.svc.record_feedback(
            self.conn,
            "uppercase-delta",
            helpful=True,
            confusing=False,
            rating=None,
            saved=False,
            learner_id=self.LEARNER,
        )
        self.svc.record_feedback(
            self.conn,
            "lowercase-delta",
            helpful=False,
            confusing=True,
            rating=None,
            saved=False,
            learner_id=self.LEARNER,
        )
        coverage, adjustments, _, _ = self.svc._learner_adaptation_context(
            self.conn,
            self.MATERIAL,
            self.LEARNER,
        )
        self.assertEqual(
            coverage,
            {
                uppercase_id: {"helpful": 1.0, "confusing": 0.0},
                lowercase_id: {"helpful": 0.0, "confusing": 1.0},
            },
        )
        self.assertAlmostEqual(adjustments[uppercase_id], 0.04)
        self.assertAlmostEqual(adjustments[lowercase_id], -0.06)

        self.conn.execute(
            "UPDATE learner_material_progress SET difficulty_reset_at = '' "
            "WHERE learner_id = ? AND material_id = ?",
            (self.LEARNER, self.MATERIAL),
        )
        self._insert_assessment_outcome(
            session_id="lowercase-delta-wrong-quiz",
            concept_id=lowercase_id,
            adjustment=-0.12,
            video_id="vb",
            difficulty=0.4,
        )
        _, combined, _, _ = self.svc._learner_adaptation_context(
            self.conn,
            self.MATERIAL,
            self.LEARNER,
        )
        self.assertAlmostEqual(combined[uppercase_id], 0.04)
        self.assertAlmostEqual(combined[lowercase_id], -0.18)

    def test_narrow_physics_families_keep_feedback_and_quiz_signals_isolated(
        self,
    ) -> None:
        facets = (
            ("units-definition", "Force measurement units", "SI force units", "va"),
            ("units-paraphrase", "Newtons as force units", "SI force units", "vb"),
            (
                "proportionality",
                "How force and mass change acceleration",
                "force-mass-acceleration proportionality",
                "vc",
            ),
            ("net-force", "Adding forces", "net force vector sum", "va"),
            (
                "frictionless",
                "Acceleration without friction",
                "frictionless Newton second-law application",
                "vb",
            ),
            (
                "friction",
                "Acceleration with friction",
                "frictional Newton second-law application",
                "vc",
            ),
        )
        concept_ids: dict[str, str] = {}
        for index, (key, title, family, video_id) in enumerate(facets):
            concept_id = ensure_clip_concept(
                self.conn,
                material_id=self.MATERIAL,
                title=title,
                semantic_identity=family,
            )[0]
            concept_ids[key] = concept_id
            reel_id = f"narrow-family-{key}"
            self._insert_reel(
                reel_id,
                concept_id,
                video_id,
                index * 30 + 1,
                0.4,
            )
            self.conn.execute(
                "UPDATE reels SET search_context_json = ? WHERE id = ?",
                (
                    json.dumps({
                        "selection_contract_version": "quality_silence_v39",
                        "selection_authority": "gemini",
                        "concept_family_contract_version": "concept_family_v3",
                        "concept_family": family,
                        "concept_aliases": [],
                    }),
                    reel_id,
                ),
            )

        self.assertEqual(
            concept_ids["units-definition"],
            concept_ids["units-paraphrase"],
        )
        self.assertEqual(len(set(concept_ids.values())), 5)

        units_id = concept_ids["units-definition"]
        proportionality_id = concept_ids["proportionality"]
        self.svc.record_feedback(
            self.conn,
            "narrow-family-units-definition",
            helpful=True,
            confusing=False,
            rating=None,
            saved=False,
            learner_id=self.LEARNER,
        )
        coverage, adjustments, _, _ = self.svc._learner_adaptation_context(
            self.conn,
            self.MATERIAL,
            self.LEARNER,
        )
        self.assertEqual(
            coverage,
            {units_id: {"helpful": 1.0, "confusing": 0.0}},
        )
        self.assertEqual(set(adjustments), {units_id})
        self.assertAlmostEqual(adjustments[units_id], 0.04)

        self.conn.execute(
            "UPDATE learner_material_progress SET difficulty_reset_at = '' "
            "WHERE learner_id = ? AND material_id = ?",
            (self.LEARNER, self.MATERIAL),
        )
        self._insert_assessment_outcome(
            session_id="narrow-family-wrong-quiz",
            concept_id=proportionality_id,
            adjustment=-0.12,
            video_id="vc",
            difficulty=0.4,
        )
        coverage, adjustments, latest, _ = self.svc._learner_adaptation_context(
            self.conn,
            self.MATERIAL,
            self.LEARNER,
        )
        self.assertEqual(
            coverage,
            {
                units_id: {"helpful": 1.0, "confusing": 0.0},
                proportionality_id: {"helpful": 0.0, "confusing": 1.0},
            },
        )
        self.assertAlmostEqual(adjustments[units_id], 0.04)
        self.assertAlmostEqual(adjustments[proportionality_id], -0.12)
        self.assertEqual(set(adjustments), {units_id, proportionality_id})
        self.assertEqual(
            set((latest or {}).get("concept_family_ids") or []),
            {proportionality_id},
        )

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
        source_reel_id: str | None = None,
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
            "VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?)",
            (
                learner,
                session_id,
                self.MATERIAL,
                concept_id,
                1 if adjustment > 0 else 0,
                1.0 if adjustment > 0 else 0.0,
                adjustment,
                source_reel_id,
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

    def test_quiz_attribution_stays_exact_but_trusted_family_order_adapts(self) -> None:
        for concept_id, title in (
            ("action", "action-reaction pairs"),
            ("identify", "identifying action-reaction pairs"),
            ("misconception", "action-reaction acceleration misconception"),
        ):
            self.conn.execute(
                "INSERT INTO concepts "
                "(id, material_id, title, keywords_json, summary, embedding_json, created_at) "
                "VALUES (?, ?, ?, '[]', '', NULL, '2026-07-09T00:00:01+00:00')",
                (concept_id, self.MATERIAL, title),
            )
        for index, concept_id in enumerate(("action", "identify", "misconception")):
            reel_id = f"quiz-family-profile-{concept_id}"
            self._insert_reel(
                reel_id,
                concept_id,
                ("va", "vb", "vc")[index],
                1 + index * 30,
                0.4,
            )
            self._set_family_context(
                reel_id=reel_id,
                concept_id=concept_id,
                family="Newton's third law of motion",
            )
        self.conn.execute(
            "UPDATE learner_material_progress SET difficulty_reset_at = '' "
            "WHERE learner_id = ? AND material_id = ?",
            (self.LEARNER, self.MATERIAL),
        )
        self._insert_assessment_outcome(
            session_id="quiz-right",
            concept_id="action",
            adjustment=0.08,
        )
        concepts = [
            dict(row)
            for row in self.conn.execute(
                "SELECT * FROM concepts WHERE material_id = ? ORDER BY id",
                (self.MATERIAL,),
            ).fetchall()
        ]
        ordered = self.svc._order_concepts(
            self.conn, self.MATERIAL, concepts, self.LEARNER
        )
        positions = {row["id"]: index for index, row in enumerate(ordered)}
        self.assertLess(positions["c2"], positions["action"])
        self.assertLess(positions["identify"], positions["action"])
        self.assertLess(positions["misconception"], positions["action"])

        self._insert_assessment_outcome(
            session_id="quiz-wrong",
            concept_id="action",
            adjustment=-0.12,
            difficulty=0.8,
        )
        coverage, adjustments, latest, _ = self.svc._learner_adaptation_context(
            self.conn, self.MATERIAL, self.LEARNER
        )
        self.assertEqual(
            coverage,
            {"action": {"helpful": 1.0, "confusing": 1.0}},
        )
        self.assertEqual(set(adjustments), {"action"})
        self.assertAlmostEqual(adjustments["action"], -0.04)
        self.assertEqual(latest["concept_family_ids"], ["action"])

        remediation = self.svc.adaptive_curriculum_order(
            self.conn,
            self.MATERIAL,
            self.LEARNER,
            [
                self._item("related-easier", "identify", "vb", 10, 1, 0.3),
                self._item("different", "c2", "vc", 10, 10, 0.3),
            ],
        )
        self.assertEqual(remediation[0]["reel_id"], "related-easier")

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
