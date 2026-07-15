"""Difficulty-stage and value ordering for versioned clipping selections."""
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

from backend.app.db import SCHEMA
from backend.app.clip_engine.provider_cache import (
    TRANSCRIPT_SCHEMA_VERSION,
    transcript_artifact_key,
)
from backend.app.services.reels import ReelService


class SelectionContractOrderingTests(unittest.TestCase):
    MATERIAL = "selection-material"
    LEARNER = "owner:selection-learner"

    def setUp(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.execute(
            "INSERT INTO materials "
            "(id, subject_tag, raw_text, source_type, source_path, knowledge_level, created_at) "
            "VALUES (?, 'chemistry', 'chemical bonding', 'topic', NULL, 'beginner', "
            "'2026-07-12T00:00:00+00:00')",
            (self.MATERIAL,),
        )
        self.conn.execute(
            "INSERT INTO concepts "
            "(id, material_id, title, keywords_json, summary, embedding_json, created_at) "
            "VALUES ('c1', ?, 'Chemical bonding', '[\"chemical bonding\"]', "
            "'How atoms form chemical bonds.', NULL, '2026-07-12T00:00:00+00:00')",
            (self.MATERIAL,),
        )
        for video_id in ("video-a", "video-b", "video-c"):
            self.conn.execute(
                "INSERT INTO videos "
                "(id, title, channel_title, description, duration_sec, created_at) "
                "VALUES (?, 'Chemical bonding explained', 'Teacher', "
                "'A chemistry lesson about chemical bonding.', 600, "
                "'2026-07-12T00:00:00+00:00')",
                (video_id,),
            )
        self.service = ReelService(embedding_service=None, youtube_service=None)
        self.service.learner_progress(self.conn, self.MATERIAL, self.LEARNER)

    def tearDown(self) -> None:
        self.conn.close()

    @staticmethod
    def _selection_item(
        reel_id: str,
        *,
        video_id: str,
        start: float,
        content_score: float,
        difficulty: float = 0.15,
        informativeness: float | None = None,
        topic_relevance: float | None = None,
        educational_importance: float | None = None,
        boundary_confidence: float = 0.90,
        is_standalone: bool = True,
        chain_id: str = "",
        chain_position: int = 0,
        prerequisite_ids: list[str] | None = None,
        intent_role: str = "primary",
        intent_coverage: float = 1.0,
    ) -> dict:
        item = {
            "reel_id": reel_id,
            "concept_id": "c1",
            "video_id": video_id,
            "t_start": start,
            "t_end": start + 20,
            "difficulty": difficulty,
            "score": 99.0,
            "_selection_contract_version": "confidence-gated-v1",
            "_selection_content_score": content_score,
            "_selection_boundary_confidence": boundary_confidence,
            "_selection_is_standalone": is_standalone,
            "_selection_chain_id": chain_id,
            "_selection_chain_position": chain_position,
            "_selection_prerequisite_ids": list(prerequisite_ids or []),
            "_selection_intent_role": intent_role,
            "_selection_intent_coverage": intent_coverage,
        }
        if informativeness is not None:
            item["_selection_informativeness"] = informativeness
        if topic_relevance is not None:
            item["_selection_topic_relevance"] = topic_relevance
        if educational_importance is not None:
            item["_selection_educational_importance"] = educational_importance
        return item

    def test_easier_independent_clip_precedes_higher_quality_advanced_clip(self) -> None:
        items = [
            self._selection_item(
                "early-low", video_id="video-a", start=10, content_score=0.61,
                difficulty=0.10,
            ),
            self._selection_item(
                "late-high", video_id="video-a", start=240, content_score=0.99,
                difficulty=0.80,
            ),
        ]

        ordered = self.service.adaptive_curriculum_order(
            self.conn, self.MATERIAL, self.LEARNER, items,
        )

        self.assertEqual([item["reel_id"] for item in ordered], ["early-low", "late-high"])

    def test_same_stage_ranks_minimum_quality_then_mean_then_relevance(self) -> None:
        items = [
            self._selection_item(
                "weaker-floor", video_id="video-a", start=5,
                content_score=0.99, difficulty=0.05,
                informativeness=0.79, topic_relevance=0.99,
                educational_importance=0.99,
            ),
            self._selection_item(
                "stronger-relevance", video_id="video-b", start=20,
                content_score=0.50, difficulty=0.32,
                informativeness=0.80, topic_relevance=0.95,
                educational_importance=0.85,
            ),
            self._selection_item(
                "weaker-relevance", video_id="video-c", start=10,
                content_score=0.99, difficulty=0.10,
                informativeness=0.80, topic_relevance=0.85,
                educational_importance=0.95,
            ),
            self._selection_item(
                "stronger-mean", video_id="video-a", start=40,
                content_score=0.10, difficulty=0.30,
                informativeness=0.80, topic_relevance=0.99,
                educational_importance=0.99,
            ),
            self._selection_item(
                "stronger-floor", video_id="video-b", start=50,
                content_score=0.10, difficulty=0.33,
                informativeness=0.90, topic_relevance=0.90,
                educational_importance=0.90,
            ),
        ]

        ordered = self.service.adaptive_curriculum_order(
            self.conn, self.MATERIAL, self.LEARNER, items,
        )

        self.assertEqual(
            [item["reel_id"] for item in ordered],
            [
                "stronger-floor",
                "stronger-mean",
                "stronger-relevance",
                "weaker-relevance",
                "weaker-floor",
            ],
        )

    def test_primary_intent_ranks_before_support_only_within_same_difficulty_stage(
        self,
    ) -> None:
        items = [
            self._selection_item(
                "beginner-support",
                video_id="video-a",
                start=10,
                content_score=0.99,
                difficulty=0.2,
                informativeness=0.99,
                topic_relevance=0.99,
                educational_importance=0.99,
                intent_role="supporting",
                intent_coverage=0.5,
            ),
            self._selection_item(
                "beginner-primary",
                video_id="video-b",
                start=20,
                content_score=0.80,
                difficulty=0.3,
                informativeness=0.80,
                topic_relevance=0.80,
                educational_importance=0.80,
                intent_role="primary",
            ),
            self._selection_item(
                "advanced-primary",
                video_id="video-c",
                start=30,
                content_score=1.0,
                difficulty=0.8,
                informativeness=1.0,
                topic_relevance=1.0,
                educational_importance=1.0,
                intent_role="primary",
            ),
        ]

        ordered = self.service.adaptive_curriculum_order(
            self.conn,
            self.MATERIAL,
            self.LEARNER,
            items,
        )

        self.assertEqual(
            [item["reel_id"] for item in ordered],
            ["beginner-primary", "beginner-support", "advanced-primary"],
        )

    def test_persisted_selection_metadata_decodes_intent_role_and_coverage(self) -> None:
        metadata = self.service._selection_metadata({
            "selection_contract_version": "quality_silence_v32",
            "intent_role": "supporting",
            "intent_coverage": 0.5,
        })

        self.assertEqual(metadata["_selection_intent_role"], "supporting")
        self.assertEqual(metadata["_selection_intent_coverage"], 0.5)

    def test_explicit_chain_and_prerequisite_edges_remain_ordered(self) -> None:
        items = [
            self._selection_item(
                "chain-2", video_id="video-a", start=100, content_score=0.99,
                is_standalone=False, chain_id="derivation", chain_position=2,
            ),
            self._selection_item(
                "independent", video_id="video-b", start=20, content_score=0.80,
                prerequisite_ids=["chain-2"],
            ),
            self._selection_item(
                "chain-1", video_id="video-a", start=30, content_score=0.70,
                chain_id="derivation", chain_position=1,
            ),
        ]

        ordered = self.service.adaptive_curriculum_order(
            self.conn, self.MATERIAL, self.LEARNER, items,
        )

        self.assertEqual(
            [item["reel_id"] for item in ordered],
            ["chain-1", "chain-2", "independent"],
        )

    def test_first_clip_gate_skips_unsafe_higher_priority_candidate(self) -> None:
        items = [
            self._selection_item(
                "unsafe", video_id="video-a", start=10, content_score=1.0,
                boundary_confidence=0.79,
            ),
            self._selection_item(
                "safe", video_id="video-b", start=10, content_score=0.70,
                boundary_confidence=0.80,
            ),
        ]

        ordered = self.service.adaptive_curriculum_order(
            self.conn, self.MATERIAL, self.LEARNER, items,
        )

        self.assertEqual([item["reel_id"] for item in ordered], ["safe", "unsafe"])

    def test_source_diversity_does_not_override_stronger_content(self) -> None:
        items = [
            self._selection_item(
                "first", video_id="video-a", start=10, content_score=1.0,
            ),
            self._selection_item(
                "same-video", video_id="video-a", start=40, content_score=0.92,
            ),
            self._selection_item(
                "other-video", video_id="video-b", start=10, content_score=0.86,
            ),
        ]

        ordered = self.service.adaptive_curriculum_order(
            self.conn, self.MATERIAL, self.LEARNER, items,
        )

        self.assertEqual(
            [item["reel_id"] for item in ordered],
            ["first", "same-video", "other-video"],
        )

    def test_source_diversity_breaks_an_exact_priority_tie(self) -> None:
        items = [
            self._selection_item(
                "first", video_id="video-a", start=10, content_score=1.0,
            ),
            self._selection_item(
                "same-video", video_id="video-a", start=40, content_score=0.86,
            ),
            self._selection_item(
                "other-video", video_id="video-b", start=10, content_score=0.86,
            ),
        ]

        ordered = self.service.adaptive_curriculum_order(
            self.conn, self.MATERIAL, self.LEARNER, items,
        )

        self.assertEqual(
            [item["reel_id"] for item in ordered],
            ["first", "other-video", "same-video"],
        )

    def test_mixed_legacy_rows_keep_source_chronology_without_disabling_new_order(self) -> None:
        items = [
            {
                "reel_id": "legacy-late",
                "concept_id": "c1",
                "video_id": "video-b",
                "t_start": 100.0,
                "difficulty": 0.15,
                "score": 0.95,
            },
            self._selection_item(
                "new-best", video_id="video-a", start=240, content_score=1.0,
            ),
            {
                "reel_id": "legacy-early",
                "concept_id": "c1",
                "video_id": "video-b",
                "t_start": 10.0,
                "difficulty": 0.15,
                "score": 0.90,
            },
        ]

        ordered = self.service.adaptive_curriculum_order(
            self.conn, self.MATERIAL, self.LEARNER, items,
        )

        self.assertEqual(
            [item["reel_id"] for item in ordered],
            ["new-best", "legacy-early", "legacy-late"],
        )

    def test_final_request_order_does_not_reapply_legacy_scheduler(self) -> None:
        from backend.app import main as main_module

        rows = [
            {"reel_id": "late-best", "_selection_ordered": True},
            {"reel_id": "early-low", "_selection_ordered": True},
        ]
        with mock.patch.object(
            main_module.reel_service, "adaptive_curriculum_order"
        ) as legacy_scheduler:
            ordered = main_module._finalize_request_reel_order(
                self.conn,
                material_id=self.MATERIAL,
                learner_id=self.LEARNER,
                rows=rows,
                previous_video_id="",
            )

        legacy_scheduler.assert_not_called()
        self.assertEqual(
            [item["reel_id"] for item in ordered],
            ["late-best", "early-low"],
        )
        self.assertTrue(all("_selection_ordered" not in item for item in ordered))

    def test_request_shaping_preserves_versioned_topological_order(self) -> None:
        from backend.app import main as main_module

        rows = [
            {
                "reel_id": "prerequisite",
                "_selection_ordered": True,
                "score": 0.70,
                "video_title": "Setup",
            },
            {
                "reel_id": "dependent",
                "_selection_ordered": True,
                "score": 0.99,
                "video_title": "Application",
            },
        ]

        shaped = main_module._shape_reels_for_request_context(
            rows,
            page=1,
            limit=10,
            subject_tag="chemical bonding",
            strict_topic_only=True,
        )

        self.assertEqual(
            [item["reel_id"] for item in shaped],
            ["prerequisite", "dependent"],
        )

    def _insert_versioned_reel(
        self,
        *,
        reel_id: str,
        video_id: str,
        start: float,
        difficulty: float,
        base_score: float,
        educational_importance: float = 0.80,
    ) -> None:
        search_context = {
            "selection_contract_version": "confidence-gated-v1",
            "directly_teaches_topic": True,
            "substantive": True,
            "factually_grounded": True,
            "topic_relevance": 0.80,
            "educational_importance": educational_importance,
            "informativeness": 0.80,
            "boundary_confidence": 0.90,
            "is_standalone": True,
            "chain_id": "",
            "chain_position": 0,
            "prerequisite_ids": [],
        }
        self.conn.execute(
            "INSERT INTO reels "
            "(id, generation_id, material_id, concept_id, video_id, video_url, "
            "t_start, t_end, transcript_snippet, takeaways_json, base_score, difficulty, "
            "informativeness, search_context_json, created_at) "
            "VALUES (?, 'selection-generation', ?, 'c1', ?, ?, ?, ?, ?, '[]', ?, ?, "
            "0.8, ?, '2026-07-12T00:00:00+00:00')",
            (
                reel_id,
                self.MATERIAL,
                video_id,
                f"https://youtube.test/{video_id}",
                start,
                start + 20,
                "Chemical bonding explains how atoms share or transfer electrons.",
                base_score,
                difficulty,
                json.dumps(search_context),
            ),
        )

    @staticmethod
    def _relevance(*_args, **_kwargs) -> dict:
        return {
            "score": 0.8,
            "concept_overlap": 1.0,
            "context_overlap": 1.0,
            "matched_terms": ["chemical bonding"],
            "off_topic_penalty": 0.0,
            "reason": "matched topic",
        }

    def _ranked(self, *, require_verified_boundaries: bool = False) -> list[dict]:
        with mock.patch.object(
            self.service, "_score_text_relevance", side_effect=self._relevance,
        ), mock.patch.object(self.service, "_build_caption_cues", return_value=[]):
            return self.service.ranked_feed(
                self.conn,
                material_id=self.MATERIAL,
                generation_id="selection-generation",
                fast_mode=True,
                learner_id=self.LEARNER,
                require_verified_boundaries=require_verified_boundaries,
            )

    def test_v7_ranked_feed_requires_each_quality_score_at_threshold(self) -> None:
        self._insert_versioned_reel(
            reel_id="v6-quality",
            video_id="video-a",
            start=10,
            difficulty=0.15,
            base_score=0.8,
        )
        row = self.conn.execute(
            "SELECT search_context_json FROM reels WHERE id = 'v6-quality'"
        ).fetchone()
        baseline = json.loads(row[0])
        baseline.update({
            "selection_contract_version": "quality_silence_v12",
            "self_contained": True,
            "is_standalone": True,
            "topic_evidence_quote": (
                "Chemical bonding explains how atoms share or transfer electrons"
            ),
            "surface_eligible": True,
            "boundary_status": "verified",
            "boundary_diagnostics": {
                "acoustic_verified": True,
                "acoustic": {"threshold_dbfs": -38.0},
            },
            "speech_corridor_verified": True,
        })

        for field in (
            "informativeness",
            "topic_relevance",
            "educational_importance",
        ):
            with self.subTest(field=field):
                context = {**baseline, field: 0.74}
                self.conn.execute(
                    "UPDATE reels SET search_context_json = ? WHERE id = 'v6-quality'",
                    (json.dumps(context),),
                )
                self.conn.execute("DELETE FROM ranked_feed_cache")
                self.assertEqual(self._ranked(require_verified_boundaries=True), [])

        accepted = {
            **baseline,
            "informativeness": 0.75,
            "topic_relevance": 0.75,
            "educational_importance": 0.75,
        }
        self.conn.execute(
            "UPDATE reels SET search_context_json = ? WHERE id = 'v6-quality'",
            (json.dumps(accepted),),
        )
        self.conn.execute("DELETE FROM ranked_feed_cache")
        self.assertEqual(
            [item["reel_id"] for item in self._ranked(require_verified_boundaries=True)],
            ["v6-quality"],
        )

    def test_ranked_feed_order_is_difficulty_first_and_stable(self) -> None:
        self._insert_versioned_reel(
            reel_id="easy", video_id="video-a", start=10,
            difficulty=0.15, base_score=0.01,
        )
        self._insert_versioned_reel(
            reel_id="hard", video_id="video-b", start=10,
            difficulty=0.85, base_score=100.0,
        )

        beginner = self._ranked()
        self.assertEqual([item["reel_id"] for item in beginner], ["easy", "hard"])
        self.assertAlmostEqual(beginner[0]["score"], 0.80)

        self.service.set_learner_level(
            self.conn, self.MATERIAL, self.LEARNER, "advanced",
        )
        advanced = self._ranked()
        self.assertEqual([item["reel_id"] for item in advanced], ["hard", "easy"])
        self.assertAlmostEqual(advanced[0]["score"], 0.80)

    def test_v4_reservoir_orders_all_valid_clips_for_current_learner_level(self) -> None:
        for reel_id, video_id, difficulty in (
            ("easy-v3", "video-a", 0.10),
            ("hard-v3", "video-b", 0.85),
        ):
            self._insert_versioned_reel(
                reel_id=reel_id,
                video_id=video_id,
                start=10,
                difficulty=difficulty,
                base_score=0.8,
            )
            row = self.conn.execute(
                "SELECT search_context_json FROM reels WHERE id = ?",
                (reel_id,),
            ).fetchone()
            context = json.loads(row[0])
            context.update({
                "selection_contract_version": "quality_silence_v4",
                "self_contained": True,
                "topic_evidence_quote": (
                    "Chemical bonding explains how atoms share or transfer electrons"
                ),
                "surface_eligible": True,
                "deferred_level": False,
            })
            self.conn.execute(
                "UPDATE reels SET search_context_json = ? WHERE id = ?",
                (json.dumps(context), reel_id),
            )

        self.assertEqual(
            [item["reel_id"] for item in self._ranked()],
            ["easy-v3", "hard-v3"],
        )
        self.service.set_learner_level(
            self.conn, self.MATERIAL, self.LEARNER, "advanced",
        )
        self.assertEqual(
            [item["reel_id"] for item in self._ranked()],
            ["hard-v3", "easy-v3"],
        )

    def test_v13_reservoir_orders_every_valid_level_by_nearest_first(self) -> None:
        def insert_current(
            reel_id: str,
            video_id: str,
            difficulty: float,
            *,
            surface_eligible: bool,
        ) -> None:
            self._insert_versioned_reel(
                reel_id=reel_id,
                video_id=video_id,
                start=10,
                difficulty=difficulty,
                base_score=0.8,
            )
            row = self.conn.execute(
                "SELECT search_context_json FROM reels WHERE id = ?",
                (reel_id,),
            ).fetchone()
            context = json.loads(row[0])
            context.update({
                "selection_contract_version": "quality_silence_v32",
                "self_contained": True,
                "topic_evidence_quote": (
                    "Chemical bonding explains how atoms share or transfer electrons"
                ),
                "surface_eligible": surface_eligible,
                "surface_reason": "" if surface_eligible else "level_mismatch",
                "deferred_level": not surface_eligible,
                "boundary_status": "verified",
                "boundary_diagnostics": {
                    "acoustic_verified": True,
                    "final_range": [10.0, 30.0],
                    "acoustic": {
                        "threshold_dbfs": -38.0,
                        "start_quiet": [9.9, 10.1],
                        "end_quiet": [29.9, 30.1],
                    },
                },
                "speech_corridor_verified": True,
            })
            self.conn.execute(
                "UPDATE reels SET search_context_json = ? WHERE id = ?",
                (json.dumps(context), reel_id),
            )

        insert_current("intermediate", "video-a", 0.50, surface_eligible=False)
        insert_current("advanced", "video-b", 0.85, surface_eligible=False)

        self.assertEqual(
            [item["reel_id"] for item in self._ranked(require_verified_boundaries=True)],
            ["intermediate", "advanced"],
        )

        insert_current("beginner", "video-c", 0.15, surface_eligible=True)
        self.conn.execute("DELETE FROM ranked_feed_cache")
        self.assertEqual(
            [item["reel_id"] for item in self._ranked(require_verified_boundaries=True)],
            ["beginner", "intermediate", "advanced"],
        )

        self.service.set_learner_level(
            self.conn, self.MATERIAL, self.LEARNER, "advanced",
        )
        self.assertEqual(
            [item["reel_id"] for item in self._ranked(require_verified_boundaries=True)],
            ["advanced", "intermediate", "beginner"],
        )

    def test_nearest_level_order_is_stable_and_never_drops_valid_inventory(self) -> None:
        def item(reel_id: str, difficulty: float) -> dict:
            return {
                "reel_id": reel_id,
                "difficulty": difficulty,
                "selection_contract_version": "quality_silence_v32",
            }

        easy = item("easy", 0.15)
        intermediate = item("intermediate", 0.50)
        advanced = item("advanced", 0.85)
        cases = (
            ("beginner", [intermediate, advanced], ["intermediate", "advanced"]),
            ("intermediate", [easy, advanced], ["easy", "advanced"]),
            ("advanced", [easy, intermediate], ["intermediate", "easy"]),
        )
        for level, inventory, expected in cases:
            with self.subTest(level=level):
                self.assertEqual(
                    [
                        row["reel_id"]
                        for row in self.service.select_difficulty_inventory(
                            inventory, level
                        )
                    ],
                    expected,
                )

    def test_same_nearest_level_uses_quality_source_and_timestamp_tiebreaks(self) -> None:
        def item(
            reel_id: str,
            *,
            quality: float,
            source_rank: int,
            start: float,
        ) -> dict:
            return {
                "reel_id": reel_id,
                "difficulty": 0.50,
                "selection_contract_version": "quality_silence_v32",
                "_selection_informativeness": quality,
                "_selection_topic_relevance": quality,
                "_selection_educational_importance": quality,
                "_selection_source_rank": source_rank,
                "t_start": start,
            }

        ordered = self.service.select_difficulty_inventory(
            [
                item("later", quality=0.90, source_rank=1, start=40),
                item("earlier", quality=0.90, source_rank=1, start=10),
                item("better-source", quality=0.90, source_rank=0, start=80),
                item("best-quality", quality=0.95, source_rank=2, start=90),
            ],
            "intermediate",
        )

        self.assertEqual(
            [item["reel_id"] for item in ordered],
            ["best-quality", "better-source", "earlier", "later"],
        )

    def test_historical_inventory_remains_viewable(self) -> None:
        for version in (
            "quality_silence_v3",
            "quality_silence_v9",
            "quality_silence_v19",
            "quality_silence_v22",
            "quality_silence_v23",
            "quality_silence_v24",
            "quality_silence_v25",
            "quality_silence_v26",
        ):
            reel_id = f"historical-{version}"
            with self.subTest(version=version):
                self._insert_versioned_reel(
                    reel_id=reel_id,
                    video_id="video-a",
                    start=10,
                    difficulty=0.10,
                    base_score=0.8,
                )
                row = self.conn.execute(
                    "SELECT search_context_json FROM reels WHERE id = ?",
                    (reel_id,),
                ).fetchone()
                context = json.loads(row[0])
                context.update({
                    "selection_contract_version": version,
                    "directly_teaches_topic": True,
                    "substantive": True,
                    "factually_grounded": True,
                    "self_contained": True,
                    "topic_evidence_quote": (
                        "Chemical bonding explains how atoms share or transfer electrons"
                    ),
                    "surface_eligible": True,
                    "deferred_level": False,
                })
                self.conn.execute(
                    "UPDATE reels SET search_context_json = ? WHERE id = ?",
                    (json.dumps(context), reel_id),
                )

                result = self._ranked()

                self.assertEqual([item["reel_id"] for item in result], [reel_id])
                self.assertEqual(result[0]["selection_contract_version"], version)
                self.conn.execute("DELETE FROM ranked_feed_cache")
                self.conn.execute("DELETE FROM reels WHERE id = ?", (reel_id,))

    def test_v4_response_distinguishes_selector_from_deterministic_relevance(self) -> None:
        self._insert_versioned_reel(
            reel_id="live-shaped-v3",
            video_id="video-a",
            start=10,
            difficulty=0.15,
            base_score=0.8,
        )
        row = self.conn.execute(
            "SELECT search_context_json FROM reels WHERE id = 'live-shaped-v3'"
        ).fetchone()
        context = json.loads(row[0])
        context.update({
            "selection_contract_version": "quality_silence_v4",
            "topic_relevance": 0.93,
            "self_contained": True,
            "topic_evidence_quote": (
                "Chemical bonding explains how atoms share or transfer electrons"
            ),
            "surface_eligible": True,
            "deferred_level": False,
        })
        self.conn.execute(
            "UPDATE reels SET search_context_json = ? WHERE id = 'live-shaped-v3'",
            (json.dumps(context),),
        )
        deterministic_relevance = {
            "score": 0.13,
            "concept_overlap": 0.13,
            "context_overlap": 0.13,
            "matched_terms": ["bonding"],
            "off_topic_penalty": 0.0,
            "reason": "deterministic overlap",
        }

        with mock.patch.object(
            self.service,
            "_score_text_relevance",
            return_value=deterministic_relevance,
        ), mock.patch.object(self.service, "_build_caption_cues", return_value=[]):
            fresh = self.service.ranked_feed(
                self.conn,
                material_id=self.MATERIAL,
                generation_id="selection-generation",
                fast_mode=True,
                learner_id=self.LEARNER,
            )
        cached = self.service.ranked_feed(
            self.conn,
            material_id=self.MATERIAL,
            generation_id="selection-generation",
            fast_mode=True,
            learner_id=self.LEARNER,
        )

        for result in (fresh, cached):
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["video_id"], "video-a")
            self.assertEqual(
                result[0]["selection_contract_version"], "quality_silence_v4"
            )
            self.assertAlmostEqual(result[0]["relevance_score"], 0.13)
            self.assertAlmostEqual(result[0]["topic_relevance"], 0.93)

    def test_v13_response_exposes_authoritative_selector_relevance(self) -> None:
        self._insert_versioned_reel(
            reel_id="live-shaped-v13",
            video_id="video-a",
            start=10,
            difficulty=0.15,
            base_score=0.8,
        )
        row = self.conn.execute(
            "SELECT search_context_json FROM reels WHERE id = 'live-shaped-v13'"
        ).fetchone()
        context = json.loads(row[0])
        context.update({
            "selection_contract_version": "quality_silence_v32",
            "topic_relevance": 0.93,
            "self_contained": True,
            "topic_evidence_quote": (
                "Chemical bonding explains how atoms share or transfer electrons"
            ),
            "surface_eligible": True,
            "deferred_level": False,
        })
        self.conn.execute(
            "UPDATE reels SET search_context_json = ? WHERE id = 'live-shaped-v13'",
            (json.dumps(context),),
        )
        deterministic_relevance = {
            "score": 0.13,
            "concept_overlap": 0.13,
            "context_overlap": 0.13,
            "matched_terms": ["bonding"],
            "off_topic_penalty": 0.0,
            "reason": "deterministic overlap",
        }

        with mock.patch.object(
            self.service,
            "_score_text_relevance",
            return_value=deterministic_relevance,
        ), mock.patch.object(self.service, "_build_caption_cues", return_value=[]):
            fresh = self.service.ranked_feed(
                self.conn,
                material_id=self.MATERIAL,
                generation_id="selection-generation",
                fast_mode=True,
                learner_id=self.LEARNER,
            )
        cached = self.service.ranked_feed(
            self.conn,
            material_id=self.MATERIAL,
            generation_id="selection-generation",
            fast_mode=True,
            learner_id=self.LEARNER,
        )

        for result in (fresh, cached):
            self.assertEqual(len(result), 1)
            self.assertEqual(
                result[0]["selection_contract_version"], "quality_silence_v32"
            )
            self.assertAlmostEqual(result[0]["relevance_score"], 0.93)
            self.assertAlmostEqual(result[0]["topic_relevance"], 0.93)

    def test_ranked_feed_uses_selector_cue_snapshot_after_artifact_refresh(self) -> None:
        response_video_id = "yt:dQw4w9WgXcQ"
        bare_video_id = "dQw4w9WgXcQ"
        correct_artifact_key = transcript_artifact_key(
            video_id=bare_video_id,
            provider="supadata",
            requested_language="en",
            returned_language="en",
            native_mode=False,
        )
        self.conn.execute(
            "INSERT INTO videos "
            "(id, title, channel_title, description, duration_sec, created_at) "
            "VALUES (?, 'Chemical bonding explained', 'Teacher', "
            "'A chemistry lesson about chemical bonding.', 600, "
            "'2026-07-12T00:00:00+00:00')",
            (response_video_id,),
        )
        self._insert_versioned_reel(
            reel_id="artifact-bound",
            video_id=response_video_id,
            start=10,
            difficulty=0.15,
            base_score=0.8,
        )
        row = self.conn.execute(
            "SELECT search_context_json FROM reels WHERE id = 'artifact-bound'"
        ).fetchone()
        context = json.loads(row[0])
        context.update(
            {
                "selection_contract_version": "quality_silence_v5",
                "self_contained": True,
                "topic_evidence_quote": (
                    "Chemical bonding explains how atoms share or transfer electrons"
                ),
                "surface_eligible": True,
                "deferred_level": False,
                "boundary_status": "verified",
                "boundary_diagnostics": {
                    "acoustic_verified": True,
                    "acoustic": {"threshold_dbfs": -38.0},
                },
                "speech_corridor_verified": True,
                "transcript_artifact_key": correct_artifact_key,
                "selection_caption_cues": [
                    {
                        "cue_id": "correct",
                        "start": 10.0,
                        "end": 30.0,
                        "text": (
                            "Chemical bonding explains how atoms share or transfer "
                            "electrons in the exact selector snapshot."
                        ),
                        "lang": "en",
                    }
                ],
            }
        )
        self.conn.execute(
            "UPDATE reels SET selected_cue_ids_json = '[\"correct\"]', "
            "search_context_json = ? WHERE id = 'artifact-bound'",
            (json.dumps(context),),
        )

        def insert_artifact(
            artifact_key: str,
            cue_id: str,
            text: str,
            created_at: str,
            *,
            native_mode: bool,
        ) -> None:
            payload = {
                "artifact_key": artifact_key,
                "video_id": bare_video_id,
                "provider": "supadata",
                "requested_language": "en",
                "returned_language": "en",
                "native_mode": native_mode,
                "schema_version": TRANSCRIPT_SCHEMA_VERSION,
                "segments": [
                    {
                        "cue_id": cue_id,
                        "start": 10.0,
                        "end": 30.0,
                        "text": text,
                        "lang": "en",
                    }
                ],
                "duration_sec": 30.0,
                "created_at": created_at,
            }
            self.conn.execute(
                "INSERT INTO transcript_artifacts "
                "(cache_key, video_id, provider, requested_language, "
                "returned_language, native_mode, schema_version, artifact_json, "
                "duration_sec, cue_count, created_at, expires_at) "
                "VALUES (?, ?, 'supadata', 'en', 'en', ?, ?, ?, 30, 1, ?, "
                "'2027-07-12T00:00:00+00:00')",
                (
                    artifact_key,
                    bare_video_id,
                    1 if native_mode else 0,
                    str(TRANSCRIPT_SCHEMA_VERSION),
                    json.dumps(payload),
                    created_at,
                ),
            )

        correct_text = (
            "Chemical bonding explains how atoms share or transfer electrons "
            "in the exact selector snapshot."
        )
        insert_artifact(
            correct_artifact_key,
            "wrong",
            "A same-profile refresh replaced the cached transcript content.",
            "2026-07-12T01:00:00+00:00",
            native_mode=False,
        )

        with mock.patch.object(
            self.service, "_score_text_relevance", side_effect=self._relevance,
        ):
            result = self.service.ranked_feed(
                self.conn,
                material_id=self.MATERIAL,
                generation_id="selection-generation",
                fast_mode=True,
                learner_id=self.LEARNER,
            )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["captions"][0]["text"], correct_text)

        context.pop("selection_caption_cues")
        context.pop("transcript_artifact_key")
        self.conn.execute(
            "UPDATE reels SET search_context_json = ? WHERE id = 'artifact-bound'",
            (json.dumps(context),),
        )
        self.conn.execute("DELETE FROM ranked_feed_cache")
        with mock.patch.object(
            self.service, "_score_text_relevance", side_effect=self._relevance,
        ):
            missing_snapshot = self.service.ranked_feed(
                self.conn,
                material_id=self.MATERIAL,
                generation_id="selection-generation",
                fast_mode=True,
                learner_id=self.LEARNER,
            )
        self.assertEqual(missing_snapshot[0]["captions"], [])

    def test_v4_deferred_boundary_clips_all_release_in_nearest_level_order(self) -> None:
        cases = (
            ("bin-33", "video-a", 10, 0.33),
            ("bin-34", "video-b", 20, 0.34),
            ("bin-66", "video-c", 30, 0.66),
            ("bin-67", "video-a", 40, 0.67),
        )
        for reel_id, video_id, start, difficulty in cases:
            self._insert_versioned_reel(
                reel_id=reel_id,
                video_id=video_id,
                start=start,
                difficulty=difficulty,
                base_score=0.8,
            )
            row = self.conn.execute(
                "SELECT search_context_json FROM reels WHERE id = ?",
                (reel_id,),
            ).fetchone()
            context = json.loads(row[0])
            context.update({
                "selection_contract_version": "quality_silence_v4",
                "self_contained": True,
                "topic_evidence_quote": (
                    "Chemical bonding explains how atoms share or transfer electrons"
                ),
                "surface_eligible": False,
                "surface_reason": "level_mismatch",
                "deferred_level": True,
                "boundary_status": "verified",
                "boundary_diagnostics": {
                    "acoustic_verified": True,
                    "acoustic": {"threshold_dbfs": -38.0},
                },
            })
            self.conn.execute(
                "UPDATE reels SET search_context_json = ? WHERE id = ?",
                (json.dumps(context), reel_id),
            )

        expected_by_level = {
            "beginner": ["bin-33", "bin-34", "bin-66", "bin-67"],
            "intermediate": ["bin-34", "bin-66", "bin-33", "bin-67"],
            "advanced": ["bin-67", "bin-34", "bin-66", "bin-33"],
        }
        for level, expected in expected_by_level.items():
            self.service.set_learner_level(
                self.conn, self.MATERIAL, self.LEARNER, level,
            )
            self.assertEqual(
                [
                    item["reel_id"]
                    for item in self._ranked(require_verified_boundaries=True)
                ],
                expected,
            )

    def test_ranked_feed_excludes_unversioned_acoustic_unavailable_row(self) -> None:
        self.conn.execute(
            "INSERT INTO reels "
            "(id, generation_id, material_id, concept_id, video_id, video_url, "
            "t_start, t_end, transcript_snippet, takeaways_json, base_score, difficulty, "
            "informativeness, search_context_json, created_at) "
            "VALUES ('deferred', 'selection-generation', ?, 'c1', 'video-a', "
            "'https://youtube.test/video-a', 10, 30, "
            "'Chemical bonding explains electron sharing.', '[]', 1, 0.2, 1, ?, "
            "'2026-07-12T00:00:00+00:00')",
            (
                self.MATERIAL,
                json.dumps({
                    "selection_candidate_id": "video-a::legacy",
                    "surface_eligible": False,
                    "boundary_status": "unavailable",
                }),
            ),
        )

        self.assertEqual(self._ranked(), [])

    def test_production_feed_accepts_strict_or_valid_transcript_boundaries(self) -> None:
        states = [
            ("missing", "", {"acoustic_verified": None}),
            ("caption", "caption_aligned", {"acoustic_verified": False}),
            ("unverified", "verified", {"acoustic_verified": False}),
            ("legacy-verified", "verified", {"acoustic_verified": True}),
            (
                "adaptive-verified",
                "verified",
                {
                    "acoustic_verified": True,
                    "acoustic": {
                        "adaptive_quiet": True,
                        "start_threshold_dbfs": -24.0,
                        "end_threshold_dbfs": -38.0,
                    },
                },
            ),
            (
                "strict-verified",
                "verified",
                {
                    "acoustic_verified": True,
                    "acoustic": {
                        "threshold_dbfs": -38.0,
                        "start_threshold_dbfs": -38.0,
                        "end_threshold_dbfs": -38.0,
                    },
                },
            ),
        ]
        for index, (reel_id, boundary_status, boundary_diagnostics) in enumerate(states):
            self._insert_versioned_reel(
                reel_id=reel_id,
                video_id="video-a",
                start=10 + index * 25,
                difficulty=0.15,
                base_score=0.8,
            )
            row = self.conn.execute(
                "SELECT search_context_json FROM reels WHERE id = ?",
                (reel_id,),
            ).fetchone()
            context = json.loads(row[0])
            context.update({
                "surface_eligible": True,
                "boundary_status": boundary_status,
                "boundary_diagnostics": boundary_diagnostics,
            })
            self.conn.execute(
                "UPDATE reels SET search_context_json = ? WHERE id = ?",
                (json.dumps(context), reel_id),
            )

        for reel_id, start, captions in (
            (
                "transcript-aligned",
                170.0,
                [{
                    "cue_id": "cue-1",
                    "start": 170.0,
                    "end": 190.0,
                    "text": "Chemical bonding explains electron sharing.",
                }],
            ),
            ("malformed-transcript", 195.0, [{}]),
        ):
            self._insert_versioned_reel(
                reel_id=reel_id,
                video_id="video-a",
                start=start,
                difficulty=0.15,
                base_score=0.8,
            )
            row = self.conn.execute(
                "SELECT search_context_json FROM reels WHERE id = ?",
                (reel_id,),
            ).fetchone()
            context = json.loads(row[0])
            context.update({
                "selection_contract_version": "quality_silence_v32",
                "self_contained": True,
                "topic_evidence_quote": (
                    "Chemical bonding explains how atoms share or transfer electrons"
                ),
                "surface_eligible": True,
                "speech_corridor_verified": True,
                "boundary_status": "context_aligned",
                "selection_caption_cues": captions,
                "boundary_diagnostics": {
                    "method": "transcript_context",
                    "context_aligned": True,
                    "acoustic_verified": False,
                    "transcript": {
                        "context_aligned": True,
                        "stage": "transcript",
                        "reason": "complete_discourse_boundary",
                        "required_speech_range": [start, start + 20.0],
                        "semantic_range": [start, start + 20.0],
                        "final_range": [start, start + 20.0],
                    },
                },
            })
            self.conn.execute(
                "UPDATE reels SET search_context_json = ? WHERE id = ?",
                (json.dumps(context), reel_id),
            )

        ranked = self._ranked(require_verified_boundaries=True)

        self.assertEqual(
            {item["reel_id"] for item in ranked},
            {"strict-verified", "transcript-aligned"},
        )

    def test_level_mismatched_verified_clip_surfaces_without_level_rejection(self) -> None:
        self._insert_versioned_reel(
            reel_id="advanced-reservoir",
            video_id="video-a",
            start=10,
            difficulty=0.85,
            base_score=0.9,
        )
        row = self.conn.execute(
            "SELECT search_context_json FROM reels WHERE id = 'advanced-reservoir'"
        ).fetchone()
        context = json.loads(row[0])
        context.update({
            "selection_contract_version": "quality_silence_v32",
            "self_contained": True,
            "topic_evidence_quote": (
                "Chemical bonding explains how atoms share or transfer electrons"
            ),
            "surface_eligible": False,
            "surface_reason": "level_mismatch",
            "deferred_level": True,
            "speech_corridor_verified": True,
            "boundary_status": "verified",
            "boundary_diagnostics": {
                "acoustic_verified": True,
                "final_range": [10.0, 30.0],
                "acoustic": {
                    "threshold_dbfs": -38.0,
                    "start_quiet": [9.9, 10.1],
                    "end_quiet": [29.9, 30.1],
                },
            },
        })
        self.conn.execute(
            "UPDATE reels SET search_context_json = ? WHERE id = 'advanced-reservoir'",
            (json.dumps(context),),
        )

        self.assertEqual(
            [
                item["reel_id"]
                for item in self._ranked(require_verified_boundaries=True)
            ],
            ["advanced-reservoir"],
        )

        self.service.set_learner_level(
            self.conn, self.MATERIAL, self.LEARNER, "advanced",
        )
        self.assertEqual(
            [
                item["reel_id"]
                for item in self._ranked(require_verified_boundaries=True)
            ],
            ["advanced-reservoir"],
        )

    def test_dependent_waits_until_its_deferred_prerequisite_is_ready(self) -> None:
        self._insert_versioned_reel(
            reel_id="deferred-parent",
            video_id="video-a",
            start=10,
            difficulty=0.10,
            base_score=0.8,
        )
        self._insert_versioned_reel(
            reel_id="advanced-dependent",
            video_id="video-a",
            start=40,
            difficulty=0.85,
            base_score=0.9,
        )
        for reel_id, updates in (
            (
                "deferred-parent",
                {
                    "selection_candidate_id": "video-a::parent",
                    "prerequisite_ids": [],
                    "is_standalone": True,
                    "surface_eligible": False,
                    "surface_reason": "level_mismatch",
                    "deferred_level": True,
                },
            ),
            (
                "advanced-dependent",
                {
                    "selection_candidate_id": "video-a::dependent",
                    "prerequisite_ids": ["video-a::parent"],
                    "is_standalone": False,
                    "surface_eligible": False,
                    "surface_reason": "prerequisite_not_surfaceable",
                    "deferred_level": False,
                },
            ),
        ):
            row = self.conn.execute(
                "SELECT search_context_json FROM reels WHERE id = ?",
                (reel_id,),
            ).fetchone()
            context = json.loads(row[0])
            context.update({
                **updates,
                "boundary_status": "verified",
                "boundary_diagnostics": {
                    "acoustic_verified": True,
                    "acoustic": {"threshold_dbfs": -38.0},
                },
            })
            self.conn.execute(
                "UPDATE reels SET search_context_json = ? WHERE id = ?",
                (json.dumps(context), reel_id),
            )

        self.service.set_learner_level(
            self.conn, self.MATERIAL, self.LEARNER, "advanced",
        )
        self.assertEqual(
            self._ranked(require_verified_boundaries=True),
            [],
        )

        self.conn.execute(
            "UPDATE reels SET difficulty = 0.85 WHERE id = 'deferred-parent'"
        )
        self.assertEqual(
            [
                item["reel_id"]
                for item in self._ranked(require_verified_boundaries=True)
            ],
            ["deferred-parent", "advanced-dependent"],
        )


if __name__ == "__main__":
    unittest.main()
