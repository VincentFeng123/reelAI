"""Importance-first ordering for versioned clipping selections."""
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
        boundary_confidence: float = 0.90,
        is_standalone: bool = True,
        chain_id: str = "",
        chain_position: int = 0,
        prerequisite_ids: list[str] | None = None,
    ) -> dict:
        return {
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
        }

    def test_later_more_important_independent_clip_beats_video_intro(self) -> None:
        items = [
            self._selection_item(
                "early-low", video_id="video-a", start=10, content_score=0.61,
            ),
            self._selection_item(
                "late-high", video_id="video-a", start=240, content_score=0.99,
            ),
        ]

        ordered = self.service.adaptive_curriculum_order(
            self.conn, self.MATERIAL, self.LEARNER, items,
        )

        self.assertEqual([item["reel_id"] for item in ordered], ["late-high", "early-low"])

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

    def test_same_video_penalty_is_soft_and_exactly_point_zero_eight(self) -> None:
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

    def _ranked(self) -> list[dict]:
        with mock.patch.object(
            self.service, "_score_text_relevance", side_effect=self._relevance,
        ), mock.patch.object(self.service, "_build_caption_cues", return_value=[]):
            return self.service.ranked_feed(
                self.conn,
                material_id=self.MATERIAL,
                generation_id="selection-generation",
                fast_mode=True,
                learner_id=self.LEARNER,
            )

    def test_ranked_feed_uses_level_neutral_content_and_current_level(self) -> None:
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
        self.assertAlmostEqual(beginner[0]["score"], 0.86)

        self.service.set_learner_level(
            self.conn, self.MATERIAL, self.LEARNER, "advanced",
        )
        advanced = self._ranked()
        self.assertEqual([item["reel_id"] for item in advanced], ["hard", "easy"])
        self.assertAlmostEqual(advanced[0]["score"], 0.86)


if __name__ == "__main__":
    unittest.main()
