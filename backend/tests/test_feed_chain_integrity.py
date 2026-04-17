"""
Chain-integrity tests for the feed layer.

Two integrity properties are verified:

1.  ``ranked_feed`` groups reels from the same YouTube video together.
    Without grouping, pure score-based sorting interleaves reels from
    different videos, so a multi-reel learning arc from one clip gets
    fragmented by reels from other clips. Users reported this as
    "videos from the same clip are interrupted by other videos."

2.  Within each same-video group, reels are ordered by ``t_start`` so
    the clip's narrative plays in chronological order (part 1, part 2,
    part 3, ...). This matters most for multi-part clusters where each
    reel continues where the previous left off.
"""

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


class FeedChainIntegrityTests(unittest.TestCase):
    MATERIAL_ID = "material-chain"
    CONCEPT_ID = "concept-chain"

    def _fresh_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA)
        conn.execute(
            "INSERT INTO materials (id, subject_tag, raw_text, source_type, source_path, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (self.MATERIAL_ID, "math", "Calculus notes", "text", None, "2026-04-16T00:00:00+00:00"),
        )
        conn.execute(
            "INSERT INTO concepts (id, material_id, title, keywords_json, summary, embedding_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                self.CONCEPT_ID,
                self.MATERIAL_ID,
                "The chain rule",
                '["chain rule", "derivative"]',
                "Computing derivatives of composite functions.",
                None,
                "2026-04-16T00:01:00+00:00",
            ),
        )
        return conn

    def _insert_video(
        self,
        conn: sqlite3.Connection,
        video_id: str,
        title: str,
    ) -> None:
        conn.execute(
            "INSERT INTO videos (id, title, channel_title, description, duration_sec, view_count, is_creative_commons, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (video_id, title, "Test Channel", f"{title} description", 600, 1000, 0, "2026-04-16T00:02:00+00:00"),
        )
        conn.execute(
            "INSERT INTO transcript_cache (video_id, transcript_json, created_at) VALUES (?, ?, ?)",
            (video_id, "[]", "2026-04-16T00:02:00+00:00"),
        )

    def _insert_reel(
        self,
        conn: sqlite3.Connection,
        *,
        reel_id: str,
        video_id: str,
        t_start: float,
        t_end: float,
        base_score: float,
        created_at: str = "2026-04-16T00:03:00+00:00",
    ) -> None:
        conn.execute(
            "INSERT INTO reels (id, generation_id, material_id, concept_id, video_id, video_url, "
            "t_start, t_end, transcript_snippet, takeaways_json, base_score, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                reel_id,
                None,
                self.MATERIAL_ID,
                self.CONCEPT_ID,
                video_id,
                f"https://example.com/watch?v={video_id}&t={int(t_start)}",
                t_start,
                t_end,
                f"Reel {reel_id} transcript snippet about the chain rule.",
                '["Chain rule intuition"]',
                base_score,
                created_at,
            ),
        )

    def _fake_relevance(self, *args, **kwargs):
        return {
            "score": 0.8,
            "concept_overlap": 0.5,
            "context_overlap": 0.2,
            "matched_terms": ["chain rule"],
            "off_topic_penalty": 0.0,
            "reason": "matched concept terms",
        }

    def _fake_caption(self, *args, **kwargs):
        return [{"start": 0.0, "end": 5.0, "text": "Chain rule intro"}]

    # ────────────────────────────────────────────────────────────────
    # Test 1: same-video reels stay grouped even when scores interleave
    # ────────────────────────────────────────────────────────────────
    def test_ranked_feed_groups_reels_from_same_video_together(self) -> None:
        conn = self._fresh_conn()
        self._insert_video(conn, "vid-A", "Video A")
        self._insert_video(conn, "vid-B", "Video B")

        # Interleaved scores: A1=1.0, B1=0.9, A2=0.8, B2=0.7, A3=0.6.
        # Before the fix, feed order is strict score-desc: A1, B1, A2, B2, A3.
        # After the fix, video groups stay together (A first since best
        # score 1.0 > 0.9): A1, A2, A3, B1, B2.
        self._insert_reel(conn, reel_id="A1", video_id="vid-A", t_start=10.0, t_end=40.0, base_score=1.0)
        self._insert_reel(conn, reel_id="B1", video_id="vid-B", t_start=20.0, t_end=50.0, base_score=0.9)
        self._insert_reel(conn, reel_id="A2", video_id="vid-A", t_start=60.0, t_end=90.0, base_score=0.8)
        self._insert_reel(conn, reel_id="B2", video_id="vid-B", t_start=80.0, t_end=110.0, base_score=0.7)
        self._insert_reel(conn, reel_id="A3", video_id="vid-A", t_start=120.0, t_end=150.0, base_score=0.6)

        service = ReelService(embedding_service=None, youtube_service=None)
        with mock.patch.object(service, "_score_text_relevance", side_effect=self._fake_relevance), mock.patch.object(
            service, "_build_caption_cues", side_effect=self._fake_caption
        ):
            ranked = service.ranked_feed(conn, self.MATERIAL_ID, fast_mode=True)
        conn.close()

        ids = [r["reel_id"] for r in ranked]
        self.assertEqual(ids, ["A1", "A2", "A3", "B1", "B2"],
                         f"Expected same-video grouping, got {ids}")

    # ────────────────────────────────────────────────────────────────
    # Test 2: within each same-video group, reels play in chronological order
    # ────────────────────────────────────────────────────────────────
    def test_ranked_feed_orders_same_video_reels_by_t_start(self) -> None:
        conn = self._fresh_conn()
        self._insert_video(conn, "vid-X", "Video X")

        # Insert reels so their scores would put them OUT of chronological
        # order. After the fix, they must play in t_start order regardless
        # of score — a clip's parts 1→2→3 must stay contiguous and in order.
        self._insert_reel(conn, reel_id="X2", video_id="vid-X", t_start=60.0, t_end=90.0, base_score=1.0)
        self._insert_reel(conn, reel_id="X3", video_id="vid-X", t_start=120.0, t_end=150.0, base_score=0.9)
        self._insert_reel(conn, reel_id="X1", video_id="vid-X", t_start=10.0, t_end=40.0, base_score=0.8)

        service = ReelService(embedding_service=None, youtube_service=None)
        with mock.patch.object(service, "_score_text_relevance", side_effect=self._fake_relevance), mock.patch.object(
            service, "_build_caption_cues", side_effect=self._fake_caption
        ):
            ranked = service.ranked_feed(conn, self.MATERIAL_ID, fast_mode=True)
        conn.close()

        ids = [r["reel_id"] for r in ranked]
        self.assertEqual(ids, ["X1", "X2", "X3"],
                         f"Expected chronological within-group order, got {ids}")

    # ────────────────────────────────────────────────────────────────
    # Test 3: the highest-scoring video's group goes first
    # ────────────────────────────────────────────────────────────────
    def test_ranked_feed_orders_groups_by_best_score(self) -> None:
        conn = self._fresh_conn()
        self._insert_video(conn, "vid-A", "Video A")
        self._insert_video(conn, "vid-B", "Video B")

        # B's best score (0.95) beats A's best score (0.90) — B's group
        # should come first.
        self._insert_reel(conn, reel_id="A1", video_id="vid-A", t_start=10.0, t_end=40.0, base_score=0.90)
        self._insert_reel(conn, reel_id="A2", video_id="vid-A", t_start=50.0, t_end=80.0, base_score=0.50)
        self._insert_reel(conn, reel_id="B1", video_id="vid-B", t_start=10.0, t_end=40.0, base_score=0.95)
        self._insert_reel(conn, reel_id="B2", video_id="vid-B", t_start=50.0, t_end=80.0, base_score=0.40)

        service = ReelService(embedding_service=None, youtube_service=None)
        with mock.patch.object(service, "_score_text_relevance", side_effect=self._fake_relevance), mock.patch.object(
            service, "_build_caption_cues", side_effect=self._fake_caption
        ):
            ranked = service.ranked_feed(conn, self.MATERIAL_ID, fast_mode=True)
        conn.close()

        ids = [r["reel_id"] for r in ranked]
        self.assertEqual(ids, ["B1", "B2", "A1", "A2"],
                         f"Expected B group first (higher best score), got {ids}")

    # ────────────────────────────────────────────────────────────────
    # Test 4: single-video feed preserves chronological order
    # ────────────────────────────────────────────────────────────────
    def test_ranked_feed_single_video_chronological(self) -> None:
        conn = self._fresh_conn()
        self._insert_video(conn, "vid-S", "Solo Video")

        # Three reels in random score order but clearly chronological t_start.
        self._insert_reel(conn, reel_id="S3", video_id="vid-S", t_start=120.0, t_end=150.0, base_score=0.5)
        self._insert_reel(conn, reel_id="S1", video_id="vid-S", t_start=10.0, t_end=40.0, base_score=1.0)
        self._insert_reel(conn, reel_id="S2", video_id="vid-S", t_start=60.0, t_end=90.0, base_score=0.7)

        service = ReelService(embedding_service=None, youtube_service=None)
        with mock.patch.object(service, "_score_text_relevance", side_effect=self._fake_relevance), mock.patch.object(
            service, "_build_caption_cues", side_effect=self._fake_caption
        ):
            ranked = service.ranked_feed(conn, self.MATERIAL_ID, fast_mode=True)
        conn.close()

        ids = [r["reel_id"] for r in ranked]
        self.assertEqual(ids, ["S1", "S2", "S3"], f"Expected chronological order, got {ids}")


if __name__ == "__main__":
    unittest.main()
