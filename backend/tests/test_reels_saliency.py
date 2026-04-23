from __future__ import annotations

import os
import sqlite3
import sys
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("VERCEL", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.services.reels import ReelService  # noqa: E402
from backend.app.services.segmenter import SegmentMatch  # noqa: E402


def _transcript_entry(start: float, duration: float, text: str) -> dict[str, object]:
    return {"start": start, "duration": duration, "text": text}


class ReelSaliencyTests(unittest.TestCase):
    def _build_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript(
            """
            CREATE TABLE reels (
                id TEXT PRIMARY KEY,
                generation_id TEXT,
                material_id TEXT NOT NULL,
                concept_id TEXT NOT NULL,
                video_id TEXT NOT NULL,
                video_url TEXT NOT NULL DEFAULT '',
                t_start REAL NOT NULL,
                t_end REAL NOT NULL,
                transcript_snippet TEXT NOT NULL DEFAULT '',
                takeaways_json TEXT NOT NULL DEFAULT '[]',
                base_score REAL NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT ''
            );
            """
        )
        return conn

    def test_rank_segments_by_relevance_excludes_structural_segments(self) -> None:
        service = ReelService(embedding_service=None, youtube_service=None)
        segments = [
            SegmentMatch(
                chunk_index=0,
                t_start=0.0,
                t_end=30.0,
                text="Hey everyone, welcome back to the channel. Today we are talking about torque.",
                score=0.82,
            ),
            SegmentMatch(
                chunk_index=1,
                t_start=30.0,
                t_end=60.0,
                text="Torque equals force times lever arm, so it measures rotational effect.",
                score=0.79,
            ),
            SegmentMatch(
                chunk_index=2,
                t_start=60.0,
                t_end=90.0,
                text="This video is sponsored by Brilliant. Use code TORQUE for a discount.",
                score=0.91,
            ),
            SegmentMatch(
                chunk_index=3,
                t_start=90.0,
                t_end=120.0,
                text="You compute torque by multiplying force by the perpendicular distance.",
                score=0.76,
            ),
        ]

        def _fake_score(*args, **kwargs) -> dict[str, object]:
            return {"score": 0.42, "matched_terms": ["torque"], "context_hits": []}

        with mock.patch.dict(os.environ, {"REELS_IMPORTANCE_RANKER_ENABLED": "1"}, clear=False):
            with mock.patch.object(service, "_score_text_relevance", side_effect=_fake_score):
                with mock.patch.object(service, "_passes_relevance_gate", return_value=True):
                    ranked = service._rank_segments_by_relevance(
                        None,
                        segments,
                        ["torque"],
                        [],
                        None,
                        None,
                        False,
                        False,
                        query_text="torque",
                        video_duration_sec=120.0,
                    )

        kept_chunk_indices = [segment.chunk_index for segment, _ in ranked]
        self.assertEqual(kept_chunk_indices, [1, 3])

    def test_create_reel_trims_edges_and_persists_query_focused_snippet(self) -> None:
        conn = self._build_conn()
        service = ReelService(embedding_service=None, youtube_service=None)
        concept = {"id": "concept-1", "title": "Gradient descent", "keywords": ["gradient", "descent"]}
        video = {
            "id": "video-1",
            "duration_sec": 60,
            "title": "Gradient descent lecture",
            "description": "A lecture about optimization.",
            "channel_title": "Math Channel",
        }
        transcript = [
            _transcript_entry(0.0, 8.0, "Hey everyone, welcome back to the channel."),
            _transcript_entry(8.0, 12.0, "Gradient descent follows the steepest direction of decrease."),
            _transcript_entry(20.0, 14.0, "The most important part is the negative gradient, which points downhill."),
            _transcript_entry(34.0, 12.0, "That is why each update subtracts the learning rate times the gradient."),
            _transcript_entry(46.0, 8.0, "Thanks for watching and subscribe for more."),
        ]
        segment = SegmentMatch(
            chunk_index=0,
            t_start=0.0,
            t_end=54.0,
            text=" ".join(str(entry["text"]) for entry in transcript),
            score=0.84,
        )

        with mock.patch.dict(os.environ, {"REELS_IMPORTANCE_RANKER_ENABLED": "1"}, clear=False):
            with mock.patch.object(service, "_brief_ai_summary", return_value=""):
                with mock.patch.object(service, "_build_caption_cues", return_value=[]):
                    reel = service._create_reel(
                        conn,
                        "material-1",
                        concept,
                        video,
                        segment,
                        transcript=transcript,
                        relevance_context={"query_text": "negative gradient", "score": 0.88},
                    )

        self.assertIsNotNone(reel)
        assert reel is not None
        self.assertEqual(reel["t_start"], 8.0)
        self.assertEqual(reel["t_end"], 46.0)
        snippet = str(reel["transcript_snippet"]).lower()
        self.assertIn("negative gradient", snippet)
        self.assertNotIn("welcome back", snippet)
        self.assertNotIn("thanks for watching", snippet)

        row = conn.execute(
            "SELECT transcript_snippet, t_start, t_end FROM reels WHERE id = ?",
            (reel["reel_id"],),
        ).fetchone()
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row["transcript_snippet"], reel["transcript_snippet"])
        self.assertEqual(row["t_start"], reel["t_start"])
        self.assertEqual(row["t_end"], reel["t_end"])

    def test_create_reel_drops_clip_when_trimmed_result_is_too_short(self) -> None:
        conn = self._build_conn()
        service = ReelService(embedding_service=None, youtube_service=None)
        concept = {"id": "concept-1", "title": "Gradient descent"}
        video = {
            "id": "video-1",
            "duration_sec": 30,
            "title": "Short gradient clip",
            "description": "",
            "channel_title": "Math Channel",
        }
        transcript = [
            _transcript_entry(0.0, 10.0, "Hey everyone, welcome back to the channel."),
            _transcript_entry(10.0, 8.0, "Gradient descent picks the steepest direction."),
            _transcript_entry(18.0, 10.0, "Thanks for watching and subscribe for more."),
        ]
        segment = SegmentMatch(
            chunk_index=0,
            t_start=0.0,
            t_end=28.0,
            text=" ".join(str(entry["text"]) for entry in transcript),
            score=0.62,
        )

        with mock.patch.dict(os.environ, {"REELS_IMPORTANCE_RANKER_ENABLED": "1"}, clear=False):
            with mock.patch.object(service, "_brief_ai_summary", return_value=""):
                with mock.patch.object(service, "_build_caption_cues", return_value=[]):
                    reel = service._create_reel(
                        conn,
                        "material-1",
                        concept,
                        video,
                        segment,
                        transcript=transcript,
                        relevance_context={"query_text": "gradient descent", "score": 0.71},
                    )

        self.assertIsNone(reel)
        row = conn.execute("SELECT COUNT(*) AS n FROM reels").fetchone()
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row["n"], 0)

    def test_clip_trim_preserves_existing_end_when_only_opener_is_removed(self) -> None:
        service = ReelService(embedding_service=None, youtube_service=None)
        transcript = [
            _transcript_entry(0.0, 8.0, "Hey everyone, welcome back to the channel."),
            _transcript_entry(8.0, 12.0, "Gradient descent follows the steepest direction of decrease."),
            _transcript_entry(20.0, 14.0, "The most important part is the negative gradient, which points downhill."),
            _transcript_entry(34.0, 5.37, "That is why each update subtracts the learning rate times the gradient."),
        ]

        trimmed = service._trim_structural_edges_from_clip(
            transcript,
            clip_start=0.0,
            clip_end=39.37,
            video_duration_sec=60,
            min_len=19,
        )

        self.assertEqual(trimmed, (8.0, 39.37))


if __name__ == "__main__":
    unittest.main()
