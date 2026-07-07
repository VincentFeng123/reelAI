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
from backend.app.services.clip_whisper_refine import WhisperRefinement, WhisperWord  # noqa: E402
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

    def test_rank_segments_preserves_topic_cut_metadata(self) -> None:
        service = ReelService(embedding_service=None, youtube_service=None)
        segment = SegmentMatch(
            chunk_index=0,
            t_start=10.0,
            t_end=40.0,
            text="Gradient descent updates parameters using the negative gradient.",
            score=0.7,
            source="topic_cut",
            final_rank_score=0.91,
            relevance_score=0.83,
            engagement_score=0.62,
            completeness_score=0.74,
            boundary_confidence=0.88,
            boundary_quality="llm-rerank",
            hook_pattern="definition",
        )

        with mock.patch.dict(os.environ, {"REELS_IMPORTANCE_RANKER_ENABLED": "0"}, clear=False):
            with mock.patch.object(
                service,
                "_score_text_relevance",
                return_value={"score": 0.5, "matched_terms": ["gradient"], "context_hits": []},
            ):
                with mock.patch.object(service, "_passes_relevance_gate", return_value=True):
                    ranked = service._rank_segments_by_relevance(
                        None,
                        [segment],
                        ["gradient"],
                        [],
                        None,
                        None,
                        False,
                        False,
                        query_text="gradient descent",
                        video_duration_sec=60.0,
                    )

        self.assertEqual(len(ranked), 1)
        ranked_segment = ranked[0][0]
        self.assertEqual(ranked_segment.source, "topic_cut")
        self.assertAlmostEqual(ranked_segment.final_rank_score, 0.91)
        self.assertAlmostEqual(ranked_segment.relevance_score or 0.0, 0.83)
        self.assertEqual(ranked_segment.boundary_quality, "llm-rerank")
        self.assertEqual(ranked_segment.hook_pattern, "definition")

    def test_strict_search_clip_window_uses_acoustic_word_edges(self) -> None:
        service = ReelService(embedding_service=None, youtube_service=None)
        refined = WhisperRefinement(
            t_start=9.8,
            t_end=35.2,
            words=[
                WhisperWord(text="So", start=10.00, end=10.12),
                WhisperWord(text="gradient", start=10.30, end=10.72),
                WhisperWord(text="descent", start=10.75, end=11.10),
                WhisperWord(text="converges.", start=34.72, end=35.00),
            ],
            first_word="gradient",
            last_word="converges.",
        )

        with mock.patch(
            "backend.app.services.clip_whisper_refine.refine_clip_with_whisper",
            return_value=refined,
        ) as refine_mock:
            result = service._finalize_search_clip_window(
                None,
                video={"id": "abc123xyz00", "provider": "youtube", "duration_sec": 80},
                clip_window=(9.8, 35.2),
                clip_min_len=20,
                clip_max_len=55,
            )

        self.assertIsNotNone(result)
        assert result is not None
        window, finalized = result
        self.assertTrue(finalized)
        self.assertEqual(window, (10.22, 35.08))
        refine_mock.assert_called_once()
        self.assertTrue(refine_mock.call_args.kwargs["force"])

    def test_topic_boundary_conversion_preserves_topic_reel_rank(self) -> None:
        from backend.app.services.topic_cut import TopicReel, VideoClassification

        service = ReelService(embedding_service=None, youtube_service=None)
        topic_reel = TopicReel(
            video_id="vid-1",
            t_start=12.0,
            t_end=42.0,
            label="Dense explanation",
            relevance_score=0.82,
            boundary_quality="llm-rerank",
            engagement_score=0.6,
            completeness_score=0.7,
            boundary_confidence=0.9,
            final_rank_score=0.94,
            hook_pattern="definition",
        )
        classification = VideoClassification(
            video_id="vid-1",
            is_short=False,
            duration_sec=90.0,
            reason="long-form",
        )

        with mock.patch(
            "backend.app.services.topic_cut.cut_video_into_topic_reels",
            return_value=(classification, [topic_reel]),
        ):
            segments = service._topic_boundary_segments_for_concept(
                transcript=[
                    _transcript_entry(12.0, 10.0, "Gradient descent is an optimization method."),
                    _transcript_entry(22.0, 10.0, "It follows the negative gradient."),
                    _transcript_entry(32.0, 10.0, "That makes the loss smaller."),
                ],
                video_id="vid-1",
                video_duration_sec=90,
                clip_min_len=20,
                clip_max_len=55,
                clip_target_len=40,
                max_segments=3,
                concept_terms=["gradient descent"],
                provider="vimeo",
            )

        self.assertEqual(len(segments), 1)
        self.assertAlmostEqual(segments[0].score, 0.94)
        self.assertAlmostEqual(segments[0].final_rank_score, 0.94)
        self.assertEqual(segments[0].boundary_quality, "llm-rerank")
        self.assertEqual(segments[0].hook_pattern, "definition")

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

    def test_create_reel_does_not_trim_finalized_word_window(self) -> None:
        conn = self._build_conn()
        service = ReelService(embedding_service=None, youtube_service=None)
        concept = {"id": "concept-1", "title": "Gradient descent"}
        video = {
            "id": "video-1",
            "duration_sec": 60,
            "title": "Gradient descent lecture",
            "description": "",
            "channel_title": "Math Channel",
        }
        transcript = [
            _transcript_entry(0.0, 8.0, "Hey everyone, welcome back to the channel."),
            _transcript_entry(8.0, 12.0, "Gradient descent follows the steepest direction."),
            _transcript_entry(20.0, 12.0, "Each update subtracts the learning rate times the gradient."),
        ]
        segment = SegmentMatch(
            chunk_index=0,
            t_start=0.0,
            t_end=32.0,
            text=" ".join(str(entry["text"]) for entry in transcript),
            score=0.9,
        )

        with mock.patch.object(service, "_brief_ai_summary", return_value=""):
            with mock.patch.object(service, "_build_caption_cues", return_value=[]):
                reel = service._create_reel(
                    conn,
                    "material-1",
                    concept,
                    video,
                    segment,
                    clip_window=(0.08, 24.08),
                    transcript=transcript,
                    relevance_context={"query_text": "gradient descent", "score": 0.91},
                    target_clip_duration_sec=55,
                    finalized_clip_window=True,
                )

        self.assertIsNotNone(reel)
        assert reel is not None
        self.assertEqual(reel["t_start"], 0.08)
        self.assertEqual(reel["t_end"], 24.08)

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
