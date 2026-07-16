"""
Tests for the clip-engine-routed ingest_topic_cut (Task 10).

Strategy: mirror test_clip_engine_ingest_url.py — mock clip_engine_run.clip and
clip_engine_meta.youtube_metadata so no network/disk I/O occurs. _persist_ingest
writes to a temp SQLite DB.

Coverage:
  * query="chain rule" → the whole-transcript selector returns one on-topic clip;
    reel_count=1, is_short=False.
  * query=None → two selector-approved informational clips are returned.
  * Each reel's video_url is the YouTube embed URL with its own start/end params.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Skip the import-time orphan sweep so tests don't poke at /tmp during import.
os.environ.setdefault("REELAI_INGEST_SKIP_IMPORT_SWEEP", "1")

from backend.app import db as db_module  # noqa: E402
from backend.app.config import get_settings  # noqa: E402
import backend.app.main as main_module  # noqa: E402
from backend.app.ingestion import pipeline as pipeline_module  # noqa: E402
from backend.app.ingestion.errors import UnsupportedSourceError  # noqa: E402


# --------------------------------------------------------------------- #
# Fake engine output — two clips: one on-topic, one off-topic
# --------------------------------------------------------------------- #


def _fake_engine_out_two_clips(video_id: str = "dQw4w9WgXcQ") -> dict:
    """
    Returns a transcript + 2 clips.
    Clip 0 (30-90s): talks about the "chain rule" in calculus → on-topic.
    Clip 1 (120-180s): talks about "cooking pasta" → off-topic when query="chain rule".
    """
    return {
        "video_id": video_id,
        "clips": [
            {
                "start": 30.0,
                "end": 90.0,
                "cut_end": 90.0,
                "title": "Chain rule differentiation",
                "facet": "calculus",
                "reason": "Explains the chain rule with examples",
                "difficulty": 0.25,
                "sequence_index": 0,
                "embed_url": (
                    f"https://www.youtube.com/embed/{video_id}?start=30&end=90"
                ),
            },
            {
                "start": 120.0,
                "end": 180.0,
                "cut_end": 180.0,
                "title": "Cooking pasta at home",
                "facet": "food",
                "reason": "How to cook spaghetti",
                "difficulty": 0.75,
                "sequence_index": 1,
                "embed_url": (
                    f"https://www.youtube.com/embed/{video_id}?start=120&end=180"
                ),
            },
        ],
        "transcript": {
            "segments": [
                # On-topic: chain rule window
                {
                    "start": 30.0,
                    "end": 60.0,
                    "text": "The chain rule lets you differentiate composite functions.",
                },
                {
                    "start": 60.0,
                    "end": 90.0,
                    "text": "For f(g(x)) the chain rule derivative is f prime of g of x times g prime of x.",
                },
                # Off-topic: cooking window
                {
                    "start": 120.0,
                    "end": 150.0,
                    "text": "Today we are cooking pasta with tomato sauce.",
                },
                {
                    "start": 150.0,
                    "end": 180.0,
                    "text": "Boil the spaghetti for eight minutes then drain.",
                },
            ],
            "words": [],
            "duration": 300.0,
        },
        "notes": "test fixture",
    }


_FAKE_META = {
    "title": "Math and Cooking",
    "duration_sec": 300.0,
    "author_name": "Test Author",
    "description": "A test video",
    "thumbnail_url": "",
    "view_count": 42,
}


# --------------------------------------------------------------------- #
# TestCase
# --------------------------------------------------------------------- #


class ClipEngineTopicCutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.previous_data_dir = os.environ.get("DATA_DIR")
        os.environ["DATA_DIR"] = self.temp_dir.name
        self.addCleanup(self._restore_environment)
        db_module._db_ready = False
        get_settings.cache_clear()
        main_module.settings = get_settings()

        os.environ["REELAI_INGEST_SKIP_IMPORT_SWEEP"] = "1"

        # Bump rate limits so a single test call never trips the per-platform cap.
        from backend.app.ingestion.pipeline import _PlatformRateLimiter
        main_module.ingestion_pipeline._rate_limiter = _PlatformRateLimiter(
            overrides={"yt": (1000, 60.0)}
        )
        prepared = pipeline_module.clip_engine_silence.AudioPreparationResult(
            "ready",
            source=pipeline_module.clip_engine_silence.PreparedAudioSource(
                "https://audio.invalid/test",
                duration_sec=300.0,
            ),
        )
        self._prepare_patch = mock.patch.object(
            pipeline_module.clip_engine_silence,
            "prepare_audio_source",
            return_value=prepared,
        )
        self._prepare_patch.start()
        self.addCleanup(self._prepare_patch.stop)

    def _restore_environment(self) -> None:
        if self.previous_data_dir is None:
            os.environ.pop("DATA_DIR", None)
        else:
            os.environ["DATA_DIR"] = self.previous_data_dir
        db_module._db_ready = False
        get_settings.cache_clear()
        main_module.settings = get_settings()

    # --------------------------------------------------------------------- #
    # query="chain rule" → only on-topic clip survives
    # --------------------------------------------------------------------- #

    def test_query_filters_to_on_topic_clip(self) -> None:
        source_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

        with (
            mock.patch.object(pipeline_module, "clip_engine_run") as mock_run,
            mock.patch.object(pipeline_module, "clip_engine_meta") as mock_meta,
            mock.patch.object(
                pipeline_module,
                "_verified_direct_adapter_clips",
                side_effect=lambda **kwargs: list(kwargs["engine_out"]["clips"]),
            ) as mock_verify,
        ):
            mock_meta.extract_video_id.return_value = "dQw4w9WgXcQ"
            mock_meta.youtube_metadata.return_value = _FAKE_META
            engine_out = _fake_engine_out_two_clips()
            engine_out["clips"] = engine_out["clips"][:1]
            mock_run.clip.return_value = engine_out

            result = main_module.ingestion_pipeline.ingest_topic_cut(
                source_url=source_url,
                query="chain rule",
                language="en",
            )

        mock_verify.assert_called_once()
        self.assertIsNone(mock_verify.call_args.kwargs["limit"])
        self.assertTrue(
            mock_verify.call_args.kwargs["require_acoustic_boundaries"]
        )
        self.assertTrue(mock_verify.call_args.kwargs["prepared_audio"].ready)
        self.assertEqual(mock_verify.call_args.kwargs["exact_topic"], "chain rule")
        self.assertIs(
            mock_verify.call_args.kwargs["embedding_service"],
            main_module.ingestion_pipeline._embedding_service,
        )

        # The exact-topic selector returns only the chain-rule teaching unit.
        self.assertFalse(result.is_short)
        self.assertEqual(result.reel_count, 1)
        self.assertEqual(len(result.reels), 1)

        reel = result.reels[0]
        # The surviving clip is the 30–90s chain rule window
        self.assertAlmostEqual(reel.t_start, 30.0)
        self.assertAlmostEqual(reel.t_end, 90.0)

        # video_url must be a YouTube embed with start/end for THIS clip
        self.assertTrue(
            reel.video_url.startswith("https://www.youtube.com/embed/dQw4w9WgXcQ?start="),
            f"Unexpected video_url: {reel.video_url}",
        )
        self.assertIn("start=30", reel.video_url)
        self.assertIn("end=90", reel.video_url)

        # General shape
        self.assertEqual(result.video_id, "dQw4w9WgXcQ")
        self.assertEqual(result.classification_reason, "long-form")
        self.assertAlmostEqual(result.duration_sec, 300.0)

    # --------------------------------------------------------------------- #
    # query=None → both selector-approved clips are returned
    # --------------------------------------------------------------------- #

    def test_no_query_returns_all_clips(self) -> None:
        source_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

        with (
            mock.patch.object(pipeline_module, "clip_engine_run") as mock_run,
            mock.patch.object(pipeline_module, "clip_engine_meta") as mock_meta,
            mock.patch.object(
                pipeline_module,
                "_verified_direct_adapter_clips",
                side_effect=lambda **kwargs: list(kwargs["engine_out"]["clips"]),
            ) as mock_verify,
        ):
            mock_meta.extract_video_id.return_value = "dQw4w9WgXcQ"
            mock_meta.youtube_metadata.return_value = _FAKE_META
            mock_run.clip.return_value = _fake_engine_out_two_clips()

            result = main_module.ingestion_pipeline.ingest_topic_cut(
                source_url=source_url,
                query=None,
                language="en",
            )

        mock_verify.assert_called_once()
        self.assertEqual(mock_verify.call_args.kwargs["exact_topic"], "")

        # With no additional interest, both selector-approved facets persist.
        self.assertFalse(result.is_short)
        self.assertEqual(result.reel_count, 2)
        self.assertEqual(len(result.reels), 2)

        starts = [r.t_start for r in result.reels]
        self.assertIn(30.0, starts)
        self.assertIn(120.0, starts)

        # Each reel has a distinct embed URL with its own start/end
        for reel in result.reels:
            self.assertIn("start=", reel.video_url)
            self.assertIn("end=", reel.video_url)
            self.assertTrue(
                reel.video_url.startswith("https://www.youtube.com/embed/dQw4w9WgXcQ"),
                f"Unexpected video_url: {reel.video_url}",
            )

        # DB sanity — both reels were written
        with db_module.get_conn() as conn:
            for reel in result.reels:
                row = db_module.fetch_one(
                    conn,
                    "SELECT id, video_id FROM reels WHERE id = ?",
                    (reel.reel_id,),
                )
                self.assertIsNotNone(row, f"reel {reel.reel_id} missing from DB")
                self.assertEqual(row["video_id"], "yt:dQw4w9WgXcQ")
            difficulty_rows = db_module.fetch_all(
                conn,
                "SELECT t_start, difficulty FROM reels WHERE video_id = ? ORDER BY t_start",
                ("yt:dQw4w9WgXcQ",),
            )
        self.assertEqual(
            [float(row["difficulty"]) for row in difficulty_rows],
            [0.25, 0.75],
        )

    # --------------------------------------------------------------------- #
    # Non-YouTube URL → UnsupportedSourceError
    # --------------------------------------------------------------------- #

    def test_non_youtube_raises_unsupported_source(self) -> None:
        with mock.patch.object(pipeline_module, "clip_engine_meta") as mock_meta:
            mock_meta.extract_video_id.return_value = None

            with self.assertRaises(UnsupportedSourceError):
                main_module.ingestion_pipeline.ingest_topic_cut(
                    source_url="https://vimeo.com/123456789",
                    query="anything",
                )

    # --------------------------------------------------------------------- #
    # Short video (duration < 60s, no clips kept) → is_short=True
    # --------------------------------------------------------------------- #

    def test_short_video_returns_is_short_true(self) -> None:
        source_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

        short_engine_out = {
            "video_id": "dQw4w9WgXcQ",
            "clips": [],  # no clips produced for a Short
            "transcript": {
                "segments": [],
                "words": [],
                "duration": 45.0,  # < 60s → short
            },
            "notes": "",
        }

        with (
            mock.patch.object(pipeline_module, "clip_engine_run") as mock_run,
            mock.patch.object(pipeline_module, "clip_engine_meta") as mock_meta,
        ):
            mock_meta.extract_video_id.return_value = "dQw4w9WgXcQ"
            mock_meta.youtube_metadata.return_value = {"duration_sec": 45.0}
            mock_run.clip.return_value = short_engine_out

            result = main_module.ingestion_pipeline.ingest_topic_cut(
                source_url=source_url,
                query="chain rule",
            )

        self.assertTrue(result.is_short)
        self.assertEqual(result.reels, [])
        self.assertEqual(result.reel_count, 0)
        self.assertEqual(result.classification_reason, "short")


if __name__ == "__main__":
    unittest.main()
