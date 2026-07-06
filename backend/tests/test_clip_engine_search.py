"""
Tests for Task 11: ingest_search routed through Supadata + clip engine (YouTube-only).

TDD flow:
  RED  — write test, confirm it fails against the OLD implementation.
  GREEN — after rewriting ingest_search, confirm it passes.

Strategy mirrors test_clip_engine_topic_cut.py:
  - mock clip_engine_search.discover (1 YouTube video returned)
  - mock clip_engine_run.clip (1 clip + transcript)
  - _persist_ingest writes to a temp SQLite DB via main_module.ingestion_pipeline
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

os.environ.setdefault("REELAI_INGEST_SKIP_IMPORT_SWEEP", "1")

from backend.app import db as db_module  # noqa: E402
from backend.app.config import get_settings  # noqa: E402
import backend.app.main as main_module  # noqa: E402
from backend.app.ingestion import pipeline as pipeline_module  # noqa: E402


# --------------------------------------------------------------------- #
# Fake data
# --------------------------------------------------------------------- #

_VIDEO_ID = "dQw4w9WgXcQ"
_VIDEO_URL = f"https://www.youtube.com/watch?v={_VIDEO_ID}"

_FAKE_DISCOVER = {
    "corrected": "calc",
    "credits_used": 1,
    "warning": None,
    "videos": [
        {
            "id": _VIDEO_ID,
            "url": _VIDEO_URL,
            "title": "V1",
            "channel": "Ch",
            "duration": 300,
            "view_count": 10,
            "thumbnail": "t",
            "upload_date": "",
        }
    ],
}

_FAKE_ENGINE_OUT = {
    "video_id": _VIDEO_ID,
    "clips": [
        {
            "start": 20.0,
            "end": 65.0,
            "cut_end": 65.0,
            "title": "Intro to calc",
            "facet": "math",
            "reason": "explains calculus basics",
            "sequence_index": 0,
            "embed_url": f"https://www.youtube.com/embed/{_VIDEO_ID}?start=20&end=65",
        }
    ],
    "transcript": {
        "segments": [
            {"start": 20.0, "end": 40.0, "text": "Calculus is the study of change."},
            {"start": 40.0, "end": 65.0, "text": "Derivatives measure instantaneous rate."},
        ],
        "words": [],
        "duration": 300.0,
    },
    "notes": "test fixture",
}


# --------------------------------------------------------------------- #
# TestCase
# --------------------------------------------------------------------- #


class ClipEngineSearchTests(unittest.TestCase):
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

        # Raise rate limits so the single test call never trips the per-platform cap.
        from backend.app.ingestion.pipeline import _PlatformRateLimiter
        main_module.ingestion_pipeline._rate_limiter = _PlatformRateLimiter(
            overrides={"yt": (1000, 60.0)}
        )

    def _restore_environment(self) -> None:
        if self.previous_data_dir is None:
            os.environ.pop("DATA_DIR", None)
        else:
            os.environ["DATA_DIR"] = self.previous_data_dir
        db_module._db_ready = False
        get_settings.cache_clear()
        main_module.settings = get_settings()

    # --------------------------------------------------------------------- #
    # Core happy-path: 1 video discovered, 1 clip produced, 1 reel persisted
    # --------------------------------------------------------------------- #

    def test_ingest_search_happy_path(self) -> None:
        """
        Caller passes platforms=["yt","ig","tt"]; result must have platforms==["yt"]
        (YouTube-only coercion). One video is discovered, one clip produced,
        one reel row written to the DB.
        """
        with (
            mock.patch.object(pipeline_module, "clip_engine_search") as mock_search,
            mock.patch.object(pipeline_module, "clip_engine_run") as mock_run,
        ):
            mock_search.discover.return_value = _FAKE_DISCOVER
            mock_run.clip.return_value = _FAKE_ENGINE_OUT

            result = main_module.ingestion_pipeline.ingest_search(
                query="calc",
                platforms=["yt", "ig", "tt"],
                max_per_platform=5,
                material_id="m1",
                concept_id=None,
                target_clip_duration_sec=45,
                target_clip_duration_min_sec=15,
                target_clip_duration_max_sec=60,
                language="en",
                exclude_video_ids=[],
            )

        # Platform coercion
        self.assertEqual(result.platforms, ["yt"])

        # Counts
        self.assertEqual(result.succeeded, 1)
        self.assertEqual(result.failed, 0)
        self.assertEqual(result.total_resolved, 1)

        # Single item
        self.assertEqual(len(result.items), 1)
        item = result.items[0]
        self.assertEqual(item.status, "ok")
        self.assertEqual(item.platform, "yt")
        self.assertEqual(item.source_url, _VIDEO_URL)

        # Reel has correct embed URL
        self.assertIsNotNone(item.reel)
        self.assertTrue(
            item.reel.video_url.startswith(f"https://www.youtube.com/embed/{_VIDEO_ID}"),
            f"Unexpected video_url: {item.reel.video_url}",
        )
        self.assertIn("start=", item.reel.video_url)
        self.assertIn("end=", item.reel.video_url)

        # DB sanity
        with db_module.get_conn() as conn:
            row = db_module.fetch_one(
                conn,
                "SELECT id, video_id FROM reels WHERE id = ?",
                (item.reel.reel_id,),
            )
        self.assertIsNotNone(row, f"reel {item.reel.reel_id} missing from DB")
        self.assertEqual(row["video_id"], f"yt:{_VIDEO_ID}")

        # terms_notice mentions YouTube-only
        self.assertIn("YouTube", result.terms_notice)

        # material_id was explicitly provided, should be preserved
        self.assertEqual(result.material_id, "m1")

        # discover called with correct args
        mock_search.discover.assert_called_once_with(
            "calc", limit=5, exclude_video_ids=[]
        )

    # --------------------------------------------------------------------- #
    # No clips → status="skipped"
    # --------------------------------------------------------------------- #

    def test_no_clips_status_skipped(self) -> None:
        empty_engine_out = {
            "video_id": _VIDEO_ID,
            "clips": [],
            "transcript": {"segments": [], "words": [], "duration": 300.0},
            "notes": "",
        }

        with (
            mock.patch.object(pipeline_module, "clip_engine_search") as mock_search,
            mock.patch.object(pipeline_module, "clip_engine_run") as mock_run,
        ):
            mock_search.discover.return_value = _FAKE_DISCOVER
            mock_run.clip.return_value = empty_engine_out

            result = main_module.ingestion_pipeline.ingest_search(
                query="calc",
                platforms=["yt"],
                max_per_platform=5,
                material_id=None,
                concept_id=None,
                target_clip_duration_sec=45,
                target_clip_duration_min_sec=15,
                target_clip_duration_max_sec=60,
                language="en",
                exclude_video_ids=[],
            )

        self.assertEqual(result.succeeded, 0)
        self.assertEqual(len(result.items), 1)
        self.assertEqual(result.items[0].status, "skipped")
        self.assertIsNone(result.items[0].reel)

    # --------------------------------------------------------------------- #
    # Per-video exception → status="error" (non-fatal)
    # --------------------------------------------------------------------- #

    def test_per_video_exception_is_non_fatal(self) -> None:
        with (
            mock.patch.object(pipeline_module, "clip_engine_search") as mock_search,
            mock.patch.object(pipeline_module, "clip_engine_run") as mock_run,
        ):
            mock_search.discover.return_value = _FAKE_DISCOVER
            mock_run.clip.side_effect = RuntimeError("network timeout")

            result = main_module.ingestion_pipeline.ingest_search(
                query="calc",
                platforms=["yt"],
                max_per_platform=5,
                material_id=None,
                concept_id=None,
                target_clip_duration_sec=45,
                target_clip_duration_min_sec=15,
                target_clip_duration_max_sec=60,
                language="en",
                exclude_video_ids=[],
            )

        self.assertEqual(result.succeeded, 0)
        self.assertEqual(result.failed, 1)
        self.assertEqual(len(result.items), 1)
        self.assertEqual(result.items[0].status, "error")
        self.assertIn("network timeout", result.items[0].error)

    # --------------------------------------------------------------------- #
    # No material_id → sentinel material created deterministically
    # --------------------------------------------------------------------- #

    def test_no_material_id_creates_sentinel(self) -> None:
        with (
            mock.patch.object(pipeline_module, "clip_engine_search") as mock_search,
            mock.patch.object(pipeline_module, "clip_engine_run") as mock_run,
        ):
            mock_search.discover.return_value = _FAKE_DISCOVER
            mock_run.clip.return_value = _FAKE_ENGINE_OUT

            result = main_module.ingestion_pipeline.ingest_search(
                query="calculus basics",
                platforms=["yt"],
                max_per_platform=5,
                material_id=None,
                concept_id=None,
                target_clip_duration_sec=45,
                target_clip_duration_min_sec=15,
                target_clip_duration_max_sec=60,
                language="en",
                exclude_video_ids=[],
            )

        # sentinel material should follow ingest-search:<hash> pattern
        self.assertTrue(
            result.material_id.startswith("ingest-search:"),
            f"Unexpected material_id: {result.material_id}",
        )
        self.assertEqual(result.succeeded, 1)


if __name__ == "__main__":
    unittest.main()
