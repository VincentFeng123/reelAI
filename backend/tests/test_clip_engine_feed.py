"""
Tests for Task 12: ingest_feed routed through clip engine (YouTube-only).

TDD flow:
  RED  — write tests, confirm they fail against the OLD implementation.
  GREEN — after implementing resolve_feed_urls + rewriting ingest_feed, confirm they pass.

Strategy mirrors test_clip_engine_search.py:
  - pass one direct YouTube video URL
  - mock clip_engine_run.clip (Supadata transcript + Gemini clips)
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

_FAKE_ENGINE_OUT = {
    "video_id": _VIDEO_ID,
    "clips": [
        {
            "start": 20.0,
            "end": 65.0,
            "cut_end": 65.0,
            "title": "Intro",
            "facet": "topic overview",
            "reason": "Explains the main subject directly.",
            "learning_objective": "Explain the main subject in context.",
            "kind": "educational",
            "informativeness": 0.9,
            "topic_relevance": 0.9,
            "educational_importance": 0.9,
            "self_contained": True,
            "is_standalone": True,
            "directly_teaches_topic": True,
            "substantive": True,
            "factually_grounded": True,
            "topic_evidence_quote": "Hello and welcome Today we discuss things",
            "cue_ids": ["cue-0", "cue-1"],
            "sequence_index": 0,
            "embed_url": f"https://www.youtube.com/embed/{_VIDEO_ID}?start=20&end=65",
        }
    ],
    "transcript": {
        "segments": [
            {"cue_id": "cue-0", "start": 20.0, "end": 40.0, "text": "Hello and welcome."},
            {"cue_id": "cue-1", "start": 40.0, "end": 65.0, "text": "Today we discuss things."},
        ],
        "words": [],
        "duration": 300.0,
        "source": "supadata",
        "artifact_key": f"supadata:{_VIDEO_ID}",
        "native_mode": True,
    },
    "notes": "test fixture",
}


# --------------------------------------------------------------------- #
# TestCase: ingest_feed integration
# --------------------------------------------------------------------- #


class ClipEngineFeedTests(unittest.TestCase):
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

        # Raise rate limits so the test never trips the per-platform cap.
        from backend.app.ingestion.pipeline import _PlatformRateLimiter
        main_module.ingestion_pipeline._rate_limiter = _PlatformRateLimiter(
            overrides={"yt": (1000, 60.0)}
        )
        prepared = pipeline_module.clip_engine_silence.AudioPreparationResult(
            "ready",
            source=pipeline_module.clip_engine_silence.PreparedAudioSource(
                "https://audio.invalid/test"
            ),
        )
        self._prepare_patch = mock.patch.object(
            pipeline_module.clip_engine_silence,
            "prepare_audio_source",
            return_value=prepared,
        )
        self._verify_patch = mock.patch.object(
            pipeline_module.clip_engine_silence,
            "verify_acoustic_boundaries",
            side_effect=lambda _url, start, end, **_kwargs: (
                pipeline_module.clip_engine_silence.SilenceVerificationResult(
                    "verified", start, end, {"threshold_dbfs": -38.0}
                )
            ),
        )
        self._prepare_patch.start()
        self._verify_patch.start()
        self.addCleanup(self._prepare_patch.stop)
        self.addCleanup(self._verify_patch.stop)
        with db_module.get_conn(transactional=True) as conn:
            db_module.insert(
                conn,
                "materials",
                {
                    "id": "m1",
                    "subject_tag": "test",
                    "raw_text": "test",
                    "source_type": "topic",
                    "source_path": None,
                    "created_at": db_module.now_iso(),
                },
            )

    def _restore_environment(self) -> None:
        if self.previous_data_dir is None:
            os.environ.pop("DATA_DIR", None)
        else:
            os.environ["DATA_DIR"] = self.previous_data_dir
        db_module._db_ready = False
        get_settings.cache_clear()
        main_module.settings = get_settings()

    # ------------------------------------------------------------------ #
    # Happy path: 1 URL resolved → 1 clip → 1 reel persisted
    # ------------------------------------------------------------------ #

    def test_ingest_feed_happy_path(self) -> None:
        with mock.patch.object(pipeline_module, "clip_engine_run") as mock_run:
            mock_run.clip.return_value = _FAKE_ENGINE_OUT

            result = main_module.ingestion_pipeline.ingest_feed(
                feed_url=_VIDEO_URL,
                max_items=6,
                material_id="m1",
                concept_id=None,
                target_clip_duration_sec=45,
                target_clip_duration_min_sec=15,
                target_clip_duration_max_sec=60,
                language="en",
            )

        self.assertEqual(result.total_resolved, 1)
        self.assertEqual(result.succeeded, 1)
        self.assertEqual(result.failed, 0)
        self.assertEqual(len(result.items), 1)

        item = result.items[0]
        self.assertEqual(item.status, "ok")
        self.assertEqual(item.source_url, _VIDEO_URL)

        # Reel must have an embed URL with start/end params
        self.assertIsNotNone(item.reel)
        self.assertTrue(
            item.reel.video_url.startswith(
                f"https://www.youtube.com/embed/{_VIDEO_ID}"
            ),
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

    # ------------------------------------------------------------------ #
    # No clips → status="skipped" (non-fatal)
    # ------------------------------------------------------------------ #

    def test_no_clips_status_skipped(self) -> None:
        empty_out = {
            "video_id": _VIDEO_ID,
            "clips": [],
            "transcript": {"segments": [], "words": [], "duration": 300.0},
            "notes": "",
        }

        with mock.patch.object(pipeline_module, "clip_engine_run") as mock_run:
            mock_run.clip.return_value = empty_out

            result = main_module.ingestion_pipeline.ingest_feed(
                feed_url=_VIDEO_URL,
                max_items=6,
                material_id="m1",
                concept_id=None,
                target_clip_duration_sec=45,
                target_clip_duration_min_sec=15,
                target_clip_duration_max_sec=60,
                language="en",
            )

        self.assertEqual(result.succeeded, 0)
        self.assertEqual(len(result.items), 1)
        self.assertEqual(result.items[0].status, "skipped")
        self.assertIsNone(result.items[0].reel)

    # ------------------------------------------------------------------ #
    # Per-URL exception → status="error" (non-fatal)
    # ------------------------------------------------------------------ #

    def test_per_url_exception_is_non_fatal(self) -> None:
        with mock.patch.object(pipeline_module, "clip_engine_run") as mock_run:
            mock_run.clip.side_effect = RuntimeError("network timeout")

            result = main_module.ingestion_pipeline.ingest_feed(
                feed_url=_VIDEO_URL,
                max_items=6,
                material_id="m1",
                concept_id=None,
                target_clip_duration_sec=45,
                target_clip_duration_min_sec=15,
                target_clip_duration_max_sec=60,
                language="en",
            )

        self.assertEqual(result.succeeded, 0)
        self.assertEqual(result.failed, 1)
        self.assertEqual(len(result.items), 1)
        self.assertEqual(result.items[0].status, "error")
        self.assertIn("network timeout", result.items[0].error)

    def test_channel_feed_is_rejected_in_supadata_only_mode(self) -> None:
        with self.assertRaises(pipeline_module.UnsupportedSourceError):
            main_module.ingestion_pipeline.ingest_feed(
                feed_url="https://www.youtube.com/@chan",
                max_items=6,
                material_id="m1",
                concept_id=None,
            )


# --------------------------------------------------------------------- #
# Unit tests for resolve_feed_urls (yt_dlp mocked at sys.modules level)
# --------------------------------------------------------------------- #


class ResolveFeedUrlsUnitTests(unittest.TestCase):
    def test_resolve_feed_urls_returns_watch_urls(self) -> None:
        """resolve_feed_urls extracts watch URLs from yt_dlp entries."""
        from backend.app.clip_engine.metadata import resolve_feed_urls

        fake_entry = {"id": "dQw4w9WgXcQ"}
        fake_info = {"entries": [fake_entry]}

        fake_ydl = mock.MagicMock()
        fake_ydl.__enter__ = mock.Mock(return_value=fake_ydl)
        fake_ydl.__exit__ = mock.Mock(return_value=False)
        fake_ydl.extract_info = mock.Mock(return_value=fake_info)

        mock_yt_dlp = mock.MagicMock()
        mock_yt_dlp.YoutubeDL.return_value = fake_ydl

        with mock.patch.dict("sys.modules", {"yt_dlp": mock_yt_dlp}):
            urls = resolve_feed_urls("https://www.youtube.com/@chan", max_items=5)

        self.assertEqual(urls, ["https://www.youtube.com/watch?v=dQw4w9WgXcQ"])

    def test_resolve_feed_urls_respects_max_items(self) -> None:
        """resolve_feed_urls truncates to max_items."""
        from backend.app.clip_engine.metadata import resolve_feed_urls

        fake_entries = [{"id": f"vid{i:08d}"} for i in range(10)]
        fake_info = {"entries": fake_entries}

        fake_ydl = mock.MagicMock()
        fake_ydl.__enter__ = mock.Mock(return_value=fake_ydl)
        fake_ydl.__exit__ = mock.Mock(return_value=False)
        fake_ydl.extract_info = mock.Mock(return_value=fake_info)

        mock_yt_dlp = mock.MagicMock()
        mock_yt_dlp.YoutubeDL.return_value = fake_ydl

        with mock.patch.dict("sys.modules", {"yt_dlp": mock_yt_dlp}):
            urls = resolve_feed_urls("https://www.youtube.com/@chan", max_items=3)

        self.assertEqual(len(urls), 3)

    def test_resolve_feed_urls_returns_empty_on_failure(self) -> None:
        """resolve_feed_urls swallows exceptions and returns []."""
        from backend.app.clip_engine.metadata import resolve_feed_urls

        mock_yt_dlp = mock.MagicMock()
        mock_yt_dlp.YoutubeDL.side_effect = RuntimeError("connection refused")

        with mock.patch.dict("sys.modules", {"yt_dlp": mock_yt_dlp}):
            urls = resolve_feed_urls("https://www.youtube.com/@chan", max_items=5)

        self.assertEqual(urls, [])


if __name__ == "__main__":
    unittest.main()
