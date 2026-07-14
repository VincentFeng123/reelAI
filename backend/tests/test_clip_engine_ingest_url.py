"""
Tests for the clip-engine-routed ingest_url (Task 9).

Strategy: mock the two heavy external surfaces (clip_engine_run.clip and
clip_engine_meta.youtube_metadata) so the test runs in <1 second with no
network access. The bridge functions are pure and run against the fake data
directly. _persist_ingest writes to a temp SQLite DB (same setUp pattern as
test_ingestion_url.py).
"""

import os
import sys
import tempfile
import threading
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
from backend.app.clip_engine.errors import CancellationError  # noqa: E402


# --------------------------------------------------------------------- #
# Fake engine output
# --------------------------------------------------------------------- #


def _verified_acoustic_result(_url: str, start: float, end: float, **kwargs):
    tolerance = (
        pipeline_module.clip_engine_silence.HANDOFF_TIMESTAMP_TOLERANCE_SEC
    )
    return pipeline_module.clip_engine_silence.SilenceVerificationResult(
        "verified",
        start,
        end,
        {
            "threshold_dbfs": -38.0,
            "semantic_start_limit_sec": kwargs["search_start_limit_sec"],
            "semantic_end_limit_sec": kwargs["search_end_limit_sec"],
            "start_speech_handoff_verified": kwargs[
                "require_start_speech_handoff"
            ],
            "end_speech_handoff_verified": kwargs[
                "require_end_speech_handoff"
            ],
            "start_two_sided_required": kwargs["require_start_two_sided"],
            "end_two_sided_required": kwargs["require_end_two_sided"],
            "start_quiet": [start - tolerance, start + tolerance],
            "end_quiet": [end - tolerance, end + tolerance],
        },
    )


def _fake_engine_out(video_id: str = "dQw4w9WgXcQ") -> dict:
    return {
        "video_id": video_id,
        "clips": [
            {
                "start": 30.0,
                "end": 75.0,
                "cut_end": 75.0,
                "title": "The chorus",
                "facet": "promise-and-response structure",
                "reason": "Directly explains the promise-and-response structure.",
                "learning_objective": "Identify the promise-and-response structure.",
                "kind": "educational",
                "informativeness": 0.9,
                "topic_relevance": 0.9,
                "educational_importance": 0.9,
                "self_contained": True,
                "is_standalone": True,
                "directly_teaches_topic": True,
                "substantive": True,
                "factually_grounded": True,
                "topic_evidence_quote": "Never gonna give you up Never gonna let you down",
                "cue_ids": ["cue-0", "cue-1"],
                "sequence_index": 0,
                "embed_url": f"https://www.youtube.com/embed/{video_id}?start=30&end=75",
            }
        ],
        "transcript": {
            "segments": [
                {"cue_id": "cue-0", "start": 30.0, "end": 55.0, "text": "Never gonna give you up."},
                {"cue_id": "cue-1", "start": 55.0, "end": 75.0, "text": "Never gonna let you down."},
            ],
            "words": [],
            "duration": 300.0,
            "source": "supadata",
            "artifact_key": f"supadata:{video_id}",
            "native_mode": True,
        },
        "notes": "test notes",
    }


# --------------------------------------------------------------------- #
# TestCase
# --------------------------------------------------------------------- #


class ClipEngineIngestUrlTests(unittest.TestCase):
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
            side_effect=_verified_acoustic_result,
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
                    "id": "mat-dedup-test",
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

    # --------------------------------------------------------------------- #
    # Happy path — YouTube URL, clip engine produces one clip
    # --------------------------------------------------------------------- #

    def test_ingest_url_uses_clip_engine(self) -> None:
        source_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

        with (
            mock.patch.object(pipeline_module, "clip_engine_run") as mock_run,
            mock.patch.object(pipeline_module, "clip_engine_meta") as mock_meta,
        ):
            mock_meta.extract_video_id.return_value = "dQw4w9WgXcQ"
            mock_meta.youtube_metadata.return_value = {
                "title": "Never Gonna Give You Up",
                "duration_sec": 300.0,
                "author_name": "Rick Astley",
                "description": "The classic",
                "thumbnail_url": "",
                "view_count": 1_000_000,
            }
            mock_run.clip.return_value = _fake_engine_out()

            result = main_module.ingestion_pipeline.ingest_url(
                source_url=source_url,
                material_id=None,
                concept_id=None,
                target_clip_duration_sec=45,
                target_clip_duration_min_sec=15,
                target_clip_duration_max_sec=60,
                language="en",
            )

        # clip() was called once with the source URL
        mock_run.clip.assert_called_once()
        call_args = mock_run.clip.call_args
        self.assertEqual(call_args.args[0], source_url)

        # Reel shape
        self.assertTrue(
            result.reel.video_url.startswith(
                "https://www.youtube.com/embed/dQw4w9WgXcQ?start="
            ),
            f"Unexpected video_url: {result.reel.video_url}",
        )
        self.assertAlmostEqual(result.reel.t_start, 30.0)
        self.assertAlmostEqual(result.reel.t_end, 75.0)
        self.assertTrue(result.reel.reel_id.startswith("ingest-"))
        self.assertTrue(result.reel.transcript_snippet)

        # Metadata
        self.assertEqual(result.metadata.platform, "yt")
        self.assertEqual(result.metadata.source_id, "dQw4w9WgXcQ")
        self.assertAlmostEqual(result.metadata.duration_sec, 300.0)

        # Envelope
        self.assertTrue(result.trace_id)
        self.assertTrue(result.terms_notice.lower().startswith("reelai"))

        # DB sanity — reel row exists
        with db_module.get_conn() as conn:
            row = db_module.fetch_one(
                conn,
                "SELECT id, video_id, video_url FROM reels WHERE id = ?",
                (result.reel.reel_id,),
            )
            self.assertIsNotNone(row)
            self.assertEqual(row["video_id"], "yt:dQw4w9WgXcQ")

    # --------------------------------------------------------------------- #
    # Non-YouTube URL → UnsupportedSourceError
    # --------------------------------------------------------------------- #

    def test_ingest_url_rejects_non_youtube(self) -> None:
        with mock.patch.object(pipeline_module, "clip_engine_meta") as mock_meta:
            mock_meta.extract_video_id.return_value = None

            with self.assertRaises(UnsupportedSourceError):
                main_module.ingestion_pipeline.ingest_url(
                    source_url="https://vimeo.com/123456789",
                    material_id=None,
                    concept_id=None,
                    target_clip_duration_sec=45,
                    target_clip_duration_min_sec=15,
                    target_clip_duration_max_sec=60,
                    language="en",
                )

        # Verify the error status code maps to a 4xx at the endpoint layer
        self.assertEqual(UnsupportedSourceError.status_code, 400)

    # --------------------------------------------------------------------- #
    # Candidate order is authoritative; deprecated duration inputs are inert.
    # --------------------------------------------------------------------- #

    def test_ingest_url_selects_first_ranked_clip_regardless_of_duration(self) -> None:
        source_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

        # The first clip ranks first even though the deprecated target is 45s.
        multi_clip_engine_out = {
            "video_id": "dQw4w9WgXcQ",
            "clips": [
                {
                    "start": 0.0,
                    "end": 20.0,
                    "cut_end": 20.0,
                    "title": "intro",
                    "facet": "first concept",
                    "reason": "Provides the first complete informational concept.",
                    "learning_objective": "Explain the first informational concept.",
                    "kind": "educational",
                    "informativeness": 0.92,
                    "topic_relevance": 0.92,
                    "educational_importance": 0.92,
                    "difficulty": 0.2,
                    "self_contained": True,
                    "is_standalone": True,
                    "directly_teaches_topic": True,
                    "substantive": True,
                    "factually_grounded": True,
                    "topic_evidence_quote": "The first concept explains a complete useful idea",
                    "cue_ids": ["cue-0"],
                    "sequence_index": 0,
                    "embed_url": "",
                },
                {
                    "start": 30.0,
                    "end": 74.0,
                    "cut_end": 74.0,
                    "title": "chorus",
                    "facet": "second concept",
                    "reason": "Provides a second complete informational concept.",
                    "learning_objective": "Explain the second informational concept.",
                    "kind": "educational",
                    "informativeness": 0.9,
                    "topic_relevance": 0.9,
                    "educational_importance": 0.9,
                    "difficulty": 0.8,
                    "self_contained": True,
                    "is_standalone": True,
                    "directly_teaches_topic": True,
                    "substantive": True,
                    "factually_grounded": True,
                    "topic_evidence_quote": "The second concept also explains a complete useful idea",
                    "cue_ids": ["cue-1"],
                    "sequence_index": 1,
                    "embed_url": "",
                },
            ],
            "transcript": {
                "segments": [
                    {"cue_id": "cue-0", "start": 0.0, "end": 20.0, "text": "The first concept explains a complete useful idea."},
                    {"cue_id": "transition", "start": 20.0, "end": 30.0, "text": "Moving to the next concept."},
                    {"cue_id": "cue-1", "start": 30.0, "end": 74.0, "text": "The second concept also explains a complete useful idea."},
                ],
                "words": [],
                "duration": 212.0,
                "source": "supadata",
                "artifact_key": "supadata:dQw4w9WgXcQ",
                "native_mode": True,
            },
            "notes": "",
        }

        with (
            mock.patch.object(pipeline_module, "clip_engine_run") as mock_run,
            mock.patch.object(pipeline_module, "clip_engine_meta") as mock_meta,
        ):
            mock_meta.extract_video_id.return_value = "dQw4w9WgXcQ"
            mock_meta.youtube_metadata.return_value = {"title": "V", "duration_sec": 212.0}
            mock_run.clip.return_value = multi_clip_engine_out

            result = main_module.ingestion_pipeline.ingest_url(
                source_url=source_url,
                material_id=None,
                concept_id=None,
                target_clip_duration_sec=45,
                target_clip_duration_min_sec=15,
                target_clip_duration_max_sec=60,
                language="en",
            )

        self.assertAlmostEqual(result.reel.t_start, 0.0)
        self.assertAlmostEqual(result.reel.t_end, 20.0)
        self.assertEqual(len(result.reels), 2)
        self.assertEqual(result.reels[0].reel_id, result.reel.reel_id)
        self.assertEqual(
            [(reel.t_start, reel.t_end) for reel in result.reels],
            [(0.0, 20.0), (30.0, 74.0)],
        )
        with db_module.get_conn() as conn:
            rows = db_module.fetch_all(
                conn,
                "SELECT t_start, difficulty FROM reels WHERE video_id = ? ORDER BY t_start",
                ("yt:dQw4w9WgXcQ",),
            )
        self.assertEqual(len(rows), 2)
        self.assertEqual(
            [(float(row["t_start"]), float(row["difficulty"])) for row in rows],
            [(0.0, 0.2), (30.0, 0.8)],
        )

    # --------------------------------------------------------------------- #
    # Idempotency: calling ingest_url twice with the same URL + material_id
    # must return the same reel_id (unique-index collision → load_existing_reel)
    # --------------------------------------------------------------------- #

    def test_ingest_url_idempotent_same_reel_id(self) -> None:
        source_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        material_id = "mat-dedup-test"

        with (
            mock.patch.object(pipeline_module, "clip_engine_run") as mock_run,
            mock.patch.object(pipeline_module, "clip_engine_meta") as mock_meta,
        ):
            mock_meta.extract_video_id.return_value = "dQw4w9WgXcQ"
            mock_meta.youtube_metadata.return_value = {
                "title": "Never Gonna Give You Up",
                "duration_sec": 300.0,
                "author_name": "Rick Astley",
                "description": "The classic",
                "thumbnail_url": "",
                "view_count": 1_000_000,
            }
            mock_run.clip.return_value = _fake_engine_out()

            result1 = main_module.ingestion_pipeline.ingest_url(
                source_url=source_url,
                material_id=material_id,
                concept_id=None,
                target_clip_duration_sec=45,
                target_clip_duration_min_sec=15,
                target_clip_duration_max_sec=60,
                language="en",
            )
            result2 = main_module.ingestion_pipeline.ingest_url(
                source_url=source_url,
                material_id=material_id,
                concept_id=None,
                target_clip_duration_sec=45,
                target_clip_duration_min_sec=15,
                target_clip_duration_max_sec=60,
                language="en",
            )

        # Both calls must resolve to the same persisted reel row
        self.assertEqual(result1.reel.reel_id, result2.reel.reel_id)
        # Verify DB only has one row for this (material_id, video_id, t_start, t_end)
        with db_module.get_conn() as conn:
            row = db_module.fetch_one(
                conn,
                "SELECT COUNT(*) AS cnt FROM reels WHERE id = ?",
                (result1.reel.reel_id,),
            )
            self.assertEqual(row["cnt"], 1)

    def test_cancellation_during_persistence_rolls_back_all_new_rows(self) -> None:
        source_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        cancelled = threading.Event()
        real_upsert_video = pipeline_module.upsert_video

        def cancel_after_video_write(*args, **kwargs):
            result = real_upsert_video(*args, **kwargs)
            cancelled.set()
            return result

        with (
            mock.patch.object(pipeline_module, "clip_engine_run") as mock_run,
            mock.patch.object(pipeline_module, "clip_engine_meta") as mock_meta,
            mock.patch.object(
                pipeline_module,
                "upsert_video",
                side_effect=cancel_after_video_write,
            ),
        ):
            mock_meta.extract_video_id.return_value = "dQw4w9WgXcQ"
            mock_meta.youtube_metadata.return_value = {
                "title": "Cancelled video",
                "duration_sec": 300.0,
            }
            mock_run.clip.return_value = _fake_engine_out()

            with self.assertRaises(CancellationError):
                main_module.ingestion_pipeline.ingest_url(
                    source_url=source_url,
                    material_id=None,
                    concept_id=None,
                    target_clip_duration_sec=45,
                    target_clip_duration_min_sec=15,
                    target_clip_duration_max_sec=60,
                    language="en",
                    should_cancel=cancelled.is_set,
                )

        with db_module.get_conn() as conn:
            self.assertEqual(
                db_module.fetch_one(conn, "SELECT COUNT(*) AS cnt FROM reels")["cnt"],
                0,
            )
            self.assertEqual(
                db_module.fetch_one(conn, "SELECT COUNT(*) AS cnt FROM videos")["cnt"],
                0,
            )


if __name__ == "__main__":
    unittest.main()
