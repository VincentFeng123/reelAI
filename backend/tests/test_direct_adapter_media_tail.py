"""Regression coverage for verified cuts beyond the final caption timestamp."""

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


VIDEO_ID = "dQw4w9WgXcQ"
SOURCE_URL = f"https://www.youtube.com/watch?v={VIDEO_ID}"


def _media_tail_engine_out() -> dict:
    text = "Photosynthesis converts light energy into stored chemical energy for plants."
    return {
        "video_id": VIDEO_ID,
        "clips": [
            {
                "start": 0.0,
                "end": 10.0,
                "title": "Photosynthesis stores light energy",
                "learning_objective": "Explain how photosynthesis stores light energy.",
                "facet": "energy conversion",
                "reason": "Directly explains the energy conversion in photosynthesis.",
                "kind": "educational",
                "informativeness": 0.9,
                "topic_relevance": 0.9,
                "educational_importance": 0.9,
                "difficulty": 0.2,
                "self_contained": True,
                "is_standalone": True,
                "directly_teaches_topic": True,
                "substantive": True,
                "factually_grounded": True,
                "topic_evidence_quote": text,
                "cue_ids": ["tail-cue"],
                "selection_candidate_id": "tail-candidate",
                "uncertainty": "low",
                "prerequisite_ids": [],
            }
        ],
        "transcript": {
            "segments": [
                {
                    "cue_id": "tail-cue",
                    "start": 0.0,
                    "end": 10.0,
                    "text": text,
                }
            ],
            "words": [],
            "duration": 10.0,
            "source": "supadata",
            "artifact_key": f"supadata:{VIDEO_ID}",
            "native_mode": True,
        },
        "notes": "",
    }


class DirectAdapterMediaTailTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.previous_data_dir = os.environ.get("DATA_DIR")
        os.environ["DATA_DIR"] = self.temp_dir.name
        self.addCleanup(self._restore_environment)
        db_module._db_ready = False
        get_settings.cache_clear()
        main_module.settings = get_settings()

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

    def _patch_media_tail(self):
        prepared = pipeline_module.clip_engine_silence.AudioPreparationResult(
            "ready",
            source=pipeline_module.clip_engine_silence.PreparedAudioSource(
                "https://audio.invalid/tail",
                format_id="140",
                duration_sec=12.0,
            ),
        )
        acoustic = pipeline_module.clip_engine_silence.SilenceVerificationResult(
            "verified",
            0.0,
            10.2,
            {"threshold_dbfs": -38.0},
        )
        return (
            mock.patch.object(pipeline_module, "clip_engine_run"),
            mock.patch.object(pipeline_module, "clip_engine_meta"),
            mock.patch.object(
                pipeline_module.clip_engine_silence,
                "prepare_audio_source",
                return_value=prepared,
            ),
            mock.patch.object(
                pipeline_module.clip_engine_silence,
                "verify_acoustic_boundaries",
                return_value=acoustic,
            ),
        )

    def _assert_persisted_tail(self, reel_id: str) -> None:
        with db_module.get_conn() as conn:
            row = db_module.fetch_one(
                conn,
                "SELECT t_start, t_end FROM reels WHERE id = ?",
                (reel_id,),
            )
        self.assertIsNotNone(row)
        self.assertAlmostEqual(float(row["t_start"]), 0.0)
        self.assertAlmostEqual(float(row["t_end"]), 10.2)

    def test_url_adapter_persists_acoustic_tail_within_prepared_media(self) -> None:
        run_patch, meta_patch, prepare_patch, verify_patch = self._patch_media_tail()
        with (
            run_patch as mock_run,
            meta_patch as mock_meta,
            prepare_patch,
            verify_patch as mock_verify,
        ):
            mock_meta.extract_video_id.return_value = VIDEO_ID
            mock_run.clip.return_value = _media_tail_engine_out()

            result = main_module.ingestion_pipeline.ingest_url(
                source_url=SOURCE_URL,
                language="en",
            )

        self.assertAlmostEqual(result.reel.t_end, 10.2)
        self.assertAlmostEqual(result.metadata.duration_sec, 12.0)
        self.assertAlmostEqual(
            mock_verify.call_args.kwargs["search_end_limit_sec"], 12.0
        )
        self._assert_persisted_tail(result.reel.reel_id)

    def test_topic_cut_adapter_persists_acoustic_tail_within_prepared_media(self) -> None:
        run_patch, meta_patch, prepare_patch, verify_patch = self._patch_media_tail()
        with (
            run_patch as mock_run,
            meta_patch as mock_meta,
            prepare_patch,
            verify_patch as mock_verify,
        ):
            mock_meta.extract_video_id.return_value = VIDEO_ID
            mock_run.clip.return_value = _media_tail_engine_out()

            result = main_module.ingestion_pipeline.ingest_topic_cut(
                source_url=SOURCE_URL,
                query="photosynthesis",
                language="en",
            )

        self.assertEqual(result.reel_count, 1)
        self.assertAlmostEqual(result.reels[0].t_end, 10.2)
        self.assertAlmostEqual(result.duration_sec, 12.0)
        self.assertAlmostEqual(result.metadata.duration_sec, 12.0)
        self.assertAlmostEqual(
            mock_verify.call_args.kwargs["search_end_limit_sec"], 12.0
        )
        self._assert_persisted_tail(result.reels[0].reel_id)

    def test_direct_adapter_rejects_acoustic_crossing_next_unselected_cue(self) -> None:
        engine_out = _media_tail_engine_out()
        engine_out["transcript"]["segments"].append(
            {
                "cue_id": "next-cue",
                "start": 10.0,
                "end": 20.0,
                "text": "A sponsor and unrelated lesson begin here.",
            }
        )
        engine_out["transcript"]["duration"] = 20.0
        prepared = pipeline_module.clip_engine_silence.AudioPreparationResult(
            "ready",
            source=pipeline_module.clip_engine_silence.PreparedAudioSource(
                "https://audio.invalid/crossing",
                format_id="140",
                duration_sec=20.0,
            ),
        )
        verify = mock.Mock(
            return_value=pipeline_module.clip_engine_silence.SilenceVerificationResult(
                "verified",
                0.0,
                12.0,
                {"threshold_dbfs": -38.0},
            )
        )

        with (
            mock.patch.object(
                pipeline_module.clip_engine_silence,
                "prepare_audio_source",
                return_value=prepared,
            ),
            mock.patch.object(
                pipeline_module.clip_engine_silence,
                "verify_acoustic_boundaries",
                verify,
            ),
        ):
            clips = pipeline_module._verified_direct_adapter_clips(
                source_url=SOURCE_URL,
                engine_out=engine_out,
                should_cancel=None,
            )

        self.assertEqual(clips, [])
        self.assertEqual(verify.call_args.kwargs["search_end_limit_sec"], 10.0)


if __name__ == "__main__":
    unittest.main()
