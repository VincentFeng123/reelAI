"""Regression coverage for quiet handoffs beyond the final caption timestamp."""

from __future__ import annotations

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
            {
                "threshold_dbfs": -38.0,
                "speech_handoff_verified": True,
                "end_speech_handoff_verified": True,
                "end_two_sided_required": False,
                "semantic_start_limit_sec": 0.0,
                "semantic_end_limit_sec": 12.0,
                "observation_start_limit_sec": 0.0,
                "observation_end_limit_sec": 11.0,
                "handoff_timestamp_tolerance_sec": 0.05,
                "start_quiet": [0.0, 0.2],
                "end_quiet": [10.0, 10.3],
            },
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

    def test_direct_selector_overlaps_audio_preparation(self) -> None:
        audio_started = threading.Event()
        selector_released_audio = threading.Event()
        prepared = pipeline_module.clip_engine_silence.AudioPreparationResult(
            "ready",
            source=pipeline_module.clip_engine_silence.PreparedAudioSource(
                "https://audio.invalid/parallel",
            ),
        )

        def prepare(*_args, **_kwargs):
            audio_started.set()
            self.assertTrue(selector_released_audio.wait(timeout=1.0))
            return prepared

        def select(*_args, **_kwargs):
            self.assertTrue(audio_started.wait(timeout=1.0))
            selector_released_audio.set()
            return _media_tail_engine_out()

        with (
            mock.patch.object(
                pipeline_module.clip_engine_silence,
                "prepare_audio_source",
                side_effect=prepare,
            ),
            mock.patch.object(pipeline_module, "_run_clip", side_effect=select),
        ):
            engine_out, result = pipeline_module._run_direct_clip_with_audio_prefetch(
                SOURCE_URL,
                topic="photosynthesis",
                language="en",
                should_cancel=None,
                generation_context=pipeline_module.GenerationContext("fast"),
            )

        self.assertEqual(engine_out["video_id"], VIDEO_ID)
        self.assertIs(result, prepared)

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
        self.assertFalse(mock_verify.call_args.kwargs["require_speech_handoff"])
        self.assertTrue(
            mock_verify.call_args.kwargs["require_end_speech_handoff"]
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
        self.assertFalse(mock_verify.call_args.kwargs["require_speech_handoff"])
        self.assertTrue(
            mock_verify.call_args.kwargs["require_end_speech_handoff"]
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

    def test_direct_adapter_requires_each_quality_score_at_green_threshold(self) -> None:
        prepared = pipeline_module.clip_engine_silence.AudioPreparationResult(
            "ready",
            source=pipeline_module.clip_engine_silence.PreparedAudioSource(
                "https://audio.invalid/quality",
                duration_sec=10.0,
            ),
        )
        acoustic = pipeline_module.clip_engine_silence.SilenceVerificationResult(
            "verified",
            0.0,
            10.0,
            {
                "semantic_start_limit_sec": 0.0,
                "semantic_end_limit_sec": 10.0,
            },
        )
        for field in (
            "informativeness",
            "topic_relevance",
            "educational_importance",
        ):
            with self.subTest(field=field):
                engine_out = _media_tail_engine_out()
                engine_out["clips"][0][field] = 0.74
                with (
                    mock.patch.object(
                        pipeline_module.clip_engine_silence,
                        "prepare_audio_source",
                        return_value=prepared,
                    ),
                    mock.patch.object(
                        pipeline_module.clip_engine_silence,
                        "verify_acoustic_boundaries",
                        return_value=acoustic,
                    ) as verify,
                ):
                    clips = pipeline_module._verified_direct_adapter_clips(
                        source_url=SOURCE_URL,
                        engine_out=engine_out,
                        should_cancel=None,
                    )

                self.assertEqual(clips, [])
                verify.assert_not_called()

        engine_out = _media_tail_engine_out()
        for field in (
            "informativeness",
            "topic_relevance",
            "educational_importance",
        ):
            engine_out["clips"][0][field] = 0.75
        with (
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
        ):
            clips = pipeline_module._verified_direct_adapter_clips(
                source_url=SOURCE_URL,
                engine_out=engine_out,
                should_cancel=None,
            )

        self.assertEqual(len(clips), 1)

    def test_direct_topic_gate_rejects_failed_hard_gemini_contract(self) -> None:
        engine_out = _media_tail_engine_out()
        unrelated = (
            "An operating system schedules processes, manages virtual memory, "
            "and provides file system abstractions."
        )
        engine_out["transcript"]["segments"][0]["text"] = unrelated
        engine_out["clips"][0]["topic_evidence_quote"] = unrelated
        engine_out["clips"][0]["directly_teaches_topic"] = False

        with (
            mock.patch.object(
                pipeline_module.clip_engine_silence,
                "prepare_audio_source",
            ) as prepare,
            mock.patch.object(
                pipeline_module.clip_engine_silence,
                "verify_acoustic_boundaries",
            ) as verify,
        ):
            clips = pipeline_module._verified_direct_adapter_clips(
                source_url=SOURCE_URL,
                engine_out=engine_out,
                should_cancel=None,
                exact_topic="biology",
            )

        self.assertEqual(clips, [])
        prepare.assert_not_called()
        verify.assert_not_called()

    def test_direct_topic_gate_accepts_paraphrases_without_semantic_model(self) -> None:
        first_quote = (
            "Giving up the next best alternative is the real economic sacrifice "
            "made whenever a person chooses."
        )
        second_quote = (
            "Past expenditures that cannot be recovered should not change the "
            "choice between future alternatives."
        )
        engine_out = _media_tail_engine_out()
        engine_out["transcript"]["segments"] = [
            {"cue_id": "choice", "start": 0.0, "end": 10.0, "text": first_quote},
            {"cue_id": "past", "start": 11.0, "end": 20.0, "text": second_quote},
        ]
        engine_out["transcript"]["duration"] = 20.0
        first_clip = engine_out["clips"][0]
        first_clip["cue_ids"] = ["choice"]
        first_clip["topic_evidence_quote"] = first_quote
        second_clip = {
            **first_clip,
            "start": 11.0,
            "end": 20.0,
            "cue_ids": ["past"],
            "topic_evidence_quote": second_quote,
            "selection_candidate_id": "past-candidate",
            "sequence_index": 1,
        }
        engine_out["clips"] = [first_clip, second_clip]
        prepared = pipeline_module.clip_engine_silence.AudioPreparationResult(
            "ready",
            source=pipeline_module.clip_engine_silence.PreparedAudioSource(
                "https://audio.invalid/comparison",
                duration_sec=20.0,
            ),
        )

        def verify(_source_url, start_sec, end_sec, **kwargs):
            tolerance = pipeline_module.clip_engine_silence.HANDOFF_TIMESTAMP_TOLERANCE_SEC
            return pipeline_module.clip_engine_silence.SilenceVerificationResult(
                "verified",
                start_sec,
                end_sec,
                {
                    "semantic_start_limit_sec": kwargs["search_start_limit_sec"],
                    "semantic_end_limit_sec": kwargs["search_end_limit_sec"],
                    "start_speech_handoff_verified": True,
                    "end_speech_handoff_verified": True,
                    "start_two_sided_required": kwargs["require_start_two_sided"],
                    "end_two_sided_required": kwargs["require_end_two_sided"],
                    "start_quiet": [start_sec - tolerance, start_sec + tolerance],
                    "end_quiet": [end_sec - tolerance, end_sec + tolerance],
                },
            )

        with mock.patch.object(
            pipeline_module.clip_engine_silence,
            "verify_acoustic_boundaries",
            side_effect=verify,
        ):
            clips = pipeline_module._verified_direct_adapter_clips(
                source_url=SOURCE_URL,
                engine_out=engine_out,
                should_cancel=None,
                prepared_audio=prepared,
                exact_topic="opportunity cost versus sunk cost",
            )

        self.assertEqual(len(clips), 2)

    def test_direct_topic_gate_accepts_explicit_comparison_component_lexically(self) -> None:
        engine_out = _media_tail_engine_out()
        quote = (
            "Opportunity cost is the value of the next best alternative given up "
            "when making a choice."
        )
        engine_out["transcript"]["segments"][0]["text"] = quote
        engine_out["clips"][0]["topic_evidence_quote"] = quote
        prepared = pipeline_module.clip_engine_silence.AudioPreparationResult(
            "ready",
            source=pipeline_module.clip_engine_silence.PreparedAudioSource(
                "https://audio.invalid/comparison-lexical",
                duration_sec=10.0,
            ),
        )
        acoustic = pipeline_module.clip_engine_silence.SilenceVerificationResult(
            "verified", 0.0, 10.0, {}
        )

        with mock.patch.object(
            pipeline_module.clip_engine_silence,
            "verify_acoustic_boundaries",
            return_value=acoustic,
        ):
            clips = pipeline_module._verified_direct_adapter_clips(
                source_url=SOURCE_URL,
                engine_out=engine_out,
                should_cancel=None,
                prepared_audio=prepared,
                exact_topic="opportunity cost versus sunk cost",
            )

        self.assertEqual(len(clips), 1)


if __name__ == "__main__":
    unittest.main()
