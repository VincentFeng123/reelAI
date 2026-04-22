from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.ingestion.models import IngestTranscriptCue, IngestTranscriptWord  # noqa: E402
from backend.app.services.clip_boundary import SnapResult  # noqa: E402
from backend.app.services.clip_whisper_refine import WhisperRefinement, WhisperWord  # noqa: E402
from backend.app.services.topic_cut import TopicReel, _apply_boundary_engine  # noqa: E402


def _cue(text: str, words: list[tuple[str, float, float]]) -> IngestTranscriptCue:
    return IngestTranscriptCue(
        start=words[0][1],
        end=words[-1][2],
        text=text,
        words=[
            IngestTranscriptWord(start=start, end=end, text=token)
            for token, start, end in words
        ],
        word_source="proportional",
    )


def _sample_ingest_cues() -> list[IngestTranscriptCue]:
    return [
        _cue(
            "Hey everyone.",
            [
                ("Hey", 0.00, 0.18),
                ("everyone", 0.22, 0.72),
            ],
        ),
        _cue(
            "Gradient descent updates the weights.",
            [
                ("Gradient", 1.52, 1.92),
                ("descent", 1.95, 2.30),
                ("updates", 2.36, 2.72),
                ("the", 2.75, 2.85),
                ("weights", 2.90, 3.28),
            ],
        ),
        _cue(
            "Backpropagation computes gradients.",
            [
                ("Backpropagation", 4.82, 5.46),
                ("computes", 5.50, 5.92),
                ("gradients", 5.96, 6.42),
            ],
        ),
    ]


class TopicCutBoundaryEnforceTests(unittest.TestCase):
    def test_sentence_edge_gate_uses_snap_helper(self) -> None:
        reels = [
            TopicReel(
                video_id="vid-1",
                t_start=1.70,
                t_end=4.40,
                label="Gradient descent",
            )
        ]
        cues = _sample_ingest_cues()
        snapped = SnapResult(
            snapped=True,
            t_start=1.63,
            t_end=4.47,
            start_sentence=None,
            end_sentence=None,
            reason="",
        )

        with mock.patch.dict(os.environ, {"CLIP_SENTENCE_EDGE_ENFORCE": "1"}, clear=False):
            with mock.patch(
                "backend.app.services.clip_boundary.snap_llm_boundary",
                return_value=snapped,
            ) as snap_mock:
                out = _apply_boundary_engine(
                    reels,
                    ingest_cues=cues,
                    chapters=None,
                    silence_ranges=None,
                    llm_topic_segments_raw=None,
                    query=None,
                    user_min_sec=2.0,
                    user_max_sec=5.0,
                    user_target_sec=3.0,
                    video_duration_sec=10.0,
                )

        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].t_start, 1.63)
        self.assertEqual(out[0].t_end, 4.47)
        self.assertEqual(snap_mock.call_count, 1)

    def test_engine_fallback_reels_get_whisper_refinement(self) -> None:
        reels = [
            TopicReel(
                video_id="vid-1",
                t_start=1.70,
                t_end=4.40,
                label="Gradient descent",
            )
        ]
        cues = _sample_ingest_cues()
        refined = WhisperRefinement(
            t_start=1.61,
            t_end=3.71,
            words=[
                WhisperWord(text="Gradient", start=1.64, end=1.90),
                WhisperWord(text="weights.", start=3.32, end=3.66),
            ],
            first_word="Gradient",
            last_word="weights.",
        )

        with mock.patch.dict(os.environ, {"CLIP_SENTENCE_EDGE_ENFORCE": "0"}, clear=False):
            with mock.patch(
                "backend.app.services.clip_whisper_refine.whisper_clip_refine_enabled",
                return_value=True,
            ), mock.patch(
                "backend.app.services.clip_whisper_refine.clip_audio_refine_conditional",
                return_value=False,
            ), mock.patch(
                "backend.app.services.clip_whisper_refine.refine_clip_with_whisper",
                return_value=refined,
            ) as refine_mock:
                out = _apply_boundary_engine(
                    reels,
                    ingest_cues=cues,
                    chapters=None,
                    silence_ranges=None,
                    llm_topic_segments_raw=None,
                    query=None,
                    user_min_sec=2.0,
                    user_max_sec=5.0,
                    user_target_sec=3.0,
                    video_duration_sec=10.0,
                )

        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].t_start, 1.59)
        self.assertEqual(out[0].t_end, 3.71)
        self.assertEqual(out[0].boundary_quality, "whisper-word")
        self.assertEqual(refine_mock.call_count, 1)

    def test_whisper_refine_drops_clip_without_terminal_close(self) -> None:
        reels = [
            TopicReel(
                video_id="vid-1",
                t_start=1.70,
                t_end=4.40,
                label="Gradient descent",
            )
        ]
        cues = _sample_ingest_cues()
        refined = WhisperRefinement(
            t_start=1.61,
            t_end=4.92,
            words=[
                WhisperWord(text="Gradient", start=1.64, end=1.90),
                WhisperWord(text="weights", start=4.40, end=4.87),
            ],
            first_word="Gradient",
            last_word="weights",
        )

        with mock.patch.dict(os.environ, {"CLIP_SENTENCE_EDGE_ENFORCE": "0"}, clear=False):
            with mock.patch(
                "backend.app.services.clip_whisper_refine.whisper_clip_refine_enabled",
                return_value=True,
            ), mock.patch(
                "backend.app.services.clip_whisper_refine.clip_audio_refine_conditional",
                return_value=False,
            ), mock.patch(
                "backend.app.services.clip_whisper_refine.refine_clip_with_whisper",
                return_value=refined,
            ):
                out = _apply_boundary_engine(
                    reels,
                    ingest_cues=cues,
                    chapters=None,
                    silence_ranges=None,
                    llm_topic_segments_raw=None,
                    query=None,
                    user_min_sec=2.0,
                    user_max_sec=5.0,
                    user_target_sec=3.0,
                    video_duration_sec=10.0,
                )

        self.assertEqual(out, [])


if __name__ == "__main__":
    unittest.main()
