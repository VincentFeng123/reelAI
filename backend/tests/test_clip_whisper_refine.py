from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.services.clip_whisper_refine import WhisperWord, refine_clip_with_whisper  # noqa: E402


class ClipWhisperRefineTests(unittest.TestCase):
    def test_refine_includes_word_overlapping_left_boundary(self) -> None:
        words = [
            WhisperWord(text="it", start=10.52, end=11.02),
            WhisperWord(text="matters.", start=11.08, end=11.52),
        ]
        with mock.patch(
            "backend.app.services.clip_whisper_refine.whisper_clip_refine_enabled",
            return_value=True,
        ), mock.patch(
            "backend.app.services.clip_whisper_refine._read_cached_words",
            return_value=words,
        ):
            refined = refine_clip_with_whisper(
                video_id="vid-1",
                t_start=10.97,
                t_end=11.55,
                conn=object(),
            )

        self.assertIsNotNone(refined)
        assert refined is not None
        self.assertAlmostEqual(refined.t_start, 10.37, places=2)
        self.assertEqual(refined.first_word, "it")

    def test_refine_skips_filler_opener(self) -> None:
        words = [
            WhisperWord(text="So", start=10.52, end=10.70),
            WhisperWord(text="derivatives", start=10.74, end=11.22),
            WhisperWord(text="matter.", start=11.28, end=11.66),
        ]
        with mock.patch(
            "backend.app.services.clip_whisper_refine.whisper_clip_refine_enabled",
            return_value=True,
        ), mock.patch(
            "backend.app.services.clip_whisper_refine._read_cached_words",
            return_value=words,
        ):
            refined = refine_clip_with_whisper(
                video_id="vid-1",
                t_start=10.60,
                t_end=11.70,
                conn=object(),
            )

        self.assertIsNotNone(refined)
        assert refined is not None
        self.assertAlmostEqual(refined.t_start, 10.59, places=2)
        self.assertEqual(refined.first_word, "derivatives")

    def test_refine_prefers_terminal_close_over_trailing_fragment(self) -> None:
        words = [
            WhisperWord(text="gradient", start=10.50, end=10.92),
            WhisperWord(text="matters.", start=10.96, end=11.34),
            WhisperWord(text="This", start=11.80, end=12.02),
            WhisperWord(text="really", start=12.02, end=12.26),
        ]
        with mock.patch(
            "backend.app.services.clip_whisper_refine.whisper_clip_refine_enabled",
            return_value=True,
        ), mock.patch(
            "backend.app.services.clip_whisper_refine._read_cached_words",
            return_value=words,
        ):
            refined = refine_clip_with_whisper(
                video_id="vid-1",
                t_start=10.60,
                t_end=12.05,
                conn=object(),
            )

        self.assertIsNotNone(refined)
        assert refined is not None
        self.assertEqual(refined.last_word, "matters.")
        self.assertAlmostEqual(refined.t_end, 11.39, places=2)


if __name__ == "__main__":
    unittest.main()
