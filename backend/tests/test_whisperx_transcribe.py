"""
Phase 4 tests for `backend/app/ingestion/whisperx_transcribe.py`.

Covered:
  * Kill-switches: `WHISPERX_ENABLED=false` disables both sites;
    `WHISPERX_FALLBACK_ENABLED=false` leaves the module loaded but disables
    only site (a).
  * `_get_whisperx_module` version-pin warning when the installed version
    differs from the pin.
  * `whisperx_align` with a mocked whisperx module — asserts cue/word
    shape, absolute seconds preservation, language pass-through,
    `word_source="whisperx"` tag, and graceful None on segment-count
    mismatch.
  * `whisperx_words_for_audio` with mocked asr + align — asserts the word
    list is flat and seconds are relative to the clip (caller adds offset).
"""

from __future__ import annotations

import importlib
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("VERCEL", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.ingestion.models import IngestTranscriptCue  # noqa: E402


def _reload_whisperx_module():
    """Re-import so env-var flips take effect for a single test case."""
    from backend.app.ingestion import whisperx_transcribe
    return importlib.reload(whisperx_transcribe)


class KillSwitchTests(unittest.TestCase):
    def test_whisperx_disabled_short_circuits(self):
        with mock.patch.dict(os.environ, {"WHISPERX_ENABLED": "false"}, clear=False):
            mod = _reload_whisperx_module()
            self.assertFalse(mod.whisperx_enabled())
            self.assertFalse(mod.whisperx_fallback_enabled())
            cues = [IngestTranscriptCue(start=0.0, end=2.0, text="hello world")]
            self.assertIsNone(mod.whisperx_align(Path("/nonexistent.wav"), cues, language="en"))
            self.assertEqual(mod.whisperx_words_for_audio(Path("/nonexistent.wav"), language="en"), [])

    def test_fallback_flag_independent(self):
        with mock.patch.dict(os.environ, {
            "WHISPERX_ENABLED": "true",
            "WHISPERX_FALLBACK_ENABLED": "false",
        }, clear=False):
            mod = _reload_whisperx_module()
            self.assertTrue(mod.whisperx_enabled())
            self.assertFalse(mod.whisperx_fallback_enabled())

    def tearDown(self):
        # Restore sane defaults for other tests.
        for k in ("WHISPERX_ENABLED", "WHISPERX_FALLBACK_ENABLED"):
            os.environ.pop(k, None)
        _reload_whisperx_module()


class WhisperxAlignWithMockTests(unittest.TestCase):
    def setUp(self):
        os.environ["WHISPERX_ENABLED"] = "true"
        self.mod = _reload_whisperx_module()

    def tearDown(self):
        os.environ.pop("WHISPERX_ENABLED", None)
        self.mod._whisperx_module = None
        self.mod._align_model_cache.clear()
        self.mod._asr_model_cache.clear()
        _reload_whisperx_module()

    def _mock_whisperx(self, align_output):
        mock_whisperx = SimpleNamespace(
            __version__="3.1.1",
            load_align_model=mock.MagicMock(return_value=("model", {"lang": "en"})),
            load_audio=mock.MagicMock(return_value=b"audio-bytes"),
            align=mock.MagicMock(return_value=align_output),
            load_model=mock.MagicMock(),
        )
        return mock_whisperx

    def test_align_transplants_word_timings_and_tags_source(self):
        cues = [
            IngestTranscriptCue(start=0.0, end=3.0, text="hello world"),
            IngestTranscriptCue(start=3.0, end=6.0, text="this is a test"),
        ]
        align_output = {
            "segments": [
                {"words": [
                    {"start": 0.1, "end": 0.5, "word": "hello", "score": 0.95},
                    {"start": 0.6, "end": 1.1, "word": "world", "score": 0.9},
                ]},
                {"words": [
                    {"start": 3.2, "end": 3.4, "word": "this", "score": 0.85},
                    {"start": 3.5, "end": 3.7, "word": "is", "score": 0.88},
                    {"start": 3.8, "end": 4.0, "word": "a", "score": 0.7},
                    {"start": 4.1, "end": 4.6, "word": "test", "score": 0.93},
                ]},
            ],
        }
        mw = self._mock_whisperx(align_output)
        self.mod._whisperx_module = mw
        self.mod._align_model_cache["en"] = ("model", {"lang": "en"})

        aligned = self.mod.whisperx_align(Path("/fake.wav"), cues, language="en")
        self.assertIsNotNone(aligned)
        self.assertEqual(len(aligned), 2)
        self.assertTrue(all(c.word_source == "whisperx" for c in aligned))
        # Absolute seconds preserved — cue text still matches.
        self.assertEqual(aligned[0].text, "hello world")
        self.assertEqual(aligned[0].start, 0.0)
        self.assertEqual(aligned[0].end, 3.0)
        self.assertEqual(len(aligned[0].words), 2)
        self.assertAlmostEqual(aligned[0].words[0].start, 0.1)
        self.assertEqual(aligned[0].words[0].text, "hello")
        self.assertAlmostEqual(aligned[0].words[0].confidence, 0.95, places=3)
        self.assertEqual(len(aligned[1].words), 4)

    def test_align_returns_none_on_segment_count_mismatch(self):
        cues = [
            IngestTranscriptCue(start=0.0, end=3.0, text="a"),
            IngestTranscriptCue(start=3.0, end=6.0, text="b"),
        ]
        # Only one segment returned — mismatch.
        mw = self._mock_whisperx({"segments": [{"words": [{"start": 0.1, "end": 0.5, "word": "a"}]}]})
        self.mod._whisperx_module = mw
        self.mod._align_model_cache["en"] = ("model", {"lang": "en"})
        self.assertIsNone(self.mod.whisperx_align(Path("/fake.wav"), cues, language="en"))

    def test_empty_cues_returns_none(self):
        self.assertIsNone(self.mod.whisperx_align(Path("/fake.wav"), [], language="en"))

    def test_align_passes_language_code(self):
        cues = [IngestTranscriptCue(start=0.0, end=3.0, text="hola mundo")]
        align_output = {
            "segments": [
                {"words": [{"start": 0.1, "end": 1.0, "word": "hola"}]},
            ],
        }
        mw = self._mock_whisperx(align_output)
        self.mod._whisperx_module = mw
        aligned = self.mod.whisperx_align(Path("/fake.wav"), cues, language="es")
        mw.load_align_model.assert_called_with(language_code="es", device=self.mod._WHISPERX_DEVICE)
        self.assertIsNotNone(aligned)


class WhisperxWordsForAudioTests(unittest.TestCase):
    def setUp(self):
        os.environ["WHISPERX_ENABLED"] = "true"
        self.mod = _reload_whisperx_module()

    def tearDown(self):
        os.environ.pop("WHISPERX_ENABLED", None)
        self.mod._whisperx_module = None
        self.mod._align_model_cache.clear()
        self.mod._asr_model_cache.clear()
        _reload_whisperx_module()

    def test_returns_flat_word_list_relative_seconds(self):
        asr_model = SimpleNamespace(
            transcribe=mock.MagicMock(return_value={
                "segments": [{"start": 0.0, "end": 1.0, "text": "hello there"}],
            }),
        )
        aligned_result = {
            "segments": [{
                "words": [
                    {"start": 0.05, "end": 0.5, "word": "hello"},
                    {"start": 0.6, "end": 1.1, "word": "there"},
                ],
            }],
        }
        mw = SimpleNamespace(
            __version__="3.1.1",
            load_align_model=mock.MagicMock(return_value=("model", {"lang": "en"})),
            load_audio=mock.MagicMock(return_value=b"audio"),
            load_model=mock.MagicMock(return_value=asr_model),
            align=mock.MagicMock(return_value=aligned_result),
        )
        self.mod._whisperx_module = mw

        words = self.mod.whisperx_words_for_audio(Path("/fake.wav"), language="en")
        self.assertEqual(len(words), 2)
        self.assertEqual(words[0].text, "hello")
        self.assertAlmostEqual(words[0].start, 0.05)
        self.assertAlmostEqual(words[0].end, 0.5)
        self.assertEqual(words[1].text, "there")

    def test_returns_empty_on_asr_exception(self):
        asr_model = SimpleNamespace(
            transcribe=mock.MagicMock(side_effect=RuntimeError("boom")),
        )
        mw = SimpleNamespace(
            __version__="3.1.1",
            load_align_model=mock.MagicMock(return_value=("model", {"lang": "en"})),
            load_audio=mock.MagicMock(return_value=b"audio"),
            load_model=mock.MagicMock(return_value=asr_model),
            align=mock.MagicMock(),
        )
        self.mod._whisperx_module = mw
        self.assertEqual(self.mod.whisperx_words_for_audio(Path("/fake.wav"), language="en"), [])


class VersionPinTests(unittest.TestCase):
    def setUp(self):
        os.environ["WHISPERX_ENABLED"] = "true"
        self.mod = _reload_whisperx_module()

    def tearDown(self):
        os.environ.pop("WHISPERX_ENABLED", None)
        self.mod._whisperx_module = None
        _reload_whisperx_module()

    def test_version_mismatch_warns(self):
        # Fake a whisperx module at a non-pinned version.
        fake_module = SimpleNamespace(
            __version__="3.3.0",
            load_align_model=mock.MagicMock(),
            load_audio=mock.MagicMock(),
            align=mock.MagicMock(),
            load_model=mock.MagicMock(),
        )
        with mock.patch.dict(sys.modules, {"whisperx": fake_module}):
            self.mod._whisperx_module = None  # force re-probe
            with self.assertLogs("backend.app.ingestion.whisperx_transcribe", level="WARNING") as captured:
                resolved = self.mod._get_whisperx_module()
            self.assertIs(resolved, fake_module)
            joined = "\n".join(captured.output)
            self.assertIn("3.3.0", joined)
            self.assertIn("3.1.1", joined)


if __name__ == "__main__":
    unittest.main()
