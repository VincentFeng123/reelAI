"""
Integration tests for `POST /api/ingest/topic-cut` and the corresponding
`IngestionPipeline.ingest_topic_cut` method.

All 5 yt-dlp/Whisper-path tests were retired in Phase 4 Wave A; live coverage
lives in test_clip_engine_*.py.  setUp is preserved so test_ingestion_url.py's
_FakeYoutubeService import keeps working.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("REELAI_INGEST_SKIP_IMPORT_SWEEP", "1")

from fastapi.testclient import TestClient  # noqa: E402

from backend.app import db as db_module  # noqa: E402
from backend.app.config import get_settings  # noqa: E402
import backend.app.main as main_module  # noqa: E402
from backend.app.main import app  # noqa: E402

from backend.tests.test_ingestion_url import _FakeYoutubeService  # noqa: E402


def _make_long_transcript_cues() -> list[dict]:
    """
    20 cues × 30 seconds = 10 minutes of fake speech, two clearly distinct topics.
    Cue 0-9 talks about pasta; cue 10-19 talks about JavaScript debugging.
    """
    pasta = [
        ("Today we cook spaghetti carbonara from scratch", 0.0),
        ("First grate the pecorino romano cheese into a bowl", 30.0),
        ("Crack three large eggs into the same mixing bowl", 60.0),
        ("Whisk the eggs and pecorino until fully combined", 90.0),
        ("Boil salted water for the dried spaghetti pasta", 120.0),
        ("Cook guanciale slowly in a wide skillet to render fat", 150.0),
        ("Drop the spaghetti into rapidly boiling salted water", 180.0),
        ("Reserve a full cup of starchy pasta cooking water", 210.0),
        ("Toss drained spaghetti directly into the guanciale skillet off heat", 240.0),
        ("Plate immediately with extra cracked pepper and pecorino on top", 270.0),
    ]
    js = [
        ("Now lets switch gears and debug a tricky JavaScript bug", 300.0),
        ("Open Chrome DevTools with command option I on macOS", 330.0),
        ("Set a breakpoint on line forty-two of the source file", 360.0),
        ("Inspect the call stack panel to trace function entry order", 390.0),
        ("Watch each variable update in the local scope panel live", 420.0),
        ("Use the network tab to inspect failed XHR requests for clues", 450.0),
        ("Add a conditional breakpoint to skip noisy iteration loops", 480.0),
        ("Memory snapshots reveal retained DOM nodes and closures", 510.0),
        ("React DevTools highlight unnecessary component re-renders", 540.0),
        ("That covers the essentials of in-browser JavaScript debugging", 570.0),
    ]
    return [
        {"start": start, "duration": 30.0, "text": text}
        for text, start in (pasta + js)
    ]


class IngestionTopicCutTests(unittest.TestCase):
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

        self._fake_yt = _FakeYoutubeService(cues=_make_long_transcript_cues())
        main_module.ingestion_pipeline._youtube_service = self._fake_yt

        self._original_openai_client = main_module.ingestion_pipeline._openai_client
        # Default: pretend no OpenAI client (heuristic fallback). Per-test we
        # override with the mock when we want to exercise the LLM path.
        main_module.ingestion_pipeline._openai_client = None

        self._original_serverless_mode = main_module.SERVERLESS_MODE
        main_module.SERVERLESS_MODE = False
        main_module.ingestion_pipeline._serverless_mode = False
        main_module._rate_limit_hits.clear()

        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def _restore_environment(self) -> None:
        if self.previous_data_dir is None:
            os.environ.pop("DATA_DIR", None)
        else:
            os.environ["DATA_DIR"] = self.previous_data_dir
        db_module._db_ready = False
        get_settings.cache_clear()
        main_module.settings = get_settings()


if __name__ == "__main__":
    unittest.main()
