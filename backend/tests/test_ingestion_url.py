"""
Tests for the reel ingestion pipeline (`backend/app/ingestion/`).

Retired tests (7 skipped yt-dlp/Whisper tests + 2 adapter unit tests +
test_unsupported_host_returns_400) were removed in Phase 4 Wave A.
Remaining coverage lives in test_clip_engine_*.py.
"""

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

from fastapi.testclient import TestClient  # noqa: E402

from backend.app import db as db_module  # noqa: E402
from backend.app.config import get_settings  # noqa: E402
import backend.app.main as main_module  # noqa: E402
from backend.app.main import app  # noqa: E402


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


class _FakeYoutubeService:
    """Drop-in stand-in for YouTubeService that returns canned transcript cues."""

    def __init__(self, cues: list[dict] | None = None) -> None:
        self._cues = cues or [
            {"start": 0.0, "duration": 5.0, "text": "Welcome to the neural network chapter."},
            {"start": 5.0, "duration": 5.0, "text": "Neurons are simple functions in a layered graph."},
            {"start": 10.0, "duration": 5.0, "text": "Training adjusts the weights via gradient descent."},
            {"start": 15.0, "duration": 5.0, "text": "Loss functions quantify how wrong the prediction is."},
            {"start": 20.0, "duration": 5.0, "text": "Backpropagation pushes error gradients backwards."},
            {"start": 25.0, "duration": 5.0, "text": "Chain rule gives a local update for every parameter."},
            {"start": 30.0, "duration": 5.0, "text": "Repeat for many batches and the network converges."},
            {"start": 35.0, "duration": 5.0, "text": "Regularization keeps the model from memorizing."},
            {"start": 40.0, "duration": 5.0, "text": "Validation splits measure generalization."},
            {"start": 45.0, "duration": 5.0, "text": "Hyperparameter search tunes the learning rate."},
            {"start": 50.0, "duration": 5.0, "text": "Deeper networks can learn more abstract patterns."},
            {"start": 55.0, "duration": 5.0, "text": "We'll see this applied to handwritten digits next."},
        ]

    def get_transcript(self, conn, video_id: str) -> list[dict]:  # noqa: ARG002
        return self._cues


# --------------------------------------------------------------------- #
# TestCase
# --------------------------------------------------------------------- #


class IngestionUrlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.previous_data_dir = os.environ.get("DATA_DIR")
        os.environ["DATA_DIR"] = self.temp_dir.name
        self.addCleanup(self._restore_environment)
        db_module._db_ready = False
        get_settings.cache_clear()
        main_module.settings = get_settings()

        # Reset the in-memory orphan sweeper to avoid touching /tmp during tests.
        os.environ["REELAI_INGEST_SKIP_IMPORT_SWEEP"] = "1"

        # Stub the YouTube service on the module-level pipeline so transcript strategy 1
        # returns canned cues without hitting the network.
        self._fake_yt = _FakeYoutubeService()
        main_module.ingestion_pipeline._youtube_service = self._fake_yt

        # Ensure Whisper isn't invoked by default — we patch it at a per-test granularity.
        self._original_openai_client = main_module.ingestion_pipeline._openai_client
        main_module.ingestion_pipeline._openai_client = None

        # Force the pipeline out of "serverless" mode so the preflight check passes.
        self._original_serverless_mode = main_module.SERVERLESS_MODE
        main_module.SERVERLESS_MODE = False
        main_module.ingestion_pipeline._serverless_mode = False

        # Clear the per-IP rate-limit buckets so tests can make multiple requests
        # back-to-back without bumping against the endpoint's own sliding-window limiter.
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

    # --------------------------------------------------------------------- #
    # Serverless gating: endpoint refuses with 503 in SERVERLESS_MODE
    # --------------------------------------------------------------------- #

    def test_serverless_gate_returns_503(self) -> None:
        previous_serverless = main_module.SERVERLESS_MODE
        main_module.SERVERLESS_MODE = True
        previous_override = os.environ.pop("ALLOW_OPENAI_IN_SERVERLESS", None)
        self.addCleanup(lambda: setattr(main_module, "SERVERLESS_MODE", previous_serverless))
        if previous_override is not None:
            self.addCleanup(lambda: os.environ.__setitem__("ALLOW_OPENAI_IN_SERVERLESS", previous_override))

        response = self.client.post(
            "/api/ingest/url",
            json={"source_url": "https://www.youtube.com/watch?v=aircAruvnKk"},
        )
        self.assertEqual(response.status_code, 503, response.text)
        body = response.json()
        self.assertIn("ServerlessUnavailable", str(body))

    # --------------------------------------------------------------------- #
    # Rate limiter — per-platform cap raises 429 after the budget is exhausted
    # --------------------------------------------------------------------- #

    def test_platform_rate_limiter_triggers_429(self) -> None:
        from backend.app.ingestion.pipeline import _PlatformRateLimiter
        from backend.app.ingestion.errors import RateLimitedError

        limiter = _PlatformRateLimiter(overrides={"yt": (1, 60.0)})
        limiter.acquire("yt")
        with self.assertRaises(RateLimitedError):
            limiter.acquire("yt")

    # --------------------------------------------------------------------- #
    # Topic search — same query reuses the same material_id
    # --------------------------------------------------------------------- #

    def _install_high_rate_limits(self) -> None:
        """Search tests can make multiple calls; bump per-platform limits to avoid noise."""
        from backend.app.ingestion.pipeline import _PlatformRateLimiter

        main_module.ingestion_pipeline._rate_limiter = _PlatformRateLimiter(
            overrides={"yt": (1000, 60.0), "ig": (1000, 60.0), "tt": (1000, 60.0)}
        )

    def test_ingest_search_same_query_reuses_material_id(self) -> None:
        """Running the same search twice must land both batches under the same material_id."""
        self._install_high_rate_limits()

        with mock.patch(
            "backend.app.ingestion.pipeline.clip_engine_search.discover",
            return_value={"corrected": "linear algebra", "videos": [], "credits_used": 0, "warning": None},
        ):
            first = self.client.post(
                "/api/ingest/search",
                json={"query": "  Linear Algebra  ", "platforms": ["yt"], "max_per_platform": 1},
            )
            second = self.client.post(
                "/api/ingest/search",
                json={"query": "linear algebra", "platforms": ["yt"], "max_per_platform": 1},
            )

        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(second.status_code, 200, second.text)
        # Same normalized query → same material_id
        self.assertEqual(first.json()["material_id"], second.json()["material_id"])


# --------------------------------------------------------------------- #
# Unit tests for normalize_clip_window — bit-compatible with reels.py:9286
# --------------------------------------------------------------------- #


class NormalizeClipWindowTests(unittest.TestCase):
    def test_basic_window(self) -> None:
        from backend.app.ingestion.segment import normalize_clip_window

        self.assertEqual(normalize_clip_window(10, 55, 120), (10, 55))

    def test_clamps_to_min_length(self) -> None:
        from backend.app.ingestion.segment import normalize_clip_window

        self.assertEqual(normalize_clip_window(10, 14, 120, min_len=15, max_len=60), (10, 25))

    def test_clamps_to_max_length(self) -> None:
        from backend.app.ingestion.segment import normalize_clip_window

        self.assertEqual(normalize_clip_window(10, 200, 300, min_len=15, max_len=60), (10, 70))

    def test_rejects_when_video_shorter_than_min(self) -> None:
        from backend.app.ingestion.segment import normalize_clip_window

        self.assertIsNone(normalize_clip_window(0, 10, 8, min_len=15, max_len=60))

    def test_end_clamped_to_video_duration(self) -> None:
        from backend.app.ingestion.segment import normalize_clip_window

        # Clip runs off the end; end is clamped and the start may be pulled back
        self.assertEqual(normalize_clip_window(20, 90, 60, min_len=15, max_len=60), (20, 60))


if __name__ == "__main__":
    unittest.main()
