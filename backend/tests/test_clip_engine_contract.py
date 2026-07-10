"""
HTTP-layer contract smoke test for POST /api/ingest/url (Task 13).

Asserts that the endpoint returns the exact ReelOut JSON shape the iOS app
decodes: top-level reel/metadata/terms_notice/trace_id, and reel has
reel_id/video_url/t_start/t_end/captions/video_duration_sec.

Strategy: mock the two external surfaces (clip_engine_run.clip and
clip_engine_meta.youtube_metadata) so the test is fully offline and fast.
Uses the same temp-DB + settings-cache-clear harness as
test_clip_engine_ingest_url.py.
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
from backend.app.ingestion import pipeline as pipeline_module  # noqa: E402


def _fake_engine_out(video_id: str = "dQw4w9WgXcQ") -> dict:
    return {
        "video_id": video_id,
        "clips": [
            {
                "start": 30.0,
                "end": 75.0,
                "cut_end": 75.0,
                "title": "The chorus",
                "facet": "music",
                "reason": "catchy hook",
                "sequence_index": 0,
                "embed_url": f"https://www.youtube.com/embed/{video_id}?start=30&end=75",
            }
        ],
        "transcript": {
            "segments": [
                {"start": 30.0, "end": 55.0, "text": "Never gonna give you up."},
                {"start": 55.0, "end": 75.0, "text": "Never gonna let you down."},
            ],
            "words": [],
            "duration": 300.0,
        },
        "notes": "test notes",
    }


class ClipEngineContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

        self.previous_data_dir = os.environ.get("DATA_DIR")
        os.environ["DATA_DIR"] = self.temp_dir.name
        # Bypass the serverless guard that returns 503 when VERCEL is set.
        os.environ["ALLOW_OPENAI_IN_SERVERLESS"] = "1"
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

        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def _restore_environment(self) -> None:
        if self.previous_data_dir is None:
            os.environ.pop("DATA_DIR", None)
        else:
            os.environ["DATA_DIR"] = self.previous_data_dir
        os.environ.pop("ALLOW_OPENAI_IN_SERVERLESS", None)
        db_module._db_ready = False
        get_settings.cache_clear()
        main_module.settings = get_settings()

    # ------------------------------------------------------------------ #
    # Contract: POST /api/ingest/url → ReelOut JSON shape
    # ------------------------------------------------------------------ #

    def test_ingest_url_response_shape(self) -> None:
        source_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

        with (
            mock.patch.object(pipeline_module, "clip_engine_run") as mock_run,
            mock.patch.object(pipeline_module, "clip_engine_meta") as mock_meta,
        ):
            mock_meta.extract_video_id.return_value = "dQw4w9WgXcQ"
            mock_meta.youtube_metadata.return_value = {
                "title": "V",
                "duration_sec": 300.0,
            }
            mock_run.clip.return_value = _fake_engine_out()

            response = self.client.post(
                "/api/ingest/url",
                json={"source_url": source_url, "material_id": "m1"},
            )

        # --- status ---
        self.assertEqual(response.status_code, 200, response.text)

        body = response.json()

        # --- top-level envelope ---
        for key in ("reel", "metadata", "terms_notice", "trace_id"):
            self.assertIn(key, body, f"missing top-level key: {key}")

        # --- reel shape ---
        reel = body["reel"]
        for key in ("reel_id", "video_url", "t_start", "t_end", "captions", "video_duration_sec"):
            self.assertIn(key, reel, f"missing reel key: {key}")

        # --- embed URL contract ---
        self.assertTrue(
            reel["video_url"].startswith("https://www.youtube.com/embed/dQw4w9WgXcQ"),
            f"Unexpected video_url: {reel['video_url']}",
        )


if __name__ == "__main__":
    unittest.main()
