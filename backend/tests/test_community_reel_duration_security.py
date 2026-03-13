import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from backend.app import db as db_module
from backend.app.config import get_settings
import backend.app.main as main_module
from backend.app.main import app


class CommunityReelDurationSecurityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.previous_data_dir = os.environ.get("DATA_DIR")
        os.environ["DATA_DIR"] = self.temp_dir.name
        self.addCleanup(self._restore_environment)
        db_module._db_ready = False
        get_settings.cache_clear()
        main_module.settings = get_settings()
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

    def test_duration_endpoint_rejects_unsupported_source_host(self) -> None:
        response = self.client.get(
            "/api/community/reels/duration",
            params={"source_url": "https://example.com/video"},
        )

        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("youtube", response.json()["detail"].lower())

    def test_duration_fetch_does_not_follow_redirects_to_unsupported_hosts(self) -> None:
        redirect_response = mock.Mock()
        redirect_response.status_code = 302
        redirect_response.headers = {"Location": "https://example.com/private-hop"}

        with mock.patch.object(main_module.requests, "get", return_value=redirect_response) as mocked_get:
            response = self.client.get(
                "/api/community/reels/duration",
                params={"source_url": "https://www.instagram.com/reel/test-id/"},
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {"duration_sec": None})
        self.assertEqual(mocked_get.call_count, 1)

    def test_duration_endpoint_allows_supported_hosts_and_extracts_duration(self) -> None:
        ok_response = mock.Mock()
        ok_response.status_code = 200
        ok_response.headers = {}
        ok_response.text = '<meta property="video:duration" content="27">'
        ok_response.raise_for_status.return_value = None

        with mock.patch.object(main_module.requests, "get", return_value=ok_response):
            response = self.client.get(
                "/api/community/reels/duration",
                params={"source_url": "https://www.instagram.com/reel/test-id/"},
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {"duration_sec": 27.0})


if __name__ == "__main__":
    unittest.main()
