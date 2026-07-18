import hashlib
import os
import sys
import tempfile
import unittest
import uuid
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
        db_module.init_db()
        account_id = str(uuid.uuid4())
        session_token = f"duration-test-session-{uuid.uuid4().hex}"
        timestamp = db_module.now_iso()
        with db_module.get_conn(transactional=True) as conn:
            db_module.insert(conn, "community_accounts", {
                "id": account_id,
                "username": f"duration-{account_id[:8]}",
                "username_normalized": f"duration-{account_id[:8]}",
                "email": f"{account_id[:8]}@example.com",
                "email_normalized": f"{account_id[:8]}@example.com",
                "password_hash": "hash",
                "password_salt": "salt",
                "verified_at": timestamp,
                "created_at": timestamp,
                "updated_at": timestamp,
            })
            db_module.insert(conn, "community_sessions", {
                "id": str(uuid.uuid4()),
                "account_id": account_id,
                "token_hash": hashlib.sha256(session_token.encode("utf-8")).hexdigest(),
                "created_at": timestamp,
                "last_used_at": timestamp,
                "expires_at": "2099-01-01T00:00:00+00:00",
            })
        self.verified_headers = {main_module.COMMUNITY_SESSION_HEADER: session_token}

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
                headers=self.verified_headers,
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
                headers=self.verified_headers,
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {"duration_sec": 27.0})

    def test_youtube_duration_uses_supadata_only(self) -> None:
        artifact = mock.Mock(duration_sec=321.0)
        with (
            mock.patch.object(
                main_module,
                "fetch_transcript_artifact",
                return_value=artifact,
            ) as fetch_transcript,
            mock.patch.object(
                main_module.requests,
                "get",
                side_effect=AssertionError("YouTube must not be queried directly"),
            ),
        ):
            response = self.client.get(
                "/api/community/reels/duration",
                params={"source_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
                headers=self.verified_headers,
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {"duration_sec": 321.0})
        fetch_transcript.assert_called_once()


if __name__ == "__main__":
    unittest.main()
