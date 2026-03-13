import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_TEST_DATA_DIR = tempfile.TemporaryDirectory()
os.environ["DATA_DIR"] = _TEST_DATA_DIR.name

from fastapi.testclient import TestClient

from backend.app.config import get_settings
import backend.app.main as main_module

get_settings.cache_clear()

from backend.app.db import execute_modify, get_conn
from backend.app.main import COMMUNITY_OWNER_HEADER, COMMUNITY_SESSION_HEADER, app


class CommunitySettingsSyncTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_verification_required = os.environ.get("COMMUNITY_EMAIL_VERIFICATION_REQUIRED")
        os.environ["COMMUNITY_EMAIL_VERIFICATION_REQUIRED"] = "1"
        get_settings.cache_clear()
        main_module.settings = get_settings()
        self.client = TestClient(app)
        self.other_client = TestClient(app)
        self.third_client = TestClient(app)
        self.addCleanup(self.client.close)
        self.addCleanup(self.other_client.close)
        self.addCleanup(self.third_client.close)
        self.addCleanup(self._restore_environment)
        with get_conn(transactional=True) as conn:
            execute_modify(conn, "DELETE FROM community_account_settings")
            execute_modify(conn, "DELETE FROM community_sessions")
            execute_modify(conn, "DELETE FROM community_accounts")

    def _restore_environment(self) -> None:
        if self.previous_verification_required is None:
            os.environ.pop("COMMUNITY_EMAIL_VERIFICATION_REQUIRED", None)
        else:
            os.environ["COMMUNITY_EMAIL_VERIFICATION_REQUIRED"] = self.previous_verification_required
        get_settings.cache_clear()
        main_module.settings = get_settings()

    def _register_and_verify(self, *, client: TestClient, owner_key: str, username: str, password: str) -> str:
        register_response = client.post(
            "/api/community/auth/register",
            headers={COMMUNITY_OWNER_HEADER: owner_key},
            json={"username": username, "email": f"{username}@example.com", "password": password},
        )
        self.assertEqual(register_response.status_code, 201, register_response.text)
        register_json = register_response.json()
        verification_code = str(register_json["verification_code_debug"])
        session_token = str(register_json["session_token"])

        verify_response = client.post(
            "/api/community/auth/verify",
            headers={
                COMMUNITY_OWNER_HEADER: owner_key,
                COMMUNITY_SESSION_HEADER: session_token,
            },
            json={"code": verification_code},
        )
        self.assertEqual(verify_response.status_code, 200, verify_response.text)
        self.assertTrue(verify_response.json()["account"]["is_verified"])
        return session_token

    def test_new_account_settings_default_to_expected_values(self) -> None:
        session_token = self._register_and_verify(
            client=self.client,
            owner_key="settings-default-owner-key-abcdefghijklmnopqrstuvwxyz",
            username="settingsdefault",
            password="studyreels-password",
        )

        response = self.client.get(
            "/api/community/settings",
            headers={COMMUNITY_SESSION_HEADER: session_token},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            response.json(),
            {
                "generation_mode": "fast",
                "default_input_mode": "source",
                "min_relevance_threshold": 0.3,
                "start_muted": True,
                "video_pool_mode": "short-first",
                "preferred_video_duration": "any",
                "target_clip_duration_sec": 55,
                "target_clip_duration_min_sec": 20,
                "target_clip_duration_max_sec": 55,
            },
        )

    def test_settings_sync_across_devices_for_same_account(self) -> None:
        username = "settingssync"
        password = "studyreels-password"
        session_token = self._register_and_verify(
            client=self.client,
            owner_key="settings-sync-owner-key-abcdefghijklmnopqrstuvwxyz",
            username=username,
            password=password,
        )

        replace_response = self.client.put(
            "/api/community/settings",
            headers={COMMUNITY_SESSION_HEADER: session_token},
            json={
                "generation_mode": "slow",
                "default_input_mode": "topic",
                "min_relevance_threshold": 0.18,
                "start_muted": False,
                "video_pool_mode": "balanced",
                "preferred_video_duration": "medium",
                "target_clip_duration_sec": 72,
                "target_clip_duration_min_sec": 45,
                "target_clip_duration_max_sec": 95,
            },
        )
        self.assertEqual(replace_response.status_code, 200, replace_response.text)

        login_response = self.other_client.post(
            "/api/community/auth/login",
            json={"username": username, "password": password},
        )
        self.assertEqual(login_response.status_code, 200, login_response.text)
        other_session_token = str(login_response.json()["session_token"])

        other_settings_response = self.other_client.get(
            "/api/community/settings",
            headers={COMMUNITY_SESSION_HEADER: other_session_token},
        )
        self.assertEqual(other_settings_response.status_code, 200, other_settings_response.text)
        self.assertEqual(other_settings_response.json()["generation_mode"], "slow")
        self.assertEqual(other_settings_response.json()["default_input_mode"], "topic")
        self.assertEqual(other_settings_response.json()["min_relevance_threshold"], 0.18)
        self.assertFalse(other_settings_response.json()["start_muted"])

        overwrite_response = self.other_client.put(
            "/api/community/settings",
            headers={COMMUNITY_SESSION_HEADER: other_session_token},
            json={
                "generation_mode": "fast",
                "default_input_mode": "file",
                "min_relevance_threshold": 0.42,
                "start_muted": True,
                "video_pool_mode": "long-form",
                "preferred_video_duration": "long",
                "target_clip_duration_sec": 120,
                "target_clip_duration_min_sec": 80,
                "target_clip_duration_max_sec": 140,
            },
        )
        self.assertEqual(overwrite_response.status_code, 200, overwrite_response.text)

        refreshed_response = self.client.get(
            "/api/community/settings",
            headers={COMMUNITY_SESSION_HEADER: session_token},
        )
        self.assertEqual(refreshed_response.status_code, 200, refreshed_response.text)
        self.assertEqual(refreshed_response.json()["default_input_mode"], "file")
        self.assertEqual(refreshed_response.json()["video_pool_mode"], "long-form")
        self.assertEqual(refreshed_response.json()["target_clip_duration_sec"], 120)

    def test_settings_are_isolated_per_account(self) -> None:
        primary_session = self._register_and_verify(
            client=self.client,
            owner_key="settings-primary-owner-key-abcdefghijklmnopqrstuvwxyz",
            username="settingsprimary",
            password="studyreels-password",
        )
        secondary_session = self._register_and_verify(
            client=self.third_client,
            owner_key="settings-secondary-owner-key-abcdefghijklmnopqrstuvwxyz",
            username="settingssecondary",
            password="studyreels-password",
        )

        replace_response = self.client.put(
            "/api/community/settings",
            headers={COMMUNITY_SESSION_HEADER: primary_session},
            json={
                "generation_mode": "slow",
                "default_input_mode": "topic",
                "min_relevance_threshold": 0.16,
                "start_muted": False,
                "video_pool_mode": "balanced",
                "preferred_video_duration": "medium",
                "target_clip_duration_sec": 70,
                "target_clip_duration_min_sec": 40,
                "target_clip_duration_max_sec": 80,
            },
        )
        self.assertEqual(replace_response.status_code, 200, replace_response.text)

        secondary_response = self.third_client.get(
            "/api/community/settings",
            headers={COMMUNITY_SESSION_HEADER: secondary_session},
        )
        self.assertEqual(secondary_response.status_code, 200, secondary_response.text)
        self.assertEqual(secondary_response.json()["generation_mode"], "fast")
        self.assertEqual(secondary_response.json()["default_input_mode"], "source")
        self.assertEqual(secondary_response.json()["target_clip_duration_sec"], 55)


if __name__ == "__main__":
    unittest.main()
