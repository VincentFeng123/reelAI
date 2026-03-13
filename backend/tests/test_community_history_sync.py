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


class CommunityHistorySyncTests(unittest.TestCase):
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
            execute_modify(conn, "DELETE FROM community_material_history")
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

    def test_history_syncs_across_devices_for_same_account(self) -> None:
        owner_key = "history-owner-key-abcdefghijklmnopqrstuvwxyz"
        username = "historysync"
        password = "studyreels-password"
        session_token = self._register_and_verify(client=self.client, owner_key=owner_key, username=username, password=password)

        replace_response = self.client.put(
            "/api/community/history",
            headers={COMMUNITY_SESSION_HEADER: session_token},
            json={
                "items": [
                    {
                        "material_id": "material-apush",
                        "title": "APUSH",
                        "updated_at": 200,
                        "starred": True,
                        "generation_mode": "fast",
                        "source": "search",
                    },
                    {
                        "material_id": "material-spanish",
                        "title": "AP Spanish",
                        "updated_at": 100,
                        "starred": False,
                        "generation_mode": "slow",
                        "source": "search",
                    },
                ]
            },
        )
        self.assertEqual(replace_response.status_code, 200, replace_response.text)

        login_response = self.other_client.post(
            "/api/community/auth/login",
            json={"username": username, "password": password},
        )
        self.assertEqual(login_response.status_code, 200, login_response.text)
        other_session_token = str(login_response.json()["session_token"])

        other_history_response = self.other_client.get(
            "/api/community/history",
            headers={COMMUNITY_SESSION_HEADER: other_session_token},
        )
        self.assertEqual(other_history_response.status_code, 200, other_history_response.text)
        self.assertEqual(
            [row["material_id"] for row in other_history_response.json()["items"]],
            ["material-apush", "material-spanish"],
        )

        overwrite_response = self.other_client.put(
            "/api/community/history",
            headers={COMMUNITY_SESSION_HEADER: other_session_token},
            json={
                "items": [
                    {
                        "material_id": "material-calc",
                        "title": "AP Calculus",
                        "updated_at": 300,
                        "starred": False,
                        "generation_mode": "fast",
                        "source": "search",
                    }
                ]
            },
        )
        self.assertEqual(overwrite_response.status_code, 200, overwrite_response.text)

        refreshed_history_response = self.client.get(
            "/api/community/history",
            headers={COMMUNITY_SESSION_HEADER: session_token},
        )
        self.assertEqual(refreshed_history_response.status_code, 200, refreshed_history_response.text)
        self.assertEqual(
            [row["material_id"] for row in refreshed_history_response.json()["items"]],
            ["material-calc"],
        )

    def test_history_is_isolated_per_account(self) -> None:
        primary_session = self._register_and_verify(
            client=self.client,
            owner_key="history-primary-owner-key-abcdefghijklmnopqrstuvwxyz",
            username="historyprimary",
            password="studyreels-password",
        )
        secondary_session = self._register_and_verify(
            client=self.third_client,
            owner_key="history-secondary-owner-key-abcdefghijklmnopqrstuvwxyz",
            username="historysecondary",
            password="studyreels-password",
        )

        replace_response = self.client.put(
            "/api/community/history",
            headers={COMMUNITY_SESSION_HEADER: primary_session},
            json={
                "items": [
                    {
                        "material_id": "material-primary",
                        "title": "Primary Account Topic",
                        "updated_at": 400,
                        "starred": False,
                        "generation_mode": "fast",
                        "source": "search",
                    }
                ]
            },
        )
        self.assertEqual(replace_response.status_code, 200, replace_response.text)

        secondary_history_response = self.third_client.get(
            "/api/community/history",
            headers={COMMUNITY_SESSION_HEADER: secondary_session},
        )
        self.assertEqual(secondary_history_response.status_code, 200, secondary_history_response.text)
        self.assertEqual(secondary_history_response.json()["items"], [])


if __name__ == "__main__":
    unittest.main()
