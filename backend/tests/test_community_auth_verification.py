import hashlib
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
from backend.app.db import fetch_one, get_conn, insert, now_iso
import backend.app.main as main_module
from backend.app.main import COMMUNITY_OWNER_HEADER, COMMUNITY_SESSION_HEADER, _hash_community_password, app


class _VerificationTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.previous_env = {key: os.environ.get(key) for key in (
            "DATA_DIR",
            "APP_ENV",
            "COMMUNITY_EMAIL_VERIFICATION_REQUIRED",
            "DATABASE_URL",
            "SMTP_HOST",
            "SMTP_PORT",
            "SMTP_USERNAME",
            "SMTP_PASSWORD",
            "SMTP_FROM_EMAIL",
            "SMTP_USE_TLS",
            "SMTP_USE_SSL",
            "VERIFICATION_HMAC_KEY",
        )}
        self.addCleanup(self._restore_environment)
        os.environ["DATA_DIR"] = self.temp_dir.name
        db_module._db_ready = False
        main_module._rate_limit_hits.clear()
        main_module._rate_limit_last_sweep = 0.0
        main_module._rate_limit_last_db_cleanup = 0.0
        main_module._local_verification_hmac_key_cache = None
        os.environ["COMMUNITY_EMAIL_VERIFICATION_REQUIRED"] = "1"
        get_settings.cache_clear()
        main_module.settings = get_settings()
        self._refresh_client()

    def _refresh_client(self) -> None:
        client = getattr(self, "client", None)
        if client is not None:
            client.close()
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def _restore_environment(self) -> None:
        for key, value in self.previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        db_module._db_ready = False
        main_module._rate_limit_hits.clear()
        main_module._rate_limit_last_sweep = 0.0
        main_module._rate_limit_last_db_cleanup = 0.0
        main_module._local_verification_hmac_key_cache = None
        get_settings.cache_clear()
        main_module.settings = get_settings()


class HostedVerificationDeliveryTests(_VerificationTestBase):
    def setUp(self) -> None:
        super().setUp()
        os.environ["APP_ENV"] = "production"
        os.environ.pop("SMTP_HOST", None)
        os.environ.pop("SMTP_FROM_EMAIL", None)
        get_settings.cache_clear()
        main_module.settings = get_settings()
        db_module._db_ready = False
        self._refresh_client()

    def test_hosted_register_requires_verification_delivery(self) -> None:
        response = self.client.post(
            "/api/community/auth/register",
            headers={COMMUNITY_OWNER_HEADER: "owner-key-abcdefghijklmnopqrstuvwxyz"},
            json={
                "username": "hostedregister",
                "email": "hostedregister@example.com",
                "password": "password123",
            },
        )

        self.assertEqual(response.status_code, 503, response.text)
        self.assertIn("verification email is not configured", response.json()["detail"].lower())

        with get_conn(transactional=True) as conn:
            account = fetch_one(
                conn,
                "SELECT id FROM community_accounts WHERE username_normalized = ? LIMIT 1",
                ("hostedregister",),
            )
            session = fetch_one(conn, "SELECT id FROM community_sessions LIMIT 1")
        self.assertIsNone(account)
        self.assertIsNone(session)

    def test_hosted_login_requires_verification_delivery(self) -> None:
        timestamp = now_iso()
        salt_hex = "11" * 16
        with get_conn(transactional=True) as conn:
            insert(
                conn,
                "community_accounts",
                {
                    "id": "unverified-account",
                    "username": "hostedlogin",
                    "username_normalized": "hostedlogin",
                    "email": "hostedlogin@example.com",
                    "email_normalized": "hostedlogin@example.com",
                    "password_hash": _hash_community_password("password123", salt_hex),
                    "password_salt": salt_hex,
                    "verified_at": None,
                    "verification_code_hash": None,
                    "verification_expires_at": None,
                    "created_at": timestamp,
                    "updated_at": timestamp,
                },
            )

        response = self.client.post(
            "/api/community/auth/login",
            headers={COMMUNITY_OWNER_HEADER: "owner-key-abcdefghijklmnopqrstuvwxyz"},
            json={"username": "hostedlogin", "password": "password123"},
        )

        self.assertEqual(response.status_code, 503, response.text)
        self.assertIn("verification email is not configured", response.json()["detail"].lower())

        with get_conn(transactional=True) as conn:
            session = fetch_one(conn, "SELECT id FROM community_sessions LIMIT 1")
        self.assertIsNone(session)

    def test_hosted_register_requires_verification_secret(self) -> None:
        os.environ["SMTP_HOST"] = "smtp.example.com"
        os.environ["SMTP_FROM_EMAIL"] = "noreply@example.com"
        os.environ.pop("VERIFICATION_HMAC_KEY", None)
        get_settings.cache_clear()
        main_module.settings = get_settings()
        db_module._db_ready = False
        self._refresh_client()

        response = self.client.post(
            "/api/community/auth/register",
            headers={COMMUNITY_OWNER_HEADER: "owner-key-abcdefghijklmnopqrstuvwxyz"},
            json={
                "username": "hostedsecret",
                "email": "hostedsecret@example.com",
                "password": "password123",
            },
        )

        self.assertEqual(response.status_code, 503, response.text)
        self.assertIn("verification secret is not configured", response.json()["detail"].lower())

        with get_conn(transactional=True) as conn:
            account = fetch_one(
                conn,
                "SELECT id FROM community_accounts WHERE username_normalized = ? LIMIT 1",
                ("hostedsecret",),
            )
            session = fetch_one(conn, "SELECT id FROM community_sessions LIMIT 1")
        self.assertIsNone(account)
        self.assertIsNone(session)


class CommunityChangeEmailTests(_VerificationTestBase):
    def test_unverified_account_can_change_email_and_only_new_code_works(self) -> None:
        register_response = self.client.post(
            "/api/community/auth/register",
            headers={COMMUNITY_OWNER_HEADER: "owner-key-abcdefghijklmnopqrstuvwxyz"},
            json={
                "username": "changeemailuser",
                "email": "old-email@example.com",
                "password": "password123",
            },
        )
        self.assertEqual(register_response.status_code, 201, register_response.text)
        register_json = register_response.json()
        original_code = str(register_json["verification_code_debug"])
        session_token = str(register_json["session_token"])

        change_email_response = self.client.post(
            "/api/community/auth/change-email",
            headers={COMMUNITY_SESSION_HEADER: session_token},
            json={
                "email": "new-email@example.com",
                "current_password": "password123",
            },
        )
        self.assertEqual(change_email_response.status_code, 200, change_email_response.text)
        change_email_json = change_email_response.json()
        self.assertEqual(change_email_json["account"]["email"], "new-email@example.com")
        self.assertFalse(change_email_json["account"]["is_verified"])
        new_code = str(change_email_json["verification_code_debug"])
        self.assertNotEqual(original_code, new_code)

        old_verify_response = self.client.post(
            "/api/community/auth/verify",
            headers={COMMUNITY_SESSION_HEADER: session_token},
            json={"code": original_code},
        )
        self.assertEqual(old_verify_response.status_code, 400, old_verify_response.text)
        self.assertIn("incorrect", old_verify_response.json()["detail"].lower())

        verify_response = self.client.post(
            "/api/community/auth/verify",
            headers={
                COMMUNITY_OWNER_HEADER: "owner-key-abcdefghijklmnopqrstuvwxyz",
                COMMUNITY_SESSION_HEADER: session_token,
            },
            json={"code": new_code},
        )
        self.assertEqual(verify_response.status_code, 200, verify_response.text)
        self.assertTrue(verify_response.json()["account"]["is_verified"])

    def test_change_email_duplicate_returns_generic_conflict(self) -> None:
        timestamp = now_iso()
        salt_hex = "44" * 16
        with get_conn(transactional=True) as conn:
            insert(
                conn,
                "community_accounts",
                {
                    "id": "existing-email-account",
                    "username": "existingemailowner",
                    "username_normalized": "existingemailowner",
                    "email": "taken@example.com",
                    "email_normalized": "taken@example.com",
                    "password_hash": _hash_community_password("password123", salt_hex),
                    "password_salt": salt_hex,
                    "verified_at": timestamp,
                    "verification_code_hash": None,
                    "verification_expires_at": None,
                    "legacy_claim_owner_key_hash": None,
                    "created_at": timestamp,
                    "updated_at": timestamp,
                },
            )

        register_response = self.client.post(
            "/api/community/auth/register",
            headers={COMMUNITY_OWNER_HEADER: "owner-key-abcdefghijklmnopqrstuvwxyz"},
            json={
                "username": "changemaildup",
                "email": "available@example.com",
                "password": "password123",
            },
        )
        self.assertEqual(register_response.status_code, 201, register_response.text)
        session_token = str(register_response.json()["session_token"])

        change_email_response = self.client.post(
            "/api/community/auth/change-email",
            headers={COMMUNITY_SESSION_HEADER: session_token},
            json={
                "email": "taken@example.com",
                "current_password": "password123",
            },
        )
        self.assertEqual(change_email_response.status_code, 409, change_email_response.text)
        self.assertEqual(
            change_email_response.json()["detail"],
            main_module.COMMUNITY_CHANGE_EMAIL_CONFLICT_DETAIL,
        )


class CommunityAuthSecurityTests(_VerificationTestBase):
    def test_verification_hash_uses_settings_secret_when_process_env_is_empty(self) -> None:
        os.environ.pop("VERIFICATION_HMAC_KEY", None)
        main_module.settings.verification_hmac_key = "settings-secret-only"

        expected_hash = hashlib.pbkdf2_hmac(
            "sha256",
            b"123456",
            b"settings-secret-only",
            100_000,
        ).hex()
        self.assertEqual(main_module._hash_community_verification_code("123456"), expected_hash)

    def test_local_postgres_keeps_verification_debug_mode_enabled(self) -> None:
        os.environ["DATABASE_URL"] = "postgresql://local-dev/reelai"
        get_settings.cache_clear()
        main_module.settings = get_settings()

        self.assertTrue(main_module._community_verification_debug_mode_enabled())
        main_module._require_hosted_verification_delivery_available()

    def test_username_normalization_rejects_short_usernames(self) -> None:
        with self.assertRaises(main_module.HTTPException) as ctx:
            main_module._normalize_community_username("a")
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("3-32 characters", str(ctx.exception.detail))

    def test_register_duplicate_username_or_email_returns_generic_conflict(self) -> None:
        timestamp = now_iso()
        salt_hex = "22" * 16
        with get_conn(transactional=True) as conn:
            insert(
                conn,
                "community_accounts",
                {
                    "id": "existing-account",
                    "username": "existinguser",
                    "username_normalized": "existinguser",
                    "email": "existing@example.com",
                    "email_normalized": "existing@example.com",
                    "password_hash": _hash_community_password("password123", salt_hex),
                    "password_salt": salt_hex,
                    "verified_at": timestamp,
                    "verification_code_hash": None,
                    "verification_expires_at": None,
                    "legacy_claim_owner_key_hash": None,
                    "created_at": timestamp,
                    "updated_at": timestamp,
                },
            )

        duplicate_username = self.client.post(
            "/api/community/auth/register",
            headers={COMMUNITY_OWNER_HEADER: "owner-key-abcdefghijklmnopqrstuvwxyz"},
            json={
                "username": "existinguser",
                "email": "different@example.com",
                "password": "password123",
            },
        )
        self.assertEqual(duplicate_username.status_code, 409, duplicate_username.text)
        self.assertEqual(
            duplicate_username.json()["detail"],
            main_module.COMMUNITY_REGISTER_CONFLICT_DETAIL,
        )

        duplicate_email = self.client.post(
            "/api/community/auth/register",
            headers={COMMUNITY_OWNER_HEADER: "owner-key-abcdefghijklmnopqrstuvwxyz"},
            json={
                "username": "anotheruser",
                "email": "existing@example.com",
                "password": "password123",
            },
        )
        self.assertEqual(duplicate_email.status_code, 409, duplicate_email.text)
        self.assertEqual(
            duplicate_email.json()["detail"],
            main_module.COMMUNITY_REGISTER_CONFLICT_DETAIL,
        )

    def test_login_username_rate_limit_is_scoped_per_ip(self) -> None:
        timestamp = now_iso()
        salt_hex = "33" * 16
        with get_conn(transactional=True) as conn:
            insert(
                conn,
                "community_accounts",
                {
                    "id": "login-target",
                    "username": "scopeduser",
                    "username_normalized": "scopeduser",
                    "email": "scopeduser@example.com",
                    "email_normalized": "scopeduser@example.com",
                    "password_hash": _hash_community_password("correct-password", salt_hex),
                    "password_salt": salt_hex,
                    "verified_at": timestamp,
                    "verification_code_hash": None,
                    "verification_expires_at": None,
                    "legacy_claim_owner_key_hash": None,
                    "created_at": timestamp,
                    "updated_at": timestamp,
                },
            )

        with mock.patch.object(main_module, "_client_ip", return_value="198.51.100.10"):
            for _ in range(main_module.COMMUNITY_LOGIN_PER_USERNAME_RATE_LIMIT):
                response = self.client.post(
                    "/api/community/auth/login",
                    json={"username": "scopeduser", "password": "wrong-password"},
                )
                self.assertEqual(response.status_code, 401, response.text)
            throttled = self.client.post(
                "/api/community/auth/login",
                json={"username": "scopeduser", "password": "wrong-password"},
            )
        self.assertEqual(throttled.status_code, 429, throttled.text)

        with mock.patch.object(main_module, "_client_ip", return_value="198.51.100.11"):
            other_ip_response = self.client.post(
                "/api/community/auth/login",
                json={"username": "scopeduser", "password": "wrong-password"},
            )
        self.assertEqual(other_ip_response.status_code, 401, other_ip_response.text)


class VerificationDisabledModeTests(_VerificationTestBase):
    def setUp(self) -> None:
        super().setUp()
        os.environ["APP_ENV"] = "production"
        os.environ["COMMUNITY_EMAIL_VERIFICATION_REQUIRED"] = "0"
        os.environ.pop("SMTP_HOST", None)
        os.environ.pop("SMTP_FROM_EMAIL", None)
        os.environ.pop("VERIFICATION_HMAC_KEY", None)
        get_settings.cache_clear()
        main_module.settings = get_settings()
        db_module._db_ready = False
        self._refresh_client()

    def test_hosted_register_auto_verifies_without_email_delivery(self) -> None:
        response = self.client.post(
            "/api/community/auth/register",
            headers={COMMUNITY_OWNER_HEADER: "owner-key-abcdefghijklmnopqrstuvwxyz"},
            json={
                "username": "autoverify",
                "email": "autoverify@example.com",
                "password": "password123",
            },
        )

        self.assertEqual(response.status_code, 201, response.text)
        payload = response.json()
        self.assertFalse(payload["verification_required"])
        self.assertIsNone(payload["verification_code_debug"])
        self.assertTrue(payload["account"]["is_verified"])

        with get_conn(transactional=True) as conn:
            account = fetch_one(
                conn,
                "SELECT verified_at FROM community_accounts WHERE username_normalized = ? LIMIT 1",
                ("autoverify",),
            )
        self.assertIsNotNone(account)
        self.assertTrue(str(account["verified_at"] or "").strip())

    def test_login_auto_verifies_existing_unverified_account_when_disabled(self) -> None:
        timestamp = now_iso()
        salt_hex = "55" * 16
        with get_conn(transactional=True) as conn:
            insert(
                conn,
                "community_accounts",
                {
                    "id": "login-autoverify-account",
                    "username": "autologin",
                    "username_normalized": "autologin",
                    "email": "autologin@example.com",
                    "email_normalized": "autologin@example.com",
                    "password_hash": _hash_community_password("password123", salt_hex),
                    "password_salt": salt_hex,
                    "verified_at": None,
                    "verification_code_hash": None,
                    "verification_expires_at": None,
                    "legacy_claim_owner_key_hash": None,
                    "created_at": timestamp,
                    "updated_at": timestamp,
                },
            )

        response = self.client.post(
            "/api/community/auth/login",
            headers={COMMUNITY_OWNER_HEADER: "owner-key-abcdefghijklmnopqrstuvwxyz"},
            json={"username": "autologin", "password": "password123"},
        )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertFalse(payload["verification_required"])
        self.assertIsNone(payload["verification_code_debug"])
        self.assertTrue(payload["account"]["is_verified"])

        with get_conn(transactional=True) as conn:
            account = fetch_one(
                conn,
                "SELECT verified_at FROM community_accounts WHERE username_normalized = ? LIMIT 1",
                ("autologin",),
            )
        self.assertIsNotNone(account)
        self.assertTrue(str(account["verified_at"] or "").strip())

    def test_change_email_keeps_account_verified_when_verification_is_disabled(self) -> None:
        register_response = self.client.post(
            "/api/community/auth/register",
            headers={COMMUNITY_OWNER_HEADER: "owner-key-abcdefghijklmnopqrstuvwxyz"},
            json={
                "username": "disableemailchange",
                "email": "old-disable@example.com",
                "password": "password123",
            },
        )
        self.assertEqual(register_response.status_code, 201, register_response.text)
        session_token = str(register_response.json()["session_token"])

        change_email_response = self.client.post(
            "/api/community/auth/change-email",
            headers={COMMUNITY_SESSION_HEADER: session_token},
            json={
                "email": "new-disable@example.com",
                "current_password": "password123",
            },
        )
        self.assertEqual(change_email_response.status_code, 200, change_email_response.text)
        payload = change_email_response.json()
        self.assertEqual(payload["account"]["email"], "new-disable@example.com")
        self.assertTrue(payload["account"]["is_verified"])
        self.assertIsNone(payload["verification_code_debug"])


if __name__ == "__main__":
    unittest.main()
