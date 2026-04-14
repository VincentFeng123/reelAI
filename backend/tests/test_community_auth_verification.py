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
from backend.app.db import dumps_json, fetch_one, get_conn, insert, now_iso
import backend.app.main as main_module
from backend.app.main import COMMUNITY_OWNER_HEADER, COMMUNITY_SESSION_HEADER, _hash_community_password, _hash_community_verification_code, app


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

        # With the pre-signup verification flow, the 503 now surfaces
        # at the send-signup-verification step when HMAC key is missing.
        response = self.client.post(
            "/api/community/auth/send-signup-verification",
            headers={COMMUNITY_OWNER_HEADER: "owner-key-abcdefghijklmnopqrstuvwxyz"},
            json={
                "email": "hostedsecret@example.com",
                "username": "hostedsecret",
            },
        )

        self.assertEqual(response.status_code, 503, response.text)
        self.assertIn("verification", response.json()["detail"].lower())

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
    def _presignup(self, *, owner_key: str, email: str, username: str) -> None:
        send_resp = self.client.post(
            "/api/community/auth/send-signup-verification",
            headers={COMMUNITY_OWNER_HEADER: owner_key},
            json={"email": email, "username": username},
        )
        self.assertEqual(send_resp.status_code, 200, send_resp.text)
        code = str(send_resp.json().get("verification_code_debug") or "")
        self.assertTrue(code)
        verify_resp = self.client.post(
            "/api/community/auth/verify-signup-email",
            headers={COMMUNITY_OWNER_HEADER: owner_key},
            json={"email": email, "code": code},
        )
        self.assertEqual(verify_resp.status_code, 200, verify_resp.text)

    def _create_unverified_account_with_session(self, *, username: str, email: str, password: str) -> tuple[str, str]:
        """Insert an unverified account + session directly into the DB."""
        import secrets as _secrets
        account_id = str(__import__("uuid").uuid4())
        salt_hex = "bb" * 16
        token_raw = _secrets.token_urlsafe(32)
        token_hash = __import__("hashlib").sha256(token_raw.encode("utf-8")).hexdigest()
        ts = now_iso()
        code = "123456"
        code_hash = _hash_community_verification_code(code)
        with get_conn(transactional=True) as conn:
            insert(conn, "community_accounts", {
                "id": account_id, "username": username, "username_normalized": username.lower(),
                "email": email, "email_normalized": email.lower(),
                "password_hash": _hash_community_password(password, salt_hex), "password_salt": salt_hex,
                "verified_at": None, "verification_code_hash": code_hash,
                "verification_expires_at": "2099-01-01T00:00:00+00:00",
                "legacy_claim_owner_key_hash": None, "created_at": ts, "updated_at": ts,
            })
            insert(conn, "community_sessions", {
                "id": str(__import__("uuid").uuid4()), "account_id": account_id,
                "token_hash": token_hash, "created_at": ts, "last_used_at": ts,
                "expires_at": "2099-01-01T00:00:00+00:00",
            })
        return token_raw, code

    def test_unverified_account_can_change_email_and_only_new_code_works(self) -> None:
        session_token, original_code = self._create_unverified_account_with_session(
            username="changeemailuser", email="old-email@example.com", password="password123",
        )

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

        session_token, _ = self._create_unverified_account_with_session(
            username="changemaildup", email="available@example.com", password="password123",
        )

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


class CommunityDeleteAccountTests(_VerificationTestBase):
    def _register_and_verify(self, *, owner_key: str, username: str, password: str) -> tuple[str, str]:
        email = f"{username}@example.com"
        # Pre-signup verification
        send_resp = self.client.post(
            "/api/community/auth/send-signup-verification",
            headers={COMMUNITY_OWNER_HEADER: owner_key},
            json={"email": email, "username": username},
        )
        self.assertEqual(send_resp.status_code, 200, send_resp.text)
        signup_code = str(send_resp.json().get("verification_code_debug") or "")
        self.assertTrue(signup_code)
        verify_signup_resp = self.client.post(
            "/api/community/auth/verify-signup-email",
            headers={COMMUNITY_OWNER_HEADER: owner_key},
            json={"email": email, "code": signup_code},
        )
        self.assertEqual(verify_signup_resp.status_code, 200, verify_signup_resp.text)

        register_response = self.client.post(
            "/api/community/auth/register",
            headers={COMMUNITY_OWNER_HEADER: owner_key},
            json={
                "username": username,
                "email": email,
                "password": password,
            },
        )
        self.assertEqual(register_response.status_code, 201, register_response.text)
        register_payload = register_response.json()
        session_token = str(register_payload["session_token"])
        account_id = str(register_payload["account"]["id"])

        if register_payload.get("verification_required") and register_payload.get("verification_code_debug"):
            verification_code = str(register_payload["verification_code_debug"])
            verify_response = self.client.post(
                "/api/community/auth/verify",
                headers={
                    COMMUNITY_OWNER_HEADER: owner_key,
                    COMMUNITY_SESSION_HEADER: session_token,
                },
                json={"code": verification_code},
            )
            self.assertEqual(verify_response.status_code, 200, verify_response.text)
            self.assertTrue(verify_response.json()["account"]["is_verified"])
        return account_id, session_token

    def test_delete_account_removes_account_sessions_and_owned_data(self) -> None:
        owner_key = "delete-owner-key-abcdefghijklmnopqrstuvwxyz"
        username = "deleteowner"
        password = "correct-password"
        account_id, session_token = self._register_and_verify(owner_key=owner_key, username=username, password=password)
        timestamp = now_iso()

        with get_conn(transactional=True) as conn:
            insert(
                conn,
                "community_sets",
                {
                    "id": "user-set-delete-me",
                    "title": "Delete Me",
                    "description": "This description is long enough to satisfy the backend validation.",
                    "tags_json": dumps_json(["cleanup"]),
                    "reels_json": dumps_json(
                        [
                            {
                                "id": "delete-reel",
                                "platform": "youtube",
                                "source_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                                "embed_url": "https://www.youtube.com/embed/dQw4w9WgXcQ",
                                "t_start_sec": 0,
                                "t_end_sec": 30,
                            }
                        ]
                    ),
                    "reel_count": 1,
                    "curator": "Delete Owner",
                    "likes": 0,
                    "learners": 1,
                    "updated_label": "Last Edited: just now",
                    "thumbnail_url": "https://example.com/thumb.png",
                    "owner_key_hash": hashlib.sha256(owner_key.encode("utf-8")).hexdigest(),
                    "owner_account_id": account_id,
                    "visibility": "private",
                    "featured": 0,
                    "created_at": timestamp,
                    "updated_at": timestamp,
                },
            )
            insert(
                conn,
                "community_material_history",
                {
                    "account_id": account_id,
                    "material_id": "material-delete-test",
                    "title": "Delete History",
                    "updated_at": 1234567890,
                    "starred": 1,
                    "generation_mode": "fast",
                    "source": "search",
                    "feed_query": "delete test",
                },
            )
            insert(
                conn,
                "community_account_settings",
                {
                    "account_id": account_id,
                    "generation_mode": "fast",
                    "default_input_mode": "source",
                    "min_relevance_threshold": 0.3,
                    "start_muted": 1,
                    "video_pool_mode": "short-first",
                    "preferred_video_duration": "any",
                    "target_clip_duration_sec": 55,
                    "target_clip_duration_min_sec": 20,
                    "target_clip_duration_max_sec": 55,
                    "updated_at": timestamp,
                },
            )

        delete_response = self.client.post(
            "/api/community/auth/delete-account",
            headers={COMMUNITY_SESSION_HEADER: session_token},
            json={"current_password": password},
        )
        self.assertEqual(delete_response.status_code, 204, delete_response.text)

        with get_conn(transactional=True) as conn:
            self.assertIsNone(fetch_one(conn, "SELECT id FROM community_accounts WHERE id = ? LIMIT 1", (account_id,)))
            self.assertIsNone(fetch_one(conn, "SELECT id FROM community_sessions WHERE account_id = ? LIMIT 1", (account_id,)))
            self.assertIsNone(fetch_one(conn, "SELECT id FROM community_sets WHERE owner_account_id = ? LIMIT 1", (account_id,)))
            self.assertIsNone(fetch_one(conn, "SELECT material_id FROM community_material_history WHERE account_id = ? LIMIT 1", (account_id,)))
            self.assertIsNone(fetch_one(conn, "SELECT account_id FROM community_account_settings WHERE account_id = ? LIMIT 1", (account_id,)))

        login_response = self.client.post(
            "/api/community/auth/login",
            json={"username": username, "password": password},
        )
        self.assertEqual(login_response.status_code, 401, login_response.text)
        self.assertIn("incorrect username or password", login_response.json()["detail"].lower())

    def test_delete_account_rejects_incorrect_password_without_deleting(self) -> None:
        owner_key = "delete-wrong-password-owner-key-abcdefghijklmnopqrstuvwxyz"
        password = "correct-password"
        account_id, session_token = self._register_and_verify(
            owner_key=owner_key,
            username="deletewrongpassword",
            password=password,
        )

        delete_response = self.client.post(
            "/api/community/auth/delete-account",
            headers={COMMUNITY_SESSION_HEADER: session_token},
            json={"current_password": "wrong-password"},
        )
        self.assertEqual(delete_response.status_code, 401, delete_response.text)
        self.assertIn("current password is incorrect", delete_response.json()["detail"].lower())

        with get_conn(transactional=True) as conn:
            self.assertIsNotNone(fetch_one(conn, "SELECT id FROM community_accounts WHERE id = ? LIMIT 1", (account_id,)))
            self.assertIsNotNone(fetch_one(conn, "SELECT id FROM community_sessions WHERE account_id = ? LIMIT 1", (account_id,)))


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
