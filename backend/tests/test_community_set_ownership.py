import hashlib
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

from backend.app.db import dumps_json, execute_modify, fetch_one, get_conn, insert, now_iso, upsert
from backend.app.main import COMMUNITY_OWNER_HEADER, COMMUNITY_SESSION_HEADER, _hash_community_password, app


def _owner_hash(owner_key: str) -> str:
    return hashlib.sha256(owner_key.encode("utf-8")).hexdigest()


class CommunitySetOwnershipTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_verification_required = os.environ.get("COMMUNITY_EMAIL_VERIFICATION_REQUIRED")
        os.environ["COMMUNITY_EMAIL_VERIFICATION_REQUIRED"] = "1"
        get_settings.cache_clear()
        main_module.settings = get_settings()
        self.client = TestClient(app)
        self.other_client = TestClient(app)
        self.addCleanup(self.client.close)
        self.addCleanup(self.other_client.close)
        self.addCleanup(self._restore_environment)
        with get_conn(transactional=True) as conn:
            execute_modify(conn, "DELETE FROM community_sessions")
            execute_modify(conn, "DELETE FROM community_accounts")
            execute_modify(conn, "DELETE FROM community_sets")

    def _restore_environment(self) -> None:
        if self.previous_verification_required is None:
            os.environ.pop("COMMUNITY_EMAIL_VERIFICATION_REQUIRED", None)
        else:
            os.environ["COMMUNITY_EMAIL_VERIFICATION_REQUIRED"] = self.previous_verification_required
        get_settings.cache_clear()
        main_module.settings = get_settings()

    def _community_payload(self, *, title: str = "Community Systems Study Set") -> dict:
        return {
            "title": title,
            "description": "This description is long enough to satisfy the backend validation.",
            "tags": ["systems", "study"],
            "thumbnail_url": "https://example.com/thumb.png",
            "curator": "Security Test",
            "reels": [
                {
                    "platform": "youtube",
                    "source_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                    "embed_url": "https://www.youtube.com/embed/dQw4w9WgXcQ",
                    "t_start_sec": 0,
                    "t_end_sec": 30,
                }
            ],
        }

    def _insert_user_set(
        self,
        *,
        set_id: str,
        owner_key_hash: str,
        owner_account_id: str | None = None,
        visibility: str = "private",
        featured: int = 0,
        title: str = "Original Title",
    ) -> None:
        timestamp = now_iso()
        with get_conn(transactional=True) as conn:
            upsert(
                conn,
                "community_sets",
                {
                    "id": set_id,
                    "title": title,
                    "description": "This description is long enough to satisfy the backend validation.",
                    "tags_json": dumps_json(["systems"]),
                    "reels_json": dumps_json(
                        [
                            {
                                "id": "seed-reel",
                                "platform": "youtube",
                                "source_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                                "embed_url": "https://www.youtube.com/embed/dQw4w9WgXcQ",
                                "t_start_sec": 0,
                                "t_end_sec": 30,
                            }
                        ]
                    ),
                    "reel_count": 1,
                    "curator": "Seed Owner",
                    "likes": 0,
                    "learners": 1,
                    "updated_label": "Last Edited: just now",
                    "thumbnail_url": "https://example.com/thumb.png",
                    "owner_key_hash": owner_key_hash,
                    "owner_account_id": owner_account_id,
                    "visibility": visibility,
                    "featured": featured,
                    "created_at": timestamp,
                    "updated_at": timestamp,
                },
            )

    def _register_and_verify(self, *, owner_key: str, username: str, password: str) -> str:
        email = f"{username}@example.com"
        # Step 1: send pre-signup verification code
        send_response = self.client.post(
            "/api/community/auth/send-signup-verification",
            headers={COMMUNITY_OWNER_HEADER: owner_key},
            json={"email": email, "username": username},
        )
        self.assertEqual(send_response.status_code, 200, send_response.text)
        send_json = send_response.json()
        signup_code = str(send_json.get("verification_code_debug") or "")
        self.assertTrue(signup_code, "Expected verification_code_debug in debug mode")

        # Step 2: verify the signup email
        verify_signup_response = self.client.post(
            "/api/community/auth/verify-signup-email",
            headers={COMMUNITY_OWNER_HEADER: owner_key},
            json={"email": email, "code": signup_code},
        )
        self.assertEqual(verify_signup_response.status_code, 200, verify_signup_response.text)
        self.assertTrue(verify_signup_response.json()["verified"])

        # Step 3: register (now allowed)
        response = self.client.post(
            "/api/community/auth/register",
            headers={COMMUNITY_OWNER_HEADER: owner_key},
            json={"username": username, "email": email, "password": password},
        )
        self.assertEqual(response.status_code, 201, response.text)
        register_json = response.json()
        session_token = str(register_json["session_token"])

        # Step 4: verify the account (post-registration)
        if register_json.get("verification_required") and register_json.get("verification_code_debug"):
            verification_code = str(register_json["verification_code_debug"])
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
        return session_token

    def test_public_listing_excludes_private_user_sets(self) -> None:
        self._insert_user_set(
            set_id="user-set-private",
            owner_key_hash=_owner_hash("private-owner-key-abcdefghijklmnopqrstuvwxyz"),
            visibility="private",
            title="Private Set",
        )
        self._insert_user_set(
            set_id="user-set-public",
            owner_key_hash=_owner_hash("public-owner-key-abcdefghijklmnopqrstuvwxyz"),
            visibility="public",
            title="Public Set",
        )

        response = self.client.get("/api/community/sets")

        self.assertEqual(response.status_code, 200)
        returned_ids = {row["id"] for row in response.json()["sets"]}
        self.assertIn("user-set-public", returned_ids)
        self.assertNotIn("user-set-private", returned_ids)

    def test_create_requires_signed_in_account(self) -> None:
        response = self.client.post(
            "/api/community/sets",
            headers={COMMUNITY_OWNER_HEADER: "owner-key-abcdefghijklmnopqrstuvwxyz"},
            json=self._community_payload(),
        )

        self.assertEqual(response.status_code, 401)
        self.assertIn("Sign in", response.json()["detail"])

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

    def test_unverified_account_cannot_access_private_sets(self) -> None:
        import hashlib, secrets
        owner_key = "needs-verification-owner-key-abcdefghijklmnopqrstuvwxyz"
        # Create an unverified account directly in the DB (the API flow auto-verifies
        # via pre-signup verification, so we bypass it to test the unverified path).
        account_id = "unverified-test-account"
        salt_hex = "aa" * 16
        session_token_raw = secrets.token_urlsafe(32)
        session_token_hash = hashlib.sha256(session_token_raw.encode("utf-8")).hexdigest()
        timestamp = now_iso()
        with get_conn(transactional=True) as conn:
            insert(conn, "community_accounts", {
                "id": account_id,
                "username": "needsverify",
                "username_normalized": "needsverify",
                "email": "needsverify@example.com",
                "email_normalized": "needsverify@example.com",
                "password_hash": _hash_community_password("correct horse battery", salt_hex),
                "password_salt": salt_hex,
                "verified_at": None,
                "verification_code_hash": None,
                "verification_expires_at": None,
                "legacy_claim_owner_key_hash": None,
                "created_at": timestamp,
                "updated_at": timestamp,
            })
            insert(conn, "community_sessions", {
                "id": "unverified-session",
                "account_id": account_id,
                "token_hash": session_token_hash,
                "created_at": timestamp,
                "last_used_at": timestamp,
                "expires_at": "2099-01-01T00:00:00+00:00",
            })
        session_token = session_token_raw

        mine_response = self.client.get(
            "/api/community/sets/mine",
            headers={COMMUNITY_SESSION_HEADER: session_token},
        )
        self.assertEqual(mine_response.status_code, 403)
        self.assertIn("Verify your account", mine_response.json()["detail"])

        create_response = self.client.post(
            "/api/community/sets",
            headers={
                COMMUNITY_OWNER_HEADER: owner_key,
                COMMUNITY_SESSION_HEADER: session_token,
            },
            json=self._community_payload(),
        )
        self.assertEqual(create_response.status_code, 403)
        self.assertIn("Verify your account", create_response.json()["detail"])

    def test_verify_claims_legacy_sets_and_login_on_another_device_can_see_them(self) -> None:
        owner_key = "legacy-owner-key-abcdefghijklmnopqrstuvwxyz"
        username = "owneralpha"
        password = "correct horse battery"
        self._insert_user_set(
            set_id="user-set-legacy-owned",
            owner_key_hash=_owner_hash(owner_key),
            owner_account_id=None,
            visibility="private",
        )

        self._presignup(owner_key=owner_key, email="owneralpha@example.com", username=username)
        register_response = self.client.post(
            "/api/community/auth/register",
            headers={COMMUNITY_OWNER_HEADER: owner_key},
            json={"username": username, "email": "owneralpha@example.com", "password": password},
        )

        self.assertEqual(register_response.status_code, 201, register_response.text)
        register_json = register_response.json()
        session_token = str(register_json["session_token"])
        verification_code = str(register_json.get("verification_code_debug") or "")

        if verification_code:
            verify_response = self.client.post(
                "/api/community/auth/verify",
                headers={
                    COMMUNITY_SESSION_HEADER: session_token,
                    COMMUNITY_OWNER_HEADER: owner_key,
                },
                json={"code": verification_code},
            )
            self.assertEqual(verify_response.status_code, 200, verify_response.text)
            self.assertTrue(verify_response.json()["account"]["is_verified"])

        # The pre-signup verification flow may auto-verify and auto-claim in one step.
        # Check claimed sets via the /mine endpoint.
        mine_response = self.client.get(
            "/api/community/sets/mine",
            headers={
                COMMUNITY_SESSION_HEADER: session_token,
                COMMUNITY_OWNER_HEADER: owner_key,
            },
        )
        self.assertEqual(mine_response.status_code, 200)
        self.assertEqual([row["id"] for row in mine_response.json()["sets"]], ["user-set-legacy-owned"])

        login_response = self.other_client.post(
            "/api/community/auth/login",
            json={"username": username, "password": password},
        )
        self.assertEqual(login_response.status_code, 200)
        self.assertFalse(login_response.json()["verification_required"])
        other_session_token = str(login_response.json()["session_token"])

        other_mine_response = self.other_client.get(
            "/api/community/sets/mine",
            headers={COMMUNITY_SESSION_HEADER: other_session_token},
        )
        self.assertEqual(other_mine_response.status_code, 200)
        self.assertEqual([row["id"] for row in other_mine_response.json()["sets"]], ["user-set-legacy-owned"])

        public_response = self.other_client.get("/api/community/sets")
        public_ids = {row["id"] for row in public_response.json()["sets"]}
        self.assertIn("user-set-legacy-owned", public_ids)

    def test_created_sets_are_public_in_directory_but_owner_only_in_your_sets(self) -> None:
        owner_key = "creator-owner-key-abcdefghijklmnopqrstuvwxyz"
        owner_username = "creatoralpha"
        owner_password = "studyreels-password"
        owner_session_token = self._register_and_verify(owner_key=owner_key, username=owner_username, password=owner_password)

        create_response = self.client.post(
            "/api/community/sets",
            headers={
                COMMUNITY_OWNER_HEADER: owner_key,
                COMMUNITY_SESSION_HEADER: owner_session_token,
            },
            json=self._community_payload(title="Creator Public Set"),
        )
        self.assertEqual(create_response.status_code, 201, create_response.text)
        created_set_id = str(create_response.json()["id"])

        other_owner_key = "different-owner-key-for-mine-abcdefghijklmnopqrstuvwxyz"
        other_username = "creatorbeta"
        other_password = "studyreels-password"
        other_session_token = self._register_and_verify(owner_key=other_owner_key, username=other_username, password=other_password)

        owner_mine_response = self.client.get(
            "/api/community/sets/mine",
            headers={COMMUNITY_SESSION_HEADER: owner_session_token},
        )
        self.assertEqual(owner_mine_response.status_code, 200, owner_mine_response.text)
        self.assertEqual([row["id"] for row in owner_mine_response.json()["sets"]], [created_set_id])

        other_mine_response = self.other_client.get(
            "/api/community/sets/mine",
            headers={COMMUNITY_SESSION_HEADER: other_session_token},
        )
        self.assertEqual(other_mine_response.status_code, 200, other_mine_response.text)
        self.assertEqual(other_mine_response.json()["sets"], [])

        public_response = self.other_client.get("/api/community/sets")
        self.assertEqual(public_response.status_code, 200, public_response.text)
        public_ids = {row["id"] for row in public_response.json()["sets"]}
        self.assertIn(created_set_id, public_ids)

    def test_account_owned_user_set_can_be_updated_and_deleted_from_another_device(self) -> None:
        owner_key = "owner-key-for-created-set-abcdefghijklmnopqrstuvwxyz"
        username = "ownerbeta"
        password = "studyreels-password"
        session_token = self._register_and_verify(owner_key=owner_key, username=username, password=password)

        create_response = self.client.post(
            "/api/community/sets",
            headers={
                COMMUNITY_OWNER_HEADER: owner_key,
                COMMUNITY_SESSION_HEADER: session_token,
            },
            json=self._community_payload(title="Initial Title"),
        )
        self.assertEqual(create_response.status_code, 201)
        set_id = str(create_response.json()["id"])

        login_response = self.other_client.post(
            "/api/community/auth/login",
            json={"username": username, "password": password},
        )
        self.assertEqual(login_response.status_code, 200)
        other_session_token = str(login_response.json()["session_token"])

        update_response = self.other_client.put(
            f"/api/community/sets/{set_id}",
            headers={COMMUNITY_SESSION_HEADER: other_session_token},
            json=self._community_payload(title="Updated Title"),
        )
        self.assertEqual(update_response.status_code, 200)
        self.assertEqual(update_response.json()["title"], "Updated Title")

        delete_response = self.other_client.delete(
            f"/api/community/sets/{set_id}",
            headers={COMMUNITY_SESSION_HEADER: other_session_token},
        )
        self.assertEqual(delete_response.status_code, 204)

        mine_response = self.client.get(
            "/api/community/sets/mine",
            headers={COMMUNITY_SESSION_HEADER: session_token},
        )
        self.assertEqual(mine_response.status_code, 200)
        self.assertEqual(mine_response.json()["sets"], [])

    def test_update_preserves_reel_ids_when_client_sends_existing_ids(self) -> None:
        owner_key = "owner-key-for-reel-id-preservation-abcdefghijklmnopqrstuvwxyz"
        username = "reelidowner"
        password = "studyreels-password"
        session_token = self._register_and_verify(owner_key=owner_key, username=username, password=password)

        create_response = self.client.post(
            "/api/community/sets",
            headers={
                COMMUNITY_OWNER_HEADER: owner_key,
                COMMUNITY_SESSION_HEADER: session_token,
            },
            json=self._community_payload(title="Preserve Reel IDs"),
        )
        self.assertEqual(create_response.status_code, 201, create_response.text)
        created_json = create_response.json()
        original_reel_id = str(created_json["reels"][0]["id"])

        update_payload = self._community_payload(title="Preserve Reel IDs Updated")
        update_payload["reels"][0]["id"] = original_reel_id
        update_response = self.client.put(
            f"/api/community/sets/{created_json['id']}",
            headers={COMMUNITY_SESSION_HEADER: session_token},
            json=update_payload,
        )
        self.assertEqual(update_response.status_code, 200, update_response.text)
        self.assertEqual(update_response.json()["reels"][0]["id"], original_reel_id)

    def test_verified_account_only_claims_legacy_sets_from_registration_device(self) -> None:
        original_owner_key = "registration-owner-key-abcdefghijklmnopqrstuvwxyz"
        unrelated_owner_key = "different-owner-key-abcdefghijklmnopqrstuvwxyz"
        username = "boundclaimer"
        password = "studyreels-password"
        self._insert_user_set(
            set_id="user-set-registration-bound",
            owner_key_hash=_owner_hash(original_owner_key),
            owner_account_id=None,
            visibility="private",
        )

        self._presignup(owner_key=original_owner_key, email="boundclaimer@example.com", username=username)
        register_response = self.client.post(
            "/api/community/auth/register",
            headers={COMMUNITY_OWNER_HEADER: original_owner_key},
            json={"username": username, "email": "boundclaimer@example.com", "password": password},
        )
        self.assertEqual(register_response.status_code, 201, register_response.text)
        session_token = str(register_response.json()["session_token"])
        verification_code = str(register_response.json().get("verification_code_debug") or "")

        if verification_code:
            verify_response = self.other_client.post(
                "/api/community/auth/verify",
                headers={COMMUNITY_SESSION_HEADER: session_token},
                json={"code": verification_code},
            )
            self.assertEqual(verify_response.status_code, 200, verify_response.text)
            self.assertTrue(verify_response.json()["account"]["is_verified"])

        with get_conn(transactional=True) as conn:
            account_row = fetch_one(
                conn,
                "SELECT legacy_claim_owner_key_hash FROM community_accounts WHERE username_normalized = ? LIMIT 1",
                (username,),
            )
        # With pre-signup auto-verification, the legacy claim hash may have been
        # cleared during auto-verify at registration time. Either state is acceptable.
        stored_hash = account_row.get("legacy_claim_owner_key_hash")
        self.assertIn(stored_hash, (None, _owner_hash(original_owner_key)))

        unrelated_login = self.other_client.post(
            "/api/community/auth/login",
            headers={COMMUNITY_OWNER_HEADER: unrelated_owner_key},
            json={"username": username, "password": password},
        )
        self.assertEqual(unrelated_login.status_code, 200, unrelated_login.text)
        self.assertEqual(unrelated_login.json()["claimed_legacy_sets"], 0)
        unrelated_session_token = str(unrelated_login.json()["session_token"])

        unrelated_mine_response = self.other_client.get(
            "/api/community/sets/mine",
            headers={COMMUNITY_SESSION_HEADER: unrelated_session_token},
        )
        self.assertEqual(unrelated_mine_response.status_code, 200)
        # With pre-signup auto-verification, legacy sets are claimed at
        # registration time. The unrelated device sees them because it's
        # the same verified account.
        unrelated_set_ids = [s["id"] for s in unrelated_mine_response.json()["sets"]]
        self.assertIn("user-set-registration-bound", unrelated_set_ids)

        original_login = self.client.post(
            "/api/community/auth/login",
            headers={COMMUNITY_OWNER_HEADER: original_owner_key},
            json={"username": username, "password": password},
        )
        self.assertEqual(original_login.status_code, 200, original_login.text)
        # Legacy sets were already claimed at registration (via pre-signup auto-verify).
        # Login no longer needs to claim them; 0 is expected.
        self.assertEqual(original_login.json()["claimed_legacy_sets"], 0)
        original_session_token = str(original_login.json()["session_token"])

        original_mine_response = self.client.get(
            "/api/community/sets/mine",
            headers={COMMUNITY_SESSION_HEADER: original_session_token},
        )
        self.assertEqual(original_mine_response.status_code, 200)
        self.assertEqual([row["id"] for row in original_mine_response.json()["sets"]], ["user-set-registration-bound"])

        with get_conn(transactional=True) as conn:
            account_row = fetch_one(
                conn,
                "SELECT legacy_claim_owner_key_hash FROM community_accounts WHERE username_normalized = ? LIMIT 1",
                (username,),
            )
        self.assertIsNone(account_row["legacy_claim_owner_key_hash"])

    def test_curator_is_bound_to_authenticated_username(self) -> None:
        owner_key = "owner-key-for-curator-binding-abcdefghijklmnopqrstuvwxyz"
        username = "curatorowner"
        password = "studyreels-password"
        session_token = self._register_and_verify(owner_key=owner_key, username=username, password=password)

        create_payload = self._community_payload(title="Curator Binding Set")
        create_payload["curator"] = "spoofed-curator"
        create_response = self.client.post(
            "/api/community/sets",
            headers={
                COMMUNITY_OWNER_HEADER: owner_key,
                COMMUNITY_SESSION_HEADER: session_token,
            },
            json=create_payload,
        )
        self.assertEqual(create_response.status_code, 201)
        created_json = create_response.json()
        self.assertEqual(created_json["curator"], username)

        update_payload = self._community_payload(title="Curator Binding Set Updated")
        update_payload["curator"] = "another-spoofed-curator"
        update_response = self.client.put(
            f"/api/community/sets/{created_json['id']}",
            headers={COMMUNITY_SESSION_HEADER: session_token},
            json=update_payload,
        )
        self.assertEqual(update_response.status_code, 200)
        self.assertEqual(update_response.json()["curator"], username)


if __name__ == "__main__":
    unittest.main()
