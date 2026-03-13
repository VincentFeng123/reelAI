import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from backend.app import db as db_module
from backend.app.config import get_settings
from backend.app.main import COMMUNITY_OWNER_HEADER, app


class CommunityAuthDbMigrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

        self.previous_data_dir = os.environ.get("DATA_DIR")
        self.previous_database_url = os.environ.get("DATABASE_URL")
        os.environ["DATA_DIR"] = self.temp_dir.name
        os.environ.pop("DATABASE_URL", None)
        self.addCleanup(self._restore_environment)

        get_settings.cache_clear()
        db_module._db_ready = False
        self.addCleanup(self._reset_db_state)

        self.db_path = Path(self.temp_dir.name) / "studyreels.db"
        self._seed_legacy_sqlite_schema()

        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def _restore_environment(self) -> None:
        if self.previous_data_dir is None:
            os.environ.pop("DATA_DIR", None)
        else:
            os.environ["DATA_DIR"] = self.previous_data_dir

        if self.previous_database_url is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = self.previous_database_url

    def _reset_db_state(self) -> None:
        db_module._db_ready = False
        get_settings.cache_clear()

    def _seed_legacy_sqlite_schema(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.executescript(
                """
                CREATE TABLE community_sets (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    reels_json TEXT NOT NULL,
                    reel_count INTEGER NOT NULL DEFAULT 0,
                    curator TEXT NOT NULL,
                    likes INTEGER NOT NULL DEFAULT 0,
                    learners INTEGER NOT NULL DEFAULT 1,
                    updated_label TEXT NOT NULL,
                    thumbnail_url TEXT NOT NULL,
                    featured INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );
                """
            )
            conn.commit()
        finally:
            conn.close()

    def test_register_migrates_legacy_sqlite_and_persists_account(self) -> None:
        response = self.client.post(
            "/api/community/auth/register",
            headers={COMMUNITY_OWNER_HEADER: "legacy-owner-key-abcdefghijklmnopqrstuvwxyz"},
            json={"username": "legacysignup", "email": "legacysignup@example.com", "password": "password123"},
        )

        self.assertEqual(response.status_code, 201, response.text)

        with db_module.get_conn(transactional=True) as conn:
            account = db_module.fetch_one(
                conn,
                "SELECT username, username_normalized FROM community_accounts WHERE username_normalized = ?",
                ("legacysignup",),
            )
        self.assertIsNotNone(account)
        self.assertEqual(account["username"], "legacysignup")

        conn = sqlite3.connect(self.db_path)
        try:
            set_columns = {
                row[1]
                for row in conn.execute("PRAGMA table_info(community_sets)").fetchall()
            }
            account_columns = {
                row[1]
                for row in conn.execute("PRAGMA table_info(community_accounts)").fetchall()
            }
        finally:
            conn.close()

        self.assertIn("updated_at", set_columns)
        self.assertIn("owner_key_hash", set_columns)
        self.assertIn("owner_account_id", set_columns)
        self.assertIn("visibility", set_columns)
        self.assertIn("email", account_columns)
        self.assertIn("email_normalized", account_columns)
        self.assertIn("verified_at", account_columns)
        self.assertIn("verification_code_hash", account_columns)
        self.assertIn("verification_expires_at", account_columns)
        self.assertIn("legacy_claim_owner_key_hash", account_columns)

    def test_unowned_legacy_sets_remain_public_after_migration(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO community_sets (
                    id,
                    title,
                    description,
                    tags_json,
                    reels_json,
                    reel_count,
                    curator,
                    likes,
                    learners,
                    updated_label,
                    thumbnail_url,
                    featured,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "user-set-legacy-public",
                    "Legacy Public Set",
                    "This description is long enough to satisfy the backend validation.",
                    "[\"systems\"]",
                    "[{\"id\":\"legacy-reel\",\"platform\":\"youtube\",\"source_url\":\"https://www.youtube.com/watch?v=dQw4w9WgXcQ\",\"embed_url\":\"https://www.youtube.com/embed/dQw4w9WgXcQ\"}]",
                    1,
                    "Legacy User",
                    0,
                    1,
                    "Last Edited: just now",
                    "https://example.com/thumb.png",
                    0,
                    "2026-01-01T00:00:00+00:00",
                ),
            )
            conn.commit()
        finally:
            conn.close()

        response = self.client.post(
            "/api/community/auth/register",
            headers={COMMUNITY_OWNER_HEADER: "legacy-owner-key-abcdefghijklmnopqrstuvwxyz"},
            json={"username": "legacyvisibility", "email": "legacyvisibility@example.com", "password": "password123"},
        )
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(response.json()["claimed_legacy_sets"], 0)

        public_response = self.client.get("/api/community/sets")
        self.assertEqual(public_response.status_code, 200, public_response.text)
        public_ids = {row["id"] for row in public_response.json()["sets"]}
        self.assertIn("user-set-legacy-public", public_ids)

        with db_module.get_conn(transactional=True) as conn:
            migrated_row = db_module.fetch_one(
                conn,
                "SELECT owner_key_hash, owner_account_id, visibility FROM community_sets WHERE id = ?",
                ("user-set-legacy-public",),
            )
        self.assertIsNotNone(migrated_row)
        self.assertEqual(migrated_row["owner_key_hash"], db_module.LEGACY_COMMUNITY_OWNER_HASH)
        self.assertIsNone(migrated_row["owner_account_id"])
        self.assertEqual(migrated_row["visibility"], db_module.PUBLIC_COMMUNITY_VISIBILITY)


if __name__ == "__main__":
    unittest.main()
