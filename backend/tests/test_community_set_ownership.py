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

get_settings.cache_clear()

from backend.app.db import dumps_json, execute_modify, fetch_all, get_conn, now_iso, upsert
from backend.app.main import COMMUNITY_OWNER_HEADER, app


def _owner_hash(owner_key: str) -> str:
    return hashlib.sha256(owner_key.encode("utf-8")).hexdigest()


class CommunitySetOwnershipTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        with get_conn(transactional=True) as conn:
            execute_modify(conn, "DELETE FROM community_sets")

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

    def _insert_user_set(self, *, set_id: str, owner_key_hash: str) -> None:
        timestamp = now_iso()
        with get_conn(transactional=True) as conn:
            upsert(
                conn,
                "community_sets",
                {
                    "id": set_id,
                    "title": "Original Title",
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
                    "featured": 0,
                    "created_at": timestamp,
                    "updated_at": timestamp,
                },
            )

    def test_legacy_user_set_cannot_be_claimed_via_update(self) -> None:
        set_id = "user-set-legacy-update"
        attacker_key = "attacker-owner-key-1234567890"
        self._insert_user_set(set_id=set_id, owner_key_hash="")

        response = self.client.put(
            f"/api/community/sets/{set_id}",
            headers={COMMUNITY_OWNER_HEADER: attacker_key},
            json=self._community_payload(title="Attacker Title"),
        )

        self.assertEqual(response.status_code, 403)
        self.assertIn("predates ownership tracking", response.json()["detail"])
        with get_conn() as conn:
            row = fetch_all(conn, "SELECT title, owner_key_hash FROM community_sets WHERE id = ?", (set_id,))[0]
        self.assertEqual(row["title"], "Original Title")
        self.assertNotEqual(str(row["owner_key_hash"] or "").strip(), _owner_hash(attacker_key))

    def test_legacy_user_set_cannot_be_claimed_via_delete(self) -> None:
        set_id = "user-set-legacy-delete"
        attacker_key = "attacker-owner-key-abcdefghij"
        self._insert_user_set(set_id=set_id, owner_key_hash="")

        response = self.client.delete(
            f"/api/community/sets/{set_id}",
            headers={COMMUNITY_OWNER_HEADER: attacker_key},
        )

        self.assertEqual(response.status_code, 403)
        self.assertIn("predates ownership tracking", response.json()["detail"])
        with get_conn() as conn:
            row = fetch_all(conn, "SELECT id FROM community_sets WHERE id = ?", (set_id,))
        self.assertEqual(len(row), 1)

    def test_owned_user_set_can_still_be_updated_and_deleted(self) -> None:
        set_id = "user-set-owned"
        owner_key = "real-owner-key-abcdefghijklmnopqrstuvwxyz"
        self._insert_user_set(set_id=set_id, owner_key_hash=_owner_hash(owner_key))

        update_response = self.client.put(
            f"/api/community/sets/{set_id}",
            headers={COMMUNITY_OWNER_HEADER: owner_key},
            json=self._community_payload(title="Updated Title"),
        )

        self.assertEqual(update_response.status_code, 200)
        self.assertEqual(update_response.json()["title"], "Updated Title")

        delete_response = self.client.delete(
            f"/api/community/sets/{set_id}",
            headers={COMMUNITY_OWNER_HEADER: owner_key},
        )

        self.assertEqual(delete_response.status_code, 204)
        with get_conn() as conn:
            row = fetch_all(conn, "SELECT id FROM community_sets WHERE id = ?", (set_id,))
        self.assertEqual(row, [])


if __name__ == "__main__":
    unittest.main()
