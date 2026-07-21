"""API contract: create-material level field, PATCH level, feed fields.
FastAPI TestClient against a temp DB (pattern from test_clip_engine_contract).
Heavy ML services (embedding, concept extraction) are mocked out so tests are
offline and don't trigger native-code segfaults."""
import os
import sys
import tempfile
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("REELAI_INGEST_SKIP_IMPORT_SWEEP", "1")

import numpy as np  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402
from starlette.requests import Request  # noqa: E402

from backend.app import db as db_module  # noqa: E402
from backend.app.config import get_settings  # noqa: E402
import backend.app.main as main_module  # noqa: E402
from backend.app.main import app, COMMUNITY_OWNER_HEADER  # noqa: E402


class _PostgresFailure(RuntimeError):
    def __init__(self, sqlstate: str) -> None:
        self.sqlstate = sqlstate
        super().__init__(f"postgres transaction failure ({sqlstate})")


def _fake_concepts(conn, text: str, subject_tag=None, max_concepts: int = 12):
    return (
        [{"id": str(uuid.uuid4()), "title": "Concept A", "keywords": ["a"], "summary": "s"}],
        [],
    )


def _fake_embed(conn, texts):
    return [np.zeros(384) for _ in texts]


class KnowledgeLevelApiTests(unittest.TestCase):
    OWNER = "owner-key-abcdefghijklmnopqrstuvwxyz"

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self._prev = os.environ.get("DATA_DIR")
        os.environ["DATA_DIR"] = self.temp_dir.name
        db_module._db_ready = False
        get_settings.cache_clear()
        main_module.settings = get_settings()

        # Patch ML services so tests never call native embedding code.
        self._patch_concepts = mock.patch.object(
            main_module.material_intelligence_service,
            "extract_concepts_and_objectives",
            side_effect=_fake_concepts,
        )
        self._patch_embed = mock.patch.object(
            main_module.embedding_service,
            "embed_texts",
            side_effect=_fake_embed,
        )
        self._patch_provider_account = mock.patch.object(
            main_module,
            "_require_verified_provider_account",
            return_value={"id": "knowledge-level-test-account"},
        )
        self._patch_concepts.start()
        self._patch_embed.start()
        self._patch_provider_account.start()

        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        self._patch_provider_account.stop()
        self._patch_embed.stop()
        self._patch_concepts.stop()
        if self._prev is None:
            os.environ.pop("DATA_DIR", None)
        else:
            os.environ["DATA_DIR"] = self._prev
        db_module._db_ready = False
        get_settings.cache_clear()

    def test_create_material_stores_level(self) -> None:
        resp = self.client.post("/api/material",
                                data={"subject_tag": "physics", "knowledge_level": "advanced"})
        self.assertEqual(resp.status_code, 200, resp.text)
        material_id = resp.json()["material_id"]
        with db_module.get_conn() as conn:
            row = db_module.fetch_one(
                conn, "SELECT knowledge_level FROM materials WHERE id = ?", (material_id,))
        self.assertEqual(row["knowledge_level"], "advanced")

    def test_create_material_default_beginner_and_invalid_422(self) -> None:
        ok = self.client.post("/api/material", data={"subject_tag": "physics"})
        self.assertEqual(ok.status_code, 200)
        mid = ok.json()["material_id"]
        with db_module.get_conn() as conn:
            row = db_module.fetch_one(conn, "SELECT knowledge_level FROM materials WHERE id = ?", (mid,))
        self.assertEqual(row["knowledge_level"], "beginner")
        bad = self.client.post("/api/material",
                               data={"subject_tag": "physics", "knowledge_level": "expert"})
        self.assertEqual(bad.status_code, 422)

    def test_create_material_persistence_retry_does_not_replay_provider(self) -> None:
        real_upsert = main_module.upsert
        material_upserts = 0
        provider = mock.Mock(side_effect=_fake_concepts)

        def flaky_upsert(conn, table, data, *args, **kwargs):
            nonlocal material_upserts
            if table == "materials":
                material_upserts += 1
                if material_upserts == 1:
                    raise _PostgresFailure("40001")
            return real_upsert(conn, table, data, *args, **kwargs)

        with mock.patch.object(
            main_module.material_intelligence_service,
            "extract_concepts_and_objectives",
            provider,
        ), mock.patch.object(main_module, "upsert", side_effect=flaky_upsert):
            response = self.client.post(
                "/api/material",
                data={"subject_tag": "transaction retries"},
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(material_upserts, 2)
        self.assertEqual(provider.call_count, 1)

    def test_create_material_lost_commit_ack_converges_without_duplicates(self) -> None:
        real_get_conn = main_module.get_conn
        transaction_exits = 0
        provider = mock.Mock(side_effect=_fake_concepts)

        @contextmanager
        def connection(*, transactional: bool = False):
            nonlocal transaction_exits
            with real_get_conn(transactional=transactional) as conn:
                yield conn
            if transactional:
                transaction_exits += 1
                if transaction_exits == 3:
                    raise _PostgresFailure("08006")

        with mock.patch.object(main_module, "get_conn", connection), mock.patch.object(
            main_module.material_intelligence_service,
            "extract_concepts_and_objectives",
            provider,
        ):
            response = self.client.post(
                "/api/material",
                data={"subject_tag": "commit acknowledgement"},
            )

        self.assertEqual(response.status_code, 200, response.text)
        material_id = response.json()["material_id"]
        self.assertEqual(transaction_exits, 4)
        self.assertEqual(provider.call_count, 1)
        with real_get_conn() as conn:
            material_count = db_module.fetch_one(
                conn,
                "SELECT COUNT(*) AS count FROM materials WHERE id = ?",
                (material_id,),
            )["count"]
            concept_count = db_module.fetch_one(
                conn,
                "SELECT COUNT(*) AS count FROM concepts WHERE material_id = ?",
                (material_id,),
            )["count"]
            chunk_count = db_module.fetch_one(
                conn,
                "SELECT COUNT(*) AS count FROM material_chunks WHERE material_id = ?",
                (material_id,),
            )["count"]
        self.assertEqual((material_count, concept_count, chunk_count), (1, 1, 1))

    def test_create_file_material_retry_reuses_one_published_object(self) -> None:
        real_upsert = main_module.upsert
        material_upserts = 0
        save_bytes = mock.Mock(return_value="s3://test/uploads/stable-notes.txt")

        def fail_after_publication(conn, table, data, *args, **kwargs):
            nonlocal material_upserts
            if table == "materials":
                material_upserts += 1
                if material_upserts == 1:
                    raise _PostgresFailure("40001")
            return real_upsert(conn, table, data, *args, **kwargs)

        with mock.patch.object(
            main_module.storage,
            "save_bytes",
            save_bytes,
        ), mock.patch.object(
            main_module,
            "upsert",
            side_effect=fail_after_publication,
        ), mock.patch.object(main_module, "_enforce_rate_limit"):
            response = self.client.post(
                "/api/material",
                headers={"Idempotency-Key": "file-publication-db-retry"},
                files={
                    "file": (
                        "notes.txt",
                        b"Newton's laws and balanced forces",
                        "text/plain",
                    )
                },
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(material_upserts, 2)
        save_bytes.assert_called_once_with(
            b"Newton's laws and balanced forces",
            "notes.txt",
        )
        with db_module.get_conn() as conn:
            row = db_module.fetch_one(
                conn,
                "SELECT source_path FROM materials WHERE id = ?",
                (response.json()["material_id"],),
            )
        self.assertEqual(row["source_path"], "s3://test/uploads/stable-notes.txt")

    def test_patch_level_updates_and_resets_adjustment(self) -> None:
        headers = {COMMUNITY_OWNER_HEADER: self.OWNER}
        created = self.client.post(
            "/api/material", data={"subject_tag": "physics"}, headers=headers
        )
        mid = created.json()["material_id"]
        with db_module.get_conn(transactional=True) as conn:
            conn.execute("UPDATE materials SET level_adjustment = 0.3 WHERE id = ?", (mid,))
            conn.execute(
                "UPDATE learner_material_progress SET global_adjustment = 0.18 "
                "WHERE material_id = ?",
                (mid,),
            )
        resp = self.client.patch(f"/api/materials/{mid}/level",
                                 json={"knowledge_level": "advanced"}, headers=headers)
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["knowledge_level"], "advanced")
        self.assertAlmostEqual(body["effective_level_target"], 0.85)
        with db_module.get_conn() as conn:
            material = db_module.fetch_one(
                conn, "SELECT knowledge_level, level_adjustment FROM materials WHERE id = ?", (mid,))
            progress = db_module.fetch_one(
                conn,
                "SELECT selected_level, global_adjustment, difficulty_reset_at, feedback_revision "
                "FROM learner_material_progress WHERE material_id = ?",
                (mid,),
            )
        # Material columns remain legacy defaults; the learner row owns new changes.
        self.assertEqual(material["knowledge_level"], "beginner")
        self.assertEqual(float(material["level_adjustment"]), 0.3)
        self.assertEqual(progress["selected_level"], "advanced")
        self.assertEqual(float(progress["global_adjustment"]), 0.0)
        self.assertTrue(progress["difficulty_reset_at"])
        self.assertEqual(int(progress["feedback_revision"]), 1)

    def test_feed_reports_level_fields(self) -> None:
        created = self.client.post("/api/material",
                                   data={"subject_tag": "physics", "knowledge_level": "advanced"})
        mid = created.json()["material_id"]
        resp = self.client.get(
            f"/api/feed?material_id={mid}",
            headers={COMMUNITY_OWNER_HEADER: self.OWNER},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["knowledge_level"], "advanced")
        self.assertAlmostEqual(body["effective_level_target"], 0.85)

    def test_levels_are_isolated_by_anonymous_owner(self) -> None:
        owner_a = {COMMUNITY_OWNER_HEADER: self.OWNER}
        owner_b = {COMMUNITY_OWNER_HEADER: "another-owner-key-abcdefghijklmnopqrstuvwxyz"}
        created = self.client.post(
            "/api/material",
            data={"subject_tag": "physics", "knowledge_level": "intermediate"},
            headers=owner_a,
        )
        mid = created.json()["material_id"]
        changed = self.client.patch(
            f"/api/materials/{mid}/level",
            json={"knowledge_level": "advanced"},
            headers=owner_a,
        )
        self.assertEqual(changed.status_code, 200, changed.text)

        feed_a = self.client.get(f"/api/feed?material_id={mid}", headers=owner_a)
        feed_b = self.client.get(f"/api/feed?material_id={mid}", headers=owner_b)
        self.assertEqual(feed_a.status_code, 200, feed_a.text)
        self.assertEqual(feed_b.status_code, 200, feed_b.text)
        self.assertEqual(feed_a.json()["knowledge_level"], "advanced")
        self.assertEqual(feed_b.json()["knowledge_level"], "intermediate")

    def test_authenticated_account_identity_takes_precedence_over_owner(self) -> None:
        request = Request(
            {
                "type": "http",
                "method": "GET",
                "path": "/",
                "headers": [
                    (COMMUNITY_OWNER_HEADER.lower().encode(), self.OWNER.encode()),
                ],
            }
        )
        with (
            db_module.get_conn() as conn,
            mock.patch.object(
                main_module, "_try_get_community_account", return_value={"id": "account-1"}
            ),
        ):
            learner_id = main_module._resolve_learner_identity(conn, request)
        self.assertEqual(learner_id, "account:account-1")

    def test_feedback_is_unique_per_learner_and_reel(self) -> None:
        created = self.client.post(
            "/api/material", data={"subject_tag": "physics"},
            headers={COMMUNITY_OWNER_HEADER: self.OWNER},
        )
        mid = created.json()["material_id"]
        reel_id = "shared-feedback-reel"
        with db_module.get_conn(transactional=True) as conn:
            concept = db_module.fetch_one(
                conn, "SELECT id FROM concepts WHERE material_id = ? LIMIT 1", (mid,)
            )
            db_module.upsert(
                conn,
                "videos",
                {
                    "id": "shared-video",
                    "title": "Shared video",
                    "channel_title": "Teacher",
                    "duration_sec": 120,
                    "created_at": db_module.now_iso(),
                },
            )
            db_module.upsert(
                conn,
                "reels",
                {
                    "id": reel_id,
                    "material_id": mid,
                    "concept_id": concept["id"],
                    "video_id": "shared-video",
                    "video_url": "https://youtu.be/shared-video",
                    "t_start": 1.0,
                    "t_end": 21.0,
                    "transcript_snippet": "A complete explanation.",
                    "takeaways_json": "[]",
                    "base_score": 1.0,
                    "created_at": db_module.now_iso(),
                },
            )
        owner_a = {COMMUNITY_OWNER_HEADER: self.OWNER}
        owner_b = {COMMUNITY_OWNER_HEADER: "another-owner-key-abcdefghijklmnopqrstuvwxyz"}
        got_it = self.client.post(
            "/api/reels/feedback",
            json={"reel_id": reel_id, "helpful": True, "confusing": False, "rating": 5, "saved": True},
            headers=owner_a,
        )
        need_help = self.client.post(
            "/api/reels/feedback",
            json={"reel_id": reel_id, "helpful": False, "confusing": True, "rating": 2, "saved": False},
            headers=owner_b,
        )
        contradictory = self.client.post(
            "/api/reels/feedback",
            json={"reel_id": reel_id, "helpful": True, "confusing": True},
            headers=owner_a,
        )
        self.assertEqual(got_it.status_code, 200, got_it.text)
        self.assertEqual(need_help.status_code, 200, need_help.text)
        self.assertEqual(contradictory.status_code, 422, contradictory.text)
        with db_module.get_conn() as conn:
            rows = db_module.fetch_all(
                conn,
                "SELECT helpful, confusing, saved FROM reel_feedback WHERE reel_id = ? ORDER BY helpful DESC",
                (reel_id,),
            )
        self.assertEqual(len(rows), 2)
        self.assertEqual((rows[0]["helpful"], rows[0]["confusing"], rows[0]["saved"]), (1, 0, 1))
        self.assertEqual((rows[1]["helpful"], rows[1]["confusing"], rows[1]["saved"]), (0, 1, 0))

    def test_patch_unknown_material_404_and_bad_level_422(self) -> None:
        self.assertEqual(
            self.client.patch("/api/materials/nope/level",
                              json={"knowledge_level": "advanced"}).status_code, 404)
        created = self.client.post("/api/material", data={"subject_tag": "physics"})
        mid = created.json()["material_id"]
        self.assertEqual(
            self.client.patch(f"/api/materials/{mid}/level",
                              json={"knowledge_level": "expert"}).status_code, 422)

    def test_identityless_legacy_patch_updates_material_default(self) -> None:
        created = self.client.post(
            "/api/material", data={"subject_tag": "physics", "knowledge_level": "beginner"}
        )
        mid = created.json()["material_id"]
        changed = self.client.patch(
            f"/api/materials/{mid}/level", json={"knowledge_level": "intermediate"}
        )
        self.assertEqual(changed.status_code, 200, changed.text)
        fresh_owner_feed = self.client.get(
            f"/api/feed?material_id={mid}",
            headers={COMMUNITY_OWNER_HEADER: "fresh-owner-key-abcdefghijklmnopqrstuvwxyz"},
        )
        self.assertEqual(fresh_owner_feed.status_code, 200, fresh_owner_feed.text)
        self.assertEqual(fresh_owner_feed.json()["knowledge_level"], "intermediate")


if __name__ == "__main__":
    unittest.main()
