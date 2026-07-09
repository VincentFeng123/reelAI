"""API contract: create-material level field, PATCH level, feed fields.
FastAPI TestClient against a temp DB (pattern from test_clip_engine_contract).
Heavy ML services (embedding, concept extraction) are mocked out so tests are
offline and don't trigger native-code segfaults."""
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

os.environ.setdefault("REELAI_INGEST_SKIP_IMPORT_SWEEP", "1")

import numpy as np  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from backend.app import db as db_module  # noqa: E402
from backend.app.config import get_settings  # noqa: E402
import backend.app.main as main_module  # noqa: E402
from backend.app.main import app  # noqa: E402


def _fake_concepts(conn, text: str, subject_tag=None, max_concepts: int = 12):
    return (
        [{"id": str(uuid.uuid4()), "title": "Concept A", "keywords": ["a"], "summary": "s"}],
        [],
    )


def _fake_embed(conn, texts):
    return [np.zeros(384) for _ in texts]


class KnowledgeLevelApiTests(unittest.TestCase):
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
        self._patch_concepts.start()
        self._patch_embed.start()

        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        self.addCleanup(self._restore)

    def _restore(self) -> None:
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

    def test_patch_level_updates_and_resets_adjustment(self) -> None:
        created = self.client.post("/api/material", data={"subject_tag": "physics"})
        mid = created.json()["material_id"]
        with db_module.get_conn(transactional=True) as conn:
            conn.execute("UPDATE materials SET level_adjustment = 0.3 WHERE id = ?", (mid,))
        resp = self.client.patch(f"/api/materials/{mid}/level",
                                 json={"knowledge_level": "advanced"})
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["knowledge_level"], "advanced")
        self.assertAlmostEqual(body["effective_level_target"], 0.85)
        with db_module.get_conn() as conn:
            row = db_module.fetch_one(
                conn, "SELECT knowledge_level, level_adjustment FROM materials WHERE id = ?", (mid,))
        self.assertEqual(row["knowledge_level"], "advanced")
        self.assertEqual(float(row["level_adjustment"]), 0.0)

    def test_patch_unknown_material_404_and_bad_level_422(self) -> None:
        self.assertEqual(
            self.client.patch("/api/materials/nope/level",
                              json={"knowledge_level": "advanced"}).status_code, 404)
        created = self.client.post("/api/material", data={"subject_tag": "physics"})
        mid = created.json()["material_id"]
        self.assertEqual(
            self.client.patch(f"/api/materials/{mid}/level",
                              json={"knowledge_level": "expert"}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
