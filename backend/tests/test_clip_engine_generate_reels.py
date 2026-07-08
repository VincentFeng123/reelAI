"""
Tests for the clip-engine-routed ReelService.generate_reels (Task T4).

The legacy search+topic_cut internals of generate_reels were hard-replaced with
a per-concept loop that routes each extracted concept through
IngestionPipeline.ingest_topic (multi-clip). This test drives the real pipeline
with the heavy engine surfaces mocked (clip_engine_search.discover +
clip_engine_run.clip), against a temp file-backed SQLite DB, and asserts the T4
contract: reels persist under generation_id, ranked_feed reads them back,
MULTIPLE clips per video survive, on_reel_created fires once per reel, the
num_reels cap holds, and the returned dicts carry the _create_reel key shape.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Skip the import-time orphan sweep so tests don't poke at /tmp during import.
os.environ.setdefault("REELAI_INGEST_SKIP_IMPORT_SWEEP", "1")

from fastapi.testclient import TestClient  # noqa: E402

from backend.app import db as db_module  # noqa: E402
from backend.app.config import get_settings  # noqa: E402
import backend.app.main as main_module  # noqa: E402
from backend.app.main import app, COMMUNITY_OWNER_HEADER  # noqa: E402
from backend.app.ingestion import pipeline as pipeline_module  # noqa: E402
from backend.app.ingestion.errors import RateLimitedError as IngestRateLimitedError  # noqa: E402

# 11-char YouTube-style id so the embed video_url round-trips through
# clip_engine.metadata.extract_video_id in _reel_attribution_to_dict.
VIDEO_ID = "vidABCDE123"
MATERIAL_ID = "mat-t4-gen"
CONCEPT_ID = "concept-t4-gen"


def _discover_result(video_id: str = VIDEO_ID) -> dict:
    return {
        "corrected": "cellular respiration",
        "videos": [
            {
                "id": video_id,
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "title": "Cellular Respiration Explained",
                "channel": "Bio Channel",
                "duration": 300,
                "thumbnail": "",
                "view_count": 1000,
                "upload_date": None,
            }
        ],
        "credits_used": 1,
        "warning": None,
    }


def _multi_clip_engine_out() -> dict:
    """Two relevance-surviving clips from one video's transcript. Each clip's
    window text contains the topic tokens ('cellular respiration') so
    clip_engine_bridge.filter_by_query keeps both."""
    return {
        "video_id": VIDEO_ID,
        "clips": [
            {
                "start": 0.0,
                "end": 45.0,
                "cut_end": 45.0,
                "title": "Cellular respiration overview",
                "facet": "biology",
                "reason": "core idea",
                "sequence_index": 0,
                "embed_url": "",
            },
            {
                "start": 50.0,
                "end": 95.0,
                "cut_end": 95.0,
                "title": "Cellular respiration in the cytoplasm",
                "facet": "biology",
                "reason": "second beat",
                "sequence_index": 1,
                "embed_url": "",
            },
        ],
        "transcript": {
            "segments": [
                {"start": 0.0, "end": 45.0, "text": "Cellular respiration releases energy in the mitochondria."},
                {"start": 50.0, "end": 95.0, "text": "The cellular respiration process continues in the cytoplasm."},
            ],
            "words": [],
            "duration": 300.0,
        },
        "notes": "",
    }


class ClipEngineGenerateReelsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.previous_data_dir = os.environ.get("DATA_DIR")
        os.environ["DATA_DIR"] = self.temp_dir.name
        self.addCleanup(self._restore_environment)
        db_module._db_ready = False
        get_settings.cache_clear()
        main_module.settings = get_settings()

        os.environ["REELAI_INGEST_SKIP_IMPORT_SWEEP"] = "1"

        # Bump rate limits so a single generation never trips the per-platform cap.
        from backend.app.ingestion.pipeline import _PlatformRateLimiter
        main_module.ingestion_pipeline._rate_limiter = _PlatformRateLimiter(
            overrides={"yt": (1000, 60.0)}
        )

        self._seed_material_and_concept()

    def _restore_environment(self) -> None:
        if self.previous_data_dir is None:
            os.environ.pop("DATA_DIR", None)
        else:
            os.environ["DATA_DIR"] = self.previous_data_dir
        db_module._db_ready = False
        get_settings.cache_clear()
        main_module.settings = get_settings()

    def _seed_material_and_concept(self) -> None:
        with db_module.get_conn(transactional=True) as conn:
            conn.execute(
                "INSERT INTO materials (id, subject_tag, raw_text, source_type, source_path, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (MATERIAL_ID, "biology", "Biology notes", "text", None, "2026-07-06T00:00:00+00:00"),
            )
            conn.execute(
                "INSERT INTO concepts (id, material_id, title, keywords_json, summary, embedding_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    CONCEPT_ID,
                    MATERIAL_ID,
                    "Cellular respiration",
                    '["cellular respiration", "mitochondria"]',
                    "How cells release energy.",
                    None,
                    "2026-07-06T00:01:00+00:00",
                ),
            )

    def _patched_engine(self, engine_out: dict):
        """Context managers patching discover + run.clip at the pipeline aliases."""
        mock_search = mock.patch.object(pipeline_module, "clip_engine_search")
        mock_run = mock.patch.object(pipeline_module, "clip_engine_run")
        search = mock_search.start()
        run = mock_run.start()
        self.addCleanup(mock_search.stop)
        self.addCleanup(mock_run.stop)
        search.discover.return_value = _discover_result()
        run.clip.return_value = engine_out
        return search, run

    # ------------------------------------------------------------------ #
    # Happy path: one concept -> one video -> MULTIPLE clips persisted
    # ------------------------------------------------------------------ #
    def test_generate_reels_multi_clip_per_video(self) -> None:
        self._patched_engine(_multi_clip_engine_out())
        collector: list[dict] = []

        with db_module.get_conn() as conn:
            result = main_module.reel_service.generate_reels(
                conn,
                material_id=MATERIAL_ID,
                concept_id=None,
                num_reels=5,
                creative_commons_only=False,
                generation_id="gen-1",
                on_reel_created=collector.append,
            )

        # 1. Reels landed in the reels table under generation_id="gen-1".
        with db_module.get_conn() as conn:
            rows = db_module.fetch_all(
                conn,
                "SELECT id, video_id FROM reels WHERE generation_id = ?",
                ("gen-1",),
            )
        self.assertGreaterEqual(len(rows), 2, "expected >=2 persisted reels under gen-1")

        # 3. MULTIPLE clips per video kept (>=2 reels from the one mocked video).
        video_ids = {r["video_id"] for r in rows}
        self.assertEqual(video_ids, {f"yt:{VIDEO_ID}"})
        self.assertGreaterEqual(len(rows), 2)

        # 2. ranked_feed reads them back under the same generation_id.
        with db_module.get_conn() as conn:
            feed = main_module.reel_service.ranked_feed(
                conn, material_id=MATERIAL_ID, generation_id="gen-1"
            )
        self.assertGreaterEqual(len(feed), 2)

        # 4. on_reel_created fired once per persisted reel.
        self.assertEqual(len(collector), len(rows))

        # 5. num_reels cap respected (<= 5).
        self.assertLessEqual(len(result), 5)

        # 6. Returned items carry the _create_reel dict key shape.
        self.assertTrue(result)
        first = result[0]
        for key in ("reel_id", "video_url", "concept_id", "score", "video_id"):
            self.assertIn(key, first)
        self.assertEqual(first["concept_id"], CONCEPT_ID)
        self.assertEqual(first["video_id"], VIDEO_ID)
        self.assertTrue(first["video_url"].startswith(f"https://www.youtube.com/embed/{VIDEO_ID}"))

    # ------------------------------------------------------------------ #
    # num_reels cap: a 2-clip video with num_reels=1 yields exactly 1 reel
    # ------------------------------------------------------------------ #
    def test_generate_reels_respects_num_reels_cap(self) -> None:
        self._patched_engine(_multi_clip_engine_out())
        collector: list[dict] = []

        with db_module.get_conn() as conn:
            result = main_module.reel_service.generate_reels(
                conn,
                material_id=MATERIAL_ID,
                concept_id=None,
                num_reels=1,
                creative_commons_only=False,
                generation_id="gen-cap",
                on_reel_created=collector.append,
            )

        with db_module.get_conn() as conn:
            rows = db_module.fetch_all(
                conn,
                "SELECT id FROM reels WHERE generation_id = ?",
                ("gen-cap",),
            )
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(collector), 1)

    # ------------------------------------------------------------------ #
    # dry_run: discover-only viability probe, zero DB writes, non-empty
    # ------------------------------------------------------------------ #
    def test_generate_reels_dry_run_zero_db_writes(self) -> None:
        self._patched_engine(_multi_clip_engine_out())

        with db_module.get_conn() as conn:
            result = main_module.reel_service.generate_reels(
                conn,
                material_id=MATERIAL_ID,
                concept_id=None,
                num_reels=4,
                creative_commons_only=False,
                generation_id="gen-dry",
                dry_run=True,
            )

        # Non-empty when videos exist.
        self.assertTrue(result)
        for item in result:
            self.assertIn("reel_id", item)
            self.assertIn("video_id", item)

        # Zero DB writes under dry_run.
        with db_module.get_conn() as conn:
            rows = db_module.fetch_all(
                conn,
                "SELECT id FROM reels WHERE generation_id = ?",
                ("gen-dry",),
            )
        self.assertEqual(len(rows), 0)

    # ------------------------------------------------------------------ #
    # Finding #3: refinement/extension excludes prior-generation video ids
    # ------------------------------------------------------------------ #
    def test_excludes_prior_generation_video_ids_from_discover(self) -> None:
        search, _run = self._patched_engine(_multi_clip_engine_out())

        # gen-1: persist reels for VIDEO_ID (stored as yt:<id>).
        with db_module.get_conn() as conn:
            main_module.reel_service.generate_reels(
                conn,
                material_id=MATERIAL_ID,
                concept_id=None,
                num_reels=5,
                creative_commons_only=False,
                generation_id="gen-1",
            )
        with db_module.get_conn() as conn:
            rows = db_module.fetch_all(
                conn, "SELECT DISTINCT video_id FROM reels WHERE generation_id = ?", ("gen-1",)
            )
        self.assertEqual({r["video_id"] for r in rows}, {f"yt:{VIDEO_ID}"})

        search.discover.reset_mock()

        # gen-2 excludes gen-1 → discover must receive gen-1's BARE video id.
        with db_module.get_conn() as conn:
            main_module.reel_service.generate_reels(
                conn,
                material_id=MATERIAL_ID,
                concept_id=None,
                num_reels=5,
                creative_commons_only=False,
                generation_id="gen-2",
                exclude_generation_ids=["gen-1"],
            )

        self.assertTrue(search.discover.called)
        passed = search.discover.call_args.kwargs.get("exclude_video_ids") or []
        self.assertIn(VIDEO_ID, passed)
        self.assertNotIn(f"yt:{VIDEO_ID}", passed)

    # ------------------------------------------------------------------ #
    # Finding #4a: one concept's engine failure must not abort the run
    # ------------------------------------------------------------------ #
    def test_one_concept_failure_does_not_abort_generation(self) -> None:
        # Seed a second concept so two concepts are processed.
        with db_module.get_conn(transactional=True) as conn:
            conn.execute(
                "INSERT INTO concepts (id, material_id, title, keywords_json, summary, embedding_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    "concept-t4-gen-2",
                    MATERIAL_ID,
                    "Cellular respiration stages",
                    '["cellular respiration", "atp"]',
                    "The stages of respiration.",
                    None,
                    "2026-07-06T00:02:00+00:00",
                ),
            )

        search, run = self._patched_engine(_multi_clip_engine_out())
        call_count = {"n": 0}

        def _discover_side_effect(topic, limit, exclude_video_ids=None, **kw):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("engine down for the first concept")
            return _discover_result()

        search.discover.side_effect = _discover_side_effect

        with db_module.get_conn() as conn:
            result = main_module.reel_service.generate_reels(
                conn,
                material_id=MATERIAL_ID,
                concept_id=None,
                num_reels=5,
                creative_commons_only=False,
                generation_id="gen-partial",
            )

        # The surviving concept still produced reels despite the first concept failing.
        self.assertTrue(result, "a failing concept must not abort the whole generation")
        with db_module.get_conn() as conn:
            rows = db_module.fetch_all(
                conn, "SELECT id FROM reels WHERE generation_id = ?", ("gen-partial",)
            )
        self.assertGreaterEqual(len(rows), 1)

    # ------------------------------------------------------------------ #
    # Finding #4b: rate-limit surfaces as HTTP 429 at the generate endpoint
    # ------------------------------------------------------------------ #
    def test_generate_endpoint_maps_rate_limit_to_429(self) -> None:
        client = TestClient(app)
        self.addCleanup(client.close)
        with mock.patch.object(
            main_module,
            "_ensure_generation_for_request",
            side_effect=IngestRateLimitedError("ingestion rate limit", retry_after_sec=42),
        ):
            resp = client.post(
                "/api/reels/generate",
                json={"material_id": MATERIAL_ID, "num_reels": 3},
                headers={COMMUNITY_OWNER_HEADER: "owner-key-abcdefghijklmnopqrstuvwxyz"},
            )
        self.assertEqual(resp.status_code, 429, resp.text)
        self.assertIn("retry-after", {k.lower() for k in resp.headers})


if __name__ == "__main__":
    unittest.main()
