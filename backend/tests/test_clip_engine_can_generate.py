"""
Tests for dry_run / can-generate parity on the clip-engine path (Task T6).

Asserts the three-point T6 contract:
  1. zero writes:  generate_reels(..., dry_run=True) → 0 reel-table rows,
                   clip_engine_run.clip NOT called, clip_engine_search.discover
                   WAS called, and the result list is non-empty when videos exist.
  2. non-empty probe: _probe_material_viability(...) → can_generate=True when
                      discover yields ≥1 video (default settings path).
  3. empty probe:  _probe_material_viability(...) → can_generate=False when
                   discover yields 0 videos.

Harness mirrors test_clip_engine_generate_reels.py: real pipeline with engine
surfaces mocked, temp file-backed SQLite DB.
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

# Skip import-time orphan sweep so tests don't touch /tmp during import.
os.environ.setdefault("REELAI_INGEST_SKIP_IMPORT_SWEEP", "1")

from backend.app import db as db_module  # noqa: E402
from backend.app.config import get_settings  # noqa: E402
import backend.app.main as main_module  # noqa: E402
from backend.app.ingestion import pipeline as pipeline_module  # noqa: E402

# 11-char YouTube-style id so it round-trips through extract_video_id.
VIDEO_ID = "cg6TestVid1"
MATERIAL_ID = "mat-t6-cg"
CONCEPT_ID = "concept-t6-cg"


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


def _empty_discover_result() -> dict:
    return {
        "corrected": "cellular respiration",
        "videos": [],
        "credits_used": 0,
        "warning": None,
    }


class ClipEngineCanGenerateTests(unittest.TestCase):
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

        # Bump rate limits so a single probe call never trips the per-platform cap.
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

    def _patch_engine_discover(self, discover_return: dict):
        """Patch both engine surfaces; only discover return value matters for dry_run."""
        mock_search = mock.patch.object(pipeline_module, "clip_engine_search")
        mock_run = mock.patch.object(pipeline_module, "clip_engine_run")
        search = mock_search.start()
        run = mock_run.start()
        self.addCleanup(mock_search.stop)
        self.addCleanup(mock_run.stop)
        search.discover.return_value = discover_return
        # clip should never be called in dry_run; configure it so tests catch it
        run.clip.side_effect = AssertionError("clip_engine_run.clip called during dry_run")
        return search, run

    # ------------------------------------------------------------------ #
    # T6-1: zero writes, run.clip not called, discover was called
    # ------------------------------------------------------------------ #
    def test_dry_run_zero_writes_clip_not_called_discover_called(self) -> None:
        search_mock, run_mock = self._patch_engine_discover(_discover_result())

        with db_module.get_conn() as conn:
            result = main_module.reel_service.generate_reels(
                conn,
                material_id=MATERIAL_ID,
                concept_id=None,
                num_reels=4,
                creative_commons_only=False,
                generation_id="gen-t6-dry",
                dry_run=True,
                retrieval_profile="bootstrap",
            )

        # Non-empty result when discover yields videos.
        self.assertTrue(result, "dry_run should return non-empty previews when discover yields videos")
        for item in result:
            self.assertIn("reel_id", item)
            self.assertIn("video_id", item)
            self.assertTrue(str(item["reel_id"]).startswith("dry-run-"))

        # Zero DB writes.
        with db_module.get_conn() as conn:
            rows = db_module.fetch_all(
                conn,
                "SELECT id FROM reels WHERE generation_id = ?",
                ("gen-t6-dry",),
            )
        self.assertEqual(len(rows), 0, "dry_run must write zero reel rows")

        # discover WAS called; run.clip was NOT called (side_effect would raise).
        search_mock.discover.assert_called()
        run_mock.clip.assert_not_called()

    # ------------------------------------------------------------------ #
    # T6-2: non-empty probe → can_generate True
    # ------------------------------------------------------------------ #
    def test_probe_can_generate_true_when_discover_has_videos(self) -> None:
        self._patch_engine_discover(_discover_result())

        with db_module.get_conn() as conn:
            result = main_module._probe_material_viability(
                conn,
                material_id=MATERIAL_ID,
                concept_id=None,
                creative_commons_only=False,
                fast_mode=False,
                min_relevance=None,
                video_pool_mode="short-first",
                preferred_video_duration="any",
                target_clip_duration_sec=main_module.DEFAULT_TARGET_CLIP_DURATION_SEC,
                target_clip_duration_min_sec=None,
                target_clip_duration_max_sec=None,
            )

        self.assertTrue(result.can_generate, "probe should return can_generate=True when discover yields videos")
        self.assertGreater(result.total_probed, 0, "total_probed should reflect discovered videos")
        self.assertGreater(result.passed_all, 0, "passed_all should be > 0 under default settings")

    # ------------------------------------------------------------------ #
    # T6-3: empty probe → can_generate False
    # ------------------------------------------------------------------ #
    def test_probe_can_generate_false_when_discover_empty(self) -> None:
        # discover returns no videos for all probe calls (initial + relaxed).
        self._patch_engine_discover(_empty_discover_result())

        with db_module.get_conn() as conn:
            result = main_module._probe_material_viability(
                conn,
                material_id=MATERIAL_ID,
                concept_id=None,
                creative_commons_only=False,
                fast_mode=False,
                min_relevance=None,
                video_pool_mode="short-first",
                preferred_video_duration="any",
                target_clip_duration_sec=main_module.DEFAULT_TARGET_CLIP_DURATION_SEC,
                target_clip_duration_min_sec=None,
                target_clip_duration_max_sec=None,
            )

        self.assertFalse(result.can_generate, "probe should return can_generate=False when discover yields no videos")
        self.assertEqual(result.total_probed, 0, "total_probed should be 0 when discover yields no videos")


if __name__ == "__main__":
    unittest.main()
