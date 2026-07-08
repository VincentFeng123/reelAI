"""
Tests for feed ranking, feedback score shifts, refinement generation swap,
and cache-version assertion (Task T5).

Proves that clip-engine reels produced by the T4 generate_reels path flow
correctly through the preserved surroundings: ranked_feed ranking/scoring,
record_feedback score shifts, and _activate_generation generation head swap.

Setup mirrors test_clip_engine_generate_reels.py (T4): real pipeline with
engine surfaces mocked, temp file-backed SQLite DB.
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

from backend.app import db as db_module  # noqa: E402
from backend.app.config import get_settings  # noqa: E402
import backend.app.main as main_module  # noqa: E402
from backend.app.ingestion import pipeline as pipeline_module  # noqa: E402
from backend.app.services.reels import ReelService  # noqa: E402

VIDEO_ID = "vidFEEDT501"
MATERIAL_ID = "mat-t5-feed"
CONCEPT_ID = "concept-t5-feed"


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


def _two_clip_engine_out(video_id: str = VIDEO_ID) -> dict:
    """Two relevance-surviving clips from one video's transcript."""
    return {
        "video_id": video_id,
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
                {
                    "start": 0.0,
                    "end": 45.0,
                    "text": "Cellular respiration releases energy in the mitochondria.",
                },
                {
                    "start": 50.0,
                    "end": 95.0,
                    "text": "The cellular respiration process continues in the cytoplasm.",
                },
            ],
            "words": [],
            "duration": 300.0,
        },
        "notes": "",
    }


class ClipEngineFeedRefineFeedbackTests(unittest.TestCase):
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

        # Bump rate limits so both generate_reels calls in the refinement test
        # never trip the per-platform cap.
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
                (
                    MATERIAL_ID,
                    "biology",
                    "Biology notes",
                    "text",
                    None,
                    "2026-07-06T00:00:00+00:00",
                ),
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
        """Patch discover + clip at the pipeline aliases; registers cleanup."""
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
    # T5-1: feed ranking
    # ------------------------------------------------------------------ #
    def test_feed_ranking_returns_scored_reels_in_descending_order(self) -> None:
        """ranked_feed returns clip-engine reels with a numeric score, descending."""
        self._patched_engine(_two_clip_engine_out())

        with db_module.get_conn() as conn:
            main_module.reel_service.generate_reels(
                conn,
                material_id=MATERIAL_ID,
                concept_id=None,
                num_reels=5,
                creative_commons_only=False,
                generation_id="gen-feed-1",
            )

        with db_module.get_conn() as conn:
            feed = main_module.reel_service.ranked_feed(
                conn, material_id=MATERIAL_ID, generation_id="gen-feed-1"
            )

        self.assertGreaterEqual(len(feed), 1, "ranked_feed must be non-empty after generate_reels")

        for item in feed:
            for key in ("reel_id", "video_id", "concept_id", "score"):
                self.assertIn(key, item, f"expected key '{key}' in ranked_feed item")
            self.assertIsInstance(item["score"], float, "score must be a float")

        scores = [item["score"] for item in feed]
        self.assertEqual(
            scores,
            sorted(scores, reverse=True),
            "ranked_feed items must be ordered by descending score",
        )

    # ------------------------------------------------------------------ #
    # T5-2: feedback shifts score
    # ------------------------------------------------------------------ #
    def test_feedback_shifts_score_up_and_down(self) -> None:
        """helpful feedback raises score; confusing feedback lowers it."""
        self._patched_engine(_two_clip_engine_out())

        with db_module.get_conn() as conn:
            main_module.reel_service.generate_reels(
                conn,
                material_id=MATERIAL_ID,
                concept_id=None,
                num_reels=5,
                creative_commons_only=False,
                generation_id="gen-feedback",
            )

        with db_module.get_conn() as conn:
            feed_before = main_module.reel_service.ranked_feed(
                conn, material_id=MATERIAL_ID, generation_id="gen-feedback"
            )

        self.assertGreaterEqual(len(feed_before), 2, "need >= 2 reels for feedback test")

        reel_a = feed_before[0]  # will receive helpful feedback -> score should rise
        reel_b = feed_before[1]  # will receive confusing feedback -> score should fall
        score_a_before = reel_a["score"]
        score_b_before = reel_b["score"]

        with db_module.get_conn(transactional=True) as conn:
            main_module.reel_service.record_feedback(
                conn,
                reel_id=reel_a["reel_id"],
                helpful=True,
                confusing=False,
                rating=5,
                saved=True,
            )

        with db_module.get_conn(transactional=True) as conn:
            main_module.reel_service.record_feedback(
                conn,
                reel_id=reel_b["reel_id"],
                helpful=False,
                confusing=True,
                rating=1,
                saved=False,
            )

        # Source fingerprint now differs (feedback_count changed) -> cache auto-busted.
        with db_module.get_conn() as conn:
            feed_after = main_module.reel_service.ranked_feed(
                conn, material_id=MATERIAL_ID, generation_id="gen-feedback"
            )

        after_by_id = {item["reel_id"]: item["score"] for item in feed_after}
        self.assertIn(reel_a["reel_id"], after_by_id, "reel_a should still be in feed after feedback")
        self.assertIn(reel_b["reel_id"], after_by_id, "reel_b should still be in feed after feedback")

        self.assertGreater(
            after_by_id[reel_a["reel_id"]],
            score_a_before,
            "helpful+saved+rating=5 feedback must raise reel_a's score",
        )
        self.assertLess(
            after_by_id[reel_b["reel_id"]],
            score_b_before,
            "confusing+rating=1 feedback must lower reel_b's score",
        )

    # ------------------------------------------------------------------ #
    # T5-3: refinement generation swap
    # ------------------------------------------------------------------ #
    def test_refinement_generation_swap_isolates_generations(self) -> None:
        """_activate_generation updates the head; gen-1 and gen-2 feeds don't bleed."""
        self._patched_engine(_two_clip_engine_out())

        # --- gen-1 ---
        with db_module.get_conn() as conn:
            main_module.reel_service.generate_reels(
                conn,
                material_id=MATERIAL_ID,
                concept_id=None,
                num_reels=5,
                creative_commons_only=False,
                generation_id="gen-refine-1",
            )

        with db_module.get_conn(transactional=True) as conn:
            main_module._activate_generation(
                conn,
                material_id=MATERIAL_ID,
                request_key="req-test",
                generation_id="gen-refine-1",
                retrieval_profile="bootstrap",
            )

        # --- gen-2: same engine mock is still active; different generation_id means
        #     no unique-index collision (key includes generation_id). ---
        with db_module.get_conn() as conn:
            main_module.reel_service.generate_reels(
                conn,
                material_id=MATERIAL_ID,
                concept_id=None,
                num_reels=5,
                creative_commons_only=False,
                generation_id="gen-refine-2",
            )

        with db_module.get_conn(transactional=True) as conn:
            main_module._activate_generation(
                conn,
                material_id=MATERIAL_ID,
                request_key="req-test",
                generation_id="gen-refine-2",
                retrieval_profile="bootstrap",
            )

        # Head must now point at gen-refine-2.
        with db_module.get_conn() as conn:
            head = db_module.fetch_one(
                conn,
                "SELECT active_generation_id FROM reel_generation_heads "
                "WHERE material_id = ? AND request_key = ?",
                (MATERIAL_ID, "req-test"),
            )
        self.assertIsNotNone(head, "reel_generation_heads row must exist after _activate_generation")
        self.assertEqual(head["active_generation_id"], "gen-refine-2")

        # gen-2 feed is non-empty.
        with db_module.get_conn() as conn:
            feed_gen2 = main_module.reel_service.ranked_feed(
                conn, material_id=MATERIAL_ID, generation_id="gen-refine-2"
            )
        self.assertGreaterEqual(len(feed_gen2), 1, "gen-2 ranked_feed must be non-empty")

        # gen-1 feed is still non-empty (not wiped by the swap).
        with db_module.get_conn() as conn:
            feed_gen1 = main_module.reel_service.ranked_feed(
                conn, material_id=MATERIAL_ID, generation_id="gen-refine-1"
            )
        self.assertGreaterEqual(len(feed_gen1), 1, "gen-1 ranked_feed must still be non-empty")

        # No reel_id overlap between gen-1 and gen-2 (generations must not bleed).
        gen1_ids = {item["reel_id"] for item in feed_gen1}
        gen2_ids = {item["reel_id"] for item in feed_gen2}
        self.assertFalse(gen1_ids & gen2_ids, "gen-1 and gen-2 reels must be completely isolated")

    # ------------------------------------------------------------------ #
    # T5-4: cache version guard
    # ------------------------------------------------------------------ #
    def test_ranked_feed_cache_version_is_6(self) -> None:
        """RANKED_FEED_CACHE_VERSION must equal 6 after the curation fixes."""
        self.assertEqual(
            ReelService.RANKED_FEED_CACHE_VERSION,
            6,
            "RANKED_FEED_CACHE_VERSION must be bumped to 6 (curation fixes — stale "
            "pre-curation ranked feeds must not be served after the quality overhaul).",
        )


if __name__ == "__main__":
    unittest.main()
