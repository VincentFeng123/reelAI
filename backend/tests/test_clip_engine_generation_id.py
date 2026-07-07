"""
Tests for generation_id threading through the ingest persistence layer (Task T1).

Uses init_db() schema directly so the UNIQUE index on
(material_id, COALESCE(generation_id, ''), video_id, t_start, t_end) is present.
Tests run against real SQLite in a temp directory — no mocks needed here.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("REELAI_INGEST_SKIP_IMPORT_SWEEP", "1")

from backend.app import db as db_module  # noqa: E402
from backend.app.config import get_settings  # noqa: E402
from backend.app.ingestion.persistence import load_existing_reel, upsert_reel_row  # noqa: E402


class GenerationIdPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.previous_data_dir = os.environ.get("DATA_DIR")
        os.environ["DATA_DIR"] = self.temp_dir.name
        self.addCleanup(self._restore_environment)
        db_module._db_ready = False
        get_settings.cache_clear()
        db_module.init_db()

    def _restore_environment(self) -> None:
        if self.previous_data_dir is None:
            os.environ.pop("DATA_DIR", None)
        else:
            os.environ["DATA_DIR"] = self.previous_data_dir
        db_module._db_ready = False
        get_settings.cache_clear()

    def _base_kwargs(self, **overrides) -> dict:
        """Base kwargs for upsert_reel_row — override specific fields as needed."""
        base = dict(
            reel_id="reel-t1-001",
            material_id="mat-t1",
            concept_id="con-t1",
            video_id="yt:abcdefg",
            video_url="https://www.youtube.com/embed/abcdefg",
            t_start=10.0,
            t_end=50.0,
            transcript_snippet="Hello world.",
            takeaways=["Point A"],
            base_score=1.0,
        )
        base.update(overrides)
        return base

    # ------------------------------------------------------------------ #
    # Test 1: generation scoping
    # ------------------------------------------------------------------ #

    def test_generation_scoping(self) -> None:
        """
        Reel persisted with generation_id='gen-x' is:
        - stored in the DB with that generation_id
        - found by load_existing_reel(..., generation_id='gen-x')
        - NOT found by load_existing_reel(..., generation_id='gen-y')
        """
        with db_module.get_conn(transactional=True) as conn:
            inserted = upsert_reel_row(conn, **self._base_kwargs(), generation_id="gen-x")
        self.assertTrue(inserted, "upsert should return True on fresh insert")

        # Row stores the correct generation_id.
        with db_module.get_conn() as conn:
            row = db_module.fetch_one(
                conn,
                "SELECT generation_id FROM reels WHERE id = ?",
                ("reel-t1-001",),
            )
        self.assertIsNotNone(row, "Row should exist in DB")
        self.assertEqual(row["generation_id"], "gen-x")

        # Matching generation finds the row.
        with db_module.get_conn() as conn:
            found = load_existing_reel(
                conn,
                material_id="mat-t1",
                video_id="yt:abcdefg",
                t_start=10.0,
                t_end=50.0,
                generation_id="gen-x",
            )
        self.assertIsNotNone(found, "load_existing_reel with matching gen must find the row")

        # Different generation does NOT find the row.
        with db_module.get_conn() as conn:
            not_found = load_existing_reel(
                conn,
                material_id="mat-t1",
                video_id="yt:abcdefg",
                t_start=10.0,
                t_end=50.0,
                generation_id="gen-y",
            )
        self.assertIsNone(not_found, "load_existing_reel with a different gen must return None")

    # ------------------------------------------------------------------ #
    # Test 2: backward-compat — None round-trip
    # ------------------------------------------------------------------ #

    def test_none_generation_id_backward_compat(self) -> None:
        """
        Reel persisted with generation_id=None (or omitted) stores NULL
        and is found by load_existing_reel(..., generation_id=None).
        Anonymous idempotency is unchanged.
        """
        with db_module.get_conn(transactional=True) as conn:
            inserted = upsert_reel_row(
                conn, **self._base_kwargs(reel_id="reel-t1-002"), generation_id=None
            )
        self.assertTrue(inserted)

        # Row stores NULL.
        with db_module.get_conn() as conn:
            row = db_module.fetch_one(
                conn,
                "SELECT generation_id FROM reels WHERE id = ?",
                ("reel-t1-002",),
            )
        self.assertIsNotNone(row)
        self.assertIsNone(row["generation_id"], "generation_id must be NULL when None was passed")

        # Anonymous lookup finds it.
        with db_module.get_conn() as conn:
            found = load_existing_reel(
                conn,
                material_id="mat-t1",
                video_id="yt:abcdefg",
                t_start=10.0,
                t_end=50.0,
                generation_id=None,
            )
        self.assertIsNotNone(found, "Anonymous (None) lookup must still find the row")

    # ------------------------------------------------------------------ #
    # Test 3: same clip, two generations → two distinct rows
    # ------------------------------------------------------------------ #

    def test_two_generations_two_rows(self) -> None:
        """
        The same (material_id, video_id, t_start, t_end) tuple under two
        different generation_ids must produce two distinct rows — the unique
        index on COALESCE(generation_id, '') treats them as separate.
        """
        with db_module.get_conn(transactional=True) as conn:
            ins1 = upsert_reel_row(
                conn, **self._base_kwargs(reel_id="reel-genx"), generation_id="gen-x"
            )
        self.assertTrue(ins1, "First insert (gen-x) must succeed")

        with db_module.get_conn(transactional=True) as conn:
            ins2 = upsert_reel_row(
                conn, **self._base_kwargs(reel_id="reel-geny"), generation_id="gen-y"
            )
        self.assertTrue(ins2, "Second insert (gen-y) must succeed — different generation, not a dupe")

        # Two rows exist for the same clip coordinates.
        with db_module.get_conn() as conn:
            count_row = db_module.fetch_one(
                conn,
                "SELECT COUNT(*) AS cnt FROM reels WHERE video_id = ? AND t_start = ? AND t_end = ?",
                ("yt:abcdefg", 10.0, 50.0),
            )
        self.assertEqual(count_row["cnt"], 2, "Two distinct generation rows must exist")


if __name__ == "__main__":
    unittest.main()
