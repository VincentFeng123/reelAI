"""
Regression test for Finding #1: client pagination exclusion across the
video_id format break.

Clip-engine reels persist with a `yt:<id>`-prefixed DB `video_id`, while legacy
rows and every client use BARE 11-char ids (derived from video_url). The feed's
exclusion filter must normalize BOTH sides so a client that sends bare ids
excludes both a legacy bare row and a prefixed clip-engine row.
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

os.environ.setdefault("REELAI_INGEST_SKIP_IMPORT_SWEEP", "1")

from backend.app import db as db_module  # noqa: E402
from backend.app.config import get_settings  # noqa: E402
import backend.app.main as main_module  # noqa: E402


class RankedExclusionNormalizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.previous_data_dir = os.environ.get("DATA_DIR")
        os.environ["DATA_DIR"] = self.temp_dir.name
        self.addCleanup(self._restore_environment)
        db_module._db_ready = False
        get_settings.cache_clear()
        main_module.settings = get_settings()

    def _restore_environment(self) -> None:
        if self.previous_data_dir is None:
            os.environ.pop("DATA_DIR", None)
        else:
            os.environ["DATA_DIR"] = self.previous_data_dir
        db_module._db_ready = False
        get_settings.cache_clear()
        main_module.settings = get_settings()

    def test_bare_client_ids_exclude_both_legacy_and_prefixed_rows(self) -> None:
        ranked = [
            {"video_id": "LEGACYIDABC", "reel_id": "r-legacy"},   # legacy bare row
            {"video_id": "yt:CLIPENGIN1", "reel_id": "r-clip"},   # clip-engine prefixed row
            {"video_id": "yt:KEEPTHIS12", "reel_id": "r-keep"},   # NOT excluded
        ]
        with (
            mock.patch.object(main_module.reel_service, "ranked_feed", return_value=ranked),
            mock.patch.object(main_module, "is_video_alive", return_value=True),
            mock.patch.object(main_module, "_shape_request_page_reels", side_effect=lambda r, **kw: r),
            db_module.get_conn() as conn,
        ):
            result = main_module._ranked_request_reels(
                conn,
                material_id="mat-excl",
                fast_mode=False,
                generation_id=None,
                min_relevance=None,
                preferred_video_duration="any",
                target_clip_duration_sec=45,
                target_clip_duration_min_sec=None,
                target_clip_duration_max_sec=None,
                # client sends BARE ids for both the legacy and the clip-engine row
                exclude_video_ids=["LEGACYIDABC", "CLIPENGIN1"],
                page=1,
                limit=5,
            )

        kept_ids = {r["video_id"] for r in result}
        self.assertEqual(
            kept_ids,
            {"yt:KEEPTHIS12"},
            "both the bare legacy row and the prefixed clip-engine row must be excluded",
        )


if __name__ == "__main__":
    unittest.main()
