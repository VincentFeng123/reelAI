"""
Tests for the multi-clip `IngestionPipeline.ingest_topic` (Task T2).

The material→reels path routes ONE study concept through the clip engine and
persists MULTIPLE relevance-surviving clips per video (unlike `ingest_search`'s
one-best `pick_best_clip`).

Strategy mirrors `test_clip_engine_ingest_url.py`: mock the heavy external
surfaces (`clip_engine_search.discover`, `clip_engine_run.clip`) at their
pipeline import aliases so the test runs offline in <1s. `_persist_ingest`
writes to a temp SQLite DB.
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
from backend.app.ingestion import pipeline as pipeline_module  # noqa: E402
from backend.app.ingestion.models import ReelOutWithAttribution  # noqa: E402


# --------------------------------------------------------------------- #
# Fake discover + engine output (single query token "photosynthesis" so the
# per-clip token-overlap relevance filter keeps every clip built below).
# --------------------------------------------------------------------- #

TOPIC = "photosynthesis"

VID_A = {
    "id": "vidAAAAAAAA",
    "url": "https://www.youtube.com/watch?v=vidAAAAAAAA",
    "title": "Photosynthesis 101",
    "channel": "BioChan",
    "duration": 600.0,
    "thumbnail": "",
    "view_count": 123,
    "upload_date": "20230101",
}
VID_B = {
    "id": "vidBBBBBBBB",
    "url": "https://www.youtube.com/watch?v=vidBBBBBBBB",
    "title": "More biology",
    "channel": "BioChan",
    "duration": 500.0,
    "thumbnail": "",
    "view_count": 55,
    "upload_date": "20230102",
}
POOL = [VID_A, VID_B]

# video_id -> list of (start, end) windows the fake engine returns
CLIP_WINDOWS = {
    "vidAAAAAAAA": [(30.0, 75.0), (120.0, 165.0)],  # 2 surviving clips
    "vidBBBBBBBB": [(10.0, 55.0)],  # 1 surviving clip
}


def _build_engine_out(video_id: str) -> dict:
    """Fresh engine_out each call (clip dicts are mutated by filter_by_query)."""
    windows = CLIP_WINDOWS[video_id]
    clips = []
    segments = []
    for i, (start, end) in enumerate(windows):
        clips.append(
            {
                "start": start,
                "end": end,
                "cut_end": end,
                "title": f"Photosynthesis part {i}",
                "facet": "",
                "reason": "",
                "sequence_index": i,
                "embed_url": f"https://www.youtube.com/embed/{video_id}?start={int(start)}&end={int(end)}",
            }
        )
        # In-window transcript text carries the query token so the clip survives.
        segments.append(
            {"start": start, "end": end, "text": f"Here we explain photosynthesis part {i}."}
        )
    return {
        "video_id": video_id,
        "clips": clips,
        "transcript": {"segments": segments, "words": [], "duration": 600.0},
        "notes": "",
    }


def _discover_side_effect(topic, limit, exclude_video_ids=None, **kw):
    excl = set(exclude_video_ids or [])
    videos = [v for v in POOL if v["id"] not in excl][:limit]
    return {"corrected": topic, "videos": videos, "credits_used": 0, "warning": None}


def _clip_side_effect(url, topic=None, settings=None):
    vid = next(v["id"] for v in POOL if v["url"] == url)
    return _build_engine_out(vid)


class _Patched:
    """Context manager returning (mock_search, mock_run)."""

    def __enter__(self):
        self._search = mock.patch.object(pipeline_module, "clip_engine_search")
        self._run = mock.patch.object(pipeline_module, "clip_engine_run")
        self.mock_search = self._search.start()
        self.mock_run = self._run.start()
        self.mock_search.discover.side_effect = _discover_side_effect
        self.mock_run.clip.side_effect = _clip_side_effect
        return self.mock_search, self.mock_run

    def __exit__(self, *exc):
        self._run.stop()
        self._search.stop()
        return False


# --------------------------------------------------------------------- #
# TestCase
# --------------------------------------------------------------------- #


class IngestTopicTests(unittest.TestCase):
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

        from backend.app.ingestion.pipeline import _PlatformRateLimiter
        main_module.ingestion_pipeline._rate_limiter = _PlatformRateLimiter(
            overrides={"yt": (1000, 60.0)}
        )

    def _restore_environment(self) -> None:
        if self.previous_data_dir is None:
            os.environ.pop("DATA_DIR", None)
        else:
            os.environ["DATA_DIR"] = self.previous_data_dir
        db_module._db_ready = False
        get_settings.cache_clear()
        main_module.settings = get_settings()

    def _reels_for_generation(self, generation_id: str) -> list[dict]:
        with db_module.get_conn() as conn:
            rows = db_module.fetch_all(
                conn,
                "SELECT id, material_id, concept_id, generation_id, video_id, video_url, "
                "t_start, t_end FROM reels WHERE generation_id = ? ORDER BY created_at, t_start",
                (generation_id,),
            )
        return [dict(r) for r in rows]

    # ---- 1. multi-clip: one video with >=2 surviving clips -> >=2 reels ---- #

    def test_multi_clip_persists_multiple_reels(self) -> None:
        with _Patched():
            reels, resolved = main_module.ingestion_pipeline.ingest_topic(
                topic=TOPIC,
                material_id="mat-1",
                concept_id="con-1",
                generation_id="gen-1",
                max_videos=3,
            )

        # 2 reels from vidA + 1 from vidB = 3 total
        self.assertEqual(len(reels), 3)
        # Every reel carries the given material/concept
        for r in reels:
            self.assertEqual(r.material_id, "mat-1")
            self.assertEqual(r.concept_id, "con-1")

        # vidA produced >=2 DISTINCT reels
        vid_a_reels = [r for r in reels if "vidAAAAAAAA" in r.video_url]
        self.assertGreaterEqual(len(vid_a_reels), 2)
        self.assertEqual(len({r.reel_id for r in vid_a_reels}), len(vid_a_reels))

        # DB rows carry generation_id
        rows = self._reels_for_generation("gen-1")
        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertEqual(row["generation_id"], "gen-1")

    # ---- 2. exclude_video_ids forwarded to discover ---- #

    def test_exclude_video_ids_forwarded(self) -> None:
        with _Patched() as (mock_search, _mock_run):
            reels, resolved = main_module.ingestion_pipeline.ingest_topic(
                topic=TOPIC,
                material_id="mat-2",
                concept_id="con-2",
                generation_id="gen-2",
                exclude_video_ids=["vidBBBBBBBB"],
                max_videos=3,
            )

        # discover received the exclusion list
        self.assertEqual(
            mock_search.discover.call_args.kwargs["exclude_video_ids"], ["vidBBBBBBBB"]
        )
        # No reel from the excluded video; resolved ids exclude it too
        self.assertNotIn("vidBBBBBBBB", resolved)
        self.assertTrue(all("vidBBBBBBBB" not in r.video_url for r in reels))
        # Only vidA's 2 clips survive
        self.assertEqual(len(reels), 2)

    # ---- 3. max_reels global cap ---- #

    def test_max_reels_global_cap(self) -> None:
        seen: list[ReelOutWithAttribution] = []
        with _Patched():
            reels, resolved = main_module.ingestion_pipeline.ingest_topic(
                topic=TOPIC,
                material_id="mat-3",
                concept_id="con-3",
                generation_id="gen-3",
                max_videos=3,
                max_reels=2,
                on_reel_created=seen.append,
            )

        # Exactly 2 persisted even though 3 clips are available
        self.assertEqual(len(reels), 2)
        self.assertEqual(len(seen), 2)
        rows = self._reels_for_generation("gen-3")
        self.assertEqual(len(rows), 2)

    # ---- 4. on_reel_created fires once per reel, in order ---- #

    def test_on_reel_created_order(self) -> None:
        seen: list[ReelOutWithAttribution] = []
        with _Patched():
            reels, resolved = main_module.ingestion_pipeline.ingest_topic(
                topic=TOPIC,
                material_id="mat-4",
                concept_id="con-4",
                generation_id="gen-4",
                max_videos=3,
                on_reel_created=seen.append,
            )

        self.assertEqual(len(seen), len(reels))
        self.assertEqual([r.reel_id for r in seen], [r.reel_id for r in reels])
        # Same objects, in order
        for a, b in zip(seen, reels):
            self.assertIs(a, b)

    # ---- 5. dry_run: zero rows, non-empty resolved_video_ids ---- #

    def test_dry_run_discovers_only(self) -> None:
        with _Patched() as (_mock_search, mock_run):
            reels, resolved = main_module.ingestion_pipeline.ingest_topic(
                topic=TOPIC,
                material_id="mat-5",
                concept_id="con-5",
                generation_id="gen-5",
                max_videos=3,
                dry_run=True,
            )

        self.assertEqual(reels, [])
        self.assertEqual(resolved, ["vidAAAAAAAA", "vidBBBBBBBB"])
        mock_run.clip.assert_not_called()
        self.assertEqual(self._reels_for_generation("gen-5"), [])

    # ---- 6. return shape ---- #

    def test_return_shape(self) -> None:
        with _Patched():
            reels, resolved = main_module.ingestion_pipeline.ingest_topic(
                topic=TOPIC,
                material_id="mat-6",
                concept_id="con-6",
                generation_id="gen-6",
                max_videos=3,
            )

        self.assertIsInstance(reels, list)
        self.assertTrue(all(isinstance(r, ReelOutWithAttribution) for r in reels))
        self.assertEqual(resolved, ["vidAAAAAAAA", "vidBBBBBBBB"])

    # ---- 7. channel_name populated from discover metadata (Finding #6) ---- #

    def test_channel_name_populated(self) -> None:
        with _Patched():
            reels, _ = main_module.ingestion_pipeline.ingest_topic(
                topic=TOPIC,
                material_id="mat-7",
                concept_id="con-7",
                generation_id="gen-7",
                max_videos=3,
            )
        self.assertTrue(reels)
        for r in reels:
            self.assertEqual(r.channel_name, "BioChan")

    # ---- 8. over-length clips PERSIST at persist (RAW-PRACTICE reversal) ---- #

    def _patch_single_video(self, engine_out: dict):
        vid = {
            "id": "vidCCCCCCCC",
            "url": "https://www.youtube.com/watch?v=vidCCCCCCCC",
            "title": "Photosynthesis deep",
            "channel": "BioChan",
            "duration": 600.0,
            "thumbnail": "",
            "view_count": 1,
            "upload_date": "20230101",
        }
        search = mock.patch.object(pipeline_module, "clip_engine_search").start()
        run = mock.patch.object(pipeline_module, "clip_engine_run").start()
        self.addCleanup(mock.patch.stopall)
        search.discover.return_value = {
            "corrected": TOPIC, "videos": [vid], "credits_used": 0, "warning": None,
        }
        run.clip.return_value = engine_out
        return search, run

    def test_over_length_clip_persists(self) -> None:
        engine_out = {
            "video_id": "vidCCCCCCCC",
            "clips": [
                {"start": 30.0, "end": 75.0, "cut_end": 75.0, "title": "photosynthesis in range",
                 "facet": "", "reason": "", "sequence_index": 0, "embed_url": ""},
                {"start": 30.0, "end": 250.0, "cut_end": 250.0, "title": "photosynthesis over length",
                 "facet": "", "reason": "", "sequence_index": 1, "embed_url": ""},
            ],
            "transcript": {
                "segments": [
                    {"start": 30.0, "end": 75.0, "text": "photosynthesis part in range."},
                    {"start": 80.0, "end": 250.0, "text": "photosynthesis deep dive continues."},
                ],
                "words": [], "duration": 600.0,
            },
            "notes": "",
        }
        self._patch_single_video(engine_out)
        reels, _ = main_module.ingestion_pipeline.ingest_topic(
            topic=TOPIC,
            material_id="mat-8",
            concept_id="con-8",
            generation_id="gen-8",
            target_clip_duration_max_sec=60,
            max_videos=1,
        )
        # RAW-PRACTICE: the 220s clip exceeds the retired 60+8 window but PERSISTS
        # exactly as cut (not clamped) alongside the 45s clip.
        self.assertEqual(len(reels), 2)
        durations = sorted(round(r.t_end - r.t_start) for r in reels)
        self.assertEqual(durations, [45, 220])
        over = next(r for r in reels if round(r.t_end - r.t_start) == 220)
        self.assertEqual(over.t_start, 30.0)
        self.assertEqual(over.t_end, 250.0)
        rows = self._reels_for_generation("gen-8")
        self.assertEqual(len(rows), 2)

    # ---- 9. captions windowed to the clip + rebased clip-relative (Finding #5) ---- #

    def test_captions_windowed_and_rebased(self) -> None:
        engine_out = {
            "video_id": "vidCCCCCCCC",
            "clips": [
                {"start": 30.0, "end": 75.0, "cut_end": 75.0, "title": "photosynthesis clip",
                 "facet": "", "reason": "", "sequence_index": 0, "embed_url": ""},
            ],
            "transcript": {
                "segments": [
                    {"start": 20.0, "end": 50.0, "text": "photosynthesis alpha spanning the start."},
                    {"start": 60.0, "end": 90.0, "text": "photosynthesis beta spanning the end."},
                    {"start": 100.0, "end": 145.0, "text": "photosynthesis gamma outside the window."},
                ],
                "words": [], "duration": 600.0,
            },
            "notes": "",
        }
        self._patch_single_video(engine_out)
        reels, _ = main_module.ingestion_pipeline.ingest_topic(
            topic=TOPIC,
            material_id="mat-9",
            concept_id="con-9",
            generation_id="gen-9",
            max_videos=1,
        )
        self.assertEqual(len(reels), 1)
        reel = reels[0]
        clip_len = reel.t_end - reel.t_start  # 45.0
        self.assertEqual(len(reel.captions), 2, "gamma cue is outside [30,75] and must be dropped")
        for cue in reel.captions:
            self.assertGreaterEqual(cue.start, 0.0)
            self.assertLessEqual(cue.end, clip_len)
            self.assertLessEqual(cue.start, cue.end)
        # alpha [20,50] -> [0,20]; beta [60,90] -> [30,45]
        self.assertEqual(reel.captions[0].start, 0.0)
        self.assertEqual(reel.captions[0].end, 20.0)
        self.assertEqual(reel.captions[1].start, 30.0)
        self.assertEqual(reel.captions[1].end, clip_len)


class PerVideoCapAndBlendTests(IngestTopicTests):
    """INGEST_TOPIC_MAX_CLIPS_PER_VIDEO cap + relevance x informativeness blend.

    Fixture order deliberately differs from blended-score order so a revert of
    the blend (back to pure token-overlap relevance) fails the ordering assert.
    """

    @staticmethod
    def _blend_engine_out(*_a, **_kw) -> dict:
        # (title, start, end, informativeness, on_topic)
        specs = [
            ("Segment 0", 30.0, 75.0, 0.5, True),     # rel 1.0 -> score 0.75
            ("Segment 1", 120.0, 165.0, 1.0, True),   # rel 1.0 -> score 1.00 (best)
            ("Segment 2", 200.0, 245.0, 1.0, False),  # rel 0.0 -> score 0.00 (capped out)
            ("Segment 3", 300.0, 345.0, 0.6, True),   # rel 1.0 -> score 0.80
        ]
        clips, segments = [], []
        for i, (title, start, end, info, on_topic) in enumerate(specs):
            clips.append(
                {
                    "start": start,
                    "end": end,
                    "cut_end": end,
                    "title": title,
                    "facet": "",
                    "reason": "",
                    "informativeness": info,
                    "sequence_index": i,
                    "embed_url": f"https://www.youtube.com/embed/vidAAAAAAAA?start={int(start)}",
                }
            )
            text = "here we explain photosynthesis" if on_topic else "unrelated words only"
            segments.append({"start": start, "end": end, "text": text})
        return {
            "video_id": "vidAAAAAAAA",
            "clips": clips,
            "transcript": {"segments": segments, "words": [], "duration": 600.0},
            "notes": "",
        }

    def _run_with_cap(self, cap: int, suffix: str):
        with _Patched() as (mock_search, mock_run):
            mock_search.discover.side_effect = (
                lambda topic, limit, exclude_video_ids=None, **kw: {
                    "corrected": topic,
                    "videos": [VID_A],
                    "credits_used": 0,
                    "warning": None,
                }
            )
            mock_run.clip.side_effect = self._blend_engine_out
            with mock.patch.object(pipeline_module, "INGEST_TOPIC_MAX_CLIPS_PER_VIDEO", cap):
                reels, _ = main_module.ingestion_pipeline.ingest_topic(
                    topic=TOPIC,
                    material_id=f"mat-{suffix}",
                    concept_id=f"con-{suffix}",
                    generation_id=f"gen-{suffix}",
                    max_videos=1,
                )
        return reels

    def test_top_k_by_blended_score(self) -> None:
        reels = self._run_with_cap(3, "cap3")
        # top-3 of 4 by relevance*(0.5+0.5*informativeness), score-descending:
        # Segment 1 (1.0) > Segment 3 (0.8) > Segment 0 (0.75); Segment 2 capped out
        self.assertEqual([r.t_start for r in reels], [120.0, 300.0, 30.0])

    def test_cap_override(self) -> None:
        reels = self._run_with_cap(1, "cap1")
        self.assertEqual([r.t_start for r in reels], [120.0])


class EmbedUrlCeilTests(IngestTopicTests):
    """Persisted video_url must floor the start and CEIL the end (>= start+1):
    int()-truncating the end cut up to ~1s off every reel's final word."""

    @staticmethod
    def _fractional_engine_out(*_a, **_kw) -> dict:
        return {
            "video_id": "vidAAAAAAAA",
            "clips": [
                {
                    "start": 30.6,
                    "end": 74.4,
                    "cut_end": 74.55,
                    "title": "Fractional",
                    "facet": "",
                    "reason": "",
                    "sequence_index": 0,
                    "embed_url": "https://www.youtube.com/embed/vidAAAAAAAA?start=30&end=75&rel=0",
                }
            ],
            "transcript": {
                "segments": [
                    {"start": 30.6, "end": 74.4, "text": "here we explain photosynthesis"}
                ],
                "words": [],
                "duration": 600.0,
            },
            "notes": "",
        }

    def test_video_url_floors_start_and_ceils_end(self) -> None:
        with _Patched() as (mock_search, mock_run):
            mock_search.discover.side_effect = (
                lambda topic, limit, exclude_video_ids=None, **kw: {
                    "corrected": topic,
                    "videos": [VID_A],
                    "credits_used": 0,
                    "warning": None,
                }
            )
            mock_run.clip.side_effect = self._fractional_engine_out
            reels, _ = main_module.ingestion_pipeline.ingest_topic(
                topic=TOPIC,
                material_id="mat-ceil",
                concept_id="con-ceil",
                generation_id="gen-ceil",
                max_videos=1,
            )
        self.assertEqual(len(reels), 1)
        self.assertIn("start=30&end=75", reels[0].video_url)


if __name__ == "__main__":
    unittest.main()
