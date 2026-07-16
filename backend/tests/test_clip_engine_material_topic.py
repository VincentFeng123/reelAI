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
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("REELAI_INGEST_SKIP_IMPORT_SWEEP", "1")

from backend.app import db as db_module  # noqa: E402
from backend.app.config import get_settings  # noqa: E402
from backend.app.clip_engine.provider_runtime import GenerationContext  # noqa: E402
import backend.app.main as main_module  # noqa: E402
from backend.app.ingestion import pipeline as pipeline_module  # noqa: E402
from backend.app.ingestion.models import ReelOutWithAttribution  # noqa: E402
from backend.pipeline import gemini_segment as segment_module  # noqa: E402


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


def _quality_v2_engine_out(engine_out: dict) -> dict:
    """Give mocked selector output the same provenance and hard gates as v2."""
    transcript = engine_out["transcript"]
    transcript.update(
        source="supadata",
        artifact_key=f"supadata:{engine_out['video_id']}",
        native_mode=True,
    )
    segments = transcript["segments"]
    for index, segment in enumerate(segments):
        segment.setdefault("cue_id", f"cue-{index}")

    for index, clip in enumerate(engine_out["clips"]):
        clip.setdefault("difficulty", 0.15)
        start = float(clip["start"])
        end = float(clip["end"])
        selected = [
            segment
            for segment in segments
            if float(segment["start"]) >= start - 1e-6
            and float(segment["end"]) <= end + 1e-6
        ]
        if not selected:
            selected = [
                segment
                for segment in segments
                if float(segment["end"]) > start and float(segment["start"]) < end
            ]
            if selected:
                clip["start"] = min(float(segment["start"]) for segment in selected)
                clip["end"] = max(float(segment["end"]) for segment in selected)
                clip["cut_end"] = clip["end"]
        if not selected:
            continue
        words = " ".join(str(segment["text"]) for segment in selected).split()
        if len(words) < 5:
            selected[0]["text"] = (
                f"{selected[0]['text']} with complete grounded explanatory context"
            )
            words = " ".join(str(segment["text"]) for segment in selected).split()
        clip.update(
            cue_ids=[str(segment["cue_id"]) for segment in selected],
            kind="educational",
            learning_objective=clip.get("learning_objective")
            or f"Explain {clip.get('title') or 'this concept'}.",
            facet=clip.get("facet") or f"facet-{index}",
            reason=clip.get("reason") or "Provides a complete grounded explanation.",
            informativeness=max(0.75, float(clip.get("informativeness", 0.9))),
            topic_relevance=float(clip.get("topic_relevance", 0.9)),
            educational_importance=float(
                clip.get("educational_importance", 0.9)
            ),
            self_contained=True,
            is_standalone=True,
            directly_teaches_topic=True,
            substantive=True,
            factually_grounded=True,
            topic_evidence_quote=" ".join(words[:40]),
            boundary_confidence=0.9,
            prerequisite_ids=[],
            selection_candidate_id=clip.get("selection_candidate_id")
            or f"candidate-{index}",
        )
    return engine_out


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
    return _quality_v2_engine_out({
        "video_id": video_id,
        "clips": clips,
        "transcript": {"segments": segments, "words": [], "duration": 600.0},
        "notes": "",
    })


def _discover_side_effect(topic, limit, exclude_video_ids=None, **kw):
    excl = set(exclude_video_ids or []) | set(kw.get("consumed_video_ids") or [])
    videos = [v for v in POOL if v["id"] not in excl][:limit]
    return {"corrected": topic, "videos": videos, "credits_used": 0, "warning": None}


def _clip_side_effect(url, topic=None, settings=None, **_kwargs):
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
        identity_suffixes = [
            "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "all", "ceil", "diff", "nodiff", "lvl",
        ]
        with db_module.get_conn(transactional=True) as conn:
            for suffix in identity_suffixes:
                material_id = f"mat-{suffix}"
                concept_id = f"con-{suffix}"
                db_module.insert(
                    conn,
                    "materials",
                    {
                        "id": material_id,
                        "subject_tag": TOPIC,
                        "raw_text": TOPIC,
                        "source_type": "topic",
                        "source_path": None,
                        "created_at": db_module.now_iso(),
                    },
                )
                db_module.insert(
                    conn,
                    "concepts",
                    {
                        "id": concept_id,
                        "material_id": material_id,
                        "title": TOPIC,
                        "keywords_json": "[]",
                        "summary": "",
                        "embedding_json": None,
                        "created_at": db_module.now_iso(),
                    },
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

    def test_open_provider_cursor_is_recorded_for_resumable_generation(self) -> None:
        context = GenerationContext("slow")
        with _Patched() as (mock_search, mock_run):
            mock_search.discover.side_effect = None
            mock_search.discover.return_value = {
                "corrected": TOPIC,
                "videos": [],
                "credits_used": 0,
                "warning": None,
                "provider_exhausted": False,
            }
            reels, resolved = main_module.ingestion_pipeline.ingest_topic(
                topic=TOPIC,
                material_id="mat-2",
                concept_id="con-2",
                generation_id="gen-open-cursor",
                max_videos=1,
                max_reels=1,
                generation_context=context,
            )

        self.assertEqual(reels, [])
        self.assertEqual(resolved, [])
        self.assertEqual(context.counters()["provider_cursor_open"], 1)
        mock_run.clip.assert_not_called()

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

        # The active inventory and callbacks honor the cap, while the remaining
        # valid later-source clip is stored for future difficulty progression.
        self.assertEqual(len(reels), 2)
        self.assertEqual(len(seen), 2)
        self.assertEqual([r.t_start for r in reels], [30.0, 120.0])
        self.assertTrue(all("vidAAAAAAAA" in r.video_url for r in reels))
        rows = self._reels_for_generation("gen-3")
        self.assertEqual(len(rows), 3)

    def test_fourth_sibling_is_cached_beyond_three_reel_surface_batch(self) -> None:
        four_windows = [
            (30.0, 75.0),
            (120.0, 165.0),
            (210.0, 255.0),
            (300.0, 345.0),
        ]
        surfaced: list[ReelOutWithAttribution] = []
        with mock.patch.dict(CLIP_WINDOWS, {"vidAAAAAAAA": four_windows}):
            with _Patched() as (mock_search, mock_run):
                mock_search.discover.side_effect = (
                    lambda topic, limit, exclude_video_ids=None, **kw: {
                        "corrected": topic,
                        "videos": [VID_A],
                        "credits_used": 0,
                        "warning": None,
                        "provider_exhausted": False,
                    }
                )
                first_batch, _ = main_module.ingestion_pipeline.ingest_topic(
                    topic=TOPIC,
                    material_id="mat-3",
                    concept_id="con-3",
                    generation_id="gen-four-siblings",
                    max_videos=1,
                    max_reels=3,
                    max_persisted_reels=None,
                    on_reel_created=surfaced.append,
                )
                selector_calls = mock_run.clip.call_count

                with db_module.get_conn() as conn:
                    reservoir = main_module.reel_service.ranked_feed(
                        conn,
                        "mat-3",
                        generation_id="gen-four-siblings",
                    )
                surfaced_ids = {reel.reel_id for reel in first_batch}
                cached_continuation = [
                    reel
                    for reel in reservoir
                    if reel["reel_id"] not in surfaced_ids
                ]

        self.assertEqual(len(first_batch), 3)
        self.assertEqual(len(surfaced), 3)
        self.assertEqual(len(self._reels_for_generation("gen-four-siblings")), 4)
        self.assertEqual(len(cached_continuation), 1)
        self.assertEqual(float(cached_continuation[0]["t_start"]), 300.0)
        self.assertEqual(selector_calls, 1)
        self.assertEqual(mock_run.clip.call_count, 1)

    def test_retry_reuse_does_not_consume_remaining_persistence_capacity(self) -> None:
        first_result = _build_engine_out("vidAAAAAAAA")
        first_result["clips"] = first_result["clips"][:1]
        retry_result = _build_engine_out("vidAAAAAAAA")

        with _Patched() as (mock_search, mock_run):
            mock_search.discover.side_effect = (
                lambda topic, limit, exclude_video_ids=None, **kw: {
                    "corrected": topic,
                    "videos": [VID_A],
                    "credits_used": 0,
                    "warning": None,
                }
            )
            mock_run.clip.side_effect = [first_result, retry_result]
            first_reels, _ = main_module.ingestion_pipeline.ingest_topic(
                topic=TOPIC,
                material_id="mat-3",
                concept_id="con-3",
                generation_id="gen-cap-retry",
                max_videos=1,
                max_reels=2,
                max_persisted_reels=1,
            )
            retry_reels, _ = main_module.ingestion_pipeline.ingest_topic(
                topic=TOPIC,
                material_id="mat-3",
                concept_id="con-3",
                generation_id="gen-cap-retry",
                max_videos=1,
                max_reels=2,
                max_persisted_reels=1,
            )

        self.assertEqual([reel.t_start for reel in first_reels], [30.0])
        self.assertEqual(
            [reel.t_start for reel in retry_reels],
            [30.0, 120.0],
        )
        self.assertEqual(
            [row["t_start"] for row in self._reels_for_generation("gen-cap-retry")],
            [30.0, 120.0],
        )

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
        run.clip.return_value = _quality_v2_engine_out(engine_out)
        return search, run

    def test_complete_clip_through_one_eighty_persists(self) -> None:
        engine_out = {
            "video_id": "vidCCCCCCCC",
            "clips": [
                {"start": 30.0, "end": 75.0, "cut_end": 75.0, "title": "photosynthesis in range",
                 "facet": "", "reason": "", "sequence_index": 0, "embed_url": ""},
                {"start": 30.0, "end": 200.0, "title": "photosynthesis long complete",
                 "facet": "", "reason": "", "sequence_index": 1, "embed_url": ""},
            ],
            "transcript": {
                "segments": [
                    {"start": 30.0, "end": 75.0, "text": "photosynthesis part in range."},
                    {"start": 80.0, "end": 200.0, "text": "photosynthesis deep dive continues."},
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
        # The 170s complete clip persists exactly as selected alongside the 45s clip.
        self.assertEqual(len(reels), 2)
        durations = sorted(round(r.t_end - r.t_start) for r in reels)
        self.assertEqual(durations, [45, 170])
        over = next(r for r in reels if round(r.t_end - r.t_start) == 170)
        self.assertEqual(over.t_start, 30.0)
        self.assertEqual(over.t_end, 200.0)
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
        clip_len = reel.t_end - reel.t_start  # 70.0; v2 includes both required cues.
        self.assertEqual(len(reel.captions), 2, "gamma cue is outside [20,90] and must be dropped")
        for cue in reel.captions:
            self.assertGreaterEqual(cue.start, 0.0)
            self.assertLessEqual(cue.end, clip_len)
            self.assertLessEqual(cue.start, cue.end)
        # The selector cannot begin/end inside required speech, so both cues are whole.
        self.assertEqual(reel.captions[0].start, 0.0)
        self.assertEqual(reel.captions[0].end, 30.0)
        self.assertEqual(reel.captions[1].start, 40.0)
        self.assertEqual(reel.captions[1].end, clip_len)


class PreserveEveryClipAndBlendTests(IngestTopicTests):
    """Every accepted engine clip persists in stable relevance order."""

    @staticmethod
    def _blend_engine_out(*_a, **_kw) -> dict:
        # (title, start, end, informativeness, on_topic)
        specs = [
            ("Segment 0", 30.0, 75.0, 0.80, True),
            ("Segment 1", 120.0, 165.0, 1.00, True),
            ("Segment 2", 200.0, 245.0, 1.00, False),
            ("Segment 3", 300.0, 345.0, 0.90, True),
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
                    "topic_relevance": 0.9 if on_topic else 0.74,
                    "educational_importance": info,
                    "sequence_index": i,
                    "embed_url": f"https://www.youtube.com/embed/vidAAAAAAAA?start={int(start)}",
                }
            )
            text = "here we explain photosynthesis" if on_topic else "unrelated words only"
            segments.append({"start": start, "end": end, "text": text})
        return _quality_v2_engine_out({
            "video_id": "vidAAAAAAAA",
            "clips": clips,
            "transcript": {"segments": segments, "words": [], "duration": 600.0},
            "notes": "",
        })

    def _run(self, suffix: str):
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
            reels, _ = main_module.ingestion_pipeline.ingest_topic(
                topic=TOPIC,
                material_id=f"mat-{suffix}",
                concept_id=f"con-{suffix}",
                generation_id=f"gen-{suffix}",
                max_videos=1,
            )
        return reels

    def test_only_hard_gate_clips_persist_in_relevance_order(self) -> None:
        reels = self._run("all")
        self.assertEqual([r.t_start for r in reels], [30.0, 120.0, 300.0])


class EmbedUrlCeilTests(IngestTopicTests):
    """Persisted video_url must floor the start and CEIL the end (>= start+1):
    int()-truncating the end cut up to ~1s off every reel's final word."""

    @staticmethod
    def _fractional_engine_out(*_a, **_kw) -> dict:
        return _quality_v2_engine_out({
            "video_id": "vidAAAAAAAA",
            "clips": [
                {
                    "start": 30.6126,
                    "end": 74.4567,
                    "title": "Fractional",
                    "facet": "",
                    "reason": "",
                    "sequence_index": 0,
                    "embed_url": "https://www.youtube.com/embed/vidAAAAAAAA?start=30&end=75&rel=0",
                }
            ],
            "transcript": {
                "segments": [
                    {"start": 30.6126, "end": 74.4567, "text": "here we explain photosynthesis"}
                ],
                "words": [],
                "duration": 600.0,
            },
            "notes": "",
        })

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
        self.assertEqual(reels[0].t_start, 30.613)
        self.assertEqual(reels[0].t_end, 74.457)
        with db_module.get_conn() as conn:
            row = db_module.fetch_one(conn, "SELECT t_start, t_end FROM reels WHERE id = ?", (reels[0].reel_id,))
        self.assertEqual(float(row["t_start"]), 30.613)
        self.assertEqual(float(row["t_end"]), 74.457)

    def test_complete_clip_over_180_seconds_survives_v3_persistence_and_level_feed(
        self,
    ) -> None:
        transcript_text = (
            "Photosynthesis converts light energy into chemical energy, and this "
            "explanation concludes by connecting the light reactions to carbon fixation."
        )
        transcript = {
            "segments": [
                {
                    "cue_id": "long-cue",
                    "start": 12.345,
                    "end": 432.789,
                    "text": transcript_text,
                }
            ],
            "words": [],
            "duration": 432.789,
            "source": "supadata",
            "artifact_key": "supadata:vidAAAAAAAA",
            "native_mode": True,
        }
        proposal = segment_module._BoundaryTopic(
            candidate_id="long-complete",
            start_line=0,
            end_line=0,
            start_quote="Photosynthesis converts light energy",
            end_quote="light reactions to carbon fixation",
            title="Photosynthesis from light to carbon",
            learning_objective=(
                "Explain how photosynthesis connects light capture to carbon fixation."
            ),
            facet="energy conversion and carbon fixation",
            reason="Directly explains the core process and reaches its conclusion.",
            informativeness=0.92,
            topic_relevance=0.97,
            educational_importance=0.94,
            difficulty=0.20,
            directly_teaches_topic=True,
            substantive=True,
            factually_grounded=True,
            topic_evidence_quote=(
                "Photosynthesis converts light energy into chemical energy"
            ),
            self_contained=True,
            is_standalone=True,
            prerequisite_candidate_ids=[],
            uncertainty="low",
            uncertainty_reasons=[],
        )
        clips = segment_module._plan_to_clips(
            segment_module._BoundaryPlan(topics=[proposal]),
            transcript["segments"],
            [],
            {"max_clips": 16},
        )
        self.assertEqual(len(clips), 1)
        self.assertGreater(clips[0]["end"] - clips[0]["start"], 180.0)
        engine_out = {
            "video_id": "vidAAAAAAAA",
            "clips": clips,
            "transcript": transcript,
            "notes": "",
        }
        prepared = pipeline_module.clip_engine_silence.AudioPreparationResult(
            "ready",
            source=pipeline_module.clip_engine_silence.PreparedAudioSource(
                url="https://media.example/audio.m4a",
                format_id="140",
                duration_sec=600.0,
            ),
        )
        verified = pipeline_module.clip_engine_silence.SilenceVerificationResult(
            "verified",
            start_sec=12.001,
            end_sec=433.012,
            diagnostics={
                "threshold_dbfs": -38.0,
                "speech_handoff_verified": True,
                "start_speech_handoff_verified": True,
                "end_speech_handoff_verified": True,
                "start_two_sided_required": False,
                "end_two_sided_required": False,
                "semantic_start_limit_sec": 0.0,
                "semantic_end_limit_sec": 600.0,
                "observation_start_limit_sec": 11.345,
                "observation_end_limit_sec": 433.789,
                "handoff_timestamp_tolerance_sec": 0.05,
                "start_quiet": [11.9, 12.4],
                "end_quiet": [432.7, 433.1],
            },
        )

        with (
            _Patched() as (mock_search, mock_run),
            mock.patch.object(
                pipeline_module.clip_engine_silence,
                "prepare_audio_source",
                return_value=prepared,
            ),
            mock.patch.object(
                pipeline_module.clip_engine_silence,
                "verify_acoustic_boundaries",
                return_value=verified,
            ),
        ):
            mock_search.discover.side_effect = (
                lambda topic, limit, exclude_video_ids=None, **kw: {
                    "corrected": topic,
                    "videos": [VID_A],
                    "credits_used": 0,
                    "warning": None,
                }
            )
            mock_run.clip.return_value = engine_out
            mock_run.clip.side_effect = None
            reels, _ = main_module.ingestion_pipeline.ingest_topic(
                topic=TOPIC,
                material_id="mat-ceil",
                concept_id="con-ceil",
                generation_id="gen-long-v3",
                knowledge_level="beginner",
                max_videos=1,
                max_reels=1,
                retrieval_profile="deep",
                generation_context=GenerationContext(
                    "fast",
                    generation_id="gen-long-v3",
                    require_acoustic_boundaries=True,
                ),
            )

        self.assertEqual(len(reels), 1)
        self.assertEqual(reels[0].selection_contract_version, "quality_silence_v38")
        self.assertEqual(reels[0].t_start, 12.001)
        self.assertEqual(reels[0].t_end, 433.012)
        self.assertGreater(reels[0].t_end - reels[0].t_start, 180.0)

        with db_module.get_conn(transactional=True) as conn:
            row = db_module.fetch_one(
                conn,
                "SELECT t_start, t_end FROM reels WHERE generation_id = ?",
                ("gen-long-v3",),
            )
            self.assertEqual(float(row["t_start"]), 12.001)
            self.assertEqual(float(row["t_end"]), 433.012)
            main_module.reel_service.set_learner_level(
                conn, "mat-ceil", "owner:long-v3", "advanced"
            )

        relevance = {
            "score": 0.97,
            "concept_overlap": 1.0,
            "context_overlap": 1.0,
            "matched_terms": ["photosynthesis"],
            "off_topic_penalty": 0.0,
            "reason": "matched topic",
        }
        with (
            mock.patch.object(
                main_module.reel_service,
                "_score_text_relevance",
                return_value=relevance,
            ),
            mock.patch.object(
                main_module.reel_service, "_build_caption_cues", return_value=[]
            ),
        ):
            with db_module.get_conn() as conn:
                advanced_feed = main_module.reel_service.ranked_feed(
                    conn,
                    material_id="mat-ceil",
                    generation_id="gen-long-v3",
                    learner_id="owner:long-v3",
                    require_verified_boundaries=True,
                )
            self.assertEqual(
                [item["reel_id"] for item in advanced_feed],
                [reels[0].reel_id],
            )

            with db_module.get_conn(transactional=True) as conn:
                main_module.reel_service.set_learner_level(
                    conn, "mat-ceil", "owner:long-v3", "beginner"
                )
            with db_module.get_conn() as conn:
                beginner_feed = main_module.reel_service.ranked_feed(
                    conn,
                    material_id="mat-ceil",
                    generation_id="gen-long-v3",
                    learner_id="owner:long-v3",
                    require_verified_boundaries=True,
                )

        self.assertEqual(len(beginner_feed), 1)
        self.assertEqual(
            beginner_feed[0]["selection_contract_version"], "quality_silence_v38"
        )
        self.assertIsInstance(beginner_feed[0]["t_start"], float)
        self.assertIsInstance(beginner_feed[0]["t_end"], float)
        self.assertEqual(beginner_feed[0]["t_start"], 12.001)
        self.assertEqual(beginner_feed[0]["t_end"], 433.012)
        self.assertGreater(
            beginner_feed[0]["t_end"] - beginner_feed[0]["t_start"], 180.0
        )


class DifficultyPersistenceTests(IngestTopicTests):
    """Engine difficulty lands in reels.difficulty; absent -> NULL."""

    @staticmethod
    def _difficulty_engine_out(*_a, **_kw) -> dict:
        return _quality_v2_engine_out({
            "video_id": "vidAAAAAAAA",
            "clips": [{
                "start": 30.0, "end": 75.0, "cut_end": 75.15,
                "title": "Scored", "facet": "", "reason": "",
                "informativeness": 0.9, "difficulty": 0.8,
                "sequence_index": 0,
                "embed_url": "https://www.youtube.com/embed/vidAAAAAAAA?start=30&end=75&rel=0",
            }],
            "transcript": {"segments": [
                {"start": 30.0, "end": 75.0, "text": "here we explain photosynthesis"}
            ], "words": [], "duration": 600.0},
            "notes": "",
        })

    def test_difficulty_round_trips_to_db(self) -> None:
        with _Patched() as (mock_search, mock_run):
            mock_search.discover.side_effect = (
                lambda topic, limit, exclude_video_ids=None, **kw: {
                    "corrected": topic, "videos": [VID_A],
                    "credits_used": 0, "warning": None,
                }
            )
            mock_run.clip.side_effect = self._difficulty_engine_out
            main_module.ingestion_pipeline.ingest_topic(
                topic=TOPIC, material_id="mat-diff", concept_id="con-diff",
                generation_id="gen-diff", max_videos=1,
            )
        with db_module.get_conn() as conn:
            row = db_module.fetch_all(
                conn, "SELECT difficulty FROM reels WHERE generation_id = ?", ("gen-diff",)
            )[0]
        self.assertAlmostEqual(float(row["difficulty"]), 0.8)

    @staticmethod
    def _no_difficulty_engine_out(*_a, **_kw) -> dict:
        return _quality_v2_engine_out({
            "video_id": "vidAAAAAAAA",
            "clips": [{
                "start": 30.0, "end": 75.0, "cut_end": 75.15,
                "title": "Unscored", "facet": "", "reason": "",
                "informativeness": 0.9, "difficulty": None,
                "sequence_index": 0,
                "embed_url": "https://www.youtube.com/embed/vidAAAAAAAA?start=30&end=75&rel=0",
            }],
            "transcript": {"segments": [
                {"start": 30.0, "end": 75.0, "text": "here we explain photosynthesis"}
            ], "words": [], "duration": 600.0},
            "notes": "",
        })

    def test_difficulty_absent_stays_null(self) -> None:
        with _Patched() as (mock_search, mock_run):
            mock_search.discover.side_effect = (
                lambda topic, limit, exclude_video_ids=None, **kw: {
                    "corrected": topic, "videos": [VID_A],
                    "credits_used": 0, "warning": None,
                }
            )
            mock_run.clip.side_effect = self._no_difficulty_engine_out
            main_module.ingestion_pipeline.ingest_topic(
                topic=TOPIC, material_id="mat-nodiff", concept_id="con-nodiff",
                generation_id="gen-nodiff", max_videos=1,
            )
        with db_module.get_conn() as conn:
            row = db_module.fetch_all(
                conn, "SELECT difficulty FROM reels WHERE generation_id = ?", ("gen-nodiff",)
            )[0]
        self.assertIsNone(row["difficulty"])


class LevelThreadingTests(IngestTopicTests):
    def test_ingest_topic_keeps_discovery_level_neutral(self) -> None:
        with _Patched() as (mock_search, mock_run):
            mock_run.clip.side_effect = _clip_side_effect
            main_module.ingestion_pipeline.ingest_topic(
                topic=TOPIC, material_id="mat-lvl", concept_id="con-lvl",
                generation_id="gen-lvl", max_videos=1, knowledge_level="advanced",
            )
            _, kwargs = mock_search.discover.call_args
            self.assertIsNone(kwargs.get("level"))
            self.assertEqual(
                mock_run.clip.call_args.kwargs["settings"]["_knowledge_level"],
                "advanced",
            )


class IngestTopicProgressTests(unittest.TestCase):
    @staticmethod
    def _video(video_id: str) -> dict:
        return {
            "id": video_id,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "title": video_id,
            "channel": "Test",
            "duration": 60.0,
            "thumbnail": "",
        }

    def _pipeline(self) -> pipeline_module.IngestionPipeline:
        return pipeline_module.IngestionPipeline(
            youtube_service=mock.Mock(),
            embedding_service=mock.Mock(),
            rate_limiter=pipeline_module._PlatformRateLimiter(
                overrides={"yt": (1000, 60.0)}
            ),
        )

    def test_fast_video_is_persisted_and_emitted_while_earlier_video_is_still_running(self) -> None:
        pipeline = self._pipeline()
        slow = self._video("slow-video")
        fast = self._video("fast-video")
        slow_started = threading.Event()
        release_slow = threading.Event()
        slow_finished = threading.Event()
        persist_states: list[tuple[str, bool]] = []
        callback_states: list[tuple[str, bool]] = []

        def clip_and_filter(video, *_args):
            if video["id"] == "slow-video":
                slow_started.set()
                self.assertTrue(release_slow.wait(1.0))
                slow_finished.set()
            else:
                self.assertTrue(slow_started.wait(1.0))
            return video, [{"title": video["id"]}], {"transcript": {}}

        def persist_engine_clip(**kwargs):
            video_id = kwargs["v"]["id"]
            persist_states.append((video_id, slow_finished.is_set()))
            return video_id, mock.sentinel.metadata

        def on_reel_created(reel: str) -> None:
            callback_states.append((reel, slow_finished.is_set()))
            if reel == "fast-video":
                release_slow.set()

        try:
            with (
                mock.patch.object(
                    pipeline_module,
                    "_discover",
                    return_value={
                        "corrected": TOPIC,
                        "videos": [slow, fast],
                        "credits_used": 0,
                        "warning": None,
                    },
                ),
                mock.patch.object(pipeline, "_clip_and_filter", side_effect=clip_and_filter),
                mock.patch.object(
                    pipeline, "_persist_engine_clip", side_effect=persist_engine_clip
                ),
            ):
                reels, _ = pipeline.ingest_topic(
                    topic=TOPIC,
                    material_id="material",
                    concept_id="concept",
                    max_videos=2,
                    on_reel_created=on_reel_created,
                )
        finally:
            release_slow.set()

        self.assertIn(("fast-video", False), persist_states)
        self.assertIn(("fast-video", False), callback_states)
        self.assertEqual(reels, ["slow-video", "fast-video"])

    def test_stalled_videos_share_one_fetch_timeout(self) -> None:
        pipeline = self._pipeline()
        videos = [self._video("slow-a"), self._video("slow-b")]
        finished = [threading.Event(), threading.Event()]
        cancelled = [threading.Event(), threading.Event()]
        analyzed: set[str] = set()

        def clip_and_filter(video, _topic, _language, should_cancel, _context):
            index = 0 if video["id"] == "slow-a" else 1
            emergency_deadline = time.monotonic() + 1.0
            while not should_cancel() and time.monotonic() < emergency_deadline:
                time.sleep(0.005)
            if should_cancel():
                cancelled[index].set()
            finished[index].set()
            return video, [], {"transcript": {}}

        with (
            mock.patch.object(
                pipeline_module,
                "_discover",
                return_value={
                    "corrected": TOPIC,
                    "videos": videos,
                    "credits_used": 0,
                    "warning": None,
                },
            ),
            mock.patch.object(
                pipeline_module, "INGEST_TOPIC_VIDEO_TIMEOUT_SEC", 0.2
            ),
            mock.patch.object(
                pipeline, "_clip_and_filter", side_effect=clip_and_filter
            ),
        ):
            started = time.monotonic()
            reels, _ = pipeline.ingest_topic(
                topic=TOPIC,
                material_id="material",
                concept_id="concept",
                max_videos=2,
                analyzed_video_ids=analyzed,
            )
            elapsed = time.monotonic() - started

        self.assertEqual(reels, [])
        self.assertLess(elapsed, 0.32)
        self.assertTrue(all(event.wait(0.1) for event in finished))
        self.assertTrue(all(event.is_set() for event in cancelled))
        self.assertEqual(analyzed, set())

    def test_first_valid_clip_streams_without_cancelling_selected_source(self) -> None:
        pipeline = self._pipeline()
        videos = [self._video("slow-video"), self._video("fast-video")]
        slow_started = threading.Event()
        streamed: list[str] = []
        analyzed: set[str] = set()

        def clip_and_filter(video, _topic, _language, should_cancel, _context):
            if video["id"] == "slow-video":
                slow_started.set()
                time.sleep(0.08)
                self.assertFalse(should_cancel())
                return video, [{"title": "slow", "score": 1.0}], {"transcript": {}}
            self.assertTrue(slow_started.wait(1.0))
            return video, [{"title": "fast", "score": 1.0}], {"transcript": {}}

        with (
            mock.patch.object(
                pipeline_module,
                "_discover",
                return_value={
                    "corrected": TOPIC,
                    "videos": videos,
                    "credits_used": 0,
                    "warning": None,
                },
            ),
            mock.patch.object(pipeline_module, "INGEST_TOPIC_VIDEO_TIMEOUT_SEC", 0.3),
            mock.patch.object(pipeline, "_clip_and_filter", side_effect=clip_and_filter),
            mock.patch.object(
                pipeline,
                "_persist_engine_clip",
                side_effect=lambda **kwargs: (kwargs["v"]["id"], mock.sentinel.metadata),
            ),
        ):
            started = time.monotonic()
            reels, _ = pipeline.ingest_topic(
                topic=TOPIC,
                material_id="material",
                concept_id="concept",
                max_videos=2,
                max_reels=8,
                on_reel_created=streamed.append,
                analyzed_video_ids=analyzed,
            )
            elapsed = time.monotonic() - started

        self.assertEqual(reels, ["slow-video", "fast-video"])
        self.assertEqual(streamed, ["fast-video", "slow-video"])
        self.assertEqual(analyzed, {"slow-video", "fast-video"})
        self.assertGreaterEqual(elapsed, 0.07)
        self.assertLess(elapsed, 0.3)

    def test_useful_inventory_bounds_the_wait_for_a_stalled_selected_source(self) -> None:
        pipeline = self._pipeline()
        videos = [self._video("stalled-video"), self._video("fast-video")]
        stalled_started = threading.Event()
        stalled_cancelled = threading.Event()
        analyzed: set[str] = set()
        streamed: list[str] = []

        def clip_and_filter(video, _topic, _language, should_cancel, _context):
            if video["id"] == "stalled-video":
                stalled_started.set()
                while not should_cancel():
                    time.sleep(0.002)
                stalled_cancelled.set()
                return video, [], {"transcript": {}}
            self.assertTrue(stalled_started.wait(1.0))
            return video, [
                {"title": f"fast-{index}", "score": 1.0}
                for index in range(3)
            ], {"transcript": {}}

        with (
            mock.patch.object(
                pipeline_module,
                "_discover",
                return_value={
                    "corrected": TOPIC,
                    "videos": videos,
                    "credits_used": 0,
                    "warning": None,
                },
            ),
            mock.patch.object(pipeline_module, "INGEST_TOPIC_VIDEO_TIMEOUT_SEC", 0.5),
            mock.patch.object(
                pipeline_module,
                "INGEST_TOPIC_USEFUL_INVENTORY_IDLE_TIMEOUT_SEC",
                0.03,
            ),
            mock.patch.object(pipeline, "_clip_and_filter", side_effect=clip_and_filter),
            mock.patch.object(
                pipeline,
                "_persist_engine_clip",
                side_effect=lambda **kwargs: (
                    kwargs["v"]["id"],
                    mock.sentinel.metadata,
                ),
            ),
        ):
            started = time.monotonic()
            reels, _ = pipeline.ingest_topic(
                topic=TOPIC,
                material_id="material",
                concept_id="concept",
                max_videos=2,
                max_reels=8,
                on_reel_created=streamed.append,
                analyzed_video_ids=analyzed,
            )
            elapsed = time.monotonic() - started

        self.assertEqual(reels, ["fast-video"] * 3)
        self.assertEqual(streamed, ["fast-video"] * 3)
        self.assertEqual(analyzed, {"fast-video"})
        self.assertLess(elapsed, 0.2)
        self.assertTrue(stalled_cancelled.wait(0.1))

    def test_deferred_valid_clip_streams_and_slow_source_still_finishes(self) -> None:
        pipeline = self._pipeline()
        videos = [self._video("slow-video"), self._video("deferred-video")]
        slow_started = threading.Event()
        stored: list[str] = []
        streamed: list[str] = []

        def clip_and_filter(video, _topic, _language, should_cancel, _context):
            if video["id"] == "slow-video":
                slow_started.set()
                time.sleep(0.06)
                self.assertFalse(should_cancel())
                return video, [{
                    "title": "valid beginner lesson",
                    "score": 1.0,
                    "difficulty": 0.15,
                    "search_context": {"surface_eligible": True},
                }], {"transcript": {}}
            self.assertTrue(slow_started.wait(1.0))
            return video, [{
                "title": "valid intermediate lesson",
                "score": 1.0,
                "difficulty": 0.50,
                "search_context": {
                    "surface_eligible": True,
                    "deferred_level": True,
                },
            }], {"transcript": {}}

        with (
            mock.patch.object(
                pipeline_module,
                "_discover",
                return_value={
                    "corrected": TOPIC,
                    "videos": videos,
                    "credits_used": 0,
                    "warning": None,
                },
            ),
            mock.patch.object(pipeline_module, "INGEST_TOPIC_VIDEO_TIMEOUT_SEC", 0.3),
            mock.patch.object(pipeline, "_clip_and_filter", side_effect=clip_and_filter),
            mock.patch.object(
                pipeline,
                "_persist_engine_clip",
                side_effect=lambda **kwargs: (
                    stored.append(kwargs["v"]["id"]) or kwargs["v"]["id"],
                    mock.sentinel.metadata,
                ),
            ),
        ):
            started = time.monotonic()
            reels, _ = pipeline.ingest_topic(
                topic=TOPIC,
                material_id="material",
                concept_id="concept",
                max_videos=2,
                max_reels=8,
                knowledge_level="beginner",
                on_reel_created=streamed.append,
            )
            elapsed = time.monotonic() - started

        self.assertEqual(reels, ["slow-video", "deferred-video"])
        self.assertEqual(stored, ["deferred-video", "slow-video"])
        self.assertEqual(streamed, ["deferred-video", "slow-video"])
        self.assertGreaterEqual(elapsed, 0.05)
        self.assertLess(elapsed, 0.3)

    def test_empty_first_source_does_not_prevent_useful_source_completion(self) -> None:
        pipeline = self._pipeline()
        videos = [self._video("empty-video"), self._video("useful-video")]

        def clip_and_filter(video, *_args):
            if video["id"] == "empty-video":
                return video, [], {"transcript": {}}
            time.sleep(0.05)
            return video, [{"title": "useful", "score": 1.0}], {"transcript": {}}

        with (
            mock.patch.object(
                pipeline_module,
                "_discover",
                return_value={
                    "corrected": TOPIC,
                    "videos": videos,
                    "credits_used": 0,
                    "warning": None,
                },
            ),
            mock.patch.object(pipeline_module, "INGEST_TOPIC_VIDEO_TIMEOUT_SEC", 0.3),
            mock.patch.object(pipeline, "_clip_and_filter", side_effect=clip_and_filter),
            mock.patch.object(
                pipeline,
                "_persist_engine_clip",
                side_effect=lambda **kwargs: (kwargs["v"]["id"], mock.sentinel.metadata),
            ),
        ):
            reels, _ = pipeline.ingest_topic(
                topic=TOPIC,
                material_id="material",
                concept_id="concept",
                max_videos=2,
                max_reels=8,
            )

        self.assertEqual(reels, ["useful-video"])

    def test_non_surfaceable_prerequisite_does_not_prevent_useful_source(self) -> None:
        pipeline = self._pipeline()
        videos = [self._video("blocked-video"), self._video("useful-video")]

        def clip_and_filter(video, *_args):
            if video["id"] == "blocked-video":
                return video, [{
                    "title": "dependent lesson",
                    "score": 1.0,
                    "prerequisite_ids": ["missing-prerequisite"],
                    "search_context": {"surface_eligible": True},
                }], {"transcript": {}}
            time.sleep(0.05)
            return video, [{"title": "useful", "score": 1.0}], {"transcript": {}}

        with (
            mock.patch.object(
                pipeline_module,
                "_discover",
                return_value={
                    "corrected": TOPIC,
                    "videos": videos,
                    "credits_used": 0,
                    "warning": None,
                },
            ),
            mock.patch.object(pipeline_module, "INGEST_TOPIC_VIDEO_TIMEOUT_SEC", 0.3),
            mock.patch.object(pipeline, "_clip_and_filter", side_effect=clip_and_filter),
            mock.patch.object(
                pipeline,
                "_persist_engine_clip",
                side_effect=lambda **kwargs: (kwargs["v"]["id"], mock.sentinel.metadata),
            ),
        ):
            reels, _ = pipeline.ingest_topic(
                topic=TOPIC,
                material_id="material",
                concept_id="concept",
                max_videos=2,
                max_reels=8,
            )

        self.assertEqual(reels, ["useful-video"])

    def test_all_selected_sources_finishing_before_deadline_are_preserved(self) -> None:
        pipeline = self._pipeline()
        videos = [self._video("first-video"), self._video("second-video")]

        def clip_and_filter(video, *_args):
            if video["id"] == "second-video":
                time.sleep(0.03)
            return video, [{"title": video["id"], "score": 1.0}], {"transcript": {}}

        with (
            mock.patch.object(
                pipeline_module,
                "_discover",
                return_value={
                    "corrected": TOPIC,
                    "videos": videos,
                    "credits_used": 0,
                    "warning": None,
                },
            ),
            mock.patch.object(pipeline_module, "INGEST_TOPIC_VIDEO_TIMEOUT_SEC", 0.3),
            mock.patch.object(pipeline, "_clip_and_filter", side_effect=clip_and_filter),
            mock.patch.object(
                pipeline,
                "_persist_engine_clip",
                side_effect=lambda **kwargs: (kwargs["v"]["id"], mock.sentinel.metadata),
            ),
        ):
            reels, _ = pipeline.ingest_topic(
                topic=TOPIC,
                material_id="material",
                concept_id="concept",
                max_videos=2,
                max_reels=8,
            )

        self.assertEqual(reels, ["first-video", "second-video"])

    def test_capped_result_streams_later_video_when_it_finishes_first(self) -> None:
        pipeline = self._pipeline()
        slow = self._video("slow-video")
        fast = self._video("fast-video")
        slow_started = threading.Event()
        fast_finished = threading.Event()
        allow_slow_return = threading.Event()
        seen: list[str] = []

        def clip_and_filter(video, *_args):
            if video["id"] == "slow-video":
                slow_started.set()
                self.assertTrue(fast_finished.wait(1.0))
                self.assertTrue(allow_slow_return.wait(1.0))
            else:
                self.assertTrue(slow_started.wait(1.0))
                fast_finished.set()
            return video, [{"title": video["id"]}], {"transcript": {}}

        def record_seen(reel_id: str) -> None:
            seen.append(reel_id)
            if reel_id == "fast-video":
                allow_slow_return.set()

        with (
            mock.patch.object(
                pipeline_module,
                "_discover",
                return_value={
                    "corrected": TOPIC,
                    "videos": [slow, fast],
                    "credits_used": 0,
                    "warning": None,
                },
            ),
            mock.patch.object(pipeline, "_clip_and_filter", side_effect=clip_and_filter),
            mock.patch.object(
                pipeline,
                "_persist_engine_clip",
                side_effect=lambda **kwargs: (kwargs["v"]["id"], mock.sentinel.metadata),
            ),
        ):
            reels, _ = pipeline.ingest_topic(
                topic=TOPIC,
                material_id="material",
                concept_id="concept",
                max_videos=2,
                max_reels=1,
                on_reel_created=record_seen,
            )

        self.assertEqual(reels, ["fast-video"])
        self.assertEqual(seen, ["fast-video"])

    def test_provider_error_after_progress_keeps_completed_reel(self) -> None:
        pipeline = self._pipeline()
        success = self._video("success-video")
        failure = self._video("failure-video")
        success_emitted = threading.Event()
        seen: list[str] = []
        analyzed: set[str] = set()

        def clip_and_filter(video, *_args):
            if video["id"] == "failure-video":
                self.assertTrue(success_emitted.wait(1.0))
                raise pipeline_module._ClipProviderError(
                    "provider unavailable",
                    provider="test",
                    operation="clip",
                )
            return video, [{"title": video["id"]}], {"transcript": {}}

        def on_reel_created(reel: str) -> None:
            seen.append(reel)
            success_emitted.set()

        with (
            mock.patch.object(
                pipeline_module,
                "_discover",
                return_value={
                    "corrected": TOPIC,
                    "videos": [success, failure],
                    "credits_used": 0,
                    "warning": None,
                },
            ),
            mock.patch.object(pipeline, "_clip_and_filter", side_effect=clip_and_filter),
            mock.patch.object(
                pipeline,
                "_persist_engine_clip",
                side_effect=lambda **kwargs: (kwargs["v"]["id"], mock.sentinel.metadata),
            ),
        ):
            reels, _ = pipeline.ingest_topic(
                topic=TOPIC,
                material_id="material",
                concept_id="concept",
                max_videos=2,
                on_reel_created=on_reel_created,
                analyzed_video_ids=analyzed,
            )

        self.assertEqual(reels, ["success-video"])
        self.assertEqual(seen, ["success-video"])
        self.assertEqual(analyzed, {"success-video"})

    def test_initial_pair_backfills_one_at_a_time_and_stops_at_buffer_cap(self) -> None:
        pipeline = self._pipeline()
        videos = [self._video(f"video-{index}") for index in range(5)]
        calls: list[str] = []
        discover_limit: list[int] = []

        def discover(*_args, **kwargs):
            discover_limit.append(kwargs["limit"])
            return {
                "corrected": TOPIC,
                "videos": videos,
                "credits_used": 0,
                "warning": None,
            }

        def clip_and_filter(video, *_args):
            calls.append(video["id"])
            clips = [] if video["id"] in {"video-0", "video-1"} else [
                {"title": f"clip-{index}"} for index in range(5)
            ]
            return video, clips, {"transcript": {}}

        with (
            mock.patch.object(pipeline_module, "_discover", side_effect=discover),
            mock.patch.object(pipeline, "_clip_and_filter", side_effect=clip_and_filter),
            mock.patch.object(
                pipeline,
                "_persist_engine_clip",
                side_effect=lambda **kwargs: (kwargs["v"]["id"], mock.sentinel.metadata),
            ),
        ):
            reels, _ = pipeline.ingest_topic(
                topic=TOPIC,
                material_id="material",
                concept_id="concept",
                max_videos=4,
            )

        self.assertEqual(discover_limit, [pipeline_module.clip_engine_config.CLIP_SEARCH_MAX_VIDEOS])
        self.assertEqual(set(calls[:2]), {"video-0", "video-1"})
        self.assertEqual(calls[2:], ["video-2"])
        self.assertEqual(reels, ["video-2"] * 5)

    def test_deep_backfill_counts_only_surfaceable_persisted_clips(self) -> None:
        pipeline = self._pipeline()
        videos = [self._video(f"video-{index}") for index in range(3)]
        calls: list[str] = []

        def clip_and_filter(video, *_args):
            calls.append(video["id"])
            return video, [{
                "title": video["id"],
                "start": 0.0,
                "end": 1.0,
                "cue_ids": ["cue-1"],
                "selection_candidate_id": video["id"],
                "search_context": {
                    "surface_eligible": video["id"] == "video-2",
                },
            }], {"transcript": {
                "source": "supadata",
                "native_mode": False,
                "artifact_key": f"supadata:{video['id']}",
                "duration": 1.0,
                "segments": [{
                    "cue_id": "cue-1",
                    "start": 0.0,
                    "end": 1.0,
                    "text": "A complete educational thought.",
                }],
            }}

        with (
            mock.patch.object(
                pipeline_module,
                "_discover",
                return_value={
                    "corrected": TOPIC,
                    "videos": videos,
                    "credits_used": 0,
                    "warning": None,
                },
            ),
            mock.patch.object(pipeline, "_clip_and_filter", side_effect=clip_and_filter),
            mock.patch.object(
                pipeline,
                "_persist_engine_clip",
                side_effect=lambda **kwargs: (kwargs["v"]["id"], mock.sentinel.metadata),
            ),
            mock.patch.object(
                pipeline_module,
                "_supadata_boundary_diagnostics",
                return_value={
                    "method": "test",
                    "start_padding_ms": 0,
                    "end_padding_ms": 0,
                },
            ),
        ):
            reels, _ = pipeline.ingest_topic(
                topic=TOPIC,
                material_id="material",
                concept_id="concept",
                max_videos=3,
                max_reels=1,
                retrieval_profile="deep",
                generation_context=GenerationContext("slow"),
            )

        self.assertEqual(set(calls[:2]), {"video-0", "video-1"})
        self.assertEqual(calls[2:], ["video-2"])
        self.assertEqual(reels, ["video-2"])

    def test_bootstrap_reserves_one_slot_per_initial_source(self) -> None:
        pipeline = self._pipeline()
        videos = [self._video("source-a"), self._video("source-b")]
        seen: list[str] = []

        def clip_and_filter(video, *_args):
            base = 0.9 if video["id"] == "source-b" else 0.8
            return video, [
                {"title": f"{video['id']}-best", "score": base},
                {"title": f"{video['id']}-second", "score": base - 0.1},
            ], {"transcript": {}}

        with (
            mock.patch.object(
                pipeline_module,
                "_discover",
                return_value={
                    "corrected": TOPIC,
                    "videos": videos,
                    "credits_used": 0,
                    "warning": None,
                },
            ),
            mock.patch.object(pipeline, "_clip_and_filter", side_effect=clip_and_filter),
            mock.patch.object(
                pipeline,
                "_persist_engine_clip",
                side_effect=lambda **kwargs: (
                    f"{kwargs['v']['id']}:{kwargs['clip']['title']}",
                    mock.sentinel.metadata,
                ),
            ),
        ):
            reels, _ = pipeline.ingest_topic(
                topic=TOPIC,
                material_id="material",
                concept_id="concept",
                max_videos=2,
                max_reels=2,
                retrieval_profile="bootstrap",
                on_reel_created=seen.append,
            )

        self.assertEqual(
            reels,
            ["source-a:source-a-best", "source-b:source-b-best"],
        )
        self.assertEqual(
            seen,
            ["source-b:source-b-best", "source-a:source-a-best"],
        )

    def test_bootstrap_third_source_backfills_missing_initial_source(self) -> None:
        pipeline = self._pipeline()
        videos = [
            self._video("source-a"),
            self._video("empty-source"),
            self._video("source-c"),
        ]
        calls: list[str] = []

        def clip_and_filter(video, *_args):
            calls.append(video["id"])
            if video["id"] == "empty-source":
                return video, [], {"transcript": {}}
            return video, [
                {"title": f"{video['id']}-best", "score": 0.9},
                {"title": f"{video['id']}-second", "score": 0.8},
            ], {"transcript": {}}

        with (
            mock.patch.object(
                pipeline_module,
                "_discover",
                return_value={
                    "corrected": TOPIC,
                    "videos": videos,
                    "credits_used": 0,
                    "warning": None,
                },
            ),
            mock.patch.object(pipeline, "_clip_and_filter", side_effect=clip_and_filter),
            mock.patch.object(
                pipeline,
                "_persist_engine_clip",
                side_effect=lambda **kwargs: (
                    f"{kwargs['v']['id']}:{kwargs['clip']['title']}",
                    mock.sentinel.metadata,
                ),
            ),
        ):
            reels, _ = pipeline.ingest_topic(
                topic=TOPIC,
                material_id="material",
                concept_id="concept",
                max_videos=3,
                max_reels=2,
                retrieval_profile="bootstrap",
            )

        self.assertEqual(set(calls), {"source-a", "empty-source", "source-c"})
        self.assertEqual(
            reels,
            ["source-a:source-a-best", "source-c:source-c-best"],
        )

    def test_bootstrap_starts_three_together_and_keeps_highest_quality_pair(
        self,
    ) -> None:
        pipeline = self._pipeline()
        videos = [
            self._video("source-a"),
            self._video("source-b"),
            self._video("source-c"),
        ]
        started: set[str] = set()
        started_lock = threading.Lock()
        all_started = threading.Event()

        def clip_and_filter(video, *_args):
            with started_lock:
                started.add(video["id"])
                if len(started) == 3:
                    all_started.set()
            self.assertTrue(
                all_started.wait(1.0),
                "all three bootstrap analyses must be dispatched concurrently",
            )
            score = {"source-a": 0.8, "source-b": 0.7, "source-c": 1.0}[
                video["id"]
            ]
            return video, [
                {"title": f"{video['id']}-best", "score": score},
            ], {"transcript": {}}

        with (
            mock.patch.object(
                pipeline_module,
                "_discover",
                return_value={
                    "corrected": TOPIC,
                    "videos": videos,
                    "credits_used": 0,
                    "warning": None,
                },
            ),
            mock.patch.object(pipeline, "_clip_and_filter", side_effect=clip_and_filter),
            mock.patch.object(
                pipeline,
                "_persist_engine_clip",
                side_effect=lambda **kwargs: (
                    f"{kwargs['v']['id']}:{kwargs['clip']['title']}",
                    mock.sentinel.metadata,
                ),
            ),
        ):
            reels, _ = pipeline.ingest_topic(
                topic=TOPIC,
                material_id="material",
                concept_id="concept",
                max_videos=3,
                max_reels=2,
                retrieval_profile="bootstrap",
            )

        self.assertEqual(started, {"source-a", "source-b", "source-c"})
        self.assertEqual(
            reels,
            ["source-a:source-a-best", "source-c:source-c-best"],
        )


if __name__ == "__main__":
    unittest.main()
