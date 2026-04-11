"""
Integration tests for `POST /api/ingest/topic-cut` and the corresponding
`IngestionPipeline.ingest_topic_cut` method.

Strategy mirrors `test_ingestion_url.py`:
  * Stub the YouTube transcript service so transcribe() returns canned cues
    without touching the network.
  * Install a `_FakeAdapter` so resolve() returns an AdapterResult that points
    at a synthesized fake video file (ffmpeg is mocked too).
  * Mock the OpenAI client so the LLM topic-segmentation call is deterministic.

The fixtures here REUSE the helpers from test_ingestion_url.py instead of
duplicating them — they're already battle-tested for the existing pipeline.

Coverage:
  * Long-form YouTube → multiple reels persisted, classification.is_short=False
  * /shorts/ URL → reels=[], is_short=True, no DB writes
  * LLM disabled (use_llm=False) → heuristic path, still produces ≥1 reel
  * Each persisted reel decodes into the iOS Reel struct shape (every required
    field present, video_url is a YouTube embed with start/end query params)
  * Re-running the same URL is idempotent (unique-index reuse, no duplicates)
"""

from __future__ import annotations

import contextlib
import json
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

from fastapi.testclient import TestClient  # noqa: E402

from backend.app import db as db_module  # noqa: E402
from backend.app.config import get_settings  # noqa: E402
import backend.app.main as main_module  # noqa: E402
from backend.app.main import app  # noqa: E402
from backend.app.ingestion import INGEST_SENTINEL_MATERIAL_ID  # noqa: E402

# Reuse the existing test fixtures verbatim — no point reinventing them.
from backend.tests.test_ingestion_url import (  # noqa: E402
    _FakeAdapter,
    _FakeYoutubeService,
    _fake_info_dict_youtube,
    _patch_ffmpeg_and_ffprobe,
)


def _make_long_transcript_cues() -> list[dict]:
    """
    20 cues × 30 seconds = 10 minutes of fake speech, two clearly distinct topics.
    Cue 0-9 talks about pasta; cue 10-19 talks about JavaScript debugging.
    """
    pasta = [
        ("Today we cook spaghetti carbonara from scratch", 0.0),
        ("First grate the pecorino romano cheese into a bowl", 30.0),
        ("Crack three large eggs into the same mixing bowl", 60.0),
        ("Whisk the eggs and pecorino until fully combined", 90.0),
        ("Boil salted water for the dried spaghetti pasta", 120.0),
        ("Cook guanciale slowly in a wide skillet to render fat", 150.0),
        ("Drop the spaghetti into rapidly boiling salted water", 180.0),
        ("Reserve a full cup of starchy pasta cooking water", 210.0),
        ("Toss drained spaghetti directly into the guanciale skillet off heat", 240.0),
        ("Plate immediately with extra cracked pepper and pecorino on top", 270.0),
    ]
    js = [
        ("Now lets switch gears and debug a tricky JavaScript bug", 300.0),
        ("Open Chrome DevTools with command option I on macOS", 330.0),
        ("Set a breakpoint on line forty-two of the source file", 360.0),
        ("Inspect the call stack panel to trace function entry order", 390.0),
        ("Watch each variable update in the local scope panel live", 420.0),
        ("Use the network tab to inspect failed XHR requests for clues", 450.0),
        ("Add a conditional breakpoint to skip noisy iteration loops", 480.0),
        ("Memory snapshots reveal retained DOM nodes and closures", 510.0),
        ("React DevTools highlight unnecessary component re-renders", 540.0),
        ("That covers the essentials of in-browser JavaScript debugging", 570.0),
    ]
    return [
        {"start": start, "duration": 30.0, "text": text}
        for text, start in (pasta + js)
    ]


def _make_mock_openai_client(segments_payload: dict | None = None) -> mock.MagicMock:
    """Build a chat-completions mock that returns `segments_payload` as JSON."""
    payload = segments_payload or {
        "segments": [
            {"start_idx": 0, "end_idx": 9, "label": "Cooking spaghetti carbonara",
             "summary": "Roman pasta from scratch"},
            {"start_idx": 10, "end_idx": 19, "label": "Debugging JavaScript with DevTools",
             "summary": "Breakpoints and the network panel"},
        ]
    }
    client = mock.MagicMock()
    response = mock.MagicMock()
    choice = mock.MagicMock()
    choice.message.content = json.dumps(payload)
    response.choices = [choice]
    client.chat.completions.create.return_value = response
    # The pipeline's brief_ai_summary helper also calls openai_client.chat.completions.create
    # — share the same mock; it returns the same canned response which brief_ai_summary
    # tolerates because it just stores the .content string in a cache row.
    return client


class IngestionTopicCutTests(unittest.TestCase):
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

        self._fake_yt = _FakeYoutubeService(cues=_make_long_transcript_cues())
        main_module.ingestion_pipeline._youtube_service = self._fake_yt

        self._original_openai_client = main_module.ingestion_pipeline._openai_client
        # Default: pretend no OpenAI client (heuristic fallback). Per-test we
        # override with the mock when we want to exercise the LLM path.
        main_module.ingestion_pipeline._openai_client = None

        main_module.ingestion_pipeline._serverless_mode = False
        main_module._rate_limit_hits.clear()

        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        self.addCleanup(self._restore_pipeline)

    def _restore_environment(self) -> None:
        if self.previous_data_dir is None:
            os.environ.pop("DATA_DIR", None)
        else:
            os.environ["DATA_DIR"] = self.previous_data_dir
        db_module._db_ready = False
        get_settings.cache_clear()
        main_module.settings = get_settings()

    def _restore_pipeline(self) -> None:
        main_module.ingestion_pipeline._openai_client = self._original_openai_client
        main_module.ingestion_pipeline._adapters = [
            adapter
            for adapter in main_module.ingestion_pipeline._adapters
            if not isinstance(adapter, _FakeAdapter)
        ]
        if not main_module.ingestion_pipeline._adapters:
            from backend.app.ingestion.adapters.yt_dlp_adapter import YtDlpAdapter

            main_module.ingestion_pipeline._adapters = [YtDlpAdapter()]

    # ----------------------------------------------------------------- #
    # Long-form happy path with LLM
    # ----------------------------------------------------------------- #

    def test_long_form_llm_path_returns_multiple_reels(self) -> None:
        source_url = "https://www.youtube.com/watch?v=aircAruvnKk"
        fake_adapter = _FakeAdapter(
            platform="yt",
            info_dict=_fake_info_dict_youtube(),
            source_url=source_url,
            source_id="aircAruvnKk",
        )
        main_module.ingestion_pipeline._adapters = [fake_adapter]
        main_module.ingestion_pipeline._openai_client = _make_mock_openai_client()

        with contextlib.ExitStack() as stack:
            # The transcript fixture spans 600s, so probe_duration must agree.
            for patch in _patch_ffmpeg_and_ffprobe(probe_duration_sec=720.0):
                stack.enter_context(patch)
            response = self.client.post(
                "/api/ingest/topic-cut",
                json={"source_url": source_url, "use_llm": True},
            )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()

        # Envelope
        self.assertEqual(payload["video_id"], "aircAruvnKk")
        self.assertFalse(payload["is_short"])
        self.assertEqual(payload["reel_count"], len(payload["reels"]))
        self.assertGreaterEqual(payload["reel_count"], 2,
                                f"expected ≥2 reels, got {payload['reel_count']}")
        self.assertTrue(payload["terms_notice"])
        self.assertTrue(payload["trace_id"])

        # Each reel decodes into the iOS Reel shape (the existing decoder
        # tolerates extra fields, but every REQUIRED field must be present).
        required_fields = {
            "reel_id", "material_id", "concept_id", "concept_title",
            "video_title", "video_description", "ai_summary", "video_url",
            "t_start", "t_end", "transcript_snippet", "takeaways", "captions",
            "score",
        }
        for reel in payload["reels"]:
            missing = required_fields - reel.keys()
            self.assertEqual(missing, set(), f"reel missing fields: {missing}")
            self.assertTrue(reel["reel_id"].startswith("topic-"))
            self.assertEqual(reel["material_id"], INGEST_SENTINEL_MATERIAL_ID)
            self.assertTrue(reel["video_url"].startswith(
                "https://www.youtube.com/embed/aircAruvnKk"))
            self.assertIn("start=", reel["video_url"])
            self.assertIn("end=", reel["video_url"])
            # Each clip must be a valid window inside [30s, 12min].
            duration = reel["t_end"] - reel["t_start"]
            self.assertGreaterEqual(duration, 30,
                                    f"clip too short: {duration}s")
            self.assertLessEqual(duration, 12 * 60,
                                 f"clip too long: {duration}s")
            self.assertEqual(reel["source_surface"], "ingest:yt:topic_cut")

        # Reels are ordered by t_start
        starts = [r["t_start"] for r in payload["reels"]]
        self.assertEqual(starts, sorted(starts))

        # DB sanity — every persisted reel exists in the reels table
        with db_module.get_conn() as conn:
            for reel in payload["reels"]:
                row = db_module.fetch_one(
                    conn,
                    "SELECT id, video_id, t_start, t_end FROM reels WHERE id = ?",
                    (reel["reel_id"],),
                )
                self.assertIsNotNone(row, f"reel {reel['reel_id']} missing from DB")
                self.assertEqual(row["video_id"], "yt:aircAruvnKk")

    # ----------------------------------------------------------------- #
    # YouTube Short → empty reels, no DB writes
    # ----------------------------------------------------------------- #

    def test_shorts_url_returns_empty_reels_and_no_db_writes(self) -> None:
        source_url = "https://www.youtube.com/shorts/1AlFJOxAY00"
        info_dict = dict(_fake_info_dict_youtube(source_id="1AlFJOxAY00"))
        info_dict["duration"] = 42.0  # consistent with a Short
        fake_adapter = _FakeAdapter(
            platform="yt",
            info_dict=info_dict,
            source_url=source_url,
            source_id="1AlFJOxAY00",
        )
        main_module.ingestion_pipeline._adapters = [fake_adapter]

        with contextlib.ExitStack() as stack:
            for patch in _patch_ffmpeg_and_ffprobe(probe_duration_sec=42.0):
                stack.enter_context(patch)
            response = self.client.post(
                "/api/ingest/topic-cut",
                json={"source_url": source_url},
            )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertTrue(payload["is_short"])
        self.assertEqual(payload["reels"], [])
        self.assertEqual(payload["reel_count"], 0)
        self.assertIn("/shorts/", payload["classification_reason"])

        # No reel rows should have been written for this video.
        with db_module.get_conn() as conn:
            row = db_module.fetch_one(
                conn,
                "SELECT COUNT(*) AS c FROM reels WHERE video_id = ?",
                ("yt:1AlFJOxAY00",),
            )
            self.assertEqual(int(row["c"]), 0)

    # ----------------------------------------------------------------- #
    # use_llm=False → heuristic path still produces reels
    # ----------------------------------------------------------------- #

    def test_heuristic_path_still_produces_reels(self) -> None:
        source_url = "https://www.youtube.com/watch?v=aircAruvnKk"
        fake_adapter = _FakeAdapter(
            platform="yt",
            info_dict=_fake_info_dict_youtube(),
            source_url=source_url,
            source_id="aircAruvnKk",
        )
        main_module.ingestion_pipeline._adapters = [fake_adapter]
        # Pass an OpenAI client too, to prove `use_llm: false` short-circuits it.
        client = _make_mock_openai_client()
        main_module.ingestion_pipeline._openai_client = client

        with contextlib.ExitStack() as stack:
            for patch in _patch_ffmpeg_and_ffprobe(probe_duration_sec=720.0):
                stack.enter_context(patch)
            response = self.client.post(
                "/api/ingest/topic-cut",
                json={"source_url": source_url, "use_llm": False},
            )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertFalse(payload["is_short"])
        self.assertGreaterEqual(payload["reel_count"], 1)
        # The chat completions endpoint must NOT have been called for topic
        # segmentation. brief_ai_summary may still call it, so we check for the
        # specific system prompt that topic_cut sends.
        for call in client.chat.completions.create.call_args_list:
            kwargs = call.kwargs
            messages = kwargs.get("messages") or []
            for msg in messages:
                content = msg.get("content") if isinstance(msg, dict) else None
                if isinstance(content, str):
                    self.assertNotIn("precise video editor", content.lower())

    # ----------------------------------------------------------------- #
    # Idempotency — re-ingesting the same URL doesn't create duplicate rows
    # ----------------------------------------------------------------- #

    def test_re_ingest_is_idempotent(self) -> None:
        source_url = "https://www.youtube.com/watch?v=aircAruvnKk"
        fake_adapter = _FakeAdapter(
            platform="yt",
            info_dict=_fake_info_dict_youtube(),
            source_url=source_url,
            source_id="aircAruvnKk",
        )
        main_module.ingestion_pipeline._adapters = [fake_adapter]
        main_module.ingestion_pipeline._openai_client = _make_mock_openai_client()

        with contextlib.ExitStack() as stack:
            for patch in _patch_ffmpeg_and_ffprobe(probe_duration_sec=720.0):
                stack.enter_context(patch)
            r1 = self.client.post("/api/ingest/topic-cut",
                                  json={"source_url": source_url})
            r2 = self.client.post("/api/ingest/topic-cut",
                                  json={"source_url": source_url})

        self.assertEqual(r1.status_code, 200, r1.text)
        self.assertEqual(r2.status_code, 200, r2.text)
        ids1 = sorted(r["reel_id"] for r in r1.json()["reels"])
        ids2 = sorted(r["reel_id"] for r in r2.json()["reels"])
        # Reused via load_existing_reel — same reel_ids both runs.
        self.assertEqual(ids1, ids2)

        # Confirm the DB only contains one row per (video_id, t_start, t_end).
        with db_module.get_conn() as conn:
            row = db_module.fetch_one(
                conn,
                """
                SELECT COUNT(*) AS c FROM (
                    SELECT video_id, t_start, t_end, COUNT(*) AS n
                      FROM reels
                     WHERE video_id = ?
                  GROUP BY video_id, t_start, t_end
                    HAVING n > 1
                )
                """,
                ("yt:aircAruvnKk",),
            )
            self.assertEqual(int(row["c"]), 0)


    # ----------------------------------------------------------------- #
    # YouTube chapters short-circuit the LLM/heuristic
    # ----------------------------------------------------------------- #

    def test_chapters_in_info_dict_drive_reels_directly(self) -> None:
        source_url = "https://www.youtube.com/watch?v=aircAruvnKk"
        # Build an info_dict with creator-authored chapter markers. The
        # _FakeAdapter passes its info_dict straight through, so the pipeline
        # will see them in the same shape yt-dlp produces.
        info_dict = dict(_fake_info_dict_youtube())
        info_dict["chapters"] = [
            {"start_time": 0.0, "end_time": 60.0, "title": "Intro"},  # fluff, dropped
            {"start_time": 60.0, "end_time": 200.0, "title": "Cooking spaghetti carbonara"},
            {"start_time": 200.0, "end_time": 360.0, "title": "Debugging JavaScript with DevTools"},
            {"start_time": 360.0, "end_time": 540.0, "title": "Closing thoughts"},
        ]
        fake_adapter = _FakeAdapter(
            platform="yt",
            info_dict=info_dict,
            source_url=source_url,
            source_id="aircAruvnKk",
        )
        main_module.ingestion_pipeline._adapters = [fake_adapter]
        # Provide a mock OpenAI client so we can prove it's NOT called for
        # topic segmentation when chapters are present.
        client = _make_mock_openai_client()
        main_module.ingestion_pipeline._openai_client = client

        with contextlib.ExitStack() as stack:
            for patch in _patch_ffmpeg_and_ffprobe(probe_duration_sec=720.0):
                stack.enter_context(patch)
            response = self.client.post(
                "/api/ingest/topic-cut",
                json={"source_url": source_url, "use_llm": True},
            )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertFalse(payload["is_short"])

        labels = [r["concept_title"] for r in payload["reels"]]
        # The "Intro" chapter is fluff and must be skipped.
        self.assertNotIn("Intro", labels)
        # The two real chapters should appear as standalone reels.
        self.assertIn("Cooking spaghetti carbonara", labels)
        self.assertIn("Debugging JavaScript with DevTools", labels)
        self.assertIn("Closing thoughts", labels)

        # The LLM topic-segmentation prompt MUST NOT have been sent. The
        # mock may still be called by brief_ai_summary for per-reel headlines,
        # but we can detect the topic-cut prompt by its system content.
        for call in client.chat.completions.create.call_args_list:
            messages = call.kwargs.get("messages") or []
            for msg in messages:
                if isinstance(msg, dict) and msg.get("role") == "system":
                    content = msg.get("content") or ""
                    self.assertNotIn(
                        "precise video editor",
                        content.lower(),
                        "topic_cut LLM prompt was sent even though chapters were present",
                    )

        # Persistence sanity — the chapter-driven reels exist in the DB.
        with db_module.get_conn() as conn:
            for reel in payload["reels"]:
                row = db_module.fetch_one(
                    conn,
                    "SELECT id, video_id FROM reels WHERE id = ?",
                    (reel["reel_id"],),
                )
                self.assertIsNotNone(row, f"chapter reel {reel['reel_id']} missing from DB")


if __name__ == "__main__":
    unittest.main()
