"""
Tests for the reel ingestion pipeline (`backend/app/ingestion/`).

Strategy: mock out the three heavy external surfaces (yt-dlp, ffmpeg subprocess, Whisper
API) so the tests run on any developer machine in <2 seconds and never make a network
request. Every path through the pipeline except the actual download is exercised.

Matches the existing `backend/tests/` convention: unittest.TestCase + TestClient(app) +
DATA_DIR override + `db_module._db_ready = False` + `get_settings.cache_clear()`.
"""

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

# Skip the import-time orphan sweep so tests don't poke at /tmp during import.
os.environ.setdefault("REELAI_INGEST_SKIP_IMPORT_SWEEP", "1")

from fastapi.testclient import TestClient  # noqa: E402

from backend.app import db as db_module  # noqa: E402
from backend.app.config import get_settings  # noqa: E402
import backend.app.main as main_module  # noqa: E402
from backend.app.main import app  # noqa: E402
from backend.app.ingestion import INGEST_SENTINEL_MATERIAL_ID  # noqa: E402
from backend.app.ingestion.adapters import base as adapter_base_module  # noqa: E402
from backend.app.ingestion import ffmpeg_tools as ffmpeg_tools_module  # noqa: E402
from backend.app.ingestion import pipeline as pipeline_module  # noqa: E402
from backend.app.ingestion import transcribe as transcribe_module  # noqa: E402
from backend.app.ingestion.models import IngestTranscriptCue  # noqa: E402


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


def _fake_info_dict_youtube(source_id: str = "aircAruvnKk") -> dict:
    return {
        "id": source_id,
        "title": "But what is a neural network? | Chapter 1",
        "uploader": "3Blue1Brown",
        "uploader_id": "3blue1brown",
        "uploader_url": "https://www.youtube.com/@3blue1brown",
        "channel": "3Blue1Brown",
        "description": (
            "What are the neurons, why are there layers, and what is the math underlying it? "
            "#neuralnetworks #math #education"
        ),
        "duration": 720.0,
        "view_count": 12345678,
        "like_count": 250000,
        "comment_count": 4200,
        "thumbnail": "https://i.ytimg.com/vi/test/hqdefault.jpg",
        "upload_date": "20171005",
        "webpage_url": f"https://www.youtube.com/watch?v={source_id}",
        "tags": ["math", "neural networks", "deep learning"],
        "categories": ["Education"],
        "automatic_captions": {},
        "subtitles": {},
    }


def _fake_info_dict_instagram(shortcode: str = "C8abc123") -> dict:
    return {
        "id": shortcode,
        "title": "NASA Reel",
        "description": "Behind the scenes at the Artemis II integration. #nasa #artemis #moon",
        "uploader": "nasa",
        "uploader_id": "nasa",
        "uploader_url": "https://www.instagram.com/nasa/",
        "duration": 42.0,
        "view_count": 98765,
        "like_count": 15000,
        "comment_count": 300,
        "thumbnail": "https://scontent.cdninstagram.com/thumb.jpg",
        "upload_date": "20250215",
        "webpage_url": f"https://www.instagram.com/reel/{shortcode}/",
        "automatic_captions": {},
        "subtitles": {},
    }


def _synthesize_fake_video_file(workspace: Path) -> Path:
    path = workspace / "source.mp4"
    # Write a non-empty placeholder; ffmpeg probe is mocked out so the content never matters.
    path.write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + b"\xff" * 1024)
    return path


class _FakeYoutubeService:
    """Drop-in stand-in for YouTubeService that returns canned transcript cues."""

    def __init__(self, cues: list[dict] | None = None) -> None:
        self._cues = cues or [
            {"start": 0.0, "duration": 5.0, "text": "Welcome to the neural network chapter."},
            {"start": 5.0, "duration": 5.0, "text": "Neurons are simple functions in a layered graph."},
            {"start": 10.0, "duration": 5.0, "text": "Training adjusts the weights via gradient descent."},
            {"start": 15.0, "duration": 5.0, "text": "Loss functions quantify how wrong the prediction is."},
            {"start": 20.0, "duration": 5.0, "text": "Backpropagation pushes error gradients backwards."},
            {"start": 25.0, "duration": 5.0, "text": "Chain rule gives a local update for every parameter."},
            {"start": 30.0, "duration": 5.0, "text": "Repeat for many batches and the network converges."},
            {"start": 35.0, "duration": 5.0, "text": "Regularization keeps the model from memorizing."},
            {"start": 40.0, "duration": 5.0, "text": "Validation splits measure generalization."},
            {"start": 45.0, "duration": 5.0, "text": "Hyperparameter search tunes the learning rate."},
            {"start": 50.0, "duration": 5.0, "text": "Deeper networks can learn more abstract patterns."},
            {"start": 55.0, "duration": 5.0, "text": "We'll see this applied to handwritten digits next."},
        ]

    def get_transcript(self, conn, video_id: str) -> list[dict]:  # noqa: ARG002
        return self._cues


class _FakeAdapter:
    """
    Stand-in for YtDlpAdapter that returns a pre-baked AdapterResult without touching yt-dlp.

    Tests install it via `pipeline._adapters = [fake]` inside a patch context.
    """

    name = "fake"

    def __init__(self, platform: str, info_dict: dict, source_url: str, source_id: str) -> None:
        self._platform = platform
        self._info_dict = info_dict
        self._source_url = source_url
        self._source_id = source_id
        self.resolve_calls = 0
        self.resolve_feed_calls = 0

    def supports(self, url: str) -> bool:
        return True

    def platform_for(self, url: str) -> str:
        return self._platform

    def resolve(self, url: str, workspace: Path):
        self.resolve_calls += 1
        video_path = _synthesize_fake_video_file(workspace)
        playback = self._source_url
        if self._platform == "yt":
            playback = f"https://www.youtube.com/embed/{self._source_id}"
        return adapter_base_module.AdapterResult(
            platform=self._platform,
            source_id=self._source_id,
            source_url=url,
            playback_url=playback,
            video_path=video_path,
            info_dict=self._info_dict,
        )

    def resolve_feed(self, url: str, *, max_items: int) -> list[str]:
        self.resolve_feed_calls += 1
        return [
            f"https://www.instagram.com/reel/feed-item-{i}/"
            for i in range(min(max_items, 3))
        ]


class _FakeSearchAdapter:
    """
    Search-capable stand-in. Exposes `build_search_url`, `resolve_feed`, and
    `extract_source_id_from_url` so the pipeline's ingest_search() path can run end-to-end
    without touching yt-dlp or the network.

    Resolves URL lists per platform from `url_lists_by_platform` (a dict of
    platform → list of (source_id, reel_url) tuples). A platform whose key is
    missing from the dict but present in `fail_platforms` raises DownloadError
    instead. Each `resolve(url, ...)` call returns an AdapterResult whose source_id
    is parsed from the reel_url (last path segment), so the per-item ingest_url()
    pipeline produces a unique reel per URL.
    """

    name = "fake-search"

    def __init__(
        self,
        *,
        url_lists_by_platform: dict[str, list[tuple[str, str]]],
        fail_platforms: set[str] | None = None,
        default_info_dict: dict | None = None,
    ) -> None:
        self._url_lists = url_lists_by_platform
        self._fail_platforms = fail_platforms or set()
        self._default_info_dict = default_info_dict or _fake_info_dict_youtube()
        # Reverse index: reel_url → (platform, source_id)
        self._url_to_meta: dict[str, tuple[str, str]] = {}
        for platform, pairs in url_lists_by_platform.items():
            for source_id, url in pairs:
                self._url_to_meta[url] = (platform, source_id)
        self.build_search_url_calls: list[tuple[str, str, int]] = []
        self.resolve_feed_calls: list[str] = []
        self.resolve_calls: list[str] = []

    def supports(self, url: str) -> bool:
        return True

    def platform_for(self, url: str) -> str:
        if url.startswith("ytsearch"):
            return "yt"
        if url in self._url_to_meta:
            return self._url_to_meta[url][0]
        if "instagram.com" in url:
            return "ig"
        if "tiktok.com" in url:
            return "tt"
        return "yt"

    def build_search_url(self, query: str, platform: str, max_items: int) -> str:
        self.build_search_url_calls.append((query, platform, max_items))
        if platform == "yt":
            return f"ytsearch{max_items}:{query}"
        if platform == "ig":
            return f"https://www.instagram.com/explore/tags/{query.replace(' ', '')}/"
        if platform == "tt":
            return f"https://www.tiktok.com/search?q={query}"
        raise ValueError(f"unknown platform {platform}")

    def resolve_feed(self, url: str, *, max_items: int) -> list[str]:
        self.resolve_feed_calls.append(url)
        platform = self.platform_for(url)
        if platform in self._fail_platforms:
            from backend.app.ingestion.errors import DownloadError

            raise DownloadError(f"fake {platform} failure")
        pairs = self._url_lists.get(platform, [])
        return [pair[1] for pair in pairs[:max_items]]

    def extract_source_id_from_url(self, url: str, platform: str | None = None):
        meta = self._url_to_meta.get(url)
        if meta:
            return meta[1]
        return None

    def resolve(self, url: str, workspace: Path):
        self.resolve_calls.append(url)
        meta = self._url_to_meta.get(url)
        if meta:
            platform_code, source_id = meta
        else:
            platform_code = self.platform_for(url)
            source_id = url.rstrip("/").rsplit("/", 1)[-1].split("?")[0] or "unknown"
        video_path = _synthesize_fake_video_file(workspace)
        info = dict(self._default_info_dict)
        info["id"] = source_id
        playback = url
        if platform_code == "yt":
            playback = f"https://www.youtube.com/embed/{source_id}"
        return adapter_base_module.AdapterResult(
            platform=platform_code,
            source_id=source_id,
            source_url=url,
            playback_url=playback,
            video_path=video_path,
            info_dict=info,
        )


def _patch_ffmpeg_and_ffprobe(probe_duration_sec: float = 720.0):
    """
    Install lightweight stand-ins for every ffmpeg_tools helper the pipeline touches.
    Returns a contextlib.ExitStack-like object that tests close in tearDown.
    """
    patches = [
        mock.patch.object(ffmpeg_tools_module, "check_ffmpeg_available", return_value=True),
        mock.patch.object(pipeline_module, "check_ffmpeg_available", return_value=True),
        mock.patch.object(ffmpeg_tools_module, "probe_duration", return_value=probe_duration_sec),
        mock.patch.object(pipeline_module, "probe_duration", return_value=probe_duration_sec),
        mock.patch.object(ffmpeg_tools_module, "silencedetect", return_value=[]),
        mock.patch.object(pipeline_module, "silencedetect", return_value=[]),
        mock.patch.object(
            ffmpeg_tools_module,
            "extract_audio_wav",
            side_effect=lambda video_path, out_path, **_: Path(out_path).write_bytes(b"RIFFfakeWAVE") or Path(out_path),
        ),
        mock.patch.object(
            transcribe_module,
            "extract_audio_wav",
            side_effect=lambda video_path, out_path, **_: Path(out_path).write_bytes(b"RIFFfakeWAVE") or Path(out_path),
        ),
    ]
    return patches


# --------------------------------------------------------------------- #
# TestCase
# --------------------------------------------------------------------- #


class IngestionUrlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.previous_data_dir = os.environ.get("DATA_DIR")
        os.environ["DATA_DIR"] = self.temp_dir.name
        self.addCleanup(self._restore_environment)
        db_module._db_ready = False
        get_settings.cache_clear()
        main_module.settings = get_settings()

        # Reset the in-memory orphan sweeper to avoid touching /tmp during tests.
        os.environ["REELAI_INGEST_SKIP_IMPORT_SWEEP"] = "1"

        # Stub the YouTube service on the module-level pipeline so transcript strategy 1
        # returns canned cues without hitting the network.
        self._fake_yt = _FakeYoutubeService()
        main_module.ingestion_pipeline._youtube_service = self._fake_yt

        # Ensure Whisper isn't invoked by default — we patch it at a per-test granularity.
        self._original_openai_client = main_module.ingestion_pipeline._openai_client
        main_module.ingestion_pipeline._openai_client = None

        # Force the pipeline out of "serverless" mode so the preflight check passes.
        main_module.ingestion_pipeline._serverless_mode = False

        # Clear the per-IP rate-limit buckets so tests can make multiple requests
        # back-to-back without bumping against the endpoint's own sliding-window limiter.
        main_module._rate_limit_hits.clear()

        self.client = TestClient(app)
        self.addCleanup(self.client.close)

        # Every test restores the pipeline state at tear-down.
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

    # --------------------------------------------------------------------- #
    # Happy path — YouTube via captions, no Whisper needed
    # --------------------------------------------------------------------- #

    def test_ingest_url_youtube_happy_path(self) -> None:
        source_url = "https://www.youtube.com/watch?v=aircAruvnKk"
        fake_adapter = _FakeAdapter(
            platform="yt",
            info_dict=_fake_info_dict_youtube(),
            source_url=source_url,
            source_id="aircAruvnKk",
        )
        main_module.ingestion_pipeline._adapters = [fake_adapter]

        with contextlib.ExitStack() as stack:
            for patch in _patch_ffmpeg_and_ffprobe():
                stack.enter_context(patch)

            response = self.client.post(
                "/api/ingest/url",
                json={
                    "source_url": source_url,
                    "target_clip_duration_sec": 45,
                    "target_clip_duration_min_sec": 15,
                    "target_clip_duration_max_sec": 60,
                },
            )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()

        # Envelope shape
        self.assertIn("reel", payload)
        self.assertIn("metadata", payload)
        self.assertIn("terms_notice", payload)
        self.assertIn("trace_id", payload)
        self.assertTrue(payload["terms_notice"].lower().startswith("reelai"))

        # Reel fields
        reel = payload["reel"]
        self.assertTrue(reel["reel_id"].startswith("ingest-"))
        self.assertEqual(reel["material_id"], INGEST_SENTINEL_MATERIAL_ID)
        self.assertTrue(reel["video_url"].startswith("https://www.youtube.com/embed/aircAruvnKk"))
        self.assertIn("start=", reel["video_url"])
        self.assertIn("end=", reel["video_url"])
        self.assertGreaterEqual(reel["t_end"] - reel["t_start"], 15)
        self.assertLessEqual(reel["t_end"] - reel["t_start"], 60)
        self.assertTrue(reel["transcript_snippet"])
        self.assertGreaterEqual(len(reel["captions"]), 1)
        self.assertEqual(reel["source_surface"], "ingest:yt")
        self.assertIsNotNone(reel.get("source_attribution"))
        self.assertIn("3blue1brown", reel["source_attribution"].lower())

        # Metadata fields
        metadata = payload["metadata"]
        self.assertEqual(metadata["platform"], "yt")
        self.assertEqual(metadata["source_id"], "aircAruvnKk")
        self.assertIn("neuralnetworks", metadata["hashtags"])
        self.assertEqual(metadata["view_count"], 12345678)

        # DB sanity — the reel row exists and a metadata blob sits in llm_cache
        with db_module.get_conn() as conn:
            row = db_module.fetch_one(
                conn,
                "SELECT id, video_id, video_url FROM reels WHERE id = ?",
                (reel["reel_id"],),
            )
            self.assertIsNotNone(row)
            self.assertEqual(row["video_id"], "yt:aircAruvnKk")

            blob = db_module.fetch_one(
                conn,
                "SELECT response_json FROM llm_cache WHERE cache_key = ?",
                (f"ingest_meta:{reel['reel_id']}",),
            )
            self.assertIsNotNone(blob)
            blob_data = json.loads(blob["response_json"])
            self.assertEqual(blob_data["platform"], "yt")
            self.assertEqual(blob_data["author_handle"], "3blue1brown")
            self.assertEqual(blob_data["source_id"], "aircAruvnKk")

        self.assertEqual(fake_adapter.resolve_calls, 1)

    # --------------------------------------------------------------------- #
    # Whisper fallback path — Instagram with no captions
    # --------------------------------------------------------------------- #

    def test_ingest_url_instagram_whisper_fallback(self) -> None:
        source_url = "https://www.instagram.com/reel/C8abc123/"
        fake_adapter = _FakeAdapter(
            platform="ig",
            info_dict=_fake_info_dict_instagram(),
            source_url=source_url,
            source_id="C8abc123",
        )
        main_module.ingestion_pipeline._adapters = [fake_adapter]
        main_module.ingestion_pipeline._openai_client = object()  # non-None stand-in; Whisper is patched below

        fake_whisper_cues = [
            IngestTranscriptCue(start=float(i * 3.0), end=float(i * 3.0 + 3.0), text=f"whisper segment {i}")
            for i in range(14)
        ]

        with contextlib.ExitStack() as stack:
            for patch in _patch_ffmpeg_and_ffprobe(probe_duration_sec=42.0):
                stack.enter_context(patch)
            stack.enter_context(
                mock.patch.object(
                    transcribe_module,
                    "_whisper_transcribe",
                    return_value=fake_whisper_cues,
                )
            )

            response = self.client.post(
                "/api/ingest/url",
                json={"source_url": source_url},
            )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        reel = payload["reel"]
        metadata = payload["metadata"]

        self.assertEqual(metadata["platform"], "ig")
        self.assertEqual(reel["video_url"], source_url, "IG playback URL must be the original")
        self.assertEqual(reel["source_surface"], "ingest:ig")
        self.assertIn("nasa", reel["source_attribution"].lower())
        self.assertGreaterEqual(len(reel["captions"]), 1)

    # --------------------------------------------------------------------- #
    # Duplicate ingest returns the existing reel id instead of crashing
    # --------------------------------------------------------------------- #

    def test_duplicate_ingest_idempotent(self) -> None:
        source_url = "https://www.youtube.com/watch?v=aircAruvnKk"
        fake_adapter = _FakeAdapter(
            platform="yt",
            info_dict=_fake_info_dict_youtube(),
            source_url=source_url,
            source_id="aircAruvnKk",
        )
        main_module.ingestion_pipeline._adapters = [fake_adapter]

        with contextlib.ExitStack() as stack:
            for patch in _patch_ffmpeg_and_ffprobe():
                stack.enter_context(patch)

            first = self.client.post("/api/ingest/url", json={"source_url": source_url})
            self.assertEqual(first.status_code, 200, first.text)
            second = self.client.post("/api/ingest/url", json={"source_url": source_url})
            self.assertEqual(second.status_code, 200, second.text)

        # Same source → same (start, end) → the second call must reuse the row
        first_reel = first.json()["reel"]
        second_reel = second.json()["reel"]
        self.assertEqual(first_reel["t_start"], second_reel["t_start"])
        self.assertEqual(first_reel["t_end"], second_reel["t_end"])
        self.assertEqual(first_reel["reel_id"], second_reel["reel_id"])

        # Only one row in the reels table
        with db_module.get_conn() as conn:
            rows = db_module.fetch_all(
                conn,
                "SELECT id FROM reels WHERE material_id = ?",
                (INGEST_SENTINEL_MATERIAL_ID,),
            )
            self.assertEqual(len(rows), 1)

    # --------------------------------------------------------------------- #
    # Feed crawl
    # --------------------------------------------------------------------- #

    def test_ingest_feed_crawls_multiple_items(self) -> None:
        fake_adapter = _FakeAdapter(
            platform="ig",
            info_dict=_fake_info_dict_instagram(),
            source_url="https://www.instagram.com/nasa/",
            source_id="C8abc123",
        )
        main_module.ingestion_pipeline._adapters = [fake_adapter]
        main_module.ingestion_pipeline._openai_client = object()

        fake_whisper_cues = [
            IngestTranscriptCue(start=float(i * 3.0), end=float(i * 3.0 + 3.0), text=f"feed segment {i}")
            for i in range(14)
        ]

        # Each reel in the feed should resolve through the fake adapter. The adapter returns
        # the same info_dict but with a different shortcode (derived from the URL).
        original_resolve = fake_adapter.resolve
        source_ids_seen: list[str] = []

        def resolve_with_id_from_url(url: str, workspace: Path):
            # Derive a unique id per URL so the unique index doesn't collide across items.
            idx = url.rstrip("/").rsplit("/", 1)[-1]
            source_ids_seen.append(idx)
            info = dict(_fake_info_dict_instagram(shortcode=idx))
            info["id"] = idx
            video_path = _synthesize_fake_video_file(workspace)
            return adapter_base_module.AdapterResult(
                platform="ig",
                source_id=idx,
                source_url=url,
                playback_url=url,
                video_path=video_path,
                info_dict=info,
            )

        fake_adapter.resolve = resolve_with_id_from_url  # type: ignore[method-assign]

        with contextlib.ExitStack() as stack:
            for patch in _patch_ffmpeg_and_ffprobe(probe_duration_sec=42.0):
                stack.enter_context(patch)
            stack.enter_context(
                mock.patch.object(
                    transcribe_module,
                    "_whisper_transcribe",
                    return_value=fake_whisper_cues,
                )
            )
            response = self.client.post(
                "/api/ingest/feed",
                json={"feed_url": "https://www.instagram.com/nasa/", "max_items": 3},
            )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["total_resolved"], 3)
        self.assertEqual(payload["succeeded"], 3)
        self.assertEqual(payload["failed"], 0)
        self.assertEqual(len(payload["items"]), 3)
        for item in payload["items"]:
            self.assertEqual(item["status"], "ok")
            self.assertIsNotNone(item["reel"])
            self.assertIsNotNone(item["metadata"])

        fake_adapter.resolve = original_resolve  # type: ignore[method-assign]
        self.assertGreaterEqual(fake_adapter.resolve_feed_calls, 1)

    # --------------------------------------------------------------------- #
    # Serverless gating: endpoint refuses with 503 in SERVERLESS_MODE
    # --------------------------------------------------------------------- #

    def test_serverless_gate_returns_503(self) -> None:
        previous_serverless = main_module.SERVERLESS_MODE
        main_module.SERVERLESS_MODE = True
        previous_override = os.environ.pop("ALLOW_OPENAI_IN_SERVERLESS", None)
        self.addCleanup(lambda: setattr(main_module, "SERVERLESS_MODE", previous_serverless))
        if previous_override is not None:
            self.addCleanup(lambda: os.environ.__setitem__("ALLOW_OPENAI_IN_SERVERLESS", previous_override))

        response = self.client.post(
            "/api/ingest/url",
            json={"source_url": "https://www.youtube.com/watch?v=aircAruvnKk"},
        )
        self.assertEqual(response.status_code, 503, response.text)
        body = response.json()
        self.assertIn("ServerlessUnavailable", json.dumps(body))

    # --------------------------------------------------------------------- #
    # Unsupported host → 400
    # --------------------------------------------------------------------- #

    def test_unsupported_host_returns_400(self) -> None:
        # Restore the real YtDlpAdapter so the host check runs.
        from backend.app.ingestion.adapters.yt_dlp_adapter import YtDlpAdapter

        main_module.ingestion_pipeline._adapters = [YtDlpAdapter()]

        with mock.patch.object(pipeline_module, "check_ffmpeg_available", return_value=True):
            response = self.client.post(
                "/api/ingest/url",
                json={"source_url": "https://vimeo.com/123456789"},
            )

        self.assertEqual(response.status_code, 400, response.text)
        body = response.json()
        # The detail payload is a dict with an "error" field
        detail = body.get("detail")
        self.assertIsInstance(detail, dict)
        self.assertEqual(detail.get("error"), "UnsupportedSourceError")

    # --------------------------------------------------------------------- #
    # Rate limiter — per-platform cap raises 429 after the budget is exhausted
    # --------------------------------------------------------------------- #

    def test_platform_rate_limiter_triggers_429(self) -> None:
        from backend.app.ingestion.pipeline import _PlatformRateLimiter
        from backend.app.ingestion.errors import RateLimitedError

        limiter = _PlatformRateLimiter(overrides={"yt": (1, 60.0)})
        limiter.acquire("yt")
        with self.assertRaises(RateLimitedError):
            limiter.acquire("yt")

    # --------------------------------------------------------------------- #
    # Topic search — happy path across all three platforms
    # --------------------------------------------------------------------- #

    def _install_high_rate_limits(self) -> None:
        """Search tests ingest 6-9 URLs, so bump per-platform limits to avoid test noise."""
        from backend.app.ingestion.pipeline import _PlatformRateLimiter

        main_module.ingestion_pipeline._rate_limiter = _PlatformRateLimiter(
            overrides={"yt": (1000, 60.0), "ig": (1000, 60.0), "tt": (1000, 60.0)}
        )

    def test_ingest_search_multi_platform_happy_path(self) -> None:
        self._install_high_rate_limits()
        main_module.ingestion_pipeline._openai_client = object()

        url_lists = {
            "yt": [
                ("ytA", "https://www.youtube.com/watch?v=ytA"),
                ("ytB", "https://www.youtube.com/watch?v=ytB"),
            ],
            "ig": [
                ("igA", "https://www.instagram.com/reel/igA/"),
                ("igB", "https://www.instagram.com/reel/igB/"),
            ],
            "tt": [
                ("ttA", "https://www.tiktok.com/@user/video/ttA"),
                ("ttB", "https://www.tiktok.com/@user/video/ttB"),
            ],
        }
        fake_adapter = _FakeSearchAdapter(url_lists_by_platform=url_lists)
        main_module.ingestion_pipeline._adapters = [fake_adapter]

        fake_whisper_cues = [
            IngestTranscriptCue(start=float(i * 3.0), end=float(i * 3.0 + 3.0), text=f"segment {i}")
            for i in range(14)
        ]

        with contextlib.ExitStack() as stack:
            for patch in _patch_ffmpeg_and_ffprobe(probe_duration_sec=60.0):
                stack.enter_context(patch)
            stack.enter_context(
                mock.patch.object(
                    transcribe_module,
                    "_whisper_transcribe",
                    return_value=fake_whisper_cues,
                )
            )

            response = self.client.post(
                "/api/ingest/search",
                json={
                    "query": "neural networks",
                    "platforms": ["yt", "ig", "tt"],
                    "max_per_platform": 2,
                },
            )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["query"], "neural networks")
        self.assertTrue(payload["material_id"].startswith("ingest-search:"))
        self.assertEqual(payload["total_resolved"], 6)
        self.assertEqual(payload["succeeded"], 6)
        self.assertEqual(payload["failed"], 0)
        self.assertEqual(len(payload["items"]), 6)

        # Per-platform counts
        self.assertEqual(payload["per_platform_resolved"], {"yt": 2, "ig": 2, "tt": 2})
        self.assertEqual(payload["per_platform_succeeded"], {"yt": 2, "ig": 2, "tt": 2})
        self.assertEqual(payload["per_platform_failed"], {"yt": 0, "ig": 0, "tt": 0})
        self.assertEqual(payload["per_platform_errors"], {})

        # Every item succeeded and has a reel
        platforms_seen = {item["platform"] for item in payload["items"]}
        self.assertEqual(platforms_seen, {"yt", "ig", "tt"})
        for item in payload["items"]:
            self.assertEqual(item["status"], "ok")
            self.assertIsNotNone(item["reel"])
            self.assertIsNotNone(item["metadata"])

        # The build_search_url helper was called once per requested platform
        called_platforms = [c[1] for c in fake_adapter.build_search_url_calls]
        self.assertEqual(set(called_platforms), {"yt", "ig", "tt"})

        # All reels landed under the same query-scoped material
        material_id = payload["material_id"]
        with db_module.get_conn() as conn:
            rows = db_module.fetch_all(
                conn,
                "SELECT id, video_id FROM reels WHERE material_id = ?",
                (material_id,),
            )
            self.assertEqual(len(rows), 6)
            video_ids = {row["video_id"] for row in rows}
            self.assertIn("yt:ytA", video_ids)
            self.assertIn("ig:igA", video_ids)
            self.assertIn("tt:ttA", video_ids)

    def test_ingest_search_instagram_failure_is_nonfatal(self) -> None:
        """If IG scraping breaks, YT and TT results still flow and IG is noted in errors."""
        self._install_high_rate_limits()
        main_module.ingestion_pipeline._openai_client = object()

        url_lists = {
            "yt": [("ytA", "https://www.youtube.com/watch?v=ytA")],
            "tt": [("ttA", "https://www.tiktok.com/@user/video/ttA")],
        }
        fake_adapter = _FakeSearchAdapter(
            url_lists_by_platform=url_lists,
            fail_platforms={"ig"},
        )
        main_module.ingestion_pipeline._adapters = [fake_adapter]

        with contextlib.ExitStack() as stack:
            for patch in _patch_ffmpeg_and_ffprobe(probe_duration_sec=60.0):
                stack.enter_context(patch)
            stack.enter_context(
                mock.patch.object(
                    transcribe_module,
                    "_whisper_transcribe",
                    return_value=[
                        IngestTranscriptCue(start=float(i * 3), end=float(i * 3 + 3), text=f"seg {i}")
                        for i in range(14)
                    ],
                )
            )

            response = self.client.post(
                "/api/ingest/search",
                json={
                    "query": "astronomy",
                    "platforms": ["yt", "ig", "tt"],
                    "max_per_platform": 1,
                },
            )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        # YT + TT succeeded, IG is listed in per_platform_errors
        self.assertEqual(payload["total_resolved"], 2)
        self.assertEqual(payload["succeeded"], 2)
        self.assertIn("ig", payload["per_platform_errors"])
        self.assertEqual(payload["per_platform_resolved"]["yt"], 1)
        self.assertEqual(payload["per_platform_resolved"]["tt"], 1)
        self.assertEqual(payload["per_platform_resolved"].get("ig", 0), 0)
        statuses = {item["platform"]: item["status"] for item in payload["items"]}
        self.assertEqual(statuses, {"yt": "ok", "tt": "ok"})

    def test_ingest_search_exclude_video_ids_skips_seen_reels(self) -> None:
        """Infinite-scroll pagination: exclude_video_ids skips already-seen reels."""
        self._install_high_rate_limits()
        main_module.ingestion_pipeline._openai_client = object()

        url_lists = {
            "yt": [
                ("seenA", "https://www.youtube.com/watch?v=seenA"),
                ("seenB", "https://www.youtube.com/watch?v=seenB"),
                ("newA", "https://www.youtube.com/watch?v=newA"),
                ("newB", "https://www.youtube.com/watch?v=newB"),
            ],
            "ig": [],
            "tt": [],
        }
        fake_adapter = _FakeSearchAdapter(url_lists_by_platform=url_lists)
        main_module.ingestion_pipeline._adapters = [fake_adapter]

        with contextlib.ExitStack() as stack:
            for patch in _patch_ffmpeg_and_ffprobe(probe_duration_sec=60.0):
                stack.enter_context(patch)
            stack.enter_context(
                mock.patch.object(
                    transcribe_module,
                    "_whisper_transcribe",
                    return_value=[
                        IngestTranscriptCue(start=float(i * 3), end=float(i * 3 + 3), text=f"seg {i}")
                        for i in range(14)
                    ],
                )
            )

            response = self.client.post(
                "/api/ingest/search",
                json={
                    "query": "physics",
                    "platforms": ["yt"],
                    "max_per_platform": 10,
                    "exclude_video_ids": ["seenA", "seenB"],
                },
            )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["total_resolved"], 2)
        self.assertEqual(payload["succeeded"], 2)
        resolved_ids = sorted(
            item["metadata"]["source_id"] for item in payload["items"] if item.get("metadata")
        )
        self.assertEqual(resolved_ids, ["newA", "newB"])

    def test_ingest_search_same_query_reuses_material_id(self) -> None:
        """Running the same search twice must land both batches under the same material_id."""
        self._install_high_rate_limits()
        main_module.ingestion_pipeline._openai_client = object()

        url_lists = {
            "yt": [("idA", "https://www.youtube.com/watch?v=idA")],
            "ig": [],
            "tt": [],
        }
        fake_adapter = _FakeSearchAdapter(url_lists_by_platform=url_lists)
        main_module.ingestion_pipeline._adapters = [fake_adapter]

        with contextlib.ExitStack() as stack:
            for patch in _patch_ffmpeg_and_ffprobe(probe_duration_sec=60.0):
                stack.enter_context(patch)
            stack.enter_context(
                mock.patch.object(
                    transcribe_module,
                    "_whisper_transcribe",
                    return_value=[
                        IngestTranscriptCue(start=float(i * 3), end=float(i * 3 + 3), text=f"seg {i}")
                        for i in range(14)
                    ],
                )
            )

            first = self.client.post(
                "/api/ingest/search",
                json={"query": "  Linear Algebra  ", "platforms": ["yt"], "max_per_platform": 1},
            )
            second = self.client.post(
                "/api/ingest/search",
                json={"query": "linear algebra", "platforms": ["yt"], "max_per_platform": 1},
            )

        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(second.status_code, 200, second.text)
        # Same normalized query → same material_id
        self.assertEqual(first.json()["material_id"], second.json()["material_id"])

    def test_build_search_url_builds_per_platform_urls(self) -> None:
        """Unit test the real adapter's search URL construction (no network)."""
        from backend.app.ingestion.adapters.yt_dlp_adapter import YtDlpAdapter

        adapter = YtDlpAdapter()
        yt_url = adapter.build_search_url("neural networks", "yt", 5)
        self.assertEqual(yt_url, "ytsearch5:neural networks")
        tt_url = adapter.build_search_url("neural networks", "tt", 5)
        self.assertIn("https://www.tiktok.com/search?q=", tt_url)
        self.assertIn("neural+networks", tt_url)
        ig_url = adapter.build_search_url("neural networks", "ig", 5)
        self.assertEqual(ig_url, "https://www.instagram.com/explore/tags/neuralnetworks/")

    def test_extract_source_id_from_url_variants(self) -> None:
        from backend.app.ingestion.adapters.yt_dlp_adapter import YtDlpAdapter

        adapter = YtDlpAdapter()
        self.assertEqual(
            adapter.extract_source_id_from_url("https://www.youtube.com/watch?v=aircAruvnKk"),
            "aircAruvnKk",
        )
        self.assertEqual(adapter.extract_source_id_from_url("https://youtu.be/abc123XYZ"), "abc123XYZ")
        self.assertEqual(
            adapter.extract_source_id_from_url("https://www.instagram.com/reel/C8xyz/"),
            "C8xyz",
        )
        self.assertEqual(
            adapter.extract_source_id_from_url("https://www.tiktok.com/@user/video/7234567890"),
            "7234567890",
        )
        self.assertIsNone(adapter.extract_source_id_from_url("https://vimeo.com/999"))


# --------------------------------------------------------------------- #
# Unit tests for normalize_clip_window — bit-compatible with reels.py:9286
# --------------------------------------------------------------------- #


class NormalizeClipWindowTests(unittest.TestCase):
    def test_basic_window(self) -> None:
        from backend.app.ingestion.segment import normalize_clip_window

        self.assertEqual(normalize_clip_window(10, 55, 120), (10, 55))

    def test_clamps_to_min_length(self) -> None:
        from backend.app.ingestion.segment import normalize_clip_window

        self.assertEqual(normalize_clip_window(10, 14, 120, min_len=15, max_len=60), (10, 25))

    def test_clamps_to_max_length(self) -> None:
        from backend.app.ingestion.segment import normalize_clip_window

        self.assertEqual(normalize_clip_window(10, 200, 300, min_len=15, max_len=60), (10, 70))

    def test_rejects_when_video_shorter_than_min(self) -> None:
        from backend.app.ingestion.segment import normalize_clip_window

        self.assertIsNone(normalize_clip_window(0, 10, 8, min_len=15, max_len=60))

    def test_end_clamped_to_video_duration(self) -> None:
        from backend.app.ingestion.segment import normalize_clip_window

        # Clip runs off the end; end is clamped and the start may be pulled back
        self.assertEqual(normalize_clip_window(20, 90, 60, min_len=15, max_len=60), (20, 60))


if __name__ == "__main__":
    unittest.main()
