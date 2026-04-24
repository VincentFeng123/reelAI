"""Unit tests for ``backend/app/services/provider_registry.py``.

Three layers:
  * Helpers (Bilibili duration parsing, highlight stripping, protocol fixup).
  * DailymotionProvider + BilibiliProvider network paths with mocked HTTP.
  * ProviderRegistry dispatch: dormant-by-default, search_all fan-out,
    provider-name routing for fetch_transcript.

Network is never hit — ``requests.get`` is patched everywhere.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

os.environ.setdefault("VERCEL", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.services.provider_registry import (  # noqa: E402
    BilibiliProvider,
    DailymotionProvider,
    ProviderCandidate,
    ProviderRegistry,
    TikTokProvider,
    TwitchProvider,
    VimeoProvider,
    _ddg_find,
    _normalize_protocol,
    _parse_bilibili_duration,
    _parse_subtitle_cues,
    _pick_english_track,
    _strip_bilibili_highlight,
    _TWITCH_CLIP_RE,
    _VIMEO_ID_RE,
    _yt_dlp_fetch_vtt_transcript,
)
import backend.app.services.provider_registry as provider_registry_module  # noqa: E402


def _fake_response(payload, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload
    resp.raise_for_status = MagicMock()
    return resp


class BilibiliHelpersTests(unittest.TestCase):
    def test_parse_duration_mm_ss(self):
        self.assertEqual(_parse_bilibili_duration("12:34"), 12 * 60 + 34)

    def test_parse_duration_hh_mm_ss(self):
        self.assertEqual(_parse_bilibili_duration("1:02:03"), 3723)

    def test_parse_duration_handles_bare_seconds(self):
        self.assertEqual(_parse_bilibili_duration("97"), 97)

    def test_parse_duration_handles_garbage(self):
        self.assertEqual(_parse_bilibili_duration("nope"), 0)
        self.assertEqual(_parse_bilibili_duration(""), 0)

    def test_strip_highlight_markers(self):
        raw = 'A <em class="keyword">physics</em> lecture'
        self.assertEqual(_strip_bilibili_highlight(raw), "A physics lecture")

    def test_normalize_protocol_relative_url(self):
        self.assertEqual(
            _normalize_protocol("//i0.hdslb.com/x/y.jpg"),
            "https://i0.hdslb.com/x/y.jpg",
        )

    def test_normalize_protocol_passthrough(self):
        self.assertEqual(
            _normalize_protocol("https://already.ok/a.jpg"),
            "https://already.ok/a.jpg",
        )


class DailymotionProviderTests(unittest.TestCase):
    def test_search_empty_query_returns_empty(self):
        provider = DailymotionProvider()
        self.assertEqual(provider.search("", 5), [])
        self.assertEqual(provider.search("   ", 5), [])

    def test_search_parses_response(self):
        payload = {
            "list": [
                {
                    "id": "x12345",
                    "title": "Calculus 101",
                    "description": "Intro to derivatives",
                    "duration": 480,
                    "thumbnail_480_url": "https://thumb/x.jpg",
                    "owner.screenname": "MathProf",
                    "created_time": "2020-01-01T00:00:00Z",
                    "views_total": 1234,
                    "url": "https://www.dailymotion.com/video/x12345",
                }
            ]
        }
        with patch(
            "backend.app.services.provider_registry.requests.get",
            return_value=_fake_response(payload),
        ) as mock_get:
            results = DailymotionProvider().search("calculus", 5)
        self.assertEqual(len(results), 1)
        cand = results[0]
        self.assertEqual(cand.provider, "dailymotion")
        self.assertEqual(cand.video_id, "x12345")
        self.assertEqual(cand.title, "Calculus 101")
        self.assertEqual(cand.duration_sec, 480)
        self.assertEqual(cand.channel, "MathProf")
        self.assertTrue(cand.playback_url.endswith("/embed/video/x12345"))
        # Timeout must be passed through.
        kwargs = mock_get.call_args.kwargs
        self.assertIn("timeout", kwargs)
        self.assertLessEqual(kwargs["timeout"], 10.0)

    def test_search_swallows_network_errors(self):
        with patch(
            "backend.app.services.provider_registry.requests.get",
            side_effect=requests.ConnectionError("boom"),
        ):
            self.assertEqual(DailymotionProvider().search("calculus", 5), [])

    def test_search_swallows_bad_json(self):
        bad = _fake_response(payload=None)
        bad.json.side_effect = ValueError("not json")
        with patch(
            "backend.app.services.provider_registry.requests.get",
            return_value=bad,
        ):
            self.assertEqual(DailymotionProvider().search("x", 1), [])

    def test_search_ignores_items_without_id(self):
        payload = {"list": [{"title": "no id here"}, {"id": "ok1", "title": "good"}]}
        with patch(
            "backend.app.services.provider_registry.requests.get",
            return_value=_fake_response(payload),
        ):
            results = DailymotionProvider().search("q", 5)
        self.assertEqual([c.video_id for c in results], ["ok1"])

    def test_fetch_transcript_empty_id(self):
        self.assertIsNone(DailymotionProvider().fetch_transcript(""))

    def test_fetch_transcript_happy_path(self):
        srt = (
            "1\n00:00:01,000 --> 00:00:05,000\nHello\n\n"
            "2\n00:00:06,000 --> 00:00:10,000\nworld\n"
        )
        list_resp = _fake_response(
            {"list": [{"language": "en", "url": "https://static.dailymotion.com/x.srt"}]}
        )
        track_resp = MagicMock(text=srt, status_code=200)
        track_resp.raise_for_status = MagicMock()
        with patch(
            "backend.app.services.provider_registry.requests.get",
            side_effect=[list_resp, track_resp],
        ):
            out = DailymotionProvider().fetch_transcript("x12345")
        self.assertIsNotNone(out)
        self.assertEqual(out.provider, "dailymotion")
        self.assertEqual(out.video_id, "x12345")
        self.assertEqual(out.language, "en")
        self.assertEqual([c.text for c in out.cues], ["Hello", "world"])

    def test_fetch_transcript_401_returns_none(self):
        resp = MagicMock(status_code=401)
        resp.raise_for_status = MagicMock()
        with patch(
            "backend.app.services.provider_registry.requests.get",
            return_value=resp,
        ):
            self.assertIsNone(DailymotionProvider().fetch_transcript("x12345"))

    def test_fetch_transcript_404_returns_none(self):
        resp = MagicMock(status_code=404)
        resp.raise_for_status = MagicMock()
        with patch(
            "backend.app.services.provider_registry.requests.get",
            return_value=resp,
        ):
            self.assertIsNone(DailymotionProvider().fetch_transcript("x12345"))

    def test_fetch_transcript_empty_track_list(self):
        with patch(
            "backend.app.services.provider_registry.requests.get",
            return_value=_fake_response({"list": []}),
        ):
            self.assertIsNone(DailymotionProvider().fetch_transcript("x12345"))

    def test_fetch_transcript_network_error_returns_none(self):
        with patch(
            "backend.app.services.provider_registry.requests.get",
            side_effect=requests.ConnectionError("boom"),
        ):
            self.assertIsNone(DailymotionProvider().fetch_transcript("x12345"))

    def test_fetch_transcript_prefers_english_over_french(self):
        srt_fr = "1\n00:00:01,000 --> 00:00:02,000\nBonjour\n"
        srt_en = "1\n00:00:01,000 --> 00:00:02,000\nHello\n"
        list_resp = _fake_response(
            {
                "list": [
                    {"language": "fr", "url": "https://cdn/fr.srt"},
                    {"language": "en", "url": "https://cdn/en.srt"},
                ]
            }
        )
        en_resp = MagicMock(text=srt_en, status_code=200)
        en_resp.raise_for_status = MagicMock()
        fr_resp = MagicMock(text=srt_fr, status_code=200)
        fr_resp.raise_for_status = MagicMock()
        # The call order we expect: list → english (french never fetched).
        with patch(
            "backend.app.services.provider_registry.requests.get",
            side_effect=[list_resp, en_resp, fr_resp],
        ):
            out = DailymotionProvider().fetch_transcript("x12345")
        self.assertIsNotNone(out)
        self.assertEqual(out.cues[0].text, "Hello")
        self.assertEqual(out.language, "en")


class SubtitleParserTests(unittest.TestCase):
    def test_parses_srt(self):
        srt = (
            "1\n00:00:01,000 --> 00:00:05,000\nHello world\n\n"
            "2\n00:00:05,500 --> 00:00:10,000\nSecond cue\n"
        )
        cues = _parse_subtitle_cues(srt)
        self.assertEqual(len(cues), 2)
        self.assertEqual(cues[0].text, "Hello world")
        self.assertAlmostEqual(cues[0].start, 1.0, places=3)
        self.assertAlmostEqual(cues[0].end, 5.0, places=3)
        self.assertAlmostEqual(cues[1].start, 5.5, places=3)

    def test_parses_webvtt(self):
        vtt = "WEBVTT\n\n00:00:01.000 --> 00:00:05.000\nHello world\n"
        cues = _parse_subtitle_cues(vtt)
        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0].text, "Hello world")

    def test_strips_inline_tags(self):
        vtt = "WEBVTT\n\n00:00:01.000 --> 00:00:05.000\n<v Speaker>Hello</v>\n"
        cues = _parse_subtitle_cues(vtt)
        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0].text, "Hello")

    def test_returns_empty_on_no_cues(self):
        self.assertEqual(_parse_subtitle_cues(""), [])
        self.assertEqual(_parse_subtitle_cues("just plain text\n"), [])

    def test_end_padded_when_end_not_greater_than_start(self):
        bad = "00:00:05,000 --> 00:00:04,000\nRegressed cue\n"
        cues = _parse_subtitle_cues(bad)
        self.assertEqual(len(cues), 1)
        self.assertGreater(cues[0].end, cues[0].start)

    def test_joins_multi_line_cue_body(self):
        vtt = (
            "WEBVTT\n\n00:00:01.000 --> 00:00:05.000\nLine one\nLine two\n"
        )
        cues = _parse_subtitle_cues(vtt)
        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0].text, "Line one Line two")


class LanguagePickerTests(unittest.TestCase):
    def test_prefers_english(self):
        tracks = [{"language": "fr"}, {"language": "en"}]
        self.assertEqual(_pick_english_track(tracks), {"language": "en"})

    def test_prefers_english_variant(self):
        tracks = [{"language": "fr"}, {"language": "en-US"}]
        picked = _pick_english_track(tracks)
        self.assertEqual(picked["language"], "en-US")

    def test_falls_back_to_first(self):
        tracks = [{"language": "fr"}, {"language": "es"}]
        self.assertEqual(_pick_english_track(tracks), {"language": "fr"})

    def test_empty_returns_none(self):
        self.assertIsNone(_pick_english_track([]))

    def test_skips_non_dict_entries(self):
        tracks = ["garbage", {"language": "en"}]
        picked = _pick_english_track(tracks)
        self.assertEqual(picked, {"language": "en"})


class VimeoProviderTests(unittest.TestCase):
    def _patched_settings(self, token: str):
        return patch(
            "backend.app.services.provider_registry.get_settings",
            return_value=MagicMock(
                vimeo_access_token=token, provider_registry_enabled=True
            ),
        )

    def test_search_no_token_falls_back_to_keyless(self):
        # When the PAT is missing we don't refuse outright — we take the
        # DDG + oEmbed keyless path. This test mocks DDG to return an id
        # and oEmbed to hydrate it, and asserts the result is a proper
        # ProviderCandidate.
        ddg_html = (
            '<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fvimeo.com%2F123456789">ok</a>'
        )
        ddg_resp = MagicMock(text=ddg_html, status_code=200)
        ddg_resp.raise_for_status = MagicMock()
        oembed_resp = MagicMock(status_code=200)
        oembed_resp.json.return_value = {
            "title": "Keyless Title",
            "author_name": "Keyless Author",
            "duration": 120,
            "thumbnail_url": "https://i.vimeo/k.jpg",
        }
        oembed_resp.raise_for_status = MagicMock()
        with self._patched_settings(""), patch(
            "backend.app.services.provider_registry.requests.get",
            side_effect=[ddg_resp, oembed_resp],
        ):
            results = VimeoProvider().search("calculus", 3)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].provider, "vimeo")
        self.assertEqual(results[0].video_id, "123456789")
        self.assertEqual(results[0].title, "Keyless Title")
        self.assertEqual(results[0].channel, "Keyless Author")

    def test_search_no_token_keyless_empty_ddg_returns_empty(self):
        ddg_resp = MagicMock(text="<html></html>", status_code=200)
        ddg_resp.raise_for_status = MagicMock()
        with self._patched_settings(""), patch(
            "backend.app.services.provider_registry.requests.get",
            return_value=ddg_resp,
        ):
            self.assertEqual(VimeoProvider().search("quantum", 3), [])

    def test_search_empty_query_returns_empty(self):
        with self._patched_settings("tok"):
            self.assertEqual(VimeoProvider().search("", 3), [])
            self.assertEqual(VimeoProvider().search("   ", 3), [])

    def test_search_happy_path(self):
        payload = {
            "data": [
                {
                    "uri": "/videos/12345",
                    "name": "Test Video",
                    "description": "desc",
                    "duration": 300,
                    "pictures": {
                        "sizes": [
                            {"width": 200, "link": "https://i.vimeo/small.jpg"},
                            {"width": 640, "link": "https://i.vimeo/med.jpg"},
                        ]
                    },
                    "user": {"name": "Alice"},
                    "created_time": "2024-01-01T00:00:00Z",
                    "stats": {"plays": 9999},
                    "link": "https://vimeo.com/12345",
                }
            ]
        }
        with self._patched_settings("fake-token"), patch(
            "backend.app.services.provider_registry.requests.get",
            return_value=_fake_response(payload),
        ) as mock_get:
            results = VimeoProvider().search("test", 5)
        self.assertEqual(len(results), 1)
        cand = results[0]
        self.assertEqual(cand.provider, "vimeo")
        self.assertEqual(cand.video_id, "12345")
        self.assertEqual(cand.title, "Test Video")
        self.assertEqual(cand.duration_sec, 300)
        self.assertEqual(cand.channel, "Alice")
        self.assertEqual(cand.view_count, 9999)
        self.assertEqual(cand.thumbnail_url, "https://i.vimeo/med.jpg")
        self.assertTrue(cand.playback_url.endswith("/video/12345"))
        headers = mock_get.call_args.kwargs.get("headers") or {}
        self.assertIn("Authorization", headers)
        self.assertIn("bearer fake-token", headers["Authorization"])

    def test_search_network_error_returns_empty(self):
        with self._patched_settings("tok"), patch(
            "backend.app.services.provider_registry.requests.get",
            side_effect=requests.Timeout("slow"),
        ):
            self.assertEqual(VimeoProvider().search("x", 3), [])

    def test_search_skips_entries_without_uri(self):
        payload = {"data": [{"name": "No URI"}, {"uri": "/videos/7", "name": "OK"}]}
        with self._patched_settings("tok"), patch(
            "backend.app.services.provider_registry.requests.get",
            return_value=_fake_response(payload),
        ):
            results = VimeoProvider().search("q", 2)
        self.assertEqual([c.video_id for c in results], ["7"])

    def test_transcript_no_token_delegates_to_yt_dlp(self):
        sentinel = MagicMock()
        with self._patched_settings(""), patch(
            "backend.app.services.provider_registry._yt_dlp_fetch_vtt_transcript",
            return_value=sentinel,
        ) as mock_helper:
            result = VimeoProvider().fetch_transcript("12345")
        self.assertIs(result, sentinel)
        kwargs = mock_helper.call_args.kwargs
        self.assertEqual(kwargs["provider"], "vimeo")
        self.assertEqual(kwargs["video_id"], "12345")
        self.assertIn("vimeo.com/12345", kwargs["video_url"])

    def test_transcript_happy_path(self):
        list_resp = _fake_response(
            {"data": [{"language": "en", "link": "https://vimeocdn/x.vtt"}]}
        )
        vtt_text = "WEBVTT\n\n00:00:01.000 --> 00:00:05.000\nHello\n"
        vtt_resp = MagicMock(text=vtt_text, status_code=200)
        vtt_resp.raise_for_status = MagicMock()
        with self._patched_settings("tok"), patch(
            "backend.app.services.provider_registry.requests.get",
            side_effect=[list_resp, vtt_resp],
        ):
            out = VimeoProvider().fetch_transcript("12345")
        self.assertIsNotNone(out)
        self.assertEqual(out.provider, "vimeo")
        self.assertEqual(out.video_id, "12345")
        self.assertEqual(out.cues[0].text, "Hello")

    def test_transcript_403_returns_none(self):
        resp = MagicMock(status_code=403)
        resp.raise_for_status = MagicMock()
        with self._patched_settings("tok"), patch(
            "backend.app.services.provider_registry.requests.get",
            return_value=resp,
        ):
            self.assertIsNone(VimeoProvider().fetch_transcript("12345"))

    def test_transcript_empty_tracks_returns_none(self):
        with self._patched_settings("tok"), patch(
            "backend.app.services.provider_registry.requests.get",
            return_value=_fake_response({"data": []}),
        ):
            self.assertIsNone(VimeoProvider().fetch_transcript("12345"))


class BilibiliProviderTests(unittest.TestCase):
    def test_search_parses_response(self):
        payload = {
            "code": 0,
            "data": {
                "result": [
                    {
                        "bvid": "BV1xx411c7mu",
                        "title": 'Lecture on <em class="keyword">calculus</em>',
                        "description": "derivatives",
                        "author": "PhysicsTeacher",
                        "duration": "10:30",
                        "pic": "//i0.hdslb.com/thumb.jpg",
                        "play": 9001,
                    }
                ]
            },
        }
        with patch(
            "backend.app.services.provider_registry.requests.get",
            return_value=_fake_response(payload),
        ):
            results = BilibiliProvider().search("calculus", 5)
        self.assertEqual(len(results), 1)
        cand = results[0]
        self.assertEqual(cand.provider, "bilibili")
        self.assertEqual(cand.video_id, "BV1xx411c7mu")
        self.assertEqual(cand.title, "Lecture on calculus")  # highlight stripped
        self.assertEqual(cand.duration_sec, 10 * 60 + 30)
        self.assertTrue(cand.thumbnail_url.startswith("https://"))
        self.assertIn(cand.video_id, cand.playback_url)

    def test_search_rejects_error_envelope(self):
        payload = {"code": 41001, "message": "auth required"}
        with patch(
            "backend.app.services.provider_registry.requests.get",
            return_value=_fake_response(payload),
        ):
            self.assertEqual(BilibiliProvider().search("x", 3), [])

    def test_search_swallows_network_errors(self):
        with patch(
            "backend.app.services.provider_registry.requests.get",
            side_effect=requests.Timeout("slow"),
        ):
            self.assertEqual(BilibiliProvider().search("x", 3), [])


class ProviderRegistryTests(unittest.TestCase):
    def test_dormant_when_flag_off(self):
        with patch(
            "backend.app.services.provider_registry.get_settings"
        ) as mock_settings:
            mock_settings.return_value = MagicMock(provider_registry_enabled=False)
            registry = ProviderRegistry()
            self.assertFalse(registry.enabled)
            self.assertEqual(registry.search_all("calculus", 5), [])

    def test_search_all_force_enabled_ignores_global_flag(self):
        stub = ProviderCandidate(
            provider="dailymotion",
            video_id="x1",
            video_url="https://www.dailymotion.com/video/x1",
            playback_url="https://www.dailymotion.com/embed/video/x1",
            title="stub",
        )

        class _Mocked(DailymotionProvider):
            def search(self, query, max_results):
                return [stub]

        registry = ProviderRegistry(providers=[_Mocked()])
        with patch(
            "backend.app.services.provider_registry.get_settings"
        ) as mock_settings:
            mock_settings.return_value = MagicMock(provider_registry_enabled=False)
            results = registry.search_all("calculus", 3, force_enabled=True)
        self.assertEqual(results, [stub])

    def test_search_all_fans_out_when_enabled(self):
        stub = ProviderCandidate(
            provider="dailymotion",
            video_id="x1",
            video_url="https://www.dailymotion.com/video/x1",
            playback_url="https://www.dailymotion.com/embed/video/x1",
            title="stub",
        )

        class _Mocked(DailymotionProvider):
            def search(self, query, max_results):
                return [stub]

        registry = ProviderRegistry(providers=[_Mocked()])
        with patch(
            "backend.app.services.provider_registry.get_settings"
        ) as mock_settings:
            mock_settings.return_value = MagicMock(provider_registry_enabled=True)
            results = registry.search_all("calculus", 3)
        self.assertEqual(results, [stub])

    def test_search_all_continues_past_a_broken_provider(self):
        class _Broken(DailymotionProvider):
            def search(self, query, max_results):
                raise RuntimeError("provider exploded")

        stub = ProviderCandidate(
            provider="bilibili",
            video_id="BV1",
            video_url="https://www.bilibili.com/video/BV1",
            playback_url="https://player.bilibili.com/player.html?bvid=BV1",
            title="ok",
        )

        class _Good(BilibiliProvider):
            def search(self, query, max_results):
                return [stub]

        registry = ProviderRegistry(providers=[_Broken(), _Good()])
        with patch(
            "backend.app.services.provider_registry.get_settings"
        ) as mock_settings:
            mock_settings.return_value = MagicMock(provider_registry_enabled=True)
            results = registry.search_all("calculus", 3)
        self.assertEqual(results, [stub])

    def test_fetch_transcript_routes_by_name(self):
        class _HasTranscript(DailymotionProvider):
            def fetch_transcript(self, video_id):
                # signal by returning a recognizable placeholder
                from backend.app.services.provider_registry import ProviderTranscript
                return ProviderTranscript(
                    provider="dailymotion", video_id=video_id, language="en", cues=[]
                )

        registry = ProviderRegistry(providers=[_HasTranscript()])
        with patch(
            "backend.app.services.provider_registry.get_settings"
        ) as mock_settings:
            mock_settings.return_value = MagicMock(provider_registry_enabled=True)
            out = registry.fetch_transcript("dailymotion", "xyz")
            self.assertIsNotNone(out)
            self.assertEqual(out.video_id, "xyz")
            # Unknown provider → None.
            self.assertIsNone(registry.fetch_transcript("nonesuch", "xyz"))


def _mock_ydl(info: dict):
    """Build a mock yt_dlp.YoutubeDL class that behaves as a context manager
    and returns `info` from `extract_info`."""
    ydl_instance = MagicMock()
    ydl_instance.extract_info.return_value = info
    ydl_class = MagicMock()
    ydl_class.return_value.__enter__.return_value = ydl_instance
    ydl_class.return_value.__exit__.return_value = False
    return ydl_class, ydl_instance


class YtDlpCaptionsHelperTests(unittest.TestCase):
    _VTT = "WEBVTT\n\n00:00:01.000 --> 00:00:05.000\nHello\n"

    def test_happy_path_prefers_manual_subs(self):
        info = {
            "subtitles": {"en": [{"ext": "vtt", "url": "https://sub.manual/x.vtt"}]},
            "automatic_captions": {"en": [{"ext": "vtt", "url": "https://sub.auto/x.vtt"}]},
        }
        ydl_class, _ = _mock_ydl(info)
        vtt_resp = MagicMock(text=self._VTT, status_code=200)
        vtt_resp.raise_for_status = MagicMock()
        fetched: list[str] = []

        def fake_get(url, **kwargs):
            fetched.append(url)
            return vtt_resp

        with patch("yt_dlp.YoutubeDL", ydl_class), patch(
            "backend.app.services.provider_registry.requests.get",
            side_effect=fake_get,
        ):
            out = _yt_dlp_fetch_vtt_transcript(
                video_url="https://example.com/x",
                provider="tiktok",
                video_id="x",
            )
        self.assertIsNotNone(out)
        self.assertEqual(out.cues[0].text, "Hello")
        self.assertEqual(fetched, ["https://sub.manual/x.vtt"])

    def test_falls_through_to_automatic_captions(self):
        info = {
            "subtitles": {},
            "automatic_captions": {"en": [{"ext": "vtt", "url": "https://sub.auto/x.vtt"}]},
        }
        ydl_class, _ = _mock_ydl(info)
        vtt_resp = MagicMock(text=self._VTT, status_code=200)
        vtt_resp.raise_for_status = MagicMock()
        with patch("yt_dlp.YoutubeDL", ydl_class), patch(
            "backend.app.services.provider_registry.requests.get",
            return_value=vtt_resp,
        ):
            out = _yt_dlp_fetch_vtt_transcript(
                video_url="https://example.com/x",
                provider="bilibili",
                video_id="BV1",
            )
        self.assertIsNotNone(out)
        self.assertEqual(out.provider, "bilibili")

    def test_english_preference(self):
        info = {
            "subtitles": {
                "fr": [{"ext": "vtt", "url": "https://sub.fr/x.vtt"}],
                "en-US": [{"ext": "vtt", "url": "https://sub.en/x.vtt"}],
            },
        }
        ydl_class, _ = _mock_ydl(info)
        captured: list[str] = []

        def fake_get(url, **kwargs):
            captured.append(url)
            return MagicMock(
                text=self._VTT,
                status_code=200,
                raise_for_status=MagicMock(),
            )

        with patch("yt_dlp.YoutubeDL", ydl_class), patch(
            "backend.app.services.provider_registry.requests.get",
            side_effect=fake_get,
        ):
            out = _yt_dlp_fetch_vtt_transcript(
                video_url="https://example.com/x",
                provider="tiktok",
                video_id="x",
            )
        self.assertIsNotNone(out)
        self.assertEqual(out.language, "en-US")
        self.assertEqual(captured, ["https://sub.en/x.vtt"])

    def test_no_captions_returns_none(self):
        ydl_class, _ = _mock_ydl({"subtitles": {}, "automatic_captions": {}})
        with patch("yt_dlp.YoutubeDL", ydl_class):
            out = _yt_dlp_fetch_vtt_transcript(
                video_url="https://example.com/x",
                provider="tiktok",
                video_id="x",
            )
        self.assertIsNone(out)

    def test_extract_info_raising_returns_none(self):
        ydl_class = MagicMock()
        ydl_class.return_value.__enter__.return_value.extract_info.side_effect = RuntimeError("nope")
        ydl_class.return_value.__exit__.return_value = False
        with patch("yt_dlp.YoutubeDL", ydl_class):
            out = _yt_dlp_fetch_vtt_transcript(
                video_url="https://example.com/x",
                provider="tiktok",
                video_id="x",
            )
        self.assertIsNone(out)

    def test_subtitle_download_failure_returns_none(self):
        info = {"subtitles": {"en": [{"ext": "vtt", "url": "https://sub/x.vtt"}]}}
        ydl_class, _ = _mock_ydl(info)
        with patch("yt_dlp.YoutubeDL", ydl_class), patch(
            "backend.app.services.provider_registry.requests.get",
            side_effect=requests.Timeout("boom"),
        ):
            self.assertIsNone(
                _yt_dlp_fetch_vtt_transcript(
                    video_url="https://example.com/x",
                    provider="tiktok",
                    video_id="x",
                )
            )

    def test_empty_vtt_parses_to_zero_cues_returns_none(self):
        info = {"subtitles": {"en": [{"ext": "vtt", "url": "https://sub/x.vtt"}]}}
        ydl_class, _ = _mock_ydl(info)
        resp = MagicMock(text="WEBVTT\n\n", status_code=200)
        resp.raise_for_status = MagicMock()
        with patch("yt_dlp.YoutubeDL", ydl_class), patch(
            "backend.app.services.provider_registry.requests.get",
            return_value=resp,
        ):
            self.assertIsNone(
                _yt_dlp_fetch_vtt_transcript(
                    video_url="https://example.com/x",
                    provider="tiktok",
                    video_id="x",
                )
            )


class TikTokProviderTests(unittest.TestCase):
    def test_search_empty_query_returns_empty(self):
        self.assertEqual(TikTokProvider().search("", 3), [])
        self.assertEqual(TikTokProvider().search("   ", 3), [])

    def test_search_happy_path_via_ddg(self):
        # yt-dlp has no TikTok search extractor (only individual video URLs),
        # so TikTokProvider.search routes through DuckDuckGo with
        # `site:tiktok.com` and extracts 19-digit video IDs from the result
        # URLs. Each hit becomes a minimal ProviderCandidate.
        ddg_html = (
            '<a class="result__a" '
            'href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.tiktok.com%2F%40mathlady%2Fvideo%2F7123456789000000000">ok</a>'
            '<a class="result__a" '
            'href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.tiktok.com%2F%40otherguy%2Fvideo%2F7998887776000000000">two</a>'
        )
        ddg_resp = MagicMock(text=ddg_html, status_code=200)
        ddg_resp.raise_for_status = MagicMock()
        with patch(
            "backend.app.services.provider_registry.requests.get",
            return_value=ddg_resp,
        ):
            results = TikTokProvider().search("calculus", 3)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].provider, "tiktok")
        self.assertEqual(results[0].video_id, "7123456789000000000")
        self.assertTrue(results[0].playback_url.endswith("/embed/v2/7123456789000000000"))
        self.assertIn("7998887776000000000", results[1].video_url)

    def test_search_empty_ddg_returns_empty(self):
        ddg_resp = MagicMock(text="<html></html>", status_code=200)
        ddg_resp.raise_for_status = MagicMock()
        with patch(
            "backend.app.services.provider_registry.requests.get",
            return_value=ddg_resp,
        ):
            self.assertEqual(TikTokProvider().search("calculus", 3), [])

    def test_search_swallows_network_errors(self):
        with patch(
            "backend.app.services.provider_registry.requests.get",
            side_effect=requests.ConnectionError("boom"),
        ):
            self.assertEqual(TikTokProvider().search("x", 3), [])

    def test_fetch_transcript_empty_id(self):
        self.assertIsNone(TikTokProvider().fetch_transcript(""))

    def test_fetch_transcript_delegates_to_helper(self):
        expected = MagicMock()
        with patch(
            "backend.app.services.provider_registry._yt_dlp_fetch_vtt_transcript",
            return_value=expected,
        ) as mock_helper:
            result = TikTokProvider().fetch_transcript("7123456789")
        self.assertIs(result, expected)
        kwargs = mock_helper.call_args.kwargs
        self.assertEqual(kwargs["provider"], "tiktok")
        self.assertEqual(kwargs["video_id"], "7123456789")
        self.assertIn("7123456789", kwargs["video_url"])


class TwitchProviderTests(unittest.TestCase):
    def setUp(self):
        # Reset the class-level token cache between tests.
        TwitchProvider._token_access = ""
        TwitchProvider._token_expires_at = 0.0

    def _patched_settings(self, client_id: str, client_secret: str):
        return patch(
            "backend.app.services.provider_registry.get_settings",
            return_value=MagicMock(
                twitch_client_id=client_id,
                twitch_client_secret=client_secret,
                provider_registry_enabled=True,
            ),
        )

    def test_search_no_credentials_falls_back_to_keyless(self):
        # Keyless Twitch: DDG is the only plausible source. Metadata is
        # degraded (no Helix → no title / duration / views) but the clip
        # slug + embed URL are enough for the client to render.
        ddg_html = (
            '<a class="result__a" '
            'href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fclips.twitch.tv%2FAwkwardHelplessSalamanderSwiftRage">x</a>'
        )
        ddg_resp = MagicMock(text=ddg_html, status_code=200)
        ddg_resp.raise_for_status = MagicMock()
        with self._patched_settings("", ""), patch(
            "backend.app.services.provider_registry.requests.get",
            return_value=ddg_resp,
        ):
            results = TwitchProvider().search("fortnite", 3)
        self.assertEqual(len(results), 1)
        cand = results[0]
        self.assertEqual(cand.provider, "twitch")
        self.assertEqual(cand.video_id, "AwkwardHelplessSalamanderSwiftRage")
        self.assertEqual(cand.title, "")  # degraded
        self.assertEqual(cand.duration_sec, 0)
        self.assertTrue(cand.playback_url.startswith("https://clips.twitch.tv/embed?clip="))

    def test_search_no_credentials_keyless_empty_ddg(self):
        ddg_resp = MagicMock(text="<html></html>", status_code=200)
        ddg_resp.raise_for_status = MagicMock()
        with self._patched_settings("", ""), patch(
            "backend.app.services.provider_registry.requests.get",
            return_value=ddg_resp,
        ):
            self.assertEqual(TwitchProvider().search("fortnite", 3), [])

    def test_search_oauth_failure_returns_empty(self):
        with self._patched_settings("cid", "sec"), patch(
            "backend.app.services.provider_registry.requests.post",
            side_effect=requests.RequestException("boom"),
        ):
            self.assertEqual(TwitchProvider().search("valorant", 3), [])

    def test_search_no_game_match_returns_empty(self):
        oauth_resp = _fake_response({"access_token": "abc", "expires_in": 3600})
        games_resp = _fake_response({"data": []})
        with self._patched_settings("cid", "sec"), patch(
            "backend.app.services.provider_registry.requests.post",
            return_value=oauth_resp,
        ), patch(
            "backend.app.services.provider_registry.requests.get",
            return_value=games_resp,
        ):
            self.assertEqual(TwitchProvider().search("calculus", 5), [])

    def test_search_happy_path_two_hop(self):
        oauth_resp = _fake_response({"access_token": "abc", "expires_in": 3600})
        games_resp = _fake_response({"data": [{"id": "33214", "name": "Fortnite"}]})
        clips_resp = _fake_response(
            {
                "data": [
                    {
                        "id": "AwkwardHelplessSalamanderSwiftRage",
                        "url": "https://clips.twitch.tv/AwkwardHelplessSalamanderSwiftRage",
                        "embed_url": "https://clips.twitch.tv/embed?clip=AwkwardHelplessSalamanderSwiftRage",
                        "title": "Clutch play",
                        "broadcaster_name": "Ninja",
                        "duration": 30.5,
                        "thumbnail_url": "https://clip-thumb/x.jpg",
                        "created_at": "2024-05-01T10:00:00Z",
                        "view_count": 123456,
                    }
                ]
            }
        )
        get_calls: list[str] = []

        def fake_get(url, **kwargs):
            get_calls.append(url)
            if "games" in url:
                return games_resp
            return clips_resp

        with self._patched_settings("cid", "sec"), patch(
            "backend.app.services.provider_registry.requests.post",
            return_value=oauth_resp,
        ) as mock_post, patch(
            "backend.app.services.provider_registry.requests.get",
            side_effect=fake_get,
        ):
            results = TwitchProvider().search("Fortnite", 5)
        self.assertEqual(len(results), 1)
        cand = results[0]
        self.assertEqual(cand.provider, "twitch")
        self.assertEqual(cand.video_id, "AwkwardHelplessSalamanderSwiftRage")
        self.assertEqual(cand.channel, "Ninja")
        self.assertEqual(cand.duration_sec, 30)  # 30.5 floored to int
        self.assertEqual(cand.view_count, 123456)
        # Two Helix calls: games then clips.
        self.assertEqual(len(get_calls), 2)
        self.assertTrue(get_calls[0].endswith("/helix/games"))
        self.assertTrue(get_calls[1].endswith("/helix/clips"))
        # OAuth was called once with client-credentials grant.
        self.assertEqual(mock_post.call_count, 1)
        data = mock_post.call_args.kwargs.get("data") or {}
        self.assertEqual(data.get("grant_type"), "client_credentials")

    def test_token_is_reused_across_calls(self):
        # First search triggers OAuth; second search within the same cache
        # window must not re-request the token.
        oauth_resp = _fake_response({"access_token": "abc", "expires_in": 3600})
        games_resp = _fake_response({"data": []})
        with self._patched_settings("cid", "sec"), patch(
            "backend.app.services.provider_registry.requests.post",
            return_value=oauth_resp,
        ) as mock_post, patch(
            "backend.app.services.provider_registry.requests.get",
            return_value=games_resp,
        ):
            provider = TwitchProvider()
            provider.search("x", 1)
            provider.search("y", 1)
        self.assertEqual(mock_post.call_count, 1)

    def test_fetch_transcript_empty_id(self):
        self.assertIsNone(TwitchProvider().fetch_transcript(""))

    def test_fetch_transcript_delegates_to_helper(self):
        sentinel = MagicMock()
        with patch(
            "backend.app.services.provider_registry._yt_dlp_fetch_vtt_transcript",
            return_value=sentinel,
        ) as mock_helper:
            result = TwitchProvider().fetch_transcript("AwkwardHelplessSalamander")
        self.assertIs(result, sentinel)
        kwargs = mock_helper.call_args.kwargs
        self.assertEqual(kwargs["provider"], "twitch")
        self.assertTrue(kwargs["video_url"].startswith("https://clips.twitch.tv/"))


class DdgFindTests(unittest.TestCase):
    def test_extracts_ids_from_uddg_wrapped_urls(self):
        html = (
            '<a class="result__a" '
            'href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fvimeo.com%2F100000001">A</a>'
            '<a class="result__a" '
            'href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fvimeo.com%2F100000002">B</a>'
        )
        resp = MagicMock(text=html, status_code=200)
        resp.raise_for_status = MagicMock()
        with patch(
            "backend.app.services.provider_registry.requests.get",
            return_value=resp,
        ):
            ids = _ddg_find(
                "calculus",
                site_filter="vimeo.com",
                pattern=_VIMEO_ID_RE,
                max_results=5,
            )
        self.assertEqual(ids, ["100000001", "100000002"])

    def test_dedups_preserving_order(self):
        html = (
            "https://vimeo.com/111222333 some text "
            "https://vimeo.com/444555666 more text "
            "https://vimeo.com/111222333 dup"
        )
        resp = MagicMock(text=html, status_code=200)
        resp.raise_for_status = MagicMock()
        with patch(
            "backend.app.services.provider_registry.requests.get",
            return_value=resp,
        ):
            ids = _ddg_find(
                "x",
                site_filter="vimeo.com",
                pattern=_VIMEO_ID_RE,
                max_results=10,
            )
        self.assertEqual(ids, ["111222333", "444555666"])

    def test_respects_max_results(self):
        urls = " ".join(f"https://vimeo.com/{1_000_000 + i}" for i in range(20))
        resp = MagicMock(text=urls, status_code=200)
        resp.raise_for_status = MagicMock()
        with patch(
            "backend.app.services.provider_registry.requests.get",
            return_value=resp,
        ):
            ids = _ddg_find(
                "x",
                site_filter="vimeo.com",
                pattern=_VIMEO_ID_RE,
                max_results=3,
            )
        self.assertEqual(len(ids), 3)

    def test_twitch_clip_pattern_matches_slug(self):
        html = 'https://clips.twitch.tv/AwkwardHelplessSalamanderSwiftRage stuff'
        resp = MagicMock(text=html, status_code=200)
        resp.raise_for_status = MagicMock()
        with patch(
            "backend.app.services.provider_registry.requests.get",
            return_value=resp,
        ):
            slugs = _ddg_find(
                "fortnite",
                site_filter="clips.twitch.tv",
                pattern=_TWITCH_CLIP_RE,
                max_results=5,
            )
        self.assertEqual(slugs, ["AwkwardHelplessSalamanderSwiftRage"])

    def test_network_error_returns_empty(self):
        with patch(
            "backend.app.services.provider_registry.requests.get",
            side_effect=requests.ConnectionError("boom"),
        ):
            self.assertEqual(
                _ddg_find(
                    "x",
                    site_filter="vimeo.com",
                    pattern=_VIMEO_ID_RE,
                    max_results=5,
                ),
                [],
            )

    def test_empty_query_returns_empty(self):
        self.assertEqual(
            _ddg_find(
                "",
                site_filter="vimeo.com",
                pattern=_VIMEO_ID_RE,
                max_results=5,
            ),
            [],
        )


class BilibiliTranscriptTests(unittest.TestCase):
    def test_fetch_transcript_empty_id(self):
        self.assertIsNone(BilibiliProvider().fetch_transcript(""))

    def test_fetch_transcript_delegates_to_helper(self):
        sentinel = MagicMock()
        with patch(
            "backend.app.services.provider_registry._yt_dlp_fetch_vtt_transcript",
            return_value=sentinel,
        ) as mock_helper:
            result = BilibiliProvider().fetch_transcript("BV1xx411c7mu")
        self.assertIs(result, sentinel)
        kwargs = mock_helper.call_args.kwargs
        self.assertEqual(kwargs["provider"], "bilibili")
        self.assertEqual(kwargs["video_id"], "BV1xx411c7mu")
        self.assertIn("BV1xx411c7mu", kwargs["video_url"])


if __name__ == "__main__":
    unittest.main()
