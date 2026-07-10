from __future__ import annotations

import sys
from unittest import mock

import pytest

from backend.app.clip_engine.metadata import (
    canonicalize_youtube_url,
    extract_video_id,
    normalize_youtube_video_id,
    resolve_feed_urls,
    youtube_metadata,
)


VIDEO_ID = "dQw4w9WgXcQ"


@pytest.mark.parametrize(
    "url",
    [
        f"https://youtube.com/watch?v={VIDEO_ID}",
        f"http://www.youtube.com/watch?v={VIDEO_ID}",
        f"https://m.youtube.com/watch?feature=share&v={VIDEO_ID}",
        f"https://music.youtube.com/watch?v={VIDEO_ID}&list=PL123",
        f"https://youtu.be/{VIDEO_ID}?t=42",
        f"https://www.youtube.com/shorts/{VIDEO_ID}?feature=share",
        f"https://www.youtube.com/embed/{VIDEO_ID}",
        f"https://www.youtube.com/live/{VIDEO_ID}?si=abc",
    ],
)
def test_extract_video_id_accepts_supported_youtube_urls(url: str) -> None:
    assert extract_video_id(url) == VIDEO_ID


@pytest.mark.parametrize(
    "url",
    [
        f"youtube.com/watch?v={VIDEO_ID}",
        f"ftp://www.youtube.com/watch?v={VIDEO_ID}",
        f"javascript:youtube.com/watch?v={VIDEO_ID}",
        f"https://www.youtube.com.evil.test/watch?v={VIDEO_ID}",
        f"https://evil.test/youtube.com/watch?v={VIDEO_ID}",
        f"https://user@www.youtube.com/watch?v={VIDEO_ID}",
        f"https://user:password@www.youtube.com/watch?v={VIDEO_ID}",
        f"https://www.youtube.com:8443/watch?v={VIDEO_ID}",
        f"https://www.youtube.com/watch?v={VIDEO_ID}extra",
        f"https://www.youtube.com/shorts/{VIDEO_ID}extra",
        f"https://www.youtube.com/embed/{VIDEO_ID}/extra",
        f"https://youtu.be/{VIDEO_ID}extra",
        f"https://youtu.be/{VIDEO_ID}/extra",
        f"https://www.youtube.com/watch?v={VIDEO_ID}&v=aaaaaaaaaaa",
        f"https://www.youtube.com/watch?v={VIDEO_ID}\n.evil.test",
    ],
)
def test_extract_video_id_rejects_malformed_or_untrusted_urls(url: str) -> None:
    assert extract_video_id(url) is None


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/playlist?list=PL123",
        "https://youtube.com.evil.test/@channel",
        "https://example.com/youtube.com/@channel",
        "javascript:youtube.com/@channel",
        "https://user@youtube.com/@channel",
        "https://youtube.com:8443/@channel",
        "https://youtube.com/redirect?q=http://127.0.0.1/private",
    ],
)
def test_resolve_feed_urls_rejects_untrusted_urls_before_ytdlp(url: str) -> None:
    fake_yt_dlp = mock.MagicMock()

    with mock.patch.dict(sys.modules, {"yt_dlp": fake_yt_dlp}):
        assert resolve_feed_urls(url, max_items=5) == []

    fake_yt_dlp.YoutubeDL.assert_not_called()


def test_resolve_feed_urls_canonicalizes_trusted_channel_url() -> None:
    fake_ydl = mock.MagicMock()
    fake_ydl.__enter__ = mock.Mock(return_value=fake_ydl)
    fake_ydl.__exit__ = mock.Mock(return_value=False)
    fake_ydl.extract_info.return_value = {"entries": [{"id": VIDEO_ID}]}
    fake_yt_dlp = mock.MagicMock()
    fake_yt_dlp.YoutubeDL.return_value = fake_ydl

    with mock.patch.dict(sys.modules, {"yt_dlp": fake_yt_dlp}):
        result = resolve_feed_urls(
            "http://m.youtube.com/@trusted_channel/videos?redirect=https://evil.test",
            max_items=5,
        )

    assert result == [f"https://www.youtube.com/watch?v={VIDEO_ID}"]
    fake_ydl.extract_info.assert_called_once_with(
        "https://www.youtube.com/@trusted_channel/videos",
        download=False,
    )


def test_resolve_feed_urls_filters_malformed_entry_ids() -> None:
    fake_ydl = mock.MagicMock()
    fake_ydl.__enter__ = mock.Mock(return_value=fake_ydl)
    fake_ydl.__exit__ = mock.Mock(return_value=False)
    fake_ydl.extract_info.return_value = {
        "entries": [
            {"id": VIDEO_ID},
            {"id": f"{VIDEO_ID}extra"},
            {"id": "too-short"},
        ]
    }
    fake_yt_dlp = mock.MagicMock()
    fake_yt_dlp.YoutubeDL.return_value = fake_ydl

    with mock.patch.dict(sys.modules, {"yt_dlp": fake_yt_dlp}):
        result = resolve_feed_urls("https://www.youtube.com/@channel", max_items=5)

    assert result == [f"https://www.youtube.com/watch?v={VIDEO_ID}"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (VIDEO_ID, VIDEO_ID),
        (f"yt:{VIDEO_ID}", VIDEO_ID),
        (f"YouTube:{VIDEO_ID}", VIDEO_ID),
        (f"https://www.youtube.com/shorts/{VIDEO_ID}?feature=share", VIDEO_ID),
        ("yt:not-a-video-id", None),
    ],
)
def test_normalize_youtube_video_id(raw: str, expected: str | None) -> None:
    assert normalize_youtube_video_id(raw) == expected


def test_canonicalize_youtube_url_returns_typed_identity() -> None:
    assert canonicalize_youtube_url(f"https://youtu.be/{VIDEO_ID}?t=42") == {
        "kind": "video",
        "canonical_url": f"https://www.youtube.com/watch?v={VIDEO_ID}",
        "video_id": VIDEO_ID,
        "playlist_id": None,
        "channel_id": None,
    }
    assert canonicalize_youtube_url("https://youtube.com/playlist?list=PL123") == {
        "kind": "playlist",
        "canonical_url": "https://www.youtube.com/playlist?list=PL123",
        "video_id": None,
        "playlist_id": "PL123",
        "channel_id": None,
    }
    assert canonicalize_youtube_url("https://m.youtube.com/@Creator/videos?x=1") == {
        "kind": "channel",
        "canonical_url": "https://www.youtube.com/@Creator",
        "video_id": None,
        "playlist_id": None,
        "channel_id": "@Creator",
    }


def test_canonicalize_allowed_kinds_can_treat_watch_list_as_playlist() -> None:
    value = canonicalize_youtube_url(
        f"https://youtube.com/watch?v={VIDEO_ID}&list=PL123",
        allowed_kinds={"playlist"},
    )
    assert value and value["kind"] == "playlist"
    assert value["canonical_url"] == "https://www.youtube.com/playlist?list=PL123"


def test_youtube_metadata_normalizes_prefixed_id_and_maps_native_fields() -> None:
    fake_ydl = mock.MagicMock()
    fake_ydl.__enter__ = mock.Mock(return_value=fake_ydl)
    fake_ydl.__exit__ = mock.Mock(return_value=False)
    fake_ydl.extract_info.return_value = {
        "title": "Title",
        "channel": "Channel",
        "channel_url": "https://youtube.com/channel/UC123",
        "duration": 12,
        "view_count": "42",
        "upload_date": "20260710",
    }
    fake_yt_dlp = mock.MagicMock()
    fake_yt_dlp.YoutubeDL.return_value = fake_ydl
    with mock.patch.dict(sys.modules, {"yt_dlp": fake_yt_dlp}):
        result = youtube_metadata(f"yt:{VIDEO_ID}")
    fake_ydl.extract_info.assert_called_once_with(
        f"https://www.youtube.com/watch?v={VIDEO_ID}", download=False
    )
    assert result["author_name"] == "Channel"
    assert result["view_count"] == 42
    assert result["upload_date_iso"] == "2026-07-10"
