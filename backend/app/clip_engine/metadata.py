# backend/app/clip_engine/metadata.py
"""YouTube id extraction + lightweight (no-download) metadata fetch."""
from __future__ import annotations

import re
from collections.abc import Collection
from typing import Any
from urllib.parse import parse_qs, urlsplit

_YT_ID = re.compile(r"[A-Za-z0-9_-]{11}\Z")
_YT_PATH_ID_KINDS = frozenset({"embed", "live", "shorts"})
_YT_FEED_PREFIXES = frozenset({"c", "channel", "user"})
_YT_FEED_TABS = frozenset(
    {"community", "featured", "playlists", "podcasts", "shorts", "streams", "videos"}
)
_YT_FEED_SEGMENT = re.compile(r"[A-Za-z0-9_.@-]+\Z")
_YT_VIDEO_PREFIX = re.compile(r"(?:yt|youtube):", re.IGNORECASE)


def _trusted_youtube_url(url: str):
    """Return a parsed, credential-free HTTP(S) YouTube URL or ``None``."""
    if not isinstance(url, str) or not url or any(
        char.isspace() or ord(char) < 32 or ord(char) == 127 for char in url
    ):
        return None
    try:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").lower()
        port = parsed.port
    except (TypeError, ValueError):
        return None
    if parsed.scheme.lower() not in {"http", "https"}:
        return None
    if parsed.username is not None or parsed.password is not None or port is not None:
        return None
    if host != "youtu.be" and host != "youtube.com" and not host.endswith(".youtube.com"):
        return None
    # Reject malformed netloc variants such as trailing colons even when
    # ``urlsplit().port`` happens to return ``None``.
    if parsed.netloc.lower() != host:
        return None
    return parsed


def _single_query_value(query: str, name: str) -> str | None:
    values = parse_qs(query, keep_blank_values=True).get(name, [])
    if len(values) != 1:
        return None
    return values[0]


def extract_video_id(url: str) -> str | None:
    parsed = _trusted_youtube_url(url)
    if parsed is None:
        return None
    host = (parsed.hostname or "").lower()
    parts = [part for part in parsed.path.split("/") if part]

    if host == "youtu.be":
        candidate = parts[0] if len(parts) == 1 else None
    elif parts == ["watch"]:
        candidate = _single_query_value(parsed.query, "v")
    elif len(parts) == 2 and parts[0].lower() in _YT_PATH_ID_KINDS:
        candidate = parts[1]
    else:
        candidate = None

    return candidate if candidate and _YT_ID.fullmatch(candidate) else None


def normalize_youtube_video_id(raw: object) -> str | None:
    """Return a bare canonical YouTube video id from a raw id, prefix, or URL."""
    if not isinstance(raw, str):
        return None
    value = raw.strip()
    if not value:
        return None
    value = _YT_VIDEO_PREFIX.sub("", value, count=1)
    if _YT_ID.fullmatch(value):
        return value
    return extract_video_id(value)


def canonicalize_youtube_url(
    url: str,
    allowed_kinds: Collection[str] | None = None,
) -> dict[str, str | None] | None:
    """Canonicalize a trusted video, playlist, or channel URL.

    The returned identifiers are bare provider ids. Channel handles remain
    handle-shaped (``@creator``); legacy custom/user routes retain their route
    prefix so unlike identities cannot collide.
    """
    allowed = {str(kind).strip().lower() for kind in allowed_kinds or ()}

    video_id = extract_video_id(url)
    if video_id:
        if not allowed or "video" in allowed:
            return {
                "kind": "video",
                "canonical_url": f"https://www.youtube.com/watch?v={video_id}",
                "video_id": video_id,
                "playlist_id": None,
                "channel_id": None,
            }

    parsed = _trusted_youtube_url(url)
    if parsed is None:
        return None
    host = (parsed.hostname or "").lower()
    parts = [part for part in parsed.path.split("/") if part]
    playlist_id = _single_query_value(parsed.query, "list")
    playlist_path = (
        parts == ["playlist"]
        or (parts == ["watch"] and host != "youtu.be")
        or (host == "youtu.be" and len(parts) == 1 and _YT_ID.fullmatch(parts[0]))
    )
    if playlist_path and playlist_id and re.fullmatch(r"[A-Za-z0-9_-]+", playlist_id):
        if allowed and "playlist" not in allowed:
            return None
        return {
            "kind": "playlist",
            "canonical_url": f"https://www.youtube.com/playlist?list={playlist_id}",
            "video_id": None,
            "playlist_id": playlist_id,
            "channel_id": None,
        }

    if host == "youtu.be" or not parts:
        return None
    first = parts[0]
    channel_id: str | None = None
    base_parts: list[str] = []
    tail: list[str] = []
    if first.startswith("@") and len(first) > 1 and _YT_FEED_SEGMENT.fullmatch(first):
        channel_id = first
        base_parts = [first]
        tail = parts[1:]
    elif first.lower() in _YT_FEED_PREFIXES and len(parts) >= 2:
        second = parts[1]
        if not _YT_FEED_SEGMENT.fullmatch(second):
            return None
        prefix = first.lower()
        channel_id = second if prefix == "channel" else f"{prefix}/{second}"
        base_parts = [prefix, second]
        tail = parts[2:]
    if channel_id is None or len(tail) > 1 or (tail and tail[0].lower() not in _YT_FEED_TABS):
        return None
    if allowed and "channel" not in allowed:
        return None
    return {
        "kind": "channel",
        "canonical_url": "https://www.youtube.com/" + "/".join(base_parts),
        "video_id": None,
        "playlist_id": None,
        "channel_id": channel_id,
    }


def _canonical_feed_url(feed_url: str) -> str | None:
    """Allowlist channel/playlist paths and discard untrusted query fields."""
    parsed = _trusted_youtube_url(feed_url)
    if parsed is None:
        return None
    host = (parsed.hostname or "").lower()
    parts = [part for part in parsed.path.split("/") if part]

    playlist_id = _single_query_value(parsed.query, "list")
    if playlist_id is not None:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", playlist_id):
            return None
        if parts == ["playlist"]:
            return f"https://www.youtube.com/playlist?list={playlist_id}"
        if parts == ["watch"] and host != "youtu.be":
            video_id = _single_query_value(parsed.query, "v")
            if video_id is not None and not _YT_ID.fullmatch(video_id):
                return None
            return f"https://www.youtube.com/playlist?list={playlist_id}"
        if host == "youtu.be" and len(parts) == 1 and _YT_ID.fullmatch(parts[0]):
            return f"https://www.youtube.com/playlist?list={playlist_id}"
        return None

    if host == "youtu.be" or not parts:
        return None

    first = parts[0]
    if first.startswith("@"):
        if len(first) == 1 or not _YT_FEED_SEGMENT.fullmatch(first):
            return None
        tail = parts[1:]
    elif first.lower() in _YT_FEED_PREFIXES:
        if len(parts) < 2 or not _YT_FEED_SEGMENT.fullmatch(parts[1]):
            return None
        tail = parts[2:]
    else:
        return None

    if len(tail) > 1 or (tail and tail[0].lower() not in _YT_FEED_TABS):
        return None
    return "https://www.youtube.com/" + "/".join(parts)


def youtube_metadata(video_id: str) -> dict:
    """Title/author/duration/thumbnail via yt-dlp (metadata only, no download).
    Lazy import; returns {} on failure (callers fall back to transcript/embed data).
    """
    normalized_id = normalize_youtube_video_id(video_id)
    if normalized_id is None:
        return {}
    try:
        import yt_dlp  # lazy — not on the hot path when search already provides metadata
        opts = {"quiet": True, "skip_download": True, "extract_flat": False, "noplaylist": True}
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(
                f"https://www.youtube.com/watch?v={normalized_id}", download=False
            )
        raw_views: Any = info.get("view_count")
        try:
            view_count = int(raw_views) if raw_views not in (None, "") else None
        except (TypeError, ValueError, OverflowError):
            view_count = None
        raw_date = str(info.get("upload_date") or "").strip()
        upload_date = (
            f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}"
            if re.fullmatch(r"\d{8}", raw_date)
            else (raw_date or None)
        )
        return {
            "title": info.get("title") or "",
            "author_name": info.get("uploader") or info.get("channel") or "",
            "author_url": info.get("uploader_url") or info.get("channel_url") or "",
            "duration_sec": float(info.get("duration")) if info.get("duration") else None,
            "thumbnail_url": info.get("thumbnail") or "",
            "view_count": view_count,
            "upload_date_iso": upload_date,
            "description": info.get("description") or "",
        }
    except Exception:
        return {}


def resolve_feed_urls(feed_url: str, max_items: int) -> list[str]:
    """Resolve a channel/playlist URL to individual YouTube watch URLs (no download)."""
    trusted_feed_url = _canonical_feed_url(feed_url)
    if trusted_feed_url is None or max_items <= 0:
        return []
    try:
        import yt_dlp  # lazy
        opts = {"quiet": True, "skip_download": True, "extract_flat": True, "playlistend": max_items}
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(trusted_feed_url, download=False)
        entries = (info or {}).get("entries") or []
        urls = []
        for e in entries[:max_items]:
            vid = (e or {}).get("id")
            if isinstance(vid, str) and _YT_ID.fullmatch(vid):
                urls.append(f"https://www.youtube.com/watch?v={vid}")
        return urls
    except Exception:
        return []
