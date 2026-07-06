# backend/app/clip_engine/metadata.py
"""YouTube id extraction + lightweight (no-download) metadata fetch."""
from __future__ import annotations

import re

_YT_ID = re.compile(
    r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/|youtube\.com/embed/|"
    r"youtube\.com/live/|m\.youtube\.com/watch\?v=)([A-Za-z0-9_-]{11})",
    re.IGNORECASE,
)


def extract_video_id(url: str) -> str | None:
    if not url:
        return None
    m = _YT_ID.search(url)
    return m.group(1) if m else None


def youtube_metadata(video_id: str) -> dict:
    """Title/author/duration/thumbnail via yt-dlp (metadata only, no download).
    Lazy import; returns {} on failure (callers fall back to transcript/embed data).
    """
    try:
        import yt_dlp  # lazy — not on the hot path when search already provides metadata
        opts = {"quiet": True, "skip_download": True, "extract_flat": False, "noplaylist": True}
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
        return {
            "title": info.get("title") or "",
            "author_name": info.get("uploader") or info.get("channel") or "",
            "author_url": info.get("uploader_url") or info.get("channel_url") or "",
            "duration_sec": float(info.get("duration")) if info.get("duration") else None,
            "thumbnail_url": info.get("thumbnail") or "",
            "view_count": info.get("view_count"),
            "upload_date_iso": info.get("upload_date") or None,
            "description": info.get("description") or "",
        }
    except Exception:
        return {}
