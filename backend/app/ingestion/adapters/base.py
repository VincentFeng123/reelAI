"""
BaseAdapter ABC and the AdapterResult dataclass.

Every platform adapter returns a uniform `AdapterResult` so the pipeline doesn't care
whether the source was YouTube, Instagram, or TikTok. Today we have exactly one adapter —
`YtDlpAdapter` — because yt-dlp natively handles all three. Future adapters (official
YouTube Data API for Creative-Commons-only flow, direct Instagram Graph API, etc.) can
slot into the same interface without touching the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

PlatformCode = Literal["yt", "ig", "tt"]


@dataclass
class AdapterResult:
    """
    The canonical output of any adapter's `resolve()` call.

    `video_path` is the on-disk location of the downloaded media inside the current
    TempWorkspace. `info_dict` is the raw platform metadata (structure differs per platform;
    the `metadata.py` mapper normalizes it). `playback_url` is the URL the iOS client will
    actually render in its WKWebView — MUST be an allow-listed host.
    """

    platform: PlatformCode
    source_id: str
    source_url: str
    playback_url: str
    video_path: Path
    info_dict: dict[str, Any] = field(default_factory=dict)
    audio_path: Path | None = None


class BaseAdapter(ABC):
    """Interface every platform adapter must implement."""

    # Human-readable identifier used in logs. Subclasses override.
    name: str = "base"

    @abstractmethod
    def supports(self, url: str) -> bool:
        """Return True iff this adapter can handle the given URL."""

    @abstractmethod
    def platform_for(self, url: str) -> PlatformCode:
        """Return the platform code ('yt' / 'ig' / 'tt') for a supported URL."""

    @abstractmethod
    def resolve(self, url: str, workspace: Path) -> AdapterResult:
        """
        Download the media referenced by `url` into `workspace` and return an AdapterResult.

        Adapters MUST raise `ingestion.errors.DownloadError` or `UnsupportedSourceError`
        for failures — never a bare `Exception`.
        """

    @abstractmethod
    def resolve_feed(self, url: str, *, max_items: int) -> list[str]:
        """
        Resolve a feed-like URL (profile / hashtag / playlist) to a list of individual reel
        URLs, up to `max_items`. Does NOT download the media; the pipeline calls `resolve()`
        on each returned URL afterward.
        """


__all__ = ["BaseAdapter", "AdapterResult", "PlatformCode"]
