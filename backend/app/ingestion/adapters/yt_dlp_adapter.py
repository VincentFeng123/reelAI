"""
Unified yt-dlp adapter for YouTube, Instagram, and TikTok.

yt-dlp (https://github.com/yt-dlp/yt-dlp) handles all three platforms through a single
extractor API. Using one adapter for all three keeps the code base small, lets us share
cookies/session handling in one place, and means a new yt-dlp release usually lifts all
three platforms at once when they change their sites.

This adapter is the ONLY place in the codebase that imports `yt_dlp`. If you need to stub
it out in tests, `unittest.mock.patch.object(yt_dlp_adapter, "yt_dlp", ...)` does the trick.

Safety features:
  * robots.txt check cached 24h per host (stdlib urllib.robotparser).
  * User-Agent "ReelAIBot/1.0 (+https://reelai.app/bot)" so platforms can identify us.
  * Explicit login-walled content detection → `UnsupportedSourceError(403)`.
  * Playback URL is always allow-listed host (youtube.com / instagram.com / tiktok.com).
  * Format selection capped at 720p to avoid downloading 4K bytes we're just going to delete.
"""

from __future__ import annotations

import logging
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from pathlib import Path
from typing import Any

from ... import config as _config
from .. import USER_AGENT
from ..errors import DownloadError, UnsupportedSourceError
from ..logging_config import get_ingest_logger, log_event
from .base import AdapterResult, BaseAdapter, PlatformCode

logger: logging.Logger = get_ingest_logger(__name__)

# Deferred import — yt_dlp is a heavy (~20MB) dependency; only load it on first use so
# `from backend.app.ingestion.adapters import yt_dlp_adapter` doesn't pay the cost at
# module import time. Tests can patch this module-level symbol to stub the whole library.
try:
    import yt_dlp  # type: ignore

    _YT_DLP_IMPORT_ERROR: Exception | None = None
except Exception as _exc:  # pragma: no cover - only hit if yt_dlp is missing at deploy time
    yt_dlp = None  # type: ignore
    _YT_DLP_IMPORT_ERROR = _exc


_YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "music.youtube.com"}
_INSTAGRAM_HOSTS = {"instagram.com", "www.instagram.com"}
_TIKTOK_HOSTS = {"tiktok.com", "www.tiktok.com", "vm.tiktok.com", "m.tiktok.com"}


def _host(url: str) -> str:
    try:
        parsed = urllib.parse.urlparse(url)
        return (parsed.hostname or "").lower()
    except Exception:
        return ""


def _normalize_host(host: str) -> str:
    if host.startswith("www."):
        return host[4:]
    return host


class _RobotsCache:
    """In-memory per-host robots.txt cache with a 24h TTL."""

    def __init__(self, ttl_sec: float = 24 * 3600) -> None:
        self._ttl_sec = ttl_sec
        self._entries: dict[str, tuple[float, urllib.robotparser.RobotFileParser | None]] = {}
        self._lock = threading.Lock()

    def _fetch(self, host: str) -> urllib.robotparser.RobotFileParser | None:
        parser = urllib.robotparser.RobotFileParser()
        robots_url = f"https://{host}/robots.txt"
        parser.set_url(robots_url)
        try:
            req = urllib.request.Request(robots_url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status != 200:
                    return None
                body = resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError, Exception):
            # If robots.txt is unreachable, default to "allowed" — this matches Google's
            # documented behavior. We'd rather miss a scraping restriction than crash.
            return None
        parser.parse(body.splitlines())
        return parser

    def allows(self, url: str) -> bool:
        host = _host(url)
        if not host:
            return True
        now = time.time()
        with self._lock:
            cached = self._entries.get(host)
            if cached and now - cached[0] < self._ttl_sec:
                parser = cached[1]
            else:
                parser = self._fetch(host)
                self._entries[host] = (now, parser)
        if parser is None:
            return True
        try:
            return parser.can_fetch(USER_AGENT, url)
        except Exception:
            return True


_robots_cache = _RobotsCache()


# yt-dlp login-wall / privacy error patterns. These are matched against the `str(DownloadError)`
# message because yt-dlp's extractor errors carry human-readable prose.
_LOGIN_WALL_PATTERNS = (
    re.compile(r"login[_\s]required", re.IGNORECASE),
    re.compile(r"this\s+video\s+is\s+private", re.IGNORECASE),
    re.compile(r"sign[\s-]?in\s+to\s+confirm", re.IGNORECASE),
    re.compile(r"members?[-\s]only", re.IGNORECASE),
    re.compile(r"account\s+.*\s+private", re.IGNORECASE),
    re.compile(r"requested\s+content\s+.*\s+unavailable", re.IGNORECASE),
    re.compile(r"age[-\s]?restricted", re.IGNORECASE),
)


def _is_login_walled(msg: str) -> bool:
    return any(p.search(msg) for p in _LOGIN_WALL_PATTERNS)


class YtDlpAdapter(BaseAdapter):
    name = "yt-dlp"

    def __init__(self, *, max_height: int = 720, socket_timeout: int = 30, retries: int = 3) -> None:
        self._max_height = max_height
        self._socket_timeout = socket_timeout
        self._retries = retries

    # --------------------------------------------------------------------- #
    # URL classification
    # --------------------------------------------------------------------- #

    def supports(self, url: str) -> bool:
        # yt-dlp's native search pseudo-URLs for topic search
        if url.startswith("ytsearch") or url.startswith("tiktoksearch"):
            return True
        host = _host(url)
        if not host:
            return False
        return host in _YOUTUBE_HOSTS or host in _INSTAGRAM_HOSTS or host in _TIKTOK_HOSTS

    def platform_for(self, url: str) -> PlatformCode:
        if url.startswith("ytsearch"):
            return "yt"
        if url.startswith("tiktoksearch"):
            return "tt"
        host = _host(url)
        if host in _YOUTUBE_HOSTS:
            return "yt"
        if host in _INSTAGRAM_HOSTS:
            return "ig"
        if host in _TIKTOK_HOSTS:
            return "tt"
        raise UnsupportedSourceError(f"Unsupported host: {host!r}")

    # --------------------------------------------------------------------- #
    # Topic search — build per-platform search URLs
    # --------------------------------------------------------------------- #

    def build_search_url(self, query: str, platform: PlatformCode, max_items: int) -> str:
        """
        Construct a URL (or yt-dlp pseudo-URL) that `resolve_feed` can hand to yt-dlp.

        - YouTube: `ytsearch{N}:{query}` — yt-dlp's native search extractor, very reliable
        - TikTok: `https://www.tiktok.com/search?q={query}` — yt-dlp's TikTok search extractor
        - Instagram: `https://www.instagram.com/explore/tags/{hashtag}/` — yt-dlp hashtag extractor,
          the flakiest of the three. Instagram anti-bot hits this path hardest; callers should
          treat empty/failing IG results as best-effort, not a hard error.
        """
        cleaned = (query or "").strip()
        if not cleaned:
            raise UnsupportedSourceError("empty search query")
        capped = max(1, min(int(max_items), 50))

        if platform == "yt":
            return f"ytsearch{capped}:{cleaned}"

        if platform == "tt":
            encoded = urllib.parse.quote_plus(cleaned)
            return f"https://www.tiktok.com/search?q={encoded}"

        if platform == "ig":
            tag = re.sub(r"[^a-z0-9_]", "", cleaned.lower())
            if not tag:
                raise UnsupportedSourceError(
                    "query cannot be converted to an Instagram hashtag",
                    detail=cleaned,
                )
            return f"https://www.instagram.com/explore/tags/{tag}/"

        raise UnsupportedSourceError(f"Unsupported platform: {platform!r}")

    def extract_source_id_from_url(self, url: str, platform: PlatformCode | None = None) -> str | None:
        """
        Extract the bare source_id from a reel URL without downloading anything.
        Used by the search path to dedup against `exclude_video_ids` before ingesting.
        Returns None if the URL doesn't match a known pattern (caller treats as "not seen").
        """
        if not url:
            return None
        if platform is None:
            try:
                platform = self.platform_for(url)
            except UnsupportedSourceError:
                return None

        if platform == "yt":
            match = (
                re.search(r"[?&]v=([A-Za-z0-9_\-]{6,})", url)
                or re.search(r"youtu\.be/([A-Za-z0-9_\-]{6,})", url)
                or re.search(r"youtube\.com/(?:shorts|embed)/([A-Za-z0-9_\-]{6,})", url)
            )
            return match.group(1) if match else None

        if platform == "ig":
            match = re.search(r"/reel/([^/?#]+)/", url) or re.search(r"/p/([^/?#]+)/", url)
            return match.group(1) if match else None

        if platform == "tt":
            match = re.search(r"/video/(\d+)", url)
            return match.group(1) if match else None

        return None

    # --------------------------------------------------------------------- #
    # Single-URL resolve
    # --------------------------------------------------------------------- #

    def resolve(self, url: str, workspace: Path) -> AdapterResult:
        self._require_yt_dlp()
        if not self.supports(url):
            raise UnsupportedSourceError(f"URL not supported by {self.name}: {url}")

        if not _robots_cache.allows(url):
            raise UnsupportedSourceError("robots.txt disallows this URL for ReelAIBot.")

        platform = self.platform_for(url)

        ydl_opts = self._ydl_opts_for_resolve(workspace)
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:  # type: ignore[attr-defined]
                info = ydl.extract_info(url, download=True)
        except yt_dlp.utils.DownloadError as exc:  # type: ignore[attr-defined]
            message = str(exc)
            if _is_login_walled(message):
                raise UnsupportedSourceError(
                    "login-walled content is not supported",
                    detail=message,
                ) from exc
            raise DownloadError("yt-dlp download failed", detail=message) from exc
        except Exception as exc:  # pragma: no cover - unexpected yt_dlp internal error
            raise DownloadError("yt-dlp raised an unexpected error", detail=str(exc)) from exc

        if info is None:
            raise DownloadError("yt-dlp returned no info dict for URL", detail=url)

        # Some extractors wrap the actual info in `entries[0]` (e.g. when a single URL points
        # at a playlist of length 1). Flatten that.
        if "entries" in info and isinstance(info.get("entries"), list) and info["entries"]:
            info = info["entries"][0]

        video_path = self._resolve_downloaded_path(info, workspace)
        source_id = self._source_id_from_info(info, platform, url)
        playback_url = self._playback_url_for(info, platform, url, source_id)

        log_event(
            logger,
            logging.INFO,
            "adapter_resolve_ok",
            platform=platform,
            source_id=source_id,
            source_url=url,
            video_path=str(video_path),
            duration_sec=info.get("duration"),
        )

        return AdapterResult(
            platform=platform,
            source_id=source_id,
            source_url=url,
            playback_url=playback_url,
            video_path=video_path,
            info_dict=info,
        )

    # --------------------------------------------------------------------- #
    # Feed / profile / hashtag resolve
    # --------------------------------------------------------------------- #

    def resolve_feed(self, url: str, *, max_items: int) -> list[str]:
        self._require_yt_dlp()
        if not self.supports(url):
            raise UnsupportedSourceError(f"Feed URL not supported by {self.name}: {url}")

        # yt-dlp pseudo-URLs (ytsearch:, tiktoksearch:) have no host → skip robots.txt
        is_pseudo_search = url.startswith("ytsearch") or url.startswith("tiktoksearch")
        if not is_pseudo_search and not _robots_cache.allows(url):
            raise UnsupportedSourceError("robots.txt disallows this feed URL for ReelAIBot.")

        if max_items <= 0:
            return []

        ydl_opts = self._ydl_opts_for_feed(max_items)
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:  # type: ignore[attr-defined]
                info = ydl.extract_info(url, download=False)
        except yt_dlp.utils.DownloadError as exc:  # type: ignore[attr-defined]
            message = str(exc)
            if _is_login_walled(message):
                raise UnsupportedSourceError("login-walled feed is not supported", detail=message) from exc
            raise DownloadError("yt-dlp feed resolve failed", detail=message) from exc
        except Exception as exc:  # pragma: no cover
            raise DownloadError("yt-dlp raised an unexpected error on feed", detail=str(exc)) from exc

        if info is None:
            return []

        entries = info.get("entries") if isinstance(info, dict) else None
        if not entries:
            return []

        urls: list[str] = []
        for entry in entries:
            if entry is None:
                continue
            candidate = self._entry_to_url(entry)
            if candidate:
                urls.append(candidate)
            if len(urls) >= max_items:
                break

        log_event(
            logger,
            logging.INFO,
            "adapter_resolve_feed_ok",
            feed_url=url,
            count=len(urls),
            max_items=max_items,
        )
        return urls

    # --------------------------------------------------------------------- #
    # Internals
    # --------------------------------------------------------------------- #

    def _require_yt_dlp(self) -> None:
        if yt_dlp is None:
            raise DownloadError(
                "yt-dlp is not installed in this environment",
                detail=str(_YT_DLP_IMPORT_ERROR),
            )

    # YouTube data-center IP workaround.
    #
    # Since 2024-2025 YouTube has been aggressively blocking the default
    # extractor when called from cloud IP ranges (AWS/GCP/Railway/etc),
    # failing with "Video unavailable. This content isn't available."
    #
    # yt-dlp's upstream recommendation as of 2026 is to add `web_embedded`
    # and `mweb` to the player_client list. The `default` value keeps
    # yt-dlp's current stable default (which changes release to release as
    # YouTube patches holes), and the extra clients are tried as fallbacks
    # when the default rejects the request. See:
    #   https://github.com/yt-dlp/yt-dlp/pull/14693  (temp workaround)
    #   https://github.com/yt-dlp/yt-dlp/issues/16150 (client status)
    #
    # This is the cheapest fix for the cloud-IP block; it doesn't work for
    # every video (age-gated + login-required still fail) but it unblocks
    # the common case without needing a proxy or cookies. yt-dlp must be
    # reasonably recent for these client names to exist — see
    # backend/requirements.txt for the pinned version.
    _YOUTUBE_EXTRACTOR_ARGS_BASE: dict[str, dict[str, list[str]]] = {
        "youtube": {
            "player_client": ["default", "web_embedded", "mweb"],
        },
    }

    def _youtube_extractor_args(self) -> dict[str, dict[str, list[str]]]:
        # Copy the base and layer in the bgutil PO token provider when
        # YTDLP_POT_PROVIDER_URL is set. On flagged IPs (cloud datacenters),
        # YouTube increasingly requires PO Tokens for GVS/caption requests;
        # the bgutil plugin fetches them from a sidecar HTTP provider. When
        # the provider URL is empty, yt-dlp behaves exactly as before (plugin
        # is installed but not configured, so it's a no-op).
        args: dict[str, dict[str, list[str]]] = {
            k: {kk: list(vv) for kk, vv in v.items()}
            for k, v in self._YOUTUBE_EXTRACTOR_ARGS_BASE.items()
        }
        provider_url = _config.get_settings().ytdlp_pot_provider_url.strip()
        if provider_url:
            args["youtubepot-bgutilhttp"] = {"base_url": [provider_url]}
        return args

    def _ydl_opts_for_resolve(self, workspace: Path) -> dict[str, Any]:
        return {
            "format": f"bestvideo[height<={self._max_height}]+bestaudio/best[height<={self._max_height}]",
            "merge_output_format": "mp4",
            "outtmpl": str(workspace / "source.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "nocheckcertificate": False,
            "socket_timeout": self._socket_timeout,
            "retries": self._retries,
            "fragment_retries": self._retries,
            "restrictfilenames": True,
            "writesubtitles": False,
            "writeautomaticsub": True,
            "subtitleslangs": ["en", "en-US", "en-GB"],
            "subtitlesformat": "vtt",
            "extractor_args": self._youtube_extractor_args(),
            "http_headers": {
                "User-Agent": USER_AGENT,
            },
            # Don't let yt-dlp call its own progress hooks (we don't use them and they
            # print to stdout under some logging configs).
            "progress_hooks": [],
        }

    def _ydl_opts_for_feed(self, max_items: int) -> dict[str, Any]:
        return {
            "extract_flat": "in_playlist",
            "playlist_items": f"1-{max_items}",
            "quiet": True,
            "no_warnings": True,
            "nocheckcertificate": False,
            "socket_timeout": self._socket_timeout,
            "retries": self._retries,
            "extractor_args": self._youtube_extractor_args(),
            "http_headers": {
                "User-Agent": USER_AGENT,
            },
            "progress_hooks": [],
        }

    def _resolve_downloaded_path(self, info: dict[str, Any], workspace: Path) -> Path:
        # yt-dlp records the final filename under `requested_downloads[0].filepath` or
        # falls back to `_filename` / `filepath` depending on version.
        requested = info.get("requested_downloads")
        if isinstance(requested, list) and requested:
            first = requested[0] or {}
            for key in ("filepath", "_filename"):
                candidate = first.get(key)
                if candidate:
                    path = Path(candidate)
                    if path.exists():
                        return path

        for key in ("filepath", "_filename"):
            candidate = info.get(key)
            if candidate:
                path = Path(candidate)
                if path.exists():
                    return path

        # Last-ditch: scan the workspace for a file whose stem is `source` or any media file.
        candidates = sorted(workspace.glob("source.*")) or sorted(workspace.iterdir())
        for entry in candidates:
            if entry.is_file() and entry.stat().st_size > 0:
                return entry

        raise DownloadError("yt-dlp finished but no downloaded file was found in workspace")

    def _source_id_from_info(self, info: dict[str, Any], platform: PlatformCode, url: str) -> str:
        raw_id = info.get("id")
        if isinstance(raw_id, str) and raw_id:
            return raw_id
        # Fallback: pull shortcode from URL
        if platform == "ig":
            match = re.search(r"/reel/([^/?#]+)/", url) or re.search(r"/p/([^/?#]+)/", url)
            if match:
                return match.group(1)
        if platform == "tt":
            match = re.search(r"/video/(\d+)", url)
            if match:
                return match.group(1)
        if platform == "yt":
            match = re.search(r"[?&]v=([A-Za-z0-9_\-]{6,})", url) or re.search(r"youtu\.be/([A-Za-z0-9_\-]{6,})", url)
            if match:
                return match.group(1)
        # Last resort — hash the URL so downstream code never sees an empty id
        return f"unknown-{abs(hash(url)) & 0xFFFFFF:x}"

    def _playback_url_for(
        self,
        info: dict[str, Any],
        platform: PlatformCode,
        source_url: str,
        source_id: str,
    ) -> str:
        """
        Return a URL the iOS WebView is allowed to load (youtube.com / instagram.com / tiktok.com).

        For YouTube we construct the embed URL with the full video duration; the pipeline
        later overrides this with a `?start=X&end=Y` variant once the clip window is chosen.
        """
        if platform == "yt":
            return f"https://www.youtube.com/embed/{source_id}"

        if platform == "ig":
            webpage = info.get("webpage_url") if isinstance(info, dict) else None
            if isinstance(webpage, str) and webpage:
                host = _host(webpage)
                if _normalize_host(host) == "instagram.com":
                    return webpage
            # Fallback: construct a canonical reel URL
            return f"https://www.instagram.com/reel/{source_id}/"

        if platform == "tt":
            webpage = info.get("webpage_url") if isinstance(info, dict) else None
            if isinstance(webpage, str) and webpage:
                host = _host(webpage)
                if _normalize_host(host) == "tiktok.com":
                    return webpage
            # Fallback — use the known-good TikTok URL pattern
            uploader = info.get("uploader_id") or info.get("uploader") or "unknown"
            return f"https://www.tiktok.com/@{uploader}/video/{source_id}"

        return source_url

    def _entry_to_url(self, entry: dict[str, Any]) -> str | None:
        url = entry.get("url")
        if isinstance(url, str) and url.startswith("http"):
            return url
        # yt-dlp flat mode sometimes returns just the video id — reconstruct the full URL
        extractor = (entry.get("ie_key") or entry.get("extractor") or "").lower()
        raw_id = entry.get("id")
        if not isinstance(raw_id, str) or not raw_id:
            return None
        if "youtube" in extractor:
            return f"https://www.youtube.com/watch?v={raw_id}"
        if "instagram" in extractor:
            return f"https://www.instagram.com/reel/{raw_id}/"
        if "tiktok" in extractor:
            uploader = entry.get("uploader") or entry.get("uploader_id") or "unknown"
            return f"https://www.tiktok.com/@{uploader}/video/{raw_id}"
        return None


__all__ = ["YtDlpAdapter"]
