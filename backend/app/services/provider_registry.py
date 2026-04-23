"""
Multi-platform video provider registry.

Pluggable adapters for Vimeo / Dailymotion / Bilibili / TikTok / Twitch.
Called from two seams:

  * `youtube._search_external_fallbacks` — when a YouTube-side search
    returns a thin pool, the registry is consulted for cross-platform
    candidates (same query, different platforms).
  * `reels._get_transcript(video_row)` — dispatches transcript fetches
    by the `provider` field stored on a video row, so Dailymotion rows
    use Dailymotion's subtitles endpoint, Twitch rows use Helix, etc.

Contract every provider MUST uphold:

  * `search(query, max_results)` returns a list of `ProviderCandidate`.
    Missing fields are OK (empty strings / zeros). NEVER raises — wrap
    network errors and return `[]`.
  * `fetch_transcript(video_id)` returns a `ProviderTranscript` or `None`
    when captions aren't available. NEVER raises.
  * All HTTP calls use short timeouts (≤ 10s) and are safe to cancel.

Preferred (non-scraping) upstreams:
    Vimeo        — api.vimeo.com         (auth: PAT)
    Dailymotion  — api.dailymotion.com   (keyless for public search)
    Bilibili     — api.bilibili.com      (keyless)
    Twitch       — api.twitch.tv/helix   (auth: client-credentials)
    TikTok       — yt-dlp tiktoksearch:  (existing adapter)

Feature flag: `PROVIDER_REGISTRY_ENABLED` (default False). While dormant
the registry exists in the import graph but returns empty for every query;
no outbound HTTP fires.
"""

from __future__ import annotations

import logging
import re
import threading
import time
import urllib.parse
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Literal

import requests

from ..config import get_settings

logger = logging.getLogger(__name__)

ProviderName = Literal["vimeo", "dailymotion", "bilibili", "tiktok", "twitch"]

_DEFAULT_TIMEOUT_SEC = 6.0

# Matches both SRT (`,`) and WebVTT (`.`) millisecond separators.
_SUBTITLE_TIMESTAMP_RE = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[.,](\d{3})\s+-->\s+(\d{1,2}):(\d{2}):(\d{2})[.,](\d{3})"
)

# Keyless scraping fallback. DuckDuckGo's html.duckduckgo.com endpoint is the
# stable bot-friendly surface; it wraps real URLs in a `/l/?uddg=` redirect,
# which we normalise by URL-unquoting the whole body before regexing.
_DDG_HTML_URL = "https://html.duckduckgo.com/html/"
_DDG_UA = "Mozilla/5.0 (compatible; ReelAI/1.0; +https://reelai.app/bot)"
_VIMEO_ID_RE = re.compile(r"vimeo\.com/(\d{6,12})")
_TWITCH_CLIP_RE = re.compile(r"clips\.twitch\.tv/([A-Za-z0-9][A-Za-z0-9_-]{7,})")


def _ddg_find(
    query: str,
    *,
    site_filter: str,
    pattern: re.Pattern[str],
    max_results: int,
    timeout: float = _DEFAULT_TIMEOUT_SEC,
) -> list[str]:
    """Keyless search hop via DuckDuckGo HTML. Narrows the query with
    `site:{site_filter}` and extracts ordered-unique captures of `pattern`
    from the decoded response body. Swallows all network errors.
    """
    cleaned = (query or "").strip()
    if not cleaned:
        return []
    try:
        resp = requests.get(
            _DDG_HTML_URL,
            params={"q": f"site:{site_filter} {cleaned}"},
            headers={"User-Agent": _DDG_UA},
            timeout=timeout,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.debug("ddg search failed (%s): %s", site_filter, exc)
        return []
    # DDG wraps real URLs as //duckduckgo.com/l/?uddg=<percent-encoded>; one
    # pass of unquote is enough to expose both the plain and wrapped forms.
    body = urllib.parse.unquote(resp.text or "")
    seen: list[str] = []
    for match in pattern.findall(body):
        if match not in seen:
            seen.append(match)
            if len(seen) >= max_results:
                break
    return seen


@dataclass
class ProviderCandidate:
    """Canonical search result. Maps 1:1 to a row inserted into `videos`."""

    provider: ProviderName
    video_id: str
    video_url: str          # canonical watch URL (what a human would open)
    playback_url: str       # embed/iframe URL (what ReelCard.tsx puts in src)
    title: str = ""
    description: str = ""
    channel: str = ""
    duration_sec: int = 0
    thumbnail_url: str = ""
    published_at: str | None = None  # ISO 8601 if known
    view_count: int = 0
    raw: dict[str, Any] = field(default_factory=dict)  # provider-specific extras


@dataclass
class ProviderTranscriptCue:
    start: float
    end: float
    text: str


@dataclass
class ProviderTranscript:
    provider: ProviderName
    video_id: str
    language: str            # BCP-47 when available ("en", "en-US", ...)
    cues: list[ProviderTranscriptCue]


class Provider(ABC):
    name: ProviderName

    @abstractmethod
    def search(self, query: str, max_results: int) -> list[ProviderCandidate]:
        """Best-effort keyword search. Returns [] on any failure."""

    @abstractmethod
    def fetch_transcript(self, video_id: str) -> ProviderTranscript | None:
        """Best-effort captions fetch. Returns None when unavailable."""


# --------------------------------------------------------------------- #
# Dailymotion — keyless search against api.dailymotion.com               #
# --------------------------------------------------------------------- #


class DailymotionProvider(Provider):
    name = "dailymotion"
    SEARCH_URL = "https://api.dailymotion.com/videos"
    SUBTITLES_URL = "https://api.dailymotion.com/video/{id}/subtitles"
    _FIELDS = (
        "id,title,description,duration,thumbnail_480_url,"
        "channel.screenname,created_time,views_total,url"
    )

    def search(self, query: str, max_results: int) -> list[ProviderCandidate]:
        query = (query or "").strip()
        if not query:
            return []
        limit = max(1, min(int(max_results or 1), 20))
        try:
            resp = requests.get(
                self.SEARCH_URL,
                params={
                    "search": query,
                    "limit": limit,
                    "fields": self._FIELDS,
                    "sort": "relevance",
                },
                timeout=_DEFAULT_TIMEOUT_SEC,
            )
            resp.raise_for_status()
            payload = resp.json()
        except (requests.RequestException, ValueError) as exc:
            logger.debug("dailymotion search failed: %s", exc)
            return []
        items = payload.get("list") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            return []
        candidates: list[ProviderCandidate] = []
        for entry in items:
            if not isinstance(entry, dict):
                continue
            video_id = str(entry.get("id") or "").strip()
            if not video_id:
                continue
            candidates.append(
                ProviderCandidate(
                    provider="dailymotion",
                    video_id=video_id,
                    video_url=str(entry.get("url") or f"https://www.dailymotion.com/video/{video_id}"),
                    playback_url=f"https://www.dailymotion.com/embed/video/{video_id}",
                    title=str(entry.get("title") or "").strip(),
                    description=str(entry.get("description") or "").strip(),
                    channel=str(entry.get("channel.screenname") or "").strip(),
                    duration_sec=int(entry.get("duration") or 0),
                    thumbnail_url=str(entry.get("thumbnail_480_url") or "").strip(),
                    published_at=str(entry.get("created_time") or "") or None,
                    view_count=int(entry.get("views_total") or 0),
                    raw=entry,
                )
            )
        return candidates

    def fetch_transcript(self, video_id: str) -> ProviderTranscript | None:
        vid = str(video_id or "").strip()
        if not vid:
            return None
        try:
            list_resp = requests.get(
                self.SUBTITLES_URL.format(id=vid),
                timeout=_DEFAULT_TIMEOUT_SEC,
            )
            # 401/403/404: captions unlisted, auth-gated, or missing — treat
            # all as "no transcript available" rather than an error.
            if list_resp.status_code in (401, 403, 404):
                return None
            list_resp.raise_for_status()
            payload = list_resp.json()
        except (requests.RequestException, ValueError) as exc:
            logger.debug("dailymotion subtitles list failed: %s", exc)
            return None
        tracks = payload.get("list") if isinstance(payload, dict) else None
        if not isinstance(tracks, list):
            return None
        track = _pick_english_track(tracks, lang_key="language")
        if track is None:
            return None
        track_url = str(track.get("url") or "").strip()
        if not track_url:
            return None
        try:
            text_resp = requests.get(track_url, timeout=_DEFAULT_TIMEOUT_SEC)
            text_resp.raise_for_status()
        except requests.RequestException as exc:
            logger.debug("dailymotion subtitle download failed: %s", exc)
            return None
        cues = _parse_subtitle_cues(text_resp.text)
        if not cues:
            return None
        return ProviderTranscript(
            provider="dailymotion",
            video_id=vid,
            language=str(track.get("language") or "en"),
            cues=cues,
        )


# --------------------------------------------------------------------- #
# Bilibili — keyless search against api.bilibili.com                     #
# --------------------------------------------------------------------- #


class BilibiliProvider(Provider):
    name = "bilibili"
    SEARCH_URL = "https://api.bilibili.com/x/web-interface/search/type"
    _HEADERS = {
        # Bilibili rejects requests without a browser-like UA. Keep this
        # modest; scraping isn't the goal here, just their public JSON API.
        "User-Agent": "Mozilla/5.0 (compatible; ReelAI/1.0)",
        "Referer": "https://www.bilibili.com/",
    }

    def search(self, query: str, max_results: int) -> list[ProviderCandidate]:
        query = (query or "").strip()
        if not query:
            return []
        page_size = max(1, min(int(max_results or 1), 20))
        try:
            resp = requests.get(
                self.SEARCH_URL,
                params={
                    "search_type": "video",
                    "keyword": query,
                    "page": 1,
                    "page_size": page_size,
                },
                headers=self._HEADERS,
                timeout=_DEFAULT_TIMEOUT_SEC,
            )
            resp.raise_for_status()
            payload = resp.json()
        except (requests.RequestException, ValueError) as exc:
            logger.debug("bilibili search failed: %s", exc)
            return []
        if not isinstance(payload, dict) or payload.get("code") not in (0, None):
            return []
        data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        items = data.get("result") if isinstance(data, dict) else None
        if not isinstance(items, list):
            return []
        candidates: list[ProviderCandidate] = []
        for entry in items:
            if not isinstance(entry, dict):
                continue
            bvid = str(entry.get("bvid") or "").strip()
            if not bvid:
                continue
            candidates.append(
                ProviderCandidate(
                    provider="bilibili",
                    video_id=bvid,
                    video_url=f"https://www.bilibili.com/video/{bvid}",
                    playback_url=f"https://player.bilibili.com/player.html?bvid={bvid}&high_quality=1",
                    title=_strip_bilibili_highlight(str(entry.get("title") or "")),
                    description=str(entry.get("description") or "").strip(),
                    channel=str(entry.get("author") or "").strip(),
                    duration_sec=_parse_bilibili_duration(str(entry.get("duration") or "")),
                    thumbnail_url=_normalize_protocol(str(entry.get("pic") or "")),
                    published_at=None,  # entry["pubdate"] is epoch seconds; leave raw
                    view_count=int(entry.get("play") or 0),
                    raw=entry,
                )
            )
        return candidates

    def fetch_transcript(self, video_id: str) -> ProviderTranscript | None:
        # Bilibili's native /x/player/v2 captions endpoint requires wbi
        # signing (reverse-engineered but fragile). yt-dlp already handles
        # that machinery; we delegate rather than reimplement.
        vid = str(video_id or "").strip()
        if not vid:
            return None
        return _yt_dlp_fetch_vtt_transcript(
            video_url=f"https://www.bilibili.com/video/{vid}",
            provider="bilibili",
            video_id=vid,
            timeout=10.0,
        )


# --------------------------------------------------------------------- #
# Stubs — each documents the preferred upstream + env var it'll need.    #
# --------------------------------------------------------------------- #


class VimeoProvider(Provider):
    """
    Vimeo API v3 (api.vimeo.com). Auth is a bearer PAT stored at
    `config.vimeo_access_token`; when unset, both methods short-circuit
    to empty/None so the registry stays silent.

    Search:     GET /videos?query=...&per_page=...&sort=relevant
    Transcript: GET /videos/{id}/texttracks  (each returns WebVTT URLs per language)
    Playback:   https://player.vimeo.com/video/{id}  (supports `#t=` segment params)
    """

    name = "vimeo"
    SEARCH_URL = "https://api.vimeo.com/videos"
    TRANSCRIPT_URL = "https://api.vimeo.com/videos/{id}/texttracks"
    _FIELDS = (
        "uri,name,description,duration,pictures.sizes,"
        "user.name,created_time,stats.plays,link"
    )

    def _auth_headers(self) -> dict[str, str] | None:
        token = (get_settings().vimeo_access_token or "").strip()
        if not token:
            return None
        return {
            "Authorization": f"bearer {token}",
            "Accept": "application/vnd.vimeo.*+json;version=3.4",
        }

    def search(self, query: str, max_results: int) -> list[ProviderCandidate]:
        query = (query or "").strip()
        if not query:
            return []
        headers = self._auth_headers()
        if headers is None:
            # Keyless fallback: DuckDuckGo for site:vimeo.com URLs, then
            # Vimeo's public oEmbed endpoint for metadata per hit.
            return self._search_keyless(query, max_results)
        per_page = max(1, min(int(max_results or 1), 25))
        try:
            resp = requests.get(
                self.SEARCH_URL,
                params={
                    "query": query,
                    "per_page": per_page,
                    "fields": self._FIELDS,
                    "direction": "desc",
                    "sort": "relevant",
                },
                headers=headers,
                timeout=_DEFAULT_TIMEOUT_SEC,
            )
            resp.raise_for_status()
            payload = resp.json()
        except (requests.RequestException, ValueError) as exc:
            logger.debug("vimeo search failed: %s", exc)
            return []
        items = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            return []
        candidates: list[ProviderCandidate] = []
        for entry in items:
            if not isinstance(entry, dict):
                continue
            # Vimeo URIs look like "/videos/{id}"; the numeric id is the tail.
            uri = str(entry.get("uri") or "")
            video_id = uri.rsplit("/", 1)[-1].strip()
            if not video_id:
                continue
            pictures = entry.get("pictures") if isinstance(entry.get("pictures"), dict) else {}
            sizes = pictures.get("sizes") if isinstance(pictures, dict) else None
            thumb_url = ""
            if isinstance(sizes, list) and sizes:
                for size in sizes:
                    if not isinstance(size, dict):
                        continue
                    if int(size.get("width") or 0) >= 480:
                        thumb_url = str(size.get("link") or "").strip()
                        break
                if not thumb_url:
                    tail = sizes[-1] if isinstance(sizes[-1], dict) else {}
                    thumb_url = str(tail.get("link") or "").strip()
            user = entry.get("user") if isinstance(entry.get("user"), dict) else {}
            stats = entry.get("stats") if isinstance(entry.get("stats"), dict) else {}
            candidates.append(
                ProviderCandidate(
                    provider="vimeo",
                    video_id=video_id,
                    video_url=str(entry.get("link") or f"https://vimeo.com/{video_id}"),
                    playback_url=f"https://player.vimeo.com/video/{video_id}",
                    title=str(entry.get("name") or "").strip(),
                    description=str(entry.get("description") or "").strip(),
                    channel=str(user.get("name") or "").strip(),
                    duration_sec=int(entry.get("duration") or 0),
                    thumbnail_url=thumb_url,
                    published_at=str(entry.get("created_time") or "") or None,
                    view_count=int(stats.get("plays") or 0),
                    raw=entry,
                )
            )
        return candidates

    def _search_keyless(self, query: str, max_results: int) -> list[ProviderCandidate]:
        capped = max(1, min(int(max_results or 1), 15))
        ids = _ddg_find(
            query,
            site_filter="vimeo.com",
            pattern=_VIMEO_ID_RE,
            max_results=capped,
        )
        if not ids:
            return []
        candidates: list[ProviderCandidate] = []
        for vid in ids:
            meta = self._oembed(vid)
            if meta is None:
                # Skip when oEmbed can't hydrate the ID (removed/private video).
                continue
            candidates.append(
                ProviderCandidate(
                    provider="vimeo",
                    video_id=vid,
                    video_url=f"https://vimeo.com/{vid}",
                    playback_url=f"https://player.vimeo.com/video/{vid}",
                    title=str(meta.get("title") or "").strip(),
                    description=str(meta.get("description") or "").strip(),
                    channel=str(meta.get("author_name") or "").strip(),
                    duration_sec=int(meta.get("duration") or 0),
                    thumbnail_url=str(meta.get("thumbnail_url") or "").strip(),
                    published_at=str(meta.get("upload_date") or "") or None,
                    view_count=0,  # oEmbed doesn't expose view counts
                    raw=meta,
                )
            )
        return candidates

    @staticmethod
    def _oembed(video_id: str) -> dict[str, Any] | None:
        try:
            resp = requests.get(
                "https://vimeo.com/api/oembed.json",
                params={"url": f"https://vimeo.com/{video_id}"},
                timeout=_DEFAULT_TIMEOUT_SEC,
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            return data if isinstance(data, dict) else None
        except (requests.RequestException, ValueError) as exc:
            logger.debug("vimeo oembed failed for %s: %s", video_id, exc)
            return None

    def fetch_transcript(self, video_id: str) -> ProviderTranscript | None:
        vid = str(video_id or "").strip()
        if not vid:
            return None
        headers = self._auth_headers()
        if headers is None:
            # Keyless transcript path: yt-dlp understands vimeo.com/{id} and
            # returns subtitle URLs without a PAT.
            return _yt_dlp_fetch_vtt_transcript(
                video_url=f"https://vimeo.com/{vid}",
                provider="vimeo",
                video_id=vid,
                timeout=10.0,
            )
        try:
            list_resp = requests.get(
                self.TRANSCRIPT_URL.format(id=vid),
                headers=headers,
                timeout=_DEFAULT_TIMEOUT_SEC,
            )
            if list_resp.status_code in (401, 403, 404):
                return None
            list_resp.raise_for_status()
            payload = list_resp.json()
        except (requests.RequestException, ValueError) as exc:
            logger.debug("vimeo texttracks list failed: %s", exc)
            return None
        tracks = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(tracks, list):
            return None
        track = _pick_english_track(tracks, lang_key="language")
        if track is None:
            return None
        track_link = str(track.get("link") or "").strip()
        if not track_link:
            return None
        # The WebVTT file at `link` is public, but fetching with the PAT is
        # harmless and keeps the auth posture consistent.
        try:
            vtt_resp = requests.get(track_link, headers=headers, timeout=_DEFAULT_TIMEOUT_SEC)
            vtt_resp.raise_for_status()
        except requests.RequestException as exc:
            logger.debug("vimeo texttrack download failed: %s", exc)
            return None
        cues = _parse_subtitle_cues(vtt_resp.text)
        if not cues:
            return None
        return ProviderTranscript(
            provider="vimeo",
            video_id=vid,
            language=str(track.get("language") or "en"),
            cues=cues,
        )


class TwitchProvider(Provider):
    """
    Twitch Helix API for clips. Helix has no free-text clip search; we
    translate the query into a game_id via /helix/games?name=... and then
    pull top clips for that game. When the query doesn't match any game
    name (common for long-tail educational queries) we simply return no
    candidates — there is no cheap broadcaster-name fallback that doesn't
    require a second round-trip per result.

    Auth: app-access token via client-credentials grant against
    https://id.twitch.tv/oauth2/token, cached at class scope until 60s
    before its `expires_in`.
    Captions: Helix exposes none. We fall through to yt-dlp subs on the
    clip URL (identical pattern to TikTok + Bilibili).
    Playback: https://clips.twitch.tv/embed?clip={id}&parent={host} — the
    client fills `parent` at render time (WKWebView host / iframe origin).
    """

    name = "twitch"
    OAUTH_URL = "https://id.twitch.tv/oauth2/token"
    GAMES_URL = "https://api.twitch.tv/helix/games"
    CLIPS_URL = "https://api.twitch.tv/helix/clips"
    _TIMEOUT_SEC = 6.0
    _TOKEN_REFRESH_MARGIN_SEC = 60.0

    # Class-level cache so every TwitchProvider instance shares one token.
    _token_access: str = ""
    _token_expires_at: float = 0.0
    _token_lock: threading.Lock = threading.Lock()

    def _get_app_auth(self) -> tuple[str, str] | None:
        """Returns (access_token, client_id) or None when unconfigured / failed."""
        settings = get_settings()
        client_id = (settings.twitch_client_id or "").strip()
        client_secret = (settings.twitch_client_secret or "").strip()
        if not client_id or not client_secret:
            return None
        now = time.time()
        with TwitchProvider._token_lock:
            if (
                TwitchProvider._token_access
                and TwitchProvider._token_expires_at > now + self._TOKEN_REFRESH_MARGIN_SEC
            ):
                return TwitchProvider._token_access, client_id
            try:
                resp = requests.post(
                    self.OAUTH_URL,
                    data={
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "grant_type": "client_credentials",
                    },
                    timeout=self._TIMEOUT_SEC,
                )
                resp.raise_for_status()
                payload = resp.json()
            except (requests.RequestException, ValueError) as exc:
                logger.debug("twitch oauth failed: %s", exc)
                return None
            token = str((payload or {}).get("access_token") or "").strip()
            expires_in = int((payload or {}).get("expires_in") or 0)
            if not token or expires_in <= 0:
                return None
            TwitchProvider._token_access = token
            TwitchProvider._token_expires_at = now + float(expires_in)
            return token, client_id

    def search(self, query: str, max_results: int) -> list[ProviderCandidate]:
        query = (query or "").strip()
        if not query:
            return []
        auth = self._get_app_auth()
        if auth is None:
            # Keyless fallback: DuckDuckGo for site:clips.twitch.tv URLs.
            # Metadata is degraded (title/duration/views all empty) because
            # Helix is the only public metadata endpoint and it needs auth.
            # Transcripts still work keyless via yt-dlp on the clip URL.
            return self._search_keyless(query, max_results)
        token, client_id = auth
        headers = {
            "Authorization": f"Bearer {token}",
            "Client-Id": client_id,
        }
        # Step 1: resolve game_id from the query as a game name (case-insensitive).
        try:
            games_resp = requests.get(
                self.GAMES_URL,
                params={"name": query},
                headers=headers,
                timeout=self._TIMEOUT_SEC,
            )
            games_resp.raise_for_status()
            games_payload = games_resp.json()
        except (requests.RequestException, ValueError) as exc:
            logger.debug("twitch games lookup failed: %s", exc)
            return []
        games = (games_payload or {}).get("data") if isinstance(games_payload, dict) else None
        if not isinstance(games, list) or not games:
            return []
        game = games[0] if isinstance(games[0], dict) else {}
        game_id = str(game.get("id") or "").strip()
        if not game_id:
            return []
        # Step 2: fetch top clips for the game.
        first = max(1, min(int(max_results or 1), 50))
        try:
            clips_resp = requests.get(
                self.CLIPS_URL,
                params={"game_id": game_id, "first": first},
                headers=headers,
                timeout=self._TIMEOUT_SEC,
            )
            clips_resp.raise_for_status()
            clips_payload = clips_resp.json()
        except (requests.RequestException, ValueError) as exc:
            logger.debug("twitch clips fetch failed: %s", exc)
            return []
        clips = (clips_payload or {}).get("data") if isinstance(clips_payload, dict) else None
        if not isinstance(clips, list):
            return []
        candidates: list[ProviderCandidate] = []
        for clip in clips:
            if not isinstance(clip, dict):
                continue
            clip_id = str(clip.get("id") or "").strip()
            if not clip_id:
                continue
            candidates.append(
                ProviderCandidate(
                    provider="twitch",
                    video_id=clip_id,
                    video_url=str(clip.get("url") or f"https://clips.twitch.tv/{clip_id}"),
                    playback_url=str(
                        clip.get("embed_url") or f"https://clips.twitch.tv/embed?clip={clip_id}"
                    ),
                    title=str(clip.get("title") or "").strip(),
                    description="",
                    channel=str(clip.get("broadcaster_name") or "").strip(),
                    duration_sec=int(float(clip.get("duration") or 0)),
                    thumbnail_url=str(clip.get("thumbnail_url") or "").strip(),
                    published_at=str(clip.get("created_at") or "") or None,
                    view_count=int(clip.get("view_count") or 0),
                    raw=clip,
                )
            )
        return candidates

    def _search_keyless(self, query: str, max_results: int) -> list[ProviderCandidate]:
        capped = max(1, min(int(max_results or 1), 15))
        slugs = _ddg_find(
            query,
            site_filter="clips.twitch.tv",
            pattern=_TWITCH_CLIP_RE,
            max_results=capped,
        )
        return [
            ProviderCandidate(
                provider="twitch",
                video_id=slug,
                video_url=f"https://clips.twitch.tv/{slug}",
                playback_url=f"https://clips.twitch.tv/embed?clip={slug}",
                title="",  # no keyless Twitch metadata API
                description="",
                channel="",
                duration_sec=0,
                thumbnail_url="",
                published_at=None,
                view_count=0,
            )
            for slug in slugs
        ]

    def fetch_transcript(self, video_id: str) -> ProviderTranscript | None:
        vid = str(video_id or "").strip()
        if not vid:
            return None
        return _yt_dlp_fetch_vtt_transcript(
            video_url=f"https://clips.twitch.tv/{vid}",
            provider="twitch",
            video_id=vid,
            timeout=self._TIMEOUT_SEC,
        )


class TikTokProvider(Provider):
    """
    TikTok via yt-dlp's `tiktoksearch{N}:{query}` pseudo-URL. yt-dlp is the
    ONLY sustainable search path for TikTok — there is no public keyword
    API, and scraping their web search HTML gets flagged immediately.

    `extract_flat=True` keeps the metadata round-trip cheap: yt-dlp only
    pulls the search-results page, not each video. Transcripts DO need a
    per-video extract_info call (to pick up auto-captions URLs), routed
    through the shared yt-dlp helper.

    Playback: https://www.tiktok.com/embed/v2/{video_id} (iframe-friendly).
    """

    name = "tiktok"
    _TIMEOUT_SEC = 10.0
    _MAX_RESULTS = 20

    def search(self, query: str, max_results: int) -> list[ProviderCandidate]:
        query = (query or "").strip()
        if not query:
            return []
        try:
            import yt_dlp  # type: ignore
        except Exception:
            return []
        capped = max(1, min(int(max_results or 1), self._MAX_RESULTS))
        pseudo_url = f"tiktoksearch{capped}:{query}"
        opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,
            "skip_download": True,
            "socket_timeout": int(self._TIMEOUT_SEC),
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(pseudo_url, download=False)
        except Exception as exc:
            logger.debug("tiktok search failed: %s", exc)
            return []
        entries = (info or {}).get("entries") if isinstance(info, dict) else None
        if not isinstance(entries, list):
            return []
        candidates: list[ProviderCandidate] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            video_id = str(entry.get("id") or "").strip()
            if not video_id:
                continue
            uploader = str(entry.get("uploader") or entry.get("channel") or "").strip()
            canonical = str(
                entry.get("webpage_url")
                or entry.get("url")
                or f"https://www.tiktok.com/@{(uploader.lstrip('@') or '_')}/video/{video_id}"
            ).strip()
            candidates.append(
                ProviderCandidate(
                    provider="tiktok",
                    video_id=video_id,
                    video_url=canonical,
                    playback_url=f"https://www.tiktok.com/embed/v2/{video_id}",
                    title=str(entry.get("title") or "").strip(),
                    description=str(entry.get("description") or "").strip(),
                    channel=uploader,
                    duration_sec=int(entry.get("duration") or 0),
                    thumbnail_url=str(entry.get("thumbnail") or "").strip(),
                    published_at=str(entry.get("upload_date") or "") or None,
                    view_count=int(entry.get("view_count") or 0),
                    raw=entry,
                )
            )
        return candidates

    def fetch_transcript(self, video_id: str) -> ProviderTranscript | None:
        vid = str(video_id or "").strip()
        if not vid:
            return None
        # TikTok's canonical URL needs the uploader handle in the path, but
        # the `@_` alias form still resolves for yt-dlp — the server 302s
        # to the real handle before captions are pulled.
        return _yt_dlp_fetch_vtt_transcript(
            video_url=f"https://www.tiktok.com/@_/video/{vid}",
            provider="tiktok",
            video_id=vid,
            timeout=self._TIMEOUT_SEC,
        )


# --------------------------------------------------------------------- #
# Registry + helpers                                                     #
# --------------------------------------------------------------------- #


class ProviderRegistry:
    """
    Thin aggregator. The order below mirrors the user-facing preference
    captured in the /ultraplan: Vimeo → Dailymotion → Bilibili → TikTok
    → Twitch. YouTube is NOT in this registry — it's the native path.
    """

    def __init__(self, providers: list[Provider] | None = None):
        self._providers: list[Provider] = providers if providers is not None else [
            VimeoProvider(),
            DailymotionProvider(),
            BilibiliProvider(),
            TikTokProvider(),
            TwitchProvider(),
        ]
        self._by_name: dict[str, Provider] = {p.name: p for p in self._providers}

    @property
    def enabled(self) -> bool:
        return bool(get_settings().provider_registry_enabled)

    def get(self, name: str) -> Provider | None:
        return self._by_name.get(name)

    def search_all(
        self,
        query: str,
        max_results_per_provider: int,
        *,
        per_provider_timeout_sec: float = 15.0,
    ) -> list[ProviderCandidate]:
        """
        Fan out search across every registered provider **in parallel**.
        Results are concatenated in registry order (Vimeo → Dailymotion →
        Bilibili → TikTok → Twitch) regardless of completion order, so the
        output is deterministic. Deduplication by video_url is the caller's
        responsibility. Dormant when the feature flag is off.
        """
        if not self.enabled:
            return []
        query = (query or "").strip()
        if not query:
            return []
        if not self._providers:
            return []
        limit = max(1, int(max_results_per_provider or 1))
        results_by_idx: dict[int, list[ProviderCandidate]] = {}
        workers = min(max(1, len(self._providers)), 5)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {
                executor.submit(provider.search, query, limit): idx
                for idx, provider in enumerate(self._providers)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    rows = future.result(timeout=per_provider_timeout_sec)
                except Exception as exc:
                    # Providers already swallow their own network errors and
                    # return []; this catches unanticipated bugs + timeouts.
                    logger.debug(
                        "%s.search raised: %s", self._providers[idx].name, exc
                    )
                    rows = []
                results_by_idx[idx] = list(rows or [])
        out: list[ProviderCandidate] = []
        for idx in range(len(self._providers)):
            out.extend(results_by_idx.get(idx, []))
        return out

    def fetch_transcript(self, provider: str, video_id: str) -> ProviderTranscript | None:
        """Dispatch by provider name. Returns None for unknown providers."""
        if not self.enabled:
            return None
        adapter = self._by_name.get(str(provider or "").strip().lower())
        if adapter is None:
            return None
        try:
            return adapter.fetch_transcript(video_id)
        except Exception as exc:
            logger.debug("%s.fetch_transcript raised: %s", adapter.name, exc)
            return None


def _parse_subtitle_cues(text: str) -> list[ProviderTranscriptCue]:
    """Parse a SRT or WebVTT payload into cues. Tolerant of format variations.
    Strips inline `<…>` tags (e.g. karaoke/word-timing markers) from cue text.
    Returns `[]` when no cue-like timestamp lines are found.
    """
    cues: list[ProviderTranscriptCue] = []
    if not text:
        return cues
    lines = text.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        match = _SUBTITLE_TIMESTAMP_RE.search(lines[i])
        if not match:
            i += 1
            continue
        sh, sm, ss, sms, eh, em, es, ems = match.groups()
        start = int(sh) * 3600 + int(sm) * 60 + int(ss) + int(sms) / 1000.0
        end = int(eh) * 3600 + int(em) * 60 + int(es) + int(ems) / 1000.0
        i += 1
        buf: list[str] = []
        while i < n and lines[i].strip():
            cleaned = re.sub(r"<[^>]+>", "", lines[i]).replace("&nbsp;", " ").strip()
            if cleaned:
                buf.append(cleaned)
            i += 1
        if buf:
            cues.append(
                ProviderTranscriptCue(
                    start=start,
                    end=max(end, start + 0.01),
                    text=" ".join(buf),
                )
            )
    return cues


def _pick_english_track(tracks: list[dict[str, Any]], lang_key: str = "language") -> dict[str, Any] | None:
    """Prefer a track whose language starts with 'en'; else return the first."""
    if not tracks:
        return None
    for track in tracks:
        if not isinstance(track, dict):
            continue
        lang = str(track.get(lang_key) or "").strip().lower()
        if lang.startswith("en"):
            return track
    return next((t for t in tracks if isinstance(t, dict)), None)


def _yt_dlp_fetch_vtt_transcript(
    *,
    video_url: str,
    provider: ProviderName,
    video_id: str,
    timeout: float = 10.0,
) -> ProviderTranscript | None:
    """Shared captions fetcher for providers that fall back to yt-dlp for
    their subtitle extraction. Prefers user-uploaded subs over auto-generated,
    English over other languages. Returns None when yt-dlp is unavailable,
    no captions exist, or the VTT body parses to zero cues.

    Flow:
      1. Lazy-import yt_dlp (heavy; not installed on every deployment).
      2. `extract_info(url, download=False)` with skip_download — info dict
         carries `subtitles` (manual) and `automatic_captions` dicts keyed by
         language code; each value is a list of track dicts with `ext`/`url`.
      3. Pick the best track, fetch it via `requests` (yt-dlp doesn't pull
         the subtitle body itself when `skip_download` is set), parse as
         SRT/VTT, wrap in ProviderTranscript.
    """
    try:
        import yt_dlp  # type: ignore
    except Exception:
        return None
    opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "socket_timeout": int(timeout),
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
    except Exception as exc:
        logger.debug("%s yt-dlp extract failed: %s", provider, exc)
        return None
    if not isinstance(info, dict):
        return None
    for kind_key in ("subtitles", "automatic_captions"):
        subs = info.get(kind_key)
        if not isinstance(subs, dict) or not subs:
            continue
        # Prefer English language variants ("en", "en-US", "en-GB", "en-orig").
        lang_key = next(
            (k for k in subs.keys() if str(k or "").lower().startswith("en")),
            next(iter(subs.keys()), None),
        )
        if not lang_key:
            continue
        tracks = subs.get(lang_key) or []
        if not isinstance(tracks, list) or not tracks:
            continue
        vtt_url = ""
        # Prefer VTT/WebVTT ext; otherwise accept the first track with a URL.
        for track in tracks:
            if not isinstance(track, dict):
                continue
            ext = str(track.get("ext") or "").lower()
            url = str(track.get("url") or "").strip()
            if url and ext in ("vtt", "webvtt"):
                vtt_url = url
                break
        if not vtt_url:
            for track in tracks:
                if isinstance(track, dict):
                    url = str(track.get("url") or "").strip()
                    if url:
                        vtt_url = url
                        break
        if not vtt_url:
            continue
        try:
            resp = requests.get(vtt_url, timeout=timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.debug("%s subtitle download failed: %s", provider, exc)
            continue
        cues = _parse_subtitle_cues(resp.text)
        if not cues:
            continue
        return ProviderTranscript(
            provider=provider,
            video_id=video_id,
            language=str(lang_key),
            cues=cues,
        )
    return None


def _strip_bilibili_highlight(text: str) -> str:
    # Bilibili returns titles with <em class="keyword">...</em> markers
    # around search-term matches. Strip them before persisting.
    return text.replace('<em class="keyword">', "").replace("</em>", "").strip()


def _parse_bilibili_duration(raw: str) -> int:
    # Bilibili reports durations as "mm:ss" (or "hh:mm:ss" for long uploads).
    raw = raw.strip()
    if not raw or ":" not in raw:
        try:
            return int(raw)
        except ValueError:
            return 0
    parts = raw.split(":")
    try:
        parts_int = [int(p) for p in parts]
    except ValueError:
        return 0
    total = 0
    for value in parts_int:
        total = total * 60 + value
    return total


def _normalize_protocol(url: str) -> str:
    # Bilibili thumbnail URLs occasionally come back protocol-relative.
    if url.startswith("//"):
        return "https:" + url
    return url


__all__ = [
    "Provider",
    "ProviderCandidate",
    "ProviderName",
    "ProviderRegistry",
    "ProviderTranscript",
    "ProviderTranscriptCue",
    "DailymotionProvider",
    "BilibiliProvider",
    "VimeoProvider",
    "TwitchProvider",
    "TikTokProvider",
    "_parse_subtitle_cues",
    "_pick_english_track",
    "_yt_dlp_fetch_vtt_transcript",
    "_ddg_find",
]
