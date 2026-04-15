import hashlib
import json
import logging
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Any, Literal
from urllib.parse import parse_qs, unquote, urlparse

import requests
from requests.adapters import HTTPAdapter
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

from ..config import get_settings
from ..db import dumps_json, fetch_one, get_conn, now_iso, upsert


# ---------------------------------------------------------------------------
# Proxy rotation & stealth helpers
# ---------------------------------------------------------------------------
# Rotating User-Agent pool — cycle through realistic browser fingerprints to
# avoid bot detection by YouTube.  The pool is intentionally small (fewer
# unique fingerprints = fewer "never-seen-before" signals) and only includes
# Chrome on mainstream OSes to match YouTube's normal traffic profile.
_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
]

# Realistic browser headers to accompany each request.
_STEALTH_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Cache-Control": "max-age=0",
    "Sec-Ch-Ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"macOS"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}


def _random_user_agent() -> str:
    return random.choice(_USER_AGENTS)


class _ProxyRotator:
    """Thread-safe round-robin proxy rotator.

    Accepts a comma-separated list of proxy URLs. Each call to ``next()``
    returns the next proxy in rotation.  If the list is empty, returns
    ``None`` (no proxy).
    """

    def __init__(self, proxy_urls_csv: str) -> None:
        self._proxies: list[str] = [
            p.strip() for p in (proxy_urls_csv or "").split(",") if p.strip()
        ]
        self._idx = 0
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return len(self._proxies) > 0

    def next(self) -> dict[str, str] | None:
        if not self._proxies:
            return None
        with self._lock:
            proxy = self._proxies[self._idx % len(self._proxies)]
            self._idx += 1
        return {"http": proxy, "https": proxy}

    def all(self) -> list[str]:
        return list(self._proxies)
from .transcript_validation import TranscriptQuality, validate_transcript

logger = logging.getLogger(__name__)


class YouTubeApiRequestError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Scraping constants
# ---------------------------------------------------------------------------
# A YouTube video ID is always exactly 11 characters, chosen from:
#   uppercase letters, lowercase letters, digits, "-" and "_"
# This pattern is the single source of truth. If we ever see something that
# claims to be a video ID but doesn't match, we reject it — that's a cheap
# guardrail against YouTube changing its format, against accidentally picking
# up unrelated junk from the HTML (e.g. a 40-char tracking hash), and against
# injecting garbage into the rest of the pipeline.
YOUTUBE_VIDEO_ID_CHARSET = r"[A-Za-z0-9_-]"
YOUTUBE_VIDEO_ID_PATTERN = rf"{YOUTUBE_VIDEO_ID_CHARSET}{{11}}"
YOUTUBE_VIDEO_ID_REGEX = re.compile(rf"^{YOUTUBE_VIDEO_ID_PATTERN}$")
# Used when scanning raw HTML for fallback IDs — the JSON escapes ensure we
# only match IDs that appear in a proper `"videoId":"xxxxxxxxxxx"` context,
# not arbitrary 11-char tokens that happen to look like a YouTube ID.
_VIDEO_ID_IN_JSON_REGEX = re.compile(rf'"videoId"\s*:\s*"({YOUTUBE_VIDEO_ID_PATTERN})"')

# Flexible match for the ytInitialData assignment used by youtube.com search
# and watch pages.  Accepts any of the observed forms:
#   var ytInitialData = {...}
#   window["ytInitialData"] = {...}
#   window['ytInitialData'] = {...}
#   window.ytInitialData = {...}
#   ytInitialData = {...}
#   ytInitialData={...}
# The ``brace`` named group captures the position of the opening ``{`` so the
# caller can kick off balanced-JSON extraction from there.
_YT_INITIAL_DATA_ASSIGN_RE = re.compile(
    r"(?:var\s+|window\s*(?:\[\s*[\"']|\.))?ytInitialData[\"']?\s*\]?\s*=\s*(?P<brace>\{)",
)

# Object-literal form: ``"ytInitialData": {...}``.  Used as a last-resort
# fallback because it also matches inside string literals; callers must
# validate by balanced-JSON + json.loads.
_YT_INITIAL_DATA_KEY_RE = re.compile(r'"ytInitialData"\s*:\s*(?P<brace>\{)')

# Page <title> extraction — used only for diagnostic logging when ytInitialData
# extraction fails, so we can distinguish a consent wall from a bot challenge
# from a real results page with a renamed variable.
_HTML_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)


def _is_valid_youtube_video_id(value: Any) -> bool:
    """Return True iff *value* is a string matching the YouTube video ID format.

    Centralising this check keeps every caller honest — anywhere we extract a
    video ID from HTML or JSON, we should filter through this gate before
    using it as a dict key, inserting it into the DB, or returning it.
    """
    return isinstance(value, str) and bool(YOUTUBE_VIDEO_ID_REGEX.match(value))


RetrievalProfile = Literal["bootstrap", "deep"]
GraphProfile = Literal["off", "light", "deep"]


def _cache_key(*parts: str) -> str:
    value = "|".join(parts)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Transcript source provenance
# ---------------------------------------------------------------------------

# Accurate names reflecting actual extraction method — "official_captions" is
# reserved for the YouTube Data API OAuth path (captions.download), which
# requires edit permission + quota.  The youtube_transcript_api library uses
# an undocumented web-client endpoint, hence "web_caption_track".
SOURCE_KIND_PRIORITY: dict[str, int] = {
    "official_captions": 6,
    "web_caption_track": 5,
    "innertube_caption_track": 4,
    "watch_page_caption_track": 3,
    "yt_dlp_subtitle": 3,
    "asr_local": 2,
    "asr_api": 1,
}

# Status-specific cache TTLs (seconds).  Non-success states get short recheck
# windows to avoid poisoning retrieval for videos whose captions appear later
# or where an extractor had a transient bad response.
CACHE_TTL_BY_STATUS: dict[str | None, int] = {
    "success": 0,                  # indefinite — provenance upgrade after 24h
    "failed_quality": 2 * 3600,    # 2 hours
    "failed_access": 1 * 3600,     # 1 hour
    "failed_rate_limit": 30 * 60,  # 30 minutes
    "failed_no_captions": 6 * 3600,  # 6 hours
    None: 0,                       # legacy entries — treat as success
}

# Granular version strings stored per transcript artifact.  Bump the relevant
# constant when the corresponding logic changes to enable targeted cache
# invalidation without discarding all cached transcripts.
NORMALIZATION_VERSION = "norm_v1"


def _vtt_timestamp_to_seconds(ts: str) -> float:
    """Convert ``HH:MM:SS.mmm`` to seconds."""
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(ts)


def _backoff_delay(
    attempt: int,
    *,
    base_sec: float = 1.0,
    cap_sec: float = 60.0,
    retry_after_sec: float | None = None,
) -> float:
    """Exponential backoff with full jitter (AWS style).

    Returns ``random(0, min(cap, base * 2^attempt))``, with
    *retry_after_sec* as a floor when a server-provided Retry-After
    header was parsed.
    """
    exponential = min(cap_sec, base_sec * (2 ** attempt))
    jittered = random.uniform(0, exponential)
    if retry_after_sec is not None and retry_after_sec > 0:
        return max(jittered, retry_after_sec)
    return jittered


def _parse_retry_after(response: "requests.Response | None") -> float | None:
    """Parse HTTP ``Retry-After`` header (RFC 6585).

    Handles both delta-seconds and HTTP-date formats.
    Returns seconds to wait, or ``None`` if absent/unparseable.
    """
    if response is None:
        return None
    value = response.headers.get("Retry-After", "").strip()
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    from email.utils import parsedate_to_datetime
    try:
        target = parsedate_to_datetime(value)
        delta = (target - datetime.now(timezone.utc)).total_seconds()
        return max(0.0, delta)
    except (TypeError, ValueError):
        return None


def parse_iso8601_duration(value: str) -> int:
    if not value:
        return 0
    match = re.match(r"^P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?$", value)
    if not match:
        return 0
    days = int(match.group(1) or 0)
    hours = int(match.group(2) or 0)
    minutes = int(match.group(3) or 0)
    seconds = int(match.group(4) or 0)
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


class YouTubeService:
    SEARCH_CACHE_VERSION = 4
    SEARCH_CACHE_TTL_SEC = 24 * 60 * 60  # 24 hours
    SEARCH_CACHE_EMPTY_TTL_SEC = 15 * 60  # 15 minutes for empty results
    DATA_API_MAX_PAGES = 10
    DATA_API_POOL_MULTIPLIER = 8
    DATA_API_POOL_CAP = 760
    SEARCH_VARIANTS_LIMIT = 16
    PRIMARY_VARIANT_LIMIT = 3
    # ---- pagination depth (speed vs. recall trade-off) ----
    # Each page is one extra HTTP roundtrip via the continuation API, so
    # halving the depth roughly halves the time spent in continuation
    # fetches. Empirically, pages 1-3 contain nearly all of the results we
    # end up using after ranking — pages 4-10 are high-latency low-yield.
    # Reduced from 6 → 3 (fast) and 10 → 6 (deep) for faster generation.
    HTML_MAX_PAGES = 3
    HTML_MAX_PAGES_DEEP = 6
    HTML_POOL_MULTIPLIER = 8
    HTML_POOL_CAP = 600
    # ---- worker pool sizes ----
    # These control how many HTTP requests are in flight at once. Bigger
    # pools = faster for multi-query batches, but too big triggers
    # YouTube's rate limiter. These values were bumped from 4/3 to 6/5 —
    # safely under the rate-limit threshold we've observed, while shaving
    # real time off concurrent queries.
    SEARCH_SURFACE_WORKERS = 6
    VIDEO_DETAILS_WORKERS = 5
    # ---- external fallback depth ----
    # DuckDuckGo and Bing only run when YouTube fails. Before: 8 pages each
    # (240/160 offset sweep). Cut to 4 pages because the fallback is mostly
    # about rescuing a few IDs — not a full re-crawl — and each additional
    # page there costs an HTTP roundtrip to a third-party search engine.
    DUCKDUCKGO_PAGE_OFFSETS = (0, 30, 60, 90)
    BING_FIRST_OFFSETS = (0, 20, 40, 60)
    # ---- timeouts ----
    # A hung YouTube request blocks an entire worker thread. With a short
    # timeout we fail fast and let the caller retry or fall through to an
    # alternative strategy, instead of burning 8 full seconds on a dead
    # connection. Reduced 8 → 5 for per-request, 12 → 8 for total budget.
    REQUEST_TIMEOUT_SEC = 5.0
    SEARCH_TIME_BUDGET_SEC = 8.0
    NETWORK_BACKOFF_SEC = 45.0
    NETWORK_BACKOFF_FAILURE_THRESHOLD = 3
    # ---- transcript quality gates ----
    TRANSCRIPT_MIN_CUE_COUNT = 3
    TRANSCRIPT_MIN_TOTAL_CHARS = 50
    TRANSCRIPT_MIN_COVERAGE = 0.40
    TRANSCRIPT_MAX_EMPTY_RATIO = 0.30
    # ---- circuit breaker ----
    CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5
    CIRCUIT_BREAKER_COOLDOWN_SEC = 120.0
    # ---- provenance upgrade ----
    PROVENANCE_UPGRADE_AGE_SEC = 24 * 3600  # 24 hours
    # ---- connection pool ----
    # Must be at least as large as the maximum number of concurrent HTTP
    # requests we'll ever have in flight: search workers + transcript
    # workers + video-details workers + continuation pages. Previously 24
    # was a bottleneck when everything ran at once. 48 gives us headroom.
    SESSION_POOL_SIZE = 48
    WATCH_PAGE_CACHE_TTL_SEC = 15 * 60
    CHANNEL_PAGE_CACHE_TTL_SEC = 15 * 60
    QUERY_VARIANT_EXACT_MIN_TOKENS = 3
    QUERY_VARIANT_EXACT_MAX_TOKENS = 9
    GRAPH_LIGHT_MAX_SEEDS = 2
    GRAPH_LIGHT_RELATED_PER_SEED = 5
    GRAPH_LIGHT_CHANNEL_SEEDS = 1
    GRAPH_LIGHT_CHANNEL_RESULTS = 4
    GRAPH_LIGHT_STAGE_MAX_SEC = 1.6
    GRAPH_DEEP_MAX_SEEDS = 5
    GRAPH_DEEP_RELATED_PER_SEED = 8
    GRAPH_DEEP_CHANNEL_SEEDS = 2
    GRAPH_DEEP_CHANNEL_RESULTS = 8
    GRAPH_DEEP_STAGE_MAX_SEC = 4.5
    GRAPH_RELATED_SURFACE = "youtube_related"
    GRAPH_CHANNEL_SURFACE = "youtube_channel"
    # Bumped from 4 → 6 to match the rest of the pool sizes. Graph fetches
    # (related + channel videos) run on a separate phase budget, so making
    # them more parallel is a pure win when multiple seeds are expanded.
    GRAPH_FETCH_WORKERS = 6
    SEARCH_QUERY_NOISE_TOKENS = {
        "basic",
        "basics",
        "course",
        "guide",
        "intro",
        "introduction",
        "learn",
        "lesson",
        "lessons",
        "overview",
        "video",
        "videos",
        "watch",
        "youtube",
    }
    SEARCH_SOURCE_PRIORITY = {
        "youtube_api": 1.0,
        "youtube_html": 0.94,
        "youtube_related": 0.9,
        "youtube_channel": 0.86,
        "duckduckgo_quoted": 0.88,
        "bing_quoted": 0.85,
        "duckduckgo_site": 0.82,
        "bing_site": 0.79,
    }
    STRATEGY_SUFFIX_BY_NAME = {
        "animation": "animation",
        "demo": "demo",
        "documentary": "documentary",
        "explained": "explained",
        "lecture": "lecture",
        "tutorial": "tutorial",
        "worked_example": "worked example",
    }
    SEARCH_VARIANT_VISUAL_STRATEGIES = {"action", "broll", "object", "scene"}
    SEARCH_VARIANT_SUFFIXES = (
        "worked example",
        "full lecture",
        "documentary",
        "animation",
        "explained",
        "tutorial",
        "lecture",
        "demo",
        "shorts",
    )

    def __init__(self) -> None:
        settings = get_settings()
        self.api_key = settings.youtube_api_key
        self.empty_transcript_ttl_sec = 6 * 60 * 60
        self._network_backoff_until = 0.0
        self._network_backoff_lock = threading.Lock()
        self._network_backoff_until_by_scope: dict[str, float] = {}
        self._network_failure_streak_by_scope: dict[str, int] = {}
        # ---- circuit breaker state ----
        self._circuit_open_until: dict[str, float] = {}
        self._circuit_failure_count: dict[str, int] = {}
        # ---- single-flight transcript locking ----
        self._transcript_flights: dict[str, threading.Event] = {}
        self._transcript_flights_lock = threading.Lock()
        self._page_cache_lock = threading.Lock()
        self._page_cache: dict[str, tuple[float, str]] = {}
        self.serverless_mode = bool(
            os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME") or os.getenv("K_SERVICE")
        )
        self.retrieval_debug_logging = bool(settings.retrieval_debug_logging)
        self.search_time_budget_sec = 3.5 if self.serverless_mode else self.SEARCH_TIME_BUDGET_SEC
        self.request_timeout_sec = 2.5 if self.serverless_mode else self.REQUEST_TIMEOUT_SEC

        # ---- Proxy rotation ------------------------------------------------
        self._proxy_rotator = _ProxyRotator(settings.proxy_urls)
        self._proxy_search = settings.proxy_search and self._proxy_rotator.available
        self._proxy_transcripts = settings.proxy_transcripts and self._proxy_rotator.available
        if self._proxy_rotator.available:
            logger.info(
                "Proxy rotation enabled: %d proxies, search=%s, transcripts=%s",
                len(self._proxy_rotator.all()),
                self._proxy_search,
                self._proxy_transcripts,
            )

        # ---- Transcript API with optional proxy ----------------------------
        if self._proxy_transcripts and self._proxy_rotator.available:
            first_proxy = self._proxy_rotator.all()[0]
            self.transcript_api = YouTubeTranscriptApi(proxies={"https": first_proxy, "http": first_proxy})
        else:
            self.transcript_api = YouTubeTranscriptApi()

        # ---- HTTP session with stealth headers + optional proxy ------------
        self._session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=self.SESSION_POOL_SIZE,
            pool_maxsize=self.SESSION_POOL_SIZE,
            max_retries=0,
        )
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)
        self._session.headers.update(_STEALTH_HEADERS)
        self._session.headers["User-Agent"] = _random_user_agent()
        # Pre-accept YouTube's cookie consent dialog so search/watch/channel
        # pages return real `ytInitialData` HTML instead of a consent wall
        # that lacks the marker entirely. Without this, requests from cloud
        # IPs (Railway, AWS, GCP) frequently get redirected to consent.youtube.com
        # and scraping silently returns zero rows for every query. These are
        # the same values used by yt-dlp and youtube-dl to bypass the gate.
        self._session.cookies.set("SOCS", "CAI", domain=".youtube.com")
        self._session.cookies.set(
            "CONSENT",
            "YES+cb.20210328-17-p0.en+FX+000",
            domain=".youtube.com",
        )

    def _session_get(self, url: str, *, deadline: float | None = None, **kwargs: Any) -> requests.Response:
        # Rotate User-Agent and optionally proxy per request.
        self._session.headers["User-Agent"] = _random_user_agent()
        if self._proxy_search and "proxies" not in kwargs:
            proxy = self._proxy_rotator.next()
            if proxy:
                kwargs["proxies"] = proxy
        return self._session.get(
            url,
            timeout=self._request_timeout(deadline),
            **kwargs,
        )

    def _session_post(self, url: str, *, deadline: float | None = None, **kwargs: Any) -> requests.Response:
        self._session.headers["User-Agent"] = _random_user_agent()
        if self._proxy_search and "proxies" not in kwargs:
            proxy = self._proxy_rotator.next()
            if proxy:
                kwargs["proxies"] = proxy
        return self._session.post(
            url,
            timeout=self._request_timeout(deadline),
            **kwargs,
        )

    def _cache_get_text(self, cache_key: str, *, ttl_sec: int) -> str | None:
        with self._page_cache_lock:
            cached = self._page_cache.get(cache_key)
            if cached is None:
                return None
            created_at, payload = cached
            if (time.monotonic() - created_at) >= ttl_sec:
                self._page_cache.pop(cache_key, None)
                return None
            return payload

    def _cache_set_text(self, cache_key: str, payload: str) -> None:
        with self._page_cache_lock:
            self._page_cache[cache_key] = (time.monotonic(), payload)

    def _query_preview(self, query: str, *, limit: int = 96) -> str:
        clean = self._clean_query_text(query)
        if len(clean) <= limit:
            return clean
        return f"{clean[: max(0, limit - 3)].rstrip()}..."

    def _row_source_counts(self, rows: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in rows:
            source = str(row.get("search_source") or "").strip() or "unknown"
            counts[source] = counts.get(source, 0) + 1
        return counts

    def _log_search_summary(
        self,
        *,
        query: str,
        max_results: int,
        retrieval_profile: RetrievalProfile,
        retrieval_stage: str,
        source_surface: str,
        creative_commons_only: bool,
        video_duration: str | None,
        rows: list[dict[str, Any]],
        cache_hit: bool,
        allow_external_fallbacks: bool,
        graph_profile: GraphProfile,
    ) -> None:
        should_log = self.retrieval_debug_logging or not rows
        if not should_log:
            return
        payload = {
            "query": self._query_preview(query),
            "max_results": int(max_results),
            "result_count": len(rows),
            "source_counts": self._row_source_counts(rows),
            "retrieval_profile": retrieval_profile,
            "retrieval_stage": retrieval_stage or "",
            "source_surface": source_surface or "",
            "creative_commons_only": bool(creative_commons_only),
            "video_duration": video_duration or "any",
            "cache_hit": bool(cache_hit),
            "allow_external_fallbacks": bool(allow_external_fallbacks),
            "graph_profile": graph_profile,
            "api_key_enabled": bool(self.api_key),
        }
        if rows:
            logger.info("YouTube search summary: %s", json.dumps(payload, sort_keys=True))
        else:
            logger.warning("YouTube search returned no rows: %s", json.dumps(payload, sort_keys=True))

    def search_videos(
        self,
        conn,
        query: str,
        max_results: int = 8,
        creative_commons_only: bool = False,
        video_duration: str | None = None,
        retrieval_strategy: str = "",
        retrieval_stage: str = "",
        source_surface: str = "youtube_api",
        retrieval_profile: RetrievalProfile = "deep",
        allow_external_fallbacks: bool = True,
        variant_limit: int | None = None,
        graph_profile: GraphProfile = "off",
        root_terms: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if conn is None:
            with get_conn() as local_conn:
                return self._search_videos_with_conn(
                    local_conn,
                    query=query,
                    max_results=max_results,
                    creative_commons_only=creative_commons_only,
                    video_duration=video_duration,
                    retrieval_strategy=retrieval_strategy,
                    retrieval_stage=retrieval_stage,
                    source_surface=source_surface,
                    retrieval_profile=retrieval_profile,
                    allow_external_fallbacks=allow_external_fallbacks,
                    variant_limit=variant_limit,
                    graph_profile=graph_profile,
                    root_terms=root_terms,
                )
        return self._search_videos_with_conn(
            conn,
            query=query,
            max_results=max_results,
            creative_commons_only=creative_commons_only,
            video_duration=video_duration,
            retrieval_strategy=retrieval_strategy,
            retrieval_stage=retrieval_stage,
            source_surface=source_surface,
            retrieval_profile=retrieval_profile,
            allow_external_fallbacks=allow_external_fallbacks,
            variant_limit=variant_limit,
            graph_profile=graph_profile,
            root_terms=root_terms,
        )

    def _search_videos_with_conn(
        self,
        conn,
        query: str,
        max_results: int,
        creative_commons_only: bool,
        video_duration: str | None,
        retrieval_strategy: str,
        retrieval_stage: str,
        source_surface: str,
        retrieval_profile: RetrievalProfile,
        allow_external_fallbacks: bool,
        variant_limit: int | None,
        graph_profile: GraphProfile = "off",
        root_terms: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        duration_key = video_duration or "any"
        normalized_root_terms = self._normalized_root_terms(query=query, root_terms=root_terms)
        key = _cache_key(
            str(self.SEARCH_CACHE_VERSION),
            query,
            str(max_results),
            str(creative_commons_only),
            duration_key,
            retrieval_strategy or "",
            retrieval_stage or "",
            source_surface or "",
            retrieval_profile,
            str(bool(allow_external_fallbacks)),
            str(variant_limit or 0),
            graph_profile,
            "||".join(normalized_root_terms),
        )
        cached = fetch_one(conn, "SELECT response_json, created_at FROM search_cache WHERE cache_key = ?", (key,))
        if cached:
            try:
                payload = json.loads(cached["response_json"])
            except (TypeError, json.JSONDecodeError):
                payload = []
            if isinstance(payload, list):
                cache_fresh = self._search_cache_is_fresh(
                    created_at=cached.get("created_at"),
                    is_empty=len(payload) == 0,
                )
                if cache_fresh:
                    finalized_rows = self._finalize_search_rows(
                        payload,
                        query=query,
                        max_results=max_results,
                        video_duration=video_duration,
                    )
                    self._log_search_summary(
                        query=query,
                        max_results=max_results,
                        retrieval_profile=retrieval_profile,
                        retrieval_stage=retrieval_stage,
                        source_surface=source_surface,
                        creative_commons_only=creative_commons_only,
                        video_duration=video_duration,
                        rows=finalized_rows,
                        cache_hit=True,
                        allow_external_fallbacks=allow_external_fallbacks,
                        graph_profile=graph_profile,
                    )
                    return finalized_rows

        deadline = time.monotonic() + self.search_time_budget_sec
        videos: list[dict[str, Any]] = []
        fast_non_api_enabled = not creative_commons_only
        if retrieval_profile == "bootstrap":
            if creative_commons_only and self.api_key:
                videos = self._search_via_data_api(
                    query=query,
                    max_results=max_results,
                    creative_commons_only=True,
                    video_duration=video_duration,
                    retrieval_strategy=retrieval_strategy,
                    retrieval_stage=retrieval_stage,
                    source_surface="youtube_api",
                    deadline=deadline,
                )
            else:
                videos = self._search_without_data_api(
                    query=query,
                    max_results=max_results,
                    creative_commons_only=creative_commons_only,
                    video_duration=video_duration,
                    retrieval_strategy=retrieval_strategy,
                    retrieval_stage=retrieval_stage,
                    source_surface="youtube_html",
                    deadline=deadline,
                    include_external_fallbacks=allow_external_fallbacks,
                    variant_limit=variant_limit or 1,
                    skip_primary_variants=False,
                    retrieval_profile=retrieval_profile,
                    graph_profile=graph_profile,
                    root_terms=normalized_root_terms,
                )
            videos = self._finalize_search_rows(
                videos,
                query=query,
                max_results=max_results,
                video_duration=video_duration,
            )
            upsert(
                conn,
                "search_cache",
                {
                    "cache_key": key,
                    "response_json": dumps_json(videos),
                    "created_at": now_iso(),
                },
                pk="cache_key",
            )
            self._log_search_summary(
                query=query,
                max_results=max_results,
                retrieval_profile=retrieval_profile,
                retrieval_stage=retrieval_stage,
                source_surface=source_surface,
                creative_commons_only=creative_commons_only,
                video_duration=video_duration,
                rows=videos,
                cache_hit=False,
                allow_external_fallbacks=allow_external_fallbacks,
                graph_profile=graph_profile,
            )
            return videos

        futures: list[tuple[str, Any]] = []
        max_workers = 1 + int(bool(self.api_key)) + int(fast_non_api_enabled)
        max_workers = max(1, min(3, max_workers))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            if self.api_key:
                futures.append(
                    (
                        "youtube_api",
                        executor.submit(
                            self._search_via_data_api,
                            query,
                            max_results,
                            creative_commons_only,
                            video_duration,
                            retrieval_strategy,
                            retrieval_stage,
                            source_surface,
                            deadline,
                            normalized_root_terms,
                        ),
                    )
                )
            if fast_non_api_enabled:
                futures.append(
                    (
                        "youtube_html_primary",
                        executor.submit(
                            self._search_without_data_api,
                            query,
                            max_results,
                            False,
                            video_duration,
                            retrieval_strategy,
                            retrieval_stage,
                            source_surface,
                            deadline,
                            False,
                            self.PRIMARY_VARIANT_LIMIT,
                            False,
                            retrieval_profile,
                            "off",
                            normalized_root_terms,
                        ),
                    )
                )

            surface_results: dict[str, list[dict[str, Any]]] = {}
            future_to_surface = {future: surface for surface, future in futures}
            for future in as_completed(future_to_surface):
                surface = future_to_surface[future]
                try:
                    rows = future.result()
                except YouTubeApiRequestError:
                    rows = []
                except Exception:
                    rows = []
                surface_results[surface] = rows

        for surface, _future in futures:
            rows = surface_results.get(surface) or []
            if rows:
                videos = self._merge_unique_videos(videos, rows, None)

        if not creative_commons_only and len(videos) < max_results:
            expanded = self._search_without_data_api(
                query=query,
                max_results=max_results,
                creative_commons_only=False,
                video_duration=video_duration,
                retrieval_strategy=retrieval_strategy,
                retrieval_stage=retrieval_stage,
                source_surface=source_surface,
                deadline=deadline,
                include_external_fallbacks=False,
                variant_limit=self.SEARCH_VARIANTS_LIMIT,
                skip_primary_variants=True,
                retrieval_profile=retrieval_profile,
                graph_profile="off",
                root_terms=normalized_root_terms,
            )
            videos = self._merge_unique_videos(videos, expanded, None)

        if (
            not creative_commons_only
            and graph_profile != "off"
            and self._should_expand_graph(
                rows=videos,
                query=query,
                max_results=max_results,
                video_duration=video_duration,
            )
        ):
            graph_rows = self._expand_videos_via_youtube_graph(
                seed_rows=videos,
                query=query,
                max_results=max_results,
                video_duration=video_duration,
                retrieval_strategy=retrieval_strategy,
                retrieval_stage=retrieval_stage,
                deadline=deadline,
                graph_profile=graph_profile,
                root_terms=normalized_root_terms,
            )
            videos = self._merge_unique_videos(videos, graph_rows, None)

        if not creative_commons_only and allow_external_fallbacks and len(videos) < max_results:
            external = self._search_external_fallbacks(
                query=query,
                max_results=max_results,
                video_duration=video_duration,
                retrieval_strategy=retrieval_strategy,
                retrieval_stage=retrieval_stage,
                source_surface=source_surface,
                retrieval_profile=retrieval_profile,
                deadline=deadline,
                variant_limit=variant_limit,
            )
            videos = self._merge_unique_videos(videos, external, None)

        if not videos and self.api_key and not creative_commons_only:
            # Secondary recovery in case primary pass got blocked by transient API errors.
            videos = self._search_without_data_api(
                query=query,
                max_results=max_results,
                creative_commons_only=False,
                video_duration=video_duration,
                retrieval_strategy=retrieval_strategy,
                retrieval_stage=retrieval_stage,
                source_surface=source_surface,
                deadline=deadline,
                include_external_fallbacks=True,
                retrieval_profile=retrieval_profile,
                graph_profile=graph_profile,
                root_terms=normalized_root_terms,
            )
        videos = self._finalize_search_rows(
            videos,
            query=query,
            max_results=max_results,
            video_duration=video_duration,
        )

        upsert(
            conn,
            "search_cache",
            {
                "cache_key": key,
                "response_json": dumps_json(videos),
                "created_at": now_iso(),
            },
            pk="cache_key",
        )
        self._log_search_summary(
            query=query,
            max_results=max_results,
            retrieval_profile=retrieval_profile,
            retrieval_stage=retrieval_stage,
            source_surface=source_surface,
            creative_commons_only=creative_commons_only,
            video_duration=video_duration,
            rows=videos,
            cache_hit=False,
            allow_external_fallbacks=allow_external_fallbacks,
            graph_profile=graph_profile,
        )
        return videos

    def _search_via_data_api(
        self,
        query: str,
        max_results: int,
        creative_commons_only: bool,
        video_duration: str | None,
        retrieval_strategy: str,
        retrieval_stage: str,
        source_surface: str,
        deadline: float | None = None,
        root_terms: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        target_pool = max(max_results, min(self.DATA_API_POOL_CAP, max_results * self.DATA_API_POOL_MULTIPLIER))
        per_page = 50
        items: list[dict[str, Any]] = []
        seen_video_ids: set[str] = set()
        next_page_token: str | None = None

        for page_idx in range(self.DATA_API_MAX_PAGES):
            if self._deadline_exceeded(deadline) or self._network_backoff_active("youtube_api"):
                break
            params = {
                "key": self.api_key,
                "part": "snippet",
                "q": query,
                "type": "video",
                "maxResults": per_page,
                "safeSearch": "moderate",
                "videoEmbeddable": "true",
                "relevanceLanguage": "en",
            }
            if creative_commons_only:
                params["videoLicense"] = "creativeCommon"
            if video_duration in {"short", "medium", "long"}:
                params["videoDuration"] = video_duration
            if next_page_token:
                params["pageToken"] = next_page_token

            try:
                resp = self._session_get(
                    "https://www.googleapis.com/youtube/v3/search",
                    params=params,
                    deadline=deadline,
                )
                resp.raise_for_status()
                data = resp.json()
                self._note_request_success("youtube_api")
            except requests.RequestException as exc:
                self._note_request_failure(exc, scope="youtube_api")
                if page_idx == 0:
                    status = getattr(getattr(exc, "response", None), "status_code", None)
                    if status == 403:
                        raise YouTubeApiRequestError(
                            "YouTube Data API request was rejected (403). Check YOUTUBE_API_KEY, API restrictions, and quota."
                        ) from exc
                    raise YouTubeApiRequestError(
                        f"YouTube Data API request failed{f' ({status})' if status else ''}."
                    ) from exc
                break

            page_items = data.get("items", [])
            if not isinstance(page_items, list):
                page_items = []

            for item in page_items:
                video_id = item.get("id", {}).get("videoId")
                if not video_id or video_id in seen_video_ids:
                    continue
                seen_video_ids.add(video_id)
                items.append(item)
                if len(items) >= target_pool:
                    break

            if len(items) >= target_pool:
                break

            next_page_token = data.get("nextPageToken")
            if not isinstance(next_page_token, str) or not next_page_token:
                break

        ids = [item.get("id", {}).get("videoId") for item in items if item.get("id", {}).get("videoId")]
        details = self._video_details([str(video_id) for video_id in ids], deadline=deadline)

        videos: list[dict[str, Any]] = []
        for item in items:
            video_id = item.get("id", {}).get("videoId")
            if not video_id:
                continue
            detail = details.get(video_id, {})
            is_cc = detail.get("license") == "creativeCommon"
            if creative_commons_only and not is_cc:
                continue
            snippet = item.get("snippet", {})
            videos.append(
                {
                    "id": video_id,
                    "title": detail.get("title") or snippet.get("title", "Untitled"),
                    "channel_id": detail.get("channel_id") or snippet.get("channelId", ""),
                    "channel_title": detail.get("channel_title") or snippet.get("channelTitle", ""),
                    "description": detail.get("description") or snippet.get("description", ""),
                    "duration_sec": detail.get("duration_sec", 0),
                    "view_count": detail.get("view_count", 0),
                    "published_at": detail.get("published_at") or snippet.get("publishedAt", ""),
                    "is_creative_commons": bool(is_cc),
                    "search_source": "youtube_api",
                    "query_strategy": retrieval_strategy or "",
                    "query_stage": retrieval_stage or "",
                    "search_query": query,
                    "discovery_path": "search:youtube_api",
                    "seed_video_id": "",
                    "seed_channel_id": item.get("snippet", {}).get("channelId", ""),
                    "crawl_depth": 0,
                }
            )
        return self._finalize_search_rows(
            videos,
            query=query,
            max_results=max_results,
            video_duration=video_duration,
        )

    def _search_without_data_api(
        self,
        query: str,
        max_results: int,
        creative_commons_only: bool,
        video_duration: str | None,
        retrieval_strategy: str,
        retrieval_stage: str,
        source_surface: str,
        deadline: float | None = None,
        include_external_fallbacks: bool = True,
        variant_limit: int | None = None,
        skip_primary_variants: bool = False,
        retrieval_profile: RetrievalProfile = "deep",
        graph_profile: GraphProfile = "off",
        root_terms: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        # License cannot be reliably verified without the Data API.
        if creative_commons_only:
            return []
        if self._deadline_exceeded(deadline):
            return []

        target_pool = max(max_results, min(self.HTML_POOL_CAP, max_results * self.HTML_POOL_MULTIPLIER))
        results: list[dict[str, Any]] = []
        query_variants = self._build_search_query_variants(
            query=query,
            video_duration=video_duration,
            source_surface=source_surface,
            retrieval_profile=retrieval_profile,
            retrieval_strategy=retrieval_strategy,
        )
        if skip_primary_variants and len(query_variants) > self.PRIMARY_VARIANT_LIMIT:
            query_variants = query_variants[self.PRIMARY_VARIANT_LIMIT :]
        if variant_limit and variant_limit > 0:
            query_variants = query_variants[:variant_limit]
        if not query_variants:
            return []

        html_futures: dict[Any, int] = {}
        max_html_workers = max(1, min(self.SEARCH_SURFACE_WORKERS, len(query_variants)))
        with ThreadPoolExecutor(max_workers=max_html_workers) as executor:
            for variant_idx, variant in enumerate(query_variants):
                search_query = str(variant.get("query") or "").strip()
                variant_surface = str(variant.get("surface") or source_surface or "youtube_html")
                if not search_query:
                    continue
                html_futures[
                    executor.submit(
                        self._search_variant_via_html,
                        search_query,
                        variant_surface,
                        target_pool,
                        video_duration,
                        retrieval_strategy,
                        retrieval_stage,
                        deadline,
                        retrieval_profile,
                    )
                ] = variant_idx
            ordered_variant_rows: list[list[dict[str, Any]]] = [[] for _ in query_variants]
            for future in as_completed(html_futures):
                if self._deadline_exceeded(deadline):
                    break
                try:
                    rows = future.result()
                except Exception:
                    rows = []
                if not rows:
                    continue
                ordered_variant_rows[html_futures[future]] = rows
        for rows in ordered_variant_rows:
            if rows:
                results = self._merge_unique_videos(results, rows, None)
        results = self._finalize_search_rows(
            results,
            query=query,
            max_results=target_pool,
            video_duration=video_duration,
        )

        normalized_root_terms = self._normalized_root_terms(query=query, root_terms=root_terms)
        if (
            graph_profile != "off"
            and self._should_expand_graph(
                rows=results,
                query=query,
                max_results=max_results,
                video_duration=video_duration,
            )
        ):
            graph_rows = self._expand_videos_via_youtube_graph(
                seed_rows=results,
                query=query,
                max_results=target_pool,
                video_duration=video_duration,
                retrieval_strategy=retrieval_strategy,
                retrieval_stage=retrieval_stage,
                deadline=deadline,
                graph_profile=graph_profile,
                root_terms=normalized_root_terms,
            )
            if graph_rows:
                results = self._merge_unique_videos(results, graph_rows, None)
                results = self._finalize_search_rows(
                    results,
                    query=query,
                    max_results=target_pool,
                    video_duration=video_duration,
                )

        if include_external_fallbacks and len(results) < max_results:
            external_rows = self._search_external_fallbacks(
                query=query,
                max_results=target_pool,
                video_duration=video_duration,
                retrieval_strategy=retrieval_strategy,
                retrieval_stage=retrieval_stage,
                source_surface=source_surface,
                retrieval_profile=retrieval_profile,
                deadline=deadline,
                variant_limit=variant_limit,
            )
            if external_rows:
                results = self._merge_unique_videos(results, external_rows, None)
                results = self._finalize_search_rows(
                    results,
                    query=query,
                    max_results=target_pool,
                    video_duration=video_duration,
                )

        return self._finalize_search_rows(
            results,
            query=query,
            max_results=max_results,
            video_duration=video_duration,
        )

    def _search_variant_via_html(
        self,
        search_query: str,
        variant_surface: str,
        target_pool: int,
        video_duration: str | None,
        retrieval_strategy: str,
        retrieval_stage: str,
        deadline: float | None,
        retrieval_profile: RetrievalProfile = "deep",
    ) -> list[dict[str, Any]]:
        if self._deadline_exceeded(deadline) or self._network_backoff_active("youtube_html"):
            return []
        html = self._fetch_search_html(search_query, deadline=deadline)
        if not html:
            return []

        rows: list[dict[str, Any]] = []
        initial_data = self._extract_yt_initial_data(html)
        parsed = self._extract_videos_from_search_data(
            initial_data,
            max_results=target_pool,
            video_duration=video_duration,
        )
        if not parsed:
            parsed = self._extract_videos_from_search_html(
                html,
                max_results=target_pool,
                video_duration=video_duration,
            )
        self._annotate_search_rows(
            parsed,
            search_source=variant_surface,
            retrieval_strategy=retrieval_strategy,
            retrieval_stage=retrieval_stage,
            search_query=search_query,
        )
        rows = self._merge_unique_videos(rows, parsed, None)
        if len(rows) >= target_pool:
            return self._finalize_search_rows(
                rows,
                query=search_query,
                max_results=target_pool,
                video_duration=video_duration,
            )

        # Last-ditch fallback strategy: scan the HTML directly for anything
        # that looks like a `"videoId":"xxxxxxxxxxx"` pair. This is much less
        # accurate than walking ytInitialData — we get no titles, durations,
        # or channels, and we can pick up unrelated mentions (e.g. a video
        # embedded in a description). But it often recovers at least *some*
        # results when YouTube changes its markup in a way that breaks the
        # primary path. The downstream scorer will deprioritise these rows
        # because their metadata is empty.
        ids: list[str] = []
        seen_ids: set[str] = set()
        for match in _VIDEO_ID_IN_JSON_REGEX.finditer(html):
            video_id = match.group(1)
            # Double-check with the canonical validator; the regex is already
            # strict but this keeps the contract explicit and consistent with
            # every other code path.
            if not _is_valid_youtube_video_id(video_id) or video_id in seen_ids:
                continue
            seen_ids.add(video_id)
            ids.append(video_id)
            if len(ids) >= target_pool:
                break
        if ids:
            fallback_rows = [row for row in (self._fallback_video_row(vid) for vid in ids) if row]
            self._annotate_search_rows(
                fallback_rows,
                search_source=variant_surface,
                retrieval_strategy=retrieval_strategy,
                retrieval_stage=retrieval_stage,
                search_query=search_query,
            )
            rows = self._merge_unique_videos(rows, fallback_rows, None)
            if len(rows) >= target_pool:
                return self._finalize_search_rows(
                    rows,
                    query=search_query,
                    max_results=target_pool,
                    video_duration=video_duration,
                )

        if not initial_data:
            return self._finalize_search_rows(
                rows,
                query=search_query,
                max_results=target_pool,
                video_duration=video_duration,
            )

        api_key, client_version = self._extract_innertube_config(html)
        continuation_token = self._extract_search_continuation_token(initial_data)
        seen_tokens: set[str] = set()
        while (
            continuation_token
            and len(rows) < target_pool
            and len(seen_tokens) < max(1, (self.HTML_MAX_PAGES_DEEP if retrieval_profile == "deep" else self.HTML_MAX_PAGES) - 1)
            and not self._deadline_exceeded(deadline)
            and not self._network_backoff_active("youtube_html")
        ):
            if continuation_token in seen_tokens:
                break
            seen_tokens.add(continuation_token)
            continuation_data = self._fetch_search_continuation(
                continuation_token=continuation_token,
                innertube_api_key=api_key,
                innertube_client_version=client_version,
                deadline=deadline,
            )
            if not continuation_data:
                break
            continuation_rows = self._extract_videos_from_search_data(
                continuation_data,
                max_results=target_pool - len(rows),
                video_duration=video_duration,
            )
            self._annotate_search_rows(
                continuation_rows,
                search_source=variant_surface,
                retrieval_strategy=retrieval_strategy,
                retrieval_stage=retrieval_stage,
                search_query=search_query,
            )
            rows = self._merge_unique_videos(rows, continuation_rows, None)
            continuation_token = self._extract_search_continuation_token(continuation_data)
        return self._finalize_search_rows(
            rows,
            query=search_query,
            max_results=target_pool,
            video_duration=video_duration,
        )

    def _annotate_search_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        search_source: str,
        retrieval_strategy: str,
        retrieval_stage: str,
        search_query: str,
    ) -> None:
        for row in rows:
            row["search_source"] = search_source
            row["query_strategy"] = retrieval_strategy or ""
            row["query_stage"] = retrieval_stage or ""
            row["search_query"] = search_query
            if not str(row.get("discovery_path") or "").strip():
                row["discovery_path"] = f"search:{search_source}"
            row["seed_video_id"] = str(row.get("seed_video_id") or "")
            row["seed_channel_id"] = str(row.get("seed_channel_id") or row.get("channel_id") or "")
            row["crawl_depth"] = max(0, int(row.get("crawl_depth") or 0))

    def _normalized_root_terms(self, *, query: str, root_terms: list[str] | None) -> list[str]:
        values: list[str] = []
        seen: set[str] = set()
        for raw in [query, self._strip_search_variant_suffix(query), *(root_terms or [])]:
            clean = self._clean_query_text(str(raw or ""))
            if not clean:
                continue
            key = clean.lower()
            if key in seen:
                continue
            seen.add(key)
            values.append(clean)
        return values[:8]

    def _graph_profile_settings(self, graph_profile: GraphProfile) -> dict[str, Any]:
        if graph_profile == "light":
            return {
                "seed_limit": self.GRAPH_LIGHT_MAX_SEEDS,
                "related_per_seed": self.GRAPH_LIGHT_RELATED_PER_SEED,
                "channel_seed_limit": self.GRAPH_LIGHT_CHANNEL_SEEDS,
                "channel_results": self.GRAPH_LIGHT_CHANNEL_RESULTS,
                "phase_max_sec": self.GRAPH_LIGHT_STAGE_MAX_SEC,
                "seed_min_score": 0.46,
            }
        if graph_profile == "deep":
            return {
                "seed_limit": self.GRAPH_DEEP_MAX_SEEDS,
                "related_per_seed": self.GRAPH_DEEP_RELATED_PER_SEED,
                "channel_seed_limit": self.GRAPH_DEEP_CHANNEL_SEEDS,
                "channel_results": self.GRAPH_DEEP_CHANNEL_RESULTS,
                "phase_max_sec": self.GRAPH_DEEP_STAGE_MAX_SEC,
                "seed_min_score": 0.4,
            }
        return {}

    def _phase_deadline(
        self,
        deadline: float | None,
        *,
        share: float,
        max_sec: float,
    ) -> float | None:
        if max_sec <= 0:
            return deadline
        now = time.monotonic()
        if deadline is None:
            return now + max_sec
        remaining = deadline - now
        if remaining <= 0:
            return deadline
        budget = min(max_sec, max(0.25, remaining * share))
        return min(deadline, now + budget)

    def _strong_direct_inventory_count(
        self,
        rows: list[dict[str, Any]],
        *,
        query: str,
        video_duration: str | None,
    ) -> int:
        count = 0
        for row in rows:
            if str(row.get("search_source") or "") not in {"youtube_api", "youtube_html"}:
                continue
            if self._search_result_score(row, query=query, video_duration=video_duration) >= 0.62:
                count += 1
        return count

    def _should_expand_graph(
        self,
        *,
        rows: list[dict[str, Any]],
        query: str,
        max_results: int,
        video_duration: str | None,
    ) -> bool:
        if not rows:
            return False
        direct_rows = [row for row in rows if str(row.get("search_source") or "") in {"youtube_api", "youtube_html"}]
        if len(direct_rows) < max_results:
            return True
        strong_direct = self._strong_direct_inventory_count(
            direct_rows,
            query=query,
            video_duration=video_duration,
        )
        return strong_direct < min(max_results, 4)

    def _search_external_fallbacks(
        self,
        *,
        query: str,
        max_results: int,
        video_duration: str | None,
        retrieval_strategy: str,
        retrieval_stage: str,
        source_surface: str,
        retrieval_profile: RetrievalProfile,
        deadline: float | None,
        variant_limit: int | None,
    ) -> list[dict[str, Any]]:
        query_variants = self._build_search_query_variants(
            query=query,
            video_duration=video_duration,
            source_surface=source_surface,
            retrieval_profile=retrieval_profile,
            retrieval_strategy=retrieval_strategy,
        )
        if variant_limit and variant_limit > 0:
            query_variants = query_variants[:variant_limit]
        if not query_variants:
            return []
        external_variants = self._build_external_query_variants(
            query_variants=query_variants,
            retrieval_strategy=retrieval_strategy,
            retrieval_profile=retrieval_profile,
        )
        if not external_variants:
            return []
        results: list[dict[str, Any]] = []
        per_variant_budget = max(3, min(8, max_results + 2))
        max_external_workers = max(1, min(self.SEARCH_SURFACE_WORKERS, len(external_variants) * 2))
        with ThreadPoolExecutor(max_workers=max_external_workers) as executor:
            future_map: dict[Any, tuple[int, str, str]] = {}
            for variant_idx, variant in enumerate(external_variants):
                search_query = str(variant.get("query") or "").strip()
                if not search_query:
                    continue
                future_map[
                    executor.submit(
                        self._search_via_duckduckgo,
                        search_query,
                        per_variant_budget,
                        deadline,
                    )
                ] = (variant_idx, "duckduckgo", search_query)
                future_map[
                    executor.submit(
                        self._search_via_bing,
                        search_query,
                        per_variant_budget,
                        deadline,
                    )
                ] = (variant_idx, "bing", search_query)
            ordered_external_ids: list[list[str]] = [[] for _ in range(len(external_variants) * 2)]
            for future in as_completed(future_map):
                if self._deadline_exceeded(deadline):
                    break
                variant_idx, engine, search_query = future_map[future]
                try:
                    ids = future.result()
                except Exception:
                    ids = []
                if not ids:
                    continue
                if engine == "duckduckgo":
                    order_idx = variant_idx * 2
                else:
                    order_idx = variant_idx * 2 + 1
                ordered_external_ids[order_idx] = ids
        all_video_ids = list(
            dict.fromkeys(
                video_id
                for ids in ordered_external_ids
                for video_id in ids
                if str(video_id or "").strip()
            )
        )
        details_by_id = self._video_details(all_video_ids, deadline=deadline) if all_video_ids and self.api_key else {}
        ordered_external_rows: list[list[dict[str, Any]]] = [[] for _ in range(len(external_variants) * 2)]
        for order_idx, ids in enumerate(ordered_external_ids):
            if not ids:
                continue
            search_query = str(external_variants[order_idx // 2].get("query") or query)
            if order_idx % 2 == 0:
                surface = "duckduckgo_quoted" if '"' in search_query else "duckduckgo_site"
            else:
                surface = "bing_quoted" if '"' in search_query else "bing_site"
            fallback_rows = self._build_fallback_rows(
                ids,
                deadline=deadline,
                details_by_id=details_by_id,
            )
            self._annotate_search_rows(
                fallback_rows,
                search_source=surface,
                retrieval_strategy=retrieval_strategy,
                retrieval_stage=retrieval_stage,
                search_query=search_query,
            )
            ordered_external_rows[order_idx] = fallback_rows
        for rows in ordered_external_rows:
            if rows:
                results = self._merge_unique_videos(results, rows, None)
        return self._finalize_search_rows(
            results,
            query=query,
            max_results=max_results,
            video_duration=video_duration,
        )

    def _graph_seed_score(
        self,
        row: dict[str, Any],
        *,
        query: str,
        root_terms: list[str],
        video_duration: str | None,
    ) -> float:
        metadata = " ".join(
            part
            for part in [
                str(row.get("title") or ""),
                str(row.get("description") or ""),
                str(row.get("channel_title") or ""),
            ]
            if part
        ).strip()
        root_tokens: list[str] = []
        for term in root_terms:
            for token in self._search_query_tokens(term):
                if token not in root_tokens:
                    root_tokens.append(token)
        title_overlap = self._query_token_overlap(str(row.get("title") or ""), root_tokens)
        metadata_overlap = self._query_token_overlap(metadata, root_tokens)
        phrase_hit = 1.0 if any(self._contains_query_phrase(metadata, term) for term in root_terms) else 0.0
        duration_score = self._duration_match_score(int(row.get("duration_sec") or 0), video_duration)
        source_score = self._search_source_priority(str(row.get("search_source") or ""))
        quality_score = self._video_row_quality(row)
        return float(
            0.34 * phrase_hit
            + 0.24 * title_overlap
            + 0.14 * metadata_overlap
            + 0.12 * duration_score
            + 0.08 * source_score
            + 0.08 * quality_score
        )

    def _graph_candidate_anchor_metrics(
        self,
        row: dict[str, Any],
        *,
        query: str,
        root_terms: list[str],
        seed_row: dict[str, Any],
    ) -> dict[str, float]:
        metadata = " ".join(
            part
            for part in [
                str(row.get("title") or ""),
                str(row.get("description") or ""),
                str(row.get("channel_title") or ""),
            ]
            if part
        ).strip()
        title = str(row.get("title") or "")
        root_tokens: list[str] = []
        for term in root_terms:
            for token in self._search_query_tokens(term):
                if token not in root_tokens:
                    root_tokens.append(token)
        seed_tokens = self._search_query_tokens(str(seed_row.get("title") or ""))
        title_overlap = self._query_token_overlap(title, root_tokens)
        metadata_overlap = self._query_token_overlap(metadata, root_tokens)
        phrase_hit = 1.0 if any(self._contains_query_phrase(metadata, term) for term in root_terms) else 0.0
        seed_overlap = self._query_token_overlap(metadata, seed_tokens)
        same_channel = 1.0 if (
            str(row.get("channel_id") or "").strip()
            and str(row.get("channel_id") or "").strip() == str(seed_row.get("channel_id") or "").strip()
        ) else 0.0
        return {
            "title_overlap": float(title_overlap),
            "metadata_overlap": float(metadata_overlap),
            "phrase_hit": float(phrase_hit),
            "seed_overlap": float(seed_overlap),
            "same_channel": float(same_channel),
        }

    def _graph_candidate_is_anchored(
        self,
        row: dict[str, Any],
        *,
        query: str,
        root_terms: list[str],
        seed_row: dict[str, Any],
        require_same_channel: bool,
    ) -> bool:
        metrics = self._graph_candidate_anchor_metrics(
            row,
            query=query,
            root_terms=root_terms,
            seed_row=seed_row,
        )
        if require_same_channel and metrics["same_channel"] <= 0.0:
            return False
        if metrics["phrase_hit"] >= 1.0:
            return True
        if metrics["title_overlap"] >= 0.5:
            return True
        if metrics["metadata_overlap"] >= 0.68:
            return True
        if metrics["same_channel"] > 0.0 and (
            metrics["metadata_overlap"] >= 0.45 or metrics["title_overlap"] >= 0.42
        ):
            return True
        return metrics["seed_overlap"] >= 0.55 and metrics["metadata_overlap"] >= 0.4

    def _expand_videos_via_youtube_graph(
        self,
        *,
        seed_rows: list[dict[str, Any]],
        query: str,
        max_results: int,
        video_duration: str | None,
        retrieval_strategy: str,
        retrieval_stage: str,
        deadline: float | None,
        graph_profile: GraphProfile,
        root_terms: list[str],
    ) -> list[dict[str, Any]]:
        settings = self._graph_profile_settings(graph_profile)
        if not settings or not seed_rows or self._deadline_exceeded(deadline):
            return []
        phase_deadline = self._phase_deadline(
            deadline,
            share=0.45,
            max_sec=float(settings.get("phase_max_sec") or 0.0),
        )
        if self._deadline_exceeded(phase_deadline):
            return []

        scored_seeds: list[tuple[float, dict[str, Any]]] = []
        for row in self._finalize_search_rows(
            seed_rows,
            query=query,
            max_results=max(len(seed_rows), int(settings.get("seed_limit") or 1) * 2),
            video_duration=video_duration,
        ):
            score = self._graph_seed_score(
                row,
                query=query,
                root_terms=root_terms,
                video_duration=video_duration,
            )
            if score >= float(settings.get("seed_min_score") or 0.0):
                scored_seeds.append((score, row))
        if not scored_seeds:
            fallback_seed = seed_rows[0]
            scored_seeds.append(
                (
                    self._graph_seed_score(
                        fallback_seed,
                        query=query,
                        root_terms=root_terms,
                        video_duration=video_duration,
                    ),
                    fallback_seed,
                )
            )
        scored_seeds.sort(key=lambda item: item[0], reverse=True)

        selected_seeds: list[tuple[float, dict[str, Any]]] = []
        seen_seed_ids: set[str] = set()
        for score, seed_row in scored_seeds:
            video_id = str(seed_row.get("id") or "").strip()
            if not video_id or video_id in seen_seed_ids:
                continue
            seen_seed_ids.add(video_id)
            selected_seeds.append((score, seed_row))
            if len(selected_seeds) >= int(settings.get("seed_limit") or 1):
                break

        results: list[dict[str, Any]] = []
        related_deadline = self._phase_deadline(
            phase_deadline,
            share=0.7,
            max_sec=max(0.6, float(settings.get("phase_max_sec") or 0.0) * 0.7),
        )
        best_channel_seed: dict[str, tuple[float, dict[str, Any]]] = {}
        for score, seed_row in selected_seeds:
            channel_id = str(seed_row.get("channel_id") or "").strip()
            if channel_id:
                best_existing = best_channel_seed.get(channel_id)
                if best_existing is None or score > best_existing[0]:
                    best_channel_seed[channel_id] = (score, seed_row)
        related_seed_rows = [
            seed_row
            for _score, seed_row in selected_seeds
            if str(seed_row.get("id") or "").strip()
        ]
        if related_seed_rows and not self._deadline_exceeded(related_deadline):
            max_workers = max(1, min(self.GRAPH_FETCH_WORKERS, len(related_seed_rows)))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {
                    executor.submit(
                        self._graph_related_rows_for_seed,
                        seed_row=seed_row,
                        query=query,
                        root_terms=root_terms,
                        video_duration=video_duration,
                        retrieval_strategy=retrieval_strategy,
                        retrieval_stage=retrieval_stage,
                        deadline=related_deadline,
                        related_per_seed=int(settings.get("related_per_seed") or 0),
                    ): seed_row
                    for seed_row in related_seed_rows
                }
                for future in as_completed(future_map):
                    if self._deadline_exceeded(related_deadline):
                        break
                    try:
                        related_rows = future.result()
                    except Exception:
                        related_rows = []
                    if related_rows:
                        results = self._merge_unique_videos(results, related_rows, None)

        strong_inventory = self._strong_direct_inventory_count(
            self._merge_unique_videos(seed_rows, results, None),
            query=query,
            video_duration=video_duration,
        )
        if strong_inventory >= min(max_results, 4):
            return results

        channel_deadline = self._phase_deadline(
            phase_deadline,
            share=0.3,
            max_sec=max(0.45, float(settings.get("phase_max_sec") or 0.0) * 0.35),
        )
        channel_candidates = sorted(best_channel_seed.values(), key=lambda item: item[0], reverse=True)
        selected_channel_seeds = [
            seed_row
            for score, seed_row in channel_candidates[: int(settings.get("channel_seed_limit") or 0)]
            if score >= max(0.45, float(settings.get("seed_min_score") or 0.0))
            and str(seed_row.get("channel_id") or "").strip()
        ]
        if selected_channel_seeds and not self._deadline_exceeded(channel_deadline):
            max_workers = max(1, min(self.GRAPH_FETCH_WORKERS, len(selected_channel_seeds)))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {
                    executor.submit(
                        self._graph_channel_rows_for_seed,
                        seed_row=seed_row,
                        query=query,
                        root_terms=root_terms,
                        video_duration=video_duration,
                        retrieval_strategy=retrieval_strategy,
                        retrieval_stage=retrieval_stage,
                        deadline=channel_deadline,
                        channel_results=int(settings.get("channel_results") or 0),
                    ): seed_row
                    for seed_row in selected_channel_seeds
                }
                for future in as_completed(future_map):
                    if self._deadline_exceeded(channel_deadline):
                        break
                    try:
                        channel_rows = future.result()
                    except Exception:
                        channel_rows = []
                    if channel_rows:
                        results = self._merge_unique_videos(results, channel_rows, None)
        return results

    def _graph_related_rows_for_seed(
        self,
        *,
        seed_row: dict[str, Any],
        query: str,
        root_terms: list[str],
        video_duration: str | None,
        retrieval_strategy: str,
        retrieval_stage: str,
        deadline: float | None,
        related_per_seed: int,
    ) -> list[dict[str, Any]]:
        if related_per_seed <= 0 or self._deadline_exceeded(deadline):
            return []
        seed_video_id = str(seed_row.get("id") or "").strip()
        if not seed_video_id:
            return []
        watch_html = self._fetch_watch_html(seed_video_id, deadline=deadline)
        if not watch_html:
            return []
        related_rows = self._extract_related_videos_from_watch_html(
            watch_html,
            max_results=related_per_seed,
            video_duration=video_duration,
        )
        anchored_rows: list[dict[str, Any]] = []
        seed_channel_id = str(seed_row.get("channel_id") or "")
        for row in related_rows:
            candidate_id = str(row.get("id") or "").strip()
            if not candidate_id or candidate_id == seed_video_id:
                continue
            if not self._graph_candidate_is_anchored(
                row,
                query=query,
                root_terms=root_terms,
                seed_row=seed_row,
                require_same_channel=False,
            ):
                continue
            row["search_source"] = self.GRAPH_RELATED_SURFACE
            row["query_strategy"] = retrieval_strategy or ""
            row["query_stage"] = retrieval_stage or ""
            row["search_query"] = query
            row["discovery_path"] = f"related:{seed_video_id}"
            row["seed_video_id"] = seed_video_id
            row["seed_channel_id"] = seed_channel_id
            row["crawl_depth"] = 1
            anchored_rows.append(row)
        return anchored_rows

    def _graph_channel_rows_for_seed(
        self,
        *,
        seed_row: dict[str, Any],
        query: str,
        root_terms: list[str],
        video_duration: str | None,
        retrieval_strategy: str,
        retrieval_stage: str,
        deadline: float | None,
        channel_results: int,
    ) -> list[dict[str, Any]]:
        if channel_results <= 0 or self._deadline_exceeded(deadline):
            return []
        channel_id = str(seed_row.get("channel_id") or "").strip()
        seed_video_id = str(seed_row.get("id") or "").strip()
        if not channel_id or not seed_video_id:
            return []
        channel_html = self._fetch_channel_videos_html(channel_id, deadline=deadline)
        if not channel_html:
            return []
        channel_rows = self._extract_channel_videos_from_channel_html(
            channel_html,
            max_results=channel_results,
            video_duration=video_duration,
        )
        anchored_rows: list[dict[str, Any]] = []
        for row in channel_rows:
            candidate_id = str(row.get("id") or "").strip()
            if not candidate_id or candidate_id == seed_video_id:
                continue
            if not self._graph_candidate_is_anchored(
                row,
                query=query,
                root_terms=root_terms,
                seed_row=seed_row,
                require_same_channel=True,
            ):
                continue
            row["search_source"] = self.GRAPH_CHANNEL_SURFACE
            row["query_strategy"] = retrieval_strategy or ""
            row["query_stage"] = retrieval_stage or ""
            row["search_query"] = query
            row["discovery_path"] = f"channel:{channel_id}"
            row["seed_video_id"] = seed_video_id
            row["seed_channel_id"] = channel_id
            row["crawl_depth"] = 2
            anchored_rows.append(row)
        return anchored_rows

    def _fetch_watch_html(self, video_id: str, *, deadline: float | None) -> str:
        # Validate the ID before we send it. A bad ID on the query string would
        # just cause YouTube to serve an "unavailable" page, which would waste
        # a request AND poison the cache under a bogus key.
        if not _is_valid_youtube_video_id(video_id):
            logger.debug("Skipping watch HTML fetch — invalid video ID: %r", video_id)
            return ""
        cache_key = f"watch:{video_id}"
        cached = self._cache_get_text(cache_key, ttl_sec=self.WATCH_PAGE_CACHE_TTL_SEC)
        if cached is not None:
            return cached
        if self._deadline_exceeded(deadline) or self._network_backoff_active("youtube_html"):
            return ""
        try:
            resp = self._session_get(
                "https://www.youtube.com/watch",
                params={"v": video_id},
                deadline=deadline,
            )
            resp.raise_for_status()
            payload = resp.text
            self._note_request_success("youtube_html")
        except requests.RequestException as exc:
            self._note_request_failure(exc, scope="youtube_html")
            return ""
        # A successful response can still be useless if YouTube served us a
        # consent wall, a captcha, or an empty shell — those are shorter than
        # a real watch page. We log (not raise) so the caller can still fall
        # through to alternative strategies, but future-you gets a breadcrumb.
        if not payload or len(payload) < 5000:
            logger.warning(
                "Watch page fetch returned suspiciously small HTML: video_id=%s bytes=%d",
                video_id,
                len(payload),
            )
        self._cache_set_text(cache_key, payload)
        return payload

    def _fetch_channel_videos_html(self, channel_id: str, *, deadline: float | None) -> str:
        # Channel IDs all start with "UC" and are exactly 24 chars. Validate
        # before sending — prevents path injection and cache-key pollution.
        if not isinstance(channel_id, str) or not re.match(r"^UC[A-Za-z0-9_-]{22}$", channel_id):
            logger.debug("Skipping channel fetch — invalid channel ID: %r", channel_id)
            return ""
        cache_key = f"channel_videos:{channel_id}"
        cached = self._cache_get_text(cache_key, ttl_sec=self.CHANNEL_PAGE_CACHE_TTL_SEC)
        if cached is not None:
            return cached
        if self._deadline_exceeded(deadline) or self._network_backoff_active("youtube_html"):
            return ""
        try:
            resp = self._session_get(
                f"https://www.youtube.com/channel/{channel_id}/videos",
                deadline=deadline,
            )
            resp.raise_for_status()
            payload = resp.text
            self._note_request_success("youtube_html")
        except requests.RequestException as exc:
            self._note_request_failure(exc, scope="youtube_html")
            return ""
        if not payload or len(payload) < 5000:
            logger.warning(
                "Channel videos fetch returned suspiciously small HTML: channel_id=%s bytes=%d",
                channel_id,
                len(payload),
            )
        self._cache_set_text(cache_key, payload)
        return payload

    def _extract_related_videos_from_watch_html(
        self,
        html: str,
        *,
        max_results: int,
        video_duration: str | None,
    ) -> list[dict[str, Any]]:
        data = self._extract_yt_initial_data(html)
        if not data:
            return []
        return self._extract_videos_from_search_data(
            data=data,
            max_results=max_results,
            video_duration=video_duration,
        )

    def _extract_channel_videos_from_channel_html(
        self,
        html: str,
        *,
        max_results: int,
        video_duration: str | None,
    ) -> list[dict[str, Any]]:
        data = self._extract_yt_initial_data(html)
        if not data:
            return []
        return self._extract_videos_from_search_data(
            data=data,
            max_results=max_results,
            video_duration=video_duration,
        )

    def _build_search_query_variants(
        self,
        query: str,
        video_duration: str | None,
        source_surface: str,
        retrieval_profile: RetrievalProfile,
        retrieval_strategy: str = "",
    ) -> list[dict[str, str]]:
        base = " ".join(str(query or "").split()).strip()
        if not base:
            return []
        strategy_key = self._clean_query_text(retrieval_strategy).lower()
        if strategy_key in self.SEARCH_VARIANT_VISUAL_STRATEGIES:
            return [{"query": base, "surface": source_surface or "youtube_html"}]

        limit = 2 if retrieval_profile == "bootstrap" else 5
        variants: list[dict[str, str]] = []
        seen: set[str] = set()

        def add_variant(raw_query: str | None) -> None:
            clean_query = self._clean_query_text(str(raw_query or ""))
            if not clean_query:
                return
            key = clean_query.lower()
            if key in seen:
                return
            seen.add(key)
            variants.append({"query": clean_query, "surface": source_surface or "youtube_html"})

        add_variant(base)
        core_query = self._strip_search_variant_suffix(base)
        if core_query != base:
            add_variant(core_query)
        elif self._should_add_pedagogical_variant(base):
            for suffix in self._search_variant_suffix_candidates(
                retrieval_strategy=strategy_key,
                video_duration=video_duration,
            ):
                add_variant(f"{base} {suffix}")
                if len(variants) >= limit:
                    return variants[:limit]

        if retrieval_profile == "deep":
            add_variant(
                self._build_exact_query_variant(
                    base,
                    retrieval_strategy=retrieval_strategy,
                    retrieval_profile=retrieval_profile,
                )
            )
            if core_query != base:
                add_variant(
                    self._build_exact_query_variant(
                        core_query,
                        retrieval_strategy="",
                        retrieval_profile=retrieval_profile,
                    )
                )
                if self._should_add_pedagogical_variant(core_query):
                    for suffix in self._search_variant_suffix_candidates(
                        retrieval_strategy=strategy_key,
                        video_duration=video_duration,
                    ):
                        add_variant(f"{core_query} {suffix}")
                        if len(variants) >= limit:
                            break
        return variants[:limit]

    def _build_external_query_variants(
        self,
        *,
        query_variants: list[dict[str, str]],
        retrieval_strategy: str,
        retrieval_profile: RetrievalProfile,
    ) -> list[dict[str, str]]:
        variants: list[dict[str, str]] = []
        seen: set[str] = set()
        for variant in query_variants:
            search_query = self._clean_query_text(str(variant.get("query") or ""))
            surface = str(variant.get("surface") or "youtube_html")
            if not search_query:
                continue
            if search_query.lower() not in seen:
                seen.add(search_query.lower())
                variants.append({"query": search_query, "surface": surface})
            exact_variant = self._build_exact_query_variant(
                search_query,
                retrieval_strategy=retrieval_strategy,
                retrieval_profile=retrieval_profile,
            )
            if exact_variant and exact_variant.lower() not in seen:
                seen.add(exact_variant.lower())
                variants.append({"query": exact_variant, "surface": surface})
        # Fix L: Add educational platform search variants in deep mode
        if retrieval_profile == "deep" and query_variants:
            first_query = self._clean_query_text(str(query_variants[0].get("query") or ""))
            if first_query:
                edu_query = f"site:youtube.com {first_query} educational"
                if edu_query.lower() not in seen:
                    seen.add(edu_query.lower())
                    variants.append({"query": edu_query, "surface": "duckduckgo_site"})
        return variants

    def _clean_query_text(self, value: str) -> str:
        return " ".join(str(value or "").split()).strip()

    def _normalize_query_key(self, value: str) -> str:
        cleaned = self._clean_query_text(value).lower()
        tokens = re.findall(r"[a-z0-9\+#]+", cleaned)
        return " ".join(tokens)

    def _search_query_tokens(self, query: str) -> list[str]:
        tokens = []
        seen: set[str] = set()
        for token in re.findall(r"[A-Za-z0-9\+#]+", self._clean_query_text(query).lower()):
            if len(token) < 3 or token in self.SEARCH_QUERY_NOISE_TOKENS or token in seen:
                continue
            seen.add(token)
            tokens.append(token)
        return tokens

    def _strip_search_variant_suffix(self, query: str) -> str:
        base = self._clean_query_text(query)
        lowered = base.lower()
        for suffix in sorted(self.SEARCH_VARIANT_SUFFIXES, key=len, reverse=True):
            if lowered.endswith(f" {suffix}"):
                candidate = self._clean_query_text(base[: -len(suffix)])
                if len(self._search_query_tokens(candidate)) >= 1:
                    return candidate
        return base

    def _query_has_educational_suffix(self, query: str) -> bool:
        lowered = self._clean_query_text(query).lower()
        return any(lowered.endswith(f" {suffix}") or lowered == suffix for suffix in self.SEARCH_VARIANT_SUFFIXES)

    def _should_add_pedagogical_variant(self, query: str) -> bool:
        if self._query_has_educational_suffix(query):
            return False
        tokens = self._search_query_tokens(query)
        return 1 <= len(tokens) <= 6

    def _search_variant_suffix_candidates(
        self,
        *,
        retrieval_strategy: str,
        video_duration: str | None,
    ) -> tuple[str, ...]:
        strategy_key = self._clean_query_text(retrieval_strategy).lower()
        if video_duration == "long":
            return ("lecture", "explained")
        if video_duration == "short":
            return ("explained", "shorts")
        if strategy_key == "worked_example":
            return ("worked example", "tutorial")
        if strategy_key == "lecture":
            return ("lecture", "explained")
        if strategy_key == "documentary":
            return ("documentary", "explained")
        if strategy_key == "animation":
            return ("animation", "explained")
        if strategy_key in {"tutorial", "demo"}:
            return ("tutorial", "explained")
        return ("explained", "lecture")

    def _build_exact_query_variant(
        self,
        query: str,
        *,
        retrieval_strategy: str,
        retrieval_profile: RetrievalProfile,
    ) -> str | None:
        if retrieval_profile != "deep":
            return None
        base = self._clean_query_text(query)
        if not base or '"' in base:
            return None
        tokens = self._search_query_tokens(base)
        if len(tokens) < self.QUERY_VARIANT_EXACT_MIN_TOKENS or len(tokens) > self.QUERY_VARIANT_EXACT_MAX_TOKENS:
            return None

        suffix = self.STRATEGY_SUFFIX_BY_NAME.get(self._clean_query_text(retrieval_strategy).lower())
        core = base
        if suffix:
            suffix_words = suffix.split()
            query_words = base.split()
            if len(query_words) > len(suffix_words) and [word.lower() for word in query_words[-len(suffix_words) :]] == [
                word.lower() for word in suffix_words
            ]:
                candidate_core = " ".join(query_words[: -len(suffix_words)]).strip()
                if len(self._search_query_tokens(candidate_core)) >= 2:
                    core = candidate_core

        if len(self._search_query_tokens(core)) < 2:
            return None
        exact_core = f"\"{core}\""
        if suffix and core != base:
            return self._clean_query_text(f"{exact_core} {suffix}")
        return exact_core

    def _fetch_search_html(self, search_query: str, deadline: float | None = None) -> str:
        # Guard against empty queries. YouTube would happily return its
        # trending page for "", which would contaminate the cache and skew
        # every downstream scorer.
        clean_query = str(search_query or "").strip()
        if not clean_query:
            logger.debug("Skipping search HTML fetch — empty query")
            return ""
        if self._deadline_exceeded(deadline) or self._network_backoff_active("youtube_html"):
            return ""
        try:
            resp = self._session_get(
                "https://www.youtube.com/results",
                params={"search_query": clean_query},
                deadline=deadline,
            )
            resp.raise_for_status()
            self._note_request_success("youtube_html")
        except requests.RequestException as exc:
            self._note_request_failure(exc, scope="youtube_html")
            return ""
        payload = resp.text or ""
        # Heuristic sanity check — a real YouTube results page is typically
        # 300-800KB. Anything under a few KB is almost always a consent
        # interstitial, a rate-limit stub, or an empty body. We still return
        # whatever we got (the extractor may still salvage something), but we
        # warn so the root cause shows up in logs instead of as a mysterious
        # empty list.
        if len(payload) < 5000:
            logger.warning(
                "Search HTML fetch returned suspiciously small payload: query=%r bytes=%d",
                self._query_preview(clean_query),
                len(payload),
            )
        return payload

    def _fallback_video_row(self, video_id: str) -> dict[str, Any] | None:
        """Create a bare-bones row from just a video ID.

        Used when primary scraping fails and we can only recover raw IDs.
        Returns None for invalid IDs so callers don't have to double-check.
        The title is synthesised as "YouTube Video <id>" so the UI has
        *something* to render — the real title will typically be filled in
        later when the row is hydrated via a different path.
        """
        if not _is_valid_youtube_video_id(video_id):
            return None
        return {
            "id": video_id,
            "title": f"YouTube Video {video_id}",
            "channel_id": "",
            "channel_title": "",
            "description": "",
            "duration_sec": 0,
            "view_count": 0,
            "published_at": "",
            "is_creative_commons": False,
            "search_source": "youtube_html",
            "query_strategy": "",
            "query_stage": "",
            "search_query": "",
            "discovery_path": "search:youtube_html",
            "seed_video_id": "",
            "seed_channel_id": "",
            "crawl_depth": 0,
        }

    def _build_fallback_rows(
        self,
        video_ids: list[str],
        deadline: float | None = None,
        details_by_id: dict[str, dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        details = details_by_id if details_by_id is not None else (self._video_details(video_ids, deadline=deadline) if self.api_key else {})
        rows: list[dict[str, Any]] = []
        for video_id in video_ids:
            row = self._fallback_video_row(video_id)
            detail = details.get(video_id, {})
            if detail:
                row["title"] = str(detail.get("title") or row["title"])
                row["channel_id"] = str(detail.get("channel_id") or row["channel_id"])
                row["channel_title"] = str(detail.get("channel_title") or row["channel_title"])
                row["description"] = str(detail.get("description") or row["description"])
                row["duration_sec"] = int(detail.get("duration_sec") or 0)
                row["view_count"] = int(detail.get("view_count") or 0)
                row["published_at"] = str(detail.get("published_at") or row["published_at"])
                row["is_creative_commons"] = detail.get("license") == "creativeCommon"
            rows.append(row)
        return rows

    def _merge_unique_videos(
        self,
        primary: list[dict[str, Any]],
        secondary: list[dict[str, Any]],
        limit: int | None,
    ) -> list[dict[str, Any]]:
        ordered_ids: list[str] = []
        merged_by_id: dict[str, dict[str, Any]] = {}

        for source in (primary, secondary):
            for row in source:
                video_id = str(row.get("id") or "").strip()
                if not video_id:
                    continue
                normalized = self._normalize_video_row(row)
                existing = merged_by_id.get(video_id)
                if existing is None:
                    ordered_ids.append(video_id)
                    merged_by_id[video_id] = normalized
                    continue
                merged_by_id[video_id] = self._merge_video_rows(existing, normalized)

        merged = [merged_by_id[video_id] for video_id in ordered_ids]
        if limit is not None:
            return merged[:limit]
        return merged

    def _normalize_video_row(self, row: dict[str, Any]) -> dict[str, Any]:
        video_id = str(row.get("id") or "").strip()
        return {
            "id": video_id,
            "title": str(row.get("title") or "Untitled"),
            "channel_id": str(row.get("channel_id") or ""),
            "channel_title": str(row.get("channel_title") or ""),
            "description": str(row.get("description") or ""),
            "duration_sec": int(row.get("duration_sec") or 0),
            "view_count": int(row.get("view_count") or 0),
            "published_at": str(row.get("published_at") or ""),
            "is_creative_commons": bool(row.get("is_creative_commons")),
            "search_source": str(row.get("search_source") or "youtube_html"),
            "query_strategy": str(row.get("query_strategy") or ""),
            "query_stage": str(row.get("query_stage") or ""),
            "search_query": str(row.get("search_query") or ""),
            "discovery_path": str(row.get("discovery_path") or ""),
            "seed_video_id": str(row.get("seed_video_id") or ""),
            "seed_channel_id": str(row.get("seed_channel_id") or row.get("channel_id") or ""),
            "crawl_depth": max(0, int(row.get("crawl_depth") or 0)),
        }

    def _merge_video_rows(self, existing: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
        winner = candidate if self._video_row_rank(candidate) > self._video_row_rank(existing) else existing
        loser = existing if winner is candidate else candidate
        video_id = str(winner.get("id") or loser.get("id") or "").strip()
        merged = dict(winner)
        merged["title"] = self._prefer_text_field(
            existing.get("title"),
            candidate.get("title"),
            video_id=video_id,
            placeholder_sensitive=True,
        )
        merged["channel_id"] = self._prefer_text_field(existing.get("channel_id"), candidate.get("channel_id"))
        merged["channel_title"] = self._prefer_text_field(existing.get("channel_title"), candidate.get("channel_title"))
        merged["description"] = self._prefer_text_field(existing.get("description"), candidate.get("description"))
        merged["duration_sec"] = self._prefer_numeric_field(existing.get("duration_sec"), candidate.get("duration_sec"))
        merged["view_count"] = max(int(existing.get("view_count") or 0), int(candidate.get("view_count") or 0))
        merged["published_at"] = self._prefer_published_at(existing.get("published_at"), candidate.get("published_at"))
        merged["is_creative_commons"] = bool(existing.get("is_creative_commons")) or bool(candidate.get("is_creative_commons"))
        merged["search_source"] = str(winner.get("search_source") or loser.get("search_source") or "youtube_html")
        merged["query_strategy"] = str(winner.get("query_strategy") or loser.get("query_strategy") or "")
        merged["query_stage"] = str(winner.get("query_stage") or loser.get("query_stage") or "")
        merged["search_query"] = str(winner.get("search_query") or loser.get("search_query") or "")
        merged["discovery_path"] = str(winner.get("discovery_path") or loser.get("discovery_path") or "")
        merged["seed_video_id"] = str(winner.get("seed_video_id") or loser.get("seed_video_id") or "")
        merged["seed_channel_id"] = str(
            winner.get("seed_channel_id")
            or loser.get("seed_channel_id")
            or winner.get("channel_id")
            or loser.get("channel_id")
            or ""
        )
        merged["crawl_depth"] = min(
            4,
            max(int(existing.get("crawl_depth") or 0), int(candidate.get("crawl_depth") or 0)),
        )
        return merged

    def _prefer_text_field(
        self,
        left: Any,
        right: Any,
        *,
        video_id: str = "",
        placeholder_sensitive: bool = False,
    ) -> str:
        left_text = str(left or "").strip()
        right_text = str(right or "").strip()
        left_score = self._text_field_score(left_text, video_id=video_id, placeholder_sensitive=placeholder_sensitive)
        right_score = self._text_field_score(right_text, video_id=video_id, placeholder_sensitive=placeholder_sensitive)
        if right_score > left_score:
            return right_text
        return left_text or right_text

    def _text_field_score(self, value: str, *, video_id: str, placeholder_sensitive: bool) -> float:
        text = str(value or "").strip()
        if not text:
            return 0.0
        score = min(1.0, 0.2 + 0.04 * len(text.split()) + 0.003 * min(len(text), 120))
        if placeholder_sensitive and self._is_placeholder_title(text, video_id):
            score -= 0.7
        return score

    def _is_placeholder_title(self, title: str, video_id: str) -> bool:
        normalized = self._normalize_query_key(title)
        if not normalized:
            return True
        placeholder = self._normalize_query_key(f"YouTube Video {video_id}")
        return normalized == placeholder or normalized.startswith("youtube video ")

    def _prefer_numeric_field(self, left: Any, right: Any) -> int:
        left_value = int(left or 0)
        right_value = int(right or 0)
        if right_value > 0 and left_value <= 0:
            return right_value
        if left_value > 0:
            return left_value
        return right_value

    def _prefer_published_at(self, left: Any, right: Any) -> str:
        left_text = str(left or "").strip()
        right_text = str(right or "").strip()
        left_dt = self._parse_cache_time(left_text)
        right_dt = self._parse_cache_time(right_text)
        if right_dt and not left_dt:
            return right_text
        if left_dt and not right_dt:
            return left_text
        if right_dt and left_dt and right_dt > left_dt:
            return right_text
        return left_text or right_text

    def _video_row_rank(self, row: dict[str, Any]) -> tuple[float, float]:
        return (
            self._search_source_priority(str(row.get("search_source") or "")),
            self._video_row_quality(row),
        )

    def _search_source_priority(self, source: str) -> float:
        return float(self.SEARCH_SOURCE_PRIORITY.get(str(source or ""), 0.76))

    def _video_row_quality(self, row: dict[str, Any]) -> float:
        video_id = str(row.get("id") or "").strip()
        title = str(row.get("title") or "").strip()
        description = str(row.get("description") or "").strip()
        quality = 0.0
        if title and not self._is_placeholder_title(title, video_id):
            quality += 0.42
        if str(row.get("channel_title") or "").strip():
            quality += 0.18
        if description:
            quality += min(0.2, 0.02 * len(description.split()))
        if int(row.get("duration_sec") or 0) > 0:
            quality += 0.12
        if int(row.get("view_count") or 0) > 0:
            quality += 0.08
        if str(row.get("published_at") or "").strip():
            quality += 0.05
        quality += 0.12 * self._search_source_priority(str(row.get("search_source") or ""))
        return quality

    def _query_token_overlap(self, text: str, query_tokens: list[str]) -> float:
        if not query_tokens:
            return 0.0
        text_tokens = set(re.findall(r"[a-z0-9\+#]+", self._normalize_query_key(text)))
        if not text_tokens:
            return 0.0
        hits = sum(1 for token in query_tokens if token in text_tokens)
        return min(1.0, hits / max(1, len(query_tokens)))

    def _contains_query_phrase(self, text: str, query: str) -> bool:
        text_key = self._normalize_query_key(text)
        query_key = self._normalize_query_key(query)
        return bool(text_key and query_key and query_key in text_key)

    def _view_count_rank(self, view_count: Any) -> float:
        try:
            views = int(view_count or 0)
        except (TypeError, ValueError):
            views = 0
        if views <= 0:
            return 0.0
        if views >= 1_000_000:
            return 1.0
        if views >= 100_000:
            return 0.8
        if views >= 10_000:
            return 0.6
        if views >= 1_000:
            return 0.4
        return 0.2

    def _duration_match_score(self, duration_sec: int, video_duration: str | None) -> float:
        if video_duration not in {"short", "medium", "long"}:
            return 0.75 if duration_sec > 0 else 0.45
        if duration_sec <= 0:
            return 0.4
        if video_duration == "short":
            return 1.0 if duration_sec < 4 * 60 else 0.0
        if video_duration == "medium":
            return 1.0 if 4 * 60 <= duration_sec <= 20 * 60 else 0.0
        return 1.0 if duration_sec > 20 * 60 else 0.0

    def _search_result_score(self, row: dict[str, Any], *, query: str, video_duration: str | None) -> float:
        title = str(row.get("title") or "")
        metadata = " ".join(
            part for part in [title, str(row.get("description") or ""), str(row.get("channel_title") or "")] if part
        ).strip()
        query_tokens = self._search_query_tokens(query)
        title_overlap = self._query_token_overlap(title, query_tokens)
        metadata_overlap = self._query_token_overlap(metadata, query_tokens)
        title_phrase_hit = 1.0 if self._contains_query_phrase(title, query) else 0.0
        metadata_phrase_hit = 1.0 if self._contains_query_phrase(metadata, query) else 0.0
        duration_score = self._duration_match_score(int(row.get("duration_sec") or 0), video_duration)
        source_score = self._search_source_priority(str(row.get("search_source") or ""))
        metadata_quality = self._video_row_quality(row)
        view_score = self._view_count_rank(row.get("view_count"))
        placeholder_penalty = 0.28 if self._is_placeholder_title(title, str(row.get("id") or "")) else 0.0
        return (
            0.3 * title_overlap
            + 0.24 * metadata_overlap
            + 0.16 * title_phrase_hit
            + 0.1 * metadata_phrase_hit
            + 0.08 * duration_score
            + 0.12 * source_score
            + 0.12 * metadata_quality
            + 0.05 * view_score
            - placeholder_penalty
        )

    def _finalize_search_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        query: str,
        max_results: int,
        video_duration: str | None,
    ) -> list[dict[str, Any]]:
        merged = self._merge_unique_videos(rows, [], None)
        indexed = list(enumerate(merged))
        indexed.sort(
            key=lambda item: (
                self._search_result_score(item[1], query=query, video_duration=video_duration),
                self._search_source_priority(str(item[1].get("search_source") or "")),
                self._video_row_quality(item[1]),
                self._view_count_rank(item[1].get("view_count")),
                -item[0],
            ),
            reverse=True,
        )
        return [row for _idx, row in indexed[:max_results]]

    def _extract_videos_from_search_html(
        self,
        html: str,
        max_results: int,
        video_duration: str | None,
    ) -> list[dict[str, Any]]:
        data = self._extract_yt_initial_data(html)
        if not data:
            return []
        return self._extract_videos_from_search_data(
            data=data,
            max_results=max_results,
            video_duration=video_duration,
        )

    def _extract_videos_from_search_data(
        self,
        data: dict[str, Any] | None,
        max_results: int,
        video_duration: str | None,
    ) -> list[dict[str, Any]]:
        if not isinstance(data, dict):
            return []

        rows: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for renderer in self._iter_video_renderers(data):
            parsed = self._video_row_from_renderer(
                renderer,
                video_duration=video_duration,
            )
            if not parsed:
                continue
            video_id = str(parsed.get("id") or "").strip()
            if not video_id or video_id in seen_ids:
                continue
            rows.append(parsed)
            seen_ids.add(video_id)
            if len(rows) >= max_results:
                break
        return rows

    def _extract_search_continuation_token(self, data: dict[str, Any] | None) -> str | None:
        if not isinstance(data, dict):
            return None

        for token in self._iter_continuation_tokens(data):
            if isinstance(token, str) and token.strip():
                return token.strip()
        return None

    def _iter_continuation_tokens(self, node: Any):
        if isinstance(node, dict):
            continuation_endpoint = node.get("continuationEndpoint")
            if isinstance(continuation_endpoint, dict):
                continuation_command = continuation_endpoint.get("continuationCommand")
                if isinstance(continuation_command, dict):
                    token = continuation_command.get("token")
                    if isinstance(token, str):
                        yield token
            next_continuation = node.get("nextContinuationData")
            if isinstance(next_continuation, dict):
                token = next_continuation.get("continuation")
                if isinstance(token, str):
                    yield token
            continuation_item = node.get("continuationItemRenderer")
            if isinstance(continuation_item, dict):
                for token in self._iter_continuation_tokens(continuation_item):
                    yield token
            for child in node.values():
                yield from self._iter_continuation_tokens(child)
        elif isinstance(node, list):
            for item in node:
                yield from self._iter_continuation_tokens(item)

    def _extract_innertube_config(self, html: str) -> tuple[str | None, str | None]:
        """Pull the InnerTube API key and client version out of a YouTube HTML page.

        When you want to paginate YouTube search results (more than the first
        page), you can't keep re-fetching the HTML — you instead POST a
        continuation token to YouTube's internal InnerTube API. To call it,
        you need two credentials that YouTube bakes into every page:
          - INNERTUBE_API_KEY        — the public API key
          - INNERTUBE_CLIENT_VERSION — the client version string

        Both change over time, so we scrape them out of the page on every
        fetch rather than hard-coding them. We try multiple regex patterns
        because YouTube formats them slightly differently depending on the
        page (e.g. in quoted JSON vs a bare assignment).
        """
        if not html:
            return (None, None)

        # For each field, try the patterns in order from most-specific to
        # fallback. `re.search` scans the whole HTML, so the order here is
        # about robustness, not performance.
        api_key_patterns = (
            r'"INNERTUBE_API_KEY"\s*:\s*"([^"]+)"',
            r'innertubeApiKey"\s*:\s*"([^"]+)"',
            r'INNERTUBE_API_KEY\s*=\s*"([^"]+)"',
        )
        client_version_patterns = (
            r'"INNERTUBE_CLIENT_VERSION"\s*:\s*"([^"]+)"',
            r'innertubeClientVersion"\s*:\s*"([^"]+)"',
            r'INNERTUBE_CLIENT_VERSION\s*=\s*"([^"]+)"',
        )

        api_key: str | None = None
        for pattern in api_key_patterns:
            match = re.search(pattern, html)
            if match:
                candidate = match.group(1).strip()
                # A real InnerTube key is ~39 chars of url-safe base64.
                # Reject empty strings and absurd lengths so we don't POST
                # garbage.
                if 20 <= len(candidate) <= 80:
                    api_key = candidate
                    break

        client_version: str | None = None
        for pattern in client_version_patterns:
            match = re.search(pattern, html)
            if match:
                candidate = match.group(1).strip()
                # Versions look like "2.20240214.00.00" — a few numeric
                # segments. We don't enforce the exact shape (YouTube could
                # reformat it), but we do guard against empty or oversized
                # values.
                if candidate and len(candidate) < 40:
                    client_version = candidate
                    break

        if html and not api_key:
            logger.debug(
                "INNERTUBE_API_KEY not found in HTML — continuation pagination will be skipped "
                "(html_bytes=%d)",
                len(html),
            )
        return (api_key, client_version)

    def _fetch_search_continuation(
        self,
        continuation_token: str,
        innertube_api_key: str | None,
        innertube_client_version: str | None,
        deadline: float | None = None,
    ) -> dict[str, Any] | None:
        if (
            not innertube_api_key
            or not continuation_token
            or self._deadline_exceeded(deadline)
            or self._network_backoff_active("youtube_html")
        ):
            return None
        client_version = innertube_client_version or "2.20240214.00.00"
        try:
            resp = self._session_post(
                f"https://www.youtube.com/youtubei/v1/search?key={innertube_api_key}",
                json={
                    "context": {
                        "client": {
                            "clientName": "WEB",
                            "clientVersion": client_version,
                            "hl": "en",
                            "gl": "US",
                        }
                    },
                    "continuation": continuation_token,
                },
                deadline=deadline,
                headers={
                    "Content-Type": "application/json",
                    "Origin": "https://www.youtube.com",
                    "x-youtube-client-name": "1",
                    "x-youtube-client-version": client_version,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            self._note_request_success("youtube_html")
        except requests.RequestException as exc:
            self._note_request_failure(exc, scope="youtube_html")
            return None
        except ValueError:
            return None
        if not isinstance(data, dict):
            return None
        return data

    def _extract_yt_initial_data(self, html: str) -> dict[str, Any] | None:
        """Pull the ytInitialData JSON blob out of a YouTube HTML page.

        YouTube embeds its initial render state as a JavaScript variable in a
        <script> tag. The blob has been given a few different names and
        spellings over the years, so we try several matching strategies in
        order of specificity.  The most flexible path is a regex that allows
        any whitespace between ``ytInitialData`` and the opening ``{`` — this
        makes the extractor robust to minor formatting changes (e.g. YouTube
        dropping the space after ``=``, or wrapping the assignment in new
        layout-specific templates) without requiring a code change every time.
        """
        if not html:
            return None

        tried_any = False

        # 1) Flexible regex match.  Covers all historically observed forms
        #    of the assignment:
        #       var ytInitialData = {...}
        #       window["ytInitialData"] = {...}
        #       window['ytInitialData'] = {...}
        #       ytInitialData = {...}
        #       ytInitialData={...}             (no spaces)
        #       window.ytInitialData = {...}    (dotted attribute form)
        #    The regex anchors on ``ytInitialData`` followed by optional
        #    whitespace, an ``=`` sign, optional whitespace, then the opening
        #    brace — which matches the start of the JSON payload.  We then
        #    balance-match the object so we still capture trailing nested
        #    structures correctly.
        for match in _YT_INITIAL_DATA_ASSIGN_RE.finditer(html):
            tried_any = True
            start = match.start("brace")
            payload = self._balanced_json_object(html, start)
            if not payload:
                continue
            try:
                loaded = json.loads(payload)
            except json.JSONDecodeError as exc:
                logger.debug(
                    "ytInitialData regex matched but JSON decode failed: %s",
                    exc,
                )
                continue
            if isinstance(loaded, dict):
                return loaded

        # 2) Fallback: object-literal form ``"ytInitialData": {...}`` used
        #    by some consent-wall / embedded-player layouts.  We already try
        #    to ignore string-literal matches via JSON balance + decode; if
        #    every stronger match above failed, this is our last resort.
        for match in _YT_INITIAL_DATA_KEY_RE.finditer(html):
            tried_any = True
            start = match.start("brace")
            payload = self._balanced_json_object(html, start)
            if not payload:
                continue
            try:
                loaded = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(loaded, dict):
                return loaded

        if tried_any:
            logger.warning(
                "ytInitialData marker(s) found but no usable JSON object could be extracted "
                "(html_bytes=%d, title=%r). YouTube may have changed its embedding format.",
                len(html),
                self._html_title_snippet(html),
            )
        else:
            logger.warning(
                "ytInitialData marker not found in HTML (html_bytes=%d, title=%r). "
                "YouTube may have renamed the variable or served a consent wall.",
                len(html),
                self._html_title_snippet(html),
            )
        return None

    @staticmethod
    def _html_title_snippet(html: str) -> str:
        """Return a short <title> snippet from HTML for diagnostic logs.

        When ytInitialData extraction fails we cannot tell whether we got a
        consent wall, a bot challenge, an error page, or a valid-but-renamed
        results page.  Logging the page <title> (truncated) gives that
        signal without dumping the whole body into the log stream.
        """
        if not html:
            return ""
        match = _HTML_TITLE_RE.search(html)
        if not match:
            return ""
        title = match.group(1).strip()
        # Collapse whitespace so multi-line titles render on one log line.
        title = re.sub(r"\s+", " ", title)
        return title[:140]

    def _balanced_json_object(self, value: str, start_idx: int) -> str | None:
        depth = 0
        in_string = False
        escaped = False
        for i in range(start_idx, len(value)):
            ch = value[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return value[start_idx : i + 1]
        return None

    # The keys YouTube uses for "here is a video entry" inside ytInitialData.
    # Each one represents the same concept (a single video in a list) but the
    # surrounding component determines which key shows up:
    #   videoRenderer         → standard search / list result
    #   compactVideoRenderer  → the "up next" sidebar on a watch page
    #   gridVideoRenderer     → the grid layout on channel pages
    #   playlistVideoRenderer → entries inside a playlist
    #   reelItemRenderer      → Shorts shelf entries
    #   shortsLockupViewModel → newer Shorts layout (uses a slightly different
    #                           shape; we still try because some sub-fields
    #                           we read will simply be missing and the row
    #                           will be filtered out later rather than crash)
    # Adding a new key here is the cheapest way to recover lost results when
    # YouTube ships a UI change.
    _VIDEO_RENDERER_KEYS: tuple[str, ...] = (
        "videoRenderer",
        "compactVideoRenderer",
        "gridVideoRenderer",
        "playlistVideoRenderer",
        "reelItemRenderer",
        "shortsLockupViewModel",
    )

    def _iter_video_renderers(self, node: Any):
        """Walk a JSON tree and yield every video renderer we find.

        The ytInitialData structure is deeply nested and changes shape based
        on which surface (search / watch / channel) it came from. Rather than
        hard-code a path like `contents.twoColumnSearchResultsRenderer.…`,
        we do a tree walk that doesn't care about the surrounding structure
        — we just look for the leaf dictionaries that describe individual
        videos. This is strictly more resilient because YouTube can reshuffle
        its wrappers without breaking us, as long as the leaf nodes keep the
        same names.
        """
        if isinstance(node, dict):
            for renderer_key in self._VIDEO_RENDERER_KEYS:
                video_renderer = node.get(renderer_key)
                if isinstance(video_renderer, dict):
                    yield video_renderer
            for child in node.values():
                yield from self._iter_video_renderers(child)
        elif isinstance(node, list):
            for item in node:
                yield from self._iter_video_renderers(item)

    def _first_run_text(self, runs: Any) -> str:
        if not isinstance(runs, list):
            return ""
        for item in runs:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    return text.strip()
        return ""

    def _runs_text(self, runs: Any) -> str:
        if not isinstance(runs, list):
            return ""
        parts: list[str] = []
        for item in runs:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return " ".join(parts).strip()

    def _text_value(self, node: Any) -> str:
        if isinstance(node, dict):
            simple = node.get("simpleText")
            if isinstance(simple, str) and simple.strip():
                return simple.strip()
            runs = node.get("runs")
            return self._runs_text(runs)
        if isinstance(node, list):
            return self._runs_text(node)
        return str(node or "").strip()

    def _thumbnail_overlay_duration_text(self, renderer: dict[str, Any]) -> str:
        overlays = renderer.get("thumbnailOverlays")
        if not isinstance(overlays, list):
            return ""
        for item in overlays:
            if not isinstance(item, dict):
                continue
            status = item.get("thumbnailOverlayTimeStatusRenderer")
            if isinstance(status, dict):
                text = self._text_value(status.get("text"))
                if text:
                    return text
        return ""

    def _extract_channel_id_from_renderer(self, renderer: dict[str, Any]) -> str:
        for key in ("ownerText", "shortBylineText", "longBylineText"):
            node = renderer.get(key)
            runs = (node or {}).get("runs") if isinstance(node, dict) else None
            if not isinstance(runs, list):
                continue
            for item in runs:
                if not isinstance(item, dict):
                    continue
                endpoint = (item.get("navigationEndpoint") or {}).get("browseEndpoint") or {}
                browse_id = str(endpoint.get("browseId") or "").strip()
                if browse_id.startswith("UC"):
                    return browse_id
        navigation = (renderer.get("navigationEndpoint") or {}).get("browseEndpoint") or {}
        browse_id = str(navigation.get("browseId") or "").strip()
        if browse_id.startswith("UC"):
            return browse_id
        return ""

    def _renderer_description_text(self, renderer: dict[str, Any]) -> str:
        detailed = renderer.get("detailedMetadataSnippets")
        if isinstance(detailed, list) and detailed:
            first = detailed[0]
            if isinstance(first, dict):
                snippet = self._text_value((first.get("snippetText") or {}).get("runs"))
                if snippet:
                    return snippet
        for key in ("descriptionSnippet", "shortDescriptionSnippet"):
            snippet = self._text_value(renderer.get(key))
            if snippet:
                return snippet
        return ""

    def _video_row_from_renderer(
        self,
        renderer: dict[str, Any],
        *,
        video_duration: str | None,
    ) -> dict[str, Any] | None:
        """Turn a raw YouTube renderer dict into a normalized video row.

        A "renderer" is YouTube's term for one UI component — in our case,
        one video tile. Each one carries the data the page would use to draw
        that tile: title, channel, thumbnail, duration, view count, etc.

        We return None when:
          1. The video ID is missing or doesn't match the 11-char format.
             This is a hard requirement — without a valid ID, there's
             nothing the rest of the pipeline can do with the row.
          2. The duration is known and doesn't match the caller's filter.
             (If the duration is unknown, we keep the row and let later
             scoring decide.)
        """
        # --- 1. ID validation ---
        # We don't trust the JSON blindly: validate the shape BEFORE doing any
        # other work, so that a malformed row fails fast and cheaply.
        raw_video_id = renderer.get("videoId")
        if not _is_valid_youtube_video_id(raw_video_id):
            # Shorts renderers sometimes stash the ID on a sub-field instead
            # of the top-level `videoId`. Try a couple of common alternates
            # before giving up.
            for alt_key in ("onTap", "entityId", "navigationEndpoint"):
                node = renderer.get(alt_key)
                candidate = self._probe_video_id_from_subtree(node)
                if candidate:
                    raw_video_id = candidate
                    break
        if not _is_valid_youtube_video_id(raw_video_id):
            return None
        video_id: str = raw_video_id  # type: ignore[assignment]

        # --- 2. Duration (optional filter) ---
        # The duration text can appear in two places depending on the
        # renderer type. We try the primary location first, then fall back
        # to the overlay badge that appears on the thumbnail. We never guess
        # — if neither is present we treat the duration as unknown (0).
        duration_text = (
            self._text_value(renderer.get("lengthText"))
            or self._thumbnail_overlay_duration_text(renderer)
        )
        duration_sec = self._parse_duration_text(duration_text)
        if not self._duration_matches(duration_sec, video_duration):
            return None

        # --- 3. Title, channel, metadata — all with tolerant fallbacks ---
        # Title is almost always present but we default to "Untitled" rather
        # than failing the whole row. A missing title is usually a layout
        # change, not missing content — we'd rather surface the video.
        title = self._text_value(renderer.get("title")) or "Untitled"

        # Channel name lives under one of several keys depending on surface.
        # We try them in the most-specific-first order and accept the first
        # non-empty result.
        channel_title = (
            self._text_value(renderer.get("ownerText"))
            or self._text_value(renderer.get("shortBylineText"))
            or self._text_value(renderer.get("longBylineText"))
            or ""
        )

        # View count: try the full text first, then the abbreviated variant.
        view_text = (
            self._text_value(renderer.get("viewCountText"))
            or self._text_value(renderer.get("shortViewCountText"))
            or ""
        )
        published_text = self._text_value(renderer.get("publishedTimeText"))

        return {
            "id": video_id,
            "title": title,
            "channel_id": self._extract_channel_id_from_renderer(renderer),
            "channel_title": channel_title,
            "description": self._renderer_description_text(renderer),
            "duration_sec": duration_sec,
            "view_count": self._parse_view_count_text(view_text),
            "published_at": self._parse_published_time_text(published_text),
            "is_creative_commons": False,
            "seed_video_id": "",
            "seed_channel_id": "",
            "crawl_depth": 0,
        }

    def _probe_video_id_from_subtree(self, node: Any, *, max_depth: int = 4) -> str | None:
        """Recursively look for a string matching the YouTube video ID format.

        Used as a fallback when a renderer stores its ID on a nested field
        instead of top-level `videoId` (this happens with newer Shorts
        renderers like shortsLockupViewModel). We limit depth so a deeply
        nested blob can't turn into a slow walk.
        """
        if max_depth <= 0:
            return None
        if isinstance(node, str):
            return node if _is_valid_youtube_video_id(node) else None
        if isinstance(node, dict):
            for key in ("videoId", "id"):
                candidate = node.get(key)
                if _is_valid_youtube_video_id(candidate):
                    return candidate  # type: ignore[return-value]
            for value in node.values():
                found = self._probe_video_id_from_subtree(value, max_depth=max_depth - 1)
                if found:
                    return found
        elif isinstance(node, list):
            for item in node:
                found = self._probe_video_id_from_subtree(item, max_depth=max_depth - 1)
                if found:
                    return found
        return None

    def _parse_duration_text(self, value: str) -> int:
        if not value:
            return 0
        clean = value.strip()
        if not clean or not re.match(r"^\d{1,2}:\d{2}(?::\d{2})?$", clean):
            return 0
        parts = [int(p) for p in clean.split(":")]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        return 0

    def _parse_view_count_text(self, value: str) -> int:
        clean = " ".join(str(value or "").lower().replace(",", "").split()).strip()
        if not clean:
            return 0
        match = re.search(r"([\d.]+)\s*([kmb]?)\s+views?\b", clean)
        if not match:
            return 0
        try:
            amount = float(match.group(1))
        except (TypeError, ValueError):
            return 0
        suffix = match.group(2)
        multiplier = 1.0
        if suffix == "k":
            multiplier = 1_000.0
        elif suffix == "m":
            multiplier = 1_000_000.0
        elif suffix == "b":
            multiplier = 1_000_000_000.0
        return max(0, int(amount * multiplier))

    def _parse_published_time_text(self, value: str) -> str:
        clean = " ".join(str(value or "").lower().split()).strip()
        if not clean:
            return ""
        clean = clean.replace("streamed ", "").replace("premiered ", "")
        match = re.search(r"(\d+)\s+(second|minute|hour|day|week|month|year)s?\s+ago\b", clean)
        if not match:
            return ""
        amount = int(match.group(1))
        unit = match.group(2)
        if amount <= 0:
            return ""
        if unit == "second":
            delta = timedelta(seconds=amount)
        elif unit == "minute":
            delta = timedelta(minutes=amount)
        elif unit == "hour":
            delta = timedelta(hours=amount)
        elif unit == "day":
            delta = timedelta(days=amount)
        elif unit == "week":
            delta = timedelta(weeks=amount)
        elif unit == "month":
            delta = timedelta(days=30 * amount)
        else:
            delta = timedelta(days=365 * amount)
        return (datetime.now(timezone.utc) - delta).isoformat()

    def _duration_matches(self, duration_sec: int, video_duration: str | None) -> bool:
        if video_duration not in {"short", "medium", "long"}:
            return True
        if duration_sec <= 0:
            # keep unknown durations so transcript scoring can decide.
            return True
        if video_duration == "short":
            return duration_sec < 4 * 60
        if video_duration == "medium":
            return 4 * 60 <= duration_sec <= 20 * 60
        return duration_sec > 20 * 60

    def _search_via_duckduckgo(self, query: str, max_results: int, deadline: float | None = None) -> list[str]:
        """Fallback search using DuckDuckGo's HTML results page.

        Why this exists:
          Even when YouTube's own search breaks (blocked, consent wall,
          rate-limited), we can often still find relevant videos by
          searching DuckDuckGo with `site:youtube.com/watch` — that filter
          forces every result to be a YouTube watch URL we can extract an
          ID from.

        Why the `uddg=` unwrapping step:
          DuckDuckGo wraps every outbound link through its own redirect
          (for privacy tracking). The real URL is inside the `uddg` query
          parameter, URL-encoded. We unwrap it before trying to extract a
          video ID, otherwise we'd always see a duckduckgo.com host.
        """
        return self._collect_youtube_ids_from_html_search(
            scope="duckduckgo",
            url="https://duckduckgo.com/html/",
            query=f"site:youtube.com/watch {query}",
            pages=self.DUCKDUCKGO_PAGE_OFFSETS,
            param_builder=lambda page_param, search_q: {"q": search_q, "s": page_param},
            max_results=max_results,
            deadline=deadline,
            unwrap_redirect=True,
        )

    def _search_via_bing(self, query: str, max_results: int, deadline: float | None = None) -> list[str]:
        """Fallback search using Bing's HTML results page.

        Same strategy as DuckDuckGo — `site:youtube.com/watch` restricts
        every result to something we can extract an ID from. Bing doesn't
        wrap URLs through a redirect, so no unwrapping step is needed.
        """
        return self._collect_youtube_ids_from_html_search(
            scope="bing",
            url="https://www.bing.com/search",
            query=f"site:youtube.com/watch {query}",
            pages=self.BING_FIRST_OFFSETS,
            param_builder=lambda page_param, search_q: {
                "q": search_q,
                "count": 50,
                "first": page_param,
            },
            max_results=max_results,
            deadline=deadline,
            unwrap_redirect=False,
        )

    def _collect_youtube_ids_from_html_search(
        self,
        *,
        scope: str,
        url: str,
        query: str,
        pages: tuple[int, ...],
        param_builder: Any,
        max_results: int,
        deadline: float | None,
        unwrap_redirect: bool,
    ) -> list[str]:
        """Shared loop for the DuckDuckGo and Bing fallbacks.

        Extracting this into a helper does three things:
          1. Removes the copy-pasted loop from both searchers (easier to
             improve one place, not two).
          2. Lets us filter the raw `href=` candidates more aggressively
             (only hrefs that look like YouTube or a known redirector).
          3. Gives us one place to add logging for empty results.
        """
        ids: list[str] = []
        seen: set[str] = set()

        # Pre-compile a narrower regex so we ignore every link on the page
        # that can't possibly be a YouTube watch URL. This matters because
        # search pages have hundreds of irrelevant links (ads, navigation,
        # "related searches") and we'd otherwise burn CPU parsing them all.
        href_regex = re.compile(
            r'href="((?:https?:)?//(?:[^"]*?youtu(?:be\.com|\.be)[^"]*|[^"]*?uddg=[^"]*))"',
            re.IGNORECASE,
        )

        for page_param in pages:
            if self._deadline_exceeded(deadline) or self._network_backoff_active(scope):
                break
            try:
                resp = self._session_get(
                    url,
                    params=param_builder(page_param, query),
                    deadline=deadline,
                )
                resp.raise_for_status()
                self._note_request_success(scope)
            except requests.RequestException as exc:
                self._note_request_failure(exc, scope=scope)
                continue

            html = resp.text or ""
            if len(html) < 2000:
                # A tiny HTML body from a search engine almost always means
                # we were rate-limited or served a captcha. Log it, skip the
                # page, but keep trying the next offset — pagination may
                # recover further down.
                logger.debug(
                    "%s returned small payload (bytes=%d) on page=%s",
                    scope,
                    len(html),
                    page_param,
                )
                continue

            for match in href_regex.finditer(html):
                href = match.group(1)
                # HTML entities in the href (like &amp;) need to be turned
                # back into their real characters before URL parsing will
                # behave. This single replace handles the most common case.
                normalized = href.replace("&amp;", "&")
                if normalized.startswith("//"):
                    normalized = "https:" + normalized

                if unwrap_redirect and "uddg=" in normalized:
                    parsed = urlparse(normalized)
                    wrapped = parse_qs(parsed.query).get("uddg", [""])[0]
                    if wrapped:
                        normalized = unquote(wrapped)

                video_id = self._extract_video_id_from_url(normalized)
                if not _is_valid_youtube_video_id(video_id) or video_id in seen:
                    continue
                seen.add(video_id)  # type: ignore[arg-type]
                ids.append(video_id)  # type: ignore[arg-type]
                if len(ids) >= max_results:
                    return ids
        if not ids:
            logger.debug(
                "%s fallback returned no video IDs for query=%r",
                scope,
                self._query_preview(query),
            )
        return ids

    def _extract_video_id_from_url(self, value: str) -> str | None:
        try:
            parsed = urlparse(value)
        except Exception:
            return None

        host = (parsed.netloc or "").lower()
        video_id_pattern = r"^[A-Za-z0-9_-]{11}$"

        if "youtube.com" in host:
            vid = parse_qs(parsed.query).get("v", [""])[0]
            if re.match(video_id_pattern, vid):
                return vid

            parts = [p for p in parsed.path.split("/") if p]
            if len(parts) >= 2 and parts[0] in {"shorts", "embed", "live"} and re.match(video_id_pattern, parts[1]):
                return parts[1]

        if "youtu.be" in host:
            cand = parsed.path.strip("/").split("/")[0]
            if re.match(video_id_pattern, cand):
                return cand

        return None

    def extract_video_id_from_url(self, value: str) -> str | None:
        return self._extract_video_id_from_url(value)

    def _video_details(self, video_ids: list[str], deadline: float | None = None) -> dict[str, dict[str, Any]]:
        if not self.api_key or not video_ids:
            return {}
        result: dict[str, dict[str, Any]] = {}
        unique_ids = list(dict.fromkeys(video_ids))

        batches = [unique_ids[i : i + 50] for i in range(0, len(unique_ids), 50)]
        if not batches:
            return result

        workers = max(1, min(self.VIDEO_DETAILS_WORKERS, len(batches)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(self._video_details_batch, batch, deadline)
                for batch in batches
                if batch and not self._deadline_exceeded(deadline)
            ]
            for future in as_completed(futures):
                if self._deadline_exceeded(deadline):
                    break
                try:
                    payload = future.result()
                except Exception:
                    payload = {}
                if payload:
                    result.update(payload)
        return result

    def video_details(self, video_ids: list[str], deadline: float | None = None) -> dict[str, dict[str, Any]]:
        return self._video_details(video_ids, deadline=deadline)

    def _video_details_batch(self, batch: list[str], deadline: float | None) -> dict[str, dict[str, Any]]:
        if not batch or self._deadline_exceeded(deadline) or self._network_backoff_active("youtube_api"):
            return {}
        params = {
            "key": self.api_key,
            "part": "snippet,contentDetails,status,statistics",
            "id": ",".join(batch),
            "maxResults": len(batch),
        }
        try:
            resp = self._session_get(
                "https://www.googleapis.com/youtube/v3/videos",
                params=params,
                deadline=deadline,
            )
            resp.raise_for_status()
            data = resp.json()
            self._note_request_success("youtube_api")
        except requests.RequestException as exc:
            self._note_request_failure(exc, scope="youtube_api")
            return {}
        payload: dict[str, dict[str, Any]] = {}
        for item in data.get("items", []):
            vid = item.get("id")
            if not vid:
                continue
            duration = parse_iso8601_duration(item.get("contentDetails", {}).get("duration", ""))
            view_count_raw = item.get("statistics", {}).get("viewCount", 0)
            try:
                view_count = int(view_count_raw or 0)
            except (TypeError, ValueError):
                view_count = 0
            payload[vid] = {
                "title": str(item.get("snippet", {}).get("title") or "Untitled"),
                "channel_id": str(item.get("snippet", {}).get("channelId") or ""),
                "channel_title": str(item.get("snippet", {}).get("channelTitle") or ""),
                "description": str(item.get("snippet", {}).get("description") or ""),
                "published_at": str(item.get("snippet", {}).get("publishedAt") or ""),
                "duration_sec": duration,
                "view_count": max(0, view_count),
                "license": item.get("status", {}).get("license", "youtube"),
            }
        return payload

    def _deadline_exceeded(self, deadline: float | None) -> bool:
        return deadline is not None and time.monotonic() >= deadline

    def _request_timeout(self, deadline: float | None) -> float:
        if deadline is None:
            return self.request_timeout_sec
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return 0.2
        return max(0.2, min(self.request_timeout_sec, remaining))

    def _search_cache_is_fresh(self, created_at: str | None, is_empty: bool) -> bool:
        if not created_at:
            return True  # Legacy rows without timestamp — accept them
        try:
            ts = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            age_sec = (datetime.now(timezone.utc) - ts).total_seconds()
        except (TypeError, ValueError):
            return True
        ttl = self.SEARCH_CACHE_EMPTY_TTL_SEC if is_empty else self.SEARCH_CACHE_TTL_SEC
        return age_sec < ttl

    def _network_backoff_active(self, scope: str | None = None) -> bool:
        with self._network_backoff_lock:
            now = time.monotonic()
            if scope:
                return now < float(self._network_backoff_until_by_scope.get(scope, 0.0) or 0.0)
            return any(now < until for until in self._network_backoff_until_by_scope.values())

    def _note_request_success(self, scope: str) -> None:
        clean_scope = str(scope or "").strip()
        if not clean_scope:
            return
        prior_streak = 0
        had_backoff = False
        with self._network_backoff_lock:
            prior_streak = int(self._network_failure_streak_by_scope.get(clean_scope, 0) or 0)
            had_backoff = float(self._network_backoff_until_by_scope.get(clean_scope, 0.0) or 0.0) > time.monotonic()
            self._network_failure_streak_by_scope[clean_scope] = 0
            self._network_backoff_until_by_scope.pop(clean_scope, None)
        if self.retrieval_debug_logging and (prior_streak > 0 or had_backoff):
            logger.info(
                "YouTube request recovered: %s",
                json.dumps(
                    {
                        "scope": clean_scope,
                        "prior_streak": prior_streak,
                        "had_backoff": had_backoff,
                    },
                    sort_keys=True,
                ),
            )

    def _note_request_failure(self, exc: requests.RequestException, *, scope: str) -> None:
        clean_scope = str(scope or "").strip()
        if not clean_scope:
            return
        response = getattr(exc, "response", None)
        request = getattr(exc, "request", None)
        request_url = str(getattr(response, "url", None) or getattr(request, "url", None) or "").strip()
        parsed_url = urlparse(request_url)
        url_target = (
            f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
            if parsed_url.scheme and parsed_url.netloc
            else request_url
        )
        status_code = int(response.status_code) if getattr(response, "status_code", None) else None
        is_transport_failure = response is None
        retry_after = _parse_retry_after(response)
        streak = 0
        backoff_remaining_sec = 0.0
        with self._network_backoff_lock:
            if is_transport_failure:
                # Silent throttle / connection drop — trip circuit breaker path.
                streak = int(self._network_failure_streak_by_scope.get(clean_scope, 0) or 0) + 1
                self._network_failure_streak_by_scope[clean_scope] = streak
                if streak >= self.NETWORK_BACKOFF_FAILURE_THRESHOLD:
                    # Jittered backoff that scales with streak, capped at 180s.
                    backoff_sec = _backoff_delay(
                        streak - self.NETWORK_BACKOFF_FAILURE_THRESHOLD,
                        base_sec=self.NETWORK_BACKOFF_SEC,
                        cap_sec=self.NETWORK_BACKOFF_SEC * 4,
                    )
                    self._network_backoff_until_by_scope[clean_scope] = max(
                        float(self._network_backoff_until_by_scope.get(clean_scope, 0.0) or 0.0),
                        time.monotonic() + backoff_sec,
                    )
            elif status_code == 429 and retry_after is not None:
                # Explicit 429 with Retry-After — respect the server's backoff.
                self._network_backoff_until_by_scope[clean_scope] = max(
                    float(self._network_backoff_until_by_scope.get(clean_scope, 0.0) or 0.0),
                    time.monotonic() + retry_after,
                )
                streak = int(self._network_failure_streak_by_scope.get(clean_scope, 0) or 0) + 1
                self._network_failure_streak_by_scope[clean_scope] = streak
            elif status_code == 429:
                # 429 without Retry-After — exponential backoff with jitter.
                streak = int(self._network_failure_streak_by_scope.get(clean_scope, 0) or 0) + 1
                self._network_failure_streak_by_scope[clean_scope] = streak
                backoff_sec = _backoff_delay(streak, base_sec=2.0, cap_sec=30.0)
                self._network_backoff_until_by_scope[clean_scope] = max(
                    float(self._network_backoff_until_by_scope.get(clean_scope, 0.0) or 0.0),
                    time.monotonic() + backoff_sec,
                )
            else:
                streak = int(self._network_failure_streak_by_scope.get(clean_scope, 0) or 0)
            backoff_until = float(self._network_backoff_until_by_scope.get(clean_scope, 0.0) or 0.0)
            backoff_remaining_sec = max(0.0, backoff_until - time.monotonic())
        logger.warning(
            "YouTube request failure: %s",
            json.dumps(
                {
                    "scope": clean_scope,
                    "kind": "transport" if is_transport_failure else "http",
                    "status_code": status_code,
                    "retry_after": retry_after,
                    "url": url_target,
                    "failure_streak": streak,
                    "backoff_remaining_sec": round(backoff_remaining_sec, 2),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                sort_keys=True,
            ),
        )

    def get_transcript(
        self, conn, video_id: str, *, video_duration_sec: float | None = None,
    ) -> list[dict[str, Any]]:
        if conn is None:
            with get_conn() as local_conn:
                return self._single_flight_transcript(
                    local_conn, video_id, video_duration_sec=video_duration_sec,
                )
        return self._single_flight_transcript(
            conn, video_id, video_duration_sec=video_duration_sec,
        )

    def get_transcript_quality(self, conn, video_id: str) -> dict[str, Any] | None:
        """Return cached coverage metadata for a transcript, or ``None``."""
        row = fetch_one(
            conn,
            "SELECT coverage_ratio, cue_count, source_kind, quality_score, quality_status "
            "FROM transcript_cache WHERE video_id = ?",
            (video_id,),
        )
        if row and row.get("coverage_ratio") is not None:
            return {
                "coverage_ratio": float(row["coverage_ratio"]),
                "cue_count": int(row.get("cue_count") or 0),
                "source_kind": row.get("source_kind"),
                "quality_score": float(row["quality_score"]) if row.get("quality_score") is not None else None,
                "quality_status": row.get("quality_status"),
            }
        return None

    def _get_transcript_with_conn(
        self, conn, video_id: str, *, video_duration_sec: float | None = None,
    ) -> list[dict[str, Any]]:
        # ------------------------------------------------------------------
        # Cache lookup with status-specific TTLs and provenance upgrades
        # ------------------------------------------------------------------
        cached = fetch_one(
            conn,
            "SELECT transcript_json, created_at, source_kind, quality_score, quality_status "
            "FROM transcript_cache WHERE video_id = ?",
            (video_id,),
        )
        if cached:
            try:
                cached_transcript = json.loads(cached["transcript_json"])
            except json.JSONDecodeError:
                cached_transcript = []

            cached_status = str(cached.get("quality_status") or "").strip() or None
            cached_source = str(cached.get("source_kind") or "").strip() or None
            cached_priority = SOURCE_KIND_PRIORITY.get(cached_source or "", 0)
            created_at = self._parse_cache_time(str(cached.get("created_at") or ""))
            age_sec = (datetime.now(timezone.utc) - created_at).total_seconds() if created_at else float("inf")

            if cached_transcript:
                # Provenance-aware upgrade: low-priority sources (ASR) get
                # re-fetched after 24h to see if better captions are now available.
                if cached_priority <= 2 and age_sec > self.PROVENANCE_UPGRADE_AGE_SEC:
                    logger.info(
                        "Provenance upgrade: re-fetching %s (source=%s, age=%.0fs)",
                        video_id, cached_source, age_sec,
                    )
                    # Fall through to re-fetch; keep cached_transcript as fallback.
                else:
                    return cached_transcript
            else:
                # Empty transcript — check status-specific TTL.
                ttl = CACHE_TTL_BY_STATUS.get(cached_status, CACHE_TTL_BY_STATUS.get(None, 0))
                if ttl > 0 and age_sec < ttl:
                    return []
                # Expired non-success entry — fall through to re-fetch.

        # Keep any existing cached transcript as fallback for provenance upgrade.
        fallback_transcript: list[dict[str, Any]] = []
        if cached:
            try:
                fallback_transcript = json.loads(cached.get("transcript_json") or "[]")
            except json.JSONDecodeError:
                pass

        # ------------------------------------------------------------------
        # Fallback chain with quality gates at each stage
        # ------------------------------------------------------------------
        transcript: list[dict[str, Any]] = []
        source_kind = ""
        quality_obj: "TranscriptQuality | None" = None
        rejection_reason: str | None = None
        final_status = "failed_no_captions"  # default if all stages produce nothing
        extractor_version = ""
        model_version: str | None = None

        # Stage 1: youtube_transcript_api (web_caption_track)
        if not self._circuit_is_open("web_caption_track"):
            max_attempts = 3 if self._proxy_transcripts else 1
            stage1_failed = True
            for attempt in range(max_attempts):
                try:
                    api = self.transcript_api
                    if attempt > 0 and self._proxy_rotator.available:
                        proxy = self._proxy_rotator.next()
                        if proxy:
                            api = YouTubeTranscriptApi(proxies=proxy)
                            logger.info("Transcript retry %d/%d with proxy for video_id=%s", attempt + 1, max_attempts, video_id)
                    transcript = api.fetch(video_id, languages=["en"]).to_raw_data()
                    stage1_failed = False
                    break
                except NoTranscriptFound:
                    transcript = self._fallback_any_transcript(video_id)
                    stage1_failed = not bool(transcript)
                    break
                except (TranscriptsDisabled, VideoUnavailable):
                    transcript = []
                    break
                except Exception as exc:
                    exc_name = type(exc).__name__
                    if "IpBlocked" in exc_name and attempt < max_attempts - 1:
                        logger.warning("IP blocked fetching transcript for %s, rotating proxy (attempt %d)", video_id, attempt + 1)
                        delay = _backoff_delay(attempt, base_sec=1.0, cap_sec=15.0)
                        time.sleep(delay)
                        continue
                    if attempt == max_attempts - 1:
                        logger.warning("All youtube_transcript_api attempts failed for %s: %s — trying Innertube", video_id, exc_name)
                    transcript = []

            if stage1_failed:
                self._circuit_record_failure("web_caption_track")
            elif transcript:
                self._circuit_record_success("web_caption_track")

            if transcript:
                source_kind = "web_caption_track"
                extractor_version = "web_caption_v1"
                passes, quality_obj, rejection_reason = self._transcript_passes_quality_gate(
                    transcript, video_duration_sec, source_kind,
                )
                if not passes:
                    logger.warning("Quality gate failed for %s (source=%s): %s", video_id, source_kind, rejection_reason)
                    transcript = []

        # Stage 2: Innertube fallback
        if not transcript and not self._circuit_is_open("innertube"):
            try:
                transcript = self._innertube_fetch_transcript(video_id)
                if transcript:
                    self._circuit_record_success("innertube")
                    source_kind = "innertube_caption_track"
                    extractor_version = "innertube_v1"
                    passes, quality_obj, rejection_reason = self._transcript_passes_quality_gate(
                        transcript, video_duration_sec, source_kind,
                    )
                    if not passes:
                        logger.warning("Quality gate failed for %s (source=%s): %s", video_id, source_kind, rejection_reason)
                        transcript = []
            except Exception as exc:
                logger.debug("Innertube fallback failed for %s: %s", video_id, exc)
                self._circuit_record_failure("innertube")

        # Stage 3: yt-dlp subtitle download (cheaper than audio ASR)
        if not transcript and not self._circuit_is_open("yt_dlp_subtitle"):
            try:
                transcript, sub_source = self._yt_dlp_subtitle_fetch(video_id)
                if transcript:
                    self._circuit_record_success("yt_dlp_subtitle")
                    source_kind = "yt_dlp_subtitle"
                    extractor_version = "yt_dlp_sub_v1"
                    passes, quality_obj, rejection_reason = self._transcript_passes_quality_gate(
                        transcript, video_duration_sec, source_kind,
                    )
                    if not passes:
                        logger.warning("Quality gate failed for %s (source=%s): %s", video_id, source_kind, rejection_reason)
                        transcript = []
            except Exception as exc:
                logger.debug("yt-dlp subtitle fallback failed for %s: %s", video_id, exc)
                self._circuit_record_failure("yt_dlp_subtitle")

        # Stage 4: Whisper audio fallback (ASR — most expensive)
        if not transcript and not self._circuit_is_open("whisper"):
            try:
                transcript, whisper_source = self._whisper_audio_fallback(video_id)
                if transcript:
                    self._circuit_record_success("whisper")
                    source_kind = whisper_source  # "asr_local" or "asr_api"
                    extractor_version = "whisper_v1"
                    model_version = (
                        os.environ.get("FASTER_WHISPER_MODEL", "base.en")
                        if whisper_source == "asr_local" else "whisper-1"
                    )
                    passes, quality_obj, rejection_reason = self._transcript_passes_quality_gate(
                        transcript, video_duration_sec, source_kind,
                    )
                    if not passes:
                        logger.warning("Quality gate failed for %s (source=%s): %s", video_id, source_kind, rejection_reason)
                        transcript = []
            except Exception as exc:
                logger.debug("Whisper audio fallback failed for %s: %s", video_id, exc)
                self._circuit_record_failure("whisper")

        # ------------------------------------------------------------------
        # Determine final status and compute quality metadata
        # ------------------------------------------------------------------
        if transcript:
            final_status = "success"
        elif rejection_reason:
            final_status = "failed_quality"
        # else: final_status stays "failed_no_captions"

        # Provenance-aware upgrade: never downgrade. If re-fetch produced a
        # lower-priority or failed result, keep the existing cached transcript.
        if not transcript and fallback_transcript:
            cached_source_str = str(cached.get("source_kind") or "") if cached else ""
            cached_priority_val = SOURCE_KIND_PRIORITY.get(cached_source_str, 0)
            new_priority = SOURCE_KIND_PRIORITY.get(source_kind, 0)
            if new_priority <= cached_priority_val:
                logger.info("Provenance upgrade: keeping existing %s transcript for %s (new source %s not better)",
                            cached_source_str, video_id, source_kind or "none")
                return fallback_transcript

        # Compute coverage metadata for downstream consumers.
        coverage_ratio: float | None = None
        cue_count = len(transcript)
        quality_score: float | None = None
        if transcript and quality_obj is None:
            quality_obj = validate_transcript(transcript, video_duration_sec)
        if quality_obj:
            coverage_ratio = quality_obj.coverage_ratio
            quality_score = self._compute_quality_score(quality_obj, source_kind)

        # ------------------------------------------------------------------
        # Persist transcript artifact
        # ------------------------------------------------------------------
        row_data: dict[str, Any] = {
            "video_id": video_id,
            "transcript_json": dumps_json(transcript),
            "created_at": now_iso(),
            "source_kind": source_kind or None,
            "quality_score": quality_score,
            "language": "en",
            "extractor_version": extractor_version or None,
            "model_version": model_version,
            "normalization_version": NORMALIZATION_VERSION,
            "quality_status": final_status,
            "quality_rejection_reason": rejection_reason if final_status != "success" else None,
        }
        if coverage_ratio is not None:
            row_data["coverage_ratio"] = coverage_ratio
        if cue_count:
            row_data["cue_count"] = cue_count
        upsert(conn, "transcript_cache", row_data, pk="video_id")
        return transcript

    def _parse_cache_time(self, value: str) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # Quality score computation
    # ------------------------------------------------------------------

    def _compute_quality_score(
        self, quality: TranscriptQuality, source_kind: str,
    ) -> float:
        """Compute a 0.0-1.0 composite quality score from validation metrics."""
        source_bonus = {
            "official_captions": 1.0, "web_caption_track": 0.9,
            "innertube_caption_track": 0.7, "watch_page_caption_track": 0.6,
            "yt_dlp_subtitle": 0.6, "asr_local": 0.3, "asr_api": 0.2,
        }.get(source_kind, 0.0)
        gap_penalty = min(1.0, quality.largest_gap_sec / 60.0)
        cue_saturation = min(1.0, quality.cue_count / 50.0)
        return round(
            0.40 * min(1.0, quality.coverage_ratio)
            + 0.20 * (1.0 - quality.empty_cue_ratio)
            + 0.15 * (1.0 - gap_penalty)
            + 0.15 * cue_saturation
            + 0.10 * source_bonus,
            4,
        )

    # ------------------------------------------------------------------
    # Transcript quality gate
    # ------------------------------------------------------------------

    def _transcript_passes_quality_gate(
        self,
        transcript: list[dict[str, Any]],
        video_duration_sec: float | None,
        source_kind: str,
    ) -> tuple[bool, "TranscriptQuality | None", str | None]:
        """Validate a transcript against lenient quality thresholds.

        Returns ``(passes, quality, rejection_reason)``.
        """
        if not transcript:
            return False, None, "empty transcript"

        if len(transcript) < self.TRANSCRIPT_MIN_CUE_COUNT:
            return False, None, f"too few cues ({len(transcript)} < {self.TRANSCRIPT_MIN_CUE_COUNT})"

        total_chars = sum(len(str(c.get("text", ""))) for c in transcript)
        if total_chars < self.TRANSCRIPT_MIN_TOTAL_CHARS:
            return False, None, f"too little text ({total_chars} chars < {self.TRANSCRIPT_MIN_TOTAL_CHARS})"

        # Timestamp sanity: no negative start or duration values
        for cue in transcript:
            start = float(cue.get("start", 0))
            dur = float(cue.get("duration", 0))
            if start < 0 or dur < 0:
                return False, None, f"negative timestamp (start={start}, duration={dur})"

        quality = validate_transcript(
            transcript,
            video_duration_sec,
            min_coverage=self.TRANSCRIPT_MIN_COVERAGE,
            max_gap_sec=120.0,
            max_first_delay_sec=30.0,
            max_empty_ratio=self.TRANSCRIPT_MAX_EMPTY_RATIO,
        )

        if video_duration_sec and video_duration_sec > 0 and quality.coverage_ratio < self.TRANSCRIPT_MIN_COVERAGE:
            return False, quality, f"low coverage ({quality.coverage_ratio:.0%} < {self.TRANSCRIPT_MIN_COVERAGE:.0%})"

        if quality.empty_cue_ratio > self.TRANSCRIPT_MAX_EMPTY_RATIO:
            return False, quality, f"too many empty cues ({quality.empty_cue_ratio:.0%} > {self.TRANSCRIPT_MAX_EMPTY_RATIO:.0%})"

        # Language plausibility: check ASCII-letter ratio for English
        all_text = " ".join(str(c.get("text", "")) for c in transcript)
        if all_text:
            ascii_letters = sum(1 for ch in all_text if ch.isascii() and ch.isalpha())
            total_alpha = sum(1 for ch in all_text if ch.isalpha())
            if total_alpha > 20 and (ascii_letters / total_alpha) < 0.3:
                return False, quality, f"language plausibility failed (ASCII ratio {ascii_letters / total_alpha:.0%})"

        return True, quality, None

    # ------------------------------------------------------------------
    # Circuit breaker
    # ------------------------------------------------------------------

    def _circuit_is_open(self, scope: str) -> bool:
        """Return True if the circuit breaker for *scope* is open."""
        with self._network_backoff_lock:
            return time.monotonic() < float(self._circuit_open_until.get(scope, 0.0))

    def _circuit_record_failure(self, scope: str) -> None:
        with self._network_backoff_lock:
            count = int(self._circuit_failure_count.get(scope, 0)) + 1
            self._circuit_failure_count[scope] = count
            if count >= self.CIRCUIT_BREAKER_FAILURE_THRESHOLD:
                self._circuit_open_until[scope] = time.monotonic() + self.CIRCUIT_BREAKER_COOLDOWN_SEC
                logger.warning(
                    "Circuit breaker opened for scope=%s after %d failures (cooldown %.0fs)",
                    scope, count, self.CIRCUIT_BREAKER_COOLDOWN_SEC,
                )

    def _circuit_record_success(self, scope: str) -> None:
        with self._network_backoff_lock:
            self._circuit_failure_count.pop(scope, None)
            self._circuit_open_until.pop(scope, None)

    # ------------------------------------------------------------------
    # Single-flight transcript locking
    # ------------------------------------------------------------------

    def _single_flight_transcript(
        self, conn, video_id: str, *, video_duration_sec: float | None = None,
    ) -> list[dict[str, Any]]:
        """Acquire single-flight lock so concurrent requests for the same
        video_id trigger exactly one transcript fetch."""
        with self._transcript_flights_lock:
            existing_event = self._transcript_flights.get(video_id)
            if existing_event is not None:
                # Another thread is already fetching — wait for it, then read cache.
                pass  # fall through below
            else:
                event = threading.Event()
                self._transcript_flights[video_id] = event

        if existing_event is not None:
            existing_event.wait(timeout=60)
            # Read from cache — the fetching thread just populated it.
            cached = fetch_one(
                conn,
                "SELECT transcript_json FROM transcript_cache WHERE video_id = ?",
                (video_id,),
            )
            if cached:
                try:
                    return json.loads(cached["transcript_json"])
                except json.JSONDecodeError:
                    pass
            # Cache miss after wait — fall through to fetch ourselves.
            return self._get_transcript_with_conn(conn, video_id, video_duration_sec=video_duration_sec)

        try:
            return self._get_transcript_with_conn(conn, video_id, video_duration_sec=video_duration_sec)
        finally:
            with self._transcript_flights_lock:
                self._transcript_flights.pop(video_id, None)
            event.set()

    # ------------------------------------------------------------------
    # Alternative transcript fetching — bypasses youtube_transcript_api
    # via two strategies:
    #   A) Innertube player API → caption track URLs → json3 content
    #   B) Watch page scraping → extract captionTracks from HTML → json3
    # Both are fallbacks when youtube_transcript_api is IP-blocked.
    # ------------------------------------------------------------------
    _INNERTUBE_PLAYER_URL = "https://www.youtube.com/youtubei/v1/player"
    _INNERTUBE_CLIENTS: list[dict[str, Any]] = [
        {"context": {"client": {"clientName": "WEB", "clientVersion": "2.20240726.00.00", "hl": "en", "gl": "US"}}},
        {"context": {"client": {"clientName": "ANDROID", "clientVersion": "19.29.37", "hl": "en", "gl": "US", "androidSdkVersion": 34}}},
    ]

    def _innertube_fetch_transcript(self, video_id: str) -> list[dict[str, Any]]:
        """Fetch transcript bypassing youtube_transcript_api.

        Strategy A: Innertube player API (POST) → caption track URLs → json3.
        Strategy B: Scrape watch page HTML → extract captionTracks → json3.

        Returns transcript in [{text, start, duration}, ...] format.
        """
        session = requests.Session()
        session.headers.update({
            "User-Agent": _random_user_agent(),
            "Accept-Language": "en-US,en;q=0.9",
            **_STEALTH_HEADERS,
        })
        proxy = self._proxy_rotator.next() if self._proxy_rotator.available else None
        if proxy:
            session.proxies.update(proxy)

        caption_tracks: list[dict[str, Any]] = []

        # Strategy A: Innertube player API (works when not IP-blocked)
        for client_ctx in self._INNERTUBE_CLIENTS:
            try:
                session.headers["Content-Type"] = "application/json"
                session.headers["Origin"] = "https://www.youtube.com"
                session.headers["Referer"] = f"https://www.youtube.com/watch?v={video_id}"
                resp = session.post(
                    self._INNERTUBE_PLAYER_URL,
                    json={**client_ctx, "videoId": video_id},
                    timeout=12,
                )
                if resp.status_code != 200:
                    continue
                player_data = resp.json()
                tracks = (
                    player_data
                    .get("captions", {})
                    .get("playerCaptionsTracklistRenderer", {})
                    .get("captionTracks", [])
                )
                if tracks:
                    caption_tracks = tracks
                    logger.debug("Innertube player API found %d tracks for %s", len(tracks), video_id)
                    break
            except Exception as exc:
                logger.debug("Innertube player %s failed for %s: %s",
                             client_ctx["context"]["client"]["clientName"], video_id, exc)

        # Strategy B: Scrape watch page HTML for embedded captionTracks
        if not caption_tracks:
            try:
                session.headers.pop("Content-Type", None)
                session.headers.pop("Origin", None)
                watch_resp = session.get(
                    f"https://www.youtube.com/watch?v={video_id}",
                    timeout=15,
                )
                if watch_resp.status_code == 200:
                    page = watch_resp.text
                    idx = page.find('"captionTracks":')
                    if idx >= 0:
                        arr_start = page.index("[", idx)
                        depth = 0
                        arr_end = arr_start
                        for i in range(min(50000, len(page) - arr_start)):
                            ch = page[arr_start + i]
                            if ch == "[":
                                depth += 1
                            elif ch == "]":
                                depth -= 1
                            if depth == 0:
                                arr_end = arr_start + i + 1
                                break
                        caption_tracks = json.loads(page[arr_start:arr_end])
                        logger.debug("Watch page scrape found %d tracks for %s", len(caption_tracks), video_id)
            except Exception as exc:
                logger.debug("Watch page scrape failed for %s: %s", video_id, exc)

        if not caption_tracks:
            return []

        # Pick best English caption track (prefer manual over auto-generated)
        best_track = None
        for track in caption_tracks:
            lang = str(track.get("languageCode", "")).lower()
            if lang.startswith("en"):
                if best_track is None or track.get("kind") != "asr":
                    best_track = track
        if not best_track:
            best_track = caption_tracks[0]

        base_url = best_track.get("baseUrl", "")
        if not base_url:
            return []

        # Fetch caption content in json3 format
        base_url = re.sub(r"[&?]fmt=[^&]*", "", base_url)
        separator = "&" if "?" in base_url else "?"
        json3_url = f"{base_url}{separator}fmt=json3"

        try:
            cap_resp = session.get(json3_url, timeout=12)
            if cap_resp.status_code != 200:
                logger.debug("Caption content fetch returned %d for %s", cap_resp.status_code, video_id)
                return []
            cap_data = cap_resp.json()
        except Exception as exc:
            logger.debug("Caption content fetch failed for %s: %s", video_id, exc)
            return []

        # Parse json3 into [{text, start, duration}, ...]
        cues: list[dict[str, Any]] = []
        for event in cap_data.get("events", []):
            segs = event.get("segs")
            if not segs:
                continue
            text = "".join(seg.get("utf8", "") for seg in segs).strip()
            if not text or text == "\n":
                continue
            cues.append({
                "text": text,
                "start": event.get("tStartMs", 0) / 1000.0,
                "duration": event.get("dDurationMs", 0) / 1000.0,
            })

        if cues:
            logger.info("Alternative transcript fetch success for %s: %d cues", video_id, len(cues))
        return cues

    # ------------------------------------------------------------------
    # Whisper audio fallback — download audio, transcribe locally or via API
    # ------------------------------------------------------------------

    def _yt_dlp_subtitle_fetch(self, video_id: str) -> tuple[list[dict[str, Any]], str]:
        """Download subtitles via yt-dlp (no audio download — much cheaper than ASR).

        Returns ``(transcript, "yt_dlp_subtitle")`` or ``([], "")``.
        """
        import shutil
        import tempfile

        try:
            import yt_dlp
        except ImportError:
            logger.debug("yt-dlp not installed, skipping subtitle fetch")
            return [], ""

        tmpdir = tempfile.mkdtemp(prefix="reelai_ytdlp_sub_")
        try:
            ydl_opts = {
                "writesubtitles": True,
                "writeautomaticsub": True,
                "subtitleslangs": ["en"],
                "subtitlesformat": "vtt",
                "skip_download": True,
                "outtmpl": os.path.join(tmpdir, f"{video_id}.%(ext)s"),
                "quiet": True,
                "no_warnings": True,
                "socket_timeout": 10,
            }
            proxy = self._proxy_rotator.next() if self._proxy_rotator.available else None
            if proxy:
                proxy_url = proxy.get("https") or proxy.get("http") or ""
                if proxy_url:
                    ydl_opts["proxy"] = proxy_url

            logger.info("yt-dlp subtitle fetch for %s", video_id)
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={video_id}"])

            # Find downloaded .vtt files
            vtt_files = [f for f in os.listdir(tmpdir) if f.endswith(".vtt")]
            if not vtt_files:
                logger.debug("No VTT subtitle file produced for %s", video_id)
                return [], ""

            vtt_path = os.path.join(tmpdir, vtt_files[0])
            cues = self._parse_vtt_file(vtt_path)
            if cues:
                logger.info("yt-dlp subtitle fetch succeeded for %s (%d cues)", video_id, len(cues))
                return cues, "yt_dlp_subtitle"
            return [], ""
        except Exception as exc:
            logger.debug("yt-dlp subtitle fetch failed for %s: %s", video_id, exc)
            return [], ""
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    @staticmethod
    def _parse_vtt_file(vtt_path: str) -> list[dict[str, Any]]:
        """Parse a WebVTT file into [{text, start, duration}, ...] format."""
        import re

        cues: list[dict[str, Any]] = []
        try:
            with open(vtt_path, "r", encoding="utf-8") as f:
                content = f.read()
        except (OSError, UnicodeDecodeError):
            return []

        # Match VTT cue blocks:  HH:MM:SS.mmm --> HH:MM:SS.mmm\ntext
        pattern = re.compile(
            r"(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})\s*\n((?:.+\n?)+)",
        )
        for m in pattern.finditer(content):
            start_str, end_str, text = m.group(1), m.group(2), m.group(3)
            start = _vtt_timestamp_to_seconds(start_str)
            end = _vtt_timestamp_to_seconds(end_str)
            clean_text = re.sub(r"<[^>]+>", "", text).strip()
            if clean_text and end > start:
                cues.append({"text": clean_text, "start": start, "duration": round(end - start, 3)})
        return cues

    def _whisper_audio_fallback(self, video_id: str) -> tuple[list[dict[str, Any]], str]:
        """Download audio via yt-dlp, transcribe with faster-whisper or OpenAI Whisper.

        Completely bypasses YouTube's caption/timedtext system.
        Returns ``(transcript, source_kind)`` where source_kind is
        ``"asr_local"`` or ``"asr_api"``, or ``([], "")`` on failure.
        """
        import shutil
        import tempfile

        tmpdir = tempfile.mkdtemp(prefix="reelai_whisper_")
        try:
            # Step 1: Download audio with yt-dlp
            audio_path = os.path.join(tmpdir, f"{video_id}.wav")
            try:
                import yt_dlp
            except ImportError:
                logger.debug("yt-dlp not installed, skipping Whisper audio fallback")
                return [], ""

            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": os.path.join(tmpdir, f"{video_id}.%(ext)s"),
                "quiet": True,
                "no_warnings": True,
                "socket_timeout": 15,
                "retries": 2,
                "postprocessors": [{
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                }],
                "postprocessor_args": ["-ac", "1", "-ar", "16000"],
            }
            proxy = self._proxy_rotator.next() if self._proxy_rotator.available else None
            if proxy:
                proxy_url = proxy.get("https") or proxy.get("http") or ""
                if proxy_url:
                    ydl_opts["proxy"] = proxy_url

            logger.info("Whisper fallback: downloading audio for %s", video_id)
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={video_id}"])

            # Find the output WAV file
            wav_files = [f for f in os.listdir(tmpdir) if f.endswith(".wav")]
            if not wav_files:
                logger.debug("No WAV file produced for %s", video_id)
                return [], ""
            audio_path = os.path.join(tmpdir, wav_files[0])
            audio_size = os.path.getsize(audio_path)
            logger.info("Whisper fallback: audio downloaded for %s (%d bytes)", video_id, audio_size)

            # Step 2: Try faster-whisper (free, local)
            cues = self._transcribe_with_faster_whisper(audio_path, video_id)
            if cues:
                return cues, "asr_local"

            # Step 3: Try OpenAI Whisper API (paid, ~$0.006/min)
            cues = self._transcribe_with_openai_whisper(audio_path, video_id)
            if cues:
                return cues, "asr_api"

            logger.debug("All Whisper strategies failed for %s", video_id)
            return [], ""
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _transcribe_with_faster_whisper(
        self, audio_path: str, video_id: str,
    ) -> list[dict[str, Any]]:
        """Transcribe audio with faster-whisper (local CPU, free)."""
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            logger.debug("faster-whisper not installed, skipping local transcription")
            return []

        model_name = os.environ.get("FASTER_WHISPER_MODEL", "base.en")
        device = os.environ.get("FASTER_WHISPER_DEVICE", "cpu")
        compute_type = os.environ.get("FASTER_WHISPER_COMPUTE_TYPE", "int8")

        try:
            logger.info("Whisper fallback: transcribing %s with faster-whisper (%s)", video_id, model_name)
            model = WhisperModel(model_name, device=device, compute_type=compute_type)
            segments_iter, _info = model.transcribe(
                audio_path,
                language="en",
                beam_size=1,
                vad_filter=True,
            )
            cues: list[dict[str, Any]] = []
            for seg in segments_iter:
                text = seg.text.strip()
                if text:
                    cues.append({
                        "text": text,
                        "start": seg.start,
                        "duration": seg.end - seg.start,
                    })
            if cues:
                logger.info("faster-whisper success for %s: %d cues", video_id, len(cues))
            return cues
        except Exception as exc:
            logger.warning("faster-whisper failed for %s: %s", video_id, exc)
            return []

    def _transcribe_with_openai_whisper(
        self, audio_path: str, video_id: str,
    ) -> list[dict[str, Any]]:
        """Transcribe audio with OpenAI Whisper API (~$0.006/min)."""
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            logger.debug("No OPENAI_API_KEY, skipping OpenAI Whisper fallback")
            return []

        # Whisper API limit is 25 MiB
        max_size = 24 * 1024 * 1024
        if os.path.getsize(audio_path) > max_size:
            logger.debug("Audio too large for Whisper API (%d bytes), skipping", os.path.getsize(audio_path))
            return []

        try:
            import openai
            client = openai.OpenAI(api_key=api_key)
            logger.info("Whisper fallback: transcribing %s with OpenAI Whisper API", video_id)
            with open(audio_path, "rb") as f:
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    response_format="verbose_json",
                    language="en",
                    timestamp_granularities=["segment"],
                )
            # Parse response — may be dict or object
            if hasattr(response, "segments"):
                segments = response.segments or []
            elif isinstance(response, dict):
                segments = response.get("segments", [])
            else:
                segments = []

            cues: list[dict[str, Any]] = []
            for seg in segments:
                if isinstance(seg, dict):
                    text = seg.get("text", "").strip()
                    start = float(seg.get("start", 0))
                    end = float(seg.get("end", 0))
                else:
                    text = getattr(seg, "text", "").strip()
                    start = float(getattr(seg, "start", 0))
                    end = float(getattr(seg, "end", 0))
                if text:
                    cues.append({
                        "text": text,
                        "start": start,
                        "duration": max(0.0, end - start),
                    })
            if cues:
                logger.info("OpenAI Whisper success for %s: %d cues", video_id, len(cues))
            return cues
        except Exception as exc:
            logger.warning("OpenAI Whisper failed for %s: %s", video_id, exc)
            return []

    def _fallback_any_transcript(self, video_id: str) -> list[dict[str, Any]]:
        try:
            transcript_list = self.transcript_api.list(video_id)
            try:
                return transcript_list.find_manually_created_transcript(["en"]).fetch().to_raw_data()
            except NoTranscriptFound:
                pass
            try:
                return transcript_list.find_generated_transcript(["en"]).fetch().to_raw_data()
            except NoTranscriptFound:
                pass
            for transcript in transcript_list:
                try:
                    return transcript.fetch().to_raw_data()
                except Exception:
                    continue
        except Exception:
            return []
        return []
