import hashlib
import json
import logging
import os
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

logger = logging.getLogger(__name__)


class YouTubeApiRequestError(RuntimeError):
    pass


RetrievalProfile = Literal["bootstrap", "deep"]
GraphProfile = Literal["off", "light", "deep"]


def _cache_key(*parts: str) -> str:
    value = "|".join(parts)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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
    HTML_MAX_PAGES = 6
    HTML_MAX_PAGES_DEEP = 10
    HTML_POOL_MULTIPLIER = 8
    HTML_POOL_CAP = 600
    SEARCH_SURFACE_WORKERS = 4
    VIDEO_DETAILS_WORKERS = 3
    DUCKDUCKGO_PAGE_OFFSETS = (0, 30, 60, 90, 120, 150, 180, 210)
    BING_FIRST_OFFSETS = (0, 20, 40, 60, 80, 100, 120, 140)
    REQUEST_TIMEOUT_SEC = 8.0
    SEARCH_TIME_BUDGET_SEC = 12.0
    NETWORK_BACKOFF_SEC = 45.0
    NETWORK_BACKOFF_FAILURE_THRESHOLD = 3
    SESSION_POOL_SIZE = 24
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
    GRAPH_FETCH_WORKERS = 4
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
        self.transcript_api = YouTubeTranscriptApi()
        self.empty_transcript_ttl_sec = 6 * 60 * 60
        self._network_backoff_until = 0.0
        self._network_backoff_lock = threading.Lock()
        self._network_backoff_until_by_scope: dict[str, float] = {}
        self._network_failure_streak_by_scope: dict[str, int] = {}
        self._page_cache_lock = threading.Lock()
        self._page_cache: dict[str, tuple[float, str]] = {}
        self.serverless_mode = bool(
            os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME") or os.getenv("K_SERVICE")
        )
        self.retrieval_debug_logging = bool(settings.retrieval_debug_logging)
        self.search_time_budget_sec = 3.5 if self.serverless_mode else self.SEARCH_TIME_BUDGET_SEC
        self.request_timeout_sec = 2.5 if self.serverless_mode else self.REQUEST_TIMEOUT_SEC
        self._session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=self.SESSION_POOL_SIZE,
            pool_maxsize=self.SESSION_POOL_SIZE,
            max_retries=0,
        )
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)
        self._session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "en-US,en;q=0.9",
            }
        )

    def _session_get(self, url: str, *, deadline: float | None = None, **kwargs: Any) -> requests.Response:
        return self._session.get(
            url,
            timeout=self._request_timeout(deadline),
            **kwargs,
        )

    def _session_post(self, url: str, *, deadline: float | None = None, **kwargs: Any) -> requests.Response:
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

        ids: list[str] = []
        seen_ids: set[str] = set()
        for video_id in re.findall(r'"videoId":"([A-Za-z0-9_-]{11})"', html):
            if video_id in seen_ids:
                continue
            seen_ids.add(video_id)
            ids.append(video_id)
            if len(ids) >= target_pool:
                break
        if ids:
            fallback_rows = [self._fallback_video_row(video_id) for video_id in ids]
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
        self._cache_set_text(cache_key, payload)
        return payload

    def _fetch_channel_videos_html(self, channel_id: str, *, deadline: float | None) -> str:
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
        if self._deadline_exceeded(deadline) or self._network_backoff_active("youtube_html"):
            return ""
        try:
            resp = self._session_get(
                "https://www.youtube.com/results",
                params={"search_query": search_query},
                deadline=deadline,
            )
            resp.raise_for_status()
            self._note_request_success("youtube_html")
            return resp.text
        except requests.RequestException as exc:
            self._note_request_failure(exc, scope="youtube_html")
            return ""

    def _fallback_video_row(self, video_id: str) -> dict[str, Any]:
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
        api_match = re.search(r'"INNERTUBE_API_KEY":"([^"]+)"', html)
        client_match = re.search(r'"INNERTUBE_CLIENT_VERSION":"([^"]+)"', html)
        api_key = api_match.group(1).strip() if api_match else None
        client_version = client_match.group(1).strip() if client_match else None
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
        markers = ["var ytInitialData = ", "ytInitialData = "]
        for marker in markers:
            idx = html.find(marker)
            if idx < 0:
                continue
            start = html.find("{", idx + len(marker))
            if start < 0:
                continue
            payload = self._balanced_json_object(html, start)
            if not payload:
                continue
            try:
                loaded = json.loads(payload)
                if isinstance(loaded, dict):
                    return loaded
            except json.JSONDecodeError:
                continue
        return None

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

    def _iter_video_renderers(self, node: Any):
        if isinstance(node, dict):
            for renderer_key in ("videoRenderer", "compactVideoRenderer", "gridVideoRenderer"):
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
        video_id = renderer.get("videoId")
        if not isinstance(video_id, str) or not video_id.strip():
            return None

        duration_text = self._text_value(renderer.get("lengthText")) or self._thumbnail_overlay_duration_text(renderer)
        duration_sec = self._parse_duration_text(duration_text)
        if not self._duration_matches(duration_sec, video_duration):
            return None

        title = self._text_value(renderer.get("title")) or "Untitled"
        channel = (
            self._text_value(renderer.get("ownerText"))
            or self._text_value(renderer.get("shortBylineText"))
            or self._text_value(renderer.get("longBylineText"))
        )
        view_text = self._text_value(renderer.get("viewCountText")) or self._text_value(renderer.get("shortViewCountText"))
        published_text = self._text_value(renderer.get("publishedTimeText"))
        return {
            "id": video_id,
            "title": title,
            "channel_id": self._extract_channel_id_from_renderer(renderer),
            "channel_title": channel,
            "description": self._renderer_description_text(renderer),
            "duration_sec": duration_sec,
            "view_count": self._parse_view_count_text(view_text),
            "published_at": self._parse_published_time_text(published_text),
            "is_creative_commons": False,
            "seed_video_id": "",
            "seed_channel_id": "",
            "crawl_depth": 0,
        }

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
        ids: list[str] = []
        seen: set[str] = set()

        for offset in self.DUCKDUCKGO_PAGE_OFFSETS:
            if self._deadline_exceeded(deadline) or self._network_backoff_active("duckduckgo"):
                break
            try:
                resp = self._session_get(
                    "https://duckduckgo.com/html/",
                    params={"q": f"site:youtube.com/watch {query}", "s": offset},
                    deadline=deadline,
                )
                resp.raise_for_status()
                self._note_request_success("duckduckgo")
            except requests.RequestException as exc:
                self._note_request_failure(exc, scope="duckduckgo")
                continue

            html = resp.text
            candidates = re.findall(r'href="([^"]+)"', html)
            for href in candidates:
                normalized = href.replace("&amp;", "&")
                if "uddg=" in normalized:
                    parsed = urlparse(normalized)
                    unwrapped = parse_qs(parsed.query).get("uddg", [""])[0]
                    if unwrapped:
                        normalized = unquote(unwrapped)

                video_id = self._extract_video_id_from_url(normalized)
                if not video_id or video_id in seen:
                    continue
                seen.add(video_id)
                ids.append(video_id)
                if len(ids) >= max_results:
                    return ids
        return ids

    def _search_via_bing(self, query: str, max_results: int, deadline: float | None = None) -> list[str]:
        ids: list[str] = []
        seen: set[str] = set()
        for first in self.BING_FIRST_OFFSETS:
            if self._deadline_exceeded(deadline) or self._network_backoff_active("bing"):
                break
            try:
                resp = self._session_get(
                    "https://www.bing.com/search",
                    params={
                        "q": f"site:youtube.com/watch {query}",
                        "count": 50,
                        "first": first,
                    },
                    deadline=deadline,
                )
                resp.raise_for_status()
                self._note_request_success("bing")
            except requests.RequestException as exc:
                self._note_request_failure(exc, scope="bing")
                continue

            html = resp.text
            candidates = re.findall(r'href="([^"]+)"', html)
            for href in candidates:
                normalized = href.replace("&amp;", "&")
                video_id = self._extract_video_id_from_url(normalized)
                if not video_id or video_id in seen:
                    continue
                seen.add(video_id)
                ids.append(video_id)
                if len(ids) >= max_results:
                    return ids
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
        streak = 0
        backoff_remaining_sec = 0.0
        with self._network_backoff_lock:
            if is_transport_failure:
                streak = int(self._network_failure_streak_by_scope.get(clean_scope, 0) or 0) + 1
                self._network_failure_streak_by_scope[clean_scope] = streak
                if streak >= self.NETWORK_BACKOFF_FAILURE_THRESHOLD:
                    self._network_backoff_until_by_scope[clean_scope] = max(
                        float(self._network_backoff_until_by_scope.get(clean_scope, 0.0) or 0.0),
                        time.monotonic() + self.NETWORK_BACKOFF_SEC,
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
                    "url": url_target,
                    "failure_streak": streak,
                    "backoff_remaining_sec": round(backoff_remaining_sec, 2),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                sort_keys=True,
            ),
        )

    def get_transcript(self, conn, video_id: str) -> list[dict[str, Any]]:
        if conn is None:
            with get_conn() as local_conn:
                return self._get_transcript_with_conn(local_conn, video_id)
        return self._get_transcript_with_conn(conn, video_id)

    def _get_transcript_with_conn(self, conn, video_id: str) -> list[dict[str, Any]]:
        cached = fetch_one(conn, "SELECT transcript_json, created_at FROM transcript_cache WHERE video_id = ?", (video_id,))
        if cached:
            try:
                cached_transcript = json.loads(cached["transcript_json"])
            except json.JSONDecodeError:
                cached_transcript = []
            if cached_transcript:
                return cached_transcript
            # Short-term cache misses to avoid hammering transcript fetch on videos with no transcript.
            created_at = self._parse_cache_time(str(cached.get("created_at") or ""))
            if created_at:
                age_sec = (datetime.now(timezone.utc) - created_at).total_seconds()
                if age_sec < self.empty_transcript_ttl_sec:
                    return []

        transcript: list[dict[str, Any]] = []
        try:
            transcript = self.transcript_api.fetch(video_id, languages=["en"]).to_raw_data()
        except NoTranscriptFound:
            transcript = self._fallback_any_transcript(video_id)
        except (TranscriptsDisabled, VideoUnavailable):
            transcript = []
        except Exception:
            transcript = []

        upsert(
            conn,
            "transcript_cache",
            {
                "video_id": video_id,
                "transcript_json": dumps_json(transcript),
                "created_at": now_iso(),
            },
            pk="video_id",
        )
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
