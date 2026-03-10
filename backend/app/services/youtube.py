import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import requests
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

from ..config import get_settings
from ..db import dumps_json, fetch_one, get_conn, now_iso, upsert


class YouTubeApiRequestError(RuntimeError):
    pass


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
    DATA_API_MAX_PAGES = 10
    DATA_API_POOL_MULTIPLIER = 8
    DATA_API_POOL_CAP = 760
    SEARCH_VARIANTS_LIMIT = 16
    PRIMARY_VARIANT_LIMIT = 3
    HTML_MAX_PAGES = 6
    HTML_POOL_MULTIPLIER = 8
    HTML_POOL_CAP = 600
    SEARCH_SURFACE_WORKERS = 4
    VIDEO_DETAILS_WORKERS = 3
    DUCKDUCKGO_PAGE_OFFSETS = (0, 30, 60, 90, 120, 150, 180, 210)
    BING_FIRST_OFFSETS = (0, 20, 40, 60, 80, 100, 120, 140)
    REQUEST_TIMEOUT_SEC = 8.0
    SEARCH_TIME_BUDGET_SEC = 12.0
    NETWORK_BACKOFF_SEC = 45.0

    def __init__(self) -> None:
        settings = get_settings()
        self.api_key = settings.youtube_api_key
        self.transcript_api = YouTubeTranscriptApi()
        self.empty_transcript_ttl_sec = 6 * 60 * 60
        self._network_backoff_until = 0.0
        self._network_backoff_lock = threading.Lock()
        self.serverless_mode = bool(
            os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME") or os.getenv("K_SERVICE")
        )
        self.search_time_budget_sec = 3.5 if self.serverless_mode else self.SEARCH_TIME_BUDGET_SEC
        self.request_timeout_sec = 2.5 if self.serverless_mode else self.REQUEST_TIMEOUT_SEC

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
    ) -> list[dict[str, Any]]:
        duration_key = video_duration or "any"
        key = _cache_key(
            query,
            str(max_results),
            str(creative_commons_only),
            duration_key,
            retrieval_strategy or "",
            retrieval_stage or "",
            source_surface or "",
        )
        cached = fetch_one(conn, "SELECT response_json FROM search_cache WHERE cache_key = ?", (key,))
        if cached:
            try:
                payload = json.loads(cached["response_json"])
            except (TypeError, json.JSONDecodeError):
                payload = []
            if isinstance(payload, list):
                return self._merge_unique_videos(payload, [], max_results)

        if self._network_backoff_active():
            return []

        deadline = time.monotonic() + self.search_time_budget_sec
        videos: list[dict[str, Any]] = []
        fast_non_api_enabled = not creative_commons_only
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
                        ),
                    )
                )

            future_to_surface = {future: surface for surface, future in futures}
            for future in as_completed(future_to_surface):
                try:
                    rows = future.result()
                except YouTubeApiRequestError:
                    rows = []
                except Exception:
                    rows = []
                if rows:
                    videos = self._merge_unique_videos(videos, rows, max_results)
                if len(videos) >= max_results:
                    break

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
                include_external_fallbacks=True,
                variant_limit=self.SEARCH_VARIANTS_LIMIT,
                skip_primary_variants=True,
            )
            videos = self._merge_unique_videos(videos, expanded, max_results)

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
            )
        videos = self._merge_unique_videos(videos, [], max_results)

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
    ) -> list[dict[str, Any]]:
        target_pool = max(max_results, min(self.DATA_API_POOL_CAP, max_results * self.DATA_API_POOL_MULTIPLIER))
        per_page = 50
        items: list[dict[str, Any]] = []
        seen_video_ids: set[str] = set()
        next_page_token: str | None = None

        for page_idx in range(self.DATA_API_MAX_PAGES):
            if self._deadline_exceeded(deadline):
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
                resp = requests.get(
                    "https://www.googleapis.com/youtube/v3/search",
                    params=params,
                    timeout=self._request_timeout(deadline),
                )
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as exc:
                self._note_request_failure(exc)
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
            videos.append(
                {
                    "id": video_id,
                    "title": item.get("snippet", {}).get("title", "Untitled"),
                    "channel_title": item.get("snippet", {}).get("channelTitle", ""),
                    "description": item.get("snippet", {}).get("description", ""),
                    "duration_sec": detail.get("duration_sec", 0),
                    "view_count": detail.get("view_count", 0),
                    "published_at": item.get("snippet", {}).get("publishedAt", ""),
                    "is_creative_commons": bool(is_cc),
                    "search_source": "youtube_api",
                    "query_strategy": retrieval_strategy or "",
                    "query_stage": retrieval_stage or "",
                    "search_query": query,
                }
            )
        return self._merge_unique_videos(videos, [], max_results)

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
    ) -> list[dict[str, Any]]:
        # License cannot be reliably verified without the Data API.
        if creative_commons_only:
            return []
        if self._deadline_exceeded(deadline) or self._network_backoff_active():
            return []

        target_pool = max(max_results, min(self.HTML_POOL_CAP, max_results * self.HTML_POOL_MULTIPLIER))
        results: list[dict[str, Any]] = []
        query_variants = self._build_search_query_variants(
            query=query,
            video_duration=video_duration,
            source_surface=source_surface,
        )
        if skip_primary_variants and len(query_variants) > self.PRIMARY_VARIANT_LIMIT:
            query_variants = query_variants[self.PRIMARY_VARIANT_LIMIT :]
        if variant_limit and variant_limit > 0:
            query_variants = query_variants[:variant_limit]
        if not query_variants:
            return []

        html_futures: list[Any] = []
        max_html_workers = max(1, min(self.SEARCH_SURFACE_WORKERS, len(query_variants)))
        with ThreadPoolExecutor(max_workers=max_html_workers) as executor:
            for variant in query_variants:
                search_query = str(variant.get("query") or "").strip()
                variant_surface = str(variant.get("surface") or source_surface or "youtube_html")
                if not search_query:
                    continue
                html_futures.append(
                    executor.submit(
                        self._search_variant_via_html,
                        search_query,
                        variant_surface,
                        target_pool,
                        video_duration,
                        retrieval_strategy,
                        retrieval_stage,
                        deadline,
                    )
                )
            for future in as_completed(html_futures):
                if self._deadline_exceeded(deadline) or self._network_backoff_active():
                    break
                try:
                    rows = future.result()
                except Exception:
                    rows = []
                if not rows:
                    continue
                results = self._merge_unique_videos(results, rows, target_pool)
                if len(results) >= target_pool:
                    break

        if include_external_fallbacks and len(results) < target_pool:
            per_variant_budget = max(4, min(20, target_pool // max(1, len(query_variants))))
            max_external_workers = max(1, min(self.SEARCH_SURFACE_WORKERS, len(query_variants) * 2))
            with ThreadPoolExecutor(max_workers=max_external_workers) as executor:
                future_map: dict[Any, tuple[str, str]] = {}
                for variant in query_variants:
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
                    ] = ("duckduckgo", search_query)
                    future_map[
                        executor.submit(
                            self._search_via_bing,
                            search_query,
                            per_variant_budget,
                            deadline,
                        )
                    ] = ("bing", search_query)

                for future in as_completed(future_map):
                    if self._deadline_exceeded(deadline) or self._network_backoff_active():
                        break
                    engine, search_query = future_map[future]
                    try:
                        ids = future.result()
                    except Exception:
                        ids = []
                    if not ids:
                        continue
                    if engine == "duckduckgo":
                        surface = "duckduckgo_quoted" if '"' in search_query else "duckduckgo_site"
                    else:
                        surface = "bing_quoted" if '"' in search_query else "bing_site"
                    fallback_rows = [self._fallback_video_row(video_id) for video_id in ids]
                    self._annotate_search_rows(
                        fallback_rows,
                        search_source=surface,
                        retrieval_strategy=retrieval_strategy,
                        retrieval_stage=retrieval_stage,
                        search_query=search_query,
                    )
                    results = self._merge_unique_videos(results, fallback_rows, target_pool)
                    if len(results) >= target_pool:
                        break

        return results[:max_results]

    def _search_variant_via_html(
        self,
        search_query: str,
        variant_surface: str,
        target_pool: int,
        video_duration: str | None,
        retrieval_strategy: str,
        retrieval_stage: str,
        deadline: float | None,
    ) -> list[dict[str, Any]]:
        if self._deadline_exceeded(deadline) or self._network_backoff_active():
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
        rows = self._merge_unique_videos(rows, parsed, target_pool)
        if len(rows) >= target_pool:
            return rows

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
            rows = self._merge_unique_videos(rows, fallback_rows, target_pool)
            if len(rows) >= target_pool:
                return rows

        if not initial_data:
            return rows

        api_key, client_version = self._extract_innertube_config(html)
        continuation_token = self._extract_search_continuation_token(initial_data)
        seen_tokens: set[str] = set()
        while (
            continuation_token
            and len(rows) < target_pool
            and len(seen_tokens) < max(1, self.HTML_MAX_PAGES - 1)
            and not self._deadline_exceeded(deadline)
            and not self._network_backoff_active()
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
            rows = self._merge_unique_videos(rows, continuation_rows, target_pool)
            continuation_token = self._extract_search_continuation_token(continuation_data)
        return rows

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

    def _build_search_query_variants(
        self,
        query: str,
        video_duration: str | None,
        source_surface: str,
    ) -> list[dict[str, str]]:
        base = " ".join(str(query or "").split()).strip()
        if not base:
            return []

        variants: list[tuple[str, str]] = [
            (base, source_surface or "youtube_html"),
            (f"{base} youtube", source_surface or "youtube_html"),
            (f"\"{base}\" site:youtube.com/watch", "duckduckgo_quoted"),
            (f"{base} b-roll footage scene site:youtube.com/watch", "duckduckgo_site"),
        ]
        if video_duration == "short":
            variants.extend(
                [
                    (f"{base} shorts", source_surface or "youtube_html"),
                    (f"{base} quick explainer", source_surface or "youtube_html"),
                    (f"{base} in 60 seconds", source_surface or "youtube_html"),
                    (f"{base} short tutorial", source_surface or "youtube_html"),
                    (f"{base} bite sized", source_surface or "youtube_html"),
                    (f"{base} quick tips", source_surface or "youtube_html"),
                ]
            )
        elif video_duration == "medium":
            variants.extend(
                [
                    (f"{base} tutorial", source_surface or "youtube_html"),
                    (f"{base} lesson", source_surface or "youtube_html"),
                    (f"{base} explained", source_surface or "youtube_html"),
                    (f"{base} walkthrough", source_surface or "youtube_html"),
                    (f"{base} examples", source_surface or "youtube_html"),
                ]
            )
        elif video_duration == "long":
            variants.extend(
                [
                    (f"{base} full lecture", source_surface or "youtube_html"),
                    (f"{base} deep dive", source_surface or "youtube_html"),
                    (f"{base} full course", source_surface or "youtube_html"),
                    (f"{base} complete tutorial", source_surface or "youtube_html"),
                    (f"{base} full class", source_surface or "youtube_html"),
                    (f"{base} masterclass", source_surface or "youtube_html"),
                ]
            )
        else:
            variants.extend(
                [
                    (f"{base} tutorial", source_surface or "youtube_html"),
                    (f"{base} lecture", source_surface or "youtube_html"),
                    (f"{base} full course", source_surface or "youtube_html"),
                    (f"{base} deep dive", source_surface or "youtube_html"),
                    (f"{base} worked examples", source_surface or "youtube_html"),
                    (f"{base} shorts", source_surface or "youtube_html"),
                    (f"{base} crash course", source_surface or "youtube_html"),
                    (f"{base} full class", source_surface or "youtube_html"),
                    (f"{base} complete guide", source_surface or "youtube_html"),
                    (f"{base} walkthrough", source_surface or "youtube_html"),
                    (f"{base} quick explainer", source_surface or "youtube_html"),
                    (f"{base} 60 seconds", source_surface or "youtube_html"),
                    (f"{base} fundamentals", source_surface or "youtube_html"),
                ]
            )

        deduped: list[dict[str, str]] = []
        seen: set[str] = set()
        for variant, variant_surface in variants:
            normalized = " ".join(variant.split()).strip()
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append({"query": normalized, "surface": variant_surface})
            if len(deduped) >= self.SEARCH_VARIANTS_LIMIT:
                break
        return deduped

    def _fetch_search_html(self, search_query: str, deadline: float | None = None) -> str:
        if self._deadline_exceeded(deadline):
            return ""
        try:
            resp = requests.get(
                "https://www.youtube.com/results",
                params={"search_query": search_query},
                timeout=self._request_timeout(deadline),
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
                    ),
                    "Accept-Language": "en-US,en;q=0.9",
                },
            )
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as exc:
            self._note_request_failure(exc)
            return ""

    def _fallback_video_row(self, video_id: str) -> dict[str, Any]:
        return {
            "id": video_id,
            "title": f"YouTube Video {video_id}",
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
        }

    def _merge_unique_videos(
        self,
        primary: list[dict[str, Any]],
        secondary: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        for source in (primary, secondary):
            for row in source:
                video_id = str(row.get("id") or "").strip()
                if not video_id or video_id in seen_ids:
                    continue
                seen_ids.add(video_id)
                merged.append(
                    {
                        "id": video_id,
                        "title": str(row.get("title") or "Untitled"),
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
                    }
                )
                if len(merged) >= limit:
                    return merged
        return merged

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
            video_id = renderer.get("videoId")
            if not isinstance(video_id, str) or video_id in seen_ids:
                continue

            duration_text = (
                (renderer.get("lengthText") or {}).get("simpleText")
                or self._first_run_text((renderer.get("lengthText") or {}).get("runs"))
                or ""
            )
            duration_sec = self._parse_duration_text(duration_text)
            if not self._duration_matches(duration_sec, video_duration):
                continue

            title = self._first_run_text((renderer.get("title") or {}).get("runs")) or "Untitled"
            channel = self._first_run_text((renderer.get("ownerText") or {}).get("runs"))
            detailed = renderer.get("detailedMetadataSnippets")
            snippet_runs: Any = None
            if isinstance(detailed, list) and detailed:
                first = detailed[0]
                if isinstance(first, dict):
                    snippet_runs = (first.get("snippetText") or {}).get("runs")
            snippet = self._runs_text(snippet_runs)
            if not snippet:
                snippet = self._runs_text((renderer.get("descriptionSnippet") or {}).get("runs"))

            rows.append(
                {
                    "id": video_id,
                    "title": title,
                    "channel_title": channel,
                    "description": snippet,
                    "duration_sec": duration_sec,
                    "view_count": 0,
                    "is_creative_commons": False,
                }
            )
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
        if not innertube_api_key or not continuation_token or self._deadline_exceeded(deadline):
            return None
        client_version = innertube_client_version or "2.20240214.00.00"
        try:
            resp = requests.post(
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
                timeout=self._request_timeout(deadline),
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
                    ),
                    "Accept-Language": "en-US,en;q=0.9",
                    "Content-Type": "application/json",
                    "Origin": "https://www.youtube.com",
                    "x-youtube-client-name": "1",
                    "x-youtube-client-version": client_version,
                },
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            self._note_request_failure(exc)
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
            video_renderer = node.get("videoRenderer")
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
            if self._deadline_exceeded(deadline) or self._network_backoff_active():
                break
            try:
                resp = requests.get(
                    "https://duckduckgo.com/html/",
                    params={"q": f"site:youtube.com/watch {query}", "s": offset},
                    timeout=self._request_timeout(deadline),
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
                        ),
                        "Accept-Language": "en-US,en;q=0.9",
                    },
                )
                resp.raise_for_status()
            except requests.RequestException as exc:
                self._note_request_failure(exc)
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
            if self._deadline_exceeded(deadline) or self._network_backoff_active():
                break
            try:
                resp = requests.get(
                    "https://www.bing.com/search",
                    params={
                        "q": f"site:youtube.com/watch {query}",
                        "count": 50,
                        "first": first,
                    },
                    timeout=self._request_timeout(deadline),
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
                        ),
                        "Accept-Language": "en-US,en;q=0.9",
                    },
                )
                resp.raise_for_status()
            except requests.RequestException as exc:
                self._note_request_failure(exc)
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
                if batch and not self._deadline_exceeded(deadline) and not self._network_backoff_active()
            ]
            for future in as_completed(futures):
                if self._deadline_exceeded(deadline) or self._network_backoff_active():
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
        if not batch or self._deadline_exceeded(deadline) or self._network_backoff_active():
            return {}
        params = {
            "key": self.api_key,
            "part": "contentDetails,status,statistics",
            "id": ",".join(batch),
            "maxResults": len(batch),
        }
        try:
            resp = requests.get(
                "https://www.googleapis.com/youtube/v3/videos",
                params=params,
                timeout=self._request_timeout(deadline),
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            self._note_request_failure(exc)
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

    def _network_backoff_active(self) -> bool:
        with self._network_backoff_lock:
            return time.monotonic() < self._network_backoff_until

    def _note_request_failure(self, exc: requests.RequestException) -> None:
        # Treat transport-level failures (DNS/connect/timeouts) as transient outages.
        if getattr(exc, "response", None) is not None:
            return
        with self._network_backoff_lock:
            self._network_backoff_until = max(
                self._network_backoff_until,
                time.monotonic() + self.NETWORK_BACKOFF_SEC,
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
