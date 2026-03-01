import hashlib
import json
import re
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
from ..db import dumps_json, fetch_one, now_iso, upsert


class YouTubeApiRequestError(RuntimeError):
    pass


def _cache_key(*parts: str) -> str:
    value = "|".join(parts)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parse_iso8601_duration(value: str) -> int:
    if not value:
        return 0
    match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", value)
    if not match:
        return 0
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    return hours * 3600 + minutes * 60 + seconds


class YouTubeService:
    def __init__(self) -> None:
        settings = get_settings()
        self.api_key = settings.youtube_api_key
        self.transcript_api = YouTubeTranscriptApi()
        self.empty_transcript_ttl_sec = 6 * 60 * 60

    def search_videos(
        self,
        conn,
        query: str,
        max_results: int = 8,
        creative_commons_only: bool = False,
        video_duration: str | None = None,
    ) -> list[dict[str, Any]]:
        duration_key = video_duration or "any"
        key = _cache_key(query, str(max_results), str(creative_commons_only), duration_key)
        cached = fetch_one(conn, "SELECT response_json FROM search_cache WHERE cache_key = ?", (key,))
        if cached:
            return json.loads(cached["response_json"])

        videos: list[dict[str, Any]] = []
        if self.api_key:
            try:
                videos = self._search_via_data_api(
                    query=query,
                    max_results=max_results,
                    creative_commons_only=creative_commons_only,
                    video_duration=video_duration,
                )
            except YouTubeApiRequestError:
                videos = self._search_without_data_api(
                    query=query,
                    max_results=max_results,
                    creative_commons_only=creative_commons_only,
                    video_duration=video_duration,
                )
        else:
            videos = self._search_without_data_api(
                query=query,
                max_results=max_results,
                creative_commons_only=creative_commons_only,
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
        return videos

    def _search_via_data_api(
        self,
        query: str,
        max_results: int,
        creative_commons_only: bool,
        video_duration: str | None,
    ) -> list[dict[str, Any]]:
        params = {
            "key": self.api_key,
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": max_results,
            "safeSearch": "moderate",
            "videoEmbeddable": "true",
            "relevanceLanguage": "en",
        }
        if creative_commons_only:
            params["videoLicense"] = "creativeCommon"
        if video_duration in {"short", "medium", "long"}:
            params["videoDuration"] = video_duration

        try:
            resp = requests.get("https://www.googleapis.com/youtube/v3/search", params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 403:
                raise YouTubeApiRequestError(
                    "YouTube Data API request was rejected (403). Check YOUTUBE_API_KEY, API restrictions, and quota."
                ) from exc
            raise YouTubeApiRequestError(
                f"YouTube Data API request failed{f' ({status})' if status else ''}."
            ) from exc

        ids = [item["id"]["videoId"] for item in data.get("items", []) if item.get("id", {}).get("videoId")]
        details = self._video_details(ids)

        videos: list[dict[str, Any]] = []
        for item in data.get("items", []):
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
                    "is_creative_commons": bool(is_cc),
                }
            )
        return videos

    def _search_without_data_api(
        self,
        query: str,
        max_results: int,
        creative_commons_only: bool,
        video_duration: str | None,
    ) -> list[dict[str, Any]]:
        # License cannot be reliably verified without the Data API.
        if creative_commons_only:
            return []

        search_query = query
        if video_duration == "short" and "shorts" not in search_query.lower():
            search_query = f"{search_query} shorts"

        try:
            resp = requests.get(
                "https://www.youtube.com/results",
                params={"search_query": search_query},
                timeout=20,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
                    ),
                    "Accept-Language": "en-US,en;q=0.9",
                },
            )
            resp.raise_for_status()
        except requests.RequestException:
            return []

        html = resp.text
        videos = self._extract_videos_from_search_html(html, max_results=max_results, video_duration=video_duration)
        if videos:
            return videos[:max_results]

        ids = []
        seen_ids = set()
        for video_id in re.findall(r'"videoId":"([A-Za-z0-9_-]{11})"', html):
            if video_id in seen_ids:
                continue
            seen_ids.add(video_id)
            ids.append(video_id)
            if len(ids) >= max_results:
                break

        if not ids:
            ids = self._search_via_duckduckgo(query=search_query, max_results=max_results)

        return [
            {
                "id": video_id,
                "title": f"YouTube Video {video_id}",
                "channel_title": "",
                "description": "",
                "duration_sec": 0,
                "is_creative_commons": False,
            }
            for video_id in ids
        ]

    def _extract_videos_from_search_html(
        self,
        html: str,
        max_results: int,
        video_duration: str | None,
    ) -> list[dict[str, Any]]:
        data = self._extract_yt_initial_data(html)
        if not data:
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
            snippet = self._runs_text((renderer.get("detailedMetadataSnippets") or [{}])[0].get("snippetText", {}).get("runs"))
            if not snippet:
                snippet = self._runs_text((renderer.get("descriptionSnippet") or {}).get("runs"))

            rows.append(
                {
                    "id": video_id,
                    "title": title,
                    "channel_title": channel,
                    "description": snippet,
                    "duration_sec": duration_sec,
                    "is_creative_commons": False,
                }
            )
            seen_ids.add(video_id)
            if len(rows) >= max_results:
                break
        return rows

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

    def _search_via_duckduckgo(self, query: str, max_results: int) -> list[str]:
        try:
            resp = requests.get(
                "https://duckduckgo.com/html/",
                params={"q": f"site:youtube.com/watch {query}"},
                timeout=20,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
                    ),
                    "Accept-Language": "en-US,en;q=0.9",
                },
            )
            resp.raise_for_status()
        except requests.RequestException:
            return []

        html = resp.text
        candidates = re.findall(r'href="([^"]+)"', html)
        ids: list[str] = []
        seen: set[str] = set()

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
                break
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

    def _video_details(self, video_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not self.api_key or not video_ids:
            return {}
        params = {
            "key": self.api_key,
            "part": "contentDetails,status",
            "id": ",".join(video_ids),
            "maxResults": len(video_ids),
        }
        try:
            resp = requests.get("https://www.googleapis.com/youtube/v3/videos", params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException:
            return {}

        result: dict[str, dict[str, Any]] = {}
        for item in data.get("items", []):
            vid = item.get("id")
            if not vid:
                continue
            duration = parse_iso8601_duration(item.get("contentDetails", {}).get("duration", ""))
            result[vid] = {
                "duration_sec": duration,
                "license": item.get("status", {}).get("license", "youtube"),
            }
        return result

    def get_transcript(self, conn, video_id: str) -> list[dict[str, Any]]:
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
