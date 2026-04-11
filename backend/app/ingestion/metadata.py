"""
Map yt-dlp info_dict → unified IngestMetadata, plus summary helpers.

yt-dlp emits wildly different keys per extractor. For Instagram you get `like_count`,
`comment_count`, `uploader`; for TikTok you additionally get `repost_count` and `track`;
for YouTube you get `tags`, `categories`, `view_count`, etc. This mapper normalizes them
into a single shape the rest of the pipeline can consume.

The `fallback_ai_summary` body is copied verbatim from `app/services/reels.py:9171-9197` —
it's a pure function already, and copying avoids importing the 9,921-line ReelService.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from ..db import DatabaseIntegrityError, dumps_json, fetch_one, now_iso, upsert
from ..services.concepts import build_takeaways
from .logging_config import get_ingest_logger
from .models import IngestMetadata, PlatformLiteral

logger: logging.Logger = get_ingest_logger(__name__)


_HASHTAG_RE = re.compile(r"#([A-Za-z0-9_\u00c0-\u024f][A-Za-z0-9_\u00c0-\u024f]{1,48})")
_AI_SUMMARY_CACHE_PREFIX = "ingest_ai_summary:"
_AI_SUMMARY_TTL_SEC = 7 * 24 * 3600  # 7 days
_SUMMARY_MODEL = "gpt-4o-mini"
_SUMMARY_MAX_TOKENS = 180


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _clean_description(raw: str, max_chars: int = 4000) -> str:
    if not raw:
        return ""
    cleaned = raw.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip() + "…"
    return cleaned


def _parse_upload_date(raw: Any) -> str | None:
    """
    yt-dlp's `upload_date` is `YYYYMMDD`; some extractors give `timestamp` as epoch seconds.
    Return an ISO-8601 date string (`YYYY-MM-DD`) or None.
    """
    if isinstance(raw, str) and len(raw) == 8 and raw.isdigit():
        return f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]}"
    if isinstance(raw, (int, float)):
        try:
            import datetime as _dt

            return _dt.datetime.fromtimestamp(float(raw), tz=_dt.timezone.utc).date().isoformat()
        except Exception:
            return None
    return None


def _extract_hashtags(description: str, fallback_tags: list[Any] | None) -> list[str]:
    tags: list[str] = []
    seen: set[str] = set()
    for match in _HASHTAG_RE.finditer(description or ""):
        tag = match.group(1).lower()
        if tag not in seen:
            tags.append(tag)
            seen.add(tag)
    if fallback_tags:
        for item in fallback_tags:
            if not isinstance(item, str):
                continue
            candidate = item.strip().lstrip("#").lower()
            if candidate and candidate not in seen:
                tags.append(candidate)
                seen.add(candidate)
    return tags[:32]


def map_info_dict_to_metadata(
    info: dict[str, Any],
    platform: PlatformLiteral,
    *,
    source_url: str,
    source_id: str,
    playback_url: str,
) -> IngestMetadata:
    """
    Normalize a yt-dlp info_dict into `IngestMetadata`. Never raises — missing fields
    become empty strings / None. The `is_private` / `is_live` flags are best-effort
    derived from the info dict.
    """
    description_raw = _as_str(info.get("description"))
    description = _clean_description(description_raw)

    title = _as_str(info.get("title") or info.get("fulltitle"))
    author_name = _as_str(info.get("uploader") or info.get("channel") or info.get("creator"))
    author_handle = _as_str(
        info.get("uploader_id") or info.get("channel_id") or info.get("creator_id")
    ).lstrip("@")
    author_url = _as_str(
        info.get("uploader_url") or info.get("channel_url")
    )

    duration_sec = info.get("duration")
    duration_val: float | None = None
    if isinstance(duration_sec, (int, float)) and duration_sec > 0:
        duration_val = float(duration_sec)

    thumbnail = _as_str(info.get("thumbnail"))

    upload_iso = _parse_upload_date(info.get("upload_date") or info.get("timestamp"))

    view_count = _as_int(info.get("view_count"))
    like_count = _as_int(info.get("like_count"))
    comment_count = _as_int(info.get("comment_count"))
    repost_count = _as_int(info.get("repost_count") or info.get("share_count"))

    hashtags = _extract_hashtags(description_raw, info.get("tags") if isinstance(info.get("tags"), list) else None)
    categories_raw = info.get("categories")
    categories = [str(c) for c in categories_raw if isinstance(c, str)] if isinstance(categories_raw, list) else []

    # TikTok: music track metadata.
    audio_title = ""
    audio_artist = ""
    track = info.get("track")
    artist = info.get("artist") or info.get("creator")
    if isinstance(track, str):
        audio_title = track
    if isinstance(artist, str):
        audio_artist = artist
    # Instagram: music info is occasionally under `music_info`
    music_info = info.get("music_info")
    if isinstance(music_info, dict):
        audio_title = audio_title or _as_str(music_info.get("song_name"))
        audio_artist = audio_artist or _as_str(music_info.get("artist_name"))

    language = _as_str(info.get("language") or info.get("subtitles_language") or "")

    location = ""
    loc_raw = info.get("location")
    if isinstance(loc_raw, str):
        location = loc_raw
    elif isinstance(loc_raw, dict):
        location = _as_str(loc_raw.get("name") or loc_raw.get("title"))

    is_private = bool(info.get("is_private") or info.get("was_private"))
    is_live = bool(info.get("is_live"))

    return IngestMetadata(
        platform=platform,
        source_id=source_id,
        source_url=source_url,
        playback_url=playback_url,
        title=title,
        description=description,
        author_handle=author_handle,
        author_name=author_name,
        author_url=author_url,
        duration_sec=duration_val,
        thumbnail_url=thumbnail,
        upload_date_iso=upload_iso,
        view_count=view_count,
        like_count=like_count,
        comment_count=comment_count,
        repost_count=repost_count,
        hashtags=hashtags,
        categories=categories,
        audio_title=audio_title,
        audio_artist=audio_artist,
        language=language,
        location=location,
        is_private=is_private,
        is_live=is_live,
    )


def format_attribution(metadata: IngestMetadata) -> str:
    """
    Build a short attribution line the iOS UI can render.
    Example: `"@nasa on Instagram"` or `"NASA on YouTube"` or `"tiktok.com"` fallback.
    """
    platform_name = {"yt": "YouTube", "ig": "Instagram", "tt": "TikTok"}.get(metadata.platform, metadata.platform)
    if metadata.author_handle:
        return f"@{metadata.author_handle} on {platform_name}"
    if metadata.author_name:
        return f"{metadata.author_name} on {platform_name}"
    return platform_name


def fallback_ai_summary(
    *,
    concept_title: str,
    video_title: str,
    video_description: str,
    transcript_snippet: str,
    takeaways: list[str],
) -> str:
    """
    Pure fallback summary. Copied verbatim from `reels.py:9171-9197` with `self` removed.
    Returns a 1-2 sentence string <= 320 chars ending in a period.
    """
    takeaway_text = " ".join(t.strip() for t in takeaways if t.strip())
    candidates = [transcript_snippet.strip(), takeaway_text.strip(), video_description.strip()]
    source = next((c for c in candidates if c), "")
    if not source:
        return f"Brief overview of {concept_title or video_title or 'this reel'}."

    compact = " ".join(source.split())
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", compact) if s.strip()]
    if not sentences:
        summary = compact
    elif len(sentences) == 1:
        summary = sentences[0]
    else:
        summary = f"{sentences[0]} {sentences[1]}"
    summary = summary[:320].strip()
    if summary and summary[-1] not in ".!?":
        summary = f"{summary}."
    return summary


def brief_ai_summary(
    conn: Any,
    *,
    openai_client: Any,
    concept_title: str,
    video_title: str,
    video_description: str,
    transcript_snippet: str,
    takeaways: list[str],
    cache_key_suffix: str,
) -> str:
    """
    LLM-backed summary. Uses the `llm_cache` table with key `ingest_ai_summary:{suffix}`
    so identical inputs never pay the LLM cost twice. Falls back to `fallback_ai_summary`
    on any error — summary is cosmetic, we never want it to fail the ingest.

    Does NOT touch `reels.py`. Does NOT instantiate a new OpenAI client — the caller
    passes one that was built via `openai_client.build_openai_client(...)`.
    """
    fallback = fallback_ai_summary(
        concept_title=concept_title,
        video_title=video_title,
        video_description=video_description,
        transcript_snippet=transcript_snippet,
        takeaways=takeaways,
    )
    if openai_client is None:
        return fallback

    cache_key = f"{_AI_SUMMARY_CACHE_PREFIX}{cache_key_suffix}"
    try:
        cached = fetch_one(
            conn,
            "SELECT response_json, created_at FROM llm_cache WHERE cache_key = ?",
            (cache_key,),
        )
        if cached and cached.get("response_json"):
            try:
                payload = json.loads(cached["response_json"])
                if isinstance(payload, dict) and isinstance(payload.get("summary"), str):
                    return payload["summary"]
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
    except Exception:
        logger.exception("brief_ai_summary: llm_cache read failed")

    try:
        prompt = (
            "You write concise, factual one-paragraph summaries of short-form video content "
            "for an educational study app. Stick to what's in the transcript. Use 1-2 sentences, "
            "<= 320 characters, plain declarative voice, no emojis, no hashtags.\n\n"
            f"Video title: {video_title or '(untitled)'}\n"
            f"Author-provided description: {video_description[:600] if video_description else '(none)'}\n"
            f"Transcript excerpt: {transcript_snippet[:1400] if transcript_snippet else '(none)'}"
        )
        completion = openai_client.chat.completions.create(
            model=_SUMMARY_MODEL,
            messages=[
                {"role": "system", "content": "Return only the summary paragraph, no preamble."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=_SUMMARY_MAX_TOKENS,
            temperature=0.2,
        )
        raw = (completion.choices[0].message.content or "").strip() if completion.choices else ""
        if not raw:
            return fallback
        summary = " ".join(raw.split())[:320]
        if summary and summary[-1] not in ".!?":
            summary = summary + "."
    except Exception:
        logger.warning("brief_ai_summary: OpenAI call failed, falling back")
        return fallback

    try:
        upsert(
            conn,
            "llm_cache",
            {
                "cache_key": cache_key,
                "response_json": dumps_json({"summary": summary}),
                "created_at": now_iso(),
            },
            pk="cache_key",
        )
    except DatabaseIntegrityError:
        pass
    except Exception:
        logger.exception("brief_ai_summary: llm_cache upsert failed")

    return summary


def build_takeaways_for_ingest(
    *,
    concept_title: str,
    transcript_snippet: str,
    hashtags: list[str],
    limit: int = 3,
) -> list[str]:
    """
    Thin wrapper around `concepts.build_takeaways` that synthesizes a concept dict when
    we're running in standalone mode (no real concept from the material).
    """
    concept = {
        "title": concept_title or "",
        "keywords": [tag for tag in hashtags if tag][:6],
        "summary": "",
    }
    return build_takeaways(concept, transcript_snippet or "", limit=limit)


__all__ = [
    "map_info_dict_to_metadata",
    "format_attribution",
    "fallback_ai_summary",
    "brief_ai_summary",
    "build_takeaways_for_ingest",
]
