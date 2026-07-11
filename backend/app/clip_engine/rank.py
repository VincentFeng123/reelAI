"""Merge results across expanded queries, dedupe by video id, rank.
Strongest signal = trusted semantic query-family evidence, with literal first.
Port of practice/lib/rank.js.
"""
from __future__ import annotations

import math
import re

from .metadata import normalize_youtube_video_id

# -- Educational ranking signal ---------------------------------------------- #
# +1.0 per distinct boost hit, -1.5 per distinct penalty hit, clamped to ±3.0.

_BOOST_PATTERNS = [
    re.compile(r'\bexplained\b', re.I),
    re.compile(r'\bexplanation\b', re.I),
    re.compile(r'\blecture\b', re.I),
    re.compile(r'\bcourse\b', re.I),
    re.compile(r'\btutorial\b', re.I),
    re.compile(r'\blesson\b', re.I),
    re.compile(r'\bhow\b.{0,50}\bworks\b', re.I),
    re.compile(r'\bintroduction\b|\bintro\s+to\b', re.I),
    re.compile(r'\bbasics\b', re.I),
    re.compile(r'\bfundamentals\b', re.I),
    re.compile(r'\bprofessor\b', re.I),
    re.compile(r'\buniversity\b', re.I),
    re.compile(r'\bdocumentary\b', re.I),
    re.compile(r'\bcrash\s+course\b', re.I),
    re.compile(r'\bkhan\s+academy\b', re.I),
]

_PENALTY_PATTERNS = [
    re.compile(r'\breaction\b', re.I),
    re.compile(r'\bprank\b', re.I),
    re.compile(r'\bfunny\b', re.I),
    re.compile(r'\bmemes?\b', re.I),
    re.compile(r'\btop\s+10\b|\btop\s+ten\b', re.I),
    re.compile(r'\bcompilation\b', re.I),
    re.compile(r'\bchallenge\b', re.I),
    re.compile(r'\bvlog\b', re.I),
]


def _edu_score(v: dict) -> float:
    text = f"{v.get('title', '')} {v.get('channel', '')}"
    boost = sum(1.0 for p in _BOOST_PATTERNS if p.search(text))
    penalty = sum(1.5 for p in _PENALTY_PATTERNS if p.search(text))
    return max(-3.0, min(3.0, boost - penalty))


# -- Knowledge-level ranking signal ------------------------------------------ #
# +1.0 per distinct hit matching the viewer's band, -1.0 per hit in the
# opposite band, clamped to ±2.0. Added to `score` only — the
# (match_count, score, view_count) sort-key structure stays unchanged,
# same convention as _edu_score.

_BEGINNER_BAND = [
    re.compile(r'\bintro(?:duction)?\b', re.I),
    re.compile(r'\bbasics\b', re.I),
    re.compile(r'\bbeginners?\b', re.I),
    re.compile(r'\b101\b'),
    re.compile(r'\bcrash\s+course\b', re.I),
    re.compile(r'\bfor\s+dummies\b', re.I),
]

_ADVANCED_BAND = [
    re.compile(r'\badvanced\b', re.I),
    re.compile(r'\bgraduate\b', re.I),
    re.compile(r'\bseminar\b', re.I),
    re.compile(r'\bresearch\b', re.I),
    re.compile(r'\bproofs?\b', re.I),
    re.compile(r'\blecture\s+\d{2,3}\b', re.I),
]


def _level_score(v: dict, level: str | None) -> float:
    lvl = (level or "").strip().lower()
    if lvl == "beginner":
        match_band, opposite_band = _BEGINNER_BAND, _ADVANCED_BAND
    elif lvl == "advanced":
        match_band, opposite_band = _ADVANCED_BAND, _BEGINNER_BAND
    else:
        return 0.0
    text = f"{v.get('title', '')} {_channel_name(v)}"
    hits = sum(1.0 for p in match_band if p.search(text))
    misses = sum(1.0 for p in opposite_band if p.search(text))
    return max(-2.0, min(2.0, hits - misses))


def _channel_name(v: dict) -> str:
    ch = v.get("channel")
    if isinstance(ch, dict):
        return ch.get("name") or ch.get("title") or ""
    return ch or ""


def _video_id(value: object) -> str:
    normalized = normalize_youtube_video_id(value)
    if normalized:
        return normalized
    raw = str(value or "").strip()
    return raw[3:].strip() if raw.casefold().startswith("yt:") else raw


def _integer(value: object) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(float(value)))
    except (TypeError, ValueError, OverflowError):
        return 0


def _duration(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0 else None


def _first_nonblank(*values: object) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def merge_and_rank(per_query: list[dict], level: str | None = None) -> list[dict]:
    by_id: dict[str, dict] = {}
    for res in per_query or []:
        query_family = str(res.get("query_family") or "").strip()
        if not query_family:
            from ..services.search_query_plan import semantic_query_family

            query_family = semantic_query_family(res.get("query") or "")
        query_trust = str(res.get("query_trust") or "trusted").strip().casefold()
        filters_applied = res.get("filters_applied") or {}
        hd_match = bool(
            res.get("hd_preferred")
            or "hd" in (filters_applied.get("features") or [])
        )
        seen_in_query: set[str] = set()
        for rank, v in enumerate(res.get("videos") or []):
            vid = _video_id(v.get("id") or v.get("videoId") or v.get("url"))
            if not vid or vid in seen_in_query:
                continue
            seen_in_query.add(vid)
            entry = by_id.get(vid)
            channel = v.get("channel") if isinstance(v.get("channel"), dict) else {}
            channel_name = _first_nonblank(
                _channel_name(v), v.get("channelTitle"), v.get("author_name")
            )
            channel_id = _first_nonblank(
                v.get("channelId"), v.get("channel_id"), channel.get("id")
            )
            channel_url = _first_nonblank(
                v.get("channelUrl"), v.get("channel_url"), channel.get("url")
            )
            title = _first_nonblank(v.get("title"))
            thumbnail = _first_nonblank(
                v.get("thumbnail"), v.get("thumbnailUrl"), v.get("thumbnail_url")
            )
            description = _first_nonblank(v.get("description"))
            published_at = _first_nonblank(
                v.get("publishedAt"), v.get("published_at"), v.get("uploadDate"), v.get("upload_date")
            )
            duration = _duration(v.get("duration", v.get("duration_sec")))
            view_count = _integer(v.get("viewCount", v.get("view_count")))
            if entry is None:
                entry = {
                    "id": vid,
                    "title": title or "(untitled)",
                    "description": description,
                    "channel": channel_name,
                    "channel_id": channel_id,
                    "channel_url": channel_url,
                    "thumbnail": thumbnail,
                    "duration": duration,
                    "view_count": view_count,
                    "upload_date": published_at,
                    "published_at": published_at,
                    "url": f"https://www.youtube.com/watch?v={vid}",
                    "match_count": 0,
                    "best_rank": rank,
                    "matched_queries": [],
                    "matched_query_provenance": {},
                    "matched_families": [],
                    "_family_trust": {},
                    "literal_match": False,
                    "canonical_match": False,
                    "trusted_match_count": 0,
                    "ai_match_count": 0,
                    "hd_match": False,
                }
                by_id[vid] = entry
            else:
                if entry["title"] == "(untitled)" and title:
                    entry["title"] = title
                for field, value in (
                    ("description", description),
                    ("channel", channel_name),
                    ("channel_id", channel_id),
                    ("channel_url", channel_url),
                    ("thumbnail", thumbnail),
                    ("upload_date", published_at),
                    ("published_at", published_at),
                ):
                    if not entry.get(field) and value:
                        entry[field] = value
                if entry.get("duration") is None and duration is not None:
                    entry["duration"] = duration
                entry["view_count"] = max(int(entry.get("view_count") or 0), view_count)
            previous_trust = str(entry["_family_trust"].get(query_family) or "")
            if not previous_trust:
                entry["matched_families"].append(query_family)
                entry["_family_trust"][query_family] = query_trust
                entry["match_count"] += 1
                if query_trust == "ai":
                    entry["ai_match_count"] += 1
                else:
                    entry["trusted_match_count"] += 1
            elif previous_trust == "ai" and query_trust != "ai":
                entry["_family_trust"][query_family] = query_trust
                entry["ai_match_count"] = max(0, int(entry["ai_match_count"]) - 1)
                entry["trusted_match_count"] += 1
            if query_trust == "literal":
                entry["literal_match"] = True
            elif query_trust == "canonical":
                entry["canonical_match"] = True
            if hd_match:
                entry["hd_match"] = True
            entry["best_rank"] = min(entry["best_rank"], rank)
            q = res.get("query")
            if q and q not in entry["matched_queries"]:
                entry["matched_queries"].append(q)
            if q:
                entry["matched_query_provenance"][str(q)] = str(
                    res.get("query_provenance") or query_trust
                )

    items = list(by_id.values())
    for v in items:
        view_score = math.log10((v["view_count"] or 0) + 10)
        rank_score = 1 / (1 + v["best_rank"])
        edu = _edu_score(v)
        v["edu_score"] = edu
        anchor_bonus = 8.0 if v["literal_match"] else (5.0 if v["canonical_match"] else 0.0)
        # AI-only query families help discovery but cannot overwhelm literal,
        # canonical, or independently trusted evidence.
        ai_boost = min(0.75, 0.25 * int(v["ai_match_count"]))
        v["score"] = (
            int(v["trusted_match_count"]) * 10
            + anchor_bonus
            + ai_boost
            + view_score
            + rank_score * 2
            + edu
            + _level_score(v, level)
        )
    items.sort(
        key=lambda v: (
            bool(v["literal_match"]),
            bool(v["canonical_match"]),
            int(v["trusted_match_count"]),
            v["score"],
            bool(v["hd_match"]),
            v["view_count"],
        ),
        reverse=True,
    )
    for v in items:
        v.pop("best_rank", None)
        v.pop("_family_trust", None)
    return items
