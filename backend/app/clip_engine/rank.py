"""Merge results across expanded queries, dedupe by video id, rank.
Strongest signal = match_count (how many queries surfaced the video).
Port of practice/lib/rank.js.
"""
from __future__ import annotations

import math
import re

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


def _channel_name(v: dict) -> str:
    ch = v.get("channel")
    if isinstance(ch, dict):
        return ch.get("name") or ch.get("title") or ""
    return ch or ""


def merge_and_rank(per_query: list[dict]) -> list[dict]:
    by_id: dict[str, dict] = {}
    for res in per_query or []:
        for rank, v in enumerate(res.get("videos") or []):
            vid = v.get("id")
            if not vid:
                continue
            entry = by_id.get(vid)
            if entry is None:
                vc = v.get("viewCount")
                entry = {
                    "id": vid,
                    "title": v.get("title") or "(untitled)",
                    "channel": _channel_name(v),
                    "thumbnail": v.get("thumbnail") or "",
                    "duration": v.get("duration") if isinstance(v.get("duration"), (int, float)) else None,
                    "view_count": vc if isinstance(vc, (int, float)) else (int(vc) if str(vc or "").isdigit() else 0),
                    "upload_date": v.get("uploadDate") or "",
                    "url": f"https://www.youtube.com/watch?v={vid}",
                    "match_count": 0,
                    "best_rank": rank,
                    "matched_queries": [],
                }
                by_id[vid] = entry
            entry["match_count"] += 1
            entry["best_rank"] = min(entry["best_rank"], rank)
            q = res.get("query")
            if q and q not in entry["matched_queries"]:
                entry["matched_queries"].append(q)

    items = list(by_id.values())
    for v in items:
        view_score = math.log10((v["view_count"] or 0) + 10)
        rank_score = 1 / (1 + v["best_rank"])
        edu = _edu_score(v)
        v["edu_score"] = edu
        v["score"] = v["match_count"] * 10 + view_score + rank_score * 2 + edu
    items.sort(key=lambda v: (v["match_count"], v["score"], v["view_count"]), reverse=True)
    for v in items:
        v.pop("best_rank", None)
    return items
