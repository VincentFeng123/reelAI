"""Merge results across expanded queries, dedupe by video id, rank.
Strongest signal = match_count (how many queries surfaced the video).
Port of practice/lib/rank.js.
"""
from __future__ import annotations

import math


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
        v["score"] = v["match_count"] * 10 + view_score + rank_score * 2
    items.sort(key=lambda v: (v["match_count"], v["score"], v["view_count"]), reverse=True)
    for v in items:
        v.pop("best_rank", None)
    return items
