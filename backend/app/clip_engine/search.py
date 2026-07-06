"""Discovery: topic -> expand -> Supadata search -> rank -> exclude -> top N."""
from __future__ import annotations

from . import config, expand, rank, supadata_search


def discover(topic: str, limit: int, exclude_video_ids: list[str] | None = None,
             breadth: int | None = None) -> dict:
    n = max(1, breadth or config.SEARCH_BREADTH)
    expansion = expand.expand_query(topic, n)
    res = supadata_search.search_all(expansion["queries"])
    ranked = rank.merge_and_rank(res["per_query"])
    exclude = set(exclude_video_ids or [])
    videos = [v for v in ranked if v["id"] not in exclude][:limit]
    return {"corrected": expansion["corrected"], "videos": videos,
            "credits_used": res["credits_used"], "warning": res["warning"]}
