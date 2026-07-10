"""Discovery: topic -> expand -> Supadata search -> rank -> exclude -> top N."""
from __future__ import annotations

from collections.abc import Callable

from . import expand, rank, supadata_search
from .cancellation import raise_if_cancelled


def _consensus_count(per_query: list[dict], excluded: set[str]) -> int:
    appearances: dict[str, int] = {}
    for result_set in per_query:
        seen_in_query: set[str] = set()
        for video in result_set.get("videos") or []:
            video_id = str(video.get("id") or "").strip()
            if not video_id or video_id in excluded or video_id in seen_in_query:
                continue
            seen_in_query.add(video_id)
            appearances[video_id] = appearances.get(video_id, 0) + 1
    return sum(1 for count in appearances.values() if count >= 2)


def discover(topic: str, limit: int, exclude_video_ids: list[str] | None = None,
             breadth: int | None = None, level: str | None = None,
             should_cancel: Callable[[], bool] | None = None) -> dict:
    # `breadth` remains in the signature for old callers, but the shipping
    # contract is fixed: one expansion call yields six queries.
    n = 6
    raise_if_cancelled(should_cancel)
    expansion = (
        expand.expand_query(topic, n, level=level)
        if should_cancel is None
        else expand.expand_query(topic, n, level=level, should_cancel=should_cancel)
    )
    exclude = set(exclude_video_ids or [])
    res = supadata_search.search_all(
        expansion["queries"],
        minimum_queries=min(3, len(expansion["queries"])),
        stop_when=lambda per_query: _consensus_count(per_query, exclude) >= limit,
        should_cancel=should_cancel,
    )
    raise_if_cancelled(should_cancel)
    ranked = rank.merge_and_rank(res["per_query"], level=level)
    videos = [v for v in ranked if v["id"] not in exclude][:limit]
    return {"corrected": expansion["corrected"], "videos": videos,
            "credits_used": res["credits_used"], "warning": res["warning"]}
