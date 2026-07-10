"""Discovery: topic -> expand -> Supadata search -> rank -> exclude -> top N."""
from __future__ import annotations

from collections.abc import Callable

from . import expand, rank, supadata_search
from .cancellation import raise_if_cancelled
from .errors import SearchError
from .metadata import normalize_youtube_video_id
from .provider_cache import ProviderCacheStore
from .provider_runtime import GenerationContext


def _consensus_count(per_query: list[dict], excluded: set[str]) -> int:
    appearances: dict[str, int] = {}
    for result_set in per_query:
        seen_in_query: set[str] = set()
        for video in result_set.get("videos") or []:
            raw_id = video.get("id") or video.get("videoId") or video.get("url")
            video_id = normalize_youtube_video_id(raw_id) or str(raw_id or "").strip()
            if not video_id or video_id in excluded or video_id in seen_in_query:
                continue
            seen_in_query.add(video_id)
            appearances[video_id] = appearances.get(video_id, 0) + 1
    return sum(1 for count in appearances.values() if count >= 2)


def discover(topic: str, limit: int, exclude_video_ids: list[str] | None = None,
             breadth: int | None = None, level: str | None = None,
             should_cancel: Callable[[], bool] | None = None,
             *, filters: dict | None = None, language: str = "en",
             context: GenerationContext | None = None,
             cache_store: ProviderCacheStore | None = None) -> dict:
    topic = " ".join(str(topic or "").split())
    if not topic:
        raise SearchError("topic must contain non-whitespace text")
    if context is not None and context.budget.mode == "fast":
        n = 3
    elif context is not None:
        # Slow jobs spend six searches on their initial acquisition pass and
        # three on each of the two possible continuations (6 + 3 + 3 = 12).
        pass_count = int(context.budget.snapshot().get("passes") or 0)
        n = 6 if pass_count <= 1 else min(3, context.budget.remaining("search"))
    else:
        n = max(1, min(6, int(breadth or 6)))
    if n <= 0:
        raise SearchError("search budget exhausted")
    raise_if_cancelled(should_cancel)
    expansion = (
        expand.expand_query(topic, n, level=level)
        if should_cancel is None
        else expand.expand_query(topic, n, level=level, should_cancel=should_cancel)
    )
    exclude = {
        normalize_youtube_video_id(video_id) or str(video_id)
        for video_id in (exclude_video_ids or [])
    }
    res = supadata_search.search_all(
        expansion["queries"][:n],
        filters,
        minimum_queries=(
            n if context is not None and context.budget.mode == "slow"
            else min(3, len(expansion["queries"][:n]))
        ),
        stop_when=lambda per_query: _consensus_count(per_query, exclude) >= limit,
        should_cancel=should_cancel,
        language=language,
        context=context,
        cache_store=cache_store,
    )
    raise_if_cancelled(should_cancel)
    ranked = rank.merge_and_rank(res["per_query"], level=level)
    videos = [v for v in ranked if v["id"] not in exclude][:limit]
    return {"corrected": expansion["corrected"], "videos": videos,
            "credits_used": res["credits_used"], "warning": res["warning"]}
