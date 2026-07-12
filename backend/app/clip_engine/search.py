"""Discovery: topic -> expand -> Supadata search -> rank -> exclude -> top N."""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from . import rank, supadata_search
from .cancellation import raise_if_cancelled
from .errors import SearchError
from .metadata import normalize_youtube_video_id
from .provider_cache import ProviderCacheStore, normalize_filters
from .provider_runtime import GenerationContext

if TYPE_CHECKING:
    from ..services.search_query_plan import SearchQueryPlan

logger = logging.getLogger(__name__)


def _request_filters(filters: dict | None, *, prefer_hd: bool) -> dict:
    """Keep mandatory filters while making HD a per-request preference."""
    normalized = normalize_filters(filters)
    features = [
        feature
        for feature in normalized.get("features") or []
        if feature == "creative-commons"
    ]
    if prefer_hd:
        features.append("hd")
    request_filters = {**normalized}
    request_filters.pop("features", None)
    if features:
        request_filters["features"] = features
    return request_filters


def _planned_query_offset(context: GenerationContext | None) -> int:
    """Count AI-term slots consumed by earlier slow acquisition passes."""
    if context is None or context.budget.mode != "slow":
        return 0
    pass_count = max(1, int(context.budget.snapshot().get("passes") or 0))
    if pass_count <= 1:
        return 0
    # Every pass starts with one unrestricted primary query. The six-call
    # initial pass therefore consumes five AI terms, and each earlier
    # three-call continuation consumes two more.
    return 5 + max(0, pass_count - 2) * 2


def _select_ranked_candidates(ranked: list[dict], *, limit: int, excluded: set[str]) -> list[dict]:
    """Keep literal priority while reserving bounded room for AI expansion."""
    eligible = [video for video in ranked if video.get("id") not in excluded]
    if limit <= 1 or len(eligible) <= limit:
        return eligible[:limit]

    non_literal = [
        video
        for video in eligible
        if not video.get("literal_match")
    ]
    if not non_literal:
        return eligible[:limit]

    reserve = min(2, max(1, limit // 3))
    selected = eligible[: max(0, limit - reserve)]
    selected_ids = {str(video.get("id") or "") for video in selected}
    selected_families = {
        str(family)
        for video in selected
        if not video.get("literal_match")
        for family in (video.get("matched_families") or [])
        if str(family or "").strip()
    }
    for video in non_literal:
        video_id = str(video.get("id") or "")
        families = {
            str(family)
            for family in (video.get("matched_families") or [])
            if str(family or "").strip()
        }
        if video_id in selected_ids or (families and families.issubset(selected_families)):
            continue
        selected.append(video)
        selected_ids.add(video_id)
        selected_families.update(families)
        if len(selected) >= limit:
            break
    for video in eligible:
        video_id = str(video.get("id") or "")
        if len(selected) >= limit:
            break
        if video_id not in selected_ids:
            selected.append(video)
            selected_ids.add(video_id)
    return selected[:limit]


def _consensus_count(per_query: list[dict], excluded: set[str]) -> int:
    appearances: dict[str, set[str]] = {}
    for result_set in per_query:
        family = str(result_set.get("query_family") or "").strip()
        if not family:
            from ..services.search_query_plan import semantic_query_family

            family = semantic_query_family(result_set.get("query") or "")
        seen_in_query: set[str] = set()
        for video in result_set.get("videos") or []:
            raw_id = video.get("id") or video.get("videoId") or video.get("url")
            video_id = normalize_youtube_video_id(raw_id) or str(raw_id or "").strip()
            if not video_id or video_id in excluded or video_id in seen_in_query:
                continue
            seen_in_query.add(video_id)
            appearances.setdefault(video_id, set()).add(family)
    return sum(1 for families in appearances.values() if len(families) >= 2)


def _load_query_plan(
    literal_topic: str,
    should_cancel: Callable[[], bool] | None,
) -> "SearchQueryPlan":
    from ..db import get_conn
    from ..services.search_query_plan import build_search_query_plan

    try:
        with get_conn() as conn:
            return build_search_query_plan(
                conn,
                literal_query=literal_topic,
                should_cancel=should_cancel,
            )
    except Exception:
        raise_if_cancelled(should_cancel)
        logger.warning("search query plan cache unavailable; using uncached validated plan")
        return build_search_query_plan(
            None,
            literal_query=literal_topic,
            should_cancel=should_cancel,
        )


def discover(topic: str, limit: int, exclude_video_ids: list[str] | None = None,
             breadth: int | None = None, level: str | None = None,
             should_cancel: Callable[[], bool] | None = None,
             *, filters: dict | None = None, language: str = "en",
             context: GenerationContext | None = None,
             cache_store: ProviderCacheStore | None = None,
             literal_topic: str | None = None,
             use_query_planner: bool = True,
             query_plan: "SearchQueryPlan | None" = None) -> dict:
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
    effective_plan = query_plan
    if effective_plan is None and use_query_planner:
        effective_plan = _load_query_plan(
            " ".join(str(literal_topic or topic).split()),
            should_cancel,
        )

    planned_queries = []
    if effective_plan is not None:
        planned_queries = list(effective_plan.queries)
    exclude = {
        normalize_youtube_video_id(video_id) or str(video_id)
        for video_id in (exclude_video_ids or [])
    }
    from ..services.search_query_plan import normalize_query, semantic_query_family

    literal_query = " ".join(
        str(
            effective_plan.literal_query
            if effective_plan is not None
            else literal_topic or topic
        ).split()
    ) or topic
    literal_key = normalize_query(literal_query)
    literal_plan = next(
        (item for item in planned_queries if item.trust == "literal"),
        None,
    )
    planned_by_key = {
        normalize_query(item.text): item
        for item in planned_queries
        if normalize_query(item.text)
    }
    use_ai_plan = effective_plan is not None and effective_plan.ai_status == "validated"
    primary_query = literal_query
    if effective_plan is not None and len(literal_key.split()) > 12:
        planned_primary = " ".join(
            str(effective_plan.primary_search_query or "").split()
        )
        if normalize_query(planned_primary):
            primary_query = planned_primary
    primary_key = normalize_query(primary_query)
    primary_plan = planned_by_key.get(primary_key)
    primary_is_literal = primary_key == literal_key
    root_query_family = semantic_query_family(
        effective_plan.one_word_topic
        if use_ai_plan and normalize_query(effective_plan.one_word_topic)
        else primary_query
    )
    primary_metadata = {
        "query": primary_query,
        "query_family": (
            primary_plan.family if primary_plan is not None
            else root_query_family
        ),
        # The first practical request remains the ranking anchor even when a
        # long literal is summarized for retrieval. The unsummarized literal
        # stays on the plan as the final relevance identity.
        "query_trust": "literal",
        "query_provenance": (
            "literal" if primary_is_literal else "ai_summary"
        ),
        "hd_preferred": False,
    }
    if primary_is_literal and literal_plan is not None:
        primary_metadata.update(
            query_family=literal_plan.family,
            query_trust="literal",
            query_provenance=literal_plan.provenance,
        )

    expansion_candidates: list[dict[str, object]] = []
    seen_query_keys = {literal_key, primary_key}
    if use_ai_plan:
        one_word_topic = effective_plan.one_word_topic
        raw_synonyms = effective_plan.one_word_synonyms
        ordered_ai_terms = [(one_word_topic, "canonical"), *(
            (synonym, "ai") for synonym in raw_synonyms
        )]
        for raw_query, query_trust in ordered_ai_terms:
            query = " ".join(str(raw_query or "").split())
            query_key = normalize_query(query)
            if (
                not query_key
                or len(query_key.split()) != 1
                or query_key in seen_query_keys
            ):
                continue
            seen_query_keys.add(query_key)
            planned = planned_by_key.get(query_key)
            expansion_candidates.append(
                {
                    "query": query,
                    # Canonical and synonym requests are alternate surface
                    # forms of one retrieval identity, not independent topic
                    # evidence. Sharing a family prevents false consensus.
                    "query_family": root_query_family,
                    "query_trust": query_trust,
                    "query_provenance": (
                        planned.provenance if planned is not None else "ai"
                    ),
                    "hd_preferred": True,
                }
            )

    requests = [primary_metadata]
    expansion_offset = _planned_query_offset(context)
    for planned in expansion_candidates[expansion_offset:]:
        if len(requests) >= n:
            break
        requests.append(planned)
    if len(requests) < n:
        requests.append({**primary_metadata, "hd_preferred": True})

    def annotate_result_sets(per_query: list[dict]) -> None:
        for result_set, request in zip(per_query, requests):
            result_set.update(
                query_family=request["query_family"],
                query_trust=request["query_trust"],
                query_provenance=request["query_provenance"],
                hd_preferred=request["hd_preferred"],
            )

    def consensus_reached(per_query: list[dict]) -> bool:
        annotate_result_sets(per_query)
        return _consensus_count(per_query, exclude) >= limit

    request_queries = [str(request["query"]) for request in requests]
    res = supadata_search.search_all(
        request_queries,
        filters,
        request_filters=[
            _request_filters(filters, prefer_hd=bool(request["hd_preferred"]))
            for request in requests
        ],
        minimum_queries=(
            len(requests) if context is not None and context.budget.mode == "slow"
            else min(3, len(requests))
        ),
        stop_when=consensus_reached,
        should_cancel=should_cancel,
        language=language,
        context=context,
        cache_store=cache_store,
    )
    raise_if_cancelled(should_cancel)
    annotate_result_sets(res["per_query"])
    ranked = rank.merge_and_rank(res["per_query"], level=level)
    videos = _select_ranked_candidates(ranked, limit=limit, excluded=exclude)
    return {"corrected": topic, "videos": videos,
            "credits_used": res["credits_used"], "warning": res["warning"],
            "query_plan": effective_plan}
