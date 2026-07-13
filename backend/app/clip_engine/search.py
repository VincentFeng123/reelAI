"""Discovery: topic -> expand -> Supadata search -> rank -> exclude -> top N."""
from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import TYPE_CHECKING

from . import expand, rank, supadata_search
from .cancellation import raise_if_cancelled
from .errors import SearchError
from .metadata import normalize_youtube_video_id
from .provider_cache import ProviderCacheStore, normalize_filters
from .provider_runtime import GenerationContext

if TYPE_CHECKING:
    from ..services.search_query_plan import SearchQueryPlan

logger = logging.getLogger(__name__)


_DIFFICULTY_QUALIFIERS = {
    "beginner": re.compile(
        r"\b(?:beginners?|intro(?:duction|ductory)?|basics?|fundamentals?|novices?)\b",
        re.IGNORECASE,
    ),
    "intermediate": re.compile(r"\b(?:intermediate|mid[ -]?level)\b", re.IGNORECASE),
    "advanced": re.compile(
        r"\b(?:advanced|experts?|graduate(?:[ -]?level)?|upper[ -]?level)\b",
        re.IGNORECASE,
    ),
}

_NICHE_SEARCH_INTENT_SUFFIX = re.compile(r"(?:^|\s)identification\s*$", re.IGNORECASE)
_SEARCH_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_SEARCH_STOPWORDS = {"a", "an", "and", "for", "in", "of", "on", "the", "to", "with"}
_LONG_TOPIC_LEAD = re.compile(
    r"^(?:how|why|explain|describe|understand|learn)\s+",
    re.IGNORECASE,
)
_LONG_TOPIC_RELATION = re.compile(
    r"\b(?:work together|interact|combine|to maintain|to preserve|to pass|across)\b",
    re.IGNORECASE,
)


def _difficulty_bootstrap_query(topic: str, level: str | None) -> str:
    """Encode the learner's level once without changing the topic identity."""
    literal = " ".join(str(topic or "").split())
    normalized_level = str(level or "").strip().casefold()
    qualifier = _DIFFICULTY_QUALIFIERS.get(normalized_level)
    if not literal or qualifier is None or qualifier.search(literal):
        return literal
    if normalized_level == "beginner":
        return f"{literal} for beginners"
    return f"{normalized_level} {literal}"


def _niche_bootstrap_backoff_query(topic: str) -> str | None:
    """Drop only trailing task words when at least two topic terms remain."""
    literal = " ".join(str(topic or "").split())
    if len(literal.split()) < 3:
        return None
    shortened = literal
    shortened = _NICHE_SEARCH_INTENT_SUFFIX.sub("", shortened).strip(" ,:;-")
    if shortened == literal or len(_search_coverage_tokens(shortened)) < 2:
        return None
    return shortened


def _long_topic_component_queries(topic: str, *, limit: int = 6) -> list[str]:
    """Extract bounded literal subtopics when a sentence is too broad for search.

    The complete user text remains the transcript-selection identity. These
    deterministic components only give Supadata searchable surfaces when the
    literal is a long list or paragraph.
    """
    literal = " ".join(str(topic or "").split())
    if len(_SEARCH_TOKEN_RE.findall(literal)) <= 12 and len(literal) <= 120:
        return []
    stripped = _LONG_TOPIC_LEAD.sub("", literal).strip(" ,:;-?.")
    parts = re.split(r"\s*[,;]\s*|\s+\b(?:and|or)\b\s+", stripped, flags=re.IGNORECASE)
    components: list[str] = []
    seen: set[str] = set()
    for raw_part in parts:
        part = re.sub(r"^(?:and|or)\s+", "", raw_part.strip(), flags=re.IGNORECASE)
        part = _LONG_TOPIC_RELATION.split(part, maxsplit=1)[0]
        words = part.strip(" ,:;-?.").split()
        if len(words) > 8:
            words = words[:8]
        candidate = " ".join(words).strip()
        key = " ".join(candidate.casefold().split())
        if len(_search_coverage_tokens(candidate)) < 2 or key in seen:
            continue
        seen.add(key)
        components.append(candidate)
        if len(components) >= max(1, int(limit)):
            break
    return components


def _search_coverage_tokens(text: object) -> set[str]:
    tokens: set[str] = set()
    for raw in _SEARCH_TOKEN_RE.findall(str(text or "").casefold()):
        if raw in _SEARCH_STOPWORDS:
            continue
        token = raw[:-1] if len(raw) > 4 and raw.endswith("s") and not raw.endswith("ss") else raw
        tokens.add(token)
    return tokens


def _bootstrap_pool_has_subject_coverage(videos: list[dict], subject: str) -> bool:
    """Check only the ranked videos the bootstrap stage can actually analyze."""
    subject_tokens = _search_coverage_tokens(subject)
    if len(subject_tokens) < 2:
        return True
    for video in videos:
        metadata_tokens = _search_coverage_tokens(
            f"{video.get('title') or ''} {video.get('description') or ''}"
        )
        if len(subject_tokens & metadata_tokens) >= 2:
            return True
    return False


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
             query_plan: "SearchQueryPlan | None" = None,
             practice_fast: bool = False,
             retrieval_profile: str = "deep",
             deadline_monotonic: float | None = None) -> dict:
    if practice_fast:
        return discover_practice_fast(
            topic,
            limit,
            exclude_video_ids,
            breadth,
            level,
            should_cancel,
            filters=filters,
            language=language,
            context=context,
            cache_store=cache_store,
            literal_topic=literal_topic,
            use_query_planner=use_query_planner,
            query_plan=query_plan,
            retrieval_profile=retrieval_profile,
            deadline_monotonic=deadline_monotonic,
        )
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


def discover_practice_fast(
    topic: str,
    limit: int,
    exclude_video_ids: list[str] | None = None,
    breadth: int | None = None,
    level: str | None = None,
    should_cancel: Callable[[], bool] | None = None,
    *,
    filters: dict | None = None,
    language: str = "en",
    context: GenerationContext | None = None,
    cache_store: ProviderCacheStore | None = None,
    literal_topic: str | None = None,
    use_query_planner: bool = True,
    query_plan: "SearchQueryPlan | None" = None,
    retrieval_profile: str = "deep",
    deadline_monotonic: float | None = None,
) -> dict:
    """Difficulty-first bootstrap or literal-first conditional expansion.

    The signature intentionally matches ``discover`` so live wiring can switch paths
    without dropping cancellation, budgets, provider caching, filters, or exclusions.
    Planner arguments remain a compatibility surface; the user's literal text always
    remains the downstream relevance identity.
    """
    del use_query_planner
    topic = " ".join(str(topic or "").split())
    if not topic:
        raise SearchError("topic must contain non-whitespace text")

    try:
        requested = int(breadth if breadth is not None else 8)
    except (TypeError, ValueError, OverflowError):
        requested = 8
    query_count = max(1, min(12, requested))
    if context is not None:
        if context.budget.mode == "fast":
            query_count = min(query_count, 3)
        query_count = min(query_count, context.budget.remaining("search"))
    if query_count <= 0:
        raise SearchError("search budget exhausted")

    raise_if_cancelled(should_cancel)
    literal_query = " ".join(str(literal_topic or topic).split()) or topic
    tutorial_query = f"{literal_query} explained tutorial"
    component_queries = _long_topic_component_queries(literal_query)
    initial_queries: list[str] = []
    seen_query_keys: set[str] = set()
    bootstrap = str(retrieval_profile or "deep").strip().casefold() == "bootstrap"
    if bootstrap:
        deterministic_queries = (_difficulty_bootstrap_query(literal_query, level),)
    else:
        # Keep the established literal/tutorial surface and leave at least one
        # provider-search slot for conditional Gemini expansion. Components
        # are only a bounded recovery surface for unusually long topics.
        component_limit = max(0, query_count - 3)
        deterministic_queries = (
            literal_query,
            tutorial_query,
            *component_queries[:component_limit],
        )
    for candidate in deterministic_queries:
        key = " ".join(candidate.casefold().split())
        if key and key not in seen_query_keys:
            seen_query_keys.add(key)
            initial_queries.append(candidate)
        if len(initial_queries) >= query_count:
            break

    search_runtime: dict[str, object] = {
        "should_cancel": should_cancel,
        "language": language,
        "context": context,
        "cache_store": cache_store,
    }
    if deadline_monotonic is not None:
        search_runtime["deadline_monotonic"] = float(deadline_monotonic)

    initial = supadata_search.search_all(
        initial_queries,
        filters,
        parallel_prefix=len(initial_queries),
        **search_runtime,
    )
    raise_if_cancelled(should_cancel)

    from ..services.search_query_plan import semantic_query_family

    root_family = semantic_query_family(literal_query)

    def annotate(result_sets: list[dict], *, expanded: bool = False) -> None:
        for result_set in result_sets:
            query = " ".join(str(result_set.get("query") or "").split())
            is_literal = query.casefold() == literal_query.casefold()
            result_set.update(
                query_family=(semantic_query_family(query) or root_family),
                query_trust=("literal" if is_literal else "ai" if expanded else "trusted"),
                query_provenance=("literal" if is_literal else "gemini" if expanded else "deterministic"),
                hd_preferred=False,
            )

    per_query = list(initial.get("per_query") or [])
    annotate(per_query)
    excluded = {
        normalize_youtube_video_id(video_id) or str(video_id or "").strip()
        for video_id in (exclude_video_ids or [])
    }
    initial_ranked = rank.merge_and_rank(per_query, level=level)
    eligible_initial = [
        video for video in initial_ranked if video.get("id") not in excluded
    ]
    niche_result = {"per_query": [], "credits_used": 0, "warning": None}
    niche_provider_order: list[str] = []
    niche_backoff = _niche_bootstrap_backoff_query(literal_query) if bootstrap else None
    if (
        niche_backoff
        and eligible_initial
        and len(initial_queries) < query_count
        and not _bootstrap_pool_has_subject_coverage(
            eligible_initial[:max(0, int(limit))], niche_backoff
        )
    ):
        recovery_query = _difficulty_bootstrap_query(niche_backoff, level)
        recovery_key = " ".join(recovery_query.casefold().split())
        if recovery_key and recovery_key not in seen_query_keys:
            seen_query_keys.add(recovery_key)
            niche_result = supadata_search.search_all(
                [recovery_query],
                filters,
                **search_runtime,
            )
            raise_if_cancelled(should_cancel)
            annotate(niche_result["per_query"])
            per_query.extend(niche_result["per_query"])
            initial_queries.append(recovery_query)
            for result_set in niche_result["per_query"]:
                for video in result_set.get("videos") or []:
                    raw_video_id = video.get("id") or video.get("videoId") or video.get("url")
                    video_id = normalize_youtube_video_id(raw_video_id) or str(raw_video_id or "").strip()
                    if video_id and video_id not in niche_provider_order:
                        niche_provider_order.append(video_id)
    if bootstrap:
        fallback_result = {"per_query": [], "credits_used": 0, "warning": None}
        component_result = {"per_query": [], "credits_used": 0, "warning": None}
        qualified_key = " ".join(initial_queries[0].casefold().split())
        literal_key = " ".join(literal_query.casefold().split())
        if (
            not eligible_initial
            and qualified_key != literal_key
            and len(initial_queries) < query_count
        ):
            fallback_result = supadata_search.search_all(
                [literal_query],
                filters,
                **search_runtime,
            )
            raise_if_cancelled(should_cancel)
            annotate(fallback_result["per_query"])
            per_query.extend(fallback_result["per_query"])
            initial_queries.append(literal_query)

        remaining_component_queries: list[str] = []
        for component in component_queries:
            if len(initial_queries) + len(remaining_component_queries) >= query_count:
                break
            query = _difficulty_bootstrap_query(component, level)
            key = " ".join(query.casefold().split())
            if not key or key in seen_query_keys:
                continue
            seen_query_keys.add(key)
            remaining_component_queries.append(query)
        if remaining_component_queries:
            component_result = supadata_search.search_all(
                remaining_component_queries,
                filters,
                parallel_prefix=len(remaining_component_queries),
                **search_runtime,
            )
            raise_if_cancelled(should_cancel)
            annotate(component_result["per_query"])
            per_query.extend(component_result["per_query"])
            initial_queries.extend(remaining_component_queries)

        ranked = rank.merge_and_rank(per_query, level=level)
        if niche_provider_order:
            ranked_by_id = {str(video.get("id") or ""): video for video in ranked}
            preferred = [
                ranked_by_id[video_id]
                for video_id in niche_provider_order
                if video_id in ranked_by_id
            ]
            preferred_ids = {str(video.get("id") or "") for video in preferred}
            ranked = [
                *preferred,
                *[video for video in ranked if str(video.get("id") or "") not in preferred_ids],
            ]
        top_n = max(0, int(limit))
        videos = [
            video for video in ranked if video.get("id") not in excluded
        ][:top_n]
        return {
            "corrected": literal_query,
            "queries": initial_queries,
            # The deterministic qualifier is retrieval-only. Transcript
            # segmentation and topic-evidence checks retain the raw topic.
            "topic_terms": [literal_query],
            "provider_used": "deterministic",
            "videos": videos,
            "credits_used": (
                int(initial.get("credits_used") or 0)
                + int(niche_result.get("credits_used") or 0)
                + int(fallback_result.get("credits_used") or 0)
                + int(component_result.get("credits_used") or 0)
            ),
            "warning": (
                component_result.get("warning")
                or fallback_result.get("warning")
                or niche_result.get("warning")
                or initial.get("warning")
            ),
            "query_plan": query_plan,
        }

    good_candidates = sum(
        1
        for video in initial_ranked
        if video.get("id") not in excluded
        and float(video.get("retrieval_score") or 0.0) >= 0.60
    )
    required_sources = (
        2
        if context is not None and context.budget.mode == "fast"
        else 3
    )

    expansion: dict[str, object] = {
        "corrected": literal_query,
        "queries": [],
        "provider_used": "skipped",
    }
    remaining_queries = max(0, query_count - len(initial_queries))
    expansion_queries: list[str] = []
    expansion_result = {"per_query": [], "credits_used": 0, "warning": None}
    if good_candidates < required_sources and remaining_queries > 0:
        expansion = expand.expand_query_practice_fast(
            literal_query,
            min(8, remaining_queries + 2),
            level=level,
            should_cancel=should_cancel,
            context=context,
        )
        for candidate in expansion.get("queries") or []:
            query = " ".join(str(candidate or "").split())
            key = " ".join(query.casefold().split())
            if not key or key in seen_query_keys:
                continue
            seen_query_keys.add(key)
            expansion_queries.append(query)
            if len(expansion_queries) >= remaining_queries:
                break
        if expansion_queries:
            expansion_result = supadata_search.search_all(
                expansion_queries,
                filters,
                **search_runtime,
            )
            annotate(expansion_result["per_query"], expanded=True)
            per_query.extend(expansion_result["per_query"])

    ranked = rank.merge_and_rank(per_query, level=level)
    top_n = max(0, int(limit))
    videos = [video for video in ranked if video.get("id") not in excluded][:top_n]
    corrected = " ".join(str(expansion.get("corrected") or literal_query).split()) or literal_query
    topic_terms = []
    seen_terms: set[str] = set()
    for term in (literal_query, corrected, *expansion_queries):
        key = " ".join(str(term or "").casefold().split())
        if key and key not in seen_terms:
            seen_terms.add(key)
            topic_terms.append(str(term))
    return {
        "corrected": corrected,
        "queries": [*initial_queries, *expansion_queries],
        "topic_terms": topic_terms,
        "provider_used": expansion.get("provider_used") or "deterministic",
        "videos": videos,
        "credits_used": int(initial.get("credits_used") or 0) + int(expansion_result.get("credits_used") or 0),
        "warning": expansion_result.get("warning") or initial.get("warning"),
        "query_plan": query_plan,
    }
