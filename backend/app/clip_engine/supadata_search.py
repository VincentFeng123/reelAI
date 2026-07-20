"""Supadata YouTube search with truthful filters, caching, budgets, and retries."""
from __future__ import annotations

import time  # compatibility surface retained for callers that import it
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx

from . import config
from .cancellation import raise_if_cancelled, run_cancellable, sleep_with_probe, wait_with_probe
from .errors import (
    ProviderAuthenticationError,
    ProviderBudgetExceededError,
    ProviderError,
    ProviderQuotaError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderTransientError,
)
from .provider_cache import (
    DEFAULT_PROVIDER_CACHE,
    ProviderCacheStore,
    normalize_filters,
    normalize_language,
    normalize_query,
    search_cache_key,
)
from .provider_runtime import GenerationContext, MAX_PROVIDER_RETRIES, bounded_retry_after


def _remaining_seconds(deadline_monotonic: float | None) -> float | None:
    if deadline_monotonic is None:
        return None
    return max(0.0, float(deadline_monotonic) - time.monotonic())


def _request_timeout(deadline_monotonic: float | None) -> float:
    remaining = _remaining_seconds(deadline_monotonic)
    if remaining is not None and remaining <= 0:
        raise ProviderTransientError(
            "Supadata search timed out.",
            provider="supadata",
            operation="search",
            detail="generation deadline exceeded",
        )
    return max(0.001, min(30.0, remaining if remaining is not None else 30.0))


async def _sleep_before_retry(
    seconds: float,
    *,
    should_cancel: Callable[[], bool] | None,
    deadline_monotonic: float | None,
) -> None:
    remaining = _remaining_seconds(deadline_monotonic)
    await sleep_with_probe(
        min(max(0.0, seconds), remaining) if remaining is not None else max(0.0, seconds),
        should_cancel,
    )
    _request_timeout(deadline_monotonic)


def _message(response: httpx.Response) -> str:
    try:
        body = response.json()
    except Exception:
        return str(getattr(response, "text", "") or "")[:300]
    if isinstance(body, dict):
        return str(body.get("details") or body.get("message") or body.get("error") or "")[:300]
    return ""


def _failure(response: httpx.Response, *, retry_after: float | None = None) -> ProviderError:
    status = int(response.status_code)
    detail = _message(response)
    kwargs = dict(
        provider="supadata",
        operation="search",
        status_code=status,
        detail=detail or None,
    )
    if status in (401, 403):
        return ProviderAuthenticationError("Supadata authentication failed.", **kwargs)
    if status == 402:
        return ProviderQuotaError("Supadata quota is exhausted.", **kwargs)
    if status == 429:
        return ProviderRateLimitError(
            "Supadata search is rate limited.", retry_after_sec=retry_after, **kwargs
        )
    if 500 <= status <= 599:
        return ProviderTransientError("Supadata search is temporarily unavailable.", **kwargs)
    return ProviderRequestError(f"Supadata rejected the search request ({status}).", **kwargs)


async def _search_one_async(
    query: str,
    filters: dict[str, Any] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    *,
    language: str = "en",
    page_token: str | None = None,
    context: GenerationContext | None = None,
    cache_store: ProviderCacheStore | None = None,
    deadline_monotonic: float | None = None,
) -> dict[str, Any]:
    store = cache_store or (context.cache_store if context is not None else None) or DEFAULT_PROVIDER_CACHE
    normalized_filters = {**normalize_filters(filters), "sort_by": "relevance"}
    normalized_language = normalize_language(language) or "en"
    cache_key = search_cache_key(
        query=query,
        filters=normalized_filters,
        language=normalized_language,
        page_token=page_token,
    )
    cached = store.get_search(cache_key)
    if cached is not None:
        payload = dict(cached.payload)
        return {
            "query": query,
            "videos": payload.get("videos") or [],
            "next_page_token": payload.get("next_page_token") or None,
            "billed": 0,
            "cache_hit": True,
            "evidence_age_sec": cached.age_sec,
            "filters_applied": normalized_filters,
        }
    key = config.require_supadata_key()

    if page_token:
        params: dict[str, Any] = {
            "query": " ".join(str(query or "").split()),
            "nextPageToken": str(page_token).strip(),
        }
    else:
        wire_features = list(normalized_filters["features"])
        if len(wire_features) == 1:
            # Supadata collapses a singleton query value to a string, then
            # rejects it because `features` must remain an array.
            wire_features.append(wire_features[0])
        params = {
            "query": " ".join(str(query or "").split()),
            "type": "video",
            "sortBy": "relevance",
        }
        if wire_features:
            params["features"] = wire_features
        if normalized_filters["upload_date"] != "all":
            params["uploadDate"] = normalized_filters["upload_date"]
        if normalized_filters["duration"] != "all":
            params["duration"] = normalized_filters["duration"]

    billed_total = 0
    async with httpx.AsyncClient(timeout=30.0) as client:
        for retry_index in range(MAX_PROVIDER_RETRIES + 1):
            raise_if_cancelled(should_cancel)
            request_timeout = _request_timeout(deadline_monotonic)
            attempt = retry_index + 1
            if context is not None:
                context.reserve("search")
            try:
                response = await client.get(
                    config.SUPADATA_SEARCH_URL,
                    headers={"x-api-key": key, "Accept": "application/json"},
                    params=params,
                    timeout=request_timeout,
                )
            except httpx.RequestError as exc:
                if context is not None:
                    context.record_http(
                        provider="supadata",
                        operation="search",
                        attempt=attempt,
                        status_code=None,
                        error_code="provider_transient",
                )
                if retry_index < MAX_PROVIDER_RETRIES:
                    await _sleep_before_retry(
                        min(30.0, 1.2 * attempt),
                        should_cancel=should_cancel,
                        deadline_monotonic=deadline_monotonic,
                    )
                    continue
                raise ProviderTransientError(
                    "Could not reach Supadata search.",
                    provider="supadata",
                    operation="search",
                    detail=str(exc),
                ) from exc

            try:
                billed = max(0, int(response.headers.get("x-billable-requests") or 0))
            except (TypeError, ValueError):
                billed = 0
            billed_total += billed
            status = int(response.status_code)
            error_code = ""
            if status == 429:
                error_code = "provider_rate_limited"
            elif status >= 500:
                error_code = "provider_transient"
            elif status >= 400:
                error_code = _failure(response).code
            if context is not None:
                context.record_http(
                    provider="supadata",
                    operation="search",
                    attempt=attempt,
                    status_code=status,
                    headers=response.headers,
                    error_code=error_code,
                )

            if status == 429 or 500 <= status <= 599:
                retry_after = bounded_retry_after(response.headers)
                if retry_index < MAX_PROVIDER_RETRIES:
                    await _sleep_before_retry(
                        retry_after if retry_after is not None else min(30.0, 1.2 * attempt),
                        should_cancel=should_cancel,
                        deadline_monotonic=deadline_monotonic,
                    )
                    continue
                raise _failure(response, retry_after=retry_after)
            if status >= 400:
                raise _failure(response)

            try:
                data = response.json()
            except Exception as exc:
                raise ProviderRequestError(
                    "Supadata returned invalid search JSON.",
                    provider="supadata",
                    operation="search",
                    status_code=status,
                ) from exc
            results = data.get("results") if isinstance(data, dict) else None
            results = results if isinstance(results, list) else []
            videos = [
                item for item in results
                if isinstance(item, dict) and (item.get("type") in (None, "video"))
            ]
            payload = store.filter_search_payload(
                {
                    "videos": videos,
                    "next_page_token": (
                        data.get("nextPageToken") or data.get("next_page_token") or None
                        if isinstance(data, dict) else None
                    ),
                }
            )
            store.put_search(
                cache_key,
                payload,
                {
                    "normalized_query": normalize_query(query),
                    "filters": normalized_filters,
                    "language": normalized_language,
                    "page_token": str(page_token or ""),
                },
            )
            return {
                "query": query,
                "videos": payload["videos"],
                "next_page_token": payload.get("next_page_token"),
                "billed": billed_total,
                "cache_hit": False,
                "evidence_age_sec": 0.0,
                "filters_applied": normalized_filters,
            }

    raise AssertionError("unreachable")


def search_one(
    query: str,
    filters: dict[str, Any] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    *,
    language: str = "en",
    page_token: str | None = None,
    context: GenerationContext | None = None,
    cache_store: ProviderCacheStore | None = None,
    deadline_monotonic: float | None = None,
) -> dict[str, Any]:
    return run_cancellable(
        lambda: _search_one_async(
            query,
            filters,
            should_cancel,
            language=language,
            page_token=page_token,
            context=context,
            cache_store=cache_store,
            deadline_monotonic=deadline_monotonic,
        ),
        should_cancel,
    )


def search_all(
    queries: list[str],
    filters: dict[str, Any] | None = None,
    *,
    page_tokens: list[str | None] | None = None,
    request_filters: list[dict[str, Any] | None] | None = None,
    minimum_queries: int = 0,
    stop_when: Callable[[list[dict]], bool] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    language: str = "en",
    context: GenerationContext | None = None,
    cache_store: ProviderCacheStore | None = None,
    parallel_prefix: int = 0,
    deadline_monotonic: float | None = None,
) -> dict[str, Any]:
    credits_used = 0
    per_query: list[dict] = []
    warning: str | None = None

    prefix_count = min(len(queries), max(0, int(parallel_prefix)))
    if prefix_count > 1:
        with ThreadPoolExecutor(max_workers=prefix_count) as executor:
            futures = []
            for index, query in enumerate(queries[:prefix_count]):
                effective_filters = (
                    request_filters[index]
                    if request_filters is not None and index < len(request_filters)
                    else filters
                )
                page_token = (
                    page_tokens[index]
                    if page_tokens is not None and index < len(page_tokens)
                    else None
                )
                futures.append(
                    executor.submit(
                        search_one,
                        query,
                        effective_filters,
                        should_cancel,
                        language=language,
                        page_token=page_token,
                        context=context,
                        cache_store=cache_store,
                        deadline_monotonic=deadline_monotonic,
                    )
                )
            for future in futures:
                raise_if_cancelled(should_cancel)
                try:
                    result = future.result()
                except ProviderBudgetExceededError:
                    if not per_query:
                        raise
                    warning = "Search budget exhausted after partial results."
                    break
                credits_used += int(result.get("billed") or 0)
                per_query.append(result)

    start_index = prefix_count if prefix_count > 1 else 0
    for index in range(start_index, len(queries)):
        query = queries[index]
        raise_if_cancelled(should_cancel)
        effective_filters = (
            request_filters[index]
            if request_filters is not None and index < len(request_filters)
            else filters
        )
        page_token = (
            page_tokens[index]
            if page_tokens is not None and index < len(page_tokens)
            else None
        )
        try:
            result = search_one(
                query,
                effective_filters,
                should_cancel,
                language=language,
                page_token=page_token,
                context=context,
                cache_store=cache_store,
                deadline_monotonic=deadline_monotonic,
            )
        except ProviderBudgetExceededError:
            if not per_query:
                raise
            warning = "Search budget exhausted after partial results."
            break
        credits_used += int(result.get("billed") or 0)
        per_query.append(result)
        if (
            stop_when is not None
            and len(per_query) >= max(0, int(minimum_queries))
            and stop_when(per_query)
        ):
            break
        if index < len(queries) - 1:
            wait_with_probe(0.25, should_cancel)
    return {"per_query": per_query, "credits_used": credits_used, "warning": warning}
