"""Supadata YouTube search with truthful filters, caching, budgets, and retries."""
from __future__ import annotations

import time  # compatibility surface retained for callers that import it
from collections.abc import Callable
from typing import Any

import httpx

from . import config
from .cancellation import raise_if_cancelled, run_cancellable, sleep_with_probe, wait_with_probe
from .errors import (
    ProviderAuthenticationError,
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
) -> dict[str, Any]:
    store = cache_store or (context.cache_store if context is not None else None) or DEFAULT_PROVIDER_CACHE
    normalized_filters = normalize_filters(filters)
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
        params: dict[str, Any] = {"nextPageToken": str(page_token).strip()}
    else:
        params = {
            "query": " ".join(str(query or "").split()),
            "type": "video",
            "features": normalized_filters["features"],
        }
        if normalized_filters["sort_by"] != "relevance":
            params["sortBy"] = normalized_filters["sort_by"]
        if normalized_filters["upload_date"] != "all":
            params["uploadDate"] = normalized_filters["upload_date"]
        if normalized_filters["duration"] != "all":
            params["duration"] = normalized_filters["duration"]

    billed_total = 0
    async with httpx.AsyncClient(timeout=30.0) as client:
        for retry_index in range(MAX_PROVIDER_RETRIES + 1):
            raise_if_cancelled(should_cancel)
            attempt = retry_index + 1
            if context is not None:
                context.reserve("search")
            try:
                response = await client.get(
                    config.SUPADATA_SEARCH_URL,
                    headers={"x-api-key": key, "Accept": "application/json"},
                    params=params,
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
                    await sleep_with_probe(min(30.0, 1.2 * attempt), should_cancel)
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
                    await sleep_with_probe(
                        retry_after if retry_after is not None else min(30.0, 1.2 * attempt),
                        should_cancel,
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
        ),
        should_cancel,
    )


def search_all(
    queries: list[str],
    filters: dict[str, Any] | None = None,
    *,
    minimum_queries: int = 0,
    stop_when: Callable[[list[dict]], bool] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    language: str = "en",
    context: GenerationContext | None = None,
    cache_store: ProviderCacheStore | None = None,
) -> dict[str, Any]:
    credits_used = 0
    per_query: list[dict] = []
    for index, query in enumerate(queries):
        raise_if_cancelled(should_cancel)
        result = search_one(
            query,
            filters,
            should_cancel,
            language=language,
            context=context,
            cache_store=cache_store,
        )
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
    return {"per_query": per_query, "credits_used": credits_used, "warning": None}
