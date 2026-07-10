"""Supadata YouTube Search client — Python port of practice/lib/supadata.js.
One query = one page (~20 results) = 1 credit. Sequential with 429 backoff.
"""
from __future__ import annotations

import time  # compatibility surface for existing callers/tests; sleeps are cancellation-aware below
from typing import Callable

import httpx

from . import config
from .cancellation import raise_if_cancelled, run_cancellable, sleep_with_probe, wait_with_probe
from .errors import SearchError

_MAX_RETRIES = 3


async def _search_one_async(
    query: str,
    filters: dict | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict:
    key = config.require_supadata_key()
    filters = filters or {}
    params = {"query": query, "type": "video"}
    if filters.get("sortBy") and filters["sortBy"] != "relevance":
        params["sortBy"] = filters["sortBy"]
    if filters.get("uploadDate") and filters["uploadDate"] != "all":
        params["uploadDate"] = filters["uploadDate"]
    if filters.get("duration") and filters["duration"] != "all":
        params["duration"] = filters["duration"]

    attempt = 0
    while True:
        raise_if_cancelled(should_cancel)
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(
                config.SUPADATA_SEARCH_URL,
                headers={"x-api-key": key},
                params=params,
            )
        billed = int(r.headers.get("x-billable-requests") or 0)
        if r.status_code == 429 and attempt < _MAX_RETRIES:
            retry_after = float(r.headers.get("retry-after") or 0) or 1.2 * (attempt + 1)
            await sleep_with_probe(retry_after, should_cancel)
            attempt += 1
            continue
        if r.status_code >= 400:
            detail = ""
            try:
                detail = r.json().get("message", "")
            except Exception:
                detail = ""
            err = SearchError(f"Supadata {r.status_code}{': ' + detail if detail else ''}")
            err.status = r.status_code  # type: ignore[attr-defined]
            err.billed = billed  # type: ignore[attr-defined]
            raise err
        data = r.json()
        results = data.get("results") if isinstance(data, dict) else None
        results = results if isinstance(results, list) else []
        videos = [it for it in results if (it.get("type") == "video" if it.get("type") else True)]
        return {"query": query, "videos": videos, "billed": billed or 1}


def search_one(
    query: str,
    filters: dict | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict:
    return run_cancellable(
        lambda: _search_one_async(query, filters, should_cancel), should_cancel
    )


def search_all(
    queries: list[str],
    filters: dict | None = None,
    *,
    minimum_queries: int = 0,
    stop_when: Callable[[list[dict]], bool] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict:
    credits_used = 0
    per_query: list[dict] = []
    errors: list[dict] = []
    for i, q in enumerate(queries):
        raise_if_cancelled(should_cancel)
        try:
            res = (
                search_one(q, filters)
                if should_cancel is None
                else search_one(q, filters, should_cancel)
            )
            credits_used += res["billed"]
            per_query.append(res)
        except (SearchError, httpx.HTTPError) as e:
            # Per-query containment (practice searchAll parity): a transient
            # network failure on one query degrades to partial results instead
            # of killing the whole discovery fan-out.
            credits_used += getattr(e, "billed", 0) or 0
            status = getattr(e, "status", None)
            errors.append({"query": q, "status": status, "message": str(e)})
            per_query.append({"query": q, "videos": [], "billed": getattr(e, "billed", 0) or 0,
                              "error": str(e), "status": status})
        should_stop = (
            stop_when is not None
            and len(per_query) >= max(0, int(minimum_queries))
            and stop_when(per_query)
        )
        if should_stop:
            break
        if i < len(queries) - 1:
            wait_with_probe(0.25, should_cancel)

    warning = None
    if errors:
        out_of_credits = any(e["status"] == 402 for e in errors)
        rate_limited = any(e["status"] == 429 for e in errors)
        reason = (" (out of Supadata credits)" if out_of_credits
                  else " (rate limited)" if rate_limited else "")
        warning = f"{len(errors)} of {len(per_query)} searches failed{reason}. Showing partial results."
    return {"per_query": per_query, "credits_used": credits_used, "warning": warning}
