import time
import threading

import pytest

from backend.app.clip_engine import supadata_search as ss
from backend.app.clip_engine.errors import (
    ProviderBudgetExceededError,
    ProviderError,
    ProviderQuotaError,
    ProviderRateLimitError,
    ProviderTransientError,
)
from backend.app.clip_engine.provider_cache import MemoryProviderCache
from backend.app.clip_engine.provider_runtime import GenerationContext


class _Resp:
    def __init__(self, status, payload, headers=None):
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}
        self.text = str(payload)

    def json(self):
        return self._payload


class _InvalidJSONResp(_Resp):
    def json(self):
        raise ValueError("invalid json")


@pytest.fixture(autouse=True)
def _offline_client(monkeypatch):
    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, headers=None, params=None, timeout=None):
            return ss.httpx.get(url, headers=headers, params=params, timeout=timeout)

    async def no_sleep(*args, **kwargs):
        return None

    monkeypatch.setattr(ss.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(ss, "sleep_with_probe", no_sleep)
    monkeypatch.setattr(ss, "DEFAULT_PROVIDER_CACHE", MemoryProviderCache())
    monkeypatch.setattr(ss.config, "SUPADATA_API_KEY", "sd_test")


def test_search_one_sends_truthful_filters_and_caches(monkeypatch):
    calls = []

    def fake_get(url, headers=None, params=None, timeout=None):
        calls.append({"headers": headers, "params": params})
        return _Resp(200, {"results": [
            {"id": "dQw4w9WgXcQ", "title": "T", "type": "video"},
            {"id": "channel", "type": "channel"},
        ], "nextPageToken": "next"}, {"x-billable-requests": "1"})

    monkeypatch.setattr(ss.httpx, "get", fake_get)
    cache = MemoryProviderCache()
    filters = {"preferred_video_duration": "medium", "creative_commons_only": True}
    first = ss.search_one(
        "Calculus", filters, language="fr", cache_store=cache
    )
    second = ss.search_one(
        "  calculus ", filters, language="FR", cache_store=cache
    )
    assert calls[0]["headers"]["x-api-key"] == "sd_test"
    assert calls[0]["params"]["features"] == ["creative-commons", "creative-commons"]
    assert calls[0]["params"]["type"] == "video"
    assert calls[0]["params"]["sortBy"] == "relevance"
    assert calls[0]["params"]["duration"] == "medium"
    assert "lang" not in calls[0]["params"]
    assert [video["id"] for video in first["videos"]] == ["dQw4w9WgXcQ"]
    assert first["next_page_token"] == "next"
    assert second["cache_hit"] is True
    assert second["billed"] == 0
    assert len(calls) == 1


def test_search_deadline_expires_before_http_dispatch(monkeypatch):
    monkeypatch.setattr(
        ss.httpx,
        "get",
        lambda *_args, **_kwargs: pytest.fail("expired work must not reach HTTP"),
    )

    with pytest.raises(ProviderTransientError) as exc_info:
        ss.search_one(
            "Calculus",
            cache_store=MemoryProviderCache(),
            deadline_monotonic=time.monotonic() - 1.0,
        )

    assert exc_info.value.detail == "generation deadline exceeded"


def test_search_request_timeout_is_clamped_to_remaining_deadline(monkeypatch):
    timeouts: list[float] = []

    def fake_get(url, headers=None, params=None, timeout=None):
        del url, headers, params
        timeouts.append(float(timeout))
        return _Resp(200, {"results": []})

    monkeypatch.setattr(ss.httpx, "get", fake_get)
    ss.search_one(
        "Calculus",
        cache_store=MemoryProviderCache(),
        deadline_monotonic=time.monotonic() + 0.5,
    )

    assert len(timeouts) == 1
    assert 0.0 < timeouts[0] <= 0.5


def test_search_one_never_turns_subtitles_into_a_requirement(monkeypatch):
    calls = []

    def fake_get(url, headers=None, params=None, timeout=None):
        calls.append(params)
        return _Resp(200, {"results": []})

    monkeypatch.setattr(ss.httpx, "get", fake_get)
    result = ss.search_one(
        "Calculus",
        {"features": ["subtitles"]},
        cache_store=MemoryProviderCache(),
    )

    assert "features" not in calls[0]
    assert calls[0]["type"] == "video"
    assert calls[0]["sortBy"] == "relevance"
    assert result["filters_applied"]["features"] == []


def test_search_one_keeps_hd_singleton_as_array_on_wire(monkeypatch):
    calls = []

    def fake_get(url, headers=None, params=None, timeout=None):
        calls.append(params)
        return _Resp(200, {"results": []})

    monkeypatch.setattr(ss.httpx, "get", fake_get)
    result = ss.search_one(
        "Calculus",
        {"features": ["hd"]},
        cache_store=MemoryProviderCache(),
    )

    assert calls[0]["features"] == ["hd", "hd"]
    assert result["filters_applied"]["features"] == ["hd"]


def test_search_all_applies_per_request_filters_without_extra_calls(monkeypatch):
    calls = []

    def fake_one(query, filters=None, *args, **kwargs):
        calls.append((query, filters))
        return {"query": query, "videos": [], "billed": 0}

    monkeypatch.setattr(ss, "search_one", fake_one)
    monkeypatch.setattr(ss, "wait_with_probe", lambda *_args: None)

    ss.search_all(
        ["literal", "literal", "expansion"],
        {"duration": "medium"},
        request_filters=[
            {"duration": "medium", "features": []},
            {"duration": "medium", "features": ["hd"]},
            {"duration": "medium", "features": ["hd"]},
        ],
    )

    assert calls == [
        ("literal", {"duration": "medium", "features": []}),
        ("literal", {"duration": "medium", "features": ["hd"]}),
        ("expansion", {"duration": "medium", "features": ["hd"]}),
    ]


def test_search_all_threads_each_provider_page_token(monkeypatch):
    calls = []

    def fake_one(query, filters=None, *args, **kwargs):
        calls.append((query, filters, kwargs.get("page_token")))
        return {"query": query, "videos": [], "billed": 0}

    monkeypatch.setattr(ss, "search_one", fake_one)
    monkeypatch.setattr(ss, "wait_with_probe", lambda *_args: None)

    ss.search_all(
        ["literal", "expanded"],
        request_filters=[{"features": []}, {"features": ["hd"]}],
        page_tokens=["literal-page-2", "expanded-page-3"],
    )

    assert calls == [
        ("literal", {"features": []}, "literal-page-2"),
        ("expanded", {"features": ["hd"]}, "expanded-page-3"),
    ]


def test_search_all_returns_primary_result_when_optional_query_exhausts_budget(monkeypatch):
    calls = []

    def fake_get(url, headers=None, params=None, timeout=None):
        calls.append(params["query"])
        return _Resp(200, {"results": [{"id": params["query"], "type": "video"}]})

    monkeypatch.setattr(ss.httpx, "get", fake_get)
    context = GenerationContext("fast", cache_store=MemoryProviderCache())
    context.reserve("search")
    context.reserve("search")
    context.reserve("search")
    context.reserve("search")

    result = ss.search_all(["primary", "optional"], context=context)

    assert calls == ["primary"]
    assert [item["query"] for item in result["per_query"]] == ["primary"]
    assert result["warning"] == "Search budget exhausted after partial results."


def test_search_all_transient_primary_keeps_logical_query_headroom(monkeypatch):
    monkeypatch.setattr(
        ss.httpx,
        "get",
        lambda *args, **kwargs: _Resp(503, {"message": "try again"}),
    )
    context = GenerationContext("fast", cache_store=MemoryProviderCache())
    context.reserve("search")
    context.reserve("search")
    context.reserve("search")

    with pytest.raises(ProviderTransientError):
        ss.search_all(["primary", "optional"], context=context)
    assert context.budget.remaining("search") == 1
    assert len(context.usage()) == 3


def test_search_all_does_not_count_optional_transport_retries_as_queries(monkeypatch):
    responses = iter(
        [
            _Resp(200, {"results": [{"id": "primary", "type": "video"}]}),
            _Resp(503, {"message": "try again"}),
            _Resp(503, {"message": "try again"}),
            _Resp(503, {"message": "try again"}),
        ]
    )
    monkeypatch.setattr(ss.httpx, "get", lambda *args, **kwargs: next(responses))
    context = GenerationContext("fast", cache_store=MemoryProviderCache())
    context.reserve("search")
    context.reserve("search")

    with pytest.raises(ProviderTransientError):
        ss.search_all(["primary", "optional"], context=context)
    assert context.budget.remaining("search") == 1
    assert len(context.usage()) == 4


def test_three_query_fast_prefix_keeps_logical_headroom_through_transient_retries(
    monkeypatch,
):
    calls: dict[str, int] = {}
    calls_lock = threading.Lock()
    first_attempts = threading.Barrier(3)

    def fake_get(url, headers=None, params=None, timeout=None):
        del url, headers, timeout
        query = params["query"]
        with calls_lock:
            calls[query] = calls.get(query, 0) + 1
            attempt = calls[query]
        if attempt == 1:
            first_attempts.wait(timeout=5)
            return _Resp(503, {"message": "try again"})
        return _Resp(200, {"results": [{"id": query, "type": "video"}]})

    monkeypatch.setattr(ss.httpx, "get", fake_get)
    context = GenerationContext("fast", cache_store=MemoryProviderCache())

    result = ss.search_all(
        ["primary", "facet-one", "facet-two"],
        context=context,
        parallel_prefix=3,
    )

    assert [item["query"] for item in result["per_query"]] == [
        "primary",
        "facet-one",
        "facet-two",
    ]
    assert calls == {"primary": 2, "facet-one": 2, "facet-two": 2}
    assert context.budget.remaining("search") == 2
    assert len(context.usage()) == 6
    assert result["warning"] is None


def test_search_page_token_keeps_required_query_without_filters(monkeypatch):
    calls = []

    def fake_get(url, headers=None, params=None, timeout=None):
        calls.append(params)
        if not params or not params.get("query"):
            return _Resp(400, {"details": "query: Required"})
        return _Resp(200, {"results": []})

    monkeypatch.setattr(ss.httpx, "get", fake_get)
    ss.search_one(
        "  Newton's   laws  ",
        {"creative_commons_only": True, "duration": "long"},
        language="fr",
        page_token="next-token",
        cache_store=MemoryProviderCache(),
    )

    assert calls == [{"query": "Newton's laws", "nextPageToken": "next-token"}]


def test_quota_is_typed_and_not_converted_to_empty_success(monkeypatch):
    calls = 0

    def quota_failure(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return _Resp(402, {"message": "out of credits"})

    monkeypatch.setattr(ss.httpx, "get", quota_failure)
    with pytest.raises(ProviderQuotaError) as exc_info:
        ss.search_all(["a", "b"], cache_store=MemoryProviderCache())
    assert exc_info.value.status_code == 402
    assert calls == 1


@pytest.mark.parametrize(
    "first_response",
    [
        _Resp(408, {"message": "timed out"}),
        _InvalidJSONResp(200, "not-json"),
        _Resp(200, ["not", "an", "object"]),
        _Resp(200, {}),
        _Resp(200, {"results": {"id": "not-a-list"}}),
    ],
    ids=[
        "http-408",
        "invalid-json",
        "non-object-json",
        "missing-results",
        "non-list-results",
    ],
)
def test_recoverable_search_response_retries_inside_logical_request(
    monkeypatch,
    first_response,
):
    responses = iter(
        [
            first_response,
            _Resp(200, {"results": [{"id": "recovered", "type": "video"}]}),
        ]
    )
    calls = 0

    def recover_on_second_attempt(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return next(responses)

    monkeypatch.setattr(ss.httpx, "get", recover_on_second_attempt)
    context = GenerationContext("slow", cache_store=MemoryProviderCache())

    result = ss.search_one("recoverable", context=context)

    assert [video["id"] for video in result["videos"]] == ["recovered"]
    assert calls == 2
    assert context.budget.snapshot()["used"]["search"] == 1
    assert len(context.usage()) == 2
    assert context.usage()[0]["error_code"] == "provider_transient"


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404])
def test_permanent_search_response_is_not_retried(monkeypatch, status):
    calls = 0

    def permanent_failure(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return _Resp(status, {"message": "permanent"})

    monkeypatch.setattr(ss.httpx, "get", permanent_failure)
    with pytest.raises(ProviderError):
        ss.search_one("permanent", cache_store=MemoryProviderCache())
    assert calls == 1


def test_network_errors_retry_at_most_twice_then_surface(monkeypatch):
    calls = 0

    def flaky_get(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise ss.httpx.ConnectError("connection reset")

    monkeypatch.setattr(ss.httpx, "get", flaky_get)
    with pytest.raises(ProviderTransientError):
        ss.search_one("flaky", cache_store=MemoryProviderCache())
    assert calls == 3


def test_rate_limit_respects_bounded_retry_and_records_each_response(monkeypatch):
    responses = iter([
        _Resp(429, {"message": "slow down"}, {"retry-after": "999", "x-billable-requests": "1"}),
        _Resp(429, {"message": "slow down"}, {"retry-after": "999", "x-billable-requests": "1"}),
        _Resp(429, {"message": "slow down"}, {"retry-after": "999", "x-billable-requests": "1"}),
    ])
    monkeypatch.setattr(ss.httpx, "get", lambda *a, **k: next(responses))
    context = GenerationContext("slow", cache_store=MemoryProviderCache())
    with pytest.raises(ProviderRateLimitError) as exc_info:
        ss.search_one("rate limited", context=context)
    assert exc_info.value.retry_after_sec == 30.0
    assert context.budget.snapshot()["used"]["search"] == 1
    assert len(context.usage()) == 3
    assert sum(row["billable_requests"] for row in context.usage()) == 3
