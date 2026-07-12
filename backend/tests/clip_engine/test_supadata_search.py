import pytest

from backend.app.clip_engine import supadata_search as ss
from backend.app.clip_engine.errors import (
    ProviderBudgetExceededError,
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


@pytest.fixture(autouse=True)
def _offline_client(monkeypatch):
    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, headers=None, params=None):
            return ss.httpx.get(url, headers=headers, params=params, timeout=None)

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


def test_search_all_returns_primary_result_when_optional_query_exhausts_budget(monkeypatch):
    calls = []

    def fake_get(url, headers=None, params=None, timeout=None):
        calls.append(params["query"])
        return _Resp(200, {"results": [{"id": params["query"], "type": "video"}]})

    monkeypatch.setattr(ss.httpx, "get", fake_get)
    context = GenerationContext("fast", cache_store=MemoryProviderCache())
    context.reserve("search")
    context.reserve("search")

    result = ss.search_all(["primary", "optional"], context=context)

    assert calls == ["primary"]
    assert [item["query"] for item in result["per_query"]] == ["primary"]
    assert result["warning"] == "Search budget exhausted after partial results."


def test_search_all_raises_when_budget_exhausts_before_primary_succeeds(monkeypatch):
    monkeypatch.setattr(
        ss.httpx,
        "get",
        lambda *args, **kwargs: _Resp(503, {"message": "try again"}),
    )
    context = GenerationContext("fast", cache_store=MemoryProviderCache())
    context.reserve("search")

    with pytest.raises(ProviderBudgetExceededError):
        ss.search_all(["primary", "optional"], context=context)


def test_search_all_keeps_primary_when_optional_retries_consume_budget(monkeypatch):
    responses = iter(
        [
            _Resp(200, {"results": [{"id": "primary", "type": "video"}]}),
            _Resp(503, {"message": "try again"}),
            _Resp(503, {"message": "try again"}),
        ]
    )
    monkeypatch.setattr(ss.httpx, "get", lambda *args, **kwargs: next(responses))
    context = GenerationContext("fast", cache_store=MemoryProviderCache())

    result = ss.search_all(["primary", "optional"], context=context)

    assert [item["query"] for item in result["per_query"]] == ["primary"]
    assert context.budget.remaining("search") == 0
    assert result["warning"] == "Search budget exhausted after partial results."


def test_search_page_token_uses_documented_parameter_without_filters(monkeypatch):
    calls = []

    def fake_get(url, headers=None, params=None, timeout=None):
        calls.append(params)
        return _Resp(200, {"results": []})

    monkeypatch.setattr(ss.httpx, "get", fake_get)
    ss.search_one(
        "ignored on continuation",
        {"creative_commons_only": True, "duration": "long"},
        language="fr",
        page_token="next-token",
        cache_store=MemoryProviderCache(),
    )

    assert calls == [{"nextPageToken": "next-token"}]


def test_quota_is_typed_and_not_converted_to_empty_success(monkeypatch):
    monkeypatch.setattr(
        ss.httpx,
        "get",
        lambda *a, **k: _Resp(402, {"message": "out of credits"}),
    )
    with pytest.raises(ProviderQuotaError) as exc_info:
        ss.search_all(["a", "b"], cache_store=MemoryProviderCache())
    assert exc_info.value.status_code == 402


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
    assert len(context.usage()) == 3
    assert sum(row["billable_requests"] for row in context.usage()) == 3
