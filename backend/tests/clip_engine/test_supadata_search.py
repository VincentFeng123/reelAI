import pytest

from backend.app.clip_engine import supadata_search as ss
from backend.app.clip_engine.errors import (
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
        "Calculus", filters, language="fr", page_token="p1", cache_store=cache
    )
    second = ss.search_one(
        "  calculus ", filters, language="FR", page_token="p1", cache_store=cache
    )
    assert calls[0]["headers"]["x-api-key"] == "sd_test"
    assert calls[0]["params"]["features"] == ["creative-commons", "subtitles"]
    assert calls[0]["params"]["duration"] == "medium"
    assert calls[0]["params"]["lang"] == "fr"
    assert calls[0]["params"]["pageToken"] == "p1"
    assert [video["id"] for video in first["videos"]] == ["dQw4w9WgXcQ"]
    assert first["next_page_token"] == "next"
    assert second["cache_hit"] is True
    assert second["billed"] == 0
    assert len(calls) == 1


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
