import pytest
from backend.app.clip_engine import supadata_search as ss
from backend.app.clip_engine.errors import SearchError


class _Resp:
    def __init__(self, status, payload, headers=None):
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}
    def json(self):
        return self._payload


def test_search_one_returns_videos(monkeypatch):
    monkeypatch.setattr(ss.config, "SUPADATA_API_KEY", "sd_test")
    calls = {}
    def fake_get(url, headers=None, params=None, timeout=None):
        calls["headers"] = headers
        return _Resp(200, {"results": [
            {"id": "abc", "title": "T", "type": "video"},
            {"id": "def", "type": "channel"},
        ]}, {"x-billable-requests": "1"})
    monkeypatch.setattr(ss.httpx, "get", fake_get)
    out = ss.search_one("calculus")
    assert calls["headers"]["x-api-key"] == "sd_test"
    assert [v["id"] for v in out["videos"]] == ["abc"]
    assert out["billed"] == 1


def test_search_all_aggregates_and_warns_on_402(monkeypatch):
    monkeypatch.setattr(ss.config, "SUPADATA_API_KEY", "sd_test")
    seq = iter([
        _Resp(200, {"results": [{"id": "a", "type": "video"}]}, {"x-billable-requests": "1"}),
        _Resp(402, {"message": "out of credits"}, {"x-billable-requests": "0"}),
    ])
    monkeypatch.setattr(ss.httpx, "get", lambda *a, **k: next(seq))
    monkeypatch.setattr(ss.time, "sleep", lambda *_: None)
    out = ss.search_all(["a", "b"])
    assert out["credits_used"] == 1
    assert "out of Supadata credits" in out["warning"]
    assert len(out["per_query"]) == 2


def test_search_all_contains_network_errors_per_query(monkeypatch):
    """A transient network failure on ONE query degrades to partial results
    (VidScout searchAll parity) instead of killing the whole fan-out."""
    monkeypatch.setattr(ss.config, "SUPADATA_API_KEY", "sd_test")
    responses = iter([
        _Resp(200, {"results": [{"id": "a", "type": "video"}]}, {"x-billable-requests": "1"}),
    ])

    def flaky_get(*a, **k):
        try:
            return next(responses)
        except StopIteration:
            raise ss.httpx.ConnectError("connection reset")

    monkeypatch.setattr(ss.httpx, "get", flaky_get)
    monkeypatch.setattr(ss.time, "sleep", lambda *_: None)
    out = ss.search_all(["good", "flaky"])
    assert [v["id"] for v in out["per_query"][0]["videos"]] == ["a"]
    assert out["per_query"][1]["videos"] == []
    assert "1 of 2 searches failed" in out["warning"]
