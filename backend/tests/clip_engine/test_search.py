# backend/tests/clip_engine/test_search.py
from backend.app.clip_engine import search


def test_discover_excludes_and_limits(monkeypatch):
    monkeypatch.setattr(search.expand, "expand_query",
                        lambda t, n: {"corrected": "calc", "queries": ["calc"], "provider_used": "free"})
    monkeypatch.setattr(search.supadata_search, "search_all",
                        lambda queries, filters=None: {"per_query": [
                            {"query": "calc", "videos": [{"id": "a"}, {"id": "b"}, {"id": "c"}]}],
                            "credits_used": 1, "warning": None})
    out = search.discover("calc", limit=2, exclude_video_ids=["a"])
    assert [v["id"] for v in out["videos"]] == ["b", "c"]
    assert out["credits_used"] == 1
    assert out["corrected"] == "calc"
