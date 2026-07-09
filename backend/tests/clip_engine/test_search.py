# backend/tests/clip_engine/test_search.py
from backend.app.clip_engine import search


def test_discover_excludes_and_limits(monkeypatch):
    monkeypatch.setattr(search.expand, "expand_query",
                        lambda t, n, level=None: {"corrected": "calc", "queries": ["calc"], "provider_used": "free"})
    monkeypatch.setattr(search.supadata_search, "search_all",
                        lambda queries, filters=None: {"per_query": [
                            {"query": "calc", "videos": [{"id": "a"}, {"id": "b"}, {"id": "c"}]}],
                            "credits_used": 1, "warning": None})
    out = search.discover("calc", limit=2, exclude_video_ids=["a"])
    assert [v["id"] for v in out["videos"]] == ["b", "c"]
    assert out["credits_used"] == 1
    assert out["corrected"] == "calc"


def test_discover_threads_level_to_expand_and_rank(monkeypatch):
    from backend.app.clip_engine import search as s
    seen = {}

    def _fake_expand(topic, n, level=None):
        seen["expand_level"] = level
        return {"corrected": topic, "queries": [topic]}

    def _fake_rank(pq, level=None):
        seen["rank_level"] = level
        return []

    monkeypatch.setattr(s.expand, "expand_query", _fake_expand)
    monkeypatch.setattr(s.supadata_search, "search_all",
                        lambda qs: {"per_query": [], "credits_used": 0, "warning": None})
    monkeypatch.setattr(s.rank, "merge_and_rank", _fake_rank)
    s.discover("physics", limit=3, level="advanced")
    assert seen["expand_level"] == "advanced"
    assert seen["rank_level"] == "advanced"
