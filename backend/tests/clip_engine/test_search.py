# backend/tests/clip_engine/test_search.py
from backend.app.clip_engine import search


def _expansion():
    return {"corrected": "calc", "queries": [f"q{i}" for i in range(1, 7)], "provider_used": "free"}


def test_discover_excludes_and_limits(monkeypatch):
    monkeypatch.setattr(search.expand, "expand_query",
                        lambda t, n, level=None: _expansion())
    monkeypatch.setattr(search.supadata_search, "search_all",
                        lambda queries, filters=None, **kwargs: {"per_query": [
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
        seen["expand_count"] = n
        return {"corrected": topic, "queries": [f"q{i}" for i in range(6)]}

    def _fake_rank(pq, level=None):
        seen["rank_level"] = level
        return []

    monkeypatch.setattr(s.expand, "expand_query", _fake_expand)
    monkeypatch.setattr(s.supadata_search, "search_all",
                        lambda qs, **kwargs: {"per_query": [], "credits_used": 0, "warning": None})
    monkeypatch.setattr(s.rank, "merge_and_rank", _fake_rank)
    s.discover("physics", limit=3, level="advanced")
    assert seen["expand_level"] == "advanced"
    assert seen["expand_count"] == 6
    assert seen["rank_level"] == "advanced"


def _run_discover(monkeypatch, videos_by_query, *, limit=1, excluded=None):
    monkeypatch.setattr(search.expand, "expand_query", lambda *a, **k: _expansion())
    calls = []

    def fake_one(query, filters=None):
        calls.append(query)
        return {"query": query, "videos": videos_by_query.get(query, []), "billed": 1}

    monkeypatch.setattr(search.supadata_search, "search_one", fake_one)
    monkeypatch.setattr(search.supadata_search.time, "sleep", lambda *_: None)
    out = search.discover("calc", limit=limit, exclude_video_ids=excluded or [])
    return out, calls


def test_normal_consensus_path_searches_exactly_three(monkeypatch):
    out, calls = _run_discover(monkeypatch, {
        "q1": [{"id": "a"}, {"id": "b"}],
        "q2": [{"id": "a"}],
        "q3": [{"id": "b"}],
    }, limit=2)
    assert calls == ["q1", "q2", "q3"]
    assert {v["id"] for v in out["videos"]} == {"a", "b"}


def test_consensus_expands_one_query_at_a_time(monkeypatch):
    _, calls = _run_discover(monkeypatch, {
        "q1": [{"id": "a"}], "q2": [{"id": "b"}], "q3": [{"id": "c"}],
        "q4": [{"id": "a"}],
    })
    assert calls == ["q1", "q2", "q3", "q4"]


def test_no_consensus_uses_at_most_all_six(monkeypatch):
    out, calls = _run_discover(
        monkeypatch, {f"q{i}": [{"id": f"v{i}"}] for i in range(1, 7)}, limit=3,
    )
    assert calls == [f"q{i}" for i in range(1, 7)]
    assert len(out["videos"]) == 3


def test_excluded_consensus_does_not_stop_expansion(monkeypatch):
    _, calls = _run_discover(monkeypatch, {
        "q1": [{"id": "excluded"}, {"id": "keep"}],
        "q2": [{"id": "excluded"}],
        "q3": [{"id": "other"}],
        "q4": [{"id": "keep"}],
    }, excluded=["excluded"])
    assert calls == ["q1", "q2", "q3", "q4"]


def test_partial_error_still_stops_when_three_query_consensus_is_enough(monkeypatch):
    monkeypatch.setattr(search.expand, "expand_query", lambda *a, **k: _expansion())
    calls = []

    def fake_one(query, filters=None):
        calls.append(query)
        if query == "q1":
            raise search.supadata_search.SearchError("temporary")
        return {"query": query, "videos": [{"id": "a"}], "billed": 1}

    monkeypatch.setattr(search.supadata_search, "search_one", fake_one)
    monkeypatch.setattr(search.supadata_search.time, "sleep", lambda *_: None)
    out = search.discover("calc", limit=1)
    assert calls == ["q1", "q2", "q3"]
    assert [video["id"] for video in out["videos"]] == ["a"]
    assert "1 of 3 searches failed" in out["warning"]
