# backend/tests/clip_engine/test_search.py
from backend.app.clip_engine import search
from backend.app.services.search_query_plan import PlannedSearchQuery, SearchQueryPlan


def _plan(topic: str = "calc") -> SearchQueryPlan:
    return SearchQueryPlan(
        literal_query=topic,
        canonical_query=topic,
        trusted_signature=[topic],
        provenance={topic: ["literal"]},
        queries=[
            PlannedSearchQuery(
                text=f"q{index}",
                family=f"family-{index}",
                provenance="literal" if index == 1 else "ai",
                trust="literal" if index == 1 else "ai",
            )
            for index in range(1, 13)
        ],
        ai_status="validated",
    )


def test_discover_excludes_and_limits(monkeypatch):
    monkeypatch.setattr(search, "_load_query_plan", lambda *_args: _plan())
    monkeypatch.setattr(search.supadata_search, "search_all",
                        lambda queries, filters=None, **kwargs: {"per_query": [
                            {"query": "calc", "videos": [{"id": "a"}, {"id": "b"}, {"id": "c"}]}],
                            "credits_used": 1, "warning": None})
    out = search.discover("calc", limit=2, exclude_video_ids=["a"])
    assert [v["id"] for v in out["videos"]] == ["b", "c"]
    assert out["credits_used"] == 1
    assert out["corrected"] == "calc"


def test_discover_threads_level_to_rank(monkeypatch):
    from backend.app.clip_engine import search as s
    seen = {}

    def _fake_rank(pq, level=None):
        seen["rank_level"] = level
        return []

    monkeypatch.setattr(s, "_load_query_plan", lambda literal, *_args: _plan(literal))
    monkeypatch.setattr(s.supadata_search, "search_all",
                        lambda qs, filters=None, **kwargs: {"per_query": [], "credits_used": 0, "warning": None})
    monkeypatch.setattr(s.rank, "merge_and_rank", _fake_rank)
    s.discover("physics", limit=3, level="advanced")
    assert seen["rank_level"] == "advanced"


def _run_discover(monkeypatch, videos_by_query, *, limit=1, excluded=None):
    monkeypatch.setattr(search, "_load_query_plan", lambda *_args: _plan())
    calls = []

    def fake_one(query, filters=None, *args, **kwargs):
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


def test_provider_error_is_not_converted_to_empty_success(monkeypatch):
    monkeypatch.setattr(search, "_load_query_plan", lambda *_args: _plan())
    calls = []

    def fake_one(query, filters=None, *args, **kwargs):
        calls.append(query)
        raise search.supadata_search.ProviderTransientError(
            "temporary", provider="supadata", operation="search"
        )

    monkeypatch.setattr(search.supadata_search, "search_one", fake_one)
    monkeypatch.setattr(search.supadata_search.time, "sleep", lambda *_: None)
    import pytest
    with pytest.raises(search.supadata_search.ProviderTransientError):
        search.discover("calc", limit=1)
    assert calls == ["q1"]


def test_fast_context_limits_initial_expansion_to_three_queries(monkeypatch):
    from backend.app.clip_engine.provider_runtime import GenerationContext

    seen = {}
    monkeypatch.setattr(search, "_load_query_plan", lambda *_args: _plan())

    def fake_search_all(queries, filters=None, **kwargs):
        seen["queries"] = queries
        return {"per_query": [], "credits_used": 0, "warning": None}

    monkeypatch.setattr(search.supadata_search, "search_all", fake_search_all)
    search.discover("calc", limit=1, context=GenerationContext("fast"))
    assert seen["queries"] == ["q1", "q2", "q3"]


def test_slow_context_requires_all_six_initial_queries(monkeypatch):
    from backend.app.clip_engine.provider_runtime import GenerationContext

    seen = {}
    monkeypatch.setattr(search, "_load_query_plan", lambda *_args: _plan())

    def fake_search_all(queries, filters=None, **kwargs):
        seen.update(queries=queries, minimum_queries=kwargs["minimum_queries"])
        return {"per_query": [], "credits_used": 0, "warning": None}

    monkeypatch.setattr(search.supadata_search, "search_all", fake_search_all)
    search.discover("calc", limit=1, context=GenerationContext("slow"))
    assert seen == {"queries": ["q1", "q2", "q3", "q4", "q5", "q6"], "minimum_queries": 6}


def test_slow_context_uses_three_queries_for_each_continuation(monkeypatch):
    from backend.app.clip_engine.provider_runtime import GenerationContext

    context = GenerationContext("slow")
    context.budget.reserve_pass()
    context.budget.reserve_pass(no_growth=True)
    seen = {}
    monkeypatch.setattr(search, "_load_query_plan", lambda *_args: _plan())

    def fake_search_all(queries, filters=None, **kwargs):
        seen.update(queries=queries, minimum_queries=kwargs["minimum_queries"])
        return {"per_query": [], "credits_used": 0, "warning": None}

    monkeypatch.setattr(search.supadata_search, "search_all", fake_search_all)
    search.discover("calc", limit=1, context=context)
    assert seen == {"queries": ["q7", "q8", "q9"], "minimum_queries": 3}


def test_whitespace_topic_is_rejected_before_expansion(monkeypatch):
    import pytest

    called = False

    def load_plan(*args, **kwargs):
        nonlocal called
        called = True
        return _plan()

    monkeypatch.setattr(search, "_load_query_plan", load_plan)
    with pytest.raises(search.SearchError):
        search.discover(" \t\n ", limit=1)
    assert called is False
