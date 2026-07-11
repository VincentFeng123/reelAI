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
                text=topic if index == 1 else f"q{index}",
                family="calc" if index == 1 else f"family-{index}",
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
    monkeypatch.setattr(search.supadata_search, "wait_with_probe", lambda *_: None)
    out = search.discover("calc", limit=limit, exclude_video_ids=excluded or [])
    return out, calls


def test_duplicate_literal_does_not_inflate_consensus(monkeypatch):
    out, calls = _run_discover(monkeypatch, {
        "calc": [{"id": "a"}, {"id": "b"}],
        "q2": [{"id": "a"}],
        "q3": [{"id": "b"}],
    }, limit=2)
    assert calls == ["calc", "calc", "q2", "q3"]
    assert {v["id"] for v in out["videos"]} == {"a", "b"}


def test_consensus_expands_one_query_at_a_time(monkeypatch):
    _, calls = _run_discover(monkeypatch, {
        "calc": [{"id": "a"}], "q2": [{"id": "b"}], "q3": [{"id": "c"}],
        "q4": [{"id": "a"}],
    })
    assert calls == ["calc", "calc", "q2", "q3", "q4"]


def test_no_consensus_uses_at_most_all_six(monkeypatch):
    out, calls = _run_discover(
        monkeypatch,
        {
            "calc": [{"id": "v1"}],
            **{f"q{i}": [{"id": f"v{i}"}] for i in range(2, 7)},
        },
        limit=3,
    )
    assert calls == ["calc", "calc", "q2", "q3", "q4", "q5"]
    assert len(out["videos"]) == 3


def test_excluded_consensus_does_not_stop_expansion(monkeypatch):
    _, calls = _run_discover(monkeypatch, {
        "calc": [{"id": "excluded"}, {"id": "keep"}],
        "q2": [{"id": "other"}],
        "q3": [{"id": "keep"}],
    }, excluded=["excluded"])
    assert calls == ["calc", "calc", "q2", "q3"]


def test_provider_error_is_not_converted_to_empty_success(monkeypatch):
    monkeypatch.setattr(search, "_load_query_plan", lambda *_args: _plan())
    calls = []

    def fake_one(query, filters=None, *args, **kwargs):
        calls.append(query)
        raise search.supadata_search.ProviderTransientError(
            "temporary", provider="supadata", operation="search"
        )

    monkeypatch.setattr(search.supadata_search, "search_one", fake_one)
    monkeypatch.setattr(search.supadata_search, "wait_with_probe", lambda *_: None)
    import pytest
    with pytest.raises(search.supadata_search.ProviderTransientError):
        search.discover("calc", limit=1)
    assert calls == ["calc"]


def test_fast_context_limits_initial_expansion_to_three_queries(monkeypatch):
    from backend.app.clip_engine.provider_runtime import GenerationContext

    seen = {}
    monkeypatch.setattr(search, "_load_query_plan", lambda *_args: _plan())

    def fake_search_all(queries, filters=None, **kwargs):
        seen["queries"] = queries
        return {"per_query": [], "credits_used": 0, "warning": None}

    monkeypatch.setattr(search.supadata_search, "search_all", fake_search_all)
    search.discover("calc", limit=1, context=GenerationContext("fast"))
    assert seen["queries"] == ["calc", "calc", "q2"]


def test_slow_context_requires_all_six_initial_queries(monkeypatch):
    from backend.app.clip_engine.provider_runtime import GenerationContext

    seen = {}
    monkeypatch.setattr(search, "_load_query_plan", lambda *_args: _plan())

    def fake_search_all(queries, filters=None, **kwargs):
        seen.update(queries=queries, minimum_queries=kwargs["minimum_queries"])
        return {"per_query": [], "credits_used": 0, "warning": None}

    monkeypatch.setattr(search.supadata_search, "search_all", fake_search_all)
    search.discover("calc", limit=1, context=GenerationContext("slow"))
    assert seen == {
        "queries": ["calc", "calc", "q2", "q3", "q4", "q5"],
        "minimum_queries": 6,
    }


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
    assert seen == {"queries": ["calc", "calc", "q6"], "minimum_queries": 3}


def test_intro_to_python_keeps_unfiltered_literal_fallback_when_hd_is_empty(monkeypatch):
    plan = _plan("Intro to Python")
    seen = {}
    literal_video = {"id": "python-video", "title": "Python for Beginners"}

    def fake_search_all(queries, filters=None, **kwargs):
        seen.update(queries=list(queries), request_filters=kwargs["request_filters"])
        return {
            "per_query": [
                {"query": queries[0], "videos": [literal_video]},
                *[{"query": query, "videos": []} for query in queries[1:]],
            ],
            "credits_used": 0,
            "warning": None,
        }

    monkeypatch.setattr(search, "_load_query_plan", lambda *_args: plan)
    monkeypatch.setattr(search.supadata_search, "search_all", fake_search_all)

    result = search.discover("Intro to Python", limit=1, breadth=3)

    assert seen["queries"] == ["Intro to Python", "Intro to Python", "q2"]
    assert seen["request_filters"][0]["features"] == []
    assert seen["request_filters"][1]["features"] == ["hd"]
    assert seen["request_filters"][2]["features"] == ["hd"]
    assert [video["id"] for video in result["videos"]] == ["python-video"]


def test_request_mix_preserves_creative_commons_and_duration(monkeypatch):
    seen = {}
    monkeypatch.setattr(search, "_load_query_plan", lambda *_args: _plan())

    def fake_search_all(queries, filters=None, **kwargs):
        seen["request_filters"] = kwargs["request_filters"]
        return {"per_query": [], "credits_used": 0, "warning": None}

    monkeypatch.setattr(search.supadata_search, "search_all", fake_search_all)
    search.discover(
        "calc",
        limit=1,
        breadth=3,
        filters={"creative_commons_only": True, "duration": "medium"},
    )

    request_filters = seen["request_filters"]
    assert [item["duration"] for item in request_filters] == ["medium"] * 3
    assert request_filters[0]["features"] == ["creative-commons"]
    assert request_filters[1]["features"] == ["creative-commons", "hd"]
    assert request_filters[2]["features"] == ["creative-commons", "hd"]
    assert all("subtitles" not in item["features"] for item in request_filters)


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
