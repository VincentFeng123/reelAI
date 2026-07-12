# backend/tests/clip_engine/test_search.py
import pytest

from backend.app.clip_engine import search
from backend.app.clip_engine.errors import ProviderBudgetExceededError
from backend.app.services.search_query_plan import (
    PlannedSearchQuery,
    SearchQueryPlan,
    semantic_query_family,
)


def _plan(topic: str = "calc") -> SearchQueryPlan:
    one_word_topic = "calculus"
    one_word_synonyms = [f"q{index}" for index in range(2, 10)]
    root_family = semantic_query_family(one_word_topic)
    return SearchQueryPlan(
        literal_query=topic,
        primary_search_query=topic,
        one_word_topic=one_word_topic,
        one_word_synonyms=one_word_synonyms,
        canonical_query=topic,
        trusted_signature=[topic],
        provenance={topic: ["literal"]},
        queries=[
            PlannedSearchQuery(
                text=topic,
                family=root_family,
                provenance="literal",
                trust="literal",
            ),
            PlannedSearchQuery(
                text=one_word_topic,
                family=root_family,
                provenance="ai",
                trust="canonical",
            ),
            *[
                PlannedSearchQuery(
                    text=value,
                    family=root_family,
                    provenance="ai",
                    trust="ai",
                )
                for value in one_word_synonyms
            ],
        ],
        ai_status="validated",
    )


def _intro_to_python_plan() -> SearchQueryPlan:
    topic = "Intro to Python"
    return SearchQueryPlan(
        literal_query=topic,
        primary_search_query="Python programming for beginners",
        one_word_topic="Python",
        one_word_synonyms=["Programming", "Coding"],
        canonical_query="Python",
        trusted_signature=["Python"],
        provenance={
            "intro to python": ["literal"],
            "python": ["ai"],
            "programming": ["ai"],
            "coding": ["ai"],
        },
        queries=[
            PlannedSearchQuery(
                text=topic,
                family="python",
                provenance="literal",
                trust="literal",
            ),
            PlannedSearchQuery(
                text="Python",
                family="python",
                provenance="ai",
                trust="canonical",
            ),
            PlannedSearchQuery(
                text="Programming",
                family="python",
                provenance="ai",
                trust="ai",
            ),
            PlannedSearchQuery(
                text="Coding",
                family="python",
                provenance="ai",
                trust="ai",
            ),
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


def test_one_word_synonyms_share_family_and_do_not_inflate_consensus(monkeypatch):
    out, calls = _run_discover(monkeypatch, {
        "calc": [{"id": "a"}, {"id": "b"}],
        "calculus": [{"id": "a"}, {"id": "b"}],
        "q2": [{"id": "a"}],
        "q3": [{"id": "b"}],
    }, limit=2)
    assert calls == ["calc", "calculus", "q2", "q3", "q4", "q5"]
    assert {v["id"] for v in out["videos"]} == {"a", "b"}
    assert all(
        video["matched_families"] == [semantic_query_family("calculus")]
        for video in out["videos"]
    )


def test_synonym_matches_do_not_stop_remaining_query_budget(monkeypatch):
    _, calls = _run_discover(monkeypatch, {
        "calc": [{"id": "a"}], "q2": [{"id": "b"}], "q3": [{"id": "c"}],
        "q4": [{"id": "a"}],
    })
    assert calls == ["calc", "calculus", "q2", "q3", "q4", "q5"]


def test_no_consensus_uses_at_most_all_six(monkeypatch):
    out, calls = _run_discover(
        monkeypatch,
        {
            "calc": [{"id": "v1"}],
            **{f"q{i}": [{"id": f"v{i}"}] for i in range(2, 6)},
        },
        limit=3,
    )
    assert calls == ["calc", "calculus", "q2", "q3", "q4", "q5"]
    assert len(out["videos"]) == 3


def test_excluded_consensus_does_not_stop_expansion(monkeypatch):
    _, calls = _run_discover(monkeypatch, {
        "calc": [{"id": "excluded"}, {"id": "keep"}],
        "q2": [{"id": "other"}],
        "q3": [{"id": "keep"}],
    }, excluded=["excluded"])
    assert calls == ["calc", "calculus", "q2", "q3", "q4", "q5"]


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
    assert seen["queries"] == ["calc", "calculus", "q2"]


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
        "queries": ["calc", "calculus", "q2", "q3", "q4", "q5"],
        "minimum_queries": 6,
    }


def test_slow_context_rejects_continuation_passes():
    from backend.app.clip_engine.provider_runtime import GenerationContext

    context = GenerationContext("slow")
    context.budget.reserve_pass()
    with pytest.raises(ProviderBudgetExceededError):
        context.budget.reserve_pass(no_growth=True)


def test_slow_context_stays_one_pass_with_empty_query_plan():
    from backend.app.clip_engine.provider_runtime import GenerationContext

    context = GenerationContext("slow")
    context.budget.reserve_pass()
    with pytest.raises(ProviderBudgetExceededError):
        context.budget.reserve_pass(no_growth=True)


def test_intro_to_python_searches_literal_before_hd_ai_terms(monkeypatch):
    plan = _intro_to_python_plan()
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

    assert seen["queries"] == ["Intro to Python", "Python", "Programming"]
    assert "features" not in seen["request_filters"][0]
    assert seen["request_filters"][1]["features"] == ["hd"]
    assert seen["request_filters"][2]["features"] == ["hd"]
    assert [video["id"] for video in result["videos"]] == ["python-video"]


def test_literal_hd_fills_only_after_validated_ai_terms_are_exhausted(monkeypatch):
    plan = _intro_to_python_plan().model_copy(update={"one_word_synonyms": []})
    seen = {}

    def fake_search_all(queries, filters=None, **kwargs):
        seen.update(queries=list(queries), request_filters=kwargs["request_filters"])
        return {"per_query": [], "credits_used": 0, "warning": None}

    monkeypatch.setattr(search, "_load_query_plan", lambda *_args: plan)
    monkeypatch.setattr(search.supadata_search, "search_all", fake_search_all)

    search.discover("Intro to Python", limit=1, breadth=3)

    assert seen["queries"] == ["Intro to Python", "Python", "Intro to Python"]
    assert "features" not in seen["request_filters"][0]
    assert seen["request_filters"][1]["features"] == ["hd"]
    assert seen["request_filters"][2]["features"] == ["hd"]


def test_long_topic_uses_ai_summary_as_unrestricted_primary_query(monkeypatch):
    literal = (
        "Explain how plants convert sunlight water and carbon dioxide into "
        "stored chemical energy during photosynthesis"
    )
    plan = SearchQueryPlan(
        literal_query=literal,
        primary_search_query="photosynthesis light energy conversion",
        one_word_topic="Photosynthesis",
        one_word_synonyms=["Photochemistry", "Bioenergetics"],
        canonical_query="Photosynthesis",
        queries=[],
        ai_status="validated",
    )
    seen = {}

    def fake_search_all(queries, filters=None, **kwargs):
        seen.update(queries=list(queries), request_filters=kwargs["request_filters"])
        return {
            "per_query": [{"query": query, "videos": []} for query in queries],
            "credits_used": 0,
            "warning": None,
        }

    def fake_rank(per_query, level=None):
        seen["annotated"] = per_query
        return []

    monkeypatch.setattr(search.supadata_search, "search_all", fake_search_all)
    monkeypatch.setattr(search.rank, "merge_and_rank", fake_rank)

    result = search.discover(literal, limit=1, breadth=3, query_plan=plan)

    assert seen["queries"] == [
        "photosynthesis light energy conversion",
        "Photosynthesis",
        "Photochemistry",
    ]
    assert "features" not in seen["request_filters"][0]
    assert seen["request_filters"][1]["features"] == ["hd"]
    assert seen["request_filters"][2]["features"] == ["hd"]
    assert seen["annotated"][0]["query_trust"] == "literal"
    assert seen["annotated"][0]["query_provenance"] == "ai_summary"
    assert result["query_plan"].literal_query == literal


def test_long_topic_uses_bounded_primary_when_ai_expansion_is_unavailable(monkeypatch):
    literal = " ".join(f"paragraphword{index}" for index in range(20))
    plan = SearchQueryPlan(
        literal_query=literal,
        primary_search_query="bounded practical search phrase",
        canonical_query=literal,
        queries=[],
        ai_status="unavailable",
    )
    seen = {}

    def fake_search_all(queries, filters=None, **kwargs):
        seen.update(queries=list(queries), request_filters=kwargs["request_filters"])
        return {"per_query": [], "credits_used": 0, "warning": None}

    monkeypatch.setattr(search.supadata_search, "search_all", fake_search_all)

    search.discover(literal, limit=1, breadth=3, query_plan=plan)

    assert seen["queries"] == [
        "bounded practical search phrase",
        "bounded practical search phrase",
    ]
    assert "features" not in seen["request_filters"][0]
    assert seen["request_filters"][1]["features"] == ["hd"]


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
