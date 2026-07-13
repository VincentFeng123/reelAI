from __future__ import annotations

import asyncio
import threading

import pytest

from backend.app.clip_engine import expand, rank, search
from backend.app.clip_engine.errors import CancellationError
from backend.app.clip_engine.provider_cache import MemoryProviderCache
from backend.app.clip_engine.provider_runtime import GenerationContext


class _FakeGeminiClient:
    def __init__(self, generate):
        class _Models:
            async def generate_content(_self, **kwargs):
                return await generate(**kwargs)

        class _Aio:
            models = _Models()

            async def aclose(_self):
                return None

        self.aio = _Aio()

    def close(self):
        return None


def test_practice_fast_expansion_uses_flash_and_normalizes_model_output(monkeypatch):
    seen = {}

    def fake_raw(topic, n, *, model, level=None, should_cancel=None):
        seen.update(
            topic=topic, n=n, model=model, level=level, cancelled=should_cancel()
        )
        return (
            '{"corrected":"Calculus","queries":['
            '"calculus", "Derivatives", " derivatives ", "Limits"]}'
        )

    monkeypatch.setattr(expand, "_practice_fast_gemini_raw", fake_raw)

    result = expand.expand_query_practice_fast(
        "calclus", 3, level="beginner", should_cancel=lambda: False,
    )

    assert expand.PRACTICE_FAST_EXPAND_MODEL == "gemini-3.1-flash-lite"
    assert expand.PRACTICE_FAST_EXPAND_TIMEOUT_MS == 8_000
    assert result == {
        "corrected": "Calculus",
        "queries": ["Calculus", "Derivatives", "Limits"],
        "provider_used": "gemini",
    }
    assert seen == {
        "topic": "calclus", "n": 3, "model": "gemini-3.1-flash-lite",
        "level": "beginner", "cancelled": False,
    }


def test_failed_expansion_dispatch_is_recorded_once(monkeypatch):
    from google import genai

    async def fail(**_kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(expand.config, "require_gemini_key", lambda: "gemini-test")
    monkeypatch.setattr(genai, "Client", lambda **_kwargs: _FakeGeminiClient(fail))
    context = GenerationContext("fast")

    with pytest.raises(RuntimeError, match="provider unavailable"):
        asyncio.run(
            expand._practice_fast_gemini_raw_async(
                "physics",
                3,
                model=expand.PRACTICE_FAST_EXPAND_MODEL,
                level=None,
                should_cancel=None,
                context=context,
            )
        )

    assert len(context.usage()) == 1
    usage = context.usage()[0]
    assert usage["operation"] == "expansion"
    assert usage["status_code"] is None
    assert usage["error_code"] == "dispatch_failed:RuntimeError"
    summary = context.usage_payload()["summary"]
    assert summary["billing_unknown_calls"] == 1
    assert summary["reserved_worst_case_cost_usd"] > 0


def test_successful_expansion_dispatch_is_not_double_recorded(monkeypatch):
    from google import genai

    class _Response:
        text = '{"corrected":"Physics","queries":["Physics"]}'
        model_version = "gemini-flash-test"
        usage_metadata = {
            "prompt_token_count": 10,
            "candidates_token_count": 5,
            "total_token_count": 15,
        }

    async def succeed(**_kwargs):
        return _Response()

    monkeypatch.setattr(expand.config, "require_gemini_key", lambda: "gemini-test")
    monkeypatch.setattr(genai, "Client", lambda **_kwargs: _FakeGeminiClient(succeed))
    context = GenerationContext("fast")

    result = asyncio.run(
        expand._practice_fast_gemini_raw_async(
            "physics",
            1,
            model=expand.PRACTICE_FAST_EXPAND_MODEL,
            level=None,
            should_cancel=None,
            context=context,
        )
    )

    assert result == _Response.text
    assert len(context.usage()) == 1
    assert context.usage()[0]["status_code"] == 200


def test_practice_fast_expansion_never_falls_back_to_pro(monkeypatch):
    calls = []

    def flash_then_pro(*args, model, **kwargs):
        calls.append(model)
        if model == expand.PRACTICE_FAST_EXPAND_MODEL:
            raise RuntimeError("flash unavailable")
        return '{"corrected":"Physics","queries":["Physics","mechanics","waves"]}'

    monkeypatch.setattr(expand, "_practice_fast_gemini_raw", flash_then_pro)

    result = expand.expand_query_practice_fast("physics", 3)

    assert calls == ["gemini-3.1-flash-lite"]
    assert result == {
        "corrected": "physics",
        "queries": ["physics", "physics explained", "physics lecture"],
        "provider_used": "deterministic",
    }


def test_practice_fast_expansion_uses_deterministic_fallback_after_flash(monkeypatch):
    calls = []

    def failed_models(*args, model, **kwargs):
        calls.append(model)
        raise RuntimeError("unavailable")

    monkeypatch.setattr(expand, "_practice_fast_gemini_raw", failed_models)

    result = expand.expand_query_practice_fast("physics", 3)

    assert calls == ["gemini-3.1-flash-lite"]
    assert result == {
        "corrected": "physics",
        "queries": ["physics", "physics explained", "physics lecture"],
        "provider_used": "deterministic",
    }


def test_practice_fast_expansion_cache_hit_skips_gemini(monkeypatch):
    context = GenerationContext("slow")
    cached = {
        "corrected": "Physics",
        "queries": ["Physics", "mechanics", "waves"],
        "provider_used": "gemini",
    }
    monkeypatch.setattr(expand, "_read_cached_expansion", lambda *_args: cached)
    monkeypatch.setattr(
        expand,
        "_practice_fast_gemini_raw",
        lambda *_args, **_kwargs: pytest.fail("Gemini must not run on a cache hit"),
    )

    result = expand.expand_query_practice_fast("physics", 3, context=context)

    assert result == cached
    assert context.counters()["expansion_cache_hits"] == 1
    usage = context.usage()[0]
    assert usage["operation"] == "expansion"
    assert usage["billable_requests"] == 0
    assert usage["metadata"]["cache_hit"] is True


def test_practice_fast_expansion_stores_success_for_reuse(monkeypatch):
    context = GenerationContext("slow")
    stored: dict[str, dict] = {}
    provider_calls = 0

    monkeypatch.setattr(
        expand,
        "_read_cached_expansion",
        lambda cache_key, _count: stored.get(cache_key),
    )
    monkeypatch.setattr(
        expand,
        "_write_cached_expansion",
        lambda cache_key, result: stored.__setitem__(cache_key, result),
    )

    def fake_raw(*_args, context=None, **_kwargs):
        nonlocal provider_calls
        assert context is not None
        provider_calls += 1
        return '{"corrected":"Physics","queries":["Physics","mechanics","waves"]}'

    monkeypatch.setattr(expand, "_practice_fast_gemini_raw", fake_raw)

    first = expand.expand_query_practice_fast("physics", 3, context=context)
    second = expand.expand_query_practice_fast("physics", 3, context=context)

    assert first == second
    assert provider_calls == 1
    assert context.counters()["expansion_cache_hits"] == 1


def test_practice_fast_expansion_does_not_swallow_cancellation(monkeypatch):
    cancelled = threading.Event()
    cancelled.set()
    monkeypatch.setattr(
        expand,
        "_practice_fast_gemini_raw",
        lambda *args, **kwargs: pytest.fail("provider must not start after cancellation"),
    )

    with pytest.raises(CancellationError):
        expand.expand_query_practice_fast(
            "physics", 3, should_cancel=cancelled.is_set,
        )


def test_practice_fast_rank_is_the_simple_practice_formula():
    ranked = rank.merge_and_rank_practice_fast([
        {
            "query": "physics",
            "query_trust": "ai",
            "videos": [
                {"id": "popular", "title": "Physics reaction", "viewCount": 1_000_000},
                {"id": "consensus", "title": "Physics lecture", "viewCount": 1},
            ],
        },
        {
            "query": "mechanics",
            "query_trust": "ai",
            "videos": [
                {"id": "consensus", "title": "Physics lecture", "viewCount": 1},
            ],
        },
    ])

    assert [video["id"] for video in ranked] == ["consensus", "popular"]
    assert ranked[0]["match_count"] == 2
    assert ranked[0]["matched_queries"] == ["physics", "mechanics"]
    assert "literal_match" not in ranked[0]
    assert "edu_score" not in ranked[0]


@pytest.mark.parametrize(
    ("level", "expected"),
    [
        ("beginner", "AP macroeconomics for beginners"),
        ("intermediate", "intermediate AP macroeconomics"),
        ("advanced", "advanced AP macroeconomics"),
    ],
)
def test_bootstrap_query_encodes_difficulty(level, expected):
    assert search._difficulty_bootstrap_query("AP macroeconomics", level) == expected


@pytest.mark.parametrize(
    ("topic", "level"),
    [
        ("AP macroeconomics for beginners", "beginner"),
        ("Beginner AP macroeconomics", "beginner"),
        ("intermediate AP macroeconomics", "intermediate"),
        ("Advanced AP macroeconomics", "advanced"),
    ],
)
def test_bootstrap_query_does_not_duplicate_equivalent_qualifier(topic, level):
    assert search._difficulty_bootstrap_query(topic, level) == topic


@pytest.mark.parametrize(
    ("topic", "expected"),
    [
        (
            "Carolingian minuscule ligature identification",
            "Carolingian minuscule ligature",
        ),
        ("identification of Carolingian minuscule ligatures", None),
        ("ligature identification", None),
        ("renormalization group in quantum chromodynamics", None),
    ],
)
def test_niche_bootstrap_backoff_removes_only_trailing_search_intent(topic, expected):
    assert search._niche_bootstrap_backoff_query(topic) == expected


def test_bootstrap_searches_qualified_query_without_gemini_and_preserves_raw_topic(monkeypatch):
    calls = []

    def fake_search_all(queries, filters=None, **kwargs):
        calls.append(list(queries))
        return {
            "per_query": [
                {
                    "query": queries[0],
                    "videos": [{"id": "matched", "title": "Physics basics"}],
                }
            ],
            "credits_used": 1,
            "warning": None,
        }

    monkeypatch.setattr(search.supadata_search, "search_all", fake_search_all)
    monkeypatch.setattr(
        search.expand,
        "expand_query_practice_fast",
        lambda *_args, **_kwargs: pytest.fail("bootstrap retrieval must not invoke Gemini"),
    )

    result = search.discover_practice_fast(
        "quantum physics",
        limit=2,
        level="beginner",
        retrieval_profile="bootstrap",
    )

    assert calls == [["quantum physics for beginners"]]
    assert result["corrected"] == "quantum physics"
    assert result["topic_terms"] == ["quantum physics"]
    assert result["queries"] == ["quantum physics for beginners"]
    assert [video["id"] for video in result["videos"]] == ["matched"]


def test_bootstrap_searches_bounded_niche_backoff_without_changing_topic_identity(monkeypatch):
    calls = []

    def fake_search_all(queries, filters=None, **kwargs):
        calls.append(list(queries))
        if len(calls) == 1:
            return {
                "per_query": [
                    {
                        "query": queries[0],
                        "videos": [
                            {"id": "popular-adjacent", "title": "Cursive handwriting", "view_count": 9_000_000},
                        ],
                    }
                ],
                "credits_used": 1,
                "warning": None,
            }
        return {
            "per_query": [
                {
                    "query": queries[0],
                    "videos": [
                        {"id": "direct-1", "title": "Carolingian minuscule ligatures"},
                        {"id": "direct-2", "title": "Reading Carolingian minuscule script"},
                    ],
                },
            ],
            "credits_used": 1,
            "warning": None,
        }

    monkeypatch.setattr(search.supadata_search, "search_all", fake_search_all)
    monkeypatch.setattr(
        search.expand,
        "expand_query_practice_fast",
        lambda *_args, **_kwargs: pytest.fail("bootstrap retrieval must not invoke Gemini"),
    )

    result = search.discover_practice_fast(
        "Carolingian minuscule ligature identification",
        limit=3,
        level="beginner",
        retrieval_profile="bootstrap",
    )

    assert calls == [
        ["Carolingian minuscule ligature identification for beginners"],
        ["Carolingian minuscule ligature for beginners"],
    ]
    assert result["corrected"] == "Carolingian minuscule ligature identification"
    assert result["topic_terms"] == ["Carolingian minuscule ligature identification"]
    assert result["queries"] == [calls[0][0], calls[1][0]]
    assert result["credits_used"] == 2
    assert [video["id"] for video in result["videos"]] == [
        "direct-1",
        "direct-2",
        "popular-adjacent",
    ]


def test_bootstrap_strong_exact_pool_skips_niche_backoff(monkeypatch):
    calls = []

    def fake_search_all(queries, filters=None, **kwargs):
        calls.append(list(queries))
        return {
            "per_query": [{
                "query": queries[0],
                "videos": [{"id": "direct", "title": "Identifying Carolingian minuscule ligatures"}],
            }],
            "credits_used": 1,
            "warning": None,
        }

    monkeypatch.setattr(search.supadata_search, "search_all", fake_search_all)

    result = search.discover_practice_fast(
        "Carolingian minuscule ligature identification",
        limit=3,
        level="beginner",
        retrieval_profile="bootstrap",
    )

    assert calls == [["Carolingian minuscule ligature identification for beginners"]]
    assert [video["id"] for video in result["videos"]] == ["direct"]


def test_bootstrap_reserves_raw_fallback_before_niche_recovery(monkeypatch):
    calls = []

    def fake_search_all(queries, filters=None, **kwargs):
        calls.append(list(queries))
        videos = [] if len(calls) < 3 else [{"id": "too-late"}]
        return {
            "per_query": [{"query": queries[0], "videos": videos}],
            "credits_used": 1,
            "warning": None,
        }

    monkeypatch.setattr(search.supadata_search, "search_all", fake_search_all)

    result = search.discover_practice_fast(
        "Carolingian minuscule ligature identification",
        limit=3,
        breadth=2,
        level="beginner",
        retrieval_profile="bootstrap",
    )

    assert calls == [
        ["Carolingian minuscule ligature identification for beginners"],
        ["Carolingian minuscule ligature identification"],
    ]
    assert result["videos"] == []
    assert result["credits_used"] == 2


def test_bootstrap_retries_raw_topic_once_when_qualified_results_are_ineligible(monkeypatch):
    calls = []
    deadline = 1234.5

    def fake_search_all(queries, filters=None, **kwargs):
        calls.append((list(queries), kwargs.get("deadline_monotonic")))
        if len(calls) == 1:
            return {
                "per_query": [
                    {"query": queries[0], "videos": [{"id": "excluded"}]}
                ],
                "credits_used": 1,
                "warning": None,
            }
        return {
            "per_query": [
                {"query": queries[0], "videos": [{"id": "raw-match"}]}
            ],
            "credits_used": 1,
            "warning": None,
        }

    monkeypatch.setattr(search.supadata_search, "search_all", fake_search_all)
    monkeypatch.setattr(
        search.expand,
        "expand_query_practice_fast",
        lambda *_args, **_kwargs: pytest.fail("raw fallback must precede and skip Gemini"),
    )

    result = search.discover(
        "physics",
        limit=2,
        exclude_video_ids=["excluded"],
        level="advanced",
        practice_fast=True,
        retrieval_profile="bootstrap",
        deadline_monotonic=deadline,
    )

    assert calls == [
        (["advanced physics"], deadline),
        (["physics"], deadline),
    ]
    assert result["queries"] == ["advanced physics", "physics"]
    assert result["credits_used"] == 2
    assert [video["id"] for video in result["videos"]] == ["raw-match"]


def test_discover_practice_fast_threads_runtime_args_and_applies_exclude_top_n(monkeypatch):
    context = GenerationContext("slow")
    cache = MemoryProviderCache()
    seen = []
    cancel_probe = lambda: False

    monkeypatch.setattr(
        search.expand,
        "expand_query_practice_fast",
        lambda topic, n, **kwargs: {
            "corrected": "Calculus",
            "queries": ["Calculus", "Derivatives", "Limits"],
            "provider_used": "gemini",
        },
    )

    def fake_search_all(queries, filters=None, **kwargs):
        seen.append({"queries": list(queries), "filters": filters, **kwargs})
        if len(seen) == 1:
            assert list(queries) == ["calclus", "calclus explained tutorial"]
            return {
                "per_query": [
                    {"query": "calclus", "videos": [
                        {"id": "excluded", "viewCount": 10_000},
                        {"id": "keep", "viewCount": 10},
                    ]},
                    {"query": "calclus explained tutorial", "videos": []},
                ],
                "credits_used": 2,
                "warning": None,
            }
        return {
            "per_query": [
                {"query": "Derivatives", "videos": [{"id": "keep", "viewCount": 10}]},
            ],
            "credits_used": 1,
            "warning": None,
        }

    monkeypatch.setattr(search.supadata_search, "search_all", fake_search_all)

    result = search.discover_practice_fast(
        "calclus",
        limit=1,
        exclude_video_ids=["excluded"],
        breadth=3,
        level="beginner",
        should_cancel=cancel_probe,
        filters={"duration": "medium"},
        language="es",
        context=context,
        cache_store=cache,
        literal_topic="calclus",
        use_query_planner=False,
        query_plan=object(),
    )

    assert result["corrected"] == "Calculus"
    assert [video["id"] for video in result["videos"]] == ["keep"]
    assert result["credits_used"] == 3
    assert seen[0].pop("should_cancel") is cancel_probe
    assert seen[0].pop("parallel_prefix") == 2
    assert seen[0] == {
        "queries": ["calclus", "calclus explained tutorial"],
        "filters": {"duration": "medium"},
        "language": "es",
        "context": context,
        "cache_store": cache,
    }
    assert seen[1]["queries"] == ["Calculus"]


def test_discover_practice_fast_respects_remaining_search_budget(monkeypatch):
    context = GenerationContext("fast")
    context.reserve("search")
    seen = {}

    def fake_expand(topic, n, **kwargs):
        seen["n"] = n
        return {
            "corrected": topic,
            "queries": [topic, f"{topic} explained"],
            "provider_used": "deterministic",
        }

    monkeypatch.setattr(search.expand, "expand_query_practice_fast", fake_expand)
    monkeypatch.setattr(
        search.supadata_search,
        "search_all",
        lambda queries, filters=None, **kwargs: {
            "per_query": [], "credits_used": 0, "warning": None,
        },
    )

    search.discover_practice_fast("physics", limit=1, breadth=8, context=context)

    assert "n" not in seen


def test_discover_practice_fast_defaults_to_eight_queries(monkeypatch):
    seen = {}

    def fake_expand(topic, n, **kwargs):
        seen["n"] = n
        return {
            "corrected": topic,
            "queries": [topic],
            "provider_used": "deterministic",
        }

    monkeypatch.setattr(search.expand, "expand_query_practice_fast", fake_expand)
    monkeypatch.setattr(
        search.supadata_search,
        "search_all",
        lambda queries, filters=None, **kwargs: {
            "per_query": [], "credits_used": 0, "warning": None,
        },
    )

    search.discover_practice_fast("physics", limit=1)

    assert seen["n"] == 8


def test_discover_practice_fast_skips_gemini_when_literal_pool_is_strong(monkeypatch):
    calls = []

    def fake_search_all(queries, filters=None, **kwargs):
        calls.append(list(queries))
        return {
            "per_query": [
                {
                    "query": "physics",
                    "videos": [
                        {"id": f"video-{index}", "title": "Physics lecture", "viewCount": 10}
                        for index in range(4)
                    ],
                },
                {"query": "physics explained tutorial", "videos": []},
            ],
            "credits_used": 2,
            "warning": None,
        }

    monkeypatch.setattr(search.supadata_search, "search_all", fake_search_all)
    monkeypatch.setattr(
        search.expand,
        "expand_query_practice_fast",
        lambda *_args, **_kwargs: pytest.fail("strong literal results must skip Gemini"),
    )

    result = search.discover_practice_fast("physics", limit=4, breadth=3)

    assert calls == [["physics", "physics explained tutorial"]]
    assert result["provider_used"] == "skipped"
    assert len(result["videos"]) == 4
    assert all(video["retrieval_score"] >= 0.60 for video in result["videos"])


def test_search_all_runs_requested_prefix_concurrently(monkeypatch):
    barrier = threading.Barrier(2)

    def fake_search_one(query, *_args, **_kwargs):
        barrier.wait(timeout=1.0)
        return {"query": query, "videos": [], "billed": 1}

    monkeypatch.setattr(search.supadata_search, "search_one", fake_search_one)

    result = search.supadata_search.search_all(
        ["literal", "literal explained tutorial"],
        parallel_prefix=2,
    )

    assert [item["query"] for item in result["per_query"]] == [
        "literal",
        "literal explained tutorial",
    ]
    assert result["credits_used"] == 2
