from __future__ import annotations

import threading

import pytest

from backend.app.clip_engine import expand, rank, search
from backend.app.clip_engine.errors import CancellationError
from backend.app.clip_engine.provider_cache import MemoryProviderCache
from backend.app.clip_engine.provider_runtime import GenerationContext


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

    assert expand.PRACTICE_FAST_EXPAND_MODEL == "gemini-3.5-flash"
    assert expand.PRACTICE_FAST_EXPAND_TIMEOUT_MS == 8_000
    assert result == {
        "corrected": "Calculus",
        "queries": ["Calculus", "Derivatives", "Limits"],
        "provider_used": "gemini",
    }
    assert seen == {
        "topic": "calclus", "n": 3, "model": "gemini-3.5-flash",
        "level": "beginner", "cancelled": False,
    }


def test_practice_fast_expansion_falls_back_from_flash_to_pro(monkeypatch):
    calls = []

    def flash_then_pro(*args, model, **kwargs):
        calls.append(model)
        if model == expand.PRACTICE_FAST_EXPAND_MODEL:
            raise RuntimeError("flash unavailable")
        return '{"corrected":"Physics","queries":["Physics","mechanics","waves"]}'

    monkeypatch.setattr(expand, "_practice_fast_gemini_raw", flash_then_pro)

    result = expand.expand_query_practice_fast("physics", 3)

    assert calls == ["gemini-3.5-flash", "gemini-3.1-pro-preview"]
    assert result == {
        "corrected": "Physics",
        "queries": ["Physics", "mechanics", "waves"],
        "provider_used": "gemini",
    }


def test_practice_fast_expansion_uses_deterministic_fallback_after_both_models(monkeypatch):
    calls = []

    def failed_models(*args, model, **kwargs):
        calls.append(model)
        raise RuntimeError("unavailable")

    monkeypatch.setattr(expand, "_practice_fast_gemini_raw", failed_models)

    result = expand.expand_query_practice_fast("physics", 3)

    assert calls == ["gemini-3.5-flash", "gemini-3.1-pro-preview"]
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


def test_discover_practice_fast_threads_runtime_args_and_applies_exclude_top_n(monkeypatch):
    context = GenerationContext("slow")
    cache = MemoryProviderCache()
    seen = {}
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
        seen.update(queries=list(queries), filters=filters, **kwargs)
        return {
            "per_query": [
                {"query": "Calculus", "videos": [
                    {"id": "excluded", "viewCount": 10_000},
                    {"id": "keep", "viewCount": 10},
                    {"id": "third", "viewCount": 1},
                ]},
                {"query": "Derivatives", "videos": [{"id": "keep", "viewCount": 10}]},
                {"query": "Limits", "videos": [{"id": "third", "viewCount": 1}]},
            ],
            "credits_used": 3,
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
    assert seen.pop("should_cancel") is cancel_probe
    assert seen == {
        "queries": ["Calculus", "Derivatives", "Limits"],
        "filters": {"duration": "medium"},
        "language": "es",
        "context": context,
        "cache_store": cache,
    }


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

    assert seen["n"] == 2


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
