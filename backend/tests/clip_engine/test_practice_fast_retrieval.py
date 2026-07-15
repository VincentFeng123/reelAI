from __future__ import annotations

import asyncio
import json
import threading
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone

import pytest

from backend.app.clip_engine import expand, rank, search, segment_cache
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


def _intent_expansion_json(
    *,
    corrected: str,
    source_phrase: str,
    queries: list[str],
) -> str:
    return json.dumps({
        "corrected": corrected,
        "intent_constraints": [{
            "constraint_id": "subject",
            "source_phrase": source_phrase,
            "requirement": f"Teach {corrected}",
        }],
        "queries": [
            {
                "text": query,
                "preserved_constraint_ids": ["subject"],
            }
            for query in queries
        ],
    })


def test_practice_fast_expansion_uses_flash_and_normalizes_model_output(monkeypatch):
    seen = {}

    def fake_raw(topic, n, *, model, level=None, should_cancel=None):
        seen.update(
            topic=topic, n=n, model=model, level=level, cancelled=should_cancel()
        )
        return _intent_expansion_json(
            corrected="Calculus",
            source_phrase="calclus",
            queries=["calculus spoken lecture", "Derivatives", " derivatives ", "Limits"],
        )

    monkeypatch.setattr(expand, "_practice_fast_gemini_raw", fake_raw)

    result = expand.expand_query_practice_fast(
        "calclus", 3, level="beginner", should_cancel=lambda: False,
    )

    assert expand.PRACTICE_FAST_EXPAND_MODEL == "gemini-3.1-flash-lite"
    assert expand.PRACTICE_FAST_EXPAND_TIMEOUT_MS == 10_000
    assert result == {
        "corrected": "Calculus",
        "queries": ["calculus spoken lecture", "Derivatives", "Limits"],
        "provider_used": "gemini",
    }
    assert seen == {
        "topic": "calclus", "n": 3, "model": "gemini-3.1-flash-lite",
        "level": "beginner", "cancelled": False,
    }


def test_practice_fast_expansion_requests_focused_sources() -> None:
    prompt = " ".join(expand._PRACTICE_FAST_SYSTEM.casefold().split())

    assert "prefer focused teaching videos" in prompt
    assert "never query for a full course" in prompt
    assert "unless the user explicitly requests that format" in prompt


def test_practice_fast_expansion_keeps_only_queries_preserving_every_intent_constraint(
    monkeypatch,
):
    payload = {
        "corrected": "chain rule worked example",
        "intent_constraints": [
            {
                "constraint_id": "subject",
                "source_phrase": "chain rule",
                "requirement": "Teach the chain rule",
            },
            {
                "constraint_id": "task",
                "source_phrase": "worked example",
                "requirement": "Work through a concrete example to its answer",
            },
        ],
        "queries": [
            {
                "text": "chain rule definition lecture",
                "preserved_constraint_ids": ["subject"],
            },
            {
                "text": "chain rule solved derivative walkthrough",
                "preserved_constraint_ids": ["subject", "task"],
            },
        ],
    }
    monkeypatch.setattr(
        expand,
        "_practice_fast_gemini_raw",
        lambda *_args, **_kwargs: json.dumps(payload),
    )

    result = expand.expand_query_practice_fast("chain rule worked example", 3)

    assert result == {
        "corrected": "chain rule worked example",
        "queries": ["chain rule solved derivative walkthrough"],
        "provider_used": "gemini",
    }


def test_practice_fast_expansion_falls_back_to_exact_request_when_contract_drops_qualifier(
    monkeypatch,
):
    payload = {
        "corrected": "chain rule worked example",
        "intent_constraints": [
            {
                "constraint_id": "subject",
                "source_phrase": "chain rule",
                "requirement": "Teach the chain rule",
            },
        ],
        "queries": [
            {
                "text": "chain rule definition lecture",
                "preserved_constraint_ids": ["subject"],
            },
        ],
    }
    monkeypatch.setattr(
        expand,
        "_practice_fast_gemini_raw",
        lambda *_args, **_kwargs: json.dumps(payload),
    )

    result = expand.expand_query_practice_fast("chain rule worked example", 3)

    assert result == {
        "corrected": "chain rule worked example",
        "queries": ["chain rule worked example"],
        "provider_used": "literal_fallback",
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
    seen = {}

    class _Response:
        text = '{"corrected":"Physics","queries":["Physics"]}'
        model_version = "gemini-flash-test"
        usage_metadata = {
            "prompt_token_count": 10,
            "candidates_token_count": 5,
            "total_token_count": 15,
        }

    async def succeed(**kwargs):
        seen["config"] = kwargs["config"]
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
    config = seen["config"]
    assert str(config.thinking_config.thinking_level).casefold().endswith("low")
    assert config.temperature is None
    assert config.max_output_tokens == 1_024


def test_practice_fast_expansion_never_falls_back_to_pro(monkeypatch):
    calls = []

    def flash_then_pro(*args, model, **kwargs):
        calls.append(model)
        if model == expand.PRACTICE_FAST_EXPAND_MODEL:
            raise RuntimeError("flash unavailable")
        return _intent_expansion_json(
            corrected="Physics",
            source_phrase="physics",
            queries=["Physics", "mechanics", "waves"],
        )

    monkeypatch.setattr(expand, "_practice_fast_gemini_raw", flash_then_pro)

    result = expand.expand_query_practice_fast("physics", 3)

    assert calls == ["gemini-3.1-flash-lite"]
    assert result == {
        "corrected": "physics",
        "queries": ["physics"],
        "provider_used": "literal_fallback",
    }


def test_practice_fast_expansion_uses_literal_fallback_after_flash(monkeypatch):
    calls = []

    def failed_models(*args, model, **kwargs):
        calls.append(model)
        raise RuntimeError("unavailable")

    monkeypatch.setattr(expand, "_practice_fast_gemini_raw", failed_models)

    result = expand.expand_query_practice_fast("physics", 3)

    assert calls == ["gemini-3.1-flash-lite"]
    assert result == {
        "corrected": "physics",
        "queries": ["physics"],
        "provider_used": "literal_fallback",
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
        return _intent_expansion_json(
            corrected="Physics",
            source_phrase="physics",
            queries=["Physics", "mechanics", "waves"],
        )

    monkeypatch.setattr(expand, "_practice_fast_gemini_raw", fake_raw)

    first = expand.expand_query_practice_fast("physics", 3, context=context)
    second = expand.expand_query_practice_fast("physics", 3, context=context)

    assert first == second
    assert provider_calls == 1
    assert context.counters()["expansion_cache_hits"] == 1


def test_practice_fast_expansion_cache_outlives_segment_ttl_and_expires(monkeypatch):
    cached_row = {
        "response_json": json.dumps({
            "version": expand.PRACTICE_FAST_EXPAND_CACHE_VERSION,
            "corrected": "Physics",
            "queries": ["physics lecture", "mechanics", "waves"],
        }),
        "created_at": (
            datetime.now(timezone.utc)
            - timedelta(seconds=segment_cache.SEGMENT_CACHE_TTL_SEC - 60)
        ).isoformat(),
    }
    monkeypatch.setattr(expand, "get_conn", lambda *args, **kwargs: nullcontext(object()))
    monkeypatch.setattr(expand, "fetch_one", lambda *_args, **_kwargs: cached_row)
    monkeypatch.setattr(expand, "_write_cached_expansion", lambda *_args: None)
    provider_calls = 0

    def fake_raw(*_args, **_kwargs):
        nonlocal provider_calls
        provider_calls += 1
        return _intent_expansion_json(
            corrected="Physics",
            source_phrase="physics",
            queries=["Physics", "mechanics", "waves"],
        )

    monkeypatch.setattr(expand, "_practice_fast_gemini_raw", fake_raw)

    context = GenerationContext("fast")
    cached = expand.expand_query_practice_fast("physics", 3, context=context)

    assert expand.PRACTICE_FAST_EXPAND_CACHE_TTL_SEC == 2 * segment_cache.SEGMENT_CACHE_TTL_SEC
    assert cached["queries"] == ["physics lecture", "mechanics", "waves"]
    assert provider_calls == 0
    assert context.counters()["expansion_cache_hits"] == 1
    assert context.usage()[0]["billable_requests"] == 0

    cached_row["created_at"] = (
        datetime.now(timezone.utc)
        - timedelta(seconds=segment_cache.SEGMENT_CACHE_TTL_SEC + 60)
    ).isoformat()
    still_cached = expand.expand_query_practice_fast(
        "physics",
        3,
        context=GenerationContext("fast"),
    )

    assert still_cached["queries"] == ["physics lecture", "mechanics", "waves"]
    assert provider_calls == 0

    cached_row["created_at"] = (
        datetime.now(timezone.utc)
        - timedelta(seconds=expand.PRACTICE_FAST_EXPAND_CACHE_TTL_SEC + 60)
    ).isoformat()
    refreshed = expand.expand_query_practice_fast(
        "physics",
        3,
        context=GenerationContext("fast"),
    )

    assert refreshed["queries"] == ["Physics", "mechanics", "waves"]
    assert provider_calls == 1


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


def test_long_topic_components_preserve_searchable_literal_subtopics():
    topic = (
        "How DNA replication proofreading, RNA transcription, ribosomal translation, "
        "membrane transport, ATP production, and feedback regulation work together to "
        "maintain cellular homeostasis and pass genetic information across cell divisions"
    )

    assert search._long_topic_component_queries(topic) == [
        "DNA replication proofreading",
        "RNA transcription",
        "ribosomal translation",
        "membrane transport",
        "ATP production",
        "feedback regulation",
    ]


def test_long_topic_bootstrap_falls_back_to_component_without_changing_identity(monkeypatch):
    topic = (
        "How DNA replication proofreading, RNA transcription, ribosomal translation, "
        "membrane transport, ATP production, and feedback regulation work together to "
        "maintain cellular homeostasis and pass genetic information across cell divisions"
    )
    calls = []

    def fake_search_all(queries, filters=None, **kwargs):
        calls.append(list(queries))
        videos = (
            [{"id": "dna-lesson", "title": "DNA replication proofreading explained"}]
            if queries == ["DNA replication proofreading for beginners"]
            else []
        )
        return {
            "per_query": [{"query": query, "videos": videos if index == 0 else []}
                          for index, query in enumerate(queries)],
            "credits_used": len(queries),
            "warning": None,
        }

    monkeypatch.setattr(search.supadata_search, "search_all", fake_search_all)
    monkeypatch.setattr(
        search.expand,
        "expand_query_practice_fast",
        lambda *_args, **_kwargs: pytest.fail("bootstrap retrieval must not invoke Gemini"),
    )

    result = search.discover_practice_fast(
        topic,
        limit=3,
        breadth=3,
        level="beginner",
        retrieval_profile="bootstrap",
    )

    assert calls == [
        [f"{topic} for beginners"],
        [topic],
        ["DNA replication proofreading for beginners"],
    ]
    assert result["corrected"] == topic
    assert result["topic_terms"] == [topic]
    assert result["queries"] == [calls[0][0], calls[1][0], calls[2][0]]
    assert [video["id"] for video in result["videos"]] == ["dna-lesson"]


def test_long_topic_deep_searches_only_bounded_ai_queries(monkeypatch):
    topic = (
        "How DNA replication proofreading, RNA transcription, ribosomal translation, "
        "membrane transport, ATP production, and feedback regulation work together to "
        "maintain cellular homeostasis and pass genetic information across cell divisions"
    )
    search_calls: list[list[str]] = []
    expansion_calls: list[tuple[str, int]] = []

    def fake_search_all(queries, filters=None, **kwargs):
        search_calls.append(list(queries))
        return {
            "per_query": [{"query": query, "videos": []} for query in queries],
            "credits_used": len(queries),
            "warning": None,
        }

    def fake_expand(expansion_topic, n, **_kwargs):
        expansion_calls.append((expansion_topic, n))
        return {
            "corrected": expansion_topic,
            "queries": [
                expansion_topic,
                f"{expansion_topic} explained tutorial",
                "cell biology information flow and homeostasis",
            ],
            "provider_used": "gemini",
        }

    monkeypatch.setattr(search.supadata_search, "search_all", fake_search_all)
    monkeypatch.setattr(search.expand, "expand_query_practice_fast", fake_expand)

    result = search.discover_practice_fast(
        topic,
        limit=3,
        breadth=8,
        level="advanced",
        retrieval_profile="deep",
    )

    expanded = "cell biology information flow and homeostasis"
    expected = [topic, f"{topic} explained tutorial", expanded]
    assert search_calls == [expected]
    assert expansion_calls == [(topic, 3)]
    assert result["queries"] == expected
    assert result["topic_terms"] == [topic]
    assert result["credits_used"] == 3


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


def test_deep_search_runs_only_ai_queries_without_broadening_selection(monkeypatch):
    topic = "chain-rule worked example"
    calls = []

    def fake_search_all(queries, filters=None, **kwargs):
        calls.append((list(queries), kwargs.get("parallel_prefix")))
        return {
            "per_query": [
                {
                    "query": query,
                    "videos": [
                        {
                            "id": f"video-{index}",
                            "title": "Complete chain rule derivative lesson",
                        }
                    ],
                }
                for index, query in enumerate(queries)
            ],
            "credits_used": len(queries),
            "warning": None,
        }

    expansion_calls = []

    def fake_expand(expansion_topic, n, **kwargs):
        expansion_calls.append((expansion_topic, n, kwargs.get("level")))
        return {
            "corrected": expansion_topic,
            "queries": [expansion_topic, "chain rule derivative example"],
            "provider_used": "gemini",
        }

    monkeypatch.setattr(search.supadata_search, "search_all", fake_search_all)
    monkeypatch.setattr(search.expand, "expand_query_practice_fast", fake_expand)

    result = search.discover_practice_fast(
        topic,
        limit=2,
        breadth=3,
        level="beginner",
        context=GenerationContext("fast"),
        retrieval_profile="deep",
    )

    assert calls == [([topic, "chain rule derivative example"], 2)]
    assert expansion_calls == [(topic, 3, None)]
    assert result["queries"] == [topic, "chain rule derivative example"]
    assert result["topic_terms"] == [topic]


def test_deep_source_ranking_does_not_filter_by_learner_level(monkeypatch):
    rank_levels = []

    monkeypatch.setattr(
        search.expand,
        "expand_query_practice_fast",
        lambda topic, _n, **_kwargs: {
            "corrected": topic,
            "queries": [topic],
            "provider_used": "gemini",
        },
    )
    monkeypatch.setattr(
        search.supadata_search,
        "search_all",
        lambda queries, filters=None, **_kwargs: {
            "per_query": [
                {
                    "query": queries[0],
                    "videos": [{"id": "topic-source", "title": "Topic lesson"}],
                }
            ],
            "credits_used": 1,
            "warning": None,
        },
    )

    def fake_rank(result_sets, level=None):
        rank_levels.append(level)
        return [
            {
                **result_sets[0]["videos"][0],
                "matched_families": ["topic"],
            }
        ]

    monkeypatch.setattr(search.rank, "merge_and_rank", fake_rank)

    result = search.discover_practice_fast(
        "topic",
        limit=1,
        level="advanced",
        retrieval_profile="deep",
    )

    assert rank_levels == [None]
    assert [video["id"] for video in result["videos"]] == ["topic-source"]


def test_deep_search_uses_ai_acronym_expansion_in_one_stage(monkeypatch):
    topic = "NLP attention mechanism"
    calls = []
    expansion_calls = []

    def fake_search_all(queries, filters=None, **kwargs):
        calls.append(list(queries))
        videos = [
            {
                "id": "direct",
                "title": "NLP Attention Mechanisms",
                "description": "Attention in natural language processing models.",
            }
        ]
        return {
            "per_query": [
                {"query": query, "videos": videos if index == 0 else []}
                for index, query in enumerate(queries)
            ],
            "credits_used": len(queries),
            "warning": None,
        }

    def fake_expand(expansion_topic, n, **_kwargs):
        expansion_calls.append((expansion_topic, n))
        return {
            "corrected": expansion_topic,
            "queries": [
                expansion_topic,
                "natural language processing attention mechanism",
            ],
            "provider_used": "gemini",
        }

    monkeypatch.setattr(search.supadata_search, "search_all", fake_search_all)
    monkeypatch.setattr(search.expand, "expand_query_practice_fast", fake_expand)

    result = search.discover_practice_fast(
        topic,
        limit=3,
        breadth=3,
        level="beginner",
        context=GenerationContext("fast"),
        retrieval_profile="deep",
    )

    assert calls == [[topic, "natural language processing attention mechanism"]]
    assert expansion_calls == [(topic, 3)]
    assert "direct" in [video["id"] for video in result["videos"]]


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


def test_bootstrap_niche_coverage_only_counts_videos_that_will_be_analyzed(monkeypatch):
    calls = []

    def fake_search_all(queries, filters=None, **kwargs):
        calls.append(list(queries))
        if len(calls) == 1:
            return {
                "per_query": [{
                    "query": queries[0],
                    "videos": [
                        {"id": "generic-1", "title": "Advanced calligraphy flourishes"},
                        {"id": "generic-2", "title": "How to improve cursive handwriting"},
                        {"id": "generic-3", "title": "Beautiful lettering tutorial"},
                        {
                            "id": "buried-match",
                            "title": "Carolingian minuscule ligature identification",
                        },
                    ],
                }],
                "credits_used": 1,
                "warning": None,
            }
        return {
            "per_query": [{
                "query": queries[0],
                "videos": [{
                    "id": "direct-recovery",
                    "title": "Carolingian minuscule ligatures explained",
                }],
            }],
            "credits_used": 1,
            "warning": None,
        }

    monkeypatch.setattr(search.supadata_search, "search_all", fake_search_all)
    monkeypatch.setattr(
        search.rank,
        "merge_and_rank",
        lambda result_sets, **_kwargs: [
            video
            for result_set in result_sets
            for video in result_set.get("videos") or []
        ],
    )

    result = search.discover_practice_fast(
        "Carolingian minuscule ligature identification",
        limit=3,
        level="advanced",
        retrieval_profile="bootstrap",
    )

    assert calls == [
        ["advanced Carolingian minuscule ligature identification"],
        ["advanced Carolingian minuscule ligature"],
    ]
    assert result["videos"][0]["id"] == "direct-recovery"


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
        videos_by_query = {
            "calclus": [{"id": "excluded", "viewCount": 10_000}],
            "Derivatives": [{"id": "keep", "viewCount": 10}],
        }
        return {
            "per_query": [
                {"query": query, "videos": videos_by_query.get(query, [])}
                for query in queries
            ],
            "credits_used": len(queries),
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
    assert [call.pop("should_cancel") for call in seen] == [cancel_probe]
    assert [call.pop("parallel_prefix") for call in seen] == [3]
    assert seen == [
        {
            "queries": ["Calculus", "Derivatives", "Limits"],
            "filters": {"duration": "medium"},
            "language": "es",
            "context": context,
            "cache_store": cache,
        },
    ]


def test_discover_practice_fast_limits_ai_queries_to_remaining_search_budget(monkeypatch):
    context = GenerationContext("fast")
    context.reserve("search")
    seen = {"queries": []}

    def fake_expand(topic, n, **kwargs):
        seen["n"] = n
        return {
            "corrected": topic,
            "queries": [topic, f"{topic} explained"],
            "provider_used": "gemini",
        }

    monkeypatch.setattr(search.expand, "expand_query_practice_fast", fake_expand)

    def fake_search_all(queries, filters=None, **kwargs):
        seen["queries"].append(list(queries))
        return {
            "per_query": [], "credits_used": 0, "warning": None,
        }

    monkeypatch.setattr(search.supadata_search, "search_all", fake_search_all)

    search.discover_practice_fast("physics", limit=1, breadth=8, context=context)

    assert seen["n"] == 3
    assert seen["queries"] == [["physics", "physics explained"]]


def test_discover_practice_fast_caps_ai_search_to_three_queries(monkeypatch):
    seen = {}

    def fake_expand(topic, n, **kwargs):
        seen["n"] = n
        return {
            "corrected": topic,
            "queries": [topic],
            "provider_used": "gemini",
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

    assert seen["n"] == 3


@pytest.mark.parametrize(("mode", "source_count"), [("fast", 2), ("slow", 3)])
def test_literal_sufficient_retrieval_still_uses_bounded_ai_diversity(
    monkeypatch,
    mode,
    source_count,
):
    calls = []

    def fake_search_all(queries, filters=None, **kwargs):
        calls.append(list(queries))
        return {
            "per_query": [
                {
                    "query": queries[0],
                    "videos": [
                        {"id": f"video-{index}", "title": "Physics lecture", "viewCount": 10}
                        for index in range(source_count)
                    ],
                },
            ],
            "credits_used": 1,
            "warning": None,
        }

    expansion_calls = []

    def fake_expand(topic, n, **_kwargs):
        expansion_calls.append((topic, n))
        return {
            "corrected": topic,
            "queries": [topic, f"{topic} explained tutorial"],
            "provider_used": "gemini",
        }

    monkeypatch.setattr(search.supadata_search, "search_all", fake_search_all)
    monkeypatch.setattr(search.expand, "expand_query_practice_fast", fake_expand)

    result = search.discover_practice_fast(
        "physics",
        limit=source_count * 2,
        breadth=8,
        context=GenerationContext(mode),
    )

    assert calls == [["physics", "physics explained tutorial"]]
    assert expansion_calls == [("physics", 3)]
    assert result["provider_used"] == "gemini"
    assert len(result["videos"]) == source_count
    assert all(video["retrieval_score"] >= 0.60 for video in result["videos"])


@pytest.mark.parametrize(
    ("mode", "source_budget", "expected_follow_up"),
    [
        ("fast", 2, ["physics mechanics"]),
        ("slow", 3, ["physics mechanics", "physics waves"]),
    ],
)
def test_deep_retrieval_runs_one_expansion_and_one_concurrent_search_stage(
    monkeypatch,
    mode,
    source_budget,
    expected_follow_up,
):
    topic = "physics"
    search_calls = []
    expansion_calls = []

    def fake_search_all(queries, filters=None, **_kwargs):
        search_calls.append(list(queries))
        return {
            "per_query": [
                {
                    "query": query,
                    "videos": [
                        {
                            "id": f"video-{len(search_calls)}-{index}",
                            "title": "Physics lesson",
                        }
                    ],
                }
                for index, query in enumerate(queries)
            ],
            "credits_used": len(queries),
            "warning": None,
        }

    def fake_expand(expansion_topic, n, **kwargs):
        expansion_calls.append((expansion_topic, n, kwargs.get("level")))
        return {
            "corrected": expansion_topic,
            "queries": [
                expansion_topic,
                "physics mechanics",
                "physics waves",
            ],
            "provider_used": "gemini",
        }

    monkeypatch.setattr(search.supadata_search, "search_all", fake_search_all)
    monkeypatch.setattr(search.expand, "expand_query_practice_fast", fake_expand)

    result = search.discover_practice_fast(
        topic,
        limit=source_budget * 2,
        breadth=8,
        context=GenerationContext(mode),
        retrieval_profile="deep",
    )

    assert search_calls == [[topic, *expected_follow_up]]
    assert expansion_calls == [(topic, 3, None)]
    assert result["queries"] == [topic, *expected_follow_up]
    assert result["topic_terms"] == [topic]


@pytest.mark.parametrize(
    ("mode", "limit", "analysis_prefix", "expected_ids"),
    [
        ("fast", 4, 2, ["literal-0", "expanded", "literal-1", "literal-2"]),
        (
            "slow",
            6,
            3,
            ["literal-0", "literal-1", "expanded", "literal-2", "literal-3", "literal-4"],
        ),
    ],
)
def test_discovery_oversampling_puts_ai_diversity_in_analysis_prefix(
    monkeypatch,
    mode,
    limit,
    analysis_prefix,
    expected_ids,
):
    ranked = [
        {
            "id": f"literal-{index}",
            "literal_match": True,
            "matched_families": ["physics"],
        }
        for index in range(limit)
    ] + [
        {
            "id": "expanded",
            "literal_match": False,
            "matched_families": ["mechanics"],
        }
    ]
    monkeypatch.setattr(
        search.expand,
        "expand_query_practice_fast",
        lambda topic, n, **_kwargs: {
            "corrected": topic,
            "queries": [topic, "mechanics"],
            "provider_used": "gemini",
        },
    )
    monkeypatch.setattr(
        search.supadata_search,
        "search_all",
        lambda queries, filters=None, **_kwargs: {
            "per_query": [{"query": query, "videos": []} for query in queries],
            "credits_used": 0,
            "warning": None,
        },
    )
    monkeypatch.setattr(
        search.rank,
        "merge_and_rank",
        lambda result_sets, **_kwargs: ranked if len(result_sets) > 1 else ranked[:1],
    )

    result = search.discover_practice_fast(
        "physics",
        limit=limit,
        breadth=8,
        context=GenerationContext(mode),
        retrieval_profile="deep",
    )

    assert [video["id"] for video in result["videos"]] == expected_ids
    assert result["videos"][0]["literal_match"] is True
    assert sum(not video["literal_match"] for video in result["videos"][:analysis_prefix]) == 1


def test_analysis_prefix_uses_a_different_channel_when_available():
    ranked = [
        {
            "id": "top-source",
            "channel": "Animation Academy",
            "literal_match": True,
            "matched_families": ["cell-division"],
        },
        {
            "id": "same-channel",
            "channel": "Animation Academy",
            "literal_match": False,
            "matched_families": ["meiosis"],
        },
        {
            "id": "different-channel",
            "channel": "Open Biology Lecture",
            "literal_match": False,
            "matched_families": ["cell-division"],
        },
    ]

    selected = search._select_ranked_candidates(
        ranked,
        limit=3,
        excluded=set(),
        analysis_prefix=2,
    )

    assert [video["id"] for video in selected] == [
        "top-source",
        "different-channel",
        "same-channel",
    ]


def test_analysis_prefix_does_not_promote_a_distant_channel_result():
    ranked = [
        {
            "id": f"ranked-{index}",
            "channel": "Top Lecture Channel" if index < 5 else "Distant Channel",
            "literal_match": False,
            "matched_families": ["cell-division"],
        }
        for index in range(6)
    ]

    selected = search._select_ranked_candidates(
        ranked,
        limit=6,
        excluded=set(),
        analysis_prefix=2,
    )

    assert [video["id"] for video in selected[:2]] == ["ranked-0", "ranked-1"]


def test_analysis_prefix_defers_multi_hour_courses_when_focused_sources_exist():
    ranked = [
        {
            "id": "full-course",
            "channel": "Open College",
            "duration": 42_828,
            "literal_match": False,
            "matched_families": ["calculus"],
        },
        {
            "id": "focused-limits",
            "channel": "Open College",
            "duration": 5_246,
            "literal_match": False,
            "matched_families": ["limits"],
        },
        {
            "id": "focused-derivatives",
            "channel": "Math Lessons",
            "duration": 2_400,
            "literal_match": False,
            "matched_families": ["derivatives"],
        },
    ]

    selected = search._select_ranked_candidates(
        ranked,
        limit=3,
        excluded=set(),
        analysis_prefix=2,
    )

    assert [video["id"] for video in selected] == [
        "focused-limits",
        "focused-derivatives",
        "full-course",
    ]


def test_multi_hour_source_remains_available_without_enough_focused_sources():
    ranked = [
        {
            "id": "full-course",
            "duration": 42_828,
            "literal_match": False,
            "matched_families": ["calculus"],
        },
        {
            "id": "focused-limits",
            "duration": 5_246,
            "literal_match": False,
            "matched_families": ["limits"],
        },
    ]

    selected = search._select_ranked_candidates(
        ranked,
        limit=2,
        excluded=set(),
        analysis_prefix=2,
    )

    assert [video["id"] for video in selected] == [
        "full-course",
        "focused-limits",
    ]


def test_multi_hour_source_remains_in_oversampled_discovery_pool():
    ranked = [
        {
            "id": "full-course",
            "duration": 42_828,
            "literal_match": False,
            "matched_families": ["calculus"],
        },
        *[
            {
                "id": f"focused-{index}",
                "duration": 1_800 + index,
                "literal_match": False,
                "matched_families": [f"facet-{index}"],
            }
            for index in range(5)
        ],
    ]

    selected = search._select_ranked_candidates(
        ranked,
        limit=4,
        excluded=set(),
        analysis_prefix=2,
    )

    assert [video["id"] for video in selected] == [
        "focused-0",
        "focused-1",
        "full-course",
        "focused-2",
    ]


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
