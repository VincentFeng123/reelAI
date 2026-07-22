from __future__ import annotations

import asyncio
import json
import threading
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone

import pytest

from backend.app.clip_engine import expand, rank, search, segment_cache
from backend.app.clip_engine.errors import (
    CancellationError,
    ProviderConfigurationError,
    ProviderRequestError,
    ProviderResponseValidationError,
    ProviderTransientError,
)
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
        "summary_preserved_constraint_ids": ["subject"],
        "joint_structures": [],
        "intent_constraints": [{
            "constraint_id": "subject",
            "kind": "subject",
            "source_phrase": source_phrase,
            "source_occurrence": 0,
            "requirement": f"Teach {corrected}",
            "relationship_topology": "not_applicable",
        }],
        "queries": [
            {
                "text": query,
                "preserved_constraint_ids": ["subject"],
            }
            for query in queries
        ],
    })


def _expansion_payload_json(payload: dict) -> str:
    completed = dict(payload)
    completed["intent_constraints"] = [
        {
            "source_occurrence": 0,
            "relationship_topology": (
                "unspecified"
                if constraint.get("kind") == "relationship"
                else "not_applicable"
            ),
            **constraint,
        }
        for constraint in completed.get("intent_constraints", [])
    ]
    completed.setdefault("joint_structures", [])
    completed.setdefault(
        "summary_preserved_constraint_ids",
        [
            constraint["constraint_id"]
            for constraint in completed.get("intent_constraints", [])
        ],
    )
    return json.dumps(completed)


def _without_intent_contract(result: dict) -> dict:
    public = dict(result)
    intent_contract = public.pop("intent_contract", None)
    if public.get("provider_used") == "gemini":
        assert intent_contract["version"] == "expansion_intent_v2"
        assert intent_contract["request_intent"]["constraints"]
    else:
        assert intent_contract is None
    return public


def _intent_contract(exact_request: str = "physics") -> dict:
    return {
        "version": "expansion_intent_v2",
        "request_intent": {
            "exact_request": exact_request,
            "constraints": [{
                "constraint_id": "subject",
                "kind": "subject",
                "source_phrase": exact_request,
                "source_occurrence": 0,
                "requirement": f"Teach {exact_request}",
                "relationship_topology": "not_applicable",
            }],
            "joint_structures": [],
        },
    }


def _selector_contract_payload(
    exact_request: str,
    constraints: list[dict],
    *,
    joint_structures: list[dict] | None = None,
) -> dict:
    constraint_ids = [constraint["constraint_id"] for constraint in constraints]
    return {
        "corrected": exact_request,
        "intent_constraints": constraints,
        "joint_structures": list(joint_structures or []),
        "summary_preserved_constraint_ids": constraint_ids,
        "queries": [{
            "text": exact_request,
            "preserved_constraint_ids": constraint_ids,
        }],
    }


def _selector_intent_contract(
    exact_request: str,
    constraints: list[dict],
    *,
    joint_structures: list[dict] | None = None,
) -> dict:
    return {
        "version": expand.PRACTICE_FAST_INTENT_CONTRACT_VERSION,
        "request_intent": {
            "exact_request": exact_request,
            "constraints": constraints,
            "joint_structures": list(joint_structures or []),
        },
    }


def _selector_constraint(
    constraint_id: str,
    kind: str,
    source_phrase: str,
    source_occurrence: int | None = 0,
) -> dict:
    constraint = {
        "constraint_id": constraint_id,
        "kind": kind,
        "source_phrase": source_phrase,
        "requirement": f"Preserve {constraint_id}",
        "relationship_topology": (
            "unspecified" if kind == "relationship" else "not_applicable"
        ),
    }
    if source_occurrence is not None:
        constraint["source_occurrence"] = source_occurrence
    return constraint


def _repeated_c_constraints(
    subject_occurrence: int | None,
    outcome_occurrence: int,
) -> list[dict]:
    return [
        _selector_constraint("subject", "subject", "C", subject_occurrence),
        _selector_constraint("outcome", "outcome", "C", outcome_occurrence),
    ]


def _comparison_constraints(
    relation_kind: str = "relationship",
    relation_phrase: str = "Compare alpha with beta",
) -> list[dict]:
    return [
        _selector_constraint("alpha", "subject", "alpha"),
        _selector_constraint("beta", "scope", "beta"),
        _selector_constraint("compare", relation_kind, relation_phrase),
    ]


def _comparison_joint_structure() -> list[dict]:
    return [{
        "member_constraint_ids": ["alpha", "beta"],
        "relation_constraint_id": "compare",
    }]


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
    assert expand.PRACTICE_FAST_EXPAND_FALLBACK_MODEL == "gemini-3.6-flash"
    assert expand.PRACTICE_FAST_EXPAND_TIMEOUT_MS == 10_000
    assert expand.PRACTICE_FAST_EXPAND_ATTEMPTS == 3
    assert expand.PRACTICE_FAST_EXPAND_FALLBACK_OUTPUT_TOKENS == 4_096
    assert _without_intent_contract(result) == {
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

    assert expand.PRACTICE_FAST_EXPAND_CACHE_VERSION == 13
    assert expand.PRACTICE_FAST_EXPAND_OUTPUT_TOKENS == 2_048
    assert "one concise, standalone, intent-preserving learning summary" in prompt
    assert "must make sense without the original request" in prompt
    assert "must be at most 200 characters" in prompt
    assert "compact wording rather than dropping a constraint" in prompt
    assert "from corrected, produce up to n concise queries" in prompt
    corrected_schema = expand._PracticeFastExpansion.model_json_schema()["properties"][
        "corrected"
    ]
    assert corrected_schema["minLength"] == 1
    assert corrected_schema["maxLength"] == 220
    expansion_schema = expand._PracticeFastExpansion.model_json_schema()
    summary_ids_schema = expansion_schema["properties"][
        "summary_preserved_constraint_ids"
    ]
    assert "summary_preserved_constraint_ids" in expansion_schema["required"]
    assert "joint_structures" in expansion_schema["required"]
    assert "default" not in expansion_schema["properties"]["joint_structures"]
    constraint_schema = expansion_schema["$defs"][
        "_PracticeFastIntentConstraint"
    ]
    assert "source_occurrence" in constraint_schema["required"]
    assert "relationship_topology" in constraint_schema["required"]
    assert "default" not in constraint_schema["properties"]["source_occurrence"]
    assert summary_ids_schema["minItems"] == 1
    assert summary_ids_schema["maxItems"] == 16
    assert "list each id exactly once" in prompt
    assert "no omissions, duplicates, or unknown ids" in prompt
    assert "prefer focused teaching videos" in prompt
    assert "never query for a full course" in prompt
    assert "unless the user explicitly requests that format" in prompt
    assert "the first broad query must preserve every constraint" in prompt
    assert "every subject constraint plus one or more distinct" in prompt
    assert "collectively target every named facet or list member" in prompt
    assert "subject means the governing named topic, law, concept, or object" in prompt
    assert "components and named list members under that governing topic are scope" in (
        prompt
    )
    assert "net force, mass, acceleration, and units are four separate scope" in prompt


def test_expansion_joint_structure_drops_only_redundant_relation_member() -> None:
    structure = expand._PracticeFastJointStructure.model_validate({
        "member_constraint_ids": ["stage-a", "sequence", "stage-b"],
        "relation_constraint_id": "sequence",
    })

    assert structure.member_constraint_ids == ["stage-a", "stage-b"]

    with pytest.raises(ValueError, match="at least two non-relation side or stage"):
        expand._PracticeFastJointStructure.model_validate({
            "member_constraint_ids": ["stage-a", "sequence"],
            "relation_constraint_id": "sequence",
        })

    schema = expand._PracticeFastJointStructure.model_json_schema()["properties"]
    assert "Never include relation_constraint_id" in schema[
        "member_constraint_ids"
    ]["description"]
    assert "not a member_constraint_id" in schema[
        "relation_constraint_id"
    ]["description"]


def test_expansion_rejects_every_kind_topology_mismatch() -> None:
    with pytest.raises(ValueError, match="relationship topology must match"):
        expand._PracticeFastExpansion.model_validate({
            "corrected": "Teach alpha, then beta",
            "intent_constraints": [{
                "constraint_id": "sequence",
                "kind": "format",
                "source_phrase": "then",
                "source_occurrence": 0,
                "requirement": "Teach alpha before beta",
                "relationship_topology": "ordered",
            }],
            "joint_structures": [],
            "summary_preserved_constraint_ids": ["sequence"],
            "queries": [{
                "text": "alpha then beta lesson",
                "preserved_constraint_ids": ["sequence"],
            }],
        })

    with pytest.raises(ValueError, match="relationship topology must match"):
        expand._PracticeFastExpansion.model_validate({
            "corrected": "Compare alpha and beta",
            "intent_constraints": [{
                "constraint_id": "relation",
                "kind": "relationship",
                "source_phrase": "Compare",
                "source_occurrence": 0,
                "requirement": "Compare alpha and beta",
                "relationship_topology": "not_applicable",
            }],
            "joint_structures": [],
            "summary_preserved_constraint_ids": ["relation"],
            "queries": [{
                "text": "alpha beta comparison",
                "preserved_constraint_ids": ["relation"],
            }],
        })


@pytest.mark.parametrize(
    ("topic", "constraints", "joint_structures"),
    [
        (
            "Teach C, then test C.",
            _repeated_c_constraints(0, 2),
            [],
        ),
        (
            "Teach C, then test C.",
            _repeated_c_constraints(None, 1),
            [],
        ),
        (
            "Teach C, then test C.",
            _repeated_c_constraints(0, 0),
            [],
        ),
        (
            "Compare alpha with beta.",
            _comparison_constraints(relation_kind="scope"),
            _comparison_joint_structure(),
        ),
        (
            "Compare alpha with beta.",
            _comparison_constraints(relation_phrase="Compare alpha"),
            _comparison_joint_structure(),
        ),
    ],
    ids=[
        "wrong-occurrence",
        "missing-repeated-occurrence",
        "duplicate-positioned-identity",
        "invalid-relation-kind",
        "invalid-relation-topology",
    ],
)
def test_expansion_retries_selector_rejected_contract_before_cache_or_return(
    monkeypatch,
    topic,
    constraints,
    joint_structures,
) -> None:
    raw = json.dumps(_selector_contract_payload(
        topic,
        constraints,
        joint_structures=joint_structures,
    ))
    provider_calls = 0
    cache_writes = []

    def fake_raw(*_args, **_kwargs):
        nonlocal provider_calls
        provider_calls += 1
        return raw

    monkeypatch.setattr(expand, "_practice_fast_gemini_raw", fake_raw)
    monkeypatch.setattr(expand, "_read_cached_expansion", lambda *_args: None)
    monkeypatch.setattr(
        expand,
        "_write_cached_expansion",
        lambda *args: cache_writes.append(args),
    )

    with pytest.raises(ProviderResponseValidationError):
        expand.expand_query_practice_fast(
            topic,
            1,
            context=GenerationContext("fast"),
        )

    assert provider_calls == expand.PRACTICE_FAST_EXPAND_ATTEMPTS
    assert cache_writes == []


@pytest.mark.parametrize(
    ("topic", "constraints", "joint_structures"),
    [
        (
            "Teach C, then test C.",
            _repeated_c_constraints(0, 1),
            [],
        ),
        (
            "Compare alpha with beta.",
            _comparison_constraints(),
            _comparison_joint_structure(),
        ),
    ],
    ids=["repeated-source-phrase", "joint-relationship"],
)
def test_expansion_accepts_selector_trusted_contract_in_one_call(
    monkeypatch,
    topic,
    constraints,
    joint_structures,
) -> None:
    raw = json.dumps(_selector_contract_payload(
        topic,
        constraints,
        joint_structures=joint_structures,
    ))
    provider_calls = 0
    cache_writes = []

    def fake_raw(*_args, **_kwargs):
        nonlocal provider_calls
        provider_calls += 1
        return raw

    monkeypatch.setattr(expand, "_practice_fast_gemini_raw", fake_raw)
    monkeypatch.setattr(expand, "_read_cached_expansion", lambda *_args: None)
    monkeypatch.setattr(
        expand,
        "_write_cached_expansion",
        lambda *args: cache_writes.append(args),
    )

    result = expand.expand_query_practice_fast(
        topic,
        1,
        context=GenerationContext("fast"),
    )

    assert provider_calls == 1
    assert result["provider_used"] == "gemini"
    assert result["intent_contract"]["request_intent"]["constraints"] == constraints
    assert (
        result["intent_contract"]["request_intent"]["joint_structures"]
        == joint_structures
    )
    assert expand._validated_selector_intent_contract(
        result["intent_contract"]
    ) == result["intent_contract"]
    assert len(cache_writes) == 1
    assert cache_writes[0][1] == result


@pytest.mark.parametrize(
    ("topic", "constraints", "joint_structures"),
    [
        (
            "Explain Newton's second law F=ma with net force and acceleration.",
            [
                ("subject", "subject", "Newton's second law F=ma"),
                ("force", "scope", "net force"),
                ("acceleration", "scope", "acceleration"),
            ],
            [],
        ),
        (
            "Compare glycolysis with the Krebs cycle during cellular respiration.",
            [
                ("subject", "subject", "cellular respiration"),
                ("glycolysis", "scope", "glycolysis"),
                ("krebs", "scope", "the Krebs cycle"),
                (
                    "compare",
                    "relationship",
                    "Compare glycolysis with the Krebs cycle",
                ),
            ],
            [{
                "member_constraint_ids": ["glycolysis", "krebs"],
                "relation_constraint_id": "compare",
            }],
        ),
        (
            "Derive the chain rule and solve one composite-function example.",
            [
                ("subject", "subject", "chain rule"),
                ("derive", "task", "Derive"),
                ("example", "outcome", "solve one composite-function example"),
            ],
            [],
        ),
        (
            "Explain quicksort partitioning and trace one recursive example.",
            [
                ("subject", "subject", "quicksort"),
                ("partition", "scope", "partitioning"),
                ("trace", "outcome", "trace one recursive example"),
            ],
            [],
        ),
        (
            "Apply negligence duty, breach, causation, and damages to one fact pattern.",
            [
                ("subject", "subject", "negligence"),
                ("duty", "scope", "duty"),
                ("breach", "scope", "breach"),
                ("causation", "scope", "causation"),
                ("damages", "scope", "damages"),
                ("apply", "outcome", "Apply"),
            ],
            [],
        ),
    ],
    ids=["physics", "biology", "math", "software", "law"],
)
def test_expansion_preserves_atomic_cross_domain_request_contracts(
    monkeypatch,
    topic,
    constraints,
    joint_structures,
) -> None:
    constraint_ids = [constraint_id for constraint_id, _kind, _phrase in constraints]
    payload = {
        "corrected": topic.rstrip("."),
        "intent_constraints": [
            {
                "constraint_id": constraint_id,
                "kind": kind,
                "source_phrase": source_phrase,
                "source_occurrence": 0,
                "requirement": f"Preserve {source_phrase}",
                "relationship_topology": (
                    "unspecified"
                    if kind == "relationship"
                    else "not_applicable"
                ),
            }
            for constraint_id, kind, source_phrase in constraints
        ],
        "joint_structures": joint_structures,
        "summary_preserved_constraint_ids": constraint_ids,
        "queries": [{
            "text": topic.rstrip("."),
            "preserved_constraint_ids": constraint_ids,
        }],
    }
    calls = 0

    def fake_raw(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return json.dumps(payload)

    monkeypatch.setattr(expand, "_practice_fast_gemini_raw", fake_raw)

    result = expand.expand_query_practice_fast(topic, 1)

    contract = result["intent_contract"]
    assert calls == 1
    assert contract["version"] == "expansion_intent_v2"
    assert contract["request_intent"]["exact_request"] == topic
    assert [
        item["constraint_id"]
        for item in contract["request_intent"]["constraints"]
    ] == constraint_ids
    assert contract["request_intent"]["joint_structures"] == joint_structures


@pytest.mark.parametrize(
    ("topic", "corrected", "constraints", "queries"),
    [
        (
            (
                "Teach cellular respiration for a beginner in a clear progression: "
                "glycolysis, the Krebs cycle, oxidative phosphorylation, and ATP "
                "yield, with each stage connected to the next."
            ),
            (
                "Beginner cellular respiration progression connecting glycolysis, "
                "the Krebs cycle, oxidative phosphorylation, and ATP yield."
            ),
            [
                ("c1", "subject", "cellular respiration"),
                ("c2", "task", "Teach"),
                ("c3", "scope", "glycolysis"),
                ("c4", "scope", "the Krebs cycle"),
                ("c5", "scope", "oxidative phosphorylation"),
                ("c6", "scope", "ATP yield"),
                ("c7", "relationship", "each stage connected to the next"),
                ("c8", "format", "clear progression"),
            ],
            [
                (
                    "cellular respiration steps explained glycolysis krebs cycle "
                    "oxidative phosphorylation connection",
                    ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"],
                ),
                (
                    "glycolysis and krebs cycle explained for beginners",
                    ["c1", "c2", "c3", "c4", "c7"],
                ),
                (
                    "oxidative phosphorylation and ATP yield explained",
                    ["c1", "c2", "c5", "c6"],
                ),
            ],
        ),
        (
            (
                "Teach derivatives for a beginner in a smooth progression: start "
                "with slope and rate-of-change intuition, derive the power rule, "
                "then solve one worked derivative problem and interpret the answer."
            ),
            (
                "Beginner derivatives progression from slope and rate-of-change "
                "intuition through the power rule, one worked problem, and "
                "interpretation."
            ),
            [
                ("c1", "subject", "derivatives"),
                ("c2", "task", "Teach"),
                ("c3", "scope", "beginner"),
                ("c4", "scope", "slope and rate-of-change intuition"),
                ("c5", "scope", "derive the power rule"),
                (
                    "c6",
                    "outcome",
                    "solve one worked derivative problem and interpret the answer",
                ),
            ],
            [
                (
                    "calculus derivatives for beginners slope rate of change power "
                    "rule derivation and examples",
                    ["c1", "c2", "c3", "c4", "c5", "c6"],
                ),
                (
                    "derivatives intuitive understanding slope and rate of change",
                    ["c1", "c2", "c3", "c4"],
                ),
                (
                    "derivative power rule proof and worked example interpretation",
                    ["c1", "c5", "c6"],
                ),
            ],
        ),
        (
            (
                "Teach nullability notation for a beginner software developer: "
                "compare Swift String? with Swift String, then explain TypeScript's "
                "non-null assertion !, with safe-use examples and common mistakes."
            ),
            (
                "Beginner software nullability: compare Swift String? with String, "
                "explain TypeScript's non-null assertion !, and cover safe use and "
                "common mistakes."
            ),
            [
                ("c1", "subject", "nullability notation"),
                ("c2", "scope", "beginner software developer"),
                ("c3", "task", "compare Swift String? with Swift String"),
                ("c4", "task", "explain TypeScript's non-null assertion !"),
                ("c5", "outcome", "safe-use examples"),
                ("c6", "outcome", "common mistakes"),
            ],
            [
                (
                    "nullability in programming for beginners Swift optionals and "
                    "TypeScript non-null assertion explained",
                    ["c1", "c2", "c3", "c4"],
                ),
                (
                    "Swift String versus String? optional explanation with safe "
                    "handling examples",
                    ["c1", "c3", "c5"],
                ),
                (
                    "TypeScript non-null assertion operator ! common mistakes and "
                    "safe usage",
                    ["c1", "c4", "c6"],
                ),
            ],
        ),
    ],
    ids=["cellular-respiration", "derivatives", "nullability"],
)
def test_live_cross_domain_expansions_reach_supadata_as_short_ai_branches(
    monkeypatch,
    topic,
    corrected,
    constraints,
    queries,
):
    payload = {
        "corrected": corrected,
        "intent_constraints": [
            {
                "constraint_id": constraint_id,
                "kind": kind,
                "source_phrase": source_phrase,
                "requirement": f"Preserve {source_phrase}",
            }
            for constraint_id, kind, source_phrase in constraints
        ],
        "queries": [
            {
                "text": query,
                "preserved_constraint_ids": preserved_constraint_ids,
            }
            for query, preserved_constraint_ids in queries
        ],
    }
    gemini_calls = 0
    search_calls = []

    def fake_raw(*_args, **_kwargs):
        nonlocal gemini_calls
        gemini_calls += 1
        return _expansion_payload_json(payload)

    def fake_search_all(search_queries, filters=None, **_kwargs):
        del filters
        search_calls.append(list(search_queries))
        return {
            "per_query": [
                {"query": query, "videos": [], "next_page_token": None}
                for query in search_queries
            ],
            "credits_used": len(search_queries),
            "warning": None,
        }

    monkeypatch.setattr(expand, "_practice_fast_gemini_raw", fake_raw)
    monkeypatch.setattr(search.supadata_search, "search_all", fake_search_all)

    result = search.discover_practice_fast(
        topic,
        limit=3,
        breadth=8,
        retrieval_profile="deep",
    )

    expected_queries = [query for query, _preserved in queries]
    assert gemini_calls == 1
    assert search_calls == [expected_queries]
    assert result["queries"] == expected_queries
    assert result["corrected"] == corrected
    assert corrected != topic
    assert result["topic_terms"] == [topic]
    assert result["provider_used"] == "gemini"
    assert topic not in expected_queries


def test_practice_fast_expansion_keeps_broad_query_then_subject_grounded_focus(
    monkeypatch,
):
    payload = {
        "corrected": "chain rule worked example",
        "intent_constraints": [
            {
                "constraint_id": "subject",
                "kind": "subject",
                "source_phrase": "chain rule",
                "requirement": "Teach the chain rule",
            },
            {
                "constraint_id": "task",
                "kind": "task",
                "source_phrase": "worked",
                "requirement": "Work through the process",
            },
            {
                "constraint_id": "outcome",
                "kind": "outcome",
                "source_phrase": "example",
                "requirement": "Reach a concrete example answer",
            },
        ],
        "queries": [
            {
                "text": "worked derivative walkthrough",
                "preserved_constraint_ids": ["task"],
            },
            {
                "text": "chain rule worked example",
                "preserved_constraint_ids": ["subject", "task", "outcome"],
            },
            {
                "text": "chain rule solved derivative walkthrough",
                "preserved_constraint_ids": ["subject", "task", "outcome"],
            },
        ],
    }
    monkeypatch.setattr(
        expand,
        "_practice_fast_gemini_raw",
        lambda *_args, **_kwargs: _expansion_payload_json(payload),
    )

    result = expand.expand_query_practice_fast("chain rule worked example", 3)

    assert _without_intent_contract(result) == {
        "corrected": "chain rule worked example",
        "queries": [
            "chain rule worked example",
            "chain rule solved derivative walkthrough",
        ],
        "provider_used": "gemini",
    }


def test_practice_fast_expansion_collectively_targets_named_request_facets(monkeypatch):
    topic = (
        "Explain Newton's second law F=ma from intuition through worked examples, "
        "including net force, mass, acceleration, units, and solving for each variable"
    )
    constraints = [
        ("task", "task", "Explain"),
        ("subject", "subject", "Newton's second law F=ma"),
        ("intuition", "format", "from intuition through"),
        ("worked_examples", "format", "through worked examples, including"),
        ("net_force", "scope", "net force"),
        ("mass", "scope", "mass"),
        ("acceleration", "scope", "acceleration"),
        ("units", "scope", "units"),
        ("solve_each", "outcome", "solving for each variable"),
    ]
    all_ids = [constraint_id for constraint_id, _kind, _phrase in constraints]
    payload = {
        "corrected": topic,
        "intent_constraints": [
            {
                "constraint_id": constraint_id,
                "kind": kind,
                "source_phrase": source_phrase,
                "requirement": f"Preserve {source_phrase}",
            }
            for constraint_id, kind, source_phrase in constraints
        ],
        "queries": [
            {
                "text": "Newton second law net force and mass worked explanation",
                "preserved_constraint_ids": [
                    "subject", "task", "intuition", "net_force", "mass",
                ],
            },
            {
                "text": (
                    "Newton second law F=ma complete worked examples for every variable"
                ),
                "preserved_constraint_ids": all_ids,
            },
            {
                "text": "Newton second law acceleration units solve each variable",
                "preserved_constraint_ids": [
                    "subject",
                    "worked_examples",
                    "acceleration",
                    "units",
                    "solve_each",
                ],
            },
        ],
    }
    monkeypatch.setattr(
        expand,
        "_practice_fast_gemini_raw",
        lambda *_args, **_kwargs: _expansion_payload_json(payload),
    )

    result = expand.expand_query_practice_fast(topic, 3)

    assert _without_intent_contract(result) == {
        "corrected": topic,
        "queries": [
            "Newton second law F=ma complete worked examples for every variable",
            "Newton second law net force and mass worked explanation",
            "Newton second law acceleration units solve each variable",
        ],
        "provider_used": "gemini",
    }


def test_practice_fast_expansion_uses_focused_queries_without_synthesizing_long_literal(
    monkeypatch,
):
    topic = (
        "Explain Newton's second law F=ma from intuition through worked examples, "
        "including net force, mass, acceleration, units, and solving for each variable"
    )
    payload = {
        "corrected": topic,
        "intent_constraints": [
            {
                "constraint_id": "subject",
                "kind": "subject",
                "source_phrase": "Newton's second law F=ma",
                "requirement": "Teach the governing law",
            },
            {
                "constraint_id": "task",
                "kind": "task",
                "source_phrase": "Explain",
                "requirement": "Explain clearly",
            },
            {
                "constraint_id": "intuition",
                "kind": "scope",
                "source_phrase": "intuition",
                "requirement": "Cover intuition",
            },
            {
                "constraint_id": "components",
                "kind": "scope",
                "source_phrase": "net force, mass, acceleration, units",
                "requirement": "Cover the named components",
            },
            {
                "constraint_id": "worked",
                "kind": "outcome",
                    "source_phrase": "worked examples",
                "requirement": "Solve each variable in worked examples",
            },
        ],
        "queries": [
            {
                "text": "Newton second law intuitive explanation and units",
                "preserved_constraint_ids": [
                    "subject", "task", "intuition", "components",
                ],
            },
            {
                "text": "Newton second law worked examples solve F m and a",
                "preserved_constraint_ids": ["subject", "worked"],
            },
            {
                "text": "Newton second law net force mass acceleration practice",
                "preserved_constraint_ids": ["subject", "components", "worked"],
            },
        ],
    }
    monkeypatch.setattr(
        expand,
        "_practice_fast_gemini_raw",
        lambda *_args, **_kwargs: _expansion_payload_json(payload),
    )

    result = expand.expand_query_practice_fast(topic, 3)

    assert _without_intent_contract(result) == {
        "corrected": topic,
        "queries": [
            "Newton second law intuitive explanation and units",
            "Newton second law worked examples solve F m and a",
            "Newton second law net force mass acceleration practice",
        ],
        "provider_used": "gemini",
    }


def test_practice_fast_expansion_uses_all_slots_for_complete_focused_cover(
    monkeypatch,
):
    topic = (
        "Newton's laws: begin with first-law inertia and balanced forces, then "
        "net force and F=ma, then free-body diagrams, then third-law "
        "action-reaction pairs, and finish with worked problems and common "
        "misconceptions."
    )
    constraints = [
        ("subject", "subject", "Newton's laws"),
        ("first_law", "scope", "first-law inertia and balanced forces"),
        ("second_law", "scope", "net force and F=ma"),
        ("free_body", "scope", "free-body diagrams"),
        ("third_law", "scope", "third-law action-reaction pairs"),
            ("sequence", "format", "begin with"),
        (
            "practice",
            "outcome",
            "worked problems and common misconceptions",
        ),
    ]
    payload = {
        "corrected": topic,
        "intent_constraints": [
            {
                "constraint_id": constraint_id,
                "kind": kind,
                "source_phrase": source_phrase,
                "requirement": f"Preserve {source_phrase}",
            }
            for constraint_id, kind, source_phrase in constraints
        ],
        "queries": [
            {
                "text": "Newton laws explained inertia F=ma action-reaction",
                "preserved_constraint_ids": [
                    "subject",
                    "first_law",
                    "second_law",
                    "third_law",
                    "sequence",
                ],
            },
            {
                "text": "Newton laws free-body diagram lesson",
                "preserved_constraint_ids": ["subject", "free_body"],
            },
            {
                "text": "Newton laws practice problems common misconceptions",
                "preserved_constraint_ids": ["subject", "practice"],
            },
        ],
    }
    monkeypatch.setattr(
        expand,
        "_practice_fast_gemini_raw",
        lambda *_args, **_kwargs: _expansion_payload_json(payload),
    )

    result = expand.expand_query_practice_fast(topic, 3)

    assert _without_intent_contract(result) == {
        "corrected": topic,
        "queries": [query["text"] for query in payload["queries"]],
        "provider_used": "gemini",
    }


def test_practice_fast_expansion_keeps_grounded_branches_when_slots_miss_one_facet(
    monkeypatch,
):
    topic = "physics force mass acceleration units"
    all_ids = ["subject", "force", "mass", "acceleration", "units"]
    payload = {
        "corrected": topic,
        "intent_constraints": [
            {
                "constraint_id": constraint_id,
                "kind": "subject" if constraint_id == "subject" else "scope",
                "source_phrase": (
                    "physics" if constraint_id == "subject" else constraint_id
                ),
                "requirement": f"Preserve {constraint_id}",
            }
            for constraint_id in all_ids
        ],
        "queries": [
            {
                "text": "physics force lesson",
                "preserved_constraint_ids": ["subject", "force"],
            },
            {
                "text": "physics mass lesson",
                "preserved_constraint_ids": ["subject", "mass"],
            },
            {
                "text": "physics acceleration lesson",
                "preserved_constraint_ids": ["subject", "acceleration"],
            },
        ],
    }
    monkeypatch.setattr(
        expand,
        "_practice_fast_gemini_raw",
        lambda *_args, **_kwargs: _expansion_payload_json(payload),
    )

    result = expand.expand_query_practice_fast(topic, 3)

    assert _without_intent_contract(result) == {
        "corrected": topic,
        "queries": [
            "physics force lesson",
            "physics mass lesson",
            "physics acceleration lesson",
        ],
        "provider_used": "gemini",
    }


def test_practice_fast_expansion_prioritizes_distinct_facet_coverage(monkeypatch):
    topic = "physics force mass acceleration units"
    constraints = [
        ("subject", "subject", "physics"),
        ("force", "scope", "force"),
        ("mass", "scope", "mass"),
        ("acceleration", "scope", "acceleration"),
        ("units", "scope", "units"),
    ]
    all_ids = [constraint_id for constraint_id, _kind, _phrase in constraints]
    payload = {
        "corrected": topic,
        "intent_constraints": [
            {
                "constraint_id": constraint_id,
                "kind": kind,
                "source_phrase": source_phrase,
                "requirement": f"Preserve {source_phrase}",
            }
            for constraint_id, kind, source_phrase in constraints
        ],
        "queries": [
            {
                "text": "physics force mass acceleration units overview",
                "preserved_constraint_ids": all_ids,
            },
            {
                "text": "physics force and mass",
                "preserved_constraint_ids": ["subject", "force", "mass"],
            },
            {
                "text": "physics force and acceleration",
                "preserved_constraint_ids": ["subject", "force", "acceleration"],
            },
            {
                "text": "physics acceleration and units",
                "preserved_constraint_ids": ["subject", "acceleration", "units"],
            },
        ],
    }
    monkeypatch.setattr(
        expand,
        "_practice_fast_gemini_raw",
        lambda *_args, **_kwargs: _expansion_payload_json(payload),
    )

    result = expand.expand_query_practice_fast(topic, 3)

    assert result["queries"] == [
        "physics force mass acceleration units overview",
        "physics force and mass",
        "physics acceleration and units",
    ]


def test_practice_fast_expansion_finds_non_greedy_complete_facet_cover(monkeypatch):
    topic = "physics force mass acceleration units energy momentum"
    all_ids = [
        "subject",
        "force",
        "mass",
        "acceleration",
        "units",
        "energy",
        "momentum",
    ]
    payload = {
        "corrected": topic,
        "intent_constraints": [
            {
                "constraint_id": constraint_id,
                "kind": "subject" if constraint_id == "subject" else "scope",
                "source_phrase": (
                    "physics" if constraint_id == "subject" else constraint_id
                ),
                "requirement": f"Preserve {constraint_id}",
            }
            for constraint_id in all_ids
        ],
        "queries": [
            {
                "text": "physics force mass acceleration and units",
                "preserved_constraint_ids": [
                    "subject",
                    "force",
                    "mass",
                    "acceleration",
                    "units",
                ],
            },
            {
                "text": "physics force mass and energy",
                "preserved_constraint_ids": [
                    "subject",
                    "force",
                    "mass",
                    "energy",
                ],
            },
            {
                "text": "physics acceleration units and momentum",
                "preserved_constraint_ids": [
                    "subject",
                    "acceleration",
                    "units",
                    "momentum",
                ],
            },
        ],
    }
    monkeypatch.setattr(
        expand,
        "_practice_fast_gemini_raw",
        lambda *_args, **_kwargs: _expansion_payload_json(payload),
    )

    result = expand.expand_query_practice_fast(topic, 3)

    assert _without_intent_contract(result) == {
        "corrected": topic,
        "queries": [
            "physics force mass acceleration and units",
            "physics force mass and energy",
            "physics acceleration units and momentum",
        ],
        "provider_used": "gemini",
    }


def test_practice_fast_expansion_keeps_partial_focus_beside_complete_broad_query(
    monkeypatch,
):
    topic = "physics force mass acceleration units"
    all_ids = ["subject", "force", "mass", "acceleration", "units"]
    payload = {
        "corrected": topic,
        "intent_constraints": [
            {
                "constraint_id": constraint_id,
                "kind": "subject" if constraint_id == "subject" else "scope",
                "source_phrase": (
                    "physics" if constraint_id == "subject" else constraint_id
                ),
                "requirement": f"Preserve {constraint_id}",
            }
            for constraint_id in all_ids
        ],
        "queries": [
            {
                "text": "physics force mass acceleration units overview",
                "preserved_constraint_ids": all_ids,
            },
            {
                "text": "physics force lesson",
                "preserved_constraint_ids": ["subject", "force"],
            },
        ],
    }
    monkeypatch.setattr(
        expand,
        "_practice_fast_gemini_raw",
        lambda *_args, **_kwargs: _expansion_payload_json(payload),
    )

    result = expand.expand_query_practice_fast(topic, 3)

    assert _without_intent_contract(result) == {
        "corrected": topic,
        "queries": [
            "physics force mass acceleration units overview",
            "physics force lesson",
        ],
        "provider_used": "gemini",
    }


@pytest.mark.parametrize(
    "intent_constraints, queries, expected",
    [
        (
            [{
                "constraint_id": "request",
                "kind": "task",
                "source_phrase": "chain rule worked example",
                "requirement": "Teach a chain rule worked example",
            }],
            [{
                "text": "chain rule worked example",
                "preserved_constraint_ids": ["request"],
            }],
            None,
        ),
        (
            [
                {
                    "constraint_id": "subject",
                    "kind": "subject",
                    "source_phrase": "chain rule",
                    "requirement": "Teach the chain rule",
                },
                {
                    "constraint_id": "task",
                    "kind": "task",
                    "source_phrase": "worked",
                    "requirement": "Work through the process",
                },
                {
                    "constraint_id": "outcome",
                    "kind": "outcome",
                    "source_phrase": "example",
                    "requirement": "Reach a concrete example answer",
                },
            ],
            [
                {
                    "text": "chain rule worked walkthrough",
                    "preserved_constraint_ids": ["subject", "task"],
                },
                {
                    "text": "chain rule example answer",
                    "preserved_constraint_ids": ["subject", "outcome"],
                },
            ],
            {
                "corrected": "chain rule worked example",
                "queries": [
                    "chain rule worked walkthrough",
                    "chain rule example answer",
                ],
                "provider_used": "gemini",
            },
        ),
    ],
)
def test_practice_fast_expansion_requires_subject_and_accepts_focused_queries(
    monkeypatch,
    intent_constraints,
    queries,
    expected,
):
    payload = {
        "corrected": "chain rule worked example",
        "intent_constraints": intent_constraints,
        "queries": queries,
    }
    monkeypatch.setattr(
        expand,
        "_practice_fast_gemini_raw",
        lambda *_args, **_kwargs: _expansion_payload_json(payload),
    )

    if expected is None:
        with pytest.raises(ProviderResponseValidationError):
            expand.expand_query_practice_fast("chain rule worked example", 3)
    else:
        result = expand.expand_query_practice_fast("chain rule worked example", 3)
        assert _without_intent_contract(result) == expected


def test_practice_fast_expansion_trusts_grounded_ai_retrieval_branch_when_plan_is_sparse(
    monkeypatch,
):
    payload = {
        "corrected": "chain rule worked example",
        "intent_constraints": [
            {
                "constraint_id": "subject",
                "kind": "subject",
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
        lambda *_args, **_kwargs: _expansion_payload_json(payload),
    )

    result = expand.expand_query_practice_fast("chain rule worked example", 3)

    assert _without_intent_contract(result) == {
        "corrected": "chain rule worked example",
        "queries": ["chain rule definition lecture"],
        "provider_used": "gemini",
    }


def test_practice_fast_expansion_rejects_ungrounded_ai_subject(monkeypatch):
    payload = {
        "corrected": "quotient rule",
        "intent_constraints": [
            {
                "constraint_id": "subject",
                "kind": "subject",
                "source_phrase": "quotient rule",
                "requirement": "Teach the quotient rule",
            },
        ],
        "queries": [
            {
                "text": "quotient rule lecture",
                "preserved_constraint_ids": ["subject"],
            },
        ],
    }
    monkeypatch.setattr(
        expand,
        "_practice_fast_gemini_raw",
        lambda *_args, **_kwargs: _expansion_payload_json(payload),
    )

    with pytest.raises(ProviderResponseValidationError):
        expand.expand_query_practice_fast("chain rule worked example", 3)


@pytest.mark.parametrize(
    "summary_ids",
    [
        ["subject", "task"],
        ["subject", "task", "outcome", "unknown"],
        ["subject", "task", "outcome", "outcome"],
    ],
    ids=["incomplete", "unknown", "duplicate"],
)
def test_practice_fast_expansion_rejects_unbound_corrected_summary(
    monkeypatch,
    summary_ids,
):
    payload = {
        "corrected": "quotient rule worked example",
        "summary_preserved_constraint_ids": summary_ids,
        "intent_constraints": [
            {
                "constraint_id": "subject",
                "kind": "subject",
                "source_phrase": "chain rule",
                "requirement": "Teach the chain rule",
            },
            {
                "constraint_id": "task",
                "kind": "task",
                "source_phrase": "worked",
                "requirement": "Work through the process",
            },
            {
                "constraint_id": "outcome",
                "kind": "outcome",
                "source_phrase": "example",
                "requirement": "Reach a concrete example answer",
            },
        ],
        "queries": [
            {
                "text": "chain rule worked example",
                "preserved_constraint_ids": ["subject", "task", "outcome"],
            },
        ],
    }
    monkeypatch.setattr(
        expand,
        "_practice_fast_gemini_raw",
        lambda *_args, **_kwargs: json.dumps(payload),
    )

    with pytest.raises(ProviderResponseValidationError):
        expand.expand_query_practice_fast("chain rule worked example", 3)


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


def test_failed_expansion_dispatch_records_typed_provider_status(monkeypatch):
    from google import genai

    class _ClientError(RuntimeError):
        code = 400

    async def fail(**_kwargs):
        raise _ClientError("invalid structured-output schema")

    monkeypatch.setattr(expand.config, "require_gemini_key", lambda: "gemini-test")
    monkeypatch.setattr(genai, "Client", lambda **_kwargs: _FakeGeminiClient(fail))
    context = GenerationContext("fast")

    with pytest.raises(_ClientError):
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
    assert usage["status_code"] == 400
    assert usage["error_code"] == "dispatch_failed:_ClientError"


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
    assert config.max_output_tokens == 2_048
    schema = config.response_json_schema
    serialized_schema = json.dumps(schema)
    assert "additionalProperties" not in serialized_schema
    assert "minLength" not in serialized_schema
    assert "maxLength" not in serialized_schema
    assert set(schema["properties"]) == {
        "corrected",
        "intent_constraints",
        "joint_structures",
        "summary_preserved_constraint_ids",
        "queries",
    }
    assert set(schema["required"]) == set(schema["properties"])


def test_fallback_expansion_gets_failure_only_output_headroom(monkeypatch):
    from google import genai

    seen = {}

    class _Response:
        text = '{"corrected":"Physics","queries":["Physics"]}'
        model_version = expand.PRACTICE_FAST_EXPAND_FALLBACK_MODEL
        usage_metadata = {}

    async def succeed(**kwargs):
        seen["config"] = kwargs["config"]
        return _Response()

    monkeypatch.setattr(expand.config, "require_gemini_key", lambda: "gemini-test")
    monkeypatch.setattr(genai, "Client", lambda **_kwargs: _FakeGeminiClient(succeed))

    asyncio.run(
        expand._practice_fast_gemini_raw_async(
            "physics",
            1,
            model=expand.PRACTICE_FAST_EXPAND_FALLBACK_MODEL,
            level=None,
            should_cancel=None,
        )
    )

    assert seen["config"].max_output_tokens == 4_096


def test_zero_result_recovery_prompt_includes_exact_request_tried_queries_and_signal(
    monkeypatch,
):
    from google import genai

    seen: dict[str, object] = {}

    class _Response:
        text = '{"corrected":"Newton laws","queries":[]}'
        model_version = "gemini-flash-test"
        usage_metadata = {
            "prompt_token_count": 10,
            "candidates_token_count": 5,
            "total_token_count": 15,
        }

    async def succeed(**kwargs):
        seen.update(kwargs)
        return _Response()

    exact_request = "Newton's laws with free-body diagram problems"
    tried_queries = ["Newton laws tutorial", "Newton laws explained"]
    monkeypatch.setattr(expand.config, "require_gemini_key", lambda: "gemini-test")
    monkeypatch.setattr(genai, "Client", lambda **_kwargs: _FakeGeminiClient(succeed))

    asyncio.run(
        expand._practice_fast_gemini_raw_async(
            exact_request,
            3,
            model=expand.PRACTICE_FAST_EXPAND_MODEL,
            level=None,
            should_cancel=None,
            tried_queries=tried_queries,
            recovery_reason=expand.RECOVERY_REASON_ZERO_SEARCH_RESULTS,
        )
    )

    prompt = str(seen["contents"])
    assert exact_request in prompt
    assert "ZERO_SEARCH_RESULT_RECOVERY" in prompt
    assert "zero eligible YouTube videos" in prompt
    assert json.dumps(tried_queries, ensure_ascii=True) in prompt


def test_zero_clip_recovery_has_distinct_signal_rejected_ids_and_cache_key(
    monkeypatch,
) -> None:
    from google import genai

    prompts: list[str] = []

    class _Response:
        text = '{"corrected":"Circuits","queries":[]}'
        model_version = "gemini-flash-test"
        usage_metadata = {
            "prompt_token_count": 10,
            "candidates_token_count": 5,
            "total_token_count": 15,
        }

    async def succeed(**kwargs):
        prompts.append(str(kwargs["contents"]))
        return _Response()

    topic = "Explain current and voltage in series circuits"
    tried_queries = ["series circuit current voltage tutorial"]
    rejected_ids = ["video-rejected-b", "video-rejected-a"]
    monkeypatch.setattr(expand.config, "require_gemini_key", lambda: "gemini-test")
    monkeypatch.setattr(genai, "Client", lambda **_kwargs: _FakeGeminiClient(succeed))

    for reason, ids in (
        (expand.RECOVERY_REASON_ZERO_SEARCH_RESULTS, []),
        (expand.RECOVERY_REASON_ZERO_VALID_CLIPS, rejected_ids),
    ):
        asyncio.run(
            expand._practice_fast_gemini_raw_async(
                topic,
                2,
                model=expand.PRACTICE_FAST_EXPAND_MODEL,
                level=None,
                should_cancel=None,
                tried_queries=tried_queries,
                recovery_reason=reason,
                rejected_video_ids=ids,
            )
        )

    assert "ZERO_SEARCH_RESULT_RECOVERY" in prompts[0]
    assert "ZERO_VALID_CLIP_RECOVERY" not in prompts[0]
    assert "ZERO_VALID_CLIP_RECOVERY" in prompts[1]
    assert "transcript retrieval, selector/audit validation, or boundary" in prompts[1]
    assert json.dumps(sorted(rejected_ids), ensure_ascii=True) in prompts[1]
    assert prompts[0] != prompts[1]
    search_key = expand._expansion_cache_key(
        topic,
        2,
        None,
        tried_queries=tried_queries,
        recovery_reason=expand.RECOVERY_REASON_ZERO_SEARCH_RESULTS,
    )
    clip_key = expand._expansion_cache_key(
        topic,
        2,
        None,
        tried_queries=tried_queries,
        recovery_reason=expand.RECOVERY_REASON_ZERO_VALID_CLIPS,
        rejected_video_ids=rejected_ids,
    )
    assert search_key != clip_key


def test_practice_fast_expansion_uses_configured_flash_on_final_attempt(
    monkeypatch,
):
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
    monkeypatch.setattr(
        expand.config,
        "GEMINI_MODEL",
        "gemini-3.1-pro-preview",
    )

    result = expand.expand_query_practice_fast("physics", 3)

    assert calls == [
        "gemini-3.1-flash-lite",
        "gemini-3.1-flash-lite",
        expand.PRACTICE_FAST_EXPAND_FALLBACK_MODEL,
    ]
    assert _without_intent_contract(result) == {
        "corrected": "Physics",
        "queries": ["Physics", "mechanics", "waves"],
        "provider_used": "gemini",
    }


def test_practice_fast_expansion_raises_typed_failure_after_all_models(
    monkeypatch,
):
    calls = []

    def failed_models(*args, model, **kwargs):
        calls.append(model)
        raise RuntimeError("unavailable")

    monkeypatch.setattr(expand, "_practice_fast_gemini_raw", failed_models)
    monkeypatch.setattr(expand.config, "GEMINI_MODEL", "gemini-3.1-pro-preview")

    with pytest.raises(ProviderTransientError):
        expand.expand_query_practice_fast("physics", 3)

    assert calls == [
        "gemini-3.1-flash-lite",
        "gemini-3.1-flash-lite",
        expand.PRACTICE_FAST_EXPAND_FALLBACK_MODEL,
    ]


def test_practice_fast_expansion_retries_failed_flash_step_once(monkeypatch):
    calls = 0

    def fail_then_succeed(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary model failure")
        return _intent_expansion_json(
            corrected="Physics",
            source_phrase="physics",
            queries=["Physics", "mechanics", "waves"],
        )

    monkeypatch.setattr(expand, "_practice_fast_gemini_raw", fail_then_succeed)

    result = expand.expand_query_practice_fast("physics", 3)

    assert calls == 2
    assert _without_intent_contract(result) == {
        "corrected": "Physics",
        "queries": ["Physics", "mechanics", "waves"],
        "provider_used": "gemini",
    }


def test_practice_fast_expansion_does_not_retry_permanent_configuration(
    monkeypatch,
):
    calls = 0

    def fail_configuration(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise ProviderConfigurationError(
            "GEMINI_API_KEY is not set.",
            provider="gemini",
            operation="expansion",
        )

    monkeypatch.setattr(expand, "_practice_fast_gemini_raw", fail_configuration)

    with pytest.raises(ProviderConfigurationError):
        expand.expand_query_practice_fast("physics", 3)

    assert calls == 1


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
            "intent_contract": _intent_contract(),
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


@pytest.mark.parametrize(
    "intent_contract",
    [
        None,
        {"version": "expansion_intent_v0", "request_intent": {}},
        {
            "version": "expansion_intent_v1",
            "request_intent": {
                "exact_request": "physics",
                "constraints": [],
                "joint_structures": [],
            },
        },
        _selector_intent_contract(
            "Teach C, then test C.",
            _repeated_c_constraints(0, 0),
        ),
        _selector_intent_contract(
            "Compare alpha with beta.",
            _comparison_constraints(relation_phrase="Compare alpha"),
            joint_structures=_comparison_joint_structure(),
        ),
    ],
)
def test_current_expansion_cache_rejects_missing_or_malformed_intent_contract(
    monkeypatch,
    intent_contract,
) -> None:
    row = {
        "response_json": json.dumps({
            "version": expand.PRACTICE_FAST_EXPAND_CACHE_VERSION,
            "corrected": "Physics",
            "queries": ["physics lecture"],
            "intent_contract": intent_contract,
        }),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    monkeypatch.setattr(
        expand,
        "get_conn",
        lambda *args, **kwargs: nullcontext(object()),
    )
    monkeypatch.setattr(expand, "fetch_one", lambda *_args, **_kwargs: row)

    assert expand._read_cached_expansion("cache-key", 3) is None


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
    intent_contract = _intent_contract(topic)

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
            "intent_contract": intent_contract,
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
    assert result["intent_contract"] == intent_contract


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


@pytest.mark.parametrize(
    ("retrieval_profile", "expected_query"),
    [
        ("deep", "gravitational force components on an incline explained"),
        ("bootstrap", "gravitational force components on an incline for beginners"),
    ],
)
def test_practice_fast_searches_narrow_focus_without_losing_literal_context(
    monkeypatch,
    retrieval_profile,
    expected_query,
):
    narrow = "gravitational force components on an incline"
    broad = (
        "Learn Newton's second law from concept to worked free-body diagram "
        "problems, including friction on an incline."
    )
    expanded_topics = []
    searched_queries = []

    def fake_expand(topic, *_args, **_kwargs):
        expanded_topics.append(topic)
        return {
            "corrected": narrow,
            "queries": [f"{topic} explained"],
            "provider_used": "gemini",
        }

    def fake_search_all(queries, filters=None, **_kwargs):
        searched_queries.extend(queries)
        return {
            "per_query": [
                {
                    "query": query,
                    "videos": [{"id": "focused-video", "viewCount": 100}],
                }
                for query in queries
            ],
            "credits_used": len(queries),
            "warning": None,
        }

    monkeypatch.setattr(search.expand, "expand_query_practice_fast", fake_expand)
    monkeypatch.setattr(search.supadata_search, "search_all", fake_search_all)

    result = search.discover_practice_fast(
        narrow,
        limit=1,
        breadth=1,
        level="beginner",
        literal_topic=broad,
        retrieval_profile=retrieval_profile,
    )

    assert expanded_topics == ([narrow] if retrieval_profile == "deep" else [])
    assert searched_queries == [expected_query]
    assert result["topic_terms"] == [narrow]
    assert [video["id"] for video in result["videos"]] == ["focused-video"]


def test_consumed_first_page_fetches_next_provider_page_before_exhaustion(monkeypatch):
    calls = []
    monkeypatch.setattr(
        search.expand,
        "expand_query_practice_fast",
        lambda topic, *_args, **_kwargs: {
            "corrected": topic,
            "queries": [topic],
            "provider_used": "literal_fallback",
        },
    )

    def fake_search_all(queries, filters=None, **kwargs):
        calls.append(kwargs.get("page_tokens"))
        if kwargs.get("page_tokens") is None:
            return {
                "per_query": [{
                    "query": queries[0],
                    "videos": [{"id": "already-consumed"}],
                    "next_page_token": "page-2",
                }],
                "credits_used": 0,
                "warning": None,
            }
        return {
            "per_query": [{
                "query": queries[0],
                "videos": [{"id": "fresh-page-2"}],
                "next_page_token": None,
            }],
            "credits_used": 1,
            "warning": None,
        }

    monkeypatch.setattr(search.supadata_search, "search_all", fake_search_all)

    result = search.discover_practice_fast(
        "AP Statistics sampling distributions",
        limit=1,
        breadth=1,
        consumed_video_ids=["already-consumed"],
        retrieval_profile="deep",
    )

    assert calls == [None, ["page-2"]]
    assert [video["id"] for video in result["videos"]] == ["fresh-page-2"]
    assert result["provider_exhausted"] is True


def test_provider_cursor_remains_open_when_current_page_fills_batch(monkeypatch):
    monkeypatch.setattr(
        search.expand,
        "expand_query_practice_fast",
        lambda topic, *_args, **_kwargs: {
            "corrected": topic,
            "queries": [topic],
            "provider_used": "literal_fallback",
        },
    )
    monkeypatch.setattr(
        search.supadata_search,
        "search_all",
        lambda queries, filters=None, **kwargs: {
            "per_query": [{
                "query": queries[0],
                "videos": [{"id": "current-page"}],
                "next_page_token": "page-2",
            }],
            "credits_used": 0,
            "warning": None,
        },
    )

    result = search.discover_practice_fast(
        "AP Statistics confidence intervals",
        limit=1,
        breadth=1,
        retrieval_profile="deep",
    )

    assert [video["id"] for video in result["videos"]] == ["current-page"]
    assert result["provider_exhausted"] is False


def test_temporarily_empty_provider_page_walks_to_later_fresh_inventory(monkeypatch):
    calls = []
    monkeypatch.setattr(
        search.expand,
        "expand_query_practice_fast",
        lambda topic, *_args, **_kwargs: {
            "corrected": topic,
            "queries": [topic],
            "provider_used": "literal_fallback",
        },
    )

    def fake_search_all(queries, filters=None, **kwargs):
        page_tokens = kwargs.get("page_tokens")
        calls.append(page_tokens)
        token = page_tokens[0] if page_tokens else None
        if token is None:
            videos = [{"id": "already-consumed"}]
            next_page_token = "page-2"
        elif token == "page-2":
            videos = []
            next_page_token = "page-3"
        else:
            videos = [{"id": "fresh-page-3"}]
            next_page_token = "page-4"
        return {
            "per_query": [{
                "query": queries[0],
                "videos": videos,
                "next_page_token": next_page_token,
            }],
            "credits_used": int(token is not None),
            "warning": None,
        }

    monkeypatch.setattr(search.supadata_search, "search_all", fake_search_all)

    result = search.discover_practice_fast(
        "AP Statistics experimental design",
        limit=1,
        breadth=1,
        consumed_video_ids=["already-consumed"],
        retrieval_profile="deep",
    )

    assert calls == [None, ["page-2"], ["page-3"]]
    assert [video["id"] for video in result["videos"]] == ["fresh-page-3"]
    assert result["provider_exhausted"] is False


def test_terminal_empty_provider_page_is_exhausted(monkeypatch):
    calls = []
    monkeypatch.setattr(
        search.expand,
        "expand_query_practice_fast",
        lambda topic, *_args, **_kwargs: {
            "corrected": topic,
            "queries": [topic],
            "provider_used": "literal_fallback",
        },
    )

    def fake_search_all(queries, filters=None, **kwargs):
        page_tokens = kwargs.get("page_tokens")
        calls.append(page_tokens)
        return {
            "per_query": [{
                "query": queries[0],
                "videos": (
                    [{"id": "already-consumed"}]
                    if page_tokens is None
                    else []
                ),
                "next_page_token": (
                    "page-2" if page_tokens is None else None
                ),
            }],
            "credits_used": int(page_tokens is not None),
            "warning": None,
        }

    monkeypatch.setattr(search.supadata_search, "search_all", fake_search_all)

    result = search.discover_practice_fast(
        "AP Statistics experimental design",
        limit=1,
        breadth=1,
        consumed_video_ids=["already-consumed"],
        retrieval_profile="deep",
    )

    assert calls == [None, ["page-2"]]
    assert result["videos"] == []
    assert result["provider_exhausted"] is True


def test_search_budget_exhaustion_keeps_unfetched_provider_cursor_open(monkeypatch):
    calls = []
    monkeypatch.setattr(
        search.expand,
        "expand_query_practice_fast",
        lambda topic, *_args, **_kwargs: {
            "corrected": topic,
            "queries": [topic],
            "provider_used": "literal_fallback",
        },
    )

    def fake_search_all(queries, filters=None, **kwargs):
        page_tokens = kwargs.get("page_tokens")
        calls.append(page_tokens)
        if page_tokens is not None:
            raise search.ProviderBudgetExceededError(
                "search budget exhausted",
                provider="supadata",
                operation="search",
            )
        return {
            "per_query": [{
                "query": queries[0],
                "videos": [{"id": "already-consumed"}],
                "next_page_token": "page-2",
            }],
            "credits_used": 0,
            "warning": None,
        }

    monkeypatch.setattr(search.supadata_search, "search_all", fake_search_all)

    result = search.discover_practice_fast(
        "AP Statistics chi-square tests",
        limit=1,
        breadth=1,
        consumed_video_ids=["already-consumed"],
        retrieval_profile="deep",
    )

    assert calls == [None, ["page-2"]]
    assert result["videos"] == []
    assert result["provider_exhausted"] is False
    assert "provider pages remaining" in str(result["warning"])


def test_rejected_provider_cursor_continues_through_another_query_branch(monkeypatch):
    calls = []
    topic = "Newton laws worked problems"
    focused = "Newton laws free body diagrams"
    practice = "Newton laws common misconceptions"
    context = GenerationContext("slow")
    monkeypatch.setattr(
        search.expand,
        "expand_query_practice_fast",
        lambda *_args, **_kwargs: {
            "corrected": topic,
            "queries": [topic, focused, practice],
            "provider_used": "gemini",
        },
    )

    def fake_search_all(queries, filters=None, **kwargs):
        assert kwargs.get("context") is context
        for _query in queries:
            context.reserve("search")
        page_tokens = kwargs.get("page_tokens")
        calls.append(page_tokens)
        if page_tokens is None:
            return {
                "per_query": [
                    {
                        "query": topic,
                        "videos": [{"id": "consumed-topic"}],
                        "next_page_token": "bad-cursor",
                    },
                    {
                        "query": focused,
                        "videos": [{"id": "consumed-focused"}],
                        "next_page_token": "good-cursor",
                    },
                    {
                        "query": practice,
                        "videos": [{"id": "consumed-practice"}],
                        "next_page_token": "unused-cursor",
                    },
                ],
                "credits_used": 0,
                "warning": None,
            }
        if page_tokens == ["bad-cursor"]:
            raise ProviderRequestError(
                "Supadata rejected the search request (400).",
                provider="supadata",
                operation="search",
                status_code=400,
                detail="Invalid or expired continuation token",
            )
        return {
            "per_query": [{
                "query": queries[0],
                "videos": [{"id": "fresh-focused"}],
                "next_page_token": None,
            }],
            "credits_used": 1,
            "warning": None,
        }

    monkeypatch.setattr(search.supadata_search, "search_all", fake_search_all)

    result = search.discover_practice_fast(
        topic,
        limit=1,
        breadth=3,
        consumed_video_ids=[
            "consumed-topic",
            "consumed-focused",
            "consumed-practice",
        ],
        retrieval_profile="deep",
        context=context,
    )

    assert calls == [None, ["bad-cursor"], ["good-cursor"]]
    assert [video["id"] for video in result["videos"]] == ["fresh-focused"]
    assert result["provider_exhausted"] is False
    assert "unusable cursor branch was skipped" in str(result["warning"])
    assert context.budget.remaining("search") == 0


def test_rejected_sole_provider_cursor_does_not_claim_another_search(
    monkeypatch,
    caplog,
):
    topic = "LEARNER_CURSOR_SECRET cell membrane transport"
    calls: list[list[str] | None] = []
    caplog.set_level("WARNING", logger=search.__name__)
    monkeypatch.setattr(
        search.expand,
        "expand_query_practice_fast",
        lambda *_args, **_kwargs: {
            "corrected": topic,
            "queries": [topic],
            "provider_used": "gemini",
            "intent_contract": _intent_contract(topic),
        },
    )

    def fake_search_all(queries, filters=None, **kwargs):
        del filters
        page_tokens = kwargs.get("page_tokens")
        calls.append(page_tokens)
        if page_tokens:
            raise ProviderRequestError(
                "Supadata rejected the search request (400).",
                provider="supadata",
                operation="search",
                status_code=400,
                detail="Invalid or expired continuation token",
            )
        return {
            "per_query": [{
                "query": queries[0],
                "videos": [{"id": "already-consumed"}],
                "next_page_token": "bad-cursor",
            }],
            "credits_used": 0,
            "warning": None,
        }

    monkeypatch.setattr(search.supadata_search, "search_all", fake_search_all)

    result = search.discover_practice_fast(
        topic,
        limit=1,
        breadth=1,
        consumed_video_ids=["already-consumed"],
        retrieval_profile="deep",
    )

    assert calls == [None, ["bad-cursor"]]
    assert result["videos"] == []
    assert "unusable cursor branch was skipped" in str(result["warning"])
    assert "continued" not in str(result["warning"])
    assert "skipping unusable cursor branch" in caplog.text
    assert "LEARNER_CURSOR_SECRET" not in caplog.text


def test_literal_only_rejected_cursor_retries_with_fresh_grounded_branch(
    monkeypatch,
):
    topic = (
        "Newton's laws: begin with first-law inertia and balanced forces, then "
        "net force and F=ma, then free-body diagrams, then third-law "
        "action-reaction pairs, and finish with worked problems and common "
        "misconceptions."
    )
    context = GenerationContext("slow")
    calls: list[tuple[list[str], list[str] | None]] = []
    expansion_calls: list[tuple[str, int, list[str]]] = []

    def fake_expand(request, count, **kwargs):
        tried_queries = list(kwargs.get("tried_queries") or [])
        expansion_calls.append((request, count, tried_queries))
        if not tried_queries:
            return {
                "corrected": topic,
                "queries": [topic],
                "provider_used": "literal_fallback",
            }
        assert tried_queries == [topic]
        return {
            "corrected": topic,
            "queries": ["Newton's laws first-law inertia"],
            "provider_used": "gemini",
            "intent_contract": _intent_contract(topic),
        }

    monkeypatch.setattr(search.expand, "expand_query_practice_fast", fake_expand)

    def fake_search_all(queries, filters=None, **kwargs):
        del filters
        page_tokens = kwargs.get("page_tokens")
        calls.append((list(queries), page_tokens))
        if page_tokens == ["bad-cursor"]:
            context.reserve("search")
            raise ProviderRequestError(
                "Supadata rejected the search request (400).",
                provider="supadata",
                operation="search",
                status_code=400,
                detail="Invalid or expired continuation token",
            )
        if queries == [topic]:
            context.reserve("search")
            return {
                "per_query": [{
                    "query": topic,
                    "videos": [{"id": "already-consumed"}],
                    "next_page_token": "bad-cursor",
                }],
                "credits_used": 0,
                "warning": None,
            }
        context.reserve("search")
        return {
            "per_query": [{
                "query": queries[0],
                "videos": [{"id": "fresh-recovery"}],
                "next_page_token": None,
            }],
            "credits_used": 1,
            "warning": None,
        }

    monkeypatch.setattr(search.supadata_search, "search_all", fake_search_all)

    result = search.discover_practice_fast(
        topic,
        limit=1,
        breadth=3,
        consumed_video_ids=["already-consumed"],
        retrieval_profile="deep",
        context=context,
    )

    assert calls == [
        ([topic], None),
        ([topic], ["bad-cursor"]),
        (["Newton's laws first-law inertia"], None),
    ]
    assert expansion_calls == [
        (topic, 3, []),
        (topic, 4, [topic]),
    ]
    assert [video["id"] for video in result["videos"]] == ["fresh-recovery"]
    assert "unusable cursor branch was skipped" in str(result["warning"])
    assert context.budget.remaining("search") == 2


@pytest.mark.parametrize(
    ("exact_request", "tried_query", "novel_query"),
    [
        (
            "Newton's laws from inertia through free-body diagram problems",
            "Newton laws tutorial",
            "Newton laws free body diagram worked examples",
        ),
        (
            "Cellular respiration from glycolysis through ATP yield",
            "cellular respiration overview",
            "cellular respiration ATP yield worked explanation",
        ),
        (
            "Python generators from iterators through async generators",
            "python generators tutorial",
            "python async generators explained with examples",
        ),
        (
            "Negligence from duty through causation and damages",
            "negligence law overview",
            "negligence causation and damages worked fact pattern",
        ),
    ],
)
def test_zero_result_recovery_preserves_cross_domain_request_shapes(
    monkeypatch,
    exact_request,
    tried_query,
    novel_query,
) -> None:
    captured: dict[str, object] = {}

    def fake_raw(topic, count, **kwargs):
        captured.update(
            topic=topic,
            count=count,
            tried_queries=list(kwargs.get("tried_queries") or []),
        )
        return _intent_expansion_json(
            corrected=exact_request,
            source_phrase=exact_request,
            queries=[tried_query, novel_query],
        )

    monkeypatch.setattr(expand, "_practice_fast_gemini_raw", fake_raw)

    result = expand.expand_query_practice_fast(
        exact_request,
        2,
        tried_queries=[tried_query],
    )

    assert captured == {
        "topic": exact_request,
        "count": 2,
        "tried_queries": [tried_query],
    }
    assert result["corrected"] == exact_request
    assert result["queries"] == [novel_query]
    assert result["provider_used"] == "gemini"
    assert result["intent_contract"]["request_intent"]["exact_request"] == exact_request


def test_zero_result_without_cursor_uses_one_ai_recovery_until_inventory_exists(
    monkeypatch,
) -> None:
    exact_request = "Teach Newton's laws with progressively harder problems"
    initial_query = "Newton laws tutorial"
    recovery_queries = [
        "Newton laws balanced forces worked problems",
        "Newton laws free body diagram problem progression",
    ]
    context = GenerationContext("slow")
    expansion_calls: list[
        tuple[str, int, list[str], str | None, list[str]]
    ] = []
    search_calls: list[str] = []

    def fake_expand(request, count, **kwargs):
        tried_queries = list(kwargs.get("tried_queries") or [])
        expansion_calls.append((request, count, tried_queries))
        if not tried_queries:
            return {
                "corrected": exact_request,
                "queries": [initial_query],
                "provider_used": "gemini",
                "intent_contract": _intent_contract(exact_request),
            }
        return {
            "corrected": exact_request,
            "queries": [initial_query, *recovery_queries],
            "provider_used": "gemini",
            "intent_contract": _intent_contract(exact_request),
        }

    def fake_search_all(queries, filters=None, **kwargs):
        del filters, kwargs
        assert len(queries) == 1
        query = queries[0]
        search_calls.append(query)
        context.reserve("search")
        return {
            "per_query": [{
                "query": query,
                "videos": (
                    [{"id": "recovered-video"}]
                    if query == recovery_queries[0]
                    else []
                ),
                "next_page_token": None,
            }],
            "credits_used": 1,
            "warning": None,
        }

    monkeypatch.setattr(search.expand, "expand_query_practice_fast", fake_expand)
    monkeypatch.setattr(search.supadata_search, "search_all", fake_search_all)

    result = search.discover_practice_fast(
        exact_request,
        limit=1,
        breadth=1,
        retrieval_profile="deep",
        context=context,
    )

    assert expansion_calls == [
        (exact_request, 1, []),
        (exact_request, 4, [initial_query]),
    ]
    assert search_calls == [initial_query, recovery_queries[0]]
    assert result["queries"] == [initial_query, recovery_queries[0]]
    assert [video["id"] for video in result["videos"]] == ["recovered-video"]
    assert result["provider_exhausted"] is False
    assert context.budget.remaining("search") == 3


def test_zero_result_recovery_uses_exact_five_search_budget(monkeypatch) -> None:
    exact_request = "Explain cellular respiration and ATP yield"
    initial_query = "cellular respiration overview"
    alternatives = [
        "cellular respiration glycolysis ATP explanation",
        "cellular respiration Krebs cycle ATP explanation",
        "cellular respiration electron transport ATP explanation",
        "cellular respiration ATP yield worked explanation",
    ]
    context = GenerationContext("slow")
    expansion_calls: list[tuple[int, list[str]]] = []
    search_calls: list[str] = []

    def fake_expand(request, count, **kwargs):
        assert request == exact_request
        tried_queries = list(kwargs.get("tried_queries") or [])
        expansion_calls.append((count, tried_queries))
        return {
            "corrected": exact_request,
            "queries": alternatives if tried_queries else [initial_query],
            "provider_used": "gemini",
            "intent_contract": _intent_contract(exact_request),
        }

    def fake_search_all(queries, filters=None, **kwargs):
        del filters, kwargs
        assert len(queries) == 1
        query = queries[0]
        search_calls.append(query)
        context.reserve("search")
        return {
            "per_query": [{
                "query": query,
                "videos": [],
                "next_page_token": None,
            }],
            "credits_used": 1,
            "warning": None,
        }

    monkeypatch.setattr(search.expand, "expand_query_practice_fast", fake_expand)
    monkeypatch.setattr(search.supadata_search, "search_all", fake_search_all)

    result = search.discover_practice_fast(
        exact_request,
        limit=1,
        breadth=1,
        retrieval_profile="deep",
        context=context,
    )

    assert expansion_calls == [(1, []), (4, [initial_query])]
    assert search_calls == [initial_query, *alternatives]
    assert result["queries"] == search_calls
    assert result["videos"] == []
    assert result["provider_exhausted"] is True
    assert context.budget.remaining("search") == 0


def test_healthy_nonempty_discovery_does_not_call_recovery(monkeypatch) -> None:
    exact_request = "Explain Python generators"
    initial_query = "Python generators tutorial"
    context = GenerationContext("slow")
    expansion_calls: list[list[str]] = []
    search_calls: list[list[str]] = []

    def fake_expand(request, _count, **kwargs):
        assert request == exact_request
        expansion_calls.append(list(kwargs.get("tried_queries") or []))
        return {
            "corrected": exact_request,
            "queries": [initial_query],
            "provider_used": "gemini",
            "intent_contract": _intent_contract(exact_request),
        }

    def fake_search_all(queries, filters=None, **kwargs):
        del filters, kwargs
        search_calls.append(list(queries))
        context.reserve("search")
        return {
            "per_query": [{
                "query": initial_query,
                "videos": [{"id": "healthy-video"}],
                "next_page_token": None,
            }],
            "credits_used": 1,
            "warning": None,
        }

    monkeypatch.setattr(search.expand, "expand_query_practice_fast", fake_expand)
    monkeypatch.setattr(search.supadata_search, "search_all", fake_search_all)

    result = search.discover_practice_fast(
        exact_request,
        limit=1,
        breadth=1,
        retrieval_profile="deep",
        context=context,
    )

    assert expansion_calls == [[]]
    assert search_calls == [[initial_query]]
    assert [video["id"] for video in result["videos"]] == ["healthy-video"]
    assert context.budget.remaining("search") == 4


def test_zero_result_recovery_duplicates_do_not_reserve_search(monkeypatch) -> None:
    exact_request = "Explain C++ RAII ownership"
    initial_query = "C++ RAII ownership tutorial"
    novel_query = "C++ RAII destructor ownership example"
    context = GenerationContext("slow")
    search_calls: list[str] = []

    def fake_expand(request, _count, **kwargs):
        assert request == exact_request
        if not kwargs.get("tried_queries"):
            queries = [initial_query]
        else:
            queries = [
                f"  {initial_query}  ",
                f"{initial_query}!",
                novel_query,
                f" {novel_query} ",
            ]
        return {
            "corrected": exact_request,
            "queries": queries,
            "provider_used": "gemini",
            "intent_contract": _intent_contract(exact_request),
        }

    def fake_search_all(queries, filters=None, **kwargs):
        del filters, kwargs
        assert len(queries) == 1
        search_calls.append(queries[0])
        context.reserve("search")
        return {
            "per_query": [{
                "query": queries[0],
                "videos": [],
                "next_page_token": None,
            }],
            "credits_used": 1,
            "warning": None,
        }

    monkeypatch.setattr(search.expand, "expand_query_practice_fast", fake_expand)
    monkeypatch.setattr(search.supadata_search, "search_all", fake_search_all)

    result = search.discover_practice_fast(
        exact_request,
        limit=1,
        breadth=1,
        retrieval_profile="deep",
        context=context,
    )

    assert search_calls == [initial_query, novel_query]
    assert result["queries"] == search_calls
    assert context.budget.remaining("search") == 3


def test_explicit_clip_rejection_recovery_searches_only_novel_ai_queries(
    monkeypatch,
) -> None:
    exact_request = "Explain series circuit current and voltage"
    tried_queries = [
        "series circuit current explained",
        "series circuit voltage tutorial",
        "series circuit current!",
    ]
    novel_queries = [
        "series circuit current voltage worked example",
        "series circuit voltage drops derivation",
    ]
    context = GenerationContext("slow")
    for _ in range(3):
        context.reserve("search")
    expansion_calls: list[tuple[str, int, list[str]]] = []
    search_calls: list[list[str]] = []

    def fake_expand(topic, count, **kwargs):
        expansion_calls.append(
            (
                topic,
                count,
                list(kwargs.get("tried_queries") or []),
                kwargs.get("recovery_reason"),
                list(kwargs.get("rejected_video_ids") or []),
            )
        )
        return {
            "corrected": exact_request,
            "queries": [tried_queries[0], *novel_queries],
            "provider_used": "gemini",
            "intent_contract": _intent_contract(exact_request),
        }

    def fake_search_all(queries, filters=None, **_kwargs):
        del filters
        search_calls.append(list(queries))
        context.budget.reserve("search", len(queries))
        return {
            "per_query": [
                {
                    "query": query,
                    "videos": [{"id": f"recovered-{index}"}],
                    "next_page_token": None,
                }
                for index, query in enumerate(queries)
            ],
            "credits_used": len(queries),
            "warning": None,
        }

    monkeypatch.setattr(search.expand, "expand_query_practice_fast", fake_expand)
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
        exact_request,
        limit=2,
        breadth=8,
        literal_topic=exact_request,
        retrieval_profile="bootstrap",
        context=context,
        recovery_tried_queries=tried_queries,
        recovery_reason=expand.RECOVERY_REASON_ZERO_VALID_CLIPS,
        recovery_rejected_video_ids=["rejected-a", "rejected-b"],
    )

    assert expansion_calls == [(
        exact_request,
        2,
        tried_queries,
        expand.RECOVERY_REASON_ZERO_VALID_CLIPS,
        ["rejected-a", "rejected-b"],
    )]
    assert search_calls == [novel_queries]
    assert result["queries"] == novel_queries
    assert [video["id"] for video in result["videos"]] == [
        "recovered-0",
        "recovered-1",
    ]
    assert context.budget.remaining("search") == 0


@pytest.mark.parametrize(("mode", "analysis_limit"), [("fast", 2), ("slow", 3)])
def test_healthy_initial_analysis_target_does_not_page_into_recovery_capacity(
    monkeypatch,
    mode: str,
    analysis_limit: int,
) -> None:
    context = GenerationContext(mode)
    search_calls: list[tuple[list[str], list[str] | None]] = []
    expansion_calls = 0

    def fake_expand(topic, _count, **kwargs):
        nonlocal expansion_calls
        expansion_calls += 1
        assert not kwargs.get("tried_queries")
        return {
            "corrected": topic,
            "queries": [topic],
            "provider_used": "gemini",
            "intent_contract": _intent_contract(topic),
        }

    def fake_search_all(queries, filters=None, **kwargs):
        del filters
        search_calls.append((list(queries), kwargs.get("page_tokens")))
        assert kwargs.get("page_tokens") is None
        context.budget.reserve("search", len(queries))
        return {
            "per_query": [{
                "query": queries[0],
                "videos": [
                    {"id": f"healthy-{index}"}
                    for index in range(analysis_limit)
                ],
                "next_page_token": "unused-open-cursor",
            }],
            "credits_used": 1,
            "warning": None,
        }

    monkeypatch.setattr(search.expand, "expand_query_practice_fast", fake_expand)
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
        "physics",
        limit=analysis_limit * 2,
        breadth=8,
        context=context,
        retrieval_profile="deep",
        analysis_limit=analysis_limit,
    )

    assert expansion_calls == 1
    assert search_calls == [(["physics"], None)]
    assert [video["id"] for video in result["videos"]][
        :analysis_limit
    ] == [f"healthy-{index}" for index in range(analysis_limit)]
    assert result["provider_exhausted"] is False


def test_initial_provider_transient_does_not_start_ai_recovery(monkeypatch) -> None:
    context = GenerationContext("slow")
    expansion_calls: list[list[str]] = []

    def fake_expand(topic, _count, **kwargs):
        expansion_calls.append(list(kwargs.get("tried_queries") or []))
        return {
            "corrected": topic,
            "queries": [topic],
            "provider_used": "gemini",
            "intent_contract": _intent_contract(topic),
        }

    def fail_search(_queries, filters=None, **kwargs):
        del filters, kwargs
        context.reserve("search")
        raise ProviderTransientError(
            "Could not reach Supadata search.",
            provider="supadata",
            operation="search",
        )

    monkeypatch.setattr(search.expand, "expand_query_practice_fast", fake_expand)
    monkeypatch.setattr(search.supadata_search, "search_all", fail_search)

    with pytest.raises(ProviderTransientError):
        search.discover_practice_fast(
            "Study negligence",
            limit=1,
            breadth=1,
            retrieval_profile="deep",
            context=context,
        )

    assert expansion_calls == [[]]
    assert context.budget.remaining("search") == 4


@pytest.mark.parametrize(
    ("status_code", "detail"),
    [
        (400, "query is required"),
        (400, "cursor cannot be combined with query"),
        (400, "Invalid request: cursor cannot be combined with query"),
        (422, "request validation failed"),
    ],
)
def test_unrelated_provider_cursor_rejection_still_propagates(
    monkeypatch,
    status_code,
    detail,
):
    monkeypatch.setattr(
        search.expand,
        "expand_query_practice_fast",
        lambda topic, *_args, **_kwargs: {
            "corrected": topic,
            "queries": [topic],
            "provider_used": "literal_fallback",
        },
    )

    def fake_search_all(queries, filters=None, **kwargs):
        if kwargs.get("page_tokens") is not None:
            raise ProviderRequestError(
                f"Supadata rejected the search request ({status_code}).",
                provider="supadata",
                operation="search",
                status_code=status_code,
                detail=detail,
            )
        return {
            "per_query": [{
                "query": queries[0],
                "videos": [{"id": "already-consumed"}],
                "next_page_token": "page-2",
            }],
            "credits_used": 0,
            "warning": None,
        }

    monkeypatch.setattr(search.supadata_search, "search_all", fake_search_all)

    with pytest.raises(ProviderRequestError) as exc_info:
        search.discover_practice_fast(
            "AP Statistics regression",
            limit=1,
            breadth=1,
            consumed_video_ids=["already-consumed"],
            retrieval_profile="deep",
        )

    assert exc_info.value.status_code == status_code


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
            "per_query": [
                {
                    "query": query,
                    "videos": ([{"id": "healthy-video"}] if index == 0 else []),
                    "next_page_token": None,
                }
                for index, query in enumerate(queries)
            ],
            "credits_used": 0,
            "warning": None,
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
        ("fast", 2, ["physics mechanics", "physics waves"]),
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


def test_ai_analysis_prefix_covers_an_unseen_query_family_before_repetition():
    ranked = [
        {
            "id": "multi-facet-overview",
            "channel": "Top Classroom",
            "literal_match": False,
            "matched_families": ["overview", "applications"],
        },
        {
            "id": "repeated-overview",
            "channel": "Second Classroom",
            "literal_match": False,
            "matched_families": ["overview"],
        },
        {
            "id": "repeated-applications",
            "channel": "Third Classroom",
            "literal_match": False,
            "matched_families": ["applications"],
        },
        {
            "id": "another-overview",
            "channel": "Fourth Classroom",
            "literal_match": False,
            "matched_families": ["overview"],
        },
        {
            "id": "unseen-worked-examples",
            "channel": "Worked Examples Lab",
            "literal_match": False,
            "matched_families": ["worked-examples"],
        },
    ]

    selected = search._select_ranked_candidates(
        ranked,
        limit=5,
        excluded=set(),
        analysis_prefix=3,
    )

    assert [video["id"] for video in selected[:3]] == [
        "multi-facet-overview",
        "unseen-worked-examples",
        "repeated-overview",
    ]


def test_ai_family_diversity_does_not_reorder_the_unanalyzed_tail():
    ranked = [
        {
            "id": f"candidate-{index}",
            "channel": channel,
            "literal_match": False,
            "matched_families": [family],
        }
        for index, (channel, family) in enumerate((
            ("Channel A", "family-a"),
            ("Channel B", "family-b"),
            ("Channel C", "family-c"),
            ("Channel A", "family-a"),
            ("Channel D", "family-a"),
        ))
    ]

    selected = search._select_ranked_candidates(
        ranked,
        limit=5,
        excluded=set(),
        analysis_prefix=3,
    )

    assert [video["id"] for video in selected] == [
        f"candidate-{index}" for index in range(5)
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


def test_analysis_prefix_prefers_known_short_calculus_sources():
    ranked = [
        {
            "id": "four-minute-overview",
            "channel": "Open College",
            "duration": 4 * 60,
            "literal_match": False,
            "matched_families": ["calculus"],
        },
        {
            "id": "thirty-six-minute-lesson",
            "channel": "Calculus Classroom",
            "duration": 36 * 60,
            "literal_match": False,
            "matched_families": ["limits"],
        },
        {
            "id": "five-minute-derivatives",
            "channel": "Math Lessons",
            "duration": 5 * 60,
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
        "four-minute-overview",
        "five-minute-derivatives",
        "thirty-six-minute-lesson",
    ]


def test_analysis_prefix_prefers_short_sources_when_every_match_is_literal():
    ranked = [
        {
            "id": "literal-full-course",
            "duration": 4 * 60 * 60,
            "literal_match": True,
        },
        {
            "id": "literal-five-minute-overview",
            "duration": 5 * 60,
            "literal_match": True,
        },
        {
            "id": "literal-eight-minute-lesson",
            "duration": 8 * 60,
            "literal_match": True,
        },
    ]

    selected = search._select_ranked_candidates(
        ranked,
        limit=3,
        excluded=set(),
        analysis_prefix=2,
    )

    assert [video["id"] for video in selected] == [
        "literal-five-minute-overview",
        "literal-eight-minute-lesson",
        "literal-full-course",
    ]


def test_analysis_prefix_does_not_promote_distant_short_results():
    ranked = [
        {
            "id": f"ranked-{index}",
            "duration": duration,
            "literal_match": False,
            "matched_families": [f"facet-{index}"],
        }
        for index, duration in enumerate((1_900, 2_100, 2_700, 3_300, 120, 180))
    ]

    selected = search._select_ranked_candidates(
        ranked,
        limit=6,
        excluded=set(),
        analysis_prefix=2,
    )

    assert [video["id"] for video in selected[:2]] == ["ranked-0", "ranked-1"]


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
            "duration": 5 * 60,
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


def test_unknown_duration_is_deferred_when_known_short_sources_fill_prefix():
    ranked = [
        {
            "id": "unknown-duration",
            "channel": "Unknown Channel",
            "literal_match": False,
            "matched_families": ["calculus"],
        },
        {
            "id": "focused-limits",
            "channel": "Limits Lab",
            "duration": 6 * 60,
            "literal_match": False,
            "matched_families": ["limits"],
        },
        {
            "id": "focused-derivatives",
            "channel": "Derivative Desk",
            "duration": 8 * 60,
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
        "unknown-duration",
    ]


def test_unknown_duration_remains_available_without_enough_known_short_sources():
    ranked = [
        {
            "id": "unknown-duration",
            "literal_match": False,
            "matched_families": ["calculus"],
        },
        {
            "id": "focused-limits",
            "duration": 6 * 60,
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
        "unknown-duration",
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
                "duration": 5 * 60 + index,
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
