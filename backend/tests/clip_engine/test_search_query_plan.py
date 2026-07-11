from __future__ import annotations

import json
import sqlite3

import numpy as np
import pytest

from backend.app.db import SCHEMA
from backend.app.clip_engine import rank, search
from backend.app.clip_engine.provider_runtime import GenerationContext
from backend.app.ingestion import pipeline as pipeline_module
from backend.app.ingestion.pipeline import (
    IngestionPipeline,
    _PlatformRateLimiter,
    _retrieval_search_context,
    _strict_topic_clips,
)
from backend.app.services import embeddings as embeddings_module
from backend.app.services import search_query_plan as query_plan_module
from backend.app.services.reels import ReelService
from backend.app.services.search_query_plan import (
    PlannedSearchQuery,
    SearchQueryPlan,
    build_search_query_plan,
    semantic_query_family,
    transcript_window_matches_topic,
)


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def _manual_plan() -> SearchQueryPlan:
    terms = [
        "Calculus Basics",
        "Calculus",
        "limits",
        "derivatives",
        "integrals",
        "fundamental theorem of calculus",
    ]
    return SearchQueryPlan(
        literal_query="Calculus Basics",
        canonical_query="Calculus",
        accepted_subtopics=terms[2:],
        trusted_signature=terms,
        literal_is_ambiguous=True,
        provenance={
            term.casefold(): (["literal"] if index == 0 else ["ai"])
            for index, term in enumerate(terms)
        },
        queries=[
            PlannedSearchQuery(
                text=f"query {index}",
                family=f"family-{index}",
                provenance="fixture",
                trust="literal" if index == 0 else "ai",
            )
            for index in range(12)
        ],
        ai_status="validated",
    )


def test_plan_runs_one_structured_expansion_and_caches_by_normalized_literal(monkeypatch) -> None:
    conn = _conn()
    ai_calls = 0

    def fake_ai(**kwargs):
        nonlocal ai_calls
        ai_calls += 1
        assert kwargs["response_schema"] is query_plan_module.AIQueryExpansion
        return json.dumps(
            {
                "canonical_query": "Calculus",
                "literal_is_ambiguous": True,
                "aliases": ["differential and integral calculus"],
                "subtopics": ["limits", "derivatives", "integrals"],
                "related_terms": ["college admissions"],
            }
        )

    monkeypatch.setattr(query_plan_module.llm_router, "chat_completion", fake_ai)
    monkeypatch.setattr(
        query_plan_module,
        "_semantic_relevance_scores",
        lambda _anchors, candidates: {
            query_plan_module.normalize_query(candidate): 0.9
            for candidate in candidates
        },
    )

    first = build_search_query_plan(conn, literal_query="  Calculus   Basics ")
    second = build_search_query_plan(conn, literal_query="calculus basics")

    assert ai_calls == 1
    assert first == second
    assert first.literal_query == "Calculus Basics"
    assert first.queries[0].text == "Calculus Basics"
    assert first.literal_is_ambiguous is True
    assert [item.text for item in first.queries] == [
        "Calculus Basics",
        "Calculus",
        "differential and integral calculus",
        "limits",
        "derivatives",
        "integrals",
    ]
    assert all(item.provenance == "ai" for item in first.queries[1:])
    assert not any(item.trust == "template" for item in first.queries)
    assert any("low-value intent" in reason for reason in first.rejection_reasons)
    assert conn.execute(
        "SELECT COUNT(*) FROM llm_cache WHERE cache_key LIKE 'search_query_plan:v2:%'"
    ).fetchone()[0] == 1
    conn.close()


def test_stale_last_good_plan_survives_transient_model_unavailability(monkeypatch) -> None:
    conn = _conn()
    monkeypatch.setattr(
        query_plan_module.llm_router,
        "chat_completion",
        lambda **_kwargs: json.dumps(
            {
                "canonical_query": "Calculus",
                "literal_is_ambiguous": False,
                "aliases": [],
                "subtopics": ["calculus differentiation"],
                "related_terms": [],
            }
        ),
    )
    monkeypatch.setattr(
        query_plan_module,
        "_semantic_relevance_scores",
        lambda _anchors, candidates: {
            query_plan_module.normalize_query(candidate): 0.9
            for candidate in candidates
        },
    )
    good = build_search_query_plan(conn, literal_query="Calculus Basics")
    conn.execute(
        "UPDATE llm_cache SET created_at = '2020-01-01T00:00:00+00:00' "
        "WHERE cache_key LIKE 'search_query_plan:v2:%'"
    )
    monkeypatch.setattr(
        query_plan_module.llm_router,
        "chat_completion",
        lambda **_kwargs: None,
    )

    fallback = build_search_query_plan(conn, literal_query="Calculus Basics")

    assert fallback == good
    assert fallback.ai_status == "validated"
    conn.close()


def test_stale_last_good_plan_survives_invalid_structured_expansion(monkeypatch) -> None:
    conn = _conn()
    monkeypatch.setattr(
        query_plan_module.llm_router,
        "chat_completion",
        lambda **_kwargs: json.dumps(
            {
                "canonical_query": "Calculus",
                "literal_is_ambiguous": False,
                "aliases": [],
                "subtopics": ["calculus differentiation"],
                "related_terms": [],
            }
        ),
    )
    monkeypatch.setattr(
        query_plan_module,
        "_semantic_relevance_scores",
        lambda _anchors, candidates: {
            query_plan_module.normalize_query(candidate): 0.9
            for candidate in candidates
        },
    )
    good = build_search_query_plan(conn, literal_query="Calculus Basics")
    conn.execute(
        "UPDATE llm_cache SET created_at = '2020-01-01T00:00:00+00:00' "
        "WHERE cache_key LIKE 'search_query_plan:v2:%'"
    )
    monkeypatch.setattr(
        query_plan_module.llm_router,
        "chat_completion",
        lambda **_kwargs: '{"unexpected":true}',
    )

    assert build_search_query_plan(conn, literal_query="Calculus Basics") == good
    conn.close()


def test_model_unavailable_plan_uses_literal_only_and_short_cache(monkeypatch) -> None:
    conn = _conn()
    ai_calls = 0

    def unavailable(**_kwargs):
        nonlocal ai_calls
        ai_calls += 1
        return None

    monkeypatch.setattr(query_plan_module.llm_router, "chat_completion", unavailable)
    first = build_search_query_plan(conn, literal_query="Calculus Basics")
    second = build_search_query_plan(conn, literal_query=" calculus basics ")

    assert ai_calls == 1
    assert first == second
    assert first.ai_status == "unavailable"
    assert first.accepted_aliases == []
    assert first.accepted_subtopics == []
    assert first.accepted_related_terms == []
    assert first.trusted_signature == ["Calculus Basics"]
    assert [query.text for query in first.queries] == ["Calculus Basics"]
    assert transcript_window_matches_topic(
        "Photosynthesis converts light into chemical energy.",
        SearchQueryPlan(
            literal_query="Photosynthesis",
            canonical_query="Photosynthesis",
            trusted_signature=["Photosynthesis"],
            provenance={"photosynthesis": ["literal"]},
            queries=[
                PlannedSearchQuery(
                    text="Photosynthesis",
                    family="photosynthesi",
                    provenance="literal",
                    trust="literal",
                )
            ],
        ),
    )
    conn.close()


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {
            "canonical_query": "Calculus",
            "aliases": [],
            "subtopics": ["derivatives"],
            "related_terms": [],
        },
    ],
)
def test_missing_required_ai_plan_fields_are_invalid_and_short_cached(monkeypatch, payload) -> None:
    conn = _conn()
    calls = 0

    def invalid_plan(**_kwargs):
        nonlocal calls
        calls += 1
        return json.dumps(payload)

    monkeypatch.setattr(query_plan_module.llm_router, "chat_completion", invalid_plan)
    first = build_search_query_plan(conn, literal_query="Calculus Basics")
    second = build_search_query_plan(conn, literal_query=" calculus basics ")

    assert calls == 1
    assert first == second
    assert first.ai_status == "invalid"
    assert [query.text for query in first.queries] == ["Calculus Basics"]
    conn.close()


def test_schema_valid_ai_drift_is_rejected_by_semantic_validation(monkeypatch) -> None:
    conn = _conn()
    monkeypatch.setattr(
        query_plan_module.llm_router,
        "chat_completion",
        lambda **_kwargs: json.dumps(
            {
                "canonical_query": "Propositional Logic",
                "literal_is_ambiguous": True,
                "aliases": [],
                "subtopics": ["truth tables"],
                "related_terms": ["logical connectives"],
            }
        ),
    )
    monkeypatch.setattr(
        query_plan_module,
        "_semantic_relevance_scores",
        lambda _anchors, candidates: {
            query_plan_module.normalize_query(candidate): 0.1
            for candidate in candidates
        },
    )

    plan = build_search_query_plan(conn, literal_query="Calculus Basics")

    assert plan.canonical_query == "Calculus Basics"
    assert plan.accepted_aliases == []
    assert plan.accepted_subtopics == []
    assert plan.accepted_related_terms == []
    assert [query.text for query in plan.queries] == ["Calculus Basics"]
    assert any("not semantically anchored" in reason for reason in plan.rejection_reasons)
    conn.close()


def test_calculus_basics_timestamped_window_corpus_is_fail_closed() -> None:
    plan = _manual_plan()
    accepted = [
        "A limit describes the value a function approaches.",
        "The derivative is an instantaneous rate of change.",
        "A definite integral accumulates signed area.",
        "The fundamental theorem connects derivatives and integrals.",
    ]
    rejected = [
        "Propositions combine under propositional logic and truth tables.",
        "Propositional calculus studies truth tables and logical connectives.",
        "This AP class ranking compares the easiest courses.",
        "College admissions officers compare application essays.",
        "Force, energy, and momentum are the main ideas in this physics lesson.",
        "Dental calculus is tartar that a hygienist removes from teeth.",
    ]

    assert all(transcript_window_matches_topic(text, plan) for text in accepted)
    assert not any(transcript_window_matches_topic(text, plan) for text in rejected)


def test_final_gate_uses_exact_timestamped_cues_for_native_or_auto_transcripts() -> None:
    plan = _manual_plan()
    clips = [
        {"start": 0.0, "end": 20.0, "cue_ids": ["cue-good"]},
        {"start": 20.0, "end": 40.0, "cue_ids": ["cue-bad"]},
    ]
    transcript = {
        "source": "supadata",
        "native_mode": False,
        "artifact_key": "supadata-transcript:v2:test",
        "segments": [
            {"cue_id": "cue-good", "start": 0.0, "end": 20.0, "text": "Derivatives measure rates of change."},
            {"cue_id": "cue-bad", "start": 20.0, "end": 40.0, "text": "Propositional logic uses truth tables."},
        ],
    }

    kept = _strict_topic_clips(clips, transcript, plan)

    assert kept == [clips[0]]
    assert kept[0]["topic_evidence_terms"]
    assert _strict_topic_clips(clips, {**transcript, "native_mode": True}, plan) == [clips[0]]
    assert _strict_topic_clips(clips, {**transcript, "artifact_key": ""}, plan) == []
    assert _strict_topic_clips(clips, {**transcript, "source": "untrusted"}, plan) == []
    assert _strict_topic_clips(
        clips,
        {**transcript, "segments": list(reversed(transcript["segments"]))},
        plan,
    ) == []
    assert _strict_topic_clips([clips[1]], transcript, plan) == []


def test_search_context_keeps_plan_and_provider_query_evidence() -> None:
    plan = _manual_plan()
    context = _retrieval_search_context(
        requested_topic="derivative intuition",
        corrected_topic="derivative intuition",
        video={
            "matched_queries": ["Calculus Basics", "derivatives"],
            "matched_families": ["calculus", "derivative"],
            "matched_query_provenance": {
                "Calculus Basics": "literal",
                "derivatives": "ai",
            },
        },
        query_plan=plan,
        creative_commons_only=False,
        source_duration="medium",
    )

    assert context["literal_query"] == "Calculus Basics"
    assert context["query_plan_version"] == plan.version
    assert context["matched_query_families"] == ["calculus", "derivative"]
    assert context["matched_query_provenance"]["derivatives"] == "ai"


def test_durable_topic_and_ingest_search_share_planner_enabled_discovery(monkeypatch) -> None:
    captured: list[dict] = []

    def fake_discover(topic, **kwargs):
        captured.append({"topic": topic, **kwargs})
        return {
            "corrected": topic,
            "videos": [],
            "credits_used": 0,
            "warning": None,
            "query_plan": _manual_plan(),
        }

    monkeypatch.setattr(pipeline_module.clip_engine_search, "discover", fake_discover)
    pipeline = IngestionPipeline(
        youtube_service=None,
        embedding_service=None,
        rate_limiter=_PlatformRateLimiter(overrides={"yt": (100, 60.0)}),
    )

    pipeline.ingest_topic(
        topic="derivative intuition",
        literal_topic="Calculus Basics",
        material_id="material",
        concept_id="concept",
        max_videos=1,
        dry_run=True,
    )
    pipeline.ingest_search(
        query="Calculus Basics",
        material_id="material",
        max_per_platform=1,
    )

    assert len(captured) == 2
    assert all(call["use_query_planner"] is True for call in captured)
    assert captured[0]["literal_topic"] == "Calculus Basics"
    assert captured[1]["literal_topic"] == "Calculus Basics"
    assert captured[1]["breadth"] == 3


def test_fast_and_slow_plans_use_three_then_six_plus_three_plus_three(monkeypatch) -> None:
    plan = _manual_plan()
    calls: list[list[str]] = []

    def fake_search_all(queries, _filters=None, **_kwargs):
        calls.append(list(queries))
        return {"per_query": [{"query": item, "videos": []} for item in queries], "credits_used": 0, "warning": None}

    monkeypatch.setattr(search.supadata_search, "search_all", fake_search_all)

    fast = GenerationContext("fast")
    fast.budget.reserve_pass()
    search.discover("concept", limit=1, context=fast, query_plan=plan)

    slow = GenerationContext("slow")
    slow.budget.reserve_pass()
    search.discover("concept", limit=1, context=slow, query_plan=plan)
    slow.budget.reserve_pass()
    search.discover("concept", limit=1, context=slow, query_plan=plan)
    slow.budget.reserve_pass()
    search.discover("concept", limit=1, context=slow, query_plan=plan)

    assert calls == [
        ["Calculus Basics", "Calculus Basics", "query 1"],
        ["Calculus Basics", "Calculus Basics", "query 1", "query 2", "query 3", "query 4"],
        ["Calculus Basics", "Calculus Basics", "query 5"],
        ["Calculus Basics", "Calculus Basics", "query 6"],
    ]


def test_consensus_counts_semantic_families_not_templates() -> None:
    assert semantic_query_family("Calculus") == semantic_query_family("calculus explained")
    repeated_templates = [
        {"query": query, "videos": [{"id": "same"}]}
        for query in ("calculus", "calculus explained", "calculus tutorial")
    ]
    distinct_subtopic = [
        *repeated_templates,
        {"query": "derivatives", "videos": [{"id": "same"}]},
    ]

    assert search._consensus_count(repeated_templates, set()) == 0
    assert search._consensus_count(distinct_subtopic, set()) == 1
    assert search._consensus_count(
        [
            {"query": "machine learning", "query_family": "machine learning", "videos": [{"id": "alias"}]},
            {"query": "ML", "query_family": "machine learning", "videos": [{"id": "alias"}]},
        ],
        set(),
    ) == 0


def test_literal_match_outranks_and_ai_family_boost_is_capped() -> None:
    per_query = [
        {
            "query": "Calculus Basics",
            "query_family": "calculus",
            "query_trust": "literal",
            "videos": [{"id": "literal", "title": "Calculus lesson", "viewCount": 10}],
        },
        *[
            {
                "query": f"ai-{index}",
                "query_family": f"ai-{index}",
                "query_trust": "ai",
                "videos": [{"id": "ai", "title": "Popular video", "viewCount": 10_000_000}],
            }
            for index in range(5)
        ],
    ]

    ranked = rank.merge_and_rank(per_query)

    assert ranked[0]["id"] == "literal"
    ai = next(item for item in ranked if item["id"] == "ai")
    baseline_without_ai = np.log10(10_000_000 + 10) + 2.0
    assert ai["score"] - baseline_without_ai <= 0.75 + 1e-6


def test_candidate_cap_reserves_room_for_ai_expansion_family() -> None:
    ranked = [
        {
            "id": f"literal-{index}",
            "literal_match": True,
            "canonical_match": False,
            "matched_families": ["calculus"],
        }
        for index in range(4)
    ] + [
        {
            "id": "derivative-video",
            "literal_match": False,
            "canonical_match": False,
            "matched_families": ["derivative"],
        }
    ]

    selected = search._select_ranked_candidates(ranked, limit=3, excluded=set())

    assert [video["id"] for video in selected] == [
        "literal-0",
        "literal-1",
        "derivative-video",
    ]


def test_candidate_cap_reserves_canonical_correction_before_ai_subtopics() -> None:
    ranked = [
        {
            "id": f"literal-{index}",
            "literal_match": True,
            "canonical_match": False,
            "matched_families": ["calclus"],
        }
        for index in range(4)
    ] + [
        {
            "id": "canonical-video",
            "literal_match": False,
            "canonical_match": True,
            "matched_families": ["calculus"],
        },
        {
            "id": "derivative-video",
            "literal_match": False,
            "canonical_match": False,
            "matched_families": ["derivative"],
        },
    ]

    selected = search._select_ranked_candidates(ranked, limit=3, excluded=set())

    assert [video["id"] for video in selected] == [
        "literal-0",
        "literal-1",
        "canonical-video",
    ]


def test_hash_embedding_never_supplies_semantic_relevance_proof() -> None:
    class HashOnlyEmbedding:
        dim = 256
        semantic_available = False

        def embed_texts(self, *_args, **_kwargs):
            raise AssertionError("hash vectors must not be used as semantic proof")

    service = ReelService(embedding_service=HashOnlyEmbedding(), youtube_service=None)
    relevance = service._score_text_relevance(
        None,
        text="Propositional logic uses truth tables.",
        concept_terms=["Calculus Basics"],
        context_terms=["limits", "derivatives", "integrals"],
        concept_embedding=np.ones(256, dtype=np.float32),
        subject_tag="Calculus Basics",
    )

    assert relevance["embedding_sim"] == 0.0
    assert service._passes_relevance_gate(relevance, require_context=True, fast_mode=False) is False


def test_runtime_semantic_failure_never_falls_back_to_hash_proof() -> None:
    class FailingSemanticEmbedding:
        dim = 384
        semantic_available = True

        def embed_semantic(self, _texts):
            return None

        def embed_texts(self, *_args, **_kwargs):
            raise AssertionError("hash fallback must not be consulted for semantic proof")

    service = ReelService(embedding_service=FailingSemanticEmbedding(), youtube_service=None)
    relevance = service._score_text_relevance(
        None,
        text="Propositional logic uses truth tables.",
        concept_terms=["Calculus Basics"],
        context_terms=["derivatives"],
        concept_embedding=np.ones(384, dtype=np.float32),
        subject_tag="Calculus Basics",
    )

    assert relevance["embedding_sim"] == 0.0
    assert service._passes_relevance_gate(relevance, require_context=True, fast_mode=False) is False


def test_loaded_semantic_backend_fails_closed_instead_of_returning_hash_vectors() -> None:
    class BrokenModel:
        def encode(self, *_args, **_kwargs):
            raise RuntimeError("inference failed")

    embedding = object.__new__(embeddings_module.EmbeddingService)
    embedding._semantic_model = BrokenModel()
    embedding.dim = 384

    with pytest.raises(RuntimeError, match="Semantic embedding inference failed"):
        embedding._embed_local(["calculus"])


def test_failed_semantic_model_load_reports_hash_backend_without_semantic_proof(monkeypatch) -> None:
    monkeypatch.setattr(embeddings_module, "_get_semantic_model", lambda: None)
    embedding = embeddings_module.EmbeddingService()

    assert embedding.semantic_available is False
    assert embedding.backend_name == "hash-lexical-fallback"


def test_strict_topic_never_backfills_off_topic_local_cache_candidate() -> None:
    conn = _conn()
    conn.execute(
        "INSERT INTO materials (id, subject_tag, raw_text, source_type, created_at) "
        "VALUES ('material', 'Calculus Basics', 'Topic: Calculus Basics', 'topic', '2026-07-11T00:00:00+00:00')"
    )
    conn.execute(
        "INSERT INTO concepts (id, material_id, title, keywords_json, summary, created_at) "
        "VALUES ('concept', 'material', 'Calculus Basics', '[\"calculus\"]', 'Math', '2026-07-11T00:00:00+00:00')"
    )
    conn.execute(
        "INSERT INTO videos (id, title, channel_title, description, duration_sec, created_at) "
        "VALUES ('yt:offtopic', 'Propositional Logic Tutorial', 'Logic Channel', "
        "'Truth tables for propositions and logical connectives.', 600, '2026-07-11T00:00:00+00:00')"
    )
    conn.execute(
        "INSERT INTO reels (id, material_id, concept_id, video_id, video_url, t_start, t_end, "
        "transcript_snippet, takeaways_json, base_score, created_at) VALUES "
        "('reel', 'material', 'concept', 'yt:offtopic', 'https://example.test/offtopic', 0, 30, "
        "'A proposition can be true or false.', '[]', 1, '2026-07-11T00:00:00+00:00')"
    )

    class HashOnlyEmbedding:
        dim = 256
        semantic_available = False

    service = ReelService(embedding_service=HashOnlyEmbedding(), youtube_service=None)
    candidates = service._recover_candidates_from_local_corpus(
        conn,
        material_id="material",
        concept_terms=["Calculus Basics", "limits", "derivatives", "integrals"],
        context_terms=["fundamental theorem of calculus"],
        concept_embedding=None,
        subject_tag="Calculus Basics",
        visual_spec={"environment": [], "objects": [], "actions": []},
        preferred_video_duration="any",
        fast_mode=False,
        strict_topic_only=True,
        existing_video_counts={},
        generated_video_counts={},
        max_segments_per_video=2,
        concept_title="Calculus Basics",
        root_topic_terms=["Calculus Basics", "Calculus"],
        bootstrap_fallback=True,
    )

    assert candidates == []
    conn.close()
