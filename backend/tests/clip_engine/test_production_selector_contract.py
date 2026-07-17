from __future__ import annotations

import json
import math
import time
from types import SimpleNamespace

import pytest

from backend import gemini_client
from backend.app.clip_engine.errors import ProviderBudgetExceededError
from backend.app.clip_engine.provider_runtime import GenerationContext
from backend.pipeline import gemini_segment


def _proposal(*, end_line: int = 0) -> gemini_segment._BoundaryTopic:
    return gemini_segment._BoundaryTopic(
        candidate_id="photosynthesis-core",
        start_line=0,
        end_line=end_line,
        start_quote="Cells use chlorophyll to capture light energy",
        end_quote="chemical reactions of photosynthesis",
        title="How photosynthesis captures energy",
        learning_objective="Explain how chlorophyll powers photosynthesis",
        facet="photosynthesis",
        reason="The span directly explains the core mechanism.",
        informativeness=0.9,
        topic_relevance=0.9,
        educational_importance=0.9,
        difficulty=0.2,
        directly_teaches_topic=True,
        substantive=True,
        factually_grounded=True,
        topic_evidence_quote=(
            "Cells use chlorophyll to capture light energy and power the chemical reactions"
        ),
        self_contained=True,
        is_standalone=True,
        prerequisite_candidate_ids=[],
        uncertainty="medium",
        uncertainty_reasons=[gemini_segment._UncertaintyReason.BOUNDARY_AMBIGUOUS],
    )


def _intent_plan(
    *,
    topic: str,
    constraints: list[dict],
    topics: list[gemini_segment._BoundaryTopic],
) -> gemini_segment._IntentBoundaryPlan:
    return gemini_segment._IntentBoundaryPlan(
        request_intent={
            "exact_request": topic,
            "constraints": constraints,
        },
        topics=[
            gemini_segment._IntentBoundaryTopic.model_validate(dict(item.__dict__))
            for item in topics
        ],
    )


def _compact_plan(
    *,
    exact_request: str,
    constraints: list[dict],
    evidence: list[dict],
) -> gemini_segment._CompactBoundaryPlan:
    proposal = _proposal()
    data = {
        key: value
        for key, value in proposal.model_dump().items()
        if key in gemini_segment._CompactBoundaryTopic.model_fields
        and key != "intent_evidence"
    }
    data["claim_quote"] = proposal.topic_evidence_quote
    data["intent_evidence"] = evidence
    return gemini_segment._CompactBoundaryPlan(
        request_intent={
            "exact_request": exact_request,
            "constraints": constraints,
        },
        topics=[gemini_segment._CompactBoundaryTopic.model_validate(data)],
    )


def _compact_custom_plan(
    *,
    request: str,
    start_quote: str,
    end_quote: str,
    claim_quote: str,
) -> gemini_segment._CompactBoundaryPlan:
    return gemini_segment._CompactBoundaryPlan(
        request_intent={
            "exact_request": request,
            "constraints": [{
                "constraint_id": "subject",
                "kind": "subject",
                "source_phrase": request,
                "requirement": f"Teach {request}",
            }],
        },
        topics=[gemini_segment._CompactBoundaryTopic(
            candidate_id="authoritative-model-cut",
            start_line=0,
            end_line=0,
            start_quote=start_quote,
            end_quote=end_quote,
            claim_quote=claim_quote,
            title="Grounded statistics lesson",
            learning_objective=f"Explain {request}",
            facet=request,
            informativeness=0.9,
            topic_relevance=0.9,
            educational_importance=0.9,
            difficulty=0.5,
            directly_teaches_topic=True,
            substantive=True,
            factually_grounded=True,
            self_contained=True,
            is_standalone=True,
            intent_evidence=[{
                "id": "subject",
                "q": claim_quote,
            }],
        )],
    )


def test_single_call_boundary_schema_caps_exhaustive_output_before_truncation() -> None:
    forty = [
        _proposal().model_copy(update={"candidate_id": f"candidate-{index}"})
        for index in range(40)
    ]
    assert len(gemini_segment._BoundaryPlan(topics=forty).topics) == 40
    with pytest.raises(ValueError):
        gemini_segment._BoundaryPlan(topics=[
            *forty,
            _proposal().model_copy(update={"candidate_id": "candidate-40"}),
        ])


def test_selector_contract_allows_short_exact_edges_without_padding() -> None:
    system, user = gemini_segment._boundary_prompts(
        "[0] 00:00 Plants convert light into stored chemical energy.",
        1,
        "photosynthesis",
    )

    assert "shortest unique 1-16" in system
    assert "one-word quote" in user
    assert "never pad" in user
    assert "ignore acoustic silence" in user
    assert "only expand outward" in user
    assert "4-8" not in f"{system}\n{user}"

    schema = gemini_segment._CompactBoundaryTopic.model_json_schema()
    assert "first spoken word" in schema["properties"]["sq"]["description"]
    assert "independently understandable spoken sentence" in (
        schema["properties"]["sq"]["description"]
    )
    assert "never the semantic span or clip duration" in (
        schema["properties"]["sq"]["description"]
    )
    assert "whole same-objective teaching arc" in schema["properties"]["eq"]["description"]
    assert "intermediate result is not an endpoint" in schema["properties"]["eq"]["description"]
    assert "complete concluding sentence or independent clause" in (
        schema["properties"]["eq"]["description"]
    )


def test_selector_contract_prioritizes_wholeness_over_clip_length() -> None:
    system, user = gemini_segment._boundary_prompts(
        "[0] 00:00 State the result. [1] 02:00 Explain why the same result holds.",
        2,
        "the result and its explanation",
    )
    prompt = f"{system}\n{user}".casefold()
    normalized = " ".join(prompt.split())

    assert "context and wholeness have absolute priority over concision" in prompt
    assert "if completeness and brevity conflict, choose the longer complete span" in prompt
    assert "a locally grammatical sentence or intermediate result is not an endpoint" in prompt
    assert "there is no numeric duration cap" in prompt
    assert '"shortest" describes only the quote used to locate that word' in normalized
    assert "never asks for a shorter semantic span or clip" in normalized
    assert "shortest concise span" not in prompt


def test_compact_selector_preserves_the_models_exact_word_interval() -> None:
    text = (
        "Welcome back. Cells use chlorophyll to capture light energy and power the "
        "chemical reactions of photosynthesis. Next, respiration releases that energy."
    )
    plan = _compact_plan(
        exact_request="photosynthesis",
        constraints=[{
            "constraint_id": "subject",
            "kind": "subject",
            "source_phrase": "photosynthesis",
            "requirement": "Teach photosynthesis",
        }],
        evidence=[{
            "id": "subject",
            "q": "power the chemical reactions of photosynthesis",
        }],
    )

    report = gemini_segment._plan_to_report(
        plan,
        [{"cue_id": "cue-0", "start": 0.0, "end": 18.0, "text": text}],
        [],
        {
            "_segment_ignore_caption_case": True,
            "_segment_model_boundary_authoritative": True,
        },
        topic="photosynthesis",
    )

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["_clip_text"] == (
        "Cells use chlorophyll to capture light energy and power the chemical "
        "reactions of photosynthesis"
    )
    assert clip["start_quote"] == "Cells use chlorophyll to capture light energy"
    assert clip["end_quote"] == "chemical reactions of photosynthesis"
    assert clip["edge_projection"] == {
        "start": {
            "required": True,
            "cue_id": "cue-0",
            "quote": "Cells use chlorophyll to capture light energy",
        },
        "end": {
            "required": True,
            "cue_id": "cue-0",
            "quote": "chemical reactions of photosynthesis",
        },
    }


def test_compact_selector_preserves_exact_edges_across_caption_cues() -> None:
    segments = [
        {
            "cue_id": "cue-0",
            "start": 0.0,
            "end": 4.0,
            "text": "Photosynthesis begins when cells",
        },
        {
            "cue_id": "cue-1",
            "start": 4.0,
            "end": 8.0,
            "text": "capture light energy and convert it into",
        },
        {
            "cue_id": "cue-2",
            "start": 8.0,
            "end": 12.0,
            "text": "stored chemical energy for later use.",
        },
    ]
    plan = _compact_plan(
        exact_request="photosynthesis",
        constraints=[{
            "constraint_id": "subject",
            "kind": "subject",
            "source_phrase": "photosynthesis",
            "requirement": "Teach photosynthesis",
        }],
        evidence=[{
            "id": "subject",
            "q": "capture light energy and convert it into",
        }],
    )
    plan = plan.model_copy(update={
        "topics": [plan.topics[0].model_copy(update={
            "start_line": 0,
            "end_line": 2,
            "start_quote": (
                "Photosynthesis begins when cells capture light"
            ),
            "end_quote": "convert it into stored chemical energy for later use",
            "claim_quote": "capture light energy and convert it into",
        })],
    })

    report = gemini_segment._plan_to_report(
        plan,
        segments,
        [],
        {
            "_segment_ignore_caption_case": True,
            "_segment_model_boundary_authoritative": True,
        },
        topic="photosynthesis",
    )

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["_clip_text"] == (
        "Photosynthesis begins when cells capture light energy and convert it "
        "into stored chemical energy for later use."
    )
    assert clip["start_quote"].startswith("Photosynthesis")
    assert clip["end_quote"].endswith("for later use")


def test_compact_selector_uses_exact_quote_inside_coarse_cue_range() -> None:
    segments = [
        {
            "cue_id": "cue-0",
            "start": 0.0,
            "end": 3.0,
            "text": "Before that, recall the null hypothesis.",
        },
        {
            "cue_id": "cue-1",
            "start": 3.0,
            "end": 7.0,
            "text": "A p-value is the probability of results at least this extreme",
        },
        {
            "cue_id": "cue-2",
            "start": 7.0,
            "end": 11.0,
            "text": "assuming the null hypothesis is true.",
        },
    ]
    plan = _compact_plan(
        exact_request="p-values",
        constraints=[{
            "constraint_id": "subject",
            "kind": "subject",
            "source_phrase": "p-values",
            "requirement": "Teach p-values",
        }],
        evidence=[{
            "id": "subject",
            "q": "probability of results at least this extreme",
        }],
    )
    plan = plan.model_copy(update={
        "topics": [plan.topics[0].model_copy(update={
            # Gemini may cite a coarse enclosing range. Its exact words—not the
            # caption provider's line boundary—are the semantic authority.
            "start_line": 0,
            "end_line": 2,
            "start_quote": "A p-value is the probability",
            "end_quote": "assuming the null hypothesis is true",
            "claim_quote": "probability of results at least this extreme",
        })],
    })

    report = gemini_segment._plan_to_report(
        plan,
        segments,
        [],
        {
            "_segment_ignore_caption_case": True,
            "_segment_model_boundary_authoritative": True,
        },
        topic="p-values",
    )

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["cue_ids"] == ["cue-1", "cue-2"]
    assert clip["_clip_text"] == (
        "A p-value is the probability of results at least this extreme "
        "assuming the null hypothesis is true."
    )


def test_authoritative_cross_cue_anchor_keeps_its_unique_full_occurrence() -> None:
    segments = [
        {
            "cue_id": "cue-0",
            "start": 0.0,
            "end": 4.0,
            "text": "A p value means something else. A p value means",
        },
        {
            "cue_id": "cue-1",
            "start": 4.0,
            "end": 8.0,
            "text": (
                "evidence under the null hypothesis. "
                "This conclusion is complete."
            ),
        },
    ]
    model_start = "A p value means evidence under the null hypothesis"
    plan = _compact_plan(
        exact_request="p value",
        constraints=[{
            "constraint_id": "subject",
            "kind": "subject",
            "source_phrase": "p value",
            "requirement": "Teach p value",
        }],
        evidence=[{
            "id": "subject",
            "q": model_start,
        }],
    )
    plan = plan.model_copy(update={
        "topics": [plan.topics[0].model_copy(update={
            "start_line": 0,
            "end_line": 1,
            "start_quote": model_start,
            "end_quote": "This conclusion is complete",
            "claim_quote": model_start,
            "title": "P value meaning",
            "learning_objective": "Explain p value meaning",
            "facet": "p value",
        })],
    })

    report = gemini_segment._plan_to_report(
        plan,
        segments,
        [],
        {
            "_segment_ignore_caption_case": True,
            "_segment_model_boundary_authoritative": True,
        },
        topic="p value",
    )

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["_clip_text"] == (
        "A p value means evidence under the null hypothesis. "
        "This conclusion is complete."
    )
    assert "something else" not in clip["_clip_text"]


def test_authoritative_start_anchor_cannot_extend_past_end_anchor() -> None:
    segments = [
        {
            "cue_id": "cue-0",
            "start": 0.0,
            "end": 4.0,
            "text": "A p value means",
        },
        {
            "cue_id": "cue-1",
            "start": 4.0,
            "end": 8.0,
            "text": "evidence under the null. This conclusion is complete.",
        },
        {
            "cue_id": "cue-2",
            "start": 8.0,
            "end": 12.0,
            "text": "with final anchor words.",
        },
    ]
    claim = "A p value means evidence under the null"
    plan = _compact_plan(
        exact_request="p value",
        constraints=[{
            "constraint_id": "subject",
            "kind": "subject",
            "source_phrase": "p value",
            "requirement": "Teach p value",
        }],
        evidence=[{
            "id": "subject",
            "q": claim,
        }],
    )
    plan = plan.model_copy(update={
        "topics": [plan.topics[0].model_copy(update={
            "start_line": 0,
            "end_line": 2,
            "start_quote": (
                "A p value means evidence under the null This conclusion is "
                "complete with final anchor words"
            ),
            "end_quote": "This conclusion is complete",
            "claim_quote": claim,
            "title": "P value meaning",
            "learning_objective": "Explain p value meaning",
            "facet": "p value",
        })],
    })

    report = gemini_segment._plan_to_report(
        plan,
        segments,
        [],
        {
            "_segment_ignore_caption_case": True,
            "_segment_model_boundary_authoritative": True,
        },
        topic="p value",
    )

    assert report.clips == []
    assert report.rejected_reasons == ["proposal_0:reversed_model_boundary"]


def test_authoritative_quotes_preserve_matching_boundary_punctuation() -> None:
    text = "“A p value means evidence under the null.”"
    plan = _compact_custom_plan(
        request="p value",
        start_quote="“A p value means",
        end_quote="under the null.”",
        claim_quote="A p value means evidence under the null",
    )

    report = gemini_segment._plan_to_report(
        plan,
        [{"cue_id": "cue-0", "start": 0.0, "end": 5.0, "text": text}],
        [],
        {
            "_segment_ignore_caption_case": True,
            "_segment_model_boundary_authoritative": True,
        },
        topic="p value",
    )

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["_clip_text"] == text
    assert clip["start_quote"] == "“A p value means"
    assert clip["end_quote"] == "under the null.”"


def test_compact_selector_rejects_instead_of_contracting_a_dirty_model_cut() -> None:
    text = (
        "Welcome back. Cells use chlorophyll to capture light energy and power the "
        "chemical reactions of photosynthesis."
    )
    plan = _compact_plan(
        exact_request="photosynthesis",
        constraints=[{
            "constraint_id": "subject",
            "kind": "subject",
            "source_phrase": "photosynthesis",
            "requirement": "Teach photosynthesis",
        }],
        evidence=[{
            "id": "subject",
            "q": "power the chemical reactions of photosynthesis",
        }],
    )
    plan = plan.model_copy(update={
        "topics": [plan.topics[0].model_copy(update={
            "start_quote": "Welcome back Cells use chlorophyll",
        })],
    })

    report = gemini_segment._plan_to_report(
        plan,
        [{"cue_id": "cue-0", "start": 0.0, "end": 12.0, "text": text}],
        [],
        {
            "_segment_ignore_caption_case": True,
            "_segment_model_boundary_authoritative": True,
        },
        topic="photosynthesis",
    )

    assert report.clips == []
    assert report.rejected_reasons == [
        "proposal_0:model_boundary_rewrite_forbidden"
    ]


def test_compact_selector_never_falls_back_from_an_ungrounded_model_quote() -> None:
    plan = _compact_plan(
        exact_request="photosynthesis",
        constraints=[{
            "constraint_id": "subject",
            "kind": "subject",
            "source_phrase": "photosynthesis",
            "requirement": "Teach photosynthesis",
        }],
        evidence=[{
            "id": "subject",
            "q": "power the chemical reactions of photosynthesis",
        }],
    )
    plan = plan.model_copy(update={
        "topics": [plan.topics[0].model_copy(update={
            "start_quote": "words that are not in the transcript",
        })],
    })

    report = gemini_segment._plan_to_report(
        plan,
        [{
            "cue_id": "cue-0",
            "start": 0.0,
            "end": 10.0,
            "text": (
                "Cells use chlorophyll to capture light energy and power the chemical "
                "reactions of photosynthesis."
            ),
        }],
        [],
        {
            "_segment_ignore_caption_case": True,
            "_segment_model_boundary_authoritative": True,
        },
        topic="photosynthesis",
    )

    assert report.clips == []
    assert report.rejected_reasons == [
        "proposal_0:ungrounded_model_start_quote"
    ]


def test_production_trusts_gemini_candidate_and_repairs_ungrounded_edge() -> None:
    plan = _compact_plan(
        exact_request="photosynthesis",
        constraints=[{
            "constraint_id": "subject",
            "kind": "subject",
            "source_phrase": "photosynthesis",
            "requirement": "Teach photosynthesis",
        }],
        evidence=[{
            "id": "subject",
            "q": "power the chemical reactions of photosynthesis",
        }],
    )
    plan = plan.model_copy(update={
        "topics": [plan.topics[0].model_copy(update={
            "start_quote": "words that are not in the transcript",
            "informativeness": 0.1,
            "topic_relevance": 0.1,
            "educational_importance": 0.1,
            "directly_teaches_topic": False,
            "substantive": False,
            "factually_grounded": False,
            "self_contained": False,
            "is_standalone": False,
        })],
    })

    report = gemini_segment._plan_to_report(
        plan,
        [{
            "cue_id": "cue-0",
            "start": 0.0,
            "end": 10.0,
            "text": (
                "Cells use chlorophyll to capture light energy and power the chemical "
                "reactions of photosynthesis."
            ),
        }],
        [],
        {
            "_segment_ignore_caption_case": True,
            "_segment_model_boundary_authoritative": True,
            "_segment_trust_gemini_semantics": True,
        },
        topic="photosynthesis",
    )

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["selection_authority"] == "gemini"
    assert clip["topic_relevance"] == 0.1
    assert clip["directly_teaches_topic"] is False
    assert clip["_clip_text"].startswith("Cells use chlorophyll")
    assert "bad_start_quote" in clip["_boundary_fallback_reasons"]


def test_production_preserves_every_schema_valid_gemini_candidate() -> None:
    text = (
        "A p value measures evidence against the null hypothesis. "
        "A small p value indicates stronger evidence against it."
    )
    first = _compact_custom_plan(
        request="p value",
        start_quote="A p value measures",
        end_quote="against the null hypothesis",
        claim_quote="A p value measures evidence against the null hypothesis",
    ).topics[0]
    second = first.model_copy(update={
        "start_line": 99,
        "end_line": 100,
        "start_quote": "missing model start edge",
        "end_quote": "missing model end edge",
        "claim_quote": "A small p value indicates stronger evidence against it",
        # Duplicate IDs are malformed selection metadata, not permission to lose a clip.
        "candidate_id": first.candidate_id,
    })
    third = second.model_copy(update={"candidate_id": "authoritative-model-cut-2"})
    plan = gemini_segment._CompactBoundaryPlan(
        request_intent=_compact_custom_plan(
            request="p value",
            start_quote="A p value measures",
            end_quote="against the null hypothesis",
            claim_quote="A p value measures evidence against the null hypothesis",
        ).request_intent,
        topics=[first, second, third],
    )

    report = gemini_segment._plan_to_report(
        plan,
        [{"cue_id": "cue-0", "start": 0.0, "end": 12.0, "text": text}],
        [],
        {
            "_segment_ignore_caption_case": True,
            "_segment_trust_gemini_semantics": True,
        },
        topic="p value",
    )

    assert report.proposed_count == report.accepted_count == 3
    assert report.rejected_reasons == []
    assert [clip["selection_candidate_id"] for clip in report.clips] == [
        "authoritative-model-cut",
        "authoritative-model-cut-2",
        "authoritative-model-cut-2-2",
    ]
    assert all(clip["selection_authority"] == "gemini" for clip in report.clips)


def test_production_labels_partial_gemini_intent_without_dropping_candidate() -> None:
    plan = _compact_custom_plan(
        request="derive x squared with the limit definition and finish at two x",
        start_quote="A derivative measures",
        end_quote="instantaneous change",
        claim_quote="A derivative measures instantaneous change in a function",
    )
    plan = plan.model_copy(update={
        "request_intent": gemini_segment._RequestIntent.model_validate({
            "exact_request": plan.request_intent.exact_request,
            "constraints": [
                {
                    "constraint_id": "subject",
                    "kind": "subject",
                    "source_phrase": "x squared",
                    "requirement": "Use x squared",
                },
                {
                    "constraint_id": "task",
                    "kind": "task",
                    "source_phrase": "limit definition",
                    "requirement": "Use the limit definition",
                },
                {
                    "constraint_id": "outcome",
                    "kind": "outcome",
                    "source_phrase": "two x",
                    "requirement": "Finish at two x",
                },
            ],
        }),
        "topics": [plan.topics[0].model_copy(update={
            "intent_evidence": [gemini_segment._CompactIntentEvidence(
                id="task",
                q="A derivative measures instantaneous change in a function",
            )],
        })],
    })

    report = gemini_segment._plan_to_report(
        plan,
        [{
            "cue_id": "cue-0",
            "start": 0.0,
            "end": 8.0,
            "text": "A derivative measures instantaneous change in a function.",
        }],
        [],
        {"_segment_trust_gemini_semantics": True},
        topic=plan.request_intent.exact_request,
    )

    assert report.accepted_count == report.proposed_count == 1
    [clip] = report.clips
    assert clip["intent_role"] == "supporting"
    assert clip["intent_coverage"] == pytest.approx(1 / 3, abs=1e-6)


def test_production_keeps_direct_false_topic_supporting_at_full_coverage() -> None:
    plan = _compact_custom_plan(
        request="the role of h in the limit definition",
        start_quote="The symbol h represents",
        end_quote="approaches zero",
        claim_quote="The symbol h represents the input change that approaches zero",
    )
    plan = plan.model_copy(update={
        "topics": [plan.topics[0].model_copy(update={
            "directly_teaches_topic": False,
        })],
    })
    text = "The symbol h represents the input change that approaches zero."

    report = gemini_segment._plan_to_report(
        plan,
        [{"cue_id": "cue-0", "start": 0.0, "end": 8.0, "text": text}],
        [],
        {"_segment_trust_gemini_semantics": True},
        topic=plan.request_intent.exact_request,
    )

    assert report.accepted_count == report.proposed_count == 1
    [clip] = report.clips
    assert clip["intent_role"] == "supporting"
    assert clip["intent_coverage"] == pytest.approx(1.0)


def test_production_passes_primary_and_multiple_supporting_units_from_one_source() -> None:
    request = "derive x squared using the limit definition to two x"
    constraints = [
        {
            "constraint_id": "object",
            "kind": "subject",
            "source_phrase": "x squared",
            "requirement": "Use x squared",
        },
        {
            "constraint_id": "method",
            "kind": "task",
            "source_phrase": "limit definition",
            "requirement": "Use the limit definition",
        },
        {
            "constraint_id": "result",
            "kind": "outcome",
            "source_phrase": "two x",
            "requirement": "Reach two x",
        },
    ]
    segment_texts = [
        "For five x minus four, use the limit definition and the answer is five.",
        "For x squared, use the limit definition and the final derivative is two x.",
        (
            "For one over x, use the limit definition and the result is negative one "
            "over x squared."
        ),
    ]
    topics = []
    specifications = [
        (
            "linear-support",
            0,
            "For five x minus four",
            "the answer is five",
            "use the limit definition and the answer",
            False,
            [{"id": "method", "q": "use the limit definition and the answer"}],
        ),
        (
            "x-squared-primary",
            1,
            "For x squared",
            "derivative is two x",
            "the final derivative is two x",
            True,
            [
                {"id": "object", "q": "For x squared, use the limit definition"},
                {"id": "method", "q": "use the limit definition and the final"},
                {"id": "result", "q": "the final derivative is two x"},
            ],
        ),
        (
            "reciprocal-support",
            2,
            "For one over x",
            "negative one over x squared",
            "use the limit definition and the result",
            False,
            [{"id": "method", "q": "use the limit definition and the result"}],
        ),
    ]
    for candidate_id, line, start_quote, end_quote, claim, direct, evidence in specifications:
        topics.append(gemini_segment._CompactBoundaryTopic.model_validate({
            "id": candidate_id,
            "s": line,
            "e": line,
            "sq": start_quote,
            "eq": end_quote,
            "cq": claim,
            "title": candidate_id,
            "obj": candidate_id,
            "facet": candidate_id,
            "info": 0.95,
            "rel": 0.95,
            "imp": 0.95,
            "diff": 0.25,
            "direct": direct,
            "sub": True,
            "fact": True,
            "self": True,
            "stand": True,
            "ie": evidence,
        }))
    plan = gemini_segment._CompactBoundaryPlan.model_validate({
        "request_intent": {
            "exact_request": request,
            "constraints": constraints,
        },
        "topics": topics,
    })
    segments = [
        {
            "cue_id": f"cue-{index}",
            "start": index * 10.0,
            "end": (index + 1) * 10.0,
            "text": text,
        }
        for index, text in enumerate(segment_texts)
    ]

    report = gemini_segment._plan_to_report(
        plan,
        segments,
        [],
        {"_segment_trust_gemini_semantics": True},
        topic=request,
    )

    assert report.accepted_count == report.proposed_count == 3
    assert report.rejected_reasons == []
    assert [clip["intent_role"] for clip in report.clips] == [
        "supporting",
        "primary",
        "supporting",
    ]
    assert [clip["intent_coverage"] for clip in report.clips] == pytest.approx([
        1 / 3,
        1.0,
        1 / 3,
    ])


def test_production_unanchored_gemini_intent_is_metadata_only() -> None:
    plan = _compact_custom_plan(
        request="derive x squared with the limit definition",
        start_quote="A derivative measures",
        end_quote="instantaneous change",
        claim_quote="A derivative measures instantaneous change",
    )
    plan = plan.model_copy(update={
        "request_intent": gemini_segment._RequestIntent.model_validate({
            "exact_request": plan.request_intent.exact_request,
            "constraints": [
                {
                    "constraint_id": "subject",
                    "kind": "subject",
                    "source_phrase": "x squared",
                    "requirement": "Use x squared",
                },
                {
                    "constraint_id": "task",
                    "kind": "task",
                    "source_phrase": "limit definition",
                    "requirement": "Use the limit definition",
                },
            ],
        }),
        "topics": [plan.topics[0].model_copy(update={
            "intent_evidence": [
                gemini_segment._CompactIntentEvidence(
                    id="subject", q="x squared appears only in invented evidence",
                ),
                gemini_segment._CompactIntentEvidence(
                    id="task", q="the missing limit definition is also invented",
                ),
            ],
        })],
    })

    report = gemini_segment._plan_to_report(
        plan,
        [{
            "cue_id": "cue-0",
            "start": 0.0,
            "end": 8.0,
            "text": "A derivative measures instantaneous change.",
        }],
        [],
        {"_segment_trust_gemini_semantics": True},
        topic=plan.request_intent.exact_request,
    )

    assert report.accepted_count == report.proposed_count == 1
    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["intent_role"] == "supporting"
    assert clip["intent_coverage"] == 0.0
    assert len(clip["intent_evidence"]) == 2


def test_production_scope_only_intent_does_not_require_spoken_scope_evidence() -> None:
    plan = _compact_custom_plan(
        request="AP Statistics",
        start_quote="A p value measures",
        end_quote="under the null hypothesis",
        claim_quote="A p value measures extremeness under the null hypothesis",
    )
    plan = plan.model_copy(update={
        "request_intent": gemini_segment._RequestIntent.model_validate({
            "exact_request": plan.request_intent.exact_request,
            "constraints": [{
                "constraint_id": "course",
                "kind": "scope",
                "source_phrase": "AP Statistics",
                "requirement": "Use AP Statistics scope",
            }],
        }),
        "topics": [plan.topics[0].model_copy(update={
            "intent_evidence": [gemini_segment._CompactIntentEvidence(
                id="course", q="this course name is not spoken",
            )],
        })],
    })

    report = gemini_segment._plan_to_report(
        plan,
        [{
            "cue_id": "cue-0",
            "start": 0.0,
            "end": 8.0,
            "text": "A p value measures extremeness under the null hypothesis.",
        }],
        [],
        {"_segment_trust_gemini_semantics": True},
        topic=plan.request_intent.exact_request,
    )

    assert report.accepted_count == report.proposed_count == 1
    [clip] = report.clips
    assert clip["intent_role"] == "primary"
    assert clip["intent_coverage"] == 1.0


def test_production_keeps_grounded_intent_quote_inside_repaired_range() -> None:
    plan = _compact_custom_plan(
        request="derive x squared",
        start_quote="The derivative is two x",
        end_quote="derivative is two x",
        claim_quote="The derivative is two x",
    )
    plan = plan.model_copy(update={
        "topics": [plan.topics[0].model_copy(update={
            "start_line": 1,
            "end_line": 1,
            "intent_evidence": [gemini_segment._CompactIntentEvidence(
                id="subject", q="Let f of x equal x squared",
            )],
        })],
    })
    segments = [
        {
            "cue_id": "cue-0",
            "start": 0.0,
            "end": 5.0,
            "text": "Let f of x equal x squared.",
        },
        {
            "cue_id": "cue-1",
            "start": 5.0,
            "end": 10.0,
            "text": "The derivative is two x.",
        },
    ]

    report = gemini_segment._plan_to_report(
        plan,
        segments,
        [],
        {"_segment_trust_gemini_semantics": True},
        topic=plan.request_intent.exact_request,
    )

    assert report.accepted_count == report.proposed_count == 1
    [clip] = report.clips
    assert clip["cue_ids"] == ["cue-0", "cue-1"]
    assert clip["intent_role"] == "primary"
    assert clip["intent_coverage"] == 1.0
    assert "start_expanded_to_intent_evidence" in (
        clip["_boundary_fallback_reasons"]
    )


def test_production_advances_generic_intro_only_to_gemini_intent_evidence() -> None:
    segments = [
        {
            "cue_id": "cue-0",
            "start": 0.0,
            "end": 20.0,
            "text": (
                "In this video, we're going to derive functions with limits."
            ),
        },
        {
            "cue_id": "cue-1",
            "start": 20.0,
            "end": 60.0,
            "text": (
                "First, for five x minus four, the limit definition gives five."
            ),
        },
        {
            "cue_id": "cue-2",
            "start": 60.0,
            "end": 105.0,
            "text": (
                "The derivative of five x minus four is five. Now let f of x equal "
                "x squared. Substitute x plus h into the limit definition."
            ),
        },
        {
            "cue_id": "cue-3",
            "start": 105.0,
            "end": 150.0,
            "text": (
                "Expand, cancel x squared, divide by h, and set h to zero. The "
                "derivative of x squared is two x."
            ),
        },
    ]
    plan = _compact_custom_plan(
        request="derive x squared with the limit definition",
        start_quote="In this video, we're going to",
        end_quote="derivative of x squared is two x",
        claim_quote="derivative of x squared is two x",
    )
    plan = plan.model_copy(update={
        "topics": [plan.topics[0].model_copy(update={
            "start_line": 0,
            "end_line": 3,
            "intent_evidence": [gemini_segment._CompactIntentEvidence(
                id="subject", q="Now let f of x equal x squared",
            )],
        })],
    })

    report = gemini_segment._plan_to_report(
        plan,
        segments,
        [],
        {"_segment_trust_gemini_semantics": True},
        topic=plan.request_intent.exact_request,
    )

    assert report.accepted_count == report.proposed_count == 1
    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["cue_ids"] == ["cue-2", "cue-3"]
    assert clip["_clip_text"].startswith("Now let f of x equal x squared")
    assert "five x minus four" not in clip["_clip_text"]
    assert clip["edge_projection"]["start"]["quote"] == (
        "Now let f of x equal x squared"
    )
    assert "generic_start_advanced_to_intent_evidence" in (
        clip["_boundary_fallback_reasons"]
    )


@pytest.mark.parametrize(
    ("text", "claim_quote", "should_advance"),
    [
        (
            "This is simply equal to 5. So that's the derivative of five x "
            "minus four. It's five. Now let's try another example. So let's "
            "say if f ofx is equal to x^2, the derivative is two x.",
            "the derivative is two x",
            True,
        ),
        (
            "This is simply equal to 5. The lesson continues. So let's say if "
            "f ofx is equal to x^2, the derivative is two x.",
            "the derivative is two x",
            False,
        ),
        (
            "This is simply equal to 5. So let's say if f ofx is equal to "
            "x^2. Now let's try another example. The derivative is two x.",
            "The derivative is two x",
            False,
        ),
        (
            "This is simply equal to 5. Now let's try another example. So "
            "let's say if f ofx is equal to x^2, the derivative is two x.",
            "This is simply equal to 5",
            False,
        ),
    ],
    ids=(
        "live-explicit-handoff",
        "no-explicit-handoff",
        "handoff-after-evidence",
        "claim-before-evidence",
    ),
)
def test_production_new_example_handoff_licenses_only_grounded_start_repair(
    text: str,
    claim_quote: str,
    should_advance: bool,
) -> None:
    evidence_quote = "if f ofx is equal to x^2"
    plan = _compact_custom_plan(
        request="derive x squared",
        start_quote="This is simply equal to 5",
        end_quote="derivative is two x",
        claim_quote=claim_quote,
    )
    plan = plan.model_copy(update={
        "topics": [plan.topics[0].model_copy(update={
            "intent_evidence": [gemini_segment._CompactIntentEvidence(
                id="subject", q=evidence_quote,
            )],
        })],
    })

    report = gemini_segment._plan_to_report(
        plan,
        [{"cue_id": "cue-0", "start": 0.0, "end": 30.0, "text": text}],
        [],
        {"_segment_trust_gemini_semantics": True},
        topic=plan.request_intent.exact_request,
    )

    assert report.accepted_count == report.proposed_count == 1
    assert report.rejected_reasons == []
    [clip] = report.clips
    repaired = "new_example_start_advanced_to_intent_evidence" in (
        clip["_boundary_fallback_reasons"]
    )
    assert repaired is should_advance
    if should_advance:
        assert evidence_quote in clip["_clip_text"]
        assert "five x minus four" not in clip["_clip_text"]
        projected_start = clip["edge_projection"]["start"]["quote"]
        assert (
            projected_start == evidence_quote
            or projected_start.startswith("Now let's try another example")
        )
    else:
        assert clip["_clip_text"].startswith("This is simply equal to 5")


def test_production_expands_anaphoric_start_without_semantic_rejection() -> None:
    segments = [
        {
            "cue_id": "cue-0",
            "start": 0.0,
            "end": 4.0,
            "text": "The null hypothesis says there is no effect.",
        },
        {
            "cue_id": "cue-1",
            "start": 4.0,
            "end": 8.0,
            "text": "This means a small p value gives evidence against it and",
        },
        {
            "cue_id": "cue-2",
            "start": 8.0,
            "end": 12.0,
            "text": "supports rejecting the null hypothesis.",
        },
    ]
    claim = (
        "small p value gives evidence against it and supports rejecting"
    )
    plan = _compact_custom_plan(
        request="p value",
        start_quote="This means a small p value",
        end_quote="evidence against it and",
        claim_quote=claim,
    )
    plan = plan.model_copy(update={
        "topics": [plan.topics[0].model_copy(update={
            "start_line": 1,
            "end_line": 1,
        })],
    })

    report = gemini_segment._plan_to_report(
        plan,
        segments,
        [],
        {
            "_segment_ignore_caption_case": True,
            "_segment_trust_gemini_semantics": True,
        },
        topic="p value",
    )

    assert report.accepted_count == report.proposed_count == 1
    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["cue_ids"] == ["cue-0", "cue-1", "cue-2"]
    assert clip["_clip_text"] == " ".join(
        segment["text"] for segment in segments
    )
    assert "end_expanded_to_claim" in clip["_boundary_fallback_reasons"]
    assert "expanded_start_context" in clip["_boundary_fallback_reasons"]


def test_production_preserves_restated_worked_example_start_inside_coarse_cue() -> None:
    segments = [
        {
            "cue_id": "cue-0",
            "start": 0.0,
            "end": 45.0,
            "text": (
                "First derive the linear function five x minus four with the limit "
                "definition."
            ),
        },
        {
            "cue_id": "cue-1",
            "start": 45.0,
            "end": 100.0,
            "text": (
                "The h terms cancel, so the derivative of five x minus four is five."
            ),
        },
        {
            "cue_id": "cue-2",
            "start": 100.0,
            "end": 155.0,
            "text": (
                "Now let's try another example. So let's say if f of x is equal to x "
                "squared, what is the derivative? Use the limit definition and replace "
                "x with x plus h."
            ),
        },
        {
            "cue_id": "cue-3",
            "start": 155.0,
            "end": 205.0,
            "text": (
                "Expanding gives x squared plus two x h plus h squared. Cancel x "
                "squared, factor h, and substitute zero. The derivative of x squared "
                "is two x."
            ),
        },
    ]
    plan = gemini_segment._CompactBoundaryPlan(
        request_intent={
            "exact_request": (
                "Use the limit definition to derive the derivative of x squared as two x"
            ),
            "constraints": [
                {
                    "constraint_id": "object",
                    "kind": "subject",
                    "source_phrase": "x squared",
                    "requirement": "Use f of x equal to x squared",
                },
                {
                    "constraint_id": "task",
                    "kind": "task",
                    "source_phrase": "limit definition",
                    "requirement": "Derive with the limit definition",
                },
                {
                    "constraint_id": "outcome",
                    "kind": "outcome",
                    "source_phrase": "two x",
                    "requirement": "Reach the final result two x",
                },
            ],
        },
        topics=[gemini_segment._CompactBoundaryTopic(
            candidate_id="x-squared-limit-derivation",
            start_line=2,
            end_line=3,
            start_quote="So let's say if f of x is equal to x squared",
            end_quote="The derivative of x squared is two x",
            claim_quote="The derivative of x squared is two x",
            title="Derive the derivative of x squared",
            learning_objective=(
                "Derive the derivative of x squared with the limit definition"
            ),
            facet="x squared limit derivation",
            informativeness=0.99,
            topic_relevance=0.99,
            educational_importance=0.99,
            difficulty=0.4,
            directly_teaches_topic=True,
            substantive=True,
            factually_grounded=True,
            self_contained=True,
            is_standalone=True,
            intent_evidence=[
                {"id": "object", "q": "f of x is equal to x squared"},
                {"id": "task", "q": "Use the limit definition and replace x"},
                {"id": "outcome", "q": "The derivative of x squared is two x"},
            ],
        )],
    )

    report = gemini_segment._plan_to_report(
        plan,
        segments,
        [],
        {
            "_segment_ignore_caption_case": True,
            "_segment_trust_gemini_semantics": True,
        },
        topic=plan.request_intent.exact_request,
    )

    assert report.accepted_count == report.proposed_count == 1
    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["cue_ids"] == ["cue-2", "cue-3"]
    assert clip["_clip_text"].startswith(
        "So let's say if f of x is equal to x squared"
    )
    assert "five x minus four" not in clip["_clip_text"]
    assert clip["edge_projection"]["start"]["quote"] == (
        "So let's say if f of x is equal to x squared"
    )


def test_production_same_cue_repair_never_excludes_gemini_claim() -> None:
    text = (
        "Plants use sunlight to make sugar during photosynthesis. "
        "Respiration releases energy from sugar."
    )
    claim = "Plants use sunlight to make sugar during photosynthesis"
    plan = _compact_custom_plan(
        request="photosynthesis",
        start_quote="Respiration releases energy",
        end_quote="energy from sugar",
        claim_quote=claim,
    )

    report = gemini_segment._plan_to_report(
        plan,
        [{"cue_id": "cue-0", "start": 0.0, "end": 10.0, "text": text}],
        [],
        {
            "_segment_ignore_caption_case": True,
            "_segment_trust_gemini_semantics": True,
        },
        topic="photosynthesis",
    )

    assert report.accepted_count == report.proposed_count == 1
    [clip] = report.clips
    assert claim in clip["_clip_text"]
    assert clip["topic_evidence_quote"] == claim
    assert "start_expanded_to_claim" in clip["_boundary_fallback_reasons"]


def test_production_preserves_gemini_fragment_projection_without_rejection() -> None:
    text = (
        "Welcome back. and its enclosing scope makes the definition precise."
    )
    claim = "its enclosing scope makes the definition precise"
    plan = _compact_custom_plan(
        request="scope",
        start_quote="and its enclosing scope",
        end_quote="makes the definition precise",
        claim_quote=claim,
    )

    report = gemini_segment._plan_to_report(
        plan,
        [{"cue_id": "cue-0", "start": 0.0, "end": 8.0, "text": text}],
        [],
        {
            "_segment_ignore_caption_case": True,
            "_segment_trust_gemini_semantics": True,
        },
        topic="scope",
    )

    assert report.accepted_count == report.proposed_count == 1
    [clip] = report.clips
    assert clip["_clip_text"] == (
        "and its enclosing scope makes the definition precise"
    )
    assert clip["edge_projection"]["start"]["quote"] == (
        "and its enclosing scope"
    )
    assert "expanded_incomplete_start" not in clip["_boundary_fallback_reasons"]


def test_trusted_candidate_expands_ordinal_opening_to_spoken_antecedent() -> None:
    segments = [
        {
            "cue_id": "cue-0",
            "start": 0.0,
            "end": 6.0,
            "text": (
                "Newton's second law says net force equals mass times acceleration."
            ),
        },
        {
            "cue_id": "cue-1",
            "start": 6.0,
            "end": 12.0,
            "text": "For a fixed mass, more net force produces more acceleration.",
        },
        {
            "cue_id": "cue-2",
            "start": 12.0,
            "end": 19.0,
            "text": (
                "The second law can be rephrased to state that acceleration is "
                "proportional to net force."
            ),
        },
    ]
    claim = "acceleration is proportional to net force"
    plan = _compact_custom_plan(
        request="Newton's second law",
        start_quote="The second law can be rephrased",
        end_quote="proportional to net force",
        claim_quote=claim,
    )
    plan = plan.model_copy(update={
        "topics": [plan.topics[0].model_copy(update={
            "start_line": 2,
            "end_line": 2,
            "intent_evidence": [gemini_segment._CompactIntentEvidence(
                id="subject", q=claim,
            )],
        })],
    })

    report = gemini_segment._plan_to_report(
        plan,
        segments,
        [],
        {"_segment_trust_gemini_semantics": True},
        topic=plan.request_intent.exact_request,
    )

    assert report.accepted_count == report.proposed_count == 1
    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["cue_ids"] == ["cue-0", "cue-1", "cue-2"]
    assert clip["_clip_text"].startswith("Newton's second law")
    assert "expanded_start_context" in clip["_boundary_fallback_reasons"]


def test_trusted_candidate_expands_unsafe_mid_cue_definition_start() -> None:
    segments = [
        {
            "cue_id": "cue-0",
            "start": 0.0,
            "end": 8.0,
            "text": (
                "Newton's second law connects net force, mass, and acceleration."
            ),
        },
        {
            "cue_id": "cue-1",
            "start": 8.0,
            "end": 18.0,
            "text": (
                "With that knowledge in hand, you are now ready to understand "
                "acceleration which is simply the rate at which velocity changes."
            ),
        },
    ]
    claim = "acceleration which is simply the rate at which velocity changes"
    plan = _compact_custom_plan(
        request="acceleration",
        start_quote="acceleration which is simply the rate",
        end_quote="rate at which velocity changes",
        claim_quote=claim,
    )
    plan = plan.model_copy(update={
        "topics": [plan.topics[0].model_copy(update={
            "start_line": 1,
            "end_line": 1,
        })],
    })

    report = gemini_segment._plan_to_report(
        plan,
        segments,
        [],
        {"_segment_trust_gemini_semantics": True},
        topic=plan.request_intent.exact_request,
    )

    assert report.accepted_count == report.proposed_count == 1
    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["cue_ids"] == ["cue-0", "cue-1"]
    assert clip["_clip_text"].startswith("Newton's second law")
    assert "expanded_start_context" in clip["_boundary_fallback_reasons"]


def test_trusted_live_proportionality_predicate_expands_to_spoken_equation() -> None:
    raw_cues = [
        (12, 37.559, 41.94, "equation, F = ma. What it means is that we can do"),
        (13, 41.94, 44.28, "quantitative calculations relating the"),
        (14, 44.28, 46.469, "magnitude of a force applied to an"),
        (15, 46.469, 49.32, "object, the mass of the object, and the"),
        (16, 49.32, 51.36, "magnitude of the acceleration that"),
        (17, 51.36, 54.3, "object will experience, and it shows the"),
        (18, 54.3, 57.27, "derivation of the Newton as the SI unit"),
        (19, 57.27, 60.629, "of force when we plug in 1 kilogram and"),
        (20, 60.629, 63.18, "one meter per second squared for mass"),
        (21, 63.18, 66.33, "and acceleration. There are a number of"),
        (22, 66.33, 68.25, "things we can say about this equation,"),
        (23, 68.25, 71.58, "which is tiny but powerful. First it"),
        (24, 71.58, 74.13, "means that heavier objects will require"),
        (25, 74.13, 76.92, "the application of greater force in"),
        (26, 76.92, 78.93, "order to achieve the same acceleration"),
        (27, 78.93, 82.409, "as lighter objects, with acceleration being"),
        (28, 82.409, 85.74, "equal to force divided by mass. If we"),
        (29, 85.74, 88.14, "want these objects of varying masses"),
        (30, 88.14, 90.42, "each to accelerate at one meter per"),
        (31, 90.42, 91.59, "second squared,"),
        (32, 91.59, 94.259, "these are the magnitudes of the forces"),
        (33, 94.259, 96.93, "that must be applied in order for the"),
        (34, 96.93, 100.17, "math to work out. This also means that"),
        (35, 100.17, 102.99, "the second law can be rephrased to state"),
        (36, 102.99, 104.88, "that the acceleration an object"),
        (37, 104.88, 107.189, "experiences will be directly"),
        (38, 107.189, 110.25, "proportional to the force applied and"),
        (39, 110.25, 114.149, "inversely proportional to its mass. It is"),
    ]
    segments = [
        {
            "cue_id": f"xzA6IBWUEDE:cue:{cue}",
            "start": start,
            "end": end,
            "text": text,
        }
        for cue, start, end, text in raw_cues
    ]
    claim = "acceleration an object experiences will be directly proportional"
    plan = _compact_custom_plan(
        request="Newton's second law",
        start_quote="means that heavier objects will require",
        end_quote="inversely proportional to its mass.",
        claim_quote=claim,
    )
    plan = plan.model_copy(update={
        "topics": [plan.topics[0].model_copy(update={
            "candidate_id": "second-law-proportionality-intuition",
            "start_line": 12,
            "end_line": 27,
            "title": "Intuition Behind Newton's Second Law",
            "learning_objective": (
                "Understand how force, mass, and acceleration relate proportionally"
            ),
            "facet": "proportionality and intuition",
        })],
    })

    report = gemini_segment._plan_to_report(
        plan,
        segments,
        [],
        {"_segment_trust_gemini_semantics": True},
        topic=plan.request_intent.exact_request,
    )

    assert report.accepted_count == report.proposed_count == 1
    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["selection_candidate_id"] == (
        "second-law-proportionality-intuition"
    )
    assert clip["start_cue_id"] == "xzA6IBWUEDE:cue:12"
    assert clip["start_quote"] == "F = ma. What it means is"
    assert clip["_clip_text"].startswith("F = ma. What it means is")
    assert "means that heavier objects will require" in clip["_clip_text"]
    assert claim in clip["_clip_text"]
    assert "expanded_subjectless_predicate_context" in (
        clip["_boundary_fallback_reasons"]
    )


def test_trusted_predicate_context_never_crosses_topic_reset_for_old_equation() -> None:
    segments = [
        {
            "cue_id": "cue-old-equation",
            "start": 0.0,
            "end": 4.0,
            "text": "For the old calculation, x = 5.",
        },
        {
            "cue_id": "cue-topic-reset",
            "start": 4.0,
            "end": 8.0,
            "text": (
                "Now let us discuss photosynthesis. Chlorophyll absorbs light and"
            ),
        },
        {
            "cue_id": "cue-model-start",
            "start": 8.0,
            "end": 13.0,
            "text": "means that plants can store energy as chemical energy",
        },
        {
            "cue_id": "cue-explanation-end",
            "start": 13.0,
            "end": 18.0,
            "text": "using chlorophyll during the light-dependent reactions.",
        },
    ]
    claim = "plants can store energy as chemical energy"
    plan = _compact_custom_plan(
        request="photosynthesis",
        start_quote="means that plants can store energy",
        end_quote="during the light-dependent reactions.",
        claim_quote=claim,
    )
    plan = plan.model_copy(update={
        "topics": [plan.topics[0].model_copy(update={
            "start_line": 2,
            "end_line": 3,
            "title": "Photosynthesis",
            "learning_objective": "Explain how plants store light energy",
            "facet": "light-dependent reactions",
        })],
    })

    report = gemini_segment._plan_to_report(
        plan,
        segments,
        [],
        {"_segment_trust_gemini_semantics": True},
        topic=plan.request_intent.exact_request,
    )

    assert report.accepted_count == report.proposed_count == 1
    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["start_cue_id"] == "cue-topic-reset"
    assert clip["_clip_text"].startswith(
        "Now let us discuss photosynthesis. Chlorophyll absorbs light and"
    )
    assert "means that plants can store energy" in clip["_clip_text"]
    assert "x = 5" not in clip["_clip_text"]
    assert "expanded_subjectless_predicate_context" not in (
        clip["_boundary_fallback_reasons"]
    )


def test_trusted_live_net_force_complement_recovers_split_copula() -> None:
    raw_cues = [
        (39, 110.25, 114.149, "inversely proportional to its mass. It is"),
        (40, 114.149, 116.67, "important to note that the net force is"),
        (41, 116.67, 119.28, "the sum of all the forces acting on an"),
        (42, 119.28, 122.28, "object. If multiple forces are acting on"),
        (43, 122.28, 124.68, "an object, which is often the case, we"),
        (44, 124.68, 126.63, "will need to represent them all in a"),
        (45, 126.63, 129.569, "free body diagram and then add up all"),
        (46, 129.569, 132.81, "the vectors to find the net force, which"),
        (47, 132.81, 133.49, "will tell us"),
        (48, 133.49, 135.71, "the direction of the acceleration that"),
        (49, 135.71, 138.23, "will occur in response to the net force."),
    ]
    segments = [
        {
            "cue_id": f"xzA6IBWUEDE:cue:{cue}",
            "start": start,
            "end": end,
            "text": text,
        }
        for cue, start, end, text in raw_cues
    ]
    claim = "the net force is the sum of all the forces"
    plan = _compact_custom_plan(
        request="Newton's second law",
        start_quote="important to note that the net",
        end_quote="in response to the net force.",
        claim_quote=claim,
    )
    plan = plan.model_copy(update={
        "topics": [plan.topics[0].model_copy(update={
            "candidate_id": "net-force-and-vector-addition",
            "start_line": 1,
            "end_line": 10,
            "title": "Calculating Net Force",
            "learning_objective": "Explain how to find net force using vector addition",
            "facet": "net force definition",
        })],
    })

    report = gemini_segment._plan_to_report(
        plan,
        segments,
        [],
        {"_segment_trust_gemini_semantics": True},
        topic=plan.request_intent.exact_request,
    )

    assert report.accepted_count == report.proposed_count == 1
    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["selection_candidate_id"] == "net-force-and-vector-addition"
    assert clip["start_cue_id"] == "xzA6IBWUEDE:cue:39"
    assert clip["start_quote"] == "It is"
    assert clip["_clip_text"].startswith("It is important to note")
    assert "inversely proportional to its mass" not in clip["_clip_text"]
    assert claim in clip["_clip_text"]
    assert "expanded_split_copula_context" in (
        clip["_boundary_fallback_reasons"]
    )


def test_trusted_live_weight_candidate_advances_to_its_complete_setup() -> None:
    raw_cues = [
        (
            3,
            98.84,
            149.84,
            "force. Newton's first law of motion is also known as the law of "
            "inertia. Now you might be wondering well what exactly is inertia? "
            "Inertia is the tendency of an object to maintain its state of rest or "
            "uniform motion. Inertia is related to mass. As the mass of an object "
            "increases, the inertia increases as well. So let's say if we have a "
            "100 kg object and 1,00 kg object, which one has more inertia? So which "
            "object wants to maintain its state of rest? If we apply a force to the "
            "smaller object, it's",
        ),
        (
            4,
            147.36,
            181.04,
            "going to be relatively easy to move. We can easily move it from a "
            "state of rest. Now, the larger object, it's going to be difficult to "
            "get it going. It requires a greater force to move it. So, it's harder "
            "to move it from a state of rest. So, the the more massive object has "
            "more inertia. And so that's the main idea behind inertia, which is the "
            "tendency of an object to resist or maintain its state of rest. It's "
            "going to resist any changes in motion that you try to apply to it.",
        ),
        (5, 179.599, 186.08, "Now the next law that you need to be familiar with is Newton's second"),
        (
            6,
            188.519,
            246.56,
            "law. Newton's second law is basically this equation. Force is equal "
            "to mass time acceleration. And this of course is the net force. So "
            "let's say if you have a mass of 5 kg and the acceleration on it is 2 "
            "m/s squared. The force is 10 newtons and this is the net force. A "
            "newton is equivalent to 1 kilogram times a meter over second squ. It "
            "turns out that the newton is not the only unit for force. Force can be "
            "measured in pounds. It turns out one pound is approximately 4.45 newtons.",
        ),
        (
            7,
            244.879,
            291.6,
            "So now you know how to convert from newtons to pounds. So let's say if "
            "you have about 100 newtons, how many pounds of force does that "
            "represent? So to convert it, let's use the conversion factor that I "
            "just gave you. So we said that there's 4.45 newtons per pound. So you "
            "want to set it up in such a way that the unit newtons cancel. So to "
            "convert newtons into pounds, simply divide the force by 4.45 and you "
            "should get 22.5 lb",
        ),
        (
            8,
            296.36,
            304.24,
            "approximately. Now what is the weight force? What's the difference "
            "between mass and weight?",
        ),
        (
            9,
            306.12,
            351.039,
            "Mass is measured in units of kilograms. So in physics whenever you see "
            "kilograms it's associated with mass. Weight is different from mass. "
            "Weight is a force. And so weight is measured in newtons sometimes "
            "pounds. Weight is equal to mass time gravitational acceleration. the "
            "same way as force is equal to mass time acceleration. As you can see, "
            "these two are the same. When you see G, G is a type of acceleration "
            "and weight is a type of force. Weight is simply the force of",
        ),
        (
            10,
            348.0,
            399.88,
            "gravity that is acting on you. And so weight is a downward force. "
            "Consider this 20 kg object. What is the weight force? acting on it. Go "
            "ahead and calculate it. So the weight force is always equal to mg mass "
            "time gravitational acceleration. And on Earth the gravitational "
            "acceleration is 9.8 m/s squared. So this is going to be about 200. You "
            "could round that to 10 if you want, but let's use the exact value. So "
            "20 * 9.8 is about 196 newtons. So that's the weight force.",
        ),
    ]
    segments = [
        {
            "cue_id": f"pL2YfC-22Uc:cue:{cue}",
            "start": start,
            "end": end,
            "text": text,
        }
        for cue, start, end, text in raw_cues
    ]
    claim = (
        "Weight is equal to mass time gravitational acceleration. the same way "
        "as force is equal to mass time acceleration."
    )
    plan = _compact_custom_plan(
        request="Newton's second law",
        start_quote="going to be relatively easy to",
        end_quote="So 20 * 9.8 is about 196 newtons. So that's the weight force.",
        claim_quote=claim,
    )
    plan = plan.model_copy(update={
        "topics": [plan.topics[0].model_copy(update={
            "candidate_id": "weight-force-calculation",
            "start_line": 1,
            "end_line": 7,
            "title": "Calculating Weight Force",
            "learning_objective": (
                "Calculate the weight force of an object using mass and "
                "gravitational acceleration."
            ),
            "facet": "Weight vs mass",
        })],
    })

    report = gemini_segment._plan_to_report(
        plan,
        segments,
        [],
        {"_segment_trust_gemini_semantics": True},
        topic=plan.request_intent.exact_request,
    )

    assert report.accepted_count == report.proposed_count == 1
    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["start_cue_id"] == "pL2YfC-22Uc:cue:8"
    assert clip["start_quote"] == "Now what is the weight force"
    assert clip["_clip_text"].startswith("Now what is the weight force?")
    assert "going to be relatively easy" not in clip["_clip_text"]
    assert "trimmed_clipped_start_to_claim_setup" in (
        clip["_boundary_fallback_reasons"]
    )


def test_trusted_live_acceleration_candidate_advances_to_its_mass_setup() -> None:
    raw_cues = [
        (
            14,
            513.279,
            527.839,
            "Newton's third law of motion states that for every action force there "
            "is an equal but opposite reaction force. So let's say if you have two people",
        ),
        (
            15,
            530.36,
            573.04,
            "skating I'm just going to draw stick figures and one person has more "
            "mass than the other person. So if the smaller person applies a force "
            "of 100 newtons on the larger person, what force will the larger person "
            "apply on the smaller person? According to Newton's third law, for "
            "every action force, there is an equal but opposite reaction force. So "
            "the other person is going to apply a force of 100 newtons. They have "
            "to be equal, but they're opposite in direction. So, the forces are always the",
        ),
        (
            16,
            570.92,
            614.72,
            "same. You might see a question like this on a test. Just remember the "
            "forces are equal and opposite in direction. Now, let's say if the mass "
            "of the smaller person is 50 kg and the mass of the larger person is "
            "100 kg, who experiences the greater acceleration? and who's going to "
            "move further. Let's say if they're on ice. So, as each of these "
            "individuals, as they push against each other, they're going to move "
            "apart. This person is going to move this way. The other person is "
            "going to move that way.",
        ),
        (
            17,
            612.279,
            657.519,
            "However, the acceleration is not the same. If you use the equation F "
            "is equal to m8, the force acting on a smaller person is 100 and he has "
            "a mass of 50 kg. So solving for the acceleration, you can see that "
            "he's going to experience an acceleration of 2 m/s squared. Now the "
            "person on the right, he's going to experience a small acceleration "
            "because he has a larger mass. he has more inertia so it's going to be "
            "harder to move the larger person. So using equation F is equal to",
        ),
        (
            18,
            654.2,
            693.68,
            "mA he experiences a force of 100 but since he has a mass of 100 the "
            "acceleration is going to be one. So because he experiences a small "
            "acceleration he's not going to move very far. the smaller person, he's "
            "going to move a lot further than the larger person because he's "
            "smaller and he experiences a greater acceleration.",
        ),
    ]
    segments = [
        {
            "cue_id": f"pL2YfC-22Uc:cue:{cue}",
            "start": start,
            "end": end,
            "text": text,
        }
        for cue, start, end, text in raw_cues
    ]
    claim = (
        "solving for the acceleration, you can see that he's going to experience "
        "an acceleration of 2 m/s squared."
    )
    plan = _compact_custom_plan(
        request="Newton's second law",
        start_quote="skating I'm just going to draw",
        end_quote=(
            "smaller person, he's going to move a lot further than the larger "
            "person because he's smaller and he experiences a greater acceleration."
        ),
        claim_quote=claim,
    )
    plan = plan.model_copy(update={
        "topics": [plan.topics[0].model_copy(update={
            "candidate_id": "f-equals-ma-acceleration",
            "start_line": 1,
            "end_line": 4,
            "title": "Calculating Acceleration from Force and Mass",
            "learning_objective": (
                "Calculate acceleration given force and mass using F=ma."
            ),
            "facet": "Solving for acceleration",
        })],
    })

    report = gemini_segment._plan_to_report(
        plan,
        segments,
        [],
        {"_segment_trust_gemini_semantics": True},
        topic=plan.request_intent.exact_request,
    )

    assert report.accepted_count == report.proposed_count == 1
    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["start_cue_id"] == "pL2YfC-22Uc:cue:16"
    assert clip["start_quote"] == "Now, let's say if the mass"
    assert clip["_clip_text"].startswith("Now, let's say if the mass")
    assert "skating I'm just going to draw" not in clip["_clip_text"]
    assert "trimmed_clipped_start_to_claim_setup" in (
        clip["_boundary_fallback_reasons"]
    )


@pytest.mark.parametrize(
    (
        "model_start_line",
        "model_start_quote",
        "model_end_line",
        "model_end_quote",
        "combined_acceleration_tail",
        "exact_fresh_v34_shape",
        "punctuated_claim_handoff",
    ),
    [
        (
            1,
            "units for speed so if you're",
            3,
            "speed doesn't change you are accelerating",
            True,
            False,
            False,
        ),
        (
            0,
            "meters per second these are all examples",
            3,
            "speed doesn't change you are accelerating",
            False,
            False,
            False,
        ),
        (
            1,
            "units for speed so if you're",
            2,
            "common units for acceleration",
            False,
            False,
            False,
        ),
        (
            1,
            "units of speed so if you're",
            2,
            "common units for acceleration",
            False,
            False,
            False,
        ),
        (
            1,
            "units for speed so if you're",
            3,
            "even if your speed doesn't change you are accelerating",
            False,
            True,
            False,
        ),
        (
            1,
            "units for speed so if you're",
            3,
            "even if your speed doesn't change you are accelerating",
            False,
            True,
            True,
        ),
    ],
    ids=[
        "clipped-speed-tail",
        "complete-speed-background",
        "live-acceleration-definition",
        "ungrounded-live-acceleration-start",
        "exact-fresh-v34-selector-output",
        "punctuated-claim-handoff",
    ],
)
def test_trusted_live_units_candidate_advances_to_acceleration_handoff(
    model_start_line: int,
    model_start_quote: str,
    model_end_line: int,
    model_end_quote: str,
    combined_acceleration_tail: bool,
    exact_fresh_v34_shape: bool,
    punctuated_claim_handoff: bool,
) -> None:
    raw_cues = [
        (
            0,
            0.269,
            33.84,
            "thanks for stopping by I'm Virgil Rick's and this is 2-minute "
            "classroom today we're talking about motion specifically we're talking "
            "about speed velocity and acceleration and let's clear up a few things "
            "right off the bat first of all speed and velocity are different and "
            "acceleration is much more than just speeding up speed is the rate at "
            "which something changes its position it's represented as distance over "
            "time miles per hour kilometers per hour meters per second these are all "
            "examples of the",
        ),
        (
            1,
            31.08,
            63.27,
            "units for speed so if you're driving in your car at 72 miles per hour "
            "then that's your speed in fact it's your instantaneous speed or your "
            "speed and that exact moment if you get to the end of your trip and "
            "realize that it took you two hours to drive 120 miles then your overall "
            "speed was 60 miles per hour and this is also known as your average "
            "speed velocity is a lot like speed except for one important difference "
            "it's a vector which means it has a direction attached to it",
        ),
        (
            2,
            60.48,
            95.729,
            "so while your speed may have been 72 miles per hour your velocity was "
            "72 miles per hour east or 72 miles per hour towards the beach there "
            "just has to be some direction attached to the speed to make it a "
            "velocity with that knowledge in hand you're now ready to understand "
            "acceleration which is simply the rate at which velocity changes it's "
            "represented as distance per time per time or distance per time squared "
            "for example meters per second squared are common units for acceleration",
        ),
        (
            3,
            92.75,
            128.97,
            "anytime you change your velocity you are accelerating that can be "
            "speeding up or slowing down which is also known as negative "
            "acceleration but Direction is also a component of velocity so when you "
            "change your direction even if your speed doesn't change you are accelerating",
        ),
    ]
    if exact_fresh_v34_shape:
        if punctuated_claim_handoff:
            cue, start, end, text = raw_cues[2]
            raw_cues[2] = (
                cue,
                start,
                end,
                text.replace(
                    "understand acceleration which is simply",
                    "understand acceleration. Acceleration which is simply",
                ),
            )
        raw_cues[3] = (
            3,
            92.75,
            128.97,
            "anytime you change your velocity you are accelerating that can be "
            "speeding up or slowing down which is also known as negative "
            "acceleration but Direction is also a component of velocity so when "
            "you change your direction even if your speed doesn't change you are "
            "accelerating does that blow your mind all right let's recap speed is "
            "the rate at which something moves distance over time velocity is speed "
            "with a direction so distance over time with a specific direction and "
            "acceleration the king of",
        )
        raw_cues.append((
            4,
            126.03,
            167.009,
            "them all is distance over x squared or the rate at which velocity "
            "changes so whether you're speeding up slow down or changing directions "
            "you're changing your velocity and thus accelerating science is so cool "
            "I hope you enjoyed this video feel free to throw any comments below "
            "whether it be questions criticisms or just idle chitchat I'd love to "
            "hear it don't forget to watch and share my other videos and I'll catch "
            "you next time [Music]",
        ))
    segments = [
        {
            "cue_id": f"Jyiw6KkedDY:cue:{cue}",
            "start": start,
            "end": end,
            "text": text,
        }
        for cue, start, end, text in raw_cues
    ]
    claim = (
        "acceleration which is simply the rate at which velocity changes"
        if exact_fresh_v34_shape
        else (
            "meters per second squared are common units for acceleration"
            if combined_acceleration_tail
            else "meters per second squared are common units"
        )
    )
    plan = _compact_custom_plan(
        request="Newton's second law",
        start_quote=model_start_quote,
        end_quote=model_end_quote,
        claim_quote=claim,
    )
    plan = plan.model_copy(update={
        "topics": [plan.topics[0].model_copy(update={
            "candidate_id": (
                "acceleration-and-units"
                if combined_acceleration_tail and not exact_fresh_v34_shape
                else "acceleration-definition-units"
            ),
            "start_line": model_start_line,
            "end_line": model_end_line,
            "title": (
                "Acceleration and its Units"
                if combined_acceleration_tail and not exact_fresh_v34_shape
                else "Defining Acceleration and Its Units"
            ),
            "learning_objective": (
                "Define acceleration, state its units, and identify causes of acceleration"
                if combined_acceleration_tail and not exact_fresh_v34_shape
                else (
                    "Define acceleration as the rate of change of velocity and identify its common units"
                    if exact_fresh_v34_shape
                    else "Define acceleration as the rate of velocity change and identify its units"
                )
            ),
            "facet": (
                "speed and acceleration definition and units"
                if combined_acceleration_tail and not exact_fresh_v34_shape
                else "acceleration definition and units"
            ),
            "directly_teaches_topic": False,
        })],
    })
    if model_end_line == 2 or combined_acceleration_tail or exact_fresh_v34_shape:
        exact_request = (
            "Explain Newton's second law F=ma from intuition through worked examples, "
            "including net force, mass, acceleration, units, and solving for each variable"
        )
        constraints = [
            ("law", "subject", "Newton's second law F=ma"),
            ("intuition", "format", "from intuition"),
            ("worked_examples", "format", "through worked examples"),
            ("net_force", "subject", "net force"),
            ("mass", "subject", "mass"),
            ("acceleration", "subject", "acceleration"),
            ("units", "outcome", "units"),
            ("variables", "task", "solving for each variable"),
        ]
        plan = plan.model_copy(update={
            "request_intent": gemini_segment._RequestIntent.model_validate({
                "exact_request": exact_request,
                "constraints": [
                    {
                        "constraint_id": constraint_id,
                        "kind": kind,
                        "source_phrase": source_phrase,
                        "requirement": source_phrase,
                    }
                    for constraint_id, kind, source_phrase in constraints
                ],
            }),
            "topics": [plan.topics[0].model_copy(update={
                "intent_evidence": [
                    gemini_segment._CompactIntentEvidence.model_validate({
                        "id": "acceleration",
                        "q": (
                            claim
                            if exact_fresh_v34_shape
                            else "the rate at which velocity changes"
                            if combined_acceleration_tail
                            else "you're now ready to understand acceleration"
                        ),
                    }),
                    gemini_segment._CompactIntentEvidence.model_validate({
                        "id": "units",
                        "q": (
                            "meters per second squared are common units for acceleration"
                            if exact_fresh_v34_shape
                            else claim
                        ),
                    }),
                ],
            })],
        })

    report = gemini_segment._plan_to_report(
        plan,
        segments,
        [],
        {"_segment_trust_gemini_semantics": True},
        topic=plan.request_intent.exact_request,
    )

    assert report.accepted_count == report.proposed_count == 1
    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["start_cue_id"] == "Jyiw6KkedDY:cue:2"
    assert clip["start_quote"] == "you're now ready to understand acceleration"
    assert clip["_clip_text"].startswith(
        "you're now ready to understand acceleration"
    )
    assert "with that knowledge" not in clip["_clip_text"]
    assert "units for speed" not in clip["_clip_text"]
    assert "trimmed_clipped_start_to_claim_setup" in (
        clip["_boundary_fallback_reasons"]
    )


def test_trusted_claim_handoff_requires_the_whole_subject_to_match() -> None:
    segments = [
        {
            "cue_id": "partial-subject:cue:0",
            "start": 0.0,
            "end": 8.0,
            "text": "The previous section covers background and units for speed",
        },
        {
            "cue_id": "partial-subject:cue:1",
            "start": 8.0,
            "end": 18.0,
            "text": (
                "with that knowledge in hand you're now ready to understand mass "
                "media. Force equals mass times acceleration."
            ),
        },
    ]
    claim = "Force equals mass times acceleration"
    plan = _compact_custom_plan(
        request="Newton force relationship",
        start_quote="and units for speed",
        end_quote=claim,
        claim_quote=claim,
    )
    plan = plan.model_copy(update={
        "topics": [plan.topics[0].model_copy(update={
            "start_line": 0,
            "end_line": 1,
            "title": "Newton force relationship",
            "learning_objective": "Explain the Newton force relationship",
            "facet": "Newton force relationship",
        })],
    })

    report = gemini_segment._plan_to_report(
        plan,
        segments,
        [],
        {"_segment_trust_gemini_semantics": True},
        topic=plan.request_intent.exact_request,
    )

    assert report.accepted_count == report.proposed_count == 1
    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["start_cue_id"] == "partial-subject:cue:0"
    assert not clip["_clip_text"].startswith(
        "you're now ready to understand mass media"
    )


def _fresh_v34_xz_segments() -> list[dict]:
    texts = {
        21: "and acceleration. There are a number of",
        22: "things we can say about this equation,",
        23: "which is tiny but powerful. First it",
        24: "means that heavier objects will require",
        25: "the application of greater force in",
        26: "order to achieve the same acceleration",
        27: "as lighter objects, with acceleration being",
        28: "equal to force divided by mass. If we",
        29: "want these objects of varying masses",
        30: "each to accelerate at one meter per",
        31: "second squared,",
        32: "these are the magnitudes of the forces",
        33: "that must be applied in order for the",
        34: "math to work out. This also means that",
        35: "the second law can be rephrased to state",
        36: "that the acceleration an object",
        37: "experiences will be directly",
        38: "proportional to the force applied and",
        39: "inversely proportional to its mass. It is",
        40: "important to note that the net force is",
        41: "the sum of all the forces acting on an",
        42: "object. If multiple forces are acting on",
        43: "an object, which is often the case, we",
        44: "will need to represent them all in a",
        45: "free body diagram and then add up all",
        46: "the vectors to find the net force, which",
        47: "will tell us",
        48: "the direction of the acceleration that",
        49: "will occur in response to the net force.",
        50: "This kind of vector addition allows us",
        51: "to make predictions about the motion of",
        52: "an object even when it is being pushed",
        53: "or pulled in a variety of ways. Since",
    }
    return [
        {
            "cue_id": f"xzA6IBWUEDE:cue:{cue}",
            "start": float(cue),
            "end": float(cue) + 1.0,
            "text": text,
        }
        for cue, text in texts.items()
    ]


@pytest.mark.parametrize(
    (
        "start_cue",
        "end_cue",
        "start_quote",
        "end_quote",
        "claim",
        "title",
        "objective",
        "facet",
        "expected_cue",
        "expected_quote",
        "excluded_prefix",
    ),
    [
        (
            24,
            28,
            "means that heavier objects will require",
            "equal to force divided by mass",
            "acceleration being equal to force divided by mass",
            "F=ma proportionality intuition",
            "Explain how mass changes the force needed for equal acceleration",
            "force, mass, and acceleration proportionality",
            23,
            "First it",
            "which is tiny but powerful",
        ),
        (
            38,
            53,
            "proportional to the force applied and",
            "or pulled in a variety of ways",
            "net force is the sum of all the forces",
            "Net force and vector addition",
            "Explain how forces add to produce net force and acceleration",
            "net force vector addition",
            39,
            "It is",
            "proportional to the force applied",
        ),
    ],
    ids=["fma-proportionality", "net-force-vector-addition"],
)
def test_fresh_v34_relational_fragments_recover_the_claim_sentence(
    start_cue: int,
    end_cue: int,
    start_quote: str,
    end_quote: str,
    claim: str,
    title: str,
    objective: str,
    facet: str,
    expected_cue: int,
    expected_quote: str,
    excluded_prefix: str,
) -> None:
    segments = _fresh_v34_xz_segments()
    cue_to_line = {
        int(segment["cue_id"].rsplit(":", 1)[1]): line
        for line, segment in enumerate(segments)
    }
    plan = _compact_custom_plan(
        request="Newton's second law F=ma",
        start_quote=start_quote,
        end_quote=end_quote,
        claim_quote=claim,
    )
    plan = plan.model_copy(update={
        "topics": [plan.topics[0].model_copy(update={
            "start_line": cue_to_line[start_cue],
            "end_line": cue_to_line[end_cue],
            "title": title,
            "learning_objective": objective,
            "facet": facet,
        })],
    })

    report = gemini_segment._plan_to_report(
        plan,
        segments,
        [],
        {"_segment_trust_gemini_semantics": True},
        topic=plan.request_intent.exact_request,
    )

    assert report.accepted_count == report.proposed_count == 1
    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["start_cue_id"] == f"xzA6IBWUEDE:cue:{expected_cue}"
    assert clip["start_quote"] == expected_quote
    assert clip["_clip_text"].startswith(expected_quote)
    assert excluded_prefix not in clip["_clip_text"]
    assert "trimmed_clipped_start_to_claim_sentence" in (
        clip["_boundary_fallback_reasons"]
    )


def test_named_subject_relational_sentence_remains_a_complete_start() -> None:
    text = "Acceleration is proportional to force when mass stays constant."
    segments = [{
        "cue_id": "named-subject:cue:0",
        "start": 0.0,
        "end": 6.0,
        "text": text,
    }]
    plan = _compact_custom_plan(
        request="acceleration proportionality",
        start_quote="Acceleration is proportional to force",
        end_quote="mass stays constant",
        claim_quote="Acceleration is proportional to force",
    )

    report = gemini_segment._plan_to_report(
        plan,
        segments,
        [],
        {"_segment_trust_gemini_semantics": True},
        topic=plan.request_intent.exact_request,
    )

    assert report.accepted_count == report.proposed_count == 1
    [clip] = report.clips
    assert clip["start_cue_id"] == "named-subject:cue:0"
    assert clip["_clip_text"].startswith(
        "Acceleration is proportional to force"
    )
    assert "trimmed_clipped_start_to_claim_sentence" not in (
        clip["_boundary_fallback_reasons"]
    )


@pytest.mark.parametrize("cross_cue_start", [False, True])
def test_worked_example_keeps_givens_before_a_later_claim_frame(
    cross_cue_start: bool,
) -> None:
    action = "and divide" if cross_cue_start else "Divide"
    segments = [
        {
            "cue_id": "worked-example:cue:0",
            "start": 0.0,
            "end": 8.0,
            "text": (
                "A cart has a ten newton net force and a two kilogram mass."
            ),
        },
        {
            "cue_id": "worked-example:cue:1",
            "start": 8.0,
            "end": 17.0,
            "text": (
                f"{action} force by mass to "
                "obtain five meters per second squared. "
                "It is important to note that acceleration is five meters per "
                "second squared."
            ),
        },
    ]
    claim = "acceleration is five meters per second squared"
    plan = _compact_custom_plan(
        request="finding acceleration from force and mass",
        start_quote=(
            "and divide force by mass"
            if cross_cue_start
            else "and a two kilogram mass"
        ),
        end_quote=claim,
        claim_quote=claim,
    )
    plan = plan.model_copy(update={
        "topics": [plan.topics[0].model_copy(update={
            "start_line": 1 if cross_cue_start else 0,
            "end_line": 1,
            "title": "Finding acceleration",
            "learning_objective": (
                "Determine acceleration from force and mass"
            ),
            "facet": "numerical application",
        })],
    })

    report = gemini_segment._plan_to_report(
        plan,
        segments,
        [],
        {"_segment_trust_gemini_semantics": True},
        topic=plan.request_intent.exact_request,
    )

    assert report.accepted_count == report.proposed_count == 1
    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["start_cue_id"] == "worked-example:cue:0"
    assert clip["_clip_text"].startswith("A cart has a ten newton net force")
    assert "trimmed_clipped_start_to_claim_sentence" not in (
        clip["_boundary_fallback_reasons"]
    )


def test_application_scope_keeps_givens_split_across_cues() -> None:
    segments = [
        {
            "cue_id": "split-application:cue:0",
            "start": 0.0,
            "end": 6.0,
            "text": "A cart has a ten newton net force",
        },
        {
            "cue_id": "split-application:cue:1",
            "start": 6.0,
            "end": 12.0,
            "text": (
                "and a two kilogram mass. Divide force by mass to get five."
            ),
        },
        {
            "cue_id": "split-application:cue:2",
            "start": 12.0,
            "end": 18.0,
            "text": (
                "It is important to note that acceleration is five meters per "
                "second squared."
            ),
        },
    ]
    claim = "acceleration is five meters per second squared"
    plan = _compact_custom_plan(
        request="acceleration from force and mass",
        start_quote="and a two kilogram mass",
        end_quote=claim,
        claim_quote=claim,
    )
    plan = plan.model_copy(update={
        "topics": [plan.topics[0].model_copy(update={
            "start_line": 1,
            "end_line": 2,
            "title": "Acceleration application",
            "learning_objective": "Use force and mass to obtain acceleration",
            "facet": "force and mass example",
        })],
    })

    report = gemini_segment._plan_to_report(
        plan,
        segments,
        [],
        {"_segment_trust_gemini_semantics": True},
        topic=plan.request_intent.exact_request,
    )

    assert report.accepted_count == report.proposed_count == 1
    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["start_cue_id"] == "split-application:cue:0"
    assert clip["_clip_text"].startswith("A cart has a ten newton net force")
    assert "Divide force by mass to get five" in clip["_clip_text"]
    assert "trimmed_clipped_start_to_claim_sentence" not in (
        clip["_boundary_fallback_reasons"]
    )


def test_coherent_pathway_keeps_premises_before_a_final_claim() -> None:
    segments = [
        {
            "cue_id": "pathway:cue:0",
            "start": 0.0,
            "end": 6.0,
            "text": "Photosynthesis captures sunlight and uses water",
        },
        {
            "cue_id": "pathway:cue:1",
            "start": 6.0,
            "end": 13.0,
            "text": (
                "and carbon dioxide to make ATP, which drives carbon fixation."
            ),
        },
        {
            "cue_id": "pathway:cue:2",
            "start": 13.0,
            "end": 19.0,
            "text": (
                "Finally this shows that sunlight becomes stored chemical energy."
            ),
        },
    ]
    claim = "sunlight becomes stored chemical energy"
    plan = _compact_custom_plan(
        request="photosynthesis energy pathway",
        start_quote="and carbon dioxide to make ATP",
        end_quote=claim,
        claim_quote=claim,
    )
    plan = plan.model_copy(update={
        "topics": [plan.topics[0].model_copy(update={
            "start_line": 1,
            "end_line": 2,
            "title": "Photosynthesis energy pathway",
            "learning_objective": (
                "Trace how sunlight becomes stored chemical energy"
            ),
            "facet": "photosynthesis mechanism",
        })],
    })

    report = gemini_segment._plan_to_report(
        plan,
        segments,
        [],
        {"_segment_trust_gemini_semantics": True},
        topic=plan.request_intent.exact_request,
    )

    assert report.accepted_count == report.proposed_count == 1
    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["start_cue_id"] == "pathway:cue:0"
    assert clip["_clip_text"].startswith("Photosynthesis captures sunlight")
    assert "which drives carbon fixation" in clip["_clip_text"]
    assert "trimmed_clipped_start_to_claim_sentence" not in (
        clip["_boundary_fallback_reasons"]
    )


def test_trusted_recovered_misaligned_start_uses_later_claim_handoff() -> None:
    segments = [
        {
            "cue_id": "Jyiw6KkedDY:cue:0",
            "start": 0.269,
            "end": 33.84,
            "text": (
                "Speed is distance over time. Meters per second are examples of speed."
            ),
        },
        {
            "cue_id": "Jyiw6KkedDY:cue:1",
            "start": 31.08,
            "end": 63.27,
            "text": (
                "units for speed so if you're driving in your car at 72 miles per hour "
                "then that's your speed and your average speed can differ velocity is "
                "speed with a direction attached to it"
            ),
        },
        {
            "cue_id": "Jyiw6KkedDY:cue:2",
            "start": 60.48,
            "end": 95.729,
            "text": (
                "with that knowledge in hand you're now ready to understand acceleration "
                "which is simply the rate at which velocity changes it's represented as "
                "distance per time squared and meters per second squared are common units "
                "for acceleration"
            ),
        },
    ]
    claim = "acceleration which is simply the rate at which velocity changes"
    exact_request = (
        "Explain Newton's second law F=ma from intuition through worked examples, "
        "including net force, mass, acceleration, units, and solving for each variable"
    )
    plan = _compact_custom_plan(
        request="acceleration",
        start_quote="units for speed so if you're",
        end_quote="common units for acceleration",
        claim_quote=claim,
    )
    plan = plan.model_copy(update={
        "request_intent": plan.request_intent.model_copy(update={
            "exact_request": exact_request,
        }),
        "topics": [plan.topics[0].model_copy(update={
            "candidate_id": "acceleration-definition-units",
            "start_line": 0,
            "end_line": 2,
            "title": "Defining Acceleration and Its Units",
            "learning_objective": (
                "Define acceleration as the rate of velocity change and identify its units."
            ),
            "facet": "acceleration definition",
            "directly_teaches_topic": False,
        })],
    })

    report = gemini_segment._plan_to_report(
        plan,
        segments,
        [],
        {"_segment_trust_gemini_semantics": True},
        topic=plan.request_intent.exact_request,
    )

    assert report.accepted_count == report.proposed_count == 1
    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["start_cue_id"] == "Jyiw6KkedDY:cue:2"
    assert clip["start_quote"] == "you're now ready to understand acceleration"
    assert clip["edge_projection"]["start"] == {
        "required": True,
        "cue_id": "Jyiw6KkedDY:cue:2",
        "quote": "you're now ready to understand acceleration",
    }
    assert "range_recovered_from_edges" in clip["_boundary_fallback_reasons"]
    assert "trimmed_clipped_start_to_claim_setup" in (
        clip["_boundary_fallback_reasons"]
    )


def test_trusted_claim_quote_mismatch_beats_polluted_metadata_anchor() -> None:
    speed_setup = "Speed uses units such as miles an hour"
    claim = "meters per second squared are common units for acceleration"
    segments = [
        {
            "cue_id": "Jyiw6KkedDY:cue:1",
            "start": 31.08,
            "end": 60.48,
            "text": f"{speed_setup}.",
        },
        {
            "cue_id": "Jyiw6KkedDY:cue:2",
            "start": 60.48,
            "end": 95.729,
            "text": (
                "with that knowledge in hand you're now ready to understand "
                "acceleration which is simply the rate at which velocity changes "
                "and meters per second squared are common units for acceleration"
            ),
        },
    ]
    plan = _compact_custom_plan(
        request="acceleration",
        start_quote=speed_setup,
        end_quote=claim,
        claim_quote=claim,
    )
    plan = plan.model_copy(update={
        "topics": [plan.topics[0].model_copy(update={
            "candidate_id": "acceleration-and-units",
            "start_line": 0,
            "end_line": 1,
            "title": "Acceleration and its Units",
            "learning_objective": (
                "Define acceleration, state its units, and identify causes of acceleration"
            ),
            # Deliberately repeats both start-quote anchors. The atomic claim,
            # not polluted metadata, must control this strong teaching handoff.
            "facet": "speed and acceleration definition and units",
        })],
    })

    report = gemini_segment._plan_to_report(
        plan,
        segments,
        [],
        {"_segment_trust_gemini_semantics": True},
        topic=plan.request_intent.exact_request,
    )

    assert report.accepted_count == report.proposed_count == 1
    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["start_cue_id"] == "Jyiw6KkedDY:cue:2"
    assert clip["start_quote"] == "you're now ready to understand acceleration"
    assert clip["edge_projection"]["start"] == {
        "required": True,
        "cue_id": "Jyiw6KkedDY:cue:2",
        "quote": "you're now ready to understand acceleration",
    }
    assert speed_setup not in clip["_clip_text"]
    assert "trimmed_clipped_start_to_claim_setup" in (
        clip["_boundary_fallback_reasons"]
    )


def test_trusted_claim_mismatch_preserves_complete_wagon_setup() -> None:
    wagon_setup = (
        "Picture a loaded wagon that barely changes its motion when shoved"
    )
    claim = (
        "Newton's second law says acceleration equals net force divided by mass"
    )
    segments = [
        {
            "cue_id": "wagon:cue:0",
            "start": 0.0,
            "end": 6.0,
            "text": f"{wagon_setup}.",
        },
        {
            "cue_id": "wagon:cue:1",
            "start": 6.0,
            "end": 14.0,
            "text": (
                "with that knowledge in hand you're now ready to understand "
                f"{claim}."
            ),
        },
    ]
    plan = _compact_custom_plan(
        request="Newton's second law",
        start_quote=wagon_setup,
        end_quote=claim,
        claim_quote=claim,
    )
    plan = plan.model_copy(update={
        "topics": [plan.topics[0].model_copy(update={
            "title": "Newton's Second Law",
            "learning_objective": (
                "Explain how Newton's second law relates force, mass, and acceleration"
            ),
            # This makes the broad metadata anchor overlap the opening while
            # the atomic claim remains lexically distinct from it.
            "facet": "loaded wagon motion under Newton's second law",
        })],
    })

    report = gemini_segment._plan_to_report(
        plan,
        segments,
        [],
        {"_segment_trust_gemini_semantics": True},
        topic=plan.request_intent.exact_request,
    )

    assert report.accepted_count == report.proposed_count == 1
    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["start_cue_id"] == "wagon:cue:0"
    assert clip["start_quote"] == wagon_setup
    assert clip["_clip_text"].startswith(wagon_setup)
    assert "trimmed_clipped_start_to_claim_setup" not in (
        clip["_boundary_fallback_reasons"]
    )


def test_trusted_claim_mismatch_ignores_unrelated_teaching_handoff() -> None:
    speed_setup = "Speed uses units such as miles an hour"
    claim = "meters per second squared are common units for acceleration"
    segments = [
        {
            "cue_id": "energy:cue:0",
            "start": 0.0,
            "end": 6.0,
            "text": f"{speed_setup}.",
        },
        {
            "cue_id": "energy:cue:1",
            "start": 6.0,
            "end": 15.0,
            "text": (
                "with that knowledge in hand you're now ready to understand "
                "energy through work and power, while meters per second squared "
                "are common units for acceleration"
            ),
        },
    ]
    plan = _compact_custom_plan(
        request="acceleration",
        start_quote=speed_setup,
        end_quote=claim,
        claim_quote=claim,
    )
    plan = plan.model_copy(update={
        "topics": [plan.topics[0].model_copy(update={
            "title": "Acceleration and its Units",
            "learning_objective": "Define acceleration and state its units",
            "facet": "speed and acceleration definition and units",
        })],
    })

    report = gemini_segment._plan_to_report(
        plan,
        segments,
        [],
        {"_segment_trust_gemini_semantics": True},
        topic=plan.request_intent.exact_request,
    )

    assert report.accepted_count == report.proposed_count == 1
    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["start_cue_id"] == "energy:cue:0"
    assert clip["start_quote"] == speed_setup
    assert clip["_clip_text"].startswith(speed_setup)
    assert "trimmed_clipped_start_to_claim_setup" not in (
        clip["_boundary_fallback_reasons"]
    )


def test_trusted_live_dependent_answer_recovers_its_question_context() -> None:
    segments = [
        {
            "cue_id": "vxFYfumAAlY:cue:3",
            "start": 102.42,
            "end": 130.58,
            "text": (
                "Will acceleration be involved here? Yes, because the velocity is "
                "changing. It was zero initially, and at the end of five seconds "
                "it's 20 kilometers per hour. In another scenario, assume that a "
                "body is moving at 30 kilometers an hour, and then it moves right at "
                "30 kilometers an hour and continues traveling at the same speed of "
                "30 kilometers per hour. Will there be acceleration in this case?"
            ),
        },
        {
            "cue_id": "vxFYfumAAlY:cue:4",
            "start": 135.32,
            "end": 169.56,
            "text": (
                "The answer is yes ! As the direction is changing, the velocity will "
                "also change. The only thing constant in this example is speed. But "
                "the velocity changes as the direction changes! So this was the most "
                "important concept you had to know about acceleration. It will exist "
                "only when there is a change in velocity!"
            ),
        },
    ]
    claim = "As the direction is changing, the velocity will also change"
    plan = _compact_custom_plan(
        request="acceleration",
        start_quote="The answer is yes ! As the",
        end_quote="It will exist only when there is a change in velocity!",
        claim_quote=claim,
    )
    plan = plan.model_copy(update={
        "topics": [plan.topics[0].model_copy(update={
            "candidate_id": "when-does-acceleration-exist",
            "start_line": 1,
            "end_line": 1,
            "title": "When Does Acceleration Exist?",
            "learning_objective": (
                "Identify that acceleration occurs whenever speed or direction changes"
            ),
            "facet": "acceleration and changing velocity",
        })],
    })

    report = gemini_segment._plan_to_report(
        plan,
        segments,
        [],
        {"_segment_trust_gemini_semantics": True},
        topic=plan.request_intent.exact_request,
    )

    assert report.accepted_count == report.proposed_count == 1
    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["start_cue_id"] == "vxFYfumAAlY:cue:3"
    assert "In another scenario, assume" in clip["_clip_text"]
    assert "Will there be acceleration in this case? The answer is yes" in (
        clip["_clip_text"]
    )
    assert not clip["_clip_text"].startswith("The answer is yes")


def test_trusted_live_weight_fragment_recovers_split_definition() -> None:
    segments = [
        {
            "cue_id": "pL2YfC-22Uc:cue:9",
            "start": 306.12,
            "end": 351.039,
            "text": (
                "Mass is measured in units of kilograms. Weight is different from "
                "mass. Weight is a force. Weight is equal to mass time gravitational "
                "acceleration. Weight is simply the force of"
            ),
        },
        {
            "cue_id": "pL2YfC-22Uc:cue:10",
            "start": 348.0,
            "end": 399.88,
            "text": (
                "gravity that is acting on you. And so weight is a downward force. "
                "Consider this 20 kg object. What is the weight force acting on it? "
                "Go ahead and calculate it. So the weight force is always equal to mg "
                "mass time gravitational acceleration. On Earth the gravitational "
                "acceleration is 9.8 meters per second squared. So 20 times 9.8 is "
                "about 196 newtons. So that's the weight force."
            ),
        },
    ]
    claim = "weight force is always equal to mg mass time gravitational acceleration"
    plan = _compact_custom_plan(
        request="Newton's second law",
        start_quote="gravity that is acting on you",
        end_quote="So that's the weight force.",
        claim_quote=claim,
    )
    plan = plan.model_copy(update={
        "topics": [plan.topics[0].model_copy(update={
            "candidate_id": "weight-force-calculation",
            "start_line": 1,
            "end_line": 1,
            "title": "Calculating Weight Force",
            "learning_objective": (
                "Apply F=ma to solve for weight using mass and gravity"
            ),
            "facet": "calculating weight force",
            "intent_evidence": [
                gemini_segment._CompactIntentEvidence.model_validate({
                    "id": "subject",
                    "q": "Weight is equal to mass time gravitational acceleration",
                }),
            ],
        })],
    })

    report = gemini_segment._plan_to_report(
        plan,
        segments,
        [],
        {"_segment_trust_gemini_semantics": True},
        topic=plan.request_intent.exact_request,
    )

    assert report.accepted_count == report.proposed_count == 1
    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["start_cue_id"] == "pL2YfC-22Uc:cue:9"
    assert clip["start_quote"] == "Weight is simply the force of"
    assert clip["_clip_text"].startswith("Weight is simply the force of gravity")
    assert clip["edge_projection"]["start"]["quote"] == (
        "Weight is simply the force of"
    )


def test_trusted_live_skating_fragment_never_starts_mid_phrase() -> None:
    segments = [
        {
            "cue_id": "pL2YfC-22Uc:cue:14",
            "start": 513.279,
            "end": 527.839,
            "text": (
                "Newton's third law states that for every action force there is an "
                "equal but opposite reaction force. So let's say if you have two people"
            ),
        },
        {
            "cue_id": "pL2YfC-22Uc:cue:15",
            "start": 530.36,
            "end": 573.04,
            "text": (
                "skating I'm just going to draw stick figures and one person has more "
                "mass than the other person. So if the smaller person applies a force "
                "of 100 newtons on the larger person, what force will the larger person "
                "apply on the smaller person? The forces are equal and opposite."
            ),
        },
        {
            "cue_id": "pL2YfC-22Uc:cue:16",
            "start": 570.92,
            "end": 614.72,
            "text": (
                "Now, let's say if the mass of the smaller person is 50 kg and the "
                "mass of the larger person is 100 kg, who experiences the greater "
                "acceleration? Let's say if they're on ice."
            ),
        },
        {
            "cue_id": "pL2YfC-22Uc:cue:17",
            "start": 612.279,
            "end": 657.519,
            "text": (
                "However, the acceleration is not the same. If you use F equals ma, "
                "the force acting on the smaller person is 100 and the mass is 50. So "
                "solving for the acceleration, you can see that he's going to experience "
                "an acceleration of 2 meters per second squared."
            ),
        },
        {
            "cue_id": "pL2YfC-22Uc:cue:18",
            "start": 654.2,
            "end": 693.68,
            "text": (
                "The larger person has a mass of 100, so the acceleration is one. "
                "The smaller person moves further because the acceleration is greater."
            ),
        },
        {
            "cue_id": "pL2YfC-22Uc:cue:19",
            "start": 689.839,
            "end": 697.936,
            "text": "he's not going to move back very much since he experiences a smaller acceleration.",
        },
    ]
    claim = (
        "solving for the acceleration, you can see that he's going to experience "
        "an acceleration"
    )
    plan = _compact_custom_plan(
        request="Newton's second law",
        start_quote="skating I'm just going to draw",
        end_quote="since he experiences a smaller acceleration.",
        claim_quote=claim,
    )
    plan = plan.model_copy(update={
        "topics": [plan.topics[0].model_copy(update={
            "candidate_id": "solving-for-acceleration",
            "start_line": 1,
            "end_line": 5,
            "title": "Solving for Acceleration",
            "learning_objective": (
                "Use F=ma to solve for acceleration given mass and force"
            ),
            "facet": "solving for acceleration",
        })],
    })

    report = gemini_segment._plan_to_report(
        plan,
        segments,
        [],
        {"_segment_trust_gemini_semantics": True},
        topic=plan.request_intent.exact_request,
    )

    assert report.accepted_count == report.proposed_count == 1
    assert report.rejected_reasons == []
    [clip] = report.clips
    assert not clip["_clip_text"].startswith("skating")
    assert clip["start_cue_id"] in {
        "pL2YfC-22Uc:cue:14",
        "pL2YfC-22Uc:cue:16",
    }


@pytest.mark.parametrize(
    "independent_opening",
    [
        "No external force acts on the object.",
        "Correct acceleration requires the net force.",
        "Exactly one net force vector determines the acceleration.",
    ],
)
def test_trusted_independent_opening_does_not_import_prior_sentence(
    independent_opening: str,
) -> None:
    prefix = "Photosynthesis stores light energy."
    text = f"{prefix} {independent_opening}"
    plan = _compact_custom_plan(
        request="acceleration",
        start_quote=independent_opening,
        end_quote=independent_opening,
        claim_quote=independent_opening,
    )

    report = gemini_segment._plan_to_report(
        plan,
        [{"cue_id": "cue-0", "start": 0.0, "end": 8.0, "text": text}],
        [],
        {"_segment_trust_gemini_semantics": True},
        topic=plan.request_intent.exact_request,
    )

    assert report.accepted_count == report.proposed_count == 1
    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["_clip_text"] == independent_opening.rstrip(".")
    assert prefix not in clip["_clip_text"]


def test_trusted_lowercase_complete_cue_does_not_import_complete_prior_cue() -> None:
    segments = [
        {
            "cue_id": "cue-0",
            "start": 0.0,
            "end": 4.0,
            "text": "Photosynthesis stores light energy.",
        },
        {
            "cue_id": "cue-1",
            "start": 4.1,
            "end": 9.0,
            "text": "acceleration equals net force divided by mass.",
        },
    ]
    claim = "acceleration equals net force divided by mass"
    plan = _compact_custom_plan(
        request="acceleration",
        start_quote=claim,
        end_quote=claim,
        claim_quote=claim,
    )
    plan = plan.model_copy(update={
        "topics": [plan.topics[0].model_copy(update={
            "start_line": 1,
            "end_line": 1,
        })],
    })

    report = gemini_segment._plan_to_report(
        plan,
        segments,
        [],
        {"_segment_trust_gemini_semantics": True},
        topic=plan.request_intent.exact_request,
    )

    assert report.accepted_count == report.proposed_count == 1
    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["start_cue_id"] == "cue-1"
    assert clip["_clip_text"] == claim


@pytest.mark.parametrize(
    "later_setup",
    [
        "Now what acceleration results when the net force acts on the mass?",
        (
            "Now let's say if a net force acts on a mass, what acceleration "
            "results?"
        ),
    ],
    ids=["later-question", "later-conditional"],
)
def test_trusted_lexically_different_same_objective_setup_is_preserved(
    later_setup: str,
) -> None:
    opening = "Picture a loaded wagon that barely changes its motion when shoved"
    claim = (
        "Newton's second law says acceleration equals net force divided by mass"
    )
    text = f"{opening}. {later_setup} {claim}."
    plan = _compact_custom_plan(
        request="Newton's second law",
        start_quote=opening,
        end_quote=claim,
        claim_quote=claim,
    )
    plan = plan.model_copy(update={
        "topics": [plan.topics[0].model_copy(update={
            "title": "Applying Newton's Second Law",
            "learning_objective": (
                "Calculate acceleration from net force and mass."
            ),
            "facet": "force-mass-acceleration calculation",
        })],
    })

    report = gemini_segment._plan_to_report(
        plan,
        [{"cue_id": "cue-0", "start": 0.0, "end": 18.0, "text": text}],
        [],
        {"_segment_trust_gemini_semantics": True},
        topic=plan.request_intent.exact_request,
    )

    assert report.accepted_count == report.proposed_count == 1
    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["start_quote"] == opening
    assert clip["_clip_text"].startswith(opening)
    assert "trimmed_clipped_start_to_claim_setup" not in (
        clip["_boundary_fallback_reasons"]
    )


def test_trusted_candidate_does_not_advance_to_an_unresolved_question() -> None:
    segments = [
        {
            "cue_id": "cue-0",
            "start": 0.0,
            "end": 5.0,
            "text": "A cart begins accelerating after a push and",
        },
        {
            "cue_id": "cue-1",
            "start": 4.8,
            "end": 14.0,
            "text": (
                "continues speeding up. Now what force causes this acceleration? "
                "Net force causes this acceleration according to Newton's second law."
            ),
        },
    ]
    claim = "Net force causes this acceleration according to Newton's second law"
    plan = _compact_custom_plan(
        request="Newton's second law",
        start_quote="continues speeding up",
        end_quote="according to Newton's second law",
        claim_quote=claim,
    )
    plan = plan.model_copy(update={
        "topics": [plan.topics[0].model_copy(update={
            "start_line": 1,
            "end_line": 1,
            "title": "Force and Acceleration",
            "learning_objective": "Explain which force causes acceleration",
            "facet": "net force and acceleration",
        })],
    })

    report = gemini_segment._plan_to_report(
        plan,
        segments,
        [],
        {"_segment_trust_gemini_semantics": True},
        topic=plan.request_intent.exact_request,
    )

    assert report.accepted_count == report.proposed_count == 1
    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["start_quote"] == "continues speeding up"
    assert clip["_clip_text"].startswith("continues speeding up")
    assert "trimmed_clipped_start_to_claim_setup" not in (
        clip["_boundary_fallback_reasons"]
    )


def test_trusted_claim_setup_ignores_later_same_cue_topic_reset() -> None:
    segments = [
        {
            "cue_id": "cue-0",
            "start": 0.0,
            "end": 4.0,
            "text": "We were discussing motion and",
        },
        {
            "cue_id": "cue-1",
            "start": 3.8,
            "end": 18.0,
            "text": (
                "continuing from before. Now how are force and acceleration "
                "related? Force equals mass times acceleration. Now let us move "
                "on to energy. Energy is conserved."
            ),
        },
    ]
    claim = "Force equals mass times acceleration"
    plan = _compact_custom_plan(
        request="Newton's second law",
        start_quote="continuing from before",
        end_quote=claim,
        claim_quote=claim,
    )
    plan = plan.model_copy(update={
        "topics": [plan.topics[0].model_copy(update={
            "start_line": 1,
            "end_line": 1,
            "title": "Force and Acceleration",
            "learning_objective": "Explain how force and acceleration are related",
            "facet": "force and acceleration relationship",
        })],
    })

    report = gemini_segment._plan_to_report(
        plan,
        segments,
        [],
        {"_segment_trust_gemini_semantics": True},
        topic=plan.request_intent.exact_request,
    )

    assert report.accepted_count == report.proposed_count == 1
    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["_clip_text"].startswith(
        "Now how are force and acceleration related?"
    )
    assert claim in clip["_clip_text"]
    assert "move on to energy" not in clip["_clip_text"]
    assert clip["model_claim_quote"] == clip["topic_evidence_quote"] == claim
    assert "claim_quote_reanchored" not in clip["_boundary_fallback_reasons"]


def test_trusted_ordinal_context_drops_speaker_tag_before_named_setup() -> None:
    segments = [
        {
            "cue_id": "cue-0",
            "start": 0.0,
            "end": 1.89,
            "text": "Professor Dave again, I want to tell you",
        },
        {
            "cue_id": "cue-1",
            "start": 1.89,
            "end": 3.85,
            "text": "about Newton's second law.",
        },
        {
            "cue_id": "cue-2",
            "start": 3.85,
            "end": 8.0,
            "text": "For a fixed mass, more net force produces more acceleration.",
        },
        {
            "cue_id": "cue-3",
            "start": 8.0,
            "end": 15.0,
            "text": (
                "The second law can be rephrased to state that acceleration is "
                "proportional to net force."
            ),
        },
    ]
    claim = "acceleration is proportional to net force"
    plan = _compact_custom_plan(
        request="Newton's second law",
        start_quote="The second law can be rephrased",
        end_quote="proportional to net force",
        claim_quote=claim,
    )
    plan = plan.model_copy(update={
        "topics": [plan.topics[0].model_copy(update={
            "start_line": 3,
            "end_line": 3,
            "intent_evidence": [gemini_segment._CompactIntentEvidence(
                id="subject", q=claim,
            )],
        })],
    })

    report = gemini_segment._plan_to_report(
        plan,
        segments,
        [],
        {"_segment_trust_gemini_semantics": True},
        topic=plan.request_intent.exact_request,
    )

    assert report.accepted_count == report.proposed_count == 1
    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["cue_ids"] == ["cue-0", "cue-1", "cue-2", "cue-3"]
    assert clip["_clip_text"].startswith(
        "I want to tell you about Newton's second law"
    )
    assert not clip["_clip_text"].startswith("Professor Dave again")
    assert clip["edge_projection"]["start"]["quote"] == (
        "I want to tell you"
    )
    assert "trimmed_speaker_framing_prefix" in (
        clip["_boundary_fallback_reasons"]
    )


def test_trusted_candidate_recovers_handoff_cut_before_its_subject() -> None:
    text = (
        "earth's gravitational field now let's talk more about the acceleration "
        "due to gravity which is approximately 9.8 meters per second squared"
    )
    claim = "acceleration due to gravity which is approximately 9.8"
    plan = _compact_custom_plan(
        request="meaning of acceleration",
        start_quote="the acceleration due to gravity",
        end_quote="9.8 meters per second squared",
        claim_quote=claim,
    )

    report = gemini_segment._plan_to_report(
        plan,
        [{"cue_id": "cue-0", "start": 0.0, "end": 12.0, "text": text}],
        [],
        {"_segment_trust_gemini_semantics": True},
        topic=plan.request_intent.exact_request,
    )

    assert report.accepted_count == report.proposed_count == 1
    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["_clip_text"].startswith(
        "now let's talk more about the acceleration"
    )
    assert not clip["_clip_text"].startswith("earth's gravitational field")
    assert clip["edge_projection"]["start"]["quote"] == (
        "now let's talk more about the"
    )
    assert "expanded_projected_start_context" in (
        clip["_boundary_fallback_reasons"]
    )


def test_trusted_candidate_keeps_unresolved_start_across_section_gap() -> None:
    segments = [
        {
            "cue_id": "cue-0",
            "start": 0.0,
            "end": 3.0,
            "text": "An unrelated earlier section ends here.",
        },
        {
            "cue_id": "cue-1",
            "start": 12.0,
            "end": 18.0,
            "text": "The second law says acceleration increases with net force.",
        },
    ]
    claim = "acceleration increases with net force"
    plan = _compact_custom_plan(
        request="Newton's second law",
        start_quote="The second law says",
        end_quote="increases with net force",
        claim_quote=claim,
    )
    plan = plan.model_copy(update={
        "topics": [plan.topics[0].model_copy(update={
            "start_line": 1,
            "end_line": 1,
        })],
    })

    report = gemini_segment._plan_to_report(
        plan,
        segments,
        [],
        {"_segment_trust_gemini_semantics": True},
        topic=plan.request_intent.exact_request,
    )

    assert report.accepted_count == report.proposed_count == 1
    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["cue_ids"] == ["cue-1"]
    assert "unresolved_start_context" in clip["_boundary_fallback_reasons"]


def test_trusted_candidate_recovers_same_cue_downward_scenario_start() -> None:
    segments = [
        {
            "cue_id": "cue-0",
            "start": 0.0,
            "end": 20.0,
            "text": "Tension is the pulling force exerted by a rope.",
        },
        {
            "cue_id": "cue-1",
            "start": 20.0,
            "end": 45.0,
            "text": (
                "For the upward-moving box, the tension must be larger than its "
                "weight, so the net force points upward."
            ),
        },
        {
            "cue_id": "cue-2",
            "start": 45.0,
            "end": 70.0,
            "text": (
                "Now if you want to compare it to the weight force, the weight force "
                "is greater. But now what if the rope is being used to allow the box "
                "to slowly descend? Intuitively, tension must be less than the weight "
                "force. So let's get the answer for part b."
            ),
        },
        {
            "cue_id": "cue-3",
            "start": 70.0,
            "end": 95.0,
            "text": (
                "The forces are weight downward and tension upward. Weight minus "
                "tension equals mass times the downward acceleration."
            ),
        },
        {
            "cue_id": "cue-4",
            "start": 95.0,
            "end": 115.0,
            "text": (
                "With m equal to 50 and g equal to 9.8, solving the equation gives a "
                "tension of 452.5 newtons, which is less than the weight force."
            ),
        },
    ]
    claim = "Weight minus tension equals mass times the downward acceleration"
    plan = _compact_custom_plan(
        request="tension while a box accelerates downward",
        start_quote="what if the rope is being used to allow the box",
        end_quote="less than the weight force",
        claim_quote=claim,
    )
    plan = plan.model_copy(update={
        "request_intent": gemini_segment._RequestIntent.model_validate({
            "exact_request": plan.request_intent.exact_request,
            "constraints": [
                {
                    "constraint_id": "accel",
                    "kind": "subject",
                    "source_phrase": "downward acceleration",
                    "requirement": "Use downward acceleration",
                },
                {
                    "constraint_id": "units",
                    "kind": "format",
                    "source_phrase": "newtons",
                    "requirement": "Give the answer in newtons",
                },
                {
                    "constraint_id": "mass",
                    "kind": "subject",
                    "source_phrase": "mass",
                    "requirement": "Use the given mass",
                },
                {
                    "constraint_id": "format",
                    "kind": "format",
                    "source_phrase": "worked example",
                    "requirement": "Work through part b",
                },
            ],
        }),
        "topics": [plan.topics[0].model_copy(update={
            "start_line": 2,
            "end_line": 4,
            "title": "Tension While a Box Descends",
            "learning_objective": (
                "Solve for tension while a rope lowers an accelerating box."
            ),
            "facet": "Downward-acceleration tension scenario",
            "intent_evidence": [
                gemini_segment._CompactIntentEvidence(id="accel", q=claim),
                gemini_segment._CompactIntentEvidence(
                    id="units", q="tension of 452.5 newtons",
                ),
                gemini_segment._CompactIntentEvidence(
                    id="mass", q="With m equal to 50 and g equal to 9.8",
                ),
                gemini_segment._CompactIntentEvidence(
                    id="format", q="So let's get the answer for part b",
                ),
            ],
        })],
    })

    report = gemini_segment._plan_to_report(
        plan,
        segments,
        [],
        {"_segment_trust_gemini_semantics": True},
        topic=plan.request_intent.exact_request,
    )

    assert report.accepted_count == report.proposed_count == 1
    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["intent_coverage"] == 1.0
    assert clip["cue_ids"] == ["cue-2", "cue-3", "cue-4"]
    assert clip["_clip_text"].startswith(
        "But now what if the rope is being used to allow the box to slowly descend"
    )
    assert "compare it to the weight force" not in clip["_clip_text"]
    assert "trimmed_scenario_before_claim" in (
        clip["_boundary_fallback_reasons"]
    )


def test_trusted_candidate_does_not_widen_clean_unit_to_prior_scenario() -> None:
    text = (
        "What if scenario A doubles its input? Scenario A would increase its output. "
        "Now let's discuss concept B. Concept B is a separate clean concept with its "
        "own complete explanation."
    )
    claim = "Concept B is a separate clean concept with its own complete explanation"
    plan = _compact_custom_plan(
        request="concept B",
        start_quote="Concept B is a separate clean concept",
        end_quote="its own complete explanation",
        claim_quote=claim,
    )
    plan = plan.model_copy(update={
        "topics": [plan.topics[0].model_copy(update={
            "title": "Concept B",
            "learning_objective": "Explain concept B as its own unit.",
            "facet": "concept B",
            "intent_evidence": [gemini_segment._CompactIntentEvidence(
                id="subject", q=claim,
            )],
        })],
    })

    report = gemini_segment._plan_to_report(
        plan,
        [{"cue_id": "cue-0", "start": 0.0, "end": 18.0, "text": text}],
        [],
        {"_segment_trust_gemini_semantics": True},
        topic=plan.request_intent.exact_request,
    )

    assert report.accepted_count == report.proposed_count == 1
    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["_clip_text"].startswith("Now let's discuss concept B")
    assert "What if scenario A" not in clip["_clip_text"]
    assert "scenario A" not in clip["_clip_text"]


def test_trusted_candidate_trims_next_problem_tail_without_rejection() -> None:
    text = (
        "For part b, the acceleration is negative 0.75 meters per second squared. "
        "The tension is 452.5 newtons, so it is less than the weight force "
        "during downward acceleration. Now let's work on this problem what is "
        "the tension in the two ropes in the picture shown below, and the crate "
        "or the box, whatever you want to call it."
    )
    claim = "The tension is 452.5 newtons"
    plan = _compact_custom_plan(
        request="tension during downward acceleration",
        start_quote="For part b, the acceleration is negative 0.75",
        end_quote="whatever you want to call it",
        claim_quote=claim,
    )

    report = gemini_segment._plan_to_report(
        plan,
        [{"cue_id": "cue-0", "start": 0.0, "end": 40.0, "text": text}],
        [],
        {"_segment_trust_gemini_semantics": True},
        topic=plan.request_intent.exact_request,
    )

    assert report.accepted_count == report.proposed_count == 1
    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["_clip_text"].endswith("during downward acceleration")
    assert "work on this problem" not in clip["_clip_text"]
    assert {
        "trimmed_same_cue_unit_after",
        "trimmed_trailing_edge_noise",
    } & set(clip["_boundary_fallback_reasons"])


def test_trusted_candidate_atomizes_broad_span_around_gemini_claim() -> None:
    segments = [
        {
            "cue_id": "cue-0",
            "start": 0.0,
            "end": 40.0,
            "text": (
                "However, the acceleration is not the same. Solving for the "
                "acceleration, you can see that the smaller person accelerates more."
            ),
        },
        {
            "cue_id": "cue-1",
            "start": 40.0,
            "end": 80.0,
            "text": (
                "Another example of Newton's third law is gravitational force "
                "between Earth and the moon."
            ),
        },
        {
            "cue_id": "cue-2",
            "start": 80.0,
            "end": 120.0,
            "text": "The resultant vector has a direction of 190.9 degrees.",
        },
        {
            "cue_id": "cue-3",
            "start": 120.0,
            "end": 165.0,
            "text": (
                "Now, let's get back to tension. So let's say if we have a "
                "horizontal surface and two blocks connected by a rope. A force of "
                "60 newtons pulls to the right."
            ),
        },
        {
            "cue_id": "cue-4",
            "start": 165.0,
            "end": 210.0,
            "text": (
                "To find the tension, we find the acceleration. Using this equation F "
                "is equal to MA we have a net force of 60 newtons and a total mass of "
                "30 kilograms."
            ),
        },
        {
            "cue_id": "cue-5",
            "start": 210.0,
            "end": 250.0,
            "text": (
                "The acceleration is 2 meters per second squared, so isolate the "
                "20 kilogram block and subtract its tension from 60 newtons."
            ),
        },
        {
            "cue_id": "cue-6",
            "start": 250.0,
            "end": 285.0,
            "text": (
                "The tension is 20 newtons and the net force is 40 newtons that "
                "propels the block to the right."
            ),
        },
    ]
    constraints = [
        {"constraint_id": "subject", "kind": "subject", "source_phrase": "F=ma", "requirement": "Use F=ma"},
        {"constraint_id": "format", "kind": "format", "source_phrase": "worked example", "requirement": "Give a worked example"},
        {"constraint_id": "components", "kind": "subject", "source_phrase": "net force", "requirement": "Use net force"},
        {"constraint_id": "task", "kind": "task", "source_phrase": "solve", "requirement": "Solve for acceleration"},
    ]
    plan = gemini_segment._CompactBoundaryPlan(
        request_intent={
            "exact_request": "Use F=ma in a worked two-block example",
            "constraints": constraints,
        },
        topics=[gemini_segment._CompactBoundaryTopic(
            candidate_id="f-ma-two-blocks-tension",
            start_line=0,
            end_line=6,
            start_quote="However, the acceleration is not the same",
            end_quote="propels the block to the right",
            claim_quote="Using this equation F is equal to MA we have a net force of 60",
            title="Applying F=ma to a Two-Block System",
            learning_objective=(
                "Use Newton's second law to solve for acceleration and tension in a "
                "two-block system."
            ),
            facet="Worked example: Two connected blocks",
            informativeness=0.9,
            topic_relevance=0.95,
            educational_importance=0.9,
            difficulty=0.35,
            directly_teaches_topic=True,
            substantive=True,
            factually_grounded=True,
            self_contained=True,
            is_standalone=True,
            intent_evidence=[
                {"id": "subject", "q": "Using this equation F is equal to MA"},
                {"id": "format", "q": "let's say if we have a horizontal surface"},
                {"id": "components", "q": "we have a net force of 60 newtons"},
                {"id": "task", "q": "Solving for the acceleration, you can see that"},
            ],
        )],
    )

    report = gemini_segment._plan_to_report(
        plan,
        segments,
        [],
        {"_segment_trust_gemini_semantics": True},
        topic=plan.request_intent.exact_request,
    )

    assert report.accepted_count == report.proposed_count == 1
    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["cue_ids"] == ["cue-3", "cue-4", "cue-5", "cue-6"]
    assert clip["_clip_text"].startswith("Now, let's get back to tension")
    assert "smaller person" not in clip["_clip_text"]
    assert "resultant vector" not in clip["_clip_text"]
    assert clip["intent_coverage"] == pytest.approx(0.75)
    assert "trimmed_adjacent_unit_before" in clip["_boundary_fallback_reasons"]


@pytest.mark.parametrize(
    ("topic_request", "text", "start_quote", "end_quote", "claim_quote"),
    [
        (
            "p value",
            "based on which one is larger when it comes to your conclusion there are "
            "two outcomes. If the p value is small, reject the null hypothesis.",
            "based on which one is larger",
            "reject the null hypothesis",
            "If the p value is small reject the null hypothesis",
        ),
        (
            "test statistic",
            "C compare once you have your test statistic and P value you need to compare "
            "these values to the critical value for your",
            "C compare once you have",
            "critical value for your",
            "your test statistic and P value you need",
        ),
        (
            "false negatives",
            "who DO have cancer will get false negatives 46% of the time. This means "
            "we miss a lot.",
            "who DO have cancer will",
            "we miss a lot",
            "cancer will get false negatives 46% of the time",
        ),
        (
            "significance test",
            "A significance test compares evidence against the null hypothesis, but "
            "some problems will specify",
            "A significance test compares evidence",
            "some problems will specify",
            "A significance test compares evidence against the null hypothesis",
        ),
        (
            "p value",
            "On the other hand, a large p value means the observed result is reasonably "
            "compatible with the null hypothesis.",
            "On the other hand a large p value",
            "compatible with the null hypothesis",
            "large p value means the observed result is reasonably compatible",
        ),
    ],
)
def test_compact_selector_rejects_real_fragmentary_model_edges(
    topic_request: str,
    text: str,
    start_quote: str,
    end_quote: str,
    claim_quote: str,
) -> None:
    report = gemini_segment._plan_to_report(
        _compact_custom_plan(
            request=topic_request,
            start_quote=start_quote,
            end_quote=end_quote,
            claim_quote=claim_quote,
        ),
        [{"cue_id": "cue-0", "start": 0.0, "end": 30.0, "text": text}],
        [],
        {
            "_segment_ignore_caption_case": True,
            "_segment_model_boundary_authoritative": True,
        },
        topic=topic_request,
    )

    assert report.clips == []
    assert any(
        reason.startswith("proposal_0:model_boundary_")
        or reason == "proposal_0:unresolved_weak_end"
        for reason in report.rejected_reasons
    )


def test_explicit_comparison_prompt_distinguishes_primary_and_supporting() -> None:
    _system, user = gemini_segment._boundary_prompts(
        "[0] 00:00 Opportunity cost differs from sunk cost.",
        1,
        "opportunity cost versus sunk cost",
    )

    assert "primary unit must teach every named side" in user
    assert "requested relationship" in user
    assert "supporting units that substantively teach a named side" in user


def test_compact_prompt_defines_every_key_and_demonstrates_exact_edges() -> None:
    _system, user = gemini_segment._boundary_prompts(
        "[0] 00:00 A complete educational statement.",
        1,
        "statistics",
    )

    definitions = {
        "id": "candidate_id",
        "s": "start_line",
        "e": "end_line",
        "sq": "start_quote",
        "eq": "end_quote",
        "cq": "claim_quote",
        "title": "a clear viewer-facing title",
        "obj": "learning_objective",
        "facet": "the narrow subtopic",
        "info": "informativeness",
        "rel": "topic_relevance",
        "imp": "educational_importance",
        "diff": "difficulty",
        "direct": "directly_teaches_topic",
        "sub": "substantive",
        "fact": "factually_grounded",
        "self": "self_contained",
        "stand": "is_standalone",
        "ie": "intent_evidence",
    }
    for key, meaning in definitions.items():
        assert f"- {key} = {meaning}" in user
    normalized = " ".join(user.split())
    assert "caption-line index, not seconds" in normalized
    assert "Its FIRST spoken word" in normalized
    assert "Its LAST spoken word" in normalized
    assert "complete independently understandable spoken sentence or independent clause" in (
        normalized
    )
    assert "never a trailing clause, complement, list item, or clipped completion" in (
        normalized
    )
    assert "never a leading clause, sentence prefix, or unfilled predicate" in normalized
    assert "never asks for a shorter semantic span or clip" in normalized
    assert (
        "cq proves where the teaching is; it does not define the start or end"
        in normalized
    )
    assert "self concerns included context; stand" in normalized
    assert "No video is supplied" in normalized
    assert '"s":18,"e":19' in user
    assert '"sq":"A small p value"' in user
    assert '"eq":"against the null hypothesis."' in user
    assert 'starts after "Welcome back"' in normalized
    assert 'ends before "Next, confidence intervals"' in normalized
    assert 'Do not start at line 32 with "Divide that change"' in normalized


def test_specific_request_prompt_returns_primary_and_complete_supporting_units() -> None:
    _system, user = gemini_segment._boundary_prompts(
        "\n".join([
            "[0] 00:00 First derive five x minus four; its derivative is five.",
            "[1] 00:10 Now try x squared with the limit definition.",
            "[2] 00:20 Expanding and cancelling gives the final derivative two x.",
        ]),
        3,
        (
            "Use the limit definition to derive f'(x)=2x for f(x)=x², including "
            "every algebra step and the final result"
        ),
    )
    normalized = " ".join(user.split())

    assert "every required non-scope constraint" in normalized
    assert "this verbatim exact_request is topic" in normalized.casefold()
    assert "A PRIMARY unit fulfills every required non-scope constraint" in normalized
    assert "A SUPPORTING unit has a substantive educational connection" in normalized
    assert "Return supporting units even when this source contains no primary unit" in normalized
    assert "If one transcript contains six complete related worked examples" in normalized
    assert "return six separate candidates" in normalized
    assert "earlier five-x-minus-four objective" in normalized.casefold()
    assert "begin at the x-squared setup" in normalized.casefold()
    assert "end at the final two-x result" in normalized.casefold()
    assert 'wrong: s=40 and sq="this is simply equal to five"' in (
        normalized.casefold()
    )
    assert "sq=\"so let's say if f of x is equal to x squared\"" in (
        normalized.casefold()
    )
    assert "preserve its first 'so'" in normalized.casefold()
    assert "x-squared-minus-three example is also a different function" in (
        normalized.casefold()
    )
    assert "does not qualify as the primary x-squared unit" in (
        normalized.casefold()
    )
    assert "it may be a supporting unit for the method only" in normalized.casefold()
    assert "same-source breadth for that original prompt" in normalized.casefold()
    assert "return each as its own supporting clip" in normalized.casefold()
    assert "give every clip its actual function and actual result" in normalized.casefold()
    assert "never stop after returning the primary x-squared clip" in normalized.casefold()
    assert "all examples below assume no learner-level restriction" in (
        normalized.casefold()
    )
    assert "never override a supplied learner band" in normalized.casefold()
    assert "one q never" in normalized.casefold()
    assert "whole-span completeness check" in normalized.casefold()


def test_newtons_second_law_prompt_requires_subject_anchoring_and_context() -> None:
    exact_request = (
        "Explain Newton's second law F=ma from intuition to worked examples, "
        "including net force, mass, acceleration, units, and solving for each variable"
    )
    system, user = gemini_segment._boundary_prompts(
        "\n".join([
            "[0] 00:00 Newton's second law says net force equals mass times acceleration.",
            "[1] 00:10 Coulomb's law computes electric force in newtons.",
            "[2] 00:20 Rearrange Coulomb's equation to solve for either charge.",
            "[3] 00:30 The impulse-momentum theorem says F delta t equals delta p.",
            "[4] 00:40 Linear momentum is mass times velocity in SI units.",
            "[5] 00:50 This law tells us force equals mass times acceleration.",
        ]),
        6,
        exact_request,
    )
    normalized = " ".join(f"{system}\n{user}".split()).casefold()

    assert exact_request.casefold() in normalized
    assert (
        "every supporting unit must stay anchored to the request by either (a) "
        "teaching the same named subject, object, or relationship from the request, or "
        "(b) applying an explicitly named technical method or mechanism within "
        "the same subject family"
    ) in normalized
    assert (
        "generic task or format such as explain, calculate, solve for a variable, "
        "show steps, give an example, or state units is never an anchor by itself"
    ) in normalized
    assert (
        'sharing only the head word of a more specific phrase is also not enough: '
        '"force" does not ground "net force"'
    ) in normalized
    assert (
        "omit a coulomb's-law unit that merely mentions force or newtons, computes "
        "net electric force, or rearranges coulomb's equation"
    ) in normalized
    assert (
        "such a unit qualifies only when one complete span explicitly connects the "
        "electric force back to f=ma"
    ) in normalized
    assert "generic algebraic isolation is not that connection" in normalized
    assert "evidence must refer to the same thing and relationship as the request" in normalized
    assert (
        "using the component only as an operand inside a different law or equation "
        "does not teach the requested component"
    ) in normalized
    assert "apply a same-referent test before adding every item" in normalized
    assert (
        "every q must ground the same referent and relationship named by its id, "
        "not merely contain one of its words, symbols, variables, dimensions, or units"
    ) in normalized
    assert (
        "also omit an impulse-momentum theorem unit f delta t = delta p and a "
        "linear-momentum unit p=mv"
    ) in normalized
    assert (
        "they teach different governing relationships and cannot evidence the "
        "newton's-law, mass-in-f=ma, or f=ma-units constraints"
    ) in normalized
    assert (
        'wrong ie: a newton\'s-law id paired with "impulse, f delta t, is equal to '
        'the change in momentum"'
    ) in normalized
    assert (
        'wrong ie: an f=ma mass id paired with "linear momentum is defined as mass '
        'times velocity"'
    ) in normalized
    assert (
        "a complete standalone definition of acceleration as the rate of change of "
        "velocity can be foundational support"
    ) in normalized
    assert "background-detour boundary example" in normalized
    assert (
        "sq, eq, cq, every ie q, title, obj, and facet on the same atomic educational unit"
        in normalized
    )
    assert 'use s=62, sq="you\'re now ready to understand acceleration"' in normalized
    assert 'also wrong: an acceleration candidate uses ie q="units for speed"' in normalized
    assert "this rule still applies for a beginner viewer" in normalized
    assert (
        "do not fold the whole completed prerequisite into the later target unit"
        in normalized
    )
    assert "the semantic opening may not rely on an unresolved referent" in normalized
    assert (
        'do not start at line 51 with "this law tells us"'
    ) in normalized
    assert (
        "the original request and video title are not spoken context"
    ) in normalized
    assert (
        "title, obj, facet, cq, ie, and a previous clip are metadata and do not supply "
        "that spoken context"
    ) in normalized
    assert "begin sq at \"newton's second law\"" in normalized
    assert (
        "a worked numerical example for a dimensioned quantity cannot end at a "
        "unitless result"
    ) in normalized
    assert "include the contiguous spoken unit" in normalized
    assert "end eq after the unit" in normalized
    assert gemini_segment.PRO_BOUNDARY_PROFILE == "pro_boundary_v10"


def test_boundary_prompt_stays_transcript_only_when_video_is_requested() -> None:
    system, user = gemini_segment._boundary_prompts(
        "[0] 00:00 This curve approaches zero as x increases.",
        1,
        "limits",
        learner_level="advanced",
        video_grounded=True,
    )
    prompt = f"{system}\n{user}".casefold()

    assert "you receive transcript text only" in prompt
    assert (
        "no video, image, audio file, frames, thumbnails, or visual metadata are attached"
        in prompt
    )
    assert "judge only the supplied transcript" in prompt
    assert "only the supplied spoken transcript" in prompt
    assert "inspect the audio and visual streams jointly" not in prompt
    assert "attached-video grounding" not in prompt
    assert "omit material that requires an unseen visual" in prompt
    assert "factually_grounded" in prompt
    assert "current level is advanced" in prompt
    assert "level fit is a gemini selection eligibility rule" in prompt
    assert "difficulty band 0.60-1.00" in prompt
    assert "genuinely advanced reasoning or application" in prompt
    assert "obey its eligibility band" in prompt
    assert "sq=start_quote" in prompt
    assert "eq=end_quote" in prompt
    assert "cq=claim_quote" in prompt
    assert "q is an exact consecutive 5-16 word transcript quote" in prompt


@pytest.mark.parametrize(
    "profile",
    [gemini_segment.FLASH_SPLIT_PROFILE, gemini_segment.PRO_BOUNDARY_PROFILE],
)
def test_boundary_selector_never_attaches_video_even_when_requested(
    monkeypatch,
    profile,
) -> None:
    calls: list[dict] = []
    reservations: list[dict] = []

    def fake_generate(system, user, schema, **kwargs):
        calls.append({"system": system, "user": user, "schema": schema, **kwargs})
        return SimpleNamespace(
            text=(
                '{"request_intent":{"exact_request":"photosynthesis",'
                '"constraints":[{"constraint_id":"subject","kind":"subject",'
                '"source_phrase":"photosynthesis","requirement":'
                '"Teach photosynthesis"}]},"topics":[]}'
            ),
            telemetry={
                "model": kwargs["model"],
                "prompt_tokens": 100,
                "candidate_tokens": 10,
                "total_tokens": 110,
            },
        )

    monkeypatch.setattr(gemini_client, "generate_json_v3", fake_generate)
    monkeypatch.setattr(
        gemini_client,
        "count_request_tokens",
        lambda *_args, **_kwargs: 1_000,
    )

    def reserve(**kwargs):
        reservations.append(kwargs)
        return {}

    result = gemini_segment.run_segment_profile(
        {
            "segments": [{
                "start": 0.0,
                "end": 5.0,
                "text": "Plants convert light into stored chemical energy.",
            }],
            "words": [],
            "duration": 600.0,
        },
        {
            "_segment_video_grounding_required": True,
            "_segment_video_url": (
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
            ),
            "_segment_media_resolution": "low",
            "_knowledge_level": "beginner",
            "_segment_budget_reserve": reserve,
        },
        profile,
        topic="photosynthesis",
    )

    assert result.error is None
    assert len(calls) == 1
    [call] = calls
    contents = call["user"]
    assert isinstance(contents, str)
    assert "Transcript (1 lines" in contents
    assert "current level is beginner" in contents
    assert "difficulty band 0.00-0.40" in contents
    assert "assume no topic-specific background" in contents
    assert "never prepend a separately complete prerequisite/background lesson" in contents
    assert "youtube.com" not in contents
    assert call["media_resolution"] is None
    assert call.get("estimated_media_tokens", 0) == 0
    if profile == gemini_segment.FLASH_SPLIT_PROFILE:
        assert call["max_retries"] == 1
        assert call["retry_status_codes"] == frozenset({503})
    else:
        assert call["max_retries"] == 0
        assert call["retry_status_codes"] is None
    [reservation] = reservations
    prompt_text = f"{call['system']}\n\n{contents}"
    schema_bytes = len(json.dumps(
        gemini_segment._CompactBoundaryPlan.model_json_schema(),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8"))
    text_estimate = math.ceil((len(prompt_text) + schema_bytes) / 3) + 1_000
    assert reservation["estimated_input_tokens"] == text_estimate


def test_required_video_grounding_flag_cannot_block_transcript_only_dispatch(
    monkeypatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        gemini_client,
        "generate_json_v3",
        lambda _system, user, _schema, **_kwargs: (
            calls.append(user)
            or SimpleNamespace(
                text=(
                    '{"request_intent":{"exact_request":"photosynthesis",'
                    '"constraints":[{"constraint_id":"subject","kind":"subject",'
                    '"source_phrase":"photosynthesis","requirement":'
                    '"Teach photosynthesis"}]},"topics":[]}'
                ),
                telemetry={},
            )
        ),
    )

    result = gemini_segment.run_segment_profile(
        {
            "segments": [{
                "start": 0.0,
                "end": 5.0,
                "text": "Plants convert light into stored chemical energy.",
            }],
            "words": [],
        },
        {"_segment_video_grounding_required": True},
        gemini_segment.FLASH_SPLIT_PROFILE,
        topic="photosynthesis",
    )

    assert len(calls) == 1
    assert isinstance(calls[0], str)
    assert result.clips == []
    assert result.error is None


def test_video_grounding_accepts_duration_sec_and_never_underbounds_last_cue() -> None:
    assert gemini_segment._video_grounding_duration_seconds(
        {"duration_sec": 600.2501},
        [{"start": 0.0, "end": 5.0, "text": "A complete lesson."}],
    ) == pytest.approx(600.2501)
    assert gemini_segment._video_grounding_duration_seconds(
        {"duration_sec": 4.0},
        [{"start": 0.0, "end": 5.25, "text": "A complete lesson."}],
    ) == pytest.approx(5.25)


def test_long_video_metadata_does_not_create_media_budget_or_block_dispatch(
    monkeypatch,
) -> None:
    context = GenerationContext("slow", generation_id="selector-long-context")
    calls: list[object] = []
    monkeypatch.setattr(
        gemini_client,
        "generate_json_v3",
        lambda *_args, **_kwargs: (
            calls.append(_kwargs)
            or SimpleNamespace(
                text=(
                    '{"request_intent":{"exact_request":"photosynthesis",'
                    '"constraints":[{"constraint_id":"subject","kind":"subject",'
                    '"source_phrase":"photosynthesis","requirement":'
                    '"Teach photosynthesis"}]},"topics":[]}'
                ),
                telemetry={
                    "model": _kwargs["model"],
                    "prompt_tokens": 100,
                    "candidate_tokens": 10,
                    "total_tokens": 110,
                },
            )
        ),
    )
    monkeypatch.setattr(
        gemini_client,
        "count_request_tokens",
        lambda *_args, **_kwargs: 1_000,
    )

    result = gemini_segment.run_segment_profile(
        {
            "segments": [{
                "start": 0.0,
                "end": 5.0,
                "text": "Plants convert light into stored chemical energy.",
            }],
            "words": [],
            "duration": 2_001.0,
        },
        {
            "_segment_video_grounding_required": True,
            "_segment_video_url": (
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
            ),
            "_segment_media_resolution": "low",
            "_segment_operation": "pro_authoritative",
            "_segment_budget_reserve": context.reserve_gemini_call,
            "_segment_budget_reconcile": context.reconcile_gemini_call,
        },
        gemini_segment.PRO_BOUNDARY_PROFILE,
        topic="photosynthesis",
    )

    assert len(calls) == 1
    assert calls[0]["media_resolution"] is None
    assert calls[0].get("estimated_media_tokens", 0) == 0
    assert result.clips == []
    assert result.error is None
    budget = context.budget.snapshot()["gemini"]
    assert budget["selector_calls"] == 1
    assert budget["inflight_reserved_cost_usd"] == 0.0


def test_long_preferred_video_url_dispatches_one_pro_transcript_only_call(
    monkeypatch,
) -> None:
    context = GenerationContext("slow", generation_id="selector-long-preferred")
    calls: list[dict] = []

    def fake_generate(_system, user, _schema, **kwargs):
        calls.append({"user": user, **kwargs})
        return SimpleNamespace(
            text=(
                '{"request_intent":{"exact_request":"photosynthesis",'
                '"constraints":[{"constraint_id":"subject","kind":"subject",'
                '"source_phrase":"photosynthesis","requirement":'
                '"Teach photosynthesis"}]},"topics":[]}'
            ),
            telemetry={
                "model": kwargs["model"],
                "prompt_tokens": 40_000,
                "candidate_tokens": 200,
                "thought_tokens": 100,
                "total_tokens": 40_300,
            },
        )

    monkeypatch.setattr(gemini_client, "generate_json_v3", fake_generate)
    monkeypatch.setattr(
        gemini_client,
        "count_request_tokens",
        lambda *_args, **_kwargs: 40_000,
    )
    segments = [
        {
            "cue_id": f"supadata-cue-{index}",
            "start": index * 7.0,
            "end": (index + 1) * 7.0,
            "text": (
                "Photosynthesis section "
                f"{index} explains how captured light energy supports electron "
                "transport, a proton gradient, and ATP synthesis."
            ),
        }
        for index in range(600)
    ]

    result = gemini_segment.run_segment_profile(
        {
            "segments": segments,
            "words": [],
            "duration": 4_200.0,
            "source": "supadata",
        },
        {
            "_segment_video_grounding_required": False,
            "_segment_video_url": (
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
            ),
            "_segment_media_resolution": "low",
            "_segment_operation": "pro_authoritative",
            "_segment_budget_reserve": context.reserve_gemini_call,
            "_segment_budget_reconcile": context.reconcile_gemini_call,
        },
        gemini_segment.PRO_BOUNDARY_PROFILE,
        topic="photosynthesis",
    )

    assert result.error is None
    assert len(calls) == 1
    [call] = calls
    assert call["model"] == "gemini-3.1-pro-preview"
    assert isinstance(call["user"], str)
    assert call["media_resolution"] is None
    assert "inspect the audio and visual streams jointly" not in call["user"]
    assert "Photosynthesis section 0 explains" in call["user"]
    assert "Photosynthesis section 599 explains" in call["user"]
    assert len(result.calls) == 1
    assert result.calls[0]["video_grounded"] is False
    assert result.calls[0]["reserved_input_tokens"] < 200_000
    assert result.calls[0]["reserved_cost_usd"] < 0.70
    assert "video_grounding_fallback_reason" not in result.calls[0]
    assert "skipped_media_tokens" not in result.calls[0]
    budget = context.budget.snapshot()["gemini"]
    assert budget["selector_calls"] == 1
    assert budget["committed_cost_usd"] == pytest.approx(
        (40_000 * 2.0 + 300 * 12.0) / 1_000_000.0
    )
    assert budget["committed_cost_usd"] < budget["cost_limit_usd"]
    assert budget["inflight_reserved_cost_usd"] == 0.0


def test_no_space_unicode_transcript_cannot_cross_long_context_price_tier(
    monkeypatch,
) -> None:
    context = GenerationContext("slow", generation_id="selector-unicode-tier")
    calls: list[object] = []
    monkeypatch.setattr(
        gemini_client,
        "generate_json_v3",
        lambda *_args, **_kwargs: calls.append(True),
    )

    def fail_count(*_args, **_kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(gemini_client, "count_request_tokens", fail_count)

    result = gemini_segment.run_segment_profile(
        {
            "segments": [{
                "start": 0.0,
                "end": 60.0,
                "text": "統" * 67_000,
            }],
            "words": [],
            "duration": 60.0,
            "source": "supadata",
        },
        {
            "_segment_video_grounding_required": False,
            "_segment_operation": "pro_authoritative",
            "_segment_budget_reserve": context.reserve_gemini_call,
            "_segment_budget_reconcile": context.reconcile_gemini_call,
        },
        gemini_segment.PRO_BOUNDARY_PROFILE,
        topic="statistics",
    )

    assert calls == []
    assert result.clips == []
    assert result.classification_reasons == [
        "request_failure:GeminiTokenPreflightError"
    ]
    assert result.calls[0]["retryable"] is False
    assert result.calls[0]["token_preflight_failed"] is True
    budget = context.budget.snapshot()["gemini"]
    assert budget["selector_calls"] == 0
    assert budget["cost_exposure_usd"] == 0.0


def test_exact_token_preflight_admits_affordable_long_unicode_text(
    monkeypatch,
) -> None:
    reservations: list[dict] = []
    token_counts: list[dict] = []
    calls: list[dict] = []

    def count_tokens(system, user_text, schema, **kwargs):
        token_counts.append({
            "system": system,
            "user_text": user_text,
            "schema": schema,
            **kwargs,
        })
        return 60_000

    def generate(_system, _user, _schema, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            text='{"topics": []}',
            telemetry={
                "model": kwargs["model"],
                "prompt_tokens": 60_000,
                "candidate_tokens": 10,
                "total_tokens": 60_010,
            },
        )

    monkeypatch.setattr(gemini_client, "count_request_tokens", count_tokens)
    monkeypatch.setattr(gemini_client, "generate_json_v3", generate)

    parsed, _call = gemini_segment._call_model(
        "system",
        "統" * 67_000,
        gemini_segment._BoundaryPlan,
        model="gemini-3.1-pro-preview",
        thinking_level="medium",
        max_output_tokens=6_000,
        timeout_s=30.0,
        deadline_monotonic=time.monotonic() + 10.0,
        operation="pro_authoritative",
        prompt_version=gemini_segment.PRO_BOUNDARY_PROFILE,
        cancelled=None,
        budget_reserve=lambda **payload: reservations.append(payload) or {},
        max_retries=0,
    )

    assert parsed.topics == []
    assert len(token_counts) == len(calls) == 1
    assert token_counts[0]["model"] == "gemini-3.1-pro-preview"
    assert token_counts[0]["schema"] is gemini_segment._BoundaryPlan
    assert reservations[0]["estimated_input_tokens"] == 60_000


@pytest.mark.parametrize("exact_tokens", [199_500, 200_000])
def test_exact_token_preflight_keeps_affordable_tier_through_its_exact_boundary(
    monkeypatch,
    exact_tokens,
) -> None:
    context = GenerationContext("slow", generation_id="selector-exact-tier")
    reservations: list[dict] = []
    calls: list[dict] = []

    monkeypatch.setattr(
        gemini_client,
        "count_request_tokens",
        lambda *_args, **_kwargs: exact_tokens,
    )

    def generate(_system, _user, _schema, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            text='{"topics": []}',
            telemetry={
                "model": kwargs["model"],
                "prompt_tokens": exact_tokens,
                "candidate_tokens": 10,
                "thought_tokens": 10,
                "total_tokens": exact_tokens + 20,
            },
        )

    monkeypatch.setattr(gemini_client, "generate_json_v3", generate)

    parsed, _call = gemini_segment._call_model(
        "system",
        "x" * 200_000,
        gemini_segment._BoundaryPlan,
        model="gemini-3.1-pro-preview",
        thinking_level="medium",
        max_output_tokens=100,
        timeout_s=30.0,
        deadline_monotonic=time.monotonic() + 10.0,
        operation="pro_authoritative",
        prompt_version=gemini_segment.PRO_BOUNDARY_PROFILE,
        cancelled=None,
        budget_reserve=lambda **payload: (
            reservations.append(payload)
            or context.reserve_gemini_call(**payload)
        ),
        budget_reconcile=context.reconcile_gemini_call,
        max_retries=0,
    )

    assert parsed.topics == []
    assert len(calls) == len(reservations) == 1
    assert reservations[0]["estimated_input_tokens"] == exact_tokens
    assert context.budget.snapshot()["gemini"]["committed_cost_usd"] == pytest.approx(
        (exact_tokens * 2.0 + 20 * 12.0) / 1_000_000.0
    )


def test_exact_preflight_prevents_high_entropy_text_from_exceeding_fast_ceiling(
    monkeypatch,
) -> None:
    context = GenerationContext("fast", generation_id="selector-high-entropy")
    generated: list[object] = []
    monkeypatch.setattr(
        gemini_client,
        "count_request_tokens",
        lambda *_args, **_kwargs: 198_000,
    )
    monkeypatch.setattr(
        gemini_client,
        "generate_json_v3",
        lambda *_args, **_kwargs: generated.append(True),
    )

    with pytest.raises(ProviderBudgetExceededError):
        gemini_segment._call_model(
            "system",
            "x" * 70_000,
            gemini_segment._BoundaryPlan,
            model="gemini-3.1-pro-preview",
            thinking_level="medium",
            max_output_tokens=6_000,
            timeout_s=30.0,
            deadline_monotonic=time.monotonic() + 10.0,
            operation="pro_authoritative",
            prompt_version=gemini_segment.PRO_BOUNDARY_PROFILE,
            cancelled=None,
            budget_reserve=context.reserve_gemini_call,
            budget_reconcile=context.reconcile_gemini_call,
            max_retries=0,
        )

    assert generated == []
    budget = context.budget.snapshot()["gemini"]
    assert budget["selector_calls"] == 0
    assert budget["cost_exposure_usd"] == 0.0


def test_preferred_video_url_failure_never_dispatches_a_media_retry(
    monkeypatch,
) -> None:
    context = GenerationContext("slow", generation_id="selector-text-failure")
    calls: list[dict] = []

    def fail_generate(*_args, **kwargs):
        calls.append(kwargs)
        raise gemini_client.GeminiTransportError(
            "provider timed out",
            gemini_client.GeminiCallTelemetry(
                model=str(kwargs["model"]),
                operation="pro_authoritative",
                prompt_version=gemini_segment.PRO_BOUNDARY_PROFILE,
                thinking_level="medium",
                latency_ms=10.0,
                retries=0,
                finish_reason=None,
                prompt_tokens=None,
                candidate_tokens=None,
                thought_tokens=None,
                total_tokens=None,
                provider_error_type="ServerError",
                provider_status_code=504,
                retryable=True,
            ),
        )

    monkeypatch.setattr(gemini_client, "generate_json_v3", fail_generate)

    result = gemini_segment.run_segment_profile(
        {
            "segments": [{
                "start": 0.0,
                "end": 5.0,
                "text": "Plants convert light into stored chemical energy.",
            }],
            "words": [],
            "duration": 4_200.0,
        },
        {
            "_segment_video_grounding_required": False,
            "_segment_video_url": (
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
            ),
            "_segment_media_resolution": "low",
            "_segment_operation": "pro_authoritative",
            "_segment_budget_reserve": context.reserve_gemini_call,
            "_segment_budget_reconcile": context.reconcile_gemini_call,
        },
        gemini_segment.PRO_BOUNDARY_PROFILE,
        topic="photosynthesis",
    )

    assert result.error is not None
    assert len(calls) == 1
    assert calls[0]["model"] == "gemini-3.1-pro-preview"
    assert calls[0]["max_retries"] == 0
    assert calls[0]["media_resolution"] is None
    assert "video_grounding_fallback_reason" not in result.calls[0]
    assert "skipped_media_tokens" not in result.calls[0]
    assert result.calls[0]["provider_status_code"] == 504
    budget = context.budget.snapshot()["gemini"]
    assert budget["selector_calls"] == 1
    assert budget["inflight_reserved_cost_usd"] == 0.0


def test_video_grounded_flash_transport_failure_never_dispatches_a_second_model(
    monkeypatch,
) -> None:
    calls = []

    def fail_once(*_args, **kwargs):
        calls.append(kwargs["model"])
        model = str(kwargs["model"])
        raise gemini_client.GeminiTransportError(
            "provider overloaded",
            gemini_client.GeminiCallTelemetry(
                model=model,
                operation="flash_boundary_selector",
                prompt_version=gemini_segment.FLASH_SPLIT_PROFILE,
                thinking_level="low",
                latency_ms=5.0,
                retries=0,
                finish_reason=None,
                prompt_tokens=None,
                candidate_tokens=None,
                thought_tokens=None,
                total_tokens=None,
                provider_error_type="ServerError",
                provider_status_code=503,
                retryable=True,
                error_history=({
                    "provider_error_type": "ServerError",
                    "provider_status_code": 503,
                    "retryable": True,
                },),
            ),
        )

    monkeypatch.setattr(gemini_client, "generate_json_v3", fail_once)
    result = gemini_segment.run_segment_profile(
        {
            "segments": [{
                "start": 0.0,
                "end": 5.0,
                "text": "Plants convert light into stored chemical energy.",
            }],
            "words": [],
        },
        {
            "_segment_video_grounding_required": True,
            "_segment_video_url": (
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
            ),
            "_segment_media_resolution": "low",
        },
        gemini_segment.FLASH_SPLIT_PROFILE,
        topic="photosynthesis",
    )

    assert calls == [gemini_segment.config.SEGMENT_FLASH_MODEL]
    assert result.clips == []
    assert result.error is not None


def test_selector_reconciles_actual_usage_before_conversion(
    monkeypatch,
) -> None:
    context = GenerationContext("fast", generation_id="selector-reconcile")
    monkeypatch.setattr(
        gemini_client,
        "generate_json_v3",
        lambda *_args, **_kwargs: SimpleNamespace(
            text=(
                '{"request_intent":{"exact_request":"test topic","constraints":['
                '{"constraint_id":"subject","kind":"subject",'
                '"source_phrase":"test topic","requirement":"Teach the test topic"}]},'
                '"topics":[]}'
            ),
            telemetry={
                "model": "gemini-3.5-flash",
                "prompt_tokens": 2_000,
                "candidate_tokens": 20,
                "thought_tokens": 10,
                "total_tokens": 2_030,
            },
        ),
    )

    parsed, telemetry = gemini_segment._call_model(
        "system",
        "user",
        gemini_segment._CompactBoundaryPlan,
        model="gemini-3.5-flash",
        thinking_level="low",
        max_output_tokens=8_192,
        timeout_s=5.0,
        deadline_monotonic=time.monotonic() + 2.0,
        operation="flash_boundary_selector",
        prompt_version="test-selector",
        cancelled=None,
        budget_reserve=context.reserve_gemini_call,
        budget_reconcile=context.reconcile_gemini_call,
        max_retries=0,
    )

    assert parsed.topics == []
    assert isinstance(telemetry["gemini_reservation_id"], int)
    budget = context.budget.snapshot()["gemini"]
    assert budget["inflight_reserved_cost_usd"] == 0.0
    assert budget["committed_cost_usd"] == pytest.approx(
        (2_000 * 1.5 + 30 * 9.0) / 1_000_000.0
    )


def test_flash_profile_retries_one_confirmed_503_on_the_same_model(monkeypatch) -> None:
    captured: dict[str, object] = {}
    empty_plan = gemini_segment._CompactBoundaryPlan(
        request_intent={
            "exact_request": "photosynthesis",
            "constraints": [{
                "constraint_id": "subject",
                "kind": "subject",
                "source_phrase": "photosynthesis",
                "requirement": "Teach photosynthesis",
            }],
        },
        topics=[],
    )

    def fake_call_model(*_args, **kwargs):
        captured.update(kwargs)
        return empty_plan, {
            "model": "gemini-3.5-flash",
            "prompt_tokens": 10,
            "candidate_tokens": 10,
            "total_tokens": 20,
        }

    monkeypatch.setattr(gemini_segment, "_call_model", fake_call_model)
    gemini_segment.run_segment_profile(
        {
            "segments": [{
                "start": 0.0,
                "end": 5.0,
                "text": "Plants convert light into stored chemical energy.",
            }],
            "words": [],
        },
        {},
        gemini_segment.PRODUCTION_FLASH_PROFILE,
        topic="photosynthesis",
    )

    assert captured["max_retries"] == 1
    assert captured["retry_status_codes"] == frozenset({503})
    assert captured["timeout_s"] == 45.0


def test_flash_selector_fails_over_immediately_after_one_503(monkeypatch) -> None:
    models: list[str] = []
    monkeypatch.setattr(
        gemini_segment.config,
        "SEGMENT_FLASH_MODEL",
        "gemini-3.5-flash",
    )

    def fake_generate(*_args, **kwargs):
        model = str(kwargs["model"])
        models.append(model)
        assert kwargs["max_retries"] == 0
        if model == "gemini-3.5-flash":
            raise gemini_client.GeminiTransportError(
                "provider overloaded",
                gemini_client.GeminiCallTelemetry(
                    model=model,
                    operation="flash_boundary_selector",
                    prompt_version="flash_split_v3",
                    thinking_level="low",
                    latency_ms=5.0,
                    retries=0,
                    finish_reason=None,
                    prompt_tokens=None,
                    candidate_tokens=None,
                    thought_tokens=None,
                    total_tokens=None,
                    provider_error_type="ServerError",
                    provider_status_code=503,
                    retryable=True,
                    error_history=({
                        "provider_error_type": "ServerError",
                        "provider_status_code": 503,
                        "retryable": True,
                    },),
                ),
            )
        return SimpleNamespace(
            text=(
                '{"request_intent":{"exact_request":"photosynthesis",'
                '"constraints":[{"constraint_id":"subject","kind":"subject",'
                '"source_phrase":"photosynthesis","requirement":'
                '"Teach photosynthesis"}]},"topics":[]}'
            ),
            telemetry={
                "model": model,
                "prompt_tokens": 100,
                "candidate_tokens": 10,
                "thought_tokens": 5,
                "total_tokens": 115,
                "retries": 0,
            },
        )

    monkeypatch.setattr(gemini_client, "generate_json_v3", fake_generate)

    parsed, telemetry = gemini_segment._call_model(
        "system",
        "user",
        gemini_segment._CompactBoundaryPlan,
        model="gemini-3.5-flash",
        thinking_level="low",
        max_output_tokens=6_000,
        timeout_s=10.0,
        deadline_monotonic=time.monotonic() + 10.0,
        operation="flash_boundary_selector",
        prompt_version="flash_split_v3",
        cancelled=None,
        max_retries=0,
        failover_model="gemini-3.1-flash-lite",
    )

    assert parsed.topics == []
    assert models == ["gemini-3.5-flash", "gemini-3.1-flash-lite"]
    assert telemetry["retries"] == 1
    assert telemetry["failover_reason"] == "primary_transient_5xx_failover"


def test_selector_releases_non_dispatched_failure_reservation(
    monkeypatch,
) -> None:
    context = GenerationContext("fast", generation_id="selector-not-dispatched")

    def fail_before_dispatch(*_args, **_kwargs):
        error = RuntimeError("provider capacity unavailable")
        error.telemetry = {
            "model": "gemini-3.5-flash",
            "dispatched": False,
        }
        raise error

    monkeypatch.setattr(gemini_client, "generate_json_v3", fail_before_dispatch)

    with pytest.raises(gemini_segment._ModelCallError) as caught:
        gemini_segment._call_model(
            "system",
            "user",
            gemini_segment._CompactBoundaryPlan,
            model="gemini-3.5-flash",
            thinking_level="low",
            max_output_tokens=6_000,
            timeout_s=5.0,
            deadline_monotonic=time.monotonic() + 2.0,
            operation="flash_boundary_selector",
            prompt_version="test-selector",
            cancelled=None,
            budget_reserve=context.reserve_gemini_call,
            budget_reconcile=context.reconcile_gemini_call,
            max_retries=0,
        )

    telemetry = caught.value.telemetry
    assert telemetry["dispatched"] is False
    assert context.budget.snapshot()["gemini"]["inflight_reserved_cost_usd"] == 0.0
    assert context.budget.snapshot()["gemini"]["committed_cost_usd"] == 0.0

    # The later usage-recording path reconciles the same reservation again.
    context.record_gemini(
        attempt=1,
        model_used="gemini-3.5-flash",
        quality_degraded=False,
        usage=telemetry,
        status_code=None,
        error_code="model_call_failed",
    )
    budget = context.budget.snapshot()["gemini"]
    assert budget["inflight_reserved_cost_usd"] == 0.0
    assert budget["committed_cost_usd"] == 0.0


def test_short_end_quote_stops_before_same_cue_outro() -> None:
    text = (
        "Studying photosynthesis explains how plants convert light into stored chemical "
        "energy. Thanks for watching and don't forget to subscribe."
    )
    proposal = _proposal().model_copy(update={
        "start_quote": "Studying photosynthesis explains",
        "end_quote": "energy",
        "topic_evidence_quote": (
            "photosynthesis explains how plants convert light into stored chemical energy"
        ),
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        [{"cue_id": "short-edge", "start": 0.0, "end": 12.0, "text": text}],
        [],
        {"_segment_ignore_caption_case": True},
        topic="photosynthesis",
    )

    assert report.rejected_reasons == []
    assert report.clips[0]["_clip_text"].endswith("chemical energy")
    assert "Thanks for watching" not in report.clips[0]["_clip_text"]


def test_inline_next_topic_tail_is_trimmed_at_the_complete_claim() -> None:
    text = (
        "Carbon dioxide and water are converted into glucose, and the plant releases "
        "oxygen during photosynthesis now let's talk about chloroplast structure."
    )
    proposal = _proposal().model_copy(update={
        "start_quote": "Carbon dioxide and water are converted",
        "end_quote": "oxygen during photosynthesis now let's",
        "topic_evidence_quote": (
            "the plant releases oxygen during photosynthesis"
        ),
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        [{"cue_id": "mixed-tail", "start": 0.0, "end": 14.0, "text": text}],
        [],
        {"_segment_ignore_caption_case": True},
        topic="photosynthesis",
    )

    assert report.rejected_reasons == []
    assert report.clips[0]["_clip_text"].endswith(
        "the plant releases oxygen during photosynthesis"
    )
    assert "now let's" not in report.clips[0]["_clip_text"].casefold()


@pytest.mark.parametrize(
    "claim_quote",
    [
        "if data gives us strong evidence that the hypothesis is wrong",
        "hypothesis but not exactly the same then the best",
    ],
)
def test_recap_evidence_rejects_the_full_production_like_coarse_span(
    claim_quote: str,
) -> None:
    texts = [
        "A hypothesis gives us a claim that an experiment can test.",
        "Drugs A and B produce measurements we can compare.",
        "The first experiment gives evidence against the hypothesis.",
        "Repeated results make that evidence stronger.",
        "The observed difference is not what the hypothesis predicts.",
        "That makes the first result unlikely under the hypothesis.",
        "every time we do the experiment we get the opposite result so we can "
        "confidently reject this hypothesis BAM",
        "now let's imagine we had two more drugs C and D just like before",
        "Their measurements are close but not identical.",
        "That similarity does not prove that the hypothesis is true.",
        "The second experiment therefore supports a weaker conclusion.",
        "the best we can do is fail to reject the hypothesis small BAM",
        "to summarize what we've covered so far we can create a hypothesis and if "
        "data gives us strong evidence that the hypothesis is wrong then we can "
        "reject the hypothesis but when we have data that is similar to the",
        "hypothesis but not exactly the same then the best we can do is fail to "
        "reject the hypothesis",
    ]
    starts = [0.08, 38.0, 76.0, 114.0, 152.0, 190.0, 228.0, 266.0,
              304.0, 342.0, 380.0, 418.0, 450.0, 500.0]
    ends = [38.0, 76.0, 114.0, 152.0, 190.0, 228.0, 266.0, 304.0,
            342.0, 380.0, 418.0, 450.0, 500.0, 523.78]
    segments = [
        {"cue_id": f"cue-{index}", "start": starts[index], "end": ends[index], "text": text}
        for index, text in enumerate(texts)
    ]
    plan = _compact_plan(
        exact_request="hypothesis testing",
        constraints=[{
            "constraint_id": "subject",
            "kind": "subject",
            "source_phrase": "hypothesis testing",
            "requirement": "Teach hypothesis testing",
        }],
        evidence=[{
            "constraint_id": "subject",
            "evidence_quote": claim_quote,
        }],
    )
    plan.topics[0] = plan.topics[0].model_copy(update={
        "candidate_id": "hypothesis-testing-summary",
        "start_line": 0,
        "end_line": 13,
        "start_quote": "A hypothesis gives us a claim",
        "end_quote": "do is fail to reject the hypothesis",
        "claim_quote": claim_quote,
        "title": "Rejecting and failing to reject a hypothesis",
        "learning_objective": "Explain when evidence rejects a hypothesis",
        "facet": "hypothesis decisions",
    })

    report = gemini_segment._plan_to_report(
        plan,
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic="hypothesis testing",
    )

    assert report.clips == []
    assert report.rejected_reasons == [
        "proposal_0:recap_evidence"
    ]
    assert segments[-1]["end"] - segments[0]["start"] > 500.0


def test_recap_inside_one_coarse_cue_is_a_hard_end_for_prior_evidence() -> None:
    text = (
        "Competitive inhibitors occupy the active site and prevent the substrate from "
        "binding. To summarize what we've covered so far, inhibitors can change enzyme "
        "activity through several distinct mechanisms."
    )
    proposal = _proposal().model_copy(update={
        "candidate_id": "competitive-inhibition",
        "start_quote": "Competitive inhibitors occupy the active site",
        "end_quote": "through several distinct mechanisms",
        "title": "How competitive inhibition works",
        "learning_objective": "Explain how competitive inhibitors block substrates",
        "facet": "competitive inhibition",
        "topic_evidence_quote": (
            "Competitive inhibitors occupy the active site and prevent the substrate"
        ),
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        [{"cue_id": "coarse-recap-cue", "start": 0.0, "end": 410.0, "text": text}],
        [],
        {"_segment_ignore_caption_case": True},
        topic="competitive inhibition",
    )

    assert report.rejected_reasons == []
    assert report.clips[0]["_clip_text"].rstrip(".") == (
        "Competitive inhibitors occupy the active site and prevent the substrate from "
        "binding"
    )
    assert "summarize" not in report.clips[0]["_clip_text"].casefold()


def test_enumerated_meta_topics_trim_the_exact_production_statquest_span() -> None:
    texts = [
        (
            "5 percent of the time we do the experiment we will get a p-value less "
            "than 0.05 aka a false positive note if it is extremely important that "
            "we are correct when we say the drugs are different then we can use a "
            "smaller threshold like 0.00001"
        ),
        (
            "using a threshold of 0.00001 means we would only get a false positive "
            "once every 100 000 experiments likewise if it's not that important for "
            "example if we're trying to decide if the ice cream truck will arrive on "
            "time then we can use a larger threshold like 0.2 using a threshold of "
            "0.2 means we are willing to get a false positive two times out of 10. "
            "that said the most common threshold is 0.05 because trying to reduce the "
            "number of false positives below 5 often costs more than it's worth"
        ),
        (
            "so if we calculate a p-value for this experiment and the p-value is less "
            "than 0.05 then we will decide that drug a is different from drug b that "
            "said the p-value is actually 0.24 so we are not confident that drug a is "
            "different from drug b bam okay before we're done let me say two more "
            "things about p-values unfortunately the first thing i want to say is just "
            "more terminology in fancy statistical lingo the idea of trying to "
            "determine if these drugs are the same or not is called hypothesis testing"
        ),
        (
            "the null hypothesis is that the drugs are the same and the p-value helps "
            "us decide if we should reject the null hypothesis or not small bam okay "
            "now that we have that fancy terminology out of the way the second thing i "
            "want to say is way more interesting while a small p-value helps us decide "
            "if drug a is different from drug b it does not tell us how different they "
            "are in other words you can have a small p-value regardless of the size of "
            "difference between drug a and drug b"
        ),
        (
            "the difference can be tiny or huge for example this experiment gives us "
            "a relatively large p-value 0.24 even though there is a six-point "
            "difference between drug a and drug b in contrast this experiment which "
            "involves a lot more people gives us a smaller p-value 0.04 even though "
            "given the new data there is a one point difference between drug a and "
            "drug b in summary a small p-value does not imply that the effect size or "
            "difference between drug a and drug b is large double bam hooray"
        ),
    ]
    starts = [447.84, 471.599, 517.76, 564.32, 604.399]
    ends = [469.8, 520.64, 567.04, 608.32, 653.16]
    segments = [
        {
            "cue_id": f"statquest-{index}",
            "start": starts[index],
            "end": ends[index],
            "text": text,
        }
        for index, text in enumerate(texts)
    ]
    evidence = "p-value helps us decide if we should reject"
    plan = _compact_plan(
        exact_request="hypothesis testing p-value",
        constraints=[
            {
                "constraint_id": "hypothesis-testing",
                "kind": "subject",
                "source_phrase": "hypothesis testing",
                "requirement": "Define hypothesis testing",
            },
            {
                "constraint_id": "p-value",
                "kind": "subject",
                "source_phrase": "p-value",
                "requirement": "Explain the p-value decision",
            },
        ],
        evidence=[
            {
                "constraint_id": "hypothesis-testing",
                "evidence_quote": "same or not is called hypothesis testing",
            },
            {
                "constraint_id": "p-value",
                "evidence_quote": evidence,
            },
        ],
    )
    plan.topics[0] = plan.topics[0].model_copy(update={
        "candidate_id": "hypothesis-testing-null",
        "start_line": 0,
        "end_line": 4,
        "start_quote": "5 percent of the time we do the experiment",
        "end_quote": "difference between drug a and drug b is large double bam hooray",
        "claim_quote": evidence,
        "title": "Hypothesis testing and the null hypothesis",
        "learning_objective": (
            "Define hypothesis testing and explain how the p-value relates to the "
            "null hypothesis"
        ),
        "facet": "hypothesis-testing null",
    })

    report = gemini_segment._plan_to_report(
        plan,
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic="hypothesis testing p-value",
    )

    assert report.rejected_reasons == []
    assert len(report.clips) == 1
    clip_text = report.clips[0]["_clip_text"].casefold()
    assert "hypothesis testing" in clip_text
    assert "p-value helps us decide if we should reject" in clip_text
    assert "5 percent of the time" not in clip_text
    assert "ice cream truck" not in clip_text
    assert "just more terminology" not in clip_text
    assert "the second thing" not in clip_text
    assert "in summary" not in clip_text
    assert len(clip_text.split()) < 80


def test_procedural_first_thing_language_stays_inside_one_worked_unit() -> None:
    text = (
        "For the chain rule the first thing you need to do is differentiate the "
        "outer function and then multiply by the derivative of the inner function."
    )
    proposal = _proposal().model_copy(update={
        "candidate_id": "chain-rule-procedure",
        "start_quote": "For the chain rule the first thing",
        "end_quote": "derivative of the inner function",
        "title": "Applying the chain rule",
        "learning_objective": "Explain the two operations in the chain rule",
        "facet": "chain rule procedure",
        "topic_evidence_quote": (
            "differentiate the outer function and then multiply by the derivative"
        ),
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        [{"cue_id": "chain-rule", "start": 0.0, "end": 14.0, "text": text}],
        [],
        {"_segment_ignore_caption_case": True},
        topic="chain rule",
    )

    assert report.rejected_reasons == []
    assert "first thing you need to do" in report.clips[0]["_clip_text"].casefold()


def test_enumerated_meta_unit_split_across_cues_still_bounds_one_topic() -> None:
    texts = [
        "A p-value below alpha is one decision rule from the earlier example.",
        "Before we're done the first thing I want",
        (
            "to say is the null hypothesis is the default claim and a p-value helps "
            "us decide whether to reject it."
        ),
        (
            "The second thing I want to say is effect size describes how large the "
            "observed difference is."
        ),
    ]
    proposal = _proposal(end_line=3).model_copy(update={
        "candidate_id": "split-caption-null-hypothesis",
        "start_line": 2,
        "start_quote": "to say is the null hypothesis",
        "end_quote": "how large the observed difference is",
        "title": "The null hypothesis",
        "learning_objective": "Explain how a p-value informs the null hypothesis",
        "facet": "null hypothesis decision",
        "topic_evidence_quote": (
            "a p-value helps us decide whether to reject it"
        ),
    })
    segments = [
        {
            "cue_id": f"split-meta-{index}",
            "start": float(index * 10),
            "end": float(index * 10 + 10),
            "text": text,
        }
        for index, text in enumerate(texts)
    ]

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic="null hypothesis and p-values",
    )

    assert report.rejected_reasons == []
    clip_text = report.clips[0]["_clip_text"].casefold()
    assert "null hypothesis is the default claim" in clip_text
    assert "earlier example" not in clip_text
    assert "first thing i want" not in clip_text
    assert "effect size" not in clip_text


def test_enumerated_meta_unit_split_across_three_cues_is_still_detected() -> None:
    texts = [
        "The first thing",
        "I want to",
        "say is the null hypothesis is the default statistical claim.",
        "The second thing I want to say is effect size measures magnitude.",
    ]
    proposal = _proposal(end_line=3).model_copy(update={
        "candidate_id": "three-cue-meta-unit",
        "start_line": 2,
        "start_quote": "say is the null hypothesis",
        "end_quote": "effect size measures magnitude",
        "title": "The null hypothesis",
        "learning_objective": "Define the null hypothesis",
        "facet": "null hypothesis",
        "topic_evidence_quote": (
            "the null hypothesis is the default statistical claim"
        ),
    })
    segments = [
        {
            "cue_id": f"three-cue-meta-{index}",
            "start": float(index * 4),
            "end": float(index * 4 + 4),
            "text": text,
        }
        for index, text in enumerate(texts)
    ]

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic="null hypothesis",
    )

    assert report.rejected_reasons == []
    clip_text = report.clips[0]["_clip_text"].casefold()
    assert "null hypothesis is the default statistical claim" in clip_text
    assert "the first thing" not in clip_text
    assert "effect size" not in clip_text


@pytest.mark.parametrize(
    ("claim", "evidence"),
    [
        (
            "background the detector sees can bias measurements",
            "the detector sees can bias measurements",
        ),
        (
            "context the parser retains determines later behavior",
            "the parser retains determines later behavior",
        ),
        (
            "terminology this field uses differs across sources",
            "this field uses differs across sources",
        ),
    ],
)
def test_enumerated_meta_navigation_never_drops_the_real_subject(
    claim: str,
    evidence: str,
) -> None:
    text = f"The first thing I want to explain is {claim}."
    proposal = _proposal().model_copy(update={
        "candidate_id": "meta-real-subject",
        "start_quote": "The first thing I want to explain",
        "end_quote": claim,
        "title": claim,
        "learning_objective": f"Explain why {claim}",
        "facet": claim,
        "topic_evidence_quote": evidence,
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        [{"cue_id": "meta-subject", "start": 0.0, "end": 12.0, "text": text}],
        [],
        {"_segment_ignore_caption_case": True},
        topic=claim,
    )

    assert report.rejected_reasons == []
    assert claim in report.clips[0]["_clip_text"].casefold()


@pytest.mark.parametrize("split_hypothetical", [False, True])
def test_explicit_new_hypothetical_starts_a_new_example(
    split_hypothetical: bool,
) -> None:
    segments = [
        {
            "cue_id": "cue-6",
            "start": 228.0,
            "end": 266.0,
            "text": (
                "every time we do the experiment we get the opposite result so we can "
                "confidently reject this hypothesis BAM"
            ),
        },
    ]
    if split_hypothetical:
        segments.extend([
            {
                "cue_id": "cue-7",
                "start": 266.0,
                "end": 280.0,
                "text": "now let's imagine we had two",
            },
            {
                "cue_id": "cue-8",
                "start": 280.0,
                "end": 304.0,
                "text": (
                    "more drugs C and D just like before and their measurements are "
                    "similar but not exactly the same"
                ),
            },
        ])
    else:
        segments.append({
            "cue_id": "cue-7",
            "start": 266.0,
            "end": 304.0,
            "text": (
                "now let's imagine we had two more drugs C and D just like before and "
                "their measurements are similar but not exactly the same"
            ),
        })
    proposal = _proposal(end_line=len(segments) - 1).model_copy(update={
        "candidate_id": "second-drug-experiment",
        "start_quote": "every time we do the experiment",
        "end_quote": "similar but not exactly the same",
        "title": "A second drug hypothesis experiment",
        "learning_objective": "Explain the evidence in the drugs C and D experiment",
        "facet": "hypothesis evidence",
        "topic_evidence_quote": (
            "their measurements are similar but not exactly the same"
        ),
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic="hypothesis testing with drugs C and D",
    )

    assert report.rejected_reasons == []
    assert report.clips[0]["cue_ids"] == (
        ["cue-7", "cue-8"] if split_hypothetical else ["cue-7"]
    )
    assert report.clips[0]["_clip_text"].startswith("now let's imagine")
    assert "opposite result" not in report.clips[0]["_clip_text"]
    assert report.clips[0]["_clip_text"].endswith("similar but not exactly the same")


@pytest.mark.parametrize(
    ("marker_text", "middle_text", "claim_prefix"),
    [
        ("To summarize what we've covered so far.", "", ""),
        ("To summarize what we've", "", "covered so far. "),
        (
            "To summarize what we've covered so far.",
            "The first result rejected the null hypothesis.",
            "",
        ),
        (
            "To summarize what we've covered so far.",
            "Kinesin proteins move on microtubules.",
            "",
        ),
    ],
)
def test_recap_marker_in_previous_cue_rejects_compact_claim_in_next_cue(
    marker_text: str,
    middle_text: str,
    claim_prefix: str,
) -> None:
    claim = "Strong evidence against a hypothesis lets us reject it"
    plan = _compact_plan(
        exact_request="hypothesis testing",
        constraints=[{
            "constraint_id": "subject",
            "kind": "subject",
            "source_phrase": "hypothesis testing",
            "requirement": "Teach hypothesis testing",
        }],
        evidence=[{"constraint_id": "subject", "evidence_quote": claim}],
    )
    claim_line = 2 if middle_text else 1
    plan.topics[0] = plan.topics[0].model_copy(update={
        "start_line": claim_line,
        "end_line": claim_line,
        "start_quote": "Strong evidence against a hypothesis",
        "end_quote": "a hypothesis lets us reject it",
        "claim_quote": claim,
        "title": "Rejecting a hypothesis",
        "learning_objective": "Explain when to reject a hypothesis",
        "facet": "hypothesis decisions",
    })
    segments = [
        {
            "cue_id": "recap-marker",
            "start": 0.0,
            "end": 4.0,
            "text": marker_text,
        },
    ]
    if middle_text:
        segments.append({
            "cue_id": "recap-middle",
            "start": 4.0,
            "end": 8.0,
            "text": middle_text,
        })
    segments.append(
        {
            "cue_id": "recap-claim",
            "start": 8.0 if middle_text else 4.0,
            "end": 80.0,
            "text": f"{claim_prefix}{claim}.",
        }
    )

    report = gemini_segment._plan_to_report(
        plan,
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic="hypothesis testing",
    )

    assert report.clips == []
    assert report.rejected_reasons == ["proposal_0:recap_evidence"]


@pytest.mark.parametrize(
    "text",
    [
        (
            "To recap, Alpha is a baseline statistical model. "
            "Beta is a flexible statistical model."
        ),
        (
            "To recap, here's another example of a statistical model. "
            "Beta is a flexible statistical model."
        ),
        (
            "To recap. Concept one is bias. Concept two is variance. "
            "Beta is a flexible statistical model."
        ),
        (
            "To recap, let's consider another example where proteins move on to the "
            "next compartment. Beta is a flexible statistical model."
        ),
        (
            "To recap, let's consider another example where cells switch to aerobic "
            "respiration. Beta is a flexible statistical model."
        ),
    ],
)
def test_ordinary_transitions_inside_a_recap_do_not_clear_recap_state(
    text: str,
) -> None:
    claim = "Beta is a flexible statistical model"
    plan = _compact_plan(
        exact_request="beta statistical model",
        constraints=[{
            "constraint_id": "subject",
            "kind": "subject",
            "source_phrase": "beta statistical model",
            "requirement": "Teach the beta statistical model",
        }],
        evidence=[{"constraint_id": "subject", "evidence_quote": claim}],
    )
    plan.topics[0] = plan.topics[0].model_copy(update={
        "start_line": 0,
        "end_line": 0,
        "start_quote": "To recap Alpha is a baseline",
        "end_quote": "Beta is a flexible statistical model",
        "claim_quote": claim,
        "title": "Beta statistical model",
        "learning_objective": "Explain the beta statistical model",
        "facet": "statistical models",
    })
    report = gemini_segment._plan_to_report(
        plan,
        [{"cue_id": "recap-facts", "start": 0.0, "end": 90.0, "text": text}],
        [],
        {"_segment_ignore_caption_case": True},
        topic="beta statistical model",
    )

    assert report.clips == []
    assert report.rejected_reasons == ["proposal_0:recap_evidence"]


@pytest.mark.parametrize(
    ("recap_prefix", "navigation"),
    [
        (
            "To recap, alpha was the earlier topic. ",
            "Now let's move on to confidence intervals.",
        ),
        (
            "To recap, alpha was the earlier topic. ",
            "Now let's turn to confidence intervals.",
        ),
        (
            "To recap, alpha was the earlier topic. ",
            "Next we'll cover confidence intervals.",
        ),
        (
            "To recap alpha was the earlier topic ",
            "now let's move on to confidence intervals ",
        ),
        (
            "To recap, alpha was the earlier topic; ",
            "now let's turn to confidence intervals; ",
        ),
    ],
)
def test_explicit_new_topic_after_recap_clears_recap_state(
    recap_prefix: str,
    navigation: str,
) -> None:
    text = (
        f"{recap_prefix}{navigation} "
        "Confidence intervals estimate a plausible range for a population mean."
    )
    proposal = _proposal().model_copy(update={
        "candidate_id": "confidence-intervals",
        "start_quote": "To recap alpha was the earlier topic",
        "end_quote": "plausible range for a population mean",
        "title": "Confidence intervals",
        "learning_objective": "Explain how confidence intervals estimate a population mean",
        "facet": "confidence intervals",
        "topic_evidence_quote": (
            "Confidence intervals estimate a plausible range for a population mean"
        ),
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        [{"cue_id": "new-topic", "start": 0.0, "end": 95.0, "text": text}],
        [],
        {"_segment_ignore_caption_case": True},
        topic="confidence intervals",
    )

    assert report.rejected_reasons == []
    assert len(report.clips) == 1
    assert "earlier topic" not in report.clips[0]["_clip_text"]
    assert "Confidence intervals estimate" in report.clips[0]["_clip_text"]


def test_relational_hypothetical_comparison_is_not_split() -> None:
    claim = "compare sample sizes ten and one hundred"
    text = (
        "With sample size ten, the sample mean varies widely. Now let's imagine a new "
        "sample size of one hundred and compare sample sizes ten and one hundred. The "
        "larger sample has a narrower sampling distribution."
    )

    transitions = gemini_segment._candidate_topic_transitions(
        [{"cue_id": "sample-size-comparison", "start": 0.0, "end": 120.0, "text": text}],
        0,
        0,
        evidence_quote=claim,
        learning_objective="Compare sample sizes ten and one hundred",
        relationship_bridge_allowed=True,
    )

    assert transitions == []


@pytest.mark.parametrize(
    "text",
    [
        "To recap, the estimate is unbiased.",
        "Let's recap the main result.",
        "To sum up, the null hypothesis is rejected.",
        "In summary the larger sample has less variability.",
    ],
)
def test_common_explicit_recap_forms_are_recognized(text: str) -> None:
    assert gemini_segment._EXPLICIT_RECAP_NAVIGATION_RE.search(text)


@pytest.mark.parametrize(
    "text",
    [
        "To summarize the measurements, compute the mean.",
        "Let's sum up the squared deviations, then divide by n.",
    ],
)
def test_substantive_summary_verbs_are_not_recap_navigation(text: str) -> None:
    assert gemini_segment._EXPLICIT_RECAP_NAVIGATION_RE.search(text) is None


def test_edge_navigation_never_manufactures_an_incomplete_ending() -> None:
    text = (
        "Photosynthesis converts light into chemical energy. The most important "
        "point is now let's discuss respiration."
    )
    proposal = _proposal().model_copy(update={
        "start_quote": "Photosynthesis converts light",
        "end_quote": "now let's discuss respiration",
        "topic_evidence_quote": (
            "Photosynthesis converts light into chemical energy"
        ),
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        [{"cue_id": "incomplete-tail", "start": 0.0, "end": 12.0, "text": text}],
        [],
        {"_segment_ignore_caption_case": True},
        topic="photosynthesis",
    )

    assert report.rejected_reasons == []
    assert report.clips[0]["_clip_text"].rstrip(".") == (
        "Photosynthesis converts light into chemical energy"
    )


def test_repeated_one_word_end_uses_the_authoritative_last_occurrence() -> None:
    text = "Photosynthesis stores energy in glucose. Now let's talk about energy."
    proposal = _proposal().model_copy(update={
        "start_quote": "Photosynthesis",
        "end_quote": "energy",
        "topic_evidence_quote": "Photosynthesis stores energy in glucose",
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        [{"cue_id": "repeated-edge", "start": 0.0, "end": 10.0, "text": text}],
        [],
        {"_segment_ignore_caption_case": True},
        topic="photosynthesis",
    )

    assert report.rejected_reasons == []
    assert report.clips[0]["_clip_text"].rstrip(" .!?") == (
        "Photosynthesis stores energy in glucose"
    )


def test_split_list_completes_before_following_navigation() -> None:
    segments = [
        {
            "cue_id": "list-a",
            "start": 0.0,
            "end": 7.0,
            "text": "The products include sugars such as glucose, NADP plus",
        },
        {
            "cue_id": "list-b",
            "start": 7.0,
            "end": 13.0,
            "text": "ADP and P so let's begin our discussion of chloroplast structure.",
        },
    ]
    proposal = _proposal().model_copy(update={
        "start_quote": "The products include sugars",
        "end_quote": "sugars such as glucose NADP plus",
        "topic_evidence_quote": (
            "The products include sugars such as glucose NADP plus"
        ),
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic="photosynthesis products",
    )

    assert report.rejected_reasons == []
    assert report.clips[0]["cue_ids"] == ["list-a", "list-b"]
    assert report.clips[0]["_clip_text"].endswith("NADP plus ADP and P")
    assert "begin our discussion" not in report.clips[0]["_clip_text"]


def test_contextual_notation_leadin_recovers_the_next_complete_setup() -> None:
    segments = [
        {
            "cue_id": "notation-transition",
            "start": 0.0,
            "end": 4.0,
            "text": "Now, there's other notations.",
        },
        {
            "cue_id": "function-setup",
            "start": 4.0,
            "end": 10.0,
            "text": "If this curve is described as y is equal to f of x.",
        },
        {
            "cue_id": "derivative-notation",
            "start": 10.0,
            "end": 18.0,
            "text": "Then dy over dx at that input is written as f prime of x.",
        },
    ]
    proposal = _proposal(end_line=2).model_copy(update={
        "candidate_id": "lagrange-notation",
        "start_quote": "Now there's other notations",
        "end_quote": "written as f prime of x",
        "title": "Lagrange derivative notation",
        "learning_objective": "Explain how f prime denotes a derivative",
        "facet": "derivative notation",
        "topic_evidence_quote": "dy over dx at that input is written as f prime",
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic="derivative notation",
    )

    assert report.rejected_reasons == []
    assert report.clips[0]["cue_ids"] == ["function-setup", "derivative-notation"]
    assert report.clips[0]["_clip_text"].startswith("If this curve is described")
    assert "other notations" not in report.clips[0]["_clip_text"]


@pytest.mark.parametrize(
    "text",
    [
        "Usually, cells generate ATP by oxidative phosphorylation.",
        "Normally, the derivative of a constant is zero.",
        "Another way to state the chain rule is to multiply outer and inner derivatives.",
    ],
)
def test_complete_frequency_or_alternative_fact_is_standalone(text: str) -> None:
    assert gemini_segment._opening_clause_is_standalone(text)


@pytest.mark.parametrize(
    "text",
    [
        (
            "Ability to convert sunlight, carbon dioxide, and water into glucose and "
            "oxygen. This is photosynthesis."
        ),
        (
            "Process to turn light into chemical energy. It is called photosynthesis."
        ),
        (
            "For FTL communication, right? One proposed scheme has Bob change a "
            "measurement setting."
        ),
        (
            "The ability to convert sunlight into food. Photosynthesis stores light "
            "energy as glucose."
        ),
        (
            "An ability to convert sunlight into food. Photosynthesis stores light "
            "energy as glucose."
        ),
        (
            "The process to turn light into food. Photosynthesis stores light energy "
            "as glucose."
        ),
        (
            "In that case, right? Photosynthesis stores light energy as glucose."
        ),
    ],
)
def test_later_sentence_does_not_make_a_fragmentary_opening_standalone(
    text: str,
) -> None:
    assert not gemini_segment._opening_clause_is_standalone(text)


def test_named_category_makes_another_example_cold_viewer_complete() -> None:
    assert gemini_segment._opening_clause_is_standalone(
        "Another example of renewable energy is wind power, which converts moving air "
        "into electricity."
    )


def test_standalone_frequency_fact_does_not_import_a_previous_topic() -> None:
    segments = [
        {
            "cue_id": "limits",
            "start": 0.0,
            "end": 7.0,
            "text": "Limits describe the value a function approaches near an input.",
        },
        {
            "cue_id": "constant",
            "start": 7.0,
            "end": 14.0,
            "text": "Normally, the derivative of a constant is zero.",
        },
    ]
    proposal = _proposal().model_copy(update={
        "candidate_id": "constant-derivative",
        "start_line": 1,
        "end_line": 1,
        "start_quote": "Normally the derivative of a constant",
        "end_quote": "derivative of a constant is zero",
        "title": "Derivative of a constant",
        "learning_objective": "Explain why a constant has zero derivative",
        "facet": "derivative",
        "topic_evidence_quote": "the derivative of a constant is zero",
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic="derivative of a constant",
    )

    assert report.rejected_reasons == []
    assert report.clips[0]["cue_ids"] == ["constant"]


def test_same_cue_recovery_keeps_the_earliest_required_worked_setup() -> None:
    text = (
        "Now, there's another example. Let f of x equal x squared. "
        "Using the power rule, f prime of x equals two x."
    )
    proposal = _proposal().model_copy(update={
        "candidate_id": "power-rule",
        "start_quote": "Now there's another example",
        "end_quote": "f prime of x equals two x",
        "title": "Power rule",
        "learning_objective": "Differentiate f of x equals x squared through the answer",
        "facet": "worked example",
        "topic_evidence_quote": "Using the power rule f prime of x equals two x",
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        [{"cue_id": "worked", "start": 0.0, "end": 14.0, "text": text}],
        [],
        {"_segment_ignore_caption_case": True},
        topic="power rule worked example",
    )

    assert report.rejected_reasons == []
    assert report.clips[0]["_clip_text"].startswith("Let f of x equal x squared")


def test_opening_joke_and_navigation_are_trimmed_before_teaching() -> None:
    text = (
        "Another example, my friends, of unintelligent design. Back to the cycle! "
        "Ribulose bisphosphate gets a carbon dioxide molecule added during carbon fixation."
    )
    proposal = _proposal().model_copy(update={
        "candidate_id": "calvin-cycle-fixation",
        "start_quote": "Another example my friends of unintelligent design",
        "end_quote": "molecule added during carbon fixation",
        "title": "Carbon fixation in the Calvin cycle",
        "learning_objective": "Explain the first carbon-fixation step of the Calvin cycle",
        "facet": "Calvin cycle carbon fixation",
        "topic_evidence_quote": (
            "Ribulose bisphosphate gets a carbon dioxide molecule added during carbon fixation"
        ),
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        [{"cue_id": "calvin", "start": 0.0, "end": 13.0, "text": text}],
        [],
        {"_segment_ignore_caption_case": True},
        topic="Calvin cycle",
    )

    assert report.rejected_reasons == []
    assert report.clips[0]["_clip_text"].startswith("Ribulose bisphosphate")
    assert "unintelligent design" not in report.clips[0]["_clip_text"]
    assert "Back to the cycle" not in report.clips[0]["_clip_text"]


def test_compact_selector_aliases_preserve_canonical_fields_and_supporting_rank() -> None:
    compact = gemini_segment._CompactBoundaryTopic(
        candidate_id="supporting-definition",
        start_line=0,
        end_line=0,
        start_quote="A derivative measures instantaneous change",
        end_quote="with respect to its input",
        claim_quote=(
            "A derivative measures instantaneous change in a function with respect"
        ),
        title="Derivative definition",
        learning_objective="Define a derivative before a worked example",
        facet="derivative definition",
        informativeness=0.9,
        topic_relevance=0.9,
        educational_importance=0.85,
        difficulty=0.2,
        directly_teaches_topic=True,
        substantive=True,
        factually_grounded=True,
        self_contained=True,
        is_standalone=True,
        intent_evidence=[{
            "constraint_id": "subject",
            "evidence_quote": (
                "A derivative measures instantaneous change in a function with respect"
            ),
        }],
    )
    payload = gemini_segment._CompactBoundaryPlan(
        request_intent={
            "exact_request": "chain rule worked example",
            "constraints": [
                {
                    "constraint_id": "subject",
                    "kind": "subject",
                    "source_phrase": "chain rule",
                    "requirement": "Teach the chain rule",
                },
                {
                    "constraint_id": "task",
                    "kind": "format",
                    "source_phrase": "worked example",
                    "requirement": "Work through an example",
                },
            ],
        },
        topics=[compact],
    ).model_dump_json(by_alias=True)
    assert '"id":"supporting-definition"' in payload
    assert '"ie":[{"id":"subject","q":' in payload
    assert '"role"' not in payload
    assert '"evidence"' not in payload
    parsed = gemini_segment._CompactBoundaryPlan.model_validate_json(payload)

    report = gemini_segment._plan_to_report(
        parsed,
        [{
            "cue_id": "definition",
            "start": 0.0,
            "end": 10.0,
            "text": (
                "A derivative measures instantaneous change in a function with respect "
                "to its input."
            ),
        }],
        [],
        {"_segment_ignore_caption_case": True},
        topic="chain rule worked example",
    )

    assert report.rejected_reasons == []
    assert report.clips[0]["intent_role"] == "supporting"
    assert report.clips[0]["intent_coverage"] == 0.5
    assert report.clips[0]["intent_evidence"][0]["constraint_id"] == "subject"
    assert report.clips[0]["topic_evidence_quote"].startswith(
        "A derivative measures instantaneous change"
    )


@pytest.mark.parametrize(
    "topic,constraints,text,claim,evidence",
    [
        (
            "opportunity cost versus sunk cost",
            [
                {
                    "constraint_id": "opportunity",
                    "kind": "subject",
                    "source_phrase": "opportunity cost",
                    "requirement": "Teach opportunity cost",
                },
                {
                    "constraint_id": "comparison",
                    "kind": "relationship",
                    "source_phrase": "versus",
                    "requirement": "Compare the two costs",
                },
                {
                    "constraint_id": "sunk",
                    "kind": "subject",
                    "source_phrase": "sunk cost",
                    "requirement": "Teach sunk cost",
                },
            ],
            (
                "Opportunity cost is the value of the next best alternative "
                "forgone when choosing."
            ),
            "Opportunity cost is the value of the next best alternative",
            [
                "opportunity",
                "comparison",
                "sunk",
            ],
        ),
        (
            "precision and recall",
            [
                {
                    "constraint_id": "precision",
                    "kind": "subject",
                    "source_phrase": "precision",
                    "requirement": "Teach precision",
                },
                {
                    "constraint_id": "recall",
                    "kind": "subject",
                    "source_phrase": "recall",
                    "requirement": "Teach recall",
                },
            ],
            "Precision is the share of predicted positives that are actually positive.",
            "Precision is the share of predicted positives that are actually positive",
            ["precision"],
        ),
    ],
)
def test_compound_request_rejects_one_sided_clips_even_when_model_marks_relevance(
    topic: str,
    constraints: list[dict],
    text: str,
    claim: str,
    evidence: list[str],
) -> None:
    plan = gemini_segment._CompactBoundaryPlan(
        request_intent={"exact_request": topic, "constraints": constraints},
        topics=[gemini_segment._CompactBoundaryTopic(
            candidate_id="one-sided",
            start_line=0,
            end_line=0,
            start_quote=" ".join(text.split()[:5]),
            end_quote=" ".join(text.split()[-5:]),
            claim_quote=claim,
            title="One-sided explanation",
            learning_objective="Explain only one requested side",
            facet="single requested side",
            informativeness=0.95,
            topic_relevance=0.95,
            educational_importance=0.95,
            difficulty=0.2,
            directly_teaches_topic=True,
            substantive=True,
            factually_grounded=True,
            self_contained=True,
            is_standalone=True,
            intent_evidence=[
                {
                    "constraint_id": constraint_id,
                    "evidence_quote": claim,
                }
                for constraint_id in evidence
            ],
        )],
    )

    report = gemini_segment._plan_to_report(
        plan,
        [{"cue_id": "one-sided", "start": 0.0, "end": 12.0, "text": text}],
        [],
        {"_segment_ignore_caption_case": True},
        topic=topic,
    )

    assert report.clips == []
    assert report.rejected_reasons == [
        "proposal_0:incomplete_joint_request_coverage"
    ]


@pytest.mark.parametrize(
    "topic,constraints",
    [
        (
            "alpha versus beta",
            [
                {
                    "constraint_id": "alpha",
                    "kind": "subject",
                    "source_phrase": "alpha",
                    "requirement": "Teach alpha",
                },
                {
                    "constraint_id": "relation",
                    "kind": "relationship",
                    "source_phrase": "versus",
                    "requirement": "Compare alpha with beta",
                },
                {
                    "constraint_id": "beta",
                    "kind": "subject",
                    "source_phrase": "beta",
                    "requirement": "Teach beta",
                },
            ],
        ),
        (
            "alpha transition to beta",
            [
                {
                    "constraint_id": "alpha",
                    "kind": "subject",
                    "source_phrase": "alpha",
                    "requirement": "Teach alpha",
                },
                {
                    "constraint_id": "relation",
                    "kind": "relationship",
                    "source_phrase": "transition to",
                    "requirement": "Explain the transition",
                },
                {
                    "constraint_id": "beta",
                    "kind": "outcome",
                    "source_phrase": "beta",
                    "requirement": "Reach beta",
                },
            ],
        ),
    ],
)
def test_joint_relationship_evidence_rejects_adjacent_definitions(
    topic: str,
    constraints: list[dict],
) -> None:
    text = (
        "Alpha is a stable source quantity; beta is a separate target quantity."
    )
    evidence_quote = (
        "Alpha is a stable source quantity beta is a separate target quantity"
    )
    plan = gemini_segment._CompactBoundaryPlan(
        request_intent={"exact_request": topic, "constraints": constraints},
        topics=[gemini_segment._CompactBoundaryTopic(
            candidate_id="adjacent-definitions",
            start_line=0,
            end_line=0,
            start_quote="Alpha is a stable source quantity",
            end_quote="beta is a separate target quantity",
            claim_quote=evidence_quote,
            title="Two adjacent definitions",
            learning_objective="Define alpha and beta separately",
            facet="definitions",
            informativeness=0.95,
            topic_relevance=0.95,
            educational_importance=0.95,
            difficulty=0.2,
            directly_teaches_topic=True,
            substantive=True,
            factually_grounded=True,
            self_contained=True,
            is_standalone=True,
            intent_evidence=[
                {
                    "constraint_id": "alpha",
                    "evidence_quote": "Alpha is a stable source quantity",
                },
                {
                    "constraint_id": "relation",
                    "evidence_quote": evidence_quote,
                },
                {
                    "constraint_id": "beta",
                    "evidence_quote": "beta is a separate target quantity",
                },
            ],
        )],
    )

    validated, error = gemini_segment._validated_intent_constraints(plan, topic)
    assert error is None
    assert not gemini_segment._joint_relationship_evidence_matches(
        evidence_quote,
        topic,
        validated,
    )

    report = gemini_segment._plan_to_report(
        plan,
        [{"cue_id": "definitions", "start": 0.0, "end": 8.0, "text": text}],
        [],
        {"_segment_ignore_caption_case": True},
        topic=topic,
    )

    assert report.clips == []
    assert report.rejected_reasons == [
        "proposal_0:incomplete_joint_request_coverage"
    ]


@pytest.mark.parametrize(
    "topic,constraints,text",
    [
        (
            "alpha versus beta",
            [
                {
                    "constraint_id": "alpha",
                    "kind": "subject",
                    "source_phrase": "alpha",
                    "requirement": "Teach alpha",
                },
                {
                    "constraint_id": "relation",
                    "kind": "relationship",
                    "source_phrase": "versus",
                    "requirement": "Compare alpha with beta",
                },
                {
                    "constraint_id": "beta",
                    "kind": "subject",
                    "source_phrase": "beta",
                    "requirement": "Teach beta",
                },
            ],
            "Alpha differs from beta because alpha retains heat while beta releases it.",
        ),
        (
            "alpha transition to beta",
            [
                {
                    "constraint_id": "alpha",
                    "kind": "subject",
                    "source_phrase": "alpha",
                    "requirement": "Teach alpha",
                },
                {
                    "constraint_id": "relation",
                    "kind": "relationship",
                    "source_phrase": "transition to",
                    "requirement": "Explain the transition",
                },
                {
                    "constraint_id": "beta",
                    "kind": "outcome",
                    "source_phrase": "beta",
                    "requirement": "Reach beta",
                },
            ],
            "Alpha converts into beta when additional energy enters the system.",
        ),
    ],
)
def test_joint_relationship_evidence_accepts_one_spoken_relation(
    topic: str,
    constraints: list[dict],
    text: str,
) -> None:
    quote = " ".join(text.rstrip(".").split())
    plan = gemini_segment._CompactBoundaryPlan(
        request_intent={"exact_request": topic, "constraints": constraints},
        topics=[gemini_segment._CompactBoundaryTopic(
            candidate_id="spoken-relation",
            start_line=0,
            end_line=0,
            start_quote=" ".join(quote.split()[:5]),
            end_quote=" ".join(quote.split()[-5:]),
            claim_quote=quote,
            title="A spoken relationship",
            learning_objective="Explain the relationship between alpha and beta",
            facet="relationship",
            informativeness=0.95,
            topic_relevance=0.95,
            educational_importance=0.95,
            difficulty=0.2,
            directly_teaches_topic=True,
            substantive=True,
            factually_grounded=True,
            self_contained=True,
            is_standalone=True,
            intent_evidence=[
                {"constraint_id": item["constraint_id"], "evidence_quote": quote}
                for item in constraints
            ],
        )],
    )

    report = gemini_segment._plan_to_report(
        plan,
        [{"cue_id": "relation", "start": 0.0, "end": 8.0, "text": text}],
        [],
        {"_segment_ignore_caption_case": True},
        topic=topic,
    )

    assert report.rejected_reasons == []
    assert report.clips[0]["intent_role"] == "primary"


def test_bare_directional_path_is_joint_but_ordinary_to_phrase_is_not() -> None:
    path_topic = "source state to target state"
    path_plan = gemini_segment._CompactBoundaryPlan(
        request_intent={
            "exact_request": path_topic,
            "constraints": [
                {
                    "constraint_id": "source",
                    "kind": "subject",
                    "source_phrase": "source state",
                    "requirement": "Teach the source state",
                },
                {
                    "constraint_id": "path",
                    "kind": "relationship",
                    "source_phrase": "to",
                    "requirement": "Explain the path",
                },
                {
                    "constraint_id": "target",
                    "kind": "outcome",
                    "source_phrase": "target state",
                    "requirement": "Reach the target state",
                },
            ],
        },
        topics=[],
    )
    path_constraints, path_error = gemini_segment._validated_intent_constraints(
        path_plan,
        path_topic,
    )

    ordinary_topic = "introduction to calculus"
    ordinary_plan = gemini_segment._CompactBoundaryPlan(
        request_intent={
            "exact_request": ordinary_topic,
            "constraints": [
                {
                    "constraint_id": "left_noun",
                    "kind": "subject",
                    "source_phrase": "introduction",
                    "requirement": "Treat introduction as a subject",
                },
                {
                    "constraint_id": "connector",
                    "kind": "relationship",
                    "source_phrase": "to",
                    "requirement": "Connect the request wording",
                },
                {
                    "constraint_id": "right_noun",
                    "kind": "outcome",
                    "source_phrase": "calculus",
                    "requirement": "Teach calculus",
                },
            ],
        },
        topics=[],
    )
    ordinary_constraints, ordinary_error = (
        gemini_segment._validated_intent_constraints(
            ordinary_plan,
            ordinary_topic,
        )
    )

    assert path_error is None
    assert ordinary_error is None
    assert gemini_segment._request_requires_joint_intent_coverage(
        path_topic,
        path_constraints,
    )
    assert not gemini_segment._request_requires_joint_intent_coverage(
        ordinary_topic,
        ordinary_constraints,
    )

    path_text = (
        "The source state converts into the target state when energy is added."
    )
    path_quote = path_text.rstrip(".")
    path_topic_candidate = gemini_segment._CompactBoundaryTopic(
        candidate_id="complete-path",
        start_line=0,
        end_line=0,
        start_quote="The source state converts into",
        end_quote="state when energy is added",
        claim_quote=path_quote,
        title="Source to target path",
        learning_objective="Explain how the source becomes the target",
        facet="path",
        informativeness=0.95,
        topic_relevance=0.95,
        educational_importance=0.95,
        difficulty=0.2,
        directly_teaches_topic=True,
        substantive=True,
        factually_grounded=True,
        self_contained=True,
        is_standalone=True,
        intent_evidence=[
            {"constraint_id": item, "evidence_quote": path_quote}
            for item in ("source", "path", "target")
        ],
    )
    complete_report = gemini_segment._plan_to_report(
        path_plan.model_copy(update={"topics": [path_topic_candidate]}),
        [{"cue_id": "path", "start": 0.0, "end": 8.0, "text": path_text}],
        [],
        {"_segment_ignore_caption_case": True},
        topic=path_topic,
    )

    assert complete_report.rejected_reasons == []
    assert complete_report.clips[0]["intent_role"] == "primary"

    one_sided_text = "The source state remains stable under ordinary conditions."
    one_sided_quote = one_sided_text.rstrip(".")
    one_sided_candidate = path_topic_candidate.model_copy(update={
        "candidate_id": "one-sided-path",
        "start_quote": "The source state remains stable",
        "end_quote": "remains stable under ordinary conditions",
        "claim_quote": one_sided_quote,
        "intent_evidence": [
            {"constraint_id": item, "evidence_quote": one_sided_quote}
            for item in ("source", "path", "target")
        ],
    })
    one_sided_report = gemini_segment._plan_to_report(
        path_plan.model_copy(update={"topics": [one_sided_candidate]}),
        [{
            "cue_id": "one-sided-path",
            "start": 0.0,
            "end": 8.0,
            "text": one_sided_text,
        }],
        [],
        {"_segment_ignore_caption_case": True},
        topic=path_topic,
    )

    assert one_sided_report.clips == []
    assert one_sided_report.rejected_reasons == [
        "proposal_0:incomplete_joint_request_coverage"
    ]


@pytest.mark.parametrize(
    "exact_request,model_constraints,expected_phrases",
    [
        (
            "precision versus recall",
            [{
                "constraint_id": "combined",
                "kind": "subject",
                "source_phrase": "precision versus recall",
                "requirement": "Compare precision with recall",
            }],
            ["precision", "versus", "recall"],
        ),
        (
            "mitosis vs. meiosis",
            [
                {
                    "constraint_id": "mitosis",
                    "kind": "subject",
                    "source_phrase": "mitosis",
                    "requirement": "Teach mitosis",
                },
                {
                    "constraint_id": "merged_relationship",
                    "kind": "relationship",
                    "source_phrase": "vs. meiosis",
                    "requirement": "Compare mitosis with meiosis",
                },
            ],
            ["mitosis", "vs.", "meiosis"],
        ),
    ],
)
def test_binary_comparison_contract_normalizes_merged_model_constraints(
    exact_request: str,
    model_constraints: list[dict],
    expected_phrases: list[str],
) -> None:
    plan = gemini_segment._CompactBoundaryPlan(
        request_intent={
            "exact_request": exact_request,
            "constraints": model_constraints,
        },
        topics=[],
    )

    constraints, error = gemini_segment._validated_intent_constraints(
        plan,
        exact_request,
    )

    assert error is None
    assert [
        constraint.kind for constraint in constraints.values()
    ] == [
        gemini_segment._IntentConstraintKind.SUBJECT,
        gemini_segment._IntentConstraintKind.RELATIONSHIP,
        gemini_segment._IntentConstraintKind.SUBJECT,
    ]
    assert [
        constraint.source_phrase for constraint in constraints.values()
    ] == expected_phrases


def test_repaired_binary_comparison_is_regrounded_before_acceptance() -> None:
    exact_request = "precision versus recall"
    text = (
        "Precision and recall are two different measures: precision checks predicted "
        "positives, while recall checks actual positives."
    )
    plan = gemini_segment._CompactBoundaryPlan(
        request_intent={
            "exact_request": exact_request,
            "constraints": [{
                "constraint_id": "combined",
                "kind": "subject",
                "source_phrase": exact_request,
                "requirement": "Compare precision with recall",
            }],
        },
        topics=[gemini_segment._CompactBoundaryTopic(
            candidate_id="precision-recall-comparison",
            start_line=0,
            end_line=0,
            start_quote="Precision and recall are two different measures",
            end_quote="recall checks actual positives",
            claim_quote="Precision and recall are two different measures",
            title="Precision versus recall",
            learning_objective="Distinguish precision from recall",
            facet="comparison",
            informativeness=0.95,
            topic_relevance=0.95,
            educational_importance=0.95,
            difficulty=0.4,
            directly_teaches_topic=True,
            substantive=True,
            factually_grounded=True,
            self_contained=True,
            is_standalone=True,
            intent_evidence=[{
                "constraint_id": "combined",
                "evidence_quote": (
                    "Precision and recall are two different measures"
                ),
            }],
        )],
    )

    report = gemini_segment._plan_to_report(
        plan,
        [{"cue_id": "comparison", "start": 0.0, "end": 10.0, "text": text}],
        [],
        {"_segment_ignore_caption_case": True},
        topic=exact_request,
    )

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert [
        evidence["constraint_id"] for evidence in clip["intent_evidence"]
    ] == ["joint_subject_1", "joint_relationship", "joint_subject_2"]


def test_compact_selector_never_uses_an_agenda_as_its_teaching_claim() -> None:
    plan = _compact_plan(
        exact_request="calculus",
        constraints=[{
            "constraint_id": "subject",
            "kind": "subject",
            "source_phrase": "calculus",
            "requirement": "Teach calculus",
        }],
        evidence=[{
            "constraint_id": "subject",
            "evidence_quote": "Today we'll cover calculus examples and applications",
        }],
    )
    plan.topics[0] = plan.topics[0].model_copy(update={
        "start_line": 0,
        "end_line": 0,
        "start_quote": "Today we'll cover calculus examples",
        "end_quote": "a function changes",
        "claim_quote": "Today we'll cover calculus examples",
    })
    segments = [{
        "cue_id": "agenda-and-teaching",
        "start": 0.0,
        "end": 12.0,
        "text": (
            "Today we'll cover calculus examples and applications. "
            "A derivative measures the instantaneous rate at which a function changes."
        ),
    }]

    report = gemini_segment._plan_to_report(
        plan,
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic="calculus",
    )

    assert report.clips == []
    assert report.rejected_reasons == [
        "proposal_0:non_substantive_claim_quote"
    ]


def test_compact_selector_rejects_short_claim_omitting_agenda_prefix() -> None:
    plan = _compact_plan(
        exact_request="calculus",
        constraints=[{
            "constraint_id": "subject",
            "kind": "subject",
            "source_phrase": "calculus",
            "requirement": "Teach calculus",
        }],
        evidence=[{
            "constraint_id": "subject",
            "evidence_quote": "cover calculus examples and useful applications",
        }],
    )
    plan.topics[0] = plan.topics[0].model_copy(update={
        "start_line": 0,
        "end_line": 0,
        "start_quote": "Today we'll cover calculus examples",
        "end_quote": "examples and useful applications",
        "claim_quote": "cover calculus examples and useful applications",
    })
    segments = [{
        "cue_id": "short-agenda-claim",
        "start": 0.0,
        "end": 8.0,
        "text": "Today we'll cover calculus examples and useful applications.",
    }]

    report = gemini_segment._plan_to_report(
        plan,
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic="calculus",
    )

    assert report.clips == []
    assert report.rejected_reasons == [
        "proposal_0:non_substantive_claim_quote"
    ]


def test_compact_selector_trims_peerless_overview_to_claimed_atomic_unit() -> None:
    evidence_quote = "A derivative measures instantaneous rate of change"
    plan = _compact_plan(
        exact_request="calculus",
        constraints=[{
            "constraint_id": "subject",
            "kind": "subject",
            "source_phrase": "calculus",
            "requirement": "Teach calculus",
        }],
        evidence=[{
            "constraint_id": "subject",
            "evidence_quote": evidence_quote,
        }],
    )
    plan.topics[0] = plan.topics[0].model_copy(update={
        "candidate_id": "calculus-foundations",
        "start_line": 0,
        "end_line": 2,
        "start_quote": "A limit describes",
        "end_quote": "quantities across an interval",
        "claim_quote": evidence_quote,
        "title": "Calculus foundations",
        "learning_objective": "Explain foundational calculus ideas",
        "facet": "calculus foundations",
    })
    segments = [
        {
            "cue_id": "limit",
            "start": 0.0,
            "end": 8.0,
            "text": "A limit describes what a function approaches near an input.",
        },
        {
            "cue_id": "derivative",
            "start": 8.0,
            "end": 16.0,
            "text": "A derivative measures instantaneous rate of change at an input.",
        },
        {
            "cue_id": "integral",
            "start": 16.0,
            "end": 24.0,
            "text": "An integral accumulates quantities across an interval.",
        },
    ]

    report = gemini_segment._plan_to_report(
        plan,
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic="calculus",
    )

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["_clip_text"] == segments[1]["text"]
    assert clip["_start_line"] == clip["_end_line"] == 1
    assert clip["intent_role"] == "primary"
    assert "limit describes" not in clip["_clip_text"].casefold()
    assert "integral accumulates" not in clip["_clip_text"].casefold()


def test_compact_selector_projects_same_cue_overview_around_atomic_claim() -> None:
    evidence_quote = "A derivative measures instantaneous rate of change"
    plan = _compact_plan(
        exact_request="calculus",
        constraints=[{
            "constraint_id": "subject",
            "kind": "subject",
            "source_phrase": "calculus",
            "requirement": "Teach calculus",
        }],
        evidence=[{
            "constraint_id": "subject",
            "evidence_quote": evidence_quote,
        }],
    )
    plan.topics[0] = plan.topics[0].model_copy(update={
        "candidate_id": "same-cue-foundations",
        "start_line": 0,
        "end_line": 0,
        "start_quote": "A limit describes",
        "end_quote": "quantities across an interval",
        "claim_quote": evidence_quote,
        "title": "Calculus foundations",
        "learning_objective": "Explain foundational calculus ideas",
        "facet": "calculus foundations",
    })
    source = (
        "A limit describes what a function approaches near an input. "
        "A derivative measures instantaneous rate of change at an input. "
        "An integral accumulates quantities across an interval."
    )

    report = gemini_segment._plan_to_report(
        plan,
        [{"cue_id": "same-cue", "start": 0.0, "end": 24.0, "text": source}],
        [],
        {"_segment_ignore_caption_case": True},
        topic="calculus",
    )

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["_clip_text"] == (
        "A derivative measures instantaneous rate of change at an input."
    )
    assert clip["start_quote"].startswith("A derivative")
    assert clip["end_quote"].endswith("an input.")


def test_compact_selector_splits_a_plain_two_topic_umbrella() -> None:
    claim = "A derivative measures instantaneous rate of change"
    plan = _compact_plan(
        exact_request="calculus",
        constraints=[{
            "constraint_id": "subject",
            "kind": "subject",
            "source_phrase": "calculus",
            "requirement": "Teach calculus",
        }],
        evidence=[{
            "constraint_id": "subject",
            "evidence_quote": claim,
        }],
    )
    plan.topics[0] = plan.topics[0].model_copy(update={
        "candidate_id": "plain-calculus-umbrella",
        "start_line": 0,
        "end_line": 1,
        "start_quote": "A limit describes",
        "end_quote": "change at an input",
        "claim_quote": claim,
        "title": "Calculus",
        "learning_objective": "Explain calculus",
        "facet": "calculus",
    })
    segments = [
        {
            "cue_id": "limit",
            "start": 0.0,
            "end": 7.0,
            "text": "A limit describes what a function approaches near an input.",
        },
        {
            "cue_id": "derivative",
            "start": 7.0,
            "end": 14.0,
            "text": "A derivative measures instantaneous rate of change at an input.",
        },
    ]

    report = gemini_segment._plan_to_report(
        plan,
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic="calculus",
    )

    assert report.rejected_reasons == []
    assert report.clips[0]["_clip_text"] == segments[1]["text"]


def test_claim_atomicity_preserves_a_contextual_causal_explanation() -> None:
    claim = "Greenhouse gases absorb that energy and reemit it"
    plan = _compact_plan(
        exact_request="greenhouse warming",
        constraints=[{
            "constraint_id": "subject",
            "kind": "subject",
            "source_phrase": "greenhouse warming",
            "requirement": "Explain greenhouse warming",
        }],
        evidence=[{
            "constraint_id": "subject",
            "evidence_quote": claim,
        }],
    )
    plan.topics[0] = plan.topics[0].model_copy(update={
        "candidate_id": "greenhouse-causal-chain",
        "start_line": 0,
        "end_line": 3,
        "start_quote": "Sunlight passes through the atmosphere",
        "end_quote": "warming the lower atmosphere",
        "claim_quote": claim,
        "title": "How greenhouse gases warm Earth",
        "learning_objective": "Explain how greenhouse gases warm Earth",
        "facet": "greenhouse warming mechanism",
    })
    texts = [
        "Sunlight passes through the atmosphere and reaches Earth's surface.",
        "Earth's surface absorbs that light and becomes warm.",
        "The warm surface emits infrared energy back upward.",
        "Greenhouse gases absorb that energy and reemit it, warming the lower atmosphere.",
    ]
    segments = [
        {
            "cue_id": f"greenhouse-{index}",
            "start": index * 7.0,
            "end": (index + 1) * 7.0,
            "text": text,
        }
        for index, text in enumerate(texts)
    ]

    report = gemini_segment._plan_to_report(
        plan,
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic="greenhouse warming",
    )

    assert report.rejected_reasons == []
    assert report.clips[0]["_clip_text"] == " ".join(texts)


def test_claim_atomicity_preserves_a_complete_worked_example() -> None:
    claim = "The factorization is x minus two times x minus three"
    plan = _compact_plan(
        exact_request="quadratic equation worked example",
        constraints=[{
            "constraint_id": "subject",
            "kind": "subject",
            "source_phrase": "quadratic equation worked example",
            "requirement": "Solve a quadratic equation",
        }],
        evidence=[{
            "constraint_id": "subject",
            "evidence_quote": claim,
        }],
    )
    plan.topics[0] = plan.topics[0].model_copy(update={
        "candidate_id": "quadratic-factoring-example",
        "start_line": 0,
        "end_line": 2,
        "start_quote": "The equation is x squared",
        "end_quote": "roots are two and three",
        "claim_quote": claim,
        "title": "Factor a quadratic equation",
        "learning_objective": "Solve a quadratic equation worked example",
        "facet": "quadratic factoring example",
    })
    texts = [
        "The equation is x squared minus five x plus six equals zero.",
        "The factorization is x minus two times x minus three.",
        "Therefore the roots are two and three.",
    ]
    segments = [
        {
            "cue_id": f"quadratic-{index}",
            "start": index * 7.0,
            "end": (index + 1) * 7.0,
            "text": text,
        }
        for index, text in enumerate(texts)
    ]

    report = gemini_segment._plan_to_report(
        plan,
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic="quadratic equation worked example",
    )

    assert report.rejected_reasons == []
    assert report.clips[0]["_clip_text"] == " ".join(texts)


@pytest.mark.parametrize(
    "source",
    [
        "Today, we are going to discuss limits. A limit describes the value a function approaches.",
        "In this video, we discuss limits. A limit describes the value a function approaches.",
        "First we discuss limits. A limit describes the value a function approaches.",
    ],
)
def test_compact_selector_trims_a_complete_agenda_sentence(source: str) -> None:
    claim = "A limit describes the value a function approaches"
    plan = _compact_plan(
        exact_request="limits",
        constraints=[{
            "constraint_id": "subject",
            "kind": "subject",
            "source_phrase": "limits",
            "requirement": "Teach limits",
        }],
        evidence=[{
            "constraint_id": "subject",
            "evidence_quote": claim,
        }],
    )
    plan.topics[0] = plan.topics[0].model_copy(update={
        "candidate_id": "agenda-then-limit",
        "start_line": 0,
        "end_line": 0,
        "start_quote": source.split(".", 1)[0],
        "end_quote": "value a function approaches",
        "claim_quote": claim,
        "title": "What a limit describes",
        "learning_objective": "Define a function limit",
        "facet": "limit definition",
    })

    report = gemini_segment._plan_to_report(
        plan,
        [{"cue_id": "agenda", "start": 0.0, "end": 12.0, "text": source}],
        [],
        {"_segment_ignore_caption_case": True},
        topic="limits",
    )

    assert report.rejected_reasons == []
    assert report.clips[0]["_clip_text"] == (
        "A limit describes the value a function approaches."
    )


def test_compact_selector_accepts_a_repeated_grounded_claim_in_one_cue() -> None:
    claim = "A derivative measures instantaneous rate of change"
    source = (
        "A derivative measures instantaneous rate of change. "
        "This definition describes slope at a point. "
        "A derivative measures instantaneous rate of change."
    )
    plan = _compact_plan(
        exact_request="derivative definition",
        constraints=[{
            "constraint_id": "subject",
            "kind": "subject",
            "source_phrase": "derivative definition",
            "requirement": "Define a derivative",
        }],
        evidence=[{
            "constraint_id": "subject",
            "evidence_quote": claim,
        }],
    )
    plan.topics[0] = plan.topics[0].model_copy(update={
        "candidate_id": "repeated-derivative-definition",
        "start_line": 0,
        "end_line": 0,
        "start_quote": "A derivative measures instantaneous",
        "end_quote": "instantaneous rate of change.",
        "claim_quote": claim,
        "title": "Derivative definition",
        "learning_objective": "Define a derivative",
        "facet": "derivative definition",
    })

    report = gemini_segment._plan_to_report(
        plan,
        [{"cue_id": "repeated", "start": 0.0, "end": 12.0, "text": source}],
        [],
        {"_segment_ignore_caption_case": True},
        topic="derivative definition",
    )

    assert report.rejected_reasons == []
    assert len(report.clips) == 1


def test_compact_selector_keeps_substantive_enumerated_exact_topic_claim() -> None:
    claim = "Three branches of government divide authority among institutions"
    plan = _compact_plan(
        exact_request="three branches of government",
        constraints=[{
            "constraint_id": "subject",
            "kind": "subject",
            "source_phrase": "three branches of government",
            "requirement": "Teach the three branches of government",
        }],
        evidence=[{
            "constraint_id": "subject",
            "evidence_quote": claim,
        }],
    )
    plan.topics[0] = plan.topics[0].model_copy(update={
        "start_line": 0,
        "end_line": 0,
        "start_quote": "Three branches of government divide",
        "end_quote": "prevent concentrated political power",
        "claim_quote": claim,
        "title": "Three branches of government",
        "learning_objective": "Explain how three branches divide authority",
        "facet": "three branches of government",
    })
    segments = [{
        "cue_id": "branches",
        "start": 0.0,
        "end": 12.0,
        "text": (
            "Three branches of government divide authority among institutions "
            "to prevent concentrated political power."
        ),
    }]

    report = gemini_segment._plan_to_report(
        plan,
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic="three branches of government",
    )

    assert report.rejected_reasons == []
    assert len(report.clips) == 1
    assert report.clips[0]["_clip_text"] == segments[0]["text"]


def test_compact_selector_rejects_a_retrieval_expansion_as_exact_request() -> None:
    plan = _compact_plan(
        exact_request="plant energy conversion",
        constraints=[{
            "constraint_id": "subject",
            "kind": "subject",
            "source_phrase": "plant energy conversion",
            "requirement": "Teach plant energy conversion",
        }],
        evidence=[{
            "constraint_id": "subject",
            "evidence_quote": (
                "Cells use chlorophyll to capture light energy and power the chemical reactions"
            ),
        }],
    )

    report = gemini_segment._plan_to_report(
        plan,
        [{
            "cue_id": "photosynthesis",
            "start": 0.0,
            "end": 10.0,
            "text": (
                "Cells use chlorophyll to capture light energy and power the chemical "
                "reactions of photosynthesis."
            ),
        }],
        [],
        {},
        topic="photosynthesis",
    )

    assert report.clips == []
    assert report.rejected_reasons == ["intent_contract_request_mismatch"]


def test_unfiltered_selector_ignores_synthetic_request_rewording() -> None:
    evidence_quote = (
        "Cells use chlorophyll to capture light energy and power the chemical reactions"
    )
    plan = _compact_plan(
        exact_request="every substantive lesson in this source",
        constraints=[{
            "constraint_id": "synthetic",
            "kind": "scope",
            "source_phrase": "every substantive lesson",
            "requirement": "Return every substantive lesson",
        }],
        evidence=[{
            "constraint_id": "synthetic",
            "evidence_quote": evidence_quote,
        }],
    )

    report = gemini_segment._plan_to_report(
        plan,
        [{
            "cue_id": "photosynthesis",
            "start": 0.0,
            "end": 10.0,
            "text": f"{evidence_quote} of photosynthesis.",
        }],
        [],
        {},
        topic="",
    )

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["intent_role"] == "primary"
    assert clip["intent_coverage"] == 1.0
    assert clip["intent_evidence"] == []


def test_compact_selector_derives_primary_and_topic_evidence_from_grounding() -> None:
    evidence_quote = (
        "Cells use chlorophyll to capture light energy and power the chemical reactions"
    )
    plan = _compact_plan(
        exact_request="photosynthesis",
        constraints=[{
            "constraint_id": "subject",
            "kind": "subject",
            "source_phrase": "photosynthesis",
            "requirement": "Teach photosynthesis",
        }],
        evidence=[{
            "constraint_id": "subject",
            "evidence_quote": evidence_quote,
        }],
    )

    report = gemini_segment._plan_to_report(
        plan,
        [{
            "cue_id": "photosynthesis",
            "start": 0.0,
            "end": 10.0,
            "text": f"{evidence_quote} of photosynthesis.",
        }],
        [],
        {},
        topic="photosynthesis",
    )

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["intent_role"] == "primary"
    assert clip["intent_coverage"] == 1.0
    assert clip["topic_evidence_quote"] == evidence_quote


def test_selector_accepts_non_lossy_descriptive_strings_beyond_prompt_limits() -> None:
    proposal = _proposal().model_copy(update={
        "candidate_id": "candidate-" + ("identifier-" * 8),
        "start_quote": "opening " * 40,
        "end_quote": "closing " * 40,
        "title": "A complete descriptive title " * 8,
        "learning_objective": "Explain the complete grounded educational relationship " * 8,
        "facet": "A detailed but valid supporting facet " * 8,
        "reason": "The model supplied a detailed optional reason. " * 10,
        "topic_evidence_quote": "grounded transcript evidence " * 40,
    })

    parsed = gemini_segment._BoundaryPlan.model_validate_json(
        gemini_segment._BoundaryPlan(topics=[proposal]).model_dump_json()
    )

    assert len(parsed.topics) == 1
    assert parsed.topics[0].facet.startswith("A detailed but valid")


def test_duration_settings_do_not_change_a_complete_clip() -> None:
    complete = [{
        "cue_id": "cue-0",
        "start": 0.0,
        "end": 80.0,
        "text": (
            "Cells use chlorophyll to capture light energy and power the chemical "
            "reactions of photosynthesis."
        ),
    }]
    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[_proposal()]),
        complete,
        [],
        {"_segment_target_min_sec": 20, "_segment_target_sec": 55,
         "_segment_target_max_sec": 55},
        topic="photosynthesis",
    )

    assert [(clip["start"], clip["end"]) for clip in report.clips] == [(0.0, 80.0)]

    long_complete = [{**complete[0], "end": 420.0}]
    long_report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[_proposal()]),
        long_complete,
        [],
        {"_segment_target_max_sec": 55},
        topic="photosynthesis",
    )
    assert [(clip["start"], clip["end"]) for clip in long_report.clips] == [
        (0.0, 420.0)
    ]


def test_exact_boundary_quote_uniquely_inside_proposed_range_is_reanchored() -> None:
    segments = [
        {
            "cue_id": "cue-0",
            "start": 0.0,
            "end": 5.0,
            "text": "Welcome to the channel.",
        },
        {
            "cue_id": "cue-1",
            "start": 5.0,
            "end": 12.0,
            "text": "Cells use chlorophyll to capture light energy.",
        },
        {
            "cue_id": "cue-2",
            "start": 12.0,
            "end": 20.0,
            "text": "That energy powers the chemical reactions of photosynthesis.",
        },
    ]
    proposal = _proposal(end_line=2).model_copy(update={
        "start_line": 0,
        "start_quote": "Cells use chlorophyll to capture light energy",
        "end_quote": "chemical reactions of photosynthesis",
        "topic_evidence_quote": (
            "Cells use chlorophyll to capture light energy"
        ),
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic="photosynthesis",
    )

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["cue_ids"] == ["cue-1", "cue-2"]
    assert clip["_quote_repaired"] is True


def test_exact_start_quote_split_across_adjacent_cues_is_projected() -> None:
    segments = [
        {
            "cue_id": "cue-0",
            "start": 0.0,
            "end": 5.0,
            "text": "Cells use chlorophyll to",
        },
        {
            "cue_id": "cue-1",
            "start": 5.0,
            "end": 12.0,
            "text": (
                "capture light energy and power the chemical reactions of photosynthesis."
            ),
        },
    ]
    proposal = _proposal(end_line=1).model_copy(update={
        "start_quote": "Cells use chlorophyll to capture light energy",
        "end_quote": "chemical reactions of photosynthesis",
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        segments,
        [],
        {},
        topic="photosynthesis",
    )

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["start_quote"] == "Cells use chlorophyll to"
    assert clip["_quote_repaired"] is True


def test_exact_end_quote_split_across_adjacent_cues_is_projected() -> None:
    segments = [
        {
            "cue_id": "cue-0",
            "start": 0.0,
            "end": 5.0,
            "text": "Cells use chlorophyll to capture light energy and power",
        },
        {
            "cue_id": "cue-1",
            "start": 5.0,
            "end": 12.0,
            "text": "the chemical reactions of photosynthesis.",
        },
    ]
    proposal = _proposal(end_line=1).model_copy(update={
        "start_quote": "Cells use chlorophyll to capture light energy",
        "end_quote": "power the chemical reactions of photosynthesis",
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        segments,
        [],
        {},
        topic="photosynthesis",
    )

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["end_quote"] == "the chemical reactions of photosynthesis"
    assert clip["_quote_repaired"] is True


def test_repeated_cross_cue_boundary_quote_falls_back_to_selected_cues() -> None:
    segments = [
        {"cue_id": "cue-0", "start": 0.0, "end": 2.0, "text": "Cells use"},
        {
            "cue_id": "cue-1",
            "start": 2.0,
            "end": 5.0,
            "text": "chlorophyll to capture light energy.",
        },
        {"cue_id": "cue-2", "start": 5.0, "end": 7.0, "text": "Cells use"},
        {
            "cue_id": "cue-3",
            "start": 7.0,
            "end": 12.0,
            "text": (
                "chlorophyll to capture light energy and power the chemical reactions "
                "of photosynthesis."
            ),
        },
    ]
    proposal = _proposal(end_line=3).model_copy(update={
        "start_quote": "Cells use chlorophyll to capture light energy",
        "end_quote": "chemical reactions of photosynthesis",
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        segments,
        [],
        {},
        topic="photosynthesis",
    )

    assert len(report.clips) == 1
    assert report.clips[0]["cue_ids"] == ["cue-0", "cue-1", "cue-2", "cue-3"]
    assert "bad_start_quote" in report.clips[0]["_boundary_fallback_reasons"]


def test_cross_cue_boundary_quote_reset_keeps_finite_selected_range() -> None:
    segments = [
        {
            "cue_id": "cue-0",
            "start": 0.0,
            "end": 2.0,
            "text": "Cells use chlorophyll to",
        },
        {
            "cue_id": "cue-1",
            "start": 10.0,
            "end": 15.0,
            "text": (
                "capture light energy and power the chemical reactions of photosynthesis."
            ),
        },
    ]
    proposal = _proposal(end_line=1).model_copy(update={
        "start_quote": "Cells use chlorophyll to capture light energy",
        "end_quote": "chemical reactions of photosynthesis",
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        segments,
        [],
        {},
        topic="photosynthesis",
    )

    assert len(report.clips) == 1
    assert report.clips[0]["cue_ids"] == ["cue-0", "cue-1"]
    assert "bad_start_quote" in report.clips[0]["_boundary_fallback_reasons"]


def test_cross_cue_reanchoring_never_discards_substantive_context() -> None:
    segments = [
        {
            "cue_id": "cue-0",
            "start": 0.0,
            "end": 4.0,
            "text": "Water reaches the leaf through the xylem before light capture.",
        },
        {
            "cue_id": "cue-1",
            "start": 4.0,
            "end": 7.0,
            "text": "Cells use chlorophyll to",
        },
        {
            "cue_id": "cue-2",
            "start": 7.0,
            "end": 14.0,
            "text": (
                "capture light energy and power the chemical reactions of photosynthesis."
            ),
        },
    ]
    proposal = _proposal(end_line=2).model_copy(update={
        "start_quote": "Cells use chlorophyll to capture light energy",
        "end_quote": "chemical reactions of photosynthesis",
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        segments,
        [],
        {},
        topic="photosynthesis",
    )

    assert len(report.clips) == 1
    assert report.clips[0]["cue_ids"] == ["cue-0", "cue-1", "cue-2"]
    assert "bad_start_quote" in report.clips[0]["_boundary_fallback_reasons"]


def test_boundary_quote_reanchoring_never_discards_substantive_context() -> None:
    segments = [
        {
            "cue_id": "cue-0",
            "start": 0.0,
            "end": 5.0,
            "text": "Water first reaches the leaf through the xylem.",
        },
        {
            "cue_id": "cue-1",
            "start": 5.0,
            "end": 12.0,
            "text": "Cells use chlorophyll to capture light energy.",
        },
        {
            "cue_id": "cue-2",
            "start": 12.0,
            "end": 20.0,
            "text": "That energy powers the chemical reactions of photosynthesis.",
        },
    ]
    proposal = _proposal(end_line=2).model_copy(update={
        "start_quote": "Cells use chlorophyll to capture light energy",
        "end_quote": "chemical reactions of photosynthesis",
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        segments,
        [],
        {},
        topic="photosynthesis",
    )

    assert report.clips == []
    assert report.rejected_reasons == [
        "proposal_0:ungrounded_topic_evidence_quote"
    ]


@pytest.mark.parametrize(
    "start_quote",
    [
        "chlorophyll captures sunlight",  # Paraphrase, not exact transcript text.
        "Cells use chlorophyll",  # Appears in two cues, so the anchor is ambiguous.
        "Outside exact anchor words",  # Exact, but outside the proposed cue range.
    ],
)
def test_boundary_quote_reanchoring_remains_exact_unique_and_in_range(
    start_quote: str,
) -> None:
    segments = [
        {
            "cue_id": "cue-0",
            "start": 0.0,
            "end": 5.0,
            "text": "A separate completed idea appears here.",
        },
        {
            "cue_id": "cue-1",
            "start": 5.0,
            "end": 12.0,
            "text": "Cells use chlorophyll to capture light energy.",
        },
        {
            "cue_id": "cue-2",
            "start": 12.0,
            "end": 20.0,
            "text": (
                "Cells use chlorophyll while chemical reactions of photosynthesis finish."
            ),
        },
        {
            "cue_id": "cue-3",
            "start": 20.0,
            "end": 24.0,
            "text": "Outside exact anchor words.",
        },
    ]
    proposal = _proposal(end_line=2).model_copy(update={
        "start_quote": start_quote,
        "end_quote": "chemical reactions of photosynthesis finish",
        "topic_evidence_quote": (
            "Cells use chlorophyll while chemical reactions of photosynthesis"
        ),
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        segments,
        [],
        {},
        topic="photosynthesis",
    )

    assert len(report.clips) == 1
    assert report.clips[0]["cue_ids"] == ["cue-0", "cue-1", "cue-2"]
    assert "bad_start_quote" in report.clips[0]["_boundary_fallback_reasons"]


def test_selector_prompt_is_exhaustive_for_primary_and_supporting_units() -> None:
    _system, user = gemini_segment._boundary_prompts(
        "[0] 00:00 Cells use chlorophyll to capture light energy.",
        1,
        "photosynthesis, cellular respiration, and DNA inheritance",
        learner_level="beginner",
    )

    assert "genuinely related to TOPIC" in user
    assert "PRIMARY unit fulfills every required non-scope constraint" in user
    assert "SUPPORTING unit has a substantive educational connection" in user
    assert "Return supporting units even when this source contains no primary unit" in user
    assert "keep the complete object in one atomic constraint" in user
    assert "every distinct educational unit" in user
    assert "whole transcript" in (_system + user).lower()
    assert (
        "informativeness, topic_relevance, and educational_importance\n"
        "  are each at least 0.75"
    ) in (_system + user)
    assert "difficulty band 0.00-0.40" in user
    assert "level fit is a gemini selection eligibility rule" in user.lower()
    assert "unseen visual" in user
    assert "every distinct qualifying primary and supporting moment" in user
    assert "internal interruption" in (_system + user)
    assert "internal filler may never remain" in (_system + user).lower()
    assert "otherwise keep it" not in (_system + user).lower()
    assert "may remain when cutting around it" not in (_system + user).lower()
    assert "never stop after one exact match" in (_system + user).lower()
    assert "title (at most 12 words)" in user
    assert "learning_objective (at most 24 words)" in user
    assert "facet (at most 12 words)" in user
    assert "explicitly distinguishing two named sides" in user
    assert user.index("Transcript (") < user.index("Exact user request:")
    assert "1. Interpret the exact request" in user
    assert "request_intent" in user
    assert "Do not output a role" in user
    assert "requested operations or tasks" in user
    assert "Do not substitute retrieval expansions" in user
    assert "2. Map every distinct educational unit" in user
    assert "up to 40 for this source" in user
    assert "3. For every qualifying unit" in user
    assert "end before the transition" in user
    assert "4. Score topic relevance, information density" in user
    assert user.count("[0] 00:00 Cells use chlorophyll") == 1
    assert "180-second" not in (_system + user)


def test_same_cue_trailing_preview_is_trimmed_from_model_end_quote() -> None:
    text = (
        "Cells use chlorophyll to capture light energy and power the chemical "
        "reactions of photosynthesis. But we'll talk more about that next time."
    )
    proposal = _proposal().model_copy(update={
        "end_quote": (
            "chemical reactions of photosynthesis. But we'll talk more about that "
            "next time"
        ),
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        [{"cue_id": "cue-0", "start": 0.0, "end": 18.0, "text": text}],
        [],
        {"_segment_ignore_caption_case": True},
        topic="photosynthesis",
    )

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["end_quote"] == "power the chemical reactions of photosynthesis."
    assert clip["edge_projection"]["end"] == {
        "required": True,
        "cue_id": "cue-0",
        "quote": "power the chemical reactions of photosynthesis.",
    }
    assert clip["_clip_text"].endswith("chemical reactions of photosynthesis.")
    assert "next time" not in clip["_clip_text"]


@pytest.mark.parametrize(
    ("candidate_id", "segments", "topic", "objective", "evidence", "forbidden"),
    [
        (
            "area-before-future-preview",
            [
                (
                    "The definite integral adds the areas of increasingly thin "
                    "rectangles and in the limit this equals the area under the curve "
                    "trust me we'll get much more involved later don't worry we'll do "
                    "that later in chapter four so let's define a limit as the value a "
                    "function approaches"
                ),
            ],
            "area under a curve",
            "Explain why a definite integral equals the area under a curve",
            "in the limit this equals the area under the curve",
            "trust me",
        ),
        (
            "differential-equation-before-assignment",
            [
                (
                    "Separating variables and integrating gives y equals c e to the "
                    "negative k t which is the complete solution"
                ),
                (
                    "There are many other examples, so I'll leave one as an exercise "
                    "to write a differential equation that describes a radioactive "
                    "substance Okay so we'll come"
                ),
            ],
            "differential equations",
            "Solve the separable differential equation through its general solution",
            "y equals c e to the negative k t",
            "many other examples",
        ),
        (
            "indefinite-parts-before-definite-version",
            [
                (
                    "For an indefinite integral integration by parts gives integral u "
                    "d v equals u v minus integral v d u and that completes the derivation "
                    "So let me spell it out So this is the definite integral's version"
                ),
            ],
            "indefinite integration by parts",
            "Derive the indefinite integration by parts identity",
            "integral u d v equals u v minus integral v d u",
            "So let me spell it out",
        ),
    ],
)
def test_coarse_caption_tail_is_trimmed_without_losing_complete_teaching(
    candidate_id: str,
    segments: list[str],
    topic: str,
    objective: str,
    evidence: str,
    forbidden: str,
) -> None:
    cues = [
        {
            "cue_id": f"{candidate_id}:{index}",
            "start": index * 12.0,
            "end": (index + 1) * 12.0,
            "text": text,
        }
        for index, text in enumerate(segments)
    ]
    proposal = _proposal(end_line=len(cues) - 1).model_copy(update={
        "candidate_id": candidate_id,
        "start_quote": " ".join(segments[0].split()[:8]),
        "end_quote": " ".join(segments[-1].split()[-8:]),
        "title": "Complete teaching before an edge transition",
        "learning_objective": objective,
        "facet": topic,
        "reason": "The retained prefix completes the requested teaching unit.",
        "topic_evidence_quote": evidence,
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        cues,
        [],
        {"_segment_ignore_caption_case": True},
        topic=topic,
    )

    assert report.rejected_reasons == []
    assert len(report.clips) == 1
    assert evidence in report.clips[0]["_clip_text"]
    assert forbidden.lower() not in report.clips[0]["_clip_text"].lower()


def test_assignment_leadin_is_retained_when_its_cue_contains_grounded_teaching() -> None:
    segments = [
        {
            "cue_id": "decay-setup",
            "start": 0.0,
            "end": 8.0,
            "text": "The decay rate is proportional to the amount remaining.",
        },
        {
            "cue_id": "decay-solution",
            "start": 8.0,
            "end": 20.0,
            "text": (
                "As an exercise, solve the differential equation by separating "
                "variables. The answer is y equals c e to the negative k t."
            ),
        },
    ]
    proposal = _proposal(end_line=1).model_copy(update={
        "candidate_id": "grounded-assignment-solution",
        "start_quote": "The decay rate is proportional",
        "end_quote": "y equals c e to the negative k t",
        "title": "Solve a radioactive decay equation",
        "learning_objective": "Solve the radioactive decay differential equation",
        "facet": "differential equation solution",
        "reason": "The assignment cue contains the grounded answer.",
        "topic_evidence_quote": "The answer is y equals c e to the negative k t",
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic="radioactive decay differential equation",
    )

    assert report.rejected_reasons == []
    assert report.clips[0]["cue_ids"] == ["decay-setup", "decay-solution"]
    assert "The answer is y equals c e to the negative k t" in report.clips[0]["_clip_text"]


def test_same_unit_future_step_is_not_unconditional_trailing_noise() -> None:
    text = (
        "We will differentiate the outer function first and multiply by the "
        "inner derivative later, which completes the chain rule."
    )

    assert gemini_segment._unconditional_trailing_edge_noise_start(text) is None


@pytest.mark.parametrize(
    "text",
    [
        "As exercise intensity increases, heart rate rises.",
        "For exercise physiology, derivatives quantify the rate of change.",
        "As an exercise physiologist, I use derivatives to measure change.",
        "As an exercise in symmetry, this proof reveals the invariant.",
    ],
)
def test_substantive_exercise_phrase_is_not_an_assignment(text: str) -> None:
    assert (
        gemini_segment._unconditional_trailing_edge_noise_start(
            text,
            require_edge_prefix=True,
        )
        is None
    )


def test_explicit_assignment_opening_is_trailing_noise() -> None:
    assert (
        gemini_segment._unconditional_trailing_edge_noise_start(
            "As an exercise, differentiate x squared.",
            require_edge_prefix=True,
        )
        == 0
    )


def test_short_complete_conclusion_survives_a_trailing_assignment_cue() -> None:
    segments = [
        {
            "cue_id": "constant-explanation",
            "start": 0.0,
            "end": 8.0,
            "text": (
                "The derivative of a constant vanishes because its output "
                "never changes."
            ),
        },
        {
            "cue_id": "constant-conclusion",
            "start": 8.0,
            "end": 9.0,
            "text": "Thus zero.",
        },
        {
            "cue_id": "next-exercise",
            "start": 9.0,
            "end": 14.0,
            "text": "As an exercise, differentiate x squared.",
        },
    ]

    trimmed_end = gemini_segment._trim_trailing_incomplete_suffix(
        segments,
        0,
        2,
        protected_quote=(
            "derivative of a constant vanishes because its output"
        ),
        learning_objective="Explain why the derivative of a constant is zero",
    )

    assert trimmed_end == 1


def test_requested_formula_version_comparison_keeps_both_versions() -> None:
    text = (
        "The indefinite integration by parts identity is integral u d v equals "
        "u v minus integral v d u. So let me spell it out. So this is the "
        "definite integral's version"
    )

    end_quote, trimmed = gemini_segment._trim_end_quote_before_edge_noise(
        text,
        "So this is the definite integral's version",
        evidence_quote="integral u d v equals u v minus integral v d u",
        learning_objective=(
            "Compare the indefinite and definite integration by parts versions"
        ),
    )

    assert not trimmed
    assert end_quote == "So this is the definite integral's version"


def test_same_cue_leading_welcome_is_trimmed_from_model_start_quote() -> None:
    text = (
        "Welcome to the channel. Cells use chlorophyll to capture light energy and "
        "power the chemical reactions of photosynthesis."
    )
    proposal = _proposal().model_copy(update={
        "start_quote": (
            "Welcome to the channel. Cells use chlorophyll to capture light energy"
        ),
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        [{"cue_id": "cue-0", "start": 0.0, "end": 18.0, "text": text}],
        [],
        {"_segment_ignore_caption_case": True},
        topic="photosynthesis",
    )

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["start_quote"] == "Cells use chlorophyll to capture light"
    assert clip["edge_projection"]["start"] == {
        "required": True,
        "cue_id": "cue-0",
        "quote": "Cells use chlorophyll to capture light",
    }
    assert clip["_clip_text"].startswith("Cells use chlorophyll")
    assert "Welcome" not in clip["_clip_text"]


def test_same_cue_leading_example_frame_is_trimmed_without_losing_internal_example() -> None:
    text = (
        "Here is another quick example. Chlorophyll absorbs blue and red light for "
        "photosynthesis. For example, accessory pigments can transfer captured energy "
        "to chlorophyll."
    )
    proposal = _proposal().model_copy(update={
        "start_quote": "Here is another quick example Chlorophyll absorbs blue",
        "end_quote": "transfer captured energy to chlorophyll",
        "topic_evidence_quote": (
            "Chlorophyll absorbs blue and red light for photosynthesis"
        ),
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        [{"cue_id": "cue-0", "start": 0.0, "end": 18.0, "text": text}],
        [],
        {"_segment_ignore_caption_case": True},
        topic="photosynthesis",
    )

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["start_quote"] == "Chlorophyll absorbs blue and red light"
    assert clip["edge_projection"]["start"] == {
        "required": True,
        "cue_id": "cue-0",
        "quote": "Chlorophyll absorbs blue and red light",
    }
    assert clip["_clip_text"].startswith("Chlorophyll absorbs")
    assert "For example, accessory pigments" in clip["_clip_text"]
    assert "another quick example" not in clip["_clip_text"]


def test_leading_example_frame_does_not_hide_an_unresolved_setup_reference() -> None:
    text = "Here is another example. If we apply it here, we get the answer."

    assert gemini_segment._leading_example_framing_quote(text) == ""


def test_trailing_preview_repair_fails_closed_on_incomplete_teaching_prefix() -> None:
    text = "Cells use chlorophyll because. But we'll talk more about that next time."
    proposal = _proposal().model_copy(update={
        "start_quote": "Cells use chlorophyll",
        "end_quote": text.rstrip("."),
        "topic_evidence_quote": "Cells use chlorophyll because",
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        [{"cue_id": "cue-0", "start": 0.0, "end": 18.0, "text": text}],
        [],
        {"_segment_ignore_caption_case": True},
        topic="photosynthesis",
    )

    assert report.clips == []
    assert report.rejected_reasons == ["proposal_0:unresolved_weak_end"]


def test_same_cue_preview_inside_teaching_is_not_shipped_as_filler() -> None:
    text = (
        "Chlorophyll captures light energy for photosynthesis. But we'll talk more "
        "about that next time. Carbon fixation then converts carbon dioxide into sugar."
    )
    proposal = _proposal().model_copy(update={
        "start_quote": "Chlorophyll captures light energy",
        "end_quote": "converts carbon dioxide into sugar",
        "topic_evidence_quote": (
            "Carbon fixation then converts carbon dioxide into sugar"
        ),
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        [{"cue_id": "cue-0", "start": 0.0, "end": 18.0, "text": text}],
        [],
        {"_segment_ignore_caption_case": True},
        topic="photosynthesis",
    )

    assert report.clips == []
    assert report.rejected_reasons == [
        "proposal_0:internal_structural_filler"
    ]


def test_real_course_logistics_opening_is_trimmed_before_biology_teaching() -> None:
    segments = [
        {
            "cue_id": "mit-biology:0",
            "start": 17.683,
            "end": 48.650,
            "text": (
                "BARBARA IMPERIALI: OK. We're going to get going. Now, we have a "
                "small class this year because of changes in the institute with pass/fail "
                "types of things, but Professor Martin and Dr. Ray and I consider this to "
                "be a special opportunity for us to run the course a little bit differently "
                "with a few more quirks and surprises. Because we have a small number of you, "
                "we can listen to you all. We can get input from you. We can even get "
                "feedback from you of something you might like to see more of."
            ),
        },
        {
            "cue_id": "mit-biology:1",
            "start": 48.650,
            "end": 77.000,
            "text": (
                "And in general, we really want to capture the sense of you. I have looked "
                "at the registration list. We have people from every year. We have people "
                "from many, many different disciplines. So this is what we're going to do "
                "today after we I start doing some introductions and so on. We're going to "
                "talk about the nitty gritty of the organization. We need to tell you this. "
                "We need to convey this information to you clearly about when exams are, "
                "and what requirements are,"
            ),
        },
        {
            "cue_id": "mit-biology:2",
            "start": 77.000,
            "end": 112.610,
            "text": (
                "and how to do well in this course without even realizing it, that kind of "
                "thing. And then I'll take you through this sort of fast track through "
                "molecules to man, all the way down to cells and organisms, to show you that "
                "there was a breakpoint in the 1950s where the structure, the non-covalent "
                "structure of DNA was elucidated. And there was an entire revolution after "
                "that which makes modern biology, the study of modern biology, so entirely "
                "different from the study"
            ),
        },
        {
            "cue_id": "mit-biology:3",
            "start": 112.610,
            "end": 146.940,
            "text": (
                "of biology in the era before that. Biology used to be considered taxonomy "
                "and dissection, like listing and looking at. But now biology, modern "
                "biology, is a molecular science."
            ),
        },
    ]
    proposal = _proposal(end_line=3).model_copy(update={
        "candidate_id": "modern-biology-shift",
        "start_quote": "BARBARA IMPERIALI OK We're going to get going",
        "end_quote": "modern biology is a molecular science",
        "title": "Why modern biology became molecular",
        "learning_objective": "Explain how DNA structure changed modern biology",
        "facet": "molecular biology history",
        "reason": "The span contrasts descriptive biology with molecular biology.",
        "topic_evidence_quote": "But now biology modern biology is a molecular science",
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic="biology",
    )

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["cue_ids"] == ["mit-biology:2", "mit-biology:3"]
    assert clip["_clip_text"].startswith("there was a breakpoint in the 1950s")
    assert "pass/fail" not in clip["_clip_text"]
    assert "registration list" not in clip["_clip_text"]
    assert "when exams are" not in clip["_clip_text"]


def test_carolingian_visual_dependent_span_is_rejected() -> None:
    raw_cues = [
        (577, 2229.04, 2234.16, "tail. Um it really is just like the end of the M."),
        (578, 2249.16, 2254.36, "So we explicitly have two Rs here."),
        (579, 2252.24, 2255.96, "We have the first R,"),
        (580, 2254.36, 2258.60, "which very much looks like the R we're used to."),
        (581, 2261.32, 2264.56, "And remember to start just a little"),
        (582, 2262.44, 2267.32, "below the line and then pull your pen up"),
        (583, 2264.56, 2267.32, "and pull it through."),
        (584, 2269.08, 2274.48, "Little below the line, pull it through."),
        (585, 2272.52, 2277.68, "The second R is what you might see when it gets"),
        (586, 2275.76, 2279.96, "written off of a letter. It's kind of a"),
        (587, 2277.68, 2286.96, "ligature R. So if I get put an O over here,"),
        (588, 2283.72, 2286.96, "then I want to draw an R,"),
        (589, 2288.36, 2291.36, "I can just do that."),
        (590, 2296.96, 2301.72, "And so this is the R and this is"),
        (591, 2298.72, 2303.84, "actually called a half R."),
        (592, 2301.72, 2308.56, "And a lot of different scripts use the half R."),
        (593, 2306.56, 2310.68, "Um I have seen this in formal documents."),
        (594, 2308.56, 2313.20, "I've seen it in formal documents. So,"),
        (595, 2310.68, 2315.16, "it's not that this is considered an"),
        (596, 2313.20, 2318.08, "informal way of writing"),
        (597, 2315.16, 2320.48, "um everywhere all the time. It's okay to do."),
        (598, 2326.80, 2330.24, "So, there's the O O."),
        (599, 2331.12, 2334.28, "Now, it doesn't have to be an O. It can"),
        (600, 2332.56, 2337.96, "be, you know, pretty much any letter"),
        (601, 2334.28, 2341.28, "that'll that precedes the R that"),
        (602, 2337.96, 2344.88, "um it fills the white space better"),
        (603, 2341.28, 2351.08, "is the easy way of saying that. And so,"),
        (604, 2346.60, 2351.08, "you start off with that same stroke."),
        (605, 2351.20, 2355.52, "And but then you bring it down."),
        (606, 2353.04, 2359.36, "And it's almost like the the Z from"),
        (607, 2355.52, 2359.36, "Uncial at this point."),
        (608, 2366.20, 2371.92, "Um I have never seen"),
        (609, 2368.52, 2374.16, "the the half R not connected, not"),
        (610, 2371.92, 2376.36, "ligatured. Um that said, I haven't seen"),
        (611, 2374.16, 2379.12, "it at all. So, it there might be a time"),
        (612, 2376.36, 2380.52, "and a place where it's okay to do that."),
        (613, 2379.12, 2383.68, "We've already done S. So, we're going to"),
        (614, 2380.52, 2386.44, "switch over to T. This is my favorite T"),
    ]
    segments = [
        {
            "cue_id": f"nHMf37SMX-Q:cue:{cue_id}",
            "start": start,
            "end": end,
            "text": text,
        }
        for cue_id, start, end, text in raw_cues
    ]
    proposal = _proposal(end_line=len(segments) - 1).model_copy(update={
        "candidate_id": "carolingian-half-r",
        "start_line": 2,
        "end_line": 14,
        "start_quote": "We have the first R",
        "end_quote": "actually called a half R",
        "title": "Identifying the Carolingian half R ligature",
        "learning_objective": "Recognize the half R ligature in Carolingian minuscule",
        "facet": "ligature identification",
        "reason": "The span demonstrates and identifies the half R ligature.",
        "topic_evidence_quote": (
            "The second R is what you might see when it gets written off of a letter"
        ),
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        segments,
        [],
        {
            "_segment_target_min_sec": 20,
            "_segment_target_sec": 55,
            "_segment_target_max_sec": 55,
            "_segment_ignore_caption_case": True,
        },
        topic="Carolingian minuscule ligature identification",
    )

    assert report.clips == []
    assert "proposal_0:requires_visual_context" in report.rejected_reasons


def test_generic_look_at_phrase_is_trimmed_from_the_opening() -> None:
    text = (
        "Look at the light-dependent reactions. Chlorophyll captures photons, "
        "and the resulting electron flow helps cells make ATP."
    )
    proposal = _proposal().model_copy(update={
        "start_quote": "Look at the light-dependent reactions",
        "end_quote": "helps cells make ATP",
        "topic_evidence_quote": (
            "Chlorophyll captures photons and the resulting electron flow"
        ),
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        [{"cue_id": "cue-0", "start": 0.0, "end": 12.0, "text": text}],
        [],
        {"_segment_ignore_caption_case": True},
        topic="photosynthesis",
    )

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["start_quote"].startswith("Chlorophyll captures photons")
    assert clip["_clip_text"].startswith("Chlorophyll captures photons")
    assert "Look at" not in clip["_clip_text"]


def test_no_article_look_at_phrase_is_trimmed_from_the_opening() -> None:
    text = (
        "Look at photosynthesis. Chlorophyll captures photons, and the resulting "
        "electron flow helps cells make ATP."
    )
    proposal = _proposal().model_copy(update={
        "start_quote": "Look at photosynthesis",
        "end_quote": "helps cells make ATP",
        "topic_evidence_quote": (
            "Chlorophyll captures photons and the resulting electron flow"
        ),
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        [{"cue_id": "cue-0", "start": 0.0, "end": 12.0, "text": text}],
        [],
        {"_segment_ignore_caption_case": True},
        topic="photosynthesis",
    )

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["_clip_text"].startswith("Chlorophyll captures photons")
    assert "Look at" not in clip["_clip_text"]


def test_bare_look_at_this_remains_visual_dependent() -> None:
    text = (
        "Look at this. Chlorophyll captures photons, and the resulting electron "
        "flow helps cells make ATP."
    )
    proposal = _proposal().model_copy(update={
        "start_quote": "Look at this",
        "end_quote": "helps cells make ATP",
        "topic_evidence_quote": (
            "Chlorophyll captures photons and the resulting electron flow"
        ),
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        [{"cue_id": "cue-0", "start": 0.0, "end": 12.0, "text": text}],
        [],
        {"_segment_ignore_caption_case": True},
        topic="photosynthesis",
    )

    assert report.clips == []
    assert report.rejected_reasons == ["proposal_0:requires_visual_context"]


def test_preserved_video_url_does_not_claim_unseen_visual_grounding() -> None:
    text = (
        "Look at this. Chlorophyll captures photons, and the arrows show how "
        "electron flow helps cells make ATP."
    )
    proposal = _proposal().model_copy(update={
        "start_quote": "Look at this",
        "end_quote": "helps cells make ATP",
        "topic_evidence_quote": (
            "Chlorophyll captures photons and the arrows show how electron"
        ),
    })

    for grounding_state in (
        {},
        {"_segment_video_grounded": False},
    ):
        report = gemini_segment._plan_to_report(
            gemini_segment._BoundaryPlan(topics=[proposal]),
            [{"cue_id": "cue-0", "start": 0.0, "end": 12.0, "text": text}],
            [],
            {
                "_segment_ignore_caption_case": True,
                "_segment_video_url": (
                    "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
                ),
                **grounding_state,
            },
            topic="photosynthesis",
        )

        assert report.clips == []
        assert report.rejected_reasons == ["proposal_0:requires_visual_context"]


@pytest.mark.parametrize(
    "opening",
    [
        "Look at how chlorophyll captures photons by exciting electrons.",
        "Look at photosynthesis because it captures light and stores energy.",
    ],
)
def test_substantive_look_at_clause_is_not_classified_as_filler(
    opening: str,
) -> None:
    assert gemini_segment._structural_filler_matches(opening) == []


def test_look_at_visual_noun_remains_visual_dependent() -> None:
    text = (
        "Look at the diagram. Chlorophyll captures photons, and the arrows show "
        "how electron flow helps cells make ATP."
    )
    proposal = _proposal().model_copy(update={
        "start_quote": "Look at the diagram",
        "end_quote": "helps cells make ATP",
        "topic_evidence_quote": (
            "Chlorophyll captures photons and the arrows show how electron flow"
        ),
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        [{"cue_id": "cue-0", "start": 0.0, "end": 12.0, "text": text}],
        [],
        {"_segment_ignore_caption_case": True},
        topic="photosynthesis",
    )

    assert report.clips == []
    assert report.rejected_reasons == ["proposal_0:requires_visual_context"]


def test_video_grounded_selector_may_keep_a_legible_required_diagram() -> None:
    text = (
        "Look at the diagram. Chlorophyll captures photons, and the arrows show "
        "how electron flow helps cells make ATP."
    )
    proposal = _proposal().model_copy(update={
        "start_quote": "Look at the diagram",
        "end_quote": "helps cells make ATP",
        "topic_evidence_quote": (
            "Chlorophyll captures photons and the arrows show how electron flow"
        ),
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        [{"cue_id": "cue-0", "start": 0.0, "end": 12.0, "text": text}],
        [],
        {
            "_segment_ignore_caption_case": True,
            "_segment_video_grounding_required": True,
            "_segment_video_url": (
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
            ),
        },
        topic="photosynthesis",
    )

    assert report.rejected_reasons == []
    assert len(report.clips) == 1


def test_articleless_look_at_visual_noun_remains_visual_dependent() -> None:
    text = (
        "Look at diagram. Chlorophyll captures photons, and the arrows show how "
        "electron flow helps cells make ATP."
    )
    proposal = _proposal().model_copy(update={
        "start_quote": "Look at diagram",
        "end_quote": "helps cells make ATP",
        "topic_evidence_quote": (
            "Chlorophyll captures photons and the arrows show how electron flow"
        ),
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        [{"cue_id": "cue-0", "start": 0.0, "end": 12.0, "text": text}],
        [],
        {"_segment_ignore_caption_case": True},
        topic="photosynthesis",
    )

    assert report.clips == []
    assert report.rejected_reasons == ["proposal_0:requires_visual_context"]


def test_short_topic_sentence_with_anaphoric_explanation_remains_a_valid_start() -> None:
    segments = [{
        "cue_id": "photosynthesis:cue:0",
        "start": 0.0,
        "end": 12.0,
        "text": (
            "Photosynthesis. It converts light energy into chemical energy that cells use."
        ),
    }]
    proposal = _proposal().model_copy(update={
        "start_quote": "Photosynthesis",
        "end_quote": "chemical energy that cells use",
        "topic_evidence_quote": (
            "It converts light energy into chemical energy that cells use"
        ),
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic="photosynthesis",
    )

    assert report.rejected_reasons == []
    assert report.clips[0]["start"] == 0.0


def test_genetic_drift_callback_end_extends_through_its_explanation() -> None:
    texts = [
        (
            "One of the biggest criticisms against The Selfish Gene is that it leaves little "
            "to chance. But many genes are invisible to natural selection. Imagine 20 blind "
            "cave fish, 10 with green eyes and 10 with blue."
        ),
        (
            "Their eye colors make no difference to survival, so they are passed down purely "
            "by chance. Repeating random selection changes the next generation."
        ),
        (
            "This shift in the frequency of gene variants is called genetic drift. It is most "
            "apparent in small populations. Look back at our replicator battle."
        ),
        (
            "If we run our simulation enough times, sometimes the winning gene will not have "
            "the traits that maximize survival. These examples show how much evolution can be "
            "due to natural selection and how much is up to chance."
        ),
    ]
    times = [(1394.24, 1421.76), (1421.76, 1447.12), (1447.12, 1473.36),
             (1473.36, 1499.12)]
    segments = [
        {
            "cue_id": f"XX7PdJIGiCw:cue:{index + 50}",
            "start": start,
            "end": end,
            "text": text,
        }
        for index, (text, (start, end)) in enumerate(zip(texts, times))
    ]
    proposal = _proposal(end_line=2).model_copy(update={
        "candidate_id": "genetic-drift",
        "start_quote": "One of the biggest criticisms",
        "end_quote": "Look back at our replicator battle",
        "title": "Genetic drift from random sampling",
        "learning_objective": "Explain how random sampling changes gene frequencies",
        "facet": "evolution",
        "reason": "The fish example explains genetic drift.",
        "topic_evidence_quote": (
            "This shift in the frequency of gene variants is called genetic drift"
        ),
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic="biology",
    )

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["cue_ids"][-1] == "XX7PdJIGiCw:cue:53"
    assert not clip["_clip_text"].endswith("Look back at our replicator battle.")
    assert clip["_clip_text"].endswith("how much is up to chance.")
    assert clip["informativeness"] == 0.9
    assert clip["uncertainty"] == "medium"


@pytest.mark.parametrize(
    "field",
    ["informativeness", "topic_relevance", "educational_importance"],
)
def test_each_quality_score_is_an_independent_numeric_hard_gate(field: str) -> None:
    segments = [{
        "start": 0.0,
        "end": 12.0,
        "text": (
            "Cells use chlorophyll to capture light energy and power the chemical "
            "reactions of photosynthesis."
        ),
    }]
    rejected = _proposal().model_copy(update={field: 0.74})
    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[rejected]),
        segments,
        [],
        {},
        topic="photosynthesis",
    )
    assert report.clips == []
    assert report.rejected_reasons == [f"proposal_0:{field}_below_green"]

    accepted = _proposal().model_copy(update={
        "informativeness": 0.75,
        "topic_relevance": 0.75,
        "educational_importance": 0.75,
    })
    accepted_report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[accepted]),
        segments,
        [],
        {},
        topic="photosynthesis",
    )
    assert len(accepted_report.clips) == 1


def test_context_expands_beyond_eight_cues_and_thirty_seconds() -> None:
    texts = [
        "A worked example begins with two values and",
        "we substitute both values into the equation and",
        "then simplify the first expression and",
        "carry the coefficient to the other side and",
        "combine the matching terms together and",
        "divide both sides by the coefficient and",
        "check the sign of the resulting value and",
        "substitute the result into the original equation and",
        "verify that both sides now agree and",
        "state the meaning of the solution and",
        "the calculation finishes with x equals two.",
    ]
    segments = [
        {"start": index * 5.0, "end": (index + 1) * 5.0, "text": text}
        for index, text in enumerate(texts)
    ]
    proposal = _proposal().model_copy(update={
        "candidate_id": "worked-example",
        "start_quote": "A worked example begins",
        "end_quote": "two values and",
        "title": "Solving the equation",
        "learning_objective": "Solve the equation through its verified result",
        "facet": "worked example",
        "reason": "The complete worked example reaches and checks its answer.",
        "topic_evidence_quote": "we substitute both values into the equation and",
    })
    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic="equation worked example",
    )
    assert report.rejected_reasons == []
    assert report.clips[0]["_end_line"] == 10
    assert report.clips[0]["end"] == 55.0


def test_real_calculus_example_intro_expands_past_dangling_or_even() -> None:
    raw_segments = [
        (114.430, 118.979, "But how are these changing quantities related to one another now? What is the formula for"),
        (118.979, 125.200, "this change? Again, the answer lies with calculus."),
        (125.200, 129.929, "So in order to tackle the problem of changing quantities calculus picks up three powerful"),
        (129.929, 134.980, "tools. These tools are: limits, derivatives, and"),
        (134.980, 139.569, "integrals. Now there are many other things you'll learn in calculus, but these 3 things"),
        (139.569, 142.879, "are the most essential. Because of this you'll want to spend as"),
        (142.879, 148.900, "much time with them as possible. Limits are the tools we use for precisely"),
        (148.900, 153.790, "describing how a function approaches a value. Derivatives are the tools we use for describing"),
        (153.790, 157.459, "how a function changes, and integrals give us the area underneith"),
        (157.459, 161.900, "the curve of a function. Using limits, derivatives and integrals calculus"),
        (161.900, 167.379, "can solve a variety of problems like where sit in a theater for optimal viewing, or even"),
        (167.379, 172.470, "how to make the perfect soup can. One of the most fascinating aspects of calculus"),
        (172.470, 176.140, "is how all of these tools are actually related to one another."),
    ]
    segments = [
        {"cue_id": f"calculus:{index}", "start": start, "end": end, "text": text}
        for index, (start, end, text) in enumerate(raw_segments)
    ]
    proposal = _proposal(end_line=10).model_copy(update={
        "candidate_id": "calculus-core-tools",
        "start_quote": "But how are these changing quantities related",
        "end_quote": "theater for optimal viewing or even",
        "title": "The three core tools of calculus",
        "learning_objective": "Explain what limits, derivatives, and integrals describe",
        "facet": "core calculus tools",
        "reason": "The span defines the central tools and what they solve.",
        "topic_evidence_quote": (
            "Limits are the tools we use for precisely describing how a function"
        ),
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic="calculus",
    )

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["_end_line"] == 11
    assert "how to make the perfect soup can" in clip["_clip_text"]
    assert clip["_clip_text"].endswith("how to make the perfect soup can.")
    assert "One of the most fascinating aspects" not in clip["_clip_text"]


def test_demonstrative_calculus_opening_expands_to_its_cold_viewer_setup() -> None:
    texts = [
        "Here is another quick example. If I want to model the volume of a balloon,",
        "you might assume that it is approximately a sphere, and use the sphere formula",
        "pi times the radius cubed. This shows that the volume of the balloon",
        "is related to the radius. Now when I let air out, things start",
        "to change. The volume is decreasing, and so is the radius.",
        "But how are these changing quantities related? What is the formula for",
        "this change? Again, the answer lies with calculus.",
        "So in order to tackle changing quantities calculus uses three powerful tools.",
        "These tools are limits, derivatives, and integrals.",
    ]
    segments = [
        {
            "cue_id": f"calculus-context:{index}",
            "start": index * 5.0,
            "end": (index + 1) * 5.0,
            "text": text,
        }
        for index, text in enumerate(texts)
    ]
    proposal = _proposal(end_line=8).model_copy(update={
        "candidate_id": "calculus-context-chain",
        "start_line": 6,
        "start_quote": "this change Again the answer lies",
        "end_quote": "tools are limits derivatives and integrals",
        "title": "Calculus tools for changing quantities",
        "learning_objective": "Explain why calculus uses limits, derivatives, and integrals",
        "facet": "calculus tools",
        "reason": "The balloon setup supplies the antecedent for changing quantities.",
        "topic_evidence_quote": "tools are limits derivatives and integrals",
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic="calculus",
    )

    assert report.rejected_reasons == []
    assert report.clips[0]["cue_ids"][0] == "calculus-context:0"
    assert report.clips[0]["start_quote"].startswith("If I want")
    assert report.clips[0]["_clip_text"].startswith("If I want")
    assert "Here is another quick example" not in report.clips[0]["_clip_text"]


def test_complete_answer_trims_a_dangling_final_phrase_instead_of_rejecting() -> None:
    text = (
        "Let h of x equal sine of x squared. The chain rule differentiates the outer "
        "sine and multiplies by the inner derivative two x. Therefore h prime of x "
        "equals two x cosine of x squared. And"
    )
    proposal = _proposal().model_copy(update={
        "candidate_id": "chain-rule-answer",
        "start_quote": "Let h of x equal sine of x squared",
        "end_quote": "two x cosine of x squared And",
        "title": "Complete chain rule derivative",
        "learning_objective": "Apply the chain rule through the final derivative",
        "facet": "worked example",
        "reason": "The worked example reaches its final answer.",
        "topic_evidence_quote": (
            "Therefore h prime of x equals two x cosine of x squared"
        ),
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        [{"cue_id": "chain-rule:0", "start": 0.0, "end": 28.0, "text": text}],
        [],
        {"_segment_ignore_caption_case": True},
        topic="chain rule worked example",
    )

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["_clip_text"].endswith("two x cosine of x squared.")
    assert "trimmed_incomplete_end_suffix" in clip["_boundary_fallback_reasons"]


def test_topic_transition_keeps_only_the_learning_objective_containing_evidence() -> None:
    segments = [
        {
            "cue_id": "limits",
            "start": 0.0,
            "end": 8.0,
            "text": "The limit equals two, which completes the limits problem.",
        },
        {
            "cue_id": "derivative-transition",
            "start": 8.0,
            "end": 18.0,
            "text": (
                "Now let's move on to derivatives. A derivative measures the "
                "instantaneous rate of change of a function."
            ),
        },
        {
            "cue_id": "derivative-example",
            "start": 18.0,
            "end": 28.0,
            "text": (
                "For example, velocity is the derivative of position with respect to time."
            ),
        },
    ]
    derivative = _proposal(end_line=2).model_copy(update={
        "candidate_id": "derivatives-after-limits",
        "start_quote": "The limit equals two which completes",
        "end_quote": "derivative of position with respect to time",
        "title": "What a derivative measures",
        "learning_objective": "Explain derivatives as instantaneous rates of change",
        "facet": "derivatives",
        "reason": "The span defines derivatives and gives a velocity example.",
        "topic_evidence_quote": (
            "A derivative measures the instantaneous rate of change of a function"
        ),
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[derivative]),
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic="calculus",
    )

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["cue_ids"] == ["derivative-transition", "derivative-example"]
    assert clip["_clip_text"].startswith("A derivative measures")
    assert "limit equals two" not in clip["_clip_text"]
    assert "move on to derivatives" not in clip["_clip_text"]


def test_same_cue_topic_transition_still_removes_the_previous_objective() -> None:
    text = (
        "The limit equals two, which completes the limits problem. "
        "Now let's move on to derivatives. A derivative measures the instantaneous "
        "rate of change of a function."
    )
    derivative = _proposal().model_copy(update={
        "candidate_id": "same-cue-derivative-transition",
        "start_quote": "The limit equals two which completes",
        "end_quote": "rate of change of a function",
        "title": "What a derivative measures",
        "learning_objective": "Explain derivatives as instantaneous rates of change",
        "facet": "derivatives",
        "reason": "The retained section defines derivatives.",
        "topic_evidence_quote": (
            "A derivative measures the instantaneous rate of change of a function"
        ),
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[derivative]),
        [{"cue_id": "calculus:mixed", "start": 0.0, "end": 20.0, "text": text}],
        [],
        {"_segment_ignore_caption_case": True},
        topic="calculus",
    )

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["_clip_text"].startswith("A derivative measures")
    assert "limit equals two" not in clip["_clip_text"]
    assert "move on to derivatives" not in clip["_clip_text"]


def test_transition_cue_drops_old_topic_prefix_before_new_evidence() -> None:
    segments = [
        {
            "cue_id": "mixed",
            "start": 0.0,
            "end": 8.0,
            "text": (
                "The limit equals two, which completes the limits problem. "
                "Now let's move on to derivatives."
            ),
        },
        {
            "cue_id": "derivative",
            "start": 8.0,
            "end": 16.0,
            "text": (
                "A derivative measures the instantaneous rate of change of a function."
            ),
        },
    ]
    proposal = _proposal(end_line=1).model_copy(update={
        "candidate_id": "derivative-only",
        "start_quote": "The limit equals two",
        "end_quote": "rate of change of a function",
        "title": "Derivative definition",
        "learning_objective": "Explain derivatives as instantaneous rates of change",
        "facet": "derivatives",
        "topic_evidence_quote": (
            "A derivative measures the instantaneous rate of change"
        ),
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic="calculus",
    )

    assert report.rejected_reasons == []
    assert report.clips[0]["cue_ids"] == ["derivative"]
    assert report.clips[0]["_clip_text"] == (
        "A derivative measures the instantaneous rate of change of a function."
    )


def test_transition_cue_keeps_answer_prefix_before_next_topic() -> None:
    segments = [
        {
            "cue_id": "setup",
            "start": 0.0,
            "end": 8.0,
            "text": "We apply the power rule to x squared.",
        },
        {
            "cue_id": "answer-transition",
            "start": 8.0,
            "end": 18.0,
            "text": (
                "Therefore the derivative is two x. Now let's move on to integrals."
            ),
        },
    ]
    proposal = _proposal(end_line=1).model_copy(update={
        "candidate_id": "power-rule-answer",
        "start_quote": "We apply the power rule",
        "end_quote": "move on to integrals",
        "title": "Power rule answer",
        "learning_objective": "Differentiate x squared through its final answer",
        "facet": "worked example",
        "topic_evidence_quote": "We apply the power rule to x squared",
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic="power rule worked example",
    )

    assert report.rejected_reasons == []
    assert report.clips[0]["cue_ids"] == ["setup", "answer-transition"]
    assert report.clips[0]["_clip_text"].rstrip(".").endswith(
        "Therefore the derivative is two x"
    )
    assert "integrals" not in report.clips[0]["_clip_text"]


def test_next_navigation_in_one_cue_drops_the_previous_topic() -> None:
    text = (
        "Photosynthesis releases oxygen. Next we'll discuss chloroplast structure. "
        "Chloroplasts contain thylakoid membranes for the light reactions."
    )
    proposal = _proposal().model_copy(update={
        "candidate_id": "chloroplast-structure",
        "start_quote": "Photosynthesis releases oxygen",
        "end_quote": "membranes for the light reactions",
        "title": "Chloroplast structure",
        "learning_objective": "Explain how thylakoid membranes support light reactions",
        "facet": "chloroplast structure",
        "topic_evidence_quote": (
            "Chloroplasts contain thylakoid membranes for the light reactions"
        ),
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        [{"cue_id": "mixed", "start": 0.0, "end": 18.0, "text": text}],
        [],
        {"_segment_ignore_caption_case": True},
        topic="chloroplast structure",
    )

    assert report.rejected_reasons == []
    assert report.clips[0]["_clip_text"].startswith("Chloroplasts contain")
    assert "Photosynthesis releases" not in report.clips[0]["_clip_text"]


@pytest.mark.parametrize(
    "navigation",
    [
        "Now let us discuss derivatives.",
        "Now let's cover derivatives.",
        "Now let us talk about derivatives.",
    ],
)
def test_named_topic_navigation_cue_drops_the_previous_topic(
    navigation: str,
) -> None:
    segments = [
        {
            "cue_id": "limits",
            "start": 0.0,
            "end": 6.0,
            "text": "Limits describe values approached by functions.",
        },
        {"cue_id": "navigation", "start": 6.0, "end": 9.0, "text": navigation},
        {
            "cue_id": "derivatives",
            "start": 9.0,
            "end": 16.0,
            "text": "Derivatives measure instantaneous rates of change.",
        },
    ]
    proposal = _proposal(end_line=2).model_copy(update={
        "candidate_id": "derivatives",
        "start_quote": "Limits describe values",
        "end_quote": "instantaneous rates of change",
        "title": "Derivative definition",
        "learning_objective": "Explain derivatives as rates of change",
        "facet": "derivatives",
        "topic_evidence_quote": "Derivatives measure instantaneous rates of change",
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic="derivatives",
    )

    assert report.rejected_reasons == []
    assert report.clips[0]["cue_ids"] == ["derivatives"]


def test_named_reset_overrides_one_shared_token_before_the_transition() -> None:
    text = (
        "Photosynthesis converts light into glucose. Now let us discuss respiration. "
        "Respiration breaks glucose down to release energy."
    )
    proposal = _proposal().model_copy(update={
        "candidate_id": "respiration",
        "start_quote": "Photosynthesis converts light",
        "end_quote": "glucose down to release energy",
        "title": "Cellular respiration",
        "learning_objective": "Explain how respiration releases energy from glucose",
        "facet": "respiration",
        "topic_evidence_quote": "Respiration breaks glucose down to release energy",
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        [{"cue_id": "mixed", "start": 0.0, "end": 18.0, "text": text}],
        [],
        {"_segment_ignore_caption_case": True},
        topic="cellular respiration",
    )

    assert report.rejected_reasons == []
    assert report.clips[0]["_clip_text"].startswith("Respiration breaks glucose")


def test_relational_word_overlap_does_not_merge_a_named_adjacent_topic() -> None:
    text = (
        "Photosynthesis converts light into glucose. Now let us discuss respiration. "
        "Respiration breaks glucose down to release energy."
    )
    proposal = _proposal().model_copy(update={
        "candidate_id": "photosynthesis",
        "start_quote": "Photosynthesis converts light",
        "end_quote": "glucose down to release energy",
        "title": "Photosynthesis produces glucose",
        "learning_objective": "Explain how photosynthesis produces glucose",
        "facet": "photosynthesis",
        "topic_evidence_quote": "Photosynthesis converts light into glucose",
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        [{"cue_id": "mixed", "start": 0.0, "end": 18.0, "text": text}],
        [],
        {"_segment_ignore_caption_case": True},
        topic="photosynthesis",
    )

    assert report.rejected_reasons == []
    assert report.clips[0]["_clip_text"] == (
        "Photosynthesis converts light into glucose."
    )


def test_relational_reset_requires_more_than_one_anchor_only_on_the_new_side() -> None:
    assert not gemini_segment._objective_bridges_sections(
        "Explain how indefinite integration by parts yields a boundary formula",
        "Indefinite integration by parts moves the derivative between factors.",
        "Boundary conditions determine a radioactive decay solution.",
        reset_subject="boundary conditions",
    )


def test_explicit_one_concept_to_one_concept_comparison_can_bridge_a_reset() -> None:
    assert gemini_segment._objective_bridges_sections(
        "Compare limits with derivatives",
        "A limit describes an approached value.",
        "A derivative describes instantaneous change.",
        reset_subject="derivatives",
    )


def test_explicit_comparison_can_bridge_when_setup_already_names_both_sides() -> None:
    assert gemini_segment._objective_bridges_sections(
        "Compare limits with derivatives",
        "Limits and derivatives differ in what they measure.",
        "Derivatives measure instantaneous change.",
        reset_subject="derivatives",
    )


def test_comparison_does_not_bridge_on_only_a_shared_head_noun() -> None:
    assert not gemini_segment._objective_bridges_sections(
        "Compare opportunity cost with sunk cost",
        "Cost represents a tradeoff.",
        "Cost accounting assigns expenditures to categories.",
        reset_subject="cost accounting",
    )


def test_teaching_inside_a_reset_sentence_starts_at_the_named_subject() -> None:
    text = (
        "A limit describes approach. Now let us move on to derivatives, which measure "
        "instantaneous change."
    )
    proposal = _proposal().model_copy(update={
        "candidate_id": "derivatives",
        "start_quote": "A limit describes approach",
        "end_quote": "which measure instantaneous change",
        "title": "Derivative definition",
        "learning_objective": "Explain derivatives",
        "facet": "derivatives",
        "topic_evidence_quote": "derivatives which measure instantaneous change",
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        [{"cue_id": "mixed", "start": 0.0, "end": 12.0, "text": text}],
        [],
        {"_segment_ignore_caption_case": True},
        topic="derivatives",
    )

    assert report.rejected_reasons == []
    assert report.clips[0]["_clip_text"].startswith("derivatives, which measure")
    assert "limit describes" not in report.clips[0]["_clip_text"]


def test_next_navigation_tail_does_not_leave_the_word_next() -> None:
    text = "The derivative of x squared is two x. Next we'll discuss integrals."
    proposal = _proposal().model_copy(update={
        "candidate_id": "derivative-answer",
        "start_quote": "The derivative of x squared",
        "end_quote": "we'll discuss integrals",
        "title": "Derivative of x squared",
        "learning_objective": "Differentiate x squared",
        "facet": "derivative",
        "topic_evidence_quote": "The derivative of x squared is two x",
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        [{"cue_id": "tail", "start": 0.0, "end": 10.0, "text": text}],
        [],
        {"_segment_ignore_caption_case": True},
        topic="derivative of x squared",
    )

    assert report.rejected_reasons == []
    assert report.clips[0]["_clip_text"].rstrip(" .!?").endswith("two x")
    assert not report.clips[0]["_clip_text"].endswith("Next")


def test_relational_objective_may_span_an_explicit_topic_transition() -> None:
    text = (
        "A limit describes the value a function approaches. Now let's move on to "
        "derivatives. A derivative is defined by a limit of difference quotients, so "
        "the two ideas are directly connected."
    )
    relationship = _proposal().model_copy(update={
        "candidate_id": "limits-define-derivatives",
        "start_quote": "A limit describes the value a function approaches",
        "end_quote": "two ideas are directly connected",
        "title": "How limits define derivatives",
        "learning_objective": "Explain how limits define derivatives",
        "facet": "limits and derivatives relationship",
        "reason": "The span explicitly relates the two calculus ideas.",
        "topic_evidence_quote": (
            "A derivative is defined by a limit of difference quotients"
        ),
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[relationship]),
        [{"cue_id": "calculus:relationship", "start": 0.0, "end": 24.0, "text": text}],
        [],
        {"_segment_ignore_caption_case": True},
        topic="calculus",
    )

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert "A limit describes" in clip["_clip_text"]
    assert "A derivative is defined by a limit" in clip["_clip_text"]


@pytest.mark.parametrize(
    "objective",
    [
        "Explain why the derivative definition uses limits",
        "Explain derivatives in terms of limits",
        "Explain the connection between limits and derivatives",
        "Show how taking a limit yields the derivative",
        "Derive the derivative from limits",
        "Explain the difference quotient limit that produces a derivative",
    ],
)
def test_relational_objective_must_anchor_both_sides_of_the_actual_reset(
    objective: str,
) -> None:
    text = (
        "A limit describes the value approached by a function. Now let us move on to "
        "derivatives. A derivative is defined as the limit of difference quotients."
    )
    proposal = _proposal().model_copy(update={
        "candidate_id": "limits-and-derivatives",
        "start_quote": "A limit describes the value",
        "end_quote": "the limit of difference quotients",
        "title": "Limits in the derivative definition",
        "learning_objective": objective,
        "facet": "limits and derivatives",
        "topic_evidence_quote": "A derivative is defined as the limit of difference quotients",
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        [{"cue_id": "relationship", "start": 0.0, "end": 20.0, "text": text}],
        [],
        {"_segment_ignore_caption_case": True},
        topic="limits and derivatives",
    )

    assert report.rejected_reasons == []
    assert report.clips[0]["_clip_text"].startswith("A limit describes")
    assert report.clips[0]["_clip_text"].rstrip(" .!?").endswith(
        "difference quotients"
    )


@pytest.mark.parametrize(
    "objective",
    [
        "Explain how derivatives affect velocity",
        "Explain derivatives in terms of velocity",
        "Explain the connection between derivatives and velocity",
        "Derive velocity change from the derivative",
    ],
)
def test_relation_to_a_third_concept_does_not_bridge_an_unrelated_reset(
    objective: str,
) -> None:
    text = (
        "Limits describe values approached by functions. Now let us move on to "
        "derivatives. Derivatives affect velocity by measuring its rate of change."
    )
    proposal = _proposal().model_copy(update={
        "candidate_id": "derivative-velocity",
        "start_quote": "Limits describe values",
        "end_quote": "measuring its rate of change",
        "title": "Derivatives and velocity",
        "learning_objective": objective,
        "facet": "derivative application",
        "topic_evidence_quote": "Derivatives affect velocity by measuring its rate of change",
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        [{"cue_id": "mixed", "start": 0.0, "end": 18.0, "text": text}],
        [],
        {"_segment_ignore_caption_case": True},
        topic="derivatives and velocity",
    )

    assert report.rejected_reasons == []
    assert report.clips[0]["_clip_text"].startswith("Derivatives affect velocity")
    assert "Limits describe" not in report.clips[0]["_clip_text"]


@pytest.mark.parametrize(
    ("segments", "objective", "evidence", "expected_start", "expected_end"),
    [
        (
            [
                "To approximate the integral, divide the interval into subintervals.",
                "Now let's cover the interval with rectangles.",
                "Adding their areas gives the Riemann sum approximation.",
            ],
            "Approximate the integral by covering its subintervals with rectangles",
            "divide the interval into subintervals",
            "To approximate the integral",
            "Riemann sum approximation.",
        ),
        (
            [
                "We need to integrate over the circular region.",
                "Now let's switch to polar coordinates.",
                "The Jacobian contributes r and the integral simplifies.",
            ],
            "Solve the circular-region integral using polar coordinates",
            "The Jacobian contributes r and the integral simplifies",
            "We need to integrate",
            "integral simplifies.",
        ),
        (
            [
                "The equation couples x and y.",
                "Now let's turn to new variables u and v.",
                "Substitution separates the equation and gives the solution.",
            ],
            "Solve the coupled equation using new variables u and v",
            "Substitution separates the equation and gives the solution",
            "The equation couples",
            "gives the solution.",
        ),
    ],
)
def test_method_navigation_inside_one_objective_preserves_the_complete_arc(
    segments: list[str],
    objective: str,
    evidence: str,
    expected_start: str,
    expected_end: str,
) -> None:
    cues = [
        {"cue_id": f"cue-{index}", "start": index * 6.0, "end": (index + 1) * 6.0, "text": text}
        for index, text in enumerate(segments)
    ]
    proposal = _proposal(end_line=2).model_copy(update={
        "candidate_id": "complete-method",
        "start_quote": " ".join(segments[0].split()[:5]),
        "end_quote": " ".join(segments[-1].split()[-5:]),
        "title": "Complete method",
        "learning_objective": objective,
        "facet": "worked method",
        "topic_evidence_quote": evidence,
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        cues,
        [],
        {"_segment_ignore_caption_case": True},
        topic=objective,
    )

    assert report.rejected_reasons == []
    assert report.clips[0]["_clip_text"].startswith(expected_start)
    assert report.clips[0]["_clip_text"].endswith(expected_end)


@pytest.mark.parametrize(
    ("text", "start_quote", "end_quote", "evidence_quote"),
    [
        (
            "A large class of proteins transports ions across the membrane.",
            "A large class of proteins transports",
            "transports ions across the membrane",
            "A large class of proteins transports ions across the membrane",
        ),
        (
            "Enrollment bias can threaten the validity of an observational study.",
            "Enrollment bias can threaten the validity",
            "validity of an observational study",
            "Enrollment bias can threaten the validity of an observational study",
        ),
        (
            "Deadline scheduling is NP-hard in this machine scheduling model.",
            "Deadline scheduling is NP-hard in",
            "this machine scheduling model",
            "Deadline scheduling is NP-hard in this machine scheduling model",
        ),
        (
            "Voter registration protects access to democratic participation in elections.",
            "Voter registration protects access to democratic",
            "democratic participation in elections",
            "Voter registration protects access to democratic participation in elections",
        ),
        (
            "Pass/fail grading changes student incentives and can affect motivation.",
            "Pass/fail grading changes student incentives",
            "and can affect motivation",
            "Pass/fail grading changes student incentives and can affect motivation",
        ),
        (
            "We need to tell you this theorem follows from compactness.",
            "We need to tell you this theorem",
            "this theorem follows from compactness",
            "We need to tell you this theorem follows from compactness",
        ),
        (
            "There are students in the treatment group and controls in the comparison group.",
            "There are students in the treatment group",
            "controls in the comparison group",
            "students in the treatment group and controls in the comparison group",
        ),
    ],
)
def test_subject_matter_admin_vocabulary_is_not_misclassified_as_edge_filler(
    text: str,
    start_quote: str,
    end_quote: str,
    evidence_quote: str,
) -> None:
    proposal = _proposal().model_copy(update={
        "candidate_id": "ambiguous-admin-vocabulary",
        "start_quote": start_quote,
        "end_quote": end_quote,
        "title": "Grounded subject matter",
        "learning_objective": "Explain the grounded subject-matter claim",
        "facet": "subject matter",
        "reason": "The sentence teaches the requested concept.",
        "topic_evidence_quote": evidence_quote,
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        [{"cue_id": "teaching", "start": 0.0, "end": 10.0, "text": text}],
        [],
        {"_segment_ignore_caption_case": True},
        topic="subject matter",
    )

    assert report.rejected_reasons == []
    assert len(report.clips) == 1


def test_instructional_preview_is_retained_when_trimming_would_start_on_an_anaphor() -> None:
    text = (
        "I'll walk you through the chain rule to show you why it multiplies the "
        "outer derivative by the inner derivative."
    )
    proposal = _proposal().model_copy(update={
        "candidate_id": "chain-rule-preview-context",
        "start_quote": "I'll walk you through the chain rule",
        "end_quote": "outer derivative by the inner derivative",
        "title": "Why the chain rule multiplies derivatives",
        "learning_objective": "Explain why the chain rule multiplies inner and outer derivatives",
        "facet": "chain rule",
        "reason": "The opening supplies the antecedent required by the explanation.",
        "topic_evidence_quote": (
            "chain rule to show you why it multiplies the outer derivative"
        ),
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        [{"cue_id": "chain-rule", "start": 0.0, "end": 12.0, "text": text}],
        [],
        {"_segment_ignore_caption_case": True},
        topic="chain rule",
    )

    assert report.rejected_reasons == []
    assert report.clips[0]["_clip_text"].startswith("I'll walk you through the chain rule")


@pytest.mark.parametrize(
    "text",
    [
        "Many of these compounds are stable under ordinary laboratory conditions",
        "One of these enzymes catalyzes the final reaction efficiently",
        "All of the measured samples remain within the expected confidence interval",
    ],
)
def test_complete_unpunctuated_nominal_sentences_are_not_dangling(text: str) -> None:
    words = text.split()
    proposal = _proposal().model_copy(update={
        "candidate_id": "complete-nominal-sentence",
        "start_quote": " ".join(words[:6]),
        "end_quote": " ".join(words[-6:]),
        "title": "Complete explanatory claim",
        "learning_objective": "Understand the complete explanatory claim",
        "facet": "complete claim",
        "reason": "The caption contains a subject and a finite predicate.",
        "topic_evidence_quote": " ".join(words[: min(12, len(words))]),
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        [{"cue_id": "complete", "start": 0.0, "end": 9.0, "text": text}],
        [],
        {"_segment_ignore_caption_case": True},
        topic="complete claim",
    )

    assert report.rejected_reasons == []
    assert len(report.clips) == 1


def test_nominal_subject_expands_only_when_next_cue_supplies_its_predicate() -> None:
    segments = [
        {
            "cue_id": "subject",
            "start": 0.0,
            "end": 6.0,
            "text": "One of the most fascinating aspects of calculus",
        },
        {
            "cue_id": "predicate",
            "start": 6.0,
            "end": 13.0,
            "text": "is how limits, derivatives, and integrals relate to one another.",
        },
    ]
    proposal = _proposal().model_copy(update={
        "candidate_id": "calculus-relationship",
        "start_quote": "One of the most fascinating aspects",
        "end_quote": "most fascinating aspects of calculus",
        "title": "How calculus tools relate",
        "learning_objective": "Explain how limits, derivatives, and integrals relate",
        "facet": "calculus relationships",
        "reason": "The next cue supplies the predicate and completes the claim.",
        "topic_evidence_quote": "most fascinating aspects of calculus",
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic="calculus",
    )

    assert report.rejected_reasons == []
    assert report.clips[0]["cue_ids"] == ["subject", "predicate"]
    assert report.clips[0]["_clip_text"].endswith("relate to one another.")


@pytest.mark.parametrize(
    ("text", "expected_start"),
    [
        (
            "But then things really started to get interesting when the first cells "
            "evolved and acquired membranes that separated their chemistry from the environment.",
            "the first cells evolved",
        ),
        (
            "So genomes differ greatly in size because organisms carry different amounts "
            "of repetitive and protein-coding DNA.",
            "genomes differ greatly in size",
        ),
    ],
)
def test_opening_discourse_marker_is_trimmed_only_to_a_standalone_teaching_claim(
    text: str,
    expected_start: str,
) -> None:
    words = text.split()
    proposal = _proposal().model_copy(update={
        "candidate_id": "standalone-after-marker",
        "start_quote": " ".join(words[:6]),
        "end_quote": " ".join(words[-6:]),
        "title": "Standalone biological explanation",
        "learning_objective": "Explain the biological mechanism in this teaching claim",
        "facet": "biological mechanism",
        "reason": "The retained sentence directly teaches a complete biological idea.",
        "topic_evidence_quote": " ".join(words[-12:]),
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        [{"cue_id": "teaching", "start": 0.0, "end": 12.0, "text": text}],
        [],
        {"_segment_ignore_caption_case": True},
        topic="biology",
    )

    assert report.rejected_reasons == []
    assert report.clips[0]["_clip_text"].startswith(expected_start)


def test_leading_so_is_retained_when_removing_it_would_create_an_anaphoric_opening() -> None:
    text = (
        "So this means the mutation changes the protein's active site and prevents "
        "the substrate from binding."
    )
    proposal = _proposal().model_copy(update={
        "candidate_id": "so-with-required-antecedent",
        "start_quote": "So this means the mutation changes",
        "end_quote": "prevents the substrate from binding",
        "title": "How the mutation changes binding",
        "learning_objective": "Explain how an active-site mutation prevents substrate binding",
        "facet": "active-site mutation",
        "reason": "The complete sentence teaches the requested causal relationship.",
        "topic_evidence_quote": (
            "the mutation changes the protein's active site and prevents the substrate"
        ),
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        [{"cue_id": "teaching", "start": 0.0, "end": 10.0, "text": text}],
        [],
        {"_segment_ignore_caption_case": True},
        topic="active-site mutation",
    )

    assert report.rejected_reasons == []
    assert report.clips[0]["_clip_text"].startswith("So this means")


def test_leading_so_is_trimmed_at_a_model_selected_mid_cue_boundary() -> None:
    text = (
        "The molecular-clock example ends here. So genomes differ greatly in size because "
        "organisms carry different amounts of repetitive and protein-coding DNA."
    )
    proposal = _proposal().model_copy(update={
        "candidate_id": "genome-size-mid-cue",
        "start_quote": "So genomes differ greatly in size",
        "end_quote": "repetitive and protein-coding DNA",
        "title": "Why genome sizes differ",
        "learning_objective": "Explain why genome sizes differ among organisms",
        "facet": "genome size",
        "reason": "The selected second sentence is a standalone teaching unit.",
        "topic_evidence_quote": (
            "genomes differ greatly in size because organisms carry different amounts"
        ),
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        [{"cue_id": "mixed", "start": 0.0, "end": 12.0, "text": text}],
        [],
        {"_segment_ignore_caption_case": True},
        topic="genome size",
    )

    assert report.rejected_reasons == []
    assert report.clips[0]["_clip_text"].startswith("genomes differ greatly in size")
    assert "molecular-clock" not in report.clips[0]["_clip_text"]


def test_grounded_explanation_does_not_expand_into_a_visual_preview_sentence() -> None:
    text = (
        "Before I move forward, I just want to quickly show you this map. I mentioned "
        "tracing evolution through a molecular clock, which estimates divergence from "
        "approximately stable mutation rates."
    )
    proposal = _proposal().model_copy(update={
        "candidate_id": "molecular-clock-after-preview",
        "start_quote": "I mentioned tracing evolution through a molecular clock",
        "end_quote": "approximately stable mutation rates",
        "title": "How a molecular clock dates divergence",
        "learning_objective": "Explain how mutation rates support molecular-clock estimates",
        "facet": "molecular clocks",
        "reason": "The selected explanation is complete without the map preview.",
        "topic_evidence_quote": (
            "molecular clock which estimates divergence from approximately stable mutation rates"
        ),
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        [{"cue_id": "clock", "start": 0.0, "end": 14.0, "text": text}],
        [],
        {"_segment_ignore_caption_case": True},
        topic="molecular clocks",
    )

    assert report.rejected_reasons == []
    assert report.clips[0]["_clip_text"].startswith("I mentioned tracing evolution")
    assert "show you this map" not in report.clips[0]["_clip_text"]


def test_topic_announcement_prefix_is_trimmed_to_the_informational_claim() -> None:
    text = (
        "So what we'll talk to you about is the discovery of fluorescent proteins, "
        "which enables researchers to label and track proteins in living cells."
    )
    proposal = _proposal().model_copy(update={
        "candidate_id": "fluorescent-protein-discovery",
        "start_quote": "So what we'll talk to you about",
        "end_quote": "track proteins in living cells",
        "title": "How fluorescent proteins support imaging",
        "learning_objective": "Explain how fluorescent proteins enable live-cell tracking",
        "facet": "fluorescent proteins",
        "reason": "The retained claim explains the educational mechanism.",
        "topic_evidence_quote": (
            "fluorescent proteins which enables researchers to label and track proteins"
        ),
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        [{"cue_id": "fluorescence", "start": 0.0, "end": 12.0, "text": text}],
        [],
        {"_segment_ignore_caption_case": True},
        topic="fluorescent proteins",
    )

    assert report.rejected_reasons == []
    assert report.clips[0]["_clip_text"].startswith(
        "the discovery of fluorescent proteins"
    )
    assert "talk to you about" not in report.clips[0]["_clip_text"]


def test_informational_prefix_is_kept_while_a_visual_demonstration_tail_is_trimmed() -> None:
    segments = [
        {
            "cue_id": "fluorescence-explanation",
            "start": 0.0,
            "end": 14.0,
            "text": (
                "So what we'll talk to you about is fluorescent proteins, which let "
                "researchers label and track proteins in living cells. Protein engineers "
                "created colors that fluoresce at different wavelengths in real time. "
                "These slides show a dividing cell."
            ),
        },
        {
            "cue_id": "visual-demo",
            "start": 14.0,
            "end": 23.0,
            "text": "In these pictures the chromosomes are red and the microtubules are green.",
        },
    ]
    proposal = _proposal(end_line=1).model_copy(update={
        "candidate_id": "fluorescence-before-demo",
        "start_quote": "So what we'll talk to you about is fluorescent proteins",
        "end_quote": "chromosomes are red and the microtubules are green",
        "title": "How fluorescent proteins support live-cell imaging",
        "learning_objective": "Explain how fluorescent proteins label living-cell structures",
        "facet": "fluorescent protein imaging",
        "reason": "The spoken mechanism is complete before the visual demonstration.",
        "topic_evidence_quote": (
            "fluorescent proteins which let researchers label and track proteins"
        ),
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic="fluorescent proteins",
    )

    assert report.rejected_reasons == []
    assert report.clips[0]["cue_ids"] == ["fluorescence-explanation"]
    assert report.clips[0]["_clip_text"].endswith("in real time.")
    assert "slides" not in report.clips[0]["_clip_text"]
    assert "pictures" not in report.clips[0]["_clip_text"]
    assert "trimmed_visual_dependent_tail" in report.clips[0][
        "_boundary_fallback_reasons"
    ]


def test_grounded_sentence_after_an_excluded_mid_cue_marker_does_not_expand_backward() -> None:
    text = (
        "We will cover this next class, because the thing that's critical to building a "
        "cell is a boundary around it. So very early in life lipid "
        "bilayers evolved to separate cellular chemistry from the environment."
    )
    proposal = _proposal().model_copy(update={
        "candidate_id": "lipid-bilayer-after-marker",
        "start_quote": "the thing that's critical to building",
        "end_quote": "cellular chemistry from the environment",
        "title": "Why lipid bilayers evolved",
        "learning_objective": "Explain how lipid bilayers compartmentalize cellular chemistry",
        "facet": "lipid bilayer compartmentalization",
        "reason": "The selected sentence is a complete biological explanation.",
        "topic_evidence_quote": (
            "lipid bilayers evolved to separate cellular chemistry from the environment"
        ),
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        [{"cue_id": "mixed", "start": 0.0, "end": 13.0, "text": text}],
        [],
        {"_segment_ignore_caption_case": True},
        topic="lipid bilayers",
    )

    assert report.rejected_reasons == []
    assert report.clips[0]["_clip_text"].startswith("very early in life lipid bilayers")
    assert "remaining logistics" not in report.clips[0]["_clip_text"]


def test_fragmentary_setup_recovers_forward_when_the_anchor_continues_in_the_next_cue() -> None:
    segments = [
        {
            "cue_id": "membrane-origin",
            "start": 0.0,
            "end": 12.0,
            "text": (
                "We will cover this next class, because the thing that's critical to build "
                "a cell is a wall around it. So very early in life lipid bilayers evolved "
                "to make compartmentalized structures."
            ),
        },
        {
            "cue_id": "membrane-function",
            "start": 12.0,
            "end": 22.0,
            "text": (
                "Cellular compartmentalization through lipid bilayers regulates what can "
                "move into or out of the cell."
            ),
        },
    ]
    proposal = _proposal(end_line=1).model_copy(update={
        "candidate_id": "compartmentalization-across-cues",
        "start_quote": "the thing that's critical to build a cell",
        "end_quote": "move into or out of the cell",
        "title": "How membranes compartmentalize cells",
        "learning_objective": "Explain how lipid bilayers create cellular compartmentalization",
        "facet": "membrane compartmentalization",
        "reason": "The two cues explain membrane origin and function.",
        "topic_evidence_quote": (
            "Cellular compartmentalization through lipid bilayers regulates what can move"
        ),
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic="cellular compartmentalization",
    )

    assert report.rejected_reasons == []
    assert report.clips[0]["cue_ids"] == ["membrane-origin", "membrane-function"]
    assert report.clips[0]["_clip_text"].startswith("very early in life lipid bilayers")
    assert "next class" not in report.clips[0]["_clip_text"]


def test_self_contained_adversative_opening_does_not_import_the_previous_topic() -> None:
    segments = [
        {
            "cue_id": "molecular-clock",
            "start": 0.0,
            "end": 9.0,
            "text": "Mutation rates let a molecular clock estimate evolutionary divergence.",
        },
        {
            "cue_id": "dna-structure",
            "start": 9.0,
            "end": 21.0,
            "text": (
                "But what's fascinating is that all organisms use the same DNA building "
                "blocks. And what we can teach from the 1950s is how its structure works."
            ),
        },
        {
            "cue_id": "dna-replication",
            "start": 21.0,
            "end": 31.0,
            "text": "The double-stranded structure explains how DNA can be copied.",
        },
    ]
    proposal = _proposal(end_line=2).model_copy(update={
        "candidate_id": "dna-structure-only",
        "start_line": 1,
        "start_quote": "we can teach from the 1950s",
        "end_quote": "explains how DNA can be copied",
        "title": "How DNA structure enables replication",
        "learning_objective": "Explain how double-stranded DNA structure enables copying",
        "facet": "DNA structure and replication",
        "reason": "The second cue is a complete, distinct DNA-structure unit.",
        "topic_evidence_quote": (
            "The double-stranded structure explains how DNA can be copied"
        ),
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic="DNA structure",
    )

    assert report.rejected_reasons == []
    assert report.clips[0]["cue_ids"] == ["dna-structure", "dna-replication"]
    assert report.clips[0]["_clip_text"].startswith("what's fascinating")
    assert "molecular clock" not in report.clips[0]["_clip_text"]


def test_complete_ordinal_subject_and_prior_conclusion_bound_one_teaching_unit() -> None:
    segments = [
        {
            "cue_id": "membranes",
            "start": 0.0,
            "end": 9.0,
            "text": "Lipid bilayers compartmentalize the chemistry inside a cell.",
        },
        {
            "cue_id": "cell-types",
            "start": 9.0,
            "end": 22.0,
            "text": (
                "The first prokaryotes were cyanobacteria. Eukaryotic cells are much larger, "
                "contain a nucleus, and can differentiate into muscle, skin, or bone. And so "
                "those eukaryotes mark a long gap of time,"
            ),
        },
        {
            "cue_id": "multicellular-life",
            "start": 22.0,
            "end": 31.0,
            "text": "but later multicellular life evolved and diversified.",
        },
    ]
    proposal = _proposal(end_line=1).model_copy(update={
        "candidate_id": "prokaryotes-versus-eukaryotes",
        "start_line": 1,
        "start_quote": "The first prokaryotes were cyanobacteria",
        "end_quote": "a long gap of time",
        "title": "Prokaryotes versus eukaryotes",
        "learning_objective": "Compare prokaryotic and eukaryotic cell structure",
        "facet": "cell-type comparison",
        "reason": "The comparison is complete before the evolutionary transition.",
        "topic_evidence_quote": (
            "Eukaryotic cells are much larger contain a nucleus and can differentiate"
        ),
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic="prokaryotes versus eukaryotes",
    )

    assert report.rejected_reasons == []
    assert report.clips[0]["cue_ids"] == ["cell-types"]
    assert report.clips[0]["_clip_text"].startswith("The first prokaryotes")
    assert report.clips[0]["_clip_text"].endswith("muscle, skin, or bone.")
    assert "long gap" not in report.clips[0]["_clip_text"]
    assert "multicellular" not in report.clips[0]["_clip_text"]


def test_complete_selected_explanation_is_not_rejected_by_a_later_same_cue_question() -> None:
    text = (
        "Each human cell has 1.8 meters of DNA in it, yet it fits inside a microscopic "
        "cell. DNA gets bundled around positively charged proteins to enable packaging. "
        "When is DNA unraveled?"
    )
    proposal = _proposal().model_copy(update={
        "candidate_id": "dna-packaging-before-next-question",
        "start_quote": "Each human cell has 1.8 meters",
        "end_quote": "positively charged proteins to enable packaging",
        "title": "How DNA fits inside a cell",
        "learning_objective": "Explain how protein binding packages DNA inside cells",
        "facet": "DNA packaging",
        "reason": "The selected span contains the complete packaging explanation.",
        "topic_evidence_quote": (
            "DNA gets bundled around positively charged proteins to enable packaging"
        ),
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        [{"cue_id": "dna", "start": 0.0, "end": 16.0, "text": text}],
        [],
        {"_segment_ignore_caption_case": True},
        topic="DNA packaging",
    )

    assert report.rejected_reasons == []
    assert report.clips[0]["_clip_text"].endswith("enable packaging")
    assert "When is DNA unraveled" not in report.clips[0]["_clip_text"]


@pytest.mark.parametrize(
    "navigation",
    [
        "Now we need to discuss the second step of this same derivation.",
        "Now let's turn to the denominator in the same calculation.",
        "Let's back up and state the theorem used by this proof.",
        "The next part substitutes the known coefficients.",
    ],
)
def test_navigation_inside_one_worked_arc_does_not_delete_required_setup(
    navigation: str,
) -> None:
    segments = [
        {
            "cue_id": "setup",
            "start": 0.0,
            "end": 8.0,
            "text": (
                "The quadratic formula begins with negative b plus or minus the square "
                "root of b squared minus four a c over two a."
            ),
        },
        {"cue_id": "navigation", "start": 8.0, "end": 13.0, "text": navigation},
        {
            "cue_id": "answer",
            "start": 13.0,
            "end": 22.0,
            "text": (
                "Substituting the coefficients gives x equals two or x equals negative three, "
                "which completes the worked example."
            ),
        },
    ]
    proposal = _proposal(end_line=2).model_copy(update={
        "candidate_id": "quadratic-worked-example",
        "start_quote": "The quadratic formula begins with negative b",
        "end_quote": "which completes the worked example",
        "title": "Complete quadratic-formula example",
        "learning_objective": "Solve a quadratic equation through both final roots",
        "facet": "worked example",
        "reason": "The formula setup is required for the substitution and answer.",
        "topic_evidence_quote": (
            "Substituting the coefficients gives x equals two or x equals negative three"
        ),
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic="quadratic formula worked example",
    )

    assert report.rejected_reasons == []
    assert report.clips[0]["cue_ids"] == ["setup", "navigation", "answer"]
    assert report.clips[0]["_clip_text"].startswith("The quadratic formula begins")


@pytest.mark.parametrize(
    ("text", "topic", "objective", "evidence", "expected_end"),
    [
        (
            "The chain rule differentiates the outer function first. Let's look at "
            "the second step: multiply by the inner derivative, giving six x squared "
            "as the final answer.",
            "chain rule worked example",
            "Apply both chain rule steps through the final answer",
            "The chain rule differentiates the outer function first",
            "six x squared as the final answer",
        ),
        (
            "Quantum entanglement creates correlated measurement outcomes. This is one "
            "of the problems with faster-than-light communication: each local outcome "
            "is random, so no controllable message is sent.",
            "entanglement FTL misconception",
            "Explain why entanglement cannot send a controllable message",
            "Quantum entanglement creates correlated measurement outcomes",
            "no controllable message is sent",
        ),
    ],
)
def test_internal_teaching_is_not_treated_as_terminal_noise(
    text: str,
    topic: str,
    objective: str,
    evidence: str,
    expected_end: str,
) -> None:
    proposal = _proposal().model_copy(update={
        "candidate_id": "complete-arc",
        "start_quote": " ".join(text.split()[:4]),
        "end_quote": expected_end,
        "title": "Complete teaching arc",
        "learning_objective": objective,
        "facet": "worked explanation",
        "topic_evidence_quote": evidence,
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        [{"cue_id": "arc", "start": 0.0, "end": 25.0, "text": text}],
        [],
        {"_segment_ignore_caption_case": True},
        topic=topic,
    )

    assert report.rejected_reasons == []
    assert report.clips[0]["_clip_text"].rstrip(" .!?").endswith(expected_end)


def test_next_same_unit_step_is_not_proof_current_setup_is_complete() -> None:
    segments = [
        {
            "cue_id": "setup",
            "start": 0.0,
            "end": 6.0,
            "text": "We differentiate the outer function first",
        },
        {
            "cue_id": "answer",
            "start": 6.0,
            "end": 14.0,
            "text": (
                "Now let's look at the second step: multiply by the inner derivative "
                "to get six x squared."
            ),
        },
    ]
    proposal = _proposal().model_copy(update={
        "candidate_id": "chain-rule-answer",
        "start_quote": "We differentiate the outer function",
        "end_quote": "differentiate the outer function first",
        "title": "Chain rule example",
        "learning_objective": "Apply both chain rule steps through the answer",
        "facet": "worked example",
        "topic_evidence_quote": "We differentiate the outer function first",
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic="chain rule worked example",
    )

    assert report.rejected_reasons == []
    assert report.clips[0]["cue_ids"] == ["setup", "answer"]
    assert report.clips[0]["_clip_text"].endswith("six x squared.")


def test_back_to_navigation_does_not_delete_worked_example_setup() -> None:
    text = (
        "Take sine of x squared. Back to the calculation! Multiplying the outer "
        "and inner derivatives gives two x cosine of x squared as the final result."
    )
    proposal = _proposal().model_copy(update={
        "candidate_id": "chain-rule-worked",
        "start_quote": "Take sine of x squared",
        "end_quote": "x squared as the final result",
        "title": "Chain rule",
        "learning_objective": "Apply the chain rule through the final derivative",
        "facet": "worked example",
        "topic_evidence_quote": (
            "Multiplying the outer and inner derivatives gives two x cosine"
        ),
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        [{"cue_id": "worked", "start": 0.0, "end": 20.0, "text": text}],
        [],
        {"_segment_ignore_caption_case": True},
        topic="chain rule worked example",
    )

    assert report.rejected_reasons == []
    assert report.clips[0]["_clip_text"].startswith("Take sine of x squared")


@pytest.mark.parametrize(
    ("text", "objective", "evidence", "expected_end"),
    [
        (
            "For sine of x squared, we use the chain rule. Now let us discuss how "
            "the rule applies. Differentiate sine, then multiply by two x to get "
            "the answer.",
            "Solve sine of x squared with the chain rule",
            "sine of x squared we use the chain rule",
            "two x to get the answer.",
        ),
        (
            "The second derivative value is negative here. Now let us discuss why it "
            "is negative. Differentiating twice gives a negative value, so the graph "
            "is concave down.",
            "Explain why the second derivative is negative and implies concavity",
            "The second derivative value is negative here",
            "graph is concave down.",
        ),
    ],
)
def test_how_or_why_navigation_keeps_required_reasoning_and_answer(
    text: str,
    objective: str,
    evidence: str,
    expected_end: str,
) -> None:
    proposal = _proposal().model_copy(update={
        "candidate_id": "complete-reasoning",
        "start_quote": " ".join(text.split()[:5]),
        "end_quote": " ".join(text.split()[-5:]),
        "title": "Complete reasoning",
        "learning_objective": objective,
        "facet": "worked explanation",
        "topic_evidence_quote": evidence,
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        [{"cue_id": "reasoning", "start": 0.0, "end": 24.0, "text": text}],
        [],
        {"_segment_ignore_caption_case": True},
        topic=(
            "chain rule worked example"
            if "sine of x squared" in text
            else "second derivative"
        ),
    )

    assert report.rejected_reasons == []
    assert report.clips[0]["_clip_text"].endswith(expected_end)


def test_distinct_topic_before_navigation_is_not_imported_as_list_completion() -> None:
    segments = [
        {
            "cue_id": "derivative",
            "start": 0.0,
            "end": 6.0,
            "text": "A derivative measures instantaneous change",
        },
        {
            "cue_id": "integral-next",
            "start": 6.0,
            "end": 14.0,
            "text": (
                "An integral accumulates area under a curve so let us move on to sequences."
            ),
        },
    ]
    proposal = _proposal().model_copy(update={
        "candidate_id": "derivative",
        "start_quote": "A derivative measures instantaneous",
        "end_quote": "derivative measures instantaneous change",
        "title": "Derivative",
        "learning_objective": "Define a derivative as instantaneous change",
        "facet": "derivative",
        "topic_evidence_quote": "A derivative measures instantaneous change",
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic="derivative",
    )

    assert report.rejected_reasons == []
    assert report.clips[0]["cue_ids"] == ["derivative"]


def test_difference_keyword_does_not_disable_a_real_topic_reset() -> None:
    text = (
        "The limit equals two, which completes the limits problem. "
        "Now let's move on to derivatives. The derivative difference quotient "
        "measures instantaneous change."
    )
    proposal = _proposal().model_copy(update={
        "candidate_id": "difference-quotient-after-limits",
        "start_quote": "The limit equals two which completes",
        "end_quote": "difference quotient measures instantaneous change",
        "title": "Derivative difference quotient",
        "learning_objective": "Explain the derivative difference quotient",
        "facet": "derivatives",
        "reason": "The retained unit explains the derivative definition.",
        "topic_evidence_quote": (
            "The derivative difference quotient measures instantaneous change"
        ),
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        [{"cue_id": "mixed", "start": 0.0, "end": 18.0, "text": text}],
        [],
        {"_segment_ignore_caption_case": True},
        topic="derivative difference quotient",
    )

    assert report.rejected_reasons == []
    assert report.clips[0]["_clip_text"].startswith("The derivative difference quotient")
    assert "limit equals two" not in report.clips[0]["_clip_text"]


def test_true_transition_keeps_both_distinct_candidates_from_one_source() -> None:
    segments = [
        {
            "cue_id": "limits",
            "start": 0.0,
            "end": 8.0,
            "text": "The limit equals two, which completes the limits problem.",
        },
        {
            "cue_id": "transition",
            "start": 8.0,
            "end": 17.0,
            "text": (
                "Now let's move on to derivatives. A derivative measures the instantaneous "
                "rate of change of a function."
            ),
        },
    ]
    limits = _proposal().model_copy(update={
        "candidate_id": "limits-answer",
        "start_quote": "The limit equals two which completes",
        "end_quote": "which completes the limits problem",
        "title": "Completed limits problem",
        "learning_objective": "Understand the completed limit result",
        "facet": "limits",
        "reason": "The first unit completes the limits result.",
        "topic_evidence_quote": "The limit equals two which completes the limits problem",
    })
    derivative = _proposal(end_line=1).model_copy(update={
        "candidate_id": "derivative-definition",
        "start_quote": "The limit equals two which completes",
        "end_quote": "rate of change of a function",
        "title": "Derivative as instantaneous change",
        "learning_objective": "Define a derivative as an instantaneous rate of change",
        "facet": "derivatives",
        "reason": "The second unit defines derivatives.",
        "topic_evidence_quote": (
            "A derivative measures the instantaneous rate of change of a function"
        ),
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[limits, derivative]),
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic="calculus",
    )

    assert report.rejected_reasons == []
    by_id = {clip["selection_candidate_id"]: clip for clip in report.clips}
    assert set(by_id) == {"limits-answer", "derivative-definition"}
    assert by_id["limits-answer"]["cue_ids"] == ["limits"]
    assert by_id["derivative-definition"]["_clip_text"].startswith(
        "A derivative measures"
    )


def test_chain_rule_query_keeps_related_prerequisite_and_worked_paraphrase() -> None:
    segments = [
        {
            "start": 0.0,
            "end": 15.0,
            "text": (
                "A composite function uses h of x equals f of g of x. The inner "
                "function g is evaluated first, and its output becomes the input to f."
            ),
        },
        {
            "start": 20.0,
            "end": 45.0,
            "text": (
                "Differentiate the sine of x squared. First differentiate the outer "
                "sine to get cosine of x squared. Then multiply by the derivative of "
                "the inner x squared, which is two x. So the final derivative is two "
                "x cosine of x squared."
            ),
        },
    ]
    notation = _proposal().model_copy(update={
        "candidate_id": "composition-notation",
        "start_line": 0,
        "end_line": 0,
        "start_quote": "A composite function uses",
        "end_quote": "becomes the input to f",
        "title": "Chain-rule worked example",
        "learning_objective": "Apply the chain rule to a composite function",
        "facet": "worked example",
        "reason": "The notation prepares a chain-rule example.",
        "topic_evidence_quote": "The inner function g is evaluated first and its output",
    })
    worked = notation.model_copy(update={
        "candidate_id": "worked-chain-rule",
        "start_line": 1,
        "end_line": 1,
        "start_quote": "Differentiate the sine of x squared",
        "end_quote": "x cosine of x squared",
        "title": "Chain-rule inner and outer derivatives",
        "learning_objective": (
            "Apply the chain rule by multiplying the outer and inner derivatives"
        ),
        "reason": "The worked steps multiply the outer and inner derivatives.",
        "topic_evidence_quote": (
            "Then multiply by the derivative of the inner x squared which is two x"
        ),
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[notation, worked]),
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic="chain-rule worked example",
    )

    assert [
        clip["selection_candidate_id"] for clip in report.clips
    ] == ["composition-notation", "worked-chain-rule"]
    assert report.rejected_reasons == []


def test_same_call_intent_contract_ranks_complete_task_before_stronger_supporting_facet() -> None:
    topic = "chain rule worked example"
    constraints = [
        {
            "constraint_id": "subject",
            "kind": "subject",
            "source_phrase": "chain rule",
            "requirement": "Teach the chain rule",
        },
        {
            "constraint_id": "task",
            "kind": "format",
            "source_phrase": "worked example",
            "requirement": "Work through a concrete example to its answer",
        },
    ]
    segments = [
        {
            "start": 0.0,
            "end": 12.0,
            "text": (
                "The chain rule differentiates a composite function by multiplying "
                "the outer derivative by the inner derivative."
            ),
        },
        {
            "start": 20.0,
            "end": 42.0,
            "text": (
                "Differentiate sine of x squared. The outer derivative is cosine of "
                "x squared, and the inner derivative is two x. Multiplying them gives "
                "the final answer two x cosine of x squared."
            ),
        },
    ]
    supporting = _proposal().model_copy(update={
        "candidate_id": "definition",
        "start_line": 0,
        "end_line": 0,
        "start_quote": "The chain rule differentiates",
        "end_quote": "by the inner derivative",
        "title": "Chain rule definition",
        "learning_objective": "Define the chain rule",
        "facet": "definition",
        "reason": "This is useful supporting background.",
        "topic_evidence_quote": (
            "The chain rule differentiates a composite function by multiplying"
        ),
        "informativeness": 0.99,
        "topic_relevance": 0.99,
        "educational_importance": 0.99,
        "difficulty": 0.2,
        "intent_role": "supporting",
        "intent_evidence": [{
            "constraint_id": "subject",
            "evidence_quote": (
                "The chain rule differentiates a composite function by multiplying"
            ),
        }],
    })
    worked = supporting.model_copy(update={
        "candidate_id": "worked-example",
        "start_line": 1,
        "end_line": 1,
        "start_quote": "Differentiate sine of x squared",
        "end_quote": "x cosine of x squared",
        "title": "Chain rule worked example",
        "learning_objective": "Apply the chain rule through the final derivative",
        "facet": "worked example",
        "reason": "The example includes setup, steps, and answer.",
        "topic_evidence_quote": (
            "The outer derivative is cosine of x squared and the inner derivative"
        ),
        "informativeness": 0.80,
        "topic_relevance": 0.80,
        "educational_importance": 0.80,
        "intent_role": "primary",
        "intent_evidence": [
            {
                "constraint_id": "subject",
                "evidence_quote": (
                    "The outer derivative is cosine of x squared and the inner derivative"
                ),
            },
            {
                "constraint_id": "task",
                "evidence_quote": (
                    "Multiplying them gives the final answer two x cosine"
                ),
            },
        ],
    })

    report = gemini_segment._plan_to_report(
        _intent_plan(
            topic=topic,
            constraints=constraints,
            topics=[supporting, worked],
        ),
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic=topic,
    )

    assert [clip["selection_candidate_id"] for clip in report.clips] == [
        "worked-example",
        "definition",
    ]
    assert [clip["intent_role"] for clip in report.clips] == [
        "primary",
        "supporting",
    ]
    assert report.clips[0]["intent_coverage"] == 1.0
    assert report.clips[1]["intent_coverage"] == 0.5
    assert report.rejected_reasons == []


def test_difficulty_stage_remains_outer_order_for_primary_and_supporting_intent() -> None:
    topic = "chain rule worked example"
    constraints = [
        {
            "constraint_id": "subject",
            "kind": "subject",
            "source_phrase": "chain rule",
            "requirement": "Teach the chain rule",
        },
        {
            "constraint_id": "task",
            "kind": "format",
            "source_phrase": "worked example",
            "requirement": "Work through a concrete example",
        },
    ]
    segments = [
        {
            "start": 0.0,
            "end": 12.0,
            "text": "The chain rule multiplies the outer derivative by the inner derivative.",
        },
        {
            "start": 20.0,
            "end": 42.0,
            "text": (
                "Differentiate sine of x squared. Multiply cosine of x squared by "
                "two x, producing the final derivative two x cosine of x squared."
            ),
        },
    ]
    beginner_support = _proposal().model_copy(update={
        "candidate_id": "beginner-support",
        "start_quote": "The chain rule multiplies",
        "end_quote": "by the inner derivative",
        "title": "Chain rule foundation",
        "learning_objective": "State the chain rule",
        "facet": "definition",
        "topic_evidence_quote": (
            "The chain rule multiplies the outer derivative by the inner derivative"
        ),
        "intent_role": "supporting",
        "intent_evidence": [{
            "constraint_id": "subject",
            "evidence_quote": (
                "The chain rule multiplies the outer derivative by the inner derivative"
            ),
        }],
        "difficulty": 0.2,
    })
    advanced_primary = beginner_support.model_copy(update={
        "candidate_id": "advanced-primary",
        "start_line": 1,
        "end_line": 1,
        "start_quote": "Differentiate sine of x squared",
        "end_quote": "x cosine of x squared",
        "title": "Advanced chain rule example",
        "learning_objective": "Complete a chain rule calculation",
        "facet": "worked example",
        "topic_evidence_quote": (
            "Multiply cosine of x squared by two x producing the final derivative"
        ),
        "intent_role": "primary",
        "intent_evidence": [
            {
                "constraint_id": "subject",
                "evidence_quote": (
                    "Multiply cosine of x squared by two x producing the final derivative"
                ),
            },
            {
                "constraint_id": "task",
                "evidence_quote": (
                    "producing the final derivative two x cosine of x squared"
                ),
            },
        ],
        "difficulty": 0.8,
    })

    report = gemini_segment._plan_to_report(
        _intent_plan(
            topic=topic,
            constraints=constraints,
            topics=[beginner_support, advanced_primary],
        ),
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic=topic,
    )

    assert [clip["selection_candidate_id"] for clip in report.clips] == [
        "beginner-support",
        "advanced-primary",
    ]


def test_partial_grounded_intent_is_demoted_to_supporting() -> None:
    topic = "chain rule worked example"
    text = (
        "The chain rule differentiates a composite function by multiplying the "
        "outer derivative by the inner derivative."
    )
    proposal = _proposal().model_copy(update={
        "candidate_id": "chain-rule-definition",
        "start_quote": "The chain rule differentiates a composite function",
        "end_quote": "outer derivative by the inner derivative",
        "title": "How the chain rule works",
        "learning_objective": "Explain the chain rule for composite functions",
        "facet": "chain rule definition",
        "reason": "The span directly teaches the chain rule relationship.",
        "topic_evidence_quote": (
            "The chain rule differentiates a composite function by multiplying"
        ),
        "intent_role": "primary",
        "intent_evidence": [{
            "constraint_id": "subject",
            "evidence_quote": (
                "The chain rule differentiates a composite function by multiplying"
            ),
        }],
    })
    report = gemini_segment._plan_to_report(
        _intent_plan(
            topic=topic,
            constraints=[
                {
                    "constraint_id": "subject",
                    "kind": "subject",
                    "source_phrase": "chain rule",
                    "requirement": "Teach the chain rule",
                },
                {
                    "constraint_id": "task",
                    "kind": "format",
                    "source_phrase": "worked example",
                    "requirement": "Work through an example",
                },
            ],
            topics=[proposal],
        ),
        [{"start": 0.0, "end": 12.0, "text": text}],
        [],
        {},
        topic=topic,
    )

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["intent_role"] == "supporting"
    assert clip["intent_coverage"] == pytest.approx(0.5)


def test_duplicate_winner_is_chosen_by_quality_before_difficulty() -> None:
    base = {
        "start": 0.0,
        "end": 12.0,
        "cue_ids": ["cue-0"],
        "learning_objective": "Explain chain rule derivative multiplication",
        "facet": "chain rule derivative",
        "intent_role": "primary",
        "intent_coverage": 1.0,
        "prerequisite_ids": [],
    }
    beginner = {
        **base,
        "selection_candidate_id": "beginner-weaker",
        "informativeness": 0.80,
        "topic_relevance": 0.80,
        "educational_importance": 0.80,
        "difficulty": 0.1,
    }
    advanced = {
        **base,
        "selection_candidate_id": "advanced-stronger",
        "informativeness": 0.99,
        "topic_relevance": 0.99,
        "educational_importance": 0.99,
        "difficulty": 0.9,
    }

    clips = gemini_segment._finalize_clips([beginner, advanced], {})

    assert [clip["selection_candidate_id"] for clip in clips] == [
        "advanced-stronger"
    ]


@pytest.mark.parametrize(
    ("topic", "text", "title"),
    [
        (
            "causes of the French Revolution",
            "Bread prices rose while regressive taxation burdened commoners, fueling anger across France.",
            "Economic pressure and popular anger",
        ),
        (
            "chain-rule worked example",
            "Differentiate the outer sine, then multiply by two x, the derivative of the inner square.",
            "Outer and inner derivative steps",
        ),
        (
            "entanglement and the FTL misconception",
            "Correlated measurements cannot transmit information faster than light because neither observer controls the outcome.",
            "Why correlations cannot send a signal",
        ),
        (
            "myocardial infarction",
            "A heart attack occurs when a blocked coronary artery deprives heart muscle of oxygen.",
            "How a heart attack damages muscle",
        ),
    ],
)
def test_semantic_paraphrases_do_not_require_query_token_echo(
    topic: str,
    text: str,
    title: str,
) -> None:
    words = text.rstrip(".").split()
    proposal = _proposal().model_copy(update={
        "candidate_id": "semantic-paraphrase",
        "start_quote": " ".join(words[:5]),
        "end_quote": " ".join(words[-5:]),
        "title": title,
        "learning_objective": title,
        "facet": title,
        "reason": "The transcript teaches a semantically related unit.",
        "topic_evidence_quote": " ".join(words[: min(10, len(words))]),
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        [{"start": 0.0, "end": 12.0, "text": text}],
        [],
        {"_segment_ignore_caption_case": True},
        topic=topic,
    )

    assert [clip["selection_candidate_id"] for clip in report.clips] == [
        "semantic-paraphrase"
    ]
    assert report.rejected_reasons == []


def test_qcd_rg_rejects_generic_renormalization_but_keeps_specific_facets() -> None:
    segments = [
        {
            "start": 0.0,
            "end": 12.0,
            "text": (
                "Renormalization can mean replacing a raw measurement with a normalized "
                "score so observations from different surveys can be compared."
            ),
        },
        {
            "start": 20.0,
            "end": 34.0,
            "text": (
                "Quarks exchange gluons through the strong interaction, and the gluons "
                "also carry color charge."
            ),
        },
        {
            "start": 40.0,
            "end": 56.0,
            "text": (
                "The coupling runs as the energy scale changes. Its beta function is "
                "negative, so the interaction becomes weaker at high energy."
            ),
        },
    ]
    generic = _proposal().model_copy(update={
        "candidate_id": "generic-renormalization",
        "start_line": 0,
        "end_line": 0,
        "start_quote": "Renormalization can mean replacing",
        "end_quote": "different surveys can be compared",
        "title": "Renormalization",
        "learning_objective": "Understand renormalization",
        "facet": "renormalization",
        "reason": "The span defines renormalization.",
        "topic_evidence_quote": (
            "replacing a raw measurement with a normalized score so observations"
        ),
        "topic_relevance": 0.40,
    })
    qcd_facet = generic.model_copy(update={
        "candidate_id": "qcd-color-charge",
        "start_line": 1,
        "end_line": 1,
        "start_quote": "Quarks exchange gluons",
        "end_quote": "also carry color charge",
        "title": "Color charge in QCD",
        "learning_objective": "Explain color charge in the strong interaction",
        "facet": "QCD prerequisite",
        "reason": "The span teaches a substantive QCD facet.",
        "topic_evidence_quote": (
            "the strong interaction and the gluons also carry color charge"
        ),
        "topic_relevance": 0.90,
    })
    rg_paraphrase = generic.model_copy(update={
        "candidate_id": "running-coupling",
        "start_line": 2,
        "end_line": 2,
        "start_quote": "The coupling runs",
        "end_quote": "becomes weaker at high energy",
        "title": "Renormalization-group beta function",
        "learning_objective": "Explain scale evolution through the beta function",
        "facet": "renormalization-group flow",
        "reason": "The span explains a renormalization-group mechanism.",
        "topic_evidence_quote": (
            "The coupling runs as the energy scale changes Its beta function is negative"
        ),
        "topic_relevance": 0.90,
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[generic, qcd_facet, rg_paraphrase]),
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic="QCD renormalization group",
    )

    assert [
        clip["selection_candidate_id"] for clip in report.clips
    ] == ["qcd-color-charge", "running-coupling"]
    assert "proposal_0:topic_relevance_below_green" in report.rejected_reasons


def test_exact_topic_gate_generalizes_to_unseen_compound_subjects() -> None:
    segments = [
        {
            "start": 0.0,
            "end": 10.0,
            "text": (
                "Attention is the cognitive process of focusing awareness on selected "
                "stimuli while ignoring distractions."
            ),
        },
        {
            "start": 20.0,
            "end": 32.0,
            "text": (
                "Each token's query vector scores the key vectors, and those scores "
                "weight a sum of the value vectors."
            ),
        },
        {
            "start": 40.0,
            "end": 52.0,
            "text": (
                "Token embeddings encode words as vectors that preserve useful language "
                "relationships."
            ),
        },
    ]
    generic = _proposal().model_copy(update={
        "candidate_id": "cognitive-attention",
        "start_line": 0,
        "end_line": 0,
        "start_quote": "Attention is the cognitive process",
        "end_quote": "while ignoring distractions",
        "title": "Cognitive attention",
        "learning_objective": "Define attention in psychology",
        "facet": "attention",
        "reason": "The span defines a broad use of attention.",
        "topic_evidence_quote": (
            "Attention is the cognitive process of focusing awareness on selected stimuli"
        ),
        "topic_relevance": 0.40,
    })
    mechanism = generic.model_copy(update={
        "candidate_id": "transformer-attention",
        "start_line": 1,
        "end_line": 1,
        "start_quote": "Each token's query vector",
        "end_quote": "sum of the value vectors",
        "title": "Transformer attention from query-key scores",
        "learning_objective": "Explain transformer attention weights",
        "facet": "attention mechanism",
        "reason": "The query-key scores determine the attention weights.",
        "topic_evidence_quote": (
            "Each token's query vector scores the key vectors and those scores"
        ),
        "topic_relevance": 0.90,
    })
    prerequisite = generic.model_copy(update={
        "candidate_id": "nlp-embeddings",
        "start_line": 2,
        "end_line": 2,
        "start_quote": "Token embeddings encode words",
        "end_quote": "useful language relationships",
        "title": "Token embeddings in NLP",
        "learning_objective": "Explain NLP token embeddings",
        "facet": "NLP prerequisite",
        "reason": "Token embeddings are a useful prerequisite facet.",
        "topic_evidence_quote": (
            "Token embeddings encode words as vectors that preserve useful language relationships"
        ),
        "topic_relevance": 0.90,
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[generic, mechanism, prerequisite]),
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic="NLP transformer attention",
    )

    assert [
        clip["selection_candidate_id"] for clip in report.clips
    ] == ["transformer-attention", "nlp-embeddings"]
    assert "proposal_0:topic_relevance_below_green" in report.rejected_reasons


def test_worked_example_query_keeps_a_grounded_prerequisite_and_the_application() -> None:
    segments = [
        {
            "start": 0.0,
            "end": 12.0,
            "text": (
                "Conditional probability notation writes the event after the vertical "
                "bar as the condition that is already known."
            ),
        },
        {
            "start": 20.0,
            "end": 38.0,
            "text": (
                "Suppose the prior odds are one to four and the evidence is three times "
                "as likely under the hypothesis. Multiply the prior by that likelihood "
                "ratio and normalize, so the final posterior probability is three sevenths."
            ),
        },
    ]
    notation = _proposal().model_copy(update={
        "candidate_id": "conditional-notation",
        "start_line": 0,
        "end_line": 0,
        "start_quote": "Conditional probability notation writes",
        "end_quote": "that is already known",
        "title": "Bayes-theorem conditional notation",
        "learning_objective": "Apply Bayes theorem with conditional probability",
        "facet": "worked example",
        "reason": "The notation prepares a Bayes-theorem calculation.",
        "educational_importance": 0.78,
        "topic_evidence_quote": (
            "Conditional probability notation writes the event after the vertical bar"
        ),
    })
    worked = notation.model_copy(update={
        "candidate_id": "bayes-calculation",
        "start_line": 1,
        "end_line": 1,
        "start_quote": "Suppose the prior odds",
        "end_quote": "probability is three sevenths",
        "title": "Bayes-theorem prior and likelihood calculation",
        "learning_objective": "Apply Bayes theorem using prior odds and likelihood",
        "reason": "The calculation combines prior odds and a likelihood ratio.",
        "educational_importance": 0.96,
        "topic_evidence_quote": (
            "Multiply the prior by that likelihood ratio and normalize so the final posterior"
        ),
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[notation, worked]),
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic="Bayes-theorem worked example",
    )

    assert [
        clip["selection_candidate_id"] for clip in report.clips
    ] == ["bayes-calculation", "conditional-notation"]
    assert report.rejected_reasons == []


def test_comparison_query_keeps_each_substantive_side_as_its_own_facet() -> None:
    segments = [
        {
            "start": 0.0,
            "end": 10.0,
            "text": (
                "Opportunity cost is the value of the best alternative you give up "
                "when making a choice."
            ),
        },
        {
            "start": 20.0,
            "end": 30.0,
            "text": (
                "A sunk cost is money already spent that cannot be recovered by a "
                "future decision."
            ),
        },
    ]
    opportunity = _proposal().model_copy(update={
        "candidate_id": "opportunity-cost",
        "start_quote": "Opportunity cost is the value",
        "end_quote": "when making a choice",
        "title": "Opportunity cost",
        "learning_objective": "Define opportunity cost",
        "facet": "opportunity cost",
        "reason": "The span teaches one requested side.",
        "topic_evidence_quote": (
            "Opportunity cost is the value of the best alternative you give up"
        ),
    })
    sunk = opportunity.model_copy(update={
        "candidate_id": "sunk-cost",
        "start_line": 1,
        "end_line": 1,
        "start_quote": "A sunk cost is money",
        "end_quote": "by a future decision",
        "title": "Sunk cost",
        "learning_objective": "Define sunk cost",
        "facet": "sunk cost",
        "topic_evidence_quote": (
            "A sunk cost is money already spent that cannot be recovered"
        ),
    })

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[opportunity, sunk]),
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic="opportunity cost versus sunk cost",
    )

    assert [
        clip["selection_candidate_id"] for clip in report.clips
    ] == ["opportunity-cost", "sunk-cost"]


def test_rephrased_facet_is_deduped_but_distinct_facet_survives() -> None:
    segments = [
        {
            "start": 0.0,
            "end": 10.0,
            "text": "Chlorophyll captures light energy that powers photosynthesis reactions.",
        },
        {
            "start": 20.0,
            "end": 30.0,
            "text": "Light absorbed by chlorophyll supplies energy for photosynthesis reactions.",
        },
        {
            "start": 40.0,
            "end": 50.0,
            "text": "Carbon fixation converts carbon dioxide into sugars used by the cell.",
        },
    ]
    first = _proposal().model_copy(update={
        "candidate_id": "energy-first",
        "start_quote": "Chlorophyll captures light energy",
        "end_quote": "photosynthesis reactions",
        "learning_objective": "Explain how chlorophyll captures light energy",
        "facet": "energy capture",
        "topic_evidence_quote": (
            "Chlorophyll captures light energy that powers photosynthesis reactions"
        ),
        "informativeness": 0.76,
        "topic_relevance": 0.99,
        "educational_importance": 0.76,
    })
    rephrased = first.model_copy(update={
        "candidate_id": "energy-better",
        "start_line": 1,
        "end_line": 1,
        "start_quote": "Light absorbed by chlorophyll",
        "end_quote": "photosynthesis reactions",
        "learning_objective": "How chlorophyll captures light energy",
        "topic_evidence_quote": (
            "Light absorbed by chlorophyll supplies energy for photosynthesis reactions"
        ),
        "informativeness": 0.95,
        "topic_relevance": 0.95,
        "educational_importance": 0.95,
    })
    distinct = first.model_copy(update={
        "candidate_id": "carbon-fixation",
        "start_line": 2,
        "end_line": 2,
        "start_quote": "Carbon fixation converts",
        "end_quote": "used by the cell",
        "learning_objective": "Explain how carbon dioxide becomes sugar",
        "facet": "carbon fixation",
        "topic_evidence_quote": (
            "Carbon fixation converts carbon dioxide into sugars used by the cell"
        ),
        "informativeness": 0.90,
        "topic_relevance": 0.90,
        "educational_importance": 0.90,
    })
    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[first, rephrased, distinct]),
        segments,
        [],
        {},
        topic="photosynthesis",
    )
    assert [
        clip["selection_candidate_id"] for clip in report.clips
    ] == ["energy-better", "carbon-fixation"]
