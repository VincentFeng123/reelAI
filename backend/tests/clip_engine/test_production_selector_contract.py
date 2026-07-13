from __future__ import annotations

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
        informativeness=0.2,
        topic_relevance=0.2,
        educational_importance=0.2,
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


def test_complete_clip_may_exceed_preferred_duration_but_not_safety_ceiling() -> None:
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

    too_long = [{**complete[0], "end": 181.0}]
    rejected = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[_proposal()]),
        too_long,
        [],
        {"_segment_target_max_sec": 55},
        topic="photosynthesis",
    )
    assert rejected.clips == []
    assert any("invalid_duration" in reason for reason in rejected.rejected_reasons)


def test_selector_prompt_prefers_foundations_and_allows_one_listed_component() -> None:
    _system, user = gemini_segment._boundary_prompts(
        "[0] 00:00 Cells use chlorophyll to capture light energy.",
        1,
        "photosynthesis, cellular respiration, and DNA inheritance",
        learner_level="beginner",
        target_sec=55,
        target_min_sec=20,
        target_max_sec=55,
    )

    assert "deeply teaches any one requested component" in user
    assert "field-wide foundations" in user
    assert "duration preference" in user
    assert "180-second safety ceiling" in user
