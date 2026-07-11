"""Strict shipping contract for the guarded Gemini educational selector."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.pipeline import gemini_segment as G


def _segs(n: int, seconds: float = 10.0) -> list[dict]:
    return [
        {
            "start": index * seconds,
            "end": (index + 1) * seconds,
            "text": f"line {index} explains lesson {index} completely end {index}",
        }
        for index in range(n)
    ]


def _words(segments: list[dict]) -> list[dict]:
    words: list[dict] = []
    for segment in segments:
        tokens = segment["text"].split()
        width = (segment["end"] - segment["start"] - 0.2) / len(tokens)
        for index, token in enumerate(tokens):
            start = segment["start"] + 0.1 + index * width
            words.append({"word": token, "start": start, "end": start + width})
    return words


def _assessment(line: int = 0) -> dict:
    return {
        "prompt": "Which lesson is explained?",
        "options": [f"Lesson {line}", "A sponsor", "A greeting", "An outro"],
        "correct_index": 0,
        "explanation": f"The clip explains lesson {line}.",
        "evidence_quote": f"explains lesson {line}",
    }


def _topic(start_line: int, end_line: int, **overrides) -> G._Topic:
    data = {
        "title": "Lesson",
        "start_line": start_line,
        "end_line": end_line,
        "start_quote": f"line {start_line}",
        "end_quote": f"end {end_line}",
        "facet": "lesson",
        "reason": "This is a complete lesson.",
        "informativeness": 0.9,
        "topic_relevance": 0.9,
        "difficulty": 0.5,
        "self_contained": True,
        "uncertainty": "low",
        "uncertainty_reasons": [],
        "summary": f"Line {start_line} explains lesson {start_line} completely.",
        "takeaways": [
            f"Line {start_line} explains lesson {start_line}.",
            f"Line {end_line} finishes end {end_line}.",
        ],
        "match_reason": f"Lesson {start_line} is explained directly.",
        "assessment": _assessment(start_line),
    }
    data.update(overrides)
    return G._Topic(**data)


def _run(topics: list[G._Topic], segments: list[dict] | None = None,
         settings: dict | None = None) -> list[dict]:
    segments = segments or _segs(20)
    return G._plan_to_clips(
        G._Plan(topics=topics),
        segments,
        _words(segments),
        {"segment_fine_snap": False, **(settings or {})},
    )


@pytest.mark.parametrize(
    "field",
    [
        "start_line", "end_line", "start_quote", "end_quote", "title", "facet", "reason",
        "informativeness", "topic_relevance", "difficulty", "self_contained", "uncertainty",
        "uncertainty_reasons", "summary", "takeaways", "match_reason", "assessment",
    ],
)
def test_every_single_pass_field_is_required(field):
    data = _topic(0, 1).model_dump()
    data.pop(field)
    with pytest.raises(ValidationError):
        G._Topic.model_validate(data)


@pytest.mark.parametrize("field", ["title", "facet", "reason", "start_quote", "end_quote"])
def test_required_text_fields_reject_blank_values(field):
    data = _topic(0, 1).model_dump()
    data[field] = "   "
    with pytest.raises(ValidationError):
        G._Topic.model_validate(data)


@pytest.mark.parametrize("field", ["start_line", "end_line"])
@pytest.mark.parametrize("value", [True, 1.5, "1"])
def test_line_ids_are_strict_integers(field, value):
    data = _topic(0, 1).model_dump()
    data[field] = value
    with pytest.raises(ValidationError):
        G._Topic.model_validate(data)


@pytest.mark.parametrize("field", ["informativeness", "topic_relevance", "difficulty"])
@pytest.mark.parametrize("value", [-0.001, 1.001, 7, 85])
def test_scores_outside_zero_to_one_are_rejected_not_normalized(field, value):
    data = _topic(0, 1).model_dump()
    data[field] = value
    with pytest.raises(ValidationError):
        G._Topic.model_validate(data)


def test_kind_is_not_model_supplied_and_is_deterministically_educational():
    data = _topic(0, 1).model_dump()
    data["kind"] = "promo"
    with pytest.raises(ValidationError):
        G._Topic.model_validate(data)
    assert _run([_topic(0, 1)])[0]["kind"] == "educational"


@pytest.mark.parametrize(
    "overrides",
    [
        {"informativeness": 0.59},
        {"topic_relevance": 0.59},
        {"self_contained": False},
    ],
)
def test_application_quality_gates_fail_closed(overrides):
    assert _run([_topic(0, 1, **overrides)]) == []


def test_quality_thresholds_are_inclusive_and_carried():
    clip = _run([_topic(0, 1, informativeness=0.6, topic_relevance=0.6)])[0]
    assert clip["informativeness"] == pytest.approx(0.6)
    assert clip["topic_relevance"] == pytest.approx(0.6)
    assert clip["self_contained"] is True


def test_request_quality_floor_overrides_are_preserved():
    proposal = _topic(0, 1, informativeness=0.7, topic_relevance=0.7)
    assert _run([proposal], settings={"segment_informativeness_min": 0.8}) == []
    assert _run([proposal], settings={"segment_topic_relevance_min": 0.8}) == []


def test_short_complete_clip_survives_legacy_fifteen_second_setting():
    segments = _segs(1, seconds=5.0)
    clips = _run([_topic(0, 0)], segments, {"segment_min_clip_s": 15})
    assert [(clip["start"], clip["end"]) for clip in clips] == [(0.0, 5.0)]


@pytest.mark.parametrize("duration", [90.0, 120.0, 180.0])
def test_complete_clips_through_one_eighty_seconds_survive(duration):
    segments = [{"start": 0.0, "end": duration,
                 "text": "line zero explains lesson zero completely end zero"}]
    clip = _run([
        _topic(
            0, 0,
            start_quote="line zero",
            end_quote="end zero",
            summary="Line zero explains lesson zero completely.",
            takeaways=["Line zero explains lesson zero.", "The lesson finishes end zero."],
            match_reason="Lesson zero is explained directly.",
            assessment={**_assessment(0), "evidence_quote": "explains lesson zero"},
        )
    ], segments)[0]
    assert clip["end"] == duration


def test_clip_over_one_eighty_seconds_is_rejected_without_hard_cut():
    segments = [{"start": 0.0, "end": 180.001,
                 "text": "line zero explains lesson zero completely end zero"}]
    proposal = _topic(
        0, 0,
        start_quote="line zero", end_quote="end zero",
        assessment={**_assessment(0), "evidence_quote": "lesson zero"},
    )
    assert _run([proposal], segments, {"segment_max_clip_s": 999}) == []


def test_explicit_max_clips_is_respected_below_forty_ceiling():
    segments = _segs(4)
    clips = _run([_topic(i, i, title=f"T{i}") for i in range(4)], segments, {"max_clips": 2})
    assert len(clips) == 2


@pytest.mark.parametrize(
    "overrides,reason",
    [
        ({"informativeness": 0.7}, "quality_score_below_green"),
        ({"uncertainty": "medium", "uncertainty_reasons": ["boundary_ambiguous"]},
         "medium_uncertainty"),
    ],
)
def test_truncation_cannot_hide_uncertain_validated_candidates(overrides, reason):
    segments = _segs(2)
    plan = G._Plan(topics=[
        _topic(0, 0, title="kept", informativeness=0.95, topic_relevance=0.95),
        _topic(1, 1, title="truncated", **overrides),
    ])
    report = G._plan_to_report(
        plan, segments, _words(segments),
        {"segment_fine_snap": False, "max_clips": 1},
    )
    assert [clip["title"] for clip in report.clips] == ["kept"]
    classification = G._classify_flash(
        report, segments, "", enrichment_required=False,
    )
    assert classification.status == "uncertain"
    assert reason in classification.reasons


def test_truncation_cannot_hide_long_validated_candidate():
    segments = _segs(3, seconds=75.0)
    plan = G._Plan(topics=[
        _topic(0, 0, title="kept", informativeness=0.95, topic_relevance=0.95),
        _topic(1, 2, title="truncated", informativeness=0.8, topic_relevance=0.8),
    ])
    report = G._plan_to_report(
        plan, segments, _words(segments),
        {"segment_fine_snap": False, "max_clips": 1},
    )
    assert [clip["title"] for clip in report.clips] == ["kept"]
    classification = G._classify_flash(
        report, segments, "", enrichment_required=False,
    )
    assert classification.status == "uncertain"
    assert "long_clip" in classification.reasons


def test_schema_enforces_forty_proposal_ceiling():
    with pytest.raises(ValidationError):
        G._Plan(topics=[_topic(0, 0) for _ in range(41)])


def test_learning_details_and_valid_assessment_are_carried_without_evidence_field():
    question = _assessment(0)
    clip = _run([_topic(0, 1, assessment=question)])[0]
    assert clip["summary"].startswith("Line 0 explains")
    assert len(clip["takeaways"]) == 2
    assert clip["match_reason"].startswith("Lesson 0")
    assert clip["assessment"] == {key: value for key, value in question.items()
                                   if key != "evidence_quote"}


@pytest.mark.parametrize(
    "question",
    [
        {**_assessment(), "options": ["a", "a", "b", "c"]},
        {**_assessment(), "options": ["a", "b", "c", "d", "e"]},
        {**_assessment(), "correct_index": 4},
        {**_assessment(), "correct_index": True},
        {**_assessment(), "evidence_quote": "outside the accepted clip"},
        {**_assessment(), "evidence_quote": "line"},
        {**_assessment(), "evidence_quote": "line 0",
         "options": ["Moon cheese", "A sponsor", "A greeting", "An outro"],
         "explanation": "The moon is cheese."},
        {**_assessment(), "options": ["A", "B", "C", "all of the above"]},
    ],
)
def test_bad_assessment_is_discarded_without_inventing_content(question):
    assert G._validated_assessment(
        question, grounding_text="line 0 explains lesson 0 completely end 0",
    ) is None


def test_bad_assessment_does_not_discard_other_grounded_learning_details():
    details, errors = G._learning_details(
        G._LegacyTopic(
            title="Lesson", start_line=0, end_line=0,
            start_quote="line zero", end_quote="end zero",
            summary="Line zero explains lesson zero.",
            takeaways=["Line zero explains the lesson.", "The lesson reaches end zero."],
            match_reason="Lesson zero is explained directly.",
            assessment={"prompt": "bad"},
        ),
        "line zero explains lesson zero completely end zero",
        "",
    )
    assert details["summary"] and len(details["takeaways"]) == 2
    assert details["match_reason"]
    assert details["assessment"] is None
    assert errors == ["assessment_invalid"]


def test_generic_function_words_cannot_make_hallucinated_text_look_grounded():
    assert not G._text_has_grounding(
        "The moon is made of cheese.",
        "The derivative measures instantaneous change.",
    )


def test_prompt_layout_and_contract_follow_gemini3_guidance():
    transcript = "[0] 00:00 hi"
    system, user = G._prompts(transcript, 1, topic="photosynthesis")
    combined = system + "\n" + user
    assert combined.index("KEEP this complete") < combined.index("Transcript")
    assert combined.index("OMIT these non-units") < combined.index("Transcript")
    assert user.index(transcript) < user.index("Based on the preceding transcript")
    assert "kind" not in _selection_task_tail(user)
    for field in (
        "informativeness", "topic_relevance", "self_contained", "difficulty",
        "start_line", "end_line", "start_quote", "end_quote", "uncertainty",
    ):
        assert field in combined
    assert "chain-of-thought" in combined


def _selection_task_tail(user: str) -> str:
    return user[user.index("Based on the preceding transcript"):]


def test_segment_clips_threads_topic_to_profile_runner(monkeypatch):
    seen: list[str] = []

    def fake_run(transcript, settings, profile, **kwargs):
        seen.append(kwargs["topic"])
        return G.SegmentResult([], "none", profile, "invalid")

    monkeypatch.setattr(G, "run_segment_profile", fake_run)
    G.segment_clips({"segments": _segs(2), "words": _words(_segs(2))}, {}, topic="linear algebra")
    assert seen == ["linear algebra"]
