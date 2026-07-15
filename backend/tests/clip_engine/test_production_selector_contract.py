from __future__ import annotations

import pytest

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


def test_compact_selector_aliases_preserve_canonical_fields_and_supporting_rank() -> None:
    compact = gemini_segment._CompactBoundaryTopic(
        candidate_id="supporting-definition",
        start_line=0,
        end_line=0,
        start_quote="A derivative measures instantaneous change",
        end_quote="with respect to its input",
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
        topic_evidence_quote=(
            "A derivative measures instantaneous change in a function with respect"
        ),
        self_contained=True,
        is_standalone=True,
        intent_role="supporting",
    )
    payload = gemini_segment._CompactBoundaryPlan(topics=[compact]).model_dump_json(
        by_alias=True
    )
    assert '"id":"supporting-definition"' in payload
    assert '"role":"supporting"' in payload
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
    assert report.clips[0]["intent_evidence"][0]["constraint_id"] == "exact_request"


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


def test_selector_prompt_is_exhaustive_and_allows_one_listed_component() -> None:
    _system, user = gemini_segment._boundary_prompts(
        "[0] 00:00 Cells use chlorophyll to capture light energy.",
        1,
        "photosynthesis, cellular respiration, and DNA inheritance",
        learner_level="beginner",
    )

    assert "deeply teaches any one requested component" in user
    assert "every distinct educational unit" in user
    assert "whole transcript" in (_system + user).lower()
    assert (
        "informativeness, topic_relevance, and educational_importance\n"
        "  are each at least 0.75"
    ) in (_system + user)
    assert "return units across that entire scale" in user.lower()
    assert "unseen visual" in user
    assert "every qualifying related unit" in (_system + user)
    assert "internal interruption" in (_system + user)
    assert "prioritize them within difficulty stages" in (_system + user)
    assert "title (at most 12 words)" in user
    assert "learning_objective (at most 24 words)" in user
    assert "facet (at most 12 words)" in user
    assert user.index("Transcript (") < user.index("Exact user request:")
    assert "1. Privately interpret the exact request" in user
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


def test_same_cue_preview_inside_teaching_is_tolerated_unchanged() -> None:
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

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert "talk more about that next time" in clip["_clip_text"]


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
    assert clip["_end_line"] == 12
    assert "how to make the perfect soup can" in clip["_clip_text"]
    assert clip["_clip_text"].endswith("related to one another.")


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
    assert report.clips[0]["_clip_text"].startswith("Here is another quick example")


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
