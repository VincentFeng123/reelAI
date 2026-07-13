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


def test_selector_prompt_prefers_foundations_and_allows_one_listed_component() -> None:
    _system, user = gemini_segment._boundary_prompts(
        "[0] 00:00 Cells use chlorophyll to capture light energy.",
        1,
        "photosynthesis, cellular respiration, and DNA inheritance",
        learner_level="beginner",
    )

    assert "deeply teaches any one requested component" in user
    assert "field-wide foundations" in user
    assert "whole transcript" in (_system + user).lower()
    assert "each at least 0.75" in (_system + user)
    assert "unseen visual" in user
    assert "180-second" not in (_system + user)


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
def test_each_quality_floor_is_independently_hard(field: str) -> None:
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
        "topic_evidence_quote": "A worked example begins with two values",
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
        "informativeness": 0.80,
        "topic_relevance": 0.85,
        "educational_importance": 0.80,
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
