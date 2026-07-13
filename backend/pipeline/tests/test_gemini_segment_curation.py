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
        "candidate_id": f"candidate-{start_line}-{end_line}",
        "title": "Lesson",
        "learning_objective": "Understand the complete lesson.",
        "start_line": start_line,
        "end_line": end_line,
        "start_quote": f"line {start_line}",
        "end_quote": f"end {end_line}",
        "facet": "lesson",
        "reason": "This is a complete lesson.",
        "informativeness": 0.9,
        "topic_relevance": 0.9,
        "educational_importance": 0.9,
        "difficulty": 0.5,
        "directly_teaches_topic": True,
        "substantive": True,
        "factually_grounded": True,
        "topic_evidence_quote": (
            f"line {start_line} explains lesson {start_line} completely"
        ),
        "self_contained": True,
        "is_standalone": True,
        "prerequisite_candidate_ids": [],
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
        "candidate_id", "start_line", "end_line", "start_quote", "end_quote", "title",
        "learning_objective", "facet", "reason", "informativeness", "topic_relevance",
        "educational_importance", "difficulty", "directly_teaches_topic", "substantive",
        "factually_grounded", "topic_evidence_quote", "self_contained", "is_standalone",
        "prerequisite_candidate_ids", "uncertainty", "uncertainty_reasons", "summary",
        "takeaways", "match_reason", "assessment",
    ],
)
def test_every_single_pass_field_is_required(field):
    data = _topic(0, 1).model_dump()
    data.pop(field)
    with pytest.raises(ValidationError):
        G._Topic.model_validate(data)


@pytest.mark.parametrize(
    "field",
    [
        "candidate_id", "title", "learning_objective", "facet", "reason",
        "start_quote", "end_quote", "topic_evidence_quote",
    ],
)
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


@pytest.mark.parametrize(
    "field",
    ["informativeness", "topic_relevance", "educational_importance", "difficulty"],
)
@pytest.mark.parametrize(
    "value",
    [-0.001, 1.001, 7, 85, True, "0.5", float("nan"), float("inf")],
)
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


def test_self_contained_gate_still_fails_closed():
    assert _run([_topic(0, 1, self_contained=False)]) == []


def test_zero_scores_are_valid_and_carried():
    clip = _run([_topic(
        0,
        1,
        informativeness=0.0,
        topic_relevance=0.0,
        educational_importance=0.0,
        difficulty=0.0,
    )])[0]
    assert clip["informativeness"] == pytest.approx(0.0)
    assert clip["topic_relevance"] == pytest.approx(0.0)
    assert clip["educational_importance"] == pytest.approx(0.0)
    assert clip["difficulty"] == pytest.approx(0.0)
    assert clip["self_contained"] is True


def test_request_quality_floor_overrides_do_not_reject_valid_scores():
    proposal = _topic(0, 1, informativeness=0.1, topic_relevance=0.1)
    assert _run([proposal], settings={"segment_informativeness_min": 0.8})
    assert _run([proposal], settings={"segment_topic_relevance_min": 0.8})


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
            topic_evidence_quote="line zero explains lesson zero completely",
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
        topic_evidence_quote="line zero explains lesson zero completely",
        assessment={**_assessment(0), "evidence_quote": "lesson zero"},
    )
    assert _run([proposal], segments, {"segment_max_clip_s": 999}) == []


def test_oversized_section_repairs_to_a_complete_cue_subunit():
    segments = [
        {
            "start": float(index * 5),
            "end": float((index + 1) * 5),
            "text": f"line {index} explains lesson {index} completely end {index}.",
        }
        for index in range(48)
    ]
    proposal = _topic(
        0,
        47,
        title="Long section",
        start_quote="line 0",
        end_quote="end 47",
    )

    clips = _run([proposal], segments)

    assert len(clips) == 1
    assert clips[0]["start"] == 0.0
    assert clips[0]["end"] == 75.0
    assert clips[0]["cue_ids"][-1] == "cue-14"


def test_oversized_range_without_complete_subunit_is_not_hard_cut():
    segments = [
        {
            "start": float(index * 10),
            "end": float((index + 1) * 10),
            "text": f"Photosynthesis step {index} continues and",
        }
        for index in range(20)
    ]

    repaired = G._repair_oversized_cue_range(
        segments,
        0,
        19,
        ignore_caption_case=True,
    )

    assert repaired is None


def test_oversized_repair_prefers_title_grounded_opening_within_setup_window():
    segments = [
        {
            "start": float(index * 5),
            "end": float((index + 1) * 5),
            "text": (
                "Chloroplast structure begins this complete explanation."
                if index == 3
                else f"Background lesson point {index} finishes completely."
            ),
        }
        for index in range(48)
    ]

    repaired = G._repair_oversized_cue_range(
        segments,
        0,
        47,
        ignore_caption_case=True,
        anchor_text="Chloroplast structure and pigments",
    )

    assert repaired == (3, 17)


def test_production_biology_range_repairs_around_channel_bump():
    raw_cues = [
        (354.960, 359.240, "Cool! There's just one issue: Your DNA and its information is in the nucleus,"),
        (359.240, 362.680, "but proteins are made in organelles called the ribosomes. How do we get the"),
        (362.680, 367.320, "information from A to B? The answer is RNA. It's kind of like DNA, just that it's most"),
        (367.320, 372.720, "often a single strand, it uses a ribose instead of deoxyribose and instead of Thymine it uses Uracil,"),
        (372.720, 376.960, "which makes it less stable, but that's besides the point, here's what RNA actually does:"),
        (376.960, 380.360, "Let's say you want to make the protein coded for by this gene. An enzyme called"),
        (380.360, 384.880, "RNA polymerase will split the DNA and make a strand of RNA with the complementary bases,"),
        (384.880, 389.720, "essentially copying the information from the DNA to the RNA. This is called transcription."),
        (389.720, 393.320, "The new strand is called messenger RNA or mRNA, because it carries this"),
        (393.320, 397.400, "message out of the nucleus to a ribosome. Remember how I said that a gene is like a"),
        (397.400, 402.480, "recipe for a protein? Well, on the mRNA, which carries the same base sequence as that gene,"),
        (402.480, 406.920, "every group of three bases, which is called a codon, codes for a specific amino acid,"),
        (406.920, 408.819, "which are the building blocks for proteins. Welcome to Biology Pro Tips Season 1, if you want"),
        (408.819, 408.920, "to decode a sequence of RNA, there is actually a chart for that! Yeah that's all have a great day."),
        (408.920, 413.160, "These amino acids are carried by special molecules called transfer RNA or tRNA,"),
        (413.160, 418.160, "which have a unique anticodon that can only attach to its matching codon on the mRNA."),
        (418.160, 423.360, "The job of the ribosome is to read over codons on the mRNA and attach the matching tRNA molecules,"),
        (423.360, 428.040, "which then leave behind their amino acid. As the ribosome moves along the mRNA and attaches more"),
        (428.040, 433.920, "tRNA, which happens a couple thousand times, the amino acids combine into a polypeptide chain,"),
        (433.920, 437.200, "which is just a really long chain of amino acids, that can be bunched up,"),
        (437.200, 443.160, "creased, smacked and folded into a protein. Okay, let's recap: A gene is copied onto mRNA,"),
        (443.160, 446.680, "which is then used to build proteins by assembling a chain of amino acids."),
    ]
    segments = [
        {"cue_id": f"3tisOnOkwzo:cue:{index + 78}", "start": start, "end": end, "text": text}
        for index, (start, end, text) in enumerate(raw_cues)
    ]
    proposal = G._BoundaryTopic(
        candidate_id="biology-transcription",
        start_line=0,
        end_line=len(segments) - 1,
        start_quote="Cool! There's just one issue",
        end_quote="assembling a chain of amino acids",
        title="RNA polymerase transcribes DNA into messenger RNA",
        learning_objective="Understand how RNA polymerase transcribes DNA into messenger RNA.",
        facet="transcription",
        reason="Explains transcription from DNA to messenger RNA.",
        informativeness=0.9,
        topic_relevance=0.9,
        educational_importance=0.9,
        difficulty=0.3,
        directly_teaches_topic=True,
        substantive=True,
        factually_grounded=True,
        topic_evidence_quote="An enzyme called RNA polymerase will split the DNA and make a strand of RNA",
        self_contained=True,
        is_standalone=True,
        prerequisite_candidate_ids=[],
        uncertainty="low",
        uncertainty_reasons=[],
    )

    report = G._plan_to_report(
        G._BoundaryPlan(topics=[proposal]),
        segments,
        [],
        {
            "_segment_target_sec": 40,
            "_segment_target_min_sec": 10,
            "_segment_target_max_sec": 55,
        },
        topic="biology",
    )

    assert len(report.clips) == 1
    clip = report.clips[0]
    assert clip["end"] - clip["start"] <= 55
    clip_text = G._cue_clip_text(segments, clip["_start_line"], clip["_end_line"])
    assert clip_text.rstrip().endswith((".", "!", "?"))
    assert "Welcome to Biology Pro Tips" not in clip_text
    assert "have a great day" not in clip_text


def test_production_biology_fragment_expands_to_question_and_complete_list():
    raw_cues = [
        (74.120, 77.600, "Ok, so enzymes make life possible by speeding up chemical reactions,"),
        (77.600, 83.760, "but what even is…life? Scientists don't really seem to agree, but obviously a cat is different"),
        (83.760, 88.200, "from a rock. The cat can produce energy by metabolizing food, it can grow and develop,"),
        (88.200, 92.320, "reproduce, and it responds to the environment, whereas the rock does not."),
        (92.320, 95.960, "Also, unlike rocks, every living thing on earth is made of cells, of which there's"),
        (95.960, 101.240, "two main categories: Eukaryotes and prokaryotes. Eukaryotes have fancy organelles which are bound"),
        (101.240, 106.040, "by membranes, like the nucleus, inside of which is DNA. Prokaryotes have none of those organelles,"),
        (106.040, 110.120, "and the DNA is just kind of chilling there, like freely floating around."),
        (110.120, 113.320, "This is why Prokaryotes are just single cell organisms like bacteria"),
        (113.320, 117.760, "and archea whereas eukaryotes can form complex organisms like protists, fungi,"),
        (117.760, 121.920, "plants and animals. These are what's known as kingdoms, which is a taxonomic rank,"),
        (121.920, 126.040, "so basically, how we classify different living things and how they're related to one another."),
    ]
    segments = [
        {"cue_id": f"3tisOnOkwzo:cue:{index + 15}", "start": start, "end": end, "text": text}
        for index, (start, end, text) in enumerate(raw_cues)
    ]
    proposal = G._BoundaryTopic(
        candidate_id="biology-cell-types",
        start_line=3,
        end_line=9,
        start_quote="reproduce, and it responds to the environment",
        end_quote="complex organisms like protists, fungi",
        title="Eukaryotic and prokaryotic cell structures",
        learning_objective="Distinguish eukaryotic and prokaryotic cells.",
        facet="cell structure",
        reason="Explains the two cell categories and their organelles.",
        informativeness=0.9,
        topic_relevance=0.9,
        educational_importance=0.9,
        difficulty=0.3,
        directly_teaches_topic=True,
        substantive=True,
        factually_grounded=True,
        topic_evidence_quote="every living thing on earth is made of cells, of which there's two main categories",
        self_contained=True,
        is_standalone=True,
        prerequisite_candidate_ids=[],
        uncertainty="low",
        uncertainty_reasons=[],
    )

    report = G._plan_to_report(
        G._BoundaryPlan(topics=[proposal]),
        segments,
        [],
        {
            "_segment_target_sec": 40,
            "_segment_target_min_sec": 20,
            "_segment_target_max_sec": 55,
            "_segment_ignore_caption_case": True,
        },
        topic="biology",
    )

    assert len(report.clips) == 1
    clip = report.clips[0]
    assert clip["cue_ids"][0] == "3tisOnOkwzo:cue:16"
    assert clip["cue_ids"][-1] == "3tisOnOkwzo:cue:26"
    assert clip["end"] - clip["start"] <= 55
    clip_text = G._cue_clip_text(segments, clip["_start_line"], clip["_end_line"])
    assert clip_text.startswith("but what even is…life?")
    assert clip_text.endswith("related to one another.")


def test_production_ligature_clip_repairs_elliptical_instruction_to_r_context():
    raw_cues = [
        (2252.24, 2255.96, "We have the first R,"),
        (2254.36, 2258.60, "which very much looks like the R we're"),
        (2255.96, 2258.60, "used to."),
        (2261.32, 2264.56, "And remember to start just a little"),
        (2262.44, 2267.32, "below the line and then pull your pen up"),
        (2264.56, 2267.32, "and pull it through."),
        (2269.08, 2272.52, "Little below the line,"),
        (2270.76, 2274.48, "pull it through."),
        (2272.52, 2275.76, "The second R"),
        (2274.48, 2277.68, "is what you might see when it gets"),
        (2275.76, 2279.96, "written off of a letter. It's kind of a"),
        (2277.68, 2283.72, "ligature R."),
        (2279.96, 2286.96, "So if I get put an O over here,"),
        (2283.72, 2286.96, "then I want to draw an R,"),
        (2288.36, 2291.36, "I can just do that."),
        (2296.96, 2301.72, "And so this is the R and this is"),
        (2298.72, 2303.84, "actually called a half R."),
        (2301.72, 2306.56, "And a lot of different scripts use the"),
        (2303.84, 2308.56, "half R."),
        (2306.56, 2310.68, "Um I have seen this in formal documents."),
        (2308.56, 2313.20, "I've seen it in formal documents. So,"),
        (2310.68, 2315.16, "it's not that this is considered an"),
        (2313.20, 2318.08, "informal way of writing"),
        (2315.16, 2320.48, "um everywhere all the time. It's okay to"),
        (2318.08, 2320.48, "do."),
    ]
    segments = [
        {"cue_id": f"nHMf37SMX-Q:cue:{index + 748}", "start": start, "end": end, "text": text}
        for index, (start, end, text) in enumerate(raw_cues)
    ]
    proposal = G._BoundaryTopic(
        candidate_id="carolingian-half-r",
        start_line=6,
        end_line=23,
        start_quote="Little below the line",
        end_quote="okay to",
        title="Identifying and writing the Carolingian half R ligature",
        learning_objective="Recognize and write the half R ligature in Carolingian minuscule.",
        facet="ligature identification",
        reason="Explains the alternate R form and demonstrates how it joins another letter.",
        informativeness=0.8,
        topic_relevance=0.9,
        educational_importance=0.8,
        difficulty=0.4,
        directly_teaches_topic=True,
        substantive=True,
        factually_grounded=True,
        topic_evidence_quote=(
            "The second R is what you might see when it gets written off of a letter"
        ),
        self_contained=True,
        is_standalone=True,
        prerequisite_candidate_ids=[],
        uncertainty="low",
        uncertainty_reasons=[],
    )

    report = G._plan_to_report(
        G._BoundaryPlan(topics=[proposal]),
        segments,
        [],
        {
            "_segment_target_sec": 55,
            "_segment_target_min_sec": 20,
            "_segment_target_max_sec": 55,
        },
        topic="Carolingian minuscule ligature identification",
    )

    assert len(report.clips) == 1
    clip = report.clips[0]
    assert clip["cue_ids"][0] == "nHMf37SMX-Q:cue:748"
    assert G._cue_clip_text(segments, clip["_start_line"], clip["_end_line"]).startswith(
        "We have the first R"
    )
    assert 55 < clip["end"] - clip["start"] <= 180


def test_complete_clean_range_is_not_shortened_to_duration_preference():
    segments = [
        {
            "cue_id": "life-definition",
            "start": 0.0,
            "end": 30.0,
            "text": "Living systems maintain chemical balance and evolve across generations.",
        },
        {
            "cue_id": "regulation",
            "start": 30.0,
            "end": 60.0,
            "text": "Regulation keeps internal conditions stable and organisms respond to environmental change.",
        },
        {
            "cue_id": "reproduction",
            "start": 60.0,
            "end": 90.0,
            "text": "Reproduction passes genetic information to offspring.",
        },
    ]
    proposal = G._BoundaryTopic(
        candidate_id="life-characteristics",
        start_line=0,
        end_line=2,
        start_quote="Living systems maintain chemical balance",
        end_quote="genetic information to offspring",
        title="Regulation, response, and reproduction define life",
        learning_objective=(
            "Identify regulation, environmental response, and reproduction as characteristics of life."
        ),
        facet="definition of life",
        reason="Explains how scientists distinguish living systems.",
        informativeness=0.8,
        topic_relevance=0.9,
        educational_importance=0.9,
        difficulty=0.3,
        directly_teaches_topic=True,
        substantive=True,
        factually_grounded=True,
        topic_evidence_quote=(
            "Living systems maintain chemical balance and evolve across generations"
        ),
        self_contained=True,
        is_standalone=True,
        prerequisite_candidate_ids=[],
        uncertainty="low",
        uncertainty_reasons=[],
    )

    report = G._plan_to_report(
        G._BoundaryPlan(topics=[proposal]),
        segments,
        [],
        {
            "_segment_target_sec": 30,
            "_segment_target_min_sec": 20,
            "_segment_target_max_sec": 40,
        },
        topic="biology",
    )

    assert len(report.clips) == 1
    clip = report.clips[0]
    assert clip["cue_ids"] == ["life-definition", "regulation", "reproduction"]
    assert clip["learning_objective"] == proposal.learning_objective
    assert clip["title"] == proposal.title


def test_range_repair_keeps_paraphrased_objective_without_lost_support():
    objective = "Explain how living systems preserve balance over time."
    assert G._objective_after_range_repair(
        objective,
        original_text=(
            "Living systems preserve balance over time. Regulation is one example."
        ),
        retained_text="Living systems preserve balance over time.",
        evidence_quote="Living systems preserve balance over time",
    ) == objective


def test_structural_filler_is_rejected_even_when_scores_are_high():
    segments = [{
        "start": 0.0,
        "end": 10.0,
        "text": "Welcome to Biology Pro Tips. Today cells are the basic unit of life.",
    }]
    report = G._plan_to_report(
        G._Plan(topics=[_topic(
            0,
            0,
            start_quote="Welcome to Biology Pro Tips",
            end_quote="basic unit of life",
            topic_evidence_quote="Today cells are the basic unit of life",
        )]),
        segments,
        [],
        {},
        topic="biology",
    )

    assert report.clips == []
    assert report.rejected_reasons == ["proposal_0:contains_filler"]


@pytest.mark.parametrize("topic", ["biology", "biology myths", "Gothic architecture"])
def test_unrequested_fictional_biology_is_rejected_even_with_real_terminology(topic):
    segments = [{
        "cue_id": "oJLA8iNUV-0:722.37-773.27",
        "start": 722.37,
        "end": 773.27,
        "text": (
            "solidified in the Middle Ages of course the greatest threat to any dark entity "
            "is the Cross of Christ but as we've seen the vampire condition is not Supernatural "
            "is there any legitimacy to this bit of lore let me explain with a bit of context "
            "our eyes contain arrays of specialized receptor cells some only activate when "
            "they see light and Shadow in conjunction some activate only when they see "
            "horizontal lines Horizons and so on in Vampires The receptors that respond to "
            "horizontal lines are cross-wired with those that respond to Vertical ones when "
            "both sets of receptors are fired simultaneously in a very specific way that is "
            "when intersecting right angles occupy more than 30 degrees of visual Arc positive "
            "feedback seems to generate a neuroelectrical overload in the visual cortex"
        ),
    }]
    report = G._plan_to_report(
        G._Plan(topics=[_topic(
            0,
            0,
            start_quote="solidified in the Middle Ages",
            end_quote="overload in the visual cortex",
            topic_evidence_quote=(
                "our eyes contain arrays of specialized receptor cells some only activate"
            ),
        )]),
        segments,
        [],
        {},
        topic=topic,
    )

    assert report.clips == []
    assert report.rejected_reasons == ["proposal_0:fictional_framing"]


def test_requested_fictional_topic_can_still_teach_its_named_subject():
    segments = [{
        "start": 0.0,
        "end": 24.0,
        "text": (
            "Vampire folklore changed from contagious corpses into aristocratic figures. "
            "This shift followed nineteenth century Gothic literature and theater."
        ),
    }]
    report = G._plan_to_report(
        G._Plan(topics=[_topic(
            0,
            0,
            start_quote="Vampire folklore changed",
            end_quote="literature and theater",
            topic_evidence_quote=(
                "Vampire folklore changed from contagious corpses into aristocratic figures"
            ),
        )]),
        segments,
        [],
        {},
        topic="history of vampire folklore",
    )

    assert len(report.clips) == 1


def test_real_teaching_that_denies_a_supernatural_claim_is_not_rejected():
    segments = [{
        "start": 0.0,
        "end": 22.0,
        "text": (
            "Quantum entanglement is not supernatural communication. "
            "Measurements are correlated, but they cannot transmit information faster than light."
        ),
    }]
    report = G._plan_to_report(
        G._Plan(topics=[_topic(
            0,
            0,
            start_quote="Quantum entanglement",
            end_quote="faster than light",
            topic_evidence_quote=(
                "Measurements are correlated but they cannot transmit information faster than light"
            ),
        )]),
        segments,
        [],
        {},
        topic="quantum entanglement",
    )

    assert len(report.clips) == 1


def test_selector_rejects_an_explicitly_ungrounded_teaching_claim():
    segments = _segs(1)
    report = G._plan_to_report(
        G._Plan(topics=[_topic(0, 0, factually_grounded=False)]),
        segments,
        _words(segments),
        {},
        topic="lesson 0",
    )

    assert report.clips == []
    assert report.rejected_reasons == ["proposal_0:not_factually_grounded"]


def test_video_plug_tail_is_rejected_even_after_substantive_teaching():
    segments = [{
        "start": 0.0,
        "end": 20.0,
        "text": (
            "Complement proteins mark bacteria and help immune cells destroy them. "
            "We made a whole video explaining them in detail. We are reaching a crossroad now."
        ),
    }]
    report = G._plan_to_report(
        G._Plan(topics=[_topic(
            0,
            0,
            start_quote="Complement proteins mark bacteria",
            end_quote="reaching a crossroad now",
            topic_evidence_quote="Complement proteins mark bacteria and help immune cells destroy them",
        )]),
        segments,
        [],
        {},
        topic="biology",
    )

    assert report.clips == []
    assert report.rejected_reasons == ["proposal_0:contains_filler"]


def test_video_recording_example_is_not_mistaken_for_a_channel_plug():
    segments = [{
        "start": 0.0,
        "end": 15.0,
        "text": "We made a video recording to observe mitosis and count each stage.",
    }]
    report = G._plan_to_report(
        G._Plan(topics=[_topic(
            0,
            0,
            start_quote="We made a video recording",
            end_quote="count each stage",
            topic_evidence_quote="observe mitosis and count each stage",
        )]),
        segments,
        [],
        {},
        topic="mitosis",
    )

    assert len(report.clips) == 1


def test_course_logistics_suffix_is_trimmed_after_complete_teaching():
    segments = [
        {
            "start": 0.0,
            "end": 26.0,
            "text": "Calculations show that this gauge theory is indeed renormalizable.",
        },
        {
            "start": 26.0,
            "end": 31.0,
            "text": "I have no hope that we'll cover that in this course.",
        },
        {
            "start": 31.0,
            "end": 35.0,
            "text": "Lots of hard combinatorics.",
        },
    ]
    report = G._plan_to_report(
        G._Plan(topics=[_topic(
            0,
            2,
            start_quote="Calculations show that",
            end_quote="hard combinatorics",
            topic_evidence_quote=(
                "Calculations show that this gauge theory is indeed renormalizable"
            ),
        )]),
        segments,
        [],
        {},
        topic="renormalization",
    )

    assert len(report.clips) == 1
    assert report.clips[0]["_end_line"] == 0
    assert report.clips[0]["end"] == 26.0


def test_explicit_max_clips_is_respected_below_forty_ceiling():
    segments = _segs(4)
    clips = _run([_topic(i, i, title=f"T{i}") for i in range(4)], segments, {"max_clips": 2})
    assert len(clips) == 2


@pytest.mark.parametrize(
    "overrides,rejected_reason",
    [
        ({"informativeness": 0.7}, None),
        ({"uncertainty": "medium", "uncertainty_reasons": ["boundary_ambiguous"]},
         None),
        ({"uncertainty": "high", "uncertainty_reasons": ["incomplete_context"]},
         "proposal_1:high_uncertainty"),
    ],
)
def test_lower_ranked_candidate_never_poison_independent_kept_clip(overrides, rejected_reason):
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
    assert classification == G._Classification("green", ())
    if rejected_reason:
        assert rejected_reason in report.rejected_reasons


def test_lower_ranked_long_candidate_never_poison_independent_kept_clip():
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
    assert classification == G._Classification("green", ())


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
        "directly_teaches_topic", "substantive", "topic_evidence_quote",
    ):
        assert field in combined
    assert "prompt (at most 16 words)" in combined
    assert "options (at most 8 words each)" in combined
    assert "explanation (one sentence, at most 24 words)" in combined
    assert "chain-of-thought" in combined


def test_boundary_prompt_includes_learner_level_and_rejects_course_framing():
    _system, user = G._boundary_prompts(
        "[0] 00:00 Biology explains cells.",
        1,
        topic="biology",
        learner_level="advanced",
    )
    assert "current level is advanced" in user
    assert "course logistics" in user
    assert "institutional framing" in user
    assert "merely names the subject" in user


def test_boundary_prompt_requires_the_requested_identification_task():
    _system, user = G._boundary_prompts(
        "[0] 00:00 Joined letter strokes reveal the ligature form.",
        1,
        topic="Carolingian minuscule ligature identification",
    )
    assert "identification, recognition, diagnosis" in user
    assert "history or definition alone is not a direct match" in user


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
