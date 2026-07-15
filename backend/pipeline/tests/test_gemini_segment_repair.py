from __future__ import annotations

import time

import pytest

from backend.pipeline import gemini_segment as G


def _topic(start: int, end: int, *, title: str) -> G._BoundaryTopic:
    return G._BoundaryTopic(
        candidate_id=f"candidate-{start}-{end}-{title}",
        start_line=start,
        end_line=end,
        start_quote="So we can now" if start in {3, 9} else "Moles convert directly",
        end_quote=(
            "x equals two"
            if end == 4
            else "result is balanced"
            if end == 10
            else "mass in stoichiometry"
        ),
        title=title,
        learning_objective=f"Understand the complete {title} idea.",
        facet="worked example",
        reason=f"Teach the complete {title} idea.",
        informativeness=0.9,
        topic_relevance=0.9,
        educational_importance=0.9,
        difficulty=0.5,
        directly_teaches_topic=True,
        substantive=True,
        factually_grounded=True,
        topic_evidence_quote=(
            "So we can now solve the equation"
            if start == 3
            else "So we can now balance the reaction"
            if start == 9
            else "Moles convert directly into mass in stoichiometry"
        ),
        self_contained=True,
        is_standalone=True,
        prerequisite_candidate_ids=[],
        uncertainty="low",
        uncertainty_reasons=[],
    )


def _transcript() -> dict:
    texts = [
        "A complete setup introduces the equation.",
        "And this gives us the first condition",
        "Because the next step depends on it",
        "So we can now solve the equation",
        "The worked result is x equals two.",
        "A transition finishes completely.",
        "A complete setup introduces equilibrium.",
        "And this gives us the second condition",
        "Because equilibrium depends on both sides",
        "So we can now balance the reaction",
        "The equilibrium result is balanced.",
        "A nearby conclusion is complete.",
        "Another nearby explanation is complete.",
        "Moles convert directly into mass in stoichiometry.",
        "A later lesson closes completely.",
        "GLOBAL DISTANT SENTINEL SHOULD NEVER ENTER REPAIR.",
    ]
    return {
        "source": "supadata",
        "words": [],
        "segments": [
            {
                "cue_id": f"cue-{index}",
                "start": index * 11.0,
                "end": (index + 1) * 11.0,
                "text": text,
            }
            for index, text in enumerate(texts)
        ],
    }


def test_unpunctuated_fixed_size_cue_edges_are_not_marked_clean() -> None:
    assert G._cue_has_weak_end(
        "we calculate the value by substituting",
        "the numbers into this equation",
        ignore_caption_case=True,
    ) is True
    assert G._cue_has_weak_end(
        "the electron moves toward the higher energy",
        "state when it absorbs a photon",
        ignore_caption_case=True,
    ) is True
    assert G._cue_boundary_confidence(
        "a short complete thought without punctuation",
        ignore_caption_case=True,
    ) < 1.0
    assert G._cue_has_weak_end(
        "ionic bonds transfer electrons between atoms",
        "covalent bonds share pairs of electrons",
        ignore_caption_case=True,
    ) is False
    assert G._cue_has_weak_end(
        "complex organisms like protists, fungi,",
        "plants and animals complete the list.",
        ignore_caption_case=True,
    ) is True


def _live_chain_rule_boundary_segments() -> list[dict]:
    texts = [
        "notation so let me make it so i have h",
        "of x and what i'm curious about is what",
        "is h prime of x",
        "so i want to know h prime of x which",
        "another way of writing it is the",
        "derivative of h",
        "with respect to x these are just",
        "different notations",
        "and to do this i'm going to use the",
        "chain rule i am going to use the chain",
        "rule the chain rule comes into play",
        "every time any time your function can be",
        "used as a composition of more than one",
        "function and as that might not seem",
        "obvious right now but it will hopefully",
        "maybe by the end of this video or the",
        "next one",
        "now what i want to do is a little bit of a thought experiment",
    ]
    starts = [
        43.36, 45.52, 47.92, 49.68, 52.399, 54.239, 55.6, 57.199,
        58.399, 60.16, 62.64, 65.36, 67.76, 69.52, 71.6, 73.6,
        75.36, 76.4,
    ]
    return [
        {
            "cue_id": f"0T0QrHO56qg:cue:{index}",
            "start": start,
            "end": starts[index + 1] if index + 1 < len(starts) else 82.0,
            "text": text,
        }
        for index, (start, text) in enumerate(zip(starts, texts))
    ]


def test_live_chain_rule_fragment_expands_to_cold_context_and_complete_close() -> None:
    segments = _live_chain_rule_boundary_segments()

    start, end, error = G._close_cue_context(
        segments,
        8,
        12,
        ignore_caption_case=True,
    )

    assert (start, end, error) == (0, 16, None)
    assert segments[start]["start"] == pytest.approx(43.36)
    assert segments[end]["text"] == "next one"
    assert "thought experiment" not in G._cue_clip_text(segments, start, end)


def test_live_chain_rule_candidate_is_rebounded_instead_of_rejected() -> None:
    segments = _live_chain_rule_boundary_segments()
    proposal = G._BoundaryTopic(
        candidate_id="chain-rule-applicability",
        start_line=8,
        end_line=12,
        start_quote="and to do this i'm going to use",
        end_quote="composition of more than one",
        title="When the chain rule applies",
        learning_objective="Explain when the chain rule applies to composite functions.",
        facet="chain-rule applicability",
        reason="Defines the condition for using the chain rule.",
        informativeness=0.92,
        topic_relevance=0.98,
        educational_importance=0.91,
        difficulty=0.35,
        directly_teaches_topic=True,
        substantive=True,
        factually_grounded=True,
        topic_evidence_quote="the chain rule comes into play every time",
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
        {"_segment_ignore_caption_case": True},
        topic="chain rule worked example",
    )

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["_start_line"] == 0
    assert clip["_end_line"] == 16
    assert clip["_clip_text"].startswith("notation so let me make it so i have h")
    assert clip["_clip_text"].endswith("the next one")
    assert "thought experiment" not in clip["_clip_text"]


def test_boundary_only_end_uncertainty_is_diagnostic_not_a_rejection() -> None:
    segments = [{
        "cue_id": "cue-0",
        "start": 0.0,
        "end": 8.0,
        "text": (
            "A catalyst lowers activation energy and is not consumed by the "
            "reaction it participates in"
        ),
    }]
    proposal = G._BoundaryTopic(
        candidate_id="catalyst-boundary",
        start_line=0,
        end_line=0,
        start_quote="A catalyst lowers activation energy",
        end_quote="reaction it participates in",
        title="Catalysts and activation energy",
        learning_objective="Explain how a catalyst changes activation energy.",
        facet="activation-energy mechanism",
        reason="Directly explains the catalyst mechanism.",
        informativeness=0.9,
        topic_relevance=0.95,
        educational_importance=0.9,
        difficulty=0.4,
        directly_teaches_topic=True,
        substantive=True,
        factually_grounded=True,
        topic_evidence_quote="A catalyst lowers activation energy",
        self_contained=True,
        is_standalone=True,
        prerequisite_candidate_ids=[],
        uncertainty="low",
        uncertainty_reasons=[],
    )

    assert G._close_cue_context(
        segments, 0, 0, ignore_caption_case=True
    ) == (0, 0, "unresolved_boundary_end")
    report = G._plan_to_report(
        G._BoundaryPlan(topics=[proposal]),
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic="catalyst activation energy",
    )

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert "unresolved_boundary_end" in clip["_boundary_fallback_reasons"]


def test_unanswered_question_remains_semantic_incompleteness() -> None:
    segments = [{
        "start": 0.0,
        "end": 8.0,
        "text": "How does a catalyst lower activation energy?",
    }]

    assert G._close_cue_context(
        segments, 0, 0, ignore_caption_case=True
    ) == (0, 0, "unresolved_weak_end")


def test_lowercase_fragment_uses_previous_unfinished_cue_as_evidence() -> None:
    segments = [
        {"text": "it can grow and develop,"},
        {"text": "reproduce, and it responds to the environment."},
        {"text": "a new sentence starts after a complete thought."},
    ]

    assert G._cue_opens_mid_thought_at(
        segments, 1, ignore_caption_case=True
    ) is True
    assert G._cue_opens_mid_thought_at(
        segments, 2, ignore_caption_case=True
    ) is False


@pytest.mark.parametrize(
    ("fragment", "guarded_prefix"),
    [
        ("“chloroplasts” inside of which photosynthesis occurs.", "“Chloroplasts”"),
        ("[chloroplasts] inside of which photosynthesis occurs.", "[Chloroplasts]"),
    ],
)
def test_opening_quotes_and_brackets_do_not_hide_lowercase_context(
    fragment: str,
    guarded_prefix: str,
) -> None:
    segments = [
        {
            "start": 0.0,
            "end": 5.0,
            "text": "Plant cells contain specialized organelles called",
        },
        {"start": 5.0, "end": 11.0, "text": fragment},
    ]

    assert G._guard_text(
        fragment, ignore_caption_case=True
    ).startswith(guarded_prefix)
    assert G._cue_opens_mid_thought_at(
        segments, 1, ignore_caption_case=True
    ) is True
    assert G._close_cue_context(
        segments, 1, 1, ignore_caption_case=True
    ) == (0, 1, None)


def test_demonstrative_noun_phrase_expands_to_its_antecedent() -> None:
    segments = [
        {
            "start": 0.0,
            "end": 6.0,
            "text": "Cells of the same type can form functional groups.",
        },
        {
            "start": 6.0,
            "end": 12.0,
            "text": "Those groups of cells working together are called tissues.",
        },
    ]

    assert G._close_cue_context(
        segments, 1, 1, ignore_caption_case=True
    ) == (0, 1, None)


@pytest.mark.parametrize(
    "text",
    [
        "Causes include fiscal crisis and inequality.",
        "Uses include generating electricity and heating water.",
        "Returns diminish as more labor is added to fixed capital.",
        "Causes can be identified by examining the evidence.",
        "Causes are classified by using their mechanisms.",
    ],
)
def test_plural_noun_subject_is_complete_at_source_edge(text: str) -> None:
    segments = [{"start": 0.0, "end": 6.0, "text": text}]

    assert G._close_cue_context(
        segments, 0, 0, ignore_caption_case=True
    ) == (0, 0, None)


def test_qcd_back_reference_expands_to_cold_viewer_context() -> None:
    contextual = {
        "start": 0.0,
        "end": 8.0,
        "text": (
            "A physical observable X cannot depend on arbitrary renormalization "
            "parameters."
        ),
    }
    back_reference = {
        "start": 8.0,
        "end": 16.0,
        "text": (
            "x could not depend on these parameters in an exactly analogous way "
            "as before."
        ),
    }

    assert G._cue_opens_mid_thought_at(
        [contextual, back_reference], 1, ignore_caption_case=True
    ) is True
    assert G._close_cue_context(
        [contextual, back_reference], 1, 1, ignore_caption_case=True
    ) == (0, 1, None)
    assert G._close_cue_context(
        [back_reference], 0, 0, ignore_caption_case=True
    ) == (0, 0, "unresolved_weak_start")


def test_qcd_back_reference_rejects_an_unrelated_prior_cue() -> None:
    segments = [
        {
            "start": 0.0,
            "end": 6.0,
            "text": "Welcome back to the lecture.",
        },
        {
            "start": 6.0,
            "end": 14.0,
            "text": (
                "X could not depend on these parameters in an exactly analogous "
                "way as before."
            ),
        },
    ]

    assert G._close_cue_context(
        segments, 1, 1, ignore_caption_case=True
    ) == (0, 1, "unresolved_weak_start")


@pytest.mark.parametrize(
    ("unrelated_context", "back_reference"),
    [
        (
            "We discussed the course schedule and office hours.",
            "The calculation proceeds as discussed earlier.",
        ),
        (
            "We defined the grading policy before class.",
            "The coupling was defined previously.",
        ),
    ],
)
def test_reference_trigger_verbs_do_not_resolve_unrelated_context(
    unrelated_context: str,
    back_reference: str,
) -> None:
    segments = [
        {"start": 0.0, "end": 6.0, "text": unrelated_context},
        {"start": 6.0, "end": 12.0, "text": back_reference},
    ]

    assert G._close_cue_context(
        segments, 1, 1, ignore_caption_case=True
    ) == (0, 1, "unresolved_weak_start")


def test_single_shared_noun_does_not_resolve_prior_context() -> None:
    segments = [
        {
            "start": 0.0,
            "end": 6.0,
            "text": "A homework exercise contains one calculation.",
        },
        {
            "start": 6.0,
            "end": 12.0,
            "text": "The calculation proceeds as discussed earlier.",
        },
    ]

    assert G._close_cue_context(
        segments, 1, 1, ignore_caption_case=True
    ) == (0, 1, "unresolved_weak_start")

    plural_segments = [
        {
            "start": 0.0,
            "end": 6.0,
            "text": "This chapter lists several couplings.",
        },
        {
            "start": 6.0,
            "end": 12.0,
            "text": "The couplings were defined previously.",
        },
    ]
    assert G._close_cue_context(
        plural_segments, 1, 1, ignore_caption_case=True
    ) == (0, 1, "unresolved_weak_start")


def test_fronted_quantified_demonstrative_expands_to_its_antecedent() -> None:
    segments = [
        {
            "start": 0.0,
            "end": 6.0,
            "text": "Genes on the X chromosome are called X-linked genes.",
        },
        {
            "start": 6.0,
            "end": 12.0,
            "text": (
                "If one of these genes is recessive, males express the trait "
                "because they have only one X chromosome."
            ),
        },
    ]

    assert G._close_cue_context(
        segments, 1, 1, ignore_caption_case=True
    ) == (0, 1, None)

    for antecedent, contextual_opener in (
        (
            segments[0]["text"],
            "If these genes are recessive, males express the trait.",
        ),
        (
            "A recessive allele can be expressed in a male.",
            "When this happens, the recessive trait becomes visible.",
        ),
        (
            "The genetic trials produced several uncommon results.",
            "Although those results are uncommon, they support the model.",
        ),
    ):
        contextual_segments = [
            {**segments[0], "text": antecedent},
            {**segments[1], "text": contextual_opener},
        ]
        assert G._close_cue_context(
            contextual_segments, 1, 1, ignore_caption_case=True
        ) == (0, 1, None)


def test_uppercase_fragment_expands_after_unfinished_function_word() -> None:
    segments = [
        {
            "start": 0.0,
            "end": 5.0,
            "text": "Fun fact: if you stretched out all the",
        },
        {
            "start": 5.0,
            "end": 11.0,
            "text": (
                "DNA of one cell, it would be about two meters long."
            ),
        },
    ]

    assert G._cue_opens_mid_thought_at(
        segments, 1, ignore_caption_case=True
    ) is True
    assert G._close_cue_context(
        segments, 1, 1, ignore_caption_case=True
    ) == (0, 1, None)


def test_structural_only_previous_tail_does_not_capture_clean_opener() -> None:
    segments = [
        {"start": 0.0, "end": 4.0, "text": "Let's move on to"},
        {
            "start": 4.0,
            "end": 10.0,
            "text": (
                "Photosynthesis converts light energy into stored chemical energy."
            ),
        },
    ]

    assert G._close_cue_context(
        segments, 1, 1, ignore_caption_case=True
    ) == (1, 1, None)


def test_by_the_way_opener_expands_or_fails_closed_at_a_section_edge() -> None:
    opener = "Oh yeah, by the way, multicellular organisms can reproduce sexually."
    with_context = [
        {
            "start": 0.0,
            "end": 6.0,
            "text": "Organisms reproduce to pass genetic information to offspring.",
        },
        {"start": 6.0, "end": 12.0, "text": opener},
    ]

    assert G._structural_filler_matches(opener)
    assert G._close_cue_context(
        with_context, 1, 1, ignore_caption_case=True
    ) == (0, 1, None)
    assert G._close_cue_context(
        [with_context[1]], 0, 0, ignore_caption_case=True
    ) == (0, 0, "unresolved_weak_start")


def test_conversational_edge_labels_are_structural_not_topic_words() -> None:
    for text in (
        "Cool! Photosynthesis converts light energy.",
        "Hey, cells contain DNA.",
        "Fun fact! One cell contains about two meters of DNA.",
        "The explanation is complete. Brilliant.",
    ):
        assert G._structural_filler_matches(text)

    assert G._structural_filler_matches(
        "Cool stars can have surface temperatures below 4,000 kelvin."
    ) == []


def test_adversative_question_expands_but_bare_question_can_open() -> None:
    contextual = [
        {
            "start": 0.0,
            "end": 5.0,
            "text": "Enzymes make life possible by speeding chemical reactions,",
        },
        {
            "start": 5.0,
            "end": 11.0,
            "text": "but what even is life? Scientists disagree about its exact definition.",
        },
    ]
    bare = [{**contextual[1], "text": "What even is life? Scientists study its traits."}]

    assert G._close_cue_context(
        contextual, 1, 1, ignore_caption_case=True
    ) == (0, 1, None)
    assert G._close_cue_context(
        bare, 0, 0, ignore_caption_case=True
    ) == (0, 0, None)


def test_live_biology_comparative_question_recovers_its_missing_setup() -> None:
    segments = [
        {
            "start": 0.0,
            "end": 5.0,
            "text": "How did simple prokaryotic cells become",
        },
        {
            "start": 5.0,
            "end": 12.0,
            "text": (
                "more complicated? This is the theory of endosymbiosis."
            ),
        },
    ]

    start, end, error = G._close_cue_context(
        segments,
        1,
        1,
        ignore_caption_case=True,
    )

    assert (start, end, error) == (0, 1, None)


def test_exact_live_biology_question_tail_never_surfaces_without_full_setup() -> None:
    texts = [
        "interesting theory that I just wanted to",
        "teach you because it is so",
        "fascinating, which is our ability or our",
        "effort to try to understand where did",
        "the eukaryotes come from? What was",
        "responsible for such a significant",
        "departure in these cells? Why do some",
        "cells have such a basic appearance and",
        "function and another cell type is much",
        "more complicated? This is the theory of endosymbiosis.",
    ]
    segments = [
        {"start": index * 2.5, "end": (index + 1) * 2.5, "text": text}
        for index, text in enumerate(texts)
    ]

    _start, _end, error = G._close_cue_context(
        segments,
        9,
        9,
        ignore_caption_case=True,
    )

    assert error == "unresolved_weak_start"


def test_live_biology_terminal_contraction_expands_to_complete_clause() -> None:
    segments = [
        {
            "start": 0.0,
            "end": 8.0,
            "text": "The eukaryotic cell is the type of cell which makes up you.",
        },
        {
            "start": 8.0,
            "end": 13.0,
            "text": "All right. Now, let's back up for just a moment. I'd",
        },
        {
            "start": 13.0,
            "end": 20.0,
            "text": "like to distinguish prokaryotic cells from eukaryotic cells.",
        },
    ]

    start, end, error = G._close_cue_context(
        segments,
        0,
        1,
        ignore_caption_case=True,
    )

    assert (start, end, error) == (0, 2, None)


def test_live_biology_terminal_contraction_finishes_the_following_explanation() -> None:
    texts = [
        "The eukaryotic cell is the type of cell which makes up you.",
        "All right. Now, let's back up for just a moment. I'd",
        "mentioned the archaea. And the archaea are",
        "the most unknown and certainly worldwide",
        "the smallest in abundance. But the archaea basically are",
        "these small little cells, single cell",
        "organisms that only live in environments",
        "where nothing else can live. In other",
        "words, even bacteria couldn't live here.",
    ]
    segments = [
        {"start": index * 3.0, "end": (index + 1) * 3.0, "text": text}
        for index, text in enumerate(texts)
    ]

    start, end, error = G._close_cue_context(
        segments,
        0,
        1,
        ignore_caption_case=True,
    )

    assert (start, end, error) == (0, 8, None)


def test_unfinished_bare_subject_expands_to_its_conclusion() -> None:
    segments = [
        {
            "start": 0.0,
            "end": 6.0,
            "text": "Helpful bacteria live in symbiosis with you, so you",
        },
        {
            "start": 6.0,
            "end": 12.0,
            "text": "give them food and they help you digest it.",
        },
    ]

    assert G._cue_has_weak_end(
        segments[0]["text"],
        segments[1]["text"],
        ignore_caption_case=True,
    ) is True
    assert G._close_cue_context(
        segments, 0, 0, ignore_caption_case=True
    ) == (0, 1, None)


def test_dangling_degree_transition_trims_to_prior_complete_teaching() -> None:
    segments = [
        {
            "start": 0.0,
            "end": 6.0,
            "text": "Translation uses ribosomes to assemble a protein.",
        },
        {
            "start": 6.0,
            "end": 12.0,
            "text": "Hey, this genetics stuff is pretty",
        },
        {
            "start": 12.0,
            "end": 18.0,
            "text": "cool, so subscribe for another genetics lesson.",
        },
    ]

    assert G._cue_has_weak_end(
        segments[1]["text"],
        segments[2]["text"],
        ignore_caption_case=True,
    ) is True
    assert G._close_cue_context(
        segments, 0, 1, ignore_caption_case=True
    ) == (0, 0, None)


def test_dangling_edge_framing_trims_without_rejecting_the_teaching() -> None:
    texts = [
        "On this strand, we talked about that.",
        "So now the next goal here is that we",
        "have to remove",
        "those primers, but before we do that",
        "I need to mention one more thing.",
        "DNA polymerase proofreads the new strand and removes a mismatched base.",
    ]
    segments = [
        {"start": index * 5.0, "end": (index + 1) * 5.0, "text": text}
        for index, text in enumerate(texts)
    ]

    assert G._close_cue_context(
        segments, 0, 5, ignore_caption_case=True
    ) == (0, 5, None)
    assert G._trim_structural_filler_edges(
        segments, 0, 5, ignore_caption_case=True
    ) == (2, 5)


def test_edge_filler_trim_keeps_teaching_with_a_weak_conjunction_opening() -> None:
    segments = [
        {"start": 0.0, "end": 4.0, "text": "Welcome back to the lesson."},
        {
            "start": 4.0,
            "end": 9.0,
            "text": "And a catalyst lowers activation energy for the reaction.",
        },
    ]

    assert G._trim_structural_filler_edges(
        segments, 0, 1, ignore_caption_case=True
    ) == (1, 1)


def test_anaphoric_function_expands_to_subject_and_completes_comparison() -> None:
    texts = [
        "How do we get rid of the RNA primers?",
        "Well, DNA polymerase III is not the answer.",
        "The next enzyme, as if there aren't enough enzymes,",
        "is called DNA polymerase I.",
        "DNA polymerase I removes each primer and replaces it with DNA.",
        "You know what else it can do? One more function.",
        "It proofreads a mismatch, cuts it out, and inserts the correct nucleotide.",
        "The last function of DNA polymerase I is proofreading incorrect base pairs.",
        "Okay, so the big difference, if you are asked between",
        (
            "DNA polymerase I and DNA polymerase III is that polymerase I also "
            "removes RNA primers."
        ),
        "The next thing is DNA ligase.",
    ]
    segments = [
        {
            "cue_id": f"cue-{index}",
            "start": index * 5.0,
            "end": (index + 1) * 5.0,
            "text": text,
        }
        for index, text in enumerate(texts)
    ]
    proposal = G._BoundaryTopic(
        candidate_id="polymerase-proofreading",
        start_line=5,
        end_line=8,
        start_quote="You know what else it can",
        end_quote="if you are asked between",
        title="Polymerase I proofreading",
        learning_objective="Explain how polymerase I proofreads mismatches.",
        facet="mismatch proofreading",
        reason="Directly explains mismatch proofreading.",
        informativeness=0.9,
        topic_relevance=0.9,
        educational_importance=0.9,
        difficulty=0.5,
        directly_teaches_topic=True,
        substantive=True,
        factually_grounded=True,
        topic_evidence_quote="proofreads a mismatch, cuts it out",
        self_contained=True,
        is_standalone=True,
        prerequisite_candidate_ids=[],
    )

    report = G._plan_to_report(
        G._BoundaryPlan(topics=[proposal]),
        segments,
        [],
        {},
        topic="DNA polymerase proofreading mismatches",
    )

    assert report.rejected_reasons == []
    assert report.clips[0]["cue_ids"] == [
        "cue-4", "cue-5", "cue-6", "cue-7", "cue-8", "cue-9",
    ]
    assert report.clips[0]["start_quote"].startswith("DNA polymerase I")
    assert report.clips[0]["end_quote"].endswith("removes RNA primers.")


def test_only_explicit_topic_resets_stop_context_expansion() -> None:
    assert G._FORWARD_TOPIC_TRANSITION_RE.match(
        "Okay, now we got to discuss RNA primers."
    )
    assert G._FORWARD_TOPIC_TRANSITION_RE.match(
        "Next we will discuss integration."
    )
    assert G._FORWARD_TOPIC_TRANSITION_RE.match("The next topic is DNA ligase.")
    assert G._FORWARD_TOPIC_TRANSITION_RE.match(
        "Now we need to move on to integration."
    )
    assert not G._FORWARD_TOPIC_TRANSITION_RE.match(
        "Now we need to multiply by the inner derivative."
    )
    assert not G._FORWARD_TOPIC_TRANSITION_RE.match(
        "Now we have to apply the chain rule."
    )
    assert not G._FORWARD_TOPIC_TRANSITION_RE.match(
        "The next step is we multiply both sides by two."
    )
    assert not G._FORWARD_TOPIC_TRANSITION_RE.match(
        "Now the next thing is we compare opportunity cost with sunk cost."
    )
    assert not G._FORWARD_TOPIC_TRANSITION_RE.match(
        "Next we will differentiate the outer function."
    )
    assert not G._FORWARD_TOPIC_TRANSITION_RE.match(
        "Now we got to multiply by two."
    )
    assert not G._FORWARD_TOPIC_TRANSITION_RE.match(
        "Now we need to look at the denominator."
    )
    assert not G._FORWARD_TOPIC_TRANSITION_RE.match(
        "Now we need to examine the sign."
    )
    assert not G._FORWARD_TOPIC_TRANSITION_RE.match(
        "The next thing is multiplication."
    )
    assert not G._FORWARD_TOPIC_TRANSITION_RE.match(
        "The next step is substitution."
    )
    assert not G._FORWARD_TOPIC_TRANSITION_RE.match("Let's say x equals two.")
    assert not G._FORWARD_TOPIC_TRANSITION_RE.match("How do we solve this example?")


def test_procedural_now_step_remains_part_of_worked_example() -> None:
    texts = [
        "What is the derivative of y equals three x squared?",
        "Now we need to multiply by",
        "the inner derivative, so the final derivative is eighteen x.",
    ]
    segments = [
        {"start": index * 5.0, "end": (index + 1) * 5.0, "text": text}
        for index, text in enumerate(texts)
    ]

    assert G._close_cue_context(
        segments, 0, 0, ignore_caption_case=True
    ) == (0, 2, None)


def test_substantive_next_step_and_thing_complete_the_selected_question() -> None:
    cases = [
        (
            "How do we isolate x?",
            "The next step is we multiply both sides by two.",
        ),
        (
            "How should we compare the two costs?",
            "Now the next thing is we compare opportunity cost with sunk cost.",
        ),
    ]

    for question, answer in cases:
        segments = [
            {"start": 0.0, "end": 5.0, "text": question},
            {"start": 5.0, "end": 10.0, "text": answer},
        ]
        assert G._close_cue_context(
            segments, 0, 0, ignore_caption_case=True
        ) == (0, 1, None)


def test_procedural_next_and_now_answers_are_not_topic_resets() -> None:
    cases = [
        (
            "How should the outer function be differentiated?",
            "Next we will differentiate the outer function.",
        ),
        (
            "What factor should multiply both sides?",
            "Now we got to multiply by two.",
        ),
        (
            "What part of the fraction should be checked?",
            "Now we need to look at the denominator.",
        ),
        (
            "How do we determine whether the result is positive?",
            "Now we need to examine the sign.",
        ),
    ]

    for question, answer in cases:
        segments = [
            {"start": 0.0, "end": 5.0, "text": question},
            {"start": 5.0, "end": 10.0, "text": answer},
        ]
        assert G._close_cue_context(
            segments, 0, 0, ignore_caption_case=True
        ) == (0, 1, None)


def test_answer_shaped_named_steps_are_not_topic_resets() -> None:
    for answer in (
        "The next thing is multiplication.",
        "The next step is substitution.",
    ):
        segments = [
                {
                    "start": 0.0,
                    "end": 5.0,
                    "text": (
                        "Two x equals four is an equation that requires one operation."
                    ),
                },
            {"start": 5.0, "end": 10.0, "text": "How do we solve it?"},
            {"start": 10.0, "end": 15.0, "text": answer},
        ]
        assert G._close_cue_context(
            segments, 0, 1, ignore_caption_case=True
        ) == (0, 2, None)


def test_substantive_next_step_is_not_structural_filler() -> None:
    substantive = "Now the next step is we multiply both sides by two."
    dangling = "So now the next goal here is that we"

    assert G._cue_is_only_structural_filler(substantive) is False
    assert G._cue_is_only_structural_filler(dangling) is True


def test_question_only_clip_fails_before_a_topic_transition() -> None:
    segments = [
        {
            "start": 0.0,
            "end": 5.0,
            "text": "How does DNA polymerase proofread a mismatched base?",
        },
        {
            "start": 5.0,
            "end": 10.0,
            "text": "The next topic is DNA ligase.",
        },
    ]

    assert G._close_cue_context(
        segments, 0, 0, ignore_caption_case=True
    ) == (0, 0, "unresolved_weak_end")


def test_question_only_model_range_fails_after_quote_projection() -> None:
    segments = [
        {
            "cue_id": "cue-0",
            "start": 0.0,
            "end": 5.0,
            "text": "How does DNA polymerase proofread a mismatched base?",
        },
        {
            "cue_id": "cue-1",
            "start": 5.0,
            "end": 10.0,
            "text": "The next topic is DNA ligase.",
        },
    ]
    proposal = G._BoundaryTopic(
        candidate_id="question-only",
        start_line=0,
        end_line=0,
        start_quote="How does DNA polymerase proofread",
        end_quote="proofread a mismatched base",
        title="DNA polymerase proofreading",
        learning_objective="Explain how DNA polymerase proofreads a mismatch.",
        facet="proofreading mechanism",
        reason="Teach DNA polymerase proofreading.",
        informativeness=0.9,
        topic_relevance=0.9,
        educational_importance=0.9,
        difficulty=0.5,
        directly_teaches_topic=True,
        substantive=True,
        factually_grounded=True,
        topic_evidence_quote="DNA polymerase proofread a mismatched base",
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
        {},
        topic="DNA polymerase proofreading",
    )

    assert report.clips == []
    assert report.rejected_reasons == ["proposal_0:unresolved_weak_end"]


def test_semantic_for_now_survives_expanded_end_projection() -> None:
    texts = [
        "The approximation works because",
        "the neglected terms are small for now.",
        "The next topic is exact solutions.",
    ]
    segments = [
        {
            "cue_id": f"cue-{index}",
            "start": index * 5.0,
            "end": (index + 1) * 5.0,
            "text": text,
        }
        for index, text in enumerate(texts)
    ]
    proposal = G._BoundaryTopic(
        candidate_id="approximation-validity",
        start_line=0,
        end_line=0,
        start_quote="The approximation works because",
        end_quote="approximation works because",
        title="Approximation validity",
        learning_objective="Explain why the approximation works.",
        facet="small neglected terms",
        reason="Explains the approximation's validity.",
        informativeness=0.9,
        topic_relevance=0.9,
        educational_importance=0.9,
        difficulty=0.5,
        directly_teaches_topic=True,
        substantive=True,
        factually_grounded=True,
        topic_evidence_quote=(
            "The approximation works because the neglected terms"
        ),
        self_contained=True,
        is_standalone=True,
        prerequisite_candidate_ids=[],
    )

    report = G._plan_to_report(
        G._BoundaryPlan(topics=[proposal]),
        segments,
        [],
        {},
        topic="approximation validity",
    )

    assert report.rejected_reasons == []
    assert report.clips[0]["_clip_text"].endswith("small for now.")
    assert report.clips[0]["end_quote"].endswith("small for now.")
    assert G._TRAILING_TRANSITION_FRAGMENT_RE.sub(
        "", "Everything else is the same though now."
    ).rstrip() == "Everything else is the same"


def test_mixed_complete_thought_and_dangling_cta_fails_closed() -> None:
    segments = [
        {
            "start": 0.0,
            "end": 6.0,
            "text": "An action potential reaches the next node and",
        },
        {
            "start": 6.0,
            "end": 14.0,
            "text": (
                "triggers another action potential, which repeats the cycle. "
                "Something in my brain is telling me that you"
            ),
        },
        {
            "start": 14.0,
            "end": 20.0,
            "text": "should subscribe to continue learning.",
        },
    ]

    assert G._close_cue_context(
        segments, 0, 1, ignore_caption_case=True
    ) == (0, 1, "unresolved_weak_end")


def test_complete_teaching_before_incomplete_edge_suffix_is_recovered() -> None:
    segments = [
        {
            "cue_id": "cue-0",
            "start": 0.0,
            "end": 6.0,
            "text": "An action potential reaches the next node and",
        },
        {
            "cue_id": "cue-1",
            "start": 6.0,
            "end": 14.0,
            "text": (
                "triggers another action potential, which repeats the cycle. "
                "Something in my brain is telling me that you"
            ),
        },
        {
            "cue_id": "cue-2",
            "start": 14.0,
            "end": 20.0,
            "text": "should subscribe to continue learning.",
        },
    ]
    proposal = G._BoundaryTopic(
        candidate_id="action-potential-cycle",
        start_line=0,
        end_line=1,
        start_quote="An action potential reaches the next",
        end_quote="brain is telling me that you",
        title="Action-potential propagation",
        learning_objective="Explain how an action potential propagates between nodes.",
        facet="propagation mechanism",
        reason="Explains the repeated electrical propagation cycle.",
        informativeness=0.92,
        topic_relevance=0.96,
        educational_importance=0.9,
        difficulty=0.45,
        directly_teaches_topic=True,
        substantive=True,
        factually_grounded=True,
        topic_evidence_quote=(
            "triggers another action potential, which repeats the cycle"
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
        {"_segment_ignore_caption_case": True},
        topic="action potential propagation",
    )

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["_clip_text"].endswith("which repeats the cycle.")
    assert "brain" not in clip["_clip_text"]
    assert "trimmed_incomplete_end_suffix" in clip["_boundary_fallback_reasons"]


def test_complete_degree_predicate_with_punctuation_remains_clean() -> None:
    assert G._cue_has_weak_end(
        "The flower is pretty.",
        "Pollinators visit the flower.",
        ignore_caption_case=True,
    ) is False
    assert G._cue_has_weak_end(
        "The flower is pretty",
        "Pollinators visit the flower.",
        ignore_caption_case=True,
    ) is False


def test_comparative_question_without_dependency_evidence_remains_eligible() -> None:
    segments = [{
        "start": 0.0,
        "end": 7.0,
        "text": "more complicated? This is the theory of endosymbiosis.",
    }]

    assert G._close_cue_context(
        segments,
        0,
        0,
        ignore_caption_case=True,
    ) == (0, 0, None)


def test_complete_questions_and_contractions_remain_clean() -> None:
    clean_questions = [
        "what is endosymbiosis? It explains organelle origins.",
        "how do mitochondria retain DNA? Endosymbiosis explains why.",
        "is endosymbiosis supported by evidence? Yes, several lines support it.",
        "can prokaryotes live independently? Many species can.",
    ]
    for text in clean_questions:
        assert G._cue_opens_mid_thought_at(
            [{"text": text}], 0, ignore_caption_case=True
        ) is False

    assert G._cue_has_weak_end(
        "I'd like to distinguish prokaryotic cells from eukaryotic cells.",
        "",
        ignore_caption_case=True,
    ) is False
    assert G._cue_has_weak_end(
        "The explanation is complete. I'd.",
        "",
        ignore_caption_case=True,
    ) is True
    for text in (
        "Can cells regulate ion balance? They can.",
        "Was the hypothesis correct? It was.",
        "Would I use this method again? I would.",
        "The final answer is yes, we can.",
    ):
        assert G._cue_has_weak_end(
            text,
            "",
            ignore_caption_case=True,
        ) is False


def test_subject_question_after_unpunctuated_cues_is_not_over_rejected() -> None:
    segments = [
        {
            "start": index * 3.0,
            "end": (index + 1) * 3.0,
            "text": "the preceding teaching explanation continues without punctuation",
        }
        for index in range(9)
    ]
    segments.append({
        "start": 27.0,
        "end": 33.0,
        "text": "cells use ATP for energy? Yes, they do.",
    })

    assert G._close_cue_context(
        segments,
        9,
        9,
        ignore_caption_case=True,
    ) == (9, 9, None)


def test_production_gene_clip_includes_the_first_selection_criterion() -> None:
    segments = [
        {
            "start": 1024.640,
            "end": 1048.640,
            "text": (
                "Genes are the core unit of natural selection. For something to "
                "undergo selection, it needs three characteristics. First, it needs "
                "to make near identical copies of itself."
            ),
        },
        {
            "start": 1048.640,
            "end": 1074.960,
            "text": (
                "Second, it needs traits that affect its interaction with the "
                "environment and its probability of survival and reproduction. "
                "What about something bigger, like a chromosome?"
            ),
        },
        {
            "start": 1074.960,
            "end": 1101.200,
            "text": "This is why the gene is the unit of natural selection.",
        },
    ]

    start, end, error = G._close_cue_context(
        segments,
        1,
        2,
        ignore_caption_case=True,
    )

    assert (start, end, error) == (0, 2, None)


def test_production_dialogue_reply_recovers_context_beyond_old_window() -> None:
    segments = [
        {
            "start": 0.080,
            "end": 30.555,
            "text": "Why does poop smell bad? How do you think it smells to flies?",
        },
        {
            "start": 30.555,
            "end": 58.640,
            "text": (
                "Yeah- They like it. Animals love stinky things. Poop smells good "
                "to flies because it is food, but dangerous bacteria make humans "
                "avoid it."
            ),
        },
    ]

    start, end, error = G._close_cue_context(
        segments,
        1,
        1,
        ignore_caption_case=True,
    )

    assert (start, end, error) == (0, 1, None)


def test_production_anaphoric_question_recovers_its_distant_antecedent() -> None:
    segments = [
        {
            "start": 137.0,
            "end": 184.0,
            "text": "Homeostasis keeps a cell's internal environment stable.",
        },
        {
            "start": 184.0,
            "end": 218.0,
            "text": (
                "Ok. But like, how does the cell do that? The secret lies in "
                "the cell membrane, which controls what goes in and out."
            ),
        },
    ]

    start, end, error = G._close_cue_context(
        segments,
        1,
        1,
        ignore_caption_case=True,
    )

    assert (start, end, error) == (0, 1, None)


def test_model_selected_complete_context_survives_a_long_caption_gap() -> None:
    segments = [
        {
            "start": 0.0,
            "end": 8.0,
            "text": "The chain rule multiplies the outer derivative because",
        },
        {
            "start": 42.0,
            "end": 50.0,
            "text": "the inner derivative supplies the remaining rate of change.",
        },
    ]

    start, end, error = G._close_cue_context(
        segments,
        0,
        1,
        ignore_caption_case=True,
    )

    assert (start, end, error) == (0, 1, None)


def test_plan_keeps_live_biology_cue_when_only_opening_boundary_is_weak() -> None:
    segments = [
        {
            "cue_id": "tZE_fQFK8EY:cue:12",
            "start": 350.0,
            "end": 360.0,
            "text": "The previous section closes before a long pause.",
        },
        {
            "cue_id": "tZE_fQFK8EY:cue:13",
            "start": 392.934,
            "end": 423.0,
            "text": (
                "And at the same time, some of these traits can be found in "
                "non-living things, too. Viruses complicate the definition of "
                "life. Let's head to the Thought Bubble."
            ),
        },
    ]
    proposal = _topic(1, 1, title="biology").model_copy(update={
        "start_quote": "And at the same time",
        "end_quote": "Let's head to the Thought Bubble",
        "topic_evidence_quote": "Viruses complicate the definition of life",
    })

    report = G._plan_to_report(
        G._BoundaryPlan(topics=[proposal]),
        segments,
        [],
        {},
        topic="biology",
    )

    assert len(report.clips) == 1
    clip = report.clips[0]
    assert clip["_clip_text"] == "Viruses complicate the definition of life"
    assert "recovered_same_cue_sentence_start" in clip["_boundary_fallback_reasons"]
    assert "trimmed_trailing_edge_noise" in clip["_boundary_fallback_reasons"]


def test_plan_keeps_unpunctuated_biology_cue_when_information_is_grounded() -> None:
    segments = [
        {
            "cue_id": "tZE_fQFK8EY:cue:12",
            "start": 350.0,
            "end": 360.0,
            "text": "The previous section closes before a long pause.",
        },
        {
            "cue_id": "tZE_fQFK8EY:cue:13",
            "start": 392.934,
            "end": 423.0,
            "text": (
                "And at the same time some of these traits can be found in non "
                "living things too which makes the definition complicated before "
                "we eventually lets head to the thought bubble"
            ),
        },
    ]
    proposal = _topic(1, 1, title="biology").model_copy(update={
        "start_quote": "And at the same time",
        "end_quote": "head to the thought bubble",
        "topic_evidence_quote": "these traits can be found in non living things",
    })

    report = G._plan_to_report(
        G._BoundaryPlan(topics=[proposal]),
        segments,
        [],
        {},
        topic="biology",
    )

    assert len(report.clips) == 1
    assert "unresolved_weak_start" in report.clips[0]["_boundary_fallback_reasons"]


def test_production_end_extends_into_following_gerund_explanation() -> None:
    segments = [
        {
            "start": 0.0,
            "end": 44.0,
            "text": (
                "The production possibilities frontier is the same thing. "
                "It's a model"
            ),
        },
        {
            "start": 44.0,
            "end": 50.0,
            "text": "showing the tradeoffs between producing two different goods.",
        },
    ]

    start, end, error = G._close_cue_context(
        segments,
        0,
        0,
        ignore_caption_case=True,
    )

    assert (start, end, error) == (0, 1, None)


def test_production_end_extends_into_short_auxiliary_continuation() -> None:
    segments = [
        {
            "start": 4865.84,
            "end": 4870.4,
            "text": "I'll certainly give you a hint as to how two",
        },
        {
            "start": 4869.08,
            "end": 4871.88,
            "text": "will happen.",
        },
    ]

    start, end, error = G._close_cue_context(
        segments,
        0,
        0,
        ignore_caption_case=True,
    )

    assert (start, end, error) == (0, 1, None)


def test_production_terminal_dangling_transition_is_trimmed_or_rejected() -> None:
    segments = [
        {
            "start": 309.45,
            "end": 393.0,
            "text": (
                "After deletion all leaves must remain at the same level and the "
                "tree must satisfy all the B+ tree properties."
            ),
        },
        {
            "start": 393.0,
            "end": 399.89,
            "text": "All right, let's",
        },
    ]

    start, end, error = G._close_cue_context(
        segments,
        0,
        1,
        ignore_caption_case=True,
    )
    assert (start, end, error) == (0, 0, None)

    start, end, error = G._close_cue_context(
        segments,
        1,
        1,
        ignore_caption_case=True,
    )
    assert (start, end, error) == (1, 1, "unresolved_weak_end")

    inside_last_cue = [{
        "start": 309.45,
        "end": 399.89,
        "text": (
            "After deletion all leaves must remain at the same level and the "
            "tree must satisfy all the B+ tree properties. All right, let's"
        ),
    }]
    start, end, error = G._close_cue_context(
        inside_last_cue,
        0,
        0,
        ignore_caption_case=True,
    )
    assert (start, end, error) == (0, 0, "unresolved_weak_end")


def test_production_trailing_forward_setup_is_trimmed_to_complete_teaching() -> None:
    segments = [
        {
            "start": 495.33,
            "end": 540.0,
            "text": (
                "A B-tree deletion can redistribute a key from a neighboring sibling."
            ),
        },
        {
            "start": 540.0,
            "end": 568.0,
            "text": (
                "So this works, as long as we have a sibling we can take from."
            ),
        },
        {
            "start": 568.0,
            "end": 583.0,
            "text": (
                "But what happens if both of our sibling nodes are already at minimum?"
            ),
        },
        {
            "start": 583.0,
            "end": 594.32,
            "text": "Now we can't take from a sibling.",
        },
        {
            "start": 594.32,
            "end": 606.0,
            "text": "Instead, we merge the node with a sibling and pull down a key.",
        },
    ]

    start, end, error = G._close_cue_context(
        segments,
        0,
        3,
        ignore_caption_case=True,
    )
    assert (start, end, error) == (0, 1, None)

    start, end, error = G._close_cue_context(
        segments,
        2,
        3,
        ignore_caption_case=True,
    )
    assert (start, end, error) == (0, 4, None)


def test_unpunctuated_prefix_expands_into_following_solution() -> None:
    segments = [
        {
            "start": 0.0,
            "end": 8.0,
            "text": "A pivot row with zero cannot be used as the divisor",
        },
        {
            "start": 8.0,
            "end": 14.0,
            "text": "But what happens if the pivot is zero?",
        },
        {
            "start": 14.0,
            "end": 19.0,
            "text": "Now we can't divide by that pivot.",
        },
        {
            "start": 19.0,
            "end": 27.0,
            "text": "Instead, swap rows and continue Gaussian elimination.",
        },
    ]

    start, end, error = G._close_cue_context(
        segments,
        0,
        2,
        ignore_caption_case=True,
    )

    assert (start, end, error) == (0, 3, None)


def test_boundary_plan_keeps_grounded_unit_when_forward_setup_is_trimmed() -> None:
    segments = [
        {
            "cue_id": "K1a2Bk8NrYQ:cue:110",
            "start": 540.0,
            "end": 568.0,
            "text": "A B-tree can borrow a key from its sibling to restore balance.",
        },
        {
            "cue_id": "K1a2Bk8NrYQ:cue:111",
            "start": 568.0,
            "end": 594.32,
            "text": (
                "But what happens if both siblings are already at minimum? "
                "Now we can't take from a sibling."
            ),
        },
        {
            "cue_id": "K1a2Bk8NrYQ:cue:113",
            "start": 594.32,
            "end": 606.0,
            "text": "Instead, merge the node with a sibling and pull down a key.",
        },
    ]
    proposal = _topic(0, 1, title="When borrowing fails, merge nodes").model_copy(update={
        "start_quote": "A B-tree can borrow a key",
        "end_quote": "can't take from a sibling",
        "learning_objective": "Explain how a merge resolves an underfull node.",
        "facet": "merge operation",
        "reason": "Shows why the underfull node must merge with a sibling.",
        "topic_evidence_quote": (
            "A B-tree can borrow a key from its sibling"
        ),
    })

    report = G._plan_to_report(
        G._BoundaryPlan(topics=[proposal]),
        segments,
        [],
        {},
        topic="B-tree deletion rebalancing",
    )

    assert len(report.clips) == 1
    assert report.clips[0]["cue_ids"] == ["K1a2Bk8NrYQ:cue:110"]
    assert "ungrounded_end_quote" in report.clips[0]["_boundary_fallback_reasons"]


def test_complete_cannot_explanation_is_not_mistaken_for_a_forward_setup() -> None:
    segments = [{
        "start": 0.0,
        "end": 8.0,
        "text": (
            "What happens if the denominator is zero? We can't divide by zero."
        ),
    }]

    start, end, error = G._close_cue_context(
        segments,
        0,
        0,
        ignore_caption_case=True,
    )

    assert (start, end, error) == (0, 0, None)


def test_dirty_edges_use_only_the_one_low_thinking_selector_call(monkeypatch):
    transcript = _transcript()
    selector = G._BoundaryPlan(topics=[
        _topic(3, 4, title="equation"),
        _topic(9, 10, title="equilibrium"),
        _topic(13, 13, title="stoichiometry"),
    ])
    calls = []
    events = []
    reserve = object()

    def fake_call(system, user, schema, **kwargs):
        calls.append((system, user, schema, kwargs))
        assert schema is G._CompactBoundaryPlan
        return selector, {"operation": kwargs["operation"]}

    monkeypatch.setattr(G, "_call_model", fake_call)
    result = G.run_segment_profile(
        transcript,
        {
            "_segment_budget_reserve": reserve,
            "_segment_telemetry": events.append,
        },
        G.FLASH_SPLIT_PROFILE,
        topic="equations",
        deadline_monotonic=time.monotonic() + 10,
    )

    assert len(calls) == 1
    _system, selector_user, schema, kwargs = calls[0]
    assert schema is G._CompactBoundaryPlan
    assert kwargs["model"] == G.config.SEGMENT_FLASH_MODEL
    assert kwargs["thinking_level"] == "low"
    assert kwargs["max_output_tokens"] == 6_000
    assert kwargs["timeout_s"] == 20.0
    assert kwargs["max_retries"] == 0
    assert kwargs["retry_status_codes"] is None
    assert kwargs["failover_model"] == G.config.SEGMENT_FLASH_FALLBACK_MODEL
    assert kwargs["operation"] == "flash_boundary_selector"
    assert kwargs["prompt_version"] == G.FLASH_SPLIT_PROFILE
    assert kwargs["budget_reserve"] is reserve
    assert (
        "GLOBAL DISTANT SENTINEL" in selector_user
    )

    assert result.classification == "green"
    assert result.accepted_count == 3
    assert {clip["title"] for clip in result.clips} == {
        "equation", "equilibrium", "stoichiometry",
    }
    assert next(
        clip for clip in result.clips if clip["title"] == "stoichiometry"
    )["boundary_confidence"] == 1.0
    assert next(
        clip for clip in result.clips if clip["title"] == "equation"
    )["boundary_confidence"] == 1.0
    assert events == []


def test_no_boundary_repair_is_attempted_after_selector_validation(monkeypatch):
    transcript = _transcript()
    selector = G._BoundaryPlan(topics=[
        _topic(3, 4, title="equation"),
        _topic(13, 13, title="stoichiometry"),
    ])
    calls = 0

    def fake_call(system, user, schema, **kwargs):
        nonlocal calls
        calls += 1
        if schema is G._CompactBoundaryPlan:
            return selector, {"operation": kwargs["operation"]}
        raise RuntimeError("repair unavailable")

    monkeypatch.setattr(G, "_call_model", fake_call)
    result = G.run_segment_profile(
        transcript,
        {},
        G.FLASH_SPLIT_PROFILE,
        topic="stoichiometry",
        deadline_monotonic=time.monotonic() + 10,
    )

    assert calls == 1
    assert result.classification == "green"
    assert result.accepted_count == 2
    assert [clip["title"] for clip in result.clips] == [
        "equation", "stoichiometry",
    ]


def test_candidates_are_validated_independently_without_model_repair(monkeypatch):
    transcript = _transcript()
    selector = G._BoundaryPlan(topics=[
        _topic(3, 4, title="equation"),
        _topic(9, 10, title="equilibrium"),
        _topic(13, 13, title="stoichiometry"),
    ])
    def fake_call(system, user, schema, **kwargs):
        assert schema is G._CompactBoundaryPlan
        return selector, {"operation": kwargs["operation"]}

    monkeypatch.setattr(G, "_call_model", fake_call)
    result = G.run_segment_profile(
        transcript,
        {},
        G.FLASH_SPLIT_PROFILE,
        topic="equations",
        deadline_monotonic=time.monotonic() + 10,
    )

    assert result.classification == "green"
    assert {clip["title"] for clip in result.clips} == {
        "equation", "equilibrium", "stoichiometry",
    }


def test_clean_fast_path_never_dispatches_boundary_repair(monkeypatch):
    transcript = _transcript()
    selector = G._BoundaryPlan(topics=[
        _topic(13, 13, title="stoichiometry"),
    ])
    calls = []

    def fake_call(system, user, schema, **kwargs):
        calls.append(schema)
        return selector, {"operation": kwargs["operation"]}

    monkeypatch.setattr(G, "_call_model", fake_call)
    result = G.run_segment_profile(
        transcript,
        {},
        G.FLASH_SPLIT_PROFILE,
        topic="chemistry",
        deadline_monotonic=time.monotonic() + 10,
    )

    assert calls == [G._CompactBoundaryPlan]
    assert result.classification == "green"
    assert result.accepted_count == 1
