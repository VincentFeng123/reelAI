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


def test_plan_rejects_live_biology_cue_despite_later_framing_sentence() -> None:
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

    assert report.clips == []
    assert report.rejected_reasons == ["proposal_0:unresolved_weak_start"]


def test_plan_rejects_long_unpunctuated_biology_cue_with_late_framing() -> None:
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

    assert report.clips == []
    assert report.rejected_reasons == ["proposal_0:unresolved_weak_start"]


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


def test_boundary_plan_rejects_a_quote_removed_with_a_forward_setup() -> None:
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

    assert report.clips == []
    assert report.rejected_reasons == ["proposal_0:ungrounded_boundary_quote"]


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
        assert schema is G._BoundaryPlan
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
    assert schema is G._BoundaryPlan
    assert kwargs["model"] == G.config.SEGMENT_FLASH_MODEL
    assert kwargs["thinking_level"] == "low"
    assert kwargs["max_output_tokens"] == 12_288
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
        if schema is G._BoundaryPlan:
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
        assert schema is G._BoundaryPlan
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

    assert calls == [G._BoundaryPlan]
    assert result.classification == "green"
    assert result.accepted_count == 1
