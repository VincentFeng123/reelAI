"""Discourse-onset primitive — text-only, offline. Decides whether a sentence used as a
clip's FIRST line drops the viewer mid-thought. A presentational ``so`` question can open a
topic, while an adversative question still depends on the preceding contrast."""
from __future__ import annotations

from backend.pipeline.discourse import (
    _has_unresolved_opening_back_reference,
    is_onset,
    opens_mid_thought,
)

# Real bad openers pulled from the shipped corpus — MUST be flagged weak.
def test_answer_first_is_weak():
    assert opens_mid_thought("So the answer is magnesium bromide.")

def test_and_then_continuation_is_weak():
    assert opens_mid_thought("And then mg which stands for magnesium,")

def test_however_continuation_is_weak():
    assert opens_mid_thought("However we do have a subscript next to O and it's a two.")


def test_dialogue_reply_and_mid_list_ordinal_are_weak():
    assert opens_mid_thought("Yeah- They like it.")
    assert opens_mid_thought("Yes, that's correct.")
    assert opens_mid_thought("Exactly, that's the point.")
    assert opens_mid_thought("Second, it needs to exhibit traits that affect survival.")
    assert opens_mid_thought("All right. To solve this, first define comparative advantage.")


def test_opening_purpose_and_reformulation_references_need_prior_context():
    assert opens_mid_thought(
        "And to do this, I'm going to use the chain rule."
    )
    assert opens_mid_thought(
        "Another way of writing it is the derivative of h with respect to x."
    )


def test_ordinal_and_reply_words_are_valid_when_they_are_subjects():
    assert is_onset("Second law relates force, mass, and acceleration.")
    assert is_onset("No force acts on the object after release.")
    assert is_onset("Right triangles satisfy the Pythagorean theorem.")
    assert is_onset("Exactly one solution exists for this equation.")
    assert is_onset("Yes votes determine the result of the referendum.")
    assert is_onset("All right triangles satisfy the Pythagorean theorem.")


def test_back_to_prior_context_is_weak():
    assert opens_mid_thought("Back to the cycle!")


def test_explicit_moving_along_transition_is_self_contained_framing():
    assert not opens_mid_thought("Because now we're moving along to the Calvin Cycle!")


def test_later_framing_sentence_does_not_rescue_a_mid_thought_opening():
    assert opens_mid_thought(
        "And at the same time, some of these traits can be found in non-living "
        "things, too. Viruses complicate the definition of life. Let's head "
        "to the Thought Bubble."
    )


def test_late_framing_does_not_rescue_a_long_unpunctuated_cue():
    assert opens_mid_thought(
        "And at the same time some of these traits can be found in non living "
        "things too which makes the definition complicated before we eventually "
        "lets head to the thought bubble"
    )
    assert is_onset("And let's define the traits shared by living organisms.")
    assert is_onset(
        "And to understand how cells release usable energy from glucose in the "
        "presence of oxygen lets examine cellular respiration step by step"
    )


def test_late_question_mark_does_not_rescue_a_long_continuation_cue():
    assert opens_mid_thought(
        "And at the same time these traits occur in many non living systems "
        "which makes the boundary difficult and requires several more examples "
        "before scientists can ask whether viruses count as life?"
    )
    assert is_onset("So what is life?")


def test_adversative_question_needs_context_but_bare_question_is_standalone():
    assert opens_mid_thought("But what even is life?")
    assert opens_mid_thought("But what about prokaryotic cells?")
    assert is_onset("What even is life?")
    assert is_onset("So what even is life?")

def test_dangling_anaphor_is_weak():
    assert opens_mid_thought("This is why the reaction proceeds so quickly.")
    assert opens_mid_thought("That gives us the final concentration.")


def test_leading_demonstrative_noun_phrase_needs_its_antecedent():
    assert opens_mid_thought("Those groups of cells are called tissues.")
    assert opens_mid_thought("This example demonstrates genetic drift.")


def test_opening_aside_marker_is_not_a_cold_viewer_onset():
    assert opens_mid_thought(
        "Oh yeah, by the way, multicellular organisms also reproduce."
    )
    assert opens_mid_thought("By the way, chloroplasts contain chlorophyll.")


def test_question_with_missing_antecedent_is_weak():
    assert opens_mid_thought("Ok. But like, how does the cell do that?")
    assert opens_mid_thought("How does that work?")
    assert opens_mid_thought(
        "And then you have to ask the question, is it renormalizable?"
    )


def test_compound_question_can_establish_its_own_pronoun_antecedent():
    assert is_onset(
        "What properties do metals have, and why do they conduct electricity?"
    )
    assert is_onset(
        "The enzyme binds DNA. How does it recognize the target sequence?"
    )
    assert is_onset("Can you explain osmosis? How does it work?")
    assert opens_mid_thought("I have one question. Why does it matter?")
    assert opens_mid_thought("We finished the introduction. How does it work?")
    assert opens_mid_thought("What does it change?")

def test_context_dependent_np_is_weak():
    assert opens_mid_thought("The answer is fifteen newtons.")
    assert opens_mid_thought("The previous equation tells us the velocity.")


def test_explicit_and_technical_back_references_are_weak():
    assert opens_mid_thought(
        "X could not depend on these parameters in an exactly analogous way as before."
    )
    assert opens_mid_thought("The calculation proceeds as discussed earlier.")


def test_locally_introduced_technical_demonstrative_is_an_onset():
    assert is_onset(
        "The model has two parameters, and these parameters control the fit."
    )
    assert is_onset(
        "The model defines one approach, and these approaches share its assumptions."
    )


def test_locally_resolved_explicit_back_reference_is_an_onset():
    assert is_onset(
        "We first define the baseline rule. As before, the baseline rule controls "
        "the calculation."
    )


def test_explicit_reference_trigger_verbs_do_not_count_as_antecedents():
    assert _has_unresolved_opening_back_reference(
        "The calculation proceeds as discussed earlier.",
        prior_text="We discussed the course schedule and office hours.",
    )
    assert _has_unresolved_opening_back_reference(
        "The coupling was defined previously.",
        prior_text="We defined the grading policy before class.",
    )


def test_single_shared_noun_does_not_resolve_an_explicit_reference():
    assert _has_unresolved_opening_back_reference(
        "The calculation proceeds as discussed earlier.",
        prior_text="A homework exercise contains one calculation.",
    )
    assert _has_unresolved_opening_back_reference(
        "The coupling was defined previously.",
        prior_text="Quantum chromodynamics contains a coupling constant.",
    )
    assert _has_unresolved_opening_back_reference(
        "The calculations proceed as discussed earlier.",
        prior_text="These calculations are difficult.",
    )
    assert _has_unresolved_opening_back_reference(
        "The couplings were defined previously.",
        prior_text="This chapter lists several couplings.",
    )


def test_punctuation_free_local_explicit_reference_has_real_antecedent():
    assert not _has_unresolved_opening_back_reference(
        "We first define the baseline rule and as before the baseline rule "
        "controls the calculation"
    )


def test_contextless_precision_reformulation_is_weak():
    assert opens_mid_thought(
        "To be exact, the signals travel from the retina to the visual cortex."
    )
    assert opens_mid_thought(
        "More precisely, these signals encode contrast rather than brightness."
    )


def test_precision_phrase_with_self_contained_subject_or_local_context_is_an_onset():
    assert is_onset("To be exact, one parsec equals about 3.26 light-years.")
    assert is_onset(
        "Retinal ganglion cells emit action potentials. To be exact, the signals "
        "encode changes in local contrast."
    )

def test_mid_clause_fragment_is_weak():
    assert opens_mid_thought("writing oxygen we're going to write a two")  # lowercase mid-clause


def test_opening_quotes_and_brackets_do_not_hide_a_lowercase_fragment():
    assert opens_mid_thought("“chloroplasts” inside of which photosynthesis occurs.")
    assert opens_mid_thought("[chloroplasts] inside of which photosynthesis occurs.")


def test_elliptical_instruction_and_action_reference_are_weak():
    assert opens_mid_thought("Little below the line, pull it through.")
    assert opens_mid_thought("I can just do that.")


def test_embedded_anaphor_and_bare_predicate_require_prior_context():
    assert opens_mid_thought("You know what else it can do: one more function.")
    assert opens_mid_thought("The next enzyme, as if there isn't enough enzymes.")
    assert opens_mid_thought("Have to remove those primers.")


def test_plural_noun_subjects_are_complete_openings():
    assert is_onset("Causes include fiscal crisis and inequality.")
    assert is_onset("Uses include generating electricity and heating water.")
    assert is_onset("Returns diminish as more labor is added to fixed capital.")
    assert is_onset("Causes can be identified by examining the evidence.")
    assert is_onset("Causes are classified by using their mechanisms.")


def test_deictic_worked_example_openers_are_weak_but_existentials_are_onsets():
    assert opens_mid_thought("What do I need to add here?")
    assert opens_mid_thought("Let's say here we had our three-prime end.")
    assert opens_mid_thought("Five-prime end here, the new strand would continue.")
    assert is_onset("Here is the proofreading rule.")
    assert is_onset("Two proofreading mechanisms operate independently.")


def test_dummy_and_existential_pronouns_do_not_require_prior_context():
    assert is_onset("DNA makes it possible to transmit hereditary information.")
    assert is_onset("When it rains, roads get slippery.")
    assert is_onset("It is important to distinguish correlation from causation.")
    assert is_onset("What happens when there is a mismatch?")
    assert is_onset("Why are there seasons?")


def test_referential_objects_and_deictic_locations_still_require_context():
    assert opens_mid_thought("We can remove them with DNA polymerase I.")
    assert opens_mid_thought("To solve it, factor the polynomial.")
    assert opens_mid_thought("What should I place there?")


def test_plural_noun_subjects_are_not_mistaken_for_bare_predicates():
    assert is_onset(
        "Causes of the French Revolution include fiscal crisis and inequality."
    )
    assert is_onset("Returns to scale describe how output changes with inputs.")
    assert is_onset("Means testing determines eligibility for benefits.")
    assert is_onset("Cuts in interest rates stimulate aggregate demand.")
    assert is_onset("Starts and stops in DNA regulate transcription.")


def test_complete_little_below_statement_is_an_onset():
    assert is_onset("A little below the boiling point, water remains liquid.")

# Real GOOD openers — MUST be treated as onsets (the critical false-positives).
def test_so_lets_start_is_onset():
    assert is_onset("So let's start with KI. How can we name this compound?")

def test_now_what_about_is_onset():
    assert is_onset("Now what about MgBr2?")

def test_bare_question_is_onset():
    assert is_onset("What is the charge on chlorine?")

def test_declarative_topic_sentence_is_onset():
    assert is_onset("Newton's second law relates force and acceleration.")

def test_hortative_framing_is_onset():
    assert is_onset("Let's consider a block sliding down a ramp.")
    assert is_onset("Suppose we have two moles of hydrogen.")

def test_empty_is_weak():
    assert opens_mid_thought("")
