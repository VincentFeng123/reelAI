"""Discourse-onset primitive — text-only, offline. Decides whether a sentence used as a
clip's FIRST line drops the viewer mid-thought. A leading continuation marker is NOT weak
when the sentence is self-contained framing or a question (the 'so' disambiguation)."""
from __future__ import annotations

from backend.pipeline.discourse import opens_mid_thought, is_onset

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

def test_dangling_anaphor_is_weak():
    assert opens_mid_thought("This is why the reaction proceeds so quickly.")
    assert opens_mid_thought("That gives us the final concentration.")


def test_question_with_missing_antecedent_is_weak():
    assert opens_mid_thought("Ok. But like, how does the cell do that?")
    assert opens_mid_thought("How does that work?")

def test_context_dependent_np_is_weak():
    assert opens_mid_thought("The answer is fifteen newtons.")
    assert opens_mid_thought("The previous equation tells us the velocity.")

def test_mid_clause_fragment_is_weak():
    assert opens_mid_thought("writing oxygen we're going to write a two")  # lowercase mid-clause

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
