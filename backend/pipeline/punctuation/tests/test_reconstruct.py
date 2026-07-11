"""Readable-output tests — the spec's worked inputs, reconstructed exactly from a good model."""
from __future__ import annotations

from backend.pipeline.punctuation.service import restore_transcript_punctuation

from .conftest import TargetProvider, make_words


def _run(text: str, target: str):
    words = make_words(text)
    return restore_transcript_punctuation(words, provider_impl=TargetProvider(target), source="")


def test_basic_statement():
    r = _run("today we are learning kinematics", "Today, we are learning kinematics.")
    assert r.status == "complete"
    assert r.readableText == "Today, we are learning kinematics."
    assert len(r.sentences) == 1


def test_definition_and_transition():
    r = _run(
        "acceleration is the rate at which velocity changes over time now lets look at an example",
        "Acceleration is the rate at which velocity changes over time. Now, lets look at an example.",
    )
    assert r.status == "complete"
    assert r.readableText == (
        "Acceleration is the rate at which velocity changes over time. "
        "Now, lets look at an example."
    )
    assert [s.text for s in r.sentences] == [
        "Acceleration is the rate at which velocity changes over time.",
        "Now, lets look at an example.",
    ]


def test_lets_is_not_rewritten():
    r = _run(
        "acceleration is the rate at which velocity changes over time now lets look at an example",
        "Acceleration is the rate at which velocity changes over time. Now, lets look at an example.",
    )
    # the underlying token must remain "lets", never "let's"
    assert [w.word for w in r.words if w.word == "lets"] == ["lets"]
    assert "let's" not in r.readableText


def test_question():
    r = _run(
        "what happens to velocity when acceleration is constant",
        "What happens to velocity when acceleration is constant?",
    )
    assert r.status == "complete"
    assert r.readableText == "What happens to velocity when acceleration is constant?"
    assert r.sentences[-1].text.endswith("?")


def test_worked_example_segmentation():
    text = ("a car starts from rest and accelerates at three meters per second squared for five "
            "seconds what is its final velocity we use v equals v zero plus a t")
    target = ("A car starts from rest and accelerates at three meters per second squared for five "
              "seconds. What is its final velocity? We use v equals v zero plus a t.")
    r = _run(text, target)
    assert r.readableText == target
    assert [s.text for s in r.sentences] == [
        "A car starts from rest and accelerates at three meters per second squared for five seconds.",
        "What is its final velocity?",
        "We use v equals v zero plus a t.",
    ]
