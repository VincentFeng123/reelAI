"""Token and timestamp preservation — the core invariant of the whole stage."""
from __future__ import annotations

from backend.pipeline.punctuation.service import restore_transcript_punctuation

from .conftest import TargetProvider, make_words, strip_readable

TEXT = ("a car starts from rest and accelerates at three meters per second squared for five "
        "seconds what is its final velocity we use v equals v zero plus a t")
TARGET = ("A car starts from rest and accelerates at three meters per second squared for five "
          "seconds. What is its final velocity? We use v equals v zero plus a t.")


def _run():
    words = make_words(TEXT)
    r = restore_transcript_punctuation(words, provider_impl=TargetProvider(TARGET), source="")
    return words, r


def test_original_words_unchanged():
    words, r = _run()
    assert [pw.word for pw in r.words] == [w["word"] for w in words]


def test_strip_reproduces_original_tokens():
    words, r = _run()
    # removing capitalization and punctuation from the restored transcript yields the input tokens
    assert strip_readable(r.readableText).split() == [w["word"].lower() for w in words]


def test_timestamps_unchanged():
    words, r = _run()
    for pw, w in zip(r.words, words):
        assert pw.start == w["start"]
        assert pw.end == w["end"]


def test_sentence_bounds_come_from_token_times():
    words, r = _run()
    for s in r.sentences:
        first = int(s.tokenIds[0][1:])
        last = int(s.tokenIds[-1][1:])
        assert s.start == words[first]["start"]
        assert s.end == words[last]["end"]
