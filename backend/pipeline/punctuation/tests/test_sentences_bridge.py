"""The bridge into the legacy Sentence model (sentences_from_punctuation), incl. the abbrev guard."""
from __future__ import annotations

from backend.pipeline.sentences import sentences_from_punctuation
from backend.pipeline.punctuation.service import restore_transcript_punctuation

from .conftest import TargetProvider, make_words


def _bridge(text: str, target: str):
    words = make_words(text)
    r = restore_transcript_punctuation(words, provider_impl=TargetProvider(target), source="")
    return words, sentences_from_punctuation(r, words)


def test_period_sentence_maps_exactly():
    words, sents = _bridge("acceleration is constant", "Acceleration is constant.")
    assert len(sents) == 1
    s = sents[0]
    assert s.text == "Acceleration is constant."
    assert s.terminator == "."
    assert s.ends_with_period is True
    assert (s.word_start_idx, s.word_end_idx) == (0, 2)
    assert s.start == words[0]["start"] and s.end == words[2]["end"]


def test_question_terminator():
    _, sents = _bridge("is it constant", "Is it constant?")
    assert sents[-1].terminator == "?"
    # a '?' (or '!') is a real sentence ending, so it counts as a valid end
    assert sents[-1].ends_with_period is True
    assert sents[-1].is_valid_end() is True


def test_legacy_path_also_counts_question_and_exclaim():
    # the legacy build_sentence_index path must agree: '?'/'!' are valid ends
    from backend.pipeline.sentences import build_sentence_index
    words = make_words("is it constant stop")
    transcript = {
        "words": words,
        "segments": [
            {"start": words[0]["start"], "end": words[2]["end"], "text": "Is it constant?"},
            {"start": words[3]["start"], "end": words[3]["end"], "text": "Stop!"},
        ],
    }
    sents = build_sentence_index(transcript)
    assert [s.terminator for s in sents] == ["?", "!"]
    assert all(s.ends_with_period for s in sents)


def test_abbreviation_period_is_guarded():
    # an LLM period after "Mr" must NOT be treated as a clip-boundary period
    _, sents = _bridge("the title is mr", "The title is Mr.")
    assert sents[-1].text == "The title is Mr."
    assert sents[-1].terminator == ""
    assert sents[-1].ends_with_period is False
