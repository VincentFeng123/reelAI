"""Speaker-change boundaries and long-pause handling."""
from __future__ import annotations

from backend.pipeline.punctuation.service import restore_transcript_punctuation

from .conftest import TargetProvider, make_words


def test_speaker_change_creates_boundary_and_propagates_speaker():
    words = make_words("he spoke she replied", speakers=["s1", "s1", "s2", "s2"])
    r = restore_transcript_punctuation(words, provider_impl=TargetProvider("He spoke. She replied."),
                                       source="")
    assert [s.text for s in r.sentences] == ["He spoke.", "She replied."]
    assert r.sentences[0].speaker == "s1"
    assert r.sentences[1].speaker == "s2"


def test_long_midsentence_pause_does_not_force_boundary():
    # a 2s gap sits between "keeps" and "rising", but the sentence continues
    words = [
        {"word": "the", "start": 0.0, "end": 0.3},
        {"word": "value", "start": 0.35, "end": 0.7},
        {"word": "keeps", "start": 0.75, "end": 1.1},
        {"word": "rising", "start": 3.10, "end": 3.5},   # long pause before this word
    ]
    r = restore_transcript_punctuation(words, provider_impl=TargetProvider("The value keeps rising."),
                                       source="")
    assert len(r.sentences) == 1
    assert r.readableText == "The value keeps rising."
