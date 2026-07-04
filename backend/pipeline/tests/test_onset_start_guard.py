"""Onset START guard — the symmetric twin of the weak-END guard. Offline (no audio/LLM).
A start that opens mid-thought extends BACKWARD to the nearest in-node onset; only-weak
starts still ship, flagged weak_start_boundary. Never drops a clip. Good onsets untouched."""
from __future__ import annotations

from backend.pipeline.refine import _is_weak_start, _snap_one
from backend.pipeline.sentences import Sentence


def _sent(idx, start, end, text, terminator="."):
    return Sentence(idx=idx, text=text, start=start, end=end, terminator=terminator,
                    ends_with_period=bool(terminator), word_start_idx=idx, word_end_idx=idx,
                    align_confidence=1.0)


_SNAP = dict(allow_qe=False, min_dur=1.0, tail_pad=0.05, max_dur=500.0)


def test_is_weak_start_matches_primitive():
    assert _is_weak_start(_sent(0, 0, 4, "So the answer is magnesium bromide."))
    assert not _is_weak_start(_sent(0, 0, 4, "Now what about MgBr2?"))


def test_weak_start_extends_back_to_onset():
    sents = [
        _sent(0, 0.0, 4.0, "Now what about MgBr2?"),                 # onset
        _sent(1, 4.1, 8.0, "Well mg is magnesium and br is bromine."),
        _sent(2, 8.1, 12.0, "So the answer is magnesium bromide."),  # weak — candidate start
    ]
    clip = _snap_one({"i_start": 2, "i_end": 2, "facet": "other"}, sents, **_SNAP)
    assert clip["sentence_start_idx"] == 0                           # moved back to the question
    assert "weak_start_boundary" not in clip["warnings"]


def test_backward_extension_bounded_by_node_span():
    sents = [
        _sent(0, 0.0, 4.0, "Earlier we discussed the periodic table."),  # DIFFERENT topic
        _sent(1, 4.1, 8.0, "So the answer is magnesium bromide."),       # weak start, node begins here
    ]
    # node_span starts at 4.1 → the guard must NOT cross back into sentence 0's topic.
    clip = _snap_one({"i_start": 1, "i_end": 1, "facet": "other", "node_span": [4.1, 8.0]},
                     sents, **_SNAP)
    assert clip["sentence_start_idx"] == 1
    assert "weak_start_boundary" in clip["warnings"]                 # only weak reachable → flagged


def test_good_onset_start_unchanged():
    sents = [
        _sent(0, 0.0, 4.0, "Newton's second law relates force and acceleration."),
        _sent(1, 4.1, 8.0, "The mass is the constant of proportionality."),
    ]
    clip = _snap_one({"i_start": 0, "i_end": 1, "facet": "other"}, sents, **_SNAP)
    assert clip["sentence_start_idx"] == 0
    assert "weak_start_boundary" not in clip["warnings"]


def test_only_weak_start_still_places_flagged():
    sents = [_sent(0, 0.0, 4.0, "So the answer is magnesium bromide.")]
    clip = _snap_one({"i_start": 0, "i_end": 0, "facet": "other"}, sents, **_SNAP)
    assert clip is not None                                          # never unplaceable
    assert clip["sentence_start_idx"] == 0
    assert "weak_start_boundary" in clip["warnings"]
