# backend/eval/tests/test_opening_onset_metric.py
from __future__ import annotations

from backend.eval.metrics import opening_onset_rate
from backend.pipeline.sentences import Sentence


def _sent(idx, text):
    return Sentence(idx=idx, text=text, start=float(idx), end=float(idx) + 1.0,
                    terminator=".", ends_with_period=True, word_start_idx=idx,
                    word_end_idx=idx, align_confidence=1.0)


def test_opening_onset_rate_counts_only_good_openers():
    sentences = [
        _sent(0, "Now what about MgBr2?"),          # onset
        _sent(1, "So the answer is magnesium bromide."),  # weak
        _sent(2, "Newton's second law relates force and acceleration."),  # onset
    ]
    specs = [
        {"sentence_start_idx": 0},
        {"sentence_start_idx": 1},
        {"sentence_start_idx": 2},
    ]
    assert opening_onset_rate(specs, sentences) == 2 / 3


def test_opening_onset_rate_empty_is_zero():
    assert opening_onset_rate([], []) == 0.0
