"""Silence-aware start/end placement (Tasks 5-6). Offline: energy_fn=None → pure gap math."""
from __future__ import annotations

from backend.pipeline.boundary import _pick_start
from backend.pipeline.sentences import Sentence

LEAD, TAIL, GAP = 0.06, 0.15, 0.12


def _s(i, a, b, term="."):
    return Sentence(idx=i, text=f"s{i}", start=a, end=b, terminator=term,
                    ends_with_period=(term in ".?!"), word_start_idx=i, word_end_idx=i,
                    align_confidence=1.0)


def test_start_cuts_into_gap_never_into_prev_word():
    # prev ends 9.50, S starts 10.00 → 0.50 s gap (> 2*lead) → cut at S.start-lead = 9.94
    sents = [_s(0, 8.0, 9.50), _s(1, 10.00, 12.0)]
    p = _pick_start(sents, rough=10.0, pad=10.0, lead_pad=LEAD, gap_min=GAP, energy_fn=None)
    assert abs(p.time - 9.94) < 1e-6
    assert 9.50 < p.time < 10.00        # strictly inside the gap
    assert p.satisfied


def test_start_small_gap_uses_midpoint():
    # prev ends 9.96, S starts 10.00 → 0.04 s gap (< 2*lead) → midpoint 9.98
    sents = [_s(0, 8.0, 9.96), _s(1, 10.00, 12.0)]
    p = _pick_start(sents, rough=10.0, pad=10.0, lead_pad=LEAD, gap_min=GAP, energy_fn=None)
    assert abs(p.time - 9.98) < 1e-6
    assert 9.96 < p.time < 10.00


def test_start_prev_absent_keep_first_uses_lead_pad():
    sents = [_s(0, 0.0, 3.0), _s(1, 3.1, 6.0)]
    p = _pick_start(sents, rough=0.02, pad=10.0, keep_first=True, lead_pad=LEAD, gap_min=GAP)
    assert p.time == 0.0                # max(0, 0.0 - lead) clamped to 0
    assert p.satisfied


def test_start_prev_unseen_triggers_growth():
    # window did NOT start at t=0; chosen S is sents[1] but its prev (sents[0]) is a FRAGMENT,
    # so its start is a real onset with no visible preceding word → grow backward (unsatisfied).
    # Here sents[0] is dropped as a fragment (not keep_first); S = sents[1]; prev index 0 is the
    # fragment, whose .end is far (< real gap) → treat as unseen when it is the window's first sent.
    sents = [_s(0, 90.0, 91.0, term=""), _s(1, 100.0, 103.0)]
    p = _pick_start(sents, rough=100.0, pad=10.0, keep_first=False, lead_pad=LEAD, gap_min=GAP)
    assert not p.satisfied
    assert "start_prev_unseen" in p.flags


def test_start_direction_safe_when_only_late_candidate():
    sents = [_s(0, 43.0, 44.0), _s(1, 58.0, 60.0)]      # only start 58 > rough+1
    p = _pick_start(sents, rough=45.0, pad=10.0, lead_pad=LEAD, gap_min=GAP)
    assert p.time == 45.0 and p.satisfied
