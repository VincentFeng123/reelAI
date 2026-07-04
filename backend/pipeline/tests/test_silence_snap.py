"""Silence-aware start/end placement (Tasks 5-6). Offline: energy_fn=None → pure gap math."""
from __future__ import annotations

from backend.pipeline.boundary import _pick_end, _pick_start
from backend.pipeline.sentences import Sentence

LEAD, TAIL, GAP = 0.06, 0.15, 0.12


def _s(i, a, b, term="."):
    # Use membership check (not substring) so term="" → ends_with_period=False, matching real Sentence.
    return Sentence(idx=i, text=f"s{i}", start=a, end=b, terminator=term,
                    ends_with_period=(term in (".", "?", "!")), word_start_idx=i, word_end_idx=i,
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


# ── END placement (Task 6) ────────────────────────────────────────────────────

def test_end_cuts_into_gap_never_into_next_word():
    # E ends 46.20, next starts 46.70 → 0.50 s gap (> 2*tail) → cut at E.end+tail = 46.35
    sents = [_s(0, 40.0, 44.5), _s(1, 44.6, 46.20), _s(2, 46.70, 49.0)]
    p = _pick_end(sents, rough=45.0, pad=10.0, allow_qe=False,
                  tail_pad=TAIL, gap_min=GAP, end_extend_max=8.0, energy_fn=None)
    assert abs(p.time - 46.35) < 1e-6
    assert 46.20 < p.time < 46.70 and p.satisfied
    assert "tight_end_no_gap" not in p.flags


def test_end_small_gap_uses_midpoint():
    # E ends 46.20, next starts 46.26 → 0.06 s gap (< 2*tail but >= gap_min? 0.06<0.12) → tight+flag
    sents = [_s(0, 44.6, 46.20), _s(1, 46.26, 49.0)]
    p = _pick_end(sents, rough=45.0, pad=10.0, allow_qe=False,
                  tail_pad=TAIL, gap_min=GAP, end_extend_max=8.0, energy_fn=None)
    # 0.06 < gap_min and no later gap within budget → tight cut at E, flagged
    assert 46.20 < p.time <= 46.23 and p.satisfied
    assert "tight_end_no_gap" in p.flags


def test_end_hybrid_nudges_to_next_gap_within_budget():
    # E @47.0 has NO gap (next @47.0); a later end @50.0 HAS a 0.5s gap, within 8s budget → advance
    sents = [_s(0, 44.0, 47.0), _s(1, 47.0, 50.0), _s(2, 50.5, 53.0)]
    p = _pick_end(sents, rough=46.5, pad=10.0, allow_qe=False,
                  tail_pad=TAIL, gap_min=GAP, end_extend_max=8.0, energy_fn=None)
    assert 50.0 < p.time < 50.5 and p.satisfied
    assert "end_extended" in p.flags


def test_end_hybrid_beyond_budget_keeps_tight():
    # E @47.0 no gap; next clean gap only at 60.0 (> 8s budget from rough 46.5) → tight at E, flagged
    sents = [_s(0, 44.0, 47.0), _s(1, 47.0, 60.0), _s(2, 60.6, 63.0)]
    p = _pick_end(sents, rough=46.5, pad=10.0, allow_qe=False,
                  tail_pad=TAIL, gap_min=GAP, end_extend_max=8.0, energy_fn=None)
    assert 47.0 <= p.time <= 47.16 and p.satisfied
    assert "tight_end_no_gap" in p.flags


def test_end_last_sentence_gap_unmeasurable_grows():
    sents = [_s(0, 44.6, 46.2)]           # E is the only/last sentence → no next → grow
    p = _pick_end(sents, rough=45.0, pad=10.0, allow_qe=False,
                  tail_pad=TAIL, gap_min=GAP, end_extend_max=8.0, energy_fn=None)
    assert not p.satisfied


def test_end_no_valid_end_grows():
    sents = [_s(0, 44.6, 46.2, term=""), _s(1, 46.3, 48.0, term="")]   # no period anywhere
    p = _pick_end(sents, rough=45.0, pad=10.0, allow_qe=False,
                  tail_pad=TAIL, gap_min=GAP, end_extend_max=8.0, energy_fn=None)
    assert not p.satisfied
