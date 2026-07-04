"""The reconstructed opener (problem-read / equation-introduction) is inlined into the clip
span, never demoted to a referential card — even when that overflows the soft span budget.
Uses the real Graph([], units) + BaseAdapter() fixture from test_closure_runs.py."""
from __future__ import annotations

from backend.adapters.base import BaseAdapter
from backend.pipeline.assemble.closure import compute_closure, ClosureBudget
from backend.pipeline.assemble.graph import Graph
from backend.pipeline.understand.models import Unit


def _u(uid, role, start, end, i, node="c0.t1"):
    return Unit(unit_id=uid, start=start, end=end, sentence_range=(i, i), role=role,
                node_id=node, transcript=f"sentence {i}.", summary=uid)


def _closure(units, anchor_id, budget):
    by_id = {u.unit_id: u for u in units}
    return compute_closure(by_id[anchor_id], Graph([], units), by_id, BaseAdapter(), units, budget)


def test_required_before_onset_is_inlined_even_past_span_budget():
    # result anchor whose GENERIC contract requires an example_setup BEFORE it. With a TIGHT
    # span budget the old code demotes the setup to a referential card; the onset must instead
    # be FORCE-inlined (overflow allowed).
    units = [
        _u("u0", "example_setup", 0.0, 4.0, 0),
        _u("u1", "worked_step", 4.0, 8.0, 1),
        _u("u2", "result", 8.0, 12.0, 2),   # anchor (payoff)
    ]
    by_id = {u.unit_id: u for u in units}
    res = compute_closure(by_id["u2"], Graph([], units), by_id, BaseAdapter(), units,
                          ClosureBudget(max_span_s=10.0))   # inlining u0 → 12s span > 10s budget
    assert "u0" in res.unit_ids                              # onset inlined, not carded
    assert "u0" not in [uid for uid, _ in res.referential]


# ── the LIVE soft<hard overflow window (the production window is (120, 240]) ──────────────
# Proven here with an explicit tight budget (soft 10 < hard 30). In production build_candidate
# now passes the RAW soft budget (closure_max_span_s=120) as max_span_s and the ship cap
# (max_clip_duration_s=240) as hard_max_span_s, so the window (soft, hard] is non-empty and the
# onset force-inline is not dead code (Task 6 Critical). These three assertions lock the
# semantics: onset overflows the SOFT budget up to the HARD cap; the exemption is onset-scoped;
# the HARD cap is absolute.
_SOFT_HARD = dict(max_span_s=10.0, hard_max_span_s=30.0)


def test_onset_overflowing_into_the_soft_hard_window_is_inlined():
    # example_setup onset + result anchor: including the onset makes the span 26s — INSIDE the
    # (soft=10, hard=30] window. Under the pre-feature behavior (no onset force-inline) 26 > 10
    # would demote it to a card; with the fix it is FORCE-inlined because it fits the hard cap.
    units = [
        _u("u0", "example_setup", 0.0, 4.0, 0),     # onset (required 'before')
        _u("u1", "result", 22.0, 26.0, 1),          # anchor → span 26s ∈ (10, 30]
    ]
    res = _closure(units, "u1", ClosureBudget(**_SOFT_HARD))
    assert "u0" in res.unit_ids                                   # INLINED (overflowed soft, ≤ hard)
    assert "u0" not in [uid for uid, _ in res.referential]


def test_non_onset_required_overflowing_the_soft_budget_is_carded():
    # SAME geometry, but the preceding unit is a worked_step — a required 'within' element, NOT a
    # 'before' onset. It overflows the soft budget (26 > 10) and, lacking the onset exemption,
    # must be CARDED. This proves the force-inline is scoped to the onset, not all required units.
    units = [
        _u("u0", "worked_step", 0.0, 4.0, 0),       # required 'within', but not a 'before' onset
        _u("u1", "result", 22.0, 26.0, 1),          # anchor → span 26s > soft 10
    ]
    res = _closure(units, "u1", ClosureBudget(**_SOFT_HARD))
    assert "u0" in [uid for uid, _ in res.referential]           # CARDED (no onset exemption)
    assert "u0" not in res.unit_ids


def test_onset_beyond_the_hard_cap_is_carded_not_inlined():
    # onset whose inclusion pushes the span to 36s — PAST the hard cap (30). The clip physically
    # cannot hold both, so even the onset falls back to a referential card.
    units = [
        _u("u0", "example_setup", 0.0, 4.0, 0),     # onset
        _u("u1", "result", 32.0, 36.0, 1),          # anchor → span 36s > hard 30
    ]
    res = _closure(units, "u1", ClosureBudget(**_SOFT_HARD))
    assert "u0" in [uid for uid, _ in res.referential]           # CARDED (exceeds hard cap)
    assert "u0" not in res.unit_ids
