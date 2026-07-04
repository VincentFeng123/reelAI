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
