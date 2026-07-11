"""W25-C granularity-invariant closure: 'before'/'after' contract-element picks take the
CONTIGUOUS run of role-matching units in the seed match's topic node instead of a single
first/last unit, so a rebuild that splits one block into micro-units no longer shrinks the
closure to a sliver (the qP items-3/4 regression: the definition-anchored vectors/scalars
clip fell from a 97s span to 24.2s when 139 units became 239). Runs stay bounded by
ClosureBudget: run length by CLOSURE_MAX_EXTRA_UNITS at pick time, span/gap/unit budgets
by the existing required-inline / recommended-frontier machinery. Offline, zero LLM."""
from __future__ import annotations

from backend.adapters.base import BaseAdapter
from backend.pipeline.assemble.closure import ClosureBudget, compute_closure
from backend.pipeline.assemble.graph import Graph
from backend.pipeline.understand.models import Unit


def _u(uid: str, role: str, start: float, end: float, i: int, node: str = "c0.t1") -> Unit:
    return Unit(unit_id=uid, start=start, end=end, sentence_range=(i, i), role=role,
                node_id=node, transcript=f"sentence {i}.", summary=uid)


def _closure(anchor, units, **budget_kw):
    return compute_closure(anchor, Graph([], units), {u.unit_id: u for u in units},
                           BaseAdapter(), units, ClosureBudget(**budget_kw))


# ── the qP c0.t1 fixture: fine-grained definition block + unpacking run ──────────────────
def _qp_c0t1():
    """Modeled on qP's rebuilt c0.t1 (102-190s): two 3.3s definition micro-units, then an
    unpacking run (explanation/intuition) and a trailing claim in the SAME topic node."""
    return [
        _u("u0", "definition", 102.0, 105.3, 0),   # anchor: 'vectors' (3.3s micro-unit)
        _u("u1", "definition", 105.3, 108.6, 1),   # 'scalars' / notation (3.3s micro-unit)
        _u("u2", "explanation", 108.6, 125.0, 2),  # unpacking run …
        _u("u3", "intuition", 125.0, 140.0, 3),
        _u("u4", "explanation", 140.0, 160.0, 4),  # … run ends here
        _u("u5", "claim", 160.0, 175.0, 5),        # role break — NOT unpacking content
        _u("u6", "explanation", 175.0, 190.0, 6),  # unreachable behind the break
    ]


def test_after_run_spans_the_full_definition_block():
    units = _qp_c0t1()
    cl = _closure(units[0], units)
    # pre-W25-C the 'unpacking' pick was cand[0] alone → u0-u2, a 23s sliver; the run now
    # carries the whole same-node explanation/intuition block. This validates the
    # run-extension MECHANISM (the qP items-3 fix: the definition closure recovers the full
    # unpacking content instead of a sliver). NOTE the honest scope, confirmed by a real-
    # structure replay at review: on qP this lifts item 3 from 0.548→0.962 (crosses ≥0.60),
    # but item 4 (142-189s) is CLAIM/exception content OUTSIDE the definition contract's
    # 'after' roles — the claim role break below is exactly why. Item 4 is NOT a closure win;
    # it rides the per-topic claim cap + coverage floor (a shipped c0.t1 claim clip),
    # arbitrated at the live gate (see spec §W25-C).
    assert set(cl.unit_ids) == {"u0", "u1", "u2", "u3", "u4"}
    assert (cl.start, cl.end) == (102.0, 160.0)
    assert "u5" not in cl.unit_ids             # the claim (item-4 analog) breaks the run — a
    assert "u6" not in cl.unit_ids             # definition clip must not swallow claim content


def test_after_run_is_bounded_by_max_extra_units():
    units = _qp_c0t1()
    cl = _closure(units[0], units, max_extra_units=2)
    # the run pick itself is capped at the unit budget AND the frontier re-enforces it:
    # u1 (required term) + u2 inline; u3 stays referential — never silently unbounded
    assert set(cl.unit_ids) == {"u0", "u1", "u2"}
    assert cl.end == 125.0
    assert any(uid == "u3" for uid, _rel in cl.referential)


def test_after_run_breaks_at_topic_node_boundary():
    units = _qp_c0t1()
    units[3] = _u("u3", "intuition", 125.0, 140.0, 3, node="c0.t2")   # next topic starts
    cl = _closure(units[0], units)
    assert set(cl.unit_ids) == {"u0", "u1", "u2"}                     # run stops at u2
    assert cl.end == 125.0


# ── symmetric 'before' runs (required elements: force-inlined span-bounded) ──────────────
def test_before_run_inlines_the_whole_problem_statement():
    units = [_u("e0", "example_setup", 0.0, 10.0, 0),    # split problem statement …
             _u("e1", "example_setup", 10.0, 20.0, 1),   # … two micro-units
             _u("w2", "worked_step", 20.0, 30.0, 2),
             _u("r3", "result", 30.0, 40.0, 3)]
    cl = _closure(units[3], units)
    # pre-W25-C problem_statement took cand[-1] (e1) alone, severing half the setup
    assert set(cl.unit_ids) == {"e0", "e1", "w2", "r3"}
    assert cl.start == 0.0


def test_before_run_respects_span_budget_and_flags_truncation():
    units = [_u("e0", "example_setup", 0.0, 10.0, 0),
             _u("e1", "example_setup", 12.0, 20.0, 1),
             _u("w2", "worked_step", 20.0, 30.0, 2),
             _u("r3", "result", 30.0, 35.0, 3)]
    cl = _closure(units[3], units, max_span_s=25.0)
    # e0 would stretch the span to 35s > 25 — the existing required-inline machinery keeps
    # it referential ('prerequisite') and marks the closure truncated; e1 still fits
    assert set(cl.unit_ids) == {"e1", "w2", "r3"}
    assert cl.start == 12.0
    assert ("e0", "prerequisite") in cl.referential
    assert cl.truncated is True
