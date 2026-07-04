"""Context closure (spec §5–6): recursively include an anchor's required context.

Two kinds of context are treated differently:

- The anchor's OWN completeness-contract elements (problem statement → steps → result …)
  define the extent of the instructional event. These are inlined span-bounded, ignoring
  temporal gaps — the "gap" between a result and its problem statement is filled by the
  reasoning steps, and the contiguous sentence range picks those up automatically.
- DISTANT prerequisites reached via the dependency graph (requires/refers_to/…) use a
  near/far split: near ones are inlined; far ones are recorded as REFERENTIAL (for the
  context card / a "watch first" hint) rather than dragged in, keeping the clip short.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ... import config
from ..understand.models import Unit


@dataclass
class ClosureBudget:
    max_extra_units: int = config.CLOSURE_MAX_EXTRA_UNITS
    max_gap_s: float = config.CLOSURE_MAX_GAP_S
    max_span_s: float = config.CLOSURE_MAX_SPAN_S
    # hard_max_span_s: the absolute ship cap (max_clip_duration_s). When set (> 0), the
    # onset force-inline is gated: we overflow the SOFT max_span_s budget but never exceed
    # the hard cap — a clip physically cannot contain both onset and anchor if the combined
    # span > max_clip_duration_s, so the onset falls back to referential. 0 means no hard
    # cap (unit tests that call compute_closure directly with a tight max_span_s; the
    # force-inline is unconditional in that case).
    hard_max_span_s: float = 0.0


@dataclass
class ClosureResult:
    anchor_id: str
    unit_ids: list[str]                        # inlined, time-ordered
    start: float
    end: float
    referential: list[tuple[str, str]] = field(default_factory=list)
    truncated: bool = False


_GRAPH_NEED_RELATIONS = ("answers", "refers_to", "requires", "visually_depends_on")
_REL_WEIGHT = {
    "answers": 1.0, "refers_to": 0.9, "requires": 0.85, "visually_depends_on": 0.8,
    "defines": 0.8, "continues": 0.6, "explains": 0.5,
    "contract_required": 0.95, "contract_recommended": 0.5,
}


def _gap_to_span(u: Unit, span: list[float]) -> float:
    if u.end < span[0]:
        return span[0] - u.end
    if u.start > span[1]:
        return u.start - span[1]
    return 0.0


def _graph_needs(uid: str, graph) -> list[tuple[str, str, float]]:
    return [(e.target, e.relation, _REL_WEIGHT.get(e.relation, 0.5))
            for e in graph.needs(uid, _GRAPH_NEED_RELATIONS)]


def _element_run(units: list[Unit], start_i: int, want: set[str], step: int,
                 max_units: int) -> list[Unit]:
    """W25-C granularity-invariant element pick: from the seed match, take the CONTIGUOUS
    run of units matching the element roles AND sharing the seed's topic node, walking
    ``step`` (+1 after / -1 before). The qP rebuild split the vectors/scalars definition
    block into 3.3s micro-units; picking cand[0] alone made the closure a 24.2s sliver of
    an ~88s event (gold item 3 at 0.548, item 4 at 0.000). A differing role OR node breaks
    the run; length is bounded by CLOSURE_MAX_EXTRA_UNITS here and the span budget where
    the picks are consumed (required inline / recommended frontier — existing machinery)."""
    seed = units[start_i]
    run = [seed]
    j = start_i + step
    while 0 <= j < len(units) and len(run) < max_units:
        u = units[j]
        if u.role not in want or u.node_id != seed.node_id:
            break
        run.append(u)
        j += step
    return run


def _required_before_ids(anchor: Unit, adapter, units: list[Unit], order: dict[str, int]) -> set[str]:
    """Unit ids satisfying a REQUIRED, position='before' contract element of the anchor — the
    onset (problem-read / equation-introduction) that the clip must open with."""
    contract = adapter.contract_for(anchor.role)
    if not contract:
        return set()
    ai = order.get(anchor.unit_id, 0)
    ids: set[str] = set()
    for el in contract.elements:
        if el.necessity != "required" or el.position != "before":
            continue
        want = set(el.roles)
        cand = [u for u in units if order[u.unit_id] < ai and u.role in want]
        if cand:
            ids.add(cand[-1].unit_id)          # nearest preceding onset unit
    return ids


def _contract_needs(anchor: Unit, adapter, units: list[Unit], order: dict[str, int],
                    budget: ClosureBudget) -> tuple[list[str], list[tuple[str, str, float]]]:
    """Return (required_element_unit_ids, recommended_needs). Required elements define the
    event extent (force-inlined span-bounded); recommended ones go through the two-mode walk."""
    contract = adapter.contract_for(anchor.role)
    if not contract:
        return [], []
    ai = order.get(anchor.unit_id, 0)
    required: list[str] = []
    recommended: list[tuple[str, str, float]] = []
    for el in contract.elements:
        if el.necessity == "optional":
            continue
        want = set(el.roles)
        if el.position == "before":
            cand = [u for u in units if order[u.unit_id] < ai and u.role in want]
            picks = (_element_run(units, order[cand[-1].unit_id], want, -1,
                                  budget.max_extra_units) if cand else [])
        elif el.position == "after":
            cand = [u for u in units if order[u.unit_id] > ai and u.role in want]
            picks = (_element_run(units, order[cand[0].unit_id], want, +1,
                                  budget.max_extra_units) if cand else [])
        else:  # within — pull matching units in a small window around the anchor
            picks = [u for u in units if u.role in want and abs(order[u.unit_id] - ai) <= 5]
        for p in picks:
            if p.unit_id == anchor.unit_id:
                continue
            if el.necessity == "required":
                required.append(p.unit_id)
            else:
                recommended.append((p.unit_id, f"contract:{el.key}", _REL_WEIGHT["contract_recommended"]))
    return required, recommended


def compute_closure(anchor: Unit, graph, units_by_id: dict[str, Unit], adapter,
                    units: list[Unit], budget: ClosureBudget) -> ClosureResult:
    order = {u.unit_id: i for i, u in enumerate(units)}
    inline = {anchor.unit_id}
    span = [anchor.start, anchor.end]
    referential: list[tuple[str, str]] = []
    truncated = False

    # 1. contract-required elements → inline. A required *before* onset (problem_statement /
    #    example_setup / practice_prompt / setup) is FORCE-inlined even past the span budget:
    #    the clip must open with it (discourse-onset invariant, overflow allowed). Other
    #    required elements keep the budget check.
    required_ids, recommended = _contract_needs(anchor, adapter, units, order, budget)
    onset_ids = _required_before_ids(anchor, adapter, units, order)   # new helper, see below
    for uid in sorted(set(required_ids), key=lambda x: order.get(x, 0)):
        u = units_by_id.get(uid)
        if not u or uid in inline:
            continue
        new_span = [min(span[0], u.start), max(span[1], u.end)]
        new_dur = new_span[1] - new_span[0]
        # Force-inline the onset (required before) even past the soft span budget, as long as
        # the combined span fits within the hard ship cap (hard_max_span_s). When no hard cap
        # is set (0), force-inline unconditionally (unit-test / direct-closure-call path).
        force = (uid in onset_ids and
                 (budget.hard_max_span_s == 0 or new_dur <= budget.hard_max_span_s))
        if force or new_dur <= budget.max_span_s:
            inline.add(uid)
            span = new_span
        else:
            referential.append((uid, "prerequisite"))
            truncated = True

    # 2. graph prerequisites + recommended elements → near/far two-mode ----------
    frontier = _graph_needs(anchor.unit_id, graph) + recommended
    for uid in list(inline):
        if uid != anchor.unit_id:
            frontier += _graph_needs(uid, graph)
    processed: set[str] = set()
    while frontier:
        frontier.sort(key=lambda nd: (-nd[2], _gap_to_span(units_by_id[nd[0]], span))
                      if nd[0] in units_by_id else (0.0, 1e9))
        uid, relation, weight = frontier.pop(0)
        if uid in inline or uid in processed:
            continue
        processed.add(uid)
        u = units_by_id.get(uid)
        if not u:
            continue
        gap = _gap_to_span(u, span)
        new_span = [min(span[0], u.start), max(span[1], u.end)]
        fits = (len(inline) - 1 < budget.max_extra_units
                and gap <= budget.max_gap_s
                and (new_span[1] - new_span[0]) <= budget.max_span_s)
        if fits:
            inline.add(uid)
            span = new_span
            frontier.extend(_graph_needs(uid, graph))
        else:
            referential.append((uid, relation))
            if weight >= _REL_WEIGHT["requires"]:
                truncated = True

    ordered = sorted(inline, key=lambda x: (units_by_id[x].start, order.get(x, 0)))
    return ClosureResult(anchor.unit_id, ordered, span[0], span[1], referential, truncated)
