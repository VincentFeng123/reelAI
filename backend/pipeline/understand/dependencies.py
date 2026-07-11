"""Stage: dependency graph (spec §5).

Cheap, high-precision RULE edges derived deterministically from the units' own
concepts/references (requires, refers_to, answers, continues), plus reference→source_unit
resolution. An optional LLM refinement pass adds semantic relations (explains, illustrates,
contradicts, summarizes, answers, requires) that rules can't infer. Closure (Part B) relies
mainly on the rule edges, so the graph is useful even if the LLM pass is skipped.
"""
from __future__ import annotations

import re
from typing import Callable, Optional

from pydantic import BaseModel, Field

from ... import config
from .models import DependencyGraph, Edge, Unit

ProgressCb = Optional[Callable[[float, str], None]]

_RELATIONS = {"defines", "requires", "explains", "illustrates", "continues", "answers",
              "contradicts", "summarizes", "visually_depends_on", "refers_to"}


def _rule_edges(units: list[Unit]) -> list[Edge]:
    order = {u.unit_id: i for i, u in enumerate(units)}
    by_id = {u.unit_id: u for u in units}
    introducers: dict[str, list[str]] = {}
    for u in units:
        for c in u.concepts_introduced:
            introducers.setdefault(c, []).append(u.unit_id)

    def _introducer(concept: str, uid: str, *, earliest: bool) -> Optional[str]:
        """Resolve a concept to an introducing unit. ``requires`` edges point at the
        EARLIEST prior introducer (the concept's first definition, per spec §6); reference
        resolution points at the NEAREST prior mention (the most recent antecedent).
        No PRIOR introducer ⇒ None, in BOTH modes (W25-B): the old ``lst[0]`` fallback
        returned a FUTURE unit, creating forward-in-time requires/refers_to edges (qP:
        u0020/u0021 'required' the magnitude def u0035 at +124s) that poisoned closure
        referential lists, cards, and judge prerequisite inputs."""
        lst = introducers.get(concept)
        if not lst:
            return None
        prior = [x for x in lst if order[x] < order[uid]]
        if not prior:
            return None
        return prior[0] if earliest else prior[-1]

    edges: list[Edge] = []
    for u in units:
        for c in u.concepts_required:
            tgt = _introducer(c, u.unit_id, earliest=True)
            if tgt and tgt != u.unit_id:
                edges.append(Edge(source=u.unit_id, target=tgt, relation="requires",
                                  rationale=f"requires '{c}'"))
        for ref in u.references:
            c = (ref.resolves_to or "").strip().lower()
            tgt = _introducer(c, u.unit_id, earliest=False) if c else None
            if tgt and tgt != u.unit_id:
                ref.source_unit = tgt
                rel = ("answers" if u.role == "solution"
                       and by_id[tgt].role in ("practice_prompt", "example_setup") else "refers_to")
                edges.append(Edge(source=u.unit_id, target=tgt, relation=rel,
                                  rationale=(ref.text or "")[:40]))
        if u.role == "solution":
            priors = [v for v in units if order[v.unit_id] < order[u.unit_id]
                      and v.role in ("practice_prompt", "example_setup")]
            if priors:
                edges.append(Edge(source=u.unit_id, target=priors[-1].unit_id, relation="answers"))
        if u.role == "direct_answer":                       # interview: answer → its question (other speaker)
            priors = [v for v in units if order[v.unit_id] < order[u.unit_id]
                      and v.role == "question"
                      and (not u.speaker or not v.speaker or v.speaker != u.speaker)]
            if priors:
                edges.append(Edge(source=u.unit_id, target=priors[-1].unit_id, relation="answers",
                                  rationale="answers prior question"))

    for a, b in zip(units, units[1:]):
        if a.node_id and a.node_id == b.node_id and (a.speaker or None) == (b.speaker or None):
            edges.append(Edge(source=b.unit_id, target=a.unit_id, relation="continues"))
    return edges


def _visual_edges(units: list[Unit]) -> list[Edge]:
    """``visually_depends_on`` edges from resolved on-screen visuals (Phase 2).

    A unit that points at a linked ``VisualEvent`` depends on the EARLIEST unit that shows
    that same event (its introducer), so closure can pull the visual's origin into the clip.
    Empty in Phase 1 / visual-free / degraded runs (no ``visual_event_id`` links exist)."""
    owner: dict[str, str] = {}                      # visual_event_id -> earliest unit showing it
    for u in units:                                 # units are in temporal order
        for vd in u.visual_dependencies:
            eid = vd.visual_event_id
            if eid and eid not in owner:
                owner[eid] = u.unit_id
    edges: list[Edge] = []
    for u in units:
        linked: set[str] = set()
        for vd in u.visual_dependencies:
            eid = vd.visual_event_id
            if not eid:
                continue
            tgt = owner.get(eid)
            if tgt and tgt != u.unit_id and tgt not in linked:
                linked.add(tgt)
                edges.append(Edge(source=u.unit_id, target=tgt, relation="visually_depends_on",
                                  weight=0.8, rationale=f"shows {vd.kind}"[:40]))
    return edges


def _mentions(text: str, concept: str) -> bool:
    """Whole-word occurrence of a (>=4 char) concept phrase in text."""
    return len(concept) >= 4 and re.search(r"\b" + re.escape(concept) + r"\b", text) is not None


def _bridge_edges(units: list[Unit], existing: list[Edge]) -> list[Edge]:
    """Bridge-concept prerequisite edges (EDM 2018, AUC ~0.80): a concept introduced EARLIER that
    REAPPEARS in a later unit — declared in its concepts_required OR mentioned in its text — signals
    that the later unit ``requires`` the earlier one. Catches prerequisites the per-unit
    concepts_required missed (the prerequisite-gap failure). High precision: prior introductions only,
    whole-word reuse, deduped against existing edges, capped per unit."""
    order = {u.unit_id: i for i, u in enumerate(units)}
    first_intro: dict[str, str] = {}                      # concept -> earliest introducing unit
    for u in units:
        for c in u.concepts_introduced:
            c = c.strip().lower()
            if c and c not in first_intro:
                first_intro[c] = u.unit_id
    have = {(e.source, e.target, e.relation) for e in existing}
    edges: list[Edge] = []
    for u in units:
        ui = order.get(u.unit_id, 0)
        text = f"{u.summary} {u.transcript}".lower()
        own = {c.strip().lower() for c in u.concepts_introduced}
        declared = {c.strip().lower() for c in u.concepts_required}
        hits: list[tuple[int, str, str]] = []             # (introducer_order, concept, introducer_id)
        for c, src in first_intro.items():
            si = order.get(src, 0)
            if c in own or si >= ui:                       # must be introduced strictly earlier
                continue
            if c in declared or _mentions(text, c):
                hits.append((si, c, src))
        hits.sort(key=lambda h: ui - h[0])                 # nearest prior introduction first
        for _si, c, src in hits[: config.BRIDGE_MAX_PER_UNIT]:
            key = (u.unit_id, src, "requires")
            if src != u.unit_id and key not in have:
                have.add(key)
                edges.append(Edge(source=u.unit_id, target=src, relation="requires",
                                  rationale=f"bridge: reuses '{c}'"[:40], derivation="bridge"))
    return edges


# ── optional LLM refinement ──────────────────────────────────────────────────
class EdgeLLM(BaseModel):
    source: str = ""
    target: str = ""
    relation: str = ""
    rationale: str = ""


class EdgesLLM(BaseModel):
    edges: list[EdgeLLM] = Field(default_factory=list)


# NOTE (graph nutrition, Q2): prompt-only change — cached structures are unaffected and the
# persisted schema is unchanged, so SCHEMA_VERSION is deliberately NOT bumped; asking for
# 'requires' here only improves future builds. Direction matches the rule edges: an edge
# (source, target, 'requires') means the LATER source unit presupposes the EARLIER target unit.
_LLM_SYSTEM = (
    "You refine a dependency graph over a video's atomic units. Given the units (id, role, "
    "summary) and rule-derived candidate edges, ADD semantic relations that rules miss: explains, "
    "illustrates, contradicts, summarizes, answers, requires. Only use the given unit ids. An edge "
    "(source, target, relation) means 'source <relation> target'. Emit 'requires' ONLY when the "
    "source unit genuinely presupposes a concept or result introduced in the target unit (the "
    "target comes earlier — same direction as the candidate requires edges). Do not repeat "
    "candidate edges. Output only the structured result."
)


def _llm_edges(units: list[Unit], candidates: list[Edge], settings: dict) -> list[Edge]:
    from ...llm import llm_json
    ids = {u.unit_id for u in units}
    rows = "\n".join(f"{u.unit_id} [{u.role}] {u.summary[:100]}" for u in units)
    cand = "\n".join(f"{e.source} {e.relation} {e.target}" for e in candidates[:200])
    user = f"UNITS:\n{rows}\n\nCANDIDATE EDGES:\n{cand}\n\nAdd missing semantic edges."
    res = llm_json(_LLM_SYSTEM, user, EdgesLLM, temperature=0.1)
    out: list[Edge] = []
    for e in res.edges:
        if e.source in ids and e.target in ids and e.source != e.target and e.relation in _RELATIONS:
            out.append(Edge(source=e.source, target=e.target, relation=e.relation,
                            rationale=e.rationale[:80], derivation="llm"))
    return out


def build_dependency_graph(units: list[Unit], settings: dict, progress: ProgressCb = None) -> DependencyGraph:
    edges = _rule_edges(units) + _visual_edges(units)
    if config.BRIDGE_PREREQ_EDGES:
        edges += _bridge_edges(units, edges)
    if progress:
        progress(0.5, f"{len(edges)} rule edges")
    degraded: list[str] = []
    if len(units) >= 2:
        try:
            edges += _llm_edges(units, edges, settings)
        except Exception as exc:  # noqa: BLE001 — refinement is optional; its LOSS is not silent (W25-B)
            # the graph stays useful on rule edges alone, but a run that shipped without the
            # semantic pass must say so (this failed silently on qP for the whole live index).
            degraded.append(f"dependency_llm_edges: {type(exc).__name__}: {exc}"[:160])
    seen: set[tuple[str, str, str]] = set()
    deduped: list[Edge] = []
    for e in edges:
        k = (e.source, e.target, e.relation)
        if k not in seen:
            seen.add(k)
            deduped.append(e)
    # W25-B graph lint: 'requires' must point BACKWARD in time (source presupposes target).
    # Rule/bridge edges are backward by construction post-fix, so any count here is an LLM
    # edge that violated the prompt's direction contract — expect 0; nonzero is telemetry.
    order = {u.unit_id: i for i, u in enumerate(units)}
    fwd = sum(1 for e in deduped if e.relation == "requires"
              and e.source in order and e.target in order and order[e.source] < order[e.target])
    if progress:
        progress(1.0, f"{len(deduped)} edges")
    return DependencyGraph(edges=deduped, degraded=degraded, forward_requires_count=fwd)
