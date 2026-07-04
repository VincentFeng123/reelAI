"""W25-B drift telemetry + graph hygiene (offline, mocked llm_json).

Pins (a) the 'boundary_clamped' Unit.warnings emission when the units.py non-overlap
clamp actually MOVES a boundary (previously a silent cascade that shifted every later
unit in the topic), (b) drift_stats + the structure-level 'unit_drift' degraded note,
(c) _introducer returning None when no PRIOR introducer exists — the old lst[0]
fallback manufactured forward-in-time requires/refers_to edges (qP: u0020/u0021
'required' u0035 at +124s), (d) the LLM edge-pass failure surfacing as a degraded
note instead of the bare except, and (e) the forward-requires graph lint.
"""
from __future__ import annotations

import backend.llm as llm_mod
from backend.adapters.detect import DetectionResult
from backend.adapters.lecture import LectureAdapter
from backend.pipeline.understand import build as build_mod
from backend.pipeline.understand.dependencies import EdgeLLM, EdgesLLM, build_dependency_graph
from backend.pipeline.understand.models import (
    ContentMap, ContentNode, DependencyGraph, Reference, Unit,
)
from backend.pipeline.understand.units import UnitLLM, UnitsLLM, drift_stats, extract_units

from .conftest import make_sents


# ── fixtures ─────────────────────────────────────────────────────────────────
def _cm(n: int) -> ContentMap:
    return ContentMap(root_id="video", nodes=[
        ContentNode(node_id="video", level="video", sentence_range=(0, n - 1)),
        ContentNode(node_id="ch1.t1", level="topic", title="T", sentence_range=(0, n - 1)),
    ])


def _extract(monkeypatch, llm_units: list[UnitLLM], n: int = 6) -> list[Unit]:
    def fake(system, user, schema, **kw):
        assert schema is UnitsLLM
        return UnitsLLM(units=llm_units)
    monkeypatch.setattr(llm_mod, "llm_json", fake)
    return extract_units(make_sents(n), _cm(n), LectureAdapter(), settings={})


def _unit(uid: str, i: int, **kw) -> Unit:
    return Unit(unit_id=uid, start=i * 10.0, end=i * 10.0 + 5.0,
                sentence_range=kw.pop("sentence_range", (i, i)), node_id="ch1.t1", topic="T",
                summary=kw.pop("summary", f"unit {uid}"),
                transcript=kw.pop("transcript", f"transcript of {uid}"), **kw)


# ── (a) boundary_clamped warning emission ────────────────────────────────────
def test_clamp_warning_emitted_when_cursor_moves_boundary(monkeypatch):
    # unit 0 claims sentences 0-2; unit 1 double-claims sentence 1 → its start is
    # clamped 1→3 by the non-overlap cursor and MUST carry the warning.
    units = _extract(monkeypatch, [
        UnitLLM(sentence_start=0, sentence_end=2, role="explanation", summary="a"),
        UnitLLM(sentence_start=1, sentence_end=5, role="explanation", summary="b"),
    ])
    assert len(units) == 2
    assert units[0].warnings == []
    assert units[1].warnings == ["boundary_clamped"]
    assert units[1].sentence_range == (3, 5)           # the clamp really moved s0

def test_no_clamp_warning_when_boundaries_are_clean(monkeypatch):
    units = _extract(monkeypatch, [
        UnitLLM(sentence_start=0, sentence_end=2, role="explanation", summary="a"),
        UnitLLM(sentence_start=3, sentence_end=5, role="explanation", summary="b"),
    ])
    assert [u.warnings for u in units] == [[], []]
    assert [u.sentence_range for u in units] == [(0, 2), (3, 5)]


# ── (b) drift_stats + the structure-level degraded note ──────────────────────
def test_drift_stats_counts_clamps_and_uncovered_gaps():
    units = [
        _unit("u0000", 0, sentence_range=(0, 1), warnings=["boundary_clamped"]),
        _unit("u0001", 1, sentence_range=(3, 4)),
    ]
    # topic partition is 0..5; units cover {0,1,3,4} → gaps {2,5}
    assert drift_stats(units, _cm(6), 6) == (1, 2)

def test_drift_stats_clean_partition_is_zero():
    units = [_unit("u0000", 0, sentence_range=(0, 2)), _unit("u0001", 1, sentence_range=(3, 5))]
    assert drift_stats(units, _cm(6), 6) == (0, 0)

def test_build_structure_emits_unit_drift_note_and_edge_pass_note(monkeypatch):
    sents = make_sents(6)
    units = [
        _unit("u0000", 0, sentence_range=(0, 1), warnings=["boundary_clamped"]),
        _unit("u0001", 1, sentence_range=(3, 4)),
    ]
    monkeypatch.setattr(build_mod, "build_content_map", lambda s, st, cb: _cm(6))
    monkeypatch.setattr(build_mod, "extract_units", lambda *a, **kw: units)
    monkeypatch.setattr(build_mod, "build_dependency_graph",
                        lambda u, s, cb: DependencyGraph(
                            degraded=["dependency_llm_edges: RuntimeError: boom"]))
    st = build_mod.build_structure("vidD", {"title": "t"}, sents, adapter=None,
                                   detection=DetectionResult(), settings={})
    assert "unit_drift: 1 boundary_clamped unit(s), 2 uncovered sentence(s)" in st.degraded
    assert "dependency_llm_edges: RuntimeError: boom" in st.degraded

def test_build_structure_clean_build_has_no_drift_note(monkeypatch):
    sents = make_sents(6)
    units = [_unit("u0000", 0, sentence_range=(0, 2)), _unit("u0001", 1, sentence_range=(3, 5))]
    monkeypatch.setattr(build_mod, "build_content_map", lambda s, st, cb: _cm(6))
    monkeypatch.setattr(build_mod, "extract_units", lambda *a, **kw: units)
    monkeypatch.setattr(build_mod, "build_dependency_graph", lambda u, s, cb: DependencyGraph())
    st = build_mod.build_structure("vidD", {"title": "t"}, sents, adapter=None,
                                   detection=DetectionResult(), settings={})
    assert not any(n.startswith("unit_drift") for n in st.degraded)


# ── (c) no PRIOR introducer ⇒ no edge (both modes) ───────────────────────────
def _future_intro_units() -> list[Unit]:
    """u0000 needs + back-references 'momentum', but the ONLY introducer is the LATER
    u0001 — under the old lst[0] fallback both resolutions produced forward edges."""
    return [
        _unit("u0000", 0, role="explanation", concepts_required=["momentum"],
              references=[Reference(text="that quantity", resolves_to="momentum")]),
        _unit("u0001", 1, role="definition", concepts_introduced=["momentum"]),
    ]

def test_no_prior_introducer_yields_no_requires_or_refers_edge(monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("offline")                    # rule/bridge edges only
    monkeypatch.setattr(llm_mod, "llm_json", boom)
    units = _future_intro_units()
    graph = build_dependency_graph(units, settings={})
    assert not any(e.relation == "requires" for e in graph.edges)   # no forward requires
    assert not any(e.relation == "refers_to" for e in graph.edges)  # nearest mode too
    assert units[0].references[0].source_unit is None    # resolution stays unresolved
    assert graph.forward_requires_count == 0             # the lint agrees post-fix

def test_prior_introducer_still_resolves(monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("offline")
    monkeypatch.setattr(llm_mod, "llm_json", boom)
    units = [
        _unit("u0000", 0, role="definition", concepts_introduced=["momentum"]),
        _unit("u0001", 1, role="explanation", concepts_required=["momentum"]),
    ]
    graph = build_dependency_graph(units, settings={})
    req = [(e.source, e.target) for e in graph.edges if e.relation == "requires"]
    assert ("u0001", "u0000") in req                     # backward edges keep working


# ── (d) LLM edge-pass failure ⇒ degraded note, not silence ───────────────────
def test_edge_pass_failure_surfaces_degraded_note(monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("provider down")
    monkeypatch.setattr(llm_mod, "llm_json", boom)
    graph = build_dependency_graph(_future_intro_units(), settings={})
    assert len(graph.degraded) == 1
    assert graph.degraded[0].startswith("dependency_llm_edges:")
    assert "RuntimeError" in graph.degraded[0] and "provider down" in graph.degraded[0]

def test_edge_pass_success_has_no_degraded_note(monkeypatch):
    monkeypatch.setattr(llm_mod, "llm_json", lambda *a, **kw: EdgesLLM(edges=[]))
    graph = build_dependency_graph(_future_intro_units(), settings={})
    assert graph.degraded == []


# ── (e) forward-requires graph lint ──────────────────────────────────────────
def test_forward_requires_lint_counts_llm_direction_violations(monkeypatch):
    # rules can no longer produce forward requires edges; an LLM edge that violates the
    # direction contract is COUNTED (telemetry, expect 0 on live runs), not filtered.
    monkeypatch.setattr(llm_mod, "llm_json", lambda *a, **kw: EdgesLLM(edges=[
        EdgeLLM(source="u0000", target="u0001", relation="requires", rationale="forward")]))
    graph = build_dependency_graph(_future_intro_units(), settings={})
    assert graph.forward_requires_count == 1
    assert any((e.source, e.target, e.relation) == ("u0000", "u0001", "requires")
               for e in graph.edges)
