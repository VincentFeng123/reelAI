"""Q2 graph nutrition: prerequisite-edge starvation fix (offline, mocked llm_json).

Pins (a) the unit-extraction prompt's ALWAYS-name-concepts_introduced instruction for
definition-style roles, (b) the LLM edge pass requesting AND accepting 'requires' edges,
(c) dedupe of LLM 'requires' edges against identical rule edges, and (d) the requires
direction convention (source = later dependent unit, target = earlier introducer).
"""
from __future__ import annotations

import backend.llm as llm_mod
from backend.adapters.lecture import LectureAdapter
from backend.pipeline.understand import dependencies as deps_mod
from backend.pipeline.understand import units as units_mod
from backend.pipeline.understand.dependencies import EdgeLLM, EdgesLLM, build_dependency_graph
from backend.pipeline.understand.models import ContentMap, ContentNode, Unit
from backend.pipeline.understand.units import UnitLLM, UnitsLLM, extract_units

from .conftest import make_sents


# ── fixtures ─────────────────────────────────────────────────────────────────
def _unit(uid: str, i: int, **kw) -> Unit:
    return Unit(unit_id=uid, start=i * 10.0, end=i * 10.0 + 5.0, sentence_range=(i, i),
                node_id="ch1.t1", topic="T",
                summary=kw.pop("summary", f"unit {uid}"),
                transcript=kw.pop("transcript", f"transcript of {uid}"), **kw)


def _graph_units() -> list[Unit]:
    """u0000 introduces 'momentum'; u0001 declares it required (→ rule requires edge);
    u0002 neither declares nor mentions it (so only the mocked LLM can link it)."""
    return [
        _unit("u0000", 0, role="definition", concepts_introduced=["momentum"],
              summary="defines momentum", transcript="momentum is mass times velocity"),
        _unit("u0001", 1, role="explanation", concepts_required=["momentum"],
              summary="uses the definition", transcript="so the quantity is conserved here"),
        _unit("u0002", 2, role="claim",
              summary="a later claim", transcript="the collision outcome follows"),
    ]


# ── Q2a: unit-extraction prompt ──────────────────────────────────────────────
def test_unit_prompt_instructs_concepts_introduced_for_definition_roles(monkeypatch):
    captured: dict[str, str] = {}

    def fake(system, user, schema, **kw):
        assert schema is UnitsLLM
        captured["system"] = system
        return UnitsLLM(units=[UnitLLM(sentence_start=0, sentence_end=3, role="definition",
                                       summary="d", concepts_introduced=["Momentum"])])

    monkeypatch.setattr(llm_mod, "llm_json", fake)
    sents = make_sents(4)
    cm = ContentMap(root_id="video", nodes=[
        ContentNode(node_id="video", level="video", sentence_range=(0, 3)),
        ContentNode(node_id="ch1.t1", level="topic", title="T", sentence_range=(0, 3)),
    ])
    units = extract_units(sents, cm, LectureAdapter(), settings={})

    system = captured["system"]
    # the added instruction: definition-style roles must ALWAYS fill concepts_introduced
    assert "ALWAYS name the concept(s)" in system
    for role in ("definition", "equation_introduction", "variable_definition"):
        assert role in system
    # the added example
    assert 'concepts_introduced: ["momentum"]' in system
    # sanity: extraction still works and normalizes concepts
    assert len(units) == 1 and units[0].concepts_introduced == ["momentum"]


def test_unit_prompt_template_carries_instruction():
    # template-level pin so the instruction can't be lost in a refactor of extract_units
    assert "ALWAYS name the concept(s)" in units_mod._SYSTEM_TMPL


# ── Q2b: LLM edge pass requests + accepts 'requires' ─────────────────────────
def test_llm_edge_prompt_requests_requires():
    sys_prompt = deps_mod._LLM_SYSTEM
    assert "requires" in sys_prompt
    # tight instruction: only genuine presupposition, direction matches candidate edges
    assert "presupposes" in sys_prompt
    assert "same direction as the candidate requires edges" in sys_prompt
    # 'requires' passes response validation
    assert "requires" in deps_mod._RELATIONS


def test_llm_requires_edge_accepted_and_deduped_against_rule_edge(monkeypatch):
    def fake(system, user, schema, **kw):
        assert schema is EdgesLLM
        return EdgesLLM(edges=[
            # exact duplicate of the rule edge u0001 -requires-> u0000: must be deduped
            EdgeLLM(source="u0001", target="u0000", relation="requires", rationale="dup"),
            # novel requires edge rules could not derive: must be accepted as derivation=llm
            EdgeLLM(source="u0002", target="u0000", relation="requires",
                    rationale="presupposes momentum"),
            # unknown unit id: must be rejected by validation
            EdgeLLM(source="u0002", target="zzz", relation="requires"),
        ])

    monkeypatch.setattr(llm_mod, "llm_json", fake)
    graph = build_dependency_graph(_graph_units(), settings={})

    keys = [(e.source, e.target, e.relation) for e in graph.edges]
    assert len(keys) == len(set(keys))                       # no duplicate pair+relation anywhere
    dup = [e for e in graph.edges if (e.source, e.target, e.relation) == ("u0001", "u0000", "requires")]
    assert len(dup) == 1 and dup[0].derivation == "rule"     # rule edge wins the dedupe
    novel = [e for e in graph.edges if (e.source, e.target, e.relation) == ("u0002", "u0000", "requires")]
    assert len(novel) == 1 and novel[0].derivation == "llm"  # LLM 'requires' accepted
    assert not any(e.target == "zzz" for e in graph.edges)   # bad id rejected


# ── Q2d: direction convention pinned by the rule-edge fixture ────────────────
def test_requires_direction_matches_rule_edge_convention(monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("offline")                        # rule/bridge edges only

    monkeypatch.setattr(llm_mod, "llm_json", boom)
    graph = build_dependency_graph(_graph_units(), settings={})
    req = [e for e in graph.edges if e.relation == "requires" and e.derivation == "rule"]
    # source = the LATER unit that needs the concept; target = the EARLIER introducer
    assert [(e.source, e.target) for e in req] == [("u0001", "u0000")]
