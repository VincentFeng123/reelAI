"""P3b/c/d — extraction plan validation (deterministic), enforcement (coverage /
saturation / arc retention / hard cap), the plan-fallback lever, and the plan path
end-to-end through assemble_clips. All offline (llm_json monkeypatched)."""
from __future__ import annotations

import pytest

import backend.llm as llm_mod
from backend import config
from backend.adapters.base import BaseAdapter
from backend.adapters.lecture import LectureAdapter
from backend.pipeline.assemble import assemble_clips
from backend.pipeline.assemble.candidates import (
    build_arc_candidate, enforce_plan, select_anchors, select_anchors_planned,
)
from backend.pipeline.assemble.graph import Graph
from backend.pipeline.assemble.plan import (
    ExtractionPlanLLM, PlanItemLLM, PlanProposal, validate_plan,
)
from backend.pipeline.assemble.validate import JudgeVerdict
from backend.pipeline.understand.arcs import ArcCheckLLM, ArcVerifyLLM, detect_arcs
from backend.pipeline.understand.models import (
    ContentMap, ContentNode, DependencyGraph, Structure, Unit,
)

from .conftest import mini_sents


def _units(roles, concepts=None, needs=None):
    concepts = concepts or {}
    needs = needs or {}
    return [Unit(unit_id=f"u{i:04d}", start=i * 10.0, end=i * 10.0 + 9.9,
                 sentence_range=(i, i), role=r, transcript=f"sentence {i}.",
                 summary=f"unit {i}",
                 concepts_introduced=list(concepts.get(i, [])),
                 concepts_required=list(needs.get(i, [])))
            for i, r in enumerate(roles)]


def _by_id(units):
    return {u.unit_id: u for u in units}


def _rel(units):
    return {u.unit_id: 1.0 for u in units}


def _cmap(n_sents, topic_ranges=None):
    nodes = [ContentNode(node_id="video", level="video", sentence_range=(0, n_sents - 1))]
    if topic_ranges:
        ch = ContentNode(node_id="c0", level="chapter", parent_id="video",
                         sentence_range=(0, n_sents - 1))
        nodes.append(ch)
        for i, (s0, s1) in enumerate(topic_ranges):
            nodes.append(ContentNode(node_id=f"c0.t{i}", level="topic", parent_id="c0",
                                     sentence_range=(s0, s1),
                                     start=s0 * 10.0, end=s1 * 10.0 + 9.9))
    return ContentMap(root_id="video", nodes=nodes)


def _structure(units, n_sents, topic_ranges=None):
    return Structure(video_id="v", units=units, dependencies=DependencyGraph(),
                     content_map=_cmap(n_sents, topic_ranges))


def _same_model(monkeypatch):
    monkeypatch.setattr(config, "JUDGE_MODEL", "")
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "same")


# ── P3b: deterministic plan validation ────────────────────────────────────────────────────
def test_validate_drops_unknown_ids_and_dedupes():
    units = _units(["claim", "definition"])
    items = [PlanItemLLM(anchor_unit_id="u0000", role="claim"),
             PlanItemLLM(anchor_unit_id="u9999", role="claim"),      # unknown unit id
             PlanItemLLM(arc_id="arc_7", role="result"),             # unknown arc id
             PlanItemLLM(anchor_unit_id="u0000", role="definition"),  # duplicate unit
             PlanItemLLM(anchor_unit_id="", arc_id="", role="claim"),  # no id at all
             PlanItemLLM(anchor_unit_id="u0001", role="definition")]
    got = validate_plan(items, _by_id(units), {}, BaseAdapter(), cap=12)
    assert [(p.kind, p.ref_id) for p in got] == [("unit", "u0000"), ("unit", "u0001")]
    assert [p.rank for p in got] == [0, 1]


def test_validate_coerces_unknown_roles_keeps_domain_roles():
    units = _units(["calculation", "explanation", "explanation"])
    items = [PlanItemLLM(anchor_unit_id="u0000", role="banter"),       # unknown → coerced
             PlanItemLLM(anchor_unit_id="u0001", role="derivation"),   # lecture domain role
             PlanItemLLM(anchor_unit_id="u0002", role="")]             # empty → unit's own
    got = validate_plan(items, _by_id(units), {}, LectureAdapter(), cap=12)
    assert [p.role for p in got] == ["explanation", "derivation", "explanation"]


def test_validate_arc_proposals_carry_terminal_role_and_win_on_both_ids():
    units = _units(["example_setup", "worked_step", "calculation"])
    arcs = detect_arcs(units)
    assert len(arcs) == 1 and arcs[0].terminal_role == "result"
    items = [PlanItemLLM(anchor_unit_id="u0000", arc_id="arc_0", role="claim")]  # both filled
    got = validate_plan(items, _by_id(units), {a.arc_id: a for a in arcs},
                        LectureAdapter(), cap=12)
    assert [(p.kind, p.ref_id, p.role) for p in got] == [("arc", "arc_0", "result")]


def test_validate_arc_supersedes_contained_unit_and_overlaps_dedupe():
    units = _units(["example_setup", "worked_step", "result", "claim"])
    arcs = detect_arcs(units)
    arcs_by_id = {a.arc_id: a for a in arcs}
    items = [PlanItemLLM(anchor_unit_id="u0001", role="claim"),   # inside the arc…
             PlanItemLLM(arc_id="arc_0", role="result"),          # …arc replaces it
             PlanItemLLM(anchor_unit_id="u0002", role="result"),  # inside kept arc → dropped
             PlanItemLLM(anchor_unit_id="u0003", role="claim")]
    got = validate_plan(items, _by_id(units), arcs_by_id, BaseAdapter(), cap=12)
    assert [(p.kind, p.ref_id) for p in got] == [("arc", "arc_0"), ("unit", "u0003")]


def test_validate_caps_at_max_anchors():
    units = _units(["claim"] * 8)
    items = [PlanItemLLM(anchor_unit_id=f"u{i:04d}", role="claim") for i in range(8)]
    got = validate_plan(items, _by_id(units), {}, BaseAdapter(), cap=3)
    assert len(got) == 3


# ── P3c: deterministic enforcement ────────────────────────────────────────────────────────
def test_enforcement_adds_best_anchor_for_plan_skipped_topic():
    units = _units(["claim", "claim", "definition", "explanation"])
    st = _structure(units, 4, topic_ranges=[(0, 1), (2, 3)])
    proposals = [PlanProposal("unit", "u0000", "claim", rank=0)]
    anchors, arc_map = enforce_plan(proposals, [], st, units, _rel(units), BaseAdapter(),
                                    {"max_anchors": 12})
    ids = [a.unit_id for a in anchors]
    # W25-C real floor: u0000 spans 9.9/19.9 = 0.497 < MIN_NODE_COVERAGE, so sliver-covered
    # c0.t0 is topped up with u0001 alongside plan-skipped c0.t1's best anchor u0002
    assert ids == ["u0000", "u0001", "u0002"]
    assert arc_map == {}


def test_enforcement_no_floor_for_topics_without_eligible_units():
    units = _units(["claim", "transition", "administrative"])   # topic B has no anchors
    st = _structure(units, 3, topic_ranges=[(0, 0), (1, 2)])
    proposals = [PlanProposal("unit", "u0000", "claim", rank=0)]
    anchors, _ = enforce_plan(proposals, [], st, units, _rel(units), BaseAdapter(),
                              {"max_anchors": 12})
    assert [a.unit_id for a in anchors] == ["u0000"]


def test_enforcement_role_saturation_drops_lowest_ranked_overflow():
    # the audit failure: six 'claim' units crowding everything else out. W25-C: the cap is
    # per (role, home_topic); with no topic nodes every unit shares the '' pseudo-topic,
    # so the wall of claims is capped at PLAN_ROLE_CAP_PER_TOPIC — lowest-ranked drop.
    units = _units(["claim"] * 6)
    st = _structure(units, 6)                  # no topic nodes → role quota only
    proposals = [PlanProposal("unit", f"u{i:04d}", "claim", rank=i) for i in range(6)]
    anchors, _ = enforce_plan(proposals, [], st, units, _rel(units), BaseAdapter(),
                              {"max_anchors": 12})
    assert [a.unit_id for a in anchors] == ["u0000", "u0001"]
    assert config.PLAN_ROLE_CAP_PER_TOPIC == 2


def test_enforcement_role_cap_is_per_topic_not_global():
    # W25-C (qP items 21-22): claim-dense zones are no longer starved by a video-global
    # role cap — each topic node holds up to PLAN_ROLE_CAP_PER_TOPIC claims of its own.
    units = _units(["claim"] * 6)
    st = _structure(units, 6, topic_ranges=[(0, 2), (3, 5)])
    proposals = [PlanProposal("unit", f"u{i:04d}", "claim", rank=i) for i in range(6)]
    anchors, _ = enforce_plan(proposals, [], st, units, _rel(units), BaseAdapter(),
                              {"max_anchors": 12})
    ids = [a.unit_id for a in anchors]
    # 2 claims per topic node survive saturation (the old global cap 4 kept u0000-u0003,
    # all from the first zone); each pair spans 19.8/29.9 = 0.66 ≥ MIN_NODE_COVERAGE so
    # the coverage floor tops up nothing
    assert ids == ["u0000", "u0001", "u0003", "u0004"]


def test_enforcement_topic_saturation_quota():
    # cap 4 over 2 topics → quota ceil(4/2)+1 = 3 anchors max per topic
    units = _units(["claim", "definition", "explanation", "intuition", "procedure",
                    "definition", "claim"])
    st = _structure(units, 7, topic_ranges=[(0, 4), (5, 6)])
    proposals = [PlanProposal("unit", f"u{i:04d}", units[i].role, rank=i) for i in range(5)]
    anchors, _ = enforce_plan(proposals, [], st, units, _rel(units), BaseAdapter(),
                              {"max_anchors": 4})
    ids = [a.unit_id for a in anchors]
    # 3 from topic A (quota) + topic B's BEST prior-scored anchor (claim 70 > definition 68)
    assert ids == ["u0000", "u0001", "u0002", "u0006"]
    assert len(ids) == 4


# ── W25-C: the coverage floor measures SPANNED time, not anchor presence ─────────────────
def test_floor_tops_up_sliver_covered_topic():
    # the items-3/4 class: one 9.9s anchor in a 49.9s node (0.198 < MIN_NODE_COVERAGE) no
    # longer marks the topic covered — its best remaining prior-scored anchor is added
    units = _units(["claim", "definition", "transition", "transition", "transition"])
    st = _structure(units, 5, topic_ranges=[(0, 4)])
    proposals = [PlanProposal("unit", "u0000", "claim", rank=0)]
    anchors, _ = enforce_plan(proposals, [], st, units, _rel(units), BaseAdapter(),
                              {"max_anchors": 12})
    assert [a.unit_id for a in anchors] == ["u0000", "u0001"]
    assert config.MIN_NODE_COVERAGE == 0.5


def test_floor_counts_union_of_selected_units():
    # two selected 9.9s claims span 19.8/29.9 = 0.66 ≥ MIN_NODE_COVERAGE jointly — the
    # eligible definition is NOT added (coverage is a union, not per-anchor)
    units = _units(["claim", "claim", "definition"])
    st = _structure(units, 3, topic_ranges=[(0, 2)])
    proposals = [PlanProposal("unit", "u0000", "claim", rank=0),
                 PlanProposal("unit", "u0001", "claim", rank=1)]
    anchors, _ = enforce_plan(proposals, [], st, units, _rel(units), BaseAdapter(),
                              {"max_anchors": 12})
    assert [a.unit_id for a in anchors] == ["u0000", "u0001"]


def test_floor_untimed_node_keeps_any_anchor_rule():
    # older maps without node timing (end <= start): the fraction is uncomputable, so any
    # selected anchor still counts as coverage (never a spurious top-up)
    units = _units(["claim", "definition", "definition", "definition"])
    nodes = [ContentNode(node_id="video", level="video", sentence_range=(0, 3)),
             ContentNode(node_id="c0.t0", level="topic", parent_id="video",
                         sentence_range=(0, 3))]          # start == end == 0.0
    st = Structure(video_id="v", units=units, dependencies=DependencyGraph(),
                   content_map=ContentMap(root_id="video", nodes=nodes))
    proposals = [PlanProposal("unit", "u0000", "claim", rank=0)]
    anchors, _ = enforce_plan(proposals, [], st, units, _rel(units), BaseAdapter(),
                              {"max_anchors": 12})
    assert [a.unit_id for a in anchors] == ["u0000"]


def test_enforcement_arcs_exempt_from_saturation_and_added_when_plan_skips_them():
    units = _units(["result", "result", "result", "result",
                    "example_setup", "worked_step", "result"])
    arcs = detect_arcs(units)
    assert len(arcs) == 1
    st = _structure(units, 7)
    proposals = [PlanProposal("unit", f"u{i:04d}", "result", rank=i) for i in range(4)]
    anchors, arc_map = enforce_plan(proposals, arcs, st, units, _rel(units), BaseAdapter(),
                                    {"max_anchors": 12})
    ids = [a.unit_id for a in anchors]
    # 2 result units exhaust the per-topic role quota (W25-C: no topic nodes → one ''
    # pseudo-topic), yet the (plan-skipped) arc is still selected
    assert ids == ["u0000", "u0001", "arc_0"]
    assert arc_map["arc_0"] is arcs[0]
    synth = anchors[-1]
    assert synth.role == "result"              # the arc's terminal role
    assert synth.sentence_range == (4, 6)      # widened to the arc hull


def test_enforcement_hard_cap_preserved_arcs_kept_over_plan_extras():
    units = _units(["result", "result", "result", "result",
                    "example_setup", "worked_step", "result"])
    arcs = detect_arcs(units)
    st = _structure(units, 7)
    proposals = [PlanProposal("unit", f"u{i:04d}", "result", rank=i) for i in range(4)]
    anchors, arc_map = enforce_plan(proposals, arcs, st, units, _rel(units), BaseAdapter(),
                                    {"max_anchors": 3})
    ids = [a.unit_id for a in anchors]
    assert len(ids) == 3                       # (iv) the hard cap always wins
    assert "arc_0" in ids                      # arcs are dropped last
    assert ids == ["u0000", "u0001", "arc_0"]  # lowest plan-ranked extras dropped first


def test_enforcement_unit_role_override_keeps_unit_identity():
    units = _units(["calculation"])
    st = _structure(units, 1)
    proposals = [PlanProposal("unit", "u0000", "demonstration", rank=0)]
    anchors, _ = enforce_plan(proposals, [], st, units, _rel(units), BaseAdapter(),
                              {"max_anchors": 12})
    assert anchors[0].unit_id == "u0000" and anchors[0].role == "demonstration"
    assert units[0].role == "calculation"      # the real unit is untouched


# ── P3d: fallback lever — byte-equivalent legacy selection, flagged ──────────────────────
_FALLBACK_ROLES = ["definition", "explanation", "claim", "summary", "transition"]


def test_plan_call_failure_falls_back_byte_equivalent_and_flagged(monkeypatch):
    units = _units(_FALLBACK_ROLES)
    st = _structure(units, len(units))
    rel = _rel(units)
    settings = {"max_anchors": 12, "arc_verify": False}

    def boom(*a, **kw):
        raise RuntimeError("plan api down")
    monkeypatch.setattr(llm_mod, "llm_json", boom)
    anchors, arc_map, notes = select_anchors_planned(st, units, rel, BaseAdapter(),
                                                     settings, "")
    legacy = select_anchors(units, rel, BaseAdapter(), settings)
    assert [a.unit_id for a in anchors] == [u.unit_id for u in legacy]   # byte-equivalent
    assert [a.role for a in anchors] == [u.role for u in legacy]
    assert arc_map == {}
    assert any("plan-fallback" in n for n in notes)


def test_empty_plan_output_falls_back_flagged(monkeypatch):
    units = _units(_FALLBACK_ROLES)
    st = _structure(units, len(units))
    rel = _rel(units)
    settings = {"max_anchors": 12, "arc_verify": False}
    monkeypatch.setattr(llm_mod, "llm_json", lambda *a, **kw: ExtractionPlanLLM())
    anchors, arc_map, notes = select_anchors_planned(st, units, rel, BaseAdapter(),
                                                     settings, "")
    assert [a.unit_id for a in anchors] == \
        [u.unit_id for u in select_anchors(units, rel, BaseAdapter(), settings)]
    assert arc_map == {} and any("plan-fallback" in n for n in notes)


def test_garbage_plan_ids_fall_back_flagged(monkeypatch):
    units = _units(_FALLBACK_ROLES)
    st = _structure(units, len(units))
    rel = _rel(units)
    settings = {"max_anchors": 12, "arc_verify": False}
    monkeypatch.setattr(llm_mod, "llm_json", lambda *a, **kw: ExtractionPlanLLM(
        extractions=[PlanItemLLM(anchor_unit_id="u9999", role="claim"),
                     PlanItemLLM(arc_id="arc_42", role="result")]))
    anchors, _, notes = select_anchors_planned(st, units, rel, BaseAdapter(), settings, "")
    assert [a.unit_id for a in anchors] == \
        [u.unit_id for u in select_anchors(units, rel, BaseAdapter(), settings)]
    assert any("plan-fallback" in n for n in notes)


def test_arc_verify_outage_note_travels_with_successful_plan(monkeypatch):
    units = _units(["example_setup", "worked_step", "calculation"])
    st = _structure(units, len(units))

    def fake(system, user, schema, **kw):
        if schema is ArcVerifyLLM:
            raise RuntimeError("verify down")              # degrade: unverified-but-kept
        if schema is ExtractionPlanLLM:
            return ExtractionPlanLLM(extractions=[PlanItemLLM(arc_id="arc_0", role="result")])
        raise AssertionError(f"unexpected schema {schema}")
    monkeypatch.setattr(llm_mod, "llm_json", fake)
    anchors, arc_map, notes = select_anchors_planned(st, units, _rel(units), BaseAdapter(),
                                                     {"max_anchors": 12, "arc_verify": True}, "")
    assert [a.unit_id for a in anchors] == ["arc_0"]       # arc kept despite verify outage
    assert arc_map["arc_0"].verified is False
    assert any("unverified" in n for n in notes)
    assert not any("plan-fallback" in n for n in notes)


# ── arc candidate construction ────────────────────────────────────────────────────────────
def test_build_arc_candidate_inlines_the_whole_hull():
    units = _units(["definition", "example_setup", "worked_step", "calculation",
                    "calculation", "worked_step", "explanation"])
    arcs = detect_arcs(units)
    sents = mini_sents(7)
    cand = build_arc_candidate(arcs[0], Graph([], units), BaseAdapter(), units, _by_id(units),
                               sents, _rel(units), {"closure_max_span_s": 999.0,
                                                    "max_clip_duration_s": 500.0})
    assert cand is not None
    # W25-D(a): the no-terminal fallback is steps[-1] regardless of step role — the
    # anchor moved from the last 'calculation' (u0004) to the last step (u0005).
    assert cand.cand_id == "c_arc_0" and cand.anchor_id == "u0005"
    assert cand.role == "result" and cand.facet == "worked_example"
    assert (cand.i_start, cand.i_end) == (1, 5)            # the full example, nothing severed
    assert set(cand.unit_ids) == {"u0001", "u0002", "u0003", "u0004", "u0005"}
    assert cand.priority == pytest.approx(BaseAdapter().anchor_priority("result"))
    assert cand.truncated is False


def test_build_arc_candidate_pair_beyond_cap_degrades_to_closure_with_referential_prompt():
    roles = ["practice_prompt"] + ["explanation"] * 29 + ["solution"]
    units = _units(roles, concepts={0: ["momentum"]}, needs={30: ["momentum"]})
    arcs = detect_arcs(units)
    assert len(arcs) == 1 and arcs[0].arc_role == "practice_pair"
    sents = mini_sents(31)                                 # hull ≈ 310s > the 100s cap
    cand = build_arc_candidate(arcs[0], Graph([], units), BaseAdapter(), units, _by_id(units),
                               sents, _rel(units), {"closure_max_span_s": 100.0,
                                                    "max_clip_duration_s": 100.0})
    assert cand is not None
    assert cand.cand_id == "c_arc_0" and cand.role == "solution"
    assert cand.truncated is True
    assert float(sents[cand.i_end].end) - float(sents[cand.i_start].start) <= 100.0
    assert any(uid == "u0000" for uid, _rel_ in cand.referential)   # prompt surfaces via card


# ── end-to-end: the plan path through assemble_clips ─────────────────────────────────────
_E2E_SETTINGS = {"min_clip_duration_s": 1.0, "max_clip_duration_s": 500.0,
                 "min_comprehension_score": 0.7, "quality_floor": 0.0, "max_clips": 12,
                 "max_anchors": 12, "closure_max_span_s": 999.0,
                 "anchor_selector": "plan", "arc_verify": True,
                 # these e2e tests pin the ROUND-0 plan seam (exact plan-call counts);
                 # Q1e refund rounds are exercised in test_selection_budget.py
                 "refund_rounds": 0}


def test_assemble_plan_path_ships_the_calculation_final_arc(monkeypatch):
    # the kinematics acceptance shape: a complete worked example whose payoff units are
    # 'calculation'-labeled becomes ONE shipped clip via the arc → plan → enforce path.
    _same_model(monkeypatch)
    sents = mini_sents(8)
    units = _units(["definition", "example_setup", "worked_step", "calculation",
                    "calculation", "worked_step", "explanation", "summary"])
    st = _structure(units, 8)
    schemas = []

    def fake(system, user, schema, **kw):
        schemas.append(schema)
        if schema is ArcVerifyLLM:
            return ArcVerifyLLM(arcs=[ArcCheckLLM(arc_id="arc_0", problem_ids=["u0001"],
                                                  step_ids=["u0002", "u0003"],
                                                  answer_ids=["u0004"])])
        if schema is ExtractionPlanLLM:
            return ExtractionPlanLLM(extractions=[
                PlanItemLLM(arc_id="arc_0", role="result", purpose="complete worked example",
                            why_valuable="problem, working and answer all present")])
        if schema is JudgeVerdict:
            return JudgeVerdict(reasoning="ok", score_10=9, understandable=True)
        raise AssertionError(f"unexpected schema {schema}")
    monkeypatch.setattr(llm_mod, "llm_json", fake)

    specs, notes, rejections = assemble_clips(st, "", sents, "u", "v", _E2E_SETTINGS,
                                              LectureAdapter())
    assert len(specs) == 1 and rejections == []
    s = specs[0]
    assert s["cand_id"] == "c_arc_0"
    assert s["role"] == "result" and s["facet"] == "worked_example"
    assert s["sentence_start_idx"] <= 1 and s["sentence_end_idx"] >= 5   # full example inside
    assert {"u0001", "u0002", "u0003", "u0004", "u0005"} <= set(s["unit_ids"])
    assert "plan-fallback" not in notes
    assert schemas.count(ArcVerifyLLM) == 1 and schemas.count(ExtractionPlanLLM) == 1


def test_assemble_priority_selector_never_calls_plan_or_verify(monkeypatch):
    _same_model(monkeypatch)
    sents = mini_sents(4)
    units = _units(["definition", "explanation", "claim", "summary"])
    st = _structure(units, 4)

    def fake(system, user, schema, **kw):
        assert schema is JudgeVerdict, f"legacy path must only judge, got {schema}"
        return JudgeVerdict(reasoning="ok", score_10=9, understandable=True)
    monkeypatch.setattr(llm_mod, "llm_json", fake)
    settings = dict(_E2E_SETTINGS, anchor_selector="priority")
    specs, notes, _rej = assemble_clips(st, "", sents, "u", "v", settings, BaseAdapter())
    assert specs and "plan-fallback" not in notes


def test_assemble_plan_failure_degrades_to_priority_and_notes_it(monkeypatch):
    _same_model(monkeypatch)
    sents = mini_sents(4)
    units = _units(["definition", "explanation", "claim", "summary"])
    st = _structure(units, 4)

    def fake(system, user, schema, **kw):
        if schema is ExtractionPlanLLM:
            raise RuntimeError("plan api down")
        if schema is JudgeVerdict:
            return JudgeVerdict(reasoning="ok", score_10=9, understandable=True)
        raise AssertionError(f"unexpected schema {schema}")
    monkeypatch.setattr(llm_mod, "llm_json", fake)
    settings = dict(_E2E_SETTINGS, arc_verify=False)
    specs, notes, _rej = assemble_clips(st, "", sents, "u", "v", settings, BaseAdapter())
    assert specs                                # clips still ship on the legacy selection
    assert "plan-fallback" in notes


def test_default_selector_is_plan_and_settings_key_inherits():
    assert config.ANCHOR_SELECTOR == "plan"
    assert config.DEFAULTS["anchor_selector"] is None      # None → inherit config
    assert config.DEFAULTS["arc_verify"] is None
