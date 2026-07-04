"""P1-contract-by-content: the completeness contract is chosen from the roles ACTUALLY in
the assembled span (choose_contract), stored on the candidate separately from the anchor
role (provenance preserved), and REBOUND after every span mutation before re-judging —
so a claim-anchored span that swallowed a worked problem gets the problem_statement/
reasoning/result judge gates instead of shipping at internal 0.95 while the judge sees a
4/10 fragment. All offline (llm_json monkeypatched)."""
from __future__ import annotations

import threading

import pytest

import backend.llm as llm_mod
from backend.adapters.base import BaseAdapter
from backend.pipeline.assemble.contracts import (
    CONTRACT_PRECEDENCE, check_contract, choose_contract,
)
from backend.pipeline.assemble.graph import Graph
from backend.pipeline.assemble.types import Candidate
from backend.pipeline.assemble.validate import (
    JudgeVerdict, is_complete, rebind_contract, validate_and_repair,
)
from backend.pipeline.understand.models import (
    ContentMap, ContentNode, DependencyGraph, Structure, Unit,
)

from .conftest import FakeAdapter, mini_sents


def role_units(roles, sents):
    return [Unit(unit_id=f"u{i:04d}", start=sents[i].start, end=sents[i].end,
                 sentence_range=(i, i), role=r, transcript=sents[i].text)
            for i, r in enumerate(roles)]


def _by_id(units):
    return {u.unit_id: u for u in units}


def _same_model(monkeypatch):
    from backend import config
    monkeypatch.setattr(config, "JUDGE_MODEL", "")
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "same")


# ── P1a: choose_contract — content wins over anchor ──────────────────────────
def test_claim_span_with_worked_problem_binds_result():
    # the audit shape: a claim anchor whose span swallowed setup+steps+answer
    sents = mini_sents(4)
    units = role_units(["claim", "example_setup", "worked_step", "result"], sents)
    assert choose_contract([u.unit_id for u in units], _by_id(units), BaseAdapter()) == "result"


def test_plain_claim_span_binds_claim():
    sents = mini_sents(2)
    units = role_units(["claim", "exception"], sents)
    assert choose_contract(["u0000"], _by_id(units), BaseAdapter()) == "claim"
    assert choose_contract([u.unit_id for u in units], _by_id(units), BaseAdapter()) == "claim"


def test_no_satisfiable_contract_returns_none():
    sents = mini_sents(1)
    units = role_units(["summary"], sents)
    assert choose_contract([u.unit_id for u in units], _by_id(units), BaseAdapter()) is None


def test_contract_free_adapter_returns_none():
    sents = mini_sents(2)
    units = role_units(["claim", "result"], sents)
    assert choose_contract([u.unit_id for u in units], _by_id(units), FakeAdapter()) is None


def test_specificity_tiebreak_more_required_elements_wins():
    # result (3/3 required) beats solution (2/2), practice_prompt/procedure/claim (1/1)
    sents = mini_sents(5)
    units = role_units(["example_setup", "worked_step", "result", "solution", "practice_prompt"], sents)
    assert choose_contract([u.unit_id for u in units], _by_id(units), BaseAdapter()) == "result"


def test_calculation_as_final_binds_result_not_procedure():
    # FLAGSHIP audit case (docs/audits/2026-07-02/kinematics_e2e.json u0021–u0025): the
    # worked example's answer lives in 'calculation'-role units — no result/solution unit
    # exists. The result contract must satisfy on 'calculation' (mirroring arcs.STEP_ROLES /
    # calculation-as-final) so it scores 3/3 with specificity 3 and beats 'procedure' (1/1,
    # specificity 1); otherwise the problem_statement/reasoning/result judge gates are
    # bypassed, the example_setup opener is unprotected from trims, and the clip is
    # invisible to metrics.worked_example_completeness.
    from backend.adapters.lecture import LectureAdapter
    sents = mini_sents(5)
    units = role_units(["example_setup", "worked_step", "calculation", "calculation",
                        "worked_step"], sents)
    ids = [u.unit_id for u in units]
    # the lecture adapter OVERRIDES the generic result contract — both must tolerate the
    # mislabeled terminal, or the flagship video (which runs under lecture) stays broken.
    for adapter in (BaseAdapter(), LectureAdapter()):
        assert choose_contract(ids, _by_id(units), adapter) == "result"
        # the span is structurally complete under the bound contract (no fill spin
        # hunting for a literal 'result' unit)…
        assert check_contract(ids, "result", _by_id(units), adapter).ok is True
        # …and the audit's judge gates now apply to this shape
        assert {"problem_statement_complete", "reasoning_complete", "result_complete"} \
            <= set(adapter.required_verdict_fields("result"))


def test_derivation_steps_also_satisfy_result_contract():
    # derivation mirrors arcs.STEP_ROLES: setup + derivation steps + calculation answer
    sents = mini_sents(3)
    units = role_units(["example_setup", "derivation", "calculation"], sents)
    assert choose_contract([u.unit_id for u in units], _by_id(units), BaseAdapter()) == "result"


def test_calculation_answer_satisfies_solution_contract():
    sents = mini_sents(2)
    units = role_units(["practice_prompt", "calculation"], sents)
    ids = [u.unit_id for u in units]
    # a calculation-role unit now satisfies the solution contract's answer element, so a
    # span bound 'solution' (e.g. a practice-pair arc) is not falsely 'missing answer'…
    assert check_contract(ids, "solution", _by_id(units), BaseAdapter()).ok is True
    # …and choose_contract binds a problem-shaped contract with the full judge gates
    # (result wins 3/3 — calculation satisfies steps AND answer) instead of the gateless
    # practice_prompt contract (1/1) it bound before the fix.
    assert choose_contract(ids, _by_id(units), BaseAdapter()) == "result"


def test_calculation_final_arc_protects_problem_statement_from_trims():
    # consequence (b) of the audit finding: under the (wrong) procedure contract a P2 trim
    # could legally drop the example_setup opener out of a detected arc. Under the content-
    # bound result contract every unit satisfying a required element is trim-protected.
    from backend.pipeline.assemble.validate import _protected_unit_ids
    sents = mini_sents(5)
    units = role_units(["example_setup", "worked_step", "calculation", "transition",
                        "calculation"], sents)
    cand = _mk_cand(units, sents, anchor_idx=4)          # anchored at the terminal calculation
    cand.contract_role = choose_contract(cand.unit_ids, _by_id(units), BaseAdapter())
    assert cand.contract_role == "result"
    protected = _protected_unit_ids(cand, _by_id(units), BaseAdapter())
    assert "u0000" in protected                          # the problem statement can't leave
    assert {"u0001", "u0002", "u0004"} <= protected      # steps/answer satisfy required elements
    assert "u0003" not in protected                      # the interleaved transition may trim


def test_calculation_final_clip_visible_to_worked_example_metric():
    # consequence (c): under contract_role='procedure' the recovered shape was excluded from
    # metrics.worked_example_completeness (_PROBLEM_ANCHORS); bound as 'result' it is counted
    # — and this complete kinematics shape scores 1.0 under both contract owners.
    from backend.adapters.lecture import LectureAdapter
    from backend.eval import metrics
    sents = mini_sents(5)
    units = role_units(["example_setup", "worked_step", "calculation", "calculation",
                        "worked_step"], sents)
    by_id = _by_id(units)
    spec = {"role": "result", "contract_role": choose_contract(list(by_id), by_id, BaseAdapter()),
            "unit_ids": list(by_id), "start": 0.0, "end": 5.0}
    assert spec["contract_role"] == "result"
    for adapter in (BaseAdapter(), LectureAdapter()):
        assert metrics.worked_example_completeness([spec], by_id, adapter) == 1.0


def test_precedence_tiebreak_deterministic():
    a = BaseAdapter()
    # {claim, definition}: both contracts fully satisfied at specificity 1 → definition > claim
    sents = mini_sents(2)
    units = role_units(["claim", "definition"], sents)
    assert choose_contract([u.unit_id for u in units], _by_id(units), a) == "definition"
    # {claim, worked_step}: claim vs procedure both 1/1 spec-1 → procedure > claim
    units = role_units(["claim", "worked_step"], sents)
    assert choose_contract([u.unit_id for u in units], _by_id(units), a) == "procedure"


def test_precedence_order_matches_spec():
    assert CONTRACT_PRECEDENCE == ("result", "derivation", "solution", "procedure",
                                   "practice_prompt", "correction", "definition", "claim")


def test_choice_deterministic_under_unit_order_and_repetition():
    sents = mini_sents(4)
    units = role_units(["claim", "example_setup", "worked_step", "result"], sents)
    ids = [u.unit_id for u in units]
    got = {choose_contract(order, _by_id(units), BaseAdapter())
           for order in (ids, list(reversed(ids)), ids[2:] + ids[:2], ids)}
    assert got == {"result"}


# ── P1b: contract_role on Candidate, gates keyed off it (anchor preserved) ───
def _mk_cand(units, sents, anchor_idx=0, unit_ids=None, role=None):
    ids = unit_ids if unit_ids is not None else [u.unit_id for u in units]
    picked = [u for u in units if u.unit_id in set(ids)]
    i0 = min(u.sentence_range[0] for u in picked)
    i1 = max(u.sentence_range[1] for u in picked)
    return Candidate(cand_id="c0", anchor_id=units[anchor_idx].unit_id,
                     role=role or units[anchor_idx].role, facet="other",
                     title="t", reason="r", unit_ids=ids, referential=[],
                     i_start=i0, i_end=i1, start=sents[i0].start, end=sents[i1].end)


def _vr(cand, sents, units, monkeypatch, fake_llm, settings=None):
    _same_model(monkeypatch)
    monkeypatch.setattr(llm_mod, "llm_json", fake_llm)
    return validate_and_repair(cand, sents, Graph([], units), units, _by_id(units), {},
                               BaseAdapter(), settings or {}, lambda s, e: "", "topic",
                               {}, threading.Lock())


def test_swallowed_problem_judge_gets_problem_gates(monkeypatch):
    sents = mini_sents(4)
    units = role_units(["claim", "example_setup", "worked_step", "result"], sents)
    users = []

    def fake(system, user, schema, **kw):
        users.append(user)
        return JudgeVerdict(reasoning="ok", score_10=9, understandable=True)
    kept, rej = _vr(_mk_cand(units, sents), sents, units, monkeypatch, fake)
    assert rej is None and kept is not None
    assert kept.role == "claim"                     # anchor provenance untouched
    assert kept.contract_role == "result"           # content-bound contract
    # the judge was briefed as a worked example, not a bare claim:
    assert "CLIP ROLE: result" in users[0]
    assert "problem_statement" in users[0] and "solution_steps" in users[0]
    assert "CLIP ROLE: claim" not in users[0]


def test_problem_gates_close_the_anchor_role_bypass(monkeypatch):
    # audit bug: score 9/10 + result_complete=False passed the claim gate; under the
    # content-bound result contract it must NOT be accepted as complete.
    sents = mini_sents(4)
    units = role_units(["claim", "example_setup", "worked_step", "result"], sents)
    v = JudgeVerdict(reasoning="missing payoff", score_10=9, understandable=True,
                     result_complete=False)
    kept, rej = _vr(_mk_cand(units, sents), sents, units, monkeypatch,
                    lambda *a, **kw: v.model_copy(deep=True))
    assert rej is None and kept is not None          # hard core intact → best-partial ships
    assert kept.contract_role == "result"
    adapter = BaseAdapter()
    assert is_complete(kept.verdict, kept.contract_role, adapter, 0.7) is False   # gated
    assert is_complete(kept.verdict, kept.role, adapter, 0.7) is True             # old bypass
    from backend.pipeline.assemble.scoring import completeness_score
    # scored under the SAME contract the judge used: 7/8 verdict fields, not 5/5
    assert completeness_score(kept.verdict, kept.contract_role, adapter) == pytest.approx(7 / 8)
    assert completeness_score(kept.verdict, kept.role, adapter) == pytest.approx(1.0)


def test_check_contract_keys_off_contract_role():
    sents = mini_sents(4)
    units = role_units(["claim", "example_setup", "worked_step", "result"], sents)
    ids = [u.unit_id for u in units]
    # under the content-bound contract the span is structurally complete…
    assert check_contract(ids, "result", _by_id(units), BaseAdapter()).ok is True
    # …and dropping the answer is DETECTED under it (the claim contract would stay ok)
    partial = ids[:3]
    assert check_contract(partial, "result", _by_id(units), BaseAdapter()).missing == ["result"]
    assert check_contract(partial, "claim", _by_id(units), BaseAdapter()).ok is True


def test_anchor_role_still_reported_in_spec_payload():
    from backend.pipeline.assemble.boundary_adapt import snap_candidates
    sents = mini_sents(4)
    units = role_units(["claim", "example_setup", "worked_step", "result"], sents)
    c = _mk_cand(units, sents)
    c.contract_role = "result"
    c.verdict = JudgeVerdict(score_10=9, understandable=True)
    specs, _rej = snap_candidates([c], sents, {"min_clip_duration_s": 1.0,
                                               "max_clip_duration_s": 500.0}, units)
    assert specs[0]["role"] == "claim"               # payload provenance preserved
    assert specs[0]["contract_role"] == "result"     # governing contract travels alongside


# ── P1c: rebinding fires on every span mutation ───────────────────────────────
def test_rebind_helper_tracks_mutations_and_falls_back_to_anchor():
    sents = mini_sents(4)
    units = role_units(["claim", "example_setup", "worked_step", "result"], sents)
    cand = _mk_cand(units, sents, unit_ids=["u0000"])
    assert rebind_contract(cand, _by_id(units), BaseAdapter()) == "claim"
    cand.unit_ids = [u.unit_id for u in units]       # span mutated (expansion/trim/merge)
    assert rebind_contract(cand, _by_id(units), BaseAdapter()) == "result"
    assert cand.contract_role == "result" and cand.role == "claim"
    # no binding possible → anchor role (pre-P1 behavior, contract-free adapters included)
    assert rebind_contract(cand, _by_id(units), FakeAdapter()) == "claim"


def test_rebind_fires_on_repair_expansion_before_rejudge(monkeypatch):
    # claim anchored alone; the failing verdict pulls in the worked problem around it →
    # the SECOND judge call must already run under the rebound result contract.
    sents = mini_sents(4)
    units = role_units(["example_setup", "worked_step", "claim", "result"], sents)
    users = []

    def fake(system, user, schema, **kw):
        users.append(user)
        if len(users) == 1:
            return JudgeVerdict(reasoning="fragment", score_10=3, understandable=False,
                                problem_statement_complete=False, result_complete=False)
        return JudgeVerdict(reasoning="complete now", score_10=9, understandable=True)
    cand = _mk_cand(units, sents, anchor_idx=2, unit_ids=["u0002"])
    kept, rej = _vr(cand, sents, units, monkeypatch, fake,
                    settings={"min_comprehension_score": 0.7})
    assert rej is None and kept is not None
    assert kept.attempts == 2                        # judged, expanded, re-judged
    assert kept.role == "claim"
    assert kept.contract_role == "result"            # rebound AFTER the expansion
    assert set(kept.unit_ids) == {u.unit_id for u in units}
    assert "CLIP ROLE: claim" in users[0]            # first pass: span was just the claim
    assert "CLIP ROLE: result" in users[1]           # re-judge ran under the new contract
    assert "problem_statement" in users[1]


class ClaimOnlyAdapter(BaseAdapter):
    """Real generic contracts; anchors restricted to 'claim' so the test yields one clip."""

    def is_anchor_role(self, role):
        return role == "claim"

    def anchor_priority(self, role):
        return 0.9 if role == "claim" else 0.0


def test_closure_swallowed_problem_binds_result_at_first_judge(monkeypatch):
    # a claim anchor NEXT TO a worked problem: closure's claim contract already inlines the
    # nearby result (statement element accepts result roles) → the FIRST judge pass must
    # already run under the content-bound result contract (one judge call, no repair spin).
    from backend.pipeline.assemble import assemble_clips
    _same_model(monkeypatch)
    users = []

    def fake(system, user, schema, **kw):
        assert schema is JudgeVerdict
        users.append(user)
        return JudgeVerdict(reasoning="ok", score_10=9, understandable=True)
    monkeypatch.setattr(llm_mod, "llm_json", fake)

    sents = mini_sents(5)
    units = role_units(["claim", "example_setup", "worked_step", "result", "summary"], sents)
    st = Structure(video_id="v", units=units, dependencies=DependencyGraph(),
                   content_map=ContentMap(nodes=[ContentNode(node_id="video", level="video",
                                                             sentence_range=(0, len(sents) - 1))]))
    settings = {"min_clip_duration_s": 1.0, "max_clip_duration_s": 500.0,
                "min_comprehension_score": 0.7, "quality_floor": 0.0, "max_clips": 12,
                "max_anchors": 12, "closure_max_span_s": 999.0,
                # pin the legacy selector: these tests count judge calls / anchor identity
                # and target the priority path (P3 keeps it byte-equivalent as the A/B lever)
                "anchor_selector": "priority"}
    specs, _notes, rejections = assemble_clips(st, "", sents, "u", "v", settings, ClaimOnlyAdapter())
    assert len(specs) == 1 and rejections == []
    s = specs[0]
    assert s["role"] == "claim" and s["contract_role"] == "result"
    assert len(users) == 1                           # bound by content BEFORE the only judge call
    assert "CLIP ROLE: result" in users[0] and "problem_statement" in users[0]


def test_post_snap_mutation_rebinds_before_rejudge(monkeypatch):
    # a claim anchor FAR (>5 units) from a worked problem: closure keeps the bare claim and
    # the accepted verdict covers "sentence 0." only; min-duration snapping then extends the
    # span over the whole worked problem → the 4b seam must REBIND (claim → result) BEFORE
    # the re-judge, and completeness must be scored under the contract the fresh verdict used.
    from backend.pipeline.assemble import assemble_clips
    _same_model(monkeypatch)
    users = []

    def fake(system, user, schema, **kw):
        assert schema is JudgeVerdict
        users.append(user)
        if "sentence 9." in user:                    # the post-snap re-judge (extended text)
            return JudgeVerdict(reasoning="answer cut off", score_10=9, understandable=True,
                                result_complete=False)
        return JudgeVerdict(reasoning="ok", score_10=9, understandable=True)
    monkeypatch.setattr(llm_mod, "llm_json", fake)

    sents = mini_sents(12, sec=4.0)                  # 4s sentences: extension reaches unit 9
    units = role_units(["claim"] + ["transition"] * 6
                       + ["example_setup", "worked_step", "result", "summary", "summary"], sents)
    st = Structure(video_id="v", units=units, dependencies=DependencyGraph(),
                   content_map=ContentMap(nodes=[ContentNode(node_id="video", level="video",
                                                             sentence_range=(0, len(sents) - 1))]))
    settings = {"min_clip_duration_s": 38.0, "max_clip_duration_s": 500.0,
                "min_comprehension_score": 0.7, "quality_floor": 0.0, "max_clips": 12,
                "max_anchors": 12, "closure_max_span_s": 999.0,
                # pin the legacy selector: these tests count judge calls / anchor identity
                # and target the priority path (P3 keeps it byte-equivalent as the A/B lever)
                "anchor_selector": "priority"}
    specs, _notes, rejections = assemble_clips(st, "", sents, "u", "v", settings, ClaimOnlyAdapter())
    assert len(specs) == 1 and all(r.stage != "post_snap_judge" for r in rejections)
    s = specs[0]
    assert s["role"] == "claim"                      # anchor provenance in the payload
    assert s["contract_role"] == "result"            # rebound from the FINAL span's roles
    assert "extended_for_min_duration" in s["warnings"]
    assert len(users) == 2                           # repair judge + post-snap re-judge
    assert "CLIP ROLE: claim" in users[0]            # judged as a bare claim originally
    assert "CLIP ROLE: result" in users[-1]          # re-judge briefed under the new contract
    assert "problem_statement" in users[-1]
    # completeness scored under the SAME contract the fresh verdict used (7/8, not 5/5)
    assert s["completeness_score"] == pytest.approx(7 / 8)
