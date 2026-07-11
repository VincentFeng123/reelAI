"""P4 — dedupe tie-break (P4a) + severed-pair linking/merging (P4b). All offline.

P4a (spec-governs: this DELIBERATELY unfreezes the pkg-1-era keep/drop tie-break):
overlap/containment losers are decided by (i) hard-core judge gates from the STORED
verdict, then (ii) contract-required-element coverage, then (iii) final_quality — the
audited NjvwWiCYLl4 shape (judge-inflated mid-clause fragment beating the complete
141s arc) must invert.

P4b: two clips holding the two halves of ONE worked example (audited kinematics bus
problem: setup clip + result clip, 42s hole, prerequisite_clips=[]) are ALWAYS linked
after sequencing; a merge is attempted only under the ship cap and kept only when the
union's never-judged text earns a clean fresh verdict.
"""
from __future__ import annotations

import pytest

import backend.llm as llm_mod
from backend.adapters.generic import GenericAdapter
from backend.pipeline.assemble.boundary_adapt import snap_candidates
from backend.pipeline.assemble.graph import Graph
from backend.pipeline.assemble.sequence import is_severed_pair, link_severed_pairs, sequence_clips
from backend.pipeline.assemble.types import Candidate
from backend.pipeline.assemble.validate import JudgeVerdict, judged_text_hash
from backend.pipeline.refine import _better
from backend.pipeline.understand.models import (
    ContentMap, ContentNode, DependencyGraph, Edge, Structure, Unit,
)

from .conftest import FakeAdapter, mini_sents, mini_units

_SETTINGS = {"min_clip_duration_s": 1.0, "max_clip_duration_s": 500.0}


def _clean_verdict(score_10=8):
    return JudgeVerdict(reasoning="ok", score_10=score_10, understandable=True)


def _cand(cand_id, s0, s1, sents, *, fq, verdict, contract_role=""):
    c = Candidate(cand_id=cand_id, anchor_id=f"u{s0:04d}", role="explanation", facet="other",
                  title=f"t-{cand_id}", reason="r",
                  unit_ids=[f"u{i:04d}" for i in range(s0, s1 + 1)], referential=[],
                  i_start=s0, i_end=s1, start=sents[s0].start, end=sents[s1].end)
    c.final_quality = fq
    c.verdict = verdict
    c.contract_role = contract_role
    return c


# ── P4a(i): hard-core gates beat judge-inflated scores ────────────────────────
def test_njv_shape_complete_arc_beats_judge_inflated_fragment():
    """The audit shape: a 140s arc that passed ALL hard-core gates at final_quality 0.855
    must beat a contained 40s fragment scored 0.948 whose stored verdict FAILED a
    hard-core gate (source_grounded — it starts and ends mid-clause)."""
    sents = mini_sents(16)
    units = mini_units(sents)
    arc = _cand("arc", 0, 13, sents, fq=0.855, verdict=_clean_verdict(8))
    frag = _cand("frag", 4, 7, sents, fq=0.948,
                 verdict=JudgeVerdict(reasoning="inflated", score_10=9, understandable=True,
                                      source_grounded=False))
    specs, rejections = snap_candidates([arc, frag], sents, _SETTINGS, units, FakeAdapter())
    assert [s["cand_id"] for s in specs] == ["arc"]
    assert len(rejections) == 1
    r = rejections[0]
    assert r.stage == "dedupe" and r.cand_id == "frag" and "arc" in r.reason


def test_error_verdict_loses_hard_gate_leg():
    """An unjudged (outage) spec never carries hard_gates_ok — a judged, gate-passing spec
    beats it even at lower final_quality."""
    sents = mini_sents(10)
    units = mini_units(sents)
    good = _cand("good", 0, 6, sents, fq=0.5, verdict=_clean_verdict())
    outage = _cand("outage", 1, 3, sents, fq=0.9,
                   verdict=JudgeVerdict(error=True, understandable=False, score=0.0,
                                        topic_identifiable=False, purpose_identifiable=False,
                                        all_references_resolved=False, source_grounded=False))
    specs, rejections = snap_candidates([good, outage], sents, _SETTINGS, units, FakeAdapter())
    assert [s["cand_id"] for s in specs] == ["good"]
    assert rejections[0].cand_id == "outage"


# ── P4a(ii): contract-required-element coverage breaks gate ties ──────────────
def _coverage_units(sents):
    roles = ["example_setup", "worked_step", "result", "explanation"]
    return [Unit(unit_id=f"u{i:04d}", start=s.start, end=s.end, sentence_range=(i, i),
                 role=roles[i], transcript=s.text) for i, s in enumerate(sents)]


def test_contract_coverage_breaks_hard_gate_ties():
    """Both pass the hard core; the container covers 3/3 required elements of the bound
    'result' contract (setup+steps+result), the contained fragment only 2/3 — coverage
    outranks the fragment's higher final_quality."""
    sents = mini_sents(4)
    units = _coverage_units(sents)
    full = _cand("full", 0, 2, sents, fq=0.5, verdict=_clean_verdict(),
                 contract_role="result")
    frag = _cand("frag", 1, 2, sents, fq=0.95, verdict=_clean_verdict(),
                 contract_role="result")
    specs, rejections = snap_candidates([full, frag], sents, _SETTINGS, units, GenericAdapter())
    assert [s["cand_id"] for s in specs] == ["full"]
    assert rejections[0].cand_id == "frag" and rejections[0].stage == "dedupe"
    assert specs[0]["contract_coverage"] == pytest.approx(1.0)


# ── P4a(iii): final_quality still breaks true ties ────────────────────────────
def test_final_quality_breaks_true_ties():
    """Equal hard-gate results and (contract-free) equal coverage → the higher
    final_quality wins, even for the CONTAINED span (pre-P4 behavior preserved)."""
    sents = mini_sents(8)
    units = mini_units(sents)
    big = _cand("big", 0, 5, sents, fq=0.4, verdict=_clean_verdict())
    small = _cand("small", 1, 2, sents, fq=0.95, verdict=_clean_verdict())
    specs, rejections = snap_candidates([big, small], sents, _SETTINGS, units, FakeAdapter())
    assert [s["cand_id"] for s in specs] == ["small"]
    assert rejections[0].cand_id == "big"


def test_legacy_better_unchanged_without_partb_fields():
    """Legacy fast-path dicts carry neither hard_gates_ok nor contract_coverage — the
    shared key reduces to the pre-P4 score-then-length tie-break."""
    a = {"score": 0.5, "start": 0.0, "end": 10.0}
    b = {"score": 0.7, "start": 0.0, "end": 5.0}
    assert _better(a, b) is b                              # higher score wins
    c = {"score": 0.7, "start": 0.0, "end": 8.0}
    assert _better(b, c) is c                              # score tie → longer span wins


# ── P4b fixtures: severed pair (setup clip + payoff clip) ─────────────────────
def _sev_units(sents, roles, node_ids=None):
    node_ids = node_ids or [""] * len(roles)
    return [Unit(unit_id=f"u{i:04d}", start=s.start, end=s.end, sentence_range=(i, i),
                 role=roles[i], node_id=node_ids[i], transcript=s.text)
            for i, s in enumerate(sents)]


def _sev_specs(sents, earlier=(0, 1), later=(3, 3)):
    e0, e1 = earlier
    l0, l1 = later
    return [
        {"start": sents[e0].start, "end": sents[e1].end,
         "unit_ids": [f"u{i:04d}" for i in range(e0, e1 + 1)]},
        {"start": sents[l0].start, "end": sents[l1].end,
         "unit_ids": [f"u{i:04d}" for i in range(l0, l1 + 1)]},
    ]


def _linked(sents, units, specs, merge_fn=None, max_dur=500.0):
    units_by_id = {u.unit_id: u for u in units}
    specs = sequence_clips(specs, Graph([], units), units_by_id)
    return link_severed_pairs(specs, Graph([], units), units_by_id, max_dur, merge_fn=merge_fn)


def test_severed_pair_link_always_added():
    sents = mini_sents(4)
    units = _sev_units(sents, ["example_setup", "worked_step", "explanation", "result"])
    out = _linked(sents, units, _sev_specs(sents))          # 10.1s gap, same (unknown) topic
    assert len(out) == 2
    later = out[1]
    assert later["prerequisite_clips"] == [1]
    assert later["notes"] == ["continues clip 1"]
    assert "notes" not in out[0]                            # only the LATER clip is annotated


def test_severed_pair_link_dedups_existing_prereq_and_note():
    sents = mini_sents(4)
    units = _sev_units(sents, ["problem_givens", "worked_step", "explanation", "solution"])
    specs = _sev_specs(sents)
    units_by_id = {u.unit_id: u for u in units}
    specs = sequence_clips(specs, Graph([], units), units_by_id)
    specs[1]["prerequisite_clips"] = [1]                    # already hinted (concept pass)
    specs[1]["notes"] = ["continues clip 1"]
    out = link_severed_pairs(specs, Graph([], units), units_by_id, 500.0)
    assert out[1]["prerequisite_clips"] == [1]              # dedup'd, not duplicated
    assert out[1]["notes"] == ["continues clip 1"]


def test_no_link_or_merge_across_different_topics():
    sents = mini_sents(4)
    units = _sev_units(sents, ["example_setup", "worked_step", "explanation", "result"],
                       node_ids=["ch1.t1", "ch1.t1", "ch1.t1", "ch1.t2"])
    calls = {"n": 0}

    def spy(a, b):
        calls["n"] += 1
        return None
    out = _linked(sents, units, _sev_specs(sents), merge_fn=spy)
    assert calls["n"] == 0                                  # merge never attempted either
    assert out[1].get("prerequisite_clips") == [] and "notes" not in out[1]


def test_no_link_beyond_60s_gap():
    sents = mini_sents(10)
    units = _sev_units(sents, ["example_setup", "worked_step"] + ["explanation"] * 7 + ["result"])
    out = _linked(sents, units, _sev_specs(sents, earlier=(0, 1), later=(9, 9)))  # 70.1s gap
    assert out[1].get("prerequisite_clips") == [] and "notes" not in out[1]


def test_no_link_when_earlier_clip_is_already_complete():
    sents = mini_sents(4)
    units = _sev_units(sents, ["example_setup", "result", "explanation", "result"])
    out = _linked(sents, units, _sev_specs(sents))          # earlier clip has its payoff
    assert out[1].get("prerequisite_clips") == [] and "notes" not in out[1]


# ── W25-F: the later-side test is PAIR-SCOPED (spec-governs: this replaces the pkg-era
# blanket 'later has no opener roles' test, which closure defeated by construction —
# required problem statements are force-inlined into every payoff clip, so the linker
# fired 0 times everywhere). An opener unit inside the later clip blocks ONLY when
# answers edges / shared concepts tie it to THIS pair's problem. ──────────────────────
def test_later_opener_blocks_when_answered_by_later_payoff():
    # the later clip restates its own setup: its result unit ANSWERS the in-clip setup —
    # genuinely self-contained, so no severed pair.
    sents = mini_sents(4)
    units = _sev_units(sents, ["example_setup", "worked_step", "example_setup", "result"])
    units_by_id = {u.unit_id: u for u in units}
    graph = Graph([Edge(source="u0003", target="u0002", relation="answers")], units)
    specs = _sev_specs(sents, earlier=(0, 1), later=(2, 3))
    assert is_severed_pair(specs[0], specs[1], units_by_id, graph=graph) is False


def test_later_opener_blocks_when_sharing_earlier_concepts():
    # no graph evidence, but the in-clip setup trades concepts with the earlier clip —
    # same problem restated, so the later clip is not severed from the earlier one.
    sents = mini_sents(4)
    units = _sev_units(sents, ["example_setup", "worked_step", "example_setup", "result"])
    units[0].concepts_introduced = ["bus problem"]
    units[2].concepts_required = ["bus problem"]
    units_by_id = {u.unit_id: u for u in units}
    specs = _sev_specs(sents, earlier=(0, 1), later=(2, 3))
    assert is_severed_pair(specs[0], specs[1], units_by_id) is False


def test_unrelated_later_opener_no_longer_vetoes_the_link():
    # the audited failure shape: the payoff clip carries a DIFFERENT problem's setup
    # (drifted in via closure / snap) — the payoff answers the EARLIER clip's prompt
    # (answers edge into the earlier clip), no tie to the in-clip opener → still a pair.
    sents = mini_sents(4)
    units = _sev_units(sents, ["practice_prompt", "worked_step", "example_setup", "result"])
    units[2].concepts_introduced = ["unrelated next problem"]
    units_by_id = {u.unit_id: u for u in units}
    graph = Graph([Edge(source="u0003", target="u0000", relation="answers")], units)
    specs = _sev_specs(sents, earlier=(0, 1), later=(2, 3))
    assert is_severed_pair(specs[0], specs[1], units_by_id, graph=graph) is True


def test_severed_opener_roles_gained_practice_prompt_and_setup():
    from backend.pipeline.assemble.sequence import SEVERED_OPENER_ROLES
    assert {"practice_prompt", "setup"} <= SEVERED_OPENER_ROLES     # W25-F additions
    assert {"example_setup", "problem_givens"} <= SEVERED_OPENER_ROLES


def test_link_scans_non_adjacent_later_clips_within_gap():
    # W25-F: an unrelated clip sits BETWEEN the two halves — the old zip(specs, specs[1:])
    # scan could never see the pair; the relaxed scan links across it (gap 20.1s ≤ 60s)
    # and annotates ONLY the payoff clip.
    sents = mini_sents(5)
    units = _sev_units(sents, ["practice_prompt", "worked_step", "explanation",
                               "explanation", "solution"])
    specs = [
        {"start": sents[0].start, "end": sents[1].end, "unit_ids": ["u0000", "u0001"]},
        {"start": sents[2].start, "end": sents[2].end, "unit_ids": ["u0002"]},
        {"start": sents[4].start, "end": sents[4].end, "unit_ids": ["u0004"]},
    ]
    out = _linked(sents, units, specs)
    assert len(out) == 3
    assert out[2]["prerequisite_clips"] == [1]              # linked across the middle clip
    assert out[2]["notes"] == ["continues clip 1"]
    assert out[1].get("prerequisite_clips") == [] and "notes" not in out[1]


def test_non_adjacent_scan_still_respects_the_gap_cap():
    sents = mini_sents(10)
    units = _sev_units(sents, ["practice_prompt", "worked_step"] + ["explanation"] * 7
                       + ["solution"])
    specs = [
        {"start": sents[0].start, "end": sents[1].end, "unit_ids": ["u0000", "u0001"]},
        {"start": sents[3].start, "end": sents[3].end, "unit_ids": ["u0003"]},
        {"start": sents[9].start, "end": sents[9].end, "unit_ids": ["u0009"]},
    ]
    out = _linked(sents, units, specs)                      # payoff at 70.1s gap: too far
    assert out[2].get("prerequisite_clips") == [] and "notes" not in out[2]


# ── P4b: merge attempted only under the cap; merge_fn verdict decides ─────────
def test_merge_only_attempted_under_cap():
    sents = mini_sents(4)
    units = _sev_units(sents, ["example_setup", "worked_step", "explanation", "result"])
    calls = {"n": 0}

    def spy(a, b):
        calls["n"] += 1
        return None
    out = _linked(sents, units, _sev_specs(sents), merge_fn=spy, max_dur=30.0)  # span 39.9 > 30
    assert calls["n"] == 0                                  # never attempted over the cap
    assert out[1]["prerequisite_clips"] == [1]              # …but the link is ALWAYS added
    out = _linked(sents, units, _sev_specs(sents), merge_fn=spy, max_dur=50.0)
    assert calls["n"] == 1                                  # under the cap → attempted


def test_merge_fn_rejection_keeps_linked_pair():
    sents = mini_sents(4)
    units = _sev_units(sents, ["example_setup", "worked_step", "explanation", "result"])
    out = _linked(sents, units, _sev_specs(sents), merge_fn=lambda a, b: None, max_dur=500.0)
    assert len(out) == 2
    assert out[1]["prerequisite_clips"] == [1]
    assert out[1]["notes"] == ["continues clip 1"]


def test_successful_merge_replaces_pair_and_resequences():
    sents = mini_sents(6)
    units = _sev_units(sents, ["example_setup", "worked_step", "explanation", "result",
                               "explanation", "explanation"])
    specs = _sev_specs(sents) + [{"start": sents[5].start, "end": sents[5].end,
                                  "unit_ids": ["u0005"]}]
    out = _linked(sents, units, specs,
                  merge_fn=lambda a, b: {"start": a["start"], "end": b["end"],
                                         "unit_ids": list(a["unit_ids"]) + list(b["unit_ids"])},
                  max_dur=500.0)
    assert len(out) == 2                                    # pair collapsed into one clip
    assert [s["sequence_index"] for s in out] == [1, 2]     # fresh indices after the merge
    assert all("notes" not in s for s in out)               # nothing left severed to link


# ── P4b integration: the real merge closure re-judges the union text ──────────
class SevAdapter(FakeAdapter):
    def is_anchor_role(self, role):
        return True

    def anchor_priority(self, role):
        return 0.9

    def facet_for(self, role):
        return "other"

    def valid_roles(self):
        return {"example_setup", "result"}


def _severed_structure():
    """u_setup on sentence 0, u_result on sentence 2 (10.1s hole at sentence 1) → two
    single-unit clips forming a severed pair whose union spans sentences 0-2."""
    sents = mini_sents(3)
    units = [
        Unit(unit_id="u_setup", start=sents[0].start, end=sents[0].end, sentence_range=(0, 0),
             role="example_setup", transcript=sents[0].text),
        Unit(unit_id="u_result", start=sents[2].start, end=sents[2].end, sentence_range=(2, 2),
             role="result", transcript=sents[2].text),
    ]
    st = Structure(video_id="v", units=units, dependencies=DependencyGraph(),
                   content_map=ContentMap(nodes=[ContentNode(node_id="video", level="video",
                                                             sentence_range=(0, 2))]))
    return sents, units, st


def _sev_settings(max_dur=500.0):
    return {"min_clip_duration_s": 1.0, "max_clip_duration_s": max_dur,
            "min_comprehension_score": 0.7, "quality_floor": 0.0, "max_clips": 12,
            "max_anchors": 12, "closure_max_span_s": 999.0,
            # legacy selector pinned: these tests count llm_json calls on the judge path
            "anchor_selector": "priority"}


def _mock_judge(monkeypatch, on_union):
    """Clean verdicts for the two single-sentence clips; `on_union(user)` answers the
    severed-merge probe (the only call whose text spans sentences 0 AND 2)."""
    from backend import config
    monkeypatch.setattr(config, "JUDGE_MODEL", "")
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "same")
    calls = {"n": 0, "union": 0}

    def judge(system, user, schema, **kw):
        assert schema is JudgeVerdict
        calls["n"] += 1
        if "sentence 0." in user and "sentence 2." in user:
            calls["union"] += 1
            return on_union(user)
        return _clean_verdict(9)
    monkeypatch.setattr(llm_mod, "llm_json", judge)
    return calls


def test_severed_merge_kept_on_clean_rejudge(monkeypatch):
    from backend.pipeline.assemble import assemble_clips
    calls = _mock_judge(monkeypatch, lambda user: _clean_verdict(9))
    sents, units, st = _severed_structure()
    specs, _notes, rejections = assemble_clips(st, "", sents, "u", "v", _sev_settings(),
                                               SevAdapter())
    assert len(specs) == 1                                  # the pair shipped as ONE clip
    m = specs[0]
    assert m["unit_ids"] == ["u_setup", "u_result"]
    assert m["sentence_start_idx"] == 0 and m["sentence_end_idx"] == 2   # hole absorbed
    assert "merged_severed_pair" in m["warnings"]
    assert "merged_overlap" not in m["warnings"]            # disjoint join, not an overlap
    # the fresh clean verdict covers the union text — the hash is refreshed (Wave-1 rule)
    assert m["judged_text_hash"] == judged_text_hash("sentence 0. sentence 1. sentence 2.")
    assert m["ship_flagged"] is False and m["judge_error"] is False
    assert m["sequence_index"] == 1 and m["prerequisite_clips"] == []
    assert calls["n"] == 3 and calls["union"] == 1          # 2 repair judges + 1 union probe
    assert rejections == []


def test_severed_merge_failing_rejudge_keeps_linked_pair(monkeypatch):
    from backend.pipeline.assemble import assemble_clips

    def bad(user):
        return JudgeVerdict(reasoning="union bad", score_10=3, understandable=False,
                            topic_identifiable=False)
    calls = _mock_judge(monkeypatch, bad)
    sents, units, st = _severed_structure()
    specs, _notes, rejections = assemble_clips(st, "", sents, "u", "v", _sev_settings(),
                                               SevAdapter())
    assert len(specs) == 2                                  # merge probe failed → pair stands
    assert all(not s.get("merged") for s in specs)
    later = next(s for s in specs if s["sequence_index"] == 2)
    assert later["prerequisite_clips"] == [1]               # …but ALWAYS linked
    assert later["notes"] == ["continues clip 1"]
    assert calls["union"] == 1
    assert rejections == []                                 # a merge probe never kills


def test_severed_merge_outage_keeps_linked_pair(monkeypatch):
    from backend.pipeline.assemble import assemble_clips

    def boom(user):
        raise RuntimeError("outage on the union probe")
    _mock_judge(monkeypatch, boom)
    sents, units, st = _severed_structure()
    specs, _notes, rejections = assemble_clips(st, "", sents, "u", "v", _sev_settings(),
                                               SevAdapter())
    assert len(specs) == 2                                  # conservative: no merge on outage
    later = next(s for s in specs if s["sequence_index"] == 2)
    assert later["prerequisite_clips"] == [1]
    assert rejections == []


def test_severed_merge_below_quality_floor_keeps_linked_pair(monkeypatch):
    """I1 reconcile of a P4 review finding: a union that re-judges CLEAN but whose honestly
    recomputed final_quality lands below the quality_floor is NOT kept — the linked pair
    (which passed the floor individually) stands."""
    from backend.pipeline.assemble import assemble_clips

    class FloorAdapter(SevAdapter):
        def required_verdict_fields(self, role):
            return ["source_grounded"]

    def docked_union(user):                    # clean (is_complete) but reason-docked:
        # completeness 1 - 3*0.05 = 0.85 → quality 0.93 < floor 0.95; the two singles
        # score 0.99 each and pass the floor.
        from backend.pipeline.assemble.validate import FailureReason
        return JudgeVerdict(reasoning="ok", score_10=8, understandable=True,
                            failure_reasons=[FailureReason(kind="other", detail=f"nit {i}")
                                             for i in range(3)])
    calls = _mock_judge(monkeypatch, docked_union)
    sents, units, st = _severed_structure()
    settings = dict(_sev_settings(), quality_floor=0.95)
    specs, _notes, rejections = assemble_clips(st, "", sents, "u", "v", settings,
                                               FloorAdapter())
    assert calls["union"] == 1                              # the merge WAS probed…
    assert len(specs) == 2                                  # …but the pair stands
    assert all(not s.get("merged") for s in specs)
    later = next(s for s in specs if s["sequence_index"] == 2)
    assert later["prerequisite_clips"] == [1]
    assert later["notes"] == ["continues clip 1"]
    assert rejections == []                                 # a merge probe never kills


# ── W25-F acceptance shape: a practice clip ships standalone + LINKS to its solution ──
def test_practice_prompt_clip_ships_standalone_and_links_to_solution(monkeypatch):
    """practice_prompt is an opener role now (W25-F): the prompt clip (deliberately
    answer-free) ships on its own and the solution clip — whose required prompt element
    fell to the closure span budget (referential, not inlined) — is LINKED back to it.
    The merge probe is cap-gated (union 59.9s > 30s), so the pair must stay two clips."""
    from backend import config
    from backend.adapters.base import BaseAdapter
    from backend.pipeline.assemble import assemble_clips
    from backend.pipeline.assemble.context_card import ContextCardDraft
    monkeypatch.setattr(config, "JUDGE_MODEL", "")
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "same")

    def fake(system, user, schema, **kw):
        if schema is ContextCardDraft:                      # card pass: nothing useful
            return ContextCardDraft()
        assert schema is JudgeVerdict, f"unexpected schema {schema}"
        return _clean_verdict(9)
    monkeypatch.setattr(llm_mod, "llm_json", fake)

    sents = mini_sents(6)
    roles = ["practice_prompt"] + ["transition"] * 4 + ["solution"]
    units = _sev_units(sents, roles)
    st = Structure(video_id="v", units=units, dependencies=DependencyGraph(),
                   content_map=ContentMap(nodes=[ContentNode(node_id="video", level="video",
                                                             sentence_range=(0, 5))]))
    settings = {"min_clip_duration_s": 1.0, "max_clip_duration_s": 30.0, "tail_pad_s": 0.0,
                "min_comprehension_score": 0.7, "quality_floor": 0.0, "max_clips": 12,
                "max_anchors": 2, "closure_max_span_s": 30.0, "anchor_selector": "priority"}
    specs, _notes, rejections = assemble_clips(st, "", sents, "u", "v", settings,
                                               BaseAdapter())
    assert len(specs) == 2                                  # prompt + solution, NOT merged
    prompt, solution = specs
    assert prompt["role"] == "practice_prompt"
    assert prompt["unit_ids"] == ["u0000"]                  # never grown into the answer
    assert solution["role"] == "solution"
    assert solution["prerequisite_clips"] == [1]            # …but linked back to the prompt
    assert "continues clip 1" in (solution.get("notes") or [])
    assert rejections == []


def test_severed_merge_skipped_over_ship_cap(monkeypatch):
    from backend.pipeline.assemble import assemble_clips
    calls = _mock_judge(monkeypatch, lambda user: _clean_verdict(9))
    sents, units, st = _severed_structure()
    specs, _notes, rejections = assemble_clips(st, "", sents, "u", "v",
                                               _sev_settings(max_dur=25.0), SevAdapter())
    assert len(specs) == 2                                  # 29.9s union > 25s cap → no probe
    assert calls["union"] == 0 and calls["n"] == 2          # zero extra judge spend
    later = next(s for s in specs if s["sequence_index"] == 2)
    assert later["prerequisite_clips"] == [1]               # the link is still ALWAYS added
    assert later["notes"] == ["continues clip 1"]
    assert rejections == []
