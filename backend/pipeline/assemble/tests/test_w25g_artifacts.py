"""W25-G pipeline-side pieces: run-artifact persistence (write_run_artifacts), the
'build' Rejection stage for candidate builders returning None, the plan/arcs stats
threading in select_anchors_planned, and the kind-level phantom inputs
(Candidate.verified_kinds/unverified_kinds) riding into specs. All offline — the one
plan test monkeypatches llm_json like the other assemble tests."""
from __future__ import annotations

import json

import backend.llm as llm_mod
import backend.pipeline.assemble as assemble_mod
from backend.pipeline.assemble import assemble_clips
from backend.pipeline.assemble.artifacts import write_run_artifacts
from backend.pipeline.assemble.boundary_adapt import candidate_to_boundary_input
from backend.pipeline.assemble.candidates import select_anchors_planned
from backend.pipeline.assemble.integrity import Rejection
from backend.pipeline.assemble.plan import ExtractionPlanLLM, PlanItemLLM, PlanProposal
from backend.pipeline.assemble.types import Candidate
from backend.pipeline.assemble.validate import (
    FailureReason, JudgeVerdict, _attach_judge_stats,
)
from backend.adapters.base import BaseAdapter
from backend.pipeline.understand.arcs import ArcCandidate
from backend.pipeline.understand.models import (
    ContentMap, ContentNode, DependencyGraph, Structure, Unit,
)

from .conftest import FakeAdapter, mini_sents, mini_units


def _units(roles):
    return [Unit(unit_id=f"u{i:04d}", start=i * 10.0, end=i * 10.0 + 9.9,
                 sentence_range=(i, i), role=r, transcript=f"sentence {i}.",
                 summary=f"unit {i}") for i, r in enumerate(roles)]


def _structure(units, n_sents):
    nodes = [ContentNode(node_id="video", level="video", sentence_range=(0, n_sents - 1))]
    return Structure(video_id="v", units=units, dependencies=DependencyGraph(),
                     content_map=ContentMap(root_id="video", nodes=nodes))


# ── write_run_artifacts ───────────────────────────────────────────────────────
def test_write_run_artifacts_writes_all_four_files(tmp_path):
    specs = [{"start": 0.0, "end": 9.9, "warnings": ("extended_for_min_duration",),
              "unit_ids": ["u0000"], "verified_kinds": ("off_topic",)}]
    rejections = [Rejection(cand_id="c1", title="t", role="claim", stage="build",
                            reason="candidate build returned None (empty closure)",
                            unverified_kinds=("other",))]
    stats = {"plan_proposals": [PlanProposal("unit", "u0000", "claim", rank=0)],
             "arcs_verified": [ArcCandidate(arc_id="arc_0", arc_role="worked_example",
                                            unit_ids=["u0001", "u0002"])]}
    d = write_run_artifacts("vid", specs, rejections, stats, work_dir=tmp_path)
    assert d is not None and d.parent == tmp_path / "vid" / "runs"
    assert {p.name for p in d.iterdir()} == {"plan.json", "arcs.json",
                                             "shipped.json", "ledger.json"}
    plan = json.loads((d / "plan.json").read_text())
    assert plan[0]["ref_id"] == "u0000" and plan[0]["kind"] == "unit"
    arcs = json.loads((d / "arcs.json").read_text())
    assert arcs[0]["arc_id"] == "arc_0" and arcs[0]["unit_ids"] == ["u0001", "u0002"]
    shipped = json.loads((d / "shipped.json").read_text())
    assert shipped[0]["warnings"] == ["extended_for_min_duration"]   # tuple → list
    assert shipped[0]["verified_kinds"] == ["off_topic"]
    ledger = json.loads((d / "ledger.json").read_text())
    assert ledger[0]["stage"] == "build" and ledger[0]["unverified_kinds"] == ["other"]


def test_write_run_artifacts_missing_stats_keys_write_empty_lists(tmp_path):
    # the priority selector never fills plan_proposals/arcs_verified — files still exist
    d = write_run_artifacts("vid", [], [], {}, work_dir=tmp_path)
    assert json.loads((d / "plan.json").read_text()) == []
    assert json.loads((d / "arcs.json").read_text()) == []
    assert json.loads((d / "shipped.json").read_text()) == []
    assert json.loads((d / "ledger.json").read_text()) == []


def test_write_run_artifacts_serializes_defensively_and_never_raises(tmp_path):
    # unknown leaves degrade to str(); NaN → null; nothing raises
    stats = {"plan_proposals": [object()]}
    specs = [{"final_quality": float("nan"), "ids": frozenset({"a"})}]
    d = write_run_artifacts("vid", specs, [], stats, work_dir=tmp_path)
    plan = json.loads((d / "plan.json").read_text())
    assert isinstance(plan[0], str)
    shipped = json.loads((d / "shipped.json").read_text())
    assert shipped[0]["final_quality"] is None and shipped[0]["ids"] == ["a"]
    # an unusable work_dir (a FILE) returns None instead of raising into the pipeline
    f = tmp_path / "not_a_dir"
    f.write_text("x")
    assert write_run_artifacts("vid", [], [], {}, work_dir=f) is None


def test_write_run_artifacts_back_to_back_runs_get_distinct_dirs(tmp_path):
    d1 = write_run_artifacts("vid", [], [], {}, work_dir=tmp_path)
    d2 = write_run_artifacts("vid", [], [], {}, work_dir=tmp_path)
    assert d1 != d2 and d1.exists() and d2.exists()


# ── select_anchors_planned threads plan/arcs into the caller-owned stats ──────
def test_planned_selection_threads_proposals_and_verified_arcs(monkeypatch):
    units = _units(["example_setup", "worked_step", "calculation"])
    st = _structure(units, len(units))
    monkeypatch.setattr(llm_mod, "llm_json", lambda *a, **kw: ExtractionPlanLLM(
        extractions=[PlanItemLLM(arc_id="arc_0", role="result")]))
    stats: dict = {}
    anchors, _arc_map, _notes = select_anchors_planned(
        st, units, {u.unit_id: 1.0 for u in units}, BaseAdapter(),
        {"max_anchors": 12, "arc_verify": False}, "", stats=stats)
    assert anchors
    assert [a.arc_id for a in stats["arcs_verified"]] == ["arc_0"]
    assert [(p.kind, p.ref_id) for p in stats["plan_proposals"]] == [("arc", "arc_0")]


def test_planned_selection_fallback_records_empty_proposals(monkeypatch):
    units = _units(["definition", "explanation", "claim"])
    st = _structure(units, len(units))

    def boom(*a, **kw):
        raise RuntimeError("plan api down")
    monkeypatch.setattr(llm_mod, "llm_json", boom)
    stats: dict = {}
    select_anchors_planned(st, units, {u.unit_id: 1.0 for u in units}, BaseAdapter(),
                           {"max_anchors": 12, "arc_verify": False}, "", stats=stats)
    assert stats["plan_proposals"] == []       # failure → empty, never a missing key
    assert stats["arcs_verified"] == []        # no arcs in this unit table


# ── 'build' stage ledgering ───────────────────────────────────────────────────
class _AnchorAdapter(FakeAdapter):
    def is_anchor_role(self, role):
        return True

    def anchor_priority(self, role):
        return 0.9

    def facet_for(self, role):
        return "other"

    def valid_roles(self):
        return {"explanation"}


def test_build_death_is_ledgered_at_stage_build(monkeypatch):
    # a builder returning None used to vanish silently; now it lands in the drop ledger
    monkeypatch.setattr(assemble_mod, "build_candidate", lambda *a, **k: None)
    sents = mini_sents(4)
    units = mini_units(sents)
    st = Structure(video_id="v", units=units, dependencies=DependencyGraph(),
                   content_map=ContentMap(nodes=[ContentNode(
                       node_id="video", level="video", sentence_range=(0, 3))]))
    settings = {"min_clip_duration_s": 1.0, "max_clip_duration_s": 500.0,
                "min_comprehension_score": 0.7, "quality_floor": 0.0, "max_clips": 12,
                "max_anchors": 12, "closure_max_span_s": 999.0,
                "anchor_selector": "priority"}    # offline: priority path, no LLM at all
    specs, notes, rejections = assemble_clips(st, "", sents, "u", "v", settings,
                                              _AnchorAdapter())
    assert specs == []
    assert rejections and all(r.stage == "build" for r in rejections)
    r = rejections[0]
    assert r.cand_id.startswith("c_u") and "returned None" in r.reason
    assert r.role == "explanation" and r.end > r.start


# ── kind-level phantom inputs (Candidate → spec plumbing) ─────────────────────
def _cand(verdict=None):
    return Candidate(cand_id="c1", anchor_id="u0000", role="explanation", facet="other",
                     title="t", reason="r", unit_ids=["u0000"], referential=[],
                     i_start=0, i_end=0, start=0.0, end=9.9, verdict=verdict)


def test_attach_judge_stats_records_kind_tuples():
    sents = mini_sents(2)
    cand = _cand(JudgeVerdict(failure_reasons=[
        FailureReason(kind="off_topic", evidence_quote="sentence 0"),        # quotable + quoted
        FailureReason(kind="missing_prerequisite", evidence_quote="never spoken words")]))
    _attach_judge_stats(cand, sents)
    assert cand.n_failure_reasons == 2 and cand.n_verified == 1
    assert cand.verified_kinds == ("off_topic",)
    assert cand.unverified_kinds == ("missing_prerequisite",)


def test_candidate_to_boundary_input_carries_kind_tuples():
    cand = _cand()
    cand.verified_kinds = ("off_topic",)
    cand.unverified_kinds = ("missing_result",)
    b = candidate_to_boundary_input(cand)
    assert b["verified_kinds"] == ("off_topic",)
    assert b["unverified_kinds"] == ("missing_result",)
    # defaults stay empty tuples (legacy candidates without the fields)
    assert candidate_to_boundary_input(_cand())["verified_kinds"] == ()
