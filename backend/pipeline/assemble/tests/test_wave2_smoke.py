"""I1b: one offline smoke of the assembly path proving P1+P2+P3+P4 compose.

ONE fixture video drives assemble_clips end-to-end (llm_json mocked, zero network) and
exercises, in a single run:
  - P3: the extraction PLAN selects (ExtractionPlanLLM consumed, plan_engine == 'plan'),
        a detected ARC (calculation-as-final worked example) becomes a candidate and ships;
  - P1: the completeness contract is bound BY CONTENT — the arc clip is anchored 'result'
        (provenance) but judged/gated under the contract its actual roles satisfy;
  - P2: an off-topic verdict fires a TRIM (bisection over the lattice) and the trimmed
        sub-span ships with n_trims recorded;
  - P4: the dedupe tie-break decides a containment overlap on the hard-core-gates leg
        (the flagged fragment loses to the gate-passing arc), ledgered at stage 'dedupe';
plus the I1 stats/eval seam: assemble_clips fills the caller's stats dict and
run_eval._wave2_columns turns the run into the new per-video columns.
"""
from __future__ import annotations

import threading

import backend.llm as llm_mod
from backend import config
from backend.adapters.base import BaseAdapter
from backend.eval.run_eval import _wave2_columns
from backend.pipeline.assemble import assemble_clips
from backend.pipeline.assemble.validate import FailureReason, JudgeVerdict
from backend.pipeline.assemble.plan import ExtractionPlanLLM, PlanItemLLM
from backend.pipeline.understand.arcs import ArcCheckLLM, ArcVerifyLLM
from backend.pipeline.understand.models import (
    ContentMap, ContentNode, DependencyGraph, Structure, Unit,
)

from .conftest import mini_sents

_SETTINGS = {"min_clip_duration_s": 1.0, "max_clip_duration_s": 500.0, "tail_pad_s": 0.0,
             "min_comprehension_score": 0.7, "quality_floor": 0.0, "max_clips": 12,
             "max_anchors": 12, "closure_max_span_s": 999.0,
             "anchor_selector": "plan", "arc_verify": True,
             # this smoke pins the ROUND-0 plan seam (exact call counts / no fallback
             # note); Q1e refund rounds are exercised in test_selection_budget.py
             "refund_rounds": 0}

_ROLES = ["administrative",     # u0000 inert
          "example_setup",      # u0001 ┐ the arc (calculation-as-final: no result/solution
          "worked_step",        # u0002 │ unit exists — the audited kinematics shape)
          "transition",         # u0003 │   ← interleaved non-member, also proposed alone
          "calculation",        # u0004 ┘ terminal (re-roled 'result')
          "transition",         # u0005 inert
          "definition",         # u0006 the trim-demo anchor
          "explanation"]        # u0007 pulled by closure, trimmed back off


def _fixture():
    sents = mini_sents(8)
    units = [Unit(unit_id=f"u{i:04d}", start=s.start, end=s.end, sentence_range=(i, i),
                  role=_ROLES[i], transcript=s.text, summary=f"unit {i}")
             for i, s in enumerate(sents)]
    nodes = [ContentNode(node_id="video", level="video", sentence_range=(0, 7)),
             ContentNode(node_id="c0.t0", level="topic", parent_id="video",
                         sentence_range=(0, 4), start=0.0, end=49.9),
             ContentNode(node_id="c0.t1", level="topic", parent_id="video",
                         sentence_range=(5, 7), start=50.0, end=79.9)]
    st = Structure(video_id="v", units=units, dependencies=DependencyGraph(),
                   content_map=ContentMap(root_id="video", nodes=nodes))
    return sents, units, st


def _off_topic():
    # phantom evidence quote → the asymmetric gate can never confirm-kill in this smoke
    return JudgeVerdict(reasoning="drifts", score_10=3, understandable=False,
                        topic_identifiable=False,
                        failure_reasons=[FailureReason(kind="off_topic", detail="tail",
                                                       evidence_quote="the previous equation")])


def _mock(monkeypatch):
    monkeypatch.setattr(config, "JUDGE_MODEL", "")
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "same")
    lock = threading.Lock()
    calls = {"schemas": [], "judge_roles": {}}

    def fake(system, user, schema, **kw):
        with lock:
            calls["schemas"].append(schema)
        if schema is ArcVerifyLLM:
            return ArcVerifyLLM(arcs=[ArcCheckLLM(arc_id="arc_0", problem_ids=["u0001"],
                                                  step_ids=["u0002", "u0004"],
                                                  answer_ids=["u0004"])])
        if schema is ExtractionPlanLLM:
            return ExtractionPlanLLM(extractions=[
                PlanItemLLM(arc_id="arc_0", role="result", purpose="worked example"),
                PlanItemLLM(anchor_unit_id="u0003", role="claim", purpose="aside"),
                PlanItemLLM(anchor_unit_id="u0006", role="", purpose="term")])
        assert schema is JudgeVerdict, f"unexpected schema {schema} (kill-confirm forbidden)"
        text = user.split("CLIP TRANSCRIPT:\n", 1)[1].rsplit("\n\nJudge whether", 1)[0].strip()
        role = user.split("CLIP ROLE: ", 1)[1].split("\n", 1)[0].split(" (", 1)[0].strip()
        with lock:
            calls["judge_roles"][text] = role
        if "sentence 4." in text:                            # the arc (hull 1–4) is clean
            return JudgeVerdict(reasoning="ok", score_10=9, understandable=True)
        if "sentence 7." in text:                            # trim-demo full span fails…
            return _off_topic()
        if text in ("sentence 6.", "sentence 5. sentence 6."):
            return JudgeVerdict(reasoning="ok", score_10=9, understandable=True)
        return _off_topic()                                  # the contained u0003 fragment
    monkeypatch.setattr(llm_mod, "llm_json", fake)
    return calls


def test_smoke_plan_arc_rebind_trim_dedupe_compose(monkeypatch):
    calls = _mock(monkeypatch)
    sents, units, st = _fixture()
    stats: dict = {}
    specs, notes, rejections = assemble_clips(st, "", sents, "u", "v", _SETTINGS,
                                              BaseAdapter(), stats=stats)

    # ── P3: the plan selected; the detected arc shipped as one clip ────────────
    assert stats["plan_engine"] == "plan" and stats["n_arcs_detected"] == 1
    assert "plan-fallback" not in notes
    assert calls["schemas"].count(ExtractionPlanLLM) == 1
    assert calls["schemas"].count(ArcVerifyLLM) == 1
    assert [s["cand_id"] for s in specs] == ["c_arc_0", "c_u0006"]
    arc = specs[0]
    assert arc["arc_id"] == "arc_0"
    assert {"u0001", "u0002", "u0003", "u0004"} <= set(arc["unit_ids"])   # full hull, truthful
    assert arc["sequence_index"] == 1

    # ── P1: contract bound by CONTENT, not by the anchor role ─────────────────
    assert arc["role"] == "result"                           # anchor provenance preserved
    # setup+steps+calculation-as-final binds the worked-example 'result' contract (3/3
    # required elements beats procedure's 1/1 on specificity) — the audit's judge gates
    # (problem_statement/reasoning/result_complete) therefore apply to this clip.
    assert arc["contract_role"] == "result"
    arc_text = "sentence 1. sentence 2. sentence 3. sentence 4."
    assert calls["judge_roles"][arc_text] == "result"        # the judge got the SAME brief
    # …and the binding RE-ran on every trim probe (P1c: rebind before every judge pass)
    assert calls["judge_roles"]["sentence 6."] == "definition"

    # ── P2: the off-topic verdict fired a trim; the trimmed sub-span shipped ──
    trimmed = specs[1]
    assert trimmed["unit_ids"] == ["u0006"]                  # u0007 trimmed back off
    assert trimmed["sentence_start_idx"] == 6 and trimmed["sentence_end_idx"] == 6
    assert trimmed["n_trims"] == 1                           # one lattice probe judged
    assert arc["n_trims"] == 0
    assert "unverified_judge_concerns" not in (trimmed.get("warnings") or ())

    # ── P4: dedupe tie-break (hard-core-gates leg) ledgered the contained loser
    # W25-C: the real coverage floor also tops up sliver-covered c0.t1 (u0006 spans 33%
    # < MIN_NODE_COVERAGE) with u0007, whose candidate cleanly loses dedupe to the
    # shipped definition clip — shipped specs are unchanged, the ledger gains one row.
    assert len(rejections) == 2
    assert all(r.stage == "dedupe" for r in rejections)
    assert rejections[0].cand_id == "c_u0003"                # flagged fragment lost to the arc
    assert rejections[1].cand_id == "c_u0007"                # coverage top-up lost to c_u0006
    assert arc["hard_gates_ok"] is True

    # ── I1: the run rolls up into the new eval columns ─────────────────────────
    cols = _wave2_columns(st, specs, stats)
    assert cols["chapter_coverage"] == 1.0                   # both topic nodes have a clip
    assert cols["plan_engine"] == "plan" and cols["plan_fallback_rate"] == 0.0
    assert cols["n_arcs_detected"] == 1 and cols["n_arc_clips_shipped"] == 1
    assert cols["n_trims"] == 1
    assert cols["severed_pairs_linked"] == 0 and cols["severed_pairs_merged"] == 0


def test_unknown_anchor_selector_is_loud_and_degrades_to_priority(monkeypatch, capsys):
    """I1 reconcile of a P3 review finding: a typo'd selector must warn + note + record
    plan_engine='priority' — never a silent legacy fallback (and never a plan/verify call)."""
    calls = _mock(monkeypatch)
    sents, units, st = _fixture()
    stats: dict = {}
    settings = dict(_SETTINGS, anchor_selector="Plan")       # typo: case matters
    specs, notes, _rej = assemble_clips(st, "", sents, "u", "v", settings,
                                        BaseAdapter(), stats=stats)
    assert stats["plan_engine"] == "priority"
    assert "unknown anchor_selector 'Plan'" in notes
    assert "anchor_selector" in capsys.readouterr().err
    assert ExtractionPlanLLM not in calls["schemas"]         # plan machinery never invoked
    assert ArcVerifyLLM not in calls["schemas"]
    assert specs                                             # legacy selection still ships
