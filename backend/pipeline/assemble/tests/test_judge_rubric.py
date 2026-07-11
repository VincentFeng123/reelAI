"""G-Eval judge: reasoning-first schema, 1-10 normalization, honest failure verdict,
repair-loop ship-but-flag. All offline (llm_json monkeypatched)."""
from __future__ import annotations

import threading

import pytest

import backend.llm as llm_mod
from backend.pipeline.assemble.graph import Graph
from backend.pipeline.assemble.types import Candidate
from backend.pipeline.assemble.validate import (
    JUDGE_SYSTEM, JudgeVerdict, is_complete, judge_clip, validate_and_repair,
)

from .conftest import FakeAdapter, mini_sents, mini_units


# ── schema ────────────────────────────────────────────────────────────────────
def test_reasoning_is_first_field():
    assert next(iter(JudgeVerdict.model_fields)) == "reasoning"


def test_score10_normalizes_to_score():
    assert JudgeVerdict(score_10=8).score == pytest.approx(0.8)
    assert JudgeVerdict(score_10=10).score == pytest.approx(1.0)


def test_score10_clamps_out_of_range():
    assert JudgeVerdict(score_10=12).score == pytest.approx(1.0)
    assert JudgeVerdict(score_10=-3).score == pytest.approx(0.1)


def test_score10_zero_keeps_legacy_score():
    assert JudgeVerdict(score=0.65).score == pytest.approx(0.65)   # no score_10 emitted


# ── prompt ────────────────────────────────────────────────────────────────────
def test_prompt_rubric_contents():
    for kind in ("unresolved_reference", "missing_prerequisite", "missing_visual",
                 "missing_problem_statement", "missing_reasoning", "missing_result",
                 "not_source_grounded", "off_topic", "other"):
        assert kind in JUDGE_SYSTEM
    assert "score_10" in JUDGE_SYSTEM
    assert "1-2" in JUDGE_SYSTEM and "9-10" in JUDGE_SYSTEM        # anchored bands
    assert "First write `reasoning`" in JUDGE_SYSTEM               # CoT-before-verdict


# ── failure path ──────────────────────────────────────────────────────────────
def test_fallback_verdict_on_llm_failure(monkeypatch):
    from backend import config
    monkeypatch.setattr(config, "JUDGE_MODEL", "")
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "same")
    def boom(*a, **kw):
        raise RuntimeError("api down")
    monkeypatch.setattr(llm_mod, "llm_json", boom)
    v = judge_clip("some clip text", "explanation", FakeAdapter())
    assert v.error is True
    assert v.understandable is False
    assert v.score == 0.0
    assert v.source_grounded is False                # unjudged never earns grounding credit
    assert v.topic_identifiable is False and v.all_references_resolved is False


def test_grounding_score_not_inflated_on_error(monkeypatch):
    from backend.pipeline.assemble.scoring import grounding_score
    sents = mini_sents(2)
    units = mini_units(sents)
    cand = _mk_candidate(sents)
    cand.verdict = JudgeVerdict(error=True, understandable=False, score=0.0,
                                source_grounded=False)
    score = grounding_score(cand, {u.unit_id: u for u in units})
    assert score == pytest.approx(0.6)               # 1.0 confidence × 0.6 ungrounded multiplier


def test_successful_parse_forces_error_false(monkeypatch):
    def fake(*a, **kw):
        return JudgeVerdict(reasoning="ok", score_10=9, understandable=True, error=True)
    monkeypatch.setattr(llm_mod, "llm_json", fake)
    v = judge_clip("text", "explanation", FakeAdapter())
    assert v.error is False                                        # LLM cannot self-flag error


def test_is_complete_false_on_error():
    v = JudgeVerdict(error=True, score=1.0, understandable=True)
    assert is_complete(v, "explanation", FakeAdapter(), min_score=0.7) is False


# ── repair loop: ship-but-flag ────────────────────────────────────────────────
def _mk_candidate(sents):
    return Candidate(cand_id="c0", anchor_id="u0000", role="explanation", facet="other",
                     title="t", reason="r", unit_ids=["u0000"], referential=[],
                     i_start=0, i_end=0, start=sents[0].start, end=sents[0].end)


def test_repair_returns_candidate_on_error_without_burning_budget(monkeypatch):
    from backend import config
    monkeypatch.setattr(config, "JUDGE_MODEL", "")
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "same")
    calls = {"n": 0}

    def boom(*a, **kw):
        calls["n"] += 1
        raise RuntimeError("api down")
    monkeypatch.setattr(llm_mod, "llm_json", boom)

    sents = mini_sents(3)
    units = mini_units(sents)
    units_by_id = {u.unit_id: u for u in units}
    cache: dict = {}
    cand, rej = validate_and_repair(
        _mk_candidate(sents), sents, Graph([], units), units, units_by_id, {},
        FakeAdapter(), {}, lambda s, e: "", "topic", cache, threading.Lock())
    assert cand is not None                                        # ship-but-flag: kept
    assert cand.verdict.error is True
    assert calls["n"] == 1                                         # ONE judge attempt, no repair spin
    assert cache == {}                                             # error verdicts are never cached
    assert rej is None


def test_cross_model_failure_retries_on_authoring_model(monkeypatch):
    from backend import config
    monkeypatch.setattr(config, "JUDGE_MODEL", "gemini-2.5-flash-lite")
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "same")
    calls = {"n": 0}

    def flaky(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("cross-model down")
        return JudgeVerdict(reasoning="ok", score_10=8, understandable=True)
    monkeypatch.setattr(llm_mod, "llm_json", flaky)
    v = judge_clip("text", "explanation", FakeAdapter())
    assert calls["n"] == 2                       # authoring-model retry happened
    assert v.error is False and v.score == pytest.approx(0.8)


# ── ship-but-flag scoring + warning plumbing ─────────────────────────────────
def test_completeness_neutral_on_error():
    from backend.pipeline.assemble.scoring import completeness_score

    class Contracty(FakeAdapter):
        def required_verdict_fields(self, role):
            return ["result_complete"]
    v = JudgeVerdict(error=True)
    assert completeness_score(v, "explanation", Contracty()) == pytest.approx(0.5)


def test_boundary_score_penalizes_unjudged():
    from backend.pipeline.assemble.scoring import boundary_score
    assert boundary_score(["unjudged"]) == pytest.approx(0.85)
    assert boundary_score([]) == pytest.approx(1.0)


def test_snap_flags_and_warns_unjudged():
    from backend.pipeline.assemble.boundary_adapt import snap_candidates

    sents = mini_sents(3)
    ok = _mk_candidate(sents)
    ok.verdict = JudgeVerdict(score_10=9, understandable=True)
    bad = _mk_candidate(sents)
    bad.cand_id, bad.i_start, bad.i_end = "c1", 1, 2
    bad.start, bad.end = sents[1].start, sents[2].end
    bad.verdict = JudgeVerdict(error=True)

    specs, _rej = snap_candidates([ok, bad], sents, {"min_clip_duration_s": 1.0,
                                                    "max_clip_duration_s": 500.0})
    flagged = {s["cand_id"]: s for s in specs}
    assert flagged["c1"]["judge_error"] is True
    assert "unjudged" in (flagged["c1"].get("warnings") or [])
    assert flagged["c0"]["judge_error"] is False
    assert "unjudged" not in (flagged["c0"].get("warnings") or [])
