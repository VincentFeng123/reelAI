"""A-budget-clamp: the effective closure/repair span budget is
min(closure_max_span_s, max_clip_duration_s) — the judge must only ever score
spans the cutter can actually ship. Pure/offline (llm_json mocked)."""
from __future__ import annotations

import threading

import backend.llm as llm_mod
import backend.pipeline.assemble.candidates as candidates_mod
import backend.pipeline.assemble.validate as validate_mod
from backend.pipeline.assemble.candidates import build_candidate
from backend.pipeline.assemble.graph import Graph
from backend.pipeline.assemble.types import Candidate
from backend.pipeline.assemble.validate import JudgeVerdict, validate_and_repair

from .conftest import FakeAdapter, mini_sents, mini_units


class AnchorAdapter(FakeAdapter):
    def is_anchor_role(self, role):
        return True

    def anchor_priority(self, role):
        return 0.9

    def facet_for(self, role):
        return "other"


# ── ClosureBudget construction (candidates.build_candidate) ──────────────────
def _built_budget(monkeypatch, settings):
    """Run build_candidate with a spy on compute_closure; return the ClosureBudget it got."""
    sents = mini_sents(6)
    units = mini_units(sents)
    units_by_id = {u.unit_id: u for u in units}
    seen = {}
    real = candidates_mod.compute_closure

    def spy(anchor, graph, ubi, adapter, us, budget):
        seen["budget"] = budget
        return real(anchor, graph, ubi, adapter, us, budget)

    monkeypatch.setattr(candidates_mod, "compute_closure", spy)
    cand = build_candidate(units[0], Graph([], units), AnchorAdapter(), units, units_by_id,
                           sents, {u.unit_id: 1.0 for u in units}, settings)
    assert cand is not None
    return seen["budget"]


def test_closure_budget_clamped_to_ship_cap(monkeypatch):
    b = _built_budget(monkeypatch, {"closure_max_span_s": 300.0, "max_clip_duration_s": 240.0})
    assert b.max_span_s == 240.0


def test_closure_budget_smaller_closure_span_wins(monkeypatch):
    b = _built_budget(monkeypatch, {"closure_max_span_s": 200.0, "max_clip_duration_s": 240.0})
    assert b.max_span_s == 200.0


# ── repair expansion cap (validate.validate_and_repair → expand_candidate) ───
def _run_repair(monkeypatch, settings, cap):
    """Anchor at t≈590–600 with far referential prereqs at 300 s and 400 s. The repair
    loop must pass expand_candidate the clamped cap and never grow past it."""
    from backend import config
    monkeypatch.setattr(config, "JUDGE_MODEL", "")
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "same")
    # failing-but-repairable verdict: low score, missing prerequisites, hard core intact
    monkeypatch.setattr(llm_mod, "llm_json",
                        lambda *a, **kw: JudgeVerdict(reasoning="gap", score_10=3,
                                                      understandable=False,
                                                      prerequisites_satisfied=False))
    sents = mini_sents(60)                       # 600 s of material, sentence i ≈ [10i, 10i+10)
    units = mini_units(sents)
    units_by_id = {u.unit_id: u for u in units}

    caps_seen: list[float] = []
    real_expand = validate_mod.expand_candidate

    def spy(cand, verdict, graph, us, ubi, intro, sentences, max_span_s, adapter=None):
        caps_seen.append(max_span_s)             # W25-F: expand_candidate now takes the adapter
        out = real_expand(cand, verdict, graph, us, ubi, intro, sentences, max_span_s,
                          adapter=adapter)
        if out is not None:                      # every expansion result must be shippable
            assert out.end - out.start <= cap
        return out

    monkeypatch.setattr(validate_mod, "expand_candidate", spy)

    cand = Candidate(cand_id="c0", anchor_id="u0059", role="explanation", facet="other",
                     title="t", reason="r", unit_ids=["u0059"],
                     referential=[("u0040", "prerequisite"), ("u0030", "prerequisite")],
                     i_start=59, i_end=59, start=sents[59].start, end=sents[59].end)
    kept, _rej = validate_and_repair(cand, sents, Graph([], units), units, units_by_id, {},
                                     FakeAdapter(), settings, lambda s, e: "", "topic",
                                     {}, threading.Lock())
    assert caps_seen, "repair loop never attempted expansion"
    assert all(c == cap for c in caps_seen)
    if kept is not None:
        assert kept.end - kept.start <= cap
    return kept


def test_repair_expansion_capped_at_ship_cap(monkeypatch):
    # closure allows 300 s but only 240 s is shippable → 240 governs; u0030 (span 299.9 s)
    # must NOT be pulled in even though it fits the unclamped closure budget.
    kept = _run_repair(monkeypatch, {"closure_max_span_s": 300.0, "max_clip_duration_s": 240.0,
                                     "min_comprehension_score": 0.7}, cap=240.0)
    assert kept is not None
    assert "u0030" not in kept.unit_ids


def test_repair_expansion_smaller_closure_span_wins(monkeypatch):
    _run_repair(monkeypatch, {"closure_max_span_s": 200.0, "max_clip_duration_s": 240.0,
                              "min_comprehension_score": 0.7}, cap=200.0)
