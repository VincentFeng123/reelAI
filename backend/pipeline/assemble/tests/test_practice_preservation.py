"""W25-F — practice-clip preservation: repair expansion is gated by the BOUND contract.

Root cause under test (spec §root-cause 13): expand_candidate keyed off verdict booleans
regardless of the bound contract — `pull_result()` fired on a practice_prompt-contract
clip whose contract DELIBERATELY excludes the answer (base.py: 'deliberately NO solution
element'), then choose_contract's specificity tie-break made the conversion permanent.
A practice clip should ship as a prompt + LINK to its solution (sequence.py), never be
grown into a solution clip. All offline (llm_json monkeypatched / pure functions).
"""
from __future__ import annotations

import threading
from dataclasses import replace

import backend.llm as llm_mod
from backend.adapters.base import BaseAdapter
from backend.pipeline.assemble.graph import Graph
from backend.pipeline.assemble.types import Candidate
from backend.pipeline.assemble.validate import (
    FailureReason, JudgeVerdict, expand_candidate, validate_and_repair,
)
from backend.pipeline.understand.models import Unit

from .conftest import mini_sents


def role_units(roles, sents):
    return [Unit(unit_id=f"u{i:04d}", start=sents[i].start, end=sents[i].end,
                 sentence_range=(i, i), role=r, transcript=sents[i].text)
            for i, r in enumerate(roles)]


def _mk_cand(units, sents, anchor_idx, unit_ids, contract_role=""):
    picked = [u for u in units if u.unit_id in set(unit_ids)]
    i0 = min(u.sentence_range[0] for u in picked)
    i1 = max(u.sentence_range[1] for u in picked)
    return Candidate(cand_id="c0", anchor_id=units[anchor_idx].unit_id,
                     role=units[anchor_idx].role, facet="other", title="t", reason="r",
                     unit_ids=list(unit_ids), referential=[], i_start=i0, i_end=i1,
                     start=sents[i0].start, end=sents[i1].end, contract_role=contract_role)


def _fixture():
    sents = mini_sents(4)
    units = role_units(["example_setup", "practice_prompt", "worked_step", "solution"], sents)
    return sents, units, {u.unit_id: u for u in units}


def _expand(cand, verdict, sents, units, units_by_id, adapter):
    return expand_candidate(cand, verdict, Graph([], units), units, units_by_id, {},
                            sents, 999.0, adapter=adapter)


# ── boolean fallback gating: result_complete=False is ADVISORY under practice_prompt ──
def test_boolean_result_pull_disabled_for_practice_prompt_contract():
    sents, units, units_by_id = _fixture()
    v = JudgeVerdict(score_10=4, result_complete=False)
    cand = _mk_cand(units, sents, 1, ["u0001"], contract_role="practice_prompt")
    # the answer the contract deliberately omits is NOT pulled — zero expansion targets
    assert _expand(cand, v, sents, units, units_by_id, BaseAdapter()) is None


def test_same_verdict_still_pulls_answer_under_solution_contract():
    # the gate is CONTRACT-scoped, not global: a solution-bound clip (whose contract
    # requires the answer, and whose required_verdict_fields include result_complete)
    # keeps the pre-W25-F growth behavior on the identical verdict.
    sents, units, units_by_id = _fixture()
    v = JudgeVerdict(score_10=4, result_complete=False)
    cand = _mk_cand(units, sents, 1, ["u0001"], contract_role="solution")
    grown = _expand(cand, v, sents, units, units_by_id, BaseAdapter())
    assert grown is not None and "u0003" in grown.unit_ids


def test_reasoning_pull_also_disabled_for_practice_prompt_contract():
    # the practice_prompt contract has no steps element either — reasoning_complete=False
    # must not drag worked_steps in (the other half of the conversion path).
    sents, units, units_by_id = _fixture()
    v = JudgeVerdict(score_10=4, reasoning_complete=False)
    cand = _mk_cand(units, sents, 1, ["u0001"], contract_role="practice_prompt")
    assert _expand(cand, v, sents, units, units_by_id, BaseAdapter()) is None


# ── kind-targeted hints obey the same gate ─────────────────────────────────────
def test_kind_hint_result_pull_gated_by_bound_contract():
    sents, units, units_by_id = _fixture()
    v = JudgeVerdict(score_10=4, failure_reasons=[
        FailureReason(kind="missing_result", detail="no answer", evidence_quote="x")])
    prompt_bound = _mk_cand(units, sents, 1, ["u0001"], contract_role="practice_prompt")
    assert _expand(prompt_bound, v, sents, units, units_by_id, BaseAdapter()) is None
    solution_bound = replace(prompt_bound, contract_role="solution")
    grown = _expand(solution_bound, v, sents, units, units_by_id, BaseAdapter())
    assert grown is not None and "u0003" in grown.unit_ids


# ── contract-free spans keep the pre-W25-F kind-agnostic behavior ─────────────
def test_contract_free_span_keeps_result_pull():
    # no adapter (and equally: adapters whose contract_for returns None) → nothing is
    # 'deliberately excluded', so the boolean fallback still grows toward the answer.
    sents, units, units_by_id = _fixture()
    v = JudgeVerdict(score_10=4, result_complete=False)
    cand = _mk_cand(units, sents, 1, ["u0001"])
    grown = expand_candidate(cand, v, Graph([], units), units, units_by_id, {}, sents, 999.0)
    assert grown is not None and "u0003" in grown.unit_ids


# ── integration: the repair loop never converts a practice clip into a solution clip ──
def test_repair_ships_practice_clip_without_the_answer(monkeypatch):
    from backend import config
    monkeypatch.setattr(config, "JUDGE_MODEL", "")
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "same")
    calls = {"texts": []}

    def fake(system, user, schema, **kw):
        assert schema is JudgeVerdict, "no kill-confirmation expected (phantom quote)"
        text = user.split("CLIP TRANSCRIPT:\n", 1)[1].rsplit("\n\nJudge whether", 1)[0].strip()
        calls["texts"].append(text)
        # the wild shape: the judge scores the answer-free prompt LOW and flags the
        # missing answer — hard core intact, evidence unquotable (absence-shaped).
        return JudgeVerdict(reasoning="question but no answer", score_10=4,
                            understandable=False, result_complete=False,
                            reasoning_complete=False,
                            failure_reasons=[FailureReason(kind="missing_result",
                                                           detail="answer never reached",
                                                           evidence_quote="ghost")])
    monkeypatch.setattr(llm_mod, "llm_json", fake)

    sents, units, units_by_id = _fixture()
    cand = _mk_cand(units, sents, 1, ["u0001"])
    kept, rej = validate_and_repair(cand, sents, Graph([], units), units, units_by_id, {},
                                    BaseAdapter(), {}, lambda s, e: "", "topic",
                                    {}, threading.Lock())
    assert rej is None and kept is not None                  # ships (hard core intact)
    assert kept.unit_ids == ["u0001"]                        # never grown into the answer
    assert kept.contract_role == "practice_prompt"           # …so the binding never flips
    assert calls["texts"] == ["sentence 1."]                 # zero growth probes judged
