"""P3a arc detection — deterministic grammar scan + cross-distance practice pairing +
the optional single-call verification (MathNet pattern). All offline; the ONLY LLM entry
point (verify_arcs) is exercised with llm_json monkeypatched."""
from __future__ import annotations

import pytest

import backend.llm as llm_mod
from backend.pipeline.understand.arcs import (
    ArcCheckLLM, ArcVerifyLLM, detect_arcs, verify_arcs,
)
from backend.pipeline.understand.models import Unit


def _units(roles, concepts=None, needs=None):
    """One 10s unit per role; concepts/needs are {index: [..]} maps."""
    concepts = concepts or {}
    needs = needs or {}
    return [Unit(unit_id=f"u{i:04d}", start=i * 10.0, end=i * 10.0 + 9.9,
                 sentence_range=(i, i), role=r, transcript=f"unit {i} text.",
                 summary=f"unit {i}",
                 concepts_introduced=list(concepts.get(i, [])),
                 concepts_required=list(needs.get(i, [])))
            for i, r in enumerate(roles)]


def _timed_units(rows, concepts=None, needs=None):
    """rows = [(role, duration_s[, lead_gap_s]), ...] laid out from t0 — for the W25-D
    substance-floor / locality fixtures where REAL durations AND gaps matter (the shared
    ``_units`` helper hardcodes 10s back-to-back members, which always clears the 12s
    floor and never exceeds the member-gap bound). Negative lead gaps model overlapping
    units (real structures have them — kinematics arc_1's gaps are all negative)."""
    concepts = concepts or {}
    needs = needs or {}
    units, t = [], 678.0                       # qP zone origin; any origin works
    for i, row in enumerate(rows):
        role, dur = row[0], row[1]
        t += row[2] if len(row) > 2 else 0.0   # optional lead gap before this unit
        units.append(Unit(unit_id=f"u{i:04d}", start=t, end=t + dur,
                          sentence_range=(i, i), role=role, transcript=f"unit {i} text.",
                          summary=f"unit {i}",
                          concepts_introduced=list(concepts.get(i, [])),
                          concepts_required=list(needs.get(i, []))))
        t += dur
    return units


def _by_id(units):
    return {u.unit_id: u for u in units}


# ── the audited kinematics shape: results labeled 'calculation' (u0021–u0025) ────────────
def test_kinematics_shape_last_step_fallback_detected():
    # docs/audits/2026-07-02/kinematics_e2e.json: u0021 example_setup (particle 100m E,
    # 150m W, 5s), u0022/u0025 worked_step, u0023/u0024 calculation — NO result/solution
    # unit; the complete example was unclippable pre-P3. Surrounding roles reproduced.
    # W25-D(a): the no-terminal fallback is now steps[-1] REGARDLESS of step role, so the
    # payoff is the last step u0005 (worked_step), not the last 'calculation' u0004.
    roles = ["definition",                                     # u0020-ish context
             "example_setup", "worked_step", "calculation", "calculation", "worked_step",
             "definition", "explanation", "variable_definition"]   # u0026-u0028-ish
    arcs = detect_arcs(_units(roles))
    assert len(arcs) == 1
    a = arcs[0]
    assert a.arc_role == "worked_example"
    assert a.unit_ids == ["u0001", "u0002", "u0003", "u0004", "u0005"]
    assert a.terminal_id == "u0005"            # the LAST step is the payoff (W25-D(a))
    assert a.terminal_role == "result"         # synthetic anchor role → worked-example contract
    assert a.calculation_as_final is False     # provenance flag tracks the terminal's real role
    assert a.arc_id == "arc_0"


def test_explicit_result_terminal():
    arcs = detect_arcs(_units(["example_setup", "worked_step", "result"]))
    assert len(arcs) == 1
    a = arcs[0]
    assert a.unit_ids == ["u0000", "u0001", "u0002"]
    assert a.terminal_id == "u0002" and a.terminal_role == "result"
    assert a.calculation_as_final is False


def test_multi_unit_setup_and_solution_terminal():
    # two setup units (example_setup + problem_givens) then steps then a solution unit
    arcs = detect_arcs(_units(["example_setup", "problem_givens", "derivation", "solution"]))
    assert len(arcs) == 1
    a = arcs[0]
    assert a.opener_ids == ["u0000", "u0001"]
    assert a.terminal_role == "solution"


# ── interleave tolerance ─────────────────────────────────────────────────────────────────
def test_two_interleaved_nonarc_units_tolerated():
    roles = ["example_setup", "worked_step", "transition", "exception", "worked_step", "result"]
    arcs = detect_arcs(_units(roles))
    assert len(arcs) == 1
    assert arcs[0].unit_ids == ["u0000", "u0001", "u0004", "u0005"]


def test_three_interleaves_truncate_the_arc_at_its_last_step():
    # W25-D update (spec governs): three NON-neutral interleaves still abort the scan
    # before the result, but the steps[-1] fallback (W25-D(a)) now closes the arc at the
    # last step instead of dropping it — the far result unit is NOT a member.
    roles = ["example_setup", "worked_step", "transition", "exception", "summary", "result"]
    arcs = detect_arcs(_units(roles))
    assert len(arcs) == 1
    assert arcs[0].unit_ids == ["u0000", "u0001"]
    assert arcs[0].terminal_id == "u0001" and arcs[0].terminal_role == "result"
    assert arcs[0].calculation_as_final is False


def test_three_interleaves_still_close_on_calculation_as_final():
    # W25-D update: 'explanation' is a NEUTRAL member now, so the third interleave is
    # 'summary' to keep this an abort-then-fallback fixture (same assertions as before).
    roles = ["example_setup", "calculation", "transition", "exception", "summary", "result"]
    arcs = detect_arcs(_units(roles))
    assert len(arcs) == 1
    assert arcs[0].unit_ids == ["u0000", "u0001"]
    assert arcs[0].terminal_id == "u0001" and arcs[0].calculation_as_final is True


# ── grammar negatives ────────────────────────────────────────────────────────────────────
def test_opener_directly_to_result_is_not_an_arc():
    assert detect_arcs(_units(["example_setup", "result"])) == []


def test_steps_without_opener_are_not_an_arc():
    assert detect_arcs(_units(["worked_step", "calculation", "result"])) == []


def test_worked_steps_only_close_via_last_step_fallback():
    # W25-D update (spec governs): pre-W25-D this shape was invisible (fallback fired for
    # literal 'calculation' only); steps[-1] now closes it regardless of step role.
    arcs = detect_arcs(_units(["example_setup", "worked_step", "worked_step"]))
    assert len(arcs) == 1
    assert arcs[0].unit_ids == ["u0000", "u0001", "u0002"]
    assert arcs[0].terminal_id == "u0002" and arcs[0].terminal_role == "result"
    assert arcs[0].calculation_as_final is False


def test_new_opener_after_steps_closes_and_restarts():
    roles = ["example_setup", "worked_step", "calculation",
             "example_setup", "worked_step", "result"]
    arcs = detect_arcs(_units(roles))
    assert [a.unit_ids for a in arcs] == [["u0000", "u0001", "u0002"],
                                          ["u0003", "u0004", "u0005"]]
    assert arcs[0].calculation_as_final is True and arcs[1].terminal_role == "result"


def test_bus_problem_slice_from_audit_yields_two_arcs():
    # u0043–u0060 role sequence from the kinematics audit (transition, example_setup, setup,
    # claim, procedure, steps…, problem_givens, step, result, part-b setup…result).
    roles = ["transition", "example_setup", "setup", "claim", "procedure",
             "worked_step", "worked_step", "calculation", "worked_step",
             "problem_givens", "worked_step", "result",
             "example_setup", "problem_givens", "physical_interpretation", "procedure",
             "worked_step", "result"]
    arcs = detect_arcs(_units(roles))
    assert [a.unit_ids for a in arcs] == [
        ["u0009", "u0010", "u0011"],                       # givens → step → result
        ["u0012", "u0013", "u0016", "u0017"],              # part b (≤2 interleave)
    ]
    assert all(a.arc_role == "worked_example" for a in arcs)


# ── W25-D(b): CLOSER terminals — payoff-adjacent roles end the example, gated on steps ───
@pytest.mark.parametrize("closer", ["claim", "physical_interpretation",
                                    "graph_interpretation", "unit_check"])
def test_closer_terminal_accepted_only_after_steps(closer):
    arcs = detect_arcs(_units(["example_setup", "worked_step", closer]))
    assert len(arcs) == 1
    a = arcs[0]
    assert a.unit_ids == ["u0000", "u0001", "u0002"]
    assert a.terminal_id == "u0002" and a.terminal_role == "result"
    assert a.arc_role == "worked_example" and a.calculation_as_final is False


@pytest.mark.parametrize("closer", ["claim", "physical_interpretation",
                                    "graph_interpretation", "unit_check"])
def test_closer_without_steps_is_not_a_terminal(closer):
    # opener straight into a closer is not an extraction event (the `if steps:` guard);
    # with zero steps the fallback has nothing to close on either → no arc.
    assert detect_arcs(_units(["example_setup", closer])) == []


def test_closer_before_steps_does_not_close_the_arc():
    # a pre-step claim interleaves (1 of 2 tolerated); the real result still terminates
    roles = ["example_setup", "claim", "worked_step", "result"]
    arcs = detect_arcs(_units(roles))
    assert len(arcs) == 1
    assert arcs[0].unit_ids == ["u0000", "u0002", "u0003"]   # claim is NOT a member
    assert arcs[0].terminal_id == "u0003"


# ── W25-D(c): interpretation/explanation-family units are NEUTRAL, not interleave ────────
def test_neutral_digression_does_not_abort_a_real_example():
    # 3 interpretation-family units between the steps and the result would have aborted
    # the scan pre-W25-D (interleave > 2); neutral units are transparent and NOT members.
    roles = ["example_setup", "worked_step", "explanation", "intuition",
             "diagram_interpretation", "worked_step", "result"]
    arcs = detect_arcs(_units(roles))
    assert len(arcs) == 1
    assert arcs[0].unit_ids == ["u0000", "u0001", "u0005", "u0006"]   # neutrals excluded
    assert arcs[0].terminal_id == "u0006" and arcs[0].terminal_role == "result"


def test_neutral_units_do_not_reset_the_interleave_budget():
    # transparent means NO reset either: transition(1), explanation(neutral),
    # transition(2), exception(3) → abort; the fallback truncates at the last step.
    roles = ["example_setup", "worked_step", "transition", "explanation",
             "transition", "exception", "result"]
    arcs = detect_arcs(_units(roles))
    assert len(arcs) == 1
    assert arcs[0].unit_ids == ["u0000", "u0001"]
    assert arcs[0].terminal_id == "u0001"


# ── W25-D review fixes: provisional closers + the MAX_ARC_MEMBER_GAP_S locality bound ─────
@pytest.mark.parametrize("closer", ["claim", "physical_interpretation",
                                    "graph_interpretation", "unit_check"])
def test_closer_between_steps_does_not_truncate_the_arc(closer):
    # Review finding 2: breaking hard at the FIRST closer after any step orphaned later
    # steps and the TRUE terminal (qP: the 700.9s arc was cut to 1 step at u0115). A
    # mid-example closer is PROVISIONAL — the later step demotes it and the real result
    # terminates the full arc; the demoted closer is not a member (old interleave shape).
    roles = ["example_setup", "worked_step", closer, "worked_step", "result"]
    arcs = detect_arcs(_units(roles))
    assert len(arcs) == 1
    a = arcs[0]
    assert a.terminal_id == "u0004" and a.terminal_role == "result"
    assert a.step_ids == ["u0001", "u0003"]
    assert a.unit_ids == ["u0000", "u0001", "u0003", "u0004"]   # u0002 demoted, not member


def test_qp_700s_shape_closer_demoted_then_last_step_fallback():
    # the REAL qP u0112-u0119 role sequence (deterministic replay of the cached
    # structure): the graph_interpretation at index 3 (u0115) no longer terminates the
    # arc — the scan continues through 3 more worked_steps and the new practice_prompt
    # (u0120) closes it at steps[-1], yielding the 4-step 700.9-738.8 arc.
    roles = ["practice_prompt", "irrelevant", "worked_step", "graph_interpretation",
             "worked_step", "irrelevant", "worked_step", "worked_step", "practice_prompt"]
    arcs = detect_arcs(_units(roles))
    assert len(arcs) == 1
    assert arcs[0].unit_ids == ["u0000", "u0002", "u0004", "u0006", "u0007"]
    assert arcs[0].terminal_id == "u0007" and len(arcs[0].step_ids) == 4


def test_pending_closer_finalizes_on_abort():
    # closer, then 2 more non-neutral interleaves → abort: the arc closes AT the pending
    # closer (it stands after the last step) — not the steps[-1] fallback, and the far
    # result past the abort is not a member.
    roles = ["example_setup", "worked_step", "claim", "transition", "exception", "result"]
    arcs = detect_arcs(_units(roles))
    assert len(arcs) == 1
    assert arcs[0].terminal_id == "u0002" and arcs[0].terminal_role == "result"
    assert arcs[0].unit_ids == ["u0000", "u0001", "u0002"]


def test_pending_closer_finalizes_on_new_opener_restart():
    roles = ["example_setup", "worked_step", "physical_interpretation",
             "example_setup", "worked_step", "result"]
    arcs = detect_arcs(_units(roles))
    assert [a.unit_ids for a in arcs] == [["u0000", "u0001", "u0002"],
                                          ["u0003", "u0004", "u0005"]]
    assert arcs[0].terminal_id == "u0002"      # the pending closer, not steps[-1]


def test_first_closer_after_the_last_step_wins_over_later_drift():
    # later closers never slide the terminal forward (crawl guard): the claim pends
    # first; the following unit_check stays neutral-transparent; scan end closes at u0002.
    roles = ["example_setup", "worked_step", "claim", "unit_check"]
    arcs = detect_arcs(_units(roles))
    assert len(arcs) == 1
    assert arcs[0].terminal_id == "u0002"
    assert "u0003" not in arcs[0].unit_ids


def test_distant_closer_is_rejected_and_the_last_step_closes():
    # Review finding 3 (terminal hop): a physical_interpretation 35s past the last step
    # is not that example's payoff — closer joins are bounded by MAX_ARC_MEMBER_GAP_S
    # (the kinematics phantom rode a 69s step→closer hop). Rejected closers stay
    # neutral-transparent; the arc closes at steps[-1].
    rows = [("example_setup", 10.0), ("worked_step", 10.0),
            ("physical_interpretation", 8.0, 35.0)]          # 35s lead gap > 30s bound
    arcs = detect_arcs(_timed_units(rows))
    assert len(arcs) == 1
    assert arcs[0].terminal_id == "u0001"
    assert "u0002" not in arcs[0].unit_ids


def test_closer_exactly_at_the_gap_bound_is_accepted():
    rows = [("example_setup", 10.0), ("worked_step", 10.0),
            ("claim", 8.0, 30.0)]                            # 30.0s == bound → still joins
    arcs = detect_arcs(_timed_units(rows))
    assert len(arcs) == 1 and arcs[0].terminal_id == "u0002"


def test_distant_opener_does_not_accumulate_kinematics_phantom_shape():
    # Review finding 3 (opener hop) — the REAL dHjWVlfNraM u0006-u0016 shape (times and
    # roles from the cached structure, deterministic replay): pre-fix the scan
    # accumulated u0010's example_setup onto the u0006 temperature prompt across a 73.5s
    # crawl (2 claims + 1 neutral explanation) and closed on the 69s-distant
    # physical_interpretation → a phantom 302s arc crossing unrelated claims/definitions.
    # Bounded joins: the prompt yields NO arc; the example opens its OWN arc and closes
    # at its last step — hull stays inside the real worked example.
    rows = [("practice_prompt", 10.0),          # u0006 temperature: scalar or vector?
            ("claim", 26.0, 2.0),               # u0007 temperature is a scalar
            ("explanation", 17.0, 1.0),         # u0008 (neutral — the pre-fix crawl hop)
            ("claim", 24.0, 2.0),               # u0009 acceleration is a vector
            ("example_setup", 9.0, 1.0),        # u0010 person travels 13 m east
            ("example_setup", 44.0, 2.0),       # u0011 then 4 m west
            ("worked_step", 47.0, 3.0),         # u0012 total distance = 17 m
            ("exception", 27.0, -2.0),          # u0013
            ("definition", 46.0, 1.0),          # u0014
            ("physical_interpretation", 45.0, -3.0),   # u0015 displacement +9 (69s hop)
            ("demonstration", 45.0, -2.0)]      # u0016 (third interleave → abort)
    arcs = detect_arcs(_timed_units(rows))
    assert len(arcs) == 1
    a = arcs[0]
    assert a.opener_ids == ["u0004", "u0005"]                # NOT the distant prompt
    assert a.unit_ids == ["u0004", "u0005", "u0006"]
    assert a.terminal_id == "u0006"                          # steps[-1]; far closer rejected


def test_member_gap_bound_is_the_config_lever(monkeypatch):
    from backend import config
    monkeypatch.setattr(config, "MAX_ARC_MEMBER_GAP_S", 120.0)
    rows = [("example_setup", 10.0), ("worked_step", 10.0),
            ("physical_interpretation", 8.0, 35.0)]
    arcs = detect_arcs(_timed_units(rows))
    assert len(arcs) == 1
    assert arcs[0].terminal_id == "u0002"      # the 35s hop is legal at a 120s bound


# ── W25-D acceptance shapes: the qP 678-883s graph-build zone + Socratic micro-pairs ──────
@pytest.mark.parametrize("terminal", ["claim", "physical_interpretation"])
def test_qp_graph_build_zone_arc_detected(terminal):
    # qP-9wwRrJbg item 16: a 205s worked example (678-883s) of 16 worked_steps whose
    # terminals were labeled claim/physical_interpretation — ZERO result/solution units
    # anywhere, so pre-W25-D the whole event was invisible to arc detection.
    rows = [("example_setup", 12.0)] + [("worked_step", 11.5)] * 16 + [(terminal, 9.0)]
    arcs = detect_arcs(_timed_units(rows))
    assert len(arcs) == 1
    a = arcs[0]
    assert a.arc_role == "worked_example"
    assert a.unit_ids == [f"u{i:04d}" for i in range(18)]    # opener + 16 steps + closer
    assert a.terminal_id == "u0017" and a.terminal_role == "result"
    assert len(a.step_ids) == 16


def test_socratic_micro_pair_not_detected():
    # a 3s prompt→solution pair (the qP sponsor-promo class): zero steps AND member
    # duration 3.0s < MIN_ARC_SUBSTANCE_S → killed at detection (W25-D(d)); downstream
    # is triple-protected (saturation-exempt / dropped-last / snap-padded), so it must
    # never leave this module.
    rows = [("practice_prompt", 1.2), ("explanation", 4.0), ("solution", 1.8)]
    assert detect_arcs(_timed_units(rows, concepts={0: ["slope"]},
                                    needs={2: ["slope"]})) == []


def test_substance_floor_keeps_pairs_at_or_above_it():
    rows = [("practice_prompt", 6.0), ("explanation", 4.0), ("solution", 6.0)]
    arcs = detect_arcs(_timed_units(rows, concepts={0: ["slope"]}, needs={2: ["slope"]}))
    assert len(arcs) == 1 and arcs[0].arc_role == "practice_pair"   # 12.0s == floor → kept


def test_substance_floor_exempts_arcs_with_steps():
    # a 30s-class worked example passes trivially; even a TINY one with real steps is
    # kept — the floor only gates zero-step arcs (a tight example beats no example).
    long_rows = [("example_setup", 6.0), ("worked_step", 12.0), ("worked_step", 12.0)]
    tiny_rows = [("example_setup", 2.0), ("worked_step", 2.0)]
    assert len(detect_arcs(_timed_units(long_rows))) == 1
    tiny = detect_arcs(_timed_units(tiny_rows))
    assert len(tiny) == 1 and tiny[0].step_ids == ["u0001"]  # 4.0s < floor, steps exempt


def test_substance_floor_is_the_config_lever(monkeypatch):
    from backend import config
    monkeypatch.setattr(config, "MIN_ARC_SUBSTANCE_S", 30.0)
    rows = [("practice_prompt", 6.0), ("explanation", 4.0), ("solution", 6.0)]
    assert detect_arcs(_timed_units(rows, concepts={0: ["slope"]},
                                    needs={2: ["slope"]})) == []   # 12s pair dies at 30s floor


# ── practice pairing across distance (shared concept) ────────────────────────────────────
def test_practice_pair_first_later_solution_sharing_a_concept():
    roles = ["practice_prompt", "explanation", "result", "claim", "explanation", "solution"]
    arcs = detect_arcs(_units(
        roles,
        concepts={0: ["momentum"], 2: ["energy"]},
        needs={5: ["Momentum"]}))                          # case-insensitive concept match
    assert len(arcs) == 1
    a = arcs[0]
    assert a.arc_role == "practice_pair"
    assert a.unit_ids == ["u0000", "u0005"]                # nearer non-sharing result skipped
    assert a.terminal_id == "u0005" and a.terminal_role == "solution"


def test_prompt_without_shared_concept_pairs_nothing():
    roles = ["practice_prompt", "explanation", "solution"]
    assert detect_arcs(_units(roles, concepts={0: ["momentum"], 2: ["energy"]})) == []


def test_prompt_consumed_as_arc_opener_is_not_paired_again():
    # prompt → step → solution is ONE grammar arc (practice_pair-shaped); no distance pair
    roles = ["practice_prompt", "worked_step", "solution", "solution"]
    arcs = detect_arcs(_units(roles, concepts={0: ["x"], 2: ["x"], 3: ["x"]}))
    assert len(arcs) == 1
    assert arcs[0].unit_ids == ["u0000", "u0001", "u0002"]
    assert arcs[0].arc_role == "practice_pair"             # prompt-opened, solution-terminated


def test_arc_ids_sequential():
    roles = ["example_setup", "worked_step", "result",
             "practice_prompt", "explanation", "solution"]
    arcs = detect_arcs(_units(roles, concepts={3: ["x"]}, needs={5: ["x"]}))
    assert [a.arc_id for a in arcs] == ["arc_0", "arc_1"]


# ── verification (one batched call; rule-checked ids; degrade-not-crash) ─────────────────
def _two_arcs():
    roles = ["example_setup", "worked_step", "result",
             "example_setup", "calculation", "result"]
    units = _units(roles)
    arcs = detect_arcs(units)
    assert len(arcs) == 2
    return arcs, _by_id(units)


def test_verify_confirms_and_drops_by_omission(monkeypatch):
    arcs, by_id = _two_arcs()
    calls = []

    def fake(system, user, schema, **kw):
        calls.append(schema)
        assert schema is ArcVerifyLLM
        return ArcVerifyLLM(arcs=[ArcCheckLLM(arc_id="arc_0", problem_ids=["u0000"],
                                              step_ids=["u0001"], answer_ids=["u0002"])])
    monkeypatch.setattr(llm_mod, "llm_json", fake)
    kept, note = verify_arcs(arcs, by_id, {"arc_verify": True})
    assert note is None and len(calls) == 1                # ONE batched call for all arcs
    assert [a.arc_id for a in kept] == ["arc_0"]           # arc_1 rejected by omission
    assert kept[0].verified is True


def test_verify_rule_checks_ids_inside_the_arc(monkeypatch):
    arcs, by_id = _two_arcs()

    def fake(system, user, schema, **kw):
        return ArcVerifyLLM(arcs=[
            # answer id points OUTSIDE arc_0 → discarded → no confirmed answer → dropped
            ArcCheckLLM(arc_id="arc_0", problem_ids=["u0000"], step_ids=["u0001"],
                        answer_ids=["u0005"]),
            ArcCheckLLM(arc_id="arc_1", problem_ids=["u0003", "u9999"], step_ids=["u0004"],
                        answer_ids=["u0005"]),             # stray u9999 discarded, arc still ok
        ])
    monkeypatch.setattr(llm_mod, "llm_json", fake)
    kept, note = verify_arcs(arcs, by_id, {"arc_verify": True})
    assert note is None
    assert [a.arc_id for a in kept] == ["arc_1"]


def test_verify_practice_pair_needs_no_steps(monkeypatch):
    roles = ["practice_prompt", "explanation", "solution"]
    units = _units(roles, concepts={0: ["x"]}, needs={2: ["x"]})
    arcs = detect_arcs(units)
    assert len(arcs) == 1 and arcs[0].arc_role == "practice_pair"

    def fake(system, user, schema, **kw):
        return ArcVerifyLLM(arcs=[ArcCheckLLM(arc_id="arc_0", problem_ids=["u0000"],
                                              step_ids=[], answer_ids=["u0002"])])
    monkeypatch.setattr(llm_mod, "llm_json", fake)
    kept, _ = verify_arcs(arcs, _by_id(units), {"arc_verify": True})
    assert len(kept) == 1 and kept[0].verified is True


def test_verify_llm_failure_keeps_arcs_unverified_with_note(monkeypatch):
    arcs, by_id = _two_arcs()

    def boom(*a, **kw):
        raise RuntimeError("api down")
    monkeypatch.setattr(llm_mod, "llm_json", boom)
    kept, note = verify_arcs(arcs, by_id, {"arc_verify": True})
    assert kept == arcs                                     # unverified-but-kept
    assert all(a.verified is False for a in kept)
    assert note is not None and "unverified" in note


def test_verify_flag_off_makes_no_llm_call(monkeypatch):
    arcs, by_id = _two_arcs()

    def boom(*a, **kw):                                     # any call would fail the test
        raise AssertionError("llm_json must not be called with arc_verify off")
    monkeypatch.setattr(llm_mod, "llm_json", boom)
    kept, note = verify_arcs(arcs, by_id, {"arc_verify": False})
    assert kept == arcs and note is None


def test_verify_no_arcs_short_circuits(monkeypatch):
    monkeypatch.setattr(llm_mod, "llm_json",
                        lambda *a, **kw: pytest.fail("no call expected"))
    assert verify_arcs([], {}, {"arc_verify": True}) == ([], None)
