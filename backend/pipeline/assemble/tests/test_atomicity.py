"""W25-E — atomicity: judge criterion (single_idea / over_inclusion), trim routing,
same-facet dedupe union guards, and node-aware min-duration snap direction.

Root causes under test (spec §9/§10): JUDGE_SYSTEM's three steps probed ONLY missing
context, so verdicts never carried the kinds _trim_flavored needs — the built-and-tested
trim bisection had no input (n_trims=0). And same-facet dedupe unioned DISTINCT problems
(every worked-example-family role shares facet 'worked_example'): qP arcs 2+4+5 mashed
into one [382,480] spec, with NO max-duration re-check on unions. All offline
(llm_json monkeypatched / pure functions)."""
from __future__ import annotations

import threading

import backend.llm as llm_mod
from backend.pipeline.assemble.boundary_adapt import _dedupe_partb, _union_guard, snap_candidates
from backend.pipeline.assemble.graph import Graph
from backend.pipeline.assemble.types import Candidate
from backend.pipeline.assemble.validate import (
    JUDGE_SYSTEM, FailureReason, JudgeVerdict, _norm_kind, _trim_flavored, expand_candidate,
    validate_and_repair,
)
from backend.pipeline.understand.models import Unit

from .conftest import FakeAdapter, mini_sents, mini_units


# ── shared scaffolding (test_repair_rework patterns) ─────────────────────────
def _mk_cand(units, sents, anchor_idx=0, unit_ids=None):
    ids = unit_ids if unit_ids is not None else [u.unit_id for u in units]
    picked = [u for u in units if u.unit_id in set(ids)]
    i0 = min(u.sentence_range[0] for u in picked)
    i1 = max(u.sentence_range[1] for u in picked)
    return Candidate(cand_id="c0", anchor_id=units[anchor_idx].unit_id,
                     role=units[anchor_idx].role, facet="other", title="t", reason="r",
                     unit_ids=ids, referential=[], i_start=i0, i_end=i1,
                     start=sents[i0].start, end=sents[i1].end)


def _vr(cand, sents, units, adapter=None, cache=None):
    units_by_id = {u.unit_id: u for u in units}
    return validate_and_repair(cand, sents, Graph([], units), units, units_by_id, {},
                               adapter or FakeAdapter(), {}, lambda s, e: "",
                               "topic", cache if cache is not None else {}, threading.Lock())


def _transcript(user: str) -> str:
    return user.split("CLIP TRANSCRIPT:\n", 1)[1].rsplit("\n\nJudge whether", 1)[0].strip()


def _script(monkeypatch, decide):
    from backend import config
    monkeypatch.setattr(config, "JUDGE_MODEL", "")
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "same")
    calls = {"n": 0, "texts": []}

    def fake(system, user, schema, **kw):
        assert schema is JudgeVerdict, "no kill-confirmation expected in these scenarios"
        calls["n"] += 1
        calls["texts"].append(_transcript(user))
        return decide(_transcript(user))
    monkeypatch.setattr(llm_mod, "llm_json", fake)
    return calls


def _pass():
    return JudgeVerdict(reasoning="fine", score_10=9, understandable=True)


def _over_inclusion(quote: str):
    """The atomicity failure shape: hard core INTACT (the clip is understandable, it just
    bundles a second idea) + a quote-friendly reason — the second idea's opening sentence
    is verbatim in the span, so it can survive the asymmetric kill gate."""
    return JudgeVerdict(reasoning="bundles a second idea", score_10=4, understandable=False,
                        single_idea=False,
                        failure_reasons=[FailureReason(kind="over_inclusion",
                                                       detail="second problem starts",
                                                       evidence_quote=quote)])


# ── schema + prompt wiring ────────────────────────────────────────────────────
def test_single_idea_field_default_true():
    assert JudgeVerdict().single_idea is True            # additive: old verdicts stay atomic
    assert JudgeVerdict(single_idea=False).single_idea is False
    assert FailureReason(kind="over_inclusion").kind == "over_inclusion"


def test_prompt_gains_atomicity_step():
    assert "Evaluate in four steps" in JUDGE_SYSTEM      # step 4 = MORE than one idea?
    assert "over_inclusion" in JUDGE_SYSTEM              # in the exact-kind vocabulary
    assert "single_idea" in JUDGE_SYSTEM                 # the boolean is instructed
    # evidence_quote for over_inclusion = the second idea's opening sentence (quotable
    # by construction — unlike absence-shaped kinds it can survive verify+confirm)
    assert "opening sentence of the SECOND idea" in JUDGE_SYSTEM


def test_over_inclusion_kinds_route_to_trim():
    def v(kind):
        return JudgeVerdict(failure_reasons=[FailureReason(kind=kind, evidence_quote="x")])
    for kind in ("over_inclusion", "over-inclusion", "multiple_ideas", "not_atomic",
                 "atomicity"):
        assert _trim_flavored(v(kind)) is True           # excess content → trim bisection
    for kind in ("missing_result", "missing_prerequisite", "unresolved_reference"):
        assert _trim_flavored(v(kind)) is False          # absence kinds keep routing to grow
    # fuzzy normalization cannot collide with the exact-equality 'other' check
    assert _norm_kind("over_inclusion").strip("_") != "other"
    assert _trim_flavored(v("another")) is False         # 'other' matches by equality only


def test_over_inclusion_produces_no_grow_targets():
    # step-1 expansion hints must not fire on the over_inclusion kind (it is not a
    # missing-content complaint) — with all booleans intact there is nothing to grow.
    sents = mini_sents(3)
    units = mini_units(sents)
    cand = _mk_cand(units, sents)
    v = _over_inclusion("sentence 2.")
    assert expand_candidate(cand, v, Graph([], units), units,
                            {u.unit_id: u for u in units}, {}, sents, 999.0) is None


# ── trim bisection reachable via an over_inclusion verdict (n_trims>0 organically) ──
def test_over_inclusion_verdict_reaches_trim_bisection(monkeypatch):
    # the judge flags the span for bundling a second idea (opening sentence: 'sentence 2.');
    # any sub-span without it passes. The bisection must ship the largest passing prefix.
    sents = mini_sents(3)
    units = mini_units(sents)

    def decide(text):
        return _over_inclusion("sentence 2.") if "sentence 2." in text else _pass()
    calls = _script(monkeypatch, decide)
    kept, rej = _vr(_mk_cand(units, sents), sents, units)
    assert rej is None and kept is not None
    assert kept.unit_ids == ["u0000", "u0001"]           # second idea trimmed off
    assert kept.i_start == 0 and kept.i_end == 1
    assert kept.n_trims >= 1                             # the n_trims=0 root cause is unblocked
    assert kept.verdict.understandable is True and kept.ship_flagged is False
    assert "single_idea_unverified" not in kept.warnings
    assert calls["n"] == 3                               # native + 2 bisection probes


def _hard_core_partial():
    """Hard-core-good but NOT complete, with no excess-content reason (prereq gap):
    the bisection's known-good side that must never outrank a judged-COMPLETE span."""
    return JudgeVerdict(reasoning="prereq gap", score_10=3, understandable=False,
                        prerequisites_satisfied=False)


def test_bisection_not_fooled_by_hard_core_intact_over_inclusion(monkeypatch):
    # Oracle-inversion regression (review fix): atomicity failures leave the hard core
    # INTACT by design, so the old known-good predicate (is_complete OR _hard_core_ok)
    # treated every still-over-inclusive trial as good — lo advanced past the true
    # boundary and the judged-COMPLETE [u0000,u0001] was OVERWRITTEN by the still-failing
    # [u0000,u0001,u0002] (second idea's opening still inside, shipped clean-looking).
    # A VERIFIED excess reason must put the trial on the known-BAD side instead.
    sents = mini_sents(4)
    units = mini_units(sents)

    def decide(text):
        # any part of the second idea [u0002,u0003] in the span → still over-inclusive
        # (hard core intact); quote = the second idea's opening sentence, verbatim.
        if "sentence 2." in text or "sentence 3." in text:
            return _over_inclusion("sentence 2.")
        return _pass()
    calls = _script(monkeypatch, decide)
    kept, rej = _vr(_mk_cand(units, sents), sents, units)
    assert rej is None and kept is not None
    assert kept.unit_ids == ["u0000", "u0001"]           # the atomic sub-span ships…
    assert kept.i_start == 0 and kept.i_end == 1         # …not the larger failing trial
    assert kept.verdict.understandable is True and kept.ship_flagged is False
    assert kept.n_trims == 2
    assert calls["n"] == 3                               # native + k=2 (pass) + k=3 (fail)


def test_bisection_never_overwrites_complete_best_good(monkeypatch):
    # Second belt of the oracle fix: even a hard-core-good trial WITHOUT any excess
    # reason (prereq gap — legitimately known-good for lo-advance) must not replace a
    # judged-COMPLETE smaller best_good; completeness outranks size.
    sents = mini_sents(5)
    units = mini_units(sents)

    def decide(text):
        if "sentence 4." in text:                        # native full span → routes to trim
            return _over_inclusion("sentence 4.")
        if "sentence 3." in text:                        # k=4 probe: partial, no excess
            return _hard_core_partial()
        return _pass()                                   # k=2 / k=3 probes: COMPLETE
    calls = _script(monkeypatch, decide)
    kept, rej = _vr(_mk_cand(units, sents), sents, units)
    assert rej is None and kept is not None
    assert kept.unit_ids == ["u0000", "u0001", "u0002"]  # largest COMPLETE prefix wins
    assert kept.verdict.understandable is True and kept.ship_flagged is False
    assert kept.n_trims == 3
    assert calls["n"] == 4                               # native + probes k=2, k=3, k=4


def test_bisection_unverified_excess_stays_known_good(monkeypatch):
    # Asymmetric-gate philosophy preserved: a PHANTOM (unquotable) over_inclusion
    # complaint on a hard-core-good trial must NOT move the bisection — the trial stays
    # known-good, the largest hard-core span ships, and the unbacked single_idea=False
    # surfaces as the warning-only 'single_idea_unverified' (never a shrink, never a kill).
    sents = mini_sents(4)
    units = mini_units(sents)

    def decide(text):
        if "sentence 3." in text:                        # native full span → routes to trim
            return _over_inclusion("sentence 2.")
        if "sentence 2." in text:                        # k=3 probe: PHANTOM evidence
            return _over_inclusion("words never spoken in this clip")
        return _hard_core_partial()                      # k=2 probe: partial, no excess
    calls = _script(monkeypatch, decide)
    kept, rej = _vr(_mk_cand(units, sents), sents, units)
    assert rej is None and kept is not None
    assert kept.unit_ids == ["u0000", "u0001", "u0002"]  # phantom did not shrink the ship
    assert kept.ship_flagged is False
    assert "single_idea_unverified" in kept.warnings     # …but the unbacked bit is visible
    assert kept.n_trims == 2
    assert calls["n"] == 3                               # native + probes k=2, k=3


def test_bisection_isolates_excess_verified_on_incomplete_prefix(monkeypatch):
    # G1 (test-quality audit): _excess_verified is CO-DEFENDED in the tests above — either a
    # COMPLETE sub-span exists (the completeness belt at :827-829 engages) or the excess is
    # PHANTOM (both belts agree) — so reverting ONLY _excess_verified leaves them all green.
    # Isolate it: a VERIFIED over_inclusion on a hard-core-good but INCOMPLETE trial where NO
    # sub-span is ever is_complete. The completeness belt can never fire, so _excess_verified
    # ALONE must keep the two-idea trial on the known-BAD side; without it the atomic prefix is
    # overwritten by the over-inclusive span (confirmed: mutating it to False ships [..,u0002]).
    sents = mini_sents(4)
    units = mini_units(sents)

    def decide(text):
        # the second idea is [u0002,u0003] (opening 'sentence 2.'); any span holding it is
        # over-inclusive with the hard core intact and a VERIFIED quote…
        if "sentence 2." in text or "sentence 3." in text:
            return _over_inclusion("sentence 2.")
        # …every atomic prefix is hard-core-good but NEVER is_complete (prereq gap), so the
        # completeness belt cannot engage and _excess_verified alone holds the boundary.
        return _hard_core_partial()
    calls = _script(monkeypatch, decide)
    kept, rej = _vr(_mk_cand(units, sents), sents, units)
    assert rej is None and kept is not None              # never killed (unverified_kill=0)
    assert kept.unit_ids == ["u0000", "u0001"]           # atomic prefix ships…
    assert kept.i_start == 0 and kept.i_end == 1         # …NOT the verified two-idea span
    assert calls["n"] == 3                               # native + k=2 (good) + k=3 (verified-bad)


# ── single_idea=False is warning-only (pre-calibration: never a gate, never a kill) ──
def test_single_idea_false_without_verified_reason_warns_only(monkeypatch):
    # a COMPLETE verdict with single_idea=False and no failure reason ships at native
    # size with a warning — single_idea is deliberately outside required_verdict_fields.
    sents = mini_sents(2)
    units = mini_units(sents)
    calls = _script(monkeypatch, lambda text: JudgeVerdict(
        reasoning="fine but two ideas", score_10=9, understandable=True, single_idea=False))
    kept, rej = _vr(_mk_cand(units, sents, unit_ids=["u0000"]), sents, units)
    assert rej is None and kept is not None and calls["n"] == 1
    assert kept.ship_flagged is False                    # warning only — not the flag path
    assert "single_idea_unverified" in kept.warnings


def test_single_idea_false_with_verified_reason_not_warned(monkeypatch):
    # the same complete verdict backed by a VERIFIED over_inclusion quote carries real
    # evidence — no unverified-atomicity warning.
    sents = mini_sents(2)
    units = mini_units(sents)

    def decide(text):
        v = JudgeVerdict(reasoning="two ideas", score_10=9, understandable=True,
                         single_idea=False,
                         failure_reasons=[FailureReason(kind="over_inclusion",
                                                        evidence_quote="sentence 1.")])
        return v
    _script(monkeypatch, decide)
    kept, rej = _vr(_mk_cand(units, sents), sents, units)
    assert rej is None and kept is not None
    assert "single_idea_unverified" not in kept.warnings


def test_single_idea_false_never_kills_on_terminal_gate(monkeypatch):
    # a failing, reason-less single_idea=False verdict (nothing to verify or confirm)
    # must ship flagged — unverified_kill stays 0 by construction.
    sents = mini_sents(2)
    units = mini_units(sents)
    calls = _script(monkeypatch, lambda text: JudgeVerdict(
        reasoning="two ideas", score_10=3, understandable=False, single_idea=False,
        topic_identifiable=False))                       # fails the hard core too
    kept, rej = _vr(_mk_cand(units, sents, unit_ids=["u0000"]), sents, units)
    assert rej is None and kept is not None              # never killed on unverifiable evidence
    assert kept.ship_flagged is True
    assert "unverified_judge_concerns" in kept.warnings
    assert "single_idea_unverified" in kept.warnings
    assert "kill_confirm_unavailable" not in kept.warnings   # confirm_kill never reached


# ── same-facet dedupe union guards (_dedupe_partb / _union_guard) ─────────────
def _node_units(sents, node_ids):
    return [Unit(unit_id=f"u{i:04d}", start=s.start, end=s.end, sentence_range=(i, i),
                 role="worked_step", node_id=node_ids[i], transcript=s.text)
            for i, s in enumerate(sents)]


def _spec(cand_id, s0, s1, sents, *, unit_ids=None, facet="worked_example", fq=0.5, **extra):
    d = {"cand_id": cand_id, "title": f"t-{cand_id}", "role": "result",
         "start": sents[s0].start, "end": sents[s1].end, "cut_end": sents[s1].end,
         "sentence_start_idx": s0, "sentence_end_idx": s1, "facet": facet,
         "final_quality": fq, "score": fq, "warnings": (), "referential": [],
         "unit_ids": unit_ids if unit_ids is not None else
         [f"u{i:04d}" for i in range(s0, s1 + 1)]}
    d.update(extra)
    return d


def _dd(specs, sents, units, max_dur=500.0):
    return _dedupe_partb(specs, sents, 1.0, units, FakeAdapter(), max_dur=max_dur)


def test_union_blocked_on_differing_arc_provenance():
    # the qP mash shape: two DISTINCT detected arcs of the same facet that touch — the
    # union is skipped and the pair falls to the trim/keep-both path (nothing dies).
    sents = mini_sents(10)
    units = mini_units(sents)
    k = _spec("a2", 0, 4, sents, arc_id="arc_2")
    c = _spec("a4", 4, 8, sents, arc_id="arc_4")
    kept, rejections = _dd([k, c], sents, units)
    assert [s["cand_id"] for s in kept] == ["a2", "a4"]  # keep-both, not one mash
    assert not any(s.get("merged") for s in kept)
    assert "trimmed_start" in kept[1]["warnings"]        # overlap resolved by trim
    assert rejections == []


def test_union_blocked_on_arc_ids_provenance_from_prior_merge():
    # W25-D arc_ids provenance counts: a spec whose arc identity lives only in arc_ids
    # (winner of an earlier merge) still guards against a different arc.
    sents = mini_sents(10)
    units = mini_units(sents)
    k = _spec("a2", 0, 4, sents, arc_id="arc_2")
    c = _spec("m45", 4, 8, sents, arc_id="", arc_ids=["arc_4", "arc_5"])
    kept, _ = _dd([k, c], sents, units)
    assert len(kept) == 2 and not any(s.get("merged") for s in kept)


def test_union_allowed_on_same_arc_provenance():
    # control: two halves of ONE arc still union (the guard compares, not bans).
    sents = mini_sents(10)
    units = mini_units(sents)
    k = _spec("h1", 0, 4, sents, arc_id="arc_2")
    c = _spec("h2", 4, 8, sents, arc_id="arc_2")
    kept, _ = _dd([k, c], sents, units)
    assert len(kept) == 1 and kept[0].get("merged") is True
    assert kept[0]["arc_id"] == "arc_2"


def test_union_blocked_on_disjoint_topic_nodes():
    # both sides carry units, but from different topic nodes → distinct problems.
    sents = mini_sents(10)
    node_ids = ["ch1.t1"] * 4 + [""] + ["ch1.t2"] * 5
    units = _node_units(sents, node_ids)
    k = _spec("t1", 0, 4, sents, unit_ids=["u0000", "u0001", "u0002", "u0003"])
    c = _spec("t2", 4, 8, sents, unit_ids=["u0005", "u0006", "u0007", "u0008"])
    kept, _ = _dd([k, c], sents, units)
    assert len(kept) == 2 and not any(s.get("merged") for s in kept)


def test_union_allowed_on_shared_topic_node():
    # control: one shared node ⇒ same problem's neighborhood ⇒ union proceeds.
    sents = mini_sents(10)
    node_ids = ["ch1.t1"] * 5 + ["ch1.t1"] + ["ch1.t2"] * 4
    units = _node_units(sents, node_ids)
    k = _spec("t1", 0, 4, sents, unit_ids=["u0000", "u0001", "u0002", "u0003"])
    c = _spec("t2", 4, 8, sents, unit_ids=["u0005", "u0006", "u0007", "u0008"])
    kept, _ = _dd([k, c], sents, units)
    assert len(kept) == 1 and kept[0].get("merged") is True


def test_union_blocked_when_both_contract_complete():
    # two contract-complete, hard-gate-passing clips are each whole — never mash them.
    sents = mini_sents(10)
    units = mini_units(sents)
    k = _spec("w1", 0, 4, sents, contract_coverage=1.0, hard_gates_ok=True)
    c = _spec("w2", 4, 8, sents, contract_coverage=1.0, hard_gates_ok=True)
    kept, _ = _dd([k, c], sents, units)
    assert len(kept) == 2 and not any(s.get("merged") for s in kept)
    # …but an INCOMPLETE side still unions (it needs the other's content):
    c2 = _spec("w3", 4, 8, sents, contract_coverage=0.5, hard_gates_ok=True)
    kept2, _ = _dd([dict(k), c2], sents, units)
    assert len(kept2) == 1 and kept2[0].get("merged") is True


def test_union_never_exceeds_max_clip_duration():
    # the kinematics arc_1+arc_2 shape: a union over the ship cap is never produced —
    # today this was unchecked and built unshippable 264s specs.
    sents = mini_sents(10)
    units = mini_units(sents)
    specs = [_spec("k1", 0, 4, sents), _spec("k2", 4, 8, sents)]   # union would be 89.9s
    kept, _ = _dd([dict(s) for s in specs], sents, units, max_dur=60.0)
    assert len(kept) == 2 and not any(s.get("merged") for s in kept)
    for s in kept:
        assert s["end"] - s["start"] <= 60.0
    kept2, _ = _dd([dict(s) for s in specs], sents, units, max_dur=500.0)   # control
    assert len(kept2) == 1 and kept2[0].get("merged") is True


def test_union_guard_reasons_pure():
    sents = mini_sents(10)
    units_by_id = {u.unit_id: u for u in mini_units(sents)}
    a, b = _spec("a", 0, 4, sents), _spec("b", 4, 8, sents)
    assert _union_guard(a, b, units_by_id, 500.0) is None          # nothing blocks
    assert _union_guard(dict(a, arc_id="x"), dict(b, arc_id="y"),
                        units_by_id, 500.0) == "differing arc provenance"
    assert _union_guard(a, b, units_by_id, 60.0) == "union exceeds max_clip_duration_s"


def test_qp_mash_shape_three_arcs_stay_three_clips():
    # end-to-end through snap_candidates: three distinct practice-question arcs that
    # touch (qP arcs 2+4+5) ship as THREE clips, not one [382,480]-style mash.
    sents = mini_sents(12)
    units = mini_units(sents)

    def arc_cand(cand_id, s0, s1, arc_id):
        c = Candidate(cand_id=cand_id, anchor_id=f"u{s0:04d}", role="result",
                      facet="worked_example", title=cand_id, reason="r",
                      unit_ids=[f"u{i:04d}" for i in range(s0, s1 + 1)], referential=[],
                      i_start=s0, i_end=s1, start=sents[s0].start, end=sents[s1].end)
        c.arc_id = arc_id
        c.final_quality = 0.8
        c.verdict = _pass()
        return c
    cands = [arc_cand("arc2", 0, 3, "arc_2"), arc_cand("arc4", 3, 6, "arc_4"),
             arc_cand("arc5", 6, 9, "arc_5")]
    specs, rejections = snap_candidates(cands, sents, {"min_clip_duration_s": 1.0,
                                                       "max_clip_duration_s": 500.0},
                                        units, FakeAdapter())
    assert len(specs) == 3
    assert not any("merged_overlap" in (s.get("warnings") or ()) for s in specs)
    assert rejections == []


# ── min-duration snap: node-aware extension direction ─────────────────────────
def _snap(cand, sents, min_dur=25.0):
    from backend.pipeline.refine import _snap_one
    return _snap_one(cand, sents, False, min_dur, 0.05, 500.0)


def test_min_snap_extends_backward_when_forward_leaves_node():
    # anchor sits at the END of its topic node: forward extension would swallow the next
    # event's units — with node_span given, the start walks backward inside the node.
    sents = mini_sents(6)
    clip = _snap({"i_start": 2, "i_end": 2, "facet": "other",
                  "node_span": (0.0, sents[2].end)}, sents)
    assert clip is not None
    assert clip["sentence_start_idx"] == 0 and clip["sentence_end_idx"] == 2
    assert "extended_for_min_duration" in clip["warnings"]


def test_min_snap_keeps_forward_when_it_stays_inside_node():
    # forward stays inside the node → forward remains the default direction.
    sents = mini_sents(6)
    clip = _snap({"i_start": 3, "i_end": 3, "facet": "other",
                  "node_span": (sents[3].start, sents[5].end)}, sents)
    assert clip["sentence_start_idx"] == 3 and clip["sentence_end_idx"] == 5


def test_min_snap_forward_default_when_both_directions_leave_node():
    # a one-sentence node: neither direction stays inside → legacy forward behavior.
    sents = mini_sents(6)
    clip = _snap({"i_start": 3, "i_end": 3, "facet": "other",
                  "node_span": (sents[3].start, sents[3].end)}, sents)
    assert clip["sentence_start_idx"] == 3 and clip["sentence_end_idx"] == 5


def test_min_snap_forward_without_node_info():
    # no node_span (legacy fast path) → behavior unchanged: forward only.
    sents = mini_sents(6)
    clip = _snap({"i_start": 2, "i_end": 2, "facet": "other"}, sents)
    assert clip["sentence_start_idx"] == 2 and clip["sentence_end_idx"] == 4


def test_snap_candidates_threads_anchor_node_span():
    # end-to-end: Part B computes the anchor's node hull from the units and _snap_one
    # extends BACKWARD (inside the node) instead of forward into the next node.
    sents = mini_sents(6)
    node_ids = ["n1", "n1", "n1", "n2", "n2", "n2"]
    units = _node_units(sents, node_ids)
    c = Candidate(cand_id="c0", anchor_id="u0002", role="worked_step",
                  facet="worked_example", title="t", reason="r", unit_ids=["u0002"],
                  referential=[], i_start=2, i_end=2,
                  start=sents[2].start, end=sents[2].end)
    c.verdict = _pass()
    specs, _ = snap_candidates([c], sents, {"min_clip_duration_s": 25.0,
                                            "max_clip_duration_s": 500.0},
                               units, FakeAdapter())
    assert len(specs) == 1
    s = specs[0]
    assert s["sentence_start_idx"] == 0 and s["sentence_end_idx"] == 2
    assert "extended_for_min_duration" in s["warnings"]
    # true_contents refreshed: the absorbed same-node units joined unit_ids
    assert s["unit_ids"] == ["u0000", "u0001", "u0002"]
