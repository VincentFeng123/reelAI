"""B-judge-integrity: quote-verified failure reasons, asymmetric 3-outcome kill gate,
fresh-context kill confirmation. All offline (llm_json monkeypatched)."""
from __future__ import annotations

import inspect
import threading

import pytest

import backend.llm as llm_mod
from backend.pipeline.assemble.graph import Graph
from backend.pipeline.assemble.integrity import Rejection
from backend.pipeline.assemble.types import Candidate
from backend.pipeline.assemble.validate import (
    JUDGE_SYSTEM, FailureReason, JudgeVerdict, KillClaimCheck, KillConfirmation,
    _verify_failure_reasons, confirm_kill, judge_clip, validate_and_repair,
)

from .conftest import FakeAdapter, mini_sents, mini_units

CLIP = "So the fog rolled in. We solved x equals two, right?"


def _fr(kind="other", quote="", **kw):
    return FailureReason(kind=kind, evidence_quote=quote, **kw)


def _verdict(*reasons, **kw):
    kw.setdefault("score_10", 3)
    kw.setdefault("understandable", False)
    return JudgeVerdict(failure_reasons=list(reasons), **kw)


# ── B1: schema + prompt ───────────────────────────────────────────────────────
def test_failure_reason_has_evidence_quote_default_empty():
    assert FailureReason().evidence_quote == ""
    assert "evidence_quote" in FailureReason.model_json_schema()["properties"]


def test_prompt_demands_verbatim_evidence():
    assert "evidence_quote" in JUDGE_SYSTEM
    assert "verbatim" in JUDGE_SYSTEM
    # kind vocabulary byte-preserved (repair loop dispatches on it)
    for kind in ("unresolved_reference", "missing_prerequisite", "missing_visual",
                 "missing_problem_statement", "missing_reasoning", "missing_result",
                 "not_source_grounded", "off_topic", "other"):
        assert kind in JUDGE_SYSTEM


def test_verification_not_part_of_llm_schema():
    props = JudgeVerdict.model_json_schema(ref_template="{model}")["properties"]
    assert "_reason_verified" not in props
    assert "reason_verified" not in str(props)


# ── B2: normalization + containment ──────────────────────────────────────────
def test_exact_quote_verifies():
    assert _verify_failure_reasons(_verdict(_fr(quote="the fog rolled in")), CLIP) == [True]


def test_punctuation_and_whitespace_variance_still_verifies():
    v = _verdict(_fr(quote="We  SOLVED x, equals two right?!"))
    assert _verify_failure_reasons(v, CLIP) == [True]


def test_phantom_quote_fails_verification():
    # the real audit shape: judge cites "the previous equation" — a phrase not in the span
    v = _verdict(_fr(kind="missing_prerequisite", quote="the previous equation"))
    assert _verify_failure_reasons(v, CLIP) == [False]


def test_empty_quote_fails_verification():
    assert _verify_failure_reasons(_verdict(_fr(quote="")), CLIP) == [False]
    assert _verify_failure_reasons(_verdict(_fr(quote="   ")), CLIP) == [False]


def test_unresolved_reference_requires_reference_text_containment():
    ok = _fr(kind="unresolved_reference", quote="the fog rolled in", reference_text="the fog")
    bad = _fr(kind="unresolved_reference", quote="the fog rolled in", reference_text="that fog")
    none = _fr(kind="unresolved_reference", quote="the fog rolled in")   # no reference_text → quote only
    assert _verify_failure_reasons(_verdict(ok, bad, none), CLIP) == [True, False, True]


def test_verification_attaches_to_verdict():
    v = _verdict(_fr(quote="the fog rolled in"), _fr(quote="not in the span"))
    flags = _verify_failure_reasons(v, CLIP)
    assert flags == [True, False]
    assert v._reason_verified == [True, False]           # travels with the verdict


# ── B3: confirm_kill ──────────────────────────────────────────────────────────
def test_confirm_kill_empty_reasons_no_llm(monkeypatch):
    monkeypatch.setattr(llm_mod, "llm_json",
                        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("must not call")))
    assert confirm_kill(CLIP, []) == []


def test_confirm_kill_prompt_and_containment(monkeypatch):
    from backend import config
    monkeypatch.setattr(config, "JUDGE_MODEL", "")
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "same")
    seen = {}

    def fake(system, user, schema, **kw):
        assert schema is KillConfirmation
        seen["system"], seen["user"], seen["kw"] = system, user, kw
        return KillConfirmation(claims=[
            KillClaimCheck(claim=1, confirmed=True, quote="the fog rolled in"),
            KillClaimCheck(claim=2, confirmed=True, quote="the previous equation"),  # phantom quote
            KillClaimCheck(claim=3, confirmed=False, quote="we solved x equals two"),
        ])
    monkeypatch.setattr(llm_mod, "llm_json", fake)
    reasons = [_fr(kind="unresolved_reference", quote="the fog rolled in", detail="dangling fog"),
               _fr(kind="missing_prerequisite", quote="we solved x", detail="x never set up"),
               _fr(kind="missing_result", quote="we solved x", detail="no result")]
    out = confirm_kill(CLIP, reasons)
    # confirmed & quote contained → True; confirmed & phantom quote → NOT confirmed; explicit false
    assert out == [True, False, False]
    assert seen["kw"].get("temperature") == 0.0
    assert "Claim 1: unresolved_reference: dangling fog." in seen["user"]
    assert "Is this actually true of the transcript below?" in seen["user"]
    assert CLIP in seen["user"]


def test_confirm_kill_claim_number_mapping_and_missing_default_false(monkeypatch):
    monkeypatch.setattr(llm_mod, "llm_json", lambda *a, **kw: KillConfirmation(claims=[
        KillClaimCheck(claim=2, confirmed=True, quote="we solved x equals two")]))
    out = confirm_kill(CLIP, [_fr(quote="a"), _fr(quote="b")])
    assert out == [False, True]                          # claim 2 mapped; claim 1 missing → False


def test_confirm_kill_outage_confirms_nothing_and_marks_outage(monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("api down")
    monkeypatch.setattr(llm_mod, "llm_json", boom)
    outage: dict = {}
    assert confirm_kill(CLIP, [_fr(quote="x"), _fr(quote="y")], outage) == [False, False]
    assert outage.get("confirm_kill") is True


# ── B4: asymmetric terminal gate ─────────────────────────────────────────────
def _mk_candidate(sents):
    return Candidate(cand_id="c0", anchor_id="u0000", role="explanation", facet="other",
                     title="T", reason="r", unit_ids=["u0000"], referential=[],
                     i_start=0, i_end=0, start=sents[0].start, end=sents[0].end)


def _run_gate(monkeypatch, fake_llm):
    from backend import config
    monkeypatch.setattr(config, "JUDGE_MODEL", "")
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "same")
    monkeypatch.setattr(llm_mod, "llm_json", fake_llm)
    sents = mini_sents(3)
    units = mini_units(sents)
    units_by_id = {u.unit_id: u for u in units}
    return validate_and_repair(_mk_candidate(sents), sents, Graph([], units), units, units_by_id,
                               {}, FakeAdapter(), {}, lambda s, e: "", "topic",
                               {}, threading.Lock())
    # candidate clip text is "sentence 0." — quotes must match that to verify


def _failing_verdict(*reasons):
    # unrepairable (no expansion targets) AND fails the hard core → reaches the terminal gate
    return JudgeVerdict(reasoning="bad", score_10=2, understandable=False,
                        topic_identifiable=False, purpose_identifiable=False,
                        failure_reasons=list(reasons))


def test_zero_verified_ships_flagged_no_rejection(monkeypatch):
    schemas = []

    def fake(system, user, schema, **kw):
        schemas.append(schema)
        return _failing_verdict(_fr(kind="missing_problem_statement",
                                    detail="no problem stated",
                                    quote="the previous equation"))     # phantom: not in span
    kept, rej = _run_gate(monkeypatch, fake)
    assert rej is None and kept is not None
    assert kept.ship_flagged is True
    assert "unverified_judge_concerns" in kept.warnings
    assert "kill_confirm_unavailable" not in kept.warnings
    assert kept.n_failure_reasons == 1 and kept.n_verified == 0
    assert KillConfirmation not in schemas               # confirm_kill never ran (accept-side path)


def test_verified_and_confirmed_rejects_with_confirmed_kinds(monkeypatch):
    def fake(system, user, schema, **kw):
        if schema is KillConfirmation:
            return KillConfirmation(claims=[
                KillClaimCheck(claim=1, confirmed=True, quote="sentence 0.")])
        return _failing_verdict(
            _fr(kind="off_topic", detail="d", quote="sentence 0."),          # verified
            _fr(kind="missing_result", detail="d2", quote="not in the span"))  # phantom
    kept, rej = _run_gate(monkeypatch, fake)
    assert kept is None
    assert isinstance(rej, Rejection) and rej.stage == "repair"
    assert rej.failure_kinds == ["off_topic"]            # confirmed kinds only
    assert rej.verified_kinds == ("off_topic",)
    assert rej.unverified_kinds == ("missing_result",)
    assert rej.kill_confirmed is True


def test_verified_but_unconfirmed_ships_flagged(monkeypatch):
    def fake(system, user, schema, **kw):
        if schema is KillConfirmation:
            return KillConfirmation(claims=[
                KillClaimCheck(claim=1, confirmed=False, quote="sentence 0.")])
        return _failing_verdict(_fr(kind="off_topic", detail="d", quote="sentence 0."))
    kept, rej = _run_gate(monkeypatch, fake)
    assert rej is None and kept is not None
    assert kept.ship_flagged is True and "unverified_judge_concerns" in kept.warnings
    assert kept.n_failure_reasons == 1 and kept.n_verified == 1


def test_confirmation_with_phantom_quote_counts_as_unconfirmed(monkeypatch):
    def fake(system, user, schema, **kw):
        if schema is KillConfirmation:                   # confirmed=True but quote not in span
            return KillConfirmation(claims=[
                KillClaimCheck(claim=1, confirmed=True, quote="the previous equation")])
        return _failing_verdict(_fr(kind="off_topic", detail="d", quote="sentence 0."))
    kept, rej = _run_gate(monkeypatch, fake)
    assert rej is None and kept is not None and kept.ship_flagged is True


def test_confirm_outage_ships_flagged_with_outage_marker(monkeypatch):
    def fake(system, user, schema, **kw):
        if schema is KillConfirmation:
            raise RuntimeError("confirm outage")
        return _failing_verdict(_fr(kind="off_topic", detail="d", quote="sentence 0."))
    kept, rej = _run_gate(monkeypatch, fake)
    assert rej is None and kept is not None              # never kill on an unconfirmable verdict
    assert "unverified_judge_concerns" in kept.warnings
    assert "kill_confirm_unavailable" in kept.warnings   # outage recorded on the shipped record


def test_accept_path_records_stats_and_no_flag(monkeypatch):
    def fake(system, user, schema, **kw):
        assert schema is JudgeVerdict
        return JudgeVerdict(reasoning="ok", score_10=9, understandable=True,
                            failure_reasons=[_fr(kind="other", quote="sentence 0."),
                                             _fr(kind="other", quote="ghost text")])
    kept, rej = _run_gate(monkeypatch, fake)
    assert rej is None and kept is not None
    assert kept.ship_flagged is False and "unverified_judge_concerns" not in kept.warnings
    assert kept.n_failure_reasons == 2 and kept.n_verified == 1
    assert kept.verdict._reason_verified == [True, False]


# ── B4/B7: flag + stats survive to the spec dict ─────────────────────────────
def test_ship_flag_and_stats_survive_snap_to_spec_dict():
    from backend.pipeline.assemble.boundary_adapt import snap_candidates
    sents = mini_sents(3)
    c = _mk_candidate(sents)
    c.verdict = JudgeVerdict(score_10=4, understandable=False)
    c.warnings = ("unverified_judge_concerns",)
    c.ship_flagged = True
    c.n_failure_reasons, c.n_verified = 3, 0
    specs, _rej = snap_candidates([c], sents, {"min_clip_duration_s": 1.0,
                                               "max_clip_duration_s": 500.0})
    s = specs[0]
    assert "unverified_judge_concerns" in s["warnings"]
    assert s["ship_flagged"] is True
    assert s["n_failure_reasons"] == 3 and s["n_verified"] == 0


def test_assemble_ships_flagged_specs_end_to_end(monkeypatch):
    from backend import config
    from backend.pipeline.assemble import assemble_clips
    from backend.pipeline.understand.models import ContentMap, ContentNode, DependencyGraph, Structure
    monkeypatch.setattr(config, "JUDGE_MODEL", "")
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "same")
    schemas = []

    def fake(system, user, schema, **kw):
        schemas.append(schema)
        return _failing_verdict(_fr(kind="missing_prerequisite", detail="phantom",
                                    quote="the previous equation"))
    monkeypatch.setattr(llm_mod, "llm_json", fake)

    class AnchorAdapter(FakeAdapter):
        def is_anchor_role(self, role):
            return True

        def anchor_priority(self, role):
            return 0.9

        def facet_for(self, role):
            return "other"

        def valid_roles(self):
            return {"explanation"}

    sents = mini_sents(4)
    units = mini_units(sents)
    st = Structure(video_id="v", units=units, dependencies=DependencyGraph(),
                   content_map=ContentMap(nodes=[ContentNode(node_id="video", level="video",
                                                             sentence_range=(0, len(sents) - 1))]))
    settings = {"min_clip_duration_s": 1.0, "max_clip_duration_s": 500.0,
                "min_comprehension_score": 0.7, "quality_floor": 0.0, "max_clips": 12,
                "max_anchors": 12, "closure_max_span_s": 999.0,
                # legacy selector pinned: this test counts llm_json schemas on the judge path
                "anchor_selector": "priority"}
    specs, _notes, rejections = assemble_clips(st, "", sents, "u", "v", settings, AnchorAdapter())
    assert specs, "phantom-only verdicts must ship flagged, not be rejected"
    assert all(r.stage != "repair" for r in rejections)
    for s in specs:
        assert s["ship_flagged"] is True
        assert "unverified_judge_concerns" in s["warnings"]
        assert s["n_failure_reasons"] >= 1 and s["n_verified"] == 0
    assert KillConfirmation not in schemas


# ── B5: Rejection record stays backward-compatible ───────────────────────────
def test_rejection_new_fields_default_safe_and_cli_print_shape():
    r = Rejection(cand_id="c1", title="t", role="claim", stage="repair", reason="x")
    assert r.verified_kinds == () and r.unverified_kinds == () and r.kill_confirmed is False
    # the CLI drop-ledger line must still format
    line = f"  [dropped/{r.stage}] {r.title[:60]} (score={r.score}, kinds={r.failure_kinds}, q={r.final_quality})"
    assert line.startswith("  [dropped/repair]")


# ── B6: warning penalty via the existing mechanism ───────────────────────────
def test_unverified_judge_concerns_docks_boundary_score():
    from backend.pipeline.assemble.scoring import boundary_score
    assert boundary_score(["unverified_judge_concerns"]) == pytest.approx(0.85)   # unjudged tier
    assert boundary_score(["unverified_judge_concerns", "unjudged"]) == pytest.approx(0.70)


# ── B8: public surface unchanged ─────────────────────────────────────────────
def test_judge_clip_public_signature_unchanged():
    params = list(inspect.signature(judge_clip).parameters.values())
    assert [p.name for p in params[:3]] == ["clip_text", "role", "adapter"]
    assert all(p.default is not inspect.Parameter.empty for p in params[3:])
