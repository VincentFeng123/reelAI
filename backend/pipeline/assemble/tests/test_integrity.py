"""integrity helpers: truthful unit_ids, Part-B merge, Rejection. Pure/offline."""
from __future__ import annotations

import pytest

from backend.pipeline.assemble.integrity import Rejection, merge_partb, true_contents

from .conftest import mini_sents, mini_units


def _setup(n=6):
    sents = mini_sents(n)
    units = mini_units(sents)          # unit i ↔ sentence i, ids u0000..u000{n-1}
    return sents, units


# ── true_contents ─────────────────────────────────────────────────────────────
def test_absorbs_spanned_units_time_ordered():
    sents, units = _setup()
    ids, ref = true_contents(["u0000", "u0003"], [], units, 0, 3)
    assert ids == ["u0000", "u0001", "u0002", "u0003"]   # gap units absorbed, ordered
    assert ref == []


def test_drops_referential_now_inside_span():
    sents, units = _setup()
    ids, ref = true_contents(["u0000", "u0002"], [("u0001", "prerequisite"), ("u0005", "prerequisite")],
                             units, 0, 2)
    assert "u0001" in ids                                # was referential, now in-span → absorbed
    assert ref == [("u0005", "prerequisite")]            # outside span → untouched


def test_idempotent():
    sents, units = _setup()
    once = true_contents(["u0000", "u0003"], [("u0005", "x")], units, 0, 3)
    twice = true_contents(once[0], once[1], units, 0, 3)
    assert once == twice


def test_partial_overlap_unit_not_absorbed():
    sents, units = _setup()
    # unit u0003 spans sentence 3 only; span [0,2] must not absorb it
    ids, _ = true_contents(["u0000"], [], units, 0, 2)
    assert "u0003" not in ids


# ── merge_partb ───────────────────────────────────────────────────────────────
def _spec(cand_id, s0, s1, sents, *, fq=0.5, facet="other", **extra):
    d = {"cand_id": cand_id, "facet": facet, "role": "explanation", "title": f"t-{cand_id}",
         "anchor_id": "u0000", "unit_ids": [f"u{i:04d}" for i in range(s0, s1 + 1)],
         "referential": [], "start": sents[s0].start, "end": sents[s1].end,
         "cut_end": sents[s1].end, "sentence_start_idx": s0, "sentence_end_idx": s1,
         "score": fq, "final_quality": fq, "warnings": ("w_a",) if cand_id == "a" else ("w_b",),
         "judge_error": False, "truncated": False, "context_card": ""}
    d.update(extra)
    return d


def test_merge_unions_span_ids_referential_warnings():
    sents, units = _setup()
    a = _spec("a", 0, 2, sents, fq=0.9, referential=[("u0005", "prerequisite")])
    b = _spec("b", 2, 4, sents, fq=0.4)
    m = merge_partb(a, b, units, sents)
    assert m["sentence_start_idx"] == 0 and m["sentence_end_idx"] == 4
    assert m["start"] == sents[0].start and m["end"] == sents[4].end
    assert m["unit_ids"] == [f"u{i:04d}" for i in range(5)]      # union + absorption
    assert m["referential"] == [("u0005", "prerequisite")]        # still outside span
    assert set(m["warnings"]) >= {"w_a", "w_b", "merged_overlap"}
    assert m["merged"] is True
    assert m["title"] == "t-a"                                    # winner metadata (higher fq)


def test_merge_ors_flags_and_keeps_winner_metadata():
    sents, units = _setup()
    a = _spec("a", 0, 1, sents, fq=0.3, judge_error=True)
    b = _spec("b", 1, 2, sents, fq=0.8, truncated=True)
    m = merge_partb(a, b, units, sents)
    assert m["judge_error"] is True and m["truncated"] is True    # OR'd
    assert m["title"] == "t-b"                                    # b wins on final_quality


def test_merge_without_units_skips_absorption():
    sents, _ = _setup()
    a = _spec("a", 0, 0, sents, fq=0.9)
    b = _spec("b", 3, 3, sents, fq=0.1)
    m = merge_partb(a, b, None, sents)
    assert m["unit_ids"] == ["u0000", "u0003"]                    # union only, no sweep


# ── W25-D: arc provenance survives merges ─────────────────────────────────────
def test_merge_winner_without_arc_id_inherits_losers():
    # pre-W25-D the winner took ALL non-span keys, silently stripping the loser's arc_id
    # → eval's n_arc_clips undercounted merged arc clips.
    sents, units = _setup()
    arc = _spec("a", 0, 2, sents, fq=0.4, arc_id="arc_3")
    plain = _spec("b", 2, 4, sents, fq=0.9, arc_id="")      # non-arc side WINS
    m = merge_partb(arc, plain, units, sents)
    assert m["title"] == "t-b"                              # winner metadata unchanged
    assert m["arc_id"] == "arc_3"                           # inherited, not stripped
    assert m["arc_ids"] == ["arc_3"]


def test_merge_unions_arc_ids_winner_id_canonical():
    sents, units = _setup()
    a = _spec("a", 0, 2, sents, fq=0.9, arc_id="arc_1")
    b = _spec("b", 2, 4, sents, fq=0.4, arc_id="arc_2")
    m = merge_partb(a, b, units, sents)
    assert m["arc_id"] == "arc_1"                           # winner's own id stays canonical
    assert m["arc_ids"] == ["arc_1", "arc_2"]               # both provenances retained


def test_merge_chains_union_prior_arc_ids():
    # a spec that already carries a merge-union list keeps EVERY arc across a second merge
    sents, units = _setup()
    a = _spec("a", 0, 2, sents, fq=0.9, arc_id="")          # arc-less winner
    b = _spec("b", 2, 4, sents, fq=0.4, arc_id="arc_2", arc_ids=["arc_0", "arc_2"])
    m = merge_partb(a, b, units, sents)
    assert m["arc_id"] == "arc_2"
    assert set(m["arc_ids"]) == {"arc_0", "arc_2"}


def test_merge_without_arc_provenance_adds_no_arc_keys():
    sents, units = _setup()
    m = merge_partb(_spec("a", 0, 2, sents, fq=0.9), _spec("b", 2, 4, sents, fq=0.4),
                    units, sents)
    assert not m.get("arc_id") and "arc_ids" not in m


# ── Rejection ─────────────────────────────────────────────────────────────────
def test_rejection_dataclass_shape():
    r = Rejection(cand_id="c1", title="t", role="claim", stage="dedupe",
                  reason="overlap loser to c0", score=0.4, failure_kinds=["off_topic"],
                  final_quality=0.3, start=1.0, end=2.0)
    assert r.stage == "dedupe" and r.failure_kinds == ["off_topic"]


# ── truthful ids at build/repair + repair rejection ──────────────────────────
import threading

import backend.llm as llm_mod
from backend.pipeline.assemble.graph import Graph
from backend.pipeline.assemble.types import Candidate
from backend.pipeline.assemble.validate import JudgeVerdict, expand_candidate, validate_and_repair

from .conftest import FakeAdapter


def test_expand_candidate_emits_true_contents():
    sents, units = _setup()
    units_by_id = {u.unit_id: u for u in units}
    cand = Candidate(cand_id="c0", anchor_id="u0003", role="explanation", facet="other",
                     title="t", reason="r", unit_ids=["u0003"],
                     referential=[("u0000", "prerequisite")], i_start=3, i_end=3,
                     start=sents[3].start, end=sents[3].end)
    verdict = JudgeVerdict(prerequisites_satisfied=False, score_10=3)
    grown = expand_candidate(cand, verdict, Graph([], units), units, units_by_id,
                             {}, sents, max_span_s=999.0)
    assert grown is not None
    # u0000 pulled in → span [0,3] → u0001/u0002 absorbed, referential emptied
    assert grown.unit_ids == ["u0000", "u0001", "u0002", "u0003"]
    assert grown.referential == []


def test_repair_drop_returns_rejection_with_last_verdict(monkeypatch):
    # spec update (B-judge-integrity): a kill now additionally requires ≥1 failure reason whose
    # evidence_quote passes containment AND survives fresh-context confirmation — a bare failing
    # verdict without verifiable evidence ships flagged instead. This test supplies both.
    from backend import config
    from backend.pipeline.assemble.validate import FailureReason, KillClaimCheck, KillConfirmation
    monkeypatch.setattr(config, "JUDGE_MODEL", "")
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "same")

    def fake_llm(system, user, schema, **kw):
        if schema is KillConfirmation:       # fresh-context confirmation upholds the claim
            return KillConfirmation(claims=[KillClaimCheck(claim=1, confirmed=True,
                                                           quote="sentence 0.")])
        # judged, failing, unrepairable (no targets: refs fine, topic bad), evidence quoted
        return JudgeVerdict(reasoning="bad", score_10=2, understandable=False,
                            topic_identifiable=False, purpose_identifiable=False,
                            failure_reasons=[FailureReason(kind="off_topic", detail="off topic",
                                                           evidence_quote="sentence 0.")])
    monkeypatch.setattr(llm_mod, "llm_json", fake_llm)

    sents, units = _setup()
    units_by_id = {u.unit_id: u for u in units}
    cand = Candidate(cand_id="c0", anchor_id="u0000", role="explanation", facet="other",
                     title="T", reason="r", unit_ids=["u0000"], referential=[],
                     i_start=0, i_end=0, start=sents[0].start, end=sents[0].end)
    kept, rej = validate_and_repair(cand, sents, Graph([], units), units, units_by_id, {},
                                    FakeAdapter(), {}, lambda s, e: "", "topic", {}, threading.Lock())
    assert kept is None
    assert rej is not None and rej.stage == "repair"
    assert rej.score == pytest.approx(0.2)
    assert rej.cand_id == "c0" and rej.title == "T"
    assert rej.failure_kinds == ["off_topic"]            # confirmed kinds only
    assert rej.kill_confirmed is True


def test_repair_keep_returns_no_rejection(monkeypatch):
    from backend import config
    monkeypatch.setattr(config, "JUDGE_MODEL", "")
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "same")
    monkeypatch.setattr(llm_mod, "llm_json",
                        lambda *a, **kw: JudgeVerdict(reasoning="ok", score_10=9, understandable=True))
    sents, units = _setup()
    units_by_id = {u.unit_id: u for u in units}
    cand = Candidate(cand_id="c0", anchor_id="u0000", role="explanation", facet="other",
                     title="t", reason="r", unit_ids=["u0000"], referential=[],
                     i_start=0, i_end=0, start=sents[0].start, end=sents[0].end)
    kept, rej = validate_and_repair(cand, sents, Graph([], units), units, units_by_id, {},
                                    FakeAdapter(), {}, lambda s, e: "", "topic", {}, threading.Lock())
    assert kept is not None and rej is None


# ── Part-B dedupe + snap rejections + mutation warnings ──────────────────────
from backend.pipeline.assemble.boundary_adapt import snap_candidates
from backend.pipeline.refine import _dedupe as refine_dedupe


def _cand(cand_id, s0, s1, sents, *, facet="other", fq=0.5, verdict=None):
    c = Candidate(cand_id=cand_id, anchor_id=f"u{s0:04d}", role="explanation", facet=facet,
                  title=f"t-{cand_id}", reason="r", unit_ids=[f"u{i:04d}" for i in range(s0, s1 + 1)],
                  referential=[], i_start=s0, i_end=s1,
                  start=sents[s0].start, end=sents[s1].end)
    c.final_quality = fq
    c.verdict = verdict or JudgeVerdict(score_10=8, understandable=True)
    return c


_SETTINGS = {"min_clip_duration_s": 1.0, "max_clip_duration_s": 500.0}


def test_same_facet_overlap_merges_and_flags():
    sents, units = _setup(8)
    a = _cand("a", 0, 3, sents, fq=0.9)
    b = _cand("b", 2, 5, sents, fq=0.4)              # overlaps a, same facet
    specs, rejections = snap_candidates([a, b], sents, _SETTINGS, units)
    assert len(specs) == 1 and specs[0]["merged"] is True
    assert specs[0]["sentence_start_idx"] == 0 and specs[0]["sentence_end_idx"] == 5
    assert specs[0]["unit_ids"] == [f"u{i:04d}" for i in range(6)]
    assert "merged_overlap" in specs[0]["warnings"]
    assert rejections == []                           # merge is not a drop


def test_containment_loser_becomes_dedupe_rejection():
    sents, units = _setup(8)
    big = _cand("big", 0, 5, sents, fq=0.9)
    small = _cand("small", 1, 2, sents, fq=0.2)      # contained in big
    specs, rejections = snap_candidates([big, small], sents, _SETTINGS, units)
    assert [s["cand_id"] for s in specs] == ["big"]
    assert len(rejections) == 1
    r = rejections[0]
    assert r.stage == "dedupe" and r.cand_id == "small" and "big" in r.reason


def test_dedupe_partb_matches_refine_on_disjoint_and_containment():
    """Characterization, UPDATED for P4 (spec-governs): this test froze the pre-Wave-2
    tie-break by pinning Part-B dedupe to refine._dedupe's keep decisions. P4a deliberately
    unfroze the tie-break — both paths now share refine._better's _keep_key (hard gates →
    contract coverage → final_quality). The pin still holds here because these fixtures
    carry uniform gate/coverage metadata, so decisions reduce to the final_quality leg;
    the NEW tie-break behavior is pinned in test_dedupe_severed.py."""
    sents, units = _setup(10)
    disjoint = [_cand("a", 0, 1, sents, fq=0.5), _cand("b", 4, 5, sents, fq=0.5)]
    contained = [_cand("c", 0, 5, sents, fq=0.9), _cand("d", 2, 3, sents, fq=0.1)]
    for group in (disjoint, contained):
        specs, _ = snap_candidates(list(group), sents, _SETTINGS, units)
        legacy_in = [snap_candidates([c], sents, _SETTINGS, units)[0][0] for c in group]
        legacy = refine_dedupe([dict(s) for s in legacy_in], sents, 1.0)
        assert sorted(s["start"] for s in specs) == sorted(c["start"] for c in legacy)


def test_truncated_carried_into_spec():
    sents, units = _setup()
    c = _cand("a", 0, 2, sents)
    c.truncated = True
    specs, _ = snap_candidates([c], sents, _SETTINGS, units)
    assert specs[0]["truncated"] is True


def test_boundary_score_penalizes_mutation_warnings():
    from backend.pipeline.assemble.scoring import boundary_score
    assert boundary_score(["extended_for_min_duration"]) == pytest.approx(0.90)
    assert boundary_score(["trimmed_start"]) == pytest.approx(0.90)
    assert boundary_score(["missing_context_card"]) == pytest.approx(0.90)


def test_snap_one_warns_on_min_duration_extension():
    from backend.pipeline.refine import _snap_one
    sents, _ = _setup(6)                              # each sentence ≈10 s
    clip = _snap_one({"i_start": 0, "i_end": 0, "facet": "other"}, sents,
                     False, 25.0, 0.05, 500.0)        # min_dur 25 s forces extension past s0
    assert clip is not None
    assert "extended_for_min_duration" in clip["warnings"]


# ── re-judge hook + ledger stages (pure step-5 logic exercised via assemble_clips) ──
from backend.pipeline.understand.models import ContentMap, ContentNode, DependencyGraph, Structure


def _structure(sents, units):
    n = len(sents)
    return Structure(video_id="v", units=units, dependencies=DependencyGraph(),
                     content_map=ContentMap(nodes=[ContentNode(node_id="video", level="video",
                                                               sentence_range=(0, n - 1))]))


class AnchorAdapter(FakeAdapter):
    def is_anchor_role(self, role):
        return True

    def anchor_priority(self, role):
        return 0.9

    def facet_for(self, role):
        return "other"

    def valid_roles(self):
        return {"explanation"}


def test_assemble_returns_rejections_and_rejudges_merges(monkeypatch):
    from backend import config
    from backend.pipeline.assemble import assemble_clips
    monkeypatch.setattr(config, "JUDGE_MODEL", "")
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "same")
    calls = {"n": 0}

    def good(*a, **kw):
        sch = a[2] if len(a) > 2 else kw.get("schema")
        from backend.pipeline.assemble.validate import JudgeVerdict as JV
        if sch is JV or (a and "self-contained" in str(a[0])):
            calls["n"] += 1
            return JV(reasoning="ok", score_10=9, understandable=True)
        raise AssertionError("unexpected llm call")
    monkeypatch.setattr(llm_mod, "llm_json", good)

    sents, units = _setup(8)
    st = _structure(sents, units)
    settings = {"min_clip_duration_s": 1.0, "max_clip_duration_s": 500.0,
                "min_comprehension_score": 0.7, "quality_floor": 0.0, "max_clips": 12,
                "max_anchors": 12, "closure_max_span_s": 999.0,
                # legacy selector pinned: these tests count llm_json calls on the judge path
                "anchor_selector": "priority"}
    specs, notes, rejections = assemble_clips(st, "", sents, "u", "v", settings, AnchorAdapter())
    assert isinstance(rejections, list)
    assert all(hasattr(r, "stage") for r in rejections)
    assert isinstance(specs, list) and isinstance(notes, str)


def test_quality_floor_and_max_clips_ledgered(monkeypatch):
    from backend import config
    from backend.pipeline.assemble import assemble_clips
    monkeypatch.setattr(config, "JUDGE_MODEL", "")
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "same")
    monkeypatch.setattr(llm_mod, "llm_json",
                        lambda *a, **kw: JudgeVerdict(reasoning="ok", score_10=9, understandable=True))
    sents, units = _setup(8)
    st = _structure(sents, units)
    # floor 2.0 is unreachable → every kept candidate must land in quality_floor rejections
    settings = {"min_clip_duration_s": 1.0, "max_clip_duration_s": 500.0,
                "min_comprehension_score": 0.7, "quality_floor": 2.0, "max_clips": 12,
                "max_anchors": 12, "closure_max_span_s": 999.0,
                # legacy selector pinned: these tests count llm_json calls on the judge path
                "anchor_selector": "priority"}
    specs, _notes, rejections = assemble_clips(st, "", sents, "u", "v", settings, AnchorAdapter())
    assert specs == []
    assert rejections and all(r.stage in {"quality_floor", "dedupe", "snap", "repair"} for r in rejections)
    assert any(r.stage == "quality_floor" for r in rejections)


# ── Step-4b re-judge coverage: gate failure and outage ──────────────────────
def _overlapping_structure():
    """Two units with OVERLAPPING sentence ranges → two same-facet candidates that merge."""
    from backend.pipeline.understand.models import Unit
    sents = mini_sents(6)
    units = [
        Unit(unit_id="u_a", start=sents[0].start, end=sents[3].end, sentence_range=(0, 3),
             role="explanation", transcript="a"),
        Unit(unit_id="u_b", start=sents[2].start, end=sents[5].end, sentence_range=(2, 5),
             role="explanation", transcript="b"),
    ]
    return sents, units


_4B_SETTINGS = {"min_clip_duration_s": 1.0, "max_clip_duration_s": 500.0,
                "min_comprehension_score": 0.7, "quality_floor": 0.0, "max_clips": 12,
                "max_anchors": 12, "closure_max_span_s": 999.0,
                # legacy selector pinned: these tests count llm_json calls on the judge path
                "anchor_selector": "priority"}


def test_merged_spec_failing_gate_becomes_post_merge_judge_rejection(monkeypatch):
    # spec update (Wave-1 FIX-1): the post-merge kill now routes through the same asymmetric
    # gate as the repair stage — it needs a quote-verified reason that survives fresh-context
    # confirmation. This test supplies both.
    from backend import config
    from backend.pipeline.assemble import assemble_clips
    from backend.pipeline.assemble.validate import FailureReason, KillClaimCheck, KillConfirmation
    monkeypatch.setattr(config, "JUDGE_MODEL", "")
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "same")

    def judge(system, user, schema, **kw):
        if schema is KillConfirmation:                  # fresh-context confirmation upholds it
            return KillConfirmation(claims=[KillClaimCheck(claim=1, confirmed=True,
                                                           quote="sentence 0.")])
        # repair-stage judges see single-unit clips (short); the 4b re-judge sees the union.
        # Distinguish by transcript length: the merged span covers all 6 sentences.
        n_sents = user.count("sentence")
        if n_sents >= 6:
            return JudgeVerdict(reasoning="union bad", score_10=3, understandable=False,
                                topic_identifiable=False,
                                failure_reasons=[FailureReason(kind="off_topic",
                                                               evidence_quote="sentence 0.")])
        return JudgeVerdict(reasoning="ok", score_10=9, understandable=True)
    monkeypatch.setattr(llm_mod, "llm_json", judge)

    sents, units = _overlapping_structure()
    st = _structure(sents, units)
    specs, _notes, rejections = assemble_clips(st, "", sents, "u", "v", _4B_SETTINGS, AnchorAdapter())
    stages = [r.stage for r in rejections]
    assert "post_merge_judge" in stages
    r = next(r for r in rejections if r.stage == "post_merge_judge")
    assert r.kill_confirmed is True and r.failure_kinds == ["off_topic"]
    assert all(not s.get("merged") for s in specs)      # the confirmed-failed merge did not ship


def test_merged_spec_failing_gate_unverified_ships_flagged(monkeypatch):
    # spec update (Wave-1 FIX-1): a would-kill union verdict with ZERO verified reasons ships
    # flagged instead of being killed — the old raw-boolean kill is gone at this stage.
    from backend import config
    from backend.pipeline.assemble import assemble_clips
    from backend.pipeline.assemble.validate import FailureReason, KillConfirmation
    monkeypatch.setattr(config, "JUDGE_MODEL", "")
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "same")
    schemas = []

    def judge(system, user, schema, **kw):
        schemas.append(schema)
        if user.count("sentence") >= 6:                 # union verdict: phantom-only evidence
            return JudgeVerdict(reasoning="union bad", score_10=3, understandable=False,
                                topic_identifiable=False,
                                failure_reasons=[FailureReason(kind="missing_prerequisite",
                                                               evidence_quote="ghost words")])
        return JudgeVerdict(reasoning="ok", score_10=9, understandable=True)
    monkeypatch.setattr(llm_mod, "llm_json", judge)

    sents, units = _overlapping_structure()
    st = _structure(sents, units)
    specs, _notes, rejections = assemble_clips(st, "", sents, "u", "v", _4B_SETTINGS, AnchorAdapter())
    merged = [s for s in specs if s.get("merged")]
    assert len(merged) == 1                              # shipped flagged, not killed
    assert merged[0]["ship_flagged"] is True
    assert "unverified_judge_concerns" in merged[0]["warnings"]
    assert all(r.stage != "post_merge_judge" for r in rejections)
    assert KillConfirmation not in schemas               # accept-side path: no confirm call


def test_merged_spec_outage_ships_flagged(monkeypatch):
    from backend import config
    from backend.pipeline.assemble import assemble_clips
    monkeypatch.setattr(config, "JUDGE_MODEL", "")
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "same")

    def judge(system, user, schema, **kw):
        if user.count("sentence") >= 6:                  # only the 4b union call fails
            raise RuntimeError("outage during re-judge")
        return JudgeVerdict(reasoning="ok", score_10=9, understandable=True)
    monkeypatch.setattr(llm_mod, "llm_json", judge)

    sents, units = _overlapping_structure()
    st = _structure(sents, units)
    specs, _notes, rejections = assemble_clips(st, "", sents, "u", "v", _4B_SETTINGS, AnchorAdapter())
    merged = [s for s in specs if s.get("merged")]
    assert len(merged) == 1                              # outage → ship-but-flag, not dropped
    assert merged[0]["judge_error"] is True
    assert "unjudged" in merged[0]["warnings"]
    assert all(r.stage != "post_merge_judge" for r in rejections)


# ── card integrity ────────────────────────────────────────────────────────────
from backend.pipeline.assemble.context_card import generate_context_card


def test_card_llm_failure_falls_back_to_referential_summary(monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("card llm down")
    monkeypatch.setattr(llm_mod, "llm_json", boom)
    sents, units = _setup()
    units_by_id = {u.unit_id: u for u in units}
    units_by_id["u0005"].summary = "Definition of the derivative from first principles."
    spec = {"referential": [("u0005", "prerequisite")], "anchor_id": "u0000",
            "unit_ids": ["u0000"], "truncated": True}
    card = generate_context_card(spec, units_by_id, FakeAdapter(), "derivatives")
    assert "derivative" in card.lower()                  # extractive, verbatim from summary


def test_card_skips_referential_already_inside_clip(monkeypatch):
    monkeypatch.setattr(llm_mod, "llm_json", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError()))
    sents, units = _setup()
    units_by_id = {u.unit_id: u for u in units}
    spec = {"referential": [("u0001", "prerequisite")], "anchor_id": "u0000",
            "unit_ids": ["u0000", "u0001"], "truncated": False}
    assert generate_context_card(spec, units_by_id, FakeAdapter(), "") == ""   # nothing outside clip


def test_missing_card_warning_appended():
    # pure: the step-6 helper logic — needed context, empty card → warning
    from backend.pipeline.assemble import _card_warning
    s = {"context_card": "", "referential": [("u0005", "prerequisite")], "truncated": False,
         "warnings": ("x",)}
    _card_warning(s)
    assert "missing_context_card" in s["warnings"] and "x" in s["warnings"]
    s2 = {"context_card": "has one", "referential": [("u0005", "p")], "warnings": ()}
    _card_warning(s2)
    assert "missing_context_card" not in s2["warnings"]


def test_snap_extension_repairs_metadata():
    """Min-duration extension absorbs newly-spanned units and drops inside-referential."""
    sents, units = _setup(6)                             # 10 s sentences
    c = _cand("a", 0, 0, sents)                          # 1-sentence judged span
    c.referential = [("u0002", "prerequisite")]
    specs, _ = snap_candidates([c], sents, {"min_clip_duration_s": 25.0,
                                            "max_clip_duration_s": 500.0}, units)
    s = specs[0]
    assert "extended_for_min_duration" in s["warnings"]
    assert s["sentence_end_idx"] >= 2
    assert "u0002" in s["unit_ids"]                      # absorbed
    assert ("u0002", "prerequisite") not in [tuple(r) for r in s["referential"]]


def test_successful_rejudge_clears_unjudged(monkeypatch):
    from backend import config
    from backend.pipeline.assemble import assemble_clips
    monkeypatch.setattr(config, "JUDGE_MODEL", "")
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "same")
    calls = {"n": 0}

    def judge(system, user, schema, **kw):
        calls["n"] += 1
        if calls["n"] == 1:                              # one repair-stage outage → unjudged input
            raise RuntimeError("outage")
        return JudgeVerdict(reasoning="ok", score_10=9, understandable=True)
    monkeypatch.setattr(llm_mod, "llm_json", judge)

    sents, units = _overlapping_structure()
    st = _structure(sents, units)
    specs, _notes, _rej = assemble_clips(st, "", sents, "u", "v", _4B_SETTINGS, AnchorAdapter())
    merged = [s for s in specs if s.get("merged")]
    if merged:                                           # merge occurred with an unjudged side
        m = merged[0]
        assert m["judge_error"] is False                 # successful union re-judge cleared it
        assert "unjudged" not in m["warnings"]


def test_dedupe_rejection_carries_quality():
    sents, units = _setup(8)
    big = _cand("big", 0, 5, sents, fq=0.9)
    small = _cand("small", 1, 2, sents, fq=0.2)
    _, rejections = snap_candidates([big, small], sents, {"min_clip_duration_s": 1.0,
                                                          "max_clip_duration_s": 500.0}, units)
    assert rejections and rejections[0].final_quality is not None


# ── pkg-3: relevance degraded flag + unmet-concept prereq hints ───────────────
from backend.pipeline.assemble.candidates import score_topic_relevance, RelevanceLLM, RelItem
from backend.pipeline.assemble.sequence import sequence_clips


def test_relevance_retry_then_success_not_degraded(monkeypatch):
    calls = {"n": 0}

    def flaky(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return RelevanceLLM(items=[RelItem(unit_id="u0000", score=0.9)])
    monkeypatch.setattr(llm_mod, "llm_json", flaky)
    sents, units = _setup(1)
    rel, degraded = score_topic_relevance(units, "calculus", {})
    assert degraded is False
    assert rel["u0000"] == pytest.approx(0.9)
    assert calls["n"] == 2                                  # one retry happened


def test_relevance_double_failure_flags_degraded(monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("down")
    monkeypatch.setattr(llm_mod, "llm_json", boom)
    sents, units = _setup(2)
    rel, degraded = score_topic_relevance(units, "calculus", {})
    assert degraded is True
    assert all(v == 0.5 for v in rel.values())              # neutral defaults, honestly flagged


def test_relevance_empty_topic_no_llm(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("must not be called")
    monkeypatch.setattr(llm_mod, "llm_json", boom)
    sents, units = _setup(2)
    rel, degraded = score_topic_relevance(units, "", {})
    assert degraded is False and all(v == 1.0 for v in rel.values())


def test_prereq_hint_skipped_when_self_defined():
    sents, units = _setup(6)
    units[0].concepts_introduced = ["derivative"]
    units[3].concepts_required = ["derivative"]
    units[3].concepts_introduced = ["derivative"]           # clip defines it itself
    units[5].concepts_required = ["derivative"]             # clip does NOT define it
    units_by_id = {u.unit_id: u for u in units}
    specs = [
        {"start": 0.0, "unit_ids": ["u0000"]},
        {"start": 30.0, "unit_ids": ["u0003"]},
        {"start": 50.0, "unit_ids": ["u0005"]},
    ]
    seq = sequence_clips(specs, Graph([], units), units_by_id)
    by_start = {s["start"]: s for s in seq}
    assert by_start[30.0]["prerequisite_clips"] == []       # self-defined → no hint
    assert by_start[50.0]["prerequisite_clips"] == [1]      # needs clip 1's definition
