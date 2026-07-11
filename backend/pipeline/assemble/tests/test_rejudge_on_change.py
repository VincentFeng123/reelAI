"""C-rejudge-on-change: verdicts are only trusted for the exact text they were issued on.

judged_text_hash is stored at judge time (on the outage path too — the text WAS sent);
the verdict cache is keyed on (frozenset(unit_ids), text hash); the post-snap seam
re-judges ANY spec whose final text hash differs from its judged hash (extension/trim/
cap/merge — hash-triggered, never warning-name-triggered). A failing fresh verdict routes
through the SAME asymmetric gate as the repair stage: kill only on quote-verified AND
fresh-context-confirmed evidence; otherwise ship flagged. All offline (llm_json
monkeypatched)."""
from __future__ import annotations

import threading

import backend.llm as llm_mod
from backend.pipeline.assemble.context_card import CardSentence, ContextCardDraft
from backend.pipeline.assemble.graph import Graph
from backend.pipeline.assemble.integrity import Rejection
from backend.pipeline.assemble.types import Candidate
from backend.pipeline.assemble.validate import (
    FailureReason, JudgeVerdict, KillClaimCheck, KillConfirmation, judged_text_hash,
    validate_and_repair,
)
from backend.pipeline.understand.models import (
    ContentMap, ContentNode, DependencyGraph, Edge, Structure, Unit,
)

from .conftest import FakeAdapter, mini_sents, mini_units


# ── shared fixtures ───────────────────────────────────────────────────────────
class AnchorAdapter(FakeAdapter):
    def is_anchor_role(self, role):
        return True

    def anchor_priority(self, role):
        return 0.9

    def facet_for(self, role):
        return "other"

    def valid_roles(self):
        return {"explanation"}


def _structure(sents, units):
    return Structure(video_id="v", units=units, dependencies=DependencyGraph(),
                     content_map=ContentMap(nodes=[ContentNode(node_id="video", level="video",
                                                               sentence_range=(0, len(sents) - 1))]))


def _one_unit_structure(n_sents=6):
    """Six 10s sentences but ONE unit on sentence 0 → a single 1-sentence candidate whose
    unit_ids can never change, while min-duration snapping can extend its TEXT."""
    sents = mini_sents(n_sents)
    units = [Unit(unit_id="u0000", start=sents[0].start, end=sents[0].end,
                  sentence_range=(0, 0), role="explanation", transcript=sents[0].text)]
    return sents, units


def _settings(min_dur):
    return {"min_clip_duration_s": min_dur, "max_clip_duration_s": 500.0,
            "min_comprehension_score": 0.7, "quality_floor": 0.0, "max_clips": 12,
            "max_anchors": 12, "closure_max_span_s": 999.0,
            # legacy selector pinned: these tests count judge calls / cache hits and target
            # the priority path (P3 keeps it byte-equivalent as the A/B lever)
            "anchor_selector": "priority"}


def _same_model(monkeypatch):
    from backend import config
    monkeypatch.setattr(config, "JUDGE_MODEL", "")
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "same")


def _mk_candidate(sents, i_start=0, i_end=0, cand_id="c0"):
    return Candidate(cand_id=cand_id, anchor_id="u0000", role="explanation", facet="other",
                     title="t", reason="r", unit_ids=["u0000"], referential=[],
                     i_start=i_start, i_end=i_end,
                     start=sents[i_start].start, end=sents[i_end].end)


def _vr(cand, sents, units, cache):
    units_by_id = {u.unit_id: u for u in units}
    return validate_and_repair(cand, sents, Graph([], units), units, units_by_id, {},
                               FakeAdapter(), {}, lambda s, e: "", "topic",
                               cache, threading.Lock())


# ── C1: judged_text_hash stored at judge time ────────────────────────────────
def test_hash_normalizes_case_and_whitespace_only():
    assert judged_text_hash("Sentence  0.") == judged_text_hash("sentence 0.")
    assert judged_text_hash(" sentence 0.\n") == judged_text_hash("sentence 0.")
    assert judged_text_hash("sentence 0.") != judged_text_hash("sentence 0. sentence 1.")
    assert judged_text_hash("") != ""                     # a real hash, never the sentinel


def test_judged_text_hash_recorded_on_candidate(monkeypatch):
    _same_model(monkeypatch)
    monkeypatch.setattr(llm_mod, "llm_json",
                        lambda *a, **kw: JudgeVerdict(reasoning="ok", score_10=9, understandable=True))
    sents = mini_sents(3)
    units = mini_units(sents)
    kept, rej = _vr(_mk_candidate(sents), sents, units, {})
    assert rej is None and kept is not None
    assert kept.judged_text_hash == judged_text_hash("sentence 0.")


def test_error_verdict_records_hash_of_text_sent(monkeypatch):
    # spec update (Wave-1 FIX-4): the outage path stores the hash of the text that WAS sent
    # to the judge, so an UNCHANGED outage spec never spends an extra call at the 4b seam
    # (it still re-judges whenever the text actually changed — hash mismatch).
    _same_model(monkeypatch)

    def boom(*a, **kw):
        raise RuntimeError("api down")
    monkeypatch.setattr(llm_mod, "llm_json", boom)
    sents = mini_sents(3)
    units = mini_units(sents)
    kept, _rej = _vr(_mk_candidate(sents), sents, units, {})
    assert kept is not None and kept.verdict.error is True
    assert kept.judged_text_hash == judged_text_hash("sentence 0.")


# ── C2: cache keyed on (unit_ids, judged text) ───────────────────────────────
def test_cache_hit_for_byte_identical_convergent_candidates(monkeypatch):
    _same_model(monkeypatch)
    calls = {"n": 0}

    def judge(*a, **kw):
        calls["n"] += 1
        return JudgeVerdict(reasoning="ok", score_10=9, understandable=True)
    monkeypatch.setattr(llm_mod, "llm_json", judge)
    sents = mini_sents(3)
    units = mini_units(sents)
    cache: dict = {}
    a, _ = _vr(_mk_candidate(sents, cand_id="cA"), sents, units, cache)
    b, _ = _vr(_mk_candidate(sents, cand_id="cB"), sents, units, cache)
    assert calls["n"] == 1                                # convergent candidates share the verdict
    assert a.verdict is b.verdict
    assert a.judged_text_hash == b.judged_text_hash


def test_cache_miss_on_text_change_with_same_unit_ids(monkeypatch):
    _same_model(monkeypatch)
    calls = {"n": 0}

    def judge(*a, **kw):
        calls["n"] += 1
        return JudgeVerdict(reasoning="ok", score_10=9, understandable=True)
    monkeypatch.setattr(llm_mod, "llm_json", judge)
    sents = mini_sents(3)
    units = [mini_units(sents)[0]]                        # one unit → unit_ids identical below
    cache: dict = {}
    a, _ = _vr(_mk_candidate(sents, 0, 0, "cA"), sents, units, cache)
    b, _ = _vr(_mk_candidate(sents, 0, 1, "cB"), sents, units, cache)   # same unit_ids, more text
    assert a.unit_ids == b.unit_ids == ["u0000"]
    assert calls["n"] == 2                                # text differs → stale verdict NOT reused
    assert a.judged_text_hash != b.judged_text_hash


# ── C3/C4: post-snap re-judge triggered by hash difference only ──────────────
def _count_judge(monkeypatch, on_extended=None, confirm=None):
    """Fake llm_json counting judge calls; `on_extended(user)` overrides the verdict for the
    post-snap call (detected by the extended text reaching past sentence 0); `confirm(user)`
    answers confirm_kill's KillConfirmation call (None → asserts confirm_kill never runs)."""
    _same_model(monkeypatch)
    calls = {"n": 0, "users": [], "confirms": 0}

    def judge(system, user, schema, **kw):
        if schema is KillConfirmation:
            assert confirm is not None, "confirm_kill must not be called on this path"
            calls["confirms"] += 1
            return confirm(user)
        assert schema is JudgeVerdict
        calls["n"] += 1
        calls["users"].append(user)
        if on_extended is not None and "sentence 2." in user:
            return on_extended(user)
        return JudgeVerdict(reasoning="ok", score_10=9, understandable=True)
    monkeypatch.setattr(llm_mod, "llm_json", judge)
    return calls


def test_unchanged_spec_spends_no_extra_judge_call(monkeypatch):
    from backend.pipeline.assemble import assemble_clips
    calls = _count_judge(monkeypatch)
    sents, units = _one_unit_structure()
    specs, _notes, rejections = assemble_clips(_structure(sents, units), "", sents, "u", "v",
                                               _settings(min_dur=1.0), AnchorAdapter())
    assert len(specs) == 1
    assert calls["n"] == 1                                # judged once at repair time, that's all
    assert rejections == []


def test_min_duration_extension_triggers_rejudge(monkeypatch):
    from backend.pipeline.assemble import assemble_clips
    calls = _count_judge(monkeypatch)
    sents, units = _one_unit_structure()
    specs, _notes, rejections = assemble_clips(_structure(sents, units), "", sents, "u", "v",
                                               _settings(min_dur=25.0), AnchorAdapter())
    assert len(specs) == 1
    s = specs[0]
    assert "extended_for_min_duration" in s["warnings"]
    assert calls["n"] == 2                                # repair judge + post-snap re-judge
    assert "sentence 2." in calls["users"][-1]            # re-judge saw the FINAL extended text
    final_text = " ".join(x.text for x in sents[s["sentence_start_idx"]:s["sentence_end_idx"] + 1])
    assert s["judged_text_hash"] == judged_text_hash(final_text)   # verdict covers shipped text
    assert s["judge_error"] is False and "unjudged" not in s["warnings"]
    assert rejections == []


def test_extension_rejudge_rejection_ledgered_as_post_snap_judge(monkeypatch):
    # spec update (Wave-1 FIX-1): the post-snap gate is now the SAME asymmetric gate as the
    # repair stage — a kill needs a quote-verified reason that survives fresh-context
    # confirmation. This test supplies both; the re-judge itself still fires on text change.
    from backend.pipeline.assemble import assemble_clips

    def bad(user):
        return JudgeVerdict(reasoning="extended text broke it", score_10=3,
                            understandable=False, topic_identifiable=False,
                            failure_reasons=[FailureReason(kind="off_topic",
                                                           evidence_quote="sentence 1.")])

    def confirm(user):
        return KillConfirmation(claims=[KillClaimCheck(claim=1, confirmed=True,
                                                       quote="sentence 1.")])
    calls = _count_judge(monkeypatch, on_extended=bad, confirm=confirm)
    sents, units = _one_unit_structure()
    specs, _notes, rejections = assemble_clips(_structure(sents, units), "", sents, "u", "v",
                                               _settings(min_dur=25.0), AnchorAdapter())
    assert specs == []                                    # confirmed kill did not ship
    stages = [r.stage for r in rejections]
    assert "post_snap_judge" in stages
    r = next(r for r in rejections if r.stage == "post_snap_judge")
    assert r.cand_id == "c_u0000" and r.score == 0.3
    assert r.kill_confirmed is True
    assert r.failure_kinds == ["off_topic"]               # confirmed kinds only
    assert calls["n"] == 2                                # repair judge + post-snap re-judge
    assert calls["confirms"] == 1                         # exactly one fresh-context call


def test_extension_rejudge_unverified_kill_ships_flagged(monkeypatch):
    # spec update (Wave-1 FIX-1): the empirical NjvwWiCYLl4 shape — the fresh verdict would
    # kill but NO failure reason survives quote verification → ship flagged, never kill,
    # and confirm_kill is never called (accept-side path spends no extra LLM call).
    from backend.pipeline.assemble import assemble_clips

    def bad(user):
        return JudgeVerdict(reasoning="extended text broke it", score_10=3,
                            understandable=False, topic_identifiable=False,
                            failure_reasons=[FailureReason(kind="missing_prerequisite",
                                                           evidence_quote="ghost words")])
    calls = _count_judge(monkeypatch, on_extended=bad)    # confirm=None → must never run
    sents, units = _one_unit_structure()
    specs, _notes, rejections = assemble_clips(_structure(sents, units), "", sents, "u", "v",
                                               _settings(min_dur=25.0), AnchorAdapter())
    assert len(specs) == 1 and rejections == []           # shipped flagged instead of killed
    s = specs[0]
    assert s["ship_flagged"] is True
    assert "unverified_judge_concerns" in s["warnings"]
    assert "kill_confirm_unavailable" not in s["warnings"]
    assert s["n_failure_reasons"] == 1 and s["n_verified"] == 0
    assert calls["n"] == 2 and calls["confirms"] == 0


def test_extension_rejudge_verified_but_unconfirmed_ships_flagged(monkeypatch):
    from backend.pipeline.assemble import assemble_clips

    def bad(user):
        return JudgeVerdict(reasoning="bad", score_10=3, understandable=False,
                            topic_identifiable=False,
                            failure_reasons=[FailureReason(kind="off_topic",
                                                           evidence_quote="sentence 1.")])

    def confirm(user):                                    # fresh context does NOT uphold it
        return KillConfirmation(claims=[KillClaimCheck(claim=1, confirmed=False,
                                                       quote="sentence 1.")])
    calls = _count_judge(monkeypatch, on_extended=bad, confirm=confirm)
    sents, units = _one_unit_structure()
    specs, _notes, rejections = assemble_clips(_structure(sents, units), "", sents, "u", "v",
                                               _settings(min_dur=25.0), AnchorAdapter())
    assert len(specs) == 1 and rejections == []
    assert specs[0]["ship_flagged"] is True
    assert "unverified_judge_concerns" in specs[0]["warnings"]
    assert calls["confirms"] == 1


def test_extension_rejudge_confirm_outage_ships_flagged_with_marker(monkeypatch):
    from backend.pipeline.assemble import assemble_clips

    def bad(user):
        return JudgeVerdict(reasoning="bad", score_10=3, understandable=False,
                            topic_identifiable=False,
                            failure_reasons=[FailureReason(kind="off_topic",
                                                           evidence_quote="sentence 1.")])

    def confirm(user):
        raise RuntimeError("confirm outage")              # conservative no-kill
    _count_judge(monkeypatch, on_extended=bad, confirm=confirm)
    sents, units = _one_unit_structure()
    specs, _notes, rejections = assemble_clips(_structure(sents, units), "", sents, "u", "v",
                                               _settings(min_dur=25.0), AnchorAdapter())
    assert len(specs) == 1 and rejections == []
    assert "unverified_judge_concerns" in specs[0]["warnings"]
    assert "kill_confirm_unavailable" in specs[0]["warnings"]   # existing outage marker
    assert specs[0]["ship_flagged"] is True


def test_extension_rejudge_outage_ships_but_flags(monkeypatch):
    from backend.pipeline.assemble import assemble_clips
    _same_model(monkeypatch)

    def judge(system, user, schema, **kw):
        if "sentence 2." in user:                         # only the post-snap re-judge fails
            raise RuntimeError("outage during re-judge")
        return JudgeVerdict(reasoning="ok", score_10=9, understandable=True)
    monkeypatch.setattr(llm_mod, "llm_json", judge)
    sents, units = _one_unit_structure()
    specs, _notes, rejections = assemble_clips(_structure(sents, units), "", sents, "u", "v",
                                               _settings(min_dur=25.0), AnchorAdapter())
    assert len(specs) == 1                                # outage → ship-but-flag, never a drop
    assert specs[0]["judge_error"] is True
    assert "unjudged" in specs[0]["warnings"]
    assert all(r.stage != "post_snap_judge" for r in rejections)


# ── 4b stats (integration): phantom_verdict_rate inputs stay truthful ─────────
def test_post_snap_rejection_records_verified_and_unverified_kinds(monkeypatch):
    # spec update (Wave-1 FIX-1): the 4b kill now requires confirmation, so the ledgered
    # rejection carries kill_confirmed=True and failure_kinds = confirmed kinds only.
    from backend.pipeline.assemble import assemble_clips

    def bad(user):
        return JudgeVerdict(reasoning="extended text broke it", score_10=3,
                            understandable=False, topic_identifiable=False,
                            failure_reasons=[
                                FailureReason(kind="off_topic", evidence_quote="sentence 1."),
                                FailureReason(kind="missing_result", evidence_quote="ghost words"),
                            ])

    def confirm(user):
        return KillConfirmation(claims=[KillClaimCheck(claim=1, confirmed=True,
                                                       quote="sentence 1.")])
    _count_judge(monkeypatch, on_extended=bad, confirm=confirm)
    sents, units = _one_unit_structure()
    specs, _notes, rejections = assemble_clips(_structure(sents, units), "", sents, "u", "v",
                                               _settings(min_dur=25.0), AnchorAdapter())
    assert specs == []
    r = next(r for r in rejections if r.stage == "post_snap_judge")
    assert r.verified_kinds == ("off_topic",)             # quote contained in the final text
    assert r.unverified_kinds == ("missing_result",)      # phantom quote
    assert r.failure_kinds == ["off_topic"]
    assert r.kill_confirmed is True                       # 4b kills only WITH confirmation now


def test_post_snap_success_refreshes_phantom_stats(monkeypatch):
    from backend.pipeline.assemble import assemble_clips

    def ok_with_reasons(user):
        return JudgeVerdict(reasoning="fine now", score_10=9, understandable=True,
                            failure_reasons=[
                                FailureReason(kind="other", evidence_quote="sentence 2."),
                                FailureReason(kind="other", evidence_quote="ghost words"),
                            ])
    _count_judge(monkeypatch, on_extended=ok_with_reasons)
    sents, units = _one_unit_structure()
    specs, _notes, rejections = assemble_clips(_structure(sents, units), "", sents, "u", "v",
                                               _settings(min_dur=25.0), AnchorAdapter())
    assert len(specs) == 1 and rejections == []
    s = specs[0]
    # stats now describe the verdict covering the SHIPPED text (repair verdict had 0 reasons)
    assert s["n_failure_reasons"] == 2 and s["n_verified"] == 1


# ── Wave-1 FIX-3: a clean fresh verdict supersedes a stale repair-gate flag ───
def _flagged_at_repair_then(monkeypatch, on_extended):
    """Drive the repair gate to ship-flag (verified failing verdict + confirm outage →
    'unverified_judge_concerns' + 'kill_confirm_unavailable'), then extend the text so the
    4b seam re-judges with `on_extended`."""
    from backend.pipeline.assemble import assemble_clips
    _same_model(monkeypatch)

    def judge(system, user, schema, **kw):
        if schema is KillConfirmation:
            raise RuntimeError("confirm outage at the repair gate")
        assert schema is JudgeVerdict
        if "sentence 2." in user:                         # the post-snap re-judge (extended text)
            return on_extended(user)
        return JudgeVerdict(reasoning="bad", score_10=2, understandable=False,
                            topic_identifiable=False, purpose_identifiable=False,
                            failure_reasons=[FailureReason(kind="off_topic",
                                                           evidence_quote="sentence 0.")])
    monkeypatch.setattr(llm_mod, "llm_json", judge)
    sents, units = _one_unit_structure()
    return assemble_clips(_structure(sents, units), "", sents, "u", "v",
                          _settings(min_dur=25.0), AnchorAdapter())


def test_clean_rejudge_clears_stale_ship_flag(monkeypatch):
    def clean(user):
        return JudgeVerdict(reasoning="extended text is fine", score_10=9, understandable=True)
    specs, _notes, rejections = _flagged_at_repair_then(monkeypatch, clean)
    assert len(specs) == 1 and rejections == []
    s = specs[0]
    # the clean verdict covers the FINAL text → stale repair-gate flags are superseded
    assert s["ship_flagged"] is False
    assert "unverified_judge_concerns" not in s["warnings"]
    assert "kill_confirm_unavailable" not in s["warnings"]


def test_failing_unverified_rejudge_keeps_ship_flag(monkeypatch):
    def still_bad(user):                                  # fails hard core, phantom-only evidence
        return JudgeVerdict(reasoning="still bad", score_10=3, understandable=False,
                            topic_identifiable=False,
                            failure_reasons=[FailureReason(kind="missing_prerequisite",
                                                           evidence_quote="ghost words")])
    specs, _notes, rejections = _flagged_at_repair_then(monkeypatch, still_bad)
    assert len(specs) == 1 and rejections == []           # unverified → still never killed
    s = specs[0]
    assert s["ship_flagged"] is True
    assert "unverified_judge_concerns" in s["warnings"]   # flag kept (verdict is NOT clean)


# ── Wave-1 FIX-4: unchanged outage specs spend no extra judge call ────────────
def test_unchanged_outage_spec_spends_no_extra_judge_call(monkeypatch):
    from backend.pipeline.assemble import assemble_clips
    _same_model(monkeypatch)
    calls = {"n": 0}

    def judge(system, user, schema, **kw):
        calls["n"] += 1
        raise RuntimeError("api down")
    monkeypatch.setattr(llm_mod, "llm_json", judge)
    sents, units = _one_unit_structure()
    specs, _notes, rejections = assemble_clips(_structure(sents, units), "", sents, "u", "v",
                                               _settings(min_dur=1.0), AnchorAdapter())
    assert len(specs) == 1 and rejections == []
    assert calls["n"] == 1                                # the outage call only — no 4b re-spend
    assert specs[0]["judge_error"] is True
    assert "unjudged" in specs[0]["warnings"]             # still honestly flagged


def test_changed_outage_spec_is_rejudged(monkeypatch):
    from backend.pipeline.assemble import assemble_clips
    _same_model(monkeypatch)
    calls = {"n": 0}

    def judge(system, user, schema, **kw):
        calls["n"] += 1
        if calls["n"] == 1:                               # repair-stage outage…
            raise RuntimeError("api down")
        return JudgeVerdict(reasoning="ok", score_10=9, understandable=True)
    monkeypatch.setattr(llm_mod, "llm_json", judge)
    sents, units = _one_unit_structure()
    specs, _notes, rejections = assemble_clips(_structure(sents, units), "", sents, "u", "v",
                                               _settings(min_dur=25.0), AnchorAdapter())
    assert len(specs) == 1 and rejections == []
    assert calls["n"] == 2                                # …but the CHANGED text is re-judged
    assert specs[0]["judge_error"] is False
    assert "unjudged" not in specs[0]["warnings"]         # successful re-judge cleared it


# ── C3: ledger + CLI/eval safety for the new stage literal ───────────────────
def test_post_snap_judge_rejection_prints_and_counts_safely():
    # exercises the REAL cli/eval code paths (not inline copies) so drift is caught here.
    import backend.eval.run_eval as R
    from backend.cli import _drop_line
    r = Rejection(cand_id="c1", title="t", role="claim", stage="post_snap_judge",
                  reason="post-snap text change failed hard-core judge gate")
    assert _drop_line(r).startswith("  [dropped/post_snap_judge]")   # stage-agnostic CLI line
    # eval's stage-keyed rejections_* columns include the new stage (wired at integration)
    assert "post_snap_judge" in R.REJECTION_STAGES
    cols = R._integrity_columns([], [r])
    assert cols["rejections_post_snap_judge"] == 1
    assert sum(v for k, v in cols.items() if k.startswith("rejections_")) == 1


# ── CARD3: card-as-repair at the post-snap/post-merge kill gate ───────────────
# A spec that ships clean at repair, then has its TEXT changed by a min-duration snap, then
# fails the fresh hard-core gate on a prereq/reference-family reason (verified+confirmed) is
# RESCUED by a grounded context card on a re-judge of the SAME (extended) span — never a new
# Rejection. The far prereq (a graph 'requires' edge > CLOSURE_MAX_GAP_S) stays REFERENTIAL,
# so it feeds the rescue card without being grown into the clip.
CARD_TEXT = "the UAM equations were derived earlier"


def _prereq_structure():
    """Anchor u0000 (sentence 0) + a FAR prereq u0005 (sentence 5, ~40s away → referential)
    whose summary grounds the rescue card; a graph 'requires' edge seeds the closure."""
    sents = mini_sents(6)
    u0 = Unit(unit_id="u0000", start=sents[0].start, end=sents[0].end, sentence_range=(0, 0),
              role="explanation", transcript=sents[0].text)
    u5 = Unit(unit_id="u0005", start=sents[5].start, end=sents[5].end, sentence_range=(5, 5),
              role="explanation", transcript=sents[5].text, summary=CARD_TEXT,
              concepts_introduced=["UAM equations"])
    units = [u0, u5]
    st = Structure(video_id="v", units=units,
                   dependencies=DependencyGraph(
                       edges=[Edge(source="u0000", target="u0005", relation="requires")]),
                   content_map=ContentMap(nodes=[ContentNode(node_id="video", level="video",
                                                             sentence_range=(0, len(sents) - 1))]))
    return sents, units, st


def _prereq_settings():
    # max_anchors=1 → only u0000 becomes a candidate; u0005 stays referential-only (no refund
    # round: budget 1 == 1 spec). min_dur=25 forces the post-snap text change.
    return {"min_clip_duration_s": 25.0, "max_clip_duration_s": 500.0,
            "min_comprehension_score": 0.7, "quality_floor": 0.0, "max_clips": 1,
            "max_anchors": 1, "closure_max_span_s": 999.0, "anchor_selector": "priority"}


def _prereq_judge(monkeypatch, carded_pass=True, prereq_quote="sentence 1.", seen=None):
    _same_model(monkeypatch)

    def judge(system, user, schema, **kw):
        if seen is not None:
            seen.append(schema)
        if schema is KillConfirmation:
            return KillConfirmation(claims=[KillClaimCheck(claim=1, confirmed=True,
                                                           quote="sentence 1.")])
        if schema is ContextCardDraft:
            return ContextCardDraft(sentences=[CardSentence(text=CARD_TEXT,
                                                            source_unit_id="u0005")])
        assert schema is JudgeVerdict
        if "sentence 2." in user:                        # the extended post-snap span
            if "CONTEXT CARD" in user:                   # the carded re-judge
                if carded_pass:
                    return JudgeVerdict(reasoning="card fixes it", score_10=9,
                                        understandable=True, prerequisites_satisfied=True,
                                        all_references_resolved=True, source_grounded=True)
                return JudgeVerdict(reasoning="still broken", score_10=3, understandable=False,
                                    source_grounded=False, prerequisites_satisfied=False,
                                    failure_reasons=[FailureReason(kind="missing_prerequisite",
                                                                   evidence_quote="sentence 1.")])
            return JudgeVerdict(reasoning="assumes prereq", score_10=3, understandable=False,
                                source_grounded=False, prerequisites_satisfied=False,
                                failure_reasons=[FailureReason(kind="missing_prerequisite",
                                                               evidence_quote=prereq_quote)])
        return JudgeVerdict(reasoning="native ok", score_10=9, understandable=True)
    monkeypatch.setattr(llm_mod, "llm_json", judge)


def test_post_snap_card_rescue_ships_same_span(monkeypatch):
    from backend.pipeline.assemble import assemble_clips
    seen: list = []
    _prereq_judge(monkeypatch, carded_pass=True, seen=seen)
    sents, units, st = _prereq_structure()
    specs, _notes, rejections = assemble_clips(st, "", sents, "u", "v", _prereq_settings(),
                                               AnchorAdapter())
    assert len(specs) == 1 and rejections == []          # rescued at the post-snap gate, not killed
    s = specs[0]
    assert s["context_card"] == CARD_TEXT
    assert "card_completed" in s["warnings"]
    assert s["ship_flagged"] is False
    assert s["unit_ids"] == ["u0000"]                    # far prereq stayed referential (no growth)
    assert s["sentence_start_idx"] == 0 and s["sentence_end_idx"] == 2   # the extended snap span
    assert "missing_context_card" not in s["warnings"]   # card_completed clip not double-docked
    assert ContextCardDraft in seen and KillConfirmation in seen


def test_post_snap_unverified_reason_never_rescues_ships_flagged(monkeypatch):
    # unverified_kill=0 at the post-snap gate: a prereq-family reason with a PHANTOM quote is
    # never verified → confirm_kill and card-rescue are both skipped → ships flagged.
    from backend.pipeline.assemble import assemble_clips
    seen: list = []
    _prereq_judge(monkeypatch, prereq_quote="ghost words", seen=seen)
    sents, units, st = _prereq_structure()
    specs, _notes, rejections = assemble_clips(st, "", sents, "u", "v", _prereq_settings(),
                                               AnchorAdapter())
    assert len(specs) == 1 and rejections == []          # unverified → ships flagged, never killed
    s = specs[0]
    assert s["ship_flagged"] is True
    assert "unverified_judge_concerns" in s["warnings"]
    assert "card_completed" not in s["warnings"]         # the rescue seam did NOT fire
    # phantom quote never reaches confirm_kill → the gate's card-rescue is never attempted;
    # any ContextCardDraft here is step-6's normal card for the shipped (flagged) clip.
    assert KillConfirmation not in seen


def test_post_snap_non_family_reason_kills_no_rescue(monkeypatch):
    # a non-family confirmed reason (off_topic) is NOT card-rescuable → the post-snap kill
    # stands (guard mutation check for the post-snap gate).
    from backend.pipeline.assemble import assemble_clips
    _same_model(monkeypatch)
    seen: list = []

    def judge(system, user, schema, **kw):
        seen.append(schema)
        if schema is KillConfirmation:
            return KillConfirmation(claims=[KillClaimCheck(claim=1, confirmed=True,
                                                           quote="sentence 1.")])
        assert schema is JudgeVerdict
        if "sentence 2." in user:                        # post-snap span → non-family kill
            return JudgeVerdict(reasoning="off the rails", score_10=3, understandable=False,
                                topic_identifiable=False,
                                failure_reasons=[FailureReason(kind="off_topic",
                                                               evidence_quote="sentence 1.")])
        return JudgeVerdict(reasoning="native ok", score_10=9, understandable=True)
    monkeypatch.setattr(llm_mod, "llm_json", judge)
    sents, units, st = _prereq_structure()
    specs, _notes, rejections = assemble_clips(st, "", sents, "u", "v", _prereq_settings(),
                                               AnchorAdapter())
    assert specs == []                                   # off_topic is not card-rescuable → killed
    r = next(r for r in rejections if r.stage == "post_snap_judge")
    assert r.kill_confirmed is True and r.failure_kinds == ["off_topic"]
    assert ContextCardDraft not in seen                  # rescue not attempted (guard held)
