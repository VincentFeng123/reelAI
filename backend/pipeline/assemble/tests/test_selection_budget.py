"""Q1 selection & budget quick wins (Wave 2.5), all offline:

Q1a  content-scaled anchor budget (compute_anchor_budget) + the ship-cap seam
     (effective_max_clips: max_clips=None inherits max(MAX_SEGMENTS, budget); an
     explicit user dial is respected exactly, even when smaller);
Q1b  topic-spread tie-break in the legacy selector (no content-map node monopolizes);
Q1c  per-topic role cap inside select_anchors (W25-C: PLAN_ROLE_CAP_PER_TOPIC keyed
     (role, node_id) — the priority/fallback arm gets the same protection the plan
     path has);
Q1d  relevance bypass when the query IS the video's own subject (verbatim containment /
     token_set_ratio against detection rationale, title, content-map root title);
Q1e  refund loop (CCQGen): after snap/dedupe, residual anchor-eligible units get up to
     two more selection rounds through the IDENTICAL machinery — bounded, zero-yield
     break, shared rejection ledger, per-round stats.
"""
from __future__ import annotations

from collections import Counter

import backend.llm as llm_mod
from backend import config
from backend.adapters.base import BaseAdapter
from backend.pipeline.assemble import assemble_clips, effective_max_clips, scoring
from backend.pipeline.assemble.candidates import (
    RelevanceLLM, compute_anchor_budget, select_anchors, topic_matches_subject,
)
from backend.pipeline.assemble.validate import (
    FailureReason, JudgeVerdict, KillClaimCheck, KillConfirmation,
)
from backend.pipeline.understand.models import (
    ContentMap, ContentNode, DependencyGraph, DetectionResult, Structure, Unit,
)

from .conftest import mini_sents


def _unit(i: int, role: str, sent_range=None, node_id: str = "") -> Unit:
    s0, s1 = sent_range if sent_range is not None else (i, i)
    return Unit(unit_id=f"u{i:04d}", start=s0 * 10.0, end=s1 * 10.0 + 9.9,
                sentence_range=(s0, s1), role=role, node_id=node_id,
                summary=f"unit {i}", transcript=f"sentence {i}.")


def _units(roles) -> list[Unit]:
    return [_unit(i, r) for i, r in enumerate(roles)]


def _rel(units, default: float = 1.0) -> dict[str, float]:
    return {u.unit_id: default for u in units}


def _det(density: str = "medium", rationale: str = "") -> DetectionResult:
    return DetectionResult(density=density, rationale=rationale)


def _structure(units, n_sents: int, *, title: str = "", rationale: str = "",
               root_title: str = "", density: str = "medium") -> Structure:
    nodes = [ContentNode(node_id="video", level="video", title=root_title,
                         sentence_range=(0, max(n_sents - 1, 0)))]
    return Structure(video_id="v", title=title, units=list(units),
                     detection=_det(density, rationale),
                     dependencies=DependencyGraph(),
                     content_map=ContentMap(root_id="video", nodes=nodes))


def _same_model(monkeypatch):
    monkeypatch.setattr(config, "JUDGE_MODEL", "")
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "same")


# ── Q1a: budget arithmetic ────────────────────────────────────────────────────────────────
def test_budget_75_eligible_high_density_is_23():
    # the qP-9wwRrJbg acceptance arithmetic: min(ceil(75/4) + 4, 32) = 23
    units = _units(["claim"] * 75)
    b = compute_anchor_budget(units, ContentMap(), _det("high"), {"max_anchors": None},
                              adapter=BaseAdapter())
    assert b == 23


def test_budget_small_video_stays_at_floor():
    units = _units(["claim"] * 20)                 # ceil(20/4) = 5 → floor
    b = compute_anchor_budget(units, ContentMap(), _det("medium"), {},
                              adapter=BaseAdapter())
    assert b == config.MAX_ANCHORS == 12


def test_budget_ceiling_clamp():
    units = _units(["claim"] * 200)                # ceil(50) + 4 = 54 → ceiling
    b = compute_anchor_budget(units, ContentMap(), _det("high"), {},
                              adapter=BaseAdapter())
    assert b == config.MAX_ANCHORS_CEIL == 32


def test_budget_density_bonus_only_when_high():
    units = _units(["claim"] * 76)                 # ceil(76/4) = 19
    med = compute_anchor_budget(units, ContentMap(), _det("medium"), {}, adapter=BaseAdapter())
    high = compute_anchor_budget(units, ContentMap(), _det("high"), {}, adapter=BaseAdapter())
    assert (med, high) == (19, 23)


def test_budget_explicit_user_dial_wins_outright():
    units = _units(["claim"] * 200)
    assert compute_anchor_budget(units, ContentMap(), _det("high"), {"max_anchors": 5},
                                 adapter=BaseAdapter()) == 5
    assert compute_anchor_budget(units, ContentMap(), _det("high"), {"max_anchors": 12},
                                 adapter=BaseAdapter()) == 12


def test_budget_counts_only_anchor_eligible_units():
    # 100 transitions/administrative are not anchor roles: 8 eligible → floor 12
    units = _units(["claim"] * 8 + ["transition", "administrative"] * 50)
    b = compute_anchor_budget(units, ContentMap(), _det("high"), {}, adapter=BaseAdapter())
    assert b == 12


def test_budget_respects_relevance_floor_in_eligibility():
    units = _units(["claim"] * 75)
    rel = {u.unit_id: (1.0 if i < 15 else 0.0) for i, u in enumerate(units)}
    b = compute_anchor_budget(units, ContentMap(), _det("high"), {},
                              adapter=BaseAdapter(), relevance=rel)
    assert b == 12                                 # ceil(15/4)+4 = 8 → floor


def test_defaults_inherit_pattern_and_new_knobs():
    assert config.DEFAULTS["max_anchors"] is None  # None → computed per-video budget
    assert config.DEFAULTS["max_clips"] is None    # None → max(MAX_SEGMENTS, budget)
    assert config.DEFAULTS["refund_rounds"] is None
    assert config.MAX_ANCHORS == 12
    assert config.MAX_ANCHORS_CEIL == 32
    assert config.REFUND_ROUNDS == 2


# ── Q1a: the final ship cap never undercuts the budget (explicit dial respected) ─────────
def test_effective_max_clips_inherits_budget_when_unset():
    assert effective_max_clips({"max_clips": None}, 23) == 23
    assert effective_max_clips({}, 23) == 23
    assert effective_max_clips({"max_clips": None}, 5) == config.MAX_SEGMENTS


def test_effective_max_clips_explicit_dial_respected_even_smaller():
    assert effective_max_clips({"max_clips": 5}, 23) == 5
    assert effective_max_clips({"max_clips": 40}, 23) == 40
    assert effective_max_clips({"max_clips": 12}, 23) == 12


# ── Q1b: topic-spread tie-break in the legacy selector ────────────────────────────────────
def _three_node_units() -> list[Unit]:
    """3 content-map nodes × (2 claims + 2 definitions), time-ordered by node — the
    stable-sort front-loading shape (all claims tie on priority)."""
    units, i = [], 0
    for n in range(3):
        for role in ("claim", "claim", "definition", "definition"):
            units.append(_unit(i, role, node_id=f"c0.t{n}"))
            i += 1
    return units


def test_tiebreak_spreads_across_content_map_nodes():
    units = _three_node_units()
    picked = select_anchors(units, _rel(units), BaseAdapter(), {"max_anchors": 6})
    counts = Counter(u.node_id for u in picked)
    assert len(picked) == 6
    assert set(counts) == {"c0.t0", "c0.t1", "c0.t2"}
    assert max(counts.values()) == 2               # no node monopolizes
    # the old stable sort put the first picks all in node t0 (video order); now the
    # first three picks cover three distinct nodes
    assert len({u.node_id for u in picked[:3]}) == 3


def test_tiebreak_is_deterministic():
    units = _three_node_units()
    a = select_anchors(units, _rel(units), BaseAdapter(), {"max_anchors": 6})
    b = select_anchors(units, _rel(units), BaseAdapter(), {"max_anchors": 6})
    assert [u.unit_id for u in a] == [u.unit_id for u in b]


def test_units_without_node_id_are_never_penalized():
    # all node_id "" → the penalty is constant zero and pure score order decides
    units = _units(["claim", "definition", "explanation", "summary"])
    picked = select_anchors(units, _rel(units), BaseAdapter(), {"max_anchors": 4})
    assert [u.role for u in picked] == ["claim", "definition", "explanation", "summary"]


def test_score_order_wins_over_node_spread_when_scores_differ():
    """Q1b fixer: node spread is a TIE-BREAK, score is PRIMARY. Pool [A:.8, A:.8, B:.7,
    '':.7] at budget 3 must keep BOTH .8 anchors — the inverted (node-primary) key
    dropped the second .8 result for the lower-scored unseen-node claims."""
    units = [_unit(0, "result", node_id="c0.tA"),      # .80
             _unit(1, "result", node_id="c0.tA"),      # .80 — must not be dropped
             _unit(2, "claim", node_id="c0.tB"),       # .70
             _unit(3, "claim")]                        # .70, unmapped
    picked = select_anchors(units, _rel(units), BaseAdapter(), {"max_anchors": 3})
    assert [u.unit_id for u in picked] == ["u0000", "u0001", "u0002"]


def test_unmapped_units_never_dominate_higher_scored_mapped_units():
    """The '' permanent-zero-penalty artifact of the inverted key: once every mapped node
    had contributed one anchor, unmapped units beat ALL mapped units regardless of score.
    Score-primary keeps the mapped .8 results ahead of the unmapped .7 claims — node A's
    second result is picked before any claim (W25-C: node B carries the third .8 result,
    since A's two slots for ('result', tA) are the per-topic cap)."""
    units = [_unit(0, "result", node_id="c0.tA"),      # .80
             _unit(1, "result", node_id="c0.tA"),      # .80
             _unit(2, "result", node_id="c0.tB"),      # .80
             _unit(3, "claim"),                        # .70, unmapped
             _unit(4, "claim")]                        # .70, unmapped
    picked = select_anchors(units, _rel(units), BaseAdapter(), {"max_anchors": 4})
    assert [u.unit_id for u in picked] == ["u0000", "u0002", "u0001", "u0003"]


# ── Q1c/W25-C: per-topic role cap in the legacy selector ─────────────────────────────────
def test_role_cap_in_legacy_selector_is_per_topic():
    # W25-C: the cap is keyed (role, node_id) — a wall of claims inside ONE node holds at
    # most PLAN_ROLE_CAP_PER_TOPIC slots while a claim-dense zone in ANOTHER node keeps
    # its own slots (the old video-global cap 4 starved qP's projectile zone).
    units = ([_unit(i, "claim", node_id="c0.t0") for i in range(5)]
             + [_unit(5 + i, "claim", node_id="c0.t1") for i in range(5)]
             + [_unit(10, "definition", node_id="c0.t0"),
                _unit(11, "definition", node_id="c0.t0")])
    picked = select_anchors(units, _rel(units), BaseAdapter(), {"max_anchors": 12})
    by = Counter((u.role, u.node_id) for u in picked)
    assert by[("claim", "c0.t0")] == config.PLAN_ROLE_CAP_PER_TOPIC == 2
    assert by[("claim", "c0.t1")] == 2                     # the second zone keeps its own
    assert by[("definition", "c0.t0")] == 2
    assert len(picked) == 6


def test_role_cap_unmapped_units_share_one_pseudo_topic():
    # node_id "" everywhere → one shared bucket per role (the audit failure shape: a wall
    # of unmapped claims cannot hold every slot)
    units = _units(["claim"] * 10 + ["definition"] * 3)
    picked = select_anchors(units, _rel(units), BaseAdapter(), {"max_anchors": 12})
    roles = Counter(u.role for u in picked)
    assert roles["claim"] == 2
    assert roles["definition"] == 2
    assert len(picked) == 4


# ── Q1d: relevance bypass ─────────────────────────────────────────────────────────────────
def test_bypass_fires_on_verbatim_containment_each_source():
    assert topic_matches_subject("kinematics", _structure(
        [], 1, rationale="The transcript discusses kinematics: velocity and acceleration."))
    assert topic_matches_subject("kinematics", _structure(
        [], 1, title="Kinematics in One Dimension"))
    assert topic_matches_subject("kinematics", _structure(
        [], 1, root_title="Intro to kinematics"))
    assert topic_matches_subject("KINEMATICS", _structure(     # case-insensitive
        [], 1, title="kinematics lecture"))


def test_bypass_fires_on_token_set_ratio_word_order():
    st = _structure([], 1, title="Kinematics in One Dimension")
    assert topic_matches_subject("one dimension kinematics", st)


def test_bypass_does_not_fire_on_narrow_or_trivial_queries():
    st = _structure([], 1, title="Kinematics in One Dimension",
                    rationale="Covers velocity, acceleration and displacement problems.")
    assert not topic_matches_subject("significant figures", st)   # genuinely narrow
    assert not topic_matches_subject("ki", st)                    # too short to trust
    assert not topic_matches_subject("", st)
    assert not topic_matches_subject("kinematics", _structure([], 1))   # nothing to match


_BYPASS_SETTINGS = {"min_clip_duration_s": 1.0, "max_clip_duration_s": 500.0,
                    "tail_pad_s": 0.0, "min_comprehension_score": 0.7,
                    "quality_floor": 0.0, "max_clips": 12, "max_anchors": 3,
                    "closure_max_span_s": 999.0, "anchor_selector": "priority"}


def test_assemble_bypass_skips_relevance_llm_and_notes_it(monkeypatch):
    _same_model(monkeypatch)
    sents = mini_sents(3)
    units = _units(["intuition", "intuition", "intuition"])    # no contracts → no inlining
    st = _structure(units, 3, title="Kinematics in One Dimension",
                    rationale="A kinematics lecture.")

    def fake(system, user, schema, **kw):
        assert schema is JudgeVerdict, f"bypass must skip the relevance LLM, got {schema}"
        return JudgeVerdict(reasoning="ok", score_10=9, understandable=True)
    monkeypatch.setattr(llm_mod, "llm_json", fake)
    stats: dict = {}
    specs, notes, _rej = assemble_clips(st, "kinematics", sents, "u", "v",
                                        _BYPASS_SETTINGS, BaseAdapter(), stats=stats)
    assert specs
    assert "topic matches video subject" in notes
    assert stats.get("relevance_bypass") is True


def test_assemble_narrow_topic_still_runs_relevance_llm(monkeypatch):
    _same_model(monkeypatch)
    sents = mini_sents(3)
    units = _units(["intuition", "intuition", "intuition"])
    st = _structure(units, 3, title="Kinematics in One Dimension",
                    rationale="A kinematics lecture.")
    rel_calls = []

    def fake(system, user, schema, **kw):
        if schema is RelevanceLLM:
            rel_calls.append(1)
            return RelevanceLLM(items=[{"unit_id": u.unit_id, "score": 1.0} for u in units])
        assert schema is JudgeVerdict, f"unexpected schema {schema}"
        return JudgeVerdict(reasoning="ok", score_10=9, understandable=True)
    monkeypatch.setattr(llm_mod, "llm_json", fake)
    stats: dict = {}
    specs, notes, _rej = assemble_clips(st, "significant figures", sents, "u", "v",
                                        _BYPASS_SETTINGS, BaseAdapter(), stats=stats)
    assert rel_calls, "narrow query must still be scored by the relevance LLM"
    assert "topic matches video subject" not in notes
    assert "relevance_bypass" not in stats


# ── Q1e: refund loop ──────────────────────────────────────────────────────────────────────
_REFUND_SETTINGS = {"min_clip_duration_s": 1.0, "max_clip_duration_s": 500.0,
                    "tail_pad_s": 0.0, "min_comprehension_score": 0.7,
                    "quality_floor": 0.0, "closure_max_span_s": 999.0,
                    "anchor_selector": "priority"}


def _collapse_fixture():
    """Round 0 dedupe-collapses its two anchors into ONE spec (u0001's claim contract
    inlines u0000, producing an identical span) while eligible units u0002/u0003 sit in
    untouched territory — the audited '12 anchors → 3 specs, 63 eligible units left'
    shape in miniature."""
    sents = mini_sents(10)
    units = [_unit(0, "result", (0, 3)),       # round-0 winner
             _unit(1, "claim", (1, 2)),        # collapses into u0000's span
             _unit(2, "definition", (6, 7)),   # residual — ships in the refund round
             _unit(3, "explanation", (8, 9))]  # absorbed into u0002's closure
    return sents, units, _structure(units, 10)


def _passing_judge(monkeypatch):
    monkeypatch.setattr(llm_mod, "llm_json",
                        lambda *a, **kw: JudgeVerdict(reasoning="ok", score_10=9,
                                                      understandable=True))


def test_refund_round_ships_from_residual_units(monkeypatch):
    _same_model(monkeypatch)
    _passing_judge(monkeypatch)
    sents, units, st = _collapse_fixture()
    stats: dict = {}
    settings = dict(_REFUND_SETTINGS, max_anchors=2)          # budget 2 (explicit)
    specs, _notes, rejections = assemble_clips(st, "", sents, "u", "v", settings,
                                               BaseAdapter(), stats=stats)
    assert stats["anchor_budget"] == 2
    assert len(specs) == 2                                    # collapsed 2→1, refunded +1
    ids = {s["cand_id"] for s in specs}
    assert "c_u0002" in ids                                   # shipped FROM the residual
    assert stats["n_refund_rounds"] == 1
    assert stats["n_refund_clips"] == 1
    assert stats["refund_rounds"][0]["n_shipped"] == 1
    assert stats["refund_rounds"][0]["n_residual"] == 2       # u0002 + u0003
    # the round-0 dedupe loser is in the same ledger the refund rounds share
    assert any(r.stage == "dedupe" and r.cand_id == "c_u0001" for r in rejections)


def test_refund_bounded_and_no_infinite_loop_when_residual_never_empties(monkeypatch):
    """u0001 is anchor-eligible but its candidate is ALWAYS killed by a verified+confirmed
    verdict, so the residual never empties. The loop must terminate (zero-yield break /
    the 2-round bound) and refund-round kills must land in the same ledger."""
    _same_model(monkeypatch)
    sents = mini_sents(7)
    units = [_unit(0, "result", (0, 1)),       # ships cleanly
             _unit(1, "claim", (5, 5))]        # poison: killed every round

    def fake(system, user, schema, **kw):
        if schema is JudgeVerdict:
            if "sentence 5" in user:           # the poison span (judge sees the transcript)
                return JudgeVerdict(
                    reasoning="bad", score_10=2, understandable=False,
                    topic_identifiable=False,
                    failure_reasons=[FailureReason(kind="off_topic", detail="drifts",
                                                   evidence_quote="sentence 5")])
            return JudgeVerdict(reasoning="ok", score_10=9, understandable=True)
        if schema is KillConfirmation:         # fresh-context confirmation upholds the kill
            return KillConfirmation(claims=[KillClaimCheck(claim=1, confirmed=True,
                                                           quote="sentence 5")])
        raise AssertionError(f"unexpected schema {schema}")
    monkeypatch.setattr(llm_mod, "llm_json", fake)

    stats: dict = {}
    settings = dict(_REFUND_SETTINGS, max_anchors=3)          # budget 3, never reachable
    specs, _notes, rejections = assemble_clips(st := _structure(units, 7), "", sents,
                                               "u", "v", settings, BaseAdapter(),
                                               stats=stats)
    assert len(specs) == 1 and specs[0]["cand_id"] == "c_u0000"
    assert stats["n_refund_rounds"] <= 2                      # hard bound
    assert stats["n_refund_rounds"] == 1                      # zero-yield round → break
    assert stats["n_refund_clips"] == 0
    assert stats["refund_rounds"][0]["n_shipped"] == 0
    # ledger integrity: the SAME candidate is killed once per round, both in one ledger
    kills = [r for r in rejections if r.stage == "repair" and r.cand_id == "c_u0001"]
    assert len(kills) == 2 and all(r.kill_confirmed for r in kills)


def _superset_fixture():
    """The qP c0.t1 lock-in shape in miniature: round 0 ships (0,3); the refund anchor's
    claim contract inlines u0000/u0001, producing a (0,5) STRICT SUPERSET of the shipped
    sliver — pre-W25-F any 1-sentence overlap rejected it outright, locking partial
    coverage in forever."""
    sents = mini_sents(6)
    units = [_unit(0, "result", (0, 3)),       # round-0 winner (the sliver incumbent)
             _unit(1, "claim", (1, 2)),        # round-0 dedupe loser
             _unit(2, "claim", (5, 5))]        # residual → refund superset candidate
    return sents, units, _structure(units, 6)


def test_refund_superset_replaces_incumbent_and_ledgers_it(monkeypatch):
    """W25-F (spec-governs: replaces the pkg-era refund-overlap drop pin on this fixture):
    a CLEAN refund newcomer strictly containing the incumbent REPLACES it — the incumbent
    is ledgered at stage 'dedupe' (kept+rejected accounting preserved), and the run stats
    carry n_refund_superset_replaced."""
    _same_model(monkeypatch)
    _passing_judge(monkeypatch)
    sents, units, st = _superset_fixture()
    stats: dict = {}
    settings = dict(_REFUND_SETTINGS, max_anchors=2)
    specs, _notes, rejections = assemble_clips(st, "", sents, "u", "v", settings,
                                               BaseAdapter(), stats=stats)
    assert len(specs) == 1                                    # replaced, not duplicated
    assert specs[0]["cand_id"] == "c_u0002"                   # the superset shipped
    assert specs[0]["sentence_start_idx"] == 0 and specs[0]["sentence_end_idx"] == 5
    assert stats["n_refund_rounds"] == 1 and stats["n_refund_clips"] == 1
    assert stats["n_refund_superset_replaced"] == 1
    # ledger invariant: the displaced incumbent is accounted for, never silently dropped
    assert any(r.stage == "dedupe" and r.cand_id == "c_u0000"
               and r.reason == "superseded by refund superset c_u0002" for r in rejections)
    assert not any(r.reason.startswith("refund overlap") for r in rejections)


def test_refund_flagged_superset_never_replaces_and_falls_back_to_reject(monkeypatch):
    """W25-F replace bar: a ship-flagged newcomer (hard core failed, phantom evidence) may
    not displace a clean incumbent; the trim fallback re-judges the trimmed text, which
    also fails → the old 'refund overlap loser' rejection stands and specs are unchanged."""
    _same_model(monkeypatch)
    sents, units, st = _superset_fixture()

    def fake(system, user, schema, **kw):
        assert schema is JudgeVerdict, f"unexpected schema {schema}"   # phantom → no confirm
        if "sentence 5." in user:              # any span holding the refund anchor fails
            return JudgeVerdict(reasoning="bad", score_10=3, understandable=False,
                                topic_identifiable=False,
                                failure_reasons=[FailureReason(kind="off_topic", detail="drifts",
                                                               evidence_quote="ghost words")])
        return JudgeVerdict(reasoning="ok", score_10=9, understandable=True)
    monkeypatch.setattr(llm_mod, "llm_json", fake)

    stats: dict = {}
    settings = dict(_REFUND_SETTINGS, max_anchors=2)
    specs, _notes, rejections = assemble_clips(st, "", sents, "u", "v", settings,
                                               BaseAdapter(), stats=stats)
    assert len(specs) == 1 and specs[0]["cand_id"] == "c_u0000"   # incumbent stands
    assert specs[0]["sentence_end_idx"] == 3
    assert stats["n_refund_clips"] == 0
    assert stats["n_refund_superset_replaced"] == 0
    assert any(r.stage == "dedupe" and r.cand_id == "c_u0002"
               and r.reason == "refund overlap loser to c_u0000" for r in rejections)


def test_refund_partial_overlap_trimmed_rejudged_and_shipped(monkeypatch):
    """W25-F trim path: a refund candidate whose contract pulls dragged its HEAD back over
    shipped territory (partial overlap, NOT a superset) is start-trimmed past the
    incumbent (refine's sentence-true _trim_start_after), its unit_ids refreshed
    truthfully, its never-judged trimmed text re-judged clean — and it SHIPS instead of
    dying as a 'refund overlap loser'."""
    _same_model(monkeypatch)
    _passing_judge(monkeypatch)
    sents = mini_sents(6)
    units = [_unit(0, "result", (0, 3)),       # round-0 winner
             _unit(1, "claim", (1, 2)),        # round-0 dedupe loser
             _unit(2, "misconception", (2, 2)),  # non-anchor; pulled by the correction contract
             _unit(3, "correction", (4, 5))]   # refund anchor — closure hull (1,5) overlaps (0,3)
    st = _structure(units, 6)
    stats: dict = {}
    settings = dict(_REFUND_SETTINGS, max_anchors=2)
    specs, _notes, rejections = assemble_clips(st, "", sents, "u", "v", settings,
                                               BaseAdapter(), stats=stats)
    assert len(specs) == 2                                    # incumbent + the TRIMMED refund
    trimmed = next(s for s in specs if s["cand_id"] == "c_u0003")
    assert trimmed["sentence_start_idx"] == 4 and trimmed["sentence_end_idx"] == 5
    assert trimmed["unit_ids"] == ["u0003"]                   # head units left truthfully
    assert "trimmed_start" in trimmed["warnings"]
    # the fresh clean verdict covers the TRIMMED text — the hash is refreshed (Wave-1 rule)
    from backend.pipeline.assemble.validate import judged_text_hash
    assert trimmed["judged_text_hash"] == judged_text_hash("sentence 4. sentence 5.")
    assert trimmed["ship_flagged"] is False and trimmed["hard_gates_ok"] is True
    assert stats["n_refund_clips"] == 1
    assert stats["n_refund_superset_replaced"] == 0
    assert not any(r.reason.startswith("refund overlap") for r in rejections)


def test_refund_disabled_by_zero_rounds(monkeypatch):
    _same_model(monkeypatch)
    _passing_judge(monkeypatch)
    sents, units, st = _collapse_fixture()
    stats: dict = {}
    settings = dict(_REFUND_SETTINGS, max_anchors=2, refund_rounds=0)
    specs, _notes, _rej = assemble_clips(st, "", sents, "u", "v", settings,
                                         BaseAdapter(), stats=stats)
    assert len(specs) == 1                                    # only the round-0 survivor
    assert stats["n_refund_rounds"] == 0 and stats["n_refund_clips"] == 0


def test_refund_below_floor_superset_does_not_evict_shippable_incumbent(monkeypatch):
    """W25-F REVIEW FIX (whole-change review, confirmed minor): superset REPLACE evicts
    incumbents on the JUDGE hard core (_refund_clean), but a superset's final_quality is not
    gated until the later quality_floor stage. A hard-core-clean superset scoring BELOW the
    floor must NOT evict a shippable sliver incumbent and then die at the floor — that ships
    NOTHING for the span. Here the incumbent (0,3) scores above floor and the (0,5) superset
    below; the incumbent must survive and no eviction may be recorded.

    final_quality is pinned to completeness (isolating the floor interaction from
    grounding/boundary/priority internals); completeness is keyed off the judge's own reasoning
    string so BOTH clips stay is_complete (score 9 → no repair mutation of the spans) while
    only the superset falls below the floor."""
    _same_model(monkeypatch)
    monkeypatch.setattr(scoring, "quality", lambda comp, grnd, bnd, pri: float(comp))
    monkeypatch.setattr(scoring, "completeness_score",
                        lambda v, role, adapter: 0.2 if getattr(v, "reasoning", "") == "thin"
                        else 0.9)

    def fake(system, user, schema, **kw):
        assert schema is JudgeVerdict, f"unexpected schema {schema}"   # no kills → no confirm
        if "sentence 0." in user and "sentence 5." in user:   # the (0,5) superset: clean, thin
            return JudgeVerdict(reasoning="thin", score_10=9, understandable=True)
        return JudgeVerdict(reasoning="ok", score_10=9, understandable=True)
    monkeypatch.setattr(llm_mod, "llm_json", fake)

    sents, units, st = _superset_fixture()
    stats: dict = {}
    settings = dict(_REFUND_SETTINGS, max_anchors=2, quality_floor=0.5)
    specs, _notes, rejections = assemble_clips(st, "", sents, "u", "v", settings,
                                               BaseAdapter(), stats=stats)
    # the above-floor incumbent survived; the below-floor superset did NOT evict it
    assert any(s["cand_id"] == "c_u0000" and s["sentence_end_idx"] == 3 for s in specs)
    assert stats["n_refund_superset_replaced"] == 0
    assert not any(r.reason.startswith("superseded by refund superset") for r in rejections)


def test_explicit_max_clips_truncates_even_below_budget(monkeypatch):
    # the user's explicit dial beats the budget-backed cap — refunded clips included
    _same_model(monkeypatch)
    _passing_judge(monkeypatch)
    sents, units, st = _collapse_fixture()
    settings = dict(_REFUND_SETTINGS, max_anchors=2, max_clips=1)
    specs, _notes, rejections = assemble_clips(st, "", sents, "u", "v", settings,
                                               BaseAdapter())
    assert len(specs) == 1
    assert any(r.stage == "max_clips" for r in rejections)


def test_unset_max_clips_never_undercuts_refunded_specs(monkeypatch):
    # max_clips absent (the DEFAULTS None shape) → cap = max(MAX_SEGMENTS, budget): both
    # the round-0 survivor and the refunded clip ship.
    _same_model(monkeypatch)
    _passing_judge(monkeypatch)
    sents, units, st = _collapse_fixture()
    settings = dict(_REFUND_SETTINGS, max_anchors=2)          # no max_clips key at all
    specs, _notes, _rej = assemble_clips(st, "", sents, "u", "v", settings, BaseAdapter())
    assert len(specs) == 2
