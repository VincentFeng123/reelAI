"""P2-repair-rework: anchor-native-first judging, verdict-routed trim moves, and
bisection over the trim lattice.

Audit findings under test (docs/audits/2026-07-02, F2): (1) _fill_contract inflated
candidates BEFORE the first judge call, so 10-60s anchors were never scored at native
size; (2) the repair loop's only move was growth — off_topic/coherence verdicts produced
zero targets and burned the budget, and candidates with one bad edge died instead of
losing a sentence. All offline (llm_json monkeypatched); scripted verdict sequences keyed
on the exact clip transcript each judge call sees."""
from __future__ import annotations

import threading

import backend.llm as llm_mod
from backend.adapters.base import CompletenessContract, ContractElement
from backend.pipeline.assemble.context_card import CardSentence, ContextCardDraft
from backend.pipeline.assemble.graph import Graph
from backend.pipeline.assemble.integrity import Rejection
from backend.pipeline.assemble.types import Candidate
from backend.pipeline.assemble.validate import (
    JUDGE_SYSTEM, FailureReason, JudgeVerdict, KillClaimCheck, KillConfirmation,
    _card_rescuable, _trim_flavored, judge_clip, judged_text_hash, validate_and_repair,
)
from backend.pipeline.understand.models import Unit

from .conftest import FakeAdapter, mini_sents, mini_units


# ── fixtures ──────────────────────────────────────────────────────────────────
class ResultContractAdapter(FakeAdapter):
    """Contract adapter: 'result' requires an example_setup before + a result within."""

    def completeness_contracts(self):
        return {"result": CompletenessContract("result", (
            ContractElement("problem", ("example_setup",), "required", "before"),
            ContractElement("answer", ("result",), "required", "within"),
        ))}

    def contract_for(self, role):
        return self.completeness_contracts().get(role)


def role_units(roles, sents):
    return [Unit(unit_id=f"u{i:04d}", start=sents[i].start, end=sents[i].end,
                 sentence_range=(i, i), role=r, transcript=sents[i].text)
            for i, r in enumerate(roles)]


def _mk_cand(units, sents, anchor_idx=0, unit_ids=None, referential=None):
    ids = unit_ids if unit_ids is not None else [u.unit_id for u in units]
    picked = [u for u in units if u.unit_id in set(ids)]
    i0 = min(u.sentence_range[0] for u in picked)
    i1 = max(u.sentence_range[1] for u in picked)
    return Candidate(cand_id="c0", anchor_id=units[anchor_idx].unit_id,
                     role=units[anchor_idx].role, facet="other", title="t", reason="r",
                     unit_ids=ids, referential=list(referential or []),
                     i_start=i0, i_end=i1, start=sents[i0].start, end=sents[i1].end)


def _vr(cand, sents, units, adapter=None, cache=None, settings=None):
    units_by_id = {u.unit_id: u for u in units}
    return validate_and_repair(cand, sents, Graph([], units), units, units_by_id, {},
                               adapter or FakeAdapter(), settings or {}, lambda s, e: "",
                               "topic", cache if cache is not None else {}, threading.Lock())


def _transcript(user: str) -> str:
    return user.split("CLIP TRANSCRIPT:\n", 1)[1].rsplit("\n\nJudge whether", 1)[0].strip()


def _script(monkeypatch, decide):
    """Fake llm_json: judge-only (asserts confirm_kill is never reached), scripted by the
    exact transcript text; records call order."""
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


def _off_topic():
    # phantom evidence quote → the terminal gate can never confirm-kill in these tests
    return JudgeVerdict(reasoning="jumps between topics", score_10=3, understandable=False,
                        topic_identifiable=False,
                        failure_reasons=[FailureReason(kind="off_topic", detail="tail drifts",
                                                       evidence_quote="the previous equation")])


def _needs_problem():
    return JudgeVerdict(reasoning="no problem stated", score_10=3, understandable=False,
                        problem_statement_complete=False,
                        failure_reasons=[FailureReason(
                            kind="missing_problem_statement", detail="question never stated",
                            evidence_quote="the previous equation")])


def _needs_prereq():
    return JudgeVerdict(reasoning="assumes prior setup", score_10=3, understandable=False,
                        prerequisites_satisfied=False)


# ── P2b routing vocabulary (pure) ─────────────────────────────────────────────
def test_trim_flavored_routing_vocabulary():
    def v(kind):
        return JudgeVerdict(failure_reasons=[FailureReason(kind=kind, evidence_quote="x")])
    for kind in ("unresolved_reference", "missing_prerequisite", "missing_visual",
                 "missing_problem_statement", "missing_reasoning", "missing_result",
                 "not_source_grounded"):
        assert _trim_flavored(v(kind)) is False          # missing content → grow
    for kind in ("off_topic", "other"):
        assert _trim_flavored(v(kind)) is True           # excess content → trim
    # reason-less coherence failure (topic/purpose core) is trim-flavored too
    assert _trim_flavored(JudgeVerdict(topic_identifiable=False)) is True
    assert _trim_flavored(JudgeVerdict()) is False


# ── P2a: native size judged FIRST, fill only after a failing verdict ──────────
def test_native_size_judged_before_any_fill(monkeypatch):
    # result anchor alone; the bound result contract is missing its required
    # example_setup — pre-P2, _fill_contract inflated the span BEFORE the first judge
    # call. Now the FIRST judged text must be the anchor's native text.
    sents = mini_sents(4)
    units = role_units(["example_setup", "worked_step", "result", "summary"], sents)

    def decide(text):
        return _pass() if text != "sentence 2." else _needs_problem()
    calls = _script(monkeypatch, decide)
    kept, rej = _vr(_mk_cand(units, sents, anchor_idx=2, unit_ids=["u0002"]),
                    sents, units, adapter=ResultContractAdapter())
    assert rej is None and kept is not None
    assert calls["texts"][0] == "sentence 2."            # native size scored first
    assert calls["n"] == 2                               # native fail → fill/grow → re-judge
    assert "sentence 0." in calls["texts"][1]            # fill pulled the example_setup
    assert kept.attempts == 2
    assert kept.judged_text_hash == judged_text_hash(calls["texts"][1])


def test_native_complete_accepts_without_inflation(monkeypatch):
    # the native verdict passes → accept at native size, no contract fill, one judge call
    # (pre-P2 the span was inflated before the judge ever saw the 1-unit anchor).
    sents = mini_sents(4)
    units = role_units(["example_setup", "worked_step", "result", "summary"], sents)
    calls = _script(monkeypatch, lambda text: _pass())
    kept, rej = _vr(_mk_cand(units, sents, anchor_idx=2, unit_ids=["u0002"]),
                    sents, units, adapter=ResultContractAdapter())
    assert rej is None and kept is not None
    assert calls["n"] == 1
    assert kept.unit_ids == ["u0002"]                    # never inflated
    assert kept.judged_text_hash == judged_text_hash("sentence 2.")


# ── P2b: off_topic routes to trim, not budget burn ────────────────────────────
def test_off_topic_triggers_trim_not_budget_burn(monkeypatch):
    # pre-P2: off_topic produced zero expansion targets → break → gate. Now the farthest
    # units are trimmed (bisection) and the anchor-only sub-span is accepted.
    sents = mini_sents(3)
    units = mini_units(sents)

    def decide(text):
        return _pass() if text == "sentence 0." else _off_topic()
    calls = _script(monkeypatch, decide)
    kept, rej = _vr(_mk_cand(units, sents), sents, units)
    assert rej is None and kept is not None
    assert kept.unit_ids == ["u0000"]                    # farthest units' sentences left the span
    assert kept.i_start == 0 and kept.i_end == 0
    assert kept.verdict.understandable is True           # accepted, not ship-flagged
    assert kept.ship_flagged is False
    assert kept.judged_text_hash == judged_text_hash("sentence 0.")   # hash refreshed (P2d)
    assert calls["texts"][0] == "sentence 0. sentence 1. sentence 2."
    assert calls["n"] == 3                               # native + 2 bisection probes ≤ budget


def test_trim_never_removes_anchor_or_contract_required_units(monkeypatch):
    # anchor (result) + contract-required example_setup are protected: every probed
    # sub-span must still contain both, however hard the judge complains.
    sents = mini_sents(4)
    units = role_units(["example_setup", "result", "explanation", "explanation"], sents)
    calls = _script(monkeypatch, lambda text: _off_topic())    # everything fails
    kept, rej = _vr(_mk_cand(units, sents, anchor_idx=1), sents, units,
                    adapter=ResultContractAdapter())
    assert rej is None and kept is not None              # phantom evidence → never killed
    for text in calls["texts"]:
        assert "sentence 0." in text and "sentence 1." in text   # protected units always kept
    # the minimal probed lattice point is exactly the protected pair
    assert "sentence 0. sentence 1." in calls["texts"]
    assert kept.ship_flagged is True                     # nothing passed → ships flagged
    assert "unverified_judge_concerns" in kept.warnings
    # confirm_kill was truly never reached: it swallows exceptions, so the fake's schema
    # assert alone can't prove it — an outage marker here would betray a confirm attempt.
    assert "kill_confirm_unavailable" not in kept.warnings


# ── P2c: bisection converges within budget to the largest passing sub-span ────
def test_bisection_returns_largest_passing_subspan_within_budget(monkeypatch):
    # 7-unit span fails coherence; scripted oracle: any span touching sentences 5/6 fails.
    # Bisection (lo=0, hi=7): probes k=3 (pass) → k=5 (pass) → k=6 (fail) and returns the
    # largest passing prefix (units 0..4) in exactly JUDGE_MAX_REPAIR+1 = 4 new judgments.
    sents = mini_sents(7)
    units = mini_units(sents)

    def decide(text):
        return _off_topic() if ("sentence 5." in text or "sentence 6." in text) else _pass()
    calls = _script(monkeypatch, decide)
    kept, rej = _vr(_mk_cand(units, sents), sents, units)
    assert rej is None and kept is not None
    assert kept.unit_ids == [f"u{i:04d}" for i in range(5)]      # largest passing sub-span
    assert kept.i_start == 0 and kept.i_end == 4
    assert kept.verdict.understandable is True
    assert calls["n"] == 4                               # native + 3 probes == budget, not linear
    assert calls["texts"][1:] == [
        " ".join(f"sentence {i}." for i in range(3)),    # k=3
        " ".join(f"sentence {i}." for i in range(5)),    # k=5
        " ".join(f"sentence {i}." for i in range(6)),    # k=6
    ]


# ── P2d: truthful unit_ids + referential restoration across grow-then-trim ────
def test_trim_restores_referential_and_truthful_unit_ids(monkeypatch):
    # native anchor fails on prerequisites → grows to absorb the far prereq (referential
    # empties); the grown span fails coherence → bisection lands between the anchor-native
    # last-known-good and the full span. The trimmed-out prereq's referential entry
    # RETURNS, and unit_ids reflect exactly the units whose sentences remain in the span.
    sents = mini_sents(4)
    units = mini_units(sents)

    def decide(text):
        if text == "sentence 3.":
            return _needs_prereq()                       # hard core intact → last-known-good
        if text == "sentence 2. sentence 3.":
            return _pass()
        return _off_topic()                              # grown span + larger probes fail
    calls = _script(monkeypatch, decide)
    cand = _mk_cand(units, sents, anchor_idx=3, unit_ids=["u0003"],
                    referential=[("u0000", "prerequisite")])
    kept, rej = _vr(cand, sents, units)
    assert rej is None and kept is not None
    # grow absorbed u0000..u0002 first (referential emptied), trim then dropped u0000/u0001
    assert calls["texts"][0] == "sentence 3."
    assert calls["texts"][1] == "sentence 0. sentence 1. sentence 2. sentence 3."
    assert kept.unit_ids == ["u0002", "u0003"]           # units that left the span left unit_ids
    assert kept.i_start == 2 and kept.i_end == 3
    assert kept.start == sents[2].start and kept.end == sents[3].end
    assert kept.referential == [("u0000", "prerequisite")]       # returned to referential
    assert kept.judged_text_hash == judged_text_hash("sentence 2. sentence 3.")
    assert calls["n"] == 4                               # native + grown + 2 probes ≤ budget


# ── P2c: cache hits are free — bisection completes on an exhausted budget ─────
def test_cache_hits_not_counted_against_budget(monkeypatch):
    from backend import config
    monkeypatch.setattr(config, "JUDGE_MAX_REPAIR", 0)   # budget = exactly 1 NEW judgment
    sents = mini_sents(7)
    units = mini_units(sents)
    calls = _script(monkeypatch, lambda text: _off_topic())     # only the native call is live
    cache: dict = {}

    def seed(n_units, verdict):                          # pre-judged lattice points
        ids = frozenset(f"u{i:04d}" for i in range(n_units))
        text = " ".join(f"sentence {i}." for i in range(n_units))
        cache[(ids, judged_text_hash(text))] = verdict
    seed(3, _pass())
    seed(5, _pass())
    seed(6, _off_topic())
    kept, rej = _vr(_mk_cand(units, sents), sents, units, cache=cache)
    assert rej is None and kept is not None
    assert calls["n"] == 1                               # the native judgment spent the budget…
    assert kept.unit_ids == [f"u{i:04d}" for i in range(5)]      # …probes rode the cache free
    assert kept.verdict.understandable is True
    assert kept.judged_text_hash == judged_text_hash(
        " ".join(f"sentence {i}." for i in range(5)))


def test_trim_impossible_falls_back_to_grow(monkeypatch):
    # a MIXED verdict (off_topic + missing prerequisite) on a single-unit span: nothing is
    # removable, so trim precedence must not starve the grow move — the prereq is pulled
    # and the grown span re-judged (pre-P2 targeting preserved).
    sents = mini_sents(4)
    units = mini_units(sents)

    def decide(text):
        if text == "sentence 3.":
            v = _off_topic()
            v.prerequisites_satisfied = False            # mixed complaint
            return v
        return _pass()
    calls = _script(monkeypatch, decide)
    cand = _mk_cand(units, sents, anchor_idx=3, unit_ids=["u0003"],
                    referential=[("u0000", "prerequisite")])
    kept, rej = _vr(cand, sents, units)
    assert rej is None and kept is not None
    assert calls["n"] == 2                               # native fail → grow → accepted
    assert set(kept.unit_ids) == {u.unit_id for u in units}
    assert kept.verdict.understandable is True


def test_uncached_probe_respects_exhausted_budget(monkeypatch):
    # budget of 1 is spent on the native judgment; with an EMPTY cache no probe may run —
    # the candidate falls through to the terminal gate instead of overspending.
    from backend import config
    monkeypatch.setattr(config, "JUDGE_MAX_REPAIR", 0)
    sents = mini_sents(3)
    units = mini_units(sents)
    calls = _script(monkeypatch, lambda text: _off_topic())
    kept, rej = _vr(_mk_cand(units, sents), sents, units)
    assert calls["n"] == 1                               # no probe judged beyond the budget
    assert rej is None and kept is not None              # phantom evidence → ships flagged
    assert kept.ship_flagged is True
    # confirm_kill never reached (see test_trim_never_removes… — the fake's schema assert
    # is swallowed by confirm_kill's except; the absent outage marker is the real proof).
    assert "kill_confirm_unavailable" not in kept.warnings
    assert set(kept.unit_ids) == {u.unit_id for u in units}


# ── CARD1-CARD4: card-as-repair at the repair terminal kill gate (THE UAM fix) ─────────
# A clip that dies verified+confirmed on a prereq/reference-family reason is RESCUED (not
# killed) when a grounded context card, shown before the SAME span, flips
# prerequisites_satisfied / all_references_resolved on a re-judge. Purely accept-side: a
# rescue converts a would-be Rejection into a ship; it NEVER creates a Rejection. All the
# LLM seams (judge, confirm_kill, context-card draft, carded re-judge) are monkeypatched.
CARD_TEXT = "the UAM equations were derived earlier"


def _uam_units(sents):
    """Anchor u0001 (in the clip) + a FAR prereq u0000 (referential-only) whose summary
    grounds the rescue card."""
    u0 = Unit(unit_id="u0000", start=sents[0].start, end=sents[0].end, sentence_range=(0, 0),
              role="explanation", transcript=sents[0].text, summary=CARD_TEXT,
              concepts_introduced=["UAM equations"])
    u1 = Unit(unit_id="u0001", start=sents[1].start, end=sents[1].end, sentence_range=(1, 1),
              role="explanation", transcript=sents[1].text)
    return [u0, u1]


def _uam_cand(sents, referential):
    return Candidate(cand_id="c0", anchor_id="u0001", role="explanation", facet="other",
                     title="t", reason="r", unit_ids=["u0001"], referential=list(referential),
                     i_start=1, i_end=1, start=sents[1].start, end=sents[1].end)


def _run_card_gate(monkeypatch, fake, referential, seen=None):
    """Drive validate_and_repair to the terminal gate with budget=1 (JUDGE_MAX_REPAIR=0 →
    the native judge exhausts the budget, so growth can NEVER absorb the prereq — exactly
    the 'growth can't fix it' shape card-as-repair targets)."""
    from backend import config
    monkeypatch.setattr(config, "JUDGE_MODEL", "")
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "same")
    monkeypatch.setattr(config, "JUDGE_MAX_REPAIR", 0)
    sents = mini_sents(2)
    units = _uam_units(sents)
    units_by_id = {u.unit_id: u for u in units}

    def wrapped(system, user, schema, **kw):
        if seen is not None:
            seen.append(schema)
        return fake(system, user, schema, **kw)
    monkeypatch.setattr(llm_mod, "llm_json", wrapped)
    kept, rej = validate_and_repair(_uam_cand(sents, referential), sents, Graph([], units),
                                    units, units_by_id, {}, FakeAdapter(), {},
                                    lambda s, e: "", "topic", {}, threading.Lock())
    return kept, rej


def _prereq_fail(kind="missing_prerequisite", quote="sentence 1."):
    return JudgeVerdict(reasoning="assumes prereq", score_10=3, understandable=False,
                        source_grounded=False, prerequisites_satisfied=False,
                        failure_reasons=[FailureReason(kind=kind, missing_concept="UAM equations",
                                                       evidence_quote=quote)])


def _uam_fake(carded_pass=True):
    def fake(system, user, schema, **kw):
        if schema is KillConfirmation:
            return KillConfirmation(claims=[KillClaimCheck(claim=1, confirmed=True,
                                                           quote="sentence 1.")])
        if schema is ContextCardDraft:
            return ContextCardDraft(sentences=[CardSentence(text=CARD_TEXT,
                                                            source_unit_id="u0000")])
        if "CONTEXT CARD" in user:                       # the carded re-judge
            if carded_pass:
                return JudgeVerdict(reasoning="card resolves the prereq", score_10=9,
                                    understandable=True, prerequisites_satisfied=True,
                                    all_references_resolved=True, source_grounded=True)
            return _prereq_fail()                         # card didn't fix the hard core
        return _prereq_fail()                             # native (no card) → fails hard core
    return fake


def test_card1_prompt_credits_the_context_card(monkeypatch):
    assert "CONTEXT CARD" in JUDGE_SYSTEM
    assert "prerequisites_satisfied" in JUDGE_SYSTEM and "all_references_resolved" in JUDGE_SYSTEM
    # judge_clip injects a provided card into the USER prompt so the credited prompt can act
    seen = {}

    def cap(system, user, schema, **kw):
        seen["user"] = user
        return JudgeVerdict(score_10=9, understandable=True)
    monkeypatch.setattr(llm_mod, "llm_json", cap)
    judge_clip("clip words", "explanation", FakeAdapter(), context_card=CARD_TEXT)
    assert f"CONTEXT CARD (shown before the clip): {CARD_TEXT}" in seen["user"]


def test_card2_uam_rescue_ships_same_span(monkeypatch):
    # the equations clip: hard core broken ONLY by a prereq whose concept is in referential →
    # the carded re-judge flips prerequisites/references → ships the SAME span carded.
    seen: list = []
    kept, rej = _run_card_gate(monkeypatch, _uam_fake(carded_pass=True),
                               referential=[("u0000", "prerequisite")], seen=seen)
    assert rej is None and kept is not None              # rescued, NOT rejected
    assert kept.i_start == 1 and kept.i_end == 1         # SAME span (no growth/trim)
    assert kept.unit_ids == ["u0001"]                    # unit_ids unchanged
    assert kept.context_card == CARD_TEXT
    assert "card_completed" in kept.warnings
    assert kept.ship_flagged is False
    assert kept.verdict.prerequisites_satisfied is True  # stats/scoring reflect the carded pass
    assert kept.verdict.all_references_resolved is True
    assert ContextCardDraft in seen                      # a card WAS generated
    assert KillConfirmation in seen                      # the kill was verified+confirmed first


def test_card2_empty_card_does_not_rescue_kill_stands(monkeypatch):
    # NEGATIVE 1: no referential → generate_context_card returns '' → NO rescue → the existing
    # verified+confirmed kill stands unchanged.
    seen: list = []
    kept, rej = _run_card_gate(monkeypatch, _uam_fake(carded_pass=True),
                               referential=[], seen=seen)
    assert kept is None and isinstance(rej, Rejection)
    assert rej.stage == "repair" and rej.kill_confirmed is True
    assert rej.failure_kinds == ["missing_prerequisite"]
    assert ContextCardDraft not in seen                  # '' returns before any draft LLM call


def test_card2_non_family_reason_does_not_rescue(monkeypatch):
    # NEGATIVE 2 (the guard mutation check): a MIXED confirmed set with one non-family reason
    # (missing_result) is NOT rescuable — deleting the 'every reason must be family' guard
    # (all→any) would wrongly rescue here.
    def fake(system, user, schema, **kw):
        if schema is KillConfirmation:
            return KillConfirmation(claims=[
                KillClaimCheck(claim=1, confirmed=True, quote="sentence 1."),
                KillClaimCheck(claim=2, confirmed=True, quote="sentence 1.")])
        if schema is ContextCardDraft:
            return ContextCardDraft(sentences=[CardSentence(text=CARD_TEXT,
                                                            source_unit_id="u0000")])
        if "CONTEXT CARD" in user:                       # a carded re-judge would PASS — so a
            return JudgeVerdict(reasoning="card would fix it", score_10=9, understandable=True,
                                prerequisites_satisfied=True, all_references_resolved=True,
                                source_grounded=True)    # weakened all→any guard would SHIP here
        return JudgeVerdict(
            reasoning="prereq + missing result", score_10=3, understandable=False,
            source_grounded=False, prerequisites_satisfied=False, result_complete=False,
            failure_reasons=[FailureReason(kind="missing_prerequisite", evidence_quote="sentence 1."),
                             FailureReason(kind="missing_result", evidence_quote="sentence 1.")])
    seen: list = []
    kept, rej = _run_card_gate(monkeypatch, fake, referential=[("u0000", "prerequisite")], seen=seen)
    assert kept is None and isinstance(rej, Rejection) and rej.kill_confirmed is True
    assert rej.failure_kinds == ["missing_prerequisite", "missing_result"]
    assert ContextCardDraft not in seen                  # rescue not even attempted (guard held)
    assert _card_rescuable([FailureReason(kind="missing_prerequisite"),
                            FailureReason(kind="missing_result")]) is False


def test_card2_carded_rejudge_still_fails_does_not_rescue(monkeypatch):
    # NEGATIVE 3: a groundable card is generated, but the carded re-judge still fails the hard
    # core → NO rescue → the kill stands.
    seen: list = []
    kept, rej = _run_card_gate(monkeypatch, _uam_fake(carded_pass=False),
                               referential=[("u0000", "prerequisite")], seen=seen)
    assert kept is None and isinstance(rej, Rejection) and rej.kill_confirmed is True
    assert rej.failure_kinds == ["missing_prerequisite"]
    assert ContextCardDraft in seen                      # a card WAS generated (rescue attempted)


def test_card2_unverified_reason_never_kills_never_rescues(monkeypatch):
    # unverified_kill=0 at the repair terminal: a prereq-family reason whose quote is PHANTOM
    # is never verified → confirm_kill and card-rescue are both skipped → ships flagged (no
    # Rejection, no card).
    def fake(system, user, schema, **kw):
        assert schema is JudgeVerdict, "phantom evidence must not reach confirm/card seams"
        return _prereq_fail(quote="ghost words")          # not contained in 'sentence 1.'
    seen: list = []
    kept, rej = _run_card_gate(monkeypatch, fake, referential=[("u0000", "prerequisite")], seen=seen)
    assert rej is None and kept is not None               # never killed on unverifiable evidence
    assert kept.ship_flagged is True
    assert "unverified_judge_concerns" in kept.warnings
    assert "card_completed" not in kept.warnings
    assert "kill_confirm_unavailable" not in kept.warnings
    assert KillConfirmation not in seen and ContextCardDraft not in seen


def test_card_completed_not_double_penalized_missing_card():
    # SCORING: a card_completed clip HAS a card, so _card_warning must NOT also stamp
    # 'missing_context_card' (which would dock boundary_score -0.10).
    from backend.pipeline.assemble import _card_warning
    from backend.pipeline.assemble.scoring import boundary_score
    s = {"context_card": CARD_TEXT, "referential": [("u0000", "prerequisite")],
         "truncated": True, "warnings": ("card_completed",)}
    _card_warning(s)
    assert "missing_context_card" not in s["warnings"]    # HAS a card → no spurious dock
    assert "card_completed" in s["warnings"]
    assert boundary_score(("card_completed",)) == 1.0     # card_completed carries no penalty
