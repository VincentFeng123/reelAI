"""CARD5 — card-rescue robustness (builds on CARD1-4). All offline (llm_json monkeypatched;
zero network/LLM). Both improvements stay PURE ACCEPT-SIDE and GROUNDED — never a new kill,
never a fabricated concept.

(a) ROBUST RESCUE SEEDING: when a card-rescuable kill's missing prerequisite is NOT in the
    clip's referential pool, seed the card from the verdict's missing_concept → introducer
    unit. If that introducer has no groundable text, generate_context_card returns '' and the
    existing verified+confirmed kill stands (unverified_kill stays 0).
(b) SUPPRESSION: skip generating a card when the clip's FIRST sentence already NAMES the
    subject (a conservative grounded rapidfuzz match). A suppressed clip simply has no card —
    and is not penalized missing_context_card.
"""
from __future__ import annotations

import threading

import backend.llm as llm_mod
from backend.pipeline.assemble import (
    _card_warning, _first_sentence_names_subject,
)
from backend.pipeline.assemble.context_card import CardSentence, ContextCardDraft
from backend.pipeline.assemble.graph import Graph
from backend.pipeline.assemble.integrity import Rejection
from backend.pipeline.assemble.types import Candidate
from backend.pipeline.assemble.validate import (
    FailureReason, JudgeVerdict, KillClaimCheck, KillConfirmation,
    _seed_referential_from_introducers, validate_and_repair,
)
from backend.pipeline.sentences import Sentence
from backend.pipeline.understand.models import Unit

from .conftest import FakeAdapter, mini_sents

CARD_TEXT = "the UAM equations were derived earlier"


# ── (a) rescue seeding: drive validate_and_repair to the terminal kill gate ────
def _run_gate(monkeypatch, fake, referential, introducers, u0_summary=CARD_TEXT,
              u0_concepts=("UAM equations",), seen=None):
    from backend import config
    monkeypatch.setattr(config, "JUDGE_MODEL", "")
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "same")
    monkeypatch.setattr(config, "JUDGE_MAX_REPAIR", 0)   # budget=1 → growth can never absorb the prereq
    sents = mini_sents(2)
    u0 = Unit(unit_id="u0000", start=sents[0].start, end=sents[0].end, sentence_range=(0, 0),
              role="explanation", transcript=sents[0].text, summary=u0_summary,
              concepts_introduced=list(u0_concepts))
    u1 = Unit(unit_id="u0001", start=sents[1].start, end=sents[1].end, sentence_range=(1, 1),
              role="explanation", transcript=sents[1].text)
    units = [u0, u1]
    units_by_id = {u.unit_id: u for u in units}
    cand = Candidate(cand_id="c0", anchor_id="u0001", role="explanation", facet="other",
                     title="t", reason="r", unit_ids=["u0001"], referential=list(referential),
                     i_start=1, i_end=1, start=sents[1].start, end=sents[1].end)

    def wrapped(system, user, schema, **kw):
        if seen is not None:
            seen.append(schema)
        return fake(system, user, schema, **kw)
    monkeypatch.setattr(llm_mod, "llm_json", wrapped)
    return validate_and_repair(cand, sents, Graph([], units), units, units_by_id, introducers,
                               FakeAdapter(), {}, lambda s, e: "", "topic", {}, threading.Lock())


def _prereq_fail():
    return JudgeVerdict(reasoning="assumes prereq", score_10=3, understandable=False,
                        source_grounded=False, prerequisites_satisfied=False,
                        failure_reasons=[FailureReason(kind="missing_prerequisite",
                                                       missing_concept="UAM equations",
                                                       evidence_quote="sentence 1.")])


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
            return _prereq_fail()
        return _prereq_fail()                             # native (no card) → fails hard core
    return fake


def test_card5_pure_seed_helper_maps_missing_concept_to_introducer():
    # the prereq's introducer is added ONLY when it isn't already in referential and IS a real
    # unit outside the clip — reuses expand_candidate's prior_introducer preference.
    reasons = [FailureReason(kind="missing_prerequisite", missing_concept="UAM equations")]
    introducers = {"UAM equations": ["u0000"]}
    units = [Unit(unit_id="u0000", start=0, end=1, sentence_range=(0, 0), role="explanation"),
             Unit(unit_id="u0001", start=1, end=2, sentence_range=(1, 1), role="explanation")]
    ubid = {u.unit_id: u for u in units}
    spec = {"unit_ids": ["u0001"], "referential": [], "anchor_id": "u0001"}
    seeded = _seed_referential_from_introducers(spec, reasons, introducers, units, ubid)
    assert seeded == [("u0000", "introduces")]           # prereq NOT in referential → seeded
    # already-present / in-clip / unknown concept → NO seed (no growth, no fabrication)
    assert _seed_referential_from_introducers(
        {"unit_ids": ["u0001"], "referential": [("u0000", "p")], "anchor_id": "u0001"},
        reasons, introducers, units, ubid) == [("u0000", "p")]
    assert _seed_referential_from_introducers(spec, reasons, {"other": ["u0000"]}, units, ubid) == []


def test_card5_seeds_from_introducer_when_not_in_referential(monkeypatch):
    # THE robustness fix: prereq NOT in referential, but introducers maps its concept → the
    # groundable u0000 → seeding adds u0000 → card generated → rescue ships the SAME span.
    seen: list = []
    kept, rej = _run_gate(monkeypatch, _uam_fake(carded_pass=True), referential=[],
                          introducers={"UAM equations": ["u0000"]}, seen=seen)
    assert rej is None and kept is not None              # rescued via seeding, NOT rejected
    assert kept.i_start == 1 and kept.i_end == 1         # SAME span (no growth/trim)
    assert kept.unit_ids == ["u0001"]                    # unit_ids unchanged — span not grown
    assert kept.referential == []                        # seeding is card-only; span untouched
    assert kept.context_card == CARD_TEXT
    assert "card_completed" in kept.warnings and kept.ship_flagged is False
    assert ContextCardDraft in seen and KillConfirmation in seen


def test_card5_no_introducer_no_rescue_kill_stands(monkeypatch):
    # NO groundable source (the concept isn't in introducers) → seeding adds nothing → card ''
    # → NO rescue → the verified+confirmed kill stands. unverified_kill stays 0.
    seen: list = []
    kept, rej = _run_gate(monkeypatch, _uam_fake(carded_pass=True), referential=[],
                          introducers={"unrelated concept": ["u0000"]}, seen=seen)
    assert kept is None and isinstance(rej, Rejection)
    assert rej.stage == "repair" and rej.kill_confirmed is True
    assert rej.failure_kinds == ["missing_prerequisite"]
    assert ContextCardDraft not in seen                  # no groundable seed → no draft LLM call


def test_card5_ungroundable_introducer_no_rescue_kill_stands(monkeypatch):
    # the introducer IS seeded, but has NO groundable text (empty summary + concepts) →
    # generate_context_card returns '' (grounding blocks fabrication) → NO rescue → kill stands.
    seen: list = []
    kept, rej = _run_gate(monkeypatch, _uam_fake(carded_pass=True), referential=[],
                          introducers={"UAM equations": ["u0000"]},
                          u0_summary="", u0_concepts=(), seen=seen)
    assert kept is None and isinstance(rej, Rejection)
    assert rej.kill_confirmed is True and rej.failure_kinds == ["missing_prerequisite"]
    assert ContextCardDraft in seen                      # rescue attempted, but grounding → ''


# ── (b) suppression: first sentence already names the subject ──────────────────
def _sent(text: str) -> Sentence:
    return Sentence(idx=0, text=text, start=0.0, end=5.0, terminator=".",
                    ends_with_period=True, word_start_idx=0, word_end_idx=0, align_confidence=1.0)


def _anchor(concept: str) -> dict:
    u = Unit(unit_id="u0000", start=0, end=5, sentence_range=(0, 0), role="explanation",
             concepts_introduced=[concept] if concept else [])
    return {"u0000": u}


def test_card5_suppress_when_first_sentence_names_anchor_concept():
    sents = [_sent("The UAM equations describe motion under constant acceleration.")]
    s = {"sentence_start_idx": 0, "anchor_id": "u0000"}
    assert _first_sentence_names_subject(s, sents, _anchor("UAM equations"), topic="") is True


def test_card5_suppress_when_first_sentence_names_topic():
    sents = [_sent("Momentum is conserved in every collision.")]
    s = {"sentence_start_idx": 0, "anchor_id": "u0000"}
    assert _first_sentence_names_subject(s, sents, _anchor(""), topic="momentum") is True


def test_card5_no_suppress_when_subject_absent():
    # conservative: an unrelated first sentence must NOT suppress (avoid removing a needed card).
    sents = [_sent("Let us begin with a short story about a traveller.")]
    s = {"sentence_start_idx": 0, "anchor_id": "u0000"}
    assert _first_sentence_names_subject(s, sents, _anchor("UAM equations"), topic="kinematics") is False


def test_card5_suppressed_card_is_not_penalized_missing_card():
    # a deliberately SUPPRESSED card (subject named) does not trip the -0.10 missing_context_card
    # dock, even with referential present — mirrors 'no card, no penalty' when referential is empty.
    s = {"referential": [("u0000", "prerequisite")], "card_suppressed": True, "warnings": ()}
    _card_warning(s)
    assert "missing_context_card" not in (s.get("warnings") or ())
    # without the suppression marker the same spec IS flagged (guard is specific)
    s2 = {"referential": [("u0000", "prerequisite")], "warnings": ()}
    _card_warning(s2)
    assert "missing_context_card" in s2["warnings"]
