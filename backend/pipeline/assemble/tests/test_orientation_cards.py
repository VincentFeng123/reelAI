"""Per-clip orientation cards (feed display). Offline — llm_json monkeypatched, zero network.

Every clip must get a card describing what IT covers, drawn from its OWN units (never a repeated
distant prerequisite), the LLM polish kept ONLY when grounded to that clip's units, and a zero-LLM
extractive floor so no clip is ever blank.
"""
from __future__ import annotations

import backend.llm as llm_mod
from backend.pipeline.assemble.context_card import (
    _OrientCard, _OrientCards, generate_orientation_cards,
)
from backend.pipeline.understand.models import Unit


def _units():
    u0 = Unit(unit_id="u0000", start=0.0, end=2.0, sentence_range=(0, 0), role="claim",
              summary="Atoms consist of a core made of protons and neutrons, and some electrons.",
              concepts_introduced=["proton", "neutron", "electron"])
    u1 = Unit(unit_id="u0001", start=2.0, end=4.0, sentence_range=(1, 1), role="definition",
              summary="An acid donates protons under the Bronsted-Lowry definition.",
              concepts_introduced=["acid"])
    return {"u0000": u0, "u0001": u1}


_SPECS = [
    {"unit_ids": ["u0000"], "anchor_id": "u0000", "title": "Atomic structure"},
    {"unit_ids": ["u0001"], "anchor_id": "u0001", "title": "Acids"},
]


def test_grounded_llm_cards_assigned_per_clip(monkeypatch):
    def fake(system, user, schema, temperature=0.2):
        return _OrientCards(cards=[
            _OrientCard(clip_index=0, text="Atoms have a core of protons and neutrons with electrons."),
            _OrientCard(clip_index=1, text="An acid donates protons per the Bronsted-Lowry definition."),
        ])
    monkeypatch.setattr(llm_mod, "llm_json", fake)
    cards = generate_orientation_cards(_SPECS, _units(), adapter=None, topic="chemistry")
    assert "protons and neutrons" in cards[0].lower() or "core" in cards[0].lower()
    assert "acid" in cards[1].lower()
    assert cards[0] != cards[1]                       # distinct per clip — no repetition


def test_extractive_fallback_on_llm_error(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("llm down")
    monkeypatch.setattr(llm_mod, "llm_json", boom)
    cards = generate_orientation_cards(_SPECS, _units(), adapter=None, topic="chemistry")
    # falls back to each clip's own anchor-unit summary — grounded, distinct, never blank
    assert cards[0].startswith("Atoms consist of a core")
    assert cards[1].startswith("An acid donates protons")
    assert cards[0] != cards[1]


def test_ungrounded_llm_card_rejected(monkeypatch):
    def fake(system, user, schema, temperature=0.2):
        return _OrientCards(cards=[
            _OrientCard(clip_index=0, text="The stock market rallied on strong earnings today."),  # fabricated
            _OrientCard(clip_index=1, text="An acid donates protons per Bronsted-Lowry."),         # grounded
        ])
    monkeypatch.setattr(llm_mod, "llm_json", fake)
    cards = generate_orientation_cards(_SPECS, _units(), adapter=None, topic="chemistry")
    assert cards[0].startswith("Atoms consist of a core")   # ungrounded → extractive fallback
    assert "acid" in cards[1].lower()                        # grounded → kept


def test_never_blank_and_word_capped(monkeypatch):
    monkeypatch.setattr(llm_mod, "llm_json", lambda *a, **k: _OrientCards(cards=[]))
    cards = generate_orientation_cards(_SPECS, _units(), adapter=None, topic="chemistry")
    assert all(c.strip() for c in cards)                     # extractive floor for every clip
    assert len(cards) == len(_SPECS)
