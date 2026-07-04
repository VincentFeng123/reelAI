"""Offline fixtures for assemble tests: minimal sentences, units, adapter. No LLM/network."""
from __future__ import annotations

from backend.pipeline.sentences import Sentence
from backend.pipeline.understand.models import Unit


def mini_sents(n: int, sec: float = 10.0) -> list[Sentence]:
    return [Sentence(idx=i, text=f"sentence {i}.", start=i * sec, end=(i + 1) * sec - 0.1,
                     terminator=".", ends_with_period=True, word_start_idx=i, word_end_idx=i,
                     align_confidence=1.0) for i in range(n)]


def mini_units(sents: list[Sentence]) -> list[Unit]:
    return [Unit(unit_id=f"u{i:04d}", start=s.start, end=s.end, sentence_range=(i, i),
                 role="explanation", transcript=s.text) for i, s in enumerate(sents)]


class FakeAdapter:
    """Contract-free adapter: every verdict field optional, no completeness contract."""

    def required_verdict_fields(self, role):
        return []

    def required_elements(self, role):
        return []

    def contract_for(self, role):
        return None
