"""opening_in_context is a required verdict field: a clip that opens mid-thought fails
is_complete even if everything else passes; a card cannot satisfy it."""
from __future__ import annotations

from backend.pipeline.assemble.validate import JudgeVerdict, is_complete
from backend.adapters import get_adapter


def test_opening_in_context_is_required_for_completeness():
    adapter = get_adapter("generic")
    v = JudgeVerdict(score_10=9, understandable=True, opening_in_context=False)
    assert not is_complete(v, "explanation", adapter, min_score=0.7)
    v_ok = JudgeVerdict(score_10=9, understandable=True, opening_in_context=True)
    assert is_complete(v_ok, "explanation", adapter, min_score=0.7)


def test_opening_in_context_in_core_fields():
    from backend.adapters.base import CORE_VERDICT_FIELDS
    assert "opening_in_context" in CORE_VERDICT_FIELDS
