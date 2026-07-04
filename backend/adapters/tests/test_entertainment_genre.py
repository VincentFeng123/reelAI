"""GEN1+GEN2 offline tests: the 'parody ships 0 clips' fix.

Every test monkeypatches ``backend.llm.llm_json`` (detect.py imports it lazily inside the
function, so patching the module attribute intercepts it) — zero network, zero live LLM.
"""
from __future__ import annotations

import pytest

import backend.llm as llm_mod
from backend.adapters import get_adapter, select_adapter
from backend.adapters.base import CORE_VERDICT_FIELDS, _PROBLEM_ROLES
from backend.adapters.detect import DetectionResult, detect_content_type
from backend.adapters.entertainment import EntertainmentAdapter

_TRANSCRIPT = {"title": "Some Video", "segments": [{"text": "hello world"}], "text": "hello world"}


def _fake_detect(content_type: str, confidence: float = 0.9, density: str = "high"):
    """A stand-in llm_json that always returns the given content_type as a DetectionResult."""
    def fake(system, user, schema, **kw):
        return schema(content_type=content_type, confidence=confidence, density=density,
                      rationale="mocked")
    return fake


# ── Test 1: metadata (categories=['Comedy']) overrides a lecture LLM guess → entertainment ──
def test_comedy_metadata_overrides_lecture(monkeypatch):
    monkeypatch.setattr(llm_mod, "llm_json", _fake_detect("lecture", confidence=0.9))
    det = detect_content_type(_TRANSCRIPT, {}, meta={"categories": ["Comedy"]})
    assert det.content_type == "entertainment"
    assert det.domain == "entertainment"
    assert det.secondary_content_type == "lecture"     # LLM guess preserved as secondary
    assert det.confidence <= 0.5 < 0.9                  # confidence dropped from the LLM's 0.9

    # and select_adapter returns the lenient EntertainmentAdapter for that override
    monkeypatch.setattr(llm_mod, "llm_json", _fake_detect("lecture", confidence=0.9))
    adapter, det2 = select_adapter(_TRANSCRIPT, {}, meta={"categories": ["Comedy"]})
    assert isinstance(adapter, EntertainmentAdapter)
    assert det2.domain == "entertainment"


# ── Test 1b: title/tags regex path (the grounded 'Calculus Rhapsody' case, categories empty) ──
def test_title_regex_overrides_lecture(monkeypatch):
    monkeypatch.setattr(llm_mod, "llm_json", _fake_detect("math", confidence=0.85))
    t = {"title": "Calculus Rhapsody", "segments": [{"text": "x"}], "text": "x"}
    det = detect_content_type(t, {}, meta={"categories": [], "tags": ["queen", "bohemian"]})
    assert det.domain == "entertainment"
    assert det.secondary_content_type == "math"


# ── Test 2: null artist/track (and non-music metadata) is NO-signal → a lecture stays a lecture ──
def test_null_music_metadata_is_no_signal(monkeypatch):
    monkeypatch.setattr(llm_mod, "llm_json", _fake_detect("lecture", confidence=0.9))
    meta = {"categories": ["Education"], "tags": ["calculus", "lecture"],
            "artist": None, "track": None, "genre": None}
    det = detect_content_type({"title": "Intro to Calculus", "segments": [{"text": "x"}], "text": "x"},
                              {}, meta=meta)
    assert det.content_type == "lecture"
    assert det.domain == "lecture"
    assert det.confidence == pytest.approx(0.9)         # untouched — no override fired


def test_empty_meta_is_no_signal(monkeypatch):
    monkeypatch.setattr(llm_mod, "llm_json", _fake_detect("lecture", confidence=0.8))
    for meta in (None, {}, {"categories": [], "tags": []}):
        det = detect_content_type(_TRANSCRIPT, {}, meta=meta)
        assert det.domain == "lecture", f"meta={meta!r} should not override"


# ── Weighted-nudge guard: a NON-lecture-family LLM guess is NOT force-replaced ──
def test_metadata_does_not_replace_non_lecture_family(monkeypatch):
    monkeypatch.setattr(llm_mod, "llm_json", _fake_detect("interview", confidence=0.9))
    det = detect_content_type(_TRANSCRIPT, {}, meta={"categories": ["Comedy"]})
    assert det.domain == "interview"                    # spoken comedy talk left alone
    assert det.content_type == "interview"


# ── Test 3: EntertainmentAdapter 'moment' anchor gates only on CORE verdict fields ──
def test_entertainment_moment_uses_core_verdict_fields_only():
    adapter = EntertainmentAdapter()
    assert "moment" not in _PROBLEM_ROLES               # so no problem/result gates attach
    fields = adapter.required_verdict_fields("moment")
    assert fields == list(CORE_VERDICT_FIELDS)
    # explicitly: no worked-problem completeness gates
    for gate in ("problem_statement_complete", "reasoning_complete", "result_complete"):
        assert gate not in fields
    # 'moment' is a real anchor with a lenient (single required, within-only) contract
    assert "moment" in adapter.anchor_roles()
    contract = adapter.contract_for("moment")
    assert contract is not None
    required = [e for e in contract.elements if e.necessity == "required"]
    assert len(required) == 1 and required[0].position == "within"


def test_entertainment_registered_and_routes():
    assert isinstance(get_adapter("entertainment"), EntertainmentAdapter)
    for ct in ("music", "song", "comedy", "parody", "entertainment"):
        from backend.adapters.detect import CONTENT_TYPE_TO_DOMAIN
        assert CONTENT_TYPE_TO_DOMAIN[ct] == "entertainment"


# ── Test 4: select_adapter stays backward-compatible when no meta is passed ──
def test_select_adapter_backward_compatible_without_meta(monkeypatch):
    monkeypatch.setattr(llm_mod, "llm_json", _fake_detect("lecture", confidence=0.9))
    adapter, det = select_adapter(_TRANSCRIPT, {})       # no meta arg at all
    assert det.domain == "lecture"
    assert get_adapter("lecture").domain == adapter.domain

    # also detect_content_type's 2-arg form (the pre-GEN1 signature) still works
    monkeypatch.setattr(llm_mod, "llm_json", _fake_detect("tutorial", confidence=0.7))
    det2 = detect_content_type(_TRANSCRIPT, {})
    assert det2.domain == "tutorial"
