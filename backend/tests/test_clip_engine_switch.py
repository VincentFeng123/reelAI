"""Tests for CLIP_ENGINE routing (Task 6).

Covers three things:
1. test_engine_resolution  — default→gemini; topic and unit remain explicit experiments.
2. test_orchestrator_imports_topic_engine  — the brief's import smoke-test.
3. test_resolve_assemble_fn_dispatch  — REAL wiring: _resolve_assemble_fn (the shared
   helper used by BOTH orchestrator and cli) returns the correct callable object.
   This tests the actual dispatch logic, not a reimplementation of it.
"""
from backend import config
from backend.pipeline.assemble import _resolve_assemble_fn, assemble_clips
from backend.pipeline.assemble.topics import assemble_topic_clips


# ── brief-specified tests ──────────────────────────────────────────────────────

def _resolve(settings):
    # mirrors the one-liner used at both call sites
    return str(settings.get("clip_engine") or config.CLIP_ENGINE).lower()


def test_engine_resolution():
    assert _resolve({}) == "gemini"
    assert _resolve({"clip_engine": "topic"}) == "topic"
    assert _resolve({"clip_engine": "unit"}) == "unit"
    assert _resolve({"clip_engine": None}) == config.CLIP_ENGINE


def test_orchestrator_imports_topic_engine():
    # the topic engine is importable where the orchestrator wires it
    from backend.pipeline.assemble.topics import assemble_topic_clips
    assert callable(assemble_topic_clips)


# ── real-wiring test ───────────────────────────────────────────────────────────

def test_resolve_assemble_fn_explicit_topic_routes_to_topic():
    fn = _resolve_assemble_fn({"clip_engine": "topic"})
    assert fn is assemble_topic_clips


def test_resolve_assemble_fn_unit_routes_to_unit():
    """clip_engine='unit' → legacy unit engine."""
    fn = _resolve_assemble_fn({"clip_engine": "unit"})
    assert fn is assemble_clips


def test_resolve_assemble_fn_rejects_gemini_because_it_dispatches_before_assembly():
    import pytest

    with pytest.raises(ValueError, match="not a full-pipeline"):
        _resolve_assemble_fn({"clip_engine": "gemini"})
    with pytest.raises(ValueError, match="not a full-pipeline"):
        _resolve_assemble_fn({})


def test_orchestrator_uses_resolve_assemble_fn():
    """_resolve_assemble_fn is imported at the orchestrator's call site."""
    import inspect
    import backend.orchestrator as orch
    src = inspect.getsource(orch)
    assert "_resolve_assemble_fn" in src, (
        "orchestrator._run_full must call _resolve_assemble_fn(settings) to route the engine"
    )
    assert 'engine == "gemini"' in src and "_run_gemini_segment" in src


def test_cli_uses_resolve_assemble_fn():
    """_resolve_assemble_fn is imported and used at the cli's call site."""
    import inspect
    import backend.cli as cli
    src = inspect.getsource(cli)
    assert "_resolve_assemble_fn" in src, (
        "backend.cli must call _resolve_assemble_fn(settings) to route the engine"
    )
    assert 'engine == "gemini"' in src and "segment_clips" in src
