"""Cross-stage pipelining neutrality (build side).

The orchestrator computes `content_map` CONCURRENTLY with the perception branch and hands the
finished map to `build_structure(..., content_map=...)`. These offline tests pin that passing a
precomputed map (a) SKIPS the internal `build_content_map` call and (b) yields a `Structure`
byte-identical (modulo the informational `built_at` stamp) to building the very same map
internally — so the overlap is a pure schedule change, never an output change. All LLM stages
are stubbed; nothing here touches a model or the network.
"""
from __future__ import annotations

from backend.adapters.detect import DetectionResult
from backend.pipeline.understand import build as build_mod
from backend.pipeline.understand.models import ContentMap, DependencyGraph, Unit

from .conftest import make_sents


def _stub_stages(monkeypatch, cm, captured):
    """Stub the three heavy stages; count build_content_map calls + record the map extract_units
    actually receives."""
    def fake_cm(sentences, settings, cb):
        captured["build_cm_calls"] += 1
        return cm

    def fake_units(sentences, content_map, adapter, settings, cb, perception):
        captured["units_cm"] = content_map
        return [Unit(unit_id="u0000", start=0.0, end=2.0, sentence_range=(0, len(sentences) - 1))]

    monkeypatch.setattr(build_mod, "build_content_map", fake_cm)
    monkeypatch.setattr(build_mod, "extract_units", fake_units)
    monkeypatch.setattr(build_mod, "build_dependency_graph",
                        lambda units, settings, cb: DependencyGraph())


def test_precomputed_content_map_skips_internal_build(monkeypatch):
    sents = make_sents(6)
    cm = ContentMap(engine="treeseg")
    captured = {"build_cm_calls": 0, "units_cm": None}
    _stub_stages(monkeypatch, cm, captured)

    build_mod.build_structure("vid", {"title": "t"}, sents, adapter=None,
                              detection=DetectionResult(), settings={}, content_map=cm)
    assert captured["build_cm_calls"] == 0        # precomputed → internal builder NOT re-run
    assert captured["units_cm"] is cm             # the exact precomputed map flows to extract_units


def test_none_content_map_builds_internally(monkeypatch):
    sents = make_sents(6)
    cm = ContentMap(engine="treeseg")
    captured = {"build_cm_calls": 0, "units_cm": None}
    _stub_stages(monkeypatch, cm, captured)

    build_mod.build_structure("vid", {"title": "t"}, sents, adapter=None,
                              detection=DetectionResult(), settings={})
    assert captured["build_cm_calls"] == 1        # default path builds it exactly as before
    assert captured["units_cm"] is cm


def test_precompute_vs_internal_structures_identical(monkeypatch):
    """Same map built internally vs passed in → identical Structure (ignoring the built_at stamp)."""
    sents = make_sents(6)
    cm = ContentMap(engine="treeseg")
    captured = {"build_cm_calls": 0, "units_cm": None}
    _stub_stages(monkeypatch, cm, captured)

    internal = build_mod.build_structure("vid", {"title": "t"}, sents, adapter=None,
                                         detection=DetectionResult(), settings={})
    passed = build_mod.build_structure("vid", {"title": "t"}, sents, adapter=None,
                                       detection=DetectionResult(), settings={}, content_map=cm)
    da, db = internal.model_dump(), passed.model_dump()
    da.pop("built_at", None)
    db.pop("built_at", None)
    assert da == db
