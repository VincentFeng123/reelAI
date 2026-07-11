"""W25-A loud index clamp: a candidate whose (cached) unit sentence_range overruns the
live sentence list is a stale-structure symptom — it must carry a spec warning and count
into stats['index_clamp_events'] instead of silently snapping to the last sentence
(which shipped clips of unrelated tail text). Off-by-one tail overshoot (exclusive-end
convention) stays silent. All offline (llm_json monkeypatched)."""
from __future__ import annotations

import backend.llm as llm_mod
from backend import config
from backend.adapters.base import BaseAdapter
from backend.pipeline.assemble import assemble_clips
from backend.pipeline.assemble.candidates import _clamped_range, build_candidate
from backend.pipeline.assemble.graph import Graph
from backend.pipeline.assemble.validate import JudgeVerdict
from backend.pipeline.understand.models import (
    ContentMap, ContentNode, DependencyGraph, Structure, Unit,
)

from .conftest import mini_sents


def _units(roles):
    return [Unit(unit_id=f"u{i:04d}", start=i * 10.0, end=i * 10.0 + 9.9,
                 sentence_range=(i, i), role=r, transcript=f"sentence {i}.",
                 summary=f"unit {i}") for i, r in enumerate(roles)]


# ── the clamp helper itself ───────────────────────────────────────────────────
def test_clamped_range_in_bounds_is_silent():
    assert _clamped_range(1, 3, 5) == (1, 3, ())


def test_clamped_range_off_by_one_tail_is_tolerated():
    # exclusive-end convention overshoots the last index by exactly 1 — benign, silent.
    lo, hi, warn = _clamped_range(2, 5, 5)
    assert (lo, hi) == (2, 4) and warn == ()


def test_clamped_range_beyond_one_is_loud():
    lo, hi, warn = _clamped_range(10, 12, 5)     # a 183-sentence app vs 322-index cache shape
    assert (lo, hi) == (4, 4)
    assert warn == ("sentence_index_clamped:+8",)


# ── build_candidate surfaces the warning ──────────────────────────────────────
def test_build_candidate_carries_clamp_warning():
    sents = mini_sents(4)
    units = _units(["definition", "explanation", "claim", "summary"])
    units[2].sentence_range = (10, 12)           # poisoned: indexes a bigger sentence universe
    by_id = {u.unit_id: u for u in units}
    cand = build_candidate(units[2], Graph([], units), BaseAdapter(), units, by_id, sents,
                           {u.unit_id: 1.0 for u in units},
                           {"closure_max_span_s": 999.0, "max_clip_duration_s": 500.0})
    assert cand is not None
    # closure may inline nearby context (i_start moves), but the poisoned end index MUST
    # land inside the live list — and loudly.
    assert cand.i_end == 3 and cand.i_start <= 3
    assert any(w.startswith("sentence_index_clamped") for w in cand.warnings)


# ── end-to-end telemetry through assemble_clips ───────────────────────────────
def test_assemble_counts_clamp_events_and_ships_the_warning(monkeypatch):
    monkeypatch.setattr(config, "JUDGE_MODEL", "")
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "same")
    sents = mini_sents(4)
    units = _units(["definition", "explanation", "claim", "summary"])
    units[2].sentence_range = (10, 12)           # ONE stale-indexed anchor unit
    st = Structure(video_id="v", units=units, dependencies=DependencyGraph(),
                   content_map=ContentMap(root_id="video", nodes=[
                       ContentNode(node_id="video", level="video", sentence_range=(0, 3))]))

    def fake(system, user, schema, **kw):
        assert schema is JudgeVerdict, f"priority path must only judge, got {schema}"
        return JudgeVerdict(reasoning="ok", score_10=9, understandable=True)
    monkeypatch.setattr(llm_mod, "llm_json", fake)

    settings = {"min_clip_duration_s": 1.0, "max_clip_duration_s": 500.0,
                "min_comprehension_score": 0.7, "quality_floor": 0.0, "max_clips": 12,
                "max_anchors": 12, "closure_max_span_s": 999.0,
                "anchor_selector": "priority", "refund_rounds": 0}
    stats: dict = {}
    specs, _notes, _rej = assemble_clips(st, "", sents, "u", "v", settings, BaseAdapter(),
                                         stats=stats)
    assert stats["index_clamp_events"] == 1      # exactly the one poisoned anchor
    assert any(any(w.startswith("sentence_index_clamped") for w in (s.get("warnings") or ()))
               for s in specs)                   # the warning rides the shipped spec


def test_assemble_clamp_counter_zero_when_fresh(monkeypatch):
    monkeypatch.setattr(config, "JUDGE_MODEL", "")
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "same")
    sents = mini_sents(4)
    units = _units(["definition", "explanation", "claim", "summary"])
    st = Structure(video_id="v", units=units, dependencies=DependencyGraph(),
                   content_map=ContentMap(root_id="video", nodes=[
                       ContentNode(node_id="video", level="video", sentence_range=(0, 3))]))
    monkeypatch.setattr(llm_mod, "llm_json",
                        lambda *a, **kw: JudgeVerdict(reasoning="ok", score_10=9,
                                                      understandable=True))
    settings = {"min_clip_duration_s": 1.0, "max_clip_duration_s": 500.0,
                "min_comprehension_score": 0.7, "quality_floor": 0.0, "max_clips": 12,
                "max_anchors": 12, "closure_max_span_s": 999.0,
                "anchor_selector": "priority", "refund_rounds": 0}
    stats: dict = {}
    specs, _notes, _rej = assemble_clips(st, "", sents, "u", "v", settings, BaseAdapter(),
                                         stats=stats)
    assert specs
    assert stats["index_clamp_events"] == 0
