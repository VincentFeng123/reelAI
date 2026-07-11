"""I1a: label-free Wave-2 extraction-quality eval columns.

chapter_coverage (content-map topic nodes with ≥1 shipped clip), n_arcs_detected /
n_arc_clips_shipped (detected-arc provenance via spec['arc_id']), plan_engine surfaced
as plan_fallback_rate, n_trims (P2 repair trim moves), severed_pairs_linked / _merged
(P4b notes/warnings) — plus their run_eval wiring (_wave2_columns), the NaN→None
convention, and aggregation behavior. Offline: plain dicts + model fixtures, zero LLM.
"""
from __future__ import annotations

import math

import pytest

import backend.eval.metrics as metrics
import backend.eval.run_eval as R
from backend.pipeline.understand.models import (
    ContentMap, ContentNode, DependencyGraph, Structure, Unit,
)


def _unit(i, node_id="", sr=None):
    s0, s1 = sr if sr is not None else (i, i)
    return Unit(unit_id=f"u{i:04d}", start=i * 10.0, end=i * 10.0 + 9.9,
                sentence_range=(s0, s1), node_id=node_id, role="explanation",
                transcript=f"sentence {i}.")


def _topics(*ranges):
    return [ContentNode(node_id=f"c0.t{k}", level="topic", parent_id="c0",
                        sentence_range=(s0, s1), start=s0 * 10.0, end=s1 * 10.0 + 9.9)
            for k, (s0, s1) in enumerate(ranges)]


def _structure(units, topic_nodes):
    nodes = [ContentNode(node_id="video", level="video",
                         sentence_range=(0, max((u.sentence_range[1] for u in units), default=0)))]
    return Structure(video_id="v", units=units, dependencies=DependencyGraph(),
                     content_map=ContentMap(root_id="video", nodes=nodes + list(topic_nodes)))


# ── chapter_coverage ──────────────────────────────────────────────────────────
def test_chapter_coverage_fraction_by_sentence_containment():
    units = [_unit(i) for i in range(6)]
    by_id = {u.unit_id: u for u in units}
    topics = _topics((0, 1), (2, 3), (4, 5))
    specs = [{"unit_ids": ["u0000"]}, {"unit_ids": ["u0004", "u0005"]}]   # t0 + t2, t1 bare
    assert metrics.chapter_coverage(specs, by_id, topics) == pytest.approx(2 / 3)


def test_chapter_coverage_uses_unit_node_id_when_present():
    u = _unit(0, node_id="c0.t1")               # explicit membership beats containment
    topics = _topics((0, 0), (5, 6))
    assert metrics.chapter_coverage([{"unit_ids": ["u0000"]}], {"u0000": u}, topics) \
        == pytest.approx(0.5)


def test_chapter_coverage_nan_without_topics_zero_without_clips():
    units = [_unit(0)]
    by_id = {u.unit_id: u for u in units}
    assert math.isnan(metrics.chapter_coverage([{"unit_ids": ["u0000"]}], by_id, []))
    assert metrics.chapter_coverage([], by_id, _topics((0, 3))) == 0.0
    # unknown unit ids contribute nothing (never a crash)
    assert metrics.chapter_coverage([{"unit_ids": ["u9999"]}], by_id, _topics((0, 3))) == 0.0


# ── topic_span_coverage (W25-C sibling: shipped seconds / node seconds) ───────
def test_topic_span_coverage_sliver_reads_low_where_chapter_coverage_reads_full():
    units = [_unit(i) for i in range(8)]
    by_id = {u.unit_id: u for u in units}
    topics = _topics((0, 7))                    # one node spanning 0-79.9
    specs = [{"unit_ids": ["u0000"], "start": 0.0, "end": 24.2}]   # the items-3/4 sliver
    assert metrics.chapter_coverage(specs, by_id, topics) == pytest.approx(1.0)  # blind
    assert metrics.topic_span_coverage(specs, topics) == pytest.approx(24.2 / 79.9)


def test_topic_span_coverage_unions_overlaps_and_clips_to_node():
    topics = _topics((0, 3))                    # 0-39.9
    specs = [{"start": 0.0, "end": 20.0}, {"start": 10.0, "end": 30.0}]   # overlap 10s
    assert metrics.topic_span_coverage(specs, topics) == pytest.approx(30.0 / 39.9)
    # spans beyond the node clip to it — coverage never exceeds 1.0
    wide = [{"start": -5.0, "end": 100.0}]
    assert metrics.topic_span_coverage(wide, topics) == pytest.approx(1.0)


def test_topic_span_coverage_nan_without_timing_zero_without_clips():
    untimed = [ContentNode(node_id="t", level="topic", sentence_range=(0, 3))]
    assert math.isnan(metrics.topic_span_coverage([{"start": 0.0, "end": 5.0}], untimed))
    assert math.isnan(metrics.topic_span_coverage([{"start": 0.0, "end": 5.0}], []))
    assert metrics.topic_span_coverage([], _topics((0, 3))) == 0.0
    assert metrics.topic_span_coverage(None, _topics((0, 3))) == 0.0


# ── n_arc_clips / trim_moves / severed_pair_counts ────────────────────────────
def test_n_arc_clips_counts_arc_provenance_only():
    specs = [{"arc_id": "arc_0"}, {"arc_id": ""}, {}, {"arc_id": "arc_1"}]
    assert metrics.n_arc_clips(specs) == 2
    assert metrics.n_arc_clips([]) == 0
    assert metrics.n_arc_clips(None) == 0


def test_n_arc_clips_counts_merge_union_provenance():
    # W25-D: a merged spec whose winner side had no arc_id carries provenance in the
    # merge-union arc_ids list — ANY arc provenance counts; empty containers do not.
    specs = [{"arc_id": "", "arc_ids": ["arc_0"]},          # inherited via merge_partb
             {"arc_ids": []},
             {"arc_id": ""},
             {"arc_id": "arc_1", "arc_ids": ["arc_1", "arc_2"]}]   # not double-counted
    assert metrics.n_arc_clips(specs) == 2


def test_trim_moves_sums_shipped_specs():
    assert metrics.trim_moves([{"n_trims": 2}, {"n_trims": 0}, {}]) == 2
    assert metrics.trim_moves([{"n_trims": None}]) == 0     # tolerant of nulls
    assert metrics.trim_moves([]) == 0


def test_severed_pair_counts_from_notes_and_warnings():
    specs = [
        {"notes": ["continues clip 1"], "warnings": ()},
        {"notes": ["some other note"], "warnings": ("merged_severed_pair",)},
        {"notes": [], "warnings": ("merged_overlap",)},
        {},
    ]
    assert metrics.severed_pair_counts(specs) == (1, 1)
    assert metrics.severed_pair_counts([]) == (0, 0)
    assert metrics.severed_pair_counts(None) == (0, 0)


# ── run_eval._wave2_columns wiring ────────────────────────────────────────────
def _cols(specs, stats, topic_ranges=((0, 1), (2, 3))):
    n = max((s1 for _s0, s1 in topic_ranges), default=0) + 1
    units = [_unit(i) for i in range(n)]
    st = _structure(units, _topics(*topic_ranges))
    return R._wave2_columns(st, specs, stats)


def test_wave2_columns_plan_engine_mapping():
    c = _cols([], {"plan_engine": "plan", "n_arcs_detected": 2})
    assert c["plan_engine"] == "plan" and c["plan_fallback_rate"] == 0.0
    assert c["n_arcs_detected"] == 2
    c = _cols([], {"plan_engine": "plan-fallback", "n_arcs_detected": 0})
    assert c["plan_fallback_rate"] == 1.0
    # priority: the plan engine never ran — no fake zeros diluting the aggregate
    c = _cols([], {"plan_engine": "priority"})
    assert c["plan_engine"] == "priority"
    assert c["plan_fallback_rate"] is None                  # NaN → None convention
    assert c["n_arcs_detected"] is None                     # detection never ran


def test_wave2_columns_full_shape():
    specs = [
        {"unit_ids": ["u0000"], "arc_id": "arc_0", "n_trims": 1,
         "notes": ["continues clip 1"], "warnings": ()},
        {"unit_ids": ["u0002"], "warnings": ("merged_severed_pair",)},
    ]
    c = _cols(specs, {"plan_engine": "plan", "n_arcs_detected": 1})
    assert c["chapter_coverage"] == pytest.approx(1.0)
    assert c["n_arc_clips_shipped"] == 1
    assert c["n_trims"] == 1
    assert c["severed_pairs_linked"] == 1
    assert c["severed_pairs_merged"] == 1


def test_wave2_columns_empty_stats_keeps_row_shape():
    c = _cols([], {})                                       # e.g. assembly early-returned
    for k in ("chapter_coverage", "topic_span_coverage", "plan_engine", "plan_fallback_rate",
              "n_arcs_detected", "n_arc_clips_shipped", "n_trims", "severed_pairs_linked",
              "severed_pairs_merged", "anchor_budget", "n_refund_rounds", "n_refund_clips"):
        assert k in c
    assert c["plan_engine"] == "" and c["plan_fallback_rate"] is None
    assert c["n_arcs_detected"] is None


def test_wave2_columns_q1_budget_and_refund_signals():
    # Q1 stats surface as columns; absent (assembly never ran) → None, never a fake 0
    c = _cols([], {"anchor_budget": 23, "n_refund_rounds": 1, "n_refund_clips": 3})
    assert c["anchor_budget"] == 23
    assert c["n_refund_rounds"] == 1 and c["n_refund_clips"] == 3
    c = _cols([], {})
    assert c["anchor_budget"] is None
    assert c["n_refund_rounds"] is None and c["n_refund_clips"] is None


# ── aggregation + reporting conventions ───────────────────────────────────────
def test_new_keys_preferred_order_and_string_engine_excluded():
    results = [{"runs": [{"chapter_coverage": 0.5, "n_trims": 1, "plan_engine": "plan",
                          "plan_fallback_rate": 0.0, "phantom_verdict_rate": 0.1}]}]
    keys = R._numeric_keys(results)
    assert "plan_engine" not in keys                        # strings never aggregate
    assert keys.index("chapter_coverage") < keys.index("phantom_verdict_rate")
    assert keys.index("plan_fallback_rate") < keys.index("n_trims")


def test_plan_fallback_rate_aggregates_and_none_drops_video():
    results = [{"runs": [{"plan_fallback_rate": 0.0}, {"plan_fallback_rate": 1.0}]},
               {"runs": [{"plan_fallback_rate": None}, {"plan_fallback_rate": None}]}]
    s = R.aggregate_over_runs(results, "plan_fallback_rate")
    assert s["videos"] == 1 and s["mean"] == pytest.approx(0.5)


# ── payload seam (P4 reconcile): the watch-first note reaches the frontend ────
def test_build_embed_clips_whitelists_notes():
    from backend.orchestrator import _build_embed_clips
    spec = {"start": 0.0, "end": 5.0, "notes": ["continues clip 1"],
            "prerequisite_clips": [1]}
    clips = _build_embed_clips([spec, {"start": 6.0, "end": 9.0}], "vid")
    assert clips[0]["notes"] == ["continues clip 1"]
    assert clips[1]["notes"] == []                          # always present, always a list
    assert clips[0]["prerequisite_clips"] == [1]            # machine-readable link intact
