"""W25-G eval columns: standing inventory recall (+ the qP-9wwRrJbg golden file),
phantom_quotable_rate, structure_source, the new count columns
(n_min_duration_extensions / index_clamp_events / forward_requires_edges /
n_refund_superset_replaced), quality_floor as a config constant, and the run_eval
wiring (gold-gated _measure column, --verbose per-item listing, eval_video's
structure_source + run-artifact persistence). Offline: judge_clip monkeypatched,
plain dict fixtures, tmp_path WORK_DIR — zero LLM/network."""
from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

import backend.eval.metrics as metrics
import backend.eval.run_eval as R
from backend import config
from backend.eval.golden import load_golden
from backend.pipeline.assemble.integrity import Rejection
from backend.pipeline.assemble.tests.conftest import FakeAdapter, mini_sents, mini_units
from backend.pipeline.understand.models import (
    ContentMap, ContentNode, DependencyGraph, Edge, Structure, Unit,
)


def _unit(i, node_id=""):
    return Unit(unit_id=f"u{i:04d}", start=i * 10.0, end=i * 10.0 + 9.9,
                sentence_range=(i, i), node_id=node_id, role="explanation",
                transcript=f"sentence {i}.")


def _topics(*ranges):
    return [ContentNode(node_id=f"c0.t{k}", level="topic", parent_id="c0",
                        sentence_range=(s0, s1), start=s0 * 10.0, end=s1 * 10.0 + 9.9)
            for k, (s0, s1) in enumerate(ranges)]


def _structure(units, topic_nodes=()):
    nodes = [ContentNode(node_id="video", level="video",
                         sentence_range=(0, max((u.sentence_range[1] for u in units), default=0)))]
    return Structure(video_id="v", units=units, dependencies=DependencyGraph(),
                     content_map=ContentMap(root_id="video", nodes=nodes + list(topic_nodes)))


def _rej(stage="repair", verified=(), unverified=(), confirmed=False):
    return Rejection(cand_id="c", title="t", role="r", stage=stage, reason="x",
                     verified_kinds=tuple(verified), unverified_kinds=tuple(unverified),
                     kill_confirmed=confirmed)


# ── inventory_recall / inventory_coverage math ────────────────────────────────
def test_inventory_recall_nan_when_no_items():
    assert math.isnan(metrics.inventory_recall([], [{"start": 0.0, "end": 10.0}]))
    assert math.isnan(metrics.inventory_recall(None, []))


def test_inventory_recall_boundary_exactly_060_counts_covered():
    item = {"start": 0.0, "end": 10.0}
    assert metrics.inventory_recall([item], [{"start": 0.0, "end": 6.0}]) == 1.0
    assert metrics.inventory_recall([item], [{"start": 0.0, "end": 5.9}]) == 0.0


def test_inventory_recall_is_one_sided_not_iou():
    # a long clip fully containing a short item covers it (IoU would read ~0.04)
    assert metrics.inventory_recall([{"start": 100.0, "end": 110.0}],
                                    [{"start": 0.0, "end": 240.0}]) == 1.0


def test_inventory_recall_needs_a_single_spec_not_a_union():
    # two half-covering slivers (union 0.8) teach neither half — deliberately NOT covered
    item = {"start": 0.0, "end": 10.0}
    specs = [{"start": 0.0, "end": 4.0}, {"start": 4.0, "end": 8.0}]
    assert metrics.inventory_recall([item], specs) == 0.0


def test_inventory_recall_fraction_and_per_item_detail():
    items = [{"n": 1, "start": 0.0, "end": 10.0}, {"n": 2, "start": 20.0, "end": 30.0}]
    specs = [{"start": 0.0, "end": 10.0}]
    assert metrics.inventory_recall(items, specs) == pytest.approx(0.5)
    cov = metrics.inventory_coverage(items, specs)
    assert cov[0][0]["n"] == 1 and cov[0][1] == pytest.approx(1.0) and cov[0][2]
    assert cov[1][1] == 0.0 and not cov[1][2]


def test_inventory_zero_duration_item_never_covered():
    assert metrics.inventory_recall([{"start": 5.0, "end": 5.0}],
                                    [{"start": 0.0, "end": 10.0}]) == 0.0


# ── the qP-9wwRrJbg golden file ───────────────────────────────────────────────
def test_qp_golden_file_loads_with_the_26_inventory_items():
    gold = load_golden("qP-9wwRrJbg")
    assert gold is not None and gold["video_id"] == "qP-9wwRrJbg"
    assert gold["topics"] == ["kinematics"]     # eval topic no longer defaults to ''
    items = gold["inventory"]
    assert [it["n"] for it in items] == list(range(1, 27))
    for it in items:
        assert {"n", "type", "start", "end", "description"} <= set(it)
        assert 0.0 <= float(it["start"]) < float(it["end"]) <= 1432.76


def test_qp_golden_reproduces_the_audited_baseline_recall():
    # the two clips the audited run actually shipped cover items 3-6 and 10-13 under the
    # ≥0.60 rule (item 7 truncates at 0.49) — the audit's 8/26; the 15/26=0.577 handoff
    # baseline is the post-quick-wins gate, measured live, not from these two clips.
    gold = load_golden("qP-9wwRrJbg")
    shipped = [{"start": 106.87, "end": 324.28}, {"start": 441.72, "end": 551.06}]
    cov = metrics.inventory_coverage(gold["inventory"], shipped)
    covered_ns = [it["n"] for it, _f, ok in cov if ok]
    assert covered_ns == [3, 4, 5, 6, 10, 11, 12, 13]
    assert metrics.inventory_recall(gold["inventory"], shipped) == pytest.approx(8 / 26)


# ── phantom_quotable_rate ─────────────────────────────────────────────────────
def test_phantom_quotable_rate_filters_to_quotable_kinds():
    specs = [{"verified_kinds": ("off_topic",),
              "unverified_kinds": ("missing_prerequisite", "over_inclusion")}]
    # quotable population: off_topic (verified) + over_inclusion (phantom) → 1/2
    assert metrics.phantom_quotable_rate(specs, []) == pytest.approx(0.5)


def test_phantom_quotable_rate_nan_when_only_absence_or_legacy_counts():
    specs = [{"unverified_kinds": ("missing_prerequisite", "missing_result")}]
    assert math.isnan(metrics.phantom_quotable_rate(specs, []))
    # count-only legacy specs (no kind tuples) contribute nothing
    assert math.isnan(metrics.phantom_quotable_rate([{"n_failure_reasons": 3,
                                                      "n_verified": 0}], []))
    assert math.isnan(metrics.phantom_quotable_rate([], []))
    assert math.isnan(metrics.phantom_quotable_rate(None, None))


def test_phantom_quotable_rate_reads_rejections_and_kind_variants():
    rejs = [_rej(verified=("not-source-grounded",), unverified=("unresolved_reference",))]
    assert metrics.phantom_quotable_rate([], rejs) == pytest.approx(0.5)
    # LLM spelling variants normalize into the quotable set
    assert metrics.phantom_quotable_rate([{"unverified_kinds": ("Over-Inclusion",)}], []) == 1.0


# ── forward_requires_edges / min_duration_extensions ──────────────────────────
def test_forward_requires_edges_counts_only_forward_requires():
    units = [_unit(0), _unit(1)]
    edges = [Edge(source="u0000", target="u0001", relation="requires"),    # forward → counts
             Edge(source="u0001", target="u0000", relation="requires"),    # backward → fine
             Edge(source="u0000", target="u0001", relation="refers_to"),   # not requires
             Edge(source="u0000", target="zzz", relation="requires")]      # unknown id ignored
    st = Structure(video_id="v", units=units, dependencies=DependencyGraph(edges=edges))
    assert metrics.forward_requires_edges(st) == 1
    assert metrics.forward_requires_edges(_structure([_unit(0)])) == 0


def test_min_duration_extensions_counts_snap_warning():
    specs = [{"warnings": ("extended_for_min_duration",)},
             {"warnings": ["extended_for_min_duration", "trimmed_start"]},   # list form too
             {"warnings": ("capped_max_duration",)}, {}]
    assert metrics.min_duration_extensions(specs) == 2
    assert metrics.min_duration_extensions([]) == 0
    assert metrics.min_duration_extensions(None) == 0


# ── column wiring (_integrity_columns / _wave2_columns) ───────────────────────
def test_integrity_columns_include_phantom_quotable_rate():
    specs = [{"n_failure_reasons": 2, "n_verified": 1,
              "verified_kinds": ("off_topic",), "unverified_kinds": ("off_topic",)}]
    cols = R._integrity_columns(specs, [])
    assert cols["phantom_quotable_rate"] == pytest.approx(0.5)
    assert R._integrity_columns([], [])["phantom_quotable_rate"] is None   # NaN → None


def _cols(specs, stats):
    units = [_unit(i) for i in range(4)]
    return R._wave2_columns(_structure(units, _topics((0, 1), (2, 3))), specs, stats)


def test_wave2_columns_w25g_none_vs_zero_semantics():
    # stats-borne counts: None when assembly never filled them (a missing measurement
    # must not dilute the aggregate as a fake 0); spec/structure-derived counts: real 0s.
    c = _cols([], {})
    assert c["index_clamp_events"] is None
    assert c["n_refund_superset_replaced"] is None
    assert c["n_min_duration_extensions"] == 0
    assert c["forward_requires_edges"] == 0
    c = _cols([{"unit_ids": [], "warnings": ("extended_for_min_duration",)}],
              {"index_clamp_events": 2, "n_refund_superset_replaced": 1})
    assert c["index_clamp_events"] == 2
    assert c["n_refund_superset_replaced"] == 1
    assert c["n_min_duration_extensions"] == 1


def test_new_columns_present_in_preferred_key_order():
    for k in ("inventory_recall", "n_refund_superset_replaced", "n_min_duration_extensions",
              "index_clamp_events", "forward_requires_edges", "phantom_quotable_rate"):
        assert k in R._PREFERRED_KEYS
    assert R._PREFERRED_KEYS.index("inventory_recall") \
        < R._PREFERRED_KEYS.index("chapter_coverage")
    assert R._PREFERRED_KEYS.index("phantom_verdict_rate") \
        < R._PREFERRED_KEYS.index("phantom_quotable_rate")


# ── quality_floor: config constant + DEFAULTS key (behavior identical) ────────
def test_quality_floor_is_a_config_constant_with_defaults_inherit():
    assert config.QUALITY_FLOOR == pytest.approx(0.45)      # the old inline literal
    assert "quality_floor" in config.DEFAULTS
    assert config.DEFAULTS["quality_floor"] is None         # None → inherit the constant


# ── _measure wiring: gold-gated inventory column + --verbose listing ──────────
def _fake_verdict(*_a, **_k):
    return SimpleNamespace(score=1.0, error=False, failure_reasons=[])


def _measure_fixture():
    sents = mini_sents(4)
    units = mini_units(sents)
    st = Structure(video_id="v", units=units, dependencies=DependencyGraph())
    specs = [{"start": 0.0, "end": 9.9, "sentence_start_idx": 0, "sentence_end_idx": 0,
              "unit_ids": ["u0000"], "role": "explanation"}]
    det = SimpleNamespace(domain="lecture")
    return st, specs, sents, det, {"min_comprehension_score": 0.7}


def test_measure_inventory_recall_is_gold_gated(monkeypatch, capsys):
    monkeypatch.setattr(metrics, "judge_clip", _fake_verdict)
    st, specs, sents, det, settings = _measure_fixture()
    gold = {"inventory": [
        {"n": 1, "type": "definition", "start": 0.0, "end": 9.9, "description": "covered item"},
        {"n": 2, "type": "concept", "start": 20.0, "end": 30.0, "description": "missed item"}]}
    m = R._measure(st, specs, sents, FakeAdapter(), det, "", gold, settings, verbose=True)
    assert m["inventory_recall"] == pytest.approx(0.5)
    out = capsys.readouterr().out
    assert "inventory coverage" in out
    assert "covered" in out and "MISSED" in out
    assert "overlap=1.00" in out and "overlap=0.00" in out
    # no gold inventory → no column (gold-gated, like anchor_recall)
    m2 = R._measure(st, specs, sents, FakeAdapter(), det, "", {}, settings)
    assert "inventory_recall" not in m2


# ── eval_video wiring: structure_source column + run-artifact persistence ─────
def test_eval_video_structure_source_and_run_artifacts(tmp_path, monkeypatch):
    vid = "vidW25G"
    (tmp_path / vid).mkdir()
    (tmp_path / vid / "transcript.json").write_text(json.dumps({"words": [], "segments": []}))
    monkeypatch.setattr(config, "WORK_DIR", tmp_path)       # artifacts land in tmp, not work/
    sents = mini_sents(3)
    st = Structure(video_id=vid, units=mini_units(sents), dependencies=DependencyGraph())
    monkeypatch.setattr(R, "_sentences", lambda t, v=None: sents)
    monkeypatch.setattr(R, "load_golden", lambda v: {})
    monkeypatch.setattr(R, "select_adapter",
                        lambda t, s: (FakeAdapter(), SimpleNamespace(domain="lecture")))
    monkeypatch.setattr(R, "_maybe_perceive", lambda *a: None)
    monkeypatch.setattr(R, "resolve_structure", lambda *a, **k: (st, "cached-stale"))
    rej = Rejection(cand_id="c", title="t", role="r", stage="build", reason="died")

    def fake_assemble(st_, topic, s_, url, video_id, settings, adapter, stats=None):
        stats["plan_proposals"] = [{"kind": "unit", "ref_id": "u0000"}]
        stats["arcs_verified"] = []
        return ([{"start": 0.0, "end": 9.9, "sentence_start_idx": 0, "sentence_end_idx": 0,
                  "unit_ids": [], "warnings": ()}], "", [rej])
    monkeypatch.setattr(R, "assemble_clips", fake_assemble)
    monkeypatch.setattr(R, "_measure", lambda *a, **k: {})

    res = R.eval_video(vid, None)
    run = res["runs"][0]
    assert run["structure_source"] == "cached-stale"        # the poisoned-cache tripwire
    assert run["rejections_build"] == 1                     # 'build' is a counted stage
    run_dirs = list((tmp_path / vid / "runs").iterdir())
    assert len(run_dirs) == 1
    assert {p.name for p in run_dirs[0].iterdir()} == {"plan.json", "arcs.json",
                                                       "shipped.json", "ledger.json"}
    ledger = json.loads((run_dirs[0] / "ledger.json").read_text())
    assert ledger[0]["stage"] == "build" and ledger[0]["reason"] == "died"
    plan = json.loads((run_dirs[0] / "plan.json").read_text())
    assert plan[0]["ref_id"] == "u0000"


# ── end-to-end: quotable phantom kinds flow assemble → columns (offline) ──────
def test_phantom_quotable_rate_from_real_assemble_run(monkeypatch):
    import backend.llm as llm_mod
    from backend.pipeline.assemble import assemble_clips
    from backend.pipeline.assemble.validate import FailureReason, JudgeVerdict
    monkeypatch.setattr(config, "JUDGE_MODEL", "")
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "same")

    def fake(system, user, schema, **kw):                   # quotable-kind phantom verdicts
        return JudgeVerdict(reasoning="bad", score_10=2, understandable=False,
                            topic_identifiable=False, purpose_identifiable=False,
                            failure_reasons=[FailureReason(
                                kind="off_topic",
                                evidence_quote="words that are not in the clip")])
    monkeypatch.setattr(llm_mod, "llm_json", fake)

    class AnchorAdapter(FakeAdapter):
        def is_anchor_role(self, role):
            return True

        def anchor_priority(self, role):
            return 0.9

        def facet_for(self, role):
            return "other"

        def valid_roles(self):
            return {"explanation"}

    sents = mini_sents(4)
    units = mini_units(sents)
    st = Structure(video_id="v", units=units, dependencies=DependencyGraph(),
                   content_map=ContentMap(nodes=[ContentNode(node_id="video", level="video",
                                                             sentence_range=(0, 3))]))
    settings = {"min_clip_duration_s": 1.0, "max_clip_duration_s": 500.0,
                "min_comprehension_score": 0.7, "quality_floor": 0.0, "max_clips": 12,
                "max_anchors": 12, "closure_max_span_s": 999.0,
                "anchor_selector": "priority"}
    specs, _notes, rejections = assemble_clips(st, "", sents, "u", "v", settings,
                                               AnchorAdapter())
    assert specs                                            # phantoms ship flagged, never kill
    assert all(s.get("unverified_kinds") == ("off_topic",) for s in specs)
    cols = R._integrity_columns(specs, rejections)
    assert cols["phantom_verdict_rate"] == 1.0
    assert cols["phantom_quotable_rate"] == 1.0             # off_topic IS quotable → counted
