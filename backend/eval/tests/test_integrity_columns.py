"""F-integration: label-free judge-integrity eval columns (Wave 1).

phantom_verdict_rate (share of ALL emitted judge failure reasons that failed quote
verification, across shipped specs AND kill rejections), n_shipped_flagged /
shipped_flagged_rate, the stage-keyed rejections_* counts including the new
post_snap_judge stage, and the verified/unverified kill split — plus their aggregation
behavior. Offline: plain dicts + Rejection records; the one end-to-end case monkeypatches
llm_json like the assemble tests do.
"""
from __future__ import annotations

import math

import pytest

import backend.eval.metrics as metrics
import backend.eval.run_eval as R
from backend.pipeline.assemble.integrity import Rejection


def _spec(n_reasons=0, n_verified=0, warnings=(), **kw):
    return {"n_failure_reasons": n_reasons, "n_verified": n_verified,
            "warnings": tuple(warnings), **kw}


def _rej(stage="repair", verified=(), unverified=(), confirmed=False):
    return Rejection(cand_id="c", title="t", role="r", stage=stage, reason="x",
                     verified_kinds=tuple(verified), unverified_kinds=tuple(unverified),
                     kill_confirmed=confirmed)


# ── phantom_verdict_rate ──────────────────────────────────────────────────────
def test_phantom_rate_counts_specs_and_rejections():
    specs = [_spec(2, 1), _spec(1, 1)]                       # 3 reasons, 1 phantom
    rejs = [_rej(verified=("off_topic",), unverified=("missing_result", "other"),
                 confirmed=True)]                            # 3 reasons, 2 phantoms
    assert metrics.phantom_verdict_rate(specs, rejs) == pytest.approx(3 / 6)


def test_phantom_rate_specs_only_and_rejections_only():
    assert metrics.phantom_verdict_rate([_spec(4, 1)], []) == pytest.approx(0.75)
    assert metrics.phantom_verdict_rate([], [_rej(unverified=("a",))]) == 1.0


def test_phantom_rate_nan_when_no_reasons_recorded():
    assert math.isnan(metrics.phantom_verdict_rate([], []))
    assert math.isnan(metrics.phantom_verdict_rate([_spec(0, 0)], []))
    # a mechanical drop (no verification stats recorded) contributes nothing
    r = Rejection(cand_id="c", title="t", role="r", stage="quality_floor", reason="x",
                  failure_kinds=["off_topic"])
    assert math.isnan(metrics.phantom_verdict_rate([], [r]))


def test_phantom_rate_tolerates_missing_keys():
    assert metrics.phantom_verdict_rate([{}, _spec(1, 0)], None) == 1.0


# ── shipped_flagged ───────────────────────────────────────────────────────────
def test_shipped_flagged_counts_only_the_flag_warning():
    specs = [_spec(warnings=("unverified_judge_concerns",)),
             _spec(warnings=["unverified_judge_concerns", "unjudged"]),   # list form too
             _spec(warnings=("merged_overlap",)),
             _spec()]
    n, rate = metrics.shipped_flagged(specs)
    assert n == 2
    assert rate == pytest.approx(0.5)


def test_shipped_flagged_empty_specs_is_nan_rate():
    n, rate = metrics.shipped_flagged([])
    assert n == 0 and math.isnan(rate)


# ── kill_counts ───────────────────────────────────────────────────────────────
def test_kill_counts_split_by_confirmation_and_stage():
    rejs = [
        _rej(stage="repair", verified=("off_topic",), confirmed=True),   # verified kill
        _rej(stage="post_snap_judge"),                                   # judge kill, unconfirmed
        _rej(stage="post_merge_judge"),                                  # judge kill, unconfirmed
        _rej(stage="dedupe"),                                            # mechanical: neither
        _rej(stage="quality_floor"),
        _rej(stage="snap"),
        _rej(stage="max_clips"),
    ]
    assert metrics.kill_counts(rejs) == (1, 2)
    assert metrics.kill_counts([]) == (0, 0)
    assert metrics.kill_counts(None) == (0, 0)


# ── run_eval column wiring ────────────────────────────────────────────────────
def test_rejection_stages_single_source_of_truth():
    # every stage literal the ledger can emit (integrity.Rejection docstring) is counted;
    # 'build' (W25-G): candidate builders returning None are now ledgered, not silent.
    assert set(R.REJECTION_STAGES) == {"build", "repair", "snap", "dedupe",
                                       "post_merge_judge", "post_snap_judge",
                                       "quality_floor", "max_clips"}


def test_integrity_columns_shapes_and_nan_convention():
    specs = [_spec(1, 0, warnings=("unverified_judge_concerns",), merged=True), _spec()]
    rejs = [_rej(stage="post_snap_judge", verified=("off_topic",), unverified=("other",)),
            _rej(stage="repair", verified=("off_topic",), confirmed=True)]
    cols = R._integrity_columns(specs, rejs)
    for stage in R.REJECTION_STAGES:
        assert f"rejections_{stage}" in cols
    assert cols["rejections_post_snap_judge"] == 1
    assert cols["rejections_repair"] == 1
    assert cols["n_merged"] == 1
    # reasons: spec1 1(1 phantom) + post_snap 2(1 phantom) + repair 1(0) → 2/4
    assert cols["phantom_verdict_rate"] == pytest.approx(0.5)
    assert cols["n_shipped_flagged"] == 1
    assert cols["shipped_flagged_rate"] == pytest.approx(0.5)
    assert cols["verified_kill"] == 1 and cols["unverified_kill"] == 1


def test_integrity_columns_empty_run_uses_nan_to_none():
    cols = R._integrity_columns([], [])
    assert cols["phantom_verdict_rate"] is None            # NaN → None (JSON null) convention
    assert cols["shipped_flagged_rate"] is None
    assert cols["n_shipped_flagged"] == 0
    assert cols["verified_kill"] == 0 and cols["unverified_kill"] == 0
    assert all(cols[f"rejections_{s}"] == 0 for s in R.REJECTION_STAGES)


def test_new_keys_aggregate_over_runs_and_order():
    # counts and rates aggregate like any other numeric metric; None runs drop the video
    results = [{"runs": [{"verified_kill": 1, "phantom_verdict_rate": 0.5},
                         {"verified_kill": 2, "phantom_verdict_rate": 0.7}]},
               {"runs": [{"verified_kill": 0, "phantom_verdict_rate": None},
                         {"verified_kill": 0, "phantom_verdict_rate": None}]}]
    s = R.aggregate_over_runs(results, "verified_kill")
    assert s["videos"] == 2 and s["mean"] == pytest.approx(0.75)
    s = R.aggregate_over_runs(results, "phantom_verdict_rate")
    assert s["videos"] == 1 and s["mean"] == pytest.approx(0.6)
    keys = R._numeric_keys(results)
    assert keys.index("phantom_verdict_rate") < keys.index("verified_kill")


# ── end-to-end: assemble run → columns (offline, llm_json mocked) ─────────────
def test_columns_from_real_assemble_run(monkeypatch):
    import backend.llm as llm_mod
    from backend import config
    from backend.pipeline.assemble import assemble_clips
    from backend.pipeline.assemble.tests.conftest import FakeAdapter, mini_sents, mini_units
    from backend.pipeline.assemble.validate import FailureReason, JudgeVerdict
    from backend.pipeline.understand.models import (
        ContentMap, ContentNode, DependencyGraph, Structure,
    )
    monkeypatch.setattr(config, "JUDGE_MODEL", "")
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "same")

    def fake(system, user, schema, **kw):                    # phantom-only failing verdicts
        return JudgeVerdict(reasoning="bad", score_10=2, understandable=False,
                            topic_identifiable=False, purpose_identifiable=False,
                            failure_reasons=[FailureReason(
                                kind="missing_prerequisite",
                                evidence_quote="the previous equation")])
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
                # legacy selector pinned: this test's fake llm returns JudgeVerdicts only
                "anchor_selector": "priority"}
    specs, _notes, rejections = assemble_clips(st, "", sents, "u", "v", settings, AnchorAdapter())
    cols = R._integrity_columns(specs, rejections)
    assert len(specs) > 0
    assert cols["n_shipped_flagged"] == len(specs)          # phantom-only verdicts ship flagged…
    assert cols["shipped_flagged_rate"] == 1.0
    assert cols["phantom_verdict_rate"] == 1.0              # …and every reason was a phantom
    assert cols["verified_kill"] == 0 and cols["unverified_kill"] == 0   # nothing was killed
