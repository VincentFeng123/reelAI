from __future__ import annotations

import pytest

from backend.eval import segment_promotion as P


STRATA = ("lecture", "speech", "interview", "podcast", "auto_caption", "comedy_negative")


def _rows():
    candidate, baseline = [], []
    for pair_index in range(25):
        stratum = STRATA[pair_index % len(STRATA)]
        positive = stratum != "comedy_negative"
        for repeat in range(1, 4):
            common = {
                "dataset_version": "segment-v1", "postprocess_version": "strict-v1",
                "video_id": f"video-{pair_index:02d}", "topic": f"topic {pair_index}",
                "transport_provenance": "in_process_gemini3_transport",
                "pair_id": f"pair-{pair_index:02d}", "repeat": repeat,
                "stratum": stratum, "positive_content": positive,
                "human_usable_rate": 0.90, "mechanical_usable_rate": 0.90,
                "mean_human_score": 4.5,
                "assessment_correct_count": 19, "assessment_total": 20,
                "severe_hallucination": False,
                "latency_ms": 1000.0, "cost_usd": 1.0, "human_usable_clips": 10,
            }
            baseline.append({**common, "profile": "corrected_pro_v1"})
            candidate.append({
                **common, "profile": "simulated_hybrid_v1",
                "latency_ms": 900.0, "cost_usd": 0.80,
                "whole_video_comparison": "better",
                "fallback_used": pair_index % 10 == 0 and repeat == 1,
                "completed_calls": 100, "total_calls": 100,
                "flash_profile": "flash_single_v1",
                "flash_classification": "green" if pair_index % 5 else "uncertain",
                "flash_latency_ms": 700.0,
            })
    return candidate, baseline


def _gate(report, name):
    return next(gate for gate in report["gates"] if gate["name"] == name)


def test_all_promotion_thresholds_pass_on_qualifying_rows():
    candidate, baseline = _rows()
    report = P.evaluate_promotion(candidate, baseline, n_resamples=300, seed=7)
    assert report["status"] == "pass"
    assert report["promote"] is True
    assert [gate["name"] for gate in report["gates"]] == list(P.PROMOTION_GATE_NAMES)
    assert all(gate["passed"] for gate in report["gates"])
    assert report["metrics"]["human_usable"]["n_pairs"] == 25


def test_missing_explicit_human_field_fails_closed_without_imputation():
    candidate, baseline = _rows()
    del candidate[0]["human_usable_rate"]
    report = P.evaluate_promotion(candidate, baseline, n_resamples=20)
    assert report["status"] == "insufficient_data"
    assert report["promote"] is False
    assert any("human_usable_rate is required" in error for error in report["errors"])
    assert all(not gate["passed"] for gate in report["gates"])


def test_missing_repeat_or_pair_cluster_fails_closed():
    candidate, baseline = _rows()
    candidate.pop()
    report = P.evaluate_promotion(candidate, baseline, n_resamples=20)
    assert report["status"] == "insufficient_data"
    assert any("repeats" in error or "keys do not match" in error for error in report["errors"])


@pytest.mark.parametrize(
    ("gate_name", "mutate"),
    [
        ("human_usable_noninferiority",
         lambda c, _b: [row.update(human_usable_rate=0.84) for row in c]),
        ("mechanical_usable_noninferiority",
         lambda c, _b: [row.update(mechanical_usable_rate=0.84) for row in c]),
        ("mean_human_score_noninferiority",
         lambda c, _b: [row.update(mean_human_score=4.1) for row in c]),
        ("assessment_correctness_absolute",
         lambda c, _b: [row.update(assessment_correct_count=18) for row in c]),
        ("assessment_correctness_relative",
         lambda c, b: ([row.update(assessment_correct_count=19) for row in c],
                       [row.update(assessment_correct_count=20) for row in b])),
        ("no_severe_hallucinations", lambda c, _b: c[0].update(severe_hallucination=True)),
        ("whole_video_better_or_tied",
         lambda c, _b: [row.update(whole_video_comparison="worse") for row in c]),
        ("green_flash_p50_latency",
         lambda c, _b: [row.update(flash_latency_ms=900.0) for row in c]),
        ("green_flash_p95_latency",
         lambda c, _b: [row.update(flash_latency_ms=1100.0) for row in c]),
        ("hybrid_p50_latency", lambda c, _b: [row.update(latency_ms=1001.0) for row in c]),
        ("hybrid_p95_latency", lambda c, _b: [row.update(latency_ms=1200.0) for row in c]),
        ("completed_calls",
         lambda c, _b: [row.update(completed_calls=98) for row in c]),
        ("fallback_rate", lambda c, _b: [row.update(fallback_used=True) for row in c]),
        ("cost_per_human_usable_clip", lambda c, _b: [row.update(cost_usd=0.90) for row in c]),
    ],
)
def test_each_threshold_can_block_promotion(gate_name, mutate):
    candidate, baseline = _rows()
    mutate(candidate, baseline)
    report = P.evaluate_promotion(candidate, baseline, n_resamples=100)
    assert _gate(report, gate_name)["passed"] is False
    assert report["promote"] is False


def test_positive_stratum_regression_gate_is_computed_from_explicit_strata():
    candidate, baseline = _rows()
    for row in candidate:
        if row["stratum"] == "lecture":
            row["human_usable_rate"] = 0.79
    report = P.evaluate_promotion(candidate, baseline, n_resamples=100)
    gate = _gate(report, "positive_strata_noninferiority")
    assert gate["passed"] is False
    assert report["metrics"]["positive_stratum_human_usable_deltas"]["lecture"] \
        == pytest.approx(-0.11)


def test_strict_confidence_margins_fail_at_equality():
    candidate, baseline = _rows()
    for row in candidate:
        row["human_usable_rate"] = 0.85
        row["mechanical_usable_rate"] = 0.85
        row["mean_human_score"] = 4.25
    report = P.evaluate_promotion(candidate, baseline, n_resamples=50)
    assert _gate(report, "human_usable_noninferiority")["passed"] is False
    assert _gate(report, "mechanical_usable_noninferiority")["passed"] is False
    assert _gate(report, "mean_human_score_noninferiority")["passed"] is False


def test_inclusive_thresholds_pass_at_their_boundaries():
    candidate, baseline = _rows()
    for row in baseline:
        row["assessment_correct_count"] = 98
        row["assessment_total"] = 100
    for index, row in enumerate(candidate):
        row["assessment_correct_count"] = 95
        row["assessment_total"] = 100
        row["flash_latency_ms"] = 800.0
        row["latency_ms"] = 1000.0
        row["completed_calls"] = 99
        row["total_calls"] = 100
        row["fallback_used"] = index < 15       # 15/75 = exactly 20%
        row["cost_usd"] = 0.85
        if row["stratum"] == "lecture":
            row["human_usable_rate"] = 0.80      # exactly -10pp in this stratum
    report = P.evaluate_promotion(candidate, baseline, n_resamples=50)
    for name in (
        "assessment_correctness_absolute", "assessment_correctness_relative",
        "positive_strata_noninferiority", "green_flash_p50_latency",
        "green_flash_p95_latency", "hybrid_p50_latency", "hybrid_p95_latency",
        "completed_calls", "fallback_rate", "cost_per_human_usable_clip",
    ):
        assert _gate(report, name)["passed"] is True, name


def test_cluster_bootstrap_uses_pair_means_not_individual_clips():
    candidate, baseline = _rows()
    c_index = {(row["pair_id"], row["repeat"]): row for row in candidate}
    b_index = {(row["pair_id"], row["repeat"]): row for row in baseline}
    for repeat in range(1, 4):
        c_index[("pair-00", repeat)]["human_usable_rate"] = 0.0
    result = P.paired_cluster_bootstrap(
        c_index, b_index, "human_usable_rate", n_resamples=200, seed=1)
    assert result["n_pairs"] == 25
    assert result["delta"] == pytest.approx(-0.9 / 25)


def test_baseline_selection_requires_explicit_quality_and_severe_regression_decisions():
    selected = P.select_pro_baseline(
        "production_pro_v0", "corrected_pro_v1",
        {"quality_delta": -0.02, "severe_regression": False,
         "production_control_source": P.HISTORICAL_PRODUCTION_CONTROL,
         "dataset_version": "segment-v1", "postprocess_version": "strict-v1"})
    assert selected["selected_profile"] == "corrected_pro_v1"
    assert selected["eligible"] is True

    retained = P.select_pro_baseline(
        "production_pro_v0", "corrected_pro_v1",
        {"quality_delta": -0.021, "severe_regression": False,
         "production_control_source": P.HISTORICAL_PRODUCTION_CONTROL,
         "dataset_version": "segment-v1", "postprocess_version": "strict-v1"})
    assert retained["selected_profile"] == "production_pro_v0"
    missing = P.select_pro_baseline("production_pro_v0", "corrected_pro_v1", {})
    assert missing["eligible"] is False
    assert missing["selected_profile"] == "production_pro_v0"

    live_control = P.select_pro_baseline(
        "production_pro_v0", "corrected_pro_v1",
        {"quality_delta": 0.0, "severe_regression": False,
         "production_control_source": "in_process_gemini3_transport",
         "dataset_version": "segment-v1", "postprocess_version": "strict-v1"})
    assert live_control["eligible"] is False
    assert any("historical_frozen_transport" in error for error in live_control["errors"])


def test_profile_selection_is_single_first_and_uses_eligible_pro_baseline():
    comparison = {
        "quality_delta": -0.01,
        "severe_regression": False,
        "production_control_source": P.HISTORICAL_PRODUCTION_CONTROL,
        "dataset_version": "segment-v1", "postprocess_version": "strict-v1",
    }
    def passed(flash_profile):
        return {"status": "pass", "promote": True, "provenance": {
            "dataset_version": "segment-v1", "postprocess_version": "strict-v1",
            "flash_profile": flash_profile, "baseline_profile": "corrected_pro_v1",
        }}
    failed = {"status": "fail", "promote": False}
    selection = P.select_evaluation_profiles(
        comparison, passed("flash_single_v1"), passed("flash_split_v2"))
    assert selection["selected_pro_profile"] == "corrected_pro_v1"
    assert selection["selected_flash_profile"] == "flash_single_v1"
    assert selection["eligible"] is True
    selection = P.select_evaluation_profiles(comparison, failed, passed("flash_split_v2"))
    assert selection["selected_flash_profile"] == "flash_split_v2"
    assert P.select_evaluation_profiles(comparison, failed, failed)["eligible"] is False


def test_promotion_provenance_requires_matching_identity_and_distinct_homogeneous_profiles():
    candidate, baseline = _rows()
    candidate[0]["dataset_version"] = "drifted"
    report = P.evaluate_promotion(candidate, baseline, n_resamples=20)
    assert report["status"] == "insufficient_data"
    assert any("dataset_version" in error for error in report["errors"])

    candidate, baseline = _rows()
    candidate[0]["profile"] = "another-profile"
    report = P.evaluate_promotion(candidate, baseline, n_resamples=20)
    assert any("homogeneous" in error for error in report["errors"])

    candidate, baseline = _rows()
    for row in candidate:
        row["profile"] = "corrected_pro_v1"
    report = P.evaluate_promotion(candidate, baseline, n_resamples=20)
    assert any("must differ" in error for error in report["errors"])

    candidate, baseline = _rows()
    for row in baseline:
        row["profile"] = "production_pro_v0"
    report = P.evaluate_promotion(candidate, baseline, n_resamples=20)
    assert any("historical_frozen_transport" in error for error in report["errors"])


def test_build_promotion_rows_joins_complete_resolved_review_coverage():
    provenance = {
        "dataset_version": "segment-v1", "postprocess_version": "strict-v1",
        "video_id": "video-1", "topic": "vectors",
        "transport_provenance": "in_process_gemini3_transport",
        "pair_id": "pair-1", "repeat": 1, "stratum": "lecture",
        "positive_content": True, "status": "ok", "proposed_count": 1,
        "accepted_count": 1, "latency_ms": 100.0, "cost_usd": 0.1,
        "calls": [{"model": "model"}],
    }
    candidate = {
        **provenance, "profile": "simulated_hybrid_v1", "classification": "green",
        "fallback_used": False, "flash_profile": "flash_single_v1", "flash_latency_ms": 70.0,
        "clips": [{"clip_id": "clip-001", "assessment": {"prompt": "q"}}],
    }
    baseline = {
        **provenance, "profile": "corrected_pro_v1",
        "clips": [{"clip_id": "clip-001", "assessment": {"prompt": "q"}}],
    }
    review_common = {
        "dataset_version": "segment-v1", "postprocess_version": "strict-v1",
        "video_id": "video-1", "topic": "vectors", "pair_id": "pair-1",
        "repeat": 1, "clip_id": "clip-001", "scores": {},
        "mean_human_score": 4.5, "human_usable": True, "assessment_correct": True,
        "severe_hallucination": False, "reviewer_count": 2,
        "resolution_validated": True,
    }
    clip_reviews = [
        {**review_common, "profile": "simulated_hybrid_v1"},
        {**review_common, "profile": "corrected_pro_v1"},
    ]
    whole = [{
        "dataset_version": "segment-v1", "postprocess_version": "strict-v1",
        "video_id": "video-1", "topic": "vectors", "pair_id": "pair-1", "repeat": 1,
        "candidate_profile": "simulated_hybrid_v1",
        "selected_pro_profile": "corrected_pro_v1",
        "whole_video_comparison": "better", "reviewer_count": 2,
        "resolution_validated": True,
    }]
    candidate_rows, baseline_rows = P.build_promotion_rows(
        [candidate, baseline], clip_reviews, whole,
        candidate_profile="simulated_hybrid_v1", baseline_profile="corrected_pro_v1")
    assert candidate_rows[0]["human_usable_rate"] == 1.0
    assert candidate_rows[0]["whole_video_comparison"] == "better"
    assert candidate_rows[0]["assessment_correct_count"] == 1
    assert baseline_rows[0]["mean_human_score"] == 4.5

    with pytest.raises(ValueError, match="coverage mismatch"):
        P.build_promotion_rows(
            [candidate, baseline], clip_reviews[:-1], whole,
            candidate_profile="simulated_hybrid_v1", baseline_profile="corrected_pro_v1")


def test_report_writer_is_json_serializable(tmp_path):
    candidate, baseline = _rows()
    report = P.evaluate_promotion(candidate, baseline, n_resamples=20)
    path = P.write_promotion_report(report, tmp_path / "reports" / "promotion.json")
    assert path.read_text().endswith("\n")


def _healthy_rollback_metrics():
    return {
        "hard_failure_rate_15m": 0.02,
        "zero_output_rate_24h": 0.10,
        "pro_zero_output_rate_24h": 0.05,
        "fallback_rate_24h": 0.25,
        "human_usability_delta": -0.05,
        "worst_positive_stratum_delta": -0.10,
        "hybrid_p95_latency_ms_1h": 1250.0,
        "pro_p95_latency_ms_1h": 1000.0,
        "model_access_or_config_failure": False,
    }


def test_rollback_threshold_boundaries_are_healthy():
    report = P.evaluate_rollback(_healthy_rollback_metrics())
    assert report["status"] == "healthy"
    assert report["rollback"] is False


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("hard_failure_rate_15m", 0.0201, "hard_failures_above_2_percent_15m"),
        ("zero_output_rate_24h", 0.1001, "zero_output_above_pro_by_5pp_24h"),
        ("fallback_rate_24h", 0.251, "fallback_rate_above_25_percent_24h"),
        ("human_usability_delta", -0.051, "human_usability_below_minus_5pp"),
        ("worst_positive_stratum_delta", -0.101, "positive_stratum_below_minus_10pp"),
        ("hybrid_p95_latency_ms_1h", 1250.1, "hybrid_p95_above_1_25x_pro_1h"),
        ("model_access_or_config_failure", True, "model_access_or_configuration_failure"),
    ],
)
def test_each_monitoring_trigger_requires_pro_only_rollback(field, value, reason):
    metrics = _healthy_rollback_metrics()
    metrics[field] = value
    report = P.evaluate_rollback(metrics)
    assert report["rollback"] is True
    assert reason in report["reasons"]


def test_missing_monitoring_signal_fails_closed_to_rollback():
    metrics = _healthy_rollback_metrics()
    metrics.pop("fallback_rate_24h")
    report = P.evaluate_rollback(metrics)
    assert report["status"] == "insufficient_data"
    assert report["rollback"] is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("hard_failure_rate_15m", 1.01),
        ("zero_output_rate_24h", -0.01),
        ("human_usability_delta", -1.01),
        ("hybrid_p95_latency_ms_1h", -1.0),
        ("pro_p95_latency_ms_1h", -1.0),
    ],
)
def test_out_of_domain_monitoring_values_fail_closed(field, value):
    metrics = _healthy_rollback_metrics()
    metrics[field] = value
    report = P.evaluate_rollback(metrics)
    assert report["status"] == "insufficient_data"
    assert report["rollback"] is True
    assert any(field in error for error in report["errors"])


@pytest.mark.parametrize(
    "stage,generations,days",
    [
        ("shadow", 300, 0),
        ("canary_1", 200, 7),
        ("percent_5", 500, 7),
        ("percent_25", 1_000, 7),
        ("percent_50", 2_000, 7),
    ],
)
def test_rollout_stage_minimums(stage, generations, days):
    evidence = {
        "generations": generations,
        "days": days,
        "quality_gates_passed": True,
    }
    if stage == "shadow":
        evidence["positive_stratum_counts"] = {
            stratum: 20 for stratum in P.POSITIVE_CONTENT_STRATA
        }
    assert P.evaluate_rollout_stage(stage, evidence)["eligible"] is True
    evidence["generations"] = max(0, generations - 1)
    assert P.evaluate_rollout_stage(stage, evidence)["eligible"] is False


def test_full_rollout_requires_new_blind_fifty_clip_sample():
    evidence = {
        "generations": 0,
        "days": 0,
        "quality_gates_passed": True,
        "blind_sample_size": 50,
        "blind_sample_quality_gates_passed": True,
    }
    assert P.evaluate_rollout_stage("percent_100", evidence)["eligible"] is True
    evidence["blind_sample_size"] = 49
    assert P.evaluate_rollout_stage("percent_100", evidence)["eligible"] is False
