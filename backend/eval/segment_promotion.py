"""Fail-closed promotion math for the Flash-first segment router.

Inputs are video-level trial rows produced elsewhere.  Human usability, severe
hallucination, and whole-video preference are explicit fields: this module never derives
those judgments from 1-5 ratings or model confidence.  Paired confidence bounds cluster
by the 25 video-query pairs, keeping each pair's three repeats together.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Iterable

from .segment_benchmark import (
    EXPECTED_PAIR_COUNT,
    EXPECTED_REPEATS,
    PROFILE_CORRECTED_PRO,
    PROFILE_FLASH_SINGLE,
    PROFILE_FLASH_SPLIT,
    PROFILE_PRODUCTION_PRO,
)


HUMAN_USABLE_MARGIN = -0.05
MECHANICAL_USABLE_MARGIN = -0.05
MEAN_HUMAN_SCORE_MARGIN = -0.25
ASSESSMENT_MIN = 0.95
ASSESSMENT_MARGIN = -0.03
STRATUM_MARGIN = -0.10
WHOLE_VIDEO_BETTER_OR_TIED_MIN = 0.65
GREEN_FLASH_P50_MAX_RATIO = 0.80
GREEN_FLASH_P95_MAX_RATIO = 1.00
HYBRID_P50_MAX_RATIO = 1.00
HYBRID_P95_MAX_RATIO = 1.10
COMPLETED_CALLS_MIN = 0.99
FALLBACK_RATE_MAX = 0.20
COST_PER_USABLE_MAX_RATIO = 0.85
ROLLBACK_HARD_FAILURE_MAX = 0.02
ROLLBACK_ZERO_OUTPUT_DELTA_MAX = 0.05
ROLLBACK_FALLBACK_MAX = 0.25
ROLLBACK_HUMAN_MARGIN = -0.05
ROLLBACK_STRATUM_MARGIN = -0.10
ROLLBACK_P95_RATIO_MAX = 1.25
ROLLOUT_REQUIREMENTS = {
    "shadow": {"generations": 300, "days": 0},
    "canary_1": {"generations": 200, "days": 7},
    "percent_5": {"generations": 500, "days": 7},
    "percent_25": {"generations": 1_000, "days": 7},
    "percent_50": {"generations": 2_000, "days": 7},
    "percent_100": {"generations": 0, "days": 0},
}
POSITIVE_CONTENT_STRATA = (
    "lecture", "speech", "interview", "podcast", "auto_caption",
)
_EPSILON = 1e-12
HISTORICAL_PRODUCTION_CONTROL = "historical_frozen_transport"

PROMOTION_GATE_NAMES = (
    "human_usable_noninferiority",
    "mechanical_usable_noninferiority",
    "mean_human_score_noninferiority",
    "assessment_correctness_absolute",
    "assessment_correctness_relative",
    "no_severe_hallucinations",
    "positive_strata_noninferiority",
    "whole_video_better_or_tied",
    "green_flash_p50_latency",
    "green_flash_p95_latency",
    "hybrid_p50_latency",
    "hybrid_p95_latency",
    "completed_calls",
    "fallback_rate",
    "cost_per_human_usable_clip",
)

_COMMON_FIELDS = (
    "dataset_version", "postprocess_version", "video_id", "topic",
    "transport_provenance", "profile", "pair_id", "repeat", "stratum", "positive_content",
    "human_usable_rate", "mechanical_usable_rate", "mean_human_score",
    "assessment_correct_count", "assessment_total", "severe_hallucination",
    "latency_ms", "cost_usd", "human_usable_clips",
)
_CANDIDATE_FIELDS = (
    "whole_video_comparison", "fallback_used", "completed_calls", "total_calls",
    "flash_profile", "flash_classification", "flash_latency_ms",
)


def _number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(float(value))


def _at_least(value: float, threshold: float) -> bool:
    return float(value) >= float(threshold) - _EPSILON


def _at_most(value: float, threshold: float) -> bool:
    return float(value) <= float(threshold) + _EPSILON


def percentile(values: Iterable[float], q: float) -> float:
    """Linear-interpolated percentile, equivalent to the common (n-1)*q definition."""
    vals = sorted(float(v) for v in values)
    if not vals:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= q <= 1.0:
        raise ValueError("percentile q must be in [0, 1]")
    rank = (len(vals) - 1) * q
    lo, hi = math.floor(rank), math.ceil(rank)
    if lo == hi:
        return vals[lo]
    return vals[lo] + (vals[hi] - vals[lo]) * (rank - lo)


def _paired_pair_means(candidate_by_key: dict, baseline_by_key: dict,
                       field: str, keys: set[tuple] | None = None) -> list[float]:
    selected = keys if keys is not None else set(candidate_by_key)
    by_pair: dict[str, list[float]] = {}
    for key in sorted(selected):
        pair_id, _repeat = key
        delta = float(candidate_by_key[key][field]) - float(baseline_by_key[key][field])
        by_pair.setdefault(pair_id, []).append(delta)
    return [sum(values) / len(values) for _pair, values in sorted(by_pair.items())]


def paired_cluster_bootstrap(candidate_by_key: dict, baseline_by_key: dict, field: str, *,
                             keys: set[tuple] | None = None, n_resamples: int = 10_000,
                             seed: int = 0, alpha: float = 0.05) -> dict:
    """Paired delta and percentile CI, bootstrapping video-query pair means."""
    pair_means = _paired_pair_means(candidate_by_key, baseline_by_key, field, keys)
    if not pair_means:
        raise ValueError("paired bootstrap has no pair clusters")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    rng = random.Random(seed)
    n = len(pair_means)
    draws = [
        sum(pair_means[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(n_resamples)
    ]
    return {
        "delta": sum(pair_means) / n,
        "lower": percentile(draws, alpha / 2.0),
        "upper": percentile(draws, 1.0 - alpha / 2.0),
        "n_pairs": n,
        "n_resamples": n_resamples,
    }


def _validate_value(row: dict, field: str, prefix: str, errors: list[str]) -> None:
    value = row.get(field)
    if field in {"positive_content", "severe_hallucination", "fallback_used"}:
        if not isinstance(value, bool):
            errors.append(f"{prefix}.{field} must be boolean")
    elif field in {"repeat", "assessment_correct_count", "assessment_total",
                   "human_usable_clips", "completed_calls", "total_calls"}:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            errors.append(f"{prefix}.{field} must be a non-negative integer")
    elif field in {"human_usable_rate", "mechanical_usable_rate"}:
        if not _number(value) or not 0.0 <= float(value) <= 1.0:
            errors.append(f"{prefix}.{field} must be in [0, 1]")
    elif field == "mean_human_score":
        if not _number(value) or not 1.0 <= float(value) <= 5.0:
            errors.append(f"{prefix}.{field} must be in [1, 5]")
    elif field in {"latency_ms", "flash_latency_ms", "cost_usd"}:
        if not _number(value) or float(value) < 0.0:
            errors.append(f"{prefix}.{field} must be non-negative")
    elif field == "whole_video_comparison":
        if value not in {"better", "tied", "worse"}:
            errors.append(f"{prefix}.{field} must be better|tied|worse")
    elif field == "flash_classification":
        if value not in {"green", "uncertain", "invalid"}:
            errors.append(f"{prefix}.{field} must be green|uncertain|invalid")
    elif not isinstance(value, str) or not value.strip():
        errors.append(f"{prefix}.{field} is required")


def _index_rows(rows: Iterable[dict], label: str, *, candidate: bool,
                expected_pairs: int, expected_repeats: int) -> tuple[dict, list[str]]:
    indexed: dict[tuple[str, int], dict] = {}
    errors: list[str] = []
    required = _COMMON_FIELDS + (_CANDIDATE_FIELDS if candidate else ())
    for index, row in enumerate(rows):
        prefix = f"{label}[{index}]"
        if not isinstance(row, dict):
            errors.append(f"{prefix} must be an object")
            continue
        missing = [field for field in required if field not in row]
        errors.extend(f"{prefix}.{field} is required" for field in missing)
        for field in required:
            if field in row:
                _validate_value(row, field, prefix, errors)
        if all(field in row for field in ("pair_id", "repeat")):
            key = (str(row["pair_id"]), row["repeat"])
            if key in indexed:
                errors.append(f"duplicate {label} row {key}")
            indexed[key] = row
        if all(field in row for field in ("assessment_correct_count", "assessment_total")):
            if (isinstance(row["assessment_correct_count"], int)
                    and isinstance(row["assessment_total"], int)
                    and row["assessment_correct_count"] > row["assessment_total"]):
                errors.append(f"{prefix}.assessment_correct_count exceeds assessment_total")
        if candidate and all(field in row for field in ("completed_calls", "total_calls")):
            if (isinstance(row["completed_calls"], int) and isinstance(row["total_calls"], int)
                    and (row["total_calls"] < 1
                         or row["completed_calls"] > row["total_calls"])):
                errors.append(f"{prefix} has invalid completed_calls/total_calls")

    pairs = {key[0] for key in indexed}
    if len(pairs) != expected_pairs:
        errors.append(f"{label} expected {expected_pairs} pair clusters, found {len(pairs)}")
    expected_repeat_set = set(range(1, expected_repeats + 1))
    for pair_id in sorted(pairs):
        repeats = {repeat for pair, repeat in indexed if pair == pair_id}
        if repeats != expected_repeat_set:
            errors.append(
                f"{label} pair {pair_id!r} repeats {sorted(repeats)} != "
                f"{sorted(expected_repeat_set)}"
            )
    return indexed, errors


def _gate(name: str, *, passed: bool, value=None, threshold=None, detail: str = "") -> dict:
    return {"name": name, "passed": bool(passed), "value": value,
            "threshold": threshold, "detail": detail}


def _missing_gates(reason: str) -> list[dict]:
    return [_gate(name, passed=False, detail=reason) for name in PROMOTION_GATE_NAMES]


def select_pro_baseline(production_profile: str, corrected_profile: str,
                        comparison: dict) -> dict:
    """Select corrected Pro only from an explicit human quality delta and regression flag.

    `quality_delta` is corrected minus production on the externally chosen quality measure.
    The measure is intentionally not inferred here because the evaluation plan does not define it.
    """
    errors: list[str] = []
    delta = comparison.get("quality_delta") if isinstance(comparison, dict) else None
    severe = comparison.get("severe_regression") if isinstance(comparison, dict) else None
    control_source = (
        comparison.get("production_control_source") if isinstance(comparison, dict) else None)
    dataset_version = comparison.get("dataset_version") if isinstance(comparison, dict) else None
    postprocess_version = (
        comparison.get("postprocess_version") if isinstance(comparison, dict) else None)
    if not _number(delta):
        errors.append("comparison.quality_delta is required and must be numeric")
    if not isinstance(severe, bool):
        errors.append("comparison.severe_regression is required and must be boolean")
    if control_source != HISTORICAL_PRODUCTION_CONTROL:
        errors.append(
            "comparison.production_control_source must be historical_frozen_transport")
    if not isinstance(dataset_version, str) or not dataset_version.strip():
        errors.append("comparison.dataset_version is required")
    if not isinstance(postprocess_version, str) or not postprocess_version.strip():
        errors.append("comparison.postprocess_version is required")
    corrected_ok = not errors and _at_least(float(delta), -0.02) and severe is False
    return {
        "selected_profile": corrected_profile if corrected_ok else production_profile,
        "corrected_selected": corrected_ok,
        "eligible": not errors,
        "quality_delta": float(delta) if _number(delta) else None,
        "threshold": -0.02,
        "severe_regression": severe if isinstance(severe, bool) else None,
        "production_control_source": control_source,
        "dataset_version": dataset_version,
        "postprocess_version": postprocess_version,
        "errors": errors,
    }


def select_evaluation_profiles(pro_comparison: dict, flash_single_report: dict,
                               flash_split_report: dict) -> dict:
    """Choose the eligible Pro baseline and the first Flash profile that clears every gate."""
    pro = select_pro_baseline(
        PROFILE_PRODUCTION_PRO, PROFILE_CORRECTED_PRO, pro_comparison)

    def passed(report: object, expected_flash: str) -> bool:
        if not isinstance(report, dict) or report.get("status") != "pass" \
                or report.get("promote") is not True:
            return False
        provenance = report.get("provenance")
        return isinstance(provenance, dict) \
            and provenance.get("dataset_version") == pro.get("dataset_version") \
            and provenance.get("postprocess_version") == pro.get("postprocess_version") \
            and provenance.get("flash_profile") == expected_flash \
            and provenance.get("baseline_profile") == pro.get("selected_profile")

    selected_flash = (
        PROFILE_FLASH_SINGLE if passed(flash_single_report, PROFILE_FLASH_SINGLE)
        else PROFILE_FLASH_SPLIT if passed(flash_split_report, PROFILE_FLASH_SPLIT)
        else None
    )
    errors = list(pro["errors"])
    if selected_flash is None:
        errors.append("neither Flash profile cleared every promotion gate")
    return {
        "eligible": not errors,
        "selected_pro_profile": pro["selected_profile"] if pro["eligible"] else None,
        "selected_flash_profile": selected_flash,
        "pro_selection": pro,
        "flash_single_passed": passed(flash_single_report, PROFILE_FLASH_SINGLE),
        "flash_split_passed": passed(flash_split_report, PROFILE_FLASH_SPLIT),
        "errors": errors,
    }


def build_promotion_rows(benchmark_rows: Iterable[dict], resolved_clip_reviews: Iterable[dict],
                         resolved_whole_video_reviews: Iterable[dict], *,
                         candidate_profile: str, baseline_profile: str
                         ) -> tuple[list[dict], list[dict]]:
    """Join frozen benchmark and resolved human records into the promotion row contract.

    Every selected clip and every candidate trial must have exactly one resolved review.
    Empty successful negative controls score as usable; empty positive/error trials receive
    the fail-closed minimum score instead of disappearing from the trial matrix.
    """
    if not candidate_profile or not baseline_profile:
        raise ValueError("candidate_profile and baseline_profile are required")
    if candidate_profile == baseline_profile:
        raise ValueError("candidate and baseline profiles must differ")

    required_benchmark = (
        "dataset_version", "postprocess_version", "video_id", "topic", "transport_provenance",
        "pair_id", "repeat", "profile", "stratum", "positive_content", "status", "clips",
        "proposed_count", "accepted_count", "latency_ms", "cost_usd", "calls",
    )
    selected: dict[tuple[str, str, int], dict] = {}
    for index, row in enumerate(benchmark_rows):
        if not isinstance(row, dict):
            raise ValueError(f"benchmark_rows[{index}] must be an object")
        if row.get("profile") not in {candidate_profile, baseline_profile}:
            continue
        missing = [field for field in required_benchmark if field not in row]
        if missing:
            raise ValueError(f"benchmark_rows[{index}] missing fields: {missing}")
        if not isinstance(row["clips"], list) or not isinstance(row["calls"], list):
            raise ValueError(f"benchmark_rows[{index}] clips/calls must be lists")
        key = (str(row["profile"]), str(row["pair_id"]), row["repeat"])
        if key in selected:
            raise ValueError(f"duplicate selected benchmark row {key}")
        selected[key] = row
    candidate_keys = {(pair, repeat) for profile, pair, repeat in selected
                      if profile == candidate_profile}
    baseline_keys = {(pair, repeat) for profile, pair, repeat in selected
                     if profile == baseline_profile}
    if not candidate_keys or candidate_keys != baseline_keys:
        raise ValueError("selected candidate/baseline trial keys must match exactly")

    provenance_fields = ("dataset_version", "postprocess_version", "video_id", "topic")
    for field in ("dataset_version", "postprocess_version"):
        if len({row[field] for row in selected.values()}) != 1:
            raise ValueError(f"selected benchmark rows must use one {field}")
    for pair_id, repeat in sorted(candidate_keys):
        candidate = selected[(candidate_profile, pair_id, repeat)]
        baseline = selected[(baseline_profile, pair_id, repeat)]
        for field in (*provenance_fields, "stratum", "positive_content"):
            if candidate[field] != baseline[field]:
                raise ValueError(f"paired benchmark rows {(pair_id, repeat)} disagree on {field}")

    expected_clips: dict[tuple[str, str, int, str], tuple[dict, dict]] = {}
    for (profile, pair_id, repeat), row in selected.items():
        seen_clip_ids: set[str] = set()
        for clip_index, clip in enumerate(row["clips"]):
            if not isinstance(clip, dict):
                raise ValueError(f"benchmark clip {(profile, pair_id, repeat, clip_index)} invalid")
            clip_id = str(clip.get("clip_id") or "").strip()
            if not clip_id or clip_id in seen_clip_ids:
                raise ValueError(f"benchmark row {(profile, pair_id, repeat)} has invalid clip IDs")
            seen_clip_ids.add(clip_id)
            expected_clips[(profile, pair_id, repeat, clip_id)] = (row, clip)

    reviews: dict[tuple[str, str, int, str], dict] = {}
    required_review = (
        *provenance_fields, "profile", "pair_id", "repeat", "clip_id", "scores",
        "mean_human_score", "human_usable", "assessment_correct", "severe_hallucination",
        "reviewer_count", "resolution_validated",
    )
    for index, review in enumerate(resolved_clip_reviews):
        if not isinstance(review, dict):
            raise ValueError(f"resolved_clip_reviews[{index}] must be an object")
        if review.get("profile") not in {candidate_profile, baseline_profile}:
            continue
        missing = [field for field in required_review if field not in review]
        if missing:
            raise ValueError(f"resolved_clip_reviews[{index}] missing fields: {missing}")
        key = (str(review["profile"]), str(review["pair_id"]), review["repeat"],
               str(review["clip_id"]))
        if key in reviews:
            raise ValueError(f"duplicate resolved clip review {key}")
        reviews[key] = review
    if set(reviews) != set(expected_clips):
        missing = set(expected_clips) - set(reviews)
        extra = set(reviews) - set(expected_clips)
        raise ValueError(
            f"resolved clip review coverage mismatch: missing={len(missing)} extra={len(extra)}")
    for key, review in reviews.items():
        row, _clip = expected_clips[key]
        for field in provenance_fields:
            if review[field] != row[field]:
                raise ValueError(f"resolved clip review {key} disagrees on {field}")
        if (isinstance(review["reviewer_count"], bool)
                or not isinstance(review["reviewer_count"], int)
                or review["reviewer_count"] < 2):
            raise ValueError(f"resolved clip review {key} requires two reviewers")
        if review["resolution_validated"] is not True:
            raise ValueError(f"resolved clip review {key} was not validated")
        if not _number(review["mean_human_score"]) \
                or not 1 <= float(review["mean_human_score"]) <= 5:
            raise ValueError(f"resolved clip review {key} has invalid mean_human_score")
        for field in ("human_usable", "assessment_correct", "severe_hallucination"):
            if not isinstance(review[field], bool):
                raise ValueError(f"resolved clip review {key}.{field} must be boolean")

    whole: dict[tuple[str, int], dict] = {}
    required_whole = (
        *provenance_fields, "pair_id", "repeat", "candidate_profile",
        "selected_pro_profile", "whole_video_comparison", "reviewer_count",
        "resolution_validated",
    )
    for index, review in enumerate(resolved_whole_video_reviews):
        if not isinstance(review, dict):
            raise ValueError(f"resolved_whole_video_reviews[{index}] must be an object")
        missing = [field for field in required_whole if field not in review]
        if missing:
            raise ValueError(f"resolved_whole_video_reviews[{index}] missing fields: {missing}")
        if (review["candidate_profile"] != candidate_profile
                or review["selected_pro_profile"] != baseline_profile):
            raise ValueError("whole-video review profiles do not match selected profiles")
        key = (str(review["pair_id"]), review["repeat"])
        if key in whole:
            raise ValueError(f"duplicate resolved whole-video verdict {key}")
        whole[key] = review
    if set(whole) != candidate_keys:
        raise ValueError("resolved whole-video verdicts must cover every candidate trial exactly")
    for key, review in whole.items():
        row = selected[(candidate_profile, *key)]
        for field in provenance_fields:
            if review[field] != row[field]:
                raise ValueError(f"resolved whole-video verdict {key} disagrees on {field}")
        if (isinstance(review["reviewer_count"], bool)
                or not isinstance(review["reviewer_count"], int)
                or review["reviewer_count"] < 2):
            raise ValueError(f"resolved whole-video verdict {key} requires two reviewers")
        if review["resolution_validated"] is not True:
            raise ValueError(f"resolved whole-video verdict {key} was not validated")
        if review["whole_video_comparison"] not in {"better", "tied", "worse"}:
            raise ValueError(f"resolved whole-video verdict {key} is invalid")

    def aggregate(row: dict, *, candidate: bool) -> dict:
        pair_id, repeat, profile = str(row["pair_id"]), row["repeat"], str(row["profile"])
        clip_reviews = [reviews[(profile, pair_id, repeat, str(clip["clip_id"]))]
                        for clip in row["clips"]]
        failed = row["status"] == "error"
        if clip_reviews:
            human_rate = sum(review["human_usable"] for review in clip_reviews) / len(clip_reviews)
            mean_score = sum(float(review["mean_human_score"])
                             for review in clip_reviews) / len(clip_reviews)
        else:
            correct_empty = not row["positive_content"] and not failed
            human_rate = 1.0 if correct_empty else 0.0
            mean_score = 5.0 if correct_empty else 1.0
        proposed, accepted = row["proposed_count"], row["accepted_count"]
        if (isinstance(proposed, bool) or not isinstance(proposed, int) or proposed < 0
                or isinstance(accepted, bool) or not isinstance(accepted, int) or accepted < 0
                or accepted > proposed):
            raise ValueError(f"benchmark row {(profile, pair_id, repeat)} has invalid clip counts")
        mechanical_rate = (0.0 if failed else accepted / proposed if proposed
                           else 1.0 if not row["positive_content"] else 0.0)
        if not _number(row["latency_ms"]) or float(row["latency_ms"]) < 0:
            raise ValueError(f"benchmark row {(profile, pair_id, repeat)} has invalid latency")
        if not _number(row["cost_usd"]) or float(row["cost_usd"]) < 0:
            raise ValueError(f"benchmark row {(profile, pair_id, repeat)} has unknown cost")
        assessment_reviews = [
            review for review, clip in zip(clip_reviews, row["clips"])
            if isinstance(clip.get("assessment"), dict)
        ]
        total_calls = max(1, len(row["calls"]))
        result = {
            **{field: row[field] for field in provenance_fields},
            "transport_provenance": row["transport_provenance"],
            "profile": profile,
            "pair_id": pair_id,
            "repeat": repeat,
            "stratum": row["stratum"],
            "positive_content": row["positive_content"],
            "human_usable_rate": human_rate,
            "mechanical_usable_rate": mechanical_rate,
            "mean_human_score": mean_score,
            "assessment_correct_count": sum(
                review["assessment_correct"] for review in assessment_reviews),
            "assessment_total": len(assessment_reviews),
            "severe_hallucination": any(
                review["severe_hallucination"] for review in clip_reviews),
            "latency_ms": float(row["latency_ms"]),
            "cost_usd": float(row["cost_usd"]),
            "human_usable_clips": sum(review["human_usable"] for review in clip_reviews),
        }
        if candidate:
            flash_latency = row.get("flash_latency_ms")
            if not _number(flash_latency) or float(flash_latency) < 0:
                raise ValueError(f"candidate row {(pair_id, repeat)} has invalid flash latency")
            flash_profile = str(row.get("flash_profile") or "").strip()
            if not flash_profile:
                raise ValueError(f"candidate row {(pair_id, repeat)} has no Flash profile")
            result.update({
                "whole_video_comparison": whole[(pair_id, repeat)]["whole_video_comparison"],
                "fallback_used": row.get("fallback_used"),
                "flash_profile": flash_profile,
                "completed_calls": total_calls if not failed else total_calls - 1,
                "total_calls": total_calls,
                "flash_classification": row.get("classification"),
                "flash_latency_ms": float(flash_latency),
            })
        return result

    candidate_rows = [aggregate(selected[(candidate_profile, pair, repeat)], candidate=True)
                      for pair, repeat in sorted(candidate_keys)]
    baseline_rows = [aggregate(selected[(baseline_profile, pair, repeat)], candidate=False)
                     for pair, repeat in sorted(baseline_keys)]
    return candidate_rows, baseline_rows


def evaluate_promotion(candidate_rows: Iterable[dict], baseline_rows: Iterable[dict], *,
                       expected_pairs: int = EXPECTED_PAIR_COUNT,
                       expected_repeats: int = EXPECTED_REPEATS,
                       n_resamples: int = 10_000, seed: int = 0) -> dict:
    """Evaluate every promotion threshold and return a JSON-serializable report.

    Missing rows, explicit human decisions, or reliability fields produce
    ``status=insufficient_data`` and a non-promotable result; no defaults are imputed.
    """
    candidate, errors = _index_rows(
        list(candidate_rows), "candidate_rows", candidate=True,
        expected_pairs=expected_pairs, expected_repeats=expected_repeats,
    )
    baseline, baseline_errors = _index_rows(
        list(baseline_rows), "baseline_rows", candidate=False,
        expected_pairs=expected_pairs, expected_repeats=expected_repeats,
    )
    errors.extend(baseline_errors)
    if set(candidate) != set(baseline):
        errors.append("candidate/baseline trial keys do not match exactly")
    for key in sorted(set(candidate) & set(baseline)):
        for field in (
            "dataset_version", "postprocess_version", "video_id", "topic",
            "stratum", "positive_content",
        ):
            if candidate[key].get(field) != baseline[key].get(field):
                errors.append(f"paired rows {key} disagree on {field}")
    candidate_profiles = {row.get("profile") for row in candidate.values()}
    baseline_profiles = {row.get("profile") for row in baseline.values()}
    for field in ("dataset_version", "postprocess_version"):
        values = {row.get(field) for row in (*candidate.values(), *baseline.values())}
        if len(values) != 1:
            errors.append(f"candidate/baseline rows must use one {field}")
    if len(candidate_profiles) != 1:
        errors.append("candidate_rows must use one homogeneous profile")
    if len(baseline_profiles) != 1:
        errors.append("baseline_rows must use one homogeneous profile")
    if candidate_profiles and candidate_profiles == baseline_profiles:
        errors.append("candidate and baseline profiles must differ")
    if PROFILE_PRODUCTION_PRO in baseline_profiles and any(
            row.get("transport_provenance") != HISTORICAL_PRODUCTION_CONTROL
            for row in baseline.values()):
        errors.append(
            "production_pro_v0 baseline requires historical_frozen_transport provenance")
    if errors:
        return {"schema_version": 1, "status": "insufficient_data", "promote": False,
                "errors": errors, "metrics": {}, "provenance": {},
                "gates": _missing_gates("required benchmark/review data is missing or invalid")}

    flash_profiles = {row["flash_profile"] for row in candidate.values()}
    if len(flash_profiles) != 1:
        errors.append("candidate_rows must use one homogeneous Flash profile")

    human = paired_cluster_bootstrap(
        candidate, baseline, "human_usable_rate", n_resamples=n_resamples, seed=seed)
    mechanical = paired_cluster_bootstrap(
        candidate, baseline, "mechanical_usable_rate", n_resamples=n_resamples, seed=seed + 1)
    human_score = paired_cluster_bootstrap(
        candidate, baseline, "mean_human_score", n_resamples=n_resamples, seed=seed + 2)

    cand_correct = sum(row["assessment_correct_count"] for row in candidate.values())
    cand_assessments = sum(row["assessment_total"] for row in candidate.values())
    base_correct = sum(row["assessment_correct_count"] for row in baseline.values())
    base_assessments = sum(row["assessment_total"] for row in baseline.values())
    if cand_assessments == 0 or base_assessments == 0:
        errors.append("assessment correctness requires at least one assessment in both profiles")

    assessment_rate = cand_correct / cand_assessments if cand_assessments else None
    base_assessment_rate = base_correct / base_assessments if base_assessments else None
    assessment_delta = (
        assessment_rate - base_assessment_rate
        if assessment_rate is not None and base_assessment_rate is not None else None
    )

    severe_count = sum(1 for row in candidate.values() if row["severe_hallucination"])
    positive_strata = sorted({
        row["stratum"] for row in candidate.values() if row["positive_content"]
    })
    stratum_deltas: dict[str, float] = {}
    for stratum in positive_strata:
        keys = {key for key, row in candidate.items()
                if row["positive_content"] and row["stratum"] == stratum}
        pair_means = _paired_pair_means(candidate, baseline, "human_usable_rate", keys)
        if pair_means:
            stratum_deltas[stratum] = sum(pair_means) / len(pair_means)
    if not stratum_deltas:
        errors.append("no explicit positive-content stratum comparisons")

    whole_rate = sum(
        1 for row in candidate.values()
        if row["whole_video_comparison"] in {"better", "tied"}
    ) / len(candidate)

    green_keys = {key for key, row in candidate.items()
                  if row["flash_classification"] == "green"}
    if not green_keys:
        errors.append("no green Flash trials for latency gates")
        green_flash_p50 = green_flash_p95 = green_pro_p50 = green_pro_p95 = None
    else:
        flash_latencies = [candidate[key]["flash_latency_ms"] for key in green_keys]
        green_pro_latencies = [baseline[key]["latency_ms"] for key in green_keys]
        green_flash_p50 = percentile(flash_latencies, 0.50)
        green_flash_p95 = percentile(flash_latencies, 0.95)
        green_pro_p50 = percentile(green_pro_latencies, 0.50)
        green_pro_p95 = percentile(green_pro_latencies, 0.95)

    hybrid_latencies = [row["latency_ms"] for row in candidate.values()]
    pro_latencies = [row["latency_ms"] for row in baseline.values()]
    hybrid_p50, hybrid_p95 = percentile(hybrid_latencies, 0.50), percentile(hybrid_latencies, 0.95)
    pro_p50, pro_p95 = percentile(pro_latencies, 0.50), percentile(pro_latencies, 0.95)

    completed = sum(row["completed_calls"] for row in candidate.values())
    total_calls = sum(row["total_calls"] for row in candidate.values())
    completed_rate = completed / total_calls
    fallback_rate = sum(1 for row in candidate.values() if row["fallback_used"]) / len(candidate)

    cand_cost = sum(float(row["cost_usd"]) for row in candidate.values())
    base_cost = sum(float(row["cost_usd"]) for row in baseline.values())
    cand_usable = sum(row["human_usable_clips"] for row in candidate.values())
    base_usable = sum(row["human_usable_clips"] for row in baseline.values())
    if cand_usable == 0 or base_usable == 0:
        errors.append("cost-per-usable requires explicit nonzero human_usable_clips")
        cand_cost_per = base_cost_per = cost_ratio = None
    else:
        cand_cost_per = cand_cost / cand_usable
        base_cost_per = base_cost / base_usable
        cost_ratio = cand_cost_per / base_cost_per if base_cost_per > 0 else None
        if cost_ratio is None:
            errors.append("baseline cost per human-usable clip must be positive")

    metrics = {
        "human_usable": human,
        "mechanical_usable": mechanical,
        "mean_human_score": human_score,
        "assessment_correctness": {"candidate": assessment_rate,
                                   "baseline": base_assessment_rate,
                                   "delta": assessment_delta},
        "severe_hallucinations": severe_count,
        "positive_stratum_human_usable_deltas": stratum_deltas,
        "whole_video_better_or_tied_rate": whole_rate,
        "latency_ms": {
            "green_flash_p50": green_flash_p50, "green_flash_p95": green_flash_p95,
            "green_pro_p50": green_pro_p50, "green_pro_p95": green_pro_p95,
            "hybrid_p50": hybrid_p50, "hybrid_p95": hybrid_p95,
            "pro_p50": pro_p50, "pro_p95": pro_p95,
        },
        "completed_call_rate": completed_rate,
        "fallback_rate": fallback_rate,
        "cost_per_human_usable_clip": {"candidate": cand_cost_per,
                                       "baseline": base_cost_per, "ratio": cost_ratio},
    }

    gates = [
        _gate("human_usable_noninferiority",
              passed=human["lower"] > HUMAN_USABLE_MARGIN + _EPSILON,
              value=human["lower"], threshold=f"> {HUMAN_USABLE_MARGIN}"),
        _gate("mechanical_usable_noninferiority",
              passed=mechanical["lower"] > MECHANICAL_USABLE_MARGIN + _EPSILON,
              value=mechanical["lower"], threshold=f"> {MECHANICAL_USABLE_MARGIN}"),
        _gate("mean_human_score_noninferiority",
              passed=human_score["lower"] > MEAN_HUMAN_SCORE_MARGIN + _EPSILON,
              value=human_score["lower"], threshold=f"> {MEAN_HUMAN_SCORE_MARGIN}"),
        _gate("assessment_correctness_absolute",
              passed=assessment_rate is not None and _at_least(assessment_rate, ASSESSMENT_MIN),
              value=assessment_rate, threshold=f">= {ASSESSMENT_MIN}"),
        _gate("assessment_correctness_relative",
              passed=assessment_delta is not None
              and _at_least(assessment_delta, ASSESSMENT_MARGIN),
              value=assessment_delta, threshold=f">= {ASSESSMENT_MARGIN}"),
        _gate("no_severe_hallucinations", passed=severe_count == 0,
              value=severe_count, threshold="== 0"),
        _gate("positive_strata_noninferiority",
              passed=bool(stratum_deltas)
              and _at_least(min(stratum_deltas.values()), STRATUM_MARGIN),
              value=min(stratum_deltas.values()) if stratum_deltas else None,
              threshold=f">= {STRATUM_MARGIN}", detail=json.dumps(stratum_deltas, sort_keys=True)),
        _gate("whole_video_better_or_tied",
              passed=_at_least(whole_rate, WHOLE_VIDEO_BETTER_OR_TIED_MIN),
              value=whole_rate, threshold=f">= {WHOLE_VIDEO_BETTER_OR_TIED_MIN}"),
        _gate("green_flash_p50_latency",
              passed=(green_flash_p50 is not None and green_pro_p50 is not None
                      and _at_most(green_flash_p50,
                                   GREEN_FLASH_P50_MAX_RATIO * green_pro_p50)),
              value=green_flash_p50,
              threshold=(None if green_pro_p50 is None
                         else f"<= {GREEN_FLASH_P50_MAX_RATIO} * {green_pro_p50}")),
        _gate("green_flash_p95_latency",
              passed=(green_flash_p95 is not None and green_pro_p95 is not None
                      and _at_most(green_flash_p95,
                                   GREEN_FLASH_P95_MAX_RATIO * green_pro_p95)),
              value=green_flash_p95,
              threshold=(None if green_pro_p95 is None
                         else f"<= {GREEN_FLASH_P95_MAX_RATIO} * {green_pro_p95}")),
        _gate("hybrid_p50_latency",
              passed=_at_most(hybrid_p50, HYBRID_P50_MAX_RATIO * pro_p50),
              value=hybrid_p50, threshold=f"<= {HYBRID_P50_MAX_RATIO} * {pro_p50}"),
        _gate("hybrid_p95_latency",
              passed=_at_most(hybrid_p95, HYBRID_P95_MAX_RATIO * pro_p95),
              value=hybrid_p95, threshold=f"<= {HYBRID_P95_MAX_RATIO} * {pro_p95}"),
        _gate("completed_calls", passed=_at_least(completed_rate, COMPLETED_CALLS_MIN),
              value=completed_rate, threshold=f">= {COMPLETED_CALLS_MIN}"),
        _gate("fallback_rate", passed=_at_most(fallback_rate, FALLBACK_RATE_MAX),
              value=fallback_rate, threshold=f"<= {FALLBACK_RATE_MAX}"),
        _gate("cost_per_human_usable_clip",
              passed=cost_ratio is not None and _at_most(cost_ratio, COST_PER_USABLE_MAX_RATIO),
              value=cost_ratio, threshold=f"<= {COST_PER_USABLE_MAX_RATIO}"),
    ]
    status = (
        "insufficient_data" if errors
        else "pass" if all(g["passed"] for g in gates)
        else "fail"
    )
    provenance = {
        "dataset_version": next(iter(candidate.values()))["dataset_version"],
        "postprocess_version": next(iter(candidate.values()))["postprocess_version"],
        "candidate_profile": next(iter(candidate_profiles)),
        "baseline_profile": next(iter(baseline_profiles)),
        "flash_profile": next(iter(flash_profiles)) if len(flash_profiles) == 1 else None,
    }
    return {"schema_version": 1, "status": status,
            "promote": not errors and all(g["passed"] for g in gates),
            "errors": errors, "metrics": metrics, "gates": gates,
            "provenance": provenance}


def write_promotion_report(report: dict, path: Path) -> Path:
    """Persist an already-computed report; callers choose the versioned result path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def evaluate_rollback(metrics: dict) -> dict:
    """Evaluate the deployment monitor's automatic return-to-Pro triggers.

    Window aggregation and the actual configuration flip belong to the deployment
    metrics/control plane.  This pure decision contract prevents each environment
    from reinterpreting thresholds and fails closed when a required signal is absent.
    """
    required_numbers = (
        "hard_failure_rate_15m",
        "zero_output_rate_24h",
        "pro_zero_output_rate_24h",
        "fallback_rate_24h",
        "human_usability_delta",
        "worst_positive_stratum_delta",
        "hybrid_p95_latency_ms_1h",
        "pro_p95_latency_ms_1h",
    )
    errors = [f"{field} is required and must be numeric"
              for field in required_numbers if not _number(metrics.get(field))]
    rate_fields = (
        "hard_failure_rate_15m", "zero_output_rate_24h", "pro_zero_output_rate_24h",
        "fallback_rate_24h",
    )
    for field in rate_fields:
        if _number(metrics.get(field)) and not 0.0 <= float(metrics[field]) <= 1.0:
            errors.append(f"{field} must be in [0, 1]")
    for field in ("human_usability_delta", "worst_positive_stratum_delta"):
        if _number(metrics.get(field)) and not -1.0 <= float(metrics[field]) <= 1.0:
            errors.append(f"{field} must be in [-1, 1]")
    for field in ("hybrid_p95_latency_ms_1h", "pro_p95_latency_ms_1h"):
        if _number(metrics.get(field)) and float(metrics[field]) < 0.0:
            errors.append(f"{field} must be non-negative")
    if not isinstance(metrics.get("model_access_or_config_failure"), bool):
        errors.append("model_access_or_config_failure is required and must be boolean")
    if errors:
        return {
            "schema_version": 1,
            "status": "insufficient_data",
            "rollback": True,
            "reasons": ["missing_required_monitoring_signal"],
            "errors": errors,
        }

    reasons: list[str] = []
    if float(metrics["hard_failure_rate_15m"]) > ROLLBACK_HARD_FAILURE_MAX:
        reasons.append("hard_failures_above_2_percent_15m")
    zero_delta = (
        float(metrics["zero_output_rate_24h"])
        - float(metrics["pro_zero_output_rate_24h"])
    )
    if zero_delta > ROLLBACK_ZERO_OUTPUT_DELTA_MAX:
        reasons.append("zero_output_above_pro_by_5pp_24h")
    if float(metrics["fallback_rate_24h"]) > ROLLBACK_FALLBACK_MAX:
        reasons.append("fallback_rate_above_25_percent_24h")
    if float(metrics["human_usability_delta"]) < ROLLBACK_HUMAN_MARGIN:
        reasons.append("human_usability_below_minus_5pp")
    if float(metrics["worst_positive_stratum_delta"]) < ROLLBACK_STRATUM_MARGIN:
        reasons.append("positive_stratum_below_minus_10pp")
    pro_p95 = float(metrics["pro_p95_latency_ms_1h"])
    hybrid_p95 = float(metrics["hybrid_p95_latency_ms_1h"])
    if pro_p95 <= 0:
        reasons.append("invalid_pro_p95_latency")
    elif hybrid_p95 > ROLLBACK_P95_RATIO_MAX * pro_p95:
        reasons.append("hybrid_p95_above_1_25x_pro_1h")
    if metrics["model_access_or_config_failure"]:
        reasons.append("model_access_or_configuration_failure")
    return {
        "schema_version": 1,
        "status": "rollback" if reasons else "healthy",
        "rollback": bool(reasons),
        "reasons": reasons,
        "errors": [],
        "zero_output_delta": zero_delta,
        "p95_latency_ratio": (hybrid_p95 / pro_p95 if pro_p95 > 0 else None),
    }


def evaluate_rollout_stage(stage: str, evidence: dict) -> dict:
    """Fail-closed evidence check before advancing one rollout stage."""
    requirement = ROLLOUT_REQUIREMENTS.get(stage)
    if requirement is None:
        raise ValueError(f"unknown rollout stage: {stage}")
    errors: list[str] = []
    generations = evidence.get("generations")
    days = evidence.get("days")
    gates_passed = evidence.get("quality_gates_passed")
    if isinstance(generations, bool) or not isinstance(generations, int) or generations < 0:
        errors.append("generations must be a non-negative integer")
    if not _number(days) or float(days) < 0:
        errors.append("days must be non-negative")
    if not isinstance(gates_passed, bool):
        errors.append("quality_gates_passed must be boolean")
    reasons: list[str] = []
    if not errors:
        if generations < requirement["generations"]:
            reasons.append("minimum_generations_not_met")
        if float(days) < requirement["days"]:
            reasons.append("minimum_days_not_met")
        if not gates_passed:
            reasons.append("quality_gates_not_met")

    if stage == "shadow":
        counts = evidence.get("positive_stratum_counts")
        if not isinstance(counts, dict):
            errors.append("positive_stratum_counts must be an object for shadow")
        else:
            missing = [name for name in POSITIVE_CONTENT_STRATA
                       if not isinstance(counts.get(name), int)
                       or isinstance(counts.get(name), bool) or counts[name] < 20]
            if missing:
                reasons.append("minimum_20_per_positive_stratum_not_met")
    if stage == "percent_100":
        sample = evidence.get("blind_sample_size")
        sample_passed = evidence.get("blind_sample_quality_gates_passed")
        if isinstance(sample, bool) or not isinstance(sample, int) or sample < 0:
            errors.append("blind_sample_size must be a non-negative integer")
        elif sample < 50:
            reasons.append("minimum_50_clip_blind_sample_not_met")
        if not isinstance(sample_passed, bool):
            errors.append("blind_sample_quality_gates_passed must be boolean")
        elif not sample_passed:
            reasons.append("blind_sample_quality_gates_not_met")
    eligible = not errors and not reasons
    return {
        "schema_version": 1,
        "stage": stage,
        "eligible": eligible,
        "status": "pass" if eligible else "insufficient_data" if errors else "hold",
        "requirements": requirement,
        "reasons": reasons,
        "errors": errors,
    }
