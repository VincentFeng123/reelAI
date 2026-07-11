"""Pure blind-review manifest and reviewer-record helpers for segment evaluation."""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Iterable


REVIEW_DIMENSIONS = (
    "opening_completeness",
    "closing_completeness",
    "topic_relevance",
    "self_containedness",
    "educational_substance",
    "summary_takeaway_grounding",
    "assessment_correctness_ambiguity",
)

REVIEW_RUBRIC = {
    "scale": {"minimum": 1, "maximum": 5},
    "dimensions": {
        "opening_completeness": "The clip begins at a complete, understandable opening.",
        "closing_completeness": "The clip reaches a complete, resolved ending.",
        "topic_relevance": "The clip directly addresses the supplied topic.",
        "self_containedness": "A cold viewer can follow it without omitted context.",
        "educational_substance": "The clip teaches a substantive idea rather than filler.",
        "summary_takeaway_grounding": "The summary and takeaways are supported by the clip.",
        "assessment_correctness_ambiguity":
            "The assessment has one supported answer and is not materially ambiguous.",
    },
    # These are explicit judgments, never inferred from the seven numeric ratings.
    "explicit_judgments": ("human_usable", "assessment_correct", "severe_hallucination"),
}

EXPLICIT_JUDGMENTS = (
    "human_usable", "assessment_correct", "severe_hallucination",
)


def _number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(float(value))


def _timed_text(items: list, start: float, end: float, text_key: str) -> str:
    words: list[str] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        try:
            item_start = float(item.get("start"))
            item_end = float(item.get("end", item_start))
        except (TypeError, ValueError):
            continue
        if min(item_end, end) - max(item_start, start) <= 0:
            continue
        text = " ".join(str(item.get(text_key) or "").split())
        if text:
            words.append(text)
    return " ".join(words).strip()


def transcript_text(transcript: dict, start: float, end: float) -> str:
    """Text overlapping a time window, preferring word timings for boundary context."""
    if end <= start:
        return ""
    words = transcript.get("words") if isinstance(transcript, dict) else None
    text = _timed_text(words if isinstance(words, list) else [], start, end, "word")
    if text:
        return text
    segments = transcript.get("segments") if isinstance(transcript, dict) else None
    return _timed_text(segments if isinstance(segments, list) else [], start, end, "text")


def _transcript_duration(transcript: dict) -> float:
    raw = transcript.get("duration") if isinstance(transcript, dict) else None
    if _number(raw) and float(raw) > 0:
        return float(raw)
    ends: list[float] = []
    for key in ("segments", "words"):
        for item in transcript.get(key) or []:
            try:
                ends.append(float(item.get("end")))
            except (AttributeError, TypeError, ValueError):
                pass
    return max(ends, default=0.0)


def _manifest_id(items: list[dict], seed: int, context_seconds: float) -> str:
    identities = sorted(
        (str(item.get("dataset_version")), str(item.get("postprocess_version")),
         str(item.get("pair_id")), str(item.get("video_id")), str(item.get("topic")),
         str(item.get("profile")), int(item.get("repeat", -1)), str(item.get("clip_id")))
        for item in items
    )
    raw = json.dumps({"identities": identities, "seed": seed,
                      "context_seconds": context_seconds}, separators=(",", ":"))
    return "segment-review-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def build_blind_review_bundle(items: Iterable[dict], transcripts: dict[str, dict], *,
                              seed: int = 0, context_seconds: float = 4.0
                              ) -> tuple[dict, dict]:
    """Return ``(public_manifest, hidden_mapping)`` with no model/profile leakage.

    ``transcripts`` may be keyed by pair_id (preferred) or video_id.  Anonymous IDs are
    assigned *after* a seeded shuffle and reveal no profile, repeat, or clip identifier.
    Each public item contains the complete clip transcript plus +/- context around both
    boundaries.  The hidden mapping is a separate object and must be stored separately.
    """
    source_items = list(items)
    if not _number(context_seconds) or float(context_seconds) <= 0:
        raise ValueError("context_seconds must be positive")
    required = (
        "dataset_version", "postprocess_version", "pair_id", "profile", "repeat",
        "clip_id", "video_id", "topic", "start", "end",
    )
    seen: set[tuple] = set()
    for index, item in enumerate(source_items):
        if not isinstance(item, dict):
            raise ValueError(f"items[{index}] must be an object")
        missing = [field for field in required if field not in item]
        if missing:
            raise ValueError(f"items[{index}] missing fields: {missing}")
        for field in required:
            if field not in {"repeat", "start", "end"} and not str(item[field]).strip():
                raise ValueError(f"items[{index}].{field} is required")
        if not _number(item["start"]) or not _number(item["end"]):
            raise ValueError(f"items[{index}] start/end must be numeric")
        if float(item["end"]) <= float(item["start"]):
            raise ValueError(f"items[{index}] end must be after start")
        if isinstance(item["repeat"], bool) or not isinstance(item["repeat"], int):
            raise ValueError(f"items[{index}] repeat must be an integer")
        identity = (str(item["pair_id"]), str(item["profile"]), item["repeat"],
                    str(item["clip_id"]))
        if identity in seen:
            raise ValueError(f"duplicate review source identity {identity}")
        seen.add(identity)

    rng = random.Random(seed)
    rng.shuffle(source_items)
    manifest_id = _manifest_id(source_items, seed, float(context_seconds))
    public_items: list[dict] = []
    hidden_items: list[dict] = []
    for ordinal, item in enumerate(source_items, 1):
        review_id = f"review-item-{ordinal:04d}"
        pair_id, video_id = str(item["pair_id"]), str(item["video_id"])
        transcript = transcripts.get(pair_id) or transcripts.get(video_id)
        if not isinstance(transcript, dict):
            raise ValueError(f"no transcript supplied for pair {pair_id!r}")
        start, end = float(item["start"]), float(item["end"])
        duration = max(_transcript_duration(transcript), end)
        opening_start = max(0.0, start - float(context_seconds))
        opening_end = min(duration, start + float(context_seconds))
        closing_start = max(0.0, end - float(context_seconds))
        closing_end = min(duration, end + float(context_seconds))
        public_items.append({
            "review_item_id": review_id,
            "video_id": video_id,
            "topic": str(item["topic"]),
            "start": start,
            "end": end,
            "clip_transcript": transcript_text(transcript, start, end),
            "opening_boundary_context": {
                "start": opening_start, "end": opening_end,
                "text": transcript_text(transcript, opening_start, opening_end),
            },
            "closing_boundary_context": {
                "start": closing_start, "end": closing_end,
                "text": transcript_text(transcript, closing_start, closing_end),
            },
            "summary": str(item.get("summary") or ""),
            "takeaways": [str(x) for x in (item.get("takeaways") or [])],
            "assessment": _blind_assessment(item.get("assessment")),
        })
        hidden_items.append({
            "review_item_id": review_id,
            "dataset_version": str(item["dataset_version"]),
            "postprocess_version": str(item["postprocess_version"]),
            "pair_id": pair_id,
            "video_id": video_id,
            "topic": str(item["topic"]),
            "profile": str(item["profile"]),
            "repeat": item["repeat"],
            "clip_id": str(item["clip_id"]),
        })

    public = {
        "schema_version": 1,
        "manifest_id": manifest_id,
        "blinded": True,
        "context_seconds": float(context_seconds),
        "rubric": REVIEW_RUBRIC,
        "items": public_items,
    }
    hidden = {"schema_version": 1, "manifest_id": manifest_id, "items": hidden_items}
    return public, hidden


def validate_review_records(records: Iterable[dict], *,
                            expected_item_ids: Iterable[str] | None = None,
                            minimum_reviewers: int = 2) -> list[str]:
    """Validate independent primary reviews and optional third-reviewer adjudications."""
    errors: list[str] = []
    seen: set[tuple[str, str]] = set()
    primary_by_item: dict[str, set[str]] = {}
    adjudicators: dict[str, set[str]] = {}
    record_list = list(records)
    expected = set(expected_item_ids or [])
    for index, record in enumerate(record_list):
        prefix = f"records[{index}]"
        if not isinstance(record, dict):
            errors.append(f"{prefix} must be an object")
            continue
        item_id = str(record.get("review_item_id") or "").strip()
        reviewer_id = str(record.get("reviewer_id") or "").strip()
        role = record.get("role")
        if not item_id:
            errors.append(f"{prefix}.review_item_id is required")
        elif expected and item_id not in expected:
            errors.append(f"{prefix}.review_item_id is not in the hidden mapping")
        if not reviewer_id:
            errors.append(f"{prefix}.reviewer_id is required")
        if role not in {"reviewer", "adjudicator"}:
            errors.append(f"{prefix}.role must be reviewer|adjudicator")
        if item_id and reviewer_id:
            key = (item_id, reviewer_id)
            if key in seen:
                errors.append(f"duplicate review by {reviewer_id!r} for {item_id!r}")
            seen.add(key)

        scores = record.get("scores")
        if not isinstance(scores, dict):
            errors.append(f"{prefix}.scores must be an object")
        else:
            missing = set(REVIEW_DIMENSIONS) - set(scores)
            extra = set(scores) - set(REVIEW_DIMENSIONS)
            if missing:
                errors.append(f"{prefix}.scores missing dimensions: {sorted(missing)}")
            if extra:
                errors.append(f"{prefix}.scores has unknown dimensions: {sorted(extra)}")
            for dimension in REVIEW_DIMENSIONS:
                value = scores.get(dimension)
                if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
                    errors.append(f"{prefix}.scores.{dimension} must be an integer in [1, 5]")
        for field in EXPLICIT_JUDGMENTS:
            if not isinstance(record.get(field), bool):
                errors.append(f"{prefix}.{field} must be an explicit boolean")
        if item_id and reviewer_id and role == "reviewer":
            primary_by_item.setdefault(item_id, set()).add(reviewer_id)
        elif item_id and reviewer_id and role == "adjudicator":
            adjudicators.setdefault(item_id, set()).add(reviewer_id)

    expected = expected or set(primary_by_item)
    for item_id in sorted(expected):
        reviewers = primary_by_item.get(item_id, set())
        if len(reviewers) < minimum_reviewers:
            errors.append(
                f"{item_id!r} has {len(reviewers)} independent reviewer(s); "
                f"requires {minimum_reviewers}"
            )
        overlap = reviewers & adjudicators.get(item_id, set())
        if overlap:
            errors.append(f"{item_id!r} adjudicator must be independent of primary reviewers")
        adjudicator_count = len(adjudicators.get(item_id, set()))
        if adjudicator_count > 1:
            errors.append(f"{item_id!r} has more than one adjudicator")
    for requirement in adjudication_requirements(record_list, expected_item_ids=expected):
        if not requirement["resolved"]:
            errors.append(f"{requirement['review_item_id']!r} requires adjudication")
    return errors


def adjudication_requirements(records: Iterable[dict], *,
                              expected_item_ids: Iterable[str] | None = None) -> list[dict]:
    """Flag dimensions where primary reviewers differ by more than one point."""
    grouped: dict[str, list[dict]] = {}
    adjudicators: dict[str, list[dict]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        item_id = str(record.get("review_item_id") or "")
        if record.get("role") == "reviewer":
            grouped.setdefault(item_id, []).append(record)
        elif record.get("role") == "adjudicator":
            adjudicators.setdefault(item_id, []).append(record)
    item_ids = set(expected_item_ids or grouped)
    out: list[dict] = []
    for item_id in sorted(item_ids):
        dimensions: list[str] = []
        judgments: list[str] = []
        reviews = grouped.get(item_id, [])
        for dimension in REVIEW_DIMENSIONS:
            values = [r.get("scores", {}).get(dimension) for r in reviews]
            valid = [v for v in values
                     if isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 5]
            if len(valid) >= 2 and max(valid) - min(valid) > 1:
                dimensions.append(dimension)
        for field in EXPLICIT_JUDGMENTS:
            values = [record.get(field) for record in reviews]
            valid = [value for value in values if isinstance(value, bool)]
            if len(valid) >= 2 and len(set(valid)) > 1:
                judgments.append(field)
        if dimensions or judgments:
            adjudicator_records = adjudicators.get(item_id, [])
            dimensions_resolved = all(any(
                isinstance(record.get("scores", {}).get(dimension), int)
                and not isinstance(record.get("scores", {}).get(dimension), bool)
                and 1 <= record["scores"][dimension] <= 5
                for record in adjudicator_records
            ) for dimension in dimensions)
            judgments_resolved = all(any(
                isinstance(record.get(field), bool)
                for record in adjudicator_records
            ) for field in judgments)
            out.append({"review_item_id": item_id, "dimensions": dimensions,
                        "judgments": judgments,
                        "adjudicator_present": bool(adjudicator_records),
                        "resolved": dimensions_resolved and judgments_resolved})
    return out


def resolve_review_records(records: Iterable[dict], hidden_mapping: dict) -> list[dict]:
    """Resolve blinded clip reviews to one deterministic record per reviewed clip."""
    record_list = list(records)
    mappings = hidden_mapping.get("items") if isinstance(hidden_mapping, dict) else None
    if not isinstance(mappings, list):
        raise ValueError("hidden_mapping.items must be a list")
    by_id: dict[str, dict] = {}
    required_mapping = (
        "dataset_version", "postprocess_version", "pair_id", "video_id", "topic",
        "profile", "repeat", "clip_id",
    )
    for index, mapping in enumerate(mappings):
        if not isinstance(mapping, dict):
            raise ValueError(f"hidden_mapping.items[{index}] must be an object")
        item_id = str(mapping.get("review_item_id") or "")
        if not item_id or item_id in by_id:
            raise ValueError("hidden mapping review item IDs must be non-empty and unique")
        missing = [field for field in required_mapping if field not in mapping]
        if missing:
            raise ValueError(f"hidden mapping {item_id!r} missing fields: {missing}")
        by_id[item_id] = mapping

    errors = validate_review_records(
        record_list, expected_item_ids=by_id, minimum_reviewers=2)
    if errors:
        raise ValueError("invalid clip reviews: " + "; ".join(errors))

    grouped: dict[str, dict[str, list[dict]]] = {
        item_id: {"reviewer": [], "adjudicator": []} for item_id in by_id
    }
    for record in record_list:
        grouped[record["review_item_id"]][record["role"]].append(record)

    resolved: list[dict] = []
    for item_id in sorted(by_id):
        mapping = by_id[item_id]
        reviewers = grouped[item_id]["reviewer"]
        adjudicators = grouped[item_id]["adjudicator"]
        adjudicator = adjudicators[0] if adjudicators else None
        scores: dict[str, float] = {}
        adjudicated_dimensions: list[str] = []
        for dimension in REVIEW_DIMENSIONS:
            values = [record["scores"][dimension] for record in reviewers]
            if max(values) - min(values) > 1:
                scores[dimension] = float(adjudicator["scores"][dimension])
                adjudicated_dimensions.append(dimension)
            else:
                scores[dimension] = sum(values) / len(values)
        judgments: dict[str, bool] = {}
        adjudicated_judgments: list[str] = []
        for field in EXPLICIT_JUDGMENTS:
            values = [record[field] for record in reviewers]
            if len(set(values)) == 1:
                judgments[field] = values[0]
            else:
                judgments[field] = adjudicator[field]
                adjudicated_judgments.append(field)
        resolved.append({
            "review_item_id": item_id,
            **{field: mapping[field] for field in required_mapping},
            "scores": scores,
            "mean_human_score": sum(scores.values()) / len(scores),
            **judgments,
            "reviewer_count": len(reviewers),
            "adjudicator_present": adjudicator is not None,
            "adjudicated_dimensions": adjudicated_dimensions,
            "adjudicated_judgments": adjudicated_judgments,
            "resolution_validated": True,
        })
    return resolved


# ---------------------------------------------------------------------------
# Whole-video blinded A/B review

WHOLE_VIDEO_CHOICES = ("a", "tied", "b")


def _index_generation_rows(rows: Iterable[dict], label: str) -> dict[tuple[str, int], dict]:
    indexed: dict[tuple[str, int], dict] = {}
    required = (
        "dataset_version", "postprocess_version", "pair_id", "repeat", "video_id",
        "topic", "profile", "clips",
    )
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{label}[{index}] must be an object")
        missing = [field for field in required if field not in row]
        if missing:
            raise ValueError(f"{label}[{index}] missing fields: {missing}")
        if isinstance(row["repeat"], bool) or not isinstance(row["repeat"], int):
            raise ValueError(f"{label}[{index}].repeat must be an integer")
        if not isinstance(row["clips"], list):
            raise ValueError(f"{label}[{index}].clips must be a list")
        for field in (
            "dataset_version", "postprocess_version", "pair_id", "video_id", "topic", "profile",
        ):
            if not str(row[field]).strip():
                raise ValueError(f"{label}[{index}].{field} is required")
        key = (str(row["pair_id"]), row["repeat"])
        if key in indexed:
            raise ValueError(f"duplicate {label} row {key}")
        indexed[key] = row
    return indexed


def _blind_assessment(value: object):
    if not isinstance(value, dict):
        return None
    allowed = ("prompt", "options", "correct_index", "explanation", "evidence_quote")
    return {field: value[field] for field in allowed if value.get(field) is not None}


def _blind_clip(clip: object, label: str) -> dict:
    if not isinstance(clip, dict):
        raise ValueError(f"{label} must be an object")
    if not _number(clip.get("start")) or not _number(clip.get("end")):
        raise ValueError(f"{label} start/end must be numeric")
    start, end = float(clip["start"]), float(clip["end"])
    if end <= start:
        raise ValueError(f"{label} end must be after start")
    return {
        "start": start,
        "end": end,
        "title": str(clip.get("title") or ""),
        "summary": str(clip.get("summary") or ""),
        "takeaways": [str(item) for item in (clip.get("takeaways") or [])],
        "assessment": _blind_assessment(clip.get("assessment")),
    }


def _whole_video_manifest_id(candidate: dict, baseline: dict, seed: int) -> str:
    identities = sorted(
        (candidate[key]["dataset_version"], candidate[key]["postprocess_version"], key[0],
         candidate[key]["video_id"], candidate[key]["topic"], key[1],
         candidate[key]["profile"], baseline[key]["profile"])
        for key in candidate
    )
    raw = json.dumps({"comparisons": identities, "seed": seed},
                     separators=(",", ":"))
    return "whole-video-review-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def build_blind_whole_video_bundle(candidate_rows: Iterable[dict], pro_rows: Iterable[dict], *,
                                   seed: int = 0) -> tuple[dict, dict]:
    """Blind candidate-vs-selected-Pro clip sets and independently randomize A/B.

    The public artifact contains only source identity needed for playback, the topic, and
    whitelisted clip content.  Pair/repeat/profile/orientation stay in the hidden mapping.
    Empty clip sets remain visible and reviewable rather than being silently dropped.
    """
    candidate = _index_generation_rows(candidate_rows, "candidate_rows")
    baseline = _index_generation_rows(pro_rows, "pro_rows")
    if set(candidate) != set(baseline):
        raise ValueError("candidate/pro pair-repeat keys must match exactly")
    candidate_profiles = {str(row["profile"]) for row in candidate.values()}
    baseline_profiles = {str(row["profile"]) for row in baseline.values()}
    if len(candidate_profiles) != 1 or len(baseline_profiles) != 1:
        raise ValueError("candidate and selected Pro inputs must each use one profile")
    if candidate_profiles == baseline_profiles:
        raise ValueError("candidate and selected Pro profiles must differ")

    rng = random.Random(seed)
    groups: list[tuple[dict, dict]] = []
    for key in sorted(candidate):
        cand, pro = candidate[key], baseline[key]
        for field in ("video_id", "topic"):
            if cand[field] != pro[field]:
                raise ValueError(f"paired rows {key} disagree on {field}")
        for field in ("dataset_version", "postprocess_version"):
            if cand[field] != pro[field]:
                raise ValueError(f"paired rows {key} disagree on {field}")
        candidate_clips = [
            _blind_clip(clip, f"candidate {key} clips[{index}]")
            for index, clip in enumerate(cand["clips"])
        ]
        pro_clips = [
            _blind_clip(clip, f"pro {key} clips[{index}]")
            for index, clip in enumerate(pro["clips"])
        ]
        candidate_side = "a" if rng.randrange(2) == 0 else "b"
        sides = ({"a": candidate_clips, "b": pro_clips}
                 if candidate_side == "a"
                 else {"a": pro_clips, "b": candidate_clips})
        groups.append(({
            "video_id": str(cand["video_id"]),
            "topic": str(cand["topic"]),
            "set_a": {"clips": sides["a"]},
            "set_b": {"clips": sides["b"]},
        }, {
            "dataset_version": str(cand["dataset_version"]),
            "postprocess_version": str(cand["postprocess_version"]),
            "pair_id": key[0],
            "repeat": key[1],
            "video_id": str(cand["video_id"]),
            "topic": str(cand["topic"]),
            "candidate_profile": next(iter(candidate_profiles)),
            "selected_pro_profile": next(iter(baseline_profiles)),
            "candidate_side": candidate_side,
        }))
    rng.shuffle(groups)

    manifest_id = _whole_video_manifest_id(candidate, baseline, seed)
    public_comparisons: list[dict] = []
    hidden_comparisons: list[dict] = []
    for ordinal, (public_group, hidden_group) in enumerate(groups, 1):
        comparison_id = f"whole-video-{ordinal:04d}"
        public_comparisons.append({"comparison_id": comparison_id, **public_group})
        hidden_comparisons.append({"comparison_id": comparison_id, **hidden_group})
    public = {
        "schema_version": 1,
        "manifest_id": manifest_id,
        "blinded": True,
        "choice_options": list(WHOLE_VIDEO_CHOICES),
        "instruction": "Choose which complete clip set is better for this topic, or tied.",
        "comparisons": public_comparisons,
    }
    hidden = {
        "schema_version": 1,
        "manifest_id": manifest_id,
        "comparisons": hidden_comparisons,
    }
    return public, hidden


def validate_whole_video_review_records(
        records: Iterable[dict], *, expected_comparison_ids: Iterable[str] | None = None,
        minimum_reviewers: int = 2) -> list[str]:
    """Require two distinct reviewers and an explicit a|tied|b choice per comparison."""
    errors: list[str] = []
    seen: set[tuple[str, str]] = set()
    reviewers_by_comparison: dict[str, set[str]] = {}
    adjudicators_by_comparison: dict[str, set[str]] = {}
    choices_by_comparison: dict[str, set[str]] = {}
    expected = set(expected_comparison_ids or [])
    for index, record in enumerate(records):
        prefix = f"records[{index}]"
        if not isinstance(record, dict):
            errors.append(f"{prefix} must be an object")
            continue
        comparison_id = str(record.get("comparison_id") or "").strip()
        reviewer_id = str(record.get("reviewer_id") or "").strip()
        role = record.get("role")
        choice = record.get("choice")
        if not comparison_id:
            errors.append(f"{prefix}.comparison_id is required")
        elif expected and comparison_id not in expected:
            errors.append(f"{prefix}.comparison_id is not in the hidden mapping")
        if not reviewer_id:
            errors.append(f"{prefix}.reviewer_id is required")
        if role not in {"reviewer", "adjudicator"}:
            errors.append(f"{prefix}.role must be reviewer|adjudicator")
        if choice not in WHOLE_VIDEO_CHOICES:
            errors.append(f"{prefix}.choice must be a|tied|b")
        if comparison_id and reviewer_id:
            key = (comparison_id, reviewer_id)
            if key in seen:
                errors.append(
                    f"duplicate whole-video review by {reviewer_id!r} for {comparison_id!r}")
            seen.add(key)
            if role == "reviewer":
                reviewers_by_comparison.setdefault(comparison_id, set()).add(reviewer_id)
                if choice in WHOLE_VIDEO_CHOICES:
                    choices_by_comparison.setdefault(comparison_id, set()).add(choice)
            elif role == "adjudicator":
                adjudicators_by_comparison.setdefault(comparison_id, set()).add(reviewer_id)
    expected = expected or set(reviewers_by_comparison)
    for comparison_id in sorted(expected):
        count = len(reviewers_by_comparison.get(comparison_id, set()))
        if count < minimum_reviewers:
            errors.append(
                f"{comparison_id!r} has {count} independent reviewer(s); "
                f"requires {minimum_reviewers}"
            )
        adjudicators = adjudicators_by_comparison.get(comparison_id, set())
        if len(adjudicators) > 1:
            errors.append(f"{comparison_id!r} has more than one adjudicator")
        if reviewers_by_comparison.get(comparison_id, set()) & adjudicators:
            errors.append(
                f"{comparison_id!r} adjudicator must be independent of primary reviewers")
        if len(choices_by_comparison.get(comparison_id, set())) > 1 and not adjudicators:
            errors.append(f"{comparison_id!r} requires adjudication")
    return errors


def decode_whole_video_review_records(records: Iterable[dict], hidden_mapping: dict) -> list[dict]:
    """Resolve blinded choices to one candidate-relative verdict per comparison."""
    record_list = list(records)
    mappings = hidden_mapping.get("comparisons") if isinstance(hidden_mapping, dict) else None
    if not isinstance(mappings, list):
        raise ValueError("hidden_mapping.comparisons must be a list")
    by_id: dict[str, dict] = {}
    for index, mapping in enumerate(mappings):
        if not isinstance(mapping, dict):
            raise ValueError(f"hidden_mapping.comparisons[{index}] must be an object")
        comparison_id = str(mapping.get("comparison_id") or "")
        if not comparison_id or comparison_id in by_id:
            raise ValueError("hidden mapping comparison IDs must be non-empty and unique")
        if mapping.get("candidate_side") not in {"a", "b"}:
            raise ValueError(f"hidden mapping {comparison_id!r} has invalid candidate_side")
        by_id[comparison_id] = mapping
    errors = validate_whole_video_review_records(
        record_list, expected_comparison_ids=by_id, minimum_reviewers=2)
    if errors:
        raise ValueError("invalid whole-video reviews: " + "; ".join(errors))

    grouped: dict[str, dict[str, list[dict]]] = {
        comparison_id: {"reviewer": [], "adjudicator": []} for comparison_id in by_id
    }
    for record in record_list:
        grouped[record["comparison_id"]][record["role"]].append(record)

    decoded: list[dict] = []
    required_mapping = (
        "dataset_version", "postprocess_version", "pair_id", "repeat", "video_id", "topic",
        "candidate_profile", "selected_pro_profile",
    )
    for comparison_id in sorted(by_id):
        mapping = by_id[comparison_id]
        missing = [field for field in required_mapping if field not in mapping]
        if missing:
            raise ValueError(f"hidden mapping {comparison_id!r} missing fields: {missing}")
        reviewer_choices = {record["choice"] for record in grouped[comparison_id]["reviewer"]}
        if len(reviewer_choices) == 1:
            choice = next(iter(reviewer_choices))
            adjudicated = False
        else:
            choice = grouped[comparison_id]["adjudicator"][0]["choice"]
            adjudicated = True
        candidate_side = mapping["candidate_side"]
        relative = "tied" if choice == "tied" else "better" if choice == candidate_side \
            else "worse"
        decoded.append({
            "comparison_id": comparison_id,
            **{field: mapping[field] for field in required_mapping},
            "whole_video_comparison": relative,
            "reviewer_count": len(grouped[comparison_id]["reviewer"]),
            "adjudicator_present": bool(grouped[comparison_id]["adjudicator"]),
            "adjudicated": adjudicated,
            "resolution_validated": True,
        })
    return decoded


resolve_whole_video_review_records = decode_whole_video_review_records


def write_review_bundle(public_manifest: dict, hidden_mapping: dict, *,
                        public_path: Path, hidden_path: Path) -> tuple[Path, Path]:
    """Write public and hidden artifacts separately; the paths may never alias."""
    public_out, hidden_out = Path(public_path), Path(hidden_path)
    if public_out.resolve() == hidden_out.resolve():
        raise ValueError("public manifest and hidden mapping must use separate files")
    public_out.parent.mkdir(parents=True, exist_ok=True)
    hidden_out.parent.mkdir(parents=True, exist_ok=True)
    public_out.write_text(json.dumps(public_manifest, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")
    hidden_out.write_text(json.dumps(hidden_mapping, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")
    return public_out, hidden_out
