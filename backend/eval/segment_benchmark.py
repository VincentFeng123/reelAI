"""Executable, incremental reporting path for the Gemini segment benchmark.

The CLI validates a tracked, frozen 25-case corpus and runs four immutable model profiles
three times before any winner is chosen.  ``synthesize_hybrid_rows`` adds the selected fifth
profile later without another model call.  Every executed trial is retained, including errors
and zero outputs.  Tests inject the profile runner so importing/reporting remains offline.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Callable, Iterable


EXPECTED_PAIR_COUNT = 25
EXPECTED_REPEATS = 3

PROFILE_PRODUCTION_PRO = "production_pro_v0"
PROFILE_CORRECTED_PRO = "corrected_pro_v1"
PROFILE_FLASH_SINGLE = "flash_single_v1"
PROFILE_FLASH_SPLIT = "flash_split_v2"
PROFILE_SIMULATED_HYBRID = "simulated_hybrid_v1"
BENCHMARK_PROFILES = (
    PROFILE_PRODUCTION_PRO,
    PROFILE_CORRECTED_PRO,
    PROFILE_FLASH_SINGLE,
    PROFILE_FLASH_SPLIT,
    PROFILE_SIMULATED_HYBRID,
)
REAL_PROFILES = BENCHMARK_PROFILES[:4]
EXPECTED_TRIAL_ROWS = EXPECTED_PAIR_COUNT * EXPECTED_REPEATS * len(BENCHMARK_PROFILES)

CONTENT_STRATA = frozenset({
    "lecture", "speech", "interview", "podcast", "auto_caption", "comedy_negative",
})

PRICING_SNAPSHOT_PATH = (
    Path(__file__).resolve().parent / "pricing" / "gemini_standard_2026-07-11.json"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def load_pricing_snapshot(path: Path = PRICING_SNAPSHOT_PATH) -> dict:
    """Load and minimally validate a versioned pricing snapshot."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or not data.get("pricing_version"):
        raise ValueError("invalid pricing snapshot metadata")
    if data.get("unit") != "per_1m_tokens" or not isinstance(data.get("models"), dict):
        raise ValueError("invalid pricing snapshot units/models")
    return data


def usage_cost_usd(model: str, usage: dict, snapshot: dict | None = None) -> float:
    """Price recorded Gemini usage; thoughts are billed with output tokens.

    The dated snapshot only covers prompts up to 200k tokens, so larger prompts fail
    closed instead of silently applying the wrong tier.
    """
    snap = snapshot or load_pricing_snapshot()
    rates = (snap.get("models") or {}).get(model)
    if not isinstance(rates, dict):
        raise ValueError(f"model {model!r} is absent from pricing snapshot")

    def count(name: str) -> int:
        value = usage.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"usage.{name} must be a non-negative integer")
        return value

    # Names intentionally match google-genai's GenerateContentResponseUsageMetadata.
    prompt = count("prompt_token_count")
    candidate = count("candidates_token_count")
    thought = count("thoughts_token_count")
    count("total_token_count")
    if prompt > int(snap.get("prompt_tier_max_tokens", 0)):
        raise ValueError("prompt exceeds the only tier captured by this pricing snapshot")
    return (
        prompt * float(rates["input"])
        + (candidate + thought) * float(rates["output_including_thoughts"])
    ) / 1_000_000.0


def _value(value: object, name: str, default=None):
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _first(value: object, names: tuple[str, ...]):
    for name in names:
        found = _value(value, name)
        if found is not None:
            return found
    return None


def _optional_token(value: object, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"telemetry.{field} must be a non-negative integer or null")
    return value


def normalize_call_telemetry(call: object) -> dict:
    """Normalize dataclass/current/internal/provider telemetry into one stable row shape."""
    if hasattr(call, "as_dict"):
        call = call.as_dict()
    elif is_dataclass(call):
        call = asdict(call)
    prompt = _optional_token(
        _first(call, ("prompt_tokens", "prompt_token_count")), "prompt_tokens")
    candidate = _optional_token(
        _first(call, ("candidate_tokens", "candidates_token_count", "candidate_token_count")),
        "candidate_tokens",
    )
    thought = _optional_token(
        _first(call, ("thought_tokens", "thoughts_token_count", "thought_token_count")),
        "thought_tokens",
    )
    total = _optional_token(
        _first(call, ("total_tokens", "total_token_count")), "total_tokens")
    latency = _first(call, ("latency_ms",))
    retries = _first(call, ("retries",))
    return {
        "model": str(_first(call, ("model",)) or ""),
        "operation": str(_first(call, ("operation",)) or ""),
        "prompt_version": str(_first(call, ("prompt_version",)) or ""),
        "thinking_level": str(_first(call, ("thinking_level",)) or ""),
        "latency_ms": float(latency) if _is_number(latency) and float(latency) >= 0 else None,
        "retries": int(retries) if isinstance(retries, int) and not isinstance(retries, bool)
        and retries >= 0 else None,
        "finish_reason": (
            None if _first(call, ("finish_reason",)) is None
            else str(_first(call, ("finish_reason",)))
        ),
        "prompt_tokens": prompt,
        "candidate_tokens": candidate,
        "thought_tokens": thought,
        "total_tokens": total,
    }


def aggregate_usage(calls: Iterable[dict]) -> dict:
    """Sum actual call usage; any missing count keeps that aggregate explicitly null."""
    call_list = list(calls)
    if not call_list:
        return {
            "prompt_tokens": None,
            "candidate_tokens": None,
            "thought_tokens": None,
            "total_tokens": None,
            "complete": False,
        }
    totals: dict[str, int | None] = {}
    for field in ("prompt_tokens", "candidate_tokens", "thought_tokens", "total_tokens"):
        values = [call.get(field) for call in call_list]
        totals[field] = sum(values) if all(isinstance(v, int) for v in values) else None
    totals["complete"] = all(totals[field] is not None for field in (
        "prompt_tokens", "candidate_tokens", "thought_tokens", "total_tokens"))
    return totals


def calls_cost_usd(calls: Iterable[dict], snapshot: dict | None = None) -> float | None:
    """Price normalized calls; null means provider usage was incomplete, never estimated."""
    snap = snapshot or load_pricing_snapshot()
    call_list = list(calls)
    if not call_list:
        return None
    total = 0.0
    for call in call_list:
        usage = {
            "prompt_token_count": call.get("prompt_tokens"),
            "candidates_token_count": call.get("candidate_tokens"),
            "thoughts_token_count": call.get("thought_tokens"),
            "total_token_count": call.get("total_tokens"),
        }
        if not all(isinstance(value, int) for value in usage.values()):
            return None
        total += usage_cost_usd(str(call.get("model") or ""), usage, snap)
    return total


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(float(value))


def _validate_transcript_data(transcript: object, label: str) -> list[str]:
    errors: list[str] = []
    segments = transcript.get("segments") if isinstance(transcript, dict) else None
    if not isinstance(segments, list) or not segments:
        return [f"transcript {label} has no segments"]
    previous_start = -1.0
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            errors.append(f"transcript {label} segment {index} is not an object")
            continue
        start, end = segment.get("start"), segment.get("end")
        if not _is_number(start) or not _is_number(end) or float(end) <= float(start):
            errors.append(f"transcript {label} segment {index} has invalid timing")
            continue
        if float(start) < previous_start:
            errors.append(f"transcript {label} segments are not chronological")
        previous_start = float(start)
    return errors


def _validate_transcript(path: Path) -> list[str]:
    try:
        transcript = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - validation should report every bad case
        return [f"unreadable transcript {path}: {exc}"]
    return _validate_transcript_data(transcript, str(path))


def validate_case_manifest(manifest: dict, *, base_dir: Path | None = None,
                           verify_files: bool = False) -> list[str]:
    """Return all frozen-corpus errors; an empty list means the manifest is usable.

    Required case fields intentionally include topic, stratum, duration, transcript path,
    and SHA-256 so timings/queries cannot drift between profiles.  The repository currently
    carries no such corpus; this validator prevents a partial local cache being mistaken for
    the required pinned 25-pair evaluation.
    """
    errors: list[str] = []
    if not isinstance(manifest, dict):
        return ["manifest must be an object"]
    if not str(manifest.get("dataset_version") or "").strip():
        errors.append("dataset_version is required")
    if not str(manifest.get("postprocess_version") or "").strip():
        errors.append("postprocess_version is required")
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        return errors + ["cases must be a list"]
    if len(cases) != EXPECTED_PAIR_COUNT:
        errors.append(f"expected {EXPECTED_PAIR_COUNT} cases, found {len(cases)}")

    seen_ids: set[str] = set()
    seen_strata: set[str] = set()
    root = Path(base_dir) if base_dir is not None else None
    for index, case in enumerate(cases):
        prefix = f"cases[{index}]"
        if not isinstance(case, dict):
            errors.append(f"{prefix} must be an object")
            continue
        pair_id = str(case.get("pair_id") or "").strip()
        if not pair_id:
            errors.append(f"{prefix}.pair_id is required")
        elif pair_id in seen_ids:
            errors.append(f"duplicate pair_id {pair_id!r}")
        seen_ids.add(pair_id)
        for field in ("video_id", "topic", "transcript_path"):
            if not str(case.get(field) or "").strip():
                errors.append(f"{prefix}.{field} is required")
        stratum = str(case.get("stratum") or "").strip()
        if stratum not in CONTENT_STRATA:
            errors.append(f"{prefix}.stratum must be one of {sorted(CONTENT_STRATA)}")
        else:
            seen_strata.add(stratum)
        positive = case.get("positive_content")
        if not isinstance(positive, bool):
            errors.append(f"{prefix}.positive_content must be boolean")
        elif stratum == "comedy_negative" and positive:
            errors.append(f"{prefix} comedy_negative must not be positive_content")
        elif stratum and stratum != "comedy_negative" and not positive:
            errors.append(f"{prefix} positive stratum must set positive_content=true")
        if not _is_number(case.get("duration_s")) or float(case.get("duration_s", 0)) <= 0:
            errors.append(f"{prefix}.duration_s must be positive")
        digest = str(case.get("transcript_sha256") or "")
        if not _SHA256_RE.fullmatch(digest):
            errors.append(f"{prefix}.transcript_sha256 must be lowercase SHA-256")

        if verify_files and root is not None and str(case.get("transcript_path") or ""):
            path = (root / str(case["transcript_path"])).resolve()
            try:
                path.relative_to(root.resolve())
            except ValueError:
                errors.append(f"{prefix}.transcript_path escapes the dataset directory")
                continue
            if not path.is_file():
                errors.append(f"{prefix} transcript is missing: {path}")
                continue
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if _SHA256_RE.fullmatch(digest) and actual != digest:
                errors.append(f"{prefix} transcript SHA-256 mismatch")
            errors.extend(_validate_transcript(path))

    missing_strata = CONTENT_STRATA - seen_strata
    if missing_strata:
        errors.append(f"missing required strata: {sorted(missing_strata)}")
    return errors


def require_valid_case_manifest(manifest: dict, *, base_dir: Path | None = None,
                                verify_files: bool = False) -> None:
    errors = validate_case_manifest(manifest, base_dir=base_dir, verify_files=verify_files)
    if errors:
        raise ValueError("invalid segment benchmark manifest: " + "; ".join(errors))


def trial_matrix(manifest: dict) -> list[dict]:
    """Expand a validated manifest into the deterministic 375-row trial identity matrix."""
    require_valid_case_manifest(manifest)
    dataset_version = str(manifest["dataset_version"])
    postprocess_version = str(manifest["postprocess_version"])
    rows: list[dict] = []
    for case in manifest["cases"]:
        for repeat in range(1, EXPECTED_REPEATS + 1):
            for profile in BENCHMARK_PROFILES:
                rows.append({
                    "dataset_version": dataset_version,
                    "postprocess_version": postprocess_version,
                    "pair_id": case["pair_id"],
                    "video_id": case["video_id"],
                    "topic": case["topic"],
                    "stratum": case["stratum"],
                    "positive_content": case["positive_content"],
                    "repeat": repeat,
                    "profile": profile,
                })
    return rows


def validate_trial_rows(rows: Iterable[dict], manifest: dict, *,
                        profiles: Iterable[str] = BENCHMARK_PROFILES) -> list[str]:
    """Validate that reporting retained one row for every trial, including errors/zeros."""
    selected_profiles = set(profiles)
    expected = {
        (r["pair_id"], r["repeat"], r["profile"])
        for r in trial_matrix(manifest)
        if r["profile"] in selected_profiles
    }
    actual: set[tuple] = set()
    errors: list[str] = []
    allowed_status = {"ok", "zero", "error"}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"rows[{index}] is not an object")
            continue
        key = (row.get("pair_id"), row.get("repeat"), row.get("profile"))
        if key in actual:
            errors.append(f"duplicate trial row {key}")
        actual.add(key)
        if row.get("status") not in allowed_status:
            errors.append(f"trial row {key} has invalid status")
    missing = expected - actual
    extra = actual - expected
    if missing:
        errors.append(f"missing {len(missing)} trial row(s)")
    if extra:
        errors.append(f"found {len(extra)} unexpected trial row(s)")
    return errors


def write_trial_rows_jsonl(rows: Iterable[dict], path: Path) -> Path:
    """Persist video-level trial rows without dropping error or zero-output records."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    out.write_text(payload, encoding="utf-8")
    return out


def _read_transcript_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _require_git_tracked(paths: Iterable[Path]) -> None:
    """Fail unless every corpus artifact is tracked in the containing Git repository."""
    path_list = [Path(path).resolve() for path in paths]
    if not path_list:
        raise ValueError("no corpus paths supplied")
    probe = subprocess.run(
        ["git", "-C", str(path_list[0].parent), "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, check=False,
    )
    if probe.returncode != 0:
        raise ValueError("benchmark manifest is not inside a Git repository")
    repo = Path(probe.stdout.strip()).resolve()
    relative: list[str] = []
    for path in path_list:
        try:
            relative.append(str(path.relative_to(repo)))
        except ValueError as exc:
            raise ValueError(f"corpus artifact is outside repository: {path}") from exc
    tracked = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "--error-unmatch", "--", *relative],
        capture_output=True, text=True, check=False,
    )
    if tracked.returncode != 0:
        raise ValueError("manifest and every frozen transcript must be Git-tracked")


def _transcript_duration(transcript: dict) -> float:
    duration = transcript.get("duration")
    if _is_number(duration) and float(duration) > 0:
        return float(duration)
    ends = [
        float(segment["end"])
        for segment in transcript.get("segments") or []
        if isinstance(segment, dict) and _is_number(segment.get("end"))
    ]
    return max(ends, default=0.0)


def load_verified_case_manifest(manifest_path: Path, *, require_tracked: bool = True
                                ) -> tuple[dict, dict[str, dict]]:
    """Load a 25-case manifest and freeze each unique transcript from disk exactly once."""
    manifest_file = Path(manifest_path).resolve()
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - turn all corpus failures into one clear error
        raise ValueError(f"unreadable benchmark manifest {manifest_file}: {exc}") from exc
    errors = validate_case_manifest(manifest)
    if errors:
        raise ValueError("invalid segment benchmark manifest: " + "; ".join(errors))

    dataset_root = manifest_file.parent.resolve()
    resolved_by_relative: dict[str, Path] = {}
    for case in manifest["cases"]:
        relative = str(case["transcript_path"])
        path = (dataset_root / relative).resolve()
        try:
            path.relative_to(dataset_root)
        except ValueError:
            errors.append(f"transcript_path escapes dataset directory: {relative}")
            continue
        resolved_by_relative[relative] = path

    corpus_paths = [manifest_file, *sorted(set(resolved_by_relative.values()))]
    if require_tracked and not errors:
        _require_git_tracked(corpus_paths)

    frozen_by_path: dict[Path, tuple[bytes, dict]] = {}
    for path in sorted(set(resolved_by_relative.values())):
        if not path.is_file():
            errors.append(f"frozen transcript is missing: {path}")
            continue
        try:
            raw = _read_transcript_bytes(path)
            transcript = json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"unreadable frozen transcript {path}: {exc}")
            continue
        errors.extend(_validate_transcript_data(transcript, str(path)))
        frozen_by_path[path] = (raw, transcript)

    by_pair: dict[str, dict] = {}
    for case in manifest["cases"]:
        relative = str(case["transcript_path"])
        path = resolved_by_relative.get(relative)
        if path not in frozen_by_path:
            continue
        raw, transcript = frozen_by_path[path]
        actual_hash = hashlib.sha256(raw).hexdigest()
        if actual_hash != case["transcript_sha256"]:
            errors.append(f"{case['pair_id']}: transcript SHA-256 mismatch")
        actual_duration = _transcript_duration(transcript)
        if abs(actual_duration - float(case["duration_s"])) > 0.01:
            errors.append(
                f"{case['pair_id']}: duration {actual_duration} != manifest "
                f"{float(case['duration_s'])}"
            )
        by_pair[str(case["pair_id"])] = transcript
    if errors:
        raise ValueError("invalid frozen segment corpus: " + "; ".join(errors))
    return manifest, by_pair


def _result_field(result: object, name: str, default=None):
    if isinstance(result, dict):
        return result.get(name, default)
    return getattr(result, name, default)


def _identity(case: dict, repeat: int, profile: str, dataset_version: str,
              postprocess_version: str) -> dict:
    return {
        "schema_version": 1,
        "dataset_version": dataset_version,
        "postprocess_version": postprocess_version,
        # Every profile executed here uses the new Gemini 3 transport.  A true current-
        # production control must come from a separately frozen historical artifact.
        "transport_provenance": "in_process_gemini3_transport",
        "pair_id": case["pair_id"],
        "video_id": case["video_id"],
        "topic": case["topic"],
        "stratum": case["stratum"],
        "positive_content": case["positive_content"],
        "repeat": repeat,
        "profile": profile,
    }


def _row_cost_fields(calls: list[dict], snapshot: dict) -> dict:
    usage = aggregate_usage(calls)
    cost_error = None
    try:
        cost = calls_cost_usd(calls, snapshot)
    except Exception as exc:  # pricing failure must not erase a completed model result
        cost = None
        cost_error = f"{type(exc).__name__}: {exc}"
    return {
        "calls": calls,
        "usage": usage,
        "cost_usd": cost,
        "cost_error": cost_error,
        "pricing_version": snapshot["pricing_version"],
        "prompt_versions": list(dict.fromkeys(
            call["prompt_version"] for call in calls if call.get("prompt_version"))),
    }


def _profile_row(identity: dict, result: object, elapsed_ms: float, snapshot: dict) -> dict:
    raw_calls = list(_result_field(result, "calls", []) or [])
    calls = [normalize_call_telemetry(call) for call in raw_calls]
    clips = copy.deepcopy(list(_result_field(result, "clips", []) or []))
    for index, clip in enumerate(clips, 1):
        if isinstance(clip, dict):
            clip.setdefault("clip_id", f"clip-{index:03d}")
    accepted = _result_field(result, "accepted_count", len(clips))
    proposed = _result_field(result, "proposed_count", accepted)
    error = _result_field(result, "error")
    call_latencies = [call["latency_ms"] for call in calls if call["latency_ms"] is not None]
    latency_ms = sum(call_latencies) if call_latencies else elapsed_ms
    status = "error" if error else "zero" if int(accepted or 0) == 0 else "ok"
    return {
        **identity,
        "status": status,
        "route": str(_result_field(result, "route", identity["profile"])),
        "classification": str(_result_field(result, "classification", "invalid")),
        "classification_reasons": list(
            _result_field(result, "classification_reasons", []) or []),
        "fallback_used": bool(_result_field(result, "fallback_reasons", [])),
        "fallback_reasons": list(_result_field(result, "fallback_reasons", []) or []),
        "proposed_count": int(proposed or 0),
        "accepted_count": int(accepted or 0),
        "clips": clips,
        "latency_ms": round(float(latency_ms), 3),
        "notes": str(_result_field(result, "notes", "") or ""),
        "error": None if error is None else str(error),
        **_row_cost_fields(calls, snapshot),
    }


def _exception_row(identity: dict, exc: Exception, elapsed_ms: float, snapshot: dict) -> dict:
    telemetry = getattr(exc, "telemetry", None)
    try:
        calls = [normalize_call_telemetry(telemetry)] if telemetry is not None else []
        cost_fields = _row_cost_fields(calls, snapshot)
    except Exception as telemetry_error:  # noqa: BLE001 - reporting must preserve the error row
        calls = []
        cost_fields = _row_cost_fields(calls, snapshot)
        cost_fields["telemetry_error"] = f"{type(telemetry_error).__name__}: {telemetry_error}"
    return {
        **identity,
        "status": "error",
        "route": identity["profile"],
        "classification": "invalid",
        "classification_reasons": [f"request_failure:{type(exc).__name__}"],
        "fallback_used": False,
        "fallback_reasons": [],
        "proposed_count": 0,
        "accepted_count": 0,
        "clips": [],
        "latency_ms": round(float(elapsed_ms), 3),
        "notes": "",
        "error": f"{type(exc).__name__}: {exc}",
        **cost_fields,
    }


def _hybrid_row(identity: dict, flash: dict, fallback_pro: dict, snapshot: dict) -> dict:
    flash_green = flash["status"] != "error" and flash["classification"] == "green"
    authoritative = flash if flash_green else fallback_pro
    selection_fallback = not flash_green
    enrichment_fallback = bool(flash.get("fallback_used"))
    calls = list(flash["calls"]) + (
        list(fallback_pro["calls"]) if selection_fallback else [])
    fallback_reasons = list(flash.get("fallback_reasons") or [])
    if selection_fallback:
        fallback_reasons.append(f"flash_{flash['classification']}")
        fallback_reasons.extend(flash.get("classification_reasons") or [])
    fallback_reasons = list(dict.fromkeys(str(reason) for reason in fallback_reasons if reason))
    latency = float(flash["latency_ms"]) \
        + (float(fallback_pro["latency_ms"]) if selection_fallback else 0.0)
    contributors = [flash, *([fallback_pro] if selection_fallback else [])]
    cost_fields = _row_cost_fields(calls, snapshot)
    if any(not isinstance(row.get("usage"), dict)
           or row["usage"].get("complete") is not True for row in contributors):
        cost_fields.update({
            "usage": {
                "prompt_tokens": None, "candidate_tokens": None,
                "thought_tokens": None, "total_tokens": None, "complete": False,
            },
            "cost_usd": None,
            "cost_error": "incomplete_call_telemetry",
        })
    return {
        **identity,
        "status": authoritative["status"],
        "route": "hybrid_pro_fallback" if selection_fallback else "hybrid_flash",
        # The classification is the Flash router decision; authoritative quality remains in clips.
        "classification": flash["classification"],
        "classification_reasons": list(flash.get("classification_reasons") or []),
        "authoritative_profile": authoritative["profile"],
        "flash_profile": flash["profile"],
        "flash_latency_ms": flash["latency_ms"],
        "fallback_used": selection_fallback or enrichment_fallback,
        "fallback_reasons": fallback_reasons,
        "proposed_count": authoritative["proposed_count"],
        "accepted_count": authoritative["accepted_count"],
        "clips": copy.deepcopy(authoritative["clips"]),
        "latency_ms": round(latency, 3),
        "notes": authoritative.get("notes", ""),
        "error": authoritative.get("error"),
        **cost_fields,
    }


def synthesize_hybrid_rows(real_rows: Iterable[dict], *, flash_profile: str,
                           pro_profile: str,
                           pricing_path: Path = PRICING_SNAPSHOT_PATH) -> list[dict]:
    """Build hybrid rows after real profiles are complete, without another model call."""
    if flash_profile not in {PROFILE_FLASH_SINGLE, PROFILE_FLASH_SPLIT}:
        raise ValueError("flash_profile must be flash_single_v1 or flash_split_v2")
    if pro_profile not in {PROFILE_PRODUCTION_PRO, PROFILE_CORRECTED_PRO}:
        raise ValueError("pro_profile must be production_pro_v0 or corrected_pro_v1")
    indexed: dict[tuple[str, int, str], dict] = {}
    for index, row in enumerate(real_rows):
        if not isinstance(row, dict):
            raise ValueError(f"real_rows[{index}] must be an object")
        if row.get("profile") not in {flash_profile, pro_profile}:
            continue
        key = (str(row.get("pair_id") or ""), row.get("repeat"), str(row["profile"]))
        if not key[0] or isinstance(key[1], bool) or not isinstance(key[1], int):
            raise ValueError(f"real_rows[{index}] has invalid trial identity")
        if key in indexed:
            raise ValueError(f"duplicate real profile row {key}")
        indexed[key] = row
    flash_keys = {(pair, repeat) for pair, repeat, profile in indexed
                  if profile == flash_profile}
    pro_keys = {(pair, repeat) for pair, repeat, profile in indexed if profile == pro_profile}
    if not flash_keys or flash_keys != pro_keys:
        raise ValueError("Flash and Pro real rows must cover the same trials")
    if pro_profile == PROFILE_PRODUCTION_PRO and any(
            indexed[(pair, repeat, pro_profile)].get("transport_provenance")
            != "historical_frozen_transport" for pair, repeat in pro_keys):
        raise ValueError(
            "production_pro_v0 hybrid fallback requires historical_frozen_transport rows")

    snapshot = load_pricing_snapshot(pricing_path)
    rows: list[dict] = []
    identity_fields = (
        "schema_version", "dataset_version", "postprocess_version", "pair_id", "video_id",
        "topic", "stratum", "positive_content", "repeat",
    )
    for pair_id, repeat in sorted(flash_keys):
        flash = indexed[(pair_id, repeat, flash_profile)]
        pro = indexed[(pair_id, repeat, pro_profile)]
        for field in (
            "dataset_version", "postprocess_version", "video_id", "topic", "stratum",
            "positive_content",
        ):
            if flash.get(field) != pro.get(field):
                raise ValueError(f"paired real rows {(pair_id, repeat)} disagree on {field}")
        missing = [field for field in identity_fields if field not in flash]
        if missing:
            raise ValueError(f"Flash row {(pair_id, repeat)} missing fields: {missing}")
        identity = {field: flash[field] for field in identity_fields}
        identity.update({
            "profile": PROFILE_SIMULATED_HYBRID,
            "transport_provenance": "simulated_hybrid_transport",
        })
        rows.append(_hybrid_row(identity, flash, pro, snapshot))
    return rows


ProfileRunner = Callable[..., object]


def run_benchmark(manifest_path: Path, out_path: Path, *, flash_winner: str | None = None,
                  pro_baseline: str | None = None,
                  run_profile: ProfileRunner | None = None, settings: dict | None = None,
                  require_tracked: bool = True,
                  pricing_path: Path = PRICING_SNAPSHOT_PATH) -> list[dict]:
    """Execute 300 real calls; optionally synthesize a preselected corrected-Pro hybrid."""
    if flash_winner is None and pro_baseline is not None:
        raise ValueError("pro_baseline cannot be selected before flash_winner")
    winner_profile = None
    fallback_profile = None
    if flash_winner is not None:
        pro_baseline = pro_baseline or "corrected"
        winner_profile = {
            "single": PROFILE_FLASH_SINGLE,
            "split": PROFILE_FLASH_SPLIT,
        }.get(str(flash_winner).lower())
        if winner_profile is None:
            raise ValueError("flash_winner must be 'single' or 'split'")
        fallback_profile = {
            "production": PROFILE_PRODUCTION_PRO,
            "corrected": PROFILE_CORRECTED_PRO,
        }.get(str(pro_baseline).lower())
        if fallback_profile is None:
            raise ValueError("pro_baseline must be 'production' or 'corrected'")
        if fallback_profile == PROFILE_PRODUCTION_PRO:
            raise ValueError(
                "in-process production_pro_v0 is not a historical production control; "
                "use synthesize_hybrid_rows with frozen historical rows")
    manifest, transcripts = load_verified_case_manifest(
        manifest_path, require_tracked=require_tracked)
    snapshot = load_pricing_snapshot(pricing_path)
    if run_profile is None:
        from ..pipeline.gemini_segment import run_segment_profile as run_profile

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    base_settings = dict(settings or {})
    with out.open("w", encoding="utf-8", buffering=1) as stream:
        for case in manifest["cases"]:
            frozen = transcripts[str(case["pair_id"])]
            for repeat in range(1, EXPECTED_REPEATS + 1):
                real_rows: dict[str, dict] = {}
                for profile in REAL_PROFILES:
                    identity = _identity(
                        case, repeat, profile, manifest["dataset_version"],
                        manifest["postprocess_version"],
                    )
                    started = time.perf_counter()
                    try:
                        result = run_profile(
                            copy.deepcopy(frozen), dict(base_settings), profile,
                            topic=str(case["topic"]),
                        )
                        elapsed = (time.perf_counter() - started) * 1000.0
                        row = _profile_row(identity, result, elapsed, snapshot)
                    except Exception as exc:  # noqa: BLE001 - error rows are mandatory
                        elapsed = (time.perf_counter() - started) * 1000.0
                        row = _exception_row(identity, exc, elapsed, snapshot)
                    real_rows[profile] = row
                    rows.append(row)
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                    stream.flush()

                if winner_profile is None or fallback_profile is None:
                    continue
                hybrid_identity = _identity(
                    case, repeat, PROFILE_SIMULATED_HYBRID, manifest["dataset_version"],
                    manifest["postprocess_version"],
                )
                hybrid_identity["transport_provenance"] = "simulated_hybrid_transport"
                try:
                    hybrid = _hybrid_row(
                        hybrid_identity, real_rows[winner_profile],
                        real_rows[fallback_profile], snapshot,
                    )
                except Exception as exc:  # noqa: BLE001 - the fifth row is mandatory too
                    hybrid = _exception_row(hybrid_identity, exc, 0.0, snapshot)
                    hybrid.update({
                        "route": "hybrid_report_error",
                        "flash_profile": winner_profile,
                        "authoritative_profile": None,
                        "fallback_used": True,
                        "fallback_reasons": ["hybrid_synthesis_failure"],
                    })
                rows.append(hybrid)
                stream.write(json.dumps(hybrid, sort_keys=True) + "\n")
                stream.flush()

    expected_profiles = BENCHMARK_PROFILES if winner_profile is not None else REAL_PROFILES
    errors = validate_trial_rows(rows, manifest, profiles=expected_profiles)
    if errors:
        raise RuntimeError("benchmark produced an invalid trial matrix: " + "; ".join(errors))
    return rows


def parse_benchmark_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m backend.eval.segment_benchmark",
        description="Run the verified 25-case Gemini segment benchmark.",
    )
    parser.add_argument("--manifest", type=Path, required=True,
                        help="tracked 25-case manifest with frozen transcript hashes")
    parser.add_argument("--out", type=Path, required=True,
                        help="JSONL destination (one row per profile/pair/repeat)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_benchmark_args(list(sys.argv[1:] if argv is None else argv))
    rows = run_benchmark(args.manifest, args.out, require_tracked=True)
    counts = {status: sum(1 for row in rows if row["status"] == status)
              for status in ("ok", "zero", "error")}
    print(f"wrote {len(rows)} benchmark rows to {args.out} ({counts})")


if __name__ == "__main__":
    main()
