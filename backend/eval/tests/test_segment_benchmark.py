from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from backend.eval import segment_benchmark as B


def _manifest(transcript_path="transcripts/shared.json", digest="0" * 64):
    strata = sorted(B.CONTENT_STRATA)
    return {
        "dataset_version": "segment-v1",
        "postprocess_version": "strict-v1",
        "cases": [
            {
                "pair_id": f"pair-{i:02d}",
                "video_id": f"video-{i:02d}",
                "topic": f"topic {i}",
                "stratum": strata[i % len(strata)],
                "positive_content": strata[i % len(strata)] != "comedy_negative",
                "duration_s": 120.0 + i,
                "transcript_path": transcript_path,
                "transcript_sha256": digest,
            }
            for i in range(B.EXPECTED_PAIR_COUNT)
        ],
    }


def test_pricing_snapshot_and_thought_billing_are_pinned():
    snapshot = B.load_pricing_snapshot()
    assert snapshot["pricing_version"] == "gemini-standard-2026-07-11"
    assert snapshot["models"]["gemini-3.5-flash"] == {
        "input": 1.5, "output_including_thoughts": 9.0}
    usage = {"prompt_token_count": 1_000, "candidates_token_count": 100,
             "thoughts_token_count": 50, "total_token_count": 1_150}
    assert B.usage_cost_usd("gemini-3.5-flash", usage, snapshot) == pytest.approx(0.00285)
    assert B.usage_cost_usd("gemini-3.1-pro-preview", usage, snapshot) == pytest.approx(0.0038)


def test_pricing_fails_closed_for_unknown_counts_model_and_uncaptured_tier():
    usage = {"prompt_token_count": -1, "candidates_token_count": 0,
             "thoughts_token_count": 0, "total_token_count": 0}
    with pytest.raises(ValueError, match="non-negative"):
        B.usage_cost_usd("gemini-3.5-flash", usage)
    usage["prompt_token_count"] = 0
    with pytest.raises(ValueError, match="absent"):
        B.usage_cost_usd("unknown-model", usage)
    usage["prompt_token_count"] = 200_001
    with pytest.raises(ValueError, match="exceeds"):
        B.usage_cost_usd("gemini-3.5-flash", usage)


def test_manifest_validation_and_full_375_row_matrix():
    manifest = _manifest()
    assert B.validate_case_manifest(manifest) == []
    rows = B.trial_matrix(manifest)
    assert len(rows) == B.EXPECTED_TRIAL_ROWS == 375
    assert {row["profile"] for row in rows} == set(B.BENCHMARK_PROFILES)
    assert {row["repeat"] for row in rows} == {1, 2, 3}
    assert {row["postprocess_version"] for row in rows} == {"strict-v1"}
    assert len({(row["pair_id"], row["repeat"], row["profile"]) for row in rows}) == 375


def test_manifest_can_verify_frozen_transcript_hash_and_timings(tmp_path):
    path = tmp_path / "transcripts" / "shared.json"
    path.parent.mkdir()
    path.write_text(json.dumps({
        "duration": 4.0,
        "segments": [
            {"start": 0.0, "end": 2.0, "text": "first"},
            {"start": 2.0, "end": 4.0, "text": "second"},
        ],
    }))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = _manifest(digest=digest)
    assert B.validate_case_manifest(manifest, base_dir=tmp_path, verify_files=True) == []
    manifest["cases"][0]["transcript_sha256"] = "f" * 64
    assert any("mismatch" in e for e in B.validate_case_manifest(
        manifest, base_dir=tmp_path, verify_files=True))


def test_manifest_rejects_partial_or_unstratified_local_cache():
    manifest = _manifest()
    manifest["cases"] = manifest["cases"][:13]
    for case in manifest["cases"]:
        case["stratum"] = "lecture"
        case["positive_content"] = True
    errors = B.validate_case_manifest(manifest)
    assert any("expected 25" in error for error in errors)
    assert any("missing required strata" in error for error in errors)


def test_trial_row_validation_keeps_error_and_zero_rows():
    manifest = _manifest()
    rows = B.trial_matrix(manifest)
    reported = []
    for i, identity in enumerate(rows):
        status = "error" if i == 0 else "zero" if i == 1 else "ok"
        reported.append({**identity, "status": status})
    assert B.validate_trial_rows(reported, manifest) == []
    errors = B.validate_trial_rows(reported[:-1], manifest)
    assert errors == ["missing 1 trial row(s)"]


def test_trial_rows_write_as_jsonl_without_dropping_error_or_zero(tmp_path):
    manifest = _manifest()
    identities = B.trial_matrix(manifest)[:3]
    rows = [{**identity, "status": status}
            for identity, status in zip(identities, ("ok", "zero", "error"))]
    path = B.write_trial_rows_jsonl(rows, tmp_path / "results" / "rows.jsonl")
    loaded = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["status"] for row in loaded] == ["ok", "zero", "error"]


def _write_verified_dataset(tmp_path):
    transcript_path = tmp_path / "transcripts" / "shared.json"
    transcript_path.parent.mkdir()
    transcript_path.write_text(json.dumps({
        "duration": 4.0,
        "segments": [
            {"start": 0.0, "end": 2.0, "text": "first idea"},
            {"start": 2.0, "end": 4.0, "text": "second idea"},
        ],
        "words": [
            {"word": "first", "start": 0.0, "end": 1.0},
            {"word": "idea", "start": 1.0, "end": 2.0},
        ],
    }))
    digest = hashlib.sha256(transcript_path.read_bytes()).hexdigest()
    manifest = _manifest(digest=digest)
    for case in manifest["cases"]:
        case["duration_s"] = 4.0
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path, transcript_path


def _profile_result(profile, *, classification="green", clips=None, error=None,
                    provider_names=False, latency=10.0):
    model = "gemini-3.5-flash" if profile.startswith("flash") else "gemini-3.1-pro-preview"
    if provider_names:
        call = {
            "model": model, "operation": profile, "prompt_version": profile,
            "thinking_level": "medium", "latency_ms": latency, "retries": 1,
            "finish_reason": "STOP", "prompt_token_count": 100,
            "candidates_token_count": 20, "thoughts_token_count": 10,
            "total_token_count": 130,
        }
    else:
        call = {
            "model": model, "operation": profile, "prompt_version": profile,
            "thinking_level": "medium", "latency_ms": latency, "retries": 0,
            "finish_reason": "STOP", "prompt_tokens": 100,
            "candidate_tokens": 20, "thought_tokens": 10, "total_tokens": 130,
        }
    clips = ([{"title": profile, "start": 0.0, "end": 2.0}]
             if clips is None else clips)
    return SimpleNamespace(
        clips=clips, notes="result", route=profile, classification=classification,
        classification_reasons=[] if classification == "green" else ["uncertain_score"],
        fallback_reasons=[], calls=[call], proposed_count=1,
        accepted_count=len(clips), error=error,
    )


def test_verified_loader_freezes_shared_transcript_once_and_checks_tracked_paths(
        tmp_path, monkeypatch):
    manifest_path, transcript_path = _write_verified_dataset(tmp_path)
    reads = []
    real_read = B._read_transcript_bytes
    monkeypatch.setattr(B, "_read_transcript_bytes",
                        lambda path: reads.append(path) or real_read(path))
    tracked = []
    monkeypatch.setattr(B, "_require_git_tracked", lambda paths: tracked.extend(paths))
    manifest, transcripts = B.load_verified_case_manifest(manifest_path, require_tracked=True)
    assert manifest["dataset_version"] == "segment-v1"
    assert len(transcripts) == 25
    assert reads == [transcript_path.resolve()]
    assert set(tracked) == {manifest_path.resolve(), transcript_path.resolve()}


def test_executable_runner_emits_375_rows_and_never_calls_hybrid(tmp_path, monkeypatch):
    manifest_path, transcript_path = _write_verified_dataset(tmp_path)
    calls = []
    per_profile_topic = {}
    transcript_reads = []
    real_read = B._read_transcript_bytes
    monkeypatch.setattr(B, "_read_transcript_bytes",
                        lambda path: transcript_reads.append(path) or real_read(path))

    def fake_runner(transcript, settings, profile, *, topic):
        assert "mutated" not in transcript             # each call sees the frozen value
        transcript["mutated"] = True
        calls.append((topic, profile))
        key = (topic, profile)
        per_profile_topic[key] = per_profile_topic.get(key, 0) + 1
        repeat = per_profile_topic[key]
        if topic == "topic 0" and profile == B.PROFILE_PRODUCTION_PRO and repeat == 1:
            raise RuntimeError("forced transport failure")
        if topic == "topic 1" and profile == B.PROFILE_CORRECTED_PRO and repeat == 1:
            return _profile_result(profile, clips=[], provider_names=True, latency=20.0)
        if profile == B.PROFILE_FLASH_SINGLE and repeat == 2:
            return _profile_result(profile, classification="uncertain", latency=10.0)
        if profile == B.PROFILE_FLASH_SINGLE and repeat == 3 and topic == "topic 0":
            return _profile_result(profile, classification="invalid", clips=[],
                                   error="forced invalid", latency=10.0)
        return _profile_result(profile, provider_names=profile.endswith("pro_v1"),
                               latency=20.0 if profile == B.PROFILE_CORRECTED_PRO else 10.0)

    out = tmp_path / "rows.jsonl"
    rows = B.run_benchmark(
        manifest_path, out, flash_winner="single", run_profile=fake_runner,
        require_tracked=False,
    )
    assert len(calls) == 25 * 3 * 4 == 300
    assert all(profile in B.REAL_PROFILES for _topic, profile in calls)
    assert len(rows) == len(out.read_text().splitlines()) == 375
    assert transcript_reads == [transcript_path.resolve()]
    assert {row["status"] for row in rows} >= {"ok", "zero", "error"}
    assert B.validate_trial_rows(rows, json.loads(manifest_path.read_text())) == []

    # Current internal and provider usage names both normalize to one stable contract.
    real = next(row for row in rows if row["profile"] == B.PROFILE_CORRECTED_PRO
                and row["status"] == "ok")
    assert real["calls"][0]["prompt_tokens"] == 100
    assert real["usage"] == {"prompt_tokens": 100, "candidate_tokens": 20,
                             "thought_tokens": 10, "total_tokens": 130,
                             "complete": True}
    assert real["cost_usd"] is not None
    assert real["cost_error"] is None
    assert real["pricing_version"] == "gemini-standard-2026-07-11"
    assert real["prompt_versions"] == [B.PROFILE_CORRECTED_PRO]


def test_hybrid_uses_same_repeat_flash_or_corrected_pro_without_extra_call(tmp_path):
    manifest_path, _transcript_path = _write_verified_dataset(tmp_path)
    counts = {}
    n_calls = 0

    def fake_runner(transcript, settings, profile, *, topic):
        nonlocal n_calls
        n_calls += 1
        key = (topic, profile)
        counts[key] = counts.get(key, 0) + 1
        repeat = counts[key]
        classification = "uncertain" if profile == B.PROFILE_FLASH_SINGLE and repeat == 2 \
            else "green"
        result = _profile_result(
            profile, classification=classification,
            latency=20.0 if profile == B.PROFILE_CORRECTED_PRO else 10.0,
        )
        result.clips[0]["title"] = f"{profile}-repeat-{repeat}"
        return result

    rows = B.run_benchmark(
        manifest_path, tmp_path / "rows.jsonl", flash_winner="single",
        pro_baseline="corrected", run_profile=fake_runner, require_tracked=False,
    )
    assert n_calls == 300
    pair_rows = [row for row in rows if row["pair_id"] == "pair-00"
                 and row["profile"] == B.PROFILE_SIMULATED_HYBRID]
    by_repeat = {row["repeat"]: row for row in pair_rows}
    assert by_repeat[1]["route"] == "hybrid_flash"
    assert by_repeat[1]["clips"][0]["title"] == "flash_single_v1-repeat-1"
    assert by_repeat[1]["latency_ms"] == 10.0
    assert len(by_repeat[1]["calls"]) == 1
    assert by_repeat[2]["route"] == "hybrid_pro_fallback"
    assert by_repeat[2]["clips"][0]["title"] == "corrected_pro_v1-repeat-2"
    assert by_repeat[2]["latency_ms"] == 30.0
    assert len(by_repeat[2]["calls"]) == 2
    assert by_repeat[2]["fallback_used"] is True
    assert "flash_uncertain" in by_repeat[2]["fallback_reasons"]


def test_split_winner_selects_split_row(tmp_path):
    manifest_path, _ = _write_verified_dataset(tmp_path)

    def fake_runner(transcript, settings, profile, *, topic):
        return _profile_result(profile)

    rows = B.run_benchmark(
        manifest_path, tmp_path / "rows.jsonl", flash_winner="split",
        run_profile=fake_runner, require_tracked=False,
    )
    hybrid = next(row for row in rows if row["profile"] == B.PROFILE_SIMULATED_HYBRID)
    assert hybrid["flash_profile"] == B.PROFILE_FLASH_SPLIT
    assert hybrid["authoritative_profile"] == B.PROFILE_FLASH_SPLIT


def test_real_profiles_can_run_before_deterministic_hybrid_selection(tmp_path):
    manifest_path, _ = _write_verified_dataset(tmp_path)
    calls = 0

    def fake_runner(transcript, settings, profile, *, topic):
        nonlocal calls
        calls += 1
        return _profile_result(profile)

    real_rows = B.run_benchmark(
        manifest_path, tmp_path / "real.jsonl", run_profile=fake_runner,
        require_tracked=False)
    assert calls == 300
    assert len(real_rows) == 300
    assert {row["profile"] for row in real_rows} == set(B.REAL_PROFILES)

    hybrid = B.synthesize_hybrid_rows(
        real_rows, flash_profile=B.PROFILE_FLASH_SINGLE,
        pro_profile=B.PROFILE_CORRECTED_PRO)
    assert calls == 300
    assert len(hybrid) == 75
    assert {row["transport_provenance"] for row in hybrid} \
        == {"simulated_hybrid_transport"}


def test_live_production_prompt_profile_cannot_claim_historical_control(tmp_path):
    manifest_path, _ = _write_verified_dataset(tmp_path)
    real_rows = B.run_benchmark(
        manifest_path, tmp_path / "real.jsonl",
        run_profile=lambda _t, _s, profile, **_k: _profile_result(profile),
        require_tracked=False)
    with pytest.raises(ValueError, match="historical production control"):
        B.run_benchmark(
            manifest_path, tmp_path / "invalid.jsonl", flash_winner="single",
            pro_baseline="production",
            run_profile=lambda _t, _s, profile, **_k: _profile_result(profile),
            require_tracked=False)
    with pytest.raises(ValueError, match="historical_frozen_transport"):
        B.synthesize_hybrid_rows(
            real_rows, flash_profile=B.PROFILE_FLASH_SINGLE,
            pro_profile=B.PROFILE_PRODUCTION_PRO)


def test_hybrid_preserves_green_split_enrichment_fallback_telemetry(tmp_path):
    manifest_path, _ = _write_verified_dataset(tmp_path)

    def fake_runner(transcript, settings, profile, *, topic):
        result = _profile_result(profile)
        if profile == B.PROFILE_FLASH_SPLIT:
            result.fallback_reasons = ["invalid_enrichment:clip-1"]
        return result

    rows = B.run_benchmark(
        manifest_path, tmp_path / "rows.jsonl", flash_winner="split",
        run_profile=fake_runner, require_tracked=False,
    )
    hybrid = next(row for row in rows if row["profile"] == B.PROFILE_SIMULATED_HYBRID)
    assert hybrid["route"] == "hybrid_flash"
    assert hybrid["fallback_used"] is True
    assert hybrid["fallback_reasons"] == ["invalid_enrichment:clip-1"]


def test_hybrid_synthesis_failure_still_emits_all_fifth_rows(tmp_path, monkeypatch):
    manifest_path, _ = _write_verified_dataset(tmp_path)
    monkeypatch.setattr(B, "_hybrid_row", lambda *_a, **_k: (_ for _ in ()).throw(
        RuntimeError("report failure")))

    rows = B.run_benchmark(
        manifest_path, tmp_path / "rows.jsonl", flash_winner="single",
        run_profile=lambda _t, _s, profile, **_k: _profile_result(profile),
        require_tracked=False,
    )
    hybrid = [row for row in rows if row["profile"] == B.PROFILE_SIMULATED_HYBRID]
    assert len(hybrid) == 75
    assert all(row["status"] == "error" and row["route"] == "hybrid_report_error"
               for row in hybrid)


def test_bad_hash_blocks_execution_before_output(tmp_path):
    manifest_path, _ = _write_verified_dataset(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["cases"][0]["transcript_sha256"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest))
    out = tmp_path / "rows.jsonl"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        B.run_benchmark(
            manifest_path, out, flash_winner="single",
            run_profile=lambda *_a, **_k: None, require_tracked=False,
        )
    assert not out.exists()


def test_cli_runs_real_profiles_without_preselecting_winner():
    with pytest.raises(SystemExit):
        B.parse_benchmark_args([])
    args = B.parse_benchmark_args([
        "--manifest", "dataset/manifest.json",
        "--out", "results.jsonl",
    ])
    assert not hasattr(args, "flash_winner")
    assert not hasattr(args, "pro_baseline")
    assert args.out.name == "results.jsonl"


def test_empty_or_missing_telemetry_never_prices_as_zero_complete_usage():
    usage = B.aggregate_usage([])
    assert usage == {
        "prompt_tokens": None, "candidate_tokens": None, "thought_tokens": None,
        "total_tokens": None, "complete": False,
    }
    assert B.calls_cost_usd([]) is None
    identity = {
        "schema_version": 1, "dataset_version": "segment-v1",
        "postprocess_version": "strict-v1",
        "transport_provenance": "in_process_gemini3_transport",
        "pair_id": "pair", "video_id": "video", "topic": "topic",
        "stratum": "lecture", "positive_content": True, "repeat": 1,
        "profile": B.PROFILE_FLASH_SINGLE,
    }
    row = B._exception_row(identity, RuntimeError("failed"), 1.0, B.load_pricing_snapshot())
    assert row["usage"]["complete"] is False
    assert row["cost_usd"] is None

    flash_identity = {**identity, "profile": B.PROFILE_FLASH_SINGLE}
    flash = B._profile_row(
        flash_identity, _profile_result(B.PROFILE_FLASH_SINGLE, classification="uncertain"),
        1.0, B.load_pricing_snapshot())
    pro_identity = {**identity, "profile": B.PROFILE_CORRECTED_PRO}
    failed_pro = B._exception_row(
        pro_identity, RuntimeError("no telemetry"), 1.0, B.load_pricing_snapshot())
    hybrid_identity = {
        **identity, "profile": B.PROFILE_SIMULATED_HYBRID,
        "transport_provenance": "simulated_hybrid_transport",
    }
    hybrid = B._hybrid_row(
        hybrid_identity, flash, failed_pro, B.load_pricing_snapshot())
    assert hybrid["usage"]["complete"] is False
    assert hybrid["cost_usd"] is None
    assert hybrid["cost_error"] == "incomplete_call_telemetry"
