from datetime import datetime, timedelta, timezone

import pytest

from backend.app.clip_engine import config as clip_engine_config
from backend.app.clip_engine.errors import ProviderBudgetExceededError
from backend.app.clip_engine.provider_cache import (
    MemoryProviderCache,
    TRANSCRIPT_PROFILE,
    TRANSCRIPT_SCHEMA_VERSION,
    TranscriptArtifact,
    normalize_filters,
    search_cache_key,
    transcript_artifact_key,
    validate_transcript_payload,
)
from backend.app.clip_engine.provider_runtime import (
    GenerationBudget,
    GenerationContext,
    bounded_retry_after,
)

VIDEO_ID = "dQw4w9WgXcQ"


def _artifact() -> TranscriptArtifact:
    created_at = datetime.now(timezone.utc).isoformat()
    key = transcript_artifact_key(
        video_id=VIDEO_ID,
        provider="supadata",
        requested_language="en",
        returned_language="en-US",
        native_mode=True,
    )
    return TranscriptArtifact(
        artifact_key=key,
        video_id=VIDEO_ID,
        provider="supadata",
        requested_language="en",
        returned_language="en-us",
        native_mode=True,
        schema_version=TRANSCRIPT_SCHEMA_VERSION,
        segments=[
            {"cue_id": "c0", "start": 0.0, "end": 1.5, "text": "one", "lang": "en"},
            {"cue_id": "c1", "start": 1.5, "end": 3.0, "text": "two", "lang": "en"},
        ],
        duration_sec=3.0,
        created_at=created_at,
    )


def test_transcript_cache_version_invalidates_coarser_cue_artifacts() -> None:
    assert TRANSCRIPT_SCHEMA_VERSION == 4
    assert TRANSCRIPT_PROFILE == f"chunk{clip_engine_config.SUPADATA_CHUNK_SIZE}-auto"
    assert _artifact().artifact_key.startswith(
        f"native-transcript:v4:{TRANSCRIPT_PROFILE}:"
    )


def test_generation_budgets_match_fast_and_slow_contracts() -> None:
    fast = GenerationBudget.for_mode("fast")
    slow = GenerationBudget.for_mode("slow")
    assert fast.snapshot()["limits"] == {"search": 3, "transcript": 2, "segmentation": 2}
    assert (fast.max_passes, fast.max_no_growth_passes) == (1, 0)
    assert slow.snapshot()["limits"] == {"search": 4, "transcript": 3, "segmentation": 3}
    assert (slow.max_passes, slow.max_no_growth_passes) == (1, 0)
    for _ in range(3):
        fast.reserve("search")
    with pytest.raises(ProviderBudgetExceededError):
        fast.reserve("search")
    fast.reserve_pass()
    with pytest.raises(ProviderBudgetExceededError):
        fast.reserve_pass()


def test_retry_after_is_parsed_and_bounded() -> None:
    assert bounded_retry_after({"Retry-After": "2.5"}) == 2.5
    assert bounded_retry_after({"retry-after": "999"}) == 30.0
    assert bounded_retry_after({"retry-after": "invalid"}) is None


def test_generation_context_records_billable_headers_and_model_usage() -> None:
    context = GenerationContext("slow", generation_id="job-1")
    context.record_http(
        provider="supadata", operation="search", attempt=1, status_code=200,
        headers={"x-billable-requests": "2"},
    )
    context.record_gemini(
        attempt=1,
        model_used="gemini-primary",
        quality_degraded=False,
        usage={"promptTokenCount": 10, "candidatesTokenCount": 4, "totalTokenCount": 14},
    )
    rows = context.usage()
    assert rows[0]["billable_requests"] == 2
    assert rows[1]["input_tokens"] == 10
    assert rows[1]["output_tokens"] == 4
    assert rows[1]["total_tokens"] == 14
    assert rows[1]["model_used"] == "gemini-primary"


def test_generation_context_maps_live_gemini_telemetry_and_cache_hits() -> None:
    context = GenerationContext("slow", generation_id="job-2")
    context.record_gemini(
        operation="expansion",
        attempt=1,
        model_used="gemini-3.5-flash",
        quality_degraded=False,
        usage={
            "video_id": "source-video-123",
            "prompt_tokens": 10,
            "candidate_tokens": 4,
            "thought_tokens": 3,
            "total_tokens": 17,
            "latency_ms": 125.5,
            "provider_error_type": "ReadError",
            "provider_status_code": 503,
            "retryable": True,
        },
    )
    context.increment_counter("segmentation_cache_hits")
    context.record_cache_hit(provider="gemini", operation="segmentation")

    provider_call, cache_hit = context.usage()
    assert provider_call["operation"] == "expansion"
    assert provider_call["input_tokens"] == 10
    assert provider_call["output_tokens"] == 7
    assert provider_call["total_tokens"] == 17
    assert provider_call["metadata"]["candidate_tokens"] == 4
    assert provider_call["metadata"]["thought_tokens"] == 3
    assert provider_call["metadata"]["video_id"] == "source-video-123"
    assert provider_call["metadata"]["latency_ms"] == 125.5
    assert provider_call["metadata"]["provider_error_type"] == "ReadError"
    assert provider_call["metadata"]["provider_status_code"] == 503
    assert provider_call["metadata"]["retryable"] is True
    assert cache_hit["billable_requests"] == 0
    assert cache_hit["metadata"] == {"provider_call": False, "cache_hit": True}
    assert context.counters()["segmentation_cache_hits"] == 1


def test_generation_context_enforces_actual_gemini_call_and_cost_budgets() -> None:
    context = GenerationContext("fast", generation_id="job-budget")
    for _ in range(2):
        context.reserve_gemini_call(
            operation="flash_boundary_selector",
            model="gemini-3.5-flash",
            prompt_text="short transcript",
            max_output_tokens=4096,
        )
    with pytest.raises(ProviderBudgetExceededError):
        context.reserve_gemini_call(
            operation="flash_boundary_selector",
            model="gemini-3.5-flash",
            prompt_text="another transcript",
            max_output_tokens=4096,
        )

    with pytest.raises(ProviderBudgetExceededError):
        context.reserve_gemini_call(
            operation="pro_authoritative",
            model="gemini-3.1-pro-preview",
            prompt_text="transcript",
            max_output_tokens=4096,
        )

    budget = context.budget.snapshot()["gemini"]
    assert budget["flash_selector_calls"] == 2
    assert budget["flash_selector_limit"] == 2
    assert budget["pro_fallback_calls"] == 0
    assert budget["pro_fallback_call_limit"] == 0

    slow_flash = GenerationContext("slow", generation_id="job-slow-flash-budget")
    for _ in range(3):
        slow_flash.reserve_gemini_call(
            operation="flash_boundary_selector",
            model="gemini-3.5-flash",
            prompt_text="short transcript",
            max_output_tokens=8192,
        )
    with pytest.raises(ProviderBudgetExceededError):
        slow_flash.reserve_gemini_call(
            operation="flash_boundary_selector",
            model="gemini-3.5-flash",
            prompt_text="fourth transcript",
            max_output_tokens=8192,
        )
    slow_budget = slow_flash.budget.snapshot()["gemini"]
    assert slow_budget["flash_selector_calls"] == 3
    assert slow_budget["flash_selector_limit"] == 3
    assert slow_budget["cost_limit_usd"] == pytest.approx(0.55)


@pytest.mark.parametrize(
    ("mode", "selector_count", "cost_limit"),
    [("fast", 2, 0.38), ("slow", 3, 0.55)],
)
def test_job_cost_budget_fits_expansion_and_realistic_long_transcript_selectors(
    mode: str,
    selector_count: int,
    cost_limit: float,
) -> None:
    context = GenerationContext(mode, generation_id=f"job-long-{mode}")
    context.reserve_gemini_call(
        operation="expansion",
        model="gemini-3.1-flash-lite",
        estimated_input_tokens=1_000,
        max_output_tokens=1_024,
    )
    for _ in range(selector_count):
        context.reserve_gemini_call(
            operation="flash_boundary_selector",
            model="gemini-3.5-flash",
            estimated_input_tokens=40_000,
            max_output_tokens=12_288,
        )

    budget = context.budget.snapshot()["gemini"]
    expected_reserved_cost = (
        1_000 * 0.25
        + 1_024 * 1.5
        + selector_count * (40_000 * 1.5 + 12_288 * 9.0)
    ) / 1_000_000.0
    assert budget["flash_selector_calls"] == selector_count
    assert budget["reserved_cost_usd"] == pytest.approx(expected_reserved_cost)
    assert budget["reserved_cost_usd"] <= cost_limit
    assert budget["cost_limit_usd"] == pytest.approx(cost_limit)


def test_flash_lite_expansion_uses_its_lower_reservation_and_usage_rates() -> None:
    context = GenerationContext("fast", generation_id="job-lite-pricing")
    reservation = context.reserve_gemini_call(
        operation="expansion",
        model="gemini-3.1-flash-lite",
        estimated_input_tokens=1_000,
        max_output_tokens=100,
    )
    context.record_gemini(
        attempt=1,
        model_used="gemini-3.1-flash-lite",
        quality_degraded=False,
        stage="expansion",
        usage={
            "prompt_tokens": 1_000,
            "candidate_tokens": 100,
            "cached_content_token_count": 400,
            "total_tokens": 1_100,
        },
    )

    reserved = (1_000 * 0.25 + 100 * 1.5) / 1_000_000.0
    actual = (600 * 0.25 + 400 * 0.025 + 100 * 1.5) / 1_000_000.0
    assert reservation["reserved_cost_usd"] == pytest.approx(reserved)
    assert context.usage_payload()["summary"]["estimated_cost_usd"] == pytest.approx(
        actual
    )
    assert context.usage_payload()["summary"]["cached_tokens"] == 400
    assert context.usage_payload()["by_stage"]["expansion"]["cached_tokens"] == 400


@pytest.mark.parametrize("mode", ["fast", "slow"])
def test_pro_dispatch_is_disabled_for_every_generation_mode(mode: str) -> None:
    context = GenerationContext(mode, generation_id=f"job-no-pro-{mode}")

    for operation in ("pro_authoritative", "pro_fallback"):
        with pytest.raises(ProviderBudgetExceededError, match="Pro"):
            context.reserve_gemini_call(
                operation=operation,
                model="gemini-3.1-pro-preview",
                prompt_text="transcript",
                max_output_tokens=4096,
            )

    budget = context.budget.snapshot()["gemini"]
    assert budget["pro_calls"] == 0
    assert budget["pro_call_limit"] == 0


def test_generation_usage_payload_aggregates_stage_tokens_cost_and_fallbacks() -> None:
    context = GenerationContext("fast", generation_id="job-summary")
    context.record_gemini(
        attempt=1,
        model_used="gemini-3.5-flash",
        quality_degraded=False,
        stage="selection",
        usage={
            "prompt_tokens": 100,
            "candidate_tokens": 20,
            "thought_tokens": 10,
            "total_tokens": 130,
        },
    )
    context.record_segment_event({"event": "pro_fallback", "reason": "invalid_edges"})
    context.record_segment_event({
        "event": "segment_completed",
        "rejection_reasons": [
            "proposal_0:bad_start_quote",
            "proposal_1:bad_start_quote",
            "request_failure:RuntimeError",
        ],
    })
    context.increment_counter("persisted_clips", 2)

    payload = context.usage_payload()
    assert payload["summary"]["gemini_calls"] == 1
    assert payload["summary"]["input_tokens"] == 100
    assert payload["summary"]["output_tokens"] == 30
    assert payload["summary"]["thought_tokens"] == 10
    assert payload["summary"]["accepted_clips"] == 2
    assert payload["summary"]["fallback_reasons"] == ["invalid_edges"]
    assert payload["summary"]["rejection_reason_counts"] == {
        "bad_start_quote": 2,
        "request_failure:RuntimeError": 1,
    }
    assert payload["by_stage"]["selection"]["calls"] == 1
    assert payload["budget"]["gemini"]["cost_limit_usd"] == 0.38


def test_one_physical_pro_fallback_counts_once_for_multiple_reasons() -> None:
    context = GenerationContext("fast")

    context.record_segment_event(
        {"event": "pro_fallback", "video_id": "video-a", "reason": "invalid_edges"}
    )
    context.record_segment_event(
        {"event": "pro_fallback", "video_id": "video-a", "reason": "low_confidence"}
    )

    assert context.counters()["pro_fallbacks"] == 1
    assert context.usage_payload()["summary"]["fallback_reasons"] == [
        "invalid_edges",
        "low_confidence",
    ]


def test_generation_context_usage_persistence_is_fail_open() -> None:
    def unavailable_sink(_record) -> None:
        raise RuntimeError("database unavailable")

    context = GenerationContext("slow", usage_sink=unavailable_sink)
    context.record_cache_hit(provider="gemini", operation="segmentation")

    assert len(context.usage()) == 1
    assert context.usage()[0]["metadata"]["cache_hit"] is True


def test_search_cache_key_normalizes_query_filters_and_language_but_not_page() -> None:
    filters = {"preferred_video_duration": "4-20m", "creative_commons_only": True}
    first = search_cache_key(
        query="  CAFÉ  calculus ", filters=filters, language="EN_us", page_token="p1"
    )
    same = search_cache_key(
        query="café calculus", filters=filters, language="en-US", page_token="p1"
    )
    next_page = search_cache_key(
        query="café calculus", filters=filters, language="en-US", page_token="p2"
    )
    assert first == same
    assert first != next_page
    assert normalize_filters(filters) == {
        "duration": "medium",
        "features": ["creative-commons"],
        "sort_by": "relevance",
        "upload_date": "all",
    }
    assert normalize_filters({"creative_commons_only": "false"})["features"] == []
    assert normalize_filters({"features": ["subtitles"]})["features"] == []
    assert normalize_filters({"features": ["hd", "not-a-real-feature"]})["features"] == ["hd"]
    assert normalize_filters({"sort_by": "viewCount"})["sort_by"] == "views"


def test_search_cache_uses_shorter_empty_ttl_and_filters_tombstones_twice() -> None:
    cache = MemoryProviderCache()
    positive_key = "positive"
    empty_key = "empty"
    cache.put_search(positive_key, {"videos": [{"id": VIDEO_ID}]}, {})
    cache.put_search(empty_key, {"videos": []}, {})
    old = (datetime.now(timezone.utc) - timedelta(minutes=20)).isoformat()
    cache.search_rows[positive_key][1]["created_at"] = old
    cache.search_rows[empty_key][1]["created_at"] = old
    assert cache.get_search(positive_key) is not None
    assert cache.get_search(empty_key) is None

    cache.blocked_video_ids.add(VIDEO_ID)
    assert cache.get_search(positive_key).payload["videos"] == []
    cache.put_search("after-block", {"videos": [{"id": VIDEO_ID}]}, {})
    assert cache.search_rows["after-block"][0]["videos"] == []


def test_transcript_validation_rejects_nonfinite_and_nonmonotonic_cues() -> None:
    artifact = _artifact()
    assert validate_transcript_payload(artifact.as_payload()) == artifact
    nonfinite = artifact.as_payload()
    nonfinite["segments"][0]["start"] = float("nan")
    assert validate_transcript_payload(nonfinite) is None
    nonmonotonic = artifact.as_payload()
    nonmonotonic["segments"][1]["start"] = -1
    assert validate_transcript_payload(nonmonotonic) is None


def test_transcript_validation_accepts_auto_mode_artifact() -> None:
    payload = _artifact().as_payload()
    payload["native_mode"] = False
    payload["artifact_key"] = transcript_artifact_key(
        video_id=VIDEO_ID,
        provider="supadata",
        requested_language="en",
        returned_language="en-us",
        native_mode=False,
    )
    artifact = validate_transcript_payload(payload)
    assert artifact is not None
    assert artifact.native_mode is False


def test_transcript_cache_rejects_tombstoned_video() -> None:
    cache = MemoryProviderCache()
    artifact = _artifact()
    cache.put_transcript(artifact)
    assert cache.get_transcript(
        video_id=VIDEO_ID,
        provider="supadata",
        requested_language="en",
        native_mode=True,
        schema_version=TRANSCRIPT_SCHEMA_VERSION,
    ) == artifact
    cache.blocked_video_ids.add(VIDEO_ID)
    assert cache.get_transcript(
        video_id=VIDEO_ID,
        provider="supadata",
        requested_language="en",
        native_mode=True,
        schema_version=TRANSCRIPT_SCHEMA_VERSION,
    ) is None
