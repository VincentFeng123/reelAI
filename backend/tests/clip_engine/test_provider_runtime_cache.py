from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
import threading
import time

import pytest

from backend import config as backend_config
from backend.app.clip_engine import expand
from backend.app.clip_engine import config as clip_engine_config
from backend.app.clip_engine import provider_runtime
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
    ProviderUsageRecord,
    bounded_retry_after,
)
from backend.pipeline import gemini_segment

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
    assert TRANSCRIPT_SCHEMA_VERSION == 5
    assert TRANSCRIPT_PROFILE == f"chunk{clip_engine_config.SUPADATA_CHUNK_SIZE}-auto"
    assert _artifact().artifact_key.startswith(
        f"native-transcript:v5:{TRANSCRIPT_PROFILE}:"
    )


def test_generation_budgets_match_fast_and_slow_contracts() -> None:
    fast = GenerationBudget.for_mode("fast")
    slow = GenerationBudget.for_mode("slow")
    assert fast.snapshot()["limits"] == {
        "search": 5,
        "transcript": 3,
        "segmentation": 3,
    }
    assert (fast.max_passes, fast.max_no_growth_passes) == (1, 0)
    assert slow.snapshot()["limits"] == {"search": 5, "transcript": 5, "segmentation": 5}
    assert (slow.max_passes, slow.max_no_growth_passes) == (1, 0)
    fast_gemini = fast.snapshot()["gemini"]
    slow_gemini = slow.snapshot()["gemini"]
    assert fast_gemini["hard_cost_cap_enabled"] is False
    assert slow_gemini["hard_cost_cap_enabled"] is False
    assert fast_gemini["cost_target_usd"] == 1.50
    assert slow_gemini["cost_target_usd"] == 2.50
    assert fast_gemini["completion_cost_target_usd"] == 2.00
    assert slow_gemini["completion_cost_target_usd"] == 3.00
    assert fast_gemini["cost_limit_usd"] == fast_gemini["cost_target_usd"]
    assert slow_gemini["completion_cost_limit_usd"] == (
        slow_gemini["completion_cost_target_usd"]
    )
    for _ in range(5):
        fast.reserve("search")
    with pytest.raises(ProviderBudgetExceededError):
        fast.reserve("search")
    fast.reserve_pass()
    with pytest.raises(ProviderBudgetExceededError):
        fast.reserve_pass()


@pytest.mark.parametrize(
    ("input_tokens", "expected_rates"),
    [
        (200_000, (1.25, 0.125, 10.00)),
        (200_001, (2.50, 0.25, 15.00)),
    ],
)
def test_stable_pro_rates_follow_the_2_5_context_tiers(
    input_tokens,
    expected_rates,
) -> None:
    assert provider_runtime._gemini_token_rates(
        "gemini-2.5-pro",
        input_tokens=input_tokens,
    ) == expected_rates
    standard = gemini_segment._model_cost({
        "model": "gemini-2.5-pro",
        "prompt_tokens": input_tokens,
        "candidate_tokens": 10,
        "thought_tokens": 5,
    })
    priority = gemini_segment._model_cost({
        "model": "gemini-2.5-pro",
        "prompt_tokens": input_tokens,
        "candidate_tokens": 10,
        "thought_tokens": 5,
        "billing_cost_multiplier": (
            gemini_segment._PRO_PRIORITY_BILLING_MULTIPLIER
        ),
    })
    assert standard == pytest.approx(
        (
            input_tokens * expected_rates[0]
            + 15 * expected_rates[2]
        ) / 1_000_000.0
    )
    assert priority == pytest.approx(
        standard * gemini_segment._PRO_PRIORITY_BILLING_MULTIPLIER
    )


@pytest.mark.parametrize(
    ("model", "expected_rates"),
    [
        ("gemini-2.5-flash", (0.30, 0.03, 2.50)),
        ("models/gemini-2.5-flash-001", (0.30, 0.03, 2.50)),
        ("gemini-2.5-flash-lite", (0.10, 0.01, 0.40)),
        ("models/gemini-2.5-flash-lite-001", (0.10, 0.01, 0.40)),
    ],
)
def test_lesson_order_models_use_exact_2_5_reservation_and_usage_rates(
    model,
    expected_rates,
) -> None:
    assert provider_runtime._gemini_token_rates(model) == expected_rates
    context = GenerationContext("slow", generation_id=f"job-order-rate-{model}")
    reservation = context.reserve_gemini_call(
        operation="ordering",
        model=model,
        estimated_input_tokens=12_079,
        max_output_tokens=10_240,
    )
    context.record_gemini(
        operation="ordering",
        attempt=1,
        model_used=model,
        quality_degraded=False,
        stage="lesson_ordering",
        usage={
            **reservation,
            "prompt_tokens": 12_079,
            "candidate_tokens": 331,
            "thought_tokens": 0,
            "cached_content_token_count": 2_000,
            "total_tokens": 12_410,
        },
    )

    input_rate, cached_rate, output_rate = expected_rates
    assert reservation["reserved_cost_usd"] == pytest.approx(
        (12_079 * input_rate + 10_240 * output_rate) / 1_000_000.0
    )
    assert context.usage_payload()["summary"]["estimated_cost_usd"] == pytest.approx(
        (
            10_079 * input_rate
            + 2_000 * cached_rate
            + 331 * output_rate
        ) / 1_000_000.0
    )


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
            "error_history": ({
                "provider_error_type": "ReadTimeout",
                "provider_status_code": None,
                "retryable": True,
            },),
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
    assert provider_call["metadata"]["error_history"] == ({
        "provider_error_type": "ReadTimeout",
        "provider_status_code": None,
        "retryable": True,
    },)
    assert cache_hit["billable_requests"] == 0
    assert cache_hit["metadata"] == {"provider_call": False, "cache_hit": True}
    assert context.counters()["segmentation_cache_hits"] == 1


def test_generation_context_preserves_selector_retry_diagnostics() -> None:
    context = GenerationContext("slow", generation_id="job-selector-diagnostics")
    diagnostics = {
        "schema_rejected_count": 1,
        "schema_rejection_reasons": ["candidate_1:invalid_claim_quote"],
        "schema_retry_attempt": 2,
        "schema_retry_reason": "invalid_structured_response",
        "schema_retry_recovered": True,
        "schema_retry_exhausted": False,
        "partial_schema_retry_attempt": 2,
        "partial_schema_retry_reason": "selector_contract_rejection",
        "partial_schema_retry_recovered": False,
        "partial_schema_retry_exhausted": True,
        "partial_schema_retry_skipped": "insufficient_deadline",
        "partial_schema_retry_retained": True,
        "selector_contract_rejected_count": 1,
        "selector_contract_rejection_reasons": [
            "intent_contract_incomplete_joint_structure",
        ],
        "selector_intent_contract_error": (
            "intent_contract_incomplete_joint_structure"
        ),
        "selector_contract_retry_attempt": 2,
        "selector_contract_retry_reason": (
            "intent_contract_incomplete_joint_structure"
        ),
        "selector_contract_retry_recovered": False,
        "selector_contract_retry_exhausted": True,
        "selector_contract_retry_skipped": "insufficient_deadline",
    }
    context.record_gemini(
        attempt=2,
        model_used="gemini-3.1-pro-preview",
        quality_degraded=False,
        usage={
            "prompt_tokens": 100,
            "candidate_tokens": 20,
            "thought_tokens": 0,
            "total_tokens": 120,
            **diagnostics,
        },
    )

    metadata = context.usage_payload()["provider_calls"][0]["metadata"]
    assert {name: metadata[name] for name in diagnostics} == diagnostics


def test_generation_context_preserves_audit_retry_diagnostics() -> None:
    context = GenerationContext("slow", generation_id="job-audit-diagnostics")
    diagnostics = {
        "selector_audit_repair_count": 1,
        "selector_audit_repair_reasons": [
            "direct_objective_fulfillment_incomplete",
        ],
        "structured_retry_attempt": 2,
        "structured_retry_reason": "invalid_structured_response",
        "structured_retry_recovered": False,
        "structured_retry_exhausted": True,
        "contract_retry_attempt": 2,
        "contract_retry_reason": "invalid_audit_contract",
        "contract_retry_recovered": False,
        "contract_retry_exhausted": True,
        "audit_error_type": "ValidationError",
        "audit_contract_rejection_reasons": [
            "audit_semantic_contract_invalid",
        ],
        "audit_partial_contract_retained": True,
        "audit_partial_contract_retained_count": 1,
        "audit_partial_contract_discarded_count": 2,
    }
    context.record_gemini(
        attempt=2,
        model_used="gemini-3.1-pro-preview",
        quality_degraded=False,
        usage={
            "prompt_tokens": 100,
            "candidate_tokens": 20,
            "thought_tokens": 0,
            "total_tokens": 120,
            **diagnostics,
        },
    )

    metadata = context.usage_payload()["provider_calls"][0]["metadata"]
    assert {name: metadata[name] for name in diagnostics} == diagnostics


def test_generation_context_enforces_actual_gemini_call_and_cost_budgets() -> None:
    context = GenerationContext("fast", generation_id="job-budget")
    for _ in range(3):
        context.reserve_gemini_call(
            operation="flash_boundary_selector",
            model="gemini-3-flash-preview",
            prompt_text="short transcript",
            max_output_tokens=4096,
        )
    with pytest.raises(ProviderBudgetExceededError):
        context.reserve_gemini_call(
            operation="flash_boundary_selector",
            model="gemini-3-flash-preview",
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
    assert budget["flash_selector_calls"] == 3
    assert budget["flash_selector_limit"] == 3
    assert budget["pro_fallback_calls"] == 0
    assert budget["pro_fallback_call_limit"] == 0

    slow_flash = GenerationContext("slow", generation_id="job-slow-flash-budget")
    for _ in range(5):
        slow_flash.reserve_gemini_call(
            operation="flash_boundary_selector",
            model="gemini-3-flash-preview",
            prompt_text="short transcript",
            max_output_tokens=8192,
        )
    with pytest.raises(ProviderBudgetExceededError):
        slow_flash.reserve_gemini_call(
            operation="flash_boundary_selector",
            model="gemini-3-flash-preview",
            prompt_text="sixth transcript",
            max_output_tokens=8192,
        )
    slow_budget = slow_flash.budget.snapshot()["gemini"]
    assert slow_budget["flash_selector_calls"] == 5
    assert slow_budget["flash_selector_limit"] == 5
    assert slow_budget["cost_limit_usd"] == pytest.approx(2.50)


@pytest.mark.parametrize(
    ("mode", "selector_count", "cost_limit"),
    [("fast", 3, 1.50), ("slow", 5, 2.50)],
)
def test_job_cost_budget_fits_expansion_and_typical_whole_transcript_selectors(
    mode: str,
    selector_count: int,
    cost_limit: float,
) -> None:
    context = GenerationContext(mode, generation_id=f"job-long-{mode}")
    context.reserve_gemini_call(
        operation="expansion",
        model="gemini-3.1-flash-lite",
        estimated_input_tokens=1_000,
        max_output_tokens=expand.PRACTICE_FAST_EXPAND_OUTPUT_TOKENS,
    )
    for _ in range(selector_count):
        context.reserve_gemini_call(
            operation="pro_authoritative",
            model=backend_config.SEGMENT_PRO_MODEL,
            estimated_input_tokens=(
                10_500
                + 600 * gemini_segment._LOW_RESOLUTION_VIDEO_TOKENS_PER_SECOND
            ),
            max_output_tokens=gemini_segment._BOUNDARY_OUTPUT_TOKENS,
        )

    budget = context.budget.snapshot()["gemini"]
    expected_reserved_cost = (
        1_000 * 0.25
        + expand.PRACTICE_FAST_EXPAND_OUTPUT_TOKENS * 1.5
        + selector_count * (
            (
                10_500
                + 600 * gemini_segment._LOW_RESOLUTION_VIDEO_TOKENS_PER_SECOND
            ) * 2.0
            + gemini_segment._BOUNDARY_OUTPUT_TOKENS * 12.0
        )
    ) / 1_000_000.0
    assert budget["pro_selector_calls"] == selector_count
    assert budget["reserved_cost_usd"] == pytest.approx(expected_reserved_cost)
    assert budget["lifetime_reserved_worst_case_cost_usd"] == pytest.approx(
        budget["reserved_cost_usd"]
    )
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
            **reservation,
            "prompt_tokens": 1_000,
            "candidate_tokens": 100,
            "thought_tokens": 0,
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


def test_gemini_3_6_flash_uses_exact_standard_reservation_and_usage_rates() -> None:
    model = "gemini-3.6-flash"
    assert provider_runtime._gemini_token_rates(model) == (1.50, 0.15, 7.50)
    context = GenerationContext("fast", generation_id="job-36-flash-pricing")
    reservation = context.reserve_gemini_call(
        operation="expansion",
        model=model,
        estimated_input_tokens=1_000,
        max_output_tokens=100,
    )
    context.record_gemini(
        operation="expansion",
        attempt=1,
        model_used=model,
        quality_degraded=True,
        stage="expansion",
        usage={
            **reservation,
            "prompt_tokens": 1_000,
            "candidate_tokens": 100,
            "thought_tokens": 0,
            "cached_content_token_count": 400,
            "total_tokens": 1_100,
        },
    )

    reserved = (1_000 * 1.50 + 100 * 7.50) / 1_000_000.0
    actual = (600 * 1.50 + 400 * 0.15 + 100 * 7.50) / 1_000_000.0
    assert reservation["reserved_cost_usd"] == pytest.approx(reserved)
    assert context.usage_payload()["summary"]["estimated_cost_usd"] == pytest.approx(
        actual
    )


def test_usage_payload_adds_groq_cost_without_changing_gemini_subtotal() -> None:
    context = GenerationContext("slow", generation_id="job-mixed-provider-cost")
    context.record_gemini(
        attempt=1,
        model_used="gemini-3.1-flash-lite",
        quality_degraded=False,
        stage="expansion",
        usage={
            "prompt_tokens": 1_000,
            "candidate_tokens": 100,
            "thought_tokens": 0,
            "total_tokens": 1_100,
        },
    )
    gemini_cost = (1_000 * 0.25 + 100 * 1.5) / 1_000_000.0
    groq_known_cost = 12.0 * 0.04 / 3600.0
    groq_unknown_cost = 10.0 * 0.04 / 3600.0
    timestamp = datetime.now(timezone.utc).isoformat()
    context.record(ProviderUsageRecord(
        provider="groq",
        operation="transcript",
        attempt=1,
        timestamp=timestamp,
        status_code=200,
        billable_requests=1,
        model_used="whisper-large-v3-turbo",
        metadata={
            "provider_call": True,
            "stage": "groq_boundary_asr",
            "billing_usage_known": True,
            "billing_unknown_attempts": 0,
            "actual_cost_usd": groq_known_cost,
            "audio_seconds": 12.0,
            "billed_audio_seconds": 12.0,
        },
    ))
    context.record(ProviderUsageRecord(
        provider="groq",
        operation="transcript",
        attempt=1,
        timestamp=timestamp,
        status_code=None,
        model_used="whisper-large-v3-turbo",
        error_code="provider_transient",
        metadata={
            "provider_call": True,
            "stage": "groq_boundary_asr",
            "billing_usage_known": False,
            "billing_unknown_attempts": 1,
            "billing_unknown_reserved_cost_usd": groq_unknown_cost,
            "audio_seconds": 2.0,
            "billed_audio_seconds": 10.0,
        },
    ))
    context.increment_counter("persisted_clips")

    payload = context.usage_payload()
    summary = payload["summary"]
    assert summary["gemini_known_billed_cost_usd"] == pytest.approx(gemini_cost)
    assert summary["groq_calls"] == 2
    assert summary["groq_known_billed_cost_usd"] == pytest.approx(
        groq_known_cost,
        abs=1e-8,
    )
    assert summary["known_billed_cost_usd"] == pytest.approx(
        gemini_cost + groq_known_cost,
        abs=1e-8,
    )
    assert summary["groq_billing_unknown_calls"] == 1
    assert summary["billing_unknown_calls"] == 1
    assert summary["billing_unknown_reserved_cost_usd"] == pytest.approx(
        groq_unknown_cost,
        abs=1e-8,
    )
    assert summary["cost_per_accepted_clip_usd"] is None
    stage = payload["by_stage"]["groq_boundary_asr"]
    assert stage["calls"] == 2
    assert stage["attempts"] == 2
    assert stage["audio_seconds"] == 14.0
    assert stage["billed_audio_seconds"] == 22.0
    assert stage["known_billed_cost_usd"] == pytest.approx(
        groq_known_cost,
        abs=1e-8,
    )
    assert stage["billing_unknown_attempts"] == 1


@pytest.mark.parametrize(
    ("input_tokens", "input_rate", "output_rate"),
    [(200_000, 2.0, 12.0), (200_001, 4.0, 18.0)],
)
def test_pro_usage_uses_the_documented_long_context_price_tier(
    input_tokens: int,
    input_rate: float,
    output_rate: float,
) -> None:
    context = GenerationContext("slow", generation_id=f"job-pro-price-{input_tokens}")
    context.record_gemini(
        attempt=1,
        model_used="gemini-3.1-pro-preview",
        quality_degraded=False,
        stage="selection",
        usage={
            "prompt_tokens": input_tokens,
            "candidate_tokens": 100,
            "thought_tokens": 0,
            "total_tokens": input_tokens + 100,
        },
    )

    expected = (input_tokens * input_rate + 100 * output_rate) / 1_000_000.0
    assert context.usage_payload()["summary"]["estimated_cost_usd"] == pytest.approx(
        expected
    )


def test_pro_reservation_applies_long_context_tier_without_a_dollar_gate() -> None:
    context = GenerationContext("slow", generation_id="job-pro-reserve-tier")
    reservation = context.reserve_gemini_call(
        operation="pro_authoritative",
        model="gemini-3.1-pro-preview",
        estimated_input_tokens=200_000,
        max_output_tokens=100,
    )

    assert reservation["reserved_cost_usd"] == pytest.approx(
        (200_000 * 2.0 + 100 * 12.0) / 1_000_000.0
    )
    over_target = context.reserve_gemini_call(
        operation="pro_authoritative",
        model="gemini-3.1-pro-preview",
        estimated_input_tokens=600_001,
        max_output_tokens=100,
    )
    assert over_target["reserved_cost_usd"] == pytest.approx(
        (600_001 * 4.0 + 100 * 18.0) / 1_000_000.0
    )
    for _ in range(3):
        context.reserve_gemini_call(
            operation="pro_authoritative",
            model="gemini-3.1-pro-preview",
            estimated_input_tokens=1_000,
            max_output_tokens=100,
        )
    with pytest.raises(ProviderBudgetExceededError, match="selector budget"):
        context.reserve_gemini_call(
            operation="pro_authoritative",
            model="gemini-3.1-pro-preview",
            estimated_input_tokens=1_000,
            max_output_tokens=100,
        )
    budget = context.budget.snapshot()["gemini"]
    assert budget["hard_cost_cap_enabled"] is False
    assert budget["cost_exposure_usd"] > budget["cost_target_usd"]
    assert budget["pro_selector_calls"] == budget["pro_selector_limit"] == 5


def test_priority_retry_reserves_and_prices_documented_pro_premium() -> None:
    context = GenerationContext("slow", generation_id="job-pro-priority-retry")
    reservation = context.reserve_gemini_call(
        operation="pro_authoritative",
        model="gemini-3.1-pro-preview",
        estimated_input_tokens=10_000,
        max_output_tokens=1_000,
        billing_cost_multiplier=1.8,
    )
    standard_reserved = (10_000 * 2.0 + 1_000 * 12.0) / 1_000_000.0

    assert reservation["billing_cost_multiplier"] == 1.8
    assert reservation["reserved_cost_usd"] == pytest.approx(
        standard_reserved * 1.8
    )

    context.record_gemini(
        attempt=1,
        model_used="gemini-3.1-pro-preview",
        quality_degraded=False,
        stage="selection",
        usage={
            **reservation,
            "prompt_tokens": 2_000,
            "candidate_tokens": 100,
            "thought_tokens": 50,
            "total_tokens": 2_150,
            "service_tier_requested": "priority",
            "service_tier_used": "priority",
            "dispatched": True,
        },
    )

    expected = (2_000 * 2.0 + 150 * 12.0) / 1_000_000.0 * 1.8
    payload = context.usage_payload()
    assert payload["summary"]["estimated_cost_usd"] == pytest.approx(expected)
    assert payload["provider_calls"][0]["metadata"][
        "service_tier_used"
    ] == "priority"


def test_current_cost_diagnostics_do_not_report_lifetime_reservations_as_spend() -> None:
    context = GenerationContext("slow", generation_id="job-reservation-history")

    for _ in range(3):
        reservation = context.reserve_gemini_call(
            operation="pro_authoritative",
            model="gemini-3.1-pro-preview",
            estimated_input_tokens=95_000,
            max_output_tokens=6_000,
        )
        context.record_gemini(
            attempt=1,
            model_used="gemini-3.1-pro-preview",
            quality_degraded=False,
            stage="selection",
            usage={
                **reservation,
                "prompt_tokens": 10_000,
                "candidate_tokens": 500,
                "thought_tokens": 500,
                "total_tokens": 11_000,
                "dispatched": True,
            },
        )

    budget = context.budget.snapshot()["gemini"]
    summary = context.usage_payload()["summary"]
    assert budget["lifetime_reserved_worst_case_cost_usd"] == pytest.approx(0.786)
    assert budget["reserved_cost_usd"] == pytest.approx(0.786)
    assert budget["cost_exposure_usd"] == pytest.approx(0.096)
    assert summary["reserved_worst_case_cost_usd"] == pytest.approx(0.786)
    assert summary["lifetime_reserved_worst_case_cost_usd"] == pytest.approx(0.786)
    assert summary["estimated_cost_usd"] == pytest.approx(0.096)
    assert summary["current_cost_exposure_usd"] == pytest.approx(0.096)
    assert summary["cost_limit_usd"] == pytest.approx(2.50)


def test_durable_retry_restores_cost_exposure_but_reopens_selector_slots() -> None:
    first = GenerationContext("slow", generation_id="job-budget-attempt-1")
    for _ in range(3):
        reservation = first.reserve_gemini_call(
            operation="pro_authoritative",
            model="gemini-3.1-pro-preview",
            estimated_input_tokens=1_000,
            max_output_tokens=100,
            max_physical_attempts=1,
        )
        first.record_gemini(
            attempt=1,
            model_used="gemini-3.1-pro-preview",
            quality_degraded=False,
            stage="selection",
            usage={
                **reservation,
                "prompt_tokens": 100,
                "candidate_tokens": 10,
                "total_tokens": 110,
                "dispatched": True,
            },
        )

    persisted = first.budget.snapshot()
    assert persisted["gemini"]["selector_calls"] == 3

    retry = GenerationContext("slow", generation_id="job-budget-attempt-2")
    retry.budget.restore_gemini_retry_exposure(persisted)
    restored = retry.budget.snapshot()["gemini"]
    assert restored["committed_cost_usd"] == pytest.approx(
        persisted["gemini"]["cost_exposure_usd"]
    )
    assert restored["selector_calls"] == 0
    retry.reserve_gemini_call(
        operation="pro_authoritative",
        model="gemini-3.1-pro-preview",
        estimated_input_tokens=1_000,
        max_output_tokens=100,
        max_physical_attempts=1,
    )
    assert retry.budget.snapshot()["gemini"]["selector_calls"] == 1


def test_durable_retry_restores_cost_telemetry_without_closing_selector_slots() -> None:
    retry = GenerationContext("fast", generation_id="job-budget-unsettled-retry")
    retry.budget.restore_gemini_retry_exposure({
        "mode": "fast",
        "gemini": {
            "committed_cost_usd": 0.1,
            "cost_exposure_usd": 0.9,
            "inflight_reserved_cost_usd": 0.8,
            "billing_unknown_cost_exposure_usd": 0.02,
        },
    })

    restored = retry.budget.snapshot()["gemini"]
    assert restored["committed_cost_usd"] == pytest.approx(0.9)
    assert restored["billing_unknown_cost_exposure_usd"] == pytest.approx(0.82)
    retry.reserve_gemini_call(
        operation="pro_authoritative",
        model="gemini-3.1-pro-preview",
        estimated_input_tokens=250_001,
        max_output_tokens=6_000,
        max_physical_attempts=1,
    )
    for _ in range(2):
        retry.reserve_gemini_call(
            operation="pro_authoritative",
            model="gemini-3.1-pro-preview",
            estimated_input_tokens=1_000,
            max_output_tokens=100,
            max_physical_attempts=1,
        )
    with pytest.raises(ProviderBudgetExceededError, match="selector budget"):
        retry.reserve_gemini_call(
            operation="pro_authoritative",
            model="gemini-3.1-pro-preview",
            estimated_input_tokens=1_000,
            max_output_tokens=100,
            max_physical_attempts=1,
        )
    budget = retry.budget.snapshot()["gemini"]
    assert budget["hard_cost_cap_enabled"] is False
    assert budget["cost_exposure_usd"] > budget["cost_target_usd"]
    assert budget["selector_calls"] == budget["selector_limit"] == 3


def test_cost_targets_are_telemetry_only_while_call_ceilings_remain_hard() -> None:
    context = GenerationContext("slow", generation_id="job-completion-envelope")
    context.budget.restore_gemini_retry_exposure({
        "mode": "slow",
        "gemini": {
            "committed_cost_usd": 2.3799052,
            "cost_exposure_usd": 2.3799052,
            "billing_unknown_cost_exposure_usd": 1.246309,
            "lifetime_reserved_worst_case_cost_usd": 2.9017975,
        },
    })

    with pytest.raises(
        ProviderBudgetExceededError,
        match="requires a prior logical selector",
    ):
        context.reserve_gemini_call(
            operation="pro_authoritative",
            model="gemini-3.1-pro-preview",
            estimated_input_tokens=1_000,
            max_output_tokens=100,
            count_logical_call=False,
        )

    context.reserve_gemini_call(
        operation="pro_boundary_audit",
        model="gemini-3.1-pro-preview",
        estimated_input_tokens=13_319,
        max_output_tokens=8_256,
    )
    context.reserve_gemini_call(
        operation="pro_authoritative",
        model="gemini-3.1-pro-preview",
        estimated_input_tokens=1_000,
        max_output_tokens=100,
    )
    context.reserve_gemini_call(
        operation="pro_authoritative",
        model="gemini-3.1-pro-preview",
        estimated_input_tokens=100_000,
        max_output_tokens=100_000,
        count_logical_call=False,
    )
    context.reserve_gemini_call(
        operation="expansion",
        model="gemini-3.1-flash-lite",
        estimated_input_tokens=100_000,
        max_output_tokens=70_000,
    )

    budget = context.budget.snapshot()["gemini"]
    assert budget["hard_cost_cap_enabled"] is False
    assert budget["cost_exposure_usd"] > budget["completion_cost_target_usd"]
    assert budget["selector_calls"] == 1
    assert budget["boundary_audit_calls"] == 1

    for _ in range(4):
        context.reserve_gemini_call(
            operation="pro_authoritative",
            model="gemini-3.1-pro-preview",
            estimated_input_tokens=1_000,
            max_output_tokens=100,
        )
    with pytest.raises(ProviderBudgetExceededError, match="selector budget"):
        context.reserve_gemini_call(
            operation="pro_authoritative",
            model="gemini-3.1-pro-preview",
            estimated_input_tokens=1_000,
            max_output_tokens=100,
        )
    for _ in range(4):
        context.reserve_gemini_call(
            operation="pro_boundary_audit",
            model="gemini-3.1-pro-preview",
            estimated_input_tokens=1_000,
            max_output_tokens=100,
        )
    with pytest.raises(ProviderBudgetExceededError, match="boundary-audit budget"):
        context.reserve_gemini_call(
            operation="pro_boundary_audit",
            model="gemini-3.1-pro-preview",
            estimated_input_tokens=1_000,
            max_output_tokens=100,
        )

    bounded = context.budget.snapshot()["gemini"]
    assert bounded["selector_calls"] == bounded["selector_limit"] == 5
    assert bounded["boundary_audit_calls"] == bounded["boundary_audit_limit"] == 5


@pytest.mark.parametrize(
    ("mode", "selector_count"),
    [("fast", 3), ("slow", 5)],
)
def test_all_planned_source_selectors_fit_concurrently_at_production_cap(
    mode: str,
    selector_count: int,
) -> None:
    context = GenerationContext(mode, generation_id=f"job-reconcile-{mode}")

    def reserve():
        return context.reserve_gemini_call(
            operation="pro_authoritative",
            model="gemini-3.1-pro-preview",
            # Production selectors are transcript-only; media grounding is
            # disabled by _boundary_selector_content.
            estimated_input_tokens=10_500,
            max_output_tokens=gemini_segment._PRO_BOUNDARY_OUTPUT_TOKENS,
            deadline_monotonic=time.monotonic() + 2.0,
        )

    with ThreadPoolExecutor(max_workers=selector_count) as executor:
        reservations = [
            future.result(timeout=0.5)
            for future in [executor.submit(reserve) for _ in range(selector_count)]
        ]

    assert all(
        reservation["admitted_physical_attempts"] == 1
        for reservation in reservations
    )
    inflight_budget = context.budget.snapshot()["gemini"]
    assert inflight_budget["pro_selector_calls"] == selector_count
    assert inflight_budget["inflight_reserved_cost_usd"] == pytest.approx(
        sum(reservation["admitted_cost_usd"] for reservation in reservations)
    )
    assert (
        inflight_budget["cost_exposure_usd"]
        <= inflight_budget["cost_limit_usd"]
    )

    for reservation in reservations:
        context.record_gemini(
            attempt=1,
            model_used="gemini-3.1-pro-preview",
            quality_degraded=False,
            usage={
                **reservation,
                "prompt_tokens": 50_000,
                "candidate_tokens": 500,
                "thought_tokens": 3_000,
                "total_tokens": 53_500,
                "dispatched": True,
            },
        )

    budget = context.budget.snapshot()["gemini"]
    assert budget["pro_selector_calls"] == selector_count
    assert budget["inflight_reserved_cost_usd"] == 0.0
    assert budget["committed_cost_usd"] == pytest.approx(
        selector_count * 0.142
    )
    assert budget["cost_exposure_usd"] <= budget["cost_limit_usd"]
    summary = context.usage_payload()["summary"]
    assert summary["gemini_calls"] == selector_count
    assert summary["gemini_attempts"] == selector_count
    assert summary["billing_unknown_attempts"] == 0


def test_three_inflight_selector_first_attempts_do_not_block_first_audit() -> None:
    context = GenerationContext("slow", generation_id="job-selector-audit-admit")
    selector_tickets = [
        context.reserve_gemini_call(
            operation="pro_authoritative",
            model="gemini-3.1-pro-preview",
            estimated_input_tokens=30_000,
            max_output_tokens=gemini_segment._PRO_BOUNDARY_OUTPUT_TOKENS,
            max_physical_attempts=1,
        )
        for _ in range(3)
    ]

    started = time.perf_counter()
    audit_ticket = context.reserve_gemini_call(
        operation="pro_boundary_audit",
        model="gemini-3.1-pro-preview",
        estimated_input_tokens=30_000,
        max_output_tokens=gemini_segment._PRO_BOUNDARY_AUDIT_OUTPUT_TOKENS,
        max_physical_attempts=1,
        deadline_monotonic=time.monotonic() + 1.0,
    )
    admission_s = time.perf_counter() - started

    budget = context.budget.snapshot()["gemini"]
    assert admission_s < 0.02
    assert budget["pro_selector_calls"] == 3
    assert budget["boundary_audit_calls"] == 1
    assert budget["cost_exposure_usd"] == pytest.approx(
        sum(ticket["admitted_cost_usd"] for ticket in selector_tickets)
        + audit_ticket["admitted_cost_usd"]
    )
    assert budget["cost_exposure_usd"] <= budget["cost_limit_usd"]


def test_single_physical_ticket_does_not_reserve_unused_retry_contingency() -> None:
    context = GenerationContext("fast", generation_id="job-retry-hard-ceiling")

    reservation = context.reserve_gemini_call(
        operation="flash_boundary_selector",
        model="gemini-3.5-flash",
        estimated_input_tokens=594_000,
        max_output_tokens=1_000,
    )

    budget = context.budget.snapshot()["gemini"]
    assert reservation["admitted_physical_attempts"] == 1
    assert reservation["admitted_cost_usd"] == pytest.approx(
        reservation["reserved_cost_usd"]
    )
    assert budget["selector_calls"] == 1
    assert budget["cost_exposure_usd"] == pytest.approx(
        reservation["reserved_cost_usd"]
    )
    assert budget["cost_exposure_usd"] <= budget["cost_limit_usd"]


@pytest.mark.parametrize(
    "operation",
    [
        "boundary_selection",
        "flash_boundary_repair",
        "flash_boundary_selector",
        "flash_grounded_enrichment",
        "flash_single_candidate",
        "pro_authoritative",
        "pro_boundary_audit",
    ],
)
def test_operations_reserve_one_physical_attempt_by_default(
    operation: str,
) -> None:
    context = GenerationContext("fast", generation_id=f"job-retry-{operation}")
    reservation = context.reserve_gemini_call(
        operation=operation,
        model=(
            "gemini-3.1-pro-preview"
            if operation.startswith("pro_")
            else "gemini-3.5-flash"
        ),
        estimated_input_tokens=1_000,
        max_output_tokens=100,
    )

    assert reservation["admitted_physical_attempts"] == 1
    assert reservation["admitted_cost_usd"] == pytest.approx(
        reservation["reserved_cost_usd"]
    )
    assert context.budget.snapshot()["gemini"]["cost_exposure_usd"] <= 1.0


def test_statusless_retry_success_retains_unknown_attempt_exposure() -> None:
    context = GenerationContext("fast", generation_id="job-statusless-retry")
    first_ticket = context.reserve_gemini_call(
        operation="flash_boundary_selector",
        model="gemini-3.5-flash",
        estimated_input_tokens=294_000,
        max_output_tokens=1_000,
    )
    assert context.reconcile_gemini_call(
        model_used="gemini-3.5-flash",
        usage={**first_ticket, "retries": 0, "dispatched": True},
        dispatched=True,
    ) is True
    second_ticket = context.reserve_gemini_call(
        operation="flash_boundary_selector",
        model="gemini-3.5-flash",
        estimated_input_tokens=294_000,
        max_output_tokens=1_000,
        count_logical_call=False,
    )
    final_usage = {
        **second_ticket,
        "retries": 1,
        "physical_dispatches": 2,
        "billing_unknown_attempts": 1,
        "billing_unknown_reserved_cost_usd": first_ticket["reserved_cost_usd"],
        "error_history": [{
            "provider_error_type": "ReadError",
            "provider_status_code": None,
            "retryable": True,
        }],
        "prompt_tokens": 294_000,
        "candidate_tokens": 1_000,
        "thought_tokens": 0,
        "total_tokens": 295_000,
        "dispatched": True,
    }
    assert context.reconcile_gemini_call(
        model_used="gemini-3.5-flash",
        usage={**final_usage, "retries": 0},
        dispatched=True,
    ) is True
    # Production records the logical telemetry after each physical ticket is
    # already settled. The reservation id therefore reconciles as a no-op.
    context.record_gemini(
        attempt=2,
        model_used="gemini-3.5-flash",
        quality_degraded=False,
        stage="selection",
        usage=final_usage,
    )

    final_attempt_cost = (294_000 * 1.5 + 1_000 * 9.0) / 1_000_000.0
    budget = context.budget.snapshot()["gemini"]
    assert budget["selector_calls"] == 1
    assert first_ticket["admitted_physical_attempts"] == 1
    assert second_ticket["admitted_physical_attempts"] == 1
    assert budget["committed_cost_usd"] == pytest.approx(
        first_ticket["reserved_cost_usd"] + final_attempt_cost
    )
    assert budget["billing_unknown_cost_exposure_usd"] == pytest.approx(
        first_ticket["reserved_cost_usd"]
    )
    assert budget["cost_exposure_usd"] <= budget["cost_limit_usd"]

    payload = context.usage_payload()
    assert payload["summary"]["gemini_calls"] == 1
    assert payload["summary"]["gemini_attempts"] == 2
    assert payload["summary"]["known_billed_cost_usd"] == pytest.approx(
        final_attempt_cost
    )
    assert payload["summary"]["billing_unknown_calls"] == 1
    assert payload["summary"]["billing_unknown_attempts"] == 1
    assert payload["summary"][
        "billing_unknown_reserved_cost_usd"
    ] == pytest.approx(first_ticket["reserved_cost_usd"])

    # Unknown retry exposure remains visible but cannot suppress later valid
    # selector work. The independent logical selector ceiling still applies.
    context.reserve_gemini_call(
        operation="flash_boundary_selector",
        model="gemini-3.5-flash",
        estimated_input_tokens=560_000,
        max_output_tokens=1_000,
    )
    context.reserve_gemini_call(
        operation="flash_boundary_selector",
        model="gemini-3.5-flash",
        estimated_input_tokens=1_000,
        max_output_tokens=100,
    )
    with pytest.raises(ProviderBudgetExceededError, match="selector budget"):
        context.reserve_gemini_call(
            operation="flash_boundary_selector",
            model="gemini-3.5-flash",
            estimated_input_tokens=1_000,
            max_output_tokens=100,
        )
    bounded = context.budget.snapshot()["gemini"]
    assert bounded["hard_cost_cap_enabled"] is False
    assert bounded["cost_exposure_usd"] > bounded["cost_target_usd"]
    assert bounded["selector_calls"] == bounded["selector_limit"] == 3


def test_failover_retains_primary_exposure_and_prices_final_model() -> None:
    context = GenerationContext("fast", generation_id="job-model-failover")
    primary_ticket = context.reserve_gemini_call(
        operation="flash_boundary_selector",
        model="gemini-3.5-flash",
        estimated_input_tokens=294_000,
        max_output_tokens=1_000,
    )
    context.reconcile_gemini_call(
        model_used="gemini-3.5-flash",
        usage={**primary_ticket, "retries": 0, "dispatched": True},
        dispatched=True,
    )
    failover_ticket = context.reserve_gemini_call(
        operation="flash_boundary_selector",
        model="gemini-3.1-flash-lite",
        estimated_input_tokens=294_000,
        max_output_tokens=1_000,
        count_logical_call=False,
    )
    final_usage = {
        **failover_ticket,
        "model": "gemini-3.1-flash-lite",
        "retries": 1,
        "physical_dispatches": 2,
        "billing_unknown_attempts": 1,
        "billing_unknown_reserved_cost_usd": primary_ticket["reserved_cost_usd"],
        "error_history": [{
            "provider_error_type": "ServerError",
            "provider_status_code": 503,
            "retryable": True,
        }],
        "failover_from_model": "gemini-3.5-flash",
        "failover_model": "gemini-3.1-flash-lite",
        "failover_reason": "primary_transient_5xx_failover",
        "prompt_tokens": 294_000,
        "candidate_tokens": 1_000,
        "thought_tokens": 0,
        "total_tokens": 295_000,
        "dispatched": True,
    }
    context.reconcile_gemini_call(
        model_used="gemini-3.1-flash-lite",
        usage={**final_usage, "retries": 0},
        dispatched=True,
    )

    context.record_gemini(
        attempt=2,
        model_used="gemini-3.1-flash-lite",
        quality_degraded=True,
        stage="selection",
        usage=final_usage,
    )

    final_attempt_cost = (294_000 * 0.25 + 1_000 * 1.5) / 1_000_000.0
    budget = context.budget.snapshot()["gemini"]
    assert budget["selector_calls"] == 1
    assert primary_ticket["admitted_physical_attempts"] == 1
    assert failover_ticket["admitted_physical_attempts"] == 1
    assert budget["committed_cost_usd"] == pytest.approx(
        primary_ticket["reserved_cost_usd"] + final_attempt_cost
    )
    assert budget["billing_unknown_cost_exposure_usd"] == pytest.approx(
        primary_ticket["reserved_cost_usd"]
    )
    assert budget["cost_exposure_usd"] <= budget["cost_limit_usd"]

    payload = context.usage_payload()
    assert payload["summary"]["known_billed_cost_usd"] == pytest.approx(
        final_attempt_cost
    )
    assert payload["summary"]["billing_unknown_attempts"] == 1
    assert payload["by_stage"]["selection"]["billing_unknown_attempts"] == 1
    assert payload["summary"][
        "billing_unknown_reserved_cost_usd"
    ] == pytest.approx(primary_ticket["reserved_cost_usd"])


def test_unknown_dispatched_usage_keeps_full_reservation_as_telemetry() -> None:
    context = GenerationContext("fast", generation_id="job-unknown-billing")
    reservation = context.reserve_gemini_call(
        operation="flash_boundary_selector",
        model="gemini-3.5-flash",
        estimated_input_tokens=200_000,
        max_output_tokens=8_192,
    )
    context.record_gemini(
        attempt=1,
        model_used="gemini-3.5-flash",
        quality_degraded=False,
        usage={**reservation, "dispatched": True},
        status_code=None,
        error_code="provider_usage_missing",
    )

    budget = context.budget.snapshot()["gemini"]
    assert budget["committed_cost_usd"] == pytest.approx(
        reservation["reserved_cost_usd"]
    )
    assert budget["inflight_reserved_cost_usd"] == 0.0
    context.increment_counter("persisted_clips")
    payload = context.usage_payload()
    assert payload["summary"]["billing_unknown_calls"] == 1
    assert payload["summary"]["known_billed_cost_usd"] == 0.0
    assert payload["summary"]["billing_unknown_reserved_cost_usd"] == pytest.approx(
        reservation["reserved_cost_usd"]
    )
    assert payload["summary"]["cost_per_accepted_clip_usd"] is None
    assert payload["by_stage"]["segmentation"]["billing_unknown_calls"] == 1
    context.reserve_gemini_call(
        operation="flash_boundary_selector",
        model="gemini-3.5-flash",
        estimated_input_tokens=3_000_000,
        max_output_tokens=8_192,
    )
    context.reserve_gemini_call(
        operation="flash_boundary_selector",
        model="gemini-3.5-flash",
        estimated_input_tokens=1_000,
        max_output_tokens=100,
    )
    with pytest.raises(ProviderBudgetExceededError, match="selector budget"):
        context.reserve_gemini_call(
            operation="flash_boundary_selector",
            model="gemini-3.5-flash",
            estimated_input_tokens=1_000,
            max_output_tokens=100,
        )
    bounded = context.budget.snapshot()["gemini"]
    assert bounded["hard_cost_cap_enabled"] is False
    assert bounded["cost_exposure_usd"] > bounded["cost_target_usd"]
    assert bounded["selector_calls"] == bounded["selector_limit"] == 3


def test_untrusted_unknown_usage_cannot_reduce_its_reserved_ceiling() -> None:
    context = GenerationContext("fast", generation_id="job-unknown-override")
    reservation = context.reserve_gemini_call(
        operation="flash_boundary_selector",
        model="gemini-3.5-flash",
        estimated_input_tokens=200_000,
        max_output_tokens=8_192,
    )

    context.record_gemini(
        attempt=1,
        model_used="gemini-3.5-flash",
        quality_degraded=False,
        usage={
            **reservation,
            "reserved_cost_usd": 0.0,
            "unknown_final_attempt_cost_usd": 0.0,
            "dispatched": True,
        },
        status_code=None,
        error_code="provider_usage_missing",
    )

    gemini = context.budget.snapshot()["gemini"]
    assert gemini["committed_cost_usd"] == pytest.approx(
        reservation["admitted_cost_usd"]
    )
    assert gemini["billing_unknown_cost_exposure_usd"] == pytest.approx(
        reservation["admitted_cost_usd"]
    )


def test_total_only_dispatched_usage_keeps_full_reservation_fail_closed() -> None:
    context = GenerationContext("fast", generation_id="job-total-only-billing")
    reservation = context.reserve_gemini_call(
        operation="flash_boundary_selector",
        model="gemini-3.5-flash",
        estimated_input_tokens=150_000,
        max_output_tokens=8_192,
    )

    context.record_gemini(
        attempt=1,
        model_used="gemini-3.5-flash",
        quality_degraded=False,
        usage={
            **reservation,
            "total_tokens": 10_000,
            "dispatched": True,
        },
    )

    budget = context.budget.snapshot()["gemini"]
    assert budget["committed_cost_usd"] == pytest.approx(
        reservation["reserved_cost_usd"]
    )
    assert budget["inflight_reserved_cost_usd"] == 0.0
    payload = context.usage_payload()
    assert payload["summary"]["billing_unknown_calls"] == 1
    assert payload["summary"]["known_billed_cost_usd"] == 0.0
    assert payload["summary"]["billing_unknown_reserved_cost_usd"] == pytest.approx(
        reservation["reserved_cost_usd"]
    )


def test_output_only_dispatched_usage_keeps_full_reservation_fail_closed() -> None:
    context = GenerationContext("fast", generation_id="job-output-only-billing")
    reservation = context.reserve_gemini_call(
        operation="flash_boundary_selector",
        model="gemini-3.5-flash",
        estimated_input_tokens=110_000,
        max_output_tokens=8_192,
    )

    context.record_gemini(
        attempt=1,
        model_used="gemini-3.5-flash",
        quality_degraded=False,
        usage={
            **reservation,
            "candidate_tokens": 10,
            "total_tokens": 10,
            "dispatched": True,
        },
    )

    budget = context.budget.snapshot()["gemini"]
    assert budget["committed_cost_usd"] == pytest.approx(
        reservation["reserved_cost_usd"]
    )
    assert budget["inflight_reserved_cost_usd"] == 0.0


def test_input_only_dispatched_usage_keeps_full_reservation_fail_closed() -> None:
    context = GenerationContext("fast", generation_id="job-input-only-billing")
    reservation = context.reserve_gemini_call(
        operation="flash_boundary_selector",
        model="gemini-3.5-flash",
        estimated_input_tokens=110_000,
        max_output_tokens=8_192,
    )

    context.record_gemini(
        attempt=1,
        model_used="gemini-3.5-flash",
        quality_degraded=False,
        usage={
            **reservation,
            "prompt_tokens": 10_000,
            "total_tokens": 10_000,
            "dispatched": True,
        },
    )

    budget = context.budget.snapshot()["gemini"]
    assert budget["committed_cost_usd"] == pytest.approx(
        reservation["reserved_cost_usd"]
    )
    assert budget["inflight_reserved_cost_usd"] == 0.0


def test_present_zero_output_split_reconciles_actual_input_cost() -> None:
    context = GenerationContext("fast", generation_id="job-zero-output-billing")
    reservation = context.reserve_gemini_call(
        operation="flash_boundary_selector",
        model="gemini-3.5-flash",
        estimated_input_tokens=110_000,
        max_output_tokens=8_192,
    )

    context.record_gemini(
        attempt=1,
        model_used="gemini-3.5-flash",
        quality_degraded=False,
        usage={
            **reservation,
            "prompt_tokens": 10_000,
            "candidate_tokens": 0,
            "thought_tokens": 0,
            "total_tokens": 10_000,
            "dispatched": True,
        },
    )

    budget = context.budget.snapshot()["gemini"]
    assert budget["committed_cost_usd"] == pytest.approx(
        10_000 * 1.5 / 1_000_000.0
    )
    assert budget["inflight_reserved_cost_usd"] == 0.0


def test_non_dispatched_reservation_releases_capacity_idempotently() -> None:
    context = GenerationContext("fast", generation_id="job-not-dispatched")
    reservation = context.reserve_gemini_call(
        operation="flash_boundary_selector",
        model="gemini-3.5-flash",
        estimated_input_tokens=30_000,
        max_output_tokens=8_192,
    )

    assert context.reconcile_gemini_call(
        model_used="gemini-3.5-flash",
        usage={**reservation, "dispatched": False},
        dispatched=False,
    ) is True
    assert context.reconcile_gemini_call(
        model_used="gemini-3.5-flash",
        usage={**reservation, "dispatched": False},
        dispatched=False,
    ) is False

    replacement = context.reserve_gemini_call(
        operation="flash_boundary_selector",
        model="gemini-3.5-flash",
        estimated_input_tokens=30_000,
        max_output_tokens=8_192,
    )
    budget = context.budget.snapshot()["gemini"]
    assert budget["flash_selector_calls"] == 2
    assert budget["committed_cost_usd"] == 0.0
    assert budget["inflight_reserved_cost_usd"] == pytest.approx(
        replacement["admitted_cost_usd"]
    )


def test_durable_gemini_ticket_is_persisted_before_reservation_returns() -> None:
    persisted: list[dict] = []
    context = GenerationContext(
        "fast",
        generation_id="job-durable-ticket",
        gemini_ticket_reserve_sink=lambda **payload: (
            persisted.append(payload) or {"id": payload["ticket_id"]}
        ),
    )

    ticket = context.reserve_gemini_call(
        operation="pro_authoritative",
        model="gemini-3.1-pro-preview",
        estimated_input_tokens=1_000,
        max_output_tokens=100,
    )

    assert len(persisted) == 1
    assert persisted[0]["ticket_id"] == ticket["gemini_ticket_id"]
    assert persisted[0]["reservation"]["gemini_reservation_id"] == (
        ticket["gemini_reservation_id"]
    )
    assert ticket["gemini_durable_ticket"] is True


def test_failed_durable_ticket_persistence_releases_local_cost_exposure() -> None:
    context = GenerationContext(
        "fast",
        generation_id="job-durable-ticket-failure",
        gemini_ticket_reserve_sink=lambda **_payload: None,
    )

    with pytest.raises(
        RuntimeError,
        match="dispatch ticket could not be persisted",
    ):
        context.reserve_gemini_call(
            operation="pro_authoritative",
            model="gemini-3.1-pro-preview",
            estimated_input_tokens=1_000,
            max_output_tokens=100,
        )

    gemini = context.budget.snapshot()["gemini"]
    assert gemini["committed_cost_usd"] == 0.0
    assert gemini["inflight_reserved_cost_usd"] == 0.0
    assert gemini["lifetime_reserved_worst_case_cost_usd"] == 0.0
    assert gemini["selector_calls"] == 0
    assert gemini["pro_selector_calls"] == 0


def test_durable_ticket_settles_once_and_replaces_append_only_usage_sink() -> None:
    persisted: list[dict] = []
    settled: list[dict] = []
    appended: list[ProviderUsageRecord] = []
    context = GenerationContext(
        "fast",
        generation_id="job-durable-ticket-settlement",
        usage_sink=appended.append,
        gemini_ticket_reserve_sink=lambda **payload: (
            persisted.append(payload) or {"id": payload["ticket_id"]}
        ),
        gemini_ticket_settle_sink=lambda **payload: settled.append(payload),
    )
    ticket = context.reserve_gemini_call(
        operation="pro_authoritative",
        model="gemini-3.1-pro-preview",
        estimated_input_tokens=1_000,
        max_output_tokens=100,
    )
    usage = {
        **ticket,
        "prompt_tokens": 1_000,
        "candidate_tokens": 10,
        "thought_tokens": 0,
        "total_tokens": 1_010,
        "dispatched": True,
    }

    assert context.reconcile_gemini_call(
        model_used="gemini-3.1-pro-preview",
        usage=usage,
        dispatched=True,
    ) is True
    context.record_gemini(
        attempt=1,
        model_used="gemini-3.1-pro-preview",
        quality_degraded=False,
        stage="selection",
        usage=usage,
    )

    assert len(persisted) == 1
    assert len(settled) == 1
    assert settled[0]["ticket_id"] == ticket["gemini_ticket_id"]
    assert settled[0]["state"] == "settled_known"
    assert settled[0]["actual_cost_usd"] == pytest.approx(
        (1_000 * 2.0 + 10 * 12.0) / 1_000_000.0
    )
    assert appended == []
    assert len(context.usage()) == 1


def test_durable_ticket_release_does_not_claim_actual_provider_cost() -> None:
    settled: list[dict] = []
    context = GenerationContext(
        "fast",
        generation_id="job-durable-ticket-release",
        gemini_ticket_reserve_sink=lambda **payload: {
            "id": payload["ticket_id"],
        },
        gemini_ticket_settle_sink=lambda **payload: settled.append(payload),
    )
    ticket = context.reserve_gemini_call(
        operation="pro_authoritative",
        model="gemini-3.1-pro-preview",
        estimated_input_tokens=1_000,
        max_output_tokens=100,
    )

    assert context.reconcile_gemini_call(
        model_used="gemini-3.1-pro-preview",
        usage={**ticket, "dispatched": False},
        dispatched=False,
    ) is True

    assert len(settled) == 1
    assert settled[0]["state"] == "released"
    assert settled[0]["actual_cost_usd"] is None
    assert settled[0]["unknown_cost_usd"] is None


def test_terminal_gemini_exposure_is_fail_closed_idempotent_and_retryable() -> None:
    context = GenerationContext("fast", generation_id="job-terminal-exposure")
    reservation = context.reserve_gemini_call(
        operation="pro_authoritative",
        model="gemini-3.1-pro-preview",
        estimated_input_tokens=100,
        max_output_tokens=100,
    )
    admitted_cost = float(reservation["admitted_cost_usd"])

    first = context.budget.finalize_gemini_exposure()
    second = context.budget.finalize_gemini_exposure()
    assert first == second == {
        "admission_closed": True,
        "terminalized_reservation_count": 1,
        "terminalized_inflight_cost_usd": pytest.approx(admitted_cost),
    }

    terminal = context.budget.snapshot()["gemini"]
    assert terminal["inflight_reserved_cost_usd"] == 0.0
    assert terminal["committed_cost_usd"] == pytest.approx(admitted_cost)
    assert terminal["billing_unknown_cost_exposure_usd"] == pytest.approx(
        admitted_cost
    )
    assert terminal["terminalized_unreconciled_reservation_count"] == 1
    assert context.budget.reconcile_gemini(
        int(reservation["gemini_reservation_id"]),
        actual_cost_usd=0.00001,
    ) is True
    assert context.budget.snapshot()["gemini"]["cost_exposure_usd"] == pytest.approx(
        0.00001
    )
    assert context.budget.snapshot()["gemini"][
        "billing_unknown_cost_exposure_usd"
    ] == 0.0
    assert context.budget.snapshot()["gemini"][
        "terminalized_unreconciled_reservation_count"
    ] == 0
    assert context.budget.reconcile_gemini(
        int(reservation["gemini_reservation_id"]),
        actual_cost_usd=0.5,
    ) is False
    assert context.budget.snapshot()["gemini"]["cost_exposure_usd"] == pytest.approx(
        0.00001
    )
    with pytest.raises(ProviderBudgetExceededError, match="admission is closed"):
        context.reserve_gemini_call(
            operation="pro_authoritative",
            model="gemini-3.1-pro-preview",
            estimated_input_tokens=100,
            max_output_tokens=100,
        )

    retry = GenerationContext("fast", generation_id="job-terminal-exposure-retry")
    retry.budget.restore_gemini_retry_exposure(context.budget.snapshot())
    retry.budget.restore_gemini_retry_exposure(context.budget.snapshot())
    restored = retry.budget.snapshot()["gemini"]
    assert restored["cost_exposure_usd"] == pytest.approx(0.00001)
    assert restored["billing_unknown_cost_exposure_usd"] == 0.0
    assert restored["admission_closed"] is False
    retry.reserve_gemini_call(
        operation="pro_authoritative",
        model="gemini-3.1-pro-preview",
        estimated_input_tokens=100,
        max_output_tokens=100,
    )


def test_late_non_dispatched_ticket_removes_terminal_placeholder() -> None:
    context = GenerationContext("fast", generation_id="job-late-not-dispatched")
    reservation = context.reserve_gemini_call(
        operation="pro_authoritative",
        model="gemini-3.1-pro-preview",
        estimated_input_tokens=100,
        max_output_tokens=100,
    )
    context.budget.finalize_gemini_exposure()

    assert context.reconcile_gemini_call(
        model_used="gemini-3.1-pro-preview",
        usage={**reservation, "dispatched": False},
        dispatched=False,
    ) is True
    gemini = context.budget.snapshot()["gemini"]
    assert gemini["committed_cost_usd"] == 0.0
    assert gemini["billing_unknown_cost_exposure_usd"] == 0.0
    assert gemini["terminalized_unreconciled_reservation_count"] == 0


def test_concurrent_terminalization_and_reconciliation_count_ticket_once() -> None:
    for iteration in range(20):
        budget = GenerationBudget.for_mode("fast")
        reservation_id = budget.reserve_gemini(
            model="gemini-3.1-pro-preview",
            operation="pro_authoritative",
            estimated_cost_usd=0.05,
        )
        barrier = threading.Barrier(2)

        def reconcile() -> bool:
            barrier.wait()
            return budget.reconcile_gemini(
                reservation_id,
                actual_cost_usd=0.01,
            )

        def finalize() -> dict[str, int | float | bool]:
            barrier.wait()
            return budget.finalize_gemini_exposure()

        with ThreadPoolExecutor(max_workers=2) as executor:
            reconciled = executor.submit(reconcile)
            finalized = executor.submit(finalize)
        assert reconciled.result(timeout=1) is True
        finalized.result(timeout=1)

        gemini = budget.snapshot()["gemini"]
        assert gemini["inflight_reserved_cost_usd"] == 0.0
        assert gemini["admission_closed"] is True
        assert gemini["committed_cost_usd"] == pytest.approx(0.01)
        assert gemini["billing_unknown_cost_exposure_usd"] == 0.0
        assert gemini["terminalized_unreconciled_reservation_count"] == 0


def test_late_known_usage_reclassifies_terminal_placeholder_without_double_count(
) -> None:
    context = GenerationContext("fast", generation_id="job-late-known-usage")
    reservation = context.reserve_gemini_call(
        operation="pro_authoritative",
        model="gemini-3.1-pro-preview",
        estimated_input_tokens=1_000,
        max_output_tokens=100,
    )
    context.budget.finalize_gemini_exposure()

    context.record_gemini(
        attempt=1,
        model_used="gemini-3.1-pro-preview",
        quality_degraded=False,
        stage="selection",
        usage={
            **reservation,
            "prompt_tokens": 1_000,
            "candidate_tokens": 10,
            "thought_tokens": 0,
            "total_tokens": 1_010,
            "dispatched": True,
        },
    )

    actual_cost = (1_000 * 2.0 + 10 * 12.0) / 1_000_000.0
    gemini = context.budget.snapshot()["gemini"]
    assert gemini["committed_cost_usd"] == pytest.approx(actual_cost)
    assert gemini["billing_unknown_cost_exposure_usd"] == 0.0
    assert gemini["terminalized_unreconciled_reservation_count"] == 0
    payload = context.usage_payload()
    assert payload["summary"]["gemini_calls"] == 1
    assert payload["summary"]["known_billed_cost_usd"] == pytest.approx(
        actual_cost
    )
    assert payload["summary"]["billing_unknown_reserved_cost_usd"] == 0.0


def test_over_target_reservation_is_immediate_but_abort_signals_still_reject() -> None:
    context = GenerationContext("fast", generation_id="job-telemetry-only-cost")
    context.reserve_gemini_call(
        operation="flash_boundary_selector",
        model="gemini-3.5-flash",
        estimated_input_tokens=30_000,
        max_output_tokens=8_192,
    )

    started = time.monotonic()
    context.reserve_gemini_call(
        operation="flash_boundary_selector",
        model="gemini-3.5-flash",
        estimated_input_tokens=900_000,
        max_output_tokens=8_192,
        deadline_monotonic=time.monotonic() + 1.0,
    )
    assert time.monotonic() - started < 0.1
    over_target = context.budget.snapshot()["gemini"]
    assert over_target["hard_cost_cap_enabled"] is False
    assert over_target["cost_exposure_usd"] > over_target["cost_target_usd"]
    assert over_target["flash_selector_calls"] == 2

    with pytest.raises(ProviderBudgetExceededError, match="deadline"):
        context.reserve_gemini_call(
            operation="flash_boundary_selector",
            model="gemini-3.5-flash",
            estimated_input_tokens=1_000,
            max_output_tokens=100,
            deadline_monotonic=time.monotonic() - 1.0,
        )

    with pytest.raises(ProviderBudgetExceededError, match="cancelled"):
        context.reserve_gemini_call(
            operation="flash_boundary_selector",
            model="gemini-3.5-flash",
            estimated_input_tokens=1_000,
            max_output_tokens=100,
            deadline_monotonic=time.monotonic() + 1.0,
            cancelled=lambda: True,
        )
    assert context.budget.snapshot()["gemini"]["flash_selector_calls"] == 2


def test_deadline_and_cancellation_are_checked_before_admission() -> None:
    expired = GenerationContext("fast", generation_id="job-expired-before-admit")
    with pytest.raises(ProviderBudgetExceededError, match="deadline"):
        expired.reserve_gemini_call(
            operation="flash_boundary_selector",
            model="gemini-3.5-flash",
            estimated_input_tokens=1_000,
            max_output_tokens=100,
            deadline_monotonic=time.monotonic() - 1.0,
        )
    assert expired.budget.snapshot()["gemini"]["flash_selector_calls"] == 0

    cancelled = GenerationContext("fast", generation_id="job-cancel-before-admit")
    with pytest.raises(ProviderBudgetExceededError, match="cancelled"):
        cancelled.reserve_gemini_call(
            operation="flash_boundary_selector",
            model="gemini-3.5-flash",
            estimated_input_tokens=1_000,
            max_output_tokens=100,
            deadline_monotonic=time.monotonic() + 1.0,
            cancelled=lambda: True,
        )
    assert cancelled.budget.snapshot()["gemini"]["flash_selector_calls"] == 0


def test_committed_cost_above_target_never_blocks_bounded_reservation() -> None:
    budget = GenerationContext("fast", generation_id="job-over-target").budget
    committed = budget.reserve_gemini(
        model="gemini-3.1-flash-lite",
        operation="query_expansion",
        estimated_cost_usd=1.49,
    )
    budget.reconcile_gemini(committed, actual_cost_usd=1.49)
    budget.reserve_gemini(
        model="gemini-3.1-flash-lite",
        operation="query_expansion",
        estimated_cost_usd=0.001,
    )

    started = time.monotonic()
    reservation_id = budget.reserve_gemini(
        model="gemini-3.1-flash-lite",
        operation="query_expansion",
        estimated_cost_usd=0.02,
        deadline_monotonic=time.monotonic() + 1.0,
    )
    assert time.monotonic() - started < 0.1
    assert isinstance(reservation_id, int)
    telemetry = budget.snapshot()["gemini"]
    assert telemetry["hard_cost_cap_enabled"] is False
    assert telemetry["cost_exposure_usd"] > telemetry["cost_target_usd"]

@pytest.mark.parametrize(("mode", "selector_limit"), [("fast", 3), ("slow", 5)])
def test_authoritative_pro_uses_shared_bounded_selector_budget(
    mode: str,
    selector_limit: int,
) -> None:
    context = GenerationContext(mode, generation_id=f"job-pro-selector-{mode}")

    for _ in range(selector_limit):
        context.reserve_gemini_call(
            operation="pro_authoritative",
            model="gemini-3.1-pro-preview",
            prompt_text="transcript",
            max_output_tokens=4096,
        )
    with pytest.raises(ProviderBudgetExceededError, match="selector budget"):
        context.reserve_gemini_call(
            operation="pro_authoritative",
            model="gemini-3.1-pro-preview",
            prompt_text="one more transcript",
            max_output_tokens=4096,
        )
    with pytest.raises(ProviderBudgetExceededError, match="fallback"):
        context.reserve_gemini_call(
            operation="pro_fallback",
            model="gemini-3.1-pro-preview",
            prompt_text="transcript",
            max_output_tokens=4096,
        )

    budget = context.budget.snapshot()["gemini"]
    assert budget["selector_calls"] == selector_limit
    assert budget["selector_limit"] == selector_limit
    assert budget["flash_selector_calls"] == 0
    assert budget["pro_selector_calls"] == selector_limit
    assert budget["pro_calls"] == 0
    assert budget["pro_call_limit"] == 0


@pytest.mark.parametrize(("mode", "audit_limit"), [("fast", 3), ("slow", 5)])
def test_pro_boundary_audit_has_a_separate_bounded_budget(
    mode: str,
    audit_limit: int,
) -> None:
    context = GenerationContext(mode, generation_id=f"job-boundary-audit-{mode}")

    for _ in range(audit_limit):
        context.reserve_gemini_call(
            operation="pro_boundary_audit",
            model="gemini-3.1-pro-preview",
            prompt_text="candidate neighboring cues",
            max_output_tokens=1024,
        )
    with pytest.raises(ProviderBudgetExceededError, match="boundary-audit budget"):
        context.reserve_gemini_call(
            operation="pro_boundary_audit",
            model="gemini-3.1-pro-preview",
            prompt_text="one more candidate window",
            max_output_tokens=1024,
        )

    budget = context.budget.snapshot()["gemini"]
    assert budget["selector_calls"] == 0
    assert budget["boundary_audit_calls"] == audit_limit
    assert budget["boundary_audit_limit"] == audit_limit


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
    assert payload["budget"]["gemini"]["cost_limit_usd"] == 1.50


def test_generation_usage_preserves_video_grounding_transport_fact() -> None:
    context = GenerationContext("slow", generation_id="job-grounding-audit")
    context.record_gemini(
        attempt=1,
        model_used="gemini-3.1-pro-preview",
        quality_degraded=False,
        stage="selection",
        usage={
            "prompt_tokens": 100,
            "candidate_tokens": 20,
            "thought_tokens": 0,
            "total_tokens": 120,
            "video_grounded": False,
        },
    )

    record = context.usage()[0]
    assert record["metadata"]["video_grounded"] is False
    assert (
        context.usage_payload()["provider_calls"][0]["metadata"]["video_grounded"]
        is False
    )


def test_generation_usage_counts_selector_retries_as_physical_attempts_once() -> None:
    context = GenerationContext("fast", generation_id="job-retried-selector")
    context.record_gemini(
        attempt=3,
        model_used="gemini-3-flash-preview",
        quality_degraded=False,
        stage="selection",
        usage={
            "retries": 2,
            "prompt_tokens": 100,
            "candidate_tokens": 20,
            "thought_tokens": 10,
            "total_tokens": 130,
            "dispatched": True,
        },
    )

    payload = context.usage_payload()
    assert payload["summary"]["gemini_calls"] == 1
    assert payload["summary"]["gemini_attempts"] == 3
    assert payload["by_stage"]["selection"]["calls"] == 1
    assert payload["by_stage"]["selection"]["attempts"] == 3
    assert payload["summary"]["input_tokens"] == 100
    assert payload["summary"]["output_tokens"] == 30
    assert payload["summary"]["estimated_cost_usd"] == pytest.approx(
        (100 * 0.5 + 30 * 3.0) / 1_000_000.0
    )
    assert payload["summary"]["billing_unknown_calls"] == 1
    assert payload["summary"]["billing_unknown_attempts"] == 2
    assert payload["by_stage"]["selection"]["billing_unknown_attempts"] == 2


def test_failed_retries_add_attempts_without_inventing_token_billing() -> None:
    context = GenerationContext("fast", generation_id="job-failed-selector")
    context.record_gemini(
        attempt=3,
        model_used="gemini-3-flash-preview",
        quality_degraded=False,
        stage="selection",
        usage={
            "retries": 2,
            "dispatched": True,
            "reserved_cost_usd": 0.05,
        },
        status_code=None,
        error_code="model_call_failed",
    )

    payload = context.usage_payload()
    assert payload["summary"]["gemini_calls"] == 1
    assert payload["summary"]["gemini_attempts"] == 3
    assert payload["by_stage"]["selection"]["attempts"] == 3
    assert payload["summary"]["input_tokens"] == 0
    assert payload["summary"]["output_tokens"] == 0
    assert payload["summary"]["estimated_cost_usd"] == 0.0
    assert payload["summary"]["billing_unknown_calls"] == 1
    assert payload["summary"]["billing_unknown_attempts"] == 3
    assert payload["summary"]["billing_unknown_reserved_cost_usd"] == pytest.approx(
        0.15
    )
    assert payload["by_stage"]["selection"]["billing_unknown_calls"] == 1


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
