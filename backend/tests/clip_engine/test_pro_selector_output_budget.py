from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from backend import gemini_client
from backend.app.clip_engine.provider_runtime import GenerationContext
from backend.pipeline import gemini_segment


_OBSERVED_PRO_THOUGHT_TOKENS = 5_757


def _newton_plan() -> gemini_segment._CompactBoundaryPlan:
    claim = (
        "net force on an object equals its mass times its acceleration"
    )
    return gemini_segment._CompactBoundaryPlan(
        request_intent={
            "exact_request": "Newton's second law F=ma",
            "constraints": [{
                "constraint_id": "law",
                "kind": "subject",
                "source_phrase": "Newton's second law F=ma",
                "requirement": "Explain Newton's second law F=ma",
            }],
        },
        topics=[gemini_segment._CompactBoundaryTopic(
            candidate_id="newton-second-law",
            start_line=0,
            end_line=0,
            start_quote="Newton's second law says that",
            end_quote="mass times its acceleration",
            claim_quote=claim,
            title="Newton's second law",
            learning_objective="Explain how net force, mass, and acceleration relate",
            facet="F equals ma",
            informativeness=0.99,
            topic_relevance=0.99,
            educational_importance=0.99,
            difficulty=0.2,
            directly_teaches_topic=True,
            substantive=True,
            factually_grounded=True,
            self_contained=True,
            is_standalone=True,
            intent_evidence=[{"id": "law", "q": claim}],
        )],
    )


def test_text_only_pro_keeps_candidate_budget_after_observed_thought_usage(
    monkeypatch,
) -> None:
    """The production truncation must not turn a relevant source into no clips."""
    plan = _newton_plan()
    context = GenerationContext("fast", generation_id="pro-thought-headroom")
    calls: list[dict] = []
    candidate_tokens = 500
    prompt_tokens = 10_006

    monkeypatch.setattr(
        gemini_client,
        "count_request_tokens",
        lambda *_args, **_kwargs: prompt_tokens,
    )

    def generate(_system, user, _schema, **kwargs):
        calls.append({"user": user, **kwargs})
        if (
            kwargs["max_output_tokens"]
            < _OBSERVED_PRO_THOUGHT_TOKENS
            + gemini_segment._BOUNDARY_OUTPUT_TOKENS
        ):
            telemetry = gemini_client.GeminiCallTelemetry(
                model=kwargs["model"],
                operation=kwargs["operation"],
                prompt_version=kwargs["prompt_version"],
                thinking_level=kwargs["thinking_level"],
                latency_ms=49_500.0,
                retries=0,
                finish_reason="MAX_TOKENS",
                prompt_tokens=prompt_tokens,
                candidate_tokens=229,
                thought_tokens=_OBSERVED_PRO_THOUGHT_TOKENS,
                total_tokens=prompt_tokens + 5_986,
            )
            raise gemini_client.GeminiTruncatedResponseError(
                "thoughts consumed the candidate budget",
                telemetry,
            )
        return SimpleNamespace(
            text=plan.model_dump_json(by_alias=True),
            telemetry={
                "model": kwargs["model"],
                "operation": kwargs["operation"],
                "prompt_version": kwargs["prompt_version"],
                "thinking_level": kwargs["thinking_level"],
                "finish_reason": "STOP",
                "prompt_tokens": prompt_tokens,
                "candidate_tokens": candidate_tokens,
                "thought_tokens": _OBSERVED_PRO_THOUGHT_TOKENS,
                "total_tokens": (
                    prompt_tokens
                    + candidate_tokens
                    + _OBSERVED_PRO_THOUGHT_TOKENS
                ),
            },
        )

    monkeypatch.setattr(gemini_client, "generate_json_v3", generate)

    result = gemini_segment.run_segment_profile(
        {
            "segments": [{
                "cue_id": "supadata-cue-0",
                "start": 0.0,
                "end": 8.0,
                "text": (
                    "Newton's second law says that the net force on an object "
                    "equals its mass times its acceleration."
                ),
            }],
            "words": [],
            "source": "supadata",
        },
        {
            "_segment_operation": "pro_authoritative",
            "_segment_budget_reserve": context.reserve_gemini_call,
            "_segment_budget_reconcile": context.reconcile_gemini_call,
        },
        gemini_segment.PRO_BOUNDARY_PROFILE,
        topic="Newton's second law F=ma",
        deadline_monotonic=time.monotonic() + 90.0,
    )

    assert result.error is None
    assert result.accepted_count == 1
    assert result.clips[0]["selection_candidate_id"] == "newton-second-law"
    assert len(calls) == 1
    [call] = calls
    assert isinstance(call["user"], str)
    assert call["media_resolution"] is None
    assert call["model"] == "gemini-3.1-pro-preview"
    assert call["thinking_level"] == "medium"
    assert call["max_retries"] == 0
    assert call["max_output_tokens"] == gemini_segment._PRO_BOUNDARY_OUTPUT_TOKENS
    assert (
        call["max_output_tokens"] - _OBSERVED_PRO_THOUGHT_TOKENS
        >= gemini_segment._BOUNDARY_OUTPUT_TOKENS
    )
    assert result.calls[0]["video_grounded"] is False
    assert result.calls[0]["reserved_output_tokens"] == call["max_output_tokens"]

    budget = context.budget.snapshot()["gemini"]
    expected_actual_cost = (
        prompt_tokens * 2.0
        + (candidate_tokens + _OBSERVED_PRO_THOUGHT_TOKENS) * 12.0
    ) / 1_000_000.0
    assert budget["selector_calls"] == 1
    assert budget["committed_cost_usd"] == pytest.approx(expected_actual_cost)
    assert budget["inflight_reserved_cost_usd"] == 0.0


@pytest.mark.parametrize(
    ("mode", "selector_count", "cost_limit"),
    [("fast", 2, 0.45), ("slow", 3, 0.70)],
)
def test_pro_thought_headroom_fits_existing_job_cost_ceiling(
    mode: str,
    selector_count: int,
    cost_limit: float,
) -> None:
    context = GenerationContext(mode, generation_id=f"pro-headroom-{mode}")
    context.reserve_gemini_call(
        operation="expansion",
        model="gemini-3.1-flash-lite",
        estimated_input_tokens=1_000,
        max_output_tokens=1_024,
    )
    reservations = [
        context.reserve_gemini_call(
            operation="pro_authoritative",
            model="gemini-3.1-pro-preview",
            estimated_input_tokens=30_000,
            max_output_tokens=gemini_segment._PRO_BOUNDARY_OUTPUT_TOKENS,
        )
        for _ in range(selector_count)
    ]

    assert all(
        reservation["reserved_output_tokens"]
        == gemini_segment._PRO_BOUNDARY_OUTPUT_TOKENS
        for reservation in reservations
    )
    budget = context.budget.snapshot()["gemini"]
    assert budget["selector_calls"] == selector_count
    assert budget["cost_exposure_usd"] <= cost_limit
    assert budget["cost_limit_usd"] == pytest.approx(cost_limit)
