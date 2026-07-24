from __future__ import annotations

import json
import math
import threading
import time
from types import SimpleNamespace

import pytest

from backend import gemini_client
from backend.app.clip_engine import expand
from backend.app.clip_engine.errors import ProviderBudgetExceededError
from backend.app.clip_engine.provider_runtime import GenerationContext
from backend.pipeline import gemini_segment


_OBSERVED_PRO_THOUGHT_TOKENS = 5_757


class _RetryHTTPError(RuntimeError):
    def __init__(self, status_code: int):
        super().__init__(f"HTTP status {status_code}")
        self.status_code = status_code
        self.response = SimpleNamespace(headers={})


class _RetryResponse:
    def __init__(self, text: str):
        self.text = text
        self.candidates = [SimpleNamespace(
            finish_reason=SimpleNamespace(value="STOP"),
        )]
        self.usage_metadata = SimpleNamespace(
            prompt_token_count=100,
            candidates_token_count=50,
            thoughts_token_count=25,
            total_token_count=175,
            cached_content_token_count=0,
        )


class _RetryClient:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []
        self.models = self

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _settle_mock_dispatch(kwargs: dict, telemetry: object) -> None:
    ticket = kwargs["before_dispatch"](
        model=kwargs["model"], attempt=1,
    )
    kwargs["after_dispatch"](
        ticket,
        model=kwargs["model"],
        attempt=1,
        telemetry=telemetry,
    )


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
            concept_family="Newton's second law",
            concept_aliases=["F=ma"],
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
            objective_constraint_ids=["law"],
            relationship_witnesses=[],
        )],
    )


def _request_mismatched_newton_plan() -> gemini_segment._CompactBoundaryPlan:
    payload = _newton_plan().model_dump(mode="json", by_alias=True)
    payload["request_intent"]["exact_request"] = "Newton's first law F=ma"
    return gemini_segment._CompactBoundaryPlan.model_validate(payload)


def _newton_audit_plan() -> gemini_segment._ProCandidateAuditPlan:
    return gemini_segment._ProCandidateAuditPlan(items=[{
        "candidate_id": "candidate-1",
        "decision": "keep",
        "actual_objective": "Explain how net force, mass, and acceleration relate",
        "title": "Newton's Second Law",
        "facet": "net force, mass, and acceleration",
        "concept_family": "Newton's second law",
        "concept_aliases": ["F=ma"],
        "directly_teaches_topic": True,
        "self_contained": True,
        "is_standalone": True,
        "intent_evidence": [{
            "id": "law",
            "q": "net force on an object equals its mass times its acceleration",
        }],
        "objective_constraint_ids": ["law"],
        "relationship_witnesses": [],
        "evidence_quote": (
            "force on an object equals its mass times its acceleration"
        ),
        "objective_witness_quote": (
            "force on an object equals its mass times its acceleration"
        ),
        "direct_start_line": 0,
        "direct_start_quote": "Newton's second law says that",
        "direct_start_context_resolved": True,
        "start_line": 0,
        "end_line": 0,
        "start_quote": "Newton's second law says that",
        "end_quote": "mass times its acceleration",
    }])


def test_v13_audit_budget_fits_forty_items_plus_observed_high_thinking() -> None:
    audit = gemini_segment._ProCandidateAuditPlan(items=[{
        "id": f"candidate-{index}",
        "d": "reject_filler_dominated",
        "obj": "o" * 120,
        "t": "t",
        "f": "f",
        "family": "x",
        "a": [],
        "direct": False,
        "self": False,
        "stand": False,
        "ie": [],
        "oi": [],
        "rw": [],
        "ev": "e" * 112,
        "ow": "e" * 112,
        "ds": 999,
        "dq": "d" * 84,
        "dc": True,
        "s": 999,
        "e": 999,
        "sq": "s" * 84,
        "eq": "q" * 84,
    } for index in range(1, 41)])
    payload_bytes = len(audit.model_dump_json(by_alias=True).encode("utf-8"))
    conservative_candidate_tokens = math.ceil(payload_bytes / 2)

    assert len(audit.items) == 40
    assert (
        conservative_candidate_tokens + _OBSERVED_PRO_THOUGHT_TOKENS
        <= gemini_segment._pro_boundary_audit_output_tokens(len(audit.items))
    )
    assert gemini_segment._pro_boundary_audit_output_tokens(40) == (
        gemini_segment._PRO_BOUNDARY_AUDIT_OUTPUT_TOKENS
    )


def test_typical_nine_candidate_audit_uses_smaller_admissible_output_reservation() -> None:
    audit = gemini_segment._ProCandidateAuditPlan(items=[{
        "id": f"candidate-{index}",
        "d": "reject_filler_dominated",
        "obj": "o" * 120,
        "t": "t",
        "f": "f",
        "family": "x",
        "a": [],
        "direct": False,
        "self": False,
        "stand": False,
        "ie": [],
        "oi": [],
        "rw": [],
        "ev": "e" * 112,
        "ow": "e" * 112,
        "ds": 999,
        "dq": "d" * 84,
        "dc": True,
        "s": 999,
        "e": 999,
        "sq": "s" * 84,
        "eq": "q" * 84,
    } for index in range(1, 10)])
    payload_tokens = math.ceil(
        len(audit.model_dump_json(by_alias=True).encode("utf-8")) / 2
    )
    reservation = gemini_segment._pro_boundary_audit_output_tokens(9)

    assert payload_tokens + _OBSERVED_PRO_THOUGHT_TOKENS <= reservation
    assert reservation < 19_200


def test_pro_selector_and_boundary_audit_recover_one_transient_failure(
    monkeypatch,
) -> None:
    plan = _newton_plan()
    audit = _newton_audit_plan()
    fake = _RetryClient(
        _RetryHTTPError(504),
        _RetryResponse(plan.model_dump_json(by_alias=True)),
        _RetryHTTPError(429),
        _RetryResponse(audit.model_dump_json(by_alias=True)),
    )
    monkeypatch.setattr(gemini_client, "get_client", lambda: fake)
    monkeypatch.setattr(gemini_client.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        gemini_client.random,
        "uniform",
        lambda lower, _upper: lower,
    )

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
        {"_segment_operation": "pro_authoritative"},
        gemini_segment.PRO_BOUNDARY_PROFILE,
        topic="Newton's second law F=ma",
        deadline_monotonic=time.monotonic() + 90.0,
    )

    assert result.error is None
    assert result.accepted_count == 1
    assert len(fake.calls) == 4
    assert [call["retries"] for call in result.calls] == [1, 1]
    assert [
        call["error_history"][0]["provider_status_code"]
        for call in result.calls
    ] == [504, 429]


def test_selector_contract_correction_retries_transient_within_five_source_dispatches(
    monkeypatch,
) -> None:
    plan = _newton_plan()
    invalid_plan = _request_mismatched_newton_plan()
    audit = _newton_audit_plan()
    fake = _RetryClient(
        _RetryHTTPError(503),
        _RetryResponse(invalid_plan.model_dump_json(by_alias=True)),
        _RetryHTTPError(504),
        _RetryResponse(plan.model_dump_json(by_alias=True)),
        _RetryResponse(audit.model_dump_json(by_alias=True)),
    )
    monkeypatch.setattr(gemini_client, "get_client", lambda: fake)
    monkeypatch.setattr(gemini_client.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        gemini_client.random,
        "uniform",
        lambda lower, _upper: lower,
    )

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
        {"_segment_operation": "pro_authoritative"},
        gemini_segment.PRO_BOUNDARY_PROFILE,
        topic="Newton's second law F=ma",
        deadline_monotonic=time.monotonic() + 180.0,
    )

    assert result.error is None
    assert result.accepted_count == 1
    assert len(fake.calls) == 5
    assert len(fake.outcomes) == 0
    assert gemini_segment._PRO_SOURCE_MAX_PHYSICAL_DISPATCHES == 5
    assert [
        int(call.get("physical_dispatches") or 0)
        for call in result.calls
    ] == [2, 2, 1]
    assert result.calls[0]["selector_contract_retry_reason"] == (
        "intent_contract_request_mismatch"
    )
    assert result.calls[1]["selector_contract_retry_recovered"] is True
    assert result.calls[1]["error_history"][0]["provider_status_code"] == 504
    assert result.calls[2]["operation"] == "pro_boundary_audit"


def test_selector_contract_correction_does_not_retry_permanent_400(
    monkeypatch,
) -> None:
    invalid_plan = _request_mismatched_newton_plan()
    fake = _RetryClient(
        _RetryResponse(invalid_plan.model_dump_json(by_alias=True)),
        _RetryHTTPError(400),
        _RetryResponse(_newton_plan().model_dump_json(by_alias=True)),
    )
    monkeypatch.setattr(gemini_client, "get_client", lambda: fake)

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
        {"_segment_operation": "pro_authoritative"},
        gemini_segment.PRO_BOUNDARY_PROFILE,
        topic="Newton's second law F=ma",
        deadline_monotonic=time.monotonic() + 180.0,
    )

    assert result.error is not None
    assert len(fake.calls) == 2
    assert len(fake.outcomes) == 1
    assert [
        int(call.get("physical_dispatches") or 0)
        for call in result.calls
    ] == [1, 1]
    assert result.calls[1]["provider_status_code"] == 400
    assert result.calls[1]["retryable"] is False
    assert result.calls[1]["selector_contract_retry_exhausted"] is True


def test_selector_contract_correction_transient_exhaustion_stays_below_five(
    monkeypatch,
) -> None:
    invalid_plan = _request_mismatched_newton_plan()
    fake = _RetryClient(
        _RetryHTTPError(503),
        _RetryResponse(invalid_plan.model_dump_json(by_alias=True)),
        _RetryHTTPError(504),
        _RetryHTTPError(504),
        _RetryResponse(_newton_audit_plan().model_dump_json(by_alias=True)),
    )
    monkeypatch.setattr(gemini_client, "get_client", lambda: fake)
    monkeypatch.setattr(gemini_client.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        gemini_client.random,
        "uniform",
        lambda lower, _upper: lower,
    )

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
        {"_segment_operation": "pro_authoritative"},
        gemini_segment.PRO_BOUNDARY_PROFILE,
        topic="Newton's second law F=ma",
        deadline_monotonic=time.monotonic() + 180.0,
    )

    assert result.error is not None
    assert len(fake.calls) == 4
    assert len(fake.outcomes) == 1
    physical_dispatches = sum(
        int(call.get("physical_dispatches") or 0)
        for call in result.calls
    )
    assert physical_dispatches == 4
    assert (
        physical_dispatches
        < gemini_segment._PRO_SOURCE_MAX_PHYSICAL_DISPATCHES
    )
    assert result.calls[1]["retries"] == 1
    assert [
        item["provider_status_code"]
        for item in result.calls[1]["error_history"]
    ] == [504, 504]
    assert not any(
        call.get("operation") == "pro_boundary_audit"
        for call in result.calls
    )


@pytest.mark.parametrize("retry_recovers", [True, False])
def test_audit_schema_correction_uses_remaining_source_budget_without_exceeding_cap(
    monkeypatch,
    retry_recovers: bool,
) -> None:
    plan = _newton_plan()
    valid_audit = _newton_audit_plan()
    invalid_audit = valid_audit.model_dump(mode="json", by_alias=True)
    invalid_audit["items"][0]["t"] = "x" * 97
    correction_outcomes = (
        [
            _RetryHTTPError(504),
            _RetryResponse(valid_audit.model_dump_json(by_alias=True)),
        ]
        if retry_recovers
        else [
            _RetryHTTPError(504),
            _RetryHTTPError(504),
            _RetryResponse(valid_audit.model_dump_json(by_alias=True)),
        ]
    )
    fake = _RetryClient(
        _RetryResponse(plan.model_dump_json(by_alias=True)),
        _RetryResponse(json.dumps(invalid_audit)),
        *correction_outcomes,
    )
    monkeypatch.setattr(gemini_client, "get_client", lambda: fake)
    monkeypatch.setattr(gemini_client.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        gemini_client.random,
        "uniform",
        lambda lower, _upper: lower,
    )

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
        {"_segment_operation": "pro_authoritative"},
        gemini_segment.PRO_BOUNDARY_PROFILE,
        topic="Newton's second law F=ma",
        deadline_monotonic=time.monotonic() + 180.0,
    )

    assert (result.error is None) is retry_recovers
    assert result.accepted_count == (1 if retry_recovers else 0)
    assert len(fake.calls) == 4
    assert len(fake.outcomes) == (0 if retry_recovers else 1)
    physical_dispatches = sum(
        int(call.get("physical_dispatches") or 0)
        for call in result.calls
    )
    assert physical_dispatches == 4
    assert (
        physical_dispatches
        <= gemini_segment._PRO_SOURCE_MAX_PHYSICAL_DISPATCHES
    )
    assert result.calls[1]["contract_retry_attempt"] == 1
    assert result.calls[2]["physical_dispatches"] == 2
    if retry_recovers:
        assert result.calls[2]["contract_retry_attempt"] == 2
        assert result.calls[2]["contract_retry_recovered"] is True
        assert (
            result.calls[2]["error_history"][0]["provider_status_code"]
            == 504
        )


def test_boundary_audit_late_transients_get_three_physical_attempts(
    monkeypatch,
) -> None:
    plan = _newton_plan()
    audit = _newton_audit_plan()
    clock = {"now": 60.0}

    class TimedAuditClient:
        def __init__(self):
            self.calls: list[dict] = []
            self.models = self

        def generate_content(self, model, contents, config):
            self.calls.append({
                "model": model,
                "contents": contents,
                "config": config,
            })
            if len(self.calls) == 1:
                clock["now"] = 119.0
                raise _RetryHTTPError(504)
            if len(self.calls) == 2:
                raise TimeoutError("audit transport timed out")
            return _RetryResponse(audit.model_dump_json(by_alias=True))

    fake = TimedAuditClient()
    context = GenerationContext("fast", generation_id="late-audit-transient")
    monkeypatch.setattr(gemini_client, "get_client", lambda: fake)
    monkeypatch.setattr(
        gemini_client,
        "count_request_tokens",
        lambda *_args, **_kwargs: 100,
    )
    monkeypatch.setattr(
        gemini_client.time,
        "monotonic",
        lambda: clock["now"],
    )
    monkeypatch.setattr(
        gemini_client.time,
        "sleep",
        lambda seconds: clock.__setitem__("now", clock["now"] + seconds),
    )
    monkeypatch.setattr(
        gemini_client.random,
        "uniform",
        lambda lower, _upper: lower,
    )

    retained, calls, rejections = gemini_segment._audit_pro_boundaries(
        plan,
        [{
            "cue_id": "supadata-cue-0",
            "start": 0.0,
            "end": 8.0,
            "text": (
                "Newton's second law says that the net force on an object "
                "equals its mass times its acceleration."
            ),
        }],
        plan.request_intent.exact_request,
        {
            "_segment_budget_reserve": context.reserve_gemini_call,
            "_segment_budget_reconcile": context.reconcile_gemini_call,
        },
        deadline=120.0,
        cancelled=None,
    )

    assert [topic.candidate_id for topic in retained.topics] == [
        "newton-second-law"
    ]
    assert rejections == []
    assert len(fake.calls) == 3
    assert [
        call["config"].http_options.timeout for call in fake.calls
    ] == [60_000, 60_000, 58_000]
    assert len(calls) == 1
    assert calls[0]["retries"] == 2
    assert calls[0]["physical_dispatches"] == 3
    assert calls[0]["error_history"][0]["provider_status_code"] == 504
    assert calls[0]["error_history"][1]["provider_error_type"] == "TimeoutError"
    assert context.budget.snapshot()["gemini"]["boundary_audit_calls"] == 1


def test_late_pro_499_retry_shares_deadline_and_reconciles_each_dispatch(
    monkeypatch,
) -> None:
    plan = _newton_plan()
    audit = _newton_audit_plan()
    clock = {"now": 0.0}

    class TimedClient:
        def __init__(self):
            self.calls: list[dict] = []
            self.models = self

        def generate_content(self, model, contents, config):
            self.calls.append({
                "model": model,
                "contents": contents,
                "config": config,
            })
            if len(self.calls) == 1:
                clock["now"] = 59.0
                raise _RetryHTTPError(499)
            if len(self.calls) == 2:
                return _RetryResponse(plan.model_dump_json(by_alias=True))
            return _RetryResponse(audit.model_dump_json(by_alias=True))

    fake = TimedClient()
    context = GenerationContext("fast", generation_id="late-pro-499")
    monkeypatch.setattr(gemini_client, "get_client", lambda: fake)
    monkeypatch.setattr(gemini_client.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(
        gemini_client.time,
        "sleep",
        lambda seconds: clock.__setitem__("now", clock["now"] + seconds),
    )
    monkeypatch.setattr(
        gemini_client.random,
        "uniform",
        lambda lower, _upper: lower,
    )
    monkeypatch.setattr(
        gemini_client,
        "count_request_tokens",
        lambda *_args, **_kwargs: 100,
    )

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
        deadline_monotonic=120.0,
    )

    assert result.error is None
    assert result.accepted_count == 1
    assert len(fake.calls) == 3
    assert [
        call["config"].http_options.timeout for call in fake.calls
    ] == [60_000, 60_000, 60_000]
    selector_call, audit_call = result.calls
    assert selector_call["retries"] == 1
    assert selector_call["physical_dispatches"] == 2
    assert selector_call["billing_unknown_attempts"] == 1
    assert selector_call["provider_status_code"] is None
    assert selector_call["error_history"][0]["provider_status_code"] == 499
    assert audit_call["operation"] == "pro_boundary_audit"
    assert audit_call["physical_dispatches"] == 1

    budget = context.budget.snapshot()["gemini"]
    assert budget["selector_calls"] == 1
    assert budget["boundary_audit_calls"] == 1
    assert budget["billing_unknown_cost_exposure_usd"] > 0.0
    assert budget["inflight_reserved_cost_usd"] == 0.0
    assert budget["cost_exposure_usd"] <= budget["cost_limit_usd"]


def test_pro_selector_and_boundary_audit_do_not_retry_permanent_4xx(
    monkeypatch,
) -> None:
    plan = _newton_plan()
    selector_fake = _RetryClient(
        _RetryHTTPError(400),
        _RetryResponse(plan.model_dump_json(by_alias=True)),
    )
    monkeypatch.setattr(gemini_client, "get_client", lambda: selector_fake)

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
        {"_segment_operation": "pro_authoritative"},
        gemini_segment.PRO_BOUNDARY_PROFILE,
        topic="Newton's second law F=ma",
        deadline_monotonic=time.monotonic() + 90.0,
    )

    assert result.error is not None
    assert len(selector_fake.calls) == 1
    assert result.calls[0]["provider_status_code"] == 400
    assert result.calls[0]["retryable"] is False

    audit_fake = _RetryClient(
        _RetryHTTPError(403),
        _RetryResponse(_newton_audit_plan().model_dump_json(by_alias=True)),
    )
    monkeypatch.setattr(gemini_client, "get_client", lambda: audit_fake)
    with pytest.raises(gemini_segment._ModelCallError) as exc_info:
        gemini_segment._audit_pro_boundaries(
            plan,
            [{
                "cue_id": "supadata-cue-0",
                "start": 0.0,
                "end": 8.0,
                "text": (
                    "Newton's second law says that the net force on an object "
                    "equals its mass times its acceleration."
                ),
            }],
            plan.request_intent.exact_request,
            {},
            deadline=time.monotonic() + 30.0,
            cancelled=None,
        )

    assert len(audit_fake.calls) == 1
    calls = exc_info.value.selection_attempt_calls
    assert calls[0]["provider_status_code"] == 403
    assert calls[0]["retryable"] is False


def test_text_only_pro_keeps_candidate_budget_after_observed_thought_usage(
    monkeypatch,
) -> None:
    """The production truncation must not turn a relevant source into no clips."""
    plan = _newton_plan()
    context = GenerationContext("fast", generation_id="pro-thought-headroom")
    calls: list[dict] = []
    candidate_tokens = 500
    prompt_tokens = 10_006
    audit_prompt_tokens = 1_000
    audit_candidate_tokens = 100

    monkeypatch.setattr(
        gemini_client,
        "count_request_tokens",
        lambda *_args, **_kwargs: prompt_tokens,
    )

    def generate(_system, user, _schema, **kwargs):
        calls.append({"user": user, "schema": _schema, **kwargs})
        if _schema is gemini_segment._ProCandidateAuditPlan:
            telemetry = {
                "model": kwargs["model"],
                "operation": kwargs["operation"],
                "prompt_version": kwargs["prompt_version"],
                "thinking_level": kwargs["thinking_level"],
                "finish_reason": "STOP",
                "prompt_tokens": audit_prompt_tokens,
                "candidate_tokens": audit_candidate_tokens,
                "thought_tokens": 0,
                "total_tokens": audit_prompt_tokens + audit_candidate_tokens,
            }
            _settle_mock_dispatch(kwargs, telemetry)
            return SimpleNamespace(
                text=_newton_audit_plan().model_dump_json(by_alias=True),
                telemetry=telemetry,
            )
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
            _settle_mock_dispatch(kwargs, telemetry)
            raise gemini_client.GeminiTruncatedResponseError(
                "thoughts consumed the candidate budget",
                telemetry,
            )
        telemetry = {
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
        }
        _settle_mock_dispatch(kwargs, telemetry)
        return SimpleNamespace(
            text=plan.model_dump_json(by_alias=True),
            telemetry=telemetry,
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
    assert len(calls) == 2
    call, audit_call = calls
    assert isinstance(call["user"], str)
    assert call["media_resolution"] is None
    assert call["model"] == "gemini-3.1-pro-preview"
    assert call["thinking_level"] == "medium"
    assert call["max_retries"] == 1
    assert call["retry_status_codes"] is None
    assert call["retry_service_tier"] == "priority"
    assert call["max_output_tokens"] == gemini_segment._PRO_BOUNDARY_OUTPUT_TOKENS
    assert (
        call["max_output_tokens"] - _OBSERVED_PRO_THOUGHT_TOKENS
        >= gemini_segment._BOUNDARY_OUTPUT_TOKENS
    )
    assert result.calls[0]["video_grounded"] is False
    assert result.calls[0]["reserved_output_tokens"] == call["max_output_tokens"]
    assert audit_call["schema"] is gemini_segment._ProCandidateAuditPlan
    assert audit_call["operation"] == "pro_boundary_audit"
    assert audit_call["prompt_version"] == "pro_candidate_audit_v14"
    assert audit_call["thinking_level"] == "medium"
    assert audit_call["media_resolution"] is None
    assert audit_call["retry_service_tier"] == "priority"
    assert gemini_segment._PRO_FINAL_AUDIT_RESERVED_S >= 60.0
    assert (
        gemini_segment._TOTAL_DEADLINE_S
        == 2 * gemini_segment._PRO_FINAL_AUDIT_RESERVED_S
    )
    assert call["timeout_s"] == gemini_segment._PRO_SELECTOR_ATTEMPT_TIMEOUT_S
    assert audit_call["deadline_monotonic"] == pytest.approx(
        call["deadline_monotonic"]
        + gemini_segment._PRO_AUDIT_RETRY_GRACE_S
    )
    assert call["initial_attempt_deadline_monotonic"] == pytest.approx(
        call["deadline_monotonic"]
        - gemini_segment._PRO_FINAL_AUDIT_RESERVED_S
    )
    assert audit_call["initial_attempt_deadline_monotonic"] is not None
    assert audit_call["initial_attempt_deadline_monotonic"] == pytest.approx(
        call["deadline_monotonic"]
    )
    assert audit_call["deadline_monotonic"] - audit_call[
        "initial_attempt_deadline_monotonic"
    ] == pytest.approx(
        gemini_segment._PRO_AUDIT_RETRY_GRACE_S
    )
    assert result.calls[1]["video_grounded"] is False

    budget = context.budget.snapshot()["gemini"]
    expected_actual_cost = (
        prompt_tokens * 2.0
        + (candidate_tokens + _OBSERVED_PRO_THOUGHT_TOKENS) * 12.0
        + audit_prompt_tokens * 2.0
        + audit_candidate_tokens * 12.0
    ) / 1_000_000.0
    assert budget["selector_calls"] == 1
    assert budget["boundary_audit_calls"] == 1
    assert budget["committed_cost_usd"] == pytest.approx(expected_actual_cost)
    assert budget["inflight_reserved_cost_usd"] == 0.0


def test_text_only_pro_retries_one_malformed_structured_response(
    monkeypatch,
) -> None:
    """A malformed Pro response gets one bounded high-reasoning correction."""
    plan = _newton_plan()
    context = GenerationContext("fast", generation_id="pro-schema-retry")
    generated: list[dict] = []

    monkeypatch.setattr(
        gemini_client,
        "count_request_tokens",
        lambda *_args, **_kwargs: 10_000,
    )

    def generate(_system, user, schema, **kwargs):
        generated.append({"user": user, "schema": schema, **kwargs})
        prompt_tokens = (
            10_000 if schema is gemini_segment._CompactBoundaryPlan else 500
        )
        candidate_tokens = 100
        thought_tokens = 25
        telemetry = {
            "model": kwargs["model"],
            "operation": kwargs["operation"],
            "prompt_version": kwargs["prompt_version"],
            "thinking_level": kwargs["thinking_level"],
            "finish_reason": "STOP",
            "prompt_tokens": prompt_tokens,
            "candidate_tokens": candidate_tokens,
            "thought_tokens": thought_tokens,
            "total_tokens": prompt_tokens + candidate_tokens + thought_tokens,
        }
        _settle_mock_dispatch(kwargs, telemetry)
        if schema is gemini_segment._CompactBoundaryPlan and sum(
            call["schema"] is gemini_segment._CompactBoundaryPlan
            for call in generated
        ) == 1:
            return SimpleNamespace(text="{malformed", telemetry=telemetry)
        if schema is gemini_segment._ProCandidateAuditPlan:
            text = _newton_audit_plan().model_dump_json(by_alias=True)
        else:
            text = plan.model_dump_json(by_alias=True)
        return SimpleNamespace(text=text, telemetry=telemetry)

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
    assert [call["schema"] for call in generated] == [
        gemini_segment._CompactBoundaryPlan,
        gemini_segment._CompactBoundaryPlan,
        gemini_segment._ProCandidateAuditPlan,
    ]
    assert generated[0]["user"] != generated[1]["user"]
    assert generated[1]["user"].startswith(generated[0]["user"])
    assert "invalid_structured_response" in generated[1]["user"]
    assert isinstance(generated[0]["user"], str)
    assert generated[0]["media_resolution"] is None
    assert generated[1]["media_resolution"] is None
    assert generated[0]["max_retries"] == 1
    assert generated[1]["max_retries"] == 2
    assert generated[1]["use_full_transient_retry_budget"] is True
    assert [call["thinking_level"] for call in generated] == [
        "medium",
        "high",
        "medium",
    ]
    assert result.calls[0]["error_type"] == "_SchemaResponseError"
    assert result.calls[0]["schema_retry_attempt"] == 1
    assert result.calls[0]["video_grounded"] is False
    assert result.calls[1]["schema_retry_attempt"] == 2
    assert result.calls[1]["schema_retry_recovered"] is True
    assert result.calls[1]["video_grounded"] is False
    assert result.calls[2]["operation"] == "pro_boundary_audit"

    budget = context.budget.snapshot()["gemini"]
    assert budget["selector_calls"] == 1
    assert budget["boundary_audit_calls"] == 1
    assert budget["committed_cost_usd"] == pytest.approx(0.0455)
    assert budget["inflight_reserved_cost_usd"] == 0.0


def test_text_only_pro_stops_after_two_malformed_structured_responses(
    monkeypatch,
) -> None:
    context = GenerationContext("fast", generation_id="pro-schema-retry-exhausted")
    generated: list[dict] = []

    monkeypatch.setattr(
        gemini_client,
        "count_request_tokens",
        lambda *_args, **_kwargs: 10_000,
    )

    def generate(_system, user, schema, **kwargs):
        generated.append({"user": user, "schema": schema, **kwargs})
        assert schema is gemini_segment._CompactBoundaryPlan
        telemetry = {
            "model": kwargs["model"],
            "operation": kwargs["operation"],
            "prompt_version": kwargs["prompt_version"],
            "thinking_level": kwargs["thinking_level"],
            "finish_reason": "STOP",
            "prompt_tokens": 10_000,
            "candidate_tokens": 100,
            "thought_tokens": 25,
            "total_tokens": 10_125,
        }
        _settle_mock_dispatch(kwargs, telemetry)
        return SimpleNamespace(
            text="{malformed",
            telemetry=telemetry,
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

    assert result.error is not None
    assert len(generated) == 2
    assert all(isinstance(call["user"], str) for call in generated)
    assert all(call["media_resolution"] is None for call in generated)
    assert len(result.calls) == 2
    assert result.calls[0]["schema_retry_attempt"] == 1
    assert result.calls[1]["schema_retry_attempt"] == 2
    assert result.calls[1]["schema_retry_exhausted"] is True
    assert all(call["video_grounded"] is False for call in result.calls)

    budget = context.budget.snapshot()["gemini"]
    assert budget["selector_calls"] == 1
    assert budget["boundary_audit_calls"] == 0
    assert budget["committed_cost_usd"] == pytest.approx(0.043)
    assert budget["inflight_reserved_cost_usd"] == 0.0


@pytest.mark.parametrize(
    ("mode", "selector_count", "cost_limit"),
    [("fast", 3, 1.50), ("slow", 5, 2.50)],
)
def test_pro_selector_and_audit_retry_headroom_stays_within_job_cost_ceiling(
    mode: str,
    selector_count: int,
    cost_limit: float,
) -> None:
    context = GenerationContext(mode, generation_id=f"pro-headroom-{mode}")
    expansion_reservation = context.reserve_gemini_call(
        operation="expansion",
        model="gemini-3.1-flash-lite",
        estimated_input_tokens=1_000,
        max_output_tokens=expand.PRACTICE_FAST_EXPAND_OUTPUT_TOKENS,
    )
    selector_reservations = []
    for _ in range(selector_count):
        selector_reservations.append(context.reserve_gemini_call(
            operation="pro_authoritative",
            model="gemini-3.1-pro-preview",
            estimated_input_tokens=30_000,
            max_output_tokens=gemini_segment._PRO_BOUNDARY_OUTPUT_TOKENS,
        ))

    assert all(
        reservation["reserved_output_tokens"]
        == gemini_segment._PRO_BOUNDARY_OUTPUT_TOKENS
        for reservation in selector_reservations
    )
    budget = context.budget.snapshot()["gemini"]
    assert budget["cost_exposure_usd"] <= cost_limit

    context.reconcile_gemini_call(
        model_used="gemini-3.1-flash-lite",
        usage={
            **expansion_reservation,
            "prompt_tokens": 1_000,
            "candidate_tokens": 100,
            "thought_tokens": 0,
            "dispatched": True,
        },
        dispatched=True,
    )
    selector_retry_reservations = []
    for reservation in selector_reservations:
        # A transport failure with no provider usage remains conservatively
        # billing-unknown, then one cost-only physical retry succeeds.
        context.reconcile_gemini_call(
            model_used="gemini-3.1-pro-preview",
            usage={
                **reservation,
                "dispatched": True,
            },
            dispatched=True,
        )
        retry_reservation = context.reserve_gemini_call(
            operation="pro_authoritative",
            model="gemini-3.1-pro-preview",
            estimated_input_tokens=30_000,
            max_output_tokens=gemini_segment._PRO_BOUNDARY_OUTPUT_TOKENS,
            count_logical_call=False,
        )
        selector_retry_reservations.append(retry_reservation)
        context.reconcile_gemini_call(
            model_used="gemini-3.1-pro-preview",
            usage={
                **retry_reservation,
                "prompt_tokens": 1_000,
                "candidate_tokens": 100,
                "thought_tokens": 0,
                "dispatched": True,
            },
            dispatched=True,
        )
        assert (
            context.budget.snapshot()["gemini"]["cost_exposure_usd"]
            <= cost_limit
        )

    audit_reservations = []
    audit_retry_reservations = []
    audit_output_tokens = gemini_segment._pro_boundary_audit_output_tokens(9)
    for _ in range(selector_count):
        reservation = context.reserve_gemini_call(
            operation="pro_boundary_audit",
            model="gemini-3.1-pro-preview",
            estimated_input_tokens=30_000,
            max_output_tokens=audit_output_tokens,
        )
        audit_reservations.append(reservation)
        assert (
            context.budget.snapshot()["gemini"]["cost_exposure_usd"]
            <= cost_limit
        )
        context.reconcile_gemini_call(
            model_used="gemini-3.1-pro-preview",
            usage={
                **reservation,
                "dispatched": True,
            },
            dispatched=True,
        )
        retry_reservation = context.reserve_gemini_call(
            operation="pro_boundary_audit",
            model="gemini-3.1-pro-preview",
            estimated_input_tokens=30_000,
            max_output_tokens=audit_output_tokens,
            count_logical_call=False,
        )
        audit_retry_reservations.append(retry_reservation)
        context.reconcile_gemini_call(
            model_used="gemini-3.1-pro-preview",
            usage={
                **retry_reservation,
                "prompt_tokens": 1_000,
                "candidate_tokens": 100,
                "thought_tokens": 0,
                "dispatched": True,
            },
            dispatched=True,
        )

    assert all(
        reservation["reserved_output_tokens"]
        == audit_output_tokens
        for reservation in [*audit_reservations, *audit_retry_reservations]
    )
    budget = context.budget.snapshot()["gemini"]
    assert budget["selector_calls"] == selector_count
    assert budget["boundary_audit_calls"] == selector_count
    assert budget["cost_exposure_usd"] <= cost_limit
    assert budget["cost_limit_usd"] == pytest.approx(cost_limit)


def test_retry_and_new_source_can_exceed_cost_target_but_selector_cap_remains(
    monkeypatch,
) -> None:
    context = GenerationContext("fast", generation_id="retry-physical-cap")
    peer_ticket = context.budget.reserve_gemini(
        model="gemini-3.5-flash",
        operation="flash_grounded_enrichment",
        estimated_cost_usd=1.00,
        max_physical_attempts=1,
    )
    fake = _RetryClient(
        _RetryHTTPError(503),
        _RetryResponse(
            gemini_segment._BoundaryPlan(topics=[]).model_dump_json()
        ),
    )
    retry_admission_started = threading.Event()
    finished = threading.Event()
    outcome: dict[str, object] = {}
    retry_admission_budgets: list[dict[str, object]] = []

    def reserve(**kwargs):
        ticket = context.reserve_gemini_call(**kwargs)
        if kwargs["count_logical_call"] is False:
            retry_admission_budgets.append(context.budget.snapshot()["gemini"])
            retry_admission_started.set()
        return ticket

    def run() -> None:
        try:
            outcome["result"] = gemini_segment._call_model(
                "system",
                "user",
                gemini_segment._BoundaryPlan,
                model="gemini-3.5-flash",
                thinking_level="medium",
                max_output_tokens=40_000,
                timeout_s=30.0,
                deadline_monotonic=time.monotonic() + 15.0,
                operation="flash_boundary_selector",
                prompt_version="physical_ticket_test_v1",
                cancelled=None,
                budget_reserve=reserve,
                budget_reconcile=context.reconcile_gemini_call,
                max_retries=1,
            )
        except Exception as exc:  # pragma: no cover - asserted below
            outcome["error"] = exc
        finally:
            finished.set()

    monkeypatch.setattr(gemini_client, "get_client", lambda: fake)
    monkeypatch.setattr(
        gemini_client,
        "count_request_tokens",
        lambda *_args, **_kwargs: 100,
    )
    monkeypatch.setattr(gemini_client, "_sleep_before_retry", lambda *_args: True)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    assert retry_admission_started.wait(timeout=1.0)
    admitted_retry_budget = retry_admission_budgets[0]
    assert admitted_retry_budget["hard_cost_cap_enabled"] is False
    assert admitted_retry_budget["cost_exposure_usd"] > admitted_retry_budget[
        "cost_target_usd"
    ]
    assert finished.wait(timeout=1.0)
    worker.join(timeout=1.0)

    assert "error" not in outcome
    assert len(fake.calls) == 2
    parsed, telemetry = outcome["result"]
    assert parsed.topics == []
    assert telemetry["physical_dispatches"] == 2
    assert telemetry["billing_unknown_attempts"] == 1
    completion_budget = context.budget.snapshot()["gemini"]
    assert completion_budget["selector_calls"] == 1
    assert completion_budget["inflight_reserved_cost_usd"] == pytest.approx(1.00)

    context.reserve_gemini_call(
        operation="flash_boundary_selector",
        model="gemini-3.5-flash",
        estimated_input_tokens=100,
        max_output_tokens=40_000,
    )
    context.reserve_gemini_call(
        operation="flash_boundary_selector",
        model="gemini-3.5-flash",
        estimated_input_tokens=100,
        max_output_tokens=200_000,
        count_logical_call=False,
    )
    context.reserve_gemini_call(
        operation="flash_boundary_selector",
        model="gemini-3.5-flash",
        estimated_input_tokens=100,
        max_output_tokens=100,
    )
    with pytest.raises(ProviderBudgetExceededError, match="selector budget"):
        context.reserve_gemini_call(
            operation="flash_boundary_selector",
            model="gemini-3.5-flash",
            estimated_input_tokens=100,
            max_output_tokens=100,
        )

    assert context.budget.reconcile_gemini(
        peer_ticket,
        actual_cost_usd=0.0,
    ) is True
    final_budget = context.budget.snapshot()["gemini"]
    assert final_budget["hard_cost_cap_enabled"] is False
    assert final_budget["cost_exposure_usd"] > final_budget[
        "completion_cost_target_usd"
    ]
    assert final_budget["selector_calls"] == final_budget["selector_limit"] == 3


def test_bound_selector_topics_only_payload_reduces_output_contract() -> None:
    full_plan = _newton_plan()
    bound_plan = gemini_segment._ContractBoundCompactBoundaryPlan(
        topics=list(full_plan.topics),
    )

    full_payload = full_plan.model_dump_json(by_alias=True)
    bound_payload = bound_plan.model_dump_json(by_alias=True)

    assert len(bound_payload) < len(full_payload)
    assert "request_intent" not in bound_payload
    assert json.loads(bound_payload) == {
        "topics": json.loads(full_payload)["topics"],
    }
