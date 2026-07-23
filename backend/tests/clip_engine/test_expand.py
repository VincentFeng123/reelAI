import json
import threading

import pytest

from backend.app.clip_engine import expand
from backend.app.clip_engine.errors import (
    CancellationError,
    ModelUnavailableError,
    ProviderAuthenticationError,
    ProviderConfigurationError,
    ProviderError,
    ProviderQuotaError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderResponseValidationError,
    ProviderTransientError,
)


class _StatusFailure(RuntimeError):
    def __init__(self, status_code: int, *, retryable: bool):
        super().__init__(f"status {status_code}")
        self.status_code = status_code
        self.retryable = retryable


class _StatuslessPermanentFailure(RuntimeError):
    retryable = False


class _RateLimitWithHeaders(_StatusFailure):
    def __init__(self, retry_after: str):
        super().__init__(429, retryable=False)
        self.headers = {"Retry-After": retry_after}


class GeminiBlockedResponseError(RuntimeError):
    retryable = True


def _valid_expansion_json() -> str:
    return json.dumps({
        "corrected": "Physics",
        "summary_preserved_constraint_ids": ["subject"],
        "intent_constraints": [{
            "constraint_id": "subject",
            "kind": "subject",
            "source_phrase": "physics",
            "requirement": "Teach physics",
        }],
        "reverse_coverage_complete": True,
        "reverse_coverage_constraint_ids": ["subject"],
        "acquisition_obligation_constraint_ids": ["subject"],
        "coordinated_groups": [],
        "queries": [{
            "text": "physics explained",
            "preserved_constraint_ids": ["subject"],
        }],
    })


def _provider_failure(error_type, *, status_code=None):
    return error_type(
        "provider failure",
        provider="gemini",
        operation="expansion",
        status_code=status_code,
    )


def test_deterministic_expand_includes_educational_templates() -> None:
    output = expand.expand_query("photosynthesis", 10)
    queries = [query.casefold() for query in output["queries"]]
    assert "photosynthesis explained" in queries
    assert "photosynthesis lecture" in queries
    assert "how photosynthesis works" in queries
    assert "photosynthesis course" in queries
    assert "photosynthesis tutorial" in queries
    assert output["queries"][0] == "photosynthesis"
    assert output["provider_used"] == "deterministic"


def test_expand_query_has_no_gemini_provider_surface(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "must-not-be-used")
    output = expand.expand_query("calculus", 4)
    assert output["corrected"] == "calculus"
    assert len(output["queries"]) == 4
    assert not hasattr(expand, "_gemini_expand_raw")
    assert not hasattr(expand, "_gemini_expand_raw_async")


def test_level_templates_are_deterministic_and_truthful() -> None:
    beginner = expand.expand_query("physics", 4, level="beginner")["queries"]
    advanced = expand.expand_query("physics", 4, level="advanced")["queries"]
    intermediate = expand.expand_query("physics", 4, level="intermediate")["queries"]
    assert beginner[1:] == [
        "introduction to physics", "physics basics", "physics explained"
    ]
    assert advanced[1:] == [
        "advanced physics", "graduate physics lecture", "physics deep dive"
    ]
    assert "graduate physics lecture" not in intermediate
    assert "physics for beginners" not in intermediate


def test_unicode_normalization_prevents_duplicate_queries() -> None:
    assert expand._normalize(["CAFÉ", "cafe\u0301", "  café  "], 5) == ["CAFÉ"]


def test_intent_tokens_preserve_attached_language_punctuation_only() -> None:
    assert expand._intent_tokens("C C++ C#") == ["C", "c++", "c#"]
    assert expand._intent_tokens("C + + memory # topic") == [
        "C",
        "memory",
        "topic",
    ]
    assert not expand._contains_token_phrase("C memory management", "C++")
    assert not expand._contains_token_phrase("C async await", "C#")
    assert not expand._contains_token_phrase("JavaScript || operator", "JavaScript &&")
    assert not expand._contains_token_phrase("JavaScript ?? operator", "JavaScript ||")
    assert not expand._contains_token_phrase("Swift String type", "Swift String?")
    assert expand._contains_token_phrase("C∗-algebra basics", "C* algebra")
    assert not expand._contains_token_phrase("C algebra basics", "C* algebra")


def test_expand_observes_cancellation_without_starting_provider_work() -> None:
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(CancellationError):
        expand.expand_query("calculus", 3, should_cancel=cancelled.is_set)


def test_practice_fast_healthy_expansion_has_one_call_and_no_retry_wait(
    monkeypatch,
) -> None:
    calls = 0

    def generate(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return _valid_expansion_json()

    monkeypatch.setattr(expand, "_practice_fast_gemini_raw", generate)
    monkeypatch.setattr(
        expand,
        "_practice_fast_failure_is_retryable",
        lambda _exc: pytest.fail("healthy expansion entered retry handling"),
    )
    result = expand.expand_query_practice_fast("physics", 1)

    assert calls == 1
    intent_contract = result.pop("intent_contract")
    acquisition_ids = result.pop("acquisition_obligation_constraint_ids")
    query_metadata = result.pop("query_metadata")
    assert result == {
        "corrected": "Physics",
        "queries": ["physics explained"],
        "provider_used": "gemini",
    }
    assert intent_contract["version"] == "expansion_intent_v2"
    assert intent_contract["request_intent"]["exact_request"] == "physics"
    assert acquisition_ids == ["subject"]
    assert query_metadata == [{
        "text": "physics explained",
        "preserved_constraint_ids": ["subject"],
        "intent_obligation_keys": sorted(
            expand.trusted_intent_obligation_keys(intent_contract)
        ),
        "focused_intent_obligation_keys": [],
        "covers_all_intent_constraints": True,
    }]


@pytest.mark.parametrize(
    "failure",
    [
        ConnectionError("connection reset"),
        ValueError("invalid local response contract"),
        _StatusFailure(408, retryable=False),
        _StatusFailure(499, retryable=False),
        _StatusFailure(500, retryable=False),
        _StatusFailure(503, retryable=False),
        _StatusFailure(599, retryable=False),
        _provider_failure(ProviderRequestError, status_code=503),
    ],
    ids=[
        "statusless-transport",
        "local-contract",
        "408",
        "499",
        "500",
        "503",
        "599",
        "status-overrides-stale-false",
    ],
)
def test_practice_fast_retryable_failure_recovers_on_second_call(
    monkeypatch,
    failure,
) -> None:
    calls = 0

    def fail_then_recover(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise failure
        return _valid_expansion_json()

    monkeypatch.setattr(
        expand,
        "_practice_fast_gemini_raw",
        fail_then_recover,
    )

    result = expand.expand_query_practice_fast("physics", 1)

    assert calls == 2
    assert result["provider_used"] == "gemini"
    assert result["queries"] == ["physics explained"]


def test_practice_fast_rate_limit_defers_to_durable_retry(
    monkeypatch,
) -> None:
    calls: list[str] = []

    def rate_limited(*_args, model, **_kwargs):
        calls.append(model)
        raise _StatusFailure(429, retryable=False)

    monkeypatch.setattr(expand, "_practice_fast_gemini_raw", rate_limited)

    with pytest.raises(ProviderRateLimitError) as exc_info:
        expand.expand_query_practice_fast("physics", 1)

    assert calls == [expand.PRACTICE_FAST_EXPAND_MODEL]
    assert exc_info.value.status_code == 429
    assert exc_info.value.retryable is True


def test_practice_fast_semantic_retry_receives_compact_validation_feedback(
    monkeypatch,
) -> None:
    calls: list[dict] = []
    invalid = json.loads(_valid_expansion_json())
    invalid["intent_constraints"][0]["kind"] = "relationship"
    invalid["intent_constraints"][0]["relationship_topology"] = "not_applicable"

    def fail_then_recover(*_args, **kwargs):
        calls.append(kwargs)
        return json.dumps(invalid) if len(calls) == 1 else _valid_expansion_json()

    monkeypatch.setattr(expand, "_practice_fast_gemini_raw", fail_then_recover)

    result = expand.expand_query_practice_fast("physics", 1)

    assert result["provider_used"] == "gemini"
    assert len(calls) == 2
    assert "validation_feedback" not in calls[0]
    assert "relationship topology must match" in calls[1][
        "validation_feedback"
    ]
    assert "input_value" not in calls[1]["validation_feedback"]
    assert expand.PRACTICE_FAST_EXPAND_ATTEMPTS == 3


def test_practice_fast_incomplete_reverse_coverage_uses_existing_correction(
    monkeypatch,
) -> None:
    calls: list[dict] = []
    invalid = json.loads(_valid_expansion_json())
    invalid["reverse_coverage_constraint_ids"] = []

    def fail_then_recover(*_args, **kwargs):
        calls.append(kwargs)
        return json.dumps(invalid) if len(calls) == 1 else _valid_expansion_json()

    monkeypatch.setattr(expand, "_practice_fast_gemini_raw", fail_then_recover)

    result = expand.expand_query_practice_fast("physics", 1)

    assert result["provider_used"] == "gemini"
    assert len(calls) == 2
    assert "validation_feedback" not in calls[0]
    assert "reverse_coverage_constraint_ids" in calls[1]["validation_feedback"]


def test_practice_fast_coordinated_member_omission_uses_existing_correction(
    monkeypatch,
) -> None:
    request = "Explain duty and breach in negligence."
    constraints = [
        {
            "constraint_id": constraint_id,
            "kind": kind,
            "source_phrase": phrase,
            "requirement": requirement,
        }
        for constraint_id, kind, phrase, requirement in (
            ("task", "task", "Explain", "Explain the requested concepts"),
            ("duty", "scope", "duty", "Explain duty"),
            ("breach", "scope", "breach", "Explain breach"),
            ("subject", "subject", "negligence", "Teach negligence"),
        )
    ]

    def response(acquisition_ids: list[str]) -> str:
        constraint_ids = [
            constraint["constraint_id"] for constraint in constraints
        ]
        return json.dumps({
            "corrected": request,
            "intent_constraints": constraints,
            "joint_structures": [],
            "summary_preserved_constraint_ids": constraint_ids,
            "reverse_coverage_complete": True,
            "reverse_coverage_constraint_ids": list(reversed(constraint_ids)),
            "acquisition_obligation_constraint_ids": acquisition_ids,
            "coordinated_groups": [{
                "member_constraint_ids": ["duty", "breach"],
            }],
            "queries": [{
                "text": request,
                "preserved_constraint_ids": constraint_ids,
            }],
        })

    calls: list[dict] = []

    def fail_then_recover(*_args, **kwargs):
        calls.append(kwargs)
        return response(
            ["task", "duty", "subject"]
            if len(calls) == 1
            else ["task", "duty", "breach", "subject"]
        )

    monkeypatch.setattr(expand, "_practice_fast_gemini_raw", fail_then_recover)

    result = expand.expand_query_practice_fast(request, 1)

    assert result["acquisition_obligation_constraint_ids"] == [
        "task",
        "duty",
        "breach",
        "subject",
    ]
    assert len(calls) == 2
    assert "acquisition obligations must retain" in calls[1][
        "validation_feedback"
    ]


def test_acquisition_obligations_exclude_gemini_marked_contextual_constraints(
    monkeypatch,
) -> None:
    request = "Teach calculus to a beginner in a concise video."
    constraints = [
        {
            "constraint_id": constraint_id,
            "kind": kind,
            "source_phrase": phrase,
            "requirement": requirement,
        }
        for constraint_id, kind, phrase, requirement in (
            ("task", "task", "Teach", "Teach calculus"),
            ("subject", "subject", "calculus", "Teach calculus"),
            ("level", "scope", "beginner", "Target a beginner"),
            ("format", "format", "concise video", "Use a concise video"),
        )
    ]
    constraint_ids = [
        constraint["constraint_id"] for constraint in constraints
    ]
    monkeypatch.setattr(
        expand,
        "_practice_fast_gemini_raw",
        lambda *_args, **_kwargs: json.dumps({
            "corrected": request,
            "intent_constraints": constraints,
            "joint_structures": [],
            "summary_preserved_constraint_ids": constraint_ids,
            "reverse_coverage_complete": True,
            "reverse_coverage_constraint_ids": list(reversed(constraint_ids)),
            "acquisition_obligation_constraint_ids": ["task", "subject"],
            "coordinated_groups": [],
            "queries": [{
                "text": request,
                "preserved_constraint_ids": constraint_ids,
            }],
        }),
    )

    result = expand.expand_query_practice_fast(request, 1)

    contract = result["intent_contract"]
    selected_keys = expand.trusted_intent_obligation_keys(
        contract,
        result["acquisition_obligation_constraint_ids"],
    )
    assert len(selected_keys) == 2
    assert len(expand.trusted_intent_obligation_keys(contract)) == 4


def test_practice_fast_order_mismatch_is_corrected_not_normalized(
    monkeypatch,
) -> None:
    calls: list[dict] = []
    invalid = json.loads(_valid_expansion_json())
    invalid["intent_constraints"][0]["kind"] = "format"
    invalid["intent_constraints"][0]["relationship_topology"] = "ordered"

    def fail_then_recover(*_args, **kwargs):
        calls.append(kwargs)
        return json.dumps(invalid) if len(calls) == 1 else _valid_expansion_json()

    monkeypatch.setattr(expand, "_practice_fast_gemini_raw", fail_then_recover)

    result = expand.expand_query_practice_fast("physics", 1)

    assert result["provider_used"] == "gemini"
    assert len(calls) == 2
    assert "relationship topology must match" in calls[1][
        "validation_feedback"
    ]


def test_practice_fast_warning_redacts_rejected_model_values(
    monkeypatch,
    caplog,
) -> None:
    secret = "LEARNER_SECRET_ENUM_VALUE"
    invalid = json.loads(_valid_expansion_json())
    invalid["intent_constraints"][0]["kind"] = secret
    monkeypatch.setattr(
        expand,
        "_practice_fast_gemini_raw",
        lambda *_args, **_kwargs: json.dumps(invalid),
    )
    caplog.set_level("WARNING", logger=expand.__name__)

    result = expand.expand_query_practice_fast("physics", 1)

    assert result == {
        "corrected": "physics",
        "queries": ["physics"],
        "provider_used": "literal_fallback",
    }
    assert secret not in caplog.text
    assert "input_value" not in caplog.text
    assert "type=ValidationError" in caplog.text
    assert expand.PRACTICE_FAST_EXPAND_MODEL in caplog.text
    assert expand.PRACTICE_FAST_EXPAND_FALLBACK_MODEL in caplog.text


@pytest.mark.parametrize(
    "failure",
    [
        _StatusFailure(400, retryable=True),
        _StatusFailure(409, retryable=True),
        _StatusFailure(410, retryable=True),
        _StatusFailure(418, retryable=True),
        _StatusFailure(422, retryable=True),
        _provider_failure(ProviderTransientError, status_code=422),
        _provider_failure(ProviderConfigurationError),
        _provider_failure(ProviderRequestError),
        GeminiBlockedResponseError("blocked response"),
        _StatuslessPermanentFailure("permanent failure"),
    ],
    ids=[
        "400",
        "409",
        "410",
        "418",
        "422",
        "status-overrides-stale-true",
        "configuration",
        "statusless-provider-request",
        "blocked",
        "statusless-explicit-permanent",
    ],
)
def test_practice_fast_permanent_failure_does_not_retry(
    monkeypatch,
    failure,
) -> None:
    calls = 0

    def fail(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise failure

    monkeypatch.setattr(expand, "_practice_fast_gemini_raw", fail)

    with pytest.raises(ProviderError) as exc_info:
        expand.expand_query_practice_fast("physics", 1)

    assert calls == 1
    assert exc_info.value.retryable is False


def test_practice_fast_exhausted_contract_is_typed_and_never_literal(
    monkeypatch,
) -> None:
    calls: list[str] = []

    def invalid(*_args, model, **_kwargs):
        calls.append(model)
        raise ValueError("selector rejected universal intent contract")

    monkeypatch.setattr(expand, "_practice_fast_gemini_raw", invalid)
    monkeypatch.setattr(expand.config, "GEMINI_MODEL", "gemini-3.1-pro-preview")

    with pytest.raises(ProviderResponseValidationError) as exc_info:
        expand.expand_query_practice_fast("cross-domain learning request", 3)

    assert calls == [
        "gemini-3.1-flash-lite",
        "gemini-3.1-flash-lite",
        expand.PRACTICE_FAST_EXPAND_FALLBACK_MODEL,
    ]
    assert exc_info.value.retryable is True
    assert exc_info.value.operation == "expansion"


def test_practice_fast_exhausted_provider_499_is_typed_transient(
    monkeypatch,
) -> None:
    calls = 0

    def provider_cancelled(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise _StatusFailure(499, retryable=False)

    monkeypatch.setattr(expand, "_practice_fast_gemini_raw", provider_cancelled)

    with pytest.raises(ProviderTransientError) as exc_info:
        expand.expand_query_practice_fast("physics", 1)

    assert calls == expand.PRACTICE_FAST_EXPAND_ATTEMPTS
    assert exc_info.value.status_code == 499
    assert exc_info.value.retryable is True


def test_practice_fast_application_cancellation_after_provider_499_does_not_retry(
    monkeypatch,
) -> None:
    calls = 0
    cancelled = False

    def provider_cancelled(*_args, **_kwargs):
        nonlocal calls, cancelled
        calls += 1
        cancelled = True
        raise _StatusFailure(499, retryable=False)

    monkeypatch.setattr(expand, "_practice_fast_gemini_raw", provider_cancelled)

    with pytest.raises(CancellationError):
        expand.expand_query_practice_fast(
            "physics",
            1,
            should_cancel=lambda: cancelled,
        )

    assert calls == 1


def test_practice_fast_typed_gemini_cancellation_does_not_retry(
    monkeypatch,
) -> None:
    calls = 0
    failure = expand.gemini_client.GeminiCancelledError(
        "application cancelled",
        expand.gemini_client.GeminiCallTelemetry(
            model=expand.PRACTICE_FAST_EXPAND_MODEL,
            operation="expansion",
            prompt_version="test",
            thinking_level="low",
            latency_ms=1.0,
            retries=0,
            finish_reason=None,
            prompt_tokens=None,
            candidate_tokens=None,
            thought_tokens=None,
            total_tokens=None,
            provider_status_code=499,
            retryable=False,
        ),
    )

    def application_cancelled(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise failure

    monkeypatch.setattr(expand, "_practice_fast_gemini_raw", application_cancelled)

    with pytest.raises(CancellationError):
        expand.expand_query_practice_fast("physics", 1)

    assert calls == 1


def test_practice_fast_model_404_uses_flash_fallback(monkeypatch) -> None:
    calls: list[str] = []

    def unavailable_then_recover(*_args, model, **_kwargs):
        calls.append(model)
        if len(calls) == 1:
            raise _StatusFailure(404, retryable=False)
        return _valid_expansion_json()

    monkeypatch.setattr(
        expand,
        "_practice_fast_gemini_raw",
        unavailable_then_recover,
    )

    result = expand.expand_query_practice_fast("physics", 1)

    assert calls == [
        expand.PRACTICE_FAST_EXPAND_MODEL,
        expand.PRACTICE_FAST_EXPAND_FALLBACK_MODEL,
    ]
    assert result["provider_used"] == "gemini"


def test_practice_fast_recovery_exhaustion_uses_exact_request(
    monkeypatch,
) -> None:
    calls: list[str] = []

    def repeated_query(*_args, model, **_kwargs):
        calls.append(model)
        return _valid_expansion_json()

    monkeypatch.setattr(expand, "_practice_fast_gemini_raw", repeated_query)

    result = expand.expand_query_practice_fast(
        "physics",
        1,
        tried_queries=["physics", "physics explained"],
        recovery_reason=expand.RECOVERY_REASON_ZERO_SEARCH_RESULTS,
    )

    assert calls == [
        expand.PRACTICE_FAST_EXPAND_MODEL,
        expand.PRACTICE_FAST_EXPAND_MODEL,
        expand.PRACTICE_FAST_EXPAND_FALLBACK_MODEL,
    ]
    assert result == {
        "corrected": "physics",
        "queries": ["physics"],
        "provider_used": "literal_fallback",
    }


@pytest.mark.parametrize(
    ("status", "expected_type", "retryable"),
    [
        (400, ProviderRequestError, False),
        (401, ProviderAuthenticationError, False),
        (402, ProviderQuotaError, False),
        (403, ProviderAuthenticationError, False),
        (404, ModelUnavailableError, False),
        (408, ProviderTransientError, True),
        (429, ProviderRateLimitError, True),
        (499, ProviderTransientError, True),
        (503, ProviderTransientError, True),
    ],
)
def test_practice_fast_provider_status_mapping(
    status,
    expected_type,
    retryable,
) -> None:
    failure = expand._practice_fast_provider_error(
        _StatusFailure(status, retryable=False)
    )

    assert isinstance(failure, expected_type)
    assert failure.status_code == status
    assert failure.retryable is retryable
    assert failure.detail == "_StatusFailure"


def test_practice_fast_rate_limit_preserves_provider_retry_after() -> None:
    failure = expand._practice_fast_provider_error(
        _RateLimitWithHeaders("12")
    )

    assert isinstance(failure, ProviderRateLimitError)
    assert failure.retry_after_sec == pytest.approx(12.0)
