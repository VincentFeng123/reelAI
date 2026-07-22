import json
import threading

import pytest

from backend.app.clip_engine import expand
from backend.app.clip_engine.errors import (
    CancellationError,
    ProviderConfigurationError,
    ProviderRequestError,
    ProviderTransientError,
)


class _StatusFailure(RuntimeError):
    def __init__(self, status_code: int, *, retryable: bool):
        super().__init__(f"status {status_code}")
        self.status_code = status_code
        self.retryable = retryable


class _StatuslessPermanentFailure(RuntimeError):
    retryable = False


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
    assert result == {
        "corrected": "Physics",
        "queries": ["physics explained"],
        "provider_used": "gemini",
    }
    assert intent_contract["version"] == "expansion_intent_v2"
    assert intent_contract["request_intent"]["exact_request"] == "physics"


@pytest.mark.parametrize(
    "failure",
    [
        ConnectionError("connection reset"),
        ValueError("invalid local response contract"),
        _StatusFailure(408, retryable=False),
        _StatusFailure(429, retryable=False),
        _StatusFailure(500, retryable=False),
        _StatusFailure(503, retryable=False),
        _StatusFailure(599, retryable=False),
        _provider_failure(ProviderRequestError, status_code=503),
    ],
    ids=[
        "statusless-transport",
        "local-contract",
        "408",
        "429",
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


@pytest.mark.parametrize(
    "failure",
    [
        _StatusFailure(400, retryable=True),
        _StatusFailure(409, retryable=True),
        _StatusFailure(410, retryable=True),
        _StatusFailure(418, retryable=True),
        _StatusFailure(422, retryable=True),
        _StatusFailure(499, retryable=True),
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
        "499",
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

    result = expand.expand_query_practice_fast("physics", 1)

    assert calls == 1
    assert result == {
        "corrected": "physics",
        "queries": ["physics"],
        "provider_used": "literal_fallback",
    }
