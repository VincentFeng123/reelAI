from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors as genai_errors
from pydantic import BaseModel, Field

from backend import gemini_client as gc


class _Schema(BaseModel):
    ok: bool


class _ConstrainedSchema(BaseModel):
    name: str = Field(min_length=1)
    labels: list[str] = Field(min_length=1, max_length=2)


class _FakeResponse:
    def __init__(self, text: str = '{"ok": true}', *, finish_reason="STOP",
                 prompt_tokens=11, candidate_tokens=7, thought_tokens=5,
                 total_tokens=23, cached_tokens=3):
        self.text = text
        self.candidates = [SimpleNamespace(
            finish_reason=SimpleNamespace(value=finish_reason),
        )]
        self.usage_metadata = SimpleNamespace(
            prompt_token_count=prompt_tokens,
            candidates_token_count=candidate_tokens,
            thoughts_token_count=thought_tokens,
            total_token_count=total_tokens,
            cached_content_token_count=cached_tokens,
        )


class _FakeModels:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _FakeClient:
    def __init__(self, *outcomes):
        self.models = _FakeModels(outcomes)


class _HTTPError(RuntimeError):
    def __init__(self, status_code: int, *, retry_after: str | None = None):
        super().__init__(f"HTTP status {status_code}")
        self.status_code = status_code
        self.response = SimpleNamespace(
            headers={} if retry_after is None else {"Retry-After": retry_after},
        )


class _RemoteProtocolError(RuntimeError):
    pass


def _enum_value(value) -> str:
    return str(getattr(value, "value", value)).lower()


def _call_v3(monkeypatch, fake, **overrides):
    monkeypatch.setattr(gc, "get_client", lambda: fake)
    schema = overrides.pop("schema", _Schema)
    kwargs = {
        "model": "gemini-3.5-flash",
        "thinking_level": "medium",
        "max_output_tokens": 24_576,
        "timeout_s": 90.0,
        "operation": "flash_single",
        "prompt_version": "flash_single_v1",
    }
    kwargs.update(overrides)
    kwargs.setdefault("deadline_monotonic", None)
    return gc.generate_json_v3("system", "user", schema, **kwargs)


@pytest.mark.parametrize(
    "operation,level,cap,timeout_s,model",
    [
        ("flash_single", "medium", 24_576, 45.0, "gemini-3.5-flash"),
        ("flash_boundary", "medium", 12_288, 45.0, "gemini-3.5-flash"),
        ("flash_enrichment", "low", 24_576, 25.0, "gemini-3.5-flash"),
        ("pro_fallback", "high", 24_576, 90.0, "gemini-3.1-pro-preview"),
    ],
)
def test_gemini3_operation_config_omits_sampling_and_sdk_retries(
    monkeypatch, operation, level, cap, timeout_s, model,
):
    fake = _FakeClient(_FakeResponse())
    result = _call_v3(
        monkeypatch,
        fake,
        operation=operation,
        thinking_level=level,
        max_output_tokens=cap,
        timeout_s=timeout_s,
        model=model,
    )

    assert result.text == '{"ok": true}'
    call = fake.models.calls[0]
    assert call["model"] == model
    cfg = call["config"]
    assert cfg.temperature is None and cfg.top_p is None and cfg.top_k is None
    assert cfg.response_schema is None
    assert cfg.response_json_schema["required"] == ["ok"]
    assert cfg.max_output_tokens == cap
    assert _enum_value(cfg.thinking_config.thinking_level).endswith(level)
    assert cfg.thinking_config.thinking_budget is None
    assert int(timeout_s * 1000) - 100 <= cfg.http_options.timeout <= int(timeout_s * 1000)
    assert cfg.http_options.retry_options.attempts == 1


def test_gemini3_uses_provider_compatible_json_schema_and_keeps_required_types(monkeypatch):
    fake = _FakeClient(_FakeResponse())
    _call_v3(monkeypatch, fake, schema=_ConstrainedSchema)

    schema = fake.models.calls[0]["config"].response_json_schema
    rendered = str(schema)
    assert "minLength" not in rendered
    assert "maxItems" not in rendered
    assert schema["required"] == ["name", "labels"]
    assert schema["properties"]["name"]["type"] == "string"
    assert schema["properties"]["labels"]["minItems"] == 1


def test_gemini3_returns_immutable_usage_and_finish_telemetry(monkeypatch):
    fake = _FakeClient(_FakeResponse())
    result = _call_v3(monkeypatch, fake)

    telemetry = result.telemetry
    assert telemetry.model == "gemini-3.5-flash"
    assert telemetry.operation == "flash_single"
    assert telemetry.prompt_version == "flash_single_v1"
    assert telemetry.thinking_level == "medium"
    assert telemetry.finish_reason == "STOP"
    assert (
        telemetry.prompt_tokens,
        telemetry.candidate_tokens,
        telemetry.thought_tokens,
        telemetry.total_tokens,
        telemetry.cached_tokens,
    ) == (11, 7, 5, 23, 3)
    assert telemetry.retries == 0 and telemetry.latency_ms >= 0
    assert telemetry.provider_error_type is None
    assert telemetry.provider_status_code is None
    assert telemetry.retryable is None
    assert telemetry.as_dict()["total_tokens"] == 23
    with pytest.raises(FrozenInstanceError):
        telemetry.retries = 9
    with pytest.raises(FrozenInstanceError):
        result.text = "changed"


def test_gemini3_tolerates_absent_usage_and_finish_metadata(monkeypatch):
    response = SimpleNamespace(
        text='{"ok": true}', candidates=[], usage_metadata=None,
    )
    result = _call_v3(monkeypatch, _FakeClient(response))
    assert result.telemetry.finish_reason is None
    assert result.telemetry.prompt_tokens is None
    assert result.telemetry.candidate_tokens is None
    assert result.telemetry.thought_tokens is None
    assert result.telemetry.total_tokens is None
    assert result.telemetry.cached_tokens is None


@pytest.mark.parametrize("status_code", [503, 504])
def test_gemini3_retries_one_transient_error_with_short_jitter(
    monkeypatch, status_code,
):
    fake = _FakeClient(_HTTPError(status_code), _FakeResponse())
    sleeps = []
    monkeypatch.setattr(gc.time, "sleep", sleeps.append)
    monkeypatch.setattr(gc.random, "uniform", lambda lo, hi: 1.0)

    result = _call_v3(monkeypatch, fake, max_retries=1)

    assert len(fake.models.calls) == 2
    assert result.telemetry.retries == 1
    assert sleeps == [1.0]
    assert result.telemetry.error_history == ({
        "provider_error_type": "_HTTPError",
        "provider_status_code": status_code,
        "retryable": True,
    },)
    assert all(call["config"].http_options.retry_options.attempts == 1
               for call in fake.models.calls)


def test_gemini3_retry_status_policy_still_retries_allowed_503(monkeypatch):
    fake = _FakeClient(_HTTPError(503), _FakeResponse())
    monkeypatch.setattr(gc.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(gc.random, "uniform", lambda lower, _upper: lower)

    result = _call_v3(
        monkeypatch,
        fake,
        max_retries=1,
        retry_status_codes=frozenset({503}),
    )

    assert len(fake.models.calls) == 2
    assert result.telemetry.retries == 1


@pytest.mark.parametrize("status_code", [408, 429, 500, 502, 504])
def test_gemini3_retry_status_policy_declines_other_transient_statuses(
    monkeypatch,
    status_code,
):
    fake = _FakeClient(_HTTPError(status_code), _FakeResponse())

    with pytest.raises(gc.GeminiTransportError) as caught:
        _call_v3(
            monkeypatch,
            fake,
            max_retries=1,
            retry_status_codes=frozenset({503}),
        )

    assert len(fake.models.calls) == 1
    assert caught.value.telemetry.retryable is True
    assert caught.value.telemetry.provider_status_code == status_code


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("status 503"),
        RuntimeError("HTTP status 504"),
        _RemoteProtocolError("unexpected EOF"),
    ],
)
def test_gemini3_retry_status_policy_declines_statusless_failures(
    monkeypatch,
    error,
):
    fake = _FakeClient(error, _FakeResponse())

    with pytest.raises(gc.GeminiTransportError) as caught:
        _call_v3(
            monkeypatch,
            fake,
            max_retries=1,
            retry_status_codes=frozenset({503}),
        )

    assert len(fake.models.calls) == 1
    assert caught.value.telemetry.retryable is True
    assert caught.value.telemetry.provider_status_code is None


def test_gemini3_honors_retry_after_over_jitter(monkeypatch):
    fake = _FakeClient(_HTTPError(429, retry_after="2.5"), _FakeResponse())
    sleeps = []
    monkeypatch.setattr(gc.time, "sleep", sleeps.append)
    monkeypatch.setattr(gc.random, "uniform", lambda _lo, _hi: 0.9)

    result = _call_v3(monkeypatch, fake, max_retries=1)

    assert result.telemetry.retries == 1
    assert sleeps == [2.5]


@pytest.mark.parametrize(
    "error",
    [
        _RemoteProtocolError("unexpected EOF"),
        RuntimeError("Server disconnected without sending a response"),
        RuntimeError("connection reset by peer"),
    ],
)
def test_gemini3_retries_common_remote_disconnects(monkeypatch, error):
    fake = _FakeClient(error, _FakeResponse())
    monkeypatch.setattr(gc.time, "sleep", lambda _: None)

    result = _call_v3(monkeypatch, fake, max_retries=1)

    assert len(fake.models.calls) == 2
    assert result.telemetry.retries == 1


@pytest.mark.parametrize(
    "error",
    [
        httpx.ReadError("socket read failed"),
        httpx.WriteError("socket write failed"),
        httpx.CloseError("socket close failed"),
        httpx.ProxyError("proxy handshake failed"),
        httpx.DecodingError("invalid compressed response"),
        genai_errors.UnknownApiResponseError("malformed provider response"),
    ],
)
def test_gemini3_retries_sdk_transport_and_malformed_response_once(
    monkeypatch, error,
):
    fake = _FakeClient(error, _FakeResponse())
    monkeypatch.setattr(gc.time, "sleep", lambda _: None)

    result = _call_v3(monkeypatch, fake, max_retries=1)

    assert len(fake.models.calls) == 2
    assert result.telemetry.retries == 1


@pytest.mark.parametrize(
    "status,status_name,message",
    [
        (400, "INVALID_ARGUMENT", "response schema field unavailable"),
        (401, "UNAUTHENTICATED", "API key is invalid"),
        (403, "PERMISSION_DENIED", "permission denied"),
        (404, "NOT_FOUND", "model not found"),
        (422, "INVALID_ARGUMENT", "invalid JSON schema"),
    ],
)
def test_gemini3_never_retries_provider_4xx_schema_or_auth_errors(
    monkeypatch, status, status_name, message,
):
    error = genai_errors.ClientError(
        status,
        {"error": {"code": status, "status": status_name, "message": message}},
    )
    fake = _FakeClient(error, _FakeResponse())

    with pytest.raises(gc.GeminiTransportError) as exc_info:
        _call_v3(monkeypatch, fake, max_retries=1)

    assert len(fake.models.calls) == 1
    telemetry = exc_info.value.telemetry
    assert telemetry.retries == 0
    assert telemetry.provider_error_type == "ClientError"
    assert telemetry.provider_status_code == status
    assert telemetry.retryable is False


def test_gemini3_retries_google_rate_limit_once(monkeypatch):
    error = genai_errors.ClientError(
        429,
        {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED"}},
    )
    fake = _FakeClient(error, _FakeResponse())
    monkeypatch.setattr(gc.time, "sleep", lambda _: None)

    result = _call_v3(monkeypatch, fake, max_retries=1)

    assert len(fake.models.calls) == 2
    assert result.telemetry.retries == 1


def test_gemini3_cancellation_during_jitter_prevents_retry(monkeypatch):
    fake = _FakeClient(_HTTPError(503, retry_after="30"), _FakeResponse())
    state = {"cancelled": False}
    sleeps = []

    def cancel_during_sleep(seconds):
        sleeps.append(seconds)
        state["cancelled"] = True

    monkeypatch.setattr(gc.time, "sleep", cancel_during_sleep)
    with pytest.raises(gc.GeminiCancelledError) as exc_info:
        _call_v3(
            monkeypatch, fake, max_retries=1,
            cancelled=lambda: state["cancelled"],
        )
    assert len(fake.models.calls) == 1
    assert sleeps == [0.05]
    assert exc_info.value.telemetry.retries == 0


def test_gemini3_one_retry_ceiling_remains_available(monkeypatch):
    fake = _FakeClient(_HTTPError(503), _HTTPError(503), _FakeResponse())
    monkeypatch.setattr(gc.time, "sleep", lambda _: None)

    with pytest.raises(gc.GeminiTransportError) as exc_info:
        _call_v3(monkeypatch, fake, max_retries=1)

    assert len(fake.models.calls) == 2
    assert exc_info.value.telemetry.retries == 1
    assert exc_info.value.telemetry.provider_error_type == "_HTTPError"
    assert exc_info.value.telemetry.provider_status_code == 503
    assert exc_info.value.telemetry.retryable is True
    assert exc_info.value.telemetry.error_history == (
        {
            "provider_error_type": "_HTTPError",
            "provider_status_code": 503,
            "retryable": True,
        },
        {
            "provider_error_type": "_HTTPError",
            "provider_status_code": 503,
            "retryable": True,
        },
    )


def test_gemini3_503_can_succeed_on_third_attempt_with_exponential_jitter(monkeypatch):
    fake = _FakeClient(_HTTPError(503), _HTTPError(503), _FakeResponse())
    sleeps = []
    jitter_ranges = []
    monkeypatch.setattr(gc.time, "sleep", sleeps.append)

    def lowest_jitter(lo, hi):
        jitter_ranges.append((lo, hi))
        return lo

    monkeypatch.setattr(gc.random, "uniform", lowest_jitter)

    result = _call_v3(monkeypatch, fake, max_retries=2)

    assert len(fake.models.calls) == 3
    assert sleeps == [1.0, 2.0]
    assert jitter_ranges == [(1.0, 2.0), (2.0, 4.0)]
    assert result.telemetry.retries == 2
    assert [item["provider_status_code"] for item in result.telemetry.error_history] == [
        503,
        503,
    ]


def test_gemini3_three_503s_fail_after_two_exponential_retries(monkeypatch):
    fake = _FakeClient(_HTTPError(503), _HTTPError(503), _HTTPError(503))
    sleeps = []
    monkeypatch.setattr(gc.time, "sleep", sleeps.append)
    monkeypatch.setattr(gc.random, "uniform", lambda lo, _hi: lo)

    with pytest.raises(gc.GeminiTransportError) as exc_info:
        _call_v3(monkeypatch, fake, max_retries=2)

    assert len(fake.models.calls) == 3
    assert sleeps == [1.0, 2.0]
    assert exc_info.value.telemetry.retries == 2
    assert len(exc_info.value.telemetry.error_history) == 3


@pytest.mark.parametrize("status_code", [429, 500, 502, 504])
def test_gemini3_non_503_remains_capped_at_one_retry(monkeypatch, status_code):
    fake = _FakeClient(
        _HTTPError(status_code),
        _HTTPError(status_code),
        _FakeResponse(),
    )
    monkeypatch.setattr(gc.time, "sleep", lambda _seconds: None)

    with pytest.raises(gc.GeminiTransportError) as exc_info:
        _call_v3(monkeypatch, fake, max_retries=2)

    assert len(fake.models.calls) == 2
    assert exc_info.value.telemetry.retries == 1


def test_gemini3_second_503_retry_honors_retry_after(monkeypatch):
    fake = _FakeClient(
        _HTTPError(503),
        _HTTPError(503, retry_after="3.5"),
        _FakeResponse(),
    )
    sleeps = []
    monkeypatch.setattr(gc.time, "sleep", sleeps.append)
    monkeypatch.setattr(gc.random, "uniform", lambda lo, _hi: lo)

    result = _call_v3(monkeypatch, fake, max_retries=2)

    assert result.telemetry.retries == 2
    assert sleeps == [1.0, 3.5]


def test_gemini3_second_503_retry_respects_absolute_deadline(monkeypatch):
    clock = {"now": 0.0}

    class Models:
        def __init__(self):
            self.calls = []

        def generate_content(self, model, contents, config):
            self.calls.append({"config": config})
            if len(self.calls) == 1:
                clock["now"] = 1.0
            else:
                clock["now"] = 6.0
            raise _HTTPError(503)

    fake = SimpleNamespace(models=Models())
    sleeps = []
    monkeypatch.setattr(gc.time, "monotonic", lambda: clock["now"])

    def advance(seconds):
        sleeps.append(seconds)
        clock["now"] += seconds

    monkeypatch.setattr(gc.time, "sleep", advance)
    monkeypatch.setattr(gc.random, "uniform", lambda lo, _hi: lo)

    with pytest.raises(gc.GeminiDeadlineExceededError) as exc_info:
        _call_v3(
            monkeypatch,
            fake,
            timeout_s=20.0,
            deadline_monotonic=12.0,
            max_retries=2,
        )

    assert len(fake.models.calls) == 2
    assert sleeps == [1.0]
    assert exc_info.value.telemetry.retries == 1


def test_gemini3_cancellation_during_second_503_backoff_prevents_third_attempt(
    monkeypatch,
):
    fake = _FakeClient(_HTTPError(503), _HTTPError(503), _FakeResponse())
    clock = {"now": 0.0}
    state = {"cancelled": False}

    def advance(seconds):
        clock["now"] += seconds
        if len(fake.models.calls) == 2:
            state["cancelled"] = True

    monkeypatch.setattr(gc.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(gc.time, "sleep", advance)
    monkeypatch.setattr(gc.random, "uniform", lambda lo, _hi: lo)

    with pytest.raises(gc.GeminiCancelledError) as exc_info:
        _call_v3(
            monkeypatch,
            fake,
            max_retries=2,
            cancelled=lambda: state["cancelled"],
        )

    assert len(fake.models.calls) == 2
    assert exc_info.value.telemetry.retries == 1


def test_gemini3_does_not_retry_non_transient_error(monkeypatch):
    fake = _FakeClient(_HTTPError(400), _FakeResponse())

    with pytest.raises(gc.GeminiTransportError):
        _call_v3(monkeypatch, fake, max_retries=1)

    assert len(fake.models.calls) == 1


def test_gemini3_absolute_deadline_caps_request_timeout(monkeypatch):
    fake = _FakeClient(_FakeResponse())
    deadline = gc.time.monotonic() + 0.25

    _call_v3(monkeypatch, fake, deadline_monotonic=deadline, timeout_s=45.0)

    timeout_ms = fake.models.calls[0]["config"].http_options.timeout
    assert 1 <= timeout_ms <= 250


def test_gemini3_expired_deadline_and_cancellation_make_no_request(monkeypatch):
    fake = _FakeClient(_FakeResponse())
    monkeypatch.setattr(gc, "get_client", lambda: fake)

    with pytest.raises(gc.GeminiDeadlineExceededError):
        gc.generate_json_v3(
            "system", "user", _Schema, model="gemini-3.5-flash",
            thinking_level="medium", max_output_tokens=100, timeout_s=45,
            operation="flash_single", prompt_version="v1",
            deadline_monotonic=gc.time.monotonic() - 1,
        )
    with pytest.raises(gc.GeminiCancelledError):
        gc.generate_json_v3(
            "system", "user", _Schema, model="gemini-3.5-flash",
            thinking_level="medium", max_output_tokens=100, timeout_s=45,
            operation="flash_single", prompt_version="v1", deadline_monotonic=None,
            cancelled=lambda: True,
        )
    assert fake.models.calls == []


def test_gemini3_truncation_and_empty_text_are_typed_and_not_retried(monkeypatch):
    truncated = _FakeClient(_FakeResponse("partial", finish_reason="MAX_TOKENS"), _FakeResponse())
    with pytest.raises(gc.GeminiTruncatedResponseError) as trunc_info:
        _call_v3(monkeypatch, truncated, max_retries=1)
    assert len(truncated.models.calls) == 1
    assert trunc_info.value.telemetry.finish_reason == "MAX_TOKENS"

    empty = _FakeClient(_FakeResponse("   "), _FakeResponse())
    with pytest.raises(gc.GeminiEmptyResponseError) as empty_info:
        _call_v3(monkeypatch, empty, max_retries=1)
    assert len(empty.models.calls) == 1
    assert empty_info.value.telemetry.total_tokens == 23


@pytest.mark.parametrize("finish_reason", ["SAFETY", "RECITATION", "BLOCKLIST"])
def test_gemini3_blocked_finish_reasons_are_invalid_and_not_retried(monkeypatch, finish_reason):
    fake = _FakeClient(_FakeResponse('{"ok": true}', finish_reason=finish_reason), _FakeResponse())
    with pytest.raises(gc.GeminiBlockedResponseError) as exc_info:
        _call_v3(monkeypatch, fake, max_retries=1)
    assert len(fake.models.calls) == 1
    assert exc_info.value.telemetry.finish_reason == finish_reason


def test_operation_timeout_is_shared_across_transient_retry(monkeypatch):
    clock = {"now": 0.0}

    class Models:
        def __init__(self):
            self.calls = []

        def generate_content(self, model, contents, config):
            self.calls.append({"config": config})
            if len(self.calls) == 1:
                clock["now"] = 30.0
                raise _HTTPError(503)
            return _FakeResponse()

    fake = SimpleNamespace(models=Models())
    monkeypatch.setattr(gc.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(gc.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(gc.random, "uniform", lambda _lo, _hi: 1.0)
    result = _call_v3(monkeypatch, fake, timeout_s=45.0, max_retries=1)
    assert result.telemetry.retries == 1
    first, second = [call["config"].http_options.timeout for call in fake.models.calls]
    assert first == 45_000
    assert second == 15_000


def test_explicit_deadline_allows_second_per_request_timeout(monkeypatch):
    clock = {"now": 0.0}

    class Models:
        def __init__(self):
            self.calls = []

        def generate_content(self, model, contents, config):
            self.calls.append({"config": config})
            if len(self.calls) == 1:
                clock["now"] = 90.0
                raise _HTTPError(503)
            return _FakeResponse()

    fake = SimpleNamespace(models=Models())
    monkeypatch.setattr(gc.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(
        gc.time, "sleep", lambda seconds: clock.__setitem__("now", clock["now"] + seconds),
    )
    monkeypatch.setattr(gc.random, "uniform", lambda _lo, _hi: 1.0)

    result = _call_v3(
        monkeypatch,
        fake,
        timeout_s=90.0,
        deadline_monotonic=150.0,
        max_retries=1,
    )

    assert result.telemetry.retries == 1
    first, second = [call["config"].http_options.timeout for call in fake.models.calls]
    assert first == 90_000
    assert second == 59_000


def test_retry_fails_closed_without_five_useful_seconds(monkeypatch):
    clock = {"now": 0.0}

    class Models:
        def __init__(self):
            self.calls = []

        def generate_content(self, model, contents, config):
            self.calls.append({"config": config})
            clock["now"] = 39.5
            raise _HTTPError(503)

    fake = SimpleNamespace(models=Models())
    sleeps = []
    monkeypatch.setattr(gc.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(gc.time, "sleep", sleeps.append)
    monkeypatch.setattr(gc.random, "uniform", lambda _lo, _hi: 1.0)

    with pytest.raises(gc.GeminiDeadlineExceededError) as exc_info:
        _call_v3(monkeypatch, fake, timeout_s=45.0, max_retries=1)

    assert len(fake.models.calls) == 1
    assert sleeps == []
    assert exc_info.value.telemetry.error_history[0]["retryable"] is True


def test_error_history_never_retains_provider_prose_or_prompt(monkeypatch):
    errors = [
        RuntimeError("connection reset private provider prose"),
        RuntimeError("connection reset second private provider prose"),
    ]
    fake = _FakeClient(*errors)
    monkeypatch.setattr(gc.time, "sleep", lambda _: None)

    with pytest.raises(gc.GeminiTransportError) as exc_info:
        _call_v3(monkeypatch, fake, max_retries=1)

    history = exc_info.value.telemetry.as_dict()["error_history"]
    assert len(history) == 2
    assert all(set(item) == {
        "provider_error_type", "provider_status_code", "retryable",
    } for item in history)
    assert "private provider prose" not in str(history)
    assert "system" not in str(history)
    assert "user" not in str(history)


def test_generic_generate_json_uses_gemini3_high_without_sampling(monkeypatch):
    fake = _FakeClient(_FakeResponse())
    monkeypatch.setattr(gc, "get_client", lambda: fake)

    text = gc.generate_json(
        "system", "user", _Schema, temperature=0.9,
        model="gemini-3.1-pro-preview", max_output_tokens=1234,
    )

    assert text == '{"ok": true}'
    cfg = fake.models.calls[0]["config"]
    assert cfg.temperature is None and cfg.top_p is None and cfg.top_k is None
    assert _enum_value(cfg.thinking_config.thinking_level).endswith("high")
    assert cfg.thinking_config.thinking_budget is None


def test_generic_generate_json_preserves_gemini25_thinking_off_then_on(monkeypatch):
    fake = _FakeClient(RuntimeError("thinking_budget unsupported"), _FakeResponse())
    monkeypatch.setattr(gc, "get_client", lambda: fake)

    text = gc.generate_json(
        "system", "user", _Schema, temperature=0.2,
        model="gemini-2.5-flash", max_output_tokens=8192,
    )

    assert text == '{"ok": true}'
    assert len(fake.models.calls) == 2
    first, second = (call["config"] for call in fake.models.calls)
    assert first.temperature == 0.2 and first.thinking_config.thinking_budget == 0
    assert second.temperature == 0.2 and second.thinking_config is None


def test_video_and_multimodal_gemini3_paths_omit_sampling(monkeypatch):
    fake = _FakeClient(_FakeResponse(), _FakeResponse())
    monkeypatch.setattr(gc, "get_client", lambda: fake)
    monkeypatch.setattr(gc.config, "VIDEO_JUDGE_MODEL", "gemini-3.5-flash")
    media = gc.types.MediaResolution.MEDIA_RESOLUTION_LOW

    assert gc.generate_json_video(
        "system", ["video"], _Schema, media_resolution=media,
    ) == '{"ok": true}'
    monkeypatch.setattr(gc.config, "GEMINI_MODEL", "gemini-3.5-flash")
    assert gc.generate_json_mm("system", ["image"], _Schema) == '{"ok": true}'

    video_cfg, mm_cfg = [call["config"] for call in fake.models.calls]
    for cfg in (video_cfg, mm_cfg):
        assert cfg.temperature is None and cfg.top_p is None and cfg.top_k is None
        assert cfg.thinking_config.thinking_budget is None
        assert _enum_value(cfg.thinking_config.thinking_level).endswith("medium")
    assert video_cfg.media_resolution == media


@pytest.mark.parametrize("model", ["", "gemini-2.5-flash"])
def test_dedicated_gemini3_api_requires_explicit_gemini3_model(monkeypatch, model):
    fake = _FakeClient(_FakeResponse())
    monkeypatch.setattr(gc, "get_client", lambda: fake)
    with pytest.raises(ValueError):
        gc.generate_json_v3(
            "system", "user", _Schema, model=model, thinking_level="medium",
            max_output_tokens=100, timeout_s=45, deadline_monotonic=None,
            operation="op", prompt_version="v1",
        )
    assert fake.models.calls == []


@pytest.mark.parametrize("max_retries", [-1, 3, True, 1.0])
def test_dedicated_gemini3_api_rejects_invalid_retry_counts(monkeypatch, max_retries):
    fake = _FakeClient(_FakeResponse())
    monkeypatch.setattr(gc, "get_client", lambda: fake)
    with pytest.raises(ValueError):
        gc.generate_json_v3(
            "system", "user", _Schema, model="gemini-3.5-flash",
            thinking_level="medium", max_output_tokens=100, timeout_s=45,
            deadline_monotonic=None, operation="op", prompt_version="v1",
            max_retries=max_retries,
        )
    assert fake.models.calls == []


@pytest.mark.parametrize(
    "retry_status_codes",
    [set(), {400}, {True}, [503]],
)
def test_dedicated_gemini3_api_rejects_invalid_retry_status_policies(
    monkeypatch,
    retry_status_codes,
):
    fake = _FakeClient(_FakeResponse())
    with pytest.raises(ValueError, match="retry_status_codes"):
        _call_v3(
            monkeypatch,
            fake,
            retry_status_codes=retry_status_codes,
        )
    assert fake.models.calls == []
