from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
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
                 total_tokens=23):
        self.text = text
        self.candidates = [SimpleNamespace(
            finish_reason=SimpleNamespace(value=finish_reason),
        )]
        self.usage_metadata = SimpleNamespace(
            prompt_token_count=prompt_tokens,
            candidates_token_count=candidate_tokens,
            thoughts_token_count=thought_tokens,
            total_token_count=total_tokens,
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
    def __init__(self, status_code: int):
        super().__init__(f"HTTP status {status_code}")
        self.status_code = status_code


def _enum_value(value) -> str:
    return str(getattr(value, "value", value)).lower()


def _call_v3(monkeypatch, fake, **overrides):
    monkeypatch.setattr(gc, "get_client", lambda: fake)
    schema = overrides.pop("schema", _Schema)
    kwargs = {
        "model": "gemini-3.5-flash",
        "thinking_level": "medium",
        "max_output_tokens": 24_576,
        "timeout_s": 45.0,
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
    ) == (11, 7, 5, 23)
    assert telemetry.retries == 0 and telemetry.latency_ms >= 0
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


def test_gemini3_retries_one_transient_error_with_short_jitter(monkeypatch):
    fake = _FakeClient(_HTTPError(503), _FakeResponse())
    sleeps = []
    monkeypatch.setattr(gc.time, "sleep", sleeps.append)
    monkeypatch.setattr(gc.random, "uniform", lambda lo, hi: 0.2)

    result = _call_v3(monkeypatch, fake, max_retries=1)

    assert len(fake.models.calls) == 2
    assert result.telemetry.retries == 1
    assert sleeps == [0.2]
    assert all(call["config"].http_options.retry_options.attempts == 1
               for call in fake.models.calls)


def test_gemini3_cancellation_during_jitter_prevents_retry(monkeypatch):
    fake = _FakeClient(_HTTPError(503), _FakeResponse())
    state = {"cancelled": False}

    def cancel_during_sleep(_):
        state["cancelled"] = True

    monkeypatch.setattr(gc.time, "sleep", cancel_during_sleep)
    with pytest.raises(gc.GeminiCancelledError) as exc_info:
        _call_v3(
            monkeypatch, fake, max_retries=1,
            cancelled=lambda: state["cancelled"],
        )
    assert len(fake.models.calls) == 1
    assert exc_info.value.telemetry.retries == 0


def test_gemini3_never_exceeds_one_application_retry(monkeypatch):
    fake = _FakeClient(_HTTPError(503), _HTTPError(503), _FakeResponse())
    monkeypatch.setattr(gc.time, "sleep", lambda _: None)

    with pytest.raises(gc.GeminiTransportError) as exc_info:
        _call_v3(monkeypatch, fake, max_retries=1)

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
    monkeypatch.setattr(gc.random, "uniform", lambda _lo, _hi: 0.2)
    result = _call_v3(monkeypatch, fake, timeout_s=45.0, max_retries=1)
    assert result.telemetry.retries == 1
    first, second = [call["config"].http_options.timeout for call in fake.models.calls]
    assert first == 45_000
    assert second == 15_000


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


@pytest.mark.parametrize("max_retries", [-1, 2, True, 1.0])
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
