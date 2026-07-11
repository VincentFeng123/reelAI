import asyncio
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from backend.app.clip_engine.clipper import gemini_client, llm
from backend.app.clip_engine.errors import ModelUnavailableError, ProviderTransientError
from backend.app.clip_engine.provider_runtime import GenerationContext


class Payload(BaseModel):
    value: str


def test_each_async_request_owns_and_closes_its_transport(monkeypatch) -> None:
    created = []

    class Client:
        def __init__(self):
            self.loop = None
            self.async_closed = False
            self.sync_closed = False
            self.aio = SimpleNamespace(models=Models(self), aclose=self.aclose)

        async def aclose(self):
            self.async_closed = True

        def close(self):
            self.sync_closed = True

    class Models:
        def __init__(self, owner):
            self.owner = owner

        async def generate_content(self, *args, **kwargs):
            loop = asyncio.get_running_loop()
            if self.owner.loop is not None and self.owner.loop is not loop:
                raise RuntimeError("the handler is closed")
            self.owner.loop = loop
            return SimpleNamespace(
                text='{"value":"ok"}', model_version="primary-v1", usage_metadata=None
            )

    shared_client = Client()

    def create_client():
        client = Client()
        created.append(client)
        return client

    monkeypatch.setattr(gemini_client, "_create_client", create_client)
    monkeypatch.setattr(gemini_client, "get_client", lambda: shared_client)

    first = gemini_client.generate_json_result("system", "one", Payload, model="primary")
    second = gemini_client.generate_json_result("system", "two", Payload, model="primary")

    assert first.text == second.text == '{"value":"ok"}'
    assert len(created) == 2
    assert shared_client.loop is None
    assert all(client.async_closed and client.sync_closed for client in created)


def test_closed_transport_is_a_transient_provider_failure() -> None:
    failure = gemini_client._typed_failure(
        RuntimeError(
            "unable to perform operation on <TCPTransport closed=True>; the handler is closed"
        )
    )

    assert isinstance(failure, ProviderTransientError)
    assert failure.retryable is True


def test_gemini_uses_developer_json_schema_field(monkeypatch) -> None:
    configs = []

    class Models:
        async def generate_content(self, *args, **kwargs):
            configs.append(kwargs["config"])
            return SimpleNamespace(
                text='{"value":"ok"}',
                model_version="primary-v1",
                usage_metadata=None,
            )

    client = SimpleNamespace(aio=SimpleNamespace(models=Models()))
    monkeypatch.setattr(gemini_client, "_create_client", lambda: client)

    result = gemini_client.generate_json_result(
        "system", "user", Payload, model="primary"
    )

    assert result.text == '{"value":"ok"}'
    assert configs[0].response_schema is None
    assert configs[0].response_json_schema == Payload.model_json_schema()


def test_only_explicit_fallback_model_can_degrade(monkeypatch) -> None:
    calls = []

    def fake_generate(system, user, schema, **kwargs):
        calls.append(kwargs)
        if kwargs["model"] == "primary":
            raise ModelUnavailableError(
                "missing", provider="gemini", operation="segmentation", status_code=404
            )
        return gemini_client.GeminiResponse(
            '{"value":"ok"}', kwargs["model"], kwargs.get("quality_degraded", False)
        )

    monkeypatch.setattr(gemini_client, "generate_json_result", fake_generate)
    result = llm.llm_json_result(
        "system", "user", Payload, model="primary", fallback_model="approved-fallback"
    )
    assert result.value.value == "ok"
    assert result.model_used == "approved-fallback"
    assert result.quality_degraded is True
    assert [call["model"] for call in calls] == ["primary", "approved-fallback"]


def test_model_unavailable_without_configured_fallback_surfaces(monkeypatch) -> None:
    def unavailable(*args, **kwargs):
        raise ModelUnavailableError(
            "missing", provider="gemini", operation="segmentation", status_code=404
        )

    monkeypatch.setattr(gemini_client, "generate_json_result", unavailable)
    with pytest.raises(ModelUnavailableError):
        llm.llm_json_result("system", "user", Payload, model="primary", fallback_model="")


def test_gemini_retries_only_transient_failures_twice_and_records_usage(monkeypatch) -> None:
    calls = 0

    class Models:
        async def generate_content(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls < 3:
                error = RuntimeError("temporarily unavailable")
                error.code = 503
                raise error
            return SimpleNamespace(
                text='{"value":"ok"}',
                model_version="primary-v1",
                usage_metadata={
                    "promptTokenCount": 8,
                    "candidatesTokenCount": 3,
                    "totalTokenCount": 11,
                },
            )

    client = SimpleNamespace(aio=SimpleNamespace(models=Models()))

    async def no_sleep(*args, **kwargs):
        return None

    monkeypatch.setattr(gemini_client, "_create_client", lambda: client)
    monkeypatch.setattr(gemini_client, "sleep_with_probe", no_sleep)
    context = GenerationContext("slow")
    result = gemini_client.generate_json_result(
        "system", "user", Payload, model="primary", context=context
    )
    assert calls == 3
    assert result.model_used == "primary-v1"
    assert context.budget.snapshot()["used"]["segmentation"] == 3
    assert len(context.usage()) == 3
    assert context.usage()[-1]["total_tokens"] == 11
