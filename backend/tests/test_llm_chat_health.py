import asyncio
import logging
import os
import sys
from types import ModuleType, SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import BaseModel

from backend.app import main
from backend.app.config import Settings
from backend.app.models import ChatRequest
from backend.app.services import llm_router
from backend.app.services.material_intelligence import MaterialIntelligenceService


class StructuredPayload(BaseModel):
    value: str


def test_text_llm_defaults_use_current_stable_flash() -> None:
    assert llm_router.GEMINI_DEFAULT_MODEL == "gemini-3.5-flash"
    assert Settings.model_fields["gemini_model"].default == "gemini-3.5-flash"


def _clear_text_provider_env(monkeypatch) -> None:
    for index in range(1, 20):
        suffix = "" if index == 1 else f"_{index}"
        monkeypatch.delenv(f"GEMINI_API_KEY{suffix}", raising=False)
    for name in (
        "GROQ_API_KEY",
        "CEREBRAS_API_KEY",
        "OLLAMA_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)


def test_gemini_builder_accepts_rotated_key_without_api_blackout(monkeypatch) -> None:
    _clear_text_provider_env(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY_2", "dedicated-chat-key")
    fake_genai = object()
    fake_google = ModuleType("google")
    fake_google.genai = fake_genai
    monkeypatch.setitem(sys.modules, "google", fake_google)

    assert llm_router._build_gemini_module() is fake_genai


def test_availability_ignores_credentials_for_disabled_providers(monkeypatch) -> None:
    _clear_text_provider_env(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "configured-but-disabled")
    monkeypatch.setenv("CEREBRAS_API_KEY", "configured-but-disabled")

    status = llm_router.text_llm_status()

    assert status["available"] is False
    assert status["provider"] is None
    assert status["providers"] == {
        "ollama": False,
        "gemini": False,
        "groq": False,
        "cerebras": False,
    }


def test_gemini_structured_output_uses_pydantic_json_schema(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "primary-key")
    configs = []

    class Models:
        async def generate_content(self, *args, **kwargs):
            configs.append(kwargs["config"])
            return SimpleNamespace(text='{"value":"ok"}')

    class Client:
        def __init__(self, *, api_key):
            assert api_key == "primary-key"
            self.aio = SimpleNamespace(models=Models())

    raw = llm_router._gemini_chat(
        genai_module=SimpleNamespace(Client=Client),
        model="gemini-test",
        system="system",
        user="user",
        temperature=0.1,
        json_mode=False,
        max_output_tokens=100,
        response_schema=StructuredPayload,
    )

    assert raw == '{"value":"ok"}'
    assert configs[0].response_mime_type == "application/json"
    assert configs[0].response_json_schema == StructuredPayload.model_json_schema()


def test_invalid_structured_cache_falls_back_to_valid_provider(monkeypatch) -> None:
    monkeypatch.setattr(llm_router, "_read_cache", lambda conn, key: '{"wrong":true}')
    writes = []
    monkeypatch.setattr(
        llm_router,
        "_write_cache",
        lambda conn, key, content: writes.append((key, content)),
    )
    monkeypatch.setattr(llm_router, "_build_ollama_client", lambda: None)
    monkeypatch.setattr(llm_router, "_build_gemini_module", lambda api_key=None: object())
    monkeypatch.setattr(llm_router, "_build_groq_client", lambda: None)
    monkeypatch.setattr(llm_router, "_build_cerebras_client", lambda: None)

    def gemini(**kwargs):
        assert kwargs["response_schema"] is StructuredPayload
        return '{"value":"fresh"}'

    monkeypatch.setattr(llm_router, "_gemini_chat", gemini)

    output = llm_router.chat_completion(
        conn=object(),
        cache_key="structured:test",
        system="system",
        user="user",
        response_schema=StructuredPayload,
    )

    assert StructuredPayload.model_validate_json(output).value == "fresh"
    assert writes == [("structured:test", '{"value":"fresh"}')]


def test_gemini_precedes_local_fallback_when_both_are_callable(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(llm_router, "_build_gemini_module", lambda api_key=None: object())
    monkeypatch.setattr(
        llm_router,
        "_build_ollama_client",
        lambda: {"base_url": "https://ollama.test", "model": "local"},
    )
    monkeypatch.setattr(
        llm_router,
        "_gemini_chat",
        lambda **kwargs: calls.append("gemini") or "Gemini answer",
    )
    monkeypatch.setattr(
        llm_router,
        "_ollama_chat",
        lambda **kwargs: calls.append("ollama") or "Local answer",
    )

    output = llm_router.chat_completion(system="system", user="user")

    assert output == "Gemini answer"
    assert calls == ["gemini"]


def test_gemini_request_has_a_bounded_timeout_and_closes(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "primary-key")
    monkeypatch.setattr(llm_router, "GEMINI_DEFAULT_TIMEOUT", 0.01)
    request_cancelled = asyncio.Event()
    closed = []

    class Models:
        async def generate_content(self, *args, **kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                request_cancelled.set()

    class AioClient:
        models = Models()

        async def aclose(self):
            closed.append("async")

    class Client:
        def __init__(self, *, api_key):
            self.aio = AioClient()

        def close(self):
            closed.append("sync")

    with pytest.raises(TimeoutError):
        llm_router._gemini_chat(
            genai_module=SimpleNamespace(Client=Client),
            model="gemini-test",
            system="system",
            user="user",
            temperature=0.1,
            json_mode=False,
            max_output_tokens=100,
        )

    assert request_cancelled.is_set()
    assert closed == ["async", "sync"]


def test_provider_logging_does_not_include_prompt_key_or_output(monkeypatch, caplog) -> None:
    monkeypatch.setattr(
        llm_router,
        "_build_ollama_client",
        lambda: {"base_url": "https://ollama.test", "model": "test"},
    )
    monkeypatch.setattr(llm_router, "_build_gemini_module", lambda api_key=None: None)
    monkeypatch.setattr(llm_router, "_build_groq_client", lambda: None)
    monkeypatch.setattr(llm_router, "_build_cerebras_client", lambda: None)

    def fail(**kwargs):
        raise RuntimeError("provider body included sensitive-output")

    monkeypatch.setattr(llm_router, "_ollama_chat", fail)
    with caplog.at_level(logging.INFO, logger=llm_router.__name__):
        assert llm_router.chat_completion(
            system="secret-system-prompt",
            user="secret-user-prompt",
        ) is None

    assert "secret-system-prompt" not in caplog.text
    assert "secret-user-prompt" not in caplog.text
    assert "sensitive-output" not in caplog.text
    assert "RuntimeError" in caplog.text


def test_chat_context_and_dedicated_key_fall_back_to_primary(monkeypatch) -> None:
    monkeypatch.setattr(
        llm_router,
        "gemini_or_groq_available",
        lambda **kwargs: True,
    )
    calls = []

    def complete(**kwargs):
        calls.append(kwargs)
        return None if kwargs.get("gemini_api_key_override") else "Grounded answer"

    monkeypatch.setattr(llm_router, "chat_completion", complete)
    service = MaterialIntelligenceService()

    answer = service.chat_assistant(
        message="Why does this work?",
        topic="Calculus",
        text="Limits describe local behavior.",
        history=[{"role": "user", "content": "What is a limit?"}],
        reel_summary="A derivative is a local rate of change.",
        video_title="Derivative intuition",
        video_description="A visual explanation.",
        transcript_snippet="The slope approaches a stable value.",
        gemini_api_key_override="dedicated-key",
    )

    assert answer == "Grounded answer"
    assert [call.get("gemini_api_key_override") for call in calls] == [
        "dedicated-key",
        None,
    ]
    assert all(call["gemini_model"] == service.model for call in calls)
    prompt = calls[0]["user"]
    assert "Derivative intuition" in prompt
    assert "The slope approaches a stable value." in prompt
    assert "User: What is a limit?" in prompt


def test_chat_service_does_not_return_a_canned_answer_without_provider(monkeypatch) -> None:
    monkeypatch.setattr(
        llm_router,
        "gemini_or_groq_available",
        lambda **kwargs: False,
    )
    service = MaterialIntelligenceService()

    with pytest.raises(llm_router.TextLLMUnavailableError):
        service.chat_assistant(message="Help", topic="Calculus")


def test_chat_provider_exhaustion_is_typed_503(monkeypatch) -> None:
    async def run_now(request, work):
        return work(lambda: False)

    monkeypatch.setattr(main, "_run_disconnect_cancellable", run_now)
    monkeypatch.setattr(main, "_enforce_rate_limit", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        main.material_intelligence_service,
        "chat_assistant",
        lambda **kwargs: (_ for _ in ()).throw(llm_router.TextLLMUnavailableError()),
    )

    with pytest.raises(HTTPException) as caught:
        asyncio.run(main.chat(SimpleNamespace(), ChatRequest(message="Help")))

    assert caught.value.status_code == 503
    assert caught.value.detail == {
        "code": "text_llm_unavailable",
        "message": "No text model is reachable right now.",
        "retryable": True,
    }
    assert caught.value.headers == {"Retry-After": "5"}


def test_chat_endpoint_prefers_configured_dedicated_key(monkeypatch) -> None:
    async def run_now(request, work):
        return work(lambda: False)

    captured = []
    monkeypatch.setenv("GEMINI_API_KEY_2", "dedicated-key")
    monkeypatch.setattr(main, "_run_disconnect_cancellable", run_now)
    monkeypatch.setattr(main, "_enforce_rate_limit", lambda *args, **kwargs: None)

    def answer(**kwargs):
        captured.append(kwargs)
        return "Answer"

    monkeypatch.setattr(main.material_intelligence_service, "chat_assistant", answer)

    output = asyncio.run(main.chat(SimpleNamespace(), ChatRequest(message="Help")))

    assert output == {"answer": "Answer"}
    assert captured[0]["gemini_api_key_override"] == "dedicated-key"


def test_admin_health_reports_callable_text_and_embedding_backends(monkeypatch) -> None:
    monkeypatch.setattr(main.pipeline_config, "SEGMENT_FLASH_MODEL", "gemini-3.5-flash")
    monkeypatch.setattr(main.pipeline_config, "SEGMENT_FLASH_FALLBACK_MODEL", "")
    monkeypatch.setattr(
        llm_router,
        "text_llm_status",
        lambda **kwargs: {
            "available": True,
            "provider": "gemini",
            "model": "gemini-health-test",
            "providers": {
                "ollama": False,
                "gemini": True,
                "groq": False,
                "cerebras": False,
            },
        },
    )
    monkeypatch.setattr(main, "embedding_service", SimpleNamespace(backend_name="hash"))

    health = main.admin_health()

    assert health["text_llm_available"] is True
    assert health["text_llm_provider"] == "gemini"
    assert health["chat_model"] == "gemini-health-test"
    assert health["embedding_backend"] == "hash"
    assert health["text_llm_providers"]["groq"] is False
    assert health["gemini_clip_selector_model"] == "gemini-3.5-flash"
    assert health["gemini_fallback_model"] is None


@pytest.mark.skipif(
    os.getenv("REELAI_RUN_LIVE_GEMINI") != "1"
    or not os.getenv("LIVE_GEMINI_API_KEY"),
    reason="set REELAI_RUN_LIVE_GEMINI=1 and LIVE_GEMINI_API_KEY for live smoke",
)
def test_live_gemini_chat_smoke(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", os.environ["LIVE_GEMINI_API_KEY"])
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

    answer = llm_router.chat_completion(
        system="Reply with a single short sentence.",
        user="What is two plus two?",
        max_tokens=32,
    )

    assert answer
