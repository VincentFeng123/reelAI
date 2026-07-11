import asyncio
import threading
import time

import pytest
from pydantic import BaseModel

from backend.app.clip_engine.cancellation import run_cancellable
from backend.app.clip_engine.errors import CancellationError
from backend.app.clip_engine import supadata_search
from backend.app.clip_engine.clipper import gemini_client
from backend.app.clip_engine.provider_cache import MemoryProviderCache
from backend.app.services import llm_router


def _cancel_shortly(event: threading.Event, delay: float = 0.05) -> None:
    threading.Timer(delay, event.set).start()


def test_active_async_request_is_cancelled_within_three_hundred_ms():
    cancel = threading.Event()
    request_cancelled = threading.Event()

    async def blocked():
        try:
            await asyncio.Event().wait()
        finally:
            request_cancelled.set()

    _cancel_shortly(cancel)
    started = time.monotonic()
    with pytest.raises(CancellationError):
        run_cancellable(blocked, cancel.is_set)
    assert time.monotonic() - started < 0.3
    assert request_cancelled.wait(0.1)


def test_supadata_active_socket_receives_cancellation_and_does_not_retry(monkeypatch):
    cancel = threading.Event()
    request_cancelled = threading.Event()
    calls = 0

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            try:
                await asyncio.Event().wait()
            finally:
                request_cancelled.set()

    monkeypatch.setattr(supadata_search.config, "SUPADATA_API_KEY", "test-key")
    monkeypatch.setattr(supadata_search.httpx, "AsyncClient", FakeAsyncClient)
    _cancel_shortly(cancel)
    started = time.monotonic()
    with pytest.raises(CancellationError):
        supadata_search.search_one(
            "vectors", should_cancel=cancel.is_set, cache_store=MemoryProviderCache()
        )
    assert time.monotonic() - started < 0.3
    assert request_cancelled.wait(0.1)
    assert calls == 1


def test_gemini_active_request_receives_cancellation(monkeypatch):
    cancel = threading.Event()
    request_cancelled = threading.Event()
    calls = 0

    class Models:
        async def generate_content(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            try:
                await asyncio.Event().wait()
            finally:
                request_cancelled.set()

    class Client:
        class Aio:
            models = Models()

        aio = Aio()

    class Payload(BaseModel):
        value: str

    monkeypatch.setattr(gemini_client, "_create_client", lambda: Client())
    _cancel_shortly(cancel)
    started = time.monotonic()
    with pytest.raises(CancellationError):
        gemini_client.generate_json(
            "system", "user", Payload, should_cancel=cancel.is_set
        )
    assert time.monotonic() - started < 0.3
    assert request_cancelled.wait(0.1)
    assert calls == 1


def test_gemini_retry_backoff_stops_before_another_provider_call(monkeypatch):
    cancel = threading.Event()
    calls = 0

    class Models:
        async def generate_content(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            cancel.set()
            error = RuntimeError("429 resource_exhausted")
            error.code = 429
            raise error

    class Client:
        class Aio:
            models = Models()

        aio = Aio()

    class Payload(BaseModel):
        value: str

    monkeypatch.setattr(gemini_client, "_create_client", lambda: Client())
    with pytest.raises(CancellationError):
        gemini_client.generate_json(
            "system", "user", Payload, should_cancel=cancel.is_set
        )
    assert calls == 1


def test_chat_http_request_receives_cancellation(monkeypatch):
    cancel = threading.Event()
    request_cancelled = threading.Event()
    calls = 0

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            try:
                await asyncio.Event().wait()
            finally:
                request_cancelled.set()

    monkeypatch.setattr(
        llm_router,
        "_build_ollama_client",
        lambda: {"base_url": "https://ollama.test", "model": "test-model"},
    )
    monkeypatch.setattr(llm_router.httpx, "AsyncClient", FakeAsyncClient)
    _cancel_shortly(cancel)
    started = time.monotonic()
    with pytest.raises(CancellationError):
        llm_router.chat_completion(
            system="system",
            user="user",
            should_cancel=cancel.is_set,
        )
    assert time.monotonic() - started < 0.3
    assert request_cancelled.wait(0.1)
    assert calls == 1
