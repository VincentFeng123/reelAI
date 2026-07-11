"""Focused cancellation coverage for reel-generation expansion providers."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
import time

import pytest

import backend.app.services.reels as reels_module
import backend.app.services.topic_expansion as topic_expansion_module
from backend.app.clip_engine.errors import CancellationError
from backend.app.db import SCHEMA
from backend.app.services.reels import ReelService
from backend.app.services.topic_expansion import TopicExpansionService


def _connection() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def _run_until_cancelled(work, started: threading.Event, cancel: threading.Event):
    errors: list[BaseException] = []

    def run() -> None:
        try:
            work()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert started.wait(1.0), "blocking provider was not called"
    cancelled_at = time.monotonic()
    cancel.set()
    thread.join(0.6)
    assert not thread.is_alive(), "cancellation did not stop the caller"
    assert time.monotonic() - cancelled_at < 0.3
    assert errors
    return errors[0]


def test_topic_expansion_cancels_active_http_and_skips_later_calls_and_cache(
    monkeypatch,
) -> None:
    conn = _connection()
    started = threading.Event()
    provider_cancelled = threading.Event()
    cancel = threading.Event()
    calls: list[str] = []

    class BlockingClient:
        def __init__(self, **_kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            pass

        async def get(self, url, **_kwargs):
            calls.append(str(url))
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                provider_cancelled.set()
                raise

    monkeypatch.setattr(topic_expansion_module.httpx, "AsyncClient", BlockingClient)
    service = TopicExpansionService()

    error = _run_until_cancelled(
        lambda: service.expand_topic(
            conn,
            topic="psychology",
            should_cancel=cancel.is_set,
        ),
        started,
        cancel,
    )

    assert isinstance(error, CancellationError)
    assert provider_cancelled.wait(0.2)
    assert calls == [service.WIKIPEDIA_API_URL]
    assert conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0] == 0
    conn.close()


def test_deep_expansion_forwards_cancellation_to_shared_query_plan(
    monkeypatch,
) -> None:
    conn = _connection()
    conn.execute(
        "INSERT INTO materials "
        "(id, subject_tag, raw_text, source_type, knowledge_level, created_at) "
        "VALUES ('material-deep-cancel', 'psychology', 'Topic: psychology', "
        "'topic', 'beginner', '2026-07-09T00:00:00+00:00')"
    )
    started = threading.Event()
    cancel = threading.Event()

    def blocking_plan(_conn, *, literal_query, should_cancel):
        assert literal_query == "psychology"
        started.set()
        while not should_cancel():
            time.sleep(0.005)
        raise CancellationError("cancelled")

    service = ReelService(embedding_service=None, youtube_service=None)
    monkeypatch.setattr(reels_module, "build_search_query_plan", blocking_plan)

    error = _run_until_cancelled(
        lambda: service._deep_topic_expansion(
            conn,
            material_id="material-deep-cancel",
            subject_tag="psychology",
            generation_id=None,
            should_cancel=cancel.is_set,
        ),
        started,
        cancel,
    )

    assert isinstance(error, CancellationError)
    assert conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0] == 0
    conn.close()
