"""Focused cancellation coverage for reel-generation expansion providers."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
import time
from unittest import mock

import pytest

import backend.app.clip_engine.expand as clip_expand
import backend.app.services.llm_router as llm_router
import backend.app.services.topic_expansion as topic_expansion_module
from backend.app.clip_engine.errors import CancellationError
from backend.app.db import SCHEMA
from backend.app.services.reels import GenerationCancelledError, ReelService
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


def test_generate_reels_cancels_spellcheck_before_material_or_expansion_writes(
    monkeypatch,
) -> None:
    conn = _connection()
    conn.execute(
        "INSERT INTO materials "
        "(id, subject_tag, raw_text, source_type, knowledge_level, created_at) "
        "VALUES ('material-cancel', 'pychology', 'Topic: pychology', 'topic', "
        "'beginner', '2026-07-09T00:00:00+00:00')"
    )
    conn.execute(
        "INSERT INTO concepts "
        "(id, material_id, title, keywords_json, summary, created_at) "
        "VALUES ('concept-cancel', 'material-cancel', 'Pychology', '[]', '', "
        "'2026-07-09T00:00:00+00:00')"
    )
    started = threading.Event()
    provider_cancelled = threading.Event()
    cancel = threading.Event()

    async def blocking_expand(*_args, **_kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            provider_cancelled.set()
            raise

    monkeypatch.setattr(clip_expand.config, "GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(clip_expand, "_gemini_expand_raw_async", blocking_expand)
    ReelService._SUBJECT_CORRECTION_CACHE.clear()
    service = ReelService(
        embedding_service=None,
        youtube_service=None,
        ingestion_pipeline=mock.Mock(),
    )
    service.topic_expansion_service.expand_topic = mock.Mock(
        side_effect=AssertionError("must not start topic expansion after cancellation")
    )

    error = _run_until_cancelled(
        lambda: service.generate_reels(
            conn,
            material_id="material-cancel",
            concept_id=None,
            num_reels=1,
            creative_commons_only=False,
            should_cancel=cancel.is_set,
        ),
        started,
        cancel,
    )

    assert isinstance(error, GenerationCancelledError)
    assert provider_cancelled.wait(0.2)
    service.topic_expansion_service.expand_topic.assert_not_called()
    subject = conn.execute(
        "SELECT subject_tag FROM materials WHERE id = 'material-cancel'"
    ).fetchone()[0]
    assert subject == "pychology"
    assert conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0] == 0
    assert "pychology" not in ReelService._SUBJECT_CORRECTION_CACHE
    ReelService._SUBJECT_CORRECTION_CACHE.clear()
    conn.close()


def test_deep_expansion_cancels_active_llm_without_cache_or_merge(
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
    provider_cancelled = threading.Event()
    cancel = threading.Event()

    class BlockingClient:
        def __init__(self, **_kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            pass

        async def post(self, _url, **_kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                provider_cancelled.set()
                raise

    service = ReelService(embedding_service=None, youtube_service=None)
    service.llm_available = True
    service.topic_expansion_service.expand_topic = mock.Mock(
        return_value={
            "canonical_topic": "Psychology",
            "aliases": [],
            "subtopics": ["cognitive psychology"],
            "related_terms": [],
        }
    )
    merge = mock.Mock(side_effect=AssertionError("must not merge after cancellation"))
    monkeypatch.setattr(service, "_merge_topic_expansions", merge)
    monkeypatch.setattr(llm_router.httpx, "AsyncClient", BlockingClient)
    monkeypatch.setattr(
        llm_router,
        "_build_ollama_client",
        lambda: {"base_url": "https://ollama.invalid", "model": "test"},
    )
    later_provider = mock.Mock(return_value=None)
    monkeypatch.setattr(llm_router, "_build_gemini_module", later_provider)

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
    assert provider_cancelled.wait(0.2)
    later_provider.assert_not_called()
    merge.assert_not_called()
    assert conn.execute(
        "SELECT COUNT(*) FROM llm_cache WHERE cache_key LIKE 'topic_deep_expand:%'"
    ).fetchone()[0] == 0
    conn.close()
