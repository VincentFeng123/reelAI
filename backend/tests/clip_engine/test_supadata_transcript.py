import pytest

from backend.app.clip_engine.clipper import supadata_client
from backend.app.clip_engine.clipper.pipeline.transcribe import transcribe_supadata
from backend.app.clip_engine.errors import (
    CaptionsUnavailableError,
    ProviderRequestError,
    ProviderTransientError,
)
from backend.app.clip_engine.provider_cache import MemoryProviderCache
from backend.app.clip_engine.provider_runtime import GenerationContext

VIDEO_ID = "dQw4w9WgXcQ"
VIDEO_URL = f"https://www.youtube.com/watch?v={VIDEO_ID}"


class _Response:
    def __init__(self, status, payload, headers=None):
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}
        self.text = str(payload)

    def json(self):
        return self._payload


def _install_client(monkeypatch, responder):
    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, params=None, headers=None):
            return responder(url, params, headers)

    async def no_sleep(*args, **kwargs):
        return None

    monkeypatch.setattr(supadata_client.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(supadata_client, "sleep_with_probe", no_sleep)
    monkeypatch.setattr(supadata_client.config, "SUPADATA_API_KEY", "sd_test")


def test_native_transcript_preserves_cue_times_language_and_caches(monkeypatch) -> None:
    calls = []

    def responder(url, params, headers):
        calls.append((url, params, headers))
        return _Response(200, {
            "lang": "es",
            "content": [
                {"id": "native-a", "offset": 250, "duration": 1250, "text": " Hola ", "lang": "es"},
                {"id": "native-b", "offset": 1500, "duration": 500, "text": "mundo", "lang": "es"},
            ],
        }, {"x-billable-requests": "1"})

    _install_client(monkeypatch, responder)
    cache = MemoryProviderCache()
    context = GenerationContext("fast", cache_store=cache)
    artifact = supadata_client.fetch_transcript_artifact(
        f"yt:{VIDEO_ID}", "ES", context=context
    )
    assert calls[0][1]["mode"] == "native"
    assert "chunkSize" not in calls[0][1]
    assert artifact.returned_language == "es"
    assert artifact.segments == [
        {"cue_id": "native-a", "start": 0.25, "end": 1.5, "text": "Hola", "lang": "es"},
        {"cue_id": "native-b", "start": 1.5, "end": 2.0, "text": "mundo", "lang": "es"},
    ]
    assert context.usage()[0]["billable_requests"] == 1

    monkeypatch.setattr(supadata_client.config, "SUPADATA_API_KEY", "")
    cached = supadata_client.fetch_transcript_artifact(VIDEO_URL, "es", cache_store=cache)
    assert cached == artifact
    assert len(calls) == 1


def test_transcription_adapter_does_not_synthesize_word_times(monkeypatch) -> None:
    def responder(url, params, headers):
        return _Response(200, {
            "lang": "en",
            "content": [{"offset": 0, "duration": 2000, "text": "native cue"}],
        })

    _install_client(monkeypatch, responder)
    result = transcribe_supadata(
        VIDEO_URL,
        VIDEO_ID,
        {"language": "en", "provider_cache": MemoryProviderCache()},
    )
    assert result["words"] == []
    assert result["segments"][0]["start"] == 0.0
    assert result["segments"][0]["end"] == 2.0
    assert result["native_mode"] is True


def test_missing_native_captions_are_typed_per_video(monkeypatch) -> None:
    _install_client(monkeypatch, lambda *args: _Response(404, {"message": "not found"}))
    with pytest.raises(CaptionsUnavailableError):
        supadata_client.fetch_transcript_artifact(
            VIDEO_URL, cache_store=MemoryProviderCache()
        )


def test_malformed_nonmonotonic_transcript_is_not_cached(monkeypatch) -> None:
    _install_client(monkeypatch, lambda *args: _Response(200, {
        "lang": "en",
        "content": [
            {"offset": 1000, "duration": 1000, "text": "later"},
            {"offset": 500, "duration": 500, "text": "earlier"},
        ],
    }))
    cache = MemoryProviderCache()
    with pytest.raises(ProviderRequestError):
        supadata_client.fetch_transcript_artifact(VIDEO_URL, cache_store=cache)
    assert cache.transcript_rows == {}


def test_network_retry_ceiling_is_two_retries(monkeypatch) -> None:
    calls = 0

    def responder(*args):
        nonlocal calls
        calls += 1
        raise supadata_client.httpx.ConnectError("offline")

    _install_client(monkeypatch, responder)
    context = GenerationContext("fast", cache_store=MemoryProviderCache())
    with pytest.raises(ProviderTransientError):
        supadata_client.fetch_transcript_artifact(VIDEO_URL, context=context)
    assert calls == 3
    assert context.budget.snapshot()["used"]["transcript"] == 3
    assert len(context.usage()) == 3


def test_tombstone_blocks_transcript_before_provider_call(monkeypatch) -> None:
    called = False

    def responder(*args):
        nonlocal called
        called = True
        return _Response(200, {})

    _install_client(monkeypatch, responder)
    cache = MemoryProviderCache()
    cache.blocked_video_ids.add(VIDEO_ID)
    with pytest.raises(ProviderRequestError):
        supadata_client.fetch_transcript_artifact(VIDEO_URL, cache_store=cache)
    assert called is False
