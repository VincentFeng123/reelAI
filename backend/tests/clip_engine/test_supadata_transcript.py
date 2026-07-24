import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest

from backend.app.clip_engine.clipper import supadata_client
from backend.app.clip_engine.clipper.pipeline.transcribe import transcribe_supadata
from backend.app.clip_engine.errors import (
    CancellationError,
    ProviderError,
    ProviderRequestError,
    ProviderTransientError,
    TranscriptUnavailableError,
)
from backend.app.clip_engine.provider_cache import (
    TRANSCRIPT_SCHEMA_VERSION,
    MemoryProviderCache,
    TranscriptArtifact,
    transcript_artifact_key,
)
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


class _InvalidJSONResponse(_Response):
    def json(self):
        raise ValueError("invalid json")


def _install_client(monkeypatch, responder):
    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, params=None, headers=None, timeout=None):
            return responder(url, params, headers)

    async def no_sleep(*args, **kwargs):
        return None

    monkeypatch.setattr(supadata_client.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(supadata_client, "sleep_with_probe", no_sleep)
    monkeypatch.setattr(supadata_client.config, "SUPADATA_API_KEY", "sd_test")


def test_auto_transcript_preserves_cue_times_language_and_caches(monkeypatch) -> None:
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
    assert calls[0][1]["mode"] == "auto"
    assert "chunkSize" not in calls[0][1]
    assert artifact.returned_language == "es"
    assert artifact.native_mode is False
    assert artifact.segments == [
        {"cue_id": "native-a", "start": 0.25, "end": 1.5, "text": "Hola", "lang": "es"},
        {"cue_id": "native-b", "start": 1.5, "end": 2.0, "text": "mundo", "lang": "es"},
    ]
    assert context.usage()[0]["billable_requests"] == 1

    monkeypatch.setattr(supadata_client.config, "SUPADATA_API_KEY", "")
    cached = supadata_client.fetch_transcript_artifact(
        VIDEO_URL,
        "es",
        context=context,
        cache_store=cache,
    )
    assert cached == artifact
    assert len(calls) == 1
    assert context.usage()[-1]["metadata"]["cache_hit"] is True
    assert context.usage()[-1]["operation"] == "transcript"
    assert context.usage_payload()["summary"]["cache_hits"] == 1


def test_synchronous_auto_generated_transcript_is_accepted(monkeypatch) -> None:
    _install_client(monkeypatch, lambda *args: _Response(200, {
        "lang": "en",
        "content": [
            {
                "id": "generated-sync",
                "offset": 1_000,
                "duration": 2_500,
                "text": "Hosted speech recognition produced this timed cue.",
            }
        ],
    }))

    artifact = supadata_client.fetch_transcript_artifact(
        VIDEO_URL, cache_store=MemoryProviderCache()
    )

    assert artifact.native_mode is False
    assert artifact.segments == [{
        "cue_id": "generated-sync",
        "start": 1.0,
        "end": 3.5,
        "text": "Hosted speech recognition produced this timed cue.",
        "lang": "en",
    }]


def test_transcript_decodes_double_escaped_caption_entities(monkeypatch) -> None:
    _install_client(monkeypatch, lambda *args: _Response(200, {
        "lang": "en",
        "content": [{
            "id": "escaped-cue",
            "offset": 0,
            "duration": 2_000,
            "text": "In our case we&amp;#39;ll reject the null &amp;amp; continue.",
        }],
    }))

    artifact = supadata_client.fetch_transcript_artifact(
        VIDEO_URL,
        cache_store=MemoryProviderCache(),
    )

    assert artifact.segments[0]["text"] == (
        "In our case we'll reject the null & continue."
    )


def test_transcription_adapter_matches_practice_chunk_and_word_timing(monkeypatch) -> None:
    calls = []

    def responder(url, params, headers):
        calls.append((url, params))
        return _Response(200, {
            "lang": "en",
            "content": [{"offset": 0, "duration": 2000, "text": "native cue words"}],
        })

    _install_client(monkeypatch, responder)
    result = transcribe_supadata(
        VIDEO_URL,
        VIDEO_ID,
        {"language": "en", "provider_cache": MemoryProviderCache()},
    )
    assert calls[0][1]["chunkSize"] == "50"
    assert result["words"] == [
        {"word": "native", "start": 0.0, "end": pytest.approx(2 / 3), "timing_source": "interpolated"},
        {"word": "cue", "start": pytest.approx(2 / 3), "end": pytest.approx(4 / 3), "timing_source": "interpolated"},
        {"word": "words", "start": pytest.approx(4 / 3), "end": 2.0, "timing_source": "interpolated"},
    ]
    assert result["word_timing_source"] == "interpolated"
    assert result["segments"][0]["start"] == 0.0
    assert result["segments"][0]["end"] == 2.0
    assert result["native_mode"] is False
    assert result["transcript_mode"] == "auto"


def test_transcription_adapter_rejects_wrong_returned_language(monkeypatch) -> None:
    artifact = TranscriptArtifact(
        artifact_key="supadata-transcript:v4:french",
        video_id=VIDEO_ID,
        provider="supadata",
        requested_language="en",
        returned_language="fr",
        native_mode=False,
        schema_version=TRANSCRIPT_SCHEMA_VERSION,
        segments=[{
            "cue_id": "fr-0",
            "start": 0.0,
            "end": 4.0,
            "text": "Les ligatures relient plusieurs formes de lettres.",
            "lang": "fr",
        }],
        duration_sec=4.0,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    monkeypatch.setattr(
        supadata_client,
        "fetch_transcript_artifact",
        lambda *_args, **_kwargs: artifact,
    )

    with pytest.raises(TranscriptUnavailableError, match="requested language"):
        transcribe_supadata(
            VIDEO_URL,
            VIDEO_ID,
            {"language": "en", "provider_cache": MemoryProviderCache()},
        )


def test_unavailable_timestamped_transcript_is_typed_per_video(monkeypatch) -> None:
    calls = 0

    def unavailable(*_args):
        nonlocal calls
        calls += 1
        return _Response(404, {"message": "not found"})

    _install_client(monkeypatch, unavailable)
    with pytest.raises(TranscriptUnavailableError):
        supadata_client.fetch_transcript_artifact(
            VIDEO_URL, cache_store=MemoryProviderCache()
        )
    assert calls == 1


def test_documented_206_transcript_unavailable_is_not_retried(
    monkeypatch,
) -> None:
    calls = 0

    def unavailable(*_args):
        nonlocal calls
        calls += 1
        return _Response(
            206,
            {
                "error": "transcript-unavailable",
                "message": "Transcript Unavailable",
                "details": "No transcript is available for this video",
            },
            {"x-billable-requests": "1"},
        )

    _install_client(monkeypatch, unavailable)
    cache = MemoryProviderCache()
    context = GenerationContext("fast", cache_store=cache)

    with pytest.raises(TranscriptUnavailableError) as exc_info:
        supadata_client.fetch_transcript_artifact(
            VIDEO_URL,
            context=context,
        )

    assert calls == 1
    assert exc_info.value.status_code == 206
    assert exc_info.value.detail == "No transcript is available for this video"
    assert len(context.usage()) == 1
    assert context.usage()[0]["status_code"] == 206
    assert context.usage()[0]["billable_requests"] == 1
    assert context.usage()[0]["error_code"] == "transcript_unavailable"
    assert cache.get_transcript(
        video_id=VIDEO_ID,
        provider="supadata",
        requested_language="en",
        native_mode=False,
        schema_version=TRANSCRIPT_SCHEMA_VERSION,
    ) is None


@pytest.mark.parametrize(
    "first_response",
    [
        _Response(408, {"message": "timed out"}),
        _InvalidJSONResponse(200, "not-json"),
        _Response(200, ["not", "an", "object"]),
        _Response(200, {}),
        _Response(200, {"content": {"offset": 0}}),
        _Response(200, {"content": ["not-a-cue"]}),
        _Response(200, {
            "content": [{"offset": "later", "duration": 2_000, "text": "bad"}],
        }),
    ],
    ids=[
        "http-408",
        "invalid-json",
        "non-object-json",
        "missing-content",
        "non-list-content",
        "non-object-cue",
        "invalid-cue-time",
    ],
)
def test_recoverable_transcript_response_retries_inside_logical_request(
    monkeypatch,
    first_response,
) -> None:
    responses = iter(
        [
            first_response,
            _Response(
                200,
                {
                    "lang": "en",
                    "content": [
                        {
                            "id": "recovered-cue",
                            "offset": 0,
                            "duration": 2_000,
                            "text": "A complete recovered explanation.",
                        }
                    ],
                },
            ),
        ]
    )
    calls = 0

    def recover_on_second_attempt(*_args):
        nonlocal calls
        calls += 1
        return next(responses)

    _install_client(monkeypatch, recover_on_second_attempt)
    context = GenerationContext("slow", cache_store=MemoryProviderCache())

    artifact = supadata_client.fetch_transcript_artifact(
        VIDEO_URL,
        context=context,
    )

    assert artifact.segments[0]["text"] == "A complete recovered explanation."
    assert calls == 2
    assert context.budget.snapshot()["used"]["transcript"] == 1
    assert len(context.usage()) == 2
    assert context.usage()[0]["error_code"] == "provider_transient"


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404])
def test_permanent_transcript_response_is_not_retried(monkeypatch, status) -> None:
    calls = 0

    def permanent_failure(*_args):
        nonlocal calls
        calls += 1
        return _Response(status, {"message": "permanent"})

    _install_client(monkeypatch, permanent_failure)
    with pytest.raises(ProviderError):
        supadata_client.fetch_transcript_artifact(
            VIDEO_URL,
            cache_store=MemoryProviderCache(),
        )
    assert calls == 1


def test_successful_transcription_with_no_speech_is_unavailable(monkeypatch) -> None:
    calls = 0

    def no_speech(*_args):
        nonlocal calls
        calls += 1
        return _Response(200, {"lang": "en", "content": []})

    _install_client(monkeypatch, no_speech)

    with pytest.raises(TranscriptUnavailableError, match="no usable"):
        supadata_client.fetch_transcript_artifact(
            VIDEO_URL, cache_store=MemoryProviderCache()
        )
    assert calls == 1


def test_blank_transcript_cue_retries_and_recovers(monkeypatch) -> None:
    calls = 0

    def blank_then_valid(*_args):
        nonlocal calls
        calls += 1
        text = "   " if calls == 1 else "Recovered spoken explanation."
        return _Response(200, {
            "lang": "en",
            "content": [{
                "offset": 0,
                "duration": 2_000,
                "text": text,
            }],
        })

    _install_client(monkeypatch, blank_then_valid)
    artifact = supadata_client.fetch_transcript_artifact(
        VIDEO_URL,
        cache_store=MemoryProviderCache(),
    )

    assert calls == 2
    assert artifact.segments[0]["text"] == "Recovered spoken explanation."


def test_existing_native_artifact_is_preferred_before_auto_provider_call(monkeypatch) -> None:
    cache = MemoryProviderCache()
    created_at = datetime.now(timezone.utc).isoformat()
    artifact = TranscriptArtifact(
        artifact_key=transcript_artifact_key(
            video_id=VIDEO_ID,
            provider="supadata",
            requested_language="en",
            returned_language="en",
            native_mode=True,
        ),
        video_id=VIDEO_ID,
        provider="supadata",
        requested_language="en",
        returned_language="en",
        native_mode=True,
        schema_version=TRANSCRIPT_SCHEMA_VERSION,
        segments=[
            {"cue_id": "native-0", "start": 0.0, "end": 2.0, "text": "cached", "lang": "en"}
        ],
        duration_sec=2.0,
        created_at=created_at,
    )
    cache.put_transcript(artifact)
    called = False

    def responder(*args):
        nonlocal called
        called = True
        return _Response(500, {})

    _install_client(monkeypatch, responder)
    assert supadata_client.fetch_transcript_artifact(VIDEO_URL, cache_store=cache) == artifact
    assert called is False


def test_async_generated_transcript_job_is_polled_and_cached(monkeypatch) -> None:
    calls = []

    def responder(url, params, headers):
        calls.append((url, params))
        if url.endswith("/transcript"):
            return _Response(202, {"jobId": "job-1"})
        return _Response(200, {
            "status": "completed",
            "result": {
                "lang": "en",
                "content": [{"offset": 500, "duration": 1500, "text": "generated cue"}],
            },
        })

    _install_client(monkeypatch, responder)
    cache = MemoryProviderCache()
    artifact = supadata_client.fetch_transcript_artifact(
        VIDEO_URL,
        cache_store=cache,
        deadline_monotonic=time.monotonic() + 5,
    )
    assert calls[0][1]["mode"] == "auto"
    assert calls[1][0].endswith("/transcript/job-1")
    assert artifact.native_mode is False
    assert artifact.segments[0]["text"] == "generated cue"
    assert cache.get_transcript(
        video_id=VIDEO_ID,
        provider="supadata",
        requested_language="en",
        native_mode=False,
        schema_version=TRANSCRIPT_SCHEMA_VERSION,
    ) == artifact


def test_malformed_completed_poll_retries_poll_without_resubmitting_job(
    monkeypatch,
) -> None:
    calls: list[str] = []

    def responder(url, _params, _headers):
        calls.append(url)
        if url.endswith("/transcript"):
            return _Response(202, {"jobId": "job-1"})
        if calls.count(url) == 1:
            return _Response(200, {
                "status": "completed",
                "result": {"lang": "en", "content": {"bad": "shape"}},
            })
        return _Response(200, {
            "status": "completed",
            "result": {
                "lang": "en",
                "content": [{
                    "offset": 500,
                    "duration": 1_500,
                    "text": "recovered poll cue",
                }],
            },
        })

    _install_client(monkeypatch, responder)
    artifact = supadata_client.fetch_transcript_artifact(
        VIDEO_URL,
        cache_store=MemoryProviderCache(),
        deadline_monotonic=time.monotonic() + 5,
    )

    initial_calls = [url for url in calls if url.endswith("/transcript")]
    poll_calls = [url for url in calls if url.endswith("/transcript/job-1")]
    assert len(initial_calls) == 1
    assert len(poll_calls) == 2
    assert artifact.segments[0]["text"] == "recovered poll cue"


def test_expired_generation_deadline_stops_before_provider_call(monkeypatch) -> None:
    called = False

    def responder(*args):
        nonlocal called
        called = True
        return _Response(200, {})

    _install_client(monkeypatch, responder)
    with pytest.raises(ProviderTransientError, match="timed out"):
        supadata_client.fetch_transcript_artifact(
            VIDEO_URL,
            cache_store=MemoryProviderCache(),
            deadline_monotonic=time.monotonic() - 1,
        )
    assert called is False


def test_async_polling_stops_at_shared_generation_deadline(monkeypatch) -> None:
    calls = []

    def responder(url, params, headers):
        calls.append((url, params))
        return _Response(202, {"jobId": "job-never-polled"})

    _install_client(monkeypatch, responder)

    async def real_sleep(seconds, _should_cancel):
        await asyncio.sleep(seconds)

    monkeypatch.setattr(supadata_client, "sleep_with_probe", real_sleep)
    with pytest.raises(ProviderTransientError, match="timed out"):
        supadata_client.fetch_transcript_artifact(
            VIDEO_URL,
            cache_store=MemoryProviderCache(),
            deadline_monotonic=time.monotonic() + 0.01,
        )
    assert calls == [(f"{supadata_client.config.SUPADATA_BASE}/transcript", {
        "url": VIDEO_URL,
        "text": "false",
        "mode": "auto",
        "lang": "en",
    })]


def test_cancelled_transcript_retrieval_never_calls_provider(monkeypatch) -> None:
    called = False

    def responder(*args):
        nonlocal called
        called = True
        return _Response(200, {})

    _install_client(monkeypatch, responder)
    with pytest.raises(CancellationError):
        supadata_client.fetch_transcript_artifact(
            VIDEO_URL,
            cache_store=MemoryProviderCache(),
            should_cancel=lambda: True,
        )
    assert called is False


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


def test_slow_network_retry_ceiling_is_two_retries(monkeypatch) -> None:
    calls = 0

    def responder(*args):
        nonlocal calls
        calls += 1
        raise supadata_client.httpx.ConnectError("offline")

    _install_client(monkeypatch, responder)
    context = GenerationContext("slow", cache_store=MemoryProviderCache())
    with pytest.raises(ProviderTransientError):
        supadata_client.fetch_transcript_artifact(VIDEO_URL, context=context)
    assert calls == 3
    assert context.budget.snapshot()["used"]["transcript"] == 1
    assert len(context.usage()) == 3


def test_three_slow_transcripts_share_budget_and_each_retry_transient_once(
    monkeypatch,
) -> None:
    video_ids = ["dQw4w9WgXcQ", "9bZkp7q19f0", "J---aiyznGQ"]
    attempts: dict[str, int] = {}
    attempts_lock = threading.Lock()
    first_attempts = threading.Barrier(3)

    def responder(_url, params, _headers):
        video_id = str(params["url"]).rsplit("=", 1)[-1]
        with attempts_lock:
            attempts[video_id] = attempts.get(video_id, 0) + 1
            attempt = attempts[video_id]
        if attempt == 1:
            first_attempts.wait(timeout=5)
            return _Response(503, {"message": "try again"})
        return _Response(
            200,
            {
                "lang": "en",
                "content": [
                    {
                        "id": f"{video_id}-cue",
                        "offset": 0,
                        "duration": 2_000,
                        "text": f"A complete explanation from {video_id}.",
                    }
                ],
            },
        )

    _install_client(monkeypatch, responder)
    context = GenerationContext("slow", cache_store=MemoryProviderCache())
    with ThreadPoolExecutor(max_workers=3) as executor:
        artifacts = list(
            executor.map(
                lambda video_id: supadata_client.fetch_transcript_artifact(
                    f"https://www.youtube.com/watch?v={video_id}",
                    context=context,
                ),
                video_ids,
            )
        )

    assert [artifact.video_id for artifact in artifacts] == video_ids
    assert attempts == {video_id: 2 for video_id in video_ids}
    assert context.budget.snapshot()["used"]["transcript"] == 3
    assert len(context.usage()) == 6
    assert sorted(row["status_code"] for row in context.usage()) == [
        200,
        200,
        200,
        503,
        503,
        503,
    ]


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
