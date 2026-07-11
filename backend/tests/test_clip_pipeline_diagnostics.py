from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import pytest

from backend.app import main
from backend.app.clip_engine.errors import (
    ProviderRateLimitError,
    ProviderTransientError,
    TranscriptUnavailableError,
)
from backend.app.clip_engine.provider_runtime import GenerationContext
from backend.app.ingestion import pipeline as pipeline_module
from backend.app.ingestion.models import (
    IngestFeedRequest,
    IngestRequest,
    IngestSearchRequest,
    IngestTopicCutRequest,
)
from backend.app.ingestion.pipeline import IngestionPipeline, _PlatformRateLimiter
from backend.app.services.search_query_plan import SearchQueryPlan


def _pipeline() -> IngestionPipeline:
    return IngestionPipeline(
        youtube_service=None,
        embedding_service=None,
        rate_limiter=_PlatformRateLimiter(overrides={"yt": (100, 60.0)}),
    )


def _plan() -> SearchQueryPlan:
    return SearchQueryPlan(
        literal_query="Intro to Python",
        canonical_query="Python",
        trusted_signature=["Python"],
    )


def _video() -> dict:
    return {
        "id": "dQw4w9WgXcQ",
        "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "title": "Intro to Python",
        "channel": "Teacher",
        "duration": 120.0,
    }


def _transcript() -> dict:
    return {
        "source": "supadata",
        "native_mode": False,
        "artifact_key": "supadata-transcript:v2:test",
        "duration": 20.0,
        "segments": [
            {
                "cue_id": "python",
                "start": 0.0,
                "end": 10.0,
                "text": "Python functions package reusable instructions.",
            },
            {
                "cue_id": "garden",
                "start": 10.0,
                "end": 20.0,
                "text": "Garden soil needs regular watering.",
            },
        ],
    }


def _discovery() -> dict:
    return {
        "corrected": "Intro to Python",
        "videos": [_video()],
        "credits_used": 0,
        "warning": None,
        "query_plan": _plan(),
    }


def test_generation_stage_counters_are_thread_safe() -> None:
    context = GenerationContext("slow")

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda _index: context.increment_counter("topic_rejections"), range(800)))

    assert context.counters() == {
        "discovered_videos": 0,
        "usable_transcripts": 0,
        "transcript_failures": 0,
        "transcript_timeouts": 0,
        "gemini_empty_results": 0,
        "topic_rejections": 800,
        "persisted_clips": 0,
        "provider_failures": 0,
    }
    with pytest.raises(ValueError):
        context.increment_counter("topic_rejections", -1)


def test_ingest_topic_records_stage_counts_and_propagates_shared_deadline(
    monkeypatch,
) -> None:
    captured_settings: list[dict] = []

    def fake_clip(_url, *, topic, settings, should_cancel):
        del topic, should_cancel
        captured_settings.append(settings)
        return {
            "clips": [],
            "transcript": _transcript(),
            "notes": "",
        }

    monkeypatch.setattr(pipeline_module, "_discover", lambda *_args, **_kwargs: _discovery())
    monkeypatch.setattr(pipeline_module.clip_engine_run, "clip", fake_clip)
    context = GenerationContext("slow")

    reels, video_ids = _pipeline().ingest_topic(
        topic="Intro to Python",
        material_id="material",
        concept_id="concept",
        generation_context=context,
        max_videos=1,
    )

    assert reels == []
    assert video_ids == ["dQw4w9WgXcQ"]
    assert captured_settings[0]["generation_context"] is context
    assert captured_settings[0]["deadline_monotonic"] > time.monotonic()
    counters = context.counters()
    assert counters["discovered_videos"] == 1
    assert counters["usable_transcripts"] == 1
    assert counters["gemini_empty_results"] == 1


def test_ingest_topic_counts_topic_rejections_and_persisted_clips(monkeypatch) -> None:
    engine_out = {
        "clips": [
            {"start": 0.0, "end": 10.0, "cue_ids": ["python"], "informativeness": 0.9},
            {"start": 10.0, "end": 20.0, "cue_ids": ["garden"], "informativeness": 0.9},
        ],
        "transcript": _transcript(),
        "notes": "",
    }
    monkeypatch.setattr(pipeline_module, "_discover", lambda *_args, **_kwargs: _discovery())
    monkeypatch.setattr(pipeline_module, "_run_clip", lambda *_args, **_kwargs: engine_out)
    pipeline = _pipeline()
    monkeypatch.setattr(
        pipeline,
        "_persist_engine_clip",
        mock.Mock(return_value=("persisted-reel", mock.sentinel.metadata)),
    )
    context = GenerationContext("slow")

    reels, _ = pipeline.ingest_topic(
        topic="Intro to Python",
        material_id="material",
        concept_id="concept",
        generation_context=context,
        max_videos=1,
    )

    assert reels == ["persisted-reel"]
    counters = context.counters()
    assert counters["usable_transcripts"] == 1
    assert counters["topic_rejections"] == 1
    assert counters["persisted_clips"] == 1


def test_ingest_topic_distinguishes_unavailable_transcript_from_provider_failure(
    monkeypatch,
) -> None:
    monkeypatch.setattr(pipeline_module, "_discover", lambda *_args, **_kwargs: _discovery())
    pipeline = _pipeline()
    context = GenerationContext("slow")
    monkeypatch.setattr(
        pipeline,
        "_clip_and_filter",
        mock.Mock(
            side_effect=TranscriptUnavailableError(
                "No usable timestamped transcript.",
                provider="supadata",
                operation="transcript",
            )
        ),
    )

    reels, _ = pipeline.ingest_topic(
        topic="Intro to Python",
        material_id="material",
        concept_id="concept",
        generation_context=context,
        max_videos=1,
    )

    assert reels == []
    assert context.counters()["transcript_failures"] == 1
    assert context.counters()["provider_failures"] == 1

    provider_context = GenerationContext("slow")
    monkeypatch.setattr(
        pipeline,
        "_clip_and_filter",
        mock.Mock(
            side_effect=ProviderTransientError(
                "Gemini is unavailable.",
                provider="gemini",
                operation="segmentation",
            )
        ),
    )
    with pytest.raises(ProviderTransientError):
        pipeline.ingest_topic(
            topic="Intro to Python",
            material_id="material",
            concept_id="concept",
            generation_context=provider_context,
            max_videos=1,
        )
    assert provider_context.counters()["provider_failures"] == 1

    timeout_context = GenerationContext("slow")
    monkeypatch.setattr(
        pipeline,
        "_clip_and_filter",
        mock.Mock(
            side_effect=ProviderTransientError(
                "Supadata transcript retrieval timed out.",
                provider="supadata",
                operation="transcript",
                detail="generation deadline exceeded",
            )
        ),
    )
    with pytest.raises(ProviderTransientError):
        pipeline.ingest_topic(
            topic="Intro to Python",
            material_id="material",
            concept_id="concept",
            generation_context=timeout_context,
            max_videos=1,
        )
    assert timeout_context.counters()["transcript_failures"] == 1
    assert timeout_context.counters()["transcript_timeouts"] == 1
    assert timeout_context.counters()["provider_failures"] == 1


def test_exhaustion_messages_are_stage_specific_and_source_neutral() -> None:
    search = main._generation_exhaustion_message({"discovered_videos": 0})
    transcript = main._generation_exhaustion_message(
        {"discovered_videos": 2, "usable_transcripts": 0, "transcript_failures": 2}
    )
    deadline = main._generation_exhaustion_message(
        {"discovered_videos": 2, "usable_transcripts": 0, "transcript_timeouts": 2}
    )
    quality = main._generation_exhaustion_message(
        {"discovered_videos": 2, "usable_transcripts": 2, "gemini_empty_results": 2}
    )
    topic = main._generation_exhaustion_message(
        {"discovered_videos": 2, "usable_transcripts": 2, "topic_rejections": 3}
    )

    assert "discovered" in search
    assert "usable timestamped transcript" in transcript
    assert "deadline" in deadline
    assert "content quality" in quality
    assert "topic and quality" in topic
    assert all("native caption" not in message.casefold() for message in (search, transcript, deadline, quality, topic))


def test_generation_status_exposes_counters_in_backward_compatible_error_detail() -> None:
    counters = {
        "discovered_videos": 3,
        "usable_transcripts": 1,
        "transcript_failures": 2,
        "transcript_timeouts": 0,
        "gemini_empty_results": 1,
        "topic_rejections": 0,
        "persisted_clips": 0,
        "provider_failures": 2,
    }

    payload = main._generation_job_status_payload(None, {
        "id": "job-diagnostics",
        "status": "exhausted",
        "material_id": "material",
        "request_key": "request",
        "terminal_error_code": "inventory_exhausted",
        "terminal_error_message": "No valid clips were produced.",
        "terminal_error_json": main.json.dumps({"counters": counters}),
    })

    assert payload["error"]["detail"]["counters"] == counters


def test_direct_ingest_endpoints_map_provider_errors_consistently(monkeypatch) -> None:
    async def run_immediately(_request, work):
        return work(lambda: False)

    def fail(**_kwargs):
        raise ProviderRateLimitError(
            "Supadata is rate limited.",
            provider="supadata",
            operation="transcript",
            retry_after_sec=2.1,
        )

    monkeypatch.setattr(main, "_enforce_rate_limit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main, "_run_disconnect_cancellable", run_immediately)
    monkeypatch.setattr(main, "SERVERLESS_MODE", False)

    cases = [
        (
            main.ingest_url_endpoint,
            "ingest_url",
            IngestRequest(source_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ"),
        ),
        (
            main.ingest_topic_cut_endpoint,
            "ingest_topic_cut",
            IngestTopicCutRequest(source_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ"),
        ),
        (
            main.ingest_search_endpoint,
            "ingest_search",
            IngestSearchRequest(query="Intro to Python"),
        ),
        (
            main.ingest_feed_endpoint,
            "ingest_feed",
            IngestFeedRequest(feed_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ"),
        ),
    ]

    for endpoint, method_name, payload in cases:
        monkeypatch.setattr(main.ingestion_pipeline, method_name, fail)
        with pytest.raises(main.HTTPException) as captured:
            asyncio.run(endpoint(object(), payload))
        assert captured.value.status_code == 429
        assert captured.value.detail["code"] == "provider_rate_limited"
        assert captured.value.detail["operation"] == "transcript"
        assert captured.value.headers == {"Retry-After": "3"}

    unavailable = main._provider_error_to_http(
        TranscriptUnavailableError(
            "No usable timestamped transcript.",
            provider="supadata",
            operation="transcript",
        )
    )
    assert unavailable.status_code == 422
    assert unavailable.detail["code"] == "transcript_unavailable"
