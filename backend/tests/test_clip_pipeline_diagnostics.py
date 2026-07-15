from __future__ import annotations

import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import pytest

from backend import gemini_client as gemini_client_module
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


class _SemanticVector:
    def __init__(self, score: float) -> None:
        self.score = score

    def dot(self, _other) -> float:
        return self.score


class _FixedSemanticEmbedding:
    semantic_available = True

    def __init__(self, score: float) -> None:
        self.score = score
        self.inputs: list[list[str]] = []

    def embed_semantic(self, texts):
        values = [str(text) for text in texts]
        self.inputs.append(values)
        return [
            _SemanticVector(1.0 if index == 0 else self.score)
            for index, _text in enumerate(values)
        ]


def _pipeline_with_semantic(score: float) -> tuple[IngestionPipeline, _FixedSemanticEmbedding]:
    embedding = _FixedSemanticEmbedding(score)
    return (
        IngestionPipeline(
            youtube_service=None,
            embedding_service=embedding,
            rate_limiter=_PlatformRateLimiter(overrides={"yt": (100, 60.0)}),
        ),
        embedding,
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


def _quality_clip(
    *,
    cue_id: str = "python",
    start: float = 0.0,
    end: float = 10.0,
    quote: str = "Python functions package reusable instructions.",
    candidate_id: str | None = None,
    score: float = 0.9,
    **overrides,
) -> dict:
    """A complete current selector result grounded to one cue."""
    clip = {
        "start": start,
        "end": end,
        "cue_ids": [cue_id],
        "informativeness": score,
        "topic_relevance": score,
        "educational_importance": score,
        "difficulty": 0.2,
        "boundary_confidence": 0.9,
        "self_contained": True,
        "is_standalone": True,
        "selection_candidate_id": candidate_id or cue_id,
        "prerequisite_ids": [],
        "uncertainty": "low",
        "directly_teaches_topic": True,
        "substantive": True,
        "factually_grounded": True,
        "topic_evidence_quote": quote,
        "title": f"Teaching unit: {candidate_id or cue_id}",
        "learning_objective": f"Understand {candidate_id or cue_id}",
        "facet": candidate_id or cue_id,
        "reason": "A complete, transcript-grounded teaching unit.",
    }
    clip.update(overrides)
    return clip


def _one_cue_selector_result(
    evidence_quote: str,
    *,
    clip_text: str | None = None,
    **clip_overrides,
) -> dict:
    text = clip_text or evidence_quote
    return {
        "clips": [
            _quality_clip(
                cue_id="unit",
                quote=evidence_quote,
                **clip_overrides,
            )
        ],
        "transcript": {
            "source": "supadata",
            "native_mode": False,
            "artifact_key": "supadata-transcript:v2:exact-topic-corroboration",
            "duration": 10.0,
            "segments": [
                {"cue_id": "unit", "start": 0.0, "end": 10.0, "text": text},
            ],
        },
        "notes": "",
    }


def test_required_speech_bounds_prefer_verified_same_cue_edge_quotes() -> None:
    assert pipeline_module._required_speech_bounds(
        {
            "start": 0.0,
            "end": 12.0,
            "required_first_speech_sec": 2.5,
            "required_last_speech_sec": 9.5,
        },
        {"start_padding_ms": 0, "end_padding_ms": 0},
    ) == (2.5, 9.5)


def test_ingestion_grounding_accepts_apostrophe_typography_only_and_returns_source_text() -> None:
    text = "A cell’s membrane won’t let every charged molecule pass freely."

    assert pipeline_module._grounded_topic_evidence_quote(
        text,
        "cell's membrane won't let every charged molecule pass",
    ) == "cell’s membrane won’t let every charged molecule pass"
    assert pipeline_module._grounded_topic_evidence_quote(
        text,
        "cell's membrane will let every charged molecule pass",
    ) == ""


def test_generation_stage_counters_are_thread_safe() -> None:
    context = GenerationContext("slow")

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda _index: context.increment_counter("topic_rejections"), range(800)))

    expected = {
        "discovered_videos": 0,
        "usable_transcripts": 0,
        "transcript_failures": 0,
        "transcript_timeouts": 0,
        "clip_fetch_timeouts": 0,
        "gemini_empty_results": 0,
        "topic_rejections": 800,
        "persisted_clips": 0,
        "provider_failures": 0,
        "segmentation_cache_hits": 0,
        "expansion_cache_hits": 0,
    }
    assert {key: context.counters()[key] for key in expected} == expected
    with pytest.raises(ValueError):
        context.increment_counter("topic_rejections", -1)


def test_ingest_topic_records_stage_counts_and_propagates_shared_deadline(
    monkeypatch,
) -> None:
    captured_settings: list[dict] = []

    def fake_clip(_url, *, topic, settings, should_cancel):
        del topic, should_cancel
        captured_settings.append(settings)
        settings["generation_context"].increment_counter("usable_transcripts")
        return {
            "clips": [],
            "transcript": _transcript(),
            "notes": "",
            "_transcript_usage_recorded": True,
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
    assert counters["transcript_failures"] == 0
    assert counters["gemini_empty_results"] == 1


def test_generation_records_discovered_sources_before_video_timeout(monkeypatch) -> None:
    monkeypatch.setattr(pipeline_module, "_discover", lambda *_args, **_kwargs: _discovery())
    monkeypatch.setattr(
        _pipeline().__class__,
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
    retrieved: set[str] = set()

    with pytest.raises(ProviderTransientError):
        _pipeline().ingest_topic(
            topic="Intro to Python",
            material_id="material",
            concept_id="concept",
            generation_context=GenerationContext("slow"),
            retrieval_profile="deep",
            max_videos=1,
            retrieved_video_ids=retrieved,
        )

    assert retrieved == {"dQw4w9WgXcQ"}


def test_mixed_provider_failure_and_successful_empty_source_is_not_total_outage(
    monkeypatch,
) -> None:
    videos = [
        {**_video(), "id": video_id, "url": f"https://youtu.be/{video_id}"}
        for video_id in ("dQw4w9WgXcQ", "abcdefghijk", "ABCDEFGHIJK")
    ]
    monkeypatch.setattr(
        pipeline_module,
        "_discover",
        lambda *_args, **_kwargs: {**_discovery(), "videos": videos},
    )
    pipeline = _pipeline()

    def clip_and_filter(video, *_args, **_kwargs):
        if video["id"] != "dQw4w9WgXcQ":
            raise ProviderTransientError(
                "Gemini is temporarily unavailable.",
                provider="gemini",
                operation="segmentation",
                status_code=503,
            )
        return video, [], {"transcript": _transcript(), "clips": []}

    monkeypatch.setattr(pipeline, "_clip_and_filter", clip_and_filter)
    context = GenerationContext("slow", require_acoustic_boundaries=False)

    reels, video_ids = pipeline.ingest_topic(
        topic="Intro to Python",
        material_id="material",
        concept_id="concept",
        generation_context=context,
        max_videos=3,
    )

    assert reels == []
    assert video_ids == [video["id"] for video in videos]
    assert context.counters()["provider_failures"] == 2


def test_mixed_provider_failures_do_not_block_a_valid_completed_source(
    monkeypatch,
) -> None:
    videos = [
        {**_video(), "id": video_id, "url": f"https://youtu.be/{video_id}"}
        for video_id in ("dQw4w9WgXcQ", "abcdefghijk", "ABCDEFGHIJK")
    ]
    monkeypatch.setattr(
        pipeline_module,
        "_discover",
        lambda *_args, **_kwargs: {**_discovery(), "videos": videos},
    )
    pipeline = _pipeline()

    def clip_and_filter(video, *_args, **_kwargs):
        if video["id"] != "dQw4w9WgXcQ":
            raise ProviderTransientError(
                "Gemini is temporarily unavailable.",
                provider="gemini",
                operation="segmentation",
                status_code=503,
            )
        return video, [_quality_clip()], {
            "transcript": _transcript(),
            "clips": [],
        }

    monkeypatch.setattr(pipeline, "_clip_and_filter", clip_and_filter)
    monkeypatch.setattr(
        pipeline,
        "_persist_engine_clip",
        mock.Mock(return_value=("valid-reel", mock.sentinel.metadata)),
    )
    context = GenerationContext("slow", require_acoustic_boundaries=False)

    reels, _ = pipeline.ingest_topic(
        topic="Intro to Python",
        material_id="material",
        concept_id="concept",
        generation_context=context,
        max_videos=3,
    )

    assert reels == ["valid-reel"]
    assert context.counters()["provider_failures"] == 2


def test_ingest_topic_uses_literal_identity_for_segmentation(monkeypatch) -> None:
    captured_topics: list[str] = []
    discovery = {
        "corrected": "Calculus",
        "topic_terms": ["calclus", "Calculus"],
        "videos": [_video()],
        "credits_used": 0,
        "warning": None,
    }

    def fake_clip(_url, *, topic, **_kwargs):
        captured_topics.append(topic)
        return {"clips": [], "transcript": _transcript(), "notes": ""}

    monkeypatch.setattr(pipeline_module, "_discover", lambda *_args, **_kwargs: discovery)
    monkeypatch.setattr(pipeline_module, "_run_clip", fake_clip)

    _pipeline().ingest_topic(
        topic="derivative intuition",
        literal_topic="calclus",
        material_id="material",
        concept_id="concept",
        max_videos=1,
    )

    assert captured_topics == ["calclus"]


def test_ingest_topic_rejects_transcript_window_without_topic_evidence(monkeypatch) -> None:
    engine_out = {
        "clips": [
            _quality_clip(),
            _quality_clip(
                cue_id="garden",
                start=10.0,
                end=20.0,
                quote="Garden soil needs regular watering.",
                directly_teaches_topic=False,
            ),
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


def test_topic_generation_streams_independent_acoustic_passes_immediately_but_returns_stable_order(
    monkeypatch,
) -> None:
    transcript = {
        "source": "supadata",
        "native_mode": False,
        "artifact_key": "supadata-transcript:v2:acoustic",
        "duration": 30.0,
        "segments": [
            {"cue_id": "one", "start": 0.0, "end": 10.0, "text": "Python functions package reusable instructions for later calls."},
            {"cue_id": "two", "start": 10.0, "end": 20.0, "text": "Python loops repeat a block while a condition remains true."},
            {"cue_id": "three", "start": 20.0, "end": 30.0, "text": "Python dictionaries connect unique keys to stored values."},
        ],
    }
    clips = [
        _quality_clip(
            cue_id="one",
            start=0.0,
            end=10.0,
            quote="Python functions package reusable instructions for later calls.",
            score=0.9,
        ),
        _quality_clip(
            cue_id="two",
            start=10.0,
            end=20.0,
            quote="Python loops repeat a block while a condition remains true.",
            score=0.85,
        ),
        _quality_clip(
            cue_id="three",
            start=20.0,
            end=30.0,
            quote="Python dictionaries connect unique keys to stored values.",
            score=0.8,
        ),
    ]
    engine_out = {"clips": clips, "transcript": transcript, "notes": ""}

    def run_clip(*_args, **_kwargs):
        return engine_out

    monkeypatch.setattr(pipeline_module, "_discover", lambda *_args, **_kwargs: _discovery())
    monkeypatch.setattr(pipeline_module, "_run_clip", run_clip)
    persisted_clips: list[dict] = []

    def persist(*, clip, **_kwargs):
        persisted_clips.append(clip)
        candidate_id = str(clip["selection_candidate_id"]).split("::")[-1]
        return (f"verified-reel-{candidate_id}", mock.sentinel.metadata)

    pipeline = _pipeline()
    monkeypatch.setattr(pipeline, "_persist_engine_clip", persist)
    prepared = mock.sentinel.prepared_audio

    prepare_audio = mock.Mock(return_value=prepared)
    verification_started: dict[float, float] = {}
    verification_finished: dict[float, float] = {}
    verification_lock = threading.Lock()
    all_started = threading.Event()

    def verify_audio(
        _source,
        start_sec,
        end_sec,
        *,
        timeout_sec,
        **_kwargs,
    ):
        with verification_lock:
            verification_started[start_sec] = time.monotonic()
            if len(verification_started) == 3:
                all_started.set()
        assert all_started.wait(timeout=1.0)
        time.sleep({0.0: 0.15, 10.0: 0.02, 20.0: 0.10}[start_sec])
        with verification_lock:
            verification_finished[start_sec] = time.monotonic()
        start_handoff = start_sec > 0.0
        end_handoff = end_sec < 30.0
        return mock.Mock(
            verified=True,
            start_sec=start_sec,
            end_sec=end_sec,
            diagnostics={
                "threshold_dbfs": -38.0,
                "start_quiet": [start_sec, start_sec + 0.2],
                "end_quiet": [end_sec - 0.2, end_sec + 0.1],
                "start_speech_handoff_verified": start_handoff,
                "end_speech_handoff_verified": end_handoff,
                "start_two_sided_required": start_handoff,
                "end_two_sided_required": end_handoff,
                "semantic_start_limit_sec": start_sec,
                "semantic_end_limit_sec": end_sec,
            },
        )

    verify_audio_mock = mock.Mock(side_effect=verify_audio)
    monkeypatch.setattr(
        pipeline_module.clip_engine_silence,
        "prepare_audio_source",
        prepare_audio,
    )
    monkeypatch.setattr(
        pipeline_module.clip_engine_silence,
        "verify_acoustic_boundaries",
        verify_audio_mock,
    )
    emitted: list[str] = []
    emitted_at: list[float] = []

    def emit(reel: str) -> None:
        emitted.append(reel)
        emitted_at.append(time.monotonic())

    context = GenerationContext("slow", require_acoustic_boundaries=True)

    reels, _ = pipeline.ingest_topic(
        topic="Intro to Python",
        material_id="material",
        concept_id="concept",
        generation_context=context,
        max_videos=1,
        max_reels=3,
        on_reel_created=emit,
        retrieval_profile="deep",
    )

    assert reels == ["verified-reel-one", "verified-reel-two", "verified-reel-three"]
    assert emitted == ["verified-reel-two", "verified-reel-three", "verified-reel-one"]
    assert len(persisted_clips) == 3
    persisted_by_candidate = {
        str(clip["selection_candidate_id"]).split("::")[-1]: clip
        for clip in persisted_clips
    }
    boundary = persisted_by_candidate["one"]["search_context"]
    assert boundary["boundary_status"] == "verified"
    assert boundary["surface_eligible"] is True
    assert boundary["boundary_diagnostics"] == {
        "method": "energy_silence",
        "acoustic_verified": True,
        "final_range": [0.0, 10.0],
        "caption": {
            "method": "supadata_cue_timing",
            "acoustic_verified": False,
            "start_cue_id": "one",
            "end_cue_id": "one",
            "start_padding_ms": 0,
            "end_padding_ms": 0,
            "preceding_gap_ms": 0,
            "following_gap_ms": 0,
        },
        "acoustic": {
            "threshold_dbfs": -38.0,
            "start_quiet": [0.0, 0.2],
            "end_quiet": [9.8, 10.1],
            "start_speech_handoff_verified": False,
            "end_speech_handoff_verified": True,
            "start_two_sided_required": False,
            "end_two_sided_required": True,
            "semantic_start_limit_sec": 0.0,
            "semantic_end_limit_sec": 10.0,
        },
    }
    assert max(verification_started.values()) < min(verification_finished.values())
    assert emitted_at[0] >= verification_finished[10.0]
    assert emitted_at[0] < verification_finished[0.0]
    assert emitted_at[0] < verification_finished[20.0]
    prepare_audio.assert_called_once()
    assert verify_audio_mock.call_count == 3
    assert all(
        call.kwargs["prepared"] is prepared
        for call in verify_audio_mock.call_args_list
    )
    assert {
        (call.kwargs["search_start_limit_sec"], call.kwargs["search_end_limit_sec"])
        for call in verify_audio_mock.call_args_list
    } == {(0.0, 10.0), (10.0, 20.0), (20.0, 30.0)}
    assert context.counters()["stored_clips"] == 3
    assert context.counters()["deferred_clips"] == 0
    assert context.counters()["persisted_clips"] == 3


def test_many_queued_acoustic_candidates_share_source_deadline_and_cancel(
    monkeypatch,
) -> None:
    candidate_count = 40
    transcript = {
        "source": "supadata",
        "native_mode": False,
        "artifact_key": "supadata-transcript:v2:queued-acoustic",
        "duration": float(candidate_count),
        "segments": [
            {
                "cue_id": f"unit-{index}",
                "start": float(index),
                "end": float(index + 1),
                "text": f"Python teaching unit {index} explains a distinct concept.",
            }
            for index in range(candidate_count)
        ],
    }
    clips = [
        {
            "start": float(index),
            "end": float(index + 1),
            "cue_ids": [f"unit-{index}"],
            "score": 0.9 - index * 0.01,
            "selection_candidate_id": f"unit-{index}",
            "prerequisite_ids": [],
            "search_context": {"surface_eligible": True},
        }
        for index in range(candidate_count)
    ]
    engine_out = {"clips": clips, "transcript": transcript, "notes": ""}
    pipeline = _pipeline()
    monkeypatch.setattr(
        pipeline_module,
        "_discover",
        lambda *_args, **_kwargs: _discovery(),
    )
    monkeypatch.setattr(
        pipeline,
        "_clip_and_filter",
        lambda selected, *_args, **_kwargs: (selected, clips, engine_out),
    )
    monkeypatch.setattr(
        pipeline_module.clip_engine_silence,
        "prepare_audio_source",
        mock.Mock(return_value=mock.sentinel.prepared_audio),
    )
    monkeypatch.setattr(
        pipeline_module.clip_engine_silence,
        "DEEP_PHASE_TIMEOUT_SEC",
        0.05,
    )

    started: list[float] = []
    timeouts: list[float] = []
    lock = threading.Lock()
    first_wave_started = threading.Event()

    def verify_audio(_source, start_sec, end_sec, *, timeout_sec, **_kwargs):
        with lock:
            started.append(start_sec)
            timeouts.append(timeout_sec)
            if len(started) == 3:
                first_wave_started.set()
        assert first_wave_started.wait(timeout=1.0)
        time.sleep(0.08)
        return mock.Mock(
            verified=True,
            start_sec=start_sec,
            end_sec=end_sec,
            diagnostics={
                "start_quiet": [start_sec, start_sec + 0.2],
                "end_quiet": [end_sec - 0.2, end_sec + 0.1],
                "start_speech_handoff_verified": start_sec > 0.0,
                "end_speech_handoff_verified": end_sec < candidate_count,
                "start_two_sided_required": start_sec > 0.0,
                "end_two_sided_required": end_sec < candidate_count,
                "semantic_start_limit_sec": start_sec,
                "semantic_end_limit_sec": end_sec,
            },
        )

    verify = mock.Mock(side_effect=verify_audio)
    monkeypatch.setattr(
        pipeline_module.clip_engine_silence,
        "verify_acoustic_boundaries",
        verify,
    )
    monkeypatch.setattr(
        pipeline,
        "_persist_engine_clip",
        lambda *, clip, **_kwargs: (
            str(clip["selection_candidate_id"]),
            mock.sentinel.metadata,
        ),
    )

    started_at = time.monotonic()
    context = GenerationContext("slow", require_acoustic_boundaries=True)
    reels, _ = pipeline.ingest_topic(
        topic="Intro to Python",
        material_id="material",
        concept_id="concept",
        generation_context=context,
        retrieval_profile="deep",
        max_videos=1,
        max_reels=candidate_count,
    )
    elapsed = time.monotonic() - started_at

    assert len(reels) == candidate_count
    assert elapsed < 0.3
    assert verify.call_count == 3
    assert len(started) == 3
    assert all(0.0 < timeout <= 0.05 for timeout in timeouts)
    assert context.counters()["boundary_unavailable"] == 0
    assert context.counters()["permanently_rejected_clips"] == 0


def test_final_caption_clip_verifies_last_speech_without_forcing_source_outro(
    monkeypatch,
) -> None:
    transcript = {
        "source": "supadata",
        "native_mode": False,
        "artifact_key": "supadata-transcript:v2:media-tail",
        "duration": 10.0,
        "segments": [
            {
                "cue_id": "tail",
                "start": 0.0,
                "end": 10.0,
                "text": "Python functions package reusable instructions for later calls.",
            },
        ],
    }
    engine_out = {
        "clips": [
            _quality_clip(
                cue_id="tail",
                start=0.0,
                end=10.0,
                quote="Python functions package reusable instructions for later calls.",
            )
        ],
        "transcript": transcript,
        "notes": "",
    }
    prepared = pipeline_module.clip_engine_silence.AudioPreparationResult(
        "ready",
        source=pipeline_module.clip_engine_silence.PreparedAudioSource(
            url="https://media.example/audio.m4a",
            format_id="140",
            duration_sec=12.0,
        ),
    )
    verify = mock.Mock(
        return_value=mock.Mock(
            verified=True,
            start_sec=0.0,
            end_sec=10.1,
            diagnostics={
                "threshold_dbfs": -38.0,
                "start_quiet": [0.0, 0.2],
                "end_quiet": [9.8, 10.2],
                "end_speech_handoff_verified": True,
                "end_two_sided_required": False,
                "semantic_start_limit_sec": 0.0,
                "semantic_end_limit_sec": 12.0,
            },
        )
    )
    monkeypatch.setattr(
        pipeline_module,
        "_discover",
        lambda *_args, **_kwargs: _discovery(),
    )
    monkeypatch.setattr(
        pipeline_module,
        "_run_clip",
        lambda *_args, **_kwargs: engine_out,
    )
    monkeypatch.setattr(
        pipeline_module.clip_engine_silence,
        "prepare_audio_source",
        mock.Mock(return_value=prepared),
    )
    monkeypatch.setattr(
        pipeline_module.clip_engine_silence,
        "verify_acoustic_boundaries",
        verify,
    )
    persisted: list[dict] = []
    pipeline = _pipeline()

    def persist(*, clip, **_kwargs):
        persisted.append(clip)
        return ("verified-tail", mock.sentinel.metadata)

    monkeypatch.setattr(pipeline, "_persist_engine_clip", persist)

    reels, _ = pipeline.ingest_topic(
        topic="Intro to Python",
        material_id="material",
        concept_id="concept",
        generation_context=GenerationContext(
            "slow", require_acoustic_boundaries=True
        ),
        retrieval_profile="deep",
        max_videos=1,
        max_reels=1,
    )

    assert reels == ["verified-tail"]
    assert persisted[0]["end"] == 10.1
    assert verify.call_args.args[1:] == (0.0, 10.0)
    assert verify.call_args.kwargs["search_end_limit_sec"] == 12.0
    assert verify.call_args.kwargs["require_speech_handoff"] is False
    assert verify.call_args.kwargs["require_end_speech_handoff"] is True
    assert verify.call_args.kwargs["require_end_two_sided"] is False
    assert persisted[0]["search_context"]["boundary_status"] == "verified"
    assert persisted[0]["search_context"]["boundary_diagnostics"]["acoustic"][
        "threshold_dbfs"
    ] == -38.0


def test_acoustic_search_anchors_to_required_speech_not_selector_padding(
    monkeypatch,
) -> None:
    transcript = {
        "source": "supadata",
        "native_mode": False,
        "artifact_key": "supadata-transcript:v2:padded-speech",
        "duration": 30.0,
        "segments": [
            {
                "cue_id": "prior",
                "start": 0.0,
                "end": 10.0,
                "text": "The prior section explains a different language feature.",
            },
            {
                "cue_id": "required-speech",
                "start": 10.0,
                "end": 20.0,
                "text": "A Python closure retains variables from its enclosing scope.",
            },
            {
                "cue_id": "following",
                "start": 20.0,
                "end": 30.0,
                "text": "The following section moves to a separate topic.",
            },
        ],
    }
    engine_out = {
        "clips": [
            _quality_clip(
                cue_id="required-speech",
                start=9.7,
                end=20.3,
                quote="A Python closure retains variables from its enclosing scope.",
            )
        ],
        "transcript": transcript,
        "notes": "",
    }
    verify = mock.Mock(
        return_value=mock.Mock(
            verified=True,
            start_sec=9.9,
            end_sec=20.0,
            diagnostics={
                "threshold_dbfs": -38.0,
                "start_speech_handoff_verified": True,
                "end_speech_handoff_verified": False,
                "start_two_sided_required": True,
                "end_two_sided_required": True,
                "semantic_start_limit_sec": 10.0,
                "semantic_end_limit_sec": 20.0,
                "observation_start_limit_sec": 9.0,
                "observation_end_limit_sec": 21.0,
                "handoff_timestamp_tolerance_sec": 0.05,
                "start_quiet": [9.8, 10.1],
                "end_quiet": [19.8, 20.1],
            },
        )
    )
    monkeypatch.setattr(
        pipeline_module,
        "_discover",
        lambda *_args, **_kwargs: _discovery(),
    )
    monkeypatch.setattr(
        pipeline_module,
        "_run_clip",
        lambda *_args, **_kwargs: engine_out,
    )
    monkeypatch.setattr(
        pipeline_module.clip_engine_silence,
        "prepare_audio_source",
        mock.Mock(return_value=mock.sentinel.prepared_audio),
    )
    monkeypatch.setattr(
        pipeline_module.clip_engine_silence,
        "verify_acoustic_boundaries",
        verify,
    )
    persisted: list[dict] = []
    pipeline = _pipeline()
    monkeypatch.setattr(
        pipeline,
        "_persist_engine_clip",
        lambda *, clip, **_kwargs: (
            persisted.append(clip) or "verified-padded-speech",
            mock.sentinel.metadata,
        ),
    )

    reels, _ = pipeline.ingest_topic(
        topic="Intro to Python",
        material_id="material",
        concept_id="concept",
        generation_context=GenerationContext(
            "slow", require_acoustic_boundaries=True
        ),
        retrieval_profile="deep",
        max_videos=1,
        max_reels=1,
    )

    assert reels == ["verified-padded-speech"]
    assert verify.call_args.args[1:] == (10.0, 20.0)
    assert verify.call_args.kwargs["require_speech_handoff"] is False
    assert verify.call_args.kwargs["require_start_speech_handoff"] is True
    assert verify.call_args.kwargs["require_end_speech_handoff"] is False
    assert verify.call_args.kwargs["require_start_two_sided"] is True
    assert verify.call_args.kwargs["require_end_two_sided"] is True
    assert persisted[0]["start"] == 9.9
    assert persisted[0]["end"] == 20.0


def test_acoustic_result_crossing_unselected_speech_falls_back_to_transcript(
    monkeypatch,
) -> None:
    transcript = _transcript()
    engine_out = {
        "clips": [_quality_clip()],
        "transcript": transcript,
        "notes": "",
    }
    verify = mock.Mock(
        return_value=mock.Mock(
            verified=True,
            start_sec=0.0,
            end_sec=12.0,
            diagnostics={"threshold_dbfs": -38.0},
        )
    )
    monkeypatch.setattr(
        pipeline_module, "_discover", lambda *_args, **_kwargs: _discovery()
    )
    monkeypatch.setattr(
        pipeline_module, "_run_clip", lambda *_args, **_kwargs: engine_out
    )
    monkeypatch.setattr(
        pipeline_module.clip_engine_silence,
        "prepare_audio_source",
        mock.Mock(return_value=mock.sentinel.prepared_audio),
    )
    monkeypatch.setattr(
        pipeline_module.clip_engine_silence,
        "verify_acoustic_boundaries",
        verify,
    )
    persisted: list[dict] = []
    pipeline = _pipeline()
    monkeypatch.setattr(
        pipeline,
        "_persist_engine_clip",
        lambda *, clip, **_kwargs: (
            persisted.append(clip) or "stored-crossing",
            mock.sentinel.metadata,
        ),
    )

    context = GenerationContext("slow", require_acoustic_boundaries=True)
    reels, _ = pipeline.ingest_topic(
        topic="Intro to Python",
        material_id="material",
        concept_id="concept",
        generation_context=context,
        retrieval_profile="deep",
        max_videos=1,
        max_reels=1,
    )

    assert reels == ["stored-crossing"]
    assert verify.call_args.kwargs["search_end_limit_sec"] == 10.0
    assert len(persisted) == 1
    assert (persisted[0]["start"], persisted[0]["end"]) == (0.0, 10.0)
    assert persisted[0]["search_context"]["boundary_status"] == "context_aligned"
    assert context.counters()["permanently_rejected_clips"] == 0


def test_acoustic_result_crossing_prior_unselected_speech_falls_back_to_transcript(
    monkeypatch,
) -> None:
    transcript = {
        "source": "supadata",
        "native_mode": False,
        "artifact_key": "supadata-transcript:v2:start-crossing",
        "duration": 20.0,
        "segments": [
            {
                "cue_id": "prior",
                "start": 0.0,
                "end": 10.0,
                "text": "A sponsor and unrelated introduction appear first.",
            },
            {
                "cue_id": "python",
                "start": 10.0,
                "end": 20.0,
                "text": "Python functions package reusable instructions.",
            },
        ],
    }
    engine_out = {
        "clips": [_quality_clip(cue_id="python", start=10.0, end=20.0)],
        "transcript": transcript,
        "notes": "",
    }
    verify = mock.Mock(
        return_value=mock.Mock(
            verified=True,
            start_sec=8.0,
            end_sec=20.0,
            diagnostics={"threshold_dbfs": -38.0},
        )
    )
    monkeypatch.setattr(
        pipeline_module, "_discover", lambda *_args, **_kwargs: _discovery()
    )
    monkeypatch.setattr(
        pipeline_module, "_run_clip", lambda *_args, **_kwargs: engine_out
    )
    monkeypatch.setattr(
        pipeline_module.clip_engine_silence,
        "prepare_audio_source",
        mock.Mock(return_value=mock.sentinel.prepared_audio),
    )
    monkeypatch.setattr(
        pipeline_module.clip_engine_silence,
        "verify_acoustic_boundaries",
        verify,
    )
    persisted: list[dict] = []
    pipeline = _pipeline()
    monkeypatch.setattr(
        pipeline,
        "_persist_engine_clip",
        lambda *, clip, **_kwargs: (
            persisted.append(clip) or "stored-start-crossing",
            mock.sentinel.metadata,
        ),
    )

    context = GenerationContext("slow", require_acoustic_boundaries=True)
    reels, _ = pipeline.ingest_topic(
        topic="Intro to Python",
        material_id="material",
        concept_id="concept",
        generation_context=context,
        retrieval_profile="deep",
        max_videos=1,
        max_reels=1,
    )

    assert reels == ["stored-start-crossing"]
    assert verify.call_args.kwargs["search_start_limit_sec"] == 10.0
    assert len(persisted) == 1
    assert (persisted[0]["start"], persisted[0]["end"]) == (10.0, 20.0)
    assert persisted[0]["search_context"]["boundary_status"] == "context_aligned"
    assert context.counters()["permanently_rejected_clips"] == 0


@pytest.mark.parametrize(
    ("text", "required_bound"),
    [
        (
            "Welcome to the channel. Python functions package reusable instructions.",
            {"required_first_speech_sec": 4.0},
        ),
        (
            "Python functions package reusable instructions. Thanks for watching.",
            {"required_last_speech_sec": 10.0},
        ),
    ],
)
def test_partial_cue_edge_filler_keeps_full_cue_without_audio_verification(
    monkeypatch, text: str, required_bound: dict,
) -> None:
    transcript = {
        "source": "supadata",
        "native_mode": False,
        "artifact_key": "supadata-transcript:v2:partial-cue-filler",
        "duration": 14.0,
        "segments": [
            {"cue_id": "unit", "start": 0.0, "end": 14.0, "text": text},
        ],
    }
    engine_out = {
        "clips": [
            _quality_clip(
                cue_id="unit",
                start=0.0,
                end=14.0,
                quote="Python functions package reusable instructions.",
                **required_bound,
            )
        ],
        "transcript": transcript,
        "notes": "",
    }
    verify = mock.Mock()
    monkeypatch.setattr(
        pipeline_module, "_discover", lambda *_args, **_kwargs: _discovery()
    )
    monkeypatch.setattr(
        pipeline_module, "_run_clip", lambda *_args, **_kwargs: engine_out
    )
    monkeypatch.setattr(
        pipeline_module.clip_engine_silence,
        "prepare_audio_source",
        mock.Mock(return_value=mock.sentinel.prepared_audio),
    )
    monkeypatch.setattr(
        pipeline_module.clip_engine_silence,
        "verify_acoustic_boundaries",
        verify,
    )
    persisted: list[dict] = []
    pipeline = _pipeline()
    monkeypatch.setattr(
        pipeline,
        "_persist_engine_clip",
        lambda *, clip, **_kwargs: (
            persisted.append(clip) or "stored-partial-cue",
            mock.sentinel.metadata,
        ),
    )

    context = GenerationContext("slow", require_acoustic_boundaries=False)
    reels, _ = pipeline.ingest_topic(
        topic="Python functions",
        material_id="material",
        concept_id="concept",
        generation_context=context,
        retrieval_profile="deep",
        max_videos=1,
        max_reels=1,
    )

    assert reels == ["stored-partial-cue"]
    verify.assert_not_called()
    assert len(persisted) == 1
    assert (persisted[0]["start"], persisted[0]["end"]) == (0.0, 14.0)
    assert persisted[0]["search_context"]["boundary_status"] == "context_aligned"
    assert context.counters()["permanently_rejected_clips"] == 0


def test_native_json3_quotes_authorize_only_the_exact_partial_cue_corridor() -> None:
    text = (
        "Welcome everyone. Python functions package reusable instructions. "
        "Thanks."
    )
    transcript = {
        "source": "supadata",
        "native_mode": False,
        "artifact_key": "supadata-transcript:v2:native-edge-projection",
        "duration": 10.0,
        "segments": [
            {"cue_id": "unit", "start": 0.0, "end": 10.0, "text": text},
        ],
    }
    clip = _quality_clip(
        cue_id="unit",
        start=0.0,
        end=10.0,
        edge_projection={
            "start": {"cue_id": "unit", "quote": "Python functions package"},
            "end": {"cue_id": "unit", "quote": "reusable instructions"},
        },
    )
    caption = pipeline_module._supadata_boundary_diagnostics(transcript, clip)
    assert caption is not None
    words = tuple(
        pipeline_module.clip_engine_silence.lexical_timing.LexicalWord(word, onset)
        for word, onset in zip(
            (
                "welcome",
                "everyone",
                "python",
                "functions",
                "package",
                "reusable",
                "instructions",
                "thanks",
            ),
            (0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0),
            strict=True,
        )
    )
    prepared = pipeline_module.clip_engine_silence.AudioPreparationResult(
        "ready",
        source=pipeline_module.clip_engine_silence.PreparedAudioSource(
            url="https://media.example/audio.m4a",
            lexical_words=words,
        ),
    )

    bounds, projection, error = pipeline_module._projected_speech_bounds(
        transcript,
        clip,
        caption,
        prepared,
    )
    corridor = pipeline_module._selected_speech_corridor(
        transcript,
        clip,
        caption,
        source_end_sec=10.0,
        required_speech_bounds=bounds,
        projection_diagnostics=projection,
    )
    cues = pipeline_module._selected_caption_cues(
        transcript,
        clip,
        boundary_bounds=(1.9, 6.8),
    )

    assert error is None
    assert bounds == (2.0, 6.002)
    assert projection["lexical_projection_verified"] is True
    assert projection["lexical_boundary_verified"] is True
    assert corridor == (1.0, 7.0, None)
    assert cues == [
        {
            "cue_id": "unit",
            "start": 1.9,
            "end": 6.8,
            "text": "Python functions package reusable instructions",
            "lang": "",
        }
    ]


def test_native_json3_spans_anchor_ordinary_internal_edges() -> None:
    transcript = {
        "source": "supadata",
        "native_mode": False,
        "artifact_key": "supadata-transcript:v2:ordinary-lexical-edges",
        "duration": 540.0,
        "segments": [
            {
                "cue_id": "prior",
                "start": 500.0,
                "end": 504.02,
                "text": "Earlier context ends now",
            },
            {
                "cue_id": "selected",
                "start": 504.02,
                "end": 530.43,
                "text": (
                    "The Calvin Cycle is sometimes called a light independent reaction "
                    "because it turns carbon dioxide into something useful for the plant"
                ),
            },
            {
                "cue_id": "next",
                "start": 530.43,
                "end": 540.0,
                "text": "Next topic begins here",
            },
        ],
    }
    clip = _quality_clip(
        cue_id="selected",
        start=504.02,
        end=530.43,
    )
    clip.update(
        start_quote="The Calvin Cycle is sometimes called",
        end_quote="useful for the plant",
    )
    caption = pipeline_module._supadata_boundary_diagnostics(transcript, clip)
    assert caption is not None
    lexical_words = tuple(
        pipeline_module.clip_engine_silence.lexical_timing.LexicalWord(word, onset)
        for word, onset in (
            ("now", 503.68),
            ("the", 504.08),
            ("calvin", 504.2),
            ("cycle", 504.4),
            ("is", 504.6),
            ("sometimes", 504.8),
            ("called", 505.0),
            ("useful", 529.8),
            ("for", 529.95),
            ("the", 530.05),
            ("plant", 530.16),
            ("next", 530.399),
        )
    )
    prepared = pipeline_module.clip_engine_silence.AudioPreparationResult(
        "ready",
        source=pipeline_module.clip_engine_silence.PreparedAudioSource(
            url="https://media.example/audio.m4a",
            lexical_words=lexical_words,
        ),
    )

    bounds, lexical, error = pipeline_module._projected_speech_bounds(
        transcript,
        clip,
        caption,
        prepared,
    )
    corridor = pipeline_module._selected_speech_corridor(
        transcript,
        clip,
        caption,
        source_end_sec=540.0,
        required_speech_bounds=bounds,
        projection_diagnostics=lexical,
    )
    plan = pipeline_module._acoustic_boundary_plan(
        transcript,
        clip,
        lexical,
        speech_bounds=bounds,
        search_limits=corridor[:2],
    )

    assert error is None
    assert bounds == pytest.approx((504.08, 530.16))
    assert lexical["lexical_boundary_verified"] is True
    assert lexical["lexical_projection_verified"] is False
    assert lexical["start"]["excluded_neighbor_onset_sec"] == 503.68
    assert lexical["end"]["excluded_neighbor_onset_sec"] == 530.399
    assert corridor == (503.68, 530.399, None)
    assert plan == pytest.approx((504.08, 530.16, False, False, True, True))


def test_ordinary_lexical_edges_survive_real_silence_verification_and_persist(
    monkeypatch,
) -> None:
    transcript = {
        "source": "supadata",
        "native_mode": False,
        "artifact_key": "supadata-transcript:v2:ordinary-lexical-persistence",
        "duration": 30.0,
        "segments": [
            {
                "cue_id": "prior",
                "start": 0.0,
                "end": 10.0,
                "text": "Earlier context ends now",
            },
            {
                "cue_id": "python",
                "start": 10.0,
                "end": 20.0,
                "text": (
                    "Python functions package reusable instructions for later calls"
                ),
            },
            {
                "cue_id": "next",
                "start": 20.0,
                "end": 30.0,
                "text": "Next topic begins here",
            },
        ],
    }
    engine_out = {
        "clips": [
            _quality_clip(
                cue_id="python",
                start=10.0,
                end=20.0,
                quote=(
                    "Python functions package reusable instructions for later calls"
                ),
                start_quote="Python functions package reusable",
                end_quote="instructions for later calls",
            )
        ],
        "transcript": transcript,
        "notes": "",
    }
    lexical_words = tuple(
        pipeline_module.clip_engine_silence.lexical_timing.LexicalWord(word, onset)
        for word, onset in (
            ("now", 9.7),
            ("python", 10.1),
            ("functions", 11.0),
            ("package", 12.0),
            ("reusable", 13.0),
            ("instructions", 14.0),
            ("for", 15.0),
            ("later", 16.0),
            ("calls", 17.0),
            ("next", 20.2),
        )
    )
    prepared = pipeline_module.clip_engine_silence.AudioPreparationResult(
        "ready",
        source=pipeline_module.clip_engine_silence.PreparedAudioSource(
            url="https://media.example/audio.m4a",
            format_id="140",
            duration_sec=30.0,
            lexical_words=lexical_words,
        ),
    )

    from backend.tests.clip_engine.test_silence import _write_wav

    def fake_decode(
        _source,
        *,
        window_start_sec,
        window_duration_sec,
        output_path,
        **_kwargs,
    ):
        quiet_start = (
            9.8 if output_path.name == "start-0.wav" else 17.3
        )
        before = quiet_start - window_start_sec
        quiet_duration = 0.3
        _write_wav(
            output_path,
            [
                (before, 12000),
                (quiet_duration, 0),
                (window_duration_sec - before - quiet_duration, 12000),
            ],
        )

    monkeypatch.setattr(
        pipeline_module, "_discover", lambda *_args, **_kwargs: _discovery()
    )
    monkeypatch.setattr(
        pipeline_module, "_run_clip", lambda *_args, **_kwargs: engine_out
    )
    monkeypatch.setattr(
        pipeline_module.clip_engine_silence,
        "prepare_audio_source",
        mock.Mock(return_value=prepared),
    )
    monkeypatch.setattr(
        pipeline_module.clip_engine_silence,
        "_decode_window",
        fake_decode,
    )
    persisted: list[dict] = []
    pipeline = _pipeline()
    monkeypatch.setattr(
        pipeline,
        "_persist_engine_clip",
        lambda *, clip, **_kwargs: (
            persisted.append(clip) or "stored-ordinary-lexical",
            mock.sentinel.metadata,
        ),
    )

    reels, _ = pipeline.ingest_topic(
        topic="Python functions",
        material_id="material",
        concept_id="concept",
        generation_context=GenerationContext(
            "slow", require_acoustic_boundaries=True
        ),
        retrieval_profile="deep",
        max_videos=1,
        max_reels=1,
    )

    assert reels == ["stored-ordinary-lexical"]
    assert len(persisted) == 1
    assert persisted[0]["start"] == pytest.approx(10.0)
    assert persisted[0]["end"] == pytest.approx(17.4)
    diagnostics = persisted[0]["search_context"]["boundary_diagnostics"][
        "acoustic"
    ]
    assert diagnostics["start_speech_handoff_verified"] is False
    assert diagnostics["end_speech_handoff_verified"] is False
    assert diagnostics["start_two_sided_required"] is True
    assert diagnostics["end_two_sided_required"] is True


def test_production_boundary_path_mixes_projected_start_with_last_speech_end(
    monkeypatch,
) -> None:
    text = (
        "Opening hook. Python functions package reusable instructions and preserve "
        "state for later calls."
    )
    transcript = {
        "source": "supadata",
        "native_mode": False,
        "artifact_key": "supadata-transcript:v2:mixed-edge-modes",
        "duration": 30.0,
        "segments": [
            {"cue_id": "unit", "start": 10.0, "end": 30.0, "text": text},
        ],
    }
    engine_out = {
        "clips": [
            _quality_clip(
                cue_id="unit",
                start=10.0,
                end=30.0,
                quote=(
                    "Python functions package reusable instructions and preserve state"
                ),
                edge_projection={
                    "start": {"cue_id": "unit", "quote": "Python functions package"}
                },
            )
        ],
        "transcript": transcript,
        "notes": "",
    }
    lexical_words = tuple(
        pipeline_module.clip_engine_silence.lexical_timing.LexicalWord(word, onset)
        for word, onset in zip(
            (
                "opening",
                "hook",
                "python",
                "functions",
                "package",
                "reusable",
                "instructions",
                "and",
                "preserve",
                "state",
                "for",
                "later",
                "calls",
            ),
            (10.5, 11.0, 12.0, 13.0, 14.0, 15.0, 17.0, 19.0, 21.0, 23.0, 25.0, 27.0, 29.0),
            strict=True,
        )
    )
    prepared = pipeline_module.clip_engine_silence.AudioPreparationResult(
        "ready",
        source=pipeline_module.clip_engine_silence.PreparedAudioSource(
            url="https://media.example/audio.m4a",
            format_id="140",
            duration_sec=40.0,
            lexical_words=lexical_words,
        ),
    )

    def fake_decode(
        _source,
        *,
        window_start_sec,
        window_duration_sec,
        output_path,
        **_kwargs,
    ):
        spans = [(window_duration_sec, 12000)]
        if output_path.name == "start-0.wav":
            quiet_start = 11.8 - window_start_sec
            spans = [
                (quiet_start, 12000),
                (0.3, 0),
                (window_duration_sec - quiet_start - 0.3, 12000),
            ]
        elif output_path.name == "end-0.wav":
            quiet_start = 29.8 - window_start_sec
            spans = [
                (quiet_start, 12000),
                (0.3, 0),
                (window_duration_sec - quiet_start - 0.3, 12000),
            ]
        _write_wav(output_path, spans)

    # Keep this regression on the real production verifier while replacing only
    # network/media decoding with deterministic PCM fixtures.
    from backend.tests.clip_engine.test_silence import _write_wav

    monkeypatch.setattr(
        pipeline_module, "_discover", lambda *_args, **_kwargs: _discovery()
    )
    monkeypatch.setattr(
        pipeline_module, "_run_clip", lambda *_args, **_kwargs: engine_out
    )
    monkeypatch.setattr(
        pipeline_module.clip_engine_silence,
        "prepare_audio_source",
        mock.Mock(return_value=prepared),
    )
    monkeypatch.setattr(
        pipeline_module.clip_engine_silence,
        "_decode_window",
        fake_decode,
    )
    real_verify = pipeline_module.clip_engine_silence.verify_acoustic_boundaries
    verify = mock.Mock(wraps=real_verify)
    monkeypatch.setattr(
        pipeline_module.clip_engine_silence,
        "verify_acoustic_boundaries",
        verify,
    )
    persisted: list[dict] = []
    pipeline = _pipeline()
    monkeypatch.setattr(
        pipeline,
        "_persist_engine_clip",
        lambda *, clip, **_kwargs: (
            persisted.append(clip) or "stored-mixed-edge",
            mock.sentinel.metadata,
        ),
    )

    reels, _ = pipeline.ingest_topic(
        topic="Python functions",
        material_id="material",
        concept_id="concept",
        generation_context=GenerationContext(
            "slow", require_acoustic_boundaries=True
        ),
        retrieval_profile="deep",
        max_videos=1,
        max_reels=1,
    )

    assert reels == ["stored-mixed-edge"]
    assert persisted[0]["start"] == 12.0
    assert persisted[0]["end"] == 30.0
    assert verify.call_args.kwargs["require_start_speech_handoff"] is False
    assert verify.call_args.kwargs["require_end_speech_handoff"] is True
    assert verify.call_args.kwargs["require_start_two_sided"] is True
    assert verify.call_args.kwargs["require_end_two_sided"] is False
    acoustic = persisted[0]["search_context"]["boundary_diagnostics"]["acoustic"]
    assert acoustic["start_windows"] == [[10.0, 15.0]]
    assert acoustic["end_windows"] == [[27.0, 33.0]]


def test_production_transcript_path_preserves_same_cue_filler_projection(
    monkeypatch,
) -> None:
    text = (
        "Welcome back. Photosynthesis converts light energy into chemical energy. "
        "Thanks for watching."
    )
    transcript = {
        "source": "supadata",
        "native_mode": False,
        "artifact_key": "supadata-transcript:v2:production-caption-projection",
        "duration": 12.0,
        "segments": [
            {"cue_id": "unit", "start": 0.0, "end": 12.0, "text": text},
        ],
    }
    engine_out = {
        "clips": [
            _quality_clip(
                cue_id="unit",
                start=0.0,
                end=12.0,
                quote="Photosynthesis converts light energy into chemical energy",
                edge_projection={
                    "start": {
                        "cue_id": "unit",
                        "quote": "Photosynthesis converts",
                    },
                    "end": {
                        "cue_id": "unit",
                        "quote": "into chemical energy",
                    },
                },
            )
        ],
        "transcript": transcript,
        "notes": "",
    }
    monkeypatch.setattr(
        pipeline_module, "_discover", lambda *_args, **_kwargs: _discovery()
    )
    monkeypatch.setattr(
        pipeline_module, "_run_clip", lambda *_args, **_kwargs: engine_out
    )
    prepare = mock.Mock()
    monkeypatch.setattr(
        pipeline_module.clip_engine_silence,
        "prepare_audio_source",
        prepare,
    )
    persisted: list[dict] = []
    pipeline = _pipeline()
    monkeypatch.setattr(
        pipeline,
        "_persist_engine_clip",
        lambda *, clip, **_kwargs: (
            persisted.append(clip) or "stored-caption-projection",
            mock.sentinel.metadata,
        ),
    )

    reels, _ = pipeline.ingest_topic(
        topic="photosynthesis",
        material_id="material",
        concept_id="concept",
        generation_context=GenerationContext(
            "fast", require_acoustic_boundaries=False
        ),
        retrieval_profile="deep",
        max_videos=1,
        max_reels=1,
    )

    assert reels == ["stored-caption-projection"]
    assert (persisted[0]["start"], persisted[0]["end"]) == pytest.approx(
        (1.85, 9.0)
    )
    projection = persisted[0]["search_context"]["boundary_diagnostics"][
        "lexical_projection"
    ]
    assert projection["caption_projection_verified"] is True
    prepare.assert_not_called()


def test_mixed_boundary_safety_rejects_missing_proof_and_speech_crossings() -> None:
    diagnostics = {
        "start_speech_handoff_verified": True,
        "end_speech_handoff_verified": False,
        "start_two_sided_required": True,
        "semantic_start_limit_sec": 12.0,
        "semantic_end_limit_sec": 40.0,
        "start_quiet": [11.8, 12.1],
        "end_quiet": [36.0, 36.3],
    }

    def safe(**overrides) -> bool:
        values = {
            "start_sec": 12.0,
            "end_sec": 36.2,
            "required_start_sec": 12.0,
            "required_end_sec": 30.0,
            "semantic_start_limit_sec": 12.0,
            "semantic_end_limit_sec": 40.0,
            "source_end_sec": 40.0,
            "diagnostics": diagnostics,
            "require_start_handoff": True,
            "require_end_handoff": False,
            "require_start_two_sided": True,
        }
        values.update(overrides)
        return pipeline_module._acoustic_range_is_safe(**values)

    assert safe() is True
    assert safe(diagnostics={**diagnostics, "start_two_sided_required": False}) is False
    assert safe(start_sec=12.1) is False
    assert safe(end_sec=29.8) is False
    assert safe(
        end_sec=40.2,
        source_end_sec=50.0,
        require_start_handoff=False,
    ) is False

    projected_end_diagnostics = {
        "start_speech_handoff_verified": False,
        "end_speech_handoff_verified": True,
        "end_two_sided_required": True,
        "semantic_start_limit_sec": 10.0,
        "semantic_end_limit_sec": 20.0,
        "start_quiet": [9.8, 10.0],
        "end_quiet": [19.7, 20.0],
    }
    assert safe(
        start_sec=10.0,
        end_sec=19.9,
        required_start_sec=10.0,
        required_end_sec=20.0,
        semantic_start_limit_sec=10.0,
        semantic_end_limit_sec=20.0,
        source_end_sec=30.0,
        diagnostics=projected_end_diagnostics,
        require_start_handoff=False,
        require_start_two_sided=False,
        require_end_handoff=True,
        require_end_two_sided=True,
    ) is True


def test_partial_cue_projection_uses_caption_tokens_without_native_words() -> None:
    transcript = {
        "source": "supadata",
        "native_mode": False,
        "artifact_key": "supadata-transcript:v2:missing-native-edge-timing",
        "duration": 10.0,
        "segments": [
            {
                "cue_id": "unit",
                "start": 0.0,
                "end": 10.0,
                "text": "Opening hook. Python functions package reusable instructions.",
            }
        ],
    }
    clip = _quality_clip(
        cue_id="unit",
        edge_projection={
            "start": {"cue_id": "unit", "quote": "Python functions package"}
        },
    )
    caption = pipeline_module._supadata_boundary_diagnostics(transcript, clip)
    assert caption is not None
    prepared = pipeline_module.clip_engine_silence.AudioPreparationResult(
        "ready",
        source=pipeline_module.clip_engine_silence.PreparedAudioSource(
            url="https://media.example/audio.m4a"
        ),
    )

    bounds, projection, error = pipeline_module._projected_speech_bounds(
        transcript,
        clip,
        caption,
        prepared,
    )

    assert error is None
    assert bounds == pytest.approx((20.0 / 7.0 - 0.15, 10.0))
    assert projection["caption_projection_verified"] is True
    assert projection["start"] == {
        "cue_id": "unit",
        "quote": "Python functions package",
        "mode": "caption_token_interpolation",
        "required_speech_sec": 2.707,
        "excluded_neighbor_onset_sec": 1.429,
    }


def test_projected_one_token_answer_end_advances_past_its_caption_onset() -> None:
    text = "The final answer is six cosine 6X now what is the next derivative"
    transcript = {
        "source": "supadata",
        "native_mode": False,
        "artifact_key": "supadata-transcript:v2:split-answer-end-coverage",
        "duration": 12.0,
        "segments": [
            {"cue_id": "answer-next", "start": 0.0, "end": 12.0, "text": text}
        ],
    }
    clip = _quality_clip(
        cue_id="answer-next",
        start=0.0,
        end=12.0,
        quote="The final answer is six cosine 6X",
        edge_projection={
            "end": {"cue_id": "answer-next", "quote": "6X"}
        },
    )
    caption = pipeline_module._supadata_boundary_diagnostics(transcript, clip)
    assert caption is not None

    bounds, projection, error = pipeline_module._projected_speech_bounds(
        transcript,
        clip,
        caption,
        None,
    )

    token_count = len(pipeline_module._QUOTE_WORD_RE.findall(text))
    last_onset = 12.0 * (6 / token_count)
    assert error is None
    assert bounds[1] > last_onset + 0.001
    assert projection["end"]["required_speech_sec"] == round(bounds[1], 3)
    assert projection["end"]["excluded_neighbor_onset_sec"] > bounds[1]


def test_first_occurrence_projection_handles_a_repeated_formula_operand() -> None:
    text = "X What is the derivative of X squared when X is positive"
    transcript = {
        "source": "supadata",
        "native_mode": False,
        "artifact_key": "supadata-transcript:v2:repeated-prefix-operand",
        "duration": 12.0,
        "segments": [
            {"cue_id": "answer-next", "start": 0.0, "end": 12.0, "text": text}
        ],
    }
    clip = _quality_clip(
        cue_id="answer-next",
        start=0.0,
        end=12.0,
        quote="X",
        edge_projection={
            "end": {
                "cue_id": "answer-next",
                "quote": "X",
                "occurrence": "first",
            }
        },
    )
    caption = pipeline_module._supadata_boundary_diagnostics(transcript, clip)
    assert caption is not None

    without_words = pipeline_module.clip_engine_silence.AudioPreparationResult(
        "ready",
        source=pipeline_module.clip_engine_silence.PreparedAudioSource(
            url="https://media.example/audio.m4a"
        ),
    )
    interpolated_bounds, interpolated, interpolated_error = (
        pipeline_module._projected_speech_bounds(
            transcript,
            clip,
            caption,
            without_words,
        )
    )

    lexical_words = tuple(
        pipeline_module.clip_engine_silence.lexical_timing.LexicalWord(word, onset)
        for word, onset in (
            ("x", 0.5),
            ("what", 1.0),
            ("is", 1.5),
            ("the", 2.0),
            ("derivative", 2.5),
            ("of", 3.0),
            ("x", 3.5),
            ("squared", 4.0),
            ("when", 4.5),
            ("x", 5.0),
            ("is", 5.5),
            ("positive", 6.0),
        )
    )
    with_words = pipeline_module.clip_engine_silence.AudioPreparationResult(
        "ready",
        source=pipeline_module.clip_engine_silence.PreparedAudioSource(
            url="https://media.example/audio.m4a",
            lexical_words=lexical_words,
        ),
    )
    lexical_bounds, lexical, lexical_error = pipeline_module._projected_speech_bounds(
        transcript,
        clip,
        caption,
        with_words,
    )
    cues = pipeline_module._selected_caption_cues(
        transcript,
        clip,
        boundary_bounds=(0.0, lexical_bounds[1]),
    )

    assert interpolated_error is None
    assert interpolated_bounds[1] < 1.1
    assert interpolated["end"]["mode"] == "caption_token_interpolation"
    assert lexical_error is None
    assert lexical_bounds[1] < 1.0
    assert lexical["end"]["excluded_neighbor_onset_sec"] == 1.0
    assert cues[0]["text"] == "X"

    ambiguous_clip = dict(clip)
    ambiguous_clip["edge_projection"] = {
        "end": {"cue_id": "answer-next", "quote": "X"}
    }
    ambiguous_caption = pipeline_module._supadata_boundary_diagnostics(
        transcript,
        ambiguous_clip,
    )
    assert ambiguous_caption is not None
    _bounds, _projection, ambiguous_error = pipeline_module._projected_speech_bounds(
        transcript,
        ambiguous_clip,
        ambiguous_caption,
        without_words,
    )
    assert ambiguous_error == "end_caption_interpolation_unavailable"


def test_selected_caption_snapshot_clamps_every_overlapping_cue_to_lexical_end() -> None:
    transcript = {
        "segments": [
            {
                "cue_id": "setup",
                "start": 284.32,
                "end": 290.639,
                "text": "squared with respect to x with respect to x",
            },
            {
                "cue_id": "answer-next",
                "start": 288.24,
                "end": 294.08,
                "text": "which is exactly what dhdx is this right over here",
            },
        ],
    }
    clip = {
        "cue_ids": ["setup", "answer-next"],
        "edge_projection": {
            "end": {
                "cue_id": "answer-next",
                "quote": "which is exactly what dhdx is",
            },
        },
    }

    cues = pipeline_module._selected_caption_cues(
        transcript,
        clip,
        boundary_bounds=(284.32, 290.351),
    )

    assert [cue["end"] for cue in cues] == [290.351, 290.351]
    assert cues[-1]["text"] == "which is exactly what dhdx is"

    context = {
        "selection_contract_version": "quality_silence_v32",
        "boundary_status": "context_aligned",
        "speech_corridor_verified": True,
        "selection_caption_cues": [
            {
                "cue_id": "opening",
                "start": 76.4,
                "end": 82.0,
                "text": "Suppose x squared equals a squared",
            },
            *cues,
        ],
        "boundary_diagnostics": {
            "method": "transcript_context",
            "context_aligned": True,
            "acoustic_verified": False,
            "transcript": {
                "context_aligned": True,
                "stage": "analyze",
                "reason": "start_silence_not_found",
                "required_speech_range": [76.4, 290.351],
                "semantic_range": [76.4, 290.351],
                "final_range": [76.4, 290.351],
            },
        },
    }

    assert pipeline_module.clip_engine_silence.persisted_boundary_is_usable(
        context,
        t_start=76.4,
        t_end=290.351,
    ) is True


def test_start_projection_clamps_rolling_cue_and_preserves_end_handoff() -> None:
    transcript = {
        "source": "supadata",
        "native_mode": False,
        "artifact_key": "supadata-transcript:v2:rolling-caption-projection",
        "duration": 20.0,
        "segments": [
            {
                "cue_id": "selected",
                "start": 0.0,
                "end": 10.0,
                "text": (
                    "Welcome back chain rule differentiates composite functions by "
                    "multiplying derivatives"
                ),
            },
            {
                "cue_id": "next",
                "start": 8.0,
                "end": 20.0,
                "text": "The next topic begins with implicit differentiation.",
            },
        ],
    }
    clip = _quality_clip(
        cue_id="selected",
        start=0.0,
        end=10.0,
        quote="chain rule differentiates composite functions by multiplying derivatives",
        edge_projection={
            "start": {
                "cue_id": "selected",
                "quote": "chain rule differentiates",
            }
        },
    )
    caption = pipeline_module._supadata_boundary_diagnostics(transcript, clip)
    assert caption is not None

    bounds, projection, error = pipeline_module._projected_speech_bounds(
        transcript,
        clip,
        caption,
        None,
    )

    assert error is None
    assert bounds == pytest.approx((1.45, 8.0))
    assert projection["start"]["required_speech_sec"] == 1.45
    assert projection["caption_overlap_end_handoff"] == {
        "mode": "next_cue_onset_two_sided_quiet",
        "selected_cue_id": "selected",
        "next_cue_id": "next",
        "display_end_sec": 10.0,
        "next_cue_onset_sec": 8.0,
        "overlap_sec": 2.0,
    }


def test_implausibly_short_rolling_projection_fails_open_to_full_cue() -> None:
    text = (
        "Welcome back chain rule differentiates composite functions by multiplying "
        "the outer and inner derivatives"
    )
    transcript = {
        "source": "supadata",
        "native_mode": False,
        "artifact_key": "supadata-transcript:v2:short-rolling-projection",
        "duration": 20.0,
        "segments": [
            {"cue_id": "selected", "start": 0.0, "end": 10.0, "text": text},
            {
                "cue_id": "next",
                "start": 0.5,
                "end": 20.0,
                "text": "The next topic starts here with a separate explanation.",
            },
        ],
    }
    clip = _quality_clip(
        cue_id="selected",
        start=0.0,
        end=10.0,
        quote="chain rule differentiates composite functions by multiplying",
        kind="educational",
        edge_projection={
            "start": {
                "cue_id": "selected",
                "quote": "chain rule differentiates",
            }
        },
    )

    verified = pipeline_module._verified_direct_adapter_clips(
        source_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        engine_out={"clips": [clip], "transcript": transcript},
        should_cancel=None,
        exact_topic="chain rule",
    )

    assert len(verified) == 1
    assert (verified[0]["start"], verified[0]["end"]) == (0.0, 10.0)
    fallback = verified[0]["search_context"]["boundary_diagnostics"][
        "lexical_projection"
    ]["context_fallback"]
    assert fallback["reason"].startswith("full_cue_fallback:")


def test_transcript_only_adapter_preserves_same_cue_filler_projection() -> None:
    text = (
        "Welcome back. Photosynthesis converts light energy into chemical energy. "
        "Thanks for watching."
    )
    transcript = {
        "source": "supadata",
        "native_mode": False,
        "artifact_key": "supadata-transcript:v2:caption-token-projection",
        "duration": 12.0,
        "segments": [
            {"cue_id": "unit", "start": 0.0, "end": 12.0, "text": text},
        ],
    }
    clip = _quality_clip(
        cue_id="unit",
        start=0.0,
        end=12.0,
        quote="Photosynthesis converts light energy into chemical energy",
        kind="educational",
        edge_projection={
            "start": {"cue_id": "unit", "quote": "Photosynthesis converts"},
            "end": {"cue_id": "unit", "quote": "into chemical energy"},
        },
    )

    verified = pipeline_module._verified_direct_adapter_clips(
        source_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        engine_out={"clips": [clip], "transcript": transcript},
        should_cancel=None,
        exact_topic="photosynthesis",
    )

    assert len(verified) == 1
    assert verified[0]["start"] == pytest.approx(1.85)
    assert verified[0]["end"] == pytest.approx(9.0)
    projection = verified[0]["search_context"]["boundary_diagnostics"][
        "lexical_projection"
    ]
    assert projection["caption_projection_verified"] is True


@pytest.mark.parametrize(
    (
        "segments",
        "expected_corridor",
        "expected_handoffs",
        "expected_two_sided",
    ),
    [
        (
            [
                {"cue_id": "prior", "start": 0.0, "end": 10.02, "text": "Prior."},
                {"cue_id": "selected", "start": 10.0, "end": 20.0, "text": "Selected."},
            ],
            (10.0, 20.0),
            (True, False),
            (True, False),
        ),
        (
            [
                {"cue_id": "selected", "start": 0.0, "end": 10.0, "text": "Selected."},
                {"cue_id": "next", "start": 9.9995, "end": 20.0, "text": "Next."},
            ],
            (0.0, 10.0),
            (False, False),
            (False, True),
        ),
    ],
)
def test_rolling_caption_overlap_uses_cue_onset_handoffs(
    segments, expected_corridor, expected_handoffs, expected_two_sided,
) -> None:
    transcript = {
        "source": "supadata",
        "native_mode": False,
        "artifact_key": "supadata-transcript:v2:overlap",
        "duration": 20.0,
        "segments": segments,
    }
    selected = next(segment for segment in segments if segment["cue_id"] == "selected")
    clip = {
        "cue_ids": ["selected"],
        "start": selected["start"],
        "end": selected["end"],
    }
    diagnostics = pipeline_module._supadata_boundary_diagnostics(transcript, clip)
    assert diagnostics is not None

    start, end, error = pipeline_module._selected_speech_corridor(
        transcript,
        clip,
        diagnostics,
        source_end_sec=20.0,
    )

    plan = pipeline_module._acoustic_boundary_plan(
        transcript,
        clip,
        {},
        speech_bounds=(selected["start"], selected["end"]),
        search_limits=(start, end),
    )

    assert error is None
    assert (start, end) == expected_corridor
    assert plan is not None
    assert plan[1] == selected["end"]
    assert plan[2:4] == expected_handoffs
    assert plan[4:6] == expected_two_sided


def test_nonlexical_start_corridor_includes_only_the_preceding_caption_gap() -> None:
    transcript = {
        "source": "supadata",
        "native_mode": False,
        "artifact_key": "supadata-transcript:v2:start-caption-gap",
        "duration": 20.0,
        "segments": [
            {
                "cue_id": "prior",
                "start": 0.0,
                "end": 4.0,
                "text": "The prior teaching unit ends here.",
            },
            {
                "cue_id": "selected",
                "start": 10.0,
                "end": 20.0,
                "text": "The selected explanation begins after a long pause.",
            },
        ],
    }
    clip = {"cue_ids": ["selected"], "start": 10.0, "end": 20.0}
    diagnostics = pipeline_module._supadata_boundary_diagnostics(transcript, clip)
    assert diagnostics is not None

    start, end, error = pipeline_module._selected_speech_corridor(
        transcript,
        clip,
        diagnostics,
        source_end_sec=20.0,
    )
    plan = pipeline_module._acoustic_boundary_plan(
        transcript,
        clip,
        {},
        speech_bounds=(10.0, 20.0),
        search_limits=(start, end),
    )

    assert error is None
    assert (start, end) == (4.0, 20.0)
    assert plan == (10.0, 20.0, True, False, True, False)


def test_nonlexical_end_progresses_from_required_speech_to_next_cue_fence() -> None:
    transcript = {
        "source": "supadata",
        "native_mode": False,
        "artifact_key": "supadata-transcript:v2:caption-gap",
        "duration": 20.0,
        "segments": [
            {
                "cue_id": "selected",
                "start": 0.0,
                "end": 10.0,
                "text": "The selected explanation reaches its conclusion.",
            },
            {
                "cue_id": "next",
                "start": 10.5,
                "end": 20.0,
                "text": "The next teaching unit begins here.",
            },
        ],
    }
    clip = {"cue_ids": ["selected"], "start": 0.0, "end": 10.0}
    diagnostics = pipeline_module._supadata_boundary_diagnostics(transcript, clip)
    assert diagnostics is not None
    start, end, error = pipeline_module._selected_speech_corridor(
        transcript,
        clip,
        diagnostics,
        source_end_sec=20.0,
    )

    plan = pipeline_module._acoustic_boundary_plan(
        transcript,
        clip,
        {},
        speech_bounds=(0.0, 10.0),
        search_limits=(start, end),
    )

    assert error is None
    assert (start, end) == (0.0, 10.5)
    assert plan == (0.0, 10.0, False, False, False, True)


def test_rolling_display_end_uses_only_a_two_sided_next_onset_handoff() -> None:
    transcript = {
        "source": "supadata",
        "native_mode": False,
        "artifact_key": "supadata-transcript:v2:rolling-display-end",
        "duration": 20.0,
        "segments": [
            {
                "cue_id": "selected",
                "start": 0.0,
                "end": 10.0,
                "text": "The selected explanation reaches its conclusion.",
            },
            {
                "cue_id": "next",
                "start": 8.0,
                "end": 20.0,
                "text": "The next teaching unit begins here.",
            },
        ],
    }
    clip = {
        "cue_ids": ["selected"],
        "start": 0.0,
        "end": 10.0,
        "start_quote": "The selected explanation reaches",
        "end_quote": "explanation reaches its conclusion",
    }
    caption = pipeline_module._supadata_boundary_diagnostics(transcript, clip)
    assert caption is not None
    prepared = pipeline_module.clip_engine_silence.AudioPreparationResult(
        "ready",
        source=pipeline_module.clip_engine_silence.PreparedAudioSource(
            url="https://media.example/audio.m4a"
        ),
    )

    bounds, projection, error = pipeline_module._projected_speech_bounds(
        transcript,
        clip,
        caption,
        prepared,
    )
    corridor = pipeline_module._selected_speech_corridor(
        transcript,
        clip,
        caption,
        source_end_sec=20.0,
        required_speech_bounds=bounds,
        projection_diagnostics=projection,
    )
    plan = pipeline_module._acoustic_boundary_plan(
        transcript,
        clip,
        projection,
        speech_bounds=bounds,
        search_limits=corridor[:2],
    )

    assert error is None
    assert bounds == (0.0, 8.0)
    assert projection == {
        "caption_overlap_end_handoff": {
            "mode": "next_cue_onset_two_sided_quiet",
            "selected_cue_id": "selected",
            "next_cue_id": "next",
            "display_end_sec": 10.0,
            "next_cue_onset_sec": 8.0,
            "overlap_sec": 2.0,
        }
    }
    assert corridor == (0.0, 8.0, None)
    assert plan == (0.0, 8.0, False, True, False, True)


def test_unverified_rolling_overlap_falls_back_to_complete_selected_cue(
    monkeypatch,
) -> None:
    transcript = {
        "source": "supadata",
        "native_mode": False,
        "artifact_key": "supadata-transcript:v2:unverified-overlap",
        "duration": 20.0,
        "segments": [
            {
                "cue_id": "selected",
                "start": 0.0,
                "end": 10.0,
                "text": (
                    "Photosynthesis converts light into stored chemical energy "
                    "for growing cells."
                ),
            },
            {
                "cue_id": "next",
                "start": 8.0,
                "end": 20.0,
                "text": "The next topic begins with cellular respiration.",
            },
        ],
    }
    clip = _quality_clip(
        cue_id="selected",
        start=0.0,
        end=10.0,
        kind="educational",
        quote=(
            "Photosynthesis converts light into stored chemical energy for "
            "growing cells."
        ),
    )
    verify_audio = mock.Mock(
        return_value=pipeline_module.clip_engine_silence.SilenceVerificationResult(
            "unavailable",
            0.0,
            8.0,
            {"stage": "end", "reason": "end_silence_not_found"},
        )
    )
    monkeypatch.setattr(
        pipeline_module.clip_engine_silence,
        "verify_acoustic_boundaries",
        verify_audio,
    )
    prepared = pipeline_module.clip_engine_silence.AudioPreparationResult(
        "ready",
        source=pipeline_module.clip_engine_silence.PreparedAudioSource(
            url="https://media.example/audio.m4a",
            duration_sec=20.0,
        ),
    )

    verified = pipeline_module._verified_direct_adapter_clips(
        source_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        engine_out={"clips": [clip], "transcript": transcript},
        should_cancel=None,
        prepared_audio=prepared,
        exact_topic="photosynthesis",
    )

    assert verify_audio.call_args.args[2] == 8.0
    assert len(verified) == 1
    assert (verified[0]["start"], verified[0]["end"]) == (0.0, 10.0)
    boundary = verified[0]["search_context"]["boundary_diagnostics"]
    assert boundary["acoustic_verified"] is False
    assert boundary["transcript"]["final_range"] == [0.0, 10.0]


def test_rolling_caption_continuation_never_shortens_required_speech() -> None:
    transcript = {
        "source": "supadata",
        "native_mode": False,
        "artifact_key": "supadata-transcript:v2:continued-overlap",
        "duration": 20.0,
        "segments": [
            {
                "cue_id": "selected",
                "start": 0.0,
                "end": 10.0,
                "text": "This fraction-like cancellation can help with intuition and then",
            },
            {
                "cue_id": "next",
                "start": 8.0,
                "end": 20.0,
                "text": "what you're left with is the derivative of the whole function.",
            },
        ],
    }
    clip = {
        "cue_ids": ["selected"],
        "start": 0.0,
        "end": 10.0,
        "start_quote": "This fraction-like cancellation",
        "end_quote": "help with intuition and then",
    }
    caption = pipeline_module._supadata_boundary_diagnostics(transcript, clip)
    assert caption is not None

    bounds, projection, error = pipeline_module._projected_speech_bounds(
        transcript,
        clip,
        caption,
        None,
    )

    assert error is None
    assert bounds == (0.0, 10.0)
    assert projection == {}


def test_native_lexical_end_wins_over_rolling_display_overlap() -> None:
    transcript = {
        "source": "supadata",
        "native_mode": False,
        "artifact_key": "supadata-transcript:v2:lexical-overlap-end",
        "duration": 20.0,
        "segments": [
            {
                "cue_id": "selected",
                "start": 0.0,
                "end": 10.0,
                "text": "Selected teaching reaches the conclusion",
            },
            {
                "cue_id": "next",
                "start": 8.0,
                "end": 20.0,
                "text": "Next topic begins here",
            },
        ],
    }
    clip = {
        "cue_ids": ["selected"],
        "start": 0.0,
        "end": 10.0,
        "end_quote": "teaching reaches the conclusion",
    }
    caption = pipeline_module._supadata_boundary_diagnostics(transcript, clip)
    assert caption is not None
    lexical_words = tuple(
        pipeline_module.clip_engine_silence.lexical_timing.LexicalWord(word, onset)
        for word, onset in (
            ("selected", 4.0),
            ("teaching", 5.0),
            ("reaches", 6.0),
            ("the", 6.5),
            ("conclusion", 7.0),
            ("next", 8.2),
        )
    )
    prepared = pipeline_module.clip_engine_silence.AudioPreparationResult(
        "ready",
        source=pipeline_module.clip_engine_silence.PreparedAudioSource(
            url="https://media.example/audio.m4a",
            lexical_words=lexical_words,
        ),
    )

    bounds, projection, error = pipeline_module._projected_speech_bounds(
        transcript,
        clip,
        caption,
        prepared,
    )
    corridor = pipeline_module._selected_speech_corridor(
        transcript,
        clip,
        caption,
        source_end_sec=20.0,
        required_speech_bounds=bounds,
        projection_diagnostics=projection,
    )
    plan = pipeline_module._acoustic_boundary_plan(
        transcript,
        clip,
        projection,
        speech_bounds=bounds,
        search_limits=corridor[:2],
    )

    assert error is None
    assert bounds == (0.0, 7.0)
    assert "caption_overlap_end_handoff" not in projection
    assert projection["end"]["excluded_neighbor_onset_sec"] == 8.2
    assert corridor == (0.0, 8.2, None)
    assert plan == (0.0, 7.0, False, False, False, True)


@pytest.mark.parametrize("next_onset", [5.0, 5.001, 5.04, 5.5, 5.749])
def test_rolling_display_handoff_never_discards_the_selected_last_cue(
    next_onset: float,
) -> None:
    transcript = {
        "source": "supadata",
        "native_mode": False,
        "artifact_key": "supadata-transcript:v2:shared-onset",
        "duration": 12.0,
        "segments": [
            {"cue_id": "a", "start": 0.0, "end": 5.0, "text": "Setup."},
            {
                "cue_id": "selected-end",
                "start": 5.0,
                "end": 10.0,
                "text": "The required conclusion is spoken here.",
            },
            {
                "cue_id": "next",
                "start": next_onset,
                "end": 12.0,
                "text": "The following cue begins here.",
            },
        ],
    }
    clip = {
        "cue_ids": ["a", "selected-end"],
        "start": 0.0,
        "end": 10.0,
    }
    caption = pipeline_module._supadata_boundary_diagnostics(transcript, clip)
    assert caption is not None
    prepared = pipeline_module.clip_engine_silence.AudioPreparationResult(
        "ready",
        source=pipeline_module.clip_engine_silence.PreparedAudioSource(
            url="https://media.example/audio.m4a"
        ),
    )

    bounds, projection, error = pipeline_module._projected_speech_bounds(
        transcript,
        clip,
        caption,
        prepared,
    )

    assert error is None
    assert bounds == (0.0, 10.0)
    assert projection == {}


@pytest.mark.parametrize(
    "selected_text",
    [
        "这是一个非常重要的结论",
        "这是 一个非常重要的结论",
        "DNA 复制这是一个非常重要的结论",
    ],
)
def test_rolling_display_handoff_counts_no_space_script_speech_units(
    selected_text: str,
) -> None:
    transcript = {
        "segments": [
            {
                "cue_id": "selected",
                "start": 0.0,
                "end": 2.0,
                "text": selected_text,
            },
            {
                "cue_id": "next",
                "start": 0.25,
                "end": 3.0,
                "text": "下一个解释开始",
            },
        ]
    }

    assert pipeline_module._overlapping_caption_end_handoff(
        transcript,
        {"cue_ids": ["selected"]},
    ) is None


def test_caption_overlap_larger_than_ownership_epsilon_fails_closed() -> None:
    transcript = {
        "source": "supadata",
        "native_mode": False,
        "artifact_key": "supadata-transcript:v2:unsafe-overlap",
        "duration": 20.0,
        "segments": [
            {
                "cue_id": "selected",
                "start": 0.0,
                "end": 10.0,
                "text": "Selected teaching ends here.",
            },
            {
                "cue_id": "next",
                "start": 9.98,
                "end": 20.0,
                "text": "Next teaching begins here.",
            },
        ],
    }
    clip = {"cue_ids": ["selected"], "start": 0.0, "end": 10.0}
    diagnostics = pipeline_module._supadata_boundary_diagnostics(transcript, clip)
    assert diagnostics is not None

    assert pipeline_module._selected_speech_corridor(
        transcript,
        clip,
        diagnostics,
        source_end_sec=20.0,
    )[2] == "selected_cue_range_unavailable"


def test_acoustic_boundary_plan_fails_closed_for_missing_cue_ids() -> None:
    transcript = _transcript()

    assert pipeline_module._acoustic_boundary_plan(
        transcript,
        {"cue_ids": ["missing"]},
        {},
        speech_bounds=(0.0, 10.0),
        search_limits=(0.0, 10.0),
    ) is None


def test_transcript_edge_speech_inside_media_uses_one_sided_handoffs() -> None:
    transcript = {
        "source": "supadata",
        "native_mode": False,
        "artifact_key": "supadata-transcript:v2:bounded-single-cue",
        "duration": 20.0,
        "segments": [
            {
                "cue_id": "selected",
                "start": 10.0,
                "end": 20.0,
                "text": "Python closures retain values from their enclosing scope.",
            }
        ],
    }
    clip = {"cue_ids": ["selected"], "start": 10.0, "end": 20.0}
    diagnostics = pipeline_module._supadata_boundary_diagnostics(transcript, clip)
    assert diagnostics is not None

    start_limit, end_limit, error = pipeline_module._selected_speech_corridor(
        transcript,
        clip,
        diagnostics,
        source_end_sec=100.0,
    )
    plan = pipeline_module._acoustic_boundary_plan(
        transcript,
        clip,
        {},
        speech_bounds=(10.0, 20.0),
        search_limits=(start_limit, end_limit),
    )

    assert error is None
    assert (start_limit, end_limit) == (0.0, 100.0)
    assert plan == (10.0, 20.0, True, True, False, False)


@pytest.mark.parametrize(
    "verification",
    [
        {
                "verified": True,
                "start_sec": 0.0,
                "end_sec": 9.5,
                "diagnostics": {},
        },
        {
                "verified": True,
                "start_sec": 0.049,
                "end_sec": 10.0,
                "diagnostics": {},
        },
        {
                "verified": True,
                "start_sec": 0.0,
                "end_sec": 9.951,
                "diagnostics": {},
        },
    ],
)
def test_unsafe_acoustic_success_falls_back_without_rejecting_good_clip(
    monkeypatch,
    verification: dict,
) -> None:
    transcript = _transcript()
    engine_out = {
        "clips": [_quality_clip(candidate_id="python-functions", score=0.8)],
        "transcript": transcript,
        "notes": "",
    }
    monkeypatch.setattr(pipeline_module, "_discover", lambda *_args, **_kwargs: _discovery())
    monkeypatch.setattr(pipeline_module, "_run_clip", lambda *_args, **_kwargs: engine_out)
    monkeypatch.setattr(
        pipeline_module.clip_engine_silence,
        "prepare_audio_source",
        mock.Mock(return_value=mock.sentinel.prepared_audio),
    )
    monkeypatch.setattr(
        pipeline_module.clip_engine_silence,
        "verify_acoustic_boundaries",
        mock.Mock(return_value=mock.Mock(**verification)),
    )
    stored: list[dict] = []
    pipeline = _pipeline()

    def persist(*, clip, **_kwargs):
        stored.append(clip)
        return ("deferred-reel", mock.sentinel.metadata)

    monkeypatch.setattr(pipeline, "_persist_engine_clip", persist)
    emitted: list[str] = []
    context = GenerationContext("slow", require_acoustic_boundaries=True)

    reels, _ = pipeline.ingest_topic(
        topic="Intro to Python",
        material_id="material",
        concept_id="concept",
        generation_context=context,
        retrieval_profile="deep",
        max_videos=1,
        max_reels=1,
        on_reel_created=emitted.append,
    )

    assert reels == ["deferred-reel"]
    assert emitted == ["deferred-reel"]
    assert len(stored) == 1
    assert (stored[0]["start"], stored[0]["end"]) == (0.0, 10.0)
    assert stored[0]["search_context"]["boundary_status"] == "context_aligned"
    assert context.counters()["stored_clips"] == 1
    assert context.counters()["deferred_clips"] == 0
    assert context.counters()["persisted_clips"] == 1
    assert context.counters()["verified_clips"] == 0
    assert context.counters()["level_deferred_clips"] == 0
    assert context.counters()["permanently_rejected_clips"] == 0


def test_production_transcript_boundary_surfaces_without_audio_work(
    monkeypatch,
) -> None:
    transcript = _transcript()
    engine_out = {
        "clips": [_quality_clip(candidate_id="python-functions", score=0.8)],
        "transcript": transcript,
        "notes": "",
    }
    monkeypatch.setattr(pipeline_module, "_discover", lambda *_args, **_kwargs: _discovery())
    monkeypatch.setattr(pipeline_module, "_run_clip", lambda *_args, **_kwargs: engine_out)
    prepare_audio = mock.Mock()
    monkeypatch.setattr(
        pipeline_module.clip_engine_silence,
        "prepare_audio_source",
        prepare_audio,
    )
    verify_audio = mock.Mock()
    monkeypatch.setattr(
        pipeline_module.clip_engine_silence,
        "verify_acoustic_boundaries",
        verify_audio,
    )
    stored: list[dict] = []
    pipeline = _pipeline()

    def persist(*, clip, **_kwargs):
        stored.append(clip)
        return ("context-aligned-reel", mock.sentinel.metadata)

    monkeypatch.setattr(pipeline, "_persist_engine_clip", persist)
    emitted: list[str] = []
    context = GenerationContext("fast", require_acoustic_boundaries=False)

    reels, _ = pipeline.ingest_topic(
        topic="Intro to Python",
        material_id="material",
        concept_id="concept",
        generation_context=context,
        retrieval_profile="deep",
        max_videos=1,
        max_reels=1,
        on_reel_created=emitted.append,
    )

    assert reels == ["context-aligned-reel"]
    assert emitted == ["context-aligned-reel"]
    assert len(stored) == 1
    assert (stored[0]["start"], stored[0]["end"]) == (0.0, 10.0)
    boundary = stored[0]["search_context"]
    assert boundary["boundary_status"] == "context_aligned"
    assert boundary["speech_corridor_verified"] is True
    assert boundary["boundary_diagnostics"]["acoustic_verified"] is False
    assert boundary["boundary_diagnostics"]["context_aligned"] is True
    assert boundary["boundary_diagnostics"]["method"] == "transcript_context"
    prepare_audio.assert_not_called()
    verify_audio.assert_not_called()
    assert context.counters()["stored_clips"] == 1
    assert context.counters()["permanently_rejected_clips"] == 0


def test_unsafe_refinement_fallback_surfaces_before_later_level_reservoir(
    monkeypatch,
) -> None:
    transcript = _transcript()
    transcript["segments"][1] = {
        "cue_id": "advanced-python",
        "start": 10.0,
        "end": 20.0,
        "text": (
            "Python generators yield values lazily and preserve suspended "
            "execution state."
        ),
    }
    engine_out = {
        "clips": [
            _quality_clip(candidate_id="invalid-boundary", score=0.95),
            _quality_clip(
                cue_id="advanced-python",
                start=10.0,
                end=20.0,
                quote=(
                    "Python generators yield values lazily and preserve suspended "
                    "execution state."
                ),
                candidate_id="advanced-reservoir",
                score=0.9,
                difficulty=0.9,
            ),
        ],
        "transcript": transcript,
        "notes": "",
    }
    monkeypatch.setattr(
        pipeline_module,
        "_discover",
        lambda *_args, **_kwargs: _discovery(),
    )
    monkeypatch.setattr(
        pipeline_module,
        "_run_clip",
        lambda *_args, **_kwargs: engine_out,
    )
    monkeypatch.setattr(
        pipeline_module.clip_engine_silence,
        "prepare_audio_source",
        mock.Mock(return_value=mock.sentinel.prepared_audio),
    )

    def verify(_source, start_sec, end_sec, **_kwargs):
        if start_sec == 0.0:
            return mock.Mock(
                verified=True,
                start_sec=start_sec,
                end_sec=end_sec - 0.5,
                diagnostics={},
            )
        return mock.Mock(
            verified=True,
            start_sec=start_sec,
            end_sec=end_sec,
            diagnostics={
                "threshold_dbfs": -38.0,
                "start_quiet": [9.9, 10.1],
                "end_quiet": [19.8, 20.0],
                "start_speech_handoff_verified": True,
                "end_speech_handoff_verified": False,
                "start_two_sided_required": True,
                "end_two_sided_required": False,
                "semantic_start_limit_sec": 10.0,
                "semantic_end_limit_sec": 20.0,
            },
        )

    monkeypatch.setattr(
        pipeline_module.clip_engine_silence,
        "verify_acoustic_boundaries",
        mock.Mock(side_effect=verify),
    )
    stored: list[dict] = []
    pipeline = _pipeline()

    def persist(*, clip, **_kwargs):
        stored.append(clip)
        return ("advanced-reservoir-reel", mock.sentinel.metadata)

    monkeypatch.setattr(pipeline, "_persist_engine_clip", persist)
    context = GenerationContext("slow", require_acoustic_boundaries=True)

    reels, _ = pipeline.ingest_topic(
        topic="Intro to Python",
        material_id="material",
        concept_id="concept",
        generation_context=context,
        retrieval_profile="deep",
        knowledge_level="beginner",
        max_videos=1,
        max_reels=2,
        max_persisted_reels=1,
    )

    assert reels == ["advanced-reservoir-reel"]
    assert len(stored) == 1
    boundary = stored[0]["search_context"]
    assert boundary["boundary_status"] == "context_aligned"
    assert boundary["surface_eligible"] is True
    assert context.counters()["stored_clips"] == 1
    assert context.counters()["deferred_clips"] == 0
    assert context.counters()["persisted_clips"] == 1
    assert context.counters()["verified_clips"] == 0
    assert context.counters()["level_deferred_clips"] == 0
    assert context.counters()["permanently_rejected_clips"] == 0


def test_generation_count_excludes_all_explicitly_deferred_boundary_rows(
    monkeypatch,
) -> None:
    strict_current = {
        "surface_eligible": True,
        "selection_contract_version": "quality_silence_v32",
        "speech_corridor_verified": True,
        "boundary_status": "verified",
        "boundary_diagnostics": {
            "acoustic_verified": True,
            "acoustic": {"threshold_dbfs": -38},
        },
    }
    transcript_current = {
        "surface_eligible": True,
        "selection_contract_version": "quality_silence_v32",
        "speech_corridor_verified": True,
        "boundary_status": "context_aligned",
        "selection_caption_cues": [
            {"start": 2.0, "end": 9.0, "text": "A complete teaching thought."}
        ],
        "boundary_diagnostics": {
            "method": "transcript_context",
            "context_aligned": True,
            "acoustic_verified": False,
            "transcript": {
                "context_aligned": True,
                "stage": "transcript",
                "reason": "complete_discourse_boundary",
                "required_speech_range": [2.0, 9.0],
                "semantic_range": [1.0, 10.0],
                "final_range": [2.0, 9.0],
            },
        },
    }
    monkeypatch.setattr(
        main,
        "fetch_all",
        lambda *_args, **_kwargs: [
            {"search_context_json": '{"surface_eligible": false}'},
            {"search_context_json": '{"surface_eligible": "false"}'},
            {"search_context_json": '{"surface_eligible": true, "boundary_status": "unavailable"}'},
            {"search_context_json": '{"surface_eligible": true}'},
            {"search_context_json": '{"surface_eligible": true, "boundary_status": "caption_aligned"}'},
            {"search_context_json": '{"surface_eligible": true, "boundary_status": "verified", "boundary_diagnostics": {"acoustic_verified": false}}'},
            {"search_context_json": '{"surface_eligible": true, "boundary_status": "verified", "boundary_diagnostics": {"acoustic_verified": true}}'},
            {"search_context_json": '{"surface_eligible": true, "boundary_status": "verified", "boundary_diagnostics": {"acoustic_verified": true, "acoustic": {"adaptive_quiet": true, "start_threshold_dbfs": -24, "end_threshold_dbfs": -38}}}'},
            {"search_context_json": '{"surface_eligible": true, "boundary_status": "verified", "boundary_diagnostics": {"acoustic_verified": true, "acoustic": {"threshold_dbfs": -38, "start_threshold_dbfs": -38, "end_threshold_dbfs": -38}}}'},
            {"search_context_json": '{"surface_eligible": true, "boundary_status": "verified", "boundary_diagnostics": {"acoustic_verified": true, "acoustic": {"start_threshold_dbfs": -37.9}}}'},
            {"search_context_json": json.dumps(strict_current)},
            {"search_context_json": json.dumps(transcript_current)},
            {"search_context_json": json.dumps({
                **transcript_current,
                "selection_contract_version": "quality_silence_v12",
            })},
            {"search_context_json": '[]'},
        ],
    )

    # Count only current strict diagnostics; missing, adaptive, noisier, and
    # non-object metadata never count as ready v2 inventory.
    assert main._count_generation_reels(object(), "generation") == 2


def test_failed_boundary_storage_does_not_consume_ready_material_cap(
    monkeypatch,
) -> None:
    deferred = [
        {
            "search_context_json": (
                '{"surface_eligible": false, "boundary_status": "unavailable"}'
            )
        }
        for _ in range(main.MAX_REELS_PER_MATERIAL + 5)
    ]
    deferred.append({
        "search_context_json": json.dumps({
            "surface_eligible": True,
            "selection_contract_version": "quality_silence_v32",
            "speech_corridor_verified": True,
            "boundary_status": "verified",
            "boundary_diagnostics": {
                "acoustic_verified": True,
                "acoustic": {"threshold_dbfs": -38},
            },
        }),
        })
    monkeypatch.setattr(main, "fetch_all", lambda *_args, **_kwargs: deferred)

    assert main._count_material_ready_reels(object(), "material") == 1


def test_one_word_biology_logistics_never_surfaces_but_concrete_teaching_does(
    monkeypatch,
) -> None:
    transcript = {
        "source": "supadata",
        "native_mode": False,
        "artifact_key": "supadata-transcript:v2:biology",
        "duration": 20.0,
        "segments": [
            {
                "cue_id": "logistics",
                "start": 0.0,
                "end": 10.0,
                "text": "Welcome to biology at Stanford University and review the course grading policy today.",
            },
            {
                "cue_id": "teaching",
                "start": 10.0,
                "end": 20.0,
                "text": "Cells convert nutrient energy into ATP through a sequence of enzyme controlled reactions.",
            },
        ],
    }
    engine_out = {
        "clips": [
            _quality_clip(
                cue_id="logistics",
                quote="Welcome to biology at Stanford University",
                candidate_id="logistics",
                score=0.0,
                directly_teaches_topic=False,
                substantive=False,
                factually_grounded=False,
            ),
            _quality_clip(
                cue_id="teaching",
                start=10.0,
                end=20.0,
                quote="Cells convert nutrient energy into ATP through a sequence of enzyme controlled reactions.",
                candidate_id="cell-energy",
                score=0.8,
            ),
            _quality_clip(
                cue_id="teaching",
                start=10.0,
                end=20.0,
                quote="Cells convert nutrient energy into ATP through a sequence of enzyme controlled reactions.",
                candidate_id="missing-grounding-contract",
                factually_grounded=None,
            ),
        ],
        "transcript": transcript,
        "notes": "",
    }
    monkeypatch.setattr(pipeline_module, "_run_clip", lambda *_args, **_kwargs: engine_out)
    video = {
        **_video(),
        "_topic_terms": ["biology"],
        "_knowledge_level": "beginner",
    }

    pipeline, _embedding = _pipeline_with_semantic(0.3)
    _video_row, clips, _engine = pipeline._clip_and_filter(
        video, "biology", "en"
    )

    assert [clip["selection_candidate_id"] for clip in clips] == [
        "dQw4w9WgXcQ::cell-energy"
    ]
    assert clips[0]["score"] == 0.8
    assert clips[0]["search_context"]["topic_evidence_quote"].startswith("Cells convert")
    assert clips[0]["search_context"]["factually_grounded"] is True


def test_modern_selector_paraphrase_survives_without_local_semantic_model(
    monkeypatch,
) -> None:
    quote = (
        "Cells convert nutrient energy into ATP through a sequence of enzyme "
        "controlled reactions."
    )
    engine_out = _one_cue_selector_result(quote, score=1.0)
    monkeypatch.setattr(pipeline_module, "_run_clip", lambda *_args, **_kwargs: engine_out)
    context = GenerationContext("slow")

    _video_row, clips, _engine = _pipeline()._clip_and_filter(
        {**_video(), "literal_match": False, "_topic_terms": ["biology"]},
        "biology",
        "en",
        generation_context=context,
    )

    assert len(clips) == 1
    assert clips[0]["topic_evidence_terms"] == [quote]
    assert context.counters()["topic_rejections"] == 0
    assert context.usage_payload()["summary"]["rejection_reason_counts"] == {}


@pytest.mark.parametrize(
    "hard_gate_override",
    [
        {"topic_relevance": 0.74},
        {"directly_teaches_topic": False},
    ],
)
def test_hard_gemini_contract_rejects_off_topic_candidates(
    monkeypatch,
    hard_gate_override: dict[str, object],
) -> None:
    quote = (
        "An operating system schedules processes, manages virtual memory, and "
        "provides file system abstractions."
    )
    engine_out = _one_cue_selector_result(
        quote,
        score=1.0,
        **hard_gate_override,
    )
    monkeypatch.setattr(pipeline_module, "_run_clip", lambda *_args, **_kwargs: engine_out)

    _video_row, clips, _engine = _pipeline()._clip_and_filter(
        {**_video(), "literal_match": False, "_topic_terms": ["biology"]},
        "biology",
        "en",
    )

    assert clips == []


def test_topic_mention_elsewhere_cannot_rescue_an_unrelated_evidence_quote(
    monkeypatch,
) -> None:
    evidence_quote = (
        "An operating system schedules processes, manages virtual memory, and "
        "provides file system abstractions."
    )
    clip_text = (
        f"{evidence_quote} Biology studies living organisms, their cells, and "
        "their interactions."
    )
    engine_out = _one_cue_selector_result(
        evidence_quote,
        clip_text=clip_text,
        score=1.0,
        directly_teaches_topic=False,
    )
    monkeypatch.setattr(pipeline_module, "_run_clip", lambda *_args, **_kwargs: engine_out)

    _video_row, clips, _engine = _pipeline()._clip_and_filter(
        {**_video(), "literal_match": False, "_topic_terms": ["biology"]},
        "biology",
        "en",
    )

    assert clips == []


def test_grounded_paraphrase_does_not_require_semantic_inference(
    monkeypatch,
) -> None:
    quote = "Cells convert nutrient energy into ATP through enzyme controlled reactions."
    engine_out = _one_cue_selector_result(quote)
    monkeypatch.setattr(pipeline_module, "_run_clip", lambda *_args, **_kwargs: engine_out)
    context = GenerationContext("slow")

    _video_row, clips, _engine = _pipeline()._clip_and_filter(
        {**_video(), "literal_match": False, "_topic_terms": ["biology"]},
        "biology",
        "en",
        generation_context=context,
    )

    assert len(clips) == 1
    assert clips[0]["topic_evidence_terms"] == [quote]
    assert context.counters()["topic_rejections"] == 0
    assert context.usage_payload()["summary"]["rejection_reason_counts"] == {}


def test_exact_topic_lexical_proof_survives_without_semantic_inference(
    monkeypatch,
) -> None:
    quote = "Biology studies living organisms, their cells, and their interactions."
    engine_out = _one_cue_selector_result(quote)
    monkeypatch.setattr(pipeline_module, "_run_clip", lambda *_args, **_kwargs: engine_out)

    _video_row, clips, _engine = _pipeline()._clip_and_filter(
        {**_video(), "literal_match": False, "_topic_terms": ["biology"]},
        "biology",
        "en",
    )

    assert len(clips) == 1


def test_exact_comparison_component_survives_without_semantic_inference(
    monkeypatch,
) -> None:
    quote = "A sunk cost has already been incurred and cannot be recovered."
    engine_out = _one_cue_selector_result(quote)
    monkeypatch.setattr(pipeline_module, "_run_clip", lambda *_args, **_kwargs: engine_out)
    topic = "opportunity cost versus sunk cost"

    _video_row, clips, _engine = _pipeline()._clip_and_filter(
        {**_video(), "literal_match": False, "_topic_terms": [topic]},
        topic,
        "en",
    )

    assert len(clips) == 1
    assert clips[0]["topic_evidence_terms"] == [quote]


def test_expanded_search_terms_cannot_override_hard_exact_topic_contract(
    monkeypatch,
) -> None:
    quote = (
        "An operating system schedules processes, manages virtual memory, and "
        "provides file system abstractions."
    )
    engine_out = _one_cue_selector_result(
        quote,
        score=1.0,
        directly_teaches_topic=False,
    )
    monkeypatch.setattr(pipeline_module, "_run_clip", lambda *_args, **_kwargs: engine_out)

    _video_row, clips, _engine = _pipeline()._clip_and_filter(
        {
            **_video(),
            "literal_match": False,
            "_topic_terms": ["biology", "operating systems"],
        },
        "biology",
        "en",
    )

    assert clips == []


def test_task_topic_keeps_recognition_teaching_and_rejects_object_history(
    monkeypatch,
) -> None:
    transcript = {
        "source": "supadata",
        "native_mode": False,
        "artifact_key": "supadata-transcript:v2:carolingian-ligatures",
        "duration": 20.0,
        "segments": [
            {
                "cue_id": "history",
                "start": 0.0,
                "end": 10.0,
                "text": "Carolingian minuscule developed in medieval scriptoria under Charlemagne's reforms.",
            },
            {
                "cue_id": "recognition",
                "start": 10.0,
                "end": 20.0,
                "text": "Recognize a Caroline minuscule ligature by the joined letter strokes between adjacent letter forms.",
            },
        ],
    }
    clip_defaults = {
        "informativeness": 0.8,
        "topic_relevance": 0.8,
        "educational_importance": 0.8,
        "difficulty": 0.4,
        "boundary_confidence": 0.9,
        "self_contained": True,
        "is_standalone": True,
        "prerequisite_ids": [],
        "uncertainty": "low",
        "substantive": True,
        "factually_grounded": True,
    }
    engine_out = {
        "clips": [
            {
                **clip_defaults,
                "start": 0.0,
                "end": 10.0,
                "cue_ids": ["history"],
                "selection_candidate_id": "history",
                "directly_teaches_topic": False,
                "topic_evidence_quote": (
                    "Carolingian minuscule developed in medieval scriptoria under Charlemagne's reforms"
                ),
            },
            {
                **clip_defaults,
                "start": 10.0,
                "end": 20.0,
                "cue_ids": ["recognition"],
                "selection_candidate_id": "recognition",
                "directly_teaches_topic": True,
                "topic_evidence_quote": (
                    "Recognize a Caroline minuscule ligature by the joined letter strokes"
                ),
            },
        ],
        "transcript": transcript,
        "notes": "",
    }
    monkeypatch.setattr(pipeline_module, "_run_clip", lambda *_args, **_kwargs: engine_out)
    topic = "Carolingian minuscule ligature identification"
    video = {**_video(), "_topic_terms": [topic], "_knowledge_level": "beginner"}

    pipeline, _embedding = _pipeline_with_semantic(0.3)
    _video_row, clips, _engine = pipeline._clip_and_filter(video, topic, "en")

    assert [clip["selection_candidate_id"] for clip in clips] == [
        "dQw4w9WgXcQ::recognition"
    ]


@pytest.mark.parametrize(
    ("quality_axis", "value", "accepted"),
    [
        ("informativeness", 0.00, False),
        ("informativeness", 0.74, False),
        ("informativeness", 0.75, True),
        ("topic_relevance", 0.74, False),
        ("topic_relevance", 0.75, True),
        ("educational_importance", 0.00, False),
        ("educational_importance", 0.74, False),
        ("educational_importance", 0.75, True),
    ],
)
def test_each_quality_axis_has_a_hard_point_seven_five_gate(
    monkeypatch,
    quality_axis: str,
    value: float,
    accepted: bool,
) -> None:
    engine_out = {
        "clips": [_quality_clip(**{quality_axis: value})],
        "transcript": _transcript(),
        "notes": "",
    }
    monkeypatch.setattr(
        pipeline_module,
        "_run_clip",
        lambda *_args, **_kwargs: engine_out,
    )

    _, clips, _ = _pipeline()._clip_and_filter(
        _video(), "Intro to Python", "en"
    )

    assert bool(clips) is accepted


@pytest.mark.parametrize(
    ("knowledge_level", "expected_starts", "expected_deferred"),
    [
        ("beginner", [0.0, 10.0], [False, True]),
        ("advanced", [10.0, 0.0], [False, True]),
    ],
)
def test_candidate_plan_prioritizes_level_eligible_then_difficulty(
    monkeypatch,
    knowledge_level: str,
    expected_starts: list[float],
    expected_deferred: list[bool],
) -> None:
    transcript = _transcript()
    transcript["segments"][1] = {
        "cue_id": "advanced-python",
        "start": 10.0,
        "end": 20.0,
        "text": "Python generators yield values lazily and preserve suspended execution state.",
    }
    engine_out = {
        "clips": [
            _quality_clip(score=0.8, difficulty=0.1),
            _quality_clip(
                cue_id="advanced-python",
                start=10.0,
                end=20.0,
                quote=(
                    "Python generators yield values lazily and preserve suspended "
                    "execution state."
                ),
                score=0.9,
                difficulty=0.9,
            ),
        ],
        "transcript": transcript,
        "notes": "",
    }
    monkeypatch.setattr(pipeline_module, "_run_clip", lambda *_args, **_kwargs: engine_out)
    video = {**_video(), "_knowledge_level": knowledge_level}

    _, clips, _ = _pipeline()._clip_and_filter(video, "Intro to Python", "en")

    assert [clip["start"] for clip in clips] == expected_starts
    assert [
        bool(clip["search_context"]["deferred_level"])
        for clip in clips
    ] == expected_deferred


def test_all_deferred_source_streams_nearest_valid_level_immediately(
    monkeypatch,
) -> None:
    engine_out = {
        "clips": [
            _quality_clip(
                candidate_id="intermediate-python",
                score=0.9,
                difficulty=0.45,
            )
        ],
        "transcript": _transcript(),
        "notes": "",
    }
    monkeypatch.setattr(
        pipeline_module,
        "_discover",
        lambda *_args, **_kwargs: _discovery(),
    )
    monkeypatch.setattr(
        pipeline_module,
        "_run_clip",
        lambda *_args, **_kwargs: engine_out,
    )
    stored: list[dict] = []
    pipeline = _pipeline()

    def persist(*, clip, **_kwargs):
        stored.append(clip)
        return "nearest-level-reel", mock.sentinel.metadata

    monkeypatch.setattr(pipeline, "_persist_engine_clip", persist)
    emitted: list[str] = []
    context = GenerationContext("fast")

    reels, _ = pipeline.ingest_topic(
        topic="Intro to Python",
        material_id="material",
        concept_id="concept",
        generation_context=context,
        retrieval_profile="deep",
        knowledge_level="beginner",
        max_videos=1,
        max_reels=1,
        max_persisted_reels=1,
        on_reel_created=emitted.append,
    )

    assert reels == ["nearest-level-reel"]
    assert emitted == ["nearest-level-reel"]
    assert stored[0]["search_context"]["deferred_level"] is True
    assert stored[0]["search_context"]["surface_reason"] == "level_mismatch"
    assert stored[0]["search_context"]["surface_eligible"] is False
    assert context.counters()["stored_clips"] == 1
    assert context.counters()["deferred_clips"] == 1
    assert context.counters()["level_deferred_clips"] == 1
    assert context.counters()["persisted_clips"] == 1


def test_all_deferred_source_streams_all_valid_clips_by_difficulty_proximity(
    monkeypatch,
) -> None:
    transcript = _transcript()
    transcript["segments"].append({
        "cue_id": "advanced-python",
        "start": 20.0,
        "end": 30.0,
        "text": "Python metaclasses customize class construction through advanced hooks.",
    })
    transcript["duration"] = 30.0
    engine_out = {
        "clips": [
            _quality_clip(
                candidate_id="intermediate-python",
                score=0.9,
                difficulty=0.45,
            ),
            _quality_clip(
                cue_id="advanced-python",
                start=20.0,
                end=30.0,
                quote=transcript["segments"][-1]["text"],
                candidate_id="advanced-python",
                score=0.95,
                difficulty=0.85,
            ),
        ],
        "transcript": transcript,
        "notes": "",
    }
    monkeypatch.setattr(
        pipeline_module,
        "_discover",
        lambda *_args, **_kwargs: _discovery(),
    )
    monkeypatch.setattr(
        pipeline_module,
        "_run_clip",
        lambda *_args, **_kwargs: engine_out,
    )
    pipeline = _pipeline()

    def persist(*, clip, **_kwargs):
        return clip["selection_candidate_id"], mock.sentinel.metadata

    monkeypatch.setattr(pipeline, "_persist_engine_clip", persist)
    emitted: list[str] = []

    reels, _ = pipeline.ingest_topic(
        topic="Intro to Python",
        material_id="material",
        concept_id="concept",
        generation_context=GenerationContext("fast"),
        retrieval_profile="deep",
        knowledge_level="beginner",
        max_videos=1,
        max_reels=2,
        max_persisted_reels=2,
        on_reel_created=emitted.append,
    )

    assert reels == [
        "dQw4w9WgXcQ::intermediate-python",
        "dQw4w9WgXcQ::advanced-python",
    ]
    assert emitted == [
        "dQw4w9WgXcQ::intermediate-python",
        "dQw4w9WgXcQ::advanced-python",
    ]


def test_current_level_candidate_precedes_without_suppressing_other_levels(
    monkeypatch,
) -> None:
    transcript = _transcript()
    transcript["segments"].append({
        "cue_id": "advanced-python",
        "start": 20.0,
        "end": 30.0,
        "text": "Python metaclasses customize class construction through advanced hooks.",
    })
    transcript["duration"] = 30.0
    engine_out = {
        "clips": [
            _quality_clip(candidate_id="beginner-python", score=0.8, difficulty=0.2),
            _quality_clip(
                cue_id="advanced-python",
                start=20.0,
                end=30.0,
                quote=transcript["segments"][-1]["text"],
                candidate_id="advanced-python",
                score=0.95,
                difficulty=0.85,
            ),
        ],
        "transcript": transcript,
        "notes": "",
    }
    monkeypatch.setattr(
        pipeline_module,
        "_discover",
        lambda *_args, **_kwargs: _discovery(),
    )
    monkeypatch.setattr(
        pipeline_module,
        "_run_clip",
        lambda *_args, **_kwargs: engine_out,
    )
    stored: list[str] = []
    pipeline = _pipeline()

    def persist(*, clip, **_kwargs):
        candidate_id = str(clip["selection_candidate_id"]).split("::")[-1]
        stored.append(candidate_id)
        return f"reel-{candidate_id}", mock.sentinel.metadata

    monkeypatch.setattr(pipeline, "_persist_engine_clip", persist)
    emitted: list[str] = []

    reels, _ = pipeline.ingest_topic(
        topic="Intro to Python",
        material_id="material",
        concept_id="concept",
        generation_context=GenerationContext("fast"),
        retrieval_profile="deep",
        knowledge_level="beginner",
        max_videos=1,
        max_reels=2,
        max_persisted_reels=2,
        on_reel_created=emitted.append,
    )

    assert stored == ["beginner-python", "advanced-python"]
    assert reels == ["reel-beginner-python", "reel-advanced-python"]
    assert emitted == ["reel-beginner-python", "reel-advanced-python"]


def test_candidate_plan_prioritizes_primary_intent_only_within_difficulty_stage(
    monkeypatch,
) -> None:
    cues = [
        {
            "cue_id": "beginner-support",
            "start": 0.0,
            "end": 10.0,
            "text": "Python functions package reusable instructions for a program.",
        },
        {
            "cue_id": "beginner-primary",
            "start": 10.0,
            "end": 20.0,
            "text": "This worked Python example defines greet and then calls greet.",
        },
        {
            "cue_id": "advanced-primary",
            "start": 20.0,
            "end": 30.0,
            "text": "This advanced Python example composes decorators around a generator function.",
        },
    ]
    engine_out = {
        "clips": [
            _quality_clip(
                cue_id="beginner-support",
                start=0.0,
                end=10.0,
                quote=cues[0]["text"],
                candidate_id="beginner-support",
                score=0.99,
                difficulty=0.2,
                intent_role="supporting",
                intent_coverage=0.5,
                intent_evidence=[{
                    "constraint_id": "subject",
                    "evidence_quote": cues[0]["text"],
                }],
            ),
            _quality_clip(
                cue_id="beginner-primary",
                start=10.0,
                end=20.0,
                quote=cues[1]["text"],
                candidate_id="beginner-primary",
                score=0.80,
                difficulty=0.3,
                intent_role="primary",
                intent_coverage=1.0,
            ),
            _quality_clip(
                cue_id="advanced-primary",
                start=20.0,
                end=30.0,
                quote=cues[2]["text"],
                candidate_id="advanced-primary",
                score=1.0,
                difficulty=0.8,
                intent_role="primary",
                intent_coverage=1.0,
            ),
        ],
        "transcript": {
            "source": "supadata",
            "native_mode": False,
            "artifact_key": "supadata-transcript:v2:intent-stage-order",
            "duration": 30.0,
            "segments": cues,
        },
        "notes": "",
    }
    monkeypatch.setattr(
        pipeline_module,
        "_run_clip",
        lambda *_args, **_kwargs: engine_out,
    )

    _, clips, _ = _pipeline()._clip_and_filter(
        {**_video(), "_knowledge_level": "beginner"},
        "Python worked example",
        "en",
    )

    assert [clip["selection_candidate_id"] for clip in clips] == [
        "dQw4w9WgXcQ::beginner-primary",
        "dQw4w9WgXcQ::beginner-support",
        "dQw4w9WgXcQ::advanced-primary",
    ]
    assert clips[0]["search_context"]["intent_role"] == "primary"
    assert clips[1]["search_context"]["intent_coverage"] == 0.5


@pytest.mark.parametrize(
    ("difficulty", "matching_level"),
    [
        (0.33, "beginner"),
        (0.34, "intermediate"),
        (0.66, "intermediate"),
        (0.67, "advanced"),
    ],
)
def test_v3_ingestion_uses_exact_non_overlapping_difficulty_bins(
    monkeypatch,
    difficulty: float,
    matching_level: str,
) -> None:
    engine_out = {
        "clips": [_quality_clip(difficulty=difficulty)],
        "transcript": _transcript(),
        "notes": "",
    }
    monkeypatch.setattr(
        pipeline_module,
        "_run_clip",
        lambda *_args, **_kwargs: engine_out,
    )
    pipeline = _pipeline()

    for level in ("beginner", "intermediate", "advanced"):
        video = {**_video(), "_knowledge_level": level}
        _, clips, _ = pipeline._clip_and_filter(video, "Intro to Python", "en")
        assert bool(clips[0]["search_context"]["deferred_level"]) is (
            level != matching_level
        )


def test_selector_contract_uses_level_neutral_content_score(monkeypatch) -> None:
    engine_out = {
        "clips": [_quality_clip(
            informativeness=0.8,
            topic_relevance=0.85,
            educational_importance=0.9,
            boundary_confidence=0.85,
            chain_id="python-functions",
            chain_position=2,
            candidate_id="python-functions-2",
            difficulty=0.95,
        )],
        "transcript": _transcript(),
        "notes": "",
    }
    monkeypatch.setattr(pipeline_module, "_run_clip", lambda *_args, **_kwargs: engine_out)
    pipeline = _pipeline()

    scores = []
    for level in ("beginner", "advanced"):
        video = {**_video(), "_knowledge_level": level}
        _, clips, _ = pipeline._clip_and_filter(video, "Intro to Python", "en")
        scores.append(clips[0]["score"])
        context = clips[0]["search_context"]
        assert context["selection_contract_version"] == "quality_silence_v32"
        assert context["boundary_confidence"] == 0.85
        assert context["is_standalone"] is True
        assert context["chain_id"] == "dQw4w9WgXcQ::python-functions"
        assert context["chain_position"] == 2
        assert context["selection_candidate_id"] == "dQw4w9WgXcQ::python-functions-2"
        assert context["prerequisite_ids"] == []
        assert context["selection_caption_cues"] == [
            {
                "cue_id": "python",
                "start": 0.0,
                "end": 10.0,
                "text": "Python functions package reusable instructions.",
                "lang": "",
            }
        ]

    assert scores == pytest.approx([0.85, 0.85])


def test_filter_rejects_non_standalone_dependent_instead_of_repairing_it(
    monkeypatch,
) -> None:
    engine_out = {
        "clips": [
            _quality_clip(candidate_id="root", score=0.8),
            _quality_clip(
                candidate_id="dependent",
                score=0.99,
                self_contained=False,
                is_standalone=False,
                prerequisite_ids=["root"],
            ),
        ],
        "transcript": _transcript(),
        "notes": "",
    }
    monkeypatch.setattr(pipeline_module, "_run_clip", lambda *_args, **_kwargs: engine_out)

    _, clips, _ = _pipeline()._clip_and_filter(_video(), "Intro to Python", "en")

    assert [clip["selection_candidate_id"] for clip in clips] == [
        "dQw4w9WgXcQ::root",
    ]


def test_multiple_flash_source_results_never_trigger_pro_fallback(
    monkeypatch,
) -> None:
    first = _video()
    second = {
        **_video(),
        "id": "9bZkp7q19f0",
        "url": "https://www.youtube.com/watch?v=9bZkp7q19f0",
    }
    discovery = {**_discovery(), "videos": [first, second]}
    monkeypatch.setattr(pipeline_module, "_discover", lambda *_args, **_kwargs: discovery)
    pipeline = _pipeline()

    def clip_and_filter(video, *_args, engine_out_override=None, **_kwargs):
        if engine_out_override is not None:
            return video, [
                {
                    "start": 0.0,
                    "end": 10.0,
                    "cue_ids": ["python"],
                    "score": 0.9,
                    "selection_candidate_id": "root",
                    "prerequisite_ids": [],
                },
                {
                    "start": 10.0,
                    "end": 20.0,
                    "cue_ids": ["garden"],
                    "score": 0.9,
                    "selection_candidate_id": "dependent",
                    "prerequisite_ids": ["root"],
                },
                {
                    "start": 0.0,
                    "end": 20.0,
                    "cue_ids": ["python", "garden"],
                    "score": 0.9,
                    "selection_candidate_id": "orphan",
                    "prerequisite_ids": ["missing"],
                },
            ], engine_out_override
        return (
            video,
            [{
                "start": 0.0,
                "end": 10.0,
                "cue_ids": ["python"],
                "score": 0.8,
                "selection_candidate_id": "root",
                "prerequisite_ids": [],
            }],
            {"transcript": _transcript(), "clips": []},
        )

    monkeypatch.setattr(pipeline, "_clip_and_filter", clip_and_filter)
    pro_fallback = mock.Mock(return_value={
        "transcript": _transcript(),
        "clips": [{"start": 10.0, "end": 20.0, "cue_ids": ["garden"]}],
        "notes": "fallback",
    })
    monkeypatch.setattr(pipeline_module.clip_engine_run, "pro_boundary_fallback", pro_fallback)
    monkeypatch.setattr(
        pipeline,
        "_persist_engine_clip",
        mock.Mock(side_effect=lambda **kwargs: (
            f"{kwargs['v']['id']}:{kwargs['clip']['start']}",
            mock.sentinel.metadata,
        )),
    )

    reels, _ = pipeline.ingest_topic(
        topic="Intro to Python",
        material_id="material",
        concept_id="concept",
        max_videos=2,
        max_reels=3,
        generation_context=GenerationContext("slow"),
    )

    assert len(reels) == 2
    pro_fallback.assert_not_called()


def test_acoustic_unavailable_uses_context_without_pro_repair(
    monkeypatch,
) -> None:
    first = _video()
    second = {
        **_video(),
        "id": "9bZkp7q19f0",
        "url": "https://www.youtube.com/watch?v=9bZkp7q19f0",
    }
    monkeypatch.setattr(
        pipeline_module,
        "_discover",
        lambda *_args, **_kwargs: {
            **_discovery(),
            "videos": [first, second],
        },
    )
    pipeline = _pipeline()

    def clip_and_filter(video, *_args, engine_out_override=None, **_kwargs):
        clips = [{
            "start": 0.0,
            "end": 10.0,
            "cue_ids": ["python"],
            "score": 0.9,
            "selection_candidate_id": f"{video['id']}::root",
            "prerequisite_ids": [],
            "search_context": {"surface_eligible": True},
        }]
        if engine_out_override is None:
            clips.append({
                "start": 10.0,
                "end": 20.0,
                "cue_ids": ["garden"],
                "score": 0.8,
                "selection_candidate_id": f"{video['id']}::second",
                "prerequisite_ids": [],
                "search_context": {"surface_eligible": True},
            })
        return video, clips, (
            engine_out_override
            or {"transcript": _transcript(), "clips": []}
        )

    monkeypatch.setattr(pipeline, "_clip_and_filter", clip_and_filter)
    pro_fallback = mock.Mock(return_value={
        "transcript": _transcript(),
        "clips": [{"start": 0.0, "end": 10.0, "cue_ids": ["python"]}],
        "notes": "fallback",
    })
    monkeypatch.setattr(
        pipeline_module.clip_engine_run,
        "pro_boundary_fallback",
        pro_fallback,
    )
    monkeypatch.setattr(
        pipeline_module.clip_engine_silence,
        "prepare_audio_source",
        mock.Mock(return_value=mock.sentinel.prepared_audio),
    )
    monkeypatch.setattr(
        pipeline_module.clip_engine_silence,
        "verify_acoustic_boundaries",
        mock.Mock(return_value=mock.Mock(
            verified=False,
            start_sec=0.0,
            end_sec=10.0,
            diagnostics={"reason": "start_silence_not_found"},
        )),
    )
    monkeypatch.setattr(
        pipeline,
        "_persist_engine_clip",
        mock.Mock(side_effect=lambda **kwargs: (
            f"{kwargs['v']['id']}:{kwargs['clip']['start']}",
            mock.sentinel.metadata,
        )),
    )

    reels, _ = pipeline.ingest_topic(
        topic="Intro to Python",
        material_id="material",
        concept_id="concept",
        max_videos=2,
        max_reels=3,
        generation_context=GenerationContext(
            "slow",
            require_acoustic_boundaries=True,
        ),
    )

    assert len(reels) == 3
    pro_fallback.assert_not_called()


def test_generation_skips_pro_repair_and_synchronous_enrichment(monkeypatch) -> None:
    video = _video()
    discovery_limits: list[int] = []

    def discover(*_args, **kwargs):
        discovery_limits.append(int(kwargs["limit"]))
        return {**_discovery(), "videos": [video]}

    monkeypatch.setattr(
        pipeline_module,
        "_discover",
        discover,
    )
    pipeline = _pipeline()
    monkeypatch.setattr(
        pipeline,
        "_clip_and_filter",
        lambda selected, *_args, **_kwargs: (
            selected,
            [{"start": 0.0, "end": 10.0, "cue_ids": ["python"], "score": 0.8}],
            {"transcript": _transcript(), "clips": []},
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_persist_engine_clip",
        mock.Mock(return_value=("v2-reel", mock.sentinel.metadata)),
    )
    pro_fallback = mock.Mock()
    monkeypatch.setattr(
        pipeline_module.clip_engine_run,
        "pro_boundary_fallback",
        pro_fallback,
    )
    context = GenerationContext("slow")
    analyzed: set[str] = set()

    reels, _ = pipeline.ingest_topic(
        topic="Intro to Python",
        material_id="material",
        concept_id="concept",
        max_videos=3,
        max_reels=2,
        generation_context=context,
        retrieval_profile="deep",
        analyzed_video_ids=analyzed,
    )

    assert reels == ["v2-reel"]
    assert len(discovery_limits) == 1
    assert analyzed == {video["id"]}
    pro_fallback.assert_not_called()
    assert not hasattr(pipeline_module, "live_gemini_segment")
    assert context.counters()["boundary_repairs"] == 0
    assert context.counters()["pro_fallbacks"] == 0


def test_clip_call_uses_one_low_thinking_flash_selector_and_ignores_duration(
    monkeypatch,
) -> None:
    captured: dict = {}

    def clip(_url, **kwargs):
        captured.update(kwargs["settings"])
        return {"clips": [], "transcript": {"segments": []}}

    clip_mock = mock.Mock(side_effect=clip)
    monkeypatch.setattr(pipeline_module.clip_engine_run, "clip", clip_mock)

    pipeline_module._run_clip(
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        topic="Intro to Python",
        language="en",
        should_cancel=None,
        retrieval_profile="deep",
        target_clip_duration_sec=40,
        target_clip_duration_min_sec=10,
        target_clip_duration_max_sec=55,
    )

    fallback_gate = captured.get("_segment_pro_fallback_gate")
    assert callable(fallback_gate)
    assert fallback_gate(accepted_count=0, video_id="dQw4w9WgXcQ") is False
    assert captured["_segment_routing_mode"] == "flash_only"
    assert captured["_segment_thinking_level"] == "low"
    assert "_segment_target_sec" not in captured
    assert "_segment_target_min_sec" not in captured
    assert "_segment_target_max_sec" not in captured
    assert clip_mock.call_count == 1


@pytest.mark.parametrize("mode", ["fast", "slow"])
def test_fast_and_slow_clip_calls_use_identical_quality_routing(
    monkeypatch, mode: str
) -> None:
    captured: dict = {}

    def clip(_url, **kwargs):
        captured.update(kwargs["settings"])
        return {"clips": [], "transcript": {"segments": []}}

    monkeypatch.setattr(pipeline_module.clip_engine_run, "clip", clip)

    pipeline_module._run_clip(
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        topic="Intro to Python",
        language="en",
        should_cancel=None,
        generation_context=GenerationContext(mode),
        retrieval_profile="deep",
    )

    assert captured["_segment_thinking_level"] == "low"
    assert captured["_segment_routing_mode"] == "flash_only"
    assert captured["_segment_pro_fallback_gate"](
        accepted_count=0,
        video_id="dQw4w9WgXcQ",
    ) is False


def test_low_yield_source_does_not_trigger_pro_or_extra_backfill(
    monkeypatch,
) -> None:
    first = _video()
    second = {
        **_video(),
        "id": "9bZkp7q19f0",
        "url": "https://www.youtube.com/watch?v=9bZkp7q19f0",
    }
    discovery = {**_discovery(), "videos": [first, second]}
    monkeypatch.setattr(pipeline_module, "_discover", lambda *_args, **_kwargs: discovery)
    pipeline = _pipeline()

    def clip_and_filter(video, *_args, engine_out_override=None, **_kwargs):
        if engine_out_override is not None:
            return video, [{
                "start": 10.0,
                "end": 20.0,
                "cue_ids": ["garden"],
                "score": 0.9,
            }], engine_out_override
        clips = (
            [
                {"start": 0.0, "end": 10.0, "cue_ids": ["python"], "score": 0.8},
                {"start": 10.0, "end": 20.0, "cue_ids": ["garden"], "score": 0.8},
            ]
            if video["id"] == first["id"]
            else []
        )
        return video, clips, {"transcript": _transcript(), "clips": []}

    monkeypatch.setattr(pipeline, "_clip_and_filter", clip_and_filter)
    pro_fallback = mock.Mock(return_value={
        "transcript": _transcript(),
        "clips": [{"start": 10.0, "end": 20.0, "cue_ids": ["garden"]}],
        "notes": "fallback",
    })
    monkeypatch.setattr(pipeline_module.clip_engine_run, "pro_boundary_fallback", pro_fallback)
    monkeypatch.setattr(
        pipeline,
        "_persist_engine_clip",
        mock.Mock(side_effect=lambda **kwargs: (
            f"{kwargs['v']['id']}:{kwargs['clip']['start']}",
            mock.sentinel.metadata,
        )),
    )

    reels, _ = pipeline.ingest_topic(
        topic="Intro to Python",
        material_id="material",
        concept_id="concept",
        max_videos=2,
        max_reels=3,
        generation_context=GenerationContext("slow"),
    )

    assert len(reels) == 2
    pro_fallback.assert_not_called()


def test_unusable_transcripts_never_trigger_pro_fallback(
    monkeypatch,
) -> None:
    first = _video()
    second = {
        **_video(),
        "id": "9bZkp7q19f0",
        "url": "https://www.youtube.com/watch?v=9bZkp7q19f0",
    }
    discovery = {**_discovery(), "videos": [first, second]}
    monkeypatch.setattr(pipeline_module, "_discover", lambda *_args, **_kwargs: discovery)
    pipeline = _pipeline()
    invalid_transcript = {"source": "supadata", "segments": _transcript()["segments"]}
    monkeypatch.setattr(
        pipeline,
        "_clip_and_filter",
        lambda video, *_args, **_kwargs: (
            video,
            [],
            {"transcript": invalid_transcript, "clips": []},
        ),
    )
    pro_fallback = mock.Mock()
    monkeypatch.setattr(pipeline_module.clip_engine_run, "pro_boundary_fallback", pro_fallback)
    context = GenerationContext("slow")

    reels, _ = pipeline.ingest_topic(
        topic="Intro to Python",
        material_id="material",
        concept_id="concept",
        max_videos=2,
        max_reels=3,
        generation_context=context,
    )

    assert reels == []
    pro_fallback.assert_not_called()
    assert context.counters()["pro_fallbacks"] == 0


def test_ingest_topic_counts_shared_clip_fetch_timeout_separately(monkeypatch) -> None:
    monkeypatch.setattr(pipeline_module, "_discover", lambda *_args, **_kwargs: _discovery())
    monkeypatch.setattr(pipeline_module, "INGEST_TOPIC_VIDEO_TIMEOUT_SEC", 0.001)
    pipeline = _pipeline()

    def slow_clip_and_filter(*_args, **_kwargs):
        time.sleep(0.05)
        return _video(), [], {"clips": [], "transcript": _transcript(), "notes": ""}

    monkeypatch.setattr(pipeline, "_clip_and_filter", slow_clip_and_filter)
    context = GenerationContext("slow")

    reels, _ = pipeline.ingest_topic(
        topic="Intro to Python",
        material_id="material",
        concept_id="concept",
        generation_context=context,
        max_videos=1,
    )

    assert reels == []
    assert context.counters()["clip_fetch_timeouts"] == 1
    assert context.counters()["transcript_timeouts"] == 0


@pytest.mark.parametrize(
    ("mode", "expected_sources"),
    [("fast", 2), ("slow", 3)],
)
def test_mode_source_budgets_are_analyzed_concurrently_without_backfill(
    monkeypatch, mode: str, expected_sources: int
) -> None:
    videos = [
        {**_video(), "id": f"video-{index}", "url": f"https://youtu.be/video-{index}"}
        for index in range(5)
    ]
    discovery_calls = 0
    analyzed: list[str] = []
    concurrency_lock = threading.Lock()
    all_started = threading.Event()
    active = 0
    max_active = 0

    def discover(*_args, **kwargs):
        nonlocal discovery_calls
        del kwargs
        discovery_calls += 1
        return {
            "corrected": "Intro to Python",
            "videos": videos,
            "credits_used": 0,
            "warning": None,
        }

    def clip_and_filter(video, *_args, **_kwargs):
        nonlocal active, max_active
        with concurrency_lock:
            analyzed.append(video["id"])
            active += 1
            max_active = max(max_active, active)
            if active == expected_sources:
                all_started.set()
        assert all_started.wait(timeout=1.0)
        time.sleep(0.02)
        with concurrency_lock:
            active -= 1
        return video, [], {"transcript": _transcript()}

    pipeline = _pipeline()
    monkeypatch.setattr(pipeline_module, "_discover", discover)
    monkeypatch.setattr(pipeline, "_clip_and_filter", clip_and_filter)

    pipeline.ingest_topic(
        topic="Intro to Python",
        material_id="material",
        concept_id="concept",
        generation_context=GenerationContext(mode),
        max_videos=5,
    )

    assert discovery_calls == 1
    assert len(analyzed) == expected_sources
    assert max_active == expected_sources


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


def test_selector_transport_failure_preserves_transcript_and_provider_counters(
    monkeypatch,
) -> None:
    telemetry = gemini_client_module.GeminiCallTelemetry(
        model="gemini-3.5-flash",
        operation="flash_boundary_selector",
        prompt_version="quality_silence_v12",
        thinking_level="low",
        latency_ms=10.0,
        retries=1,
        finish_reason=None,
        prompt_tokens=None,
        candidate_tokens=None,
        thought_tokens=None,
        total_tokens=None,
        provider_error_type="ServerError",
        provider_status_code=503,
        retryable=True,
        error_history=({
            "provider_error_type": "ServerError",
            "provider_status_code": 503,
            "retryable": True,
        },),
    )

    def fail_selector(*_args, **_kwargs):
        raise gemini_client_module.GeminiTransportError(
            "private provider response",
            telemetry,
        )

    monkeypatch.setattr(pipeline_module, "_discover", lambda *_args, **_kwargs: _discovery())
    monkeypatch.setattr(
        pipeline_module.clip_engine_run,
        "_transcribe",
        lambda *_args, **_kwargs: _transcript(),
    )
    monkeypatch.setattr(
        pipeline_module.clip_engine_run.segment_cache,
        "cache_enabled",
        lambda: False,
    )
    monkeypatch.setattr(gemini_client_module, "generate_json_v3", fail_selector)
    monkeypatch.setattr(
        pipeline_module.clip_engine_silence,
        "prepare_audio_source",
        lambda *_args, **_kwargs: None,
    )
    context = GenerationContext("fast")

    with pytest.raises(ProviderTransientError):
        _pipeline().ingest_topic(
            topic="Intro to Python",
            material_id="material",
            concept_id="concept",
            generation_context=context,
            max_videos=1,
        )

    counters = context.counters()
    assert counters["usable_transcripts"] == 1
    assert counters["transcript_failures"] == 0
    assert counters["provider_failures"] == 1
    assert context.usage()[0]["metadata"]["error_history"] == telemetry.error_history


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
    mixed = main._generation_exhaustion_message(
        {
            "discovered_videos": 3,
            "usable_transcripts": 1,
            "clip_fetch_timeouts": 2,
            "topic_rejections": 1,
        }
    )

    assert "discovered" in search
    assert "usable timestamped transcript" in transcript
    assert "deadline" in deadline
    assert "content quality" in quality
    assert "topic and quality" in topic
    assert "did not finish before" in mixed
    assert "topic and quality" in mixed
    assert all(
        "native caption" not in message.casefold()
        for message in (search, transcript, deadline, quality, topic, mixed)
    )


def test_generation_status_exposes_counters_in_backward_compatible_error_detail() -> None:
    counters = {
        "discovered_videos": 3,
        "usable_transcripts": 1,
        "transcript_failures": 2,
        "transcript_timeouts": 0,
        "clip_fetch_timeouts": 0,
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
