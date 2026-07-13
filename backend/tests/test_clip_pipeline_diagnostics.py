from __future__ import annotations

import asyncio
import threading
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
    """A complete quality_silence_v3 selector result grounded to one cue."""
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
        return mock.Mock(
            verified=True,
            start_sec=start_sec,
            end_sec=end_sec,
            diagnostics={
                "threshold_dbfs": -38.0,
                "start_quiet": [start_sec, start_sec + 0.2],
                "end_quiet": [end_sec - 0.2, end_sec + 0.1],
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
    } == {(0.0, 30.0)}
    assert context.counters()["stored_clips"] == 3
    assert context.counters()["deferred_clips"] == 0
    assert context.counters()["persisted_clips"] == 3


def test_final_caption_clip_searches_to_true_media_end_and_keeps_verified_quiet_pad(
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
            end_sec=10.2,
            diagnostics={
                "threshold_dbfs": -38.0,
                "start_quiet": [0.0, 0.2],
                "end_quiet": [10.0, 10.3],
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

    assert reels == ["verified-tail"]
    assert persisted[0]["end"] == 10.2
    assert verify.call_args.kwargs["search_end_limit_sec"] == 12.0
    assert persisted[0]["search_context"]["boundary_status"] == "verified"
    assert persisted[0]["search_context"]["boundary_diagnostics"]["acoustic"][
        "threshold_dbfs"
    ] == -38.0


@pytest.mark.parametrize(
    ("verification", "expected_reason"),
    [
        (
            {
                "verified": False,
                "start_sec": 0.0,
                "end_sec": 10.0,
                "diagnostics": {
                    "stage": "analyze",
                    "reason": "start_silence_not_found",
                },
            },
            "start_silence_not_found",
        ),
        (
            {
                "verified": True,
                "start_sec": 0.0,
                "end_sec": 9.5,
                "diagnostics": {},
            },
            "acoustic_boundary_outside_source_or_required_speech",
        ),
    ],
)
def test_acoustic_failure_is_stored_but_never_emitted(
    monkeypatch,
    verification: dict,
    expected_reason: str,
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

    assert reels == []
    assert emitted == []
    assert len(stored) == 1
    boundary = stored[0]["search_context"]
    assert boundary["surface_eligible"] is False
    assert boundary["boundary_status"] == "unavailable"
    assert boundary["surface_reason"] == expected_reason
    assert context.counters()["stored_clips"] == 1
    assert context.counters()["deferred_clips"] == 1
    assert context.counters()["persisted_clips"] == 0
    assert context.usage_payload()["summary"]["rejection_reason_counts"] == {
        (
            "acoustic:"
            f"{verification['diagnostics'].get('stage', 'verify')}:"
            f"{expected_reason}"
        ): 1,
    }


def test_generation_count_excludes_all_explicitly_deferred_boundary_rows(
    monkeypatch,
) -> None:
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
            {"search_context_json": '[]'},
        ],
    )

    # Count only current strict diagnostics; missing, adaptive, noisier, and
    # non-object metadata never count as ready v2 inventory.
    assert main._count_generation_reels(object(), "generation") == 1


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
        "search_context_json": (
                '{"surface_eligible": true, "boundary_status": "verified", '
                '"boundary_diagnostics": {"acoustic_verified": true, '
                '"acoustic": {"threshold_dbfs": -38}}}'
            ),
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

    _video_row, clips, _engine = _pipeline()._clip_and_filter(
        video, "biology", "en"
    )

    assert [clip["selection_candidate_id"] for clip in clips] == [
        "dQw4w9WgXcQ::cell-energy"
    ]
    assert clips[0]["score"] == 0.8
    assert clips[0]["search_context"]["topic_evidence_quote"].startswith("Cells convert")
    assert clips[0]["search_context"]["factually_grounded"] is True


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

    _video_row, clips, _engine = _pipeline()._clip_and_filter(video, topic, "en")

    assert [clip["selection_candidate_id"] for clip in clips] == [
        "dQw4w9WgXcQ::recognition"
    ]


@pytest.mark.parametrize(
    ("quality_axis", "value", "accepted"),
    [
        ("informativeness", 0.00, True),
        ("informativeness", 0.75, True),
        ("topic_relevance", 0.74, False),
        ("topic_relevance", 0.75, True),
        ("educational_importance", 0.00, True),
        ("educational_importance", 0.75, True),
    ],
)
def test_only_topic_relevance_has_a_hard_point_seven_five_gate(
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
    engine_out = {
        "clips": [
            _quality_clip(score=0.8, difficulty=0.1),
            _quality_clip(
                cue_id="garden",
                start=10.0,
                end=20.0,
                quote="Garden soil needs regular watering.",
                score=0.9,
                difficulty=0.9,
            ),
        ],
        "transcript": _transcript(),
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
        assert context["selection_contract_version"] == "quality_silence_v3"
        assert context["boundary_confidence"] == 0.85
        assert context["is_standalone"] is True
        assert context["chain_id"] == "dQw4w9WgXcQ::python-functions"
        assert context["chain_position"] == 2
        assert context["selection_candidate_id"] == "dQw4w9WgXcQ::python-functions-2"
        assert context["prerequisite_ids"] == []

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


def test_acoustic_failures_fail_closed_without_pro_repair(
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

    assert reels == []
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
