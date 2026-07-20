from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest

from backend import gemini_client
from backend.app.clip_engine import config
from backend.app.clip_engine.errors import CancellationError
from backend.app.services import lesson_ordering


@pytest.fixture(autouse=True)
def _isolate_persistent_cache(monkeypatch) -> None:
    monkeypatch.setattr(
        lesson_ordering,
        "_read_cached_lesson_order",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        lesson_ordering,
        "_write_cached_lesson_order",
        lambda *_args, **_kwargs: None,
    )


def _reel(
    reel_id: str,
    *,
    video_id: str,
    start: float,
    concept: str,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "reel_id": reel_id,
        "video_id": video_id,
        "video_url": f"https://www.youtube.com/watch?v={video_id}",
        "t_start": start,
        "t_end": start + 20.0,
        "concept_title": concept,
        "video_title": f"{concept} lesson",
        "ai_summary": f"Explains {concept}",
        "takeaways": [concept],
        "transcript_snippet": f"Here is {concept}.",
        "difficulty": 0.3,
        **extra,
    }


def _generation_result(
    ordered_ids: list[str],
    checkpoint_ids: list[str] | None = None,
) -> gemini_client.GenerationResult:
    return gemini_client.GenerationResult(
        text=json.dumps(
            {
                "ordered_reel_ids": ordered_ids,
                "assessment_checkpoint_reel_ids": checkpoint_ids or [],
            }
        ),
        telemetry=gemini_client.GeminiCallTelemetry(
            model=config.LESSON_ORDER_MODEL,
            operation="ordering",
            prompt_version=lesson_ordering.LESSON_ORDER_PROMPT_VERSION,
            thinking_level="disabled",
            latency_ms=4.0,
            retries=0,
            finish_reason="STOP",
            prompt_tokens=120,
            candidate_tokens=20,
            thought_tokens=0,
            total_tokens=140,
        ),
    )


def test_orders_every_clip_and_returns_organizer_checkpoints(monkeypatch) -> None:
    reels = [
        _reel("worked", video_id="worked-video", start=30, concept="worked example"),
        _reel("intro", video_id="intro-video", start=0, concept="introduction"),
        _reel("core", video_id="core-video", start=10, concept="core definition"),
    ]
    captured: dict[str, str] = {}

    def fake_generate(system_prompt, user_prompt, *, should_cancel, dispatch_state):
        captured["system"] = system_prompt
        captured["user"] = user_prompt
        assert should_cancel is None
        return _generation_result(
            ["intro", "core", "worked"],
            ["worked"],
        )

    monkeypatch.setattr(lesson_ordering, "_generate_lesson_order", fake_generate)

    result = lesson_ordering.order_lesson_batch(
        reels,
        topic="gradient descent",
        learner_level="beginner",
    )

    assert result.ordered_reel_ids == ["intro", "core", "worked"]
    assert result.reels == [reels[1], reels[2], reels[0]]
    assert result.assessment_checkpoint_reel_ids == ["worked"]
    assert result.degraded is False
    assert "short batch may have none" in captured["system"]
    assert "gradient descent" in captured["user"]
    assert "beginner" in captured["user"]
    assert "assessment_checkpoint_reel_ids" in captured["user"]


def test_organizer_may_omit_a_mastered_concept(monkeypatch) -> None:
    reels = [
        _reel(
            "mastered-repeat",
            video_id="a",
            start=0,
            concept="force definition",
            concept_id="force",
        ),
        _reel(
            "net-force",
            video_id="b",
            start=0,
            concept="net force",
            concept_id="net-force",
        ),
        _reel(
            "worked",
            video_id="c",
            start=0,
            concept="worked example",
            concept_id="worked-example",
        ),
    ]
    captured: dict[str, str] = {}

    def fake_generate(system_prompt, user_prompt, **_kwargs):
        captured["system"] = system_prompt
        captured["user"] = user_prompt
        return _generation_result(["net-force", "worked"], ["worked"])

    monkeypatch.setattr(lesson_ordering, "_generate_lesson_order", fake_generate)

    result = lesson_ordering.order_lesson_batch(
        reels,
        topic="Newton's second law",
        concept_signals={
            "force": {
                "helpful": 2,
                "confusing": 0,
                "adjustment": 0.08,
            }
        },
    )

    assert result.ordered_reel_ids == ["net-force", "worked"]
    assert result.reels == [reels[1], reels[2]]
    assert result.degraded is False
    assert "may omit" in captured["system"]
    assert '"concept_id":"force"' in captured["user"]
    assert '"helpful":2.0' in captured["user"]
    assert '"adjustment":0.08' in captured["user"]


def test_organizer_subset_cannot_orphan_a_declared_prerequisite(monkeypatch) -> None:
    reels = [
        _reel(
            "definition",
            video_id="a",
            start=0,
            concept="definition",
            selection_candidate_id="candidate-definition",
        ),
        _reel(
            "example",
            video_id="b",
            start=0,
            concept="example",
            prerequisite_ids=["candidate-definition"],
        ),
    ]
    monkeypatch.setattr(
        lesson_ordering,
        "_generate_lesson_order",
        lambda *args, **kwargs: _generation_result(["example"]),
    )

    result = lesson_ordering.order_lesson_batch(reels, topic="topic")

    assert result.reels == reels
    assert result.degraded is True
    assert result.fallback_reason == "invalid_model_order"


def test_organizer_subset_cannot_skip_an_earlier_chain_member(monkeypatch) -> None:
    reels = [
        _reel(
            "chain-one",
            video_id="a",
            start=0,
            concept="setup",
            chain_id="derivation",
            chain_position=1,
        ),
        _reel(
            "chain-two",
            video_id="a",
            start=20,
            concept="result",
            chain_id="derivation",
            chain_position=2,
        ),
        _reel("independent", video_id="b", start=0, concept="recap"),
    ]
    monkeypatch.setattr(
        lesson_ordering,
        "_generate_lesson_order",
        lambda *args, **kwargs: _generation_result(["chain-two", "independent"]),
    )

    result = lesson_ordering.order_lesson_batch(reels, topic="topic")

    assert result.reels == reels
    assert result.degraded is True


def test_explicit_empty_checkpoint_list_is_authoritative(monkeypatch) -> None:
    reels = [
        _reel("intro", video_id="a", start=0, concept="intro"),
        _reel("core", video_id="b", start=0, concept="core"),
    ]
    monkeypatch.setattr(
        lesson_ordering,
        "_generate_lesson_order",
        lambda *args, **kwargs: _generation_result(["intro", "core"], []),
    )

    result = lesson_ordering.order_lesson_batch(reels, topic="topic")

    assert result.degraded is False
    assert result.assessment_checkpoint_reel_ids == []


def test_single_reel_still_asks_organizer_to_choose_checkpoint(monkeypatch) -> None:
    reel = _reel("only", video_id="a", start=0, concept="core")
    calls = 0

    def fake_generate(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _generation_result(["only"], ["only"])

    monkeypatch.setattr(lesson_ordering, "_generate_lesson_order", fake_generate)

    result = lesson_ordering.order_lesson_batch([reel], topic="topic")

    assert calls == 1
    assert result.reels == [reel]
    assert result.assessment_checkpoint_reel_ids == ["only"]
    assert result.degraded is False


@pytest.mark.parametrize(
    "checkpoint_ids",
    [["unknown"], ["core", "core"], ["core", "intro"]],
)
def test_invalid_checkpoint_plan_degrades_atomically(monkeypatch, checkpoint_ids) -> None:
    reels = [
        _reel("intro", video_id="a", start=0, concept="intro"),
        _reel("core", video_id="b", start=0, concept="core"),
    ]
    monkeypatch.setattr(
        lesson_ordering,
        "_generate_lesson_order",
        lambda *args, **kwargs: _generation_result(
            ["intro", "core"], checkpoint_ids
        ),
    )

    result = lesson_ordering.order_lesson_batch(reels, topic="topic")

    assert result.reels == reels
    assert result.degraded is True
    assert result.assessment_checkpoint_reel_ids is None


def test_same_source_chronology_cannot_be_reversed(monkeypatch) -> None:
    reels = [
        _reel("later", video_id="same", start=40, concept="example"),
        _reel("earlier", video_id="same", start=5, concept="definition"),
    ]
    monkeypatch.setattr(
        lesson_ordering,
        "_generate_lesson_order",
        lambda *args, **kwargs: _generation_result(["later", "earlier"]),
    )

    result = lesson_ordering.order_lesson_batch(reels, topic="topic")

    assert result.reels == reels
    assert result.degraded is True
    assert result.assessment_checkpoint_reel_ids is None


def test_invalid_permutation_and_dependency_order_fall_back_without_dropping_clips(
    monkeypatch,
) -> None:
    reels = [
        _reel(
            "definition",
            video_id="a",
            start=0,
            concept="definition",
            selection_candidate_id="candidate-definition",
        ),
        _reel(
            "example",
            video_id="b",
            start=0,
            concept="example",
            prerequisite_ids=["candidate-definition"],
        ),
    ]
    monkeypatch.setattr(
        lesson_ordering,
        "_generate_lesson_order",
        lambda *args, **kwargs: _generation_result(["example", "definition"]),
    )
    dependency_result = lesson_ordering.order_lesson_batch(reels, topic="topic")
    assert dependency_result.reels == reels
    assert dependency_result.degraded is True

    monkeypatch.setattr(
        lesson_ordering,
        "_generate_lesson_order",
        lambda *args, **kwargs: _generation_result(["definition", "definition"]),
    )
    permutation_result = lesson_ordering.order_lesson_batch(reels, topic="topic")
    assert permutation_result.reels == reels
    assert permutation_result.ordered_reel_ids == ["definition", "example"]
    assert permutation_result.degraded is True


def test_provider_failure_degrades_without_dropping_clips(monkeypatch) -> None:
    reels = [
        _reel("one", video_id="a", start=0, concept="one"),
        _reel("two", video_id="b", start=0, concept="two"),
    ]
    monkeypatch.setattr(
        lesson_ordering,
        "_generate_lesson_order",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )

    result = lesson_ordering.order_lesson_batch(reels, topic="topic")

    assert result.reels == reels
    assert result.degraded is True
    assert result.assessment_checkpoint_reel_ids is None


def test_generation_context_reserves_and_records_ordering(monkeypatch) -> None:
    reels = [
        _reel("one", video_id="a", start=0, concept="one"),
        _reel("two", video_id="b", start=0, concept="two"),
    ]

    class Context:
        def __init__(self) -> None:
            self.reservations: list[dict[str, Any]] = []
            self.records: list[dict[str, Any]] = []

        def reserve_gemini_call(self, **kwargs):
            self.reservations.append(kwargs)
            return {"gemini_reservation_id": 7, "reserved_cost_usd": 0.01}

        def record_gemini(self, **kwargs):
            self.records.append(kwargs)

    context = Context()
    monkeypatch.setattr(
        lesson_ordering,
        "_generate_lesson_order",
        lambda *args, **kwargs: _generation_result(["one", "two"]),
    )

    result = lesson_ordering.order_lesson_batch(
        reels,
        topic="topic",
        generation_context=context,
    )

    assert result.degraded is False
    assert context.reservations[0]["operation"] == "ordering"
    assert context.reservations[0]["model"] == config.LESSON_ORDER_MODEL
    assert context.records[0]["operation"] == "ordering"
    assert context.records[0]["usage"]["gemini_reservation_id"] == 7
    assert context.records[0]["usage"]["dispatched"] is True


def test_cache_read_observes_cancellation_before_return(monkeypatch) -> None:
    reels = [
        _reel("one", video_id="a", start=0, concept="one"),
        _reel("two", video_id="b", start=0, concept="two"),
    ]
    cancelled = False

    def read_cache(*args, **kwargs):
        nonlocal cancelled
        cancelled = True
        return lesson_ordering.LessonOrderResult(
            reels=reels,
            ordered_reel_ids=["one", "two"],
            model_used=config.LESSON_ORDER_MODEL,
            degraded=False,
            fallback_reason=None,
            provider_called=False,
            assessment_checkpoint_reel_ids=[],
        )

    monkeypatch.setattr(lesson_ordering, "_read_cached_lesson_order", read_cache)

    with pytest.raises(CancellationError):
        lesson_ordering.order_lesson_batch(
            reels,
            topic="topic",
            should_cancel=lambda: cancelled,
        )


def test_post_cache_write_cancellation_records_and_reconciles_billed_call(
    monkeypatch,
) -> None:
    reels = [
        _reel("one", video_id="a", start=0, concept="one"),
        _reel("two", video_id="b", start=0, concept="two"),
    ]
    cancelled = False

    class Context:
        def __init__(self) -> None:
            self.records: list[dict[str, Any]] = []

        def reserve_gemini_call(self, **kwargs):
            return {"gemini_reservation_id": 9, "reserved_cost_usd": 0.01}

        def record_gemini(self, **kwargs):
            self.records.append(kwargs)

    context = Context()
    monkeypatch.setattr(
        lesson_ordering,
        "_generate_lesson_order",
        lambda *args, **kwargs: _generation_result(["one", "two"]),
    )

    def write_cache(*args, **kwargs):
        nonlocal cancelled
        cancelled = True

    monkeypatch.setattr(lesson_ordering, "_write_cached_lesson_order", write_cache)

    with pytest.raises(CancellationError):
        lesson_ordering.order_lesson_batch(
            reels,
            topic="topic",
            should_cancel=lambda: cancelled,
            generation_context=context,
        )

    assert len(context.records) == 1
    assert context.records[0]["operation"] == "ordering"
    assert context.records[0]["error_code"] == "cancelled"
    assert context.records[0]["usage"]["gemini_reservation_id"] == 9
    assert context.records[0]["usage"]["dispatched"] is True


def test_generate_content_receives_text_only_and_no_media_configuration(
    monkeypatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeModels:
        async def generate_content(self, **kwargs):
            captured.update(kwargs)
            assert isinstance(kwargs.get("contents"), str)
            return SimpleNamespace(
                text=json.dumps(
                    {
                        "ordered_reel_ids": ["intro", "core"],
                        "assessment_checkpoint_reel_ids": [],
                    }
                ),
                model_version=config.LESSON_ORDER_MODEL,
                usage_metadata=None,
                candidates=[SimpleNamespace(finish_reason="STOP")],
            )

    class FakeAio:
        def __init__(self) -> None:
            self.models = FakeModels()

        async def aclose(self) -> None:
            return None

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            captured["client_kwargs"] = kwargs
            self.aio = FakeAio()

        def close(self) -> None:
            return None

    monkeypatch.setattr(config, "GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("google.genai.Client", FakeClient)
    user_prompt = lesson_ordering._user_prompt(
        [
            _reel("intro", video_id="abc123", start=0, concept="intro"),
            _reel("core", video_id="def456", start=20, concept="core"),
        ],
        topic="Newton's second law",
        learner_level="beginner",
    )

    result = asyncio.run(
        lesson_ordering._generate_lesson_order_async(
            lesson_ordering._SYSTEM_PROMPT,
            user_prompt,
            should_cancel=None,
            dispatch_state=lesson_ordering._DispatchState(),
        )
    )

    assert result.text
    assert set(captured) == {"client_kwargs", "model", "contents", "config"}
    assert captured["contents"] == user_prompt
    assert isinstance(captured["contents"], str)
    assert "https://" not in captured["contents"]
    assert "youtube.com" not in captured["contents"]
    request_config = captured["config"]
    assert getattr(request_config, "media_resolution", None) is None
    assert getattr(request_config, "response_mime_type", None) == "application/json"
    assert not isinstance(captured["contents"], (list, dict, bytes, bytearray))


def test_non_youtube_url_is_reduced_to_an_opaque_text_source_id() -> None:
    payload = lesson_ordering._clip_payload(
        {
            **_reel("clip", video_id="", start=0, concept="concept"),
            "video_url": "https://media.example.test/private/video.mp4?token=secret",
        }
    )

    assert payload["source_video_id"].startswith("source-")
    assert "http" not in payload["source_video_id"]
    assert ".mp4" not in payload["source_video_id"]

    explicit_url_payload = lesson_ordering._clip_payload(
        {
            **_reel(
                "explicit-url",
                video_id="https://youtube.com/watch?v=secret",
                start=0,
                concept="concept",
            ),
        }
    )
    assert explicit_url_payload["source_video_id"].startswith("source-")
    assert "http" not in explicit_url_payload["source_video_id"]
