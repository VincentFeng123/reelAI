# backend/tests/clip_engine/test_run.py
import pytest
from backend.app.clip_engine import expand, run
from backend.app.clip_engine.errors import (
    ProviderAuthenticationError,
    ProviderQuotaError,
    UnsupportedURLError,
)
from backend.app.clip_engine.provider_runtime import GenerationContext


@pytest.fixture(autouse=True)
def disable_persistent_segment_cache(monkeypatch):
    monkeypatch.setattr(run.segment_cache, "load_segment_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run.segment_cache, "store_segment_result", lambda *_args, **_kwargs: None)


def test_rejects_non_youtube():
    with pytest.raises(UnsupportedURLError):
        run.clip("https://vimeo.com/123", "topic")


def test_clip_builds_embed_urls_and_forwards_topic(monkeypatch):
    fake_tx = {"segments": [{"start": 0.0, "end": 5.0, "text": "hello world"}],
               "words": [], "duration": 5.0}
    seen: dict = {}

    def fake_segment(transcript, settings, progress=None, topic="", video_id=""):
        seen["topic"] = topic
        seen["video_id"] = video_id
        seen["accept_partial_flash"] = settings.get("segment_accept_partial_flash")
        return ([{"start": 1.0, "end": 4.0, "cut_end": 4.15, "title": "Bit",
                  "facet": "concept", "reason": "", "sequence_index": 1}], "1 clip")

    def fake_transcribe(url, video_id, settings):
        seen["transcript_url"] = url
        return fake_tx

    monkeypatch.setattr(run, "_transcribe", fake_transcribe)
    monkeypatch.setattr(run.gemini_segment, "segment_clips", fake_segment)
    out = run.clip(
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ&redirect=https://evil.test",
        "topic",
    )
    assert out["video_id"] == "dQw4w9WgXcQ"
    assert out["clips"][0]["embed_url"] == "https://www.youtube.com/embed/dQw4w9WgXcQ?start=1&end=4&rel=0"
    assert seen["topic"] == "topic"
    assert seen["video_id"] == "dQw4w9WgXcQ"
    assert seen["accept_partial_flash"] is True
    assert seen["transcript_url"] == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


def test_live_runner_uses_the_canonical_practice_segmenter():
    assert run.gemini_segment.__name__ == "backend.pipeline.gemini_segment"


def test_direct_url_segment_cache_hit_skips_all_gemini_and_budget(monkeypatch):
    assert run.segment_cache.SELECTION_CONTRACT_VERSION == "quality_silence_v6"
    transcript = {
        "segments": [{
            "start": 0.0,
            "end": 5.0,
            "text": "hello world explains the complete cached concept clearly",
        }],
        "words": [],
        "duration": 5.0,
    }
    cached_clip = {
        "start": 1.0,
        "end": 4.0,
        "title": "Bit",
        "facet": "concept",
        "reason": "Explains the concept",
        "learning_objective": "Understand the cached concept",
        "kind": "educational",
        "informativeness": 0.9,
        "topic_relevance": 0.9,
        "educational_importance": 0.9,
        "self_contained": True,
        "is_standalone": True,
        "directly_teaches_topic": True,
        "substantive": True,
        "factually_grounded": True,
        "topic_evidence_quote": "hello world explains the complete cached concept",
        "difficulty": 0.5,
        "uncertainty": "low",
        "uncertainty_reasons": [],
        "summary": "",
        "takeaways": [],
        "match_reason": "",
        "assessment": None,
        "sequence_index": 1,
    }
    assert run.segment_cache._valid_clips(
        [cached_clip], transcript=transcript, settings={}
    ) is not None
    context = GenerationContext("slow")
    expansion_dispatches = 0

    def fail_expansion(*_args, **_kwargs):
        nonlocal expansion_dispatches
        expansion_dispatches += 1
        pytest.fail("A direct URL cache hit must not depend on expansion cache state")

    # A URL-adapter source is already known. Even with no usable expansion
    # cache, its valid segment result must remain a zero-Gemini path.
    monkeypatch.setattr(expand, "_read_cached_expansion", lambda *_args: None)
    monkeypatch.setattr(expand, "_practice_fast_gemini_raw", fail_expansion)
    monkeypatch.setattr(run, "_transcribe", lambda *_args, **_kwargs: transcript)
    monkeypatch.setattr(
        run.segment_cache,
        "load_segment_result",
        lambda *_args, **_kwargs: ([cached_clip], "cached"),
    )
    monkeypatch.setattr(
        run.gemini_segment,
        "segment_clips",
        lambda *_args, **_kwargs: pytest.fail("Gemini must not run on a cache hit"),
    )

    output = run.clip(
        "https://youtu.be/dQw4w9WgXcQ",
        "topic",
        settings={"generation_context": context},
    )

    assert output["clips"][0]["title"] == "Bit"
    assert context.budget.snapshot()["used"]["segmentation"] == 0
    assert context.counters()["segmentation_cache_hits"] == 1
    assert expansion_dispatches == 0
    assert len(context.usage()) == 1
    assert context.usage()[0]["metadata"]["cache_hit"] is True


def test_segment_cache_miss_reserves_generic_and_per_dispatch_budgets(monkeypatch):
    transcript = {
        "segments": [{"start": 0.0, "end": 5.0, "text": "hello world"}],
        "words": [],
        "duration": 5.0,
    }
    context = GenerationContext("slow")

    def fake_segment(_transcript, settings, **_kwargs):
        settings["_segment_budget_reserve"](
            operation="flash_boundary_selector",
            model="gemini-3.5-flash",
            max_output_tokens=4096,
            prompt_text="short transcript",
        )
        return ([{"start": 1.0, "end": 4.0, "title": "Bit"}], "one")

    monkeypatch.setattr(run, "_transcribe", lambda *_args, **_kwargs: transcript)
    monkeypatch.setattr(run.gemini_segment, "segment_clips", fake_segment)

    run.clip(
        "https://youtu.be/dQw4w9WgXcQ",
        "topic",
        settings={"generation_context": context},
    )

    budget = context.budget.snapshot()
    assert budget["used"]["segmentation"] == 1
    assert budget["gemini"]["flash_selector_calls"] == 1


def test_empty_selector_result_is_cached(monkeypatch):
    transcript = {
        "segments": [{"start": 0.0, "end": 5.0, "text": "hello world"}],
        "words": [],
        "duration": 5.0,
    }
    stored: list[list[dict]] = []
    monkeypatch.setattr(run, "_transcribe", lambda *_args, **_kwargs: transcript)
    monkeypatch.setattr(
        run.gemini_segment,
        "segment_clips",
        lambda *_args, **_kwargs: ([], "No qualifying clips."),
    )
    monkeypatch.setattr(
        run.segment_cache,
        "store_segment_result",
        lambda _key, clips, *_args, **_kwargs: stored.append(clips),
    )

    output = run.clip("https://youtu.be/dQw4w9WgXcQ", "topic")

    assert output["clips"] == []
    assert stored == [[]]


def test_boundary_repair_usage_is_recorded_in_repair_stage():
    context = GenerationContext("slow")
    settings = {"generation_context": context}
    run._wire_segment_runtime(settings, "video")

    settings["_segment_telemetry"]({
        "event": "model_call",
        "operation": "flash_boundary_repair",
        "model": "gemini-3.5-flash",
        "prompt_tokens": 100,
        "candidate_tokens": 20,
        "thought_tokens": 10,
        "total_tokens": 130,
    })
    settings["_segment_telemetry"]({
        "event": "boundary_repair",
        "attempted_count": 2,
        "accepted_count": 1,
    })

    usage = context.usage_payload()
    assert usage["by_stage"]["repair"]["calls"] == 1
    assert usage["counters"]["boundary_repairs"] == 1


def test_shadow_routing_bypasses_segment_cache(monkeypatch):
    transcript = {
        "segments": [{"start": 0.0, "end": 5.0, "text": "hello world"}],
        "words": [],
        "duration": 5.0,
    }
    segment_calls = 0

    def fake_segment(*_args, **_kwargs):
        nonlocal segment_calls
        segment_calls += 1
        return ([{"start": 1.0, "end": 4.0, "title": "Bit"}], "shadow")

    monkeypatch.setattr(run, "_transcribe", lambda *_args, **_kwargs: transcript)
    monkeypatch.setattr(run.segment_cache, "cache_enabled", lambda: False)
    monkeypatch.setattr(
        run.segment_cache,
        "load_segment_result",
        lambda *_args, **_kwargs: pytest.fail("shadow mode must not read the cache"),
    )
    monkeypatch.setattr(run.gemini_segment, "segment_clips", fake_segment)

    output = run.clip("https://youtu.be/dQw4w9WgXcQ", "topic")

    assert output["notes"] == "shadow"
    assert segment_calls == 1


def test_transcript_provider_error_is_re_raised_unchanged(monkeypatch):
    from backend.app.clip_engine.clipper.pipeline import transcribe

    error = ProviderQuotaError(
        "quota", provider="supadata", operation="transcript", status_code=402
    )

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(transcribe, "transcribe_supadata", fail)
    with pytest.raises(ProviderQuotaError) as exc_info:
        run._transcribe(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "dQw4w9WgXcQ",
            {},
        )
    assert exc_info.value is error


def test_segmentation_provider_error_is_re_raised_unchanged(monkeypatch):
    transcript = {
        "segments": [{"start": 0.0, "end": 2.0, "text": "native cue"}],
        "words": [],
    }
    error = ProviderAuthenticationError(
        "auth", provider="gemini", operation="segmentation", status_code=403
    )
    monkeypatch.setattr(run, "_transcribe", lambda *args, **kwargs: transcript)

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(run.gemini_segment, "segment_clips", fail)
    with pytest.raises(ProviderAuthenticationError) as exc_info:
        run.clip("https://youtu.be/dQw4w9WgXcQ", "topic")
    assert exc_info.value is error
