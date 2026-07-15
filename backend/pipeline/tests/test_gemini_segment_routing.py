from __future__ import annotations

import hashlib
import json
import threading
import time
from types import SimpleNamespace

import pytest

from backend import gemini_client as GC
from backend.app.clip_engine.provider_runtime import GenerationContext
from backend.pipeline import gemini_segment as G


def _segments(duration: float = 100.0) -> list[dict]:
    return [{"start": 0.0, "end": duration, "text": "alpha lesson concept closes cleanly"}]


def test_provider_schemas_avoid_unsupported_additional_properties():
    for schema in (
        G._Plan,
        G._BoundaryPlan,
        G._CompactBoundaryPlan,
        G._IntentBoundaryPlan,
        G._BoundaryRepairPlan,
        G._EnrichmentPlan,
    ):
        assert "additionalProperties" not in str(GC._gemini3_json_schema(schema))


def _transcript(duration: float = 100.0) -> dict:
    return {"segments": _segments(duration), "words": [
        {"word": "alpha", "start": 0.0, "end": 0.5},
        {"word": "lesson", "start": 0.5, "end": 1.0},
        {"word": "concept", "start": 1.0, "end": 1.5},
        {"word": "closes", "start": duration - 1.0, "end": duration - 0.5},
        {"word": "cleanly", "start": duration - 0.5, "end": duration},
    ]}


def _empty_plan(schema: type, *, topic: str = ""):
    if schema in {G._CompactBoundaryPlan, G._IntentBoundaryPlan}:
        exact_request = topic.strip() or "(all educational topics)"
        return schema(
            request_intent={
                "exact_request": exact_request,
                "constraints": [{
                    "constraint_id": "subject",
                    "kind": "subject",
                    "source_phrase": topic.strip() or "all educational topics",
                    "requirement": "Teach the exact requested subject",
                }],
            },
            topics=[],
        )
    return schema(topics=[])


def _empty_selector_response(topic: str = "calculus") -> SimpleNamespace:
    return SimpleNamespace(
        text=json.dumps({
            "request_intent": {
                "exact_request": topic,
                "constraints": [{
                    "constraint_id": "subject",
                    "kind": "subject",
                    "source_phrase": topic,
                    "requirement": f"Teach {topic}",
                }],
            },
            "topics": [],
        }),
        candidates=[],
        usage_metadata=SimpleNamespace(
            prompt_token_count=120,
            candidates_token_count=10,
            thoughts_token_count=0,
            total_token_count=130,
            cached_content_token_count=0,
        ),
    )


class _HTTPStatusError(RuntimeError):
    def __init__(self, status_code: int):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


class _ModelSequence:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _install_model_sequence(monkeypatch, *outcomes) -> _ModelSequence:
    models = _ModelSequence(*outcomes)
    monkeypatch.setattr(
        GC,
        "get_client",
        lambda: type("Client", (), {"models": models})(),
    )
    monkeypatch.setattr(GC.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(GC.random, "uniform", lambda lower, _upper: lower)
    return models


def _clip(**overrides) -> dict:
    data = {
        "start": 0.0,
        "end": 100.0,
        "informativeness": 0.9,
        "topic_relevance": 0.9,
        "_uncertainty": "low",
        "_clip_text": "alpha lesson concept closes cleanly",
    }
    data.update(overrides)
    return data


def _report(**overrides) -> G._Conversion:
    data = {"clips": [_clip()], "proposed_count": 1}
    data.update(overrides)
    return G._Conversion(**data)


@pytest.mark.parametrize("report,topic", [
    (_report(), "alpha lesson"),
    (_report(clips=[_clip(end=150.0)]), "alpha"),
    (_report(near_duplicate=True), "alpha"),
    (_report(), "beta"),
    (_report(rejected_reasons=["proposal_0:bad_index"]), "alpha"),
])
def test_flash_classification_keeps_any_independently_valid_candidate(report, topic):
    classified = G._classify_flash(report, _segments(), topic, enrichment_required=False)
    assert classified == G._Classification("green", ())


def test_flash_classification_never_marks_below_green_output_green():
    classified = G._classify_flash(
        _report(score_below_green=True),
        _segments(),
        "alpha",
        enrichment_required=False,
    )
    assert classified == G._Classification(
        "invalid", ("quality_score_below_green",),
    )


def test_zero_valid_candidates_is_invalid_at_any_transcript_length():
    empty = G._Conversion(proposed_count=0)
    for duration in (30.0, 120.0):
        classified = G._classify_flash(
            empty, _segments(duration), "", enrichment_required=False,
        )
        assert classified.status == "invalid"
        assert "zero_valid_candidates" in classified.reasons


def test_optional_enrichment_errors_do_not_invalidate_boundaries():
    report = _report(enrichment_errors=["clip-001:assessment_invalid"])
    classified = G._classify_flash(report, _segments(), "alpha", enrichment_required=True)
    assert classified == G._Classification("green", ())


@pytest.mark.parametrize("topic", ["AI", "ML", "R", "Go", "C++"])
def test_short_technical_topics_participate_in_lexical_support(topic):
    token = next(iter(G._content_tokens(topic)))
    report = _report(clips=[_clip(_clip_text=f"This lesson teaches {token}")])
    assert G._classify_flash(
        report, _segments(), topic, enrichment_required=False,
    ).status == "green"


def _result(profile: str, status: str, *, title: str, error: str | None = None) -> G.SegmentResult:
    clips = [] if error else [{"start": 0.0, "end": 10.0, "title": title}]
    return G.SegmentResult(
        clips,
        "one clip",
        profile,
        status,
        [] if status == "green" else ["quality_score_below_green"],
        calls=[{
            "model": "gemini-3.5-flash" if profile.startswith("flash") else "gemini-3.1-pro-preview",
            "operation": profile,
            "prompt_version": profile,
            "thinking_level": "medium",
            "latency_ms": 10.0,
            "retries": 0,
            "finish_reason": "STOP",
            "prompt_tokens": 100,
            "candidate_tokens": 20,
            "thought_tokens": 10,
            "total_tokens": 130,
        }],
        proposed_count=1,
        accepted_count=len(clips),
        error=error,
    )


def test_pro_only_calls_only_boundary_pro_profile(monkeypatch):
    seen = []

    def fake_run(transcript, settings, profile, **kwargs):
        seen.append(profile)
        return _result(profile, "green", title="pro")

    monkeypatch.setattr(G, "run_segment_profile", fake_run)
    result = G.segment_clips_detailed(_transcript(), {}, video_id="video", routing_mode="pro_only")
    assert seen == [G.AUTHORITATIVE_PRO_PROFILE]
    assert result.clips[0]["title"] == "pro"


def test_all_authoritative_pro_calls_use_selected_baseline(monkeypatch):
    seen = []
    monkeypatch.setattr(G, "AUTHORITATIVE_PRO_PROFILE", G.CORRECTED_PRO_PROFILE)

    def fake_run(transcript, settings, profile, **kwargs):
        seen.append((profile, settings.get("_segment_operation")))
        return _result(profile, "green", title="pro")

    monkeypatch.setattr(G, "run_segment_profile", fake_run)
    authoritative = G._authoritative_pro(
        _transcript(), {}, "", time.monotonic() + 10, None,
    )
    fallback = G._authoritative_pro(
        _transcript(), {}, "", time.monotonic() + 10, None, fallback=True,
    )
    assert seen == [
        (G.CORRECTED_PRO_PROFILE, "pro_authoritative"),
        (G.PRO_BOUNDARY_PROFILE, "pro_fallback"),
    ]
    assert authoritative.calls[0]["operation"] == "pro_authoritative"
    assert fallback.calls[0]["operation"] == "pro_fallback"


def test_route_event_uses_selected_authoritative_prompt_version(monkeypatch):
    events = []
    monkeypatch.setattr(G, "AUTHORITATIVE_PRO_PROFILE", G.CORRECTED_PRO_PROFILE)
    monkeypatch.setattr(
        G,
        "run_segment_profile",
        lambda transcript, settings, profile, **kwargs: _result(
            profile, "green", title="pro"
        ),
    )
    G.segment_clips_detailed(
        _transcript(), {"_segment_telemetry": events.append},
        video_id="video", routing_mode="pro_only",
    )
    selected = next(event for event in events if event["event"] == "route_selected")
    assert selected["prompt_version"] == G.CORRECTED_PRO_PROFILE


@pytest.mark.parametrize(
    ("configured_deadline", "expected_deadline"),
    [(None, 136.0), (120.0, 120.0), (200.0, 136.0)],
)
def test_production_route_caps_and_respects_the_shared_deadline(
    monkeypatch,
    configured_deadline,
    expected_deadline,
):
    captured = {}

    def fake_run(transcript, settings, profile, **kwargs):
        captured.update(kwargs)
        return _result(profile, "green", title="flash")

    monkeypatch.setattr(G.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(G, "run_segment_profile", fake_run)
    settings = {}
    if configured_deadline is not None:
        settings["deadline_monotonic"] = configured_deadline

    G.segment_clips_detailed(
        _transcript(),
        settings,
        video_id="video",
        routing_mode="flash_only",
    )

    assert captured["deadline_monotonic"] == expected_deadline


def test_green_hybrid_flash_never_calls_pro(monkeypatch):
    seen = []
    monkeypatch.setattr(G.config, "SEGMENT_HYBRID_PERCENT", 100.0)

    def fake_run(transcript, settings, profile, **kwargs):
        seen.append(profile)
        return _result(profile, "green", title="flash")

    monkeypatch.setattr(G, "run_segment_profile", fake_run)
    result = G.segment_clips_detailed(_transcript(), {}, video_id="video", routing_mode="hybrid")
    assert seen == [G.PRODUCTION_FLASH_PROFILE]
    assert result.route == "hybrid_flash" and result.clips[0]["title"] == "flash"


def test_flash_only_ignores_rollout_and_never_calls_pro(monkeypatch):
    seen = []
    monkeypatch.setattr(G.config, "SEGMENT_HYBRID_PERCENT", 0.0)
    monkeypatch.setattr(G, "_flash_disabled_reason", "previous flash failure")

    def fake_run(transcript, settings, profile, **kwargs):
        seen.append(profile)
        return _result(profile, "invalid", title="unsafe flash")

    monkeypatch.setattr(G, "run_segment_profile", fake_run)
    result = G.segment_clips_detailed(
        _transcript(), {}, video_id="video", routing_mode="flash_only",
    )

    assert seen == [G.PRODUCTION_FLASH_PROFILE]
    assert result.route == "hybrid_flash_deferred"
    assert result.clips == []


def test_public_adapter_forwards_internal_flash_only_route(monkeypatch):
    captured = {}

    def fake_detailed(transcript, settings, **kwargs):
        captured.update(kwargs)
        return _result(G.PRODUCTION_FLASH_PROFILE, "green", title="flash")

    monkeypatch.setattr(G, "segment_clips_detailed", fake_detailed)
    clips, _notes = G.segment_clips(
        _transcript(), {"_segment_routing_mode": "flash_only"}, video_id="video",
    )

    assert captured["routing_mode"] == "flash_only"
    assert clips[0]["title"] == "flash"


def test_generation_gate_defers_non_green_flash_without_shipping_it(monkeypatch):
    seen = []
    gate_calls = []
    events = []
    monkeypatch.setattr(G.config, "SEGMENT_HYBRID_PERCENT", 100.0)

    def fake_run(transcript, settings, profile, **kwargs):
        seen.append(profile)
        return _result(profile, "invalid", title="unsafe flash")

    def gate(**kwargs):
        gate_calls.append(kwargs)
        return False

    monkeypatch.setattr(G, "run_segment_profile", fake_run)
    result = G.segment_clips_detailed(
        _transcript(),
        {
            "_segment_pro_fallback_gate": gate,
            "_segment_telemetry": events.append,
        },
        video_id="video",
        routing_mode="hybrid",
    )

    assert seen == [G.PRODUCTION_FLASH_PROFILE]
    assert gate_calls == [{"accepted_count": 1, "video_id": "video"}]
    assert result.route == "hybrid_flash_deferred"
    assert result.clips == []
    assert not any(event["event"] == "pro_fallback" for event in events)


@pytest.mark.parametrize("classification", ["uncertain", "invalid"])
def test_live_adapter_rejects_partial_flash_and_uses_pro(monkeypatch, classification):
    seen = []
    monkeypatch.setattr(G.config, "SEGMENT_HYBRID_PERCENT", 100.0)

    def fake_run(transcript, settings, profile, **kwargs):
        seen.append(profile)
        if profile == G.PRODUCTION_FLASH_PROFILE:
            return _result(profile, classification, title="flash")
        return _result(profile, "green", title="pro")

    monkeypatch.setattr(G, "run_segment_profile", fake_run)
    result = G.segment_clips_detailed(
        _transcript(),
        {"segment_accept_partial_flash": True},
        video_id="video",
        routing_mode="hybrid",
    )
    assert seen == [G.PRODUCTION_FLASH_PROFILE, G.PRO_BOUNDARY_PROFILE]
    assert result.route == "hybrid_pro_fallback"
    assert result.clips[0]["title"] == "pro"


def test_live_adapter_uses_pro_when_flash_has_no_surviving_clip(monkeypatch):
    seen = []
    monkeypatch.setattr(G.config, "SEGMENT_HYBRID_PERCENT", 100.0)

    def fake_run(transcript, settings, profile, **kwargs):
        seen.append(profile)
        if profile == G.PRODUCTION_FLASH_PROFILE:
            return _result(profile, "invalid", title="flash", error="empty flash")
        return _result(profile, "green", title="pro")

    monkeypatch.setattr(G, "run_segment_profile", fake_run)
    result = G.segment_clips_detailed(
        _transcript(),
        {"segment_accept_partial_flash": True},
        video_id="video",
        routing_mode="hybrid",
    )
    assert seen == [G.PRODUCTION_FLASH_PROFILE, G.PRO_BOUNDARY_PROFILE]
    assert result.route == "hybrid_pro_fallback"
    assert result.clips[0]["title"] == "pro"


@pytest.mark.parametrize("classification", ["uncertain", "invalid"])
def test_non_green_flash_calls_corrected_pro_exactly_once(monkeypatch, classification):
    seen = []
    monkeypatch.setattr(G.config, "SEGMENT_HYBRID_PERCENT", 100.0)

    def fake_run(transcript, settings, profile, **kwargs):
        seen.append(profile)
        if profile == G.PRODUCTION_FLASH_PROFILE:
            return _result(profile, classification, title="flash")
        return _result(profile, "green", title="pro")

    monkeypatch.setattr(G, "run_segment_profile", fake_run)
    result = G.segment_clips_detailed(_transcript(), {}, video_id="video", routing_mode="hybrid")
    assert seen == [G.PRODUCTION_FLASH_PROFILE, G.PRO_BOUNDARY_PROFILE]
    assert result.route == "hybrid_pro_fallback"
    assert result.clips[0]["title"] == "pro"
    assert result.fallback_reasons == ["quality_score_below_green"]


def test_failed_pro_never_exposes_non_green_flash(monkeypatch):
    monkeypatch.setattr(G.config, "SEGMENT_HYBRID_PERCENT", 100.0)

    def fake_run(transcript, settings, profile, **kwargs):
        if profile == G.PRODUCTION_FLASH_PROFILE:
            return _result(profile, "uncertain", title="flash")
        return _result(profile, "invalid", title="pro", error="provider failed")

    monkeypatch.setattr(G, "run_segment_profile", fake_run)
    result = G.segment_clips_detailed(_transcript(), {}, video_id="video", routing_mode="hybrid")
    assert result.clips == []
    assert result.error == "provider failed"


def test_flash_model_access_failure_rolls_process_back_to_pro_only(monkeypatch):
    seen = []
    monkeypatch.setattr(G.config, "SEGMENT_HYBRID_PERCENT", 100.0)
    monkeypatch.setattr(G, "_flash_disabled_reason", None)

    def fake_run(transcript, settings, profile, **kwargs):
        seen.append(profile)
        if profile == G.PRODUCTION_FLASH_PROFILE:
            return _result(
                profile, "invalid", title="flash",
                error=("GeminiTransportError: 404 NOT_FOUND. "
                       "models/gemini-3.5-flash is not found for API version v1beta"),
            )
        return _result(profile, "green", title="pro")

    monkeypatch.setattr(G, "run_segment_profile", fake_run)
    first = G.segment_clips_detailed(
        _transcript(), {}, video_id="video-a", routing_mode="hybrid",
    )
    second = G.segment_clips_detailed(
        _transcript(), {}, video_id="video-b", routing_mode="hybrid",
    )
    assert first.route == "hybrid_pro_fallback"
    assert second.route == "pro_only"
    assert seen == [
        G.PRODUCTION_FLASH_PROFILE,
        G.PRO_BOUNDARY_PROFILE,
        G.AUTHORITATIVE_PRO_PROFILE,
    ]


def test_split_enrichment_configuration_failure_rolls_future_requests_to_pro(monkeypatch):
    seen = []
    monkeypatch.setattr(G.config, "SEGMENT_HYBRID_PERCENT", 100.0)
    monkeypatch.setattr(G, "PRODUCTION_FLASH_PROFILE", G.FLASH_SPLIT_PROFILE)
    monkeypatch.setattr(G, "_flash_disabled_reason", None)

    def fake_run(transcript, settings, profile, **kwargs):
        seen.append(profile)
        result = _result(profile, "green", title="flash" if "flash" in profile else "pro")
        if profile == G.FLASH_SPLIT_PROFILE:
            result.flash_configuration_error = (
                "GeminiTransportError: 400 INVALID_ARGUMENT. Invalid value at model"
            )
            result.fallback_reasons = ["invalid_enrichment:clip-a"]
        return result

    monkeypatch.setattr(G, "run_segment_profile", fake_run)
    first = G.segment_clips_detailed(
        _transcript(), {}, video_id="video-a", routing_mode="hybrid",
    )
    second = G.segment_clips_detailed(
        _transcript(), {}, video_id="video-b", routing_mode="hybrid",
    )
    assert first.route == "hybrid_flash"
    assert second.route == "pro_only"
    assert seen == [G.FLASH_SPLIT_PROFILE, G.AUTHORITATIVE_PRO_PROFILE]


@pytest.mark.parametrize("message", [
    "400 INVALID_ARGUMENT. Invalid value at model",
    "401 UNAUTHENTICATED. API key not valid",
    "403 PERMISSION_DENIED. The caller lacks permission",
    "404 NOT_FOUND. models/gemini-3.5-flash is not found",
])
def test_realistic_provider_configuration_errors_disable_flash(message):
    assert G._flash_configuration_failure(message) is True


def test_shadow_always_returns_pro_and_records_flash_comparison(monkeypatch):
    seen = []
    events = []
    shadow_done = threading.Event()

    def fake_run(transcript, settings, profile, **kwargs):
        seen.append(profile)
        if profile == G.AUTHORITATIVE_PRO_PROFILE:
            return _result(profile, "green", title="pro")
        return _result(profile, "invalid", title="shadow-flash")

    monkeypatch.setattr(G, "run_segment_profile", fake_run)
    def sink(event):
        events.append(event)
        if event["event"] == "shadow_comparison":
            shadow_done.set()

    result = G.segment_clips_detailed(
        _transcript(), {"_segment_telemetry": sink},
        video_id="video", routing_mode="shadow",
    )
    assert shadow_done.wait(timeout=1)
    assert set(seen) == {G.AUTHORITATIVE_PRO_PROFILE, G.PRODUCTION_FLASH_PROFILE}
    assert result.clips[0]["title"] == "pro"
    assert any(event["event"] == "shadow_comparison" for event in events)


def test_shadow_split_records_enrichment_fallback_reasons_and_rate(monkeypatch):
    events = []
    shadow_done = threading.Event()
    monkeypatch.setattr(G, "PRODUCTION_FLASH_PROFILE", G.FLASH_SPLIT_PROFILE)

    def fake_run(transcript, settings, profile, **kwargs):
        result = _result(profile, "green", title="flash" if "flash" in profile else "pro")
        if profile == G.FLASH_SPLIT_PROFILE:
            result.fallback_reasons = ["invalid_enrichment:clip-a"]
        return result

    def sink(event):
        events.append(event)
        if event["event"] == "shadow_comparison":
            shadow_done.set()

    monkeypatch.setattr(G, "run_segment_profile", fake_run)
    G.segment_clips_detailed(
        _transcript(), {"_segment_telemetry": sink},
        video_id="video", routing_mode="shadow",
    )
    assert shadow_done.wait(timeout=1)
    comparison = next(event for event in events if event["event"] == "shadow_comparison")
    assert comparison["fallback_reasons"] == ["invalid_enrichment:clip-a"]
    assert comparison["fallback_rate"] == 1.0
    assert comparison["cost_per_accepted_clip_usd"] is not None
    assert any(event["event"] == "pro_fallback" and event.get("shadow") is True
               for event in events)


def test_shadow_flash_never_delays_authoritative_pro_response(monkeypatch):
    release_flash = threading.Event()
    flash_started = threading.Event()
    shadow_done = threading.Event()

    def fake_run(transcript, settings, profile, **kwargs):
        if profile == G.PRODUCTION_FLASH_PROFILE:
            flash_started.set()
            release_flash.wait(timeout=2)
            shadow_done.set()
            return _result(profile, "green", title="flash")
        return _result(profile, "green", title="pro")

    monkeypatch.setattr(G, "run_segment_profile", fake_run)
    started = time.perf_counter()
    result = G.segment_clips_detailed(
        _transcript(), {}, video_id="video", routing_mode="shadow",
    )
    elapsed = time.perf_counter() - started
    assert flash_started.wait(timeout=1)
    assert elapsed < 0.5
    assert not shadow_done.is_set()
    assert result.clips[0]["title"] == "pro"
    release_flash.set()
    assert shadow_done.wait(timeout=1)


def test_hybrid_assignment_is_sha256_stable_and_missing_id_fails_closed():
    video_id = "abc123"
    bucket = int.from_bytes(hashlib.sha256(video_id.encode()).digest()[:8], "big") % 10_000
    assert G._hybrid_selected(video_id, bucket / 100.0) is False
    assert G._hybrid_selected(video_id, (bucket + 1) / 100.0) is True
    assert G._hybrid_selected("", 100.0) is False


def test_request_segment_model_cannot_redirect_pro(monkeypatch):
    seen_models = []

    def fake_call(system, user, schema, **kwargs):
        seen_models.append(kwargs["model"])
        return G._LegacyPlan(topics=[]), {}

    monkeypatch.setattr(G, "_call_model", fake_call)
    G.run_segment_profile(
        _transcript(), {"segment_model": "gemini-2.5-flash"}, G.PRODUCTION_PRO_PROFILE,
    )
    assert seen_models == [G.config.SEGMENT_PRO_MODEL]


def test_schema_failure_preserves_successful_call_usage_telemetry(monkeypatch):
    telemetry = GC.GeminiCallTelemetry(
        model="gemini-3.5-flash", operation="flash_single_candidate",
        prompt_version=G.FLASH_SINGLE_PROFILE, thinking_level="medium",
        latency_ms=12.0, retries=0, finish_reason="STOP",
        prompt_tokens=100, candidate_tokens=10, thought_tokens=20, total_tokens=130,
    )
    monkeypatch.setattr(
        GC, "generate_json_v3",
        lambda *args, **kwargs: GC.GenerationResult("not-json", telemetry),
    )
    with pytest.raises(G._SchemaResponseError) as exc_info:
        G._call_model(
            "system", "user", G._Plan,
            model=G.config.SEGMENT_FLASH_MODEL, thinking_level="medium",
            max_output_tokens=24_576, timeout_s=45.0,
            deadline_monotonic=time.monotonic() + 10,
            operation="flash_single_candidate", prompt_version=G.FLASH_SINGLE_PROFILE,
            cancelled=None,
        )
    assert G._exception_telemetry(exc_info.value)["total_tokens"] == 130
    assert G._exception_telemetry(exc_info.value)["dispatched"] is True


def test_boundary_schema_rejects_one_bad_topic_without_losing_valid_sibling(
    monkeypatch,
):
    valid = G._CompactBoundaryTopic(
        candidate_id="candidate-alpha",
        start_line=0,
        end_line=0,
        start_quote="Alpha lesson defines the concept",
        end_quote="closes with a clear conclusion",
        title="Alpha lesson",
        learning_objective="Understand the complete alpha lesson.",
        facet="alpha",
        informativeness=0.9,
        topic_relevance=0.9,
        educational_importance=0.9,
        difficulty=0.5,
        directly_teaches_topic=True,
        substantive=True,
        factually_grounded=True,
        self_contained=True,
        is_standalone=True,
        intent_evidence=[{
            "constraint_id": "subject",
            "evidence_quote": "Alpha lesson defines the concept completely",
        }],
    ).model_dump(mode="json", by_alias=True)
    malformed = {**valid, "id": "candidate-bad", "rel": "high"}
    telemetry = GC.GeminiCallTelemetry(
        model=G.config.SEGMENT_FLASH_MODEL,
        operation="flash_boundary_selector",
        prompt_version=G.FLASH_SPLIT_PROFILE,
        thinking_level="low",
        latency_ms=12.0,
        retries=0,
        finish_reason="STOP",
        prompt_tokens=100,
        candidate_tokens=100,
        thought_tokens=0,
        total_tokens=200,
    )
    monkeypatch.setattr(
        GC,
        "generate_json_v3",
        lambda *args, **kwargs: GC.GenerationResult(
            json.dumps({
                "request_intent": {
                    "exact_request": "alpha lesson",
                    "constraints": [{
                        "constraint_id": "subject",
                        "kind": "subject",
                        "source_phrase": "alpha lesson",
                        "requirement": "Teach the alpha lesson",
                    }],
                },
                "topics": [malformed, valid],
            }),
            telemetry,
        ),
    )

    result = G.run_segment_profile(
        {
            "segments": [{
                "start": 0.0,
                "end": 10.0,
                "text": (
                    "Alpha lesson defines the concept completely. Therefore, the lesson "
                    "closes with a clear conclusion."
                ),
            }],
            "words": [],
            "source": "supadata",
        },
        {},
        G.FLASH_SPLIT_PROFILE,
        topic="alpha lesson",
    )

    assert result.accepted_count == 1
    assert result.proposed_count == 2
    assert result.classification == "green"
    assert result.rejection_reasons == [
        "proposal_0:schema_invalid:rel:float_type"
    ]
    assert result.calls[0]["schema_rejected_count"] == 1


def test_dispatched_transport_failure_preserves_call_identity(monkeypatch):
    monkeypatch.setattr(
        GC,
        "generate_json_v3",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("transport down")),
    )

    with pytest.raises(G._ModelCallError) as exc_info:
        G._call_model(
            "system",
            "user",
            G._BoundaryPlan,
            model=G.config.SEGMENT_FLASH_MODEL,
            thinking_level="medium",
            max_output_tokens=4_096,
            timeout_s=45.0,
            deadline_monotonic=time.monotonic() + 10,
            operation="flash_boundary_selector",
            prompt_version=G.FLASH_SPLIT_PROFILE,
            cancelled=None,
        )

    telemetry = G._exception_telemetry(exc_info.value)
    assert telemetry["dispatched"] is True
    assert telemetry["operation"] == "flash_boundary_selector"
    assert telemetry["model"] == G.config.SEGMENT_FLASH_MODEL


def test_production_selector_retries_one_503_with_one_reservation(monkeypatch):
    models = _install_model_sequence(
        monkeypatch,
        _HTTPStatusError(503),
        _empty_selector_response(),
    )
    reservations = []

    def reserve(**kwargs):
        reservations.append(kwargs)
        return {"reserved_cost_usd": 0.25}

    result = G.run_segment_profile(
        _transcript(),
        {"_segment_budget_reserve": reserve},
        G.PRODUCTION_FLASH_PROFILE,
        topic="calculus",
        deadline_monotonic=time.monotonic() + 10,
    )

    assert len(models.calls) == 2
    assert {call["model"] for call in models.calls} == {
        G.config.SEGMENT_FLASH_MODEL
    }
    assert len(reservations) == 1
    assert result.calls[0]["retries"] == 1
    assert result.calls[0]["error_history"] == ({
        "provider_error_type": "_HTTPStatusError",
        "provider_status_code": 503,
        "retryable": True,
    },)
    assert result.calls[0]["dispatched"] is True
    assert result.calls[0]["reserved_cost_usd"] == 0.25
    assert result.error is None
    assert result.classification_reasons == ["zero_valid_candidates"]


@pytest.mark.parametrize(
    ("primary_statuses", "failover_reason"),
    [
        ((503, 503), "primary_503_retry_exhausted"),
        ((503, 504), "primary_503_retry_exhausted"),
        ((500,), "primary_transient_5xx_failover"),
        ((502,), "primary_transient_5xx_failover"),
        ((504,), "primary_transient_5xx_failover"),
    ],
)
def test_production_selector_failover_reuses_one_reservation(
    monkeypatch,
    primary_statuses,
    failover_reason,
):
    models = _install_model_sequence(
        monkeypatch,
        *(_HTTPStatusError(status) for status in primary_statuses),
        _empty_selector_response(),
    )
    context = GenerationContext("fast", generation_id="selector-failover")
    reservations = []
    reconciliations = []

    def reserve(**kwargs):
        reservations.append(kwargs)
        return context.reserve_gemini_call(**kwargs)

    def reconcile(**kwargs):
        reconciliations.append(kwargs)
        return context.reconcile_gemini_call(**kwargs)

    result = G.run_segment_profile(
        _transcript(),
        {
            "_segment_budget_reserve": reserve,
            "_segment_budget_reconcile": reconcile,
        },
        G.PRODUCTION_FLASH_PROFILE,
        topic="calculus",
        deadline_monotonic=time.monotonic() + 10,
    )

    assert [call["model"] for call in models.calls] == [
        *([G.config.SEGMENT_FLASH_MODEL] * len(primary_statuses)),
        G.config.SEGMENT_FLASH_FALLBACK_MODEL,
    ]
    assert len(reservations) == 1
    assert len(reconciliations) == 1
    assert reconciliations[0]["model_used"] == G.config.SEGMENT_FLASH_FALLBACK_MODEL
    call = result.calls[0]
    assert call["model"] == G.config.SEGMENT_FLASH_FALLBACK_MODEL
    assert call["retries"] == len(primary_statuses)
    assert len(call["error_history"]) == len(primary_statuses)
    assert call["failover_from_model"] == G.config.SEGMENT_FLASH_MODEL
    assert call["failover_model"] == G.config.SEGMENT_FLASH_FALLBACK_MODEL
    assert call["failover_reason"] == failover_reason
    assert call["quality_degraded"] is True
    assert isinstance(call["gemini_reservation_id"], int)
    assert result.error is None
    budget = context.budget.snapshot()["gemini"]
    assert budget["flash_selector_calls"] == 1
    assert budget["inflight_reserved_cost_usd"] == 0.0
    assert budget["committed_cost_usd"] == pytest.approx(0.000045)


@pytest.mark.parametrize(
    "primary_errors",
    [
        (_HTTPStatusError(503), _HTTPStatusError(400)),
        (_HTTPStatusError(503), _HTTPStatusError(408)),
        (_HTTPStatusError(503), _HTTPStatusError(429)),
        (RuntimeError("status 504"),),
    ],
)
def test_production_selector_does_not_fail_over_without_eligible_typed_statuses(
    monkeypatch,
    primary_errors,
):
    models = _install_model_sequence(monkeypatch, *primary_errors)

    result = G.run_segment_profile(
        _transcript(),
        {},
        G.PRODUCTION_FLASH_PROFILE,
        topic="calculus",
        deadline_monotonic=time.monotonic() + 10,
    )

    assert len(models.calls) == len(primary_errors)
    assert {call["model"] for call in models.calls} == {
        G.config.SEGMENT_FLASH_MODEL
    }
    call = result.calls[0]
    assert "failover_model" not in call
    assert "failover_reason" not in call
    assert result.error is not None


def test_flash_failover_requires_time_and_no_cancellation(monkeypatch):
    transport_error = type("GeminiTransportError", (RuntimeError,), {})()
    telemetry = {
        "provider_status_code": 503,
        "retryable": True,
        "retries": 1,
        "dispatched": True,
        "error_history": ({
            "provider_error_type": "ServerError",
            "provider_status_code": 503,
            "retryable": True,
        }, {
            "provider_error_type": "ServerError",
            "provider_status_code": 503,
            "retryable": True,
        }),
    }
    monkeypatch.setattr(G.config, "SEGMENT_FLASH_MODEL", "gemini-3.5-flash")
    monkeypatch.setattr(G.time, "monotonic", lambda: 100.0)
    common = {
        "primary_exception": transport_error,
        "primary_model": "gemini-3.5-flash",
        "failover_model": "gemini-3.1-flash-lite",
        "operation": "flash_boundary_selector",
    }

    assert G._flash_failover_reason(
        telemetry,
        **common,
        deadline_monotonic=106.0,
        cancelled=None,
    ) == "primary_503_retry_exhausted"
    assert G._flash_failover_reason(
        telemetry,
        **common,
        deadline_monotonic=104.999,
        cancelled=None,
    ) is None
    assert G._flash_failover_reason(
        telemetry,
        **common,
        deadline_monotonic=106.0,
        cancelled=lambda: True,
    ) is None
    assert G._flash_failover_reason(
        {**telemetry, "provider_status_code": 500},
        **common,
        deadline_monotonic=106.0,
        cancelled=None,
    ) is None
    assert G._flash_failover_reason(
        telemetry,
        **{**common, "failover_model": "gemini-3.1-pro-preview"},
        deadline_monotonic=106.0,
        cancelled=None,
    ) is None

    monkeypatch.setattr(G.config, "SEGMENT_FLASH_MODEL", "models/gemini-3.5-flash")
    assert G._flash_failover_reason(
        telemetry,
        **{
            **common,
            "primary_model": "models/gemini-3.5-flash",
            "failover_model": "models/gemini-3.1-flash-lite",
        },
        deadline_monotonic=106.0,
        cancelled=None,
    ) == "primary_503_retry_exhausted"

    immediate_504 = {
        **telemetry,
        "provider_status_code": 504,
        "retries": 0,
        "error_history": ({
            "provider_error_type": "ServerError",
            "provider_status_code": 504,
            "retryable": True,
        },),
    }
    assert G._flash_failover_reason(
        immediate_504,
        **common,
        deadline_monotonic=106.0,
        cancelled=None,
    ) == "primary_transient_5xx_failover"
    assert G._flash_failover_reason(
        {
            **telemetry,
            "provider_status_code": 400,
            "retryable": False,
            "error_history": (
                telemetry["error_history"][0],
                {
                    "provider_error_type": "ClientError",
                    "provider_status_code": 400,
                    "retryable": False,
                },
            ),
        },
        **common,
        deadline_monotonic=106.0,
        cancelled=None,
    ) is None


def test_production_selector_does_not_retry_failed_lite_failover(monkeypatch):
    models = _install_model_sequence(
        monkeypatch,
        _HTTPStatusError(503),
        _HTTPStatusError(503),
        _HTTPStatusError(503),
    )

    result = G.run_segment_profile(
        _transcript(),
        {},
        G.PRODUCTION_FLASH_PROFILE,
        topic="calculus",
        deadline_monotonic=time.monotonic() + 10,
    )

    assert [call["model"] for call in models.calls] == [
        G.config.SEGMENT_FLASH_MODEL,
        G.config.SEGMENT_FLASH_MODEL,
        G.config.SEGMENT_FLASH_FALLBACK_MODEL,
    ]
    call = result.calls[0]
    assert call["model"] == G.config.SEGMENT_FLASH_FALLBACK_MODEL
    assert call["retries"] == 2
    assert len(call["error_history"]) == 3
    assert call["quality_degraded"] is True
    assert result.error is not None


def test_transport_failure_reports_inner_type_and_retry_telemetry(monkeypatch):
    provider_telemetry = GC.GeminiCallTelemetry(
        model=G.config.SEGMENT_FLASH_MODEL,
        operation="flash_boundary_selector",
        prompt_version=G.FLASH_SPLIT_PROFILE,
        thinking_level="low",
        latency_ms=321.0,
        retries=1,
        finish_reason=None,
        prompt_tokens=None,
        candidate_tokens=None,
        thought_tokens=None,
        total_tokens=None,
        provider_error_type="ReadError",
        provider_status_code=None,
        retryable=True,
    )
    monkeypatch.setattr(
        GC,
        "generate_json_v3",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            GC.GeminiTransportError(
                "status 503 private provider prose", provider_telemetry,
            )
        ),
    )
    events = []

    result = G.segment_clips_detailed(
        _transcript(),
        {"_segment_telemetry": events.append},
        topic="calculus",
        routing_mode="flash_only",
    )

    assert result.classification_reasons == [
        "request_failure:GeminiTransportError"
    ]
    assert result.rejection_reasons == [
        "request_failure:GeminiTransportError"
    ]
    assert result.calls[0]["retries"] == 1
    assert result.calls[0]["latency_ms"] == 321.0
    assert result.calls[0]["error_type"] == "GeminiTransportError"
    assert result.calls[0]["provider_error_type"] == "ReadError"
    assert result.calls[0]["provider_status_code"] is None
    assert result.calls[0]["retryable"] is True
    assert [event["event"] for event in events if event["event"] == "segment_error"] == [
        "segment_error"
    ]
    assert "private provider prose" not in str(result.error)
    assert "private provider prose" not in str(events)


@pytest.mark.parametrize(
    "profile,expected",
    [
        (G.PRODUCTION_PRO_PROFILE,
         ("high", 24_576, 90.0, "pro_authoritative", "gemini-3.1-pro-preview", 0)),
        (G.CORRECTED_PRO_PROFILE,
         ("high", 24_576, 90.0, "pro_fallback", "gemini-3.1-pro-preview", 0)),
    (G.FLASH_SINGLE_PROFILE,
         ("medium", 24_576, 45.0, "flash_single_candidate", "gemini-3.5-flash", 0)),
    (G.FLASH_SPLIT_PROFILE,
         ("low", 6_000, 20.0, "flash_boundary_selector", "gemini-3.5-flash", 1)),
    (G.PRO_BOUNDARY_PROFILE,
         ("high", 6_000, 90.0, "pro_fallback", "gemini-3.1-pro-preview", 0)),
    ],
)
def test_profile_operation_settings_are_wired_to_client(monkeypatch, profile, expected):
    captured = {}

    def fake_call(system, user, schema, **kwargs):
        captured.update(kwargs)
        plan = G._LegacyPlan(topics=[]) if schema is G._LegacyPlan else _empty_plan(schema)
        return plan, {}

    monkeypatch.setattr(G, "_call_model", fake_call)
    G._run_selection_profile(
        profile, _transcript(), "", {},
        deadline=time.monotonic() + 10, cancelled=None,
    )
    level, cap, timeout, operation, default_model, max_retries = expected
    expected_model = (G.config.SEGMENT_PRO_MODEL if "pro" in default_model
                      else G.config.SEGMENT_FLASH_MODEL)
    assert (
        captured["thinking_level"], captured["max_output_tokens"],
        captured["timeout_s"], captured["operation"], captured["model"],
        captured["max_retries"],
    ) == (
        level,
        cap,
        timeout,
        operation,
        expected_model,
        max_retries,
    )
    assert captured["failover_model"] == (
        G.config.SEGMENT_FLASH_FALLBACK_MODEL
        if profile == G.FLASH_SPLIT_PROFILE
        else None
    )
    assert captured["retry_status_codes"] == (
        frozenset({503}) if profile == G.FLASH_SPLIT_PROFILE else None
    )


def test_flash_boundary_profile_accepts_bootstrap_low_thinking_override(monkeypatch):
    captured = {}

    def fake_call(system, user, schema, **kwargs):
        captured.update(kwargs)
        return _empty_plan(schema), {}

    monkeypatch.setattr(G, "_call_model", fake_call)
    G._run_selection_profile(
        G.FLASH_SPLIT_PROFILE,
        _transcript(),
        "",
        {"_segment_thinking_level": "low"},
        deadline=time.monotonic() + 10,
        cancelled=None,
    )

    assert captured["thinking_level"] == "low"
    assert captured["max_output_tokens"] == 6_000


def test_boundary_profile_keeps_grounded_clip_with_bad_edge_quote(monkeypatch):
    proposal = G._BoundaryTopic(
        candidate_id="candidate-bad-quote",
        start_line=0,
        end_line=0,
        start_quote="missing quote",
        end_quote="closes cleanly",
        title="Alpha",
        learning_objective="Understand alpha.",
        facet="alpha",
        reason="Complete lesson.",
        informativeness=0.9,
        topic_relevance=0.9,
        educational_importance=0.9,
        difficulty=0.5,
        directly_teaches_topic=True,
        substantive=True,
        factually_grounded=True,
        topic_evidence_quote="alpha lesson concept closes cleanly",
        self_contained=True,
        is_standalone=True,
        prerequisite_candidate_ids=[],
        uncertainty="low",
        uncertainty_reasons=[],
    )
    monkeypatch.setattr(
        G,
        "_call_model",
        lambda *args, **kwargs: (G._BoundaryPlan(topics=[proposal]), {}),
    )

    result = G.run_segment_profile(
        _transcript(),
        {"_segment_ignore_caption_case": True},
        G.PRO_BOUNDARY_PROFILE,
        deadline_monotonic=time.monotonic() + 10,
    )

    assert result.classification == "green"
    assert result.accepted_count == 1
    assert result.rejection_reasons == []


def test_production_boundary_selector_keeps_every_candidate_beyond_sixteen(monkeypatch):
    segments = [
        {
            "cue_id": f"c{index}",
            "start": index * 10.0,
            "end": (index + 1) * 10.0,
            "text": f"line {index} teaches concept {index} and finishes end {index}",
        }
        for index in range(17)
    ]
    topics = [
        G._BoundaryTopic(
            candidate_id=f"candidate-{index}",
            start_line=index,
            end_line=index,
            start_quote=f"line {index}",
            end_quote=f"end {index}",
            title=f"T{index}",
            learning_objective=f"Understand concept {index}.",
            facet="concept",
            reason="Complete educational moment.",
            informativeness=0.80 + index * 0.004,
            topic_relevance=0.80 + index * 0.004,
            educational_importance=0.80 + index * 0.004,
            difficulty=0.5,
            directly_teaches_topic=True,
            substantive=True,
            factually_grounded=True,
            topic_evidence_quote=(
                f"line {index} teaches concept {index} and finishes"
            ),
            self_contained=True,
            is_standalone=True,
            prerequisite_candidate_ids=[],
            uncertainty="low",
            uncertainty_reasons=[],
        )
        for index in range(17)
    ]
    assert len(G._BoundaryPlan(topics=topics).topics) == 17

    captured = {}
    monkeypatch.setattr(
        G,
        "_call_model",
        lambda system, user, *args, **kwargs: (
            captured.update({"user": user}) or G._BoundaryPlan(topics=topics),
            {},
        ),
    )

    report, classification, _ = G._run_selection_profile(
        G.PRODUCTION_FLASH_PROFILE,
        {"segments": segments, "words": [], "source": "supadata"},
        "",
        {"max_clips": 1},
        deadline=time.monotonic() + 10,
        cancelled=None,
    )

    assert classification.status == "green"
    assert len(report.clips) == 17
    assert "return every distinct qualifying moment" in captured["user"].casefold()
    assert "up to 16" not in captured["user"].casefold()
    assert "T0" in {clip["title"] for clip in report.clips}
    assert "T16" in {clip["title"] for clip in report.clips}


def test_production_order_uses_information_and_importance_with_relevance(monkeypatch):
    segments = [
        {
            "cue_id": f"c{index}",
            "start": index * 10.0,
            "end": (index + 1) * 10.0,
            "text": f"line {index} teaches concept {index} and finishes end {index}",
        }
        for index in range(2)
    ]
    topics = [
        G._BoundaryTopic(
            candidate_id=f"candidate-{index}",
            start_line=index,
            end_line=index,
            start_quote=f"line {index}",
            end_quote=f"end {index}",
            title=title,
            learning_objective=f"Understand concept {index}.",
            facet=f"facet {index}",
            reason="Complete educational moment.",
            informativeness=informativeness,
            topic_relevance=relevance,
            educational_importance=importance,
            difficulty=0.5,
            directly_teaches_topic=True,
            substantive=True,
            factually_grounded=True,
            topic_evidence_quote=(
                f"line {index} teaches concept {index} and finishes"
            ),
            self_contained=True,
            is_standalone=True,
            prerequisite_candidate_ids=[],
            uncertainty="low",
            uncertainty_reasons=[],
        )
        for index, title, informativeness, relevance, importance in (
            (0, "Most relevant", 0.75, 0.99, 0.75),
            (1, "Less relevant", 1.0, 0.80, 1.0),
        )
    ]
    monkeypatch.setattr(
        G,
        "_call_model",
        lambda *_args, **_kwargs: (G._BoundaryPlan(topics=topics), {}),
    )

    report, classification, _ = G._run_selection_profile(
        G.PRODUCTION_FLASH_PROFILE,
        {"segments": segments, "words": [], "source": "supadata"},
        "concept",
        {"max_clips": 1},
        deadline=time.monotonic() + 10,
        cancelled=None,
    )

    assert classification.status == "green"
    assert [clip["title"] for clip in report.clips] == [
        "Less relevant",
        "Most relevant",
    ]


def test_production_profile_preserves_optional_learning_detail_schema(monkeypatch):
    captured = {}

    def fake_call(system, user, schema, **kwargs):
        captured.update({"system": system, "user": user, "schema": schema})
        return G._LegacyPlan(topics=[]), {}

    monkeypatch.setattr(G, "_call_model", fake_call)
    G._run_selection_profile(
        G.PRODUCTION_PRO_PROFILE, _transcript(), "", {},
        deadline=time.monotonic() + 10, cancelled=None,
    )
    assert captured["schema"] is G._LegacyPlan
    assert "summary" in captured["user"] and "assessment" in captured["user"]


def _private_clip(clip_id: str, text: str) -> dict:
    return {
        "_clip_id": clip_id,
        "_clip_text": text,
        "summary": "",
        "takeaways": [],
        "match_reason": "",
        "assessment": None,
    }


def _enrichment(clip_id: str, text: str, evidence: str) -> G._EnrichmentItem:
    return G._EnrichmentItem(
        clip_id=clip_id,
        summary=f"{text} is explained.",
        takeaways=[f"{text} is taught.", f"The {text} closes."],
        match_reason=f"The {text} lesson is relevant.",
        assessment={
            "prompt": "What is taught?",
            "options":[text, "Sponsor", "Greeting", "Outro"],
            "correct_index": 0,
            "explanation": f"The {text} is taught.",
            "evidence_quote": evidence,
        },
    )


def test_split_invalid_item_is_cleared_without_pro_retry(monkeypatch):
    clips = [_private_clip("clip-a", "alpha lesson"), _private_clip("clip-b", "beta lesson")]
    calls = []

    def fake_call(system, user, schema, **kwargs):
        calls.append((kwargs, user))
        return G._EnrichmentPlan(items=[
            _enrichment("clip-a", "alpha lesson", "alpha lesson"),
            _enrichment("clip-b", "beta lesson", "not in beta"),
        ]), {"model": kwargs["model"]}

    monkeypatch.setattr(G, "_call_model", fake_call)
    enriched, _telemetry, reasons, configuration_error = G._enrich_split(
        clips, "", deadline=time.monotonic() + 10, cancelled=None,
    )
    assert [kwargs["model"] for kwargs, _user in calls] == [G.config.SEGMENT_FLASH_MODEL]
    assert (calls[0][0]["thinking_level"], calls[0][0]["max_output_tokens"],
            calls[0][0]["timeout_s"]) == ("low", 2_048, 25.0)
    assert enriched[0]["summary"]
    assert enriched[1]["summary"] == ""
    assert reasons == []
    assert configuration_error is None


def test_split_enrichment_failure_preserves_clip_without_details(monkeypatch):
    clips = [_private_clip("clip-a", "alpha lesson")]

    def fail(*args, **kwargs):
        raise RuntimeError("provider failed")

    monkeypatch.setattr(G, "_call_model", fail)
    enriched, _calls, reasons, configuration_error = G._enrich_split(
        clips, "", deadline=time.monotonic() + 10, cancelled=None,
    )
    assert enriched[0]["summary"] == ""
    assert enriched[0]["takeaways"] == []
    assert enriched[0]["assessment"] is None
    assert reasons == []
    assert configuration_error is None


def test_semantically_invalid_enrichment_is_nonfatal_without_pro(monkeypatch):
    clips = [_private_clip("clip-a", "alpha lesson")]
    attempts = 0

    def invalid_evidence(system, user, schema, **kwargs):
        nonlocal attempts
        attempts += 1
        return G._EnrichmentPlan(items=[
            _enrichment("clip-a", "alpha lesson", "not in alpha"),
        ]), {"model": kwargs["model"]}

    monkeypatch.setattr(G, "_call_model", invalid_evidence)
    enriched, _calls, reasons, configuration_error = G._enrich_split(
        clips, "", deadline=time.monotonic() + 10, cancelled=None,
    )
    assert attempts == 1
    assert reasons == []
    assert enriched[0]["assessment"] is None
    assert configuration_error is None


def test_split_enrichment_configuration_failure_does_not_invalidate_boundaries(monkeypatch):
    clips = [_private_clip("clip-a", "alpha lesson")]
    attempts = 0

    def fake_call(system, user, schema, **kwargs):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("400 INVALID_ARGUMENT. Invalid value at model")

    monkeypatch.setattr(G, "_call_model", fake_call)
    enriched, _calls, reasons, configuration_error = G._enrich_split(
        clips, "", deadline=time.monotonic() + 10, cancelled=None,
    )
    assert attempts == 1
    assert enriched[0]["summary"] == ""
    assert reasons == []
    assert configuration_error is None


def test_uncertain_split_boundaries_skip_enrichment(monkeypatch):
    report = _report(clips=[{
        **_clip(_uncertainty="medium"),
        "_clip_id": "clip-a",
        "summary": "",
        "takeaways": [],
        "match_reason": "",
        "assessment": None,
    }])

    def fake_selection(*args, **kwargs):
        return report, G._Classification("uncertain", ("medium_uncertainty",)), []

    monkeypatch.setattr(G, "_run_selection_profile", fake_selection)
    monkeypatch.setattr(
        G, "_enrich_split",
        lambda *args, **kwargs: pytest.fail("uncertain boundaries must skip enrichment"),
    )
    result = G.run_segment_profile(_transcript(), {}, G.FLASH_SPLIT_PROFILE)
    assert result.classification == "uncertain"


def test_green_split_skips_synchronous_enrichment_by_default(monkeypatch):
    report = _report(clips=[{
        **_clip(),
        **_private_clip("clip-a", "alpha lesson"),
    }])

    monkeypatch.setattr(
        G,
        "_run_selection_profile",
        lambda *args, **kwargs: (report, G._Classification("green", ()), []),
    )
    monkeypatch.setattr(
        G, "_enrich_split",
        lambda *args, **kwargs: pytest.fail("default selector must not enrich synchronously"),
    )

    result = G.run_segment_profile(_transcript(), {}, G.FLASH_SPLIT_PROFILE)

    assert result.classification == "green"
    assert result.clips[0]["summary"] == ""


def test_short_zero_clip_split_skips_empty_enrichment_call(monkeypatch):
    report = G._Conversion(clips=[], proposed_count=0)

    def fake_selection(*args, **kwargs):
        return report, G._Classification("green", ()), []

    monkeypatch.setattr(G, "_run_selection_profile", fake_selection)
    monkeypatch.setattr(
        G, "_enrich_split",
        lambda *args, **kwargs: pytest.fail("zero clips must not trigger enrichment"),
    )
    result = G.run_segment_profile(_transcript(100.0), {}, G.FLASH_SPLIT_PROFILE)
    assert result.classification == "green"
    assert result.clips == []


def test_telemetry_cost_uses_recorded_usage_and_stays_internal(monkeypatch):
    events = []
    monkeypatch.setattr(G.config, "SEGMENT_HYBRID_PERCENT", 100.0)
    monkeypatch.setattr(
        G, "run_segment_profile",
        lambda transcript, settings, profile, **kwargs: _result(profile, "green", title="flash"),
    )
    result = G.segment_clips_detailed(
        _transcript(), {"_segment_telemetry": events.append},
        video_id="video", routing_mode="hybrid",
    )
    completed = next(event for event in events if event["event"] == "segment_completed")
    assert completed["estimated_cost_usd"] > 0
    assert completed["cost_per_accepted_clip_usd"] > 0
    assert "model" not in result.clips[0]
    assert "gemini" not in result.notes.lower()


def test_preview_flash_cost_uses_current_input_and_output_rates():
    assert G._model_cost({
        "model": "gemini-3-flash-preview",
        "prompt_tokens": 1_000_000,
        "candidate_tokens": 100_000,
        "thought_tokens": 50_000,
    }) == pytest.approx(0.95)


def test_flash_lite_failover_cost_uses_its_lower_rates():
    assert G._model_cost({
        "model": "gemini-3.1-flash-lite",
        "prompt_tokens": 1_000_000,
        "candidate_tokens": 100_000,
        "thought_tokens": 50_000,
    }) == pytest.approx(0.475)


def test_cancelled_worker_never_publishes_late_boundary_or_done_progress(monkeypatch):
    state = {"cancelled": False}
    progress = []

    def fake_run(transcript, settings, profile, **kwargs):
        state["cancelled"] = True
        return _result(profile, "invalid", title="cancelled", error="cancelled")

    monkeypatch.setattr(G, "run_segment_profile", fake_run)
    G.segment_clips_detailed(
        _transcript(), {"_segment_cancelled": lambda: state["cancelled"]},
        video_id="video", routing_mode="pro_only",
        progress=lambda fraction, message: progress.append((fraction, message)),
    )
    assert [fraction for fraction, _message in progress] == [0.1]


def test_clip_grounding_text_matches_the_selected_semantic_projection():
    segments = [{
        "start": 0.0,
        "end": 180.0,
        "text": "outside before alpha educational lesson closes cleanly outside after",
    }]
    words = [
        {"word": "outside", "start": 0.0, "end": 1.0},
        {"word": "before", "start": 1.0, "end": 2.0},
        {"word": "alpha", "start": 40.0, "end": 41.0},
        {"word": "educational", "start": 41.0, "end": 42.0},
        {"word": "lesson", "start": 42.0, "end": 43.0},
        {"word": "closes", "start": 60.0, "end": 61.0},
        {"word": "cleanly", "start": 61.0, "end": 62.0},
        {"word": "outside", "start": 120.0, "end": 121.0},
        {"word": "after", "start": 121.0, "end": 122.0},
    ]
    proposal = G._Topic(
        candidate_id="candidate-alpha",
        start_line=0, end_line=0,
        start_quote="alpha educational lesson", end_quote="closes cleanly",
        title="Alpha", learning_objective="Understand the alpha lesson.",
        facet="lesson", reason="A complete alpha lesson.",
        informativeness=0.9, topic_relevance=0.9,
        educational_importance=0.9, difficulty=0.5,
        directly_teaches_topic=True, substantive=True, factually_grounded=True,
        topic_evidence_quote="alpha educational lesson closes cleanly",
        self_contained=True, is_standalone=True,
        prerequisite_candidate_ids=[], uncertainty="low", uncertainty_reasons=[],
        summary="The alpha lesson closes cleanly.",
        takeaways=["The alpha lesson is taught.", "The lesson closes cleanly."],
        match_reason="The alpha lesson is directly relevant.",
        assessment={
            "prompt": "What closes?",
            "options": ["The lesson", "Sponsor", "Greeting", "Outro"],
            "correct_index": 0,
            "explanation": "The lesson closes cleanly.",
            "evidence_quote": "closes cleanly",
        },
    )
    report = G._plan_to_report(
        G._Plan(topics=[proposal]), segments, words,
        {"segment_fine_snap": True}, topic="alpha", require_enrichment=True,
    )
    assert report.clips[0]["_clip_text"] == (
        "alpha educational lesson closes cleanly"
    )
