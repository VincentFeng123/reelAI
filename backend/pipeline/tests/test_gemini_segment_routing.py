from __future__ import annotations

import hashlib
import threading
import time

import pytest

from backend import gemini_client as GC
from backend.pipeline import gemini_segment as G


def _segments(duration: float = 100.0) -> list[dict]:
    return [{"start": 0.0, "end": duration, "text": "alpha lesson concept closes cleanly"}]


def test_provider_schemas_avoid_unsupported_additional_properties():
    for schema in (
        G._Plan,
        G._BoundaryPlan,
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
    (_report(clips=[_clip(informativeness=0.61)]), "alpha"),
    (_report(clips=[_clip(end=150.0)]), "alpha"),
    (_report(near_duplicate=True), "alpha"),
    (_report(), "beta"),
    (_report(rejected_reasons=["proposal_0:bad_index"]), "alpha"),
])
def test_flash_classification_keeps_any_independently_valid_candidate(report, topic):
    classified = G._classify_flash(report, _segments(), topic, enrichment_required=False)
    assert classified == G._Classification("green", ())


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


@pytest.mark.parametrize(
    "profile,expected",
    [
        (G.PRODUCTION_PRO_PROFILE,
         ("high", 24_576, 90.0, "pro_authoritative", "gemini-3.1-pro-preview")),
        (G.CORRECTED_PRO_PROFILE,
         ("high", 24_576, 90.0, "pro_fallback", "gemini-3.1-pro-preview")),
        (G.FLASH_SINGLE_PROFILE,
         ("medium", 24_576, 45.0, "flash_single_candidate", "gemini-3.5-flash")),
        (G.FLASH_SPLIT_PROFILE,
         ("medium", 4_096, 45.0, "flash_boundary_selector", "gemini-3.5-flash")),
        (G.PRO_BOUNDARY_PROFILE,
         ("high", 4_096, 90.0, "pro_fallback", "gemini-3.1-pro-preview")),
    ],
)
def test_profile_operation_settings_are_wired_to_client(monkeypatch, profile, expected):
    captured = {}

    def fake_call(system, user, schema, **kwargs):
        captured.update(kwargs)
        plan = G._LegacyPlan(topics=[]) if schema is G._LegacyPlan else schema(topics=[])
        return plan, {}

    monkeypatch.setattr(G, "_call_model", fake_call)
    G._run_selection_profile(
        profile, _transcript(), "", {},
        deadline=time.monotonic() + 10, cancelled=None,
    )
    level, cap, timeout, operation, default_model = expected
    expected_model = (G.config.SEGMENT_PRO_MODEL if "pro" in default_model
                      else G.config.SEGMENT_FLASH_MODEL)
    assert (
        captured["thinking_level"], captured["max_output_tokens"],
        captured["timeout_s"], captured["operation"], captured["model"],
    ) == (level, cap, timeout, operation, expected_model)


def test_production_boundary_selector_caps_global_candidates_at_eight(monkeypatch):
    segments = [
        {
            "cue_id": f"c{index}",
            "start": index * 10.0,
            "end": (index + 1) * 10.0,
            "text": f"line {index} teaches concept {index} and finishes end {index}",
        }
        for index in range(9)
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
            informativeness=0.80 + index * 0.02,
            topic_relevance=0.80 + index * 0.02,
            educational_importance=0.80 + index * 0.02,
            difficulty=0.5,
            self_contained=True,
            is_standalone=True,
            prerequisite_candidate_ids=[],
            uncertainty="low",
            uncertainty_reasons=[],
        )
        for index in range(9)
    ]
    monkeypatch.setattr(
        G,
        "_call_model",
        lambda *args, **kwargs: (G._BoundaryPlan(topics=topics), {}),
    )

    report, classification, _ = G._run_selection_profile(
        G.PRODUCTION_FLASH_PROFILE,
        {"segments": segments, "words": [], "source": "supadata"},
        "",
        {"max_clips": 40},
        deadline=time.monotonic() + 10,
        cancelled=None,
    )

    assert classification.status == "green"
    assert len(report.clips) == 8
    assert "T0" not in {clip["title"] for clip in report.clips}
    assert "T8" in {clip["title"] for clip in report.clips}


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


def test_clip_grounding_text_matches_the_delivered_complete_cue_window():
    segments = [{
        "start": 0.0,
        "end": 180.0,
        "text": "outside before alpha lesson closes cleanly outside after",
    }]
    words = [
        {"word": "outside", "start": 0.0, "end": 1.0},
        {"word": "before", "start": 1.0, "end": 2.0},
        {"word": "alpha", "start": 40.0, "end": 41.0},
        {"word": "lesson", "start": 41.0, "end": 42.0},
        {"word": "closes", "start": 60.0, "end": 61.0},
        {"word": "cleanly", "start": 61.0, "end": 62.0},
        {"word": "outside", "start": 120.0, "end": 121.0},
        {"word": "after", "start": 121.0, "end": 122.0},
    ]
    proposal = G._Topic(
        candidate_id="candidate-alpha",
        start_line=0, end_line=0,
        start_quote="alpha lesson", end_quote="closes cleanly",
        title="Alpha", learning_objective="Understand the alpha lesson.",
        facet="lesson", reason="A complete alpha lesson.",
        informativeness=0.9, topic_relevance=0.9,
        educational_importance=0.9, difficulty=0.5,
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
    assert report.clips[0]["_clip_text"] == segments[0]["text"]
