from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.pipeline import gemini_segment as G


def _segs(n: int, step: float = 10.0) -> list[dict]:
    return [
        {
            "start": i * step,
            "end": (i + 1) * step,
            "text": f"line {i} teaches concept {i} and finishes end {i}",
        }
        for i in range(n)
    ]


def _words(segs: list[dict]) -> list[dict]:
    out: list[dict] = []
    for seg in segs:
        tokens = seg["text"].split()
        width = (seg["end"] - seg["start"] - 0.2) / len(tokens)
        for index, token in enumerate(tokens):
            start = seg["start"] + 0.1 + index * width
            out.append({"word": token, "start": start, "end": start + width})
    return out


def _assessment(line: int) -> G._AssessmentDraft:
    return G._AssessmentDraft(
        prompt="Which concept is taught?",
        options=[f"Concept {line}", "A sponsor", "A greeting", "An outro"],
        correct_index=0,
        explanation=f"The clip teaches concept {line}.",
        evidence_quote=f"teaches concept {line}",
    )


def _topic(title: str, start_line: int, end_line: int, **overrides) -> G._Topic:
    data = {
        "candidate_id": f"candidate-{start_line}-{end_line}",
        "title": title,
        "learning_objective": f"Understand {title}.",
        "start_line": start_line,
        "end_line": end_line,
        "start_quote": f"line {start_line}",
        "end_quote": f"end {end_line}",
        "facet": "concept",
        "reason": "Teaches the requested concept completely.",
        "informativeness": 0.9,
        "topic_relevance": 0.9,
        "educational_importance": 0.9,
        "difficulty": 0.5,
        "directly_teaches_topic": True,
        "substantive": True,
        "topic_evidence_quote": (
            f"line {start_line} teaches concept {start_line} and finishes"
        ),
        "self_contained": True,
        "is_standalone": True,
        "prerequisite_candidate_ids": [],
        "uncertainty": "low",
        "uncertainty_reasons": [],
        "summary": f"Line {start_line} teaches the concept and finishes it.",
        "takeaways": [f"Line {start_line} teaches concept {start_line}.", f"Line {end_line} finishes end {end_line}."],
        "match_reason": f"The concept is taught in line {start_line}.",
        "assessment": _assessment(start_line),
    }
    data.update(overrides)
    return G._Topic(**data)


def _plan(*triples) -> G._Plan:
    return G._Plan(topics=[_topic(title, start, end) for title, start, end in triples])


def _convert(plan: G._Plan, segs: list[dict], **settings) -> list[dict]:
    return G._plan_to_clips(
        plan, segs, _words(segs), {"segment_fine_snap": False, **settings},
    )


def test_maps_lines_to_chunk_times_after_required_alignment():
    segs = _segs(3)
    clips = _convert(_plan(("T1", 0, 1)), segs)
    assert len(clips) == 1
    assert (clips[0]["start"], clips[0]["end"]) == (0.0, 20.0)
    assert clips[0]["sequence_index"] == 1
    assert clips[0]["kind"] == "educational"
    assert clips[0]["cue_ids"] == ["cue-0", "cue-1"]
    assert clips[0]["selection_candidate_id"] == "candidate-0-1"
    assert clips[0]["learning_objective"] == "Understand T1."
    assert clips[0]["educational_importance"] == pytest.approx(0.9)
    assert clips[0]["boundary_confidence"] == 0.9
    assert clips[0]["is_standalone"] is True
    assert clips[0]["chain_id"] == ""
    assert clips[0]["chain_position"] == 0
    assert clips[0]["prerequisite_ids"] == []
    assert "cut_end" not in clips[0]


def test_selector_rejects_non_teaching_non_substantive_and_ungrounded_evidence():
    segs = _segs(4)
    plan = G._Plan(topics=[
        _topic("Not teaching", 0, 0, directly_teaches_topic=False),
        _topic("Filler", 1, 1, substantive=False),
        _topic(
            "Invented evidence",
            2,
            2,
            topic_evidence_quote="this exact teaching quote is not present",
        ),
        _topic("Useful", 3, 3),
    ])

    report = G._plan_to_report(
        plan, segs, _words(segs), {"segment_fine_snap": False},
    )

    assert [clip["title"] for clip in report.clips] == ["Useful"]
    assert report.rejected_reasons == [
        "proposal_0:does_not_directly_teach_topic",
        "proposal_1:not_substantive",
        "proposal_2:ungrounded_topic_evidence_quote",
    ]


@pytest.mark.parametrize(
    "quote,reason",
    [
        ("line 0 teaches concept", "proposal_0:invalid_topic_evidence_quote_length"),
        (
            "line 0 teaches concept 0 and finishes end 0 extra words beyond the source "
            "one two three four five six seven eight nine ten eleven twelve thirteen "
            "fourteen fifteen sixteen seventeen eighteen nineteen twenty twentyone "
            "twentytwo twentythree twentythree twentyfour twentyfive twentysix "
            "twentyseven twentyeight twentynine thirty thirtyone thirtytwo thirtythree",
            "proposal_0:invalid_topic_evidence_quote_length",
        ),
    ],
)
def test_topic_evidence_quote_requires_five_to_forty_words(quote, reason):
    segs = _segs(1)
    report = G._plan_to_report(
        G._Plan(topics=[_topic("Evidence", 0, 0, topic_evidence_quote=quote)]),
        segs,
        _words(segs),
        {"segment_fine_snap": False},
    )
    assert report.clips == []
    assert report.rejected_reasons == [reason]


def test_selector_metadata_preserves_explicit_prerequisite_chain():
    segs = _segs(3)
    setup = _topic(
        "Setup",
        0,
        0,
        candidate_id="setup",
        educational_importance=0.7,
    )
    application = _topic(
        "Application",
        1,
        1,
        candidate_id="application",
        educational_importance=0.95,
        is_standalone=False,
        prerequisite_candidate_ids=["setup"],
    )

    clips = _convert(G._Plan(topics=[setup, application]), segs)
    by_id = {clip["selection_candidate_id"]: clip for clip in clips}
    assert by_id["setup"]["chain_id"] == "chain:setup"
    assert by_id["setup"]["chain_position"] == 0
    assert by_id["application"]["chain_id"] == "chain:setup"
    assert by_id["application"]["chain_position"] == 1
    assert by_id["application"]["prerequisite_ids"] == ["setup"]


def test_prerequisite_bundle_refills_when_dependent_does_not_fit_limit():
    segs = _segs(3)
    setup = _topic(
        "Setup",
        0,
        0,
        candidate_id="setup",
        educational_importance=0.61,
    )
    dependent = _topic(
        "Dependent",
        2,
        2,
        candidate_id="dependent",
        educational_importance=0.99,
        is_standalone=False,
        prerequisite_candidate_ids=["setup"],
    )

    clips = _convert(G._Plan(topics=[setup, dependent]), segs, max_clips=1)
    assert [clip["selection_candidate_id"] for clip in clips] == ["setup"]


def test_clip_limit_uses_importance_before_chronology():
    segs = _segs(3)
    early = _topic(
        "Early",
        0,
        0,
        candidate_id="early",
        educational_importance=0.61,
        informativeness=0.8,
        topic_relevance=0.8,
    )
    later = _topic(
        "Later",
        2,
        2,
        candidate_id="later",
        educational_importance=0.99,
        informativeness=0.8,
        topic_relevance=0.8,
    )

    clips = _convert(G._Plan(topics=[early, later]), segs, max_clips=1)
    assert [clip["selection_candidate_id"] for clip in clips] == ["later"]


def test_post_acceptance_enrichment_is_grounded_batched_and_quiz_free(monkeypatch):
    captured = {}
    plan = G._CardEnrichmentPlan(items=[
        G._CardEnrichmentItem(
            clip_id="reel-1",
            summary="Gradients reduce loss during training.",
            takeaways=[
                "Gradients reduce loss.",
                "Training follows the gradient direction.",
            ],
            match_reason="Alpha uses gradients to reduce loss.",
        )
    ])

    def fake_call(system, user, schema, **kwargs):
        captured.update({"system": system, "user": user, "schema": schema, **kwargs})
        return plan, {"operation": kwargs["operation"]}

    monkeypatch.setattr(G, "_call_model", fake_call)
    enriched, calls = G.enrich_accepted_clips(
        [
            {
                "clip_id": "reel-1",
                "title": "Gradient descent",
                "learning_objective": "Understand loss reduction.",
                "text": "Alpha training follows the gradient direction and gradients reduce loss.",
            },
            {"clip_id": "reel-2", "text": "Second excerpt."},
            {"clip_id": "reel-3", "text": "Third excerpt."},
            {"clip_id": "reel-4", "text": "Must not enter this batch."},
        ],
        topic="alpha",
    )

    assert set(enriched) == {"reel-1"}
    assert calls == [{"operation": "flash_grounded_enrichment"}]
    assert captured["schema"] is G._CardEnrichmentPlan
    assert captured["thinking_level"] == "low"
    assert captured["max_output_tokens"] == 2_048
    assert "reel-4" not in captured["user"]
    assert "quiz" in captured["system"].casefold()


@pytest.mark.parametrize("start,end", [(5, 99), (2, 1)])
def test_bad_line_indices_are_rejected_instead_of_clamped(start, end):
    segs = _segs(3)
    assert _convert(G._Plan(topics=[_topic("T", start, end)]), segs) == []


def test_contextual_overlap_below_duplicate_threshold_is_preserved():
    segs = _segs(4)
    clips = _convert(_plan(("A", 0, 2), ("B", 1, 3)), segs)
    assert [(clip["start"], clip["end"]) for clip in clips] == [(0.0, 30.0), (10.0, 40.0)]


def test_span_covering_eighty_percent_of_shorter_is_deduplicated():
    segs = _segs(4)
    clips = _convert(_plan(("A", 0, 2), ("B", 0, 3)), segs)
    assert [clip["title"] for clip in clips] == ["A"]


def test_cue_quotes_are_authoritative_when_word_timing_is_missing_or_untrusted():
    segs = _segs(2)
    plan = _plan(("T", 0, 1))
    assert len(G._plan_to_clips(plan, segs, [], {"segment_fine_snap": False})) == 1
    bad_words = [{"word": "unrelated", "start": 0.0, "end": 1.0}]
    assert len(G._plan_to_clips(plan, segs, bad_words, {"segment_fine_snap": False})) == 1


def test_quote_on_a_different_declared_line_is_rejected():
    segs = _segs(2)
    plan = G._Plan(topics=[_topic("T", 0, 1, start_quote="line 1")])
    assert _convert(plan, segs) == []


def test_locate_quote_finds_start_and_latest_end():
    words = [
        {"word": "hello", "start": 1.0, "end": 1.5},
        {"word": "world", "start": 1.5, "end": 2.0},
        {"word": "hello", "start": 2.0, "end": 2.5},
        {"word": "world", "start": 2.5, "end": 3.0},
    ]
    assert G._locate_quote(words, "hello world", 0.0, 4.0, "start") == pytest.approx(1.0)
    assert G._locate_quote(words, "hello world", 0.0, 4.0, "end") == pytest.approx(3.0)
    assert G._locate_quote(words, "zzz qqq", 0.0, 4.0, "start") is None


def test_interpolated_words_never_tighten_physical_boundaries():
    segs = [{
        "start": 0.0,
        "end": 100.0,
        "text": "intro then the real topic starts here and ends now",
    }]
    words = [
        {"word": "intro", "start": 0.0, "end": 5.0, "timing_source": "interpolated"},
        {"word": "topic", "start": 40.0, "end": 42.0, "timing_source": "interpolated"},
        {"word": "starts", "start": 42.0, "end": 44.0, "timing_source": "interpolated"},
        {"word": "ends", "start": 90.0, "end": 92.0, "timing_source": "interpolated"},
        {"word": "now", "start": 92.0, "end": 95.0, "timing_source": "interpolated"},
    ]
    plan = G._Plan(topics=[_topic(
        "T", 0, 0,
        start_quote="topic starts",
        end_quote="ends now",
        topic_evidence_quote="intro then the real topic starts here",
        summary="The real topic starts and ends now.",
        takeaways=["The topic starts here.", "The topic ends now."],
        match_reason="The topic starts in this explanation.",
        assessment=G._AssessmentDraft(
            prompt="What starts?",
            options=["The topic", "A sponsor", "A greeting", "An outro"],
            correct_index=0,
            explanation="The topic starts here.",
            evidence_quote="topic starts",
        ),
    )])
    clips = G._plan_to_clips(plan, segs, words, {"segment_fine_snap": True})
    assert (clips[0]["start"], clips[0]["end"]) == pytest.approx((0.0, 100.0))


def test_cue_boundaries_get_gap_clamped_three_hundred_ms_padding():
    segs = [
        {"start": 0.0, "end": 9.0, "text": "before context"},
        {"start": 10.0, "end": 20.0, "text": "topic starts and ends now"},
        {"start": 21.0, "end": 30.0, "text": "after context"},
    ]
    words = [
        {"word": "topic", "start": 10.2, "end": 10.5},
        {"word": "starts", "start": 10.5, "end": 10.9},
        {"word": "ends", "start": 19.0, "end": 19.3},
        {"word": "now", "start": 19.3, "end": 19.6},
    ]
    proposal = _topic(
        "T", 1, 1,
        start_quote="topic starts",
        end_quote="ends now",
        topic_evidence_quote="topic starts and ends now",
        summary="The topic starts and ends now.",
        takeaways=["The topic starts.", "The topic ends now."],
        match_reason="The topic is explained here.",
        assessment=G._AssessmentDraft(
            prompt="What ends?",
            options=["The topic", "A sponsor", "A greeting", "An outro"],
            correct_index=0,
            explanation="The topic ends now.",
            evidence_quote="ends now",
        ),
    )
    clips = G._plan_to_clips(
        G._Plan(topics=[proposal]), segs, words, {"segment_fine_snap": True},
    )
    assert (clips[0]["start"], clips[0]["end"]) == (9.7, 20.3)


def test_alignment_cannot_select_repeated_quote_from_adjacent_line():
    words = [
        {"word": "repeat", "start": 9.0, "end": 9.2},
        {"word": "phrase", "start": 9.2, "end": 9.4},
        {"word": "repeat", "start": 10.2, "end": 10.4},
        {"word": "phrase", "start": 10.4, "end": 10.6},
        {"word": "ending", "start": 19.0, "end": 19.2},
        {"word": "phrase", "start": 19.2, "end": 19.4},
        {"word": "ending", "start": 20.2, "end": 20.4},
        {"word": "phrase", "start": 20.4, "end": 20.6},
    ]
    assert G._locate_quote(words, "repeat phrase", 10.0, 20.0, "start") == 10.2
    assert G._locate_quote(words, "ending phrase", 10.0, 20.0, "end") == 19.4


def test_chemistry_fast_path_closes_two_cues_of_missing_setup():
    segs = [
        {
            "start": 0.0,
            "end": 7.9,
            "text": "Let's name the ionic compound MgBr2.",
        },
        {
            "start": 8.2,
            "end": 16.0,
            "text": "And then Mg stands for magnesium while Br means bromide.",
        },
        {
            "start": 16.4,
            "end": 22.0,
            "text": "So the answer is magnesium bromide.",
        },
        {
            "start": 22.8,
            "end": 30.0,
            "text": "Next we will name potassium oxide.",
        },
    ]
    proposal = _topic(
        "Magnesium bromide",
        2,
        2,
        start_quote="So the answer",
        end_quote="magnesium bromide",
        topic_evidence_quote="So the answer is magnesium bromide",
    )

    [clip] = G._plan_to_clips(G._Plan(topics=[proposal]), segs, [], {})

    assert clip["start"] == 0.0
    assert clip["end"] == 22.3
    assert clip["title"] == "Magnesium bromide"


def test_weak_conjunction_end_expands_to_the_completed_answer():
    segs = [
        {"start": 0.0, "end": 6.0, "text": "We combine magnesium and"},
        {"start": 6.2, "end": 12.0, "text": "bromide to form magnesium bromide."},
        {"start": 12.8, "end": 20.0, "text": "Now consider potassium oxide."},
    ]
    proposal = _topic(
        "Formation",
        0,
        0,
        start_quote="We combine",
        end_quote="magnesium and",
        topic_evidence_quote="We combine magnesium and bromide",
    )

    [clip] = G._plan_to_clips(G._Plan(topics=[proposal]), segs, [], {})

    assert clip["end"] == 12.3


def test_global_quality_limit_ranks_stronger_clip_over_valid_low_scored_clip():
    segs = _segs(5)
    plan = G._Plan(topics=[
        _topic(
            "early .01",
            0,
            0,
            informativeness=0.01,
            topic_relevance=0.01,
            educational_importance=0.01,
        ),
        _topic(
            "late .99",
            4,
            4,
            informativeness=0.99,
            topic_relevance=0.99,
        ),
    ])

    clips = G._plan_to_clips(plan, segs, [], {"max_clips": 1})

    assert [clip["title"] for clip in clips] == ["late .99"]


def test_medium_uncertainty_is_accepted_but_high_uncertainty_is_rejected():
    segs = _segs(3)
    plan = G._Plan(topics=[
        _topic(
            "ambiguous",
            0,
            0,
            uncertainty="medium",
            uncertainty_reasons=["boundary_ambiguous"],
        ),
        _topic("clean", 2, 2),
        _topic(
            "incomplete",
            1,
            1,
            uncertainty="high",
            uncertainty_reasons=["incomplete_context"],
        ),
    ])

    report = G._plan_to_report(plan, segs, [], {})
    classification = G._classify_flash(
        report, segs, "", enrichment_required=False,
    )

    assert [clip["title"] for clip in report.clips] == ["ambiguous", "clean"]
    assert report.clips[0]["uncertainty"] == "medium"
    assert report.clips[0]["uncertainty_reasons"] == ["boundary_ambiguous"]
    assert "proposal_0:medium_uncertainty" not in report.rejected_reasons
    assert "proposal_2:high_uncertainty" in report.rejected_reasons
    assert classification == G._Classification("green", ())


def test_medium_uncertainty_ranks_below_equally_scored_clean_clip():
    segs = _segs(2)
    plan = G._Plan(topics=[
        _topic(
            "ambiguous",
            0,
            0,
            uncertainty="medium",
            uncertainty_reasons=["boundary_ambiguous"],
        ),
        _topic("clean", 1, 1),
    ])

    clips = G._plan_to_clips(plan, segs, [], {"max_clips": 1})

    assert [clip["title"] for clip in clips] == ["clean"]


def test_production_flash_is_compact_global_boundary_first():
    system, user = G._boundary_prompts("[0] 00:00 lesson", 1, "chemistry")
    prompt = f"{system}\n{user}".casefold()

    assert G.PRODUCTION_FLASH_PROFILE == G.FLASH_SPLIT_PROFILE
    assert "at most 150 seconds" in prompt
    assert "20 to 90 seconds" in prompt
    assert "low or medium uncertainty" in prompt
    assert "omit only high-uncertainty" in prompt
    assert G._BOUNDARY_OUTPUT_TOKENS == 8192
    assert "globally" in prompt
    assert "do not favor the beginning" in prompt
    assert "at most 8" in prompt
    assert "learning details and assessments are generated later" in prompt


def test_production_flash_receives_requested_duration_contract():
    _system, user = G._boundary_prompts(
        "[0] 00:00 lesson",
        1,
        "chemistry",
        target_sec=40,
        target_min_sec=10,
        target_max_sec=55,
    )

    assert "requested range is 10 to 55 seconds" in user
    assert "40-second target" in user
    assert "MUST be at most 55 seconds" in user
    assert "minimum is a preference, not a rejection rule" in user


def test_budget_is_reserved_once_before_dispatch_and_provider_retry_is_disabled(monkeypatch):
    order = []
    payload = {}

    def reserve(**kwargs):
        order.append("reserve")
        payload.update(kwargs)

    def generate(*args, **kwargs):
        order.append("dispatch")
        assert kwargs["max_retries"] == 0
        return SimpleNamespace(text='{"topics": []}', telemetry={})

    monkeypatch.setattr("backend.gemini_client.generate_json_v3", generate)
    parsed, _ = G._call_model(
        "system",
        "user",
        G._BoundaryPlan,
        model="gemini-3.5-flash",
        thinking_level="medium",
        max_output_tokens=4096,
        timeout_s=45.0,
        deadline_monotonic=10_000.0,
        operation="flash_boundary_selector",
        prompt_version=G.FLASH_SPLIT_PROFILE,
        cancelled=None,
        budget_reserve=reserve,
    )

    assert parsed.topics == []
    assert order == ["reserve", "dispatch"]
    assert payload == {
        "operation": "flash_boundary_selector",
        "model": "gemini-3.5-flash",
        "max_output_tokens": 4096,
        "prompt_text": "system\n\nuser",
        "estimated_input_tokens": 3,
    }


def test_invalid_partial_flash_is_never_shipped_even_when_legacy_setting_is_true(monkeypatch):
    events = []
    partial = G.SegmentResult(
        clips=[{"title": "unsafe partial"}],
        notes="partial",
        route=G.FLASH_SPLIT_PROFILE,
        classification="invalid",
        classification_reasons=["proposal_1:unresolved_weak_end"],
        proposed_count=2,
        accepted_count=1,
        rejection_reasons=["proposal_1:unresolved_weak_end"],
    )
    pro = G.SegmentResult(
        clips=[{"title": "safe pro"}],
        notes="pro",
        route=G.PRODUCTION_PRO_PROFILE,
        classification="green",
        proposed_count=1,
        accepted_count=1,
        rejection_reasons=["proposal_0:bad_start_quote"],
    )
    monkeypatch.setattr(G.config, "SEGMENT_HYBRID_PERCENT", 100.0)
    monkeypatch.setattr(G, "_flash_disable_reason", lambda: None)
    monkeypatch.setattr(G, "run_segment_profile", lambda *args, **kwargs: partial)
    monkeypatch.setattr(G, "_authoritative_pro", lambda *args, **kwargs: pro)

    result = G.segment_clips_detailed(
        {"segments": _segs(2)},
        {
            "segment_accept_partial_flash": True,
            "_segment_telemetry": events.append,
        },
        video_id="dQw4w9WgXcQ",
        routing_mode="hybrid",
    )

    assert [clip["title"] for clip in result.clips] == ["safe pro"]
    assert result.route == "hybrid_pro_fallback"
    assert result.rejection_reasons == [
        "proposal_1:unresolved_weak_end",
        "proposal_0:bad_start_quote",
    ]
    completed = next(event for event in events if event["event"] == "segment_completed")
    assert completed["rejection_reasons"] == result.rejection_reasons


def test_optional_enrichment_failure_keeps_valid_boundary_without_pro_retry(monkeypatch):
    calls = 0

    def fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("optional metadata unavailable")

    monkeypatch.setattr(G, "_call_model", fail)
    clip = {
        "_clip_id": "clip-001-0-0",
        "_clip_text": "A complete chemistry explanation.",
        "summary": "",
        "takeaways": [],
        "match_reason": "",
        "assessment": None,
    }

    clips, model_calls, fallback_reasons, configuration_error = G._enrich_split(
        [clip],
        "chemistry",
        {},
        deadline=10_000.0,
        cancelled=None,
    )

    assert calls == 1
    assert clips == [clip]
    assert model_calls == []
    assert fallback_reasons == []
    assert configuration_error is None
