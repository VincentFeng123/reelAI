from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
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
        "factually_grounded": True,
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


def test_short_grounded_topic_evidence_is_extended_to_five_words():
    segs = _segs(1)
    report = G._plan_to_report(
        G._Plan(topics=[_topic(
            "Evidence",
            0,
            0,
            topic_evidence_quote="line 0 teaches concept",
        )]),
        segs,
        _words(segs),
        {"segment_fine_snap": False},
    )

    assert len(report.clips) == 1
    assert report.clips[0]["topic_evidence_quote"] == "line 0 teaches concept 0"


def test_oversized_ungrounded_topic_evidence_is_rejected():
    segs = _segs(1)
    quote = (
        "line 0 teaches concept 0 and finishes end 0 extra words beyond the source "
        "one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen twenty twentyone "
        "twentytwo twentythree twentythree twentyfour twentyfive twentysix "
        "twentyseven twentyeight twentynine thirty thirtyone thirtytwo thirtythree"
    )
    report = G._plan_to_report(
        G._Plan(topics=[_topic("Evidence", 0, 0, topic_evidence_quote=quote)]),
        segs,
        _words(segs),
        {"segment_fine_snap": False},
    )

    assert report.clips == []
    assert report.rejected_reasons == ["proposal_0:ungrounded_topic_evidence_quote"]


def test_near_exact_topic_evidence_is_rejected_instead_of_rewritten():
    segments = [{
        "start": 0.0,
        "end": 12.0,
        "text": "Mitochondria transform chemical energy from nutrients into ATP for cells.",
    }]
    proposal = _topic(
        "Mitochondria",
        0,
        0,
        start_quote="Mitochondria transform",
        end_quote="ATP for cells",
        topic_evidence_quote=(
            "Mitochondria transform chemical energy in nutrients into ATP for cells"
        ),
    )

    report = G._plan_to_report(
        G._BoundaryPlan(topics=[proposal]),
        segments,
        [],
        {},
        topic="mitochondria",
    )

    assert report.clips == []
    assert report.rejected_reasons == ["proposal_0:ungrounded_topic_evidence_quote"]


def test_near_exact_topic_evidence_cannot_invert_negation():
    text = "Vaccination does increase immune memory after controlled antigen exposure."
    proposal = _topic(
        "Immune memory",
        0,
        0,
        start_quote="Vaccination does increase",
        end_quote="controlled antigen exposure",
        topic_evidence_quote=(
            "Vaccination does not increase immune memory after controlled antigen exposure"
        ),
    )

    report = G._plan_to_report(
        G._BoundaryPlan(topics=[proposal]),
        [{"start": 0.0, "end": 12.0, "text": text}],
        [],
        {},
        topic="vaccination",
    )

    assert report.clips == []
    assert report.rejected_reasons == ["proposal_0:ungrounded_topic_evidence_quote"]


def test_selector_rejects_clip_that_requires_a_separate_prerequisite():
    segs = _segs(3)
    setup = _topic(
        "Setup",
        0,
        0,
        candidate_id="setup",
        educational_importance=0.75,
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

    report = G._plan_to_report(
        G._Plan(topics=[setup, application]), segs, _words(segs), {},
    )
    assert [clip["selection_candidate_id"] for clip in report.clips] == ["setup"]
    assert report.rejected_reasons == ["proposal_1:not_standalone"]


def test_nonstandalone_candidate_cannot_displace_standalone_candidate():
    segs = _segs(3)
    setup = _topic(
        "Setup",
        0,
        0,
        candidate_id="setup",
        educational_importance=0.75,
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


def test_clip_limit_uses_overall_quality_before_stable_chronology():
    segs = _segs(3)
    early = _topic(
        "Early",
        0,
        0,
        candidate_id="early",
        educational_importance=0.75,
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


def test_shared_cue_overlap_is_deduplicated_even_below_eighty_percent():
    segs = _segs(4)
    clips = _convert(_plan(("A", 0, 2), ("B", 1, 3)), segs)
    assert [(clip["start"], clip["end"]) for clip in clips] == [(0.0, 30.0)]


def test_non_overlapping_distinct_facets_from_one_source_are_preserved():
    segs = _segs(4)
    clips = _convert(_plan(("Alpha", 0, 1), ("Beta", 2, 3)), segs)

    assert [clip["title"] for clip in clips] == ["Alpha", "Beta"]


def test_shared_cue_overlap_keeps_the_higher_quality_candidate():
    segs = _segs(4)
    plan = G._Plan(topics=[
        _topic(
            "Lower quality",
            0,
            2,
            candidate_id="lower",
            informativeness=0.76,
            topic_relevance=0.8,
            educational_importance=0.77,
        ),
        _topic(
            "Higher quality",
            1,
            3,
            candidate_id="higher",
            informativeness=0.94,
            topic_relevance=0.92,
            educational_importance=0.93,
        ),
    ])

    clips = _convert(plan, segs)

    assert [clip["selection_candidate_id"] for clip in clips] == ["higher"]


def test_span_covering_eighty_percent_of_shorter_is_deduplicated():
    segs = _segs(4)
    clips = _convert(_plan(("A", 0, 2), ("B", 0, 3)), segs)
    assert [clip["title"] for clip in clips] == ["A"]


def test_one_shared_topic_token_does_not_collapse_distinct_facets():
    first = {"learning_objective": "Photosynthesis", "facet": ""}
    second = {"learning_objective": "Photosynthesis", "facet": ""}

    assert not G._semantic_restatement(first, second)


def test_two_token_reworded_objectives_are_semantic_restatements():
    first = {
        "learning_objective": "Explain photosynthesis light reactions",
        "facet": "energy capture",
    }
    second = {
        "learning_objective": "Understand photosynthesis light reactions",
        "facet": "energy capture",
    }

    assert G._semantic_restatement(first, second)


def test_cue_quotes_are_authoritative_when_word_timing_is_missing_or_untrusted():
    segs = _segs(2)
    plan = _plan(("T", 0, 1))
    assert len(G._plan_to_clips(plan, segs, [], {"segment_fine_snap": False})) == 1
    bad_words = [{"word": "unrelated", "start": 0.0, "end": 1.0}]
    assert len(G._plan_to_clips(plan, segs, bad_words, {"segment_fine_snap": False})) == 1


def test_quote_on_a_different_declared_line_uses_declared_cue_fallback():
    segs = _segs(2)
    plan = G._Plan(topics=[_topic("T", 0, 1, start_quote="line 1")])
    clips = _convert(plan, segs)
    assert len(clips) == 1
    assert (clips[0]["start"], clips[0]["end"]) == (0.0, 20.0)


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
        topic_evidence_quote="topic starts here and ends now",
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


def test_high_uncertainty_with_only_boundary_reasons_is_retained():
    segs = _segs(1)
    report = G._plan_to_report(
        G._Plan(topics=[
            _topic(
                "boundary fallback",
                0,
                0,
                uncertainty="high",
                uncertainty_reasons=["boundary_ambiguous", "overlap_risk"],
            ),
        ]),
        segs,
        [],
        {},
    )

    assert [clip["title"] for clip in report.clips] == ["boundary fallback"]
    assert report.clips[0]["uncertainty"] == "high"
    assert report.clips[0]["uncertainty_reasons"] == [
        "boundary_ambiguous",
        "overlap_risk",
    ]
    assert report.clips[0]["_boundary_fallback_reasons"] == [
        "model_boundary_ambiguous",
        "model_overlap_risk",
    ]
    assert "proposal_0:high_uncertainty" not in report.rejected_reasons


@pytest.mark.parametrize("reason", ["topic_ambiguous", "incomplete_context"])
def test_high_topic_or_context_uncertainty_is_rejected(reason):
    report = G._plan_to_report(
        G._Plan(topics=[
            _topic(
                "content uncertainty",
                0,
                0,
                uncertainty="high",
                uncertainty_reasons=[reason],
            ),
        ]),
        _segs(1),
        [],
        {},
    )

    assert report.clips == []
    assert report.rejected_reasons == ["proposal_0:high_uncertainty"]


def test_equal_quality_ranking_is_stable_and_not_duration_or_uncertainty_shaped():
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

    assert [clip["title"] for clip in clips] == ["ambiguous"]


def test_production_flash_is_compact_exhaustive_boundary_first():
    system, user = G._boundary_prompts("[0] 00:00 lesson", 1, "chemistry")
    prompt = f"{system}\n{user}".casefold()

    assert G.PRODUCTION_FLASH_PROFILE == G.FLASH_SPLIT_PROFILE
    assert "duration is never a selection criterion" in prompt
    assert (
        "informativeness, topic_relevance, and educational_importance\n"
        "  are each at least 0.75"
    ) in prompt
    assert "let deterministic post-processing refine it" in prompt
    assert (
        "never omit a substantive grounded unit solely because its boundary is uncertain"
        in prompt
    )
    assert G._BOUNDARY_OUTPUT_TOKENS == 6_000
    assert "scan the whole transcript from first to last" in prompt
    assert "every distinct" in prompt
    assert "return every distinct qualifying moment" in prompt
    assert "arbitrary count" in prompt
    assert "up to 40" in prompt
    assert "never add filler or incomplete material" in prompt
    assert "difficulty is metadata, not an eligibility filter" in prompt
    assert "learning details and assessments are generated later" in prompt


def test_production_flash_has_a_bounded_latency_tail():
    assert G._TOTAL_DEADLINE_S == 36.0
    assert G._FLASH_BOUNDARY_TIMEOUT_S == 28.0
    assert G._FLASH_BOUNDARY_TIMEOUT_S < G._TOTAL_DEADLINE_S


def test_production_flash_selector_allows_one_bounded_transport_retry(monkeypatch):
    dispatched = []

    def call_model(*args, **kwargs):
        dispatched.append(kwargs)
        return G._BoundaryPlan(topics=[]), {}

    monkeypatch.setattr(G, "_call_model", call_model)

    G._run_selection_profile(
        G.PRODUCTION_FLASH_PROFILE,
        {"segments": _segs(1), "words": []},
        "chemistry",
        {},
        deadline=10_000.0,
        cancelled=None,
    )

    assert len(dispatched) == 1
    assert dispatched[0]["operation"] == "flash_boundary_selector"
    assert dispatched[0]["max_retries"] == 1
    assert dispatched[0]["failover_model"] == G.config.SEGMENT_FLASH_FALLBACK_MODEL


def test_production_flash_has_no_requested_duration_contract():
    system, user = G._boundary_prompts(
        "[0] 00:00 lesson", 1, "chemistry",
    )
    prompt = f"{system}\n{user}"

    assert "requested 10 to 55 second range" not in prompt
    assert "40-second target" not in prompt
    assert "180-second safety ceiling" not in prompt
    assert "regardless of its duration" in prompt


def test_compound_topic_allows_grounded_related_facets():
    _system, user = G._boundary_prompts(
        "[0] 00:00 lesson",
        1,
        "renormalization group in quantum chromodynamics",
    )

    assert "multiple linked ideas" in user
    assert "useful prerequisite facet is relevant" in user
    assert "unrelated domain is not enough" in user
    assert (
        "shared vocabulary, a loose analogy, or general systems thinking alone"
        in user.casefold()
    )
    assert (
        "clear educational connection to the exact requested topic" in user.casefold()
    )
    assert "even if it is not strictly required background" in user.casefold()


def test_budget_is_reserved_once_and_default_call_allows_one_transient_retry(monkeypatch):
    order = []
    payload = {}

    def reserve(**kwargs):
        order.append("reserve")
        payload.update(kwargs)

    def generate(*args, **kwargs):
        order.append("dispatch")
        assert kwargs["max_retries"] == 1
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
        deadline_monotonic=time.monotonic() + 10.0,
        operation="flash_boundary_selector",
        prompt_version=G.FLASH_SPLIT_PROFILE,
        cancelled=None,
        budget_reserve=reserve,
    )

    assert parsed.topics == []
    assert order == ["reserve", "dispatch"]
    deadline_monotonic = payload.pop("deadline_monotonic")
    assert isinstance(deadline_monotonic, float)
    assert deadline_monotonic > time.monotonic()
    assert payload == {
        "operation": "flash_boundary_selector",
        "model": "gemini-3.5-flash",
        "max_output_tokens": 4096,
        "prompt_text": "system\n\nuser",
        "estimated_input_tokens": 1_004,
        "cancelled": None,
    }


def test_selector_dispatches_are_capped_process_wide_at_three(monkeypatch):
    slots = threading.BoundedSemaphore(G._SELECTOR_CALL_LIMIT)
    monkeypatch.setattr(G, "_selector_call_slots", slots)
    lock = threading.Lock()
    release = threading.Event()
    saturated = threading.Event()
    state = {"active": 0, "maximum": 0}

    def generate(*args, **kwargs):
        del args, kwargs
        with lock:
            state["active"] += 1
            state["maximum"] = max(state["maximum"], state["active"])
            if state["active"] == G._SELECTOR_CALL_LIMIT:
                saturated.set()
        try:
            assert release.wait(timeout=2)
            return SimpleNamespace(text='{"topics": []}', telemetry={})
        finally:
            with lock:
                state["active"] -= 1

    monkeypatch.setattr("backend.gemini_client.generate_json_v3", generate)

    def dispatch():
        return G._call_model(
            "system",
            "user",
            G._BoundaryPlan,
            model="gemini-3.5-flash",
            thinking_level="low",
            max_output_tokens=4096,
            timeout_s=45.0,
            deadline_monotonic=time.monotonic() + 3.0,
            operation="flash_boundary_selector",
            prompt_version=G.FLASH_SPLIT_PROFILE,
            cancelled=None,
        )

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(dispatch) for _ in range(6)]
        assert saturated.wait(timeout=1)
        with lock:
            assert state["active"] == G._SELECTOR_CALL_LIMIT
            assert state["maximum"] == G._SELECTOR_CALL_LIMIT
        release.set()
        for future in futures:
            parsed, _telemetry = future.result(timeout=2)
            assert parsed.topics == []

    assert state["maximum"] == G._SELECTOR_CALL_LIMIT


@pytest.mark.parametrize(
    ("abort_kind", "expected_error_type"),
    [
        ("cancel", "GeminiCancelledError"),
        ("deadline", "GeminiDeadlineExceededError"),
    ],
)
def test_selector_capacity_wait_exits_without_dispatch(
    monkeypatch,
    abort_kind,
    expected_error_type,
):
    slots = threading.BoundedSemaphore(G._SELECTOR_CALL_LIMIT)
    for _ in range(G._SELECTOR_CALL_LIMIT):
        assert slots.acquire(blocking=False)
    monkeypatch.setattr(G, "_selector_call_slots", slots)
    dispatched = []
    monkeypatch.setattr(
        "backend.gemini_client.generate_json_v3",
        lambda *args, **kwargs: dispatched.append((args, kwargs)),
    )
    cancelled = threading.Event()
    timer = None
    deadline = time.monotonic() + 0.1
    if abort_kind == "cancel":
        deadline = time.monotonic() + 2.0
        timer = threading.Timer(0.075, cancelled.set)
        timer.start()

    try:
        with pytest.raises(G._ModelCallError) as exc_info:
            G._call_model(
                "system",
                "user",
                G._BoundaryPlan,
                model="gemini-3.5-flash",
                thinking_level="low",
                max_output_tokens=4096,
                timeout_s=45.0,
                deadline_monotonic=deadline,
                operation="flash_boundary_selector",
                prompt_version=G.FLASH_SPLIT_PROFILE,
                cancelled=cancelled.is_set,
            )
    finally:
        if timer is not None:
            timer.cancel()
            timer.join(timeout=1)
        for _ in range(G._SELECTOR_CALL_LIMIT):
            slots.release()

    assert dispatched == []
    assert exc_info.value.telemetry["error_type"] == expected_error_type
    assert exc_info.value.telemetry["dispatched"] is False


def test_boundary_repair_without_explicit_clip_limit_does_not_compare_none(
    monkeypatch,
):
    segments = _segs(1)
    proposal = _topic("Repair", 0, 0)
    report = G._Conversion(repair_candidates=[
        G._BoundaryRepairCandidate(
            candidate_id="candidate-0-0",
            prefix="proposal_0",
            proposal=proposal,
            start_line=0,
            end_line=0,
            reason="bad_boundary",
        )
    ])
    monkeypatch.setattr(
        G,
        "_call_model",
        lambda *_args, **_kwargs: (G._BoundaryRepairPlan(items=[]), {}),
    )

    calls = G._repair_failed_boundaries(
        report,
        segments,
        _words(segments),
        "concept",
        {},
        deadline=10_000.0,
        cancelled=None,
    )

    assert calls == [{}]
    assert "proposal_0:repair_omitted" in report.rejected_reasons


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
