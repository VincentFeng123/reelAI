# backend/tests/clip_engine/test_gemini_segment_curation.py
"""Curation gates in the gemini-segment engine (offline — no LLM, no network).

Covers the shared one-pass quality/duration/boundary contract.
"""
import json

import pytest

from backend.app.clip_engine.clipper.pipeline.gemini_segment import (
    _norm_informativeness,
    _Plan,
    _Topic,
    _cue_batches,
    _plan_to_clips,
    _prompts,
    segment_clips,
)


def test_developer_api_schema_omits_unsupported_additional_properties():
    schema_json = json.dumps(_Plan.model_json_schema(), sort_keys=True)
    assert "additionalProperties" not in schema_json


def _segs(n: int, sec: float = 30.0) -> list[dict]:
    return [
        {"start": i * sec, "end": (i + 1) * sec, "text": f"line {i} text"}
        for i in range(n)
    ]


def _topic(start_line: int, end_line: int, **kw) -> _Topic:
    base = dict(
        title="T", start_line=start_line, end_line=end_line,
        start_quote=f"line {start_line}", end_quote="text", kind="content",
        informativeness=0.9, topic_relevance=0.9, self_contained=True,
    )
    base.update(kw)
    return _Topic(**base)


def _run(
    topics: list[_Topic], n_segs: int = 10, settings: dict | None = None, sec: float = 30.0,
) -> list[dict]:
    return _plan_to_clips(_Plan(topics=topics), _segs(n_segs, sec), [], settings or {})


class TestKindGate:
    def test_intro_outro_admin_promo_dropped(self):
        topics = [
            _topic(0, 1, kind="intro", informativeness=0.9),
            _topic(2, 3, kind="content", informativeness=0.9, title="Keep"),
            _topic(4, 5, kind="outro", informativeness=0.9),
            _topic(6, 7, kind="admin", informativeness=0.9),
            _topic(8, 9, kind="promo", informativeness=0.9),
        ]
        clips = _run(topics)
        assert [c["title"] for c in clips] == ["Keep"]

    def test_unknown_or_missing_kind_fails_closed(self):
        assert _run([_topic(0, 1, kind="lesson")]) == []
        assert _run([_topic(0, 1, kind=None)]) == []

    def test_educational_kind_kept(self):
        clips = _run([_topic(0, 1, kind="educational")])
        assert clips[0]["kind"] == "educational"

    def test_blank_quote_anchor_fails_closed(self):
        assert _run([_topic(0, 1, start_quote="")]) == []
        assert _run([_topic(0, 1, end_quote="   ")]) == []

    def test_kind_case_insensitive(self):
        clips = _run([_topic(0, 1, kind="INTRO", informativeness=0.9)])
        assert clips == []

    def test_near_miss_labels_dropped(self):
        for label in ("sponsor", "ad", "advertisement", "introduction", "greeting"):
            assert _run([_topic(0, 1, kind=label, informativeness=0.9)]) == [], label


class TestInformativenessGate:
    def test_below_default_min_dropped(self):
        clips = _run([_topic(0, 1, informativeness=0.4)])
        assert clips == []

    def test_at_default_min_kept(self):
        clips = _run([_topic(0, 1, informativeness=0.6)])
        assert len(clips) == 1

    def test_settings_override(self):
        clips = _run(
            [_topic(0, 1, informativeness=0.6)],
            settings={"segment_informativeness_min": 0.7},
        )
        assert clips == []

    def test_informativeness_carried_on_clip(self):
        clips = _run([_topic(0, 1, informativeness=0.8)])
        assert clips[0]["informativeness"] == pytest.approx(0.8)

    @pytest.mark.parametrize("field", ["informativeness", "topic_relevance", "self_contained"])
    def test_missing_confidence_fails_closed(self, field):
        assert _run([_topic(0, 1, **{field: None})]) == []

    @pytest.mark.parametrize("field", ["informativeness", "topic_relevance"])
    def test_each_quality_score_has_point_six_floor(self, field):
        assert _run([_topic(0, 1, **{field: 0.59})]) == []
        assert len(_run([_topic(0, 1, **{field: 0.6})])) == 1

    def test_self_contained_must_be_true(self):
        assert _run([_topic(0, 1, self_contained=False)]) == []


class TestDurationContract:
    def test_short_complete_clip_survives(self):
        clips = _run(
            [_topic(0, 0)], n_segs=2,
            settings={"segment_fine_snap": False, "segment_min_clip_s": 15}, sec=5.0,
        )
        assert len(clips) == 1
        assert clips[0]["end"] - clips[0]["start"] == 5.0

    def test_ninety_to_one_eighty_seconds_kept_untrimmed(self):
        clips = _run([_topic(0, 5)])
        assert len(clips) == 1
        assert clips[0]["start"] == 0.0
        assert clips[0]["end"] == 180.0

    def test_above_one_eighty_kept_intact(self):
        clips = _run([_topic(0, 6)])
        assert len(clips) == 1
        assert clips[0]["start"] == 0.0
        assert clips[0]["end"] == 210.0

    def test_one_second_validity_guard(self):
        segs = [{"start": 1.0, "end": 1.999, "text": "tiny"}]
        assert _plan_to_clips(_Plan(topics=[_topic(0, 0)]), segs, [], {}) == []

    def test_canonical_millisecond_timestamps_do_not_impose_an_upper_cap(self):
        rounds_to_180 = [{"start": 0.0, "end": 180.0004, "text": "complete"}]
        rounds_above_180 = [{"start": 0.0, "end": 180.0006, "text": "complete"}]
        topic = _topic(0, 0, start_quote="complete", end_quote="complete")
        assert _plan_to_clips(_Plan(topics=[topic]), rounds_to_180, [], {})[0]["end"] == 180.0
        assert (
            _plan_to_clips(_Plan(topics=[topic]), rounds_above_180, [], {})[0]["end"]
            == 180.001
        )

    def test_safety_ceiling_remains_forty(self):
        segs = _segs(45, sec=1.0)
        topics = [_topic(index, index, title=f"T{index}") for index in range(45)]
        clips = _plan_to_clips(
            _Plan(topics=topics), segs, [], {"segment_fine_snap": False, "max_clips": 100}
        )
        assert len(clips) == 40

    def test_no_cut_end_or_low_confidence_fallback_fields(self):
        clip = _run([_topic(0, 1)])[0]
        assert "cut_end" not in clip and "low_confidence" not in clip


class TestExactCueBoundaries:
    def test_boundaries_are_cited_cue_start_and_end(self):
        segs = [
            {"start": 0.0, "end": 9.0, "text": "before"},
            {"start": 10.0, "end": 20.0, "text": "teaching"},
            {"start": 21.0, "end": 30.0, "text": "after"},
        ]
        clips = _plan_to_clips(
            _Plan(topics=[_topic(1, 1, start_quote="teaching", end_quote="teaching")]),
            segs, [], {},
        )
        assert clips[0]["start"] == 10.0
        assert clips[0]["end"] == 20.0

    def test_small_or_distant_gaps_do_not_move_semantic_boundary(self):
        segs = [
            {"start": 0.0, "end": 5.0, "text": "before"},
            {"start": 10.0, "end": 20.0, "text": "teaching"},
            {"start": 20.2, "end": 30.0, "text": "after"},
        ]
        clips = _plan_to_clips(
            _Plan(topics=[_topic(1, 1, start_quote="teaching", end_quote="teaching")]),
            segs, [], {},
        )
        assert clips[0]["start"] == 10.0
        assert clips[0]["end"] == 20.0


class TestOverlapHandling:
    def test_contextual_overlap_below_eighty_percent_is_kept(self):
        clips = _run([_topic(0, 9, title="A"), _topic(4, 13, title="B")], n_segs=20,
                     settings={"segment_fine_snap": False}, sec=1.0)
        assert [c["title"] for c in clips] == ["A", "B"]
        assert clips[1]["start"] < clips[0]["end"]

    def test_eighty_percent_coverage_dedupes_without_boundary_mutation(self):
        clips = _run([_topic(0, 9, title="A"), _topic(2, 11, title="B")], n_segs=20,
                     settings={"segment_fine_snap": False}, sec=1.0)
        assert len(clips) == 1
        assert (clips[0]["start"], clips[0]["end"]) in {(0.0, 10.0), (2.0, 12.0)}

    def test_near_duplicate_keeps_higher_quality_span(self):
        clips = _run([
            _topic(0, 9, title="lower", informativeness=0.7),
            _topic(2, 11, title="higher", informativeness=0.95),
        ], n_segs=20, settings={"segment_fine_snap": False}, sec=1.0)
        assert [clip["title"] for clip in clips] == ["higher"]

    def test_timestamps_round_to_milliseconds(self):
        segs = [{"start": 1.23456, "end": 4.56789, "text": "complete idea"}]
        clips = _plan_to_clips(
            _Plan(topics=[_topic(0, 0, start_quote="complete", end_quote="idea")]),
            segs, [], {},
        )
        assert clips[0]["start"] == 1.235
        assert clips[0]["end"] == 4.568


class TestInformativenessScale:
    def test_zero_to_one_passthrough(self):
        assert _norm_informativeness(0.7) == pytest.approx(0.7)

    def test_ten_scale_normalized(self):
        assert _norm_informativeness(7) == pytest.approx(0.7)
        assert _norm_informativeness(3) == pytest.approx(0.3)

    def test_percent_scale_normalized(self):
        assert _norm_informativeness(85) == pytest.approx(0.85)

    def test_clamped(self):
        assert _norm_informativeness(-2) == 0.0
        assert _norm_informativeness(1.0) == 1.0


class TestTopicThreading:
    def test_prompt_names_the_topic(self):
        system, _ = _prompts("[0] 00:00 hi", 1, topic="photosynthesis")
        assert "photosynthesis" in system

    def test_prompt_without_topic_has_no_topic_rule(self):
        system, _ = _prompts("[0] 00:00 hi", 1)
        assert "viewer is studying" not in system

    def test_segment_clips_is_live_compatibility_reexport(self):
        from backend.pipeline import gemini_segment as live_segmenter

        assert segment_clips is live_segmenter.segment_clips


class TestLearningDetailsContract:
    def test_summary_takeaways_match_reason_and_question_are_carried(self):
        question = {
            "prompt": "Which line states the central idea?",
            "options": ["Line zero", "A tangent", "An outro", "A sponsor"],
            "correct_index": 0,
            "explanation": "Line zero introduces the idea taught in the clip.",
            "cue_ids": ["cue-0"],
        }
        clip = _run([
            _topic(
                0,
                1,
                summary="The clip explains the central idea in two steps.",
                summary_cue_ids=["cue-0", "cue-1"],
                takeaways=["First idea", "Second idea"],
                takeaway_cue_ids=[["cue-0"], ["cue-1"]],
                match_reason="It directly explains the line used by this topic.",
                match_reason_cue_ids=["cue-0"],
                assessment=question,
            )
        ])[0]
        assert clip["summary"].startswith("The clip explains")
        assert clip["takeaways"] == ["First idea", "Second idea"]
        assert clip["match_reason"].startswith("It directly")
        assert clip["assessment"] == question

    def test_ungrounded_metadata_is_discarded_without_rejecting_clip(self):
        clip = _run([_topic(
            0, 1, summary="Unsupported", takeaways=["Unsupported"],
            match_reason="Unsupported", assessment={
                "prompt": "Q", "options": ["a", "b", "c", "d"],
                "correct_index": 0, "explanation": "line zero",
            },
        )])[0]
        assert clip["summary"] == ""
        assert clip["takeaways"] == []
        assert clip["match_reason"] == ""
        assert clip["assessment"] is None

    @pytest.mark.parametrize(
        "question",
        [
            {"prompt": "Q", "options": ["a", "a", "b", "c"], "correct_index": 0, "explanation": "line"},
            {"prompt": "Q", "options": ["a", "b", "c", "d", "e"], "correct_index": 0, "explanation": "line"},
            {"prompt": "Q", "options": ["a", "b", "c", "d"], "correct_index": 4, "explanation": "line"},
            {"prompt": "Q", "options": ["a", "b", "c", "d"], "correct_index": True, "explanation": "line"},
            {"prompt": "Q", "options": ["a", "b", "c", "d"], "correct_index": 1.5, "explanation": "line"},
            {"prompt": "Q", "options": ["a", "b", "c", "d"], "correct_index": 0, "explanation": ""},
            {"prompt": "Q", "options": ["a", "b", "c", "d"], "correct_index": 0, "explanation": "Unrelated generic praise"},
        ],
    )
    def test_malformed_question_is_discarded_without_rejecting_clip(self, question):
        clips = _run([_topic(0, 1, assessment=question)])
        assert len(clips) == 1
        assert clips[0]["assessment"] is None

    def test_prompt_requests_complete_learning_detail_contract(self):
        system, user = _prompts("[0] 00:00 line text", 1, topic="vectors")
        prompt = system + user
        for field in ("summary", "takeaways", "match_reason", "assessment", "correct_index", "explanation"):
            assert field in prompt


class TestProposalEvidence:
    def test_out_of_range_indices_are_rejected_instead_of_clamped(self):
        assert _run([_topic(-1, 1)]) == []
        assert _run([_topic(0, 10)], n_segs=2) == []

    def test_quote_must_appear_inside_its_cited_cue(self):
        assert _run([_topic(0, 1, start_quote="hallucinated phrase")]) == []
        assert _run([_topic(0, 1, end_quote="line 0")]) == []

    def test_global_quality_selection_happens_before_chronological_order(self):
        topics = [
            _topic(index, index, title=f"T{index}", informativeness=0.6)
            for index in range(40)
        ]
        topics.append(_topic(40, 40, title="best late clip", informativeness=1.0))
        clips = _run(topics, n_segs=41, settings={"max_clips": 1}, sec=1.0)
        assert [clip["title"] for clip in clips] == ["best late clip"]

    def test_clip_carries_exact_cue_and_model_metadata(self):
        [clip] = _run(
            [_topic(0, 1)],
            settings={"_model_used": "gemini-primary", "_quality_degraded": True},
        )
        assert clip["cue_ids"] == ["cue-0", "cue-1"]
        assert clip["transcript_text"] == "line 0 text line 1 text"
        assert clip["model_used"] == "gemini-primary"
        assert clip["quality_degraded"] is True


class TestCueBatching:
    def test_batches_are_bounded_and_overlap_by_cue(self):
        segments = _segs(8, sec=1.0)
        batches = _cue_batches(
            segments, max_cues=3, max_input_tokens=10_000, overlap_cues=1
        )
        assert [offset for offset, _ in batches] == [0, 2, 4, 6]
        assert all(len(batch) <= 3 for _, batch in batches)
        assert batches[0][1][-1]["text"] == batches[1][1][0]["text"]

    def test_input_token_limit_splits_long_cues(self):
        segments = [
            {"start": i, "end": i + 1, "text": "word " * 400}
            for i in range(3)
        ]
        batches = _cue_batches(
            segments, max_cues=10, max_input_tokens=2200, overlap_cues=0
        )
        assert [len(batch) for _, batch in batches] == [1, 1, 1]
