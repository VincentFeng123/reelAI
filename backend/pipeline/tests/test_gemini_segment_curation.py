"""Shipping contract for the one-pass Gemini educational selector."""

import pytest

from backend.pipeline import gemini_segment
from backend.pipeline.gemini_segment import (
    _norm_informativeness,
    _Plan,
    _Topic,
    _plan_to_clips,
    _prompts,
    segment_clips,
)


def _segs(n: int, sec: float = 10.0) -> list[dict]:
    return [
        {"start": i * sec, "end": (i + 1) * sec, "text": f"line {i} text"}
        for i in range(n)
    ]


def _topic(start_line: int, end_line: int, **overrides) -> _Topic:
    data = {
        "title": "T",
        "start_line": start_line,
        "end_line": end_line,
        "start_quote": f"line {start_line}",
        "end_quote": f"line {end_line}",
        "kind": "content",
        "informativeness": 0.9,
        "topic_relevance": 0.9,
        "self_contained": True,
    }
    data.update(overrides)
    return _Topic(**data)


def _run(topics: list[_Topic], segs: list[dict] | None = None, settings: dict | None = None) -> list[dict]:
    return _plan_to_clips(
        _Plan(topics=topics), segs or _segs(20), [], {"segment_fine_snap": False, **(settings or {})}
    )


@pytest.mark.parametrize("kind", ["intro", "outro", "admin", "promo", "lesson", None])
def test_only_declared_educational_kinds_ship(kind):
    assert _run([_topic(0, 1, kind=kind)]) == []


@pytest.mark.parametrize("kind", ["content", "educational", "CONTENT"])
def test_content_and_educational_kinds_ship_case_insensitively(kind):
    assert len(_run([_topic(0, 1, kind=kind)])) == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("informativeness", 0.59),
        ("topic_relevance", 0.59),
        ("self_contained", False),
        ("informativeness", None),
        ("topic_relevance", None),
        ("self_contained", None),
    ],
)
def test_quality_gates_fail_closed(field, value):
    assert _run([_topic(0, 1, **{field: value})]) == []


def test_quality_thresholds_are_inclusive_and_carried():
    clips = _run([_topic(0, 1, informativeness=0.6, topic_relevance=0.6)])
    assert len(clips) == 1
    assert clips[0]["informativeness"] == pytest.approx(0.6)
    assert clips[0]["topic_relevance"] == pytest.approx(0.6)
    assert clips[0]["self_contained"] is True
    assert clips[0]["kind"] == "content"


@pytest.mark.parametrize(
    "field,setting",
    [
        ("informativeness", "segment_informativeness_min"),
        ("topic_relevance", "segment_topic_relevance_min"),
    ],
)
def test_settings_cannot_weaken_point_six_quality_floors(field, setting):
    assert _run([_topic(0, 1, **{field: 0.59})], settings={setting: 0.1}) == []


@pytest.mark.parametrize(
    "field,value", [("start_quote", ""), ("end_quote", "   ")]
)
def test_blank_boundary_quote_rejects_proposal(field, value):
    assert _run([_topic(0, 1, **{field: value})]) == []


def test_short_complete_clip_survives_legacy_fifteen_second_setting():
    clips = _run([_topic(0, 0)], segs=_segs(1, sec=5.0), settings={"segment_min_clip_s": 15})
    assert [(c["start"], c["end"]) for c in clips] == [(0.0, 5.0)]


@pytest.mark.parametrize("duration", [90.0, 120.0, 180.0])
def test_complete_clips_from_ninety_through_one_eighty_seconds_survive(duration):
    segs = [{"start": 0.0, "end": duration, "text": "complete explanation"}]
    clips = _run(
        [_topic(0, 0, start_quote="complete", end_quote="explanation")], segs=segs
    )
    assert len(clips) == 1 and clips[0]["end"] == duration


def test_clip_over_one_eighty_seconds_is_rejected_without_hard_cut_fallback():
    segs = [{"start": 0.0, "end": 180.001, "text": "too long"}]
    clips = _run(
        [_topic(0, 0, start_quote="too", end_quote="long")],
        segs=segs,
        settings={"segment_max_clip_s": 999},
    )
    assert clips == []


def test_duration_gate_uses_canonical_millisecond_timestamps():
    accepted = [{"start": 0.0, "end": 180.0004, "text": "complete"}]
    rejected = [{"start": 0.0, "end": 180.0006, "text": "complete"}]
    topic = _topic(0, 0, start_quote="complete", end_quote="complete")
    assert _run([topic], segs=accepted)[0]["end"] == 180.0
    assert _run([topic], segs=rejected) == []


def test_safety_ceiling_keeps_first_forty_qualified_proposals():
    segs = _segs(45, sec=1.0)
    topics = [_topic(i, i, title=f"T{i}") for i in range(45)]
    clips = _run(topics, segs=segs, settings={"max_clips": 100})
    assert len(clips) == 40
    assert [c["title"] for c in clips] == [f"T{i}" for i in range(40)]


def test_no_cut_end_or_low_confidence_fallback_fields():
    clip = _run([_topic(0, 1)])[0]
    assert "cut_end" not in clip and "low_confidence" not in clip


def test_highest_quality_near_duplicate_wins_then_results_are_chronological():
    topics = [
        _topic(0, 9, title="weaker", informativeness=0.6, topic_relevance=0.6),
        _topic(2, 11, title="stronger", informativeness=0.9, topic_relevance=0.9),
        _topic(15, 16, title="later"),
    ]
    clips = _run(topics, segs=_segs(20, sec=1.0))
    assert [clip["title"] for clip in clips] == ["stronger", "later"]
    assert (clips[0]["start"], clips[0]["end"]) == (2.0, 12.0)


def test_score_scale_normalization_is_preserved():
    assert _norm_informativeness(0.7) == pytest.approx(0.7)
    assert _norm_informativeness(7) == pytest.approx(0.7)
    assert _norm_informativeness(85) == pytest.approx(0.85)
    assert _norm_informativeness(-2) == 0.0


def test_prompt_names_topic_and_full_required_contract():
    system, user = _prompts("[0] 00:00 hi", 1, topic="photosynthesis")
    assert "photosynthesis" in system
    for field in (
        "kind", "informativeness", "topic_relevance", "self_contained", "difficulty",
        "start_line", "end_line", "start_quote", "end_quote",
    ):
        assert field in system + user
    assert "20-90" in system and "up to 180" in system and "Split longer sections" in system


def test_segment_clips_passes_topic_to_single_llm_call(monkeypatch):
    seen: list[str] = []

    def fake_llm_json(system, user, schema, **kwargs):
        seen.append(system)
        return _Plan(topics=[])

    monkeypatch.setattr(gemini_segment, "llm_json", fake_llm_json)
    segment_clips({"segments": _segs(2), "words": []}, {}, topic="linear algebra")
    assert len(seen) == 1 and "linear algebra" in seen[0]
