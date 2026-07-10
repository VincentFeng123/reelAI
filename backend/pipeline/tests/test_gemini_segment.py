from __future__ import annotations

import pytest

from backend.pipeline import gemini_segment as G


def _segs(n: int, step: float = 10.0) -> list[dict]:
    return [{"start": i * step, "end": (i + 1) * step, "text": f"line {i}"} for i in range(n)]


def _topic(title: str, start_line: int, end_line: int, **overrides) -> G._Topic:
    data = {
        "title": title,
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
    return G._Topic(**data)


def _plan(*triples) -> G._Plan:
    return G._Plan(topics=[_topic(t, a, b) for (t, a, b) in triples])


def test_maps_lines_to_chunk_times_no_finesnap():
    clips = G._plan_to_clips(_plan(("T1", 0, 1)), _segs(3), [], {"segment_fine_snap": False})
    assert len(clips) == 1
    assert (clips[0]["start"], clips[0]["end"]) == (0.0, 20.0)
    assert clips[0]["sequence_index"] == 1 and clips[0]["title"] == "T1"
    assert "cut_end" not in clips[0]


def test_clamps_out_of_range_line_indices():
    clips = G._plan_to_clips(_plan(("T", 5, 99)), _segs(3), [], {"segment_fine_snap": False})
    assert len(clips) == 1
    assert (clips[0]["start"], clips[0]["end"]) == (20.0, 30.0)


def test_contextual_overlap_is_preserved_without_boundary_mutation():
    clips = G._plan_to_clips(
        _plan(("A", 0, 2), ("B", 1, 3)), _segs(4), [], {"segment_fine_snap": False}
    )
    assert [(c["start"], c["end"]) for c in clips] == [(0.0, 30.0), (10.0, 40.0)]


def test_span_covering_eighty_percent_of_shorter_is_deduplicated():
    clips = G._plan_to_clips(
        _plan(("A", 0, 2), ("B", 0, 3)), _segs(4), [], {"segment_fine_snap": False}
    )
    assert [c["title"] for c in clips] == ["A"]


def test_one_second_validity_guard_cannot_be_disabled_by_legacy_setting():
    segs = [{"start": 0.0, "end": 0.8, "text": "brief"}]
    plan = G._Plan(topics=[_topic("brief", 0, 0, start_quote="brief", end_quote="brief")])
    assert G._plan_to_clips(plan, segs, [], {"segment_min_clip_s": 0}) == []


def test_locate_quote_finds_start_and_end():
    words = [
        {"word": "hello", "start": 1.0, "end": 1.5},
        {"word": "world", "start": 1.5, "end": 2.0},
        {"word": "foo", "start": 2.0, "end": 2.5},
    ]
    assert G._locate_quote(words, "hello world", 0.0, 3.0, "start") == pytest.approx(1.0)
    assert G._locate_quote(words, "world foo", 0.0, 3.0, "end") == pytest.approx(2.5)
    assert G._locate_quote(words, "zzz qqq", 0.0, 3.0, "start") is None


def test_fine_snap_tightens_boundary_when_quote_matches():
    segs = [{"start": 0.0, "end": 100.0, "text": "intro then the real topic starts here and ends now"}]
    words = [
        {"word": "intro", "start": 0.0, "end": 5.0},
        {"word": "topic", "start": 40.0, "end": 42.0},
        {"word": "starts", "start": 42.0, "end": 44.0},
        {"word": "ends", "start": 90.0, "end": 92.0},
        {"word": "now", "start": 92.0, "end": 95.0},
    ]
    plan = G._Plan(topics=[_topic(
        "T", 0, 0, start_quote="topic starts", end_quote="ends now"
    )])
    clips = G._plan_to_clips(plan, segs, words, {"segment_fine_snap": True})
    assert clips[0]["start"] == pytest.approx(40.0)
    assert clips[0]["end"] == pytest.approx(95.0)


def test_quote_boundaries_move_outward_to_nearby_caption_gap_midpoints():
    segs = [
        {"start": 0.0, "end": 9.0, "text": "before"},
        {"start": 10.0, "end": 20.0, "text": "topic starts and ends now"},
        {"start": 21.0, "end": 30.0, "text": "after"},
    ]
    words = [
        {"word": "topic", "start": 10.2, "end": 10.5},
        {"word": "starts", "start": 10.5, "end": 10.9},
        {"word": "ends", "start": 19.0, "end": 19.3},
        {"word": "now", "start": 19.3, "end": 19.6},
    ]
    plan = G._Plan(topics=[_topic(
        "T", 1, 1, start_quote="topic starts", end_quote="ends now"
    )])
    clips = G._plan_to_clips(plan, segs, words, {"segment_fine_snap": True})
    assert (clips[0]["start"], clips[0]["end"]) == (9.5, 20.5)


def test_no_qualifying_gap_retains_semantic_quote_boundaries():
    segs = [
        {"start": 0.0, "end": 10.0, "text": "before"},
        {"start": 10.0, "end": 20.0, "text": "topic starts and ends now"},
        {"start": 20.0, "end": 30.0, "text": "after"},
    ]
    words = [
        {"word": "topic", "start": 10.2, "end": 10.5},
        {"word": "starts", "start": 10.5, "end": 10.9},
        {"word": "ends", "start": 19.0, "end": 19.3},
        {"word": "now", "start": 19.3, "end": 19.6},
    ]
    plan = G._Plan(topics=[_topic(
        "T", 1, 1, start_quote="topic starts", end_quote="ends now"
    )])
    clips = G._plan_to_clips(plan, segs, words, {"segment_fine_snap": True})
    assert (clips[0]["start"], clips[0]["end"]) == (10.2, 19.6)
