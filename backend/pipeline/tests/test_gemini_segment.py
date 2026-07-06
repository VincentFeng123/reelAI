from __future__ import annotations

import pytest

from backend.pipeline import gemini_segment as G


def _segs(n, step=10.0):
    return [{"start": i * step, "end": (i + 1) * step, "text": f"line {i}"} for i in range(n)]


def _plan(*triples):
    return G._Plan(topics=[G._Topic(title=t, start_line=a, end_line=b) for (t, a, b) in triples])


def test_maps_lines_to_chunk_times_no_finesnap():
    segs = _segs(3)                                   # 0-10, 10-20, 20-30
    clips = G._plan_to_clips(_plan(("T1", 0, 1)), segs, [], {"segment_fine_snap": False})
    assert len(clips) == 1
    c = clips[0]
    assert (c["start"], c["end"]) == (0.0, 20.0)
    assert c["sequence_index"] == 1 and c["title"] == "T1"
    assert c["cut_end"] >= c["end"]                   # tail pad applied


def test_clamps_out_of_range_line_indices():
    segs = _segs(3)
    clips = G._plan_to_clips(
        _plan(("T", 5, 99)), segs, [], {"segment_fine_snap": False, "segment_min_clip_s": 1})
    assert len(clips) == 1
    assert (clips[0]["start"], clips[0]["end"]) == (20.0, 30.0)   # clamped to last line


def test_overlaps_are_trimmed_to_previous_end():
    segs = _segs(4)                                   # 0-10,10-20,20-30,30-40
    clips = G._plan_to_clips(
        _plan(("A", 0, 2), ("B", 1, 3)), segs, [], {"segment_fine_snap": False, "segment_min_clip_s": 1})
    assert len(clips) == 2
    a, b = clips
    assert (a["start"], a["end"]) == (0.0, 30.0)
    assert b["start"] == 30.0 and b["end"] == 40.0     # B trimmed to start at A's end (no overlap)
    assert b["sequence_index"] == 2


def test_too_short_after_trim_is_dropped():
    segs = _segs(3)
    clips = G._plan_to_clips(
        _plan(("A", 0, 1), ("B", 1, 1)), segs, [], {"segment_fine_snap": False, "segment_min_clip_s": 15})
    # A = 0-20 (ok); B = 10-20 -> trimmed start to 20 -> 0s -> dropped
    assert [c["title"] for c in clips] == ["A"]


def test_locate_quote_finds_start_and_end():
    words = [{"word": "hello", "start": 1.0, "end": 1.5},
             {"word": "world", "start": 1.5, "end": 2.0},
             {"word": "foo", "start": 2.0, "end": 2.5}]
    assert G._locate_quote(words, "hello world", 0.0, 3.0, "start") == pytest.approx(1.0)
    assert G._locate_quote(words, "world foo", 0.0, 3.0, "end") == pytest.approx(2.5)
    assert G._locate_quote(words, "zzz qqq", 0.0, 3.0, "start") is None


def test_fine_snap_tightens_boundary_when_quote_matches():
    # one 0-100s chunk; words interpolated inside it
    segs = [{"start": 0.0, "end": 100.0, "text": "intro then the real topic starts here and ends now"}]
    words = [{"word": "intro", "start": 0.0, "end": 5.0},
             {"word": "topic", "start": 40.0, "end": 42.0},
             {"word": "starts", "start": 42.0, "end": 44.0},
             {"word": "ends", "start": 90.0, "end": 92.0},
             {"word": "now", "start": 92.0, "end": 95.0}]
    plan = G._Plan(topics=[G._Topic(title="T", start_line=0, end_line=0,
                                    start_quote="topic starts", end_quote="ends now")])
    clips = G._plan_to_clips(plan, segs, words, {"segment_fine_snap": True, "segment_min_clip_s": 1})
    assert len(clips) == 1
    # start snapped near 40 (not 0), end near 95 (not 100)
    assert clips[0]["start"] == pytest.approx(40.0, abs=1.0)
    assert clips[0]["end"] == pytest.approx(95.0, abs=1.0)
