"""
Pure unit tests for backend.app.clip_engine.bridge — no DB, no network.
"""
from backend.app.clip_engine import bridge

TRANSCRIPT = {
    "segments": [
        {"start": 0.0, "end": 5.0, "text": "the chain rule in calculus"},
        {"start": 5.0, "end": 10.0, "text": "unrelated cooking tips"},
    ],
    "words": [],
    "duration": 600.0,
}


# ── to_cues ──────────────────────────────────────────────────────────────────


def test_to_cues_maps_segments():
    tx = {
        "segments": [
            {"start": 0.0, "end": 3.0, "text": "hello world"},
            {"start": 3.0, "end": 6.0, "text": "second segment"},
        ],
        "words": [],
    }
    cues = bridge.to_cues(tx)
    assert len(cues) == 2
    assert cues[0].start == 0.0
    assert cues[0].end == 3.0
    assert cues[0].text == "hello world"
    assert cues[1].text == "second segment"


def test_to_cues_skips_empties():
    tx = {
        "segments": [
            {"start": 0, "end": 1, "text": "hi"},
            {"start": 1, "end": 2, "text": ""},
            {"start": 2, "end": 3, "text": "  "},
        ],
        "words": [],
    }
    cues = bridge.to_cues(tx)
    assert [c.text for c in cues] == ["hi"]


def test_to_cues_empty_transcript():
    cues = bridge.to_cues({"segments": [], "words": []})
    assert cues == []


# ── to_metadata ───────────────────────────────────────────────────────────────


def test_to_metadata_platform_fields():
    meta = bridge.to_metadata("abc123", {}, "https://www.youtube.com/watch?v=abc123")
    assert meta.platform == "yt"
    assert meta.source_id == "abc123"
    assert meta.source_url == "https://www.youtube.com/watch?v=abc123"
    assert meta.playback_url == "https://www.youtube.com/embed/abc123"


def test_to_metadata_copies_fields():
    raw = {
        "title": "My Video",
        "description": "Great desc",
        "author_name": "Author A",
        "duration_sec": 120.5,
        "thumbnail_url": "https://img.example.com/thumb.jpg",
        "view_count": 9999,
    }
    meta = bridge.to_metadata("xyz", raw, "https://www.youtube.com/watch?v=xyz")
    assert meta.title == "My Video"
    assert meta.description == "Great desc"
    assert meta.author_name == "Author A"
    assert meta.duration_sec == 120.5
    assert meta.thumbnail_url == "https://img.example.com/thumb.jpg"
    assert meta.view_count == 9999


def test_to_metadata_missing_fields_use_defaults():
    meta = bridge.to_metadata("vid", {}, "https://www.youtube.com/watch?v=vid")
    assert meta.title == ""
    assert meta.description == ""
    assert meta.author_name == ""
    assert meta.duration_sec is None
    assert meta.thumbnail_url == ""
    assert meta.view_count is None


def test_to_metadata_view_count_digit_string_coerced():
    raw = {"view_count": "42000"}
    meta = bridge.to_metadata("v", raw, "https://www.youtube.com/watch?v=v")
    assert meta.view_count == 42000


def test_to_metadata_view_count_non_digit_string_ignored():
    raw = {"view_count": "not-a-number"}
    meta = bridge.to_metadata("v", raw, "https://www.youtube.com/watch?v=v")
    assert meta.view_count is None


# ── synth_adapter_result ──────────────────────────────────────────────────────


def test_synth_adapter_result_fields():
    ar = bridge.synth_adapter_result("dQw4w9WgXcQ", "https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    assert ar.platform == "yt"
    assert ar.source_id == "dQw4w9WgXcQ"
    assert ar.playback_url == "https://www.youtube.com/embed/dQw4w9WgXcQ"


def test_synth_adapter_result_source_url_preserved():
    url = "https://www.youtube.com/watch?v=abc"
    ar = bridge.synth_adapter_result("abc", url)
    assert ar.source_url == url


def test_synth_adapter_result_video_path_is_path():
    from pathlib import Path
    ar = bridge.synth_adapter_result("v", "https://www.youtube.com/watch?v=v")
    assert isinstance(ar.video_path, Path)


# ── window_text ───────────────────────────────────────────────────────────────


def test_window_text_overlapping_segments():
    text = bridge.window_text(TRANSCRIPT, 0.0, 5.0)
    assert "chain rule" in text


def test_window_text_second_segment_only():
    text = bridge.window_text(TRANSCRIPT, 5.0, 10.0)
    assert "cooking" in text
    assert "chain rule" not in text


def test_window_text_no_overlap():
    text = bridge.window_text(TRANSCRIPT, 100.0, 200.0)
    assert text == ""


def test_window_text_both_segments():
    text = bridge.window_text(TRANSCRIPT, 0.0, 10.0)
    assert "chain rule" in text
    assert "cooking" in text


# ── to_segment ────────────────────────────────────────────────────────────────


def test_to_segment_in_window_text():
    clip = {"start": 0.0, "end": 5.0, "title": "Chain Rule", "score": 0.8}
    seg = bridge.to_segment(clip, TRANSCRIPT)
    assert seg.t_start == 0.0
    assert seg.t_end == 5.0
    assert "chain rule" in seg.text.lower()
    assert seg.score == 0.8


def test_to_segment_fallback_to_title_when_no_window_text():
    clip = {"start": 999.0, "end": 1000.0, "title": "My Title"}
    seg = bridge.to_segment(clip, TRANSCRIPT)
    assert seg.text == "My Title"


def test_to_segment_default_score():
    clip = {"start": 0.0, "end": 5.0, "title": "X"}
    seg = bridge.to_segment(clip, TRANSCRIPT)
    assert seg.score == 1.0


# ── relevance_score ───────────────────────────────────────────────────────────


def test_relevance_score_empty_query_returns_one():
    clip = {"start": 0.0, "end": 5.0, "title": "anything"}
    assert bridge.relevance_score(clip, TRANSCRIPT, "") == 1.0
    assert bridge.relevance_score(clip, TRANSCRIPT, None) == 1.0


def test_relevance_score_on_topic_higher_than_off_topic():
    on_topic = {"start": 0.0, "end": 5.0, "title": "chain rule"}
    off_topic = {"start": 5.0, "end": 10.0, "title": "cooking tips"}
    score_on = bridge.relevance_score(on_topic, TRANSCRIPT, "chain rule calculus")
    score_off = bridge.relevance_score(off_topic, TRANSCRIPT, "chain rule calculus")
    assert score_on > score_off


def test_relevance_score_perfect_overlap():
    clip = {"start": 0.0, "end": 5.0, "title": "chain rule calculus"}
    score = bridge.relevance_score(clip, TRANSCRIPT, "chain rule calculus")
    assert score == 1.0


def test_relevance_score_no_overlap_returns_zero():
    clip = {"start": 5.0, "end": 10.0, "title": "cooking tips eggs"}
    score = bridge.relevance_score(clip, TRANSCRIPT, "quantum physics")
    assert score == 0.0


# ── filter_by_query ───────────────────────────────────────────────────────────


def test_filter_by_query_keeps_all_when_no_query():
    clips = [
        {"start": 0.0, "end": 5.0, "title": "A"},
        {"start": 5.0, "end": 10.0, "title": "B"},
    ]
    result = bridge.filter_by_query(clips, TRANSCRIPT, None)
    assert result == clips


def test_filter_by_query_keeps_all_when_empty_string():
    clips = [{"start": 0.0, "end": 5.0, "title": "A"}]
    result = bridge.filter_by_query(clips, TRANSCRIPT, "")
    assert result == clips


def test_filter_by_query_ranks_on_topic_first():
    clips = [
        {"start": 5.0, "end": 10.0, "title": "cooking"},
        {"start": 0.0, "end": 5.0, "title": "chain rule"},
    ]
    out = bridge.filter_by_query(clips, TRANSCRIPT, "chain rule calculus", floor=0.0)
    assert out[0]["title"] == "chain rule"
    assert out[0]["score"] >= out[-1]["score"]


def test_filter_by_query_annotates_scores():
    clips = [{"start": 0.0, "end": 5.0, "title": "chain rule"}]
    out = bridge.filter_by_query(clips, TRANSCRIPT, "chain rule")
    assert "score" in out[0]
    assert isinstance(out[0]["score"], float)


def test_filter_by_query_drops_at_floor():
    clips = [
        {"start": 0.0, "end": 5.0, "title": "chain rule"},
        {"start": 5.0, "end": 10.0, "title": "xyz uvw"},
    ]
    out = bridge.filter_by_query(clips, TRANSCRIPT, "chain rule calculus", floor=0.0)
    # The off-topic clip's score should be 0, which is <= floor so it's dropped
    titles = [c["title"] for c in out]
    assert "chain rule" in titles
    # xyz uvw clip should have 0 score and be dropped
    assert "xyz uvw" not in titles


def test_filter_by_query_no_mutate_on_no_query():
    clips = [{"start": 0.0, "end": 5.0, "title": "A"}]
    original_clip = clips[0].copy()
    bridge.filter_by_query(clips, TRANSCRIPT, None)
    assert clips[0] == original_clip


# ── pick_best_clip ────────────────────────────────────────────────────────────


def _make_clip(start: float, end: float) -> dict:
    return {"start": start, "end": end, "title": f"{int(end - start)}s clip"}


def test_pick_best_clip_prefers_in_bounds():
    # clips: 70s, 40s, 200s; target=45, max=60 → in-bounds are 40s and 70s is out
    # Wait: 70s > 60 so in-bounds = [40s]. Returns 40s.
    clips = [_make_clip(0, 70), _make_clip(100, 140), _make_clip(200, 400)]
    result = bridge.pick_best_clip(clips, target_sec=45.0, max_sec=60.0)
    assert float(result["end"]) - float(result["start"]) == 40.0


def test_pick_best_clip_fallback_when_none_in_bounds():
    # clips: 90s, 200s; both exceed max=60 → fallback to full pool, closest to 45 is 90s
    clips = [_make_clip(0, 90), _make_clip(100, 300)]
    result = bridge.pick_best_clip(clips, target_sec=45.0, max_sec=60.0)
    assert float(result["end"]) - float(result["start"]) == 90.0
