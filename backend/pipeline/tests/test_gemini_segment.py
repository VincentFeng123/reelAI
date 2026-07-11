from __future__ import annotations

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
        "title": title,
        "start_line": start_line,
        "end_line": end_line,
        "start_quote": f"line {start_line}",
        "end_quote": f"end {end_line}",
        "facet": "concept",
        "reason": "Teaches the requested concept completely.",
        "informativeness": 0.9,
        "topic_relevance": 0.9,
        "difficulty": 0.5,
        "self_contained": True,
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
    assert "cut_end" not in clips[0]


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


def test_missing_words_and_unaligned_quotes_are_rejected():
    segs = _segs(2)
    plan = _plan(("T", 0, 1))
    assert G._plan_to_clips(plan, segs, [], {"segment_fine_snap": False}) == []
    bad_words = [{"word": "unrelated", "start": 0.0, "end": 1.0}]
    assert G._plan_to_clips(plan, segs, bad_words, {"segment_fine_snap": False}) == []


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


def test_fine_snap_tightens_boundary_only_after_quote_validation():
    segs = [{
        "start": 0.0,
        "end": 100.0,
        "text": "intro then the real topic starts here and ends now",
    }]
    words = [
        {"word": "intro", "start": 0.0, "end": 5.0},
        {"word": "topic", "start": 40.0, "end": 42.0},
        {"word": "starts", "start": 42.0, "end": 44.0},
        {"word": "ends", "start": 90.0, "end": 92.0},
        {"word": "now", "start": 92.0, "end": 95.0},
    ]
    plan = G._Plan(topics=[_topic(
        "T", 0, 0,
        start_quote="topic starts",
        end_quote="ends now",
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
    assert (clips[0]["start"], clips[0]["end"]) == pytest.approx((40.0, 95.0))


def test_quote_boundaries_snap_outward_to_caption_gap_midpoints():
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
    assert (clips[0]["start"], clips[0]["end"]) == (9.5, 20.5)


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
