"""FE1 (embed bleed fix) + FE2 (quality payload) — the serving embed path.

FE1 root cause: orchestrator stamped end=ceil(end)+1, bleeding ~1-1.7s into the NEXT sentence
on every clip, while the labeling twin used the correct floor/ceil/guard form. Both now share
backend.embed.embed_url, so the drift (and the +1) is gone.
"""
from __future__ import annotations

from backend.embed import embed_url
from backend.orchestrator import _build_embed_clips


def test_embed_url_no_plus_one_bleed():
    # the old serving bug produced end=47 for 45.2 (ceil 46 + 1); the shared helper yields 46.
    assert embed_url("v", 12.3, 45.2) == "https://www.youtube.com/embed/v?start=12&end=46&rel=0"
    assert embed_url("v", 0.0, 30.0) == "https://www.youtube.com/embed/v?start=0&end=30&rel=0"
    # guarded: negative start clamps to 0, and end is always >= start+1 (never zero-length)
    assert embed_url("v", -1.0, 0.2) == "https://www.youtube.com/embed/v?start=0&end=1&rel=0"


def test_build_embed_clips_uses_shared_helper_and_surfaces_quality():
    spec = {"start": 12.3, "end": 45.2, "sequence_index": 1,
            "final_quality": 0.7123, "warnings": ("trimmed_start",), "ship_flagged": True}
    c = _build_embed_clips([spec], "vid")[0]
    # FE1: shared helper — no +1 bleed (was end=47 under the bug)
    assert c["embed_url"] == "https://www.youtube.com/embed/vid?start=12&end=46&rel=0"
    # FE2: quality signals surfaced into the payload
    assert c["final_quality"] == 0.712 and c["warnings"] == ["trimmed_start"]
    assert c["ship_flagged"] is True


def test_build_embed_clips_quality_defaults_never_keyerror():
    # a spec missing the quality fields yields safe defaults, not a KeyError
    c = _build_embed_clips([{"start": 0.0, "end": 30.0}], "vid")[0]
    assert c["final_quality"] is None and c["warnings"] == [] and c["ship_flagged"] is False
    assert c["embed_url"] == "https://www.youtube.com/embed/vid?start=0&end=30&rel=0"


def test_build_embed_clips_preserves_milliseconds_and_selector_fields():
    spec = {
        "start": 1.23456, "end": 4.56789, "kind": "educational",
        "informativeness": 0.8, "topic_relevance": 0.9,
        "self_contained": True, "difficulty": 0.4,
        "summary": "A grounded summary.",
        "takeaways": ["One", "Two"],
        "match_reason": "It explains the requested idea.",
        "assessment": {
            "prompt": "What is taught?", "options": ["A", "B", "C", "D"],
            "correct_index": 0, "explanation": "The transcript teaches A.",
        },
    }
    clip = _build_embed_clips([spec], "vid")[0]
    assert (clip["start"], clip["end"], clip["duration"]) == (1.235, 4.568, 3.333)
    assert {key: clip[key] for key in (
        "kind", "informativeness", "topic_relevance", "self_contained", "difficulty"
    )} == {
        "kind": "educational", "informativeness": 0.8, "topic_relevance": 0.9,
        "self_contained": True, "difficulty": 0.4,
    }
    assert clip["embed_url"] == "https://www.youtube.com/embed/vid?start=1&end=5&rel=0"
    assert clip["summary"] == "A grounded summary."
    assert clip["takeaways"] == ["One", "Two"]
    assert clip["match_reason"].startswith("It explains")
    assert clip["assessment"]["correct_index"] == 0
