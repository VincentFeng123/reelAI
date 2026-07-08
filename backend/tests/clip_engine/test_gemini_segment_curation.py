# backend/tests/clip_engine/test_gemini_segment_curation.py
"""Curation gates in the gemini-segment engine (offline — no LLM, no network).

Covers the quality fixes that closed the gap with the practice topic engine:
  * kind gate: intro/outro/admin/promo (and near-miss labels) never ship
  * informativeness gate: < SEGMENT_INFORMATIVENESS_MIN drops; mis-scaled
    scores (0-10 / 0-100) are normalized, not saturated to 1.0
  * max-duration fitting: whole-chapter slabs are trimmed back to the last
    chunk boundary within SEGMENT_MAX_CLIP_S (practice walk-back analogue)
  * low-confidence backstop: a video never zeroes out when content topics exist
  * topic threading: the viewer's topic reaches the prompt
"""
import pytest

from backend.app.clip_engine.clipper.pipeline import gemini_segment
from backend.app.clip_engine.clipper.pipeline.gemini_segment import (
    _norm_informativeness,
    _Plan,
    _Topic,
    _plan_to_clips,
    _prompts,
    segment_clips,
)


def _segs(n: int, sec: float = 30.0) -> list[dict]:
    return [
        {"start": i * sec, "end": (i + 1) * sec, "text": f"line {i} text"}
        for i in range(n)
    ]


def _topic(start_line: int, end_line: int, **kw) -> _Topic:
    base = dict(title="T", start_line=start_line, end_line=end_line)
    base.update(kw)
    return _Topic(**base)


def _run(topics: list[_Topic], n_segs: int = 10, settings: dict | None = None) -> list[dict]:
    return _plan_to_clips(_Plan(topics=topics), _segs(n_segs), [], settings or {})


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

    def test_unknown_kind_treated_as_content(self):
        clips = _run([_topic(0, 1, kind="lesson", informativeness=0.9)])
        assert len(clips) == 1

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
        clips = _run([_topic(0, 1, informativeness=0.5)])
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


class TestMaxDurationFitting:
    def test_whole_chapter_slab_trimmed_to_chunk_boundary(self):
        # 4 lines x 30s = 120s > default 75s ceiling → end walks back to the
        # last chunk boundary within budget (60.0), instead of dropping.
        clips = _run([_topic(0, 3, informativeness=0.9)])
        assert len(clips) == 1
        assert clips[0]["start"] == 0.0
        assert clips[0]["end"] == 60.0

    def test_normal_clip_kept_untrimmed(self):
        # 2 lines x 30s = 60s <= 75s
        clips = _run([_topic(0, 1, informativeness=0.9)])
        assert len(clips) == 1
        assert clips[0]["end"] == 60.0

    def test_settings_override_trims_tighter(self):
        clips = _run(
            [_topic(0, 1, informativeness=0.9)],
            settings={"segment_max_clip_s": 45.0},
        )
        assert len(clips) == 1
        assert clips[0]["end"] == 30.0  # only boundary within 45s budget


class TestLowConfidenceBackstop:
    def test_video_never_zeroes_when_content_exists(self):
        # max 20s budget: no 30s-chunk boundary fits → main loop drops the topic
        # → backstop ships ONE hard-cut low-confidence clip instead of nothing.
        clips = _run(
            [_topic(0, 3, informativeness=0.9)],
            settings={"segment_max_clip_s": 20.0, "segment_min_clip_s": 15.0},
        )
        assert len(clips) == 1
        assert clips[0]["low_confidence"] is True
        assert clips[0]["start"] == 0.0
        assert clips[0]["end"] == 20.0
        assert clips[0]["sequence_index"] == 1

    def test_backstop_picks_most_informative(self):
        clips = _run(
            [
                _topic(0, 3, informativeness=0.6, title="meh"),
                _topic(4, 7, informativeness=0.9, title="best"),
            ],
            settings={"segment_max_clip_s": 20.0, "segment_min_clip_s": 15.0},
        )
        assert len(clips) == 1
        assert clips[0]["title"] == "best"

    def test_no_backstop_when_nothing_survives_the_gates(self):
        # Everything is filler → an empty result is the CORRECT outcome.
        clips = _run([_topic(0, 3, kind="intro", informativeness=0.9)])
        assert clips == []


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

    def test_segment_clips_passes_topic_to_prompt(self, monkeypatch):
        seen: dict = {}

        def fake_llm_json(system, user, schema, **kw):
            seen["system"] = system
            return _Plan(topics=[])

        monkeypatch.setattr(gemini_segment, "llm_json", fake_llm_json)
        tx = {"segments": _segs(2), "words": [], "duration": 60.0}
        segment_clips(tx, {}, topic="linear algebra")
        assert "linear algebra" in seen["system"]
