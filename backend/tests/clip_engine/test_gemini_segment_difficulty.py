"""Per-clip difficulty: parsed, normalized, carried — NEVER gates a clip."""
import pytest

from backend.app.clip_engine.clipper.pipeline.gemini_segment import (
    _Plan,
    _Topic,
    _plan_to_clips,
    _prompts,
)


def _segs(n: int, sec: float = 30.0) -> list[dict]:
    return [{"start": i * sec, "end": (i + 1) * sec, "text": f"line {i}"} for i in range(n)]


def _run(topics, n=10, settings=None):
    return _plan_to_clips(_Plan(topics=topics), _segs(n), [], settings or {})


def test_difficulty_carried_on_clip():
    t = _Topic(title="T", start_line=0, end_line=1, start_quote="line 0", end_quote="line 1",
               kind="content", informativeness=0.9, topic_relevance=0.9,
               self_contained=True, difficulty=0.8)
    clips = _run([t])
    assert clips[0]["difficulty"] == pytest.approx(0.8)


def test_difficulty_defaults_to_half_when_omitted():
    # Simulate parsed JSON with no difficulty key at all (the model omitted it).
    t = _Topic.model_validate({"title": "T", "start_line": 0, "end_line": 1,
                               "start_quote": "line 0", "end_quote": "line 1", "kind": "content",
                               "informativeness": 0.9, "topic_relevance": 0.9,
                               "self_contained": True})
    clips = _run([t])
    assert clips[0]["difficulty"] == pytest.approx(0.5)


def test_misscaled_difficulty_normalized():
    t = _Topic(title="T", start_line=0, end_line=1, start_quote="line 0", end_quote="line 1",
               kind="content", informativeness=0.9, topic_relevance=0.9,
               self_contained=True, difficulty=7)
    clips = _run([t])
    assert clips[0]["difficulty"] == pytest.approx(0.7)
    # 0-100 scale branch
    t = _Topic(title="T", start_line=0, end_line=1, start_quote="line 0", end_quote="line 1",
               kind="content", informativeness=0.9, topic_relevance=0.9,
               self_contained=True, difficulty=85)
    clips = _run([t])
    assert clips[0]["difficulty"] == pytest.approx(0.85)


def test_extreme_difficulty_never_gates():
    common = {"kind": "content",
              "informativeness": 0.9, "topic_relevance": 0.9, "self_contained": True}
    hard = _Topic(title="H", start_line=0, end_line=1, start_quote="line 0",
                  end_quote="line 1", difficulty=1.0, **common)
    easy = _Topic(title="E", start_line=2, end_line=3, start_quote="line 2",
                  end_quote="line 3", difficulty=0.0, **common)
    clips = _run([hard, easy])
    assert {c["title"] for c in clips} == {"H", "E"}


def test_prompt_documents_difficulty_scale():
    system, user = _prompts("[0] 00:00 hi", 1)
    assert "difficulty on a 0..1 scale" in system
    assert "difficulty" in user


def test_prompt_documents_latex_assessment_contract():
    system, _ = _prompts("[0] 00:00 hi", 1)
    assert r"\( ... \)" in system
    assert r"\[ ... \]" in system
    assert "only when it improves mathematical clarity" in system
    assert "Escape every LaTeX backslash in JSON" in system
