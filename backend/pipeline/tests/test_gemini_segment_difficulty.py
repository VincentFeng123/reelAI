"""Per-clip difficulty is strict, carried, and never used as a quality gate."""

import pytest
from pydantic import ValidationError

from backend.pipeline import gemini_segment as G


def _segments() -> list[dict]:
    return [
        {"start": 0.0, "end": 30.0, "text": "line zero teaches easy lesson end zero"},
        {"start": 30.0, "end": 60.0, "text": "line one teaches hard lesson end one"},
    ]


def _words() -> list[dict]:
    out = []
    for segment in _segments():
        tokens = segment["text"].split()
        width = 29.8 / len(tokens)
        for index, token in enumerate(tokens):
            start = segment["start"] + 0.1 + index * width
            out.append({"word": token, "start": start, "end": start + width})
    return out


def _topic(**overrides) -> G._Topic:
    start_line = int(overrides.get("start_line", 0))
    line_word = "one" if start_line == 1 else "zero"
    data = {
        "candidate_id": (
            f"candidate-{overrides.get('title', 'Lesson')}-"
            f"{overrides.get('start_line', 0)}"
        ),
        "title": "Lesson",
        "learning_objective": "Understand the lesson.",
        "start_line": 0,
        "end_line": 0,
        "start_quote": "line zero",
        "end_quote": "end zero",
        "facet": "lesson",
        "reason": "A complete lesson.",
        "informativeness": 0.9,
        "topic_relevance": 0.9,
        "educational_importance": 0.9,
        "difficulty": 0.5,
        "directly_teaches_topic": True,
        "substantive": True,
        "topic_evidence_quote": f"line {line_word} teaches {overrides.get('title', 'easy').lower()} lesson end",
        "self_contained": True,
        "is_standalone": True,
        "prerequisite_candidate_ids": [],
        "uncertainty": "low",
        "uncertainty_reasons": [],
        "summary": "Line zero teaches an easy lesson.",
        "takeaways": ["Line zero teaches the lesson.", "The lesson reaches end zero."],
        "match_reason": "The easy lesson is taught directly.",
        "assessment": {
            "prompt": "Which lesson is taught?",
            "options": ["Easy", "Sponsor", "Greeting", "Outro"],
            "correct_index": 0,
            "explanation": "The easy lesson is taught.",
            "evidence_quote": "easy lesson",
        },
    }
    data.update(overrides)
    return G._Topic(**data)


def _run(topics):
    return G._plan_to_clips(
        G._Plan(topics=topics), _segments(), _words(), {"segment_fine_snap": False},
    )


def test_difficulty_is_carried():
    assert _run([_topic(difficulty=0.8)])[0]["difficulty"] == pytest.approx(0.8)


def test_difficulty_is_required():
    data = _topic().model_dump()
    data.pop("difficulty")
    with pytest.raises(ValidationError):
        G._Topic.model_validate(data)


@pytest.mark.parametrize("value", [-0.01, 1.01, 7, 85])
def test_misscaled_difficulty_is_rejected_not_repaired(value):
    data = _topic().model_dump()
    data["difficulty"] = value
    with pytest.raises(ValidationError):
        G._Topic.model_validate(data)


def test_extreme_valid_difficulty_never_gates():
    easy = _topic(title="Easy", difficulty=0.0)
    hard = _topic(
        title="Hard",
        start_line=1,
        end_line=1,
        start_quote="line one",
        end_quote="end one",
        difficulty=1.0,
        summary="Line one teaches a hard lesson.",
        takeaways=["Line one teaches the lesson.", "The lesson reaches end one."],
        match_reason="The hard lesson is taught directly.",
        assessment={
            "prompt": "Which lesson is taught?",
            "options": ["Hard", "Sponsor", "Greeting", "Outro"],
            "correct_index": 0,
            "explanation": "The hard lesson is taught.",
            "evidence_quote": "hard lesson",
        },
    )
    assert {clip["title"] for clip in _run([easy, hard])} == {"Easy", "Hard"}


def test_prompt_requires_difficulty():
    system, user = G._prompts("[0] 00:00 hi", 1)
    assert "difficulty" in system + user
