from __future__ import annotations

from unittest import mock

import pytest

from backend.app.clip_engine import lexical_timing, silence
from backend.app.ingestion import pipeline as pipeline_module


def _transcript(*segments: dict) -> dict:
    return {
        "source": "supadata",
        "native_mode": False,
        "artifact_key": "native-transcript:v4:acoustic-context",
        "duration": max(float(segment["end"]) for segment in segments),
        "segments": list(segments),
    }


def _clip(*, cue_ids: list[str], start: float, end: float) -> dict:
    return {
        "start": start,
        "end": end,
        "cue_ids": cue_ids,
        "start_quote": "its enclosing scope",
        "end_quote": "available to later calls",
        "title": "Python closure scope",
        "learning_objective": "Explain how a Python closure retains enclosing scope.",
        "facet": "closure scope",
        "reason": "Explains closure scope.",
        "kind": "educational",
        "informativeness": 0.9,
        "topic_relevance": 0.9,
        "educational_importance": 0.9,
        "difficulty": 0.4,
        "directly_teaches_topic": True,
        "substantive": True,
        "factually_grounded": True,
        "self_contained": True,
        "is_standalone": True,
        "topic_evidence_quote": "closure retains its enclosing scope",
        "selection_candidate_id": "closure-scope",
        "prerequisite_ids": [],
        "uncertainty": "low",
        "intent_role": "primary",
        "intent_coverage": 1.0,
        "intent_evidence": [],
        "summary": "A closure retains enclosing-scope values.",
        "takeaways": ["Closures retain values.", "Later calls can reuse them."],
        "match_reason": "Directly explains closure scope.",
    }


def test_coarse_mid_thought_start_recovers_same_objective_context() -> None:
    transcript = _transcript(
        {
            "cue_id": "setup",
            "start": 0.0,
            "end": 10.0,
            "text": "A Python closure retains its enclosing scope, making",
        },
        {
            "cue_id": "core",
            "start": 10.0,
            "end": 20.0,
            "text": "its enclosing scope available to later calls.",
        },
    )
    clip = _clip(cue_ids=["core"], start=10.0, end=20.0)

    bounds, projection, error = pipeline_module._apply_selector_acoustic_context(
        transcript,
        clip,
        {},
        (10.0, 20.0),
        sibling_clips=[clip],
    )

    assert error is None
    assert bounds == (0.0, 20.0)
    assert projection["acoustic_context"]["context_cue_ids"] == ["setup", "core"]
    assert projection["acoustic_context"]["start_explicit"] is False


def test_sponsor_context_is_not_authorized() -> None:
    transcript = _transcript(
        {
            "cue_id": "sponsor",
            "start": 0.0,
            "end": 8.0,
            "text": "Welcome back, and thanks to today's sponsor.",
        },
        {
            "cue_id": "core",
            "start": 8.0,
            "end": 18.0,
            "text": "and its enclosing scope remains available to later calls.",
        },
    )
    clip = _clip(cue_ids=["core"], start=8.0, end=18.0)

    bounds, projection, error = pipeline_module._apply_selector_acoustic_context(
        transcript,
        clip,
        {},
        (8.0, 18.0),
        sibling_clips=[clip],
    )
    context = pipeline_module._selector_authorized_acoustic_context(
        transcript,
        clip,
        {},
        sibling_clips=[clip],
    )

    assert bounds == (8.0, 18.0)
    assert error == "unresolved_acoustic_context_start"
    assert projection == {}
    assert context["context_cue_ids"] == ["core"]
    assert context["start_context_error"] == "unsafe_context_expansion"
    assert "sponsor" not in context["context_cue_ids"]


@pytest.mark.parametrize(
    ("neighbor_text", "core_text", "title", "facet", "objective", "evidence"),
    [
        (
            "A limit of a function gives the value it approaches.",
            "And the derivative of a function gives its instantaneous rate of change.",
            "Derivative of a function",
            "function",
            "Explain how the derivative of a function gives instantaneous rate of change.",
            "derivative of a function gives its instantaneous rate of change",
        ),
        (
            "The mean of a data set measures its center.",
            "And standard deviation of a data set measures its spread.",
            "Standard deviation of a data set",
            "data",
            "Explain how standard deviation of a data set measures spread.",
            "standard deviation of a data set measures its spread",
        ),
    ],
)
def test_unmarked_different_topic_prefix_is_not_authorized(
    neighbor_text: str,
    core_text: str,
    title: str,
    facet: str,
    objective: str,
    evidence: str,
) -> None:
    transcript = _transcript(
        {
            "cue_id": "neighbor",
            "start": 0.0,
            "end": 10.0,
            "text": neighbor_text,
        },
        {
            "cue_id": "core",
            "start": 10.0,
            "end": 20.0,
            "text": core_text,
        },
    )
    clip = _clip(cue_ids=["core"], start=10.0, end=20.0)
    clip["title"] = title
    clip["facet"] = facet
    clip["learning_objective"] = objective
    clip["topic_evidence_quote"] = evidence

    bounds, projection, error = pipeline_module._apply_selector_acoustic_context(
        transcript,
        clip,
        {},
        (10.0, 20.0),
        sibling_clips=[clip],
    )
    context = pipeline_module._selector_authorized_acoustic_context(
        transcript,
        clip,
        {},
        sibling_clips=[clip],
    )

    assert bounds == (10.0, 20.0)
    assert projection == {}
    assert error == "unresolved_acoustic_context_start"
    assert context["context_cue_ids"] == ["core"]
    assert context["start_context_error"] == "unsafe_context_expansion"


@pytest.mark.parametrize(
    ("core_text", "neighbor_text", "title", "facet", "objective", "evidence"),
    [
        (
            "The derivative of a function gives its instantaneous rate of change, and",
            "a limit of a function gives the value it approaches.",
            "Derivative of a function",
            "function",
            "Explain how the derivative of a function gives instantaneous rate of change.",
            "derivative of a function gives its instantaneous rate of change",
        ),
        (
            "Standard deviation of a data set measures its spread, and",
            "the mean of a data set measures its center.",
            "Standard deviation of a data set",
            "data",
            "Explain how standard deviation of a data set measures spread.",
            "standard deviation of a data set measures its spread",
        ),
    ],
)
def test_unmarked_different_topic_suffix_is_not_authorized(
    core_text: str,
    neighbor_text: str,
    title: str,
    facet: str,
    objective: str,
    evidence: str,
) -> None:
    transcript = _transcript(
        {
            "cue_id": "core",
            "start": 0.0,
            "end": 10.0,
            "text": core_text,
        },
        {
            "cue_id": "neighbor",
            "start": 10.0,
            "end": 20.0,
            "text": neighbor_text,
        },
    )
    clip = _clip(cue_ids=["core"], start=0.0, end=10.0)
    clip["title"] = title
    clip["facet"] = facet
    clip["learning_objective"] = objective
    clip["topic_evidence_quote"] = evidence

    bounds, projection, error = pipeline_module._apply_selector_acoustic_context(
        transcript,
        clip,
        {},
        (0.0, 10.0),
        sibling_clips=[clip],
    )
    context = pipeline_module._selector_authorized_acoustic_context(
        transcript,
        clip,
        {},
        sibling_clips=[clip],
    )

    assert bounds == (0.0, 10.0)
    assert projection == {}
    assert error == "unresolved_acoustic_context_end"
    assert context["context_cue_ids"] == ["core"]
    assert context["end_context_error"] == "unsafe_context_expansion"


def test_explicit_projected_and_overlap_edges_remain_exact_handoffs() -> None:
    transcript = _transcript(
        {
            "cue_id": "core",
            "start": 0.0,
            "end": 10.0,
            "text": "Welcome back. A Python closure retains its enclosing scope.",
        },
        {
            "cue_id": "next",
            "start": 9.0,
            "end": 20.0,
            "text": "Next topic begins here.",
        },
    )
    clip = _clip(cue_ids=["core"], start=0.0, end=10.0)
    clip["edge_projection"] = {
        "start": {"cue_id": "core", "quote": "A Python closure"}
    }
    overlap_handoff = pipeline_module._overlapping_caption_end_handoff(
        transcript, clip
    )
    assert overlap_handoff is not None
    projection = {
        "start": {
            "cue_id": "core",
            "mode": "projected",
            "required_speech_sec": 2.0,
            "excluded_neighbor_onset_sec": 1.5,
        },
        "caption_overlap_end_handoff": overlap_handoff[1],
    }

    context = pipeline_module._selector_authorized_acoustic_context(
        transcript,
        clip,
        projection,
        sibling_clips=[clip],
    )
    plan = pipeline_module._acoustic_boundary_plan(
        transcript,
        clip,
        projection,
        speech_bounds=(2.0, 9.0),
        search_limits=(1.5, 9.0),
    )

    assert context["start_explicit"] is True
    assert context["end_explicit"] is True
    assert plan is not None
    assert plan[2] is True
    assert plan[3] is True


def test_observation_shift_requires_no_intervening_lexical_speech() -> None:
    prepared = silence.AudioPreparationResult(
        "ready",
        source=silence.PreparedAudioSource(
            url="https://media.example/audio.m4a",
            lexical_words=tuple(
                lexical_timing.LexicalWord(f"word-{index}", float(index))
                for index in range(30)
            ),
        ),
    )
    clean = mock.Mock(
        diagnostics={
            "start_quiet": [9.7, 9.9],
            "end_quiet": [20.0, 20.2],
        }
    )
    crossed = mock.Mock(
        diagnostics={
            "start_quiet": [7.7, 7.9],
            "end_quiet": [20.0, 20.2],
        }
    )

    assert pipeline_module._acoustic_observation_shift_is_safe(
        clean,
        prepared,
        speech_bounds=(10.0, 20.0),
    )
    assert not pipeline_module._acoustic_observation_shift_is_safe(
        crossed,
        prepared,
        speech_bounds=(10.0, 20.0),
    )


def test_direct_adapter_persists_expanded_context_cues(monkeypatch) -> None:
    transcript = _transcript(
        {
            "cue_id": "setup",
            "start": 0.0,
            "end": 10.0,
            "text": "A Python closure retains its enclosing scope, making",
        },
        {
            "cue_id": "core",
            "start": 10.0,
            "end": 20.0,
            "text": "its enclosing scope available to later calls.",
        },
    )
    clip = _clip(cue_ids=["core"], start=10.0, end=20.0)
    clip["topic_evidence_quote"] = (
        "its enclosing scope available to later calls"
    )
    prepared = silence.AudioPreparationResult(
        "ready",
        source=silence.PreparedAudioSource(
            url="https://media.example/audio.m4a",
            duration_sec=20.0,
        ),
    )
    verify = mock.Mock(
        return_value=silence.SilenceVerificationResult(
            "verified",
            0.0,
            20.0,
            {
                "threshold_dbfs": -38.0,
                "start_quiet": [0.0, 0.2],
                "end_quiet": [19.9, 20.1],
                "start_speech_handoff_verified": False,
                "end_speech_handoff_verified": False,
                "start_two_sided_required": True,
                "end_two_sided_required": False,
                "semantic_start_limit_sec": 0.0,
                "semantic_end_limit_sec": 20.0,
            },
        )
    )
    monkeypatch.setattr(
        pipeline_module.clip_engine_silence,
        "verify_acoustic_boundaries",
        verify,
    )

    clips = pipeline_module._verified_direct_adapter_clips(
        source_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        engine_out={"transcript": transcript, "clips": [clip]},
        should_cancel=None,
        prepared_audio=prepared,
        require_acoustic_boundaries=True,
    )

    assert len(clips) == 1
    assert float(clip["start"]) - float(clips[0]["start"]) > 3.0
    assert clips[0]["cue_ids"] == ["setup", "core"]
    assert clips[0]["search_context"]["selection_core_cue_ids"] == ["core"]
    assert [
        cue["cue_id"]
        for cue in clips[0]["search_context"]["selection_caption_cues"]
    ] == ["setup", "core"]
