from __future__ import annotations

import time

from backend.pipeline import gemini_segment as G


def _topic(start: int, end: int, *, title: str) -> G._BoundaryTopic:
    return G._BoundaryTopic(
        candidate_id=f"candidate-{start}-{end}-{title}",
        start_line=start,
        end_line=end,
        start_quote="So we can now" if start in {3, 9} else "Moles convert directly",
        end_quote=(
            "x equals two"
            if end == 4
            else "result is balanced"
            if end == 10
            else "mass in stoichiometry"
        ),
        title=title,
        learning_objective=f"Understand the complete {title} idea.",
        facet="worked example",
        reason=f"Teach the complete {title} idea.",
        informativeness=0.9,
        topic_relevance=0.9,
        educational_importance=0.9,
        difficulty=0.5,
        directly_teaches_topic=True,
        substantive=True,
        topic_evidence_quote=(
            "So we can now solve the equation"
            if start == 3
            else "So we can now balance the reaction"
            if start == 9
            else "Moles convert directly into mass in stoichiometry"
        ),
        self_contained=True,
        is_standalone=True,
        prerequisite_candidate_ids=[],
        uncertainty="low",
        uncertainty_reasons=[],
    )


def _transcript() -> dict:
    texts = [
        "A complete setup introduces the equation.",
        "And this gives us the first condition",
        "Because the next step depends on it",
        "So we can now solve the equation",
        "The worked result is x equals two.",
        "A transition finishes completely.",
        "A complete setup introduces equilibrium.",
        "And this gives us the second condition",
        "Because equilibrium depends on both sides",
        "So we can now balance the reaction",
        "The equilibrium result is balanced.",
        "A nearby conclusion is complete.",
        "Another nearby explanation is complete.",
        "Moles convert directly into mass in stoichiometry.",
        "A later lesson closes completely.",
        "GLOBAL DISTANT SENTINEL SHOULD NEVER ENTER REPAIR.",
    ]
    return {
        "source": "supadata",
        "words": [],
        "segments": [
            {
                "cue_id": f"cue-{index}",
                "start": index * 10.0,
                "end": (index + 1) * 10.0,
                "text": text,
            }
            for index, text in enumerate(texts)
        ],
    }


def test_unpunctuated_fixed_size_cue_edges_are_not_marked_clean() -> None:
    assert G._cue_has_weak_end(
        "we calculate the value by substituting",
        "the numbers into this equation",
        ignore_caption_case=True,
    ) is True
    assert G._cue_has_weak_end(
        "the electron moves toward the higher energy",
        "state when it absorbs a photon",
        ignore_caption_case=True,
    ) is True
    assert G._cue_boundary_confidence(
        "a short complete thought without punctuation",
        ignore_caption_case=True,
    ) < 1.0
    assert G._cue_has_weak_end(
        "ionic bonds transfer electrons between atoms",
        "covalent bonds share pairs of electrons",
        ignore_caption_case=True,
    ) is False


def test_dirty_edges_use_one_localized_low_thinking_flash_batch(monkeypatch):
    transcript = _transcript()
    selector = G._BoundaryPlan(topics=[
        _topic(3, 4, title="equation"),
        _topic(9, 10, title="equilibrium"),
        _topic(13, 13, title="stoichiometry"),
    ])
    repair = G._BoundaryRepairPlan(items=[
        G._BoundaryRepairItem(
            candidate_id="candidate-3-4-equation",
            start_line=0,
            end_line=4,
            start_quote="complete setup introduces",
            end_quote="x equals two",
        ),
        G._BoundaryRepairItem(
            candidate_id="candidate-9-10-equilibrium",
            start_line=6,
            end_line=10,
            start_quote="complete setup introduces equilibrium",
            end_quote="result is balanced",
        ),
    ])
    calls = []
    events = []
    reserve = object()

    def fake_call(system, user, schema, **kwargs):
        calls.append((system, user, schema, kwargs))
        if schema is G._BoundaryPlan:
            return selector, {"operation": kwargs["operation"]}
        assert schema is G._BoundaryRepairPlan
        return repair, {"operation": kwargs["operation"]}

    monkeypatch.setattr(G, "_call_model", fake_call)
    result = G.run_segment_profile(
        transcript,
        {
            "_segment_budget_reserve": reserve,
            "_segment_telemetry": events.append,
        },
        G.FLASH_SPLIT_PROFILE,
        topic="equations",
        deadline_monotonic=time.monotonic() + 10,
    )

    assert len(calls) == 2
    _system, repair_user, schema, kwargs = calls[1]
    assert schema is G._BoundaryRepairPlan
    assert kwargs["model"] == G.config.SEGMENT_FLASH_MODEL
    assert kwargs["thinking_level"] == "low"
    assert kwargs["max_output_tokens"] == 1_024
    assert kwargs["operation"] == "flash_boundary_repair"
    assert kwargs["prompt_version"] == G._BOUNDARY_REPAIR_PROMPT_VERSION
    assert kwargs["budget_reserve"] is reserve
    assert (
        "candidate-3-4-equation" in repair_user
        and "candidate-9-10-equilibrium" in repair_user
    )
    assert "GLOBAL DISTANT SENTINEL" not in repair_user

    assert result.classification == "green"
    assert result.accepted_count == 3
    assert {clip["title"] for clip in result.clips} == {
        "equation", "equilibrium", "stoichiometry",
    }
    assert next(
        clip for clip in result.clips if clip["title"] == "stoichiometry"
    )["boundary_confidence"] == 1.0
    assert next(
        clip for clip in result.clips if clip["title"] == "equation"
    )["boundary_confidence"] == 0.85
    assert [event["event"] for event in events] == ["boundary_repair"]
    assert events[0]["attempted_count"] == 2
    assert events[0]["accepted_count"] == 2


def test_repair_failure_preserves_independently_valid_original(monkeypatch):
    transcript = _transcript()
    selector = G._BoundaryPlan(topics=[
        _topic(3, 4, title="equation"),
        _topic(13, 13, title="stoichiometry"),
    ])
    calls = 0

    def fake_call(system, user, schema, **kwargs):
        nonlocal calls
        calls += 1
        if schema is G._BoundaryPlan:
            return selector, {"operation": kwargs["operation"]}
        raise RuntimeError("repair unavailable")

    monkeypatch.setattr(G, "_call_model", fake_call)
    result = G.run_segment_profile(
        transcript,
        {},
        G.FLASH_SPLIT_PROFILE,
        topic="stoichiometry",
        deadline_monotonic=time.monotonic() + 10,
    )

    assert calls == 2
    assert result.classification == "green"
    assert result.accepted_count == 1
    assert [clip["title"] for clip in result.clips] == ["stoichiometry"]


def test_repaired_candidates_are_validated_independently(monkeypatch):
    transcript = _transcript()
    selector = G._BoundaryPlan(topics=[
        _topic(3, 4, title="equation"),
        _topic(9, 10, title="equilibrium"),
        _topic(13, 13, title="stoichiometry"),
    ])
    repair = G._BoundaryRepairPlan(items=[
        G._BoundaryRepairItem(
            candidate_id="candidate-3-4-equation",
            start_line=0,
            end_line=4,
            start_quote="complete setup introduces",
            end_quote="x equals two",
        ),
        G._BoundaryRepairItem(
            candidate_id="candidate-9-10-equilibrium",
            start_line=6,
            end_line=10,
            start_quote="quote not present in selected cue",
            end_quote="result is balanced",
        ),
    ])

    def fake_call(system, user, schema, **kwargs):
        if schema is G._BoundaryPlan:
            return selector, {"operation": kwargs["operation"]}
        return repair, {"operation": kwargs["operation"]}

    monkeypatch.setattr(G, "_call_model", fake_call)
    result = G.run_segment_profile(
        transcript,
        {},
        G.FLASH_SPLIT_PROFILE,
        topic="equations",
        deadline_monotonic=time.monotonic() + 10,
    )

    assert result.classification == "green"
    assert {clip["title"] for clip in result.clips} == {
        "equation", "equilibrium", "stoichiometry",
    }


def test_clean_fast_path_never_dispatches_boundary_repair(monkeypatch):
    transcript = _transcript()
    selector = G._BoundaryPlan(topics=[
        _topic(13, 13, title="stoichiometry"),
    ])
    calls = []

    def fake_call(system, user, schema, **kwargs):
        calls.append(schema)
        return selector, {"operation": kwargs["operation"]}

    monkeypatch.setattr(G, "_call_model", fake_call)
    result = G.run_segment_profile(
        transcript,
        {},
        G.FLASH_SPLIT_PROFILE,
        topic="chemistry",
        deadline_monotonic=time.monotonic() + 10,
    )

    assert calls == [G._BoundaryPlan]
    assert result.classification == "green"
    assert result.accepted_count == 1
