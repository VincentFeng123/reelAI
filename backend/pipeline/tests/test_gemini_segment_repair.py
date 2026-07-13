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
    assert G._cue_has_weak_end(
        "complex organisms like protists, fungi,",
        "plants and animals complete the list.",
        ignore_caption_case=True,
    ) is True


def test_lowercase_fragment_uses_previous_unfinished_cue_as_evidence() -> None:
    segments = [
        {"text": "it can grow and develop,"},
        {"text": "reproduce, and it responds to the environment."},
        {"text": "a new sentence starts after a complete thought."},
    ]

    assert G._cue_opens_mid_thought_at(
        segments, 1, ignore_caption_case=True
    ) is True
    assert G._cue_opens_mid_thought_at(
        segments, 2, ignore_caption_case=True
    ) is False


def test_production_gene_clip_includes_the_first_selection_criterion() -> None:
    segments = [
        {
            "start": 1024.640,
            "end": 1048.640,
            "text": (
                "Genes are the core unit of natural selection. For something to "
                "undergo selection, it needs three characteristics. First, it needs "
                "to make near identical copies of itself."
            ),
        },
        {
            "start": 1048.640,
            "end": 1074.960,
            "text": (
                "Second, it needs traits that affect its interaction with the "
                "environment and its probability of survival and reproduction. "
                "What about something bigger, like a chromosome?"
            ),
        },
        {
            "start": 1074.960,
            "end": 1101.200,
            "text": "This is why the gene is the unit of natural selection.",
        },
    ]

    start, end, error = G._close_cue_context(
        segments,
        1,
        2,
        ignore_caption_case=True,
    )

    assert (start, end, error) == (0, 2, None)


def test_production_dialogue_reply_is_rejected_when_context_exceeds_repair_window() -> None:
    segments = [
        {
            "start": 0.080,
            "end": 30.555,
            "text": "Why does poop smell bad? How do you think it smells to flies?",
        },
        {
            "start": 30.555,
            "end": 58.640,
            "text": (
                "Yeah- They like it. Animals love stinky things. Poop smells good "
                "to flies because it is food, but dangerous bacteria make humans "
                "avoid it."
            ),
        },
    ]

    start, end, error = G._close_cue_context(
        segments,
        1,
        1,
        ignore_caption_case=True,
    )

    assert (start, end, error) == (1, 1, "unresolved_weak_start")


def test_production_anaphoric_question_is_rejected_without_its_antecedent() -> None:
    segments = [
        {
            "start": 137.0,
            "end": 184.0,
            "text": "Homeostasis keeps a cell's internal environment stable.",
        },
        {
            "start": 184.0,
            "end": 218.0,
            "text": (
                "Ok. But like, how does the cell do that? The secret lies in "
                "the cell membrane, which controls what goes in and out."
            ),
        },
    ]

    start, end, error = G._close_cue_context(
        segments,
        1,
        1,
        ignore_caption_case=True,
    )

    assert (start, end, error) == (1, 1, "unresolved_weak_start")


def test_production_end_extends_into_following_gerund_explanation() -> None:
    segments = [
        {
            "start": 0.0,
            "end": 44.0,
            "text": (
                "The production possibilities frontier is the same thing. "
                "It's a model"
            ),
        },
        {
            "start": 44.0,
            "end": 50.0,
            "text": "showing the tradeoffs between producing two different goods.",
        },
    ]

    start, end, error = G._close_cue_context(
        segments,
        0,
        0,
        ignore_caption_case=True,
    )

    assert (start, end, error) == (0, 1, None)


def test_production_end_extends_into_short_auxiliary_continuation() -> None:
    segments = [
        {
            "start": 4865.84,
            "end": 4870.4,
            "text": "I'll certainly give you a hint as to how two",
        },
        {
            "start": 4869.08,
            "end": 4871.88,
            "text": "will happen.",
        },
    ]

    start, end, error = G._close_cue_context(
        segments,
        0,
        0,
        ignore_caption_case=True,
    )

    assert (start, end, error) == (0, 1, None)


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
