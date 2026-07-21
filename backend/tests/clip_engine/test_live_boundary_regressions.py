import pytest

from backend.pipeline import gemini_segment as G


def _cue(video_id: str, index: int, start: float, end: float, text: str) -> dict:
    return {
        "cue_id": f"{video_id}:cue:{index}",
        "start": start,
        "end": end,
        "text": text,
    }


def _plan(
    *,
    request: str,
    start_line: int,
    end_line: int,
    start_quote: str,
    end_quote: str,
    claim_quote: str,
    objective: str,
    intent_quote: str | None = None,
) -> G._CompactBoundaryPlan:
    return G._CompactBoundaryPlan(
        request_intent={
            "exact_request": request,
            "constraints": [{
                "constraint_id": "subject",
                "kind": "subject",
                "source_phrase": request,
                "requirement": f"Teach {request}",
            }],
        },
        topics=[G._CompactBoundaryTopic(
            candidate_id="live-boundary-regression",
            start_line=start_line,
            end_line=end_line,
            start_quote=start_quote,
            end_quote=end_quote,
            claim_quote=claim_quote,
            title=objective,
            learning_objective=objective,
            facet=request,
            informativeness=0.95,
            topic_relevance=0.95,
            educational_importance=0.95,
            difficulty=0.5,
            directly_teaches_topic=True,
            substantive=True,
            factually_grounded=True,
            self_contained=True,
            is_standalone=True,
            intent_evidence=[{
                "constraint_id": "subject",
                "evidence_quote": intent_quote or claim_quote,
            }],
        )],
    )


def _report(plan: G._CompactBoundaryPlan, segments: list[dict], topic: str, *, video=False):
    settings = {"_segment_ignore_caption_case": True}
    if video:
        settings["_segment_video_url"] = "https://www.youtube.com/watch?v=testvideo01"
    return G._plan_to_report(plan, segments, [], settings, topic=topic)


def test_live_calculus_deictic_secant_clip_is_rejected_even_with_video_grounding() -> None:
    segments = [
        _cue("N2PpRnFqnqY", 29, 290.087, 297.839, (
            "one way to think about the slope of the tangent line is, well, let's "
            "calculate the slope of secant lines. Let's say between that point and that point,"
        )),
        _cue("N2PpRnFqnqY", 30, 297.839, 303.209, (
            "but then let's get even closer, say that point and that point, and then "
            "let's get even closer and that point and that point"
        )),
        _cue("N2PpRnFqnqY", 31, 303.209, 308.063, (
            "and let's see what happens as the change in x approaches zero,"
        )),
        _cue("N2PpRnFqnqY", 32, 309.346, 320.856, (
            'and so using these d\'s instead of deltas, this was Leibniz\'s way of saying, '
            '"Hey, what happens if my changes in, say, x become close to zero?" So this idea,'
        )),
        _cue("N2PpRnFqnqY", 33, 320.856, 332.63, (
            "this is known as differential notation, using super small changes in y "
            "for a super small change in x."
        )),
    ]
    plan = _plan(
        request="calculus limits to derivatives",
        start_line=0,
        end_line=3,
        start_quote="calculate the slope of secant lines",
        end_quote="Hey, what happens if my changes",
        claim_quote="see what happens as the change in x approaches zero",
        objective="Explain secant slopes approaching a tangent slope",
    )

    report = _report(plan, segments, "calculus limits to derivatives", video=True)

    assert report.clips == []
    assert "proposal_0:requires_visual_context" in report.rejected_reasons


@pytest.mark.parametrize(
    "claim_quote",
    [
        "costs that have already been incurred and cannot be recovered",
        "opportunity cost and sunk cost are two distinct Concepts",
    ],
)
def test_live_economics_comparison_drops_prior_investment_and_next_movie_examples(
    claim_quote: str,
) -> None:
    texts = [
        "will be investing in the stock market where you have the potential to earn a higher return but with more risk",
        "if you choose the low-risk savings account your opportunity cost is the potential return from the stock market",
        "the opportunity cost is the difference between the returns of the two options which is six percent",
        "opportunity cost illustrates that you are forgoing the chance to earn a higher return",
        "this concept is crucial in investment decisions as it helps investors compare potential gains and losses",
        "associated with different choices and it plays a significant role in asset allocation and portfolio management opportunity cost and sunk",
        "cost are two distinct Concepts in economics and decision-making opportunity cost indicates the value of the next best alternative that must be foregone when a",
        "decision is made and sunk cost refers to costs that have already been incurred and cannot be recovered regardless of the decision made Let's still use that movie option as an",
        "example suppose you've paid five hundred dollars for a non-refundable movie ticket but you've fallen ill",
    ]
    segments = [
        _cue("xWQgxirOFe8", index, index * 10.0, (index + 1) * 10.0, text)
        for index, text in enumerate(texts)
    ]
    comparison_evidence = "opportunity cost and sunk cost are two distinct Concepts"
    plan = G._CompactBoundaryPlan(
        request_intent={
            "exact_request": "opportunity cost versus sunk cost",
            "constraints": [
                {
                    "constraint_id": "opportunity",
                    "kind": "subject",
                    "source_phrase": "opportunity cost",
                    "requirement": "Teach opportunity cost",
                },
                {
                    "constraint_id": "sunk",
                    "kind": "subject",
                    "source_phrase": "sunk cost",
                    "requirement": "Teach sunk cost",
                },
                {
                    "constraint_id": "comparison",
                    "kind": "relationship",
                    "source_phrase": "versus",
                    "requirement": "Compare opportunity cost with sunk cost",
                },
            ],
            "joint_structures": [{
                "member_constraint_ids": ["opportunity", "sunk"],
                "relation_constraint_id": "comparison",
            }],
        },
        topics=[G._CompactBoundaryTopic(
        candidate_id="live-economics-comparison",
        start_line=0,
        end_line=8,
        start_quote="will be investing in the stock market",
        end_quote="Let's still use that movie option as an",
        claim_quote=claim_quote,
        title="Opportunity cost versus sunk cost",
        learning_objective="Distinguish opportunity cost from sunk cost",
        facet="comparison",
        informativeness=0.95,
        topic_relevance=0.95,
        educational_importance=0.95,
        difficulty=0.5,
        directly_teaches_topic=True,
        substantive=True,
        factually_grounded=True,
        self_contained=True,
        is_standalone=True,
        intent_evidence=[
            {
                "constraint_id": "opportunity",
                "evidence_quote": (
                    "opportunity cost indicates the value of the next best alternative"
                ),
            },
            {
                "constraint_id": "sunk",
                "evidence_quote": (
                    "sunk cost refers to costs that have already been incurred"
                ),
            },
            {
                "constraint_id": "comparison",
                "evidence_quote": comparison_evidence,
            },
        ],
    )],
    )

    report = _report(plan, segments, "opportunity cost versus sunk cost")

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["_clip_text"].startswith("opportunity cost and sunk cost are two distinct Concepts")
    assert clip["_clip_text"].endswith("regardless of the decision made")
    assert "stock market" not in clip["_clip_text"].casefold()
    assert "movie option" not in clip["_clip_text"].casefold()


def test_live_biology_trims_opening_back_reference_and_relative_next_video_preview() -> None:
    opening_segments = [
        _cue("Le7KOX91w7U", 4, 119.289, 151.349, (
            "This is where that oxygen comes from! Now, back to the electron transport chain. "
            "The electron transport chain transports electrons. As it does, the energy in "
            "the electrons is used to pump hydrogen ions across the thylakoid membrane."
        )),
        _cue("Le7KOX91w7U", 5, 151.349, 183.359, (
            "This flow of hydrogen ions through ATP synthase causes ATP synthase to spin and produce ATP."
        )),
    ]
    opening_plan = _plan(
        request="photosynthesis light-dependent reactions",
        start_line=0,
        end_line=1,
        start_quote="This is where that oxygen comes from",
        end_quote="ATP synthase to spin and produce ATP",
        claim_quote="energy in the electrons is used to pump hydrogen",
        objective="Explain how electron transport drives ATP production",
    )
    opening_report = _report(
        opening_plan,
        opening_segments,
        "photosynthesis light-dependent reactions",
    )
    assert opening_report.rejected_reasons == []
    assert opening_report.clips[0]["_clip_text"].startswith(
        "The electron transport chain transports electrons"
    )

    preview_segments = [_cue("Le7KOX91w7U", 6, 183.359, 211.81, (
        "NADPH is an electron carrier and is another key product of the light-dependent "
        "reactions. Both ATP and NADPH are critical products of the light-dependent "
        "reactions that are needed to make sugar in the Calvin cycle, which we will "
        "examine in our next video."
    ))]
    preview_plan = _plan(
        request="photosynthesis light-dependent reactions",
        start_line=0,
        end_line=0,
        start_quote="NADPH is an electron carrier",
        end_quote="which we will examine in our next video",
        claim_quote="NADPH is an electron carrier and is another key product",
        objective="Explain why ATP and NADPH are products of the light-dependent reactions",
    )
    preview_report = _report(
        preview_plan,
        preview_segments,
        "photosynthesis light-dependent reactions",
    )
    assert preview_report.rejected_reasons == []
    [clip] = preview_report.clips
    assert clip["_clip_text"].endswith("needed to make sugar in the Calvin cycle")
    assert "next video" not in clip["_clip_text"].casefold()


def test_live_biology_recovers_antecedent_before_relative_where_fragment() -> None:
    segments = [
        _cue("Le7KOX91w7U", 5, 151.349, 183.359, (
            "This flow of hydrogen ions through ATP synthase causes ATP synthase "
            "to spin and produce ATP. This ATP is a key product of the light-dependent "
            "reactions. When the electrons reach the end of this first electron "
            "transport chain, they go to photosystem I where light excites them once "
            "again. They travel down a second, shorter electron transport chain where "
            "they are accepted by a molecule called NADP+."
        )),
        _cue("Le7KOX91w7U", 6, 183.359, 211.81, (
            "When it accepts the electrons, it also accepts hydrogen and becomes NADPH. "
            "NADPH is an electron carrier and is another key product of the light-dependent "
            "reactions. It carries electrons and hydrogens to the next set of reactions "
            "in photosynthesis, the Calvin Cycle. Both ATP and NADPH are critical products "
            "of the light-dependent reactions that are needed to make sugar in the Calvin cycle."
        )),
    ]
    plan = _plan(
        request="photosynthesis light-dependent reactions",
        start_line=0,
        end_line=1,
        start_quote="where they are accepted by a",
        end_quote="sugar in the Calvin cycle",
        claim_quote="When it accepts the electrons, it also accepts hydrogen",
        objective="Explain how photosystem I produces NADPH",
    )

    report = _report(
        plan,
        segments,
        "photosynthesis light-dependent reactions",
    )

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["_clip_text"].startswith(
        "When the electrons reach the end of this first electron transport chain"
    )
    assert not clip["_clip_text"].casefold().startswith("where they")
    assert "ATP synthase causes" not in clip["_clip_text"]


def test_context_expansion_trims_to_latest_topic_anchored_sentence() -> None:
    segments = [
        _cue("Le7KOX91w7U", 5, 151.349, 183.359, (
            "This flow of hydrogen ions through ATP synthase causes ATP synthase "
            "to spin and produce ATP. This ATP is a key product of the light-dependent "
            "reactions. When the electrons reach the end of this first electron "
            "transport chain, they go to photosystem I where light excites them once "
            "again. They travel down a second, shorter electron transport chain where "
            "they are accepted by a molecule called NADP+."
        )),
        _cue("Le7KOX91w7U", 6, 183.359, 211.81, (
            "When it accepts the electrons, it also accepts hydrogen and becomes NADPH. "
            "NADPH is an electron carrier and is another key product of the light-dependent "
            "reactions. It carries electrons and hydrogens to the next set of reactions "
            "in photosynthesis, the Calvin Cycle. Both ATP and NADPH are critical products "
            "of the light-dependent reactions that are needed to make sugar in the Calvin cycle."
        )),
    ]
    plan = _plan(
        request="photosynthesis light-dependent reactions",
        start_line=1,
        end_line=1,
        start_quote="When it accepts the electrons, it",
        end_quote="make sugar in the Calvin cycle",
        claim_quote="When it accepts the electrons, it also accepts hydrogen",
        objective="Explain how photosystem I produces NADPH",
    )

    report = _report(
        plan,
        segments,
        "photosynthesis light-dependent reactions",
    )

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["_clip_text"].startswith(
        "When the electrons reach the end of this first electron transport chain"
    )
    assert "ATP synthase causes" not in clip["_clip_text"]
    assert "trimmed_context_expansion_to_topic_sentence" in (
        clip["_boundary_fallback_reasons"]
    )


def test_transition_evidence_semantics_are_not_reclassified_locally() -> None:
    request = "calculus limits transition to derivatives"
    evidence = "Limits describe what happens as a change approaches zero"
    segments = [_cue("transition", 0, 0.0, 8.0, evidence + ".")]
    plan = G._CompactBoundaryPlan(
        request_intent={
            "exact_request": request,
            "constraints": [
                    {
                        "constraint_id": "limits",
                        "kind": "subject",
                        "source_phrase": "calculus limits",
                        "requirement": "Teach calculus limits",
                },
                {
                    "constraint_id": "transition",
                    "kind": "relationship",
                    "source_phrase": "transition to",
                    "requirement": "Teach the transition",
                },
                {
                    "constraint_id": "derivatives",
                    "kind": "subject",
                    "source_phrase": "derivatives",
                    "requirement": "Teach derivatives",
                },
            ],
            "joint_structures": [{
                "member_constraint_ids": ["limits", "derivatives"],
                "relation_constraint_id": "transition",
            }],
        },
        topics=[G._CompactBoundaryTopic(
            candidate_id="limits-only",
            start_line=0,
            end_line=0,
            start_quote="Limits describe what happens",
            end_quote="a change approaches zero",
            claim_quote="Limits describe what happens as a change approaches zero",
            title="Limits",
            learning_objective="Explain limits as change approaches zero",
            facet="limits",
            informativeness=0.95,
            topic_relevance=0.95,
            educational_importance=0.95,
            difficulty=0.5,
            directly_teaches_topic=True,
            substantive=True,
            factually_grounded=True,
            self_contained=True,
            is_standalone=True,
            intent_evidence=[
                {"constraint_id": constraint_id, "evidence_quote": evidence}
                for constraint_id in ("limits", "transition", "derivatives")
            ],
        )],
    )

    report = _report(plan, segments, request)

    assert len(report.clips) == 1
    assert report.rejected_reasons == []


def test_dangling_article_and_same_sentence_preposition_are_incomplete_edges() -> None:
    assert G._terminal_content_is_explicitly_incomplete(
        "Let's still use that movie option as an"
    )
    text = '"Hey, what happens if my changes in, say, x become close to zero?"'
    span = G._quote_character_span(text, "Hey, what happens if my changes")
    assert span is not None
    assert G._projected_end_continues_same_sentence(text, span)


def test_live_newton_unit_derivation_requires_the_grounded_result() -> None:
    request = "how force, mass, and acceleration are related by Newton's second law"
    objective = "Explain how the Newton is derived as the SI unit of force"
    incomplete_segments = [
        _cue(
            "newton-unit",
            0,
            0.0,
            1.0,
            "equation, F = ma. What it means is that we can do",
        ),
        _cue(
            "newton-unit",
            1,
            1.0,
            14.0,
            "quantitative calculations relating the magnitude of a force applied "
            "to an object, the mass of the object, and the magnitude of the "
            "acceleration that object will experience,",
        ),
        _cue(
            "newton-unit",
            2,
            14.0,
            23.0,
            "and it shows the derivation of the Newton as the SI unit of force "
            "when we plug in 1 kilogram and one meter per second squared for mass "
            "and acceleration.",
        ),
    ]
    plan = _plan(
        request=request,
        start_line=0,
        end_line=2,
        start_quote="we can do quantitative calculations",
        end_quote="for mass and acceleration",
        claim_quote="shows the derivation of the Newton as the SI unit of force",
        objective=objective,
    )

    incomplete = G._plan_to_report(
        plan,
        incomplete_segments,
        [],
        {
            "_segment_ignore_caption_case": True,
            "_segment_trust_gemini_semantics": True,
        },
        topic=request,
    )

    assert incomplete.clips == []
    assert "proposal_0:claimed_worked_result_missing" in incomplete.rejected_reasons

    complete_segments = [
        *incomplete_segments,
        _cue(
            "newton-unit",
            3,
            23.0,
            27.0,
            "That gives one Newton, equal to one kilogram meter per second squared.",
        ),
    ]
    complete_plan = plan.model_copy(update={
        "topics": [
            plan.topics[0].model_copy(update={
                "end_line": 3,
                "end_quote": "one kilogram meter per second squared",
            })
        ],
    })

    complete = G._plan_to_report(
        complete_plan,
        complete_segments,
        [],
        {
            "_segment_ignore_caption_case": True,
            "_segment_trust_gemini_semantics": True,
        },
        topic=request,
    )

    assert complete.rejected_reasons == []
    [clip] = complete.clips
    assert clip["_clip_text"].endswith(
        "That gives one Newton, equal to one kilogram meter per second squared"
    )


def test_split_condition_is_included_before_the_next_independent_topic() -> None:
    segments = [
        _cue("condition", 0, 0.0, 2.0, "The probability measures the observed result"),
        _cue("condition", 1, 2.0, 4.0, "given that the assumed model is true."),
        _cue(
            "condition",
            2,
            4.0,
            6.0,
            "A separate misconception reverses that conditional probability.",
        ),
    ]
    plan = _plan(
        request="conditional probability",
        start_line=0,
        end_line=0,
        start_quote="The probability measures the observed result",
        end_quote="measures the observed result",
        claim_quote="The probability measures the observed result",
        objective="Explain a probability conditional on an assumed model",
    )

    report = _report(plan, segments, "conditional probability")

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["_end_line"] == 1
    assert clip["_clip_text"].endswith("given that the assumed model is true.")
    assert "misconception" not in clip["_clip_text"].casefold()


def test_sequence_only_cues_trim_before_a_substantive_decision_rule() -> None:
    segments = [
        _cue("sequence", 0, 0.0, 2.0, "So let me call that step five."),
        _cue("sequence", 1, 2.0, 4.0, "Step five, there are two situations."),
        _cue(
            "sequence",
            2,
            4.0,
            8.0,
            "If the score is below the threshold, reject the default assumption.",
        ),
    ]

    assert G._trim_structural_filler_edges(
        segments,
        0,
        2,
        ignore_caption_case=True,
    ) == (2, 2)


def test_projected_noun_fragment_is_trimmed_after_a_complete_conclusion() -> None:
    segments = [
        _cue(
            "noun-fragment",
            0,
            0.0,
            4.0,
            "A score above the threshold supports rejecting the default assumption.",
        ),
        _cue(
            "noun-fragment",
            1,
            4.0,
            6.0,
            "A world where the default assumption remains true",
        ),
        _cue(
            "noun-fragment",
            2,
            6.0,
            8.0,
            "would produce a different distribution of scores.",
        ),
    ]
    plan = _plan(
        request="decision thresholds",
        start_line=0,
        end_line=1,
        start_quote="A score above the threshold",
        end_quote="A world",
        claim_quote="A score above the threshold supports rejecting",
        objective="Explain the decision rule for a threshold",
    )

    report = _report(plan, segments, "decision thresholds")

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["_end_line"] == 0
    assert clip["_clip_text"].endswith("the default assumption.")
    assert "A world" not in clip["_clip_text"]


def test_same_cue_sequence_label_is_trimmed_before_the_teaching_claim() -> None:
    text = (
        "Step five, there are two situations. If the score is below the threshold, "
        "reject the default assumption."
    )
    plan = _plan(
        request="decision thresholds",
        start_line=0,
        end_line=0,
        start_quote="Step five there are two situations",
        end_quote="reject the default assumption",
        claim_quote="If the score is below the threshold reject the default",
        objective="Apply a threshold decision rule",
    )

    report = _report(
        plan,
        [_cue("sequence", 0, 0.0, 8.0, text)],
        "decision thresholds",
    )

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["_clip_text"].startswith("If the score is below the threshold")
    assert "Step five" not in clip["_clip_text"]


def test_mixed_same_cue_music_marker_cannot_ride_inside_teaching() -> None:
    text = (
        "Supply shortages raise prices. [Theme music] Anyway, a joke follows. "
        "Higher prices reduce demand."
    )
    plan = _plan(
        request="market adjustment",
        start_line=0,
        end_line=0,
        start_quote="Supply shortages raise prices",
        end_quote="Higher prices reduce demand",
        claim_quote="Supply shortages raise prices Theme music Anyway",
        objective="Explain a market adjustment",
    )

    report = _report(
        plan,
        [_cue("mixed-marker", 0, 0.0, 12.0, text)],
        "market adjustment",
    )

    assert report.clips == []
    assert report.rejected_reasons == [
        "proposal_0:internal_structural_filler"
    ]


def test_binary_comparison_keeps_only_the_explicit_requested_relation() -> None:
    request = "alpha versus beta"
    relation = (
        "Alpha differs from beta because alpha retains configuration whereas beta"
    )
    text = (
        "Alpha and gamma are paired side-channel mechanisms. "
        "Beta and delta are paired side-channel mechanisms. "
        f"{relation} inverts configuration."
    )
    constraints = [
        {
            "constraint_id": "alpha",
            "kind": "subject",
            "source_phrase": "alpha",
            "requirement": "Teach alpha",
        },
        {
            "constraint_id": "relation",
            "kind": "relationship",
            "source_phrase": "versus",
            "requirement": "Compare alpha with beta",
        },
        {
            "constraint_id": "beta",
            "kind": "subject",
            "source_phrase": "beta",
            "requirement": "Teach beta",
        },
    ]
    plan = G._CompactBoundaryPlan(
        request_intent={
            "exact_request": request,
            "constraints": constraints,
            "joint_structures": [{
                "member_constraint_ids": ["alpha", "beta"],
                "relation_constraint_id": "relation",
            }],
        },
        topics=[G._CompactBoundaryTopic(
            candidate_id="binary-comparison",
            start_line=0,
            end_line=0,
            start_quote="Alpha and gamma are paired side-channel mechanisms",
            end_quote="whereas beta inverts configuration",
            claim_quote=relation,
            title="Alpha versus beta",
            learning_objective="Compare alpha versus beta",
            facet="binary comparison",
            informativeness=0.95,
            topic_relevance=0.95,
            educational_importance=0.95,
            difficulty=0.5,
            directly_teaches_topic=True,
            substantive=True,
            factually_grounded=True,
            self_contained=True,
            is_standalone=True,
            intent_evidence=[
                {"constraint_id": item["constraint_id"], "evidence_quote": relation}
                for item in constraints
            ],
        )],
    )

    report = _report(plan, [_cue("comparison", 0, 0.0, 18.0, text)], request)

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["_clip_text"].startswith("Alpha differs from beta")
    assert "gamma" not in clip["_clip_text"].casefold()
    assert "delta" not in clip["_clip_text"].casefold()


def test_embedded_subject_split_expands_through_its_predicate() -> None:
    segments = [
        _cue(
            "embedded-predicate",
            0,
            0.0,
            6.0,
            "This is the chance that our measured value when the trial is repeated",
        ),
        _cue(
            "embedded-predicate",
            1,
            6.0,
            10.0,
            "is greater than the observed value.",
        ),
    ]
    plan = _plan(
        request="repeated measurements",
        start_line=0,
        end_line=0,
        start_quote="This is the chance that our measured value",
        end_quote="when the trial is repeated",
        claim_quote="chance that our measured value when the trial is repeated",
        objective="Explain a repeated-measurement chance",
    )

    report = _report(plan, segments, "repeated measurements")

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["_end_line"] == 1
    assert clip["_clip_text"].endswith("is greater than the observed value.")


def test_conditional_complement_is_kept_despite_caption_punctuation() -> None:
    segments = [
        _cue(
            "conditional-complement",
            0,
            0.0,
            5.0,
            "The estimate summarizes the observed measurements.",
        ),
        _cue(
            "conditional-complement",
            1,
            5.0,
            9.0,
            "given that the assumed model is true.",
        ),
    ]
    plan = _plan(
        request="conditional estimates",
        start_line=0,
        end_line=0,
        start_quote="The estimate summarizes the observed measurements",
        end_quote="summarizes the observed measurements",
        claim_quote="The estimate summarizes the observed measurements",
        objective="Explain a conditional estimate",
    )

    report = _report(plan, segments, "conditional estimates")

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["_end_line"] == 1
    assert clip["_clip_text"].endswith("given that the assumed model is true.")


def test_generic_atomicity_splits_unlinked_process_definitions() -> None:
    segments = [
        _cue("language-process", 0, 0.0, 5.0, "Phonology describes sound patterns."),
        _cue(
            "language-process",
            1,
            5.0,
            10.0,
            "Morphology describes how words are structured.",
        ),
    ]
    plan = _plan(
        request="language learning process",
        start_line=0,
        end_line=1,
        start_quote="Phonology describes sound patterns",
        end_quote="how words are structured",
        claim_quote="Morphology describes how words are structured",
        objective="Explain morphology in language learning",
    )

    report = _report(plan, segments, "language learning process")

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["_clip_text"] == "Morphology describes how words are structured."


def test_generic_atomicity_preserves_an_explicit_causal_process() -> None:
    segments = [
        _cue("storm-process", 0, 0.0, 5.0, "Warm water evaporates into humid air."),
        _cue("storm-process", 1, 5.0, 10.0, "The humid air then rises and cools."),
        _cue(
            "storm-process",
            2,
            10.0,
            15.0,
            "Condensation releases heat, which drives further uplift.",
        ),
    ]
    plan = _plan(
        request="how a storm develops",
        start_line=0,
        end_line=2,
        start_quote="Warm water evaporates into humid air",
        end_quote="which drives further uplift",
        claim_quote="Condensation releases heat which drives further uplift",
        objective="Explain how a storm develops",
    )

    report = _report(plan, segments, "how a storm develops")

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["_start_line"] == 0
    assert clip["_end_line"] == 2


def test_embedded_subject_split_across_cues_keeps_its_predicate_and_condition() -> None:
    segments = [
        _cue("embedded-rule", 0, 0.0, 2.0, "The rule states that the selected value"),
        _cue("embedded-rule", 1, 2.0, 4.0, "when the condition holds"),
        _cue("embedded-rule", 2, 4.0, 6.0, "is at least ten,"),
        _cue("embedded-rule", 3, 6.0, 8.0, "given the stated assumption."),
        _cue(
            "embedded-rule",
            4,
            8.0,
            10.0,
            "A separate rule applies to another case.",
        ),
    ]

    start_line, end_line, error = G._close_cue_context(
        segments,
        0,
        1,
        ignore_caption_case=True,
        cue_limit=len(segments),
    )

    assert error is None
    assert (start_line, end_line) == (0, 3)
    assert G._cue_clip_text(segments, start_line, end_line).endswith(
        "is at least ten, given the stated assumption."
    )
