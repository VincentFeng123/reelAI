from backend.pipeline import gemini_segment as G


def _segment(index: int, text: str) -> dict:
    return {
        "cue_id": f"stats-{index}",
        "start": index * 8.0,
        "end": (index + 1) * 8.0,
        "text": text,
    }


def _plan(
    *,
    exact_request: str,
    start_line: int,
    end_line: int,
    start_quote: str,
    end_quote: str,
    claim_quote: str,
    title: str,
    objective: str,
    facet: str,
) -> G._CompactBoundaryPlan:
    return G._CompactBoundaryPlan(
        request_intent={
            "exact_request": exact_request,
            "constraints": [{
                "constraint_id": "subject",
                "kind": "subject",
                "source_phrase": exact_request,
                "requirement": f"Teach {exact_request}",
            }],
        },
        topics=[G._CompactBoundaryTopic(
            candidate_id="ap-statistics-unit",
            start_line=start_line,
            end_line=end_line,
            start_quote=start_quote,
            end_quote=end_quote,
            claim_quote=claim_quote,
            title=title,
            learning_objective=objective,
            facet=facet,
            informativeness=0.95,
            topic_relevance=0.95,
            educational_importance=0.95,
            difficulty=0.6,
            directly_teaches_topic=True,
            substantive=True,
            factually_grounded=True,
            self_contained=True,
            is_standalone=True,
            intent_evidence=[{
                "constraint_id": "subject",
                "evidence_quote": claim_quote,
            }],
        )],
    )


def test_sampling_distribution_keeps_its_context_and_stops_before_confidence_interval() -> None:
    segments = [
        _segment(
            0,
            "A sampling distribution describes how a statistic varies across repeated "
            "random samples.",
        ),
        _segment(
            1,
            "Its center is the population parameter when the statistic is an unbiased "
            "estimator.",
        ),
        _segment(
            2,
            "A confidence interval uses a sample statistic and margin of error to estimate "
            "a population parameter.",
        ),
    ]
    plan = _plan(
        exact_request="sampling distributions",
        start_line=0,
        end_line=2,
        start_quote="A sampling distribution describes",
        end_quote="estimate a population parameter",
        claim_quote="Its center is the population parameter when the statistic",
        title="Center of a sampling distribution",
        objective="Explain the center of a sampling distribution",
        facet="sampling distribution center",
    )

    report = G._plan_to_report(
        plan,
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic="sampling distributions",
    )

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["_clip_text"] == " ".join(
        str(segment["text"]) for segment in segments[:2]
    )
    assert clip["_start_line"] == 0
    assert clip["_end_line"] == 1
    assert "confidence interval" not in clip["_clip_text"].casefold()


def test_p_value_definition_drops_prior_hypothesis_test_and_later_type_errors() -> None:
    segments = [
        _segment(
            0,
            "A hypothesis test starts with null and alternative hypotheses about a "
            "population parameter.",
        ),
        _segment(
            1,
            "The hypothesis test statistic summarizes how far the sample result falls from "
            "what the null predicts.",
        ),
        _segment(
            2,
            "A p-value measures how surprising the observed result would be if the null "
            "hypothesis were true.",
        ),
        _segment(
            3,
            "A Type I error rejects a true null, whereas a Type II error fails to reject a "
            "false null.",
        ),
    ]
    plan = _plan(
        exact_request="p-values",
        start_line=0,
        end_line=3,
        start_quote="A hypothesis test starts",
        end_quote="reject a false null",
        claim_quote="A p-value measures how surprising the observed result would be",
        title="Interpreting a p-value",
        objective="Define and interpret a p-value",
        facet="p-value definition",
    )

    report = G._plan_to_report(
        plan,
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic="p-values",
    )

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["_clip_text"] == segments[2]["text"]
    assert clip["_start_line"] == clip["_end_line"] == 2
    assert "hypothesis test starts" not in clip["_clip_text"].casefold()
    assert "type i error" not in clip["_clip_text"].casefold()


def test_hypothesis_test_keeps_its_setup_and_stops_before_p_value() -> None:
    segments = [
        _segment(
            0,
            "A hypothesis test starts with null and alternative hypotheses about a "
            "population parameter.",
        ),
        _segment(
            1,
            "The hypothesis test statistic summarizes how far the sample result falls from "
            "what the null predicts.",
        ),
        _segment(
            2,
            "A p-value measures how surprising the observed result would be if the null "
            "hypothesis were true.",
        ),
        _segment(
            3,
            "A Type I error rejects a true null, whereas a Type II error fails to reject a "
            "false null.",
        ),
    ]
    plan = _plan(
        exact_request="hypothesis tests",
        start_line=0,
        end_line=3,
        start_quote="A hypothesis test starts",
        end_quote="reject a false null",
        claim_quote="The hypothesis test statistic summarizes how far the sample result",
        title="Hypothesis test evidence",
        objective="Explain how a hypothesis test summarizes sample evidence",
        facet="hypothesis test statistic",
    )

    report = G._plan_to_report(
        plan,
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic="hypothesis tests",
    )

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["_clip_text"] == " ".join(
        str(segment["text"]) for segment in segments[:2]
    )
    assert clip["_start_line"] == 0
    assert clip["_end_line"] == 1
    assert "p-value" not in clip["_clip_text"].casefold()
    assert "type i error" not in clip["_clip_text"].casefold()


def test_dangling_p_value_end_completes_the_thought_but_not_the_next_error_topic() -> None:
    segments = [
        _segment(0, "A smaller p-value provides stronger evidence against"),
        _segment(1, "the null hypothesis under its assumed sampling model."),
        _segment(2, "A Type I error occurs when a true null hypothesis is rejected."),
    ]
    plan = _plan(
        exact_request="p-values",
        start_line=0,
        end_line=0,
        start_quote="A smaller p-value provides",
        end_quote="stronger evidence against",
        claim_quote="A smaller p-value provides stronger evidence against",
        title="Evidence from a p-value",
        objective="Explain what a smaller p-value means",
        facet="p-value evidence",
    )

    report = G._plan_to_report(
        plan,
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic="p-values",
    )

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["_clip_text"] == " ".join(
        str(segment["text"]) for segment in segments[:2]
    )
    assert clip["_end_line"] == 1
    assert clip["_clip_text"].endswith("sampling model.")
    assert "Type I error" not in clip["_clip_text"]


def test_embedded_subject_split_across_captions_keeps_its_predicate() -> None:
    segments = [
        _segment(0, "A p-value is a conditional probability."),
        _segment(1, "It is the probability that our sample mean"),
        _segment(2, "when we take a sample of size n equals one hundred"),
        _segment(3, "is greater than or equal to twenty five minutes,"),
        _segment(4, "given that the null hypothesis is true."),
        _segment(5, "A confidence interval estimates a population parameter."),
    ]
    plan = _plan(
        exact_request="p-values",
        start_line=0,
        end_line=2,
        start_quote="A p-value is a conditional probability",
        end_quote="sample of size n equals one hundred",
        claim_quote="A p-value is a conditional probability",
        title="Interpreting a p-value",
        objective="Explain a p-value under the null hypothesis",
        facet="p-value interpretation",
    )

    report = G._plan_to_report(
        plan,
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic="p-values",
    )

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["_end_line"] == 4
    assert clip["_clip_text"].endswith("null hypothesis is true.")
    assert "is greater than or equal" in clip["_clip_text"]
    assert "confidence interval" not in clip["_clip_text"].casefold()


def test_boundary_prompt_demands_shortest_complete_unit_without_duration_cap() -> None:
    _system, user = G._boundary_prompts(
        "[0] 00:00 A sampling distribution describes repeated samples.\n"
        "[1] 00:08 A confidence interval estimates a parameter.",
        2,
        topic="AP Statistics",
    )
    prompt = user.casefold()

    assert "shortest concise" in prompt
    assert "necessary setup and context" in prompt
    assert "ignore acoustic silence" in prompt
    assert "only expand outward" in prompt
    assert "adjacent named concepts" in prompt
    assert "procedures" in prompt
    assert "worked examples" in prompt
    assert "decision rules" in prompt
    assert "misconceptions" in prompt
    assert "error cases" in prompt
    assert "duration is unrestricted" in prompt
    assert "duration is never a selection criterion" in prompt


def test_joint_request_subject_matching_handles_plural_query_and_singular_evidence() -> None:
    request = "hypothesis tests and p-values"
    plan = G._CompactBoundaryPlan(
        request_intent={
            "exact_request": request,
            "constraints": [
                {
                    "constraint_id": "tests",
                    "kind": "subject",
                    "source_phrase": "hypothesis tests",
                    "requirement": "Teach hypothesis tests",
                },
                {
                    "constraint_id": "join",
                    "kind": "relationship",
                    "source_phrase": "and",
                    "requirement": "Teach both components together",
                },
                {
                    "constraint_id": "pvalues",
                    "kind": "subject",
                    "source_phrase": "p-values",
                    "requirement": "Teach p-values",
                },
            ],
        },
        topics=[],
    )
    constraints, error = G._validated_intent_constraints(plan, request)

    assert error is None
    assert G._joint_subject_evidence_matches(
        constraints["pvalues"],
        "A p-value measures how surprising the observed result would be",
        constraints,
    )


def test_bare_next_step_action_is_removed_as_low_density_framing() -> None:
    text = "Then the next step is we calculate a p-value."

    assert G._cue_is_only_structural_filler(text)
    assert not G._cue_is_only_structural_filler(
        "Then the next step is we calculate the integral using substitution."
    )


def test_ordinal_sequence_labels_are_filler_but_teaching_steps_are_not() -> None:
    assert G._cue_is_only_structural_filler("So let me call that step five.")
    assert G._cue_is_only_structural_filler(
        "Then stage three, there are two situations."
    )
    assert G._cue_is_only_structural_filler(
        "We label this as phase 2."
    )
    assert not G._cue_is_only_structural_filler(
        "Step five calculates the probability under the assumed model."
    )


def test_clarification_announcements_and_cross_lesson_references_are_filler() -> None:
    assert G._cue_is_only_structural_filler(
        "Now there's one last point of clarification."
    )
    assert G._cue_is_only_structural_filler(
        "That I want to make very, very clear."
    )
    assert G._cue_is_only_structural_filler(
        "In other lessons, we have talked about how to do this."
    )
    assert G._cue_is_only_structural_filler(
        "And you will see how this comes into play in a second."
    )
    assert G._cue_is_only_structural_filler(
        "I'm putting an exclamation mark."
    )
    assert G._cue_is_only_structural_filler(
        "Because it's so conceptually important here."
    )
    assert not G._cue_is_only_structural_filler(
        "The clarification is that conditional probability reverses the order."
    )
    assert not G._cue_is_only_structural_filler(
        "This is important because it changes the probability model."
    )


def test_caption_split_dependent_condition_requires_the_following_cue() -> None:
    assert G._cue_has_weak_end(
        "This is the probability of obtaining the observed statistic",
        "given that the assumed model is true.",
        ignore_caption_case=True,
    )
    assert G._cue_has_weak_end(
        "The result remains valid",
        "under the assumption of independent observations.",
        ignore_caption_case=True,
    )
    assert not G._cue_has_weak_end(
        "This statement is complete",
        "Given that example, we now consider a separate case.",
        ignore_caption_case=True,
    )


def test_course_scope_does_not_require_spoken_evidence_for_a_joint_unit() -> None:
    request = "AP Statistics hypothesis tests and p-values"
    text = (
        "A p-value measures how surprising sample evidence would be if the null "
        "hypothesis in a hypothesis test were true."
    )
    segments = [_segment(0, text)]
    plan = G._CompactBoundaryPlan(
        request_intent={
            "exact_request": request,
            "constraints": [
                {
                    "constraint_id": "course",
                    "kind": "scope",
                    "source_phrase": "AP Statistics",
                    "requirement": "Use the AP Statistics scope",
                },
                {
                    "constraint_id": "tests",
                    "kind": "subject",
                    "source_phrase": "hypothesis tests",
                    "requirement": "Teach hypothesis tests",
                },
                {
                    "constraint_id": "join",
                    "kind": "relationship",
                    "source_phrase": "and",
                    "requirement": "Teach both components in one unit",
                },
                {
                    "constraint_id": "pvalues",
                    "kind": "subject",
                    "source_phrase": "p-values",
                    "requirement": "Teach p-values",
                },
            ],
        },
        topics=[G._CompactBoundaryTopic(
            candidate_id="joint-scope-unit",
            start_line=0,
            end_line=0,
            start_quote="A p-value measures how surprising",
            end_quote="hypothesis test were true",
            claim_quote="A p-value measures how surprising sample evidence would be",
            title="P-values in hypothesis tests",
            learning_objective="Explain a p-value under the null hypothesis",
            facet="p-value interpretation",
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
                    "constraint_id": "tests",
                    "evidence_quote": "the null hypothesis in a hypothesis test were true",
                },
                {
                    "constraint_id": "pvalues",
                    "evidence_quote": "A p-value measures how surprising sample evidence would be",
                },
            ],
        )],
    )

    report = G._plan_to_report(
        plan,
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic=request,
    )

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["intent_role"] == "primary"
    assert clip["intent_coverage"] == 1.0
    assert {
        evidence["constraint_id"] for evidence in clip["intent_evidence"]
    } == {"tests", "join", "pvalues"}
