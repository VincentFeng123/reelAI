from __future__ import annotations

from backend.pipeline import gemini_segment


def _proposal(
    *,
    candidate_id: str,
    start_line: int,
    end_line: int,
    start_quote: str,
    end_quote: str,
    evidence: str,
    objective: str,
) -> gemini_segment._BoundaryTopic:
    return gemini_segment._BoundaryTopic(
        candidate_id=candidate_id,
        start_line=start_line,
        end_line=end_line,
        start_quote=start_quote,
        end_quote=end_quote,
        title=objective,
        learning_objective=objective,
        facet="calculus",
        reason="The span directly teaches the requested calculus concept.",
        informativeness=0.9,
        topic_relevance=0.9,
        educational_importance=0.9,
        difficulty=0.3,
        directly_teaches_topic=True,
        substantive=True,
        factually_grounded=True,
        topic_evidence_quote=evidence,
        self_contained=True,
        is_standalone=True,
        prerequisite_candidate_ids=[],
        uncertainty="medium",
        uncertainty_reasons=[gemini_segment._UncertaintyReason.BOUNDARY_AMBIGUOUS],
    )


def _report(
    segments: list[dict],
    proposal: gemini_segment._BoundaryTopic,
    *,
    topic: str = "calculus limits",
) -> gemini_segment._Conversion:
    return gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=[proposal]),
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic=topic,
    )


def _cue(cue_id: str, start: float, end: float, text: str) -> dict:
    return {"cue_id": cue_id, "start": start, "end": end, "text": text}


def _youtube_cues(raw_cues: list[tuple[str | int, float, float, str]]) -> list[dict]:
    return [
        _cue(f"YNstP0ESndU:cue:{cue_id}", start, end, text)
        for cue_id, start, end, text in raw_cues
    ]


def test_course_preview_recovers_forward_to_the_complete_limit_problem() -> None:
    raw_cues = [
        (0, 1.520, 5.600, "in this video we're just going to go"),
        (1, 2.720, 8.320, "over a basic introduction into limits"),
        (2, 5.600, 9.920, "and how to evaluate them analytically"),
        (3, 8.320, 14.400, "and graphically so here's a simple example"),
        (4, 12.080, 16.480, "let's say if we want to find the limit"),
        (5, 14.400, 19.359, "as x approaches two"),
        (6, 16.480, 23.199, "of the function x squared minus four"),
        (7, 19.359, 26.880, "divided by x minus two so how can we do so"),
        (8, 25.439, 31.359, "well one way is to use direct substitution"),
        (
            9,
            31.359,
            37.000,
            "values on both sides of two make the function approach four",
        ),
        (10, 37.000, 41.000, "therefore the limit as x approaches two is four."),
    ]
    segments = _youtube_cues(raw_cues)
    proposal = _proposal(
        candidate_id="limit-introduction",
        start_line=0,
        end_line=10,
        start_quote="in this video we're just going to go",
        end_quote="limit as x approaches two is four",
        evidence="values on both sides of two make the function approach four",
        objective="Evaluate a limit from nearby function values",
    )

    report = _report(segments, proposal)

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["cue_ids"][0] == "YNstP0ESndU:cue:4"
    assert clip["_clip_text"].startswith("let's say if we want to find the limit")
    assert "in this video" not in clip["_clip_text"].casefold()
    assert "basic introduction" not in clip["_clip_text"].casefold()


def test_contextual_example_requires_setup_but_a_local_restatement_is_valid() -> None:
    assert not gemini_segment._opening_clause_is_standalone(
        "in this particular example we could factor"
    )
    assert gemini_segment._opening_clause_is_standalone(
        "In this particular example, the limit as x approaches two of x squared "
        "minus four over x minus two gives zero over zero, so factor the numerator."
    )

    missing_setup = [
        _cue(
            "YNstP0ESndU:cue:44", 157.599, 162.959,
            "in this particular example we could factor",
        ),
        _cue(
            "YNstP0ESndU:cue:45", 160.240, 165.840,
            "x squared minus four you can write it as",
        ),
        _cue(
            "YNstP0ESndU:cue:46", 162.959, 168.240,
            "x plus two times x minus two.",
        ),
    ]
    missing_proposal = _proposal(
        candidate_id="factor-without-limit-setup",
        start_line=0,
        end_line=2,
        start_quote="in this particular example we could factor",
        end_quote="x plus two times x minus two",
        evidence="x squared minus four you can write it as x plus two times x minus two",
        objective="Evaluate a limit by factoring and canceling its undefined denominator",
    )
    assert _report(missing_setup, missing_proposal).clips == []

    local_setup = [
        _cue(
            "local:0", 0.0, 9.0,
            "In this particular example, the limit as x approaches two of x "
            "squared minus four over x minus two gives zero over zero, so factor "
            "the numerator.",
        ),
        _cue(
            "local:1", 9.0, 17.0,
            "Write the numerator as x plus two times x minus two and cancel "
            "the common x minus two factor.",
        ),
        _cue(
            "local:2", 17.0, 23.0,
            "Substituting two into x plus two gives four, so the limit is four.",
        ),
    ]
    local_proposal = _proposal(
        candidate_id="factor-with-local-limit-setup",
        start_line=0,
        end_line=2,
        start_quote="In this particular example the limit as x approaches two",
        end_quote="so the limit is four",
        evidence="cancel the common x minus two factor",
        objective="Evaluate a limit by factoring and canceling its undefined denominator",
    )

    local_report = _report(local_setup, local_proposal)

    assert local_report.rejected_reasons == []
    assert local_report.clips[0]["cue_ids"] == ["local:0", "local:1", "local:2"]


def test_cross_cue_evidence_trims_the_overlapping_next_example_transition() -> None:
    raw_cues = [
        (
            "setup",
            180.000,
            188.000,
            "After factoring and canceling x minus two, the expression is x plus two.",
        ),
        (
            "54",
            186.159,
            193.120,
            "Now all we need to do is find the limit",
        ),
        ("55", 189.280, 195.519, "as x approaches two of x plus two"),
        (
            "56",
            193.120,
            197.200,
            "so now we can replace x with two and two",
        ),
        ("57", 195.519, 202.159, "plus two is four and so that's the limit"),
        ("58", 198.800, 205.200, "it approaches a value of 4."),
        ("59", 202.159, 207.120, "now let's look at another example"),
        ("60", 205.200, 210.799, "what is the limit as x approaches 5"),
    ]
    segments = _youtube_cues(raw_cues)
    proposal = _proposal(
        candidate_id="factored-limit-answer",
        start_line=0,
        end_line=6,
        start_quote="After factoring and canceling x minus two",
        end_quote="now let's look at another example",
        evidence="find the limit as x approaches two of x plus two",
        objective="Evaluate a factored limit after canceling the undefined factor",
    )

    report = _report(segments, proposal)

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["cue_ids"][-1] == "YNstP0ESndU:cue:58"
    assert "look at another example" not in clip["_clip_text"].casefold()
    assert "approaches 5" not in clip["_clip_text"].casefold()


def test_terminal_now_sometimes_completes_the_following_teaching_sentence() -> None:
    raw_cues = [
        (
            "setup",
            141.000,
            147.120,
            "A two-sided limit exists when both sides approach the same value.",
        ),
        ("41", 147.120, 151.840, "if the limit exists it's going to"),
        ("42", 148.640, 156.239, "converge to a certain value now sometimes"),
        (
            "43",
            153.840,
            159.360,
            "you have to use other techniques to get the answer",
        ),
        (
            "44",
            157.599,
            162.959,
            "in this particular example we could factor",
        ),
    ]
    segments = _youtube_cues(raw_cues)
    proposal = _proposal(
        candidate_id="limit-convergence-methods",
        start_line=0,
        end_line=2,
        start_quote="A two-sided limit exists when both sides approach",
        end_quote="converge to a certain value now sometimes",
        evidence="converge to a certain value",
        objective="Explain when a limit converges and when another technique is needed",
    )

    report = _report(segments, proposal)

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["cue_ids"][-1] == "YNstP0ESndU:cue:43"
    assert clip["_clip_text"].endswith("you have to use other techniques to get the answer")
    assert "in this particular example" not in clip["_clip_text"].casefold()


def test_terminal_example_outro_is_trimmed_without_broad_or_internal_matches() -> None:
    separate_cues = [
        _cue(
            "direct:setup", 232.000, 240.879,
            "For the polynomial limit as x approaches five, direct substitution "
            "gives twenty-five plus ten minus four.",
        ),
        _cue(
            "YNstP0ESndU:cue:70", 240.879, 247.360,
            "so the limit is going to be 31.",
        ),
        _cue(
            "YNstP0ESndU:cue:71", 244.319, 248.799,
            "and so that's it for that example",
        ),
        _cue(
            "YNstP0ESndU:cue:72", 247.360, 253.439,
            "but now what about this one what is the limit",
        ),
    ]
    separate_proposal = _proposal(
        candidate_id="direct-substitution-answer",
        start_line=0,
        end_line=2,
        start_quote="For the polynomial limit as x approaches five",
        end_quote="and so that's it for that example",
        evidence="direct substitution gives twenty-five plus ten minus four",
        objective="Apply direct substitution to a polynomial limit",
    )

    separate_report = _report(separate_cues, separate_proposal)

    assert separate_report.rejected_reasons == []
    [separate_clip] = separate_report.clips
    assert separate_clip["cue_ids"][-1] == "YNstP0ESndU:cue:70"
    assert "that's it for that example" not in separate_clip["_clip_text"].casefold()

    same_cue_text = (
        "Direct substitution gives thirty-one, so the limit is thirty-one. "
        "And so that's it for that example"
    )
    same_cue_proposal = _proposal(
        candidate_id="same-cue-example-outro",
        start_line=0,
        end_line=0,
        start_quote="Direct substitution gives thirty-one",
        end_quote="that's it for that example",
        evidence="Direct substitution gives thirty-one so the limit is thirty-one",
        objective="Apply direct substitution to a polynomial limit",
    )
    same_cue_report = _report(
        [_cue("same", 0.0, 10.0, same_cue_text)],
        same_cue_proposal,
    )
    assert same_cue_report.rejected_reasons == []
    assert same_cue_report.clips[0]["_clip_text"].rstrip(" .!?").endswith(
        "the limit is thirty-one"
    )

    substantive = (
        "That's it: the derivative is zero because constant functions do not change."
    )
    assert gemini_segment._unconditional_trailing_edge_noise_start(
        substantive,
        require_edge_prefix=True,
    ) is None

    internal = (
        "The limit equals four. And so that's it for that example. This result "
        "establishes that factoring preserves the limit after cancellation."
    )
    internal_proposal = _proposal(
        candidate_id="internal-example-closure",
        start_line=0,
        end_line=0,
        start_quote="The limit equals four",
        end_quote="preserves the limit after cancellation",
        evidence="factoring preserves the limit after cancellation",
        objective="Explain why cancellation preserves the factored limit",
    )
    internal_report = _report(
        [_cue("internal", 0.0, 12.0, internal)],
        internal_proposal,
    )
    assert internal_report.rejected_reasons == []
    assert "that's it for that example" in internal_report.clips[0][
        "_clip_text"
    ].casefold()


def test_spatially_deictic_derivative_narration_needs_visual_context() -> None:
    visual_text = (
        "If we calculate the average rate of change between this point and this "
        "point, it is the slope of this line, the secant line. Pick this point and "
        "this point; it looks like it has a higher slope. What if we draw a tangent "
        "line to this point, a line that touches the graph right over there? It might "
        "look something like that. Its slope is the instantaneous rate of change at "
        "that point."
    )
    visual_proposal = _proposal(
        candidate_id="khan-deictic-derivative",
        start_line=0,
        end_line=0,
        start_quote="If we calculate the average rate of change",
        end_quote="instantaneous rate of change at that point",
        evidence="Its slope is the instantaneous rate of change at that point",
        objective="Explain instantaneous rate of change using secant and tangent lines",
    )

    visual_report = _report(
        [_cue("N2PpRnFqnqY:deictic", 128.0, 260.0, visual_text)],
        visual_proposal,
        topic="calculus derivatives",
    )

    assert visual_report.clips == []
    assert visual_report.rejected_reasons == ["proposal_0:requires_visual_context"]


    verbal_text = (
        "A derivative at x equals two is the limit of average rates of change as a "
        "second input approaches two. This limit gives the instantaneous rate of "
        "change at x equals two."
    )
    verbal_proposal = _proposal(
        candidate_id="verbal-derivative-definition",
        start_line=0,
        end_line=0,
        start_quote="A derivative at x equals two",
        end_quote="instantaneous rate of change at x equals two",
        evidence="limit of average rates of change as a second input approaches two",
        objective="Define a derivative as a limit of average rates of change",
    )
    verbal_report = _report(
        [_cue("verbal", 0.0, 14.0, verbal_text)],
        verbal_proposal,
        topic="calculus derivatives",
    )
    assert verbal_report.rejected_reasons == []
    assert len(verbal_report.clips) == 1

    locally_grounded = (
        "Let x equal two. At this point, the derivative equals four, so the tangent "
        "slope is four."
    )
    assert not gemini_segment._clip_requires_visual_context(locally_grounded)
    assert not gemini_segment._clip_requires_visual_context(
        "Let L be y equals two x plus one. The slope of this line is two."
    )


def test_nonvisual_long_clip_skips_sentence_level_visual_scan(monkeypatch) -> None:
    def unexpected_scan(*_args, **_kwargs):
        raise AssertionError("ordinary nonvisual speech should use the whole-text fast path")

    monkeypatch.setattr(
        gemini_segment,
        "_sentence_requires_visual_context",
        unexpected_scan,
    )

    text = " ".join(
        f"Step {index} explains a complete mathematical idea with spoken context."
        for index in range(1_000)
    )
    assert not gemini_segment._clip_requires_visual_context(text)


def test_long_complete_clip_checks_trailing_suffix_once(monkeypatch) -> None:
    segments = [
        _cue(
            f"cue-{index}",
            float(index),
            float(index + 1),
            f"This complete sentence explains educational step {index}.",
        )
        for index in range(500)
    ]
    original = gemini_segment._cue_clip_text
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(gemini_segment, "_cue_clip_text", counted)

    assert gemini_segment._trim_trailing_incomplete_suffix(
        segments,
        0,
        len(segments) - 1,
    ) == len(segments) - 1
    assert calls <= 2


def test_long_reset_free_clip_skips_per_cue_transition_scans(monkeypatch) -> None:
    segments = [
        _cue(
            f"cue-{index}",
            float(index),
            float(index + 1),
            f"Educational step {index} develops the same coherent objective.",
        )
        for index in range(1_000)
    ]

    def unexpected_scan(_text):
        raise AssertionError("reset-free candidates should not segment each cue")

    monkeypatch.setattr(
        gemini_segment,
        "_sentence_character_spans",
        unexpected_scan,
    )

    assert gemini_segment._candidate_topic_transitions(
        segments,
        0,
        len(segments) - 1,
        evidence_quote="Educational step 500 develops the same coherent objective",
        learning_objective="Develop one coherent educational objective",
    ) == []


def test_agenda_recovery_keeps_a_substantive_setup_before_later_work() -> None:
    segments = [
        _cue("agenda", 0.0, 3.0, "In this video we're going to cover limits"),
        _cue(
            "setup",
            3.0,
            8.0,
            "The limit problem is x squared minus four divided by x minus two.",
        ),
        _cue(
            "method",
            8.0,
            13.0,
            "Calculate values near two to see the function approach four.",
        ),
        _cue("answer", 13.0, 16.0, "Therefore the limit is four."),
    ]
    proposal = _proposal(
        candidate_id="agenda-before-required-setup",
        start_line=0,
        end_line=3,
        start_quote="In this video we're going to cover limits",
        end_quote="the limit is four",
        evidence="values near two to see the function approach four",
        objective="Evaluate the stated limit from nearby values",
    )

    report = _report(segments, proposal)

    assert report.rejected_reasons == []
    assert report.clips[0]["cue_ids"][0] == "setup"
    assert report.clips[0]["_clip_text"].startswith("The limit problem is")


def test_ordinary_here_is_exposition_is_not_a_topic_reset() -> None:
    segments = [
        _cue(
            "intuition",
            0.0,
            5.0,
            "Here is the intuition for why the derivative measures change.",
        ),
        _cue(
            "definition",
            5.0,
            10.0,
            "The limit of secant slopes gives the tangent slope.",
        ),
    ]

    assert gemini_segment._candidate_topic_transitions(
        segments,
        0,
        1,
        evidence_quote="limit of secant slopes",
        learning_objective="Explain derivative intuition using secant slopes",
    ) == []


def test_agenda_recovery_keeps_same_cue_and_question_setups() -> None:
    same_cue = [
        _cue(
            "agenda-and-setup",
            0.0,
            9.0,
            "In this video we're going to cover limits. The problem is the limit "
            "as x approaches two of x squared minus four over x minus two.",
        ),
        _cue(
            "method",
            9.0,
            15.0,
            "Calculate by factoring the numerator and canceling x minus two.",
        ),
        _cue("answer", 15.0, 19.0, "Substitution then gives the limit four."),
    ]
    same_cue_proposal = _proposal(
        candidate_id="same-cue-agenda-setup",
        start_line=0,
        end_line=2,
        start_quote="In this video we're going to cover limits",
        end_quote="gives the limit four",
        evidence="factoring the numerator and canceling x minus two",
        objective="Evaluate the stated limit by factoring",
    )

    same_cue_report = _report(same_cue, same_cue_proposal)

    assert same_cue_report.rejected_reasons == []
    assert same_cue_report.clips[0]["_clip_text"].startswith(
        "The problem is the limit"
    )

    split_question = [
        _cue("agenda", 0.0, 3.0, "In this video we're going to go over limits"),
        _cue(
            "question",
            3.0,
            9.0,
            "And how do we evaluate the limit at two? First factor the numerator.",
        ),
        _cue(
            "answer",
            9.0,
            14.0,
            "Canceling the common factor and substituting gives four.",
        ),
    ]
    question_proposal = _proposal(
        candidate_id="question-after-agenda",
        start_line=0,
        end_line=2,
        start_quote="In this video we're going to go over limits",
        end_quote="substituting gives four",
        evidence="evaluate the limit at two First factor the numerator",
        objective="Evaluate a limit by factoring",
    )

    question_report = _report(split_question, question_proposal)

    assert question_report.rejected_reasons == []
    assert question_report.clips[0]["cue_ids"][0] == "question"


def test_contextual_example_requires_a_grounded_object_not_a_pronoun() -> None:
    assert not gemini_segment._opening_clause_is_standalone(
        "In this particular example, calculate it by factoring."
    )
    assert not gemini_segment._opening_clause_is_standalone(
        "In this particular example, consider the next step."
    )
    for generic_setup in (
        "In this example, calculate the answer.",
        "In this example, find the result.",
        "In this example, solve the problem.",
        "In this example, let it equal zero.",
        "In this example, assume it is defined as x squared.",
        "In this example, consider this is written as x plus two.",
        "In this example, the problem is this: factor it.",
        "In this example, the function is this one.",
        "In this example, the equation is this, so solve it.",
        "In this example, the problem is to calculate the answer.",
        "In this example, the equation is given.",
        "In this example, the function is unknown.",
        "In this example, the expression is the same.",
        "In this example, let x.",
        "In this example, given x.",
        "In this example, given x, find the answer.",
    ):
        assert not gemini_segment._opening_clause_is_standalone(generic_setup)
    for grounded_setup in (
        "In this example, calculate the derivative of x squared.",
        "In this example, find x.",
        "In this example, let x equal zero.",
        "In this example, assume this curve is defined as y equals x squared.",
        "In this example, the function f equals x squared, so differentiate it.",
        "In this example, the equation is x equals zero.",
        "In this example, the function is sine.",
        "In this example, the expression is zero.",
        "In this example, let x equal two.",
        "In this example, given x squared plus one, factor the expression.",
    ):
        assert gemini_segment._opening_clause_is_standalone(grounded_setup)
    assert gemini_segment._opening_clause_is_standalone(
        "In this particular example, the limit as x approaches two is undefined, "
        "so factor it. After cancellation, substitution gives four."
    )


def test_contextual_example_accepts_a_later_same_cue_grounded_setup() -> None:
    segments = [
        _cue(
            "same-cue-setup",
            0.0,
            10.0,
            "In this example, we will solve it. The problem is the limit as x "
            "approaches two of x squared minus four over x minus two.",
        ),
        _cue(
            "method",
            10.0,
            16.0,
            "Factor the numerator and cancel the common x minus two factor.",
        ),
        _cue("answer", 16.0, 20.0, "Substitution then gives the limit four."),
    ]
    proposal = _proposal(
        candidate_id="same-cue-context-recovery",
        start_line=0,
        end_line=2,
        start_quote="In this example we will solve it",
        end_quote="gives the limit four",
        evidence="Factor the numerator and cancel the common x minus two factor",
        objective="Evaluate the stated limit by factoring",
    )

    report = _report(segments, proposal)

    assert report.rejected_reasons == []
    assert report.clips[0]["_clip_text"].startswith("The problem is the limit")


def test_generic_backward_context_does_not_masquerade_as_an_example_setup() -> None:
    segments = [
        _cue("generic", 0.0, 3.0, "Calculate the answer."),
        _cue("contextual", 3.0, 7.0, "In this example we could factor."),
        _cue("answer", 7.0, 11.0, "After cancellation, the answer is four."),
    ]
    proposal = _proposal(
        candidate_id="generic-backward-context",
        start_line=1,
        end_line=2,
        start_quote="In this example we could factor",
        end_quote="the answer is four",
        evidence="After cancellation the answer is four",
        objective="Evaluate a limit by factoring",
    )

    report = _report(segments, proposal)

    assert report.clips == []
    assert report.rejected_reasons == ["proposal_0:unresolved_example_setup"]


def test_contextual_example_accepts_a_complete_target_restated_later() -> None:
    segments = [
        _cue(
            "prior-method",
            0.0,
            4.0,
            "Direct substitution completes the previous limit when its denominator is nonzero.",
        ),
        _cue(
            "factor",
            4.0,
            10.0,
            "In this particular example we could factor x squared minus four. "
            "It becomes x plus two times x minus two.",
        ),
        _cue(
            "cancel",
            10.0,
            16.0,
            "Cancel x minus two because that factor creates zero in the denominator.",
        ),
        _cue(
            "restated-target",
            16.0,
            21.0,
            "Now find the limit as x approaches two of x plus two.",
        ),
        _cue(
            "answer",
            21.0,
            26.0,
            "Substitution gives four, so the limit approaches a value of four.",
        ),
    ]
    proposal = _proposal(
        candidate_id="later-restated-target",
        start_line=1,
        end_line=4,
        start_quote="In this particular example we could factor",
        end_quote="limit approaches a value of four",
        evidence="Cancel x minus two because that factor creates zero in the denominator",
        objective="Evaluate a limit by factoring and cancellation",
    )

    report = _report(segments, proposal)

    assert report.rejected_reasons == []
    assert len(report.clips) == 1
    assert report.clips[0]["cue_ids"][0] == "factor"
    assert "prior-method" not in report.clips[0]["cue_ids"]
    assert "accepted_later_restated_example_target" in report.clips[0][
        "_boundary_fallback_reasons"
    ]


def test_method_framing_does_not_invalidate_a_complete_recovered_setup() -> None:
    segments = [
        _cue(
            "setup",
            0.0,
            8.0,
            "There are different ways to evaluate the limit as x approaches two "
            "of x squared minus four over x minus two.",
        ),
        _cue("contextual", 8.0, 12.0, "In this example we could factor."),
        _cue(
            "work-and-answer",
            12.0,
            20.0,
            "Cancel the common x minus two factor; substitution then gives four.",
        ),
    ]
    proposal = _proposal(
        candidate_id="grounded-method-framing",
        start_line=1,
        end_line=2,
        start_quote="In this example we could factor",
        end_quote="substitution then gives four",
        evidence="Cancel the common x minus two factor substitution then gives four",
        objective="Evaluate the stated limit by factoring",
    )

    report = _report(segments, proposal)

    assert report.rejected_reasons == []
    assert report.clips[0]["cue_ids"][0] == "setup"


def test_contextual_educational_unit_heads_require_their_setup() -> None:
    for fragment in (
        "In this proof, we now apply induction.",
        "In this derivation, substitute it.",
        "In this calculation, cancel the term.",
        "In this exercise, find the answer.",
    ):
        assert not gemini_segment._opening_clause_is_standalone(fragment)
    assert gemini_segment._opening_clause_is_standalone(
        "In this proof, assume n equals one and establish the base case."
    )
    for grounded in (
        "In this example, a bacterial population doubles every hour.",
        "In this case, mitochondria produce ATP through oxidative phosphorylation.",
        "In this proof, every continuous function on a closed interval attains a maximum.",
        "For this derivation, energy conservation gives the particle speed.",
        "In this exercise, identify the Carolingian minuscule ligature.",
        "In this demonstration, adding acid turns the indicator red.",
        "In this example, how does photosynthesis convert light into chemical energy?",
        "In this example, a bacterial population doubled every hour.",
        "In this case, mitochondria synthesized ATP.",
        "In this example, Louis XVI convened the Estates General.",
        "In this case, the Treaty of Versailles imposed reparations.",
        "In this example, Carolingian minuscule standardized letterforms.",
        "In this case, photosynthesis transforms light into chemical energy.",
        "In this example, the process of photosynthesis transforms light energy.",
    ):
        assert gemini_segment._opening_clause_is_standalone(grounded)
    for unresolved in (
        "In this example, we factor it.",
        "In this proof, we now apply induction.",
        "In this derivation, substitute it.",
        "In this calculation, cancel the term.",
        "In this exercise, find the answer.",
        "In this exercise, explain the thing carefully.",
        "In this exercise, analyze it carefully.",
        "In this exercise, identify the answer exactly.",
        "In this exercise, describe this briefly.",
        "In this exercise, compare them carefully.",
        "In this example, the process changes.",
        "In this example, the concept explains growth.",
    ):
        assert not gemini_segment._opening_clause_is_standalone(unresolved)


def test_unpunctuated_agenda_keeps_the_same_cue_problem_setup() -> None:
    segments = [
        _cue(
            "agenda-and-problem",
            0.0,
            10.0,
            "In this video we're going to cover limits the problem is the limit "
            "as x approaches two of x squared minus four over x minus two",
        ),
        _cue(
            "method",
            10.0,
            15.0,
            "Factor the numerator and cancel the common x minus two factor.",
        ),
        _cue("answer", 15.0, 19.0, "Substitution gives the limit four."),
    ]
    proposal = _proposal(
        candidate_id="unpunctuated-agenda-setup",
        start_line=0,
        end_line=2,
        start_quote="In this video we're going to cover limits",
        end_quote="gives the limit four",
        evidence="Factor the numerator and cancel the common x minus two factor",
        objective="Evaluate the stated limit by factoring",
    )

    report = _report(segments, proposal)

    assert report.rejected_reasons == []
    assert report.clips[0]["_clip_text"].startswith("the problem is the limit")
    assert "going to cover" not in report.clips[0]["_clip_text"].casefold()


def test_navigation_split_across_cues_cannot_mix_two_learning_units() -> None:
    segments = [
        _cue(
            "limits",
            0.0,
            8.0,
            "After factoring, the limit is four. Now let us look at",
        ),
        _cue(
            "derivatives",
            8.0,
            16.0,
            "another example. The derivative is the instantaneous rate of change.",
        ),
    ]
    old_side = _proposal(
        candidate_id="old-side-limit",
        start_line=0,
        end_line=1,
        start_quote="After factoring the limit is four",
        end_quote="instantaneous rate of change",
        evidence="After factoring the limit is four",
        objective="Explain the result of the factored limit",
    )
    old_report = _report(segments, old_side)
    assert old_report.rejected_reasons == []
    assert old_report.clips[0]["cue_ids"] == ["limits"]
    assert "look at" not in old_report.clips[0]["_clip_text"].casefold()

    new_side = _proposal(
        candidate_id="new-side-derivative",
        start_line=0,
        end_line=1,
        start_quote="After factoring the limit is four",
        end_quote="instantaneous rate of change",
        evidence="derivative is the instantaneous rate of change",
        objective="Define a derivative as instantaneous rate of change",
    )
    new_report = _report(segments, new_side, topic="calculus derivatives")
    assert new_report.rejected_reasons == []
    assert new_report.clips[0]["cue_ids"] == ["derivatives"]
    assert new_report.clips[0]["_clip_text"].startswith("The derivative is")

    stitched = old_side.model_copy(update={
        "candidate_id": "stitched-reset-evidence",
        "topic_evidence_quote": (
            "limit is four Now let us look at another example"
        ),
    })
    stitched_report = _report(segments, stitched)
    assert stitched_report.clips == []
    assert stitched_report.rejected_reasons == [
        "proposal_0:topic_evidence_crosses_topic_reset"
    ]


def test_complete_projected_end_does_not_expand_for_unselected_now_sometimes() -> None:
    segments = [
        _cue(
            "limit",
            0.0,
            10.0,
            "Direct substitution gives four. So the limit is four. Now sometimes",
        ),
        _cue(
            "derivative",
            10.0,
            15.0,
            "we study derivative rules instead.",
        ),
    ]
    proposal = _proposal(
        candidate_id="complete-before-unselected-leadin",
        start_line=0,
        end_line=0,
        start_quote="Direct substitution gives four",
        end_quote="the limit is four",
        evidence="Direct substitution gives four So the limit is four",
        objective="Evaluate a limit by direct substitution",
    )

    report = _report(segments, proposal)

    assert report.rejected_reasons == []
    assert report.clips[0]["cue_ids"] == ["limit"]
    assert "derivative" not in report.clips[0]["_clip_text"].casefold()
    assert not gemini_segment._clip_requires_visual_context(
        "Let this point be x equals one and that point be x equals three. Values "
        "between this point and that point lie in the interval from one to three."
    )


def test_spoken_formula_after_looks_like_this_is_not_visual_only() -> None:
    assert not gemini_segment._clip_requires_visual_context(
        "The quadratic formula looks like this: x equals negative b plus or minus "
        "the square root of b squared minus four a c, all over two a."
    )
    assert not gemini_segment._clip_requires_visual_context(
        "The derivative looks like this: f prime of x equals two x."
    )
    assert gemini_segment._clip_requires_visual_context(
        "The curve looks like this."
    )
    for spoken in (
        "Let's look at this another way: substitution preserves the value.",
        "Look at this from an algebraic perspective: both factors cancel.",
        "Let's look at this question: why does the limit equal four?",
        "Watch this variable cancel algebraically: x minus two cancels.",
        "I'm writing a proof in complete spoken sentences.",
    ):
        assert not gemini_segment._clip_requires_visual_context(spoken)
    for visual in (
        "Look at this graph and notice the curve rises.",
        "As you can see, the curve rises here.",
        "The diagram shows how the process works.",
        "This graph illustrates exponential growth.",
    ):
        assert gemini_segment._clip_requires_visual_context(visual)
    assert gemini_segment._clip_requires_visual_context(
        "Let this point be x equals one. The distance between this point and "
        "that point is the secant interval."
    )


def test_internal_visual_aside_does_not_erase_later_verbal_teaching() -> None:
    segments = [
        _cue(
            "definition",
            0.0,
            8.0,
            "The derivative is the limit of average rates of change and measures "
            "instantaneous change.",
        ),
        _cue("visual-aside", 8.0, 11.0, "Look at this graph."),
        _cue(
            "verbal-return",
            11.0,
            20.0,
            "The difference quotient subtracts the two function values and divides "
            "by the input change. Taking its limit gives the derivative.",
        ),
    ]
    proposal = _proposal(
        candidate_id="internal-visual-aside",
        start_line=0,
        end_line=2,
        start_quote="The derivative is the limit",
        end_quote="limit gives the derivative",
        evidence="derivative is the limit of average rates of change",
        objective="Explain the derivative through the difference quotient",
    )

    report = _report(segments, proposal, topic="calculus derivatives")

    assert report.rejected_reasons == []
    assert report.clips[0]["cue_ids"] == [
        "definition",
        "visual-aside",
        "verbal-return",
    ]


def test_runaway_caption_scan_detects_visual_signal_split_across_cues() -> None:
    blocks = [
        " ".join(["background"] * 65),
        "as you can",
        "see the half R ligature shape must be recognized visually",
        " ".join(["explanation"] * 65),
    ]
    assert gemini_segment._clip_requires_visual_context(
        " ".join(blocks),
        learning_objective="Recognize the half R ligature shape",
        speech_blocks=blocks,
    )


def test_terminal_future_preview_is_trimmed_after_a_complete_explanation() -> None:
    segments = [
        _cue(
            "notation",
            0.0,
            10.0,
            "Leibniz notation writes the tangent slope as dy over dx, which is "
            "change in y over change in x.",
        ),
        _cue("future", 10.0, 13.0, "As you'll see in future videos,"),
        _cue(
            "preview",
            13.0,
            18.0,
            "one way to think about tangent slope is to calculate secant lines.",
        ),
    ]
    proposal = _proposal(
        candidate_id="terminal-future-preview",
        start_line=0,
        end_line=2,
        start_quote="Leibniz notation writes the tangent slope",
        end_quote="calculate secant lines",
        evidence="tangent slope as dy over dx which is change in y",
        objective="Explain Leibniz derivative notation",
    )

    report = _report(segments, proposal, topic="calculus derivatives")

    assert report.rejected_reasons == []
    assert report.clips[0]["cue_ids"] == ["notation"]
    assert "future videos" not in report.clips[0]["_clip_text"].casefold()


def test_terminal_mastery_recap_is_trimmed_after_the_answer() -> None:
    segments = [
        _cue(
            "setup",
            0.0,
            8.0,
            "Multiplying by the common denominator and canceling gives negative "
            "one over three x.",
        ),
        _cue(
            "answer-a",
            8.0,
            12.0,
            "Substituting three makes the final answer negative one divided",
        ),
        _cue(
            "answer-b-and-recap",
            12.0,
            16.0,
            "by nine so now you know how to evaluate",
        ),
        _cue(
            "recap",
            16.0,
            19.0,
            "limits that are associated with complex fractions",
        ),
    ]
    proposal = _proposal(
        candidate_id="terminal-mastery-recap",
        start_line=0,
        end_line=3,
        start_quote="Multiplying by the common denominator",
        end_quote="associated with complex fractions",
        evidence="common denominator and canceling gives negative one over three x",
        objective="Evaluate a complex-fraction limit",
    )

    report = _report(segments, proposal)

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["cue_ids"][-1] == "answer-b-and-recap"
    assert clip["_clip_text"].endswith("by nine")
    assert "now you know" not in clip["_clip_text"].casefold()


def test_punctuationless_internal_meta_does_not_erase_later_teaching() -> None:
    cases = [
        (
            [
                _cue("evidence", 0.0, 6.0, "Isolate x by undoing operations in reverse order"),
                _cue("aside", 6.0, 10.0, "so now you know how to start solving the equation"),
                _cue("step", 10.0, 14.0, "next divide both sides by three"),
                _cue("answer", 14.0, 18.0, "therefore the final solution is x equals four"),
            ],
            "Isolate x by undoing operations in reverse order",
            "the final solution is x equals four",
            "Isolate x by undoing operations in reverse order",
            "Solve a linear equation by inverse operations",
        ),
        (
            [
                _cue("evidence", 0.0, 7.0, "The chain rule differentiates an outer and inner function"),
                _cue("aside", 7.0, 10.0, "as you'll see in future videos this rule appears often"),
                _cue("application", 10.0, 15.0, "apply the rule to f of g of x by differentiating both functions"),
                _cue("answer", 15.0, 19.0, "therefore multiply f prime of g of x by g prime of x"),
            ],
            "The chain rule differentiates",
            "by g prime of x",
            "chain rule differentiates an outer and inner function",
            "Apply the chain rule to a composite function",
        ),
    ]
    for index, (segments, start, end, evidence, objective) in enumerate(cases):
        proposal = _proposal(
            candidate_id=f"punctuationless-internal-meta-{index}",
            start_line=0,
            end_line=3,
            start_quote=start,
            end_quote=end,
            evidence=evidence,
            objective=objective,
        )
        report = _report(segments, proposal, topic=objective)
        assert report.rejected_reasons == []
        assert report.clips[0]["cue_ids"][-1] == "answer"


def test_multisentence_terminal_meta_is_still_trimmed() -> None:
    segments = [
        _cue(
            "teaching",
            0.0,
            10.0,
            "Leibniz notation writes the derivative as dy over dx and connects it "
            "to change in y over change in x.",
        ),
        _cue(
            "future-filler",
            10.0,
            15.0,
            "As you'll see in future videos, we will prove why this works.",
        ),
        _cue(
            "more-filler",
            15.0,
            19.0,
            "Be sure to watch those lessons for the full story.",
        ),
    ]
    proposal = _proposal(
        candidate_id="multisentence-terminal-meta",
        start_line=0,
        end_line=2,
        start_quote="Leibniz notation writes the derivative",
        end_quote="lessons for the full story",
        evidence="derivative as dy over dx and connects it to change",
        objective="Explain Leibniz derivative notation",
    )
    report = _report(segments, proposal, topic="calculus derivatives")
    assert report.rejected_reasons == []
    assert report.clips[0]["cue_ids"] == ["teaching"]


def test_described_next_examples_are_hard_one_topic_boundaries() -> None:
    for transition in (
        "Here is another worked problem",
        "Let's consider one more simple case",
        "Here is another short proof",
        "Let's consider another derivation",
        "Here is another calculation",
        "Let's do another example",
        "Let's take another problem",
        "Let's go through another example",
        "Let's work out another problem",
        "Now consider another example",
        "Now for another example",
        "Our next example is",
    ):
        segments = [
            _cue(
                "old-unit",
                0.0,
                8.0,
                f"After factoring, the limit is four. {transition}",
            ),
            _cue(
                "new-unit",
                8.0,
                14.0,
                "The derivative is the instantaneous rate of change.",
            ),
        ]
        proposal = _proposal(
            candidate_id=f"described-reset-{transition[:4]}",
            start_line=0,
            end_line=1,
            start_quote="After factoring the limit is four",
            end_quote="instantaneous rate of change",
            evidence="derivative is the instantaneous rate of change",
            objective="Define a derivative as instantaneous rate of change",
        )

        report = _report(
            segments,
            proposal,
            topic="calculus derivatives",
        )

        assert report.rejected_reasons == []
        assert report.clips[0]["cue_ids"] == ["new-unit"]


def test_grounded_worked_problem_isolated_from_prior_examples_and_next_problem() -> None:
    segments = [
        _cue(
            "rule",
            0.0,
            12.0,
            "The chain rule multiplies the outer derivative by the inner derivative.",
        ),
        _cue(
            "example-one",
            12.0,
            30.0,
            "Find the derivative of five x plus three to the fourth power. "
            "The answer is twenty times five x plus three cubed.",
        ),
        _cue(
            "example-two",
            30.0,
            49.0,
            "Find the derivative of x squared minus three x to the fifth power. "
            "Apply the outer and inner derivatives to finish the answer.",
        ),
        _cue(
            "target",
            49.0,
            67.0,
            "Find the derivative of sine of six x. Differentiate sine first and "
            "then multiply by the inner derivative six.",
        ),
        _cue(
            "target-answer-next",
            67.0,
            83.0,
            "The final answer is six cosine of six x. Now find the derivative of "
            "cosine of four x.",
        ),
    ]
    proposal = _proposal(
        candidate_id="sine-six-x",
        start_line=0,
        end_line=4,
        start_quote="The chain rule multiplies",
        end_quote="cosine of four x",
        evidence="Find the derivative of sine of six x",
        objective="Differentiate sine of six x with the chain rule",
    )

    report = _report(segments, proposal, topic="chain rule worked example")

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["cue_ids"] == ["target", "target-answer-next"]
    assert clip["_clip_text"].startswith("Find the derivative of sine of six x")
    assert clip["_clip_text"].rstrip(" .").endswith("six cosine of six x")
    assert "five x plus three" not in clip["_clip_text"]
    assert "cosine of four x" not in clip["_clip_text"]


def test_grounded_problem_trims_completed_prior_problem_and_next_topic_in_coarse_cues() -> None:
    segments = [
        _cue(
            "prior-problem",
            0.0,
            16.0,
            "What is the derivative of one over x cubed minus seven. Rewrite the "
            "denominator with a negative exponent.",
        ),
        _cue(
            "prior-answer-target",
            16.0,
            34.0,
            "The final answer is negative three x squared over x cubed minus seven "
            "squared. Find the derivative of one over x squared plus eight raised "
            "to the third power.",
        ),
        _cue(
            "target-reasoning",
            34.0,
            52.0,
            "Rewrite the expression with exponent negative three, use the chain "
            "rule, and multiply by the inner derivative two x.",
        ),
        _cue(
            "target-answer-next",
            52.0,
            70.0,
            "The simplified answer is negative six x over x squared plus eight to "
            "the fourth power. Now what if we have a trigonometric function? Find "
            "the derivative of sine of x squared.",
        ),
    ]
    proposal = _proposal(
        candidate_id="rational-target",
        start_line=0,
        end_line=3,
        start_quote="What is the derivative of one over x cubed",
        end_quote="derivative of sine of x squared",
        evidence=(
            "Find the derivative of one over x squared plus eight raised to the "
            "third power"
        ),
        objective=(
            "Differentiate one over x squared plus eight cubed by rewriting a "
            "negative exponent"
        ),
    )

    report = _report(segments, proposal, topic="chain rule worked example")

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["cue_ids"] == [
        "prior-answer-target",
        "target-reasoning",
        "target-answer-next",
    ]
    assert clip["_clip_text"].startswith("Find the derivative of one over x squared")
    assert clip["_clip_text"].rstrip(" .").endswith(
        "x squared plus eight to the fourth power"
    )
    assert "x cubed minus seven" not in clip["_clip_text"]
    assert "trigonometric function" not in clip["_clip_text"]


def test_first_worked_problem_keeps_required_rule_context() -> None:
    segments = [
        _cue(
            "rule",
            0.0,
            10.0,
            "The chain rule differentiates the outside and multiplies by the "
            "derivative of the inside.",
        ),
        _cue(
            "problem",
            10.0,
            20.0,
            "Find the derivative of sine of six x.",
        ),
        _cue(
            "answer",
            20.0,
            30.0,
            "The outside derivative is cosine and the inner derivative is six, so "
            "the answer is six cosine of six x.",
        ),
    ]
    proposal = _proposal(
        candidate_id="first-problem-with-rule",
        start_line=0,
        end_line=2,
        start_quote="The chain rule differentiates the outside",
        end_quote="six cosine of six x",
        evidence="Find the derivative of sine of six x",
        objective="Use the chain rule to differentiate sine of six x",
    )

    report = _report(segments, proposal, topic="chain rule worked example")

    assert report.rejected_reasons == []
    assert report.clips[0]["cue_ids"] == ["rule", "problem", "answer"]


def test_reasoning_words_inside_one_solution_do_not_create_new_units() -> None:
    segments = [
        _cue(
            "problem",
            0.0,
            8.0,
            "Find the derivative of sine of x squared.",
        ),
        _cue(
            "reasoning",
            8.0,
            18.0,
            "To find the inner derivative, calculate the derivative of x squared, "
            "which is two x.",
        ),
        _cue(
            "answer",
            18.0,
            27.0,
            "Then multiply to get the final answer two x cosine of x squared.",
        ),
    ]
    proposal = _proposal(
        candidate_id="single-solution",
        start_line=0,
        end_line=2,
        start_quote="Find the derivative of sine",
        end_quote="two x cosine of x squared",
        evidence="To find the inner derivative calculate the derivative of x squared",
        objective="Differentiate sine of x squared with the chain rule",
    )

    report = _report(segments, proposal, topic="chain rule worked example")

    assert report.rejected_reasons == []
    assert report.clips[0]["cue_ids"] == ["problem", "reasoning", "answer"]


def test_explicit_comparison_may_keep_two_worked_units() -> None:
    segments = [
        _cue(
            "power-problem",
            0.0,
            9.0,
            "Find the derivative of x squared using the power rule.",
        ),
        _cue(
            "power-answer",
            9.0,
            17.0,
            "The power rule gives two x.",
        ),
        _cue(
            "chain-problem",
            17.0,
            27.0,
            "Now find the derivative of sine of x squared using the chain rule.",
        ),
        _cue(
            "comparison",
            27.0,
            40.0,
            "Unlike the first result, this answer multiplies cosine of x squared by "
            "the inner derivative two x.",
        ),
        _cue(
            "unrelated-third-problem",
            40.0,
            50.0,
            "Now find the derivative of tangent of x cubed.",
        ),
    ]
    proposal = _proposal(
        candidate_id="compare-rules",
        start_line=0,
        end_line=4,
        start_quote="Find the derivative of x squared",
        end_quote="inner derivative two x",
        evidence="Unlike the first result this answer multiplies cosine",
        objective="Compare the power rule example with the chain rule example",
    )

    report = _report(segments, proposal, topic="power rule versus chain rule")

    assert report.rejected_reasons == []
    assert report.clips[0]["cue_ids"] == [
        "power-problem",
        "power-answer",
        "chain-problem",
        "comparison",
    ]


def test_compact_comparison_uses_relationship_evidence_and_stops_at_third_unit() -> None:
    segments = [
        _cue(
            "power-problem",
            0.0,
            8.0,
            "Find the derivative of x squared using the power rule.",
        ),
        _cue("power-answer", 8.0, 14.0, "The power rule gives two x."),
        _cue(
            "chain-problem",
            14.0,
            24.0,
            "Find the derivative of sine of x squared using the chain rule.",
        ),
        _cue(
            "comparison",
            24.0,
            36.0,
            "Unlike the power rule result the chain rule answer multiplies cosine of "
            "x squared by the inner derivative two x.",
        ),
        _cue(
            "third-problem",
            36.0,
            46.0,
            "Find the derivative of tangent of x cubed.",
        ),
    ]
    relationship_evidence = (
        "Unlike the power rule result the chain rule answer multiplies"
    )
    plan = gemini_segment._CompactBoundaryPlan(
        request_intent={
            "exact_request": "power rule versus chain rule",
            "constraints": [
                {
                    "constraint_id": "power",
                    "kind": "subject",
                    "source_phrase": "power rule",
                    "requirement": "Teach the power rule side",
                },
                {
                    "constraint_id": "versus",
                    "kind": "relationship",
                    "source_phrase": "versus",
                    "requirement": "Compare the two rules",
                },
                {
                    "constraint_id": "chain",
                    "kind": "subject",
                    "source_phrase": "chain rule",
                    "requirement": "Teach the chain rule side",
                },
            ],
        },
        topics=[
            gemini_segment._CompactBoundaryTopic(
                candidate_id="compare-rules",
                start_line=0,
                end_line=4,
                start_quote="Find the derivative of x squared",
                end_quote="derivative of tangent of x cubed",
                title="Power rule versus chain rule",
                learning_objective=(
                    "Compare the power rule example with the chain rule example"
                ),
                facet="power rule and chain rule comparison",
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
                        "constraint_id": "power",
                        "evidence_quote": (
                            "derivative of x squared using the power rule"
                        ),
                    },
                    {
                        "constraint_id": "versus",
                        "evidence_quote": relationship_evidence,
                    },
                    {
                        "constraint_id": "chain",
                        "evidence_quote": relationship_evidence,
                    },
                ],
            )
        ],
    )

    report = gemini_segment._plan_to_report(
        plan,
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic="power rule versus chain rule",
    )

    assert report.rejected_reasons == []
    assert report.clips[0]["cue_ids"] == [
        "power-problem",
        "power-answer",
        "chain-problem",
        "comparison",
    ]


def test_one_source_preserves_several_distinct_worked_units_as_separate_clips() -> None:
    segments = [
        _cue(
            "square",
            0.0,
            10.0,
            "Find the derivative of x squared. The final answer is two x.",
        ),
        _cue(
            "cube",
            10.0,
            20.0,
            "Find the derivative of x cubed. The final answer is three x squared.",
        ),
        _cue(
            "fourth",
            20.0,
            30.0,
            "Find the derivative of x to the fourth. The final answer is four x cubed.",
        ),
    ]
    proposals = [
        _proposal(
            candidate_id=candidate_id,
            start_line=0,
            end_line=2,
            start_quote="Find the derivative of x squared",
            end_quote="four x cubed",
            evidence=evidence,
            objective=objective,
        )
        for candidate_id, evidence, objective in (
            ("square", "Find the derivative of x squared", "Differentiate x squared"),
            ("cube", "Find the derivative of x cubed", "Differentiate x cubed"),
            (
                "fourth",
                "Find the derivative of x to the fourth",
                "Differentiate x to the fourth power",
            ),
        )
    ]

    report = gemini_segment._plan_to_report(
        gemini_segment._BoundaryPlan(topics=proposals),
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic="power rule worked examples",
    )

    assert report.rejected_reasons == []
    assert [clip["cue_ids"] for clip in report.clips] == [
        ["square"],
        ["cube"],
        ["fourth"],
    ]


def test_evidence_anchored_problem_trims_naturally_completed_prior_examples() -> None:
    segments = [
        _cue(
            "a",
            0.0,
            10.0,
            "Find the derivative of x squared. Using the power rule gives two x.",
        ),
        _cue(
            "b",
            10.0,
            20.0,
            "Find the derivative of x cubed. Bring down the exponent to get three x squared.",
        ),
        _cue(
            "c",
            20.0,
            30.0,
            "Find the derivative of sine of six x. Differentiate sine and multiply "
            "by six to get six cosine of six x.",
        ),
        _cue(
            "d",
            30.0,
            40.0,
            "Find the derivative of cosine of four x. Differentiate cosine and "
            "multiply by four.",
        ),
    ]
    proposal = _proposal(
        candidate_id="natural-worked-completions",
        start_line=0,
        end_line=3,
        start_quote="Find the derivative of x squared",
        end_quote="multiply by four",
        evidence="Find the derivative of sine of six x",
        objective="Differentiate sine of six x with the chain rule",
    )

    report = _report(segments, proposal, topic="chain rule")

    assert report.rejected_reasons == []
    assert report.clips[0]["cue_ids"] == ["c"]


def test_discourse_idiom_is_not_mistaken_for_a_new_question() -> None:
    segments = [
        _cue(
            "photosynthesis",
            0.0,
            12.0,
            "Photosynthesis converts light energy into chemical energy. The result "
            "is glucose. What is more, it releases oxygen that supports aerobic life.",
        )
    ]
    proposal = _proposal(
        candidate_id="photosynthesis-discourse-continuation",
        start_line=0,
        end_line=0,
        start_quote="Photosynthesis converts light energy into chemical energy",
        end_quote="oxygen that supports aerobic life",
        evidence="it releases oxygen that supports aerobic life",
        objective="Explain how photosynthesis releases oxygen that supports aerobic life",
    )

    report = _report(segments, proposal, topic="photosynthesis")

    assert report.rejected_reasons == []
    assert report.clips[0]["_clip_text"].startswith("Photosynthesis converts")


def test_anaphoric_show_continuation_keeps_the_reasoning_after_a_result() -> None:
    segments = [
        _cue(
            "power-rule-reasoning",
            0.0,
            10.0,
            "The result is two x. Show this by applying the power rule: bring down "
            "the exponent and subtract one.",
        )
    ]
    proposal = _proposal(
        candidate_id="power-rule-reasoning",
        start_line=0,
        end_line=0,
        start_quote="The result is two x",
        end_quote="bring down the exponent and subtract one",
        evidence="The result is two x",
        objective="Explain why the result is two x using the power rule",
    )

    report = _report(segments, proposal, topic="power rule")

    assert report.rejected_reasons == []
    assert "applying the power rule" in report.clips[0]["_clip_text"]
    assert report.clips[0]["_clip_text"].endswith("subtract one.")


def test_wh_question_units_are_isolated_across_non_calculus_topics() -> None:
    segments = [
        _cue(
            "reign-of-terror",
            0.0,
            10.0,
            "Who led the Reign of Terror? Robespierre led its most radical phase.",
        ),
        _cue(
            "social-contract",
            10.0,
            20.0,
            "Who wrote The Social Contract? Jean-Jacques Rousseau wrote it and "
            "argued for popular sovereignty.",
        ),
        _cue(
            "napoleon",
            20.0,
            30.0,
            "Who became emperor in 1804? Napoleon became emperor after the revolution.",
        ),
    ]
    proposal = _proposal(
        candidate_id="social-contract-author",
        start_line=0,
        end_line=2,
        start_quote="Who led the Reign of Terror",
        end_quote="Napoleon became emperor after the revolution",
        evidence="Jean-Jacques Rousseau wrote it and argued for popular sovereignty",
        objective="Identify who wrote The Social Contract and the political idea it taught",
    )

    report = _report(segments, proposal, topic="The Social Contract author")

    assert report.rejected_reasons == []
    assert report.clips[0]["cue_ids"] == ["social-contract"]


def test_wh_explanatory_continuation_is_not_split_from_its_result() -> None:
    segments = [
        _cue(
            "why-result",
            0.0,
            10.0,
            "The result is two x. This shows how the power rule reduces the exponent.",
        )
    ]
    proposal = _proposal(
        candidate_id="why-result",
        start_line=0,
        end_line=0,
        start_quote="The result is two x",
        end_quote="power rule reduces the exponent",
        evidence="power rule reduces the exponent",
        objective="Explain how the power rule produces two x",
    )

    report = _report(segments, proposal, topic="power rule")

    assert report.rejected_reasons == []
    assert report.clips[0]["_clip_text"].startswith("The result is two x")


def test_interrogative_prerequisite_is_kept_for_an_anaphoric_solution() -> None:
    segments = [
        _cue(
            "rule",
            0.0,
            12.0,
            "Why does the chain rule multiply by the inner derivative? Because a "
            "composite function changes at both the outer and inner rates.",
        ),
        _cue(
            "problem",
            12.0,
            18.0,
            "Find the derivative of sine of six x.",
        ),
        _cue(
            "solution",
            18.0,
            25.0,
            "Applying that rule gives six cosine of six x.",
        ),
    ]
    proposal = _proposal(
        candidate_id="anaphoric-chain-rule-solution",
        start_line=0,
        end_line=2,
        start_quote="Why does the chain rule multiply by the inner derivative",
        end_quote="gives six cosine of six x",
        evidence="Find the derivative of sine of six x",
        objective="Use the chain rule explanation to differentiate sine of six x",
    )

    report = _report(segments, proposal, topic="chain rule")

    assert report.rejected_reasons == []
    assert report.clips[0]["cue_ids"] == ["rule", "problem", "solution"]


def test_relative_wh_continuations_keep_their_antecedents() -> None:
    cases = (
        (
            "atp",
            "ATP hydrolysis releases usable energy. The result is ADP. Which powers "
            "cellular work by coupling exergonic and endergonic reactions.",
            "powers cellular work by coupling exergonic and endergonic reactions",
            "Explain how ATP hydrolysis powers cellular work",
            "ATP hydrolysis",
        ),
        (
            "catalyst",
            "A catalyst lowers the activation energy. The result is an easier reaction "
            "pathway. Which increases the reaction rate without changing equilibrium.",
            "increases the reaction rate without changing equilibrium",
            "Explain how a catalyst lowers activation energy, which increases reaction rate",
            "A catalyst",
        ),
        (
            "supply",
            "The upward-sloping graph compares higher prices with greater quantity "
            "supplied. The result is a positive relationship. What this shows is that "
            "producers supply more when price rises.",
            "producers supply more when price rises",
            "Explain the positive price and quantity-supplied relationship",
            "The upward-sloping graph",
        ),
    )
    for candidate_id, text, evidence, objective, expected_start in cases:
        segments = [_cue(candidate_id, 0.0, 12.0, text)]
        proposal = _proposal(
            candidate_id=candidate_id,
            start_line=0,
            end_line=0,
            start_quote=expected_start,
            end_quote=evidence,
            evidence=evidence,
            objective=objective,
        )

        report = _report(segments, proposal, topic=objective)

        assert report.rejected_reasons == []
        assert report.clips[0]["_clip_text"].startswith(expected_start)


def test_cross_cue_evidence_with_a_repeated_tail_still_anchors_target_unit() -> None:
    segments = [
        _cue(
            "prior",
            0.0,
            10.0,
            "Find the derivative of x squared. The result is two x.",
        ),
        _cue(
            "target-head",
            10.0,
            18.0,
            "That completes the prior example. Find the derivative of sine of",
        ),
        _cue(
            "target-tail",
            18.0,
            28.0,
            "six x. Differentiate sine and multiply by six to get six cosine of six x.",
        ),
        _cue(
            "next",
            28.0,
            36.0,
            "Find the derivative of cosine of four x. Multiply by four.",
        ),
    ]
    proposal = _proposal(
        candidate_id="repeated-cross-cue-tail",
        start_line=0,
        end_line=3,
        start_quote="Find the derivative of x squared",
        end_quote="Multiply by four",
        evidence="Find the derivative of sine of six x",
        objective="Differentiate sine of six x with the chain rule",
    )

    report = _report(segments, proposal, topic="chain rule")

    assert report.rejected_reasons == []
    assert report.clips[0]["cue_ids"] == ["target-head", "target-tail"]
    assert report.clips[0]["_clip_text"].startswith("Find the derivative of sine")


def test_unpunctuated_which_noun_question_is_a_distinct_learning_unit() -> None:
    segments = [
        _cue(
            "terror",
            0.0,
            10.0,
            "Who led the Reign of Terror Robespierre led its radical phase",
        ),
        _cue(
            "financial-crisis",
            10.0,
            22.0,
            "Which factor most directly caused the financial crisis The regressive "
            "tax system and war debt drove the monarchy toward bankruptcy",
        ),
        _cue(
            "napoleon",
            22.0,
            30.0,
            "Who became emperor in 1804 Napoleon became emperor",
        ),
    ]
    proposal = _proposal(
        candidate_id="financial-crisis-cause",
        start_line=0,
        end_line=2,
        start_quote="Who led the Reign of Terror",
        end_quote="Napoleon became emperor",
        evidence="The regressive tax system and war debt drove the monarchy toward bankruptcy",
        objective="Identify the regressive tax system and war debt as causes of the crisis",
    )

    report = _report(segments, proposal, topic="French financial crisis")

    assert report.rejected_reasons == []
    assert report.clips[0]["cue_ids"] == ["financial-crisis"]


def test_locally_defined_result_resolves_later_demonstrative_reference() -> None:
    segments = [
        _cue(
            "square",
            0.0,
            8.0,
            "Find the area of a square with side three. The answer is nine.",
        ),
        _cue(
            "circle-question",
            8.0,
            14.0,
            "Find the area of a circle with radius two.",
        ),
        _cue(
            "circle-answer",
            14.0,
            23.0,
            "The result is four pi. Using this result, we know the circle covers four "
            "pi square units.",
        ),
        _cue(
            "cube",
            23.0,
            31.0,
            "Find the volume of a cube with side three. The answer is twenty seven.",
        ),
    ]
    proposal = _proposal(
        candidate_id="circle-area",
        start_line=0,
        end_line=3,
        start_quote="Find the area of a square with side three",
        end_quote="The answer is twenty seven",
        evidence="Find the area of a circle with radius two",
        objective="Calculate the area of a circle with radius two",
    )

    report = _report(segments, proposal, topic="circle area")

    assert report.rejected_reasons == []
    assert report.clips[0]["cue_ids"] == ["circle-question", "circle-answer"]


def test_unpunctuated_which_plural_noun_question_is_isolated() -> None:
    segments = [
        _cue("prior", 0.0, 8.0, "Who led the radical phase Robespierre led it"),
        _cue(
            "economic-causes",
            8.0,
            20.0,
            "Which causes of the French Revolution were economic Fiscal inequality "
            "war debt and food prices destabilized the monarchy",
        ),
        _cue("next", 20.0, 28.0, "Who became emperor Napoleon became emperor"),
    ]
    proposal = _proposal(
        candidate_id="economic-causes",
        start_line=0,
        end_line=2,
        start_quote="Who led the radical phase",
        end_quote="Napoleon became emperor",
        evidence="Fiscal inequality war debt and food prices destabilized the monarchy",
        objective="Explain the economic causes that destabilized the French monarchy",
    )

    report = _report(segments, proposal, topic="French Revolution economic causes")

    assert report.rejected_reasons == []
    assert report.clips[0]["cue_ids"] == ["economic-causes"]


def test_procedural_numerator_step_stays_with_its_problem_and_answer() -> None:
    segments = [
        _cue(
            "limit-question",
            0.0,
            10.0,
            "Find the limit of x squared minus four over x minus two as x approaches two.",
        ),
        _cue(
            "factor-step",
            10.0,
            18.0,
            "Now calculate the numerator by factoring it as x minus two times x plus two.",
        ),
        _cue(
            "limit-answer",
            18.0,
            25.0,
            "Cancel x minus two and substitute two. The final answer is four.",
        ),
        _cue(
            "next-limit",
            25.0,
            34.0,
            "Find the limit of x cubed minus eight over x minus two.",
        ),
    ]
    proposal = _proposal(
        candidate_id="factored-limit",
        start_line=0,
        end_line=3,
        start_quote="Find the limit of x squared minus four",
        end_quote="x cubed minus eight over x minus two",
        evidence="Find the limit of x squared minus four over x minus two",
        objective="Evaluate the stated limit by factoring and cancellation",
    )

    report = _report(segments, proposal, topic="factoring limits")

    assert report.rejected_reasons == []
    assert report.clips[0]["cue_ids"] == [
        "limit-question",
        "factor-step",
        "limit-answer",
    ]


def test_live_style_fraction_intuition_recovers_setup_and_formal_conclusion() -> None:
    segments = [
        _cue(
            "live:0",
            0.0,
            7.0,
            "The quotient rule differentiates a ratio, and that completes the previous lesson.",
        ),
        _cue(
            "live:1",
            7.0,
            15.0,
            "For h of x equals sine of x squared, the chain rule differentiates a composite function.",
        ),
        _cue("live:2", 15.0, 19.0, "so the derivative of h"),
        _cue(
            "live:3",
            19.0,
            25.0,
            "with respect to x and this is where the notation can build intuition",
        ),
        _cue(
            "live:4",
            25.0,
            33.0,
            "A quick aside: the notation resembles a fraction even though it is not literally one.",
        ),
        _cue(
            "live:5",
            33.0,
            40.0,
            "The intermediate differential appears to cancel, which encodes composition.",
        ),
        _cue(
            "live:6",
            40.0,
            46.0,
            "This cancellation can help build intuition and then",
        ),
        _cue(
            "live:7",
            46.0,
            54.0,
            "the formal rule multiplies the derivative of sine by the derivative of x squared.",
        ),
        _cue(
            "live:8",
            54.0,
            61.0,
            "Now let us move on to the product rule.",
        ),
    ]
    proposal = _proposal(
        candidate_id="fraction-intuition",
        start_line=3,
        end_line=6,
        start_quote="with respect to x and this",
        end_quote="help build intuition and then",
        evidence="intermediate differential appears to cancel which encodes composition",
        objective="Explain why fraction-like differential cancellation conveys chain-rule composition",
    )

    report = _report(segments, proposal, topic="chain rule worked example")

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["cue_ids"] == [f"live:{index}" for index in range(1, 8)]
    assert clip["_clip_text"].startswith("For h of x equals sine")
    assert clip["_clip_text"].endswith("derivative of x squared.")
    assert "A quick aside" in clip["_clip_text"]
    assert "quotient rule" not in clip["_clip_text"].casefold()
    assert "product rule" not in clip["_clip_text"].casefold()


def test_dependent_preposition_and_terminal_then_are_not_complete_edges() -> None:
    assert not gemini_segment._opening_clause_is_standalone(
        "with respect to x and this is where the notation helps"
    )
    assert gemini_segment._opening_clause_is_standalone(
        "With respect to x, the derivative of x squared is two x."
    )
    assert gemini_segment._opening_clause_is_standalone(
        "By applying the chain rule, we multiply the outer and inner derivatives."
    )
    assert gemini_segment._terminal_content_is_explicitly_incomplete(
        "This cancellation can help build intuition and then"
    )
    assert not gemini_segment._terminal_content_is_explicitly_incomplete(
        "Differentiate the outside and then multiply by the inner derivative."
    )


def test_independent_of_onset_does_not_complete_the_previous_caption_cue() -> None:
    for current, following in (
        (
            "The derivative of x squared is two x",
            "Of course integration reverses differentiation and is our next topic",
        ),
        (
            "The derivative of x squared is two x, which is one complete example",
            "Of the available methods substitution belongs to the next topic",
        ),
    ):
        segments = [
            _cue("derivative", 0.0, 8.0, current),
            _cue("integration", 8.0, 16.0, following),
        ]
        proposal = _proposal(
            candidate_id="x-squared-derivative",
            start_line=0,
            end_line=0,
            start_quote="The derivative of x squared",
            end_quote=(
                "complete example" if current.endswith("complete example") else "is two x"
            ),
            evidence="derivative of x squared",
            objective="Differentiate x squared",
        )

        report = _report(segments, proposal, topic="derivative of x squared")

        assert report.rejected_reasons == []
        assert report.clips[0]["cue_ids"] == ["derivative"]


def test_independent_preposition_and_auxiliary_question_keep_clean_openings() -> None:
    assert gemini_segment._opening_clause_is_standalone(
        "With respect to x and y, the function is symmetric."
    )
    coordinated = (
        "With respect to x and this parameter, the function is symmetric and stable."
    )
    assert gemini_segment._opening_clause_is_standalone(coordinated)
    coordinated_report = _report(
        [
            _cue("prior", 0.0, 5.0, "The previous lesson covered limits."),
            _cue(
                "symmetric",
                5.0,
                12.0,
                coordinated + " Its mixed partials agree.",
            ),
        ],
        _proposal(
            candidate_id="symmetric-function",
            start_line=1,
            end_line=1,
            start_quote="With respect to x and this parameter",
            end_quote="mixed partials agree",
            evidence="function is symmetric and stable Its mixed partials agree",
            objective="Explain symmetry with respect to two parameters",
        ),
        topic="symmetric functions",
    )
    assert coordinated_report.rejected_reasons == []
    assert coordinated_report.clips[0]["cue_ids"] == ["symmetric"]

    question_report = _report(
        [
            _cue("prior", 0.0, 5.0, "the previous lesson covered elements"),
            _cue("question", 5.0, 10.0, "Is oxygen a metal"),
            _cue(
                "answer",
                10.0,
                16.0,
                "Oxygen is classified as a nonmetal under standard conditions.",
            ),
        ],
        _proposal(
            candidate_id="oxygen-classification",
            start_line=1,
            end_line=2,
            start_quote="Is oxygen a metal",
            end_quote="standard conditions",
            evidence="oxygen is classified as a nonmetal under standard conditions",
            objective="Classify oxygen as a nonmetal",
        ),
        topic="oxygen classification",
    )
    assert question_report.rejected_reasons == []
    assert question_report.clips[0]["cue_ids"] == ["question", "answer"]


def test_visual_pointer_is_not_trimmed_after_a_required_gerund_complement() -> None:
    text = (
        "Chain rule intuition comes from the final derivative by comparing both "
        "this right over here with that right over there."
    )
    assert gemini_segment._trailing_edge_noise_start(text) is None
    report = _report(
        [_cue("visual-comparison", 0.0, 10.0, text)],
        _proposal(
            candidate_id="visual-comparison",
            start_line=0,
            end_line=0,
            start_quote="Chain rule intuition",
            end_quote="that right over there",
            evidence="Chain rule intuition comes from the final derivative",
            objective="Explain chain rule intuition from the final derivative",
        ),
        topic="chain rule intuition",
    )
    assert report.clips == []
    assert report.rejected_reasons == ["proposal_0:requires_visual_context"]


def test_now_and_then_idiom_is_a_complete_educational_ending() -> None:
    text = "Intermittent symptoms can appear now and then."
    assert not gemini_segment._terminal_content_is_explicitly_incomplete(text)
    report = _report(
        [_cue("intermittent", 0.0, 10.0, text)],
        _proposal(
            candidate_id="intermittent-symptoms",
            start_line=0,
            end_line=0,
            start_quote="Intermittent symptoms",
            end_quote="now and then",
            evidence="Intermittent symptoms can appear now and then",
            objective="Explain that intermittent symptoms recur occasionally",
        ),
        topic="intermittent symptoms",
    )
    assert report.rejected_reasons == []
    assert report.clips[0]["cue_ids"] == ["intermittent"]


def test_exact_live_rolling_chain_rule_clip_recovers_the_complete_thought() -> None:
    raw_cues = [
        ("36", 76.4, 79.2, "now, what, i, want, to, do, is, a little, bit, of"),
        ("37", 77.6, 81.52, "a thought experiment"),
        ("38", 79.2, 83.6, "a little bit of a thought experiment if"),
        ("39", 81.52, 85.28, "i were to ask you what is the derivative"),
        ("40", 83.6, 87.759, "with respect to x if i were to supply"),
        ("41", 85.28, 89.759, "the derivative operator to"),
        ("42", 87.759, 92.159, "x squared with respect to x what do i"),
        ("43", 89.759, 94.56, "get well this gives me 2x we've seen"),
        ("44", 92.159, 95.92, "that many many many many times now what"),
        ("45", 94.56, 99.92, "if i were to take the derivative with respect to a"),
        ("46", 97.439, 102.64, "of a squared well it's the exact same"),
        ("47", 99.92, 105.6, "thing i just swapped an a for the x's"),
        ("48", 102.64, 107.2, "this is still going to be equal to 2a"),
        ("49", 105.6, 109.119, "now i will do something that might be a"),
        ("50", 107.2, 112.24, "little bit more bizarre what if i were"),
        ("51", 109.119, 115.68, "to take the derivative with respect to"),
        ("52", 112.24, 119.04, "sine of x with respect to sine of x of"),
        ("53", 116.88, 123.92, "of sine of x sine of x squared"),
        ("54", 121.6, 125.52, "well wherever i had the x's up here or"),
        ("55", 123.92, 128.08, "the a's over here i just replace it with"),
        ("56", 125.52, 130.72, "a sine of x so this is just going to be"),
        ("57", 128.08, 131.92, "2 times the thing that i had so whatever"),
        ("58", 130.72, 133.84, "i'm taking the derivative with respect"),
        ("59", 131.92, 135.84, "to here with respect to x here with"),
        ("60", 133.84, 138.16, "respect to a here's with respect to sine"),
        ("61", 135.84, 144.08, "of x so it's going to be 2 times sine of x"),
        ("62", 141.68, 147.12, "now so the chain rule tells us that this"),
        ("63", 144.08, 149.84, "derivative is going to be the derivative"),
        ("64", 147.12, 151.76, "of our whole function with respect"),
        ("65", 149.84, 156.56, "or the derivative of this outer function x squared"),
        ("66", 153.36, 159.44, "the derivative of x squared"),
        ("67", 156.56, 161.84, "the derivative of this outer function"),
        ("68", 159.44, 164.959, "with respect to sine of x"),
        ("69", 161.84, 167.84, "so that's going to be 2 sine of x 2"),
        ("70", 166.08, 169.92, "sine of x so we could view it as the"),
        ("71", 167.84, 172.319, "derivative of the outer function with"),
        ("72", 169.92, 173.84, "respect to the inner 2 sine of x we"),
        ("73", 172.319, 175.36, "could just treat sine of x like it's"),
        ("74", 173.84, 177.04, "kind of an x and it would have been just"),
        ("75", 175.36, 178.56, "2x but instead it's a sine of x so we"),
        ("76", 177.04, 182.48, "say 2 sine of x times"),
        ("77", 179.92, 185.04, "times the derivative we do this in green"),
        ("78", 182.48, 186.64, "times the derivative of sine of x with"),
        ("79", 185.04, 189.28, "respect to x"),
        ("80", 186.64, 190.8, "times the derivative of sine of x with"),
        ("81", 189.28, 192.08, "respect to x well that's more"),
        ("82", 190.8, 194.48, "straightforward a little bit more"),
        ("83", 192.08, 196.48, "intuitive the derivative of sine of x"),
        ("84", 194.48, 201.519, "with respect to x we've seen multiple times"),
        ("85", 197.44, 203.36, "is cosine of x so times cosine of x and"),
        ("86", 201.519, 204.959, "so there we've applied the chain rule it"),
        ("87", 203.36, 207.519, "was the derivative of the outer function"),
        ("88", 204.959, 209.04, "with respect to the inner so derivative"),
        ("89", 207.519, 211.84, "of sine of x squared with respect to"),
        ("90", 209.04, 213.44, "sine of x is 2 sine of x and then we"),
        ("91", 211.84, 216.959, "multiply that times the derivative of sine of x"),
        ("92", 214.64, 218.72, "with respect to x"),
        ("93", 216.959, 221.84, "so let me make it clear this right over here"),
        ("94", 219.599, 223.36, "is the derivative we're taking the"),
        ("95", 221.84, 226.959, "derivative of we're taking the derivative of sine"),
        ("96", 225.519, 230.879, "of x squared so let me make it clear that's"),
        ("97", 229.04, 233.2, "what we're taking the derivative of with"),
        ("98", 230.879, 236.0, "respect to sine of x"),
        ("99", 233.2, 238.64, "with respect to sine of x and then we're"),
        ("100", 236.0, 242.48, "multiplying that times the derivative of"),
        ("101", 238.64, 245.12, "sine of x the derivative of sine"),
        ("102", 242.48, 247.76, "of x with respect to"),
        ("103", 245.12, 249.599, "with respect to x and this is where it"),
        ("104", 247.76, 252.48, "start might start making a little bit of intuition"),
        ("105", 251.04, 254.72, "you can't really treat these"),
        ("106", 252.48, 258.32, "differentials this d whatever this dx"),
        ("107", 254.72, 259.84, "this d sine of x as as a as a as a"),
        ("108", 258.32, 261.44, "number and you really can't this"),
        ("109", 259.84, 262.88, "notation makes it look like a fraction"),
        ("110", 261.44, 264.32, "because intuitively that's what we're"),
        ("111", 262.88, 266.479, "doing but if you were to treat them like"),
        ("112", 264.32, 268.24, "fractions then you could think about"),
        ("113", 266.479, 269.68, "canceling that and that and once again"),
        ("114", 268.24, 271.199, "this isn't a rigorous thing to do but it"),
        ("115", 269.68, 273.36, "can help with the intuition and then"),
        ("116", 271.199, 275.759, "what you're left with is the derivative"),
        ("117", 273.36, 278.639, "of this whole sine of x squared with"),
        ("118", 275.759, 280.88, "respect to x so you're left with"),
        ("119", 278.639, 282.96, "you're left with the derivative"),
        ("120", 280.88, 285.44, "of essentially our original function sine of x"),
        ("121", 284.32, 290.639, "squared with respect to x with respect to x"),
        ("122", 288.24, 294.08, "which is exactly what dhdx is this right over here"),
        ("123", 291.759, 298.479, "this right over here is our original function"),
        ("124", 295.36, 300.56, "h that's our original function h so it"),
        ("125", 298.479, 302.08, "might seem a little bit daunting now"),
    ]
    segments = _youtube_cues(raw_cues)
    proposal = _proposal(
        candidate_id="live-fraction-intuition",
        start_line=67,
        end_line=79,
        start_quote="with respect to x and this",
        end_quote="help with the intuition and then",
        evidence="if you were to treat them like fractions then you could think about canceling",
        objective="Use fraction-like cancellation of differentials to conceptualize the chain rule logic",
    )

    report = _report(segments, proposal, topic="chain rule worked example")

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["cue_ids"][0] == "YNstP0ESndU:cue:38"
    assert clip["cue_ids"][-1] == "YNstP0ESndU:cue:122"
    assert clip["_clip_text"].startswith("if i were to ask you")
    assert clip["_clip_text"].endswith("which is exactly what dhdx is")
    assert "what, i, want, to, do" not in clip["_clip_text"]
    assert "thought experiment" not in clip["_clip_text"]
    assert not clip["_clip_text"].startswith("with respect to x")
    assert not clip["_clip_text"].endswith("and then")
    assert "next video" not in clip["_clip_text"].casefold()

    worked_proposal = _proposal(
        candidate_id="live-sine-squared-chain-rule",
        start_line=0,
        end_line=len(segments) - 1,
        start_quote="now what i want to do",
        end_quote="might seem a little bit daunting now",
        evidence="so there we've applied the chain rule it was the derivative",
        objective="Apply the chain rule to differentiate sine squared of x",
    )
    worked_report = _report(
        segments,
        worked_proposal,
        topic="chain rule worked example",
    )

    assert worked_report.rejected_reasons == []
    [worked_clip] = worked_report.clips
    assert worked_clip["cue_ids"][0] == "YNstP0ESndU:cue:38"
    assert worked_clip["cue_ids"][-1] == "YNstP0ESndU:cue:103"
    worked_text = worked_clip["_clip_text"].casefold()
    assert worked_text.startswith("if i were to ask you what is the derivative")
    assert "x squared with respect to x" in worked_text
    assert "a squared well it's the exact same" in worked_text
    assert "so there we've applied the chain rule" in worked_text
    assert "thought experiment" not in worked_text
    assert "differentials" not in worked_text
    assert "fraction" not in worked_text

    line_by_cue_id = {
        str(cue_id): line
        for line, (cue_id, _start, _end, _text) in enumerate(raw_cues)
    }
    split_restatement_proposal = _proposal(
        candidate_id="live-chain-rule-split-restatement",
        start_line=line_by_cue_id["38"],
        end_line=line_by_cue_id["86"],
        start_quote="if i were to ask you what is the derivative",
        end_quote="so there we've applied the chain rule it",
        evidence="so there we've applied the chain rule",
        objective="Apply the chain rule to differentiate sine squared of x",
    )
    split_restatement_report = _report(
        segments,
        split_restatement_proposal,
        topic="chain rule worked example",
    )

    assert split_restatement_report.rejected_reasons == []
    [split_restatement_clip] = split_restatement_report.clips
    assert split_restatement_clip["cue_ids"][-1] == "YNstP0ESndU:cue:92"
    assert split_restatement_clip["_clip_text"].endswith("with respect to x")
    assert "so let me make it clear" not in split_restatement_clip["_clip_text"]


def test_pronoun_tail_continuation_is_question_and_agreement_aware() -> None:
    assert gemini_segment._cue_has_explicit_dangling_end(
        "so there we've applied the chain rule it",
        "was the derivative of the outer function",
    )
    assert gemini_segment._cue_has_explicit_dangling_end(
        "the explanation says I",
        "am applying the chain rule",
    )
    assert not gemini_segment._cue_has_explicit_dangling_end(
        "The chain rule derivation verifies it.",
        "Was a second example needed?",
    )
    assert not gemini_segment._cue_has_explicit_dangling_end(
        "The chain rule derivation verifies it",
        "Was a second example needed?",
    )
    assert not gemini_segment._cue_has_explicit_dangling_end(
        "The caption ends with we",
        "is a different lesson",
    )


def test_live_coarse_captions_isolate_sine_six_x_worked_unit() -> None:
    segments = [
        _cue(
            "HaHsqDjWMLU:cue:0",
            0.919,
            34.879,
            "let's move on to the chain rule we're going to cover a lot of examples "
            "the first Formula you need to be familiar with is the derivative of the "
            "composite function f of g ofx a composite function is one where you have "
            "one function inside of another notice that g is inside of f which makes "
            "it a composite function so the first thing you need to do is differentiate "
            "the outside portion of the function that is f and you need to keep the "
            "inside the same and then multiply it by the derivative",
        ),
        _cue(
            "HaHsqDjWMLU:cue:1",
            32.8,
            69.159,
            "of the inside that's the main idea behind the chain rule if you follow "
            "this process you're going to get the answer right so let's say for example "
            "if we have a function U raised to the N where U is another function in "
            "terms of X using the chain Rule and the power rule combined it's going to "
            "be n * U you have to keep that the same raised to the N minus one times "
            "the derivative of what's on the inside that's the general power rule "
            "formula with the chain rule combine so never forget to",
        ),
        _cue(
            "HaHsqDjWMLU:cue:2",
            64.92,
            77.119,
            "multiply by the derivative of the inside function so let's use an "
            "example let's say if we want to find the",
        ),
        _cue(
            "HaHsqDjWMLU:cue:3",
            78.28,
            124.6,
            "derivative of 5x + 3 raised to the 4th power so the first thing we're "
            "going to do is we're going to move the constant I mean the exponent to "
            "the front so it's going to be four and then keep the inside stuff the "
            "same * 5x + 3 subtract the exponent by 1 4 - 1 is 3 and then multiply by "
            "the derivative of the inside the inside function is four it's 5x + 3 the "
            "derivative of 5x + 3 is just 5 and so that's the answer we can multiply "
            "four and 5 that's going to give us 20 so it's 20 * 5x + 3 ra the thir",
        ),
        _cue(
            "HaHsqDjWMLU:cue:4",
            120.92,
            165.92,
            "power so that's the final answer fully simplified now let's work on some "
            "more examples find the derivative of x^2 - 3x raised to the 5th power so "
            "first let's bring down to five so it's going to be five and then keep the "
            "inside function the same and then subtract the exponent by 1 so this is "
            "four and then multiply by the derivative of the inside the derivative of "
            "x^2 - 3x is 2x - 3 and so that's the answer once you get used to the "
            "process it's not that bad here's another example that you can",
        ),
        _cue(
            "HaHsqDjWMLU:cue:5",
            159.959,
            185.44,
            "try find the derivative of s of 6X the derivative of the outside part of "
            "the function s is cosine and you got to keep the inside function the same "
            "then you multiply by the derivative of the inside function the derivative "
            "of 6X is 6 so the answer is simply 6 cosine",
        ),
        _cue(
            "HaHsqDjWMLU:cue:6",
            189.519,
            227.56,
            "6X now what is the derivative of cosine x^2 so first differentiate the "
            "outside part of the function cosine the derivative of cosine is negative "
            "sign now the inside part of the function has to remain the same that is "
            "the angle of cosine so it's going to be x^2 and then differentiate the "
            "inside function x^2 which is 2x so basically you're working away from the "
            "outside towards the inside the final answer is -2X sin",
        ),
    ]
    proposal = _proposal(
        candidate_id="ex3",
        start_line=0,
        end_line=5,
        start_quote="let's move on to the chain",
        end_quote="6 cosine",
        evidence="try find the derivative of s of 6X the derivative of",
        objective=(
            "Show how to differentiate a sine function with a linear inner "
            "function using the chain rule."
        ),
    )

    report = _report(segments, proposal, topic="chain rule worked example")

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["_clip_text"].casefold().startswith(
        "find the derivative of s of 6x"
    )
    assert clip["_clip_text"].rstrip(" .!?").endswith("6 cosine 6X")
    assert "5x + 3" not in clip["_clip_text"]
    assert "x^2 - 3x" not in clip["_clip_text"]
    assert "derivative of cosine x^2" not in clip["_clip_text"]
    assert clip["start"] >= 159.959
    assert clip["edge_projection"]["end"] == {
        "required": True,
        "cue_id": "HaHsqDjWMLU:cue:6",
        "quote": "6X",
    }

    compact_plan = gemini_segment._CompactBoundaryPlan(
        request_intent={
            "exact_request": "chain rule worked example",
            "constraints": [
                {
                    "constraint_id": "topic_01",
                    "kind": "subject",
                    "source_phrase": "chain rule",
                    "requirement": "Teach the chain rule",
                },
                {
                    "constraint_id": "task_01",
                    "kind": "format",
                    "source_phrase": "worked example",
                    "requirement": "Work through an example",
                },
            ],
        },
        topics=[gemini_segment._CompactBoundaryTopic(
            candidate_id="chain_rule_trig_example",
            start_line=0,
            end_line=5,
            start_quote="let's move on to the chain",
            end_quote="6 cosine",
            title="Chain Rule Worked Example for Trigonometric Functions",
            learning_objective=(
                "Show how to differentiate a sine function with a linear inner "
                "function using the chain rule."
            ),
            facet="Worked Example",
            informativeness=0.9,
            topic_relevance=1.0,
            educational_importance=0.9,
            difficulty=0.45,
            directly_teaches_topic=True,
            substantive=True,
            factually_grounded=True,
            self_contained=True,
            is_standalone=True,
            intent_evidence=[
                {
                    "constraint_id": "topic_01",
                    "evidence_quote": (
                        "derivative of s of 6X the derivative of the outside"
                    ),
                },
                {
                    "constraint_id": "task_01",
                    "evidence_quote": (
                        "try find the derivative of s of 6X the derivative of"
                    ),
                },
            ],
        )],
    )
    compact_report = gemini_segment._plan_to_report(
        compact_plan,
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic="chain rule worked example",
    )

    assert compact_report.rejected_reasons == []
    [compact_clip] = compact_report.clips
    assert compact_clip["_clip_text"].casefold().startswith(
        "find the derivative of s of 6x"
    )
    assert compact_clip["topic_evidence_quote"].casefold().startswith(
        "find the derivative of s of 6x"
    )
    assert compact_clip["intent_evidence"][1]["evidence_quote"].casefold().startswith(
        "find the derivative of s of 6x"
    )


def test_sine_candidate_starting_at_target_cue_recovers_split_example_prompt() -> None:
    segments = [
        _cue(
            "HaHsqDjWMLU:cue:4",
            120.92,
            165.92,
            "power so that's the final answer fully simplified now let's work on some "
            "more examples find the derivative of x^2 - 3x raised to the 5th power so "
            "first let's bring down to five so it's going to be five and then keep the "
            "inside function the same and then subtract the exponent by 1 so this is "
            "four and then multiply by the derivative of the inside the derivative of "
            "x^2 - 3x is 2x - 3 and so that's the answer once you get used to the "
            "process it's not that bad here's another example that you can",
        ),
        _cue(
            "HaHsqDjWMLU:cue:5",
            159.959,
            185.44,
            "try find the derivative of s of 6X the derivative of the outside part of "
            "the function s is cosine and you got to keep the inside function the same "
            "then you multiply by the derivative of the inside function the derivative "
            "of 6X is 6 so the answer is simply 6 cosine",
        ),
        _cue("HaHsqDjWMLU:cue:6", 189.519, 189.995, "6X"),
    ]
    proposal = _proposal(
        candidate_id="production-sine-six-x",
        start_line=1,
        end_line=2,
        start_quote="try find the derivative of s of 6X",
        end_quote="6X",
        evidence="find the derivative of s of 6X the derivative of the outside part",
        objective="Apply the chain rule to find the derivative of sin(6x).",
    )

    report = _report(segments, proposal, topic="chain rule worked example")

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["_clip_text"].casefold().startswith(
        "find the derivative of s of 6x"
    )
    assert clip["_clip_text"].endswith("6X")
    assert "x^2 - 3x" not in clip["_clip_text"]
    assert clip["cue_ids"] == [
        "HaHsqDjWMLU:cue:5",
        "HaHsqDjWMLU:cue:6",
    ]


def test_completed_prior_unit_allows_universal_one_cue_prompt_recovery() -> None:
    prior_endings = (
        "The final answer is four. As another example you can",
        "The final answer is four. For our next problem you can",
        "The final answer is four. Example three asks you to",
        "The final answer is four. Let's do one more and",
        "That completes the second example. As another example you can",
    )
    for prior_ending in prior_endings:
        segments = [
            _cue("prior-example", 0.0, 20.0, prior_ending),
            _cue(
                "target-example",
                19.8,
                35.0,
                "try find the derivative of sine of six x. The outside derivative "
                "is cosine, and the inner derivative is six, so the final answer is "
                "six cosine of six x.",
            ),
        ]
        proposal = _proposal(
            candidate_id="universal-split-prompt",
            start_line=1,
            end_line=1,
            start_quote="try find the derivative of sine of six x",
            end_quote="six cosine of six x",
            evidence="find the derivative of sine of six x",
            objective="Apply the chain rule to differentiate sine of six x.",
        )

        report = _report(segments, proposal, topic="chain rule worked example")

        assert report.rejected_reasons == [], prior_ending
        [clip] = report.clips
        assert clip["_clip_text"].casefold().startswith(
            "find the derivative of sine of six x"
        ), prior_ending
        assert "final answer is four" not in clip["_clip_text"].casefold(), prior_ending
        assert clip["cue_ids"] == ["target-example"], prior_ending


def test_live_coarse_captions_isolate_first_power_example_and_complete_answer() -> None:
    segments = [
        _cue(
            "HaHsqDjWMLU:cue:0",
            0.919,
            34.879,
            "let's move on to the chain rule we're going to cover a lot of examples "
            "the first Formula you need to be familiar with is the derivative of the "
            "composite function f of g ofx a composite function is one where you have "
            "one function inside of another notice that g is inside of f which makes "
            "it a composite function so the first thing you need to do is differentiate "
            "the outside portion of the function that is f and you need to keep the "
            "inside the same and then multiply it by the derivative",
        ),
        _cue(
            "HaHsqDjWMLU:cue:1",
            32.8,
            69.159,
            "of the inside that's the main idea behind the chain rule if you follow "
            "this process you're going to get the answer right so let's say for example "
            "if we have a function U raised to the N where U is another function in "
            "terms of X using the chain Rule and the power rule combined it's going to "
            "be n * U you have to keep that the same raised to the N minus one times "
            "the derivative of what's on the inside that's the general power rule "
            "formula with the chain rule combine so never forget to",
        ),
        _cue(
            "HaHsqDjWMLU:cue:2",
            64.92,
            77.119,
            "multiply by the derivative of the inside function so let's use an "
            "example let's say if we want to find the",
        ),
        _cue(
            "HaHsqDjWMLU:cue:3",
            78.28,
            124.6,
            "derivative of 5x + 3 raised to the 4th power so the first thing we're "
            "going to do is we're going to move the constant I mean the exponent to "
            "the front so it's going to be four and then keep the inside stuff the "
            "same * 5x + 3 subtract the exponent by 1 4 - 1 is 3 and then multiply by "
            "the derivative of the inside the inside function is four it's 5x + 3 the "
            "derivative of 5x + 3 is just 5 and so that's the answer we can multiply "
            "four and 5 that's going to give us 20 so it's 20 * 5x + 3 ra the thir",
        ),
        _cue(
            "HaHsqDjWMLU:cue:4",
            120.92,
            165.92,
            "power so that's the final answer fully simplified now let's work on some "
            "more examples find the derivative of x^2 - 3x raised to the 5th power so "
            "first let's bring down to five so it's going to be five and then keep the "
            "inside function the same and then subtract the exponent by 1 so this is "
            "four and then multiply by the derivative of the inside the derivative of "
            "x^2 - 3x is 2x - 3 and so that's the answer",
        ),
    ]
    proposal = _proposal(
        candidate_id="example-power-rule-chain",
        start_line=3,
        end_line=3,
        start_quote="derivative of 5x + 3 raised to the 4th power",
        end_quote="20 * 5x + 3 ra the thir",
        evidence="derivative of 5x + 3 raised to the 4th power",
        objective="Differentiate a polynomial function raised to a power using the chain rule.",
    )

    report = _report(segments, proposal, topic="chain rule worked example")

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["_clip_text"].casefold().startswith(
        "find the derivative of 5x + 3 raised to the 4th power"
    )
    assert clip["_clip_text"].casefold().endswith(
        "power so that's the final answer fully simplified"
    )
    assert "going to cover" not in clip["_clip_text"].casefold()
    assert "x^2 - 3x" not in clip["_clip_text"]
    assert clip["edge_projection"]["start"]["cue_id"] == "HaHsqDjWMLU:cue:2"
    assert clip["edge_projection"]["end"]["cue_id"] == "HaHsqDjWMLU:cue:4"

    compact_plan = gemini_segment._CompactBoundaryPlan(
        request_intent={
            "exact_request": "chain rule worked example",
            "constraints": [
                {
                    "constraint_id": "c1",
                    "kind": "subject",
                    "source_phrase": "chain rule",
                    "requirement": "Teach the chain rule",
                },
                {
                    "constraint_id": "c2",
                    "kind": "format",
                    "source_phrase": "worked example",
                    "requirement": "Work through an example",
                },
            ],
        },
        topics=[gemini_segment._CompactBoundaryTopic(
            candidate_id="poly_chain_rule",
            start_line=3,
            end_line=3,
            start_quote="derivative of 5x + 3 raised to the 4th power",
            end_quote="20 * 5x + 3 ra the thir",
            title="Derivative of a polynomial to a power",
            learning_objective=(
                "Differentiate a polynomial function raised to a power using the chain rule."
            ),
            facet="Polynomial examples",
            informativeness=0.9,
            topic_relevance=1.0,
            educational_importance=0.9,
            difficulty=0.3,
            directly_teaches_topic=True,
            substantive=True,
            factually_grounded=True,
            self_contained=True,
            is_standalone=True,
            intent_evidence=[
                {
                    "constraint_id": "c1",
                    "evidence_quote": "derivative of 5x + 3 raised to the 4th power",
                },
                {
                    "constraint_id": "c2",
                    "evidence_quote": "derivative of 5x + 3 raised to the 4th power",
                },
            ],
        )],
    )
    compact_report = gemini_segment._plan_to_report(
        compact_plan,
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic="chain rule worked example",
    )

    assert compact_report.rejected_reasons == []
    [compact_clip] = compact_report.clips
    assert compact_clip["_clip_text"].casefold().startswith(
        "find the derivative of 5x + 3 raised to the 4th power"
    )
    assert "going to cover" not in compact_clip["_clip_text"].casefold()
    assert "x^2 - 3x" not in compact_clip["_clip_text"]


def test_grounded_want_to_find_prompt_does_not_retain_the_prior_rule() -> None:
    segments = [
        _cue(
            "rule",
            0.0,
            12.0,
            "The general chain rule multiplies the outer derivative by the inner derivative.",
        ),
        _cue(
            "prompt-head",
            12.0,
            18.0,
            "For example, suppose we want to find the",
        ),
        _cue(
            "worked-answer",
            18.0,
            30.0,
            "derivative of three x plus one squared. Bring down the exponent, keep "
            "the inside, and multiply by three, so the answer is six times three x "
            "plus one.",
        ),
    ]
    proposal = _proposal(
        candidate_id="generic-want-to-find-example",
        start_line=0,
        end_line=2,
        start_quote="The general chain rule multiplies",
        end_quote="six times three x plus one",
        evidence="find the derivative of three x plus one squared",
        objective="Differentiate three x plus one squared with the chain rule",
    )

    report = _report(segments, proposal, topic="chain rule worked example")

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["_clip_text"].startswith(
        "find the derivative of three x plus one squared"
    )
    assert "general chain rule" not in clip["_clip_text"].casefold()


def test_cross_cue_future_action_separates_a_completed_prior_topic() -> None:
    segments = [
        _cue(
            "limits-then-prompt",
            0.0,
            40.0,
            "A limit describes the value a function approaches near an input. "
            "One-sided limits can differ, and if they differ the two-sided limit "
            "does not exist. That completes our limits discussion. We're now "
            "going to find",
        ),
        _cue(
            "derivative-answer",
            40.2,
            75.0,
            '"the derivative of 5x plus 3 raised to the fourth power. Bring down four, '
            "keep 5x plus 3 cubed, and multiply by five. The final answer is 20 "
            "times 5x plus 3 cubed.",
        ),
    ]
    proposal = _proposal(
        candidate_id="limits-to-derivative-transition",
        start_line=1,
        end_line=1,
        start_quote="the derivative of 5x plus 3 raised to the fourth power",
        end_quote="20 times 5x plus 3 cubed",
        evidence="the derivative of 5x plus 3 raised to the fourth power",
        objective="Differentiate a polynomial raised to a power using the chain rule.",
    )

    report = _report(segments, proposal, topic="chain rule worked example")

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["_clip_text"].casefold().startswith(
        'find "the derivative of 5x plus 3 raised to the fourth power'
    )
    assert "limit" not in clip["_clip_text"].casefold()
    assert clip["edge_projection"]["start"] == {
        "required": True,
        "cue_id": "limits-then-prompt",
        "quote": "find",
    }


def test_cross_cue_future_action_keeps_an_inner_step_with_its_problem() -> None:
    segments = [
        _cue(
            "problem-and-inner-step",
            0.0,
            30.0,
            "For y equals 5x plus 3 to the fourth, bring down four and keep the "
            "inside cubed. We are now going to find the",
        ),
        _cue(
            "inner-answer",
            30.2,
            50.0,
            '"inner derivative, which is five, then multiply to get 20 times 5x plus 3 cubed. '
            "That is the final answer.",
        ),
    ]
    proposal = _proposal(
        candidate_id="same-problem-inner-derivative",
        start_line=1,
        end_line=1,
        start_quote="inner derivative which is five",
        end_quote="That is the final answer",
        evidence="inner derivative which is five then multiply to get 20 times",
        objective="Differentiate 5x plus 3 to the fourth using the chain rule.",
    )

    report = _report(segments, proposal, topic="chain rule worked example")

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["_clip_text"].startswith("For y equals 5x plus 3 to the fourth")
    assert 'find the "inner derivative, which is five' in clip["_clip_text"]


def test_grounded_action_keeps_its_same_sentence_problem_setup() -> None:
    segments = [
        _cue(
            "prior",
            0.0,
            8.0,
            "The previous example is complete, and its final answer is four.",
        ),
        _cue(
            "equation",
            8.0,
            22.0,
            "Given the equation x plus three equals five, we need to solve for x. "
            "Subtract three from both sides, so x equals two, which is the solution.",
        ),
    ]
    proposal = _proposal(
        candidate_id="same-sentence-equation-setup",
        start_line=0,
        end_line=1,
        start_quote="Given the equation x plus three equals five",
        end_quote="x equals two, which is the solution",
        evidence="solve for x",
        objective="Solve the equation x plus three equals five for x",
    )

    report = _report(segments, proposal, topic="solve a linear equation")

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["_clip_text"].startswith(
        "Given the equation x plus three equals five"
    )
    assert not clip["_clip_text"].startswith("solve for x")
    assert "previous example" not in clip["_clip_text"].casefold()


def test_example_label_does_not_erase_the_equation_before_the_action() -> None:
    text = (
        "In this example, the equation is x plus three equals five, so solve "
        "for x. Subtract three, so x equals two. That is the final answer."
    )
    proposal = _proposal(
        candidate_id="labeled-equation-setup",
        start_line=0,
        end_line=0,
        start_quote="In this example, the equation is x plus three equals five",
        end_quote="That is the final answer",
        evidence="solve for x",
        objective="Solve x plus three equals five for x",
    )

    report = _report(
        [_cue("labeled-equation", 0.0, 18.0, text)],
        proposal,
        topic="solve a linear equation",
    )

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["_clip_text"].startswith(
        "In this example, the equation is x plus three equals five"
    )
    assert not clip["_clip_text"].startswith("solve for x")


def test_use_example_framing_does_not_erase_the_problem_setup() -> None:
    text = (
        "Let's use an example: let x plus three equal five, and solve for x. "
        "Subtract three, so x equals two. That's the final answer."
    )
    proposal = _proposal(
        candidate_id="use-example-equation-setup",
        start_line=0,
        end_line=0,
        start_quote="Let's use an example: let x plus three equal five",
        end_quote="That's the final answer",
        evidence="solve for x",
        objective="Solve x plus three equals five for x",
    )

    report = _report(
        [_cue("use-example-equation", 0.0, 18.0, text)],
        proposal,
        topic="solve a linear equation",
    )

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["_clip_text"].startswith(
        "Let's use an example: let x plus three equal five"
    )
    assert not clip["_clip_text"].startswith("solve for x")


def test_plain_more_examples_transition_does_not_leak_the_next_unit() -> None:
    text = (
        "Photosynthesis converts light into chemical energy. That is the key "
        "result. Let's do more examples carbon fixation stores energy in sugars."
    )
    proposal = _proposal(
        candidate_id="plain-more-examples-transition",
        start_line=0,
        end_line=0,
        start_quote="Photosynthesis converts light into chemical energy",
        end_quote="carbon fixation stores energy in sugars",
        evidence="Photosynthesis converts light into chemical energy",
        objective="Explain the energy conversion performed by photosynthesis",
    )

    report = _report(
        [_cue("mixed-biology-units", 0.0, 16.0, text)],
        proposal,
        topic="photosynthesis energy conversion",
    )

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["_clip_text"].endswith("That is the key result")
    assert "carbon fixation" not in clip["_clip_text"].casefold()


def test_completed_problem_resets_setup_guard_inside_one_unpunctuated_cue() -> None:
    text = (
        "given x squared find its derivative use the power rule and the final "
        "answer is two x find the derivative of x cubed use the power rule and "
        "the final answer is three x squared"
    )
    segments = [_cue("two-power-rule-problems", 0.0, 24.0, text)]
    first_proposal = _proposal(
        candidate_id="first-power-rule-problem",
        start_line=0,
        end_line=0,
        start_quote="given x squared",
        end_quote="three x squared",
        evidence="find its derivative",
        objective="Find the derivative of x squared with the power rule",
    )
    second_proposal = _proposal(
        candidate_id="second-power-rule-problem",
        start_line=0,
        end_line=0,
        start_quote="given x squared",
        end_quote="three x squared",
        evidence="find the derivative of x cubed",
        objective="Find the derivative of x cubed with the power rule",
    )

    first_report = _report(
        segments,
        first_proposal,
        topic="power rule worked examples",
    )
    second_report = _report(
        segments,
        second_proposal,
        topic="power rule worked examples",
    )

    assert first_report.rejected_reasons == []
    [first_clip] = first_report.clips
    assert first_clip["_clip_text"].startswith("given x squared")
    assert first_clip["_clip_text"].endswith("final answer is two x")
    assert "x cubed" not in first_clip["_clip_text"]

    assert second_report.rejected_reasons == []
    [second_clip] = second_report.clips
    assert second_clip["_clip_text"].startswith("find the derivative of x cubed")
    assert second_clip["_clip_text"].endswith("final answer is three x squared")
    assert "x squared find its derivative" not in second_clip["_clip_text"]


def test_live_coarse_captions_complete_formula_operand_before_next_problem() -> None:
    segments = [
        _cue(
            "HaHsqDjWMLU:cue:8",
            259.0,
            308.28,
            "Cub as you can see it's not that bad here's another problem what is the "
            "derivative of secant 4X the derivative of secant 4X is going to be "
            "secant tangent that's the derivative of secant now the inside function "
            "has to remain the same for secant and tangent next we need to "
            "differentiate 4X so it's just going to be time4 and so that's the "
            "solution what is the derivative of Ln X raised to the 7th power try that "
            "problem so this is going to be S keep the inside part the same and then",
        ),
        _cue(
            "HaHsqDjWMLU:cue:9",
            304.8,
            323.56,
            "subtract the exponent by one so 7 - 1 is 6 now we got to multiply by "
            "the derivative of the inside function the derivative of Ln X is simply "
            "1 /x so the final answer is 7 Ln X raised to 6 power /",
        ),
        _cue(
            "HaHsqDjWMLU:cue:10",
            328.08,
            376.72,
            "X What is the dtive of theun of XB - 7 take a minute and work on that "
            "example the first thing I would do is rewrite it so this is the same as X "
            "Cub - 7 raised to the 12 and so that's going to be equal to2 we got to "
            "bring the exponent to the front keep the inside function the same and then "
            "subtract the exponent by one 1 12 - 1 which is 12 - 2 2 that's a half and "
            "then we got to multiply by the derivative of the inside the derivative of "
            "x Cub - 7 is simply 3x^2 so we could bring this back to the",
        ),
    ]
    proposal = _proposal(
        candidate_id="example-ln-chain",
        start_line=0,
        end_line=1,
        start_quote="what is the derivative of Ln",
        end_quote="Ln X raised to 6 power /",
        evidence="what is the derivative of Ln X raised to the 7th power",
        objective=(
            "Differentiate a natural logarithm function raised to a power using "
            "the chain rule."
        ),
    )

    report = _report(segments, proposal, topic="chain rule worked example")

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["_clip_text"].casefold().startswith(
        "what is the derivative of ln x raised to the 7th power"
    )
    assert clip["_clip_text"].rstrip().endswith("7 Ln X raised to 6 power / X")
    assert "dtive of theun" not in clip["_clip_text"]
    assert clip["edge_projection"]["end"] == {
        "required": True,
        "cue_id": "HaHsqDjWMLU:cue:10",
        "quote": "X",
        "occurrence": "first",
    }


def test_live_coarse_captions_isolate_product_chain_problem_and_solution() -> None:
    segments = [
        _cue(
            "HaHsqDjWMLU:cue:19",
            626.44,
            664.279,
            "can combine terms you can multiply five and three to get 15 what is the "
            "derivative of cosine raised to the 7th power of s of secant x^2 so try "
            "that problem so first let's rewrite it as cosine of s of secant x^2 and "
            "let's put the seven and it's exponent position so let's use the power rule",
        ),
        _cue(
            "HaHsqDjWMLU:cue:20",
            667.12,
            718.56,
            "same and then subtract the exponent by one so this is going to be six "
            "now let's find the derivative of cosine let's work our way towards the "
            "inside the derivative of cosine is negative sign and the stuff inside of "
            "that is s secant X2 so now we got to find the derivative of s the "
            "derivative of s is cosine and the stuff inside of s is secant x^2 so now "
            "we got to find the derivative of secant so that's secant tangent so it's "
            "going to be secant x^2 tangent x^2 and the derivative of x^2 is",
        ),
        _cue(
            "HaHsqDjWMLU:cue:21",
            716.079,
            759.199,
            "2x so whenever you have multiple composite functions just work your way "
            "from the outside towards the inside and everything is multiplied by each "
            "other and then when you're done simply collect terms so we have a seven "
            "a 2X and a negative so you can move that to the front and write it as "
            "-4x if you want to here's the next problem find the derivative of x Cub "
            "* 4x + 5 raised to the 4th power so what we have here is a product rule "
            "we could say that this is f and this is",
        ),
        _cue(
            "HaHsqDjWMLU:cue:22",
            755.519,
            787.68,
            "G and for G we have to use the quotient rule I mean not the quotient rule "
            "but the chain rule so using the product rule we need to differentiate the "
            "first part F the derivative of the first part is 3x^2 and we need to keep "
            "the second part the same so we just have to rewrite G plus now we need to "
            "keep the first part the same now for the second part we need to use the "
            "chain room so let's bring the four to the front let's keep the inside "
            "stuff the same and subtract the four by",
        ),
        _cue(
            "HaHsqDjWMLU:cue:23",
            785.12,
            827.639,
            "one then multiply by the derivative of the inside function which is 4x + "
            "5 the derivative of that is four so now let's simplify so the first part "
            "we don't really need to change anything we can just leave it like this now "
            "for the second part we can multiply four and four that's 16 so this is 16 "
            "x Cub * 4x + 5 raised to the thir power you can leave your answer like "
            "this or if you want to you can take out the GCF we can take out an x s and "
            "we could take out three 4x+ 5S or",
        ),
        _cue(
            "HaHsqDjWMLU:cue:24",
            823.48,
            855.079,
            "basically 4x + 5 to the third power so this is gone we took three of these "
            "one is left over and we have a a three left so this is going to be a 3 * "
            "4x + 5 and this is gone we took out all three of these well there's an X "
            "left over and it's 16 so plus 16 x because we took out an X squ now what "
            "we can do is basically simplify what we have",
        ),
        _cue(
            "HaHsqDjWMLU:cue:25",
            856.839,
            920.36,
            "here so this is x^2 4x + 5 to the 3 power and let's distribute the three "
            "so 3 * 4X is 12x 3 * 5 is 15 + 16x now let's add 12x and 16x so the "
            "final answer is x^2 * 4x + 5 to the 3 power and then 28x + 15 so that's "
            "the solution let's try one more example let's find the derivative of "
            "2x - 3 / 4 + 5x X raised to the 4th power",
        ),
        _cue(
            "HaHsqDjWMLU:cue:26",
            917.44,
            967.56,
            "one next we need to multiply by derivative of the inside so that's when "
            "we have to use a quotient rule so f is 2x - 3 and G is 4 + 5x frime is "
            "2 G Prime is 4 and the formula for the quotient rule is g f Prime minus "
            "FG Prime over G squared so as you can see this is a long",
        ),
    ]
    proposal = _proposal(
        candidate_id="example-product-chain",
        start_line=0,
        end_line=5,
        start_quote="can combine terms you can multiply",
        end_quote="we took out an X squ",
        evidence="find the derivative of x Cub * 4x + 5 raised to the 4th power",
        objective=(
            "Combine the product rule and chain rule to find the derivative of a "
            "function involving multiplication."
        ),
    )

    report = _report(segments, proposal, topic="chain rule worked example")

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["_clip_text"].casefold().startswith(
        "find the derivative of x cub * 4x + 5 raised to the 4th power"
    )
    assert clip["_clip_text"].casefold().endswith("so that's the solution")
    assert "cosine raised to the 7th" not in clip["_clip_text"]
    assert "let's try one more example" not in clip["_clip_text"].casefold()
    assert clip["edge_projection"]["start"]["cue_id"] == "HaHsqDjWMLU:cue:21"
    assert clip["edge_projection"]["end"]["cue_id"] == "HaHsqDjWMLU:cue:25"


def test_completion_lookahead_does_not_drop_reasoning_after_an_early_result() -> None:
    segments = [
        _cue(
            "partial-result",
            0.0,
            8.0,
            "Find the derivative of sine x squared. The inner derivative is two x, "
            "and that is the result of differentiating the inside.",
        ),
        _cue(
            "complete-result",
            8.0,
            16.0,
            "Now multiply by cosine x squared, so the complete derivative equals "
            "two x cosine x squared.",
        ),
        _cue(
            "next-problem",
            16.0,
            24.0,
            "Let's try another example: find the derivative of tangent x squared.",
        ),
    ]
    proposal = _proposal(
        candidate_id="complete-sine-chain-example",
        start_line=0,
        end_line=1,
        start_quote="Find the derivative of sine x squared",
        end_quote="two x cosine x squared",
        evidence="Find the derivative of sine x squared",
        objective="Differentiate sine x squared with the chain rule",
    )

    report = _report(segments, proposal, topic="chain rule worked example")

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert "multiply by cosine x squared" in clip["_clip_text"].casefold()
    assert clip["_clip_text"].casefold().endswith("two x cosine x squared.")
    assert "tangent" not in clip["_clip_text"].casefold()


def test_completion_lookahead_preserves_a_substantive_post_answer_qualifier() -> None:
    segments = [
        _cue(
            "answer",
            0.0,
            7.0,
            "Substitution gives four, so that is the final answer.",
        ),
        _cue(
            "qualifier",
            7.0,
            14.0,
            "For some cases the denominator is zero, so direct substitution is undefined.",
        ),
        _cue(
            "next-problem",
            14.0,
            21.0,
            "Let's try another example and factor a different rational expression.",
        ),
    ]
    proposal = _proposal(
        candidate_id="answer-with-domain-qualifier",
        start_line=0,
        end_line=1,
        start_quote="Substitution gives four",
        end_quote="direct substitution is undefined",
        evidence="Substitution gives four so that is the final answer",
        objective="Explain direct substitution and its zero-denominator limitation",
    )

    report = _report(segments, proposal, topic="calculus limits")

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert "denominator is zero" in clip["_clip_text"].casefold()
    assert "different rational expression" not in clip["_clip_text"].casefold()


def test_live_compact_selector_evidence_isolates_rational_worked_unit() -> None:
    segments = [
        _cue(
            "HaHsqDjWMLU:cue:9",
            304.8,
            323.56,
            "subtract the exponent by one so 7 - 1 is 6 now we got to multiply by "
            "the derivative of the inside function the derivative of Ln X is simply "
            "1 /x so the final answer is 7 Ln X raised to 6 power /",
        ),
        _cue(
            "HaHsqDjWMLU:cue:10",
            328.08,
            376.72,
            "X What is the dtive of theun of XB - 7 take a minute and work on that "
            "example the first thing I would do is rewrite it so this is the same as X "
            "Cub - 7 raised to the 12 and so that's going to be equal to2 we got to "
            "bring the exponent to the front keep the inside function the same and then "
            "subtract the exponent by one 1 12 - 1 which is 12 - 2 2 that's a half and "
            "then we got to multiply by the derivative of the inside the derivative of "
            "x Cub - 7 is simply 3x^2 so we could bring this back to the",
        ),
        _cue(
            "HaHsqDjWMLU:cue:11",
            374.319,
            428.72,
            "bottom since it has a negative exponent so it's 3x^2 / we have a two on "
            "the bottom 2 XB - 7 and now the exponent is going to change from negative "
            "half to positive half and now we could put it back in its radical form so "
            "it's 3x^2 / 2 < TK XB - 7 and so that's the final answer for this problem "
            "find the derivative of 1 / x^2 + 8 raised to the 3 power so first let's "
            "rewrite the expression let's bring the variables to the top so this is is "
            "x^2 + 8 raed Theus 3 and now we can use the chain",
        ),
        _cue(
            "HaHsqDjWMLU:cue:12",
            426.12,
            469.0,
            "rule combined with the power rule let's move the3 to the front and let's "
            "keep the inside function let's rewrite it exactly the way we see it and "
            "then let's subtract this by 1 -3 - 1 is4 and now let's multiply by the "
            "derivative of the inside function which is 2x so now let's take this term "
            "move it back to the bottom so we have -3 * 2x which is -6x on top and on "
            "the bottom it's x^2 + 8 raised to the 4th power and so that's all we need "
            "to do for this problem so for some examples you need to",
        ),
        _cue(
            "HaHsqDjWMLU:cue:13",
            465.199,
            469.0,
            "rewrite it before you find the",
        ),
        _cue(
            "HaHsqDjWMLU:cue:14",
            471.039,
            483.199,
            "derivative now what if we have a trig function inside another trig "
            "function find the Der of this uh",
        ),
    ]
    evidence = "find the derivative of 1 / x^2 + 8 raised to the 3 power"
    plan = gemini_segment._CompactBoundaryPlan(
        request_intent={
            "exact_request": "chain rule worked example",
            "constraints": [
                {
                    "constraint_id": "c1",
                    "kind": "subject",
                    "source_phrase": "chain rule",
                    "requirement": "Teach the chain rule",
                },
                {
                    "constraint_id": "c2",
                    "kind": "format",
                    "source_phrase": "worked example",
                    "requirement": "Include a worked example",
                },
            ],
        },
        topics=[
            gemini_segment._CompactBoundaryTopic(
                candidate_id="ex6",
                start_line=1,
                end_line=5,
                start_quote="X What is the dtive of",
                end_quote="find the Der of this uh",
                title="Derivative of 1/(x^2+8)^3",
                learning_objective=(
                    "Differentiate a rational function using the chain rule by "
                    "rewriting it with a negative exponent."
                ),
                facet="Rational function chain rule example",
                informativeness=0.95,
                topic_relevance=1.0,
                educational_importance=0.95,
                difficulty=0.5,
                directly_teaches_topic=True,
                substantive=True,
                factually_grounded=True,
                self_contained=True,
                is_standalone=True,
                intent_evidence=[
                    {"constraint_id": "c1", "evidence_quote": evidence},
                    {"constraint_id": "c2", "evidence_quote": evidence},
                ],
            )
        ],
    )

    report = gemini_segment._plan_to_report(
        plan,
        segments,
        [],
        {"_segment_ignore_caption_case": True},
        topic="chain rule worked example",
    )

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["_clip_text"].startswith(evidence)
    assert "-6x on top" in clip["_clip_text"]
    assert "x^2 + 8 raised to the 4th power" in clip["_clip_text"]
    assert "XB - 7" not in clip["_clip_text"]
    assert "3x^2 / 2" not in clip["_clip_text"]
    assert "trig function inside another trig function" not in clip["_clip_text"]
    assert 374.319 <= clip["start"] < 428.72
    assert clip["edge_projection"]["start"]["cue_id"] == (
        "HaHsqDjWMLU:cue:11"
    )
    assert clip["edge_projection"]["end"] == {
        "required": True,
        "cue_id": "HaHsqDjWMLU:cue:14",
        "quote": "derivative",
    }


def test_split_answer_prefix_stops_before_following_bare_problem() -> None:
    segments = [
        _cue(
            "answer-head",
            0.0,
            10.0,
            "The final answer is negative two x sine",
        ),
        _cue(
            "answer-tail-next",
            10.0,
            24.0,
            "x squared find the derivative of tangent x cubed and then apply the "
            "chain rule.",
        ),
    ]
    proposal = _proposal(
        candidate_id="split-answer-before-next",
        start_line=0,
        end_line=1,
        start_quote="The final answer is negative",
        end_quote="apply the chain rule",
        evidence="The final answer is negative two x sine",
        objective="State the completed derivative answer negative two x sine x squared",
    )

    report = _report(segments, proposal, topic="chain rule worked example")

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["_clip_text"].rstrip(" .!?").endswith(
        "negative two x sine x squared"
    )
    assert "tangent x cubed" not in clip["_clip_text"]


def test_ambiguous_evidence_cannot_span_a_topic_reset() -> None:
    segments = [
        _cue(
            "secant",
            0.0,
            8.0,
            "For the secant line, the rate of change is four. Now let's look at "
            "another example",
        ),
        _cue(
            "tangent",
            8.0,
            14.0,
            "For the tangent line, the rate of change is called the derivative.",
        ),
    ]
    proposal = _proposal(
        candidate_id="ambiguous-reset-evidence",
        start_line=0,
        end_line=1,
        start_quote="For the secant line",
        end_quote="called the derivative",
        evidence="the rate of change is",
        objective="Define a derivative as instantaneous rate of change",
    )

    report = _report(segments, proposal, topic="calculus derivatives")

    assert report.clips == []
    assert report.rejected_reasons == [
        "proposal_0:topic_evidence_crosses_topic_reset"
    ]


def test_same_cue_described_unit_label_is_not_the_new_clip_opening() -> None:
    for transition in (
        "Now for another example,",
        "Let's do another example:",
    ):
        text = (
            "After factoring, the limit is four. "
            f"{transition} the derivative is the instantaneous rate of change."
        )
        proposal = _proposal(
            candidate_id=f"inline-described-label-{transition[:3]}",
            start_line=0,
            end_line=0,
            start_quote="After factoring the limit is four",
            end_quote="instantaneous rate of change",
            evidence="derivative is the instantaneous rate of change",
            objective="Define a derivative as instantaneous rate of change",
        )
        report = _report(
            [_cue("mixed", 0.0, 15.0, text)],
            proposal,
            topic="calculus derivatives",
        )
        assert report.rejected_reasons == []
        assert report.clips[0]["_clip_text"].startswith("the derivative is")


def test_split_question_tail_recovers_the_complete_notation_context() -> None:
    texts = [
        "So how can we denote a derivative?",
        "One way is known as Leibniz's notation,",
        "and in his notation the slope of the tangent line equals dy over dx.",
        "Now why do I like this notation?",
        "Because it really comes from this idea of a slope,",
        "which is change in y over change in x.",
        "And let's see what happens as the change",
        "in x approaches zero,",
        "and so using these d's instead of deltas,",
        "this was Leibniz's way of saying,",
        '"Hey, what happens if my changes',
        'in, say, x become close to zero?" So this idea,',
        "this is known as differential notation,",
        "and it is how we will calculate the derivative.",
    ]
    segments = [
        _cue(f"split-question:{index}", index * 3.0, (index + 1) * 3.0, text)
        for index, text in enumerate(texts)
    ]
    proposal = _proposal(
        candidate_id="split-leibniz-question",
        start_line=11,
        end_line=13,
        start_quote="in, say, x become close to",
        end_quote="calculate the derivative",
        evidence="this is known as differential notation",
        objective="Explain Leibniz derivative notation",
    )

    report = _report(segments, proposal, topic="calculus derivatives")

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["cue_ids"][0] == "split-question:6"
    assert clip["_clip_text"].startswith(
        "And let's see what happens as the change in x approaches zero"
    )
    assert not clip["_clip_text"].startswith("in, say")
    assert '"Hey, what happens if my changes in, say, x become close to zero?"' in clip[
        "_clip_text"
    ]


def test_preposition_fronted_question_is_a_valid_independent_opening() -> None:
    questions = (
        "In which direction does the gradient point?",
        "During which phase does mitosis separate sister chromatids?",
        "For what values does the series converge?",
        "To what extent is the approximation valid?",
        "In calculus, what does a derivative mean?",
    )
    for question in questions:
        assert not gemini_segment._opening_is_dependent_question_tail(question)
        assert gemini_segment._opening_clause_is_standalone(question)
        assert not gemini_segment._cue_opens_mid_thought_at(
            [
                _cue("unpunctuated-prior", 0.0, 4.0, "the prior section ended"),
                _cue("independent-question", 4.0, 8.0, question),
            ],
            1,
            ignore_caption_case=True,
        )

    segments = [
        _cue("prior", 0.0, 4.0, "the previous section explained algebra"),
        _cue("question", 4.0, 8.0, "In calculus, what does a derivative mean?"),
        _cue(
            "answer",
            8.0,
            13.0,
            "A derivative is the instantaneous rate of change of a function.",
        ),
    ]
    proposal = _proposal(
        candidate_id="independent-framed-question",
        start_line=1,
        end_line=2,
        start_quote="In calculus, what does a derivative mean",
        end_quote="instantaneous rate of change of a function",
        evidence="derivative is the instantaneous rate of change",
        objective="Define a derivative",
    )

    report = _report(segments, proposal, topic="calculus derivatives")

    assert report.rejected_reasons == []
    assert report.clips[0]["cue_ids"][0] == "question"


def test_pedagogical_meta_frame_trims_punctuated_and_asr_openings() -> None:
    assert gemini_segment._leading_pedagogical_meta_quote(
        "A thought experiment. Suppose the input doubles."
    ) == "Suppose the input doubles"
    assert gemini_segment._leading_pedagogical_meta_quote(
        "a little bit of a thought experiment if"
    ) == "if"


def test_lens_handoff_requires_completion_and_never_creates_new_side_start() -> None:
    segments = [
        _cue(
            "completed",
            0.0,
            6.0,
            "Therefore we have solved the worked example and found the result.",
        ),
        _cue(
            "handoff",
            6.0,
            11.0,
            "And this is where the intuition for differential calculus comes in.",
        ),
        _cue(
            "notation",
            11.0,
            17.0,
            "Differential notation writes the derivative as dy over dx.",
        ),
        _cue(
            "meaning",
            17.0,
            23.0,
            "The notation represents how y changes with respect to x.",
        ),
    ]
    evidence = "Differential notation writes the derivative as dy over dx"

    [old_side_transition] = gemini_segment._candidate_topic_transitions(
        segments,
        0,
        len(segments) - 1,
        evidence_quote="we have solved the worked example and found the result",
        learning_objective="Solve the worked example",
    )
    assert old_side_transition.navigation_line == 1
    assert old_side_transition.navigation_left == 0

    assert gemini_segment._candidate_topic_transitions(
        segments,
        0,
        len(segments) - 1,
        evidence_quote=evidence,
        learning_objective="Explain differential notation",
    ) == []

    incomplete_segments = [dict(segment) for segment in segments]
    incomplete_segments[0]["text"] = (
        "The worked example still needs its final multiplication step."
    )
    assert gemini_segment._candidate_topic_transitions(
        incomplete_segments,
        0,
        len(incomplete_segments) - 1,
        evidence_quote="worked example still needs its final multiplication step",
        learning_objective="Solve the worked example",
    ) == []

    proposal = _proposal(
        candidate_id="differential-notation-after-lens-handoff",
        start_line=2,
        end_line=3,
        start_quote="Differential notation writes the derivative",
        end_quote="with respect to x",
        evidence=evidence,
        objective="Explain differential notation",
    )
    report = _report(segments, proposal, topic="differential notation")

    assert report.rejected_reasons == []
    [clip] = report.clips
    assert clip["cue_ids"][0] == "notation"
    assert gemini_segment._opening_clause_is_standalone(clip["_clip_text"])
