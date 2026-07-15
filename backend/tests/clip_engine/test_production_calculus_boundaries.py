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
