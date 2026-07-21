from backend.intent_obligations import (
    intent_obligation,
    intent_obligation_key,
    normalize_intent_obligations,
)


def test_obligation_identity_ignores_enum_and_requirement_rewording() -> None:
    assert intent_obligation_key("  merge   sort  ") == intent_obligation_key(
        "merge sort"
    )
    subject = intent_obligation(
        kind="subject",
        source_phrase="merge sort",
        requirement="Teach the merge-sort algorithm",
        evidence_quote="Merge sort divides the input into smaller arrays",
    )
    scope = intent_obligation(
        kind="scope",
        source_phrase="merge sort",
        requirement="Explain how merge sort works",
        evidence_quote="The merge step combines the sorted arrays",
    )
    assert subject is not None and scope is not None
    assert subject["key"] == scope["key"]
    assert intent_obligation_key("C") != intent_obligation_key("c")
    assert intent_obligation_key("ℂ") != intent_obligation_key("C")
    assert intent_obligation_key("C", 8) != intent_obligation_key("C", 24)


def test_obligation_round_trip_preserves_compatibility_math_identity() -> None:
    complex_numbers = intent_obligation(
        kind="subject",
        source_phrase="ℂ",
        source_start=8,
        requirement="Explain ℂ as the complex-number field",
        evidence_quote="ℂ denotes the field of complex numbers",
    )
    plain_c = intent_obligation(
        kind="subject",
        source_phrase="C",
        source_start=8,
        requirement="Explain C as the programming language",
        evidence_quote="C is a compiled programming language",
    )

    assert complex_numbers is not None and plain_c is not None
    assert complex_numbers["key"] != plain_c["key"]
    assert complex_numbers["source_phrase"] == "ℂ"
    assert complex_numbers["requirement"].startswith("Explain ℂ")
    assert normalize_intent_obligations(
        [complex_numbers, plain_c],
        require_evidence=True,
    ) == [complex_numbers, plain_c]


def test_obligation_normalization_rejects_tampering_and_missing_evidence() -> None:
    obligation = intent_obligation(
        kind="relationship",
        source_phrase="compare time and space complexity",
        requirement="Compare both time and space complexity",
        evidence_quote="Merge sort uses linear auxiliary space",
    )
    assert obligation is not None
    assert normalize_intent_obligations(
        [obligation], require_evidence=True
    ) == [obligation]
    assert normalize_intent_obligations(
        [{**obligation, "key": "io:forged"}], require_evidence=True
    ) == []
    without_position = {
        key: value
        for key, value in obligation.items()
        if key != "source_start"
    }
    assert normalize_intent_obligations(
        [without_position], require_evidence=True
    ) == []
    without_evidence = {
        key: value
        for key, value in obligation.items()
        if key != "evidence_quote"
    }
    assert normalize_intent_obligations(
        [without_evidence], require_evidence=True
    ) == []
