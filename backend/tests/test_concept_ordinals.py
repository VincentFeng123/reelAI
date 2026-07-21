from backend.concept_ordinals import (
    canonicalize_ordinal_tokens,
    concept_identifier_indexes,
    concept_ordinal_indexes,
    numbered_ordinal_indexes,
    ordinal_indexes,
)


KINDS = {
    "chapter", "chapters", "class", "classes", "equation", "equations",
    "factor", "factors", "law", "laws", "phase", "phases", "type", "types",
    "war", "wars",
}


def test_word_numeric_and_compound_ordinals_share_one_identity() -> None:
    assert canonicalize_ordinal_tokens(["5th", "law"], numbered_kind_tokens=KINDS) == (
        "fifth",
        "law",
    )
    assert canonicalize_ordinal_tokens(["fifth", "law"], numbered_kind_tokens=KINDS) == (
        "fifth",
        "law",
    )
    assert canonicalize_ordinal_tokens(
        ["twenty", "first", "law"], numbered_kind_tokens=KINDS
    ) == ("ordinal_21", "law")
    assert canonicalize_ordinal_tokens(["21st", "law"], numbered_kind_tokens=KINDS) == (
        "ordinal_21",
        "law",
    )
    assert canonicalize_ordinal_tokens(
        ["One", "Hundred", "First", "Airborne", "Division"],
        numbered_kind_tokens=KINDS,
    ) == ("ordinal_101", "airborne", "division")
    assert canonicalize_ordinal_tokens(
        ["101st", "Airborne", "Division"],
        numbered_kind_tokens=KINDS,
    ) == ("ordinal_101", "airborne", "division")
    assert canonicalize_ordinal_tokens(
        ["one", "hundredth", "anniversary"],
        numbered_kind_tokens=KINDS,
    ) == ("ordinal_100", "anniversary")


def test_plain_numeric_numbered_lists_canonicalize_every_member() -> None:
    maxwell = canonicalize_ordinal_tokens(
        ["maxwell", "equations", "1", "and", "2"],
        numbered_kind_tokens=KINDS,
    )
    laws = canonicalize_ordinal_tokens(
        ["laws", "5", "and", "6"],
        numbered_kind_tokens=KINDS,
    )
    assert ordinal_indexes(maxwell) == {1, 2}
    assert ordinal_indexes(laws) == {5, 6}
    assert numbered_ordinal_indexes(maxwell, numbered_kind_tokens=KINDS) == {1, 2}
    assert numbered_ordinal_indexes(laws, numbered_kind_tokens=KINDS) == {5, 6}


def test_plain_numbers_without_a_numbered_concept_kind_remain_values() -> None:
    for phrase in (
        ["vitamin", "b12", "level", "5"],
        ["ph", "7"],
        ["glucose", "level", "5"],
        ["solving", "x", "=", "5"],
        ["probability", "of", "rolling", "a", "6"],
    ):
        assert canonicalize_ordinal_tokens(
            phrase,
            numbered_kind_tokens=KINDS,
        ) == tuple(phrase)


def test_rate_units_and_examples_are_not_numbered_concepts() -> None:
    for phrase in (
        ["radioactive", "decay", "law", "probability", "per", "second"],
        ["second", "order", "rate", "law"],
        ["law", "applied", "in", "the", "second", "example"],
    ):
        normalized = canonicalize_ordinal_tokens(
            phrase,
            numbered_kind_tokens=KINDS,
        )
        assert numbered_ordinal_indexes(normalized, numbered_kind_tokens=KINDS) == set()

    first_order = canonicalize_ordinal_tokens(
        ["first", "order", "rate", "law"], numbered_kind_tokens=KINDS
    )
    second_order = canonicalize_ordinal_tokens(
        ["second", "order", "rate", "law"], numbered_kind_tokens=KINDS
    )
    assert concept_ordinal_indexes(first_order) == {1}
    assert concept_ordinal_indexes(second_order) == {2}


def test_true_numbered_concept_patterns_remain_recognized() -> None:
    for phrase, expected in (
        (["second", "law"], {2}),
        (["law", "2"], {2}),
        (["law", "number", "2"], {2}),
        (["law", "number", "two"], {2}),
        (["type", "II", "diabetes"], {2}),
        (["type", "I", "diabetes"], {1}),
        (["HLA", "class", "II"], {2}),
        (["factor", "V", "deficiency"], {5}),
        (["first", "and", "second", "equations"], {1, 2}),
        (["laws", "5", "and", "6"], {5, 6}),
    ):
        normalized = canonicalize_ordinal_tokens(
            phrase,
            numbered_kind_tokens=KINDS,
        )
        assert numbered_ordinal_indexes(normalized, numbered_kind_tokens=KINDS) == expected


def test_roman_cardinal_and_numeric_notations_share_context_bound_identity() -> None:
    for left, right in (
        (["type", "II", "diabetes"], ["type", "2", "diabetes"]),
        (["type", "I", "diabetes"], ["type", "1", "diabetes"]),
        (["phase", "II"], ["phase", "two"]),
        (["chapter", "IV"], ["chapter", "4"]),
        (["chapter", "XL"], ["chapter", "40"]),
        (["world", "war", "II"], ["world", "war", "2"]),
        (["HLA", "class", "II"], ["HLA", "class", "2"]),
        (["factor", "V"], ["factor", "5"]),
    ):
        assert canonicalize_ordinal_tokens(left, numbered_kind_tokens=KINDS) == (
            canonicalize_ordinal_tokens(right, numbered_kind_tokens=KINDS)
        )

    assert canonicalize_ordinal_tokens(
        ["i", "think", "this", "law", "applies"],
        numbered_kind_tokens=KINDS,
    )[0] == "i"
    assert canonicalize_ordinal_tokens(
        ["chapter", "mix"], numbered_kind_tokens=KINDS
    ) != canonicalize_ordinal_tokens(
        ["chapter", "1009"], numbered_kind_tokens=KINDS
    )
    assert canonicalize_ordinal_tokens(
        ["chapter", "iv"], numbered_kind_tokens=KINDS
    ) != canonicalize_ordinal_tokens(
        ["chapter", "4"], numbered_kind_tokens=KINDS
    )


def test_ambiguous_roman_acronyms_and_product_letters_remain_lexical() -> None:
    for left, right in (
        (["washington", "DC"], ["washington", "600"]),
        (["CI", "pipeline"], ["101", "pipeline"]),
        (["IV", "therapy"], ["4", "therapy"]),
        (["MI", "treatment"], ["1001", "treatment"]),
        (["CIV", "game"], ["104", "game"]),
        (["model", "MIX"], ["model", "1009"]),
        (["model", "X"], ["model", "10"]),
        (["type", "C"], ["type", "100"]),
        (["version", "X"], ["version", "10"]),
    ):
        assert canonicalize_ordinal_tokens(left, numbered_kind_tokens=KINDS) != (
            canonicalize_ordinal_tokens(right, numbered_kind_tokens=KINDS)
        )


def test_explicit_ordinals_are_concept_identifiers_without_a_kind_allowlist() -> None:
    for phrase, expected in (
        (["beethoven", "fifth", "symphony"], {5}),
        (["first", "crusade"], {1}),
        (["fifth", "cranial", "nerve"], {5}),
    ):
        normalized = canonicalize_ordinal_tokens(
            phrase,
            numbered_kind_tokens=KINDS,
        )
        assert concept_ordinal_indexes(normalized) == expected


def test_counts_before_plural_kinds_remain_counts() -> None:
    for phrase in (
        ["asimov", "3", "laws", "of", "robotics"],
        ["top", "10", "principles", "of", "design"],
        ["systems", "of", "3", "equations"],
        ["solve", "1", "or", "2", "equations"],
        ["top", "1", "and", "2", "principles"],
    ):
        normalized = canonicalize_ordinal_tokens(
            phrase,
            numbered_kind_tokens={*KINDS, "principles"},
        )
        assert ordinal_indexes(normalized) == set()


def test_bare_numeric_values_are_not_inferred_as_concept_identifiers() -> None:
    for phrase in (
        ["apollo", "11", "mission"],
        ["python", "3", "12", "typing"],
        ["windows", "11"],
        ["formula", "1"],
    ):
        normalized = canonicalize_ordinal_tokens(
            phrase, numbered_kind_tokens=KINDS
        )
        assert concept_identifier_indexes(normalized) == set()


def test_compound_magnitude_ordinal_does_not_collapse_to_unit_ordinal() -> None:
    one_hundred_first = canonicalize_ordinal_tokens(
        ["one", "hundred", "first", "airborne", "division"],
        numbered_kind_tokens=KINDS,
    )
    first = canonicalize_ordinal_tokens(
        ["first", "airborne", "division"], numbered_kind_tokens=KINDS
    )
    assert concept_ordinal_indexes(one_hundred_first) == {101}
    assert concept_ordinal_indexes(first) == {1}
    assert one_hundred_first != first
