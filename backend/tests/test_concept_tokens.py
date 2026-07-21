from backend.concept_tokens import (
    concept_semantic_key,
    is_structural_case_token,
    semantic_key,
    semantic_token_case_key,
    semantic_tokens,
)


def test_notation_variants_share_keys_without_collapsing_plain_concepts() -> None:
    for unicode_value, ascii_value in (
        ("C♯ generics", "C# generics"),
        ("C∗-algebra", "C* algebra"),
        ("C⋆-algebra", "C* algebra"),
        ("A∗ search", "A* search"),
        ("Cl− ion", "Cl- ion"),
    ):
        assert semantic_key(unicode_value) == semantic_key(ascii_value)

    for left, right in (
        ("C memory", "C++ memory"),
        ("C memory", "C# memory"),
        ("C algebra", "C* algebra"),
        ("A search", "A* search"),
        ("Cl ion", "Cl- ion"),
        ("e field", "e- field"),
    ):
        assert semantic_key(left) != semantic_key(right)


def test_operator_concepts_retain_identity() -> None:
    keys = {
        semantic_key("JavaScript && operator"),
        semantic_key("JavaScript || operator"),
        semantic_key("JavaScript ?? operator"),
    }
    assert len(keys) == 3
    assert semantic_key("C bitwise &") != semantic_key("C bitwise |")
    assert concept_semantic_key(
        "Swift String?",
        preserve_terminal_suffix=True,
    ) != concept_semantic_key("Swift String")
    assert semantic_tokens("x >= y") == ("x", ">=", "y")


def test_operator_identity_is_independent_of_spacing() -> None:
    for compact, spaced, expected in (
        ("x+y", "x + y", ("x", "+", "y")),
        ("x-y", "x - y", ("x", "-", "y")),
        ("x*y", "x * y", ("x", "*", "y")),
        ("x/y", "x / y", ("x", "/", "y")),
        ("x!=y", "x != y", ("x", "!=", "y")),
        ("x??y", "x ?? y", ("x", "??", "y")),
        ("x->y", "x -> y", ("x", "->", "y")),
        ("x**y", "x ** y", ("x", "**", "y")),
    ):
        assert semantic_tokens(compact) == expected
        assert semantic_tokens(spaced) == expected

    assert len({semantic_key("x-y"), semantic_key("x/y"), semantic_key("x y")}) == 3


def test_unicode_math_and_scientific_notation_retain_identity() -> None:
    assert semantic_key("x ≥ y") == semantic_key("x >= y")
    assert semantic_key("A ∩ B") != semantic_key("A ∪ B")
    assert semantic_key("−5 eigenvalue") == semantic_key("-5 eigenvalue")
    assert semantic_key("-5 eigenvalue") != semantic_key("5 eigenvalue")
    assert semantic_key("OH• radical") != semantic_key("OH radical")
    for mathematical, plain in (
        ("ℂ vector space", "C vector space"),
        ("ℝ vector space", "R vector space"),
        ("ℤ integers", "Z integers"),
    ):
        assert semantic_key(mathematical) != semantic_key(plain)


def test_structural_case_is_preserved_without_case_sensitizing_prose() -> None:
    assert semantic_key("Co oxidation") != semantic_key("CO oxidation")
    assert semantic_token_case_key("Co") == "Co"
    assert semantic_token_case_key("CO") == "CO"
    assert semantic_token_case_key("HLA") == "HLA"
    assert is_structural_case_token("Co")
    assert is_structural_case_token("CO")
    assert is_structural_case_token("HLA")
    assert not is_structural_case_token("Bayes")
    assert semantic_key("Bayes Theorem") == semantic_key("bayes theorem")
    assert semantic_key("PHOTOSYNTHESIS") == semantic_key("photosynthesis")
    assert semantic_key("CAFÉ") == semantic_key("cafe\u0301")


def test_attached_language_suffixes_do_not_consume_the_next_token() -> None:
    assert semantic_tokens("C C++ C#") == ("c", "c++", "c#")
    assert semantic_tokens("C + + memory # topic") == ("c", "memory", "topic")


def test_terminal_suffix_requires_explicit_preservation() -> None:
    assert semantic_key("Bayes Theorem?") == semantic_key("Bayes Theorem")
    assert semantic_tokens("Swift String?") == semantic_tokens("Swift String")
    assert semantic_tokens(
        "Swift String?",
        preserve_terminal_suffix=True,
    ) != semantic_tokens("Swift String")
    assert semantic_tokens(
        "Explain photosynthesis!",
        preserve_terminal_suffix=True,
    ) != semantic_tokens("Explain photosynthesis")
    assert semantic_tokens(
        "factorial n!",
        preserve_terminal_suffix=True,
    ) == ("factorial", "n!")
    assert semantic_tokens(
        "TypeScript non-null assertion !",
        preserve_terminal_suffix=True,
    )[-1] == "!"
    assert semantic_key("Explain photosynthesis!") == semantic_key(
        "Explain photosynthesis"
    )


def test_prose_punctuation_and_compound_hyphens_are_not_identity() -> None:
    for punctuated, plain in (
        ("What is photosynthesis?", "What is photosynthesis"),
        ("Explain photosynthesis!", "Explain photosynthesis"),
        ("What is Photosynthesis?", "What is Photosynthesis"),
        ("Explain Gravity!", "Explain Gravity"),
        ("Bayes Theorem?", "Bayes Theorem"),
        ("Newton Laws?", "Newton Laws"),
        ("T-cell activation", "T cell activation"),
        ("first-order kinetics", "first order kinetics"),
        ("input-output", "input output"),
        ("net-force / acceleration", "net force acceleration"),
        ("C*-algebra", "C* algebra"),
        ("input/output", "input output"),
    ):
        assert semantic_key(punctuated) == semantic_key(plain)
