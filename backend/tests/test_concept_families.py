from backend.concept_families import (
    concept_family_identity_key,
    validate_concept_family_contract,
    validate_concept_family_labels,
)
from backend.app.services.reels import ReelService


def _validate(
    family: str,
    *,
    title: str = "",
    facet: str = "",
    objective: str = "",
    evidence: str = "",
    aliases: list[str] | None = None,
) -> str | None:
    return validate_concept_family_contract(
        family,
        aliases or [],
        title=title,
        facet=facet,
        objective=objective,
        evidence=evidence,
    )


def test_canonical_identity_requires_a_domain_qualified_label() -> None:
    for label in (
        "concept",
        "equations",
        "laws",
        "principles",
        "theories",
    ):
        assert concept_family_identity_key(label) == ""
        assert validate_concept_family_labels(label, []) is not None

    for label in (
        "force units",
        "photosynthesis",
        "Newton's first law",
        "Python 3.12 typing",
    ):
        assert concept_family_identity_key(label)
        assert validate_concept_family_labels(label, []) is None


def test_canonical_identity_preserves_protected_distinctions() -> None:
    for left, right in (
        ("Newton's first law", "Newton's second law"),
        ("Apollo 11 mission", "Apollo 13 mission"),
        ("Windows 11 features", "Windows 10 features"),
        ("Highway 101 routes", "Highway 102 routes"),
        ("Formula 1 aerodynamics", "Formula 2 aerodynamics"),
        ("HLA Class II presentation", "HLA Class I presentation"),
        ("Factor V deficiency", "Factor VIII deficiency"),
        ("hepatitis infection", "hepatitis C infection"),
        ("blood type A", "blood type B"),
        ("World Wars", "World War II"),
        ("photosynthesis", "cellular respiration"),
        ("kinematics", "dynamics"),
        ("C language", "C++ language"),
        ("Swift nullable type String?", "Swift nullable type String"),
        ("factorial operation n!", "factorial operation n"),
        (
            "TypeScript non-null assertion !",
            "TypeScript non-null assertion",
        ),
        ("Co chemistry", "CO chemistry"),
        ("ℂ numbers", "C numbers"),
    ):
        assert concept_family_identity_key(left)
        assert concept_family_identity_key(right)
        assert concept_family_identity_key(left) != concept_family_identity_key(right)

    for label in (
        "Swift nullable type String?",
        "factorial operation n!",
        "TypeScript non-null assertion !",
    ):
        assert validate_concept_family_labels(label, []) is None


def test_python_version_word_is_canonical_but_version_number_is_not_erased() -> None:
    assert concept_family_identity_key("Python version 3.12 typing") == (
        concept_family_identity_key("Python 3.12 typing")
    )
    assert concept_family_identity_key("Python version 3.12 typing") != (
        concept_family_identity_key("Python 3.11 typing")
    )


def test_canonical_profile_ordinals_round_trip_without_reserving_user_labels() -> None:
    for label in (
        "Twenty-first Amendment",
        "21st Amendment",
        "101st Airborne Division",
    ):
        key = concept_family_identity_key(label)
        assert key
        assert ReelService._concept_family_ordinal_indexes({key})

    assert concept_family_identity_key("ordinal_21 variable") != (
        concept_family_identity_key("21st variable")
    )


def test_trusted_label_contract_does_not_accept_aliases() -> None:
    for family, alias in (
        ("Newton's first law", "law of inertia"),
        ("photosynthesis", "cellular respiration"),
        ("C language", "C++ language"),
    ):
        assert validate_concept_family_labels(family, [alias]) is not None
        assert _validate(
            family,
            aliases=[alias],
            title=family,
            evidence=f"This clip teaches {family}",
        ) is not None


def test_full_contract_requires_evidence_but_does_not_redecide_ai_semantics() -> None:
    assert _validate("quantum mechanics") is not None

    for family, fields in (
        (
            "quantum mechanics",
            {
                "title": "Photosynthesis",
                "facet": "chloroplast light reactions",
                "objective": "Explain how plants convert light energy",
                "evidence": "Plants convert light into stored chemical energy",
            },
        ),
        (
            "Newton's second law of motion",
            {
                "title": "F equals m a",
                "facet": "net force equation",
                "objective": "Relate net force, mass, and acceleration",
                "evidence": "Net force equals mass times acceleration",
            },
        ),
        (
            "hepatitis C infection",
            {
                "title": "HCV infection",
                "facet": "HCV transmission",
                "objective": "Explain how HCV spreads",
                "evidence": "HCV spreads through exposure to infected blood",
            },
        ),
        (
            "quantum mechanics",
            {
                "title": "Quantum mechanics",
                "evidence": "Current clip evidence is present",
            },
        ),
        (
            "quantum mechanics",
            {"evidence": "Quantum mechanics describes microscopic systems"},
        ),
    ):
        assert _validate(family, **fields) is None


def test_incidental_acronyms_counts_and_discourse_do_not_reclassify_family() -> None:
    cases = (
        (
            "force units",
            "SI unit of force",
            "force units",
            "First explain the SI unit for force",
            "Force 10 newtons acts rightward in this first example",
        ),
        (
            "photosynthesis",
            "ATP production in photosynthesis",
            "light reactions",
            "In the second part explain photosynthesis",
            "The light reactions produce 2 ATP for later reactions",
        ),
        (
            "cellular respiration",
            "NADH in cellular respiration",
            "electron carriers",
            "Explain the second stage of cellular respiration",
            "During the second stage NADH transfers 2 electrons",
        ),
    )
    for family, title, facet, objective, evidence in cases:
        assert _validate(
            family,
            title=title,
            facet=facet,
            objective=objective,
            evidence=evidence,
        ) is None


def test_grounded_broad_ai_family_is_not_semantically_narrowed_by_code() -> None:
    for family, title in (
        ("Newton's laws", "Newton's first law"),
        ("Apollo missions", "Apollo 11 lunar mission"),
        ("Python typing", "Python version 3.12 typing"),
        ("C memory management", "C++ memory management"),
    ):
        assert _validate(
            family,
            title=title,
            facet=family,
            objective=f"Explain {title}",
            evidence=f"This lesson covers {title}",
        ) is None
