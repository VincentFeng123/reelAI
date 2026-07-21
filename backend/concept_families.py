"""Shared safety contract for trusted adaptive concept-family metadata."""
from __future__ import annotations

from typing import Sequence

from .concept_ordinals import (
    NUMBERED_CONCEPT_KIND_TOKENS,
    canonicalize_concept_identifier_tokens,
    concept_ordinal_indexes,
    is_canonical_ordinal_token,
)
from .concept_tokens import semantic_tokens


CONCEPT_FAMILY_GENERIC_TOKENS = frozenset({
    "concept", "concepts", "effect", "effects", "equation", "equations",
    "formula", "formulas", "identity", "identities", "law", "laws",
    "method", "methods", "model", "models", "principle", "principles",
    "process", "processes", "relationship", "relationships", "rule", "rules",
    "system", "systems", "theorem", "theorems", "theory", "theories",
    "topic", "topics",
})
CONCEPT_FAMILY_NOISE_TOKENS = frozenset({
    "a", "an", "and", "application", "applications", "applying", "basic",
    "basics", "common", "definition", "definitions", "demonstration",
    "derivation", "example", "examples", "explained", "explaining",
    "explanation", "for", "from", "fundamental", "fundamentals",
    "identification", "identify", "identifying", "in", "intro",
    "introduction", "intuition", "intuitive", "misconception",
    "misconceptions", "of", "on", "overview", "pair", "pairs", "practice",
    "problem", "problems", "recap", "summary", "the", "to", "using", "via",
    "with", "worked",
})
def concept_family_identity_tokens(
    value: object,
) -> tuple[str, ...]:
    raw_tokens = semantic_tokens(
        value,
        casefold=False,
        preserve_terminal_suffix=True,
    )
    tokens: list[str] = []
    for index, raw_token in enumerate(raw_tokens):
        token = raw_token[:-2] if raw_token.casefold().endswith("'s") else raw_token
        # Decimal punctuation is intentionally absent from semantic tokens. Drop
        # the optional lexical marker before a numeric version run so
        # ``Python version 3.12`` and ``Python 3.12`` share one stable identity;
        # the numeric components remain in the key and keep 3.11 distinct.
        if (
            token.casefold() in {"version", "versions"}
            and index + 1 < len(raw_tokens)
            and raw_tokens[index + 1].isdigit()
        ):
            continue
        if token:
            tokens.append(token)
    return canonicalize_concept_identifier_tokens(
        tokens,
        numbered_kind_tokens=NUMBERED_CONCEPT_KIND_TOKENS,
    )


def concept_family_identity_key(value: object) -> str:
    tokens = concept_family_identity_tokens(value)
    if not tokens or len(concept_ordinal_indexes(tokens)) > 1:
        return ""
    domain_tokens = {
        token
        for token in tokens
        if (
            not is_canonical_ordinal_token(token)
            and any(character.isalpha() for character in token)
            and token.casefold() not in CONCEPT_FAMILY_GENERIC_TOKENS
            and token.casefold() not in CONCEPT_FAMILY_NOISE_TOKENS
        )
    }
    return " ".join(tokens) if domain_tokens else ""


def validate_concept_family_labels(
    family: object,
    aliases: Sequence[object],
) -> str | None:
    if aliases:
        return "aliases_not_supported"
    family_key = concept_family_identity_key(family)
    if not family_key:
        return "family_not_domain_qualified"
    return None


def validate_concept_family_contract(
    family: object,
    aliases: Sequence[object],
    *,
    title: object = "",
    facet: object = "",
    objective: object = "",
    evidence: object = "",
) -> str | None:
    label_error = validate_concept_family_labels(family, aliases)
    if label_error is not None:
        return label_error

    # Semantic family selection belongs to the independent high-thinking Pro
    # audit. The audit boundary verifies its evidence quote against transcript
    # text; this shared layer only requires that evidence to exist and never
    # re-decides synonyms, qualifiers, or broad/narrow meaning with word rules.
    if not str(evidence or "").strip():
        return "family_not_clip_grounded"
    return None
