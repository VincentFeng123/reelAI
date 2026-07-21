"""Domain-independent ordinal normalization for concept-family identities."""
from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

_SIMPLE_WORD_ORDINALS = {
    "zeroth": 0,
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
    "thirteenth": 13,
    "fourteenth": 14,
    "fifteenth": 15,
    "sixteenth": 16,
    "seventeenth": 17,
    "eighteenth": 18,
    "nineteenth": 19,
    "twentieth": 20,
    "thirtieth": 30,
    "fortieth": 40,
    "fiftieth": 50,
    "sixtieth": 60,
    "seventieth": 70,
    "eightieth": 80,
    "ninetieth": 90,
}
_MAGNITUDE_WORD_ORDINALS = {
    "hundredth": 100,
}
_CARDINAL_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_CARDINAL_UNITS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_UNIT_WORD_ORDINALS = {
    word: value
    for word, value in _SIMPLE_WORD_ORDINALS.items()
    if 1 <= value <= 9
}
_NUMERIC_ORDINAL = re.compile(r"^(?P<number>\d+)(?:st|nd|rd|th)$", re.IGNORECASE)
_PLAIN_NUMBER = re.compile(r"^\d+$")
_ROMAN_NUMBER = re.compile(r"^[ivxlcdm]+$", re.IGNORECASE)
_CANONICAL_PREFIX = "ordinal_"
NUMBERED_CONCEPT_KIND_TOKENS = frozenset({
    "amendment", "amendments", "article", "articles", "axiom", "axioms",
    "chapter", "chapters", "class", "classes", "equation", "equations",
    "factor", "factors", "identity", "identities", "law", "laws", "model",
    "models", "movement", "movements", "phase", "phases", "postulate",
    "postulates", "principle", "principles", "rule", "rules", "section",
    "sections", "stage", "stages", "step", "steps", "theorem", "theorems",
    "type", "types", "version", "versions", "war", "wars",
})
PLURAL_NUMBERED_CONCEPT_KIND_TOKENS = frozenset({
    "amendments", "articles", "axioms", "chapters", "classes", "equations",
    "factors", "identities", "laws", "models", "movements", "phases",
    "postulates", "principles", "rules", "sections", "stages", "steps",
    "theorems", "types", "versions", "wars",
})
_NUMBERED_LIST_CONNECTORS = frozenset({"and", "no", "number", "or"})
_AMBIGUOUS_ROMAN_LABEL_KINDS = frozenset({
    "model", "models", "type", "types", "version", "versions",
})
_RATE_MARKERS = frozenset({"each", "every", "per"})
_ENUMERATION_FOLLOWERS = frozenset({
    "attempt", "attempts", "case", "cases", "example", "examples",
    "instance", "instances", "time", "times",
})
_NUMERIC_CONTEXT_LEADS = frozenset({
    "a", "about", "an", "at", "by", "each", "every", "for", "from", "given", "in",
    "of", "on", "over", "per", "through", "to", "top", "under", "using", "with",
})
_NUMERIC_ENUMERATION_LEADS = frozenset({
    "attempt", "attempts", "case", "cases", "example", "examples", "item", "items",
    "part", "parts", "problem", "problems", "question", "questions", "step", "steps",
})
_NUMERIC_UNIT_FOLLOWERS = frozenset({
    "amp", "amps", "ampere", "amperes", "byte", "bytes", "celsius", "cm", "day",
    "days", "degree", "degrees", "feet", "foot", "g", "gram", "grams", "hertz",
    "hour", "hours", "hz", "inch", "inches", "joule", "joules", "kelvin", "kg",
    "kilogram", "kilograms", "km", "liter", "liters", "litre", "litres", "m", "meter",
    "meters", "metre", "metres", "mile", "miles", "minute", "minutes", "ml", "mm",
    "mole", "moles", "newton", "newtons", "ohm", "ohms", "percent", "percentage",
    "second", "seconds", "volt", "volts", "watt", "watts", "year", "years",
})
_CANONICAL_WORD_BY_VALUE = {
    value: word for word, value in _SIMPLE_WORD_ORDINALS.items()
}


def _canonical_ordinal_token(value: int) -> str:
    return _CANONICAL_WORD_BY_VALUE.get(value, f"{_CANONICAL_PREFIX}{value}")


def _roman_number(value: str) -> int | None:
    raw_token = str(value or "").strip()
    if not raw_token or raw_token != raw_token.upper():
        return None
    token = raw_token.upper()
    if not _ROMAN_NUMBER.fullmatch(token):
        return None
    numerals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    previous = 0
    for character in reversed(token):
        current = numerals[character]
        total += -current if current < previous else current
        previous = max(previous, current)
    if not 0 < total < 4000:
        return None
    pairs = (
        (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
        (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
        (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
    )
    remaining = total
    canonical = []
    for amount, symbol in pairs:
        while remaining >= amount:
            canonical.append(symbol)
            remaining -= amount
    return total if "".join(canonical) == token else None


def _plain_number_at(
    tokens: Sequence[str],
    index: int,
    *,
    raw_tokens: Sequence[str] | None = None,
) -> tuple[int, int] | None:
    token = tokens[index]
    if _PLAIN_NUMBER.fullmatch(token):
        return int(token), 1
    roman = _roman_number(raw_tokens[index] if raw_tokens is not None else token)
    if roman is not None:
        return roman, 1
    cardinal = _CARDINAL_UNITS.get(token)
    if cardinal is not None:
        return cardinal, 1
    if token in _CARDINAL_TENS:
        if index + 1 < len(tokens) and tokens[index + 1] in _CARDINAL_UNITS:
            unit = _CARDINAL_UNITS[tokens[index + 1]]
            if 1 <= unit <= 9:
                return _CARDINAL_TENS[token] + unit, 2
        return _CARDINAL_TENS[token], 1
    return None


def _plain_value_list_count(
    tokens: Sequence[str],
    kind_index: int,
    *,
    raw_tokens: Sequence[str] | None = None,
) -> int:
    count = 0
    index = kind_index + 1
    limit = min(len(tokens), kind_index + 8)
    while index < limit:
        if tokens[index] in _NUMBERED_LIST_CONNECTORS:
            index += 1
            continue
        parsed = _plain_number_at(tokens, index, raw_tokens=raw_tokens)
        if parsed is None:
            break
        _value, width = parsed
        count += 1
        index += width
    return count


def _plain_value_is_numbered_label(
    tokens: Sequence[str],
    index: int,
    width: int,
    numbered_kind_tokens: frozenset[str],
    *,
    raw_tokens: Sequence[str] | None = None,
) -> bool:
    if not numbered_kind_tokens:
        return False
    for kind_index in range(index - 1, max(-1, index - 7), -1):
        kind = tokens[kind_index]
        if kind not in numbered_kind_tokens:
            continue
        raw_number = (
            raw_tokens[index]
            if raw_tokens is not None and index < len(raw_tokens)
            else tokens[index]
        )
        roman = _roman_number(raw_number)
        if (
            roman is not None
            and kind in _AMBIGUOUS_ROMAN_LABEL_KINDS
            and (
                (len(raw_number) == 1 and raw_number != "I")
                or any(
                    character in {"L", "C", "D", "M"}
                    for character in raw_number.upper()
                )
            )
        ):
            continue
        between = tokens[kind_index + 1 : index]
        if not all(
            token in _NUMBERED_LIST_CONNECTORS
            or _plain_number_at(tokens, position, raw_tokens=raw_tokens) is not None
            for position, token in enumerate(
                between,
                start=kind_index + 1,
            )
        ):
            continue
        is_plural = kind in PLURAL_NUMBERED_CONCEPT_KIND_TOKENS
        if not is_plural or _plain_value_list_count(
            tokens,
            kind_index,
            raw_tokens=raw_tokens,
        ) >= 2:
            return True
    return False


def _ordinal_under_100_at(
    tokens: Sequence[str],
    index: int,
) -> tuple[int, int] | None:
    if index >= len(tokens):
        return None
    token = tokens[index]
    if token in _CARDINAL_TENS:
        if index + 1 < len(tokens) and tokens[index + 1] in _UNIT_WORD_ORDINALS:
            return _CARDINAL_TENS[token] + _UNIT_WORD_ORDINALS[tokens[index + 1]], 2
        ordinal_value = _SIMPLE_WORD_ORDINALS.get(token)
        if ordinal_value is not None:
            return ordinal_value, 1
    ordinal_value = _SIMPLE_WORD_ORDINALS.get(token)
    if ordinal_value is not None:
        return ordinal_value, 1
    return None


def _word_ordinal_at(
    tokens: Sequence[str],
    index: int,
) -> tuple[int, int] | None:
    """Parse an explicit English ordinal phrase bounded to values below 1000."""
    if index >= len(tokens):
        return None
    token = tokens[index]
    if token == "hundredth":
        return 100, 1
    leading = _CARDINAL_UNITS.get(token)
    if (
        leading is not None
        and 1 <= leading <= 9
        and index + 1 < len(tokens)
    ):
        magnitude = tokens[index + 1]
        if magnitude == "hundredth":
            return leading * 100, 2
        if magnitude == "hundred":
            remainder_index = index + 2
            if remainder_index < len(tokens) and tokens[remainder_index] == "and":
                remainder_index += 1
            remainder = _ordinal_under_100_at(tokens, remainder_index)
            if remainder is not None:
                value, width = remainder
                return leading * 100 + value, remainder_index - index + width
    return _ordinal_under_100_at(tokens, index)


def _lexical_identity_token(raw_token: str) -> str:
    # Imported lazily because concept_tokens intentionally imports this module
    # only when concept identifier canonicalization is requested.
    try:
        from .concept_tokens import semantic_token_case_key
    except (ImportError, AttributeError):
        return raw_token.casefold()
    return semantic_token_case_key(raw_token)


def canonicalize_ordinal_tokens(
    values: Iterable[object],
    *,
    numbered_kind_tokens: Iterable[str] = (),
) -> tuple[str, ...]:
    """Canonicalize word, numeric, and compound ordinals without domain lists.

    Plain integers are treated as ordinals only beside a caller-supplied numbered
    concept kind (for example ``law 5``); explicit ``5th`` and word ordinals are
    unambiguous and normalize everywhere.
    """
    raw_tokens = tuple(str(value or "").strip() for value in values)
    tokens = tuple(value.casefold() for value in raw_tokens)
    kinds = frozenset(str(value).casefold() for value in numbered_kind_tokens)
    normalized: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        word_ordinal = _word_ordinal_at(tokens, index)
        if word_ordinal is not None:
            value, width = word_ordinal
            normalized.append(_canonical_ordinal_token(value))
            index += width
            continue
        numeric = _NUMERIC_ORDINAL.fullmatch(token)
        if numeric is not None:
            normalized.append(_canonical_ordinal_token(int(numeric.group("number"))))
            index += 1
            continue
        plain_number = _plain_number_at(tokens, index, raw_tokens=raw_tokens)
        if plain_number is not None:
            value, width = plain_number
            if _plain_value_is_numbered_label(
                tokens,
                index,
                width,
                kinds,
                raw_tokens=raw_tokens,
            ):
                normalized.append(_canonical_ordinal_token(value))
                index += width
                continue
        normalized.append(_lexical_identity_token(raw_tokens[index]))
        index += 1
    return tuple(normalized)


def ordinal_indexes(tokens: Iterable[object]) -> frozenset[int]:
    """Return canonical ordinal numbers from already-normalized tokens."""
    indexes: set[int] = set()
    for raw in tokens:
        token = str(raw or "")
        word_value = _SIMPLE_WORD_ORDINALS.get(token)
        if word_value is not None:
            indexes.add(word_value)
            continue
        if not token.startswith(_CANONICAL_PREFIX):
            continue
        try:
            indexes.add(int(token[len(_CANONICAL_PREFIX) :]))
        except ValueError:
            continue
    return frozenset(indexes)


def numbered_ordinal_indexes(
    tokens: Iterable[object],
    *,
    numbered_kind_tokens: Iterable[str] = NUMBERED_CONCEPT_KIND_TOKENS,
) -> frozenset[int]:
    """Return ordinals syntactically attached to a numbered concept phrase."""
    values = tuple(str(value or "").casefold() for value in tokens)
    kinds = frozenset(str(value).casefold() for value in numbered_kind_tokens)
    ordinal_positions = {
        index: next(iter(ordinal_indexes((token,))), None)
        for index, token in enumerate(values)
        if is_canonical_ordinal_token(token)
    }
    kind_positions = [
        index for index, token in enumerate(values) if token in kinds
    ]
    found: set[int] = set()
    for ordinal_position, ordinal_value in ordinal_positions.items():
        if ordinal_value is None:
            continue
        if (
            ordinal_position > 0
            and values[ordinal_position - 1] in _RATE_MARKERS
        ):
            continue
        for kind_position in kind_positions:
            left, right = sorted((ordinal_position, kind_position))
            between = values[left + 1 : right]
            if len(between) > 5:
                continue
            if all(
                token in _NUMBERED_LIST_CONNECTORS
                or is_canonical_ordinal_token(token)
                for token in between
            ):
                found.add(ordinal_value)
                break
    return frozenset(found)


def concept_ordinal_indexes(tokens: Iterable[object]) -> frozenset[int]:
    """Return explicit concept ordinals while excluding discourse/rate uses."""
    values = tuple(str(value or "").casefold() for value in tokens)
    found = set(numbered_ordinal_indexes(values))
    for index, token in enumerate(values):
        indexes = ordinal_indexes((token,))
        if not indexes:
            continue
        if index > 0 and values[index - 1] in _RATE_MARKERS:
            continue
        if index + 1 < len(values) and values[index + 1] in _ENUMERATION_FOLLOWERS:
            continue
        found.update(indexes)
    return frozenset(found)


def standalone_numeric_identifiers(tokens: Iterable[object]) -> frozenset[str]:
    """Return conservative named-entity numbers, excluding counts and quantities."""
    values = tuple(str(value or "").casefold() for value in tokens)
    found: set[str] = set()
    for index, token in enumerate(values):
        if not _PLAIN_NUMBER.fullmatch(token):
            continue
        previous = values[index - 1] if index else ""
        following = values[index + 1] if index + 1 < len(values) else ""
        if previous in _NUMERIC_CONTEXT_LEADS or previous in _NUMERIC_ENUMERATION_LEADS:
            continue
        if following in _NUMERIC_UNIT_FOLLOWERS:
            continue
        if following.endswith("s") and len(following) > 2 and not following.endswith("ss"):
            continue
        found.add(str(int(token)))
    return frozenset(found)


def canonicalize_concept_identifier_tokens(
    values: Iterable[object],
    *,
    numbered_kind_tokens: Iterable[str] = NUMBERED_CONCEPT_KIND_TOKENS,
) -> tuple[str, ...]:
    """Canonicalize only explicit or strong-grammar concept identifiers."""
    return canonicalize_ordinal_tokens(
        values,
        numbered_kind_tokens=numbered_kind_tokens,
    )


def concept_identifier_indexes(tokens: Iterable[object]) -> frozenset[int]:
    """Return only explicit or strong-grammar ordinal concept identifiers."""
    return concept_ordinal_indexes(tuple(tokens))


def is_canonical_ordinal_token(value: object) -> bool:
    token = str(value or "")
    return token in _SIMPLE_WORD_ORDINALS or (
        token.startswith(_CANONICAL_PREFIX)
        and token[len(_CANONICAL_PREFIX) :].isdigit()
    )
