"""Shared lexical normalization for concept identities and semantic query keys."""
from __future__ import annotations

import re
import unicodedata
from typing import NamedTuple


_SEMANTIC_SYMBOLS = str.maketrans({
    "\u2018": "'",
    "\u2019": "'",
    "\u02bc": "'",
    "♯": "#",
    "−": "-",
    "⁻": "-",
    "⁺": "+",
    "∗": "*",
    "⋆": "*",
    "≥": ">=",
    "≤": "<=",
    "≠": "!=",
    "→": "->",
    "⇒": "=>",
    "×": "*",
    "÷": "/",
})
_SEMANTIC_TOKEN_RE = re.compile(
    r"(?P<word>[^\W_]+(?:'[^\W_]+)*)"
    r"|(?P<operator>&&|\|\||\?\?|==|!=|<=|>=|<<|>>|\*\*|//|::|->|=>|\+\+|--|[&|=<>+*/%^~?!#-])"
    r"|(?P<symbol>[^\w\s])",
    re.UNICODE,
)
_ATTACHABLE_SUFFIX_CHARACTERS = frozenset("+#*?!-")
_ATTACHABLE_SUFFIX_SYMBOLS = frozenset({"•"})
_STRUCTURAL_MATH_LETTERS = frozenset({"ℂ", "ℍ", "ℕ", "ℙ", "ℚ", "ℝ", "ℤ"})
_CASE_INSENSITIVE_SHORT_WORDS = frozenset({
    "a", "an", "and", "as", "at", "be", "by", "do", "for", "from", "go",
    "he", "i", "if", "in", "is", "it", "law", "no", "of", "on", "or", "rule",
    "set", "so", "step", "the", "to", "type", "up", "us", "war", "we",
    "zero", "one", "two", "four", "five", "nine", "ten",
})
_WORD_PREFIX_RE = re.compile(r"^[^\W_]+(?:'[^\W_]+)*", re.UNICODE)


class _Lexeme(NamedTuple):
    value: str
    start: int
    end: int
    kind: str


def normalize_semantic_text(value: object, *, casefold: bool = True) -> str:
    """Normalize safe notation variants without erasing compatibility symbols."""
    text = unicodedata.normalize("NFC", str(value or "")).translate(_SEMANTIC_SYMBOLS)
    return text.casefold() if casefold else text


def is_structural_case_token(token: object) -> bool:
    """Return whether case is part of a compact formula, acronym, or math symbol."""
    raw = normalize_semantic_text(token, casefold=False)
    match = _WORD_PREFIX_RE.match(raw)
    if match is None:
        return False
    core = match.group(0)
    if any(character in _STRUCTURAL_MATH_LETTERS for character in core):
        return True
    letters = [character for character in core if character.isalpha()]
    if len(letters) == 1:
        return bool(
            raw == core
            and core.isupper()
            and core.casefold() not in _CASE_INSENSITIVE_SHORT_WORDS
        )
    if len(letters) < 2 or core.casefold() in _CASE_INSENSITIVE_SHORT_WORDS:
        return False
    if (
        len(letters) <= 4
        and all(character.isascii() and character.isupper() for character in letters)
    ):
        return True
    if len(letters) == 2 and letters[0].isupper() and letters[1].islower():
        return True
    return (
        not all(character.isupper() for character in letters)
        and any(character.isupper() for character in letters[1:])
    )


def semantic_token_case_key(token: object) -> str:
    """Case-fold prose tokens while retaining structurally meaningful token case."""
    raw = normalize_semantic_text(token, casefold=False)
    return raw if is_structural_case_token(raw) else raw.casefold()


def _raw_lexemes(text: str) -> list[_Lexeme]:
    lexemes: list[_Lexeme] = []
    for match in _SEMANTIC_TOKEN_RE.finditer(text):
        value = match.group(0)
        kind = str(match.lastgroup or "")
        if kind == "symbol":
            if value in _ATTACHABLE_SUFFIX_SYMBOLS:
                kind = "operator"
            elif unicodedata.category(value) != "Sm":
                continue
        lexemes.append(_Lexeme(value, match.start(), match.end(), kind))
    return lexemes


def _word_core(value: str) -> str:
    match = _WORD_PREFIX_RE.match(value)
    return match.group(0) if match is not None else ""


def _compact_operand(value: str) -> bool:
    core = _word_core(value)
    return bool(core) and (core.isdigit() or len(core) == 1)


def _operator_can_attach(value: str) -> bool:
    return bool(value) and (
        value in _ATTACHABLE_SUFFIX_SYMBOLS
        or all(character in _ATTACHABLE_SUFFIX_CHARACTERS for character in value)
    )


def _has_right_operand(
    lexemes: list[_Lexeme],
    operator_index: int,
    left_word: str,
) -> bool:
    if operator_index + 1 >= len(lexemes):
        return False
    following = lexemes[operator_index + 1]
    if following.kind != "word":
        return False
    operator = lexemes[operator_index]
    if operator.end == following.start:
        return True
    if operator.value in {"++", "--", "#"}:
        return False
    return _compact_operand(left_word) and _compact_operand(following.value)


def _combine_suffixes(lexemes: list[_Lexeme]) -> list[_Lexeme]:
    combined: list[_Lexeme] = []
    index = 0
    while index < len(lexemes):
        current = lexemes[index]
        if current.kind != "word":
            combined.append(current)
            index += 1
            continue
        value = current.value
        end = current.end
        cursor = index + 1
        while cursor < len(lexemes):
            suffix = lexemes[cursor]
            if (
                suffix.kind != "operator"
                or suffix.start != end
                or not _operator_can_attach(suffix.value)
                or _has_right_operand(lexemes, cursor, value)
            ):
                break
            value += suffix.value
            end = suffix.end
            cursor += 1
        combined.append(_Lexeme(value, current.start, end, "word"))
        index = cursor
    return combined


def _is_prose_separator(
    lexemes: list[_Lexeme],
    index: int,
) -> bool:
    current = lexemes[index]
    if current.value not in {"-", "/"} or not 0 < index < len(lexemes) - 1:
        return False
    previous = lexemes[index - 1]
    following = lexemes[index + 1]
    if (
        previous.kind != "word"
        or following.kind != "word"
    ):
        return False
    left = _word_core(previous.value)
    right = _word_core(following.value)
    return bool(
        left
        and right
        and left.isalpha()
        and right.isalpha()
        and (len(left) > 1 or len(right) > 1)
    )


def _is_detached_suffix_punctuation(
    lexemes: list[_Lexeme],
    index: int,
) -> bool:
    current = lexemes[index]
    if current.kind != "operator":
        return False
    if current.value == "#":
        return True
    if current.value != "+":
        return False
    return any(
        0 <= neighbor < len(lexemes) and lexemes[neighbor].value == "+"
        for neighbor in (index - 1, index + 1)
    )


def semantic_tokens(
    value: object,
    *,
    casefold: bool = True,
    preserve_terminal_suffix: bool = False,
) -> tuple[str, ...]:
    """Return words and bounded operators used by concept/query identity layers."""
    text = normalize_semantic_text(value, casefold=False)
    lexemes = _combine_suffixes(_raw_lexemes(text))
    filtered = [
        lexeme
        for index, lexeme in enumerate(lexemes)
        if not _is_prose_separator(lexemes, index)
        and not _is_detached_suffix_punctuation(lexemes, index)
    ]
    tokens: list[str] = []
    for index, lexeme in enumerate(filtered):
        token = lexeme.value
        terminal = index == len(filtered) - 1
        if terminal and not preserve_terminal_suffix:
            if lexeme.kind == "word":
                token = token.rstrip("!?")
            elif token in {"!", "?"}:
                continue
        if not token:
            continue
        tokens.append(
            semantic_token_case_key(token) if casefold else token
        )
    return tuple(tokens)


def semantic_key(value: object) -> str:
    """Return a stable, whitespace-delimited semantic identity key."""
    return " ".join(semantic_tokens(value))


def concept_semantic_tokens(
    value: object,
    *,
    preserve_terminal_suffix: bool = False,
) -> tuple[str, ...]:
    """Return concept-label tokens with strong-grammar identifiers normalized."""
    from .concept_ordinals import canonicalize_concept_identifier_tokens

    return canonicalize_concept_identifier_tokens(
        semantic_tokens(
            value,
            casefold=False,
            preserve_terminal_suffix=preserve_terminal_suffix,
        )
    )


def concept_semantic_key(
    value: object,
    *,
    preserve_terminal_suffix: bool = False,
) -> str:
    return " ".join(concept_semantic_tokens(
        value,
        preserve_terminal_suffix=preserve_terminal_suffix,
    ))
