"""Classify a user's search query as ``broad`` / ``narrow`` / ``medium`` / ``none``
and extract the anchor terms the importance ranker should chase.

A *broad* query asks for survey-style coverage ("Physics 1 review",
"AP Bio everything", "intro to calculus"). The downstream ranker should
return 3-5 clips spanning distinct subtopics.

A *narrow* query points at one specific concept ("torque", "B=qvBsinθ",
"Treaty of Versailles", "photosynthesis"). The ranker should return 1-2
clips concentrated on that concept.

The classifier is pure stdlib: regex + a closed school-subject bucket.
No ML, no model load. Runs in microseconds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from .text_utils import STOPWORDS


IntentType = Literal["broad", "narrow", "medium", "none"]


@dataclass(frozen=True)
class QueryIntent:
    type: IntentType
    confidence: float
    anchors: tuple[str, ...]
    raw_query: str
    normalized_query: str


_BROAD_VERBS: frozenset[str] = frozenset({
    "review", "overview", "intro", "introduction", "summary", "recap",
    "explain", "explained", "cover", "covers", "everything", "all",
    "complete", "comprehensive", "guide", "tour", "survey", "fundamentals",
    "basics",
})

# Closed school-subject bucket. ONLY these tokens trigger the bare-subject
# broad path. Concept terms ("torque", "photosynthesis", "federalism") are
# explicitly NOT in here — they fall through to narrow.
_SCHOOL_SUBJECTS: frozenset[str] = frozenset({
    "physics", "biology", "chemistry", "history", "spanish", "english",
    "calculus", "algebra", "geometry", "statistics", "psychology",
    "sociology", "economics", "government", "literature", "geography",
    "anatomy", "civics", "trigonometry", "precalculus",
})

# AP/IB course-name shortlist — bare or with the AP/IB prefix counts as a
# broad query.
_AP_IB_COURSE_TOKENS: frozenset[str] = frozenset({
    "ap", "ib",
    # course suffixes that pair with AP/IB
    "macro", "micro", "macroeconomics", "microeconomics", "comp",
    "lang", "lit", "world", "us", "european", "human",
})

_COURSE_CODE_RE = re.compile(r"\b([A-Z]{2,4})\s?-?\s?(\d{2,4})\b")
_EQUATION_RE = re.compile(
    r"[A-Za-zα-ω][A-Za-z_α-ω0-9]{0,3}\s*=\s*[A-Za-zα-ω0-9·\*\(\)/\^\+\-]"
)
# Multi-word capitalized proper nouns. Two-or-more capitalized words OR a
# hyphenated multi-cap.
_PROPER_NOUN_RE = re.compile(
    r"\b[A-Z][a-zA-Z]+(?:\s+(?:of|the|de|and|von|la)\s+[A-Z][a-zA-Z]+|\s+[A-Z][a-zA-Z]+|-[A-Z][a-zA-Z]+)+"
)
_FORMULA_THEOREM_RE = re.compile(
    r"\b(?:formula|theorem|equation|law|principle|rule|identity)\s+(?:for|of)\b"
    r"|\bthe\s+\w+\s+(?:formula|theorem|equation|law|principle|rule|identity)\b",
    re.IGNORECASE,
)


def _content_tokens(text: str) -> list[str]:
    """Lowercase non-stopword alphanumeric tokens (≥2 chars)."""
    raw = re.findall(r"[A-Za-zα-ω][A-Za-zα-ω0-9\-']*", text.lower())
    return [t for t in raw if t not in STOPWORDS and len(t) > 1]


def _extract_anchors(query: str, tokens: list[str]) -> list[str]:
    """Anchors = (multi-word proper nouns) ∪ (equation matches) ∪ (content tokens)."""
    anchors: list[str] = []
    seen: set[str] = set()

    for m in _PROPER_NOUN_RE.finditer(query):
        span = m.group(0).strip()
        key = span.lower()
        if key not in seen:
            seen.add(key)
            anchors.append(span)

    for m in _EQUATION_RE.finditer(query):
        span = m.group(0).strip()
        key = span.lower()
        if key not in seen:
            seen.add(key)
            anchors.append(span)

    for tok in tokens:
        if tok in seen:
            continue
        seen.add(tok)
        anchors.append(tok)

    return anchors


def classify_query(query: str | None) -> QueryIntent:
    """Return the intent classification + anchors for ``query``.

    Order of checks: empty → broad → narrow → medium fallback.
    """
    raw = (query or "").strip()
    if not raw:
        return QueryIntent(
            type="none", confidence=1.0, anchors=(), raw_query="", normalized_query="",
        )

    normalized = re.sub(r"\s+", " ", raw)
    tokens = _content_tokens(raw)

    has_broad_verb = any(t in _BROAD_VERBS for t in tokens)
    has_course_code = bool(_COURSE_CODE_RE.search(raw))
    has_ap_ib = any(t in _AP_IB_COURSE_TOKENS for t in tokens) and len(tokens) >= 2
    is_bare_subject = (
        len(tokens) <= 2
        and any(t in _SCHOOL_SUBJECTS for t in tokens)
    )

    if has_broad_verb or has_course_code or has_ap_ib or is_bare_subject:
        anchors = _extract_anchors(raw, tokens)
        return QueryIntent(
            type="broad", confidence=0.85, anchors=tuple(anchors),
            raw_query=raw, normalized_query=normalized,
        )

    has_equation = bool(_EQUATION_RE.search(raw))
    has_proper_noun = bool(_PROPER_NOUN_RE.search(raw))
    has_formula_pattern = bool(_FORMULA_THEOREM_RE.search(raw))
    short_concept = (
        1 <= len(tokens) <= 3
        and not any(t in _SCHOOL_SUBJECTS for t in tokens)
    )

    if has_equation or has_proper_noun or has_formula_pattern or short_concept:
        anchors = _extract_anchors(raw, tokens)
        return QueryIntent(
            type="narrow", confidence=0.85, anchors=tuple(anchors),
            raw_query=raw, normalized_query=normalized,
        )

    anchors = _extract_anchors(raw, tokens)
    return QueryIntent(
        type="medium", confidence=0.4, anchors=tuple(anchors),
        raw_query=raw, normalized_query=normalized,
    )


__all__ = ["QueryIntent", "IntentType", "classify_query"]
