"""Structural role classifier for transcript passages.

Labels a passage as INTRO / OUTRO / SPONSOR / RECAP / TRANSITION /
SUBSTANTIVE based on lexical cues plus a position-in-video prior. Pure
regex — runs in microseconds on CPU, no model load.

Companion to ``_is_back_reference_opener`` / ``_classify_hook_pattern`` in
clip_boundary.py — those classify *opener style*, this classifies *passage
role*. They compose: a sponsor read may also have a backref opener.

Why position matters: the phrase "thanks for watching" near second 0 is
almost always quoted ("a lot of channels start with 'thanks for watching'");
near the end of the video it is almost always an outro. The `t_start` /
`video_duration` arguments encode this prior.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class StructuralLabel:
    name: str
    confidence: float
    matched: str | None = None


SUBSTANTIVE = StructuralLabel("substantive", 0.0, None)


_INTRO_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"^\s*(hey|hi|hello|what'?s up|yo)\s+(guys|everyone|folks|y'?all|all|friends|team)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(hey|hi|hello)[, ]+\w+[, ]+(here|speaking)\b",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*welcome\s+(back\s+)?to\b", re.IGNORECASE),
    re.compile(
        r"\bmy\s+name\s+is\s+\w+\b[^.]*\b(today|this video|in this video|going to)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bin\s+(today'?s|this)\s+(video|episode|lesson|talk|tutorial)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\btoday\s+(we'?re|i'?m|we\s+will|i\s+will)\s+"
        r"(going\s+to\s+|gonna\s+)?(talk\s+about|cover|discuss|explore|look\s+at|learn\s+about)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(what|here'?s\s+what)\s+we'?re\s+going\s+to\s+(talk\s+about|cover|do|build)\b",
        re.IGNORECASE,
    ),
)


_OUTRO_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\bthank(s|\s+you)\b[^.]{0,40}\b(for\s+watching|for\s+tuning|for\s+listening|for\s+joining)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(don'?t\s+forget\s+to|please|make\s+sure\s+to|be\s+sure\s+to|remember\s+to)\s+"
        r"(like|subscribe|comment|share|hit|smash|click|tap)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(hit|smash|tap|click|ring)\s+(that|the)\s+(like|subscribe|bell|notification)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(see|catch)\s+you\s+(in\s+the\s+next|next\s+time|tomorrow|soon|later)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\buntil\s+next\s+time\b", re.IGNORECASE),
    re.compile(
        r"\bleave\s+(a|your)\s+(comment|thoughts|question)s?\s+(below|down\s+below|in\s+the\s+comments)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bif\s+you\s+(enjoyed|liked|found this).{0,30}\b(like|subscribe|share)\b",
        re.IGNORECASE,
    ),
)


_SPONSOR_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(this|today'?s)\s+(video|episode)\s+is\s+(brought\s+to\s+you\s+by|sponsored\s+by|made\s+possible\s+by)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(thanks|thank\s+you)\s+to\s+(our|today'?s|this\s+episode'?s)\s+sponsor\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bsponsored\s+by\b", re.IGNORECASE),
    re.compile(r"\b(use|with)\s+(the\s+)?(code|promo\s+code|discount\s+code)\s+[A-Z0-9]{3,}\b"),
    re.compile(
        r"\b(go\s+to|visit|head\s+(to|over\s+to)|check\s+out)\s+[a-z0-9-]+\.(com|io|co|org|net|ai|app)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bsupport(ers)?\s+on\s+patreon\b|\bpatreon\s+supporters\b|\bon\s+patreon\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bthank(s|\s+you)?\s+to\s+(my|our|all\s+the)\s+(patreon|patrons|supporters|backers)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(first\s+\d+|next\s+\d+)\s+(people|viewers|subscribers|to\s+sign\s+up)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(get|receive|enjoy)\s+\d+%?\s+(off|discount)\b",
        re.IGNORECASE,
    ),
)


_RECAP_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"^\s*(as|like)\s+(we|i)\s+(discussed|mentioned|saw|covered|talked\s+about|said|noted)\s+"
        r"(earlier|before|previously|in\s+the\s+last|already)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(remember|recall|recap)\b[^.]{0,40}\b(when|that|how|the|earlier)\b",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*(so|now)[, ]+(going|coming)\s+back\s+to\b", re.IGNORECASE),
    re.compile(r"^\s*to\s+(recap|summarize|review|wrap\s+up)\b", re.IGNORECASE),
    re.compile(r"^\s*let'?s\s+(recap|summarize|review)\b", re.IGNORECASE),
)


_TRANSITION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(moving|jumping|switching)\s+on\b", re.IGNORECASE),
    re.compile(r"^\s*next\s+(up|on\s+the\s+agenda|i\s+want\s+to|we'?ll)\b", re.IGNORECASE),
    re.compile(
        r"^\s*(let'?s|we'?ll)\s+(switch\s+gears|move\s+on|turn\s+to|take\s+a\s+look\s+at)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*before\s+(we|i)\s+(continue|move\s+on|get\s+into|dive\s+into)\b",
        re.IGNORECASE,
    ),
)


PENALTY_BY_LABEL: dict[str, float] = {
    "intro": 0.45,
    "outro": 0.50,
    "sponsor": 0.60,
    "recap": 0.20,
    "transition": 0.10,
    "substantive": 0.0,
}


def classify_passage(
    text: str,
    *,
    t_start: float | None = None,
    video_duration: float | None = None,
) -> StructuralLabel:
    """Return the structural role of ``text`` with a position-aware prior.

    Pass the *first sentence* of a window for INTRO/RECAP/TRANSITION
    detection (those cues live in the opener). For OUTRO/SPONSOR, pass the
    *full window text* — sign-offs and sponsor reads can arrive at any
    point inside the window.

    ``t_start`` (seconds into video) and ``video_duration`` (seconds) shift
    priors: an INTRO match in the last 30% of the video is treated as a
    quoted reference, not a real intro, and analogously for OUTRO matches
    in the first 50%. Without timing, classification is text-only.
    """
    if not text:
        return SUBSTANTIVE
    snippet = text[:600]

    fraction: float | None = None
    if t_start is not None and video_duration and video_duration > 0:
        fraction = max(0.0, min(1.0, t_start / video_duration))

    intro_boost = 0.0
    outro_boost = 0.0
    if fraction is not None:
        if fraction < 0.05:
            intro_boost = 0.30
        elif fraction < 0.10:
            intro_boost = 0.20
        if fraction > 0.92:
            outro_boost = 0.30
        elif fraction > 0.85:
            outro_boost = 0.20

    for pat in _SPONSOR_PATTERNS:
        if pat.search(snippet):
            return StructuralLabel(
                "sponsor", min(1.0, 0.75 + outro_boost), pat.pattern
            )

    for pat in _INTRO_PATTERNS:
        if pat.search(snippet):
            if fraction is not None and fraction > 0.30:
                continue
            return StructuralLabel(
                "intro", min(1.0, 0.70 + intro_boost), pat.pattern
            )

    for pat in _OUTRO_PATTERNS:
        if pat.search(snippet):
            if fraction is not None and fraction < 0.50:
                continue
            return StructuralLabel(
                "outro", min(1.0, 0.70 + outro_boost), pat.pattern
            )

    for pat in _RECAP_PATTERNS:
        if pat.search(snippet):
            return StructuralLabel("recap", 0.65, pat.pattern)

    for pat in _TRANSITION_PATTERNS:
        if pat.search(snippet):
            return StructuralLabel("transition", 0.55, pat.pattern)

    return SUBSTANTIVE


def label_penalty(label: StructuralLabel) -> float:
    """Confidence-scaled penalty for ``label`` — caller subtracts from score."""
    base = PENALTY_BY_LABEL.get(label.name, 0.0)
    return base * label.confidence
