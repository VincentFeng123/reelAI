"""Discourse-onset detection (text-only, genre-independent).

Decides whether a sentence, used as a clip's FIRST line, drops a cold viewer mid-thought.
Grounded in: Decontextualization (Choi et al., TACL 2021) — the dominant context-dependence
is a dangling referring expression (~40%); and cue-phrase disambiguation (Hirschberg & Litman
1993) — a leading discourse marker ("so", "now") is a genuine onset when the sentence is
self-contained framing or a question, and a mid-thought continuation otherwise.

Shared by the snap START guard (refine.py) and the opening_onset_rate metric (eval/metrics.py).
"""
from __future__ import annotations

import re

_WORD_RE = re.compile(r"[a-z0-9']+", re.IGNORECASE)

# Leading tokens that, without self-contained framing, signal continuation of a prior thought.
CONTINUATION_MARKERS: frozenset[str] = frozenset({
    "and", "so", "but", "because", "cuz", "cause", "therefore", "thus", "hence",
    "anyway", "anyways", "also", "plus", "or", "nor", "yet", "then", "well",
    "okay", "ok", "alright", "actually", "basically", "meanwhile", "however",
    "moreover", "furthermore", "additionally", "consequently", "again",
    "yeah", "yep",
})

_MID_LIST_OR_REPLY_RE = re.compile(
    r"^(?:(?:second|third|fourth)(?:ly)?|finally|no|right|yes|exactly)"
    r"(?:\s*[,;:]|\s*[-–—]\s+)",
    re.IGNORECASE,
)

_ALL_RIGHT_REPLY_RE = re.compile(
    r"^all\s+right\s*[.!?,;:]",
    re.IGNORECASE,
)

_DANGLING_QUESTION_REFERENCE_RE = re.compile(
    r"(?:\b(?:this|that|these|those|it|they|them)\s*\?|"
    r"\b(?:how|why|when|where|what)\s+"
    r"(?:does|do|did|is|are|was|were|can|could|would|should|will)\s+"
    r"(?:this|that|these|those|it|they|them)\b)",
    re.IGNORECASE,
)

# Bare anaphora that, as the first word before a verb/aux, lack an in-clip antecedent.
ANAPHORS: frozenset[str] = frozenset({
    "this", "that", "these", "those", "it", "they", "them", "he", "she",
    "him", "her", "here", "there", "its", "their",
})

# Definite-NP heads that are context-dependent ("the answer", "the previous equation").
CONTEXT_DEP_HEADS: frozenset[str] = frozenset({
    "answer", "result", "value", "number", "equation", "formula", "expression",
    "problem", "reason", "difference", "ratio", "sum", "product", "solution",
    "previous", "next", "first", "second", "third", "latter", "former", "same",
    "above", "below", "point", "step", "one", "thing",
})

# Verb/aux tokens; a leading anaphor immediately followed by one is a dangling reference.
_AUX_VERB: frozenset[str] = frozenset({
    "is", "are", "was", "were", "be", "been", "'s", "gives", "shows", "means",
    "tells", "makes", "gets", "goes", "comes", "has", "have", "had", "will",
    "would", "can", "could", "should", "does", "do", "did", "equals", "becomes",
})

# Framing / segment-onset cues: their presence makes even a marker-led sentence an onset.
_FRAMING_PATTERNS = (
    "let's", "lets", "let us", "we're going to", "we are going to", "we will",
    "we'll", "i want to", "i'm going to", "consider", "suppose", "imagine",
    "here's", "here is", "take ", "let me", "start with", "starting with",
    "look at", "move on to", "moving on", "turn to", "next up", "first,",
    "moving along to",
    "to begin", "begin with", "picture ", "think about", "what about",
    "how about", "say we", "say you",
)

_INTERROGATIVE_WORDS = frozenset({
    "what", "how", "why", "where", "when", "which", "who", "whose", "whom",
})

_INTERROGATIVE_FORMS = frozenset({
    "is", "are", "can", "could", "would", "should", "does", "do", "did", "will",
})


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text or "")


def _is_framing_or_question(text: str, words: list[str]) -> bool:
    """A leading marker is fine when the sentence stands on its own as new framing/a question."""
    low = (text or "").lower()
    if "?" in low:
        return True
    if any(pat in low for pat in _FRAMING_PATTERNS):
        return True
    # A question that lost its '?' in ASR: starts with a pure interrogative word.
    if words and words[0].lower() in _INTERROGATIVE_WORDS:
        return True
    return False


def opens_mid_thought(text: str) -> bool:
    """True when this sentence, as a clip's first line, drops the viewer mid-thought."""
    words = _words(text)
    if not words:
        return True

    # 4) mid-clause fragment: begins lowercase (checked first—overrides framing)
    #    (post-punctuation-restoration sentences are capitalized at true onsets)
    stripped = (text or "").lstrip()
    if stripped and stripped[0].islower():
        return True

    if _MID_LIST_OR_REPLY_RE.match(stripped) or _ALL_RIGHT_REPLY_RE.match(stripped):
        return True

    # A question is not self-contained merely because it has punctuation.
    # "How does the cell do that?" still requires the missing prior action.
    if _DANGLING_QUESTION_REFERENCE_RE.search(stripped):
        return True

    framing = _is_framing_or_question(text, words)
    if framing:
        return False                       # self-contained framing/question wins outright

    w0 = words[0].lower()
    w1 = words[1].lower() if len(words) > 1 else ""

    # 1) leading continuation marker without framing → mid-thought
    if w0 in CONTINUATION_MARKERS:
        return True
    if w0 == "back" and w1 == "to":
        return True
    # 2) dangling anaphor (pronoun immediately before a verb/aux) → unresolved reference
    if w0 in ANAPHORS and (w1 in _AUX_VERB or w1 == "s"):
        return True
    # 3) context-dependent definite NP: "the answer/previous/…"
    if w0 == "the" and w1 in CONTEXT_DEP_HEADS:
        return True
    # Too few words to be a complete opener
    if len(words) < 3:
        return True
    return False


def is_onset(text: str) -> bool:
    return not opens_mid_thought(text)
