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

_OPENING_ASIDE_RE = re.compile(
    r"^(?:oh\s*[,;:]?\s*)?(?:yeah\s*[,;:]?\s*)?by\s+the\s+way\b",
    re.IGNORECASE,
)

_ELLIPTICAL_SPATIAL_INSTRUCTION_RE = re.compile(
    r"^(?:little|slightly|just)\s+(?:above|below|behind|inside|outside|over|under)\b",
    re.IGNORECASE,
)

_UNRESOLVED_ACTION_REFERENCE_RE = re.compile(
    r"^(?:i|we|you)\s+can\s+(?:just\s+)?do\s+(?:that|this|it)\b",
    re.IGNORECASE,
)

_UNRESOLVED_EMBEDDED_ANAPHOR_RE = re.compile(
    r"^(?:(?:do|did)\s+)?you\s+know\s+(?:what|how|why)\s+"
    r"(?:else\s+)?(?:it|this|that|these|those|they|he|she)\b|"
    r"^(?:what|how|why)\s+(?:else\s+)?"
    r"(?:it|this|that|these|those|they|he|she)\b|"
    r"^(?:one|another)\s+(?:thing|function)\s+"
    r"(?:it|this|that|these|those|they|he|she)\s+can\s+do\b",
    re.IGNORECASE,
)

_OPENING_UNRESOLVED_OBJECT_RE = re.compile(
    r"^(?:[a-z0-9']+\s+){1,5}(?:(?:this|that)\s+one|it|them)\b"
    r"(?=\s*(?:$|[,.!?;:]|\b(?:away|by|from|into|onto|out|using|via|with)\b))",
    re.IGNORECASE,
)

_OPENING_DEICTIC_LOCATION_RE = re.compile(
    r"^(?!(?:here|there)\s+(?:is|are|was|were)\b)"
    r"(?![^.!?]{0,80}\b(?:(?:is|are|was|were)\s+there|"
    r"there\s+(?:is|are|was|were))\b)(?:"
    r"(?:(?:right|over|up|down)\s+)?(?:here|there)\b|"
    r"(?:what|where|how|why|which|who)\b[^.!?]{0,80}\b(?:here|there)\b|"
    r"let(?:['’]?s|\s+us)\s+(?:say\s+)?(?:here|there)\b|"
    r"(?:[a-z0-9'-]+\s+){0,5}(?:base|end|line|point|position|side|strand)\s+"
    r"(?:right\s+)?(?:here|there)\b"
    r")",
    re.IGNORECASE,
)

_OPENING_BARE_PREDICATE_RE = re.compile(
    r"^(?:(?:have|has|had)\s+to|"
    r"(?:is|are|was|were)\s+(?:called|defined|found|known|located|made|used))\b",
    re.IGNORECASE,
)

_OPENING_QUANTIFIED_DEMONSTRATIVE_RE = re.compile(
    r"^(?:"
    r"(?:one|any|some|each|either|neither)\s+of\s+(?:these|those)|"
    r"(?:if|when|because|although|while|since)\s+"
    r"(?:(?:one|any|some|each|either|neither)\s+of\s+)?"
    r"(?:this|that|these|those)"
    r")\b",
    re.IGNORECASE,
)

_CONTEXTUAL_REFORMULATION_RE = re.compile(
    r"^(?:(?:to\s+be\s+(?:more\s+)?(?:exact|precise))|"
    r"(?:more\s+(?:exactly|precisely))|(?:strictly|technically)\s+speaking|"
    r"in\s+other\s+words|put\s+differently|"
    r"to\s+put\s+(?:it|that)\s+another\s+way)\s*[,;:]?\s+"
    r"(?:the|this|that|these|those|it|they|them|such|both|either|neither|"
    r"former|latter)\b",
    re.IGNORECASE,
)

_EXPLICIT_BACK_REFERENCE_RE = re.compile(
    r"\b(?:as\s+(?:before|above|earlier|previously)|"
    r"(?:shown|mentioned|discussed|described|defined|noted)\s+"
    r"(?:above|before|earlier|previously))\b",
    re.IGNORECASE,
)

_CONTEXTUAL_DEMONSTRATIVE_HEADS: frozenset[str] = frozenset({
    "answer", "answers", "approach", "approaches", "assumption", "assumptions",
    "calculation", "calculations", "case", "cases", "condition", "conditions",
    "dependency", "dependencies", "difference", "differences", "equation", "equations",
    "expression", "expressions", "method", "methods", "parameter", "parameters",
    "property", "properties", "quantity", "quantities", "reason", "reasons",
    "relationship", "relationships", "result", "results", "rule", "rules",
    "solution", "solutions", "step", "steps", "term", "terms", "value", "values",
    "variable", "variables",
})

_DEMONSTRATIVE_NP_RE = re.compile(
    r"\b(?:these|those)\s+(?P<head>[a-z][a-z'-]*)\b",
    re.IGNORECASE,
)

_REFERENCE_TERM_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "before", "by",
    "could", "did", "do", "does", "earlier", "for", "from", "had", "has",
    "have", "in", "is", "it", "not", "of", "on", "or", "previously", "that",
    "the", "these", "this", "those", "to", "was", "we", "were", "will", "with",
    "would",
})

_EXPLICIT_REFERENCE_TRIGGER_WORDS: frozenset[str] = frozenset({
    "above", "before", "defined", "described", "discussed", "earlier",
    "mentioned", "noted", "previously", "shown",
})

_MIN_EXPLICIT_REFERENCE_SHARED_TERMS = 2

_DANGLING_QUESTION_REFERENCE_RE = re.compile(
    r"(?:\b(?:this|that|these|those|it|they|them)\s*\?|"
    r"\b(?:how|why|when|where|what)\s+"
    r"(?:does|do|did|is|are|was|were|can|could|would|should|will)\s+"
    r"(?:this|that|these|those|it|they|them)\b)",
    re.IGNORECASE,
)

_ANAPHORIC_QUESTION_RE = re.compile(
    r"\b(?:is|are|was|were|does|do|did|can|could|would|should|will)\s+"
    r"(?:it|this|that|these|those|they)\b[^?]*\?",
    re.IGNORECASE,
)

_LOCAL_QUESTION_CONTEXT_RE = re.compile(
    r"(?:[.!?]\s+|(?:,\s*)?\b(?:and|or)\s+)$",
    re.IGNORECASE,
)

_QUESTION_CONTEXT_GENERIC_WORDS = frozenset({
    "a", "an", "and", "are", "be", "been", "can", "clarify", "could", "did",
    "do", "does", "explain", "finished", "have", "here", "how", "i", "in",
    "introduction", "is", "it", "matter", "one", "or", "part", "process",
    "question", "section", "show", "tell", "that", "the", "these", "they",
    "thing", "this", "those", "to", "topic", "was", "we", "were", "what",
    "when", "where", "which", "why", "will", "work", "would", "you",
})

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

# These markers can introduce a fresh pedagogical question without requiring a
# prior contrast. Adversative/causal markers such as "but" and "because" cannot.
_QUESTION_FRAMING_MARKERS = frozenset({"now", "so"})

_DEMONSTRATIVE_DETERMINERS = frozenset({"this", "that", "these", "those"})

_FRAMING_WINDOW_WORDS = 20

_ANTICIPATORY_IT_RE = re.compile(
    r"^it\s+(?:is|was|can\s+be|could\s+be|may\s+be|might\s+be|"
    r"will\s+be|would\s+be)\s+(?:(?:especially|often|sometimes|very)\s+)?"
    r"(?:clear|crucial|difficult|easy|easier|essential|hard|harder|helpful|"
    r"important|likely|necessary|possible|unlikely|useful)\b",
    re.IGNORECASE,
)


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text or "")


def first_lexical_character_index(text: str) -> int | None:
    """Return the first letter/digit index, ignoring opening quotes/brackets."""
    return next(
        (index for index, character in enumerate(text or "") if character.isalnum()),
        None,
    )


def _is_framing_or_question(text: str, words: list[str]) -> bool:
    """A leading marker is fine when the sentence stands on its own as new framing/a question."""
    low = (text or "").lower()
    opening_match = re.match(r"^([^.!?]*)([.!?]?)", low, re.DOTALL)
    opening_text = opening_match.group(1) if opening_match else low
    opening_terminator = opening_match.group(2) if opening_match else ""
    opening_words = list(_WORD_RE.finditer(opening_text))
    contextual_question_marker = bool(
        words
        and words[0].lower() in CONTINUATION_MARKERS
        and words[0].lower() not in _QUESTION_FRAMING_MARKERS
    )
    if opening_terminator == "?" and contextual_question_marker:
        return False
    if (
        opening_terminator == "?"
        and len(opening_words) <= _FRAMING_WINDOW_WORDS
    ):
        return True
    if len(opening_words) > _FRAMING_WINDOW_WORDS:
        opening_text = opening_text[:opening_words[_FRAMING_WINDOW_WORDS - 1].end()]
    if any(
        re.search(
            rf"(?<![a-z0-9']){re.escape(pat.strip())}(?![a-z0-9'])",
            opening_text,
        )
        for pat in _FRAMING_PATTERNS
    ):
        return True
    # A question that lost its '?' in ASR: starts with a pure interrogative word.
    if words and words[0].lower() in _INTERROGATIVE_WORDS:
        return True
    return False


def _has_unresolved_question_reference(text: str) -> bool:
    """Reject only question references without context earlier in the same cue.

    Caption providers can place several complete sentences or coordinated questions in
    one cue. A later pronoun can therefore resolve to a noun introduced earlier in that
    cue; treating every later ``why do they`` as dangling makes those valid openings
    impossible to recover by expanding farther backward.
    """
    for match in _DANGLING_QUESTION_REFERENCE_RE.finditer(text):
        prefix = text[:match.start()]
        boundary = _LOCAL_QUESTION_CONTEXT_RE.search(prefix)
        if boundary is None:
            return True
        context = prefix[:boundary.start()].strip()
        context_words = [word.casefold() for word in _words(context)]
        has_concrete_antecedent = any(
            len(word) >= 3 and word not in _QUESTION_CONTEXT_GENERIC_WORDS
            for word in context_words
        )
        if not has_concrete_antecedent:
            return True
    return False


def _reference_word_variants(word: str) -> set[str]:
    normalized = word.casefold()
    variants = {normalized}
    if normalized.endswith("ies"):
        variants.add(f"{normalized[:-3]}y")
    if normalized.endswith("es"):
        variants.add(normalized[:-2])
    if normalized.endswith("s"):
        variants.add(normalized[:-1])
    return variants


def _canonical_reference_term(word: str) -> str:
    normalized = word.casefold()
    if len(normalized) > 4 and normalized.endswith("ies"):
        return f"{normalized[:-3]}y"
    if len(normalized) > 4 and normalized.endswith(
        ("ches", "shes", "xes", "zes", "sses", "oes")
    ):
        return normalized[:-2]
    if (
        len(normalized) > 3
        and normalized.endswith("s")
        and not normalized.endswith(("ss", "us", "is"))
    ):
        return normalized[:-1]
    return normalized


def _reference_terms(text: str) -> set[str]:
    return {
        _canonical_reference_term(word)
        for word in _words(text)
        if (
            len(word) >= 3
            and word.casefold() not in _REFERENCE_TERM_STOP_WORDS
            and word.casefold() not in _EXPLICIT_REFERENCE_TRIGGER_WORDS
        )
        and len(_canonical_reference_term(word)) >= 3
    }


def _explicit_reference_has_antecedent(
    text: str,
    match: re.Match[str],
    *,
    prior_text: str,
) -> bool:
    sentence_start = max(
        text.rfind(".", 0, match.start()),
        text.rfind("!", 0, match.start()),
        text.rfind("?", 0, match.start()),
    ) + 1
    sentence_end_candidates = [
        index
        for marker in ".!?"
        for index in [text.find(marker, match.end())]
        if index >= 0
    ]
    sentence_end = min(sentence_end_candidates, default=len(text))
    external_context = f"{prior_text} {text[:sentence_start]}".strip()
    sentence_prefix = text[sentence_start:match.start()]
    sentence_suffix = text[match.end():sentence_end]
    reference_terms = _reference_terms(f"{sentence_prefix} {sentence_suffix}")
    if len(_reference_terms(external_context) & reference_terms) >= (
        _MIN_EXPLICIT_REFERENCE_SHARED_TERMS
    ):
        return True

    # ASR cues do not always contain sentence punctuation. Permit an antecedent
    # established earlier in the same clause only when its content is repeated
    # after the backward-reference phrase; the phrase's own trigger verb is not
    # evidence of context.
    return len(
        _reference_terms(sentence_prefix)
        & _reference_terms(sentence_suffix)
    ) >= (
        _MIN_EXPLICIT_REFERENCE_SHARED_TERMS
    )


def _has_unresolved_opening_back_reference(
    text: str,
    *,
    prior_text: str = "",
) -> bool:
    """Find opening references without a lexical antecedent in accumulated context."""
    for match in _EXPLICIT_BACK_REFERENCE_RE.finditer(text):
        if not _explicit_reference_has_antecedent(
            text,
            match,
            prior_text=prior_text,
        ):
            return True
    for match in _DEMONSTRATIVE_NP_RE.finditer(text):
        head = match.group("head").casefold()
        if head not in _CONTEXTUAL_DEMONSTRATIVE_HEADS:
            continue
        prefix_words = {
            variant
            for word in _words(f"{prior_text} {text[:match.start()]}")
            for variant in _reference_word_variants(word)
        }
        variants = _reference_word_variants(head)
        if not variants.intersection(prefix_words):
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
    lexical_index = first_lexical_character_index(stripped)
    lexical_text = stripped[lexical_index:] if lexical_index is not None else ""
    if lexical_text and lexical_text[0].islower():
        return True

    if (
        _MID_LIST_OR_REPLY_RE.match(lexical_text)
        or _ALL_RIGHT_REPLY_RE.match(lexical_text)
        or _OPENING_ASIDE_RE.match(lexical_text)
        or _ELLIPTICAL_SPATIAL_INSTRUCTION_RE.match(lexical_text)
        or _UNRESOLVED_ACTION_REFERENCE_RE.match(lexical_text)
        or _UNRESOLVED_EMBEDDED_ANAPHOR_RE.match(lexical_text)
        or _OPENING_UNRESOLVED_OBJECT_RE.match(lexical_text)
        or _OPENING_DEICTIC_LOCATION_RE.match(lexical_text)
        or _OPENING_BARE_PREDICATE_RE.match(lexical_text)
        or _OPENING_QUANTIFIED_DEMONSTRATIVE_RE.match(lexical_text)
        or _CONTEXTUAL_REFORMULATION_RE.match(lexical_text)
    ):
        return True

    if _has_unresolved_opening_back_reference(lexical_text):
        return True

    # A question is not self-contained merely because it has punctuation.
    # "How does the cell do that?" still requires the missing prior action.
    if _has_unresolved_question_reference(stripped):
        return True

    w0 = words[0].lower()
    w1 = words[1].lower() if len(words) > 1 else ""
    if _ANTICIPATORY_IT_RE.match(lexical_text):
        return False
    if w0 in CONTINUATION_MARKERS and _ANAPHORIC_QUESTION_RE.search(stripped):
        return True
    if w0 in CONTINUATION_MARKERS and w1 in _DEMONSTRATIVE_DETERMINERS:
        return True

    framing = _is_framing_or_question(lexical_text, words)
    if framing:
        return False                       # self-contained framing/question wins outright

    # 1) leading continuation marker without framing → mid-thought
    if w0 in CONTINUATION_MARKERS:
        return True
    if w0 == "back" and w1 == "to":
        return True
    # 2) dangling anaphor (pronoun immediately before a verb/aux) → unresolved reference
    if w0 in _DEMONSTRATIVE_DETERMINERS:
        return True
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
