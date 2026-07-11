"""Rebuild readable output from ORIGINAL tokens + reconciled annotations.

Every function here reads only ``word/start/end`` from the source tokens — text is never taken
from the model — so word and timestamp preservation is guaranteed by construction. ``displayWord``
carries capitalization only (punctuation is a separate field, per the spec's example); the readable
string re-attaches ``punctuationAfter``.
"""
from __future__ import annotations

from typing import Optional

from .types import Annotation, PunctuatedWord, TimedWord, TranscriptSentence, TERMINALS


def clean_core(word: str) -> str:
    """The display core of a raw token: no surrounding whitespace, no pre-existing edge punctuation.
    The underlying ``word`` field is kept verbatim elsewhere — this only shapes ``displayWord``."""
    return word.strip().strip(".,?!;:\"'“”‘’ ")


def capitalize_core(core: str) -> str:
    for i, ch in enumerate(core):
        if ch.isalpha():
            return core[:i] + ch.upper() + core[i + 1:]
        if ch.isdigit():
            return core
    return core


def sentence_spans(order: list[int], dense: dict[int, Annotation]) -> list[tuple[int, int]]:
    """Derive sentence spans deterministically: a sentence ends at a token whose annotation has
    ``sentenceEnd`` or a terminal ``punctuationAfter``; the next token starts a new one."""
    spans: list[tuple[int, int]] = []
    cur: Optional[int] = None
    for gi in order:
        if cur is None:
            cur = gi
        a = dense[gi]
        if a.sentenceEnd or a.punctuationAfter in TERMINALS:
            spans.append((cur, gi))
            cur = None
    if cur is not None:
        spans.append((cur, order[-1]))
    return spans


def build_punctuated_words(words: list[TimedWord],
                           dense: dict[int, Annotation]) -> list[PunctuatedWord]:
    out: list[PunctuatedWord] = []
    for gi, w in enumerate(words):
        a = dense[gi]
        core = clean_core(w.word)
        display = capitalize_core(core) if (a.capitalize and core) else core
        out.append(PunctuatedWord(
            id=w.id, word=w.word, displayWord=display, start=w.start, end=w.end,
            speaker=w.speaker, confidence=w.confidence,
            capitalize=a.capitalize, punctuationAfter=a.punctuationAfter,
            sentenceStart=a.sentenceStart, sentenceEnd=a.sentenceEnd,
            paragraphStart=a.paragraphStart, punctuationConfidence=a.confidence))
    return out


def _span_text(words: list[TimedWord], dense: dict[int, Annotation], s: int, e: int) -> str:
    parts: list[str] = []
    for gi in range(s, e + 1):
        a = dense[gi]
        core = clean_core(words[gi].word)
        if not core:
            continue
        if a.capitalize:
            core = capitalize_core(core)
        parts.append(core + a.punctuationAfter)
    return " ".join(parts)


def build_sentences(words: list[TimedWord], dense: dict[int, Annotation]) -> list[TranscriptSentence]:
    order = list(range(len(words)))
    sentences: list[TranscriptSentence] = []
    for si, (s, e) in enumerate(sentence_spans(order, dense)):
        confs = [dense[k].confidence for k in range(s, e + 1) if dense[k].confidence is not None]
        sentences.append(TranscriptSentence(
            id=f"sentence_{si}", text=_span_text(words, dense, s, e),
            start=words[s].start, end=words[e].end, speaker=words[s].speaker,
            tokenIds=[words[k].id for k in range(s, e + 1)],
            confidence=(min(confs) if confs else None),
            paragraphStart=dense[s].paragraphStart))
    return sentences


def render_readable(sentences: list[TranscriptSentence]) -> str:
    """Join sentences with spaces; a ``paragraphStart`` sentence begins a new paragraph."""
    parts: list[str] = []
    for i, s in enumerate(sentences):
        if i > 0:
            parts.append("\n\n" if s.paragraphStart else " ")
        parts.append(s.text)
    return "".join(parts).strip()
