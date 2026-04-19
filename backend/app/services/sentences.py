"""
Sentence boundary detection for transcript cues (Phase A.2).

This module turns a list of `IngestTranscriptCue`s (each carrying word-level
timestamps, from Phase A.1) into a list of `SentenceSpan`s — the canonical unit
every clip-boundary decision downstream resolves to (`find_topic_start`,
`find_topic_end`, max-duration truncation, etc.).

Uses `pysbd` (Segment Any Boundary Disambiguator — ~50 KB, pure Python) when
available. Falls back to a reasonably-robust regex splitter that handles common
abbreviations (Dr., Mr., Mrs., e.g., i.e., U.S., etc.), decimal numbers, and
ellipses, so the pipeline doesn't hard-fail if pysbd is unavailable in a
serverless/vercel environment that skipped the extra dep.

A `SentenceSpan` carries:
  - text: joined sentence text (from cues)
  - t_start, t_end: tight sentence-aligned timestamps derived from the first /
    last matching word in the underlying cue words
  - cue_start_idx, cue_end_idx: inclusive indices into the input cue list
  - word_start_idx, word_end_idx: inclusive indices into a flat "all words"
    array (lexicographic concatenation of cue words), so the engine can snap
    a boundary to an exact word position
  - terminal_punct: the terminal punctuation char of the sentence
    ('.', '!', '?', '…') or empty if the sentence was truncated mid-stream
  - confidence: 0.0-1.0 — higher when a word-level timestamp anchors the end
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Iterable

from ..ingestion.models import IngestTranscriptCue, IngestTranscriptWord

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Public data class
# --------------------------------------------------------------------------- #


@dataclass
class SentenceSpan:
    text: str
    t_start: float
    t_end: float
    cue_start_idx: int
    cue_end_idx: int
    word_start_idx: int
    word_end_idx: int
    terminal_punct: str
    confidence: float = 0.5


@dataclass
class _FlatWord:
    """A word with its containing cue index, used for back-reference."""

    text: str
    start: float
    end: float
    cue_idx: int
    global_idx: int
    sentence_idx: int = -1


# --------------------------------------------------------------------------- #
# pysbd lazy loader
# --------------------------------------------------------------------------- #


_pysbd_segmenter = None
_pysbd_tried = False


def _get_pysbd_segmenter(language: str = "en"):
    global _pysbd_segmenter, _pysbd_tried
    if _pysbd_tried:
        return _pysbd_segmenter
    _pysbd_tried = True
    try:
        import pysbd  # type: ignore
    except Exception:
        logger.info("pysbd not installed; using fallback regex sentence splitter")
        _pysbd_segmenter = None
        return None
    try:
        _pysbd_segmenter = pysbd.Segmenter(language=language, clean=False, char_span=True)
    except Exception:
        logger.exception("pysbd segmenter init failed; using fallback regex splitter")
        _pysbd_segmenter = None
    return _pysbd_segmenter


# --------------------------------------------------------------------------- #
# Fallback regex splitter (used when pysbd unavailable)
# --------------------------------------------------------------------------- #


_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "sr", "jr", "st", "prof", "rev", "hon", "gen",
    "col", "lt", "capt", "cmdr", "sgt", "cpl", "pvt", "pres", "gov",
    "inc", "ltd", "co", "corp", "llc", "llp",
    "vs", "etc", "eg", "ie", "al", "viz", "ca", "approx", "est",
    "u.s", "u.k", "u.n", "e.u", "a.m", "p.m",
}


_SENTENCE_TERMINATOR_RE = re.compile(
    r"""
    (                                  # sentence terminator group
        [.!?…]+                        # one-or-more terminal punct
        ['"\)\]]*                      # optional closing quote/bracket
    )
    (?=\s+[A-Z0-9(]|\s*$)              # followed by whitespace+capital OR end
    """,
    re.VERBOSE,
)


def _is_abbreviation(text: str, pos: int) -> bool:
    """Heuristic: a period preceded by a single letter or known abbrev is not a sentence end."""
    if pos <= 0:
        return False
    # Back-scan up to 8 chars for the preceding token.
    start = max(0, pos - 16)
    preceding = text[start:pos].rstrip(".!?…)\"'] \t")
    # Last token: keep letters, digits, dots.
    tail = re.search(r"([A-Za-z0-9.]+)$", preceding)
    if not tail:
        return False
    token = tail.group(1).lower().rstrip(".")
    if not token:
        return False
    # Single-letter with period (e.g., "A.") or multi-letter known abbrev.
    if len(token) == 1 and token.isalpha():
        return True
    if token in _ABBREVIATIONS:
        return True
    # Decimal numbers: digit.digit (e.g., "3.14") — preceding char is digit, next is digit
    if token.isdigit() and pos + 1 < len(text) and text[pos + 1].isdigit():
        return True
    return False


def _regex_split_sentences(text: str) -> list[tuple[int, int, str]]:
    """
    Return list of (char_start, char_end, terminal_punct) tuples covering the text.
    char_end is inclusive of the terminal punct. Any trailing text with no terminator
    is returned as a final span with terminal_punct=''.
    """
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    n = len(text)
    for m in _SENTENCE_TERMINATOR_RE.finditer(text):
        punct = m.group(1)
        # Locate the actual last sentence-terminator char (handles ellipsis "..." and "!?!").
        end = m.end()
        # Skip false positives inside abbreviations.
        if _is_abbreviation(text, m.start()):
            continue
        if end > cursor:
            terminal = punct[-1] if punct else ""
            # Translate "..." into "…" for consistency.
            if terminal == "." and punct.endswith("..."):
                terminal = "…"
            spans.append((cursor, end, terminal))
            cursor = end
    if cursor < n and text[cursor:].strip():
        # Tail with no terminal punct.
        spans.append((cursor, n, ""))
    return spans


# --------------------------------------------------------------------------- #
# Flatten cues → words
# --------------------------------------------------------------------------- #


def _flatten_words(cues: list[IngestTranscriptCue]) -> list[_FlatWord]:
    out: list[_FlatWord] = []
    global_idx = 0
    for cue_idx, cue in enumerate(cues):
        for w in cue.words:
            out.append(
                _FlatWord(
                    text=w.text,
                    start=float(w.start),
                    end=float(w.end),
                    cue_idx=cue_idx,
                    global_idx=global_idx,
                )
            )
            global_idx += 1
        if not cue.words:
            # Last-resort proportional placeholder so the flat array is never empty when cues have text.
            text = (cue.text or "").strip()
            if text:
                out.append(
                    _FlatWord(
                        text=text,
                        start=float(cue.start),
                        end=float(cue.end),
                        cue_idx=cue_idx,
                        global_idx=global_idx,
                    )
                )
                global_idx += 1
    return out


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def split_sentences(
    cues: list[IngestTranscriptCue],
    *,
    language: str = "en",
) -> list[SentenceSpan]:
    """
    Split the concatenated cue text into sentences, mapping each sentence back
    to precise (t_start, t_end) via word-level timing when available, else via
    proportional-character fallback on cue timing.
    """
    if not cues:
        return []

    flat_words = _flatten_words(cues)
    if not flat_words:
        return []

    # Build a continuous string and a char→word index map so we can slice the
    # text through the sentence segmenter, then resolve back to word positions.
    buf_parts: list[str] = []
    word_char_ranges: list[tuple[int, int]] = []  # per word: (char_start, char_end) in buf
    char_cursor = 0
    for fw in flat_words:
        token = fw.text
        if not token:
            continue
        if buf_parts:
            buf_parts.append(" ")
            char_cursor += 1
        start_char = char_cursor
        buf_parts.append(token)
        char_cursor += len(token)
        end_char = char_cursor
        word_char_ranges.append((start_char, end_char))
    full_text = "".join(buf_parts)

    if not full_text.strip():
        return []

    # Sentence-split the concatenated text.
    segmenter = _get_pysbd_segmenter(language=language)
    raw_spans: list[tuple[int, int, str]] = []
    if segmenter is not None:
        try:
            results = segmenter.segment(full_text)
        except Exception:
            logger.exception("pysbd segmentation raised; falling back to regex")
            results = []
        for r in results:
            # pysbd with char_span=True returns `TextSpan` objects with `.start` and `.end`.
            try:
                s = int(getattr(r, "start", None) if not isinstance(r, dict) else r.get("start"))
                e = int(getattr(r, "end", None) if not isinstance(r, dict) else r.get("end"))
                sent_text = full_text[s:e]
            except Exception:
                continue
            if s >= e or not sent_text.strip():
                continue
            # Derive terminal punct from the last non-space char of the span.
            stripped = sent_text.rstrip()
            terminal = stripped[-1] if stripped and stripped[-1] in {".", "!", "?", "…"} else ""
            # Map "..." to "…" for consistency.
            if terminal == "." and stripped.endswith("..."):
                terminal = "…"
            raw_spans.append((s, e, terminal))
    if not raw_spans:
        raw_spans = _regex_split_sentences(full_text)

    # Map each sentence's (char_start, char_end) to word indices.
    def _word_for_char(char_pos: int, *, side: str) -> int:
        """Return the word index whose char range contains char_pos (approx)."""
        if not word_char_ranges:
            return 0
        if side == "start":
            # First word whose end > char_pos.
            for i, (_s, e) in enumerate(word_char_ranges):
                if e > char_pos:
                    return i
            return len(word_char_ranges) - 1
        # side == "end" — last word whose start < char_pos.
        best = len(word_char_ranges) - 1
        for i, (s, _e) in enumerate(word_char_ranges):
            if s >= char_pos:
                return max(0, i - 1)
            best = i
        return best

    spans: list[SentenceSpan] = []
    for (s_char, e_char, terminal) in raw_spans:
        sent_text = full_text[s_char:e_char].strip()
        if not sent_text:
            continue
        w_start_idx = _word_for_char(s_char, side="start")
        w_end_idx = _word_for_char(e_char, side="end")
        if w_end_idx < w_start_idx:
            w_end_idx = w_start_idx
        first_word = flat_words[w_start_idx]
        last_word = flat_words[w_end_idx]
        t_start = first_word.start
        t_end = max(last_word.end, t_start + 0.05)
        # Confidence: higher when underlying cues have non-legacy word_source.
        cue_sources = {cues[flat_words[i].cue_idx].word_source for i in range(w_start_idx, w_end_idx + 1)}
        if "whisperx" in cue_sources:
            confidence = 0.95 if terminal else 0.85
        elif "whisper" in cue_sources or "openai" in cue_sources:
            confidence = 0.9 if terminal else 0.75
        elif "whisper_aligned" in cue_sources:
            confidence = 0.88 if terminal else 0.78
        elif "proportional" in cue_sources:
            confidence = 0.6 if terminal else 0.45
        else:  # legacy only
            confidence = 0.4 if terminal else 0.3
        spans.append(
            SentenceSpan(
                text=sent_text,
                t_start=t_start,
                t_end=t_end,
                cue_start_idx=first_word.cue_idx,
                cue_end_idx=last_word.cue_idx,
                word_start_idx=w_start_idx,
                word_end_idx=w_end_idx,
                terminal_punct=terminal,
                confidence=confidence,
            )
        )
    return spans


def sentences_in_range(
    sentences: list[SentenceSpan],
    t0: float,
    t1: float,
) -> list[SentenceSpan]:
    """Return sentences whose [t_start, t_end] overlap [t0, t1]."""
    out: list[SentenceSpan] = []
    for s in sentences:
        if s.t_end < t0:
            continue
        if s.t_start > t1:
            break
        out.append(s)
    return out


def latest_sentence_ending_before(
    sentences: list[SentenceSpan],
    t_cap: float,
    *,
    require_terminal: bool = True,
) -> SentenceSpan | None:
    """
    Return the latest SentenceSpan whose t_end <= t_cap. When `require_terminal`
    is True, skip sentences that don't end on '.', '!', '?', or '…'. If no
    qualifying sentence is found before t_cap, returns None.
    """
    chosen: SentenceSpan | None = None
    for s in sentences:
        if s.t_end > t_cap:
            break
        if require_terminal and s.terminal_punct not in {".", "!", "?", "…"}:
            continue
        chosen = s
    return chosen


def first_sentence_at_or_after(
    sentences: list[SentenceSpan],
    t_floor: float,
) -> SentenceSpan | None:
    """Return the first SentenceSpan whose t_start >= t_floor."""
    for s in sentences:
        if s.t_start >= t_floor:
            return s
    return None


__all__ = [
    "SentenceSpan",
    "split_sentences",
    "sentences_in_range",
    "latest_sentence_ending_before",
    "first_sentence_at_or_after",
]
