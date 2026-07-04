"""Deterministic pause/segment-based annotation — the safe degraded path.

Adds sentence boundaries at ASR segment starts (and, in strong-timing mode, at long pauses) and
capitalizes sentence starts. It invents no text and never touches word ids or timestamps; the result
is a valid, if coarse, punctuation that keeps the pipeline running when the LLM can't produce a
clean result.
"""
from __future__ import annotations

from .types import Annotation, TimedWord, TranscriptChunk


def _annotate(ids: list[int], words: list[TimedWord], seg_bounds: set[int], gaps: list[float],
              weak_timing: bool, sentence_gap_s: float) -> dict[int, Annotation]:
    dense = {gi: Annotation(confidence=0.3) for gi in ids}
    for pos, gi in enumerate(ids):
        if pos == 0:
            continue
        boundary = (gi in seg_bounds) if weak_timing else (gi in seg_bounds or gaps[gi] >= sentence_gap_s)
        if boundary:
            prev = ids[pos - 1]
            dense[prev].punctuationAfter = "."
            dense[prev].sentenceEnd = True
            dense[gi].sentenceStart = True
            dense[gi].capitalize = True
    first = ids[0]
    dense[first].sentenceStart = True
    dense[first].capitalize = True
    return dense


def pause_based_annotations(chunk: TranscriptChunk, words: list[TimedWord], seg_bounds: set[int],
                            gaps: list[float], weak_timing: bool,
                            sentence_gap_s: float = 0.7) -> dict[int, Annotation]:
    return _annotate(chunk.token_ids, words, seg_bounds, gaps, weak_timing, sentence_gap_s)


def pause_based_full(words: list[TimedWord], seg_bounds: set[int], gaps: list[float],
                     weak_timing: bool, sentence_gap_s: float = 0.7) -> dict[int, Annotation]:
    return _annotate(list(range(len(words))), words, seg_bounds, gaps, weak_timing, sentence_gap_s)
