"""Transcript chunking + boundary hints for the content-map pass.

Long videos exceed a single LLM context, so we split the sentence-indexed transcript
into chunks at natural seams (long pauses between sentences, discourse markers like
"so", "next", "let's"). Pure-Python and dependency-free — the sentence-transformers
embedding signal is a Phase-2 upgrade layered on top of these hints.
"""
from __future__ import annotations

import re

from ... import config
from ..sentences import Sentence

# Discourse markers that commonly open a new topic/segment.
_DISCOURSE = re.compile(
    r"^\s*(so|now|next|okay|ok|alright|right|first|second|third|finally|"
    r"let'?s|moving on|another|the next|to sum up|in summary|to recap|recap|"
    r"for example|for instance|consider|suppose|imagine|today|in this (video|lecture|section))\b",
    re.IGNORECASE,
)


def discourse_hits(sentences: list[Sentence]) -> set[int]:
    return {s.idx for s in sentences if _DISCOURSE.match(s.text or "")}


def gap_before(sentences: list[Sentence], i: int) -> float:
    if i <= 0 or i >= len(sentences):
        return 0.0
    return max(0.0, float(sentences[i].start) - float(sentences[i - 1].end))


def chunk_sentences(
    sentences: list[Sentence],
    max_per_chunk: int | None = None,
) -> list[tuple[int, int]]:
    """Split sentence indices into ``[(i0, i1), …]`` chunks of ≤ max_per_chunk, breaking
    at the strongest natural seam near each cut so a topic isn't split mid-thought."""
    n = len(sentences)
    cap = max_per_chunk or config.CONTENT_MAP_MAX_SENTS_PER_CALL
    if n == 0:
        return []
    if n <= cap:
        return [(0, n - 1)]

    hits = discourse_hits(sentences)
    chunks: list[tuple[int, int]] = []
    start = 0
    while start < n:
        hard_end = min(start + cap - 1, n - 1)
        if hard_end == n - 1:
            chunks.append((start, hard_end))
            break
        # search a window near hard_end for the best seam (discourse marker, else max gap)
        lo = max(start + cap // 2, hard_end - cap // 3)
        best_i, best_score = hard_end, -1.0
        for i in range(lo, hard_end + 1):
            score = gap_before(sentences, i) + (2.0 if i in hits else 0.0)
            if score > best_score:
                best_i, best_score = i, score
        cut = best_i if best_i > start else hard_end
        chunks.append((start, cut - 1 if cut > start else cut))
        start = cut
    return chunks
