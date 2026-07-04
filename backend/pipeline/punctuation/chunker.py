"""Split a long timed transcript into overlapping chunks with clean, reconcilable seams.

Each interior cut owns ONE symmetric overlap band of half-width ``H``; that band is chunk m's
``right_overlap`` and chunk m+1's ``left_overlap`` — the same integer ids in both, so a band token
is predicted by exactly the two adjacent chunks and reconciliation is a 2-way decision.

Cut *centers* are chosen near the word-target, preferring ASR segment boundaries / speaker changes /
long pauses, while avoiding isolating a tiny sliver against the seam. All iteration is over integer
ranges so the output is fully deterministic (no dict/set ordering).
"""
from __future__ import annotations

import bisect

from .types import TimedWord, TranscriptChunk


def _pause_gaps(words: list[TimedWord]) -> list[float]:
    gaps = [0.0] * len(words)
    for i in range(1, len(words)):
        gaps[i] = max(0.0, words[i].start - words[i - 1].end)
    return gaps


def _robust_norm(gaps: list[float]) -> list[float]:
    positive = sorted(g for g in gaps if g > 0)
    if not positive:
        return [0.0] * len(gaps)
    p95 = positive[min(len(positive) - 1, int(0.95 * len(positive)))]
    if p95 <= 0:
        return [0.0] * len(gaps)
    return [min(1.0, g / p95) for g in gaps]


def _segment_boundary_indices(words: list[TimedWord], segments) -> set[int]:
    """Word index that begins each ASR segment (nearest-start match with tolerance)."""
    if not segments or not words:
        return set()
    starts = [w.start for w in words]
    bounds: set[int] = set()
    for seg in segments:
        s = float(seg.get("start", 0.0))
        j = bisect.bisect_left(starts, s)
        best, best_d = None, 1e18
        for k in (j - 1, j, j + 1):
            if 0 <= k < len(words):
                d = abs(starts[k] - s)
                if d < best_d:
                    best, best_d = k, d
        if best is not None:
            bounds.add(best)
    bounds.discard(0)  # token 0 always starts the doc; not an interior cut hint
    return bounds


def _speaker_change_at(c: int, words: list[TimedWord]) -> bool:
    if c <= 0 or c >= len(words):
        return False
    a, b = words[c - 1].speaker, words[c].speaker
    return bool(a) and bool(b) and a != b


def _short_phrase_penalty(c: int, boundaries: list[int]) -> float:
    """Penalize a cut whose center is only a few tokens after the nearest boundary."""
    if not boundaries:
        return 0.0
    i = bisect.bisect_right(boundaries, c) - 1
    if i < 0:
        return 0.0
    d = c - boundaries[i]
    if d < 4:
        return 1.0
    if d < 8:
        return 0.5
    return 0.0


def _pick_cut(lo: int, hi: int, ideal: int, seg_bounds: set[int], gnorm: list[float],
              boundaries: list[int], words: list[TimedWord], weak_timing: bool,
              has_speakers: bool) -> int:
    w_seg = 3.0 if weak_timing else 2.0
    w_pause = 0.5 if weak_timing else 2.5
    w_spk = 1.5 if has_speakers else 0.0
    w_dist = 1.0
    w_short = 2.0
    span = max(1, hi - lo)
    best_c: int | None = None
    best_key: tuple | None = None
    for c in range(lo, hi + 1):
        s = 0.0
        s += w_seg * (1.0 if c in seg_bounds else 0.0)
        s += w_pause * gnorm[c]
        s += w_spk * (1.0 if _speaker_change_at(c, words) else 0.0)
        s -= w_dist * abs(c - ideal) / span
        s -= w_short * _short_phrase_penalty(c, boundaries)
        key = (s, -abs(c - ideal), -c)  # higher score → nearer ideal → smaller index
        if best_key is None or key > best_key:
            best_key, best_c = key, c
    return best_c if best_c is not None else lo


def make_chunks(words: list[TimedWord], target_words: int = 500, overlap_words: int = 60,
                max_words: int = 700, *, segments=None, speaker_turns=None,
                weak_timing: bool = False, min_words: int = 300) -> list[TranscriptChunk]:
    n = len(words)
    if n == 0:
        return []
    if n <= max_words:
        ids = list(range(n))
        return [TranscriptChunk(id="c0", token_ids=ids, primary_token_ids=ids,
                                left_overlap_token_ids=[], right_overlap_token_ids=[])]

    h = max(1, min(overlap_words // 2, max_words // 4))
    min_len = max(2 * h + 1, min(min_words, max_words))
    win = min(round(0.25 * target_words), (max_words - 2 * h) // 2)

    seg_bounds = _segment_boundary_indices(words, segments)
    gaps = _pause_gaps(words)
    gnorm = _robust_norm(gaps)
    has_speakers = any(w.speaker for w in words)
    bset = set(seg_bounds)
    if not weak_timing:
        bset.update(i for i in range(1, n) if gaps[i] >= 0.7)
    boundaries = sorted(bset)

    centers: list[int] = []
    start = 0
    while n - start > max_words:
        ideal_c = start + target_words - h
        lo_h = max(start + min_len - h, h)
        hi_h = min(start + max_words - h, n - h - 1)
        if hi_h < lo_h:                          # geometry forced → largest legal step
            c = hi_h
        else:
            lo = max(lo_h, ideal_c - win)
            hi = min(hi_h, ideal_c + win)
            if hi < lo:                          # window outside feasibility → search full range
                lo, hi = lo_h, hi_h
            c = _pick_cut(lo, hi, ideal_c, seg_bounds, gnorm, boundaries, words,
                          weak_timing, has_speakers)
        if c - h <= start:                       # never regress → force progress
            c = min(start + max_words - h, n - h - 1)
        centers.append(c)
        start = c - h

    if centers:                                  # tail guard: avoid a sliver final chunk
        last_start = centers[-1] - h
        if n - last_start < min_len:
            prev_start = (centers[-2] - h) if len(centers) >= 2 else 0
            if (n - prev_start) <= max_words:
                centers.pop()                    # merge tail into previous chunk
            else:
                new_c = n - min_len + h
                if new_c - h > prev_start and (new_c + h - prev_start) <= max_words:
                    centers[-1] = new_c

    return _assemble_chunks(n, centers, h)


def _assemble_chunks(n: int, centers: list[int], h: int) -> list[TranscriptChunk]:
    bands = [(max(0, c - h), min(n - 1, c + h - 1)) for c in centers]
    chunks: list[TranscriptChunk] = []
    for m in range(len(centers) + 1):
        left = bands[m - 1] if m > 0 else None
        right = bands[m] if m < len(centers) else None
        a = left[0] if left else 0
        b = right[1] if right else n - 1
        left_ov = list(range(left[0], left[1] + 1)) if left else []
        right_ov = list(range(right[0], right[1] + 1)) if right else []
        p0 = (left[1] + 1) if left else 0
        p1 = (right[0] - 1) if right else n - 1
        primary = list(range(p0, p1 + 1)) if p1 >= p0 else []
        chunks.append(TranscriptChunk(
            id=f"c{m}", token_ids=list(range(a, b + 1)),
            primary_token_ids=primary, left_overlap_token_ids=left_ov,
            right_overlap_token_ids=right_ov))
    return chunks
