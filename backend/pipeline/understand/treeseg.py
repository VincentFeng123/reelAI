"""Deterministic embedding-based divisive topic segmentation (TreeSeg, arXiv:2407.12028).

Boundaries come 100% from sentence embeddings: bisecting splits that maximize the Ward-style
between-segment scatter gain, plus a small deterministic pause/discourse prior at candidate
boundaries. One tree yields both the topic cut and (via the earliest splits) the chapter cut.
No LLM, no randomness — same input, same output. Labeling happens later in content_map.
"""
from __future__ import annotations

import heapq

import numpy as np

from ... import config
from ..sentences import Sentence
from .segment import discourse_hits, gap_before

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(config.BI_ENCODER, device=config.TORCH_DEVICE)
    return _model


def embed_sentences(sentences: list[Sentence]) -> np.ndarray:
    """L2-normalized float32 sentence embeddings (all-MiniLM-L6-v2, locally cached).
    Raises on model-load/encode failure — the caller falls back to the legacy engine."""
    texts = [(s.text or "") for s in sentences]
    emb = _get_model().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(emb, dtype=np.float32)


def boundary_priors(sentences: list[Sentence], weight: float) -> np.ndarray:
    """Prior bonus for cutting BEFORE sentence k (shape n+1): long pauses and discourse
    markers ("so", "next", …) make a boundary slightly more attractive. Deterministic."""
    n = len(sentences)
    pr = np.zeros(n + 1, dtype=np.float64)
    if weight <= 0.0 or n < 2:
        return pr
    hits = discourse_hits(sentences)
    gaps = np.array([gap_before(sentences, i) for i in range(n)], dtype=np.float64)
    gmax = float(gaps.max()) or 1.0
    for i in range(1, n):
        pr[i] = weight * (min(gaps[i] / gmax, 1.0) + (1.0 if i in hits else 0.0)) / 2.0
    return pr


def divisive_segments(emb: np.ndarray, *, target_k: int, min_size: int,
                      coherence_floor: float = 0.0,
                      priors: np.ndarray | None = None) -> tuple[list[tuple[int, int]], list[int]]:
    """Bisecting segmentation of [0, n-1] into ≤ target_k contiguous segments.

    A cut at k splits [a,b] into [a,k-1],[k,b]; its gain is the between-segment scatter
    ‖S_L‖²/n_L + ‖S_R‖²/n_R − ‖S‖²/n (O(1) per candidate via prefix sums) + priors[k].
    Highest-gain span splits first; stops at target_k or when the best gain < coherence_floor.
    Ties break to the earliest index. Returns (sorted segments, boundaries in split order)."""
    n = int(emb.shape[0])
    if n == 0:
        return [], []
    if target_k <= 1 or n < 2 * min_size:
        return [(0, n - 1)], []
    prefix = np.zeros((n + 1, emb.shape[1]), dtype=np.float64)
    np.cumsum(emb, axis=0, out=prefix[1:])
    pri = priors if priors is not None else np.zeros(n + 1, dtype=np.float64)

    def best_cut(a: int, b: int):
        total = prefix[b + 1] - prefix[a]
        base = float(total @ total) / (b - a + 1)
        best_gain = best_k = None
        for k in range(a + min_size, b - min_size + 2):
            left = prefix[k] - prefix[a]
            right = total - left
            gain = (float(left @ left) / (k - a) + float(right @ right) / (b - k + 1)
                    - base + float(pri[k]))
            if best_gain is None or gain > best_gain + 1e-12:   # strict → earliest k wins ties
                best_gain, best_k = gain, k
        return (best_gain, best_k) if best_k is not None else None

    segments = [(0, n - 1)]
    split_order: list[int] = []
    heap: list[tuple[float, int, int, int]] = []
    first = best_cut(0, n - 1)
    if first:
        heapq.heappush(heap, (-first[0], 0, n - 1, first[1]))
    while heap and len(segments) < target_k:
        neg_gain, a, b, k = heapq.heappop(heap)
        if -neg_gain < coherence_floor:
            break
        segments.remove((a, b))
        segments.extend([(a, k - 1), (k, b)])
        split_order.append(k)
        for x, y in ((a, k - 1), (k, b)):
            c = best_cut(x, y)
            if c:
                heapq.heappush(heap, (-c[0], x, y, c[1]))
    segments.sort()
    return segments, split_order


def chapter_cut(split_order: list[int], segments: list[tuple[int, int]],
                max_per_chapter: int) -> list[tuple[int, int]]:
    """Chapter grouping from the SAME tree: the earliest splits are the coarsest seams.
    Returns inclusive (first_topic_idx, last_topic_idx) ranges covering all topics."""
    n_topics = len(segments)
    if n_topics == 0:
        return []
    n_chapters = max(1, round(n_topics / max_per_chapter))
    if n_chapters >= n_topics:
        return [(i, i) for i in range(n_topics)]
    bounds = sorted(split_order[: n_chapters - 1])
    starts = [s0 for s0, _ in segments]
    ranges, t0 = [], 0
    for b in bounds:
        t = starts.index(b)              # splits nest → every early boundary survives in the cut
        ranges.append((t0, t - 1))
        t0 = t
    ranges.append((t0, n_topics - 1))
    return ranges
