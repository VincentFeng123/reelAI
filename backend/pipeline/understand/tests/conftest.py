"""Offline fixtures: fabricated Sentences + block-structured fake embeddings (no model, no LLM)."""
from __future__ import annotations

import numpy as np

from backend.pipeline.sentences import Sentence


def make_sents(n: int, *, sec: float = 2.0, gap_at: tuple[int, ...] = (), gap: float = 6.0,
               texts: list[str] | None = None) -> list[Sentence]:
    """n sentences, `sec` seconds each; a `gap`-second pause BEFORE each index in gap_at."""
    sents, t = [], 0.0
    for i in range(n):
        if i in gap_at:
            t += gap
        s, e = t, t + sec
        sents.append(Sentence(idx=i, text=(texts[i] if texts else f"sentence number {i}"),
                              start=s, end=e, terminator=".", ends_with_period=True,
                              word_start_idx=i, word_end_idx=i, align_confidence=1.0))
        t = e + 0.1
    return sents


def block_emb(sizes: list[int], dim: int = 8) -> np.ndarray:
    """Unit vectors; block j points along axis j%dim → maximal between-block scatter at
    block boundaries. Deterministic by construction."""
    rows = []
    for j, sz in enumerate(sizes):
        v = np.zeros(dim)
        v[j % dim] = 1.0
        rows += [v] * sz
    return np.asarray(rows, dtype=np.float64)
