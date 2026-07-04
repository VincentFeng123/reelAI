"""divisive_segments / boundary_priors / chapter_cut — pure, deterministic, offline."""
from __future__ import annotations

import numpy as np

from backend.pipeline.understand.treeseg import boundary_priors, chapter_cut, divisive_segments

from .conftest import block_emb, make_sents


def _coverage_ok(segments, n):
    assert segments == sorted(segments)
    assert segments[0][0] == 0 and segments[-1][1] == n - 1
    for (a0, a1), (b0, b1) in zip(segments, segments[1:]):
        assert b0 == a1 + 1


def test_two_blocks_cut_on_boundary():
    emb = block_emb([5, 5])
    segs, order = divisive_segments(emb, target_k=2, min_size=2)
    assert segs == [(0, 4), (5, 9)]
    assert order == [5]


def test_three_blocks_found():
    segs, _ = divisive_segments(block_emb([4, 4, 4]), target_k=3, min_size=2)
    assert segs == [(0, 3), (4, 7), (8, 11)]


def test_target_k_stops_splitting():
    segs, _ = divisive_segments(block_emb([3, 3, 3, 3]), target_k=2, min_size=2)
    assert len(segs) == 2


def test_uniform_never_splits_past_floor():
    emb = np.tile(np.array([1.0, 0, 0, 0]), (12, 1))          # identical vectors → gain 0
    segs, order = divisive_segments(emb, target_k=4, min_size=2, coherence_floor=1e-6)
    assert segs == [(0, 11)] and order == []


def test_min_size_respected():
    segs, _ = divisive_segments(block_emb([2, 10]), target_k=4, min_size=3)
    assert all(b - a + 1 >= 3 for a, b in segs)


def test_coverage_and_determinism():
    emb = block_emb([5, 4, 6, 5])
    r1 = divisive_segments(emb, target_k=4, min_size=2)
    r2 = divisive_segments(emb, target_k=4, min_size=2)
    assert r1 == r2
    _coverage_ok(r1[0], 20)


def test_degenerate_inputs():
    assert divisive_segments(block_emb([3]), target_k=1, min_size=2) == ([(0, 2)], [])
    assert divisive_segments(block_emb([2]), target_k=4, min_size=2) == ([(0, 1)], [])  # n < 2*min
    assert divisive_segments(np.zeros((0, 4)), target_k=2, min_size=2) == ([], [])


def test_prior_places_cut_on_uniform_embeddings():
    emb = np.tile(np.array([1.0, 0, 0, 0]), (12, 1))          # no embedding signal at all
    pr = np.zeros(13)
    pr[7] = 0.5
    segs, order = divisive_segments(emb, target_k=2, min_size=2, priors=pr)
    assert order == [7] and segs == [(0, 6), (7, 11)]


def test_boundary_priors_gap_and_discourse():
    sents = make_sents(10, gap_at=(5,), texts=[
        "alpha one", "alpha two", "alpha three", "alpha four", "alpha five",
        "beta one", "beta two", "So let us move on", "beta four", "beta five"])
    pr = boundary_priors(sents, weight=0.15)
    assert pr.shape == (11,)
    assert pr[5] > 0 and pr[7] > 0                            # pause seam + discourse marker
    assert pr[5] > pr[2] and pr[7] > pr[2]                    # plain boundary scores lower
    assert boundary_priors(sents, weight=0.0).sum() == 0.0


def test_chapter_cut_groups_by_earliest_splits():
    segs = [(0, 3), (4, 7), (8, 11), (12, 15), (16, 19), (20, 23)]
    order = [12, 4, 8, 16, 20]                                # first split at 12 = coarsest seam
    ranges = chapter_cut(order, segs, max_per_chapter=3)      # round(6/3)=2 chapters
    assert ranges == [(0, 2), (3, 5)]                         # boundary 12 = topic index 3


def test_chapter_cut_small_counts():
    segs = [(0, 5), (6, 11)]
    assert chapter_cut([6], segs, max_per_chapter=5) == [(0, 1)]   # round(2/5)=0 → 1 chapter
    assert chapter_cut([], [(0, 9)], max_per_chapter=5) == [(0, 0)]


def test_chapter_cut_multiple_boundaries():
    segs = [(0, 3), (4, 7), (8, 11), (12, 15), (16, 19), (20, 23), (24, 27), (28, 31)]
    order = [16, 8, 24, 4, 12, 20, 28]                    # earliest splits: 16, then 8, then 24
    ranges = chapter_cut(order, segs, max_per_chapter=3)  # round(8/3)=3 chapters → bounds {8, 16}
    assert ranges == [(0, 1), (2, 3), (4, 7)]             # topic idx of starts 8→2, 16→4
    assert [segs[a][0] for a, _ in ranges] == [0, 8, 16]  # chapter starts on the split boundaries
