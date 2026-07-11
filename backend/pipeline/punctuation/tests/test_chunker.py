"""Chunker unit tests — coverage, identical shared bands, determinism, tail guard."""
from __future__ import annotations

from backend.pipeline.punctuation.chunker import make_chunks
from backend.pipeline.punctuation.types import TimedWord


def _timed(n: int) -> list[TimedWord]:
    return [TimedWord(id=f"w{i}", word=f"tok{i}", start=i * 0.3, end=i * 0.3 + 0.25)
            for i in range(n)]


def test_single_chunk_when_short():
    chunks = make_chunks(_timed(50), 500, 60, 700)
    assert len(chunks) == 1
    assert chunks[0].primary_token_ids == list(range(50))
    assert chunks[0].left_overlap_token_ids == []
    assert chunks[0].right_overlap_token_ids == []


def test_multi_chunk_coverage_bands_and_size():
    n = 1800
    chunks = make_chunks(_timed(n), 500, 60, 700)
    assert len(chunks) >= 2

    # every chunk is a contiguous id range and within the max size
    for ch in chunks:
        assert ch.token_ids == list(range(ch.token_ids[0], ch.token_ids[-1] + 1))
        assert len(ch.token_ids) <= 700

    # adjacent shared bands are byte-identical (right_overlap(m) == left_overlap(m+1))
    for m in range(len(chunks) - 1):
        assert chunks[m].right_overlap_token_ids == chunks[m + 1].left_overlap_token_ids
        assert chunks[m].right_overlap_token_ids  # non-empty interior band

    # union of token_ids covers the whole document
    covered = set()
    for ch in chunks:
        covered.update(ch.token_ids)
    assert covered == set(range(n))

    # primaries are disjoint; band tokens belong to exactly two chunks
    prim_count = {}
    tok_count = {}
    for ch in chunks:
        for gi in ch.primary_token_ids:
            prim_count[gi] = prim_count.get(gi, 0) + 1
        for gi in ch.token_ids:
            tok_count[gi] = tok_count.get(gi, 0) + 1
    assert all(v == 1 for v in prim_count.values())
    assert set(tok_count.values()) <= {1, 2}


def test_deterministic():
    a = make_chunks(_timed(1500), 500, 60, 700)
    b = make_chunks(_timed(1500), 500, 60, 700)
    assert [c.token_ids for c in a] == [c.token_ids for c in b]


def test_tiny_tail_guard():
    # n just over max would leave a sliver last chunk without the guard
    chunks = make_chunks(_timed(720), 500, 60, 700, min_words=300)
    assert len(chunks) == 2
    assert all(len(c.token_ids) >= 300 for c in chunks)


def test_empty():
    assert make_chunks([], 500, 60, 700) == []
