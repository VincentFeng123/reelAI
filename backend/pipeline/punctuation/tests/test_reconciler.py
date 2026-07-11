"""Reconciler unit tests — deterministic overlap resolution + seam de-dup."""
from __future__ import annotations

from backend.pipeline.punctuation.reconciler import reconcile
from backend.pipeline.punctuation.types import Annotation, TimedWord, TranscriptChunk


def _words(n: int) -> list[TimedWord]:
    return [TimedWord(id=f"w{i}", word=f"tok{i}", start=i * 0.3, end=i * 0.3 + 0.25)
            for i in range(n)]


def _two_chunks():
    # c0: [0..59] primary [0..39] right_overlap [40..59]
    # c1: [40..99] primary [60..99] left_overlap [40..59]
    c0 = TranscriptChunk(id="c0", token_ids=list(range(0, 60)),
                         primary_token_ids=list(range(0, 40)),
                         left_overlap_token_ids=[], right_overlap_token_ids=list(range(40, 60)))
    c1 = TranscriptChunk(id="c1", token_ids=list(range(40, 100)),
                         primary_token_ids=list(range(60, 100)),
                         left_overlap_token_ids=list(range(40, 60)), right_overlap_token_ids=[])
    return [c0, c1]


def _ann(chunk):
    return {gi: Annotation() for gi in chunk.token_ids}


def test_farther_from_boundary_wins_on_overlap():
    words = _words(100)
    chunks = _two_chunks()
    a0, a1 = _ann(chunks[0]), _ann(chunks[1])
    # gi=45 is in both overlaps; c0 is farther from its nearest edge (dist 14) than c1 (dist 5)
    a0[45] = Annotation(punctuationAfter=".", sentenceEnd=True)
    a1[45] = Annotation(punctuationAfter=",")
    merged, conflicts = reconcile(chunks, {"c0": a0, "c1": a1}, words, [0.0] * 100)
    assert merged[45].punctuationAfter == "."
    assert any(c.token == "w45" for c in conflicts)


def test_agreement_keeps_shared_no_conflict():
    words = _words(100)
    chunks = _two_chunks()
    a0, a1 = _ann(chunks[0]), _ann(chunks[1])
    a0[50] = Annotation(punctuationAfter=",")
    a1[50] = Annotation(punctuationAfter=",")
    merged, conflicts = reconcile(chunks, {"c0": a0, "c1": a1}, words, [0.0] * 100)
    assert merged[50].punctuationAfter == ","
    assert not any(c.token == "w50" for c in conflicts)


def test_seam_dedup_terminal_then_comma():
    words = _words(100)
    chunks = _two_chunks()
    a0, a1 = _ann(chunks[0]), _ann(chunks[1])
    for a in (a0, a1):
        a[45] = Annotation(punctuationAfter=".", sentenceEnd=True)
        a[46] = Annotation(punctuationAfter=",")
    merged, conflicts = reconcile(chunks, {"c0": a0, "c1": a1}, words, [0.0] * 100)
    assert merged[45].punctuationAfter == "."
    assert merged[46].punctuationAfter == ""      # contradictory comma cleared
    assert any(c.reason == "term_then_comma" for c in conflicts)


def test_deterministic():
    words = _words(100)
    chunks = _two_chunks()
    a0, a1 = _ann(chunks[0]), _ann(chunks[1])
    a0[45] = Annotation(punctuationAfter=".", confidence=0.9)
    a1[45] = Annotation(punctuationAfter=",", confidence=0.1)
    m1, _ = reconcile(chunks, {"c0": dict(a0), "c1": dict(a1)}, words, [0.0] * 100)
    m2, _ = reconcile(chunks, {"c0": dict(a0), "c1": dict(a1)}, words, [0.0] * 100)
    assert m1[45].punctuationAfter == m2[45].punctuationAfter
