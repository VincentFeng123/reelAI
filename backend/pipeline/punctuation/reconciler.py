"""Deterministically merge per-chunk annotations where overlap bands disagree.

A band token is predicted by exactly the two adjacent chunks. When they disagree we pick a winner by
a fully-ordered priority ladder (primary region > farther from the chunk edge > higher confidence >
stronger corroborating timing > lower chunk order) — every rung a total order, so the result never
depends on dict/set iteration order. A seam-repair pass then removes duplicated punctuation across a
seam, and ``normalize_full`` fixes sentence-boundary flags on the merged whole.
"""
from __future__ import annotations

from collections import defaultdict

from .reconstructor import sentence_spans
from .types import Annotation, Conflict, TERMINALS, TimedWord, TranscriptChunk


def _diff(a: Annotation, b: Annotation) -> str:
    fields = []
    if a.punctuationAfter != b.punctuationAfter:
        fields.append(f"p:{a.punctuationAfter!r}!={b.punctuationAfter!r}")
    if a.sentenceEnd != b.sentenceEnd:
        fields.append(f"se:{a.sentenceEnd}!={b.sentenceEnd}")
    if a.sentenceStart != b.sentenceStart:
        fields.append(f"ss:{a.sentenceStart}!={b.sentenceStart}")
    if a.capitalize != b.capitalize:
        fields.append(f"cap:{a.capitalize}!={b.capitalize}")
    return ",".join(fields)


def _rank(gi: int, ann: Annotation, meta: tuple, gaps: list[float]) -> tuple:
    order, primary_set, first, last = meta
    primary = 1 if gi in primary_set else 0
    dist = min(gi - first, last - gi)
    conf = ann.confidence if ann.confidence is not None else 0.5
    g = gaps[gi + 1] if gi + 1 < len(gaps) else 0.0
    timing = g if ann.punctuationAfter in TERMINALS else -g
    return (primary, dist, conf, timing, -order)  # larger tuple wins; -order breaks final ties


def reconcile(chunks: list[TranscriptChunk], per_chunk_ann: dict[str, dict[int, Annotation]],
              words: list[TimedWord], gaps: list[float]) -> tuple[dict[int, Annotation], list[Conflict]]:
    n = len(words)
    meta: dict[str, tuple] = {}
    for order, ch in enumerate(chunks):
        meta[ch.id] = (order, set(ch.primary_token_ids), ch.token_ids[0], ch.token_ids[-1])

    cand: dict[int, list[str]] = defaultdict(list)
    for ch in chunks:                                    # chunks already in ascending order
        for gi in ch.token_ids:
            cand[gi].append(ch.id)

    merged: dict[int, Annotation] = {}
    conflicts: list[Conflict] = []
    for gi in range(n):
        ids = sorted(cand[gi], key=lambda cid: meta[cid][0])
        if len(ids) == 1:
            merged[gi] = per_chunk_ann[ids[0]][gi]
            continue
        best_id = ids[0]
        best_ann = per_chunk_ann[best_id][gi]
        for other in ids[1:]:
            other_ann = per_chunk_ann[other][gi]
            if best_ann == other_ann:
                continue
            if _rank(gi, other_ann, meta[other], gaps) > _rank(gi, best_ann, meta[best_id], gaps):
                conflicts.append(Conflict(token=f"w{gi}", won=other, lost=best_id,
                                          reason="priority", field_diff=_diff(best_ann, other_ann)))
                best_id, best_ann = other, other_ann
            else:
                conflicts.append(Conflict(token=f"w{gi}", won=best_id, lost=other,
                                          reason="priority", field_diff=_diff(best_ann, other_ann)))
        merged[gi] = best_ann

    seam_ids: set[int] = set()
    for ch in chunks:
        seam_ids.update(ch.left_overlap_token_ids)
        seam_ids.update(ch.right_overlap_token_ids)
    _seam_repair(merged, n, seam_ids, conflicts)
    return merged, conflicts


def _seam_repair(merged: dict[int, Annotation], n: int, seam_ids: set[int],
                 conflicts: list[Conflict]) -> None:
    for i in range(n - 1):
        j = i + 1
        ai, aj = merged[i], merged[j]
        # a terminal followed by a comma/semicolon is contradictory ONLY when the next token isn't a
        # sentence start (e.g. "time. Now," is fine — the comma follows a new sentence's first word)
        if ai.punctuationAfter in TERMINALS and aj.punctuationAfter in {",", ";"} \
                and not aj.sentenceStart:
            aj.punctuationAfter = ""
            conflicts.append(Conflict(token=f"w{j}", won=f"w{i}", lost=f"w{j}",
                                      reason="term_then_comma"))
        # a double terminal straddling a seam makes a 1-token "sentence" → collapse the second
        if i in seam_ids and j in seam_ids and ai.sentenceEnd and ai.punctuationAfter in TERMINALS \
                and aj.sentenceEnd and aj.punctuationAfter in TERMINALS:
            aj.punctuationAfter = ""
            aj.sentenceEnd = False
            conflicts.append(Conflict(token=f"w{j}", won=f"w{i}", lost=f"w{j}",
                                      reason="dup_terminal"))


def normalize_full(merged: dict[int, Annotation], words: list[TimedWord]) -> dict[int, Annotation]:
    """Make the merged whole internally consistent: every sentence has a capitalized first token and
    a flagged last token, and the document ends on a terminal mark (adding one is punctuation-only)."""
    n = len(words)
    if n == 0:
        return merged
    order = list(range(n))
    last = order[-1]
    if merged[last].punctuationAfter not in TERMINALS:
        merged[last].punctuationAfter = "."
        merged[last].sentenceEnd = True
    for (s, e) in sentence_spans(order, merged):
        merged[s].sentenceStart = True
        merged[s].capitalize = True
        merged[e].sentenceEnd = True
    merged[0].sentenceStart = True
    merged[0].capitalize = True
    return merged
