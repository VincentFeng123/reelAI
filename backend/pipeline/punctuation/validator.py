"""Strict validation of punctuation annotations, before we accept a result.

``densify`` turns the model's sparse edits into a dense per-token annotation and rejects unknown /
duplicate ids up front. ``validate`` runs the structural checks. Messages are prefixed ``ERROR:``
(hard-fail → retry/repair/fallback) or ``WARN:`` (logged, auto-repairable, never blocks). Chunk edges
legitimately start mid-sentence, so first-token-starts-a-sentence is required only for the full doc.
"""
from __future__ import annotations

from dataclasses import dataclass

from .reconstructor import sentence_spans
from .types import Annotation, PUNCT_ENUM, TERMINALS, TimedWord, TokenEdit, TranscriptChunk

# Sentence-length guards (tokens / seconds).
RUNAWAY_TOKENS = 160
RUNAWAY_SECS = 120.0
LONG_TOKENS = 90
LONG_SECS = 60.0


@dataclass
class Scope:
    expected_ids: list[int]      # global word indices, ascending
    is_full: bool


def hard_errors(errors: list[str]) -> list[str]:
    return [e for e in errors if e.startswith("ERROR")]


def densify(chunk: TranscriptChunk, words: list[TimedWord],
            edits: list[TokenEdit]) -> tuple[dict[int, Annotation], list[str]]:
    """Sparse edits → dense annotation over ``chunk.token_ids``; report unknown/duplicate/enum errors."""
    id_to_index = {words[gi].id: gi for gi in chunk.token_ids}
    dense: dict[int, Annotation] = {gi: Annotation() for gi in chunk.token_ids}
    errors: list[str] = []
    seen: set[str] = set()
    for e in edits:
        if e.id in seen:
            errors.append(f"ERROR: duplicate token id {e.id}")
            continue
        seen.add(e.id)
        gi = id_to_index.get(e.id)
        if gi is None:
            errors.append(f"ERROR: unknown token id {e.id}")
            continue
        if e.p not in PUNCT_ENUM:
            errors.append(f"ERROR: invalid punctuationAfter {e.p!r} at {e.id}")
            continue
        dense[gi] = Annotation(
            capitalize=e.cap, punctuationAfter=e.p, sentenceStart=e.ss,
            sentenceEnd=e.se, paragraphStart=e.pg, confidence=e.conf)
    return dense, errors


def validate(scope: Scope, words: list[TimedWord], dense: dict[int, Annotation]) -> list[str]:
    errors: list[str] = []
    expected = scope.expected_ids
    if not expected:
        return ["ERROR: empty scope"]

    keys = set(dense.keys())
    exp = set(expected)
    missing = exp - keys
    unknown = keys - exp
    if missing:
        errors.append(f"ERROR: missing annotations for {len(missing)} ids (e.g. w{min(missing)})")
    if unknown:
        errors.append(f"ERROR: annotations for unknown ids (e.g. {sorted(unknown)[:3]})")

    for gi in expected:
        a = dense.get(gi)
        if a is None:
            continue
        if a.punctuationAfter not in PUNCT_ENUM:
            errors.append(f"ERROR: invalid punctuationAfter {a.punctuationAfter!r} at w{gi}")

    order = sorted(expected)
    if scope.is_full and not dense[order[0]].sentenceStart:
        errors.append("ERROR: first token does not start a sentence")

    # sentence-length guards
    for (s, e) in sentence_spans(order, dense):
        span = e - s + 1
        secs = float(words[e].end) - float(words[s].start)
        if span > RUNAWAY_TOKENS or secs > RUNAWAY_SECS:
            errors.append(f"ERROR: runaway sentence w{s}-w{e} ({span} tokens, {secs:.0f}s)")
        elif span > LONG_TOKENS or secs > LONG_SECS:
            errors.append(f"WARN: long sentence w{s}-w{e} ({span} tokens, {secs:.0f}s)")

    # sentence-flag consistency (auto-repairable)
    open_sent = False
    for gi in order:
        a = dense[gi]
        if a.sentenceStart:
            if open_sent:
                errors.append(f"WARN: sentenceStart at w{gi} without prior sentenceEnd")
            open_sent = True
            if not a.capitalize:
                errors.append(f"WARN: sentence-initial w{gi} not capitalized")
        if a.sentenceEnd:
            if a.punctuationAfter not in TERMINALS and gi != order[-1]:
                errors.append(f"WARN: sentenceEnd w{gi} lacks terminal punctuation")
            open_sent = False

    # no duplicate punctuation across a boundary (full-doc backstop for the reconciler). A terminal
    # then comma is only contradictory when the comma token does NOT begin a new sentence.
    if scope.is_full:
        for i, j in zip(order, order[1:]):
            if (dense[i].punctuationAfter in TERMINALS and dense[j].punctuationAfter in {",", ";"}
                    and not dense[j].sentenceStart):
                errors.append(f"ERROR: terminal then comma at w{i}->w{j}")

    # balanced quotes — near-vacuous (enum has no quote glyph); WARN over original tokens only
    quote_count = sum(words[gi].word.count('"') for gi in expected)
    if quote_count % 2:
        errors.append("WARN: unbalanced quotation marks in source tokens")

    return errors
