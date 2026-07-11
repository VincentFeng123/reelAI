"""Validator + densify: reject invalid model output; classify hard vs. soft."""
from __future__ import annotations

from backend.pipeline.punctuation.types import Annotation, TimedWord, TokenEdit, TranscriptChunk
from backend.pipeline.punctuation.validator import Scope, densify, hard_errors, validate


def _words(n: int) -> list[TimedWord]:
    return [TimedWord(id=f"w{i}", word=f"tok{i}", start=i * 0.3, end=i * 0.3 + 0.25)
            for i in range(n)]


def _chunk(n: int) -> TranscriptChunk:
    ids = list(range(n))
    return TranscriptChunk(id="c0", token_ids=ids, primary_token_ids=ids,
                           left_overlap_token_ids=[], right_overlap_token_ids=[])


def test_densify_rejects_unknown_id():
    words, chunk = _words(5), _chunk(5)
    _, errs = densify(chunk, words, [TokenEdit(id="w99", p=".")])
    assert any("unknown token id" in e for e in errs)


def test_densify_rejects_duplicate_id():
    words, chunk = _words(5), _chunk(5)
    _, errs = densify(chunk, words, [TokenEdit(id="w0"), TokenEdit(id="w0")])
    assert any("duplicate token id" in e for e in errs)


def test_bad_enum_rejected_by_densify():
    # `p` is a plain str at the LLM boundary (providers reject empty-string enums); densify enforces it
    words, chunk = _words(5), _chunk(5)
    _, errs = densify(chunk, words, [TokenEdit(id="w0", p="x")])
    assert any("invalid punctuationAfter" in e for e in errs)


def test_runaway_sentence_is_hard_error():
    n = 200
    words = _words(n)
    dense = {i: Annotation() for i in range(n)}
    dense[0] = Annotation(sentenceStart=True, capitalize=True)
    errs = validate(Scope(list(range(n)), is_full=True), words, dense)
    assert any("runaway sentence" in e for e in hard_errors(errs))


def test_terminal_then_comma_is_hard_error():
    words = _words(3)
    dense = {
        0: Annotation(sentenceStart=True, capitalize=True, punctuationAfter="."),
        1: Annotation(punctuationAfter=","),
        2: Annotation(punctuationAfter=".", sentenceEnd=True),
    }
    errs = validate(Scope([0, 1, 2], is_full=True), words, dense)
    assert any("terminal then comma" in e for e in hard_errors(errs))


def test_full_first_token_must_start_sentence():
    words = _words(3)
    dense = {i: Annotation() for i in range(3)}
    dense[2] = Annotation(punctuationAfter=".", sentenceEnd=True)
    errs = validate(Scope([0, 1, 2], is_full=True), words, dense)
    assert any("first token does not start a sentence" in e for e in hard_errors(errs))


def test_chunk_edge_may_start_midsentence():
    # a non-full scope must NOT require the first token to start a sentence
    words = _words(3)
    dense = {i: Annotation() for i in range(3)}
    dense[2] = Annotation(punctuationAfter=".", sentenceEnd=True)
    errs = validate(Scope([0, 1, 2], is_full=False), words, dense)
    assert not any("first token" in e for e in errs)
