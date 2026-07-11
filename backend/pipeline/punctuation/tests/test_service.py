"""Service-level tests: retry ladder, fallback, invalid output, edge inputs."""
from __future__ import annotations

from backend.pipeline.punctuation.service import restore_transcript_punctuation

from .conftest import (
    FlakyProvider,
    RaisingProvider,
    TargetProvider,
    UnknownIdProvider,
    make_words,
)


class _Reverse:
    """Returns the good edits but in reversed order — order must not matter (ids are keys)."""

    def __init__(self, inner):
        self.inner = inner

    def infer(self, system, user, *, est_tokens=0):
        ce = self.inner.infer(system, user, est_tokens=est_tokens)
        return ce.__class__(edits=list(reversed(ce.edits)))


TEXT = "today we are learning kinematics"
TARGET = "Today, we are learning kinematics."


def test_retry_then_success_reports_repairs():
    words = make_words(TEXT)
    prov = FlakyProvider(TargetProvider(TARGET), bad=1, mode="unknown")
    r = restore_transcript_punctuation(words, provider_impl=prov, source="")
    assert r.status == "complete_with_repairs"
    assert r.readableText == TARGET
    assert r.metadata.retryCount >= 1


def test_persistent_invalid_output_degrades_but_preserves():
    words = make_words(TEXT)
    r = restore_transcript_punctuation(words, provider_impl=UnknownIdProvider(), source="")
    assert r.status == "degraded"
    assert [pw.word for pw in r.words] == [w["word"] for w in words]
    assert r.sentences                       # fallback still yields sentences


def test_provider_exception_degrades_safely():
    words = make_words(TEXT)
    r = restore_transcript_punctuation(words, provider_impl=RaisingProvider(), source="")
    assert r.status == "degraded"
    assert [pw.word for pw in r.words] == [w["word"] for w in words]
    for pw, w in zip(r.words, words):
        assert pw.start == w["start"] and pw.end == w["end"]


def test_reordered_annotations_still_correct():
    words = make_words(TEXT)
    r = restore_transcript_punctuation(words, provider_impl=_Reverse(TargetProvider(TARGET)),
                                       source="")
    assert r.readableText == TARGET


def test_empty_transcript_is_failed():
    r = restore_transcript_punctuation([], provider_impl=TargetProvider(""), source="")
    assert r.status == "failed"
    assert r.words == []


def test_single_word():
    r = restore_transcript_punctuation([{"word": "hello", "start": 0.0, "end": 0.3}],
                                       provider_impl=TargetProvider("Hello."), source="")
    assert r.status == "complete"
    assert r.readableText == "Hello."
    assert len(r.sentences) == 1


def test_missing_timestamps_do_not_crash():
    words = [{"word": "hello"}, {"word": "world"}]        # no start/end
    r = restore_transcript_punctuation(words, provider_impl=TargetProvider("Hello world."),
                                       source="")
    assert [pw.word for pw in r.words] == ["hello", "world"]
    assert r.readableText == "Hello world."
