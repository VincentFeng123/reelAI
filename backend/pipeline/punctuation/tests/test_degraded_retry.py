"""W25-B degraded-artifact retry-once semantics (offline; fake providers, tmp WORK_DIR).

A cached ``status='degraded'`` punctuation artifact used to be served forever (qP's live
index was 183 run-on "sentences" replayed from this cache). Now it is retry-eligible —
ONE fresh attempt per run: a successful retry replaces the artifact; a failed retry keeps
the degraded artifact byte-for-byte (never worse than today) and stays retry-eligible on
the NEXT run. Non-degraded artifacts keep the old serve-from-cache behavior.
"""
from __future__ import annotations

from backend import config
from backend.pipeline.punctuation.service import restore_transcript_punctuation

from .conftest import RaisingProvider, TargetProvider, make_words

TEXT = "today we are learning kinematics"
TARGET = "Today, we are learning kinematics."


class CountingProvider:
    """Delegates to ``inner`` while counting infer() calls — proves a retry ran (or didn't)."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = 0

    def infer(self, system, user, *, est_tokens=0):
        self.calls += 1
        return self.inner.infer(system, user, est_tokens=est_tokens)


def _run(words, provider, video_id):
    return restore_transcript_punctuation(words, provider_impl=provider, source="",
                                          video_id=video_id)


def test_degraded_artifact_retried_and_replaced_on_success(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WORK_DIR", tmp_path)
    words = make_words(TEXT)
    assert _run(words, RaisingProvider(), "vidR").status == "degraded"
    assert (tmp_path / "vidR" / "punctuation.json").exists()   # degraded artifact persisted

    good = CountingProvider(TargetProvider(TARGET))
    r2 = _run(words, good, "vidR")
    assert good.calls >= 1                       # degraded cache did NOT short-circuit the run
    assert r2.status == "complete"
    assert r2.readableText == TARGET             # the degraded chunk was retried too, not replayed

    guard = CountingProvider(TargetProvider(TARGET))
    r3 = _run(words, guard, "vidR")
    assert guard.calls == 0                      # healed artifact serves from cache again
    assert r3.status == "complete" and r3.metadata.cacheHitCount >= 1


def test_failed_retry_keeps_degraded_artifact_and_stays_eligible(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WORK_DIR", tmp_path)
    words = make_words(TEXT)
    assert _run(words, RaisingProvider(), "vidK").status == "degraded"
    art = tmp_path / "vidK" / "punctuation.json"
    before = art.read_text(encoding="utf-8")

    prov = CountingProvider(RaisingProvider())
    r2 = _run(words, prov, "vidK")
    assert prov.calls >= 1                       # one fresh attempt was made this run
    assert r2.status == "degraded"               # never worse: still a usable degraded result
    assert [pw.word for pw in r2.words] == [w["word"] for w in words]
    assert art.read_text(encoding="utf-8") == before   # artifact KEPT, not overwritten

    prov2 = CountingProvider(RaisingProvider())
    _run(words, prov2, "vidK")
    assert prov2.calls >= 1                      # retry-eligible again NEXT run (once per run)


def test_non_degraded_artifact_is_never_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WORK_DIR", tmp_path)
    words = make_words(TEXT)
    assert _run(words, TargetProvider(TARGET), "vidC").status == "complete"
    guard = CountingProvider(TargetProvider(TARGET))
    r2 = _run(words, guard, "vidC")
    assert guard.calls == 0                      # complete artifacts keep serve-from-cache
    assert r2.status == "complete" and r2.metadata.cacheHitCount >= 1
