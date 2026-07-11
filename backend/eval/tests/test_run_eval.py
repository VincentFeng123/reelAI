"""Unit tests for the trustworthy-eval harness (freeze + average-N-runs + variance).

Pure logic only — no LLM, no network. Covers CLI parsing, the freeze/rebuild
structure-resolution decision, and the run-aggregation math that turns N noisy runs
into a mean ± std so comprehension deltas can be told apart from judge noise.
"""
from __future__ import annotations

import math

import pytest

import backend.eval.run_eval as R


# ── CLI parsing ───────────────────────────────────────────────────────────────
def test_parse_defaults():
    a = R.parse_eval_args([])
    assert a["video_ids"] == []
    assert a["topic"] is None
    assert a["verbose"] is False
    assert a["runs"] == 1
    assert a["freeze"] is False
    assert a["freeze_specs"] is False
    assert a["rebuild"] is False


def test_parse_full():
    a = R.parse_eval_args(
        ["vid1", "vid2", "--runs", "3", "--freeze", "--topic", "the derivative", "--verbose"]
    )
    assert a["video_ids"] == ["vid1", "vid2"]
    assert a["runs"] == 3
    assert a["freeze"] is True
    assert a["topic"] == "the derivative"
    assert a["verbose"] is True
    assert a["freeze_specs"] is False


def test_parse_freeze_specs_and_rebuild_are_distinct_flags():
    a = R.parse_eval_args(["--freeze-specs", "--rebuild"])
    assert a["freeze_specs"] is True
    assert a["rebuild"] is True
    assert a["freeze"] is False


def test_parse_runs_invalid_falls_back_to_one():
    assert R.parse_eval_args(["--runs", "abc"])["runs"] == 1
    assert R.parse_eval_args(["--runs", "0"])["runs"] == 1
    assert R.parse_eval_args(["--runs"])["runs"] == 1


def test_parse_topic_value_not_captured_as_video_id():
    a = R.parse_eval_args(["vid", "--topic", "x"])
    assert a["video_ids"] == ["vid"]
    assert a["topic"] == "x"


def test_parse_runs_does_not_swallow_following_flag():
    # a non-numeric token after --runs must fall through to normal handling, not be eaten
    a = R.parse_eval_args(["--runs", "--freeze"])
    assert a["runs"] == 1
    assert a["freeze"] is True


def test_parse_runs_does_not_swallow_video_id():
    a = R.parse_eval_args(["--runs", "vid1"])
    assert a["runs"] == 1
    assert a["video_ids"] == ["vid1"]


# ── video selection (explicit ids are never truncated) ────────────────────────
def test_select_videos_caps_only_the_default_set():
    discovered = [f"v{i}" for i in range(20)]
    assert R.select_videos([], discovered, cap=8) == discovered[:8]


def test_select_videos_keeps_all_explicit_ids():
    explicit = [f"v{i}" for i in range(10)]
    assert R.select_videos(explicit, ["auto1", "auto2"], cap=8) == explicit


# ── headline aggregation over a FIXED video set (composition-clean) ───────────
def test_aggregate_over_runs_aligned():
    results = [{"runs": [{"c": 0.4}, {"c": 0.6}]},
               {"runs": [{"c": 0.5}, {"c": 0.5}]}]
    s = R.aggregate_over_runs(results, "c")
    assert s["videos"] == 2
    assert s["mean"] == pytest.approx(0.5)      # per-run means 0.45, 0.55 → mean 0.5
    assert s["std"] == pytest.approx(0.0707, abs=1e-3)


def test_aggregate_over_runs_excludes_videos_that_flip_to_none():
    # video 1 flips numeric→None on run 2; it must be dropped ENTIRELY, not partially,
    # so the per-run means aren't taken over different video sets per index.
    results = [{"runs": [{"c": 0.4}, {"c": None}]},
               {"runs": [{"c": 0.5}, {"c": 0.5}]}]
    s = R.aggregate_over_runs(results, "c")
    assert s["videos"] == 1                      # only the always-numeric video counts
    assert s["mean"] == pytest.approx(0.5)
    assert s["std"] == 0.0


def test_aggregate_over_runs_ragged_uses_common_run_count():
    results = [{"runs": [{"c": 0.4}, {"c": 0.6}, {"c": 0.8}]},
               {"runs": [{"c": 0.5}]}]
    s = R.aggregate_over_runs(results, "c")      # common run count = 1
    assert s["n"] == 1                           # only 1 comparable run index
    assert s["videos"] == 2
    assert s["mean"] == pytest.approx(0.45)


def test_aggregate_over_runs_empty_is_none():
    assert R.aggregate_over_runs([], "c") is None
    assert R.aggregate_over_runs([{"runs": [{"c": None}]}], "c") is None


# ── freeze / rebuild structure resolution ─────────────────────────────────────
# load_structure now takes (video_id, sentences, allow_stale=…) — the W25-A freshness
# seam; these stubs accept the full signature. --freeze passes allow_stale=True (the
# explicit hold-anyway override) and classifies the result via structure_is_stale.
def test_resolve_uses_cache_when_frozen(monkeypatch):
    cached = object()
    calls = {"build": 0, "save": 0}
    monkeypatch.setattr(R, "load_structure", lambda vid, sents=None, **kw: cached)
    monkeypatch.setattr(R, "structure_is_stale", lambda st, sents: None)   # fresh
    monkeypatch.setattr(R, "save_structure", lambda st: calls.__setitem__("save", calls["save"] + 1))

    def build():
        calls["build"] += 1
        return object()

    st, src = R.resolve_structure("vid", build, freeze=True, rebuild=False, sentences=["s"])
    assert st is cached
    assert src == "cached"
    assert calls["build"] == 0          # frozen → never rebuilt
    assert calls["save"] == 0           # cache hit → not re-saved


def test_resolve_frozen_stale_cache_loads_with_override_and_reports_cached_stale(monkeypatch):
    # --freeze on a stale cache: the override must LOAD it (holding structure constant is
    # the whole point of freeze) but the source string must say so — never plain "cached".
    cached = object()
    seen = {}
    monkeypatch.setattr(R, "load_structure",
                        lambda vid, sents=None, **kw: seen.update(kw) or cached)
    monkeypatch.setattr(R, "structure_is_stale",
                        lambda st, sents: "n_sentences 322 != live 183")
    monkeypatch.setattr(R, "save_structure", lambda st: (_ for _ in ()).throw(AssertionError))

    st, src = R.resolve_structure("vid", lambda: object(), freeze=True, rebuild=False,
                                  sentences=["s"])
    assert st is cached
    assert src == "cached-stale"
    assert seen.get("allow_stale") is True      # freeze IS the explicit override


def test_resolve_builds_and_saves_when_frozen_but_no_cache(monkeypatch):
    built = object()
    saved = []
    monkeypatch.setattr(R, "load_structure", lambda vid, sents=None, **kw: None)
    monkeypatch.setattr(R, "save_structure", lambda st: saved.append(st))

    st, src = R.resolve_structure("vid", lambda: built, freeze=True, rebuild=False)
    assert st is built
    assert src == "built"
    assert saved == [built]             # newly built structure is persisted for next freeze


def test_resolve_without_freeze_always_builds(monkeypatch):
    built = object()
    saved = []
    monkeypatch.setattr(R, "load_structure", lambda vid, sents=None, **kw: object())  # cache present…
    monkeypatch.setattr(R, "save_structure", lambda st: saved.append(st))

    st, src = R.resolve_structure("vid", lambda: built, freeze=False, rebuild=False)
    assert st is built                  # …but ignored: no-freeze rebuilds every time
    assert src == "built"
    assert saved == [built]


def test_resolve_rebuild_ignores_cache(monkeypatch):
    built = object()
    saved = []
    monkeypatch.setattr(R, "load_structure", lambda vid, sents=None, **kw: object())  # cache present
    monkeypatch.setattr(R, "save_structure", lambda st: saved.append(st))

    st, src = R.resolve_structure("vid", lambda: built, freeze=True, rebuild=True)
    assert st is built                  # rebuild forces a fresh build even when frozen
    assert src == "rebuilt"
    assert saved == [built]


# ── sentence-path unification (W25-A) ─────────────────────────────────────────
def test_sentences_uses_the_app_build_sentences_seam(monkeypatch):
    # run_eval must build the SAME punctuation-restored index the orchestrator uses —
    # the legacy pysbd path produced a different sentence universe (322 vs 183 on qP)
    # and its saved structures poisoned the shared app cache.
    from backend import config
    calls = []

    def fake_build_sentences(transcript, video_id, settings, progress=None):
        calls.append((transcript, video_id, settings, progress))
        return ["sentinel"]

    monkeypatch.setattr(R, "build_sentences", fake_build_sentences)
    t = {"words": [], "segments": []}
    assert R._sentences(t, "vidX") == ["sentinel"]
    transcript, video_id, settings, progress = calls[0]
    assert transcript is t
    assert video_id == "vidX"           # keys the punctuation chunk cache (offline on cached vids)
    assert settings == dict(config.DEFAULTS)
    assert progress is None


# ── aggregation math ──────────────────────────────────────────────────────────
def test_is_num():
    assert R._is_num(0.5)
    assert R._is_num(3)
    assert not R._is_num(None)
    assert not R._is_num(float("nan"))
    assert not R._is_num("x")
    assert not R._is_num(True)          # bools are not treated as metric numbers


def test_mean():
    assert R.mean([0.4, 0.5, 0.6]) == pytest.approx(0.5)
    assert R.mean([]) is None


def test_metric_values_filters_none_and_nan():
    rows = [{"c": 0.4}, {"c": None}, {"c": float("nan")}, {"c": 0.6}, {}]
    assert R._metric_values(rows, "c") == [0.4, 0.6]


def test_summarize_multi_run_reports_sample_std():
    s = R.summarize([0.4, 0.5, 0.6])
    assert s["n"] == 3
    assert s["mean"] == pytest.approx(0.5)
    assert s["std"] == pytest.approx(0.1)      # sample stdev (n-1), not population
    assert s["min"] == 0.4
    assert s["max"] == 0.6


def test_summarize_single_run_has_zero_std():
    s = R.summarize([0.7])
    assert s["n"] == 1
    assert s["mean"] == 0.7
    assert s["std"] == 0.0
    assert s["min"] == 0.7 and s["max"] == 0.7


def test_summarize_empty_is_none():
    assert R.summarize([]) is None


def test_summarize_ignores_nan_and_none_entries():
    s = R.summarize([0.4, None, float("nan"), 0.6])
    assert s["n"] == 2
    assert s["mean"] == pytest.approx(0.5)
    assert not math.isnan(s["std"])
