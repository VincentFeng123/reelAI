"""E1d judge-calibration math — kappa hand-checked, bootstrap CI shape, the 0.5s join,
and the per-kind refusal below 10 human positives. Pure math, zero LLM."""
from __future__ import annotations

import math

import pytest

import backend.eval.judge_calibration as C


# hand-checkable 2x2 table: a=20 (yes/yes), b=5 (yes/no), c=10 (no/yes), d=15 (no/no)
# po = 35/50 = 0.7; pe = 0.5*0.6 + 0.5*0.4 = 0.5; kappa = (0.7-0.5)/(1-0.5) = 0.4
TABLE_04 = ([(True, True)] * 20 + [(True, False)] * 5
            + [(False, True)] * 10 + [(False, False)] * 15)


# ── Cohen's kappa ─────────────────────────────────────────────────────────────
def test_kappa_hand_checked_contingency_table():
    assert C.cohens_kappa(TABLE_04) == pytest.approx(0.4)


def test_kappa_perfect_agreement():
    assert C.cohens_kappa([(True, True), (False, False)] * 5) == pytest.approx(1.0)


def test_kappa_chance_only_agreement_is_zero():
    # independent raters at 50/50: po = pe = 0.5 → kappa 0
    pairs = [(True, True), (True, False), (False, True), (False, False)]
    assert C.cohens_kappa(pairs) == pytest.approx(0.0)


def test_kappa_undefined_cases_are_nan():
    assert math.isnan(C.cohens_kappa([]))
    assert math.isnan(C.cohens_kappa([(True, True)] * 5))    # both raters constant → pe=1


# ── bootstrap CI ──────────────────────────────────────────────────────────────
def test_bootstrap_ci_shape_and_determinism():
    assert C.DEFAULT_RESAMPLES == 1000
    ci = C.bootstrap_kappa_ci(TABLE_04, seed=7)
    assert ci == C.bootstrap_kappa_ci(TABLE_04, seed=7)      # deterministic given the seed
    lo, hi, n = ci
    assert -1.0 <= lo <= hi <= 1.0
    assert lo <= 0.4 <= hi                                   # point estimate inside the CI
    assert 990 <= n <= 1000                                  # ~all resamples defined


def test_bootstrap_ci_empty_and_degenerate():
    assert C.bootstrap_kappa_ci([]) is None
    # every resample of an all-constant table is degenerate → no defined kappa at all
    assert C.bootstrap_kappa_ci([(True, True)] * 4, n_resamples=20, seed=1) is None


# ── join by video_id + 0.5s tolerance ─────────────────────────────────────────
def _entry(vid="v", start=10.0, end=20.0, status="shipped", understandable=True,
           kinds=(), score=0.8):
    return {"video_id": vid, "start": start, "end": end, "status": status,
            "stratum": "random",
            "judge": {"score": score, "understandable": understandable,
                      "failure_kinds": list(kinds)}}


def _human(clips):
    return {"clips": clips, "video_note": ""}


def test_join_matches_within_half_second_both_endpoints():
    entries = [_entry()]
    ok = {"v": _human([{"start": 10.3, "end": 20.4, "understandable": False,
                        "failure_kinds": ["off_topic"]}])}
    joined = C.join_labels(entries, ok)
    assert len(joined) == 1
    assert joined[0]["human_understandable"] is False
    assert joined[0]["human_kinds"] == ["off_topic"]
    assert joined[0]["judge_understandable"] is True

    too_far = {"v": _human([{"start": 10.6, "end": 20.0, "understandable": True}])}
    assert C.join_labels(entries, too_far) == []
    wrong_video = {"other": _human([{"start": 10.0, "end": 20.0, "understandable": True}])}
    assert C.join_labels(entries, wrong_video) == []


def test_join_skips_unanswered_labels():
    unanswered = {"v": _human([{"start": 10.0, "end": 20.0, "understandable": None}])}
    assert C.join_labels([_entry()], unanswered) == []


def test_kappa_pairs_come_from_joined_rows():
    joined = C.join_labels(
        [_entry(understandable=True), _entry(start=30.0, end=40.0, understandable=False)],
        {"v": _human([{"start": 10.0, "end": 20.0, "understandable": True},
                      {"start": 30.0, "end": 40.0, "understandable": True}])})
    assert C.kappa_pairs(joined) == [(True, True), (True, False)]


# ── per-kind precision/recall + the <10 refusal ───────────────────────────────
def _row(status, judge_kinds=(), human_kinds=(), human_ok=False):
    return {"status": status, "judge_kinds": list(judge_kinds),
            "human_kinds": list(human_kinds), "human_understandable": human_ok,
            "judge_understandable": status != "rejected"}


def test_per_kind_refuses_below_ten_human_positives():
    joined = [_row("rejected", ["missing_reasoning"], ["missing_reasoning"])] * 9
    stats = C.per_kind_stats(joined)
    st = stats["missing_reasoning"]
    assert st["n_human"] == 9
    assert st["sufficient"] is False
    assert "precision" not in st and "recall" not in st     # REFUSED, not caveated
    assert st["action"] == "need-more-labels"


def test_per_kind_precision_recall_and_actions():
    joined = (
        # off_topic: 10 judge kills, 8 true positives; 12 human positives overall
        [_row("rejected", ["off_topic"], ["off_topic"])] * 8
        + [_row("rejected", ["off_topic"], [])] * 2
        + [_row("shipped", [], ["off_topic"])] * 4
        # missing_result: 5 kills, 2 tp; 10 human positives
        + [_row("rejected", ["missing_result"], ["missing_result"])] * 2
        + [_row("rejected", ["missing_result"], [])] * 3
        + [_row("shipped", [], ["missing_result"])] * 8
    )
    stats = C.per_kind_stats(joined)
    ot = stats["off_topic"]
    assert ot["n_human"] == 12 and ot["n_judge_kills"] == 10
    assert ot["precision"] == pytest.approx(0.8)
    assert ot["recall"] == pytest.approx(8 / 12)
    assert ot["action"] == "trust"                          # precision ≥ 0.7 on ≥10 labels
    mr = stats["missing_result"]
    assert mr["precision"] == pytest.approx(0.4)
    assert mr["recall"] == pytest.approx(0.2)
    assert mr["action"] == "distrust"


def test_per_kind_human_only_kind_with_no_kills():
    # a human-only boundary kind: enough labels, but the judge never kills on it
    joined = [_row("shipped", [], ["starts_mid_thought"])] * 10
    st = C.per_kind_stats(joined)["starts_mid_thought"]
    assert st["sufficient"] is True
    assert st["n_judge_kills"] == 0
    assert st["precision"] is None                          # nothing to trust or distrust
    assert st["action"] == "need-more-labels"


def test_recommended_action_thresholds():
    assert C.recommended_action({"sufficient": False}) == "need-more-labels"
    assert C.recommended_action({"sufficient": True, "precision": 0.70}) == "trust"
    assert C.recommended_action({"sufficient": True, "precision": 0.69}) == "distrust"
    assert C.recommended_action({"sufficient": True, "precision": None}) == "need-more-labels"


# ── human-block discovery ─────────────────────────────────────────────────────
def test_load_human_blocks_skips_files_without_labels(tmp_path):
    (tmp_path / "a.json").write_text(
        '{"video_id": "a", "human": {"clips": [{"start": 1, "end": 2, '
        '"understandable": true}], "video_note": ""}}')
    (tmp_path / "b.json").write_text('{"video_id": "b", "units": []}')   # no human block
    (tmp_path / "c.json").write_text('not json')                          # unreadable: skipped
    blocks = C.load_human_blocks(tmp_path)
    assert set(blocks) == {"a"}
    assert blocks["a"]["clips"][0]["understandable"] is True
