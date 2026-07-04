"""comprehension() excludes error verdicts; judge_error_rate accounting. Offline."""
from __future__ import annotations

import pytest

import backend.eval.metrics as metrics
from backend.pipeline.assemble.validate import JudgeVerdict


class _Sent:
    def __init__(self, i):
        self.text, self.start, self.end, self.ends_with_period = f"s{i}.", float(i), i + 0.9, True


def _specs(n):
    return [{"sentence_start_idx": 0, "sentence_end_idx": 0, "role": "explanation",
             "context_card": "", "start": 0.0, "end": 1.0} for _ in range(n)]


def test_comprehension_excludes_error_verdicts(monkeypatch):
    verdicts = [JudgeVerdict(score_10=9), JudgeVerdict(error=True), JudgeVerdict(score_10=5)]
    it = iter(verdicts)
    monkeypatch.setattr(metrics, "judge_clip", lambda *a, **kw: next(it))
    mean, rate, n_judged, n_error = metrics.comprehension(_specs(3), [_Sent(0)], None, "t", 0.7)
    assert n_judged == 2 and n_error == 1
    assert mean == pytest.approx((0.9 + 0.5) / 2)
    assert rate == pytest.approx(0.5)                    # 0.9 passes 0.7; 0.5 doesn't


def test_comprehension_all_errors(monkeypatch):
    monkeypatch.setattr(metrics, "judge_clip", lambda *a, **kw: JudgeVerdict(error=True))
    mean, rate, n_judged, n_error = metrics.comprehension(_specs(2), [_Sent(0)], None, "t", 0.7)
    assert (mean, rate, n_judged, n_error) == (0.0, 0.0, 0, 2)


def test_comprehension_empty_specs():
    assert metrics.comprehension([], [], None, "t", 0.7) == (0.0, 0.0, 0, 0)


def test_judge_failures_carries_error_flag(monkeypatch):
    monkeypatch.setattr(metrics, "judge_clip", lambda *a, **kw: JudgeVerdict(error=True))
    rows = metrics.judge_failures(_specs(1), [_Sent(0)], None, "t")
    role, score, fails, error = rows[0]
    assert error is True
