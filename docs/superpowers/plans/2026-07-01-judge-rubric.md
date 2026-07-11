# Judge Rubric + CoT + 1-10 Scale Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the clip-only judge G-Eval-shaped — explicit rubric, reasoning-first CoT, 1-10 integer score normalized to the existing 0-1 internals — and replace the silent score-0.7 judge-failure fallback with an honest ship-but-flag error verdict.

**Architecture:** All verdict changes live in `assemble/validate.py` (schema field order drives Gemini generation order; a pydantic validator normalizes `score_10`→`score` so no consumer thresholds change). Error verdicts flow: `judge_clip` fallback → repair-loop early-return (ship-but-flag) → neutral completeness + `judge_error` flag through `snap_candidates` → `"unjudged"` warning penalized by `boundary_score` → excluded from eval comprehension with a new `judge_error_rate` metric.

**Tech Stack:** Python 3.12, pydantic 2 (`model_validator`), pytest 9.1.1, mocked `llm_json` (offline tests).

**Spec:** `docs/superpowers/specs/2026-07-01-judge-rubric-design.md`

## Global Constraints

- Run from `/Users/vincentfeng/Documents/practice/clips` with `.venv/bin/python`.
- **No git repo** — "Checkpoint" steps (full pytest + compileall) replace commits.
- Tests offline: no network/LLM; monkeypatch `backend.llm.llm_json` (resolved at call time — `judge_clip` does `from ...llm import llm_json` inside the function).
- Boolean gates and the `failure_reasons` kind vocabulary are UNCHANGED (repair-loop contract): kinds are exactly `unresolved_reference, missing_prerequisite, missing_visual, missing_problem_statement, missing_reasoning, missing_result, not_source_grounded, off_topic, other`.
- `reasoning` MUST be the first declared field of `JudgeVerdict`; `error` is never emitted by the LLM (forced False on successful parse; True only in the fallback).
- Ship-but-flag policy: error verdict → candidate kept, repair skipped, completeness 0.5, `"unjudged"` warning, eval exclusion. Never score 0.7, never `understandable=True` on failure.
- Suite baseline before this plan: 88 passed. Old-judge eval baseline is already captured (scratchpad `judge-baseline-old.log`) — do NOT re-run it after the prompt changes.

---

### Task 1: Verdict schema, rubric prompt, honest fallback, repair early-return

**Files:**
- Modify: `backend/pipeline/assemble/validate.py`
- Create: `backend/pipeline/assemble/tests/__init__.py` (empty)
- Create: `backend/pipeline/assemble/tests/conftest.py`
- Test: `backend/pipeline/assemble/tests/test_judge_rubric.py`

**Interfaces:**
- Consumes: existing `JudgeVerdict`, `judge_clip`, `is_complete`, `validate_and_repair` in validate.py; `backend.llm.llm_json` (lazily imported inside `judge_clip`).
- Produces: `JudgeVerdict` with `reasoning: str` (FIRST field), `score_10: int`, `error: bool`; normalization `score_10>0 → score = clamp(score_10,1,10)/10`; fallback verdict `error=True, understandable=False, score=0.0`; `is_complete` False on error; `validate_and_repair` returns the candidate immediately on an error verdict without caching it. Later tasks rely on `verdict.error` existing and being trustworthy.

- [ ] **Step 1: Write shared fixtures**

```python
# backend/pipeline/assemble/tests/conftest.py
"""Offline fixtures for assemble tests: minimal sentences, units, adapter. No LLM/network."""
from __future__ import annotations

from backend.pipeline.sentences import Sentence
from backend.pipeline.understand.models import Unit


def mini_sents(n: int, sec: float = 10.0) -> list[Sentence]:
    return [Sentence(idx=i, text=f"sentence {i}.", start=i * sec, end=(i + 1) * sec - 0.1,
                     terminator=".", ends_with_period=True, word_start_idx=i, word_end_idx=i,
                     align_confidence=1.0) for i in range(n)]


def mini_units(sents: list[Sentence]) -> list[Unit]:
    return [Unit(unit_id=f"u{i:04d}", start=s.start, end=s.end, sentence_range=(i, i),
                 role="explanation", transcript=s.text) for i, s in enumerate(sents)]


class FakeAdapter:
    """Contract-free adapter: every verdict field optional, no completeness contract."""

    def required_verdict_fields(self, role):
        return []

    def required_elements(self, role):
        return []

    def contract_for(self, role):
        return None
```

- [ ] **Step 2: Write the failing tests**

```python
# backend/pipeline/assemble/tests/test_judge_rubric.py
"""G-Eval judge: reasoning-first schema, 1-10 normalization, honest failure verdict,
repair-loop ship-but-flag. All offline (llm_json monkeypatched)."""
from __future__ import annotations

import threading

import pytest

import backend.llm as llm_mod
from backend.pipeline.assemble.graph import Graph
from backend.pipeline.assemble.types import Candidate
from backend.pipeline.assemble.validate import (
    JUDGE_SYSTEM, JudgeVerdict, is_complete, judge_clip, validate_and_repair,
)

from .conftest import FakeAdapter, mini_sents, mini_units


# ── schema ────────────────────────────────────────────────────────────────────
def test_reasoning_is_first_field():
    assert next(iter(JudgeVerdict.model_fields)) == "reasoning"


def test_score10_normalizes_to_score():
    assert JudgeVerdict(score_10=8).score == pytest.approx(0.8)
    assert JudgeVerdict(score_10=10).score == pytest.approx(1.0)


def test_score10_clamps_out_of_range():
    assert JudgeVerdict(score_10=12).score == pytest.approx(1.0)
    assert JudgeVerdict(score_10=-3).score == pytest.approx(0.1)


def test_score10_zero_keeps_legacy_score():
    assert JudgeVerdict(score=0.65).score == pytest.approx(0.65)   # no score_10 emitted


# ── prompt ────────────────────────────────────────────────────────────────────
def test_prompt_rubric_contents():
    for kind in ("unresolved_reference", "missing_prerequisite", "missing_visual",
                 "missing_problem_statement", "missing_reasoning", "missing_result",
                 "not_source_grounded", "off_topic", "other"):
        assert kind in JUDGE_SYSTEM
    assert "score_10" in JUDGE_SYSTEM
    assert "1-2" in JUDGE_SYSTEM and "9-10" in JUDGE_SYSTEM        # anchored bands
    assert "First write `reasoning`" in JUDGE_SYSTEM               # CoT-before-verdict


# ── failure path ──────────────────────────────────────────────────────────────
def test_fallback_verdict_on_llm_failure(monkeypatch):
    from backend import config
    monkeypatch.setattr(config, "JUDGE_MODEL", "")   # disable cross-model override: 1 call, config-independent

    def boom(*a, **kw):
        raise RuntimeError("api down")
    monkeypatch.setattr(llm_mod, "llm_json", boom)
    v = judge_clip("some clip text", "explanation", FakeAdapter())
    assert v.error is True
    assert v.understandable is False
    assert v.score == 0.0


def test_successful_parse_forces_error_false(monkeypatch):
    def fake(*a, **kw):
        return JudgeVerdict(reasoning="ok", score_10=9, understandable=True, error=True)
    monkeypatch.setattr(llm_mod, "llm_json", fake)
    v = judge_clip("text", "explanation", FakeAdapter())
    assert v.error is False                                        # LLM cannot self-flag error


def test_is_complete_false_on_error():
    v = JudgeVerdict(error=True, score=1.0, understandable=True)
    assert is_complete(v, "explanation", FakeAdapter(), min_score=0.7) is False


# ── repair loop: ship-but-flag ────────────────────────────────────────────────
def _mk_candidate(sents):
    return Candidate(cand_id="c0", anchor_id="u0000", role="explanation", facet="other",
                     title="t", reason="r", unit_ids=["u0000"], referential=[],
                     i_start=0, i_end=0, start=sents[0].start, end=sents[0].end)


def test_repair_returns_candidate_on_error_without_burning_budget(monkeypatch):
    from backend import config
    monkeypatch.setattr(config, "JUDGE_MODEL", "")   # isolate "no repair spin" from the retry ladder
    calls = {"n": 0}

    def boom(*a, **kw):
        calls["n"] += 1
        raise RuntimeError("api down")
    monkeypatch.setattr(llm_mod, "llm_json", boom)

    sents = mini_sents(3)
    units = mini_units(sents)
    units_by_id = {u.unit_id: u for u in units}
    cache: dict = {}
    cand = validate_and_repair(
        _mk_candidate(sents), sents, Graph([], units), units, units_by_id, {},
        FakeAdapter(), {}, lambda s, e: "", "topic", cache, threading.Lock())
    assert cand is not None                                        # ship-but-flag: kept
    assert cand.verdict.error is True
    assert calls["n"] == 1                                         # ONE judge attempt, no repair spin
    assert cache == {}                                             # error verdicts are never cached


def test_cross_model_failure_retries_on_authoring_model(monkeypatch):
    """The cross-model → authoring-model retry ladder must survive (spec: 'Keep the existing
    cross-model judge retry ladder'). With a JUDGE_MODEL override active, a failed first call
    retries once on the authoring model before any error verdict."""
    from backend import config
    monkeypatch.setattr(config, "JUDGE_MODEL", "gemini-2.5-flash-lite")
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "same")
    calls = {"n": 0}

    def flaky(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("cross-model down")
        return JudgeVerdict(reasoning="ok", score_10=8, understandable=True)
    monkeypatch.setattr(llm_mod, "llm_json", flaky)
    v = judge_clip("text", "explanation", FakeAdapter())
    assert calls["n"] == 2                       # authoring-model retry happened
    assert v.error is False and v.score == pytest.approx(0.8)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest backend/pipeline/assemble/tests/test_judge_rubric.py -q`
Expected: FAIL — `reasoning`/`score_10`/`error` unknown fields; `test_fallback_verdict_on_llm_failure` gets score 0.7; repair test gets non-error verdict.

- [ ] **Step 4: Implement in `backend/pipeline/assemble/validate.py`**

a) Import: change `from pydantic import BaseModel, Field` → `from pydantic import BaseModel, Field, model_validator`.

b) Replace `JudgeVerdict` with (field ORDER is load-bearing — google-genai preserves declaration order in `response_schema`, so Gemini generates `reasoning` first):

```python
class JudgeVerdict(BaseModel):
    reasoning: str = ""      # 2-3 sentence CoT the judge writes BEFORE the verdict fields
    understandable: bool = False
    score: float = 0.0       # normalized 0-1; derived from score_10 when the judge emits it
    score_10: int = 0        # judge emits an integer 1-10; 0 = not emitted (legacy/fallback)
    topic_identifiable: bool = True
    purpose_identifiable: bool = True
    all_references_resolved: bool = True
    prerequisites_satisfied: bool = True
    visuals_sufficient: bool = True
    problem_statement_complete: bool = True
    reasoning_complete: bool = True
    result_complete: bool = True
    source_grounded: bool = True
    failure_reasons: list[FailureReason] = Field(default_factory=list)
    error: bool = False      # judge call failed (set ONLY by the fallback path, never the LLM)

    @model_validator(mode="after")
    def _normalize_score(self):
        if self.score_10 > 0:
            self.score = min(max(self.score_10, 1), 10) / 10.0
        return self
```

c) Replace `JUDGE_SYSTEM` with:

```python
JUDGE_SYSTEM = (
    "You are a strict judge of whether a short video clip is SELF-CONTAINED. You see ONLY the "
    "clip's transcript (and any on-screen text), never the surrounding video. Decide whether a "
    "brand-new viewer, watching only this clip, could understand what it is about and follow it "
    "to a complete thought.\n"
    "Evaluate in three steps:\n"
    "1. Identify what the clip is about and why it matters (topic and purpose).\n"
    "2. Hunt for dangling references — 'this', 'that', 'the previous equation' with no antecedent "
    "inside the clip — and for concepts the clip assumes but never introduces (and that are not "
    "common knowledge).\n"
    "3. If the clip is a worked problem, check the question is stated, the reasoning shown, and "
    "the result reached.\n"
    "Score bands for score_10 (integer 1-10):\n"
    "1-2: incomprehensible without the source video. 3-4: major gaps — the topic is unclear OR a "
    "key reference/prerequisite is missing. 5-6: partially followable — topic clear but real gaps "
    "remain. 7-8: fully understandable with minor rough edges. 9-10: flawlessly self-contained.\n"
    "First write `reasoning`: 2-3 sentences applying the steps to THIS clip. Then set every "
    "boolean truthfully and score_10 per the bands:\n"
    "- topic_identifiable / purpose_identifiable: is it clear what this is about and why?\n"
    "- all_references_resolved: no dangling 'this/that/the previous equation' without an antecedent.\n"
    "- prerequisites_satisfied: it doesn't assume a concept it never introduces or that isn't common.\n"
    "- visuals_sufficient: you cannot see the frames — set TRUE unless the transcript explicitly leans "
    "on an unshown, undescribed visual that is essential to follow it.\n"
    "- problem_statement_complete / reasoning_complete / result_complete: for a worked problem, is the "
    "question stated, the reasoning shown, and the result reached? (Set true/NA if not a problem.)\n"
    "- source_grounded: the clip stands on its own words (not obviously mid-argument).\n"
    "Give an overall 'understandable' boolean. For each problem add a failure_reason whose 'kind' "
    "is EXACTLY one of: unresolved_reference, missing_prerequisite, missing_visual, "
    "missing_problem_statement, missing_reasoning, missing_result, not_source_grounded, off_topic, "
    "other — and set missing_concept (for a prerequisite) or reference_text (the dangling phrase) "
    "when relevant. Never set 'error' or 'score' yourself; emit score_10 only. "
    "Output only the structured result."
)
```

d) In `judge_clip`, force `error=False` on every successful parse and replace the final fallback. The try/except block becomes:

```python
    try:
        v = llm_json(JUDGE_SYSTEM, user, JudgeVerdict, temperature=0.0, provider=provider, model=model)
        v.error = False                      # the LLM can never self-flag a transport error
        return v
    except Exception:
        if provider is not None or model is not None:  # cross-model judge failed → authoring model
            try:
                v = llm_json(JUDGE_SYSTEM, user, JudgeVerdict, temperature=0.0)
                v.error = False
                return v
            except Exception:
                pass
        # judge unavailable after retries: honest error verdict — never a free pass (was 0.7)
        return JudgeVerdict(error=True, understandable=False, score=0.0,
                            reasoning="judge unavailable")
```

e) `is_complete` gains the error guard as its first line:

```python
def is_complete(v: JudgeVerdict, role: str, adapter, min_score: float) -> bool:
    if v.error:
        return False
    required = adapter.required_verdict_fields(role)
    return v.score >= min_score and all(getattr(v, f, True) for f in required)
```

f) In `validate_and_repair`: (1) don't cache error verdicts — replace the verdict-fetch block:

```python
        key = frozenset(cand.unit_ids)
        verdict = _cached(key)
        if verdict is None:
            verdict = validate_candidate(cand, sentences, adapter,
                                         visual_summary_fn(cand.start, cand.end), topic)
            if not verdict.error:            # an outage verdict must not poison the cache
                _store(key, verdict)
        cand.verdict = verdict
        cand.attempts = attempt + 1

        if verdict.error:
            return cand                      # ship-but-flag: keep the clip, skip repair entirely
```

(the `if verdict.error: return cand` lines are NEW, inserted immediately after `cand.attempts = attempt + 1` and before the `is_complete` check).

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest backend/pipeline/assemble/tests/test_judge_rubric.py -q`
Expected: 10 passed.

- [ ] **Step 6: Checkpoint**

Run: `.venv/bin/python -m pytest backend -q && .venv/bin/python -m compileall -q backend`
Expected: 98 passed (88 + 10), compile clean.

---

### Task 2: Scoring neutrality + unjudged-warning plumbing

**Files:**
- Modify: `backend/pipeline/assemble/scoring.py` (`completeness_score`, `boundary_score`)
- Modify: `backend/pipeline/assemble/boundary_adapt.py` (`candidate_to_boundary_input`, `snap_candidates`)
- Test: append to `backend/pipeline/assemble/tests/test_judge_rubric.py`

**Interfaces:**
- Consumes: `verdict.error` (Task 1); existing `Candidate`, `snap_candidates(cands, sentences, settings) -> list[dict]`.
- Produces: `completeness_score` → 0.5 on error verdicts; snap dicts carry `"judge_error": bool` and error clips have `"unjudged"` appended to `warnings`; `boundary_score` penalizes `"unjudged"` by 0.15.

- [ ] **Step 1: Write the failing tests** (append to test_judge_rubric.py)

```python
# ── ship-but-flag scoring + warning plumbing ─────────────────────────────────
def test_completeness_neutral_on_error():
    from backend.pipeline.assemble.scoring import completeness_score

    class Contracty(FakeAdapter):
        def required_verdict_fields(self, role):
            return ["result_complete"]
    v = JudgeVerdict(error=True)
    assert completeness_score(v, "explanation", Contracty()) == pytest.approx(0.5)


def test_boundary_score_penalizes_unjudged():
    from backend.pipeline.assemble.scoring import boundary_score
    assert boundary_score(["unjudged"]) == pytest.approx(0.85)
    assert boundary_score([]) == pytest.approx(1.0)


def test_snap_flags_and_warns_unjudged():
    from backend.pipeline.assemble.boundary_adapt import snap_candidates

    sents = mini_sents(3)
    ok = _mk_candidate(sents)
    ok.verdict = JudgeVerdict(score_10=9, understandable=True)
    bad = _mk_candidate(sents)
    bad.cand_id, bad.i_start, bad.i_end = "c1", 1, 2
    bad.start, bad.end = sents[1].start, sents[2].end
    bad.verdict = JudgeVerdict(error=True)

    specs = snap_candidates([ok, bad], sents, {"min_clip_duration_s": 1.0,
                                               "max_clip_duration_s": 500.0})
    flagged = {s["cand_id"]: s for s in specs}
    assert flagged["c1"]["judge_error"] is True
    assert "unjudged" in (flagged["c1"].get("warnings") or [])
    assert flagged["c0"]["judge_error"] is False
    assert "unjudged" not in (flagged["c0"].get("warnings") or [])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest backend/pipeline/assemble/tests/test_judge_rubric.py -q`
Expected: the 3 new tests FAIL (`judge_error` KeyError; completeness ≠ 0.5; boundary_score == 1.0).

- [ ] **Step 3: Implement**

In `scoring.py::completeness_score`, add before the `req` line:

```python
    if getattr(verdict, "error", False):
        return 0.5                # unjudged (ship-but-flag): below judged-good, above judged-bad
```

In `scoring.py::boundary_score`, add to the penalty sum:

```python
               + 0.15 * ("unjudged" in w)
```

In `boundary_adapt.py::candidate_to_boundary_input`, add to the returned dict:

```python
        "judge_error": bool(getattr(cand.verdict, "error", False)),
```

In `boundary_adapt.py::snap_candidates`, after the `specs.append({**b, **snapped})` loop and BEFORE `_dedupe`, insert:

```python
    for s in specs:
        if s.get("judge_error"):   # unjudged (judge outage): user-visible + boundary_score penalty
            s["warnings"] = list(s.get("warnings") or []) + ["unjudged"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest backend/pipeline/assemble/tests/ -q`
Expected: 13 passed.

- [ ] **Step 5: Checkpoint**

Run: `.venv/bin/python -m pytest backend -q && .venv/bin/python -m compileall -q backend`
Expected: 101 passed, compile clean.

---

### Task 3: Eval exclusion + judge_error_rate

**Files:**
- Modify: `backend/eval/metrics.py` (`comprehension`, `judge_failures`)
- Modify: `backend/eval/run_eval.py` (`_measure`)
- Test: `backend/eval/tests/test_judge_metrics.py` (new file in the existing eval tests package)

**Interfaces:**
- Consumes: `verdict.error` (Task 1); `metrics.judge_clip` (module-level import in metrics.py — monkeypatchable as `backend.eval.metrics.judge_clip`).
- Produces: `comprehension(specs, sentences, adapter, topic, threshold=0.7) -> tuple[float, float, int, int]` (mean over judged, rate over judged, n_judged, n_error); `judge_failures` tuples gain a 4th element `error: bool`; `_measure` adds `"judge_error_rate"` and unpacks the 4-tuple.

- [ ] **Step 1: Write the failing test**

```python
# backend/eval/tests/test_judge_metrics.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest backend/eval/tests/test_judge_metrics.py -q`
Expected: FAIL — `comprehension` returns a 2-tuple; `judge_failures` rows are 3-tuples.

- [ ] **Step 3: Implement**

Replace `metrics.comprehension`:

```python
def comprehension(specs, sentences, adapter, topic, threshold=0.7) -> tuple[float, float, int, int]:
    """(mean judge score, fraction ≥ threshold, n_judged, n_error). Clip-only — the headline.
    Error verdicts (judge outage) are EXCLUDED from mean/rate so outages can never inflate
    comprehension; the caller reports them via judge_error_rate."""
    if not specs:
        return 0.0, 0.0, 0, 0
    scores, ok, n_error = [], 0, 0
    for s in specs:
        v = judge_clip(_clip_text(s, sentences), s.get("role", ""), adapter,
                       visual_summary="", topic=topic, context_card=s.get("context_card", ""))
        if v.error:
            n_error += 1
            continue
        scores.append(v.score)
        ok += 1 if v.score >= threshold else 0
    if not scores:
        return 0.0, 0.0, 0, n_error
    return sum(scores) / len(scores), ok / len(scores), len(scores), n_error
```

In `metrics.judge_failures`, change the append to a 4-tuple:

```python
        out.append((s.get("role", ""), round(v.score, 2), [f.kind for f in v.failure_reasons],
                    bool(v.error)))
```

In `run_eval._measure`: change the comprehension lines to

```python
    mean_score, comp_rate, n_judged, n_err = metrics.comprehension(specs, sents, adapter, topic, thr)
```

add to the metrics dict (after `"mean_judge_score"`):

```python
        "judge_error_rate": round(n_err / (n_judged + n_err), 3) if (n_judged + n_err) else 0.0,
```

and update the `--verbose` loop to unpack 4-tuples:

```python
        for role, score, fails, err in metrics.judge_failures(specs, sents, adapter, topic):
            tag = " UNJUDGED" if err else ""
            print(f"      [{role or '-':20}] score={score}  fails={fails}{tag}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest backend/eval/tests/ -q`
Expected: 28 passed (24 prior + 4 new).

- [ ] **Step 5: Checkpoint**

Run: `.venv/bin/python -m pytest backend -q && .venv/bin/python -m compileall -q backend`
Expected: 105 passed, compile clean.

---

### Task 4: Live verification + A/B vs old-judge baseline (controller-run)

**Files:** none (measurement + docs only).

- [ ] **Step 1: Reasoning-first live check** — one real judge call, inspect raw ordering:

```bash
cd /Users/vincentfeng/Documents/practice/clips && .venv/bin/python - <<'EOF'
from backend.pipeline.assemble.validate import judge_clip
from backend.adapters.generic import GenericAdapter
v = judge_clip("So this gives us the same answer as before, which confirms the theorem.",
               "explanation", GenericAdapter(), topic="the theorem")
print("reasoning:", repr(v.reasoning[:200]))
print("score_10:", v.score_10, "score:", v.score, "error:", v.error)
EOF
```

Expected: non-empty `reasoning`; `score_10` in 1-10; `score == score_10/10`; `error False`.
(If reasoning comes back empty, google-genai did not honor field order — fall back per spec:
strengthen the prompt's reasoning-first instruction and re-test; the schema order stays.)

- [ ] **Step 2: NEW-judge frozen-specs runs** (baseline with the OLD judge was captured
  BEFORE this plan landed — scratchpad `judge-baseline-old.log`):

```bash
.venv/bin/python -m backend.eval.run_eval uqwC41RDPyg NjvwWiCYLl4 yfajBIaDf1Q --freeze-specs --runs 3 --verbose
```

Compare against the old-judge baseline: score distribution spread (expect wider than the
0.7-cluster), run-to-run flip rate (expect ≤ old), comprehension delta (report, don't over-read),
judge_error_rate (expect 0.0 in a healthy run).

- [ ] **Step 3: Docs** — RESEARCH.md roadmap row "Judge rubric+CoT, 1-10 scale" → shipped (+ the
  measured numbers); clipper memory gains the judge-rubric entry incl. the ship-but-flag policy
  and the removed 0.7 fallback.

---

## Self-Review (done)

- **Spec coverage:** verdict schema/prompt/fallback → Task 1; is_complete + repair early-return +
  no-cache-on-error → Task 1; completeness 0.5 + unjudged warning + boundary penalty (spec's
  "penalized like other warnings" made concrete at 0.15) → Task 2; snap plumbing (moved from
  assemble/__init__ to snap_candidates — same semantics, standalone-testable seam; warnings exist
  before boundary_score reads them in assemble step 5) → Task 2; eval exclusion + judge_error_rate
  + verbose annotation → Task 3; live order check + A/B + docs → Task 4. No gaps.
- **Placeholders:** none — full code in every step.
- **Type consistency:** `comprehension` 4-tuple matches `_measure` unpack; `judge_failures`
  4-tuple matches the verbose loop; `JudgeVerdict.error` consumed via attribute everywhere;
  `judge_error` dict key consistent between boundary_adapt and its test.
