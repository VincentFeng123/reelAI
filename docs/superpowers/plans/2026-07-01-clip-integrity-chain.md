# Clip Integrity Chain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make shipped clips provably consistent with what the judge approved — truthful unit_ids at every span change, re-judged merges, warned mutations, a drop-reason ledger for every silently-discarded candidate, and cards that never fail silently.

**Architecture:** New pure module `assemble/integrity.py` (span-truth helpers + Part-B merge + `Rejection`). `boundary_adapt` gets its own metadata-aware dedupe (shared `refine.py` keeps only ADDITIVE warning strings — its merge/dedupe behavior is untouched because the legacy fast path uses it). `assemble_clips` re-judges merged spans (hybrid policy), collects `Rejection`s at all six stages, and returns them as a third tuple element consumed by the CLI/orchestrator/eval.

**Tech Stack:** Python 3.12, pydantic 2, pytest 9.1.1, mocked `llm_json` (offline tests).

**Spec:** `docs/superpowers/specs/2026-07-01-clip-integrity-chain-design.md`

## Global Constraints

- Run from `/Users/vincentfeng/Documents/practice/clips` with `.venv/bin/python`. **No git repo** — Checkpoint steps (full pytest + compileall) replace commits.
- Tests offline: no network/LLM; monkeypatch `backend.llm.llm_json` (call-time lazy import in judge_clip / card generator).
- `refine.py` and `boundary.py` may gain ONLY additive warning strings; keep/drop/merge decisions there are frozen (legacy fast path).
- Rejection stages, exact strings: `"repair"`, `"snap"`, `"dedupe"`, `"post_merge_judge"`, `"quality_floor"`, `"max_clips"`.
- New warning strings, exact: `"merged_overlap"`, `"extended_for_min_duration"`, `"trimmed_start"`, `"missing_context_card"`; boundary_score penalty 0.10 each (existing penalties and `"unjudged"` 0.15 unchanged).
- Hybrid re-judge: ONLY specs with `"merged": True` get a re-judge; outage → ship-but-flag (existing `error` semantics); hard-core gate = `topic_identifiable ∧ purpose_identifiable ∧ source_grounded ∧ all_references_resolved`.
- Suite baseline before this plan: 106 passed.

---

### Task 1: `integrity.py` — pure helpers + Rejection

**Files:**
- Create: `backend/pipeline/assemble/integrity.py`
- Test: `backend/pipeline/assemble/tests/test_integrity.py` (uses existing `tests/conftest.py` fixtures `mini_sents`, `mini_units`)

**Interfaces:**
- Produces:
  - `true_contents(unit_ids: list[str], referential: list[tuple[str, str]], units: list, i_start: int, i_end: int) -> tuple[list[str], list[tuple[str, str]]]`
  - `merge_partb(a: dict, b: dict, units: list | None, sentences: list) -> dict`
  - `@dataclass Rejection(cand_id, title, role, stage, reason, score, failure_kinds, final_quality, start, end)`

- [ ] **Step 1: Write the failing tests**

```python
# backend/pipeline/assemble/tests/test_integrity.py
"""integrity helpers: truthful unit_ids, Part-B merge, Rejection. Pure/offline."""
from __future__ import annotations

import pytest

from backend.pipeline.assemble.integrity import Rejection, merge_partb, true_contents

from .conftest import mini_sents, mini_units


def _setup(n=6):
    sents = mini_sents(n)
    units = mini_units(sents)          # unit i ↔ sentence i, ids u0000..u000{n-1}
    return sents, units


# ── true_contents ─────────────────────────────────────────────────────────────
def test_absorbs_spanned_units_time_ordered():
    sents, units = _setup()
    ids, ref = true_contents(["u0000", "u0003"], [], units, 0, 3)
    assert ids == ["u0000", "u0001", "u0002", "u0003"]   # gap units absorbed, ordered
    assert ref == []


def test_drops_referential_now_inside_span():
    sents, units = _setup()
    ids, ref = true_contents(["u0000", "u0002"], [("u0001", "prerequisite"), ("u0005", "prerequisite")],
                             units, 0, 2)
    assert "u0001" in ids                                # was referential, now in-span → absorbed
    assert ref == [("u0005", "prerequisite")]            # outside span → untouched


def test_idempotent():
    sents, units = _setup()
    once = true_contents(["u0000", "u0003"], [("u0005", "x")], units, 0, 3)
    twice = true_contents(once[0], once[1], units, 0, 3)
    assert once == twice


def test_partial_overlap_unit_not_absorbed():
    sents, units = _setup()
    # unit u0003 spans sentence 3 only; span [0,2] must not absorb it
    ids, _ = true_contents(["u0000"], [], units, 0, 2)
    assert "u0003" not in ids


# ── merge_partb ───────────────────────────────────────────────────────────────
def _spec(cand_id, s0, s1, sents, *, fq=0.5, facet="other", **extra):
    d = {"cand_id": cand_id, "facet": facet, "role": "explanation", "title": f"t-{cand_id}",
         "anchor_id": "u0000", "unit_ids": [f"u{i:04d}" for i in range(s0, s1 + 1)],
         "referential": [], "start": sents[s0].start, "end": sents[s1].end,
         "cut_end": sents[s1].end, "sentence_start_idx": s0, "sentence_end_idx": s1,
         "score": fq, "final_quality": fq, "warnings": ("w_a",) if cand_id == "a" else ("w_b",),
         "judge_error": False, "truncated": False, "context_card": ""}
    d.update(extra)
    return d


def test_merge_unions_span_ids_referential_warnings():
    sents, units = _setup()
    a = _spec("a", 0, 2, sents, fq=0.9, referential=[("u0005", "prerequisite")])
    b = _spec("b", 2, 4, sents, fq=0.4)
    m = merge_partb(a, b, units, sents)
    assert m["sentence_start_idx"] == 0 and m["sentence_end_idx"] == 4
    assert m["start"] == sents[0].start and m["end"] == sents[4].end
    assert m["unit_ids"] == [f"u{i:04d}" for i in range(5)]      # union + absorption
    assert m["referential"] == [("u0005", "prerequisite")]        # still outside span
    assert set(m["warnings"]) >= {"w_a", "w_b", "merged_overlap"}
    assert m["merged"] is True
    assert m["title"] == "t-a"                                    # winner metadata (higher fq)


def test_merge_ors_flags_and_keeps_winner_metadata():
    sents, units = _setup()
    a = _spec("a", 0, 1, sents, fq=0.3, judge_error=True)
    b = _spec("b", 1, 2, sents, fq=0.8, truncated=True)
    m = merge_partb(a, b, units, sents)
    assert m["judge_error"] is True and m["truncated"] is True    # OR'd
    assert m["title"] == "t-b"                                    # b wins on final_quality


def test_merge_without_units_skips_absorption():
    sents, _ = _setup()
    a = _spec("a", 0, 0, sents, fq=0.9)
    b = _spec("b", 3, 3, sents, fq=0.1)
    m = merge_partb(a, b, None, sents)
    assert m["unit_ids"] == ["u0000", "u0003"]                    # union only, no sweep


# ── Rejection ─────────────────────────────────────────────────────────────────
def test_rejection_dataclass_shape():
    r = Rejection(cand_id="c1", title="t", role="claim", stage="dedupe",
                  reason="overlap loser to c0", score=0.4, failure_kinds=["off_topic"],
                  final_quality=0.3, start=1.0, end=2.0)
    assert r.stage == "dedupe" and r.failure_kinds == ["off_topic"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest backend/pipeline/assemble/tests/test_integrity.py -q`
Expected: FAIL — `ModuleNotFoundError: integrity`.

- [ ] **Step 3: Implement `integrity.py`**

```python
# backend/pipeline/assemble/integrity.py
"""Clip-integrity helpers (audit pkg 1): keep unit_ids truthful at every span change,
merge Part-B specs without losing metadata, and record WHY candidates are dropped.

Pure module — no LLM, no I/O. `refine.py`'s merge/dedupe stays untouched (the legacy
fast path shares it); Part B routes overlaps through merge_partb instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Rejection:
    """One dropped candidate + the stage and reason — the assemble run's drop ledger."""
    cand_id: str
    title: str
    role: str
    stage: str            # repair | snap | dedupe | post_merge_judge | quality_floor | max_clips
    reason: str
    score: Optional[float] = None            # last judge score, when one exists
    failure_kinds: list[str] = field(default_factory=list)
    final_quality: Optional[float] = None
    start: float = 0.0
    end: float = 0.0


def true_contents(unit_ids, referential, units, i_start: int, i_end: int):
    """(unit_ids ∪ every unit whose sentence_range lies inside [i_start, i_end], time-ordered;
    referential minus entries now inside the span). Keeps unit_ids the TRUE clip contents so
    contract checks, grounding, repair targeting, and cards see what the viewer sees."""
    have = set(unit_ids)
    for u in units or []:
        s0, s1 = u.sentence_range
        if s0 >= i_start and s1 <= i_end:
            have.add(u.unit_id)
    by_id = {u.unit_id: u for u in units or []}
    ordered = sorted(have, key=lambda uid: (by_id[uid].start if uid in by_id else 0.0, uid))
    new_ref = [(uid, rel) for uid, rel in referential if uid not in have]
    return ordered, new_ref


def merge_partb(a: dict, b: dict, units, sentences) -> dict:
    """Union two overlapping same-facet Part-B specs WITHOUT losing metadata: span/ids/
    referential/warnings union, flags OR'd, other keys from the higher-final_quality side.
    Marks the result merged — the caller MUST re-judge it (its text was never judged)."""
    winner = a if a.get("final_quality", a.get("score", 0.0)) >= b.get("final_quality", b.get("score", 0.0)) else b
    s0 = min(a["sentence_start_idx"], b["sentence_start_idx"])
    s1 = max(a["sentence_end_idx"], b["sentence_end_idx"])
    ids = list(dict.fromkeys(list(a.get("unit_ids", [])) + list(b.get("unit_ids", []))))
    ref = list(dict.fromkeys(list(map(tuple, a.get("referential", []))) +
                             list(map(tuple, b.get("referential", [])))))
    ids, ref = true_contents(ids, ref, units, s0, s1) if units else (ids, [(u, r) for u, r in ref if u not in set(ids)])
    return {
        **winner,
        "start": min(a["start"], b["start"]),
        "end": max(a["end"], b["end"]),
        "cut_end": max(a.get("cut_end", a["end"]), b.get("cut_end", b["end"])),
        "sentence_start_idx": s0,
        "sentence_end_idx": s1,
        "unit_ids": ids,
        "referential": ref,
        "warnings": tuple(set(a.get("warnings") or ()) | set(b.get("warnings") or ()) | {"merged_overlap"}),
        "judge_error": bool(a.get("judge_error")) or bool(b.get("judge_error")),
        "truncated": bool(a.get("truncated")) or bool(b.get("truncated")),
        "merged": True,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest backend/pipeline/assemble/tests/test_integrity.py -q`
Expected: 9 passed.

- [ ] **Step 5: Checkpoint**

Run: `.venv/bin/python -m pytest backend -q && .venv/bin/python -m compileall -q backend`
Expected: 115 passed, compile clean.

---

### Task 2: Truthful unit_ids at build/repair + repair Rejections

**Files:**
- Modify: `backend/pipeline/assemble/candidates.py` (`build_candidate`, ~lines 88-109)
- Modify: `backend/pipeline/assemble/validate.py` (`expand_candidate` final replace ~line 262; `validate_and_repair` return contract)
- Modify: `backend/pipeline/assemble/__init__.py` (`_assemble_one` unpack; internal rejection list — NOT yet returned)
- Modify: `backend/pipeline/assemble/tests/test_judge_rubric.py` (repair test unpack)
- Test: append to `backend/pipeline/assemble/tests/test_integrity.py`

**Interfaces:**
- Consumes: Task 1's `true_contents`, `Rejection`.
- Produces: `validate_and_repair(...) -> tuple[Optional[Candidate], Optional[Rejection]]` — `(cand, None)` on keep, `(None, Rejection(stage="repair", …))` on drop, carrying the LAST verdict's score + failure kinds; `build_candidate`/`expand_candidate` always emit `true_contents`-consistent `unit_ids`/`referential`.

- [ ] **Step 1: Write the failing tests** (append to `test_integrity.py`)

```python
# ── truthful ids at build/repair + repair rejection ──────────────────────────
import threading

import backend.llm as llm_mod
from backend.pipeline.assemble.graph import Graph
from backend.pipeline.assemble.types import Candidate
from backend.pipeline.assemble.validate import JudgeVerdict, expand_candidate, validate_and_repair

from .conftest import FakeAdapter


def test_expand_candidate_emits_true_contents():
    sents, units = _setup()
    units_by_id = {u.unit_id: u for u in units}
    cand = Candidate(cand_id="c0", anchor_id="u0003", role="explanation", facet="other",
                     title="t", reason="r", unit_ids=["u0003"],
                     referential=[("u0000", "prerequisite")], i_start=3, i_end=3,
                     start=sents[3].start, end=sents[3].end)
    verdict = JudgeVerdict(prerequisites_satisfied=False, score_10=3)
    grown = expand_candidate(cand, verdict, Graph([], units), units, units_by_id,
                             {}, sents, max_span_s=999.0)
    assert grown is not None
    # u0000 pulled in → span [0,3] → u0001/u0002 absorbed, referential emptied
    assert grown.unit_ids == ["u0000", "u0001", "u0002", "u0003"]
    assert grown.referential == []


def test_repair_drop_returns_rejection_with_last_verdict(monkeypatch):
    from backend import config
    monkeypatch.setattr(config, "JUDGE_MODEL", "")
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "same")

    def bad_verdict(*a, **kw):     # judged, failing, unrepairable (no targets: refs fine, topic bad)
        return JudgeVerdict(reasoning="bad", score_10=2, understandable=False,
                            topic_identifiable=False, purpose_identifiable=False)
    monkeypatch.setattr(llm_mod, "llm_json", bad_verdict)

    sents, units = _setup()
    units_by_id = {u.unit_id: u for u in units}
    cand = Candidate(cand_id="c0", anchor_id="u0000", role="explanation", facet="other",
                     title="T", reason="r", unit_ids=["u0000"], referential=[],
                     i_start=0, i_end=0, start=sents[0].start, end=sents[0].end)
    kept, rej = validate_and_repair(cand, sents, Graph([], units), units, units_by_id, {},
                                    FakeAdapter(), {}, lambda s, e: "", "topic", {}, threading.Lock())
    assert kept is None
    assert rej is not None and rej.stage == "repair"
    assert rej.score == pytest.approx(0.2)
    assert rej.cand_id == "c0" and rej.title == "T"


def test_repair_keep_returns_no_rejection(monkeypatch):
    from backend import config
    monkeypatch.setattr(config, "JUDGE_MODEL", "")
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "same")
    monkeypatch.setattr(llm_mod, "llm_json",
                        lambda *a, **kw: JudgeVerdict(reasoning="ok", score_10=9, understandable=True))
    sents, units = _setup()
    units_by_id = {u.unit_id: u for u in units}
    cand = Candidate(cand_id="c0", anchor_id="u0000", role="explanation", facet="other",
                     title="t", reason="r", unit_ids=["u0000"], referential=[],
                     i_start=0, i_end=0, start=sents[0].start, end=sents[0].end)
    kept, rej = validate_and_repair(cand, sents, Graph([], units), units, units_by_id, {},
                                    FakeAdapter(), {}, lambda s, e: "", "topic", {}, threading.Lock())
    assert kept is not None and rej is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest backend/pipeline/assemble/tests/test_integrity.py -q`
Expected: new tests FAIL — expand doesn't absorb; validate_and_repair returns a single value.

- [ ] **Step 3: Implement**

a) `candidates.py::build_candidate` — replace the final `return Candidate(...)` block's id fields:

```python
    from .integrity import true_contents
    unit_ids, referential = true_contents(list(cl.unit_ids), list(cl.referential),
                                          units, i_start, i_end)
    return Candidate(
        cand_id=f"c_{anchor.unit_id}",
        anchor_id=anchor.unit_id,
        role=anchor.role,
        facet=adapter.facet_for(anchor.role),
        title=anchor.topic or (anchor.summary[:60]),
        reason=anchor.summary,
        unit_ids=unit_ids,
        referential=referential,
        i_start=i_start, i_end=i_end,
        start=float(sentences[i_start].start), end=float(sentences[i_end].end),
        relevance=relevance.get(anchor.unit_id, 0.0),
        priority=adapter.anchor_priority(anchor.role),
        truncated=cl.truncated,
    )
```

(move the `from .integrity import true_contents` to the module imports.)

b) `validate.py::expand_candidate` — the final replace becomes:

```python
    new_ids = sorted(chosen, key=lambda x: (units_by_id[x].start, order.get(x, 0)))
    i0, i1 = _reindex(new_ids, units_by_id, sentences)
    from .integrity import true_contents
    new_ids, new_ref = true_contents(new_ids, cand.referential, units, i0, i1)
    return replace(cand, unit_ids=new_ids, referential=new_ref, i_start=i0, i_end=i1,
                   start=sentences[i0].start, end=sentences[i1].end)
```

c) `validate.py::validate_and_repair` — change the return contract. Signature line:

```python
def validate_and_repair(cand: Candidate, sentences: list[Sentence], graph, units: list[Unit],
                        units_by_id: dict[str, Unit], introducers: dict[str, list[str]], adapter,
                        settings: dict, visual_summary_fn, topic: str,
                        cache: dict, cache_lock=None) -> tuple[Optional[Candidate], Optional["Rejection"]]:
```

Every existing `return cand` / `return best` becomes `return cand, None` / `return best, None`
(three sites: the is_complete return, the error-verdict early return, the best-partial return).
The final `return None` becomes:

```python
    from .integrity import Rejection
    last = best.verdict if best is not None else getattr(cand, "verdict", None)
    return None, Rejection(
        cand_id=cand.cand_id, title=cand.title, role=cand.role, stage="repair",
        reason="judge verdict incomplete after repair budget",
        score=(float(last.score) if last is not None else None),
        failure_kinds=[f.kind for f in (last.failure_reasons if last is not None else [])],
        final_quality=None, start=cand.start, end=cand.end)
```

NOTE the best-partial gate: when `best` exists but fails the hard-core gate, execution falls
through to this same Rejection return (previously `return None`).

d) `assemble/__init__.py::_assemble_one` and its collection loop — unpack the tuple and stash
rejections in a thread-safe list (internal only in this task; returned in Task 4):

```python
    rejections: list = []
    rej_lock = threading.Lock()

    def _assemble_one(anchor):
        cand = build_candidate(anchor, graph, adapter, units, units_by_id, sentences, relevance, settings)
        if cand is None:
            return None
        cand, rejection = validate_and_repair(cand, sentences, graph, units, units_by_id, introducers,
                                              adapter, settings, structure.visual_summary, topic,
                                              cache, cache_lock)
        if rejection is not None:
            with rej_lock:
                rejections.append(rejection)
        if cand is None:
            return None
        cand.completeness_score = scoring.completeness_score(cand.verdict, cand.role, adapter)
        cand.grounding_score = scoring.grounding_score(cand, units_by_id)
        cand.final_quality = scoring.final_quality(cand)
        return cand
```

e) `tests/test_judge_rubric.py::test_repair_returns_candidate_on_error_without_burning_budget` —
the call becomes `cand, rej = validate_and_repair(...)`; add `assert rej is None` after the
existing assertions (error path is ship-but-flag: kept, not rejected).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest backend/pipeline/assemble/tests/ -q`
Expected: 26 passed (14 rubric + 12 integrity).

- [ ] **Step 5: Checkpoint**

Run: `.venv/bin/python -m pytest backend -q && .venv/bin/python -m compileall -q backend`
Expected: 118 passed, compile clean.

---

### Task 3: Part-B dedupe + snap rejections + additive mutation warnings

**Files:**
- Modify: `backend/pipeline/assemble/boundary_adapt.py` (snap_candidates signature + `_dedupe_partb`; `candidate_to_boundary_input` carries `truncated`)
- Modify: `backend/pipeline/refine.py` (two additive warnings ONLY)
- Modify: `backend/pipeline/boundary.py::_resolve_overlaps` (one additive warning)
- Modify: `backend/pipeline/assemble/scoring.py::boundary_score` (three new penalties)
- Modify: `backend/pipeline/assemble/tests/test_judge_rubric.py::test_snap_flags_and_warns_unjudged` (new snap return shape)
- Test: append to `backend/pipeline/assemble/tests/test_integrity.py`

**Interfaces:**
- Consumes: Task 1's `merge_partb`, `Rejection`; refine's `_better`, `_trim_start_after`, `NEAR_DUP_EPS` (imported, unmodified).
- Produces: `snap_candidates(cands, sentences, settings, units=None) -> tuple[list[dict], list[Rejection]]`; specs may carry `"merged": True` and `"truncated": bool`; warnings `"extended_for_min_duration"`/`"trimmed_start"` appear in shared snap output.

- [ ] **Step 1: Write the failing tests** (append to `test_integrity.py`)

```python
# ── Part-B dedupe + snap rejections + mutation warnings ──────────────────────
from backend.pipeline.assemble.boundary_adapt import snap_candidates
from backend.pipeline.refine import _dedupe as refine_dedupe


def _cand(cand_id, s0, s1, sents, *, facet="other", fq=0.5, verdict=None):
    c = Candidate(cand_id=cand_id, anchor_id=f"u{s0:04d}", role="explanation", facet=facet,
                  title=f"t-{cand_id}", reason="r", unit_ids=[f"u{i:04d}" for i in range(s0, s1 + 1)],
                  referential=[], i_start=s0, i_end=s1,
                  start=sents[s0].start, end=sents[s1].end)
    c.final_quality = fq
    c.verdict = verdict or JudgeVerdict(score_10=8, understandable=True)
    return c


_SETTINGS = {"min_clip_duration_s": 1.0, "max_clip_duration_s": 500.0}


def test_same_facet_overlap_merges_and_flags():
    sents, units = _setup(8)
    a = _cand("a", 0, 3, sents, fq=0.9)
    b = _cand("b", 2, 5, sents, fq=0.4)              # overlaps a, same facet
    specs, rejections = snap_candidates([a, b], sents, _SETTINGS, units)
    assert len(specs) == 1 and specs[0]["merged"] is True
    assert specs[0]["sentence_start_idx"] == 0 and specs[0]["sentence_end_idx"] == 5
    assert specs[0]["unit_ids"] == [f"u{i:04d}" for i in range(6)]
    assert "merged_overlap" in specs[0]["warnings"]
    assert rejections == []                           # merge is not a drop


def test_containment_loser_becomes_dedupe_rejection():
    sents, units = _setup(8)
    big = _cand("big", 0, 5, sents, fq=0.9)
    small = _cand("small", 1, 2, sents, fq=0.2)      # contained in big
    specs, rejections = snap_candidates([big, small], sents, _SETTINGS, units)
    assert [s["cand_id"] for s in specs] == ["big"]
    assert len(rejections) == 1
    r = rejections[0]
    assert r.stage == "dedupe" and r.cand_id == "small" and "big" in r.reason


def test_dedupe_partb_matches_refine_on_disjoint_and_containment():
    """Characterization: identical keep decisions as refine._dedupe on non-merge shapes."""
    sents, units = _setup(10)
    disjoint = [_cand("a", 0, 1, sents, fq=0.5), _cand("b", 4, 5, sents, fq=0.5)]
    contained = [_cand("c", 0, 5, sents, fq=0.9), _cand("d", 2, 3, sents, fq=0.1)]
    for group in (disjoint, contained):
        specs, _ = snap_candidates(list(group), sents, _SETTINGS, units)
        legacy_in = [snap_candidates([c], sents, _SETTINGS, units)[0][0] for c in group]
        legacy = refine_dedupe([dict(s) for s in legacy_in], sents, 1.0)
        assert sorted(s["start"] for s in specs) == sorted(c["start"] for c in legacy)


def test_truncated_carried_into_spec():
    sents, units = _setup()
    c = _cand("a", 0, 2, sents)
    c.truncated = True
    specs, _ = snap_candidates([c], sents, _SETTINGS, units)
    assert specs[0]["truncated"] is True


def test_boundary_score_penalizes_mutation_warnings():
    from backend.pipeline.assemble.scoring import boundary_score
    assert boundary_score(["extended_for_min_duration"]) == pytest.approx(0.90)
    assert boundary_score(["trimmed_start"]) == pytest.approx(0.90)
    assert boundary_score(["missing_context_card"]) == pytest.approx(0.90)


def test_snap_one_warns_on_min_duration_extension():
    from backend.pipeline.refine import _snap_one
    sents, _ = _setup(6)                              # each sentence ≈10 s
    clip = _snap_one({"i_start": 0, "i_end": 0, "facet": "other"}, sents,
                     False, 25.0, 0.05, 500.0)        # min_dur 25 s forces extension past s0
    assert clip is not None
    assert "extended_for_min_duration" in clip["warnings"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest backend/pipeline/assemble/tests/test_integrity.py -q`
Expected: new tests FAIL — snap_candidates returns a list (not tuple), no `merged`/`truncated`
keys, boundary_score returns 1.0, no extension warning.

- [ ] **Step 3: Implement**

a) `refine.py::_snap_one` — in the min-duration loop ONLY (additive):

```python
    # enforce minimum duration by extending the end outward to full period sentences
    guard = 0
    extended = False
    while sentences[ei].end - sentences[si].start < min_dur and guard < n:
        nxt = _valid_end_at_or_after(sentences, ei + 1, allow_qe)
        if nxt is None or nxt == ei:
            warnings.append("min_duration_unreachable")
            break
        ei = nxt
        extended = True
        guard += 1
    if extended:
        warnings.append("extended_for_min_duration")   # content beyond the judged span
```

b) `refine.py::_trim_start_after` — additive warning on the successful-trim path:

```python
            if c["end"] - new_start >= min_dur:
                d = dict(c)
                d["start"] = round(new_start, 3)
                d["sentence_start_idx"] = i
                d["warnings"] = tuple(set(c.get("warnings") or ()) | {"trimmed_start"})
                return d
```

c) `boundary.py::_resolve_overlaps` — additive warning when a start is actually trimmed:

```python
    for c in clips:
        c = dict(c)
        if c["start"] < last_end:
            c["start"] = round(last_end, 3)
            c["warnings"] = tuple(set(c.get("warnings") or ()) | {"trimmed_start"})
        ...
```

d) `scoring.py::boundary_score` — extend the penalty sum:

```python
               + 0.10 * ("extended_for_min_duration" in w)
               + 0.10 * ("trimmed_start" in w)
               + 0.10 * ("missing_context_card" in w)
```

e) `boundary_adapt.py` — new signature + Part-B dedupe. `candidate_to_boundary_input` adds
`"truncated": bool(getattr(cand, "truncated", False)),` next to `judge_error`. Then:

```python
def snap_candidates(cands: list[Candidate], sentences: list[Sentence], settings: dict,
                    units=None) -> tuple[list[dict], list[Rejection]]:
    from ..refine import _snap_one
    allow_qe = bool(settings.get("allow_question_exclaim_ends", config.DEFAULTS["allow_question_exclaim_ends"]))
    min_dur = float(settings.get("min_clip_duration_s", config.DEFAULTS["min_clip_duration_s"]))
    max_dur = float(settings.get("max_clip_duration_s", config.DEFAULTS["max_clip_duration_s"]))
    tail_pad = float(settings.get("tail_pad_s", config.DEFAULTS["tail_pad_s"]))

    rejections: list[Rejection] = []
    specs: list[dict] = []
    for c in cands:
        b = candidate_to_boundary_input(c)
        snapped = _snap_one(b, sentences, allow_qe, min_dur, tail_pad, max_dur)
        if snapped is None:
            rejections.append(Rejection(cand_id=c.cand_id, title=c.title, role=c.role,
                                        stage="snap", reason="unsnappable (end<=start)",
                                        final_quality=c.final_quality, start=c.start, end=c.end))
            continue
        specs.append({**b, **snapped})
    for s in specs:
        if s.get("judge_error"):   # unjudged (judge outage): user-visible + boundary_score penalty
            s["warnings"] = list(s.get("warnings") or []) + ["unjudged"]
    specs, dedupe_rejections = _dedupe_partb(specs, sentences, min_dur, units)
    rejections.extend(dedupe_rejections)
    specs.sort(key=lambda s: s["start"])
    return specs, rejections


def _dedupe_partb(clips: list[dict], sentences: list[Sentence], min_dur: float,
                  units) -> tuple[list[dict], list[Rejection]]:
    """refine._dedupe's keep/drop decisions, but metadata-aware: same-facet overlaps merge via
    integrity.merge_partb (union + merged flag, re-judged upstream) and every dropped spec
    becomes a Rejection. refine.py itself is untouched (legacy fast path)."""
    from ..refine import NEAR_DUP_EPS, _better, _trim_start_after

    def _reject(loser: dict, winner: dict) -> None:
        rejections.append(Rejection(
            cand_id=loser.get("cand_id", ""), title=loser.get("title", ""),
            role=loser.get("role", ""), stage="dedupe",
            reason=f"overlap loser to {winner.get('cand_id', '?')}",
            final_quality=loser.get("final_quality"), start=loser["start"], end=loser["end"]))

    rejections: list[Rejection] = []
    clips = sorted(clips, key=lambda c: (c["start"], -c["end"]))
    kept: list[dict] = []
    for c in clips:
        if not kept:
            kept.append(c)
            continue
        k = kept[-1]
        if c["start"] >= k["end"]:                      # disjoint
            kept.append(c)
            continue
        overlapping_pairs = (
            (c["start"] >= k["start"] and c["end"] <= k["end"]) or
            (c["start"] <= k["start"] and c["end"] >= k["end"]) or
            (abs(c["start"] - k["start"]) <= NEAR_DUP_EPS and abs(c["end"] - k["end"]) <= NEAR_DUP_EPS)
        )
        if overlapping_pairs:                            # containment / near-dup → keep better
            winner = _better(k, c)
            _reject(c if winner is k else k, winner)
            kept[-1] = winner
            continue
        if c["facet"] == k["facet"]:                    # same facet → metadata-aware union
            kept[-1] = merge_partb(k, c, units, sentences)
            continue
        trimmed = _trim_start_after(c, k["end"], sentences, min_dur)
        if trimmed is not None:
            kept.append(trimmed)
        else:
            winner = _better(k, c)
            _reject(c if winner is k else k, winner)
            kept[-1] = winner
    return kept, rejections
```

(`from .integrity import Rejection, merge_partb` in module imports; drop the now-unused
`refine._dedupe` import.)

f) `tests/test_judge_rubric.py::test_snap_flags_and_warns_unjudged` — the call becomes
`specs, _rej = snap_candidates([ok, bad], sents, {...})` (units omitted → default None).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest backend/pipeline/assemble/tests/ -q`
Expected: 32 passed.

- [ ] **Step 5: Checkpoint**

Run: `.venv/bin/python -m pytest backend -q && .venv/bin/python -m compileall -q backend`
Expected: 124 passed, compile clean.

---

### Task 4: Re-judge hook + full ledger + 3-tuple return + caller updates

**Files:**
- Modify: `backend/pipeline/assemble/__init__.py` (steps 4-5-7 region + signature + docstring)
- Modify: `backend/cli.py:80` region (unpack + rejection lines)
- Modify: `backend/orchestrator.py:201-204` (unpack; rejections currently unused there — `_, `)
- Modify: `backend/eval/run_eval.py:260` region (unpack + per-stage counts into the run dict)
- Test: append to `backend/pipeline/assemble/tests/test_integrity.py`

**Interfaces:**
- Consumes: Tasks 1-3 (`Rejection`, snap tuple, `merged` flag, repair rejections list).
- Produces: `assemble_clips(...) -> tuple[list[dict], str, list[Rejection]]`; eval run dicts gain
  `rejections_repair, rejections_snap, rejections_dedupe, rejections_post_merge_judge,
  rejections_quality_floor, rejections_max_clips` (ints).

- [ ] **Step 1: Write the failing tests** (append to `test_integrity.py`)

```python
# ── re-judge hook + ledger stages (pure step-5 logic exercised via assemble_clips) ──
from backend.pipeline.understand.models import ContentMap, ContentNode, DependencyGraph, Structure


def _structure(sents, units):
    n = len(sents)
    return Structure(video_id="v", units=units, dependencies=DependencyGraph(),
                     content_map=ContentMap(nodes=[ContentNode(node_id="video", level="video",
                                                               sentence_range=(0, n - 1))]))


class AnchorAdapter(FakeAdapter):
    def is_anchor_role(self, role):
        return True

    def anchor_priority(self, role):
        return 0.9

    def facet_for(self, role):
        return "other"

    def valid_roles(self):
        return {"explanation"}


def test_assemble_returns_rejections_and_rejudges_merges(monkeypatch):
    from backend import config
    from backend.pipeline.assemble import assemble_clips
    monkeypatch.setattr(config, "JUDGE_MODEL", "")
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "same")
    calls = {"n": 0}

    def good(*a, **kw):
        sch = a[2] if len(a) > 2 else kw.get("schema")
        from backend.pipeline.assemble.validate import JudgeVerdict as JV
        if sch is JV or (a and "self-contained" in str(a[0])):
            calls["n"] += 1
            return JV(reasoning="ok", score_10=9, understandable=True)
        raise AssertionError("unexpected llm call")
    monkeypatch.setattr(llm_mod, "llm_json", good)

    sents, units = _setup(8)
    st = _structure(sents, units)
    settings = {"min_clip_duration_s": 1.0, "max_clip_duration_s": 500.0,
                "min_comprehension_score": 0.7, "quality_floor": 0.0, "max_clips": 12,
                "max_anchors": 12, "closure_max_span_s": 999.0}
    specs, notes, rejections = assemble_clips(st, "", sents, "u", "v", settings, AnchorAdapter())
    assert isinstance(rejections, list)
    assert all(hasattr(r, "stage") for r in rejections)
    assert isinstance(specs, list) and isinstance(notes, str)


def test_quality_floor_and_max_clips_ledgered(monkeypatch):
    from backend import config
    from backend.pipeline.assemble import assemble_clips
    monkeypatch.setattr(config, "JUDGE_MODEL", "")
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "same")
    monkeypatch.setattr(llm_mod, "llm_json",
                        lambda *a, **kw: JudgeVerdict(reasoning="ok", score_10=9, understandable=True))
    sents, units = _setup(8)
    st = _structure(sents, units)
    # floor 2.0 is unreachable → every kept candidate must land in quality_floor rejections
    settings = {"min_clip_duration_s": 1.0, "max_clip_duration_s": 500.0,
                "min_comprehension_score": 0.7, "quality_floor": 2.0, "max_clips": 12,
                "max_anchors": 12, "closure_max_span_s": 999.0}
    specs, _notes, rejections = assemble_clips(st, "", sents, "u", "v", settings, AnchorAdapter())
    assert specs == []
    assert rejections and all(r.stage in {"quality_floor", "dedupe", "snap", "repair"} for r in rejections)
    assert any(r.stage == "quality_floor" for r in rejections)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest backend/pipeline/assemble/tests/test_integrity.py -q`
Expected: FAIL — `assemble_clips` returns a 2-tuple; snap call inside it breaks on the new
snap signature only if Task 3 landed (it did) — the 2-tuple unpack error is the expected RED.

- [ ] **Step 3: Implement**

a) `assemble/__init__.py` — imports gain `from .integrity import Rejection` and
`from .validate import judge_clip`; step 4 onward becomes:

```python
    # 4. snap boundaries (metadata-aware dedupe; drops ledgered) -------------
    emit(0.80, "Refining boundaries…")
    specs, snap_rejections = snap_candidates(kept, sentences, settings, units)
    rejections.extend(snap_rejections)

    # 4b. hybrid re-judge: a merged span's text was never judged --------------
    surviving = []
    for s in specs:
        if not s.get("merged"):
            surviving.append(s)
            continue
        text = " ".join((sentences[i].text or "")
                        for i in range(s["sentence_start_idx"], s["sentence_end_idx"] + 1)).strip()
        key = frozenset(s.get("unit_ids", []))
        with cache_lock:
            verdict = cache.get(key)
        if verdict is None:
            verdict = judge_clip(text, s.get("role", ""), adapter,
                                 visual_summary=structure.visual_summary(s["start"], s["end"]),
                                 topic=topic, context_card="")
            if not verdict.error:
                with cache_lock:
                    cache[key] = verdict
        if verdict.error:                                 # outage: ship-but-flag (rubric policy)
            s["judge_error"] = True
            s["warnings"] = tuple(set(s.get("warnings") or ()) | {"unjudged"})
            surviving.append(s)
            continue
        if not (verdict.topic_identifiable and verdict.purpose_identifiable
                and verdict.source_grounded and verdict.all_references_resolved):
            rejections.append(Rejection(
                cand_id=s.get("cand_id", ""), title=s.get("title", ""), role=s.get("role", ""),
                stage="post_merge_judge", reason="merged span failed hard-core judge gate",
                score=float(verdict.score), failure_kinds=[f.kind for f in verdict.failure_reasons],
                final_quality=s.get("final_quality"), start=s["start"], end=s["end"]))
            continue
        s["completeness_score"] = scoring.completeness_score(verdict, s.get("role", ""), adapter)
        surviving.append(s)
    specs = surviving

    # 5. boundary score + quality filter (drops ledgered) --------------------
    for s in specs:
        s["boundary_score"] = scoring.boundary_score(s.get("warnings"))
        s["final_quality"] = scoring.quality(
            s.get("completeness_score", 0.0), s.get("grounding_score", 0.0),
            s["boundary_score"], s.get("priority", 0.0))
    floor = float(settings.get("quality_floor", 0.45))
    weak = [s for s in specs if s.get("final_quality", 0.0) < floor]
    specs = scoring.drop_weak(specs, floor)
    for s in weak:
        rejections.append(Rejection(cand_id=s.get("cand_id", ""), title=s.get("title", ""),
                                    role=s.get("role", ""), stage="quality_floor",
                                    reason=f"final_quality {s.get('final_quality', 0.0):.2f} < floor {floor}",
                                    final_quality=s.get("final_quality"), start=s["start"], end=s["end"]))
    specs.sort(key=lambda s: s["final_quality"], reverse=True)
    cap = int(settings.get("max_clips", config.DEFAULTS["max_clips"]))
    for s in specs[cap:]:
        rejections.append(Rejection(cand_id=s.get("cand_id", ""), title=s.get("title", ""),
                                    role=s.get("role", ""), stage="max_clips",
                                    reason=f"beyond max_clips={cap}",
                                    final_quality=s.get("final_quality"), start=s["start"], end=s["end"]))
    specs = specs[:cap]
```

All three `return [], "…"` early-exits become `return [], "…", rejections` (empty list at the
anchors exit — define `rejections` before step 3). The final return:
`return specs, notes, rejections`. Docstring updated: "Returns (clips_spec, notes, rejections)".

b) `cli.py:80` — `clips_spec, notes = …` → `clips_spec, notes, rejections = …`; after the
`print(f"  notes: …")` line add:

```python
    for r in (rejections or []):
        print(f"  [dropped/{r.stage}] {r.title[:60]} (score={r.score}, kinds={r.failure_kinds}, q={r.final_quality})")
```

(the fast-path branch sets `rejections = []` next to its existing assignments.)

c) `orchestrator.py:201-204` —

```python
    clips_spec, notes, rejections = await run(
        assemble_clips, structure, job.topic, sentences, job.url, video_id,
        settings, adapter, emit("assembling", 72, 90),
    )
    if rejections:
        registry.publish(job, ProgressEvent(
            "assembling", 90.0, f"{len(rejections)} candidate(s) dropped ({', '.join(sorted({r.stage for r in rejections}))})"))
```

d) `eval/run_eval.py` — the call site becomes
`specs, _notes, rejections = assemble_clips(st, topic, sents, url, video_id, settings, adapter)`
and immediately after the `_measure` call appends counts to the run dict:

```python
            m = _measure(st, specs, sents, adapter, det, topic, gold, settings,
                         verbose=verbose and r == 0)
            for stage in ("repair", "snap", "dedupe", "post_merge_judge", "quality_floor", "max_clips"):
                m[f"rejections_{stage}"] = sum(1 for rj in rejections if rj.stage == stage)
            run_dicts.append(m)
```

(the `rejections` variable is bound in the same guarded block where `assemble_clips` runs;
`--freeze-specs` reuses the same rejections for all runs — correct, since specs are frozen.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest backend/pipeline/assemble/tests/ backend/eval/tests/ -q`
Expected: 62 passed (34 assemble + 28 eval).

- [ ] **Step 5: Checkpoint**

Run: `.venv/bin/python -m pytest backend -q && .venv/bin/python -m compileall -q backend`
Expected: 126 passed, compile clean.

---

### Task 5: Card integrity — extractive fallback + missing-card warning

**Files:**
- Modify: `backend/pipeline/assemble/context_card.py::generate_context_card`
- Modify: `backend/pipeline/assemble/__init__.py` step 6 (warning append)
- Test: append to `backend/pipeline/assemble/tests/test_integrity.py`

**Interfaces:**
- Consumes: `spec["referential"]`, `spec["truncated"]`, `spec["unit_ids"]` (Tasks 2-4).
- Produces: `generate_context_card` never returns "" when referential context exists (extractive fallback); step 6 appends `"missing_context_card"` when a needed card is still empty.

- [ ] **Step 1: Write the failing tests** (append to `test_integrity.py`)

```python
# ── card integrity ────────────────────────────────────────────────────────────
from backend.pipeline.assemble.context_card import generate_context_card


def test_card_llm_failure_falls_back_to_referential_summary(monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("card llm down")
    monkeypatch.setattr(llm_mod, "llm_json", boom)
    sents, units = _setup()
    units_by_id = {u.unit_id: u for u in units}
    units_by_id["u0005"].summary = "Definition of the derivative from first principles."
    spec = {"referential": [("u0005", "prerequisite")], "anchor_id": "u0000",
            "unit_ids": ["u0000"], "truncated": True}
    card = generate_context_card(spec, units_by_id, FakeAdapter(), "derivatives")
    assert "derivative" in card.lower()                  # extractive, verbatim from summary


def test_card_skips_referential_already_inside_clip(monkeypatch):
    monkeypatch.setattr(llm_mod, "llm_json", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError()))
    sents, units = _setup()
    units_by_id = {u.unit_id: u for u in units}
    spec = {"referential": [("u0001", "prerequisite")], "anchor_id": "u0000",
            "unit_ids": ["u0000", "u0001"], "truncated": False}
    assert generate_context_card(spec, units_by_id, FakeAdapter(), "") == ""   # nothing outside clip


def test_missing_card_warning_appended():
    # pure: the step-6 helper logic — needed context, empty card → warning
    from backend.pipeline.assemble import _card_warning
    s = {"context_card": "", "referential": [("u0005", "prerequisite")], "truncated": False,
         "warnings": ("x",)}
    _card_warning(s)
    assert "missing_context_card" in s["warnings"] and "x" in s["warnings"]
    s2 = {"context_card": "has one", "referential": [("u0005", "p")], "warnings": ()}
    _card_warning(s2)
    assert "missing_context_card" not in s2["warnings"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest backend/pipeline/assemble/tests/test_integrity.py -q`
Expected: FAIL — card returns "" on LLM failure; no `_card_warning`.

- [ ] **Step 3: Implement**

a) `context_card.py::generate_context_card` — filter inlined referential up front, and add the
extractive fallback:

```python
def generate_context_card(spec: dict, units_by_id: dict[str, Unit], adapter, topic: str) -> str:
    in_clip = set(spec.get("unit_ids", []))
    ref_pairs = [(uid, rel) for uid, rel in spec.get("referential", [])
                 if uid in units_by_id and uid not in in_clip]
    ref_units = [units_by_id[uid] for uid, _rel in ref_pairs]
    if not ref_units:
        return ""
    anchor = units_by_id.get(spec.get("anchor_id", ""))
    allowed = {u.unit_id: u for u in (([anchor] if anchor else []) + ref_units)}
    rows = "\n".join(f"{u.unit_id}: {u.summary}" for u in allowed.values())
    from ...llm import llm_json
    system = _CARD_SYSTEM.format(max_words=config.CONTEXT_CARD_MAX_WORDS)
    user = f"TOPIC: {topic or '(general)'}\n\nUNITS (earlier context for the clip):\n{rows}\n\nWrite the preface."
    card = ""
    try:
        draft = llm_json(system, user, ContextCardDraft, temperature=0.2)
        kept = [cs.text.strip() for cs in draft.sentences
                if cs.source_unit_id in allowed and _grounded(cs.text, allowed[cs.source_unit_id])]
        card = " ".join(t for t in kept if t)
    except Exception:
        card = ""
    if not card:
        # extractive fallback: verbatim referential-unit summary — grounded by construction
        for u in ref_units:
            if (u.summary or "").strip():
                card = u.summary.strip()
                break
    words = card.split()
    if len(words) > config.CONTEXT_CARD_MAX_WORDS:     # hard word budget (prompt asks for it too)
        card = " ".join(words[:config.CONTEXT_CARD_MAX_WORDS])
    return card
```

b) `assemble/__init__.py` — module-level helper + step-6 use:

```python
def _card_warning(s: dict) -> None:
    """A clip that NEEDS earlier context (referential prereqs or truncated closure) but has no
    card ships with an explicit, penalized marker instead of failing silently."""
    if not s.get("context_card") and (s.get("referential") or s.get("truncated")):
        s["warnings"] = tuple(set(s.get("warnings") or ()) | {"missing_context_card"})
```

```python
    # 6. context cards -------------------------------------------------------
    emit(0.90, "Writing context cards…")
    for s in specs:
        s["context_card"] = generate_context_card(s, units_by_id, adapter, topic)
        _card_warning(s)
```

NOTE ordering: `_card_warning` runs AFTER step 5 computed `boundary_score`, so re-derive the
two scores for flagged specs right there:

```python
        if "missing_context_card" in (s.get("warnings") or ()):
            s["boundary_score"] = scoring.boundary_score(s.get("warnings"))
            s["final_quality"] = scoring.quality(
                s.get("completeness_score", 0.0), s.get("grounding_score", 0.0),
                s["boundary_score"], s.get("priority", 0.0))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest backend/pipeline/assemble/tests/ -q`
Expected: 37 passed.

- [ ] **Step 5: Checkpoint**

Run: `.venv/bin/python -m pytest backend -q && .venv/bin/python -m compileall -q backend`
Expected: 129 passed, compile clean.

---

### Task 6: Real-world verification (controller-run)

**Files:** none (measurement + docs).

- [ ] **Step 1: Frozen A/B** — `--freeze --runs 3` on 3 cached videos before/after was captured
implicitly (pre-change numbers exist from the judge-rubric A/B). Run after:

```bash
cd /Users/vincentfeng/Documents/practice/clips
.venv/bin/python -m backend.eval.run_eval uqwC41RDPyg NjvwWiCYLl4 yfajBIaDf1Q --freeze --runs 3
```

Expected: `rejections_*` columns appear and are numeric; for every video,
kept(n_clips) + Σ rejections is constant across runs OR the varying stage is identifiable;
comprehension within noise of the judge-rubric run; no `judge_error_rate` > 0.

- [ ] **Step 2: CLI smoke** on a cached video:

```bash
PRECISE_BOUNDARIES=0 .venv/bin/python -m backend.cli "https://youtu.be/NjvwWiCYLl4" "" full
```

Expected: any dropped candidate prints a `[dropped/<stage>]` line; merged clips (if any) carry
`merged_overlap`; cards non-empty or the clip shows the `missing_context_card` penalty in q.

- [ ] **Step 3: Docs** — audit doc: mark the pkg-1 findings addressed (one line each);
clipper memory: integrity-chain entry (mutation warnings, re-judge policy, ledger, card
fallback, measured numbers).

---

## Self-Review (done)

- **Spec coverage:** true_contents at build/expand/merge → Tasks 1-3; validate_and_repair tuple +
  repair Rejection → Task 2; Part-B dedupe + snap/dedupe rejections + truncated carry → Task 3;
  additive refine/boundary warnings + 3 penalties → Task 3; re-judge hook + quality/max_clips
  ledger + 3-tuple + 3 call sites + eval counts → Task 4; card fallback + inlined-referential
  skip + missing_context_card + score re-derive → Task 5; measurement → Task 6. No gaps.
- **Placeholders:** none — full code in every step.
- **Type consistency:** `Rejection` fields used identically across Tasks 1-4; `snap_candidates`
  4-arg/2-tuple shape consistent between Task 3 impl, Task 3/4 tests, and Task 4's call;
  `merge_partb(a, b, units, sentences)` matches its Task 3 call; `validate_and_repair` 2-tuple
  consistent with Task 2's caller update and Task 2 test unpacks; stage strings match the
  Global Constraints set everywhere.
