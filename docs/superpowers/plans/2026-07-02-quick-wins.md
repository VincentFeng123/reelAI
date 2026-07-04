# Quick Wins (Robustness Pkg 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close seven silent-failure holes: precise-boundary crashes, topic-blind selection, poisoned mp4 caches, wrong-range cache hits, self-defined prerequisite hints, embed-mode zip crashes, and wrong-direction Whisper snaps.

**Architecture:** Seven surgical fixes in three file clusters — boundary.py (resilience + direction-safe picks), output/serving (atomic tmp-rename writes, range-keyed export names, guarded zip), assemble-side (relevance degraded flag surfaced in notes, unmet-concept prerequisite hints). Each fix has offline tests; no structural refactors.

**Tech Stack:** Python 3.12, pytest 9.1.1, mocked `llm_json`/`_ensure_audio`, real temp files for the atomic-write helper.

**Spec:** `docs/superpowers/specs/2026-07-02-quick-wins-design.md`

## Global Constraints

- Run from `/Users/vincentfeng/Documents/practice/clips` with `.venv/bin/python`. **No git repo** — Checkpoint steps (full pytest + compileall) replace commits.
- Tests offline: no network, no LLM, no ffmpeg (the finalize helper is tested with plain files).
- `refine.py` stays untouched (frozen). `boundary.py` changes are exactly fixes 1 and 7 — `_resolve_overlaps` and `_whisper_window` unchanged.
- Notes suffix, exact: `" (topic filtering degraded — clips selected by role priority)"`.
- Zip 409 detail, exact: `"No rendered clips to zip — use output_mode=cut or export clips first"`.
- Suite baseline before this plan: 133 passed.

---

### Task 1: boundary.py — resilience + direction-safe picks

**Files:**
- Modify: `backend/pipeline/boundary.py` (`refine_clip_boundaries` audio guard; `_pick_start`, `_pick_end`)
- Create: `backend/pipeline/tests/__init__.py` (empty; first tests for this package level)
- Test: `backend/pipeline/tests/test_boundary_safety.py`

**Interfaces:**
- Produces: `_pick_start(sents, rough, pad, keep_first: bool = False) -> float`;
  `_pick_end` unchanged signature, direction-safe fallback; `refine_clip_boundaries` returns the
  input clips unchanged when `_ensure_audio` raises.

- [ ] **Step 1: Write the failing tests**

```python
# backend/pipeline/tests/test_boundary_safety.py
"""Precise-boundary resilience + direction-safe Whisper picks. Offline (no audio, no whisper)."""
from __future__ import annotations

import pytest

import backend.pipeline.boundary as boundary_mod
from backend.pipeline.boundary import _pick_end, _pick_start, refine_clip_boundaries
from backend.pipeline.sentences import Sentence


def _sent(i, start, end, terminator="."):
    return Sentence(idx=i, text=f"s{i}.", start=start, end=end, terminator=terminator,
                    ends_with_period=(terminator in ".?!"), word_start_idx=i, word_end_idx=i,
                    align_confidence=1.0)


# ── fix 1: audio failure returns input unchanged ─────────────────────────────
def test_refine_returns_input_unchanged_on_audio_failure(monkeypatch):
    def boom(url, video_id):
        raise RuntimeError("yt-dlp throttled")
    monkeypatch.setattr(boundary_mod, "_ensure_audio", boom)
    clips = [{"start": 10.0, "end": 40.0, "cut_end": 40.05, "facet": "other"}]
    out = refine_clip_boundaries(clips, "https://youtu.be/x", "vid", {})
    assert out is clips                                   # the very same list, untouched


# ── fix 7: direction-safe fallbacks ──────────────────────────────────────────
def test_pick_end_never_moves_earlier_than_window_floor():
    # only valid end is at 31.0, far BEFORE rough-1.0 (rough=45): old code returned 31.0
    sents = [_sent(0, 28.0, 31.0)]
    assert _pick_end(sents, rough=45.0, pad=10.0, allow_qe=False) == 45.0


def test_pick_end_normal_path_unchanged():
    sents = [_sent(0, 40.0, 44.5), _sent(1, 44.6, 46.2)]
    assert _pick_end(sents, rough=45.0, pad=10.0, allow_qe=False) == 46.2


def test_pick_start_never_moves_later_than_window_ceiling():
    # only candidate start is at 58.0, far AFTER rough+1.0 (rough=45): old code returned 58.0
    sents = [_sent(0, 43.0, 44.0), _sent(1, 58.0, 60.0)]  # sents[0] dropped as fragment
    assert _pick_start(sents, rough=45.0, pad=10.0) == 45.0


def test_pick_start_normal_path_unchanged():
    sents = [_sent(0, 35.0, 36.0), _sent(1, 44.2, 47.0), _sent(2, 47.1, 49.0)]
    assert _pick_start(sents, rough=45.0, pad=10.0) == 44.2


def test_pick_start_keep_first_at_video_start():
    # window clamped at t=0 → sents[0] is a REAL sentence start, not a fragment
    sents = [_sent(0, 0.0, 3.0), _sent(1, 3.1, 6.0)]
    assert _pick_start(sents, rough=0.5, pad=10.0, keep_first=True) == 0.0
    assert _pick_start(sents, rough=0.5, pad=10.0) == 0.5   # fragment dropped → direction-safe rough
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest backend/pipeline/tests/test_boundary_safety.py -q`
Expected: FAIL — refine raises RuntimeError; `_pick_end` returns 31.0; `_pick_start` returns
58.0; `keep_first` is an unexpected keyword.

- [ ] **Step 3: Implement in `backend/pipeline/boundary.py`**

a) `_pick_start` becomes:

```python
def _pick_start(sents: list[Sentence], rough: float, pad: float,
                keep_first: bool = False) -> float:
    """Latest real sentence-start at/just before the rough start (begin at a thought).
    keep_first: the window began at t<=0, so sents[0] is a real start, not a cut fragment."""
    if not sents:
        return rough
    pool = sents if keep_first else sents[1:]
    starts = [s.start for s in pool] or [s.start for s in sents]
    before = [x for x in starts if x <= rough + 1.0]
    if not before:
        return rough              # direction-safe: never move the start LATER than rough+1s
    cand = max(before)
    return cand if abs(cand - rough) <= pad else rough
```

b) `_pick_end`'s fallback becomes direction-safe (two-tier: prefer no truncation, tolerate the
1s epsilon, else keep rough — adjudicated during execution; the single-tier snippet originally
here contradicted the normal-path test):

```python
    at_or_after = [x for x in ends if x >= rough]
    near = [x for x in ends if rough - 1.0 <= x < rough]
    if at_or_after:
        cand = min(at_or_after)
    elif near:
        cand = max(near)
    else:
        return rough              # direction-safe: never truncate EARLIER than rough-1s
    return cand if abs(cand - rough) <= pad else rough
```

c) In `refine_clip_boundaries`, guard the audio step (replacing the bare `audio = _ensure_audio(url, video_id)`):

```python
    try:
        audio = _ensure_audio(url, video_id)
    except Exception:
        return clips              # precise pass unavailable → coarse (judged) boundaries stand
```

and pass `keep_first` at both `_pick_start` call sites:

```python
                new_start = _pick_start(w, s0, pad, keep_first=(s0 - pad <= 0.0))
```
```python
                new_start = _pick_start(_whisper_window(audio, s0 - pad, s0 + pad), s0, pad,
                                        keep_first=(s0 - pad <= 0.0))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest backend/pipeline/tests/test_boundary_safety.py -q`
Expected: 6 passed.

- [ ] **Step 5: Checkpoint**

Run: `.venv/bin/python -m pytest backend -q && .venv/bin/python -m compileall -q backend`
Expected: 139 passed, compile clean.

---

### Task 2: Atomic writes + range-keyed export + zip guard

**Files:**
- Modify: `backend/pipeline/export.py` (`finalize_output` new; `export_clip` tmp/rename, fname, cache check)
- Modify: `backend/pipeline/cut.py` (`one()` tmp/rename via `finalize_output`)
- Modify: `backend/main.py` (`_zipable_files` helper; `download_zip` guard)
- Test: `backend/pipeline/tests/test_output_safety.py`

**Interfaces:**
- Consumes: existing `build_cmd`, `PipelineError`.
- Produces: `export.finalize_output(tmp: Path, out: Path, rc: int, err: str) -> None` (raises
  `PipelineError` on failure, atomically renames on success — cut.py imports it);
  export filenames `clip_{n}_{slug}_{start_ms}_{end_ms}.mp4`;
  `main._zipable_files(clips: list[dict], folder: Path) -> list[Path]`.

- [ ] **Step 1: Write the failing tests**

```python
# backend/pipeline/tests/test_output_safety.py
"""Atomic ffmpeg output finalization, range-keyed export names, zip guard. Offline."""
from __future__ import annotations

from pathlib import Path

import pytest

from backend.errors import PipelineError
from backend.main import _zipable_files
from backend.pipeline.export import _export_fname, finalize_output


# ── fix 3: atomic finalize ────────────────────────────────────────────────────
def test_finalize_success_renames(tmp_path):
    tmp, out = tmp_path / "c.tmp.mp4", tmp_path / "c.mp4"
    tmp.write_bytes(b"fake-mp4-bytes")
    finalize_output(tmp, out, rc=0, err="")
    assert out.exists() and not tmp.exists()
    assert out.read_bytes() == b"fake-mp4-bytes"


def test_finalize_failure_cleans_tmp_and_raises(tmp_path):
    tmp, out = tmp_path / "c.tmp.mp4", tmp_path / "c.mp4"
    tmp.write_bytes(b"partial")
    with pytest.raises(PipelineError):
        finalize_output(tmp, out, rc=1, err="boom\nlast line")
    assert not tmp.exists() and not out.exists()


def test_finalize_zero_byte_output_is_failure(tmp_path):
    tmp, out = tmp_path / "c.tmp.mp4", tmp_path / "c.mp4"
    tmp.write_bytes(b"")
    with pytest.raises(PipelineError):
        finalize_output(tmp, out, rc=0, err="")
    assert not tmp.exists() and not out.exists()


# ── fix 4: range-keyed export name ───────────────────────────────────────────
def test_export_fname_embeds_range():
    assert _export_fname(2, "definition", 12.345, 67.89) == "clip_2_definition_12345_67890.mp4"
    assert _export_fname(2, "definition", 12.345, 68.0) != _export_fname(2, "definition", 12.345, 67.89)


# ── fix 6: zip guard ─────────────────────────────────────────────────────────
def test_zipable_files_skips_none_and_missing(tmp_path):
    real = tmp_path / "clip_1_other.mp4"
    real.write_bytes(b"x")
    clips = [{"path": None},                                # embed mode
             {"path": "/clips/v/clip_1_other.mp4"},         # exists
             {"path": "/clips/v/clip_9_gone.mp4"}]          # missing on disk
    files = _zipable_files(clips, tmp_path)
    assert files == [real]


def test_zipable_files_empty_for_embed_job(tmp_path):
    assert _zipable_files([{"path": None}, {"path": None}], tmp_path) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest backend/pipeline/tests/test_output_safety.py -q`
Expected: FAIL — ImportError on `finalize_output`, `_export_fname`, `_zipable_files`.

- [ ] **Step 3: Implement**

a) `backend/pipeline/export.py` — add (module level, near `_slug`):

```python
def _export_fname(n: int, facet: str, start: float, end: float) -> str:
    """Range-keyed name: different boundaries → different file (stale cache can't serve)."""
    return f"clip_{n}_{_slug(facet)}_{int(round(start * 1000))}_{int(round(end * 1000))}.mp4"


def finalize_output(tmp: Path, out: Path, rc: int, err: str) -> None:
    """Atomically promote an ffmpeg output: rename tmp→out only on success with real bytes;
    otherwise remove the partial tmp and raise. A crash/failure can never poison the cache."""
    if rc == 0 and tmp.exists() and tmp.stat().st_size > 0:
        tmp.replace(out)
        return
    tmp.unlink(missing_ok=True)
    tail = err.strip().splitlines()
    raise PipelineError(f"Clip cut failed: {tail[-1] if tail else rc}")
```

(ensure `from pathlib import Path` is imported.)

b) `export_clip` — replace the fname/cache/cut block:

```python
    fname = _export_fname(n, facet, float(start), float(end))
    out = out_dir / fname
    served = f"/clips/{video_id}/{fname}"
    if out.exists() and out.stat().st_size > 0:
        return {"path": served}

    src = await asyncio.to_thread(_ensure_source, url, video_id, int(res))

    dur = max(0.1, round(float(end) - float(start), 3))
    tmp = out.with_name(out.stem + ".tmp.mp4")
    proc = await asyncio.create_subprocess_exec(
        *build_cmd(str(src), float(start), dur, tmp),
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
    )
    err = (await proc.stderr.read()).decode(errors="ignore") if proc.stderr else ""
    rc = await proc.wait()
    finalize_output(tmp, out, rc, err)
    return {"path": served}
```

c) `backend/pipeline/cut.py::one()` — route through the same helper: cut to a tmp name and
finalize (import `from .export import finalize_output` — if that would create a circular
import because export.py imports from cut.py, put `finalize_output`+`_export_fname` in cut.py
instead and have export.py import from cut.py; keep ONE definition, wherever the import
direction already flows):

```python
        out = out_dir / fname
        tmp = out.with_name(out.stem + ".tmp.mp4")
        ...
            proc = await asyncio.create_subprocess_exec(
                *build_cmd(src, start, dur, tmp),
                ...
            )
            ...
            rc = await proc.wait()

        if rc != 0 or not tmp.exists() or tmp.stat().st_size == 0:
            tmp.unlink(missing_ok=True)
            tail = err.strip().splitlines()
            raise PipelineError(f"ffmpeg failed on clip {i + 1}: {tail[-1] if tail else rc}")
        tmp.replace(out)
```

(cut.py's failure message format is preserved; using inline finalize logic there is acceptable
if the import direction makes the shared helper awkward — but prefer the shared helper.)

d) `backend/main.py` — add near the top (after imports):

```python
def _zipable_files(clips: list[dict], folder: Path) -> list[Path]:
    """Existing rendered files for a job — embed-mode (path=None) and missing files skipped."""
    out: list[Path] = []
    for c in clips or []:
        p = c.get("path")
        if not p:
            continue
        fp = folder / Path(p).name
        if fp.exists():
            out.append(fp)
    return out
```

and `download_zip` becomes:

```python
@app.get("/jobs/{job_id}/zip")
def download_zip(job_id: str):
    job = registry.get(job_id)
    if not job or job.status != Status.DONE or not job.video_id:
        raise HTTPException(404, "No finished job to zip")
    folder = config.OUTPUT_DIR / job.video_id
    files = _zipable_files(job.clips, folder)
    has_manifest = (folder / "clips.json").exists()
    if not files and not has_manifest:
        raise HTTPException(409, "No rendered clips to zip — use output_mode=cut or export clips first")

    def gen():
        from zipstream import ZipStream
        zs = ZipStream(sized=False)
        for fp in files:
            zs.add_path(fp, fp.name)
        if has_manifest:
            zs.add_path(folder / "clips.json", "clips.json")
        yield from zs
```

(the existing `StreamingResponse(...)` return stays as-is.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest backend/pipeline/tests/ -q`
Expected: 12 passed.

- [ ] **Step 5: Checkpoint**

Run: `.venv/bin/python -m pytest backend -q && .venv/bin/python -m compileall -q backend`
Expected: 145 passed, compile clean.

---

### Task 3: Relevance-degraded surfacing + unmet-concept prerequisite hints

**Files:**
- Modify: `backend/pipeline/assemble/candidates.py` (`score_topic_relevance` retry + tuple return)
- Modify: `backend/pipeline/assemble/sequence.py` (`attach_prerequisites`, `sequence_clips`)
- Modify: `backend/pipeline/assemble/__init__.py` (unpack relevance tuple; notes suffix; pass `units_by_id` to `sequence_clips`)
- Test: append to `backend/pipeline/assemble/tests/test_integrity.py`

**Interfaces:**
- Consumes: existing assemble flow (`relevance` used by `select_anchors`/`build_candidate` —
  the dict itself is unchanged in shape).
- Produces: `score_topic_relevance(units, topic, settings, progress) -> tuple[dict[str, float], bool]`;
  `sequence_clips(specs, graph, units_by_id) -> list[dict]`;
  notes suffix exactly `" (topic filtering degraded — clips selected by role priority)"`.

- [ ] **Step 1: Write the failing tests** (append to `test_integrity.py`)

```python
# ── pkg-3: relevance degraded flag + unmet-concept prereq hints ───────────────
from backend.pipeline.assemble.candidates import score_topic_relevance
from backend.pipeline.assemble.sequence import sequence_clips


def test_relevance_retry_then_success_not_degraded(monkeypatch):
    from backend.pipeline.assemble.candidates import RelevanceLLM, RelItem
    calls = {"n": 0}

    def flaky(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return RelevanceLLM(items=[RelItem(unit_id="u0000", score=0.9)])
    monkeypatch.setattr(llm_mod, "llm_json", flaky)
    sents, units = _setup(1)
    rel, degraded = score_topic_relevance(units, "calculus", {})
    assert degraded is False
    assert rel["u0000"] == pytest.approx(0.9)
    assert calls["n"] == 2                                  # one retry happened


def test_relevance_double_failure_flags_degraded(monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("down")
    monkeypatch.setattr(llm_mod, "llm_json", boom)
    sents, units = _setup(2)
    rel, degraded = score_topic_relevance(units, "calculus", {})
    assert degraded is True
    assert all(v == 0.5 for v in rel.values())              # neutral defaults, honestly flagged


def test_relevance_empty_topic_no_llm(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("must not be called")
    monkeypatch.setattr(llm_mod, "llm_json", boom)
    sents, units = _setup(2)
    rel, degraded = score_topic_relevance(units, "", {})
    assert degraded is False and all(v == 1.0 for v in rel.values())


def test_prereq_hint_skipped_when_self_defined():
    sents, units = _setup(6)
    units[0].concepts_introduced = ["derivative"]
    units[3].concepts_required = ["derivative"]
    units[3].concepts_introduced = ["derivative"]           # clip defines it itself
    units[5].concepts_required = ["derivative"]             # clip does NOT define it
    units_by_id = {u.unit_id: u for u in units}
    specs = [
        {"start": 0.0, "unit_ids": ["u0000"]},
        {"start": 30.0, "unit_ids": ["u0003"]},
        {"start": 50.0, "unit_ids": ["u0005"]},
    ]
    seq = sequence_clips(specs, Graph([], units), units_by_id)
    by_start = {s["start"]: s for s in seq}
    assert by_start[30.0]["prerequisite_clips"] == []       # self-defined → no hint
    assert by_start[50.0]["prerequisite_clips"] == [1]      # needs clip 1's definition
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest backend/pipeline/assemble/tests/test_integrity.py -q`
Expected: FAIL — relevance returns a plain dict (unpack error); `sequence_clips` takes 2 args;
self-defined clip still gets a hint.

- [ ] **Step 3: Implement**

a) `candidates.py::score_topic_relevance`:

```python
def score_topic_relevance(units: list[Unit], topic: str, settings: dict,
                          progress: ProgressCb = None) -> tuple[dict[str, float], bool]:
    """(unit_id → 0..1 relevance, degraded). degraded=True means the relevance LLM failed
    (after one retry per batch) and scores are neutral defaults — the topic filter is
    effectively OFF and callers must surface that instead of pretending scores exist."""
    if not topic or not topic.strip():
        return {u.unit_id: 1.0 for u in units}, False
    from ...llm import llm_json
    rel: dict[str, float] = {}
    degraded = False
    B = 120
    batches = [units[i:i + B] for i in range(0, len(units), B)] or [[]]
    for bi, batch in enumerate(batches):
        rows = "\n".join(
            f"{u.unit_id}: {u.summary[:120]}"
            + (f" | concepts: {', '.join((u.concepts_introduced + u.concepts_required)[:6])}"
               if (u.concepts_introduced or u.concepts_required) else "")
            for u in batch
        )
        user = f"TOPIC: {topic}\n\nUNITS:\n{rows}\n\nScore each unit id 0.0–1.0 for relevance to the topic."
        got: dict[str, float] = {}
        for attempt in range(2):                            # one retry on transient failure
            try:
                res = llm_json(_REL_SYSTEM, user, RelevanceLLM, temperature=0.0)
                got = {it.unit_id: max(0.0, min(1.0, float(it.score))) for it in res.items}
                break
            except Exception:
                if attempt == 1:
                    degraded = True                          # batch stays neutral — flag it
        for u in batch:
            rel[u.unit_id] = got.get(u.unit_id, 0.5)   # neutral if the model omitted it
        if progress:
            progress((bi + 1) / len(batches), "Scoring topic relevance")
    return rel, degraded
```

b) `assemble/__init__.py` step 1 + notes:

```python
    relevance, relevance_degraded = score_topic_relevance(
        units, topic, settings, lambda f, m="": emit(0.05 + 0.10 * f, m))
```

```python
    notes = f"{len(specs)} clip(s) about “{topic}”." if topic else f"{len(specs)} clip(s)."
    if relevance_degraded and topic:
        notes += " (topic filtering degraded — clips selected by role priority)"
    return specs, notes, rejections
```

c) `sequence.py`:

```python
def attach_prerequisites(specs: list[dict], graph, units_by_id: dict) -> None:
    """Hint an earlier clip only for concepts this clip REQUIRES but does not itself
    introduce — a clip that contains its own definition needs no 'watch first' pointer."""
    def _units(s):
        return [units_by_id[u] for u in s.get("unit_ids", []) if u in units_by_id]

    for s in specs:
        mine = _units(s)
        intro = set().union(*[set(u.concepts_introduced) for u in mine]) if mine else set()
        req = set().union(*[set(u.concepts_required) for u in mine]) if mine else set()
        unmet = {c for c in (req - intro) if c}
        prereqs: list[int] = []
        # Earliest-per-concept semantics (adjudicated during execution: the all-earlier-clips
        # reference originally here contradicted the test below — and produces redundant
        # "watch first" pointers; the earliest introducer of each unmet concept suffices).
        for concept in unmet:
            for other in specs:                               # specs are in temporal order
                if other is s or other["start"] >= s["start"]:
                    continue
                ou = [units_by_id[u] for u in other.get("unit_ids", []) if u in units_by_id]
                ou_intro = set().union(*[set(u.concepts_introduced) for u in ou]) if ou else set()
                if concept in ou_intro:
                    prereqs.append(other["sequence_index"])
                    break                                     # only the earliest for this concept
        s["prerequisite_clips"] = sorted(set(prereqs))


def sequence_clips(specs: list[dict], graph, units_by_id: dict) -> list[dict]:
    specs.sort(key=lambda s: s["start"])
    for i, s in enumerate(specs):
        s["sequence_index"] = i + 1
    attach_prerequisites(specs, graph, units_by_id)
    return specs
```

d) `assemble/__init__.py` step 7: `specs = sequence_clips(specs, graph, units_by_id)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest backend/pipeline/assemble/tests/ -q`
Expected: all pass (4 new).

- [ ] **Step 5: Checkpoint**

Run: `.venv/bin/python -m pytest backend -q && .venv/bin/python -m compileall -q backend`
Expected: 149 passed, compile clean.

---

### Task 4: Verification (controller-run)

**Files:** none.

- [ ] **Step 1:** Full suite + compile (`149 passed` expected — report actuals).
- [ ] **Step 2:** CLI smoke on a cached video (`PRECISE_BOUNDARIES=0 .venv/bin/python -m backend.cli "https://youtu.be/NjvwWiCYLl4" "" full`) — no behavior change expected on the happy path (clips + drop lines as before).
- [ ] **Step 3:** Docs — audit doc gains a "PKG 3 SHIPPED" header line; clipper memory entry.

---

## Self-Review (done)

- **Spec coverage:** fix 1 → T1; fix 7 → T1; fix 3 → T2 (helper + both call sites); fix 4 → T2
  (`_export_fname`); fix 6 → T2 (`_zipable_files` + 409); fix 2 → T3 (retry/tuple/notes);
  fix 5 → T3 (unmet-concept hints + signature). No gaps.
- **Placeholders:** none — full code every step. The cut.py circular-import contingency names
  both acceptable resolutions explicitly (shared helper preferred, inline fallback allowed with
  preserved message format).
- **Type consistency:** `finalize_output(tmp, out, rc, err)` matches tests and both call sites;
  `_export_fname(n, facet, start, end)` matches test expectations (12345 = round(12.345*1000));
  `score_topic_relevance` 2-tuple matches T3's unpack; `sequence_clips(specs, graph, units_by_id)`
  matches caller update and test.
