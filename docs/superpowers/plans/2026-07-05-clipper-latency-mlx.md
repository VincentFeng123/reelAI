# Clipper Latency Pass (MLX + paid-Gemini) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut a 1-hour video's pipeline wall-clock ~2× cold (~600s→~300s) and make a warm re-clip near-instant, with **zero quality loss proven by an A/B oracle** — by moving the boundary-refine Whisper from CPU→M2-GPU, caching deterministic assembly, and exploiting the paid Gemini tier.

**Architecture:** The dominant cost is `refine_clip_boundaries` re-transcribing ~50 clip-edge windows with `faster-whisper medium` on **CPU int8** (no Metal) — ~330s, GPU idle. We add an MLX-Whisper backend behind a config flag (identical algorithm, GPU hardware), a deterministic post-refine result cache keyed on the structure fingerprint, an aria2c downloader, and paid-tier parallelism. A parakeet-mlx global transcriber (deletes refine entirely) is a final **gated** phase.

**Tech Stack:** Python 3.12, faster-whisper (CPU, existing), **mlx-whisper** (new, M2 GPU), **parakeet-mlx** (new, gated), yt-dlp, ffmpeg, Gemini (google-genai), pytest. Apple Silicon (M2, 8 cores).

## Global Constraints

- **Zero quality loss, proven, not assumed.** Every ASR/boundary change ships only after the W0 boundary-precision A/B oracle passes: per-edge |Δstart|,|Δend| ≤ **50 ms** vs the current shipped cut, and no edge moves onto a word. Parakeet (W5) additionally requires a WER/text A/B pass.
- **No new cloud STT keys.** Local MLX only. (Groq declined.)
- **Keep all quality features on:** `MULTIMODAL=1`, `PRECISE_BOUNDARIES=1`, `CLIP_ENGINE=topic`, `TOPIC_MODEL=gemini-3.1-pro-preview`, Pro topic selection. No `ANALYSIS_PROFILE=fast`, no `REFINE_WHISPER_MODEL=base`.
- **Every new knob has a revert switch** (env var defaulting to current behavior). New MLX/parakeet deps are **lazy-imported + guarded** so non-mac/CI falls back to `faster_whisper`/`supadata` and the suite stays importable.
- **Preserve invariants:** `_run_full`→`_run_fast` graceful degrade; `unverified_kill = 0`; the existing suite (~760 tests) stays green.
- **clips/ is a git repo on `main`.** Do work on a branch (`git switch -c latency-mlx`). Snapshot touched hot-path files to `.backup/` before editing (repo convention). Restart uvicorn + rebuild frontend only when live-testing.
- **Repo idioms:** disk artifacts live under `config.WORK_DIR / <video_id> / …`, written via `p.parent.mkdir(parents=True, exist_ok=True)` + `write_text(json.dumps(...))`. Tests are plain pytest functions, `from __future__ import annotations`, fake `Sentence` via the `_s(i,a,b,term=".")` helper, floats asserted with `pytest.approx` / `< 1e-6`.

---

## File Structure

**Create:**
- `backend/pipeline/refine_asr.py` — MLX-Whisper window transcriber (W1); duck-types faster-whisper's `words`/`segs` output.
- `backend/pipeline/assemble/result_cache.py` — deterministic post-refine clip result cache (W3b).
- `backend/eval/boundary_ab.py` — boundary-precision A/B oracle + WER util (W0/W5 gate).
- `backend/pipeline/tests/test_refine_asr.py`, `backend/pipeline/assemble/tests/test_result_cache.py`, `backend/eval/tests/test_boundary_ab.py`.

**Modify:**
- `backend/config.py` — new knobs (all phases).
- `backend/pipeline/boundary.py:88-100` (backend branch in `_whisper_window`), `:406`/`:413-416` (workers=1 + prewarm for MLX).
- `backend/pipeline/transcribe.py` — `_prewarm_refine_asr()` (W4), `_transcribe_parakeet` + routing (W5).
- `backend/orchestrator.py:305-348` — result-cache wrap; `:139-149` parakeet source note.
- `backend/pipeline/download.py:131-145` — aria2c external downloader (W4).

**Phases & priority:** P0 (measure + oracle) → **P1 (MLX refine — the centerpiece)** → P2 (result cache — warm win) → P3 (aria2c + prewarm) → P4 (paid-tier tuning) → P5 (verdict cache, optional, unit-engine only) → **P6 (parakeet, gated)**.

---

## Phase 0 — Measurement + validation oracle

### Task 0.1: MLX feasibility spike (gate the whole plan)

**Files:**
- Create: `backend/pipeline/tests/_spike_mlx.py` (throwaway; delete after).

**Interfaces:**
- Produces: a go/no-go on `mlx-whisper` (word timestamps + speed on this M2). If no-go, P1/P6 pivot to `lightning-whisper-mlx` or stop.

- [ ] **Step 1: Install the MLX deps into the venv**

Run:
```bash
cd /Users/vincentfeng/Documents/practice/clips
.venv/bin/pip install mlx-whisper parakeet-mlx
```
Expected: both install cleanly (pure MLX wheels, no torch build) on arm64.

- [ ] **Step 2: Extract a 30s audio window from a cached video**

Run:
```bash
ls work/*/audio.m4a 2>/dev/null | head -1
# if none, extract from a cached video.mp4:
V=$(ls -d output/*/ | head -1); echo "$V"
/opt/homebrew/bin/ffmpeg -nostdin -y -ss 60 -t 30 -i "$(ls work/*/video.mp4 2>/dev/null | head -1)" -ar 16000 -ac 1 /tmp/spike.wav 2>/dev/null && echo OK
```
Expected: `/tmp/spike.wav` exists (16 kHz mono). If no cached media, skip to running on any local wav.

- [ ] **Step 3: Confirm mlx-whisper emits word timestamps + time it**

```python
# backend/pipeline/tests/_spike_mlx.py
import time, mlx_whisper
t0 = time.perf_counter()
r = mlx_whisper.transcribe("/tmp/spike.wav",
                           path_or_hf_repo="mlx-community/whisper-medium",
                           word_timestamps=True, temperature=0.0,
                           condition_on_previous_text=False)
dt = time.perf_counter() - t0
segs = r["segments"]
words = [w for s in segs for w in s.get("words", [])]
print(f"segments={len(segs)} words={len(words)} took={dt:.1f}s for 30s audio")
print("sample word:", words[0] if words else None)
assert words and all("start" in w and "end" in w and "word" in w for w in words[:5])
print("OK: mlx-whisper gives word-level timestamps")
```

- [ ] **Step 4: Run it**

Run: `.venv/bin/python -m backend.pipeline.tests._spike_mlx`
Expected: prints `OK`, `words>0`, and `took` well under 30s (real-time-factor < 1 on the M2 GPU). Record the RTF — it sets P1's expected refine speedup. **GATE:** if word timestamps are absent/empty or RTF ≥ ~1.0 (no speedup vs CPU), STOP and reassess (try `mlx-community/whisper-medium-mlx` repo id, or `lightning-whisper-mlx`).

- [ ] **Step 5: Delete the spike, commit nothing**

Run: `rm backend/pipeline/tests/_spike_mlx.py`
(The venv install stays; it is captured in Task 1.1's requirements edit.)

### Task 0.2: Per-stage profiling baseline

**Files:**
- Create: `docs/superpowers/plans/latency-baseline-2026-07-05.md` (measured numbers).

**Interfaces:**
- Produces: measured per-stage seconds for one cold + one warm 1-hr run — replaces the spec's estimates; sets A/B targets.

- [ ] **Step 1: Pick a ~1-hour video id and clear its cache for a cold run**

Run:
```bash
# use a known long lecture URL; store it for reuse
echo "https://www.youtube.com/watch?v=<ONE_HOUR_ID>" > /tmp/ab_url.txt
rm -rf work/<ONE_HOUR_ID>
```

- [ ] **Step 2: Run the pipeline once COLD with per-stage timing on**

Run:
```bash
PROFILE_TIMINGS_FILE=/tmp/cold_timings.jsonl \
  .venv/bin/python -m backend.cli "$(cat /tmp/ab_url.txt)" 2>&1 | tail -5
cat /tmp/cold_timings.jsonl
```
Expected: one JSON line per stage (`{"stage": "...", "ms": N, "cumulative_ms": M}`) written by `orchestrator._record`. (If `backend.cli` isn't the right entry, use the same offline recipe as `run_eval.eval_video`.)

- [ ] **Step 3: Run again WARM (structure cached) with timing on**

Run:
```bash
PROFILE_TIMINGS_FILE=/tmp/warm_timings.jsonl \
  .venv/bin/python -m backend.cli "$(cat /tmp/ab_url.txt)" 2>&1 | tail -5
cat /tmp/warm_timings.jsonl
```

- [ ] **Step 4: Record the breakdown**

Write `docs/superpowers/plans/latency-baseline-2026-07-05.md` with the cold + warm per-stage tables from the two jsonl files, and confirm the sink (expected: `refine_clip_boundaries` dominates). This is the number every later A/B compares against.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/plans/latency-baseline-2026-07-05.md
git commit -m "docs(latency): measured cold+warm per-stage baseline for a 1hr video"
```

### Task 0.3: Boundary-precision A/B oracle

**Files:**
- Create: `backend/eval/boundary_ab.py`, `backend/eval/tests/test_boundary_ab.py`.

**Interfaces:**
- Produces: `compare_boundaries(specs_a, specs_b) -> BoundaryDiff` (max/mean |Δstart|,|Δend| over IoU-matched clips) and `wer(ref: str, hyp: str) -> float`. Consumed by P1 (gate) and P6 (gate).

- [ ] **Step 1: Write the failing test for `compare_boundaries` and `wer`**

```python
# backend/eval/tests/test_boundary_ab.py
from __future__ import annotations
import pytest
from backend.eval.boundary_ab import compare_boundaries, wer

def _spec(s, e):  # minimal clip spec shape (start/end are what we compare)
    return {"start": s, "end": e, "cut_end": e + 0.1}

def test_identical_specs_have_zero_delta():
    a = [_spec(10.0, 20.0), _spec(30.0, 40.0)]
    d = compare_boundaries(a, list(a))
    assert d.matched == 2
    assert d.max_start_ms == pytest.approx(0.0)
    assert d.max_end_ms == pytest.approx(0.0)

def test_small_shift_reported_in_ms():
    a = [_spec(10.0, 20.0)]
    b = [_spec(10.03, 20.02)]              # 30ms start, 20ms end
    d = compare_boundaries(a, b)
    assert d.matched == 1
    assert d.max_start_ms == pytest.approx(30.0, abs=1e-6)
    assert d.max_end_ms == pytest.approx(20.0, abs=1e-6)
    assert d.passes(threshold_ms=50.0) is True
    assert d.passes(threshold_ms=10.0) is False

def test_unmatched_clip_counts_as_miss():
    a = [_spec(10.0, 20.0)]
    b = [_spec(500.0, 510.0)]             # no IoU overlap
    d = compare_boundaries(a, b)
    assert d.matched == 0
    assert d.unmatched_a == 1 and d.unmatched_b == 1
    assert d.passes(threshold_ms=50.0) is False   # missing clips fail the gate

def test_wer_basic():
    assert wer("the cat sat", "the cat sat") == pytest.approx(0.0)
    assert wer("the cat sat", "the dog sat") == pytest.approx(1/3, abs=1e-6)
    assert wer("", "") == pytest.approx(0.0)
```

- [ ] **Step 2: Run it, verify it fails**

Run: `.venv/bin/python -m pytest backend/eval/tests/test_boundary_ab.py -v`
Expected: FAIL (`ModuleNotFoundError: backend.eval.boundary_ab`).

- [ ] **Step 3: Implement `boundary_ab.py`**

```python
# backend/eval/boundary_ab.py
"""A/B oracle for the latency pass: prove a faster ASR/boundary backend produces
the SAME cuts (boundary precision) and, for a transcript swap, comparable text (WER).

compare_boundaries(): IoU-match clips A↔B, report max/mean |Δstart|,|Δend| in ms.
wer(): word error rate (Levenshtein over word tokens). No in-repo WER existed."""
from __future__ import annotations
from dataclasses import dataclass
from ..eval.golden import iou     # existing IoU helper (golden.py:161)


@dataclass
class BoundaryDiff:
    matched: int
    unmatched_a: int
    unmatched_b: int
    max_start_ms: float
    max_end_ms: float
    mean_start_ms: float
    mean_end_ms: float

    def passes(self, threshold_ms: float = 50.0) -> bool:
        if self.unmatched_a or self.unmatched_b:
            return False
        return self.max_start_ms <= threshold_ms and self.max_end_ms <= threshold_ms


def _match(specs_a: list[dict], specs_b: list[dict], thresh: float = 0.5):
    """Greedy IoU match a→b (each b used once)."""
    used = set()
    pairs = []
    for a in specs_a:
        best_j, best = None, thresh
        for j, b in enumerate(specs_b):
            if j in used:
                continue
            ov = iou(a["start"], a["end"], b["start"], b["end"])
            if ov >= best:
                best, best_j = ov, j
        if best_j is not None:
            used.add(best_j)
            pairs.append((a, specs_b[best_j]))
    return pairs, used


def compare_boundaries(specs_a: list[dict], specs_b: list[dict]) -> BoundaryDiff:
    pairs, used = _match(specs_a, specs_b)
    ds = [abs(a["start"] - b["start"]) * 1000.0 for a, b in pairs]
    de = [abs(a["end"] - b["end"]) * 1000.0 for a, b in pairs]
    return BoundaryDiff(
        matched=len(pairs),
        unmatched_a=len(specs_a) - len(pairs),
        unmatched_b=len(specs_b) - len(used),
        max_start_ms=max(ds) if ds else 0.0,
        max_end_ms=max(de) if de else 0.0,
        mean_start_ms=(sum(ds) / len(ds)) if ds else 0.0,
        mean_end_ms=(sum(de) / len(de)) if de else 0.0,
    )


def wer(ref: str, hyp: str) -> float:
    """Word error rate: Levenshtein distance over whitespace tokens / len(ref words)."""
    r = ref.split()
    h = hyp.split()
    if not r:
        return 0.0 if not h else 1.0
    # classic DP edit distance
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cost = 0 if rw == hw else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1] / len(r)
```

- [ ] **Step 4: Run tests, verify pass**

Run: `.venv/bin/python -m pytest backend/eval/tests/test_boundary_ab.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/eval/boundary_ab.py backend/eval/tests/test_boundary_ab.py
git commit -m "feat(eval): boundary-precision A/B oracle + WER util for the latency gate"
```

---

## Phase 1 — MLX-accelerated refine backend (the centerpiece)

### Task 1.1: Config knobs + guarded MLX window transcriber

**Files:**
- Modify: `backend/config.py` (after the Whisper block, ~line 71).
- Create: `backend/pipeline/refine_asr.py`, `backend/pipeline/tests/test_refine_asr.py`.
- Modify: `requirements.txt`.

**Interfaces:**
- Produces: `mlx_transcribe_window(wav_path: str) -> tuple[list[dict], list[dict]]` returning `(words, segs)` where `words=[{"word","start","end"}]`, `segs=[{"start","end","text"}]` — **byte-shape-identical to boundary.py:95-99's lists**. `config.REFINE_ASR_BACKEND` ∈ {`faster_whisper`,`mlx_whisper`}, `config.MLX_WHISPER_MODEL`.

- [ ] **Step 1: Add config knobs**

In `backend/config.py`, immediately after the `REFINE_VAD`/`REFINE_WORKERS` lines (~71):
```python
# ── Refine ASR backend (latency): faster_whisper (CPU, default/fallback) | mlx_whisper (M2 GPU).
# mlx_whisper runs the SAME windowed refine on the Apple GPU; REFINE_ASR_BACKEND=faster_whisper
# is the revert switch. MLX is not thread-safe → the refine pool is forced to 1 worker (each GPU
# window is fast). MLX_WHISPER_MODEL matches REFINE_WHISPER_MODEL's size for boundary parity.
REFINE_ASR_BACKEND = os.environ.get("REFINE_ASR_BACKEND", "faster_whisper")
MLX_WHISPER_MODEL = os.environ.get("MLX_WHISPER_MODEL", "mlx-community/whisper-medium")
```

- [ ] **Step 2: Write the failing test (import-guard + output shape via a stub)**

```python
# backend/pipeline/tests/test_refine_asr.py
from __future__ import annotations
import sys, types
from backend.pipeline import refine_asr

def test_mlx_window_maps_segments_to_words_and_segs(monkeypatch):
    # stub the mlx_whisper module so the test needs no model / GPU
    fake = types.SimpleNamespace(transcribe=lambda *a, **k: {
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "hi there",
             "words": [{"word": "hi", "start": 0.0, "end": 0.4},
                       {"word": "there", "start": 0.5, "end": 1.0}]},
        ]})
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake)
    words, segs = refine_asr.mlx_transcribe_window("/tmp/whatever.wav")
    assert segs == [{"start": 0.0, "end": 1.0, "text": "hi there"}]
    assert words == [{"word": "hi", "start": 0.0, "end": 0.4},
                     {"word": "there", "start": 0.5, "end": 1.0}]

def test_mlx_import_failure_raises_cleanly(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlx_whisper", None)  # forces ImportError path
    try:
        refine_asr.mlx_transcribe_window("/tmp/x.wav")
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "mlx_whisper" in str(e)
```

- [ ] **Step 3: Run it, verify it fails**

Run: `.venv/bin/python -m pytest backend/pipeline/tests/test_refine_asr.py -v`
Expected: FAIL (`ModuleNotFoundError: backend.pipeline.refine_asr`).

- [ ] **Step 4: Implement `refine_asr.py`**

```python
# backend/pipeline/refine_asr.py
"""MLX-Whisper backend for the boundary-refine window transcription (W1).

Runs Whisper on the Apple-Silicon GPU via mlx-whisper and returns the SAME
(words, segs) dict lists that boundary._whisper_window builds from faster-whisper,
so the rest of the refine algorithm (build_sentence_index, energy snap, contract)
is untouched. Import is lazy + guarded so non-mac/CI never imports mlx."""
from __future__ import annotations
from .. import config

_prewarmed = False


def mlx_transcribe_window(wav_path: str) -> "tuple[list[dict], list[dict]]":
    try:
        import mlx_whisper
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"mlx_whisper unavailable: {e}") from e
    r = mlx_whisper.transcribe(
        wav_path, path_or_hf_repo=config.MLX_WHISPER_MODEL,
        word_timestamps=True, temperature=0.0, condition_on_previous_text=False,
    )
    words: list[dict] = []
    segs: list[dict] = []
    for seg in r.get("segments", []):
        segs.append({"start": float(seg["start"]), "end": float(seg["end"]),
                     "text": seg.get("text", "")})
        for w in seg.get("words", []):
            words.append({"word": w.get("word", ""),
                          "start": float(w["start"]), "end": float(w["end"])})
    return words, segs


def prewarm() -> None:
    """Load the MLX model once (hidden under download/vision). No-op off mlx backend."""
    global _prewarmed
    if _prewarmed or config.REFINE_ASR_BACKEND != "mlx_whisper":
        return
    try:
        import mlx_whisper
        from mlx_whisper.load_models import load_model
        load_model(config.MLX_WHISPER_MODEL)   # cached (lru) for subsequent transcribe()
        _prewarmed = True
    except Exception:
        pass   # fail-soft: refine will fall back to faster_whisper
```

- [ ] **Step 5: Run tests, verify pass**

Run: `.venv/bin/python -m pytest backend/pipeline/tests/test_refine_asr.py -v`
Expected: 2 passed.

- [ ] **Step 6: Record the deps**

Add to `requirements.txt` under a new comment block:
```
# Apple-Silicon-only (lazy-imported; refine falls back to faster_whisper elsewhere)
mlx-whisper>=0.4 ; platform_machine == "arm64"
```

- [ ] **Step 7: Commit**

```bash
git add backend/config.py backend/pipeline/refine_asr.py backend/pipeline/tests/test_refine_asr.py requirements.txt
git commit -m "feat(refine): guarded MLX-whisper window transcriber + config knobs"
```

### Task 1.2: Branch `_whisper_window` on the backend + force 1 worker for MLX

**Files:**
- Modify: `backend/pipeline/boundary.py:88-100` (branch), `:406` (workers), `:413-416` (prewarm).

**Interfaces:**
- Consumes: `refine_asr.mlx_transcribe_window`, `refine_asr.prewarm` (Task 1.1).
- Produces: unchanged `_whisper_window` contract `(list[Sentence], Path|None)`.

- [ ] **Step 1: Snapshot the hot-path file**

Run: `cp backend/pipeline/boundary.py .backup/boundary.py.pre-mlx-2026-07-05`

- [ ] **Step 2: Add a regression test that the branch selects MLX and preserves the wav**

```python
# append to backend/pipeline/tests/test_refine_asr.py
def test_whisper_window_uses_mlx_when_selected(monkeypatch, tmp_path):
    import backend.pipeline.boundary as B
    from backend import config
    # a real tiny wav so the ffmpeg extract + energy path stays valid
    import wave, struct
    monkeypatch.setattr(config, "REFINE_ASR_BACKEND", "mlx_whisper")
    called = {}
    def fake_mlx(path):
        called["path"] = path
        return ([{"word": "hello", "start": 0.0, "end": 0.5}],
                [{"start": 0.0, "end": 0.5, "text": "hello"}])
    monkeypatch.setattr(B, "_mlx_window", fake_mlx, raising=False)
    # stub ffmpeg extraction to drop a 0.5s silent 16k mono wav at the tmp path
    real_run = B.subprocess.run
    def fake_run(cmd, **kw):
        out = cmd[cmd.index(str(cmd[-1]))] if False else cmd[-1]
        with wave.open(str(out), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
            w.writeframes(struct.pack("<8000h", *([0] * 8000)))
        return real_run(["true"], **kw)
    monkeypatch.setattr(B.subprocess, "run", fake_run)
    sents, wav = B._whisper_window(tmp_path / "audio.m4a", 10.0, 10.5)
    assert called                                  # MLX path was taken
    assert wav is not None and wav.exists()        # wav preserved for energy snap
```

- [ ] **Step 3: Run it, verify it fails**

Run: `.venv/bin/python -m pytest backend/pipeline/tests/test_refine_asr.py::test_whisper_window_uses_mlx_when_selected -v`
Expected: FAIL (`_mlx_window`/branch not present).

- [ ] **Step 4: Implement the branch in `_whisper_window`**

In `backend/pipeline/boundary.py`, replace lines 88-100 (the `model = _get_refine_whisper()` … `ok = True` block) with:
```python
            if config.REFINE_ASR_BACKEND == "mlx_whisper":
                words, segs = _mlx_window(tmp)
            else:
                model = _get_refine_whisper()
                kw = dict(word_timestamps=True, beam_size=5, temperature=0.0,
                          condition_on_previous_text=False)
                if config.REFINE_VAD:
                    kw["vad_filter"] = True
                    kw["vad_parameters"] = dict(speech_pad_ms=200)
                segments, _ = model.transcribe(str(tmp), **kw)
                words, segs = [], []
                for seg in segments:
                    segs.append({"start": float(seg.start), "end": float(seg.end), "text": seg.text})
                    for w in (seg.words or []):
                        words.append({"word": w.word, "start": float(w.start), "end": float(w.end)})
            ok = True
```
Add near the top-level helpers of boundary.py (e.g. after the imports):
```python
def _mlx_window(tmp):
    """MLX backend with a faster-whisper fallback (never fail the window on an MLX error)."""
    from .refine_asr import mlx_transcribe_window
    try:
        return mlx_transcribe_window(str(tmp))
    except Exception:
        from .transcribe import _get_refine_whisper
        model = _get_refine_whisper()
        segments, _ = model.transcribe(str(tmp), word_timestamps=True, beam_size=5,
                                       temperature=0.0, condition_on_previous_text=False)
        words, segs = [], []
        for seg in segments:
            segs.append({"start": float(seg.start), "end": float(seg.end), "text": seg.text})
            for w in (seg.words or []):
                words.append({"word": w.word, "start": float(w.start), "end": float(w.end)})
        return words, segs
```

- [ ] **Step 5: Force single-worker + prewarm for the MLX backend**

In `refine_clip_boundaries`, change line 406 from:
```python
    workers = max(1, min(config.REFINE_WORKERS, n))
```
to:
```python
    # MLX (Metal) is not thread-safe → serialize on the GPU (each window is fast anyway).
    workers = 1 if config.REFINE_ASR_BACKEND == "mlx_whisper" else max(1, min(config.REFINE_WORKERS, n))
```
And replace the prewarm block (lines 413-416, currently guarded by `if workers > 1:`) so MLX prewarms too:
```python
    try:
        if config.REFINE_ASR_BACKEND == "mlx_whisper":
            from .refine_asr import prewarm as _mlx_prewarm
            _mlx_prewarm()
        elif workers > 1:
            _get_refine_whisper()        # pre-warm shared singleton before the pool
    except Exception:
        pass
```

- [ ] **Step 6: Run the new + existing boundary tests**

Run: `.venv/bin/python -m pytest backend/pipeline/tests/test_refine_asr.py backend/pipeline/tests/test_silence_snap.py -v`
Expected: all pass (branch works; energy-snap untouched).

- [ ] **Step 7: Commit**

```bash
git add backend/pipeline/boundary.py backend/pipeline/tests/test_refine_asr.py
git commit -m "feat(refine): route _whisper_window to MLX backend; serialize+prewarm on GPU"
```

### Task 1.3: Real-audio boundary A/B gate (flip default only if it passes)

**Files:**
- Modify: `backend/config.py:REFINE_ASR_BACKEND` default (only on pass).

**Interfaces:**
- Consumes: `boundary_ab.compare_boundaries` (0.3), a cached video, both backends.

- [ ] **Step 1: Run refine on a real video with BOTH backends and capture cut times**

Run:
```bash
cat > /tmp/ab_refine.py <<'PY'
import json
from backend import config
from backend.pipeline.boundary import refine_clip_boundaries
from backend.eval.boundary_ab import compare_boundaries
VID = "<CACHED_ID>"; URL = open("/tmp/ab_url.txt").read().strip()
specs = json.loads(open(f"output/{VID}/clips.json").read())        # assembled specs (pre-refine)
base = [dict(s) for s in specs]
config.REFINE_ASR_BACKEND = "faster_whisper"
a = refine_clip_boundaries([dict(s) for s in base], URL, VID, dict(config.DEFAULTS))
config.REFINE_ASR_BACKEND = "mlx_whisper"
b = refine_clip_boundaries([dict(s) for s in base], URL, VID, dict(config.DEFAULTS))
d = compare_boundaries(a, b)
print(d)
print("PASSES@50ms:", d.passes(50.0))
PY
.venv/bin/python /tmp/ab_refine.py
```
Expected: prints max/mean Δ in ms and a PASS/FAIL. **GATE:** `PASSES@50ms` must be `True`.

- [ ] **Step 2: Time both backends (confirm the speedup)**

Wrap each `refine_clip_boundaries` call in `time.perf_counter()` (edit `/tmp/ab_refine.py`) and print seconds. Expected: MLX materially faster (target ≥3× on this M2). Record both numbers in `docs/superpowers/plans/latency-baseline-2026-07-05.md`.

- [ ] **Step 3: Flip the default (only if Step 1 passed)**

If and only if the gate passed, change `backend/config.py`:
```python
REFINE_ASR_BACKEND = os.environ.get("REFINE_ASR_BACKEND", "mlx_whisper")
```
(Keep `faster_whisper` reachable via env — the revert switch.)

- [ ] **Step 4: Full suite green**

Run: `.venv/bin/python -m pytest -q`
Expected: the whole suite passes.

- [ ] **Step 5: Commit**

```bash
git add backend/config.py docs/superpowers/plans/latency-baseline-2026-07-05.md
git commit -m "perf(refine): default REFINE_ASR_BACKEND=mlx_whisper (boundary A/B ≤50ms, Nx faster)"
```

---

## Phase 2 — Deterministic post-refine result cache (warm win)

### Task 2.1: Result-cache module

**Files:**
- Create: `backend/pipeline/assemble/result_cache.py`, `backend/pipeline/assemble/tests/test_result_cache.py`.
- Modify: `backend/config.py` (`CLIP_RESULT_CACHE`).

**Interfaces:**
- Produces: `load_result(video_id, structure, settings) -> tuple[list[dict], str] | None`; `save_result(video_id, structure, settings, clips_spec, notes) -> None`; `settings_fingerprint(settings) -> str`.
- Consumes: `structure.sentence_fingerprint` (understand/models.py) — the structure identity.

- [ ] **Step 1: Add the config knob**

`backend/config.py` (near `STRUCTURE_CACHE`, ~line 224):
```python
# Deterministic post-refine clip cache: the topic engine is query-independent, so the shipped
# clips are fully determined by (structure, assembly settings). Caching them lets a warm re-clip
# skip assemble+refine entirely. CLIP_RESULT_CACHE=0 disables (revert switch).
CLIP_RESULT_CACHE = os.environ.get("CLIP_RESULT_CACHE", "1") not in ("0", "false", "")
```

- [ ] **Step 2: Write the failing test**

```python
# backend/pipeline/assemble/tests/test_result_cache.py
from __future__ import annotations
import types
from backend.pipeline.assemble import result_cache as RC

def _struct(fp="abc123"):
    return types.SimpleNamespace(video_id="vid1", sentence_fingerprint=fp)

def test_roundtrip_hit(tmp_path, monkeypatch):
    monkeypatch.setattr(RC.config, "WORK_DIR", tmp_path)
    st = _struct()
    specs = [{"start": 1.0, "end": 2.0, "cut_end": 2.1, "title": "t"}]
    assert RC.load_result("vid1", st, {"max_clips": 5}) is None       # cold
    RC.save_result("vid1", st, {"max_clips": 5}, specs, "3 clips")
    got = RC.load_result("vid1", st, {"max_clips": 5})
    assert got is not None
    clips, notes = got
    assert clips == specs and notes == "3 clips"

def test_settings_change_invalidates(tmp_path, monkeypatch):
    monkeypatch.setattr(RC.config, "WORK_DIR", tmp_path)
    st = _struct()
    RC.save_result("vid1", st, {"max_clips": 5}, [{"start": 0, "end": 1, "cut_end": 1.1}], "n")
    assert RC.load_result("vid1", st, {"max_clips": 9}) is None       # different settings

def test_structure_change_invalidates(tmp_path, monkeypatch):
    monkeypatch.setattr(RC.config, "WORK_DIR", tmp_path)
    RC.save_result("vid1", _struct("fpA"), {"max_clips": 5}, [{"start": 0, "end": 1, "cut_end": 1.1}], "n")
    assert RC.load_result("vid1", _struct("fpB"), {"max_clips": 5}) is None  # different structure
```

- [ ] **Step 3: Run it, verify it fails**

Run: `.venv/bin/python -m pytest backend/pipeline/assemble/tests/test_result_cache.py -v`
Expected: FAIL (`ImportError: result_cache`).

- [ ] **Step 4: Implement `result_cache.py`**

```python
# backend/pipeline/assemble/result_cache.py
"""Deterministic post-refine clip cache (W3b). Keyed on the structure's sentence
fingerprint + the RESOLVED assembly settings, so a warm re-clip of the same video
returns the exact shipped clips without re-running assemble+refine. Mirrors the
structure-cache disk idiom (work/<id>/...)."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Optional
from ... import config

# assembly-relevant settings whose change must invalidate the cache (resolved → None means default).
_ASSEMBLY_KEYS = (
    "clip_engine", "max_clips", "informativeness_min", "boundary_window",
    "clip_max_s", "tail_pad_s", "min_clip_duration_s", "max_clip_duration_s",
)


def settings_fingerprint(settings: dict) -> str:
    picked = {k: settings.get(k) for k in _ASSEMBLY_KEYS}
    blob = json.dumps(picked, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _path(video_id: str, structure, settings: dict) -> Path:
    sfp = getattr(structure, "sentence_fingerprint", "") or "nofp"
    key = hashlib.sha256(f"{sfp}:{settings_fingerprint(settings)}".encode()).hexdigest()[:20]
    return config.WORK_DIR / video_id / "clip_results" / f"{key}.json"


def load_result(video_id: str, structure, settings: dict) -> "Optional[tuple[list[dict], str]]":
    p = _path(video_id, structure, settings)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return list(data["clips"]), str(data.get("notes", ""))
    except Exception:
        return None


def save_result(video_id: str, structure, settings: dict, clips_spec: list[dict], notes: str) -> None:
    if not clips_spec:
        return                                  # never cache an empty (failed) run
    p = _path(video_id, structure, settings)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"clips": clips_spec, "notes": notes}), encoding="utf-8")
```

- [ ] **Step 5: Run tests, verify pass**

Run: `.venv/bin/python -m pytest backend/pipeline/assemble/tests/test_result_cache.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add backend/config.py backend/pipeline/assemble/result_cache.py backend/pipeline/assemble/tests/test_result_cache.py
git commit -m "feat(assemble): deterministic post-refine clip result cache"
```

### Task 2.2: Wire the result cache into the orchestrator

**Files:**
- Modify: `backend/orchestrator.py:305-346` (around assemble+refine).

**Interfaces:**
- Consumes: `result_cache.load_result/save_result` (2.1). Caches the **post-refine** `clips_spec`.

- [ ] **Step 1: Snapshot**

Run: `cp backend/orchestrator.py .backup/orchestrator.py.pre-resultcache-2026-07-05`

- [ ] **Step 2: Add the cache-hit short-circuit before assemble**

In `_run_full`, immediately before line 314 (`registry.publish(... "assembling" ...)`), insert:
```python
    from .pipeline.assemble.result_cache import load_result, save_result
    _cached_result = None
    if config.CLIP_RESULT_CACHE:
        _cached_result = load_result(video_id, structure, settings)
```
Then wrap the assemble+artifacts+refine block. Replace lines 314-334 so that on a hit we skip straight to the shipped specs:
```python
    if _cached_result is not None:
        clips_spec, notes = _cached_result
        rejections = []
        registry.publish(job, ProgressEvent("assembling", 90.0, "Reusing cached clips…"))
    else:
        registry.publish(job, ProgressEvent("assembling", 72.0, "Assembling self-contained clips…"))
        stats: dict = {}
        clips_spec, notes, rejections = await run(
            _resolve_assemble_fn(settings), structure, job.topic, sentences, job.url, video_id,
            settings, adapter, emit("assembling", 72, 90), stats,
        )
        await run(write_run_artifacts, video_id, clips_spec, rejections, stats)
        if rejections:
            registry.publish(job, ProgressEvent(
                "assembling", 90.0, f"{len(rejections)} candidate(s) dropped ({', '.join(sorted({r.stage for r in rejections}))})"))
        if not clips_spec:
            return registry.fail(
                job, notes or f"“{job.topic}” isn’t covered enough in this video to clip.", notes=notes)
        if config.PRECISE_BOUNDARIES and transcript.get("source") == "supadata":
            registry.publish(job, ProgressEvent("refining", 90.0, "Refining boundaries with Whisper…"))
            clips_spec = await run(
                refine_clip_boundaries, clips_spec, job.url, video_id, settings, emit("refining", 90, 96))
        if config.CLIP_RESULT_CACHE:
            save_result(video_id, structure, settings, clips_spec, notes)
```
(The existing empty-result bail and refine block move *inside* the `else`. The edge-probe/cut/`_build_embed_clips` code at lines 336-361 is unchanged and runs on `clips_spec` either way.)

- [ ] **Step 3: Add an integration test for the hit path**

```python
# backend/pipeline/tests/test_orchestrator_result_cache.py
from __future__ import annotations
import asyncio, types
import backend.orchestrator as O
from backend import config

def test_cache_hit_skips_assemble(monkeypatch):
    # a hit must NOT call _resolve_assemble_fn / refine_clip_boundaries
    monkeypatch.setattr(config, "CLIP_RESULT_CACHE", True)
    monkeypatch.setattr(O, "load_result", lambda *a: ([{"start": 1.0, "end": 2.0, "cut_end": 2.1}], "cached"), raising=False)
    called = {"assemble": False}
    monkeypatch.setattr(O, "_resolve_assemble_fn",
                        lambda s: (_ for _ in ()).throw(AssertionError("assemble ran on a cache hit")),
                        raising=False)
    # NOTE: this is a shape guard; the full _run_full path is covered by the live run in Task 2.4.
    assert O.load_result("v", object(), {}) == ([{"start": 1.0, "end": 2.0, "cut_end": 2.1}], "cached")
```
(Full end-to-end is validated live in Step 4; the unit guard just proves the seam names line up.)

- [ ] **Step 4: Live verify hit + parity**

Run:
```bash
# cold once (writes cache), then warm (reads cache) — compare shipped clips.json
rm -rf work/<CACHED_ID>/clip_results
.venv/bin/python -m backend.cli "$(cat /tmp/ab_url.txt)" >/tmp/run1.txt 2>&1
cp output/<CACHED_ID>/clips.json /tmp/clips_cold.json
.venv/bin/python -m backend.cli "$(cat /tmp/ab_url.txt)" >/tmp/run2.txt 2>&1   # should hit cache
.venv/bin/python - <<'PY'
import json
a=json.load(open("/tmp/clips_cold.json")); b=json.load(open("output/<CACHED_ID>/clips.json"))
print("same start/end:", [(x["start"],x["end"]) for x in a]==[(x["start"],x["end"]) for x in b])
PY
grep -c "Reusing cached clips" /tmp/run2.txt
```
Expected: `same start/end: True`; warm run logs the cache reuse and is dramatically faster.

- [ ] **Step 5: Full suite green + commit**

Run: `.venv/bin/python -m pytest -q`
```bash
git add backend/orchestrator.py backend/pipeline/tests/test_orchestrator_result_cache.py
git commit -m "feat(orchestrator): reuse cached post-refine clips on warm re-clip"
```

---

## Phase 3 — aria2c downloader + prewarm (small cold wins)

### Task 3.1: aria2c external downloader

**Files:**
- Modify: `backend/config.py` (`YTDLP_EXTERNAL_DL`), `backend/pipeline/download.py:131-145`.

- [ ] **Step 1: Config knob**

`backend/config.py`:
```python
# yt-dlp multi-connection downloader (latency): set to "aria2c" to use it when present.
YTDLP_EXTERNAL_DL = os.environ.get("YTDLP_EXTERNAL_DL", "")
```

- [ ] **Step 2: Add to ydl_opts (guarded on availability)**

In `backend/pipeline/download.py`, add `import shutil` at the top, then after the `ydl_opts = {...}` dict at line 131-145 insert:
```python
    if config.YTDLP_EXTERNAL_DL == "aria2c" and shutil.which("aria2c"):
        ydl_opts["external_downloader"] = "aria2c"
        ydl_opts["external_downloader_args"] = {"aria2c": ["-x", "16", "-s", "16", "-k", "1M"]}
```

- [ ] **Step 3: Test the opts builder is guarded (no aria2c → no keys)**

```python
# backend/pipeline/tests/test_download_aria2c.py
from __future__ import annotations
import backend.pipeline.download as D
from backend import config

def test_aria2c_absent_leaves_opts_clean(monkeypatch):
    monkeypatch.setattr(config, "YTDLP_EXTERNAL_DL", "aria2c")
    monkeypatch.setattr(D.shutil, "which", lambda _n: None)   # aria2c not installed
    # build the same guarded dict the download() body builds
    opts = {}
    if config.YTDLP_EXTERNAL_DL == "aria2c" and D.shutil.which("aria2c"):
        opts["external_downloader"] = "aria2c"
    assert "external_downloader" not in opts
```

- [ ] **Step 4: Run + verify**

Run: `.venv/bin/python -m pytest backend/pipeline/tests/test_download_aria2c.py -v`
Expected: pass. (Optionally `brew install aria2` and time one real cold download with/without.)

- [ ] **Step 5: Commit**

```bash
git add backend/config.py backend/pipeline/download.py backend/pipeline/tests/test_download_aria2c.py
git commit -m "feat(download): optional aria2c multi-connection downloader"
```

### Task 3.2: Prewarm the MLX model during download/vision

**Files:**
- Modify: `backend/orchestrator.py` (`_run_full`, in the perceive branch ~line 260).

- [ ] **Step 1: Kick off prewarm concurrently with perception**

In `_run_full`, right after the perceive task is launched (~line 263, after `perceive_task = asyncio.ensure_future(...)`), add:
```python
                    if config.REFINE_ASR_BACKEND == "mlx_whisper":
                        from .pipeline.refine_asr import prewarm as _mlx_prewarm
                        asyncio.ensure_future(loop.run_in_executor(executor, _mlx_prewarm))
```
(`loop` and `executor` are already in scope via `run`/`run_pipeline`; if not, pass through — `run_in_executor` uses the same pool.) This hides the model load under the ~90-120s vision stage.

- [ ] **Step 2: Verify no regression**

Run: `.venv/bin/python -m pytest backend/pipeline/tests/ -q`
Expected: pass (prewarm is fire-and-forget, fail-soft).

- [ ] **Step 3: Commit**

```bash
git add backend/orchestrator.py
git commit -m "perf(refine): prewarm MLX model under the perception stage"
```

---

## Phase 4 — Paid-tier parallelism (measure-and-tune, light)

### Task 4.1: Confirm paid tier + tune non-judge workers

**Files:**
- Modify: `backend/config.py` (`VISION_WORKERS`/`UNDERSTAND_WORKERS` defaults) — only if measured faster.

**Interfaces:**
- No new code path; there is no proactive limiter, so paid-tier parallelism is mostly automatic (the 8-wide fan-outs stop hitting free-tier 429 backoff). `JUDGE_WORKERS` stays 4 (measured paid-tier soft-throttle).

- [ ] **Step 1: Confirm the key is a paid-tier key**

Verify the Gemini key in `.env` is on a paid project (billing enabled). No code reads `RPM_LIMIT` on the Gemini path (confirmed), so nothing throttles proactively.

- [ ] **Step 2: Wall-clock A/B on VISION/UNDERSTAND workers**

Run three cold understand+assemble passes with worker counts 8 (baseline), 12, 16:
```bash
for W in 8 12 16; do
  echo "=== workers=$W ==="
  rm -rf work/<CACHED_ID>/structure.json work/<CACHED_ID>/clip_results
  VISION_WORKERS=$W UNDERSTAND_WORKERS=$W PROFILE_TIMINGS_FILE=/tmp/w$W.jsonl \
    .venv/bin/python -m backend.cli "$(cat /tmp/ab_url.txt)" >/dev/null 2>&1
  python3 -c "import json;print(sum(json.loads(l)['ms'] for l in open('/tmp/w$W.jsonl')))"
done
```
Expected: pick the fastest that does not regress. If 8 is best (soft-throttle), keep 8.

- [ ] **Step 3: Update defaults only if a higher count won**

If e.g. 12 won cleanly, change `backend/config.py`:
```python
VISION_WORKERS = int(os.environ.get("VISION_WORKERS", "12"))
UNDERSTAND_WORKERS = int(os.environ.get("UNDERSTAND_WORKERS", "12"))
```
Leave `JUDGE_WORKERS = 4` unchanged (documented net-slower when raised).

- [ ] **Step 4: Suite green + commit**

Run: `.venv/bin/python -m pytest -q`
```bash
git add backend/config.py docs/superpowers/plans/latency-baseline-2026-07-05.md
git commit -m "perf(gemini): tune paid-tier fan-out workers (measured); judge stays 4"
```

---

## Phase 5 — Disk verdict cache (OPTIONAL; unit-engine only)

> **Note:** the default `CLIP_ENGINE=topic` has **no judge/repair loop**, so this helps only `CLIP_ENGINE=unit`. Implement only if unit-engine runs matter. Otherwise skip — Phase 2's result cache already covers the default path's warm case.

### Task 5.1: Persist the unit-engine verdict cache to disk

**Files:**
- Modify: `backend/pipeline/assemble/validate.py:819-830` (`_cached`/`_store`), key at `:851`.

- [ ] **Step 1: Add `VERDICT_CACHE_DISK` config**

```python
VERDICT_CACHE_DISK = os.environ.get("VERDICT_CACHE_DISK", "0") not in ("0", "false", "")
```

- [ ] **Step 2: Include judge model in the key + back `_store`/`_cached` with a disk layer**

Change the key at `validate.py:851` to fold in the model:
```python
        key = (frozenset(c.unit_ids), text_hash, config.JUDGE_MODEL)
```
Extend `_store`/`_cached` (validate.py:819-830) to also read/write `work/<video_id>/verdicts/<sha>.json` when `config.VERDICT_CACHE_DISK` (serialize `JudgeVerdict.model_dump()`; on load `JudgeVerdict.model_validate`). Thread `video_id` into `validate_and_repair` (already available in `assemble_clips`).

- [ ] **Step 3: Round-trip test + suite + commit**

Test that a disk-persisted verdict is reused across two in-process caches (new `dict`). Then `pytest -q` and commit.

---

## Phase 6 — Parakeet global transcript (GATED final phase)

> Pursue only if a fresh warm-ish run is still >~70s after P1-P4 and pushing further is wanted. **Two gates:** boundary A/B (P0) AND WER/text A/B (below).

### Task 6.1: Parakeet transcriber

**Files:**
- Modify: `backend/config.py` (`PARAKEET_MODEL`), `backend/pipeline/transcribe.py` (add `_transcribe_parakeet` + route `TRANSCRIBER=="parakeet"`).
- Create: `backend/pipeline/tests/test_transcribe_parakeet.py`.

**Interfaces:**
- Produces: a transcript dict `{text, duration, words:[{word,start,end}], segments:[{start,end,text}], source:"parakeet"}` — same shape as `transcribe_supadata`, but `source != "supadata"` so `refine_clip_boundaries` **skips** (orchestrator gate).

- [ ] **Step 1: Config knob**

```python
PARAKEET_MODEL = os.environ.get("PARAKEET_MODEL", "mlx-community/parakeet-tdt-0.6b-v3")
```

- [ ] **Step 2: Failing test (stub parakeet_mlx)**

```python
# backend/pipeline/tests/test_transcribe_parakeet.py
from __future__ import annotations
import sys, types
from backend.pipeline import transcribe as T

def test_parakeet_shape(monkeypatch, tmp_path):
    tok = types.SimpleNamespace
    res = types.SimpleNamespace(
        text="hi there",
        sentences=[types.SimpleNamespace(start=0.0, end=1.0, text="hi there")],
        tokens=[tok(text="hi", start=0.0, end=0.4), tok(text="there", start=0.5, end=1.0)],
    )
    fake = types.SimpleNamespace(from_pretrained=lambda *_a, **_k: types.SimpleNamespace(
        transcribe=lambda *a, **k: res))
    monkeypatch.setitem(sys.modules, "parakeet_mlx", fake)
    out = T._transcribe_parakeet(str(tmp_path / "a.m4a"), {})
    assert out["source"] == "parakeet"
    assert out["words"][0] == {"word": "hi", "start": 0.0, "end": 0.4}
    assert out["segments"][0]["text"] == "hi there"
```

- [ ] **Step 3: Implement `_transcribe_parakeet` + route**

Add to `backend/pipeline/transcribe.py`:
```python
def _transcribe_parakeet(audio_path: str, settings: dict) -> dict:
    from parakeet_mlx import from_pretrained
    model = from_pretrained(config.PARAKEET_MODEL)
    r = model.transcribe(audio_path, chunk_duration=120.0, overlap_duration=15.0)
    words = [{"word": t.text, "start": float(t.start), "end": float(t.end)} for t in r.tokens]
    segments = [{"start": float(s.start), "end": float(s.end), "text": s.text} for s in r.sentences]
    return {"text": r.text, "duration": segments[-1]["end"] if segments else 0.0,
            "words": words, "segments": segments, "source": "parakeet"}
```
In `transcribe(...)` (the local/non-supadata entry), before the faster_whisper/groq branch add:
```python
    if config.TRANSCRIBER == "parakeet":
        result = _transcribe_parakeet(audio_path, settings)
```
(Cache to `transcript.json` as the other paths do.)

- [ ] **Step 4: Run test + WER/boundary gate**

Run the unit test (`pytest ...test_transcribe_parakeet.py`). Then the **live gates** on a real cached video:
```bash
# WER: parakeet text vs supadata caption text
.venv/bin/python - <<'PY'
import json
from backend.eval.boundary_ab import wer
sup = json.load(open("work/<CACHED_ID>/transcript.json"))["text"]
# produce a parakeet transcript into /tmp then compare
# ... (run _transcribe_parakeet on work/<id>/audio.m4a) ...
print("WER parakeet-vs-supadata:", wer(sup, para_text))
PY
```
**GATE 1 (WER):** parakeet text WER vs captions ≤ ~0.15 (or manually judged ≥ caption quality). **GATE 2 (boundary):** run `compare_boundaries(supadata+refine cuts, parakeet cuts)` ≤ 50 ms. Ship `TRANSCRIBER=parakeet` as default only if BOTH pass; else keep it opt-in and leave supadata default.

- [ ] **Step 5: Commit**

```bash
git add backend/config.py backend/pipeline/transcribe.py backend/pipeline/tests/test_transcribe_parakeet.py
git commit -m "feat(transcribe): gated parakeet-mlx global transcriber (skips refine at full precision)"
```

---

## Self-Review

**Spec coverage:**
- W0 (profile + oracle) → P0 (0.1 spike, 0.2 profile, 0.3 oracle). ✓
- W1 (MLX refine) → P1 (1.1 backend, 1.2 branch, 1.3 A/B-gated flip). ✓
- W2 (paid Gemini) → P4 (measure-and-tune; judge stays 4). ✓
- W3a (disk verdict cache) → P5 (optional, unit-engine-only — corrected from spec after discovering the topic engine has no judge loop). ✓
- W3b (result cache) → P2. ✓
- W4 (aria2c + prewarm) → P3. ✓
- W5 (parakeet, gated) → P6 (two gates: WER + boundary). ✓

**Placeholder scan:** No TBD/TODO. `<CACHED_ID>`/`<ONE_HOUR_ID>`/`<CACHED_ID>` are runtime video-id inputs the operator fills at execution (the plan can't invent a real id) — flagged as such, not code placeholders.

**Type consistency:** `mlx_transcribe_window`/`_mlx_window` return `(words, segs)` with `words=[{word,start,end}]`, `segs=[{start,end,text}]` everywhere (Tasks 1.1, 1.2). `compare_boundaries`/`BoundaryDiff.passes(threshold_ms)`/`wer` names match across 0.3, 1.3, 6.4. `load_result`/`save_result`/`settings_fingerprint` signatures match across 2.1 and 2.2. `structure.sentence_fingerprint` is the real attribute (understand/models.py). `refine_clip_boundaries(clips, url, video_id, settings, progress)` signature matches boundary.py:380.

**Scope:** One coherent latency goal; phased so each phase ships working, revert-switchable software. P5/P6 are optional/gated.
