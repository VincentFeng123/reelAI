# Precise, Silence-Snapped Clip Cutting — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every clip boundary land in the inter-word silence (never on a word), so clips never clip off the first or last word ("on-the-dot cutting").

**Architecture:** The precise-boundary pass (`backend/pipeline/boundary.py`) transcribes a small audio window around each rough clip boundary with faster-whisper, snaps the start to a sentence start and the end to a period-terminated sentence end, then places the actual cut **in the inter-word gap** (word-gap midpoint capped by a small pad, refined to the quietest 10 ms audio frame). When no period-terminated end with a usable trailing gap is in the window, the window **grows and re-transcribes** (bounded). A ±100 ms Whisper timestamp error is then harmless — it moves the cut *within* the silence, not into speech.

**Tech Stack:** Python 3, faster-whisper (CTranslate2), numpy, stdlib `wave`, ffmpeg (already wired), pytest.

## Global Constraints

- **Run tests with the venv, from the `clips/` dir:** `.venv/bin/python -m pytest backend/<path> -v` (NOT bare `pytest`). Full suite: `.venv/bin/python -m pytest backend/ -q` (baseline: 701 passed).
- **No new pip dependencies.** numpy (2.2.6) and stdlib `wave` are already available; the energy step is pure-numpy.
- **Two boundary systems exist — touch only one.** `backend/pipeline/refine.py` (sentence-level text snapping: `_snap_one`, `refine_and_snap`, `_is_weak_end`, `_is_weak_start`) is **NOT** part of this work. The BND1 and onset guards live there. Do **not** modify `refine.py`. This plan changes only `backend/pipeline/boundary.py`, `backend/pipeline/transcribe.py`, and `backend/config.py`.
- **Do NOT regress** `backend/pipeline/tests/test_bnd1_boundary_guards.py`, `backend/pipeline/tests/test_onset_start_guard.py`, or `backend/pipeline/tests/test_discourse_onset.py` (they cover `refine.py`, untouched here) — nor the merged discourse-onset work (merge `1dff8b3`).
- **`backend/pipeline/tests/test_boundary_safety.py` and `test_refine_parallel_order.py` ARE fair game** — they test `boundary.py`, which this plan rewrites. Update their assertions to the new behavior, but **preserve their intent**: direction-safety invariants (start never later than `rough+1s`; end never earlier than `rough-1s`) and parallel order-invariance (reversed completion == serial).
- **End-extension policy (handoff §8 — CONFIRMED with user: HYBRID):** When the chosen end `E` is a complete (period-terminated) sentence but has no trailing pause (gap `< SILENCE_MIN_GAP_S`), keep the tight cut at `E`; only nudge the clip end forward to the next period-terminated sentence *with* a usable gap when that alternative is within a small budget `END_EXTEND_MAX_S` (~one sentence). If none is within budget, take the best-available cut at the original end (flagged). The **START** never re-selects an earlier sentence — backward window growth only widens the transcription window so the previous word becomes visible.
- **`cut.py` is unchanged** (verify empirically in §6 of the spec; only change if a real-cut measurement proves truncation). Not in this plan's scope.
- **Commit message footer (every commit):**
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  ```
- **Branch:** `precise-cutting` (already created off `main`).

## File Structure

- `backend/config.py` — add 5 constants + 2 DEFAULTS changes (Task 1).
- `backend/pipeline/transcribe.py` — add `_get_refine_whisper()` singleton (Task 2).
- `backend/pipeline/boundary.py` — the core rewrite (Tasks 3–8): energy snap, richer `_whisper_window`, silence-aware `_pick_start`/`_pick_end`, window-extending `_refine_start`/`_refine_end`, and `refine_clip_boundaries` wiring.
- New tests under `backend/pipeline/tests/`: `test_precise_cutting_config.py`, `test_refine_whisper_model.py`, `test_energy_snap.py`, `test_silence_snap.py`, `test_boundary_extension.py`, `test_whisper_window.py`. Updated: `test_boundary_safety.py`, `test_refine_parallel_order.py`.

## Shared Module Contract (target end-state of `boundary.py`)

All tasks converge on these signatures. Later tasks rely on names/types defined here.

```python
from collections import namedtuple
# time: absolute video seconds for the cut. flags: warnings to merge onto the clip.
# satisfied: True when the cut lands in a real (measurable) inter-word gap — i.e. no window
# growth is needed. False signals _refine_* to grow the window and re-transcribe.
Pick = namedtuple("Pick", ["time", "flags", "satisfied"])

def _energy_min_snap(wav_path, win_start: float, a: float, b: float,
                     frame_ms: int = 10) -> "float | None": ...
def _whisper_window(audio, win_start: float, win_end: float) -> "tuple[list[Sentence], Path | None]": ...
def _gap_before(sents, idx: int) -> "tuple[float, float] | None": ...   # (prev.end, S.start)
def _gap_after(sents, idx: int) -> "tuple[float, float] | None": ...    # (E.end, next.start)
def _snap_start_cut(s_start, prev_end, lead_pad, energy_fn) -> "tuple[float, tuple]": ...
def _snap_end_cut(e_end, next_start, tail_pad, energy_fn) -> "tuple[float, tuple]": ...
def _pick_start(sents, rough, pad, keep_first=False, *, lead_pad, gap_min, energy_fn=None) -> Pick: ...
def _pick_end(sents, rough, pad, allow_qe, *, tail_pad, gap_min, end_extend_max, energy_fn=None) -> Pick: ...
def _refine_start(audio, s0, pad, *, lead_pad, gap_min, max_search) -> Pick: ...
def _refine_end(audio, e0, pad, allow_qe, *, tail_pad, gap_min, end_extend_max, max_search, max_clip_end) -> Pick: ...
def _refine_one(c, audio, pad, allow_qe, tail_pad, lead_pad, gap_min, end_extend_max, max_search, max_clip_dur) -> dict: ...
```

`energy_fn` is `Callable[[float, float], float | None]` taking absolute `(a, b)` seconds and returning the snapped absolute time, or `None` to skip (the window origin is closed over by `_refine_*`). Passing `energy_fn=None` disables energy snapping → pure gap math (used by the offline `_pick_*` unit tests).

---

### Task 1: Config — pads, refine model, search bounds

**Files:**
- Modify: `backend/config.py` (near L55 `WHISPER_MODEL`; near L59-69 precise-boundary block; DEFAULTS L127/L122-161)
- Test: `backend/pipeline/tests/test_precise_cutting_config.py`

**Interfaces:**
- Produces: `config.REFINE_WHISPER_MODEL: str`, `config.REFINE_VAD: bool`, `config.MAX_BOUNDARY_SEARCH_S: float`, `config.SILENCE_MIN_GAP_S: float`, `config.END_EXTEND_MAX_S: float`, `config.DEFAULTS["tail_pad_s"] == 0.15`, `config.DEFAULTS["lead_pad_s"] == 0.06`.

- [ ] **Step 1: Write the failing test**

Create `backend/pipeline/tests/test_precise_cutting_config.py`:
```python
"""Precise-cutting config surface (Task 1). Pure constants — no audio/whisper."""
from __future__ import annotations

from backend import config


def test_pads_updated():
    assert config.DEFAULTS["tail_pad_s"] == 0.15      # was 0.05 — trailing cushion
    assert config.DEFAULTS["lead_pad_s"] == 0.06      # new — leading cushion


def test_refine_model_defaults_to_medium():
    # default "medium"; empty REFINE_WHISPER_MODEL falls back to WHISPER_MODEL
    assert config.REFINE_WHISPER_MODEL in ("medium", config.WHISPER_MODEL)
    assert isinstance(config.REFINE_WHISPER_MODEL, str) and config.REFINE_WHISPER_MODEL


def test_refine_vad_on_by_default():
    assert config.REFINE_VAD is True


def test_search_and_gap_bounds():
    assert config.MAX_BOUNDARY_SEARCH_S == 45.0
    assert config.SILENCE_MIN_GAP_S == 0.12
    assert config.END_EXTEND_MAX_S == 8.0
    # the hybrid end-nudge budget must not exceed the window search cap
    assert config.END_EXTEND_MAX_S <= config.MAX_BOUNDARY_SEARCH_S
    # the initial window half-width stays the drift cap
    assert config.BOUNDARY_PAD_S == 10.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest backend/pipeline/tests/test_precise_cutting_config.py -v`
Expected: FAIL (`AttributeError: module ... has no attribute 'REFINE_WHISPER_MODEL'`, and `KeyError`/assert on `lead_pad_s`).

- [ ] **Step 3: Write minimal implementation**

In `backend/config.py`, under the faster-whisper block (after L57, the `WHISPER_DEVICE` line), add:
```python
# Precise-boundary REFINE uses a dedicated (usually larger) Whisper model than full
# transcription: the window is small so cost is modest, and word timestamps are more precise.
# Default "medium"; set REFINE_WHISPER_MODEL="" to fall back to WHISPER_MODEL. (~1.5 GB one-time
# download on first CPU/int8 run.)
REFINE_WHISPER_MODEL = os.environ.get("REFINE_WHISPER_MODEL", "medium") or WHISPER_MODEL
# VAD on the refine window pads speech segments (speech_pad_ms), giving usable silence margin.
REFINE_VAD = os.environ.get("REFINE_VAD", "1") not in ("0", "false", "")
```

In the precise-boundary block (after L63 `BOUNDARY_PAD_S`), add:
```python
# When no period-terminated sentence end with a usable trailing gap is in the window, the refine
# pass GROWS the window (pad→2·pad→4·pad…) and re-transcribes, up to this forward/backward reach.
MAX_BOUNDARY_SEARCH_S = float(os.environ.get("MAX_BOUNDARY_SEARCH_S", "45"))
# A word-gap must be at least this wide to count as a clean cut site (the cut lands inside it).
SILENCE_MIN_GAP_S = float(os.environ.get("SILENCE_MIN_GAP_S", "0.12"))
# HYBRID end policy (handoff §8): when the chosen complete-sentence end has no trailing pause,
# advance the clip END to the next period-terminated sentence WITH a gap only within this budget
# (~one sentence). Beyond it, best-available tight cut at the original end.
END_EXTEND_MAX_S = float(os.environ.get("END_EXTEND_MAX_S", "8"))
```

In `DEFAULTS`, change the `tail_pad_s` line (L127) and add `lead_pad_s` immediately after:
```python
    "tail_pad_s": 0.15,   # trailing cushion — the cut lands in the gap AFTER the last word
    "lead_pad_s": 0.06,   # leading cushion — the cut lands in the gap BEFORE the first word
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest backend/pipeline/tests/test_precise_cutting_config.py -v`
Expected: PASS (4 tests).

Also confirm no config-consumer regressed: `.venv/bin/python -m pytest backend/pipeline/tests/test_config_targets.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/config.py backend/pipeline/tests/test_precise_cutting_config.py
git commit -m "$(cat <<'EOF'
feat(boundary): add precise-cutting config (pads, refine model, search bounds)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Dedicated refine-model Whisper singleton

**Files:**
- Modify: `backend/pipeline/transcribe.py` (after `_get_whisper`, ~L38)
- Test: `backend/pipeline/tests/test_refine_whisper_model.py`

**Interfaces:**
- Consumes: `config.REFINE_WHISPER_MODEL`, `config.REFINE_WORKERS`, `config.WHISPER_MODEL`, existing `_get_whisper()`.
- Produces: `transcribe._get_refine_whisper()` — a threadsafe CTranslate2 singleton for `REFINE_WHISPER_MODEL`; returns the same object as `_get_whisper()` when the two model names are equal (no double-load).

- [ ] **Step 1: Write the failing test**

Create `backend/pipeline/tests/test_refine_whisper_model.py`:
```python
"""Refine-model Whisper singleton (Task 2). WhisperModel is stubbed — no real model loads."""
from __future__ import annotations

import backend.config as config
import backend.pipeline.transcribe as tr


class _FakeModel:
    def __init__(self, name, device=None, compute_type=None, num_workers=1):
        self.name, self.num_workers = name, num_workers


def _install_fake(monkeypatch):
    built = []

    class _FakeWM(_FakeModel):
        def __init__(self, name, **kw):
            super().__init__(name, **kw)
            built.append(name)

    # faster_whisper.WhisperModel is imported INSIDE the getters
    import faster_whisper
    monkeypatch.setattr(faster_whisper, "WhisperModel", _FakeWM)
    monkeypatch.setattr(tr, "_whisper_model", None, raising=False)
    monkeypatch.setattr(tr, "_refine_whisper_model", None, raising=False)
    return built


def test_refine_singleton_builds_refine_model(monkeypatch):
    built = _install_fake(monkeypatch)
    monkeypatch.setattr(config, "REFINE_WHISPER_MODEL", "medium")
    monkeypatch.setattr(config, "WHISPER_MODEL", "small")
    m = tr._get_refine_whisper()
    assert m.name == "medium"
    assert "medium" in built
    assert tr._get_refine_whisper() is m          # cached singleton


def test_refine_reuses_full_singleton_when_models_match(monkeypatch):
    built = _install_fake(monkeypatch)
    monkeypatch.setattr(config, "REFINE_WHISPER_MODEL", "small")
    monkeypatch.setattr(config, "WHISPER_MODEL", "small")
    full = tr._get_whisper()
    refine = tr._get_refine_whisper()
    assert refine is full                          # same object, one load
    assert built.count("small") == 1


def test_refine_singleton_uses_refine_workers(monkeypatch):
    _install_fake(monkeypatch)
    monkeypatch.setattr(config, "REFINE_WHISPER_MODEL", "medium")
    monkeypatch.setattr(config, "WHISPER_MODEL", "small")
    monkeypatch.setattr(config, "REFINE_WORKERS", 4)
    m = tr._get_refine_whisper()
    assert m.num_workers == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest backend/pipeline/tests/test_refine_whisper_model.py -v`
Expected: FAIL (`AttributeError: module ... has no attribute '_get_refine_whisper'`).

- [ ] **Step 3: Write minimal implementation**

In `backend/pipeline/transcribe.py`, after `_get_whisper()` (L38) add:
```python
# Dedicated singleton for the boundary-REFINE pass, which uses a (usually larger) model than full
# transcription for more precise word timestamps. Keyed by REFINE_WHISPER_MODEL. When it equals
# WHISPER_MODEL we reuse the full-transcription singleton so the model loads only once. Threadsafe
# once built (CTranslate2 num_workers); refine_clip_boundaries pre-warms it before its thread pool.
_refine_whisper_model = None


def _get_refine_whisper():
    global _refine_whisper_model
    if config.REFINE_WHISPER_MODEL == config.WHISPER_MODEL:
        return _get_whisper()
    if _refine_whisper_model is None:
        from faster_whisper import WhisperModel
        _refine_whisper_model = WhisperModel(
            config.REFINE_WHISPER_MODEL, device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE, num_workers=max(1, config.REFINE_WORKERS),
        )
    return _refine_whisper_model
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest backend/pipeline/tests/test_refine_whisper_model.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/pipeline/transcribe.py backend/pipeline/tests/test_refine_whisper_model.py
git commit -m "$(cat <<'EOF'
feat(transcribe): dedicated refine-model Whisper singleton

Reuses the full-transcription singleton when REFINE_WHISPER_MODEL == WHISPER_MODEL.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Energy-minimum snap (the "on the dot" point)

**Files:**
- Modify: `backend/pipeline/boundary.py` (add imports + `_energy_min_snap`)
- Test: `backend/pipeline/tests/test_energy_snap.py`

**Interfaces:**
- Produces: `boundary._energy_min_snap(wav_path, win_start, a, b, frame_ms=10) -> float | None`. Reads the 16 kHz mono window wav, returns the absolute time of the lowest-RMS `frame_ms` frame within `[a, b]`. Returns `None` on unreadable wav / empty or sub-frame interval (caller falls back to the pad/midpoint value).

- [ ] **Step 1: Write the failing test**

Create `backend/pipeline/tests/test_energy_snap.py`:
```python
"""Energy-minimum snap (Task 3). Synthesizes a tone+silence wav — no whisper, no ffmpeg."""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from backend.pipeline.boundary import _energy_min_snap


def _write_wav(path: Path, sr: int, samples: np.ndarray) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)               # int16
        wf.setframerate(sr)
        wf.writeframes(samples.astype(np.int16).tobytes())


def _tone_then_silence(sr=16000):
    # [0.0,0.5): loud 200 Hz tone; [0.5,1.0): silence
    t = np.arange(int(sr * 1.0)) / sr
    sig = (8000 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    sig[int(sr * 0.5):] = 0.0
    return sig


def test_snaps_to_silent_frame(tmp_path):
    sr = 16000
    wav = tmp_path / "w.wav"
    _write_wav(wav, sr, _tone_then_silence(sr))
    # gap spans the tone→silence transition; the quietest frame is in the silent half
    t = _energy_min_snap(wav, win_start=0.0, a=0.4, b=0.95)
    assert t is not None
    assert 0.5 <= t <= 0.95           # landed in the silence, not in the tone


def test_absolute_offset_respected(tmp_path):
    sr = 16000
    wav = tmp_path / "w.wav"
    _write_wav(wav, sr, _tone_then_silence(sr))
    # window starts at video-time 100.0 → gap [100.4,100.95] maps to wav [0.4,0.95]
    t = _energy_min_snap(wav, win_start=100.0, a=100.4, b=100.95)
    assert t is not None and 100.5 <= t <= 100.95


def test_missing_wav_returns_none():
    assert _energy_min_snap(None, 0.0, 0.1, 0.2) is None
    assert _energy_min_snap(Path("/nonexistent/x.wav"), 0.0, 0.1, 0.2) is None


def test_subframe_interval_returns_none(tmp_path):
    sr = 16000
    wav = tmp_path / "w.wav"
    _write_wav(wav, sr, _tone_then_silence(sr))
    assert _energy_min_snap(wav, 0.0, 0.5, 0.5) is None       # empty
    assert _energy_min_snap(wav, 0.0, 0.5, 0.5005) is None    # < one 10 ms frame
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest backend/pipeline/tests/test_energy_snap.py -v`
Expected: FAIL (`ImportError: cannot import name '_energy_min_snap'`).

- [ ] **Step 3: Write minimal implementation**

At the top of `backend/pipeline/boundary.py`, add to the imports:
```python
import wave

import numpy as np
```

Add the function (place it above `_pick_start`):
```python
def _energy_min_snap(wav_path, win_start: float, a: float, b: float,
                     frame_ms: int = 10) -> "float | None":
    """Absolute time of the lowest-RMS ``frame_ms`` frame within ``[a, b]`` — the quietest instant
    in the pause. ``win_start`` is the wav's absolute start time. Returns None on a bad/short read
    so the caller keeps its pad/midpoint fallback (never raises)."""
    if wav_path is None or b <= a:
        return None
    try:
        with wave.open(str(wav_path), "rb") as wf:
            sr = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    except Exception:
        return None
    if samples.size == 0:
        return None
    frame = max(1, int(sr * frame_ms / 1000))
    lo = max(0, int((a - win_start) * sr))
    hi = min(samples.size, int((b - win_start) * sr))
    if hi - lo < frame:
        return None
    best_i, best_rms = lo, None
    for i in range(lo, hi - frame + 1, frame):
        seg = samples[i:i + frame]
        rms = float(np.sqrt(np.mean(seg * seg)))
        if best_rms is None or rms < best_rms:
            best_rms, best_i = rms, i
    return win_start + (best_i + frame / 2) / sr
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest backend/pipeline/tests/test_energy_snap.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/pipeline/boundary.py backend/pipeline/tests/test_energy_snap.py
git commit -m "$(cat <<'EOF'
feat(boundary): energy-minimum snap to the quietest frame in a gap

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Structural refactor — `_whisper_window` returns the wav; `Pick` return type

Pure refactor: identical boundary values, but `_whisper_window` now returns `(sents, wav_path)` (keeping the extracted wav for the energy step, cleaned up by the caller), uses the refine model + VAD, and `_pick_start`/`_pick_end` return a `Pick` namedtuple. Behavior is unchanged so the suite stays green; Tasks 5–7 fill in the silence logic.

**Files:**
- Modify: `backend/pipeline/boundary.py` (`_whisper_window` L71-96, `_pick_start` L99-111, `_pick_end` L114-128, `_refine_one` L151-176)
- Modify (tests): `backend/pipeline/tests/test_boundary_safety.py`, `backend/pipeline/tests/test_refine_parallel_order.py`
- Test (new): `backend/pipeline/tests/test_whisper_window.py`

**Interfaces:**
- Consumes: `transcribe._get_refine_whisper` (Task 2), `config.REFINE_VAD`.
- Produces: `Pick` namedtuple; `_whisper_window(audio, win_start, win_end) -> (list[Sentence], Path | None)`; `_pick_start(...) -> Pick`, `_pick_end(...) -> Pick` (values identical to the pre-refactor floats, `flags=()`, `satisfied=True`).

- [ ] **Step 1: Write the failing test**

Create `backend/pipeline/tests/test_whisper_window.py`:
```python
"""_whisper_window: refine model + VAD kwargs, and returns (sents, wav_path). Model+ffmpeg stubbed."""
from __future__ import annotations

from pathlib import Path

import backend.pipeline.boundary as bmod


class _Seg:
    def __init__(self, start, end, text, words):
        self.start, self.end, self.text, self.words = start, end, text, words


class _W:
    def __init__(self, word, start, end):
        self.word, self.start, self.end = word, start, end


def test_whisper_window_returns_sents_and_wavpath(monkeypatch, tmp_path):
    captured = {}

    class _Model:
        def transcribe(self, path, **kw):
            captured["path"] = path
            captured["kw"] = kw
            segs = [_Seg(0.0, 1.0, "Hi there.", [_W("Hi", 0.0, 0.3), _W("there.", 0.4, 0.9)])]
            return segs, object()

    # ffmpeg is not run: fake subprocess.run and make the temp wav "exist"
    monkeypatch.setattr(bmod, "_get_refine_whisper", lambda: _Model())
    monkeypatch.setattr(bmod.subprocess, "run", lambda *a, **k: None)
    real_mkstemp = bmod.tempfile.mkstemp
    made = {}

    def _fake_mkstemp(suffix="", dir=None):
        fd, p = real_mkstemp(suffix=suffix, dir=dir)
        made["path"] = p
        return fd, p
    monkeypatch.setattr(bmod.tempfile, "mkstemp", _fake_mkstemp)

    sents, wav = bmod._whisper_window(Path(tmp_path) / "audio.m4a", 10.0, 20.0)

    assert wav is not None and str(wav) == made["path"]        # wav path returned, NOT deleted
    assert Path(wav).exists()
    assert captured["kw"].get("condition_on_previous_text") is False
    assert captured["kw"].get("temperature") == 0.0
    assert captured["kw"].get("beam_size") == 5
    assert captured["kw"].get("word_timestamps") is True
    assert captured["kw"].get("vad_filter") is True            # REFINE_VAD default on
    assert [s.text for s in sents]                              # sentences built + shifted
    Path(wav).unlink(missing_ok=True)


def test_pick_helpers_return_pick(monkeypatch):
    from backend.pipeline.boundary import Pick, _pick_end, _pick_start
    from backend.pipeline.sentences import Sentence

    def _s(i, a, b, term="."):
        return Sentence(idx=i, text=f"s{i}", start=a, end=b, terminator=term,
                        ends_with_period=(term in ".?!"), word_start_idx=i, word_end_idx=i,
                        align_confidence=1.0)

    pe = _pick_end([_s(0, 40.0, 44.5), _s(1, 44.6, 46.2)], rough=45.0, pad=10.0, allow_qe=False,
                   tail_pad=0.15, gap_min=0.12, end_extend_max=8.0)
    assert isinstance(pe, Pick)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest backend/pipeline/tests/test_whisper_window.py -v`
Expected: FAIL (`_whisper_window` returns a list, not a tuple; `Pick` not importable; `_pick_end` missing kwargs).

- [ ] **Step 3: Write minimal implementation**

In `backend/pipeline/boundary.py`:

Add the `Pick` type near the top (after the imports):
```python
from collections import namedtuple

Pick = namedtuple("Pick", ["time", "flags", "satisfied"])
```

Replace the `from .transcribe import _get_whisper` import with:
```python
from .transcribe import _get_refine_whisper, _get_whisper
```

Rewrite `_whisper_window` to keep the wav and use the refine model + VAD:
```python
def _whisper_window(audio: Path, win_start: float, win_end: float) -> "tuple[list[Sentence], Path | None]":
    win_start = max(0.0, win_start)
    win_len = max(1.0, win_end - win_start)
    tmp = Path(tempfile.mkstemp(suffix=".wav", dir=str(audio.parent))[1])
    ok = False
    try:
        subprocess.run(
            [config.FFMPEG_BIN, "-nostdin", "-y", "-ss", f"{win_start:.3f}", "-t", f"{win_len:.3f}",
             "-i", str(audio), "-ar", "16000", "-ac", "1", str(tmp)],
            capture_output=True,
        )
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
    finally:
        if not ok:
            tmp.unlink(missing_ok=True)
    if not words:
        tmp.unlink(missing_ok=True)
        return [], None
    sents = build_sentence_index({"words": words, "segments": segs})
    for s in sents:  # shift to absolute video time
        s.start += win_start
        s.end += win_start
    return sents, tmp        # caller owns tmp cleanup (energy step reads it first)
```

Wrap the existing `_pick_start`/`_pick_end` bodies to return `Pick` (values unchanged, new kwargs accepted but not yet used for snapping):
```python
def _pick_start(sents, rough, pad, keep_first=False, *, lead_pad=0.06, gap_min=0.12,
                energy_fn=None) -> Pick:
    if not sents:
        return Pick(rough, (), True)
    pool = sents if keep_first else sents[1:]
    starts = [s.start for s in pool] or [s.start for s in sents]
    before = [x for x in starts if x <= rough + 1.0]
    if not before:
        return Pick(rough, (), True)
    cand = max(before)
    return Pick(cand if abs(cand - rough) <= pad else rough, (), True)


def _pick_end(sents, rough, pad, allow_qe, *, tail_pad=0.15, gap_min=0.12, end_extend_max=8.0,
              energy_fn=None) -> Pick:
    if not sents:
        return Pick(rough, (), True)
    ends = [s.end for s in sents if s.is_valid_end(allow_qe)]
    if not ends:
        return Pick(rough, (), True)
    after = [x for x in ends if x >= rough] or [x for x in ends if x >= rough - 1.0]
    if not after:
        return Pick(rough, (), True)
    cand = min(after)
    return Pick(cand if abs(cand - rough) <= pad else rough, (), True)
```

Update `_refine_one` to unpack the window tuple, use `.time`, and clean up the wav:
```python
def _refine_one(c: dict, audio: Path, pad: float, allow_qe: bool, tail_pad: float) -> dict:
    s0, e0 = float(c["start"]), float(c["end"])
    wavs: list = []
    try:
        def _win(a, b):
            sents, wav = _whisper_window(audio, a, b)
            if wav is not None:
                wavs.append(wav)
            return sents
        if e0 - s0 <= 2 * pad + 20:
            w = _win(s0 - pad, e0 + pad)
            new_start = _pick_start(w, s0, pad, keep_first=(s0 - pad <= 0.0)).time
            new_end = _pick_end(w, e0, pad, allow_qe).time
        else:
            new_start = _pick_start(_win(s0 - pad, s0 + pad), s0, pad,
                                    keep_first=(s0 - pad <= 0.0)).time
            new_end = _pick_end(_win(e0 - pad, e0 + pad), e0, pad, allow_qe).time
        if new_end <= new_start:
            new_start, new_end = s0, e0
    except Exception:
        new_start, new_end = s0, e0
    finally:
        for wav in wavs:
            wav.unlink(missing_ok=True)
    d = dict(c)
    d["start"] = round(new_start, 3)
    d["end"] = round(new_end, 3)
    d["cut_end"] = round(new_end + tail_pad, 3)
    return d
```

- [ ] **Step 4: Update the two boundary.py tests to the new shapes (no behavior change)**

In `backend/pipeline/tests/test_boundary_safety.py`, the direct `_pick_end`/`_pick_start` calls now return `Pick`. Update the assertions to `.time` and pass the new kwargs, values unchanged:
```python
def test_pick_end_never_moves_earlier_than_window_floor():
    sents = [_sent(0, 28.0, 31.0)]
    assert _pick_end(sents, rough=45.0, pad=10.0, allow_qe=False,
                     tail_pad=0.15, gap_min=0.12, end_extend_max=8.0).time == 45.0


def test_pick_end_normal_path_unchanged():
    sents = [_sent(0, 40.0, 44.5), _sent(1, 44.6, 46.2)]
    assert _pick_end(sents, rough=45.0, pad=10.0, allow_qe=False,
                     tail_pad=0.15, gap_min=0.12, end_extend_max=8.0).time == 46.2


def test_pick_start_never_moves_later_than_window_ceiling():
    sents = [_sent(0, 43.0, 44.0), _sent(1, 58.0, 60.0)]
    assert _pick_start(sents, rough=45.0, pad=10.0, lead_pad=0.06, gap_min=0.12).time == 45.0


def test_pick_start_normal_path_unchanged():
    sents = [_sent(0, 35.0, 36.0), _sent(1, 44.2, 47.0), _sent(2, 47.1, 49.0)]
    assert _pick_start(sents, rough=45.0, pad=10.0, lead_pad=0.06, gap_min=0.12).time == 44.2


def test_pick_start_keep_first_at_video_start():
    sents = [_sent(0, 0.0, 3.0), _sent(1, 3.1, 6.0)]
    assert _pick_start(sents, rough=0.5, pad=10.0, keep_first=True,
                       lead_pad=0.06, gap_min=0.12).time == 0.0
    assert _pick_start(sents, rough=0.5, pad=10.0, lead_pad=0.06, gap_min=0.12).time == 0.5
```
(These `_pick_*` values will change once Task 5/6 add silence-snapping; that's expected — those tasks update these same assertions again. Here they only lose the `Pick` wrapper.)

In `backend/pipeline/tests/test_refine_parallel_order.py`, the mock `_whisper_window` must return `(sents, None)`. Update `_make_windows` and `_make_reversed_windows` to return the tuple:
```python
    def _win(audio, win_start, win_end):
        s0 = round(win_start + PAD, 3)
        _idx, tgt_start, tgt_end = spec[s0]
        return [_sent(0, s0 - 20.0, s0 - 19.0, ""), _sent(1, tgt_start, tgt_end, ".")], None
```
and in `_make_reversed_windows`, end with `return sents, None` instead of `return sents`. All numeric expectations stay (Task 4 does not change boundary values). Order-invariance assertions unchanged.

- [ ] **Step 5: Run tests to verify they pass**

Run:
```
.venv/bin/python -m pytest backend/pipeline/tests/test_whisper_window.py backend/pipeline/tests/test_boundary_safety.py backend/pipeline/tests/test_refine_parallel_order.py -v
```
Expected: PASS (all). Then confirm no wider regression: `.venv/bin/python -m pytest backend/pipeline/ -q`.

- [ ] **Step 6: Commit**

```bash
git add backend/pipeline/boundary.py backend/pipeline/tests/test_whisper_window.py backend/pipeline/tests/test_boundary_safety.py backend/pipeline/tests/test_refine_parallel_order.py
git commit -m "$(cat <<'EOF'
refactor(boundary): _whisper_window returns wav + refine model/VAD; Pick return type

Structural only — boundary values unchanged. Prepares the silence-snap + extension work.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Silence-aware START placement (`_pick_start`)

**Files:**
- Modify: `backend/pipeline/boundary.py` (`_gap_before`, `_snap_start_cut`, `_pick_start`)
- Modify (tests): `backend/pipeline/tests/test_boundary_safety.py`
- Test (new): `backend/pipeline/tests/test_silence_snap.py` (start cases)

**Interfaces:**
- Consumes: `_energy_min_snap` (Task 3).
- Produces: `_gap_before(sents, idx)`, `_snap_start_cut(s_start, prev_end, lead_pad, energy_fn)`, and the silence-aware `_pick_start` — the START cut = `max(S.start - lead_pad, midpoint(prev.end, S.start))`, energy-snapped within `[that, S.start]`, never into `prev`. `satisfied=False` only when the chosen start's previous word is not visible in the window and the window began after video-time 0 (→ backward growth in Task 7).

**Behavior contract (drives the tests):**
- Choose `S` = latest sentence start `≤ rough+1s` (direction-safe), within `pad` of `rough`; else keep `rough`.
- Leading gap = `(prev.end, S.start)` where `prev` = the sentence before `S` in `sents`.
- `pad_cut = max(S.start - lead_pad, midpoint(prev.end, S.start))`. Energy-snap within `[pad_cut, S.start]`; clamp result to `[pad_cut, S.start]`. Never `< prev.end`, never `> S.start`.
- `prev` absent + `keep_first` (window at t=0) → `cut = max(0, S.start - lead_pad)` (still energy-snapped), `satisfied=True`.
- `prev` absent + not `keep_first` → `cut = max(0, S.start - lead_pad)`, `satisfied=False`, flag `start_prev_unseen` (Task 7 grows backward).

- [ ] **Step 1: Write the failing test**

Create `backend/pipeline/tests/test_silence_snap.py` with the START cases:
```python
"""Silence-aware start/end placement (Tasks 5-6). Offline: energy_fn=None → pure gap math."""
from __future__ import annotations

from backend.pipeline.boundary import _pick_start
from backend.pipeline.sentences import Sentence

LEAD, TAIL, GAP = 0.06, 0.15, 0.12


def _s(i, a, b, term="."):
    return Sentence(idx=i, text=f"s{i}", start=a, end=b, terminator=term,
                    ends_with_period=(term in ".?!"), word_start_idx=i, word_end_idx=i,
                    align_confidence=1.0)


def test_start_cuts_into_gap_never_into_prev_word():
    # prev ends 9.50, S starts 10.00 → 0.50 s gap (> 2*lead) → cut at S.start-lead = 9.94
    sents = [_s(0, 8.0, 9.50), _s(1, 10.00, 12.0)]
    p = _pick_start(sents, rough=10.0, pad=10.0, lead_pad=LEAD, gap_min=GAP, energy_fn=None)
    assert abs(p.time - 9.94) < 1e-6
    assert 9.50 < p.time < 10.00        # strictly inside the gap
    assert p.satisfied


def test_start_small_gap_uses_midpoint():
    # prev ends 9.96, S starts 10.00 → 0.04 s gap (< 2*lead) → midpoint 9.98
    sents = [_s(0, 8.0, 9.96), _s(1, 10.00, 12.0)]
    p = _pick_start(sents, rough=10.0, pad=10.0, lead_pad=LEAD, gap_min=GAP, energy_fn=None)
    assert abs(p.time - 9.98) < 1e-6
    assert 9.96 < p.time < 10.00


def test_start_prev_absent_keep_first_uses_lead_pad():
    sents = [_s(0, 0.0, 3.0), _s(1, 3.1, 6.0)]
    p = _pick_start(sents, rough=0.02, pad=10.0, keep_first=True, lead_pad=LEAD, gap_min=GAP)
    assert p.time == 0.0                # max(0, 0.0 - lead) clamped to 0
    assert p.satisfied


def test_start_prev_unseen_triggers_growth():
    # window did NOT start at t=0; chosen S is sents[1] but its prev (sents[0]) is a FRAGMENT,
    # so its start is a real onset with no visible preceding word → grow backward (unsatisfied).
    # Here sents[0] is dropped as a fragment (not keep_first); S = sents[1]; prev index 0 is the
    # fragment, whose .end is far (< real gap) → treat as unseen when it is the window's first sent.
    sents = [_s(0, 90.0, 91.0, term=""), _s(1, 100.0, 103.0)]
    p = _pick_start(sents, rough=100.0, pad=10.0, keep_first=False, lead_pad=LEAD, gap_min=GAP)
    assert not p.satisfied
    assert "start_prev_unseen" in p.flags


def test_start_direction_safe_when_only_late_candidate():
    sents = [_s(0, 43.0, 44.0), _s(1, 58.0, 60.0)]      # only start 58 > rough+1
    p = _pick_start(sents, rough=45.0, pad=10.0, lead_pad=LEAD, gap_min=GAP)
    assert p.time == 45.0 and p.satisfied
```

Note on `test_start_prev_unseen_triggers_growth`: the chosen `S` is `sents[1]`; `sents[0]` is the *window's first* sentence and, when a clip's start is mid-video, the first window sentence is a cut fragment whose `.end` is not a reliable previous-word boundary. `_pick_start` treats "the sentence before `S` is the window's first sentence AND not `keep_first`" as **prev-unseen** → `satisfied=False`.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest backend/pipeline/tests/test_silence_snap.py -v`
Expected: FAIL (start cuts equal `S.start`/`rough`, not the gap-snapped values; no `start_prev_unseen`).

- [ ] **Step 3: Write minimal implementation**

In `backend/pipeline/boundary.py`, add helpers above `_pick_start`:
```python
def _gap_before(sents, idx: int) -> "tuple[float, float] | None":
    """(prev.end, S.start) for the sentence at idx, or None if idx is the window's first sentence."""
    if idx <= 0 or idx >= len(sents):
        return None
    return sents[idx - 1].end, sents[idx].start


def _snap_start_cut(s_start: float, prev_end: float, lead_pad: float, energy_fn) -> "tuple[float, tuple]":
    """Cut into the gap BEFORE s_start: max(s_start-lead_pad, midpoint(prev_end, s_start)), then
    energy-snap within [that, s_start]. Never < prev_end, never > s_start."""
    mid = (prev_end + s_start) / 2.0
    pad_cut = max(s_start - lead_pad, mid)
    cut = pad_cut
    if energy_fn is not None:
        snapped = energy_fn(pad_cut, s_start)
        if snapped is not None:
            cut = min(max(snapped, pad_cut), s_start)
    return max(prev_end, min(cut, s_start)), ()
```

Rewrite `_pick_start`:
```python
def _pick_start(sents, rough, pad, keep_first=False, *, lead_pad=0.06, gap_min=0.12,
                energy_fn=None) -> Pick:
    """Begin at a thought's onset, cutting into the leading inter-word gap (never into the prev
    word). Direction-safe: never chooses a start later than rough+1s. The START never re-selects
    an earlier sentence — if the previous word isn't visible, it asks _refine_start to grow the
    window (satisfied=False)."""
    if not sents:
        return Pick(rough, (), True)
    # candidate starts: exclude the window's first sentence as a fragment unless keep_first
    cand = [(i, s.start) for i, s in enumerate(sents) if (keep_first or i >= 1)]
    before = [(i, x) for (i, x) in cand if x <= rough + 1.0]
    if not before:
        return Pick(rough, (), True)                 # direction-safe: keep rough
    idx, s_start = max(before, key=lambda t: t[1])
    if abs(s_start - rough) > pad:
        return Pick(rough, (), True)                 # nearest onset too far → keep rough
    gap = _gap_before(sents, idx)
    if gap is None:
        cut = max(0.0, s_start - lead_pad)
        if energy_fn is not None:
            snapped = energy_fn(cut, s_start)
            if snapped is not None:
                cut = min(max(snapped, cut), s_start)
        if keep_first:
            return Pick(round(cut, 3), (), True)     # window at t=0 → real onset, no prev needed
        return Pick(round(cut, 3), ("start_prev_unseen",), False)   # prev not visible → grow back
    prev_end, _ = gap
    cut, flags = _snap_start_cut(s_start, prev_end, lead_pad, energy_fn)
    return Pick(round(cut, 3), flags, True)
```

- [ ] **Step 4: Update `test_boundary_safety.py` start assertions to the snapped values**

The two direction-safe start tests keep returning `rough` (no valid nearer onset) — unchanged. The "normal path" test now snaps into the gap. Update it:
```python
def test_pick_start_normal_path_unchanged():
    # S = sents[1] @44.2, prev = sents[0] end 36.0 → big gap → cut = 44.2 - lead(0.06) = 44.14
    sents = [_sent(0, 35.0, 36.0), _sent(1, 44.2, 47.0), _sent(2, 47.1, 49.0)]
    assert abs(_pick_start(sents, rough=45.0, pad=10.0, lead_pad=0.06, gap_min=0.12).time - 44.14) < 1e-6
```
`test_pick_start_keep_first_at_video_start` stays: `keep_first=True` → `max(0, 0-0.06)=0.0`; `keep_first=False` → `rough=0.5`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest backend/pipeline/tests/test_silence_snap.py backend/pipeline/tests/test_boundary_safety.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/pipeline/boundary.py backend/pipeline/tests/test_silence_snap.py backend/pipeline/tests/test_boundary_safety.py
git commit -m "$(cat <<'EOF'
feat(boundary): silence-aware START placement — cut into the leading gap

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Silence-aware END placement + hybrid gap policy (`_pick_end`, single window)

**Files:**
- Modify: `backend/pipeline/boundary.py` (`_gap_after`, `_snap_end_cut`, `_pick_end`)
- Modify (tests): `backend/pipeline/tests/test_boundary_safety.py`
- Test (add to): `backend/pipeline/tests/test_silence_snap.py` (end cases)

**Interfaces:**
- Consumes: `_energy_min_snap`.
- Produces: `_gap_after(sents, idx)`, `_snap_end_cut(e_end, next_start, tail_pad, energy_fn)`, and the hybrid `_pick_end`. END cut = `min(E.end + tail_pad, midpoint(E.end, next.start))`, energy-snapped within `[E.end, that]`, never into `next`.

**Behavior contract (hybrid, per handoff §8):**
- `E` = earliest period-terminated end `≥ rough` (direction-safe fallback `≥ rough-1s`). If none → `Pick(rough, (), False)` (grow window).
- If `E` is the window's **last** sentence (no `next` → gap unmeasurable) → `Pick(E.end + tail_pad, (), False)` (grow so `next` becomes visible).
- Let `gapE = next.start - E.end`. If `gapE ≥ gap_min` → tight cut at `E`, `satisfied=True`.
- Else scan later period-terminated ends `E'` with `E'.end ≤ rough + end_extend_max`; the first with a measurable gap `≥ gap_min` → cut at `E'`, flag `end_extended`, `satisfied=True`.
- Else (no in-budget alternative) → best-available tight cut at `E`, flag `tight_end_no_gap`, `satisfied=True` (don't grow — the clip stays tight per the user's hybrid choice).

- [ ] **Step 1: Write the failing test** (append to `test_silence_snap.py`)

```python
from backend.pipeline.boundary import _pick_end


def test_end_cuts_into_gap_never_into_next_word():
    # E ends 46.20, next starts 46.70 → 0.50 s gap (> 2*tail) → cut at E.end+tail = 46.35
    sents = [_s(0, 40.0, 44.5), _s(1, 44.6, 46.20), _s(2, 46.70, 49.0)]
    p = _pick_end(sents, rough=45.0, pad=10.0, allow_qe=False,
                  tail_pad=TAIL, gap_min=GAP, end_extend_max=8.0, energy_fn=None)
    assert abs(p.time - 46.35) < 1e-6
    assert 46.20 < p.time < 46.70 and p.satisfied
    assert "tight_end_no_gap" not in p.flags


def test_end_small_gap_uses_midpoint():
    # E ends 46.20, next starts 46.26 → 0.06 s gap (< 2*tail but >= gap_min? 0.06<0.12) → tight+flag
    sents = [_s(0, 44.6, 46.20), _s(1, 46.26, 49.0)]
    p = _pick_end(sents, rough=45.0, pad=10.0, allow_qe=False,
                  tail_pad=TAIL, gap_min=GAP, end_extend_max=8.0, energy_fn=None)
    # 0.06 < gap_min and no later gap within budget → tight cut at E, flagged
    assert 46.20 < p.time <= 46.23 and p.satisfied
    assert "tight_end_no_gap" in p.flags


def test_end_hybrid_nudges_to_next_gap_within_budget():
    # E @47.0 has NO gap (next @47.0); a later end @50.0 HAS a 0.5s gap, within 8s budget → advance
    sents = [_s(0, 44.0, 47.0), _s(1, 47.0, 50.0), _s(2, 50.5, 53.0)]
    p = _pick_end(sents, rough=46.5, pad=10.0, allow_qe=False,
                  tail_pad=TAIL, gap_min=GAP, end_extend_max=8.0, energy_fn=None)
    assert 50.0 < p.time < 50.5 and p.satisfied
    assert "end_extended" in p.flags


def test_end_hybrid_beyond_budget_keeps_tight():
    # E @47.0 no gap; next clean gap only at 60.0 (> 8s budget from rough 46.5) → tight at E, flagged
    sents = [_s(0, 44.0, 47.0), _s(1, 47.0, 60.0), _s(2, 60.6, 63.0)]
    p = _pick_end(sents, rough=46.5, pad=10.0, allow_qe=False,
                  tail_pad=TAIL, gap_min=GAP, end_extend_max=8.0, energy_fn=None)
    assert 47.0 <= p.time <= 47.16 and p.satisfied
    assert "tight_end_no_gap" in p.flags


def test_end_last_sentence_gap_unmeasurable_grows():
    sents = [_s(0, 44.6, 46.2)]           # E is the only/last sentence → no next → grow
    p = _pick_end(sents, rough=45.0, pad=10.0, allow_qe=False,
                  tail_pad=TAIL, gap_min=GAP, end_extend_max=8.0, energy_fn=None)
    assert not p.satisfied


def test_end_no_valid_end_grows():
    sents = [_s(0, 44.6, 46.2, term=""), _s(1, 46.3, 48.0, term="")]   # no period anywhere
    p = _pick_end(sents, rough=45.0, pad=10.0, allow_qe=False,
                  tail_pad=TAIL, gap_min=GAP, end_extend_max=8.0, energy_fn=None)
    assert not p.satisfied
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest backend/pipeline/tests/test_silence_snap.py -v`
Expected: FAIL (end still returns `min(after)` float-equivalent, no gap snap / hybrid flags / grow signals).

- [ ] **Step 3: Write minimal implementation**

Add helpers above `_pick_end`:
```python
def _gap_after(sents, idx: int) -> "tuple[float, float] | None":
    """(E.end, next.start) for the sentence at idx, or None if idx is the window's last sentence."""
    if idx < 0 or idx + 1 >= len(sents):
        return None
    return sents[idx].end, sents[idx + 1].start


def _snap_end_cut(e_end: float, next_start: float, tail_pad: float, energy_fn) -> "tuple[float, tuple]":
    """Cut into the gap AFTER e_end: min(e_end+tail_pad, midpoint(e_end, next_start)), then
    energy-snap within [e_end, that]. Never > next_start, never < e_end."""
    mid = (e_end + next_start) / 2.0
    pad_cut = min(e_end + tail_pad, mid)
    cut = pad_cut
    if energy_fn is not None:
        snapped = energy_fn(e_end, pad_cut)
        if snapped is not None:
            cut = min(max(snapped, e_end), pad_cut)
    return min(next_start, max(cut, e_end)), ()
```

Rewrite `_pick_end`:
```python
def _pick_end(sents, rough, pad, allow_qe, *, tail_pad=0.15, gap_min=0.12, end_extend_max=8.0,
              energy_fn=None) -> Pick:
    """Complete the thought, cutting into the trailing inter-word gap (never into the next word).
    HYBRID (handoff §8): tight cut at the chosen complete sentence when it has a usable trailing
    gap; only nudge forward to a later gap within end_extend_max; else keep tight + flag. Direction-
    safe: never truncates earlier than rough-1s. satisfied=False only when no valid end with a
    MEASURABLE gap is in the window (→ _refine_end grows)."""
    if not sents:
        return Pick(rough, (), False)
    valids = [i for i, s in enumerate(sents) if s.is_valid_end(allow_qe)]
    at_after = [i for i in valids if sents[i].end >= rough] or \
               [i for i in valids if sents[i].end >= rough - 1.0]
    if not at_after:
        return Pick(rough, (), False)                    # no valid end at all → grow
    at_after.sort(key=lambda i: sents[i].end)
    e_idx = at_after[0]
    if abs(sents[e_idx].end - rough) > pad and sents[e_idx].end > rough + pad:
        return Pick(rough, (), False)                    # nearest end beyond the window → grow
    gap = _gap_after(sents, e_idx)
    if gap is None:
        return Pick(round(sents[e_idx].end + tail_pad, 3), (), False)   # last in window → grow
    e_end, nxt = gap
    if (nxt - e_end) >= gap_min:
        cut, flags = _snap_end_cut(e_end, nxt, tail_pad, energy_fn)
        return Pick(round(cut, 3), flags, True)          # tight cut in a real gap
    # hybrid: look forward within budget for a later end WITH a usable gap
    for i in at_after:
        if sents[i].end > rough + end_extend_max:
            break
        g = _gap_after(sents, i)
        if g and (g[1] - g[0]) >= gap_min:
            cut, _ = _snap_end_cut(g[0], g[1], tail_pad, energy_fn)
            return Pick(round(cut, 3), ("end_extended",), True)
    # none in budget → best-available tight cut at E, flagged
    cut, _ = _snap_end_cut(e_end, nxt, tail_pad, energy_fn)
    return Pick(round(cut, 3), ("tight_end_no_gap",), True)
```

- [ ] **Step 4: Update `test_boundary_safety.py` end assertions to the new behavior**

The direction-safe floor test: only end `31.0`, far before `rough-1` (rough=45) → no `at_after` (with the `>= rough-1` fallback also empty since 31 < 44) → `satisfied=False`, `time == rough`:
```python
def test_pick_end_never_moves_earlier_than_window_floor():
    sents = [_sent(0, 28.0, 31.0)]
    p = _pick_end(sents, rough=45.0, pad=10.0, allow_qe=False,
                  tail_pad=0.15, gap_min=0.12, end_extend_max=8.0)
    assert p.time == 45.0 and not p.satisfied
```
The normal-path test: `E = sents[1]` (end 46.2) is the **last** sentence → gap unmeasurable → grow signal. Add a trailing sentence so the gap is measurable and assert the snapped end:
```python
def test_pick_end_normal_path_unchanged():
    # E @46.2, next @46.7 → 0.5 s gap → tight cut at 46.2 + tail(0.15) = 46.35
    sents = [_sent(0, 40.0, 44.5), _sent(1, 44.6, 46.2), _sent(2, 46.7, 48.0)]
    p = _pick_end(sents, rough=45.0, pad=10.0, allow_qe=False,
                  tail_pad=0.15, gap_min=0.12, end_extend_max=8.0)
    assert abs(p.time - 46.35) < 1e-6 and p.satisfied
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest backend/pipeline/tests/test_silence_snap.py backend/pipeline/tests/test_boundary_safety.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/pipeline/boundary.py backend/pipeline/tests/test_silence_snap.py backend/pipeline/tests/test_boundary_safety.py
git commit -m "$(cat <<'EOF'
feat(boundary): silence-aware END placement + hybrid gap policy (handoff §8)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Window extension + energy wiring (`_refine_start`, `_refine_end`, `_refine_one`)

**Files:**
- Modify: `backend/pipeline/boundary.py` (`_refine_start`, `_refine_end`, rewrite `_refine_one`)
- Modify (tests): `backend/pipeline/tests/test_refine_parallel_order.py` (expected values → snapped)
- Test (new): `backend/pipeline/tests/test_boundary_extension.py`

**Interfaces:**
- Consumes: `_whisper_window` (returns `(sents, wav)`), `_pick_start`/`_pick_end` (Pick), `_energy_min_snap`.
- Produces: `_refine_start(audio, s0, pad, *, lead_pad, gap_min, max_search) -> Pick`, `_refine_end(audio, e0, pad, allow_qe, *, tail_pad, gap_min, end_extend_max, max_search, max_clip_end) -> Pick`, and the extension-aware `_refine_one(c, audio, pad, allow_qe, tail_pad, lead_pad, gap_min, end_extend_max, max_search, max_clip_dur) -> dict`.

**Behavior contract:**
- `_refine_end`: transcribe `[e0-pad, min(e0+grow, e0+max_search, max_clip_end)]`; build `energy_fn` from the wav (`win_start = max(0, e0-pad)`); call `_pick_end`. If `satisfied` → return. Else grow (`grow *= 2`) until the forward reach hits `max_search` or `max_clip_end`; then return the last pick with `boundary_search_exhausted` added and `satisfied=False`. **Never advances the clip end past `max_clip_end` (= `s0 + max_clip_dur`).**
- `_refine_start`: transcribe `[max(0, s0-grow), s0+pad]`; `keep_first = (s0-grow <= 0)`; call `_pick_start`. If `satisfied` → return. Else grow backward until `win_start` hits 0 or `s0-win_start ≥ max_search`; then return last pick + `start_prev_unseen`, `satisfied=False`. **Never re-selects an earlier sentence** — growth only reveals the previous word.
- `_refine_one`: for short clips (`e0-s0 <= 2*pad+20`) try ONE combined window `[s0-pad, e0+pad]` and run both picks with a shared `energy_fn`; if either pick is unsatisfied, fall back to `_refine_start`/`_refine_end` for that side. Merge flags onto `warnings`. Clean up every wav.

- [ ] **Step 1: Write the failing test**

Create `backend/pipeline/tests/test_boundary_extension.py`:
```python
"""Window extension + refine orchestration (Task 7). Fully offline: _whisper_window is a stub
returning staged sentence sets keyed by the requested window; no audio/whisper/ffmpeg."""
from __future__ import annotations

import backend.config as config
import backend.pipeline.boundary as bmod
from backend.pipeline.sentences import Sentence


def _s(i, a, b, term="."):
    return Sentence(idx=i, text=f"s{i}", start=a, end=b, terminator=term,
                    ends_with_period=(term in ".?!"), word_start_idx=i, word_end_idx=i,
                    align_confidence=1.0)


def _stub_window(fn):
    """fn(win_start, win_end) -> list[Sentence]; adapts to the (sents, wav=None) contract."""
    def _w(audio, win_start, win_end):
        return fn(win_start, win_end), None
    return _w


# ── END: no period in the initial window → grow → period found in the extension ───────────
def test_end_grows_until_period_found(monkeypatch):
    calls = []

    def fn(ws, we):
        calls.append((round(ws, 2), round(we, 2)))
        if we - ws <= 20.5:                       # initial window [e0-pad, e0+pad] = 20 s: no period
            return [_s(0, 95.0, 99.0, term=""), _s(1, 99.1, 104.0, term="")]
        # grown window: a period-terminated end at 108 with a trailing gap at 108.6
        return [_s(0, 99.0, 104.0, term=""), _s(1, 104.0, 108.0, "."), _s(2, 108.6, 112.0, ".")]

    monkeypatch.setattr(bmod, "_whisper_window", _stub_window(fn))
    p = bmod._refine_end(audio=None, e0=100.0, pad=10.0, allow_qe=False, tail_pad=0.15,
                         gap_min=0.12, end_extend_max=30.0, max_search=45.0, max_clip_end=1e9)
    assert p.satisfied
    assert 108.0 < p.time < 108.6                 # cut in the gap after the found period
    assert len(calls) >= 2                        # it actually grew


# ── END: exhaustion → flagged fallback, still returns a usable time (clip ships) ──────────
def test_end_exhaustion_flags_and_ships(monkeypatch):
    def fn(ws, we):
        return [_s(0, 95.0, 99.0, term=""), _s(1, 99.1, 140.0, term="")]   # never a period

    monkeypatch.setattr(bmod, "_whisper_window", _stub_window(fn))
    p = bmod._refine_end(audio=None, e0=100.0, pad=10.0, allow_qe=False, tail_pad=0.15,
                         gap_min=0.12, end_extend_max=30.0, max_search=45.0, max_clip_end=1e9)
    assert not p.satisfied
    assert "boundary_search_exhausted" in p.flags
    assert p.time >= 99.0                          # a real, usable time — clip is not dropped


# ── END: never advances past max_clip_end even when a later clean gap exists ──────────────
def test_end_respects_max_clip_end(monkeypatch):
    def fn(ws, we):
        # a clean period+gap at 150 exists, but is only VISIBLE once the window reaches it;
        # max_clip_end caps the window at 120, so it can never be transcribed.
        sents = [_s(0, 99.0, 104.0, term="")]
        if we >= 150.6:
            sents += [_s(1, 104.0, 150.0, "."), _s(2, 150.6, 153.0, ".")]
        else:
            sents += [_s(1, 104.0, we, term="")]     # run-on to the (capped) window edge
        return sents

    monkeypatch.setattr(bmod, "_whisper_window", _stub_window(fn))
    p = bmod._refine_end(audio=None, e0=100.0, pad=10.0, allow_qe=False, tail_pad=0.15,
                         gap_min=0.12, end_extend_max=30.0, max_search=45.0, max_clip_end=120.0)
    # window never reaches 150 → the 150 end is never seen → exhausted fallback, not an advance
    assert p.time < 150.0
    assert not p.satisfied


# ── END: a far period found ONLY after growth is ACCEPTED, not rejected for being > pad away ─
# (guards against re-introducing the §4d bug: a period-terminated end far from the coarse rough
#  must be snapped to, never discarded for the coarse mid-sentence fallback.)
def test_end_accepts_far_period_found_via_growth(monkeypatch):
    def fn(ws, we):
        if we >= 118.6:                              # period+gap at 118 (rough+18) visible now
            return [_s(0, 99.0, 110.0, term=""), _s(1, 110.0, 118.0, "."), _s(2, 118.6, 121.0, ".")]
        return [_s(0, 95.0, 99.0, term=""), _s(1, 99.1, we, term="")]    # run-on to window edge

    monkeypatch.setattr(bmod, "_whisper_window", _stub_window(fn))
    p = bmod._refine_end(audio=None, e0=100.0, pad=10.0, allow_qe=False, tail_pad=0.15,
                         gap_min=0.12, end_extend_max=8.0, max_search=45.0, max_clip_end=1e9)
    assert p.satisfied                               # far period accepted, NOT exhausted
    assert 118.0 < p.time < 118.6                    # cut in the gap after the found period
    assert "boundary_search_exhausted" not in p.flags


# ── START: backward growth reveals the previous word so the leading gap can be measured ───
def test_start_grows_backward_to_see_prev(monkeypatch):
    def fn(ws, we):
        if ws >= 89.5:                             # initial [s0-pad, s0+pad]: S is window-first
            return [_s(0, 100.0, 103.0, ".")]      # only the chosen sentence → prev unseen
        # grown backward: previous sentence now visible ending at 99.4 → 0.6 s leading gap
        return [_s(0, 96.0, 99.4, "."), _s(1, 100.0, 103.0, ".")]

    monkeypatch.setattr(bmod, "_whisper_window", _stub_window(fn))
    p = bmod._refine_start(audio=None, s0=100.0, pad=10.0, lead_pad=0.06, gap_min=0.12,
                           max_search=45.0)
    assert p.satisfied
    assert 99.4 < p.time < 100.0                   # cut in the leading gap (≈ 99.94)


# ── _refine_one: short clip uses ONE combined window when both picks are satisfied ────────
def test_refine_one_combined_window_fast_path(monkeypatch):
    windows = []

    def _w(audio, ws, we):
        windows.append((round(ws, 2), round(we, 2)))
        # one window [s0-pad, e0+pad] contains a clean start and end with measurable gaps
        return [_s(0, 88.0, 89.4, "."), _s(1, 90.0, 92.0, "."), _s(2, 100.0, 104.0, "."),
                _s(3, 104.6, 108.0, ".")], None

    monkeypatch.setattr(bmod, "_whisper_window", _w)
    c = {"start": 90.0, "end": 104.0, "cut_end": 104.05, "facet": "x"}
    out = bmod._refine_one(c, audio=None, pad=10.0, allow_qe=False, tail_pad=0.15, lead_pad=0.06,
                           gap_min=0.12, end_extend_max=8.0, max_search=45.0, max_clip_dur=180.0)
    assert len(windows) == 1                        # single combined transcription (fast path)
    assert 89.4 < out["start"] < 90.0               # start snapped into its gap
    assert 104.0 < out["end"] < 104.6               # end snapped into its gap
    assert out["cut_end"] == round(out["end"] + 0.15, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest backend/pipeline/tests/test_boundary_extension.py -v`
Expected: FAIL (`_refine_start`/`_refine_end` not defined; `_refine_one` signature differs).

- [ ] **Step 3: Write minimal implementation**

**First, correct `_pick_end` for the grown-window case.** From Task 6, `_pick_end` still carries a
forward drift cap that is dead in a single window but becomes HARMFUL once `_refine_end` grows the
window. Make exactly these two edits to `_pick_end`:

1. **Delete** this rejection entirely — it would discard a legitimately-found far period and grow
   to exhaustion, landing on the coarse mid-sentence fallback (the exact §4d bug this feature fixes):
   ```python
   if abs(sents[e_idx].end - rough) > pad and sents[e_idx].end > rough + pad:
       return Pick(rough, (), False)
   ```
   The earliest valid period-terminated end at/after `rough` is now always accepted; clip-end
   distance is bounded by `max_clip_end` and `MAX_BOUNDARY_SEARCH_S` inside `_refine_end`.
2. **Re-anchor the hybrid nudge budget to the chosen end `E`** (§8: advance ~one sentence beyond the
   complete thought, not beyond the coarse rough):
   ```python
   for i in at_after:
       if sents[i].end > sents[e_idx].end + end_extend_max:   # was: rough + end_extend_max
           break
   ```

Both edits keep every Task 6 test in `test_silence_snap.py` green (there `E.end ≈ rough`) and are
covered by the new `test_end_accepts_far_period_found_via_growth`. Do NOT change any other part of
`_pick_end` (the direction-safe `at_after`, the last-sentence grow, the tight/`end_extended`/
`tight_end_no_gap` returns all stay).

Then add `_refine_start`/`_refine_end` and rewrite `_refine_one` in `backend/pipeline/boundary.py`:
```python
def _refine_end(audio, e0, pad, allow_qe, *, tail_pad, gap_min, end_extend_max,
                max_search, max_clip_end) -> Pick:
    grow = pad
    last = Pick(e0, (), False)
    while True:
        win_start = max(0.0, e0 - pad)
        win_end = min(e0 + grow, e0 + max_search, max_clip_end)
        sents, wav = _whisper_window(audio, win_start, win_end)
        try:
            energy_fn = (lambda a, b: _energy_min_snap(wav, win_start, a, b)) if wav else None
            last = _pick_end(sents, e0, pad, allow_qe, tail_pad=tail_pad, gap_min=gap_min,
                             end_extend_max=end_extend_max, energy_fn=energy_fn)
        finally:
            if wav is not None:
                wav.unlink(missing_ok=True)
        if last.satisfied:
            return last
        if win_end >= e0 + max_search - 1e-6 or win_end >= max_clip_end - 1e-6:
            break
        grow *= 2
    return Pick(last.time, tuple(sorted(set(last.flags) | {"boundary_search_exhausted"})), False)


def _refine_start(audio, s0, pad, *, lead_pad, gap_min, max_search) -> Pick:
    grow = pad
    last = Pick(s0, (), False)
    while True:
        win_start = max(0.0, s0 - grow)
        win_end = s0 + pad
        sents, wav = _whisper_window(audio, win_start, win_end)
        try:
            energy_fn = (lambda a, b: _energy_min_snap(wav, win_start, a, b)) if wav else None
            last = _pick_start(sents, s0, pad, keep_first=(win_start <= 0.0),
                               lead_pad=lead_pad, gap_min=gap_min, energy_fn=energy_fn)
        finally:
            if wav is not None:
                wav.unlink(missing_ok=True)
        if last.satisfied:
            return last
        if win_start <= 0.0 or (s0 - win_start) >= max_search - 1e-6:
            break
        grow *= 2
    return Pick(last.time, tuple(sorted(set(last.flags) | {"start_prev_unseen"})), False)


def _refine_one(c, audio, pad, allow_qe, tail_pad, lead_pad, gap_min, end_extend_max,
                max_search, max_clip_dur) -> dict:
    s0, e0 = float(c["start"]), float(c["end"])
    flags: tuple = ()
    try:
        max_clip_end = s0 + max_clip_dur
        if e0 - s0 <= 2 * pad + 20:                       # short clip → try one combined window
            win_start = max(0.0, s0 - pad)
            sents, wav = _whisper_window(audio, s0 - pad, e0 + pad)
            try:
                energy_fn = (lambda a, b: _energy_min_snap(wav, win_start, a, b)) if wav else None
                sp = _pick_start(sents, s0, pad, keep_first=(win_start <= 0.0),
                                 lead_pad=lead_pad, gap_min=gap_min, energy_fn=energy_fn)
                ep = _pick_end(sents, e0, pad, allow_qe, tail_pad=tail_pad, gap_min=gap_min,
                               end_extend_max=end_extend_max, energy_fn=energy_fn)
            finally:
                if wav is not None:
                    wav.unlink(missing_ok=True)
            if not sp.satisfied:
                sp = _refine_start(audio, s0, pad, lead_pad=lead_pad, gap_min=gap_min,
                                   max_search=max_search)
            if not ep.satisfied:
                ep = _refine_end(audio, e0, pad, allow_qe, tail_pad=tail_pad, gap_min=gap_min,
                                 end_extend_max=end_extend_max, max_search=max_search,
                                 max_clip_end=max_clip_end)
        else:
            sp = _refine_start(audio, s0, pad, lead_pad=lead_pad, gap_min=gap_min,
                               max_search=max_search)
            ep = _refine_end(audio, e0, pad, allow_qe, tail_pad=tail_pad, gap_min=gap_min,
                             end_extend_max=end_extend_max, max_search=max_search,
                             max_clip_end=max_clip_end)
        new_start, new_end = sp.time, ep.time
        flags = tuple(sorted(set(sp.flags) | set(ep.flags)))
        if new_end <= new_start:
            new_start, new_end, flags = s0, e0, tuple(sorted(set(flags) | {"refine_degenerate"}))
    except Exception:
        new_start, new_end, flags = s0, e0, ()
    d = dict(c)
    d["start"] = round(new_start, 3)
    d["end"] = round(new_end, 3)
    d["cut_end"] = round(new_end + tail_pad, 3)
    if flags:
        d["warnings"] = tuple(sorted(set(d.get("warnings") or ()) | set(flags)))
    return d
```

- [ ] **Step 4: Update `refine_clip_boundaries`'s `_refine_one` call (temporary shim) to keep the suite green**

`refine_clip_boundaries` still calls `_refine_one(c, audio, pad, allow_qe, tail_pad)` (Task 4 signature). Update the two `pool.submit(_refine_one, ...)` / serial call sites to pass the new args from config defaults (Task 8 threads them from settings):
```python
        fut_to_idx = {pool.submit(_refine_one, c, audio, pad, allow_qe, tail_pad,
                                  config.DEFAULTS["lead_pad_s"], config.SILENCE_MIN_GAP_S,
                                  config.END_EXTEND_MAX_S, config.MAX_BOUNDARY_SEARCH_S,
                                  config.DEFAULTS["max_clip_duration_s"]): i
                      for i, c in enumerate(clips)}
```

- [ ] **Step 5: Update `test_refine_parallel_order.py` expected offsets (order-invariance preserved)**

The stub windows now produce silence-snapped boundaries, so the exact offsets shift. Keep the **order-invariance** and **tie-break** assertions; recompute the numeric expectations. In `_make_windows`/`_make_reversed_windows`, the sentences are `[_sent(0, s0-20, s0-19, ""), _sent(1, tgt_start, tgt_end, ".")]`:
- Start: `S=sents[1]@tgt_start`, `prev=sents[0]` end `s0-19` → huge gap → `cut = tgt_start - lead_pad(0.06)`.
- End: `E=sents[1]@tgt_end` is the window's **last** sentence → gap unmeasurable → `_pick_end` returns `satisfied=False` → `_refine_end` grows. The stub ignores the widened window (returns the same 2 sentences) → exhaustion → `time = tgt_end + tail? ` (the last pick's `E.end + tail_pad`), flag `boundary_search_exhausted`.

To keep this test focused on **order-invariance** (not on exact snap values), simplify the stub so BOTH picks are satisfied in one window by adding a trailing sentence after the end. Replace the sentence list in both `_win` helpers with three sentences that give measurable gaps:
```python
        sents = [_sent(0, s0 - 20.0, s0 - 19.0, ""),          # leading fragment (prev for S)
                 _sent(1, tgt_start, tgt_end, "."),           # the chosen sentence
                 _sent(2, tgt_end + 0.5, tgt_end + 3.0, ".")]  # trailing → measurable end gap
        return sents, None
```
Then the deterministic boundaries are `start = tgt_start - 0.06`, `end = tgt_end + 0.15` (tight, 0.5 s gap ≥ gap_min). Update the assertions:
```python
    assert [round(c["start"], 3) for c in serial] == [98.94, 197.94, 296.94, 395.94]  # tgt_start-lead
```
and any `end`/`cut_end` checks accordingly (`end = tgt_end + 0.15`; for clip i, `tgt_end = s0+25+(i+1)`). Keep `_key(serial) == _key(parallel)` and the `["c0","c1","c2","c3"]` / `["A"]` survivor assertions **unchanged** — those are the actual contract. For the start-tie test, keep both clips' targets so A still wins the tie after snapping (A start `100.0-0.06=99.94`, B start `100.0-0.06=99.94` collide → A kept, B trimmed & dropped) — the tie survivor is unchanged.

- [ ] **Step 6: Run tests to verify they pass**

Run:
```
.venv/bin/python -m pytest backend/pipeline/tests/test_boundary_extension.py backend/pipeline/tests/test_refine_parallel_order.py backend/pipeline/tests/test_boundary_safety.py backend/pipeline/tests/test_silence_snap.py -v
```
Expected: PASS (all).

- [ ] **Step 7: Commit**

```bash
git add backend/pipeline/boundary.py backend/pipeline/tests/test_boundary_extension.py backend/pipeline/tests/test_refine_parallel_order.py
git commit -m "$(cat <<'EOF'
feat(boundary): window-extending refine with energy snap + hybrid end policy

Grows/re-transcribes the window until a period-terminated end with a usable gap is found
(bounded by MAX_BOUNDARY_SEARCH_S and max_clip_duration_s); exhaustion ships a flagged
best-available cut. START growth only reveals the previous word — never re-selects a sentence.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Wire config/settings through `refine_clip_boundaries` + pre-warm; full-suite green

**Files:**
- Modify: `backend/pipeline/boundary.py` (`refine_clip_boundaries`)
- Test (new): `backend/pipeline/tests/test_refine_clip_boundaries_wiring.py`

**Interfaces:**
- Consumes: `_refine_one` (Task 7 signature), `transcribe._get_refine_whisper`, `config.*`.
- Produces: `refine_clip_boundaries` reads `lead_pad_s` (settings→DEFAULTS), `tail_pad_s`, `SILENCE_MIN_GAP_S`, `END_EXTEND_MAX_S`, `MAX_BOUNDARY_SEARCH_S`, and `max_clip_duration_s` (settings→DEFAULTS), pre-warms the refine singleton, and threads them into every `_refine_one`. The audio-failure short-circuit and the parallel order-invariance are unchanged.

- [ ] **Step 1: Write the failing test**

Create `backend/pipeline/tests/test_refine_clip_boundaries_wiring.py`:
```python
"""refine_clip_boundaries wiring (Task 8). Offline: audio + _refine_one stubbed."""
from __future__ import annotations

import backend.config as config
import backend.pipeline.boundary as bmod
from backend.pipeline.boundary import refine_clip_boundaries


def test_audio_failure_returns_input_unchanged(monkeypatch):
    def boom(url, video_id):
        raise RuntimeError("yt-dlp throttled")
    monkeypatch.setattr(bmod, "_ensure_audio", boom)
    clips = [{"start": 10.0, "end": 40.0, "cut_end": 40.05, "facet": "other"}]
    assert refine_clip_boundaries(clips, "https://youtu.be/x", "vid", {}) is clips


def test_settings_lead_pad_threaded_into_refine_one(monkeypatch):
    seen = {}

    def fake_refine_one(c, audio, pad, allow_qe, tail_pad, lead_pad, gap_min,
                        end_extend_max, max_search, max_clip_dur):
        seen.update(lead_pad=lead_pad, tail_pad=tail_pad, gap_min=gap_min,
                    end_extend_max=end_extend_max, max_search=max_search, max_clip_dur=max_clip_dur)
        return dict(c)

    monkeypatch.setattr(bmod, "_ensure_audio", lambda url, vid: "/x/audio.m4a")
    monkeypatch.setattr(bmod, "_refine_one", fake_refine_one)
    monkeypatch.setattr(bmod, "_get_refine_whisper", lambda: object())   # pre-warm no-op
    monkeypatch.setattr(config, "REFINE_WORKERS", 1)
    clips = [{"start": 10.0, "end": 40.0, "cut_end": 40.05, "facet": "other"}]
    refine_clip_boundaries(clips, "https://youtu.be/x", "vid",
                           {"lead_pad_s": 0.09, "tail_pad_s": 0.2, "max_clip_duration_s": 90.0})
    assert seen["lead_pad"] == 0.09
    assert seen["tail_pad"] == 0.2
    assert seen["max_clip_dur"] == 90.0
    assert seen["gap_min"] == config.SILENCE_MIN_GAP_S
    assert seen["end_extend_max"] == config.END_EXTEND_MAX_S
    assert seen["max_search"] == config.MAX_BOUNDARY_SEARCH_S


def test_prewarms_refine_singleton_when_pooled(monkeypatch):
    warmed = {"n": 0}
    monkeypatch.setattr(bmod, "_ensure_audio", lambda url, vid: "/x/audio.m4a")
    monkeypatch.setattr(bmod, "_refine_one", lambda *a, **k: dict(a[0]))
    monkeypatch.setattr(bmod, "_get_refine_whisper", lambda: warmed.__setitem__("n", warmed["n"] + 1))
    monkeypatch.setattr(config, "REFINE_WORKERS", 4)
    clips = [{"start": i * 100.0, "end": i * 100.0 + 30.0, "facet": "f"} for i in range(3)]
    refine_clip_boundaries(clips, "https://youtu.be/x", "vid", {})
    assert warmed["n"] >= 1                        # pre-warmed the refine model before the pool
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest backend/pipeline/tests/test_refine_clip_boundaries_wiring.py -v`
Expected: FAIL (`_refine_one` called with the old 5-arg signature; refine singleton not pre-warmed).

- [ ] **Step 3: Write minimal implementation**

Rewrite the body of `refine_clip_boundaries` to read the new tunables and thread them, and pre-warm `_get_refine_whisper` (replacing the `_get_whisper()` pre-warm):
```python
def refine_clip_boundaries(clips, url, video_id, settings, progress=None):
    allow_qe = bool(settings.get("allow_question_exclaim_ends", False))
    tail_pad = float(settings.get("tail_pad_s", config.DEFAULTS["tail_pad_s"]))
    lead_pad = float(settings.get("lead_pad_s", config.DEFAULTS["lead_pad_s"]))
    min_dur = float(settings.get("min_clip_duration_s", config.DEFAULTS["min_clip_duration_s"]))
    max_dur = float(settings.get("max_clip_duration_s", config.DEFAULTS["max_clip_duration_s"]))
    gap_min = config.SILENCE_MIN_GAP_S
    end_extend_max = config.END_EXTEND_MAX_S
    max_search = config.MAX_BOUNDARY_SEARCH_S
    pad = config.BOUNDARY_PAD_S
    try:
        audio = _ensure_audio(url, video_id)
    except Exception:
        return clips
    n = len(clips)
    total = max(1, n)
    results = [None] * n
    workers = max(1, min(config.REFINE_WORKERS, n))
    if workers > 1:
        try:
            _get_refine_whisper()                         # pre-warm the shared refine singleton
        except Exception:
            pass
    args = (pad, allow_qe, tail_pad, lead_pad, gap_min, end_extend_max, max_search, max_dur)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        fut_to_idx = {pool.submit(_refine_one, c, audio, *args): i for i, c in enumerate(clips)}
        for done, fut in enumerate(as_completed(fut_to_idx), start=1):
            results[fut_to_idx[fut]] = fut.result()
            if progress:
                progress(done / total, f"Refining boundary {done}/{n}")
    out = [d for d in results if d is not None]
    return _resolve_overlaps(out, min_dur, tail_pad)
```
Keep the existing module-level docstring/comments on ordering; the `results`-by-index + serial `_resolve_overlaps` contract is unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest backend/pipeline/tests/test_refine_clip_boundaries_wiring.py backend/pipeline/tests/test_refine_parallel_order.py -v`
Expected: PASS.

- [ ] **Step 5: Full suite green (regression gate)**

Run: `.venv/bin/python -m pytest backend/ -q`
Expected: PASS — at least the 701 baseline plus the new tests; **zero** failures. Pay special attention that these are green:
```
.venv/bin/python -m pytest backend/pipeline/tests/test_bnd1_boundary_guards.py backend/pipeline/tests/test_onset_start_guard.py backend/pipeline/tests/test_discourse_onset.py -v
```
If anything regressed, fix it before committing (use superpowers:systematic-debugging).

- [ ] **Step 6: Commit**

```bash
git add backend/pipeline/boundary.py backend/pipeline/tests/test_refine_clip_boundaries_wiring.py
git commit -m "$(cat <<'EOF'
feat(boundary): thread pads/search bounds through refine_clip_boundaries + pre-warm refine model

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Post-implementation (outside the task loop)

1. **Whole-branch review** (subagent-driven-development's final stage; superpowers:requesting-code-review): correctness of the extension/growth bounds, thread-safety of the new refine singleton under the pool, wav-cleanup on every path, and confirm `refine.py`/BND1/onset are untouched and green.
2. **Real-cut A/B (spec §6 — host env, needs local Whisper + ffmpeg):** cut a sample video old vs new; measure (a) fraction of cut points in silence (≈0%→≈100%), (b) RMS of the clip's final 100 ms, (c) first 100 ms. Acceptance: no clip's first/last 100 ms contains speech-level energy. Only if this shows truncation, revisit `cut.py` (spec §5) — otherwise leave it unchanged.
3. **finishing-a-development-branch**, then restart the host (`host.sh`, with the double-fork reparent if the harness backgrounds it) to go live. First run downloads the ~1.5 GB `medium` refine model.

## Self-Review Notes (author check against the spec)

- **Spec §4a** (richer refine window: refine model, VAD, `condition_on_previous_text=False`, `temperature=0`, `beam_size=5`, keep wav) → Tasks 2, 4. **§4b** (silence-aware picks, direction-safety) → Tasks 5, 6. **§4c** (energy-minimum snap) → Tasks 3, 7. **§4d** (adaptive window extension, asymmetry) → Task 7. **§4e** (config) → Task 1. **§6** (unit tests: gap-landing, asymmetric pad, prev/next absent, gap<min → extension, exhaustion flagged, energy snap) → Tasks 3, 5, 6, 7. **§8 hybrid** (user-confirmed) → Task 6 (`_pick_end`) + Task 7 (budget bound).
- **Type consistency:** `Pick(time, flags, satisfied)` is introduced in Task 4 and used unchanged through Task 8. `_whisper_window` returns `(sents, wav)` from Task 4 on; all stubs updated to match. `_refine_one`'s signature is set in Task 7 and its callers (Task 7 shim, Task 8 final) match it.
- **No placeholders:** every step has concrete code/commands/expected output.
