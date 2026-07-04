"""Order-invariance of the parallel boundary REFINE (latency lever).

The per-clip edge-window Whisper passes now run over a ThreadPoolExecutor, but the output
(each clip's start/end/cut_end, and the post-`_resolve_overlaps` order) MUST be byte-identical
to the serial baseline. These tests force OUT-OF-ORDER (reversed) task completion and assert the
result equals the in-order (REFINE_WORKERS=1) result.

Fully offline: `_ensure_audio` is mocked away and `_whisper_window` returns deterministic,
per-clip sentences — the real Whisper model is never constructed.
"""
from __future__ import annotations

import threading
import time

import backend.config as config_mod
import backend.pipeline.boundary as boundary_mod
from backend.pipeline.boundary import refine_clip_boundaries
from backend.pipeline.sentences import Sentence

PAD = config_mod.BOUNDARY_PAD_S   # 10.0 — window half-width used by refine


def _sent(idx: int, start: float, end: float, terminator: str) -> Sentence:
    return Sentence(idx=idx, text=f"s{idx}", start=start, end=end, terminator=terminator,
                    ends_with_period=(terminator in ".?!"), word_start_idx=idx, word_end_idx=idx,
                    align_confidence=1.0)


def _spec_from_clips(clips):
    """spec[round(s0,3)] = (idx, target_start, target_end) — the deterministic per-clip snap."""
    spec = {}
    for idx, c in enumerate(clips):
        spec[round(float(c["start"]), 3)] = (idx, c["_target_start"], c["_target_end"])
    return spec


def _make_windows(spec):
    """A plain (no coordination) `_whisper_window`: recover the clip from the window start and
    return sentences engineered so `_pick_start`→target_start and `_pick_end`→target_end.
    Three sentences: sents[0] is a prev-word with terminator "." (gap before sents[1] is
    measurable); sents[2] is a trailing sentence giving a 0.5 s gap after sents[1] so
    _pick_end is satisfied in one combined window."""
    def _win(audio, win_start, win_end):
        s0 = round(win_start + PAD, 3)                    # short-clip window is (s0-PAD, e0+PAD)
        _idx, tgt_start, tgt_end = spec[s0]
        sents = [_sent(0, s0 - 20.0, s0 - 19.0, "."),     # prev word (terminated) → start gap measurable
                 _sent(1, tgt_start, tgt_end, "."),        # the chosen sentence
                 _sent(2, tgt_end + 0.5, tgt_end + 3.0, ".")]  # trailing → measurable end gap
        return sents, None
    return _win


def _make_reversed_windows(spec, n):
    """Same deterministic sentences, but forces STRICTLY REVERSED completion: clip idx does not
    return until clips idx+1..n-1 already returned. Requires n workers so all run concurrently."""
    events = [threading.Event() for _ in range(n)]

    def _win(audio, win_start, win_end):
        s0 = round(win_start + PAD, 3)
        idx, tgt_start, tgt_end = spec[s0]
        sents = [_sent(0, s0 - 20.0, s0 - 19.0, "."),     # prev word (terminated) → start gap measurable
                 _sent(1, tgt_start, tgt_end, "."),
                 _sent(2, tgt_end + 0.5, tgt_end + 3.0, ".")]  # trailing → measurable end gap
        if idx + 1 < n:                       # wait for the next-higher clip to finish first
            events[idx + 1].wait(5.0)
            time.sleep(0.02)                  # let that clip's _refine_one fully return
        events[idx].set()
        return sents, None
    return _win


def _run(monkeypatch, clips, window_fn, workers):
    monkeypatch.setattr(boundary_mod, "_ensure_audio", lambda url, vid: "/nonexistent/audio.m4a")
    monkeypatch.setattr(boundary_mod, "_whisper_window", window_fn)
    monkeypatch.setattr(config_mod, "REFINE_WORKERS", workers)
    spec_clips = [{k: v for k, v in c.items() if not k.startswith("_")} for c in clips]
    return refine_clip_boundaries(spec_clips, "https://youtu.be/x", "vid", {})


def _key(clips):
    return [(c.get("id"), c["start"], c["end"], c["cut_end"],
             tuple(sorted(c.get("warnings") or ()))) for c in clips]


# ── distinct clips: reversed completion must not permute or cross-contaminate results ─────────
def test_parallel_reversed_completion_matches_serial_distinct(monkeypatch):
    # 4 well-separated clips; each snaps to a UNIQUE offset so a mis-routed result would show up.
    clips = []
    for i in range(4):
        s0 = 100.0 + 100.0 * i
        e0 = s0 + 25.0
        clips.append({"id": f"c{i}", "start": s0, "end": e0, "cut_end": e0 + 0.05,
                      "_target_start": s0 - (i + 1), "_target_end": e0 + (i + 1)})
    spec = _spec_from_clips(clips)

    serial = _run(monkeypatch, clips, _make_windows(spec), workers=1)
    parallel = _run(monkeypatch, clips, _make_reversed_windows(spec, 4), workers=4)

    # every clip snapped to its own offset, and order is preserved
    assert _key(serial) == _key(parallel)
    assert [c["id"] for c in serial] == ["c0", "c1", "c2", "c3"]
    # Task 5: _pick_start now cuts into the leading gap (tgt_start - lead_pad=0.06);
    # stubs use term="." for sents[0] → terminated-prev / measurable-gap path → cut = tgt_start - lead_pad.
    assert [round(c["start"], 3) for c in serial] == [98.94, 197.94, 296.94, 395.94]


# ── start-tie: THE guard. Two clips snap to the SAME start; _resolve_overlaps' stable sort
#   tie-break (and thus which clip survives) depends on the ORIGINAL index order feeding it.
#   If the pool appended results in COMPLETION order, the survivor would flip → different output. ─
def test_parallel_preserves_index_order_on_start_tie(monkeypatch):
    clips = [
        {"id": "A", "start": 100.0, "end": 128.0, "cut_end": 128.05,
         "_target_start": 100.0, "_target_end": 128.0},   # kept when processed first
        {"id": "B", "start": 103.0, "end": 138.0, "cut_end": 138.05,
         "_target_start": 100.0, "_target_end": 138.0},   # collides on start=100 → trimmed & dropped
    ]
    spec = _spec_from_clips(clips)

    serial = _run(monkeypatch, clips, _make_windows(spec), workers=1)
    parallel = _run(monkeypatch, clips, _make_reversed_windows(spec, 2), workers=2)

    # index order wins the tie: A survives (100→128), B is trimmed to 128 then dropped (<min_dur)
    assert [c["id"] for c in serial] == ["A"]
    assert _key(serial) == _key(parallel)   # reversed completion did NOT flip the survivor
