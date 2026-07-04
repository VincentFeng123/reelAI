"""Window extension + refine orchestration (Task 7). Fully offline: _whisper_window is a stub
returning staged sentence sets keyed by the requested window; no audio/whisper/ffmpeg."""
from __future__ import annotations

import backend.config as config
import backend.pipeline.boundary as bmod
from backend.pipeline.sentences import Sentence


def _s(i, a, b, term="."):
    # Use tuple membership so term="" → ends_with_period=False (matches real Sentence behaviour).
    return Sentence(idx=i, text=f"s{i}", start=a, end=b, terminator=term,
                    ends_with_period=(term in (".", "?", "!")), word_start_idx=i, word_end_idx=i,
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
            # sents[0] is an unterminated fragment → prev_unseen → grow
            return [_s(0, 90.0, 99.0, term=""), _s(1, 100.0, 103.0, ".")]
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
