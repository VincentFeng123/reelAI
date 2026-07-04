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
    assert _pick_end(sents, rough=45.0, pad=10.0, allow_qe=False,
                     tail_pad=0.15, gap_min=0.12, end_extend_max=8.0).time == 45.0


def test_pick_end_normal_path_unchanged():
    sents = [_sent(0, 40.0, 44.5), _sent(1, 44.6, 46.2)]
    assert _pick_end(sents, rough=45.0, pad=10.0, allow_qe=False,
                     tail_pad=0.15, gap_min=0.12, end_extend_max=8.0).time == 46.2


def test_pick_start_never_moves_later_than_window_ceiling():
    # only candidate start is at 58.0, far AFTER rough+1.0 (rough=45): old code returned 58.0
    sents = [_sent(0, 43.0, 44.0), _sent(1, 58.0, 60.0)]  # sents[0] dropped as fragment
    assert _pick_start(sents, rough=45.0, pad=10.0, lead_pad=0.06, gap_min=0.12).time == 45.0


def test_pick_start_normal_path_unchanged():
    # S = sents[1] @44.2, prev = sents[0] end 36.0 → big gap → cut = 44.2 - lead(0.06) = 44.14
    sents = [_sent(0, 35.0, 36.0), _sent(1, 44.2, 47.0), _sent(2, 47.1, 49.0)]
    assert abs(_pick_start(sents, rough=45.0, pad=10.0, lead_pad=0.06, gap_min=0.12).time - 44.14) < 1e-6


def test_pick_start_keep_first_at_video_start():
    # window clamped at t=0 → sents[0] is a REAL sentence start, not a fragment
    sents = [_sent(0, 0.0, 3.0), _sent(1, 3.1, 6.0)]
    assert _pick_start(sents, rough=0.5, pad=10.0, keep_first=True,
                       lead_pad=0.06, gap_min=0.12).time == 0.0
    assert _pick_start(sents, rough=0.5, pad=10.0, lead_pad=0.06, gap_min=0.12).time == 0.5
