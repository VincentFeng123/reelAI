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
