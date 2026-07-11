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
