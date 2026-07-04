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
