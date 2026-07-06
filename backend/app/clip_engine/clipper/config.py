# backend/app/clip_engine/clipper/config.py
"""Vendored clipper config shim → delegates to app.clip_engine.config.
Adds the extra constants the vendored gemini path reads that aren't in the
engine config, with practice defaults. NO .env file load, NO paths/mkdir.
"""
from __future__ import annotations

from .. import config as _engine

SEGMENT_MODEL = _engine.SEGMENT_MODEL
SEGMENT_FINE_SNAP = _engine.SEGMENT_FINE_SNAP
SEGMENT_MIN_CLIP_S = _engine.SEGMENT_MIN_CLIP_S
SEGMENT_MAX_CLIPS = _engine.SEGMENT_MAX_CLIPS
SEGMENT_MAX_OUTPUT_TOKENS = _engine.SEGMENT_MAX_OUTPUT_TOKENS
GEMINI_MODEL = _engine.GEMINI_MODEL
GEMINI_API_KEY = _engine.GEMINI_API_KEY
LLM_PROVIDER = "gemini"
CHARS_PER_TOKEN = 4
DEFAULTS = {"tail_pad_s": _engine.TAIL_PAD_S, "lead_pad_s": 0.06}
# Supadata transcript
SUPADATA_API_KEY = _engine.SUPADATA_API_KEY
SUPADATA_BASE = _engine.SUPADATA_BASE
SUPADATA_CHUNK_SIZE = 180
