# backend/app/clip_engine/clipper/config.py
"""Vendored clipper config shim → delegates to app.clip_engine.config.
Adds the extra constants the vendored gemini path reads that aren't in the
engine config, with practice defaults. NO .env file load, NO paths/mkdir.
"""
from __future__ import annotations

import os
from pathlib import Path

from .. import config as _engine

# ── Delegated to engine config ────────────────────────────────────────────────
SEGMENT_MODEL = _engine.SEGMENT_MODEL
SEGMENT_FINE_SNAP = _engine.SEGMENT_FINE_SNAP
SEGMENT_MIN_CLIP_S = _engine.SEGMENT_MIN_CLIP_S
SEGMENT_MAX_CLIP_S = _engine.SEGMENT_MAX_CLIP_S
SEGMENT_INFORMATIVENESS_MIN = _engine.SEGMENT_INFORMATIVENESS_MIN
SEGMENT_MAX_CLIPS = _engine.SEGMENT_MAX_CLIPS
SEGMENT_MAX_OUTPUT_TOKENS = _engine.SEGMENT_MAX_OUTPUT_TOKENS
GEMINI_MODEL = _engine.GEMINI_MODEL
GEMINI_API_KEY = _engine.GEMINI_API_KEY
SUPADATA_API_KEY = _engine.SUPADATA_API_KEY
SUPADATA_BASE = _engine.SUPADATA_BASE

# ── LLM provider ──────────────────────────────────────────────────────────────
LLM_PROVIDER = "gemini"
CHARS_PER_TOKEN = 4
DEFAULTS = {"tail_pad_s": _engine.TAIL_PAD_S, "lead_pad_s": 0.06}
SUPADATA_CHUNK_SIZE = 180

# ── Token budgeting ────────────────────────────────────────────────────────────
EXPECTED_OUTPUT_TOKENS = 2_500

# ── Backoff ────────────────────────────────────────────────────────────────────
BACKOFF_MAX_RETRIES = 6
BACKOFF_BASE = 1.0
BACKOFF_CAP = 60.0

# ── Video judge ────────────────────────────────────────────────────────────────
VIDEO_JUDGE_MODEL = os.environ.get("VIDEO_JUDGE_MODEL", "gemini-2.5-flash-lite")

# ── Transcription provider / local Whisper ────────────────────────────────────
TRANSCRIBER = os.environ.get("TRANSCRIBER", "supadata")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE = os.environ.get("WHISPER_COMPUTE", "int8")
REFINE_WORKERS = int(os.environ.get("REFINE_WORKERS", "4"))
REFINE_WHISPER_MODEL = os.environ.get("REFINE_WHISPER_MODEL", "medium") or WHISPER_MODEL

# ── Work directory (Path only; mkdir is runtime-concern, not import-time) ──────
WORK_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent / "work"
