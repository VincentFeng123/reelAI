# backend/app/clip_engine/clipper/config.py
"""Vendored clipper config shim → delegates to app.clip_engine.config.
Adds the extra constants the vendored gemini path reads that aren't in the
engine config, with practice defaults. NO .env file load, NO paths/mkdir.
"""
from __future__ import annotations

import os

from .. import config as _engine

# ── Delegated to engine config ────────────────────────────────────────────────
SEGMENT_MODEL = _engine.SEGMENT_MODEL
SEGMENT_FALLBACK_MODEL = _engine.SEGMENT_FALLBACK_MODEL
SEGMENT_FINE_SNAP = _engine.SEGMENT_FINE_SNAP
SEGMENT_MIN_CLIP_S = _engine.SEGMENT_MIN_CLIP_S
SEGMENT_MAX_CLIP_S = _engine.SEGMENT_MAX_CLIP_S
SEGMENT_INFORMATIVENESS_MIN = _engine.SEGMENT_INFORMATIVENESS_MIN
SEGMENT_TOPIC_RELEVANCE_MIN = _engine.SEGMENT_TOPIC_RELEVANCE_MIN
SEGMENT_MAX_CLIPS = _engine.SEGMENT_MAX_CLIPS
SEGMENT_MAX_OUTPUT_TOKENS = _engine.SEGMENT_MAX_OUTPUT_TOKENS
SEGMENT_MAX_INPUT_TOKENS = _engine.SEGMENT_MAX_INPUT_TOKENS
SEGMENT_BATCH_MAX_CUES = _engine.SEGMENT_BATCH_MAX_CUES
SEGMENT_BATCH_OVERLAP_CUES = _engine.SEGMENT_BATCH_OVERLAP_CUES
ASSESSMENT_MODEL = _engine.ASSESSMENT_MODEL
GEMINI_MODEL = _engine.GEMINI_MODEL
GEMINI_API_KEY = _engine.GEMINI_API_KEY
SUPADATA_API_KEY = _engine.SUPADATA_API_KEY
SUPADATA_BASE = _engine.SUPADATA_BASE
SUPADATA_CHUNK_SIZE = _engine.SUPADATA_CHUNK_SIZE

# ── LLM provider ──────────────────────────────────────────────────────────────
LLM_PROVIDER = "gemini"
CHARS_PER_TOKEN = 4
DEFAULTS = {"lead_pad_s": 0.06}

# ── Token budgeting ────────────────────────────────────────────────────────────
EXPECTED_OUTPUT_TOKENS = 2_500

# ── Backoff ────────────────────────────────────────────────────────────────────
BACKOFF_MAX_RETRIES = 2
BACKOFF_BASE = 1.0
BACKOFF_CAP = 30.0

# ── Video judge ────────────────────────────────────────────────────────────────
VIDEO_JUDGE_MODEL = os.environ.get("VIDEO_JUDGE_MODEL", "gemini-2.5-flash-lite")
