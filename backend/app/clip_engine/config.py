"""Clip-engine config, bridged from env with practice-folder defaults.
Import stays key-free: missing keys raise only when a call needs them.
"""
from __future__ import annotations

import os

from .errors import SearchError, ClipError


def _flag(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() not in ("0", "false", "no", "")


SUPADATA_API_KEY = os.environ.get("SUPADATA_API_KEY", "")
SUPADATA_BASE = os.environ.get("SUPADATA_BASE", "https://api.supadata.ai/v1")
SUPADATA_SEARCH_URL = f"{SUPADATA_BASE}/youtube/search"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
SEGMENT_MODEL = os.environ.get("SEGMENT_MODEL", GEMINI_MODEL)

CLIP_ENGINE = os.environ.get("CLIP_ENGINE", "gemini")
OUTPUT_MODE = os.environ.get("OUTPUT_MODE", "embed")
PRECISE_BOUNDARIES = _flag("PRECISE_BOUNDARIES", False)
SEGMENT_FINE_SNAP = _flag("SEGMENT_FINE_SNAP", True)

SEGMENT_MIN_CLIP_S = float(os.environ.get("SEGMENT_MIN_CLIP_S", "15"))
SEGMENT_MAX_CLIPS = int(os.environ.get("SEGMENT_MAX_CLIPS", "40"))
SEGMENT_MAX_OUTPUT_TOKENS = int(os.environ.get("SEGMENT_MAX_OUTPUT_TOKENS", "24576"))
TAIL_PAD_S = float(os.environ.get("SEGMENT_TAIL_PAD_S", "0.15"))

CLIP_SEARCH_MAX_VIDEOS = int(os.environ.get("CLIP_SEARCH_MAX_VIDEOS", "5"))
SEARCH_BREADTH = int(os.environ.get("CLIP_SEARCH_BREADTH", "5"))


def require_supadata_key() -> str:
    if not SUPADATA_API_KEY:
        raise SearchError("SUPADATA_API_KEY is not set.")
    return SUPADATA_API_KEY


def require_gemini_key() -> str:
    if not GEMINI_API_KEY:
        raise ClipError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set.")
    return GEMINI_API_KEY
