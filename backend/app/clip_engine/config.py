"""Clip-engine config, bridged from env with practice-folder defaults.
Import stays key-free: missing keys raise only when a call needs them.
"""
from __future__ import annotations

import os

from .errors import ProviderConfigurationError


def _flag(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() not in ("0", "false", "no", "")


SUPADATA_API_KEY = os.environ.get("SUPADATA_API_KEY", "")
SUPADATA_BASE = os.environ.get("SUPADATA_BASE", "https://api.supadata.ai/v1")
SUPADATA_SEARCH_URL = f"{SUPADATA_BASE}/youtube/search"
SUPADATA_CHUNK_SIZE = int(os.environ.get("SUPADATA_CHUNK_SIZE", "180"))

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
TOPIC_MODEL = os.environ.get("TOPIC_MODEL", "gemini-3.1-pro-preview")
SEGMENT_MODEL = os.environ.get("SEGMENT_MODEL", GEMINI_MODEL)
SEGMENT_FALLBACK_MODEL = os.environ.get("SEGMENT_FALLBACK_MODEL", TOPIC_MODEL).strip()
# Query-expansion uses the cheaper lite tier; segmentation keeps its own model.
EXPAND_MODEL = os.environ.get("EXPAND_MODEL", "gemini-2.5-flash-lite")

CLIP_ENGINE = os.environ.get("CLIP_ENGINE", "gemini")
OUTPUT_MODE = os.environ.get("OUTPUT_MODE", "embed")
PRECISE_BOUNDARIES = _flag("PRECISE_BOUNDARIES", False)
SEGMENT_FINE_SNAP = _flag("SEGMENT_FINE_SNAP", True)

SEGMENT_MIN_CLIP_S = float(os.environ.get("SEGMENT_MIN_CLIP_S", "1"))
SEGMENT_MAX_CLIP_S = float(os.environ.get("SEGMENT_MAX_CLIP_S", "180"))
SEGMENT_INFORMATIVENESS_MIN = float(os.environ.get("SEGMENT_INFORMATIVENESS_MIN", "0.6"))
SEGMENT_TOPIC_RELEVANCE_MIN = float(os.environ.get("SEGMENT_TOPIC_RELEVANCE_MIN", "0.6"))
SEGMENT_MAX_CLIPS = int(os.environ.get("SEGMENT_MAX_CLIPS", "40"))
SEGMENT_MAX_OUTPUT_TOKENS = int(os.environ.get("SEGMENT_MAX_OUTPUT_TOKENS", "24576"))
SEGMENT_MAX_INPUT_TOKENS = int(os.environ.get("SEGMENT_MAX_INPUT_TOKENS", "12000"))
SEGMENT_BATCH_MAX_CUES = int(os.environ.get("SEGMENT_BATCH_MAX_CUES", "160"))
SEGMENT_BATCH_OVERLAP_CUES = int(os.environ.get("SEGMENT_BATCH_OVERLAP_CUES", "4"))

CLIP_SEARCH_MAX_VIDEOS = int(os.environ.get("CLIP_SEARCH_MAX_VIDEOS", "5"))
SEARCH_BREADTH = int(os.environ.get("CLIP_SEARCH_BREADTH", "3"))


def require_supadata_key() -> str:
    if not SUPADATA_API_KEY:
        raise ProviderConfigurationError(
            "SUPADATA_API_KEY is not set.", provider="supadata", operation="search"
        )
    return SUPADATA_API_KEY


def require_gemini_key() -> str:
    if not GEMINI_API_KEY:
        raise ProviderConfigurationError(
            "GEMINI_API_KEY (or GOOGLE_API_KEY) is not set.",
            provider="gemini",
            operation="segmentation",
        )
    return GEMINI_API_KEY
