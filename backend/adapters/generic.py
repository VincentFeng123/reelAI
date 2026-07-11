"""Generic fallback adapter — universal roles only, generic contracts.

Routes here for any content type without a dedicated adapter. Every other adapter
inherits the generic completeness contracts and overrides/extends where its genre
differs.
"""
from __future__ import annotations

from .base import BaseAdapter


class GenericAdapter(BaseAdapter):
    domain = "generic"
    content_types = ("other", "vlog", "commentary", "podcast", "unknown")

    def labeling_hints(self) -> str:
        return ("Content genre is unknown. Use the universal roles literally. Label sponsor "
                "reads, subscribe prompts, and channel housekeeping as administrative; label "
                "filler/repetition as irrelevant.")
