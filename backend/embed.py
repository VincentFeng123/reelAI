"""Canonical YouTube embed-URL helper — the single source of truth shared by the serving
path (orchestrator._build_embed_clips) and the labeling manifest (eval.sample_for_labeling).

Two copies existed and drifted: the serving copy stamped ``end = ceil(end) + 1`` (FE1 bug —
~1-1.7s bleed into the NEXT sentence on every clip), while the labeling copy used the correct
floor/ceil/guard form. Whole seconds only (the embed API takes integer seconds); floor(start)
and ceil(end) so the embed never trims real span content, and ``end >= start + 1`` so a clip is
never zero-length.
"""
from __future__ import annotations

import math


def embed_url(video_id: str, start: float, end: float) -> str:
    """YouTube embed URL that plays only [start, end] (whole seconds: floor start / ceil end so
    the embed never trims real span content, guarded to a ≥1s window)."""
    s = max(0, int(math.floor(float(start))))
    e = max(s + 1, int(math.ceil(float(end))))
    return f"https://www.youtube.com/embed/{video_id}?start={s}&end={e}&rel=0"
