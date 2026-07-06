# backend/app/clip_engine/run.py
"""Inline clip runner — replaces the practice clipper's job/SSE surface with a
plain function. Supadata transcript -> one Gemini pass -> embed clip specs.
"""
from __future__ import annotations

from . import config
from .clipper import embed
from .clipper.pipeline import gemini_segment
from .errors import ClipError, TranscriptError, UnsupportedURLError
from .metadata import extract_video_id


def _transcribe(url: str, video_id: str, settings: dict) -> dict:
    """Fetch a timestamped Supadata transcript as {segments, words, duration, ...}."""
    from .clipper.pipeline.transcribe import transcribe_supadata  # lazy
    try:
        return transcribe_supadata(url, video_id, settings)
    except Exception as exc:  # normalize to engine error
        raise TranscriptError(f"Supadata transcript failed for {video_id}: {exc}") from exc


def clip(url: str, topic: str, settings: dict | None = None) -> dict:
    video_id = extract_video_id(url)
    if not video_id:
        raise UnsupportedURLError(f"Not a recognized YouTube URL: {url}")
    settings = dict(settings or {})
    settings.setdefault("segment_model", config.SEGMENT_MODEL)
    settings.setdefault("segment_fine_snap", config.SEGMENT_FINE_SNAP)
    settings.setdefault("segment_min_clip_s", config.SEGMENT_MIN_CLIP_S)

    transcript = _transcribe(url, video_id, settings)
    if not (transcript.get("segments")):
        raise TranscriptError(f"Empty transcript for {video_id}")

    try:
        clips, notes = gemini_segment.segment_clips(transcript, settings)
    except Exception as exc:
        raise ClipError(f"Gemini segmentation failed for {video_id}: {exc}") from exc

    for c in clips:
        c["embed_url"] = embed.embed_url(video_id, c["start"], c["end"])
    return {"video_id": video_id, "clips": clips, "transcript": transcript, "notes": notes}
