# backend/app/clip_engine/run.py
"""Inline adapter for practice's fast Supadata -> Gemini -> iframe path."""
from __future__ import annotations

from . import config
from .cancellation import raise_if_cancelled
from .clipper import embed
from ...pipeline import gemini_segment
from .errors import (
    CancellationError,
    ClipError,
    ProviderError,
    TranscriptError,
    UnsupportedURLError,
)
from .metadata import extract_video_id


def _transcribe(url: str, video_id: str, settings: dict) -> dict:
    """Fetch a hosted timestamped transcript as {segments, words, duration, ...}."""
    from .clipper.pipeline.transcribe import transcribe_supadata  # lazy
    try:
        return transcribe_supadata(url, video_id, settings)
    except CancellationError:
        raise
    except ProviderError:
        raise
    except Exception as exc:  # normalize to engine error
        raise TranscriptError(f"Supadata transcript failed for {video_id}: {exc}") from exc


def _wire_segment_runtime(settings: dict, video_id: str) -> None:
    """Bridge the hosted job ledger into practice's unchanged segment router."""
    context = settings.get("generation_context") or settings.get("provider_context")
    if context is not None:
        context.reserve("segmentation")

    existing_sink = settings.get("_segment_telemetry")

    def sink(event: dict) -> None:
        if callable(existing_sink):
            existing_sink(event)
        if context is None or event.get("event") != "model_call":
            return
        context.record_gemini(
            attempt=max(1, int(event.get("retries") or 0) + 1),
            model_used=str(event.get("model") or ""),
            quality_degraded=str(event.get("operation") or "").startswith("pro_fallback"),
            usage=event,
        )

    settings["_segment_telemetry"] = sink
    settings["_segment_cancelled"] = settings.get("should_cancel")
    settings["video_id"] = video_id


def clip(url: str, topic: str, settings: dict | None = None, *, should_cancel=None) -> dict:
    raise_if_cancelled(should_cancel)
    video_id = extract_video_id(url)
    if not video_id:
        raise UnsupportedURLError(f"Not a recognized YouTube URL: {url}")
    settings = dict(settings or {})
    settings.setdefault("segment_fine_snap", config.SEGMENT_FINE_SNAP)
    settings.setdefault("segment_min_clip_s", config.SEGMENT_MIN_CLIP_S)
    # Practice's hard validators already discard unsafe clips. Ship surviving
    # Flash clips immediately; reserve Pro for an empty/failed Flash response.
    settings.setdefault("segment_accept_partial_flash", True)
    settings["should_cancel"] = should_cancel

    canonical_url = f"https://www.youtube.com/watch?v={video_id}"
    transcript = _transcribe(canonical_url, video_id, settings)
    raise_if_cancelled(should_cancel)
    if not (transcript.get("segments")):
        raise TranscriptError(f"Empty transcript for {video_id}")

    try:
        _wire_segment_runtime(settings, video_id)
        clips, notes = gemini_segment.segment_clips(
            transcript,
            settings,
            topic=topic or "",
            video_id=video_id,
        )
        raise_if_cancelled(should_cancel)
    except CancellationError:
        raise
    except ProviderError:
        raise
    except Exception as exc:
        raise ClipError(f"Gemini segmentation failed for {video_id}: {exc}") from exc

    for c in clips:
        raise_if_cancelled(should_cancel)
        c["embed_url"] = embed.embed_url(video_id, c["start"], c["end"])
    return {"video_id": video_id, "clips": clips, "transcript": transcript, "notes": notes}
