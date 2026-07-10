"""Native-caption transcription adapter; local ASR is intentionally unsupported."""
from __future__ import annotations

from typing import Callable, Optional

from ...cancellation import raise_if_cancelled

ProgressCb = Optional[Callable[[float, str], None]]


def transcribe_supadata(
    url: str,
    video_id: str,
    settings: dict,
    progress: ProgressCb = None,
) -> dict:
    """Fetch and expose exact native-caption cues without invented word times."""
    del video_id

    def emit(fraction: float, message: str = "") -> None:
        if progress:
            progress(max(0.0, min(1.0, fraction)), message)

    should_cancel = settings.get("should_cancel")
    raise_if_cancelled(should_cancel)
    from ..supadata_client import fetch_transcript_artifact

    emit(0.2, "Fetching native captions (Supadata)…")
    artifact = fetch_transcript_artifact(
        url,
        settings.get("language", "en"),
        should_cancel=should_cancel,
        context=settings.get("generation_context") or settings.get("provider_context"),
        cache_store=settings.get("provider_cache"),
    )
    segments = [dict(cue) for cue in artifact.segments]
    result = {
        "text": " ".join(segment["text"] for segment in segments),
        "duration": artifact.duration_sec,
        "words": [],
        "segments": segments,
        "source": "supadata",
        "chunks": segments,
        "video_id": artifact.video_id,
        "requested_language": artifact.requested_language,
        "returned_language": artifact.returned_language,
        "native_mode": True,
        "artifact_key": artifact.artifact_key,
    }
    raise_if_cancelled(should_cancel)
    emit(1.0, "Native captions ready")
    return result
