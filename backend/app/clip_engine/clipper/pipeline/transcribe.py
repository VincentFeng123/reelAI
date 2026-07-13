"""Practice fast-path Supadata transcript adapter; local ASR is unsupported."""
from __future__ import annotations

from typing import Callable, Optional

from .. import config
from ...cancellation import raise_if_cancelled
from ...errors import TranscriptUnavailableError
from ...provider_cache import normalize_language

ProgressCb = Optional[Callable[[float, str], None]]


def transcribe_supadata(
    url: str,
    video_id: str,
    settings: dict,
    progress: ProgressCb = None,
) -> dict:
    """Fetch timed cues and retain an explicitly untrusted word approximation."""
    del video_id

    def emit(fraction: float, message: str = "") -> None:
        if progress:
            progress(max(0.0, min(1.0, fraction)), message)

    should_cancel = settings.get("should_cancel")
    raise_if_cancelled(should_cancel)
    from ..supadata_client import fetch_transcript_artifact

    emit(0.2, "Fetching timestamped transcript (Supadata)…")
    artifact = fetch_transcript_artifact(
        url,
        settings.get("language", "en"),
        should_cancel=should_cancel,
        context=settings.get("generation_context") or settings.get("provider_context"),
        cache_store=settings.get("provider_cache"),
        deadline_monotonic=settings.get("deadline_monotonic"),
        chunk_size=config.SUPADATA_CHUNK_SIZE,
    )
    requested_language = normalize_language(
        artifact.requested_language or settings.get("language", "en")
    )
    returned_language = normalize_language(artifact.returned_language)
    requested_base = requested_language.split("-", 1)[0]
    returned_base = returned_language.split("-", 1)[0]
    if requested_base and returned_base and requested_base != returned_base:
        raise TranscriptUnavailableError(
            "Supadata returned a transcript outside the requested language.",
            provider="supadata",
            operation="transcript",
            detail=(
                f"requested_language={requested_language};"
                f"returned_language={returned_language}"
            ),
        )
    segments = [dict(cue) for cue in artifact.segments]
    words: list[dict] = []
    for segment in segments:
        tokens = str(segment.get("text") or "").split()
        count = max(1, len(tokens))
        start = float(segment["start"])
        duration = max(0.01, float(segment["end"]) - start)
        for index, token in enumerate(tokens):
            words.append(
                {
                    "word": token,
                    "start": start + duration * index / count,
                    "end": start + duration * (index + 1) / count,
                    "timing_source": "interpolated",
                }
            )
    result = {
        "text": " ".join(segment["text"] for segment in segments),
        "duration": artifact.duration_sec,
        "words": words,
        "word_timing_source": "interpolated",
        "segments": segments,
        "source": "supadata",
        "chunks": segments,
        "video_id": artifact.video_id,
        "requested_language": artifact.requested_language,
        "returned_language": artifact.returned_language,
        "native_mode": artifact.native_mode,
        "transcript_mode": "native" if artifact.native_mode else "auto",
        "artifact_key": artifact.artifact_key,
    }
    raise_if_cancelled(should_cancel)
    emit(1.0, "Timestamped transcript ready")
    return result
