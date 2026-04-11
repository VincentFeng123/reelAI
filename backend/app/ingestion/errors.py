"""
Exception hierarchy for the ingestion pipeline.

Each class carries a `.status_code` hint that the endpoint layer maps into an HTTPException.
Keep messages terse and safe to surface to the client — callers should NOT include stack traces
or internal paths in the `args`. Use `logger.exception(...)` for the debugging detail instead.
"""

from __future__ import annotations


class IngestError(Exception):
    """Base class for all ingestion failures."""

    status_code: int = 500

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "error": self.__class__.__name__,
            "message": self.message,
        }
        if self.detail:
            payload["detail"] = self.detail
        return payload


class UnsupportedSourceError(IngestError):
    """The URL is not a supported platform, is private, or robots.txt disallows it."""

    status_code = 400


class DownloadError(IngestError):
    """yt-dlp or ffmpeg failed to retrieve / decode the source media."""

    status_code = 502


class TranscriptionError(IngestError):
    """All transcript strategies (platform captions / yt-dlp subs / Whisper) failed."""

    status_code = 502


class SegmentationError(IngestError):
    """Transcript was usable but no valid clip window could be selected."""

    status_code = 500


class MetadataError(IngestError):
    """Platform metadata was malformed or missing required fields we can't synthesize."""

    status_code = 502


class ServerlessUnavailable(IngestError):
    """The pipeline refused to run because the process is in SERVERLESS_MODE."""

    status_code = 503


class RateLimitedError(IngestError):
    """The per-platform outbound rate limiter rejected this call."""

    status_code = 429

    def __init__(self, message: str, *, retry_after_sec: float = 60.0, detail: str | None = None) -> None:
        super().__init__(message, detail=detail)
        self.retry_after_sec = retry_after_sec


__all__ = [
    "IngestError",
    "UnsupportedSourceError",
    "DownloadError",
    "TranscriptionError",
    "SegmentationError",
    "MetadataError",
    "ServerlessUnavailable",
    "RateLimitedError",
]
