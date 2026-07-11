"""Pipeline error type + raw-exception → human-readable message mapping."""
from __future__ import annotations


class PipelineError(Exception):
    """Raised by pipeline stages with an already-friendly, user-facing message."""


def friendly_error(e: Exception) -> str:
    """Map a raw exception to a readable line for the UI."""
    if isinstance(e, PipelineError):
        return str(e)
    s = str(e).lower()
    if "private video" in s or "this video is private" in s:
        return "This video is private and can't be downloaded."
    if "age" in s and "restrict" in s:
        return "This video is age-restricted and can't be fetched."
    if "members-only" in s or "join this channel" in s:
        return "This is a members-only video."
    if "not available in your country" in s or "geo" in s or "region" in s:
        return "This video is region-locked."
    if "video unavailable" in s or "removed" in s:
        return "Video unavailable (removed, or the URL is wrong)."
    if "resource_exhausted" in s or "quota" in s:
        return "Gemini free-tier quota hit — wait a bit and try again."
    if "api key" in s or "api_key" in s or "permission_denied" in s or "unauthenticated" in s:
        return "API key missing or invalid — check GEMINI_API_KEY in .env."
    if "429" in s or "rate limit" in s or "too many requests" in s:
        return "Rate limit hit — wait a minute and try again."
    if "ffmpeg" in s:
        return f"Video cutting failed: {e}"
    return f"Unexpected error: {e}"
