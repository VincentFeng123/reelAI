"""
Reel ingestion pipeline for YouTube native-caption sources.

This package provides a YouTube-only pipeline that:
  1. Canonicalizes YouTube video, playlist, and channel URLs.
  2. Retrieves search metadata and native caption cues through Supadata.
  3. Uses Gemini to propose clips grounded in exact caption cue IDs.
  4. Persists only clips inside the global 1-180 second safety envelope.
  5. Returns `ReelOut`-compatible payloads for the iOS and web clients.

Design rules (non-negotiable):
  * NEVER imports `app.services.reels` — the 9,921-line ReelService god-file stays quarantined.
  * Source media is never downloaded. Versioned search and transcript artifacts live in the DB.
  * Playback `video_url` returned to clients is ALWAYS an allow-listed host
    (youtube.com). The backend never acts as a video CDN — see
    `reelai/reelai/SharedViews.swift:1097` for the iOS WKWebView host allow-list that enforces this.
  * Only PUBLIC content is supported. Login-walled reels are refused with a clear 403.
  * Railway is the only Python runtime; Vercel hosts the Next.js client and proxies API requests.

Good-faith crawler posture:
  * User-Agent: "ReelAIBot/1.0 (+https://reelai.app/bot)"
  * robots.txt respected (cached 24h)
  * Per-platform rate limits on top of per-IP limits
  * Structured DMCA-friendly logging at INFO level (searchable by source_url)
  * `persistence.takedown_by_source_url(conn, url)` admin helper for takedown responses

Terms notice (surfaced in API responses and in every module's logging context):

    ReelAI ingests publicly-available media to generate educational summaries.
    Users are responsible for ensuring they have rights to process any content submitted.
    ReelAI preserves attribution and supports DMCA takedown requests at dmca@reelai.app.
"""

USER_AGENT = "ReelAIBot/1.0 (+https://reelai.app/bot)"

TERMS_NOTICE = (
    "ReelAI ingests publicly-available media to generate educational summaries. "
    "Users are responsible for ensuring they have rights to process any content submitted. "
    "ReelAI preserves attribution and supports DMCA takedown requests at dmca@reelai.app."
)

INGEST_SENTINEL_MATERIAL_ID = "ingest-scratch"
INGEST_SENTINEL_CONCEPT_ID = "ingest-scratch-concept"

__all__ = [
    "USER_AGENT",
    "TERMS_NOTICE",
    "INGEST_SENTINEL_MATERIAL_ID",
    "INGEST_SENTINEL_CONCEPT_ID",
]
