"""
Reel ingestion pipeline for YouTube, Instagram, and TikTok.

This package provides a fully decoupled, modular pipeline that:
  1. Downloads a reel URL to a temp workspace via yt-dlp (unified across platforms).
  2. Extracts platform-agnostic metadata (author, description, hashtags, counts, date, audio).
  3. Fetches a timestamped transcript (YouTube captions → yt-dlp subtitles → OpenAI Whisper API).
  4. Picks a smart 15-60s clip window using transcript-gap + ffmpeg silencedetect boundaries.
  5. Persists to the existing `reels` table via sentinel material/concept rows.
  6. Returns a `ReelOut`-compatible payload the iOS/web clients can render unchanged.
  7. UNCONDITIONALLY cleans up every temp file, plus a janitor sweep on import for orphans.

Design rules (non-negotiable):
  * NEVER imports `app.services.reels` — the 9,921-line ReelService god-file stays quarantined.
  * Videos are downloaded, processed, and deleted within a single function call. Nothing persists
    on disk past return. Only derived data (transcript text, metadata, timestamps, summaries)
    is stored in the DB.
  * Playback `video_url` returned to clients is ALWAYS an allow-listed host
    (youtube.com / instagram.com / tiktok.com). The backend never acts as a video CDN — see
    `reelai/reelai/SharedViews.swift:1097` for the iOS WKWebView host allow-list that enforces this.
  * Only PUBLIC content is supported. Login-walled reels are refused with a clear 403.
  * Refused in SERVERLESS_MODE (Vercel / Lambda) because yt-dlp + ffmpeg + Whisper exceed
    typical serverless time limits. Railway (single-instance) is the target runtime.

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
