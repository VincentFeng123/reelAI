"""YouTube video liveness probe backed by the ``video_liveness_cache`` table.

Ranked feeds can serve reels whose source video has been deleted or privated.
Each unique video is probed via the YouTube oEmbed endpoint (HEAD request,
~200-400ms) and the result is cached with a TTL so the probe doesn't run on
every feed request. When the cached entry is still fresh we return it
directly; stale entries trigger a re-probe.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import httpx

from ..db import fetch_one, now_iso, upsert

logger = logging.getLogger(__name__)

_OEMBED_ENDPOINT = "https://www.youtube.com/oembed"
_DEFAULT_TTL_SECONDS = 28800  # 8 hours — tuned to keep the hit rate high without masking fresh deletions for too long.
_PROBE_TIMEOUT_SEC = 3.0


def _parse_iso(raw: str) -> datetime | None:
    if not raw:
        return None
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _probe(video_id: str) -> bool:
    url = f"{_OEMBED_ENDPOINT}?url=https://www.youtube.com/watch?v={video_id}&format=json"
    try:
        response = httpx.head(url, timeout=_PROBE_TIMEOUT_SEC, follow_redirects=True)
    except httpx.HTTPError as exc:
        logger.debug("liveness probe network error for %s: %s", video_id, exc)
        # Fail open: if the probe network-errors we still want to surface the reel.
        return True
    status = response.status_code
    if status == 200:
        return True
    if status in (401, 403, 404):
        return False
    return True


def is_video_alive(video_id: str, *, conn: Any) -> bool:
    """Return True iff ``video_id`` appears live on YouTube.

    Consults ``video_liveness_cache`` first; re-probes and upserts when the
    cache row is missing or stale.
    """
    if not video_id:
        return True

    row = fetch_one(
        conn,
        "SELECT alive, checked_at, ttl_seconds FROM video_liveness_cache WHERE video_id = ?",
        (video_id,),
    )
    now = datetime.now(timezone.utc)
    if row:
        checked_at = _parse_iso(str(row.get("checked_at") or ""))
        ttl = int(row.get("ttl_seconds") or _DEFAULT_TTL_SECONDS)
        if checked_at is not None:
            age = (now - checked_at).total_seconds()
            if age < ttl:
                return bool(int(row.get("alive") or 0))

    alive = _probe(video_id)
    try:
        upsert(
            conn,
            "video_liveness_cache",
            {
                "video_id": video_id,
                "alive": 1 if alive else 0,
                "checked_at": now_iso(),
                "ttl_seconds": _DEFAULT_TTL_SECONDS,
            },
            pk="video_id",
        )
    except Exception:
        logger.exception("failed to upsert video_liveness_cache for %s", video_id)
    return alive


__all__ = ["is_video_alive"]
