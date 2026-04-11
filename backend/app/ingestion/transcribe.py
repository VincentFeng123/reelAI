"""
Transcript extraction with a three-strategy fallback chain:

    1. YouTube only: reuse the existing `YouTubeService.get_transcript` (caches via
       `transcript_cache` table; free; best quality when captions are authored).
    2. Any platform: parse yt-dlp's scraped subtitles from the `info_dict`
       (`automatic_captions` / `subtitles`). Free; ~90% of IG Reels with captions and
       most English YouTube videos land here.
    3. Fallback: OpenAI Whisper API (`whisper-1`) on the extracted audio. Paid
       (~$0.006/min) but works on anything that has speech.

Results are cached in the existing `llm_cache` table under key
`ingest_transcript:{platform}:{source_id}` so repeated ingests of the same URL skip
the expensive Whisper call.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from ..db import DatabaseIntegrityError, dumps_json, fetch_one, loads_json, now_iso, upsert
from .errors import ServerlessUnavailable, TranscriptionError
from .ffmpeg_tools import extract_audio_wav
from .logging_config import get_ingest_logger, log_event
from .models import IngestTranscriptCue, PlatformLiteral

logger: logging.Logger = get_ingest_logger(__name__)

_TRANSCRIPT_CACHE_PREFIX = "ingest_transcript:"
_WHISPER_MODEL = "whisper-1"
_WHISPER_MAX_FILE_BYTES = 24 * 1024 * 1024  # 24 MiB (Whisper API caps at 25 MiB)


def _serialize_cues(cues: list[IngestTranscriptCue]) -> str:
    return dumps_json([{"start": c.start, "end": c.end, "text": c.text} for c in cues])


def _deserialize_cues(raw: Any) -> list[IngestTranscriptCue]:
    payload = loads_json(raw, default=[])
    cues: list[IngestTranscriptCue] = []
    if not isinstance(payload, list):
        return cues
    for item in payload:
        if not isinstance(item, dict):
            continue
        try:
            cues.append(
                IngestTranscriptCue(
                    start=float(item.get("start", 0.0)),
                    end=float(item.get("end", 0.0)),
                    text=str(item.get("text", "")).strip(),
                )
            )
        except Exception:
            continue
    return cues


def _cache_key(platform: PlatformLiteral, source_id: str, language: str) -> str:
    return f"{_TRANSCRIPT_CACHE_PREFIX}{platform}:{source_id}:{language}"


def _load_cached(conn: Any, key: str) -> list[IngestTranscriptCue] | None:
    try:
        row = fetch_one(
            conn,
            "SELECT response_json FROM llm_cache WHERE cache_key = ?",
            (key,),
        )
    except Exception:
        logger.exception("transcript cache read failed for key=%s", key)
        return None
    if not row or not row.get("response_json"):
        return None
    cues = _deserialize_cues(row["response_json"])
    return cues or None


def _store_cache(conn: Any, key: str, cues: list[IngestTranscriptCue]) -> None:
    if not cues:
        return
    try:
        upsert(
            conn,
            "llm_cache",
            {
                "cache_key": key,
                "response_json": _serialize_cues(cues),
                "created_at": now_iso(),
            },
            pk="cache_key",
        )
    except DatabaseIntegrityError:
        pass
    except Exception:
        logger.exception("transcript cache write failed for key=%s", key)


# --------------------------------------------------------------------- #
# Strategy 1: YouTube-only, via the existing service
# --------------------------------------------------------------------- #


def _cues_from_youtube_service(conn: Any, youtube_service: Any, yt_video_id: str) -> list[IngestTranscriptCue]:
    if youtube_service is None:
        return []
    try:
        raw = youtube_service.get_transcript(conn, yt_video_id)
    except Exception:
        logger.exception("youtube_service.get_transcript raised for video_id=%s", yt_video_id)
        return []
    cues: list[IngestTranscriptCue] = []
    if not isinstance(raw, list):
        return cues
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        try:
            start = float(entry.get("start", 0.0))
            duration = float(entry.get("duration", 0.0))
            text = str(entry.get("text", "")).replace("\n", " ").strip()
            if not text:
                continue
            cues.append(IngestTranscriptCue(start=start, end=start + max(duration, 0.01), text=text))
        except Exception:
            continue
    return cues


# --------------------------------------------------------------------- #
# Strategy 2: yt-dlp's scraped subtitles / automatic captions
# --------------------------------------------------------------------- #


_VTT_TIMESTAMP_RE = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[.,](\d{3})\s+-->\s+(\d{1,2}):(\d{2}):(\d{2})[.,](\d{3})"
)


def _vtt_time_to_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def _parse_vtt(text: str) -> list[IngestTranscriptCue]:
    """Parse a WEBVTT or SRT payload into cues. Tolerant of minor format variants."""
    cues: list[IngestTranscriptCue] = []
    lines = text.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        match = _VTT_TIMESTAMP_RE.search(line)
        if not match:
            i += 1
            continue
        start = _vtt_time_to_seconds(*match.groups()[:4])
        end = _vtt_time_to_seconds(*match.groups()[4:])
        i += 1
        buffer: list[str] = []
        while i < n and lines[i].strip():
            clean = re.sub(r"<[^>]+>", "", lines[i])
            clean = clean.replace("&nbsp;", " ").strip()
            if clean:
                buffer.append(clean)
            i += 1
        text_joined = " ".join(buffer).strip()
        if text_joined:
            cues.append(IngestTranscriptCue(start=start, end=max(end, start + 0.01), text=text_joined))
    return cues


def _cues_from_info_dict_subtitles(info: dict[str, Any], language: str) -> list[IngestTranscriptCue]:
    """
    Parse cues from yt-dlp's info_dict. yt-dlp stores subtitles under `subtitles` (manual)
    and `automatic_captions` (auto-generated) keyed by language, each a list of format
    dicts with `url` and `ext`.
    """
    candidates: list[dict[str, Any]] = []
    for key in ("subtitles", "automatic_captions"):
        container = info.get(key)
        if not isinstance(container, dict):
            continue
        # Try the exact language first, then common English variants, then any.
        for lang_key in (language, "en", "en-US", "en-GB"):
            entries = container.get(lang_key)
            if isinstance(entries, list):
                candidates.extend(entries)
        if not candidates:
            for entries in container.values():
                if isinstance(entries, list):
                    candidates.extend(entries)
                    break
    if not candidates:
        return []

    # Some yt-dlp extractors inline the cues under `data`; others provide a `url` to fetch.
    import urllib.request

    for entry in candidates:
        if not isinstance(entry, dict):
            continue
        ext = (entry.get("ext") or "").lower()
        if ext not in ("vtt", "srv3", "srt", "ttml"):
            continue
        body: str | None = None
        inline = entry.get("data")
        if isinstance(inline, str) and inline.strip():
            body = inline
        else:
            url = entry.get("url")
            if isinstance(url, str):
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "ReelAIBot/1.0 (+https://reelai.app/bot)"})
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        body = resp.read().decode("utf-8", errors="replace")
                except Exception:
                    logger.exception("yt-dlp subtitle fetch failed for %s", url)
                    continue
        if not body:
            continue
        cues = _parse_vtt(body)
        if cues:
            return cues
    return []


# --------------------------------------------------------------------- #
# Strategy 3: OpenAI Whisper API
# --------------------------------------------------------------------- #


def _whisper_transcribe(
    audio_path: Path,
    *,
    openai_client: Any,
    language: str,
) -> list[IngestTranscriptCue]:
    """
    Upload the wav to OpenAI Whisper API and return timestamped cues. Kept as a module-level
    function so tests can `patch.object(transcribe, "_whisper_transcribe", ...)` without
    spinning up the full pipeline.
    """
    if openai_client is None:
        raise TranscriptionError(
            "Whisper fallback requested but no OpenAI client is configured",
        )

    try:
        size = audio_path.stat().st_size
    except OSError as exc:
        raise TranscriptionError("extracted audio file is missing", detail=str(exc)) from exc

    if size == 0:
        raise TranscriptionError("extracted audio file is empty")
    if size > _WHISPER_MAX_FILE_BYTES:
        raise TranscriptionError(
            f"audio file too large for Whisper API (size={size}, max={_WHISPER_MAX_FILE_BYTES})"
        )

    try:
        with open(audio_path, "rb") as f:
            completion = openai_client.audio.transcriptions.create(
                model=_WHISPER_MODEL,
                file=f,
                response_format="verbose_json",
                language=language if language else None,
                timestamp_granularities=["segment"],
            )
    except Exception as exc:
        raise TranscriptionError("Whisper API call failed", detail=str(exc)) from exc

    segments = getattr(completion, "segments", None)
    if segments is None and isinstance(completion, dict):
        segments = completion.get("segments")
    if not segments:
        raw_text = getattr(completion, "text", None) or (completion.get("text") if isinstance(completion, dict) else "")
        if not raw_text:
            raise TranscriptionError("Whisper returned no segments and no text")
        return [IngestTranscriptCue(start=0.0, end=max(1.0, float(audio_path.stat().st_size) / 32000.0), text=str(raw_text).strip())]

    cues: list[IngestTranscriptCue] = []
    for segment in segments:
        if isinstance(segment, dict):
            start = float(segment.get("start", 0.0))
            end = float(segment.get("end", start))
            text = str(segment.get("text", "")).strip()
        else:
            start = float(getattr(segment, "start", 0.0))
            end = float(getattr(segment, "end", start))
            text = str(getattr(segment, "text", "")).strip()
        if not text:
            continue
        cues.append(IngestTranscriptCue(start=start, end=max(end, start + 0.01), text=text))
    if not cues:
        raise TranscriptionError("Whisper segments contained no usable text")
    return cues


# --------------------------------------------------------------------- #
# Public orchestrator
# --------------------------------------------------------------------- #


def transcribe(
    conn: Any,
    *,
    platform: PlatformLiteral,
    source_id: str,
    info_dict: dict[str, Any],
    video_path: Path,
    workspace: Path,
    youtube_service: Any,
    openai_client: Any,
    language: str = "en",
    serverless_mode: bool = False,
    allow_openai_in_serverless: bool = False,
) -> list[IngestTranscriptCue]:
    """
    Return a timestamped transcript for the ingested reel.

    Strategy chain:
      (1) cache hit
      (2) YouTube service (yt platform only) — reuses existing cache
      (3) yt-dlp scraped subtitles from the info_dict
      (4) Whisper API fallback (refused in SERVERLESS_MODE unless explicitly overridden)

    Raises `TranscriptionError` if every strategy fails AND Whisper is unavailable.
    """
    cache_key = _cache_key(platform, source_id, language)

    cached = _load_cached(conn, cache_key)
    if cached:
        log_event(logger, logging.INFO, "transcript_cache_hit", platform=platform, source_id=source_id, count=len(cached))
        return cached

    # Strategy 2: YouTube transcript service
    if platform == "yt":
        cues = _cues_from_youtube_service(conn, youtube_service, source_id)
        if cues:
            _store_cache(conn, cache_key, cues)
            log_event(logger, logging.INFO, "transcript_from_youtube_service", source_id=source_id, count=len(cues))
            return cues

    # Strategy 3: yt-dlp scraped subtitles
    cues = _cues_from_info_dict_subtitles(info_dict, language)
    if cues:
        _store_cache(conn, cache_key, cues)
        log_event(
            logger,
            logging.INFO,
            "transcript_from_yt_dlp_subs",
            platform=platform,
            source_id=source_id,
            count=len(cues),
        )
        return cues

    # Strategy 4: Whisper API fallback
    if serverless_mode and not allow_openai_in_serverless:
        raise ServerlessUnavailable(
            "Whisper fallback is unavailable in serverless mode. Set ALLOW_OPENAI_IN_SERVERLESS=1 to override."
        )
    if openai_client is None:
        raise TranscriptionError(
            "No transcript available from platform sources and Whisper is not configured."
        )

    audio_path = workspace / "audio_16k.wav"
    try:
        extract_audio_wav(video_path, audio_path)
    except Exception as exc:
        raise TranscriptionError(
            "Could not extract audio for Whisper fallback",
            detail=str(exc),
        ) from exc

    cues = _whisper_transcribe(audio_path, openai_client=openai_client, language=language)
    _store_cache(conn, cache_key, cues)
    log_event(
        logger,
        logging.INFO,
        "transcript_from_whisper",
        platform=platform,
        source_id=source_id,
        count=len(cues),
    )
    return cues


__all__ = ["transcribe"]
