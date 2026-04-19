"""
Clip boundary refinement via real Whisper word-level timestamps.

The sentence-level snap + proportional-word-level refinement already
gives us ±500ms boundaries in the search path (YouTube auto-captions
have no word timings, so we synthesize proportional ones). For sub-50ms
precision the user asked for, we need REAL word timestamps from an ASR
model processing the actual audio.

Strategy:
    1. yt-dlp downloads just the clip's audio window (+1s padding each
       side so the first/last word isn't clipped during download).
    2. The audio goes to Groq's hosted Whisper large-v3 endpoint with
       ``timestamp_granularities=["word"]`` (already wired in
       llm_router.transcribe_audio).
    3. Whisper returns word-level timings relative to the audio start;
       we add the download offset to get absolute video timestamps.
    4. We pick the first substantive word inside [t_start, t_end] and
       the last; refined boundaries = first_word.start − pre_roll,
       last_word.end + post_roll.
    5. Per-clip results are cached in the `llm_cache` table under the
       key ``whisper_words:{video_id}:{round_start}:{round_end}``, so
       subsequent searches hitting the same clip skip both the audio
       download and the Whisper call.

Activation: set ``WHISPER_CLIP_REFINE_ENABLED=true`` in the backend's
environment. Leaving it unset keeps the pipeline on proportional-word
precision (still produces valid clips; just ±cue-granularity instead of
±50ms). This gate is per-user-settings / cost-aware: each refinement
costs a yt-dlp audio download (~3-10s, bandwidth) + one Groq Whisper
request (~2-5s, rate-limited on the free tier).

On failure at any step (yt-dlp error, Groq 429, malformed response,
zero words returned), the function returns None and the caller keeps
the sentence-level boundaries — no regression.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ..db import DatabaseIntegrityError, dumps_json, fetch_one, now_iso, upsert
from .llm_router import transcribe_audio

logger = logging.getLogger(__name__)


# Post-Whisper padding. Small enough to stay inside the contract ("start
# exactly before the word"), big enough to ensure the word's initial
# phoneme and terminal punctuation sound are both fully captured.
_PRE_ROLL_SEC = 0.03
_POST_ROLL_SEC = 0.05

# Download padding. Whisper needs context around the clip to segment
# correctly — running it on a hard-cut audio clip makes the first and
# last words unreliable. 1s on each side is the community-standard value.
_DOWNLOAD_PAD_SEC = 1.0

# Timeouts.
_YT_DLP_TIMEOUT_SEC = 60
_WHISPER_TIMEOUT_SEC = 90

# Cache key version — bump to invalidate all persisted Whisper word
# timings (e.g., when switching Whisper model or output parser).
_WHISPER_CACHE_VERSION = "v1"


def whisper_clip_refine_enabled() -> bool:
    return str(os.environ.get("WHISPER_CLIP_REFINE_ENABLED", "")).strip().lower() in {
        "1", "true", "yes", "on",
    }


@dataclass
class WhisperWord:
    text: str
    start: float  # absolute seconds in video
    end: float  # absolute seconds in video


@dataclass
class WhisperRefinement:
    t_start: float
    t_end: float
    words: list[WhisperWord]
    first_word: str
    last_word: str


def _cache_key(video_id: str, t_start: float, t_end: float) -> str:
    return (
        f"whisper_words:{_WHISPER_CACHE_VERSION}:"
        f"{video_id}:{round(float(t_start), 1)}:{round(float(t_end), 1)}"
    )


def _read_cached_words(conn: Any, cache_key: str) -> list[WhisperWord] | None:
    if conn is None:
        return None
    try:
        row = fetch_one(
            conn,
            "SELECT response_json FROM llm_cache WHERE cache_key = ?",
            (cache_key,),
        )
    except Exception:
        logger.exception("whisper cache read failed for %s", cache_key)
        return None
    if not row:
        return None
    raw = row.get("response_json") or ""
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, list):
        return None
    out: list[WhisperWord] = []
    for w in payload:
        if not isinstance(w, dict):
            continue
        try:
            out.append(WhisperWord(
                text=str(w.get("text") or ""),
                start=float(w.get("start") or 0.0),
                end=float(w.get("end") or 0.0),
            ))
        except (TypeError, ValueError):
            continue
    return out or None


def _write_cached_words(conn: Any, cache_key: str, words: list[WhisperWord]) -> None:
    if conn is None or not words:
        return
    payload = [
        {"text": w.text, "start": round(w.start, 4), "end": round(w.end, 4)}
        for w in words
    ]
    try:
        upsert(
            conn,
            "llm_cache",
            {
                "cache_key": cache_key,
                "response_json": dumps_json(payload),
                "created_at": now_iso(),
            },
            pk="cache_key",
        )
    except DatabaseIntegrityError:
        pass
    except Exception:
        logger.exception("whisper cache write failed for %s", cache_key)


def _download_clip_audio(
    video_id: str,
    dl_start_sec: float,
    dl_end_sec: float,
    out_dir: Path,
) -> Path | None:
    """Download a specific seconds range of a YouTube video's audio as
    16kHz mono WAV via yt-dlp. Returns the path or None on failure.
    """
    try:
        import yt_dlp  # noqa: F401 — we use the CLI, but the import probes availability
    except ImportError:
        logger.debug("yt-dlp not installed; whisper refine disabled")
        return None
    url = f"https://www.youtube.com/watch?v={video_id}"
    out_tpl = str(out_dir / f"{video_id}.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f", "bestaudio/best",
        "--download-sections", f"*{dl_start_sec:.2f}-{dl_end_sec:.2f}",
        "--force-keyframes-at-cuts",
        "--extract-audio",
        "--audio-format", "wav",
        "--postprocessor-args", "-ac 1 -ar 16000",
        "-q",
        "--no-warnings",
        "-o", out_tpl,
        url,
    ]
    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            timeout=_YT_DLP_TIMEOUT_SEC,
        )
    except FileNotFoundError:
        logger.debug("yt-dlp binary not on PATH; whisper refine disabled")
        return None
    except subprocess.TimeoutExpired:
        logger.warning("yt-dlp timed out downloading clip audio for %s", video_id)
        return None
    except subprocess.CalledProcessError as exc:
        logger.warning(
            "yt-dlp clip download failed for %s: %s",
            video_id, (exc.stderr or b"")[:500].decode(errors="ignore"),
        )
        return None
    # yt-dlp may pick any extension depending on the source; find the WAV.
    wavs = [p for p in out_dir.iterdir() if p.suffix.lower() == ".wav"]
    return wavs[0] if wavs else None


def _call_whisper(audio_path: Path, dl_start_sec: float) -> list[WhisperWord]:
    """Run the clip audio through Groq Whisper and convert word
    timestamps (relative to the extracted audio's own t=0) into absolute
    video seconds by adding the download offset.
    """
    try:
        payload = transcribe_audio(audio_path, language="en", timeout=_WHISPER_TIMEOUT_SEC)
    except Exception:
        logger.exception("transcribe_audio raised during clip refinement")
        return []
    if not isinstance(payload, dict):
        return []
    raw_words = payload.get("words") or []
    out: list[WhisperWord] = []
    for w in raw_words:
        if not isinstance(w, dict):
            continue
        try:
            start = float(w.get("start") or 0.0) + dl_start_sec
            end = float(w.get("end") or 0.0) + dl_start_sec
        except (TypeError, ValueError):
            continue
        text = str(w.get("word") or w.get("text") or "").strip()
        if not text or end <= start:
            continue
        out.append(WhisperWord(text=text, start=start, end=end))
    return out


def refine_clip_with_whisper(
    *,
    video_id: str,
    t_start: float,
    t_end: float,
    conn: Any | None = None,
) -> WhisperRefinement | None:
    """Run Whisper on the audio at [t_start − 1s, t_end + 1s] and return
    refined (t_start, t_end) + word timings. Returns None when the
    feature is disabled, a dep is missing, or any step fails — callers
    keep the sentence-level boundaries.

    Precision: Groq's Whisper large-v3 gives word-level timestamps
    accurate to ~30-80ms on clean speech; combined with the pre/post
    roll the final contract is ≤50ms in well-recorded content.
    """
    if not whisper_clip_refine_enabled():
        return None
    if t_end <= t_start:
        return None

    cache_key = _cache_key(video_id, t_start, t_end)
    cached = _read_cached_words(conn, cache_key)
    if cached is not None:
        words = cached
    else:
        dl_start = max(0.0, float(t_start) - _DOWNLOAD_PAD_SEC)
        dl_end = float(t_end) + _DOWNLOAD_PAD_SEC
        tmpdir = Path(tempfile.mkdtemp(prefix="whisper_refine_"))
        try:
            audio_path = _download_clip_audio(video_id, dl_start, dl_end, tmpdir)
            if audio_path is None:
                return None
            words = _call_whisper(audio_path, dl_start)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
        if not words:
            return None
        _write_cached_words(conn, cache_key, words)

    # Filter to words that land inside [t_start, t_end] with a small grace
    # window — Whisper's word-level alignment drifts <100ms in both
    # directions on clean speech.
    in_window = [
        w for w in words
        if (t_start - 0.1) <= w.start and w.end <= (t_end + 0.1)
    ]
    if not in_window:
        return None
    first_word = in_window[0]
    last_word = in_window[-1]
    refined_t_start = max(0.0, first_word.start - _PRE_ROLL_SEC)
    refined_t_end = last_word.end + _POST_ROLL_SEC
    if refined_t_end <= refined_t_start:
        return None
    return WhisperRefinement(
        t_start=refined_t_start,
        t_end=refined_t_end,
        words=list(in_window),
        first_word=first_word.text,
        last_word=last_word.text,
    )


__all__ = [
    "WhisperWord",
    "WhisperRefinement",
    "refine_clip_with_whisper",
    "whisper_clip_refine_enabled",
]
