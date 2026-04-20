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


def clip_audio_refine_conditional() -> bool:
    """Phase 1 A1 gate. When True (default), refinement only fires on clips
    whose existing cue timings are untrustworthy (proportional/legacy source,
    or sparse sub-60-word windows with >500ms inter-word gaps). When False,
    falls back to the pre-A1 behaviour of refining every clip whenever
    `WHISPER_CLIP_REFINE_ENABLED` is on.
    """
    raw = os.environ.get("CLIP_AUDIO_REFINE_CONDITIONAL")
    if raw is None:
        return True
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


# Thresholds for `should_use_clip_audio_refine`. The inter-word-gap trigger
# guards against cue windows where Whisper reported timings but they're
# suspiciously stretched — only meaningful when the clip is genuinely sparse
# (<60 words across the whole window), because a slow TED-style speaker can
# have 500ms pauses with perfectly valid word timings.
_REFINE_INTER_WORD_GAP_SEC = 0.5
_REFINE_SPARSE_WORD_COUNT = 60


def should_use_clip_audio_refine(cues_in_clip: Sequence[Any]) -> bool:
    """Return True iff the clip's cue window has untrustworthy word timings
    and therefore benefits from acoustic Whisper refinement.

    Inputs are duck-typed against `IngestTranscriptCue` (any object with
    `word_source` and iterable `words` containing `start`/`end` floats).

    Trigger conditions (either):
      - ANY cue in the window has `word_source` in {"proportional", "legacy"}
        (YouTube manual captions / pre-Phase-A.1 persisted cues — no acoustic
        word timings, only character-proportional interpolation).
      - AVG inter-word gap in the window exceeds 500 ms AND the total word
        count is under 60. The word-count guard prevents false positives on
        slow speakers / speeches where 500 ms pauses are expected with valid
        timings.
    """
    if not cues_in_clip:
        # No cues → nothing to refine against. Caller keeps sentence-level bounds.
        return False

    untrusted_sources = {"proportional", "legacy"}
    for cue in cues_in_clip:
        src = getattr(cue, "word_source", None)
        if src in untrusted_sources:
            return True

    word_starts_ends: list[tuple[float, float]] = []
    for cue in cues_in_clip:
        words = getattr(cue, "words", None) or []
        for w in words:
            try:
                ws = float(getattr(w, "start", 0.0))
                we = float(getattr(w, "end", 0.0))
            except (TypeError, ValueError):
                continue
            if we > ws:
                word_starts_ends.append((ws, we))

    word_count = len(word_starts_ends)
    if word_count < 2:
        # Too few words to compute a meaningful gap — if we got here the cues
        # claimed a non-untrusted source, so trust them and skip refinement.
        return False
    if word_count >= _REFINE_SPARSE_WORD_COUNT:
        return False

    word_starts_ends.sort(key=lambda we: we[0])
    gaps: list[float] = []
    for i in range(1, len(word_starts_ends)):
        gap = word_starts_ends[i][0] - word_starts_ends[i - 1][1]
        if gap > 0:
            gaps.append(gap)
    if not gaps:
        return False
    avg_gap = sum(gaps) / len(gaps)
    return avg_gap > _REFINE_INTER_WORD_GAP_SEC


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

    Used as the fallback when the video-level audio cache is disabled or
    the full-video download failed. The primary path is
    `_ensure_full_video_audio` + `_slice_audio_for_clip` which amortises the
    yt-dlp cost across all clips from the same video.
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


# Process-local cache of full-video audio paths. Keyed by video_id so every
# clip from the same video reuses the single yt-dlp download. The directory
# is created lazily (`_ensure_clip_audio_cache_dir`) under the system tempdir
# with the `reelai-ingest-` prefix so the existing orphan sweeper in
# `app/ingestion/download.py` cleans up any leaks on the next worker boot.
_FULL_VIDEO_AUDIO_CACHE: dict[str, Path] = {}
_CLIP_AUDIO_CACHE_DIR: Path | None = None

# Full-video yt-dlp downloads can be large — bump the timeout well above the
# per-clip case. 6 min handles a 2-hour video at 320kbps on a slow connection.
_FULL_VIDEO_YT_DLP_TIMEOUT_SEC = 360


def _ensure_clip_audio_cache_dir() -> Path:
    """Lazily create (and memoise) the per-process directory that holds full
    video audio downloads. Uses `mkdtemp` with the `reelai-ingest-` prefix so
    `sweep_orphans` in `app/ingestion/download.py` reaps it after an hour."""
    global _CLIP_AUDIO_CACHE_DIR
    if _CLIP_AUDIO_CACHE_DIR is not None and _CLIP_AUDIO_CACHE_DIR.exists():
        return _CLIP_AUDIO_CACHE_DIR
    _CLIP_AUDIO_CACHE_DIR = Path(tempfile.mkdtemp(prefix="reelai-ingest-clip-audio-"))
    return _CLIP_AUDIO_CACHE_DIR


def _ensure_full_video_audio(video_id: str) -> Path | None:
    """Return a path to the full video's 16kHz mono WAV, downloading once
    on first call per video. Returns None if yt-dlp is unavailable or the
    download fails — the caller falls back to per-clip windowed download.

    The download includes the *entire* video because slicing locally with
    ffmpeg on a cached WAV is two orders of magnitude faster than re-issuing
    yt-dlp per clip (N clips × ~5-10 s network + decode vs one ~10-30 s
    download + N × ~0.5 s ffmpeg slices)."""
    cached = _FULL_VIDEO_AUDIO_CACHE.get(video_id)
    if cached is not None and cached.exists() and cached.stat().st_size > 0:
        return cached

    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        return None

    cache_dir = _ensure_clip_audio_cache_dir()
    out_path = cache_dir / f"{video_id}.wav"
    out_tpl = str(cache_dir / f"{video_id}.%(ext)s")
    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = [
        "yt-dlp",
        "-f", "bestaudio/best",
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
            timeout=_FULL_VIDEO_YT_DLP_TIMEOUT_SEC,
        )
    except FileNotFoundError:
        return None
    except subprocess.TimeoutExpired:
        logger.warning("yt-dlp full-video download timed out for %s", video_id)
        return None
    except subprocess.CalledProcessError as exc:
        logger.warning(
            "yt-dlp full-video download failed for %s: %s",
            video_id, (exc.stderr or b"")[:500].decode(errors="ignore"),
        )
        return None

    if not out_path.exists() or out_path.stat().st_size == 0:
        # yt-dlp may have named the output differently (e.g. .m4a before
        # the postprocessor ran). Pick any .wav in the cache dir named after
        # the video id.
        candidates = [p for p in cache_dir.glob(f"{video_id}.*") if p.suffix.lower() == ".wav"]
        if not candidates:
            return None
        out_path = candidates[0]

    _FULL_VIDEO_AUDIO_CACHE[video_id] = out_path
    return out_path


def _slice_audio_for_clip(
    full_audio: Path,
    dl_start_sec: float,
    dl_end_sec: float,
    out_path: Path,
) -> Path | None:
    """Extract `[dl_start_sec, dl_end_sec]` from the cached full-video WAV
    into `out_path` using ffmpeg's pcm_s16le re-encode (fast, sample-accurate
    on a mono 16kHz file). Returns the output path or None on failure."""
    duration = max(0.05, dl_end_sec - dl_start_sec)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-ss", f"{max(0.0, dl_start_sec):.3f}",
        "-i", str(full_audio),
        "-t", f"{duration:.3f}",
        "-ac", "1",
        "-ar", "16000",
        "-acodec", "pcm_s16le",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=_YT_DLP_TIMEOUT_SEC)
    except FileNotFoundError:
        logger.debug("ffmpeg binary not on PATH; cannot slice cached audio")
        return None
    except subprocess.TimeoutExpired:
        logger.warning("ffmpeg slice timed out for %s", full_audio.name)
        return None
    except subprocess.CalledProcessError as exc:
        logger.warning(
            "ffmpeg slice failed for %s: %s",
            full_audio.name, (exc.stderr or b"")[:500].decode(errors="ignore"),
        )
        return None
    if not out_path.exists() or out_path.stat().st_size == 0:
        return None
    return out_path


def _faster_whisper_words(audio_path: Path) -> list[WhisperWord]:
    """Transcribe a short clip with locally-installed faster-whisper and
    return word timings relative to the audio's own t=0. Returns [] if the
    package is missing, the model fails to load, or transcription fails —
    the caller then falls through to the Groq hosted endpoint.

    Used as the middle tier between WhisperX and Groq so local runs without
    the heavy `whisperx` wheel still get free, offline word timings.
    """
    try:
        from ..ingestion.transcribe import (
            _faster_whisper_transcribe,
            _load_faster_whisper_model,
            TranscriptionError,
        )
    except Exception:
        return []
    if _load_faster_whisper_model() is None:
        return []
    try:
        cues = _faster_whisper_transcribe(audio_path, language="en")
    except TranscriptionError:
        return []
    except Exception:
        logger.exception("faster-whisper raised during clip refinement")
        return []
    if not cues:
        return []
    out: list[WhisperWord] = []
    for cue in cues:
        for w in cue.words:
            text = (w.text or "").strip()
            if not text or w.end <= w.start:
                continue
            out.append(WhisperWord(text=text, start=float(w.start), end=float(w.end)))
    return out


def _call_whisper(audio_path: Path, dl_start_sec: float) -> list[WhisperWord]:
    """Run the clip audio through WhisperX (preferred), then local
    faster-whisper, then Groq hosted Whisper as the last-resort fallback.
    Word timestamps (relative to the extracted audio's own t=0) are
    converted to absolute video seconds by adding the download offset.

    Phase 4(b): when `WHISPERX_ENABLED=true` (default) and the package is
    installed, WhisperX wav2vec2 alignment runs locally for ±30 ms word
    precision. When WhisperX is unavailable, faster-whisper (also local)
    runs at ±80-120 ms precision. Groq is only reached when both local
    paths return nothing.
    """
    try:
        from ..ingestion.whisperx_transcribe import (
            whisperx_enabled,
            whisperx_words_for_audio,
        )
    except Exception:
        whisperx_enabled = lambda: False  # type: ignore[assignment]
        whisperx_words_for_audio = None  # type: ignore[assignment]

    if whisperx_enabled() and whisperx_words_for_audio is not None:
        try:
            wx_words = whisperx_words_for_audio(audio_path, language="en")
        except Exception:
            logger.exception("whisperx_words_for_audio raised during clip refinement")
            wx_words = []
        if wx_words:
            return [
                WhisperWord(
                    text=w.text,
                    start=w.start + dl_start_sec,
                    end=w.end + dl_start_sec,
                )
                for w in wx_words
            ]

    fw_words = _faster_whisper_words(audio_path)
    if fw_words:
        return [
            WhisperWord(
                text=w.text,
                start=w.start + dl_start_sec,
                end=w.end + dl_start_sec,
            )
            for w in fw_words
        ]

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
            # Primary path: reuse the full-video audio cached on first refine
            # for this video_id, slice locally with ffmpeg. Falls back to the
            # per-clip windowed yt-dlp download only when the cache is unusable.
            audio_path: Path | None = None
            full = _ensure_full_video_audio(video_id)
            if full is not None:
                slice_out = tmpdir / f"{video_id}_{round(float(t_start), 1)}_{round(float(t_end), 1)}.wav"
                audio_path = _slice_audio_for_clip(full, dl_start, dl_end, slice_out)
            if audio_path is None:
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
    "clip_audio_refine_conditional",
    "should_use_clip_audio_refine",
]
