"""
Thin subprocess wrappers around ffmpeg / ffprobe.

No third-party deps — we shell out to the `ffmpeg` and `ffprobe` binaries that Railway
installs via `railpack.toml`. Every function here:

  * Uses `subprocess.run([...], check=True, capture_output=True, timeout=...)` with a per-call
    timeout so a hung decoder can't deadlock the worker.
  * Catches `FileNotFoundError` (ffmpeg missing) and `CalledProcessError` and wraps them
    in `DownloadError` / `SegmentationError` with the last ~800 chars of stderr.
  * Never constructs shell strings — all arguments are lists so paths with spaces don't break.

`check_ffmpeg_available()` is the one query you should call at process startup to fail fast
if ffmpeg is missing from the deploy environment.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path

from .errors import DownloadError, SegmentationError
from .logging_config import get_ingest_logger

logger: logging.Logger = get_ingest_logger(__name__)

# Per-call timeouts (seconds). ffmpeg operations on short reels should be much faster than this;
# these are "something is hung" backstops, not normal-case budgets.
_PROBE_TIMEOUT_SEC = 15
_EXTRACT_TIMEOUT_SEC = 120
_CUT_TIMEOUT_SEC = 120
_SILENCE_TIMEOUT_SEC = 60
_THUMBNAIL_TIMEOUT_SEC = 30

_STDERR_TAIL_CHARS = 800


def _tail(data: bytes | str | None) -> str:
    if data is None:
        return ""
    if isinstance(data, bytes):
        try:
            text = data.decode("utf-8", errors="replace")
        except Exception:
            text = repr(data)
    else:
        text = data
    text = text.strip()
    if len(text) <= _STDERR_TAIL_CHARS:
        return text
    return "..." + text[-_STDERR_TAIL_CHARS:]


def check_ffmpeg_available() -> bool:
    """
    True iff both `ffmpeg` and `ffprobe` are resolvable on PATH.

    Called at `IngestionPipeline.__init__` so an inbound request can refuse with a clear
    503 rather than mysteriously failing at download time.
    """
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def probe_duration(path: Path | str) -> float:
    """
    Return the container duration in seconds. Raises `DownloadError` on failure.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            timeout=_PROBE_TIMEOUT_SEC,
        )
    except FileNotFoundError as exc:
        raise DownloadError("ffprobe is not installed on this host", detail=str(exc)) from exc
    except subprocess.TimeoutExpired as exc:
        raise DownloadError("ffprobe timed out", detail=_tail(exc.stderr)) from exc
    except subprocess.CalledProcessError as exc:
        raise DownloadError(
            "ffprobe failed to read container duration",
            detail=_tail(exc.stderr),
        ) from exc

    text = result.stdout.decode("utf-8", errors="replace").strip()
    try:
        return float(text)
    except ValueError as exc:
        raise DownloadError(
            f"ffprobe returned non-numeric duration: {text!r}",
        ) from exc


def extract_audio_wav(video_path: Path | str, out_path: Path | str, *, sample_rate: int = 16000) -> Path:
    """
    Extract a mono 16kHz wav from the video. 16kHz is the native Whisper sample rate —
    sending it at 16kHz saves Whisper the resample step and keeps the upload small.
    """
    out = Path(out_path)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-acodec",
        "pcm_s16le",
        str(out),
    ]
    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            timeout=_EXTRACT_TIMEOUT_SEC,
        )
    except FileNotFoundError as exc:
        raise DownloadError("ffmpeg is not installed on this host", detail=str(exc)) from exc
    except subprocess.TimeoutExpired as exc:
        raise DownloadError("ffmpeg audio extraction timed out", detail=_tail(exc.stderr)) from exc
    except subprocess.CalledProcessError as exc:
        raise DownloadError(
            "ffmpeg audio extraction failed",
            detail=_tail(exc.stderr),
        ) from exc

    if not out.exists() or out.stat().st_size == 0:
        raise DownloadError(f"ffmpeg produced empty audio file at {out}")
    return out


def cut_clip(
    input_path: Path | str,
    start_sec: float,
    end_sec: float,
    out_path: Path | str,
    *,
    copy_streams: bool = True,
) -> Path:
    """
    Cut [start_sec, end_sec] from the input video into `out_path`.

    `copy_streams=True` uses stream copy (no re-encode) which is fast and lossless but aligns
    to keyframes — cuts may be ±1-2 sec from the requested boundary. `copy_streams=False`
    re-encodes for frame accuracy at the cost of CPU. Default True since the clip is not
    being served to end users; it's only used for audio extraction / thumbnails.
    """
    start = max(0.0, float(start_sec))
    end = max(start + 0.05, float(end_sec))
    duration = end - start

    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    if copy_streams:
        # -ss before -i seeks on input (fast, keyframe-aligned)
        cmd += ["-ss", f"{start:.3f}", "-i", str(input_path), "-t", f"{duration:.3f}", "-c", "copy"]
    else:
        # -ss after -i seeks on output (frame-accurate, slower)
        cmd += ["-i", str(input_path), "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac"]
    cmd += ["-movflags", "+faststart", str(out_path)]

    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            timeout=_CUT_TIMEOUT_SEC,
        )
    except FileNotFoundError as exc:
        raise SegmentationError("ffmpeg is not installed on this host", detail=str(exc)) from exc
    except subprocess.TimeoutExpired as exc:
        raise SegmentationError("ffmpeg cut timed out", detail=_tail(exc.stderr)) from exc
    except subprocess.CalledProcessError as exc:
        raise SegmentationError(
            "ffmpeg cut failed",
            detail=_tail(exc.stderr),
        ) from exc

    out = Path(out_path)
    if not out.exists() or out.stat().st_size == 0:
        raise SegmentationError(f"ffmpeg produced empty clip at {out}")
    return out


_SILENCE_START_RE = re.compile(r"silence_start:\s*([0-9.]+)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*([0-9.]+)\s*\|\s*silence_duration:\s*([0-9.]+)")


def silencedetect(
    audio_path: Path | str,
    *,
    noise_db: float = -30.0,
    min_silence_sec: float = 0.35,
) -> list[tuple[float, float]]:
    """
    Run ffmpeg's silencedetect filter over the audio and return [(start, end)] ranges of silence.

    These ranges are ideal cut-point anchors: the pipeline picks transcript windows that
    start / end inside (or touching) a silence range so clips never chop a word in half.

    `noise_db=-30` is a sensible default for speech on IG/TT/YT reels. Lower it if content
    is loud-by-design (music, sports commentary).
    """
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        str(audio_path),
        "-af",
        f"silencedetect=noise={noise_db}dB:d={min_silence_sec}",
        "-f",
        "null",
        "-",
    ]
    try:
        # silencedetect writes to stderr; we don't check returncode because ffmpeg sometimes
        # exits non-zero when piping to `null` even on success. We instead rely on parsing stderr.
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=_SILENCE_TIMEOUT_SEC,
        )
    except FileNotFoundError as exc:
        raise SegmentationError("ffmpeg is not installed on this host", detail=str(exc)) from exc
    except subprocess.TimeoutExpired as exc:
        raise SegmentationError("ffmpeg silencedetect timed out", detail=_tail(exc.stderr)) from exc

    stderr = (result.stderr or b"").decode("utf-8", errors="replace")
    ranges: list[tuple[float, float]] = []
    pending_start: float | None = None
    for line in stderr.splitlines():
        m_start = _SILENCE_START_RE.search(line)
        if m_start:
            try:
                pending_start = float(m_start.group(1))
            except ValueError:
                pending_start = None
            continue
        m_end = _SILENCE_END_RE.search(line)
        if m_end and pending_start is not None:
            try:
                end_sec = float(m_end.group(1))
            except ValueError:
                pending_start = None
                continue
            if end_sec > pending_start:
                ranges.append((pending_start, end_sec))
            pending_start = None
    return ranges


def thumbnail(video_path: Path | str, t_sec: float, out_path: Path | str) -> Path:
    """
    Grab a single frame at `t_sec` as a JPEG thumbnail. Used for preview cards.
    Failures raise `SegmentationError` but are not fatal to the pipeline — the caller
    can choose to skip thumbnails and continue.
    """
    out = Path(out_path)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, float(t_sec)):.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "4",
        str(out),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=_THUMBNAIL_TIMEOUT_SEC)
    except FileNotFoundError as exc:
        raise SegmentationError("ffmpeg is not installed on this host", detail=str(exc)) from exc
    except subprocess.TimeoutExpired as exc:
        raise SegmentationError("ffmpeg thumbnail timed out", detail=_tail(exc.stderr)) from exc
    except subprocess.CalledProcessError as exc:
        raise SegmentationError("ffmpeg thumbnail failed", detail=_tail(exc.stderr)) from exc

    if not out.exists() or out.stat().st_size == 0:
        raise SegmentationError(f"ffmpeg produced empty thumbnail at {out}")
    return out


__all__ = [
    "check_ffmpeg_available",
    "probe_duration",
    "extract_audio_wav",
    "cut_clip",
    "silencedetect",
    "thumbnail",
]
