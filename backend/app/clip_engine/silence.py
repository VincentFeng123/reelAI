"""Bounded, energy-only acoustic verification for hosted YouTube clips.

The production selector already supplies complete transcript cue boundaries.
This module performs the smaller final check that captions cannot provide: it
seeks two six-second audio windows and proves that each cut lands in silence.
It never downloads or transcribes the full source and every failure is returned
as ``unavailable`` so callers can keep searching for another clip.
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import wave
from array import array
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence
from urllib.parse import parse_qs, urlparse

from backend.app.config import get_settings


CancelCheck = Callable[[], bool]
VerificationStatus = Literal["verified", "unavailable"]
PreparationStatus = Literal["ready", "unavailable"]

DEFAULT_TIMEOUT_SEC = 20.0
DEFAULT_PREPARE_TIMEOUT_SEC = 10.0
EDGE_WINDOW_SEC = 6.0
QUIET_THRESHOLD_DBFS = -38.0
MIN_QUIET_MS = 120
START_CUSHION_MS = 100
END_CUSHION_MS = 200
_FRAME_MS = 10
_YT_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
_YT_HOSTS = frozenset({"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"})


@dataclass(frozen=True)
class PreparedAudioSource:
    """Short-lived direct audio source resolved by yt-dlp.

    URL and headers may contain signed credentials, so repr intentionally omits
    them and final diagnostics expose only the non-sensitive format identifier.
    """

    url: str = field(repr=False)
    headers: Mapping[str, str] = field(default_factory=dict, repr=False)
    proxy_url: str = field(default="", repr=False)
    format_id: str = ""


@dataclass(frozen=True)
class AudioPreparationResult:
    status: PreparationStatus
    source: PreparedAudioSource | None = field(default=None, repr=False)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status == "ready" and self.source is not None


@dataclass(frozen=True)
class SilenceVerificationResult:
    status: VerificationStatus
    start_sec: float
    end_sec: float
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def verified(self) -> bool:
        return self.status == "verified"


@dataclass(frozen=True)
class _QuietInterval:
    start_sec: float
    end_sec: float
    preceded_by_sound: bool
    followed_by_sound: bool

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


class _Unavailable(RuntimeError):
    def __init__(self, stage: str, reason: str) -> None:
        super().__init__(reason)
        self.stage = stage
        self.reason = reason


def _is_cancelled(cancel_check: CancelCheck | None) -> bool:
    if cancel_check is None:
        return False
    try:
        return bool(cancel_check())
    except Exception:
        return True


def _canonical_watch_url(value: str) -> str | None:
    clean = str(value or "").strip()
    if _YT_ID.fullmatch(clean):
        return f"https://www.youtube.com/watch?v={clean}"
    try:
        parsed = urlparse(clean)
    except Exception:
        return None
    host = (parsed.hostname or "").lower()
    video_id = ""
    if host == "youtu.be":
        video_id = parsed.path.strip("/").split("/", 1)[0]
    elif host in _YT_HOSTS:
        if parsed.path.rstrip("/") == "/watch":
            video_id = (parse_qs(parsed.query).get("v") or [""])[0]
        else:
            parts = [part for part in parsed.path.split("/") if part]
            if len(parts) >= 2 and parts[0].lower() in {"embed", "live", "shorts"}:
                video_id = parts[1]
    if not _YT_ID.fullmatch(video_id):
        return None
    return f"https://www.youtube.com/watch?v={video_id}"


def _remaining(deadline: float, stage: str) -> float:
    value = deadline - time.monotonic()
    if value <= 0:
        raise _Unavailable(stage, "deadline_exceeded")
    return value


def _run_command(
    command: Sequence[str],
    *,
    deadline: float,
    cancel_check: CancelCheck | None,
    stage: str,
) -> tuple[bytes, bytes]:
    if _is_cancelled(cancel_check):
        raise _Unavailable(stage, "cancelled")
    _remaining(deadline, stage)
    try:
        process = subprocess.Popen(
            list(command),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, ValueError):
        raise _Unavailable(stage, "process_start_failed") from None

    while True:
        if _is_cancelled(cancel_check):
            process.kill()
            process.communicate()
            raise _Unavailable(stage, "cancelled")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            process.kill()
            process.communicate()
            raise _Unavailable(stage, "deadline_exceeded")
        wait_for = min(0.1, remaining)
        try:
            stdout, stderr = process.communicate(timeout=wait_for)
            break
        except subprocess.TimeoutExpired:
            continue
        except Exception:
            process.kill()
            process.communicate()
            raise _Unavailable(stage, "process_failed") from None

    if process.returncode != 0:
        raise _Unavailable(stage, "process_failed")
    return stdout, stderr


def _proxy_url() -> str:
    proxies = str(get_settings().proxy_urls or "")
    return next((part.strip() for part in proxies.split(",") if part.strip()), "")


def _yt_dlp_command(watch_url: str) -> list[str]:
    settings = get_settings()
    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--dump-single-json",
        "--skip-download",
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        "--socket-timeout",
        "8",
        "--retries",
        "1",
        "--fragment-retries",
        "1",
        "--remote-components",
        "ejs:github",
        "--format",
        "bestaudio/best",
    ]
    cookie_file = str(os.environ.get("YT_COOKIES_FILE") or "").strip()
    if cookie_file and os.path.isfile(cookie_file):
        command.extend(["--cookies", cookie_file])
    else:
        browser = str(os.environ.get("YT_COOKIES_FROM_BROWSER") or "").strip()
        if browser:
            command.extend(["--cookies-from-browser", browser])
    proxy = _proxy_url()
    if proxy:
        command.extend(["--proxy", proxy])
    provider_url = str(settings.ytdlp_pot_provider_url or "").strip()
    if provider_url:
        command.extend(
            ["--extractor-args", f"youtubepot-bgutilhttp:base_url={provider_url}"]
        )
    command.append(watch_url)
    return command


def _audio_entry(info: Mapping[str, Any]) -> Mapping[str, Any]:
    requested = info.get("requested_downloads")
    if isinstance(requested, list):
        for entry in requested:
            if isinstance(entry, Mapping) and isinstance(entry.get("url"), str):
                return entry
    if isinstance(info.get("url"), str):
        return info
    formats = info.get("formats")
    if not isinstance(formats, list):
        return {}
    audio = [
        entry
        for entry in formats
        if isinstance(entry, Mapping)
        and isinstance(entry.get("url"), str)
        and str(entry.get("acodec") or "none") != "none"
        and str(entry.get("vcodec") or "none") == "none"
    ]
    if not audio:
        return {}
    return max(audio, key=lambda entry: float(entry.get("abr") or entry.get("tbr") or 0.0))


def _prepare_audio_source(
    video_id_or_url: str,
    *,
    deadline: float,
    cancel_check: CancelCheck | None,
) -> PreparedAudioSource:
    watch_url = _canonical_watch_url(video_id_or_url)
    if watch_url is None:
        raise _Unavailable("resolve", "invalid_youtube_source")
    stdout, _ = _run_command(
        _yt_dlp_command(watch_url),
        deadline=deadline,
        cancel_check=cancel_check,
        stage="resolve",
    )
    try:
        info = json.loads(stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
        raise _Unavailable("resolve", "invalid_ytdlp_response") from None
    if not isinstance(info, Mapping):
        raise _Unavailable("resolve", "invalid_ytdlp_response")
    entry = _audio_entry(info)
    media_url = str(entry.get("url") or "").strip()
    if not media_url.startswith(("https://", "http://")):
        raise _Unavailable("resolve", "audio_url_missing")
    raw_headers = entry.get("http_headers") or info.get("http_headers") or {}
    headers = {
        str(key): str(value)
        for key, value in (raw_headers.items() if isinstance(raw_headers, Mapping) else ())
        if re.fullmatch(r"[A-Za-z0-9-]+", str(key))
        and "\r" not in str(value)
        and "\n" not in str(value)
        and str(key).lower() not in {"host", "content-length", "range"}
    }
    return PreparedAudioSource(
        url=media_url,
        headers=headers,
        proxy_url=_proxy_url(),
        format_id=str(entry.get("format_id") or info.get("format_id") or ""),
    )


def prepare_audio_source(
    video_id_or_url: str,
    *,
    timeout_sec: float = DEFAULT_PREPARE_TIMEOUT_SEC,
    cancel_check: CancelCheck | None = None,
) -> AudioPreparationResult:
    """Resolve bestaudio without downloading it, suitable for parallel prefetch.

    Callers may start this while transcript selection runs, then pass the result
    to :func:`verify_acoustic_boundaries`. Signed URLs are never placed in the
    returned diagnostics.
    """

    started = time.monotonic()
    if not math.isfinite(timeout_sec) or timeout_sec <= 0:
        return AudioPreparationResult(
            "unavailable", diagnostics={"stage": "resolve", "reason": "invalid_timeout"}
        )
    try:
        source = _prepare_audio_source(
            video_id_or_url,
            deadline=started + timeout_sec,
            cancel_check=cancel_check,
        )
    except _Unavailable as exc:
        return AudioPreparationResult(
            "unavailable",
            diagnostics={
                "stage": exc.stage,
                "reason": exc.reason,
                "elapsed_ms": round((time.monotonic() - started) * 1000),
            },
        )
    except Exception:
        return AudioPreparationResult(
            "unavailable",
            diagnostics={
                "stage": "resolve",
                "reason": "unexpected_failure",
                "elapsed_ms": round((time.monotonic() - started) * 1000),
            },
        )
    return AudioPreparationResult(
        "ready",
        source=source,
        diagnostics={
            "format_id": source.format_id,
            "elapsed_ms": round((time.monotonic() - started) * 1000),
        },
    )


def _ffmpeg_headers(headers: Mapping[str, str]) -> str:
    return "".join(f"{key}: {value}\r\n" for key, value in headers.items())


def _decode_window(
    source: PreparedAudioSource,
    *,
    window_start_sec: float,
    window_duration_sec: float,
    output_path: Path,
    ffmpeg_bin: str,
    deadline: float,
    cancel_check: CancelCheck | None,
) -> None:
    duration = min(EDGE_WINDOW_SEC, max(4.0, window_duration_sec))
    command = [ffmpeg_bin, "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
    if source.proxy_url.startswith(("http://", "https://")):
        command.extend(["-http_proxy", source.proxy_url])
    if source.headers:
        command.extend(["-headers", _ffmpeg_headers(source.headers)])
    command.extend(
        [
            "-rw_timeout",
            str(max(1, int(min(8.0, _remaining(deadline, "decode")) * 1_000_000))),
            "-ss",
            f"{max(0.0, window_start_sec):.3f}",
            "-i",
            source.url,
            "-t",
            f"{duration:.3f}",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
    )
    _run_command(command, deadline=deadline, cancel_check=cancel_check, stage="decode")
    if not output_path.is_file() or output_path.stat().st_size <= 44:
        raise _Unavailable("decode", "empty_audio_window")


def _quiet_intervals(
    wav_path: Path,
    *,
    absolute_start_sec: float,
    threshold_dbfs: float,
    min_quiet_ms: int,
) -> list[_QuietInterval]:
    try:
        with wave.open(str(wav_path), "rb") as handle:
            if handle.getsampwidth() != 2 or handle.getnchannels() != 1:
                raise _Unavailable("analyze", "unsupported_pcm")
            sample_rate = handle.getframerate()
            raw = handle.readframes(handle.getnframes())
    except _Unavailable:
        raise
    except (OSError, EOFError, wave.Error):
        raise _Unavailable("analyze", "invalid_audio_window") from None
    if sample_rate <= 0 or not raw:
        raise _Unavailable("analyze", "empty_audio_window")

    samples = array("h")
    samples.frombytes(raw)
    if sys.byteorder != "little":
        samples.byteswap()
    frame_samples = max(1, round(sample_rate * _FRAME_MS / 1000))
    quiet: list[bool] = []
    for offset in range(0, len(samples), frame_samples):
        frame = samples[offset : offset + frame_samples]
        if len(frame) < frame_samples:
            break
        rms = math.sqrt(sum(float(value) * float(value) for value in frame) / len(frame))
        dbfs = -math.inf if rms <= 0 else 20.0 * math.log10(rms / 32768.0)
        quiet.append(dbfs <= threshold_dbfs)

    minimum_frames = max(1, math.ceil(min_quiet_ms / _FRAME_MS))
    intervals: list[_QuietInterval] = []
    run_start: int | None = None
    for index in range(len(quiet) + 1):
        is_quiet = index < len(quiet) and quiet[index]
        if is_quiet and run_start is None:
            run_start = index
        elif not is_quiet and run_start is not None:
            if index - run_start >= minimum_frames:
                intervals.append(
                    _QuietInterval(
                        start_sec=absolute_start_sec + run_start * _FRAME_MS / 1000.0,
                        end_sec=absolute_start_sec + index * _FRAME_MS / 1000.0,
                        preceded_by_sound=run_start > 0 and not quiet[run_start - 1],
                        followed_by_sound=index < len(quiet) and not quiet[index],
                    )
                )
            run_start = None
    return intervals


def _pick_start_interval(
    intervals: Sequence[_QuietInterval], rough_start: float
) -> _QuietInterval | None:
    candidates = [
        interval
        for interval in intervals
        if interval.followed_by_sound
        and interval.end_sec >= rough_start - 2.5
        and interval.start_sec <= rough_start + 0.35
        and interval.duration_sec + 1e-9 >= START_CUSHION_MS / 1000.0
    ]
    if not candidates:
        return None

    def priority(interval: _QuietInterval) -> tuple[int, float]:
        if interval.start_sec <= rough_start <= interval.end_sec:
            return 0, abs(interval.end_sec - rough_start)
        if interval.end_sec <= rough_start:
            return 1, rough_start - interval.end_sec
        return 2, interval.start_sec - rough_start

    return min(candidates, key=priority)


def _pick_end_interval(
    intervals: Sequence[_QuietInterval], rough_end: float
) -> _QuietInterval | None:
    candidates = [
        interval
        for interval in intervals
        if interval.preceded_by_sound
        and interval.end_sec >= rough_end
        and interval.start_sec >= rough_end - 0.35
        and interval.start_sec <= rough_end + 2.5
        and interval.duration_sec + 1e-9 >= END_CUSHION_MS / 1000.0
    ]
    if not candidates:
        return None

    def priority(interval: _QuietInterval) -> tuple[int, float]:
        if interval.start_sec <= rough_end <= interval.end_sec:
            return 0, abs(interval.start_sec - rough_end)
        return 1, interval.start_sec - rough_end

    return min(candidates, key=priority)


def _unavailable(
    start_sec: float,
    end_sec: float,
    *,
    stage: str,
    reason: str,
    started: float,
    extra: Mapping[str, Any] | None = None,
) -> SilenceVerificationResult:
    diagnostics: dict[str, Any] = {
        "stage": stage,
        "reason": reason,
        "elapsed_ms": round((time.monotonic() - started) * 1000),
    }
    if extra:
        diagnostics.update(extra)
    return SilenceVerificationResult("unavailable", start_sec, end_sec, diagnostics)


def verify_acoustic_boundaries(
    video_id_or_url: str,
    start_sec: float,
    end_sec: float,
    *,
    prepared: AudioPreparationResult | None = None,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    cancel_check: CancelCheck | None = None,
    ffmpeg_bin: str = "ffmpeg",
) -> SilenceVerificationResult:
    """Verify and adjust one clip to measured quiet edge intervals.

    ``prepared`` may be produced concurrently with transcript selection. On any
    resolution, decoding, cancellation, timeout, or silence failure, this
    function returns the original range with ``status='unavailable'``.
    """

    started = time.monotonic()
    try:
        original_start = float(start_sec)
        original_end = float(end_sec)
    except (TypeError, ValueError, OverflowError):
        return _unavailable(0.0, 0.0, stage="input", reason="invalid_range", started=started)
    if (
        not math.isfinite(original_start)
        or not math.isfinite(original_end)
        or original_start < 0
        or original_end <= original_start
        or not math.isfinite(timeout_sec)
        or timeout_sec <= 0
    ):
        return _unavailable(
            original_start, original_end, stage="input", reason="invalid_range", started=started
        )
    deadline = started + timeout_sec
    if _is_cancelled(cancel_check):
        return _unavailable(
            original_start, original_end, stage="verify", reason="cancelled", started=started
        )

    try:
        if prepared is None:
            source = _prepare_audio_source(
                video_id_or_url, deadline=deadline, cancel_check=cancel_check
            )
            preparation_diagnostics: Mapping[str, Any] = {}
        elif not prepared.ready or prepared.source is None:
            return _unavailable(
                original_start,
                original_end,
                stage=str(prepared.diagnostics.get("stage") or "resolve"),
                reason=str(prepared.diagnostics.get("reason") or "media_unavailable"),
                started=started,
            )
        else:
            source = prepared.source
            preparation_diagnostics = prepared.diagnostics

        start_window = max(0.0, original_start - EDGE_WINDOW_SEC / 2.0)
        end_window = max(0.0, original_end - EDGE_WINDOW_SEC / 2.0)
        with tempfile.TemporaryDirectory(prefix="reelai_silence_") as temp_dir:
            edge_paths = {
                "start": Path(temp_dir) / "start.wav",
                "end": Path(temp_dir) / "end.wav",
            }
            windows = {"start": start_window, "end": end_window}
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="silence-edge") as pool:
                futures = {
                    pool.submit(
                        _decode_window,
                        source,
                        window_start_sec=window_start,
                        window_duration_sec=EDGE_WINDOW_SEC,
                        output_path=edge_paths[name],
                        ffmpeg_bin=ffmpeg_bin,
                        deadline=deadline,
                        cancel_check=cancel_check,
                    ): name
                    for name, window_start in windows.items()
                }
                for future in as_completed(futures):
                    future.result()

            start_intervals = _quiet_intervals(
                edge_paths["start"],
                absolute_start_sec=start_window,
                threshold_dbfs=QUIET_THRESHOLD_DBFS,
                min_quiet_ms=MIN_QUIET_MS,
            )
            end_intervals = _quiet_intervals(
                edge_paths["end"],
                absolute_start_sec=end_window,
                threshold_dbfs=QUIET_THRESHOLD_DBFS,
                min_quiet_ms=MIN_QUIET_MS,
            )
            start_quiet = _pick_start_interval(start_intervals, original_start)
            end_quiet = _pick_end_interval(end_intervals, original_end)
            if start_quiet is None:
                return _unavailable(
                    original_start,
                    original_end,
                    stage="analyze",
                    reason="start_silence_not_found",
                    started=started,
                )
            if end_quiet is None:
                return _unavailable(
                    original_start,
                    original_end,
                    stage="analyze",
                    reason="end_silence_not_found",
                    started=started,
                )

            adjusted_start = max(
                max(0.0, original_start - START_CUSHION_MS / 1000.0),
                start_quiet.start_sec,
                start_quiet.end_sec - START_CUSHION_MS / 1000.0,
            )
            adjusted_end = min(
                original_end + END_CUSHION_MS / 1000.0,
                end_quiet.end_sec,
                end_quiet.start_sec + END_CUSHION_MS / 1000.0,
            )
            if (
                start_quiet.end_sec - adjusted_start + 1e-9
                < START_CUSHION_MS / 1000.0
            ):
                return _unavailable(
                    original_start,
                    original_end,
                    stage="analyze",
                    reason="start_cushion_outside_selected_range",
                    started=started,
                )
            if (
                adjusted_end - end_quiet.start_sec + 1e-9
                < END_CUSHION_MS / 1000.0
            ):
                return _unavailable(
                    original_start,
                    original_end,
                    stage="analyze",
                    reason="end_cushion_outside_selected_range",
                    started=started,
                )
            if adjusted_end <= adjusted_start:
                return _unavailable(
                    original_start,
                    original_end,
                    stage="analyze",
                    reason="adjusted_range_invalid",
                    started=started,
                )
    except _Unavailable as exc:
        return _unavailable(
            original_start,
            original_end,
            stage=exc.stage,
            reason=exc.reason,
            started=started,
        )
    except Exception:
        return _unavailable(
            original_start,
            original_end,
            stage="verify",
            reason="unexpected_failure",
            started=started,
        )

    diagnostics = {
        "format_id": source.format_id,
        "threshold_dbfs": QUIET_THRESHOLD_DBFS,
        "min_quiet_ms": MIN_QUIET_MS,
        "start_cushion_ms": START_CUSHION_MS,
        "end_cushion_ms": END_CUSHION_MS,
        "start_window": [round(start_window, 3), round(start_window + EDGE_WINDOW_SEC, 3)],
        "end_window": [round(end_window, 3), round(end_window + EDGE_WINDOW_SEC, 3)],
        "start_quiet": [round(start_quiet.start_sec, 3), round(start_quiet.end_sec, 3)],
        "end_quiet": [round(end_quiet.start_sec, 3), round(end_quiet.end_sec, 3)],
        "start_shift_sec": round(adjusted_start - original_start, 3),
        "end_shift_sec": round(adjusted_end - original_end, 3),
        "elapsed_ms": round((time.monotonic() - started) * 1000),
    }
    if preparation_diagnostics:
        diagnostics["prepare_elapsed_ms"] = preparation_diagnostics.get("elapsed_ms")
    return SilenceVerificationResult(
        "verified", round(adjusted_start, 3), round(adjusted_end, 3), diagnostics
    )


__all__ = [
    "AudioPreparationResult",
    "PreparedAudioSource",
    "SilenceVerificationResult",
    "prepare_audio_source",
    "verify_acoustic_boundaries",
]
