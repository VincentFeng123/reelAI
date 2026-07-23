"""Progressive, energy-only acoustic verification for hosted YouTube clips.

The production selector already supplies complete transcript cue boundaries.
This module performs the smaller final check that captions cannot provide: it
seeks short audio windows outward from each required speech edge and proves that
each cut lands in silence. It never downloads or transcribes the full source and
every failure is returned as ``unavailable`` so callers can keep searching for
another clip.
"""
from __future__ import annotations

import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import wave
from array import array
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, Mapping, Sequence
from urllib.parse import parse_qs, urlparse

from backend.app.config import get_settings
from . import lexical_timing


CancelCheck = Callable[[], bool]
VerificationStatus = Literal["verified", "context_aligned", "unavailable"]
PreparationStatus = Literal["ready", "unavailable"]

DEFAULT_TIMEOUT_SEC = 20.0
DEEP_PHASE_TIMEOUT_SEC = 50.0
DEFAULT_PREPARE_TIMEOUT_SEC = 24.0
EDGE_WINDOW_SEC = 6.0
QUIET_THRESHOLD_DBFS = -38.0
MIN_QUIET_MS = 100
START_CUSHION_MS = 100
END_CUSHION_MS = 100
HANDOFF_OBSERVATION_HALO_SEC = 1.0
HANDOFF_TIMESTAMP_TOLERANCE_SEC = 0.05
_FRAME_MS = 10
_FRAME_SEC = _FRAME_MS / 1000.0
_YT_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
_YT_HOSTS = frozenset({"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"})
_MAX_CONCURRENT_DECODES = 4
_MAX_SOURCE_CONCURRENT_DECODES = 2
_REFRESHABLE_MEDIA_FAILURES = frozenset(
    {
        "media_http_401",
        "media_http_403",
        "media_http_404",
        "media_http_410",
        "media_http_429",
    }
)
_TRANSIENT_DECODE_FAILURES = frozenset(
    {"media_http_5xx", "media_network_error", "process_failed"}
)
_decode_slots = threading.BoundedSemaphore(_MAX_CONCURRENT_DECODES)


class _SourceDecodeState:
    def __init__(self) -> None:
        self.slots = threading.BoundedSemaphore(_MAX_SOURCE_CONCURRENT_DECODES)
        self.condition = threading.Condition()
        self.active_source: PreparedAudioSource | None = None
        self.generation = 0
        self.refresh_callback: (
            Callable[[float, CancelCheck | None], PreparedAudioSource] | None
        ) = None
        self.refreshing = False
        self.refresh_used = False
        self.terminal_reason = ""
        self.terminal_attempt_reasons: tuple[str, ...] = ()

    def configure_initial(self, source: PreparedAudioSource) -> None:
        with self.condition:
            if self.active_source is None:
                self.active_source = source

    def configure_refresh(
        self,
        callback: Callable[[float, CancelCheck | None], PreparedAudioSource],
    ) -> None:
        with self.condition:
            self.refresh_callback = callback

    def failure(self) -> tuple[str, tuple[str, ...]]:
        with self.condition:
            return self.terminal_reason, self.terminal_attempt_reasons

    def trip(self, reason: str) -> None:
        with self.condition:
            if not self.terminal_reason:
                self.terminal_reason = reason
            self.condition.notify_all()

    def snapshot(self) -> tuple[PreparedAudioSource, int]:
        with self.condition:
            if self.terminal_reason:
                raise _Unavailable(
                    "decode",
                    self.terminal_reason,
                    attempt_reasons=self.terminal_attempt_reasons,
                )
            if self.active_source is None:
                raise _Unavailable("decode", "media_route_unavailable")
            return self.active_source, self.generation

    def refresh_after_failure(
        self,
        *,
        failed_generation: int,
        reason: str,
        deadline: float,
        cancel_check: CancelCheck | None,
    ) -> None:
        callback: Callable[[float, CancelCheck | None], PreparedAudioSource] | None
        while True:
            with self.condition:
                if self.terminal_reason:
                    raise _Unavailable(
                        "decode",
                        self.terminal_reason,
                        attempt_reasons=self.terminal_attempt_reasons,
                    )
                if self.generation != failed_generation:
                    return
                if not self.refreshing:
                    if self.refresh_used or self.refresh_callback is None:
                        self.terminal_reason = reason
                        self.condition.notify_all()
                        raise _Unavailable("decode", reason)
                    self.refresh_used = True
                    self.refreshing = True
                    callback = self.refresh_callback
                    break
                if _is_cancelled(cancel_check):
                    raise _Unavailable("decode", "cancelled")
                remaining = _remaining(deadline, "decode")
                self.condition.wait(timeout=min(0.05, remaining))

        try:
            refreshed = callback(deadline, cancel_check)
        except _Unavailable as exc:
            with self.condition:
                self.refreshing = False
                self.terminal_reason = reason
                self.terminal_attempt_reasons = tuple(exc.attempt_reasons)
                self.condition.notify_all()
            raise _Unavailable(
                "decode",
                reason,
                attempt_reasons=exc.attempt_reasons,
            ) from None
        except Exception:
            with self.condition:
                self.refreshing = False
                self.terminal_reason = reason
                self.condition.notify_all()
            raise _Unavailable("decode", reason) from None

        with self.condition:
            self.active_source = refreshed
            self.generation += 1
            self.refreshing = False
            self.condition.notify_all()


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
    duration_sec: float | None = None
    lexical_words: tuple[lexical_timing.LexicalWord, ...] = field(
        default_factory=tuple,
        repr=False,
    )
    lexical_language: str = ""
    _decode_state: _SourceDecodeState = field(
        default_factory=_SourceDecodeState,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        self._decode_state.configure_initial(self)


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

    @property
    def usable(self) -> bool:
        return self.status in {"verified", "context_aligned"}


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
    def __init__(
        self,
        stage: str,
        reason: str,
        *,
        attempt_reasons: Sequence[str] = (),
    ) -> None:
        super().__init__(reason)
        self.stage = stage
        self.reason = reason
        self.attempt_reasons = tuple(attempt_reasons)


def persisted_boundary_is_verified(context: object) -> bool:
    """Accept only rows with explicit strict energy-threshold evidence.

    Measured energy verification has always persisted its threshold. Missing
    or malformed threshold evidence therefore fails closed rather than letting
    a forged/partial legacy row bypass the acoustic gate.
    """
    if not isinstance(context, Mapping):
        return False
    diagnostics = context.get("boundary_diagnostics")
    if (
        str(context.get("boundary_status") or "").strip().lower() != "verified"
        or not isinstance(diagnostics, Mapping)
        or diagnostics.get("acoustic_verified") is not True
    ):
        return False

    acoustic = diagnostics.get("acoustic")
    if not isinstance(acoustic, Mapping):
        return False
    detail_sets = [acoustic]
    saw_threshold = False
    for details in detail_sets:
        adaptive = details.get("adaptive_quiet")
        if adaptive is True or (
            isinstance(adaptive, str)
            and adaptive.strip().lower() in {"1", "true", "yes", "on"}
        ):
            return False
        for key in ("threshold_dbfs", "start_threshold_dbfs", "end_threshold_dbfs"):
            if key not in details:
                continue
            saw_threshold = True
            value = details.get(key)
            if isinstance(value, bool):
                return False
            try:
                threshold = float(value)
            except (TypeError, ValueError, OverflowError):
                return False
            if not math.isfinite(threshold) or threshold > QUIET_THRESHOLD_DBFS:
                return False
    return saw_threshold


def _diagnostic_range(value: object) -> tuple[float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        return None
    try:
        start, end = (float(item) for item in value)
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        not math.isfinite(start)
        or not math.isfinite(end)
        or start < 0.0
        or end <= start
    ):
        return None
    return start, end


def persisted_boundary_is_usable(
    context: object,
    *,
    t_start: object | None = None,
    t_end: object | None = None,
) -> bool:
    """Accept strict silence or an explicit, bounded transcript fallback.

    Context-aligned rows remain distinguishable from acoustically verified
    rows. Current rows and the immediately preceding history contract remain
    viewable only when their final range contains every required word while
    staying inside the already validated semantic corridor. Fresh-generation
    reuse is gated separately by the current request and selector contracts.
    """

    if not isinstance(context, Mapping):
        return False
    persisted_range: tuple[float, float] | None = None
    if t_start is not None or t_end is not None:
        persisted_range = _diagnostic_range([t_start, t_end])
        if persisted_range is None:
            return False
    if persisted_boundary_is_verified(context):
        if persisted_range is None:
            return True
        diagnostics = context.get("boundary_diagnostics")
        if not isinstance(diagnostics, Mapping):
            return False
        is_current_contract = bool(
            str(context.get("selection_contract_version") or "").strip()
            == "quality_silence_v41"
        )
        final = _diagnostic_range(diagnostics.get("final_range"))
        if final is not None:
            final_matches = bool(
                abs(persisted_range[0] - final[0]) <= 1e-3
                and abs(persisted_range[1] - final[1]) <= 1e-3
            )
            if not final_matches or not is_current_contract:
                return final_matches
        acoustic = diagnostics.get("acoustic")
        if not isinstance(acoustic, Mapping):
            return False
        start_quiet = _diagnostic_range(acoustic.get("start_quiet"))
        end_quiet = _diagnostic_range(acoustic.get("end_quiet"))
        if start_quiet is None or end_quiet is None:
            return not is_current_contract
        return bool(
            start_quiet[0] - 1e-3
            <= persisted_range[0]
            <= start_quiet[1] + 1e-3
            and end_quiet[0] - 1e-3
            <= persisted_range[1]
            <= end_quiet[1] + 1e-3
        )
    diagnostics = context.get("boundary_diagnostics")
    captions = context.get("selection_caption_cues")
    if (
        str(context.get("selection_contract_version") or "").strip()
        not in {
            "quality_silence_v21",
            "quality_silence_v22",
            "quality_silence_v23",
            "quality_silence_v24",
            "quality_silence_v25",
            "quality_silence_v26",
            "quality_silence_v27",
            "quality_silence_v28",
            "quality_silence_v29",
            "quality_silence_v30",
            "quality_silence_v31",
            "quality_silence_v32",
            "quality_silence_v34",
            "quality_silence_v36",
            "quality_silence_v37",
            "quality_silence_v38",
            "quality_silence_v39",
            "quality_silence_v40",
            "quality_silence_v41",
        }
        or str(context.get("boundary_status") or "").strip().lower()
        != "context_aligned"
        or context.get("speech_corridor_verified") is not True
        or not isinstance(captions, list)
        or not captions
        or not isinstance(diagnostics, Mapping)
        or diagnostics.get("method") != "transcript_context"
        or diagnostics.get("context_aligned") is not True
        or diagnostics.get("acoustic_verified") is not False
    ):
        return False
    transcript = diagnostics.get("transcript")
    if (
        not isinstance(transcript, Mapping)
        or transcript.get("context_aligned") is not True
        or not str(transcript.get("stage") or "").strip()
        or not str(transcript.get("reason") or "").strip()
    ):
        return False
    required = _diagnostic_range(transcript.get("required_speech_range"))
    semantic = _diagnostic_range(transcript.get("semantic_range"))
    final = _diagnostic_range(transcript.get("final_range"))
    if required is None or semantic is None or final is None:
        return False
    if persisted_range is not None:
        if (
            abs(persisted_range[0] - final[0]) > 1e-3
            or abs(persisted_range[1] - final[1]) > 1e-3
        ):
            return False
    if not (
        semantic[0] <= final[0] <= required[0]
        and required[1] <= final[1] <= semantic[1]
    ):
        return False
    caption_ranges: list[tuple[float, float]] = []
    for caption in captions:
        if (
            not isinstance(caption, Mapping)
            or not str(caption.get("text") or "").strip()
        ):
            return False
        caption_range = _diagnostic_range(
            [caption.get("start"), caption.get("end")]
        )
        if caption_range is None:
            return False
        if caption_ranges and (
            caption_range[0] + 1e-3 < caption_ranges[-1][0]
            or caption_range[1] + 1e-3 < caption_ranges[-1][1]
        ):
            return False
        caption_ranges.append(caption_range)
    caption_start = min(item[0] for item in caption_ranges)
    caption_end = max(item[1] for item in caption_ranges)
    required_start_covered = any(
        caption_start_sec <= required[0] + 1e-3
        and caption_end_sec > required[0] + 1e-3
        for caption_start_sec, caption_end_sec in caption_ranges
    )
    required_end_covered = any(
        caption_start_sec < required[1] - 1e-3
        and caption_end_sec + 1e-3 >= required[1]
        for caption_start_sec, caption_end_sec in caption_ranges
    )
    return bool(
        caption_start <= final[0] + 1e-3
        and caption_end + 1e-3 >= final[1]
        and caption_start <= required[0] + 1e-3
        and caption_end + 1e-3 >= required[1]
        and required_start_covered
        and required_end_covered
    )


def _is_cancelled(cancel_check: CancelCheck | None) -> bool:
    if cancel_check is None:
        return False
    try:
        return bool(cancel_check())
    except Exception:
        return True


def _process_failure_reason(stage: str, stderr: bytes) -> str:
    """Classify resolver failures without retaining sensitive command output."""
    decoded = stderr.decode("utf-8", errors="ignore")
    if stage == "decode":
        message = decoded.casefold()
        http_status = re.search(
            r"(?:http (?:error )?|server returned\s+)(401|403|404|410|429|5\d\d)\b",
            message,
        )
        if http_status:
            status = int(http_status.group(1))
            return (
                f"media_http_{status}"
                if status in {401, 403, 404, 410, 429}
                else "media_http_5xx"
            )
        if any(
            marker in message
            for marker in (
                "connection refused",
                "connection reset",
                "connection timed out",
                "failed to resolve",
                "network is unreachable",
                "no route to host",
                "temporary failure in name resolution",
            )
        ):
            return "media_network_error"
        return "process_failed"
    if stage != "resolve":
        return "process_failed"
    terminal_errors = [
        line for line in decoded.splitlines() if line.lstrip().casefold().startswith("error:")
    ]
    message = (terminal_errors[-1] if terminal_errors else decoded).casefold()
    if "sign in to confirm" in message or "not a bot" in message:
        return "youtube_bot_challenge"
    if "proxy" in message and any(
        marker in message
        for marker in ("unable", "failed", "refused", "timed out", "tunnel", "407")
    ):
        return "proxy_failed"
    if any(
        marker in message
        for marker in (
            "requested format is not available",
            "no video formats found",
            "no formats found",
        )
    ):
        return "format_unavailable"
    if ("remote component" in message or "ejs" in message) and "fail" in message:
        return "component_failed"
    return "process_failed"


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


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    """Kill yt-dlp and any Deno child without an unbounded pipe wait."""
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except (OSError, ValueError):
        try:
            process.kill()
        except OSError:
            pass
    try:
        process.communicate(timeout=1.0)
    except (subprocess.TimeoutExpired, OSError, ValueError):
        for stream in (process.stdout, process.stderr):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass


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
            start_new_session=os.name == "posix",
        )
    except (OSError, ValueError):
        raise _Unavailable(stage, "process_start_failed") from None

    while True:
        if _is_cancelled(cancel_check):
            _terminate_process(process)
            raise _Unavailable(stage, "cancelled")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _terminate_process(process)
            raise _Unavailable(stage, "deadline_exceeded")
        wait_for = min(0.1, remaining)
        try:
            stdout, stderr = process.communicate(timeout=wait_for)
            break
        except subprocess.TimeoutExpired:
            continue
        except Exception:
            _terminate_process(process)
            raise _Unavailable(stage, "process_failed") from None

    if process.returncode != 0:
        raise _Unavailable(stage, _process_failure_reason(stage, stderr))
    return stdout, stderr


def _proxy_urls() -> list[str]:
    proxies = str(get_settings().proxy_urls or "")
    return list(dict.fromkeys(part.strip() for part in proxies.split(",") if part.strip()))


def _proxy_url() -> str:
    return next(iter(_proxy_urls()), "")


def _yt_dlp_cookie_args() -> list[str]:
    cookie_file = str(os.environ.get("YT_COOKIES_FILE") or "").strip()
    if cookie_file and os.path.isfile(cookie_file):
        return ["--cookies", cookie_file]
    browser = str(os.environ.get("YT_COOKIES_FROM_BROWSER") or "").strip()
    return ["--cookies-from-browser", browser] if browser else []


def _yt_dlp_command(
    watch_url: str,
    *,
    proxy_url: str | None = None,
    player_client: str = "default",
    use_cookies: bool = True,
) -> list[str]:
    settings = get_settings()
    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--ignore-config",
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
        "--no-remote-components",
        "--format",
        "worstaudio[acodec!=none][vcodec=none]",
    ]
    if use_cookies:
        command.extend(_yt_dlp_cookie_args())
    proxy = _proxy_url() if proxy_url is None else proxy_url
    if proxy:
        command.extend(["--proxy", proxy])
    provider_url = str(settings.ytdlp_pot_provider_url or "").strip()
    if provider_url:
        command.extend(
            ["--extractor-args", f"youtubepot-bgutilhttp:base_url={provider_url}"]
        )
    if player_client != "default":
        command.extend(
            [
                "--extractor-args",
                f"youtube:player_client={player_client}",
            ]
        )
    command.append(watch_url)
    return command


def _audio_bitrate(entry: Mapping[str, Any]) -> float | None:
    for key in ("tbr", "abr"):
        try:
            bitrate = float(entry.get(key))
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(bitrate) and bitrate > 0.0:
            return bitrate
    return None


def _is_audio_only_entry(entry: object) -> bool:
    return bool(
        isinstance(entry, Mapping)
        and isinstance(entry.get("url"), str)
        and str(entry.get("url") or "").strip()
        and str(entry.get("acodec") or "none").casefold() != "none"
        and str(entry.get("vcodec") or "none").casefold() == "none"
        and _audio_bitrate(entry) is not None
    )


def _audio_entry(info: Mapping[str, Any]) -> Mapping[str, Any]:
    requested = info.get("requested_downloads")
    if isinstance(requested, list):
        requested_audio = [entry for entry in requested if _is_audio_only_entry(entry)]
        if requested_audio:
            return min(
                requested_audio,
                key=lambda entry: _audio_bitrate(entry) or math.inf,
            )
    if _is_audio_only_entry(info):
        return info
    formats = info.get("formats")
    if not isinstance(formats, list):
        return {}
    fallback_audio = [entry for entry in formats if _is_audio_only_entry(entry)]
    if not fallback_audio:
        return {}
    return min(fallback_audio, key=lambda entry: _audio_bitrate(entry) or math.inf)


def _prepare_audio_source(
    video_id_or_url: str,
    *,
    deadline: float,
    cancel_check: CancelCheck | None,
    language: str = "en",
    excluded_attempts: frozenset[tuple[str, str, bool]] = frozenset(),
    configure_refresh: bool = True,
    fetch_lexical: bool = True,
) -> PreparedAudioSource:
    watch_url = _canonical_watch_url(video_id_or_url)
    if watch_url is None:
        raise _Unavailable("resolve", "invalid_youtube_source")
    configured_routes = _proxy_urls()[:3]
    provider_configured = bool(
        str(get_settings().ytdlp_pot_provider_url or "").strip()
    )
    if not configured_routes:
        default_routes = [""]
    elif len(configured_routes) == 1:
        default_routes = [configured_routes[0], ""]
    else:
        default_routes = configured_routes[:2]
    primary_client = "mweb" if provider_configured else "default"
    cookies_available = bool(_yt_dlp_cookie_args())
    primary_uses_cookies = bool(not provider_configured and cookies_available)
    attempts: list[tuple[str, str, bool]] = [
        (route, primary_client, primary_uses_cookies)
        for route in default_routes
    ]
    fallback_profiles = (
        (("default", cookies_available), ("web_embedded", False))
        if provider_configured
        else (("web_embedded", False),)
    )
    for profile, use_cookies in fallback_profiles:
        if len(attempts) >= 3:
            break
        # Public educational videos use the recommended cookieless mweb+POT
        # route first. Keep one bounded cookie-backed default fallback for
        # account-gated media, plus a cookieless embedded-client fallback when
        # the three-attempt cap leaves room.
        attempts.append(("", profile, use_cookies))
    attempts = [
        attempt for attempt in attempts[:3] if attempt not in excluded_attempts
    ]
    if not attempts:
        raise _Unavailable("resolve", "media_route_exhausted")
    attempt_reasons: list[str] = []
    for attempt_index, (proxy, player_client, use_cookies) in enumerate(attempts):
        try:
            remaining = _remaining(deadline, "resolve")
            attempts_left = len(attempts) - attempt_index
            attempt_deadline = time.monotonic() + (remaining / max(1, attempts_left))
            stdout, _ = _run_command(
                _yt_dlp_command(
                    watch_url,
                    proxy_url=proxy,
                    player_client=player_client,
                    use_cookies=use_cookies,
                ),
                deadline=attempt_deadline,
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
            raw_duration = info.get("duration") or entry.get("duration")
            try:
                duration_sec = float(raw_duration)
            except (TypeError, ValueError, OverflowError):
                duration_sec = None
            if duration_sec is not None and (
                not math.isfinite(duration_sec) or duration_sec <= 0.0
            ):
                duration_sec = None
            raw_headers = entry.get("http_headers") or info.get("http_headers") or {}
            headers = {
                str(key): str(value)
                for key, value in (
                    raw_headers.items() if isinstance(raw_headers, Mapping) else ()
                )
                if re.fullmatch(r"[A-Za-z0-9-]+", str(key))
                and "\r" not in str(value)
                and "\n" not in str(value)
                and str(key).lower() not in {"host", "content-length", "range"}
            }
            lexical_words: tuple[lexical_timing.LexicalWord, ...] = ()
            lexical_language = ""
            if fetch_lexical:
                tracks = lexical_timing.select_original_json3_tracks(
                    info,
                    expected_language=language,
                )
                for track in tracks:
                    lexical_deadline = min(
                        deadline,
                        time.monotonic() + lexical_timing.MAX_FETCH_TIMEOUT_SEC,
                    )
                    try:
                        lexical_words = lexical_timing.fetch_json3_words(
                            track,
                            headers=headers,
                            proxy_url=proxy,
                            deadline=lexical_deadline,
                            cancel_check=cancel_check,
                        )
                    except Exception:
                        if _is_cancelled(cancel_check):
                            raise _Unavailable("resolve", "cancelled") from None
                        lexical_words = ()
                    if lexical_words:
                        lexical_language = track.language
                        break
            prepared = PreparedAudioSource(
                url=media_url,
                headers=headers,
                proxy_url=proxy,
                format_id=str(entry.get("format_id") or info.get("format_id") or ""),
                duration_sec=duration_sec,
                lexical_words=lexical_words,
                lexical_language=lexical_language,
            )
            if configure_refresh:
                consumed_attempts = frozenset(
                    [*excluded_attempts, *attempts[: attempt_index + 1]]
                )

                def refresh_source(
                    refresh_deadline: float,
                    refresh_cancel_check: CancelCheck | None,
                ) -> PreparedAudioSource:
                    return _prepare_audio_source(
                        video_id_or_url,
                        deadline=refresh_deadline,
                        cancel_check=refresh_cancel_check,
                        language=language,
                        excluded_attempts=consumed_attempts,
                        configure_refresh=False,
                        fetch_lexical=False,
                    )

                prepared._decode_state.configure_refresh(refresh_source)
            return prepared
        except _Unavailable as exc:
            if exc.reason in {"cancelled", "invalid_youtube_source"}:
                raise
            reason = exc.reason
            if reason == "deadline_exceeded":
                if deadline - time.monotonic() <= 0:
                    reason = "deadline_exceeded"
                else:
                    reason = "attempt_timeout"
            route = "proxy" if proxy else "direct"
            auth = "cookies" if use_cookies else "cookieless"
            attempt_reasons.append(f"{route}:{player_client}:{auth}:{reason}")
            if reason == "deadline_exceeded":
                break

    reason_priority = (
        "youtube_bot_challenge",
        "proxy_failed",
        "format_unavailable",
        "component_failed",
        "deadline_exceeded",
        "attempt_timeout",
        "media_route_exhausted",
        "process_failed",
        "audio_url_missing",
        "invalid_ytdlp_response",
    )
    final_reason = next(
        (
            reason
            for reason in reason_priority
            if any(item.endswith(f":{reason}") for item in attempt_reasons)
        ),
        "process_failed",
    )
    raise _Unavailable(
        "resolve",
        final_reason,
        attempt_reasons=attempt_reasons,
    )


def prepare_audio_source(
    video_id_or_url: str,
    *,
    timeout_sec: float = DEFAULT_PREPARE_TIMEOUT_SEC,
    cancel_check: CancelCheck | None = None,
    language: str = "en",
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
            language=language,
        )
    except _Unavailable as exc:
        return AudioPreparationResult(
            "unavailable",
            diagnostics={
                "stage": exc.stage,
                "reason": exc.reason,
                "attempt_reasons": list(exc.attempt_reasons),
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
            "lexical_word_count": len(source.lexical_words),
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
    max_duration_sec: float = EDGE_WINDOW_SEC,
) -> None:
    source_acquired = False
    while not source_acquired:
        terminal_reason, terminal_attempt_reasons = source._decode_state.failure()
        if terminal_reason:
            raise _Unavailable(
                "decode",
                terminal_reason,
                attempt_reasons=terminal_attempt_reasons,
            )
        if _is_cancelled(cancel_check):
            raise _Unavailable("decode", "cancelled")
        remaining = _remaining(deadline, "decode")
        source_acquired = source._decode_state.slots.acquire(
            timeout=min(0.1, remaining)
        )
    try:
        duration = min(max_duration_sec, max(0.01, window_duration_sec))
        transient_failures = 0
        transient_generation = -1
        while True:
            global_acquired = False
            generation: int | None = None
            failure: _Unavailable | None = None
            try:
                while not global_acquired:
                    terminal_reason, terminal_attempt_reasons = (
                        source._decode_state.failure()
                    )
                    if terminal_reason:
                        raise _Unavailable(
                            "decode",
                            terminal_reason,
                            attempt_reasons=terminal_attempt_reasons,
                        )
                    if _is_cancelled(cancel_check):
                        raise _Unavailable("decode", "cancelled")
                    remaining = _remaining(deadline, "decode")
                    global_acquired = _decode_slots.acquire(
                        timeout=min(0.1, remaining)
                    )
                active_source, generation = source._decode_state.snapshot()
                if generation != transient_generation:
                    transient_failures = 0
                    transient_generation = generation
                if _is_cancelled(cancel_check):
                    raise _Unavailable("decode", "cancelled")
                command = [
                    ffmpeg_bin,
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                ]
                if active_source.proxy_url.startswith(("http://", "https://")):
                    command.extend(["-http_proxy", active_source.proxy_url])
                if active_source.headers:
                    command.extend(["-headers", _ffmpeg_headers(active_source.headers)])
                command.extend(
                    [
                        "-rw_timeout",
                        str(
                            max(
                                1,
                                int(
                                    min(8.0, _remaining(deadline, "decode"))
                                    * 1_000_000
                                ),
                            )
                        ),
                        "-ss",
                        f"{max(0.0, window_start_sec):.3f}",
                        "-threads",
                        "1",
                        "-i",
                        active_source.url,
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
                _run_command(
                    command,
                    deadline=deadline,
                    cancel_check=cancel_check,
                    stage="decode",
                )
            except _Unavailable as exc:
                failure = exc
            finally:
                if global_acquired:
                    _decode_slots.release()
            if failure is None:
                break
            output_path.unlink(missing_ok=True)
            if failure.stage != "decode":
                raise failure
            if failure.reason in _REFRESHABLE_MEDIA_FAILURES:
                if generation is None:
                    raise failure
                source._decode_state.refresh_after_failure(
                    failed_generation=generation,
                    reason=failure.reason,
                    deadline=deadline,
                    cancel_check=cancel_check,
                )
                continue
            if failure.reason in _TRANSIENT_DECODE_FAILURES:
                transient_failures += 1
                if transient_failures < 2:
                    continue
                source._decode_state.trip(failure.reason)
            raise failure
    finally:
        if source_acquired:
            source._decode_state.slots.release()
    if not output_path.is_file() or output_path.stat().st_size <= 44:
        raise _Unavailable("decode", "empty_audio_window")


@contextmanager
def decode_audio_window(
    source: PreparedAudioSource,
    *,
    window_start_sec: float,
    window_end_sec: float,
    max_duration_sec: float,
    timeout_sec: float,
    cancel_check: CancelCheck | None = None,
    ffmpeg_bin: str = "ffmpeg",
) -> Iterator[Path]:
    """Yield one bounded 16 kHz mono PCM WAV and remove it on exit."""

    try:
        start = float(window_start_sec)
        end = float(window_end_sec)
        maximum = float(max_duration_sec)
        timeout = float(timeout_sec)
    except (TypeError, ValueError, OverflowError):
        raise _Unavailable("decode", "invalid_audio_window") from None
    if (
        not all(math.isfinite(value) for value in (start, end, maximum, timeout))
        or start < 0
        or end <= start
        or maximum <= 0
        or end - start > maximum
        or timeout <= 0
    ):
        raise _Unavailable("decode", "invalid_audio_window")
    source_duration = source.duration_sec
    if (
        source_duration is not None
        and math.isfinite(source_duration)
        and end > source_duration + HANDOFF_TIMESTAMP_TOLERANCE_SEC
    ):
        raise _Unavailable("decode", "invalid_audio_window")

    with tempfile.TemporaryDirectory(prefix="reelai_audio_window_") as temp_dir:
        output_path = Path(temp_dir) / "window.wav"
        _decode_window(
            source,
            window_start_sec=start,
            window_duration_sec=end - start,
            output_path=output_path,
            ffmpeg_bin=ffmpeg_bin,
            deadline=time.monotonic() + timeout,
            cancel_check=cancel_check,
            max_duration_sec=maximum,
        )
        yield output_path


def _quiet_intervals(
    wav_path: Path,
    *,
    absolute_start_sec: float,
    threshold_dbfs: float,
    min_quiet_ms: int,
    include_boundary_fragments: bool = False,
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
            if (
                index - run_start >= minimum_frames
                or (
                    include_boundary_fragments
                    and (run_start == 0 or index == len(quiet))
                )
            ):
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
    intervals: Sequence[_QuietInterval],
    rough_start: float,
    *,
    handoff_band: tuple[float, float] | None = None,
    require_two_sided: bool = False,
) -> _QuietInterval | None:
    candidates = [
        interval
        for interval in intervals
        if interval.followed_by_sound
        and (not require_two_sided or interval.preceded_by_sound)
        and (
            interval.start_sec <= rough_start
            if handoff_band is None
            else (
                interval.start_sec <= handoff_band[0] + _FRAME_SEC
                and interval.end_sec
                >= handoff_band[1] - HANDOFF_TIMESTAMP_TOLERANCE_SEC
            )
        )
        and interval.duration_sec + 1e-9
        >= max(MIN_QUIET_MS, START_CUSHION_MS) / 1000.0
    ]
    if not candidates:
        return None

    def priority(interval: _QuietInterval) -> tuple[int, float]:
        if interval.start_sec <= rough_start <= interval.end_sec:
            return 0, abs(interval.end_sec - rough_start)
        return 1, rough_start - interval.end_sec

    return min(candidates, key=priority)


def _pick_end_interval(
    intervals: Sequence[_QuietInterval],
    rough_end: float,
    *,
    allow_source_edge: bool = False,
    handoff_band: tuple[float, float] | None = None,
    require_two_sided: bool = False,
) -> _QuietInterval | None:
    candidates = [
        interval
        for interval in intervals
        if interval.preceded_by_sound
        and (not require_two_sided or interval.followed_by_sound)
        and (
            (
                interval.end_sec - _FRAME_SEC + 1e-9 >= rough_end
                if handoff_band is None
                else (
                    interval.start_sec
                    <= handoff_band[0] + HANDOFF_TIMESTAMP_TOLERANCE_SEC
                    and interval.end_sec
                    >= handoff_band[1] - HANDOFF_TIMESTAMP_TOLERANCE_SEC
                )
            )
            or (
                allow_source_edge
                and abs(interval.end_sec - rough_end) <= _FRAME_SEC + 1e-9
            )
        )
        and interval.duration_sec + 1e-9
        >= max(MIN_QUIET_MS / 1000.0, END_CUSHION_MS / 1000.0 + _FRAME_SEC)
    ]
    if not candidates:
        return None

    def priority(interval: _QuietInterval) -> tuple[int, float]:
        if interval.start_sec <= rough_end <= interval.end_sec:
            return 0, abs(interval.start_sec - rough_end)
        return 1, interval.start_sec - rough_end

    return min(candidates, key=priority)


@dataclass(frozen=True)
class _EdgeSearchResult:
    quiet: _QuietInterval | None
    windows: tuple[tuple[float, float], ...]


def _start_search_windows(
    required_start: float,
    *,
    search_start_limit: float,
    search_end_limit: float,
) -> Iterator[tuple[float, float]]:
    """Return nearest-first, non-overlapping windows extending backward."""
    half_window = EDGE_WINDOW_SEC / 2.0
    nearest_start = max(search_start_limit, required_start - half_window)
    nearest_end = min(search_end_limit, required_start + half_window)
    yield nearest_start, nearest_end
    cursor = nearest_start
    while cursor > search_start_limit + 1e-9:
        window_start = max(search_start_limit, cursor - EDGE_WINDOW_SEC)
        yield window_start, cursor
        cursor = window_start


def _end_search_windows(
    required_end: float,
    *,
    search_start_limit: float,
    search_end_limit: float,
) -> Iterator[tuple[float, float]]:
    """Return nearest-first, non-overlapping windows extending forward."""
    half_window = EDGE_WINDOW_SEC / 2.0
    nearest_start = max(search_start_limit, required_end - half_window)
    nearest_end = min(search_end_limit, required_end + half_window)
    yield nearest_start, nearest_end
    cursor = nearest_end
    while cursor < search_end_limit - 1e-9:
        window_end = min(search_end_limit, cursor + EDGE_WINDOW_SEC)
        yield cursor, window_end
        cursor = window_end


def _search_quiet(
    source: PreparedAudioSource,
    *,
    edge: Literal["start", "end"],
    required_boundary: float,
    search_start_limit: float,
    search_end_limit: float,
    allow_source_edge: bool = False,
    handoff_band: tuple[float, float] | None = None,
    require_two_sided: bool = False,
    temp_dir: Path,
    ffmpeg_bin: str,
    deadline: float,
    cancel_check: CancelCheck | None,
) -> _EdgeSearchResult:
    if edge == "start":
        windows = _start_search_windows(
            required_boundary,
            search_start_limit=search_start_limit,
            search_end_limit=search_end_limit,
        )
        picker = _pick_start_interval
    else:
        windows = _end_search_windows(
            required_boundary,
            search_start_limit=search_start_limit,
            search_end_limit=search_end_limit,
        )
        picker = _pick_end_interval

    searched: list[tuple[float, float]] = []
    adjacent_boundary_quiet: _QuietInterval | None = None
    for index, (window_start, window_end) in enumerate(windows):
        output_path = temp_dir / f"{edge}-{index}.wav"
        _decode_window(
            source,
            window_start_sec=window_start,
            window_duration_sec=window_end - window_start,
            output_path=output_path,
            ffmpeg_bin=ffmpeg_bin,
            deadline=deadline,
            cancel_check=cancel_check,
        )
        searched.append((window_start, window_end))
        intervals = _quiet_intervals(
            output_path,
            absolute_start_sec=window_start,
            threshold_dbfs=QUIET_THRESHOLD_DBFS,
            min_quiet_ms=MIN_QUIET_MS,
            include_boundary_fragments=True,
        )
        if index:
            seam_fragment = next(
                (
                    interval
                    for interval in intervals
                    if (
                        abs(interval.end_sec - window_end) <= _FRAME_SEC
                        if edge == "start"
                        else abs(interval.start_sec - window_start) <= _FRAME_SEC
                    )
                ),
                None,
            )
            if seam_fragment is not None and adjacent_boundary_quiet is not None:
                intervals.append(
                    _QuietInterval(
                        (
                            seam_fragment.start_sec
                            if edge == "start"
                            else adjacent_boundary_quiet.start_sec
                        ),
                        (
                            adjacent_boundary_quiet.end_sec
                            if edge == "start"
                            else seam_fragment.end_sec
                        ),
                        (
                            seam_fragment.preceded_by_sound
                            if edge == "start"
                            else adjacent_boundary_quiet.preceded_by_sound
                        ),
                        (
                            adjacent_boundary_quiet.followed_by_sound
                            if edge == "start"
                            else seam_fragment.followed_by_sound
                        ),
                    )
                )
            elif seam_fragment is not None:
                # The nearer window proves sound immediately across the seam.
                replacement = _QuietInterval(
                    seam_fragment.start_sec,
                    seam_fragment.end_sec,
                    (
                        seam_fragment.preceded_by_sound
                        if edge == "start"
                        else True
                    ),
                    (
                        True
                        if edge == "start"
                        else seam_fragment.followed_by_sound
                    ),
                )
                intervals = [
                    replacement if interval is seam_fragment else interval
                    for interval in intervals
                ]
        quiet = (
            picker(
                intervals,
                required_boundary,
                handoff_band=handoff_band,
                require_two_sided=require_two_sided,
            )
            if edge == "start"
            else _pick_end_interval(
                intervals,
                required_boundary,
                allow_source_edge=allow_source_edge,
                handoff_band=handoff_band,
                require_two_sided=require_two_sided,
            )
        )
        if quiet is not None:
            return _EdgeSearchResult(quiet, tuple(searched))
        boundary_fragments = [
            interval
            for interval in intervals
            if (
                abs(interval.start_sec - window_start) <= _FRAME_SEC
                if edge == "start"
                else abs(interval.end_sec - window_end) <= _FRAME_SEC
            )
        ]
        if handoff_band is not None:
            if edge != "start":
                return _EdgeSearchResult(None, tuple(searched))
            # Continue backward only while proving the same quiet run that
            # reaches the required speech onset. Sound at a seam ends the
            # search, so an earlier unrelated pause can never qualify.
            boundary_fragments = [
                interval
                for interval in boundary_fragments
                if (
                    interval.start_sec
                    <= handoff_band[0] + HANDOFF_TIMESTAMP_TOLERANCE_SEC
                    and interval.end_sec
                    >= handoff_band[1] - HANDOFF_TIMESTAMP_TOLERANCE_SEC
                )
            ]
            if not boundary_fragments:
                return _EdgeSearchResult(None, tuple(searched))
        adjacent_boundary_quiet = max(
            boundary_fragments,
            key=lambda interval: interval.duration_sec,
            default=None,
        )
    return _EdgeSearchResult(None, tuple(searched))


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
    search_start_limit_sec: float | None = None,
    search_end_limit_sec: float | None = None,
    require_speech_handoff: bool = False,
    require_start_speech_handoff: bool | None = None,
    require_end_speech_handoff: bool | None = None,
    require_start_two_sided: bool = False,
    require_end_two_sided: bool = False,
    prepared: AudioPreparationResult | None = None,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    cancel_check: CancelCheck | None = None,
    ffmpeg_bin: str = "ffmpeg",
) -> SilenceVerificationResult:
    """Verify and adjust one clip to measured quiet edge intervals.

    ``start_sec`` and ``end_sec`` are required speech boundaries. Ordinarily a
    verified start never moves later and a verified end never moves earlier.
    With ``require_speech_handoff``, both caption timestamps are treated as
    semantic fences: a one-second decode-only halo may complete a quiet run
    across each fence, but no later quiet run separated by sound can qualify.
    The per-edge handoff arguments override that legacy all-edges switch. The
    same observation-only halo gives two-sided lexical searches enough audio
    to prove surrounding sound. A start gap must begin on the selected side of
    its fence; an end gap may straddle its fence, but the persisted cut remains
    on the authorized side.

    ``prepared`` may be produced concurrently with transcript selection. On any
    resolution, decoding, cancellation, timeout, or silence failure, this
    function returns the original range with ``status='unavailable'``.
    """

    started = time.monotonic()
    start_speech_handoff = (
        bool(require_speech_handoff)
        if require_start_speech_handoff is None
        else bool(require_start_speech_handoff)
    )
    end_speech_handoff = (
        bool(require_speech_handoff)
        if require_end_speech_handoff is None
        else bool(require_end_speech_handoff)
    )
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
    try:
        semantic_start_limit = (
            max(0.0, original_start - EDGE_WINDOW_SEC / 2.0)
            if search_start_limit_sec is None
            else float(search_start_limit_sec)
        )
        semantic_end_limit = (
            original_end + EDGE_WINDOW_SEC / 2.0
            if search_end_limit_sec is None
            else float(search_end_limit_sec)
        )
    except (TypeError, ValueError, OverflowError):
        return _unavailable(
            original_start,
            original_end,
            stage="input",
            reason="invalid_search_limits",
            started=started,
        )
    if (
        not math.isfinite(semantic_start_limit)
        or not math.isfinite(semantic_end_limit)
        or semantic_start_limit < 0
        or semantic_start_limit > original_start
        or semantic_end_limit < original_end
        or semantic_end_limit <= semantic_start_limit
    ):
        return _unavailable(
            original_start,
            original_end,
            stage="input",
            reason="invalid_search_limits",
            started=started,
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
            prepared_extra = {
                key: value
                for key, value in prepared.diagnostics.items()
                if key not in {"stage", "reason", "elapsed_ms"}
            }
            if "elapsed_ms" in prepared.diagnostics:
                prepared_extra["prepare_elapsed_ms"] = prepared.diagnostics["elapsed_ms"]
            return _unavailable(
                original_start,
                original_end,
                stage=str(prepared.diagnostics.get("stage") or "resolve"),
                reason=str(prepared.diagnostics.get("reason") or "media_unavailable"),
                started=started,
                extra=prepared_extra,
            )
        else:
            source = prepared.source
            preparation_diagnostics = prepared.diagnostics

        source_duration = source.duration_sec
        if (
            source_duration is not None
            and math.isfinite(source_duration)
            and source_duration >= original_end
        ):
            semantic_end_limit = min(semantic_end_limit, source_duration)
        start_observation_start_limit = (
            max(0.0, semantic_start_limit - HANDOFF_OBSERVATION_HALO_SEC)
            if start_speech_handoff or require_start_two_sided
            else semantic_start_limit
        )
        start_observation_end_limit = semantic_end_limit
        end_observation_start_limit = semantic_start_limit
        end_observation_end_limit = (
            semantic_end_limit + HANDOFF_OBSERVATION_HALO_SEC
            if end_speech_handoff or require_end_two_sided
            else semantic_end_limit
        )
        if (
            source_duration is not None
            and math.isfinite(source_duration)
            and source_duration >= original_end
        ):
            start_observation_end_limit = min(
                start_observation_end_limit, source_duration
            )
            end_observation_end_limit = min(
                end_observation_end_limit, source_duration
            )
        end_is_physical_source_edge = bool(
            source_duration is not None
            and math.isfinite(source_duration)
            and abs(original_end - source_duration) <= _FRAME_SEC + 1e-9
            and abs(semantic_end_limit - source_duration) <= _FRAME_SEC + 1e-9
        )

        with tempfile.TemporaryDirectory(prefix="reelai_silence_") as temp_dir:
            edge_temp_dir = Path(temp_dir)
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="silence-edge") as pool:
                futures = {
                    pool.submit(
                        _search_quiet,
                        source,
                        edge="start",
                        required_boundary=original_start,
                        search_start_limit=start_observation_start_limit,
                        search_end_limit=start_observation_end_limit,
                        handoff_band=(original_start, original_start)
                        if start_speech_handoff
                        else None,
                        require_two_sided=require_start_two_sided,
                        temp_dir=edge_temp_dir,
                        ffmpeg_bin=ffmpeg_bin,
                        deadline=deadline,
                        cancel_check=cancel_check,
                    ): "start",
                    pool.submit(
                        _search_quiet,
                        source,
                        edge="end",
                        required_boundary=original_end,
                        search_start_limit=end_observation_start_limit,
                        search_end_limit=end_observation_end_limit,
                        allow_source_edge=end_is_physical_source_edge,
                        handoff_band=(original_end, original_end)
                        if end_speech_handoff
                        else None,
                        require_two_sided=require_end_two_sided,
                        temp_dir=edge_temp_dir,
                        ffmpeg_bin=ffmpeg_bin,
                        deadline=deadline,
                        cancel_check=cancel_check,
                    ): "end",
                }
                edge_results: dict[str, _EdgeSearchResult] = {}
                for future in as_completed(futures):
                    edge_results[futures[future]] = future.result()

            start_result = edge_results["start"]
            end_result = edge_results["end"]
            start_quiet = start_result.quiet
            end_quiet = end_result.quiet
            if (
                start_quiet is not None
                and require_start_two_sided
                and not start_speech_handoff
                and start_quiet.start_sec < semantic_start_limit - _FRAME_SEC
            ):
                start_quiet = None
            if (
                end_quiet is not None
                and require_end_two_sided
                and end_quiet.start_sec
                > semantic_end_limit + HANDOFF_TIMESTAMP_TOLERANCE_SEC
            ):
                end_quiet = None
            if start_quiet is None:
                return _unavailable(
                    original_start,
                    original_end,
                    stage="analyze",
                    reason="start_silence_not_found",
                    started=started,
                    extra={
                        "start_windows": [
                            [round(start, 3), round(end, 3)]
                            for start, end in start_result.windows
                        ]
                    },
                )
            if end_quiet is None:
                return _unavailable(
                    original_start,
                    original_end,
                    stage="analyze",
                    reason="end_silence_not_found",
                    started=started,
                    extra={
                        "end_windows": [
                            [round(start, 3), round(end, 3)]
                            for start, end in end_result.windows
                        ]
                    },
                )

            quiet_start_cut = max(
                start_quiet.start_sec,
                start_quiet.end_sec - START_CUSHION_MS / 1000.0,
            )
            adjusted_start = min(original_start, quiet_start_cut)
            if not start_speech_handoff:
                adjusted_start = max(semantic_start_limit, adjusted_start)
            end_is_verified_source_edge = bool(
                end_is_physical_source_edge
                and abs(end_quiet.end_sec - end_observation_end_limit)
                <= _FRAME_SEC + 1e-9
            )
            quiet_end_cut = (
                original_end
                if end_is_verified_source_edge
                else min(
                    end_quiet.end_sec - _FRAME_SEC,
                    end_quiet.start_sec + END_CUSHION_MS / 1000.0,
                )
            )
            adjusted_end = max(original_end, quiet_end_cut)
            if not end_speech_handoff:
                adjusted_end = min(semantic_end_limit, adjusted_end)
            end_cushion_is_preserved = bool(
                end_is_verified_source_edge
                or adjusted_end + 1e-9
                >= end_quiet.start_sec + END_CUSHION_MS / 1000.0
            )
            cuts_are_inside_quiet = bool(
                start_quiet.start_sec - 1e-9
                <= adjusted_start
                < start_quiet.end_sec + 1e-9
                and end_quiet.start_sec + 1e-9
                <= adjusted_end
                and (
                    adjusted_end < end_quiet.end_sec
                    or (
                        end_is_verified_source_edge
                        and adjusted_end
                        <= end_quiet.end_sec + _FRAME_SEC + 1e-9
                    )
                )
                and end_cushion_is_preserved
            )
            if adjusted_end <= adjusted_start or not cuts_are_inside_quiet:
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
            extra=(
                {"attempt_reasons": list(exc.attempt_reasons)}
                if exc.attempt_reasons
                else None
            ),
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
        "search_start_limit_sec": round(semantic_start_limit, 3),
        "search_end_limit_sec": round(semantic_end_limit, 3),
        "start_window": [
            round(start_result.windows[0][0], 3),
            round(start_result.windows[0][1], 3),
        ],
        "end_window": [
            round(end_result.windows[0][0], 3),
            round(end_result.windows[0][1], 3),
        ],
        "start_windows": [
            [round(start, 3), round(end, 3)]
            for start, end in start_result.windows
        ],
        "end_windows": [
            [round(start, 3), round(end, 3)]
            for start, end in end_result.windows
        ],
        "start_quiet": [round(start_quiet.start_sec, 3), round(start_quiet.end_sec, 3)],
        "end_quiet": [round(end_quiet.start_sec, 3), round(end_quiet.end_sec, 3)],
        "start_shift_sec": round(adjusted_start - original_start, 3),
        "end_shift_sec": round(adjusted_end - original_end, 3),
        "elapsed_ms": round((time.monotonic() - started) * 1000),
    }
    diagnostics.update(
        start_speech_handoff_verified=start_speech_handoff,
        end_speech_handoff_verified=end_speech_handoff,
    )
    if (
        start_speech_handoff
        or end_speech_handoff
        or require_start_two_sided
        or require_end_two_sided
    ):
        diagnostics.update(
            speech_handoff_verified=(
                start_speech_handoff and end_speech_handoff
            ),
            semantic_start_limit_sec=round(semantic_start_limit, 3),
            semantic_end_limit_sec=round(semantic_end_limit, 3),
            observation_start_limit_sec=round(
                start_observation_start_limit, 3
            ),
            observation_end_limit_sec=round(end_observation_end_limit, 3),
            handoff_timestamp_tolerance_sec=HANDOFF_TIMESTAMP_TOLERANCE_SEC,
            start_two_sided_required=bool(require_start_two_sided),
            end_two_sided_required=bool(require_end_two_sided),
        )
    if source.duration_sec is not None:
        diagnostics["source_duration_sec"] = round(source.duration_sec, 3)
    if preparation_diagnostics:
        diagnostics["prepare_elapsed_ms"] = preparation_diagnostics.get("elapsed_ms")
    return SilenceVerificationResult(
        "verified", round(adjusted_start, 3), round(adjusted_end, 3), diagnostics
    )


__all__ = [
    "AudioPreparationResult",
    "PreparedAudioSource",
    "SilenceVerificationResult",
    "persisted_boundary_is_verified",
    "persisted_boundary_is_usable",
    "prepare_audio_source",
    "verify_acoustic_boundaries",
]
