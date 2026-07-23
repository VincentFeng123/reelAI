"""Bounded audio-only Groq timing for one already-selected source window.

The adapter never receives or uploads video. Every configuration, decode,
provider, and response-validation failure returns no timing so callers can keep
their existing boundary fallback.
"""
from __future__ import annotations

import math
import os
import re
import time
import uuid
import wave
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from . import lexical_timing, silence
from .lexical_timing import LexicalWord
from .provider_runtime import GenerationContext, ProviderUsageRecord


GROQ_BOUNDARY_MODEL = "whisper-large-v3-turbo"
GROQ_BOUNDARY_PRICE_USD_PER_AUDIO_HOUR = 0.04
GROQ_BOUNDARY_MIN_BILLED_AUDIO_SEC = 10.0
GROQ_BOUNDARY_PRICING_VERSION = "groq-on-demand-2026-07-23"
MAX_BOUNDARY_WINDOW_SEC = 20.0
MAX_BOUNDARY_WAV_BYTES = 2 * 1024 * 1024
EXPECTED_SAMPLE_RATE = 16_000
MIN_EXACT_EDGE_FRAGMENT_TOKENS = 4
MAX_WORD_TIMESTAMP_OVERLAP_SEC = 0.25
_LANGUAGE_RE = re.compile(r"^[a-z]{2}$")


def _create_client(*, api_key: str, timeout_sec: float) -> Any:
    from groq import Groq

    return Groq(api_key=api_key, timeout=timeout_sec, max_retries=0)


def _cancelled(cancel_check: Callable[[], bool] | None) -> bool:
    if cancel_check is None:
        return False
    try:
        return bool(cancel_check())
    except Exception:
        return False


def _field(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _read_pcm_wav(path: Path) -> tuple[bytes, float] | None:
    try:
        if path.suffix.casefold() != ".wav" or not path.is_file():
            return None
        size = path.stat().st_size
        if size <= 44 or size > MAX_BOUNDARY_WAV_BYTES:
            return None
        with wave.open(str(path), "rb") as handle:
            sample_rate = handle.getframerate()
            frame_count = handle.getnframes()
            if (
                handle.getnchannels() != 1
                or handle.getsampwidth() != 2
                or handle.getcomptype() != "NONE"
                or sample_rate != EXPECTED_SAMPLE_RATE
                or frame_count <= 0
            ):
                return None
            duration_sec = frame_count / sample_rate
        if (
            not math.isfinite(duration_sec)
            or duration_sec <= 0
            or duration_sec > MAX_BOUNDARY_WINDOW_SEC
        ):
            return None
        payload = path.read_bytes()
    except (OSError, EOFError, wave.Error):
        return None
    if not payload.startswith(b"RIFF") or payload[8:12] != b"WAVE":
        return None
    return payload, duration_sec


def _absolute_words(
    response: object,
    *,
    window_start_sec: float,
    decoded_duration_sec: float,
) -> tuple[LexicalWord, ...]:
    raw_words = _field(response, "words")
    if (
        not isinstance(raw_words, Sequence)
        or isinstance(raw_words, (str, bytes, bytearray))
        or not raw_words
    ):
        return ()

    words: list[LexicalWord] = []
    previous_end = 0.0
    for index, raw_word in enumerate(raw_words):
        text = _field(raw_word, "word")
        local_start = _finite_number(_field(raw_word, "start"))
        local_end = _finite_number(_field(raw_word, "end"))
        if (
            not isinstance(text, str)
            or not text.strip()
            or local_start is None
            or local_end is None
            or local_start < 0
            or local_end <= local_start
        ):
            return ()
        if local_start < previous_end:
            if (
                previous_end - local_start > MAX_WORD_TIMESTAMP_OVERLAP_SEC
                or local_end <= previous_end
            ):
                return ()
            # Whisper occasionally reports a small overlap between consecutive
            # words. Preserve the word sequence while assigning the shared
            # acoustic interval to the preceding word.
            local_start = previous_end
        trailing_partial = bool(
            index == len(raw_words) - 1
            and 0 <= local_start < decoded_duration_sec < local_end
        )
        if trailing_partial:
            break
        if (
            local_end > decoded_duration_sec
        ):
            return ()
        absolute_start = window_start_sec + local_start
        absolute_end = window_start_sec + local_end
        if not math.isfinite(absolute_start) or not math.isfinite(absolute_end):
            return ()
        words.append(
            LexicalWord(
                text=text.strip(),
                onset_sec=absolute_start,
                end_sec=absolute_end,
            )
        )
        previous_end = local_end
    return tuple(words)


def _language(source: object) -> str:
    language = str(getattr(source, "lexical_language", "") or "")
    primary = language.strip().replace("_", "-").casefold().split("-", 1)[0]
    return primary if _LANGUAGE_RE.fullmatch(primary) else ""


def _record_provider_usage(
    generation_context: GenerationContext | None,
    *,
    decoded_duration_sec: float,
    started_monotonic: float,
    dispatch_id: str,
    response: object | None = None,
    error: BaseException | None = None,
) -> None:
    """Record one physical Groq dispatch without changing the fail-open result."""
    if generation_context is None:
        return
    billed_audio_sec = max(
        GROQ_BOUNDARY_MIN_BILLED_AUDIO_SEC,
        decoded_duration_sec,
    )
    bounded_cost = (
        billed_audio_sec
        * GROQ_BOUNDARY_PRICE_USD_PER_AUDIO_HOUR
        / 3600.0
    )
    status_code: int | None = 200 if error is None else None
    if error is not None:
        try:
            raw_status = getattr(error, "status_code", None)
            if raw_status is None:
                raw_status = getattr(getattr(error, "response", None), "status_code", None)
            status_code = int(raw_status) if raw_status is not None else None
        except (TypeError, ValueError, OverflowError):
            status_code = None
    if error is None:
        error_code = ""
    elif status_code == 429:
        error_code = "provider_rate_limited"
    elif status_code == 408 or (status_code is not None and status_code >= 500):
        error_code = "provider_transient"
    elif status_code is not None and status_code >= 400:
        error_code = "provider_request_rejected"
    else:
        error_code = "provider_transient"
    request_id = str(
        _field(_field(response, "x_groq"), "id") or ""
    ).strip()
    try:
        generation_context.record(
            ProviderUsageRecord(
                provider="groq",
                operation="transcript",
                attempt=1,
                timestamp=datetime.now(timezone.utc).isoformat(),
                status_code=status_code,
                billable_requests=1 if error is None else 0,
                model_used=GROQ_BOUNDARY_MODEL,
                error_code=error_code,
                metadata={
                    "provider_call": True,
                    "stage": "groq_boundary_asr",
                    "dispatched": True,
                    "physical_dispatches": 1,
                    "groq_dispatch_id": dispatch_id,
                    "audio_seconds": round(decoded_duration_sec, 6),
                    "billed_audio_seconds": round(billed_audio_sec, 6),
                    "price_per_audio_hour_usd": (
                        GROQ_BOUNDARY_PRICE_USD_PER_AUDIO_HOUR
                    ),
                    "minimum_billed_audio_seconds": (
                        GROQ_BOUNDARY_MIN_BILLED_AUDIO_SEC
                    ),
                    "pricing_version": GROQ_BOUNDARY_PRICING_VERSION,
                    "billing_usage_known": error is None,
                    "billing_unknown_attempts": 0 if error is None else 1,
                    "billing_unknown_reserved_cost_usd": (
                        0.0 if error is None else round(bounded_cost, 8)
                    ),
                    "actual_cost_usd": (
                        round(bounded_cost, 8) if error is None else None
                    ),
                    "latency_ms": max(
                        0,
                        round((time.monotonic() - started_monotonic) * 1000),
                    ),
                    **({"provider_request_id": request_id} if request_id else {}),
                    **(
                        {"provider_error_type": error.__class__.__name__}
                        if error is not None
                        else {}
                    ),
                },
            )
        )
    except Exception:
        # Accounting can never make a valid related clip disappear.
        return


def align_groq_edge_anchor(
    words: Sequence[LexicalWord],
    *,
    cue_text: str,
    quote: str,
    edge: Literal["start", "end"],
    cue_start_sec: float,
    cue_end_sec: float,
    occurrence: Literal["first", "last"] | None = None,
) -> lexical_timing.EdgeAnchor | None:
    """Align an edge, tolerating only an exact Groq boundary fragment.

    The provider-neutral aligner remains authoritative. Its failure permits a
    longest-first exact prefix (start) or suffix (end) retry, never a semantic
    or fuzzy match. Both the caption cue and timed Groq words must contain the
    fragment exactly once, and the generic aligner still proves the excluded
    neighbor onset used by the cut.
    """

    anchor = lexical_timing.align_edge_anchor(
        words,
        cue_text=cue_text,
        quote=quote,
        edge=edge,
        cue_start_sec=cue_start_sec,
        cue_end_sec=cue_end_sec,
        occurrence=occurrence,
    )
    if anchor is not None or edge not in {"start", "end"}:
        return anchor

    cue_start = lexical_timing._number(cue_start_sec)
    cue_end = lexical_timing._number(cue_end_sec)
    cue_tokens = lexical_timing._tokens(cue_text)
    quote_tokens = lexical_timing._tokens(quote)
    if (
        cue_start is None
        or cue_end is None
        or cue_end <= cue_start
        or len(quote_tokens) <= MIN_EXACT_EDGE_FRAGMENT_TOKENS
    ):
        return None

    timed = [
        word
        for word in words
        if (
            math.isfinite(word.onset_sec)
            and cue_start - lexical_timing.CUE_TIME_TOLERANCE_SEC
            <= word.onset_sec
            <= cue_end + lexical_timing.CUE_TIME_TOLERANCE_SEC
            and len(lexical_timing._tokens(word.text)) == 1
        )
    ]
    timed.sort(key=lambda word: word.onset_sec)
    timed_tokens = tuple(lexical_timing._tokens(word.text)[0] for word in timed)

    for width in range(
        len(quote_tokens) - 1,
        MIN_EXACT_EDGE_FRAGMENT_TOKENS - 1,
        -1,
    ):
        fragment = (
            quote_tokens[:width]
            if edge == "start"
            else quote_tokens[len(quote_tokens) - width :]
        )
        cue_matches = lexical_timing._sequence_matches(cue_tokens, fragment)
        timed_matches = lexical_timing._sequence_matches(timed_tokens, fragment)
        if len(cue_matches) != 1 or len(timed_matches) != 1:
            continue
        timed_start = timed_matches[0]
        timed_end = timed_start + width
        if (edge == "start" and timed_start == 0) or (
            edge == "end" and timed_end >= len(timed)
        ):
            continue
        anchor = lexical_timing.align_edge_anchor(
            words,
            cue_text=cue_text,
            quote=" ".join(fragment),
            edge=edge,
            cue_start_sec=cue_start,
            cue_end_sec=cue_end,
        )
        if anchor is not None:
            return anchor
    return None


def transcribe_boundary_words(
    prepared: object,
    *,
    window_start_sec: float,
    window_end_sec: float,
    timeout_sec: float,
    cancel_check: Callable[[], bool] | None = None,
    generation_context: GenerationContext | None = None,
) -> tuple[LexicalWord, ...]:
    """Return validated Groq word spans in absolute source time.

    Only the requested bounded WAV is uploaded. An empty tuple is the complete
    fail-open contract, including cancellation and all provider failures.
    """

    try:
        start = _finite_number(window_start_sec)
        end = _finite_number(window_end_sec)
        timeout = _finite_number(timeout_sec)
        source = getattr(prepared, "source", None)
        api_key = (os.environ.get("GROQ_API_KEY") or "").strip()
        if (
            not api_key
            or not bool(getattr(prepared, "ready", False))
            or source is None
            or start is None
            or end is None
            or timeout is None
            or start < 0
            or end <= start
            or end - start > MAX_BOUNDARY_WINDOW_SEC
            or timeout <= 0
            or _cancelled(cancel_check)
        ):
            return ()

        deadline = time.monotonic() + timeout
        with silence.decode_audio_window(
            source,
            window_start_sec=start,
            window_end_sec=end,
            max_duration_sec=MAX_BOUNDARY_WINDOW_SEC,
            timeout_sec=timeout,
            cancel_check=cancel_check,
        ) as wav_path:
            audio = _read_pcm_wav(wav_path)
            remaining = deadline - time.monotonic()
            if audio is None or remaining <= 0 or _cancelled(cancel_check):
                return ()
            wav_bytes, decoded_duration = audio
            if decoded_duration > end - start + (1 / EXPECTED_SAMPLE_RATE):
                return ()
            client = _create_client(api_key=api_key, timeout_sec=remaining)
            try:
                request: dict[str, object] = {
                    "file": ("boundary.wav", wav_bytes, "audio/wav"),
                    "model": GROQ_BOUNDARY_MODEL,
                    "response_format": "verbose_json",
                    "temperature": 0,
                    "timestamp_granularities": ["word"],
                    "timeout": remaining,
                }
                language = _language(source)
                if language:
                    request["language"] = language
                provider_started = time.monotonic()
                dispatch_id = str(uuid.uuid4())
                try:
                    response = client.audio.transcriptions.create(**request)
                except Exception as exc:
                    _record_provider_usage(
                        generation_context,
                        decoded_duration_sec=decoded_duration,
                        started_monotonic=provider_started,
                        dispatch_id=dispatch_id,
                        error=exc,
                    )
                    raise
                _record_provider_usage(
                    generation_context,
                    decoded_duration_sec=decoded_duration,
                    started_monotonic=provider_started,
                    dispatch_id=dispatch_id,
                    response=response,
                )
            finally:
                close = getattr(client, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass
            if _cancelled(cancel_check):
                return ()
            return _absolute_words(
                response,
                window_start_sec=start,
                decoded_duration_sec=decoded_duration,
            )
    except Exception:
        return ()
