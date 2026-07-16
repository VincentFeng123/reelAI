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
import wave
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from . import lexical_timing, silence
from .lexical_timing import LexicalWord


GROQ_BOUNDARY_MODEL = "whisper-large-v3-turbo"
MAX_BOUNDARY_WINDOW_SEC = 20.0
MAX_BOUNDARY_WAV_BYTES = 2 * 1024 * 1024
EXPECTED_SAMPLE_RATE = 16_000
MIN_EXACT_EDGE_FRAGMENT_TOKENS = 4
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
    for raw_word in raw_words:
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
            or local_start < previous_end
            or local_end > decoded_duration_sec
        ):
            return ()
        absolute_start = window_start_sec + local_start
        if not math.isfinite(absolute_start):
            return ()
        words.append(LexicalWord(text=text.strip(), onset_sec=absolute_start))
        previous_end = local_end
    return tuple(words)


def _language(source: object) -> str:
    language = str(getattr(source, "lexical_language", "") or "")
    primary = language.strip().replace("_", "-").casefold().split("-", 1)[0]
    return primary if _LANGUAGE_RE.fullmatch(primary) else ""


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
) -> tuple[LexicalWord, ...]:
    """Return validated Groq word onsets in absolute source time.

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
                response = client.audio.transcriptions.create(**request)
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
