"""Exact lexical anchors from original-language YouTube JSON3 captions.

Supadata remains the semantic transcript.  This module only corroborates an
edge quote with word offsets already exposed by the yt-dlp metadata lookup.
Translated tracks, inferred timings, ambiguous quotes, and incomplete context
all fail closed.
"""
from __future__ import annotations

import asyncio
import json
import math
import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Sequence
from urllib.parse import parse_qs, urlparse

import httpx

from .cancellation import run_cancellable


CancelCheck = Callable[[], bool]
Edge = Literal["start", "end"]

MAX_FETCH_TIMEOUT_SEC = 2.0
MAX_JSON3_BYTES = 8 * 1024 * 1024
CUE_TIME_TOLERANCE_SEC = 0.05
_WORD_RE = re.compile(r"[^\W_]+(?:['\-][^\W_]+)*", re.UNICODE)
_APOSTROPHES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u02bc": "'"})


@dataclass(frozen=True)
class Json3CaptionTrack:
    """One original-language automatic-caption resource.

    Caption URLs are signed credentials.  Keeping the URL out of ``repr`` also
    keeps it out of ordinary exception and diagnostic rendering.
    """

    language: str
    url: str = field(repr=False)


@dataclass(frozen=True)
class LexicalWord:
    """A lexical token with a provider-supplied onset (never interpolated)."""

    text: str
    onset_sec: float


@dataclass(frozen=True)
class EdgeAnchor:
    """Verified timing evidence for one partial cue edge."""

    edge: Edge
    anchor_sec: float
    quote_start_sec: float
    quote_last_onset_sec: float
    excluded_neighbor_onset_sec: float


def _normalize_language(value: object) -> str:
    return "-".join(
        part for part in str(value or "").strip().replace("_", "-").casefold().split("-")
        if part
    )


def _language_matches(actual: str, expected: str) -> bool:
    if not actual or not expected:
        return False
    if actual == expected:
        return True
    return actual.split("-", 1)[0] == expected.split("-", 1)[0]


def _is_untranslated_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except (TypeError, ValueError):
        return False
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False
    return "tlang" not in parse_qs(parsed.query, keep_blank_values=True)


def select_original_json3_track(
    info: Mapping[str, Any],
    *,
    expected_language: str,
) -> Json3CaptionTrack | None:
    """Select one original, automatic JSON3 track from yt-dlp metadata.

    Current yt-dlp marks the authoritative ASR language with a ``-orig`` key;
    translated compatibility entries are deliberately never guessed from.
    The first usable JSON3 format preserves yt-dlp's deterministic client order.
    """

    expected = _normalize_language(expected_language)
    automatic = info.get("automatic_captions") if isinstance(info, Mapping) else None
    if not expected or not isinstance(automatic, Mapping):
        return None

    for raw_language, raw_formats in automatic.items():
        keyed_language = _normalize_language(raw_language)
        if not keyed_language.endswith("-orig"):
            continue
        language = keyed_language[: -len("-orig")]
        if not _language_matches(language, expected) or not isinstance(raw_formats, list):
            continue
        for raw_format in raw_formats:
            if not isinstance(raw_format, Mapping):
                continue
            if str(raw_format.get("ext") or "").strip().casefold() != "json3":
                continue
            url = str(raw_format.get("url") or "").strip()
            if _is_untranslated_url(url):
                return Json3CaptionTrack(language=language, url=url)
    return None


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _tokens(value: object) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", str(value or ""))
    normalized = normalized.translate(_APOSTROPHES).casefold()
    return tuple(match.group(0) for match in _WORD_RE.finditer(normalized))


def _parse_event(raw_event: object) -> list[LexicalWord]:
    if not isinstance(raw_event, Mapping):
        return []
    event_start_ms = _number(raw_event.get("tStartMs"))
    raw_segments = raw_event.get("segs")
    if event_start_ms is None or event_start_ms < 0 or not isinstance(raw_segments, list):
        return []

    lexical: list[tuple[str, float | None]] = []
    for raw_segment in raw_segments:
        if not isinstance(raw_segment, Mapping):
            return []
        segment_tokens = _tokens(raw_segment.get("utf8"))
        if not segment_tokens:
            continue
        # One segment offset cannot establish the onset of two lexical words.
        if len(segment_tokens) != 1:
            return []
        offset = _number(raw_segment.get("tOffsetMs")) if "tOffsetMs" in raw_segment else None
        if offset is not None and offset < 0:
            return []
        lexical.append((segment_tokens[0], offset))

    if not lexical:
        return []
    if lexical[0][1] is None:
        # JSON3 commonly omits zero for the first word.  It is safe only when
        # every sibling proves that this is a word-timed event.
        if len(lexical) < 2 or any(offset is None for _, offset in lexical[1:]):
            return []
        lexical[0] = (lexical[0][0], 0.0)
    elif any(offset is None for _, offset in lexical):
        return []

    offsets = [float(offset) for _, offset in lexical if offset is not None]
    if any(current < previous for previous, current in zip(offsets, offsets[1:])):
        return []
    return [
        LexicalWord(text=text, onset_sec=(event_start_ms + float(offset)) / 1000.0)
        for text, offset in lexical
        if offset is not None
    ]


def parse_json3_words(payload: object) -> tuple[LexicalWord, ...]:
    """Parse explicit JSON3 segment offsets without synthesizing word timing."""

    if not isinstance(payload, Mapping) or not isinstance(payload.get("events"), list):
        return ()
    words: list[LexicalWord] = []
    for event in payload["events"]:
        words.extend(_parse_event(event))
    return tuple(sorted(words, key=lambda word: word.onset_sec))


async def _fetch_payload(
    track: Json3CaptionTrack,
    *,
    headers: Mapping[str, str],
    proxy_url: str,
    timeout_sec: float,
) -> object | None:
    timeout = httpx.Timeout(timeout_sec)
    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            proxy=proxy_url or None,
            follow_redirects=False,
            trust_env=False,
        ) as client:
            response = await asyncio.wait_for(
                client.get(track.url, headers=dict(headers)),
                timeout=timeout_sec,
            )
    except (asyncio.TimeoutError, httpx.HTTPError, OSError, ValueError):
        return None
    if response.status_code != 200 or len(response.content) > MAX_JSON3_BYTES:
        return None
    try:
        return json.loads(response.content)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return None


def fetch_json3_words(
    track: Json3CaptionTrack,
    *,
    headers: Mapping[str, str] | None = None,
    proxy_url: str = "",
    deadline: float,
    cancel_check: CancelCheck | None = None,
) -> tuple[LexicalWord, ...]:
    """Fetch exactly one selected URL within an absolute monotonic deadline."""

    remaining = _number(deadline)
    if remaining is None:
        return ()
    remaining -= time.monotonic()
    if remaining <= 0:
        return ()
    timeout_sec = min(MAX_FETCH_TIMEOUT_SEC, remaining)
    payload = run_cancellable(
        lambda: _fetch_payload(
            track,
            headers=headers or {},
            proxy_url=str(proxy_url or "").strip(),
            timeout_sec=timeout_sec,
        ),
        cancel_check,
    )
    return parse_json3_words(payload)


def _sequence_matches(haystack: Sequence[str], needle: Sequence[str]) -> list[int]:
    if not needle or len(needle) > len(haystack):
        return []
    width = len(needle)
    return [
        index
        for index in range(len(haystack) - width + 1)
        if tuple(haystack[index:index + width]) == tuple(needle)
    ]


def align_edge_anchor(
    words: Sequence[LexicalWord],
    *,
    cue_text: str,
    quote: str,
    edge: Edge,
    cue_start_sec: float,
    cue_end_sec: float,
) -> EdgeAnchor | None:
    """Align a uniquely quoted partial edge to explicit lexical onsets.

    ``start`` returns the first required quote onset and requires a nonempty
    excluded cue prefix.  ``end`` returns the first excluded suffix onset and
    requires a nonempty excluded cue suffix.  The quote and its immediately
    excluded neighbor must both match; unrelated provider drift elsewhere in a
    coarse cue cannot invalidate an otherwise corroborated edge.
    """

    if edge not in {"start", "end"}:
        return None
    cue_start = _number(cue_start_sec)
    cue_end = _number(cue_end_sec)
    if cue_start is None or cue_end is None or cue_end <= cue_start:
        return None
    cue_tokens = _tokens(cue_text)
    quote_tokens = _tokens(quote)
    quote_matches = _sequence_matches(cue_tokens, quote_tokens)
    if len(quote_matches) != 1:
        return None
    quote_start_index = quote_matches[0]
    quote_end_index = quote_start_index + len(quote_tokens)
    if edge == "start" and quote_start_index == 0:
        return None
    if edge == "end" and quote_end_index == len(cue_tokens):
        return None

    timed = [
        word
        for word in words
        if (
            math.isfinite(word.onset_sec)
            and cue_start - CUE_TIME_TOLERANCE_SEC
            <= word.onset_sec
            <= cue_end + CUE_TIME_TOLERANCE_SEC
            and len(_tokens(word.text)) == 1
        )
    ]
    timed.sort(key=lambda word: word.onset_sec)
    timed_tokens = [_tokens(word.text)[0] for word in timed]
    timed_quote_matches = _sequence_matches(timed_tokens, quote_tokens)
    if len(timed_quote_matches) != 1:
        return None
    timed_quote_start = timed_quote_matches[0]
    timed_quote_end = timed_quote_start + len(quote_tokens)
    quote_start_word = timed[timed_quote_start]
    quote_last_word = timed[timed_quote_end - 1]
    if edge == "start":
        if (
            timed_quote_start == 0
            or timed_tokens[timed_quote_start - 1] != cue_tokens[quote_start_index - 1]
        ):
            return None
        excluded_neighbor = timed[timed_quote_start - 1]
        anchor = quote_start_word.onset_sec
    else:
        if (
            timed_quote_end >= len(timed)
            or timed_tokens[timed_quote_end] != cue_tokens[quote_end_index]
        ):
            return None
        excluded_neighbor = timed[timed_quote_end]
        anchor = excluded_neighbor.onset_sec
    return EdgeAnchor(
        edge=edge,
        anchor_sec=anchor,
        quote_start_sec=quote_start_word.onset_sec,
        quote_last_onset_sec=quote_last_word.onset_sec,
        excluded_neighbor_onset_sec=excluded_neighbor.onset_sec,
    )
