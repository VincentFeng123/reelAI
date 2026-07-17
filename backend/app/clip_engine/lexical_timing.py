"""Exact lexical anchors from original-language YouTube JSON3 captions.

Supadata remains the semantic transcript.  This module only corroborates an
edge quote with word offsets already exposed by the yt-dlp metadata lookup.
Translated tracks, inferred timings, ambiguous quotes, and incomplete context
all fail closed.
"""
from __future__ import annotations

import asyncio
from difflib import SequenceMatcher
import json
import math
import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Sequence
from urllib.parse import parse_qs, urlparse

import httpx
from curl_cffi.const import CurlOpt
from curl_cffi import requests as curl_requests

from .cancellation import run_cancellable


CancelCheck = Callable[[], bool]
Edge = Literal["start", "end"]

MAX_FETCH_TIMEOUT_SEC = 2.0
MAX_AUTHORITATIVE_TRACKS = 2
MAX_JSON3_BYTES = 8 * 1024 * 1024
CUE_TIME_TOLERANCE_SEC = 0.05
APPROXIMATE_QUOTE_MIN_TOKENS = 4
APPROXIMATE_QUOTE_MIN_EXACT_TOKENS = 3
APPROXIMATE_QUOTE_MIN_CHARACTER_SCORE = 0.82
APPROXIMATE_QUOTE_MIN_EDGE_SCORE = 0.82
APPROXIMATE_QUOTE_MIN_MARGIN = 0.04
_WORD_RE = re.compile(r"[^\W_]+(?:['\-][^\W_]+)*", re.UNICODE)
_APOSTROPHES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u02bc": "'"})


@dataclass(frozen=True)
class Json3CaptionTrack:
    """One original-language authoritative caption resource.

    Caption URLs are signed credentials.  Keeping the URL out of ``repr`` also
    keeps it out of ordinary exception and diagnostic rendering.
    """

    language: str
    url: str = field(repr=False)
    impersonate: bool = False


@dataclass(frozen=True)
class LexicalWord:
    """A lexical token with provider-supplied timing (never interpolated)."""

    text: str
    onset_sec: float
    end_sec: float | None = None


@dataclass(frozen=True)
class EdgeAnchor:
    """Verified timing evidence for one partial cue edge."""

    edge: Edge
    anchor_sec: float
    quote_start_sec: float
    quote_last_onset_sec: float
    excluded_neighbor_onset_sec: float
    quote_last_end_sec: float | None = None
    excluded_neighbor_end_sec: float | None = None


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


def _is_original_asr_alias(url: str, *, expected_language: str) -> bool:
    """Accept yt-dlp's duplicate exact-language ASR key, never a translation."""
    try:
        query = parse_qs(urlparse(url).query, keep_blank_values=True)
    except (TypeError, ValueError):
        return False
    languages = [_normalize_language(value) for value in query.get("lang", [])]
    kinds = {str(value or "").strip().casefold() for value in query.get("kind", [])}
    return bool(
        "tlang" not in query
        and "asr" in kinds
        and any(_language_matches(language, expected_language) for language in languages)
    )


def select_original_json3_tracks(
    info: Mapping[str, Any],
    *,
    expected_language: str,
) -> tuple[Json3CaptionTrack, ...]:
    """Select up to two unique untranslated JSON3 tracks in trust order.

    Prefer yt-dlp's authoritative automatic ``-orig`` key, then its validated
    exact-language ASR alias, then exact-language manual captions.  Format and
    mapping order remain deterministic and duplicate signed URLs are attempted
    only once.
    """

    expected = _normalize_language(expected_language)
    automatic = info.get("automatic_captions") if isinstance(info, Mapping) else None
    manual = info.get("subtitles") if isinstance(info, Mapping) else None
    if not expected:
        return ()

    selected: list[Json3CaptionTrack] = []
    seen_urls: set[str] = set()

    def add_tracks(
        collection: object,
        *,
        kind: Literal["original", "alias", "manual"],
    ) -> bool:
        if not isinstance(collection, Mapping):
            return False
        for raw_language, raw_formats in collection.items():
            keyed_language = _normalize_language(raw_language)
            is_original_key = keyed_language.endswith("-orig")
            language = (
                keyed_language[: -len("-orig")]
                if is_original_key
                else keyed_language
            )
            if kind == "original" and not is_original_key:
                continue
            if kind == "alias" and is_original_key:
                continue
            if (
                not _language_matches(language, expected)
                or not isinstance(raw_formats, list)
            ):
                continue
            for raw_format in raw_formats:
                if not isinstance(raw_format, Mapping):
                    continue
                if str(raw_format.get("ext") or "").strip().casefold() != "json3":
                    continue
                url = str(raw_format.get("url") or "").strip()
                if not _is_untranslated_url(url):
                    continue
                if kind == "alias" and not _is_original_asr_alias(
                    url, expected_language=expected
                ):
                    continue
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                selected.append(
                    Json3CaptionTrack(
                        language=language,
                        url=url,
                        impersonate=raw_format.get("impersonate") is True,
                    )
                )
                if len(selected) >= MAX_AUTHORITATIVE_TRACKS:
                    return True
        return False

    for collection, kind in (
        (automatic, "original"),
        (automatic, "alias"),
        (manual, "manual"),
    ):
        if add_tracks(collection, kind=kind):
            break
    return tuple(selected)


def select_original_json3_track(
    info: Mapping[str, Any],
    *,
    expected_language: str,
) -> Json3CaptionTrack | None:
    """Return the first authoritative track for compatibility with callers."""

    tracks = select_original_json3_tracks(
        info,
        expected_language=expected_language,
    )
    return tracks[0] if tracks else None


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
        # JSON3 commonly omits zero for the first word. A one-token event has
        # no second lexical position to interpolate: its explicit event start
        # is the onset of that sole word. Multi-token events still need every
        # later sibling offset to prove that the event is word-timed.
        if len(lexical) > 1 and any(
            offset is None for _, offset in lexical[1:]
        ):
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
    try:
        if track.impersonate:
            request_headers = {
                str(key): str(value)
                for key, value in headers.items()
                if str(key).strip().casefold()
                not in {"user-agent", "accept-encoding"}
                and not str(key).strip().casefold().startswith("sec-ch-ua")
            }
            async with curl_requests.AsyncSession(
                impersonate="chrome",
                max_clients=1,
                trust_env=False,
                curl_options=(
                    {}
                    if proxy_url
                    else {CurlOpt.PROXY: ""}
                ),
            ) as client:
                response = await asyncio.wait_for(
                    client.get(
                        track.url,
                        headers=request_headers,
                        proxy=proxy_url or None,
                        timeout=timeout_sec,
                        allow_redirects=False,
                    ),
                    timeout=timeout_sec,
                )
        else:
            timeout = httpx.Timeout(timeout_sec)
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
    except (
        asyncio.TimeoutError,
        curl_requests.errors.RequestsError,
        httpx.HTTPError,
        OSError,
        ValueError,
    ):
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


def _token_key(value: str) -> str:
    """Collapse punctuation and compound separators for cross-ASR comparison."""
    return "".join(character for character in value if character.isalnum())


def _character_score(left: Sequence[str], right: Sequence[str]) -> float:
    left_text = "".join(_token_key(token) for token in left)
    right_text = "".join(_token_key(token) for token in right)
    if not left_text or not right_text:
        return 0.0
    return SequenceMatcher(None, left_text, right_text, autojunk=False).ratio()


def _token_lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    left_keys = tuple(_token_key(token) for token in left)
    right_keys = tuple(_token_key(token) for token in right)
    previous = [0] * (len(right_keys) + 1)
    for left_token in left_keys:
        current = [0]
        for index, right_token in enumerate(right_keys, start=1):
            if left_token and left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def _approximate_quote_span(
    timed_tokens: Sequence[str],
    quote_tokens: Sequence[str],
    *,
    edge: Edge,
    cue_neighbor: str,
) -> tuple[int, int] | None:
    """Find one strong edge-preserving ASR match without inventing timing.

    Supadata and YouTube JSON3 often disagree on compounds or one inflected
    word.  A candidate still needs at least three exact ordered tokens, strong
    character agreement, a strong match on the actual cut-edge word, and one
    unambiguous timed anchor.  Returned indices always refer to real JSON3
    words; no word onset is interpolated.
    """

    quote_width = len(quote_tokens)
    if quote_width < APPROXIMATE_QUOTE_MIN_TOKENS:
        return None
    minimum_exact = max(
        APPROXIMATE_QUOTE_MIN_EXACT_TOKENS,
        math.ceil(quote_width * 0.6),
    )
    minimum_width = max(2, quote_width - 2)
    maximum_width = min(len(timed_tokens), quote_width + 2)
    by_anchor: dict[int, tuple[float, int, int]] = {}

    for width in range(minimum_width, maximum_width + 1):
        for start in range(0, len(timed_tokens) - width + 1):
            end = start + width
            if edge == "start" and start == 0:
                continue
            if edge == "end" and end >= len(timed_tokens):
                continue
            window = timed_tokens[start:end]
            edge_score = _character_score(
                (quote_tokens[0] if edge == "start" else quote_tokens[-1],),
                (window[0] if edge == "start" else window[-1],),
            )
            if edge_score < APPROXIMATE_QUOTE_MIN_EDGE_SCORE:
                continue
            exact_tokens = _token_lcs_length(quote_tokens, window)
            if exact_tokens < minimum_exact:
                continue
            character_score = _character_score(quote_tokens, window)
            if character_score < APPROXIMATE_QUOTE_MIN_CHARACTER_SCORE:
                continue
            excluded_token = timed_tokens[start - 1] if edge == "start" else timed_tokens[end]
            neighbor_score = _character_score((cue_neighbor,), (excluded_token,))
            score = (
                0.70 * character_score
                + 0.20 * (exact_tokens / quote_width)
                + 0.08 * edge_score
                + 0.02 * neighbor_score
            )
            anchor_index = start if edge == "start" else end - 1
            current = by_anchor.get(anchor_index)
            candidate = (score, start, end)
            if current is None or candidate > current:
                by_anchor[anchor_index] = candidate

    ranked = sorted(by_anchor.values(), reverse=True)
    if not ranked:
        return None
    if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < APPROXIMATE_QUOTE_MIN_MARGIN:
        return None
    return ranked[0][1], ranked[0][2]


def align_edge_anchor(
    words: Sequence[LexicalWord],
    *,
    cue_text: str,
    quote: str,
    edge: Edge,
    cue_start_sec: float,
    cue_end_sec: float,
    occurrence: Literal["first", "last"] | None = None,
) -> EdgeAnchor | None:
    """Align a uniquely quoted partial edge to explicit lexical onsets.

    ``start`` returns the first required quote onset and requires a nonempty
    excluded cue prefix.  ``end`` returns the first excluded suffix onset and
    requires a nonempty excluded cue suffix. Exact quotes are preferred; a
    unique, edge-preserving approximate match tolerates minor ASR spelling and
    compound drift. The excluded neighbor must have a real onset, but its text
    may differ between providers.
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
    if not quote_matches or (
        len(quote_matches) != 1 and occurrence not in {"first", "last"}
    ):
        return None
    quote_start_index = (
        quote_matches[0] if occurrence != "last" else quote_matches[-1]
    )
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
    if timed_quote_matches and (
        len(timed_quote_matches) == 1 or occurrence in {"first", "last"}
    ):
        timed_quote_start = (
            timed_quote_matches[0]
            if occurrence != "last"
            else timed_quote_matches[-1]
        )
        timed_quote_end = timed_quote_start + len(quote_tokens)
    elif timed_quote_matches:
        return None
    else:
        cue_neighbor = (
            cue_tokens[quote_start_index - 1]
            if edge == "start"
            else cue_tokens[quote_end_index]
        )
        approximate_span = _approximate_quote_span(
            timed_tokens,
            quote_tokens,
            edge=edge,
            cue_neighbor=cue_neighbor,
        )
        if approximate_span is None:
            return None
        timed_quote_start, timed_quote_end = approximate_span
    quote_start_word = timed[timed_quote_start]
    quote_last_word = timed[timed_quote_end - 1]
    if edge == "start":
        if timed_quote_start == 0:
            return None
        excluded_neighbor = timed[timed_quote_start - 1]
        anchor = quote_start_word.onset_sec
    else:
        if timed_quote_end >= len(timed):
            return None
        excluded_neighbor = timed[timed_quote_end]
        anchor = excluded_neighbor.onset_sec
    return EdgeAnchor(
        edge=edge,
        anchor_sec=anchor,
        quote_start_sec=quote_start_word.onset_sec,
        quote_last_onset_sec=quote_last_word.onset_sec,
        excluded_neighbor_onset_sec=excluded_neighbor.onset_sec,
        quote_last_end_sec=_number(getattr(quote_last_word, "end_sec", None)),
        excluded_neighbor_end_sec=_number(
            getattr(excluded_neighbor, "end_sec", None)
        ),
    )
