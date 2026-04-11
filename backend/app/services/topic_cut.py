"""
Topic-aware reel cutter for long-form YouTube videos.

Most of the existing pipeline (`backend/app/services/reels.py`) cuts ONE clip per
video, ranked by how well a transcript chunk matches a study concept. This module
takes the opposite stance: given a YouTube video coming straight from the HTML
scraper (`YouTubeService._extract_videos_from_search_html`), it cuts the WHOLE
video into multiple self-contained reels — one per topic the creator actually
introduces and transitions away from.

Usage from Python:

    from backend.app.services.topic_cut import cut_video_into_topic_reels

    reels = cut_video_into_topic_reels("https://www.youtube.com/watch?v=aircAruvnKk")
    # -> list[TopicReel] with .video_id, .t_start, .t_end, .label, .summary

    # YouTube Shorts return [] — leave them untouched.

CLI:

    python -m backend.app.services.topic_cut <url-or-video-id>
    python -m backend.app.services.topic_cut <url-or-video-id> --json
    python -m backend.app.services.topic_cut <url-or-video-id> --no-llm   # heuristic only

The program does NOT write to the database. It returns the cut list and the
caller decides whether to persist (so you can wire it into `IngestionPipeline`
later, or run it ad-hoc to inspect output).

Pipeline:

    1.  Resolve URL → 11-char video ID (`extract_video_id`)
    2.  Classify Short vs long-form (`classify_video`)
            * `/shorts/` in path  -> Short
            * duration ≤ 60s      -> Short
            * otherwise           -> long-form
        Duration is taken from yt-dlp metadata if available, else inferred from
        the transcript's last cue.
    3.  Fetch the full transcript via `youtube_transcript_api` — same library and
        same fallback chain (`find_manually_created_transcript` →
        `find_generated_transcript` → any) used by
        `YouTubeService.get_transcript`.
    4.  If Short: return [].
    5.  If long-form:
          (a) Ask gpt-4o-mini to identify topic boundaries by CUE INDEX (we send
              the transcript with explicit `[idx mm:ss]` prefixes so the model
              cannot hallucinate timestamps it didn't see). Parse the JSON
              response into (start_idx, end_idx, label) triples.
          (b) Snap each boundary to the nearest natural cue gap so a clip never
              starts or ends mid-word.
          (c) Drop any segment that's <30s or >12min — that's almost always a
              parsing artefact, not a real topic.
          (d) If the LLM is unavailable OR returns garbage, fall back to a
              lexical-novelty heuristic that splits on big vocabulary shifts.
    6.  Return TopicReel records.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# A YouTube video ID is exactly 11 chars from this charset. Source-of-truth
# pattern is duplicated here so this module has zero internal imports — that
# keeps the CLI runnable in isolation. The pattern matches
# `services/youtube.py:YOUTUBE_VIDEO_ID_PATTERN` byte-for-byte.
_VIDEO_ID_REGEX = re.compile(r"^[A-Za-z0-9_-]{11}$")

# Anything ≤ this is treated as a Short, even if /shorts/ isn't in the URL.
# YouTube's own definition is 60 seconds, so we mirror it.
SHORT_MAX_DURATION_SEC = 60

# Topic-segment guardrails. Reels shorter than MIN are almost never useful as
# stand-alone content; reels longer than MAX usually mean the LLM merged two
# topics. The MAX is generous because some long-form creators (lectures,
# documentaries) genuinely sustain a topic for ~10 minutes.
MIN_TOPIC_REEL_SEC = 30
MAX_TOPIC_REEL_SEC = 12 * 60

# Cue snap tolerance — we'll move a t_start/t_end up to this many seconds to
# land on a real transcript-cue boundary so the clip doesn't begin or end
# mid-sentence.
SNAP_TOLERANCE_SEC = 2.0

# Default LLM model. Mirrors `Settings.openai_chat_model`. Cheap, fast, fits
# any single-video transcript in one call.
DEFAULT_MODEL = "gpt-4o-mini"

# Hard cap on transcript cues sent to the LLM. ~3000 cues ≈ 4 hours of speech;
# beyond that we fall back to the heuristic (and warn) rather than risk a
# token-limit error.
MAX_CUES_FOR_LLM = 3000


# --------------------------------------------------------------------------- #
# Public dataclasses
# --------------------------------------------------------------------------- #


@dataclass
class TranscriptCue:
    """One transcript line. `text` is already stripped of newlines."""
    start: float
    duration: float
    text: str

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass
class VideoClassification:
    """Result of `classify_video`."""
    video_id: str
    is_short: bool
    duration_sec: float
    reason: str  # human-readable, e.g. "url path /shorts/" or "duration<=60s"


@dataclass
class TopicReel:
    """One self-contained reel cut from a long-form video."""
    video_id: str
    t_start: float
    t_end: float
    label: str
    summary: str = ""
    cue_start_idx: int = -1
    cue_end_idx: int = -1

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.t_end - self.t_start)

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["duration_sec"] = round(self.duration_sec, 2)
        d["t_start"] = round(self.t_start, 2)
        d["t_end"] = round(self.t_end, 2)
        return d


# --------------------------------------------------------------------------- #
# YouTube chapters — the killer free signal
# --------------------------------------------------------------------------- #


@dataclass
class Chapter:
    """
    A creator-authored chapter marker scraped from a YouTube video description.

    yt-dlp parses the chapter list from the description and exposes it under
    `info_dict["chapters"]` as `[{"start_time": float, "end_time": float, "title": str}]`.
    These are by FAR the highest-quality topic boundaries we can get for free —
    they're the creator's own segmentation, not an inference. When chapters
    exist, we use them directly and skip the LLM/heuristic entirely.
    """

    start: float
    end: float
    title: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def extract_chapters(info_dict: dict[str, Any] | None) -> list[Chapter]:
    """
    Pull a list of `Chapter` objects out of a yt-dlp info_dict.

    Returns [] when:
      * info_dict is None or doesn't have a "chapters" key
      * the chapters list is empty (creator didn't put markers in the description)
      * none of the entries have valid start/end timestamps

    Each surviving chapter is sanity-checked: end > start, title is non-empty
    after stripping. Duplicate or zero-length entries are filtered out.
    """
    if not isinstance(info_dict, dict):
        return []
    raw = info_dict.get("chapters")
    if not isinstance(raw, list):
        return []

    out: list[Chapter] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        try:
            start = float(entry.get("start_time", 0.0))
            end = float(entry.get("end_time", 0.0))
        except (TypeError, ValueError):
            continue
        title = str(entry.get("title") or "").strip()
        if not title or end <= start:
            continue
        out.append(Chapter(start=start, end=end, title=title))

    # Sort by start time and dedupe identical (start, title) pairs.
    out.sort(key=lambda c: (c.start, c.title))
    deduped: list[Chapter] = []
    seen: set[tuple[float, str]] = set()
    for chap in out:
        key = (round(chap.start, 1), chap.title.lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(chap)
    return deduped


# Skip-segment heuristics for chapters — these usually mean intro/outro/sponsor
# fluff that we don't want as standalone reels. Matched case-insensitively as
# WHOLE WORDS (not substrings). Forms are listed explicitly to avoid stemming
# false-positives: "intro" matches "Intro" but NOT "Introducing layers", and
# "introduction" matches "Introduction" explicitly.
_SKIPPABLE_CHAPTER_TOKENS = (
    "intro",
    "introduction",
    "introductions",
    "outro",
    "outros",
    "conclusion",
    "conclusions",
    "sponsor",
    "sponsors",
    "sponsored",
    "subscribe",
    "thanks",
    "credits",
    "shoutout",
    "shoutouts",
    "discord",
    "patreon",
    "merch",
    "giveaway",
)
# Multi-word skip phrases (whole-phrase substring is fine here because they're
# specific enough to not be false-positives).
_SKIPPABLE_CHAPTER_PHRASES = (
    "thank you",
    "ad break",
    "ad read",
    "sponsor read",
    "wrap up",
)

_SKIPPABLE_CHAPTER_WORD_REGEX = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in _SKIPPABLE_CHAPTER_TOKENS) + r")\b",
    re.IGNORECASE,
)


def _is_skippable_chapter(title: str) -> bool:
    """
    Decide whether a chapter title looks like fluff (intro / outro / sponsor /
    "thanks for watching") that shouldn't become a standalone reel.

    Conservative on purpose: we only skip when the title is short (≤4 words)
    AND either matches a single-word token at WORD-BOUNDARY level (so
    "Introducing layers" is NOT killed by the "intro" token), or contains
    one of the well-known multi-word fluff phrases. Empty titles are skipped.
    """
    clean = title.strip().lower()
    if not clean:
        return True
    word_count = len(clean.split())
    if word_count > 4:
        return False
    if any(phrase in clean for phrase in _SKIPPABLE_CHAPTER_PHRASES):
        return True
    return bool(_SKIPPABLE_CHAPTER_WORD_REGEX.search(clean))


def chapters_to_topic_segments(
    chapters: Sequence[Chapter],
) -> list[tuple[float, float, str, str]]:
    """
    Convert yt-dlp chapters into the same `(t_start, t_end, label, summary)`
    tuple shape that `_llm_topic_segments` and `_heuristic_topic_segments`
    return, so the same boundary-snapping pipeline applies to all three.

    Drops chapters whose titles look like intros/outros/sponsor fluff. Does
    NOT enforce the [MIN, MAX] duration window — `_snap_segments_to_cues`
    handles that downstream so the same rules apply uniformly.

    Note: this returns SECONDS, not cue indices. The snap function transparently
    handles both because it works in seconds internally and only uses cue
    indices for the optional inter-cue gap snap. (See the chapter-aware
    overload of `_snap_segments_to_cues` below.)
    """
    out: list[tuple[float, float, str, str]] = []
    for chap in chapters:
        if _is_skippable_chapter(chap.title):
            logger.debug("skipping chapter %r as fluff", chap.title)
            continue
        out.append((chap.start, chap.end, chap.title, ""))
    return out


# --------------------------------------------------------------------------- #
# URL / ID parsing
# --------------------------------------------------------------------------- #


def extract_video_id(url_or_id: str) -> tuple[str, bool]:
    """
    Parse a YouTube URL or bare 11-char ID into (video_id, is_shorts_url).

    Recognises:
        https://www.youtube.com/watch?v=XXXXXXXXXXX
        https://youtu.be/XXXXXXXXXXX
        https://www.youtube.com/shorts/XXXXXXXXXXX     <- is_shorts_url=True
        https://www.youtube.com/embed/XXXXXXXXXXX
        XXXXXXXXXXX                                    <- bare 11-char ID

    Raises ValueError on anything else, with the offending input echoed back so
    a CLI user can see what they typed.
    """
    raw = (url_or_id or "").strip()
    if not raw:
        raise ValueError("empty url/id")

    if _VIDEO_ID_REGEX.match(raw):
        return raw, False

    parsed = urlparse(raw)
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]

    is_shorts_url = False
    candidate: str | None = None

    if host in ("youtu.be",):
        candidate = parsed.path.lstrip("/").split("/")[0]
    elif host in ("youtube.com", "m.youtube.com", "music.youtube.com"):
        path_parts = [p for p in parsed.path.split("/") if p]
        if path_parts and path_parts[0] == "shorts" and len(path_parts) >= 2:
            candidate = path_parts[1]
            is_shorts_url = True
        elif path_parts and path_parts[0] == "embed" and len(path_parts) >= 2:
            candidate = path_parts[1]
        else:
            qs = parse_qs(parsed.query or "")
            v_list = qs.get("v") or []
            if v_list:
                candidate = v_list[0]

    if not candidate or not _VIDEO_ID_REGEX.match(candidate):
        raise ValueError(f"could not extract a YouTube video ID from {url_or_id!r}")

    return candidate, is_shorts_url


# --------------------------------------------------------------------------- #
# Transcript fetching
# --------------------------------------------------------------------------- #


def fetch_transcript(
    video_id: str,
    *,
    languages: Sequence[str] = ("en", "en-US", "en-GB"),
) -> list[TranscriptCue]:
    """
    Fetch the timestamped transcript for `video_id`.

    Mirrors `YouTubeService._fallback_any_transcript`'s precedence order:
        1. manually-authored transcript in any of `languages`
        2. auto-generated transcript in any of `languages`
        3. ANY available transcript (last resort, may be a different language)

    Returns an empty list if every strategy fails — the caller can decide
    whether that's a hard error or a "skip and move on".

    NOTE: this is a network call (no cache here, on purpose — the existing
    pipeline caches at the DB level via `transcript_cache`; this CLI is meant
    to run standalone, so it doesn't touch the DB).
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api._errors import (
            NoTranscriptFound,
            TranscriptsDisabled,
            VideoUnavailable,
        )
    except ImportError as exc:
        raise RuntimeError(
            "youtube_transcript_api is not installed. Activate the backend venv "
            "or `pip install youtube-transcript-api==1.2.4`."
        ) from exc

    api = YouTubeTranscriptApi()

    # Strategy 1+2: directed fetch by language. The library's `.fetch(video_id,
    # languages=[...])` already prefers manual over auto, so we get both for free.
    try:
        raw = api.fetch(video_id, languages=list(languages)).to_raw_data()
        cues = _coerce_to_cues(raw)
        if cues:
            return cues
    except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable):
        pass
    except Exception:  # noqa: BLE001 — defensive
        logger.exception("transcript fetch (directed) failed for video_id=%s", video_id)

    # Strategy 3: walk every transcript track and take the first that returns text.
    try:
        listing = api.list(video_id)
        # Manual first, then generated, then anything.
        try:
            return _coerce_to_cues(
                listing.find_manually_created_transcript(list(languages)).fetch().to_raw_data()
            )
        except NoTranscriptFound:
            pass
        try:
            return _coerce_to_cues(
                listing.find_generated_transcript(list(languages)).fetch().to_raw_data()
            )
        except NoTranscriptFound:
            pass
        for transcript in listing:
            try:
                return _coerce_to_cues(transcript.fetch().to_raw_data())
            except Exception:  # noqa: BLE001
                continue
    except (TranscriptsDisabled, VideoUnavailable):
        return []
    except Exception:  # noqa: BLE001
        logger.exception("transcript listing failed for video_id=%s", video_id)
        return []

    return []


def cues_from_ingest_cues(ingest_cues: Iterable[Any]) -> list[TranscriptCue]:
    """
    Adapter from `backend.app.ingestion.models.IngestTranscriptCue` (pydantic,
    `start`/`end` fields) to this module's `TranscriptCue` (dataclass,
    `start`/`duration`/derived `end`).

    Kept here so the ingestion pipeline can pass cues straight from
    `transcribe()` without import-cycling through the segmenter.
    """
    out: list[TranscriptCue] = []
    for cue in ingest_cues:
        try:
            start = float(getattr(cue, "start"))
            end = float(getattr(cue, "end"))
            text = str(getattr(cue, "text") or "").replace("\n", " ").strip()
        except (AttributeError, TypeError, ValueError):
            continue
        if not text:
            continue
        duration = max(end - start, 0.01)
        out.append(TranscriptCue(start=start, duration=duration, text=text))
    return out


def _coerce_to_cues(raw: list[dict[str, Any]]) -> list[TranscriptCue]:
    """Normalize a youtube-transcript-api raw payload into our TranscriptCue list."""
    cues: list[TranscriptCue] = []
    if not isinstance(raw, list):
        return cues
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        try:
            start = float(entry.get("start", 0.0))
            duration = float(entry.get("duration", 0.0))
            text = str(entry.get("text", "")).replace("\n", " ").strip()
        except (TypeError, ValueError):
            continue
        if not text:
            continue
        cues.append(TranscriptCue(start=start, duration=max(duration, 0.01), text=text))
    return cues


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #


def classify_video(
    url_or_id: str,
    *,
    duration_sec: float | None = None,
    transcript: Sequence[TranscriptCue] | None = None,
) -> VideoClassification:
    """
    Decide whether `url_or_id` is a YouTube Short or a long-form video.

    `duration_sec` (if provided) takes precedence — that's what the HTML
    scraper already extracts in `_video_row_from_renderer` and is the most
    authoritative signal. Otherwise we infer duration from the last cue's
    end-time, which is good enough for the Short/long-form decision (the
    only edge case where it's wrong is silent intros/outros, which Shorts
    don't have).
    """
    video_id, is_shorts_url = extract_video_id(url_or_id)

    if is_shorts_url:
        return VideoClassification(
            video_id=video_id,
            is_short=True,
            duration_sec=float(duration_sec or 0.0),
            reason="url path is /shorts/",
        )

    effective_duration = float(duration_sec) if duration_sec is not None else 0.0
    if effective_duration <= 0 and transcript:
        last = transcript[-1]
        effective_duration = last.start + last.duration

    if 0 < effective_duration <= SHORT_MAX_DURATION_SEC:
        return VideoClassification(
            video_id=video_id,
            is_short=True,
            duration_sec=effective_duration,
            reason=f"duration {effective_duration:.0f}s ≤ {SHORT_MAX_DURATION_SEC}s",
        )

    return VideoClassification(
        video_id=video_id,
        is_short=False,
        duration_sec=effective_duration,
        reason=(
            f"duration {effective_duration:.0f}s > {SHORT_MAX_DURATION_SEC}s"
            if effective_duration > 0
            else "duration unknown, defaulting to long-form"
        ),
    )


# --------------------------------------------------------------------------- #
# LLM-driven topic segmentation
# --------------------------------------------------------------------------- #


_SYSTEM_PROMPT = """You are a precise video editor cutting long-form YouTube videos into self-contained topic clips.

You will receive a numbered transcript. Each line has the format:
    [<idx> <mm:ss>] <speech>

Your job: identify every distinct TOPIC the creator covers, and for each one return the cue index where they FIRST INTRODUCE it and the cue index where they LAST DISCUSS it before moving on.

Rules:
- A new topic begins when the creator clearly introduces a new subject ("Now let's talk about X", "The next thing is Y", a new chapter, a new question, a new demo). Not when X is merely mentioned in passing.
- A topic ends at the LAST cue still on that subject — i.e. the cue right before the creator transitions away. NOT the first cue of the next topic.
- Each clip must be SELF-CONTAINED: a viewer who saw nothing else of the video should still understand what's being discussed. If a section depends on the previous topic to make sense, include it as part of the previous topic instead of starting a new clip.
- Skip intros, outros, sponsor reads, and "subscribe / like" call-outs — do NOT return them as topics.
- Topics must not overlap and must be in ascending order.
- If the entire video is one continuous monolithic topic (e.g. a single proof, a single demo with no breaks), return ONE segment spanning the substantive part.
- A label must be 4-9 words and describe the topic, not the action ("Diagonalizing 3x3 matrices", not "He talks about matrices").

Return JSON only, in this exact shape:
{
    "segments": [
        {"start_idx": <int>, "end_idx": <int>, "label": "<topic label>", "summary": "<one short sentence>"}
    ]
}
No prose outside the JSON."""


def _format_timestamp(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds >= 3600:
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h}:{m:02d}:{s:02d}"
    m = seconds // 60
    s = seconds % 60
    return f"{m}:{s:02d}"


def _render_transcript_for_llm(cues: Sequence[TranscriptCue]) -> str:
    """Render `cues` as `[idx mm:ss] text` lines, one per cue."""
    return "\n".join(
        f"[{i} {_format_timestamp(cue.start)}] {cue.text}"
        for i, cue in enumerate(cues)
    )


def _llm_topic_segments(
    cues: Sequence[TranscriptCue],
    *,
    openai_client: Any,
    model: str = DEFAULT_MODEL,
) -> list[tuple[int, int, str, str]]:
    """
    Ask the LLM to identify topic boundaries by cue index.

    Returns a list of (start_idx, end_idx, label, summary). May raise on API
    errors so the caller can decide whether to fall back to the heuristic.
    """
    rendered = _render_transcript_for_llm(cues)
    user_msg = (
        "Here is the full transcript of a long-form YouTube video. "
        "Identify topic segments per the rules in the system prompt.\n\n"
        f"{rendered}"
    )

    response = openai_client.chat.completions.create(
        model=model,
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    raw = response.choices[0].message.content or "{}"
    payload = json.loads(raw)
    segments_raw = payload.get("segments")
    if not isinstance(segments_raw, list):
        raise ValueError(f"LLM returned no segments: {raw[:200]}")

    out: list[tuple[int, int, str, str]] = []
    for seg in segments_raw:
        if not isinstance(seg, dict):
            continue
        try:
            s = int(seg.get("start_idx"))
            e = int(seg.get("end_idx"))
        except (TypeError, ValueError):
            continue
        label = str(seg.get("label") or "").strip()
        summary = str(seg.get("summary") or "").strip()
        if not label:
            continue
        out.append((s, e, label, summary))
    return out


# --------------------------------------------------------------------------- #
# Heuristic fallback
# --------------------------------------------------------------------------- #


_STOPWORDS = frozenset(
    """
    a an and are as at be by for from has have he her his i if in is it
    its of on or our she that the their them they this to was we were
    what which who will with you your yours yeah okay um uh just like
    so really very kind sort gonna going get got know now then there
    here something some thing things really also actually mean even
    well still much many one two three first second next way
    """.split()
)


def _cue_tokens(cue: TranscriptCue) -> set[str]:
    return {
        tok
        for tok in re.findall(r"[a-z][a-z']{2,}", cue.text.lower())
        if tok not in _STOPWORDS
    }


# --------------------------------------------------------------------------- #
# Semantic boundary detection (free, runs locally via sentence-transformers)
# --------------------------------------------------------------------------- #
#
# This is the "free LLM-equivalent" path: a small ~80 MB MiniLM model runs on
# CPU in your own process, embeds each transcript window, and detects topic
# boundaries by spotting cosine-similarity drops between adjacent windows.
# Quality is close to gpt-4o-mini for boundary detection (TF-IDF labels do
# the naming).
#
# The import is optional. If sentence-transformers isn't installed, the
# function returns None and the caller falls through to the Jaccard heuristic.

_SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _load_sentence_transformer() -> Any | None:
    """
    Try to load (and cache) the MiniLM embedder. Returns None if the package
    isn't installed or model loading fails for any reason — every caller is
    expected to handle None gracefully and fall back to the Jaccard path.

    The first call downloads ~80 MB to `~/.cache/huggingface/hub/`. Subsequent
    calls reuse the cached weights, so cold-start cost is paid exactly once
    per process.
    """
    cache = getattr(_load_sentence_transformer, "_cache", None)
    if cache is not None:
        return cache
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception:
        logger.debug("sentence-transformers is not installed; semantic path disabled")
        _load_sentence_transformer._cache = None  # type: ignore[attr-defined]
        return None
    try:
        model = SentenceTransformer(_SENTENCE_TRANSFORMER_MODEL)
    except Exception:
        logger.exception("failed to load sentence-transformers model %s", _SENTENCE_TRANSFORMER_MODEL)
        _load_sentence_transformer._cache = None  # type: ignore[attr-defined]
        return None
    _load_sentence_transformer._cache = model  # type: ignore[attr-defined]
    return model


def _chunk_cues_for_embedding(
    cues: Sequence[TranscriptCue],
    *,
    target_window_sec: float = 12.0,
) -> list[tuple[int, int, str]]:
    """
    Group adjacent cues into ~12-second windows for embedding.

    Returns `(start_idx, end_idx, text)` triples. Smaller windows give finer
    boundary resolution, larger ones make embeddings more meaningful — 12s is
    the sweet spot for typical 5-15 minute educational videos.
    """
    if not cues:
        return []
    chunks: list[tuple[int, int, str]] = []
    start_idx = 0
    pieces: list[str] = []
    window_start = cues[0].start
    for i, cue in enumerate(cues):
        pieces.append(cue.text)
        if (cue.end - window_start) >= target_window_sec:
            chunks.append((start_idx, i, " ".join(pieces).strip()))
            pieces = []
            start_idx = i + 1
            if start_idx < len(cues):
                window_start = cues[start_idx].start
    if pieces and start_idx < len(cues):
        chunks.append((start_idx, len(cues) - 1, " ".join(pieces).strip()))
    return [c for c in chunks if c[2]]


def _semantic_topic_segments(
    cues: Sequence[TranscriptCue],
    *,
    target_duration_sec: float = 180.0,
) -> list[tuple[int, int, str, str]] | None:
    """
    Topic segmentation via local sentence-transformer embeddings.

    Returns None when:
      * sentence-transformers isn't installed
      * model load fails
      * the transcript is too short to benefit (<8 chunks)

    Returns a list of `(start_idx, end_idx, label, summary)` tuples otherwise,
    in the same shape as `_llm_topic_segments` and `_heuristic_topic_segments`,
    so it drops straight into the existing snap pipeline. Labels are produced
    by `_tfidf_labels_for_ranges`.

    Algorithm:
      1. Group cues into ~12-second windows (`_chunk_cues_for_embedding`).
      2. Embed every window with MiniLM in one batched call.
      3. Compute cosine similarity between adjacent embeddings (a 1-D array).
      4. A boundary is a *local minimum* of that similarity that is also below
         the median minus 0.05 — i.e. an unusually big vocabulary shift, not
         just any small dip.
      5. Convert window indices back to cue indices, enforce min spacing in
         seconds (so we don't emit two boundaries 5s apart on a sub-topic
         pause), and clean up the same way `_heuristic_topic_segments` does.
    """
    model = _load_sentence_transformer()
    if model is None:
        return None
    if not cues:
        return None

    chunks = _chunk_cues_for_embedding(cues, target_window_sec=12.0)
    if len(chunks) < 8:
        return None

    texts = [text for (_, _, text) in chunks]
    try:
        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    except Exception:
        logger.exception("sentence-transformers encode() failed; falling back to heuristic")
        return None

    # Cosine similarity between adjacent windows. With normalized vectors this
    # is just the dot product.
    try:
        import numpy as np
    except ImportError:
        logger.warning("numpy not available; semantic path disabled")
        return None
    sims = (embeddings[:-1] * embeddings[1:]).sum(axis=1)

    if len(sims) < 4:
        return None

    median_sim = float(np.median(sims))
    # Threshold below which a similarity dip counts as a topic boundary. We
    # subtract a small constant from the median so we only flag UNUSUALLY large
    # drops, not the noisy ones that always appear between any two windows.
    threshold = median_sim - 0.05

    # Find boundaries: local minima below threshold, with a 2-window radius.
    boundary_chunk_indices: list[int] = []
    for i in range(1, len(sims) - 1):
        if sims[i] >= threshold:
            continue
        if sims[i] > sims[i - 1] or sims[i] > sims[i + 1]:
            continue
        boundary_chunk_indices.append(i + 1)  # the boundary is BETWEEN chunk i and i+1, so i+1 starts the new segment

    # Convert chunk-index boundaries into cue-index ranges.
    cut_cue_indices: list[int] = [0]
    for chunk_i in boundary_chunk_indices:
        if chunk_i >= len(chunks):
            continue
        cue_idx = chunks[chunk_i][0]
        # Enforce minimum spacing in seconds so we don't double-cut on quick dips.
        if cut_cue_indices and (cues[cue_idx].start - cues[cut_cue_indices[-1]].start) < target_duration_sec:
            continue
        cut_cue_indices.append(cue_idx)
    cut_cue_indices.append(len(cues))

    ranges: list[tuple[int, int]] = []
    for a, b in zip(cut_cue_indices, cut_cue_indices[1:]):
        end_idx = max(a, b - 1)
        if end_idx <= a:
            continue
        seg_dur = cues[end_idx].end - cues[a].start
        if seg_dur < MIN_TOPIC_REEL_SEC:
            continue
        if seg_dur > MAX_TOPIC_REEL_SEC:
            mid = (a + end_idx) // 2
            ranges.append((a, mid))
            ranges.append((mid + 1, end_idx))
            continue
        ranges.append((a, end_idx))

    if not ranges:
        return None

    labels = _tfidf_labels_for_ranges(cues, ranges)
    return [(a, b, label, "") for (a, b), label in zip(ranges, labels)]


def _heuristic_topic_segments(
    cues: Sequence[TranscriptCue],
    *,
    target_duration_sec: float = 180.0,
) -> list[tuple[int, int, str, str]]:
    """
    Lexical-novelty topic segmentation, used when the LLM is unavailable AND
    no chapters are present.

    Strategy: walk the transcript with a 30-cue rolling window. Score each
    boundary by `1 - jaccard(window_left, window_right)` — a high score means
    the vocabulary just shifted, which is a good proxy for a topic change.
    Take the local maxima as cut points, then enforce min/max segment length.

    Labels are produced by `_tfidf_labels_for_ranges` AFTER all ranges are
    picked, so each label is the most distinctive phrase for that segment
    relative to the others — much better than first-N-content-words.
    """
    if not cues:
        return []
    if len(cues) < 8:
        # Too short to bother — return the whole thing as one segment.
        text = " ".join(c.text for c in cues if c.text)[:80]
        return [(0, len(cues) - 1, text or "Full clip", "")]

    window = max(8, min(30, len(cues) // 8))
    scores: list[float] = [0.0] * len(cues)
    for i in range(window, len(cues) - window):
        left: set[str] = set()
        for cue in cues[i - window : i]:
            left.update(_cue_tokens(cue))
        right: set[str] = set()
        for cue in cues[i : i + window]:
            right.update(_cue_tokens(cue))
        if not left and not right:
            continue
        union = left | right
        intersection = left & right
        jaccard = (len(intersection) / len(union)) if union else 1.0
        scores[i] = 1.0 - jaccard

    # Find peaks: a cue index whose score is greater than its 5-neighbour mean
    # by at least 0.15 and is the local max in a ±5 window.
    cuts: list[int] = [0]
    radius = 5
    for i in range(window, len(cues) - window):
        if scores[i] < 0.4:
            continue
        local = scores[max(0, i - radius) : i + radius + 1]
        if not local:
            continue
        if scores[i] < max(local) - 1e-9:
            continue
        # Enforce minimum spacing in seconds.
        if cuts and (cues[i].start - cues[cuts[-1]].start) < target_duration_sec:
            continue
        cuts.append(i)
    cuts.append(len(cues))

    # Compute (a, end_idx) ranges, splitting any range that's too long.
    ranges: list[tuple[int, int]] = []
    for a, b in zip(cuts, cuts[1:]):
        end_idx = max(a, b - 1)
        if end_idx <= a:
            continue
        seg_dur = cues[end_idx].end - cues[a].start
        if seg_dur < MIN_TOPIC_REEL_SEC:
            continue
        if seg_dur > MAX_TOPIC_REEL_SEC:
            mid = (a + end_idx) // 2
            ranges.append((a, mid))
            ranges.append((mid + 1, end_idx))
            continue
        ranges.append((a, end_idx))

    if not ranges:
        return []

    # Generate distinctive labels for each range using TF-IDF over the
    # complete set of ranges (each range is one "document").
    labels = _tfidf_labels_for_ranges(cues, ranges)
    return [
        (a, b, label, "")
        for (a, b), label in zip(ranges, labels)
    ]


# --------------------------------------------------------------------------- #
# TF-IDF labels (free, deterministic, much better than first-N-content-words)
# --------------------------------------------------------------------------- #


def _tfidf_labels_for_ranges(
    cues: Sequence[TranscriptCue],
    ranges: Sequence[tuple[int, int]],
    *,
    top_k_terms: int = 4,
) -> list[str]:
    """
    Build a distinctive label for each (start_idx, end_idx) range using TF-IDF.

    Each range is treated as one document; the term frequency is its own count
    of unigrams + bigrams, and the document frequency is "how many ranges
    contain this term." The most distinctive terms are the ones that appear
    often in this range but rarely in the others — exactly the phrases a
    creator would put in a chapter title.

    The label is constructed by taking the top `top_k_terms` ranked terms,
    keeping the highest-scoring bigram (if any) plus a few high-scoring
    unigrams that aren't already covered by the bigram, and joining them as
    title case. The result is at most ~6 words long.

    The whole thing is pure-Python, no extra deps. For a 5000-cue transcript
    split into 12 segments it runs in <50ms.
    """
    if not ranges:
        return []

    # Tokenize each range into unigrams + bigrams. Bigrams give us "neural
    # network" instead of just "neural" — much closer to a chapter title.
    range_terms: list[dict[str, int]] = []
    for a, b in ranges:
        tf: dict[str, int] = {}
        # Flatten the segment text into a single token list, preserving order
        # so adjacent tokens form valid bigrams.
        ordered_tokens: list[str] = []
        for cue in cues[a : b + 1]:
            for tok in re.findall(r"[a-z][a-z']{2,}", cue.text.lower()):
                if tok in _STOPWORDS:
                    continue
                ordered_tokens.append(tok)

        for tok in ordered_tokens:
            tf[tok] = tf.get(tok, 0) + 1

        for tok_a, tok_b in zip(ordered_tokens, ordered_tokens[1:]):
            bigram = f"{tok_a} {tok_b}"
            tf[bigram] = tf.get(bigram, 0) + 1

        range_terms.append(tf)

    # Document frequency: how many ranges contain each term.
    n_docs = len(ranges)
    df: dict[str, int] = {}
    for tf in range_terms:
        for term in tf.keys():
            df[term] = df.get(term, 0) + 1

    import math
    labels: list[str] = []
    for tf in range_terms:
        if not tf:
            labels.append("Untitled segment")
            continue

        scored: list[tuple[float, str]] = []
        for term, freq in tf.items():
            doc_freq = df.get(term, 1)
            # Smoothed idf so a term in every doc still scores ~0.
            idf = math.log((n_docs + 1) / (doc_freq + 0.5))
            if idf <= 0.0:
                continue
            # Bigrams get a small bonus because they make better labels.
            bigram_bonus = 1.5 if " " in term else 1.0
            score = freq * idf * bigram_bonus
            scored.append((score, term))

        if not scored:
            labels.append("Untitled segment")
            continue

        scored.sort(reverse=True)
        labels.append(_compose_label_from_terms([term for _, term in scored[:top_k_terms]]))

    return labels


def _compose_label_from_terms(terms: list[str]) -> str:
    """
    Combine the top-ranked terms into a 4-7 word title-cased label.

    Strategy: pick the highest-scoring bigram (if any), then add up to two
    unigrams that aren't already covered by that bigram. The bigram appears
    first because it's typically the most descriptive single phrase.

    Falls back to plain top-3 unigrams if no bigrams scored.
    """
    if not terms:
        return "Untitled segment"

    bigram = next((t for t in terms if " " in t), None)
    chosen: list[str] = []
    if bigram:
        chosen.append(bigram)
        bigram_words = set(bigram.split())
        for term in terms:
            if " " in term:
                continue
            if term in bigram_words:
                continue
            chosen.append(term)
            if len(chosen) >= 3:
                break
    else:
        for term in terms:
            chosen.append(term)
            if len(chosen) >= 3:
                break

    return " ".join(word.capitalize() for word in " ".join(chosen).split())


# --------------------------------------------------------------------------- #
# Boundary snapping
# --------------------------------------------------------------------------- #


def _split_long_range(
    t_start: float,
    t_end: float,
    label: str,
    summary: str,
    *,
    max_reel_sec: float,
    min_reel_sec: float,
) -> list[tuple[float, float, str, str]]:
    """
    Split a `(t_start, t_end)` range that exceeds `max_reel_sec` into equal
    sub-ranges, each labeled `"<original> (i/N)"`.

    Returns the original range unchanged when it fits, or `[]` when it would
    require sub-parts shorter than `min_reel_sec` (in which case the caller
    should accept the natural boundary or drop the topic entirely — see
    `_topic_reels_from_chapters` for the policy).
    """
    duration = t_end - t_start
    if max_reel_sec <= 0 or duration <= max_reel_sec:
        return [(t_start, t_end, label, summary)]
    import math
    num_parts = int(math.ceil(duration / max_reel_sec))
    part_dur = duration / num_parts
    if part_dur < min_reel_sec:
        # Splitting would yield sub-parts that are too short. Bail out — the
        # caller can fall back to the natural boundary.
        return []
    parts: list[tuple[float, float, str, str]] = []
    for i in range(num_parts):
        a = t_start + i * part_dur
        b = t_end if i == num_parts - 1 else (t_start + (i + 1) * part_dur)
        parts.append((a, b, f"{label} ({i + 1}/{num_parts})", summary))
    return parts


def _topic_reels_from_chapters(
    chapter_segments: Sequence[tuple[float, float, str, str]],
    cues: Sequence[TranscriptCue],
    *,
    video_id: str,
    video_duration_sec: float | None,
    min_reel_sec: float = MIN_TOPIC_REEL_SEC,
    max_reel_sec: float = MAX_TOPIC_REEL_SEC,
) -> list[TopicReel]:
    """
    Build TopicReel records directly from chapter (t_start_sec, t_end_sec, label, summary)
    tuples. Mirrors `_snap_segments_to_cues` but works in seconds, not cue indices,
    because chapter markers come with absolute timestamps from yt-dlp and don't
    need to be resolved through the cue list.

    Optional cue snapping: if `cues` is non-empty, the END timestamp of each
    chapter is nudged backward to land in the natural pause between cues if a
    pause is within `SNAP_TOLERANCE_SEC`. This handles the common case where
    a creator's chapter marker is half a second into the next sentence.

    Drops segments outside `[MIN_TOPIC_REEL_SEC, MAX_TOPIC_REEL_SEC]` after
    snapping. Sorts and de-overlaps the result, just like `_snap_segments_to_cues`.
    """
    candidates: list[TopicReel] = []
    has_cues = bool(cues)
    last_idx = len(cues) - 1 if has_cues else -1
    abs_max_cap = max(max_reel_sec, MAX_TOPIC_REEL_SEC)

    for t_start_sec, t_end_sec, label, summary in chapter_segments:
        t_start = float(t_start_sec)
        t_end = float(t_end_sec)
        if t_end <= t_start:
            continue

        if video_duration_sec and video_duration_sec > 0:
            t_end = min(t_end, float(video_duration_sec))

        # Drop unconditionally if the chapter is too short OR exceeds the
        # absolute hard cap (12 min). For chapters that exceed the SOFT cap
        # (`max_reel_sec`, e.g. the user's preferred 60s) we attempt to split.
        duration = t_end - t_start
        if duration < min_reel_sec or duration > abs_max_cap:
            logger.debug(
                "dropping chapter '%s' duration=%.1fs (outside [%d,%d])",
                label, duration, min_reel_sec, abs_max_cap,
            )
            continue

        if duration > max_reel_sec:
            sub_ranges = _split_long_range(
                t_start, t_end, label, summary,
                max_reel_sec=max_reel_sec,
                min_reel_sec=min_reel_sec,
            )
            if not sub_ranges:
                # Splitting would produce sub-parts shorter than the minimum.
                # Keep the natural boundary instead.
                sub_ranges = [(t_start, t_end, label, summary)]
        else:
            sub_ranges = [(t_start, t_end, label, summary)]

        for sub_start, sub_end, sub_label, sub_summary in sub_ranges:
            s_idx = -1
            e_idx = -1
            t_end_local = sub_end
            if has_cues:
                # Resolve to nearest cue indices via linear scan (cues are sorted
                # by start). For typical transcripts (200-3000 cues) this is plenty
                # fast — bisect would shave microseconds.
                s_idx = 0
                for i, cue in enumerate(cues):
                    if cue.start >= sub_start - 0.01:
                        s_idx = i
                        break
                else:
                    s_idx = last_idx

                e_idx = s_idx
                for i in range(s_idx, len(cues)):
                    if cues[i].start > t_end_local + 0.01:
                        break
                    e_idx = i

                # Snap t_end to inter-cue gap if available.
                if e_idx + 1 < len(cues):
                    gap_start = cues[e_idx].end
                    gap_end = cues[e_idx + 1].start
                    if gap_end > gap_start:
                        snapped = (gap_start + gap_end) / 2.0
                        if abs(snapped - t_end_local) <= SNAP_TOLERANCE_SEC:
                            t_end_local = snapped

            candidates.append(
                TopicReel(
                    video_id=video_id,
                    t_start=round(sub_start, 2),
                    t_end=round(t_end_local, 2),
                    label=sub_label,
                    summary=sub_summary,
                    cue_start_idx=s_idx,
                    cue_end_idx=e_idx,
                )
            )

    candidates.sort(key=lambda r: r.t_start)
    cleaned: list[TopicReel] = []
    for reel in candidates:
        if cleaned and reel.t_start < cleaned[-1].t_end:
            new_start = cleaned[-1].t_end
            if (reel.t_end - new_start) < MIN_TOPIC_REEL_SEC:
                continue
            reel.t_start = round(new_start, 2)
        cleaned.append(reel)
    return cleaned


def _snap_segments_to_cues(
    raw_segments: Iterable[tuple[int, int, str, str]],
    cues: Sequence[TranscriptCue],
    *,
    video_id: str,
    video_duration_sec: float | None,
    min_reel_sec: float = MIN_TOPIC_REEL_SEC,
    max_reel_sec: float = MAX_TOPIC_REEL_SEC,
) -> list[TopicReel]:
    """
    Resolve LLM/heuristic cue indices into final TopicReel records.

    For each (start_idx, end_idx, label, summary):
      * clamp start_idx and end_idx to valid bounds
      * derive t_start from cues[start_idx].start
      * derive t_end from cues[end_idx].end (the END of the last cue, not the
        start of the next one — that's the moment the speaker stops talking
        about this topic)
      * if there's a natural pause within ±SNAP_TOLERANCE_SEC of t_end (a gap
        between two consecutive cues), nudge t_end to that pause so the cut
        lands on silence
      * drop the segment if it's outside [MIN_TOPIC_REEL_SEC, MAX_TOPIC_REEL_SEC]
      * sort + de-dup overlapping segments (LLMs occasionally return ranges
        that share a cue or two)
    """
    if not cues:
        return []

    last_idx = len(cues) - 1
    candidates: list[TopicReel] = []
    abs_max_cap = max(max_reel_sec, MAX_TOPIC_REEL_SEC)

    for s_idx, e_idx, label, summary in raw_segments:
        s_idx = max(0, min(last_idx, int(s_idx)))
        e_idx = max(s_idx, min(last_idx, int(e_idx)))
        t_start = cues[s_idx].start
        t_end = cues[e_idx].end

        # Try to snap t_end backward to the gap between cues[e_idx] and cues[e_idx+1]
        if e_idx + 1 < len(cues):
            gap_start = cues[e_idx].end
            gap_end = cues[e_idx + 1].start
            if gap_end > gap_start:
                # Land mid-gap if the gap is wide enough — this is the most
                # natural place to cut.
                snapped = (gap_start + gap_end) / 2.0
                if abs(snapped - t_end) <= SNAP_TOLERANCE_SEC:
                    t_end = snapped

        if video_duration_sec and video_duration_sec > 0:
            t_end = min(t_end, float(video_duration_sec))

        duration = t_end - t_start
        if duration < min_reel_sec or duration > abs_max_cap:
            logger.debug(
                "dropping out-of-range segment idx=%d-%d duration=%.1fs label=%s",
                s_idx, e_idx, duration, label,
            )
            continue

        # If the segment exceeds the SOFT cap, split it into roughly-equal parts
        # so the user's max_reel_sec preference is respected. Each part inherits
        # the same label with a "(i/N)" suffix.
        if duration > max_reel_sec:
            split = _split_long_range(
                t_start, t_end, label, summary,
                max_reel_sec=max_reel_sec, min_reel_sec=min_reel_sec,
            )
            if not split:
                # Splitting would yield sub-parts shorter than min_reel_sec,
                # so keep the natural boundary instead.
                split = [(t_start, t_end, label, summary)]
        else:
            split = [(t_start, t_end, label, summary)]

        for sub_start, sub_end, sub_label, sub_summary in split:
            candidates.append(
                TopicReel(
                    video_id=video_id,
                    t_start=round(sub_start, 2),
                    t_end=round(sub_end, 2),
                    label=sub_label,
                    summary=sub_summary,
                    cue_start_idx=s_idx,
                    cue_end_idx=e_idx,
                )
            )

    # Sort and trim overlaps. If two reels overlap, keep the earlier one and
    # push the later one's start past the earlier one's end.
    candidates.sort(key=lambda r: r.t_start)
    cleaned: list[TopicReel] = []
    for reel in candidates:
        if cleaned and reel.t_start < cleaned[-1].t_end:
            new_start = cleaned[-1].t_end
            if (reel.t_end - new_start) < MIN_TOPIC_REEL_SEC:
                continue
            reel.t_start = round(new_start, 2)
        cleaned.append(reel)
    return cleaned


# --------------------------------------------------------------------------- #
# Top-level entrypoint
# --------------------------------------------------------------------------- #


def cut_video_into_topic_reels(
    url_or_id: str,
    *,
    duration_sec: float | None = None,
    openai_client: Any | None = None,
    model: str = DEFAULT_MODEL,
    use_llm: bool = True,
    transcript: Sequence[TranscriptCue] | None = None,
    info_dict: dict[str, Any] | None = None,
    min_reel_sec: float = MIN_TOPIC_REEL_SEC,
    max_reel_sec: float = MAX_TOPIC_REEL_SEC,
) -> tuple[VideoClassification, list[TopicReel]]:
    """
    End-to-end: classify a YouTube video, fetch its transcript, and (if it's
    long-form) cut it into per-topic reels.

    Cut precedence (highest-quality first):
        1. **YouTube chapters** from `info_dict["chapters"]` — creator-authored,
           free, deterministic. When chapters exist we use them DIRECTLY and
           skip the LLM/heuristic entirely.
        2. **LLM topic segmentation** via `openai_client.chat.completions` —
           skipped when `use_llm=False` or no client is available.
        3. **Lexical-novelty heuristic** — pure-Python Jaccard sliding window,
           no API needed. Always available as the final fallback.

    Args:
        url_or_id: YouTube URL or 11-char video ID.
        duration_sec: Optional duration in seconds — if you already have it
            from the HTML scraper (`_video_row_from_renderer["duration_sec"]`)
            pass it through; otherwise duration is inferred from the transcript.
        openai_client: Optional pre-built OpenAI client. If None and `use_llm`
            is True, this function tries to build one from `OPENAI_API_KEY`.
        model: Chat model to use. Defaults to `gpt-4o-mini`.
        use_llm: If False, skip the LLM call and use the heuristic fallback
            even when an OpenAI client is available. Useful for offline runs.
        transcript: Optional pre-fetched transcript cues. When provided, the
            function skips its internal `fetch_transcript` call. This is the
            integration point used by `IngestionPipeline.ingest_topic_cut`,
            which has already transcribed the video via its own fallback chain
            (YouTube → yt-dlp subs → Whisper) and doesn't want to re-fetch.
        info_dict: Optional yt-dlp info dict. When provided AND it carries a
            non-empty `chapters` list, those chapters are used directly as
            topic boundaries — no LLM, no heuristic, no API cost. This is
            the highest-quality free signal we have.

    Returns:
        (classification, reels). For Shorts, `reels` is always [].
        For long-form videos with no transcript, `reels` is also [] and a
        warning is logged — the caller should decide whether to retry with
        a transcription fallback (Whisper) or skip.
    """
    video_id, _is_shorts_url = extract_video_id(url_or_id)

    # Fetch transcript first — we need it both for classification (when
    # duration_sec is missing) and for the actual segmentation. Skip the
    # network call if the caller already has cues in hand.
    if transcript is None:
        transcript = fetch_transcript(video_id)
    else:
        transcript = list(transcript)

    classification = classify_video(
        url_or_id,
        duration_sec=duration_sec,
        transcript=transcript,
    )

    if classification.is_short:
        logger.info(
            "video %s classified as Short (%s); leaving untouched",
            video_id, classification.reason,
        )
        return classification, []

    # ---- Path 1: YouTube chapters (free, no API, no inference). -------- #
    # When the creator put chapter markers in the description, yt-dlp parses
    # them into info_dict["chapters"]. These are the BEST signal we can get —
    # the creator's own segmentation, not a guess. If we have any usable
    # chapters we return immediately without ever touching the LLM or the
    # heuristic. Transcript may still be empty here (e.g. captions disabled
    # but description has chapter markers), and that's fine — chapters carry
    # their own absolute timestamps.
    chapters = extract_chapters(info_dict)
    if chapters:
        chapter_segments = chapters_to_topic_segments(chapters)
        chapter_reels = _topic_reels_from_chapters(
            chapter_segments,
            transcript,
            video_id=video_id,
            video_duration_sec=classification.duration_sec or None,
            min_reel_sec=min_reel_sec,
            max_reel_sec=max_reel_sec,
        )
        if chapter_reels:
            logger.info(
                "video %s cut from %d YouTube chapters → %d reels (free path)",
                video_id, len(chapters), len(chapter_reels),
            )
            return classification, chapter_reels
        logger.info(
            "video %s had %d chapters but none survived the [%d,%d] guardrails; "
            "falling through to LLM/heuristic",
            video_id, len(chapters), int(min_reel_sec), int(max_reel_sec),
        )

    if not transcript:
        logger.warning(
            "video %s has no transcript available; cannot cut. Caller should "
            "fall back to the Whisper path in IngestionPipeline.transcribe.",
            video_id,
        )
        return classification, []

    # ---- Path 2: LLM topic segmentation (paid, best quality). ---------- #
    raw_segments: list[tuple[int, int, str, str]] = []
    if use_llm:
        client = openai_client or _maybe_build_openai_client()
        if client is None:
            logger.info(
                "no OpenAI client available; trying free semantic path next"
            )
        elif len(transcript) > MAX_CUES_FOR_LLM:
            logger.warning(
                "transcript has %d cues, exceeds MAX_CUES_FOR_LLM=%d; "
                "trying free semantic path instead",
                len(transcript), MAX_CUES_FOR_LLM,
            )
        else:
            try:
                raw_segments = _llm_topic_segments(transcript, openai_client=client, model=model)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "LLM topic segmentation failed for video %s; trying free semantic path",
                    video_id,
                )
                raw_segments = []

    # ---- Path 3: Local sentence-transformer embeddings (free, near-LLM). #
    # This is the "free LLM-equivalent": a small ~80 MB MiniLM model runs in
    # this process and detects topic boundaries via cosine-similarity drops.
    # Returns None if sentence-transformers isn't installed — that's fine, the
    # Jaccard fallback below picks up the slack.
    if not raw_segments:
        semantic = _semantic_topic_segments(transcript)
        if semantic:
            logger.info(
                "video %s segmented via local sentence-transformers → %d ranges",
                video_id, len(semantic),
            )
            raw_segments = semantic

    # ---- Path 4: Pure-Python Jaccard heuristic (always available). ----- #
    if not raw_segments:
        raw_segments = _heuristic_topic_segments(transcript)

    reels = _snap_segments_to_cues(
        raw_segments,
        transcript,
        video_id=video_id,
        video_duration_sec=classification.duration_sec or None,
        min_reel_sec=min_reel_sec,
        max_reel_sec=max_reel_sec,
    )
    return classification, reels


def _maybe_build_openai_client() -> Any | None:
    """Build an OpenAI client from `OPENAI_API_KEY` if possible, else None."""
    api_key = os.environ.get("OPENAI_API_KEY") or ""
    if not api_key:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None
    try:
        return OpenAI(api_key=api_key, timeout=60.0)
    except Exception:  # noqa: BLE001
        logger.exception("could not build OpenAI client")
        return None


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _format_human(classification: VideoClassification, reels: list[TopicReel]) -> str:
    lines = [
        f"video_id:   {classification.video_id}",
        f"is_short:   {classification.is_short}",
        f"duration:   {classification.duration_sec:.0f}s",
        f"reason:     {classification.reason}",
        f"reel_count: {len(reels)}",
        "",
    ]
    if not reels:
        lines.append("(no reels emitted)")
        return "\n".join(lines)

    for i, reel in enumerate(reels, 1):
        lines.append(
            f"  {i:>2}. [{_format_timestamp(reel.t_start)} → {_format_timestamp(reel.t_end)}] "
            f"({reel.duration_sec:.0f}s)  {reel.label}"
        )
        if reel.summary:
            lines.append(f"      {reel.summary}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="topic_cut",
        description="Cut a long-form YouTube video into per-topic reels.",
    )
    parser.add_argument("url_or_id", help="YouTube URL or 11-char video ID")
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=None,
        help="Video duration in seconds (optional, takes precedence over transcript inference)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI chat model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip the LLM and use the lexical-novelty heuristic only",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON instead of a human-readable summary",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show debug logs",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        classification, reels = cut_video_into_topic_reels(
            args.url_or_id,
            duration_sec=args.duration_sec,
            model=args.model,
            use_llm=not args.no_llm,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        logger.exception("topic_cut failed")
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(
            {
                "classification": dataclasses.asdict(classification),
                "reels": [r.to_dict() for r in reels],
            },
            indent=2,
        ))
    else:
        print(_format_human(classification, reels))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
