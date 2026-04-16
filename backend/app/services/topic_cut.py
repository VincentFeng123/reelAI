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

from .transcript_validation import TranscriptQuality, validate_transcript

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
SNAP_TOLERANCE_SEC = 1.5

# Default LLM model. Mirrors `Settings.openai_chat_model`. Cheap, fast, fits
# any single-video transcript in one call.
DEFAULT_MODEL = "gpt-4o-mini"

# Hard cap on transcript cues sent to the LLM. ~6000 cues ≈ 8 hours of speech;
# beyond that we fall back to the heuristic (and warn) rather than risk a
# token-limit error. Raised from 3000 to 6000 — Gemini Flash and Llama 3
# handle large contexts well.
MAX_CUES_FOR_LLM = 6000


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
    relevance_score: float | None = None
    # Phase A.4: tier set by the ClipBoundaryEngine when it refines t_start/t_end.
    # Values: "legacy" (pre-engine), "chapter", "sentence", "precise", "degraded".
    boundary_quality: str = "legacy"

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.t_end - self.t_start)

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["duration_sec"] = round(self.duration_sec, 2)
        d["t_start"] = round(self.t_start, 2)
        d["t_end"] = round(self.t_end, 2)
        if self.relevance_score is not None:
            d["relevance_score"] = round(self.relevance_score, 3)
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


_SYSTEM_PROMPT_LEGACY = """You are a precise video editor cutting long-form YouTube videos into self-contained topic clips.

You will receive a numbered transcript. Each line has the format:
    [<idx> <mm:ss>] <speech>

Your job: identify every distinct TOPIC the creator covers, and for each one return the cue index where they FIRST INTRODUCE it and the cue index where they LAST DISCUSS it before moving on.

Rules for TOPIC INTRODUCTIONS (start_idx):
- A topic INTRODUCTION is when the creator begins SUSTAINED discussion of a new subject. Strong signals include phrases like "Now let me show you", "The next concept is", "Moving on to", "Let's dive into", "So what is X?", a new chapter, a new question, or a new demo.
- A mere passing reference ("as we discussed earlier", "similar to X", "which relates to Y") is NOT a topic introduction. The creator must be setting context for a new discussion, not referencing a previous one.
- `start_idx` must be the FIRST cue where the creator begins the new topic — NOT the preceding segue cue like "let's move on" or "alright so". Start at the actual substance.

Rules for TOPIC ENDINGS (end_idx):
- `end_idx` is the LAST cue where the creator is ACTIVELY discussing this topic — NOT the first cue of the next topic.
- Look for transition signals: "Alright so", "Moving on", "Next up", "So that covers", "Now let's talk about something else", or a noticeable pause followed by new vocabulary.
- Place end_idx on the cue BEFORE such a transition signal, not on the transition itself.
- Prefer ending at a complete sentence rather than mid-thought — if the creator finishes their point 1-2 cues before the transition phrase, end there.

General rules:
- Each segment should span at least 10 cues of substantive content. If you have fewer, consider whether it's a passing reference rather than a true topic.
- Each clip must be SELF-CONTAINED: a viewer should understand the discussion without seeing the rest of the video.
- Skip intros, outros, sponsor reads, and "subscribe / like" call-outs — do NOT return them as topics.
- Topics must not overlap and must be in ascending order.
- If the entire video is one continuous monolithic topic, return ONE segment spanning the substantive part.
- A label must be 4-9 words and describe the topic, not the action ("Diagonalizing 3x3 matrices", not "He talks about matrices").
- Prefer tighter, more precise boundaries — it is better to trim a transition sentence than to include 10 seconds of the wrong topic.

Return JSON only, in this exact shape:
{
    "segments": [
        {"start_idx": <int>, "end_idx": <int>, "label": "<topic label>", "summary": "<one short sentence>", "confidence": <float 0.0-1.0>}
    ]
}
No prose outside the JSON."""


def _build_system_prompt(*, query: str | None = None) -> str:
    """Build the timestamp-based topic segmentation system prompt.

    When *query* is provided the prompt also instructs the LLM to return a
    ``relevance_score`` (0.0-1.0) per segment indicating how relevant it is
    to the user's search.
    """
    query_block = ""
    if query:
        query_block = f"""
The user is searching for: "{query}"
For each segment, also return a "relevance_score" (float 0.0-1.0) indicating
how relevant this segment is to the user's query.
  1.0 = directly and substantively addresses the query topic.
  0.5 = partially related or tangentially relevant.
  0.0 = completely unrelated.
Score based on semantic relevance, not just keyword overlap."""

    relevance_field = ', "relevance_score": <float 0.0-1.0>' if query else ""

    return f"""You are a precise video editor cutting long-form YouTube videos into self-contained topic clips.

You will receive a transcript. Each line has the format:
    [<start_seconds>-<end_seconds>s] <speech>

For example: [45.2-48.7s] And that brings us to the concept of gradient descent.

Your job: identify every distinct TOPIC the creator covers, and for each one return the EXACT start and end timestamps where they discuss it.
{query_block}
Rules for TOPIC STARTS (start_time):
- A topic START is when the creator begins SUSTAINED discussion of a new subject. Strong signals include phrases like "Now let me show you", "The next concept is", "Moving on to", "Let's dive into", "So what is X?", a new chapter, a new question, or a new demo.
- A mere passing reference ("as we discussed earlier", "similar to X", "which relates to Y") is NOT a topic start. The creator must be setting context for a new discussion, not referencing a previous one.
- start_time must be the timestamp of the FIRST cue where the creator begins the new topic — NOT the preceding segue cue like "let's move on" or "alright so". Start at the actual substance.

Rules for TOPIC ENDS (end_time):
- end_time is the END timestamp of the LAST cue where the creator is ACTIVELY discussing this topic — NOT the start of the next topic's first cue.
- Look for transition signals: "Alright so", "Moving on", "Next up", "So that covers", "Now let's talk about something else", or a noticeable pause followed by new vocabulary.
- Place end_time on the cue BEFORE such a transition signal, not on the transition itself.
- Prefer ending at a complete sentence rather than mid-thought.

IMPORTANT: start_time and end_time MUST be exact timestamp values that appear in the transcript's [X.X-Y.Ys] ranges. For start_time, use the START value of a cue. For end_time, use the END value of a cue. Do not interpolate or invent timestamps.

Edge-case handling:
- REVISITED TOPICS: If the creator returns to a topic they discussed earlier, treat each occurrence as a SEPARATE segment. Append "(continued)" or "(part 2)" to the label for later occurrences. Do NOT merge non-contiguous discussions.
- BRIEF TANGENTS: A tangent shorter than roughly 10 seconds that interrupts a topic should be ABSORBED into the surrounding topic, not treated as its own segment. Only create a separate segment for tangents that last at least 15 seconds with substantive content.
- INTROS: Skip any intro (channel branding, "hey guys welcome back", recap of previous episodes). If the intro contains a preview or table of contents, skip it — the actual discussion appears later.
- OUTROS: Skip any outro ("don't forget to like and subscribe", "thanks for watching", sign-off).
- SPONSOR READS: Skip sponsor segments ("this video is brought to you by", "use code X for 20% off"). These are NOT topics.

General rules:
- Each clip must be SELF-CONTAINED: a viewer should understand the discussion without seeing the rest of the video.
- Topics must not overlap and must be in ascending chronological order.
- If the entire video is one continuous monolithic topic, return ONE segment spanning the substantive part.
- A label must be 4-9 words and describe the topic, not the action ("Diagonalizing 3x3 matrices", not "He talks about matrices").
- Prefer tighter, more precise boundaries — it is better to trim a transition sentence than to include 10 seconds of the wrong topic.

Return JSON only, in this exact shape:
{{
    "segments": [
        {{"start_time": <float>, "end_time": <float>, "topic_label": "<4-9 word label>", "summary": "<one short sentence>", "confidence": <float 0.0-1.0>{relevance_field}}}
    ]
}}
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
    """Render cues as `[start-end s] text` lines with precise timestamps.

    Format: ``[45.2-48.7s] Hello everyone...``

    Both start and end times are exposed so the LLM can pick precise
    boundaries for both segment starts and ends.
    """
    return "\n".join(
        f"[{cue.start:.1f}-{cue.end:.1f}s] {cue.text}"
        for cue in cues
    )


# --------------------------------------------------------------------------- #
# Transcript coverage validation
# --------------------------------------------------------------------------- #
# Delegates to the shared ``transcript_validation`` module. The local alias
# ``TranscriptValidation`` and wrapper ``_validate_transcript_coverage`` keep
# existing callers working without changes.

# Re-export the shared type under the old name used by callers in this module.
TranscriptValidation = TranscriptQuality


def _validate_transcript_coverage(
    cues: Sequence[TranscriptCue],
    video_duration_sec: float | None,
    *,
    min_coverage_ratio: float = 0.80,
    max_gap_sec: float = 30.0,
    max_first_cue_delay_sec: float = 10.0,
) -> TranscriptQuality:
    """Check whether the transcript adequately covers the video.

    Delegates to the shared :func:`validate_transcript` module.  Returns a
    :class:`TranscriptQuality` (aliased as ``TranscriptValidation`` for
    backward compatibility) with coverage stats.
    """
    quality = validate_transcript(
        cues,
        video_duration_sec,
        min_coverage=min_coverage_ratio,
        max_gap_sec=max_gap_sec,
        max_first_delay_sec=max_first_cue_delay_sec,
    )
    for w in quality.warnings:
        logger.warning("transcript validation: %s", w)
    return quality


# -- Timestamp-based segment tuple:
#    (start_time, end_time, label, summary, relevance_score_or_None)
_SegmentTuple = tuple[float, float, str, str, float | None]


def _strip_code_fences(raw: str) -> str:
    """Strip markdown code fences that some LLMs wrap around JSON."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    return raw


def _parse_llm_segments_json(raw: str) -> list[_SegmentTuple]:
    """Parse timestamp-based JSON from any LLM into segment tuples.

    Expected JSON shape::

        {"segments": [{"start_time": 45.2, "end_time": 312.8,
                        "topic_label": "...", "summary": "...",
                        "confidence": 0.9, "relevance_score": 0.85}]}

    Returns ``(start_time, end_time, label, summary, relevance_score)``
    tuples.  ``relevance_score`` is ``None`` when the field is absent.
    """
    raw = _strip_code_fences(raw)
    payload = json.loads(raw)
    segments_raw = payload.get("segments")
    if not isinstance(segments_raw, list):
        raise ValueError(f"LLM returned no segments: {raw[:200]}")

    out: list[_SegmentTuple] = []
    for seg in segments_raw:
        if not isinstance(seg, dict):
            continue
        try:
            s = float(seg.get("start_time"))
            e = float(seg.get("end_time"))
        except (TypeError, ValueError):
            continue
        if s < 0 or e < 0 or s >= e:
            continue
        label = str(seg.get("topic_label") or seg.get("label") or "").strip()
        summary = str(seg.get("summary") or "").strip()
        if not label:
            continue
        rel = seg.get("relevance_score")
        relevance: float | None = None
        if rel is not None:
            try:
                relevance = max(0.0, min(1.0, float(rel)))
            except (TypeError, ValueError):
                pass
        out.append((s, e, label, summary, relevance))
    return out


def _llm_topic_segments(
    cues: Sequence[TranscriptCue],
    *,
    openai_client: Any,
    model: str = DEFAULT_MODEL,
    query: str | None = None,
) -> list[_SegmentTuple]:
    """
    Ask the LLM to identify topic boundaries by timestamp.

    Returns a list of (start_time, end_time, label, summary, relevance).
    May raise on API errors so the caller can fall back.
    """
    rendered = _render_transcript_for_llm(cues)
    system_prompt = _build_system_prompt(query=query)
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
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
    )
    raw = response.choices[0].message.content or "{}"
    return _parse_llm_segments_json(raw)


# --------------------------------------------------------------------------- #
# Gemini Flash (free tier: 15 RPM, 1M tokens/day)
# --------------------------------------------------------------------------- #


def _collect_gemini_api_keys() -> list[str]:
    """Collect all Gemini API keys from environment variables.

    Reads GEMINI_API_KEY, GEMINI_API_KEY_2, GEMINI_API_KEY_3, ...
    Returns a list of non-empty keys in order.
    """
    keys: list[str] = []
    primary = os.environ.get("GEMINI_API_KEY") or ""
    if primary:
        keys.append(primary)
    for i in range(2, 20):  # support up to 19 backup keys
        k = os.environ.get(f"GEMINI_API_KEY_{i}") or ""
        if k:
            keys.append(k)
    return keys


# Track which key index to start with next (rotates on rate-limit).
_gemini_key_offset: int = 0


def _build_gemini_client(api_key: str | None = None) -> Any | None:
    """Build a Google Gemini client.

    If *api_key* is provided, uses that key directly. Otherwise reads
    the primary GEMINI_API_KEY env var.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY") or ""
    if not key:
        return None
    try:
        import google.generativeai as genai
    except ImportError:
        logger.debug("google-generativeai is not installed; Gemini path disabled")
        return None
    try:
        genai.configure(api_key=key)
        return genai
    except Exception:
        logger.exception("could not configure Gemini client")
        return None


def _llm_topic_segments_gemini(
    cues: Sequence[TranscriptCue],
    *,
    genai_module: Any,
    model: str = "gemini-2.0-flash",
    query: str | None = None,
) -> list[_SegmentTuple]:
    """
    Identify topic boundaries using Google Gemini Flash (free tier).

    Automatically rotates through all available GEMINI_API_KEY_* env vars
    when a key hits its rate limit (HTTP 429).
    """
    global _gemini_key_offset

    rendered = _render_transcript_for_llm(cues)
    system_prompt = _build_system_prompt(query=query)
    user_msg = (
        "Here is the full transcript of a long-form YouTube video. "
        "Identify topic segments per the rules in the system prompt.\n\n"
        f"{rendered}"
    )

    all_keys = _collect_gemini_api_keys()
    if not all_keys:
        # No keys at all — use the already-configured genai_module as-is
        all_keys = [""]

    # Try each key starting from where we left off last time.
    last_exc: Exception | None = None
    for attempt in range(len(all_keys)):
        idx = (_gemini_key_offset + attempt) % len(all_keys)
        key = all_keys[idx]

        # Reconfigure the module with this key (skip if empty = already configured)
        if key:
            try:
                genai_module.configure(api_key=key)
            except Exception:
                continue

        try:
            model_obj = genai_module.GenerativeModel(
                model_name=model,
                system_instruction=system_prompt,
                generation_config=genai_module.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                ),
            )
            response = model_obj.generate_content(user_msg)
            raw = response.text or "{}"
            # Success — remember this key for next call
            _gemini_key_offset = idx
            return _parse_llm_segments_json(raw)
        except Exception as exc:
            last_exc = exc
            exc_str = str(exc).lower()
            is_rate_limit = (
                "429" in exc_str
                or "resource_exhausted" in exc_str
                or "rate" in exc_str
                or "quota" in exc_str
            )
            if is_rate_limit and len(all_keys) > 1:
                logger.info(
                    "Gemini key %d/%d rate-limited; rotating to next key",
                    idx + 1, len(all_keys),
                )
                # Advance offset so next call starts with the next key
                _gemini_key_offset = (idx + 1) % len(all_keys)
                continue
            # Non-rate-limit error — don't rotate, just raise
            raise

    # All keys exhausted
    if last_exc is not None:
        raise last_exc
    return []


# --------------------------------------------------------------------------- #
# Groq / Llama 3 (free tier fallback)
# --------------------------------------------------------------------------- #


def _build_groq_client() -> Any | None:
    """Build a Groq client from GROQ_API_KEY if available."""
    api_key = os.environ.get("GROQ_API_KEY") or ""
    if not api_key:
        return None
    try:
        from groq import Groq
    except ImportError:
        logger.debug("groq package is not installed; Groq path disabled")
        return None
    try:
        return Groq(api_key=api_key)
    except Exception:
        logger.exception("could not build Groq client")
        return None


def _llm_topic_segments_groq(
    cues: Sequence[TranscriptCue],
    *,
    groq_client: Any,
    model: str = "llama-3.3-70b-versatile",
    query: str | None = None,
) -> list[_SegmentTuple]:
    """
    Identify topic boundaries using Groq (Llama 3, free tier).
    Returns timestamp-based segment tuples with optional relevance scores.
    """
    rendered = _render_transcript_for_llm(cues)
    system_prompt = _build_system_prompt(query=query)
    user_msg = (
        "Here is the full transcript of a long-form YouTube video. "
        "Identify topic segments per the rules in the system prompt.\n\n"
        f"{rendered}"
    )

    response = groq_client.chat.completions.create(
        model=model,
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
    )
    raw = response.choices[0].message.content or "{}"
    return _parse_llm_segments_json(raw)


# --------------------------------------------------------------------------- #
# Two-pass boundary refinement
# --------------------------------------------------------------------------- #

_REFINE_PROMPT = """You are refining topic boundaries in a YouTube transcript.

I have identified a topic segment labeled "{label}". Below are the cues around the proposed START boundary (cues {start_range}) and END boundary (cues {end_range}).

START BOUNDARY CUES:
{start_cues}

END BOUNDARY CUES:
{end_cues}

For the START: which cue index is the EXACT first cue where the creator begins discussing "{label}"? Look for where they set context or introduce the topic, not transition phrases from the previous topic.

For the END: which cue index is the EXACT last cue where the creator is still discussing "{label}"? Look for the last substantive cue before they transition away. End at a complete sentence.

Return JSON only:
{{"refined_start_idx": <int>, "refined_end_idx": <int>}}"""


def _refine_boundaries_llm(
    segments: list[tuple[int, int, str, str]],
    cues: Sequence[TranscriptCue],
    *,
    llm_call: Any,
) -> list[tuple[int, int, str, str]]:
    """
    Two-pass refinement: for each segment, extract a small window around each
    boundary and ask the LLM for the EXACT cue index. `llm_call` is a callable
    that accepts a user prompt string and returns the raw text response.

    Returns the same segments with refined start/end indices.
    """
    refined: list[tuple[int, int, str, str]] = []
    window = 5  # cues before/after the boundary

    for s_idx, e_idx, label, summary in segments:
        try:
            start_lo = max(0, s_idx - window)
            start_hi = min(len(cues) - 1, s_idx + window)
            end_lo = max(0, e_idx - window)
            end_hi = min(len(cues) - 1, e_idx + window)

            start_cue_text = "\n".join(
                f"[{i} {_format_timestamp(cues[i].start)}] {cues[i].text}"
                for i in range(start_lo, start_hi + 1)
            )
            end_cue_text = "\n".join(
                f"[{i} {_format_timestamp(cues[i].start)}] {cues[i].text}"
                for i in range(end_lo, end_hi + 1)
            )

            prompt = _REFINE_PROMPT.format(
                label=label,
                start_range=f"{start_lo}-{start_hi}",
                end_range=f"{end_lo}-{end_hi}",
                start_cues=start_cue_text,
                end_cues=end_cue_text,
            )

            raw = llm_call(prompt)
            payload = json.loads(raw)
            new_s = int(payload.get("refined_start_idx", s_idx))
            new_e = int(payload.get("refined_end_idx", e_idx))

            # Sanity check: refined indices must be within the window range
            if start_lo <= new_s <= start_hi:
                s_idx = new_s
            if end_lo <= new_e <= end_hi:
                e_idx = new_e
        except Exception:
            logger.debug("boundary refinement failed for segment '%s'; keeping original", label)

        refined.append((s_idx, e_idx, label, summary))

    return refined


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
        t_start = max(0.0, float(t_start_sec) - 1.0)
        t_end = float(t_end_sec) + 1.0
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


# --------------------------------------------------------------------------- #
# Query-based topic relevance filtering
# --------------------------------------------------------------------------- #

# Minimum combined score for a topic reel to be considered relevant to the
# user's search query.  Kept low because Jaccard on short labels is naturally
# sparse; the transcript overlap (weight 0.5) does most of the heavy lifting.
_MIN_QUERY_RELEVANCE = 0.03


def _tokenize_for_relevance(text: str) -> set[str]:
    """Lowercase, strip stopwords, keep tokens ≥3 chars — same spirit as `_cue_tokens`."""
    return {
        tok
        for tok in re.findall(r"[a-z][a-z']{2,}", text.lower())
        if tok not in _STOPWORDS
    }


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _compute_semantic_scores(
    query: str,
    texts: list[str],
) -> list[float] | None:
    """
    Compute cosine similarity between query and each text using the local
    sentence-transformer (all-MiniLM-L6-v2). Returns None if the model
    is not available (falls back to Jaccard-only scoring).
    """
    model = _load_sentence_transformer()
    if model is None:
        return None
    try:
        import numpy as np
    except ImportError:
        return None
    try:
        all_texts = [query] + texts
        embeddings = model.encode(
            all_texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        query_vec = embeddings[0]
        text_vecs = embeddings[1:]
        # Cosine similarity with normalized vectors = dot product
        scores = (text_vecs @ query_vec).tolist()
        return scores
    except Exception:
        logger.debug("semantic scoring failed; falling back to Jaccard")
        return None


def _filter_reels_by_query(
    reels: list[TopicReel],
    query: str,
    *,
    cues: Sequence[TranscriptCue],
    transcript_validation: TranscriptValidation | None = None,
) -> list[TopicReel]:
    """
    Score each TopicReel against the user's search *query* and keep only
    relevant ones.

    Uses **adaptive blending** when the reel already carries an LLM-assigned
    ``relevance_score``:

    * Strong transcript (coverage >= 0.80) + specific query (>= 3 tokens):
      trust LLM more (70% LLM, 30% local signals).
    * Weak transcript or vague query: trust local signals more
      (30% LLM, 70% local signals).

    When ``relevance_score`` is ``None`` the original local-only scoring
    applies unchanged.

    If ALL reels fall below the threshold, the single highest-scoring reel is
    returned so we never silently discard every topic for a video that has content.
    """
    query_tokens = _tokenize_for_relevance(query)
    if not query_tokens:
        return reels  # no meaningful query — return all

    # Determine adaptive blend weight based on transcript quality + query specificity.
    has_strong_transcript = (
        transcript_validation is not None
        and transcript_validation.coverage_ratio >= 0.80
        and transcript_validation.largest_gap_sec <= 30.0
    )
    has_specific_query = len(query_tokens) >= 3
    llm_weight = 0.70 if (has_strong_transcript and has_specific_query) else 0.30

    # Build transcript text for each reel.
    reel_transcripts: list[str] = []
    for reel in reels:
        transcript_text = ""
        if reel.cue_start_idx >= 0 and reel.cue_end_idx >= 0:
            transcript_text = " ".join(
                cues[i].text
                for i in range(
                    max(0, reel.cue_start_idx),
                    min(len(cues), reel.cue_end_idx + 1),
                )
            )
        reel_transcripts.append(transcript_text)

    # Try semantic scoring (free, local sentence-transformer)
    semantic_scores = _compute_semantic_scores(query, reel_transcripts)

    scored: list[tuple[float, TopicReel]] = []
    for i, reel in enumerate(reels):
        label_score = _jaccard(query_tokens, _tokenize_for_relevance(reel.label))
        summary_score = _jaccard(query_tokens, _tokenize_for_relevance(reel.summary))

        # Compute local signal score.
        if semantic_scores is not None:
            sem_score = max(0.0, semantic_scores[i])
            local_score = 0.25 * label_score + 0.15 * summary_score + 0.60 * sem_score
        else:
            transcript_score = _jaccard(
                query_tokens, _tokenize_for_relevance(reel_transcripts[i])
            )
            local_score = 0.4 * label_score + 0.5 * transcript_score + 0.1 * summary_score

        # Blend with LLM relevance score when available.
        if reel.relevance_score is not None:
            combined = llm_weight * reel.relevance_score + (1.0 - llm_weight) * local_score
        else:
            combined = local_score

        scored.append((combined, reel))

    scored.sort(key=lambda pair: pair[0], reverse=True)

    # Keep reels above the relevance threshold.
    kept = [reel for score, reel in scored if score >= _MIN_QUERY_RELEVANCE]

    # Fallback: never return empty — keep the best match.
    if not kept and scored:
        kept = [scored[0][1]]

    # Restore chronological order.
    kept.sort(key=lambda r: r.t_start)
    return kept


_SENTENCE_END_RE = re.compile(r"[.?!][\s\"']*$")


def _find_sentence_end_cue(
    cues: Sequence[TranscriptCue],
    anchor_idx: int,
    *,
    scan_backward: bool = True,
    max_scan: int = 3,
) -> int | None:
    """
    Scan up to `max_scan` cues from `anchor_idx` looking for a cue whose text
    ends with sentence-ending punctuation (. ? !). Returns the cue index if
    found, None otherwise.

    When `scan_backward=True`, scans from anchor_idx toward 0 (for end boundaries).
    When `scan_backward=False`, scans from anchor_idx toward len(cues) (for start boundaries).
    """
    if scan_backward:
        for i in range(anchor_idx, max(anchor_idx - max_scan, -1), -1):
            if 0 <= i < len(cues) and _SENTENCE_END_RE.search(cues[i].text.strip()):
                return i
    else:
        for i in range(anchor_idx, min(anchor_idx + max_scan, len(cues))):
            # For start boundaries: find a cue that begins a new sentence
            # (the PREVIOUS cue ends with sentence-ending punctuation).
            if i > 0 and _SENTENCE_END_RE.search(cues[i - 1].text.strip()):
                return i
    return None


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
      * snap start_idx FORWARD to the nearest cue that begins a new sentence
      * snap end_idx BACKWARD to the nearest cue that ends a sentence
      * derive t_start and t_end from the snapped cue boundaries
      * if there's a natural pause within ±SNAP_TOLERANCE_SEC, nudge to the
        gap midpoint so the cut lands on silence
      * drop the segment if it's outside [MIN_TOPIC_REEL_SEC, MAX_TOPIC_REEL_SEC]
      * sort + de-dup overlapping segments
    """
    if not cues:
        return []

    last_idx = len(cues) - 1
    candidates: list[TopicReel] = []
    abs_max_cap = max(max_reel_sec, MAX_TOPIC_REEL_SEC)

    for s_idx, e_idx, label, summary in raw_segments:
        s_idx = max(0, min(last_idx, int(s_idx)))
        e_idx = max(s_idx, min(last_idx, int(e_idx)))

        # Sentence-boundary snapping: try to land on natural sentence edges.
        # For the END: scan backward from e_idx to find a cue that ends with
        # sentence-ending punctuation — this ensures clips end after a complete thought.
        sentence_end = _find_sentence_end_cue(cues, e_idx, scan_backward=True)
        if sentence_end is not None and sentence_end >= s_idx:
            e_idx = sentence_end

        # For the START: scan forward from s_idx to find a cue that begins a
        # new sentence (previous cue ends with punctuation).
        sentence_start = _find_sentence_end_cue(cues, s_idx, scan_backward=False)
        if sentence_start is not None and sentence_start <= e_idx:
            s_idx = sentence_start

        t_start = max(0.0, cues[s_idx].start - 1.0)
        t_end = cues[e_idx].end + 1.0

        # Try to snap t_end to the gap between cues[e_idx] and cues[e_idx+1]
        # if the gap falls within tolerance of our +1s offset.
        if e_idx + 1 < len(cues):
            gap_start = cues[e_idx].end
            gap_end = cues[e_idx + 1].start
            if gap_end > gap_start:
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
# Timestamp-based directional snapping (new pipeline for LLM results)
# --------------------------------------------------------------------------- #


def _validate_timestamp_against_cues(
    timestamp: float,
    cues: Sequence[TranscriptCue],
    *,
    epsilon: float = 1.0,
    check_start: bool = True,
) -> bool:
    """Check whether *timestamp* is within *epsilon* of an actual cue boundary.

    When *check_start* is True, matches against ``cue.start`` values.
    When False, matches against ``cue.end`` values.
    """
    for cue in cues:
        ref = cue.start if check_start else cue.end
        if abs(ref - timestamp) <= epsilon:
            return True
    return False


def _snap_timestamps_to_cues(
    raw_segments: Iterable[_SegmentTuple],
    cues: Sequence[TranscriptCue],
    *,
    video_id: str,
    video_duration_sec: float | None,
    min_reel_sec: float = MIN_TOPIC_REEL_SEC,
    max_reel_sec: float = MAX_TOPIC_REEL_SEC,
    timestamp_epsilon: float = 1.0,
) -> list[TopicReel]:
    """Resolve LLM-returned timestamps into final TopicReel records.

    Uses **directional snapping** to avoid leaking neighbouring topics:

    * ``start_time`` -> first cue whose ``.start >= returned_timestamp``
      (snap forward so we never include the tail of the previous topic).
    * ``end_time``   -> last cue whose ``.end <= returned_timestamp``
      (snap backward so we never include the head of the next topic).

    After snapping, an optional sentence-boundary expansion pass widens the
    clip by up to 3 cues if a natural sentence boundary exists just outside.

    Segments whose timestamps do not match any real cue boundary within
    *timestamp_epsilon* are coerced to the nearest valid boundary (with a
    warning), not silently dropped.  Only segments that fail duration
    guardrails after snapping are discarded.
    """
    if not cues:
        return []

    abs_max_cap = max(max_reel_sec, MAX_TOPIC_REEL_SEC)
    candidates: list[TopicReel] = []

    for start_time, end_time, label, summary, relevance in raw_segments:
        # --- Validate that timestamps match real cue boundaries ----------- #
        if not _validate_timestamp_against_cues(start_time, cues, epsilon=timestamp_epsilon, check_start=True):
            logger.debug(
                "start_time %.1f does not match any cue start within %.1fs; coercing",
                start_time, timestamp_epsilon,
            )
        if not _validate_timestamp_against_cues(end_time, cues, epsilon=timestamp_epsilon, check_start=False):
            logger.debug(
                "end_time %.1f does not match any cue end within %.1fs; coercing",
                end_time, timestamp_epsilon,
            )

        # --- Directional snap: start forward, end backward ---------------- #
        # start -> first cue whose .start >= start_time
        s_idx = 0
        for i, cue in enumerate(cues):
            if cue.start >= start_time - 0.01:
                s_idx = i
                break
        else:
            s_idx = len(cues) - 1

        # end -> last cue whose .end <= end_time
        e_idx = s_idx
        for i in range(len(cues) - 1, s_idx - 1, -1):
            if cues[i].end <= end_time + 0.01:
                e_idx = i
                break

        if e_idx < s_idx:
            e_idx = s_idx

        # --- Optional sentence-boundary expansion ------------------------- #
        # Try to widen slightly if a sentence boundary is just outside.
        sentence_end = _find_sentence_end_cue(cues, e_idx, scan_backward=False, max_scan=2)
        if sentence_end is not None and sentence_end > e_idx:
            expansion_cost = cues[sentence_end].end - cues[e_idx].end
            if expansion_cost <= 2.0:
                e_idx = sentence_end

        sentence_start = _find_sentence_end_cue(cues, s_idx, scan_backward=True, max_scan=2)
        if sentence_start is not None and sentence_start < s_idx:
            expansion_cost = cues[s_idx].start - cues[sentence_start].start
            if expansion_cost <= 2.0:
                s_idx = sentence_start

        # --- Derive final timestamps from snapped cues -------------------- #
        t_start = cues[s_idx].start
        t_end = cues[e_idx].end

        # Snap t_end to inter-cue gap if one exists right after e_idx.
        if e_idx + 1 < len(cues):
            gap_start = cues[e_idx].end
            gap_end = cues[e_idx + 1].start
            if gap_end > gap_start:
                snapped = (gap_start + gap_end) / 2.0
                if abs(snapped - t_end) <= SNAP_TOLERANCE_SEC:
                    t_end = snapped

        if video_duration_sec and video_duration_sec > 0:
            t_end = min(t_end, float(video_duration_sec))

        duration = t_end - t_start
        if duration < min_reel_sec or duration > abs_max_cap:
            logger.debug(
                "dropping out-of-range segment t=%.1f-%.1f duration=%.1fs label=%s",
                t_start, t_end, duration, label,
            )
            continue

        # Split segments that exceed the soft max.
        if duration > max_reel_sec:
            split = _split_long_range(
                t_start, t_end, label, summary,
                max_reel_sec=max_reel_sec, min_reel_sec=min_reel_sec,
            )
            if not split:
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
                    relevance_score=relevance,
                )
            )

    # Sort and trim overlaps.
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
# Phase A.4: ClipBoundaryEngine integration
# --------------------------------------------------------------------------- #


def _apply_boundary_engine(
    reels: list[TopicReel],
    *,
    ingest_cues: Sequence[Any] | None,  # list[IngestTranscriptCue]
    chapters: Sequence[Any] | None,
    silence_ranges: Sequence[tuple[float, float]] | None,
    llm_topic_segments_raw: Sequence[dict[str, Any]] | None,
    query: str | None,
    user_min_sec: float,
    user_max_sec: float,
    video_duration_sec: float | None,
) -> list[TopicReel]:
    """
    Phase A.4: refine every TopicReel's (t_start, t_end) via `ClipBoundaryEngine`.

    The LLM / chapter / novelty path above already identified WHICH topic each
    reel represents. This pass tightens the boundaries to:
      - start at the first substantive-introduction sentence (skipping boilerplate)
      - end at either a natural topic-switch sentence boundary or, under
        max-duration truncation, the latest terminal-punct sentence that fits.

    Tagged `boundary_quality` per tier so downstream code can surface the
    confidence level to clients.

    `ingest_cues` carries word-level timestamps from the Whisper paths (Phase A.1);
    when absent (older transcripts without word data) the engine falls back to
    proportional word splits and tags quality `"sentence"` instead of `"precise"`.
    """
    if not reels or not ingest_cues:
        # No word-level data available — leave reels as-is, but tag quality.
        for r in reels:
            if r.boundary_quality == "legacy":
                r.boundary_quality = "chapter" if chapters else "sentence"
        return reels

    try:
        from .clip_boundary import ClipBoundaryEngine
        from .sentences import split_sentences
    except Exception:
        logger.exception("ClipBoundaryEngine unavailable; skipping boundary refinement")
        return reels

    try:
        cue_list = list(ingest_cues)
        sentences = split_sentences(cue_list)
        if not sentences:
            return reels
    except Exception:
        logger.exception("sentence splitting raised; skipping boundary refinement")
        return reels

    engine = ClipBoundaryEngine(llm_pick_start=None)

    # Convert chapters to the dict shape the engine expects
    chapter_dicts: list[dict[str, Any]] = []
    if chapters:
        for c in chapters:
            try:
                chapter_dicts.append(
                    {
                        "start_time": float(getattr(c, "start", getattr(c, "start_time", 0.0))),
                        "end_time": float(getattr(c, "end", getattr(c, "end_time", 0.0))),
                        "title": str(getattr(c, "title", "")),
                    }
                )
            except (AttributeError, TypeError, ValueError):
                continue

    # LLM topic segments → shape the engine expects (dicts with t_end + label)
    llm_hints: list[dict[str, Any]] = []
    if llm_topic_segments_raw:
        for seg in llm_topic_segments_raw:
            if isinstance(seg, dict) and isinstance(seg.get("t_end"), (int, float)):
                llm_hints.append(seg)

    refined: list[TopicReel] = []
    for r in reels:
        try:
            result = engine.refine(
                raw_t_start=r.t_start,
                raw_t_end=r.t_end,
                cues=cue_list,
                sentences=sentences,
                query=query,
                chapters=chapter_dicts or None,
                llm_topic_segments=llm_hints or None,
                silence_ranges=silence_ranges,
                user_min_sec=user_min_sec,
                user_max_sec=user_max_sec,
                video_duration_sec=video_duration_sec,
            )
            # Only adopt refined bounds if they're saner than the raw ones.
            if result.t_end > result.t_start:
                r.t_start = round(float(result.t_start), 2)
                r.t_end = round(float(result.t_end), 2)
            r.boundary_quality = result.boundary_quality
        except Exception:
            logger.exception(
                "boundary refinement failed for video %s raw=(%.2f, %.2f); keeping raw",
                r.video_id, r.t_start, r.t_end,
            )
        refined.append(r)
    return refined


# --------------------------------------------------------------------------- #
# Top-level entrypoint
# --------------------------------------------------------------------------- #


def cut_video_into_topic_reels(
    url_or_id: str,
    *,
    query: str | None = None,
    duration_sec: float | None = None,
    openai_client: Any | None = None,
    model: str = DEFAULT_MODEL,
    use_llm: bool = True,
    refine_boundaries: bool = True,
    transcript: Sequence[TranscriptCue] | None = None,
    info_dict: dict[str, Any] | None = None,
    min_reel_sec: float = MIN_TOPIC_REEL_SEC,
    max_reel_sec: float = MAX_TOPIC_REEL_SEC,
    # Phase A.4: optional word-level precision inputs.
    ingest_cues_for_precision: Sequence[Any] | None = None,
    silence_ranges: Sequence[tuple[float, float]] | None = None,
    user_min_sec: float | None = None,
    user_max_sec: float | None = None,
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

    # ---- Transcript validation ----------------------------------------- #
    transcript_validation: TranscriptValidation | None = None
    if transcript:
        effective_duration = duration_sec or classification.duration_sec or None
        transcript_validation = _validate_transcript_coverage(
            transcript, effective_duration,
        )

    # Resolve soft user-setting bounds — default to the hard min/max guardrails.
    _user_min = float(user_min_sec) if user_min_sec is not None else float(min_reel_sec)
    _user_max = float(user_max_sec) if user_max_sec is not None else float(max_reel_sec)

    # ---- Path 1: YouTube chapters (free, no API, no inference). -------- #
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
            # Phase A.4: tighten chapter bounds to skip intro/outro boilerplate and
            # land on sentence endings, not hard chapter-boundary cuts mid-sentence.
            chapter_reels = _apply_boundary_engine(
                chapter_reels,
                ingest_cues=ingest_cues_for_precision,
                chapters=chapters,
                silence_ranges=silence_ranges,
                llm_topic_segments_raw=None,
                query=query,
                user_min_sec=_user_min,
                user_max_sec=_user_max,
                video_duration_sec=classification.duration_sec or None,
            )
            if query and chapter_reels:
                chapter_reels = _filter_reels_by_query(
                    chapter_reels, query, cues=transcript or [],
                    transcript_validation=transcript_validation,
                )
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

    # ---- Path 2: LLM topic segmentation (timestamp-based). ------------- #
    # Fallback chain: Gemini Flash (free) → Groq/Llama 3 (free) → OpenAI (paid)
    # LLM functions now return timestamp-based _SegmentTuples, not cue indices.
    llm_segments: list[_SegmentTuple] = []
    used_llm = False

    if use_llm and len(transcript) <= MAX_CUES_FOR_LLM:
        # 2a: Google Gemini Flash (free tier: 15 RPM, 1M tokens/day)
        gemini = _build_gemini_client()
        if gemini:
            try:
                llm_segments = _llm_topic_segments_gemini(
                    transcript, genai_module=gemini, query=query,
                )
                if llm_segments:
                    used_llm = True
                    logger.info(
                        "video %s segmented via Gemini Flash → %d segments",
                        video_id, len(llm_segments),
                    )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Gemini topic segmentation failed for video %s; trying Groq next",
                    video_id,
                )
                llm_segments = []

        # 2b: Groq / Llama 3 (free tier fallback)
        if not llm_segments:
            groq = _build_groq_client()
            if groq:
                try:
                    llm_segments = _llm_topic_segments_groq(
                        transcript, groq_client=groq, query=query,
                    )
                    if llm_segments:
                        used_llm = True
                        logger.info(
                            "video %s segmented via Groq/Llama 3 → %d segments",
                            video_id, len(llm_segments),
                        )
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "Groq topic segmentation failed for video %s; trying OpenAI next",
                        video_id,
                    )
                    llm_segments = []

        # 2c: OpenAI (paid, optional last resort for LLM path)
        if not llm_segments:
            client = openai_client or _maybe_build_openai_client()
            if client:
                try:
                    llm_segments = _llm_topic_segments(
                        transcript, openai_client=client, model=model, query=query,
                    )
                    if llm_segments:
                        used_llm = True
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "OpenAI topic segmentation failed for video %s; trying semantic path",
                        video_id,
                    )
                    llm_segments = []

    elif use_llm and len(transcript) > MAX_CUES_FOR_LLM:
        logger.warning(
            "transcript has %d cues, exceeds MAX_CUES_FOR_LLM=%d; "
            "trying free semantic path instead",
            len(transcript), MAX_CUES_FOR_LLM,
        )

    # If an LLM produced timestamp-based segments, snap them directionally.
    if used_llm and llm_segments:
        reels = _snap_timestamps_to_cues(
            llm_segments,
            transcript,
            video_id=video_id,
            video_duration_sec=classification.duration_sec or None,
            min_reel_sec=min_reel_sec,
            max_reel_sec=max_reel_sec,
        )
        # Phase A.4: tighten LLM-picked topic windows to precise sentence ends.
        # Also pass the raw LLM segments as topic_switch_hints so the engine can
        # prefer natural closes over truncation.
        llm_hints_for_engine = [
            {"t_end": float(seg[1]), "label": str(seg[2]) if len(seg) > 2 else ""}
            for seg in llm_segments
        ]
        reels = _apply_boundary_engine(
            reels,
            ingest_cues=ingest_cues_for_precision,
            chapters=chapters or None,
            silence_ranges=silence_ranges,
            llm_topic_segments_raw=llm_hints_for_engine,
            query=query,
            user_min_sec=_user_min,
            user_max_sec=_user_max,
            video_duration_sec=classification.duration_sec or None,
        )
        if query and reels:
            reels = _filter_reels_by_query(
                reels, query, cues=transcript,
                transcript_validation=transcript_validation,
            )
        return classification, reels

    # ---- Path 3: Local sentence-transformer embeddings (free, near-LLM). #
    # These fallback paths still use the legacy index-based pipeline.
    legacy_segments: list[tuple[int, int, str, str]] = []
    semantic = _semantic_topic_segments(transcript)
    if semantic:
        logger.info(
            "video %s segmented via local sentence-transformers → %d ranges",
            video_id, len(semantic),
        )
        legacy_segments = semantic

    # ---- Path 4: Pure-Python Jaccard heuristic (always available). ----- #
    if not legacy_segments:
        legacy_segments = _heuristic_topic_segments(transcript)

    reels = _snap_segments_to_cues(
        legacy_segments,
        transcript,
        video_id=video_id,
        video_duration_sec=classification.duration_sec or None,
        min_reel_sec=min_reel_sec,
        max_reel_sec=max_reel_sec,
    )
    # Phase A.4: tighten heuristic/semantic bounds via ClipBoundaryEngine.
    reels = _apply_boundary_engine(
        reels,
        ingest_cues=ingest_cues_for_precision,
        chapters=chapters or None,
        silence_ranges=silence_ranges,
        llm_topic_segments_raw=None,
        query=query,
        user_min_sec=_user_min,
        user_max_sec=_user_max,
        video_duration_sec=classification.duration_sec or None,
    )
    if query and reels:
        reels = _filter_reels_by_query(
            reels, query, cues=transcript,
            transcript_validation=transcript_validation,
        )
    return classification, reels


def _maybe_build_openai_client() -> Any | None:
    """OpenAI integration is permanently disabled."""
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
