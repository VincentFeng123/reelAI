"""
Clip segmentation — pick a 15-60s window that starts and ends on a natural boundary.

Two exported functions:

  * `normalize_clip_window(...)`: pure int-clamping logic, copied verbatim from
    `app/services/reels.py:9286-9325` with `self` removed. This keeps the legacy and
    the new pipeline bit-compatible on window math so tests comparing outputs stay valid.

  * `pick_segments(...)`: the smart picker. Uses transcript cue gaps + optional
    ffmpeg silencedetect ranges to find a window that does NOT start or end mid-word.
    If concept info (embedding + terms) is supplied, falls through to
    `segmenter.select_segments` for relevance-ranked picks. Otherwise picks the longest
    "interesting" contiguous window from the middle-third of the video, snapped to the
    nearest silence/gap on each side.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

from ..services.segmenter import (
    TranscriptChunk,
    chunk_transcript,
    lexical_overlap_score,
    merge_adjacent,
    select_segments,
)
from .logging_config import get_ingest_logger
from .models import IngestSegment, IngestTranscriptCue

logger: logging.Logger = get_ingest_logger(__name__)


# --------------------------------------------------------------------- #
# Verbatim copy of `reels.py:9286-9325` (self-less)
# --------------------------------------------------------------------- #


def normalize_clip_window(
    t_start: float,
    t_end: float,
    video_duration_sec: int,
    min_len: int = 15,
    max_len: int = 60,
    allow_exceed_max: bool = False,
    allow_below_min: bool = False,
) -> tuple[int, int] | None:
    """
    Copied verbatim from `reels.py:9286-9325` with `self` removed. Do NOT edit without
    keeping the original in sync, or rip the original out once the refactor merges.
    """
    if min_len < 1:
        min_len = 1

    start_sec = max(0, int(float(t_start)))
    end_sec = max(start_sec + 1, int(float(t_end)))

    if max_len > 0 and not allow_exceed_max and end_sec - start_sec > max_len:
        end_sec = start_sec + max_len
    if not allow_below_min and end_sec - start_sec < min_len:
        end_sec = start_sec + min_len

    if video_duration_sec > 0:
        if not allow_below_min and video_duration_sec < min_len:
            return None
        if end_sec > video_duration_sec:
            end_sec = video_duration_sec
        if not allow_below_min and end_sec - start_sec < min_len:
            start_sec = max(0, end_sec - min_len)
        if max_len > 0 and not allow_exceed_max and end_sec - start_sec > max_len:
            end_sec = start_sec + max_len
        if end_sec > video_duration_sec:
            end_sec = video_duration_sec

    if end_sec <= start_sec:
        return None
    if not allow_below_min and end_sec - start_sec < min_len:
        return None
    if max_len > 0 and not allow_exceed_max and end_sec - start_sec > max_len:
        return None
    return (start_sec, end_sec)


# --------------------------------------------------------------------- #
# Smart boundary-aware picker
# --------------------------------------------------------------------- #


@dataclass
class _Candidate:
    t_start: float
    t_end: float
    text: str
    score: float


def _snap_to_boundary(
    raw_time: float,
    *,
    cues: Sequence[IngestTranscriptCue],
    silence_ranges: Sequence[tuple[float, float]],
    max_adjust: float = 1.5,
) -> float:
    """
    Move `raw_time` to the nearest transcript-gap OR silence-range boundary within
    `max_adjust` seconds. If no boundary is close enough, return `raw_time` unchanged.
    """
    best = raw_time
    best_distance = float("inf")

    # Transcript cue boundaries
    for cue in cues:
        for boundary in (cue.start, cue.end):
            distance = abs(boundary - raw_time)
            if distance < best_distance and distance <= max_adjust:
                best_distance = distance
                best = boundary

    # Silence range midpoints
    for start, end in silence_ranges:
        if start >= end:
            continue
        mid = 0.5 * (start + end)
        for boundary in (start, mid, end):
            distance = abs(boundary - raw_time)
            if distance < best_distance and distance <= max_adjust:
                best_distance = distance
                best = boundary

    return best


def _cue_to_entry(cue: IngestTranscriptCue) -> dict:
    return {
        "start": float(cue.start),
        "duration": max(0.0, float(cue.end) - float(cue.start)),
        "text": cue.text,
    }


def _candidate_from_chunk(
    chunk: TranscriptChunk,
    *,
    cues: Sequence[IngestTranscriptCue],
    silence_ranges: Sequence[tuple[float, float]],
    target_sec: int,
    min_sec: int,
    max_sec: int,
    video_duration_sec: float,
) -> _Candidate | None:
    if not chunk.text.strip():
        return None

    raw_start = float(chunk.t_start)
    raw_end = float(chunk.t_end)
    midpoint = 0.5 * (raw_start + raw_end)

    # If the chunk is already within the target window, just snap boundaries.
    length = raw_end - raw_start
    if length < min_sec:
        # Expand symmetrically around the midpoint to reach target_sec.
        half = target_sec / 2.0
        raw_start = max(0.0, midpoint - half)
        raw_end = min(video_duration_sec or (midpoint + half), midpoint + half)
    elif length > max_sec:
        raw_end = raw_start + max_sec

    snapped_start = _snap_to_boundary(raw_start, cues=cues, silence_ranges=silence_ranges)
    snapped_end = _snap_to_boundary(raw_end, cues=cues, silence_ranges=silence_ranges)

    if snapped_end <= snapped_start:
        return None

    window = normalize_clip_window(
        snapped_start,
        snapped_end,
        int(video_duration_sec or 0),
        min_len=min_sec,
        max_len=max_sec,
    )
    if window is None:
        return None

    w_start_f = float(window[0])
    w_end_f = float(window[1])
    # Score: how well this window lines up with a silence boundary on each side.
    boundary_bonus = 0.0
    for ts in (w_start_f, w_end_f):
        for s, e in silence_ranges:
            if s <= ts <= e:
                boundary_bonus += 0.25
                break
    score = 1.0 + boundary_bonus + min(1.0, (w_end_f - w_start_f) / max(float(target_sec), 1.0))
    return _Candidate(t_start=w_start_f, t_end=w_end_f, text=chunk.text, score=score)


def pick_segments(
    cues: list[IngestTranscriptCue],
    video_duration_sec: float,
    *,
    target_sec: int = 45,
    min_sec: int = 15,
    max_sec: int = 60,
    silence_ranges: list[tuple[float, float]] | None = None,
    concept_embedding: list[float] | None = None,
    chunk_embeddings: list[list[float]] | None = None,
    concept_terms: list[str] | None = None,
    top_k: int = 1,
) -> list[IngestSegment]:
    """
    Choose up to `top_k` IngestSegment windows, each satisfying the 15-60s bounds and
    snapped to silence/transcript boundaries.

    When `concept_embedding` + `chunk_embeddings` are provided (the flow where the caller
    has a real material/concept), we delegate the "which chunks are most relevant" question
    to the existing `segmenter.select_segments` then boundary-snap the result.

    When concept info is missing (anonymous ingest), we pick the longest chunk from the
    middle third of the video (to avoid intros/outros) and snap it.
    """
    silence = silence_ranges or []
    if not cues:
        return []

    entries = [_cue_to_entry(cue) for cue in cues]
    chunks = chunk_transcript(entries, target_sec=target_sec, min_sec=min_sec, max_sec=max_sec)
    if not chunks:
        # Fallback: wrap all cues into a single synthetic chunk of the whole video.
        total_text = " ".join(cue.text for cue in cues if cue.text).strip()
        if not total_text:
            return []
        chunks = [TranscriptChunk(chunk_index=0, t_start=float(cues[0].start), t_end=float(cues[-1].end), text=total_text)]

    # Concept-guided path
    if concept_embedding and chunk_embeddings and len(chunk_embeddings) == len(chunks):
        try:
            matches = select_segments(
                concept_embedding,
                chunk_embeddings,
                chunks,
                concept_terms=concept_terms or [],
                top_k=max(1, top_k),
            )
            merged = merge_adjacent(matches, max_total_sec=max_sec)
            results: list[IngestSegment] = []
            for match in merged[: max(1, top_k)]:
                window = normalize_clip_window(
                    match.t_start,
                    match.t_end,
                    int(video_duration_sec or 0),
                    min_len=min_sec,
                    max_len=max_sec,
                )
                if window is None:
                    continue
                w_start_f = float(window[0])
                w_end_f = float(window[1])
                snapped_start = _snap_to_boundary(w_start_f, cues=cues, silence_ranges=silence)
                snapped_end = _snap_to_boundary(w_end_f, cues=cues, silence_ranges=silence)
                final = normalize_clip_window(
                    snapped_start,
                    snapped_end,
                    int(video_duration_sec or 0),
                    min_len=min_sec,
                    max_len=max_sec,
                )
                if final is None:
                    continue
                results.append(
                    IngestSegment(
                        t_start=float(final[0]),
                        t_end=float(final[1]),
                        text=match.text,
                        score=float(match.score),
                    )
                )
            if results:
                return results
        except Exception:
            logger.exception("concept-guided segment selection failed; falling back to heuristic")

    # Heuristic fallback: pick longest chunk in middle third, with lexical bonus for any
    # terms in `concept_terms` (if provided), then boundary-snap.
    total_duration = float(video_duration_sec) if video_duration_sec and video_duration_sec > 0 else float(chunks[-1].t_end)
    lower = total_duration * 0.2
    upper = total_duration * 0.9 if total_duration > 0 else float("inf")

    scored: list[_Candidate] = []
    for chunk in chunks:
        midpoint = 0.5 * (chunk.t_start + chunk.t_end)
        center_bonus = 0.0
        if lower <= midpoint <= upper:
            center_bonus = 0.5

        candidate = _candidate_from_chunk(
            chunk,
            cues=cues,
            silence_ranges=silence,
            target_sec=target_sec,
            min_sec=min_sec,
            max_sec=max_sec,
            video_duration_sec=total_duration,
        )
        if candidate is None:
            continue
        lex_bonus = 0.0
        if concept_terms:
            lex_bonus = 0.4 * lexical_overlap_score(candidate.text, concept_terms)
        candidate.score += center_bonus + lex_bonus
        scored.append(candidate)

    if not scored:
        # Last-ditch: just clamp the first cue into the min_sec window.
        first = cues[0]
        window = normalize_clip_window(
            first.start,
            first.start + target_sec,
            int(total_duration or 0),
            min_len=min_sec,
            max_len=max_sec,
        )
        if window is None:
            return []
        return [
            IngestSegment(
                t_start=float(window[0]),
                t_end=float(window[1]),
                text=first.text,
                score=0.5,
            )
        ]

    scored.sort(key=lambda c: c.score, reverse=True)
    take = max(1, top_k)
    picked = scored[:take]
    return [
        IngestSegment(t_start=p.t_start, t_end=p.t_end, text=p.text, score=p.score)
        for p in picked
    ]


def snippet_for_window(
    cues: list[IngestTranscriptCue],
    t_start: float,
    t_end: float,
    *,
    max_chars: int = 700,
) -> str:
    """Join all cue text whose midpoint falls inside [t_start, t_end] and cap at max_chars."""
    pieces: list[str] = []
    for cue in cues:
        mid = 0.5 * (cue.start + cue.end)
        if mid < t_start:
            continue
        if mid > t_end:
            break
        if cue.text:
            pieces.append(cue.text)
    snippet = " ".join(pieces).strip()
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars].rstrip() + "…"
    return snippet


__all__ = ["normalize_clip_window", "pick_segments", "snippet_for_window"]
