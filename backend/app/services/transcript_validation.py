"""
Shared transcript quality validation.

This module is intentionally dependency-free — no database, config, or service
imports — so it can be used from:

  * ``services/reels.py``   (main scoring pipeline)
  * ``services/topic_cut.py`` (LLM-based segmentation)
  * ``ingestion/transcribe.py`` (fallback-chain ingestion)

It accepts transcript cues in any of the three formats used across the codebase:

  1. Raw dicts from ``youtube_transcript_api``:
     ``{"start": float, "duration": float, "text": str}``
  2. ``IngestTranscriptCue`` (Pydantic, from ``ingestion/models.py``):
     ``.start``, ``.end``, ``.text``
  3. ``TranscriptCue`` (dataclass, from ``services/topic_cut.py``):
     ``.start``, ``.duration``, ``.text``, property ``.end``

All are normalised to ``(start, end, text)`` tuples internally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class TranscriptQuality:
    """Result of :func:`validate_transcript`."""

    coverage_ratio: float
    """Fraction of video duration covered by transcript (last cue end / duration)."""

    largest_gap_sec: float
    """Longest silence gap between consecutive cues (seconds)."""

    first_cue_delay_sec: float
    """Seconds before the first cue starts."""

    empty_cue_ratio: float
    """Fraction of cues with fewer than 2 non-whitespace characters."""

    avg_cue_duration: float
    """Mean cue duration in seconds."""

    cue_count: int
    """Total number of cues in the transcript."""

    is_adequate: bool
    """``True`` when all quality checks pass."""

    warnings: list[str] = field(default_factory=list)
    """Human-readable descriptions of any quality issues found."""


# ---------------------------------------------------------------------------
# Internal cue normalisation
# ---------------------------------------------------------------------------

def _normalise_cue(raw: Any) -> tuple[float, float, str] | None:
    """Convert any supported cue format to ``(start, end, text)``.

    Returns ``None`` when the cue cannot be parsed.
    """
    if isinstance(raw, dict):
        try:
            start = float(raw.get("start", 0.0))
            text = str(raw.get("text", ""))
            # youtube_transcript_api uses "duration"; IngestTranscriptCue-style dicts use "end"
            if "end" in raw:
                end = float(raw["end"])
            elif "duration" in raw:
                end = start + float(raw["duration"])
            else:
                end = start
            return (start, max(end, start), text)
        except (TypeError, ValueError):
            return None

    # Object with .start / .end / .text  (IngestTranscriptCue, or anything duck-typed)
    try:
        start = float(getattr(raw, "start", 0.0))
        text = str(getattr(raw, "text", ""))
        if hasattr(raw, "end"):
            end = float(raw.end)
        elif hasattr(raw, "duration"):
            end = start + float(raw.duration)
        else:
            end = start
        return (start, max(end, start), text)
    except (TypeError, ValueError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_transcript(
    cues: Sequence[Any],
    video_duration_sec: float | None,
    *,
    min_coverage: float = 0.90,
    max_gap_sec: float = 20.0,
    max_first_delay_sec: float = 8.0,
    max_empty_ratio: float = 0.10,
) -> TranscriptQuality:
    """Check whether a transcript adequately covers a video.

    Parameters
    ----------
    cues:
        Transcript cues in any of the three supported formats (raw dicts,
        ``IngestTranscriptCue``, ``TranscriptCue``).  Invalid cues are
        silently skipped.
    video_duration_sec:
        Known video duration.  When ``None`` or ``<= 0``, coverage is
        estimated from the transcript itself (last cue end).
    min_coverage:
        Minimum ``last_cue_end / video_duration`` to pass (default 0.90).
    max_gap_sec:
        Maximum gap between consecutive cues before a warning is raised.
    max_first_delay_sec:
        Maximum seconds before the first cue before a warning is raised.
    max_empty_ratio:
        Maximum fraction of cues that are empty/trivial (< 2 chars).

    Returns
    -------
    TranscriptQuality
        Quality metrics and ``is_adequate`` flag.  The caller decides whether
        to penalise, retry, or proceed.
    """
    if not cues:
        return TranscriptQuality(
            coverage_ratio=0.0,
            largest_gap_sec=0.0,
            first_cue_delay_sec=0.0,
            empty_cue_ratio=0.0,
            avg_cue_duration=0.0,
            cue_count=0,
            is_adequate=False,
            warnings=["Transcript has no cues"],
        )

    # --- normalise ---
    normalised: list[tuple[float, float, str]] = []
    for raw in cues:
        parsed = _normalise_cue(raw)
        if parsed is not None:
            normalised.append(parsed)

    if not normalised:
        return TranscriptQuality(
            coverage_ratio=0.0,
            largest_gap_sec=0.0,
            first_cue_delay_sec=0.0,
            empty_cue_ratio=0.0,
            avg_cue_duration=0.0,
            cue_count=0,
            is_adequate=False,
            warnings=["All cues failed to parse"],
        )

    # Sort by start time for gap analysis
    normalised.sort(key=lambda c: c[0])

    # --- basic stats ---
    first_cue_delay = normalised[0][0]
    last_cue_end = max(c[1] for c in normalised)
    durations = [max(0.0, c[1] - c[0]) for c in normalised]
    avg_dur = sum(durations) / len(durations) if durations else 0.0
    empty_count = sum(1 for c in normalised if len(c[2].strip()) < 2)
    empty_ratio = empty_count / len(normalised)

    # --- largest gap between consecutive cues ---
    largest_gap = 0.0
    for i in range(1, len(normalised)):
        gap = normalised[i][0] - normalised[i - 1][1]
        if gap > largest_gap:
            largest_gap = gap

    # --- coverage ratio ---
    effective_duration = (
        video_duration_sec
        if (video_duration_sec and video_duration_sec > 0)
        else last_cue_end
    )
    coverage = last_cue_end / effective_duration if effective_duration > 0 else 1.0

    # --- assemble warnings ---
    warnings: list[str] = []

    if coverage < min_coverage:
        warnings.append(
            f"Transcript covers {coverage:.0%} of {effective_duration:.0f}s video "
            f"(last cue ends at {last_cue_end:.1f}s)"
        )

    if largest_gap > max_gap_sec:
        warnings.append(
            f"Largest gap between cues is {largest_gap:.1f}s (threshold {max_gap_sec:.0f}s)"
        )

    if first_cue_delay > max_first_delay_sec:
        warnings.append(
            f"First cue starts at {first_cue_delay:.1f}s (threshold {max_first_delay_sec:.0f}s)"
        )

    if empty_ratio > max_empty_ratio:
        warnings.append(
            f"{empty_count}/{len(normalised)} cues are empty or very short"
        )

    if avg_dur > 30.0:
        warnings.append(
            f"Average cue duration is {avg_dur:.1f}s "
            f"(unusually long, may indicate chunked captions)"
        )

    if avg_dur < 0.1 and len(normalised) > 10:
        warnings.append(
            f"Average cue duration is {avg_dur:.3f}s (unusually short)"
        )

    return TranscriptQuality(
        coverage_ratio=round(coverage, 4),
        largest_gap_sec=round(largest_gap, 2),
        first_cue_delay_sec=round(first_cue_delay, 2),
        empty_cue_ratio=round(empty_ratio, 4),
        avg_cue_duration=round(avg_dur, 3),
        cue_count=len(normalised),
        is_adequate=len(warnings) == 0,
        warnings=warnings,
    )
