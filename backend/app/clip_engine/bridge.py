"""
Bridge: translates clip_engine output into IngestionPipeline._persist_ingest() inputs.

All functions are pure (no I/O, no DB) so they can be unit-tested in isolation.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..ingestion.models import IngestTranscriptCue, IngestMetadata, IngestSegment
from ..ingestion.adapters.base import AdapterResult


# ── Transcript helpers ────────────────────────────────────────────────────────


def to_cues(transcript: dict) -> list[IngestTranscriptCue]:
    """Convert transcript segments to IngestTranscriptCues, skipping blank text."""
    cues = []
    for seg in transcript.get("segments", []):
        if not seg.get("text", "").strip():
            continue
        cues.append(IngestTranscriptCue(start=seg["start"], end=seg["end"], text=seg["text"]))
    return cues


def window_text(transcript: dict, t0: float, t1: float) -> str:
    """Return joined text of segments whose window strictly overlaps (t0, t1).

    A segment that merely touches the boundary (end==t0 or start==t1) is excluded
    so that consecutive non-overlapping clips do not bleed into each other.
    """
    parts = []
    for seg in transcript.get("segments", []):
        if seg["end"] <= t0:
            continue
        if seg["start"] >= t1:
            break
        if seg.get("text", "").strip():
            parts.append(seg["text"])
    return " ".join(parts)


# ── Model constructors ────────────────────────────────────────────────────────


def to_metadata(video_id: str, meta: dict, source_url: str) -> IngestMetadata:
    """Build IngestMetadata from a raw meta dict (Supadata / yt-dlp info_dict)."""
    raw_vc = meta.get("view_count")
    if isinstance(raw_vc, int):
        view_count = raw_vc
    elif isinstance(raw_vc, str) and raw_vc.strip().isdigit():
        view_count = int(raw_vc.strip())
    else:
        view_count = None

    return IngestMetadata(
        platform="yt",
        source_id=video_id,
        source_url=source_url,
        playback_url=f"https://www.youtube.com/embed/{video_id}",
        title=meta.get("title", ""),
        description=meta.get("description", ""),
        author_name=meta.get("author_name", ""),
        duration_sec=meta.get("duration_sec"),
        thumbnail_url=meta.get("thumbnail_url", ""),
        view_count=view_count,
    )


def synth_adapter_result(video_id: str, source_url: str) -> AdapterResult:
    """Construct a minimal AdapterResult satisfying _persist_ingest's contract."""
    return AdapterResult(
        platform="yt",
        source_id=video_id,
        source_url=source_url,
        playback_url=f"https://www.youtube.com/embed/{video_id}",
        video_path=Path("."),
        info_dict={},
    )


def to_segment(clip: dict, transcript: dict) -> IngestSegment:
    """Build IngestSegment from a clip dict; text is in-window transcript or clip title."""
    t0, t1 = clip["start"], clip["end"]
    text = window_text(transcript, t0, t1) or clip.get("title", "")
    return IngestSegment(
        t_start=t0,
        t_end=t1,
        text=text,
        score=float(clip.get("score", 1.0)),
    )


# ── Relevance filter ──────────────────────────────────────────────────────────


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def relevance_score(clip: dict, transcript: dict, query: str | None) -> float:
    """
    Token-overlap relevance of a clip against a query.

    Returns 1.0 when query is falsy (no filtering intent).
    Otherwise: |query_tokens ∩ clip_tokens| / max(1, |query_tokens|).
    clip_tokens = tokens of (clip title + in-window transcript text).
    """
    if not query:
        return 1.0

    query_tokens = _tokens(query)
    if not query_tokens:
        return 1.0

    t0, t1 = clip["start"], clip["end"]
    clip_text = clip.get("title", "") + " " + window_text(transcript, t0, t1)
    clip_tokens = _tokens(clip_text)

    overlap = len(query_tokens & clip_tokens)
    return overlap / max(1, len(query_tokens))


def filter_by_query(
    clips: list[dict],
    transcript: dict,
    query: str | None,
    *,
    floor: float = 0.0,
) -> list[dict]:
    """
    When query is falsy, return clips unchanged.
    Otherwise: annotate each clip with clip["score"], drop clips at/below floor,
    sort by score descending.
    """
    if not query:
        return clips

    scored = []
    for clip in clips:
        clip["score"] = relevance_score(clip, transcript, query)
        if clip["score"] > floor:
            scored.append(clip)

    scored.sort(key=lambda c: c["score"], reverse=True)
    return scored


__all__ = [
    "to_cues",
    "window_text",
    "to_metadata",
    "synth_adapter_result",
    "to_segment",
    "relevance_score",
    "filter_by_query",
]
