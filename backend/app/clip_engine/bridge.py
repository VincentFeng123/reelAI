"""
Bridge: translates clip_engine output into IngestionPipeline._persist_ingest() inputs.

All functions are pure (no I/O, no DB) so they can be unit-tested in isolation.
"""

from __future__ import annotations

import re
import unicodedata

from ..ingestion.models import IngestTranscriptCue, IngestMetadata, IngestSegment, YouTubeSourceRef
from .metadata import normalize_youtube_video_id

_TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)


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


def cue_text(transcript: dict, cue_ids: object) -> str:
    """Return text from exactly the requested immutable cue ids, in cue order."""
    if not isinstance(cue_ids, list) or not cue_ids:
        return ""
    selected = {str(cue_id) for cue_id in cue_ids}
    parts: list[str] = []
    for index, segment in enumerate(transcript.get("segments", [])):
        cue_id = str(segment.get("cue_id") or f"cue-{index}")
        text = str(segment.get("text") or "").strip()
        if cue_id in selected and text:
            parts.append(text)
    return " ".join(parts)


# ── Model constructors ────────────────────────────────────────────────────────


def to_metadata(video_id: str, meta: dict, source_url: str) -> IngestMetadata:
    """Build IngestMetadata from a raw meta dict (Supadata / yt-dlp info_dict)."""
    raw_vc = meta.get("view_count", meta.get("viewCount"))
    if isinstance(raw_vc, int):
        view_count = raw_vc
    elif isinstance(raw_vc, str) and raw_vc.strip().isdigit():
        view_count = int(raw_vc.strip())
    else:
        view_count = None

    source_id = normalize_youtube_video_id(video_id) or str(video_id).strip()
    duration = meta.get("duration_sec", meta.get("duration"))
    try:
        duration_sec = float(duration) if duration not in (None, "") else None
    except (TypeError, ValueError, OverflowError):
        duration_sec = None
    return IngestMetadata(
        platform="yt",
        source_id=source_id,
        source_url=source_url,
        playback_url=f"https://www.youtube.com/embed/{source_id}",
        title=meta.get("title") or "",
        description=meta.get("description") or "",
        author_name=meta.get("author_name") or meta.get("channel") or meta.get("channelTitle") or "",
        author_url=meta.get("author_url") or meta.get("channel_url") or "",
        channel_id=meta.get("channel_id") or meta.get("channelId") or "",
        duration_sec=duration_sec,
        thumbnail_url=meta.get("thumbnail_url") or meta.get("thumbnail") or "",
        upload_date_iso=meta.get("upload_date_iso") or meta.get("published_at") or meta.get("uploadDate") or None,
        view_count=view_count,
    )


def synth_adapter_result(video_id: str, source_url: str) -> YouTubeSourceRef:
    """Construct the canonical YouTube source reference used by persistence."""
    return YouTubeSourceRef(
        source_id=video_id,
        source_url=source_url,
        playback_url=f"https://www.youtube.com/embed/{video_id}",
    )


def to_segment(clip: dict, transcript: dict) -> IngestSegment:
    """Build IngestSegment from a clip dict; text is in-window transcript or clip title."""
    t0, t1 = round(float(clip["start"]), 3), round(float(clip["end"]), 3)
    text = cue_text(transcript, clip.get("cue_ids")) or window_text(transcript, t0, t1)
    text = text or clip.get("title", "")
    return IngestSegment(
        t_start=t0,
        t_end=t1,
        text=text,
        score=float(clip.get("score", 1.0)),
    )


# ── Relevance filter ──────────────────────────────────────────────────────────


def _tokens(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return {match.group(0) for match in _TOKEN_RE.finditer(normalized)}


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
    exact = cue_text(transcript, clip.get("cue_ids"))
    clip_text = clip.get("title", "") + " " + (exact or window_text(transcript, t0, t1))
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


# ── Best-clip selection ───────────────────────────────────────────────────────


def pick_best_clip(clips: list[dict], target_sec: float, max_sec: float) -> dict:
    """Pick the single best clip for a single-clip endpoint: prefer clips whose
    duration is within max_sec (short-form UX), and among those the one closest to
    target_sec. If NO clip is within max_sec, return the closest-to-target overall —
    we return the full topic clip rather than clamping it mid-thought (the gemini
    engine ships whole self-contained topics)."""
    def dur(c: dict) -> float:
        return float(c["end"]) - float(c["start"])
    in_bounds = [c for c in clips if dur(c) <= max_sec]
    pool = in_bounds or clips
    return min(pool, key=lambda c: abs(dur(c) - target_sec))


__all__ = [
    "to_cues",
    "window_text",
    "cue_text",
    "to_metadata",
    "synth_adapter_result",
    "to_segment",
    "relevance_score",
    "filter_by_query",
    "pick_best_clip",
]
