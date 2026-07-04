"""Content-type detection → adapter routing (spec §11).

One cheap LLM call over a head+tail transcript sample classifies the genre and its
information density, then maps to an adapter key. Never hard-crashes: on any failure it
returns a generic/medium result. Density feeds the per-video anchor budget.
"""
from __future__ import annotations

import re

from pydantic import BaseModel

from .. import config


class DetectionResult(BaseModel):
    content_type: str = "other"
    domain: str = "generic"           # resolved post-hoc from CONTENT_TYPE_TO_DOMAIN
    density: str = "medium"           # low | medium | high
    confidence: float = 0.5
    secondary_content_type: str = ""
    rationale: str = ""


# content_type (as emitted by the model) → adapter domain key. Unknown → generic.
CONTENT_TYPE_TO_DOMAIN = {
    "lecture": "lecture", "physics": "lecture", "math": "lecture",
    "science_explainer": "lecture", "class": "lecture", "course": "lecture",
    "tutorial": "tutorial", "howto": "tutorial", "how_to": "tutorial",
    "coding": "coding", "programming": "coding", "software": "coding",
    "interview": "interview", "podcast": "interview", "talk_show": "interview",
    "debate": "debate",
    "recipe": "recipe", "cooking": "recipe",
    "product_review": "review", "review": "review",
    "story": "story", "storytelling": "story",
    "sports": "sports", "sports_analysis": "sports",
    "news": "news", "news_report": "news",
    # entertainment family (GEN2): music/song/comedy/parody all route to the lenient
    # EntertainmentAdapter so a parody/song isn't gated by worked-problem completeness.
    "music": "entertainment", "song": "entertainment", "comedy": "entertainment",
    "parody": "entertainment", "entertainment": "entertainment",
}

DETECT_SYSTEM = (
    "You classify a video's genre from a transcript sample so a downstream system can pick "
    "the right clipping strategy. Choose the single best content_type from: lecture, physics, "
    "math, tutorial, coding, interview, podcast, debate, recipe, product_review, story, sports, "
    "news, vlog, commentary, music, song, comedy, parody, entertainment, other. Also rate "
    "information density (low|medium|high): how densely packed with distinct clip-worthy ideas "
    "it is. Output only the structured result."
)

# Metadata signal (GEN2). yt-dlp categories that mark non-lecture entertainment, plus a
# title/tags regex for music/parody. When either fires AND the LLM guessed a lecture-family
# type, metadata takes precedence and we override to entertainment (the parody-ships-0 fix).
_ENTERTAINMENT_CATEGORIES = frozenset({"music", "comedy", "entertainment"})
_LECTURE_FAMILY_DOMAINS = frozenset({"lecture", "tutorial", "coding"})
_MUSIC_TITLE_RE = re.compile(
    r"rhapsody|song|parody|lyrics|official\s+(?:music\s+)?video|cover|remix", re.IGNORECASE
)


def _metadata_entertainment_hit(meta: dict | None, title: str) -> bool:
    """True when yt-dlp metadata marks this as music/comedy/entertainment. Null artist/track is
    treated as no-signal — we never force music off a null artist (GEN1 grounded fact)."""
    if not meta:
        return False
    cats = {str(c).lower() for c in (meta.get("categories") or [])}
    if cats & _ENTERTAINMENT_CATEGORIES:
        return True
    tags = [str(t) for t in (meta.get("tags") or [])]
    haystack = " ".join([title or ""] + tags)
    return bool(_MUSIC_TITLE_RE.search(haystack))


def _apply_metadata_signal(det: DetectionResult, meta: dict | None, title: str) -> DetectionResult:
    """Metadata-precedence nudge: if metadata says entertainment but the LLM said a lecture-family
    type, override to entertainment, drop confidence, and stash the LLM guess as secondary. A
    weighted nudge — non-lecture-family LLM guesses (interview, story, …) are left untouched."""
    if not _metadata_entertainment_hit(meta, title):
        return det
    if det.domain not in _LECTURE_FAMILY_DOMAINS:
        return det
    det.secondary_content_type = det.content_type or det.secondary_content_type
    det.content_type = "entertainment"
    det.domain = "entertainment"
    det.confidence = min(det.confidence, 0.5)
    det.rationale = (det.rationale + " | metadata→entertainment").strip(" |")
    return det


def _sample(transcript: dict) -> str:
    segs = transcript.get("segments") or []
    head = " ".join((s.get("text") or "") for s in segs[: config.DETECT_SAMPLE_SEGMENTS])
    tail = " ".join((s.get("text") or "") for s in segs[-config.DETECT_TAIL_SEGMENTS:]) if segs else ""
    text = (head + (" ... " + tail if tail else "")).strip()
    if not text:                       # no segments → fall back to raw text
        text = (transcript.get("text") or "")
    return text[: config.DETECT_SAMPLE_CHARS]


def detect_content_type(
    transcript: dict, settings: dict | None = None, meta: dict | None = None
) -> DetectionResult:
    """Classify content type from a transcript sample (+ optional yt-dlp `meta`).

    `meta` is an optional yt-dlp metadata dict ({categories, tags, artist, track, genre}); it is
    backward-compatible (defaults None → no metadata signal). When present it can OVERRIDE a
    lecture-family LLM guess to entertainment (metadata precedence) — see _apply_metadata_signal.
    """
    from ..llm import llm_json
    title = transcript.get("title", "")
    user = f"TITLE: {title}\n\nSAMPLE:\n{_sample(transcript)}"
    try:
        det = llm_json(DETECT_SYSTEM, user, DetectionResult, temperature=0.0)
    except Exception:
        det = DetectionResult()
    det.domain = CONTENT_TYPE_TO_DOMAIN.get((det.content_type or "").lower(), "generic")
    det = _apply_metadata_signal(det, meta, title)
    return det
