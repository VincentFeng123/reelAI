"""Adapter registry + selection.

Concrete adapters self-register on import here. ``select_adapter`` runs content-type
detection (or honors a ``domain_override`` setting) and returns the adapter to use plus
the ``DetectionResult``. Unknown domains fall back to the generic adapter.
"""
from __future__ import annotations

from .base import BaseAdapter
from .coding import CodingAdapter
from .debate import DebateAdapter
from .detect import DetectionResult, detect_content_type
from .entertainment import EntertainmentAdapter
from .generic import GenericAdapter
from .interview import InterviewAdapter
from .lecture import LectureAdapter
from .news import NewsAdapter
from .recipe import RecipeAdapter
from .review import ReviewAdapter
from .sports import SportsAdapter
from .story import StoryAdapter
from .tutorial import TutorialAdapter

_REGISTRY: dict[str, BaseAdapter] = {}


def register(adapter: BaseAdapter) -> None:
    _REGISTRY[adapter.domain] = adapter


def get_adapter(domain: str) -> BaseAdapter:
    return _REGISTRY.get(domain, _REGISTRY["generic"])


def select_adapter(
    transcript: dict, settings: dict | None = None, meta: dict | None = None
) -> tuple[BaseAdapter, DetectionResult]:
    """Pick the adapter for this video. ``meta`` is an optional yt-dlp metadata dict threaded to
    the detector for the genre-metadata signal; it defaults None (backward-compatible — callers
    that pass no meta get the pure-LLM detection they had before)."""
    settings = settings or {}
    override = settings.get("domain_override")
    if override:
        a = get_adapter(override)
        return a, DetectionResult(content_type="forced", domain=a.domain, confidence=1.0)
    det = detect_content_type(transcript, settings, meta)
    return get_adapter(det.domain), det


# Adapters. Order: generic first so it's the fallback.
register(GenericAdapter())
register(LectureAdapter())
register(InterviewAdapter())
register(CodingAdapter())
register(TutorialAdapter())
register(DebateAdapter())
register(RecipeAdapter())
register(ReviewAdapter())
register(StoryAdapter())
register(SportsAdapter())
register(NewsAdapter())
register(EntertainmentAdapter())

__all__ = ["register", "get_adapter", "select_adapter", "DetectionResult", "BaseAdapter"]
