"""Run perception → a ``Perception`` bundle, cached to ``work/<id>/perception.json``.

Orchestrates the topic-independent visual pass (ffmpeg scenes/keyframes → Gemini-vision
captions → optional local OCR merge) ahead of unit extraction. Each sub-stage degrades
independently — recorded in ``Perception.degraded`` — so a missing dep/key/exception yields
a Phase-1-equivalent (transcript-only) run rather than a crashed job. Cached by ``video_id``
so re-clipping the same video for a new topic skips perception entirely.

The cache is *not* sticky against transient failures: on a cache hit whose ``degraded`` names a
capability that is now available (ffmpeg installed, GEMINI_API_KEY set, an API outage passed),
perception is recomputed so a recovered environment upgrades the result. A ``schema_version``
mismatch also invalidates the cache, mirroring ``load_structure``. A legitimately visual-free
video (vision ran, found nothing) is NOT marked degraded, so it is cached once and not retried.

``audio_path`` is accepted (and unused here) so Phase-3 diarization can slot in without
changing the call sites.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, Optional

from ... import config
from ..sentences import Sentence
from .models import SCHEMA_VERSION, Perception
from . import diarize as diarize_mod
from . import ocr as ocr_mod
from . import scenes as scenes_mod
from . import vision as vision_mod

ProgressCb = Optional[Callable[[float, str], None]]


def _cache_path(video_id: str) -> Path:
    return config.WORK_DIR / video_id / "perception.json"


def _band(progress: ProgressCb, lo: float, hi: float):
    def cb(frac: float, msg: str = "") -> None:
        if progress:
            progress(lo + (hi - lo) * max(0.0, min(1.0, frac)), msg)
    return cb


def _capability_available(cap: str) -> bool:
    """Is the capability behind a ``degraded`` marker usable right now?"""
    if cap == "vision":
        return vision_mod.available()
    if cap == "ocr":
        return ocr_mod.available()
    if cap == "diarization":
        return diarize_mod.available()
    if cap == "scenes":
        b = config.FFMPEG_BIN
        return bool(shutil.which(b) or Path(b).exists())
    return False                                            # unknown marker → don't force recompute


def _load_cached(video_id: str) -> Optional[Perception]:
    cache = _cache_path(video_id)
    if not (config.STRUCTURE_CACHE and cache.exists()):
        return None
    try:
        cached = Perception.model_validate_json(cache.read_text(encoding="utf-8"))
    except Exception:
        return None
    if getattr(cached, "schema_version", None) != SCHEMA_VERSION:
        return None                                        # stale shape → rebuild
    if any(_capability_available(cap) for cap in cached.degraded):
        return None                                        # a degraded capability recovered → recompute
    return cached


def perceive(video_path: str, video_id: str, transcript: dict, sentences: list[Sentence],
             settings: dict, progress: ProgressCb = None, audio_path: Optional[str] = None) -> Perception:
    cached = _load_cached(video_id)
    if cached is not None:
        return cached

    duration = float(transcript.get("duration") or 0.0) or (sentences[-1].end if sentences else 0.0)
    degraded: list[str] = []

    def nearby_text(t: float) -> str:
        return " ".join((s.text or "") for s in sentences if s.start <= t + 8 and s.end >= t - 8)

    # 1. scenes + keyframes (ffmpeg) ----------------------------------------
    try:
        scs = scenes_mod.detect_and_extract(video_path, video_id, duration, _band(progress, 0.0, 0.45))
    except Exception:
        scs = []
    if not scs:
        degraded.append("scenes")                          # any real video has frames → empty ⇒ failure

    # 2. vision captions (Gemini) — primary visual pass ---------------------
    #    Degrade only when vision is unavailable or every batch failed; an empty result from a
    #    successful pass is a legitimately visual-free video, not a degradation (so it isn't retried).
    ves = []
    if scs:
        if not vision_mod.available():
            degraded.append("vision")
        else:
            try:
                ves = vision_mod.describe_keyframes(scs, nearby_text, _band(progress, 0.45, 0.90))
            except Exception:
                ves, degraded = [], degraded + ["vision"]

    # 3. optional local OCR merge (default off) -----------------------------
    if ves and ocr_mod.available():
        try:
            blocks = ocr_mod.ocr_keyframes(scs, progress=_band(progress, 0.90, 0.95))
            ocr_mod.merge_into_events(ves, blocks)
        except Exception:
            degraded.append("ocr")

    # 4. optional speaker diarization (Phase 3; default off) ----------------
    #    Same degrade contract as vision: mark degraded only when requested-but-unavailable or it
    #    raises, so a recovered token/dep triggers a recompute via _capability_available.
    turns = []
    if settings.get("diarization", config.DIARIZATION_ENABLED):
        if not diarize_mod.available() or not audio_path:
            degraded.append("diarization")
        else:
            try:
                turns = diarize_mod.diarize(audio_path, video_id, _band(progress, 0.95, 1.0))
            except Exception:
                degraded.append("diarization")
    if progress:
        progress(1.0, "Perception complete")

    per = Perception(video_id=video_id, scenes=scs, visual_events=ves,
                     diarization=turns, degraded=degraded)
    try:
        _cache_path(video_id).write_text(per.model_dump_json(), encoding="utf-8")
    except Exception:
        pass
    return per
