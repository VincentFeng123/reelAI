"""build_structure — run the topic-independent understanding stages (spec §1–6).

Phase 1 is transcript-only: content map → atomic units (role + concepts + references
in one pass) → dependency graph. Perception (scenes/OCR/vision/diarization) is layered
in ahead of unit extraction in Phase 2; here ``has_perception`` stays False and visual
fields stay empty. The result is a ``Structure`` the caller caches per video_id.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Optional

from ... import config
from ..sentences import Sentence
from .content_map import build_content_map
from .dependencies import build_dependency_graph
from .models import Structure, sentence_fingerprint
from .units import drift_stats, extract_units

ProgressCb = Optional[Callable[[float, str], None]]


def _scale(lo: float, hi: float, frac: float) -> float:
    return lo + (hi - lo) * max(0.0, min(1.0, frac))


def build_structure(video_id: str, transcript: dict, sentences: list[Sentence], adapter,
                    detection, settings: dict, progress: ProgressCb = None,
                    perception=None, content_map=None) -> Structure:
    def sub(lo: float, hi: float):
        def cb(frac: float, msg: str = "") -> None:
            if progress:
                progress(_scale(lo, hi, frac), msg)
        return cb

    # content_map may be precomputed by the caller so it can run CONCURRENTLY with the perception
    # branch (cross-stage pipelining — content_map needs only `sentences`, perception needs the
    # video). When None we build it here exactly as before; identical `sentences`/`settings` yield
    # an identical map, so a precomputed one is byte-for-byte equivalent — only the schedule moves.
    if content_map is None:
        content_map = build_content_map(sentences, settings, sub(0.00, 0.30))
    units = extract_units(sentences, content_map, adapter, settings, sub(0.30, 0.80), perception)
    dependencies = build_dependency_graph(units, settings, sub(0.80, 1.00))

    visual_events = list(perception.visual_events) if perception else []
    degraded = list(perception.degraded) if perception else []
    if content_map.engine == "llm-fallback":
        degraded.append("content_map")             # treeseg failed; legacy LLM engine used
    # W25-B drift telemetry: boundary clamps (LLM double-claimed sentences; every later unit
    # in the topic shifted) + topic sentences no unit covers — both previously silent.
    n_clamped, n_uncovered = drift_stats(units, content_map, len(sentences))
    if n_clamped or n_uncovered:
        degraded.append(f"unit_drift: {n_clamped} boundary_clamped unit(s), "
                        f"{n_uncovered} uncovered sentence(s)")
    degraded.extend(dependencies.degraded)         # W25-B: LLM edge-pass failure, not silent
    return Structure(
        video_id=video_id,
        title=transcript.get("title", "") or "",
        duration=float(transcript.get("duration", 0.0) or 0.0),
        detection=detection,
        content_map=content_map,
        units=units,
        dependencies=dependencies,
        visual_events=visual_events,
        has_perception=bool(visual_events),
        degraded=degraded,
        # W25-A build provenance: every sentence index above is only valid against THIS
        # sentence list — load_structure refuses the cache when the live list differs.
        n_sentences=len(sentences),
        sentence_fingerprint=sentence_fingerprint(sentences),
        prompt_version=config.UNDERSTANDING_PROMPT_VERSION,
        built_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
