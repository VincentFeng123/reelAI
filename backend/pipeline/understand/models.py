"""The source-of-truth data model for video understanding.

Persisted per video to ``work/<video_id>/structure.json`` (bumping ``SCHEMA_VERSION``
invalidates the cache). Phase 1 populates the transcript-derived fields; visual fields
(``VisualEvent``, ``VisualDependency``) stay empty until Phase 2 perception is wired.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from ... import config
from ...adapters.detect import DetectionResult

SCHEMA_VERSION = 4   # 4: content-map boundaries are embedding-derived (TreeSeg); cached topic
                     #    partitions / sentence_ranges from the LLM engine are stale; rebuild.
                     # 3: punctuation-restoration stage changes sentence segmentation
                     # 2: Phase-2 perception (visual_events populated, visual_dependencies linked)


# ── atomic building blocks ───────────────────────────────────────────────────
class Reference(BaseModel):
    text: str                                  # e.g. "as we saw earlier", "this integral"
    resolves_to: str = ""                      # concept/entity it points at
    source_unit: Optional[str] = None          # unit_id it refers back to (filled in deps)


class VisualDependency(BaseModel):
    kind: str = "slide"                        # slide|equation|diagram|code|chart|demo
    keyframe_time: float = 0.0
    description: str = ""
    visual_event_id: Optional[str] = None


class SourceConfidence(BaseModel):
    transcript: float = 1.0
    visual: float = 0.0
    overall: float = 1.0


class VisualEvent(BaseModel):
    event_id: str
    start: float
    end: float
    kind: str = "other"
    text: str = ""
    description: str = ""
    confidence: float = 0.0


class Scene(BaseModel):
    """A keyframe-anchored span of the video (from ffmpeg scene cuts ∪ a uniform grid)."""
    index: int
    start: float
    end: float
    keyframe_time: float
    keyframe_path: Optional[str] = None


class SpeakerTurn(BaseModel):
    """A diarized speaker turn (Phase 3; empty list until diarization runs)."""
    start: float
    end: float
    speaker: str                               # e.g. "SPEAKER_00"


class Unit(BaseModel):
    unit_id: str
    start: float
    end: float
    sentence_range: tuple[int, int]
    node_id: str = ""
    topic: str = ""
    role: str = ""
    role_domain: str = ""
    summary: str = ""
    transcript: str = ""
    claims: list[str] = Field(default_factory=list)
    concepts_introduced: list[str] = Field(default_factory=list)
    concepts_required: list[str] = Field(default_factory=list)
    equations: list[str] = Field(default_factory=list)
    references: list[Reference] = Field(default_factory=list)
    visual_dependencies: list[VisualDependency] = Field(default_factory=list)
    speaker: Optional[str] = None
    source_confidence: float = 1.0             # scalar 0..1 (Part B scoring reads this)
    warnings: list[str] = Field(default_factory=list)


class ContentNode(BaseModel):
    node_id: str                               # "ch1", "ch1.t2", "ch1.t2.s1"
    level: str                                 # video|chapter|topic|subtopic
    title: str = ""
    summary: str = ""
    start: float = 0.0
    end: float = 0.0
    parent_id: Optional[str] = None
    children_ids: list[str] = Field(default_factory=list)
    sentence_range: tuple[int, int] = (0, 0)
    keywords: list[str] = Field(default_factory=list)


class ContentMap(BaseModel):
    root_id: str = "video"
    nodes: list[ContentNode] = Field(default_factory=list)
    engine: str = ""                           # "treeseg" | "llm" | "llm-fallback" (see content_map)

    def _by_level(self, *levels: str) -> list[ContentNode]:
        return [n for n in self.nodes if n.level in levels]

    def chapters(self) -> list[ContentNode]:
        return self._by_level("chapter")

    def topics(self) -> list[ContentNode]:
        """The unit-bearing partition — one atomic-unit LLM pass runs per topic. Subtopics are
        a finer structural annotation and are NOT returned here (they must not re-partition the
        transcript for unit extraction)."""
        return self._by_level("topic") or self._by_level("subtopic")

    def subtopics(self) -> list[ContentNode]:
        return self._by_level("subtopic")


class Edge(BaseModel):
    source: str
    target: str
    relation: str                              # defines|requires|explains|illustrates|
                                               # continues|answers|contradicts|summarizes|
                                               # visually_depends_on|refers_to
    weight: float = 1.0
    rationale: str = ""
    derivation: str = "rule"                   # rule | llm


class DependencyGraph(BaseModel):
    edges: list[Edge] = Field(default_factory=list)
    # W25-B telemetry (pydantic-additive; old caches default to clean). ``degraded`` carries
    # graph-build notes (e.g. the LLM edge pass failing — previously a bare except) which
    # build_structure folds into Structure.degraded; ``forward_requires_count`` is the
    # graph lint for requires edges pointing FORWARD in time (must be 0 post-W25-B — the
    # rule-edge lst[0] future-introducer fallback was the producer).
    degraded: list[str] = Field(default_factory=list)
    forward_requires_count: int = 0


# ── the persisted bundle ─────────────────────────────────────────────────────
class Structure(BaseModel):
    schema_version: int = SCHEMA_VERSION
    video_id: str
    title: str = ""
    duration: float = 0.0
    detection: DetectionResult = Field(default_factory=DetectionResult)
    content_map: ContentMap = Field(default_factory=ContentMap)
    units: list[Unit] = Field(default_factory=list)
    dependencies: DependencyGraph = Field(default_factory=DependencyGraph)
    visual_events: list[VisualEvent] = Field(default_factory=list)
    has_perception: bool = False
    degraded: list[str] = Field(default_factory=list)
    # build provenance (W25-A) — pydantic-additive; the defaults MEAN "unknown ⇒ stale".
    # Every sentence index in this bundle (unit/node sentence_ranges) is only meaningful
    # against the exact sentence list it was built on; a cache built on a different
    # sentence universe (legacy pysbd vs punctuation-restored — 322 vs 183 sentences on
    # qP) silently poisons assembly, so load_structure refuses it unless overridden.
    n_sentences: int = 0                       # len(sentences) at build; 0 ⇒ pre-provenance cache
    sentence_fingerprint: str = ""             # sentence_fingerprint(sentences); "" ⇒ unknown
    prompt_version: str = ""                   # config.UNDERSTANDING_PROMPT_VERSION at build
    built_at: str = ""                         # ISO-8601 UTC build time (informational)

    # runtime helpers (not persisted state) ----------------------------------
    def units_by_id(self) -> dict[str, Unit]:
        return {u.unit_id: u for u in self.units}

    def visual_summary(self, start: float, end: float) -> str:
        """On-screen text/description overlapping [start,end] (empty in Phase 1)."""
        parts = []
        for ve in self.visual_events:
            if ve.end >= start and ve.start <= end:
                bit = (ve.text or ve.description or "").strip()
                if bit:
                    parts.append(f"[{ve.kind}] {bit}")
        return " | ".join(parts)[:1200]


# ── perception bundle (Phase 2/3) ────────────────────────────────────────────
class Perception(BaseModel):
    """Topic-independent visual/audio perception, cached to ``work/<id>/perception.json``.

    Produced by ``perceive()`` before unit extraction; its ``visual_events`` become
    ``Structure.visual_events`` and steer ``Unit.visual_dependencies``. Every sub-stage
    degrades independently (recorded in ``degraded``), never hard-crashing the job.
    """
    schema_version: int = SCHEMA_VERSION           # gate the cache like Structure does
    video_id: str
    scenes: list[Scene] = Field(default_factory=list)
    visual_events: list[VisualEvent] = Field(default_factory=list)
    diarization: list[SpeakerTurn] = Field(default_factory=list)   # Phase 3
    degraded: list[str] = Field(default_factory=list)


# ── caching ──────────────────────────────────────────────────────────────────
def _structure_path(video_id: str) -> Path:
    return config.WORK_DIR / video_id / "structure.json"


def sentence_fingerprint(sentences) -> str:
    """Deterministic sha256 over the sentence TEXTS (order-sensitive, record-separated so
    ("ab","c") ≠ ("a","bc")). Texts alone suffice: a different segmentation/punctuation
    pass changes the texts, and the texts are what every sentence index points into."""
    h = hashlib.sha256()
    for s in sentences:
        h.update((s.text or "").encode("utf-8", "replace"))
        h.update(b"\x1e")
    return h.hexdigest()


def structure_is_stale(structure: Structure, sentences) -> Optional[str]:
    """None when the cached structure is fresh against the LIVE sentence list, else a
    human-readable staleness reason. Missing provenance (pre-W25-A cache) IS stale —
    'unknown' must never pass for 'fresh' (that silence is how the cross-indexer cache
    poisoning shipped clips indexed against the wrong sentence universe)."""
    if structure.n_sentences <= 0 or not structure.sentence_fingerprint \
            or not structure.prompt_version:
        return "missing build provenance (pre-freshness cache)"
    if structure.prompt_version != config.UNDERSTANDING_PROMPT_VERSION:
        return (f"prompt_version {structure.prompt_version!r} != "
                f"live {config.UNDERSTANDING_PROMPT_VERSION!r}")
    if structure.n_sentences != len(sentences):
        return f"n_sentences {structure.n_sentences} != live {len(sentences)}"
    if structure.sentence_fingerprint != sentence_fingerprint(sentences):
        return "sentence_fingerprint mismatch (same count, different texts)"
    return None


def save_structure(structure: Structure) -> None:
    p = _structure_path(structure.video_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(structure.model_dump_json(), encoding="utf-8")


def load_structure(video_id: str, sentences=None, *,
                   allow_stale: bool = False) -> Optional[Structure]:
    """Load the cached Structure, gated on schema AND freshness (W25-A).

    ``sentences`` is the caller's LIVE sentence list (every call site that goes on to
    assemble threads it); a stale cache — wrong fingerprint/count/prompt_version or
    missing provenance — returns None so the normal path rebuilds. ``allow_stale=True``
    is the explicit hold-anyway override (eval --freeze / cached-inputs-only probes):
    it loads the stale bundle with a LOUD stderr warning, never silently.
    ``sentences=None`` skips the freshness check (no live index to compare against)."""
    p = _structure_path(video_id)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if data.get("schema_version") != SCHEMA_VERSION:
            return None
        st = Structure.model_validate(data)
    except Exception:
        return None
    if sentences is None:
        return st
    reason = structure_is_stale(st, sentences)
    if reason is None:
        return st
    if allow_stale:
        print(f"[structure] WARNING: cached structure for {video_id} is STALE ({reason}) — "
              "loading anyway (allow_stale/freeze override); sentence indices may not match "
              "the live transcript", file=sys.stderr)
        return st
    return None
