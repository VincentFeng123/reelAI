"""Topic-independent video understanding (spec stages 1–6).

``build_structure`` turns a transcript (+ sentences, + optional perception) into a
``Structure``: a hierarchical content map, atomic discourse units with universal/domain
roles + concepts + references + equations, and a dependency graph. The result is cached
per ``video_id`` so re-clipping the same video for a new topic skips all of this.
"""
from .models import (  # noqa: F401
    ContentMap,
    ContentNode,
    DependencyGraph,
    Edge,
    Perception,
    Reference,
    Scene,
    SourceConfidence,
    SpeakerTurn,
    Structure,
    Unit,
    VisualDependency,
    VisualEvent,
    load_structure,
    save_structure,
    sentence_fingerprint,
    structure_is_stale,
)
from .build import build_structure  # noqa: F401
