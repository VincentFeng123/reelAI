"""Shared types for clip assembly."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Candidate:
    cand_id: str
    anchor_id: str
    role: str
    facet: str
    title: str
    reason: str
    unit_ids: list[str]                        # inlined units (anchor + inline context), time-ordered
    referential: list[tuple[str, str]]         # (unit_id, relation) surfaced via card, not inlined
    i_start: int                               # sentence indices (for boundary reuse)
    i_end: int
    start: float
    end: float
    relevance: float = 0.0                     # topic relevance of the anchor (0..1)
    priority: float = 0.0                      # anchor role priority (0..1)
    truncated: bool = False                    # closure hit its budget with needs unmet
    arc_id: str = ""                           # detected-arc provenance (P3): set by
                                               # build_arc_candidate, "" for unit anchors —
                                               # eval counts n_arc_clips_shipped off this
    # filled during validation / scoring:
    contract_role: str = ""                    # content-bound contract (P1): completeness gates,
                                               # judge hints, and scoring key off THIS — rebound
                                               # after every span mutation. `role` above stays the
                                               # ANCHOR's role (provenance). "" → falls back to role.
    verdict: Optional[object] = None
    judged_text_hash: str = ""                 # hash of the EXACT text sent to the judge (set on the outage path too)
    attempts: int = 0
    n_trims: int = 0                           # repair trim moves taken (P2): trim-lattice
                                               # probes judged during bisection — eval sums
                                               # these into the per-video n_trims column
    warnings: tuple[str, ...] = ()             # candidate-level warnings (union'd into the spec's)
    ship_flagged: bool = False                 # shipped past the kill gate on unverifiable evidence
    n_failure_reasons: int = 0                 # failure reasons on the final verdict (B7 stats)
    n_verified: int = 0                        # …of which passed evidence-quote verification
    verified_kinds: tuple[str, ...] = ()       # W25-G kind-level mirror of the counts above —
    unverified_kinds: tuple[str, ...] = ()     # phantom_quotable_rate needs WHICH kinds failed
    completeness_score: float = 0.0
    grounding_score: float = 0.0
    boundary_score: float = 1.0
    final_quality: float = 0.0
    context_card: str = ""
    prerequisite_anchor_ids: list[str] = field(default_factory=list)


@dataclass
class Need:
    unit_id: str
    relation: str
    weight: float
    gap: float
