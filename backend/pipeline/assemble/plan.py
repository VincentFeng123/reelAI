"""Per-video extraction plan (Wave 2 P3b): the model proposes WHAT to extract.

One llm_json call (authoring model, temperature 0) whose input is the video's ACTUAL
inventory — the content map (chapters/topics + spans), a compact unit table (id, role,
span, one-line summary), the detected instructional arcs, and the adapter's anchor-
priority table labeled explicitly as a PRIOR (a golf video yields demonstrations, not
worked examples — the adapter parameterizes, it does not dictate). The output is an
ordered most-valuable-first list of proposed extractions, each naming one anchor unit OR
one arc, with a role, purpose, and one-line why_valuable.

Validation is fully deterministic (``validate_plan``): unknown unit/arc ids are dropped,
duplicate/overlapping proposals are deduped (an arc supersedes unit proposals it
contains — arcs are the highest-value extraction class), unknown roles are coerced via
``roles.coerce_role``, and the list is capped at MAX_ANCHORS. Enforcement (coverage /
saturation / arc retention / hard cap) lives in ``candidates.enforce_plan``.
"""
from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from ... import config
from ...roles import coerce_role
from ..understand.arcs import ArcCandidate
from ..understand.models import Structure, Unit


class PlanItemLLM(BaseModel):
    anchor_unit_id: str = ""                   # exactly one of anchor_unit_id / arc_id
    arc_id: str = ""
    role: str = ""                             # must be a known universal/domain role
    purpose: str = ""                          # free text
    why_valuable: str = ""                     # one line


class ExtractionPlanLLM(BaseModel):
    extractions: list[PlanItemLLM] = Field(default_factory=list)


@dataclass
class PlanProposal:
    """One validated extraction proposal (plan order = rank, most valuable first)."""
    kind: str                                  # "unit" | "arc"
    ref_id: str                                # unit_id or arc_id
    role: str                                  # validated / coerced role
    purpose: str = ""
    why_valuable: str = ""
    rank: int = 0


PLAN_SYSTEM = (
    "You plan what to clip from ONE specific video. You see its actual inventory: the "
    "content map, every labeled unit, and any detected instructional arcs (complete worked "
    "examples / practice question+answer pairs). Judge WHAT is worth extracting from THIS "
    "video — a golf video yields demonstrations, not worked examples; a physics lecture "
    "yields worked examples, derivations, and practice problems. The adapter's anchor-"
    "priority table is a PRIOR reflecting what usually matters in this genre — deviate when "
    "this video's actual content warrants it. Detected arcs are the most valuable extraction "
    "class: prefer proposing an arc over any single unit inside it. Cover the video's "
    "distinct topics rather than piling proposals onto one. Order proposals most-valuable-"
    "first. Each proposal names EITHER one anchor_unit_id OR one arc_id (never both), a role "
    "from the role menu describing what the extraction IS, a short purpose, and a one-line "
    "why_valuable. Output only the structured result."
)


def _span(a: float, b: float) -> str:
    return f"{a:.0f}-{b:.0f}s"


def render_inventory(structure: Structure, units: list[Unit], arcs: list[ArcCandidate],
                     adapter, topic: str, cap: int) -> str:
    """The plan call's user prompt: the video's actual inventory + the adapter PRIOR."""
    lines: list[str] = []
    if topic and topic.strip():
        lines.append(f"VIEWER TOPIC: {topic.strip()}\n")
    cm = structure.content_map
    map_rows: list[str] = []
    for ch in cm.chapters():
        map_rows.append(f"{ch.node_id} “{ch.title}” ({_span(ch.start, ch.end)})")
    for t in cm.topics():
        summary = f": {t.summary}" if t.summary else ""
        map_rows.append(f"  {t.node_id} “{t.title}” ({_span(t.start, t.end)}){summary}")
    lines.append("CONTENT MAP:\n" + ("\n".join(map_rows) if map_rows else "(flat — no map)"))
    unit_rows = "\n".join(
        f"{u.unit_id} [{u.role}] ({_span(u.start, u.end)}) {(u.summary or u.transcript)[:110]}"
        for u in units)
    lines.append("UNITS (id [role] (span) summary):\n" + unit_rows)
    if arcs:
        arc_rows = "\n".join(
            f"{a.arc_id} [{a.arc_role}] units {','.join(a.unit_ids)}"
            + (" (verified)" if a.verified else "")
            for a in arcs)
        lines.append("DETECTED ARCS (complete instructional events — highest value):\n" + arc_rows)
    else:
        lines.append("DETECTED ARCS: (none)")
    specs_fn = getattr(adapter, "role_specs", None)
    prior_rows: list[str] = []
    if callable(specs_fn):
        anchor_specs = sorted((s for s in specs_fn().values() if s.is_anchor),
                              key=lambda s: -s.anchor_priority)
        prior_rows = [f"{s.name}: {s.anchor_priority}" for s in anchor_specs]
    lines.append("ADAPTER PRIOR — typical anchor-role priorities for this genre "
                 "(0-100; a PRIOR, not a rule):\n"
                 + ("; ".join(prior_rows) if prior_rows else "(none available)"))
    lines.append(f"Propose up to {cap} extractions, ordered most valuable first.")
    return "\n\n".join(lines)


def validate_plan(items: list[PlanItemLLM], units_by_id: dict[str, Unit],
                  arcs_by_id: dict[str, ArcCandidate], adapter, cap: int) -> list[PlanProposal]:
    """Deterministic validation: unknown ids dropped; duplicates/overlaps deduped (a later
    arc REPLACES unit proposals it contains); unknown roles coerced via roles.coerce_role;
    arc proposals always carry the arc's terminal role; capped at ``cap``."""
    valid_fn = getattr(adapter, "valid_roles", None)
    known_roles = set(valid_fn()) if callable(valid_fn) else set()
    kept: list[PlanProposal] = []
    claimed: dict[str, int] = {}               # unit_id -> index in kept that claims it

    def _claims(p: PlanProposal) -> list[str]:
        return list(arcs_by_id[p.ref_id].unit_ids) if p.kind == "arc" else [p.ref_id]

    for item in items:
        arc_id = (item.arc_id or "").strip()
        unit_id = (item.anchor_unit_id or "").strip()
        if arc_id and arc_id in arcs_by_id:    # arc wins when both are (mis)filled
            arc = arcs_by_id[arc_id]
            prop = PlanProposal("arc", arc_id, arc.terminal_role,
                                purpose=item.purpose, why_valuable=item.why_valuable)
        elif unit_id and unit_id in units_by_id:
            r = (item.role or "").strip().lower().replace(" ", "_")
            if not r:
                role = units_by_id[unit_id].role       # no role proposed → the unit's own
            elif r in known_roles:
                role = r                               # known universal/domain role
            else:
                role = coerce_role(r)                  # unknown → coerced universal role
            prop = PlanProposal("unit", unit_id, role,
                                purpose=item.purpose, why_valuable=item.why_valuable)
        else:
            continue                           # unknown / missing ids → dropped
        if any(k.kind == prop.kind and k.ref_id == prop.ref_id for k in kept):
            continue                           # exact duplicate
        ids = _claims(prop)
        overlap = [claimed[i] for i in ids if i in claimed]
        if overlap:
            if prop.kind == "arc" and all(kept[i].kind == "unit" for i in set(overlap)):
                # the arc supersedes unit proposals it contains (highest-value class)
                for i in sorted(set(overlap), reverse=True):
                    kept.pop(i)
                claimed = {uid: idx for idx, k in enumerate(kept) for uid in _claims(k)}
            else:
                continue                       # overlapping proposal → deduped
        kept.append(prop)
        for uid in ids:
            claimed[uid] = len(kept) - 1
        if len(kept) >= cap:
            break
    for rank, p in enumerate(kept):
        p.rank = rank
    return kept


def propose_plan(structure: Structure, units: list[Unit], arcs: list[ArcCandidate],
                 adapter, topic: str, settings: dict) -> list[PlanProposal]:
    """One authoring-model plan call + deterministic validation. Raises on LLM failure —
    the caller (candidates.select_anchors_planned) owns the plan-fallback policy."""
    from ...llm import llm_json
    cap = int(settings.get("max_anchors", config.MAX_ANCHORS))
    user = render_inventory(structure, units, arcs, adapter, topic, cap)
    res = llm_json(PLAN_SYSTEM, user, ExtractionPlanLLM, temperature=0.0)
    units_by_id = {u.unit_id: u for u in units}
    arcs_by_id = {a.arc_id: a for a in arcs}
    return validate_plan(res.extractions, units_by_id, arcs_by_id, adapter, cap)
