"""Anchor selection + candidate construction (spec §6, Wave 2 P3).

Two selectors (config.ANCHOR_SELECTOR / settings["anchor_selector"]):

- "priority" — the legacy path (the eval A/B lever): anchors are units whose role is
  clip-worthy (per the adapter) AND relevant to the search topic, ranked by priority
  under the per-video anchor budget. Since Wave 2.5 (Q1b/Q1c) the flat stable sort —
  which resolved large priority ties by list order = video order, front-loading every
  anchor into the video's opening minutes — is a greedy pass that spreads picks across
  content-map nodes and caps any role at PLAN_ROLE_CAP_PER_TOPIC per content-map node
  (the same protection the plan path already had).
- "plan" (default) — the model proposes WHAT to extract per video (assemble/plan.py) from
  the actual inventory, with the adapter as a PRIOR; ``enforce_plan`` then does what LLMs
  are bad at deterministically: topic coverage, saturation quotas, arc retention, and the
  hard cap. Detected arcs (understand/arcs.py) BYPASS adapter.is_anchor_role and enter as
  synthetic anchors carrying the arc's terminal role. Plan failure or empty/garbage output
  falls back to the priority sort ("plan-fallback": stderr-logged + degraded note).

Topic relevance is scored by a single LLM pass over unit summaries (no embedding model
needed). Each anchor's context closure is turned into a sentence-anchored candidate using
the units' own sentence ranges (exact — units were built from sentence spans); arc anchors
inline their whole arc hull when it fits the ship cap.
"""
from __future__ import annotations

import math
import sys
from collections import Counter
from dataclasses import dataclass, replace
from typing import Callable, Optional

from pydantic import BaseModel, Field

from ... import config
from ..sentences import Sentence
from ..understand.arcs import ArcCandidate, detect_arcs, verify_arcs
from ..understand.models import Structure, Unit
from .closure import ClosureBudget, compute_closure
from .integrity import true_contents
from .types import Candidate

ProgressCb = Optional[Callable[[float, str], None]]


class RelItem(BaseModel):
    unit_id: str
    score: float = 0.0


class RelevanceLLM(BaseModel):
    items: list[RelItem] = Field(default_factory=list)


_REL_SYSTEM = (
    "You score how relevant each unit of a video is to a viewer's TOPIC. For each unit id, "
    "return a score from 0.0 (unrelated) to 1.0 (squarely about the topic). Judge by the unit's "
    "summary and concepts. Output only the structured result."
)


def score_topic_relevance(units: list[Unit], topic: str, settings: dict,
                          progress: ProgressCb = None) -> tuple[dict[str, float], bool]:
    """(unit_id → 0..1 relevance, degraded). degraded=True means the relevance LLM failed
    (after one retry per batch) and scores are neutral defaults — the topic filter is
    effectively OFF and callers must surface that instead of pretending scores exist."""
    if not topic or not topic.strip():
        return {u.unit_id: 1.0 for u in units}, False
    from ...llm import llm_json
    rel: dict[str, float] = {}
    degraded = False
    B = 120
    batches = [units[i:i + B] for i in range(0, len(units), B)] or [[]]
    for bi, batch in enumerate(batches):
        rows = "\n".join(
            f"{u.unit_id}: {u.summary[:120]}"
            + (f" | concepts: {', '.join((u.concepts_introduced + u.concepts_required)[:6])}"
               if (u.concepts_introduced or u.concepts_required) else "")
            for u in batch
        )
        user = f"TOPIC: {topic}\n\nUNITS:\n{rows}\n\nScore each unit id 0.0–1.0 for relevance to the topic."
        got: dict[str, float] = {}
        for attempt in range(2):                            # one retry on transient failure
            try:
                res = llm_json(_REL_SYSTEM, user, RelevanceLLM, temperature=0.0)
                got = {it.unit_id: max(0.0, min(1.0, float(it.score))) for it in res.items}
                break
            except Exception:
                if attempt == 1:
                    degraded = True                          # batch stays neutral — flag it
        for u in batch:
            rel[u.unit_id] = got.get(u.unit_id, 0.5)   # neutral if the model omitted it
        if progress:
            progress((bi + 1) / len(batches), "Scoring topic relevance")
    return rel, degraded


def compute_anchor_budget(units: list[Unit], content_map, detection, settings: dict, *,
                          adapter=None, relevance: Optional[dict[str, float]] = None) -> int:
    """Q1a content-scaled per-video anchor budget.

    An EXPLICIT settings["max_anchors"] (a number — user dial / test pin) wins outright.
    Otherwise the budget scales with how much anchor-eligible material the video actually
    has: clamp(ceil(n_anchor_eligible / 4) + 4·[detection.density == "high"],
    [MAX_ANCHORS, MAX_ANCHORS_CEIL]). Eligibility is the legacy rule (_anchor_eligible:
    anchor role + priority floor + relevance floor); without an adapter every unit counts
    (conservative overcount — the clamp still bounds it). ``content_map`` is accepted for
    future topic-count-aware scaling (enforce_plan's quota arithmetic already wants
    ~2·n_topics slots) but is not part of the current formula.
    Acceptance arithmetic: 75 eligible + density high → min(ceil(75/4)+4, 32) = 23; a
    small video stays at the floor 12."""
    explicit = settings.get("max_anchors")
    if explicit is not None:
        return max(1, int(explicit))
    floor_cap = int(config.MAX_ANCHORS)
    ceil_cap = max(floor_cap, int(config.MAX_ANCHORS_CEIL))
    rel = relevance if relevance is not None else {u.unit_id: 1.0 for u in units}
    rel_floor = float(settings.get("anchor_rel_floor", config.ANCHOR_REL_FLOOR))
    if adapter is not None:
        n_eligible = sum(1 for u in units if _anchor_eligible(u, rel, adapter, rel_floor))
    else:
        n_eligible = len(units)
    scaled = math.ceil(n_eligible / 4)
    if str(getattr(detection, "density", "") or "").lower() == "high":
        scaled += 4
    return max(floor_cap, min(scaled, ceil_cap))


def topic_matches_subject(topic: str, structure: Structure) -> bool:
    """Q1d relevance bypass predicate: True when the search query IS the video's own
    subject — asking a kinematics lecture for "kinematics" should select everything
    important, not run an LLM relevance pass whose noise re-ranks the whole video.
    Conservative by construction: case-insensitive verbatim containment (query ≥ 4 chars)
    or rapidfuzz token_set_ratio ≥ 85 against the detection rationale, the video title,
    or the content-map root title. Never raises (a bad structure just returns False)."""
    q = " ".join((topic or "").lower().split())
    if len(q) < 4:
        return False
    from rapidfuzz import fuzz
    det = getattr(structure, "detection", None)
    cm = getattr(structure, "content_map", None)
    root_title = ""
    if cm is not None:
        root = next((n for n in cm.nodes if n.node_id == cm.root_id), None)
        root_title = getattr(root, "title", "") if root is not None else ""
    subjects = (getattr(det, "rationale", "") if det is not None else "",
                getattr(structure, "title", ""), root_title)
    for s in subjects:
        s_norm = " ".join((s or "").lower().split())
        if not s_norm:
            continue
        if q in s_norm:
            return True                            # verbatim containment
        if fuzz.token_set_ratio(q, s_norm) >= 85:  # word-order/inflection tolerant
            return True
    return False


def select_anchors(units: list[Unit], relevance: dict[str, float], adapter, settings: dict) -> list[Unit]:
    """The legacy 'priority' selector, upgraded in place (Q1b/Q1c — same eligibility, same
    scoring formula, no longer a flat stable sort):

    - Q1b topic-spread tie-break: a greedy pass picks, at each step, the highest-SCORED
      remaining unit; ties resolve toward the content-map node that has contributed the
      FEWEST already-selected anchors (unknown node "" is never penalized), then time,
      then unit_id. Score stays PRIMARY — node spread only breaks score ties — so a
      strong anchor is never dropped for a weaker one from an unseen node, while the
      stable-sort artifact (a 27-way priority tie resolved by list order putting every
      anchor in the video's opening minutes) still round-robins across nodes.
    - Q1c/W25-C per-topic role cap: no role holds more than PLAN_ROLE_CAP_PER_TOPIC slots
      WITHIN one content-map node (keyed (role, node_id) — the same saturation protection
      enforce_plan gives the plan path). The old video-global PLAN_ROLE_CAP=4 starved
      claim-dense zones: 46 eligible qP claims competed for 4 video-wide slots. Unmapped
      units (node_id "") share one pseudo-topic bucket.
    Fully deterministic; the cap is the per-video anchor budget (settings["max_anchors"],
    injected by assemble_clips; None falls back to the MAX_ANCHORS floor)."""
    floor = float(settings.get("anchor_rel_floor", config.ANCHOR_REL_FLOOR))
    cap = int(settings.get("max_anchors") or config.MAX_ANCHORS)
    scored = [(_legacy_anchor_score(u, relevance, adapter), u)
              for u in units if _anchor_eligible(u, relevance, adapter, floor)]
    # deterministic base order: score desc, then time, then unit id
    scored.sort(key=lambda t: (-t[0], t[1].start, t[1].unit_id))
    picked: list[Unit] = []
    node_count: Counter = Counter()
    role_count: Counter = Counter()                # keyed (role, node_id) — W25-C
    pool = list(scored)
    while pool and len(picked) < cap:
        pool = [t for t in pool
                if role_count[(t[1].role, t[1].node_id)] < config.PLAN_ROLE_CAP_PER_TOPIC]
        if not pool:
            break
        i = min(range(len(pool)),
                key=lambda j: (-pool[j][0],
                               (node_count[pool[j][1].node_id] if pool[j][1].node_id else 0),
                               pool[j][1].start, pool[j][1].unit_id))
        _score, u = pool.pop(i)
        picked.append(u)
        if u.node_id:
            node_count[u.node_id] += 1
        role_count[(u.role, u.node_id)] += 1
    return picked


# ── Wave 2 P3: plan-driven selection + deterministic enforcement ─────────────────────────

@dataclass
class _Sel:
    """One selected anchor entry during enforcement (anchor unit + optional arc + rank)."""
    anchor: Unit
    arc: Optional[ArcCandidate]
    kind: str                                  # "plan" | "arc" | "coverage"
    rank: int


def _legacy_anchor_score(u: Unit, relevance: dict[str, float], adapter) -> float:
    """EXACTLY the legacy select_anchors scoring formula (shared so coverage additions are
    'best prior-scored' in the same sense the priority path ranks anchors)."""
    rel = relevance.get(u.unit_id, 0.0)
    return adapter.anchor_priority(u.role) * (0.4 + 0.6 * rel) * (0.5 + 0.5 * u.source_confidence)


def _anchor_eligible(u: Unit, relevance: dict[str, float], adapter, floor: float) -> bool:
    """The legacy eligibility gate (anchor role + priority floor + relevance floor)."""
    return (adapter.is_anchor_role(u.role)
            and adapter.anchor_priority(u.role) * 100.0 >= config.ANCHOR_MIN_PRIORITY
            and relevance.get(u.unit_id, 0.0) >= floor)


def _unit_topic_id(u: Unit, topic_nodes) -> str:
    """The content-map topic node a unit belongs to ('' when the map has no topic level)."""
    ids = {n.node_id for n in topic_nodes}
    if u.node_id in ids:
        return u.node_id
    for n in topic_nodes:
        if n.sentence_range[0] <= u.sentence_range[0] <= n.sentence_range[1]:
            return n.node_id
    return ""


def _node_covered(node, members: list[Unit]) -> bool:
    """W25-C real-coverage predicate: True when ``members`` (the selected entries' units
    attributed to this topic node) span ≥ config.MIN_NODE_COVERAGE of the node's timed
    span (interval UNION clipped to the node — overlapping units never double-count).
    A node without timing (end <= start, older maps) keeps the legacy any-anchor rule:
    covered iff any member exists (the fraction is uncomputable, not zero).

    HONEST SCOPE (confirmed by a real-structure replay at review): ``members`` are the
    entries' ANCHOR units (arc members for arcs), NOT their built closures — closures don't
    exist yet at selection time. On fine-grained rebuilt structures anchors are 3-15s while
    topic nodes run 50-300s, so a single anchor rarely reaches 0.5 and the floor reads
    'under-covered' and tops the node up with its best prior-scored anchor. That is the
    intended coverage-beats-saturation direction (it is exactly what adds a c0.t1 CLAIM
    anchor for qP item 4 alongside the definition clip), bounded by the MAX_ANCHORS cap; the
    downstream eval metric ``topic_span_coverage`` measures the SHIPPED spans instead, so the
    two are deliberately different bases (selection prior vs shipped truth)."""
    if not members:
        return False
    lo, hi = float(node.start), float(node.end)
    if hi <= lo:
        return True                            # untimed node → legacy any-anchor rule
    ivs = sorted((max(lo, float(u.start)), min(hi, float(u.end)))
                 for u in members if float(u.end) > lo and float(u.start) < hi)
    covered, cur_lo, cur_hi = 0.0, None, None
    for s, e in ivs:
        if cur_hi is None or s > cur_hi:       # disjoint — flush the running interval
            covered += (cur_hi - cur_lo) if cur_hi is not None else 0.0
            cur_lo, cur_hi = s, e
        else:
            cur_hi = max(cur_hi, e)
    if cur_hi is not None:
        covered += cur_hi - cur_lo
    return covered / (hi - lo) >= config.MIN_NODE_COVERAGE


def _arc_anchor_unit(arc: ArcCandidate, units_by_id: dict[str, Unit]) -> Optional[Unit]:
    """The synthetic anchor a detected arc enters selection as: the arc's TERMINAL unit,
    re-roled to the arc's terminal role (result/solution — the worked-example contracts)
    and widened to the arc hull, bypassing adapter.is_anchor_role by construction."""
    members = [units_by_id[uid] for uid in arc.unit_ids if uid in units_by_id]
    terminal = units_by_id.get(arc.terminal_id)
    if not members or terminal is None:
        return None
    opener = units_by_id.get(arc.opener_ids[0]) if arc.opener_ids else None
    return terminal.model_copy(update={
        "unit_id": arc.arc_id,
        "role": arc.terminal_role,
        "start": min(u.start for u in members),
        "end": max(u.end for u in members),
        "sentence_range": (min(u.sentence_range[0] for u in members),
                           max(u.sentence_range[1] for u in members)),
        "topic": (opener.topic if opener and opener.topic else terminal.topic),
        "summary": (opener.summary if opener and opener.summary else terminal.summary),
    })


def enforce_plan(proposals, arcs: list[ArcCandidate], structure: Structure, units: list[Unit],
                 relevance: dict[str, float], adapter, settings: dict) -> tuple[list[Unit], dict[str, ArcCandidate]]:
    """Deterministic enforcement over the validated plan (P3c) — the plan is the primary
    selector; this pass guarantees what the LLM can't: (i) topic coverage — every content-
    map topic node with ≥1 anchor-eligible unit contributes ≥1 candidate, and (W25-C) a
    topic only counts COVERED when the selected entries' units span ≥ MIN_NODE_COVERAGE of
    the node's timed span (one 3.3s anchor no longer covers an 81s topic — the items-3/4
    sliver class; untimed nodes keep the legacy any-anchor rule); an uncovered topic gets
    its best prior-scored anchor; (ii) saturation — no topic contributes more than
    ceil(cap/n_topics)+1 anchors, no role more than PLAN_ROLE_CAP_PER_TOPIC anchors WITHIN
    its home topic (W25-C: keyed (role, home_topic), replacing the video-global
    PLAN_ROLE_CAP=4 that starved claim-dense zones — 46 eligible qP claims for 4 slots;
    lowest plan-ranked overflow drops); (iii) detected arcs are never dropped by saturation
    (highest-value extraction class; arcs the plan skipped are appended); (iv) the
    MAX_ANCHORS hard cap is preserved. Precedence when (i) and (ii) conflict: COVERAGE
    WINS — an under-covered topic's best anchor is added even when its role is already at
    quota (deliberate: an uncovered topic is a worse failure than one more anchor of a
    role; only the hard cap can still drop it). Returns (anchor units, arc_id →
    ArcCandidate for the synthetic anchors)."""
    cap = int(settings.get("max_anchors") or config.MAX_ANCHORS)   # None → the budget floor
    floor = float(settings.get("anchor_rel_floor", config.ANCHOR_REL_FLOOR))
    units_by_id = {u.unit_id: u for u in units}
    arcs_by_id = {a.arc_id: a for a in arcs}
    topic_nodes = structure.content_map.topics()

    entries: list[_Sel] = []
    planned_arcs: set[str] = set()
    for p in proposals:
        if p.kind == "arc":
            arc = arcs_by_id.get(p.ref_id)
            anchor = _arc_anchor_unit(arc, units_by_id) if arc else None
            if anchor is None:
                continue
            entries.append(_Sel(anchor, arc, "plan", p.rank))
            planned_arcs.add(arc.arc_id)
        else:
            u = units_by_id.get(p.ref_id)
            if u is None:
                continue
            anchor = u if (not p.role or p.role == u.role) else u.model_copy(update={"role": p.role})
            entries.append(_Sel(anchor, None, "plan", p.rank))
    # (iii) arcs the plan skipped are appended — they are the highest-value extraction class
    for arc in arcs:
        if arc.arc_id in planned_arcs:
            continue
        anchor = _arc_anchor_unit(arc, units_by_id)
        if anchor is not None:
            entries.append(_Sel(anchor, arc, "arc", len(entries)))

    def _entry_units(e: _Sel) -> list[Unit]:
        """The real units an entry contributes to coverage arithmetic: the arc's members
        for arc entries (the synthetic anchor's hull-wide span would overstate coverage
        across gaps), else the anchor unit itself."""
        if e.arc is not None:
            return [units_by_id[uid] for uid in e.arc.unit_ids if uid in units_by_id]
        return [e.anchor]

    def _home_topic(e: _Sel) -> str:
        if e.arc is not None:
            term = units_by_id.get(e.arc.terminal_id)
            return _unit_topic_id(term, topic_nodes) if term is not None else ""
        return _unit_topic_id(e.anchor, topic_nodes)

    # (ii) saturation in plan-rank order — arcs are exempt (iii) ---------------------------
    topic_quota = (math.ceil(cap / len(topic_nodes)) + 1) if topic_nodes else None
    role_quota = config.PLAN_ROLE_CAP_PER_TOPIC
    kept: list[_Sel] = []
    topic_count: Counter = Counter()
    role_count: Counter = Counter()            # keyed (role, home_topic) — W25-C
    for e in entries:
        home = _home_topic(e)
        if e.arc is None:                      # arcs are never dropped by saturation
            if topic_quota is not None and home and topic_count[home] >= topic_quota:
                continue
            if role_count[(e.anchor.role, home)] >= role_quota:
                continue
        if home:
            topic_count[home] += 1
        role_count[(e.anchor.role, home)] += 1
        kept.append(e)

    # (i) topic coverage floor — REAL coverage (W25-C): an under-covered topic (selected
    # units spanning < MIN_NODE_COVERAGE of its timed span, sliver class) contributes its
    # best prior-scored anchor, same as a fully skipped topic ------------------------------
    node_members: dict[str, list[Unit]] = {}
    for e in kept:
        for u in _entry_units(e):
            t = _unit_topic_id(u, topic_nodes)
            if t:
                node_members.setdefault(t, []).append(u)
    selected_ids = {e.anchor.unit_id for e in kept} \
        | {uid for e in kept if e.arc is not None for uid in e.arc.unit_ids}
    for node in topic_nodes:
        if _node_covered(node, node_members.get(node.node_id, [])):
            continue
        eligible = [u for u in units
                    if u.unit_id not in selected_ids
                    and _unit_topic_id(u, topic_nodes) == node.node_id
                    and _anchor_eligible(u, relevance, adapter, floor)]
        if not eligible:
            continue                           # nothing anchor-eligible → no floor for it
        best = max(eligible, key=lambda u: (_legacy_anchor_score(u, relevance, adapter), -u.start))
        kept.append(_Sel(best, None, "coverage", len(entries) + len(kept)))
        selected_ids.add(best.unit_id)

    # (iv) hard cap — drop lowest-ranked plan extras first, then coverage, arcs last -------
    for droppable in (("plan",), ("plan", "coverage"), ("plan", "coverage", "arc")):
        while len(kept) > cap:
            idx = next((i for i in range(len(kept) - 1, -1, -1)
                        if kept[i].kind in droppable and kept[i].arc is None), None)
            if idx is None and "arc" in droppable:
                idx = len(kept) - 1            # all arcs: the cap still wins
            if idx is None:
                break
            kept.pop(idx)
    kept = kept[:cap]

    anchors = [e.anchor for e in kept]
    arc_map = {e.anchor.unit_id: e.arc for e in kept if e.arc is not None}
    return anchors, arc_map


def select_anchors_planned(structure: Structure, units: list[Unit], relevance: dict[str, float],
                           adapter, settings: dict, topic: str, *,
                           stats: Optional[dict] = None) \
        -> tuple[list[Unit], dict[str, ArcCandidate], list[str]]:
    """The 'plan' selector (P3a/b/c/d): detect + optionally verify arcs, ask the model for a
    per-video extraction plan, then deterministically enforce coverage/saturation/caps.
    Plan-call failure or empty/garbage output falls back to the legacy priority sort
    ('plan-fallback') — byte-equivalent anchors, stderr-logged, degraded note appended.
    ``stats`` (optional, additive — I1 eval plumbing) is filled with machine-readable run
    signals: n_arcs_detected (pre-verify detection count), plan_engine
    ('plan' | 'plan-fallback'), and (W25-G run artifacts) 'arcs_verified' (the post-verify
    ArcCandidate survivors) + 'plan_proposals' (the validated PlanProposals; [] on
    failure/garbage) — write_run_artifacts serializes both defensively into
    runs/<ts>/{arcs,plan}.json. Returns (anchor units, arc_id → ArcCandidate for synthetic
    arc anchors, notes)."""
    from .plan import propose_plan
    if stats is None:
        stats = {}
    # normalize the budget once so every downstream reader (propose_plan's cap,
    # enforce_plan, the fallback select_anchors) sees a concrete number, never None.
    settings = {**settings, "max_anchors": int(settings.get("max_anchors") or config.MAX_ANCHORS)}
    notes: list[str] = []
    units_by_id = {u.unit_id: u for u in units}
    arcs = detect_arcs(units)
    stats["n_arcs_detected"] = len(arcs)       # detection count BEFORE the verify pass
    arcs, arc_note = verify_arcs(arcs, units_by_id, settings)
    stats["arcs_verified"] = list(arcs)        # W25-G artifact plumbing (post-verify survivors)
    if arc_note:
        notes.append(arc_note)
    proposals = None
    try:
        proposals = propose_plan(structure, units, arcs, adapter, topic, settings)
    except Exception as e:
        print(f"[plan] extraction-plan call failed ({e!r}); falling back to priority selection",
              file=sys.stderr)
    stats["plan_proposals"] = list(proposals or [])   # W25-G artifact ([] on failure/garbage)
    if not proposals:                          # failure OR empty/garbage output → fallback
        if proposals is not None:
            print("[plan] extraction plan empty after validation; "
                  "falling back to priority selection", file=sys.stderr)
        notes.append("extraction plan degraded — legacy priority selection used (plan-fallback)")
        stats["plan_engine"] = "plan-fallback"
        return select_anchors(units, relevance, adapter, settings), {}, notes
    anchors, arc_map = enforce_plan(proposals, arcs, structure, units, relevance, adapter, settings)
    if not anchors:
        notes.append("extraction plan degraded — legacy priority selection used (plan-fallback)")
        stats["plan_engine"] = "plan-fallback"
        return select_anchors(units, relevance, adapter, settings), {}, notes
    stats["plan_engine"] = "plan"
    return anchors, arc_map, notes


def _clamped_range(i_start: int, i_end: int, n_sentences: int) -> tuple[int, int, tuple[str, ...]]:
    """Clamp a unit-derived sentence range into the live sentence list — LOUDLY (W25-A).
    A range past the live list means the structure was built on a DIFFERENT sentence
    universe (the cross-indexer cache-poisoning class: qP's cache indexed to 321 against
    the app's 183); silently snapping to the last sentence shipped clips of unrelated
    tail text. Freshness gating should prevent this upstream, so any overshoot beyond an
    off-by-one tail (exclusive-end convention) returns a candidate warning the caller
    puts on the spec and counts into stats['index_clamp_events']."""
    raw_end = max(i_start, i_end)
    lo = max(0, min(i_start, n_sentences - 1))
    hi = max(lo, min(i_end, n_sentences - 1))
    overshoot = raw_end - (n_sentences - 1)
    if overshoot > 1:
        return lo, hi, (f"sentence_index_clamped:+{overshoot}",)
    return lo, hi, ()


def build_arc_candidate(arc: ArcCandidate, graph, adapter, units: list[Unit],
                        units_by_id: dict[str, Unit], sentences: list[Sentence],
                        relevance: dict[str, float], settings: dict) -> Optional[Candidate]:
    """Turn a detected arc into a candidate. The arc IS the known-complete extraction, so
    its whole hull is inlined when it fits the ship cap (no closure guesswork). A hull
    that cannot ship (cross-distance practice pair, oversized example) degrades to the
    normal closure build anchored on the terminal unit re-roled to the arc's terminal
    role — near context inlines, far context goes referential (card / watch-first)."""
    # The arc hull is a known-complete extraction; it inlines whole up to the HARD ship cap
    # (max_clip_duration_s). closure_max_span_s is the SOFT context budget and plays no role
    # here — clamping to it (the old min(...)) would wrongly shrink arcs once soft < hard.
    ship_cap = float(settings.get("max_clip_duration_s", config.DEFAULTS["max_clip_duration_s"]))
    members = [units_by_id[uid] for uid in arc.unit_ids if uid in units_by_id]
    terminal = units_by_id.get(arc.terminal_id)
    if not members or terminal is None:
        return None
    i_start = min(u.sentence_range[0] for u in members)
    i_end = max(u.sentence_range[1] for u in members)
    i_start, i_end, clamp_warn = _clamped_range(i_start, i_end, len(sentences))
    hull_s = float(sentences[i_end].end) - float(sentences[i_start].start)
    opener = units_by_id.get(arc.opener_ids[0]) if arc.opener_ids else None
    if hull_s > ship_cap:
        synth = (terminal if terminal.role == arc.terminal_role
                 else terminal.model_copy(update={"role": arc.terminal_role}))
        cand = build_candidate(synth, graph, adapter, units, units_by_id, sentences,
                               relevance, settings)
        if cand is None:
            return None
        # Overflow (locked decision): keep the opener IN the span, never SILENTLY drop it.
        # Onset within the HARD ship cap → re-inline (extend the unit set + i_start/start back
        # to the opener, and strip it from referential so it is not both inlined AND carded —
        # Finding 2). Onset beyond the ship cap → surface it as a watch-first card (referential)
        # rather than dropping it, preserving the old "opener always surfaces" guarantee
        # (Finding 3). compute_closure may already have carded it (gap / max_extra_units demote),
        # in which case we leave that card in place.
        if opener is not None and opener.unit_id not in cand.unit_ids:
            new_start = min(cand.start, float(sentences[opener.sentence_range[0]].start))
            new_dur = float(sentences[cand.i_end].end) - new_start
            if new_dur <= ship_cap:
                new_ids = [opener.unit_id] + list(cand.unit_ids)
                i_start = min(cand.i_start, opener.sentence_range[0])
                cand = replace(cand, unit_ids=new_ids, i_start=i_start, start=new_start,
                               referential=[(uid, r) for uid, r in cand.referential
                                            if uid != opener.unit_id])
            elif opener.unit_id not in {uid for uid, _ in cand.referential}:
                rel = "answers" if arc.arc_role == "practice_pair" else "prerequisite"
                cand = replace(cand, referential=list(cand.referential)
                               + [(opener.unit_id, rel)], truncated=True)
        return replace(cand, cand_id=f"c_{arc.arc_id}", arc_id=arc.arc_id,
                       warnings=tuple(set(cand.warnings or ()) | set(clamp_warn)))
    unit_ids, referential = true_contents([m.unit_id for m in members], [], units, i_start, i_end)
    return Candidate(
        cand_id=f"c_{arc.arc_id}",
        arc_id=arc.arc_id,
        anchor_id=arc.terminal_id,
        role=arc.terminal_role,
        facet=adapter.facet_for(arc.terminal_role),
        title=(opener.topic if opener and opener.topic else terminal.topic)
              or (opener.summary[:60] if opener and opener.summary else terminal.summary[:60]),
        reason=(opener.summary if opener and opener.summary else terminal.summary),
        unit_ids=unit_ids,
        referential=referential,
        i_start=i_start, i_end=i_end,
        start=float(sentences[i_start].start), end=float(sentences[i_end].end),
        relevance=max((relevance.get(m.unit_id, 0.0) for m in members), default=0.0),
        priority=adapter.anchor_priority(arc.terminal_role),
        truncated=False,
        warnings=clamp_warn,
    )


def build_candidate(anchor: Unit, graph, adapter, units: list[Unit], units_by_id: dict[str, Unit],
                    sentences: list[Sentence], relevance: dict[str, float], settings: dict) -> Optional[Candidate]:
    # Soft/hard split. max_span_s is the SOFT closure budget for NON-onset context; it stays
    # clamped to the ship cap so the judge never scores a span the cutter can't ship (the
    # test_budget_clamp invariant). hard_max_span_s is the ship cap and gates the onset
    # force-inline. The overflow window (max_span_s, hard_max_span_s] is now LIVE because the
    # SOFT budget default was lowered below the ship cap (closure_max_span_s=120 < 240) — the
    # clamp is a no-op there (min(120,240)=120), so an onset overflows 120 up to 240.
    _ship_cap = float(settings.get("max_clip_duration_s", config.DEFAULTS["max_clip_duration_s"]))
    budget = ClosureBudget(
        max_span_s=min(float(settings.get("closure_max_span_s", config.CLOSURE_MAX_SPAN_S)),
                       _ship_cap),
        hard_max_span_s=_ship_cap)
    cl = compute_closure(anchor, graph, units_by_id, adapter, units, budget)
    inline = [units_by_id[uid] for uid in cl.unit_ids if uid in units_by_id]
    if not inline:
        return None
    i_start = min(u.sentence_range[0] for u in inline)
    i_end = max(u.sentence_range[1] for u in inline)
    i_start, i_end, clamp_warn = _clamped_range(i_start, i_end, len(sentences))
    unit_ids, referential = true_contents(list(cl.unit_ids), list(cl.referential),
                                          units, i_start, i_end)
    return Candidate(
        cand_id=f"c_{anchor.unit_id}",
        anchor_id=anchor.unit_id,
        role=anchor.role,
        facet=adapter.facet_for(anchor.role),
        title=anchor.topic or (anchor.summary[:60]),
        reason=anchor.summary,
        unit_ids=unit_ids,
        referential=referential,
        i_start=i_start, i_end=i_end,
        start=float(sentences[i_start].start), end=float(sentences[i_end].end),
        relevance=relevance.get(anchor.unit_id, 0.0),
        priority=adapter.anchor_priority(anchor.role),
        truncated=cl.truncated,
        warnings=clamp_warn,
    )
