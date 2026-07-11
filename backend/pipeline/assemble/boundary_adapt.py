"""Physical boundary refinement (spec §9) — REUSE the existing snapping code.

Validated candidates carry Part-B metadata (role, context_card, scores, unit_ids) that
``refine_and_snap`` would strip when it rebuilds dicts. So we call its internals directly
— ``boundary._snap_one`` (start→sentence-start, end→period, min/max duration, tail pad)
and ``_dedupe_partb`` (overlap removal with metadata-aware merge) — and merge the snapped
result over the metadata, preserving every field. The optional targeted-Whisper precise pass
(``boundary.refine_clip_boundaries``) is applied by the orchestrator afterwards.
"""
from __future__ import annotations

from ... import config
from ..sentences import Sentence
from .contracts import contract_coverage
from .integrity import Rejection, merge_partb, true_contents
from .types import Candidate


def candidate_to_boundary_input(cand: Candidate) -> dict:
    from .validate import _hard_core_ok
    return {
        "i_start": cand.i_start, "i_end": cand.i_end,
        "start": cand.start, "end": cand.end,
        "facet": cand.facet, "reason": cand.reason,
        "score": cand.final_quality, "rel": cand.final_quality,
        # Part-B metadata carried through the snap/dedupe (which preserve dict keys).
        # "role" stays the ANCHOR role (provenance, reported in payloads); "contract_role"
        # is the content-bound contract the verdict/scoring were computed under (P1b):
        "role": cand.role, "title": cand.title, "cand_id": cand.cand_id, "anchor_id": cand.anchor_id,
        "contract_role": cand.contract_role or cand.role,
        "unit_ids": list(cand.unit_ids), "referential": list(cand.referential),
        "context_card": cand.context_card,
        "completeness_score": cand.completeness_score, "grounding_score": cand.grounding_score,
        "priority": cand.priority,
        "judge_error": bool(getattr(cand.verdict, "error", False)),
        # P4a dedupe tie-break input (i): did the STORED verdict pass every hard-core gate
        # (topic + purpose + grounded + references)? Read from the verdict the repair stage
        # issued — the freshest one at dedupe time (the 4b seam runs after dedupe).
        "hard_gates_ok": _hard_core_ok(cand.verdict),
        # hash of the exact text the verdict was issued on — the post-snap re-judge seam
        # compares it against the FINAL span text to detect any post-judge mutation:
        "judged_text_hash": str(getattr(cand, "judged_text_hash", "") or ""),
        "truncated": bool(getattr(cand, "truncated", False)),
        # I1 eval plumbing (additive; payload whitelist in orchestrator is unaffected):
        # detected-arc provenance + repair trim moves for the per-video eval columns.
        "arc_id": str(getattr(cand, "arc_id", "") or ""),
        "n_trims": int(getattr(cand, "n_trims", 0) or 0),
        # judge-integrity gate (ship-flagged visibility + phantom_verdict_rate inputs):
        "warnings": tuple(getattr(cand, "warnings", ()) or ()),
        "ship_flagged": bool(getattr(cand, "ship_flagged", False)),
        "n_failure_reasons": int(getattr(cand, "n_failure_reasons", 0)),
        "n_verified": int(getattr(cand, "n_verified", 0)),
        # W25-G kind-level phantom inputs (phantom_quotable_rate filters on these):
        "verified_kinds": tuple(getattr(cand, "verified_kinds", ()) or ()),
        "unverified_kinds": tuple(getattr(cand, "unverified_kinds", ()) or ()),
    }


def snap_candidates(cands: list[Candidate], sentences: list[Sentence], settings: dict,
                    units=None, adapter=None) -> tuple[list[dict], list[Rejection]]:
    from ..refine import _snap_one
    allow_qe = bool(settings.get("allow_question_exclaim_ends", config.DEFAULTS["allow_question_exclaim_ends"]))
    min_dur = float(settings.get("min_clip_duration_s", config.DEFAULTS["min_clip_duration_s"]))
    max_dur = float(settings.get("max_clip_duration_s", config.DEFAULTS["max_clip_duration_s"]))
    tail_pad = float(settings.get("tail_pad_s", config.DEFAULTS["tail_pad_s"]))

    rejections: list[Rejection] = []
    specs: list[dict] = []
    # W25-E: the anchor's topic-node time bounds ride into _snap_one so the min-duration
    # extension can prefer the direction that stays INSIDE the anchor's node — the
    # forward-only default extended 2s prompts into the NEXT event's units (their answers),
    # manufacturing overlaps. Node span = hull of the units sharing the anchor's node_id
    # (the ContentMap isn't threaded here; the unit hull is the same partition).
    units_by_id = {u.unit_id: u for u in units} if units else {}
    node_bounds: dict[str, tuple[float, float]] = {}
    for u in units or []:
        if u.node_id:
            lo, hi = node_bounds.get(u.node_id, (u.start, u.end))
            node_bounds[u.node_id] = (min(lo, u.start), max(hi, u.end))
    for c in cands:
        b = candidate_to_boundary_input(c)
        anchor_u = units_by_id.get(c.anchor_id)
        if anchor_u is not None and anchor_u.node_id in node_bounds:
            b["node_span"] = node_bounds[anchor_u.node_id]
        snapped = _snap_one(b, sentences, allow_qe, min_dur, tail_pad, max_dur)
        if snapped is None:
            rejections.append(Rejection(cand_id=c.cand_id, title=c.title, role=c.role,
                                        stage="snap", reason="unsnappable (end<=start)",
                                        final_quality=c.final_quality, start=c.start, end=c.end))
            continue
        spec = {**b, **snapped}
        # candidate-level warnings (e.g. unverified_judge_concerns) must survive the snap
        spec["warnings"] = tuple(set(b.get("warnings") or ()) | set(snapped.get("warnings") or ()))
        specs.append(spec)
    if units is not None:
        # snap may extend the span (min-duration) — keep unit_ids/referential truthful
        for s in specs:
            s["unit_ids"], s["referential"] = true_contents(
                s.get("unit_ids", []), s.get("referential", []), units,
                s["sentence_start_idx"], s["sentence_end_idx"])
    for s in specs:
        if s.get("judge_error"):   # unjudged (judge outage): user-visible + boundary_score penalty
            s["warnings"] = tuple(set(s.get("warnings") or ()) | {"unjudged"})
        # P4a tie-break input (ii): required-element coverage of the spec's bound contract,
        # computed AFTER the true_contents refresh (so absorbed units count). Neutral (0)
        # without an adapter/contract — the tie-break then falls through to final_quality.
        s["contract_coverage"] = float(contract_coverage(
            s.get("unit_ids", []), s.get("contract_role") or s.get("role", ""),
            units_by_id, adapter))
    specs, dedupe_rejections = _dedupe_partb(specs, sentences, min_dur, units, adapter,
                                             max_dur=max_dur)
    rejections.extend(dedupe_rejections)
    specs.sort(key=lambda s: s["start"])
    return specs, rejections


def _arc_provenance(s: dict) -> frozenset:
    """Every detected-arc id the spec carries: arc_id + prior merges' arc_ids union
    (W25-D provenance), empties dropped."""
    return frozenset(x for x in (s.get("arc_id"), *(s.get("arc_ids") or ())) if x)


def _union_guard(a: dict, b: dict, units_by_id: dict, max_dur: float):
    """W25-E same-facet union guards: the reason two overlapping same-facet specs must NOT
    union, else None. Facet equality is too coarse — every worked-example-family role
    shares facet 'worked_example', so DISTINCT problems that merely touch used to union
    into one mash (qP arcs 2+4+5 → one [382,480] spec) — and unions had NO max-duration
    re-check (kinematics arc_1+arc_2 would union to 264s > the 240s cap). Guarded pairs
    fall through to the cross-facet trim/keep-both path, so nothing dies that didn't
    before; both sides' contract_coverage/hard_gates_ok are already stored at dedupe time."""
    pa, pb = _arc_provenance(a), _arc_provenance(b)
    if pa and pb and pa != pb:                       # two different detected arcs
        return "differing arc provenance"
    na = {u.node_id for u in (units_by_id.get(uid) for uid in a.get("unit_ids") or ())
          if u is not None and u.node_id}
    nb = {u.node_id for u in (units_by_id.get(uid) for uid in b.get("unit_ids") or ())
          if u is not None and u.node_id}
    if na and nb and not (na & nb):                  # no shared topic node
        return "disjoint topic nodes"
    if (a.get("hard_gates_ok") and b.get("hard_gates_ok")
            and float(a.get("contract_coverage") or 0.0) >= 1.0
            and float(b.get("contract_coverage") or 0.0) >= 1.0):
        return "both contract-complete"              # two whole clips → keep both, not a mash
    if max_dur and (max(a["end"], b["end"]) - min(a["start"], b["start"])) > max_dur:
        return "union exceeds max_clip_duration_s"
    return None


def _dedupe_partb(clips: list[dict], sentences: list[Sentence], min_dur: float,
                  units, adapter=None, max_dur: float = 0.0) -> tuple[list[dict], list[Rejection]]:
    """refine._dedupe's keep/drop structure, but metadata-aware: same-facet overlaps merge via
    integrity.merge_partb (union + merged flag, re-judged upstream) and every dropped spec
    becomes a Rejection. The keep/drop TIE-BREAK was deliberately unfrozen by P4a (Wave 2
    §16): overlap/containment losers are decided by refine._better's shared _keep_key —
    hard-core gate results, then contract coverage, then final_quality — driven by the
    ``hard_gates_ok``/``contract_coverage`` fields only Part-B specs carry (the legacy fast
    path stays on the pure score tie-break because it never sets them)."""
    from ..refine import NEAR_DUP_EPS, _better, _trim_start_after

    def _reject(loser: dict, winner: dict) -> None:
        rejections.append(Rejection(
            cand_id=loser.get("cand_id", ""), title=loser.get("title", ""),
            role=loser.get("role", ""), stage="dedupe",
            reason=f"overlap loser to {winner.get('cand_id', '?')}",
            final_quality=loser.get("final_quality", loser.get("score")), start=loser["start"], end=loser["end"]))

    rejections: list[Rejection] = []
    units_by_id = {u.unit_id: u for u in units} if units else {}
    clips = sorted(clips, key=lambda c: (c["start"], -c["end"]))
    kept: list[dict] = []
    for c in clips:
        if not kept:
            kept.append(c)
            continue
        k = kept[-1]
        if c["start"] >= k["end"]:                      # disjoint
            kept.append(c)
            continue
        overlapping_pairs = (
            (c["start"] >= k["start"] and c["end"] <= k["end"]) or
            (c["start"] <= k["start"] and c["end"] >= k["end"]) or
            (abs(c["start"] - k["start"]) <= NEAR_DUP_EPS and abs(c["end"] - k["end"]) <= NEAR_DUP_EPS)
        )
        if overlapping_pairs:                            # containment / near-dup → keep better
            winner = _better(k, c)
            _reject(c if winner is k else k, winner)
            kept[-1] = winner
            continue
        if c["facet"] == k["facet"] and _union_guard(k, c, units_by_id, max_dur) is None:
            kept[-1] = merge_partb(k, c, units, sentences)  # same facet → metadata-aware union
            # the union's coverage is a pure function of its (refreshed) unit_ids — keep it
            # truthful for any LATER overlap comparison in this sweep. hard_gates_ok stays
            # the winner's (the union's own verdict arrives at the 4b re-judge seam).
            kept[-1]["contract_coverage"] = float(contract_coverage(
                kept[-1].get("unit_ids", []),
                kept[-1].get("contract_role") or kept[-1].get("role", ""),
                units_by_id, adapter))
            continue
        # cross-facet overlap — or a GUARDED same-facet pair (distinct arcs / disjoint
        # topic nodes / both contract-complete / oversized union): trim c, else keep better.
        trimmed = _trim_start_after(c, k["end"], sentences, min_dur)
        if trimmed is not None:
            kept.append(trimmed)
        else:
            winner = _better(k, c)
            _reject(c if winner is k else k, winner)
            kept[-1] = winner
    return kept, rejections
