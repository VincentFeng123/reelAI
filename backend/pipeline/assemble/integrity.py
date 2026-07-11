"""Clip-integrity helpers (audit pkg 1): keep unit_ids truthful at every span change,
merge Part-B specs without losing metadata, and record WHY candidates are dropped.

Pure module — no LLM, no I/O. `refine.py`'s merge/dedupe stays untouched (the legacy
fast path shares it); Part B routes overlaps through merge_partb instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Rejection:
    """One dropped candidate + the stage and reason — the assemble run's drop ledger."""
    cand_id: str
    title: str
    role: str
    stage: str            # build | repair | snap | dedupe | post_merge_judge | post_snap_judge | quality_floor | max_clips
    reason: str
    score: Optional[float] = None            # last judge score, when one exists
    failure_kinds: list[str] = field(default_factory=list)
    final_quality: Optional[float] = None
    start: float = 0.0
    end: float = 0.0
    # judge-integrity gate (additive, safe defaults — existing constructors keep working):
    verified_kinds: tuple[str, ...] = ()     # kinds whose evidence_quote passed containment
    unverified_kinds: tuple[str, ...] = ()   # kinds that failed quote verification (phantoms)
    kill_confirmed: bool = False             # fresh-context confirmation upheld ≥1 verified kind


def true_contents(unit_ids, referential, units, i_start: int, i_end: int):
    """(unit_ids ∪ every unit whose sentence_range lies inside [i_start, i_end], time-ordered;
    referential minus entries now inside the span). Keeps unit_ids the TRUE clip contents so
    contract checks, grounding, repair targeting, and cards see what the viewer sees."""
    have = set(unit_ids)
    for u in units or []:
        s0, s1 = u.sentence_range
        if s0 >= i_start and s1 <= i_end:
            have.add(u.unit_id)
    by_id = {u.unit_id: u for u in units or []}
    ordered = sorted(have, key=lambda uid: (by_id[uid].start if uid in by_id else 0.0, uid))
    new_ref = [(uid, rel) for uid, rel in referential if uid not in have]
    return ordered, new_ref


def merge_partb(a: dict, b: dict, units, sentences) -> dict:
    """Union two overlapping same-facet Part-B specs WITHOUT losing metadata: span/ids/
    referential/warnings union, flags OR'd, arc provenance inherited (W25-D: a non-arc
    winner used to take ALL non-span keys and silently strip the loser's arc_id, so
    eval's n_arc_clips undercounted), other keys from the higher-final_quality side.
    Marks the result merged — the caller MUST re-judge it (its text was never judged)."""
    a_q = a.get("final_quality", a.get("score", 0.0))
    b_q = b.get("final_quality", b.get("score", 0.0))
    winner, loser = (a, b) if a_q >= b_q else (b, a)
    s0 = min(a["sentence_start_idx"], b["sentence_start_idx"])
    s1 = max(a["sentence_end_idx"], b["sentence_end_idx"])
    ids = list(dict.fromkeys(list(a.get("unit_ids", [])) + list(b.get("unit_ids", []))))
    ref = list(dict.fromkeys(list(map(tuple, a.get("referential", []))) +
                             list(map(tuple, b.get("referential", [])))))
    ids, ref = true_contents(ids, ref, units, s0, s1) if units else (ids, [(u, r) for u, r in ref if u not in set(ids)])
    # W25-D arc provenance union: every arc either side ever carried (arc_id + prior
    # merges' arc_ids), first-seen order, empties dropped — multi-merge chains stay auditable.
    arc_prov = [x for x in dict.fromkeys((a.get("arc_id"), *(a.get("arc_ids") or ()),
                                          b.get("arc_id"), *(b.get("arc_ids") or ()))) if x]
    out = {
        **winner,
        "start": min(a["start"], b["start"]),
        "end": max(a["end"], b["end"]),
        "cut_end": max(a.get("cut_end", a["end"]), b.get("cut_end", b["end"])),
        "sentence_start_idx": s0,
        "sentence_end_idx": s1,
        "unit_ids": ids,
        "referential": ref,
        "warnings": tuple(set(a.get("warnings") or ()) | set(b.get("warnings") or ()) | {"merged_overlap"}),
        "judge_error": bool(a.get("judge_error")) or bool(b.get("judge_error")),
        "truncated": bool(a.get("truncated")) or bool(b.get("truncated")),
        "ship_flagged": bool(a.get("ship_flagged")) or bool(b.get("ship_flagged")),
        "merged": True,
    }
    if arc_prov:
        # winner's own arc_id stays canonical; a winner without one INHERITS the loser's.
        out["arc_id"] = winner.get("arc_id") or loser.get("arc_id") or arc_prov[0]
        out["arc_ids"] = arc_prov
    return out
