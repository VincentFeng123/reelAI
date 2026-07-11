"""Per-clip quality scoring (spec §12)."""
from __future__ import annotations

from ..understand.models import Unit
from .types import Candidate


def completeness_score(verdict, role: str, adapter) -> float:
    if getattr(verdict, "error", False):
        return 0.5                # unjudged (ship-but-flag): below judged-good, above judged-bad
    req = adapter.required_verdict_fields(role)
    if not req:
        return 1.0
    passed = sum(1 for f in req if bool(getattr(verdict, f, True)))
    base = passed / len(req)
    penalty = 0.05 * len(getattr(verdict, "failure_reasons", []) or [])
    return max(0.0, min(1.0, base - penalty))


def grounding_score(cand: Candidate, units_by_id: dict[str, Unit]) -> float:
    units = [units_by_id[u] for u in cand.unit_ids if u in units_by_id]
    if not units:
        return 0.0
    conf = sum(u.source_confidence for u in units) / len(units)
    grounded = 1.0 if (cand.verdict and getattr(cand.verdict, "source_grounded", False)) else 0.6
    return max(0.0, min(1.0, conf * grounded))


def boundary_score(warnings, whisper_moved: bool = False) -> float:
    w = set(warnings or ())
    penalty = (0.25 * ("no_period_terminated_end" in w)
               + 0.15 * ("min_duration_unreachable" in w)
               + 0.10 * ("capped_max_duration" in w)
               + 0.10 * ("no_period_after_used_last_period" in w)
               + 0.15 * ("unjudged" in w)
               + 0.15 * ("unverified_judge_concerns" in w)
               + 0.10 * ("extended_for_min_duration" in w)
               + 0.10 * ("trimmed_start" in w)
               + 0.10 * ("missing_context_card" in w)
               # VID2 edge probe (ADVISORY): a small dock only — these warnings are added
               # post-ship by the edge probe (default OFF), so with it off no spec carries
               # them and boundary_score is byte-identical.
               + 0.05 * ("starts_mid_sentence_audio" in w)
               + 0.05 * ("ends_mid_sentence_audio" in w))
    return max(0.0, min(1.0, 1.0 - penalty + 0.10 * bool(whisper_moved)))


def quality(completeness: float, grounding: float, boundary: float, priority: float) -> float:
    return round(0.40 * completeness + 0.30 * grounding + 0.20 * boundary + 0.10 * priority, 4)


def final_quality(cand: Candidate) -> float:
    return quality(cand.completeness_score, cand.grounding_score, cand.boundary_score, cand.priority)


def drop_weak(specs: list[dict], floor: float) -> list[dict]:
    return [s for s in specs if s.get("final_quality", 0.0) >= floor]
