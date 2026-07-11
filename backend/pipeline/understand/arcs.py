"""Deterministic instructional-arc detection (Wave 2 P3a).

Scans the time-ordered units for the worked-example grammar

    (example_setup | problem_givens | practice_prompt)
        -> 1+ (worked_step | calculation | derivation)
        -> (result | solution | closer-with-steps | last-step-as-final)

tolerating up to ``MAX_ARC_INTERLEAVE`` interleaved non-arc units between members
(``NEUTRAL_ROLES`` — interpretation/explanation asides — are transparent to that budget,
W25-D), plus cross-distance practice_prompt->solution pairing by shared concept. Closers
are PROVISIONAL terminals (a later step or true result/solution supersedes them) and the
two W25-D-broadened join paths (closer acceptance, pre-step opener accumulation) are
locality-bounded by ``MAX_ARC_MEMBER_GAP_S`` — review fixes for closer-truncation and
the phantom-crawl arc both replayed on the cached qP/kinematics structures. Arcs
recover complete instructional events whose payoff units were labeled with NON-anchor
roles (the audited kinematics example: results labeled 'calculation' were unclippable;
the qP graph-build example terminated in claim/physical_interpretation) — they bypass
``adapter.is_anchor_role`` and enter anchor selection as synthetic anchors carrying the
arc's terminal role (result/solution), i.e. the worked-example contracts. Zero-step arcs
below ``MIN_ARC_SUBSTANCE_S`` are dropped at detection (micro Socratic pairs are
saturation-exempt / dedupe-protected / snap-padded downstream, so only here works).

Detection is pure (zero LLM). ``verify_arcs`` optionally spends ONE batched llm_json call
per video (MathNet pattern: the model returns only the unit ids confirming each arc's
problem/steps/answer; ids outside the arc are discarded by rule; arcs the model rejects
are dropped). LLM failure degrades to unverified-but-kept with a note — never a crash.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel, Field

from ... import config
from .models import Unit

OPENER_ROLES = frozenset({"example_setup", "problem_givens", "practice_prompt"})
STEP_ROLES = frozenset({"worked_step", "calculation", "derivation"})
TERMINAL_ROLES = frozenset({"result", "solution"})
# W25-D(b) closers: payoff-adjacent roles that legitimately END a worked example when the
# labeler never emitted result/solution (the qP 678-883s graph-build example: 16
# worked_steps, terminals labeled claim/physical_interpretation — invisible pre-W25-D).
# Accepted ONLY once steps exist (an opener straight into a claim is not an extraction
# event); the synthetic role stays 'result' (worked-example contract). Review-fixed
# semantics: a closer is PROVISIONAL — a later step demotes it to a mid-example
# interpretation and a later true result/solution outranks it (breaking hard at the first
# closer truncated qP's 700.9s arc to 1 step at u0115, orphaning 3 later worked_steps) —
# and it must sit within MAX_ARC_MEMBER_GAP_S of the last member (a 69s step→closer hop
# manufactured a phantom 302s kinematics arc).
CLOSER_ROLES = frozenset({"claim", "physical_interpretation", "graph_interpretation",
                          "unit_check"})
# W25-D(c) neutral members: interpretation/explanation-family asides that unpack a step
# WITHOUT leaving the example. Transparent to the interleave budget — they neither count
# toward nor reset MAX_ARC_INTERLEAVE — and NOT arc members (the hull spans them anyway).
# Deliberately excluded: transition/administrative/irrelevant (real topic exits — exactly
# what the interleave budget exists to catch) and structure-OPENING roles (setup,
# definition, variable_definition, claim, evidence, procedure, demonstration, summary,
# misconception/correction/exception): treating those as neutral would let one example's
# scan crawl across unrelated extractable content. Closer∩neutral roles
# (physical_/graph_interpretation, unit_check) become the PROVISIONAL closer once steps
# exist; when they don't qualify (steps empty / a closer already pends / farther than
# MAX_ARC_MEMBER_GAP_S) they stay neutral-transparent. Opener accumulation (pre-step
# multi-unit setup) is bounded by the same gap — real setups sit <5s apart, and unbounded
# accumulation is how the kinematics phantom crawled 73.5s across two claims (W25-D review).
NEUTRAL_ROLES = frozenset({"explanation", "intuition", "physical_interpretation",
                           "graph_interpretation", "diagram_interpretation", "unit_check"})
MAX_ARC_INTERLEAVE = 2       # non-arc, non-neutral units tolerated between consecutive members


@dataclass
class ArcCandidate:
    """One detected instructional arc — a complete extraction event, not a single unit."""
    arc_id: str
    arc_role: str                              # "worked_example" | "practice_pair"
    unit_ids: list[str]                        # member units, time-ordered
    opener_ids: list[str] = field(default_factory=list)
    step_ids: list[str] = field(default_factory=list)
    terminal_id: str = ""
    terminal_role: str = "result"              # synthetic anchor role: "result" | "solution"
    calculation_as_final: bool = False         # payoff lives in a 'calculation'-role unit
    verified: bool = False                     # confirmed by the LLM verification pass


def _concepts(u: Unit) -> set[str]:
    return {c.strip().lower() for c in (u.concepts_introduced + u.concepts_required)
            if c and c.strip()}


def _scan_worked_arcs(units: list[Unit]) -> list[ArcCandidate]:
    by_id = {u.unit_id: u for u in units}
    arcs: list[ArcCandidate] = []
    n = len(units)
    i = 0
    while i < n:
        if units[i].role not in OPENER_ROLES:
            i += 1
            continue
        openers = [units[i].unit_id]
        steps: list[str] = []
        terminal_id = ""
        terminal_role = ""
        last_member = i
        interleave = 0
        restart_at: Optional[int] = None       # a NEW opener after steps → next example
        pending_closer: Optional[int] = None   # first closer AFTER the last step (W25-D(b))
        j = i + 1
        while j < n:
            r = units[j].role
            gap = units[j].start - units[last_member].end   # locality vs last real member
            if r in TERMINAL_ROLES:
                if steps:                      # grammar needs 1+ steps before the terminal
                    terminal_id = units[j].unit_id
                    terminal_role = "solution" if r == "solution" else "result"
                    last_member = j
                break
            if (r in CLOSER_ROLES and steps and pending_closer is None
                    and gap <= config.MAX_ARC_MEMBER_GAP_S):
                # W25-D(b), review-fixed: a closer is a PROVISIONAL terminal, not a hard
                # break — a later step re-opens the example (the closer was a mid-example
                # interpretation, cf. (c)) and a later TRUE result/solution outranks it;
                # only a scan exit (abort / new opener / end) with no later step finalizes
                # it. First-closer-after-the-last-step wins (later closers are drift) and
                # it must be LOCAL — the kinematics phantom rode a 69s step→closer hop.
                pending_closer = j
                if r not in NEUTRAL_ROLES:     # claim keeps costing interleave budget
                    interleave += 1            # (pre-W25-D semantics); closer∩neutral
                    if interleave > MAX_ARC_INTERLEAVE:   # stays transparent per (c).
                        break
            elif r in STEP_ROLES:
                steps.append(units[j].unit_id)
                last_member = j
                interleave = 0
                pending_closer = None          # mid-example closer ≠ the payoff
            elif r in OPENER_ROLES:
                if steps:                      # a new problem starts — close the current arc
                    restart_at = j
                    break
                if gap > config.MAX_ARC_MEMBER_GAP_S:
                    break                      # distant opener = a DIFFERENT event (the
                                               # kinematics phantom rode a 73.5s prompt→
                                               # setup hop); the outer scan re-finds it.
                openers.append(units[j].unit_id)   # multi-unit setup (setup + givens …)
                last_member = j
                interleave = 0
            elif r in NEUTRAL_ROLES:
                pass                           # W25-D(c): transparent — no count, no reset
            else:
                interleave += 1
                if interleave > MAX_ARC_INTERLEAVE:
                    break
            j += 1
        if not terminal_id and pending_closer is not None:
            # W25-D(b): the scan exited with a closer standing after the last step — that
            # closer is the example's payoff (synthetic role 'result', same as below).
            terminal_id = units[pending_closer].unit_id
            terminal_role = "result"
            last_member = pending_closer
        if not terminal_id and steps:
            # last-step-as-final: no terminal/closer unit exists — the example's payoff
            # lives in its LAST step regardless of step role (W25-D(a) generalizes the
            # audited kinematics calculation-as-final shape; synthetic role 'result').
            terminal_id = steps[-1]
            terminal_role = "result"
        if terminal_id and steps:
            member_ids = openers + steps + ([terminal_id] if terminal_id not in steps else [])
            order = {u.unit_id: k for k, u in enumerate(units)}
            member_ids = sorted(set(member_ids), key=lambda uid: order[uid])
            arc_role = ("practice_pair"
                        if terminal_role == "solution"
                        and any(by_id[o].role == "practice_prompt" for o in openers)
                        else "worked_example")
            arcs.append(ArcCandidate(
                arc_id="", arc_role=arc_role, unit_ids=member_ids,
                opener_ids=list(openers), step_ids=list(steps),
                terminal_id=terminal_id, terminal_role=terminal_role,
                calculation_as_final=(by_id[terminal_id].role == "calculation")))
            i = restart_at if restart_at is not None else last_member + 1
        else:
            i += 1
    return arcs


def _pair_practice_prompts(units: list[Unit], worked: list[ArcCandidate]) -> list[ArcCandidate]:
    """Cross-distance practice_prompt -> solution pairing: the candidate solution is the
    FIRST later solution/result unit sharing >=1 concept with the prompt. A solution unit
    is CONSUMED by its first pairing (I1 reconcile of a P3 review finding): two prompts can
    no longer both pair to the same solution and flood assembly with nested arcs — a later
    prompt scans on to the next unused qualifying unit instead."""
    consumed = {uid for a in worked for uid in a.opener_ids}
    covered = [set(a.unit_ids) for a in worked]
    used_solutions: set[str] = set()
    pairs: list[ArcCandidate] = []
    for idx, u in enumerate(units):
        if u.role != "practice_prompt" or u.unit_id in consumed:
            continue
        pc = _concepts(u)
        if not pc:
            continue
        for v in units[idx + 1:]:
            if v.unit_id in used_solutions:
                continue                         # already answers an earlier prompt
            if v.role in ("solution", "result") and (_concepts(v) & pc):
                members = {u.unit_id, v.unit_id}
                if not any(members <= cov for cov in covered):   # already inside a worked arc
                    used_solutions.add(v.unit_id)
                    pairs.append(ArcCandidate(
                        arc_id="", arc_role="practice_pair",
                        unit_ids=[u.unit_id, v.unit_id],
                        opener_ids=[u.unit_id], step_ids=[],
                        terminal_id=v.unit_id, terminal_role="solution"))
                break                            # first qualifying later unit only
    return pairs


def _substance_s(arc: ArcCandidate, by_id: dict[str, Unit]) -> float:
    """Member-duration sum (s) — the substance the arc actually puts on screen."""
    return sum(max(0.0, by_id[uid].end - by_id[uid].start)
               for uid in arc.unit_ids if uid in by_id)


def detect_arcs(units: list[Unit]) -> list[ArcCandidate]:
    """Deterministic arc detection — zero LLM calls. Returns worked-example arcs (grammar
    scan) followed by cross-distance practice pairs, with sequential arc ids. W25-D(d):
    ZERO-step arcs (concept-paired prompt→solution) summing below MIN_ARC_SUBSTANCE_S of
    member duration are dropped HERE — on qP 8 of 11 arcs were 2-15s Socratic micro-pairs
    (two inside the sponsor promo), and downstream they are saturation-exempt
    (candidates), dropped-last (dedupe) and snap-padded to 20s (refine), so detection is
    the only stage a floor can work at. Arcs with real steps are never substance-gated
    (a tight complete example beats no example)."""
    worked = _scan_worked_arcs(units)
    arcs = worked + _pair_practice_prompts(units, worked)
    by_id = {u.unit_id: u for u in units}
    arcs = [a for a in arcs
            if a.step_ids or _substance_s(a, by_id) >= config.MIN_ARC_SUBSTANCE_S]
    for k, arc in enumerate(arcs):
        arc.arc_id = f"arc_{k}"
    return arcs


# ── optional single-call verification (MathNet pattern) ─────────────────────────────────
class ArcCheckLLM(BaseModel):
    arc_id: str = ""
    problem_ids: list[str] = Field(default_factory=list)
    step_ids: list[str] = Field(default_factory=list)
    answer_ids: list[str] = Field(default_factory=list)


class ArcVerifyLLM(BaseModel):
    arcs: list[ArcCheckLLM] = Field(default_factory=list)


ARC_VERIFY_SYSTEM = (
    "You verify DETECTED instructional arcs in a video transcript. Each arc claims to be a "
    "complete worked example or practice question+answer: a problem statement, the working, "
    "and the answer. For each arc you are shown its member units (id, role, text). Return, per "
    "arc, ONLY the unit ids you can actually confirm from the text: problem_ids (units stating "
    "the problem/question/givens), step_ids (units doing the working), answer_ids (units "
    "containing the final answer/result). Use ONLY unit ids listed for that arc. If an arc is "
    "not a real problem->working->answer event, omit it entirely (or return empty id lists). "
    "For a practice question+answer pair, step_ids may be empty. Output only the structured result."
)


def _arc_verify_enabled(settings: dict) -> bool:
    flag = (settings or {}).get("arc_verify")
    return config.ARC_VERIFY if flag is None else bool(flag)


def verify_arcs(arcs: list[ArcCandidate], units_by_id: dict[str, Unit],
                settings: dict) -> tuple[list[ArcCandidate], Optional[str]]:
    """ONE batched llm_json call verifying every detected arc (config flag ARC_VERIFY,
    default on). Rule-check: returned ids must fall inside their arc (others discarded);
    arcs the model rejects (omitted / empty confirmation) are dropped. On LLM failure the
    arcs are kept unverified and a degraded note is returned — never a hard failure."""
    if not arcs or not _arc_verify_enabled(settings):
        return arcs, None
    from ...llm import llm_json
    blocks = []
    for arc in arcs:
        rows = "\n".join(
            f"  {uid} [{units_by_id[uid].role}] "
            f"{(units_by_id[uid].transcript or units_by_id[uid].summary)[:160]}"
            for uid in arc.unit_ids if uid in units_by_id)
        blocks.append(f"ARC {arc.arc_id} ({arc.arc_role}):\n{rows}")
    user = ("DETECTED ARCS:\n\n" + "\n\n".join(blocks)
            + "\n\nConfirm each arc's problem_ids / step_ids / answer_ids "
              "(only ids listed under that arc).")
    try:
        res = llm_json(ARC_VERIFY_SYSTEM, user, ArcVerifyLLM, temperature=0.0)
    except Exception as e:
        print(f"[arcs] verification failed ({e!r}); keeping {len(arcs)} arc(s) unverified",
              file=sys.stderr)
        return arcs, "arc verification degraded — arcs kept unverified"
    checks = {c.arc_id: c for c in res.arcs}
    kept: list[ArcCandidate] = []
    for arc in arcs:
        chk = checks.get(arc.arc_id)
        if chk is None:
            continue                            # rejected by omission
        valid = set(arc.unit_ids)               # rule-check: ids must fall inside the arc
        problem = [i for i in chk.problem_ids if i in valid]
        steps = [i for i in chk.step_ids if i in valid]
        answer = [i for i in chk.answer_ids if i in valid]
        needs_steps = arc.arc_role == "worked_example"
        if problem and answer and (steps or not needs_steps):
            arc.verified = True
            kept.append(arc)
    return kept, None
